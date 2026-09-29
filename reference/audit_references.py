#!/usr/bin/env python3
"""Verify BibTeX metadata and download open-access reference PDFs.

The script is intentionally conservative: it downloads only open-access PDFs
found through arXiv, OpenAlex, Crossref links, or explicit BibTeX URLs. It does
not attempt to bypass publisher access controls.
"""

from __future__ import annotations

import csv
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


ROOT = Path(__file__).resolve().parents[1]
BIB_PATH = ROOT / "paper" / "references.bib"
REF_DIR = ROOT / "reference"
PDF_DIR = REF_DIR / "pdfs"
META_DIR = REF_DIR / "metadata"
CSV_PATH = REF_DIR / "reference_audit.csv"
MD_PATH = REF_DIR / "reference_audit.md"
JSON_PATH = REF_DIR / "reference_audit.json"

HEADERS = {
    "User-Agent": "TransitDuet-reference-audit/1.0 (mailto:example@example.com)",
}

MANUAL_METADATA: dict[str, dict[str, Any]] = {
    # PMLR/ICLR/NIPS proceedings are better represented by their official
    # conference records than by arXiv/OpenAlex preprint records.
    "haarnoja2018soft": {
        "source": "manual-pmlr",
        "title": "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor",
        "author": [
            "Haarnoja, Tuomas",
            "Zhou, Aurick",
            "Abbeel, Pieter",
            "Levine, Sergey",
        ],
        "year": "2018",
        "container": "Proceedings of the 35th International Conference on Machine Learning",
        "volume": "80",
        "pages": "1861--1870",
        "publisher": "PMLR",
        "url": "https://proceedings.mlr.press/v80/haarnoja18b.html",
        "pdf_url": "https://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf",
    },
    "gkiotsalitis2021bus": {
        "source": "manual-crossref",
        "title": "At-Stop Control Measures in Public Transport: Literature Review and Research Agenda",
        "author": ["Gkiotsalitis, Konstantinos", "Cats, Oded"],
        "year": "2021",
        "container": "Transportation Research Part E: Logistics and Transportation Review",
        "volume": "145",
        "pages": "102176",
        "doi": "10.1016/j.tre.2020.102176",
        "url": "https://doi.org/10.1016/j.tre.2020.102176",
        "pdf_url": "https://repository.tudelft.nl/file/File_d117b4dc-8cbc-4286-8f32-d9829ca3a273",
    },
    "vezhnevets2017feudal": {
        "source": "manual-pmlr",
        "title": "FeUdal Networks for Hierarchical Reinforcement Learning",
        "author": [
            "Vezhnevets, Alexander Sasha",
            "Osindero, Simon",
            "Schaul, Tom",
            "Heess, Nicolas",
            "Jaderberg, Max",
            "Silver, David",
            "Kavukcuoglu, Koray",
        ],
        "year": "2017",
        "container": "Proceedings of the 34th International Conference on Machine Learning",
        "volume": "70",
        "pages": "3540--3549",
        "publisher": "PMLR",
        "url": "https://proceedings.mlr.press/v70/vezhnevets17a.html",
        "pdf_url": "https://proceedings.mlr.press/v70/vezhnevets17a/vezhnevets17a.pdf",
    },
    "duan2021distributional": {
        "source": "manual-ieee",
        "title": "Distributional Soft Actor-Critic: Off-Policy Reinforcement Learning for Addressing Value Estimation Errors",
        "author": [
            "Duan, Jingliang",
            "Guan, Yang",
            "Li, Shengbo Eben",
            "Ren, Yangang",
            "Sun, Qi",
            "Cheng, Bo",
        ],
        "year": "2022",
        "container": "IEEE Transactions on Neural Networks and Learning Systems",
        "volume": "33",
        "number": "11",
        "pages": "6584--6598",
        "doi": "10.1109/TNNLS.2021.3082568",
        "url": "https://doi.org/10.1109/TNNLS.2021.3082568",
        "pdf_url": "https://arxiv.org/pdf/2001.02811.pdf",
    },
    "fujimoto2018addressing": {
        "source": "manual-pmlr",
        "title": "Addressing Function Approximation Error in Actor-Critic Methods",
        "author": ["Fujimoto, Scott", "van Hoof, Herke", "Meger, David"],
        "year": "2018",
        "container": "Proceedings of the 35th International Conference on Machine Learning",
        "volume": "80",
        "pages": "1587--1596",
        "publisher": "PMLR",
        "url": "https://proceedings.mlr.press/v80/fujimoto18a.html",
        "pdf_url": "https://proceedings.mlr.press/v80/fujimoto18a/fujimoto18a.pdf",
    },
    "stooke2020responsive": {
        "source": "manual-pmlr",
        "title": "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods",
        "author": ["Stooke, Adam", "Achiam, Joshua", "Abbeel, Pieter"],
        "year": "2020",
        "container": "Proceedings of the 37th International Conference on Machine Learning",
        "volume": "119",
        "pages": "9133--9143",
        "publisher": "PMLR",
        "url": "https://proceedings.mlr.press/v119/stooke20a.html",
        "pdf_url": "https://proceedings.mlr.press/v119/stooke20a/stooke20a.pdf",
    },
    "metelli2020control": {
        "source": "manual-pmlr",
        "title": "Control Frequency Adaptation via Action Persistence in Batch Reinforcement Learning",
        "author": [
            "Metelli, Alberto Maria",
            "Mazzolini, Flavio",
            "Bisi, Lorenzo",
            "Sabbioni, Luca",
            "Restelli, Marcello",
        ],
        "year": "2020",
        "container": "Proceedings of the 37th International Conference on Machine Learning",
        "volume": "119",
        "pages": "6862--6873",
        "publisher": "PMLR",
        "url": "https://proceedings.mlr.press/v119/metelli20a.html",
        "pdf_url": "https://proceedings.mlr.press/v119/metelli20a/metelli20a.pdf",
    },
    "tessler2019reward": {
        "source": "manual-openreview",
        "title": "Reward Constrained Policy Optimization",
        "author": [
            "Tessler, Chen",
            "Mankowitz, Daniel J.",
            "Mannor, Shie",
        ],
        "year": "2019",
        "container": "Proceedings of the 7th International Conference on Learning Representations",
        "url": "https://openreview.net/forum?id=SkfrvsA9FX",
        "pdf_url": "https://openreview.net/pdf?id=SkfrvsA9FX",
    },
    "dayan1992feudal": {
        "source": "manual-neurips",
        "title": "Feudal Reinforcement Learning",
        "author": ["Dayan, Peter", "Hinton, Geoffrey E."],
        "year": "1992",
        "container": "Advances in Neural Information Processing Systems",
        "volume": "5",
        "pages": "271--278",
        "url": "https://papers.nips.cc/paper/714-feudal-reinforcement-learning",
        "pdf_url": "https://proceedings.neurips.cc/paper/1992/file/d14220ee66aeec73c49038385428ec4c-Paper.pdf",
    },
    "li2019hierarchical": {
        "source": "manual-neurips",
        "title": "Hierarchical Reinforcement Learning with Advantage-Based Auxiliary Rewards",
        "author": ["Li, Siyuan", "Wang, Rui", "Tang, Minxue", "Zhang, Chongjie"],
        "year": "2019",
        "container": "Advances in Neural Information Processing Systems",
        "volume": "32",
        "pages": "1409--1419",
        "url": "https://proceedings.neurips.cc/paper/2019/hash/81e74d678581a3bb7a720b019f4f1a93-Abstract.html",
        "pdf_url": "https://proceedings.neurips.cc/paper/2019/file/81e74d678581a3bb7a720b019f4f1a93-Paper.pdf",
    },
    "lee2020reinforcement": {
        "source": "manual-neurips",
        "title": "Reinforcement Learning for Control with Multiple Frequencies",
        "author": ["Lee, Jongmin", "Lee, Byung-Jun", "Kim, Kee-Eung"],
        "year": "2020",
        "container": "Advances in Neural Information Processing Systems",
        "volume": "33",
        "pages": "3254--3264",
        "url": "https://proceedings.neurips.cc/paper/2020/hash/216f44e2d28d4e175a194492bde9148f-Abstract.html",
        "pdf_url": "https://proceedings.neurips.cc/paper/2020/file/216f44e2d28d4e175a194492bde9148f-Paper.pdf",
    },
    "lillicrap2016continuous": {
        "source": "manual-iclr",
        "title": "Continuous Control with Deep Reinforcement Learning",
        "author": [
            "Lillicrap, Timothy P.",
            "Hunt, Jonathan J.",
            "Pritzel, Alexander",
            "Heess, Nicolas",
            "Erez, Tom",
            "Tassa, Yuval",
            "Silver, David",
            "Wierstra, Daan",
        ],
        "year": "2016",
        "container": "Proceedings of the 4th International Conference on Learning Representations",
        "url": "https://arxiv.org/abs/1509.02971",
        "pdf_url": "https://arxiv.org/pdf/1509.02971.pdf",
    },
    "zhang2026single": {
        "source": "manual-tse",
        "title": "Single agent robust deep reinforcement learning for bus fleet control",
        "author": [
            "Zhang, Yifan",
            "Zheng, Liang",
            "Zhang, Qifan",
            "Tang, Hewei",
        ],
        "year": "2026",
        "container": "Transportation Safety and Environment",
        "volume": "00",
        "number": "0",
        "pages": "tdag005",
        "doi": "10.1093/tse/tdag005",
        "url": "https://doi.org/10.1093/tse/tdag005",
    },
}

MANUAL_PDF_URLS: dict[str, list[str]] = {
    key: [meta["pdf_url"]]
    for key, meta in MANUAL_METADATA.items()
    if meta.get("pdf_url")
}

MANUAL_PDF_URLS.update({
    "pan2024roboduet": ["https://arxiv.org/pdf/2403.17367.pdf"],
    "wang2020real": ["https://lijunsun.github.io/files/papers/2020-TRC-Bus.pdf"],
    "wang2023robust": ["https://arxiv.org/pdf/2111.01946.pdf"],
    "wang2023multiobjective": [
        "https://papers.ssrn.com/sol3/Delivery.cfm/4a2e6716-b6fc-4621-92a4-23c416f52f55-MECA.pdf?abstractid=4305641&mirid=1",
    ],
    "he2022dynamic": ["https://arxiv.org/pdf/2006.08706.pdf"],
    "rodriguez2023cooperative": ["https://www.osti.gov/servlets/purl/2577182"],
    "yu2024hierarchical": ["https://ieeexplore.ieee.org/iel7/6979/10621861/10440179.pdf"],
    "xu2025systematic": ["https://ieeexplore.ieee.org/iel8/6979/4358928/10909364.pdf"],
    "yu2025llm": ["https://arxiv.org/pdf/2410.10212.pdf"],
    "alesiani2018reinforcement": ["https://ieeexplore.ieee.org/iel7/8543039/8569013/08569473.pdf"],
    "gkiotsalitis2021bus": [
        "https://repository.tudelft.nl/file/File_d117b4dc-8cbc-4286-8f32-d9829ca3a273",
    ],
    "cats2019frequency": ["https://journals.sagepub.com/doi/pdf/10.1177/0361198118822292"],
    "sutton1999between": [
        "https://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf",
    ],
})


@dataclass
class BibEntry:
    entry_type: str
    key: str
    fields: dict[str, str]
    raw: str


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def strip_outer(value: str) -> str:
    value = value.strip().rstrip(",").strip()
    if (value.startswith("{") and value.endswith("}")) or (
        value.startswith('"') and value.endswith('"')
    ):
        value = value[1:-1]
    return re.sub(r"\s+", " ", value).strip()


def parse_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    i = 0
    n = len(body)
    while i < n:
        while i < n and (body[i].isspace() or body[i] == ","):
            i += 1
        m = re.match(r"([A-Za-z][A-Za-z0-9_-]*)\s*=", body[i:])
        if not m:
            i += 1
            continue
        name = m.group(1).lower()
        i += m.end()
        while i < n and body[i].isspace():
            i += 1
        if i >= n:
            break
        if body[i] == "{":
            start = i
            depth = 0
            while i < n:
                if body[i] == "{" and (i == 0 or body[i - 1] != "\\"):
                    depth += 1
                elif body[i] == "}" and (i == 0 or body[i - 1] != "\\"):
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            value = body[start:i]
        elif body[i] == '"':
            start = i
            i += 1
            while i < n:
                if body[i] == '"' and body[i - 1] != "\\":
                    i += 1
                    break
                i += 1
            value = body[start:i]
        else:
            start = i
            while i < n and body[i] != ",":
                i += 1
            value = body[start:i]
        fields[name] = strip_outer(value)
    return fields


def parse_bib(text: str) -> list[BibEntry]:
    entries: list[BibEntry] = []
    pos = 0
    while True:
        at = text.find("@", pos)
        if at < 0:
            break
        m = re.match(r"@(\w+)\s*\{\s*([^,]+),", text[at:])
        if not m:
            pos = at + 1
            continue
        entry_type, key = m.group(1), m.group(2).strip()
        body_start = at + m.end()
        depth = 1
        i = body_start
        while i < len(text) and depth:
            if text[i] == "{" and text[i - 1] != "\\":
                depth += 1
            elif text[i] == "}" and text[i - 1] != "\\":
                depth -= 1
            i += 1
        raw = text[at:i]
        fields = parse_fields(text[body_start : i - 1])
        entries.append(BibEntry(entry_type=entry_type, key=key, fields=fields, raw=raw))
        pos = i
    return entries


def normalize_text(value: str) -> str:
    value = value.replace("{", "").replace("}", "")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = value.replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\b(a|an|the|of|for|in|on|to|and|with|by|via|using)\b", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def title_similarity(a: str, b: str) -> float:
    ta = set(normalize_text(a).split())
    tb = set(normalize_text(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def parse_author_surnames(author_field: str) -> list[str]:
    authors = [a.strip() for a in re.split(r"\s+and\s+", author_field) if a.strip()]
    surnames = []
    for author in authors:
        if "," in author:
            surname = author.split(",", 1)[0]
        else:
            parts = author.split()
            surname = parts[-1] if parts else ""
        surnames.append(normalize_text(surname))
    return surnames


def request_json(url: str, *, params: dict[str, Any] | None = None, sleep_s: float = 0.0) -> dict[str, Any] | None:
    if sleep_s:
        time.sleep(sleep_s)
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=25)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def crossref_by_doi(doi: str) -> dict[str, Any] | None:
    data = request_json(f"https://api.crossref.org/works/{quote(doi)}")
    if not data:
        return None
    return data.get("message")


def crossref_by_title(title: str, first_author: str | None = None) -> dict[str, Any] | None:
    params = {"query.title": title, "rows": 5}
    if first_author:
        params["query.author"] = first_author
    data = request_json("https://api.crossref.org/works", params=params)
    items = (data or {}).get("message", {}).get("items", [])
    if not items:
        return None
    best = max(items, key=lambda item: title_similarity(title, " ".join(item.get("title") or [])))
    if title_similarity(title, " ".join(best.get("title") or [])) < 0.55:
        return None
    return best


def arxiv_by_id(arxiv_id: str) -> dict[str, Any] | None:
    url = "https://export.arxiv.org/api/query"
    try:
        time.sleep(3.1)
        r = requests.get(url, params={"id_list": arxiv_id}, headers=HEADERS, timeout=25)
        if r.status_code != 200 or "<entry>" not in r.text:
            return None
        return parse_arxiv_atom(r.text)
    except Exception:
        return None


def arxiv_by_title(title: str) -> dict[str, Any] | None:
    try:
        time.sleep(3.1)
        r = requests.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": f'ti:"{title}"', "max_results": 3},
            headers=HEADERS,
            timeout=8,
        )
        if r.status_code != 200 or "<entry>" not in r.text:
            return None
        entries = re.findall(r"<entry>(.*?)</entry>", r.text, re.S)
        parsed = [parse_arxiv_atom(f"<entry>{e}</entry>") for e in entries]
        parsed = [p for p in parsed if p]
        if not parsed:
            return None
        best = max(parsed, key=lambda item: title_similarity(title, item.get("title", "")))
        if title_similarity(title, best.get("title", "")) < 0.55:
            return None
        return best
    except Exception:
        return None


def xml_text(block: str, tag: str) -> str:
    m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", block, re.S)
    if not m:
        return ""
    value = re.sub(r"<[^>]+>", "", m.group(1))
    return re.sub(r"\s+", " ", value).strip()


def parse_arxiv_atom(text: str) -> dict[str, Any] | None:
    entry_m = re.search(r"<entry>(.*?)</entry>", text, re.S)
    block = entry_m.group(1) if entry_m else text
    title = xml_text(block, "title")
    if not title:
        return None
    authors = re.findall(r"<author>\s*<name>(.*?)</name>\s*</author>", block, re.S)
    authors = [re.sub(r"\s+", " ", a).strip() for a in authors]
    arxiv_url = xml_text(block, "id")
    arxiv_id = arxiv_url.rstrip("/").split("/")[-1] if arxiv_url else ""
    year = ""
    published = xml_text(block, "published")
    if published:
        year = published[:4]
    return {
        "source": "arxiv",
        "title": title,
        "author": authors,
        "year": year,
        "eprint": arxiv_id,
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else "",
        "url": arxiv_url,
    }


def openalex_lookup(entry: BibEntry, doi: str | None) -> dict[str, Any] | None:
    if doi:
        data = request_json(f"https://api.openalex.org/works/doi:{quote(doi)}")
        if data and data.get("id"):
            return data
    title = entry.fields.get("title", "")
    if not title:
        return None
    data = request_json("https://api.openalex.org/works", params={"search": title, "per-page": 5})
    results = (data or {}).get("results", [])
    if not results:
        return None
    best = max(results, key=lambda item: title_similarity(title, item.get("title") or ""))
    if title_similarity(title, best.get("title") or "") < 0.55:
        return None
    return best


def crossref_to_meta(item: dict[str, Any]) -> dict[str, Any]:
    authors = []
    for a in item.get("author") or []:
        given = a.get("given") or ""
        family = a.get("family") or ""
        if family and given:
            authors.append(f"{family}, {given}")
        elif family:
            authors.append(family)
    issued = item.get("published-print") or item.get("published-online") or item.get("issued") or {}
    date_parts = issued.get("date-parts") or []
    year = str(date_parts[0][0]) if date_parts and date_parts[0] else ""
    return {
        "source": "crossref",
        "title": " ".join(item.get("title") or []),
        "author": authors,
        "year": year,
        "container": " ".join(item.get("container-title") or []),
        "volume": item.get("volume") or "",
        "number": item.get("issue") or "",
        "pages": (item.get("page") or "").replace("-", "--"),
        "doi": item.get("DOI") or "",
        "type": item.get("type") or "",
        "url": item.get("URL") or "",
        "links": item.get("link") or [],
        "publisher": item.get("publisher") or "",
    }


def openalex_to_meta(item: dict[str, Any]) -> dict[str, Any]:
    authors = []
    for au in item.get("authorships") or []:
        name = ((au.get("author") or {}).get("display_name") or "").strip()
        if name:
            authors.append(name)
    loc = item.get("primary_location") or {}
    source = loc.get("source") or {}
    biblio = item.get("biblio") or {}
    best = item.get("best_oa_location") or {}
    return {
        "source": "openalex",
        "title": item.get("title") or "",
        "author": authors,
        "year": str(item.get("publication_year") or ""),
        "container": source.get("display_name") or "",
        "volume": biblio.get("volume") or "",
        "number": biblio.get("issue") or "",
        "pages": biblio.get("first_page") + "--" + biblio.get("last_page")
        if biblio.get("first_page") and biblio.get("last_page")
        else (biblio.get("first_page") or ""),
        "doi": (item.get("doi") or "").replace("https://doi.org/", ""),
        "url": item.get("id") or "",
        "pdf_url": best.get("pdf_url") or loc.get("pdf_url") or "",
        "oa_url": best.get("landing_page_url") or "",
        "is_oa": bool((item.get("open_access") or {}).get("is_oa")),
    }


def pdf_candidates(
    entry: BibEntry,
    verified: dict[str, Any],
    openalex: dict[str, Any] | None,
    arxiv: dict[str, Any] | None,
) -> list[str]:
    urls: list[str] = []
    urls.extend(MANUAL_PDF_URLS.get(entry.key, []))
    eprint = entry.fields.get("eprint", "").strip()
    if eprint:
        urls.append(f"https://arxiv.org/pdf/{eprint}.pdf")
    if arxiv and arxiv.get("pdf_url"):
        urls.append(arxiv["pdf_url"])
    if verified.get("source") == "arxiv" and verified.get("pdf_url"):
        urls.append(verified["pdf_url"])
    if openalex:
        oa = openalex_to_meta(openalex)
        for key in ["pdf_url", "oa_url"]:
            if oa.get(key):
                urls.append(oa[key])
    for link in verified.get("links") or []:
        if "pdf" in (link.get("content-type") or "").lower() and link.get("URL"):
            urls.append(link["URL"])
    if entry.fields.get("url"):
        urls.append(entry.fields["url"])
    seen = set()
    out = []
    for url in urls:
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


def download_pdf(urls: list[str], out_path: Path) -> tuple[str, str]:
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=45, allow_redirects=True)
            ctype = r.headers.get("content-type", "").lower()
            content = r.content
            looks_pdf = content[:4] == b"%PDF" or "pdf" in ctype
            if r.status_code == 200 and looks_pdf and len(content) > 10_000:
                out_path.write_bytes(content)
                return "downloaded", url
        except Exception:
            continue
    if out_path.exists() and out_path.stat().st_size > 10_000:
        return "existing", str(out_path)
    return "not_downloaded", ""


def compare(entry: BibEntry, verified: dict[str, Any]) -> tuple[str, list[str], float]:
    notes = []
    title_sim = title_similarity(entry.fields.get("title", ""), verified.get("title", ""))
    if title_sim < 0.82:
        notes.append(f"title similarity {title_sim:.2f}")
    year_current = entry.fields.get("year", "")
    year_verified = verified.get("year", "")
    if year_current and year_verified and year_current != year_verified:
        notes.append(f"year {year_current} != {year_verified}")
    for field, verified_key in [
        ("volume", "volume"),
        ("number", "number"),
        ("pages", "pages"),
        ("doi", "doi"),
    ]:
        cur = normalize_pages(entry.fields.get(field, ""))
        ver = normalize_pages(verified.get(verified_key, ""))
        if cur and ver and cur.lower() != ver.lower():
            notes.append(f"{field} {cur} != {ver}")
    bib_surnames = parse_author_surnames(entry.fields.get("author", ""))
    ver_authors = verified.get("author") or []
    ver_surnames = parse_author_surnames(" and ".join(ver_authors))
    if bib_surnames and ver_surnames:
        if bib_surnames[0] != ver_surnames[0]:
            notes.append(f"first author {bib_surnames[0]} != {ver_surnames[0]}")
        if abs(len(bib_surnames) - len(ver_surnames)) >= 2:
            notes.append(f"author count {len(bib_surnames)} != {len(ver_surnames)}")
    if not verified:
        return "not_found", ["no metadata source matched"], title_sim
    if title_sim < 0.55:
        return "not_found", notes, title_sim
    if notes:
        return "mismatch", notes, title_sim
    return "verified", [], title_sim


def normalize_pages(value: str) -> str:
    value = value.replace("–", "-").replace("—", "-")
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"-+", "--", value).strip()
    m = re.fullmatch(r"(.+)--\1", value)
    if m:
        return m.group(1)
    return value


def audit_entry(entry: BibEntry) -> dict[str, Any]:
    doi = entry.fields.get("doi", "").strip()
    eprint = entry.fields.get("eprint", "").strip()
    crossref = crossref_by_doi(doi) if doi else None
    if not crossref:
        first = parse_author_surnames(entry.fields.get("author", ""))
        crossref = crossref_by_title(entry.fields.get("title", ""), first[0] if first else None)

    arxiv = None
    if eprint:
        arxiv = arxiv_by_id(eprint)
    elif "arxiv" in entry.fields.get("journal", "").lower():
        arxiv = arxiv_by_title(entry.fields.get("title", ""))

    openalex = openalex_lookup(entry, doi or (crossref or {}).get("DOI"))

    verified: dict[str, Any] = {}
    if entry.key in MANUAL_METADATA:
        verified = MANUAL_METADATA[entry.key]
    elif crossref:
        verified = crossref_to_meta(crossref)
    elif arxiv:
        verified = arxiv
    elif openalex:
        verified = openalex_to_meta(openalex)

    status, notes, title_sim = compare(entry, verified) if verified else (
        "not_found",
        ["no metadata source matched"],
        0.0,
    )

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = PDF_DIR / f"{entry.key}.pdf"
    pdf_status, pdf_url = download_pdf(pdf_candidates(entry, verified, openalex, arxiv), pdf_path)
    pdf_notes: list[str] = []
    if pdf_status == "not_downloaded":
        oa = (openalex or {}).get("open_access") or {}
        oa_status = oa.get("oa_status") or "unknown"
        is_oa = oa.get("is_oa")
        if is_oa is False:
            pdf_notes.append(f"OpenAlex OA status is {oa_status}; no open PDF URL found")
        elif is_oa is True:
            pdf_notes.append(
                f"OpenAlex OA status is {oa_status}, but the landing page did not provide a machine-downloadable PDF"
            )
        else:
            pdf_notes.append("No open PDF URL found in Crossref/OpenAlex/manual sources")

    metadata = {
        "key": entry.key,
        "entry_type": entry.entry_type,
        "current": entry.fields,
        "verified": verified,
        "crossref_raw": crossref,
        "arxiv_raw": arxiv,
        "openalex_raw": openalex,
        "status": status,
        "notes": notes,
        "title_similarity": title_sim,
        "pdf_status": pdf_status,
        "pdf_url": pdf_url,
        "pdf_path": str(pdf_path.relative_to(ROOT)) if pdf_path.exists() else "",
        "pdf_notes": pdf_notes,
    }
    META_DIR.mkdir(parents=True, exist_ok=True)
    (META_DIR / f"{entry.key}.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def write_reports(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "key",
        "entry_type",
        "status",
        "title_similarity",
        "current_title",
        "verified_title",
        "current_year",
        "verified_year",
        "current_venue",
        "verified_venue",
        "current_volume",
        "verified_volume",
        "current_number",
        "verified_number",
        "current_pages",
        "verified_pages",
        "current_doi",
        "verified_doi",
        "pdf_status",
        "pdf_path",
        "pdf_url",
        "pdf_notes",
        "notes",
        "metadata_source",
        "metadata_url",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            cur = row["current"]
            ver = row["verified"]
            writer.writerow({
                "key": row["key"],
                "entry_type": row["entry_type"],
                "status": row["status"],
                "title_similarity": f"{row['title_similarity']:.3f}",
                "current_title": cur.get("title", ""),
                "verified_title": ver.get("title", ""),
                "current_year": cur.get("year", ""),
                "verified_year": ver.get("year", ""),
                "current_venue": cur.get("journal") or cur.get("booktitle", ""),
                "verified_venue": ver.get("container", ""),
                "current_volume": cur.get("volume", ""),
                "verified_volume": ver.get("volume", ""),
                "current_number": cur.get("number", ""),
                "verified_number": ver.get("number", ""),
                "current_pages": cur.get("pages", ""),
                "verified_pages": ver.get("pages", ""),
                "current_doi": cur.get("doi", ""),
                "verified_doi": ver.get("doi", ""),
                "pdf_status": row["pdf_status"],
                "pdf_path": row["pdf_path"],
                "pdf_url": row["pdf_url"],
                "pdf_notes": "; ".join(row.get("pdf_notes") or []),
                "notes": "; ".join(row["notes"]),
                "metadata_source": ver.get("source", ""),
                "metadata_url": ver.get("url", ""),
            })

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    pdf_counts: dict[str, int] = {}
    for row in rows:
        pdf_counts[row["pdf_status"]] = pdf_counts.get(row["pdf_status"], 0) + 1

    lines = [
        "# Reference audit",
        "",
        f"- BibTeX file: `{BIB_PATH.relative_to(ROOT)}`",
        f"- Total entries: {len(rows)}",
        f"- Metadata status: {counts}",
        f"- PDF status: {pdf_counts}",
        "",
        "| Key | Status | PDF | Main notes | PDF notes |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        pdf = row["pdf_status"]
        if row["pdf_path"]:
            md_pdf_path = row["pdf_path"]
            if md_pdf_path.startswith("reference/"):
                md_pdf_path = md_pdf_path[len("reference/") :]
            pdf = f"[{pdf}]({md_pdf_path})"
        lines.append(
            f"| `{row['key']}` | {row['status']} | {pdf} | "
            f"{'; '.join(row['notes']) if row['notes'] else 'OK'} | "
            f"{'; '.join(row.get('pdf_notes') or []) if row.get('pdf_notes') else 'OK'} |"
        )
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    JSON_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    entries = parse_bib(read_text(BIB_PATH))
    rows = []
    for idx, entry in enumerate(entries, 1):
        print(f"[{idx:02d}/{len(entries)}] {entry.key}", flush=True)
        rows.append(audit_entry(entry))
    write_reports(rows)
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {MD_PATH}")
    print(f"Wrote {JSON_PATH}")


if __name__ == "__main__":
    main()
