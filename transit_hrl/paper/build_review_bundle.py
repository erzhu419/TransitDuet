"""Package offline analysis without operational files or local archive metadata."""

from io import BytesIO
from pathlib import Path, PurePosixPath
import sys
import tarfile
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "paper/artifact_files.txt"


def include(name: str) -> bool:
    path = PurePosixPath(name)
    if path.parts[0] in {"paper", "freq_hrl"}:
        return True
    if path.parts[0] == "results":
        return path.name == "summary.json" or path.parent.name == "report"
    return name in {
        "tests/test_paper_plan_validity_evidence.py",
        "tests/test_plan_validity_figures.py",
    }


def anonymous(member: tarfile.TarInfo) -> tarfile.TarInfo:
    member.uid = member.gid = 0
    member.uname = member.gname = ""
    return member


def build(output: Path) -> None:
    names = [name for name in (ROOT / MANIFEST).read_text().splitlines() if name and include(name)]
    manifest = ("\n".join(names) + "\n").encode()
    if output.suffix == ".zip":
        with ZipFile(output, "w", compression=ZIP_DEFLATED) as bundle:
            for name in names:
                if name == MANIFEST:
                    continue
                source = ROOT / name
                files = sorted(source.rglob("*")) if source.is_dir() else [source]
                for path in files:
                    if path.is_file():
                        bundle.writestr(path.relative_to(ROOT).as_posix(), path.read_bytes())
            bundle.writestr(MANIFEST, manifest)
        return
    with tarfile.open(output, "w:gz") as bundle:
        for name in names:
            if name != MANIFEST:
                bundle.add(ROOT / name, arcname=name, filter=anonymous)
        member = tarfile.TarInfo(MANIFEST)
        member.size = len(manifest)
        bundle.addfile(member, BytesIO(manifest))


if __name__ == "__main__":
    build(Path(sys.argv[1]))
