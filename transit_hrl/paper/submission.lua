-- Keep Markdown as the only manuscript source; use native PDF captions.
local function caption(block, prefix)
  local inlines = block.content:clone()
  local head = inlines[1].content
  for _ = 1, (prefix == "Figure" and 6 or 4) do
    head:remove(1)
  end
  return pandoc.Caption({pandoc.Plain(inlines)})
end

local function is_wide_transit_table(tbl)
  if #tbl.colspecs ~= 4 or not tbl.head or not tbl.head.rows or #tbl.head.rows == 0 then
    return false
  end
  local first_cell = pandoc.utils.stringify(tbl.head.rows[1].cells[1])
  return first_cell:match("Study / contrast") ~= nil or first_cell:match("Contrast / endpoint") ~= nil
end

local function tmlr_heading(block, blocks)
  local heading = block.content
  if heading[1].text == "Appendix" then
    if heading[3].text == "A." then
      blocks:insert(pandoc.RawBlock("latex",
        "\\clearpage\n\\bibliographystyle{tmlr}\n\\bibliography{references}\n\\clearpage\n\\appendix"))
    end
    for _ = 1, 4 do
      heading:remove(1)
    end
  elseif heading[1].text and heading[1].text:match("^%d[%d.]*$") then
    heading:remove(1)
    heading:remove(1)
  end
  block.content = heading
end

function Pandoc(doc)
  local source = doc.blocks
  doc.meta.title = pandoc.MetaInlines(source[1].content)
  doc.meta.abstract = pandoc.MetaBlocks({source[4]})
  local blocks = pandoc.List()
  local i = 5
  while i <= #source do
    local block = source[i]
    local next_block = source[i + 1]
    local flush = false
    if block.t == "Header" then
      block.level = block.level - 1
      if doc.meta.tmlr then
        tmlr_heading(block, blocks)
      end
    elseif block.t == "Figure" then
      block.caption = caption(next_block, "Figure")
      block = block:walk({Image = function(image)
        image.src = image.src:gsub("%.png$", ".pdf")
        if image.src:match("transit_boundary%.pdf$") then
          image.attributes.width = "88%"
          flush = true
        end
        return image
      end})
      i = i + 1
    elseif next_block and next_block.t == "Table" then
      next_block.caption = caption(block, "Table")
      if is_wide_transit_table(next_block) then
        local colspecs = next_block.colspecs
        for column, width in ipairs({0.39, 0.13, 0.24, 0.24}) do
          colspecs[column][2] = width
        end
        next_block.colspecs = colspecs
      end
      block = next_block
      i = i + 1
    end
    blocks:insert(block)
    if flush then
      blocks:insert(pandoc.RawBlock("latex", "\\clearpage"))
    end
    i = i + 1
  end
  doc.blocks = blocks
  if doc.meta.tmlr then
    -- Local supplement filenames are not valid links in a standalone review PDF.
    return doc:walk({Link = function(link)
      if not link.target:match("^%a+:") and not link.target:match("^#") then
        return link.content
      end
      return link
    end})
  end
  return doc
end
