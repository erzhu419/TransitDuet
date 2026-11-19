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
  return doc
end
