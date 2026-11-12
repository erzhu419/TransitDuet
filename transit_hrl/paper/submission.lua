-- Keep Markdown as the only manuscript source; use native PDF captions.
local function caption(block, prefix)
  local inlines = block.content:clone()
  local head = inlines[1].content
  for _ = 1, (prefix == "Figure" and 6 or 4) do
    head:remove(1)
  end
  return pandoc.Caption({pandoc.Plain(inlines)})
end

function Pandoc(doc)
  local source = doc.blocks
  doc.meta.title = pandoc.MetaInlines(source[1].content)
  doc.meta.abstract = pandoc.MetaBlocks({source[4]})
  local blocks = pandoc.List()
  local i, table_number = 5, 0
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
        if image.src:match("fig3_transit_boundary%.pdf$") then
          image.attributes.width = "88%"
          flush = true
        end
        return image
      end})
      i = i + 1
    elseif next_block and next_block.t == "Table" then
      table_number = table_number + 1
      next_block.caption = caption(block, "Table")
      if table_number == 3 or table_number == 4 then
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
