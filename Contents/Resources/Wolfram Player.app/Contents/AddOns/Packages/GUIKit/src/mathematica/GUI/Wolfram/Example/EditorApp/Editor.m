Widget["Frame", {
  "title" -> "Text Editor",
  Widget["ScrollPane", {
    "viewportView" -> Widget["TextArea", {
      "font" -> Widget["FixedWidthFont"],
      "text" -> "",
      "rows" -> 20,
      "columns" -> 50
      }, Name -> "contentTarget", WidgetLayout -> None]
    }],
  PropertyValue[{"contentTarget", "document"}, Name -> "contentTargetDocument"],
  "menus" -> Widget["EditorMenuBar", WidgetLayout -> None]
  }, Name -> "frame"]