Widget["Frame", {
  WidgetGroup[{
    Widget["Label", {
      "text" -> WidgetReference["#2"]
      }, Name -> "label"],
    Widget["ProgressBar", {
      "minimum" -> 0, 
      "maximum" -> 100,
      "value" -> WidgetReference["#3"],
      "preferredSize" -> 
        Widget["Dimension", {"width" -> 300, "height" -> 25}]
      }, Name -> "bar"] 
    }, WidgetLayout -> {
        "Grouping" -> Column, 
        "Border" -> {{15, 15}, {25, 20}}}
    ],

  "title" -> WidgetReference["#1"],
  "resizable" -> False},

  Name -> "frame"]
  