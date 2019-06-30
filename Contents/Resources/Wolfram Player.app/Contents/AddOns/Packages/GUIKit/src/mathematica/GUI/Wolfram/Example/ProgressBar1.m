Widget["Frame", {
  WidgetGroup[{
    Widget["Label", {
      "text" -> "Percent complete:"
      }, Name -> "label"],
    Widget["ProgressBar", {
      "minimum" -> 0, 
      "maximum" -> 100,
      "preferredSize" -> 
        Widget["Dimension", {"width" -> 300, "height" -> 25}]
      }, Name -> "bar"] 
    }, WidgetLayout -> {
        "Grouping" -> Column, 
        "Border" -> {{15, 15}, {25, 20}}}
    ],

  "title" -> "Computation Progress",
  "resizable" -> False},

  Name -> "frame"]