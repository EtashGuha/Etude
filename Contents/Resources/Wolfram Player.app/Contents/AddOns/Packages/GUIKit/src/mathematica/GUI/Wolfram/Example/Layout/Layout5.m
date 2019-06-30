Widget["Panel", 
  {
    { Widget["Label", {"text" -> "int"}],
      Widget["TextField", {"columns" -> 5}],
      Widget["Label", {"text" -> "double"}],
      Widget["TextField", {"columns" -> 5},
        WidgetLayout -> { "Stretching" -> {Maximize, None} }],
      Widget["Label", {"text" -> "double"}],
      Widget["TextField", {"columns" -> 5}, 
        WidgetLayout -> { "Stretching" -> {Maximize, None} }]
    },
    {
      { Widget["Label", {"text" -> "reallyreallyreallyreallyreallylonglabel"}],
        Widget["TextArea", {},
          WidgetLayout -> { "Stretching" -> {WidgetAlign, True} }]
      },
      { WidgetFill[],
        Widget["Button", {"text" -> "Submit"}]
      }
    }
  }
]