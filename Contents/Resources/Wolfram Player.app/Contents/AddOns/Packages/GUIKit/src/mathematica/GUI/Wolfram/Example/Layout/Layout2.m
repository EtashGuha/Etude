Widget["Panel", 
    WidgetGroup[{
      Widget["Label", {"text" -> "Inputs"}], 
      WidgetGroup[
        {
          { Widget["Label", {"text" -> "A"}],
            Widget["TextField", {"columns" -> 10}],
            Widget["Label", {"text" -> "ft"}],
            WidgetSpace[5]
          },
          { Widget["Label", {"text" -> "BB"}],
            Widget["TextField", {"columns" -> 10}],
            Widget["Label", {"text" -> "in"}],
            Widget["CheckBox", {"label" -> "round"}]
          },
          { Widget["Label", {"text" -> "CCC"}],
            Widget["TextField", {"columns" -> 10}],
            Widget["Label", {"text" -> "m"}],
            WidgetGroup[{
              Widget["Label", {"text" -> " OR "}],
              Widget["TextField", {"columns" -> 10}],
              Widget["Label", {"text" -> "km"}]
              }, WidgetLayout -> {
                  "Grouping" -> Row,
                  "Border" -> {{RGBColor[1, 0, 0], 3}, {{5, 5}, {5, 5}} }
                  }]
            }
          }, WidgetLayout -> Grid]
      }, WidgetLayout -> {
        "Border" -> {{RGBColor[0, 0, 1], 3}, {{5, 5}, {5, 5}} }  
      }]
]
