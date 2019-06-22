Widget["Frame", {
  Widget["Graph/GraphPanel", {
    "preferredSize" -> Widget["Dimension", {"width" -> 300, "height" -> 400}]
    },
    ExposeWidgetReferences -> {"graphModel"},
    Name -> "graphPanel"],
 
  BindEvent["endModal",
    Script[ 
      PropertyValue[{"graphPanel", "expr"}] 
      ]
    ]
    
 }, Name -> "graphFrame"]