 Widget["Panel", WidgetGroup[{
 
   {Widget["Label", {"text" -> "f[x]: ",
     "horizontalAlignment" -> PropertyValue["Right"]}],
    WidgetAlign[],
    Widget["TextField",  {
      "text" -> "Sin[x]",
      BindEvent["action", Script[updatePlot[]] ]
      }, WidgetLayout -> {"Stretching" -> {Maximize, None}},
      Name -> "functionField"]
    },
     
  {Widget["Label", {"text" -> "x min: ",
    "horizontalAlignment" -> PropertyValue["Right"]}],
   WidgetAlign[],
   Widget["TextField",  {
      "text" -> "0",
      "horizontalAlignment" -> PropertyValue["Right"],
      BindEvent["action", Script[updatePlot[]] ]
      }, WidgetLayout -> {"Stretching" -> {Maximize, None}},
        Name -> "xminField"],
   WidgetSpace[10],
   Widget["Label", {"text" -> "x max: ",
     "horizontalAlignment" -> PropertyValue["Right"]}],
   WidgetAlign[],
   Widget["TextField",  {
      "text" -> "2 Pi",
      "horizontalAlignment" -> PropertyValue["Right"],
      BindEvent["action", Script[updatePlot[]] ]
      }, WidgetLayout -> {"Stretching" -> {Maximize, None}},
      Name -> "xmaxField"]
  },
  
  {Widget["Label", {"text" -> "Line: color: ",
    "horizontalAlignment" -> PropertyValue["Right"]}],
   WidgetAlign[],
   Widget["ComboBox", {
      "items" -> {"Black", "Red", "Green", "Blue"},
      "editable" -> False,
      BindEvent["action", Script[updatePlot[]] ]
      }, WidgetLayout -> {"Stretching" -> {Maximize, None}},
      Name -> "colorName"],
  WidgetSpace[10],
  Widget["Label", {"text" -> "style: ",
     "horizontalAlignment" -> PropertyValue["Right"]}],
  WidgetAlign[],
  Widget["ComboBox", {
        "items" -> {"Solid Thin", "Solid Thick", "Dashed"},
        "editable" -> False,
        BindEvent["action", Script[updatePlot[]] ]
        }, WidgetLayout -> {"Stretching" -> {Maximize, None}},
       Name -> "lineStyleName"]
   },
   
  Widget["MathPanel", {
    "preferredSize" -> Widget["Dimension", {"width" -> 150, "height"-> 150}],
    "usesFE" -> True
    }, WidgetLayout -> {"Stretching" -> {Maximize, Maximize}},
    Name -> "canvas"],

  BindEvent["componentResized", Script[updatePlot[]] ],
  BindEvent["endModal", Script[ Show[expr] ] ],
  
  Script[
    updatePlot[] := Block[{functionExpr, xmin, xmax, col, style,
        $DisplayFunction = Identity},
      functionExpr = ToExpression[PropertyValue[{"functionField", "text"}]];
      xmin = ToExpression[PropertyValue[{"xminField", "text"}]];
      xmax = ToExpression[PropertyValue[{"xmaxField", "text"}]];
      col = ToExpression[PropertyValue[{"colorName", "selectedItem"}]];
      style = PropertyValue[{"lineStyleName", "selectedItem"}];
      Switch[ style,
        "Solid Thick", style = {col, Thickness[0.02]},
        "Dashed", style = {col, Dashing[{0.05, 0.05}]},
        _, style = {col}];
      
      expr = Show[ Plot[functionExpr, {x, xmin, xmax},
        PlotStyle -> style],
        PlotRange -> All];
      SetPropertyValue[{"canvas", "mathCommand"}, ToString[expr, InputForm]];
      InvokeMethod[{"canvas", "repaintNow"}];
      ];
      
    updatePlot[]
    ]
    
  }, WidgetLayout -> Column], 
  
 WidgetLayout -> {"Stretching" -> {True, True}}
 ]