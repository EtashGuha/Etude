 Widget["Panel", WidgetGroup[{
   { Widget["TextField",  {
        "text" -> "30",
        "columns" -> 5,
        "horizontalAlignment" -> PropertyValue["RIGHT"],
        BindEvent["action", 
          Script[updatePlot[]] 
          ]
        }, WidgetLayout -> {"Stretching" -> {None, None}},
         Name -> "ruleField"],
      Widget["Slider", {
        "value" -> 5,
        "minimum" -> 0, 
        "maximum" -> 50,
        BindEvent["change", 
          Script[updatePlot[]] 
          ]
        }, WidgetLayout -> {"Stretching" -> {Maximize, None}},
         Name -> "stepSlider"]},
        
  Widget["MathPanel", {
    "preferredSize" -> Widget["Dimension", {"width" ->150, "height"->150}]
    }, WidgetLayout -> {"Stretching" -> {Maximize, Maximize}},
    Name -> "canvas"],

  BindEvent["componentResized",
    Script[updatePlot[]] 
    ],
    
  BindEvent["endModal",
    Script[ Show[expr] ] 
    ],
    
  Script[
  
    RasterGraphics[(state_)?MatrixQ, colors_Integer:2, opts___] :=
         Graphics[Raster[Reverse[1 - state/(colors - 1)]], 
           AspectRatio -> (AspectRatio /. {opts} /. AspectRatio -> Automatic), opts];

    updatePlot[] := Block[{steps, ruleNum, $DisplayFunction = Identity},
      ruleNum = Round[ToExpression[PropertyValue[{"ruleField", "text"}]]];
      ruleNum = Min[Max[ruleNum, 0], 255];
      SetPropertyValue[{"ruleField", "text"}, ToString[ruleNum, InputForm] ];
      
      steps = Round[ToExpression[PropertyValue[{"stepSlider", "value"}]]];

      expr = Show[ RasterGraphics[ CellularAutomaton[ruleNum, {{1},0}, steps]],
          PlotRange -> All];
          
      SetPropertyValue[{"canvas", "mathCommand"}, ToString[expr, InputForm]];
      InvokeMethod[{"canvas", "repaintNow"}];
      ];
          
    updatePlot[]
    ]
  }, WidgetLayout -> Column], 
  WidgetLayout -> {"Stretching" -> {True, True}}
]