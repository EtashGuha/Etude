Widget["Frame", {
  "title" -> "Calc with Mathematica",
      
 {Widget["Label", {
    "text" -> "First argument:"}], 
  WidgetAlign[],
  Widget["TextField", {
    "columns" -> "20", 
    "horizontalAlignment" -> PropertyValue["RIGHT"],
    BindEvent["action", 
      Script[calculateResults[]]]
    }, Name -> "FirstArgumentField"] }, 
      
 {Widget["Label", {
    "text" -> "Second argument:"}], 
  WidgetAlign[],
  Widget["TextField", {
    "horizontalAlignment" -> PropertyValue["RIGHT"],
    BindEvent["action", 
      Script[calculateResults[]]]
    }, Name -> "SecondArgumentField"]}, 
        
 {WidgetFill[],
  Widget["Button", {"text" -> "Compute",
    BindEvent["action", 
      Script[calculateResults[]]]
    }]},
    
  Widget["Label", {"text" -> "Results: "}], 
      
 {Widget["Label", {"text" -> "Sum: "}], 
  WidgetAlign[],
  Widget["TextField", {
    "horizontalAlignment" -> PropertyValue["RIGHT"],
    "editable" -> "false"
    }, Name -> "SumField"]}, 
 {Widget["Label", {"text" -> "Difference:"}], 
  WidgetAlign[],
  Widget["TextField", {
    "horizontalAlignment" -> PropertyValue["RIGHT"],
    "editable" -> "false"
    }, Name -> "DifferenceField"]}, 
 {Widget["Label", {"text" -> "Product:"}], 
  WidgetAlign[],
  Widget["TextField", {
    "horizontalAlignment" -> PropertyValue["RIGHT"],
    "editable" -> "false"
    }, Name -> "ProductField"]}, 
 {Widget["Label", {"text" -> "Quotient:"}], 
  WidgetAlign[],
  Widget["TextField", {
    "horizontalAlignment" -> PropertyValue["RIGHT"],  
    "editable" -> "false"
    }, Name -> "QuotientField"]}, 
  
  BindEvent["endModal", 
    Script[
      fields = {"FirstArgumentField", "SecondArgumentField", "SumField", 
          "DifferenceField", "ProductField", "QuotientField"}; 
      (#1 -> ToExpression[PropertyValue[{#1, "text"}]] & ) /@ fields
      ]
    ], 
  
  Script[
    calculateResults[] := Module[{x, y}, 
      {x, y} = ToExpression /@ PropertyValue[{{"FirstArgumentField", "SecondArgumentField"}, "text"}]; 
      SetPropertyValue[{"SumField", "text"}, ToString[x + y, InputForm]]; 
      SetPropertyValue[{"DifferenceField", "text"}, ToString[x - y, InputForm]]; 
      SetPropertyValue[{"ProductField", "text"}, ToString[x*y, InputForm]]; 
      SetPropertyValue[{"QuotientField", "text"}, ToString[x/y, InputForm]]; 
      ]; 
    ]
    
  }
]
