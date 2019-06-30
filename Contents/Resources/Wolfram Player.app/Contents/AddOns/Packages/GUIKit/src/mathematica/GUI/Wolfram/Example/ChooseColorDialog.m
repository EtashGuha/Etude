Widget["Panel", {

  Widget["ColorChooser", {
    PropertyValue[{"chooser", "selectionModel"}, 
      Name -> "colorSelectionModel"],
    BindEvent[{"colorSelectionModel", "change"}, Script[updateColor[]]]
    }, Name -> "chooser"],
      
  {
    Widget["TextField", {
      "text" -> "    ", "editable" -> False},
      WidgetLayout -> {"Stretching" -> {None, None}
      }, Name -> "myColorWell"],
      
    Widget["ComboBox", {
      "items" -> {"RGBColor", "GrayLevel", "CMYKColor"},
      BindEvent["action", Script[updateColor[]]]
      }, Name -> "myColorModel"],
  

    Widget["TextField", {"text" -> ""
      }, Name -> "myTextField"]
  },
      
  Script[
    colorExpr = RGBColor[0, 0, 0];

    updateColor[] := Module[{newColor},
      newColor = PropertyValue[{"colorSelectionModel", "selectedColor"}];
      SetPropertyValue[{"myColorWell", "background"}, newColor ];

      colorExpr = RGBColor[ PropertyValue[{newColor, "red"}]/255.,
        PropertyValue[{newColor, "green"}]/255., PropertyValue[{newColor, "blue"}]/255.];
      colorExpr = Chop[#, 10^-5] & /@ 
        ToColor[colorExpr, ToExpression[PropertyValue[{"myColorModel", "selectedItem"}]]];
      SetPropertyValue[{"myTextField", "text"}, ToString[NumberForm[colorExpr, 5]]];
      ];
    ],

  BindEvent["endModal", Script[colorExpr]]
    
}]