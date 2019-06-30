Widget["Panel", {
  WidgetGroup[{
   Widget["TextField", {
      "text" -> "0", "columns" -> 6,
      "horizontalAlignment" -> PropertyValue["RIGHT"],
      BindEvent["action",
        Script[
          value = ToExpression[PropertyValue[{"textField", "text"}]]
          ]
        ]}, 
      Name -> "textField"],
      
    Widget["Button", {
      "label" -> "update",
      BindEvent["action", 
        Script[
          buttonClicked[]
          ]
        ]}],   

    Widget["CheckBox", {}, Name -> "checkBox"]
    }, WidgetLayout -> Row],

  Script[
    value = 0;
    buttonClicked[] := (
      If[ PropertyValue[{"checkBox", "selected"}],
        value++, value--];
      SetPropertyValue[{"textField", "text"}, ToString[value, InputForm] ])
    ]
    
  }]