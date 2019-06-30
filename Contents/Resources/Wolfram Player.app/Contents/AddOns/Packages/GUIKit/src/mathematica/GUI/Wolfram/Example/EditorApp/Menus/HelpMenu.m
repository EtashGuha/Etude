Widget["Menu", {
  "text" -> "Help",
  
  Widget["MenuItem", {
    "text" -> "About Text Editor...",
    BindEvent["action", 
      Script[
        If[ WidgetReference["helpDialog"] === Null,
          Widget["HelpDialog", Name -> "helpDialog", 
            InitialArguments -> {Widget["HelpString"]}];
          ];
        InvokeMethod[{"helpDialog", "show"}];
        ]
      ]
    }]
  
}]
