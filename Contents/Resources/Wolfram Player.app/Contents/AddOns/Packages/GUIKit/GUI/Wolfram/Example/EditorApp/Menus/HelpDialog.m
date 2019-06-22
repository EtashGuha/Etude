Widget["Dialog", {
  "resizable" -> False,
  
  WidgetGroup[{
   Widget["Label", {
    "text" -> WidgetReference["#1"]}], 
    
   WidgetSpace[50],
   
   { WidgetFill[],
    Widget["Button", {
      "text" -> "Close",
      BindEvent["action", 
        InvokeMethod[{"dialog", "dispose"}] 
        ]
      }]
    }
   }, WidgetLayout -> {"Grouping" -> Column, "Border" -> {{5,5},{10,15}}}], 

  BindEvent["windowClosing", 
    InvokeMethod["dispose"]
    ]
  }, 
  Name -> "dialog",
  InitialArguments -> {WidgetReference["frame"], "About Text Editor", True}
]

