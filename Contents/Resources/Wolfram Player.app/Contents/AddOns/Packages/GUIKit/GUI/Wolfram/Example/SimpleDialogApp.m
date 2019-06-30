Widget["Frame", {
  "title" -> "Simple Dialog App",
  
  Widget["TextField", {"text" -> "Initial app text"}, Name -> "appTextField"],

  Widget["Dialog", {
    "title" -> "Simple Dialog",
    "modal" -> True,
   
    Widget["TextField", {
      "text" -> "Initial dialog text"
      }, Name -> "dialogTextField"],
    Widget["Button", {"text" -> "Done",
      BindEvent["action",
        InvokeMethod[{"myDialog", "dispose"}] ]
        }],
   
    InvokeMethod["center"]
    }, Name -> "myDialog",
    InitialArguments -> {WidgetReference["myFrame"]}],
      
  Widget["Button", {"text" -> "Run Modal Dialog",
    BindEvent["action", {
      SetPropertyValue[{"dialogTextField", "text"}, 
        PropertyValue[{"appTextField", "text"}] ],
      InvokeMethod[{"myDialog", "show"}]
      }]
    }],
    
  BindEvent[{"myDialog", "windowClosed"},
    SetPropertyValue[{"appTextField", "text"}, 
      PropertyValue[{"dialogTextField", "text"}]
    ]]
    
  }, Name -> "myFrame"]