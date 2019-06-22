Widget["Frame", {
  "title" -> "ActionApp",

  (* Here we define the Action objects along with all their properties and event bindings *)
  Widget["Action", {
    (* Buttons and menu items will use an action's name property for their text property *)
    "name" -> "Stop",
    (* Buttons and menu items will use an action's icon as their icon *)
    "icon" -> Widget["Icon", {"path" -> "Stop16.gif"}],
    (* Buttons and menu items will use an action's short description as tool tips *)
    "shortDescription" -> "Switch to stopped state",
    (* Menu items will use an action's accelerator for their accelerator keystroke *)
    "accelerator" -> Widget["MenuShortcut", InitialArguments -> {"T"}],
    (* Buttons and menu items will use an action's action for their action *)
    BindEvent["action",
      Script[ 
        setRunning[False];
        InvokeMethod[{"messageArea", "append"}, "Stop called...\n"];
        ]]
    }, Name -> "stopAction"],
  Widget["Action", {"name" -> "Start",
    "icon" -> Widget["Icon", {"path" -> "Play16.gif"}],
    "shortDescription" -> "Switch to running state",
    "accelerator" -> Widget["MenuShortcut", InitialArguments -> {"S"}],
    BindEvent["action",
      Script[ 
        setRunning[True];
        InvokeMethod[{"messageArea", "append"}, "Start called...\n"];
        ]]
    }, Name -> "startAction"],
  Widget["Action", {"name" -> "Forward",
    "icon" -> Widget["Icon", {"path" -> "StepForward16.gif"}],
    "shortDescription" -> "Move forward one step",
    "accelerator" -> Widget["MenuShortcut", InitialArguments -> {"F"}],
    BindEvent["action",
      InvokeMethod[{"messageArea", "append"}, "Forward called...\n"]
      ]
    }, Name -> "stepForwardAction"],
  Widget["Action", {"name" -> "Back",
    "icon" -> Widget["Icon", {"path" -> "StepBack16.gif"}],
    "shortDescription" -> "Move back one step",
    "accelerator" -> Widget["MenuShortcut", InitialArguments -> {"B"}],
    BindEvent["action",
      InvokeMethod[{"messageArea", "append"}, "Back called...\n"]
      ]
    }, Name -> "stepBackAction"],
  Widget["Action", {"name" -> "Clear",
      "icon" -> Widget["Icon", {"path" -> "Delete16.gif"}],
      "shortDescription" -> "Clear message area",
      "accelerator" -> Widget["MenuShortcut", InitialArguments -> {"K"}],
      BindEvent["action",
        SetPropertyValue[{"messageArea", "text"}, ""]
        ]
    }, Name -> "clearAction"],
     
  (* This is a scrolling text area where we log our messages *)
  Widget["ScrollPane", {
    "viewportView" -> 
      Widget["TextArea", {
        "columns" -> 30,
        "rows" -> 12}, Name -> "messageArea"]
    }],
    
  (* Here we create a set of buttons within the panel layout associated with the actions,
     We set the icons to Null because we only want action text in these buttons *)
  {
    WidgetFill[],
    Widget["Button", {"action" -> WidgetReference["stopAction"], "icon" -> Null}],
    Widget["Button", {"action" -> WidgetReference["startAction"], "icon" -> Null}],
    WidgetSpace[5],
    Widget["Button", {"action" -> WidgetReference["stepBackAction"], "icon" -> Null}],
    Widget["Button", {"action" -> WidgetReference["stepForwardAction"], "icon" -> Null}],
    WidgetSpace[10],
    Widget["Button", {"action" -> WidgetReference["clearAction"], "icon" -> Null}],
    WidgetFill[]
  },

  (* Here we create a menu bar with menu items associated with the actions *)
  "menus" -> 
    Widget["MenuBar", {
      Widget["Menu", {
        "text" -> "Actions",
        Widget["MenuItem", {"action" -> WidgetReference["stopAction"]}], 
        Widget["MenuItem", {"action" -> WidgetReference["startAction"]}], 
        Widget["MenuSeparator"], 
        Widget["MenuItem", {"action" -> WidgetReference["stepForwardAction"]}], 
        Widget["MenuItem", {"action" -> WidgetReference["stepBackAction"]}], 
        Widget["MenuSeparator"],
        Widget["MenuItem", {"action" -> WidgetReference["clearAction"]}]
        }]
      }],
  
  (* Here we define a contextual popup menu associated with the actions *)
  Widget["PopupMenu", {
    Widget["MenuItem", {"action" -> WidgetReference["stopAction"]}], 
    Widget["MenuItem", {"action" -> WidgetReference["startAction"]}], 
    Widget["MenuSeparator"], 
    Widget["MenuItem", {"action" -> WidgetReference["stepForwardAction"]}], 
    Widget["MenuItem", {"action" -> WidgetReference["stepBackAction"]}], 
    Widget["MenuSeparator"],
    Widget["MenuItem", {"action" -> WidgetReference["clearAction"]}]
    }, Name -> "popupMenu"],
  
  
	(* Here we define a toolbar associated with the actions 
		 We explicitly set the text of toolbar buttons to "" so they will
		 only show the action icons
	 *)
	Widget["ToolBar", {
		Widget["Button", {"action" -> WidgetReference["stopAction"], "text" -> ""}],
		Widget["Button", {"action" -> WidgetReference["startAction"], "text" -> ""}],
		Widget["MenuSeparator"], 
		Widget["Button", {"action" -> WidgetReference["stepBackAction"], "text" -> ""}],
		Widget["Button", {"action" -> WidgetReference["stepForwardAction"], "text" -> ""}],
		Widget["MenuSeparator"], 
		Widget["Button", {"action" -> WidgetReference["clearAction"], "text" -> ""}],
		"floatable" -> True,
		"rollover" -> True
	}, Name -> "toolBar"],
    
  (* Here we hook up mouse bindings to the messageArea to allow the
     popup menu to appear when appropriate *)
  BindEvent[{"messageArea", "mousePressed"},
    Script[
      If[ TrueQ[ PropertyValue[{"#", "popupTrigger"}]],
        InvokeMethod[{"popupMenu", "show"}, 
          PropertyValue[{"#", "component"}], PropertyValue[{"#", "x"}], PropertyValue[{"#", "y"}]]];
      ]
    ],
  BindEvent[{"messageArea", "mouseReleased"},
    Script[
      If[ TrueQ[ PropertyValue[{"#", "popupTrigger"}]],
        InvokeMethod[{"popupMenu", "show"}, 
          PropertyValue[{"#", "component"}], PropertyValue[{"#", "x"}], PropertyValue[{"#", "y"}]]];
      ]
    ],
    
  Script[
    (* Note that by simply setting the enabled state of the actions, all components
       connected to the action will update their enabled state appropriately *)
    setRunning[isActive_] := (
      SetPropertyValue[{"stopAction", "enabled"}, TrueQ[isActive]];
      SetPropertyValue[{"startAction", "enabled"}, !TrueQ[isActive]];
      SetPropertyValue[{"stepForwardAction", "enabled"}, !TrueQ[isActive]];
      SetPropertyValue[{"stepBackAction", "enabled"}, !TrueQ[isActive]];
      );
    
    (* He we setup the intial stopped action state *)
    setRunning[False];
    ]
    
  }, Name -> "actionFrame"]