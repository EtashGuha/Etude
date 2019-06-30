Widget["Frame", { 
   "title" -> "Get angle...",
   "resizable" -> False,
   
   WidgetGroup[{
     Widget["Label", {"text" -> "Enter an angle:"}], 
     Widget["TextField", {
       "horizontalAlignment" -> PropertyValue["RIGHT"]
       }, Name -> "inputField"]
     }, WidgetLayout -> {"Border" -> {{15,15}, {10, 5}}}  ], 
       
   WidgetGroup[{
     Widget["RadioButton",
       {"text" -> "degrees", "selected" -> True}, 
         Name -> "degreesButton"],
     Widget["RadioButton", 
        {"text" -> "radians", "selected" -> False}, 
         Name -> "radiansButton"]
      }, WidgetLayout -> {
            "Border" -> { {{15,15},{5,5}}, "Units", {{15,15},{5,5}}},
            "Stretching" -> {Maximize, Maximize}}],
      
    (* We create an action component here because we can share
       it as the action property of the Cancel button and as an
       action binding with the Escape keypress within the dialog *)
    Widget["Action", {
      "name" -> "Cancel",
      BindEvent["action", 
        Script[
          returnValue = False; 
          InvokeMethod[{"frame", "dispose"}]; ]]
      }, Name -> "cancelAction"],
   
   WidgetGroup[{
    WidgetFill[],
    
    Widget["Button", 
      {"text" -> "OK",
      BindEvent["action", 
       Script[
        returnValue = True; 
        InvokeMethod[{"frame", "dispose"}]; ]]
     }, Name -> "okButton"],
      
    WidgetSpace[5],
    
    Widget["Button", 
      {"action" -> WidgetReference["cancelAction"],
       "text" -> "Cancel"
      }, Name -> "cancelButton"]
      
    }, WidgetLayout -> {"Border" -> {{5,5}, {5, 5}}}],

      
   (* A group of radio buttons where only one can be selected at a given time
      must be added and managed by a ButtonGroup component.
      A ButtonGroup component is not a GUI component that is added to a parent or layout at all
    *)
  Widget["ButtonGroup", {
    WidgetReference["degreesButton"], 
    WidgetReference["radiansButton"]
    }],
       
  PropertyValue[{"frame", "rootPane"}, Name -> "rootPane"],
  SetPropertyValue[{"rootPane", "defaultButton"}, WidgetReference["okButton"]],
  
  (* We register the Escape Key and a unique ID action command name with the input map of the rootPane
     that is checked whenever the dialog is in focus *)
  InvokeMethod[{"rootPane", "getInputMap"}, PropertyValue[{"rootPane", "WHEN_IN_FOCUSED_WINDOW"}], Name -> "inputMap"],
  InvokeMethod[{"inputMap", "put"}, 
    Widget["KeyStroke", InitialArguments -> {"ESCAPE"}], "window.canceled"],
    
  (* We then register the unique ID action commmand name and the action component to be called
     when needed and when the action component is enabled *)
  PropertyValue[{"rootPane", "actionMap"}, Name -> "actionMap"],
  InvokeMethod[{"actionMap", "put"}, 
    "window.canceled", WidgetReference["cancelAction"]],
    
  BindEvent["endModal", 
    Script[
        If[TrueQ[returnValue], 
          angle = ToExpression[PropertyValue[{"inputField", "text"}]]; 
          If[angle =!= Null && PropertyValue[{"degreesButton", "selected"}], 
              angle *= Pi/180], 
          angle = Null]; 
        returnValue = False; 
        angle
     ]
   ]
 
  }, 
Name -> "frame"]