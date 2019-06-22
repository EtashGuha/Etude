Widget["Frame", WidgetGroup[{
  "title" -> "AuthorTools Help Browser Search",

  WidgetGroup[{        
    { Widget["Button", {"text" -> "Search",
        "toolTipText" -> "Perform search", 
        
        (* cancel binding will interrupt kernel from Java *)
        BindEvent["action",
          InvokeMethod[{"ScriptEvaluator", "abort"}],
          InvokeThread -> "New",
          Name -> "cancelActionListener"]
        }, 
        WidgetLayout -> {"Stretching" -> {None, None}},
        Name -> "searchButton"],  
        
      Widget["TextField", {
        "toolTipText" -> "Enter keywords to use for search"}, 
        Name -> "searchTextField"]
    },

   { Widget["Label", {"text" -> "Match:",
      "horizontalAlignment" -> PropertyValue["RIGHT"]}],
                        
     Widget["ComboBox", {
      "toolTipText" -> "Multiple words search mode",
      "prototypeDisplayValue" -> "At least one ",
      "items" -> {"At least one", "All"},
      "editable" -> False
      }, Name -> "multiWordSelect"],

     Widget["Label", {"text" -> " Results as:",
      "horizontalAlignment" -> PropertyValue["RIGHT"]}],
                         
     Widget["ComboBox", { 
      "toolTipText" -> "Choose how results will be displayed",
      "prototypeDisplayValue" -> "Notebook ",
      "items" -> {"Table", "Notebook"},
      "editable" -> False
      }, Name -> "resultModeSelect"]
    }
    
   }, WidgetLayout -> {
      "Grouping" -> Column,
      "Border" -> { "Keywords:", {{2,2},{0,0}}} }],

  WidgetGroup[{
    WidgetGroup[{ 
      Widget["CheckBox", {"text" -> "Built-in Functions",  "selected" -> False}, 
        Name -> "category_RefGuide", WidgetLayout -> {"Stretching" -> {Maximize, None}}],
      Widget["CheckBox", {"text" -> "Add-ons & Links",  "selected" -> False}, 
        Name -> "category_AddOns", WidgetLayout -> {"Stretching" -> {Maximize, None}}],      
      Widget["CheckBox", {"text" -> "The Mathematica Book", "selected" -> False}, 
        Name -> "category_MainBook", WidgetLayout -> {"Stretching" -> {Maximize, None}}],
      Widget["CheckBox", {"text" -> "Front End", "selected" -> False}, 
        Name -> "category_OtherInformation", WidgetLayout -> {"Stretching" -> {Maximize, None}}] },
          WidgetLayout -> Column],
    WidgetGroup[{ 
      Widget["CheckBox", {"text" -> "Getting Started", "selected" -> True}, 
        Name -> "category_GettingStarted", WidgetLayout -> {"Stretching" -> {Maximize, None}}],
      Widget["CheckBox", {"text" -> "Demos", "selected" -> False}, 
        Name -> "category_Demos", WidgetLayout -> {"Stretching" -> {Maximize, None}}],   
      Widget["CheckBox", {"text" -> "Tour", "selected" -> False}, 
        Name -> "category_Tour", WidgetLayout -> {"Stretching" -> {Maximize, None}}],    
      Widget["CheckBox", {"text" -> "Master Index", "selected" -> False}, 
        Name -> "category_MasterIndex", WidgetLayout -> {"Stretching" -> {Maximize, None}}] },
          WidgetLayout -> Column]
    }, WidgetLayout -> {"Border" -> "Help Browser Categories:"}],
           
  WidgetGroup[{
    WidgetGroup[{ 
      Widget["CheckBox", {"text" -> "All", "selected" -> True}, 
          Name -> "cellStyle_All", WidgetLayout -> {"Stretching" -> {Maximize, None}}],
      Widget["CheckBox", {"text" -> "Input", "selected" -> False}, 
          Name -> "cellStyle_Input", WidgetLayout -> {"Stretching" -> {Maximize, None}}]},
           WidgetLayout -> Column],      
    WidgetGroup[{  
      Widget["CheckBox", {"text" -> "Text", "selected" -> False}, 
        Name -> "cellStyle_Text", WidgetLayout -> {"Stretching" -> {Maximize, None}}], 
      Widget["CheckBox", {"text" -> "Output", "selected" -> False}, 
        Name -> "cellStyle_Output", WidgetLayout -> {"Stretching" -> {Maximize, None}}]},
           WidgetLayout -> Column],  
    WidgetGroup[{  
       Widget["CheckBox", {"text" -> "MathCaption", "selected" -> False}, 
        Name -> "cellStyle_MathCaption", WidgetLayout -> {"Stretching" -> {Maximize, None}}],
       Widget["CheckBox", {"text" -> "Usage", "selected" -> False}, 
        Name -> "cellStyle_Usage", WidgetLayout -> {"Stretching" -> {Maximize, None}}] },
           WidgetLayout -> Column]
    }, WidgetLayout -> {"Border" -> "Cell Styles:"}],

  Widget["ScrollPane", {
    "preferredSize" -> Widget["Dimension", {"width" -> 275, "height" -> 90}],
    "viewportView" -> 
      Widget["DisplayOnlyTable", {
        "columnSortable" -> True,
        PropertyValue["model", Name -> "notebookResultTableModel"],
        PropertyValue["selectionModel", Name -> "notebookResultTableSelectionModel"],
        "selectionMode" -> PropertyValue[{"notebookResultTableSelectionModel", "SINGLE_SELECTION"}]
        }, Name -> "notebookResultTable"]
      }, Name -> "notebookResultScrollPane"],

  Widget["Label", {"text" -> " "}, Name -> "statusResultText"],
  
  (* This GUIKit definition calls external AuthorTools NotebookSearch[] *)
  Script[ Needs["AuthorTools`Experimental`"]; ],
  
  (* To manage the Mathematica code we store it in a relative separate file *)
  Script[{}, ScriptSource -> "HelpBrowserSearchScriptCode.m"]
    
  }, WidgetLayout -> {"Border" -> {{2,2},{0,0}} }], 
Name -> "frame"]
