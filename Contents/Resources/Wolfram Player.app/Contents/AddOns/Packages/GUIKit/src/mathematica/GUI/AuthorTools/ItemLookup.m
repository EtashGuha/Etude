Widget["Frame", WidgetGroup[{
  "title" -> "HelpBrowser Item Lookup",

  WidgetGroup[{        
    { Widget["TextField", {
        "columns" -> 15,
        "toolTipText" -> "Enter phrase to use for lookup",
        "text" -> "",
        PropertyValue[{"lookupTextField", "document"}, Name -> "myDocument"]}, 
        Name -> "lookupTextField"],
        
      Widget["Timer", {
        "delay" -> 0400},
        Name -> "myTimer"],
     
     Widget["Label", {"text" -> "in"}],
     Widget["ComboBox", {
       "prototypeDisplayValue" -> "The Mathematica Book ",
       "items" -> {
         "All Help Categories",
         "Built-in Functions",
         "Add-ons & Links",
         "The Mathematica Book",
         "Front End",
         "Getting Started",
         "Tour",
         "Demos",
         "Master Index"},
       "toolTipText" -> "Which part of the Help Browser to search", 
       "editable" -> False,
       "maximumRowCount" -> 10},
       Name -> "lookupInCategory"]

    },

   {
     {Widget["CheckBox", {
       "text" -> "Search Item Names",
       "toolTipText" -> "Whether to include item names in the lookup",
       "selected" -> True,
       "enabled" -> False,
       "visible"->True},
       Name -> "lookupItemNamesQ"
     ],
     
     Widget["CheckBox", {
       "text" -> "Search Category Names",
       "toolTipText" -> "Whether to include category names in the lookup",
       "selected" -> False,
       "visible"->True},
       Name -> "lookupCategoryNamesQ"
     ]},

     WidgetSpace[15],
     WidgetFill[],

     {Widget["CheckBox", {
       "text" -> "Match Whole Name",
       "toolTipText" -> "Whether to look inside name strings or not",
       "selected" -> False,
       "visible"->True},
       Name -> "wholeNameSelected"],
      
      Widget["CheckBox", {
       "text" -> "Automatic Navigation",
       "toolTipText" -> "Turn this on to navigate the Help Browser with single-clicks in the table below.",
       "selected" -> True,
       "visible"->True},
       Name -> "autoNavigationSelected"]
     }
     
    }
    
   }, WidgetLayout -> {
      "Grouping" -> Column,
      "Border" -> { "Lookup Phrase:", {{2,2},{0,0}}} }],

  
  Widget["Label", {"text" -> " "}, Name -> "resultSummary"],
      
  
  Widget["ScrollPane", {
    "preferredSize" -> Widget["Dimension", {"width" -> 350, "height" -> 150}],
    "viewportView" -> 
      Widget["DisplayOnlyTable", {
        "columnSortable" -> True,
        PropertyValue["model", Name -> "notebookResultTableModel"],
        PropertyValue["selectionModel", Name -> "notebookResultTableSelectionModel"],
        "selectionMode" -> PropertyValue[{"notebookResultTableSelectionModel", "SINGLE_SELECTION"}]
        }, Name -> "notebookResultTable"]
      }, Name -> "notebookResultScrollPane"],

  Widget["TextArea", {
    "text" -> " ",
    "editable" -> False,
    "enabled" -> False,
    "rows" -> 0,
    "lineWrap" -> True,
    "wrapStyleWord" -> True},
    Name -> "statusResultText"],
    
  (* This GUIKit definition calls external AuthorTools NotebookSearch[] *)
  Script[ Needs["AuthorTools`Experimental`"]; ],
  
  
  (* To manage the Mathematica code we store it in a relative separate file *)
  Script[{}, ScriptSource -> "ItemLookupScriptCode.m"]
  
    
    
    
  }, WidgetLayout -> {"Border" -> {{2,2},{0,0}} }], 
Name -> "frame"]
