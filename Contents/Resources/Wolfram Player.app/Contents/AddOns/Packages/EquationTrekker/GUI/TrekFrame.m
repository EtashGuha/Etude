Widget["Frame", {
  "title" -> "EquationTrekker",

    WidgetGroup[{
     Widget["TrekCanvas", {
      PropertyValue[{"canvas", "canvasPane"}, Name -> "trekPane"],
      PropertyValue[{"trekPane", "trekController"}, Name -> "trekController"]
      }, Name -> "canvas",
      WidgetLayout -> {"Stretching" -> {Maximize, Maximize}}],

     {
     WidgetSpace[5],

     {WidgetFill[], Widget["ImageLabel", Name -> "typesetEquationsLabel"], WidgetFill[]},

     WidgetSpace[5],

     WidgetGroup[{
      Widget["ScrollPane", {
        "viewportView" -> 
          Widget["class:com.wolfram.guikit.trek.ParameterPanel", {}, Name -> "parameterPanel"],
         "preferredSize" -> Widget["Dimension", {"width" -> 140, "height" -> 170}],
         "minimumSize" -> Widget["Dimension", {"width" -> 50, "height" -> 60}]
         }, WidgetLayout -> {"Stretching" -> {Maximize, Maximize}}]
      }, WidgetLayout -> {"Border" -> "Parameters"}],

      Widget["class:com.wolfram.guikit.trek.TrekInspectorPanel", Name -> "trekInspectorPanel",
        InitialArguments -> {WidgetReference["trekPane"]}]
     }
    },  WidgetLayout -> {"Grouping" -> {Split, Horizontal}}],
  
  Widget["class:com.wolfram.guikit.trek.TrekToolBar", {
    "name" -> "EquationTrekker Toolbar",
    "parentFrame" -> WidgetReference["frame"]
    },
    Name -> "trekToolBar",
    InitialArguments -> {WidgetReference["trekPane"]}],
    
  (* This produces result content when the user interface ends from a modal session
     with the kernel *)
  BindEvent["endModal",
    Script[ CreateEndModalResults[]] ],
      
  Script[
   Needs["EquationTrekker`"];
   ],
  (* This external script file contains all the Mathematica code,
     separated out for easier editing *)
  Script[{}, ScriptSource -> "TrekFrameScriptCode.m"]
    
}, Name -> "frame"]
