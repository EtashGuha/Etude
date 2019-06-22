Widget["class:java.util.ArrayList", {
  InvokeMethod["addAll",
    InvokeMethod[{"class:java.util.Arrays", "asList"},
      {Widget["Label", {"text" -> "Label1"}, WidgetLayout -> None], 
      "string1", 
      "string2", 
      Widget["Label", {"text" -> "Label2"}, WidgetLayout -> None], 
      Widget["class:java.util.Date"]}]
    ], 
  
  Script[
    InvokeMethod[{"myList", "add"}, #]& /@
      {ToString[10!], 34};
    ]
  }, Name -> "myList"]