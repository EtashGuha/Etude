Widget["Panel", 
{
  Widget["Label", {"text" -> "These are inputs"},  
      WidgetLayout -> {"Alignment" -> Center},
      Name -> "mainLabel"],
  {
    Widget["Label", {"text" -> "BB"}],
    WidgetAlign[],
    Widget["TextField", {"columns" -> 10}],
    WidgetAlign[],
    Widget["Label", {"text" -> "in"}],
    WidgetAlign[],
    Widget["CheckBox", {"label" -> "round"}]
    },
  {
    Widget["Label", {"text" -> "CCC"}],
    WidgetAlign[],
    Widget["TextField", {"columns" -> 10}],
    WidgetAlign[],
    Widget["Label", {"text" -> "m"}],
    Widget["Label", {"text" -> " OR "}],
    WidgetAlign[],
    Widget["TextField", {"columns" -> 10}],
    Widget["Label", {"text" -> "km"}]
    },
  {
    WidgetFill[],
    WidgetAlign[{"mainLabel", After}, Before],
    Widget["Button", {"text" -> "Submit"}],
    WidgetFill[]
    }
  }
]
