Widget["Panel", 
  { 
   WidgetGroup[{
     WidgetGroup[{
         WidgetGroup[{
             {Widget["Label", {"text" -> "A"},
                    WidgetLayout -> {"Alignment" -> {Right, Automatic}}],
               Widget["TextField", {"columns" -> 10}],
               Widget["Label", {"text" -> "ft"}]},
             {Widget["Label", {"text" -> "BB"},
                   WidgetLayout -> {"Alignment" -> {Right, Automatic}}],
               Widget["TextField", {"columns" -> 10}],
               Widget["Label", {"text" -> "in"}]},
             {Widget["Label", {"text" -> "CCC"}, 
                   WidgetLayout -> {"Alignment" -> {Right, Automatic}}],
               Widget["TextField", {"columns" -> 10}],
               Widget["Label", {"text" -> "m"}]}
               }, WidgetLayout -> {
                    "Grouping" -> Grid,
                    "Border" ->  {{ RGBColor[1, 0, 0], 3}, {{5, 5}, {5, 5}} }}
                        ],
         WidgetFill[]
         }, WidgetLayout -> {
                  "Border" -> {{RGBColor[1, 1, 0], 3}, {{5, 5}, {5, 5}} }
             }],
     WidgetSpace[5],
     {
       WidgetFill[],
       Widget["Button", {"text" -> "foo"}]
       }
     }, WidgetLayout -> {
            "Grouping" -> Row,
            "Border" ->  {{ RGBColor[0, 0, 1], 3}, {{5, 5}, {5, 5}} }
         }]
  }
]