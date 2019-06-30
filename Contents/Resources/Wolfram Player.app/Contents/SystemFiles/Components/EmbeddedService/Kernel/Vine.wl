Vine[id_] := Module[{},
 template = FileTemplate[ FileNameJoin[ {$TemplatesDirectory, "vine.template"} ] ];
 embedding = TemplateApply[ template, <| "id" -> id |> ];
 EmbeddedHTML[ embedding, ImageSize->{500,500} ]
]
