GoogleSheets[id_] := Module[{},
 template = FileTemplate[ FileNameJoin[ {$TemplatesDirectory, "googlesheets.template"} ] ];
 embedding = TemplateApply[ template, <| "id" -> id |> ];
 EmbeddedHTML[ embedding ]
]
