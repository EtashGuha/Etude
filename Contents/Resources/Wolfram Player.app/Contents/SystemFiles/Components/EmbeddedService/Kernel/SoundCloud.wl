SoundCloud[id_] := Module[{},
 template = FileTemplate[ FileNameJoin[ {$TemplatesDirectory, "soundcloud.template"} ] ];
 embedding = TemplateApply[ template, <| "id" -> id |> ];
 EmbeddedHTML[ embedding, ImageSize -> {800,200} ]
]
