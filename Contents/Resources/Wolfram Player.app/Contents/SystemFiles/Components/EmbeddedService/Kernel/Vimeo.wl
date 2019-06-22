(* ::Package:: *)

Vimeo[id_] := Module[{template,embedding},
 template = FileTemplate[ FileNameJoin[ {$TemplatesDirectory, "vimeo.template"} ] ];
 embedding = TemplateApply[ template, <| "id" -> id |> ];
 EmbeddedHTML[ embedding, ImageSize->1.1*({500,281}) ]
]
