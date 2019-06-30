(* ::Package:: *)

Instagram[id_] := Module[{template,embedding},
 template = FileTemplate[ FileNameJoin[ {$TemplatesDirectory, "instagram.template"} ] ];
 embedding = TemplateApply[ template, <| "id" -> id |> ];
 EmbeddedHTML[ embedding, ImageSize->{520,520} ]
]
