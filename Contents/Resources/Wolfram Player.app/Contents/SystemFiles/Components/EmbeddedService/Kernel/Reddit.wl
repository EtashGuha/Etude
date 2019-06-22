(* ::Package:: *)

Reddit[sub_,id_] := Module[{template,embedding},
 template = FileTemplate[ FileNameJoin[ {$TemplatesDirectory, "reddit.template"} ] ];
 embedding = TemplateApply[ template, <| "sub" -> sub, "id" -> id |> ];
 EmbeddedHTML[ embedding, ImageSize->{600,300} ]
]
