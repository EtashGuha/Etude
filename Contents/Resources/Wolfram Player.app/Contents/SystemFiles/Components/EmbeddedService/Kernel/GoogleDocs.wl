(* ::Package:: *)

GoogleDocs[id_] := Module[{},
 template = FileTemplate[ FileNameJoin[ {$TemplatesDirectory, "googledocs.template"} ] ];
 embedding = TemplateApply[ template, <| "id" -> id |> ];
 EmbeddedHTML[ embedding, ImageSize->{800,500} ]
]
