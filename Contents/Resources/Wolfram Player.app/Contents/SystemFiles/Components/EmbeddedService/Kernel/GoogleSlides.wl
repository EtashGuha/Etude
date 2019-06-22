(* ::Package:: *)

GoogleSlides[id_] := Module[{},
 template = FileTemplate[ FileNameJoin[ {$TemplatesDirectory, "googleslides.template"} ] ];
 embedding = TemplateApply[ template, <| "id" -> id |> ];
 EmbeddedHTML[ embedding , ImageSize->{960+20,569+20}]
]
