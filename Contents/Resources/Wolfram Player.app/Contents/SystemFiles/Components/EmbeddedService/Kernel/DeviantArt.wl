(* ::Package:: *)

DeviantArt[id_String] :=  Module[{template, embedding},
template = FileTemplate[ FileNameJoin[ {$TemplatesDirectory, "deviantart.template"} ] ];
embedding = TemplateApply[template, <| "id" ->  id  |>];
EmbeddedHTML[embedding, ImageSize->{480,300} ]
]
