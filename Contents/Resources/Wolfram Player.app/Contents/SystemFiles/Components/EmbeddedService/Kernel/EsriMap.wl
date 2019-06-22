(* ::Package:: *)

EsriMap[pos_List] := Module[{template, embedding},
 template = FileTemplate[ FileNameJoin[ {$TemplatesDirectory, "esrimap.template"} ] ];
 embedding = TemplateApply[ template, <| "lat" -> Part[pos,1], "lon" -> Part[pos,2]|> ];
 EmbeddedHTML[ embedding , ImageSize->{1370,420}]
]
