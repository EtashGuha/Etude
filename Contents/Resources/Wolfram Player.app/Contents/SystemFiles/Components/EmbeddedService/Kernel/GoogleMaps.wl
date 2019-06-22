(* ::Package:: *)

GoogleMaps[pos_List] := Module[{},
 template = FileTemplate[ FileNameJoin[ {$TemplatesDirectory, "googlemaps.template"} ] ];
 embedding = TemplateApply[ template, <| "lat" -> Part[pos,1], "lon" -> Part[pos,2] |> ];
 EmbeddedHTML[ embedding , ImageSize->{620,470}]
]
