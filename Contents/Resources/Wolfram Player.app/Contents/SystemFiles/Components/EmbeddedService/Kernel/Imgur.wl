(* ::Package:: *)

Imgur[id_] := Module[{},
 template = FileTemplate[ FileNameJoin[ {$TemplatesDirectory, "imgur.template"} ] ];
 embedding = TemplateApply[ template, <| "id" -> id |> ];
 {height,width} = ImageDimensions@Import[StringJoin["http://i.imgur.com/",id,".jpg"]];
 EmbeddedHTML[ embedding, ImageSize->{height + 20, width + 20} ]
]
