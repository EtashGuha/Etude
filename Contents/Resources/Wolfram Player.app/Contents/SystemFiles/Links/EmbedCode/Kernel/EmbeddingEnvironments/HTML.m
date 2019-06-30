(* Embedding for Java language using no extra libs. *)

(* Pick a unique private context for each implementation file. *)
Begin["EmbedCode`HTML`Private`"]


iframeTemplate = StringTemplate["<iframe src=\"`src`\" width=\"`width`\" height=\"`height`\"></iframe>"];

EmbedCode`Common`iEmbedCode["html", expr_, uri_, opts:OptionsPattern[]] :=
    Module[{size, width, height, template},
        size = OptionValue[ImageSize];
        If[size === Automatic,
            width = 600; height = 800,
        (* else *)
            If[!MatchQ[size, {_Integer?Positive, _Integer?Positive}],
                Message[EmbedCode::invsize, size];
                Return[$Failed];
            ];
            {width, height} = size
        ];
        Association[
            {"EnvironmentName" -> "HTML",
             "CodeSection" -> <|"Content" -> iframeTemplate[<|"src"->URLBuild[uri, {"_embed" -> "iframe"}], "width"->width, "height"->height|>],
                                (* Leave out Description field to indicate that there should be no description text (redundant when no lib files are present) *)
                                "Title" -> Automatic,
                                "Filename" -> "cloudembed.html"
                              |>
            }
        ]
    ]
    
(* EmbedCode for Data Drop *)
EmbedCode`Common`iEmbedCode["html", databin_Databin, uri_, opts:OptionsPattern[]] :=
    Module[{size, width, height, requestUrl},
        size = OptionValue[ImageSize];
        If[size === Automatic,
            width = 600; height = 800,
        (* else *)
            If[!MatchQ[size, {_Integer?Positive, _Integer?Positive}],
                Message[EmbedCode::invsize, size];
                Return[$Failed];
            ];
            {width, height} = size
        ];
        requestUrl = TemplateApply[
        	"https://datadrop.wolframcloud.com/api/v1.0/Recent?Bin=`binId`&_exportform=JSON",
        	<|"binId" -> databin[[1]]|>
        ];
        Association[
            {"EnvironmentName" -> "HTML",
             "CodeSection" -> <|"Content" -> iframeTemplate[<|"src"->URLBuild[requestUrl, {"_embed" -> "iframe"}], "width"->width, "height"->height|>],
                                (* Leave out Description field to indicate that there should be no description text (redundant when no lib files are present) *)
                                "Title" -> Automatic,
                                "Filename" -> "cloudembed.html"
                              |>
            }
        ]
    ]
    

End[]

