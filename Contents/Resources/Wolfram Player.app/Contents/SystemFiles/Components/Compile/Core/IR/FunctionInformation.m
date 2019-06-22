


BeginPackage["Compile`Core`IR`FunctionInformation`"]


CreateFunctionInformation;
FunctionInformationQ;
DeserializeFunctionInformation;

Begin["`Private`"] 

Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["Compile`Utilities`Serialization`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`FunctionInlineInformation`"]


(* 

FunctionInformation[<|
    "inlineInformation" -> InlineInformation,
    "sideEffecting" -> ...,
    "Throws" -> ...,
    "linkage" -> ...
|>]

*)
 
RegisterCallback["DeclareCompileClass", Function[{st},
FunctionInformationClass = DeclareClass[
    xFunctionInformation,
    <|

        "addMetaData" -> Function[{meta}, addMetaData[Self, meta]],
        "dispose" -> Function[{}, dispose[Self]],
        "serialize" -> (serialize[Self, #]&),
        "toString" -> Function[{}, toString[Self]],
        "toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
    |>,
    {
        "inlineInformation",
        "linkage",
        "ArgumentAlias",
        "Throws",
        "sideEffecting"
    },
    Predicate -> FunctionInformationQ
]
]]

CreateFunctionInformation[fm_] :=
    Module[ {info},
        info = CreateObject[xFunctionInformation, <|
            "inlineInformation" -> CreateFunctionInlineInformation[fm],
            "linkage" -> Missing["use the FunctionLinkagePass to compute whether the linkage of the function"],
            "Throws" -> Missing["use the FunctionThrowsPass to compute whether the function can throw"],
            "sideEffecting" -> Missing["use the FunctionSideEffectingPass to compute whether the function has any side effects"],
            "ArgumentAlias" -> False
        |>];
        info
    ]


addMetaData[self_, meta_] :=
	Module[{argumentAlias = meta["getData", "ArgumentAlias", False] },
		self["setArgumentAlias", argumentAlias];
		self["inlineInformation"]["addMetaData", meta];
	]


   
dispose[self_] := (
    self["inlineInformation"]["dispose"];
);


getFields[self_] :=
    With[{
        fields = {
		    "inlineInformation",
		    "linkage",
		    "Throws",
		    "ArgumentAlias",
		    "sideEffecting"
        }
    },
	    Association[Map[
	        Function[{key},
	            If[MissingQ[self[key]],
	                Nothing,
	                key -> self[key]
	            ]
	        ],
	        fields
	    ]]
	];




DeserializeFunctionInformation[ env_, "FunctionInformation"[ data_]] :=
    deserialize[env, data]

serialize[ self_, env_] :=
    Module[{data = getFields[self]},
        If[KeyExistsQ[data, "inlineInformation"],
            data["inlineInformation"] = data["inlineInformation"]["serialize", env]
        ];
        "FunctionInformation"[data]
    ]
    
deserialize[env_, data_] :=
    With[{
        info = CreateFunctionInformation[None]
    },
        If[KeyExistsQ[data, "inlineInformation"],
            info["setInlineInformation", env["deserialize", data["inlineInformation"]]]
        ];
        If[KeyExistsQ[data, "linkage"],
            info["setLinkage", data["linkage"]]
        ];
        If[KeyExistsQ[data, "Throws"],
            info["setThrows", data["Throws"]]
        ];
        If[KeyExistsQ[data, "sideEffecting"],
            info["setSideEffecting", data["sideEffecting"]]
        ];
        If[KeyExistsQ[data, "ArgumentAlias"],
            info["setArgumentAlias", data["ArgumentAlias"]]
        ];
        info
    ];

toString[self_] :=
    StringJoin[
        "{",
        StringRiffle[
            KeyValueMap[
                Function[{key, val},
                    StringJoin[
                        quote[key],
                         "->",
                         Which[
                             StringQ[val],
                                quote[val],
                             FunctionInlineInformationQ[val],
                                val["toString"],
                             True,
                                ToString[val]
                         ]
                    ]
                ],
                getFields[self]
            ]
            , 
            ", "
        ],
        "}"
    ]
   
quote[k_] := "\"" <> k <> "\"" 
    
icon := icon = Graphics[Text[
  Style["Fun\nInfo", GrayLevel[0.7], Bold, 
   CurrentValue["FontCapHeight"]/
     AbsoluteCurrentValue[Magnification]]], $FormatingGraphicsOptions];  
      
toBoxes[self_, fmt_] :=
    BoxForm`ArrangeSummaryBox[
        "FunctionInformation",
        self,
        icon,
        Flatten[{
	        KeyValueMap[
	            Function[{key, val},
	                BoxForm`SummaryItem[{Pane[key <> ": ",     {90, Automatic}], val}]
	            ],
	            getFields[self]
	        ]
        }],
        {
        }, 
        fmt
    ]
    


    
End[]
EndPackage[]

