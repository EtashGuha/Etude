


BeginPackage["Compile`Core`IR`FunctionInlineInformation`"]


CreateFunctionInlineInformation;
FunctionInlineInformationQ;
DeserializeFunctionInlineInformation;

Begin["`Private`"] 

Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["Compile`Utilities`Serialization`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]



(* 

FunctionInlineInformation[<|
    "InlineCost" -> Number | Infinity,
    "CanInlineQ" -> True | False,
    "InlineValue" -> True | False | Indeterminate,
    "IsTrivial" -> True | False
    ...
|>]

*)
 
RegisterCallback["DeclareCompileClass", Function[{st},
FunctionInlineInformationClass = DeclareClass[
    xFunctionInlineInformation,
    <|
    	"addMetaData" -> Function[{meta}, addMetaData[Self, meta]],
		"shouldInline" -> Function[{pm}, shouldInline[Self, pm]],
        "dispose" -> Function[{}, dispose[Self]],
        "serialize" -> (serialize[Self, #]&),
        "toString" -> Function[{}, toString[Self]],
        "toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
    |>,
    {
        "inlineCost",
        "isInlinable",
        "inlineValue",
        "isTrivial"
    },
    Predicate -> FunctionInlineInformationQ
]
]]

CreateFunctionInlineInformation[fm_] :=
    Module[ {info},
        info = CreateObject[xFunctionInlineInformation, <|
            "inlineCost" -> Missing["use the FunctionInlineCostPass to compute the cost of the function"],
            "isInlinable" -> Missing["use the FunctionIsInlinablePass to compute whether a function can be inlined"],
            "inlineValue" -> False,
			"isTrivial" -> False
        |>];
        info
    ]


(*
 Inline metadata settings
    "Inline" -> "Always"
    "Inline" -> "Never"
    "Inline" -> Automatic  leave it up to system  default
    "Inline" -> "Hint"   try to inline
*)

$inlineVals = {"Always", "Never", "Hint", Automatic}

addMetaData[self_, meta_] :=
	Module[{inline = meta["getData", "Inline", Automatic]},
		If[ !MemberQ[$inlineVals, inline], 
			inline = Automatic];
		self["setInlineValue", inline];
	]

shouldInline[self_, pm_] :=
	Module[{val = self["inlineValue"]},
		(*
		
		*)
		Switch[val,
			"Always",
				True
			,
			"Hint",   (*  Should probably look at options here *)
				True
			,
			"Never",
				False
			,
			_,       (*  value should be Automatic *)
				TrueQ[self["isTrivial"]]]
	]
 

dispose[self_] :=
    Null


getFields[self_] :=
    With[{
        fields = {
            "inlineCost",
            "isInlinable",
            "inlineValue",
            "isTrivial"
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
    
DeserializeFunctionInlineInformation[ env_, "FunctionInlineInformation"[ data_]] :=
    deserialize[env, data]

serialize[ self_, env_] :=
    With[{data = getFields[self]},
        "FunctionInlineInformation"[data]
    ]
    
deserialize[env_, data_] :=
    With[{
        info = CreateFunctionInlineInformation[None]
    },
        If[!MissingQ[data["inlineCost"]],
            info["setInlineCost", data["inlineCost"]]
        ];
        If[!MissingQ[data["isInlinable"]],
            info["setIsInlinable", data["isInlinable"]]
        ];
        If[!MissingQ[data["inlineValue"]],
            info["setInlineValue", data["inlineValue"]]
        ];
        If[!MissingQ[data["isTrivial"]],
            info["setIsTrivial", data["isTrivial"]]
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
                         If[StringQ[val],
                             quote[val],
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
  Style["Fun\nInlInfo", GrayLevel[0.7], Bold, 
   CurrentValue["FontCapHeight"]/
     AbsoluteCurrentValue[Magnification]]], $FormatingGraphicsOptions];  
    
toBoxes[self_, fmt_] :=
    BoxForm`ArrangeSummaryBox[
        "FunctionInlineInformation",
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
