
BeginPackage["TypeFramework`TypeObjects`TypeVariable`"]


CreateTypeVariable
TypeVariableQ
TypeVariableClassName

Begin["`Private`"]

Needs["TypeFramework`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["TypeFramework`TypeObjects`TypeBase`"]
Needs["TypeFramework`TypeObjects`Kind`"]
Needs["TypeFramework`Utilities`Error`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`TypeObjects`TypeForAll`"]


sameQ[self_, other_?TypeVariableQ] :=
	self["id"] === other["id"]

sameQ[arg___] :=
	Module[{},
		False
	]


RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
DeclareClass[
	TypeVariableClassName,
	<|
		"free" -> (computeFree[Self]&),
		"computeFree" -> (computeFree[Self]&),
		"variableCount" -> (variableCount[Self]&),
		"format" -> (format[Self, ##]&),
		"sameQ" -> (sameQ[Self, ##]&),
        "toScheme" -> (toScheme[Self]&),
		"unresolve" -> Function[ {}, unresolve[Self]],
		"skolemVariableQ" -> (skolemVariableQ[Self]&),
        "metaVariableQ" -> (metaVariableQ[Self]&),
		"clone" -> (clone[Self, ##]&),
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
	|>,
	{
	    "kind",
		"variablename"
	},
	Predicate -> TypeVariableQ,
	Extends -> TypeBaseClass
];
RegisterTypeObject[TypeVariableClassName];
]]


CreateTypeVariable[name_String] :=
    CreateTypeVariable[name, CreateUnknownKind[]];
CreateTypeVariable[name_String, kind_?KindQ] :=
	CreateObject[TypeVariableClassName, <|
        "id" -> GetNextTypeId[],
        "kind" -> kind,
        "variablename" -> name
    |>]

CreateTypeVariable[args___] :=
	TypeFailure["TypeVariable", "Unrecognized call to CreateTypeVariable", args]



free[self_] :=
	(
	If[self["freeCache"] === Null,
		self["setFreeCache", self["computeFree"]]];
	self["freeCache"]
	)


computeFree[self_] :=
	If[SkolemKindQ[self["kind"]], 
		<||>, 
		<|self["id"] -> self|>]
	
	
clone[self_] :=
	clone[self, CreateReference[<||>]];

clone[self_, varmap_] :=
	If[varmap["keyExistsQ", self["id"]],
		varmap["lookup", self["id"]],
		With[{
			tv = CreateTypeVariable[self["variablename"], self["kind"]]
		},
			tv["setProperties", self["properties"]["clone"]];
			varmap["associateTo", self["id"] -> tv];
			tv
		]
	];

unresolve[ self_] :=
	TypeSpecifier[ TypeVariable[ self["variablename"]]]

skolemVariableQ[self_] :=
    SkolemKindQ[self["kind"]];
metaVariableQ[self_] :=
    TrueQ[self["getProperty", "MetaVariable"]];

(**************************************************)

variableCount[self_] :=
	1


toScheme[self_] :=
    CreateTypeForAll[{}, self];

(**************************************************)

icon := Graphics[Text[
	Style["TVar",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]

toBoxes[typ_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"TypeVariable",
		typ,
  		icon,
		{
            BoxForm`SummaryItem[{Pane["id: ", {90, Automatic}], typ["id"]}],
			BoxForm`SummaryItem[{Pane["name: ", {90, Automatic}], 
			                                     Which[
			                                         metaVariableQ[typ],
				                                         Style[
				                                             typ["variablename"],
				                                             FontWeight->Bold,
				                                             FontColor -> RGBColor[0.263, 0.537, 0.345]
				                                         ],
                                                     skolemVariableQ[typ],
                                                         Style[
                                                             typ["variablename"],
                                                             FontWeight->Bold,
                                                             FontColor -> Red
                                                         ],
                                                     True,
                                                        typ["variablename"]
			                                     ]
			}],
            If[skolemVariableQ[typ],
                BoxForm`SummaryItem[{Pane["skolem: ", {90, Automatic}], True}],
                Nothing
            ],
			If[metaVariableQ[typ],
                BoxForm`SummaryItem[{Pane["metaVariable: ", {90, Automatic}], True}],
                Nothing
			],
            BoxForm`SummaryItem[{Pane["kind: ", {90, Automatic}], typ["kind"]["toString"]}]
  		},
  		{},
  		fmt
  	]


toString[typ_] :=
    "v" <> ToString[typ["id"]]


format[ self_, shortQ_:True] :=
	If[shortQ, "v" <> ToString[self["id"]], self["variablename"]]

End[]

EndPackage[]

