
BeginPackage["TypeFramework`TypeObjects`TypeApplication`"]


TypeApplicationQ
CreateTypeApplication


Begin["`Private`"]


Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["TypeFramework`TypeObjects`TypeBase`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`TypeObjects`TypeForAll`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["TypeFramework`"]
Needs["TypeFramework`TypeObjects`TypeConstructor`"]
Needs["TypeFramework`TypeObjects`TypeLiteral`"]


sameQ[self_, other_?TypeApplicationQ] :=
	self["type"]["sameQ", other["type"]] &&
	Length[self["arguments"]] === Length[other["arguments"]] &&
	AllTrue[Transpose[{self["arguments"], other["arguments"]}], #[[1]]["sameQ", #[[2]]]&]

sameQ[arg___] := False


computeFree[self_] :=
	Join @@ Map[
	   #["free"]&,
	   Append[self["arguments"], self["type"]]
	]

variableCount[self_] :=
	Total[ Map[#["variableCount"]&, self["arguments"]]] + self["type"]["variableCount"]


stripType[ Type[arg_]] :=
	arg

stripType[ TypeSpecifier[arg_]] :=
	arg

unresolve[ self_] :=
	Module[ {args, app},
		args = Map[ stripType[#["unresolve"]]&, self["arguments"]];
		app = stripType[self["type"]["unresolve"]];
		TypeSpecifier[ app @@ args]
	]

clone[self_] :=
	clone[self, CreateReference[<||>]];

clone[self_, varmap_] :=
	If[varmap["keyExistsQ", self["id"]],
		varmap["lookup", self["id"]],
		With[{
			ty = CreateTypeApplication[
				self["type"]["clone", varmap],
				#["clone", varmap]& /@ self["arguments"]
			]
		},
			ty["setProperties", self["properties"]["clone"]];
			ty
		]
	];

RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
TypeApplicationClass = DeclareClass[
	TypeApplication,
	<|
        "toScheme" -> (toScheme[Self]&),
        "canonicalize" -> (canonicalize[Self, ##]&),
		"sameQ" -> (sameQ[Self, ##]&),
		"clone" -> (clone[Self, ##]&),
		"computeFree" -> (computeFree[Self]&),
		"variableCount" -> (variableCount[Self]&),
		"unresolve" -> Function[ {}, unresolve[Self]],
		"isNamedApplication" -> (isNamedApplication[Self, ##]&),
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	{
		"arguments",
		"type"
	},
	Predicate -> TypeApplicationQ,
	Extends -> TypeBaseClass
];
RegisterTypeObject[ TypeApplication];
]]


CreateTypeApplication[type_, arg_, opts_:<||>] :=
	CreateTypeApplication[type, {arg}, opts]

CreateTypeApplication[type_, arguments_?ListQ, opts_:<||>] :=
	Module[{expectedKindArguments, actualKindArguments},
		(*
			This checking can be revisited to better handle:
			Unknown kind
			TypeEvaluate
			TypeApplication
			etc.
		*)
		If[isEasilyChecked[type] && (And @@ (isEasilyChecked /@ arguments)),
			expectedKindArguments = type["kind"]["arguments"];
			actualKindArguments = (#["kind"])& /@ arguments;
			If[Length[expectedKindArguments] =!= Length[actualKindArguments],
				ThrowException[{
					"Length of arguments given to CreateTypeApplication does not equal the expected kind", type, arguments}]
			];
			If[!(And @@ (MapThread[#1["sameQ", #2]&, {expectedKindArguments, actualKindArguments}])),
				ThrowException[{
					"Kind of type and arguments are not sameQ", expectedKindArguments, actualKindArguments}]
			];
		];
		CreateObject[TypeApplication,
			<|
			"id" -> GetNextTypeId[],
			"arguments" -> arguments,
			"type" -> type
			|>
		]
	]

CreateTypeApplication[args___] :=
	ThrowException[{"Unrecognized call to CreateTypeApplication", {args}}]


isEasilyChecked[ty_] :=
	TypeConstructorQ[ty] || TypeLiteralQ[ty]






isNamedApplication[self_, name_] :=
	self["type"]["isConstructor", name]



(**************************************************)

toScheme[self_] :=
    CreateTypeForAll[{}, self];
    
    
canonicalize[self_] :=
    canonicalize[self, CreateReference[1]];
canonicalize[self_, idx_] :=
    CreateTypeApplication[
    	self["type"]["canonicalize", idx],
        #["canonicalize", idx]& /@ self["arguments"]
    ];
    
(**************************************************)

icon := Graphics[Text[
	Style["TApp",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[typ_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"TypeApplication",
		typ,
  		icon,
  		Flatten[
		{
			BoxForm`SummaryItem[{Pane["type: ", {90, Automatic}], typ["type"]["toString"]}],
			BoxForm`SummaryItem[{Pane["arguments: ", {90, Automatic}], #["toString"]& /@ typ["arguments"]}]    
  		}],
  		{}, 
  		fmt
  	]


toString[typ_] := StringJoin[
	typ["type"]["toString"],
	"[",
	Riffle[#["toString"]& /@ typ["arguments"], ", "],
	"]"
]

format[ self_, shortQ_:True] :=
	With[{
	    args = Map[ #["format", shortQ]&, self["arguments"]]
	},
		StringJoin[ self["type"]["format", shortQ], "[", Riffle[args, ","], "]"]
	]
   
    
End[]

EndPackage[]

