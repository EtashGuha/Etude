
BeginPackage["TypeFramework`TypeObjects`TypeArrow`"]


TypeArrowQ
CreateTypeArrow
TypeArrow


Begin["`Private`"]

Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["TypeFramework`TypeObjects`TypeBase`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`TypeObjects`TypeForAll`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)


sameQ[self_, other_?TypeArrowQ] :=
	Length[self["arguments"]] === Length[other["arguments"]] &&
	AllTrue[Transpose[{self["arguments"], other["arguments"]}], #[[1]]["sameQ", #[[2]]]&] &&
	self["result"]["sameQ", other["result"]]


sameQ[arg___] := False


computeFree[self_] :=
	Join @@ Map[
	   #["free"]&,
	   Append[self["arguments"], self["result"]]
	]
    
variableCount[self_] :=
	Total[ Map[#["variableCount"]&, self["arguments"]]] + self["result"]["variableCount"]

stripType[ Type[arg_]] :=
	arg
	
stripType[ TypeSpecifier[arg_]] :=
	arg

unresolve[ self_] :=
	Module[ {args, res},
		args = Map[ stripType[#["unresolve"]]&, self["arguments"]];
		res = stripType[self["result"]["unresolve"]];
		TypeSpecifier[ args -> res]
	]

clone[self_] :=
	clone[self, CreateReference[<||>]];

clone[self_, varmap_] :=
	If[varmap["keyExistsQ", self["id"]],
		varmap["lookup", self["id"]],
		With[{
			ty = CreateTypeArrow[
				#["clone", varmap]& /@ self["arguments"],
				self["result"]["clone", varmap]
			]
		},
			ty["setProperties", self["properties"]["clone"]];
			ty
		]
	];

RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
TypeArrowClass = DeclareClass[
	TypeArrow,
	<|
        "toScheme" -> (toScheme[Self]&),
        "canonicalize" -> (canonicalize[Self, ##]&),
		"sameQ" -> (sameQ[Self, ##]&),
		"clone" -> (clone[Self, ##]&),
		"computeFree" -> (computeFree[Self]&),
		"variableCount" -> (variableCount[Self]&),
		"unresolve" -> Function[ {}, unresolve[Self]],
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"format" -> (format[ Self, ##]&)
	|>,
	{
		"arguments",
		"result"
	},
	Predicate -> TypeArrowQ,
	Extends -> TypeBaseClass
];
RegisterTypeObject[ TypeArrow];
]]

(* Todo: need to check the kind *)
CreateTypeArrow[arg_, result_,opts_:<||>] :=
	CreateTypeArrow[{arg}, result, opts]

(*
 The Flatten for the arguments is needed in case we have a bound sequence.
*)
CreateTypeArrow[arguments_?ListQ, result_, opts_:<||>] :=
		CreateObject[TypeArrow,
			<|
			"id" -> GetNextTypeId[],
			"arguments" -> arguments,
			"result" -> result
			|>
		]

CreateTypeArrow[args___] :=
	ThrowException[{"Unrecognized call to CreateTypeArrow", {args}}]


(**************************************************)

toScheme[self_] :=
    CreateTypeForAll[{}, self];
    
    
canonicalize[self_] :=
    canonicalize[self, CreateReference[1]];
canonicalize[self_, idx_] :=
    CreateTypeArrow[
        #["canonicalize", idx]& /@ self["arguments"],
        self["result"]["canonicalize", idx]
    ];
    
(**************************************************)

icon := Graphics[Text[
	Style["TArr",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[typ_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"TypeArrow",
		typ,
  		icon,
  		Flatten[
			{
				BoxForm`SummaryItem[{Pane["arguments: ", {90, Automatic}], #["toString"]& /@ typ["arguments"]}],
	  		    BoxForm`SummaryItem[{Pane["result: ", {90, Automatic}], typ["result"]["toString"]}]
	  		}
  		],
  		{
            BoxForm`SummaryItem[{Pane["arguments: ", {90, Automatic}], typ["arguments"]}],
            BoxForm`SummaryItem[{Pane["result: ", {90, Automatic}], typ["result"]}]
        }, 
  		fmt
  	]


toString[typ_] := StringJoin[
	"{",
	Riffle[#["toString"]& /@ typ["arguments"], ", "],
	"}",
	"\[Rule]",
	typ["result"]["toString"]
]

format[ self_, shortQ_:True] :=
	With[{args = Map[ #["format", shortQ]&, self["arguments"]]},
		StringJoin[ "(", Riffle[args, ","], ")", "\[Rule]", self["result"]["format", shortQ]]
	]


End[]

EndPackage[]

