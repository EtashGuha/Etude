
BeginPackage["TypeFramework`TypeObjects`TypeForAll`"]


TypeForAllQ
CreateTypeForAll


Begin["`Private`"]


Needs["TypeFramework`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["TypeFramework`TypeObjects`TypeBase`"]
Needs["TypeFramework`TypeObjects`TypeVariable`"]
Needs["TypeFramework`TypeObjects`TypeQualified`"]
Needs["TypeFramework`Inference`Substitution`"]
Needs["TypeFramework`Utilities`TypeInstantiate`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Error`Exceptions`"]


(* Todo: this is not correct, since 
 * we want the ForAll to be sameQ upto
 * alpha equivalence
 *)
sameQ[self_, other_?TypeForAllQ] :=
	Length[self["variables"]] === Length[other["variables"]] &&
	AllTrue[Transpose[{self["variables"], other["variables"]}], #[[1]]["sameQ", #[[2]]]&] &&
	self["type"]["sameQ", other["type"]]
	
sameQ[___] := False




RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
TypeForAllClass = DeclareClass[
	TypeForAll,
	<|
	    "computeFree" -> (computeFree[Self]&),
        "instantiate" -> (instantiate[Self, ##]&),
        "toScheme" -> (toScheme[Self]&),
        "canonicalize" -> (canonicalize[Self, ##]&),
        "format" -> (format[Self, ##]&),
		"sameQ" -> (sameQ[Self, ##]&),
		"clone" -> (clone[Self, ##]&),
		"unresolve" -> Function[ {}, unresolve[Self]],
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
	|>,
	{
		"kind",
		"variables",
		"type"
	},
	Predicate -> TypeForAllQ,
	Extends -> TypeBaseClass
];
RegisterTypeObject[TypeForAll];
]]


CreateTypeForAll[variable_, type_?TypeQualifiedQ] :=
	CreateTypeForAll[{variable}, type]

CreateTypeForAll[variable_, type_?TypeObjectQ] :=
	CreateTypeForAll[{variable}, type]

CreateTypeForAll[variables_List, type_?TypeQualifiedQ] :=
    CreateObject[TypeForAll,
        <|
            "id" -> GetNextTypeId[],
            "variables" -> variables,
            "type" -> type
        |>
    ]

CreateTypeForAll[variables_List, type_?TypeObjectQ] :=
	CreateObject[TypeForAll,
		<|
			"id" -> GetNextTypeId[],
			"variables" -> variables,
			"type" -> CreateTypeQualified[{}, type]
		|>
	]

CreateTypeForAll[args___] :=
	ThrowException[{"Unrecognized call to CreateTypeForAll", {args}}]


prepType[ Type[arg_]] :=
	prepType[ arg]
	
prepType[ TypeSpecifier[arg_]] :=
	prepType[ arg]
	
prepType[ args_List -> res_] :=
	Map[ prepType, args] -> prepType[res]
	
prepType[ args_List -> res_] :=
	Map[ prepType, args] -> prepType[res]
	
prepType[ TypeQualified[preds_, ty_]] :=
	TypeQualified[ Map[ prepType, preds], prepType[ty]]
	
prepType[ TypePredicate[types_, tests_]] :=
	TypePredicate[ Map[ prepType, types], tests]
	
prepType[ TypeVariable[ name_]] :=
	name

prepType[ other_] :=
	other


unresolve[ self_] :=
	Module[ {vars, type},
		vars = Map[ prepType[#["unresolve"]]&, self["variables"]];
		type = prepType[self["type"]["unresolve"]];
		TypeSpecifier[ TypeForAll[ vars, type]]
	]

clone[self_] :=
	clone[self, CreateReference[<||>]];

clone[self_, varmap_] :=
	If[varmap["keyExistsQ", self["id"]],
		varmap["lookup", self["id"]],
		With[{
			ty = CreateTypeForAll[
				#["clone", varmap]& /@ self["variables"],
				self["type"]["clone", varmap]
			]
		},
			ty["setProperties", self["properties"]["clone"]];
			ty
		]
	];


(**************************************************)

computeFree[self_] :=
    With[{
        boundVarIds = self["variables"],
        freeVars = self["type"]["free"]
    },

    KeySelect[freeVars, NoneTrue[boundVarIds, Function[{b}, b["id"] === #]]&]
]

typeSameQ[t1_, t2_] := t1["sameQ", t2];

instantiate[self_, opts:OptionsPattern[TypeInstantiate]] :=
    TypeInstantiate[self, opts];
    
    
toScheme[self_] :=
    self;
    
    
canonicalize[self_] :=
    canonicalize[self, CreateReference[1]];
canonicalize[self_, idx_] :=
    With[{
        newVars = Table[
            CreateTypeVariable["v" <> ToString[idx["increment"]]],
            {var, self["variables"]}
        ]
    },
    With[{
        sub = CreateTypeSubstitution[
            AssociationThread[self["variables"] -> newVars],
            "TypeEnvironment" -> Undefined
        ]
    },
    With[{
        body = sub["apply", self["type"]]
    },
        CreateTypeForAll[
            newVars,
            body["canonicalize", idx]
        ]
    ]]]; 


(**************************************************)

icon := Graphics[Text[
	Style["\[ForAll]Typ",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[typ_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"TypeForAll",
		typ,
  		icon,
		{
			BoxForm`SummaryItem[{Pane["variables: ", {90, Automatic}], #["toString"]& /@ typ["variables"]}],
  		    BoxForm`SummaryItem[{Pane["type: ", {90, Automatic}], typ["type"]["toString"]}]
  		},
  		{}, 
  		fmt
  	]


toString[typ_] := StringJoin[
	"\[ForAll]",
	"(",
	Riffle[#["toString"]& /@ typ["variables"], ", "],
	")",
	" : ",
	typ["type"]["toString"]
]


format[self_, shortQ_:True] :=
	With[{
	   args = Map[ #["format", shortQ]&, self["variables"]]
	},
	   StringJoin[ "\[ForAll] ", Riffle[args, ","], " : ", self["type"]["format", shortQ]]
	]
	

End[]

EndPackage[]

