
BeginPackage["TypeFramework`TypeObjects`TypeProjection`"]

TypeProjectionQ
CreateTypeProjection
TypeProjectionObject

Begin["`Private`"]

Needs["TypeFramework`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["TypeFramework`TypeObjects`TypeBase`"]
Needs["TypeFramework`TypeObjects`TypeApplication`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Error`Exceptions`"]
Needs["CompileUtilities`Reference`"]
Needs["TypeFramework`TypeObjects`TypeForAll`"]
Needs["TypeFramework`TypeObjects`TypeLiteral`"]

  
sameValue[t1_?TypeObjectQ, t2_?TypeObjectQ] :=
    t1["sameQ", t2];
    
sameValue[t1_, t2_] :=
    !TypeObjectQ[t1] &&
    !TypeObjectQ[t2] &&
    t1 === t2;
    

sameQ[self_, other_?TypeProjectionQ] :=
    self["id"] === other["id"] ||
    (
        sameValue[self["value"], other["value"]] &&
        self["type"]["sameQ", other["type"]]
    )
        
sameQ[___] := 
    Module[{},
        False
    ]

format[ self_, shortQ_:True] :=
    "TypeProjection[" <> self["type"]["format", shortQ] <> ", " <> ToString[self["value"]] <> "]"

accept[ self_, vst_] :=
    vst["visitProjection", self]


RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
TypeProjectionClass = DeclareClass[
    TypeProjectionObject,
    <|
        "computeFree" -> (computeFree[Self]&),
        "variableCount" -> (variableCount[Self]&),
        "toScheme" -> (toScheme[Self]&),
        "sameQ" -> (sameQ[Self, ##]&),
        "clone" -> (clone[Self, ##]&),
        "project" -> (project[Self]&),
       	"unresolve" -> Function[ {}, unresolve[Self]],
        "accept" -> Function[{vst}, accept[Self, vst]],
        "toString" -> Function[{}, toString[Self]],
        "toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
        "format" -> (format[ Self, ##]&)
    |>,
    {
        "value",
        "type"
    },
    Predicate -> TypeProjectionQ,
    Extends -> TypeBaseClass
];
RegisterTypeObject[TypeProjectionObject];
]]
    
CreateTypeProjection[type_?TypeObjectQ, value_, opts_:<||>] :=
    With[{
        tycon = CreateObject[TypeProjectionObject, <|
            "id" -> GetNextTypeId[],
            "value" -> value,
            "type" -> type
        |>]
    },
        tycon
    ]



project[ self_] :=
	project[self, self["type"]];
	
project[ self_, ty_?TypeApplicationQ] :=
	Module[ {val = self["value"], args = ty["arguments"]},
		Assert[TypeLiteralQ[val]];
		If[ !TypeLiteralQ[val],
			ThrowException[{"Invalid value in projection (only integral literal values allowed).", val}]
		];
		val = val["value"];
		Assert[val >= 0 && val < Length[args]];
		If[ val < 0 || val >= Length[args],
			ThrowException[{"Projection into type cannot be made.", ty, val}]
		];
		args[[val+1]]
	]
	
	
		
project[ self_, args___] :=
	ThrowException[{"Projection into a type that is not a type application cannot be made.", {args}}]


stripType[ Type[arg_]] :=
	arg
	
stripType[ TypeSpecifier[arg_]] :=
	arg
	
stripType[ arg_] :=
	arg

unresolve[ self_] :=
    With[{
        val = stripType[self["value"]["unresolve"]]
    },
	    TypeSpecifier[TypeProjection[
	    		stripType[ self["type"]["unresolve"]],
	    		val]]
    ]



clone[self_] :=
    clone[self, CreateReference[<||>]];

clone[self_, varmap_] :=
    If[varmap["keyExistsQ", self["id"]],
        varmap["lookup", self["id"]],
	    With[{
	        val = self["value"]["clone"]
	    },
	    With[{
	        ty = CreateTypeProjection[ self["type"]["clone", varmap], val]
	    },
	        ty["setProperties", self["properties"]["clone"]];
	        ty
	    ]]
    ];

(**************************************************)


computeFree[self_] := 
	Join[
		If[TypeObjectQ[self["value"]],
			self["value"]["free"],
			<||>
		],
		self["type"]["free"]
	]

variableCount[self_] := 
	self["type"]["variableCount"];

toScheme[self_] :=
    CreateTypeForAll[{}, self];
    
(**************************************************)

icon := Graphics[Text[
    Style["TProj",
          GrayLevel[0.7],
          Bold,
          1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
    ]], $FormatingGraphicsOptions
]
      
toBoxes[typ_, fmt_]  :=
    BoxForm`ArrangeSummaryBox[
        "TypeProjection",
        typ,
        icon,
        {
            BoxForm`SummaryItem[{Pane["type: ", {90, Automatic}], typ["type"]["toString"]}],
            BoxForm`SummaryItem[{Pane["value: ", {90, Automatic}], typ["value"]}]
        },
        {},
        fmt
    ]


toString[typ_] := StringJoin[
	"TypeProjection[",
	typ["type"]["toString"],
	", ",
	typ["value"]["toString"],
	"]"
]

End[]

EndPackage[]

