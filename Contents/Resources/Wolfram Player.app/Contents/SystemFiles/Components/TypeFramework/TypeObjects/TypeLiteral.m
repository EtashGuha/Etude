
BeginPackage["TypeFramework`TypeObjects`TypeLiteral`"]

TypeLiteralQ
CreateTypeLiteral
TypeLiteralObject

Begin["`Private`"]

Needs["TypeFramework`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["TypeFramework`TypeObjects`TypeBase`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Reference`"]
Needs["TypeFramework`TypeObjects`TypeForAll`"]
Needs["TypeFramework`TypeObjects`TypeConstructor`"]



sameValue[t1_?TypeObjectQ, t2_?TypeObjectQ] :=
    t1["sameQ", t2];
    
sameValue[t1_, t2_] :=
    !TypeObjectQ[t1] &&
    !TypeObjectQ[t2] &&
    t1 === t2;
    

sameQ[self_, other_?TypeLiteralQ] :=
    self["id"] === other["id"] ||
    (
        sameValue[self["value"], other["value"]] &&
        self["type"]["sameQ", other["type"]]
    )
        
sameQ[___] := 
    Module[{},
        False
    ]

TypeBottomQ[ty_?TypeLiteralQ] := ty["getProperty", "Bottom"];
TypeTopQ[ty_?TypeLiteralQ] := ty["getProperty", "Top"];

format[ self_, shortQ_:True] :=
    ToString[self["value"]] <> ":" <>self["type"]["format", shortQ]

accept[ self_, vst_] :=
    vst["visitLiteral", self]

RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
TypeLiteralClass = DeclareClass[
    TypeLiteralObject,
    <|
        "computeFree" -> (computeFree[Self]&),
        "variableCount" -> (variableCount[Self]&),
        "toScheme" -> (toScheme[Self]&),
        "sameQ" -> (sameQ[Self, ##]&),
        "clone" -> (clone[Self, ##]&),
        "unresolve" -> Function[ {}, unresolve[Self]],
        "accept" -> Function[{vst}, accept[Self, vst]],
        "toString" -> Function[{}, toString[Self]],
        "toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
        "format" -> (format[ Self, ##]&)
    |>,
    {
        "kind",
        "value",
        "type"
    },
    Predicate -> TypeLiteralQ,
    Extends -> TypeBaseClass
];
RegisterTypeObject[ TypeLiteralObject];
]]
    
CreateTypeLiteral[value_, type_?TypeObjectQ, opts_:<||>] :=
    With[{
        tycon = CreateObject[TypeLiteralObject, <|
            "id" -> GetNextTypeId[],
            "value" -> value,
            "type" -> type,
            (* Kind objects do not have "clone" method, otherwise should clone here *)
            "kind" -> type["kind"]
        |>]
    },
        tycon
    ]


stripType[ Type[arg_]] :=
	arg

stripType[ TypeSpecifier[arg_]] :=
	arg

unresolve[ self_] :=
    With[{
        val = self["value"]
    },
	    TypeSpecifier[TypeLiteral[
	        If[TypeObjectQ[val], val["unresolve"], val],
	        stripType[ self["type"]["unresolve"]]
	    ]]
    ]



clone[self_] :=
    clone[self, CreateReference[<||>]];

clone[self_, varmap_] :=
    If[varmap["keyExistsQ", self["id"]],
        varmap["lookup", self["id"]],
	    With[{
	        val = self["value"]
	    },
	    With[{
	        ty = CreateTypeLiteral[
	            If[TypeObjectQ[val], val["clone", varmap], val],
	            self["type"]["clone", varmap]   
	        ]
	    },
	        ty["setProperties", self["properties"]["clone"]];
	        ty
	    ]]
    ];

(**************************************************)


computeFree[self_] := 
	self["type"]["free"];

variableCount[self_] := 
	self["type"]["variableCount"];

toScheme[self_] :=
    CreateTypeForAll[{}, self];
    
(**************************************************)

icon := Graphics[Text[
    Style["TLit",
          GrayLevel[0.7],
          Bold,
          1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
    ]], $FormatingGraphicsOptions
]
      
toBoxes[typ_, fmt_]  :=
    BoxForm`ArrangeSummaryBox[
        "TypeLiteral",
        typ,
        icon,
        {
            BoxForm`SummaryItem[{Pane["value: ", {90, Automatic}], typ["value"]}],
            BoxForm`SummaryItem[{Pane["type: ", {90, Automatic}], typ["type"]["toString"]}]
        },
        {},
        fmt
    ]


toString[typ_] := typ["name"]

End[]

EndPackage[]

