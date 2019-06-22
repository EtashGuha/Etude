
BeginPackage["TypeFramework`TypeObjects`TypeAssumption`"]


TypeAssumptionQ
CreateTypeAssumption


Begin["`Private`"]

Needs["TypeFramework`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["TypeFramework`TypeObjects`TypeBase`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`Utilities`Error`"]
Needs["TypeFramework`TypeObjects`TypeForAll`"]



sameQ[self_, other_?TypeAssumptionQ] :=
    self["name"] === other["name"] &&
    self["type"]["sameQ", other["type"]]
    
sameQ[___] := False

RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
TypeAssumptionClass = DeclareClass[
    TypeAssumption,
    <|
        "underlyingAbstractType" -> (Self["type"]["underlyingAbstractType"]&),
        "computeFree" -> (computeFree[Self]&),
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
        "typename",
        "type"
    },
    Predicate -> TypeAssumptionQ,
    Extends -> TypeBaseClass
];
RegisterTypeObject[ TypeAssumption];
]]


CreateTypeAssumption[name_, type_?TypeQ] :=
    CreateObject[TypeAssumption,
        <|
            "id" -> GetNextTypeId[],
            "typename" -> name,
            "type" -> type
        |>
    ]


stripWrapper[ Type[TypeAssumption[ name_, typ_]]] :=
    TypeAssumption[name, typ]

stripWrapper[ TypeSpecifier[TypeAssumption[ name_, typ_]]] :=
    TypeAssumption[name, typ]


unresolve[ self_] :=
    TypeFailure["unresolve not implemented for TypeAssumption"]

clone[self_] :=
	clone[self, CreateReference[<||>]];

clone[self_, varmap_] :=
	If[varmap["keyExistsQ", self["id"]],
		varmap["lookup", self["id"]],
		With[{
			ty = CreateTypeAssumption[
				self["typename"],
				self["type"]["clone", varmap]
			]
		},
			ty["setProperties", self["properties"]["clone"]];
			ty
		]
	];


(**************************************************)

computeFree[self_] :=
    self["type"]["free"]
    
toScheme[self_] :=
    CreateTypeForAll[{}, self];

canonicalize[self_] :=
    canonicalize[self, CreateReference[1]];
canonicalize[self_, idx_] :=
    CreateTypeAssumption[
        self["typename"],
        self["type"]["canonicalize", idx]
    ];
    
(**************************************************)

icon := icon = Graphics[Text[
    Style["TYP\nASS",
          GrayLevel[0.7],
          Bold,
          1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
    ]], $FormatingGraphicsOptions
]
      
toBoxes[typ_, fmt_]  :=
    BoxForm`ArrangeSummaryBox[
        "TypeAssumption",
        typ,
        icon,
        {
            BoxForm`MakeSummaryItem[{Pane["name: ", {90, Automatic}], typ["name"]}, fmt],
            BoxForm`MakeSummaryItem[{Pane["type: ", {90, Automatic}], typ["type"]}, fmt]
        },
        {}, 
        fmt
    ]


toString[typ_] := StringJoin[
    "Assumption(",
    typ["typename"],
    " === ",
    typ["type"]["toString"],
    ")"
]


format[self_, shortQ_:True] :=
    StringJoin[
        "Assumption(",
	    self["typename"],
	    " === ",
	    self["type"]["format", shortQ],
	    ")"
    ]
    

End[]

EndPackage[]

