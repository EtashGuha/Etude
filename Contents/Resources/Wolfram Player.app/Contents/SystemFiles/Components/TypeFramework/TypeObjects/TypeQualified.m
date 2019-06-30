
BeginPackage["TypeFramework`TypeObjects`TypeQualified`"]


TypeQualifiedQ
CreateTypeQualified


Begin["`Private`"]


Needs["TypeFramework`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["TypeFramework`TypeObjects`TypeBase`"]
Needs["TypeFramework`TypeObjects`TypePredicate`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`TypeObjects`TypeForAll`"]



(* Todo: this is not correct, since 
 * we want the ForAll to be sameQ upto
 * alpha equivalence
 *)
sameQ[self_, other_?TypeQualifiedQ] :=
    AllTrue[Transpose[{self["predicates"], other["predicates"]}], #[[1]]["sameQ", #[[2]]]&] &&
    self["type"]["sameQ", other["type"]]
    
sameQ[___] := False




RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
TypeQualifiedClass = DeclareClass[
    TypeQualified,
    <|
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
        "predicates",
        "type"
    },
    Predicate -> TypeQualifiedQ,
    Extends -> TypeBaseClass
];
RegisterTypeObject[TypeQualified];
]]

CreateTypeQualified[type_] :=
    CreateTypeQualified[{}, type]
    
CreateTypeQualified[pred_?TypePredicateQ, type_] :=
    CreateTypeQualified[{pred}, type];
CreateTypeQualified[preds:{___?TypePredicateQ}, type_] :=
    CreateObject[TypeQualified,
        <|
            "id" -> GetNextTypeId[],
            "predicates" -> preds,
            "type" -> type
        |>
    ]

CreateTypeQualified[args___] :=
    ThrowException[{"Unknown arguments to CreateTypeQualified", {args}}]


stripType[ Type[arg_]] :=
	arg
    
stripType[ TypeSpecifier[arg_]] :=
	arg
    
unresolve[ self_] :=
    Module[ {preds, type},
        preds = Map[ stripType[#["unresolve"]]&, self["predicates"]];
        type = stripType[self["type"]["unresolve"]];
        TypeSpecifier[ TypeQualified[ preds, type]]
    ]

clone[self_] :=
	clone[self, CreateReference[<||>]];

clone[self_, varmap_] :=
	If[varmap["keyExistsQ", self["id"]],
		varmap["lookup", self["id"]],
		With[{
			ty = CreateTypeQualified[
				#["clone", varmap]& /@ self["predicates"],
				self["type"]["clone", varmap]
			]
		},
			ty["setProperties", self["properties"]["clone"]];
			ty
		]
	];

(**************************************************)

computeFree[self_] :=
    Join @@ Map[
        #["free"]&,
        Append[self["predicates"], self["type"]]
    ]

toScheme[self_] :=
    CreateTypeForAll[{}, self];
    
canonicalize[self_] :=
    canonicalize[self, CreateReference[1]];
canonicalize[self_, idx_] :=
    CreateTypeQualified[
        #["canonicalize", idx]& /@ self["predicates"],
        self["type"]["canonicalize", idx]
    ];

(**************************************************)

icon := icon = Graphics[Text[
    Style["Typ\nQual",
          GrayLevel[0.7],
          Bold,
          1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
    ]], $FormatingGraphicsOptions
]
      
toBoxes[typ_, fmt_]  :=
    BoxForm`ArrangeSummaryBox[
        "TypeQualified",
        typ,
        icon,
        {
            BoxForm`SummaryItem[{Pane["predicates: ", {90, Automatic}], #["toString"]& /@ typ["predicates"]}],
            BoxForm`SummaryItem[{Pane["type: ", {90, Automatic}], typ["type"]["toString"]}]
        },
        {}, 
        fmt
    ]


toString[typ_] := StringJoin[
    "(",
    Riffle[#["toString"]& /@ typ["predicates"], ", "],
    ")",
    "\[Implies]",
    typ["type"]["toString"]
]


format[self_, shortQ_:True] :=
    With[{
        preds = Map[ #["format", shortQ]&, self["predicates"]]
    },
        StringJoin[
            "(",
            Riffle[preds, ","],
            ") \[Implies]",
            self["type"]["format", shortQ]
        ]
    ]
    

End[]

EndPackage[]

