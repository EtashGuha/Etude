
BeginPackage["TypeFramework`TypeObjects`TypePredicate`"]


TypePredicateQ
CreateTypePredicate


Begin["`Private`"]


Needs["TypeFramework`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["TypeFramework`TypeObjects`TypeBase`"]
Needs["TypeFramework`TypeObjects`TypeVariable`"]
Needs["TypeFramework`TypeObjects`TypeConstructor`"]
Needs["TypeFramework`TypeObjects`TypeApplication`"]
Needs["TypeFramework`TypeObjects`TypeArrow`"]
Needs["TypeFramework`TypeObjects`TypeForAll`"]
Needs["TypeFramework`TypeObjects`TypeLiteral`"]
Needs["TypeFramework`TypeObjects`TypeEvaluate`"]
Needs["TypeFramework`Environments`AbstractTypeEnvironment`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`Utilities`Error`"]




sameQ[self_, other_?TypePredicateQ] :=
    self["test"] === other["test"] &&
    AllTrue[Transpose[{self["types"], other["types"]}], #[[1]]["sameQ", #[[2]]]&]
    
sameQ[___] := False


RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
TypePredicateClass = DeclareClass[
    TypePredicate,
    <|
        "computeFree" -> (computeFree[Self]&),
        "hasAbstractType" -> (hasAbstractType[Self]&),
        "underlyingAbstractType" -> (underlyingAbstractType[Self]&),
        "toScheme" -> (toScheme[Self]&),
        "canonicalize" -> (canonicalize[Self]&),
        "format" -> (format[Self, ##]&),
        "sameQ" -> (sameQ[Self, ##]&),
        "clone" -> (clone[Self, ##]&),
        "isHnf" -> (isHnf[Self]&),
        "entailedBy" -> (entailedBy[Self, ##]&),
        "unresolve" -> Function[ {}, unresolve[Self]],
        "toString" -> Function[{}, toString[Self]],
        "toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
    |>,
    {
        "test",
        "types"
    },
    Predicate -> TypePredicateQ,
    Extends -> TypeBaseClass
];
RegisterTypeObject[TypePredicate];
]]
    
CreateTypePredicate[type_?TypeObjectQ, test_?validTestQ] :=
    CreateObject[TypePredicate,
        <|
            "id" -> GetNextTypeId[],
            "test" -> canonicalizeTest[test],
            "types" -> {type}
        |>
    ]
    
CreateTypePredicate[types_?ListQ, test_?validTestQ] :=
    CreateObject[TypePredicate,
        <|
            "id" -> GetNextTypeId[],
            "test" -> canonicalizeTest[test],
            "types" -> types
        |>
    ]
    
canonicalizeTest[MemberQ[s_String]] := MemberQ[AbstractType[s]];

canonicalizeTest[a_] := a;

validTestQ[ TrueQ] := True

validTestQ[ Unequal] := True

validTestQ[MemberQ[_String]] := True;

validTestQ[MemberQ[AbstractType[_String]]] := True;

validTestQ[__] := False;

stripType[ Type[arg_]] :=
	arg

stripType[ TypeSpecifier[arg_]] :=
	arg

unresolve[ self_] :=
    With[{
        test = self["test"],
        types = stripType[#["unresolve"]]& /@ self["types"]
    },
        TypeSpecifier[ TypePredicate[types, test]]
    ]

clone[self_] :=
	clone[self, CreateReference[<||>]];

clone[self_, varmap_] :=
	If[varmap["keyExistsQ", self["id"]],
		varmap["lookup", self["id"]],
		With[{
			pred = CreateTypePredicate[
				#["clone", varmap]& /@ self["types"],
				self["test"]
			]
		},
			pred["setProperties", self["properties"]["clone"]];
			pred
		]
	];

hasAbstractType[ self_] :=
	MatchQ[ self["test"], _MemberQ]

underlyingAbstractType[self_] := 
	getAbstractType[self, self["test"]];
	
getAbstractType[self_, MemberQ[cls_]] := cls;

getAbstractType[self_, other_] := 
	ThrowException[{"TypePredicate does not have an abstract type", self}]



(**************************************************)

computeFree[self_] :=
    Join @@ Map[#["free"]&, self["types"]]

toScheme[self_] :=
    CreateTypeForAll[{}, self];
    
    
canonicalize[self_] :=
    canonicalize[self, CreateReference[1]];
canonicalize[self_, idx_] :=
    CreateTypePredicate[
        #["canonicalize", idx]& /@ self["types"],
        self["test"]
    ]; 
    
(**********************************************************************)

entailedBy[self_, absEnv_?AbstractTypeEnvironmentQ, preds_] :=
    absEnv["predicatesEntailQ", preds, self];  

(**********************************************************************
 * From: http://stackoverflow.com/a/6889335
 * To determine whether an expression is in weak head normal form, 
 * we only have to look at the outermost part of the expression. 
 * If it's a data constructor or a lambda, it's in weak head normal 
 * form. If it's a function application, it's not.
 *********************************************************************)

isHnf[t_] := AllTrue[t["types"], isHnf0];
isHnf0[t_] :=
    Which[
        TypeVariableQ[t], True,
        TypeConstructorQ[t], False,
        TypeApplicationQ[t], isHnf0[t["type"]],
        TypeArrowQ[t], False,
        TypeForAllQ[t], False,
        TypeLiteralQ[t], False,
        TypeEvaluateQ[t], False,
        True,
            TypeFailure[
	            "IsHNF",
	            "invalid type `1` encountered while checking if type is in head-normal form",
	            t
	        ]
    ];


(**************************************************)

icon := icon = Graphics[Text[
    Style["TYP\nPRED",
          GrayLevel[0.7],
          Bold,
          1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
    ]], $FormatingGraphicsOptions
]
      
toBoxes[typ_, fmt_]  :=
    BoxForm`ArrangeSummaryBox[
        "TypePredicate",
        typ,
        icon,
        {
            BoxForm`SummaryItem[{Pane["test: ", {90, Automatic}], typ["test"]}],
            BoxForm`SummaryItem[{Pane["types: ", {90, Automatic}], #["toString"]& /@ typ["types"]}]
        },
        {}, 
        fmt
    ]


(*
Save the $ContextPath at package load time, and use it when calling ToString
This will allow ToString to stringify symbols as, e.g., "AbstractType" instead of the full-qualified "TypeFramework`AbstractType"
*)
$contextPathAtLoadTime = $ContextPath

toStringAbstract[ typ_] :=
StringJoin[
    "Predicate[",
    Riffle[
        #["toString"]& /@ typ["types"],
        ", "
    ],
    " \[Element] ",
    Block[{$ContextPath = $contextPathAtLoadTime}, ToString[typ["test"]]],
    "]"
]

toStringGeneral[ typ_] :=
StringJoin[
    "Predicate[",
    Block[{$ContextPath = $contextPathAtLoadTime}, ToString[typ["test"]]],
    "[",
    Riffle[
        #["toString"]& /@ typ["types"],
        ", "
    ],
    "]",
    
    "]"
]


toString[typ_] := 
	If[ typ["hasAbstractType"],
		toStringAbstract[typ],
		toStringGeneral[typ]]


format[self_, shortQ_:True] :=
    StringJoin[
        "Pred(",
        Riffle[#["format", shortQ]& /@ self["types"], ", "],
        " \[Element] ",
        Block[{$ContextPath = $contextPathAtLoadTime}, ToString[self["test"]]],
        ")"
    ]
    

End[]

EndPackage[]

