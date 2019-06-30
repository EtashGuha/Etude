
BeginPackage["TypeFramework`TypeObjects`Kind`"]

Kind;
KindQ;
KindClass;

RowKind;
RowKindQ;
RowKindClass;
CreateRowKind;

UnknownKind;
UnknownKindQ;
UnknownKindClass;
CreateUnknownKind;

NullaryKind;
NullaryKindQ;
NullaryKindClass;
CreateNullaryKind;

LabelKind;
LabelKindQ;
LabelKindClass;
CreateLabelKind;

FunctionKind;
FunctionKindQ;
FunctionKindClass;
CreateFunctionKind;


SkolemKind;
SkolemKindQ;
SkolemKindClass;
CreateSkolemKind;

Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Asserter`Assert`"]
Needs["CompileUtilities`Callback`"]

RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
KindClass = DeclareClass[
	Kind,
	<|
		"isFunction" -> Function[{}, False],
		"kindList" -> Function[{}, ThrowException[{"not implemented"}]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
	|>,
	{},
(*
 Set a predicate with an internal name that will not be called.
 This class is not designed to be instantiated.
*)
	Predicate -> nullKindQ
]
]]

(*
  Add functionality for KindQ
*)
$kindBaseObjects = <||>

RegisterKind[ name_] :=
	AssociateTo[$kindBaseObjects, name -> True]
	
KindQ[ obj_] :=
	ObjectInstanceQ[obj] && KeyExistsQ[$kindBaseObjects, obj["_class"]]



sameNullKindQ[self_, other_?NullaryKindQ] := True
sameNullKindQ[___] := False

RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
NullaryKindClass = DeclareClass[
	NullaryKind,
	<|
		"sameQ" -> (sameNullKindQ[Self, ##]&),
		"toString" -> Function[{}, "*"],
		"kindList" -> Function[{}, {}]
	|>,
	{},
	Predicate -> NullaryKindQ,
	Extends -> KindClass
];
RegisterKind[ NullaryKind];
]]

CreateNullaryKind[] := CreateObject[NullaryKind]

sameLabelKindQ[self_, other_?LabelKindQ] := self["name"] === other["name"]
sameLabelKindQ[___] := False

RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
LabelKindClass = DeclareClass[
	LabelKind,
	<|
		"sameQ" -> (sameLabelKindQ[Self, ##]&),
		"toString" -> Function[{}, If[StringQ[Self["name"]], Self["name"], "lbl" <>Self["id"]]],
		"kindList" -> Function[{}, {}]
	|>,
	{
		"id",
		"name"
	},
	Predicate -> LabelKindQ,
	Extends -> KindClass
];
RegisterKind[ LabelKind];
]]


CreateLabelKind[] := CreateObject[LabelKind, <| "id" -> SymbolName[Unique[]] |>]
CreateLabelKind[name_?StringQ] := CreateObject[LabelKind, <| "name" -> name, "id" -> SymbolName[Unique[]] |>]

sameRowKindQ[self_, other_?RowKindQ] := True
sameRowKindQ[___] := False

RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
RowKindClass = DeclareClass[
	RowKind,
	<|
		"sameQ" -> (sameRowKindQ[Self, ##]&),
		"toString" -> Function[{}, "R"],
		"kindList" -> Function[{}, {}]
	|>,
	{},
	Predicate -> RowKindQ,
	Extends -> KindClass
];
RegisterKind[ RowKind];
]]

CreateRowKind[] := CreateObject[RowKind]


sameUnknownKindQ[self_, other_?UnknownKindQ] := True
sameUnknownKindQ[___] := False

RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
UnknownKindClass = DeclareClass[
	UnknownKind,
	<|
		"sameQ" -> (sameUnknownKindQ[Self, ##]&),
		"toString" -> Function[{}, "U"],
		"kindList" -> Function[{}, {}]
	|>,
	{},
	Predicate -> UnknownKindQ,
	Extends -> KindClass
];
RegisterKind[ UnknownKind];
]]

CreateUnknownKind[] := CreateObject[UnknownKind]

sameFunctionKindQ[self_, other_?FunctionKindQ] :=
	Length[self["kindList"]] === Length[other["kindList"]] &&
	AllTrue[Transpose[{self["kindList"], other["kindList"]}], #[[1]]["sameQ", #[[2]]]&]
sameFunctionKindQ[___] := False

RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
FunctionKindClass = DeclareClass[
	FunctionKind,
	<|
		"isFunction" -> Function[{}, True],
		"sameQ" -> (sameFunctionKindQ[Self, ##]&),
		"toString" -> (functionKindToString[Self]&),
		"kindList" -> (kindList[Self]&)
	|>,
	{
		"arguments" -> {},
		"result"
	},
	Predicate -> FunctionKindQ,
	Extends -> KindClass
];
RegisterKind[ FunctionKind];
]]

CreateFunctionKind[args___, ret_] :=
	CreateFunctionKind[{args}, ret]
CreateFunctionKind[args_List, ret_] := Module[{},
	AssertThat["All the inputs are kinds", Flatten[{args, ret}]
		]["named", "kind list"
		]["elementsSatisfy", KindQ
	];
	CreateObject[FunctionKind, <| "arguments" -> args, "result" -> ret |>]
]

RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
SkolemKindClass = DeclareClass[
    SkolemKind,
    <|
        "sameQ" -> (Self["kind"]["sameQ", ##]&),
        "toString" -> ("skol(" <> Self["kind"]["toString"] <> ")"&),
        "kindList" -> (kindList[Self["kind"]]&)
    |>,
    {
        "kind"
    },
    Predicate -> SkolemKindQ,
    Extends -> KindClass
];
RegisterKind[ SkolemKind];
]]

CreateSkolemKind[] :=
    CreateSkolemKind[CreateNullaryKind[]]

CreateSkolemKind[kind_] :=
    CreateObject[SkolemKind, <| "kind" -> kind |>]

(**************************************************)
(**************************************************)
(**************************************************)

makeIcon[kind_] := 
	Graphics[Text[
		Style[kind["toString"],
			  GrayLevel[0.7],
			  Bold,
			  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
		]], $FormatingGraphicsOptions
	]
  	
toBoxes[kind_?FunctionKindQ, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		kind["_class"],
		kind,
  		makeIcon[kind],
		{
			BoxForm`SummaryItem[{Pane["arguments: ", {90, Automatic}], #["toString"]& /@ kind["arguments"]}],
			BoxForm`SummaryItem[{Pane["result: ", {90, Automatic}], kind["result"]["toString"]}],
			BoxForm`SummaryItem[{Pane["kindList: ", {90, Automatic}], #["toString"]& /@ kind["kindList"]}]
  		},
  		{}, 
  		fmt
  	]
  	
toBoxes[kind_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		kind["_class"],
		kind,
  		makeIcon[kind],
		{
			BoxForm`SummaryItem[{"kind: ", kind["toString"]}]
  		},
  		{}, 
  		fmt
  	]
kindList[self_] := Flatten[
	With[{kl = #["kindList"]},
		If[kl === {}, #, kl]
	]& /@ Join[self["arguments"], {self["result"]}]
]
functionKindToString[self_] :=
	StringJoin[
		"(", 
		Riffle[#["toString"]& /@ self["arguments"], "\[Rule]"],
		") \[Rule] ",
		self["result"]["toString"]
	]

End[]

EndPackage[]

