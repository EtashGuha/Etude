Package["Databases`Database`"]


PackageImport["Databases`"]
PackageImport["Databases`Common`"]
PackageImport["Databases`Schema`"] (* Types *)


PackageScope["DBRegisterFunctionSignatures"]
PackageScope["DBGetTypeSignatures"]
PackageScope["DBSortFunctionSignatures"]
PackageScope["DBAggregatingOperatorQ"]

$DBFunctionSignatures = <||>


SetAttributes[DBRegisterFunctionSignatures, HoldFirst]

DBRegisterFunctionSignatures[
	{fun_?StringQ, argumentType_DBType?DBTypeQ} -> {returnFun_, returnType_}
] := (
	If[!KeyExistsQ[$DBFunctionSignatures, fun],
		$DBFunctionSignatures[fun] = {}
	];
	$DBFunctionSignatures[fun] = Join[
		$DBFunctionSignatures[fun],
		{argumentType -> {returnFun, Replace[returnType, f: Except[_Function] :> Function[f]]}}
	];
)

DBRegisterFunctionSignatures[
	{fun_, argumentType_} :> {returnFun_, returnType_}
] := With[
	{rules = Cases[
		{fun, argumentType},
		Verbatim[Pattern][symbol_Symbol, Verbatim[Alternatives][args___]] :>
      		Thread[symbol -> {args}],
		{0, Infinity}
	]},
	Map[
		DBRegisterFunctionSignatures,
		Rule @@@ Map[
			{
				ReplaceAll[
					{fun, argumentType},
					Function[
						{sym, val},
						Verbatim[Pattern][sym, _] -> val
					] @@@ #
				],
				ReplaceAll[
					{returnFun, returnType},
					#
				]
			}&,
			Tuples[rules]
		]
	]
]

DBRegisterFunctionSignatures[{fun: Except[_?StringQ], _} -> _] := DBRaise[
	DBRegisterFunctionSignatures,
	"operator_not_string",
	{fun}
]

DBRegisterFunctionSignatures[{_, argumentType: Except[_DBType?DBTypeQ]} -> _] := DBRaise[
	DBRegisterFunctionSignatures,
	"argument_not_DBTypeQ",
	{argumentType}
]

DBRegisterFunctionSignatures[(Rule|RuleDelayed)[_, ret: Except[{_, _}]]] := DBRaise[
	DBRegisterFunctionSignatures,
	"return_not_2ple",
	{ret}
]

DBRegisterFunctionSignatures[(Rule|RuleDelayed)[_, {_, t: Except[_DBType?DBTypeQ]}]] := DBRaise[
	DBRegisterFunctionSignatures,
	"return_not_type",
	{t}
]


DBDefError @ DBRegisterFunctionSignatures

DBGetTypeSignatures[] :=
	$DBFunctionSignatures

DBGetTypeSignatures[operator_?StringQ] :=
	Lookup[$DBFunctionSignatures, operator, {}]

DBGetTypeSignatures[operator_?StringQ, type_] :=
	Apply[
		Function[
			{in, out},
			{out[[1]], out[[2]][type], in}
		],
		SelectFirst[
			DBGetTypeSignatures[operator],
			DBTypeContainsQ[#[[1]], type]&,
			Automatic -> {None, Function[DBType[DBTypeUnion[]]]}
		]
	]


DBSortFunctionSignatures[] :=
	$DBFunctionSignatures = Map[
		Sort[#, ! DBTypeContainsQ[First[#1], First[#2]]&]&,
		$DBFunctionSignatures
	]

DBAggregatingOperatorQ[operator_?StringQ] := AllTrue[
	DeleteCases[
        DBGetTypeSignatures[operator], 
        DBType[CompoundElement[{DBType["Query"]}]] -> _
    ],
	And[
		MatchQ[First[#], DBType[CompoundElement[{DBType[_RepeatingElement, ___]}], ___]],
		FreeQ[#[[-1, -1]][First[#]], RepeatingElement]
	]&
]