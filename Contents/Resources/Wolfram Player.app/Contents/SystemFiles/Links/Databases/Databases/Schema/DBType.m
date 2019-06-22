Package["Databases`Schema`"]


PackageImport["Databases`"]
PackageImport["Databases`Common`"]

PackageExport["DBType"]
PackageExport["DBTyped"]
PackageExport["DBTypeQ"]
PackageExport["DBTypeUnion"]
PackageExport["DBTypeIntersection"]
PackageExport["DBDeduceExpressionType"]


createDeserializer[type_, rule:_Rule|_RuleDelayed] := createDeserializer[type, {rule}]
createDeserializer[type_, rules_List] := 
    Replace[
        Join[
            {None :> Missing["NotAvailable"], m_Missing :> m}, 
            rules
        ]
    ]

defaultDeserializer["SQLite"|"MySQL"|"Oracle"|"MicrosoftSQL", dbt : DBType["Boolean", ___]] =
    createDeserializer[
        dbt, {
            0|False -> False, 
            _ -> True
        }
    ]

defaultDeserializer["SQLite", dbt : DBType["Date", ___]] = 
    createDeserializer[
        dbt,
    	num_?NumericQ :> DateObject[FromUnixTime[num, TimeZone -> "UTC"], "Day"]
    ]

defaultDeserializer["SQLite", dbt: DBType["DateTime", ___]] /; Not[dbt["TimeZone"]] := 
    createDeserializer[
        dbt, 
        num_?NumericQ :> ReplacePart[FromUnixTime[num, TimeZone -> "UTC"], -1 -> None]
    ]
    
defaultDeserializer["SQLite", dbt : DBType["DateTime", ___]] =
    createDeserializer[
        dbt, 
        num_?NumericQ :> FromUnixTime[num]
    ]

defaultDeserializer["SQLite"|"Oracle"|"MySQL"|"MicrosoftSQL", dbt : DBType["Time", ___]] = 
    createDeserializer[
        dbt, 
        num_Real|num_Integer|Quantity[num_, "Seconds"] :> DateValue[
            FromUnixTime[num, TimeZone -> "UTC"], 
            "Second", 
            TimeObject
        ]
    ]
    
defaultDeserializer["MySQL", dbt : DBType["String", ___]] =
    createDeserializer[
        dbt, {
            bytes_ByteArray :> ByteArrayToString[bytes],
            {} -> ""
        }
    ]
    
defaultDeserializer["MySQL"|"Oracle", dbt: DBType["DateTime", ___]] /; dbt["TimeZone"] := 
    createDeserializer[
        dbt, 
        DateObject[start___, None] :> DateObject[start, "UTC"]
    ]

(* those mssql serializers are currently not used with cpython, they were relevant with jython*)
defaultDeserializer["Oracle", dbt : DBType["Date", ___]] =
    createDeserializer[
        dbt, 
        s_DateObject :> DateObject[s, "Day"]
    ]

defaultDeserializer["MicrosoftSQL", dbt : DBType["Date", ___]] =
    createDeserializer[
        dbt, 
        s_String :> DateObject @ Map[FromDigits, StringSplit[s, "-"]]
    ]

defaultDeserializer["MicrosoftSQL", dbt : DBType["Time", ___]] =
	createDeserializer[
        dbt, 
        s_String :> TimeObject @ Map[Internal`StringToDouble, StringSplit[s, ":"]]
    ]

defaultDeserializer["SQLite"|"MySQL"|"MicrosoftSQL", dbt : DBType["TimeQuantity", ___]] =
    createDeserializer[
        dbt, 
        num_ :> Quantity[num, "Seconds"]
    ]

defaultDeserializer[_, _] := Identity


$registry = <|
    "String"    -> <|
        "Test"     -> StringQ, 
        "Examples" -> {"foo", "bar"}
    |>,
    "Real"      -> <|
        "Test"     -> Developer`MachineRealQ, 
        "Examples" -> {1.23, 10.25}
    |>,
    "Boolean"   -> <|
        "Test"     -> BooleanQ, 
        "Examples" -> {True, False}
    |>,
    "Choices" -> <|
        "Test" -> StringQ,
        "Examples" -> {"small", "large"}
    |>,
    "Integer"   -> <|
        "Test"     -> Developer`MachineIntegerQ, 
        "Examples" -> {1, 2}
    |>,
    "Decimal"   -> <|
        "Test"     -> MatchQ[_Real], 
        "Examples" -> {1.23``4, 100``4}
    |>,
    "Date"  -> <|
        "Test"     :> Function[d, DateObjectQ[d] && d["Granularity"] === "Day"],
        "Examples" :> {
            DateObject[{2018, 6, 21}, "Day", "Gregorian", 2.],
            DateObject[{2016, 6, 21}, "Day", "Gregorian", 2.]
        }
    |>,
    "DateTime"  -> <|
        "Test"     :> Function[d, DateObjectQ[d] && MatchQ[d["Granularity"], "Second" | "Instant"]],
        "Examples" :> {
            DateObject[{2018, 6, 21, 16, 6, 8}, "Instant", "Gregorian", 2.],
            DateObject[{2016, 6, 21, 16, 6, 8}, "Instant", "Gregorian", 2.]
        }
    |>,
    "Time"      -> <|
        "Test"     :> TimeObjectQ, 
        "Examples" :> {
            TimeObject[{16, 6, 2}, TimeZone -> 2.],
            TimeObject[{08, 6, 2}, TimeZone -> 2.]
        }
    |>,
	"TimeQuantity" -> <|
		"Test" :> Function[QuantityQ[#] && UnitDimensions[#] === {{"TimeUnit", 1}}],
		"Examples" :> {
			Quantity[2, "Hours"],
			MixedRadixQuantity[{1, 9}, {"Months", "Days"}]
		}
	|>,
    "ByteArray" -> <|
        "Test"     -> ByteArrayQ,
        "Examples" :> {
            ByteArray[{104, 101, 108, 108, 111}],
            ByteArray[{119, 111, 114, 108, 100}]
        }
    |>,
    "Unsupported" -> <|
        "Test" -> Function[False],
        "Examples" -> {}
    |>
|>

$metaPattern = meta: _Association?(DBUnevaluatedPatternCheck[AssociationQ]): <||>;

DBType[DBTyped[_, type_]] := DBType[type]

(* Utility to get all patterns, results are memoized because they are used in other places *)
DBType[All, rest___] := 
    DBType[Map[DBType, DBTypeUnion @@ Keys[$registry]], rest]

(* Generic Patterns *)

DBType[f_DBType] := 
    f 

DBType[f_Failure] := 
    DBRaise @ f


DBType[t_, <||>] :=
    DBType[t]


(* Handling DBTypeUnion, DBTypeIntersection *)
DBType[DBTypeIntersection[]] := DBType[DBTypeUnion[]]
DBType[(DBTypeUnion | DBTypeIntersection)[s_DBType?DBTypeQ], rest___] :=
    DBType[s, rest]

replaceAllRepeated := ReplaceAll[
	{
		HoldPattern[DBType[DBTypeUnion[x___]]] :> replaceAllRepeated[Or[x]],
		HoldPattern[DBType[DBTypeIntersection[x___]]] :> replaceAllRepeated[And[x]],
		DBTypeUnion[x___] :> replaceAllRepeated[Or[x]],
		DBTypeIntersection[x___] :> replaceAllRepeated[And[x]]
	}
]

iDBTypeSimplify[t_] := Block[{$simplification = True},
	ReplaceAll[
		LogicalExpand[
			replaceAllRepeated[t]
		],
		{
			a_Association?AssociationQ :> a,
			Or :> Function[
				DBType @* DBTypeUnion @@ Map[DBType, {##}]
			],
			And :> Function[
				DBType @* DBTypeIntersection @@ Map[DBType, {##}]
			],
			False :> DBType[DBTypeUnion[]]
		}
	]
]

DBType[t: (DBTypeUnion | DBTypeIntersection)[__], rest___] /; !TrueQ[$simplification] := With[
    {res = iDBTypeSimplify[t]},
	DBType[res, rest] /; res =!= t
]
DBType[type: (_DBTypeUnion | _DBTypeIntersection), $metaPattern]["AddDefaults"] := DBType[
    Map[#["AddDefaults"]&, type]
]
DBType[HoldPattern[DBTypeUnion[]], ___]["Repr"] := "EmptyType"
DBType[HoldPattern[DBTypeIntersection[types___]], ___]["Repr"] := StringRiffle[#["Repr"]& /@ {types}, " | "]
DBType[HoldPattern[DBTypeUnion[types__]], ___]["Repr"] := StringRiffle[#["Repr"]& /@ {types}, " \[Union] "]
DBType[type: (_DBTypeUnion | _DBTypeIntersection), ___]["Type"] :=
    type
DBType[type: (_DBTypeUnion | _DBTypeIntersection), ___]["Constituents"] :=
    List @@ type
DBType[type: (_DBTypeUnion | _DBTypeIntersection), ___]["Properties"] :=
    {
		"Type", "AddDefaults", "Test", "Examples", "DeduceExpressionFunction",
		"Constituents", "Dimensions", "Depth", "Repr"
	}
(type: DBType[_DBTypeUnion, ___])["DeduceExpressionFunction"] :=
    With[
        {tests = Map[#["DeduceExpressionFunction"] &, type["Constituents"]]},
        Function[
            {expr},
			Replace[
				DeleteCases[
					#[expr]& /@ tests,
					DBType[DBTypeUnion[]]
				],
				{
					{s_} :> s,
					s_ :> DBType[DBTypeIntersection @@ s]
				}
			]
		]
	]


(t: DBType[_DBTypeUnion, ___])["Examples"] :=
    Join @@ Transpose[Map[#["Examples"] &, t["Constituents"]]]

(t: DBType[_DBTypeIntersection, ___])["Examples"] :=
    Select[
        DeleteDuplicates[Join @@ Map[#["Examples"] &, t["Constituents"]]],
        t["Test"]
    ]

DBType[type_DBTypeUnion, ___]["Test"] :=
    Function[
        value,
        AnyTrue[type, #["Test"][value] &]
    ]

DBType[type_DBTypeIntersection, ___]["Test"] :=
	Function[
		value,
		AllTrue[type, #["Test"][value] &]
	]


(* Repeating Element *)

DBType[i: RepeatingElement[type: Except[_DBType?DBTypeQ], bounds___]] := With[
    {res = RepeatingElement[DBType[type], bounds]},
    DBType[res] /; i =!= res
]
DBType[type_RepeatingElement, ___]["Type"] :=
    type
DBType[RepeatingElement[type_, ___], ___]["Repr"] := "RepeatingElement[" <> type["Repr"] <> "]"
DBType[type_RepeatingElement, ___]["Properties"] :=
    {
		"Type", "AddDefaults", "Test", "Examples", "Constituents", "Dimensions", "Depth",
		"Min", "Max", "DeduceExpressionFunction", "Repr"
	}
DBType[type_RepeatingElement, ___]["Min"] := Replace[
    First[Rest[type], {0, Infinity}],
    {
        {min_, _} :> min,
        _ -> 0
	}
]
DBType[type_RepeatingElement, ___]["Max"] := Replace[
	First[Rest[type], {0, Infinity}],
    {_, max_} :> max
]
DBType[type_RepeatingElement, ___]["Constituents"] := {First[type]}

(dbt: DBType[type_RepeatingElement, ___])["Test"] := Function @ And[
    ListQ[#],
    IntervalMemberQ[Interval[{dbt["Min"], dbt["Max"]}], Length[#]],
    AllTrue[#, First[type]["Test"]]
]

(dbt: DBType[_RepeatingElement, ___])["Examples"] :=
    With[
        {examples = First[dbt["Constituents"]]["Examples"]},
        Map[
            PadRight[
                RotateRight[examples, #],
                Max[dbt["Min"], 1],
                RotateRight[examples, #]
            ] &,
            Range[Min[4, Length[examples]]]
        ]
    ]

(* SquareRepeatingElement *)

DBType[i: SquareRepeatingElement[type: Except[_DBType?DBTypeQ], bounds___]] := With[
	{res = SquareRepeatingElement[DBType[type], bounds]},
	DBType[res] /; i =!= res
]
DBType[type_SquareRepeatingElement, ___]["Type"] :=
	type
DBType[SquareRepeatingElement[type_, ___], ___]["Repr"] := "SquareRepeatingElement[" <> type["Repr"] <> "]"
DBType[type_SquareRepeatingElement, ___]["Properties"] :=
	{
		"Type", "AddDefaults", "Test", "Examples", "Constituents", "Dimensions", "Depth",
		"Min", "Max", "DeduceExpressionFunction", "Repr"
	}
DBType[type_SquareRepeatingElement, ___]["Min"] := Replace[
	First[Rest[type], {0, Infinity}],
	{
		{min_, _} :> min,
		_ -> 0
	}
]
DBType[type_SquareRepeatingElement, ___]["Max"] := Replace[
	First[Rest[type], {0, Infinity}],
	{_, max_} :> max
]
DBType[type_SquareRepeatingElement, ___]["Constituents"] := {First[type]}

(dbt: DBType[type_SquareRepeatingElement, ___])["Test"] := Function @ And[
	SquareMatrixQ[#],
	IntervalMemberQ[Interval[{dbt["Min"], dbt["Max"]}], Length[#]],
	IntervalMemberQ[Interval[{dbt["Min"], dbt["Max"]}], Length[First[#]]],
	AllTrue[#, First[type]["Test"], 2]
]

(dbt: DBType[type_SquareRepeatingElement, ___])["Examples"] :=
    With[
        {t = DBType @ RectangularRepeatingElement[
                First[type],
                ConstantArray[{dbt["Min"], dbt["Max"]}, 2]
            ]
        },
        t["Examples"]
    ]


(* RectangularRepeatingElement *)

DBType[i: RectangularRepeatingElement[type: Except[_DBType?DBTypeQ], bounds___]] := With[
	{res = RectangularRepeatingElement[DBType[type], bounds]},
	DBType[res] /; i =!= res
]
DBType[type_RectangularRepeatingElement, ___]["Type"] :=
	type
DBType[RectangularRepeatingElement[type_, ___], ___]["Repr"] := "RectangularRepeatingElement[" <> type["Repr"] <> "]"
DBType[type_RectangularRepeatingElement, ___]["Properties"] :=
	{
		"Type", "AddDefaults", "Test", "Examples", "Constituents", "Dimensions", "Depth",
		"MinRows", "MaxRows", "MinColumns", "MaxColumns", "DeduceExpressionFunction", "Repr"
	}
DBType[type_RectangularRepeatingElement, ___]["MinRows"] := Replace[
	First[Rest[type], {{0, Infinity}, {0, Infinity}}],
	{
        {{min_, _}, _} :> min,
		_ -> 0
	}
]
DBType[type_RectangularRepeatingElement, ___]["MaxRows"] := Replace[
	First[Rest[type], {{0, Infinity}, {0, Infinity}}],
	{
		{{_, max_}, _} :> max,
		{max_, _} :> max
	}
]
DBType[type_RectangularRepeatingElement, ___]["MinColumns"] := Replace[
	First[Rest[type], {{0, Infinity}, {0, Infinity}}],
	{
		{_, {min_, _}} :> min,
		_ -> 0
	}
]
DBType[type_RectangularRepeatingElement, ___]["MaxColumns"] := Replace[
	First[Rest[type], {{0, Infinity}, {0, Infinity}}],
	{
		{_, {_, max_}} :> max,
		{_, max_} :> max
	}

]
DBType[type_RectangularRepeatingElement, ___]["Constituents"] := {First[type]}

(dbt: DBType[type_RectangularRepeatingElement, ___])["Test"] := Function @ And[
	MatrixQ[#],
	IntervalMemberQ[Interval[{dbt["MinRows"], dbt["MaxRows"]}], Length[#]],
	IntervalMemberQ[Interval[{dbt["MinColumns"], dbt["MaxColumns"]}], Length[First[#]]],
	AllTrue[#, First[type]["Test"], 2]
]

(dbt: DBType[_RectangularRepeatingElement, ___])["Examples"] :=
    With[
        {examples = First[dbt["Constituents"]]["Examples"]},
        Map[
            ArrayPad[
                {{First @ RotateRight[examples, #]}},
                {{0, Max[dbt["MinRows"], 1] - 1}, {0, Max[dbt["MinColumns"], 1] - 1}},
                RotateRight[examples, #]
            ] &,
            Range[Min[4, Length[examples]]]
        ]
    ]

(*CompoundElement*)
DBType[CompoundElement[types: {__Rule}]] := DBType[Association[types]]
DBType[CompoundElement[types: _List | _Association?AssociationQ]] := With[
    {res = DBType /@ types},
    DBType[CompoundElement[res]] /; types =!= res
]
DBType[elem_CompoundElement, ___]["Type"] :=
    elem
DBType[CompoundElement[types_List, ___], ___]["Repr"] :=
    "CompoundElement[{" <> StringRiffle[#["Repr"]& /@ types, ", "] <> "}]"
DBType[type_CompoundElement, ___]["Properties"] :=
    {
		"Type", "AddDefaults", "Test", "Examples", "Constituents",
		"Dimensions", "Depth", "DeduceExpressionFunction", "Repr"
	}
DBType[type_CompoundElement, ___]["Constituents"] := Replace[
    First[type],
    a_?AssociationQ :> Values[a]
]
DBType[CompoundElement[elems_], ___]["Examples"] :=
    With[
        {data   = MapIndexed[RotateRight[#["Examples"], Last[#2] - 1] &, Values[elems]]},
        {length = Max @ Map[Length, data]},
        Transpose[
            Map[
                PadRight[#, length, #] &,
                data
            ],
            AllowedHeads -> All
        ]
    ]

DBType[CompoundElement[elems_], ___]["Test"] := Function[
    And[
        Or[
            ListQ[elems] && ListQ[#] && Length[elems] === Length[#],
            AssociationQ[elems] && AssociationQ[#] && Keys[elems] === Keys[#]
        ],
        AllTrue[
            Transpose[{elems, #}, AllowedHeads -> All],
            Function[a, a[[1]]["Test"][a[[2]]]]
        ]
    ]
]

(* Single type implementation *)

$DBTypePattern = Alternatives @@ Keys[$registry]

DBType[type: $DBTypePattern, ___]["Type"] :=
    type

DBType[type: $DBTypePattern, ___]["Repr"] := type

DBType[type: $DBTypePattern, ___]["Properties"] :=
    Join[
        {
			"Type", "AddDefaults", "Test", "Examples", "DeduceExpressionFunction",
			"Dimensions", "Depth", "CursorProcessor", "Repr"
		},
        Keys @ getTypeInfo[type][["Arguments"]]
    ]

dbt_DBType["CursorProcessor", backend_] :=
    defaultDeserializer[backend, dbt]

DBType[type: $DBTypePattern, $metaPattern]["AddDefaults"] := DBType[
	type,
	Join[
		getTypeInfo[type][["Arguments", All, "Default"]],
		meta
	]
]

DBType[type: $DBTypePattern, ___][p:"Test"|"Examples"] :=
    $registry[[type, p]]

DBType[s: _Rule | _RuleDelayed | {_Rule | _RuleDelayed...}, rest___] :=
    DBType[<|s|>, rest]
DBType[s: {_String, ___}, rest___] :=
    DBType[validateType[s], rest]

DBType[a: KeyValuePattern[(Rule | RuleDelayed)["Type", t_]]] :=
    DBType[t, KeyDrop[a, "Type"]]

(* Query and DBModelInstance types *)

DBType["Query" | "DatabaseModelInstance", a_Association?AssociationQ]["Properties"] := {
	"PrimaryKey", "Fields", "Type", "Repr"
}

DBType[t: "Query" | "DatabaseModelInstance", a: _Association?AssociationQ : <||>]["Type"] :=
	t

DBType["DatabaseModelInstance", a:_Association?AssociationQ: <||>]["Repr"] := "Entity"

DBType["Query", a:_Association?AssociationQ: <||>]["Repr"] := "EntityClass"

DBType[t: "Query" | "DatabaseModelInstance", a: _Association?AssociationQ : <||>]["Depth"] := 0

DBType[t: "Query" | "DatabaseModelInstance", a: _Association?AssociationQ : <||>]["Dimensions"] := {}

DBType["Query" | "DatabaseModelInstance", a_Association?AssociationQ][k: "PrimaryKey" | "Fields"] :=
    a[k]

(* Fallback patterns for undefined types *)

type_DBType["Properties"] /; Not[DBTypeQ[type]] := 
    {"Type", "AddDefaults", "Test", "Examples", "DeduceExpressionFunction"}
type_DBType["Test"] /; Not[DBTypeQ[type]] := 
    Function[False]

(type: DBType[t: "DateTime" | "Time"])["DeduceExpressionFunction"] :=
    With[
		{test = type["Test"]},
    	If[
			test[#],
			DBType[t, <|"TimeZone" -> #["TimeZone"] =!= None|>],
			DBType[DBTypeUnion[]]
		]&
	]

type_DBType["DeduceExpressionFunction"] := 
    With[
        {test = type["Test"]},
        If[test[#], type, DBType[DBTypeUnion[]]] &
    ]
type_DBType["Type"] /; Not[DBTypeQ[type]] := 
    DBRaise @ validateType[First[type, None]]

type_DBType[All] := 
    type[type["Properties"]]
type_DBType[s_List] := 
    AssociationMap[type, s]

type_DBType["Dimensions"] := getDimensions[type]
type_DBType["Depth"] := Length[type["Dimensions"]]

(* Generic property extraction*)
(dbt: (DBType[type: $DBTypePattern]))[prop_?StringQ] /; MemberQ[dbt["Properties"], prop] :=
	Automatic
(dbt: (DBType[type: $DBTypePattern]))[prop_?StringQ] /; !MemberQ[dbt["Properties"], prop] :=
	Missing["NotAProperty", prop]
(dbt: (DBType[t_, a:_Association?AssociationQ:<||>]))[prop_?StringQ] :=
		Lookup[a, prop, Missing["NotAProperty", prop]]



(* dimensions *)

getDimensions[dbt: DBType[RepeatingElement[t1_DBType?DBTypeQ, ___], ___]] :=
	Join[{Interval[{dbt["Min"], dbt["Max"]}]}, getDimensions[t1]]
getDimensions[dbt: DBType[SquareRepeatingElement[t1_DBType?DBTypeQ, ___], ___]] := Join[
	{
		Interval[{dbt["Min"], dbt["Max"]}],
		Interval[{dbt["Min"], dbt["Max"]}]
	},
	getDimensions[t1]
]
getDimensions[dbt: DBType[RectangularRepeatingElement[t1_DBType?DBTypeQ, ___], ___]] := Join[
	{
		Interval[{dbt["MinRows"], dbt["MaxRows"]}],
		Interval[{dbt["MinColumns"], dbt["MaxColumns"]}]
	},
	getDimensions[t1]
]
getDimensions[dbt: DBType[CompoundElement[l_List], ___], ___] := Join[
	{Interval[Length[l]]},
	Replace[
		getDimensions /@ l,
		{
			{a_, b___} /; SameQ[a, b] :> Replace[
				a,
				int_?IntegerQ :> Interval[int],
				{1}
			],
			_ -> {}
		}
	]
]
getDimensions[_] := {}

(* Adding DBTypeQ *)

DBTypeQ[DBType["Query" | "DatabaseModelInstance", $metaPattern]] := True


DBTypeQ[DBType[
    type_?(DBUnevaluatedPatternCheck[StringQ]),
	$metaPattern
]] := MemberQ[Keys[$registry], type]
DBTypeQ[DBType[
    (DBTypeUnion | DBTypeIntersection)[args___DBType?(DBUnevaluatedPatternCheck[DBTypeQ])],
	$metaPattern
]] := True
DBTypeQ[DBType[
    RepeatingElement[_DBType?(DBUnevaluatedPatternCheck[DBTypeQ])],
	$metaPattern
]] := True
DBTypeQ[DBType[
    RepeatingElement[
        _DBType?(DBUnevaluatedPatternCheck[DBTypeQ]),
        y_?((IntegerQ[#] && NonNegative[#]) || # == Infinity&)
    ],
	$metaPattern
]] := True

DBTypeQ[DBType[
    RepeatingElement[
        _?(DBUnevaluatedPatternCheck[DBTypeQ]), {x_, y_}
    ],
	$metaPattern
]] /; And[
    IntegerQ[Unevaluated[x]] && NonNegative[Unevaluated[x]],
    (IntegerQ[Unevaluated[y]] && NonNegative[Unevaluated[y]]) || Unevaluated[y] == Infinity,
    y >= x
] := True
DBTypeQ[DBType[
	SquareRepeatingElement[_DBType?(DBUnevaluatedPatternCheck[DBTypeQ])],
	$metaPattern
]] := True
DBTypeQ[DBType[
	SquareRepeatingElement[
		_DBType?(DBUnevaluatedPatternCheck[DBTypeQ]),
		y_?((IntegerQ[#] && NonNegative[#]) || # == Infinity&)
	],
	$metaPattern
]] := True

DBTypeQ[DBType[
	SquareRepeatingElement[
		_?(DBUnevaluatedPatternCheck[DBTypeQ]), {x_, y_}
	],
	$metaPattern
]] /; And[
	IntegerQ[Unevaluated[x]] && NonNegative[Unevaluated[x]],
	(IntegerQ[Unevaluated[y]] && NonNegative[Unevaluated[y]]) || Unevaluated[y] == Infinity,
	y >= x
] := True
DBTypeQ[DBType[
	RectangularRepeatingElement[_DBType?(DBUnevaluatedPatternCheck[DBTypeQ])],
	$metaPattern
]] := True
DBTypeQ[DBType[
	RectangularRepeatingElement[
		_DBType?(DBUnevaluatedPatternCheck[DBTypeQ]),
		{
			x_?((IntegerQ[#] && NonNegative[#]) || # == Infinity&),
			y_?((IntegerQ[#] && NonNegative[#]) || # == Infinity&)
		}
	],
	$metaPattern
]] := True

DBTypeQ[DBType[
	RectangularRepeatingElement[
		_?(DBUnevaluatedPatternCheck[DBTypeQ]), {{min1_, max1_}, {min2_, max2_}}
	],
	$metaPattern
]] /; And[
	IntegerQ[Unevaluated[min1]] && NonNegative[Unevaluated[min1]],
	(IntegerQ[Unevaluated[max1]] && NonNegative[Unevaluated[max1]]) || Unevaluated[max1] == Infinity,
	max1 >= min1,
	IntegerQ[Unevaluated[min2]] && NonNegative[Unevaluated[min2]],
	(IntegerQ[Unevaluated[max2]] && NonNegative[Unevaluated[max2]]) || Unevaluated[max2] == Infinity,
	max2 >= min2
] := True
DBTypeQ[DBType[
    CompoundElement[{___DBType?(DBUnevaluatedPatternCheck[DBTypeQ])}],
    $metaPattern
]] := True
DBTypeQ[DBType[
    CompoundElement[<|(_ -> _DBType?(DBUnevaluatedPatternCheck[DBTypeQ]))...|>],
    $metaPattern
]] := True

DBTypeQ[_] := False


iDBDeduceExpressionType = DBType[All]["DeduceExpressionFunction"];


(* TODO: this might be not entirely correct, perhaps we need istead type promotion *)

(* This rule deduces a list of values as a "column" of general type. This is needed
for IN, and may be a few other places *)
DBDeduceExpressionType[expr:{Except[_List]...}] :=
    DBType @ RepeatingElement[DBTypeUnion @@ Map[DBDeduceExpressionType, expr]]

DBDeduceExpressionType[None | _?MissingQ] := DBType[DBTypeIntersection @@ Keys[$registry]]

DBDeduceExpressionType[expr_] :=  iDBDeduceExpressionType[expr]
