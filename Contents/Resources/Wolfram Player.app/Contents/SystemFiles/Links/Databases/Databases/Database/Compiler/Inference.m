Package["Databases`Database`"]


PackageImport["Databases`"]
PackageImport["Databases`Common`"]
PackageImport["Databases`SQL`"]
PackageImport["Databases`Schema`"]


PackageExport["DBTypeAwareCompiler"]


(*
**  Returns a function that should be applied to an intermediate AST to compile it
*)
DBTypeAwareCompiler[
    compilationContext_Association?AssociationQ,
    isAggregate: (True | False)
] := typeDecorate[#, compilationContext, isAggregate]&


getDBType[types: _List|_?AssociationQ] /; AllTrue[types, MatchQ[_DBTyped]] := Map[getDBType, types]
getDBType[DBTyped[_, type_]] := type

getExpr[types: _List|_?AssociationQ] /; AllTrue[types, MatchQ[_DBTyped]] := Map[getExpr, types]
getExpr[DBTyped[expr_, _]] := expr

DBDefError[getDBType, getExpr]

decorateHelper[name_, oldArgType_, isAggregate_] := Module[
	{isAggregateButNotAggregatingSymbol, transform, type, argType},
	argType = oldArgType;
	isAggregateButNotAggregatingSymbol = And[
		TrueQ[isAggregate],
		!DBAggregatingOperatorQ[name],
		!FreeQ[argType, RepeatingElement]
	];
	(*
		if isAggregate is true but we have an astSymbol that is not performing aggregation
		we can treat the inner fields as scalars. This guarantees that aggregation is performed
		exactly once
	*)
	If[
		isAggregateButNotAggregatingSymbol,
		argType = Replace[
			argType,
			RepeatingElement[t_] :> t,
			If[name === "Case", {7}, {4}]
		]
	];
	(*This removes the RepeatingElement from the arg types*)
	{transform, type} = Most[DBGetTypeSignatures[name, argType]];
	If[type === DBType[DBTypeUnion[]],
		DBRaise[
			typeDecorate, "unknown_operator_or_type_signature", {},
			<|
				"Operator" -> name,
				"ArgumentsCompoundType" -> argType,
				"FailingExpr" -> Last[$annotationStack]
			|>
		]
	];
	If[
		isAggregateButNotAggregatingSymbol,
		type = DBType[RepeatingElement[type]]
	];
	(*This puts it back in the return type*)
	{transform, type}

]

extractHead[DBAnnotation[a_, ___]] := extractHead[a]

extractHead[astSymbol[_][args___]] := Flatten[extractHead /@ {args}]

extractHead[head_[_]] := {head}

extractHead[atom_] := {}


checkEmptyTypes[args_, name] := With[{empty = Cases[args, DBTyped[_, DBType[DBTypeUnion[]]]]},
	If[empty =!= {},
		DBRaise[checkEmptyTypes, "some_argument_types_unresolved", {},
			<|"Operator" -> name, "FailingArguments" -> empty|>
		]
	]
]


$annotationStack = {}

typeDecorate[DBAnnotation[expr_, annot_], rest___] := 
    Block[{$annotationStack = Append[$annotationStack, annot]},
        typeDecorate[expr, rest]
    ]
    
typeDecorate[
	astSymbol[a: "Any" | "All"][first_, astSymbol[op_], list_],
	compilationContext_,
	isAggregate_
] := Module[{args, argType, transform, type},
	args = typeDecorate[#, compilationContext, isAggregate]& /@ {first, list};
	checkEmptyTypes[args, a];
	If[
		!MatchQ[Last[args], DBType[_RepeatingElement, ___]],
		DBRaise[typeDecorate, "scalar_in_all_or_any", <|"FailingExpr" -> Last[$annotationStack]|>]
	];
	argType = DBType[CompoundElement[{
		getDBType[First[args]],
		DBType[DBTypeUnion @@ getDBType[Last[args]]["Constituents"]]
	}]];
	{transform, type} = decorateHelper[op, argType, isAggregate];
	transform = Replace[
		transform,
		{
			HoldPattern[Function[DBSQLSymbol[sym_][_]]] :> Function[DBSQLSymbol[sym][#1, DBSQLSymbol[a][#2]]],
			any_ :> DBRaise[typeDecorate, "invalid_all_or_any_spec", {any, a}]
		}
	];
	DBTyped[Apply[transform, getExpr[args]], type]
]

aggregationHelper[qbo_, transform_, type_, compilationContext_, propnameann_:None] := With[
	{propnameagg = DBGenerateUniquePropertyName[]},
	typeDecorate[
		DBTyped[
			(qbo @ DBAggregate[{propnameagg -> DBTyped[
				Apply[
					transform,
					{DBSQLField[propnameann]}
				],
				type
			]}]),
			DBType["DatabaseModelInstance"]
		][propnameagg],
		compilationContext,
		False
	]
]

$relationChainPattern = (_DBSQLSlot[_]) | (x_ /; MatchQ[x, $relationChainPattern[_]])

typeDecorate[astSymbol[name_][a_], compilationContext_, False] /; DBAggregatingOperatorQ[name] :=
	Module[{type, transform, argType, arg, h, qbo, x, newCC, tag, partialTypes,
		propnameann = DBGenerateUniquePropertyName[]},
		x = DBUniqueTemporary["x"];
		If[
			DBQueryBuilderObjectQ[a],
			qbo = a,
			If[
				And[
					MatchQ[a, $relationChainPattern],
					MatchQ[
						DBGetTypeSignatures[name],
						{___, DBType[CompoundElement[{DBType["Query"]}]] -> _, ___}
					]
				],
				h = a,
				h = extractHead[a];
				If[
					Length[DeleteDuplicates[h]] =!= 1,
					DBRaise[
						typeDecorate,
						"attempting_aggregation_on_multiple_subqueries",
						{h},
						<|"FailingExpr" -> Last[$annotationStack]|>
					];,
					h = First[h]
				]
			];
			{h, {partialTypes}} = Block[{$inner =  True}, Reap[
				SelectFirst[
					Reverse[Most[NestWhileList[Head, h, Length[#] =!= 0&]]],
					MatchQ[
						Sow[#, tag] & @ getDBType @ typeDecorate[
							#,
							compilationContext,
							False
						],
						DBType["Query", ___]
					]&
				],
				tag,
				#2&
			]];

			If[
				And[
					MissingQ[h],
					MatchQ[partialTypes, {DBType["DatabaseModelInstance", ___] ..}]
				],
				DBRaise[
					typeDecorate,
					"attempting_aggregation_on_many_to_one_relation_chain",
					{},
					<|"FailingExpr" -> Last[$annotationStack]|>
				]
			];

			qbo = Block[{$inner = True},
				First @ typeDecorate[
					h,
					compilationContext,
					False
				]
			]
		];
		newCC = Append[
			compilationContext,
			"SlotQueryMap" -> Append[compilationContext["SlotQueryMap"], DBSQLSlot[x] -> qbo]
		];
		arg = typeDecorate[ReplaceAll[a, h -> DBSQLSlot[x]], newCC, True];
		arg = ReplaceAll[
			arg,
			q_?DBQueryBuilderObjectQ /; q @ DBGetName[] === qbo @ DBGetName[] :> DBSQLField
		];
       	checkEmptyTypes[{arg}, name];
		qbo = qbo @ DBAnnotate[propnameann -> arg];
		argType = DBType[CompoundElement[{getDBType[arg]}]];
		{transform, type} = decorateHelper[name, argType, True];
		aggregationHelper[qbo, transform, type, newCC, propnameann]
	]

typeDecorate[astSymbol[name_][a___], compilationContext_, isAggregate_] :=
	Module[{type, transform, argType, args},
		args = typeDecorate[#, compilationContext, isAggregate]& /@ {a};
		checkEmptyTypes[args, name];
		argType = DBType[CompoundElement[getDBType[args]]];
		{transform, type} = decorateHelper[name, argType, isAggregate];
		DBTyped[Apply[transform, getExpr[args]], type]
	]

typeDecorate[slot_DBSQLSlot[_], compilationContext_, _] /;
    !KeyExistsQ[compilationContext["SlotQueryMap"], slot] :=
		DBRaise[typeDecorate, "unbound_slot", {slot}]


typeDecorate[
	slot_DBSQLSlot,
	compilationContext_,
	isAggregate: True | False
] := DBTyped[
        (* 
        ** Note: this is a non-trivial operation, here we replace the slot with 
        ** the DBQueryBuilderObject that corresponds to that slot. So, this is 
        ** the place where we dispose of DBSQLSlot-s.
        *)
		compilationContext["SlotQueryMap"][slot],
		qboType[
			compilationContext["SlotQueryMap"][slot],
			isAggregate
		]
	]

typeDecorate[
	DBTyped[query_?DBInertQueryQ, DBType[type_, meta_: <||>]],
	compilationContext_,
	isAggregate_
] := With[
	{qbo = compileQuery[query, compilationContext]},
	DBTyped[
		If[TrueQ[$inner], qbo, compilationContext["OriginalQBO"]],
		qboType[
			qbo,
			Not[qbo @ DBIsSingleRow[]] || isAggregate,
			meta
		]
	]
]


typeDecorate[
	qbo_?DBQueryBuilderObjectQ,
	compilationContext_,
	isAggregate_
] := DBTyped[
	qbo,
	qboType[
		qbo,
		If[
			AnyTrue[Values[compilationContext["SlotQueryMap"]], # @ DBGetName[] === qbo @ DBGetName[]&],
			isAggregate,
			! qbo @ DBIsSingleRow[]
		]
	]
]

sort[l_List] := Sort[l]
sort[None] := None

qboType[
	qbo_?DBQueryBuilderObjectQ,
	isAggregate: True | False,
	meta_: <||>
] := DBType[
	If[TrueQ[isAggregate], "Query", "DatabaseModelInstance"],
	(*here the signature disagrees with the qbo, but the signature is right*)
	Join[
		meta,
		Map[
			sort @* DBNormalizePrefixedFields,
			<|
				"PrimaryKey" -> qbo @ DBGetPrefixedFields[qbo @ DBPrimaryKey[]],
				"Fields" -> qbo @ DBGetPrefixedSelectedFields[]
			|>
		]
	]
]
(*
	the downvalue below is a shortcut. The only function signature that will match	
	is Exists, so we don't really need to care about the dependencies or internal types
*)
typeDecorate[
	e: DBTyped[
		qbo_?DBQueryBuilderObjectQ, DBType[t: "Query" | "DatabaseModelInstance", ___]
	][fields: {$DBTopLevelFieldNamePattern..}],
	compilationContext_,
	isAggregate: True | False
] := DBTyped[
	qbo[fields],
	DBType[RectangularRepeatingElement[All]]
]

typeDecorate[
	e: DBTyped[
        qbo_?DBQueryBuilderObjectQ, DBType[t: "Query" | "DatabaseModelInstance", ___]
    ][field: $DBTopLevelFieldNamePattern],
	compilationContext_,
	isAggregate: True | False
] := With[{
	isRepeating = TrueQ[isAggregate] || t === "Query",
	type = qbo @ DBGetFieldType[field, True]},
	If[
		And[
			isRepeating,
			MatchQ[type, DBType["Query", ___]]
		],
		DBRaise[
			typeDecorate,
			"attempting_chaining_of_multiple_one_to_many",
			{},
			<|"FailingExpr" -> Last[$annotationStack]|>
		]
	];
	Replace[
		Replace[
			{
				type["Deserializer"],
				type["Dependencies"]
			},
			{
				{_?MissingQ, _} :> qbo[field],
				(* types that have a serializer but no dependencies are pass-through
				   for what concerns the inference step: operations on dates should work transparently
				 *)
				{func_, _?MissingQ} :> qbo[field],
				{func_, _} :> func[qbo]
			}
		],
		{
			(q_?DBQueryBuilderObjectQ)[f_] :> DBTyped[
				q[f],
				If[isRepeating, DBType @* RepeatingElement, Identity][
					qbo @ DBGetFieldType[f]
				]
			],
			any_ :> typeDecorate[
				DBTyped[any, type],
				If[!TrueQ[$inner], Append["OriginalQBO" -> qbo], Identity][
					compilationContext
				],
				isAggregate
			]
		}
	]
]

typeDecorate[
	DBTyped[
		_, DBType[Except["Query" | "DatabaseModelInstance"], ___]
	][f: $DBTopLevelFieldNamePattern | {$DBTopLevelFieldNamePattern..}],
	_,
	_
] := DBRaise[typeDecorate, "field_extraction_from_scalar", {f}]


$inner = False;

typeDecorate[(x: Except[_DBTyped | List | DBSQLSecondsToTimeQuantity])[y_], compilationContext_, isAggregate_] :=
	typeDecorate[
		Block[{$inner = True},
			typeDecorate[x, compilationContext, isAggregate]
			]
		[y],
		compilationContext,
		isAggregate
	]

typeDecorate[list: _List|_?AssociationQ, compilationContext_, isAggregate_] :=
	typeDecorate[
		Map[
			typeDecorate[#, compilationContext, isAggregate]&,
			list
		],
		compilationContext,
		isAggregate
	]


typeDecorate[list: _List|_?AssociationQ, _, _] /; AllTrue[list, MatchQ[_DBTyped]] :=
	DBTyped[
		getExpr[list],
		DBType[CompoundElement[getDBType[list]]]
	]

typeDecorate[boole: True | False, _, _] := DBTyped[DBSQLBoolean[boole], DBType["Boolean"]]

typeDecorate[
	DBSQLSecondsToTimeQuantity[t_],
	compilationContext_,
	isAggregate_
] := Replace[
	typeDecorate[
		t,
		compilationContext,
		isAggregate
	],
	DBTyped[
		n_,
		DBType["Integer" | "Decimal" | "Real", ___]
	] :> DBTyped[
		DBSQLSecondsToTimeQuantity[n],
		DBType["TimeQuantity"]
	]
]

typeDecorate[do: (_DateObject | _TimeObject), _, _] :=
    dateTimeDecorate[
        do, 
        Replace[
			do["TimeZone"],
			{
				(*TODO: I suspect this is a bug in 12*)
				list: {__} :> TimeZoneOffset[First[list], "UTC"],
				tz : Except[None] :> TimeZoneOffset[tz, "UTC"]
			}
		]
    ]

dateTimeDecorate[d_TimeObject, tz_] :=
    (* on 12 TimeObject[][{"Hour"}] is returning errors https://bugs.wolfram.com/show?number=356608 *)
    DBTyped[
		DBSQLTime[tz, Sequence @@ d[{"Hour", "Minute", "SecondExact" }]],
		DBType["Time", <|"TimeZone" -> tz =!= None|>]
	]

dateTimeDecorate[do_DateObject, tz_] := With[{d = CalendarConvert[do, "Gregorian"]},
    Switch[d["Granularity"],
		"Day", 
            DBTyped[
                DBSQLDate[tz, Sequence @@ d[{"Year", "Month", "Day"}]], 
                DBType["Date"]
            ],
		"Instant"|"Second",
            DBTyped[
                DBSQLDateTime[tz, Sequence @@ d[{"Year", "Month", "Day", "Hour", "Minute", "SecondExact" }]],
                DBType["DateTime", <|"TimeZone" -> tz =!= None|>]
            ],
		_, 
            DBRaise[dateTimeDecorate, "unsupported_date", {do, tz}]
	]
]

typeDecorate[e_, __] :=
	With[{type = DBDeduceExpressionType[e]},
		DBTyped[e, type] /; type =!= DBType[DBTypeUnion[]]
	]

typeDecorate[e_, __] := 
    DBRaise[
        typeDecorate, 
        "invalid_arguments", 
        {First @ $annotationStack, Last @ $annotationStack}
    ]

DBDefError @ typeDecorate
