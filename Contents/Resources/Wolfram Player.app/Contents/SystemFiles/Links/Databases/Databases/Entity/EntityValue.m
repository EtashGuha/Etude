Package["Databases`Entity`"]

PackageImport["EntityFramework`"]

PackageImport["Databases`"]  (* RelationalDatabase *)
PackageImport["Databases`Common`"]
PackageImport["Databases`Database`"] (* DBRunQuery *)
PackageImport["Databases`SQL`"]
PackageImport["Databases`Schema`"] (* TODO: check if this import is still needed *)


PackageScope["entityValue"]
PackageScope["canonicalProperties"]

entityValue[es_][args___] := DBHandleError[entityValue, $errorHandler][iEntityValue[es][args]]

iEntityValue[es_EntityStore][eps: {__EntityProperty}, subprops: {___}] := ConstantArray[
	Missing["NotAvailable"],
	{Length[eps], Length[subprops]}
]

iEntityValue[es_EntityStore][EntityPropertyClass[q_, {}], {"CanonicalName"}] :=
	Map[
		List @* Last @* toEntityProperty,
		DBEntityQueryToQueryBuilderObject[es, q] @ Databases`SQL`DBGetPrefixedSelectedFields[]
	]



iEntityValue[es_EntityStore][q_, props: {___, EntityProperty[t_, "CanonicalName"], ___}] :=
	Module[{pks, pos, res},
		pks = canonicalProperties[es, q];
		pos = FirstPosition[props, EntityProperty[t, "CanonicalName"], None, {1}, Heads -> False];
		Replace[
			Length[props],
			{
				1 :> If[
					pks === {},
					ConstantArray[
						{Missing["UnknownProperty", {q, "CanonicalName"}]},
						iEntityValue[es][q, "EntityCount"]
					],
					If[Length[pks] > 1, Map[List], Identity] @ iEntityValue[es][q, pks]
				],
				_ :> (
					res = iEntityValue[es][q, Join[Delete[props, pos], pks]];
					If[res === {}, Return[res]];
					res  = Transpose @ res;
					Transpose @ Insert[
						Drop[res, - Length[pks]],
						Replace[
							Take[res, - Length[pks]],
							{
								{a_} :> a,
								{} :> ConstantArray[
									Missing["UnknownProperty", {q, "CanonicalName"}],
									Length[First[res]]
								],
								l_ :> Transpose[l]
							}
						],
						pos
					]
				)
			}
		]
	]

(*
**  TODO I have replaced the call to
**
**     	First @ EntityValue[q, "count"]
**
**  with
**
** 		First @ First @ iEntityValue[es][q, {"count"}],
**
** because the former led to infinite recursion for reasons yet to be understood. Not
** sure to what extent this replacement is correct.
*)
iEntityValue[es_EntityStore][q_, "EntityCount"] :=  First @ First @ iEntityValue[es][
	AggregatedEntityClass[
		q,
		"count" -> EntityFunction[x, Length[x]]
	],
	{"count"}
]

iEntityValue[es_EntityStore][l_List, "EntityCount"] :=  Length[l]

iEntityValue[es_EntityStore][q_, "SampleEntities"] := EntityList[SampledEntityClass[q, 10]]

iEntityValue[es_EntityStore][RandomEntity, q_, n_] := 
    With[{result = EntityList[SampledEntityClass[q, n]]},
        Which[
            n === 1,
                Message[RandomEntity::unsupported1],
            n === Length @ result,
                Message[RandomEntity::unsupported, n],
            n > Length[result],
                Message[RandomEntity::insfct, n, Length @ result]
        ];
        result
    ]

(*
	TODO: so far Toni doesn't provide hooks to override this,
	so this is dead code pending changes from him
*)
iEntityValue[es_EntityStore][x_,
	props: {
		EntityProperty[_, _, {}, Total | Mean | StandardDeviation | Variance | Min | Max | Length]..
	}
] := With[
	{names = propNames[props]},
	First[iEntityValue[es][
		AggregatedEntityClass[
			x,
			Thread[Rule[
				names,
				Replace[
					props,
					EntityProperty[_, prop_, {}, func_] :>
						EntityFunction[y, func[y[prop]]],
					{1}
				]
			]]
		],
		names
	]]
]

iEntityValue[es_EntityStore][
	x: {Repeated[Entity[q_, _], {2, Infinity}]},
	props_List
] := With[{pk = canonicalProperties[es, q]},
	With[{names = propNames[pk]},
		With[{annotated = ExtendedEntityClass[x, Thread[names -> pk]]},
			Replace[
				iEntityValue[es][annotated, Join[props, names]],
				{
					miss_?MissingQ :> ConstantArray[miss, {Length[x], Length[props]}],
					nonmiss_ :>
						Replace[
							Values[Part[
								Association[Apply[
									Entity[q, Replace[#2, {a_} :> a]] -> #1&,
									Map[
										TakeDrop[#, Length[props]]&,
										nonmiss
									],
									{1}
								]],
								Key /@ x
							]],
							Missing["KeyAbsent", ent_] :>
								Missing["UnknownEntity", List @@ ent],
							{1}
						]
				}
			]
		]
	]
]

iEntityValue[es: EntityStore[_, _RelationalDatabase, ___]][x_, props_List] :=
    With[{processed = processProperties[x, props]},
        Replace[processed,
            {Hold[q_], p_} :> iEntityValue[es][q, p]
        ] /; processed =!= {Hold[x], props}
    ]

(*
**  The main call that constructs DBQueryBuilderObject, then DBQueryObject, executes
**  the query on the database, and processes the result.
*)

iEntityValue[es: EntityStore[_, _RelationalDatabase, ___]][x_, props_List] /;
    !DuplicateFreeQ[props] := With[{
		dedup = DeleteDuplicates[props]},
		Replace[
			iEntityValue[es][x, dedup], {
                {} -> {},
    			l: {__List} :> With[
    				{pos = MapThread[
    					Function[{prop, data},
    						Map[
    							# -> data&,
    							Position[props, Verbatim[prop], {1}, Heads -> False]
    						]
    					],
    					{dedup, Transpose[l]}
    				]},
    				Transpose[ReplacePart[
    					ConstantArray[Missing[], Length[props]],
    					Join @@ pos
    				]]
    			]
        }]
	]


iEntityValue[es: EntityStore[_, _RelationalDatabase, ___]][x_, props_List] /;
    DuplicateFreeQ[props] := Module[{fieldMap, queryObject, qbo, missing, data, len},
	qbo = DBEntityQueryToQueryBuilderObject[es, x];
	missing = Position[
		qbo @ DBGetPrefixedFields[toPrefixedField /@ props],
		$Failed,
		{1},
		Heads -> False
	];
	If[
		Length[missing] === Length[props],
		len = iEntityValue[es][x, "EntityCount"];
		data = {},
		qbo = qbo @ DBValues[Delete[toPrefixedField /@ props, missing]];
		queryObject = DBQueryObject[
			qbo,
			"RelationFunction" -> EntityRelationFunction
		];
		data = DBRunQuery[
			queryObject,
			"Columns"
		];
		len = Length[First[data]];
	];
	Transpose[Fold[
		Insert[#, #2[[2]], #2[[1]]] &,
		data,
		Map[
			# -> ConstantArray[Missing["UnknownProperty", List @@ Extract[props, #]], len]&,
			missing
		]
	]]
]

SetAttributes[canonicalProperties, HoldRest]

canonicalProperties[_, AggregatedEntityClass[_, _]] := {}

canonicalProperties[
	es: EntityStore[meta_, _RelationalDatabase, ___],
    Alternatives[
	   type_?(DBUnevaluatedPatternCheck[StringQ]),
       EntityClass[type_?(DBUnevaluatedPatternCheck[StringQ]), {}]
    ]
] := Map[
	EntityProperty[type, #]&,
    meta[["Types", type, "CanonicalNameProperties"]]
]

canonicalProperties[es: EntityStore[meta_, _RelationalDatabase, ___], {Entity[query_, _]..}] :=
    canonicalProperties[es, query]

canonicalProperties[
	EntityStore[meta_, schema : _RelationalDatabase, ___],
	query: Except[_List, _?(DBUnevaluatedPatternCheck[Internal`PossibleEntityListQ])]
] := With[
	{es = Internal`InheritedBlock[{DatabaseStore},
		Options[DatabaseStore] = {"AddRelations" -> True};
		Quiet[
			Check[
                Block[{$DBQueryLightWeightCompilation = True},
                    EntityStore[{"dummy" -> query}, schema]
                ]
				,
				DBRaise[
					Internal`CheckCache[
						iEntityToDatabaseStore[{"dummy" -> query}, schema]
					]
				],
				EntityStore::invent
			],
			{EntityStore::nocanon, EntityStore::invent}
		]
	]},
	If[
		Head[es] === EntityStore,
		Map[
			EntityProperty[
				es[[1, "Types", "dummy", "Properties", #, "ColumnPrefix"]],
				es[[1, "Types", "dummy", "Properties", #, "ColumnName"]]
			]&,
			es[[1, "Types", "dummy", "CanonicalNameProperties"]]
		],
		es
	]
]

DBDefError["invalid_class_in_canonicalProperties"] @ canonicalProperties