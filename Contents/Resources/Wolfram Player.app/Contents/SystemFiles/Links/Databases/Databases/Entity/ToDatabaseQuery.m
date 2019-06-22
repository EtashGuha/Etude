Package["Databases`Entity`"]

PackageImport["EntityFramework`"]


(*
	RelationalDatabase,
   	DatabaseSQLDistinct, DatabaseWhere, DatabaseAnnotate, DatabaseAggregate,
	DatabaseOrderBy, DatabaseGroupBy, DatabaseSelectFields, DatabaseExcludeFields,
	DatabaseJoin
*)
PackageImport["Databases`"]

(* TODO: perhaps, move this to some higher-level / different module *)
(*
	DBRunQuery, DBQueryToSymbolicSQL
*)
PackageImport["Databases`Database`"]

PackageImport["Databases`Common`"] (* DBRaise *)


(* TODO: check if this import is still needed *)
PackageImport["Databases`SQL`"]

PackageImport["Databases`Schema`"] (* TODO: check if this import is still needed  *)


PackageExport["DBToDatabaseQuery"]
PackageExport["DBEntityQueryToQueryBuilderObject"]
PackageExport["DBEntityDBQuerySQLString"]
PackageExport["DBEntityQueryToSymbolicSQL"]

PackageScope["$CompiledEntityQueries"]
PackageScope["DBHeldAlias"]

(* 
** Inert holding wrapper to wrap entity-level aliases in subqueries. This is 
** needed because parts of the original entity queries present in subqueries,
** may evaluate and produce leaks - see bug # 369568
*)
SetAttributes[DBHeldAlias, HoldAll]

SetAttributes[makeAlias, HoldRest]
makeAlias[q_, alias_, rest___] := 
    With[{realAlias = If[TrueQ[$DBInEntityFunction], DBHeldAlias[alias], alias]},
        DatabaseQueryMakeAlias[q, realAlias, rest]
    ]

$posInteger = _Integer?Positive
$inf = Infinity | DirectedInfinity[1]

(* Custom assigning operator that injects patterns *)
SetAttributes[def, HoldAll]
def /: SetDelayed[def[patterns__][lhs_], rhs_] :=
    SetDelayed @@ ReplaceAll[
        Hold[lhs, rhs],
        Thread[Map[HoldPattern, Unevaluated[{patterns}]] -> {patterns}]
    ]
    


$CompiledEntityQueries = <||>

SetAttributes[DBToDatabaseQuery, HoldAll]


DBToDatabaseQuery[any_] := With[{dbq = DBToDatabaseQuery[None, any]},
	AppendTo[$CompiledEntityQueries, dbq :> any];
	dbq
]


DBToDatabaseQuery[es_, EntityClass[type_, All | {}]] := DBToDatabaseQuery[es, type]
DBToDatabaseQuery[es_, EntityClass[type_, rule_Rule]] :=
	DBToDatabaseQuery[es, EntityClass[type, {rule}]]

DBToDatabaseQuery[es_, EntityClass[type_, allRules: {__Rule}]] :=
    With[{
		rules = Select[allRules, FreeQ[TakeLargest | TakeSmallest | TakeLargestBy | TakeSmallestBy]],
		sorting = Select[allRules, Not @* FreeQ[TakeLargest | TakeSmallest | TakeLargestBy | TakeSmallestBy]]
	},
		With[{ep = EntityProperties[type]},
			With[
				{invalid = Select[
					rules[[All, 1]],
					!MemberQ[ep, Replace[#, s_String :> s | EntityProperty[_, s]]]&
				]},
				If[
					!MatchQ[
						invalid,
						{EntityProperty[_, "Entity" | "CanonicalName"] | "Entity" | "CanonicalName"...}
					],
					DBRaise[DBToDatabaseQuery, "invalid_implicit_entityclass", {invalid}]
				]
			]
		];
		DBToDatabaseQuery[
			es,
			Evaluate[
				normalizeSorting[sorting] @ If[
					rules === {},
					Identity,
					FilteredEntityClass[#, normalizeRules[rules, type]]&
				][type]
			]
		]
	]

DBToDatabaseQuery[es_, Entity[type_, pk_]] := With[
	{cp = toPrefixedField /@ canonicalProperties[es, type]},
	If[
		Length[cp] === Replace[Unevaluated[pk], {l_List :> Length[Unevaluated[l]], _ -> 1}],
		DatabaseModelInstance[
			DBToDatabaseQuery[es, type],
			AssociationThread[cp :> pk]
		],
		DBRaise[DBToDatabaseQuery, "wrong_number_of_pk", {cp, pk}]
	]
]

DBToDatabaseQuery[es_, Entity[type_String?StringQ]] := DBToDatabaseQuery[es, type]
DBToDatabaseQuery[es_, type_String?StringQ] := DatabaseModel[type]


DBToDatabaseQuery[es_, FilteredEntityClass[type_, pred_EntityFunction]] := DatabaseWhere[
	DBToDatabaseQuery[es, type],
	DBCompileEntityFunction[es, pred]
]

DBToDatabaseQuery[es_, q: ExtendedEntityClass[type_, rules_]] := makeAlias[
	DatabaseAnnotate[
		DBToDatabaseQuery[es, type],
		compileAnnotations[es, rules]
	],
	q
]

DBToDatabaseQuery[es_, q: SortedEntityClass[type_, props_]] := DatabaseOrderBy[
	DBToDatabaseQuery[es, type],
	compileSortingFields[es, props]
]

DBToDatabaseQuery[es_, q: SortedEntityClass[type_, props_, lim_]] := Which[
	MatchQ[-lim, Alternatives[
        $inf,
		$posInteger,
		{$posInteger,  All | $inf},
		{m: $posInteger, n: $posInteger } /; n >= m
	]],
	With[{inverted = invertSortingFields[props], minus = -lim},
		DBToDatabaseQuery[
			es,
			SampledEntityClass[SortedEntityClass[type, inverted], minus]
		]
	]	,
	MatchQ[lim, Alternatives[
        $inf,
		$posInteger,
		{$posInteger,  All | $inf},
		{m: $posInteger, n: $posInteger} /; n >= m
	]],
	DBToDatabaseQuery[
		es,
		SampledEntityClass[SortedEntityClass[type, props], lim]
	],
	True,
	DBRaise[DBToDatabaseQuery, "invalid_take_spec_in_SortedEntityClass", {lim}]
]


DBToDatabaseQuery[
	es_,
	q: AggregatedEntityClass[type_, annotation_, aggregator_: {}]
] := makeAlias[
	DatabaseGroupBy[
		DBToDatabaseQuery[es, type],
		compileFields[es, aggregator],
		compileAnnotations[es, annotation]
	],
	q
]


def[$posInteger, $inf] @ DBToDatabaseQuery[
	es_,
	SampledEntityClass[type_, n:$posInteger | $inf]
] := DBToDatabaseQuery[
	es,
	SampledEntityClass[type, {1, n}]
]

def[$posInteger, $inf] @ DBToDatabaseQuery[
	es_,
	SampledEntityClass[type_, {m: $posInteger, $inf | All }]
] := DatabaseOffset[
	DBToDatabaseQuery[es, type],
	m - 1
]

def[$posInteger, $inf] @ DBToDatabaseQuery[
	es_,
	SampledEntityClass[type_, {m: $posInteger, n: $posInteger}]
] := DatabaseLimit[
	DatabaseOffset[
		DBToDatabaseQuery[es, type],
		m - 1
	],
	n - m + 1
] /; n >= m

DBToDatabaseQuery[
	es_,
	SampledEntityClass[type_, any_]
] := DBRaise[DBToDatabaseQuery, "invalid_take_spec_in_SampledEntityClass", {any}]

DBToDatabaseQuery[es_, q: {Entity[x_, _]..}] := If[
	es === None,
	DBRaise[DBToDatabaseQuery, "need_EntityStore_for_single_Entity", {q}],
	DBToDatabaseQuery[
		es,
		Evaluate @ FilteredEntityClass[
			x,
			primaryKeyInverse[es, x, q[[All, 2]]]
		]
	]
]

DBToDatabaseQuery[
	es_,
	q: CombinedEntityClass[q1_, q2_, spec_: Automatic, jtype_: "Inner"]
] := With[{
	left = DBToDatabaseQuery[es, q1],
	right = DBToDatabaseQuery[es, q2]},
	$CompiledEntityQueries = <|
		$CompiledEntityQueries,
		<|left :> q1, right :> q2|>
	|>;
	DatabaseJoin[
		DBToDatabaseQuery[es, q1],
		DBToDatabaseQuery[es, q2],
		compileEntityJoinSpec[es, spec],
		jtype
	]
]

DBToDatabaseQuery[es_, (r: Rule | RuleDelayed)[alias_, query_]] := makeAlias[
	DBToDatabaseQuery[es, query],
	alias,
	True
]

DBToDatabaseQuery[es_, EntityValue[query_, fields_]] :=
	DatabaseSelectFields[
		DBToDatabaseQuery[es, query],
		compileFields[es, fields]
	]

DBToDatabaseQuery[es_, EntityValue[query_, fields_, None]] :=
	DBToDatabaseQuery[es, EntityValue[query, fields]]

DBToDatabaseQuery[es_, EntityValue[query_, fields_, f_]] := With[{
	dummy = DBGenerateUniquePropertyName[],
	x = DBUniqueTemporary["x"]},
	DBToDatabaseQuery[
		es,
		EntityValue[
			Entity[
				AggregatedEntityClass[query, dummy -> EntityFunction[x, f[x[fields]]]],
				{}
			],
			dummy
		]
	]
]

DBToDatabaseQuery[es_, EntityValue[query_, fields_, DeleteDuplicates]] :=
	DatabaseSQLDistinct @ DBToDatabaseQuery[es, EntityValue[query, fields]]

DBDefError @ DBToDatabaseQuery


normalizeRules[rules: {__Rule}, type_] := With[{ent = DBUniqueTemporary["ent"]},
	Replace[
		Replace[
			Hold @@ rules,
			{
				Verbatim[Rule]["Entity" | EntityProperty[_, "Entity"], val_] :> ent == val,
				Verbatim[Rule]["CanonicalName" | EntityProperty[_, "CanonicalName"], val_] :>
        			ent == Entity[type, val],
				Verbatim[Rule][key_, head_[val_]] /; MemberQ[
					Append[$DBBinaryComparisonOperatorForms, MemberQ], head
				] :> head[val][ent[key]]
				,
				Verbatim[Rule][key_, Verbatim[Alternatives][vals__]] :> MemberQ[{vals}, ent[key]],
				Verbatim[Rule][key_, any_] :> any == ent[key]
			},
			{1}
		],
		{
			Hold[r_] :> EntityFunction[ent, r],
			Hold[r__] :> EntityFunction[ent, And[r]]
		}
	]
]


normalizeSorting[rules: {___Rule}] := Composition @@ Map[
	normalizeSorting,
	rules
]

normalizeSorting[prop_ -> TakeLargest[n_]] := SortedEntityClass[#, prop -> "Descending", n]&

normalizeSorting[prop_ -> TakeSmallest[n_]] := SortedEntityClass[#, prop -> "Ascending", n]&

normalizeSorting[prop_ -> TakeLargestBy[f_, n_]] := With[{
	x = DBUniqueTemporary["x"]},
	SortedEntityClass[#, EntityFunction[x, f[x[prop]]] -> "Descending", n]&
]

normalizeSorting[prop_ -> TakeSmallestBy[f_, n_]] := With[{
	x = DBUniqueTemporary["x"]},
	SortedEntityClass[#, EntityFunction[x, f[x[prop]]] -> "Ascending", n]&
]


PackageScope[propNames]

propNames[funcs_List] :=Table[DBGenerateUniquePropertyName[], {i, Length[funcs]}]
propNames[func_] := First[propNames[{func}]]


toPKAssoc[pk: {_}, vals: {Except[_List]..}] := toPKAssoc[pk, List /@ vals]
toPKAssoc[pks_, vals: {{___}..}] := AssociationThread[pks, #]& /@ vals

primaryKeyInverse[es_, x_, vals: {Except[_?AssociationQ]..}] := With[
	{pks = canonicalProperties[es, x]},
	primaryKeyInverse[
		es,
		x,
		toPKAssoc[pks, vals]
	]
]

primaryKeyInverse[es_, x_, {<||>}] := With[
	{ent = DBUniqueTemporary["ent"]},
	EntityFunction[ent, True]
]

primaryKeyInverse[
	es_,
	x_,
	vals: {<|key_ -> _|>..}
] /; Length[First[vals]] == 1 := With[
	{ent = DBUniqueTemporary["ent"], v = vals[[All, 1]]},
	EntityFunction[
		ent,
		MemberQ[v, ent[key]]
	]
]
primaryKeyInverse[es_, x_, vals: {_?AssociationQ..}] /; Length[First[vals]] > 1 := With[
	{ent = DBUniqueTemporary["ent"]},
	With[{body = Or @@ (And @@@ KeyValueMap[ent[#1] == #2&] /@ vals)},
		EntityFunction[
			ent,
			body
		]
	]
]

SetAttributes[DBEntityDBQuerySQLString, HoldAll]
DBEntityDBQuerySQLString[
	es_, query_, formatter: Except[_?OptionQ]: Identity, opts___
] /; MatchQ[es, _EntityStore] :=
	formatter @ DBRunQuery[DBToDatabaseQuery[es, query], "SQLString", opts];


SetAttributes[DBEntityQueryToQueryBuilderObject, HoldRest]
DBEntityQueryToQueryBuilderObject[es: EntityStore[_, _RelationalDatabase, ___], query_] :=
	DBQueryToQueryBuilderObject[DBToDatabaseQuery[es, query], DatabaseStore[es]]

DBDefError @ DBEntityQueryToQueryBuilderObject


SetAttributes[DBEntityQueryToSymbolicSQL, HoldRest]
DBEntityQueryToSymbolicSQL[es_, query_] :=
	DBQueryToSymbolicSQL @ DBEntityQueryToQueryBuilderObject[es, query]

DBEntityQueryToSymbolicSQL[es_] :=
	Function[query, DBEntityQueryToSymbolicSQL[es, query], HoldAll]
    
    
    
SetAttributes[getEntityStore, HoldFirst]  
getEntityStore["CoreElement", q_] := 
    Replace[
        FindEntityStore[EntityFramework`General`Private`baseType[q]],
        _Missing -> None
    ]
  
getEntityStore[(EntityValue | EntityList)[q_, ___]] := 
    getEntityStore["CoreElement", q]
    
getEntityStore[
    q: EntityClass[_String, __] | Entity[_String, _] | EntityProperty[_String, _]
] := getEntityStore["CoreElement", q]   
    
getEntityStore[q_] :=
    With[{queryElem = FirstCase[
            Unevaluated[q], 
            EntityClass[_String, __] | Entity[_String, _] | EntityProperty[_String, _],
            Heads -> True
        ]},
        getEntityStore[queryElem] /; !MissingQ[queryElem]
    ]
    
getEntityStore[_] := None    
    
    
SetAttributes[getDatabaseQuery, HoldFirst]    
getDatabaseQuery[q_] :=
    Module[{tag, inToDatabaseQuery = False},
        Internal`InheritedBlock[{DBToDatabaseQuery},
            PrependTo[
                DownValues[DBToDatabaseQuery],
                HoldPattern[
                    (call:DBToDatabaseQuery[___]) /; !TrueQ[inToDatabaseQuery]
                ] :> Block[{inToDatabaseQuery = True},
                    Throw[call, tag]
                ]
            ];
            Catch[q, tag]
        ]
    ]
    
    
SetAttributes[DBEntityQueryToSQLString, HoldFirst]  
DBEntityQueryToSQLString::nostore = "Unable to detect EntityStore for the query";
DBEntityQueryToSQLString::invstore = "EntityStore for the query is not database-backed";  
DBEntityQueryToSQLString[q_] :=
    With[{es = getEntityStore[q]},
        If[es === None, 
            Message[DBEntityQueryToSQLString::nostore];
            Return[$Failed]
        ];
        Replace[
            es, 
            {
                EntityStore[_, schema_RelationalDatabase] :> 
                    DatabaseRunQuery[
                        getDatabaseQuery[q], schema, "SQLString"
                    ],
                _ :> (
                    Message[DBEntityQueryToSQLString::invstore];
                    $Failed
                )
            }
        ]
    ]
