Package["Databases`Entity`"]


PackageImport["Databases`"]
PackageImport["Databases`Common`"]
PackageImport["Databases`Database`"]
PackageImport["Databases`SQL`"]


PackageScope["$errorHandler"]


EntityFunction::nonscal = "EntityFunction can only return a scalar."
EntityFunction::invexpr = "The expression `1` cannot be compiled to SQL."
EntityFunction::invprop = "Unknown property `1` in EntityFunction."
EntityFunction::invprop2 = "Cannot extract property `1` from something that is not an entity or an entity class."
EntityFunction::type = "EntityFunction has return type `1` instead of `2`."
EntityFunction::invtype = "Invalid types in EntityFunction. Cannot compile `1` with arguments of type `2` in expression `3`."
EntityFunction::lists = "Attempting listable operation over lists that might not have the same length in expression `1`."
EntityFunction::aggscalar = "Attempting aggregation over a scalar in `1`."
EntityFunction::manymany = "Attempting to follow multiple entity-class-valued properties in `1`."
EntityFunction::nonlist = "AllTrue and AnyTrue expect a list or an entity class as the first argument."
EntityFunction::nonclass = "`1` is not a valid entity class."
CombinedEntityClass::alias = "Cannot combine two identical classes. use the form \"alias\" -> class to disambiguate."
SortedEntityClass::seqs = "The sequence specification `` in SortedEntityClass is either invalid or not supported on this backend."
SampledEntityClass::seqs = "The sequence specification `` in SampledEntityClass is either invalid or not supported on this backend."
CombinedEntityClass::manyrel = "Multiple relations were found between the two classes. Specify the third argument of CombinedEntityClass."
CombinedEntityClass::norel = "No relations were found between the two classes. Specify the third argument of CombinedEntityClass."
CombinedEntityClass::badjoincond = "The third argument of CombinedEntityClass should be an EntityFunction, an EntityProperty, a list of properties or a list of rules."
CombinedEntityClass::invpropcombined = "Property `` not found in ``."
EntityValue::dberror = "Generic database error."
RelationalDatabase::dberror = "Generic database error."
EntityClass::invprop = "Invalid properties in implicit EntityClass."
RandomEntity::unsupported = "RandomEntity is not supported on this backend. Returning the first `` entities."
RandomEntity::unsupported1 = "RandomEntity is not supported on this backend. Returning the first entity."

(*if assocs are present in a held expression the function below turns them into their non AssociationQ form*)

inactivateAssocs[body_] := ReplaceAll[
	ReplaceAll[
		body,
		Association -> fakeAssoc
	],
	fakeAssoc -> Association
]

normalizeExpr[h_] := ReplaceAll[
	ReplaceAll[
		ReplaceAll[
			inactivateAssocs[h],
			DBSQLSlot[sym_] :> sym
		],
		inactivateAssocs[Normal[$CompiledEntityQueries]]
	],
	{
		DBSQLSlot[sym_] :> sym,
		DatabaseFunction -> EntityFunction,
		db_DBPrefixedField :> RuleCondition[toEntityProperty[db]],
		dbq_?DBUncompiledQueryQ :> RuleCondition[
			Lookup[$CompiledEntityQueries, dbq, DBRaise[EntityStore, "no_conversion_known", {dbq}]]
		]
	}
]

SetAttributes[buildFailureAndEmitMessage, HoldRest]

buildFailureAndEmitMessage[
	originalFailure_,  msgName_, msgParams_: {}, extraParams_: <||>
] := (
	Message[msgName, Sequence @@ msgParams];
	Failure[
		"DatabaseFailure",
		KeyDrop[
			<|
				"MessageTemplate" -> msgName,
				"MessageParameters" -> msgParams,
				If[
					TrueQ[$DBDebugMode],
					"OriginalFailure" -> originalFailure,
					{}
				],
				extraParams
			|>,
			{"FailingFunctionArgs", "FailingFunction"}
		]
	]
)

$errorHandler = Function[
	{func, failure},
	Replace[
		failure,
		{
			f : Failure["invalid_implicit_entityclass", assoc_] :>
				buildFailureAndEmitMessage[
					f,
					EntityClass::invprop,
					{},
					"FailingProperties" -> assoc[["FailingFunctionArgs", 1]]
				],
			f : Failure["cannot_annotate_with_a_nonscalar", _] :>
				buildFailureAndEmitMessage[
					f,
					EntityFunction::nonscal
				],
			f : Failure["unknown_element", assoc_] :> With[
				{failedExpr = normalizeExpr[assoc[["FailingFunctionArgs", 1]]]},
				buildFailureAndEmitMessage[
					f,
					EntityFunction::invexpr,
					{failedExpr},
					"FailingExpression" -> failedExpr
				]
			],
			f : Failure["type_mismatch", assoc_] :>
				buildFailureAndEmitMessage[
					f,
					EntityFunction::type,
					{assoc["PassedType"]["Type"], assoc["ExpectedType"]["Type"]},
					<|
						"PassedType" -> assoc["PassedType"]["Type"],
						"ExpectedType" -> assoc["ExpectedType"]["Type"]
					|>
				],
			f : Failure["no_field_found", assoc_] :> With[
				{invProp = normalizeExpr[assoc[["FailingFunctionArgs", 1]]]},
				buildFailureAndEmitMessage[
					f,
					EntityFunction::invprop,
					{invProp},
					"MissingProperty" -> invProp
				]
			],
			f : Failure["no_field_found_join", assoc_] :> With[
				{invProp = normalizeExpr[assoc[["FailingFunctionArgs", 2]]],
				query = invProp = normalizeExpr[assoc[["FailingFunctionArgs", 1]]]},
				buildFailureAndEmitMessage[
					f,
					CombinedEntityClass::invpropcombined,
					{invProp, query},
					<|"MissingProperty" -> invProp, "Query" -> query|>
				]
			],
			f : Failure["field_extraction_from_scalar", assoc_] :> With[
				{invProp = normalizeExpr[assoc[["FailingFunctionArgs", 1]]]},
				buildFailureAndEmitMessage[
					f,
					EntityFunction::invprop2,
					{invProp},
					"MissingProperty" -> invProp
				]
			],
			f : Failure["ambiguous_aliases_in_queries_for_join", _] :>
				buildFailureAndEmitMessage[
					f,
					CombinedEntityClass::alias
				],
			f : Failure["more_than_one_relation_found", _] :> buildFailureAndEmitMessage[
				f,
				CombinedEntityClass::manyrel
			],
			f : Failure["invalid_join_spec", _] :> buildFailureAndEmitMessage[
				f,
				CombinedEntityClass::badjoincond
			],
			f : Failure["no_relation_found", _] :> buildFailureAndEmitMessage[
				f,
				CombinedEntityClass::norel
			],
			f : Failure["unknown_operator_or_type_signature", assoc_] :> With[{
				expr = normalizeExpr[assoc["FailingExpr"]]},
				buildFailureAndEmitMessage[
					f,
					EntityFunction::invtype,
					{
						Replace[expr, HoldForm[h_[___]] :> h],
						StringRiffle[#["Repr"]& /@ assoc["ArgumentsCompoundType"]["Constituents"], ", "],
						expr
					},
					<|
						"FailingExpression" -> expr,
						"FailingTypes" ->
							#["Repr"]& /@ assoc["ArgumentsCompoundType"]["Constituents"]
					|>
				]
			],
			f : Failure["attempting_aggregation_on_multiple_subqueries", assoc_] :> With[
				{failedExpr = normalizeExpr[assoc["FailingExpr"]]},
				buildFailureAndEmitMessage[
					f,
					EntityFunction::lists,
					{failedExpr},
					"FailingExpression" -> failedExpr
				]
			],
			f : Failure["attempting_aggregation_on_many_to_one_relation_chain", assoc_] :> With[
				{failedExpr = normalizeExpr[assoc["FailingExpr"]]},
				buildFailureAndEmitMessage[
					f,
					EntityFunction::aggscalar,
					{failedExpr},
					"FailingExpression" -> failedExpr
				]
			],
			f : Failure["attempting_chaining_of_multiple_one_to_many", assoc_] :> With[
				{failedExpr = normalizeExpr[assoc["FailingExpr"]]},
				buildFailureAndEmitMessage[
					f,
					EntityFunction::manymany,
					{failedExpr},
					"FailingExpression" -> failedExpr
				]
			],
			f : Failure["scalar_in_all_or_any", assoc_] :> With[
				{failedExpr = normalizeExpr[assoc["FailingExpr"]]},
				buildFailureAndEmitMessage[
					f,
					EntityFunction::nonlist,
					{failedExpr},
					"FailingExpression" -> failedExpr
				]
			],
			f: Failure["invalid_class_in_canonicalProperties", assoc_] :> With[
				{failedExpr = assoc["FailingFunctionArgs"][[2]]},
				buildFailureAndEmitMessage[
					f,
					EntityFunction::nonclass,
					{failedExpr},
					"FailingExpression" -> failedExpr
				]
			],
			f: Failure["invalid_take_spec_in_SortedEntityClass", assoc_] :> With[
				{failedExpr = assoc["FailingFunctionArgs"][[1]]},
				buildFailureAndEmitMessage[
					f,
					SortedEntityClass::seqs,
					{failedExpr},
					"FailingExpression" -> failedExpr
				]
			],
			f: Failure["invalid_take_spec_in_SampledEntityClass", assoc_] :> With[
				{failedExpr = assoc["FailingFunctionArgs"][[1]]},
				buildFailureAndEmitMessage[
					f,
					SampledEntityClass::seqs,
					{failedExpr},
					"FailingExpression" -> failedExpr
				]
			],
			f : Failure["DatabaseFailure", _] :> (
				Message[EntityValue::dberror];
				f
			),
			f_ :> buildFailureAndEmitMessage[
				f,
				RelationalDatabase::dberror
			]
		}
	]
]
