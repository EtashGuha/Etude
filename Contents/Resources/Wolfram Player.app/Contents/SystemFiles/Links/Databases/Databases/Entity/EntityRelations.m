Package["Databases`Entity`"]

PackageImport["Databases`"]
PackageImport["Databases`Database`"]

PackageScope["EntityRelationFunction"]

EntityRelationFunction[args___][compilationContext_][left_] :=
    EntityRelationFunction[compilationContext["Store"], args][left]

EntityRelationFunction[store_DatabaseStore, spec_?AssociationQ, "ManyToOne"][left_] := With[
	{list = Map[
		left[#]&,
		DBCompileJoinRules[store, left, spec["DestinationModel"], spec["ModelMapping"]][[All, 1]]
	]},
	If[
		AnyTrue[list, MissingQ],
		Missing["NotAvailable"],
		Entity[
			Replace[
				spec["DestinationModel"],
				dbq: Except[_?StringQ] :>
					DBRaise[EntityRelationFunction, "single_entity_with_complex_query", {dbq}]
			],
			Replace[
				list,
				{a_} :> a
			]
		]
	]
]

EntityRelationFunction[store_DatabaseStore, spec_?AssociationQ, type_][left_] := If[
	MatchQ[spec["ModelMapping"], _DatabaseFunction],
	FilteredEntityClass,
	EntityClass
][
	Replace[
		spec["DestinationModel"],
		dbq: Except[_?StringQ] :>
			Lookup[
				$CompiledEntityQueries,
				dbq,
				DBRaise[EntityRelationFunction, "no_conversion_known", {dbq}]
			]
	],
	Replace[
		spec["ModelMapping"],
		{
			dbf_DatabaseFunction :> Lookup[
				$CompiledEntityFunctions,
				dbf,
				DBRaise[EntityRelationFunction, "no_conversion_known", {dbf}]
			],
			any_ :> MapAt[
				left[#]&,
				Reverse[DBCompileJoinRules[store, left, spec["DestinationModel"], any], 2],
				{All, 2}
			]
		}
	]
]