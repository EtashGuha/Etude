Package["Databases`Entity`"]

PackageImport["Databases`"]
PackageImport["Databases`Schema`"]
PackageImport["Databases`Common`"]

PackageScope["makeEntityStoreBoxes"]

$entityStoreIcon = BinaryDeserialize[
	ReadByteArray @ FileNameJoin @ {
		PacletManager`PacletResource["Databases", "Data"],
		"EntityStoreIcon.wxf"
	}
]


makeEntityStoreBoxes[es: EntityStore[meta_, schema : _RelationalDatabase], fmt_] := With[
	{dbs = Catch[DatabaseStore[es], DBError]},
	If[
		FailureQ[dbs],
		$Failed,
		BoxForm`ArrangeSummaryBox[
			EntityStore,
			es,
			$entityStoreIcon,
			{
				BoxForm`MakeSummaryItem[
					{"Type count: ", Length[dbs["Models"]]},
					fmt
				],
				BoxForm`MakeSummaryItem[
					{
						"RelationalDatabase: ",
						OpenerView[{schema["Connection"]["Backend"], dbs["Schema"]}]
					},
					fmt
				]
			},
			{
				Grid[
					{{
						RawBoxes @ TagBox[MakeBoxes["Types: ", StandardForm], "SummaryItemAnnotation"],
						Column[Map[
							Function[
								model,
								OpenerView[{
									If[
										dbs[model, "ConcreteModelQ"],
										model,
										Style[model, Italic]
									],
									Column[Map[
										Function[
											field,
											OpenerView[{
												Which[
													MemberQ[dbs[model, "CanonicalFields"], field],
													Style[#, Bold]&,
													dbs[model, field, "FieldType"] === "Function",
													Style[#, Italic]&,
													dbs[model, field, "FieldType"] === "Relation",
													Row[{
														If[
															dbs[model, field, "RelationType"] === "OneToMany",
															"\[LeftArrow] ",
															"\[RightArrow] "
														],
														Style[#, Underlined]
													}]&,
													True,
													Identity
												][field],
												Grid[
													MapAt[
														Style[# <> ":", Gray]&,
														Prepend[
															KeyValueMap[
																{
																	Replace[
																		#,
																		Reverse[$mappings[[-1, "Keys"]]]
																	],
																	Replace[
																		#2,
																		First[$mappings[[-1, "Values"]]]
																	]
																}&,
																dbs[model, field, "FieldExtractor"]
															],
															{"PropertySourceType", dbs[model, field, "FieldType"]}
														],
														{All, 1}
													],
													Alignment -> Left
												]
											}]
										],
										dbs[model, "Fields"]
									]]
								}]
							],
							dbs["Models"]
						]]
					}},
					Alignment -> Top
				]
			},
			fmt
		]
	]
]