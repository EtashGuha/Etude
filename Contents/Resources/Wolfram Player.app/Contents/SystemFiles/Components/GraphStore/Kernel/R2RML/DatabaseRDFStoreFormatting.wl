BeginPackage["GraphStore`R2RML`DatabaseRDFStoreFormatting`", {"GraphStore`", "GraphStore`R2RML`"}];
Begin["`Private`"];

DatabaseRDFStore /: MakeBoxes[store_DatabaseRDFStore, fmt_] := With[{res = Catch[iDatabaseRDFStoreMakeBoxes[store, fmt], $failTag]}, res /; res =!= $failTag];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


clear[iDatabaseRDFStoreMakeBoxes];
iDatabaseRDFStoreMakeBoxes[store_DatabaseRDFStore, fmt_] := BoxForm`ArrangeSummaryBox[
	DatabaseRDFStore,
	store,
	None,
	{
		BoxForm`MakeSummaryItem[
			{
				"Triples map count: ",
				Length[store["Mapping"]]
			},
			fmt
		],
		BoxForm`MakeSummaryItem[
			{
				"Database: ",
				OpenerView[{
					If[StringQ[store["Database"]],
						"SQLite",
						fail[]
					],
					store["Database"]
				}]
			},
			fmt
		]
	},
	{
		Grid[
			{{
				RawBoxes[TagBox[MakeBoxes["Triples maps: ", fmt], "SummaryItemAnnotation"]],
				Column[MapIndexed[
					Function[{tm, tmpos},
						OpenerView[{
							formatNode[Lookup[tm, "Node", tmpos]],
							Column[{
								OpenerView[{
									"logical table",
									Lookup[tm, "LogicalTable", fail[]]
								}],
								OpenerView[{
									"subject map",
									Lookup[tm, "SubjectMap", fail[]]
								}],
								OpenerView[{
									"predicate-object maps",
									Column[MapIndexed[
										Function[{pom, pompos},
											OpenerView[{
												formatNode[Lookup[pom, "Node", pompos]],
												Column[{
													OpenerView[{
														"predicate maps",
														Column[Lookup[pom, "PredicateMaps", fail[]]]
													}],
													OpenerView[{
														"object maps",
														Column[Lookup[pom, "ObjectMaps", fail[]]]
													}],
													OpenerView[{
														"graph maps",
														Column[Lookup[pom, "GraphMaps", {}]]
													}]
												}]
											}]
										],
										Lookup[tm, "PredicateObjectMaps", {}]
									]]
								}]
							}]
						}]
					],
					store["Mapping"]
				]] /. Column[{}] :> Style["none", Gray]
			}},
			Alignment -> Top
		]
	},
	fmt
];

clear[formatNode];
formatNode[RDFBlankNode[b_String]] := "_:" <> b;
formatNode[IRI[i_String]] := i;
formatNode[{pos_Integer}] := pos;

End[];
EndPackage[];
