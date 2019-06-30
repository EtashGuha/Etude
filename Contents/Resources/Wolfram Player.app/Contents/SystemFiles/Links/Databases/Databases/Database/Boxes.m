(* Wolfram Language package *)

Package["Databases`Database`"]

PackageImport["Databases`"] (* DatabaseStore *)
PackageImport["Databases`SQL`"] (* DBQueryObjectQ *)

DBResultSet /: MakeBoxes[
    set : DBResultSet[_Association],
    StandardForm
    ]:=
    BoxForm`ArrangeSummaryBox[
        DBResultSet,
        set,
        None,
        {
            BoxForm`MakeSummaryItem[
                {"Columns: ", showColumns[set["ColumnNames"]]},
                StandardForm
            ],
            BoxForm`MakeSummaryItem[
                {"Count: ", If[set["Lazy"], Indeterminate, set["RowCount"]]},
                StandardForm
            ]
        },
        {
            BoxForm`MakeSummaryItem[
                { "Details: ", showDetails @ set},
                StandardForm
            ]
        }
        ,
        StandardForm
    ]

showColumns[cols_List, limit_ : 4] /; Length[cols] <= limit :=
    Row @ Riffle[cols, ", "]

showColumns[cols_List, limit_ : 4] :=
    OpenerView[{
        Tooltip[
            Framed[
                Row[{showColumns[Take[cols, limit]],  ", ... "}],
                FrameStyle -> LightGray
            ],
            Column @ cols
        ],
        Column @ cols
    }]

showDetails[set : DBResultSet[query_, ___]]:=
    Replace[
        Normal @ Map[pane[400, 200]] @ Association @ {
            "Query" -> Replace[query["Query"],
                q_?DBSymbolicSQLQueryQ :> DBDisplaySymbolicSQL[q]
            ]
            ,
            If[
                set["Lazy"],
                {},
                {
                    "SQL" -> formatSQL @ set["SQLString"],
                    "Result" -> set["Dataset"]
                }
            ]
        }
        ,
        {
            {content_} :> content,
            content_List :> TabView[content, Length[content]]
        }
    ]

pane[maxWidth_, maxHeight_] := Function[expr, Pane[
    expr,
    ImageSize -> {UpTo[maxWidth], UpTo[maxHeight]},
    Scrollbars -> True
]]

formatSQL[sql_String] /; TrueQ[DatabasesUtilities`$DatabasesUtilitiesAvailable] :=
    DatabasesUtilities`SQLFormat[sql]

formatSQL[sql_String] := sql

$databaseStoreIcon :=
	Graphics[{
		White,
		{Rectangle[{0, 0}, {6, 4}]},
		Black,
		Circle[{3.5, 3}, {1.5, 0.5}],
		Line[{{2, 3}, {2, 1}}],
		Line[{{5, 3}, {5, 1}}],
		Circle[{3.5, 2}, {1.5, 0.5}, {-Pi, 0}],
		Circle[{3.5, 1}, {1.5, 0.5}, {-Pi, 0}]
	},
		AspectRatio -> 1,
		Background -> GrayLevel[0.93],
		ImageSize -> {
			Automatic,
			Dynamic[3.5*(CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification])]
		},
		Frame -> {{True, True}, {True, True}},
		FrameStyle -> Directive[Thickness[Tiny], GrayLevel[0.9]],
		FrameTicks -> {{None, None}, {None, None}},
		GridLines -> {None, None}
	];

DatabaseStore /: MakeBoxes[dbs_DatabaseStore, StandardForm] /;
    System`Private`NoEntryQ[dbs] := BoxForm`ArrangeSummaryBox[
	DatabaseStore,
	dbs,
	$databaseStoreIcon,
    {
		BoxForm`MakeSummaryItem[
            {
                "RelationalDatabase: ",
                OpenerView[{dbs["Schema"]["Connection"]["Backend"], dbs["Schema"]}]
            },
            StandardForm
        ],
		BoxForm`MakeSummaryItem[
			{
				"Structure: ",
				OpenerView[{
					Row[{"Models (", Length[dbs["Models"]], ")"}],
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
														"\[LeftArrow]",
														"\[RightArrow]"
													],
													Spacer[.1],
													Style[#, Underlined]
												}]&,
												True,
												Identity
											][field],
											Grid[
												Prepend[
													KeyValueMap[
														List,
														dbs[model, field, "FieldExtractor"]
													],
													{"FieldType", dbs[model, field, "FieldType"]}
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
				}]
			},
			StandardForm
		]
	},
    {},
    StandardForm
]

DBRelationFunction /: MakeBoxes[dbr: DBRelationFunction[spec_?AssociationQ, type_][_], StandardForm] :=
	With[{inside = If[type === "OneToMany", "\[LeftArrow]", "\[RightArrow]"] <> spec["DestinationModel"]},
		InterpretationBox[
			RowBox[{"DBRelationFunction", "[", PanelBox[inside], "]"}],
			dbr
		]
	]