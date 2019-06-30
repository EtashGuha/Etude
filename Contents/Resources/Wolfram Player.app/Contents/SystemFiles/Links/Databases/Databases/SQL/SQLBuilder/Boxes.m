Package["Databases`SQL`"]

PackageImport["Databases`"]
PackageImport["Databases`Common`"]


DBRawFieldName /: MakeBoxes[e : DBRawFieldName[name_, _, type_], StandardForm] :=
    With[{colorize = Style[
            #,
            FontColor -> Darker[Green, 0.5], FontWeight -> Bold, FontFamily -> "Bookman",
            ShowStringCharacters -> False
        ] &
        },
        InterpretationBox[#, e] & @ MakeBoxes[#, StandardForm] & @ Tooltip[
            Row[{
                colorize["["], colorize["["],
                name,
                Spacer[3],
                Style[
                    "\[RightArrow]",
                    FontWeight -> Plain, FontColor->Gray, FontFamily->"Verdana"
                ],
                Spacer[3],
                type,
                colorize["]"], colorize["]"]
            }],
            RawBoxes[RowBox[{"DBRawFieldName", "[",
				RowBox[
					Riffle[MakeBoxes[#, StandardForm]& /@ Apply[List, e], ","]
				],
				"]"
			}]]
        ]
    ]


DBPrefixedField /: MakeBoxes[e : DBPrefixedField[field_], StandardForm] :=
    With[{colorize = Style[
            #,
            FontColor -> Orange, FontWeight -> Bold, FontFamily -> "Bookman",
            ShowStringCharacters -> False
        ] &
        },
        InterpretationBox[#, e] & @ MakeBoxes[#, StandardForm] & @
            Row[{
                Row[{colorize["["], colorize["["]}],
                field,
                Row[{colorize["]"], colorize["]"]}]
            }," "]
    ]



DBQueryBuilderObject /: MakeBoxes[
    q: DBQueryBuilderObject[info_]?(
        Function[qo, DBQueryBuilderObjectQ[Unevaluated @ qo], HoldAllComplete]
    ),
    StandardForm
] :=
If[TrueQ[$DBInErrorHandler],
    BoxForm`ArrangeSummaryBox[DBQueryBuilderObject, q, None, {}, {}, StandardForm ],
    (* else *)
    BoxForm`ArrangeSummaryBox[
        DBQueryBuilderObject,
        q,
        None,
        {
            BoxForm`MakeSummaryItem[{"Name: ", q @ DBGetName[]}, StandardForm],
            If[q @ getType[] === "NativeTable",
                BoxForm`MakeSummaryItem[
                    {
                        "Table name:  " ,
                        Style[q @ get["TableName"], FontFamily-> "Monaco", FontWeight -> Bold]
                    }, StandardForm
                ],
                (* else *)
                Sequence @@ {}
            ],
            BoxForm`MakeSummaryItem[{
                    "Type: ", 
                    If[TrueQ[q @ DBIsSingleRow[]], "Single row", "Query"]
                }, StandardForm
            ],
            BoxForm`MakeSummaryItem[
                {
                    "Fields: ",
                    Pane[
                        Style[
                            Row[
                                If[q @ getType[] === "NativeTable",
                                    q @ getSelectedFieldNames[],
                                    (* else *)
                                    q @ DBGetPrefixedSelectedFields[]
                                ],
                                ","
                            ],
                            FontSize -> 9, FontFamily-> "Monaco"
                        ],
                        ImageSize -> {250, Automatic}, Scrollbars -> {False, False}
                    ]
                }, StandardForm]
        },
        {
            If[q @ getProperFields[] =!= {},
                BoxForm`MakeSummaryItem[
                    {
                        "Annotations:  ",
                        Style[
                            Column @ Reverse[q @ getProperFields[], {2}],
                            FontFamily-> "Monaco", FontSize -> 10
                        ]
                    },
                    StandardForm
                ],
                (* else *)
                Sequence @@ {}
            ],
            If[Length[q @ getInnerFields[]] > 0,
                BoxForm`MakeSummaryItem[
                    {
                        "Inner fields: ",
                        Style[
                            Grid[Transpose @ {
                                Map[Style[#, FontWeight -> Bold]&] @ Keys @ #,
                                Map[Style[Row[#, ","], FontFamily-> "Monaco"]&]@ Values @ #
                            }, Alignment -> Left],
                            FontSize -> 8
                        ]& @ q @ getInnerFields[]
                    }, StandardForm
                ],
                (* else *)
                Sequence @@ {}
            ],
            If[q @ getTables[] =!= {},
                BoxForm`MakeSummaryItem[
                    {
                        "Tables: ",
                        OpenerView[{
                            Style[
                                Row[#, " , "],
                                FontWeight -> Bold
                            ] & @ Map[# @ DBGetName[]&] @ q @ getTables[],
                            Column @ q @ getTables[]
                        }]
                    }, StandardForm
                ],
                (* else *)
                Sequence @@ {}
            ],
            If[q @ get["Joins"] =!= {},
                BoxForm`MakeSummaryItem[
                    {
                        "Joins:  ",
                        Column @ Map[Rule[
                            Row[Map[Style[#, FontWeight -> Bold]&] @ #["JoinedTables"],"   "],
                            {
                                Style[#, FontWeight -> Bold, FontSlant -> Italic]& @ #["JoinType"],
                                Style[#["On"], FontFamily-> "Monaco"]
                            }
                        ]&] @ q @ get["Joins"]
                    }, StandardForm
                ],
                (* else *)
                Sequence @@ {}
            ],
            If[q @ get["Where"] =!={},
                BoxForm`MakeSummaryItem[
                    {
                        "Where: ",
                        Style[Apply[And] @ q @ get["Where"], FontFamily-> "Monaco", FontSize -> 10]
                    },
                    StandardForm
                ],
                (* else *)
                Sequence @@ {}
            ],
            If[q @ get["GroupBy"] =!= None,
                BoxForm`MakeSummaryItem[
                    {
                        "GroupBy:  ",
                        Style[
                            Row[stripFullNames @ (q @ get["GroupBy"])["Fields"], " , "],
                            FontFamily-> "Monaco", FontSize -> 10
                        ]
                    },
                    StandardForm
                ],
                (* else *)
                Sequence @@ {}
            ],
            If[q @ get["OrderBy"] =!= None,
                BoxForm`MakeSummaryItem[
                    {
                        "OrderBy:  ",
                        Style[
                            Row[
                                Replace[
                                    stripFullNames @ (q @ get["OrderBy"]),
                                    (f_ -> flag_) :> (f -> If[flag, "ASC", "DESC"]),
                                    {1}
                                ],
                                " , "],
                            FontFamily-> "Monaco", FontSize -> 10
                        ]
                    },
                    StandardForm
                ],
                (* else *)
                Sequence @@ {}
            ],
			If[q @ get["Limit"] =!= None,
				BoxForm`MakeSummaryItem[
					{
						"Limit:  ",
						q @ get["Limit"]
					},
					StandardForm
				],
			(* else *)
				Sequence @@ {}
			],
			If[q @ get["Offset"] =!= None,
				BoxForm`MakeSummaryItem[
					{
						"Offset:  ",
						q @ get["Offset"]
					},
					StandardForm
				],
			(* else *)
				Sequence @@ {}
			]
        },
        StandardForm
   ]
 ]

$displayQueryFieldMapping[mapping_Association?AssociationQ] :=
    With[{rules = Normal @ mapping},
        If[rules === {},
            {},
            (* else *)
            OpenerView[{
                Tooltip[
                    Framed[
                        Row[{"{ ", First @ rules, ", ... }"}],
                        FrameStyle -> LightGray
                    ],
                    Column @ rules
                ],
                Column @ rules
            }]
        ]
    ]

DBQueryObject /: MakeBoxes[q_DBQueryObject, StandardForm] /; DBQueryObjectQ[Unevaluated @ q] :=
    BoxForm`ArrangeSummaryBox[
        DBQueryObject,
        q,
        None,
        {
            BoxForm`MakeSummaryItem[
                {
                    "Symbolic SQL: ",
                    DBDisplaySymbolicSQL[q["SymbolicSQL"], True]
                },
                StandardForm
            ]
            ,
            BoxForm`MakeSummaryItem[
                {
                    "Field mapping: ",
                    $displayQueryFieldMapping @ DBKeyValueReverse @ KeyMap[
                        q["FieldMap"],
                        q ["RawToPrefixedFieldsMap"]
                    ]
                },
                StandardForm
            ]
        },
        {
            BoxForm`MakeSummaryItem[
                {"Query builder instance:  ", q["QueryBuilderInstance"]},
                StandardForm
            ]
        },
        StandardForm
    ]
