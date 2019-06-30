(* Wolfram Language package *)

Package["Databases`Schema`"]

(*
    RelationalDatabase, DatabaseReference
*)
PackageImport["Databases`"]
PackageImport["Databases`Common`"]

makeBoxIcon[text_String, color_: Blue] :=
    Graphics[{
        color,
        Disk[],
        Text[
            Style[text, Directive[13 - StringLength[text], White, Bold]],
            Scaled[{.53, .48}]
        ]},
        ImageSize -> Dynamic[{
            Automatic,
            (3 * CurrentValue["FontCapHeight"]) / AbsoluteCurrentValue[Magnification]
        }],
        Background -> None
    ]
makeBoxIcon[text_, rest___] := makeBoxIcon[ToString[text, rest]]

$databaseConnectionIcon = BinaryDeserialize[
	ReadByteArray @ FileNameJoin @ {
		PacletManager`PacletResource["Databases", "Data"],
		"DatabaseReferenceIcon.wxf"
	}
]

makeSummary[a_Association?AssociationQ] := KeyValueMap[makeSummary, a]

makeSummary[key_, val_] := BoxForm`MakeSummaryItem[{key <> ": ", val}, StandardForm]

DatabaseReference /: MakeBoxes[
    conn: DatabaseReference[meta_Association?AssociationQ, ___],
    StandardForm
    ] :=
    With[
        {elements = ReplacePart[
			DeleteCases[conn[All], None],
			Key["Password"] -> "********"
		]},
        BoxForm`ArrangeSummaryBox[
            DatabaseReference,
            conn,
			$databaseConnectionIcon,
			makeSummary @ KeyTake[elements, {"Name", "Backend"}],
			makeSummary @ KeyDrop[elements, {"Name", "Backend"}],
            StandardForm
        ]
    ]


$SQLDatabaseObjectIcon = BinaryDeserialize[
	ReadByteArray @ FileNameJoin @ {
		PacletManager`PacletResource["Databases", "Data"],
		"RelationalDatabaseIcon.wxf"
	}
]

RelationalDatabase /: MakeBoxes[
	db: RelationalDatabase[expr_Association?AssociationQ, conn_DatabaseReference: None],
	StandardForm
] /; IntegerString[Hash[{KeyDrop[expr, "ID"], conn}], 36] === expr["ID"] :=
	BoxForm`ArrangeSummaryBox[
		RelationalDatabase,
		db,
		$SQLDatabaseObjectIcon,
		{
			BoxForm`MakeSummaryItem[{"Table count: ", Length[db["Tables"]]}, StandardForm],
			BoxForm`MakeSummaryItem[{"Backend: ", If[conn =!= None, conn["Backend"], None]}, StandardForm]
		},
        Replace[
            db["Tables"], {
                {} :> {},
                s_List :> {Grid[{{
                    RawBoxes @ TagBox[MakeBoxes["Tables: ", StandardForm], "SummaryItemAnnotation"],
                    Column @ Map[
                        Function[
                            table,
                            OpenerView @ {table, Column @ Map[
                                Function[
                                    column,
                                    OpenerView @ {
                                        column,
                                        Grid[
                                            KeyValueMap[
                                                {Style[# <> ":", Gray], #2}&,
                                                db[table, column, All]
                                            ],
                                            Alignment -> Left
                                        ]
                                    }
                                ],
                                db[table, "Columns"]
                            ]}
                        ],
                        db["Tables"]
                    ]
                }}, Alignment -> Top]}
            }
        ],
		StandardForm
	]



$horizontalBoxes = False


DBType /: MakeBoxes[
	t: DBType[_RepeatingElement, ___]?(DBUnevaluatedPatternCheck[DBTypeQ]),
	StandardForm
] :=
    InterpretationBox @@ {
        StyleBox[
            FrameBox[
                Block[{$horizontalBoxes = !$horizontalBoxes},
                    If[
                        $horizontalBoxes,
                        GridBox[
                            {{ToBoxes[First[t["Constituents"]]], "\[Ellipsis]"}},
                            RowAlignments -> Center
                        ],
                        GridBox[
                            {{ToBoxes[First[t["Constituents"]]]}, {"\[VerticalEllipsis]"}},
                            ColumnAlignments -> Center
                        ]
                    ]
                ],
                FrameStyle -> Lighter[Brown, 0.2],
                StripOnInput -> False
            ],
			FontFamily->"Avenir Next Condensed",
			FontColor-> Lighter[Brown, 0.2],
			FontSize->12,
			ShowStringCharacters -> False
		],
        t
    }

DBType /: MakeBoxes[
	t: DBType[_CompoundElement, ___]?(DBUnevaluatedPatternCheck[DBTypeQ]),
	StandardForm
] :=
	With[{assocq = AssociationQ[t[[1, 1]]]},
	With[{delims = If[assocq, {"\[LeftAssociation]", "\[RightAssociation]"}, {"{", "}"}]},
		InterpretationBox @@ {
			StyleBox[
				FrameBox[
					Block[{$horizontalBoxes = !$horizontalBoxes},
						If[
							$horizontalBoxes,
							RowBox[
								Join[
									{StyleBox[delims[[1]], Bold, StripOnInput -> False]},
									Riffle[
										If[
											assocq,
											KeyValueMap[
												RowBox[{ToBoxes[#1], "\[Rule]", ToBoxes[#2]}]&,
												t[[1, 1]]
											],
											ToBoxes /@ t["Constituents"]
										],
										StyleBox[",", Bold, StripOnInput -> False]
									],
									{StyleBox[delims[[2]], Bold, StripOnInput -> False]}
								]
							],
							GridBox[
								PadRight[#, Automatic, "\[SpanFromLeft]"]& @ Join[
									{{RotationBox[
										StyleBox[delims[[1]], Bold, StripOnInput -> False],
										BoxRotation -> -Pi/2
									]}},
									Riffle[
										If[
											assocq,
											KeyValueMap[
												{ToBoxes[#1], "\[Rule]", ToBoxes[#2]}&,
												t[[1, 1]]
											],
											List @* ToBoxes /@ t["Constituents"]
										],
										{{RotationBox[
											StyleBox[",", Bold, StripOnInput -> False],
											BoxRotation -> -Pi/2
										]}}
									],
									{{RotationBox[
										StyleBox[delims[[2]], Bold, StripOnInput -> False],
										BoxRotation -> -Pi/2
									]}}
								],
								ColumnAlignments -> Center
							]
						]
					],
					FrameStyle -> Lighter[Brown, 0.2],
					StripOnInput -> False
				],
				FontFamily->"Avenir Next Condensed",
				FontColor-> Lighter[Brown, 0.2],
				FontSize->12,
				ShowStringCharacters -> False
			],
			t
		}
	]
	]


DBType /: MakeBoxes[
    t: DBType[_RectangularRepeatingElement | _SquareRepeatingElement, ___]?(
		DBUnevaluatedPatternCheck[DBTypeQ]
	),
    StandardForm
] :=
	InterpretationBox @@ {
		StyleBox[
            FrameBox[
                FrameBox[
					GridBox[
						{{ToBoxes[First[t["Constituents"]]], ""}, {"", "\[DescendingEllipsis]"}},
						RowAlignments -> Center,
                        ColumnAlignments -> Center
					],
                    FrameStyle -> Lighter[Brown, 0.2],
					RoundingRadius -> If[Head[First[t]] === SquareRepeatingElement, 5, 0],
                    StripOnInput -> False
                ],
				FrameStyle -> Lighter[Brown, 0.2],
                RoundingRadius -> If[Head[First[t]] === SquareRepeatingElement, 5, 0],
				StripOnInput -> False
			],
			FontFamily->"Avenir Next Condensed",
			FontColor-> Lighter[Brown, 0.2],
			FontSize->12,
			ShowStringCharacters -> False
		],
		t
	}

DBType /: MakeBoxes[t_DBType?(DBUnevaluatedPatternCheck[DBTypeQ]), StandardForm] :=
	InterpretationBox @@ {
		If[
			Length[t] > 1,
			TooltipBox[
				#,
				ToBoxes[Grid[List @@@ Normal[t[[2]]], Alignment -> {{Left, Right}}]]
			]&,
			Identity
		] @ StyleBox[
				RowBox[{"[", ToBoxes[t["Type"]],"]"}],
				FontFamily->"Avenir Next Condensed",
				FontColor-> Lighter[Brown, 0.2],
				FontSize->12,
				ShowStringCharacters -> False
			],
		t
	}
DBTypeUnion /: MakeBoxes[
	t: DBTypeUnion[types__DBType?(DBUnevaluatedPatternCheck[DBTypeQ])],
	StandardForm
] :=
    InterpretationBox @@ {
        RowBox[Riffle[ToBoxes /@ {types}, "\[Union]"]],
        t
    }
DBTypeIntersection /: MakeBoxes[
	t: DBTypeIntersection[types__DBType?(DBUnevaluatedPatternCheck[DBTypeQ])],
	StandardForm
] :=
	InterpretationBox @@ {
		RowBox[Riffle[ToBoxes /@ {types}, "\[Intersection]"]],
			t
	}
DBTypeUnion /: MakeBoxes[t: DBTypeUnion[], StandardForm] :=
	InterpretationBox @@ {
		"\[EmptySet]",
		t
	}
