Package["Databases`SQL`"]

PackageImport["Databases`"]

PackageImport["Databases`Common`"] (* DBRaise *)

(* TODO: captialize exported heads *)
PackageExport["DBSQLDistinct"]
PackageExport["DBSQLField"]
PackageExport["DBSQLSymbol"]
PackageExport["DBSQLDate"]
PackageExport["DBSQLTime"]
PackageExport["DBSQLDateTime"]
PackageExport["DBSQLSecondsToTimeQuantity"]
PackageExport["DBSQLBoolean"]
PackageExport["DBSQLJoin"]
PackageExport["DBSQLSelect"]
PackageExport["DBSQLWhere"]
PackageExport["DBSQLGroupBy"]
PackageExport["DBSQLOrderBy"]
PackageExport["DBSQLDesc"]
PackageExport["DBSQLAsc"]
PackageExport["DBSQLOuterRef"]
PackageExport["DBSQLOffsetLimit"]

PackageExport["DBValidateSQLExpression"]
PackageExport["DBSymbolicSQLQueryQ"]
PackageExport["DBDisplaySymbolicSQL"]


(* TODO: add cases for SQLUpdate, SQLInsert etc, once primitives get implemented *)
DBSymbolicSQLQueryQ[_DBSQLSelect] := True
DBSymbolicSQLQueryQ[_] := False


$validASTExpressionCompositeTokens = {
	{DBSQLField, _String | _DBPrefixedField},
	{DBSQLField, _String, _String | _DBRawFieldName},
	{DBSQLSymbol, _String},
    {DBQueryBuilderObject, _Association?AssociationQ} (* Subqueries *)
}


$validASTExpressionCompositeTokenPattern = Alternatives @@ Replace[
	$validASTExpressionCompositeTokens,
	{head_, args___} :> head[args],
	{1}
]

$validASTExpressionTokenPattern = Alternatives[
	DBSQLDistinct,
    DBSQLDate,
    DBSQLTime,
    DBSQLDateTime,
	DBSQLSecondsToTimeQuantity,
	DBSQLBoolean,
    DBSQLAsc,
    DBSQLDesc,
	DBSQLOffsetLimit,
	_Integer,
    _String,
    _Real,
    _ByteArray,
    List,
    True,
    False,
    None
]


SetAttributes[myHold, HoldAll];

DBValidateSQLExpression[expr_, extraValidTokens_List: {}] :=
	Block[{check, $token},
		check[{}] := Null;
		check[invalidTokens_List] :=
			DBRaise[
				DBValidateSQLExpression,
				"invalid_tokens_found",
				{},
				<|"Expression" -> expr, "InvalidTokens" -> invalidTokens|>
			];
		check @ Cases[
			expr,
			inv: Except[
				$validASTExpressionCompositeTokenPattern,
				Apply[Alternatives, $validASTExpressionCompositeTokens[[All, 1]]][__]
			] :> HoldForm[inv],
			Infinity,
			Heads->True
		];
		check @ Complement[
			DeleteDuplicates @ Cases[
				ReplaceAll[
					myHold[expr],
					Alternatives[
						$validASTExpressionTokenPattern,
						$validASTExpressionCompositeTokenPattern,
						Sequence @@ extraValidTokens
					] -> $token
				],
				x_ :> HoldForm[x],
				{-1},
				Heads -> True
			],
			{HoldForm[$token], HoldForm[myHold]}
		];
		expr
	]


$highlightSymbolicSQL = ReplaceAll[
    {
        s_Symbol /; Context[s] === "Databases`SQL`" :>
            Style[s, FontColor -> Purple, FontFamily -> "Monaco"]
        ,
        str_String :> Style[str, FontColor -> RGBColor[0.5, 0., 0], FontFamily -> "Monaco"]
        ,
        num_?NumericQ :> Style[num, FontColor -> Darker[Blue, 0.5], FontFamily -> "Monaco"]
    }
]

$formatSymbolicSQL[expr_] :=
    If[TrueQ[DatabasesUtilities`$DatabasesUtilitiesAvailable],
        RawBoxes[DatabasesUtilities`Formatting`DBMakeFormattedBoxes[expr, 200]],
        (* else *)
        expr
    ]

DBDisplaySymbolicSQL[expr_, wrapInOpener_:False] :=
    With[{styled = Style[expr, FontFamily -> "Monaco", ShowStringCharacters -> True]},
        With[{formatted = $formatSymbolicSQL[$highlightSymbolicSQL[styled]]},
            If[!TrueQ[wrapInOpener],
                formatted,
                (* else *)
                OpenerView[{
                    Tooltip[
                        Framed[
                            Style[Row[{Head @ expr, "[...]"}], FontFamily -> "Monaco"],
                            FrameStyle -> LightGray
                        ],
                        Pane[formatted, ImageSize -> {600, Automatic}]
                    ],
                    Framed @ styled
                }]
            ]
        ]
    ]

(* TODO: fix evaluaiton leaks / interpretation issues with boxes below *)

(*
MakeBoxes[DBSQLSymbol[op1: "Any"|"All"][val1_, DBSQLSymbol[op2_], val2_], StandardForm] := RowBox[{MakeBoxes[val1], op2, op1, MakeBoxes[val2]}]
MakeBoxes[DBSQLSymbol[s_][arg_], StandardForm] := RowBox[{s, " ", MakeBoxes[arg]}]
MakeBoxes[DBSQLSymbol[s_][args__], StandardForm] := RowBox[Riffle[MakeBoxes /@ {args}, s]]
MakeBoxes[DBSQLSymbol[s_][args__], StandardForm] := RowBox[{s, "(", Sequence @@ Riffle[MakeBoxes /@ {args}, ", "], ")"}]
MakeBoxes[DBSQLSlot[s_], StandardForm] := StyleBox[ToString[Unevaluated[s]], Darker[Green]]
*)

(* MakeBoxes[SQLList[args___], StandardForm] := RowBox[{"(", Sequence @@ Riffle[MakeBoxes /@ {args}, ", "]	, ")"}] *)

(*
MakeBoxes[(DBSQLDateTime | DBSQLDate)[tz_, rest__], StandardForm] := ToBoxes[DateObject[{rest}, TimeZone -> tz]]
MakeBoxes[DBSQLTime[tz_, rest__], StandardForm] := ToBoxes[TimeObject[{rest}, TimeZone -> tz]]
*)
