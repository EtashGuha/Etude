Package["Databases`Database`"]

PackageImport["Databases`"]
PackageImport["Databases`Common`"] (*DBHandleError*)
PackageImport["Databases`Schema`"] (* Types *)
PackageImport["Databases`SQL`"]

$allAlternatives = Alternatives @@ DBType[All]["Constituents"];

DBRegisterFunctionSignatures::invsig = "Invalid signature: ``, ``."
DBHandleError[
	Function[
		{func, failure},
		Message[DBRegisterFunctionSignatures::invsig, failure[[2, "MessageParameters", "etype"]], failure[[2, "FailingFunctionArgs"]]]
	]
][
	DBRegisterFunctionSignatures[
		{
			op: "+" | "*",
			DBType[RepeatingElement[a: "Integer" | "Real" | "Decimal", {2, Infinity}]]
		} :> {Function[DBSQLSymbol[op][##]], DBType[a]}
	];

	DBRegisterFunctionSignatures[
		{
			op: "+" | "*",
			DBType[RepeatingElement[DBTypeUnion["Integer", "Real", "Decimal"], {2, Infinity}]]
		} :> {
			Function[DBSQLSymbol[op][##]],
			DBType["Decimal"]
		}
	];

	DBRegisterFunctionSignatures[
		{
			"+" ,
			DBType[RepeatingElement["TimeQuantity", {2, Infinity}]]
		} -> {
			Function[DBSQLSymbol["+"][##]],
			DBType["TimeQuantity"]
		}
	];

	DBRegisterFunctionSignatures[
		{
			"*" ,
			DBType[CompoundElement[{
				DBTypeUnion["Integer", "Real", "Decimal"],
				"TimeQuantity"
			}]
			]
		} -> {
			Function[DBSQLSymbol["*"][##]],
			DBType["TimeQuantity"]
		}
	];

	DBRegisterFunctionSignatures[
		{
			"*" ,
			DBType[CompoundElement[{
				"TimeQuantity",
				DBTypeUnion["Integer", "Real", "Decimal"]
			}]
			]
		} -> {
			Function[DBSQLSymbol["*"][##]],
			DBType["TimeQuantity"]
		}
	];

	DBRegisterFunctionSignatures[
		{
			"-",
			DBType[CompoundElement[{
				a: "Integer" | "Real" | "Decimal" | "TimeQuantity",
				a_
			}]]
		} :> {Function[DBSQLSymbol["-"][##]], DBType[a]}
	];

	DBRegisterFunctionSignatures[
		{
			"Abs",
			DBType[CompoundElement[{
				a: "Integer" | "Real" | "Decimal" | "TimeQuantity"
			}]]
		} :> {Function[DBSQLSymbol["Abs"][##]], DBType[a]}
	];

	DBRegisterFunctionSignatures[
		{
			"-",
			DBType[CompoundElement[{a: "Integer" | "Real" | "Decimal"}]]
		} :> {Function[DBSQLSymbol["-"][##]], DBType[a]}
	];

	DBRegisterFunctionSignatures[
		{
			"+",
			DBType[CompoundElement[{
				a: "Date" | "Time" | "DateTime",
				"TimeQuantity"
			}]]
		} :> {Function[DBSQLSymbol["DateAdd"][##]], DBType[a]}
	];

	DBRegisterFunctionSignatures[
		{
			"+",
			DBType[CompoundElement[{
				"TimeQuantity",
				a: "Date" | "Time" | "DateTime"
			}]]
		} :> {Function[DBSQLSymbol["DateAdd"][#2, #1]], DBType[a]}
	];

	DBRegisterFunctionSignatures[
		{
			"-",
			DBType[CompoundElement[{
				a: "Date" | "Time" | "DateTime",
				"TimeQuantity"
			}]]
		} :> {Function[DBSQLSymbol["DateAdd"][#1, DBSQLSymbol["-"][#2]]], DBType[a]}
	];

	DBRegisterFunctionSignatures[
		{
			"-",
			DBType[CompoundElement[{
				"TimeQuantity",
				a: "Date" | "Time" | "DateTime"
			}]]
		} :> {Function[DBSQLSymbol["DateAdd"][#2, DBSQLSymbol["-"][#1]]], DBType[a]}
	]

	DBRegisterFunctionSignatures[
		{
			"-",
			DBType[CompoundElement[{
				a: "Date" | "Time" | "DateTime",
				a_
			}]]
		} :> {
			Function[DBSQLSymbol["DateDifference"][##]],
			Function[If[
				And[
					DBTypeContainsQ @@ #["Constituents"],
					DBTypeContainsQ @@ Reverse[#["Constituents"]]
				],
				DBType["TimeQuantity"],
				DBType[DBTypeUnion[]]
			]]
		}
	];

	DBRegisterFunctionSignatures[
		{
			"-",
			DBType[CompoundElement[{
				DBTypeUnion["Integer", "Real", "Decimal"],
				DBTypeUnion["Integer", "Real", "Decimal"]
			}]]
		} -> {
			Function[DBSQLSymbol["-"][##]],
			DBType["Decimal"]
		}
	];

	DBRegisterFunctionSignatures[
		{
			"/",
			DBType[CompoundElement[{
				"TimeQuantity",
				"TimeQuantity"
			}]]
		} -> {Function[DBSQLSymbol["/"][##]], DBType["Real"]}
	];

	DBRegisterFunctionSignatures[
		{
			"/",
			DBType[CompoundElement[{
				"TimeQuantity",
				DBTypeUnion["Integer", "Real", "Decimal"]
			}]]
		} -> {Function[DBSQLSymbol["/"][##]], DBType["TimeQuantity"]}
	];

	DBRegisterFunctionSignatures[
		{
			"/",
			DBType[CompoundElement[{
				DBTypeUnion["Real", "Decimal"],
				DBTypeUnion["Real", "Decimal"]
			}]]
		} -> {Function[DBSQLSymbol["/"][##]], DBType["Real"]}
	];

	DBRegisterFunctionSignatures[
		{
			"/",
			DBType[CompoundElement[{
				"Integer",
				DBTypeUnion["Real", "Decimal"]
			}]]
		} -> {
			Function[DBSQLSymbol["/"][DBSQLSymbol["Cast"][#1, "Real"], #2]],
			DBType["Real"]
		}
	];

	DBRegisterFunctionSignatures[
		{
			"/",
			DBType[CompoundElement[{
				DBTypeUnion["Real", "Decimal"],
				"Integer"
			}]]
		} -> {
			Function[DBSQLSymbol["/"][#1, DBSQLSymbol["Cast"][#2, "Real"]]],
			DBType["Real"]
		}
	];

	DBRegisterFunctionSignatures[
		{
			"/",
			DBType[CompoundElement[{
				"Integer",
				"Integer"
			}]]
		} -> {
			Function[DBSQLSymbol["/"][DBSQLSymbol["Cast"][#1, "Real"], DBSQLSymbol["Cast"][#2, "Real"]]],
			DBType["Real"]
		}
	];

	DBRegisterFunctionSignatures[
		{
			"QUOTIENT",
			DBType[CompoundElement[{
				"Integer",
				"Integer"
			}]]
		} -> {Function[DBSQLSymbol["Quotient"][##]], DBType["Integer"]}
	];

	DBRegisterFunctionSignatures[
		{
			"%",
			DBType[CompoundElement[{
				"Integer",
				"Integer"
			}]]
		} -> {Function[DBSQLSymbol["%"][##]], DBType["Integer"]}
	];

	DBRegisterFunctionSignatures[
		{
			"^",
			DBType[CompoundElement[{
				DBTypeUnion["Integer"],
				DBTypeUnion["Integer"]
			}]]
		} -> {Function[DBSQLSymbol["^"][#1, #2]], DBType["Integer"]}
	];

	DBRegisterFunctionSignatures[
		{
			"^",
			DBType[CompoundElement[{
				DBTypeUnion["Integer", "Real", "Decimal"],
				DBTypeUnion["Integer", "Real", "Decimal"]
			}]]
		} -> {Function[DBSQLSymbol["^"][DBSQLSymbol["Cast"][#1, "Real"], #2]], DBType["Real"]}
	];

	DBRegisterFunctionSignatures[
		{
			op: "Regexp" | "IRegexp",
			DBType[CompoundElement[{
				"String",
				"String"
			}]]
		} :> {Function[DBSQLSymbol[op][##]], DBType["Boolean"]}
	];

	DBRegisterFunctionSignatures[
		{
			"-",
			DBType[CompoundElement[{
				a: "Integer" | "Real" | "Decimal"
			}]]
		} :> {Function[DBSQLSymbol["-"][##]], DBType[a]}
	];

	DBRegisterFunctionSignatures[
		{
			op: "Log" | "Sin" | "Cos" | "Tan" | "ArcTan" | "ArcSin" | "ArcCos",
			DBType[CompoundElement[{
				DBTypeUnion["Integer", "Real", "Decimal"]
			}]]
		} :> {Function[DBSQLSymbol[op][##]], DBType["Real"]}
	];

	DBRegisterFunctionSignatures[
		{
			"ArcTan2",
			DBType[CompoundElement[{
				DBTypeUnion["Integer", "Real", "Decimal"],
				DBTypeUnion["Integer", "Real", "Decimal"]
			}]]
		} -> {Function[DBSQLSymbol["ArcTan2"][#]], DBType["Real"]}
	];

	DBRegisterFunctionSignatures[
		{
			op: "Round"|"Ceiling"|"Floor",
			DBType[CompoundElement[{
				DBTypeUnion["Integer", "Real", "Decimal"]
			}]]
		} :> {Function[DBSQLSymbol["Cast"][DBSQLSymbol[op][#], "Integer"]], DBType["Integer"]}
	];

	DBRegisterFunctionSignatures[
		{
			op: "BitAnd" | "BitOr" | "BitXor",
			DBType[RepeatingElement["Integer",{2, Infinity}]]
		} :> {Function[DBSQLSymbol[op][##]], DBType["Integer"]}
	];

	DBRegisterFunctionSignatures[
		{
			"BitNot",
			DBType[CompoundElement[{"Integer"}]]
		} -> {Function[DBSQLSymbol["BitNot"][##]], DBType["Integer"]}
	];

	DBRegisterFunctionSignatures[
		{
			op: "BitShiftLeft" | "BitShiftRight",
			DBType[CompoundElement[{"Integer", "Integer"}]]
		} :> {Function[DBSQLSymbol[op][##]], DBType["Integer"]}
	];

	(* OrderedQ *)
	DBRegisterFunctionSignatures[
		{
			"<=",
			DBType[CompoundElement[{
				"String",
				"String"
			}]]
		} -> {
			Function[DBSQLSymbol["<="][##]],
			DBType["Boolean"]
		}
	];

	DBRegisterFunctionSignatures[
		{
			op : ">" | "<" | ">=" | "<=" | "=" | "<>",
			DBType[CompoundElement[{
				a: DBTypeUnion["Integer", "Real", "Decimal"] | "DateTime" | "Time" | "Date",
				a_
			}]]
		} :> {
			Function[DBSQLSymbol[op][##]],
			DBType["Boolean"]
		}
	];

	DBRegisterFunctionSignatures[
		{
			op : "=" | "<>",
			DBType[CompoundElement[{
				a: "Boolean" | "String" | "ByteArray",
				a_
			}]]
		} :> {
			Function[DBSQLSymbol[op][##]],
			DBType["Boolean"]
		}
	];
	(*
	DBRegisterFunctionSignatures[
		{
			"In",
			DBType[CompoundElement[{
				RectangularRepeatingElement[DBTypeUnion[DBType[All]]],
				RepeatingElement[DBTypeUnion[DBType[All]]]
			}]]
		} -> {
			Function[DBSQLSymbol["In"][##]],
			Function[
				With[{const = #["Constituents"]},
					If[
					(* this ensures that the two queries being compared are effectively of the same type*)
						SameQ[
							First[const]["Constituents"],
							Function[c, First @ c["Constituents"]] /@ Last[const]["Constituents"]
						],
						DBType["Boolean"],
						DBType[DBTypeUnion[]]
					]
				]
			]
		}
	];
	*)
	DBRegisterFunctionSignatures[
		{
			"In",
			DBType[CompoundElement[{
				"Query",
				"DatabaseModelInstance"
			}]]
		} -> {
			Function[
				{left, right},
				With[{
					pk = DBNormalizePrefixedFields[left @ DBGetPrefixedFields[left @ DBPrimaryKey[]]]},
					Apply[
						If[
							Length[pk] > 1,
							DBSQLSymbol["And"],
							Identity
						],
						Map[
							DBSQLSymbol["In"][right[#], left[#]]&,
							pk
						]
					]
				]
			],
			Function[
				If[
				(* this ensures that the two queries being compared are effectively of the same type*)
					SameQ @@ KeyTake[#["Constituents"][[All, 2]], {"PrimaryKey", "Fields"}],
					DBType["Boolean"],
					DBType[DBTypeUnion[]]
				]
			]
		}
	]

	DBRegisterFunctionSignatures[
		{
			op : "=" | "<>",
			DBType[CompoundElement[{
				"DatabaseModelInstance",
				"DatabaseModelInstance"
			}]]
		} :> {
			Function[
				{left, right},
				With[{
					pk = DBNormalizePrefixedFields[left @ DBGetPrefixedFields[left @ DBPrimaryKey[]]]},
					Apply[
						If[
							Length[pk] > 1,
							If[op === "=", DBSQLSymbol["And"], DBSQLSymbol["Or"]],
							Identity
						],
						Map[
							DBSQLSymbol[op][left[#], right[#]]&,
							pk
						]
					]
				]
			],
			Function[
				If[
					(* this ensures that the two queries being compared are effectively of the same type*)
					Length[DeleteDuplicates[#["Constituents"]]] === 1,
					DBType["Boolean"],
					DBType[DBTypeUnion[]]
				]
			]
		}
	];

	DBRegisterFunctionSignatures[
		{
			"<>",
			DBType[CompoundElement[{
				DBTypeUnion[All],
				DBTypeUnion[All]
			}]]
		} :> {
			Function[True],
			DBType["Boolean"]
		}
	];

	DBRegisterFunctionSignatures[
		{
			"=",
			DBType[CompoundElement[{
				DBTypeUnion[All],
				DBTypeUnion[All]
			}]]
		} :> {
			Function[False],
			DBType["Boolean"]
		}
	];

	DBRegisterFunctionSignatures[
		{
			"Between",
			DBType[CompoundElement[{
				a: DBTypeUnion["Integer", "Real", "Decimal"] | DBTypeUnion["Date", "DateTime"] | "Time",
				a_,
				a_
			}]]
		} :> {
			Function[DBSQLSymbol["Between"][##]],
			DBType["Boolean"]
		}
	];


	DBRegisterFunctionSignatures[
		{
			"Not",
			DBType[CompoundElement[{"Boolean"}]]
		} -> {
			Function[DBSQLSymbol["Not"][##]],
			DBType["Boolean"]
		}
	];


    DBRegisterFunctionSignatures[
        {
            "Exists",
            DBType[CompoundElement[{RepeatingElement[DBType[All]]}]]
        } -> {
            Function[DBSQLSymbol["Exists"][##]],
            DBType["Boolean"]
        }
    ];


	DBRegisterFunctionSignatures[
		{
			"Exists",
			DBType[CompoundElement[{RectangularRepeatingElement[DBType[All]]}]]
		} -> {
			Function[DBSQLSymbol["Exists"][##]],
			DBType["Boolean"]
		}
	];

	With[{all = $allAlternatives, allOrNum = Append[$allAlternatives, DBTypeUnion["Integer", "Real", "Decimal"]]},
		DBRegisterFunctionSignatures[
			{
				"In",
				DBType[CompoundElement[{RepeatingElement[a_], a: allOrNum}]]
			} :> {
				Function[DBSQLSymbol["In"][#2, #1]],
				DBType["Boolean"]
			}
		];

		DBRegisterFunctionSignatures[
			{
				"DeleteDuplicates",
				DBType[CompoundElement[{a: all}]]
			} :> {
				Function[DBSQLSymbol["DeleteDuplicates"][##]],
				DBType[a]
			}
		];

		DBRegisterFunctionSignatures[
			{
				"Count",
				DBType[CompoundElement[{RepeatingElement[a: all]}]]
			} :> {Function[DBSQLSymbol["Count"][#]], DBType["Integer"]}
		]

	];

	DBRegisterFunctionSignatures[
		{
			"Count",
			DBType[CompoundElement[{"Query"}]]
		} :> {Function[DBSQLSymbol["CountAll"][]], DBType["Integer"]}
	];

	DBRegisterFunctionSignatures[
		{
			op: "Least" | "Greatest",
			DBType[RepeatingElement[DBTypeUnion["Integer", "Real", "Decimal"], {2, Infinity}]]
		} :> {Function[DBSQLSymbol[op][##]], DBType[DBTypeUnion["Integer", "Real", "Decimal"]]}
	];

	DBRegisterFunctionSignatures[
		{
			op: "Least" | "Greatest",
			DBType[RepeatingElement[a: "Integer" | "Real" | "Decimal", {2, Infinity}]]
		} :> {Function[DBSQLSymbol[op][##]], DBType[a]}
	];

	DBRegisterFunctionSignatures[
		{
			op: "Least" | "Greatest",
			DBType[RepeatingElement[a: "Date" | "DateTime" | "Time", {2, Infinity}]]
		} :> {Function[DBSQLSymbol[op][##]], DBType[a]}
	];

	DBRegisterFunctionSignatures[
		{
			op: "Min" | "Max",
			DBType[CompoundElement[{RepeatingElement[a: "Date" | "DateTime" | "Time"]}]]
		} :> {Function[DBSQLSymbol[op][##]], DBType[a]}
	];

	DBRegisterFunctionSignatures[
		{
			op: "Sum" | "Min" | "Max",
			DBType[CompoundElement[{RepeatingElement[a: "Integer" | "Real" | "Decimal"]}]]
		} :> {Function[DBSQLSymbol[op][##]], DBType[a]}
	];

	DBRegisterFunctionSignatures[
		{
			op: "Mean" | "StandardDeviation" | "Variance",
			DBType[CompoundElement[{RepeatingElement[a: "Integer" | "Real" | "Decimal"]}]]
		} :> {Function[DBSQLSymbol[op][##]], DBType["Decimal"]}
	];

	DBRegisterFunctionSignatures[
		{
			op: "Or" | "And",
			DBType[RepeatingElement["Boolean", {2, Infinity}]]
		} :> {Function[DBSQLSymbol[op][##]], DBType["Boolean"]}
	];

	DBRegisterFunctionSignatures[
		{
			"Concat",
			DBType[RepeatingElement["String"]]
		} -> {Function[DBSQLSymbol["Concat"][##]], DBType["String"]}
	];

	DBRegisterFunctionSignatures[
		{
			"IsNull",
			DBType[CompoundElement[{DBType[All]}]]
		} -> {Function[DBSQLSymbol["IsNull"][##]], DBType["Boolean"]}
	];

	DBRegisterFunctionSignatures[
		{
			"Coalesce",
			DBType[RepeatingElement[DBType[All]]]
		} -> {Function[DBSQLSymbol["Coalesce"][##]], Function[DBType[DBTypeUnion @@ #["Constituents"]]]}
	];

	DBRegisterFunctionSignatures[
		{
			"Case",
			DBType[CompoundElement[{RepeatingElement["Boolean"], RepeatingElement[DBType[All]]}]]
		} -> {Function[DBSQLSymbol["Case"][##]], Function[DBType[DBTypeUnion @@ Last[#["Constituents"]]["Constituents"]]]}
	];
	(* CASE is column oriented !! *)

	DBRegisterFunctionSignatures[
		{
			"Now",
			DBType[CompoundElement[{}]]
		} -> {
			Function[DBSQLSymbol["Now"][]],
			DBType["DateTime", <|"TimeZone" -> True|>]
		}
	];

	DBRegisterFunctionSignatures[
		{
			"Today",
			DBType[CompoundElement[{}]]
		} -> {
			Function[DBSQLSymbol["Today"][]],
			DBType["Date"]
		}
	];

	DBRegisterFunctionSignatures[
		{
			"ToString",
			DBType[CompoundElement[{DBType[All]}]]
		} -> {
			Function[DBSQLSymbol["Cast"][#, "String"]],
			DBType["String"]
		}
	];

	DBRegisterFunctionSignatures[
		{
			"FromDigits",
			DBType[CompoundElement[{"String"}]]
		} -> {
			Function[DBSQLSymbol["Cast"][#, "Integer"]],
			DBType["Integer"]
		}
	];

	DBRegisterFunctionSignatures[
		{
			"N",
			DBType[CompoundElement[{DBTypeUnion["Decimal", "Integer", "Real"]}]]
		} -> {
			Function[DBSQLSymbol["Cast"][#, "Real"]],
			DBType["Real"]
		}
	];

	DBRegisterFunctionSignatures[
		{
			"IntegerPart",
			DBType[CompoundElement[{DBTypeUnion["Decimal", "Integer", "Real"]}]]
		} -> {
			Function[DBSQLSymbol["Cast"][#, "Integer"]],
			DBType["Integer"]
		}
	];

	DBRegisterFunctionSignatures[
		{
			"Boole",
			DBType[CompoundElement[{"Boolean"}]]
		} -> {
			Function[DBSQLSymbol["Cast"][#, "Integer"]],
			DBType["Integer"]
		}
	];

	DBRegisterFunctionSignatures[
		{
			"DateInterpreter",
			DBType[CompoundElement[{"String"}]]
		} -> {
			Function[DBSQLSymbol["Cast"][#, "Date"]],
			DBType["Date"]
		}
	];

	DBRegisterFunctionSignatures[
		{
			"TimeInterpreter",
			DBType[CompoundElement[{"String"}]]
		} -> {
			Function[DBSQLSymbol["Cast"][#, "Time"]],
			DBType["Time"]
		}
	];

	DBRegisterFunctionSignatures[
		{
			"DateTimeInterpreter",
			DBType[CompoundElement[{"String"}]]
		} -> {
			Function[DBSQLSymbol["Cast"][#, {"DateTime", True}]],
			DBType["DateTime", <|"TimeZone" -> True|>]
		}
	];

	DBRegisterFunctionSignatures[
		{
			"UnixTime",
			DBType[CompoundElement[{DBTypeUnion["String", "Date", "DateTime"]}]]
		} -> {
			Function[DBSQLSymbol["UnixTime"][#]],
			DBType["Decimal"]
		}
	];


	DBRegisterFunctionSignatures[
		{
			"FromUnixTime",
			DBType[CompoundElement[{DBTypeUnion["Decimal", "Real", "Integer"]}]]
		} -> {
			Function[DBSQLSymbol["FromUnixTime"][#]],
			DBType["DateTime"]
		}
	];

	DBRegisterFunctionSignatures[
		{
			"SecondsFromMidnight",
			DBType[CompoundElement[{"Time"}]]
		} -> {
			Function[DBSQLSymbol["SecondsFromMidnight"][#]],
			DBType["Decimal"]
		}
	];

	DBRegisterFunctionSignatures[
		{
			"FromSecondsFromMidnight",
			DBType[CompoundElement[{DBTypeUnion["Decimal", "Real", "Integer"]}]]
		} -> {
			Function[DBSQLSymbol["FromSecondsFromMidnight"][#]],
			DBType["Time"]
		}
	];

	DBRegisterFunctionSignatures[
		{
			"StringLength",
			DBType[CompoundElement[{"String"}]]
		} -> {
			Function[DBSQLSymbol["StringLength"][#]],
			DBType["Integer"]
		}
	];

	DBRegisterFunctionSignatures[
		{
			"Substr",
			DBType[CompoundElement[{"String", "Integer", "Integer"}]]
		} -> {
			Function[DBSQLSymbol["Substr"][##]],
			DBType["String"]
		}
	];
    
    DBRegisterFunctionSignatures[
		{
			"ByteArray",
			DBType[CompoundElement[{
                DBTypeUnion[DBDeduceExpressionType[{}], DBType["ByteArray"]]
            }]]
		} -> {
			Function[DBSQLSymbol["ByteArray"][##]],
			DBType["ByteArray"]
		}
	];
]

DBSortFunctionSignatures[]
