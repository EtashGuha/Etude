Package["Databases`Database`"]


PackageImport["Databases`"]
PackageImport["Databases`Common`"]
PackageImport["Databases`SQL`"]


PackageExport["DBExprToAST"]
PackageExport["$DBBinaryComparisonOperatorForms"]
PackageExport["$DBInPlaceEvaluation"]
PackageExport["DBDataToAST"]
PackageExport["DBAnnotation"]
PackageExport["DBExprToASTAnnotated"]

PackageScope["astSymbol"]
PackageScope["$dbAnnotationWrapper"]


(* ----------------           Patterns for primitive types          ---------------*)

$needsServerEvaluationPatt = Alternatives[
	DBSQLField,
	DBSQLSlot,
	DBPrefixedField,
	DBRawFieldName,
	$queryOperationHead,
	HoldPattern[Now],
	HoldPattern[Today],
    _ByteArray
]

fUneval[f_] := fUneval[f] = Function[arg, f[Unevaluated[arg]], HoldAll]

missingQ[arg_] := MissingQ[Unevaluated @ arg] || (Unevaluated[arg] === None)

$primitivePredicates = {
    NumberQ,
    StringQ,
    ByteArrayQ,
    BooleanQ,
    missingQ
}

$primitivePattern = Alternatives @@ Map[
    Function[PatternTest[Blank[], #]],
    $primitivePredicates
]

$primitiveHomogeneousListPattern = Alternatives @@ Map[
    Function[List[PatternTest[BlankNullSequence[], #]]],
    $primitivePredicates
]

{ $primitivePatternUnevaluated, $primitiveHomogeneousListPatternUnevaluated } =
    ReplaceAll[
        {$primitivePattern, $primitiveHomogeneousListPattern},
        Verbatim[PatternTest][patt_, f_] :>
            With[{ev = fUneval[f]},
                PatternTest[patt, ev] /; True
            ]
    ]

$ExplicitDataPatternUnevaluated = Alternatives[
    $primitivePatternUnevaluated,
    $primitiveHomogeneousListPatternUnevaluated,
    _DateObject,
    _TimeObject
]

(* ----------------              Main implementation                ---------------*)

SetAttributes[applyListable, HoldFirst]
applyListable[elem_List] := DBExprToAST /@ Unevaluated[elem]
applyListable[elem_] := DBExprToAST[elem]

$DBBinaryComparisonOperatorForms = {
	EqualTo, GreaterEqualThan, GreaterThan, LessEqualThan, LessThan, UnequalTo, SameQ, UnsameQ
}

$DBInPlaceEvaluation = True;

(* TODO: we need to figure out the interaction between slot names and aliasing for joins*)
SetAttributes[DBExprToAST, HoldAll]


(* In-place evaluation *)
DBExprToAST[expr_] /; And[
    TrueQ[$DBInPlaceEvaluation],
    FreeQ[Unevaluated[expr], $needsServerEvaluationPatt]
] := 
    With[{eval = expr}, 
        DBExprToAST[eval] /; Hold[eval] =!= Hold[expr]
    ]
    
DBExprToAST[b_ByteArray] := astSymbol["ByteArray"][b]    
           
DBExprToAST[expr_] /; And[
    !NumberQ[Unevaluated[expr]],
    NumericQ[Unevaluated[expr]]
] := N[expr]

DBExprToAST[(a_Association?AssociationQ)[key_]] := Extract[a, key, DBExprToAST]

DBExprToAST[Alternatives[
	(SameQ|Equal)[0, Length[list_]],
	(SameQ|Equal)[Length[list_], 0],
	(SameQ|Equal)[list_, {}],
	(SameQ|Equal)[{}, list_]
]] := astSymbol["Not"][astSymbol["Exists"][DBExprToAST[list]]]

DBExprToAST[Alternatives[
	(UnsameQ|Unequal)[0, Length[list_]],
	(UnsameQ|Unequal)[Length[list_], 0],
	(UnsameQ|Unequal)[list_, {}],
	(UnsameQ|Unequal)[{}, list_]
]] := astSymbol["Exists"][DBExprToAST[list]]

DBExprToAST[MissingQ[a_]|MatchQ[a_, Verbatim[_Missing]]] :=
    astSymbol["IsNull"][DBExprToAST[a]]

DBExprToAST[
	Alternatives[
		SelectFirst[list_, Not @* MissingQ|Function[!Missing[#]]],
		FirstCase[list_, Verbatim[Except[_Missing]]|Verbatim[Except[_?MissingQ]]]
	]
] := Apply[astSymbol["Coalesce"], applyListable[list]]


DBExprToAST[(SelectFirst|FirstCase)[{args___}, any_, default_]] :=
    DBExprToAST[SelectFirst[{args, default}, any]]

DBExprToAST[If[cond_, true_, false_]] := DBExprToAST[Which[cond, true, True, false]]
DBExprToAST[Which[args___]] := If[
	EvenQ[Length[Hold[args]]],
	Replace[
		DeleteCases[
			Cases[
				Partition[HoldComplete[args], 2],
				HoldComplete[c_, val_] :> {
					If[
						And[
							TrueQ[$DBInPlaceEvaluation],
							FreeQ[Unevaluated[c], $needsServerEvaluationPatt]
						],
						DBExprToAST @@ {c},
						DBExprToAST[c]
					],
					DBExprToAST[val]
				},
				{1}
			],
			{False | DBAnnotation[False, _], _}
		],
		{
			{} :> None,
			{{True | DBAnnotation[True, _], expr_}, ___} :> expr,
			list_ :> Apply[astSymbol["Case"], Transpose[list]]
		}
	],
	DBRaise[DBExprToAST, "which_called_with_odd_arguments", {args}]
]
(* CASE is column oriented !! *)

DBExprToAST[field_SQLF] := field;

DBExprToAST[Plus[a_, Times[-1, b_]]] := DBExprToAST[Subtract[a, b]]
DBExprToAST[(Plus|Times)[a_]] := DBExprToAST[a]
DBExprToAST[Plus[args__]] := Apply[astSymbol["+"], DBExprToAST /@ Hold[args]]
DBExprToAST[Subtract[a_, b_]] := astSymbol["-"][DBExprToAST[a], DBExprToAST[b]]
DBExprToAST[Minus[a_]] := astSymbol["-"][DBExprToAST[a]]
DBExprToAST[Times[num_, Power[den_, -1]]] := DBExprToAST[Divide[num, den]]
DBExprToAST[Times[args__]] := Apply[astSymbol["*"], DBExprToAST /@ Hold[args]]
DBExprToAST[(Divide|Rational)[num_, den_]] := astSymbol["/"][DBExprToAST[num], DBExprToAST[den]]
DBExprToAST[Quotient[num_, den_]] := astSymbol["QUOTIENT"][DBExprToAST[num], DBExprToAST[den]]
DBExprToAST[Power[base_, exp_]] := astSymbol["^"][DBExprToAST[base], DBExprToAST[exp]]
DBExprToAST[Sqrt[expr_]] := DBExprToAST[Power[expr, 0.5]]
DBExprToAST[Exp[expr_]] := DBExprToAST[Power[E, expr]]
(* TODO check that ^ in SQL works with negatives and floats *)
DBExprToAST[Mod[num_, den_]] := astSymbol["%"][DBExprToAST[num], DBExprToAST[den]]
DBExprToAST[Abs[num_]] := astSymbol["Abs"][DBExprToAST[num]]
DBExprToAST[OddQ[value_]]  := DBExprToAST[Mod[value, 2] > 0]
DBExprToAST[EvenQ[value_]] := DBExprToAST[Mod[value, 2] == 0]
DBExprToAST[(h: Log|Sin|Cos|Tan|ArcTan|ArcSin|ArcCos)[z_]] :=
    astSymbol[SymbolName[h]][DBExprToAST[z]]
DBExprToAST[Cot[x_]] := DBExprToAST[1/Tan[x]]
DBExprToAST[Sinh[x_]] := DBExprToAST[(Exp[x] - Exp[-x])/2]
DBExprToAST[Cosh[x_]] := DBExprToAST[(Exp[x] + Exp[-x])/2]
DBExprToAST[Tanh[x_]] := DBExprToAST[Sinh[x]/Cosh[x]]
DBExprToAST[Coth[x_]] := DBExprToAST[Cosh[x]/Sinh[x]]
DBExprToAST[ArcSinh[x_]] := DBExprToAST[Log[x + Sqrt[1 + x^2]]]
DBExprToAST[ArcCosh[x_]] := DBExprToAST[Log[x + Sqrt[x^2 - 1]]]
DBExprToAST[ArcTanh[x_]] := DBExprToAST[- 1/2 Log[1 - x] + 1/2 Log[1 + x]]
(*inversion of args below is intentional*)
DBExprToAST[ArcTan[x_, y_]] := astSymbol["ArcTan2"][DBExprToAST[y], DBExprToAST[x]]
DBExprToAST[Log[b_, z_]] := DBExprToAST[Log[z]/Log[b]]
DBExprToAST[Log2[z_]] := DBExprToAST[Log[z]/Log[2]]
DBExprToAST[Log10[z_]] := DBExprToAST[Log[z]/Log[10]]
DBExprToAST[Round[n_]] := astSymbol["Round"][DBExprToAST[n]]
DBExprToAST[Floor[n_]] := astSymbol["Floor"][DBExprToAST[n]]
DBExprToAST[Ceiling[n_]] := astSymbol["Ceiling"][DBExprToAST[n]]
DBExprToAST[(h:Round|Floor|Ceiling)[n_, a_]] := DBExprToAST[
	If[
		a > 0,
		h[n / a] * a,
		None
	]
]
DBExprToAST[BitOr[]|BitXor[]] := DBExprToAST[0]
DBExprToAST[BitAnd[]] := DBExprToAST[-1]
DBExprToAST[(h: BitAnd|BitOr|BitXor)[args___]] := Apply[
    astSymbol[SymbolName[h]],
    DBExprToAST /@ Hold[args]
]
DBExprToAST[(h: BitAnd|BitOr|BitXor)[args_]] := DBExprToAST[args]
DBExprToAST[(h: BitAnd|BitOr|BitXor)[args___]] := Apply[
	astSymbol[SymbolName[h]],
	DBExprToAST /@ Hold[args]
]
DBExprToAST[BitNot[a_]] := astSymbol["BitNot"][DBExprToAST[a]]
DBExprToAST[(h: BitShiftLeft|BitShiftRight)[a_, b_:1]] := astSymbol[SymbolName[h]][
	DBExprToAST[a],
	DBExprToAST[b]
]
DBExprToAST[DatePlus[a_, incr: {Except[_List], Except[_List]}]] :=
    DBExprToAST[DatePlus[a, {incr}]]
DBExprToAST[DatePlus[a_, incr: {{Except[_List], Except[_List]}..}]] :=
	Replace[
		{Hold[a], Extract[incr, {All, 1}, Hold], Extract[incr, {All, 2}, Hold]},
		{Hold[x_], Hold[y_], Hold[z_]} :> DBExprToAST[Plus[x, MixedRadixQuantity[y, z]]]
	]
DBExprToAST[DatePlus[a_, n: Except[_Quantity|_MixedRadixQuantity]]] :=
    DBExprToAST[Plus[a, Quantity[n, "Days"]]]
DBExprToAST[DatePlus[a_, n_]] := DBExprToAST[Plus[a, n]]
DBExprToAST[DateDifference[d1_, d2_]] := DBExprToAST[Subtract[d1, d2]]

DBExprToAST[
	expr: Alternatives[
		Nand,
		Nor,
		Xor,
		Xnor,
		Majority,
		BooleanCountingFunction[__],
		BooleanConsecutiveFunction[__],
		BooleanFunction[__]
	][___]
] := Internal`InheritedBlock[{And, Or, Not},
	SetAttributes[{And, Or, Not}, HoldAll];
	DBExprToAST @@ {LogicalExpand[Unevaluated[expr]]}
]

DBExprToAST[And[]] := True
DBExprToAST[Or[]] := False
DBExprToAST[(And|Or)[arg_]] := DBExprToAST[arg]
DBExprToAST[And[args__]] := Apply[astSymbol["And"], DBExprToAST /@ Hold[args]]
DBExprToAST[Or[args__]] := Apply[astSymbol["Or"], DBExprToAST /@ Hold[args]]
DBExprToAST[Not[arg_]] := astSymbol["Not"][DBExprToAST[arg]]

DBExprToAST[Greater[a_, b_]] := astSymbol[">"][DBExprToAST[a], DBExprToAST[b]]
DBExprToAST[Less[a_, b_]] := astSymbol["<"][DBExprToAST[a], DBExprToAST[b]]
DBExprToAST[GreaterEqual[a_, b_]] := astSymbol[">="][DBExprToAST[a], DBExprToAST[b]]
DBExprToAST[LessEqual[a_, b_]] := astSymbol["<="][DBExprToAST[a], DBExprToAST[b]]
DBExprToAST[(UnsameQ|Unequal)[a_, b_]] := astSymbol["<>"][DBExprToAST[a], DBExprToAST[b]]
DBExprToAST[(SameQ|Equal)[a_, b_]] /; Or[
	MissingQ[Unevaluated[a]],
	MissingQ[Unevaluated[b]],
	Unevaluated[a] === None,
	Unevaluated[b] === None
] := False
DBExprToAST[(SameQ|Equal)[a_, b_]] := astSymbol["="][DBExprToAST[a], DBExprToAST[b]]

DBExprToAST[OrderedQ[{a_, b_}]] := astSymbol["<="][DBExprToAST[a], DBExprToAST[b]]

DBExprToAST[Between[x_, {a_, b_}]] :=
    astSymbol["Between"][DBExprToAST[x], DBExprToAST[a], DBExprToAST[b]]
(*DBExprToAST[Unequal[args__]] := DBExprToAST[Evaluate[Apply[And, Unequal @@@ Subsets[DBExprToAST /@ {args}, {2}]]]]*)
(*this is probably going to do evaluation leaks and all sorts of bad stuff*)
DBExprToAST[MemberQ[list_, elem_]] := astSymbol["In"][applyListable[list], DBExprToAST[elem]]
DBExprToAST[MemberQ[list_][elem_]] := DBExprToAST[MemberQ[list, elem]]

DBExprToAST[MatchQ[elem_, Verbatim[Alternatives][stuff___]]] :=
    DBExprToAST[MemberQ[{stuff}, elem]]

DBExprToAST[MatchQ[Verbatim[Alternatives[stuff___]]][elem]] :=
    DBExprToAST[MemberQ[{stuff}, elem]]

DBExprToAST[(op: head_[val1_])[val2_]] /; MemberQ[$DBBinaryComparisonOperatorForms, head] :=
    With[{operator = compileOperator[op]},
	   operator[[2]][operator[[1]], DBExprToAST[val2]]
    ]

SetAttributes[deepFlatten, HoldAll]
deepFlatten[args___] := Flatten[
	List @@ (DBExprToAST /@ Flatten[Hold[{args}], Infinity, List]),
	Infinity,
	List
]

DBExprToAST[StringJoin[args___]] := Apply[
    astSymbol["Concat"],
    deepFlatten[args]
]

DBExprToAST[(h: Min|Max)[args___]] /; Or[
	!FreeQ[Hold[args], List],
	Length[Hold[args]] > 1
] := Apply[
	astSymbol[If[h === Min, "Least", "Greatest"]],
	deepFlatten[args]
]


DBExprToAST[func: (StringMatchQ|StringStartsQ|StringEndsQ|StringFreeQ)[
	patt_,
	opt: (IgnoreCase -> _): (IgnoreCase -> False)
][string_]] := DBExprToAST[func[string, patt, opt]]


SetAttributes[toPattern, HoldFirst]

toPattern[string_, patt_, ignorecase_:False, pre_:"", post_:""] :=
    astSymbol[If[TrueQ[ignorecase], "IRegexp", "Regexp"]][
        DBExprToAST[string],
        StringJoin[
            pre,
            StringReplace[
                (*
                WordCharacter is exploded into something that is making all backends complain
                https://stash.wolfram.com/projects/KERN/repos/kernel/browse/StartUp/StringPattern.m#56
                *)
                First @ StringPattern`PatternConvert @ ReplaceAll[
                    patt, {
                        LetterCharacter :> RegularExpression["(?!\\d|_)\\w"],
                        DigitCharacter  :> RegularExpression["\\d"],
                        WordCharacter   :> RegularExpression["(?!_)\\w"],
                        HexadecimalCharacter :> RegularExpression["[0-9a-fA-F]"],
                        PunctuationCharacter :> RegularExpression["[!'#S%&'\\(\\)\\*\\+,-\\./:;<=>\\?@\\[/\\]\\^_\\{\\|\\}~]"], (* using https://www.petefreitag.com/cheatsheets/regex/character-classes/ *)
                        EndOfString     :> RegularExpression["\\Z"] (* postgres is complaining about \z, need to use \Z *)
                    }
                ], {
                    (* mysql does not support modifiers, we need to remove them and re-implement them manually *)
                    "(?ms)" :> "", (* multiline mode *)
                    "?:" :> "" (* noncapturing group *)
                }
            ],
            post
        ]
    ]

DBExprToAST[StringMatchQ[string_, patt_, opt: (IgnoreCase -> _): (IgnoreCase -> False)]] :=
    toPattern[string, patt, Last[opt], "^", "$"]

DBExprToAST[StringStartsQ[string_, patt_, opt: (IgnoreCase -> _): (IgnoreCase -> False)]] :=
    toPattern[string, patt, Last[opt], "^"]

DBExprToAST[StringEndsQ[string_, patt_, opt: (IgnoreCase -> _): (IgnoreCase -> False)]] :=
    toPattern[string, patt, Last[opt], "", "$"]

DBExprToAST[StringContainsQ[string_, patt_, opt: (IgnoreCase -> _): (IgnoreCase 	-> False)]] :=
    toPattern[string, patt, Last[opt]]

DBExprToAST[StringFreeQ[string_, patt_, opt: (IgnoreCase -> _): (IgnoreCase -> False)]] :=
    DBExprToAST[! StringContainsQ[string, patt, opt]]

DBExprToAST[Now] := astSymbol["Now"][]
DBExprToAST[Today] := astSymbol["Today"][]

(* TODO: check if explicit lists of values can be used on the RHS of ALL and ANY *)
DBExprToAST[AllTrue[list_, operator_]] :=
    astSymbol["All"][Sequence @@ compileOperator[operator], DBExprToAST[list]]

DBExprToAST[AnyTrue[list_, operator_]] :=
    astSymbol["Any"][Sequence @@ compileOperator[operator], DBExprToAST[list]]

SetAttributes[compileOperator, HoldAll]

(* !! due to the semantics of ALL and ANY in SQL these are reverted !! *)

compileOperator[GreaterThan[val_]] := {DBExprToAST[val], astSymbol["<"]}
compileOperator[LessThan[val_]] := {DBExprToAST[val], astSymbol[">"]}
compileOperator[GreaterEqualThan[val_]] := {DBExprToAST[val], astSymbol["<="]}
compileOperator[LessEqualThan[val_]] := {DBExprToAST[val], astSymbol[">="]}
compileOperator[EqualTo[val_]] := {DBExprToAST[val], astSymbol["="]}
compileOperator[UnequalTo[val_]] := {DBExprToAST[val], astSymbol["<>"]}
(*TODO do ANY and ALL support other operators?*)

DBExprToAST[Total[list_]] := astSymbol["Sum"][compileAggregateArg[list]]
DBExprToAST[Mean[list_]] := astSymbol["Mean"][compileAggregateArg[list]]
DBExprToAST[StandardDeviation[list_]] := astSymbol["StandardDeviation"][compileAggregateArg[list]]
DBExprToAST[Variance[list_]] := astSymbol["Variance"][compileAggregateArg[list]]
DBExprToAST[Min[list_]] := astSymbol["Min"][compileAggregateArg[list]]
DBExprToAST[Max[list_]] := astSymbol["Max"][compileAggregateArg[list]]
DBExprToAST[Length[list_]|Count[list_, Verbatim[_]]] :=
    astSymbol["Count"][compileAggregateArg[list]]

DBExprToAST[Function[vars_, body_][args___]] := Function[vars, DBExprToAST[body], HoldAll][args]
DBExprToAST[Function[body_][args___]] := Function[Null, DBExprToAST[body], HoldAll][args]

DBExprToAST[substr[s_, m_, n_]] := astSymbol["Substr"][DBExprToAST[s], DBExprToAST[m], DBExprToAST[Abs[n]]]
DBExprToAST[negsubstr[s_, m_, n_]] := astSymbol["Concat"][
	astSymbol["Substr"][DBExprToAST[s], 1, DBExprToAST[Abs[m - 1]]],
	astSymbol["Substr"][DBExprToAST[s], DBExprToAST[m + n], DBExprToAST[Abs[StringLength[s] - m - n + 1]]]
]

DBExprToAST[(h: StringTake | StringDrop)[s_, UpTo[n_]]] := DBExprToAST[
	Which[
		n < 0, None,
		n > StringLength[s], h[s, StringLength[s]],
		True, h[s, n]
	]
]
DBExprToAST[(h: StringTake | StringDrop)[s_, {m_, UpTo[n_]}]] := DBExprToAST[
	Which[
		n < 0, None,
		n > StringLength[s], h[s, {m, StringLength[s]}],
		True, h[s, {m, n}]
	]
]
DBExprToAST[(h: StringTake | StringDrop)[s_, {UpTo[m_], n_}]] := DBExprToAST[
	Which[
		m < 0, None,
		m > StringLength[s], h[s, {StringLength[s], n}],
		True, h[s, {m, n}]
	]
]

DBExprToAST[(h: StringTake | StringDrop)[s_, n: Except[_List]]] := With[
	{fun = If[h === StringTake, substr, negsubstr]},
	DBExprToAST[
		Which[
			Abs[n] > StringLength[s],
				None,
			n >= 0,
				fun[s, 1, n],
			n < 0,
				fun[s, StringLength[s] + n + 1, - n]
		]
	]
]
DBExprToAST[(h: StringTake | StringDrop)[s_, {n_}]] := DBExprToAST[h[s, {n, n}]]
DBExprToAST[(h: StringTake | StringDrop)[s_, {m_, n_}]] := With[
	{fun = If[h === StringTake, substr, negsubstr]},
    DBExprToAST[
		Which[
			m > 0 && n >= 0,
				If[
					Or[
						n - m + 1 < 0, (*length >= 0*)
						m > StringLength[s] + 1, (*start <= stringlen + 1*)
						n + 1 > StringLength[s] + 1 (*end <= stringlen + 1*)
					],
					None,
					fun[s, m , n - m + 1]
				],
			m <= 0 && n < 0,
				If[
					Or[
						n - m + 1 < 0, (*length >= 0*)
						StringLength[s] + m + 1 <= 0, (*start > 0*)
						2 + n + StringLength[s] <= 0 (*end > 0*)
					],
					None,
					fun[s, StringLength[s] + m + 1, n - m + 1]
				],
			m > 0 && n < 0,
				If[
					Or[
						n + 2 > 1, (*end <= stringlen + 1*)
						m > StringLength[s] + 1, (*start <= stringlen + 1*)
						n - m + StringLength[s] + 2 < 0 (*length >= 0*)
					],
					None,
					fun[s, m, n - m + StringLength[s] + 2]
				],
			m <= 0 && n >= 0,
				If[
					Or[
						StringLength[s] + m + 1 <= 0, (*start > 0*)
						n - m - StringLength[s] < 0, (*length >= 0*)
						n + 1 > StringLength[s] + 1 (*end <= stringlen + 1*)
					],
					None,
					fun[s, StringLength[s] + m + 1, n - m - StringLength[s]]
				]
		]
	]
]

DBExprToAST[StringLength[s_]] := astSymbol["StringLength"][DBExprToAST[s]]




(* CAST-like heads *)
SetAttributes[myHold, HoldAll]

DBExprToAST[q: (h: Quantity)[m_: 1, unit_String]|(h: MixedRadixQuantity)[m_List, unit: {__String}]] /;
    UnitDimensions[Unevaluated[q]] === {{"TimeUnit", 1}} :=
	(*
		TODO because of a bug in QuantityQ we can't really support things like "Knots"/"Meters"
		since all unit functionality works symbolically
		this will work also with expressions like:
		MixedRadixQuantity[{x["hours"], x["minutes"]}, {"Hours", "Minutes"}]
	*)
	Replace[
		ReplaceAll[
			Hold @ Evaluate @ QuantityMagnitude @ UnitConvert[
				If[h === Quantity,
					Quantity[myHold[m], unit],
					MixedRadixQuantity[Map[myHold, Unevaluated @ m], unit]
				],
				"Seconds"
			],
			myHold[x_] :> x
		],
		Hold[expr_] :> DBSQLSecondsToTimeQuantity @ DBExprToAST @ expr
	]
DBExprToAST[ToString[arg_]] := astSymbol["ToString"][DBExprToAST[arg]]
DBExprToAST[(Interpreter["Integer"]|FromDigits)[arg_]] := astSymbol["FromDigits"][DBExprToAST[arg]]
DBExprToAST[N[arg_]] := astSymbol["N"][DBExprToAST[arg]]
DBExprToAST[IntegerPart[arg_]] := astSymbol["IntegerPart"][DBExprToAST[arg]]
DBExprToAST[Boole[arg_]] := astSymbol["Boole"][DBExprToAST[arg]]
(*

This cannot be done on V1, disabling interpreter for now

DBExprToAST[Interpreter["Date"|"StructuredDate"][arg_]] := astSymbol["DateInterpreter"][DBExprToAST[arg]]
DBExprToAST[Interpreter["Time"|"StructuredTime"][arg_]] := astSymbol["TimeInterpreter"][DBExprToAST[arg]]
DBExprToAST[Interpreter["DateTime"|"StructuredDateTime"][arg_]] :=
    astSymbol["DateTimeInterpreter"][DBExprToAST[arg]]
*)
DBExprToAST[UnixTime[expr_:Now]] := astSymbol["UnixTime"][DBExprToAST[expr]]
DBExprToAST[FromUnixTime[expr_]] := astSymbol["FromUnixTime"][DBExprToAST[expr]]
DBExprToAST[JulianDate[expr_:Now]] := DBExprToAST[UnixTime[expr] / 86400.0 + 2440587.5]
DBExprToAST[FromJulianDate[expr_]] := DBExprToAST[FromUnixTime[86400.0 * expr - 210866760000.]]
DBExprToAST[DateValue[expr_, "SecondsFromMidnight"]] :=
    astSymbol["SecondsFromMidnight"][DBExprToAST[expr]]
DBExprToAST[DateValue[UnixTime[arg_], "TimeObject"]] :=
    astSymbol["FromSecondsFromMidnight"][DBExprToAST[arg]]

SetAttributes[compileAggregateArg, HoldAll]
compileAggregateArg[(DeleteDuplicates|Union)[arg_]] :=
	astSymbol["DeleteDuplicates"][compileAggregateArg @ arg];

compileAggregateArg[arg_] := DBExprToAST @ arg;

(* ------------------     Data / primitive types processing     -------------------*)

With[{patt = $ExplicitDataPatternUnevaluated},
    (* Need to inject the pattern, since DBExprToAST is HoldAll *)
    DBExprToAST[data: patt] := DBDataToAST[data]
]

(*
** Once we established the we are dealing with data, no need to keep argument
** unevaluated, so DBDataToAST is not holding args, and we use simpler patterns.
*)
DBDataToAST[p: $primitivePattern] := Replace[p, _?MissingQ -> None];
DBDataToAST[p: $primitiveHomogeneousListPattern] := Replace[p, _?MissingQ -> None, {1}];
DBDataToAST[do: (_DateObject|_TimeObject)] := do

(* ------------------            Catch - all case               -------------------*)

DBExprToAST[expr_] := DBRaise[DBExprToAST, "unknown_element", {HoldForm[expr]}]



DBExprToASTAnnotated[fun_, wrapper_ : $dbAnnotationWrapper] :=
    DBBlockModify[
        DBExprToAST,
        Function[{args, rhs}, DBAnnotation[rhs, wrapper[args]], HoldAll],
        fun
    ]

$dbAnnotationWrapper = Function[x, Apply[HoldForm, Unevaluated[x]], HoldFirst]       
