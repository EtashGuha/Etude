BeginPackage["GraphStore`Parsing`"];

GrammarApply1;

Begin["`Private`"];

Options[GrammarApply1] = {
	"InputCacheSize" -> 15000,
	"WhitespacePattern" -> Whitespace
};
GrammarApply1[args___] := With[{res = Catch[iGrammarApply1[args], $failTag]}, res /; res =!= $failTag];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


clear[iGrammarApply1];
Options[iGrammarApply1] = Options[GrammarApply1];

iGrammarApply1[HoldPattern[GrammarRules][rules_List, defs_], input_, opts : OptionsPattern[]] := Module[{tag}, Catch[
	Function[r,
		With[
			{res = iGrammarApply1[GrammarRules[r, defs], input, opts]},
			If[Unevaluated[res] =!= $Failed,
				Throw[Unevaluated[res], tag]
			]
		]
	] /@ rules;
	$Failed,
	tag
]];
iGrammarApply1[r : HoldPattern[GrammarRules][Except[_List], _List], input_String, opts : OptionsPattern[]] := iGrammarApply1[r, StringToStream[input], opts];
iGrammarApply1[HoldPattern[GrammarRules][Except[_List, rule_], defs_List], input_InputStream, OptionsPattern[]] := Block[{
	$input,
	$defs = <|defs|> // Map[List /* Flatten],
	$whitespacePattern = OptionValue["WhitespacePattern"],
	$firstCache = <||>,
	$ruleLookupTableCache = <||>,
	$fixInputCache = False,
	hasNext,
	next,
	$hasLeftRecursionCache = <||>,
	$leftRecursions = <||>,
	$leftRecursionParseCache = <||>
},
	Module[{
		inputCacheSize = OptionValue["InputCacheSize"],
		inputCache,
		actualInputCacheSize,
		hasNextCache,
		nextCache,
		checkInputCache
	},
		$input /: Set[$input, value_] := (
			inputCache = value;
			actualInputCacheSize = StringLength[inputCache];
			hasNextCache = inputCache =!= "";
			If[hasNext[],
				(* https://bugs.wolfram.com/show?number=335934 *)
				nextCache = First[StringTake[inputCache, {{1, 1}}]]
			];
			checkInputCache = True;
			value
		);
		$input := (
			If[! $fixInputCache && checkInputCache && actualInputCacheSize < inputCacheSize,
				$input = inputCache <> readMore[input, 2 * inputCacheSize];
				checkInputCache = False
			];
			inputCache
		);
		hasNext[] := hasNextCache;
		next[] := nextCache;
		$input = readMore[input, 2 * inputCacheSize];
	];
	trimWhitespace[];
	Module[
		{res = parseRule[rule]},
		trimWhitespace[];
		If[hasNext[],
			res = $Failed
		];
		res
	]
];
iGrammarApply1[HoldPattern[GrammarRules][rules_], input_, opts : OptionsPattern[]] := iGrammarApply1[GrammarRules[rules, {}], input, opts];
iGrammarApply1[rules_, file_File, rest___] := iGrammarApply1[rules, OpenRead[file], rest];

clear[readMore];
readMore[input_InputStream, chars_Integer] := Quiet[
	FromCharacterCode[ToCharacterCode[StringJoin[ReadList[input, Character, chars]]], "UTF-8"],
	{$CharacterEncoding::utf8}
];


clear[tokenQ];
tokenQ[_GrammarToken | _String] := True;
tokenQ[x_] := ! FreeQ[x, GrammarToken];

clear[tryParse];
tryParse[f_, patt_] := With[
	{oldInput = $input, res = f[patt]},
	With[
		{rule = $input -> res},
		$input = oldInput;
		rule
	]
];

clear[getFirst];
getFirst[patt_] := Lookup[
	$firstCache,
	Key[patt],
	$firstCache[[Key[patt]]] = Replace[
		Block[{$GFVisitedTokens = {}},
			Catch[iGetFirst[patt], $getFirstTag]
		],
		{
			{x_} :> x,
			l_List :> Alternatives @@ l
		}
	]
];

clear[iGetFirst];
iGetFirst[l : _Alternatives | {(_GrammarToken | _Rule | _RuleDelayed) ...}] := Union @@ iGetFirst /@ l;
iGetFirst[CaseSensitive[s_]] := Replace[CaseSensitive /@ iGetFirst[s], CaseSensitive[c_ /; ToUpperCase[c] === ToLowerCase[c]] :> c, {1}];
iGetFirst[d : DigitCharacter | LetterCharacter] := {d};
iGetFirst[FixedOrder[x_, ___]] := iGetFirst[x];
iGetFirst[t_GrammarToken] := If[MemberQ[$GFVisitedTokens, t],
	{},
	Block[{$GFVisitedTokens = Append[$GFVisitedTokens, t]},
		iGetFirst[tokenLookup[t]]
	]
];
iGetFirst[Verbatim[Pattern][_, p_]] := iGetFirst[p];
iGetFirst[Verbatim[Repeated][p_]] := iGetFirst[p];
iGetFirst[Verbatim[Repeated][p_, {_?(GreaterThan[0]), _}]] := iGetFirst[p];
iGetFirst[(Rule | RuleDelayed)[lhs_, _]] := iGetFirst[lhs];
iGetFirst[Except["", s_String]] := {StringTake[s, 1]};
iGetFirst[Verbatim[StringExpression][x_]] := iGetFirst[x];
iGetFirst[Verbatim[StringExpression][Verbatim[Repeated][p_, {0, _}] | Verbatim[RepeatedNull][p_], rest__]] := Union[iGetFirst[p], iGetFirst[StringExpression[rest]]];
iGetFirst[Verbatim[StringExpression][s_, __]] := iGetFirst[s];
iGetFirst[Verbatim[Verbatim][Except["", s_String]]] := {Verbatim[StringTake[s, 1]]};
iGetFirst[_] := Throw[Missing[], $getFirstTag];

clear[possibleMatchQ];
possibleMatchQ[patt_] := Or[
	! hasNext[],
	With[{first = getFirst[patt]},
		Or[
			MissingQ[first],
			StringMatchQ[next[], first, IgnoreCase -> True]
		]
	]
];

clear[pickRule];
pickRule[l : {__GrammarToken}, first_String] := Lookup[ruleLookupTable[l], first, $Failed];
pickRule[__] := $Failed;

clear[ruleLookupTable];
ruleLookupTable[l_List] := Lookup[
	$ruleLookupTableCache,
	Key[l],
	$ruleLookupTableCache[[Key[l]]] = makeRuleLookupTable[l]
];

clear[makeRuleLookupTable];
makeRuleLookupTable[l_List] := With[
	{first = getFirst /@ l},
	If[MemberQ[first, _Missing],
		Return[<||>]
	];
	With[
		{indicative = Table[
			If[
				And[
					StringQ[first[[i]]],
					ToLowerCase[first[[i]]] === ToUpperCase[first[[i]]],
					! StringMatchQ[first[[i]], Alternatives @@ Delete[first, i], IgnoreCase -> True]
				],
				first[[i]],
				Nothing
			],
			{i, Length[first]}
		]},
		AssociationThread[first, l][[indicative]]
	]
];

clear[hasLeftRecursion];
hasLeftRecursion[rule : (Rule | RuleDelayed)[patt_, _]] := Lookup[
	$hasLeftRecursionCache,
	rule,
	$hasLeftRecursionCache[rule] = iHasLeftRecursion[rule, patt]
];
hasLeftRecursion[t_GrammarToken] := Lookup[
	$hasLeftRecursionCache,
	t,
	$hasLeftRecursionCache[t] = AnyTrue[tokenLookup[t], Curry[iHasLeftRecursion, 2][t]]
];

clear[iHasLeftRecursion];
iHasLeftRecursion[rule_, rule_] := True;
iHasLeftRecursion[rule_, t_GrammarToken] := If[ListQ[seenTokens],
	If[MemberQ[seenTokens, t],
		True,
		AppendTo[seenTokens, t];
		AnyTrue[tokenLookup[t], Curry[iHasLeftRecursion, 2][rule]]
	],
	Block[{seenTokens = {}},
		iHasLeftRecursion[rule, t]
	]
];
iHasLeftRecursion[rule_, a_Alternatives] := AnyTrue[a, Curry[iHasLeftRecursion, 2][rule]];
iHasLeftRecursion[rule_, FixedOrder[x_, ___]] := iHasLeftRecursion[rule, x];
iHasLeftRecursion[rule_, Verbatim[Pattern][_, x_]] := iHasLeftRecursion[rule, x];
iHasLeftRecursion[rule_, Verbatim[Repeated][x_, ___]] := iHasLeftRecursion[rule, x];
iHasLeftRecursion[rule_, Verbatim[RepeatedNull][x_, ___]] := iHasLeftRecursion[rule, x];
iHasLeftRecursion[rule_, (Rule | RuleDelayed)[x_, _]] := iHasLeftRecursion[rule, x];
iHasLeftRecursion[_, _?(FreeQ[GrammarToken])] := False;

clear[tokenLookup];
tokenLookup[GrammarToken[token_?StringQ]] := Lookup[
	$defs,
	token,
	Message[GrammarRules::undsym, token];
	fail[]
] // validateToken;

clear[validateToken];
validateToken[x : _GrammarToken | _Rule | _RuleDelayed | {(_GrammarToken | _Rule | _RuleDelayed) ...}] := x;
validateToken[x_] := (
	Message[GrammarRules::arg2, x];
	fail[]
);


clear[parseRule];

parseRule[t_GrammarToken] := If[possibleMatchQ[t], parseRule[tokenLookup[t]], $Failed];
parseRule[{rule_}] := parseRule[rule];
parseRule[rules_List] := (
	If[hasNext[],
		With[
			{rule = pickRule[rules, next[]]},
			If[rule =!= $Failed,
				Return[parseRule[rule]]
			]
		]
	];
	DeleteCases[
		Block[
			{$fixInputCache = True},
			Function[rule,
				If[KeyExistsQ[$leftRecursions, rule],
					If[$leftRecursions[rule] > 0,
						$leftRecursions[rule]--;
						tryParse[parseRule, rule],
						Null -> $Failed
					],
					If[hasLeftRecursion[rule],
						Lookup[
							$leftRecursionParseCache,
							Key[{rule, $input}],
							$leftRecursionParseCache[{rule, $input}] = Module[{leftRecursionDepth = 0, parseRes = Null -> $Failed, previousParseRes},
								While[
									previousParseRes = parseRes;
									AssociateTo[$leftRecursions, Rule @@ {rule, leftRecursionDepth}];
									parseRes = tryParse[parseRule, rule];
									KeyDropFrom[$leftRecursions, rule];
									And[
										! MatchQ[parseRes, _ -> $Failed],
										! StringQ[First[previousParseRes]] || StringLength[First[parseRes]] < StringLength[First[previousParseRes]]
									],
									leftRecursionDepth++;
								];
								previousParseRes
							]
						],
						Block[{$leftRecursions = <||>},
							tryParse[parseRule, rule]
						]
					]
				]
			] /@ rules
		],
		_ -> $Failed
	] // Replace[{
		{} -> $Failed,
		{i_ -> r_} :> ($input = i; Unevaluated[r]),
		l_ :> Replace[First[TakeSmallestBy[l, First /* StringLength, 1]], (i_ -> r_) :> ($input = i; Unevaluated[r])]
	}]
);
parseRule[(Rule | RuleDelayed)[patt : (_Alternatives | _FixedOrder | _GrammarToken | (Repeated | RepeatedNull)[_?tokenQ, ___])?(FreeQ[Pattern]), rhs_]] := With[
	{r = parse[patt]},
	If[Unevaluated[r] === $Failed,
		$Failed,
		Unevaluated[rhs]
	]
];
parseRule[(Rule | RuleDelayed)[Verbatim[Pattern][name_Symbol, patt : (_Alternatives | _FixedOrder | _GrammarToken | (Repeated | RepeatedNull)[_?tokenQ, ___])?(FreeQ[Pattern])], rhs_]] := With[
	{r = parse[patt]},
	If[Unevaluated[r] === $Failed,
		$Failed,
		ReleaseHold[Hold[rhs] /. name :> r]
	]
];
parseRule[(Rule | RuleDelayed)[_?(Not @* possibleMatchQ), _]] := $Failed;
parseRule[(Rule | RuleDelayed)[fo : FixedOrder[___], rhs_]] := Module[{tag}, Catch[
	With[
		{oldInput = $input},
		ReleaseHold[Hold[rhs] /. Function[{var, patt},
			trimWhitespace[];
			var -> parse[patt] // Replace[
				(_ -> $Failed) :> (
					$input = oldInput;
					Throw[$Failed, tag]
				)
			]
		] @@@ Replace[List @@ fo, Except[_Pattern, p_] :> Pattern @@ {Unique[], p}, {1}]]
	],
	tag
]];
parseRule[(Rule | RuleDelayed)[Verbatim[Pattern][name_Symbol, patt_?(FreeQ[Pattern])], rhs_]] := Replace[
	StringPosition[$input, StartOfString ~~ patt, 1, IgnoreCase -> True],
	{
		{{1, n_}} :> With[
			{split = StringTake[$input, {{1, n}, {n + 1, -1}}]},
			$input = Last[split];
			ReleaseHold[Hold[rhs] /. name -> First[split]]
		],
		_ :> $Failed
	}
];
parseRule[(Rule | RuleDelayed)[patt_?(FreeQ[Pattern]), rhs_]] := Module[
	{pos, stringpatternQ = True},
	pos = Quiet[Check[
		StringPosition[$input, StartOfString ~~ patt, 1, IgnoreCase -> True],
		stringpatternQ = False,
		{StringExpression::invld}
	], {StringExpression::invld}];
	Replace[
		pos,
		{
			{{1, n_}} :> (
				$input = StringDrop[$input, n];
				Unevaluated[rhs]
			),
			_ :> $Failed
		}
	] /; stringpatternQ
];
parseRule[(Rule | RuleDelayed)[patt_, rhs_]] := Module[
	{res = $Failed, stringpatternQ = True},
	$input = Quiet[Check[
		StringReplace[
			$input,
			StartOfString ~~ patt :> (
				res = rhs;
				""
			),
			1,
			IgnoreCase -> True
		],
		stringpatternQ = False;
		$input,
		{StringExpression::invld}
	], {StringExpression::invld}];
	res /; stringpatternQ
];

parseRule[r : (Rule | RuleDelayed)[patt_, _]] := (
	Message[GrammarRules::bdrule2, patt, r];
	fail[]
);


clear[parse];

parse[t_GrammarToken] := parseRule[t];
parse[a : Verbatim[Alternatives][__GrammarToken]] := parseRule[List @@ a];
parse[a_Alternatives] := parseRule[With[{var = Unique[]}, Pattern @@ {var, #} -> var & /@ List @@ a]];
parse[fo_FixedOrder] := Module[{tag}, Catch[
	With[
		{oldInput = $input},
		Sequence @@ Function[p,
			trimWhitespace[];
			With[
				{r = parse[p]},
				If[Unevaluated[r] === $Failed,
					$input = oldInput;
					Throw[$Failed, tag],
					Unevaluated[r]
				]
			]
		] /@ fo
	],
	tag
]];
parse[Verbatim[Repeated][patt_?tokenQ, {min : _Integer | Infinity, max : _Integer | Infinity}]] := Module[
	{res = {}, i = 0},
	While[
		And[
			i < max,
			trimWhitespace[];
			With[
				{r = parse[patt]},
				If[Unevaluated[r] === $Failed,
					False,
					(* https://bugs.wolfram.com/show?number=332786 *)
					res = Append[res, Unevaluated[r]];
					i++;
					True
				]
			]
		]
	];
	If[Length[res] < min,
		Return[$Failed, Module]
	];
	Sequence @@ res
];
parse[Verbatim[Repeated][patt_]] := parse[Repeated[patt, {1, Infinity}]];
parse[Verbatim[RepeatedNull][patt_]] := parse[Repeated[patt, {0, Infinity}]];
parse[patt_] := parseRule[s : patt -> s];


clear[trimWhitespace];
trimWhitespace[] := (
	$input = StringDelete[$input, StartOfString ~~ $whitespacePattern ..];
)


End[];
EndPackage[];
