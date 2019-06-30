
BeginPackage["CompileAST`Export`Format`Information`Character`"]

$SyntaxCharacterInformation
SyntaxCharacterInformation

Begin["`Private`"]

Needs["CompileAST`Language`ShortNames`"]

ClearAll[SyntaxCharacterInformation]

If[!ValueQ[characterMap],
	characterMap = <||>
]

shortNameToSymbolMap := shortNameToSymbolMap = 
	AssociationMap[Reverse, $SystemShortNames]


If[!ValueQ[$SyntaxCharacterInformation],
	$SyntaxCharacterInformation = <||>
]

notLetterQ[Nothing] = False
notLetterQ[fixity_] := MemberQ[{"Infix", "Prefix", "Postfix"}, fixity]

ClearAll[formatRawChar]
formatRawChar[fixity_?notLetterQ, _, {{_?StringQ, name_?StringQ}, ___}, _] := name
formatRawChar[fixity_?notLetterQ, _, {{name_?StringQ}, ___}, _] := name
formatRawChar[_, _, None, {{"Alias", name_}}] := name
formatRawChar[_, code_?StringQ, _, _] := FromCharacterCode[FromDigits[code, 16]]
formatRawChar[_, None, _, _] := Nothing

ClearAll[getPrecedence]
getPrecedence[{___, {"Raw" | "Math", prec_?IntegerQ}}] := prec
getPrecedence[___] := Nothing

ClearAll[getType]
getType[{{"Alias", "raw operator"}}] := "Type" -> "RawOperator"
getType[{{"Letter", ""}}] := "Type" -> "Letter"
getType[op_] := With[{flt = Flatten[{op}]},
	If[notLetterQ[flt],
		"Type" -> "Operator",
		"Other" -> op
	]
]
getType[other_] := "Other" -> other

ClearAll[getAssociativity]
iGetAssociativity[{_, _, assoc_, _, _, _}] :=
	If[assoc === "None",
		Nothing,
		assoc
	]
iGetAssociativity[{___}] :=
	Nothing
getAssociativity[None] = {}
getAssociativity[{}] = {}
getAssociativity[{assoc_?ListQ, rest___}] :=
	Join[{iGetAssociativity[assoc]}, getAssociativity[{rest}]]
		
	

symbolName[char_] :=
	Lookup[shortNameToSymbolMap, char, Nothing]
parse[record:{hex_, longname_, {escapes__} | None, other_, rawChar_, info_, prec_}] :=
	Module[{name, fullname, lname, char, fixity},
		lname = Replace[longname, ASCII[v_] :> v];
		lname=If[MatchQ[prec, {"TextTable",{___}}],
			First@Last@prec,
			If[MatchQ[longname, ASCII[_]],
				First@longname,
				longname
			]
		];
		If[lname =!=None && StringQ[lname] && StringMatchQ[lname, "\\"~~___],
			StringDrop[lname, 1],
			lname
		];
		If[lname === None,
			Return[Nothing]
		];
		
		fixity = If[Length[Flatten[{prec}]] === 5,
			Which[
				Or @@ StringMatchQ[Flatten[prec], "INFIX" ~~ ___],
					"Infix",
				Or @@ StringMatchQ[Flatten[prec], "PREFIX" ~~ ___],
					"Prefix",
				Or @@ StringMatchQ[Flatten[prec], "POSTFIX" ~~ ___],
					"Postfix",
				Or @@ StringMatchQ[Flatten[prec], "LETTER" ~~ ___],
					"Letter",
				Or @@ StringMatchQ[Flatten[prec], "OPEN" ~~ ___],
					"Open",
				Or @@ StringMatchQ[Flatten[prec], "CLOSE" ~~ ___],
					"Close",
				Or @@ StringMatchQ[Flatten[prec], "WHITESPACE" ~~ ___],
					"WhiteSpace",
				True,
					Nothing
			],
			If[Length[Flatten[{prec}]] === 3,
				With[{f = Last@Last[prec]},
					Capitalize[ToLowerCase[f]]
				],
				Nothing
			]
		];
		char = formatRawChar[fixity, hex, {escapes}, rawChar];
		fullname = Replace[longname, ASCII[v_] :> v];
		name = If[StringQ[fullname],
			StringTrim[fullname, "Raw"],
			fullname
		];
		AssociateTo[
			characterMap,
			char -> name
		];
		name -> Association[Flatten[{
			"Code" -> hex,
			"Name" -> name,
			"TextName" -> lname,
			"LongName" -> fullname,
         	"Character" -> char,
         	"SymbolName" -> symbolName[char],
			"ShortForms" -> {escapes},
			"Fixity" -> fixity,
			"Precedence" -> getPrecedence[info],
			"Associativity" -> With[{tmp = getAssociativity[other]},
				If[tmp === {},
					Nothing,
					tmp
				]
			],
			getType[other]
			(* , "Raw" -> record *)
		}]]
	]

init[] :=
	Module[{unicode},		
		unicode = Get["CompileAST`Language`UnicodeTable`"];
		$SyntaxCharacterInformation = Association[Map[
	         parse,
	         unicode
		]];
		$SyntaxCharacterInformation
	];

init[]

error[template_, params_] :=
	Failure[
		"SyntaxCharacterInformation",
		<|
			"MessageTemplate" :> template,
			"MessageParameters" -> params
		|>
	]

iSyntaxCharacterInformation[tok_] :=
	Lookup[
		$SyntaxCharacterInformation,
		tok,
		error["Invalid token `Token`", <| "Token" -> tok |>]
	]
ClearAll[SyntaxCharacterInformation]
SyntaxCharacterInformation[tok_?StringQ] := SyntaxCharacterInformation[tok] =
	Module[{res, tmp},
		res = iSyntaxCharacterInformation[tok];
		If[FailureQ[res],
			tmp = Which[
				KeyExistsQ[shortNameToSymbolMap, tok],
					SyntaxCharacterInformation[shortNameToSymbolMap[tok]],
				KeyExistsQ[$ShortNames, tok] && KeyExistsQ[characterMap, $ShortNames[tok]],
					iSyntaxCharacterInformation[characterMap[$ShortNames[tok]]],
				KeyExistsQ[characterMap, tok],
					iSyntaxCharacterInformation[characterMap[tok]],
				KeyExistsQ[$SyntaxCharacterInformation, StringJoin["Raw", tok]],
					iSyntaxCharacterInformation[StringJoin["Raw", tok]],
				True,
					res
			];
			If[FailureQ[tmp],
				res,
				tmp
			],
			res
		]
	]
SyntaxCharacterInformation[tok_Symbol] :=
	SyntaxCharacterInformation[SymbolName[tok]]
SyntaxCharacterInformation[tok_?AtomQ] :=
	SyntaxCharacterInformation[ToString[tok]]
SyntaxCharacterInformation[Infinity] :=
	SyntaxCharacterInformation["Infinity"]
SyntaxCharacterInformation[tok_Symbol, All] :=
	SyntaxCharacterInformation[SymbolName[tok]]
SyntaxCharacterInformation[Infinity, All] :=
	SyntaxCharacterInformation["Infinity", All]
SyntaxCharacterInformation[tok_?AtomQ, All] :=
	SyntaxCharacterInformation[ToString[tok]]
	
SyntaxCharacterInformation[tok_, query:(
		"Code" | "Name" | "LongName" | "Character" |
		"SymbolName" | "ShortForms" | "Fixity" | "Precedence"
	)] :=
	With[{info = SyntaxCharacterInformation[tok]},
		If[FailureQ[info],
			info,
			Lookup[info, query]
		]
	]
SyntaxCharacterInformation[tok_, query_?ListQ] :=
	SyntaxCharacterInformation[tok, #]& /@ query
	
End[]

EndPackage[]
