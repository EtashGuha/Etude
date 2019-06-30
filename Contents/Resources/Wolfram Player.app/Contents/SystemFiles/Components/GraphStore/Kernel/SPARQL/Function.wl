BeginPackage["GraphStore`SPARQL`Function`", {"GraphStore`", "GraphStore`SPARQL`"}];

Needs["GraphStore`IRI`"];
Needs["GraphStore`RDF`"];

Begin["`Private`"];


EvaluateSPARQLFunction[i_IRI, args___] := evalIRI[i, args];
EvaluateSPARQLFunction[s_String, args___] := evalString[s, args];
EvaluateSPARQLFunction[SPARQLEvaluation[f : _String | _IRI], args___] := EvaluateSPARQLFunction[f, args];
EvaluateSPARQLFunction[SPARQLEvaluation[f_], args___] := f[args];
EvaluateSPARQLFunction[___] := $Failed;


(* 17.4 Function Definitions *)


(* 17.4.1 Functional Forms *)

(* 17.4.1.1 bound *)
evalString["BOUND", var_] := ! MatchQ[var, _SPARQLVariable];
(* 17.4.1.2 IF *)
evalString["IF", cond_, t_, f_] := evalIRI[IRI["http://www.w3.org/2005/xpath-functions#boolean"], cond] // Replace[b_?BooleanQ :> If[b, t, f]];
(* 17.4.1.3 COALESCE *)
evalString["COALESCE", args___] := FirstCase[{args}, Except[$Failed | _SPARQLVariable], $Failed];
(* 17.4.1.4 NOT EXISTS and EXISTS *)
(* 17.4.1.5 logical-or *)
(* 17.4.1.6 logical-and *)
(* 17.4.1.7 RDFterm-equal *)
evalString["RDFTERM-EQUAL", term1_, term2_] := term1 === term2;
(* 17.4.1.8 sameTerm *)
(* 17.4.1.9 IN, 17.4.1.10 NOT IN *)
evalString["IN", x_, y_] := ContainsAny[y, {x}];


(* 17.4.2 Functions on RDF Terms *)

(* 17.4.2.1 isIRI *)
evalString["ISIRI", _IRI] := True;
evalString["ISIRI", _] := False;
(* 17.4.2.2 isBlank *)
evalString["ISBLANK", _RDFBlankNode] := True;
evalString["ISBLANK", _] := False;
(* 17.4.2.3 isLiteral *)
evalString["ISLITERAL", _IRI | _RDFBlankNode] := False;
evalString["ISLITERAL", _] := True;
(* 17.4.2.4 isNumeric *)
evalString["ISNUMERIC", x_] := NumericQ[x];
(* 17.4.2.5 str *)
evalString["STR", s_String] := s;
evalString["STR", RDFString[s_, _]] := s;
evalString["STR", RDFLiteral[s_, _]] := s;
evalString["STR", IRI[i_]] := i;
evalString["STR", x_] := ToString[x];
(* 17.4.2.6 lang *)
evalString["LANG", RDFString[_, lang_]] := lang;
evalString["LANG", _] := "";
(* 17.4.2.7 datatype *)
evalString["DATATYPE", RDFLiteral[_, dt_String | IRI[dt_String]]] := IRI[dt];
evalString["DATATYPE", _String] := IRI["http://www.w3.org/2001/XMLSchema#string"];
evalString["DATATYPE", _RDFString] := IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#langString"];
evalString["DATATYPE", x_] := ToRDFLiteral[x] // Replace[{RDFLiteral[_, dt_String] :> IRI[dt], _ :> $Failed}];
(* 17.4.2.8 IRI *)
evalString["IRI" | "URI", i_IRI] := i;
evalString["IRI" | "URI", s_] := IRI[ExpandIRI[IRI[lexicalForm[s]], $Base]];
(* 17.4.2.9 BNODE *)
evalString["BNODE"] := RDFBlankNode[CreateUUID["b-"]];
evalString["BNODE", s_String] := HeldBNode[s];
(* 17.4.2.10 STRDT *)
evalString["STRDT", s_String, IRI[dt_String]] := FromRDFLiteral[RDFLiteral[s, dt]];
(* 17.4.2.11 STRLANG *)
evalString["STRLANG", s_String, lang_String] := RDFString[
	s,
	With[{parts = StringSplit[lang, "-"]},
		StringRiffle[
			Join[
				{ToLowerCase[First[parts]]},
				StringReplace[
					Rest[parts],
					{
						StartOfString ~~ two : Repeated[_, {2}] ~~ EndOfString :> ToUpperCase[two],
						StartOfString ~~ first_ ~~ three : Repeated[_, {3}] ~~ EndOfString :> ToUpperCase[first] <> three,
						other__ :> ToLowerCase[other]
					}
				]
			],
			"-"
		]
	]
];
(* 17.4.2.12 UUID *)
evalString["UUID"] := IRI[CreateUUID["urn:uuid:"]];
(* 17.4.2.13 STRUUID *)
evalString["STRUUID"] := CreateUUID[];


(* 17.4.3 Functions on Strings *)

(* 17.4.3.1.1 String arguments *)
$stringLiteralPattern = _String | RDFString[_String, _String] | RDFLiteral[_String, "http://www.w3.org/2001/XMLSchema#string" | IRI["http://www.w3.org/2001/XMLSchema#string"]];
(* 17.4.3.1.2 Argument Compatibility Rules *)
argumentsCompatibleQ[_String | RDFLiteral[_String, "http://www.w3.org/2001/XMLSchema#string" | IRI["http://www.w3.org/2001/XMLSchema#string"]], _String | RDFLiteral[_String, "http://www.w3.org/2001/XMLSchema#string" | IRI["http://www.w3.org/2001/XMLSchema#string"]]] := True;
argumentsCompatibleQ[RDFString[_String, lang_], RDFString[_String, lang_]] := True;
argumentsCompatibleQ[RDFString[_String, _], _String | RDFLiteral[_String, "http://www.w3.org/2001/XMLSchema#string" | IRI["http://www.w3.org/2001/XMLSchema#string"]]] := True;
argumentsCompatibleQ[_, _] := False;
(* 17.4.3.2 STRLEN *)
evalString["STRLEN", s : $stringLiteralPattern] := StringLength[lexicalForm[s]];
(* 17.4.3.3 SUBSTR *)
evalString["SUBSTR", source : $stringLiteralPattern, startingLoc_] := StringTake[lexicalForm[source], {Interpreter["Integer"][startingLoc], -1}] // restoreForm[source];
evalString["SUBSTR", source : $stringLiteralPattern, startingLoc_, length_] := StringTake[lexicalForm[source], {Interpreter["Integer"][startingLoc], Interpreter["Integer"][startingLoc] + Interpreter["Integer"][length] - 1}] // restoreForm[source];
(* 17.4.3.4 UCASE *)
evalString["UCASE", s : $stringLiteralPattern] := ToUpperCase[lexicalForm[s]] // restoreForm[s];
(* 17.4.3.5 LCASE *)
evalString["LCASE", s : $stringLiteralPattern] := ToLowerCase[lexicalForm[s]] // restoreForm[s];
(* 17.4.3.6 STRSTARTS *)
evalString["STRSTARTS", arg1 : $stringLiteralPattern, arg2 : $stringLiteralPattern] := StringStartsQ[lexicalForm[arg1], lexicalForm[arg2]];
(* 17.4.3.7 STRENDS *)
evalString["STRENDS", arg1 : $stringLiteralPattern, arg2 : $stringLiteralPattern] := StringEndsQ[lexicalForm[arg1], lexicalForm[arg2]];
(* 17.4.3.8 CONTAINS *)
evalString["CONTAINS", arg1 : $stringLiteralPattern, arg2 : $stringLiteralPattern] := StringContainsQ[lexicalForm[arg1], lexicalForm[arg2]];
(* 17.4.3.9 STRBEFORE *)
evalString["STRBEFORE", s1 : $stringLiteralPattern, s2 : $stringLiteralPattern] := If[argumentsCompatibleQ[s1, s2],
	StringSplit[lexicalForm[s1], lexicalForm[s2], 2] // Replace[{{res_, _} :> restoreForm[s1][res], _ :> ""}],
	$Failed
];
(* 17.4.3.10 STRAFTER *)
evalString["STRAFTER", s1 : $stringLiteralPattern, s2 : $stringLiteralPattern] := If[argumentsCompatibleQ[s1, s2],
	StringSplit[lexicalForm[s1], lexicalForm[s2], 2] // Replace[{{_, res_} :> restoreForm[s1][res], _ :> ""}],
	$Failed
];
(* 17.4.3.11 ENCODE_FOR_URI *)
evalString["ENCODE_FOR_URI", s : $stringLiteralPattern] := URLEncode[lexicalForm[s]];
(* 17.4.3.12 CONCAT *)
evalString["CONCAT", s___String] := StringJoin[s];
evalString["CONCAT", s : RDFString[_, lang_] ..] := RDFString[StringJoin[{s}[[All, 1]]], lang];
evalString["CONCAT", s : RDFLiteral[_, dt_] ..] := RDFLiteral[StringJoin[{s}[[All, 1]]], dt];
evalString["CONCAT", s : $stringLiteralPattern ..] := StringJoin[lexicalForm /@ {s}];
(* 17.4.3.13 langMatches *)
evalString["LANGMATCHES", tag : $stringLiteralPattern, range : $stringLiteralPattern] := lexicalForm /@ {tag, range} // Replace[{
	{_, "*"} :> True,
	{t_, r_} :> StringMatchQ[t, r, IgnoreCase -> True] || StringStartsQ[t, r <> "-", IgnoreCase -> True]
}];
(* 17.4.3.14 REGEX *)
evalString["REGEX", text : $stringLiteralPattern, pattern : $stringLiteralPattern, flags : $stringLiteralPattern : ""] := StringContainsQ[
	lexicalForm[text],
	RegularExpression[lexicalForm[pattern]],
	IgnoreCase -> Replace[lexicalForm[flags], {_String?(StringContainsQ["i"]) -> True, _ -> False}]
] // restoreForm[text];
(* 17.4.3.15 REPLACE *)
evalString["REPLACE", arg : $stringLiteralPattern, pattern : $stringLiteralPattern, replacement : $stringLiteralPattern, flags : $stringLiteralPattern : ""] := StringReplace[
	lexicalForm[arg],
	RegularExpression[lexicalForm[pattern]] -> lexicalForm[replacement],
	IgnoreCase -> Replace[lexicalForm[flags], {_String?(StringContainsQ["i"]) -> True, _ -> False}]
] // restoreForm[arg];


(* 17.4.4 Functions on Numerics *)

(* 17.4.4.1 abs *)
evalString["ABS", x_] := Abs[x];
(* 17.4.4.2 round *)
evalString["ROUND", i_Integer] := i;
evalString["ROUND", x_?Positive /; Mod[x, 1] == 0.5] := N[Round[x] + 1];
evalString["ROUND", x_] := N[Round[x]];
(* 17.4.4.3 ceil *)
evalString["CEIL", x_] := N[Ceiling[x]];
(* 17.4.4.4 floor *)
evalString["FLOOR", x_] := N[Floor[x]];
(* 17.4.4.5 RAND *)
evalString["RAND"] := RandomReal[];


(* 17.4.5 Functions on Dates and Times *)

(* 17.4.5.1 now *)
evalString["NOW"] := ToRDFLiteral[Now];
(* 17.4.5.2 year *)
evalString["YEAR", x_?DateObjectQ] := DateValue[x, "Year"];
(* 17.4.5.3 month *)
evalString["MONTH", x_?DateObjectQ] := DateValue[x, "Month"];
(* 17.4.5.4 day *)
evalString["DAY", x_?DateObjectQ] := DateValue[x, "Day"];
(* 17.4.5.5 hours *)
evalString["HOURS", x_?DateObjectQ] := DateValue[x, "Hour24"];
(* 17.4.5.6 minutes *)
evalString["MINUTES", x_?DateObjectQ] := DateValue[x, "Minute"];
(* 17.4.5.7 seconds *)
evalString["SECONDS", x_?DateObjectQ] := DateValue[x, "SecondExact"];
(* 17.4.5.8 timezone *)
evalString["TIMEZONE", x_?DateObjectQ] := DateValue[x, "TimeZone"] // Replace[{
	None :> $Failed,
	tz_ :> FromRDFLiteral[RDFLiteral[
		StringJoin[{
			If[tz < 0, "-", Nothing],
			"P",
			"T",
			DateValue[TimeObject[{Abs[tz], 0}], {"Hour", "Minute"}] // Round // Replace[{
				{0, 0} :> "0S",
				{h_, 0} :> ToString[h] <> "H",
				{0, m_} :> ToString[m] <> "M",
				{h_, m_} :> ToString[h] <> "H" <> ToString[m] <> "M"
			}]
		}],
		"http://www.w3.org/2001/XMLSchema#dayTimeDuration"
	]]
}];
(* 17.4.5.9 tz *)
evalString["TZ", RDFLiteral[x_String, "http://www.w3.org/2001/XMLSchema#dateTime"]] := StringReplace[x, {
	StartOfString ~~ __ ~~ "T" ~~ __ ~~ tz : ("+" | "-" ~~ __) ~~ EndOfString :> tz,
	StartOfString ~~ __ ~~ "T" ~~ __ ~~ "Z" ~~ EndOfString :> "Z",
	___ :> ""
}];
evalString["TZ", x_?DateObjectQ] := evalString["TZ", ToRDFLiteral[x]];

evalString[fn : "YEAR" | "MONTH" | "DAY" | "HOURS" | "MINUTES" | "SECONDS", RDFLiteral[x_String, "http://www.w3.org/2001/XMLSchema#dateTime"]] := evalString[fn, DateObject[x]];


(* 17.4.6 Hash Functions *)
evalString[type : "MD5" | "SHA1" | "SHA256" | "SHA384" | "SHA512", s_] := Hash[lexicalForm[s], type, "HexString"];

evalString[f_String, args___] := With[{upper = ToUpperCase[f]}, If[f === upper, $Failed, evalString[upper, args]]];

evalString[___] := $Failed;


(* 17.5 XPath Constructor Functions *)
evalIRI[IRI["http://www.w3.org/2001/XMLSchema#boolean"], x_] := Interpreter["Boolean"][x] // Replace[_?FailureQ :> $Failed];
evalIRI[IRI["http://www.w3.org/2001/XMLSchema#double"], x_] := Interpreter["Real"][x] // Replace[_?FailureQ :> $Failed];
evalIRI[IRI["http://www.w3.org/2001/XMLSchema#float"], x_] := Interpreter["Real"][x] // Replace[_?FailureQ :> $Failed];
evalIRI[IRI["http://www.w3.org/2001/XMLSchema#decimal"], x_] := Interpreter["Real"][x] // Replace[_?FailureQ :> $Failed];
evalIRI[IRI["http://www.w3.org/2001/XMLSchema#integer"], x_] := Interpreter["Integer"][x] // Replace[_?FailureQ :> $Failed];


(* XPath functions *)
(* https://www.w3.org/TR/xpath-functions-31/ *)

(* 5.3.6 fn:compare *)
evalIRI[IRI["http://www.w3.org/2005/xpath-functions#compare"], a_String, b_String] := -Order[a, b];
(* 7.3.1 fn:boolean *)
evalIRI[IRI["http://www.w3.org/2005/xpath-functions#boolean"], x_] := Switch[x,
	_?BooleanQ, x,
	"", False,
	_String, True,
	0 | 0., False,
	_Integer | _Real, True,
	_, $Failed
];
(* 7.3.2 fn:not *)
evalIRI[IRI["http://www.w3.org/2005/xpath-functions#not"], x_] := evalIRI[IRI["http://www.w3.org/2005/xpath-functions#boolean"], x] // Replace[b_?BooleanQ :> ! b];

evalIRI[___] := $Failed;


lexicalForm[s_String] := s;
lexicalForm[RDFString[s_String, _]] := s;
lexicalForm[RDFLiteral[s_String, _]] := s;
lexicalForm[x_] := ToString[x];

restoreForm[_String][s_] := s;
restoreForm[(head : RDFString | RDFLiteral)[_, arg_]][s_] := head[s, arg];

End[];
EndPackage[];
