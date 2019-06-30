BeginPackage["GraphStore`RDF`Literal`", {"GraphStore`", "GraphStore`RDF`"}];

Needs["GraphStore`LanguageTag`"];

Begin["`Private`"];

DatatypeIRI[args___] := With[{res = Catch[iDatatypeIRI[args], $failTag]}, res /; res =!= $failTag];
FromRDFLiteral[args___] := With[{res = Catch[iFromRDFLiteral[args], $failTag]}, res /; res =!= $failTag];
LexicalForm[args___] := With[{res = Catch[iLexicalForm[args], $failTag]}, res /; res =!= $failTag];
ToRDFLiteral[args___] := With[{res = Catch[iToRDFLiteral[args], $failTag]}, res /; res =!= $failTag];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


(* -------------------------------------------------- *)
(* vocabulary *)

clear[rdf];
rdf[s_String] := "http://www.w3.org/1999/02/22-rdf-syntax-ns#" <> s;

clear[xsd];
xsd[s_String] := "http://www.w3.org/2001/XMLSchema#" <> s;

clear[geo];
geo[s_String] := "http://www.opengis.net/ont/geosparql#" <> s;

(* end vocabulary *)
(* -------------------------------------------------- *)


(* -------------------------------------------------- *)
(* datatype IRI *)

clear[iDatatypeIRI];
iDatatypeIRI[RDFLiteral[_, IRI[dt_String] | dt_String]] := IRI[dt];
iDatatypeIRI[_RDFString] := IRI[rdf["langString"]];
iDatatypeIRI[_String] := IRI[xsd["string"]];
iDatatypeIRI[_?BooleanQ] := IRI[xsd["boolean"]];
iDatatypeIRI[x_] := iDatatypeIRI[iToRDFLiteral[x]];

(* end datatype IRI *)
(* -------------------------------------------------- *)


(* -------------------------------------------------- *)
(* lexical form *)

clear[iLexicalForm];
iLexicalForm[RDFLiteral[s_String, _]] := s;
iLexicalForm[RDFString[s_String, _]] := s;
iLexicalForm[s_String] := s;
iLexicalForm[True] := "true";
iLexicalForm[False] := "false";
iLexicalForm[x_] := iLexicalForm[iToRDFLiteral[x]];

(* end lexical form *)
(* -------------------------------------------------- *)


(* -------------------------------------------------- *)
(* from RDF literal *)

(* https://www.w3.org/TR/rdf11-concepts/#section-Datatypes *)
(* 5. Datatypes *)
clear[iFromRDFLiteral];


(* 5.1 The XML Schema Built-in Datatypes *)

(* Core types *)
(* xsd:string *)
iFromRDFLiteral[RDFLiteral[s_String, xsd["string"]]] := s;
(* xsd:boolean *)
iFromRDFLiteral[RDFLiteral[s_String, xsd["boolean"]]] := Interpreter["Boolean"][s];
(* xsd:decimal *)
iFromRDFLiteral[l : RDFLiteral[s_String, xsd["decimal"]]] := If[StringMatchQ[s, NumberString],
	If[StringContainsQ[s, "."],
		Module[
			{i, f},
			{i, f} = StringSplit[StringReplace[s, {"." ~~ EndOfString :> ".0", StartOfString ~~ "." :> "0."}], ".", 2];
			i = ToExpression[i];
			i + If[NonNegative[i], 1, -1] * ToExpression[f] / 10^StringLength[f]
		],
		ToExpression[s]
	],
	l
];
(* xsd:integer *)
iFromRDFLiteral[l : RDFLiteral[s_String, xsd["integer"]]] := First[StringReplace[s, {
   StartOfString ~~ pm : Repeated["+" | "-", {0, 1}] ~~ d : DigitCharacter .. ~~ EndOfString :> If[{pm} === {"-"}, -1, 1] * FromDigits[d],
   ___ :> l
}, 1]];

(* IEEE floating-point numbers *)
(* xsd:double *)
iFromRDFLiteral[RDFLiteral[s_String, xsd["double"]]] := Interpreter["Real"][s];
(* xsd:float *)
iFromRDFLiteral[RDFLiteral[s_String, xsd["float"]]] := Interpreter["Real"][s];

(* Time and date *)

(* W3C XML Schema Definition Language (XSD) 1.1 Part 2: Datatypes *)
(* http://www.w3.org/TR/xmlschema11-2/ *)

(* [56] *) yearFrag := Repeated["-", {0, 1}] ~~ (CharacterRange["1", "9"] ~~ Repeated[DigitCharacter, {3, Infinity}]) | ("0" ~~ Repeated[DigitCharacter, {3}]);
(* [57] *) monthFrag := ("0" ~~ CharacterRange["1", "9"]) | ("1" ~~ CharacterRange["0", "2"]);
(* [58] *) dayFrag := ("0" ~~ CharacterRange["1", "9"]) | ("1" | "2" ~~ DigitCharacter) | ("3" ~~ "0" | "1");
(* [59] *) hourFrag := ("0" | "1" ~~ DigitCharacter) | ("2" ~~ CharacterRange["0", "3"]);
(* [60] *) minuteFrag := CharacterRange["0", "5"] ~~ DigitCharacter;
(* [61] *) secondFrag := CharacterRange["0", "5"] ~~ DigitCharacter ~~ Repeated["." ~~ DigitCharacter .., {0, 1}];
(* [62] *) endOfDayFrag := "24:00:00" ~~ Repeated["." ~~ "0" .., {0, 1}];
(* [63] *) timezoneFrag := "Z" | ("+" | "-" ~~ (("0" ~~ DigitCharacter) | ("1" ~~ CharacterRange["0", "3"]) ~~ ":" ~~ minuteFrag) | "14:00");

clear[timezoneFragToTimeZone];
timezoneFragToTimeZone["Z"] := 0;
timezoneFragToTimeZone[tz_String] := # + Sign[#] * #2 / 60. & @@ Interpreter["Integer"][StringSplit[tz, ":"]];

(* xsd:date *)
iFromRDFLiteral[l : RDFLiteral[s_String, xsd["date"]]] := First[StringReplace[s, {
	StartOfString ~~ year : yearFrag ~~ "-" ~~ month : monthFrag ~~ "-" ~~ day : dayFrag ~~ tz : Repeated[timezoneFrag, {0, 1}] ~~ EndOfString :> DateObject[
		Interpreter["Integer"][{year, month, day}],
		TimeZone -> If[tz === "", None, timezoneFragToTimeZone[tz]]
	],
	___ :> l
}, 1]];
(* xsd:time *)
iFromRDFLiteral[l : RDFLiteral[s_String, xsd["time"]]] := First[StringReplace[s, {
	StartOfString ~~ time : (hourFrag ~~ ":" ~~ minuteFrag ~~ ":" ~~ secondFrag) | endOfDayFrag ~~ tz : Repeated[timezoneFrag, {0, 1}] ~~ EndOfString :> TimeObject[
		Interpreter["Number"][StringSplit[time, ":"]],
		TimeZone -> If[tz === "", None, timezoneFragToTimeZone[tz]]
	],
	___ :> l
}, 1]];
(* xsd:dateTime *)
iFromRDFLiteral[l : RDFLiteral[s_String, xsd["dateTime"]]] := First[StringReplace[s, {
	StartOfString ~~
	year : yearFrag ~~ "-" ~~ month : monthFrag ~~ "-" ~~ day : dayFrag ~~
	"T" ~~
	time : (hourFrag ~~ ":" ~~ minuteFrag ~~ ":" ~~ secondFrag) | endOfDayFrag ~~
	tz : Repeated[timezoneFrag, {0, 1}] ~~
	EndOfString :> DateObject[
		Join[
			Interpreter["Integer"][{year, month, day}],
			Interpreter["Number"][StringSplit[time, ":"]]
		],
		TimeZone -> If[tz === "", None, timezoneFragToTimeZone[tz]]
	],
	___ :> l
}, 1]];
(* xsd:dateTimeStamp *)
iFromRDFLiteral[l : RDFLiteral[s_String, xsd["dateTimeStamp"]]] := First[StringReplace[s, {
	StartOfString ~~
	year : yearFrag ~~ "-" ~~ month : monthFrag ~~ "-" ~~ day : dayFrag ~~
	"T" ~~
	time : (hourFrag ~~ ":" ~~ minuteFrag ~~ ":" ~~ secondFrag) | endOfDayFrag ~~
	tz : timezoneFrag ~~
	EndOfString :> DateObject[
		Join[
			Interpreter["Integer"][{year, month, day}],
			Interpreter["Number"][StringSplit[time, ":"]]
		],
		TimeZone -> timezoneFragToTimeZone[tz]
	],
	___ :> l
}, 1]];

(* Recurring and partial dates *)
(* xsd:gYear *)
iFromRDFLiteral[l : RDFLiteral[s_String, xsd["gYear"]]] := First[StringReplace[s, {
	StartOfString ~~ year : yearFrag ~~ tz : Repeated[timezoneFrag, {0, 1}] ~~ EndOfString :> DateObject[
		{Interpreter["Integer"][year]},
		TimeZone -> If[tz === "", None, timezoneFragToTimeZone[tz]]
	],
	___ :> l
}, 1]];
(* xsd:gMonth *)
(* xsd:gDay *)
(* xsd:gYearMonth *)
iFromRDFLiteral[l : RDFLiteral[s_String, xsd["gYearMonth"]]] := First[StringReplace[s, {
	StartOfString ~~ year : yearFrag ~~ "-" ~~ month : monthFrag ~~ tz : Repeated[timezoneFrag, {0, 1}] ~~ EndOfString :> DateObject[
		Interpreter["Integer"][{year, month}],
		TimeZone -> If[tz === "", None, timezoneFragToTimeZone[tz]]
	],
	___ :> l
}, 1]];
(* xsd:gMonthDay *)
(* xsd:duration *)
iFromRDFLiteral[l : RDFLiteral[s_String, xsd["duration"]]] := First[StringReplace[s, {
	StartOfString ~~
	minus : Repeated["-", {0, 1}] ~~
	"P" ~~
	years : Repeated[NumberString ~~ "Y", {0, 1}] ~~
	months : Repeated[NumberString ~~ "M", {0, 1}] ~~
	days : Repeated[NumberString ~~ "D", {0, 1}] ~~
	Repeated[
		"T" ~~
		hours : Repeated[NumberString ~~ "H", {0, 1}] ~~
		minutes : Repeated[NumberString ~~ "M", {0, 1}] ~~
		seconds : Repeated[NumberString ~~ "S", {0, 1}],
		{0, 1}
	] ~~
	EndOfString :> With[
		{components = Cases[
			{{years, "Years"}, {months, "Months"}, {days, "Days"}, {hours, "Hours"}, {minutes, "Minutes"}, {seconds, "Seconds"}},
			{Except["", mag_], unit_} :> {Interpreter[Restricted["Number", {0, Infinity}]][StringDrop[mag, -1]], unit}
		]},
		If[NoneTrue[components[[All, 1]], FailureQ],
			If[minus === "", 1, -1] * Quantity[
				MixedMagnitude[components[[All, 1]]],
				MixedUnit[components[[All, 2]]]
			],
			l
		]
	],
	___ :> l
}, 1]];
(* xsd:yearMonthDuration *)
iFromRDFLiteral[l : RDFLiteral[s_String, xsd["yearMonthDuration"]]] := First[StringReplace[s, {
	StartOfString ~~
	minus : Repeated["-", {0, 1}] ~~
	"P" ~~
	years : Repeated[NumberString ~~ "Y", {0, 1}] ~~
	months : Repeated[NumberString ~~ "M", {0, 1}] ~~
	EndOfString :> With[
		{components = Cases[
			{{years, "Years"}, {months, "Months"}},
			{Except["", mag_], unit_} :> {Interpreter[Restricted["Number", {0, Infinity}]][StringDrop[mag, -1]], unit}
		]},
		If[NoneTrue[components[[All, 1]], FailureQ],
			If[minus === "", 1, -1] * Quantity[
				MixedMagnitude[components[[All, 1]]],
				MixedUnit[components[[All, 2]]]
			],
			l
		]
	],
	___ :> l
}, 1]];
(* xsd:dayTimeDuration *)
iFromRDFLiteral[l : RDFLiteral[s_String, xsd["dayTimeDuration"]]] := First[StringReplace[s, {
	StartOfString ~~
	minus : Repeated["-", {0, 1}] ~~
	"P" ~~
	days : Repeated[NumberString ~~ "D", {0, 1}] ~~
	Repeated[
		"T" ~~
		hours : Repeated[NumberString ~~ "H", {0, 1}] ~~
		minutes : Repeated[NumberString ~~ "M", {0, 1}] ~~
		seconds : Repeated[NumberString ~~ "S", {0, 1}],
		{0, 1}
	] ~~
	EndOfString :> With[
		{components = Cases[
			{{days, "Days"}, {hours, "Hours"}, {minutes, "Minutes"}, {seconds, "Seconds"}},
			{Except["", mag_], unit_} :> {Interpreter[Restricted["Number", {0, Infinity}]][StringDrop[mag, -1]], unit}
		]},
		If[NoneTrue[components[[All, 1]], FailureQ],
			If[minus === "", 1, -1] * Quantity[
				MixedMagnitude[components[[All, 1]]],
				MixedUnit[components[[All, 2]]]
			],
			l
		]
	],
	___ :> l
}, 1]];

(* Limited-range integer numbers *)
(* xsd:byte *)
iFromRDFLiteral[l : RDFLiteral[s_String, xsd["byte"]]] := Interpreter[Restricted["Integer", {-128, 127}]][s] // Replace[_?FailureQ :> l];
(* xsd:short *)
iFromRDFLiteral[l : RDFLiteral[s_String, xsd["short"]]] := Interpreter[Restricted["Integer", {-32768, 32767}]][s] // Replace[_?FailureQ :> l];
(* xsd:int *)
iFromRDFLiteral[l : RDFLiteral[s_String, xsd["int"]]] := Interpreter[Restricted["Integer", {-2147483648, 2147483647}]][s] // Replace[_?FailureQ :> l];
(* xsd:long *)
iFromRDFLiteral[l : RDFLiteral[s_String, xsd["long"]]] := Interpreter[Restricted["Integer", {-9223372036854775808, 9223372036854775807}]][s] // Replace[_?FailureQ :> l];
(* xsd:unsignedByte *)
iFromRDFLiteral[l : RDFLiteral[s_String, xsd["unsignedByte"]]] := Interpreter[Restricted["Integer", {0, 255}]][s] // Replace[_?FailureQ :> l];
(* xsd:unsignedShort *)
iFromRDFLiteral[l : RDFLiteral[s_String, xsd["unsignedShort"]]] := Interpreter[Restricted["Integer", {0, 65535}]][s] // Replace[_?FailureQ :> l];
(* xsd:unsignedInt *)
iFromRDFLiteral[l : RDFLiteral[s_String, xsd["unsignedInt"]]] := Interpreter[Restricted["Integer", {0, 4294967295}]][s] // Replace[_?FailureQ :> l];
(* xsd:unsignedLong *)
iFromRDFLiteral[l : RDFLiteral[s_String, xsd["unsignedLong"]]] := Interpreter[Restricted["Integer", {0, 18446744073709551615}]][s] // Replace[_?FailureQ :> l];
(* xsd:positiveInteger *)
iFromRDFLiteral[l : RDFLiteral[s_String, xsd["positiveInteger"]]] := Interpreter[Restricted["Integer", {1, Infinity}]][s] // Replace[_?FailureQ :> l];
(* xsd:nonNegativeInteger *)
iFromRDFLiteral[l : RDFLiteral[s_String, xsd["nonNegativeInteger"]]] := Interpreter[Restricted["Integer", {0, Infinity}]][s] // Replace[_?FailureQ :> l];
(* xsd:negativeInteger *)
iFromRDFLiteral[l : RDFLiteral[s_String, xsd["negativeInteger"]]] := Interpreter[Restricted["Integer", {-Infinity, -1}]][s] // Replace[_?FailureQ :> l];
(* xsd:nonPositiveInteger *)
iFromRDFLiteral[l : RDFLiteral[s_String, xsd["nonPositiveInteger"]]] := Interpreter[Restricted["Integer", {-Infinity, 0}]][s] // Replace[_?FailureQ :> l];

(* Encoded binary data *)
(* xsd:hexBinary *)
iFromRDFLiteral[l : RDFLiteral[s_String, xsd["hexBinary"]]] := If[Divisible[StringLength[s], 2], ByteArray[IntegerDigits[FromDigits[s, 16], 256, StringLength[s] / 2]], l];
(* xsd:base64Binary *)
iFromRDFLiteral[l : RDFLiteral[s_String, xsd["base64Binary"]]] := BaseDecode[s];

(* Miscellaneous XSD types *)
(* xsd:anyURI *)
(* xsd:language *)
(* xsd:normalizedString *)
(* xsd:token *)
(* xsd:NMTOKEN *)
(* xsd:Name *)
(* xsd:NCName *)


(* 5.2 The rdf:HTML Datatype *)


(* 5.3 The rdf:XMLLiteral Datatype *)
iFromRDFLiteral[l : RDFLiteral[s_String, "http://www.w3.org/1999/02/22-rdf-syntax-ns#XMLLiteral"]] := ImportString["<x>" <> s <> "</x>", "XML"] // Replace[{
	XMLObject["Document"][{}, XMLElement[_, _, {e_}], {}] :> e,
	___ :> l
}];


(* rdf:PlainLiteral: A Datatype for RDF Plain Literals (Second Edition) *)
(* https://www.w3.org/TR/rdf-plain-literal/ *)
iFromRDFLiteral[l : RDFLiteral[s_String, rdf["PlainLiteral"]]] := Which[
	StringFreeQ[s, "@"],
	l,
	StringEndsQ[s, "@"],
	StringDrop[s, -1],
	True,
	First[StringReplace[
		s,
		StartOfString ~~ a___ ~~ "@" ~~ Shortest[b__] ~~ EndOfString :> If[LanguageTagQ[b],
			RDFString[a, ToLowerCase[b]],
			l
		],
		1
	]]
];


(* OGC GeoSPARQL - A Geographic Query Language for RDF Data *)
(* http://www.opengis.net/doc/IS/geosparql/1.0 *)

(* Simple Feature Access - Part 1: Common Architecture *)
(* http://www.opengeospatial.org/standards/sfa *)
(* examples on page 60 *)

iFromRDFLiteral[l : RDFLiteral[s_String, geo["wktLiteral"]]] := With[
	{pair = WhitespaceCharacter ... ~~ NumberString ~~ Whitespace ~~ NumberString ~~ WhitespaceCharacter ...},
	With[
		{pairList = (pair ~~ "," ~~ WhitespaceCharacter ...) ... ~~ pair},
		First[StringReplace[s, {
			(* point *)
			StartOfString ~~ "Point(" ~~ point : pair ~~ ")" ~~ EndOfString :> Point[GeoPosition[
				Reverse[ToExpression /@ StringSplit[StringTrim[point]]]
			]],
			(* line string *)
			StartOfString ~~ "LineString(" ~~ line : pairList ~~ ")" ~~ EndOfString :> Line[GeoPosition[
				Reverse[Map[ToExpression, StringSplit[StringTrim[StringSplit[line, ","]]], {2}], 2]
			]],
			(* polygon *)
			StartOfString ~~ "Polygon(" ~~ "(" ~~ line : pairList ~~ ")" ~~ ")" ~~ EndOfString :> Polygon[GeoPosition[
				Reverse[Map[ToExpression, StringSplit[StringTrim[StringSplit[line, ","]]], {2}], 2]
			]],
			___ :> l
		}, 1, IgnoreCase -> True]]
	]
];


iFromRDFLiteral[RDFLiteral[s_String, IRI[i_String]]] := iFromRDFLiteral[RDFLiteral[s, i]];
iFromRDFLiteral[l : RDFLiteral[_String, _String]] := l;

(* end from RDF literal *)
(* -------------------------------------------------- *)


(* -------------------------------------------------- *)
(* to RDF literal *)

(* https://www.w3.org/TR/rdf11-concepts/#xsd-datatypes *)
(* 5. Datatypes *)
clear[iToRDFLiteral];


(* 5.1 The XML Schema Built-in Datatypes *)

(* Core types *)
iToRDFLiteral[s_String] := RDFLiteral[s, xsd["string"]];
iToRDFLiteral[True] := RDFLiteral["true", xsd["boolean"]];
iToRDFLiteral[False] := RDFLiteral["false", xsd["boolean"]];
iToRDFLiteral[r_Real?(Not @* MachineNumberQ)] := RDFLiteral[
	ToString[DecimalForm[r, Infinity]],
	xsd["decimal"]
];
iToRDFLiteral[r_Rational] := RDFLiteral[
	StringDelete[ToString[DecimalForm[N[r, 100], Infinity]], "0" .. ~~ EndOfString],
	xsd["decimal"]
];
iToRDFLiteral[i_Integer] := RDFLiteral[ToString[i], xsd["integer"]];

(* IEEE floating-point numbers *)
iToRDFLiteral[r_Real?MachineNumberQ] := RDFLiteral[
	If[r == 0,
		"0.0E0",
		Module[
			{m, e},
			{m, e} = MantissaExponent[r];
			m *= 10;
			e--;
			ToString[m] <> "E" <> ToString[e]
		]
	],
	xsd["double"]
];

(* Time and date *)
iToRDFLiteral[d_?DateObjectQ] := With[
	{dateTimeQ = MatchQ[DateValue[d, "Granularity"], "Hour" | "Minute" | "Second" | "Instant"]},
	RDFLiteral[
		(* https://bugs.wolfram.com/show?number=339513 *)
		StringJoin[
			DateString[d, If[dateTimeQ, "ISODateTime", "ISODate"]],
			formatTimeZone[d]
		],
		xsd[If[dateTimeQ, "dateTime", "date"]]
	]
];
iToRDFLiteral[t_?TimeObjectQ] := RDFLiteral[
	StringJoin[
		Last[StringSplit[DateString[t, "ISODateTime"], "T", 2]],
		formatTimeZone[t]
	],
	xsd["time"]
];

clear[formatTimeZone];
formatTimeZone[d_] := Switch[DateValue[d, "TimeZone"],
	None, "",
	_?(EqualTo[0]), "Z",
	_?NumberQ, DateString[d, "ISOTimeZone"],
	_, fail[]
];

(* Recurring and partial dates *)

(* Limited-range integer numbers *)

(* Encoded binary data *)

(* Miscellaneous XSD types *)


(* 5.2 The rdf:HTML Datatype *)


(* 5.3 The rdf:XMLLiteral Datatype *)
iToRDFLiteral[e_XMLElement] := RDFLiteral[ExportString[e, "XML"], "http://www.w3.org/1999/02/22-rdf-syntax-ns#XMLLiteral"];


(* OGC GeoSPARQL - A Geographic Query Language for RDF Data *)
(* http://www.opengis.net/doc/IS/geosparql/1.0 *)

(* Simple Feature Access - Part 1: Common Architecture *)
(* http://www.opengeospatial.org/standards/sfa *)
(* examples on page 60 *)

(* point *)
iToRDFLiteral[HoldPattern[Point[GeoPosition[{lat_, lon_}]]]] := RDFLiteral["Point(" <> ToString[lon] <> " " <> ToString[lat] <> ")", geo["wktLiteral"]];
iToRDFLiteral[HoldPattern[pos_GeoPosition]] := iToRDFLiteral[Point[pos]];
(* line string *)
iToRDFLiteral[HoldPattern[Line[line : {GeoPosition[{_, _}] ..}]]] := RDFLiteral["LineString(" <> StringRiffle[Reverse[line[[All, 1]], 2], ", ", " "] <> ")", geo["wktLiteral"]];
iToRDFLiteral[HoldPattern[Line[GeoPosition[line : {{_, _} ..}]]]] := iToRDFLiteral[Line[GeoPosition /@ line]];
(* polygon *)
iToRDFLiteral[HoldPattern[Polygon[line : {GeoPosition[{_, _}] ..}]]] := RDFLiteral["Polygon((" <> StringRiffle[Reverse[line[[All, 1]], 2], ", ", " "] <> "))", geo["wktLiteral"]];
iToRDFLiteral[HoldPattern[Polygon[GeoPosition[polygon : {{_, _} ..}]]]] := iToRDFLiteral[Polygon[GeoPosition /@ polygon]];

(* end to RDF literal *)
(* -------------------------------------------------- *)


End[];
EndPackage[];
