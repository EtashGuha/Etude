(* Tags for Identifying Languages *)
(* https://tools.ietf.org/html/rfc5646 *)

(* to do: *)
(* GraphStore`LanguageTag`LanguageTagQ["ar-a-aaa-b-bbb-a-ccc"] should give False *)

BeginPackage["GraphStore`LanguageTag`"];

LanguageTagQ;

Begin["`Private`"];

LanguageTagQ[x_] := StringMatchQ[x, $languageTagPattern];

$languageTagPattern := Alternatives[
	langtag,
	privateuse,
	grandfathered
];

langtag := StringExpression[
	language,
	Repeated["-" ~~ script, {0, 1}],
	Repeated["-" ~~ region, {0, 1}],
	("-" ~~ variant) ...,
	("-" ~~ extension) ...,
	Repeated["-" ~~ privateuse, {0, 1}]
];

language := Alternatives[
	StringExpression[
		Repeated[LetterCharacter, {2, 3}],
		Repeated["-" ~~ extlang, {0, 1}]
	]
	(*Repeated[LetterCharacter, {4}]*)
	(*Repeated[LetterCharacter, {5, 8}]*)
];

extlang := StringExpression[
	Repeated[LetterCharacter, {3}],
	Repeated["-" ~~ Repeated[LetterCharacter, {3}], {0, 2}]
];

script := Repeated[LetterCharacter, {4}];

region := Alternatives[
	Repeated[LetterCharacter, {2}],
	Repeated[DigitCharacter, {3}]
];

variant := Alternatives[
	Repeated[WordCharacter, {5, 8}],
	DigitCharacter ~~ Repeated[WordCharacter, {3}]
];

extension := singleton ~~ ("-" ~~ Repeated[WordCharacter, {2, 8}]) ..;

singleton := Except["x" | "X", WordCharacter];

privateuse := "x" ~~ ("-" ~~ Repeated[WordCharacter, {1, 8}]) ..;

grandfathered := Alternatives[
	irregular,
	regular
];

irregular := Alternatives[
	"en-GB-oed",
	"i-ami",
	"i-bnn",
	"i-default",
	"i-enochian",
	"i-hak",
	"i-klingon",
	"i-lux",
	"i-mingo",
	"i-navajo",
	"i-pwn",
	"i-tao",
	"i-tay",
	"i-tsu",
	"sgn-BE-FR",
	"sgn-BE-NL",
	"sgn-CH-DE"
];

regular := Alternatives[
	"art-lojban",
	"cel-gaulish",
	"no-bok",
	"no-nyn",
	"zh-guoyu",
	"zh-hakka",
	"zh-min",
	"zh-min-nan",
	"zh-xiang"
];

End[];
EndPackage[];
