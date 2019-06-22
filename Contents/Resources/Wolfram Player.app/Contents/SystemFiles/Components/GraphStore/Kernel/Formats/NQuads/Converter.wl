(* RDF 1.1 N-Quads *)
(* https://www.w3.org/TR/n-quads/ *)

BeginPackage["GraphStore`Formats`NQuads`", {"GraphStore`"}];

Needs["GraphStore`Formats`Utilities`"];
Needs["GraphStore`IRI`"];
Needs["GraphStore`RDF`"];

ExportNQuads;

ImportNQuads;
Options[ImportNQuads] = {
	"ProduceGeneralizedRDF" -> False
};

Begin["`Private`"];

ExportNQuads[args___] := Catch[iExportNQuads[args], $failTag, (Message[Export::fmterr, "NQuads"]; #) &];
ImportNQuads[file_, opts : OptionsPattern[]] := Catch[iImportNQuads[file, FilterRules[{opts}, Options[ImportNQuads]]], $failTag, (Message[Import::fmterr, "NQuads"]; #) &];


fail[___] := Throw[$Failed, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


(* -------------------------------------------------- *)
(* export *)

clear[iExportNQuads];
iExportNQuads[file_, data_, OptionsPattern[]] := Export[
	file,
	nquadsDocToString[data],
	"Text",
	CharacterEncoding -> "UTF-8"
];


(* [1] *)
clear[nquadsDocToString];
nquadsDocToString[store_RDFStore] := StringRiffle[Flatten[{
	statementToString /@ store["DefaultGraph"],
	KeyValueMap[
		Function[{gl, g},
			statementToString[{#, gl}] & /@ g
		],
		store["NamedGraphs"]
	]
}], "\n"];

(* [2] *)
clear[statementToString];
statementToString[RDFTriple[s_, p_, o_]] := StringRiffle[{subjectToString[s], predicateToString[p], objectToString[o], "."}];
statementToString[{RDFTriple[s_, p_, o_], gl_}] := StringRiffle[{subjectToString[s], predicateToString[p], objectToString[o], graphLabelToString[gl], "."}];

(* [3] *)
clear[subjectToString];
subjectToString[i_IRI] := IRIREFToString[i];
subjectToString[b_RDFBlankNode] := BLANKNODELABELToString[b];

(* [4] *)
clear[predicateToString];
predicateToString[i_IRI] := IRIREFToString[i];

(* [5] *)
clear[objectToString];
objectToString[i_IRI] := IRIREFToString[i];
objectToString[b_RDFBlankNode] := BLANKNODELABELToString[b];
objectToString[x_] := literalToString[x];

(* [6] *)
clear[graphLabelToString];
graphLabelToString[i_IRI] := IRIREFToString[i];
graphLabelToString[b_RDFBlankNode] := BLANKNODELABELToString[b];

(* [7] *)
clear[literalToString];
literalToString[s_String] := STRINGLITERALQUOTEToString[s];
literalToString[RDFLiteral[s_, "http://www.w3.org/2001/XMLSchema#string"]] := STRINGLITERALQUOTEToString[s];
literalToString[RDFLiteral[s_, dt_?StringQ]] := STRINGLITERALQUOTEToString[s] <> "^^" <> IRIREFToString[IRI[dt]];
literalToString[RDFLiteral[s_, IRI[dt_]]] := literalToString[RDFLiteral[s, dt]];
literalToString[RDFString[s_, lang_?StringQ]] := STRINGLITERALQUOTEToString[s] <> "@" <> lang;
literalToString[x_] := literalToString[ToRDFLiteral[x] // Replace[_ToRDFLiteral :> fail[]]];


(* [10] *)
clear[IRIREFToString];
IRIREFToString[IRI[s_?StringQ]] := "<" <> s <> ">";

(* [11] *)
clear[STRINGLITERALQUOTEToString];
STRINGLITERALQUOTEToString[s_?StringQ] := "\"" <> (s // StringEncode) <> "\"";

(* [141s] *)
clear[BLANKNODELABELToString];
BLANKNODELABELToString[RDFBlankNode[s_?StringQ]] := "_:" <> s;

(* end export *)
(* -------------------------------------------------- *)


(* -------------------------------------------------- *)
(* import *)

clear[iImportNQuads];
Options[iImportNQuads] = Options[ImportNQuads];
iImportNQuads[stream_, OptionsPattern[]] := {"Data" -> (Flatten[StringCases[
	readList[stream],
	{
		StringExpression[
			StartOfString ~~ ws,
			(* subject *)
			(si : IRIREF) | (sb : BLANKNODELABEL),
			ws,
			(* predicate *)
			If[TrueQ[OptionValue["ProduceGeneralizedRDF"]],
				(pi : IRIREF) | (pb : BLANKNODELABEL),
				pi : IRIREF
			],
			ws,
			(* object *)
			(oi : IRIREF) | (ob : BLANKNODELABEL) | ((os : STRINGLITERALQUOTE) ~~ Repeated[((ox : "^^") ~~ (odt : IRIREF)) | (ol : LANGTAG), {0, 1}]),
			ws,
			(* graph *)
			Repeated[(gi : IRIREF) | (gb : BLANKNODELABEL), {0, 1}],
			ws ~~ "." ~~ ws ~~ comment ~~ EndOfString
		] :> {
			RDFTriple[
				importSubject[si, sb],
				If[TrueQ[OptionValue["ProduceGeneralizedRDF"]],
					importSubject[pi, pb],
					importIRI[pi]
				],
				importObject[oi, ob, {os, ox, odt, ol}]
			],
			importGraphLabel[gi, gb]
		},
		StartOfString ~~ ws ~~ comment ~~ EndOfString :> {},
		__ :> fail[]
	},
	1
], 1] // Function[l,
	RDFStore[
		l // Cases[{t_, "default"} :> t],
		l // Cases[{_, Except["default"]}] // GroupBy[Last -> First]
	]
])};

clear[readList];
readList[stream_] := ReadByteArray[stream] // Replace[{
	EndOfFile :> {},
	ba_ :> StringSplit[ByteArrayToString[ba], "\n" | "\r"]
}];


ws = (" " | "\t") ...;
comment = Repeated["#" ~~ Except["\n"] ..., {0, 1}];


clear[importSubject];
importSubject[Except["", i_], ___] := importIRI[i];
importSubject[_, Except["", b_], ___] := importBlankNode[b];

clear[importObject];
importObject[Except["", i_], ___] := importIRI[i];
importObject[_, Except["", b_], ___] := importBlankNode[b];
importObject[_, _, Except[{"", "", "", ""}, l_], ___] := importLiteral[l];

clear[importGraphLabel];
importGraphLabel[Except["", i_], ___] := importIRI[i];
importGraphLabel[_, Except["", b_], ___] := importBlankNode[b];
importGraphLabel[___] := "default";

clear[importIRI];
importIRI[s_] := With[
	{i = IRI[StringTake[s, {2, -2}] // NumericDecode]},
	If[! AbsoluteIRIQ[i],
		fail[]
	];
	i
];

clear[importBlankNode];
importBlankNode[b_] := RDFBlankNode[StringDrop[b, 2]];

clear[importString];
importString[s_] := StringTake[s, {2, -2}] // StringDecode // NumericDecode;

clear[importLiteral];
importLiteral[{s_, "", "", ""}] := importString[s];
importLiteral[{s_, "^^", dt_, ""}] := FromRDFLiteral[RDFLiteral[importString[s], First[importIRI[dt]]]];
importLiteral[{s_, "", "", lang_}] := RDFString[importString[s], StringDrop[lang, 1]];


(* terminals *)
(* [144s] *) LANGTAG := RegularExpression["@[a-zA-Z]+(-[a-zA-Z0-9]+)*"];
(* [8] *)    EOL := ("\n" | "\r") ..;
(* [10] *)   IRIREF := "<" ~~ (Except[" " | "<" | ">" | "\"" | "{" | "}" | "|" | "^" | "`" | "\\"] | UCHAR) ... ~~ ">";
(* [11] *)   STRINGLITERALQUOTE := "\"" ~~ (Except["\"" | "\\" | "\n" | "\r"] | ECHAR | UCHAR) ... ~~ "\"";
(* [141s] *) BLANKNODELABEL := "_:" ~~ PNCHARSU | DigitCharacter ~~ Repeated[(PNCHARS | ".") ... ~~ PNCHARS, {0, 1}];
(* [12] *)   UCHAR := ("\\u" ~~ Repeated[HEX, {4}]) | ("\\U" ~~ Repeated[HEX, {8}]);
(* [153s] *) ECHAR := "\\" ~~ "t" | "b" | "n" | "r" | "f" | "\"" | "'" | "\\";
(* [157s] *) PNCHARSBASE := LetterCharacter;
(* [158s] *) PNCHARSU := PNCHARSBASE | "_" | ":";
(* [160s] *) PNCHARS := PNCHARSU | "-" | DigitCharacter;
(* [162s] *) HEX := HexadecimalCharacter;

(* end import *)
(* -------------------------------------------------- *)


End[];
EndPackage[];
