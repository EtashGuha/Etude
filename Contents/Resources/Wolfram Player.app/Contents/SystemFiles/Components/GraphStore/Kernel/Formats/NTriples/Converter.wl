(* RDF 1.1 N-Triples *)
(* https://www.w3.org/TR/n-triples/ *)

BeginPackage["GraphStore`Formats`NTriples`", {"GraphStore`"}];

Needs["GraphStore`Formats`Utilities`"];
Needs["GraphStore`Libraries`SerdLink`"];
Needs["GraphStore`RDF`"];

ExportNTriples;
ImportNTriples;

Begin["`Private`"];

ExportNTriples[file_, data_, opts : OptionsPattern[]] := Quiet[Export[file, data, "NQuads", opts]] // Replace[f_?FailureQ :> (Message[Export::fmterr, "NTriples"]; f)];
ImportNTriples[file_, opts : OptionsPattern[]] := Catch[iImportNTriples[file, FilterRules[{opts}, Options[ImportNTriples]]], $failTag, (Message[Import::fmterr, "NTriples"]; #) &];


fail[___] := Throw[$Failed, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


(* -------------------------------------------------- *)
(* import *)

clear[iImportNTriples];
iImportNTriples[file_, OptionsPattern[]] := {
	"Data" -> Replace[
		SerdImport[file, 2 (* SERD_NTRIPLES *), ""],
		{
			_?FailureQ :> fail[],
			l_List :> RDFStore[l /. {
				lit_RDFLiteral :> FromRDFLiteral[lit],
				s_String :> NumericDecode[StringDecode[s]]
			}]
		}
	]
};

(* end import *)
(* -------------------------------------------------- *)


End[];
EndPackage[];
