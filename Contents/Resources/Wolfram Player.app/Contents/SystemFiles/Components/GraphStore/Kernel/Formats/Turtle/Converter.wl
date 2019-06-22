(* RDF 1.1 Turtle *)
(* https://www.w3.org/TR/turtle/ *)

BeginPackage["GraphStore`Formats`Turtle`", {"GraphStore`"}];

Needs["GraphStore`Formats`Utilities`"];
Needs["GraphStore`Libraries`SerdLink`"];
Needs["GraphStore`RDF`"];

ExportTurtle;
Options[ExportTurtle] = {
	"Base" -> None,
	"Indentation" -> "  ",
	"Prefixes" -> <||>
};

ImportTurtle;
Options[ImportTurtle] = {
	"Base" -> Automatic
};

ImportTurtleBase;

ImportTurtlePrefixes;

Begin["`Private`"];

ExportTurtle[file_, data_, opts : OptionsPattern[]] := Quiet[Export[file, data, "TriG", opts]] // Replace[f_?FailureQ :> (Message[Export::fmterr, "Turtle"]; f)];
ImportTurtle[file_, opts : OptionsPattern[]] := Catch[iImportTurtle[file, FilterRules[{opts}, Options[ImportTurtle]]], $failTag, (Message[Import::fmterr, "Turtle"]; #) &];
ImportTurtleBase[file_, opts : OptionsPattern[]] := {"Base" -> Import[file, {"TriG", "Base"}, opts]};
ImportTurtlePrefixes[file_, opts : OptionsPattern[]] := {"Prefixes" -> Import[file, {"TriG", "Prefixes"}, opts]};


fail[___] := Throw[$Failed, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


(* -------------------------------------------------- *)
(* import *)

clear[iImportTurtle];
Options[iImportTurtle] = Options[ImportTurtle];
iImportTurtle[file_, OptionsPattern[]] := {
	"Data" -> Replace[
		SerdImport[file, 1 (* SERD_TURTLE *), ChooseBase[OptionValue["Base"], file] // Replace[None :> ""]],
		{
			_?FailureQ :> fail[],
			l_List :> RDFStore[l /. lit_RDFLiteral :> FromRDFLiteral[lit]]
		}
	]
};

(* end import *)
(* -------------------------------------------------- *)


End[];
EndPackage[];
