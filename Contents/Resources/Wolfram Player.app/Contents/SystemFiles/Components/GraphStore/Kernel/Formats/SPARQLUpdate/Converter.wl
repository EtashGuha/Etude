(* SPARQL 1.1 Update *)
(* https://www.w3.org/TR/sparql11-update/ *)

BeginPackage["GraphStore`Formats`SPARQLUpdate`", {"GraphStore`"}];

ExportSPARQLUpdate;
Options[ExportSPARQLUpdate] = {
	"Base" -> None,
	"Indentation" -> "  ",
	"Prefixes" -> <||>
};

ImportSPARQLUpdate;
Options[ImportSPARQLUpdate] = {
	"Base" -> Automatic
};

ImportSPARQLUpdateBase;
ImportSPARQLUpdatePrefixes;

Begin["GraphStore`Formats`SPARQLQuery`Private`"]; (* use private functions from SPARQLQuery/Converter.wl *)

ExportSPARQLUpdate[args___] := Catch[iExportSPARQLUpdate[args], $failTag, (Message[Export::fmterr, "SPARQLUpdate"]; #) &];
ImportSPARQLUpdate[file_, opts : OptionsPattern[]] := Catch[iImportSPARQLUpdate[file, FilterRules[{opts}, Options[ImportSPARQLUpdate]]], $failTag, (Message[Import::fmterr, "SPARQLUpdate"]; #) &];
ImportSPARQLUpdateBase[file_, opts : OptionsPattern[]] := {"Base" -> Replace[Import[file, "SPARQLUpdate", opts], {SPARQLUpdate[__, "Base" -> base_, ___] :> base, _ :> None}]};
ImportSPARQLUpdatePrefixes[file_, opts : OptionsPattern[]] := Catch[iImportSPARQLUpdatePrefixes[file, FilterRules[{opts}, Options[ImportSPARQLUpdatePrefixes]]], $failTag, (Message[Import::fmterr, "SPARQLUpdate"]; #) &];

End[];
EndPackage[];
