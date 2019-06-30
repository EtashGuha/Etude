(* SPARQL 1.1 Query Results JSON Format *)
(* https://www.w3.org/TR/sparql11-results-json/ *)

BeginPackage["GraphStore`Formats`SPARQLResultsJSON`", {"GraphStore`"}];

Needs["GraphStore`RDF`"];

ExportSPARQLResultsJSON;
ImportSPARQLResultsJSON;

Begin["`Private`"];

ExportSPARQLResultsJSON[args___] := Catch[iExportSPARQLResultsJSON[args], $failTag, (Message[Export::fmterr, "SPARQLResultsJSON"]; #) &];
ImportSPARQLResultsJSON[file_, opts : OptionsPattern[]] := Catch[iImportSPARQLResultsJSON[file, FilterRules[{opts}, Options[ImportSPARQLResultsJSON]]], $failTag, (Message[Import::fmterr, "SPARQLResultsJSON"]; #) &];


fail[___] := Throw[$Failed, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


(* -------------------------------------------------- *)
(* export *)

clear[iExportSPARQLResultsJSON];
iExportSPARQLResultsJSON[file_, data_List, OptionsPattern[]] := exportSelect[file, data];
iExportSPARQLResultsJSON[file_, data_?BooleanQ, OptionsPattern[]] := exportAsk[file, data];

clear[exportSelect];
exportSelect[file_, l : {___?AssociationQ}] := Export[
	file,
	<|
		"head" -> <|
			"vars" -> DeleteDuplicates[Join @@ Keys[l]]
		|>,
		"results" -> <|
			"bindings" -> Map[
				encodeTerm,
				l,
				{2}
			]
		|>
	|>,
	"RawJSON",
	"Compact" -> 4
];

clear[encodeTerm];
encodeTerm[(URL | IRI)[i_?StringQ]] := <|"type" -> "uri", "value" -> i|>;
encodeTerm[s_?StringQ] := <|"type" -> "literal", "value" -> s|>;
encodeTerm[RDFLiteral[s_?StringQ, "http://www.w3.org/2001/XMLSchema#string"]] := encodeTerm[s];
encodeTerm[RDFString[s_?StringQ, lang_?StringQ]] := <|"type" -> "literal", "value" -> s, "xml:lang" -> lang|>;
encodeTerm[RDFLiteral[s_?StringQ, dt_?StringQ]] := <|"type" -> "literal", "value" -> s, "datatype" -> dt|>;
encodeTerm[RDFLiteral[s_, (URL | IRI)[dt_?StringQ]]] := encodeTerm[RDFLiteral[s, dt]];
encodeTerm[RDFBlankNode[l_?StringQ]] := <|"type" -> "bnode", "value" -> l|>;
encodeTerm[x_] := encodeTerm[ToRDFLiteral[x] // Replace[_ToRDFLiteral :> fail[]]];

clear[exportAsk];
exportAsk[file_, b_?BooleanQ] := Export[file, <|"head" -> <||>, "boolean" -> b|>, "RawJSON"];

(* end export *)
(* -------------------------------------------------- *)


(* -------------------------------------------------- *)
(* import *)

clear[iImportSPARQLResultsJSON];
iImportSPARQLResultsJSON[file_, OptionsPattern[]] := {"Data" -> (Quiet[Import[file, "RawJSON"]] // Replace[{
	KeyValuePattern[{"head" -> KeyValuePattern["vars" -> vars_List], "results" -> KeyValuePattern["bindings" -> bindings_List]}] :> Replace[
		bindings,
		{
			KeyValuePattern[{"type" -> "uri", "value" -> iri_String}] :> IRI[iri],
			KeyValuePattern[{"type" -> "literal", "value" -> value_String, "xml:lang" -> lang_String}] :> RDFString[value, lang],
			(* support "typed-literal" during import only *)
			(* https://www.w3.org/2013/sparql-errata#sparql11-results-json *)
			KeyValuePattern[{"type" -> "literal" | "typed-literal", "value" -> value_String, "datatype" -> dt_String}] :> FromRDFLiteral[RDFLiteral[value, dt]],
			KeyValuePattern[{"type" -> "literal", "value" -> value_String}] :> value,
			KeyValuePattern[{"type" -> "bnode", "value" -> name_String}] :> RDFBlankNode[name],
			_ :> fail[]
		},
		{2}
	],
	KeyValuePattern["boolean" -> boolean : False | True] :> boolean,
	_ :> fail[]
}])};

(* end import *)
(* -------------------------------------------------- *)


End[];
EndPackage[];
