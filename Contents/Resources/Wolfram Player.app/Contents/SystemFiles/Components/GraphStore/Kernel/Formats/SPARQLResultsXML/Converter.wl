(* SPARQL Query Results XML Format (Second Edition) *)
(* https://www.w3.org/TR/rdf-sparql-XMLres/ *)

BeginPackage["GraphStore`Formats`SPARQLResultsXML`", {"GraphStore`"}];

Needs["GraphStore`RDF`"];

ExportSPARQLResultsXML;
ImportSPARQLResultsXML;

Begin["`Private`"];

ExportSPARQLResultsXML[args___] := Catch[iExportSPARQLResultsXML[args], $failTag, (Message[Export::fmterr, "SPARQLResultsXML"]; #) &];
ImportSPARQLResultsXML[file_, opts : OptionsPattern[]] := Catch[iImportSPARQLResultsXML[file, FilterRules[{opts}, Options[ImportSPARQLResultsXML]]], $failTag, (Message[Import::fmterr, "SPARQLResultsXML"]; #) &];


fail[___] := Throw[$Failed, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


(* -------------------------------------------------- *)
(* export *)

clear[iExportSPARQLResultsXML];
iExportSPARQLResultsXML[file_, data_List, OptionsPattern[]] := exportSelect[file, data];
iExportSPARQLResultsXML[file_, data_?BooleanQ, OptionsPattern[]] := exportAsk[file, data];

clear[exportSelect];
exportSelect[file_, l : {___?AssociationQ}] := Export[
	file,
	XMLObject["Document"][
		{XMLObject["Declaration"]["Version" -> "1.0"]},
		XMLElement[
			"sparql",
			{"xmlns" -> "http://www.w3.org/2005/sparql-results#"},
			{
				XMLElement[
					"head",
					{},
					Function[var, XMLElement["variable", {"name" -> var}, {}]] /@ DeleteDuplicates[Join @@ Keys[l]]
				],
				XMLElement[
					"results",
					{},
					Function[sol,
						XMLElement[
							"result",
							{},
							KeyValueMap[
								Function[{var, term},
									XMLElement[
										"binding",
										{"name" -> var},
										{encodeTerm[term]}
									]
								],
								sol
							]
						]
					] /@ l
				]
			}
		],
		{}
	],
	"XML"
];

clear[encodeTerm];
encodeTerm[(URL | IRI)[i_?StringQ]] := XMLElement["uri", {}, {i}];
encodeTerm[s_?StringQ] := XMLElement["literal", {}, {s}];
encodeTerm[RDFLiteral[s_?StringQ, "http://www.w3.org/2001/XMLSchema#string"]] := encodeTerm[s];
encodeTerm[RDFString[s_?StringQ, lang_?StringQ]] := XMLElement["literal", {"xml:lang" -> lang}, {s}];
encodeTerm[RDFLiteral[s_?StringQ, dt_?StringQ]] := XMLElement["literal", {"datatype" -> dt}, {s}];
encodeTerm[RDFLiteral[s_, (URL | IRI)[dt_?StringQ]]] := encodeTerm[RDFLiteral[s, dt]];
encodeTerm[RDFBlankNode[l_?StringQ]] := XMLElement["bnode", {}, {l}];
encodeTerm[x_] := encodeTerm[ToRDFLiteral[x] // Replace[_ToRDFLiteral :> fail[]]];

clear[exportAsk];
exportAsk[file_, b_?BooleanQ] := Export[
	file,
	XMLObject["Document"][
		{XMLObject["Declaration"]["Version" -> "1.0"]},
		XMLElement[
			"sparql",
			{"xmlns" -> "http://www.w3.org/2005/sparql-results#"},
			{
				XMLElement["head", {}, {}],
				XMLElement["boolean", {}, {booleanToString[b]}]
			}
		],
		{}
	],
	"XML"
];

clear[booleanToString];
booleanToString[False] := "false";
booleanToString[True] := "true";

(* end export *)
(* -------------------------------------------------- *)


(* -------------------------------------------------- *)
(* import *)

clear[iImportSPARQLResultsXML];
iImportSPARQLResultsXML[file_, OptionsPattern[]] := {"Data" -> importDocument[Quiet[Import[file, "XML"]] // Replace[_?FailureQ :> fail[]]]};

clear[importDocument];
importDocument[XMLObject["Document"][_, sparql_, _]] := importSparql[sparql];

clear[importSparql];
importSparql[XMLElement["sparql", _, {XMLElement["head", _, _], XMLElement["results", _, results_List]}]] := importResult /@ results;
importSparql[XMLElement["sparql", _, {XMLElement["head", _, _], XMLElement["boolean", _, {boolean_String}]}]] := Interpreter["Boolean"][boolean];

clear[importResult];
importResult[XMLElement["result", _, bindings_List]] := <|importBinding /@ bindings|>;

clear[importBinding];
importBinding[XMLElement["binding", KeyValuePattern["name" -> name_String], {term_}]] := name -> importTerm[term];

clear[importTerm];
importTerm[XMLElement["uri", {}, {uri_String}]] := IRI[uri];
importTerm[XMLElement["literal", {}, {s_String : ""}]] := s;
importTerm[XMLElement["literal", {{"http://www.w3.org/XML/1998/namespace", "lang"} -> lang_String}, {s_String : ""}]] := RDFString[s, lang];
importTerm[XMLElement["literal", {"datatype" -> dt_String}, {s_String : ""}]] := FromRDFLiteral[RDFLiteral[s, dt]];
importTerm[XMLElement["bnode", {}, {s_String : ""}]] := RDFBlankNode[s];

(* end import *)
(* -------------------------------------------------- *)


End[];
EndPackage[];
