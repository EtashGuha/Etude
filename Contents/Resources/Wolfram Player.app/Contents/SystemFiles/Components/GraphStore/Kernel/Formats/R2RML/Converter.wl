(* R2RML: RDB to RDF Mapping Language *)
(* https://www.w3.org/TR/r2rml/ *)

BeginPackage["GraphStore`Formats`R2RML`", {"GraphStore`"}];

Needs["GraphStore`SPARQL`"];

ExportR2RML;

ImportR2RML;
Options[ImportR2RML] = {
	"Base" -> Automatic
};

Begin["`Private`"];

ExportR2RML[args___] := Catch[iExportR2RML[args], $failTag, (Message[Export::fmterr, "R2RML"]; #) &];
ImportR2RML[file_, opts : OptionsPattern[]] := Catch[iImportR2RML[file, FilterRules[{opts}, Options[ImportR2RML]]], $failTag, (Message[Import::fmterr, "R2RML"]; #) &];


fail[f_Failure] := Throw[{"Data" -> f}, $failTag];
fail[___] := Throw[$Failed, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


(* -------------------------------------------------- *)
(* common *)

clear[rdf];
rdf[s_String] := IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#" <> s];

clear[rr];
rr[s_String] := IRI["http://www.w3.org/ns/r2rml#" <> s];

(* end common *)
(* -------------------------------------------------- *)


(* -------------------------------------------------- *)
(* export *)

clear[iExportR2RML];
iExportR2RML[file_, data_, OptionsPattern[]] := Export[
	file,
	RDFStore[
		Block[
			{$blankNodeCounter = 0},
			First[Last[Reap[mappingToTriples[data], $tripleTag]], {}]
		]
	],
	"Turtle",
	"Prefixes" -> <|"rr" -> "http://www.w3.org/ns/r2rml#"|>
];


clear[nodeLookup];
nodeLookup[map_?AssociationQ] := Lookup[map, "Node", RDFBlankNode["b" <> ToString[$blankNodeCounter++]]];

clear[sowTriple];
sowTriple[s_, p_, o_] := Sow[RDFTriple[s, p, o], $tripleTag];


clear[mappingToTriples];
mappingToTriples[mapping_List] := triplesMapToTriples /@ mapping;


(* 5 Defining Logical Tables *)
clear[logicalTableToTriples];
logicalTableToTriples[lt_?AssociationQ] := With[
	{node = nodeLookup[lt]},
	sowTriple[node, rdf["type"], rr["LogicalTable"]];
	lt // Replace[{
		KeyValuePattern["TableName" -> tableName_String] :> (
			sowTriple[node, rdf["type"], rr["BaseTableOrView"]];
			sowTriple[node, rr["tableName"], tableName]
		),
		KeyValuePattern["SQLQuery" -> sqlQuery_String] :> (
			sowTriple[node, rdf["type"], rr["R2RMLView"]];
			sowTriple[node, rr["sqlQuery"], sqlQuery];
			lt // Replace[KeyValuePattern["SQLVersion" -> sqlVersion_IRI] :> sowTriple[node, rr["sqlVersion"], sqlVersion]];
		),
		_ :> fail[]
	}];
	node
];


(* 6 Mapping Logical Tables to RDF with Triples Maps *)
clear[triplesMapToTriples];
triplesMapToTriples[tm_?AssociationQ] := With[
	{node = nodeLookup[tm]},
	sowTriple[node, rdf["type"], rr["TriplesMap"]];
	sowTriple[node, rr["logicalTable"], logicalTableToTriples[tm["LogicalTable"]]];
	sowTriple[node, rr["subjectMap"], subjectMapToTriples[tm["SubjectMap"]]];
	Lookup[tm, "PredicateObjectMaps", {}] // Scan[Function[pom,
		sowTriple[node, rr["predicateObjectMap"], predicateObjectMapToTriples[pom]];
	]];
	node
];

clear[subjectMapToTriples];
subjectMapToTriples[sm_?AssociationQ] := With[
	{node = termMapToTriples[sm]},
	sowTriple[node, rdf["type"], rr["SubjectMap"]];
	Lookup[sm, "Classes", {}] // Scan[Function[class,
		sowTriple[node, rr["class"], class];
	]];
	Lookup[sm, "GraphMaps", {}] // Scan[Function[gm,
		sowTriple[node, rr["graphMap"], graphMapToTriples[gm]];
	]];
	node
];

clear[predicateObjectMapToTriples];
predicateObjectMapToTriples[pom_?AssociationQ] := With[
	{node = nodeLookup[pom]},
	sowTriple[node, rdf["type"], rr["PredicateObjectMap"]];
	Lookup[pom, "PredicateMaps", {}] // Scan[Function[pm,
		sowTriple[node, rr["predicateMap"], predicateMapToTriples[pm]];
	]];
	Lookup[pom, "ObjectMaps", {}] // Scan[Function[om,
		sowTriple[node, rr["objectMap"], If[KeyExistsQ[om, "ParentTriplesMap"], referencingObjectMapToTriples[om], objectMapToTriples[om]]];
	]];
	Lookup[pom, "GraphMaps", {}] // Scan[Function[gm,
		sowTriple[node, rr["graphMap"], graphMapToTriples[gm]];
	]];
	node
];

clear[predicateMapToTriples];
predicateMapToTriples[pm_?AssociationQ] := With[
	{node = termMapToTriples[pm]},
	sowTriple[node, rdf["type"], rr["PredicateMap"]];
	node
];

clear[objectMapToTriples];
objectMapToTriples[om_?AssociationQ] := With[
	{node = termMapToTriples[om]},
	sowTriple[node, rdf["type"], rr["ObjectMap"]];
	node
];


(* 7 Creating RDF Terms with Term Maps *)
clear[termMapToTriples];
termMapToTriples[tm_?AssociationQ] := With[
	{node = nodeLookup[tm]},
	sowTriple[node, rdf["type"], rr["TermMap"]];
	tm // Replace[{
		KeyValuePattern["Constant" -> constant_] :> sowTriple[node, rr["constant"], constant],
		KeyValuePattern["Column" -> column_] :> sowTriple[node, rr["column"], column],
		KeyValuePattern["Template" -> template_] :> sowTriple[node, rr["template"], template],
		_ :> fail[]
	}];
	tm // Replace[KeyValuePattern["TermType" -> termType_] :> sowTriple[node, rr["termType"], termType]];
	tm // Replace[KeyValuePattern["Language" -> language_] :> sowTriple[node, rr["language"], language]];
	tm // Replace[KeyValuePattern["Datatype" -> datatype_] :> sowTriple[node, rr["datatype"], datatype]];
	tm // Replace[KeyValuePattern["InverseExpression" -> ie_] :> sowTriple[node, rr["inverseExpression"], ie]];
	node
];


(* 8 Foreign Key Relationships among Logical Tables (rr:parentTriplesMap, rr:joinCondition, rr:child and rr:parent) *)
clear[referencingObjectMapToTriples];
referencingObjectMapToTriples[rom_?AssociationQ] := With[
	{node = nodeLookup[rom]},
	sowTriple[node, rdf["type"], rr["RefObjectMap"]];
	sowTriple[node, rr["parentTriplesMap"], rom["ParentTriplesMap"]];
	Lookup[rom, "JoinConditions", {}] // Scan[Function[jc,
		sowTriple[node, rr["joinCondition"], joinConditionToTriples[jc]];
	]];
	node
];

clear[joinConditionToTriples];
joinConditionToTriples[jc_?AssociationQ] := With[
	{node = nodeLookup[jc]},
	sowTriple[node, rdf["type"], rr["Join"]];
	sowTriple[node, rr["child"], jc["Child"]];
	sowTriple[node, rr["parent"], jc["Parent"]];
	node
];


(* 9 Assigning Triples to Named Graphs *)
clear[graphMapToTriples];
graphMapToTriples[gm_?AssociationQ] := With[
	{node = termMapToTriples[gm]},
	sowTriple[node, rdf["type"], rr["GraphMap"]];
	node
];

(* end export *)
(* -------------------------------------------------- *)


(* -------------------------------------------------- *)
(* import *)

clear[iImportR2RML];
Options[iImportR2RML] = Options[ImportR2RML];
iImportR2RML[file_, OptionsPattern[]] := {
	"Data" -> (Quiet[Import[file, "Turtle", "Base" -> OptionValue["Base"]]] // Replace[_?FailureQ :> fail[]] // triplesMapsQuery[] // triplesMapsToTree)
};


clear[triplesMapsToTree];
triplesMapsToTree[tml_List] := Values[GroupBy[
	tml,
	Key["triplesMap"],
	Function[tm,
		<|
			"Node" -> tm[[1, "triplesMap"]],
			(* logical table *)
			"LogicalTable" -> Replace[
				Values[GroupBy[
					Select[tm, KeyExistsQ["logicalTable"]],
					Key["logicalTable"],
					Function[lt,
						<|
							"Node" -> lt[[1, "logicalTable"]],
							lt // First // KeyTake[{"tableName", "sqlQuery", "sqlVersion"}] // KeyMap[Replace[{"tableName" -> "TableName", "sqlQuery" -> "SQLQuery", "sqlVersion" -> "SQLVersion"}]]
						|>
					]
				]],
				{
					{x_} :> x,
					{} :> fail[Failure["MissingLogicalTable", <|"MessageTemplate" -> "The triples map `1` has no logical table.", "MessageParameters" -> {tm[[1, "triplesMap"]]}|>]],
					{__} :> fail[Failure["MultipleLogicalTables", <|"MessageTemplate" -> "The triples map `1` has multiple logical tables.", "MessageParameters" -> {tm[[1, "triplesMap"]]}|>]]
				}
			],
			(* subject map *)
			"SubjectMap" -> Replace[
				Values[GroupBy[
					Select[tm, KeyExistsQ["subjectMap"]],
					Key["subjectMap"],
					Function[sm,
						<|
							"Node" -> sm[[1, "subjectMap"]],
							sm // First // KeyTake["subjectMap" <> # & /@ {"constant", "column", "template", "termType", "language", "datatype", "inverseExpression"}] // KeyMap[StringDelete[StartOfString ~~ "subjectMap"] /* Capitalize],
							(* classes *)
							"Classes" -> DeleteDuplicates[Cases[sm, KeyValuePattern["class" -> class_] :> class]],
							(* graph maps *)
							"GraphMaps" -> Values[GroupBy[
								Select[sm, KeyExistsQ["subjectMapgraphMap"]],
								Key["subjectMapgraphMap"],
								Function[gm,
									<|
										"Node" -> gm[[1, "subjectMapgraphMap"]],
										gm // First // KeyTake["subjectMapgraphMap" <> # & /@ {"constant", "column", "template", "termType", "language", "datatype", "inverseExpression"}] // KeyMap[StringDelete[StartOfString ~~ "subjectMapgraphMap"] /* Capitalize]
									|>
								]
							]]
						|> // DeleteCases[{}]
					]
				]],
				{
					{x_} :> x,
					{} :> fail[Failure["MissingSubjectMap", <|"MessageTemplate" -> "The triples map `1` has no subject map.", "MessageParameters" -> {tm[[1, "triplesMap"]]}|>]],
					{__} :> fail[Failure["MultipleSubjectMaps", <|"MessageTemplate" -> "The triples map `1` has multiple subject maps.", "MessageParameters" -> {tm[[1, "triplesMap"]]}|>]]
				}
			],
			(* predicate object maps *)
			"PredicateObjectMaps" -> Values[GroupBy[
				Select[tm, KeyExistsQ["predicateObjectMap"]],
				Key["predicateObjectMap"],
				Function[pom,
					<|
						"Node" -> pom[[1, "predicateObjectMap"]],
						(* predicate maps *)
						"PredicateMaps" -> Values[GroupBy[
							Select[pom, KeyExistsQ["predicateMap"]],
							Key["predicateMap"],
							Function[pm,
								<|
									"Node" -> pm[[1, "predicateMap"]],
									pm // First // KeyTake["predicateMap" <> # & /@ {"constant", "column", "template", "termType", "language", "datatype", "inverseExpression"}] // KeyMap[StringDelete[StartOfString ~~ "predicateMap"] /* Capitalize]
								|>
							]
						]],
						(* object maps *)
						"ObjectMaps" -> Values[GroupBy[
							Select[pom, KeyExistsQ["objectMap"]],
							Key["objectMap"],
							Function[om,
								<|
									"Node" -> om[[1, "objectMap"]],
									If[KeyFreeQ[First[om], "parentTriplesMap"],
										(* object map *)
										om // First // KeyTake["objectMap" <> # & /@ {"constant", "column", "template", "termType", "language", "datatype", "inverseExpression"}] // KeyMap[StringDelete[StartOfString ~~ "objectMap"] /* Capitalize],
										(* referencing object map *)
										<|
											om // First // KeyTake[{"parentTriplesMap"}] // KeyMap[Capitalize],
											"JoinConditions" -> Values[GroupBy[
												Select[om, KeyExistsQ["joinCondition"]],
												Key["joinCondition"],
												Function[jc,
													<|
														"Node" -> jc[[1, "joinCondition"]],
														jc // First // KeyTake[{"child", "parent"}] // KeyMap[Capitalize]
													|>
												]
											]]
										|> // DeleteCases[{}]
									]
								|>
							]
						]],
						(* graph maps *)
						"GraphMaps" -> Values[GroupBy[
							Select[pom, KeyExistsQ["predicateObjectMapgraphMap"]],
							Key["predicateObjectMapgraphMap"],
							Function[gm,
								<|
									"Node" -> gm[[1, "predicateObjectMapgraphMap"]],
									gm // First // KeyTake["predicateObjectMapgraphMap" <> # & /@ {"constant", "column", "template", "termType", "language", "datatype", "inverseExpression"}] // KeyMap[StringDelete[StartOfString ~~ "predicateObjectMapgraphMap"] /* Capitalize]
								|>
							]
						]]
					|> // DeleteCases[{}]
				]
			]]
		|> // DeleteCases[{}]
	]
]];


(* 5 Defining Logical Tables *)
clear[logicalTableQuery];
logicalTableQuery[var_] := SPARQLOptional[{
	RDFTriple[var, rr["logicalTable"], SPARQLVariable["logicalTable"]],
	Alternatives[
		RDFTriple[SPARQLVariable["logicalTable"], rr["tableName"], SPARQLVariable["tableName"]],
		{
			RDFTriple[SPARQLVariable["logicalTable"], rr["sqlQuery"], SPARQLVariable["sqlQuery"]],
			SPARQLOptional[RDFTriple[SPARQLVariable["logicalTable"], rr["sqlVersion"], SPARQLVariable["sqlVersion"]]]
		}
	]
}];


(* 6 Mapping Logical Tables to RDF with Triples Maps *)
clear[triplesMapsQuery];
triplesMapsQuery[] := SPARQLSelect[{
	SPARQLSelect[Alternatives[
		RDFTriple[SPARQLVariable["triplesMap"], rdf["type"], rr["TriplesMap"]],
		RDFTriple[SPARQLVariable["triplesMap"], rr["logicalTable"], RDFBlankNode[]],
		RDFTriple[SPARQLVariable["triplesMap"], rr["subject"], RDFBlankNode[]],
		RDFTriple[SPARQLVariable["triplesMap"], rr["subjectMap"], RDFBlankNode[]],
		RDFTriple[SPARQLVariable["triplesMap"], rr["predicateObjectMap"], RDFBlankNode[]]
	]] /* SPARQLProject["triplesMap"] /* SPARQLDistinct[],
	logicalTableQuery[SPARQLVariable["triplesMap"]],
	subjectMapQuery[SPARQLVariable["triplesMap"]],
	predicateObjectMapsQuery[SPARQLVariable["triplesMap"]]
}];

(* 6.1 Creating Resources with Subject Maps *)
clear[subjectMapQuery];
subjectMapQuery[var_] := SPARQLOptional[{
	Alternatives[
		{
			RDFTriple[var, rr["subjectMap"], SPARQLVariable["subjectMap"]],
			termMapQuery[SPARQLVariable["subjectMap"]]
		},
		{
			RDFTriple[var, rr["subject"], SPARQLVariable["subjectMap" <> "constant"]],
			"subjectMap" -> SPARQLEvaluation["BNODE"][]
		}
	],
	classesQuery[SPARQLVariable["subjectMap"]],
	graphMapsQuery[SPARQLVariable["subjectMap"]]
}];

(* 6.2 Typing Resources (rr:class) *)
clear[classesQuery];
classesQuery[var_] := SPARQLOptional[RDFTriple[var, rr["class"], SPARQLVariable["class"]]];

(* 6.3 Creating Properties and Values with Predicate-Object Maps *)
clear[predicateObjectMapsQuery];
predicateObjectMapsQuery[var_] := SPARQLOptional[{
	RDFTriple[var, rr["predicateObjectMap"], SPARQLVariable["predicateObjectMap"]],
	predicateMapQuery[SPARQLVariable["predicateObjectMap"]],
	Alternatives[
		objectMapQuery[SPARQLVariable["predicateObjectMap"]],
		referencingObjectMapsQuery[SPARQLVariable["predicateObjectMap"]]
	],
	graphMapsQuery[SPARQLVariable["predicateObjectMap"]]
}];

clear[predicateMapQuery];
predicateMapQuery[var_] := Alternatives[
	{
		RDFTriple[var, rr["predicateMap"], SPARQLVariable["predicateMap"]],
		termMapQuery[SPARQLVariable["predicateMap"]]
	},
	{
		RDFTriple[var, rr["predicate"], SPARQLVariable["predicateMap" <> "constant"]],
		"predicateMap" -> SPARQLEvaluation["BNODE"][]
	}
];

clear[objectMapQuery];
objectMapQuery[var_] := Alternatives[
	{
		RDFTriple[var, rr["objectMap"], SPARQLVariable["objectMap"]],
		termMapQuery[SPARQLVariable["objectMap"]]
	},
	{
		RDFTriple[var, rr["object"], SPARQLVariable["objectMap" <> "constant"]],
		"objectMap" -> SPARQLEvaluation["BNODE"][]
	}
];


(* 7 Creating RDF Terms with Term Maps *)
clear[termMapQuery];
termMapQuery[var : SPARQLVariable[map_String]] := {
	Alternatives[
		RDFTriple[var, rr["constant"], SPARQLVariable[map <> "constant"]],
		RDFTriple[var, rr["column"], SPARQLVariable[map <> "column"]],
		RDFTriple[var, rr["template"], SPARQLVariable[map <> "template"]]
	],
	termTypeQuery[var],
	languageTagQuery[var],
	datatypeQuery[var],
	inverseExpressionQuery[var]
};

(* 7.4 IRIs, Literal, Blank Nodes (rr:termType) *)
clear[termTypeQuery];
termTypeQuery[var : SPARQLVariable[map_String]] := SPARQLOptional[RDFTriple[var, rr["termType"], SPARQLVariable[map <> "termType"]]];

(* 7.5 Language Tags (rr:language) *)
clear[languageTagQuery];
languageTagQuery[var : SPARQLVariable[map_String]] := SPARQLOptional[RDFTriple[var, rr["language"], SPARQLVariable[map <> "language"]]];

(* 7.6 Typed Literals (rr:datatype) *)
clear[datatypeQuery];
datatypeQuery[var : SPARQLVariable[map_String]] := SPARQLOptional[RDFTriple[var, rr["datatype"], SPARQLVariable[map <> "datatype"]]];

(* 7.7 Inverse Expressions (rr:inverseExpression) *)
clear[inverseExpressionQuery];
inverseExpressionQuery[var : SPARQLVariable[map_String]] := SPARQLOptional[RDFTriple[var, rr["inverseExpression"], SPARQLVariable[map <> "inverseExpression"]]];


(* 8 Foreign Key Relationships among Logical Tables (rr:parentTriplesMap, rr:joinCondition, rr:child and rr:parent) *)
clear[referencingObjectMapsQuery];
referencingObjectMapsQuery[var_] := {
	RDFTriple[var, rr["objectMap"], SPARQLVariable["objectMap"]],
	RDFTriple[SPARQLVariable["objectMap"], rr["parentTriplesMap"], SPARQLVariable["parentTriplesMap"]],
	joinConditionsQuery[SPARQLVariable["objectMap"]]
};

clear[joinConditionsQuery];
joinConditionsQuery[var_] := SPARQLOptional[{
	RDFTriple[var, rr["joinCondition"], SPARQLVariable["joinCondition"]],
	RDFTriple[SPARQLVariable["joinCondition"], rr["child"], SPARQLVariable["child"]],
	RDFTriple[SPARQLVariable["joinCondition"], rr["parent"], SPARQLVariable["parent"]]
}];


(* 9 Assigning Triples to Named Graphs *)
clear[graphMapsQuery];
graphMapsQuery[var : SPARQLVariable[map_String]] := SPARQLOptional[Alternatives[
	{
		RDFTriple[var, rr["graphMap"], SPARQLVariable[map <> "graphMap"]],
		termMapQuery[SPARQLVariable[map <> "graphMap"]]
	},
	{
		RDFTriple[var, rr["graph"], SPARQLVariable[map <> "graphMap" <> "constant"]],
		map <> "graphMap" -> SPARQLEvaluation["BNODE"][]
	}
]];

(* end import *)
(* -------------------------------------------------- *)


End[];
EndPackage[];
