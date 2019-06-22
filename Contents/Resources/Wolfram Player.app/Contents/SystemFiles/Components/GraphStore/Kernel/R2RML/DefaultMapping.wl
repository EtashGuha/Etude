(* A Direct Mapping of Relational Data to RDF *)
(* https://www.w3.org/TR/rdb-direct-mapping/ *)

BeginPackage["GraphStore`R2RML`DefaultMapping`", {"GraphStore`", "GraphStore`R2RML`"}];

Needs["DatabaseLink`"];

Options[R2RMLDefaultMapping] = {
	"PreserveDuplicateRows" -> True
};

Begin["`Private`"];

R2RMLDefaultMapping[args___] := With[{res = Catch[iR2RMLDefaultMapping[args], $failTag]}, res /; res =!= $failTag];


fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


clear[rdf];
rdf[s_String] := IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#" <> s];

clear[rr];
rr[s_String] := IRI["http://www.w3.org/ns/r2rml#" <> s];


clear[iR2RMLDefaultMapping];
Options[iR2RMLDefaultMapping] = Options[R2RMLDefaultMapping];
iR2RMLDefaultMapping[conn_SQLConnection, base_String, OptionsPattern[]] := Block[{
	$blankNodeCounter = 1
},
	Module[
		{tm, lt, sm, pk, pom, om, ptm, jc},
		RDFStore[
			Flatten[SQLTableNames[conn] // Map[Function[table,
				{tm, lt, sm} = Table[createBlankNode[], 3];
				pk = StringReplace[getPrimaryKey[conn, table], "\"" ~~ s__ ~~ "\"" :> s];
				{
					(* triples map *)
					RDFTriple[tm, rdf["type"], rr["TriplesMap"]],
					(* logical table *)
					RDFTriple[tm, rr["logicalTable"], lt],
					RDFTriple[lt, rr["tableName"], table],
					(* subject map *)
					RDFTriple[tm, rr["subjectMap"], sm],
					RDFTriple[sm, rr["class"], IRI[base <> tableIRI[table]]],
					Switch[pk,
						{__String},
						RDFTriple[sm, rr["template"], base <> URLEncode[table] <> "/" <> StringRiffle[# <> "=" <> "{" <> # <> "}" & /@ URLEncode /@ Flatten[{pk}], ";"]],
						{},
						{
							RDFTriple[sm, rr["template"], If[OptionValue["PreserveDuplicateRows"],
								"b-" <> table <> "-{ROWID}",
								"b-" <> table <> StringRiffle["{" <> # <> "}" & /@ SQLColumnNames[conn, table][[All, 2]], "-"],
								fail[]
							]],
							RDFTriple[sm, rr["termType"], rr["BlankNode"]]
						},
						_, fail[]
					],
					(* predicate object maps *)
					(* columns *)
					SQLColumnNames[conn, table][[All, 2]] // Map[Function[column,
						{pom, om} = Table[createBlankNode[], 2];
						{
							RDFTriple[tm, rr["predicateObjectMap"], pom],
							(* predicate map *)
							RDFTriple[pom, rr["predicate"], IRI[base <> literalPropertyIRI[table, column]]],
							(* object map *)
							RDFTriple[pom, rr["objectMap"], om],
							RDFTriple[om, rr["column"], column]
						}
					]],
					(* foreign keys *)
					ptm[table] = tm;
					getForeignKeys[conn, table] // Map[Function[fk,
						{pom, om} = Table[createBlankNode[], 2];
						{
							RDFTriple[tm, rr["predicateObjectMap"], pom],
							(* predicate map *)
							RDFTriple[pom, rr["predicate"], IRI[base <> referencePropertyIRI[table, fk["FKCOLUMN_NAME"]]]],
							(* referencing object map *)
							RDFTriple[pom, rr["objectMap"], om],
							RDFTriple[om, rr["parentTriplesMap"], ptm[fk["PKTABLE_NAME"]]],
							fk // Lookup[{"FKCOLUMN_NAME", "PKCOLUMN_NAME"}] // Map[Replace[Except[_List, x_] :> {x}]] // MapThread[Function[{child, parent},
								jc = createBlankNode[];
								{
									RDFTriple[om, rr["joinCondition"], jc],
									RDFTriple[jc, rr["child"], child],
									RDFTriple[jc, rr["parent"], parent]
								}
							]]
						}
					]]
				}
			]]]
		]
	]
];
iR2RMLDefaultMapping[db_, (IRI | URL)[base_String], rest___] := iR2RMLDefaultMapping[db, base, rest];
iR2RMLDefaultMapping[file_String | File[file_String], rest___] := Module[
	{res, conn},
	conn = OpenSQLConnection[JDBC["SQLite", file]];
	res = iR2RMLDefaultMapping[conn, rest];
	CloseSQLConnection[conn];
	res
];

clear[createBlankNode];
createBlankNode[] := RDFBlankNode["b" <> ToString[$blankNodeCounter++]];

clear[getPrimaryKey];
(* https://bugs.wolfram.com/show?number=362254 *)
(*getPrimaryKey[db_, table_] := SQLTablePrimaryKeys[db, table][[All, 4]];*)
getPrimaryKey[db_, table_] := SQLExecute[
	db,
	"pragma table_info([" <> table <> "]);"
] //
DeleteCases[{___, 0}] //
SortBy[Last] //
Query[All, 2];

clear[getForeignKeys];
getForeignKeys[db_, table_] := Module[
	{fk},
	fk = SQLTableImportedKeys[db, table, "ShowColumnHeadings" -> True];
	fk = AssociationThread[First[fk], #] & /@ Rest[fk];
	If[fk === {},
		Return[fk, Module];
	];
	fk = SplitBy[
		fk,
		Module[
			{last = 0},
			If[#["KEY_SEQ"] == ++last, True, last = 0; False] &
		]
	];
	fk = Merge[Identity] /@ fk;
	fk = MapAt[First, fk, {All, "PKTABLE_NAME"}];
	fk
];


(* table IRI *)
clear[tableIRI];
tableIRI[table_String] := URLEncode[table];

(* literal property IRI *)
clear[literalPropertyIRI];
literalPropertyIRI[table_String, column_String] := URLEncode[table] <> "#" <> URLEncode[column];

(* reference property IRI *)
clear[referencePropertyIRI];
referencePropertyIRI[table_String, columns : {__String}] := URLEncode[table] <> "#ref-" <> StringRiffle[URLEncode /@ columns, ";"];


End[];
EndPackage[];
