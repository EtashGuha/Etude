(* R2RML: RDB to RDF Mapping Language *)
(* https://www.w3.org/TR/r2rml/ *)


(* Semantics preserving SPARQL-to-SQL translation *)
(* Artem Chebotko, Shiyong Lu, Farshad Fotouhi *)
(* https://doi.org/10.1016/j.datak.2009.04.001 *)

(* Formalisation and Experiences of R2RML-based SPARQL to SQL query translation using Morph *)
(* Freddy Priyatna, Oscar Corcho, Juan Sequeda *)
(* http://wwwconference.org/proceedings/www2014/proceedings/p479.pdf *)

(* Efficient SPARQL-to-SQL with R2RML mappings *)
(* Mariano Rodriguez-Muro, Martin Rezk *)
(* https://doi.org/10.1016/j.websem.2015.03.001 *)

BeginPackage["GraphStore`R2RML`DatabaseRDFStore`", {"GraphStore`", "GraphStore`R2RML`"}];

Needs["DatabaseLink`"];
Needs["GraphStore`Formats`"];
Needs["GraphStore`IRI`"];
Needs["GraphStore`LanguageTag`"];
Needs["GraphStore`RDF`"];
Needs["GraphStore`SPARQL`Algebra`"];
Needs["GraphStore`SQL`"];

Options[DatabaseRDFStore] = {
	"Base" -> None
};

Begin["`Private`"];

DatabaseRDFStore[db_, file : _File | _IRI | _URL, rest___] := DatabaseRDFStore[db, Quiet[ImportInternal[file, "R2RML"]], rest];
DatabaseRDFStore[db_, g_RDFStore, rest___] := DatabaseRDFStore[db, Quiet[ImportStringInternal[ExportString[g, "Turtle"], "R2RML"]], rest];
DatabaseRDFStore[db_, ___]["Database"] := db;
DatabaseRDFStore[_, mapping_List, ___]["Mapping"] := mapping;

DatabaseRDFStoreEvaluateAlgebraExpression[args___] := With[{res = Catch[iDatabaseRDFStoreEvaluateAlgebraExpression[args], $failTag]}, res /; res =!= $failTag];


fail[___, f_Failure, ___] := Throw[f, $failTag];
fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


clear[rdf];
rdf[s_String] := IRI["http://www.w3.org/1999/02/22-rdf-syntax-ns#" <> s];

clear[rr];
rr[s_String] := IRI["http://www.w3.org/ns/r2rml#" <> s];

clear[xsd];
xsd[s_String] := "http://www.w3.org/2001/XMLSchema#" <> s;


clear[iDatabaseRDFStoreEvaluateAlgebraExpression];
iDatabaseRDFStoreEvaluateAlgebraExpression[DatabaseRDFStore[db_, mapping_List, opts : OptionsPattern[]], algebraExpr_] := Block[{
	$activeGraph = rr["defaultGraph"],
	$base = OptionValue[DatabaseRDFStore, "Base"],
	$blankNodeCounter = 1,
	$blankNodes = <||>
},
	Module[
		{res, conn},
		conn = openConnection[db];
		res = evalQuery[conn, mapping, algebraExpr];
		CloseSQLConnection[conn];
		res = res // Map[KeyMap[First]];
		res
	]
];
iDatabaseRDFStoreEvaluateAlgebraExpression[DatabaseRDFStore[___, f_Failure, ___], ___] := fail[f];

clear[openConnection];
openConnection[file_String | File[file_String]] := OpenSQLConnection[JDBC["SQLite", file]];

clear[evalQuery];
evalQuery[db_, mapping_, patt_] := Module[
	{res},
	res = SQLExecute[db, patternToSQL[patt, mapping, db], "ShowColumnHeadings" -> True];
	With[
		{headings = First[res]},
		res = AssociationThread[headings, #] & /@ Rest[res];
	];
	res
] // Replace[f_?FailureQ :> fail[f]] // Query[
	DeleteCases[#, _Missing, {2}] &,
	patt // inScopeVariables // AssociationMap[Function[var,
		generateRDFTerm @@ Lookup[#, {variableToSQL[var], variableToSQL[var] <> "Datatype"}] &
	]]
];

clear[inScopeVariables];
inScopeVariables[bgp_BGP] := DeleteDuplicates[Cases[bgp, _SPARQLVariable, {3}]];
inScopeVariables[join[x_, y_]] := DeleteDuplicates[Join @@ inScopeVariables /@ {x, y}];
inScopeVariables[graph[g_, patt_]] := DeleteDuplicates[Join[
	If[MatchQ[g, _SPARQLVariable], {g}, {}],
	inScopeVariables[patt]
]];
inScopeVariables[union[x_, y_]] := DeleteDuplicates[Join @@ inScopeVariables /@ {x, y}];
inScopeVariables[project[_, vars_List]] := SPARQLVariable /@ vars;
inScopeVariables[filter[_, patt_]] := inScopeVariables[patt];
inScopeVariables[distinct[x_]] := inScopeVariables[x];
inScopeVariables[extend[patt_, var_, _]] := DeleteDuplicates[Append[
	inScopeVariables[patt],
	SPARQLVariable[var]
]];
inScopeVariables[algValues[patt_, var_, _]] := DeleteDuplicates[Append[
	inScopeVariables[patt],
	SPARQLVariable[var]
]];
inScopeVariables[solutionListIdentity[]] := {};
inScopeVariables[leftJoin[patt1_, patt2_, _]] := DeleteDuplicates[Join[
	inScopeVariables[patt1],
	inScopeVariables[patt2]
]];


clear[patternToSQL];
Options[patternToSQL] = {
	"NullVariables" -> {}
};

(* triple pattern *)
patternToSQL[RDFTriple[s_, p_, o_], mapping_, db_, opts : OptionsPattern[]] := StringRiffle[{
	"SELECT",
	"*",
	"FROM",
	"(" <> StringRiffle[
		mapping // Map[Function[tm,
			{
				(* predicate - object map *)
				Prepend[
					Lookup[tm, "PredicateObjectMaps", {}],
					(* classes *)
					<|
						"PredicateMaps" -> {<|"Constant" -> rdf["type"]|>},
						"ObjectMaps" -> Function[class, <|"Constant" -> class|>] /@ Lookup[tm["SubjectMap"], "Classes", {}]
					|>
				] // Map[Function[pom,
					Outer[
						Function[{pm, om, gm},
							StringRiffle[{
								"SELECT",
								StringRiffle[{
									termMapToSQL[{tm, db}, "SubjectMap" -> tm["SubjectMap"]] <> " as s",
									datatypeToSQL[{tm, db}, "SubjectMap" -> tm["SubjectMap"]] <> " as sDatatype",
									termMapToSQL[{tm, db}, "PredicateMap" -> pm] <> " as p",
									datatypeToSQL[{tm, db}, "PredicateMap" -> pm] <> " as pDatatype",
									termMapToSQL[{tm, db}, "ObjectMap" -> om] <> " as o",
									datatypeToSQL[{tm, db}, "ObjectMap" -> om] <> " as oDatatype",
									termMapToSQL[{tm, db}, "GraphMap" -> gm] <> " as g",
									datatypeToSQL[{tm, db}, "GraphMap" -> gm] <> " as gDatatype"
								}, ", "],
								"FROM",
								logicalTableToSQL[tm["LogicalTable"]]
							}, "\n  "]
						],
						pom["PredicateMaps"],
						pom["ObjectMaps"] // Select[KeyFreeQ["ParentTriplesMap"]],
						Join @@ Lookup[{tm["SubjectMap"], pom}, "GraphMaps", {}] // Replace[{} :> {<|"Constant" -> rr["defaultGraph"]|>}]
					]
				]],
				(* predicate - referencing object map *)
				Lookup[tm, "PredicateObjectMaps", {}] // Map[Function[pom,
					Outer[
						Function[{pm, rom, gm},
							With[{ptm = FirstCase[mapping, KeyValuePattern["Node" -> rom["ParentTriplesMap"]]]},
								StringRiffle[{
									"SELECT",
									StringRiffle[{
										termMapToSQL[{tm, db}, "SubjectMap" -> tm["SubjectMap"], "child."] <> " as s",
										datatypeToSQL[{tm, db}, "SubjectMap" -> tm["SubjectMap"]] <> " as sDatatype",
										termMapToSQL[{tm, db}, "PredicateMap" -> pm, "child."] <> " as p",
										datatypeToSQL[{tm, db}, "PredicateMap" -> pm] <> " as pDatatype",
										termMapToSQL[{tm, db}, "SubjectMap" -> ptm["SubjectMap"], "parent."] <> " as o",
										datatypeToSQL[{tm, db}, "SubjectMap" -> ptm["SubjectMap"]] <> " as oDatatype",
										termMapToSQL[{tm, db}, "GraphMap" -> gm] <> " as g",
										datatypeToSQL[{tm, db}, "GraphMap" -> gm] <> " as gDatatype"
									}, ", "],
									"FROM",
									StringRiffle[{
										logicalTableToSQL[tm["LogicalTable"]] <> " as child",
										logicalTableToSQL[ptm["LogicalTable"]] <> " as parent"
									}, ", "],
									"WHERE",
									StringRiffle[
										Lookup[rom, "JoinConditions", {}] // Map[Function[
											"child."<> #Child <> " = " <> "parent." <> #Parent
										]],
										" AND "
									]
								} // SequenceReplace[{"WHERE", ""} :> Nothing], "\n  "]
							]
						],
						pom["PredicateMaps"],
						pom["ObjectMaps"] // Select[KeyExistsQ["ParentTriplesMap"]],
						Join @@ Lookup[{tm["SubjectMap"], pom}, "GraphMaps", {}] // Replace[{} :> {<|"Constant" -> rr["defaultGraph"]|>}]
					]
				]]
			}
		]] // Flatten,
		" UNION\n"
	] <> ")",
	"WHERE",
	StringRiffle[
		{
			If[MatchQ[s, _SPARQLVariable], "s IS NOT NULL", "s = " <> constantToSQL[s]],
			If[MatchQ[p, _SPARQLVariable], "p IS NOT NULL", "p = " <> constantToSQL[p]],
			If[MatchQ[o, _SPARQLVariable], "o IS NOT NULL", "o = " <> constantToSQL[o]]
		},
		" AND "
	]
} // SequenceReplace[{"WHERE", ""} :> Nothing], "\n "];

(* basic graph pattern *)
patternToSQL[BGP[{}], ___] := "SELECT 1";
patternToSQL[BGP[triples : {__RDFTriple}], mapping_, db_, opts : OptionsPattern[]] := StringRiffle[{
	"SELECT",
	StringRiffle[
		Flatten[{
			<|triples // MapIndexed[Function[{t, i},
				With[{table = "t" <> ToString[First[i]]},
					{
						t[[1]] -> table <> ".s",
						t[[2]] -> table <> ".p",
						t[[3]] -> table <> ".o",
						$activeGraph -> table <> ".g"
					}
				]
			]]|> // KeySelect[MatchQ[_SPARQLVariable]] // KeyValueMap[Function[{var, s},
				{
					s <> " as " <> variableToSQL[var],
					s <> "Datatype as " <> variableToSQL[var] <> "Datatype"
				}
			]],
			{
				(* "NULL" instead of NULL: This avoids casting non-NULL values to NULL if this is the first part of a UNION clause. *)
				"\"NULL\" as " <> variableToSQL[#],
				"\"\" as " <> variableToSQL[#] <> "Datatype"
			} & /@ OptionValue["NullVariables"]
		}],
		", "
	] // Replace["" -> "*"],
	"FROM",
	StringRiffle[
		triples // MapIndexed[Function[{t, i},
			StringRiffle[{
				"(" <> patternToSQL[
					t /. RDFBlankNode[] :> SPARQLVariable[StringDelete[CreateUUID["v"], "-"]],
					mapping,
					db,
					opts
				] <> ")",
				"as",
				"t" <> ToString[First[i]]
			}]
		]],
		"\nJOIN\n"
	],
	"WHERE",
	StringRiffle[
		{
			Position[triples, _SPARQLVariable, {2}] // GroupBy[Curry[Extract, 2][triples]] // Select[Length /* GreaterEqualThan[2]] // KeyValueMap[Function[{var, pos},
				BlockMap[
					Apply[Function[{p1, p2},
						With[{
							table = "t" <> ToString[First[#]] &,
							col = Switch[Last[#], 1, "s", 2, "p", 3, "o"] &
						},
							table[p1] <> "." <> col[p1] <> " = " <> table[p2] <> "." <> col[p2]
						]
					]],
					pos,
					2,
					1
				]
			]],
			If[MatchQ[$activeGraph, _SPARQLVariable],
				Table["t" <> ToString[i] <> ".g != " <> constantToSQL[rr["defaultGraph"]], {i, Length[triples]}],
				Table["t" <> ToString[i] <> ".g = " <> constantToSQL[$activeGraph], {i, Length[triples]}]
			]
		} // Flatten,
		" AND "
	]
} // SequenceReplace[{"WHERE", ""} :> Nothing], "\n"];

(* filter *)
patternToSQL[filter[expr_, patt_], mapping_, db_, opts : OptionsPattern[]] := StringRiffle[{
	"SELECT",
	"*",
	"FROM",
	"(" <> patternToSQL[patt, mapping, db, opts] <> ")",
	"WHERE",
	expressionToSQL[expr]
}, "\n"];

clear[expressionToSQL];
expressionToSQL[x : _Equal | _Unequal] := StringRiffle[Riffle[
	If[MatchQ[#, _SPARQLVariable], variableToSQL[#], constantToSQL[#]] & /@ List @@ x,
	operatorToSQL[Head[x]]
]];
expressionToSQL[x : _Greater | _GreaterEqual | _Less | _LessEqual] := StringRiffle[Riffle[
	If[MatchQ[#, _SPARQLVariable], "CAST(" <> variableToSQL[#] <> " as number)", ToString[#]] & /@ List @@ x,
	operatorToSQL[Head[x]]
]];
expressionToSQL[x : _And | _Or] := StringRiffle[Riffle[
	expressionToSQL /@ List @@ x,
	operatorToSQL[Head[x]]
]];
expressionToSQL[x_SPARQLVariable] := variableToSQL[x];
expressionToSQL[SPARQLEvaluation["ISIRI"][var_SPARQLVariable]] := StringRiffle[{
	variableToSQL[var] <> "Datatype",
	"=",
	"\"IRI\""
}];
expressionToSQL[SPARQLEvaluation[f_String][x___]] := With[
	{u = ToUpperCase[f]},
	expressionToSQL[SPARQLEvaluation[u][x]] /; u =!= f
];
expressionToSQL[expr_] := constantToSQL[expr];

clear[operatorToSQL];
operatorToSQL[Greater] := ">";
operatorToSQL[GreaterEqual] := ">=";
operatorToSQL[Less] := "<";
operatorToSQL[LessEqual] := "<=";
operatorToSQL[Equal] := "=";
operatorToSQL[Unequal] := "<>";
operatorToSQL[And] := "AND";
operatorToSQL[Or] := "OR";

clear[expressionToSQLDatatype];
expressionToSQLDatatype[_IRI] := "\"IRI\"";
expressionToSQLDatatype[_String] := "\"SQL:CHARACTER\"";
expressionToSQLDatatype[var_SPARQLVariable] := variableToSQL[var] <> "Datatype";

(* join *)
patternToSQL[join[x_, y_], mapping_, db_, opts : OptionsPattern[]] := StringRiffle[{
	"SELECT",
	"*",
	"FROM",
	"(" <> patternToSQL[x, mapping, db, opts] <> ") as j1",
	"JOIN",
	"(" <> patternToSQL[y, mapping, db, opts] <> ") as j2",
	"ON",
	StringRiffle["j1." <> variableToSQL[#] <> " = j2." <> variableToSQL[#] & /@ Intersection @@ inScopeVariables /@ {x, y}, " AND "]
} // SequenceReplace[{"ON", ""} :> Nothing], "\n"];

(* left join *)
patternToSQL[lj : leftJoin[x_, y_, True], mapping_, db_, opts : OptionsPattern[]] := StringRiffle[{
	"SELECT",
	StringRiffle[
		Flatten[
			inScopeVariables[lj] // Map[Function[var,
				With[{ji = Pick[{"j1", "j2"}, {MemberQ[inScopeVariables[x], var], MemberQ[inScopeVariables[y], var]}]},
					{
						"COALESCE(" <> StringRiffle[Append[# <> "." <> variableToSQL[var] & /@ ji, "\"NULL\""], ", "] <> ") as " <> variableToSQL[var],
						"COALESCE(" <> StringRiffle[Append[# <> "." <> variableToSQL[var] <> "Datatype" & /@ ji, "\"NULL\""], ", "] <> ") as " <> variableToSQL[var] <> "Datatype"
					}
				]
			]]
		],
		", "
	],
	"FROM",
	"(" <> patternToSQL[x, mapping, db, opts] <> ") as j1",
	"LEFT JOIN",
	"(" <> patternToSQL[y, mapping, db, opts] <> ") as j2",
	"ON",
	StringRiffle["j1." <> variableToSQL[#] <> " = j2." <> variableToSQL[#] & /@ Intersection @@ inScopeVariables /@ {x, y}, " AND "]
} // SequenceReplace[{"ON", ""} :> Nothing], "\n"];

(* union *)
patternToSQL[union[x_, y_], mapping_, db_, opts : OptionsPattern[]] := StringRiffle[{
	patternToSQL[x, mapping, db, "NullVariables" -> Union[OptionValue["NullVariables"], Complement @@ inScopeVariables /@ {y, x}], opts],
	"UNION ALL",
	patternToSQL[y, mapping, db, "NullVariables" -> Union[OptionValue["NullVariables"], Complement @@ inScopeVariables /@ {x, y}], opts]
}, "\n\n"];

(* graph *)
patternToSQL[graph[g_, patt_], mapping_, db_, opts : OptionsPattern[]] := Block[{$activeGraph = g}, patternToSQL[patt, mapping, db, opts]];

(* project *)
patternToSQL[project[patt_, vars_List], mapping_, db_, opts : OptionsPattern[]] := StringRiffle[{
	"SELECT",
	StringRiffle[Flatten[{variableToSQL[#], variableToSQL[#] <> "Datatype"} & /@ SPARQLVariable /@ vars], ", "],
	"FROM",
	"(" <> patternToSQL[patt, mapping, db, opts] <> ")"
}];

(* distinct *)
patternToSQL[distinct[patt_], mapping_, db_, opts : OptionsPattern[]] := StringRiffle[{
	"SELECT",
	"DISTINCT",
	"*",
	"FROM",
	"(" <> patternToSQL[patt, mapping, db, opts] <> ")"
}];

(* extend *)
patternToSQL[extend[patt_, var_String, expr_], mapping_, db_, opts : OptionsPattern[]] := StringRiffle[{
	"SELECT",
	"*",
	",",
	expressionToSQL[expr],
	"as",
	variableToSQL[SPARQLVariable[var]],
	",",
	expressionToSQLDatatype[expr],
	"as",
	variableToSQL[SPARQLVariable[var]] <> "Datatype",
	"FROM",
	"(" <> patternToSQL[patt, mapping, db, opts] <> ")"
}];

(* values *)
patternToSQL[algValues[solutionListIdentity[], var_String, values : {__}], mapping_, db_, opts : OptionsPattern[]] := StringRiffle[
	Prepend[
		StringRiffle[{
			"SELECT",
			constantToSQL[#],
			",",
			expressionToSQLDatatype[#]
		}] & /@ Rest[values],
		StringRiffle[{
			"SELECT",
			constantToSQL[First[values]],
			"as",
			variableToSQL[SPARQLVariable[var]],
			",",
			expressionToSQLDatatype[First[values]],
			"as",
			variableToSQL[SPARQLVariable[var]] <> "Datatype"
		}]
	],
	" UNION ALL\n"
];


clear[logicalTableToSQL];
logicalTableToSQL[KeyValuePattern["TableName" -> table_String]] := "[" <> StringTrim[table, "\""] <> "]";
logicalTableToSQL[lt : KeyValuePattern["SQLQuery" -> sqlQuery_String]] := (
	If[KeyExistsQ[lt, "SQLVersion"] && ! MemberQ[{rr["SQL2008"]}, lt["SQLVersion"]],
		fail[Failure["UnknownSQLVersionIdentifier", <|"MessageTemplate" -> "SQL version identifier `1` is unknown.", "MessageParameters" -> {lt["SQLVersion"]}|>]]
	];
	"(" <> StringDelete[sqlQuery, ";" ~~ WhitespaceCharacter ... ~~ EndOfString] <> ")"
);

clear[termMapToSQL];
termMapToSQL[_, map_String -> KeyValuePattern["Constant" -> constant_], ___] := constantToSQL[constant];
termMapToSQL[{tm_, db_}, map_String -> termMap : KeyValuePattern["Column" -> column_String], prefix_String : ""] := With[
	{c = prefix <> "[" <> StringTrim[column, "\""] <> "]"},
	If[binaryColumnQ[column, tm, db],
		"hex(" <> c <> ")",
		"'' || " <> c
	]
];
termMapToSQL[{tm_, db_}, map_String -> termMap : KeyValuePattern["Template" -> template_String], prefix_String : ""] := StringDelete[
	"'" <> StringReplace[
		template,
		start : StartOfString | Except["\\"] ~~ "{" ~~ Shortest[column__] ~~ "}" :> start <> "' || " <> With[
			{c = prefix <> "[" <> StringTrim[URLDecode[column], "\""] <> "]"},
			If[
				binaryColumnQ[column, tm, db],
				"hex(" <> c <> ")",
				Fold[
					"replace(" <> # <> ", '" <> #2 <> "', '" <> URLEncode[#2] <> "')" &,
					c,
					{
						";", "/", "?", ":", "@", "=", "&",
						" ", ",", "(", ")"
					}
				]
			]
		] <> " || '"
	] <> "'",
	"\\"
];

clear[binaryColumnQ];
binaryColumnQ[column_, tm_, db_] := With[
	{type = getColumnDatatype[column, tm, db]},
	StringQ[type] && StringStartsQ[type, Alternatives[
		"BINARY", "BINARY VARYING", "BINARY LARGE OBJECT",
		"VARBINARY"
	]]
];

clear[termType];
termType[map_String -> termMap_Association] := If[KeyExistsQ[termMap, "TermType"],
	(* validation *)
	With[
		{tt = termMap["TermType"]},
		Switch[map,
			"SubjectMap", If[! MemberQ[{rr["IRI"], rr["BlankNode"]}, tt], fail[Failure["SubjectMapInvalidTermType", <|"MessageTemplate" -> "The subject map's term type `1` is invalid.", "MessageParameters" -> {tt}|>]]],
			"PredicateMap", If[! MemberQ[{rr["IRI"]}, tt], fail[Failure["PredicateMapInvalidTermType", <|"MessageTemplate" -> "The predicate map's term type `1` is invalid.", "MessageParameters" -> {tt}|>]]],
			"ObjectMap", If[! MemberQ[{rr["IRI"], rr["BlankNode"], rr["Literal"]}, tt], fail[Failure["ObjectMapInvalidTermType", <|"MessageTemplate" -> "The object map's term type `1` is invalid.", "MessageParameters" -> {tt}|>]]],
			"GraphMap", If[! MemberQ[{rr["IRI"]}, tt], fail[Failure["GraphMapInvalidTermType", <|"MessageTemplate" -> "The graph map's term type `1` is invalid.", "MessageParameters" -> {tt}|>]]],
			_, Null
		];
		tt
	],
	(* default *)
	If[
		And[
			map === "ObjectMap",
			AnyTrue[
				{"Column", "Language", "Datatype"},
				KeyExistsQ[termMap, #] &
			]
		],
		rr["Literal"],
		rr["IRI"]
	]
];

clear[constantToSQL];
constantToSQL[IRI[i_String]] := "\"" <> i <> "\"";
constantToSQL[RDFLiteral[s_String, _]] := "\"" <> s <> "\"";
constantToSQL[s_String] := "\"" <> s <> "\"";
constantToSQL[x_] := constantToSQL[ToRDFLiteral[x] // Replace[_ToRDFLiteral :> fail[]]];

clear[variableToSQL];
variableToSQL[SPARQLVariable[s_String]] := s;

clear[datatypeToSQL];
datatypeToSQL[_, _ -> KeyValuePattern["Constant" -> Except[_IRI | _RDFBlankNode, const_]]] := With[{l = ToRDFLiteral[const]}, "\"Literal:\" || " <> constantToSQL[IRI[Last[l]]] /; MatchQ[l, _RDFLiteral]];
datatypeToSQL[{tm_, db_}, map_String -> termMap_Association] := Switch[termType[map -> termMap],
	rr["IRI"],
	"\"IRI\"",
	rr["BlankNode"],
	"\"BlankNode\"",
	rr["Literal"],
	termMap // Replace[{
		KeyValuePattern["Datatype" -> dt_] :> "\"Literal:\" || " <> constantToSQL[dt],
		KeyValuePattern["Language" -> lang_String] :> "\"LangString:" <> lang <> "\"",
		KeyValuePattern["Column" -> column_] :> "\"SQL:" <> Replace[
			getColumnDatatype[column, tm, db],
			{
				f_Failure :> fail[f],
				_Missing :> fail[Failure["UnknownColumn", <|"MessageTemplate" -> "Unknown column `2` in table `1`.", "MessageParameters" -> {tm["LogicalTable", "TableName"], column}|>]]
			}
		] <> "\"",
		_ :> "\"SQL:\""
	}],
	_,
	fail[]
];

clear[getColumnDatatype];
getColumnDatatype[column_String, KeyValuePattern["LogicalTable" -> KeyValuePattern["TableName" -> table_String]], db_] := SQLColumnType[StringTrim[column, "\""], StringTrim[table, "\""], db];
getColumnDatatype[column_String, KeyValuePattern["LogicalTable" -> KeyValuePattern["SQLQuery" -> query_]], db_] := SQLColumnTypeFromQuery[StringTrim[column, "\""], query, db];


(* 10 Datatype Conversions *)

(* 10.2 Natural Mapping of SQL Values *)
clear[naturalRDFLiteral];
naturalRDFLiteral[value_String, "CHARACTER" | "CHARACTER VARYING" | "CHARACTER LARGE OBJECT" | "NATIONAL CHARACTER" | "NATIONAL CHARACTER VARYING" | "NATIONAL CHARACTER LARGE OBJECT"] := value;
naturalRDFLiteral[value_String, "BINARY" | "BINARY VARYING" | "BINARY LARGE OBJECT" | "VARBINARY"] := FromRDFLiteral[RDFLiteral[value, xsd["hexBinary"]]];
naturalRDFLiteral[value_String, "NUMERIC" | "DECIMAL"] := FromRDFLiteral[RDFLiteral[value, xsd["decimal"]]];
naturalRDFLiteral[value_String, "SMALLINT" | "INTEGER" | "BIGINT" | "INT"] := FromRDFLiteral[RDFLiteral[value, xsd["integer"]]];
naturalRDFLiteral[value_String, "FLOAT" | "REAL" | "DOUBLE PRECISION"] := FromRDFLiteral[RDFLiteral[value, xsd["double"]]];
naturalRDFLiteral[value_String, "BOOLEAN"] := FromRDFLiteral[RDFLiteral[value, xsd["boolean"]]];
naturalRDFLiteral[value_String, "DATE"] := FromRDFLiteral[RDFLiteral[value, xsd["date"]]];
naturalRDFLiteral[value_String, "TIME"] := FromRDFLiteral[RDFLiteral[value, xsd["time"]]];
naturalRDFLiteral[value_String, "TIMESTAMP"] := FromRDFLiteral[RDFLiteral[StringReplace[value, " " -> "T"], xsd["dateTime"]]];
naturalRDFLiteral[value_String, dt_String?(StringMatchQ["CHAR(" ~~ DigitCharacter .. ~~ ")", IgnoreCase -> True])] := StringPadRight[value, ToExpression[StringTake[dt, {6, -2}]]];
naturalRDFLiteral[value_String, dt_String?(StringContainsQ["("])] := naturalRDFLiteral[value, StringDelete[dt, "(" ~~ ___ ~~ EndOfString]];
naturalRDFLiteral[value_String, _] := value;


(* 11 The Output Dataset *)

(* 11.2 The Generated RDF Term of a Term Map *)
clear[generateRDFTerm];
generateRDFTerm[s_String, "IRI"] := Module[
	{res},
	res = IRI[s];
	If[AbsoluteIRIQ[res],
		Return[res];
	];
	If[! StringQ[$base],
		fail[]
	];
	res = IRI[$base <> First[res]];
	If[! AbsoluteIRIQ[res],
		fail[Failure["DataError", <|"MessageTemplate" -> "An invalid RDF term `1` was generated.", "MessageParameters" -> {res}|>]];
	];
	res
];
generateRDFTerm[s_String, "BlankNode"] := RDFBlankNode[Lookup[$blankNodes, s, $blankNodes[s] = "b" <> ToString[$blankNodeCounter++]]];
generateRDFTerm[s_String, dt_String?(StringStartsQ["Literal:"])] := FromRDFLiteral[RDFLiteral[s, StringDrop[dt, 8]]];
generateRDFTerm[s_String, dt_String?(StringStartsQ["SQL:"])] := naturalRDFLiteral[s, StringDrop[dt, 4]];
generateRDFTerm[s_String, l_String?(StringStartsQ["LangString:"])] := Module[
	{lang},
	lang = StringDrop[l, 11];
	If[! LanguageTagQ[lang],
		fail[Failure["InvalidLanguageTag", <|"MessageTemplate" -> "The language tag `1` is invalid.", "MessageParameters" -> {lang}|>]]
	];
	RDFString[s, lang]
];
generateRDFTerm["NULL", ___] := Missing[];


End[];
EndPackage[];
