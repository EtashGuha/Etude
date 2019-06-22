BeginPackage["GraphStore`SQL`"];

Needs["DatabaseLink`"];
Needs["GraphStore`Formats`"];

SQLCase1;
SQLColumnType;
SQLColumnTypeFromQuery;
SQLEvaluation1;
SQLIdentifier1;
SQLSelect1;

Begin["`Private`"];

SQLColumnType[args___] := With[{res = Catch[iSQLColumnType[args], $failTag]}, res /; res =!= $failTag];
SQLColumnTypeFromQuery[args___] := With[{res = Catch[iSQLColumnTypeFromQuery[args], $failTag]}, res /; res =!= $failTag];


fail[___, f_Failure, ___] := Throw[f, $failTag];
fail[___] := Throw[$failTag, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


clear[iSQLColumnType];
iSQLColumnType["ROWID", ___] := "INTEGER";
iSQLColumnType[column_String, table_String, conn_SQLConnection] := First[
	SQLColumnInformation[conn, {table, column}],
	fail[Failure["UnknownColumn", <|"MessageTemplate" -> "Unknown column `2` in table `1`.", "MessageParameters" -> {table, column}|>]]
][[6]];


clear[iSQLColumnTypeFromQuery];
iSQLColumnTypeFromQuery[column_String, query_String, db_] := Quiet[ImportStringInternal[query, "SQL"]] // Replace[{
	f_?FailureQ :> fail[Failure["SQLQueryParseFailue", <|"MessageTemplate" -> "Failed to parse SQL query `1`.", "MessageParameters" -> {query}|>]],
	symbolicQuery_ :> iSQLColumnTypeFromQuery[column, symbolicQuery, db]
}];
iSQLColumnTypeFromQuery[_, SQLSelect1[id_List, ___], _] := With[
	{c = CountsBy[id, Replace[r_Rule :> First[r]]] // Select[GreaterThan[1]]},
	fail[Failure["SQLQueryDuplicateColumnName", <|"MessageTemplate" -> "Query contains duplicate column name `1`.", "MessageParameters" -> {First[First[Keys[c]]]}|>]] /; Length[c] > 0
];
iSQLColumnTypeFromQuery[column_String, SQLSelect1[
	{___, SQLIdentifier1[column_], ___},
	{SQLIdentifier1[table_String]},
	___
], db_] := iSQLColumnType[column, table, db];
iSQLColumnTypeFromQuery[column_String, SQLSelect1[
	{___, SQLIdentifier1[column_] -> SQLIdentifier[baseColumn_], ___},
	{SQLIdentifier1[table_String]},
	___
], db_] := iSQLColumnType[baseColumn, table, db];
iSQLColumnTypeFromQuery[column_String, SQLSelect1[
	{___, {SQLIdentifier1[table_String], SQLIdentifier1[column_]}, ___},
	{___, SQLIdentifier1[table_String], ___},
	___
], db_] := iSQLColumnType[column, table, db];
iSQLColumnTypeFromQuery[column_String, SQLSelect1[
	{___, SQLIdentifier1[column_] -> {SQLIdentifier1[table_String], SQLIdentifier1[baseColumn_]}, ___},
	{___, SQLIdentifier1[table_String], ___},
	___
], db_] := iSQLColumnType[baseColumn, table, db];
iSQLColumnTypeFromQuery[column_String, SQLSelect1[
	{___, SQLIdentifier1[column_] -> Inactive[StringJoin][___], ___},
	___
], db_] := "CHARACTER";
iSQLColumnTypeFromQuery[column_String, SQLSelect1[
	{___, SQLIdentifier1[column_] -> SQLEvaluation1[_String?(StringMatchQ["COUNT", IgnoreCase -> True])][___], ___},
	___
], db_] := "INTEGER";
iSQLColumnTypeFromQuery[column_String, Except[_String, query_], db_] := fail[Failure["SQLQueryUnknownColumnType", <|"MessageTemplate" -> "Unable to determine type of column `1` in query `2` agains database `3`", "MessageParameters" -> {column, query, db}|>]];


End[];
EndPackage[];
