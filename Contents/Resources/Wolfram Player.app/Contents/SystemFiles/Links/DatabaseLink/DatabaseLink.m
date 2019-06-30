(* Author:          Christopher Williamson *)
(* Copyright:       Copyright 2004-2013, Wolfram Research, Inc. *)

BeginPackage["DatabaseLink`", {"JLink`"}]

`Information`$Version = "DatabaseLink Version 3.1.0 (February 2016)";

`Information`$VersionNumber = 3.1;

`Information`$ReleaseNumber = 0;

(* Usage Statements *)

AddDatabaseResources::usage = 
"AddDatabaseResources[dir] adds a directory to the DatabaseResources path."

CloseSQLConnection::usage = 
"CloseSQLConnection[conn] disconnects the current connection associated with a data source."

DatabaseExplorer::usage=
"DatabaseExplorer[] launches a graphical user interface to DatabaseLink."

DataSourceNames::usage = 
"DataSourceNames[] returns a list of the names of data sources made available through the DatabaseResourcesPath[]."

DataSources::usage = 
"DataSources[] returns a list of information about named data sources made available through the DatabaseResourcesPath[].
DataSources[\"name\"] returns a list of information about the data sources called name."

DatabaseResourcesPath::usage = 
"DatabaseResourcesPath[] gives the list of directories that are searched to find database resources."

(*
InstallStoredProcedures::usage = ""
*)

(*
InstallJDBCDriver::usage = "InstallJDBCDriver[\"name\"] downloads and installs the specified driver, \
creating a JDBC configuration if necessary.";
*)

JDBC::usage = 
"JDBC[args] is an object that holds parameters for making a JDBC connection to a database."

JDBCDriver::usage = 
"JDBCDriver[args] specifies the configuration for connecting to a database produced by a specific vendor."

JDBCDriverNames::usage = 
"JDBCDriverNames[] returns a list of the names of databases for which JDBC drivers are available through the DatabaseResourcesPath[]."

JDBCDrivers::usage = 
"JDBCDrivers[] returns a list of information about JDBC drivers available through the DatabaseResourcesPath[].
JDBCDrivers[\"name\"] returns a list of information about the JDBC driver called name."

WriteDataSource::usage = 
"WriteDataSource[name, opts] stores on the file system an SQLConnection configuration."

OpenSQLConnection::usage = 
"OpenSQLConnection[src, opts] connects to the specified data source.
OpenSQLConnection[] opens a GUI for selecting and editing data sources."

SQLConnectionOpenQ::usage =
"SQLConnectionOpenQ[conn] returns a boolean indicating whether conn is open or closed on the client."

SQLConnectionUsableQ::usage =
"SQLConnectionUsableQ[conn] returns a boolean indicating whether queries may be executed with conn."

SQLArgument::usage = 
"SQLArgument[arg1, arg2, ...] holds a sequence of arguments to an SQL query."

SQLBeginTransaction::usage = 
"SQLBeginTransaction[conn] initiates an SQL transaction. A group of SQL commands grouped into a transaction will only take effect permanently when the transaction is committed."

SQLBinary::usage = 
"SQLBinary[data] represents raw binary data that can be stored in a database."

SQLCatalogNames::usage = 
"SQLCatalogNames[conn] returns the names of the catalogs in an SQL connection."

SQLColumn::usage = 
"SQLColumn[args] represents a column in an SQL table."

SQLColumnInformation::usage = 
"SQLColumnInformation[conn] returns a list of information about the columns in an SQL connection."

SQLColumnNames::usage = 
"SQLColumnNames[conn] returns a list of elements, {table, name}, for each column in an SQL connection."

SQLColumnPrivileges::usage = 
"SQLColumnPrivileges[conn] returns a table of access rights about the columns in an SQL connection."

SQLColumns::usage = 
"SQLColumns[conn] returns the SQLColumn objects for each column in an SQL connection."

SQLCommitTransaction::usage = 
"SQLCommitTransaction[conn] commits an SQL transaction. A group of SQL commands grouped into a transaction will only take effect permanently when the transaction is committed."

SQLConnection::usage = 
"SQLConnection[args] is an object that represents a connection to a data source."

SQLConnectionInformation::usage = 
"SQLConnectionInformation[conn] returns a list of information about the SQL connection.
SQLConnectionInformation[conn, mdi] returns the value of the meta data item (mdi) requested.
SQLConnectionInformation[conn, {mdi1, mdi2, ...}] returns a list of the values of the meta data items (mdi) requested."

SQLConnectionWarnings::usage = 
"SQLConnectionWarnings[conn] returns a list of warnings for the SQL connection."

SQLConnectionPool::usage = 
"SQLConnectionPool[args] is an object that represents a connection pool to a data source."

SQLConnectionPoolClose::usage = 
"SQLConnectionPoolClose[args] closes a connection pool."

SQLCreateTable::usage = 
"SQLCreateTable[conn, table, columns, opts] creates a new table in an SQL connection."

SQLDataTypeInformation::usage = 
"SQLDataTypeInformation[conn] returns information about the data types that can be stored in an SQL connection."

SQLDataTypeNames::usage = 
"SQLDataTypeNames[conn] returns the names of datatypes that can be stored in an SQL connection."

SQLDateTime::usage = 
"SQLDateTime[date|time] represents date and time information that can be stored in a database."

SQLDelete::usage = 
"SQLDelete[conn, table] deletes the data in a table in an SQL connection.
SQLDelete[conn, table, cond] deletes data that matches cond. This function should be used cautiously."

SQLDropTable::usage = 
"SQLDropTable[conn, table] drops a table in an SQL connection. This function should be used with caution."

SQLExecute::usage = 
"SQLExecute[conn, comm, args...] executes a command in an SQL connection."

SQLExpr::usage = 
"SQLExpr[expr] allows a Mathematica expression to be stored in a database."

SQLInsert::usage = 
"SQLInsert[conn, table, cols, data] inserts data into a table in an SQL connection."

SQLMemberQ::usage =
"SQLMemberQ[data, column] is used to test the value of data in a column when using a condition as part of an SQL query."

SQLQueries::usage = 
"SQLQueries[] returns a list of datasources made available through the DatabaseResourcesPath[]."

SQLQueryNames::usage = 
"SQLQueryNames[] returns a list of the names of datasources made available through the DatabaseResourcesPath[].  Each name can be used to execute a query."

SQLResultSet::usage = 
"SQLResultSet[args] is an object that represents the results from an SQL query."

SQLResultSetColumnNames::usage = 
"SQLResultSetColumnNames[rs] returns a list of elements, {table, name}, for each column in a result set."

SQLResultSetClose::usage = 
"SQLResultSetClose[rs] closes a result set."

SQLResultSetCurrent::usage = 
"SQLResultSetCurrent[rs] reads the current row from a result set."

SQLResultSetGoto::usage = 
"SQLResultSetGoto[rs, pos] sets the current position of a result set to pos."

SQLResultSetOpen::usage = 
"SQLResultSetOpen[query, opts] makes a result set from an SQL query."

SQLResultSetPosition::usage = 
"SQLResultSetPosition[rs] returns an integer that specifies the current position in a result set."

SQLResultSetRead::usage = 
"SQLResultSetRead[rs] shifts the current position and then reads a row from a result set.
SQLResultSetRead[rs, num] reads num rows from a result set."

SQLResultSetShift::usage = 
"SQLResultSetShift[rs, num] shifts the current position of a result set by num."

SQLResultSetTake::usage = 
"SQLResultSetTake[rs, {m, n}] reads rows m through n from a result set."

SQLRollbackTransaction::usage = 
"SQLRollbackTransaction[conn] is used to terminate an SQL transaction.
SQLRollbackTransaction[conn, savepoint] is used to return to an SQLSavepoint.
A group of SQL commands grouped into a transaction will only take effect permanently when the transaction is committed."

SQLReleaseSavepoint::usage = 
"SQLReleaseSavepoint[conn, savepoint] removes the given savepoint from the current transaction."

SQLSavepoint::usage = 
"SQLSavepoint[args] is an object that represents a savepoint in an SQL transaction."

SQLSchemaInformation::usage=
"SQLSchemaInformation[conn] returns information about the schemas available through an SQL connection."

SQLSchemaNames::usage = 
"SQLSchemaNames[conn] returns the names of the schema in an SQL connection."

SQLSelect::usage = 
"SQLSelect[conn, table]  extracts data from a table in an SQL connection.
SQLSelect[conn, table, cols]  extracts data from particular columns.
SQLSelect[conn, table, cols, cond]  only extracts data that matches cond."

SQLServer::usage = 
"SQLServer[args] is an object that represents a server process started in Mathematica."

SQLServerInformation::usage = 
"SQLServerInformation[server] returns a list of information about the SQL server."

SQLServerLaunch::usage = 
"SQLServerLaunch[{name->location .. }] launches a database server that hosts access to the databases specified in the parameters."

SQLServerShutdown::usage = 
"SQLServerShutdown[server] shuts down an active SQLServer started in Mathematica."

SQLSetSavepoint::usage = 
"SQLSetSavepoint[conn, name] creates a savepoint to be used as part of an SQL transaction."

SQLStringMatchQ::usage =
"SQLStringMatchQ[col, patt]  uses patt to test the value of data in a column when using a condition as part of an SQL query. The actual format for the pattern varies from one database to another."

SQLTable::usage = 
"SQLTable[args] represents a table in an SQL connection."

SQLTableExportedKeys::usage = 
"SQLTableExportedKeys[conn] returns a table of foreign key descriptions that reference the table's primary key."

SQLTableImportedKeys::usage = 
"SQLTableImportedKeys[conn] returns a table of primary key descriptions that are referenced by the table's foreign key."

SQLTableIndexInformation::usage = 
"SQLTableIndexInformation[conn] returns a table of indices and statistics for a table."

SQLTableInformation::usage = 
"SQLTableInformation[conn] returns a list of information about the tables in an SQL connection."

SQLTableNames::usage = 
"SQLTableNames[conn] returns the names of each table in an SQL connection."

SQLTablePrimaryKeys::usage = 
"SQLTablePrimaryKeys[conn] returns a table of primary key descriptions."

SQLTablePrivileges::usage = 
"SQLTablePrivileges[conn] returns a table of access rights about the tables in an SQL connection."

SQLTables::usage = 
"SQLTables[conn] returns the SQLTable objects for each table in an SQL connection."

SQLTableTypeNames::usage = 
"SQLTableTypeNames[datasourceobject] returns the names of the table types in the current data source."

SQLTableVersionColumns::usage = 
"SQLTableVersionColumns[conn] retrieves an unordered description of a table's columns that are automatically updated when any value in a row is updated."

SQLUserDefinedTypeInformation::usage = 
"SQLUserDefinedTypeInformation[conn] retrieves a description of the user-defined types (UDTs) defined in a particular schema."

SQLUpdate::usage = 
"SQLUpdate[conn, table, cols, data]  updates data in a table in an SQL connection."

SQLConnections::usage = 
"SQLConnections[] returns a list of the open SQLConnections."

SQLServers::usage = 
"SQLServers[] returns a list of the open SQLServers."

SQLResultSets::usage = 
"SQLResultSets[] returns a list of the open SQLResultSets."

SQLConnectionPools::usage = 
"SQLConnectionPools[] returns a list of the open SQLConnectionPools."

$SQLTimeout::usage = 
"$SQLTimeout gives the default time in seconds that DatabaseLink waits while opening connections and executing database queries.";

$SQLUseConnectionPool::usage = 
"$SQLUseConnectionPool specifies whether a connection pool is used to retrieve a connection.";

$DatabaseLinkDirectory::usage =
"$DatabaseLinkDirectory gives the directory where DatabaseLink is installed."

Begin["`Package`"];

(*
 It is better to just create these package symbols alone in 
 the Package context, not their implementation as well.
*)
canonicalOptions
testOpt
JoinOptions
ThrowException
optionsErrorMessage
Spew
$databaseLinkPackageDirectory
$DatabaseLinkDirectory

End[]; (* DatabaseLink`Package` *)

(* Make the Package` symbols visible to all implementation files as they are read in. *)
AppendTo[$ContextPath, "DatabaseLink`Package`"]

Begin["`Private`"];

SetAttributes[canonicalOptions, {Listable}];
canonicalOptions[name_Symbol -> val_] := SymbolName[name] -> val;
canonicalOptions[expr___] := expr;

(*
 Utility for joining options
*)
testOpt[hash_, _[name_ , val_]] :=
	If[hash[name] === True, False, hash[name] = True]

JoinOptions[opts___] :=
	Module[{optList, found, ef},
		optList = Join[opts];
		ef = Select[optList, testOpt[found,#]&];
		Clear[found];
		ef
	]

ThrowException[symbol_Symbol, tagname_String, message_String] := Module[
	{exception = GetJavaException[]}, 
    Which[
        InstanceOf[exception, LoadJavaClass["java.lang.ClassNotFoundException"]],
        Message[JDBC::classnotfound, exception@getMessage[]],
                 
        exception@getMessage[] === Null,
        JLink`Exceptions`Private`$internalJavaExceptionHandler[symbol, tagname, message],
        
        True,
        Message[JDBC::error, exception@getMessage[]]
    ];
    Throw[$Failed];
];

optionsErrorMessage[opts_, sym_, e_] := Module[
	{notOpts, notResetOpts},
    notOpts = FilterRules[opts, Except[Options[sym]]];
    Scan[Message[sym::optx, #1, e]&, Map[First, notOpts]];
    notResetOpts = FilterRules[opts, Options[sym]];
    Scan[Message[sym::optreset, #1, e]&, Map[First, notResetOpts]];
]

Spew[args___] := If[TrueQ[DatabaseLink`SQL`Private`$Debug], Print[args]];


(* Set package directory used to find implementation files *)
$databaseLinkPackageDirectory = DirectoryName[System`Private`FindFile[$Input]];

$DatabaseLinkDirectory = $databaseLinkPackageDirectory;

End[]; (* DatabaseLink`Private` *)

Get[FileNameJoin[{$databaseLinkPackageDirectory, "Kernel", #}]] & /@ {
	"DataSources.m",
	"Execute.m",
	"Connections.m",
	"TablesAndColumns.m",
	"SCUD.m",
	"Transactions.m",
	"Servers.m",
	"JDBC.m",
	"Pools.m",
	"ResultSets.m",
	"StoredProcedures.m",
	
	"UI.m",
	"DataSourceWizard.m",
	"JDBCWizard.m",
	"DatabaseExplorer.m"
};

Begin["`SQL`Private`"];

(*===================================================================*)
(*====================== Formatting Resources =======================*)
(*===================================================================*)

summaryBoxIcon = Graphics[{Thickness[0.0625], 
  Style[{FilledCurve[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, 
      {0, 1, 0}}}, {{{15.236999999999998, 15.07}, {11.078, 17.829}, {11.078, 
      15.975000000000001}, {1.625, 15.975000000000001}, {1.625, 14.165000000000001}, 
      {11.078, 14.165000000000001}, {11.078, 12.31}, {15.236999999999998, 15.07}}}], 
    FilledCurve[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 
      1, 0}}}, {{{0., 9.792}, {4.159, 7.033}, {4.159, 8.887}, {13.612, 8.887}, {13.612, 
      10.697}, {4.159, 10.697}, {4.159, 12.551}, {0., 9.792}}}], 
    FilledCurve[{{{1, 4, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {0, 
      1, 0}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {0, 1, 0}, 
      {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {0, 1, 0}, {1, 3, 
      3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {0, 1, 0}}}, 
     {{{2.4789999999999996, 1.508}, {2.504, 1.327}, {2.554, 1.1920000000000002}, {2.63, 
      1.103}, {2.7670000000000003, 0.9390000000000001}, {3.002, 0.8580000000000001}, 
      {3.3339999999999996, 0.8580000000000001}, {3.533, 0.8580000000000001}, {3.695, 
      0.88}, {3.82, 0.923}, {4.055, 1.005}, {4.1739999999999995, 1.1580000000000001}, 
      {4.1739999999999995, 1.3820000000000001}, {4.1739999999999995, 1.513}, 
      {4.114999999999999, 1.613}, {4., 1.6860000000000002}, {3.885, 1.755}, {3.701, 
      1.817}, {3.4499999999999997, 1.87}, {3.022, 1.9649999999999999}, {2.601, 2.058}, 
      {2.3109999999999995, 2.159}, {2.154, 2.269}, {1.887, 2.4499999999999997}, {1.754, 
      2.7359999999999998}, {1.754, 3.125}, {1.754, 3.4789999999999996}, 
      {1.8840000000000001, 3.773}, {2.145, 4.009}, {2.4059999999999997, 4.243}, {2.789, 
      4.359999999999999}, {3.295, 4.359999999999999}, {3.718, 4.359999999999999}, {4.077, 
      4.25}, {4.376, 4.028}, {4.6739999999999995, 3.808}, {4.83, 3.4859999999999998}, 
      {4.843999999999999, 3.065}, {4.05, 3.065}, {4.035, 3.304}, {3.928, 3.473}, {3.73, 
      3.573}, {3.598, 3.64}, {3.4339999999999997, 3.673}, {3.238, 3.673}, 
      {3.0189999999999997, 3.673}, {2.8449999999999998, 3.63}, {2.715, 3.544}, {2.584, 
      3.4579999999999997}, {2.5189999999999997, 3.3379999999999996}, {2.5189999999999997, 
      3.184}, {2.5189999999999997, 3.042}, {2.583, 2.9359999999999995}, 
      {2.7119999999999997, 2.867}, {2.794, 2.82}, {2.969, 2.766}, {3.238, 
      2.7030000000000003}, {3.9319999999999995, 2.5389999999999997}, {4.237, 2.468}, 
      {4.465, 2.3719999999999994}, {4.616999999999999, 2.252}, {4.853, 2.065}, 
      {4.971000000000001, 1.796}, {4.971000000000001, 1.4429999999999998}, 
      {4.971000000000001, 1.082}, {4.831, 0.7809999999999999}, {4.552, 0.543}, {4.272, 
      0.304}, {3.877, 0.185}, {3.367, 0.185}, {2.8449999999999998, 0.185}, 
      {2.4359999999999995, 0.302}, {2.137, 0.537}, {1.839, 0.772}, {1.689, 1.097}, 
      {1.689, 1.508}, {2.4789999999999996, 1.508}}}], 
    FilledCurve[{{{1, 4, 3}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {1, 3, 3}, {1, 3, 3}, {1, 
      3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}}, {{1, 4, 
      3}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, 
      {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}}}, 
     {{{8.173, 0.912}, {8.220999999999998, 0.925}, {8.283, 0.9470000000000001}, 
      {8.359000000000002, 0.979}, {7.958, 1.36}, {8.383000000000001, 1.804}, {8.785, 
      1.4249999999999998}, {8.847999999999999, 1.554}, {8.892000000000001, 1.667}, 
      {8.917, 1.764}, {8.956000000000001, 1.908}, {8.976, 2.077}, {8.976, 2.27}, {8.976, 
      2.715}, {8.885000000000002, 3.0589999999999997}, {8.703, 3.3009999999999997}, 
      {8.522, 3.543}, {8.256, 3.665}, {7.907, 3.665}, {7.579000000000001, 3.665}, {7.318, 
      3.548}, {7.122999999999999, 3.3149999999999995}, {6.927, 3.083}, {6.83, 2.734}, 
      {6.83, 2.27}, {6.83, 1.728}, {6.970000000000001, 1.34}, {7.2490000000000006, 
      1.105}, {7.430000000000001, 0.9530000000000001}, {7.646999999999999, 
      0.8770000000000001}, {7.899, 0.8770000000000001}, {7.994, 0.8770000000000001}, 
      {8.086, 0.889}, {8.173, 0.912}}, {{9.674, 1.4429999999999998}, {9.604, 
      1.2169999999999999}, {9.502, 1.028}, {9.366000000000001, 0.8780000000000001}, 
      {9.821, 0.45}, {9.389000000000001, 0.}, {8.914, 0.451}, {8.769, 0.363}, 
      {8.642999999999999, 0.301}, {8.537999999999998, 0.265}, {8.360000000000001, 
      0.20600000000000002}, {8.147999999999998, 0.17600000000000002}, {7.901000000000001, 
      0.17600000000000002}, {7.385, 0.17600000000000002}, {6.958, 0.32999999999999996}, 
      {6.6209999999999996, 0.638}, {6.213, 1.009}, {6.009, 1.553}, {6.009, 2.27}, {6.009, 
      2.9939999999999998}, {6.218, 3.541}, {6.636, 3.9109999999999996}, 
      {6.979000000000001, 4.2139999999999995}, {7.404, 4.364999999999999}, {7.912, 
      4.364999999999999}, {8.425, 4.364999999999999}, {8.854000000000001, 4.205}, 
      {9.200999999999999, 3.885}, {9.602, 3.5149999999999997}, {9.803, 
      2.9959999999999996}, {9.803, 2.3299999999999996}, {9.803, 1.978}, {9.76, 
      1.6820000000000002}, {9.674, 1.4429999999999998}}}]}, 
   FaceForm[RGBColor[0.5, 0.5, 0.5, 1.]]], 
  Style[{FilledCurve[{{{0, 2, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 
      0}}}, {{{11.039, 4.245}, {11.866000000000001, 4.245}, {11.866000000000001, 0.998}, 
      {13.842, 0.998}, {13.842, 0.28600000000000003}, {11.039, 0.28600000000000003}, 
      {11.039, 4.245}}}]}, FaceForm[RGBColor[0.5, 0.5, 0.5, 1.]]]}, 
 PlotRangePadding -> 4, Background -> GrayLevel[0.93], Axes -> False, AspectRatio -> 1, 
 ImageSize -> {Automatic, Dynamic[3.5*(CurrentValue["FontCapHeight"]/
      AbsoluteCurrentValue[Magnification])]}, Frame -> True, FrameTicks -> None, 
 FrameStyle -> Directive[Thickness[Tiny], GrayLevel[0.7 ]]];

End[] (* `SQL`Private` *)

EndPackage[] (* DatabaseLink` *)
