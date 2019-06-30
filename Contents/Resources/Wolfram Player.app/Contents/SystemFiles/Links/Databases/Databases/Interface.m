Package["Databases`"]


PackageExport["$Databases"]               (* container for all defined schemas *)
PackageExport["$DatabaseAuthentications"] (* container for all authentications *)
PackageExport["$DBResourceObject"]

PackageExport["DatabaseModel"]
PackageExport["DatabaseView"] (* Not yet implemented, a stub *)
PackageExport["Databases"] (* An inert symbol to attach various general messages to *)

PackageExport["DatabaseRunQuery"]

PackageExport["DatabaseReferences"]
PackageExport["DatabaseQueryToSymbolicSQL"]
PackageExport["DBEntityQueryToSQLString"]

PackageExport["DatabaseStore"] (* container *)
PackageExport["DatabaseModelInstance"]
(*
** Inert query primitives. The query compiler which compiles a query built with these,
** is implemented in Databases`Database`
*)
PackageExport["DatabaseFunction"]
PackageExport["DatabaseSQLDistinct"]
PackageExport["DatabaseSQLAscending"]
PackageExport["DatabaseSQLDescending"]
PackageExport["DatabaseQueryMakeAlias"]
PackageExport["DatabaseWhere"]
PackageExport["DatabaseAnnotate"]
PackageExport["DatabaseAggregate"]
PackageExport["DatabaseOrderBy"]
PackageExport["DatabaseGroupBy"]
PackageExport["DatabaseSelectFields"]
PackageExport["DatabaseExcludeFields"]
PackageExport["DatabaseJoin"]
PackageExport["DatabaseLimit"]
PackageExport["DatabaseOffset"]
