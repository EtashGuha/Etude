Package["Databases`Database`"]

PackageImport["Databases`"]
PackageImport["Databases`Common`"] (* DBUnevaluatedPatternCheck *)
PackageImport["Databases`SQL`"] (* DBQueryBuilderObjectQ *)


PackageExport["DBUncompiledQueryQ"]
PackageExport["DBInertQueryQ"]
PackageExport["DBQueryQ"]


PackageScope["dbPropertyExtractorQ"]


$dbQueryPattern = Alternatives[
    _DatabaseQueryMakeAlias,
    _DatabaseWhere,
    _DatabaseJoin,
    _DatabaseSelectFields,
    _DatabaseGroupBy,
    _DatabaseOrderBy,
    _DatabaseSQLDistinct,
    _DatabaseAggregate,
    _DatabaseAnnotate,
    _DatabaseOffset,
    _DatabaseLimit,
    _DatabaseModel,
    _DatabaseModelInstance
]

$dbPropertyExtractorPattern = 
    _String | _DBPrefixedField | _Rule| {(_String | _Rule | _DBPrefixedField)...}

$dbOperatorFormQueryPattern = $dbQueryPattern[
    Except[$dbPropertyExtractorPattern]..
]


DBInertQueryQ[$dbQueryPattern] := True
DBInertQueryQ[$dbOperatorFormQueryPattern] := True
DBInertQueryQ[_] := False


dbPropertyExtractorQ[$dbPropertyExtractorPattern] := True
dbPropertyExtractorQ[_] := False


DBUncompiledQueryQ[_?(DBUnevaluatedPatternCheck[DBQueryBuilderObjectQ])] := True
DBUncompiledQueryQ[_?(DBUnevaluatedPatternCheck[DBInertQueryQ])] := True
DBUncompiledQueryQ[_] := False


DBQueryQ[_?(DBUnevaluatedPatternCheck[DBUncompiledQueryQ])] := True
DBQueryQ[_?DBSymbolicSQLQueryQ] := True
DBQueryQ[_SAFullQuery] := True (* To allow to use this in DatabaseQuery *)
DBQueryQ[_SAQueryString] := True (* To allow to use this in DatabaseQuery *)
DBQueryQ[_] = False
