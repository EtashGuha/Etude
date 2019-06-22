Package["Databases`SQL`"]

(*
**  The DBQueryObject encapsulates the symbolic SQL generation, and keeps the necessary
**  information to be able to reconstruct original top-level fields in the result of
**  the query execution. It wraps the DBQueryBuilderObject, and generates symbolic SQL
**  at the point of construction. This cleanly separates the state of query-building
**  from the state of symbolic SQL generation / query execution.
*)
PackageImport["Databases`"]  
PackageImport["Databases`Common`"] (* DBRaise, DBHandleError *)


PackageExport["DBQueryObject"]
PackageExport["DBQueryObjectQ"]


$externalFieldMapPattern[valuePattern_] :=
    KeyValuePattern[{(_ -> valuePattern) ...}]

$fullResolvedExternalFieldMapPattern[valuePattern_] := KeyValuePattern[{
    "Map" -> $externalFieldMapPattern[valuePattern],
    "Normalizer" -> _
}]

$externalFieldMapInputPattern = $externalFieldMapPattern[_DBPrefixedField]

$fullExternalFieldMapInputPattern = $fullResolvedExternalFieldMapPattern[_DBPrefixedField]

$missingField = Missing["NotAvailable", "InvalidFieldName"]

$queryObjectPattern = DBQueryObject @ KeyValuePattern[{
	"QueryBuilderInstance" -> _?DBQueryBuilderObjectQ,
	"SymbolicSQL" -> _?DBSymbolicSQLQueryQ,
	"FieldMap" -> _Association?AssociationQ,
	"ReverseFieldMap" -> _Association?AssociationQ,
	"PrefixedFields" -> {___DBPrefixedField},
	"RawToPrefixedFieldsMap" ->  _Association?AssociationQ,
	"ExternalFieldMap" -> Alternatives[
        $fullResolvedExternalFieldMapPattern[ _DBPrefixedField | $missingField ],
        None
    ],
	"RelationFunction" -> _
}]


DBQueryObjectQ[$queryObjectPattern] := True
DBQueryObjectQ[_] := False


(*
**  DBQueryObject Constructor
**
**  Takes a DBQueryBuilderObject, and optionally a field map, that maps external names
**  one may be using for query fields, to top-level field names (_DBPrefixedField).
**  It is allowed to pass non-existent _DBPrefixedField fields, DBQueryObject will
**  not raise an error for those, but will treat them correctly.
*)

Options[DBQueryObject] = Options[createQueryObjectData] = {"RelationFunction" -> Automatic}

DBQueryObject[
	builder_?DBQueryBuilderObjectQ,
	fieldMap:_:None,
	opts: OptionsPattern[]
] /; MatchQ[ fieldMap, Alternatives[
        $externalFieldMapInputPattern,
        $fullExternalFieldMapInputPattern,
        None
    ]
] :=
	DBQueryObject @ createQueryObjectData[builder, fieldMap, opts]

createQueryObjectData[builder_, None, opts: OptionsPattern[]] :=
	Append[
		builder @ asSQLWithFieldMap[], {
			"QueryBuilderInstance" -> builder,
			"PrefixedFields" -> builder @ DBGetPrefixedSelectedFields[],
			"ExternalFieldMap" -> None,
			"FieldPostProcess" -> builder @ DBGetFieldPostProcess[],
			"RelationFunction" -> OptionValue["RelationFunction"]
		}
	]

createQueryObjectData[
	builder_,
	fieldMap :$externalFieldMapInputPattern,
	opts: OptionsPattern[]
] :=
    createQueryObjectData[
        builder, <| "Map" -> fieldMap, "Normalizer" -> Identity|>,
		opts
    ]

createQueryObjectData[
	builder_,
	fieldMap : $fullExternalFieldMapInputPattern,
	opts: OptionsPattern[]
] :=
	Module[{resolvedMap, validFields, finalQuery},
		resolvedMap = AssociationThread[
			Keys @ fieldMap["Map"],
			Replace[
                builder @ DBGetPrefixedFields[Values @ fieldMap["Map"]],
                $Failed -> $missingField,
                {1}
            ]
		];
		validFields = Cases[resolvedMap, _DBPrefixedField];
        (*
        ** NOTE: currently we assume, that if external map is provided / passed,
        ** then only the fields mentioned in the map will be available.
        *)
		finalQuery = builder @ keepOnlySelectedFields[validFields];
		Append[
			createQueryObjectData[finalQuery, None, opts],
			"ExternalFieldMap" -> Append[fieldMap, "Map" -> resolvedMap]
		]
	]

(*  Property extractor *)
DBQueryObject[assoc_]?DBQueryObjectQ[prop_String] /; KeyExistsQ[assoc, prop] :=
	assoc[prop]


qo_DBQueryObject?DBQueryObjectQ["KeyConversionFunction"] :=
	With[{
        map = qo["FieldMap"],
        builder = qo["QueryBuilderInstance"],
        externalMap = qo["ExternalFieldMap"]
        },
        Function[fieldName,
            With[
                { dbFieldName = If[externalMap === None,
                    fieldName,
                    externalMap["Map"][externalMap["Normalizer"] @ fieldName]
                  ]
                },
                If[MissingQ[dbFieldName],
                    $missingField,
                    (* else *)
                    Replace[
                        builder @ DBResolveTopLevelField[dbFieldName, False], {
                            None :> $missingField,
                            resolvedRawField_ :> map[resolvedRawField]
                        }
                    ]
                ]
            ]
        ]
	]

qo_DBQueryObject?DBQueryObjectQ["PrefixedColumnNames"] :=
    If[qo["ExternalFieldMap"] =!= None,
        Values @ qo["ExternalFieldMap"]["Map"],
        (* else *)
        Values @ qo["RawToPrefixedFieldsMap"]
    ]

qo_DBQueryObject?DBQueryObjectQ["ColumnNames"] :=
    If[qo["ExternalFieldMap"] =!= None,
        Keys @ qo["ExternalFieldMap"]["Map"],
        (* else *)
        Values @ GroupBy[
            qo["PrefixedColumnNames"],
            extractName,
            Function @ If[
                Length[#] === 1,
                extractName[First[#]],
                Sequence @@ #
            ]
        ]
    ]

qo_DBQueryObject?DBQueryObjectQ["Schema"] :=
    qo["QueryBuilderInstance"] @ getSchema[]

extractName[DBPrefixedField[name_String]] := name
extractName[DBPrefixedField[_ -> name_]]  := extractName[DBPrefixedField[name]]
