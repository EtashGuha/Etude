Package["Databases`Database`"]


PackageImport["Databases`"]
PackageImport["Databases`Common`"]
PackageImport["Databases`SQL`"]


PackageScope["synthesizeAnnotations"]
PackageScope["syntheticFieldsDecorator"]


(*  ============       Synthetic fields and auto-annotations        ============   *)


(*
**  A helper function to process the passed field of list of fields. It looks for the
**  fields of the form _DatabaseFunction or name -> _DatabaseFunction. For the former,
**  it generates synthetic names. For an individual field, it returns a list of 3
**  elements: {final-field-name, annotation, was-field-syntehtic}. So, for example,
**
**  autoAnnotateFields[DBPrefixedField["prefix" -> "field"], True] -> {
**      DBPrefixedField["prefix" -> "field"], None, False
**  }
**
**  autoAnnotateFields[DatabaseFunction[x, x["prop"]], True] -> {
**      "new_synthetic_name_1", "new_synthetic_name_1" -> DatabaseFunction[x, x["prop"]], True
**  }
**
**  autoAnnotateFields["myField" -> DatabaseFunction[x, x["prop"]], True] -> {
**      "myField", "myField" -> DatabaseFunction[x, x["prop"]], False
**  }
**
**  If the second argument is False, then synthetic fields are not allowed, and the
**  error is raised.
**
**  For a list of fields, it processes each one, and returns an assoc with a total
**  list of field names, annotations, and new (synthetic) field names.
**
*)
autoAnnotateFields[(head: DatabaseSQLAscending | DatabaseSQLDescending)[field_], allowSyntheticFields_] :=
    Replace[
        autoAnnotateFields[field, allowSyntheticFields],
        {f_, annot_, isSynthetic_} :> {head[f], annot, isSynthetic}
    ]

autoAnnotateFields[field: $DBTopLevelFieldNamePattern, _] := {field, None, False}

autoAnnotateFields[field_DatabaseFunction, True] :=
    With[{newFieldName = DBGenerateUniquePropertyName[]},
        {newFieldName, newFieldName -> field, True}
    ]

autoAnnotateFields[field_DatabaseFunction, False] :=
	DBRaise[autoAnnotateFields, "synthetic_fields_not_allowed_by_an_operation", {field}]

autoAnnotateFields[field: (name_String -> _DatabaseFunction), _] := {name, field, False}

autoAnnotateFields[fields_List, allowSyntheticFields_] :=
	With[{processed = Transpose @ Map[autoAnnotateFields[#, allowSyntheticFields]&, fields]},
		<|
			"Fields" -> First @ processed,
			"Annotations" -> DeleteCases[processed[[2]], None],
			"NewFieldNames" -> Pick[First @ processed, Last @ processed]
		|>
	]

DBDefError @ autoAnnotateFields


(*
**  This function returns a spec to be used for auto-annotation / syntehtic field
**  generation. NOTE: it does not itself operate on the query, but rather prepares
**  functions which can be applied to it.
**
**  It takes a list of fields, where fields like fieldName ->  _DatabaseFunction and
**  just _DatabaseFunction are allowed, and the flags instructing whether the pure
**  _DatabaseFunction fields are allowed and whether synthetic fields should be
**  removed after the main operation.
**
**  It returns an assoc with: a list of processed field names (where f_DatabaseFunction
**  - based fields are no longer, but only their names, passed in or generated), and
**  two functions: annotation function (to be applied before the main query operation),
**  and field removal function (to be applied after the main query operation).
*)
synthesizeAnnotations[
    fields: {DBDelayedPattern[$orderByFieldPattern] ... },
    allowSyntheticFields: True | False : True,
    removeFieldsAfter: True | False | Automatic : Automatic
] :=
	With[{processedFields = autoAnnotateFields[fields, allowSyntheticFields]},
		<|
			"AnnotationFunction" -> DBFnOrIdentity[
                Function[annots, annots =!= {}],
                Function[annots, DatabaseAnnotate[annots]]
            ] @ processedFields["Annotations"]
            ,
			"FieldRemovalFunction" ->
                DBFnOrIdentity[
                    Function[newNames, newNames =!= {} && !TrueQ[!removeFieldsAfter]],
                    Function[newNames, DatabaseExcludeFields[
                        Replace[newNames, {(DatabaseSQLAscending | DatabaseSQLDescending)[f_] :> f}, {1}]
                    ]]
                ] @ processedFields["NewFieldNames"]
			,
			"Fields" -> processedFields["Fields"]
		|>
	]

DBDefError @ synthesizeAnnotations


(*
**  A convenience decorator to transform the main query operation for syntehtic field
**  generation.
**
**  Takes: 1. a function encapsulating the new query operation, to be normally applied
**  to the inner query to form the total query, and 2. The synthetic annotation spec,
**  as defined by / returned from  synthesizeAnnotations[]
**
**  Returns a decorated function, to be applied to the inner query, which encapsulates
**  the synthetic fields annotation logic.
*)
syntheticFieldsDecorator[queryOp_, syntheticAnnotationsSpec_Association?AssociationQ] :=
    Composition[
        syntheticAnnotationsSpec["FieldRemovalFunction"],
        queryOp[syntheticAnnotationsSpec["Fields"]],
        syntheticAnnotationsSpec["AnnotationFunction"]
    ]

DBDefError @ syntheticFieldsDecorator
