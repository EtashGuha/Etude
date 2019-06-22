Package["Databases`SQL`"]

PackageImport["Databases`"]  
PackageImport["Databases`Common`"] (* DBRaise, DBHandleError *)
PackageImport["Databases`Schema`"] (* DBTyped *)


PackageExport["DBRemoveSeveralFieldsFromSelected"]
PackageExport["DBGetPrefixedSelectedFields"]
PackageExport["DBGetPrefixedFields"]
PackageExport["DBResolveTopLevelSelectedField"]
PackageExport["DBGetFieldType"]


PackageScope["stripFullNames"]
PackageScope["fieldToF"]
PackageScope["getFieldName"]
PackageScope["addField"]
PackageScope["getProperFields"]
PackageScope["getProperFieldNames"]
PackageScope["getSelectedFieldNames"]
PackageScope["getSelectedFields"]
PackageScope["getInnerFields"]
PackageScope["removeFieldFromSelected"]
PackageScope["keepOnlySelectedFields"]
PackageScope["getResolvableFieldNames"]
PackageScope["getRawToPrefixedFieldsMap"]
PackageScope["searchRawField"]



fieldToF[table_ -> field_] := DBSQLField[table, field]

getFieldName[_ -> field_] := field
getFieldName[DBSQLField[_: None, f_]] := f

fieldQueryName[name_ -> _] := name
fieldQueryName[DBSQLField[name_: None, _]] := name


stripFullNames[fields_] := fields /. {field_SQLF :> getFieldName[field]}


(* ============       Selected and proper fields - accessors        ========== *)

q_DBQueryBuilderObject @ getProperFields[filterPattern_: All] :=
    Replace[
        Normal @ KeySelect[
            q @ get @ "ProperFields",
            MatchQ[filterPattern /. {All -> _, None -> Except[_]}]
        ],
        (fieldName_ -> assoc_?AssociationQ) :> (assoc["Expression"] -> fieldName),
        {1}
    ]

q_DBQueryBuilderObject @ getProperFieldNames[filterPattern_: All] :=
    Values @ q @ getProperFields[filterPattern]


q_DBQueryBuilderObject @ getSelectedFieldNames[] := q @ get["SelectedFields"]


(*
** This gives all "exported" fields, coming from various "constituent" tables,
** but now they are qualified by the actual name / alias of the current table / query
*)
q_DBQueryBuilderObject @ getSelectedFields[] := q @ DBGetName[] -> q @ getSelectedFieldNames[]

(*
** Inner fields for a given query are all "exported" / selected fields of its inner tables.
** In general, inner fields can have conflicting short names, and that's Ok.
*)
q_DBQueryBuilderObject @ getInnerFields[] :=
    Association @ Map[# @ getSelectedFields[]&, q @ getTables[]]

(*
** Returns a list of field names which are resolvable (from the outside) for this
** query. These are basically fields, that can be used in SQL expressions for
** operations chained on this query.
*)
q_DBQueryBuilderObject @ getResolvableFieldNames[] :=
    Cases[
        (* Excluding string fields when q has type "NativeTable" *)
        $rawFieldNamePattern
    ] @ Join[
        Flatten @ Values @ q @ getInnerFields[],
        q @ getProperFieldNames[]
    ]


q_DBQueryBuilderObject @  searchRawField[
    field_, table:_String | All : All, searchInner: True | False : False 
] := iSearchRawField[q, field, table, searchInner]


iSearchRawField[q_, field_, ___] /; KeyExistsQ[q @ get["ProperFields"], field] := 
	{ None -> field }
	
iSearchRawField[q_, field_, table:_String | All : All, searchInner: True | False : False ] := 
	With[{
			tables = If[StringQ[table], 
				Select[# @ DBGetName[] === table&],
				Identity
			][Select[q @ getTables[], # @ getType[] =!= "NativeTable"&]],
			selectedInMain = q @ getSelectedFieldNames[]
		},
		If[# === Null, {}, {#}] & @ Scan[
			Function[t,
				If[And[
					MemberQ[t @ getSelectedFieldNames[], field],
					TrueQ[searchInner] || MemberQ[selectedInMain, field]	
				],
					Return[t @ DBGetName[] -> field]
				]
			],
			tables
		]
	]
    		

(* =======    Selected field manipulations (adding, removing, renaming)   ===== *)

(*
** Removes a field from a list of selected fields in a given query.
** The field in  question may be one of the inner fields (that is, belonging to
** the list of selected fields on one of the inner tables), or it can be one of
** the annotated / generated fields (for which the table is None)
*)

q_DBQueryBuilderObject @ removeFieldFromSelected[
    table: _String | None -> field: $DBTopLevelFieldNamePattern,
    errorOnFieldNotFound: (True | False ) : True,
    strictFieldResolutionMode: (True | False ) : True
] := With[{resolved = q @ DBResolveTopLevelField[
            field, errorOnFieldNotFound, strictFieldResolutionMode
        ]},
        If[resolved === None,
            q,
            (* else *)
            q @ removeFieldFromSelected[table -> resolved ]
        ]
    ]


q_DBQueryBuilderObject @ removeFieldFromSelected[
    table: _String | None -> field : $rawFieldNamePattern
] :=
    q @ transform[
        "SelectedFields", DeleteCases[field]
    ]

q_DBQueryBuilderObject @ removeFieldFromSelected[
    field: $DBTopLevelFieldNamePattern,
    errorOnFieldNotFound: (True | False ): True,
    strictFieldResolutionMode: (True | False ) : True
] :=
    With[{resolved = q @ DBResolveTopLevelField[
            field, errorOnFieldNotFound, strictFieldResolutionMode
        ]},
        If[resolved === None,
            Return[q] (* No field found, but that is Ok *)
        ];
        q @ removeFieldFromSelected[resolved, errorOnFieldNotFound]
    ]


q_DBQueryBuilderObject @ removeFieldFromSelected[
    field : $rawFieldNamePattern,
    errorOnFieldNotFound: (True | False ) : True
] :=
    Module[{resolved, resolvedTable},
        (* Resolve the inner table, to which the field belongs inside the query *)
        If[errorOnFieldNotFound,
            resolvedTable = fieldQueryName @ q @ resolveFieldEnsureUnique[
                field, All, <|"SearchInnerFields" -> False|>
            ],
            (* else *)
            resolved = q @ resolveField[field, All, <|"SearchInnerFields" -> False|>];
            If[Length[resolved] == 0,
                (* No field found, return back the original query *)
                Return[q]
            ];
            resolvedTable = fieldQueryName @ First @ resolved
            (* We know there will be exactly one here *)
        ];
        q @ removeFieldFromSelected[resolvedTable -> field]
    ]

(*
** Can't add this def to  removeFieldFromSelected, due to the prefix field / list of
** string fields ambiguity. TODO: well, now that we changed the field resolution
** mechanism, we can.
*)
q_DBQueryBuilderObject @ DBRemoveSeveralFieldsFromSelected[
    fields: {$DBTopLevelFieldNamePattern...},
    errorOnFieldNotFound_: True,
    strictFieldResolutionMode: (True | False ) : True
] :=
    Fold[
        # @ removeFieldFromSelected[#2, errorOnFieldNotFound, strictFieldResolutionMode]&,
        q,
        fields
    ]

(*
** Low - level annotation. Add a new field / aliased expression, to the SELECT list
*)

q_DBQueryBuilderObject @ addField[fieldName_String -> DBTyped[expr_, type_], addToParent_: False] :=
    Module[{result = q, rawFieldName = createFieldName[fieldName, type], previousRawField},
        (*
        ** If the field with this name was in selected fields, de-select it and then
        ** actually hard-remove it from annotated fields of the query.
        *)
        previousRawField =  q @ DBResolveTopLevelField[
            DBPrefixedField[fieldName],
            False,
            True
        ];
        If[previousRawField =!= None,
            result = result @ removeFieldFromSelected[
                previousRawField, False
            ] @ transform[
                "ProperFields", KeyDrop[previousRawField]
            ]
        ];

        result @  append[ (* Add full field spec to "ProperFields" property *)
            "ProperFields",
            rawFieldName -> <|
                (*
                ** Setting the full name resolution to False to allow correlated subqueries
                *)
                "Expression" -> q @ resolveExpression[expr, False],
                "Alias" -> rawFieldName
            |>
        ] @ append[ (* Add field name to selected fields  *)
            "SelectedFields", rawFieldName
        ] @ transform[ (* Add field to the field prefix trie, and return *)
            "FieldPrefixTrie",
            Function[trie, trie @ "addField"[fieldName -> rawFieldName, addToParent]]
        ]
    ]

q_DBQueryBuilderObject @ addField[fieldName_String -> expr_] :=
    q @ addField[fieldName -> DBTyped[expr, DBType[DBTypeUnion[]]]]

q_DBQueryBuilderObject @ addField[annotations_?AssociationQ] :=
    q @ addField[Normal @ annotations]

q_DBQueryBuilderObject @ addField[annotations: {(_String -> _)...}] :=
    Fold[# @ addField[#2]&, q, annotations]


(*
** Only keeps the fields from a passed list, as selected fields.
** Resolves fields as needed
*)

q_DBQueryBuilderObject @ keepOnlySelectedFields[fields_List] := With[
    {f = Map[q @ resolveFieldName[#]&, fields]},
    If[
        DuplicateFreeQ[f],
        q @ set["SelectedFields" -> f] @ set[
            If[
                SubsetQ[
                    q @ DBGetPrefixedFields[fields],
                    q @ DBGetPrefixedFields[
                        Replace[
                            q @ get["PrimaryKey"],
                            None -> {}
                        ]
                    ]
                ],
                <||>,
                "PrimaryKey" -> None
            ]
        ],
        DBRaise[keepOnlySelectedFields, "duplicate_fields_found", {fields, f}]
    ]
]
    
(*
**  Returns prefixed form for fields, specified by their (short) string names, raw
**  fields, or actually prefixed fields (in which case, a canonical form of the
**  prefixed field will be returned for each input prefixed field - which may differ
**  since the resolver is liberal and allows shorter prefixed forms).
**
**  By default (All), all available / resolvable fields in the query are considered.
*)
q_DBQueryBuilderObject @ DBGetPrefixedFields[] := q @ DBGetPrefixedFields[All]

q_DBQueryBuilderObject @ DBGetPrefixedFields[All] :=
    q @ DBGetPrefixedFields[q @ getResolvableFieldNames[]]

q_DBQueryBuilderObject @ DBGetPrefixedFields[None] :=
	None


(*  The case of a single field *)
q_DBQueryBuilderObject @ DBGetPrefixedFields[name: _String | _DBRawFieldName | _DBPrefixedField] :=
    First @ q @ DBGetPrefixedFields[{name}]

(*  The case when all fields are raw fields *)
q_DBQueryBuilderObject @ DBGetPrefixedFields[rawFields: {___DBRawFieldName}] :=
    resolveWithFailed[
        q @ getFieldPrefixTrie[] @ "getPrefixedFields"[rawFields, True],
        Identity
    ]

(*  Wrap all string names, if any,  into DBPrefixedField *)
q_DBQueryBuilderObject @ DBGetPrefixedFields[
    fields: {(_String | _DBPrefixedField | _DBRawFieldName)...}
] :=
    With[{stringDressed = Replace[fields, name_String :> DBPrefixedField[name], {1}]},
        q @ DBGetPrefixedFields[stringDressed] /; stringDressed =!= fields
    ]

(*
**  Resolve all prefixed fields (if any) to raw fields, and then call DBGetPrefixedFields
**  on the result. Note that DBResolveTopLevelField[] will only resolve the field if
**  is is currently resolveable, as per getResolvableFieldNames[].
*)
q_DBQueryBuilderObject @ DBGetPrefixedFields[fields: {(_DBPrefixedField | _DBRawFieldName)...}] :=
    With[{raw = Replace[fields, f_DBPrefixedField :> q @ DBResolveTopLevelField[f, False], {1}]},
        If[!DuplicateFreeQ[DeleteCases[raw, None]],
            DBRaise[DBGetPrefixedFields, "duplicate_fields_encountered", {fields}],
            (* else *)
            resolveWithFailed[raw, Function[q @ DBGetPrefixedFields[#]]]
        ]
    ]
    
    
resolveWithFailed[fields_, resolver_] :=
    Module[{ pos, result},
        result = ConstantArray[$Failed, Length[fields]];
        pos = Flatten @ Position[fields, Except[None], {1}, Heads -> False];
        result[[pos]] = resolver[fields[[pos]]];
        result
    ]


(*
**  Returns a map of raw fields to prefixed fields, for all currently selected fields
**  in the query (all fields currently in the SELECT list).
*)
q_DBQueryBuilderObject @ getRawToPrefixedFieldsMap[] :=
    With[{rawFields = q @ getSelectedFieldNames[]},
        AssociationThread[rawFields, q @ DBGetPrefixedFields[rawFields]]
    ]

(*
**  Returns prefixed form for all currently selected fields in the query. The order
**  in which the fields are returned, is implementation-defined.
*)
q_DBQueryBuilderObject @ DBGetPrefixedSelectedFields[] := Values @ q @ getRawToPrefixedFieldsMap[]

q_DBQueryBuilderObject @ DBResolveTopLevelSelectedField[
    field: _String | _DBPrefixedField
] := 
With[{resolved = q @ DBResolveTopLevelField[field, True]},
    If[MemberQ[DBRawFieldID[q @ getSelectedFieldNames[]], DBRawFieldID[resolved]],
        resolved, 
        (* else *)
        DBRaise[
            DBResolveTopLevelSelectedField, 
            "no_field_found",
            {field}
        ]
    ]
]

q_DBQueryBuilderObject @ DBGetFieldType[
    field: _String | _DBPrefixedField, strict: True | False : False
] :=
    DBType[
        If[TrueQ[strict], 
            q @ DBResolveTopLevelSelectedField[field],
            q @ DBResolveTopLevelField[field]
        ]
    ]
    
q_DBQueryBuilderObject @ DBGetFieldType[
    fields:_List | _Association?AssociationQ, strict: True | False : False
] :=
    DBType[CompoundElement[Map[q @ DBGetFieldType[#, strict] & , fields]]]
