Package["Databases`SQL`"]


PackageImport["Databases`"]
PackageImport["Databases`Common`"]


PackageScope["removeUnusedAnnotations"]
PackageScope["collapseQuery"]
PackageScope["optimizeQuery"]
PackageScope["hasNontrivialAnnotations"]


$maxCallapseIterations = 10

q_DBQueryBuilderObject @ removeUnusedAnnotations[ _ : {}] /; SameQ[
        q @ getType[], "NativeTable"
] := q

q_DBQueryBuilderObject @ removeUnusedAnnotations[] := 
	q @ removeUnusedAnnotations[q @ getSelectedFieldNames[]]

q_DBQueryBuilderObject @ removeUnusedAnnotations[usedRawFields_List] :=
	Module[{annots, allUsed, newq, allPotentiallyUsedFromChildren, prefixedUsed},
		annots =  q @ get["ProperFields"];
		allUsed = getAllUsedRawFields[usedRawFields, annots];
		(* Keep only really used annotations *)
		newq = q @ transform["ProperFields", KeyTake[allUsed]];
        newq = newq @ transform[
            "SelectedFields", 
            Function[fields, Intersection[fields, newq @ getResolvableFieldNames[]]]
        ];
		allPotentiallyUsedFromChildren = Join[
			Complement[allUsed, Keys @ annots],
			collectRawFields[
                newq @ get[{"Where", "Joins", "GroupBy", "OrderBy", "Limit", "Offset"}]
            ]
		];
		prefixedUsed = Values @ KeyTake[newq @ getRawToPrefixedFieldsMap[],allUsed];
		newq @ DBValues[prefixedUsed] @ transform[
			"Tables", 
            Map[ # @ removeUnusedAnnotations[allPotentiallyUsedFromChildren]&]
		]
	]
    
    
q_DBQueryBuilderObject @ hasNontrivialAnnotations[
    fields:{___DBRawFieldName} | All : All
] :=
    With[{selectedFields = q @ getSelectedFieldNames[]},
        Composition[
            MemberQ[Except[_DBSQLField]],
            Lookup[#, "Expression", {}]&,
            Values,
            KeyTake @ If[rawFields === All, 
                selectedFields, 
                Intersection[fields, selectedFields]
            ]
        ] @ q @ get["ProperFields"]     
    ]
    

q_DBQueryBuilderObject @  collapseChildQuery[
    child_?nativeTableWrapperQ
] /; AnyTrue[q @ getTables[], # === child&] :=
	Module[{annots, innerTable, newq},
		annots = child @ get["ProperFields"];
		innerTable = First[child @ getTables[]];
		newq = ReplaceAll[
			q @ transform[
                "Tables", 
                Function[tables, 
					Replace[tables, t_ /; t === child :> innerTable, {1}]
				]
			],
			Normal @ KeyMap[DBSQLField[_,#]&] @ Map[#Expression&] @ annots
		];
		newq @ append["ProperFields",  KeyTake[
			annots, newq @ getSelectedFieldNames[]
		]] @ transform[ "Joins", ReplaceAll[
            (* TODO: use a more precise rule here  *)
			child @ DBGetName[] -> innerTable @ DBGetName[]
		]]
	]    
    
    
q_DBQueryBuilderObject @ collapseQuery[] := 
	Switch[q @ getTables[],
		{} | ({table_} /; table @ getType[] === "NativeTable"),
			q,
		{___, t_ /; childQueryCollapsibleQ[q, t], ___},
			Composition[
				# @ transform[
                    "ProperFields", 
                    KeyTake[# @ getSelectedFieldNames[]]
                ]&,
				Fold[
					Function[{query, ch}, query @ collapseChildQuery[ch]],
					#,
					Select[# @ getTables[], Function[ch, childQueryCollapsibleQ[#, ch]]]
				]&
			] @ q,
		{__},
			If[# === q, #, # @ collapseQuery[]]& @
				FixedPoint[
					Composition[
						# @ transform["Tables", Map[# @ collapseQuery[]&]]&,
						DBIterationLimiter[$maxCallapseIterations, collapseQuery]
					],
					q
				],
		_,
			DBRaise[collapseQuery, "invalid_state", {q}]
	]    
    

q_DBQueryBuilderObject @ optimizeQuery[] := 
    q @ removeUnusedAnnotations[] @ collapseQuery[]
    	    
    
nativeTableWrapperQ[q_?DBQueryBuilderObjectQ] := And[
	!q @ isAggregateQuery[],
    !MemberQ[q @ get[{"Limit", "Offset"}], Except[None], {1}],
    MatchQ[
		q @ getTables[], 
		{inner_ /; inner @ getType[] === "NativeTable"}
	],
    FreeQ[ q @ get["ProperFields"], DBQueryBuilderObject]
] 
    
nativeTableWrapperQ[_] := False	    

DBDefError @ nativeTableWrapperQ


childQueryCollapsibleQ[
    parent_?DBQueryBuilderObjectQ, 
    child_?DBQueryBuilderObjectQ
] := 
    And[
        nativeTableWrapperQ[child],
        !child @ hasNontrivialAnnotations[
            collectRawFields[parent @ get[{"GroupBy", "OrderBy"}]]
        ]
    ]

childQueryCollapsibleQ[parent_?DBQueryBuilderObjectQ] := 
    Function[child, childQueryCollapsibleQ[parent, child]]

DBDefError @ childQueryCollapsibleQ


getAllUsedRawFields[usedRawFields_List, annotations_?AssociationQ] :=
	DeleteDuplicates @ Flatten @ FixedPointList[
		collectRawFields[
			Lookup[Values @ KeyTake[annotations, #], "Expression"]
		]&,
		usedRawFields
	]
    
DBDefError @ getAllUsedRawFields   


collectRawFields[expr_] := 
    DeleteDuplicates @ Cases[
        expr, _DBRawFieldName, {0, Infinity}, Heads -> True
    ]
	
    
    