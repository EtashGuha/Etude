Package["Databases`SQL`"]


PackageImport["Databases`"]
PackageImport["Databases`Common`"]
PackageImport["Databases`Schema`"]


PackageExport["DBQueryPrefixTrie"]
PackageExport["DBQueryPrefixTrieQ"]
PackageExport["DBPrefixedField"]
PackageExport["DBPrefixedFieldParts"]
PackageExport["DBPrefixedFieldPartsList"]
PackageExport["DBReallyPrefixedFieldQ"]
PackageExport["DBNormalizePrefixedFields"]


DBPrefixedField[DBPrefixedField[f_]] := DBPrefixedField[f]
DBPrefixedField[prefix_ -> DBPrefixedField[f_]] := DBPrefixedField[prefix -> f]
DBPrefixedField[pre_Rule, rest_] := DBPrefixedField @ MapAt[
	Function[lhs, lhs -> rest],
	pre,
	ConstantArray[-1, Depth[pre] -1]
]
DBPrefixedField[pre_, rest_] := DBPrefixedField[pre -> rest]

DBPrefixedFieldPartsList[DBPrefixedField[field_]] := DBPrefixedFieldPartsList[field]

DBPrefixedFieldPartsList[field_String] := {Inherited, field}

DBPrefixedFieldPartsList[prefix_ -> field_] := DBPrefixedFieldPartsList[prefix-> field, DBToLinkedList[{}]]

DBPrefixedFieldPartsList[prefix_ -> field_, prefixes : DBLinkedListPattern[]] :=
	DBPrefixedFieldPartsList[field, DBLinkedList[prefix, prefixes]]

DBPrefixedFieldPartsList[field_String, prefixes : DBLinkedListPattern[]] := Join[
	DBFromLinkedList @ prefixes,
	{field}
]


DBPrefixedFieldParts[field_DBPrefixedField] := With[{parts = DBPrefixedFieldPartsList[field]},
	{
		Fold[Rule[#2, #1]&, Most[parts]],
		Last[parts]
	}
]


DBReallyPrefixedFieldQ[DBPrefixedField[prefix_ -> field_]] := True
DBReallyPrefixedFieldQ[_] = False

DBNormalizePrefixedFields[None] := None

DBNormalizePrefixedFields[l: {___DBPrefixedField}] := normalizeField /@ l
DBDefError @ DBNormalizePrefixedFields  


normalizeField[field:DBPrefixedField[_String]] := field    
normalizeField[field_DBPrefixedField] := DBPrefixedField[Apply[
	Rule,
	Take[DBPrefixedFieldPartsList[field], -2]
]]


normalizeRules[rule_Rule] := normalizeRules[{rule}]
normalizeRules[rules: {___Rule}] := normalizeRules[Association @ rules]
normalizeRules[rules_Association?AssociationQ] := rules;


$fieldRulePattern = _String -> _

$fieldsMapPattern = Replace[$fieldRulePattern, x_ :> x | { x ... } | <| x ... |>]
(* One way to inject into assoc *)

$prefixedFieldPattern = _String | (_ -> inner_ /; MatchQ[inner, $prefixedFieldPattern])

$properFieldsPattern = Replace[$fieldRulePattern, p_ :> <| p ... |>] (* Inject into assoc *)


DBQueryPrefixTrieQ[
    DBQueryPrefixTrie[
        KeyValuePattern[{
            "Alias" -> _,
            "Parents" -> {___Association?AssociationQ},
            "ProperFields" -> $properFieldsPattern
        }],
        meta_Association?AssociationQ
    ]
] := True

DBQueryPrefixTrieQ[_] := False

Options[DBQueryPrefixTrie] = {
    "KeepDefaultAliasesInPrefixedFields" -> False
}

DBQueryPrefixTrie @ (t_new)[args___] := DBQueryPrefixTrie[t][args]

addCoreMethods[DBQueryPrefixTrie]

DBDefMethodChaining[DBQueryPrefixTrie]


(*  Constructor *)
DBQueryPrefixTrie @ new[
    rawFieldPattern_,
    fieldRules: $fieldsMapPattern : <||>,
    startingAlias_:None,
    opts : OptionsPattern[{DBQueryPrefixTrie}]
] :=
	DBQueryPrefixTrie[
		<|
			"Alias" -> <| "Value" -> startingAlias, "Sticky" -> False |>,
			"Parents" -> {},
			"ProperFields" -> normalizeRules @ fieldRules
		|>,
        Association @ Flatten @ Join[
            Options[DBQueryPrefixTrie], 
            { opts },
            { "RawFieldPattern" -> rawFieldPattern }
        ]
	]


DBDefMethod @ "getMeta"[t_DBQueryPrefixTrie]:= Last @ t


DBDefMethod @ "getAlias"[t_DBQueryPrefixTrie] := t @ get["Alias" -> "Value"]


DBDefMethod @ "setAlias"[t_DBQueryPrefixTrie, alias_] :=
    t @ set["Alias" -> "Value" -> alias]


DBDefMethod @ "setSticky"[t_DBQueryPrefixTrie, sticky_:True] :=
    t @ set["Alias" -> "Sticky" -> sticky]

(*  Adds another trie as a parent node for the current trie / node *)
DBDefMethod @ "addParent"[
    t_DBQueryPrefixTrie,
    DBQueryPrefixTrie[assoc_, _] ? DBQueryPrefixTrieQ
] :=
	t @ append["Parents", assoc]


(*  Adds one or more fields of the form "fieldName" -> raw-field-pattern *)
DBDefMethod @ "addField"[t_DBQueryPrefixTrie, fieldInfo : $fieldsMapPattern, addToParent_: False] :=
    If[
		TrueQ[addToParent],
		t @ transform[
			"Parents",
			Function[
				MapAt[
					Append[ normalizeRules @ fieldInfo ],
					#,
					{1, "ProperFields"}
				]
			]
		],
		t @ transform["ProperFields", Append[ normalizeRules @ fieldInfo ]]
	]


(*
**  Grows a trie by adding a child node with alias <alias>, so that the current node
**  becomes its parent node. Can optionally add some fields, which would be attached
**  to this new child node. This is an operation one needs when generically aliasing
**  some query with a new alias.
*)
DBDefMethod @ "extend"[
	t_DBQueryPrefixTrie,
	alias : Except[$fieldsMapPattern] : None,
	fieldInfo : $fieldsMapPattern : <||>
] := 
    With[{meta = t @ "getMeta"[]},
        DBQueryPrefixTrie @ new[
            meta["RawFieldPattern"], fieldInfo, alias, Normal @ meta
        ] @ "addParent"[t]
    ]
	

DBDefMethod @ "setAliasOrExtend"[t_DBQueryPrefixTrie, alias_, sticky_:False] :=
    If[ And[
            t @ "getAlias"[] === None,
            (*  
            ** The check below is needed since otherwise all proper fields that 
            ** currently exist for t, will not carry this new alias. So, resetting 
            ** None to alias on the same node is only fine it the node does not 
            ** have proper fields.
            *)
            t @ get["ProperFields"] === <||> 
        ],
        t @ "setAlias"[alias],
        (* else *)
        t @ "extend"[alias]
    ] @ "setSticky"[sticky]


(*
**  Resolves one or several prefixed fields to their corresponding raw fields. If
**  a field can not be resolved / corresponding raw field found, None is returned.
*)
DBDefMethod @ "fieldResolve"[
    t_DBQueryPrefixTrie, fields_List, strictResolve: (True | False) : False
] :=
	Map[t @ "fieldResolve"[#, strictResolve]&, fields]

DBDefMethod @ "fieldResolve"[
	t_DBQueryPrefixTrie,
	DBPrefixedField[field: $prefixedFieldPattern] | (field: $prefixedFieldPattern),
    strictResolve: (True | False) : False
] := fieldResolve[field, t, strictResolve]


(*
**  Collects raw fields in a trie, and returns their prefixed field counterparts.
**  If a specific list of raw fields is added as a second argument (it is All by
**  default), then only for those fields the prefixed fields are returned. If the
**  order of resulting prefixed fields should correspond to the orider of the raw
**  fields in the passed list, the third argument must be set to True (it is False
**  by default).
**
**  Note that this operation can be viewed as an inverse for <fieldResolve>.
*)
DBDefMethod @ "getPrefixedFields"[
    t_DBQueryPrefixTrie,
    rawfields: (_List | All) : All,
    keepOrder: (True | False) : False
] :=
    With[{
        prefixedFieldPostprocessor = If[
            TrueQ[(t @ "getMeta"[])["KeepDefaultAliasesInPrefixedFields"]],
            Identity,
            Replace[#, (None -> x_) :> x, {0, Infinity}]&
        ],
        (* We need this dummy extension to inspect the very last (latest) 
        ** alias. So this is a trick. *)
        dummyExtendedTrie = t @ "extend"[]
        },
        With[{
            prefixed = prefixedFieldPostprocessor @ getPrefixedFields[
                dummyExtendedTrie, 
                rawfields, 
                (dummyExtendedTrie @ "getMeta"[])["RawFieldPattern"]
            ]},
            Which[
                !TrueQ[keepOrder] || rawfields === All,
                    prefixed,
                Length[rawfields] =!= Length[prefixed],
                    DBRaise[
                        DBQueryPrefixTrie,
                        "can_not_keep_original_fields_order",
                        {},
                        <|
                            "OriginalFields" -> rawfields,
                            "PrefixedFields" -> prefixed,
                            "Details" ->
                                "raw and prefixed field sets should be the same length to keep field order",
                            "PrefixeTrie" -> t
                        |>
                     ],
                True,
                    Permute[
                        prefixed,
                        Check[
                            FindPermutation[Map[t @ "fieldResolve"[#] &, prefixed], rawfields],
                            DBRaise[
                                FindPermutation, 
                                "prefix_raw_fields_mismatch", 
                                {t, prefixed, rawfields}
                            ]
                        ]
                    ]
            ]
        ]
    ]

(*  ==============     Prefixed field resolution to raw fields     =============== *)


(* Entry point *)
fieldResolve[field_, trie_?DBQueryPrefixTrieQ, strictResolve: (True | False) : False] :=
    fieldResolve[field, trie @ getDataAssoc[], strictResolve, False]

(* The case of string field name, no prefixes *)
fieldResolve[
    field_String,
    trieAssoc_Association?AssociationQ,
    strictResolve: True | False,
    isJoinParentNode: True | False
] :=
	With[{properField = trieAssoc["ProperFields", field]},
        Which[
            (*
            **  If field exists / was defined at the level of the current node,
            **  look up the corresponding raw field from the field map, and return it.
            *)
            !MissingQ[properField],
                properField,
            (*
            **  If no field for the current node was found, and we are resolving in
            **  a strict mode, then field is not found and we return None
            *)
            strictResolve,
                None,
            (*
            **  Otherwise, search parent nodes left to right, depth-first, until the
            **  first field with this top-level name is found. If no match has been
            **  found, return None.
            *)
            True,
                fieldResolveAgainstParentNodes[field, trieAssoc, strictResolve]
        ]
	]


(* The case of prefixed field *)
fieldResolve[
    prefix_ -> field_,
    trieAssoc_Association?AssociationQ,
    strictResolve: True | False,
    isJoinParentNode: True | False
] :=
    Which[
        (*
        **  If we deal with a prefixed field, and the current node's alias matches the
        **  outer prefix, strip the left-most prefix, and keep resolving the field.
        *)
        prefix === trieAssoc["Alias", "Value"],
            fieldResolveAgainstParentNodes[field, trieAssoc, strictResolve],
        (*
        **  If we are in a strict mode and the prefix didn't match, this means that
        **  the field wasn't found / resolved successfully.
        *)
        strictResolve || ( isJoinParentNode && trieAssoc["Alias", "Value"] =!= None ),
            None,
        (*
        **  Prefix did not match this node's alias, but we are not in a strict mode,
        **  so we should check it against parent nodes.
        *)
        True,
            fieldResolveAgainstParentNodes[
                prefix -> field, trieAssoc, strictResolve
            ]
    ]

(*
**  String field was not found / resolved in a node itself, so we search for it in
**  parent nodes. This can  happen either for non-strict resolve mode, or when we
**  stripped some prefix(es) off the field until we have  a string.
*)
fieldResolveAgainstParentNodes[
    field_String,
    trieAssoc_Association?AssociationQ,
    strictResolve_
] :=
    With[{parents = trieAssoc["Parents"]},
        Replace[Null -> None] @ Scan[
            With[
                {result = fieldResolve[
                    field, #, strictResolve, Length[parents] > 1
                ]},
                If[result =!= None, Return[result, Scan]]
            ]&,
            parents
        ]
    ]


(*
**  If we deal with a prefixed field, and the prefix doesn't match the prefix of
**  the current node, try resolving the field against this node's parents.
*)
fieldResolveAgainstParentNodes[
    prefix_ -> field_,
    trieAssoc_Association?AssociationQ,
    strictResolve: (True | False)
] :=
    With[{
        parents = trieAssoc["Parents"],
        return = Replace[res: Except[None] :> Return[res]]
        },
        Switch[parents,
            {},
                None,
            {_}, (* Annotation *)
                return @ fieldResolve[
                    prefix -> field, First @ parents, strictResolve, False
                ],
            _, (* Join *)
                Replace[Null -> None] @ Scan[
                    return[
                        fieldResolve[prefix -> field, #, strictResolve, True]
                    ]&,
                    parents
                ]
        ]
    ]


(*  ===========     Reconstructing field prefixes for raw fields     ============= *)


(*
**  Constructs a position tree for a given trie, for all the fields matching
**  <fieldPattern>, which are restricted to be members of the third argument, if
**  it is not <All> (default)
*)
fieldPositionTree[trie_?DBQueryPrefixTrieQ, rest__] :=
    fieldPositionTree[trie @ getDataAssoc[], rest]

fieldPositionTree[
    trie_Association?AssociationQ,
    fieldPattern_,
    fields: (_List | All) : All
] :=
	DBPositionTree[trie, fieldPattern, fields, fieldPattern]

(*
**  Given a field prefix trie and a list of raw fields (or All for all of them),
**  creates a position tree for those fields, and then traverses the trie using that
**  position tree, and picks up prefixes and top-level names for these raw fields.
*)
getPrefixedFields[trie_?DBQueryPrefixTrieQ, fields: (_List | All) : All, fieldPattern_] :=
	traverse[
        trie,
        fieldPositionTree[trie, fieldPattern, fields]
    ]


(*
**  Traversal function to traverse fiel prefix trie, driven by a position tree, passed
**  as a second argument. The result of the traversal is a list of prefixed fields,
**  which corresponds to this prefix trie and position tree.
**
**  The general strategy is to keep collecting the nodes we pass through, as we traverse
**  the trie, for nodes whose aliases have to be retained. When we reach the leafs
**  (fields), extract aliases from collected nodes. The main rule is that the node
**  should be kept in the prefix node list, if this is the first node "after the join"
**  (when moving "up" the trie), along the second, third etc. parent branches (we don't
**  keep prefix nodes for joins for the "main" branch (first parent of join node)).
**
**  We need to pass the child node as a separate parameter, because it is that child
**  node that carries the alias to be used for fields defined in its parent node,
**  rather than the fields-containing (parent) node itself, and since nodes are
**  immutable, we can't have a reference from a parent node to its children.
*)
traverse[trie_?DBQueryPrefixTrieQ, pos_] := traverse[trie @ getDataAssoc[], pos]

traverse[trie_Association?AssociationQ, pos_] :=
    Flatten @ traverse[trie, pos, {}, None]  (* None stands for initial child node *)

(* Expand position assoc into the first and the rest *)
traverse[trie_, pos_?AssociationQ, prefixNodes_List, childNode_] :=
	KeyValueMap[
		traverse[trie, ##, prefixNodes, childNode]&,
		pos
	]

(* Go to the parent node(s). Pass the current node as the child node, in the last arg *)
traverse[
    node_?AssociationQ, Key["Parents"], pos_?AssociationQ, prefixNodes_List, childNode_
] :=
	traverse[node["Parents"], pos, prefixNodes, node]


(*
** Traverse second and other queries / branches at the point of a join. In this
** case, the node is added to the prefix node list.
*)
traverse[
    parents_List, p_Integer, pos_?AssociationQ, prefixNodes_List, childNode_
] :=
	With[{node = parents[[p]]},
        {
    		traverse[
                node,
                KeyTake[pos, Key["ProperFields"]],
                (* Do not add current node to prefix nodes for proper fields *)
                prefixNodes, 
                childNode
            ],
            traverse[
                node,
                KeyDrop[pos, Key @ Key["ProperFields"]],
                If[addNodeQ[parents, node, p], {node, prefixNodes}, prefixNodes],
                childNode
            ]
        }
	]

(*
**  This function decides whether we keep a node / alias, in the node / alias
**  list, for a given prefixed field we are extracting. The current rule is
**  to keep the alias for
**    1. Left-most query if it is non-None, *and* either there is a single
**       parent (annotation) but the alias is sticky, or there is more than 1
**       parent (join)
**    2. All other branches / parents - always keep an alias
*)
addNodeQ[{node_}, node_,  1]:= And[
    node["Alias", "Value"] =!= None,
    TrueQ[node["Alias", "Sticky"]]
]
addNodeQ[{__}, node_, 1]:= node["Alias", "Value"] =!= None
addNodeQ[{__}, _, p_Integer]:=True

(*
** We finally get to the node that actually defines the fields. We always add the
** last child node to prefix node list, and then recostruct full prefixed names for
** all fields, defined at the level of this node.
*)
traverse[
    node_?AssociationQ,
    Key["ProperFields"],
    fields: {Key[_String]...},
    prefixNodes_List,
    childNode_
] :=
	With[{fullPrefixes = flatPrefixListFromNodeLinkedList[{childNode, prefixNodes}]},
		Map[
			DBPrefixedField[Fold[Rule[#2, #1]&, #, fullPrefixes]]&,
			Replace[fields, Key[name_] :> name, {1}]
		]
	]

(* A helper function to flatten prfix nodes linked list and extract aliases from it *)
flatPrefixListFromNodeLinkedList[nodes_List] :=
    Lookup[#, "Value", {}] & @ Lookup[
        Replace[ DeleteDuplicates @ Flatten @ nodes, {p___, None} :> {p}],
        "Alias",
        {}
    ]
