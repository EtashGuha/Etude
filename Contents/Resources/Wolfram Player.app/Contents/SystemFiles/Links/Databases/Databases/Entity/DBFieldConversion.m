Package["Databases`Entity`"]


PackageImport["EntityFramework`"]
PackageImport["Databases`"]
PackageImport["Databases`SQL`"]
PackageImport["Databases`Common`"]
PackageImport["Databases`Database`"] (* DBGenerateUniquePropertyName *)


PackageScope["toPrefixedField"]
PackageScope["compileFields"]
PackageScope["compileAnnotations"]
PackageScope["compileSortingFields"]
PackageScope["invertSortingFields"]
PackageScope["compileEntityJoinSpec"]
PackageScope["toEntityProperty"]
PackageScope["processProperties"]


$string = _String?StringQ


(*
**  Converts string or _EntityProperty field to prefixed field
*)
toPrefixedField[prop: $string] := DBPrefixedField[prop]

toPrefixedField[EntityProperty[Inherited | Automatic, field: $string]] :=
	DBPrefixedField[field]

toPrefixedField[EntityProperty[type_, field: $string]] :=
	toPrefixedField[type, field, DBToLinkedList[{}]]

toPrefixedField[props: {($string | _EntityProperty)...}] :=
	Map[toPrefixedField, props]

toPrefixedField[arg_] := DBRaise[toPrefixedField, "invalid_entity_property", {arg}]

toPrefixedField[prefix_ -> t_, field_, prev : DBLinkedListPattern[]] :=
	toPrefixedField[t, field, DBLinkedList[prefix, prev]]

toPrefixedField[t_, field_, prev: DBLinkedListPattern[]] :=
	DBPrefixedField @ Fold[
		Rule[#2, #1]&,
		Join[{field, t}, DBFromLinkedList @ prev]
	]

DBDefError @ toPrefixedField


(*
**  "Compiles" entity-level fields, including _EntityFunction - valued ones, to
**   Database* - level fields.
*)
compileFields[es_, fields_List] := Map[compileFields[es, #]&, fields]

compileFields[es_, field: $string | _EntityProperty] :=
	toPrefixedField[field]

compileFields[es_, field_EntityFunction] :=
	DBCompileEntityFunction[es, field]

compileFields[es_, name: $string -> fn_EntityFunction] :=
	name -> compileFields[es, fn]

DBDefError @ compileFields


compileSortingFields[es_, fields_List] := Map[
	compileSortingFields[es, #]&,
	fields
]

compileSortingFields[es_, field: $string | _EntityProperty] :=
    compileFields[es, field]

compileSortingFields[es_, fn_EntityFunction] :=
	compileFields[es, fn]

compileSortingFields[es_, any_ -> "Ascending"] :=
    DatabaseSQLAscending[compileSortingFields[es, any]]

compileSortingFields[es_, any_ -> "Descending"] :=
	DatabaseSQLDescending[compileSortingFields[es, any]]

invertSortingFields[list_List] := invertSortingFields /@ list

invertSortingFields[field: $string | _EntityProperty | _EntityFunction] :=
    field -> "Descending"

invertSortingFields[any_ -> "Ascending"] := any -> "Descending"

invertSortingFields[any_ -> "Descending"] := any -> "Ascending"

(*
**  "Compiles" entity-level annotations to Database* - level annotations
*)
compileAnnotations[
    es_, 
    annots: {($string -> ($string | _EntityFunction | _Symbol | _EntityProperty ))...}
] :=
	Map[compileAnnotations[es, #]&, annots]

compileAnnotations[es_, name: $string -> s: ( $string | _EntityProperty )] :=
	compileAnnotations[
		es,
		name -> Replace[DBUniqueTemporary["ent"], ent_ :> EntityFunction[ent, ent[s]]]
	]

compileAnnotations[es_, annot: ($string -> _EntityFunction)] :=
	compileFields[es, annot]

compileAnnotations[es_, s: $string -> h_Symbol] :=
	compileAnnotations[
		es,
		s -> Replace[DBUniqueTemporary["ent"], ent_ :> EntityFunction[ent, h[ent[s]]]]
	]

DBDefError @ compileAnnotations

compileEntityJoinSpec[es_, ef_EntityFunction] := DBCompileEntityFunction[es, ef]

compileEntityJoinSpec[es_, any_] := ReplaceAll[
	any,
	ep_EntityProperty :> toPrefixedField[ep]
]

DBDefError @ compileEntityJoinSpec



(*
**  Converts Database* - level prefixed fields to entity-level fields.
*)
toEntityProperty[any_] := EntityProperty @@ DBPrefixedFieldParts[any]


(* 
**  This function processes synthetic fields for entityValue - like usage
*)
processProperties[q_, prop:Except[_List]] :=
    MapAt[First, processProperties[Unevaluated @ q, {prop}], 2]
(*
**  Generates syntehtic field names for fields of the form _EntityFunction.
*)
processProperties[q_, props_List] :=
    With[{newProps = Replace[
            props,
            p: EntityFunction[_Symbol | {_Symbol}, __] :> 
                (DBGenerateUniquePropertyName[] -> p),
            {1}
        ]},
        processProperties[Unevaluated @ q, newProps] /; newProps =!= props
    ]
    
(*
**  Looks for the fields of the form name -> _EntityFunction, and performs query
**  annotation with these fields. The resulting call does not contain such fields,
**  only _String | _EntityProperty
*)
processProperties[q_, props_List] := 
    With[{
        annots = Cases[
            props,
            Verbatim[Rule][_, EntityFunction[_Symbol | {_Symbol}, __]],
            {1}
        ],
        pureProps = Replace[props, (f_ -> _EntityFunction) :> f, {1}]
        },
        If[annots === {},
            {Hold[q], props},
            (* else *)
            {Hold[ExtendedEntityClass[q, annots]], pureProps}
        ]
    ]
