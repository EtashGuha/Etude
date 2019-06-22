(* Wolfram Language package *)

Package["Databases`Database`"]

PackageImport["Databases`"]
PackageImport["Databases`Common`"] (* DBDefError *)
PackageImport["Databases`Python`"]
PackageImport["Databases`Schema`"] (* $DBReferencePattern, $DBTablePattern, $DBSchemaPattern *)
PackageImport["Databases`SQL`"] (* DBQueryObjectQ *)


PackageExport[DBCreate]
PackageExport[DBDrop]
PackageExport[DBDump]
PackageExport[DBInsert]

Options[DBCreate]  := {Authentication -> Inherited, Path -> None}
Options[DBDrop]    := {Authentication -> Inherited, Path -> None}
Options[DBDump]    := {Authentication -> Inherited, Path -> None}
Options[DBInsert]  := {Authentication -> Inherited, Path -> None}

DBCreate[tables: $DBTablePattern: All, schema: $DBSchemaPattern: Inherited, OptionsPattern[]] :=
    With[
        {using = DBSchemaGet[schema]},
        DBRunPython[
            "DatabaseCreate"[tables],
            Authentication -> DBAuthenticationGet[OptionValue[Authentication], using],
            "Schema"       -> using,
            Path           -> OptionValue[Path]
        ]
    ]
DBDefError @ DBCreate

DBDrop[tables: $DBTablePattern: All, schema: $DBSchemaPattern: Inherited, OptionsPattern[]] :=
    With[
        {using = DBSchemaGet[schema]},
        DBRunPython[
            "DatabaseDrop"[tables],
            Authentication -> DBAuthenticationGet[OptionValue[Authentication], using],
            "Schema"       -> using,
            Path           -> OptionValue[Path]
        ]
    ]
DBDefError @ DBDrop

DBDump[tables: $DBTablePattern: All, schema: $DBSchemaPattern: Inherited, OptionsPattern[]] :=
    With[
        {using = DBSchemaGet[schema]},
        DBRunPython[
            "DatabaseDump"[tables],
            Authentication -> DBAuthenticationGet[OptionValue[Authentication], using],
            "Schema"       -> using,
            Path           -> OptionValue[Path]
        ]
    ]
DBDefError @ DBDump

DBInsert[
    data: KeyValuePattern[_ -> {RepeatedNull[_?AssociationQ]}],
    schema: $DBSchemaPattern: Inherited,
    OptionsPattern[]
] :=
    With[
        {using = DBSchemaGet[schema]},
        DBRunPython[
            "DatabaseInsert"[serializeData[data, schema["Connection"]["Backend"]]],
            Authentication -> DBAuthenticationGet[OptionValue[Authentication], using],
            "Schema"       -> using,
            Path           -> OptionValue[Path]
        ]
    ]

DBInsert[f_File, rest___] :=
    DBInsert[Import[f], rest]

DBDefError @ DBInsert
