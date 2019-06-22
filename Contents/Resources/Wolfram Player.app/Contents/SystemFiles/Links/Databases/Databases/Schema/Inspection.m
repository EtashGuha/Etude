(* Wolfram Language package *)

Package["Databases`Schema`"]

PackageImport["Databases`"]
PackageImport["Databases`Python`"]
PackageImport["Databases`Common`"]

PackageExport["DBInspect"]

DBInspect[
    conn: $DBReferencePattern, 
    tables: $DBTablePattern: All, 
    OptionsPattern[{RelationalDatabase, DBInspect}]
] := 
    DBRunPython[
        "DatabaseInspect"[
            Replace[
                tables, 
                All|Automatic -> None
            ],
            Replace[
                OptionValue[IncludeRelatedTables], {
                    All|Automatic -> None,
                    any_ :> TrueQ[any]
                }
            ]
        ],
        Authentication -> DBAuthenticationGet[conn],
        "Schema"       -> None
    ]

