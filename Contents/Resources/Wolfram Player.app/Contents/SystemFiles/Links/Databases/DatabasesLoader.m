BeginPackage["Databases`"]

Begin["`Private`"]

Needs["PacletManager`"]

$DevelopmentMode = MatchQ[
    PacletInformation["DatabasesTests"], 
    KeyValuePattern["Location" -> _String]
];

(*
** This makes sure that we reload all sub-modules (which are independent packages),
** every time when we reload the project.
*)
Unprotect[$Packages];
$Packages = DeleteCases[$Packages, c_/; StringMatchQ[c, "Databases`" ~~ ___]];
Protect[$Packages];

(* Cleaning up all definitions except those we want to persist - dev. mode only *)
If[TrueQ[$DevelopmentMode],
    With[
        {
            devMode = $DevelopmentMode,
            db = Databases`$Databases,
            auth = Databases`$DatabaseAuthentications,
            clearPackageCode = Hold[ClearPackage[name_String, base_:"Databases`"] :=
                With[{mainContext = base <> name <> "`"},
                    Scan[
                        (Unprotect[#]; ClearAll[#]) &, {
                            mainContext <> "*",
                            mainContext <> "*`*",
                            mainContext <> "*`*`*"
                        }
                    ]
                ]
            ] (* Prevent self - destruction *)
        },
        ReleaseHold[clearPackageCode];
        ClearPackage["Databases", ""];
        ReleaseHold[clearPackageCode]; (* Reconstruct after self - destruction *)
        $DevelopmentMode = devMode;
        Databases`$Databases = db;
        Databases`$DatabaseAuthentications = auth;
    ];

    $Path = DeleteDuplicates @ Prepend[$Path, DirectoryName[$InputFileName]];
]


End[]

EndPackage[]


Block[{$Path = DeleteDuplicates @ Prepend[$Path, DirectoryName[$InputFileName]]},

    PacletManager`Package`loadWolframLanguageCode[
        "Databases",
        "Databases`",
        DirectoryName[$InputFileName],
        "Databases/Init.m",
        "AutoUpdate" -> True,
        "SymbolsToProtect" -> Automatic,
        "AutoloadSymbols" -> {
            "System`DatabaseReference",
            "System`RelationalDatabase",
            "System`DatabaseConnect",
            "System`DatabaseDisconnect",
            "System`IncludeRelatedTables"
        },
        "HiddenImports" -> {}
    ]

]
