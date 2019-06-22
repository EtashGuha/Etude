(* All loading of the paclet's Wolfram Language code should go through this file. *)

Developer`$CurrentPackage = "TypeSystem`";
PacletManager`Package`loadWolframLanguageCode["TypeSystem", "TypeSystem`", DirectoryName[$InputFileName], "Types.m",
        "AutoUpdate" -> True, "ForceMX" -> False, "Lock" -> False, 
        "HiddenImports" -> {"Macros`", "GeneralUtilities`"},
        "AutoloadSymbols" -> {}
]
Developer`$CurrentPackage = None;

General::invty = "`` is in an invalid type spec.";