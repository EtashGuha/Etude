(* All loading of the paclet's Wolfram Language code should go through this file. *)

(* Developer maintains this list of symbols.
   autoloadSymbols must agree with the symbols listed in the Kernel extension in the PacletInfo.m file.
*)

TextSearch`Private`autoloadSymbols = {
    "System`Autocomplete",
    "System`AutocompletionFunction",
    "System`ContentObject",
    "System`CreateSearchIndex",
    "System`DeleteSearchIndex",
    "System`DocumentWeightingRules",
    "System`SearchIndexObject",
    "System`SearchIndices",
    "System`SearchResultObject",
    "System`Snippet",
    "System`TextSearch",
    "System`TextSearchReport",
    "System`UpdateSearchIndex",
    "System`ContentFieldOptions",
    "System`ContentLocationFunction",
    "System`SearchAdjustment",
    "System`MaxWordGap",
    "System`ContentFieldOptions",
    "System`SearchQueryString"
};

PacletManager`Package`loadWolframLanguageCode["TextSearch", "TextSearch`", DirectoryName[$InputFileName], "Utils.m",
          "AutoUpdate" -> False, "ForceMX" -> False, "Lock" -> False, 
          "AutoloadSymbols" -> TextSearch`Private`autoloadSymbols,
          "HiddenImports" -> {"Macros`", "JLink`"}
];
