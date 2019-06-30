(* 
    This file loaded by Get/Needs["Forms`"]. 
    It must load the package files and also ensure that Forms` context is on $ContextPath, which is not done by FormsLoader.
*)

BeginPackage["Forms`"]
EndPackage[]

(* 
    All loading of the paclet's Wolfram Language code should go through this file. 
    Developer maintains this list of symbols.
    autoloadSymbols must agree with the symbols listed in the Kernel extension in the PacletInfo.m file.
*)

Forms`Private`autoloadSymbols = {
    "System`FormObject", 
    "System`FormTheme", 
    "System`PageTheme" ,
    "System`FormFunction",
    "System`FormLayoutFunction", 
    "System`AppearanceRules", 
    "System`APIFunction",
    "System`AutoSubmitting",
    "System`AskFunction",
    "System`Ask",
    "System`AskAppend",
    "System`AskConfirm",
    "System`AskedValue",
    "System`AskedQ",
    "System`AskDisplay",
    "System`AskTemplateDisplay",
    "System`AskState"
}

Map[
    (Unprotect[#];ClearAll[#]) &, Join[
        Forms`Private`autoloadSymbols, {
            "Forms`*",
            "Forms`PackageScope`*",
            "Forms`*`PackagePrivate`*"
        }
    ]
]

toProtect[]

PacletManager`Package`loadWolframLanguageCode[
    "Forms", 
    "Forms`", 
    DirectoryName[$InputFileName], 
    "Primitives.m",
    "AutoUpdate"       -> True,
    "ForceMX"          -> False, 
    "Lock"             -> False,
    "AutoloadSymbols"  -> Forms`Private`autoloadSymbols,
    "SymbolsToProtect" -> Automatic, 
    "HiddenImports"    -> {"Interpreter`"}
]