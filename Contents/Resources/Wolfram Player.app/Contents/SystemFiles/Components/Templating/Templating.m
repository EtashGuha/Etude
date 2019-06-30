(* 
    This file loaded by Get/Needs["Templating`"]. 
    It must load the package files and also ensure that Templating` context is on $ContextPath, which is not done by TemplatingLoader.
*)

BeginPackage["Templating`"]
EndPackage[]

(* 
    All loading of the paclet's Wolfram Language code should go through this file.
    Developer maintains this list of symbols.
    autoloadSymbols must agree with the symbols listed in the Kernel extension in the PacletInfo.m file.
*)

Templating`Private`autoloadSymbols = {

    (* Templating symbols *)
    "System`$TemplatePath",
    "System`FileTemplate",
    "System`FileTemplateApply", 
    "System`StringTemplate", 
    "System`TemplateApply", 
    "System`TemplateEvaluate",
    "System`TemplateExpression",
    "System`TemplateIf", 
    "System`TemplateObject",
    "System`TemplateSequence", 
    "System`TemplateSlot", 
    "System`TemplateUnevaluated",
    "System`TemplateVerbatim",
    "System`TemplateWith", 
    "System`XMLTemplate",

    (* HTML Utilities *)
    "Templating`ExportHTML",
    "System`$HTMLExportRules",

    (* Symbolic pages *)
    "System`GalleryView",
    "Templating`Webpage",
    "Templating`HTMLTemplate",

    (* Panel language *)
    "Templating`HorizontalLayout",
    "Templating`InterfaceSwitched",
    "Templating`LayoutItem",
    "Templating`VerticalLayout",
    "Templating`DynamicLayout"
};

Map[
    (Unprotect[#];ClearAll[#]) &, Join[
        Templating`Private`autoloadSymbols, {
            "Templating`*",
            "Templating`PackageScope`*",
            "Templating`*`PackagePrivate`*"
        }
    ]
]

PacletManager`Package`loadWolframLanguageCode[
    "Templating", 
    "Templating`", 
    DirectoryName[$InputFileName], 
    "Primitives.m",
    "AutoUpdate"       -> True, 
    "ForceMX"          -> False, 
    "Lock"             -> False,
    "AutoloadSymbols"  -> Templating`Private`autoloadSymbols,
    "SymbolsToProtect" -> Automatic, 
    "HiddenImports"    -> {"GeneralUtilities`"}
]