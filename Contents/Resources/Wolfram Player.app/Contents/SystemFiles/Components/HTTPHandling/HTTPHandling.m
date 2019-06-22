(* 
    This file loaded by Get or by Needs["HTTPHandling`"]. 
    It must load the package files by calling HTTPHandling.m,
    and also ensure that HTTPHandling` context is on $ContextPath.
*)

BeginPackage["HTTPHandling`"]
EndPackage[]

(* 
    All loading of the paclet's Wolfram Language code should go through this file.
    Developer maintains this list of symbols.
    AutoloadSymbols must agree with the symbols listed in the Kernel extension in the PacletInfo.m file.
*)

Map[
    Function[
        Unprotect[#];
        ClearAll[#];
    ], {
        "HTTPHandling`WebServer",
        "HTTPHandling`StartWebServer",
        "HTTPHandling`*`*",
        "HTTPHandling`*`*`*"
    }
]

PacletManager`Package`loadWolframLanguageCode[
    "HTTPHandling", 
    "HTTPHandling`", 
    DirectoryName[$InputFileName], 
    "Main.m",
    "AutoUpdate" -> True,
    "ForceMX" -> False, 
    "Lock" -> False,
    "AutoloadSymbols" -> {
        "HTTPHandling`WebServer",
        "HTTPHandling`StartWebServer"
    },
    "HiddenImports" -> {"MQTTLink`"}
]