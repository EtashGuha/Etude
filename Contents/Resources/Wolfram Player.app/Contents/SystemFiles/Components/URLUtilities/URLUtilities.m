(* 
    This file loaded by Get/Needs["URLUtilities`"]. 
    It must load the package files and also ensure that URLUtilities` context is on $ContextPath, which is not done by URLUtilitiesLoader.
*)

BeginPackage["URLUtilities`"]
EndPackage[]

(* 
    All loading of the paclet's Wolfram Language code should go through this file.
    Developer maintains this list of symbols. 
    autoloadSymbols must agree with the symbols listed in the Kernel extension in the PacletInfo.m file.
*)

URLUtilities`Private`autoloadSymbols = {
    "System`URLBuild",
    "System`URLParse",
    "System`URLEncode",
    "System`URLDecode",
    "System`URLQueryDecode",
    "System`URLQueryEncode",
    "System`URLShorten",
    "System`URLExpand",
    "System`URLExecute",
    "System`URLDownload",
    "System`URLRead",
    "System`URLSubmit",
    "System`URLDownloadSubmit",
    "URLUtilities`URLCorrect",
    "URLUtilities`URLCopyFile"
};

Map[
    Quiet[Unprotect[#];ClearAll[#]] &, Join[
        URLUtilities`Private`autoloadSymbols, {
            "URLUtilities`*",
            "URLUtilities`PackageScope`*",
            "URLUtilities`*`PackagePrivate`*"
        }
    ]
];

PacletManager`Package`loadWolframLanguageCode[
    "URLUtilities", 
    "URLUtilities`", 
    DirectoryName[$InputFileName], 
    "Main.m",
    "AutoUpdate"       -> True,
    "ForceMX"          -> False, 
    "Lock"             -> False,
    "AutoloadSymbols"  -> URLUtilities`Private`autoloadSymbols,
    "SymbolsToProtect" -> URLUtilities`Private`autoloadSymbols,
    "HiddenImports"    -> {"JSONTools`"}
]