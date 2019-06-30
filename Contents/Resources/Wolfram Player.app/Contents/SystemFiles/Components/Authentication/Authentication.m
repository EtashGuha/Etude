(* 
    This file loaded by Get/Needs["Authentication`"]. 
    It must load the package files and also ensure that Authentication` context is on $ContextPath, which is not done by AuthenticationLoader.
*)

BeginPackage["Authentication`"]
EndPackage[]
(* 
    All loading of the paclet's Wolfram Language code should go through this file.
    Developer maintains this list of symbols.
    autoloadSymbols must agree with the symbols listed in the Kernel extension in the PacletInfo.m file.
*)

Authentication`Private`autoloadSymbols = {
    "System`Authenticate",
    "System`SecuredAuthenticationKey",
    "System`SecuredAuthenticationKeys",
    "System`GenerateSecuredAuthenticationKey",
    "System`SetSecuredAuthenticationKey",
    "System`$SecuredAuthenticationKeyTokens",
    "Authentication`AuthenticateHTTPRequest"
};

Map[
    Quiet[
        Unprotect[#];
        ClearAll[#];
    ] &, Join[
        Authentication`Private`autoloadSymbols, {
            "Authentication`*",
            "Authentication`PackageScope`*",
            "Authentication`*`PackagePrivate`*"
        }
    ]
];

PacletManager`Package`loadWolframLanguageCode[
    "Authentication", 
    "Authentication`", 
    DirectoryName[$InputFileName], 
    "SecuredAuthenticationKey.m",
    "AutoUpdate"       -> True,
    "ForceMX"          -> False, 
    "Lock"             -> False,
    "AutoloadSymbols"  -> Authentication`Private`autoloadSymbols,
    "SymbolsToProtect" -> Authentication`Private`autoloadSymbols
]