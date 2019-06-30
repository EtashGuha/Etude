
BeginPackage["Blockchain`"]
EndPackage[]

Blockchain`Private`autoloadSymbols = {
    
    "System`BlockchainData",
    "System`BlockchainBlockData",
    "System`BlockchainTransactionData",
    "System`BlockchainTransaction",
    "System`BlockchainTransactionSign",
    "System`BlockchainTransactionSubmit",
    "System`BlockchainAddressData",
    "System`BlockchainTokenData",
    "System`BlockchainGet",
    "System`BlockchainPut",
    "System`BlockchainKeyEncode",
    "System`BlockchainContractValue",
    "System`BlockchainBase",
    "Blockchain`ContractEncode",
    "Blockchain`ContractDecode",
    "Blockchain`BlockchainContract",    
    "Blockchain`ValidBlockchainBaseQ",
    "Blockchain`$TemplateBase"
};

Map[
    Quiet[
        Unprotect[#];
        ClearAll[#];
    ] &, Join[
        Blockchain`Private`autoloadSymbols, {
            "Blockchain`*",
            "Blockchain`PackageScope`*",
            "Blockchain`*`PackagePrivate`*"
        }
    ]
];

PacletManager`Package`loadWolframLanguageCode[
    "Blockchain", 
    "Blockchain`", 
    DirectoryName[$InputFileName], 
    "Main.m",
    "AutoUpdate"       -> True,
    "ForceMX"          -> False, 
    "Lock"             -> False,
    "AutoloadSymbols"  -> Blockchain`Private`autoloadSymbols,
    "SymbolsToProtect" -> Blockchain`Private`autoloadSymbols
]
