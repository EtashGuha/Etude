(* ::Package:: *)

BeginPackage["Cryptography`"]
EndPackage[]

Cryptography`Private`autoloadSymbols = { 
				"System`SymmetricKey",
				"System`GenerateSymmetricKey",
				"System`PrivateKey",
				"System`PublicKey",
				"System`GenerateAsymmetricKeyPair",
				"System`Encrypt",
				"System`Decrypt",
				"System`EncryptedObject",
				"System`EncryptFile",
				"System`DecryptFile",
				"System`GenerateDerivedKey",
				"System`DerivedKey",
				"System`VerifyDerivedKey",
				"System`VerifyDigitalSignature",
				"System`GenerateDigitalSignature",
				"System`DigitalSignature",
				"Cryptography`AddEllipticCurvePoints",
				"Cryptography`MultiplyEllipticCurvePoints",
				"Cryptography`GenerateEllipticCurvePublicKey",
				"Cryptography`PublicKeyFormat",
				"Cryptography`$EllipticCurves",
				"Cryptography`$EllipticCurvesParameters",
				"Cryptography`$Cryptocurrencies"
};

Map[
    Quiet[
        Unprotect[#];
        ClearAll[#];
    ] &, Join[
        Cryptography`Private`autoloadSymbols, {
            "Cryptography`*",
            "Cryptography`PackageScope`*",
            "Cryptography`*`PackagePrivate`*"
        }
    ]
];

PacletManager`Package`loadWolframLanguageCode[
    "Cryptography", 
    "Cryptography`", 
    DirectoryName[$InputFileName], 
    "EncryptDecrypt.m",
    "AutoUpdate"       -> True,
    "ForceMX"          -> False, 
    "Lock"             -> False,
    "AutoloadSymbols"  -> Cryptography`Private`autoloadSymbols,
    "SymbolsToProtect" -> Cryptography`Private`autoloadSymbols
]
