(* ::Package:: *)

Paclet[
	Name -> "Cryptography",
	Version -> "2.0.0",
	MathematicaVersion -> "12+",
	Description -> "Cryptographic primitives and EncryptedData",
	Loading -> Automatic,
	Extensions -> {
		{"Kernel", 
			Context -> {"Cryptography`"}, 
			Symbols -> {		
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
			},
			(* Prevent modification of $ContextPath *)
			HiddenImport -> True
		},
		{"Resource", Root -> "Resources", Resources -> {"Libraries"} }
	}
]
