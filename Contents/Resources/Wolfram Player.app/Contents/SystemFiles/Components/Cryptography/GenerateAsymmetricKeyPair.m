(* ::Package:: *)

Package["Cryptography`"]

(*General::invba = "`1` is not a valid ByteArray."*)

byteArrayToString[head_, ba_?ByteArrayQ] := StringJoin[IntegerString[Developer`FromByteArray[ba], 16, 2]];
byteArrayToString[head_, arg_] := (Message[MessageName[head, "invba"], arg]; Missing["KeyAbsent"]);

byteArrayToBitLength[head_, ba_?ByteArrayQ] := 8*Length[ba];
byteArrayToBitLength[head_, ba_?ByteArrayQ, "Header"] := 8*(Length[ba]-1);
byteArrayToBitLength[head_, arg_] := (Message[MessageName[head, "invba"], arg]; Missing["KeyAbsent"]);

PackageExport["PublicKey"]

(*PublicKey::usage = "Represents a public key"*)

PublicKey[data_Association][key_] := data[key];
PublicKey[data_Association]["PublicKeyHexString"] := byteArrayToString[PublicKey, data["PublicByteArray"]];
PublicKey[data_Association]["PublicHexString"] := byteArrayToString[PublicKey, data["PublicByteArray"]];

PublicKey[data_Association]["PublicKeySize"] := byteArrayToBitLength[PublicKey, data["PublicByteArray"]] /; data["Cipher"] == "RSA";
PublicKey[data_Association]["PublicKeySize"] := byteArrayToBitLength[PublicKey, data["PublicByteArray"], "Header"] /; data["Type"] == "EllipticCurve";

PublicKey[data_Association]["Properties"] := Join[Keys[data], {"PublicHexString", "PublicKeySize"}];

PackageExport["PrivateKey"]

(*PrivateKey::usage = "Represents private key"*)

PrivateKey[data_Association][key_] := data[key];
PrivateKey[data_Association]["PrivateHexString"] := byteArrayToString[PrivateKey, data["PrivateByteArray"]];
PrivateKey[data_Association]["PublicHexString"] := byteArrayToString[PrivateKey, data["PublicByteArray"]];
PrivateKey[data_Association]["PrivateKeyHexString"] := byteArrayToString[PrivateKey, data["PrivateByteArray"]];
PrivateKey[data_Association]["PublicKeyHexString"] := byteArrayToString[PrivateKey, data["PublicByteArray"]];

PrivateKey[data_Association]["PrivateKeySize"] := byteArrayToBitLength[PrivateKey, data["PrivateByteArray"]];

PrivateKey[data_Association]["PublicKeySize"] := byteArrayToBitLength[PublicKey, data["PublicByteArray"]]/; data["Cipher"] == "RSA";
PrivateKey[data_Association]["PublicKeySize"] := byteArrayToBitLength[PublicKey, data["PublicByteArray"], "Header"]/; data["Type"] == "EllipticCurve";

PrivateKey[data_Association]["Properties"] := Join[Keys[data], {"PublicHexString", "PrivateHexString", "PublicKeySize", "PrivateKeySize"}];


PackageExport["GenerateAsymmetricKeyPair"]

(*GenerateAsymmetricKeyPair::usage = "Generates a PrivateKey and PublicKey."*)
(*
GenerateAsymmetricKeyPair::failure = "Couldn't generate a key pair.";
GenerateAsymmetricKeyPair::invcipher = "`1` is not a valid \"Cipher\"." 
GenerateAsymmetricKeyPair::invcompress = "The value of \"Compressed\" `1` should be either True or False";
GenerateAsymmetricKeyPair::invcurvename = "`1` is not a valid \"CurveName\"."
GenerateAsymmetricKeyPair::invkeysize = "`1` is not a valid \"KeySize\".";
GenerateAsymmetricKeyPair::invmethod = "The value of the option Method -> `` should be a string or an association.";
GenerateAsymmetricKeyPair::invpexp = "The \"PublicExponent\" `1` should be an odd integer greater than 3.";
GenerateAsymmetricKeyPair::invpadding = "The \"Padding\" `1` should be one of ``.";
GenerateAsymmetricKeyPair::invspecs = "The value of the option Method -> `` is not a valid combination of specifications.";
GenerateAsymmetricKeyPair::invtype = "`1` is not a valid \"Type\"."
GenerateAsymmetricKeyPair::nocipher = "No cipher or elliptic curve specified."
GenerateAsymmetricKeyPair::notype = "Warning: \"Type\" parameter is not provided. Assuming \"`1`\""
*)

(*GenerateRSAKey[bits_Integer, publicExponent_Integer] :=
 
 Module[{p, q, modulus, totient, privateExponent},
  
  {p, q} = RandomPrime[{2^((bits/2) - 1), 2^(bits/2) - 1}, 2];
  modulus = p*q;
  totient = (p - 1)*(q - 1);
  
  While[! CoprimeQ[totient, publicExponent],
   {p, q} = RandomPrime[{2^((bits/2) - 1), 2^(bits/2) - 1}, 2];
   modulus = p*q;
   totient = (p - 1)*(q - 1);
   ];
  privateExponent = PowerMod[publicExponent, -1, totient];
  <|"PublicExponent" -> publicExponent, "PublicModulus" -> modulus, "PrivateExponent" -> privateExponent|>]*)

PackageExport["GeneratePrivateKey"]

GeneratePrivateKey[bits_Integer, publicExponent_Integer, padding_] := Module[
	{pubmod, privmod},
	{pubmod, privmod} = ToExpression[FromCharacterCode[llgenerateRSAKey[bits, publicExponent]]];
	If[!IntegerQ[pubmod] || !IntegerQ[privmod], 
		Message[GenerateAsymmetricKeyPair::failure];
		Throw[$Failed]
	];
	<|
		"Type" -> "RSA",
		"Cipher" -> "RSA",
		"Padding" -> padding,
		"PublicExponent" -> publicExponent, 
		"PublicModulus" -> pubmod, 
		"PrivateExponent" -> privmod,
		"PublicByteArray" -> ByteArray[IntegerDigits[pubmod, 256, bits/8]],
		"PrivateByteArray" -> ByteArray[IntegerDigits[privmod, 256, bits/8]]
	|>
];


$AsymmetricKeyPairMethods = {"RSA", "EllipticCurve", "Bitcoin", "BTC", "Ethereum", "ETH"};
(*do not change order of $AsymmetricKeyPairKeys*)
$AsymmetricKeyPairKeys = {"Type", "Cipher", "KeySize", "PublicExponent", "Padding", "CurveName", "Compressed"};
$ECKeys = {"Type", "CurveName", "Compressed"};
$RSAKeys = {"Type", "Cipher", "Padding", "KeySize", "PublicExponent"};
  
Options[GenerateAsymmetricKeyPair] = {Method -> "RSA"};

GenerateAsymmetricKeyPair[name_String] /; MemberQ[$AsymmetricKeyPairMethods, name] :=
	GenerateAsymmetricKeyPair[Method -> name];

GenerateAsymmetricKeyPair[Method -> <||>] := GenerateAsymmetricKeyPair[Method -> "RSA"];

GenerateAsymmetricKeyPair["EllipticCurve"] := 
	GenerateAsymmetricKeyPair[Method -> <|"Type" -> "EllipticCurve", "CurveName" -> "secp256k1", "Compressed" -> False|>];
	
GenerateAsymmetricKeyPair[OptionsPattern[]] := Catch @ Module[
	{method, type, cipher, keysize, pexponent, padding, ec, compress, info},
	
	method = OptionValue[Method];
	
	If[method == Automatic, method = <|"Cipher" -> "RSA"|>];
	
	If[StringQ[method],
		Which[
			MemberQ[Join[$AsymmetricCiphers, {"EllipticCurve"}], method], 
					method = <|"Type" -> method|>,
			MemberQ[Join[$EllipticCurves, $Cryptocurrencies], method], 
					method = <|"Type" -> "EllipticCurve", "CurveName" -> method|>,
			True, 
					Message[GenerateAsymmetricKeyPair::invcipher, method]; 
					Throw[$Failed]
			]
	,
		If[!AssociationQ[method], 
			Message[GenerateAsymmetricKeyPair::invmethod, method]; 
			Throw[$Failed]
		];
	];

	checkInvalidMethodKeys[GenerateAsymmetricKeyPair, method, $AsymmetricKeyPairKeys];
		
	{type, cipher, keysize, pexponent, padding, ec, compress} = Lookup[method, $AsymmetricKeyPairKeys];
	
	(*first check if Type is present and correct*)
	If[!MissingQ[type],
	
		If[!MemberQ[Join[$AsymmetricCiphers, {"EllipticCurve"}], type],
			(* if Type is wrong fail *)
			Message[GenerateAsymmetricKeyPair::invtype, type]; 
					Throw[$Failed]],
					
		(*if Type is missing try to guess*)			
		Which[
			SubsetQ[$RSAKeys, Keys[method]],
				type = "RSA";
				Message[GenerateAsymmetricKeyPair::notype, type],
								
			SubsetQ[$ECKeys, Keys[method]],
				type = "EllipticCurve";
				Message[GenerateAsymmetricKeyPair::notype, type],
					
			True,
				Message[GenerateAsymmetricKeyPair::invspecs, method]; 
				Throw[$Failed]			
		];
	];		
		
	Switch[type, 
	
		"RSA",
		
			If[!SubsetQ[$RSAKeys, Keys[method]],
				Message[GenerateAsymmetricKeyPair::invspecs, method]; 
				Throw[$Failed]];
				
			If[!MissingQ[cipher]&&!MemberQ[$AsymmetricCiphers, cipher],
				Message[GenerateAsymmetricKeyPair::invcipher, cipher]; 
				Throw[$Failed]
			];
		
			If[missingQ[keysize],
				keysize = 2048;
			,
				If[!IntegerQ[keysize] || keysize < 17 || keysize > 65536,
					Message[GenerateAsymmetricKeyPair::invkeysize, keysize]; 
					Throw[$Failed];
				];
			];
			
			If[missingQ[pexponent],
				pexponent = 65537;
			,
				If[!IntegerQ[pexponent] || EvenQ[pexponent] || pexponent < 3,
					Message[GenerateAsymmetricKeyPair::invpexp, pexponent]; 
					Throw[$Failed];
				];
			];
			
			If[missingQ[padding],
				padding = "PKCS1";
			,
				If[!MemberQ[$RSAPaddingModes, padding],
					Message[GenerateAsymmetricKeyPair::invpadding, padding, DeleteCases[$RSAPaddingModes, "None"]];
					Throw[$Failed]
				];
			];
			
			info = GeneratePrivateKey[keysize, pexponent, padding];

			<|
				"PrivateKey" -> PrivateKey[info], 
				"PublicKey" -> PublicKey[KeyTake[info, {"Type", "Cipher", "Padding", "PublicExponent", "PublicModulus", "PublicByteArray"}]]
			|>,
			
			
		"EllipticCurve",
		
			If[!SubsetQ[$ECKeys, Keys[method]],
				Message[GenerateAsymmetricKeyPair::invspecs, method]; 
				Throw[$Failed]];
				
			If[MissingQ[ec],
				ec = "secp256k1",
				
				If[!MemberQ[Join[$EllipticCurves, $Cryptocurrencies], ec],
					Message[GenerateAsymmetricKeyPair::invcurvename, ec]; 
					Throw[$Failed]
				];	
			];			
		
			If[MissingQ[compress],
				compress = False,
				
				If[!BooleanQ[compress],
					Message[GenerateAsymmetricKeyPair::invcompress, compress];
					Throw[$Failed]
				];
			];
			
			info = GenerateECKeyPair[ec, "Compressed" -> compress];
			
			<|
				"PrivateKey" -> PrivateKey[info], 
				"PublicKey" -> PublicKey[KeyTake[info, {"Type", "CurveName", "Compressed", "PublicByteArray", "PublicCurvePoint"}]]
			|>
	]
];




