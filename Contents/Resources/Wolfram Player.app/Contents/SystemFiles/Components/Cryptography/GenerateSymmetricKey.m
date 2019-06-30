(* ::Package:: *)

Package["Cryptography`"]


PackageExport["SymmetricKey"]

(*SymmetricKey::usage = "Represents a symmetric key"*)

SymmetricKey[data_Association][key_] := data[key]
SymmetricKey[data_Association]["Properties"] := Keys[data];

PackageExport["GenerateSymmetricKey"]

Clear[GenerateSymmetricKey];
(*
GenerateSymmetricKey::invcipher = "`1` is not a valid cipher."
GenerateSymmetricKey::invmethodkey = GenerateAsymmetricKeyPair::invmethodkey = GenerateDigitalSignature::invmethodkey = "The method key `1` is not one of ``."
GenerateSymmetricKey::invblockmode = "`1` is not a known block mode for this cypher."
GenerateSymmetricKey::invivlen = "Cipher requires initialization vector of `1` bytes; `2` bytes given."
GenerateSymmetricKey::inviv = "Initialization vector `` should be Automatic, None, or a valid ByteArray.";
GenerateSymmetricKey::invkeysize = "`1` is not a valid key size."
GenerateSymmetricKey::nocipher = "No cipher given."
GenerateSymmetricKey::invdkey = "Derived key `1` is not valid."
GenerateSymmetricKey::invkeysz = "`1` has length `2`; `3` is expected."
GenerateSymmetricKey::nbyte = "`1` is not a list of integer values between 0 and 255."
*)
Options[GenerateSymmetricKey] = {Method->"AES256"}

PackageScope["checkInvalidMethodKeys"]

checkInvalidMethodKeys[head_, assoc_, allowed_] := 
	Replace[Complement[Keys[assoc], allowed],
		{{} -> None,
		 l_List :> (
		 	Message[MessageName[head, "invmethodkey"], First[l], allowed];
		 	Throw[$Failed];
		 )}];


PackageScope["missingQ"]

missingQ[_Missing] := True;
missingQ[_] := False;

parseMethod[cipher_String] := parseMethod[<|"Cipher" -> cipher|>];

parseMethod[assoc_Association] := Module[
	{cipher, blockmode, keysize, iv, validLen},
	
	checkInvalidMethodKeys[GenerateSymmetricKey, assoc, {"Cipher", "BlockMode", "KeySize", "InitializationVector"}];
	
	{cipher, blockmode, keysize, iv} = Lookup[assoc, {"Cipher", "BlockMode", "KeySize", "InitializationVector"}];
		
	If[missingQ[cipher], 
		Message[GenerateSymmetricKey::nocipher]; 
		Throw[$Failed]];
		
	If[!MemberQ[$SymmetricCiphers, cipher], 
		Message[GenerateSymmetricKey::invcipher, cipher]; 
		Throw[$Failed]];
	
	If[missingQ[blockmode],
		blockmode = If[cipher === "RC4", None, "CBC"];
	,
		If[!ValidBlockModeQ[blockmode, cipher],
			Message[GenerateSymmetricKey::invblockmode, blockmode];
			Throw[$Failed]
		]
	];
	
	If[missingQ[keysize]  || keysize === Automatic,
		keysize = $SymmetricKeySizes[cipher];
	,
		If[!(MatchQ[keysize, _Integer] && keysize > 0) || !ValidKeySizeQ[keysize, cipher],
			Message[GenerateSymmetricKey::invkeysize, keysize];
			Throw[$Failed]
		];
	];
	
	validLen = $SymmetricIVSizes[cipher] / 8;
	iv = parseIV[iv, validLen];
	
	{cipher, blockmode, keysize, iv}
];

parseMethod[method_] := (
	Message[GenerateSymmetricKey::invmethod, method];
	Throw[$Failed]
); 

parseIV[None|_Missing, _] := None;
parseIV[Automatic, n_] := randomBytes[n];
parseIV[iv_ByteArray ? ByteArrayQ, n_] := If[Length[iv] === n, iv,
	Message[GenerateSymmetricKey::invivlen, n, Length[iv]];
	Throw[$Failed]
];
parseIV[iv_, _] := (
	Message[GenerateSymmetricKey::inviv, iv];
	Throw[$Failed]
);

toKey[head_, Automatic, keysize_] := randomBytes[keysize];

toKey[head_, password_String, keysize_] := ByteArray[scrypt[
	ToCharacterCode[password, "UTF8"],
	ToCharacterCode["wolframKey", "UTF8"], keysize]]

toKey[head_, list:{___Integer}, keysize_] := Module[
	{size = Length[list]},
	If[!byteListQ[list],
		Message[GenerateSymmetricKey::nbyte, list];
		Throw[$Failed]
	];
	If[size != keysize,
		Message[GenerateSymmetricKey::invkeysz, list, size, keysize];
		Throw[$Failed]					
	];
	ByteArray[list]
]

toKey[head_, ba_ByteArray, keysize_] := Module[
	{size = Length[ba]},
	If[size != keysize,
		Message[GenerateSymmetricKey::invkeysz, ba, size, keysize];
		Throw[$Failed]					
	];
	ba
]

toKey[head_, k:DerivedKey[dk_], keysize_] /; AssociationQ[dk] := Module[
	{key = Lookup[dk, "DerivedKey", $Failed], size},
	If[key === $Failed || !ByteArrayQ[key],
		Message[GenerateSymmetricKey::invdkey, k];
		Throw[$Failed]
	];
	size = Length[key];
	If[size != keysize,
		Message[GenerateSymmetricKey::invkeysz, k, size, keysize];
		Throw[$Failed]		
	];
	key
]

toKey[head_, __] := Throw[$Failed];

GenerateSymmetricKey[opts:OptionsPattern[]] := GenerateSymmetricKey[Automatic, opts];

GenerateSymmetricKey[keyspec_, OptionsPattern[]] := Module[
	{key, cipher, blockmode, keysize, iv},
	Catch[
		{cipher, blockmode, keysize, iv} = parseMethod[OptionValue[Method]];
		key = toKey[GenerateSymmetricKey, keyspec, keysize/8];
		SymmetricKey[<|
			"Cipher" -> cipher,
			"BlockMode" -> blockmode,
			"Key" -> key,
			"KeySize" -> keysize,
			"InitializationVector" -> iv
		|>]
	]
];
