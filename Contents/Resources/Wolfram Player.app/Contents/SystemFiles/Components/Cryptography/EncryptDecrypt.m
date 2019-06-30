(* ::Package:: *)

Package["Cryptography`"]

PackageExport["Macros`"]

PackageExport["$LastEncryptionError"]

checkList[l_List] := l;
checkList[err_LibraryFunctionError] := ($LastEncryptionError = err; Throw[$Failed]); 
checkList[err_] := ($LastEncryptionError = err; Throw[$Failed]);

Clear[encryptInternal];

encryptInternal[SymmetricKey[info_Association], data_List, dir_Integer] := 
	ByteArray @ checkList @ llencryptSym[
		$SymmetricCipherNumbering[info["Cipher"], info["BlockMode"]],
		data,
		Normal[info["Key"]],
		Replace[Normal[info["InitializationVector"]], None -> {}],
		dir
	];

encryptInternal[PublicKey[info_Association], data_List, dir_Integer] := (
	If[dir == 0 && !MatchQ[info["Padding"], "PKCS1" | "None" | None],
		Message[Decrypt::padding]; Throw[$Failed]];
	ByteArray @ checkList @ llrsaPublic[
		data, 
		ToCharacterCode[ToString[info["PublicModulus"]]],
		ToCharacterCode[ToString[info["PublicExponent"]]],
		$RSAPaddingModeNumbering[info["Padding"]],
		dir
	]
);

encryptInternal[PrivateKey[info_Association], data_, dir_Integer] := (
	If[dir == 1 && !MatchQ[info["Padding"], "PKCS1" | "None" | None],
		 Message[Encrypt::padding]; Throw[$Failed]];
	ByteArray @ checkList @ llrsaPrivate[
		data,
		ToCharacterCode[ToString[info["PublicModulus"]]],
		ToCharacterCode[ToString[info["PublicExponent"]]],
		ToCharacterCode[ToString[info["PrivateExponent"]]],
		$RSAPaddingModeNumbering[info["Padding"]],
		dir
	]
);

encryptInternal[args___] := (
	Message[Encrypt::failed]; 
	Throw[$Failed];
);
(*
Encrypt::invkeyspec = Decrypt::invkeyspec = "Key should be a string or valid SymmetricKey, PrivateKey, or PublicKey."
Encrypt::ecckeyspec = Decrypt::ecckeyspec = "Encryption is not available with elliptic curves."
Encrypt::failed = "Data could not be encrypted.";
Decrypt::failed = "Data could not be decrypted."
Decrypt::key = "Key is not compatible with the encrypted data." 
Encrypt::keylen = Decrypt::keylen = "Key is not of suitable length."
Encrypt::ivlen = Decrypt::ivlen = "Initialization vector is not of suitable length."
Encrypt::padding = Decrypt::padding = "Specified padding could not be used."
Encrypt::len = Decrypt::len = "Input too long for this encryption method."
Encrypt::invdata = "Input is not a valid ByteArray.";
Decrypt::invdata = "`` is not a ByteArray or valid EncryptedObject.";
*)
PackageExport["Encrypt"]

encryptForm[data_ByteArray] := {If[!System`ByteArrayQ[data], Message[Encrypt::invdata]; Throw[$Failed]]; Normal[data], ByteArray};
encryptForm[data_String] := {ToCharacterCode[data, "UTF8"], String};
encryptForm[expr_] := {Normal @ ByteArray[StringDrop[Compress[expr], 2]], Expression};

Clear[Encrypt];
Encrypt[key_, data_] := Module[
	{key2, data2, form, iv, encrypted},
	Catch[
		key2 = toKey[Encrypt, key, randomBytes];
		If[key2 === $Failed, 
			Message[MessageName[Encrypt, "invkeyspec"]];
			Throw[$Failed];
		];
		iv = key2["InitializationVector"];
		{data2, form} = encryptForm[data];
		encrypted = encryptInternal[key2, data2, 1];
		EncryptedObject[<|
			"Data" -> encrypted,
			If[ByteArrayQ[iv], "InitializationVector" -> iv, Nothing],
			"OriginalForm" -> form
		|>]
	]
];

(* 1-arg form displays authentication dialog to get the password *)
Encrypt[data_] := Module[
	{res},
	Catch[
		res = AuthenticationDialog["Password", WindowTitle -> "Encrypt"];
		If[!AssociationQ[res], Throw[$Failed]];
		res = Lookup[res, "Password", $Failed];
		If[!StringQ[res], Throw[$Failed]];
		Encrypt[res, data]
	]	
]


Clear[toKey];

assocMatchQ[assoc_Association, rules_] := MatchQ[Sort @ Normal @ assoc, Sort @ rules];

toKey[head_, SymmetricKey[assoc_Association], ivfunc_] := Module[
	{iv, cipher, blockmode, assoc2 = assoc, key, ivsize},
	
	{iv, cipher, blockmode, key} = Lookup[assoc, {"InitializationVector", "Cipher", "BlockMode", "Key"}];
	
	If[!MemberQ[$SymmetricCiphers, cipher], Return[$Failed]];
	
	If[!ValidBlockModeQ[blockmode, cipher], Return[$Failed]];
	
	If[CipherRequiresIVQ[cipher, blockmode],
		ivsize = $SymmetricIVSizes[cipher];
		Which[
			iv === None,
				assoc2["InitializationVector"] = ivfunc[ivsize/8],
			System`ByteArrayQ[iv],
				If[Length[iv]*8 =!= ivsize, Return[$Failed]],
			True,
				Return[$Failed]
		];

	];
	
	If[!System`ByteArrayQ[key] || !ValidKeySizeQ[Length[key], cipher], 
		Return[$Failed]
	];

	SymmetricKey[assoc2]
];

toKey[head_, key:PublicKey[KeyValuePattern[{
		"Type" -> "RSA",
		"Cipher" -> "RSA",
		"Padding" -> p_ /; MemberQ[$RSAPaddingModes, p],
		"PublicExponent" -> _Integer,
		"PublicModulus" -> _Integer
	}]], _] := key;
	
toKey[head_, key:PrivateKey[KeyValuePattern[{
		"Type" -> "RSA",
		"Cipher" -> "RSA",
		"Padding" -> p_ /; MemberQ[$RSAPaddingModes, p],
		"PrivateExponent" -> _Integer,
		"PublicExponent" -> _Integer,
		"PublicModulus" -> _Integer
	}]], _] := key;
	
toKey[head_, s_String, ivfunc_] := 
	Replace[Quiet @ GenerateSymmetricKey[s], {
		k_SymmetricKey :> toKey[head, k, ivfunc],
		_ :> $Failed
	}];

toKey[head:(Encrypt|Decrypt), key:(PublicKey|PrivateKey)[KeyValuePattern[{"Type" -> "EllipticCurve"}]], _] :=
(
	Message[MessageName[head, "ecckeyspec"]];
	Throw[$Failed]
)

toKey[head_, key_, ___] := $Failed;


PackageExport["Decrypt"]

unwrapEncryptedData[e:EncryptedObject[data_Association]] := With[{keys = Keys[data]}, If[
	SubsetQ[keys, {"Data", "OriginalForm"}] && ByteArrayQ @ data["Data"] && MemberQ[{ByteArray, String, Expression}, data["OriginalForm"]],
	MapAt[Normal, 1] @ Lookup[data, {"Data", "OriginalForm", "InitializationVector"}],
	Message[Decrypt::invdata , e]; Throw[$Failed]]
];

unwrapEncryptedData[ba_ByteArray ? ByteArrayQ] := {Normal[ba], ByteArray, None};

unwrapEncryptedData[e_] := (Message[Decrypt::invdata, e]; Throw[$Failed]);

wrapForm[ByteArray, data_List] := ByteArray[data];

wrapForm[String, data_List] := Quiet @ Check[
	FromCharacterCode[data, "UTF8"],
	Throw[Unevaluated[Message[Decrypt::failed]; $Failed]]];

wrapForm[Expression, data_List] := Quiet @ Check[
	Uncompress @ StringJoin["1:", Developer`EncodeBase64[data]], (* hate how ugly this is, but we don't yet have very good ByteArray support *)
	Throw[Unevaluated[Message[Decrypt::failed]; $Failed]]];
	
wrapForm[_, _] := (
	Message[Decrypt::failed];
	Throw[$Failed];
)

Clear[Decrypt];

Decrypt[key_, data_] := Module[
	{key2, data2, form, iv, decrypted},
	Catch[
		{data2, form, iv} = unwrapEncryptedData[data];
		key2 = toKey[Decrypt, key, iv&];
		If[key2 === $Failed, 
			Message[MessageName[Decrypt, "invkeyspec"]];
			Throw[$Failed];
		];
		decrypted = encryptInternal[key2, data2, 0];
		wrapForm[form, Normal@decrypted]
	]
]

(* 1-arg form displays authentication dialog to get the password *)
Decrypt[data_] := Module[
	{res},
	Catch[
		res = AuthenticationDialog["Password", WindowTitle -> "Decrypt"];
		If[!AssociationQ[res], Throw[$Failed]];
		res = Lookup[res, "Password", $Failed];
		If[!StringQ[res], Throw[$Failed]];
		Decrypt[res, data]
	]	
]
