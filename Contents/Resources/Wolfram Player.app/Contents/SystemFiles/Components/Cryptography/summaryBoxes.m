(* ::Package:: *)

Package["Cryptography`"]


getLength[s_String] := StringLength[s];
getLength[l: _List | _ByteArray] := Row[{Length[l]*8, " bits"}];
getLength[n_Integer] := Row[{IntegerLength[n, 2], " bits"}];
(*gets length of full bytes needed to store the number*)
getLength[n_Integer, "Full"] := Row[{Ceiling[BitLength[n], 8], " bits"}];

(* 
   "Length" of ByteArray, which packs x,y coordinates of elliptic curve.
   The first byte specifies packing type, so is not counted towards the length. 
*)
getPointLenth[ba_ByteArray] := Row[{(Length[ba] - 1) * 8, " bits"}];

getByteLength[b_ByteArray] := bytesToSize @ Length[b];
getByteLength[_] := Indeterminate;

bytesToSize[bytes_Integer] := Module[{i, k = 1000, sizes = {"bytes", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"}},
	If[bytes == 0,
		Return[ToString[bytes]  <> " bytes"];
	];
	i = Floor[Log[bytes]/Log[k]];
	If[i == 0,
		Return[ToString[bytes]  <> " bytes"];
		,
		Return[ToString[NumberForm[N[bytes/k^i], {4, 1}]] <> " " <> sizes[[i + 1]]]
	]
];

tofield["Type", vec_] := {"type: ", vec};
tofield["InitializationVector", vec_] := {"IV length: ", getLength[vec]};
tofield["InitializationVector", None] := {"IV: ", None};
tofield["OriginalForm", type_] := {"original form: ", type};
tofield["Key", key_] := {"key size: ", getLength[key]};
tofield["KeySize", keysize_] := {"key size: ", Row[{keysize, " bits"}]};
tofield["Cipher", method_] := {"cipher: ", method};
tofield["BlockMode", mode_] := {"block mode: ", mode}
tofield["Data", ciphertext_] := {"data length: ", getByteLength[ciphertext]};
tofield["Padding", packing_] := {"padding method: ", packing};
tofield["PublicExponent", exponent_] := {"public exponent: ", exponent};
tofield["PrivateExponent", exponent_] := {"private exponent size: ", getLength[exponent]};
tofield["PublicModulus", exponent_] := {"public modulus size: ", getLength[exponent]};
tofield["PrivateKey", key_] := {"private key size: ", getByteLength[key]};
tofield["PublicKey", key_] := {"public key size: ", getByteLength[key]};
tofield["EllipticCurve", method_] := {"elliptic curve: ", method};
tofield["PrivateMultiplier", key_] := {"private multiplier: ", getLength[key]};
tofield["PublicCurvePoint", key_] := {"public curve point: ", getLength[key]};
tofield["SignatureType", mode_] := {"signature type: ", decamelize[mode]};
tofield[fieldname_, _] := $Failed[fieldname];

makeKeyIcon[data_, colorFunction_] := ArrayPlot[
	Partition[IntegerDigits[Hash[data,"SHA256"],4,64],8],
		ImageSize -> 42,
		ColorFunction -> colorFunction, ColorFunctionScaling -> False,
		Frame -> None, PlotRangePadding -> None];

decamelize[str_String] := StringTrim @ StringReplace[str, s:LetterCharacter /; UpperCaseQ[s] :> (" " <> ToLowerCase[s])];
toextrafield[str_, value_] := {decamelize[str] <> ": ", value};
toextrafield[str_, value:_Integer] := {decamelize[str] <> ": ", Short[value, 0.25]};
toextrafield[str_, value:_String] := {
	If[UpperCaseQ[str],
		str <> ": ",
		decamelize[str] <> ": "],
	If[StringLength[value] >= 12,
		StringTake[value, 12]<>"...",
		value
	]
	};


$SymKeyValidKeys = {"Cipher", "BlockMode", "Key", "KeySize", "InitializationVector"};

SymmetricKey /: MakeBoxes[SymmetricKey[data_Association?AssociationQ], StandardForm] /; 
	ContainsAll[Keys[data], $SymKeyValidKeys] :=
	BoxForm`ArrangeSummaryBox[
		SymmetricKey, 
		SymmetricKey[data],
		makeKeyIcon[data, 106],
		System`KeyValueMap[tofield /* BoxForm`SummaryItem, KeyDrop[data, {"InitializationVector", "Key"}]],
		{},
		StandardForm];
	
$EncObjValidKeys = {"Data", "OriginalForm"};
	
EncryptedObject /: MakeBoxes[EncryptedObject[data_Association?AssociationQ], StandardForm] /; 
	ContainsAll[Keys[data], $EncObjValidKeys] :=
	BoxForm`ArrangeSummaryBox[
		EncryptedObject, 
		EncryptedObject[data],
		makeKeyIcon[data, 47],
		(tofield /* BoxForm`SummaryItem) @@@ Normal[data],
		{},
		StandardForm];
	
$PrKeyValidRSAKeys = {"Cipher", "Padding", "PublicExponent", "PublicModulus", "PrivateExponent"};

formatPrivate["RSA", data_] := Module[
	{main, extra, cipher, padding, pubexp, pubmod, privexp},
	{cipher, padding, pubexp, pubmod, privexp} = Lookup[data, $PrKeyValidRSAKeys, $Failed];
	(
		main = {
			BoxForm`SummaryItem[{"type: ", "RSA"}],
			BoxForm`SummaryItem[{"public modulus size: ", getLength[pubmod, "Full"]}],
			BoxForm`SummaryItem[{"private exponent size: ", getLength[privexp, "Full"]}]
		};
		extra = {
			BoxForm`SummaryItem[{"cipher: ", cipher}],
			BoxForm`SummaryItem[{"padding method: ", padding}],
			BoxForm`SummaryItem[{"public modulus: ", Short[pubmod, 0.25]}],
			BoxForm`SummaryItem[{"private exponent: ", Short[privexp, 0.25]}],
			BoxForm`SummaryItem[{"public exponent: ", pubexp}]
		};
		BoxForm`ArrangeSummaryBox[
			PrivateKey, 
			PrivateKey[data],
			makeKeyIcon[data, 100],
			main,
			extra, 
			StandardForm
		]
	) /; FreeQ[{cipher, padding, pubexp, pubmod, privexp}, $Failed]
]

$PubKeyValidRSAKeys = {"Cipher", "Padding", "PublicExponent", "PublicModulus"};

formatPublic["RSA", data_] := Module[
	{main, extra, cipher, padding, pubmod, pubexp},
	{cipher, padding, pubexp, pubmod} = Lookup[data, $PubKeyValidRSAKeys, $Failed];
	(
		main = {
			BoxForm`SummaryItem[{"type: ", "RSA"}],
			BoxForm`SummaryItem[{"public modulus size: ", getLength[pubmod, "Full"]}]
		};
		extra = {
			BoxForm`SummaryItem[{"cipher: ", cipher}],
			BoxForm`SummaryItem[{"padding method: ", padding}],
			BoxForm`SummaryItem[{"public modulus: ", Short[pubmod, 0.25]}],
			BoxForm`SummaryItem[{"public exponent: ", Short[pubexp, 0.25]}]
		};
		BoxForm`ArrangeSummaryBox[
			PublicKey, 
			PublicKey[data],
			makeKeyIcon[data, 63],
			main,
			extra, 
			StandardForm
		]
	) /; FreeQ[{cipher, padding, pubexp, pubmod}, $Failed]
]

$PrKeyValidECKeys = {(*"EllipticCurve",*) "CurveName", "PrivateMultiplier", "PublicCurvePoint", "Compressed", "PrivateByteArray", "PublicByteArray"};

formatPrivate["EllipticCurve", data_] := Module[
	{main, extra, curve, privmult, point, compressed, privba, pubba},
	{curve, privmult, point, compressed, privba, pubba} = Lookup[data, $PrKeyValidECKeys, $Failed];
	(
		main = {
			BoxForm`SummaryItem[{"type: ", StringJoin[{"elliptic curve (", curve, ")"}]}],
			BoxForm`SummaryItem[{"private key size: ", getLength[privba]}],
			BoxForm`SummaryItem[{"public key size: ", getPointLenth[pubba]}]
		};
		extra = {
			BoxForm`SummaryItem[{"elliptic curve: ", curve}],
			BoxForm`SummaryItem[{"private multiplier: ", Short[privmult, 0.25]}],
			BoxForm`SummaryItem[{"public curve point: ", Short[point, 0.25]}],
			BoxForm`SummaryItem[{"compressed: ", compressed}]	
		};
		BoxForm`ArrangeSummaryBox[
			PrivateKey, 
			PrivateKey[data],
			makeKeyIcon[data, 100],
			main,
			extra, 
			StandardForm
		]
	) /; FreeQ[{curve, privmult, point, compressed, privba, pubba}, $Failed]
]

$PubKeyValidECKeys = {(*"EllipticCurve",*) "CurveName", "PublicCurvePoint", "PublicByteArray"};

formatPublic["EllipticCurve", data_] := Module[
	{main, extra, curve, point, pubba},
	{curve, point, pubba} = Lookup[data, $PubKeyValidECKeys, $Failed];
	(
		main = {
			BoxForm`SummaryItem[{"type: ", StringJoin[{"elliptic curve (", curve, ")"}]}],
			BoxForm`SummaryItem[{"public key size: ", getPointLenth[pubba]}]
		};
		extra = {
			BoxForm`SummaryItem[{"elliptic curve: ", curve}],
			BoxForm`SummaryItem[{"public curve point: ", Short[point, 0.25]}]
		};
		BoxForm`ArrangeSummaryBox[
			PublicKey, 
			PublicKey[data],
			makeKeyIcon[data, 63],
			main,
			extra, 
			StandardForm
		]
	) /; FreeQ[{curve, point, pubba}, $Failed]
]

formatPrivate[___] := $Failed
formatPublic[___] := $Failed

PrivateKey /: MakeBoxes[PrivateKey[data_Association?AssociationQ], StandardForm] /; 
	ContainsAll[Keys[data], $PrKeyValidRSAKeys] || ContainsAll[Keys[data], $PrKeyValidECKeys] :=
	Module[{type, cipher, curve, res},
		{type, cipher, curve} = Lookup[data, {"Type", "Cipher", (*"EllipticCurve",*) "CurveName"}, $Failed];
		(
			res = Which[
				MatchQ[type, "RSA" | "EllipticCurve"],
					formatPrivate[type, data],
				type === $Failed && MatchQ[cipher, "RSA"], (* old RSA key *)
					data["Type"] = "RSA";
					formatPrivate["RSA", data],
				type === $Failed && curve =!= $Failed, (* old EllipticCurve key *)
					data["Type"] = "EllipticCurve";
					formatPrivate["EllipticCurve", data],
				True,
					$Failed
			];
			res		
		) /; FreeQ[res, $Failed]
	]
	

PublicKey /: MakeBoxes[PublicKey[data_Association?AssociationQ], StandardForm] /; 
	ContainsAll[Keys[data], $PubKeyValidRSAKeys] || ContainsAll[Keys[data], $PubKeyValidECKeys] :=
	Module[{type, cipher, curve, res},
		{type, cipher, curve} = Lookup[data, {"Type", "Cipher", (*"EllipticCurve",*) "CurveName"}, $Failed];
		(
			res = Which[
				MatchQ[type, "RSA" | "EllipticCurve"],
					formatPublic[type, data],
				type === $Failed && MatchQ[cipher, "RSA"], (* old RSA key *)
					data["Type"] = "RSA";
					formatPublic["RSA", data],
				type === $Failed && curve =!= $Failed, (* old EllipticCurve key *)
					data["Type"] = "EllipticCurve";
					formatPublic["EllipticCurve", data],
				True,
					$Failed
			];
			res
		) /; FreeQ[res, $Failed]
	];
		


$DigSigValidKeys = {"CurveName", "SignatureType", "HashingMethod", "R", "S"};
		
DigitalSignature /: MakeBoxes[DigitalSignature[data_Association?AssociationQ], StandardForm] /; 
	ContainsAll[Keys[data], $DigSigValidKeys] :=
	BoxForm`ArrangeSummaryBox[
		DigitalSignature, 
		DigitalSignature[data],
		makeKeyIcon[data, 55],
		{
		BoxForm`SummaryItem[{"type: ", StringJoin[{"elliptic curve (", data["CurveName"], ")"}]}],
		BoxForm`SummaryItem[{"signature type: ", decamelize[data["SignatureType"]]}]},
		(toextrafield /* BoxForm`SummaryItem) @@@ Normal[KeyDrop[data, {"Type", "CurveName", "SignatureType"}]],
		StandardForm];


toHexString[data:{___Integer}] := StringJoin[IntegerString[data, 16]]

toInfoField["KeySize", val_] := {"key size: ", val};
toInfoField[name_String, val_] := {name <> ": ", val};
toInfoField[___] := $Failed;

System`DerivedKey /: MakeBoxes[System`DerivedKey[data_Association?AssociationQ], StandardForm] := Module[
	{main, extra, function, derivedKey, salt, params, info},
	{function, derivedKey, salt, params} = Lookup[data, {"Function", "DerivedKey", "Salt", "Parameters"}, $Failed];
	(
		info = toInfoField @@@ Normal[params];
		(
			main = {
				BoxForm`SummaryItem[{"function: ", function}],
				BoxForm`SummaryItem[{"derived key: ", derivedKey}]
			};
			extra = Join[
				{BoxForm`SummaryItem[{"salt: ", salt}]},
				BoxForm`SummaryItem /@ info
			];
			BoxForm`ArrangeSummaryBox[
				System`DerivedKey,
				System`DerivedKey[data],
				makeKeyIcon[data, 96],
				main,
				extra,
				StandardForm
			]
		) /; FreeQ[info, $Failed]
	) /; FreeQ[{function, derivedKey, salt, params}, $Failed] && AssociationQ[params]
];

