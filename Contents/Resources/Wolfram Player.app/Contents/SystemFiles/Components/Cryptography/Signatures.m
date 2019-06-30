(* ::Package:: *)

Package["Cryptography`"]


PackageExport["DigitalSignature"]

DigitalSignature[data_Association]["Parameters"] := data;
DigitalSignature[data_Association]["Properties"] := Keys[data];
DigitalSignature[data_Association][key_] := data[key];


PackageScope["DEREncode"]

(*
	Encodes a signature (pair of integers r and s) to a DER format, consists of:
		- 0x30: a header byte indicating a compound structure.
		- b1: A 1-byte length descriptor for all what follows.
		- 0x02: a header byte indicating an integer.
		- b2: A 1-byte length descriptor for the r value.
		- r: first part of the signature as a signed big-endian integer.
		- 0x02: a header byte indicating an integer.
		- A 1-byte length descriptor for the s value.
		- s: second part of the signature as a signed big-endian integer.
	
*)

DEREncode[r_Integer, s_Integer] := Module[
	{b1, b2, b3, vr, vs},
		
		(* divide by 8 to get number of bytes needed to represent r, multiply by 2 because 1 byte is 2 string places*)
		vr = IntegerString[r, 16, 2 Ceiling[BitLength[r]/8]];
		If[FromDigits[StringTake[vr, 2], 16] > 127,
			(* 00 is added so the first bit of the first byte is 0 and indicates a positive "signed" number*)
			vr = "00" <> vr;
			];
		(* 3.2 is used, because when vr is 32 bytes(StringLength is 64) b2 is 20, and when vr is 33 bytes b2 is 21*)
		b2 = Ceiling[StringLength[vr]/3.2];
		
		vs = IntegerString[s, 16, 2 Ceiling[BitLength[s]/8]];
		If[FromDigits[StringTake[vs, 2], 16] > 127,
			vs = "00" <> vs;
			];
		b3 = Ceiling[StringLength[vs]/3.2];
		
		(* b1 is the total "length" of all that follows: total of b2 and b3 plus 4 bytes 0x02 b2 and 0x02 b3 occupy *)
		b1= 4 + b2 + b3;
		
		"30" <> ToString[b1] <> "02" <> ToString[b2] <> vr <> "02" <> ToString[b3] <> vs
]
	
PackageScope["DERDecode"]

(* takes a DER-encoded signature String "304.... and takes the (r, s) parts from their places *)

DERDecode[sig_?StringQ] := Catch@Module[
	{r, s},
	
	Which[
		StringTake[sig, {7, 8}] == "20",
			r = StringTake[sig, {9, 72}];
			s = StringDrop[sig, 76],
			
		StringTake[sig, {7, 8}] == "21",
			r = StringTake[sig, {9, 74}];
			s = StringDrop[sig, 78],
		True,
			Throw[$Failed]
		];
	
	{r, s}
	
	] /; StringMatchQ[sig, HexadecimalCharacter..]


PackageExport["GenerateDigitalSignature"]
(*
GenerateDigitalSignature::invec = "`1` is not a supported elliptic curve specification.";
GenerateDigitalSignature::invkey = "`1` is not a valid elliptic curve private key.";
GenerateDigitalSignature::invmethod = "The value of the option Method -> `` should be an association or Automatic.";
GenerateDigitalSignature::invmode = "The value of the option \"SignatureType\" -> `1` should be either Deterministic or NonDeterministic.";
GenerateDigitalSignature::invarg = "Invalid set of arguments."; 
GenerateDigitalSignature::invdeterk = "Failed to calculate deterministic signature.";
GenerateDigitalSignature::invtype = GenerateAsymmetricKeyPair::invtype;
GenerateDigitalSignature::invcurvename = GenerateAsymmetricKeyPair::invcurvename;
GenerateDigitalSignature::invhash = "`1` is not a valid hashing method.";
GenerateDigitalSignature::hashnone = VerifyDigitalSignature::hashnone =
		"Input `1` should be a string of hexadecimal characters or a ByteArray.";
*)
(* not used messages *)
(* GenerateDigitalSignature::notstr = VerifyDigitalSignature::notstr = "Input `1` should be a string of hexadecimal characters.";*)
(* GenerateDigitalSignature::notstrint = "Input `1` should be a string of hexadecimal characters or an integer.";*)

(*
	Parts of the signature (r, s) and secret number k are standard notation
		ref: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.186-4.pdf
		
	z is not a standard notation
*)

Options[GenerateDigitalSignature] = {Method -> <|"Type"-> "EllipticCurve", "HashingMethod" -> "SHA256", "CurveName" -> "secp256k1", "SignatureType" -> Automatic|>};

$MethodKeys = {"Type", "EllipticCurve", "CurveName", "SignatureType", "HashingMethod"}

GenerateDigitalSignature[args___] := Module[
	{res},
	res = Catch[doGenerateDigitalSignature[args], iGenerateDigitalSignature];
	res /; res =!= $Failed
]

Options[doGenerateDigitalSignature] = Options[GenerateDigitalSignature]

doGenerateDigitalSignature[expr_, privkey:Except[_Rule], opts:OptionsPattern[GenerateDigitalSignature]] := Module[
	{method, hash, hashres},
	method = OptionValue[Method];
	If[method === Automatic, method = <||>];
	If[!AssociationQ[method],
		Message[GenerateDigitalSignature::invmethod, method]; 
		Throw[$Failed, iGenerateDigitalSignature]
	];
	hash = Lookup[method, "HashingMethod", "SHA256"];
	Switch[hash,
		None, (* expr is a digest string or a ByteArray *)
			Which[
				StringQ[expr] && StringMatchQ[expr, HexadecimalCharacter..],
					iGenerateDigitalSignature[ByteArray[IntegerDigits[FromDigits[expr, 16], 256]], privkey, method],
				ByteArrayQ[expr],
					iGenerateDigitalSignature[expr, privkey, method],
				True,
					Message[GenerateDigitalSignature::hashnone, expr]; 
					$Failed],
			
		_?StringQ,
			hashres = Hash[expr, hash, "ByteArray"]; (* check if evaluated correctly *)
			If[ByteArrayQ[hashres],
				iGenerateDigitalSignature[hashres, privkey, method]
				,
				Message[GenerateDigitalSignature::invhash, hash];
				$Failed
			],
		_,
			Message[GenerateDigitalSignature::invhash, hash];
			$Failed
	]
]

doGenerateDigitalSignature[args___:Except[_Rule], opts:OptionsPattern[GenerateDigitalSignature]] := Block[
	{len = Length[{args}]}
	,
	If[len =!= 2, Message[GenerateDigitalSignature::argrx, GenerateDigitalSignature, len, 2]];
	$Failed
]

(*"SignatureType" shouldn't be alone*)
(*iGenerateDigitalSignature[digest_?ByteArrayQ, privkey_, <|"SignatureType" -> value_|>] :=
	iGenerateDigitalSignature[digest, privkey, <|"EllipticCurve" -> "secp256k1", "SignatureType" -> value|>];*)
	
iGenerateDigitalSignature[digest_?ByteArrayQ, privkey_, <||>] :=
	iGenerateDigitalSignature[digest, privkey, <|"Type"->"EllipticCurve", "CurveName" -> "secp256k1"|>];
	
iGenerateDigitalSignature[digest_?ByteArrayQ, privkey_PrivateKey, method_] := 
	(*at least one of PrivateMultiplier or PrivateByteArray should be present in the object*)
	If[MissingQ[privkey["PrivateMultiplier"]],
		If[MissingQ[privkey["PrivateByteArray"]],
			Message[GenerateDigitalSignature::invkey, privkey];
			Throw[$Failed, iGenerateDigitalSignature],
		
			iGenerateDigitalSignature[digest, FromDigits[Developer`FromByteArray[privkey["PrivateByteArray"]], 256], method]
			],
			
		iGenerateDigitalSignature[digest, privkey["PrivateMultiplier"], method]
			
	]/;privkey["Type"]=="EllipticCurve";
	
iGenerateDigitalSignature[digest_?ByteArrayQ, privkey_PrivateKey, method_] := 
	(
			Message[GenerateDigitalSignature::invkey, privkey];
			Throw[$Failed, iGenerateDigitalSignature]
			
	)/;privkey["Cipher"]=="RSA";
	
iGenerateDigitalSignature[digest_?ByteArrayQ, privkey_?ByteArrayQ, method_] := 
		iGenerateDigitalSignature[digest, FromDigits[Developer`FromByteArray[privkey], 256], method]
	
iGenerateDigitalSignature[digest_?ByteArrayQ, privkey_?StringQ, method_] := 
	If[StringMatchQ[privkey, HexadecimalCharacter..],
		iGenerateDigitalSignature[digest, FromDigits[privkey, 16], method],
		Message[GenerateDigitalSignature::invkey, privkey];
		Throw[$Failed, iGenerateDigitalSignature]
	];

iGenerateDigitalSignature[digest_?ByteArrayQ, privkey_?IntegerQ, method_] := Module[
	{inmethod, type, ec, curr = Null, mode, G, n, a, p, z, k, r, rec, s, outextra = <||>, hash, keysize},
	
	If[StringQ[method],
		Which[
			MemberQ[Join[$EllipticCurves, $Cryptocurrencies], method],
					inmethod = <|"CurveName" -> method|>,
			True, 
					Message[GenerateDigitalSignature::invec, method]; 
					Throw[$Failed, iGenerateDigitalSignature]
			]
		,
		If[AssociationQ[method], 
			inmethod = method,
			Message[GenerateDigitalSignature::invmethod, method]; 
			Throw[$Failed, iGenerateDigitalSignature]
		];
	];
	
	If[!MemberQ[$MethodKeys, #],
		 	Message[GenerateDigitalSignature::invmethodkey, #, $MethodKeys];
		 	Throw[$Failed, iGenerateDigitalSignature];
		 ]&/@ Keys[inmethod];
		 
	type = Lookup[inmethod, "Type", "EllipticCurve"];
	
	If[type =!= "EllipticCurve",
		Message[GenerateDigitalSignature::invtype, type]; 
					Throw[$Failed, iGenerateDigitalSignature]];
		
		
	If[MemberQ[Keys[inmethod], "EllipticCurve"],
		ec = Lookup[inmethod, "EllipticCurve"],
		ec = Lookup[inmethod, "CurveName", "secp256k1"]
		];
	
	hash = Lookup[inmethod, "HashingMethod", "SHA256"];
	mode = Lookup[inmethod, "SignatureType", "NonDeterministic"];
	
	If[mode == Automatic, mode = "NonDeterministic"];
	
	If[!StringQ[mode] || !StringMatchQ[mode, "NonDeterministic" |"Deterministic" ],
		Message[GenerateDigitalSignature::invmode, mode];
		Throw[$Failed, iGenerateDigitalSignature]
	];
			
	If[MemberQ[$Cryptocurrencies, ec],
		curr = ec;
		ec = $CurrenciesEC[ec]
		];
		
	If[!MemberQ[$EllipticCurves, ec],
		Message[GenerateDigitalSignature::invcurvename, ec]; 
		Throw[$Failed, iGenerateDigitalSignature]
	];
		
	{G, n, a, p, keysize} = Lookup[$EllipticCurvesParameters[ec], {"G", "n", "a", "p", "KeySize"}];

	If[privkey > n,
		Message[GenerateDigitalSignature::invkey, privkey];
		Throw[$Failed, iGenerateDigitalSignature]
	];

	z = Mod[FromDigits[Developer`FromByteArray[digest], 256], n];
	
	Switch[mode,
		"NonDeterministic",
			k = Mod[FromDigits[Developer`FromByteArray[randomBytes[keysize/8]],  256], n],
		"Deterministic",
			If[MemberQ[{"Adler32", "CRC32", None, "None", "Expression"}, hash], (*for strange cases just use SHA256*)
				k = DeterministicK[IntegerDigits[privkey, 256], Developer`FromByteArray[digest], "SHA256", n],
				
				k = DeterministicK[IntegerDigits[privkey, 256], Developer`FromByteArray[digest], hash, n]]
		];

	r = MultiplyEllipticCurvePoints[k, G, p, a];
   
    rec = If[OddQ[r[[2]]], 1, 0];
    
    r = Mod[r[[1]], n];

	s = Mod[PowerMod[k, -1, n](z + r*privkey), n];
	(* ensure that s value is always at the lower part of its range (known malleability) *)
	If[s > Floor[n/2], s = n - s; rec = Switch[rec, 0, 1, 1, 0]];
	
	Switch[ curr,
		"Bitcoin"|"BTC",
			outextra = <|"DER" -> DEREncode[r, s]|>,
		"Ethereum"|"ETH",
			outextra = <|"RecoveryParameter" -> rec|>
	];

	DigitalSignature[
		Join[
			<|
				"Type" -> "EllipticCurve",
				"CurveName" -> ec, 
				"SignatureType" -> mode,
				"HashingMethod" -> hash,
				"R" -> ByteArray[IntegerDigits[r, 256, keysize/8]],
				"S" -> ByteArray[IntegerDigits[s, 256, keysize/8]]			
			|>,
			outextra
		]		
	]
];

iGenerateDigitalSignature[_, privkey: Except[_?IntegerQ|_?StringQ|_PrivateKey], _] := (
	Message[GenerateDigitalSignature::invkey, privkey]; 
	Throw[$Failed, iGenerateDigitalSignature]
);
	
iGenerateDigitalSignature[args___] := If[ 
	!ArgumentCountQ[GenerateDigitalSignature, Length[{args}], 2, 2],
	Throw[$Failed, iGenerateDigitalSignature]
];
	


PackageScope["SignatureVerify"]

(*
   Bitcoin uses secp256k1. Our implementation does NOT require reversed digest.
*)
SignatureVerify[signature_?StringQ, digest_?StringQ, pubkey_?StringQ, "secp256k1"] := Module[
	{n, digdata, sigdata, pubkeydata}
	,
	n = Lookup[$EllipticCurvesParameters["secp256k1"], "n"];
	sigdata = IntegerDigits[FromDigits[signature, 16], 256];
	pubkeydata = IntegerDigits[FromDigits[pubkey, 16], 256];
	digdata = IntegerDigits[Mod[FromDigits[digest, 16], n], 256];
	
	Catch[
		Replace[sigVerify[digdata, pubkeydata, sigdata],
			_LibraryFunctionError -> $Failed]
	]
];

SignatureVerify[{rval_Integer, sval_Integer}, digest_?StringQ, pubkey_?StringQ, "secp256k1"] := Module[
	{n, digdata, pubkeydata, r, s}
	,
	n = Lookup[$EllipticCurvesParameters["secp256k1"], "n"];
	pubkeydata = IntegerDigits[FromDigits[pubkey, 16], 256];
	r = IntegerDigits[rval, 256];
	s = IntegerDigits[sval, 256];
	
	digdata = IntegerDigits[Mod[FromDigits[digest, 16], n], 256];
	(* taking the digest modulo n if it is too big ( > bit than the group order n) *)
	
	Catch[
		Replace[sigVerifyRS[digdata, pubkeydata, r, s],
			_LibraryFunctionError -> $Failed]	
	]
];

SignatureVerify[{rval_?ByteArrayQ, sval_?ByteArrayQ}, digest_?ByteArrayQ, pubkey_?ByteArrayQ, "secp256k1"] := Module[
	{n, digdata}
	,
	n = Lookup[$EllipticCurvesParameters["secp256k1"], "n"];
	
	digdata = IntegerDigits[Mod[FromDigits[Developer`FromByteArray[digest], 256], n], 256];
	(* taking the digest modulo n if it is too big ( > bit than the field size n) *)
	
	Catch[
		Replace[sigVerifyRS[digdata, Developer`FromByteArray[pubkey], Developer`FromByteArray[rval], Developer`FromByteArray[sval]],
			_LibraryFunctionError -> $Failed]	
	]
];

SignatureVerify[___] := $Failed


PackageExport["VerifyDigitalSignature"]
(*
VerifyDigitalSignature::invsig = "Signature `1` should be a valid DigitalSignature object.";
VerifyDigitalSignature::invpubkey = "`1` is not a valid elliptic curve public key.";
VerifyDigitalSignature::invarg = "`1` is not a pair {expr, sig} or a list of such pairs."
*)
VerifyDigitalSignature[expr_, pubkey_] := Module[
	{res},
	res = Catch[
		listVerifyDigitalSignature[expr, pubkey],
		iVerifyDigitalSignature
	];
	res /; res =!= $Failed && Head[res] =!= listVerifyDigitalSignature
]

VerifyDigitalSignature[pubkey_][expr_] := VerifyDigitalSignature[expr, pubkey]

VerifyDigitalSignature[args___] := Module[
	{res, len = Length[{args}]},
	(
		res = (
			Message[VerifyDigitalSignature::argt, VerifyDigitalSignature, len, 1, 2];
			$Failed	
		);
		res /; res =!= $Failed
	) /; len < 1 || len > 2
]

ExprSigPairQ[{_, _DigitalSignature}] := True
ExprSigPairQ[___] := False

ExprSigListQ[list_List] := AllTrue[list, ExprSigPairQ]
ExprSigListQ[___] := False

listVerifyDigitalSignature[list_?ExprSigListQ, pubkey_] := 
	iVerifyDigitalSignature[#, pubkey]& /@ list;
	
listVerifyDigitalSignature[{expr_, sig_DigitalSignature}, pubkey_] := 
	iVerifyDigitalSignature[expr, sig, pubkey];
	
listVerifyDigitalSignature[{expr_, sig:Except[_DigitalSignature]}, pubkey_] := 
	(
	Message[VerifyDigitalSignature::invsig, sig]; 
	Throw[$Failed, iVerifyDigitalSignature]
	);
	
listVerifyDigitalSignature[pubkey_][list_List] := listVerifyDigitalSignature[list, pubkey];

listVerifyDigitalSignature[arg_?(!ExprSigListQ[#]&), ___] := (
	Message[VerifyDigitalSignature::invarg, arg];
	Throw[$Failed, iVerifyDigitalSignature]
)

listVerifyDigitalSignature[___] := Throw[$Failed, iVerifyDigitalSignature]

iVerifyDigitalSignature[{expr_, sig_}, pubkey_] := iVerifyDigitalSignature[expr, sig, pubkey];

iVerifyDigitalSignature[expr_, sig_DigitalSignature, pubkey_List] :=
	If[!MatchQ[pubkey, {_Integer, _Integer}],		
		Message[VerifyDigitalSignature::invpubkey, pubkey]; 
		Throw[$Failed, iVerifyDigitalSignature],		
		iVerifyDigitalSignature[expr, sig, ByteArray[Flatten@Join[{4}, IntegerDigits[#, 256]&/@pubkey]]]
	];

iVerifyDigitalSignature[expr_, sig_DigitalSignature, pubkey_?StringQ] := 
	If[!StringMatchQ[pubkey, HexadecimalCharacter..],		
		Message[VerifyDigitalSignature::invpubkey, pubkey]; 
		Throw[$Failed, iVerifyDigitalSignature],		
		iVerifyDigitalSignature[expr, sig, ByteArray[IntegerDigits[FromDigits[pubkey, 16], 256]]]
	];
	
iVerifyDigitalSignature[expr_, sig_DigitalSignature, pubkey_PublicKey] :=
	(*at least one of PublicCurvePoint or PublicByteArray should be present in the object*)
	If[!MissingQ[pubkey["PublicByteArray"]],
	
		iVerifyDigitalSignature[expr, sig, pubkey["PublicByteArray"]],
		
		If[And@@(!MissingQ[#]&/@{pubkey["PublicCurvePoint"], pubkey["CurveName"], pubkey["Compressed"]}),
					
			iVerifyDigitalSignature[expr, sig, PublicKeyFormat[pubkey["PublicCurvePoint"], pubkey["CurveName"], "Compressed" -> pubkey["Compressed"], "ByteArray"]],
			
			Message[VerifyDigitalSignature::invpubkey, pubkey];
			Throw[$Failed, iVerifyDigitalSignature]			
			]	
		
	];

iVerifyDigitalSignature[expr_, sig:DigitalSignature[assoc__], pubkey_?ByteArrayQ] := Module[
	{ec, r, s, hash}
	,
	If[Length[{assoc}] > 1||!AssociationQ[assoc],
		Message[VerifyDigitalSignature::invsig, sig];
		Throw[$Failed, iVerifyDigitalSignature]
	];
	{ec, r, s, hash} = Lookup[assoc, {"CurveName", "R", "S", "HashingMethod"}, $Failed];
	If[MemberQ[{ec, r, s, hash}, $Failed],
		Message[VerifyDigitalSignature::invsig, sig]; 
		Throw[$Failed, iVerifyDigitalSignature]
	];

	Switch[hash,
			None, (* expr is a digest string or a ByteArray*)
				Which[
					StringQ[expr] && StringMatchQ[expr, HexadecimalCharacter..],
						SignatureVerify[{r, s}, ByteArray[IntegerDigits[FromDigits[expr, 16], 256]], pubkey, ec],
					ByteArrayQ[expr],
						SignatureVerify[{r, s}, expr, pubkey, ec],
					True,
						Message[VerifyDigitalSignature::hashnone, expr]; 
						Throw[$Failed, iVerifyDigitalSignature]],
				
			_?StringQ,
				SignatureVerify[{r, s}, Hash[expr, hash, "ByteArray"], pubkey, ec],
			_,
				Throw[$Failed, iVerifyDigitalSignature]
		]
];

iVerifyDigitalSignature[_, sig: Except[_DigitalSignature], _] := (
	Message[VerifyDigitalSignature::invsig, sig]; 
	Throw[$Failed, iVerifyDigitalSignature]
	);
	
iVerifyDigitalSignature[_, sig: DigitalSignature[], _] := (
	Message[VerifyDigitalSignature::invsig, sig]; 
	Throw[$Failed, iVerifyDigitalSignature]
	);
	
iVerifyDigitalSignature[_, _, pubkey:Except[_PublicKey|_?StringQ|_?ByteArrayQ]] := (
	Message[VerifyDigitalSignature::invpubkey, pubkey]; 
	Throw[$Failed, iVerifyDigitalSignature]
	);
	
iVerifyDigitalSignature[args___] := If[
	!ArgumentCountQ[VerifyDigitalSignature, Length[{args}], 3, 3],
	Throw[$Failed, iVerifyDigitalSignature]
	];
