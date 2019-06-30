(* ::Package:: *)

Package["Cryptography`"]


PackageExport["AddEllipticCurvePoints"]
(*EC point addition for curves y^2 = x^3 + ax + b (including Bitcoin curve)*)

AddEllipticCurvePoints[point1_, point2_, mod_Integer, acoef_Integer:0]:= 0 /; point1 == 0 && point2 == 0
AddEllipticCurvePoints[point1_, point2_, mod_Integer, acoef_Integer:0]:= point2 /; point1 == 0		
AddEllipticCurvePoints[point1_, point2_, mod_Integer, acoef_Integer:0]:= point1 /; point2 == 0	
AddEllipticCurvePoints[point1_, point2_, mod_Integer, acoef_Integer:0]:= Module[
	{alpha, x1, y1, y2, x2, x3, y3},
	 
	{x1, y1} = Mod[point1, mod];
	{x2, y2} = Mod[point2, mod];
	
	Which[
		(y1 === y2 === 0)||y1 === Mod[-y2, mod],
			Return[0], 
		{x1, y1} === {x2, y2}, 
			alpha = Mod[(3*PowerMod[x1, 2, mod] + acoef)*PowerMod[2y1, -1, mod], mod], 
		{x1, y1} =!= {x2, y2}, 
			alpha = Mod[(y1 - y2) PowerMod[(x1 - x2), -1, mod], mod]
		];
								
	x3 = Mod[PowerMod[alpha, 2, mod] - x1 - x2, mod];
	y3 = Mod[-y1 + alpha(x1 - x3), mod];
	
	{x3, y3}
	];				


PackageExport["MultiplyEllipticCurvePoints"]
(*EC point multiplication using double-and-add method*)

MultiplyEllipticCurvePoints[times_Integer, {point_, mod_Integer, acoef_Integer:0}] := 
	MultiplyEllipticCurvePoints[times, point, mod, acoef];
	
MultiplyEllipticCurvePoints[times_Integer, point_, mod_Integer, acoef_Integer:0] := Module[
	{Q = 0, tbin}, 
	
	tbin = IntegerDigits[times, 2];
	
	For[i = 1, i <= Length[tbin], i++, 
		Q = AddEllipticCurvePoints[Q, Q, mod, acoef];
		If[tbin[[i]] == 1, 
			Q = AddEllipticCurvePoints[Q, point, mod, acoef]
			]
		];
		
	Q
];


PackageExport["PublicKeyFormat"]
(* for adding here formattings for different cryptocurrencies *)

(*PublicKeyFormat::invpubkey = "`1` is not a valid public key format";*)

PublicKeyFormat[pubkey_List, "Bitcoin"|"Ethereum"|"secp256k1"] := PublicKeyFormat[pubkey, "secp256k1", "Compressed" -> False];

PublicKeyFormat[pubkey_List, "Bitcoin"|"Ethereum"|"secp256k1", "Compressed" -> False] := 
		"04" <> {IntegerString[#,  16, 64]& /@ pubkey};

PublicKeyFormat[pubkey_List, "Bitcoin"|"Ethereum"|"secp256k1", "Compressed" -> True] := 
		If[EvenQ[pubkey[[2]]], 
			"02" <> IntegerString[pubkey[[1]], 16, 64], 
			"03" <> IntegerString[pubkey[[1]], 16, 64]
			];

PublicKeyFormat[pubkey_List, "Bitcoin"|"Ethereum"|"secp256k1", "Compressed" -> False, "ByteArray"] := 
		ByteArray[Flatten[Prepend[IntegerDigits[pubkey, 256, 32], 4]]];

PublicKeyFormat[pubkey_List, "Bitcoin"|"Ethereum"|"secp256k1", "Compressed" -> True, "ByteArray"] := 
		If[EvenQ[pubkey[[2]]], 
			ByteArray[Flatten[Prepend[IntegerDigits[pubkey[[1]], 256, 32], 2]]], 
			ByteArray[Flatten[Prepend[IntegerDigits[pubkey[[1]], 256, 32], 3]]]
			];
			
PublicKeyFormat[pubkey_List, ec_, "Coordinates"] := pubkey;

PublicKeyFormat[pubkey_List, "Coordinates"] := pubkey;

PublicKeyFormat[pubkey_List] := PublicKeyFormat[pubkey, "secp256k1", "Compressed" -> False];

PublicKeyFormat[pubkey_String, ec_, "Coordinates"] := Catch @ Module[
	{a, b, p, x, y},
	
	Which[
		(StringLength[pubkey] == 130) && (StringTake[pubkey, 2] == "04"),
			Return[FromDigits[#, 16]& /@ StringTake[pubkey, {{3, 66}, {67, 130}}]],
			
		(StringLength[pubkey] == 66) && ((StringTake[pubkey, 2] == "03") || (StringTake[pubkey, 2] == "02")),
			If[MemberQ[$Cryptocurrencies, ec], 
					{a, b, p} = Lookup[$EllipticCurvesParameters[$CurrenciesEC[ec]],{"a", "b", "p"}],
					{a, b, p} = Lookup[$EllipticCurvesParameters[ec], {"a", "b", "p"}]
				];
				
			x = FromDigits[StringDrop[pubkey, 2], 16];
			y = PowerMod[PowerMod[x, 3, p] + a x + b, 1/2, p];
			
			If[(StringTake[pubkey, 2] == "03" && EvenQ[y]) || (StringTake[pubkey, 2] == "02" && OddQ[y]),
					y = p - y]; 
			
			Return[{x, y}],
		
		True, 
			Message[PublicKeyFormat::invpubkey, pubkey];
			Throw[$Failed]	
		]
	
];

PublicKeyFormat[pubkey_ByteArray, "Bitcoin"|"Ethereum"|"secp256k1", "Coordinates"] := Catch @ Module[
	{listpubkey, a, b, p, x, y},
	
	listpubkey = Developer`FromByteArray[pubkey];
	Which[
		(Length[listpubkey] == 65) && (listpubkey[[1]] == 4),
			Return[FromDigits[#, 256]& /@{listpubkey[[2;;33]],listpubkey[[34;;]]}],
			
		(Length[pubkey] == 33) && ((listpubkey[[1]]== 3) || (listpubkey[[1]] == 2)),
			If[MemberQ[$Cryptocurrencies, ec], 
					{a, b, p} = Lookup[$EllipticCurvesParameters[$CurrenciesEC[ec]],{"a", "b", "p"}],
					{a, b, p} = Lookup[$EllipticCurvesParameters[ec], {"a", "b", "p"}]
				];
				
			x = FromDigits[Drop[listpubkey, 1], 256];
			y = PowerMod[PowerMod[x, 3, p] + a x + b, 1/2, p];
			
			If[(listpubkey[[1]] == 3 && EvenQ[y]) || (listpubkey[[1]] == 2 && OddQ[y]),
					y = p - y]; 
			
			Return[{x, y}],
		
		True, 
			Message[PublicKeyFormat::invpubkey, pubkey];
			Throw[$Failed]	
		]
	
]

PublicKeyFormat[___] := $Failed


PackageExport["GenerateEllipticCurvePublicKey"]
(* generates a public key on a given curve from a given integer or hex string private key  *)
GenerateEllipticCurvePublicKey[privkey_String, ec_String, opts___] := GenerateEllipticCurvePublicKey[FromDigits[privkey, 16], ec, opts];

GenerateEllipticCurvePublicKey[privkey_Integer, ec_String, opts___] := 
	PublicKeyFormat[
		MultiplyEllipticCurvePoints[privkey, Lookup[$EllipticCurvesParameters[ec], {"G", "p", "a"}]],
		ec,
		opts
		] /; MemberQ[$EllipticCurves, ec];

GenerateEllipticCurvePublicKey[privkey_Integer, curr_String, opts___] := 
	PublicKeyFormat[
		MultiplyEllipticCurvePoints[privkey, Lookup[$EllipticCurvesParameters[$CurrenciesEC[curr]], {"G", "p", "a"}]],
		curr,
		opts
		] /; MemberQ[$Cryptocurrencies, curr];

GenerateEllipticCurvePublicKey[privkey_Integer, G_List, p_Integer, a_Integer:0] := 
	PublicKeyFormat[
		MultiplyEllipticCurvePoints[privkey, G, p, a]
		]

GenerateEllipticCurvePublicKey[privkey_Integer, G_List, p_Integer, a_Integer:0, "Coordinates"] := 
	PublicKeyFormat[
		MultiplyEllipticCurvePoints[privkey, G, p, a], 
		"Coordinates"
		]


PackageScope["GenerateECKeyPair"]

GenerateECKeyPair[ec_, compress_]:= Module[
	{privkey, intprivkey, newec = ec, n, pointpubkey, keysize},
	
	If[MemberQ[$Cryptocurrencies, ec], 
		newec = $CurrenciesEC[ec]
	];
	
	{n, keysize} = Lookup[$EllipticCurvesParameters[newec], {"n", "KeySize"}];
	
	privkey = randomBytes[keysize/8];
	intprivkey = FromDigits[Developer`FromByteArray[privkey], 256];
	
	If[intprivkey >= n, 
		Message[GenerateAsymmetricKeyPair::failure]; 
		Throw[$Failed]
		];
		
	pointpubkey = GenerateEllipticCurvePublicKey[intprivkey, newec, "Coordinates"];
	
	<|
		"Type" -> "EllipticCurve",
		"CurveName" -> newec, 
		"PublicCurvePoint" -> pointpubkey,
		"PrivateMultiplier"-> intprivkey,
		"Compressed" -> Values[compress], 
		"PublicByteArray" -> PublicKeyFormat[pointpubkey, newec, "Compressed" -> Values[compress], "ByteArray"], 
		"PrivateByteArray" -> privkey
	|>
	];
