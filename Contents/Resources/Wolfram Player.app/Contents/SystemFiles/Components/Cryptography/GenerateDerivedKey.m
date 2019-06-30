(* ::Package:: *)

Package["Cryptography`"]

PackageExport["System`GenerateDerivedKey"]
PackageExport["System`VerifyDerivedKey"]
PackageExport["System`DerivedKey"]
(*
GenerateDerivedKey::optx1 = "Unknown option `1`."
GenerateDerivedKey::nassoc = "`1` is not an Association."
GenerateDerivedKey::nparm = "Incorrect parameters for `1` specified."
GenerateDerivedKey::keysize = "No output key size specified."
GenerateDerivedKey::passwd = VerifyDerivedKey::passwd = "Password is not a string, a list of positive integers, or a ByteArray."
GenerateDerivedKey::salt = "Salt is not a string, a list of positive integers, or a ByteArray."
GenerateDerivedKey::alg = "Unknown key derivation algorithm `1`."
GenerateDerivedKey::msize = "Parameters specified in `1` must be machine-size integers."
GenerateDerivedKey::nbyte = "`1` is not a list of integer values between 0 and 255."

GenerateDerivedKey::tmax = "The value `1` of `2` is too large."
GenerateDerivedKey::tpos = "The value `1` of `2` must be a positive integer."

GenerateDerivedKey::tmin = "The value of t is too small."
GenerateDerivedKey::mmin = "The value of m is too small."
GenerateDerivedKey::pmin = "The value of p is too small."

GenerateDerivedKey::sout = "Output size is too small."
GenerateDerivedKey::lout = "Output size is too large."
GenerateDerivedKey::spwd = "Password size is too small."
GenerateDerivedKey::lpwd = "Password size is too large."
GenerateDerivedKey::sslt = "Salt size is too small."
GenerateDerivedKey::lslt = "Salt size is too large."

GenerateDerivedKey::type = "Unknown algorithm `1`."
GenerateDerivedKey::invn = "The value `1` of N must be a power of 2 greater than 1."
GenerateDerivedKey::err = "Could not compute derived key."

DerivedKey::invstr = "String `1` cannot be interpreted as a derived key."
*)
(* "2017Default" *)
$ScryptDefaultParameters = <|"KeySize" -> 64, "N" -> 32768, "r" -> 8, "p" -> 1|>;

$Argon2DefaultParameters = <|"KeySize" -> 64, "t" -> 2, "m" -> 2^16, "p" -> 1|>

$DefaultGenerateDerivedKeyMethod = <|
	"Function" ->"scrypt",
	"Parameters" -> $ScryptDefaultParameters
|>;

Options[System`GenerateDerivedKey] = {Method -> $DefaultGenerateDerivedKeyMethod}

checkOpts[sym_, opts_, allowed_] := Module[
	{extra = Complement[opts, allowed]},
	If[Length[extra] != 0,
		Message[MessageName[sym, "optx1"], First[extra]];
		Throw[$Failed]
	]
]

parseGenerateDerivedKeyMethod[method_] := Module[
	{function, params}
	,
	If[!AssociationQ[method],
		Message[GenerateDerivedKey::"nassoc", method];
		Throw[$Failed]
	];
	checkOpts[GenerateDerivedKey, Keys[method], {"Function", "Parameters"}];
	function = Lookup[method, "Function", "scrypt"];
	params = Lookup[method, "Parameters", $Failed];
	If[params === $Failed,
		Switch[function,
			"scrypt",
				params = $ScryptDefaultParameters
			,
			"Argon2i" | "Argon2d" | "Argon2id",
				params = $Argon2DefaultParameters
			,
			_,
				Message[GenerateDerivedKey::nparm, function];
				Throw[$Failed]
		]
	];
	If[!AssociationQ[params],
		Message[GenerateDerivedKey::"nassoc", params];
		Throw[$Failed]
	];
	{function, params}
]

doGenerateDerivedKey[alg:"scrypt",
	pw:{___Integer}, salt:{___Integer}, params_] := Module[
	{keysize, n, r, p, key}
	,
	checkOpts[GenerateDerivedKey, Keys[params], {"KeySize", "N", "r", "p"}];
	n = Lookup[params, "N", Lookup[$ScryptDefaultParameters, "N", $Failed]];
	r = Lookup[params, "r", Lookup[$ScryptDefaultParameters, "r", $Failed]];
	p = Lookup[params, "p", Lookup[$ScryptDefaultParameters, "p", $Failed]];
	
	keysize = Lookup[params, "KeySize", $Failed];
	If[!IntegerQ[keysize],
		Message[GenerateDerivedKey::keysize];
		Throw[$Failed]
	];
	If[!FreeQ[{n, r, p}, $Failed],
		Message[GenerateDerivedKey::nparm, alg];
		Throw[$Failed]
	];
	If[!Developer`MachineIntegerQ[n] || !Developer`MachineIntegerQ[r] || !Developer`MachineIntegerQ[p],
		Message[GenerateDerivedKey::msize, params];
		Throw[$Failed]
	];
	key = scrypt[pw, salt, keysize, n, r, p];

	If[!ListQ[key],
		Throw[$Failed]
	];
	<|
	   "Function" -> "scrypt",
	   "DerivedKey" -> ByteArray[key],
       "Salt" -> ByteArray[salt],
	   "Parameters" -> <|
	       "KeySize" -> keysize,
	       "N" -> n,
	       "r" -> r,
	       "p" -> p
	   |> 
	|>
]

doGenerateDerivedKey[alg:("Argon2i" | "Argon2d" | "Argon2id"),
	pw:{___Integer}, salt:{___Integer}, params_] := Module[
	{keysize, t, m, p, idx, key}
	,
	checkOpts[GenerateDerivedKey, Keys[params], {"KeySize", "t", "m", "p"}];
	t = Lookup[params, "t", Lookup[$Argon2DefaultParameters, "t", $Failed]];
	m = Lookup[params, "m", Lookup[$Argon2DefaultParameters, "m", $Failed]];
	p = Lookup[params, "p", Lookup[$Argon2DefaultParameters, "p", $Failed]];

	idx = Lookup[$Argon2Types, alg, $Failed];
	If[idx === $Failed,
		Message[GenerateDerivedKey::alg, alg];
		Throw[$Failed]
	];
	keysize = Lookup[params, "KeySize", $Failed];
	If[!IntegerQ[keysize],
		Message[GenerateDerivedKey::keysize];
		Throw[$Failed]
	];
	If[!FreeQ[{t, m, p}, $Failed],
		Message[GenerateDerivedKey::nparm, alg];
		Throw[$Failed]
	];
	If[!Developer`MachineIntegerQ[t] || !Developer`MachineIntegerQ[m] || !Developer`MachineIntegerQ[p],
		Message[GenerateDerivedKey::msize, params];
		Throw[$Failed]
	];
	key = argon2[pw, salt, keysize, t, m, p, idx];
	
	If[!ListQ[key],
		Throw[$Failed]
	];
	<|
	   "Function" -> alg,
	   "DerivedKey" -> ByteArray[key],
       "Salt" -> ByteArray[salt],
	   "Parameters" -> <|
	       "KeySize" -> keysize,
	       "t" -> t,
	       "m" -> m,
	       "p" -> p
	   |> 
	|>
]

doGenerateDerivedKey[___] := Throw[$Failed]

derivedKeyParameters[alg:"scrypt", args_Association] := Module[
	{n, r, p, params}
	,
	{n, r, p} = Lookup[args, {"N", "r", "p"}, $Failed];
	If[!FreeQ[{n, r, p}, $Failed],
		Throw[$Failed, DerivedKey]
	];
	params = ToString /@ {n, r, p};
	{
		alg,
		StringJoin[{"N=", params[[1]], ",", "r=", params[[2]] , ",", "p=", params[[3]]}]
	}
]

derivedKeyParameters[alg:("Argon2i" | "Argon2d" | "Argon2id"), args_Association] := Module[
	{t, m, p, params}
	,
	{t, m, p} = Lookup[args, {"t", "m", "p"}, $Failed];
	If[!FreeQ[{t, m, p}, $Failed],
		Throw[$Failed, DerivedKey]
	];
	params = ToString /@ {t, m, p};
	{ToLowerCase[alg], StringJoin[{"t=", params[[1]], ",", "m=", params[[2]] , ",", "p=", params[[3]]}]}
]

derivedKeyParameters[___] := Throw[$Failed]

algStringToProper["argon2i"] = "Argon2i"
algStringToProper["argon2d"] = "Argon2d"
algStringToProper["argon2id"] = "Argon2id"
algStringToProper[str_String] := str
algStringToProper[___] := Throw[$Failed, DerivedKey]

parseParams2[{a_String, b_String}] := Rule[a, ToExpression[b]]
parseParams2[___] := Throw[$Failed]

parseParams[params_String] := Module[
	{args}
	,
	args = StringSplit[#,"="]& /@ StringSplit[params, ","];
	Association[parseParams2 /@ args]
]

paramsToAssoc[alg:"scrypt", key_, salt_, params_String] := Module[
	{keysize, args, n, r, p}
	,
	keysize = Length[key];
	args = parseParams[params];
	{n, r, p} = Lookup[args, {"N", "r", "p"}, Throw[$Failed, DerivedKey]];
	<|
		"Function" -> alg,
		"DerivedKey" -> key,
		"Salt" -> salt,
		"Parameters" -> <|
			"KeySize" -> keysize,
			"N" -> n, "r" -> r, "p" -> p
		|>
	|>
]

paramsToAssoc[alg:("Argon2i" | "Argon2d" | "Argon2id"), key_, salt_, params_String] := Module[
	{keysize, args, t, m, p}
	,
	keysize = Length[key];
	args = parseParams[params];
	{t, m, p} = Lookup[args, {"t", "m", "p"}, Throw[$Failed, DerivedKey]];
	<|
		"Function" -> alg,
		"DerivedKey" -> key,
		"Salt" -> salt,
		"Parameters" -> <|
			"KeySize" -> keysize,
			"t" -> t, "m" -> m, "p" -> p
		|>	
	|>
]

paramsToAssoc[___] := Throw[$Failed, DerivedKey]

(* Encode byte array as hex string *)
hexEncode[data_ByteArray] := StringJoin[IntegerString[Developer`FromByteArray[data], 16, 2]]
hexEncode[___] := Throw[$Failed]
	
(* Decode hex string into a ByteArray *)
hexDecode[str_String] := Module[
	{res},
	If[Mod[StringLength[str], 2] != 0,
		Throw[$Failed]
	];
	res = Quiet[FromDigits[#, 16] & /@ StringPartition[str, 2]];
	If[!ListQ[res] || Max[res] > 255 || Min[res] < 0,
		Throw[$Failed]
	];
	ByteArray[res]
]
hexDecode[___] := Throw[$Failed]

b64Encode[data_ByteArray] := Module[
	{res},
	res = BaseEncode[data];
	Which[
		StringEndsQ[res, "=="], StringDrop[res, -2],
		StringEndsQ[res, "="], StringDrop[res, -1],
		True, res
	]
]
b64Encode[___] := $Failed

b64Decode[str_String] := Module[
	{res},
	res = Switch[Mod[StringLength[str], 4],
		0, str,
		2, StringJoin[str, "=="],
		3, StringJoin[str, "="],
		_, Throw[$Failed, DerivedKey]
	];
	Check[Quiet[BaseDecode[res]], Throw[$Failed, DerivedKey]]		
]
b64Decode[___] := $Failed

DerivedKey[data_Association][key_] := Lookup[data, key, (* Message *) Missing["NotApplicable"]]

DerivedKey[data_Association]["Properties"] := Keys[data];

DerivedKey[data_Association]["PHCString"] := Module[
	{function, key, salt, params}
	,
	Catch[
		{function, key, salt, params} = Lookup[data, {"Function", "DerivedKey", "Salt", "Parameters"}, Throw[$Failed]];
		{function, params} = derivedKeyParameters[function, params];
		salt = b64Encode[salt];
		key = b64Encode[key];
		StringJoin[{"$", function, "$", params, "$", salt, "$", key}]
	]
]

DerivedKey[str_String] := Module[
	{chunks, alg, salt, key}
	,
	Catch[
		If[StringLength[str] < 1 || StringTake[str, 1] =!= "$",
			Throw[$Failed, DerivedKey]
		]; 
		chunks = StringSplit[StringDrop[str, 1], "$"];
		If[Length[chunks] < 4,
			Throw[$Failed, DerivedKey]
		];
		alg = algStringToProper[chunks[[1]]];
		key = b64Decode[chunks[[-1]]];
		salt = b64Decode[chunks[[-2]]];
		DerivedKey[paramsToAssoc[alg, key, salt, chunks[[-3]]]]
		,
		DerivedKey
		,
		Function[{val,tag}, Message[DerivedKey::invstr, str]; $Failed]
	]
]


PackageScope[byteListQ]
byteListQ[e:{___Integer}] := (Min[e] >= 0) && (Max[e] <= 255) 
byteListQ[___] := False

(*
   If "Salt" is not given, it is generated as randomBytes[32].
*)
GenerateDerivedKey[passwd_, saltIn:Except[_Rule]:Automatic, opts:OptionsPattern[]] := Module[
	{function, salt, params, pw},
	Catch[
		checkOpts[GenerateDerivedKey, Keys[{opts}], {Method}];
		{function, params} = parseGenerateDerivedKeyMethod[OptionValue[Method]];
		salt = Switch[saltIn,
			_String,
				ToCharacterCode[saltIn, "UTF8"],
			{___Integer},
				If[!byteListQ[saltIn],
					Message[GenerateDerivedKey::nbyte, saltIn];
					Throw[$Failed]
				];
				saltIn,
			_ByteArray,
				Normal[saltIn],
			$Failed | Automatic,
				Normal[randomBytes[32]],
			_,
				Message[GenerateDerivedKey::salt, saltIn];
				Throw[$Failed]
		];
		pw = Switch[passwd,
			_String,
				ToCharacterCode[passwd, "UTF8"],
			{___Integer},
				If[!byteListQ[passwd],
					Message[GenerateDerivedKey::nbyte, passwd];
					Throw[$Failed]
				];
				passwd,
			_ByteArray,
				Normal[passwd],
			_,
				Message[GenerateDerivedKey::passwd, passwd];
				Throw[$Failed]
		];
		DerivedKey[doGenerateDerivedKey[function, pw, salt, params]]
	]
]


doVerifyDerivedKey[System`DerivedKey[assoc_], pw_] := Module[
	{function, key, salt, params, newdata, newkey}
	,
	{function, key, salt, params} = Lookup[assoc, {"Function", "DerivedKey", "Salt", "Parameters"}, $Failed];
	If[!FreeQ[{key, salt, params}, $Failed],
		Throw[False]
	];
	If[!ByteArray[salt],
		Throw[False]
	];
	salt = Developer`FromByteArray[salt];
	newdata = doGenerateDerivedKey[function, pw, salt, params];
	newkey = Lookup[newdata, "DerivedKey", $Failed];
	newkey == key	
]

doVerifyDerivedKey[___] := False

VerifyDerivedKey[key_System`DerivedKey, passwd_] := Block[
	{pw}
	,
	Catch[
		pw = Switch[passwd,
			_String,
				ToCharacterCode[passwd, "UTF8"],
			{___Integer},
				passwd,
			_ByteArray,
				Normal[passwd],
			_,
				Message[VerifyDerivedKey::passwd, passwd];
				Throw[False]
		];
		doVerifyDerivedKey[key, pw]
	] 
]


VerifyDerivedKey[str_String, passwd_] := Block[
	{pw, key}
	,
	Catch[
		key = DerivedKey[str];
		If[key === $Failed,
			Throw[False]
		];
		pw = Switch[passwd,
			_String,
				ToCharacterCode[passwd, "UTF8"],
			{___Integer},
				passwd,
			_ByteArray,
				Normal[passwd],
			_,
				Message[VerifyDerivedKey::passwd, passwd];
				Throw[False]
		];
		doVerifyDerivedKey[key, pw]
	]		
]
