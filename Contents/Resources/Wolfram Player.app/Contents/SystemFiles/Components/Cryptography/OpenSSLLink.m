(* ::Package:: *)

Package["Cryptography`"]
(*
Encrypt::libopenssl = "Couldn't load OpenSSL library."
*)
SetAttributes[checkLibraryLoad, HoldAll];
checkedLibraryFunctionLoad[args___] := 
	Replace[
		Quiet[LibraryFunctionLoad[args]], 
		$Failed :> (Message[Encrypt::libopenssl]; Throw[$Failed])
	];

SetAttributes[checkedLibraryLoad, HoldAll];
checkedLibraryLoad[args___] := 
	Replace[
		Quiet[LibraryLoad[args]], 
		$Failed :> (Message[Encrypt::libopenssl]; Throw[$Failed])
	];

$systemLibraries = FileNameJoin[{$InstallationDirectory, "SystemFiles", "Libraries", $SystemID}];

libraryPath := libraryPath = (
	Which[
		$OperatingSystem === "Windows",
			checkedLibraryLoad[FileNameJoin[{$systemLibraries, "libeay32"}]];,
		$OperatingSystem === "Unix" && $SystemID =!= "Linux-ARM",
			checkedLibraryLoad[FileNameJoin[{$systemLibraries, "libcrypto.so.1.0.0"}]];
		 ];
	If[$OperatingSystem === "iOS", FindLibrary["OpenSSLLink"], ToFileName[ PacletManager`PacletResource["Cryptography", "Libraries"], "OpenSSLLink" <> ToString[$SystemWordLength] ]]
);

(*
    scrypt KDF
 *)
 
llscrypt := llscrypt = checkedLibraryFunctionLoad[libraryPath, "scrypt", 
	{{Integer, 1}, {Integer, 1}, Integer, Integer, Integer, Integer}, {Integer, 1}];
  
PackageScope["scrypt"]

scrypt[password:{___Integer}, salt:{___Integer}, outputLength_Integer, N_Integer:8192, r_Integer:8, p_Integer:16] :=
	Replace[
		llscrypt[password, salt, N, r, p, outputLength],
		_LibraryFunctionError :> Throw[$Failed]
	]


(*
   Argon2 KDF
 *)
 
 PackageExport[$Argon2Types]
 
 $Argon2Types = <|
     "Argon2d" -> 0,
     "Argon2i" -> 1,
     "Argon2id" -> 2
 |>

llargon2 := llargon2 = checkedLibraryFunctionLoad[libraryPath, "argon2",
	{{Integer, 1}, {Integer, 1}, Integer, Integer, Integer, Integer, Integer}, {Integer, 1}];
  
PackageScope["argon2"]

(* TODO: tweak default parameters *)
argon2[password:{___Integer}, salt:{___Integer}, outputLength_Integer, t_Integer:2, m_Integer:16384, p_Integer:2, typeidx_Integer:3] :=
	Replace[
		llargon2[password, salt, t, m, p, outputLength, typeidx],
		_LibraryFunctionError :> Throw[$Failed]
	]



PackageScope["randomBytes"]

llrandomBytes := llrandomBytes = checkedLibraryFunctionLoad[libraryPath, "randomBytes", {Integer}, {Integer, 1}];

randomBytes[length_Integer] := ByteArray[llrandomBytes[length]]

PackageScope["llgenerateRSAKey"]
	
llgenerateRSAKey := llgenerateRSAKey = checkedLibraryFunctionLoad[libraryPath, "generateRSAKey", {Integer, Integer}, {Integer, 1}];

PackageScope["llrsaPublic"]

llrsaPublic := llrsaPublic = 
  checkedLibraryFunctionLoad[libraryPath, 
   "rsaPublic", {{Integer, 1}, {Integer, 1}, {Integer, 1}, Integer, 
    Integer}, {Integer, 1}];


PackageScope["llrsaPrivate"]

llrsaPrivate := llrsaPrivate =  
  checkedLibraryFunctionLoad[libraryPath, 
   "rsaPrivate", {{Integer, 1}, {Integer, 1}, {Integer, 1}, {Integer, 
     1}, Integer, Integer}, {Integer, 1}];


PackageExport["$AsymmetricCiphers"]

$AsymmetricCiphers = {"RSA"};


PackageScope["llencryptSym"]
llencryptSym := llencryptSym = 
  checkedLibraryFunctionLoad[libraryPath, 
   "encryptSym", {Integer, {Integer, 1}, {Integer, 1}, {Integer, 1}, 
    Integer}, {Integer, 1}];


PackageExport["$SymmetricCiphers"]

$SymmetricCiphers = {"Blowfish", "CAST5", "DES", "RC4", "IDEA", "AES128", "AES192", "AES256"};


PackageExport["$BlockModes"]

$BlockModes = {"ECB", "CBC", "CFB", "OFB", "CTR"};


PackageExport["$SymmetricCipherNumbering"]

$SymmetricCipherNumbering = 
<|
 "Blowfish" -> <|"ECB" -> 0, "CBC" -> 1, "CFB" -> 2, "OFB" -> 3|>, 
 "CAST5"    -> <|"ECB" -> 4, "CBC" -> 5, "CFB" -> 6, "OFB" -> 7|>, 
 "DES"      -> <|"ECB" -> 8, "CBC" -> 9, "CFB" -> 10, "OFB" -> 11|>, 
 "RC4"      -> <|None -> 12, "None" -> 12|>, 
 "IDEA"     -> <|"ECB" -> 13, "CBC" -> 14, "CFB" -> 15, "OFB" -> 16|>, 
 "AES128"   -> <|"ECB" -> 17, "CBC" -> 18, "CFB" -> 19, "OFB" -> 20, "CTR" -> 29|>, 
 "AES192"   -> <|"ECB" -> 21, "CBC" -> 22, "CFB" -> 23, "OFB" -> 24, "CTR" -> 30|>, 
 "AES256"   -> <|"ECB" -> 25, "CBC" -> 26, "CFB" -> 27, "OFB" -> 28, "CTR" -> 31|>
|>;


PackageExport["CipherRequiresIVQ"]

CipherRequiresIVQ["RC4", _] = False;
CipherRequiresIVQ[_, "ECB"] = False;
CipherRequiresIVQ[_, _] = True;


PackageExport["$VariableSizeSymmetricKeys"]

$VariableSizeSymmetricKeys = {"Blowfish", "RC4", "CAST5"};


PackageExport["ValidKeySizeQ"]

ValidKeySizeQ[size_Integer, cipher_String] := 
	If[MemberQ[$VariableSizeSymmetricKeys, cipher],
		Mod[size, 8] == 0,
		$SymmetricKeySizes[cipher] / 8 === size
	];


PackageExport["ValidBlockModeQ"]

ValidBlockModeQ[block_, cipher_] := Block[
	{data, res},
	data = Lookup[$SymmetricCipherNumbering, cipher, $Failed];
	(data =!= $Failed) && (Lookup[data, block, $Failed] =!= $Failed)
]

ValidBlockModeQ[___] := False


PackageExport["$SymmetricKeySizes"]

$SymmetricKeySizes = 
<|"Blowfish" -> 256, "CAST5" -> 256, "DES" -> 64, "RC4" -> 256, 
 "IDEA" -> 128, "AES128" -> 128, "AES192" -> 192, "AES256" -> 256|>


PackageExport["$SymmetricIVSizes"]

$SymmetricIVSizes =
<|"Blowfish" -> 64, "CAST5" -> 64, "DES" -> 64, "IDEA" -> 64, 
 "AES128" -> 128, "AES192" -> 128, "AES256" -> 128, "RC4" -> 0|>
 

PackageExport["$RSAPaddingModeNumbering"]

$RSAPaddingModeNumbering = <|
	"PKCS1" -> 1,
	"SSLV23" -> 2,
	None -> 3,
	"None" -> 3,
	"OAEP" -> 4
|>;

PackageExport["$RSAPaddingModes"]

$RSAPaddingModes = Keys[$RSAPaddingModeNumbering];

(*----------------------------------------------------------------------------*)
(* File Hashing Functions *)

fHash := fHash = 
	checkedLibraryFunctionLoad[libraryPath, "file_hash", {"UTF8String", Integer}, "UTF8String"];

PackageExport["fastFileHash"]

$HashMethods = <|
	"MD5" -> 1,
	"SHA" -> 2,
	"SHA256" -> 3,
	"SHA384" -> 4,
	"SHA512" -> 5
|>;

fastFileHash[file_String, type_String] := Module[
	{hashCode, out},
	hashCode = Lookup[$HashMethods, type, Return[$Failed]];
	out = fHash[file, hashCode];
	If[Head[out] === LibraryFunctionError, Return[$Failed]];
	out
]

fastFileHash[File[file_String], type_String] := fastFileHash[file, type]

(*---------------------------------------------------------------------------*)

PackageScope["sigVerify"]

sigVerify := sigVerify =
	checkedLibraryFunctionLoad[libraryPath, "ecdsa_verify_signature_secp256k1",
		{{Integer, 1}, {Integer, 1}, {Integer, 1}}, "Boolean"]
		
PackageScope["sigVerifyRS"]

sigVerifyRS := sigVerifyRS =
	checkedLibraryFunctionLoad[libraryPath, "ecdsa_verify_signature_secp256k1_RS",
		{{Integer, 1}, {Integer, 1}, {Integer, 1}, {Integer, 1}}, "Boolean"]


PackageExport["$EllipticCurvesParameters"]
(* add here other curves parameters, ref: http://www.secg.org/sec2-v2.pdf *)

$EllipticCurvesParameters = 
<|
	"secp256k1" -> <|
		"KeySize" -> 256,
		(*prime field size*)
		"p" -> 115792089237316195423570985008687907853269984665640564039457584007908834671663,
		"a" -> 0, 
		"b" -> 7, 
		(*group generator point*)
		"G" -> {
				55066263022277343669578718895168534326250603453777594175500187360389116729240, 
				32670510020758816978083085130507043184471273380659243275938904335757337482424
			},
		(*group order*)
		"n" -> 115792089237316195423570985008687907852837564279074904382605163141518161494337
		|>
|>;


PackageScope["$CurrenciesEC"]

$CurrenciesEC = <|
	"Bitcoin" -> "secp256k1",
	"BTC" -> "secp256k1",
	"Ethereum" -> "secp256k1",
	"ETH" -> "secp256k1"
	|>;


PackageExport["$Cryptocurrencies"]

$Cryptocurrencies = Keys[$CurrenciesEC];


PackageExport["$EllipticCurves"]

$EllipticCurves = Keys[$EllipticCurvesParameters];
