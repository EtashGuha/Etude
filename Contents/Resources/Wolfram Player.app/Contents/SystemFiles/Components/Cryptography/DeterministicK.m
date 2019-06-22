(* ::Package:: *)

Package["Cryptography`"]


(* 
Deterministic generation of Digital Signatures as specified in RFC 6979
https://tools.ietf.org/html/rfc6979
*)

PackageScope["DeterministicK"]

DeterministicK[privKey_List, digest_List, hash_String, qmod_Integer]:= Module[
		{h1, v, k, outsize, T={}, detk},
		outsize = $hmacHashes[hash][[1]]/8;
		h1 = bytes2octets[digest, qmod];
        v = ConstantArray[1, outsize];
        k = ConstantArray[0, outsize];
        k = HMAChash[Join[v, {0}, privKey, h1], k, hash];
        v = HMAChash[v, k, hash];
        k = HMAChash[Join[v, {1}, privKey, h1], k, hash];
        v = HMAChash[HMAChash[v, k, hash], k, hash];
        While[8*Length[T] < BitLength[qmod],
			T = Flatten@AppendTo[T, v]];
        detk = bytes2int[T, qmod];
        While[detk > qmod - 1,
			k = HMAChash[Join[v, {0}], k, hash];
			v = HMAChash[HMAChash[v, k, hash], k, hash];
			While[8*Length[T] < BitLength[qmod],
				T = Flatten@AppendTo[T, v]];
			detk = bytes2int[T, qmod]
         ];
        detk
        ]


(*
"hash" \[Rule] {output size in bits, block size in bits}
 TODO: import from the library that provides them 
 *)
$hmacHashes = <|
	"MD2" -> {128,128},"MD4" -> {128,512},"MD5" -> {128,512},
	"SHA" -> {160,512},"SHA1" -> {160,512},"SHA256" -> {256,512},
	"SHA384" -> {384,1024},"SHA512" -> {512,1024},"RIPEMD160" -> {160,512},
	"RIPEMD160SHA256" -> {160,512},"SHA256SHA256" -> {256,512},"SHA3-224" -> {224,1152},
	"SHA3-256" -> {256,1088},"SHA3-384" -> {384,832},"SHA3-512" -> {512,576},
	"Keccak224" -> {224,1152},"Keccak256" -> {256,1088},"Keccak384" -> {384,832},
	"Keccak512" -> {512,576}
	|>; 

PackageScope["HMAChash"]

HMAChash[invec_List, inkey_List, hash_String] := Module[
		{key0, blocksize, ipad, opad, ipadkey, opadkey, temp, result},
		
		blocksize = $hmacHashes[hash][[2]]/8;
		key0 = PadRight[inkey, blocksize];
		ipad = Table[54, blocksize];
		opad = Table[92, blocksize];  
		ipadkey = BitXor[key0, ipad];
		opadkey = BitXor[key0, opad];
		temp = Hash[ByteArray[Join[ipadkey, invec]], hash, "ByteArray"];
		result = Hash[ByteArray[Join[opadkey, Developer`FromByteArray[temp]]], hash, "ByteArray"];
		Developer`FromByteArray[result]
        ];


(*helper functions from RFC 6979*)
bytes2int[bytes_List, qmod_Integer]:= Block[
	{b},
	b = FromDigits[bytes, 256];
	If[8*Length[bytes] > BitLength[qmod],
		b = BitShiftRight[b, 8*Length[bytes] - BitLength[qmod]]];
	b];
	
bytes2octets[bytes_List, qmod_Integer]:= Block[
	{z1, z2},
	z1 = bytes2int[bytes, qmod];
	z2 = Mod[z1, qmod];
	int2octets[z2, qmod]
	];
	
int2octets[x_Integer, qmod_Integer]:=
	IntegerDigits[x, 256, Ceiling[BitLength[qmod]/8]];
