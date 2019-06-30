Package["Blockchain`"]

(* Key encoding and hashing functions *)

PackageScope[AddressToPublicKeyHash]
AddressToPublicKeyHash[address_,tag_:BlockchainTransaction] := Module[{hex, pubKeyHash, checksum},
  If[StringLength[address] < 8 || StringCases[address, Except[Alternatives @@ Characters["123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"]]] =!= {},
    BlockchainMessage[tag, "addchk", address]; Throw[$Failed]
  ];
  hex = IntegerString[FromDigits[(Characters[address] /. MapThread[Rule, {Characters["123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"], Range[58]}]) - 1, 58], 16];
  pubKeyHash = StringDrop[hex, -8];
  checksum = StringTake[hex, -8];
  pubKeyHash = Which[
    StringStartsQ[address, "1"], "00" <> pubKeyHash,(*special case for mainnet addresses starting \with 1*)
    StringStartsQ[address, "3"], "0" <> pubKeyHash,(*mainnet addresses*) 
    StringStartsQ[address, "m" | "n" | "2"], pubKeyHash,(*testnet addresses*)
    True, BlockchainMessage[tag, "addsup", address]; Throw[$Failed]
  ];
  If[Not[StringTake[Hash[ByteArray[FromDigits[#, 16] & /@ StringPartition[pubKeyHash, 2]], "SHA256SHA256", "HexString"], 8] === checksum], BlockchainMessage[tag, "addchk", address]; Throw[$Failed]];
  StringDrop[pubKeyHash, 2]
]

PackageScope[WifToPrivateKeyHex]
WifToPrivateKeyHex[wif_] := Module[{hex, privKeyHex, checksum},
  Which[
    StringLength[wif] < 8 || StringCases[wif, Except[Alternatives @@ Characters["123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"]]] =!= {},
      BlockchainMessage[BlockchainTransactionSign, "wifchk", wif]; Throw[$Failed],
    !StringStartsQ[wif, "c" | "9" | "K" | "L" | "5"],
      BlockchainMessage[BlockchainTransactionSign, "wifsup", wif]; Throw[$Failed]
  ];
  hex = IntegerString[FromDigits[(Characters[wif] /. MapThread[Rule, {Characters["123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"], Range[58]}]) - 1, 58], 16];
  Which[
    StringLength[hex] < 8, BlockchainMessage[BlockchainTransactionSign, "wifchk", wif]; Throw[$Failed],
    !StringStartsQ[wif, "c" | "9" | "K" | "L" | "5"], BlockchainMessage[BlockchainTransactionSign, "wifsup", wif]; Throw[$Failed]
  ];
  privKeyHex = StringDrop[hex, -8];
  checksum = StringTake[hex, -8];
  If[Not[StringTake[Hash[ByteArray[FromDigits[#, 16] & /@ StringPartition[privKeyHex, 2]], "SHA256SHA256", "HexString"], 8] === checksum], BlockchainMessage[BlockchainTransactionSign, "wifchk", wif]; Throw[$Failed]];
  If[StringLength[privKeyHex] === 68, 
    {StringDrop[StringDrop[privKeyHex, -2], 2], True},
    {StringDrop[privKeyHex, 2], False}
  ]
]

PackageScope[FromPrivateKeyToWIF]
Options[FromPrivateKeyToWIF] = {"Compressed" -> True, "Network" -> "mainnet"};

FromPrivateKeyToWIF[privatekey_String, OptionsPattern[]] := Block[
  {version = OptionValue["Network"] /. {"mainnet" -> "80", "testnet" -> "ef"}, 
   suffix = OptionValue["Compressed"] /. {True -> "01", False -> ""}, extendedkey, checksum
  },
  extendedkey = version <> privatekey <> suffix;
  checksum = StringTake[Hash[ByteArray[FromDigits[#, 16] & /@ StringPartition[extendedkey, 2]], "SHA256SHA256", "HexString"], 8];
  IntegerString[FromDigits[extendedkey <> checksum, 16], "Base58"]
]

PackageScope[FromPublicKeyToAddress]
Options[FromPublicKeyToAddress] = {"Network" -> "mainnet", "Type" -> "P2PKH"};

FromPublicKeyToAddress[publickey_String, OptionsPattern[]] := Block[
  {version = Switch[OptionValue["Network"],
     "mainnet",
     OptionValue["Type"] /. {"P2PKH" -> "00", "P2SH" -> "05"},
     "testnet",
     OptionValue["Type"] /. {"P2PKH" -> "6f", "P2SH" -> "c4"}
    ],
    hash160, extendedhash, checksum, ones
  },
  hash160 = HashPubKeyP2PKH[publickey];
  extendedhash = version <> hash160;
  checksum = StringTake[Hash[ByteArray[FromDigits[#, 16] & /@ StringPartition[extendedhash, 2]], "SHA256SHA256", "HexString"], 8];
  If[version === "00",
    ones = StringLength[First[StringCases[extendedhash <> checksum, RegularExpression["^(00)+"]]]]/2;
    StringJoin[Table["1", ones]] <> IntegerString[FromDigits[extendedhash <> checksum, 16], "Base58"],
    IntegerString[FromDigits[extendedhash <> checksum, 16], "Base58"]
  ]
]

PackageScope[HashPubKeyP2PKH]
HashPubKeyP2PKH[pubkey_String] := Hash[ByteArray[FromDigits[#, 16] & /@ StringPartition[pubkey, 2]], "RIPEMD160SHA256", "HexString"]

(* Auxiliary functions for ScriptSerialize*)
$opcodes = <|"OP_0" -> 0, "OP_FALSE" -> 0, "OP_PUSHDATA1" -> 76, "OP_PUSHDATA2" -> 77, "OP_PUSHDATA4" -> 78, "OP_1NEGATE" -> 79, "OP_1" -> 78, "OP_TRUE" -> 81, "OP_2" -> 82, "OP_3" -> 83, 
 "OP_4" -> 84, "OP_5" -> 85, "OP_6" -> 86, "OP_7" -> 87, "OP_8" -> 88, "OP_9" -> 89, "OP_10" -> 90, "OP_11" -> 91, "OP_12" -> 92, "OP_13" -> 93, "OP_14" -> 94, "OP_15" -> 95, "OP_16" -> 96, 
 "OP_PUSHDATA(1)" -> 1, "OP_PUSHDATA(2)" -> 2, "OP_PUSHDATA(3)" -> 3, "OP_PUSHDATA(4)" -> 4, "OP_PUSHDATA(5)" -> 5, "OP_PUSHDATA(6)" -> 6, "OP_PUSHDATA(7)" -> 7, "OP_PUSHDATA(8)" -> 8, "OP_PUSHDATA(9)" -> 9, 
 "OP_PUSHDATA(10)" -> 10, "OP_PUSHDATA(11)" -> 11, "OP_PUSHDATA(12)" -> 12, "OP_PUSHDATA(13)" -> 13, "OP_PUSHDATA(14)" -> 14, "OP_PUSHDATA(15)" -> 15, "OP_PUSHDATA(16)" -> 16, "OP_PUSHDATA(17)" -> 17, 
 "OP_PUSHDATA(18)" -> 18, "OP_PUSHDATA(19)" -> 19, "OP_PUSHDATA(20)" -> 20, "OP_PUSHDATA(21)" -> 21, "OP_PUSHDATA(22)" -> 22, "OP_PUSHDATA(23)" -> 23, "OP_PUSHDATA(24)" -> 24, "OP_PUSHDATA(25)" -> 25, 
 "OP_PUSHDATA(26)" -> 26, "OP_PUSHDATA(27)" -> 27, "OP_PUSHDATA(28)" -> 28, "OP_PUSHDATA(29)" -> 29, "OP_PUSHDATA(30)" -> 30, "OP_PUSHDATA(31)" -> 31, "OP_PUSHDATA(32)" -> 32, "OP_PUSHDATA(33)" -> 33, 
 "OP_PUSHDATA(34)" -> 34, "OP_PUSHDATA(35)" -> 35, "OP_PUSHDATA(36)" -> 36, "OP_PUSHDATA(37)" -> 37, "OP_PUSHDATA(38)" -> 38, "OP_PUSHDATA(39)" -> 39, "OP_PUSHDATA(40)" -> 40, "OP_PUSHDATA(41)" -> 41, 
 "OP_PUSHDATA(42)" -> 42, "OP_PUSHDATA(43)" -> 43, "OP_PUSHDATA(44)" -> 44, "OP_PUSHDATA(45)" -> 45, "OP_PUSHDATA(46)" -> 46, "OP_PUSHDATA(47)" -> 47, "OP_PUSHDATA(48)" -> 48, "OP_PUSHDATA(49)" -> 49, 
 "OP_PUSHDATA(50)" -> 50, "OP_PUSHDATA(51)" -> 51, "OP_PUSHDATA(52)" -> 52, "OP_PUSHDATA(53)" -> 53, "OP_PUSHDATA(54)" -> 54, "OP_PUSHDATA(55)" -> 55, "OP_PUSHDATA(56)" -> 56, "OP_PUSHDATA(57)" -> 57, 
 "OP_PUSHDATA(58)" -> 58, "OP_PUSHDATA(59)" -> 59, "OP_PUSHDATA(60)" -> 60, "OP_PUSHDATA(61)" -> 61, "OP_PUSHDATA(62)" -> 62, "OP_PUSHDATA(63)" -> 63, "OP_PUSHDATA(64)" -> 64, "OP_PUSHDATA(65)" -> 65, 
 "OP_PUSHDATA(66)" -> 66, "OP_PUSHDATA(67)" -> 67, "OP_PUSHDATA(68)" -> 68, "OP_PUSHDATA(69)" -> 69, "OP_PUSHDATA(70)" -> 70, "OP_PUSHDATA(71)" -> 71, "OP_PUSHDATA(72)" -> 72, "OP_PUSHDATA(73)" -> 73, 
 "OP_PUSHDATA(74)" -> 74, "OP_PUSHDATA(75)" -> 75, "OP_NOP" -> 97, "OP_IF" -> 99, "OP_NOTIF" -> 100, "OP_ELSE" -> 103, "OP_ENDIF" -> 104, "OP_VERIFY" -> 105, "OP_RETURN" -> 106, 
 "OP_TOALTSTACK" -> 107, "OP_FROMALTSTACK" -> 108, "OP_IFDUP" -> 115, "OP_DEPTH" -> 116, "OP_DROP" -> 117, "OP_DUP" -> 118, "OP_NIP" -> 119, "OP_OVER" -> 120, "OP_PICK" -> 121, 
 "OP_ROLL" -> 122, "OP_ROT" -> 123, "OP_SWAP" -> 124, "OP_TUCK" -> 125, "OP_2DROP" -> 109, "OP_2DUP" -> 110, "OP_3DUP" -> 111, "OP_2OVER" -> 112, "OP_2ROT" -> 113, 
 "OP_2SWAP" -> 114, "OP_INVERT" -> 131, "OP_AND" -> 132, "OP_OR" -> 133, "OP_XOR" -> 134, "OP_EQUAL" -> 135, "OP_EQUALVERIFY" -> 136, "OP_RIPEMD160" -> 166, "OP_SHA1" -> 167, 
 "OP_SHA256" -> 168, "OP_HASH160" -> 169, "OP_HASH256" -> 170, "OP_CODESEPARATOR" -> 171, "OP_CHECKSIG" -> 172, "OP_CHECKSIGVERIFY" -> 173, "OP_CHECKMULTISIG" -> 174, 
 "OP_CHECKMULTISIGVERIFY" -> 175, "OP_CAT" -> 126, "OP_SUBSTR" -> 127, "OP_LEFT" -> 128, "OP_RIGHT" -> 129, "OP_SIZE" -> 130, "OP_1ADD" -> 139, "OP_1SUB" -> 140, "OP_2MUL" -> 141, 
 "OP_2DIV" -> 142, "OP_NEGATE" -> 143, "OP_ABS" -> 144, "OP_NOT" -> 145, "OP_0NOTEQUAL" -> 146, "OP_ADD" -> 147, "OP_SUB" -> 148, "OP_MUL" -> 149, "OP_DIV" -> 150, "OP_MOD" -> 151, 
 "OP_LSHIFT" -> 152, "OP_RSHIFT" -> 153, "OP_BOOLAND" -> 154, "OP_BOOLOR" -> 155, "OP_NUMEQUAL" -> 156, "OP_NUMEQUALVERIFY" -> 157, "OP_NUMNOTEQUAL" -> 158, "OP_LESSTHAN" -> 159, 
 "OP_GREATERTHAN" -> 160, "OP_LESSTHANOREQUAL" -> 161, "OP_GREATERTHANOREQUAL" -> 162, "OP_MIN" -> 163, "OP_MAX" -> 164, "OP_WITHIN" -> 165, "OP_CHECKLOCKTIMEVERIFY" -> 177, 
 "OP_CHECKSEQUENCEVERIFY" -> 178, "OP_PUBKEYHASH" -> 253, "OP_PUBKEY" -> 254, "OP_INVALIDOPCODE" -> 255, "OP_RESERVED" -> 80, "OP_VER" -> 98, "OP_VERIF" -> 101, "OP_VERNOTIF" -> 102, 
 "OP_RESERVED1" -> 137, "OP_RESERVED2" -> 138, "OP_NOP1" -> 176, "OP_NOP4" -> 179, "OP_NOP5" -> 180, "OP_NOP6" -> 181, "OP_NOP7" -> 182, "OP_NOP8" -> 183, "OP_NOP9" -> 184, "OP_NOP10" -> 185|>;

$sighash = {"[ALL]" -> "01", "[NONE]" -> "02", "[SINGLE]" -> "03", "[ALL|ANYONECANPAY]" -> "81", "[NONE|ANYONECANPAY]" -> "82", "[SINGLE|ANYONECANPAY]" -> "83"};

scriptToBytes[opcode_String] := Block[{length},
	If[StringTake[opcode, 2]=="OP",
		{$opcodes[opcode]},
		length = StringLength[opcode]/2;
		Which[
			length <= 75, Prepend[FromDigits[#, 16]& /@ StringPartition[opcode, 2], length],
			76 <= length <= 255, Prepend[Prepend[FromDigits[#, 16]& /@ StringPartition[opcode, 2],length], 76],
			256 <= length <= 65535, Prepend[FromDigits[#, 16]& /@ StringPartition[IntegerString[length, 16, 4] <> opcode, 2], 77],
			65536 <= length <= 4294967295, Prepend[FromDigits[#, 16]& /@ StringPartition[IntegerString[length, 16, 8] <> opcode, 2], 78]
		]
	]
]

ScriptSerialize[script_String | script_List, format_String: "Bytes"] := Block[{bytes, scriptlist = If[Head[script] === String, StringSplit[script], script]},
	bytes = Flatten[scriptToBytes /@ StringReplace[scriptlist, $sighash]];
  	Which[
		format == "Bytes", bytes,
		format == "ByteArray", ByteArray[bytes],
		format == "HexString", StringJoin[IntegerString[bytes, 16, 2]]
   	]
]

(* Auxiliary functions for TransactionSerializeBTC*)

swapEndian[s_] := StringJoin[Reverse[StringPartition[s, 2]]]

toVarInt[n_] := Block[{nHex},
  	nHex = IntegerString[n, 16];
  	Which[
   		n <= 252,
   		IntegerString[n, 16, 2],
   
   		3 <= StringLength[nHex] <= 4,
   		"fd" <> swapEndian[IntegerString[n, 16, 4]],
   
   		4 < StringLength[nHex] <= 8,
   		"fe" <> swapEndian[IntegerString[n, 16, 8]],
   
   		8 < StringLength[nHex] <= 16,
   		"ff" <> swapEndian[IntegerString[n, 16, 16]]
   	]
]

fromVarInt[varInt_] := If[StringLength[varInt] <= 2,
  	FromDigits[varInt, 16],
  	FromDigits[swapEndian[StringDrop[varInt, 2]], 16]
]

serializeInput[inputAssoc_] := Block[{txid, vout, scriptSize, script, sequence, serializedScript, scriptByteArrayQ = MemberQ[Keys[inputAssoc], "ScriptByteArray"]},
  	serializedScript = If[! scriptByteArrayQ, ScriptSerialize[Lookup[inputAssoc, "ScriptString", ""], "HexString"]];
  	txid = swapEndian[Lookup[inputAssoc, "TransactionID"]];
  	vout = swapEndian[IntegerString[Lookup[inputAssoc, "Index"], 16, 8]];
  	scriptSize = toVarInt@If[scriptByteArrayQ, Length[Lookup[inputAssoc, "ScriptByteArray"]], StringLength[serializedScript]/2];
  	script = If[scriptByteArrayQ, StringJoin[IntegerString[Normal[Lookup[inputAssoc, "ScriptByteArray"]], 16, 2]], serializedScript];
  	sequence = swapEndian[IntegerString[Lookup[inputAssoc, "SequenceNumber", 4294967295], 16, 8]];
  	StringJoin[{txid, vout, scriptSize, script, sequence}]
]

serializeOutput[outputAssoc_] := Block[{value, scriptSize, script, serializedScript, amount, scriptByteArrayQ = MemberQ[Keys[outputAssoc], "ScriptByteArray"]},
 	amount = With[{a = Lookup[outputAssoc, "Amount"]}, If[Head[a] === Quantity, QuantityMagnitude[a], a]];
  	serializedScript = If[! scriptByteArrayQ, ScriptSerialize[Lookup[outputAssoc, "ScriptString", ""], "HexString"]];
  	value = swapEndian[IntegerString[Round[amount*10^8], 16, 16]];
  	scriptSize = toVarInt@If[scriptByteArrayQ, Length[Lookup[outputAssoc, "ScriptByteArray"]], StringLength[serializedScript]/2];
  	script = If[scriptByteArrayQ, StringJoin[IntegerString[Normal[Lookup[outputAssoc, "ScriptByteArray"]], 16,2]], serializedScript];
  	StringJoin[{value, scriptSize, script}]
]

PackageScope[TransactionSerializeBTC]
TransactionSerializeBTC[txAssoc_] := Block[{version, inputCount, inputs, outputCount, outputs, locktime},
	version = swapEndian[IntegerString[Lookup[txAssoc, "Version", 1], 16, 8]];
	inputCount = toVarInt[Length[txAssoc["Inputs"]]];
  	inputs = StringJoin[serializeInput /@ txAssoc["Inputs"]];
	outputCount = toVarInt[Length[txAssoc["Outputs"]]];
	outputs = StringJoin[serializeOutput /@ txAssoc["Outputs"]];
	locktime = swapEndian[IntegerString[Lookup[txAssoc, "LockTime", 0], 16, 8]];
  	StringJoin[{version, inputCount, inputs, outputCount, outputs, locktime}]
]

(* Message and digest functions for signature generation *)
PackageScope[MessageFromTransactionBTC]
MessageFromTransactionBTC[tx_Association, InputIndex_Integer, network_String, SigHash_String] := Module[
  	{input, sourceTX, UTXOScript, copyTX, sighash = StringSplit[SigHash, "|"] /. {{a_, "ANYONECANPAY"} :> {a, True}, {a_} :> {a, False}}, pubKeyHash},
    input = tx["Inputs"][[InputIndex + 1]];
    If[(First[sighash] === "SINGLE") && ((InputIndex + 1) > Length[tx["Outputs"]]), BlockchainMessage[BlockchainTransactionSign, "shsing"]; Throw[$Failed]];
  	sourceTX = BlockchainTransactionDataBTC[input["TransactionID"], network];
  	UTXOScript = sourceTX["Outputs"][[input["Index"] + 1]]["ScriptString"];
  	pubKeyHash = StringCases[UTXOScript, "OP_HASH160 " ~~ pkh__ ~~ " OP_EQUALVERIFY" :> pkh];
    copyTX = tx;
  	(*Inputs*)
  	If[Last[sighash],
   		copyTX["Inputs"] = {Append[KeyDrop[input, "ScriptByteArray"], "ScriptString" -> UTXOScript]},
   		copyTX["Inputs"] = KeyDrop[copyTX["Inputs"], {"ScriptByteArray", "ScriptString"}];
   		copyTX["Inputs"] = MapAt[Append[#, "ScriptString" -> UTXOScript] &, copyTX["Inputs"], InputIndex + 1];
   		If[MatchQ[First[sighash], "NONE" | "SINGLE"],
    		copyTX["Inputs"] = MapAt[Append[#, "SequenceNumber" -> 0] &, copyTX["Inputs"], DeleteCases[Partition[Range[Length[copyTX["Inputs"]]], 1], {InputIndex + 1}]]
    	]
   	];
  	(*Outputs*)
  	Switch[First[sighash],
		"NONE", copyTX["Outputs"] = {},
		"SINGLE", copyTX["Outputs"] = Append[ConstantArray[<|"Amount" -> 184467440737.09551615|>, InputIndex], copyTX["Outputs"][[InputIndex + 1]]]
   	];
  	(*Append SigHash bytes*)
  	sighash = SigHash /. {"ALL" -> "01000000", "NONE" -> "02000000", "SINGLE" -> "03000000", "ALL|ANYONECANPAY" -> "81000000", "NONE|ANYONECANPAY" -> "82000000", "SINGLE|ANYONECANPAY" -> "83000000"};
  	
	{TransactionSerializeBTC[copyTX] <> sighash,pubKeyHash}
]

PackageScope[MessageDigestBTC]
MessageDigestBTC[message_String] := Hash[ByteArray[(FromDigits[#, 16] & /@ StringPartition[message, 2])], "SHA256SHA256", "HexString"]