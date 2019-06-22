Package["Blockchain`"]

(*Messages*)
PackageScope[Blockchain]
(*General - Multichain*)
Blockchain::errcon = "Error connecting to server (StatusCode: `1`): `2`";
Blockchain::invinp = "Invalid input.";
Blockchain::invopt = "Invalid set of options.";
Blockchain::invnet = "Not supported for `1` BlockchainBase.";
Blockchain::bbase = "Invalid BlockchainBase specification `1`.";
Blockchain::errcon0 = "Error connecting to server";
Blockchain::notcon = "User not connected.";
Blockchain::notos = "To use this function, you must accept the terms of service.";
Blockchain::sizlim = "Data size exceeds maximum permitted.";
Blockchain::invcb = "Not supported for `1` CloudBase.";
Blockchain::invprm = "`1` is not a valid `2`.";
Blockchain::nobase = "\"BlockchainBase\" needs to be specified in the transaction.";
Blockchain::argn = "`1` called with `2` arguments; `3` arguments are expected.";
Blockchain::argn1 = "`1` called with `2` arguments; 1 argument is expected.";
Blockchain::invarg = "`1` is expected at position `2`.";
Blockchain::invtime = "Input can't contain both \"TimeInterval\" and \"BlockNumberInterval\" elements.";
Blockchain::curvpri = "Invalid \"EllipticCurve\" of private key object.";
Blockchain::curvpub = "Invalid \"EllipticCurve\" of public key object.";
Blockchain::invint = "The second element of \"`1`\" property must be greater or equal than the first element.";
Blockchain::invelem = "Invalid element \"`1`\" in `2` association.";
Blockchain::invelist = "Invalid element \"`1`\" in one of the associations of \"`2`\".";
Blockchain::misselem = "Missing element \"`1`\" in `2` association.";
Blockchain::invval = "Invalid value `1` for \"`2`\" element. `3` is expected.";
Blockchain::invassoc = "Invalid association `1` for \"`2`\" element.";
Blockchain::invtyp = "Invalid Typed expression `1` for \"`2`\" element.";
Blockchain::invinas = "Invalid input association.";
(*Bitcoin*)
Blockchain::addsup = "`1` is not a supported \"Address\".";
Blockchain::addchk = "`1` is not a valid \"Address\". Check your address for typos or missing characters.";
Blockchain::invpvk = "The output with \"Index\" `1` of source transaction `2` is not associated to any of the provided private keys.";
Blockchain::wifsup = "`1` is not a supported private key.";
Blockchain::wifchk = "`1` is not a valid private key. Check your private key for typos or missing characters.";
Blockchain::nopriv = "A private key must be provided to obtain the Wallet Import Format.";
Blockchain::invsti = "The output with \"Index\" `1` of source transaction `2` does not exist.";
Blockchain::insfam = "The total input amount is insufficient for the total output amount.";
Blockchain::shsing = "No matching output index was found when signing input with \"Single\" or \"SingleAnyoneCanPay\" as \"SignatureHash\".";
Blockchain::txnadd = "The transaction `1` has not been added to the blockchain yet.";
Blockchain::txnnet = "The transaction `1` was not found for `2` BlockchainBase.";
Blockchain::btcerr = "`1`.";
Blockchain::invtxa = "\"`1`\" needs to be specified in the transaction as a list of associations with the required elements.";
(*Ethereum*)
Blockchain::smerr = "Calling contract returned following error: `1`";
Blockchain::ipfsadd = "Missing \"Address\" element in IPFSObject.";
Blockchain::invtkn = "Invalid token specification `1`.";
Blockchain::dupfil = "Giving priority to \"Addresses\" element over \"HolderAddresses\".";
Blockchain::nofile = "The input file `1` doesn't exist.";
Blockchain::etherr = "`1`.";
Blockchain::ethadd = "Invalid Ethereum address `1`.";
Blockchain::abienc = "Could not encode input `1` using `2` as data type.";
Blockchain::abidec = "Could not decode output using `1` as data type.";
Blockchain::abityp = "Invalid Solidity data type `1`.";
Blockchain::abilen = "Length of inputs must be the same as input data types.";
Blockchain::abiinp = "Inputs are expected in `1` association.";
Blockchain::solerr = "Solidity compiler returned following error: `1`.";
Blockchain::solwar = "Solidity compiler returned following warning: `1`.";
Blockchain::solcon = "Contract's constructor doesn't allow initial funds.";
Blockchain::solnop = "Contract's constructor doesn't allow initial parameters.";
Blockchain::solimp = "When importing a Solidity contract, please define its filepath as \"./filename.sol\". `1` doesn't comply.";
Blockchain::solpar = "Contract's contructor expects `1` parameters. \"Parameters\" has `2` elements.";
Blockchain::solfil = "Invalid Solidity contract `1`.";

(* Loading Cryptography paclet*)
<< Cryptography`;

(* Main Blockchain functions *)

Options[BlockchainData] = {
    BlockchainBase :> $BlockchainBase
}

BlockchainData[fields : (_String | {___String} | All) : {}, opt : OptionsPattern[BlockchainData]] := Catch[Module[
	{blockchain = getBlockchainBase[BlockchainData, OptionValue[BlockchainBase]], result},
	
	result = Switch[First@blockchain,
		"Bitcoin", BlockchainDataBTC[Last@blockchain],
		"Ethereum", BlockchainDataETH[Last@blockchain],
		"Multichain", BlockchainDataWL[],
		_, BlockchainMessage[BlockchainData, "invnet", blockchain]; Throw[$Failed]
	];
	If[SameQ[fields, {}] || SameQ[result, $Failed] || SameQ[fields, All],
    	result,        
    	Lookup[result, fields, Missing["NotAvailable"]]
    ]
]]

BlockchainData[_, opt : OptionsPattern[BlockchainData]] := (BlockchainMessage[BlockchainData, "invarg", "A blockchain property", 1]; $Failed)

BlockchainData[arg___, opt : OptionsPattern[BlockchainData]] := (BlockchainMessage[BlockchainData, "argn", "BlockchainData", Length[{arg}], "0 or 1"]; $Failed)

Options[BlockchainBlockData] = {
    BlockchainBase :> $BlockchainBase
}

BlockchainBlockData[block:First | Last | _Integer | _String, fields : (_String | {___String} | All) : {}, opt : OptionsPattern[BlockchainBlockData]] := Catch[Module[
	{blockchain = getBlockchainBase[BlockchainBlockData, OptionValue[BlockchainBase]], result},
	
	result = Switch[First@blockchain,
		"Bitcoin", BlockchainBlockDataBTC[block, Last@blockchain],
		"Ethereum", BlockchainBlockDataETH[block, Last@blockchain],
		"Multichain", BlockchainBlockDataWL[block],
		_, BlockchainMessage[BlockchainBlockData, "invnet", blockchain]; Throw[$Failed]
	];
	If[SameQ[fields, {}] || SameQ[result, $Failed] || SameQ[fields, All],
    	result,        
    	Lookup[result, fields, Missing["NotAvailable"]]
    ]
]]

BlockchainBlockData[blocks :  {(_String | _Integer) ..}, fields : (_String | {___String}) : {}, opt : OptionsPattern[BlockchainBlockData]] := Module[{},
	BlockchainBlockData[#, fields, opt] & /@ blocks
]

BlockchainBlockData[arg1_, arg2 : RepeatedNull[_,1], opt : OptionsPattern[BlockchainBlockData]] := If[MatchQ[{arg2}, ({_String} | {{___String}} | {})],
	BlockchainMessage[BlockchainBlockData, "invarg", "A block hash or height given by an integer", 1]; $Failed,
	BlockchainMessage[BlockchainBlockData, "invarg", "A block property", 2]; $Failed
]

BlockchainBlockData[arg___, opt : OptionsPattern[BlockchainBlockData]] := (BlockchainMessage[BlockchainBlockData, "argn", "BlockchainBlockData", Length[{arg}], "1 or 2"]; $Failed)

Options[BlockchainTransactionData] = {
    BlockchainBase :> $BlockchainBase
}

BlockchainTransactionData[txid_String, fields : (_String | {___String} | All) : {}, opt : OptionsPattern[BlockchainTransactionData]] := Catch[Module[
	{blockchain = getBlockchainBase[BlockchainTransactionData, OptionValue[BlockchainBase]], result},

	result = Switch[First@blockchain,
		"Bitcoin", BlockchainTransactionDataBTC[txid, Last@blockchain],
		"Ethereum", BlockchainTransactionDataETH[txid, Last@blockchain],
		"Multichain", BlockchainTransactionDataWL[txid],
		_, BlockchainMessage[BlockchainTransactionData, "invnet", blockchain]; Throw[$Failed]
	];
	If[SameQ[fields, {}] || SameQ[result, $Failed] || SameQ[fields, All],
    	result,        
    	Lookup[result, fields, Missing["NotAvailable"]]
    ]
]]

BlockchainTransactionData[txids:  {__String}, fields : (_String | {___String}) : {}, opt : OptionsPattern[BlockchainTransactionData]] := Module[{},
	BlockchainTransactionData[#, fields, opt] & /@ txids
]

BlockchainTransactionData[arg1_, arg2 : RepeatedNull[_,1], opt : OptionsPattern[BlockchainTransactionData]] := If[MatchQ[{arg2}, ({_String} | {{___String}} | {})],
	BlockchainMessage[BlockchainTransactionData, "invarg", "A transaction ID", 1]; $Failed,
	BlockchainMessage[BlockchainTransactionData, "invarg", "A transaction property", 2]; $Failed
]

BlockchainTransactionData[arg___, opt : OptionsPattern[BlockchainTransactionData]] := (BlockchainMessage[BlockchainTransactionData, "argn", "BlockchainTransactionData", Length[{arg}], "1 or 2"]; $Failed)

Options[BlockchainKeyEncode] = {
    BlockchainBase :> $BlockchainBase
}

BlockchainKeyEncode[key_, format: "Address" | "WIF", opt : OptionsPattern[BlockchainKeyEncode]] := Catch[Module[
	{blockchain = getBlockchainBase[BlockchainKeyEncode, OptionValue[BlockchainBase]], result, newKey},

	If[StringQ[key], newKey = ToLowerCase[StringDelete[key, "0x"]], newKey = key];

	If[!MatchQ[newKey, _String?(hexStringQ[#] && (StringLength[#] === 64 || StringLength[#] === 66 || StringLength[#] === 130) &) | _PublicKey | _PrivateKey],
		BlockchainMessage[BlockchainKeyEncode, "invprm", newKey, "HexString, PublicKey object or PrivateKey object"]; Throw[$Failed]
	];

	result = Switch[First@blockchain,
		"Bitcoin", BlockchainKeyEncodeBTC[newKey, format, Last@blockchain],
		"Ethereum", If[format === "WIF",
        	BlockchainMessage[BlockchainKeyEncode, "invnet", "Ethereum"]; Throw[$Failed],
        	BlockchainKeyEncodeETH[newKey]
        ],
       	_, BlockchainMessage[BlockchainKeyEncode, "invnet", blockchain]; Throw[$Failed]
   	];
       
   	result

]]

BlockchainKeyEncode[arg1_, arg2_, opt : OptionsPattern[BlockchainKeyEncode]] := (BlockchainMessage[BlockchainKeyEncode, "invarg", "\"Address\" or \"WIF\"", 2]; $Failed)

BlockchainKeyEncode[arg___, opt : OptionsPattern[BlockchainKeyEncode]] := (BlockchainMessage[BlockchainKeyEncode, "argn", "BlockchainKeyEncode", Length[{arg}], "2"]; $Failed)

(*****************  Blockchain WRITE functions  *********************)


BlockchainTransactionSign[obj_BlockchainTransaction, privKey: _String | _PrivateKey | List[__?(StringQ[#] || MatchQ[#, _PrivateKey] &)]] := Catch[Module[
	{blockchain = getBlockchainBase[BlockchainTransactionSign, obj["BlockchainBase"]], result},

	(*Needs error handling for input*)

	result = Switch[First@blockchain,
		"Bitcoin", BlockchainTransactionSignBTC[obj, privKey, Last@blockchain],
		"Ethereum", BlockchainTransactionSignETH[obj, privKey],
		_, BlockchainMessage[BlockchainTransactionSign, "invnet", blockchain]; Throw[$Failed]
	];

    result 

]]

BlockchainTransactionSign[a_, b_] := Which[
	!MatchQ[a, _BlockchainTransaction], (BlockchainMessage[BlockchainTransactionSign, "invarg", "A BlockchainTransaction object", 1]; $Failed),
	!MatchQ[b, _String | _PrivateKey | List[__?(StringQ[#] || MatchQ[#, _PrivateKey] &)]], (BlockchainMessage[BlockchainTransactionSign, "invarg", "A private key", 2]; $Failed)
]

BlockchainTransactionSign[arg___] := (BlockchainMessage[BlockchainTransactionSign, "argn", "BlockchainTransactionSign", Length[{arg}], "2"]; $Failed)

BlockchainTransactionSubmit[obj_BlockchainTransaction] := Catch[Module[
	{blockchain = getBlockchainBase[BlockchainTransactionSubmit, obj["BlockchainBase"]], result},

	(*Needs error handling for input*)

	result = Switch[First@blockchain,
		"Bitcoin", BlockchainTransactionSendBTC[obj, Last@blockchain],
		"Ethereum", BlockchainTransactionSendETH[obj, Last@blockchain],
		_, BlockchainMessage[BlockchainTransactionSubmit, "invnet", blockchain]; Throw[$Failed]
	];

    result 

]]

BlockchainTransactionSubmit[a_?(!MatchQ[#, _BlockchainTransaction]&)] := (BlockchainMessage[BlockchainTransactionSubmit, "invarg", "A BlockchainTransaction object", 1]; $Failed)

BlockchainTransactionSubmit[arg___] := (BlockchainMessage[BlockchainTransactionSubmit, "argn1", "BlockchainTransactionSubmit", Length[{arg}]]; $Failed)

(*****************  BlockchainAddressData  *********************)

Options[BlockchainAddressData] = {
    BlockchainBase :> $BlockchainBase, 
    MaxItems -> 20
}

BlockchainAddressData[account: (_String | _Association | Rule[_, (_String | _Association)]), fields: (__String | {__String..} | All): {}, opt : OptionsPattern[BlockchainAddressData]] := Catch[Module[
	{blockchain, account0, result},

	If[Head[account] === Rule,
		blockchain = getBlockchainBase[BlockchainAddressData, Keys[account]];
		account0 = Values[account];
		,
		blockchain = getBlockchainBase[BlockchainAddressData, OptionValue[BlockchainBase]];
		account0 = account;
	];

	Switch[First@blockchain,
		(* "Bitcoin", BlockchainAddressDataBTC[account0, fields, Last@blockchain, ToString@OptionValue[MaxItems]], *)
		"Ethereum", BlockchainAddressDataETH[account0, fields, Last@blockchain, ToString@OptionValue[MaxItems]],
		_, BlockchainMessage[BlockchainAddressData, "invnet", blockchain]; Throw[$Failed]
	]

	(*If[SameQ[fields, {}] || SameQ[result, $Failed] || SameQ[fields, All],
    	result,        
    	Lookup[result, fields, Missing["NotAvailable"]]
    ]*)
		
]]

BlockchainAddressData[arg1_, arg2 : RepeatedNull[_,1], opt : OptionsPattern[BlockchainAddressData]] := If[MatchQ[{arg2}, ({_String} | {{___String}} | {})],
	BlockchainMessage[BlockchainAddressData, "invarg", "A blockchain account given by a string, an association or a rule", 1]; $Failed,
	BlockchainMessage[BlockchainAddressData, "invarg", "A property", 2]; $Failed
]

BlockchainAddressData[arg___, opt : OptionsPattern[BlockchainAddressData]] := (BlockchainMessage[BlockchainAddressData, "argn", "BlockchainAddressData", Length[{arg}], "1 or 2"]; $Failed)

(*****************  BlockchainContractCreate  *********************)
(*Options[BlockchainContractCreate] = {
    BlockchainBase :> $BlockchainBase
}
ValidContractFileQ[input_String] := If[FileFormat[input] === "Package" || FileExtension[input] === "sol", True, False]
ValidContractFileQ[___] := False

BlockchainContractCreate[input: _?ValidContractFileQ | _ExternalCloudStorageLink`IPFSObject, opt : OptionsPattern[BlockchainContractCreate]] := Module[
    {blockchain = Catch[getBlockchainBaseContracts[BlockchainContractCreate, OptionValue[BlockchainBase]]], result},

	result = Switch[First@blockchain,
		"Ethereum", Catch[BlockchainContractCreateETH[input, Last@blockchain]],
		_, $Failed
	];
	result

]*)

BlockchainContractCreate[___, opt : OptionsPattern[BlockchainContractCreate]] := (BlockchainMessage[BlockchainContractCreate, "invinp"]; $Failed)

(*****************  BlockchainContractValue  *********************)

Options[BlockchainContractValue] = {
    BlockchainBase :> $BlockchainBase
}

BlockchainContractValue[contractAddress_String, functionCall:(_Association | _String | _Typed | All): All, opt : OptionsPattern[BlockchainContractValue]] := Catch[Module[
	    {blockchain = getBlockchainBaseContracts[BlockchainContractValue, OptionValue[BlockchainBase]], result},

		result = Switch[First@blockchain,
			"Ethereum", BlockchainContractValueETH[contractAddress, functionCall, Last@blockchain],
			_, BlockchainMessage[BlockchainContractValue, "invnet", blockchain]; Throw[$Failed]
		];

		result

]]

BlockchainContractValue[arg1_, arg2 : RepeatedNull[_,1], opt : OptionsPattern[BlockchainContractValue]] := If[MatchQ[{arg2}, ({_Association} | {_String} | {_Typed} | {})],
	BlockchainMessage[BlockchainContractValue, "invarg", "A contract address", 1]; $Failed,
	BlockchainMessage[BlockchainContractValue, "invarg", "A contract function definition given by an association, a string or a Typed expression", 2]; $Failed
]

BlockchainContractValue[arg___, opt : OptionsPattern[BlockchainContractValue]] := (BlockchainMessage[BlockchainContractValue, "argn", "BlockchainContractValue", Length[{arg}], "1 or 2"]; $Failed)

(*****************  ContractEncoder  *********************)

PackageExport[ContractEncode]

Options[ContractEncode] = {
    BlockchainBase :> $BlockchainBase
}

ContractEncode[x_, type_, opt: OptionsPattern[ContractEncode]] := Catch[Module[
    {blockchain = getBlockchainBaseContracts[ContractEncode, OptionValue[BlockchainBase]], result},

	result = Switch[First@blockchain,
		"Ethereum", ContractEncodeETH[x, type],
		_, BlockchainMessage[ContractEncode, "invnet", blockchain]; Throw[$Failed]
	];

	result

]]

ContractEncode[___, opt : OptionsPattern[ContractEncode]] := (BlockchainMessage[ContractEncode, "invinp"]; $Failed)

(*****************  ContractDecode  *********************)

PackageExport[ContractDecode]

Options[ContractDecode] = {
    BlockchainBase :> $BlockchainBase
}

ContractDecode[x_, type_, opt: OptionsPattern[ContractDecode]] := Catch[Module[
    {blockchain = getBlockchainBaseContracts[ContractDecode, OptionValue[BlockchainBase]], result},

	result = Switch[First@blockchain,
		"Ethereum", ContractDecodeETH[x, type],
		_, BlockchainMessage[ContractDecode, "invnet", blockchain]; Throw[$Failed]
	];
	
	result
]]

ContractDecode[___, opt : OptionsPattern[ContractDecode]] := (BlockchainMessage[ContractDecode, "invinp"]; $Failed)


BlockchainGet[txid:(_String | {__String})] := Module[
	{result}, Catch[

	If[Not[hexStringQ[#] && (StringLength[#] === 64)],
		BlockchainMessage[BlockchainGet, "invprm", #,"transaction ID"]; Throw[$Failed]
	] & /@ Flatten[{txid}];	

	result = BlockchainGetWL[txid];
	If[StringQ[txid] && !FailureQ[result], First[result], result]
]]

BlockchainGet[arg_] := (BlockchainMessage[BlockchainGet, "invarg", "A transaction ID", 1]; $Failed)

BlockchainGet[arg___] := (BlockchainMessage[BlockchainGet, "argn1", "BlockchainGet", Length[{arg}]]; $Failed)

BlockchainPut[data_] := Module[
	{bytes, strData},

	bytes = If[!MatchQ[data, _ByteArray], BinarySerialize[data], data];
	strData = Compress[bytes, Method -> {"Version" -> 3}];
	Catch[BlockchainPutWL[strData]]
]

BlockchainPut[arg___] := (BlockchainMessage[BlockchainPut, "argn1", "BlockchainPut", Length[{arg}]]; $Failed)

(*****************  BlockchainContract  *********************)

PackageExport[BlockchainContract]

Options[BlockchainContract] = {
    BlockchainBase :> $BlockchainBase
}
ValidContractFileQ[input_String] := If[FileFormat[input] === "Package" || FileExtension[input] === "sol", True, False]
ValidContractFileQ[input_Association] := If[KeyExistsQ[input, "ByteCode"], False, True]
ValidContractFileQ[input_ExternalStorage`IPFSObject] := True
ValidContractFileQ[___] := False

BlockchainContract[input: _?ValidContractFileQ, opt:OptionsPattern[]] := Module[
{blockchain, result},

Catch[
	blockchain = getBlockchainBaseContracts[BlockchainContract, OptionValue[BlockchainBase]];
	result = Switch[First@blockchain, 
				"Ethereum", BlockchainContractETH[input, Last@blockchain],
				_, BlockchainMessage[BlockchainContract, "invnet", blockchain]; Throw[$Failed]
	];
	BlockchainContract[result]
	]
]


(*Error Handling*)
BlockchainContract[opt : OptionsPattern[BlockchainContract]] := (BlockchainMessage[BlockchainContract, "invinp"]; $Failed)
BlockchainContract[Except[_Association],___, opt : OptionsPattern[BlockchainContract]] := (BlockchainMessage[BlockchainContract, "invinp"]; $Failed)


HoldPattern[BlockchainContract[a_Association][c_]]:= a[c]

BoxForm`MakeConditionalTextFormattingRule[BlockchainContract];

BlockchainContract /: MakeBoxes[f:BlockchainContract[a_Association], form: StandardForm | TraditionalForm] := Module[
    {icon, alwaysGrid, sometimesGrid, dd},
    icon = makeIcon[f["IconData"], 10];

    alwaysGrid = {
    BoxForm`SummaryItem[{"blockchain: ",f["Blockchain"]}],
    BoxForm`SummaryItem[{"network: ",f["Network"]}]
    };

    dd = Association @@ DeleteCases[Normal[a], ("Blockchain" -> _) | ("Network" -> _) | ("IconData" -> _)];
    If[ KeyExistsQ[dd, "ApplicationBinaryInterface"], 
    	dd["ApplicationBinaryInterface"] = Developer`WriteRawJSONString[dd["ApplicationBinaryInterface"], "Compact" -> True], 
    	AppendTo[dd, "ApplicationBinaryInterface" -> Missing["NotAvailable"]]
    ];
    
    sometimesGrid = (toextrafield /* BoxForm`SummaryItem) @@@ Normal[dd];

    BoxForm`ArrangeSummaryBox[BlockchainContract, f, icon, alwaysGrid, sometimesGrid, form, "Interpretable" -> True]
]

(*****************  BlockchainTransaction  *********************)

BlockchainTransaction[a_Association/;Not[KeyExistsQ[a, "MessageHash"]]] := Module[
	{blockchain, result, aa}, Catch[
	If[KeyExistsQ[a, "BlockchainBase"], 
		blockchain = getBlockchainBase[BlockchainTransaction, a["BlockchainBase"]]
	,
		blockchain = getBlockchainBase[BlockchainTransaction, $BlockchainBase]
	];
	aa = Append[a, "BlockchainBase" -> blockchain];

	result = Switch[First@blockchain,
		"Bitcoin", BlockchainTransactionCreateBTC[aa, Last@blockchain],
		"Ethereum", BlockchainTransactionCreateETH[aa, Last@blockchain],
		_, BlockchainMessage[BlockchainTransaction, "invnet", blockchain]; Throw[$Failed]
	];
    BlockchainTransaction[result]

]]

HoldPattern[BlockchainTransaction[a_Association][c_]]:= a[c]

BoxForm`MakeConditionalTextFormattingRule[BlockchainTransaction];

BlockchainTransaction /: MakeBoxes[f : BlockchainTransaction[a_Association], form : StandardForm | TraditionalForm] := Catch[Module[
	{blockchain, icon, alwaysGrid, sometimesGrid},

	blockchain = getBlockchainBase[BlockchainTransaction, a["BlockchainBase"]];

    Which[
        KeyExistsQ[a, "TransactionID"], icon = makeIcon[a["MessageHash"], 102],
        a["Signed"], icon = makeIcon[a["MessageHash"], 47],
        True, icon = ColorConvert[makeIcon[a["MessageHash"], 47], "Grayscale"]
    ];
    alwaysGrid = {
        BoxForm`SummaryItem[{"blockchain base: ", a["BlockchainBase"]}],
        BoxForm`SummaryItem[{"signed: ", a["Signed"]}]
    };
    AppendTo[alwaysGrid, BoxForm`SummaryItem[{Style["fee: ", FontColor -> Red, Bold], Style[a["Fee"], Bold]}]];
    sometimesGrid = (toextrafield /* BoxForm`SummaryItem) @@@ DeleteCases[ Normal[a], ("BlockchainBase" -> _) | ("Signed" -> _) | ("Fee" -> _)];
    BoxForm`ArrangeSummaryBox[BlockchainTransaction, f, icon, alwaysGrid, sometimesGrid, form, "Interpretable" -> True]
]]

BlockchainTransaction[a_?(!MatchQ[#, _Association]&)] := (BlockchainMessage[BlockchainTransaction, "invarg", "A transaction definition given by an association", 1]; $Failed)

BlockchainTransaction[] := (BlockchainMessage[BlockchainTransaction, "argn1", "BlockchainTransaction", 0]; $Failed)

BlockchainTransaction[_, arg__] := (BlockchainMessage[BlockchainTransaction, "argn1", "BlockchainTransaction", Length[{arg}] + 1]; $Failed)
