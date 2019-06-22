Package["Blockchain`"]

$ethPath = "/eth";
$SolidityTemplateDirectory = PacletManager`PacletResource["Blockchain", "Solidity_Templates"];
$SolidityTemplateV1Filename = "Template1v1.sol"; (*Template for ExternalStorageObject*)
$SolidityTemplateV2Filename = "Template2v1.sol"; (*Template for WLS*)
$WServerToken = Compress[<|"Application" -> "Wolfram BlockchainContract", "Version" -> "1.0"|>];

PackageScope[BlockchainDataETH]
BlockchainDataETH[network_String] := Module[
    {blockchainPath = $ethPath, requestType = "blockchainData", response, params, gasPrices},

    params = {"network" -> network};
    response = ExplorerRequest[blockchainPath, requestType, params];

    Association[
        "Type" -> "Ethereum",
        "Name" -> Which[network === "mainnet", "ETH.main",
                        network === "testnet", "ETH.test"
                    ],
        "Core" -> "Ethereum",        
        "Blocks" -> response["Blocks"],
        "LatestHash" -> StringDelete[response["LatestHash"], "0x"],
        "LowestGasPrice" -> Quantity[ToExpression[response["LowestGasPrice"]]/10.^9, "gwei"],
        "AverageGasPrice" -> Quantity[ToExpression[response["AverageGasPrice"]]/10.^9, "gwei"],
        "HighestGasPrice" -> Quantity[ToExpression[response["HighestGasPrice"]]/10.^9, "gwei"]

    ]
]

PackageScope[BlockchainBlockDataETH]
BlockchainBlockDataETH[block: First | Last | _Integer | _String, network_String] := Module[
{blockchainPath = $ethPath, requestType = "blockData", response, params},

    params = Which[
        ToString[block] === "First", {"blockID" -> "0"},
        ToString[block] === "Last", {"blockID" -> "-1"},
        IntegerQ[block], {"blockID" -> ToString[block]},
        hexStringQ[StringDelete[block, "0x"]] && StringLength[StringDelete[block, "0x"]] === 64, {"blockID" -> StringJoin["0x", StringDelete[block, "0x"]]},
        True, (BlockchainMessage[BlockchainBlockData, "invprm", block, "block hash"]; Throw[$Failed])
    ];

    AppendTo[params, "network" -> network];

    response = ExplorerRequest[blockchainPath, requestType, params];

    Association[
        "BlockHash" -> StringDelete[response["BlockHash"], "0x"],
        "BlockNumber" -> response["BlockNumber"],        
        "TimeMined" -> TimeZoneConvert[FromUnixTime[response["TimeMined"], TimeZone -> 0], $TimeZone],        
        "ByteCount" -> response["ByteCount"],        
        "Nonce" -> FromDigits[StringDelete[response["Nonce"], "0x"], 16],
        "Version" -> response["Version"],
        "Confirmations" -> response["Confirmations"],
        (*"TotalEvents" -> response["TotalEvents"],
        "TotalInternalTransactions" -> response["TotalInternalTransactions"],*)
        "PreviousBlockHash" -> StringDelete[response["PreviousBlockHash"], "0x"],
        "MerkleRoot" -> StringDelete[response["MerkleRoot"], "0x"],
        "TransactionList" -> (StringDelete[#, "0x"]&) /@ response["TransactionList"]
    ]
]

PackageScope[BlockchainTransactionDataETH]
BlockchainTransactionDataETH[txid_, network_String] := Module[
    {blockchainPath = $ethPath, requestType = "transactionData", response, result, params, txid0},

    txid0 = StringDelete[txid, "0x"];
    
    If[hexStringQ[txid0] && StringLength[txid0] === 64,
        params = {"transactionID" -> "0x" <> txid0, "network" -> network},
        BlockchainMessage[BlockchainTransactionData, "invprm", txid, "transaction ID"]; Throw[$Failed]
    ];
    response = ExplorerRequest[blockchainPath, requestType, params];

    result = DeleteMissing[Association[
        "TransactionID" -> StringDelete[response["TransactionID"], "0x"],
        "BlockHash" -> StringDelete[response["BlockHash"], "0x"],
        "BlockNumber" -> response["BlockNumber"],
        "Confirmations" -> response["Confirmations"],
        "Time" -> TimeZoneConvert[FromUnixTime[response["Time"], TimeZone -> 0], $TimeZone],
        "StateRoot" -> With[{tmp = response["StateRoot"]}, If[MissingQ[tmp], tmp, StringDelete[tmp, "0x"]]],
        "Status" -> response["Status"],
        "TransactionIndex" -> response["TransactionIndex"],
        "Sender" -> StringDelete[response["From"], "0x"],
        "Receiver" -> With[{tmp = response["To"]}, If[MissingQ[tmp], tmp, StringDelete[tmp, "0x"]]],
        "ContractAddress" -> With[{tmp = response["ContractAddress"]}, If[MissingQ[tmp], tmp, StringDelete[tmp, "0x"]]],
        "Amount" -> Quantity[ToExpression[response["TotalWei"]], "Wei"],
        "GasUsed" -> response["GasUsed"],
        "GasPrice" -> Quantity[ToExpression[response["GasPrice"]], "Wei"],
        "Fee" -> Quantity[response["GasUsed"] * ToExpression[response["GasPrice"]], "Wei"],
        "TransactionCount" -> response["Nonce"],
        "ByteCount" -> response["ByteCount"],
        "InputData" -> ByteArray[FromDigits[#, 16] & /@ StringPartition[StringDelete[response["Input"], "0x"], 2]]/.{{} :> Missing["NotAvailable"]}
    ]];

    If[KeyExistsQ[response, "Events"], result = Join[result, Association["EventList" -> 
        (Association[
            "Address" -> StringDrop[#["address"], 2],
            "Topics" -> (StringDrop[#, 2] &/@ #["topics"]),
            "Data" -> With[{dat = #["data"]}, If[dat === "0x", Null, StringDrop[dat, 2]]]
        ] &/@ response["Events"])
    ]]];

    If[KeyExistsQ[response, "TokenTransfers"], result = Join[result, Association["TokenTransferList" -> (With[{temp = Switch[Length[#["Topics"]],
            1, {StringTake[StringDrop[#["Data"],2], {1, 64}], StringTake[StringDrop[#["Data"],2], {65, 128}], StringTake[StringDrop[#["Data"],2], {129, 192}]},
            3, {StringDrop[#["Topics"][[2]],2], StringDrop[#["Topics"][[3]],2], StringDrop[#["Data"],2]},
            4, {StringDrop[#["Topics"][[2]],2], StringDrop[#["Topics"][[3]],2], StringDrop[#["Topics"][[4]],2]}
        ]}, Association[
            "Name" -> #["Name"],
            "Symbol" -> #["Symbol"],
            "Sender" -> checksum[ToLowerCase[StringTake[temp[[1]],-40]]],
            "Receiver" -> checksum[ToLowerCase[StringTake[temp[[2]],-40]]],
            Which[MatchQ[#["Standard"], 20] || MemberQ[#["Standard"], 20], "Amount", MatchQ[#["Standard"], 721] || MemberQ[#["Standard"], 721], "TokenID"] -> FromDigits[temp[[3]],16]
        ]  
    ] &/@ response["TokenTransfers"])]]];

    If[KeyExistsQ[response, "InternalTxs"], result = Join[result, Module[{intTxs, intTxs0, calls, maxChild},
        calls = Transpose[{#, "_" <> ToString[#] &/@ Range[Length[#]]}] &@ response["InternalTxs"];
        intTxs = {};
        While[!MatchQ[calls, {}],
            intTxs0 = With[{suffix = #[[2]]}, MapAll[Replace[#, (<|"type" -> typ_, res___|> :> <|"type" -> typ <> suffix, res|>)] &, #[[1]]]] &/@ calls;
            intTxs = Join[intTxs, Flatten[If[KeyExistsQ[#, "calls"], {Delete[#, "calls"]}, {#}] &/@ intTxs0]];
            calls = DeleteCases[Flatten[If[KeyExistsQ[#, "calls"], Transpose[{#, "_" <> ToString[#] &/@ Range[Length[#]]}] &@ #["calls"]] & /@ intTxs0, 1], Null];
        ];
        maxChild = Max[Length[StringCases[#, RegularExpression["(?<=_)\\d+"]]] & /@ (#["type"] & /@ intTxs)];
        intTxs = SortBy[intTxs, With[{temp = ToExpression[StringCases[#["type"], RegularExpression["(?<=_)\\d+"]]]}, Join[temp, Table[0, maxChild - Length[temp]]]]& ];
        Association["InternalTransactionList" -> (DeleteMissing[Association[
            "CallType" -> #["type"],
            "Sender" -> If[KeyExistsQ[#,"from"], checksum[ToLowerCase[StringDrop[#["from"], 2]]], Missing["NotAvailable"]],
            "Receiver" -> If[KeyExistsQ[#,"to"], checksum[ToLowerCase[StringDrop[#["to"], 2]]], Missing["NotAvailable"]],
            "Amount" -> If[KeyExistsQ[#,"value"], Quantity[FromDigits[StringDrop[#["value"], 2], 16], "Wei"], Missing["NotAvailable"]],
            "GasLimit" -> If[KeyExistsQ[#,"gas"], FromDigits[StringDrop[#["gas"], 2], 16], Missing["NotAvailable"]],
            "GasUsed" -> If[KeyExistsQ[#,"gasUsed"], FromDigits[StringDrop[#["gasUsed"], 2], 16], Missing["NotAvailable"]],
            "Input" -> If[KeyExistsQ[#,"input"], ByteArray[(FromDigits[#,16] &/@ StringPartition[StringDrop[#["input"], 2], 2])] /.{{} :> Missing["NotAvailable"]}, Missing["NotAvailable"]],
            "Output" -> If[KeyExistsQ[#,"output"], ByteArray[(FromDigits[#,16] &/@ StringPartition[StringDrop[#["output"], 2], 2])] /.{{} :> Missing["NotAvailable"]}, Missing["NotAvailable"]]
    ]] &/@ intTxs)]]]];

    result

]

PackageScope[BlockchainKeyEncodeETH]
BlockchainKeyEncodeETH[key_] := Module[{pubkey},

    Switch[Head[key], 
        String,
            Switch[StringLength[key],
                64, 
                    pubkey = StringDrop[Cryptography`GenerateEllipticCurvePublicKey[key, "secp256k1"], 2];
                ,66,
                    pubkey = StringJoin[IntegerString[Cryptography`PublicKeyFormat[key, "secp256k1", "Coordinates"], 16, 64]];
                ,130, 
                    pubkey = StringDrop[key,2];
            ]
        ,PublicKey,
            If[key["CurveName"] =!= "secp256k1", BlockchainMessage[BlockchainKeyEncode, "curvpub"]; Throw[$Failed]];
            Switch[StringLength[key["PublicKeyHexString"]],
                66,
                    pubkey = StringJoin[IntegerString[Cryptography`PublicKeyFormat[key["PublicKeyHexString"], "secp256k1", "Coordinates"], 16, 64]];
                ,130,
                    pubkey = StringDrop[key["PublicKeyHexString"], 2];
            ]
        ,PrivateKey,
            If[key["CurveName"] =!= "secp256k1", BlockchainMessage[BlockchainKeyEncode, "curvpri"]; Throw[$Failed]];
            pubkey = StringDrop[Cryptography`GenerateEllipticCurvePublicKey[key["PrivateKeyHexString"], "secp256k1"], 2];
    ];
    
    checksum[StringTake[Hash[ByteArray[FromDigits[#, 16] & /@ StringPartition[pubkey, 2]], "Keccak256", "HexString"], -40]]

]

PackageScope[BlockchainTransactionCreateETH]

BlockchainTransactionCreateETH[a_Association, network_String] := Module[
    {
        blockchainPath = $ethPath, requestType = "estimateGas", response, chainId, messageHash, a0, a00, a000, gasPrice0, 
        value0, inputTypes, params0, constructorData, result0, abiraw, id, paramsABI, func0, to0, auxdata, fee,
        nonce = a["TransactionCount"], to = a["Address"], value = a["Amount"], gasPrice = a["GasPrice"], blockchainBase = {"Ethereum", Switch[network, "mainnet", "MainNet", "testnet", "TestNet"]}
    },

    If[!MissingQ[to], If[validateEthereumAddress[to], to = StringDelete[to, "0x"];, (BlockchainMessage[BlockchainTransaction, "ethadd", to]; Throw[$Failed])]];
    If[MissingQ[nonce], 
        (BlockchainMessage[BlockchainTransaction, "misselem", "TransactionCount", "input"]; Throw[$Failed]),
        If[!Internal`NonNegativeIntegerQ[nonce], (BlockchainMessage[BlockchainTransaction, "invval", nonce, "TransactionCount", "A non negative integer"]; Throw[$Failed])]
    ];
    If[MissingQ[gasPrice], (BlockchainMessage[BlockchainTransaction, "misselem", "GasPrice", "input"]; Throw[$Failed])];

    If[!MemberQ[{"BlockchainBase", "TransactionCount", "GasPrice", "Address", "Amount", "Contract", "WolframExpression", "WolframExpressionReference", "FunctionCall"}, #],
        (BlockchainMessage[BlockchainTransaction, "invelem", #, "input"]; Throw[$Failed])
    ] &/@ Keys[a];

    gasPrice0 = If[!MatchQ[gasPrice, _?Internal`NonNegativeIntegerQ | _Quantity?(MatchQ[QuantityUnit[#], "Wei" | "Gwei" | "Ethers"] && NonNegative[QuantityMagnitude[#]] &)],
        BlockchainMessage[BlockchainTransaction, "invval", gasPrice, "GasPrice", "A non negative value given by an integer or a quantity in Ether, Gwei or Wei"];
        Throw[$Failed],
        If[Head[gasPrice] === Quantity, Switch[Head[QuantityUnit[gasPrice]],
            Integer, Switch[QuantityUnit[gasPrice],
                    "Ethers", QuantityMagnitude[gasPrice]*10^18
                    ,"Gwei", QuantityMagnitude[gasPrice]*10^9
                    ,"Wei", QuantityMagnitude[gasPrice]
                ]
            ,String, Switch[QuantityUnit[gasPrice],
                    "Ethers", Round[ToExpression[QuantityMagnitude[gasPrice]]*10^18]
                    ,"Gwei", Round[ToExpression[QuantityMagnitude[gasPrice]]*10^9]
                    ,"Wei", Round[ToExpression[QuantityMagnitude[gasPrice]]]
                ]
            ,Real, Switch[QuantityUnit[gasPrice],
                    "Ethers", Round[QuantityMagnitude[gasPrice]*10^18]
                    ,"Gwei", Round[QuantityMagnitude[gasPrice]*10^9]
                    ,"Wei", Round[QuantityMagnitude[gasPrice]]
                ]
            ], gasPrice
        ]
    ];

    If[!MissingQ[value],
        If[!MatchQ[value, _?Internal`NonNegativeIntegerQ | _Quantity?(MatchQ[QuantityUnit[#], "Wei" | "Gwei" | "Ethers"] && NonNegative[QuantityMagnitude[#]] &)],
            BlockchainMessage[BlockchainTransaction, "invval", value, "Amount", "A non negative value given by an integer or a quantity in Ether, Gwei or Wei"];
            Throw[$Failed],
            value0 = If[Head[value] === Quantity, Switch[Head[QuantityUnit[value]],
                Integer, Switch[QuantityUnit[value],
                        "Ethers", QuantityMagnitude[value]*10^18
                        ,"Gwei", QuantityMagnitude[value]*10^9
                        ,"Wei", QuantityMagnitude[value]
                    ]
                ,String, Switch[QuantityUnit[value],
                        "Ethers", Round[ToExpression[QuantityMagnitude[value]]*10^18]
                        ,"Gwei", Round[ToExpression[QuantityMagnitude[value]]*10^9]
                        ,"Wei", Round[ToExpression[QuantityMagnitude[value]]]
                    ]
                ,Real, Switch[QuantityUnit[value],
                        "Ethers", Round[QuantityMagnitude[value]*10^18]
                        ,"Gwei", Round[QuantityMagnitude[value]*10^9]
                        ,"Wei", Round[QuantityMagnitude[value]]
                    ]
                ], value
            ];
        ];
        a000 = Association[SortBy[Normal[Delete[Delete[Delete[Delete[a, "BlockchainBase"], "TransactionCount"], "GasPrice"], "Amount"]], Keys]];
        ,
        value0 = 0;
        a000 = Association[SortBy[Normal[Delete[Delete[Delete[a, "BlockchainBase"], "TransactionCount"], "GasPrice"]], Keys]];
    ];

    Which[
        MatchQ[a000, <|"Address" -> _String|>],
            to0 = to;
            auxdata = <|"dataBin" -> "", "extra" -> <||>|>;
        ,MatchQ[a000, <|"Contract" -> _|>],
            to0 = "";
            auxdata = processContract[a000["Contract"], value0];  
        ,MatchQ[a000, <|"WolframExpression" -> _|>],
            to0 = "";
            auxdata = processWolframExpression[a000["WolframExpression"], value0];
        ,MatchQ[a000, <|"WolframExpressionReference" -> _|>],
            to0 = "";
            auxdata = processWolframExpressionReference[a000["WolframExpressionReference"], value0];
        ,KeyExistsQ[a000, "Address"] && KeyExistsQ[a000, "FunctionCall"],
            to0 = to;
            auxdata = processFunctionCall[a000["FunctionCall"]];
        ,True, 
            BlockchainMessage[BlockchainTransaction, "invinas"]; Throw[$Failed]
    ];

    params = <|"network" -> network, "to" -> "0x" <> to0, "data" -> "0x" <> auxdata["dataBin"], "value" -> "0x" <> IntegerString[value0, 16],
            "sender" -> With[{tmp = auxdata["sender"]}, If[MissingQ[tmp], "0x", "0x" <> ToLowerCase[tmp]]]
        |>;
    gasLimit = ExplorerRequest[blockchainPath, requestType, params];
    fee = Quantity[gasPrice0*gasLimit, "wei"];
    chainId = Switch[network, "mainnet", 1, "testnet", 3];
    a0 = <|"TransactionCount" -> nonce, "GasPrice" -> gasPrice0, "GasLimit" -> gasLimit, "DestinationAddress" -> to0, "Amount" -> value0, "Data" -> auxdata["dataBin"], "ChainID" -> chainId|>;
    a00 = Join[<|"TransactionCount" -> nonce, "GasPrice" -> gasPrice, "GasLimit" -> gasLimit, "Fee" -> fee|>, DeleteCases[<|"Address" -> to, "Amount" -> value|>, 0 | "" | _Missing]];
    a00 = Join[a00, auxdata["extra"]];
    messageHash = Hash[ByteArray[FromDigits[#, 16] & /@ StringPartition[ethRlpEncode[a0], 2]], "Keccak256", "HexString"];

    Join[<|"BlockchainBase" -> blockchainBase, "Signed" -> False|>, a00, <|"MessageHash" -> messageHash|>]

]

PackageScope[BlockchainTransactionSignETH]
BlockchainTransactionSignETH[obj_BlockchainTransaction, privKey_] := Module[{blockchain, sigObj, rawTx, data = obj["ByteCode"], value = obj["Amount"], gasPrice = obj["GasPrice"], data0, value0, gasPrice0},
    
    If[ListQ[privKey], (BlockchainMessage[BlockchainTransactionSign, "invprm", privKey, "HexString or PrivateKey object"]; Throw[$Failed])];

    blockchain = getBlockchainBase[BlockchainTransactionSign, obj["BlockchainBase"]];

    privKeyHex = StringDelete[If[Head[privKey] === PrivateKey, privKey["PrivateKeyHexString"], privKey], "0x"];
    If[Not[hexStringQ[ToLowerCase[privKeyHex]] && StringLength[privKeyHex] === 64],
        BlockchainMessage[BlockchainTransactionSign, "invprm", privKey, "HexString or PrivateKey object"]; Throw[$Failed]
    ];

    sigObj = GenerateDigitalSignature[obj["MessageHash"], privKeyHex, Method -> <|"Type" -> "EllipticCurve", "CurveName" -> "Ethereum", "SignatureType" -> "Deterministic", "HashingMethod" -> None|>];

    gasPrice0 = If[Head[gasPrice] === Quantity, Switch[Head[QuantityUnit[gasPrice]],
        Integer, Switch[QuantityUnit[gasPrice],
                "Ethers", QuantityMagnitude[gasPrice]*10^18
                ,"Gwei", QuantityMagnitude[gasPrice]*10^9
                ,"Wei", QuantityMagnitude[gasPrice]
            ]
        ,String, Switch[QuantityUnit[gasPrice],
                "Ethers", Round[ToExpression[QuantityMagnitude[gasPrice]]*10^18]
                ,"Gwei", Round[ToExpression[QuantityMagnitude[gasPrice]]*10^9]
                ,"Wei", Round[ToExpression[QuantityMagnitude[gasPrice]]]
            ]
        ,Real, Switch[QuantityUnit[gasPrice],
                "Ethers", Round[QuantityMagnitude[gasPrice]*10^18]
                ,"Gwei", Round[QuantityMagnitude[gasPrice]*10^9]
                ,"Wei", Round[QuantityMagnitude[gasPrice]]
            ]
        ], gasPrice
    ];

    value0 = If[!MissingQ[value],
        If[Head[value] === Quantity, Switch[Head[QuantityUnit[value]],
            Integer, Switch[QuantityUnit[value],
                    "Ethers", QuantityMagnitude[value]*10^18
                    ,"Gwei", QuantityMagnitude[value]*10^9
                    ,"Wei", QuantityMagnitude[value]
                ]
            ,String, Switch[QuantityUnit[value],
                    "Ethers", Round[ToExpression[QuantityMagnitude[value]]*10^18]
                    ,"Gwei", Round[ToExpression[QuantityMagnitude[value]]*10^9]
                    ,"Wei", Round[ToExpression[QuantityMagnitude[value]]]
                ]
            ,Real, Switch[QuantityUnit[value],
                    "Ethers", Round[QuantityMagnitude[value]*10^18]
                    ,"Gwei", Round[QuantityMagnitude[value]*10^9]
                    ,"Wei", Round[QuantityMagnitude[value]]
                ]
            ], value
        ], 0
    ];

    data0 = If[!MissingQ[data], StringJoin[IntegerString[#, 16, 2] & /@ Normal[data]], ""];

    rawTx = ethRlpEncode[Association[
        "TransactionCount" -> obj["TransactionCount"], 
        "GasPrice" -> gasPrice0, 
        "GasLimit" -> obj["GasLimit"], 
        "DestinationAddress" -> If[KeyExistsQ[First[obj], "Address"], obj["Address"], ""],
        "Amount" -> value0,
        "Data" -> data0, 
        "v" -> Switch[ToLowerCase[Last[blockchain]], "mainnet", 37 + sigObj["RecoveryParameter"], "testnet", 41 + sigObj["RecoveryParameter"]],
        "r" -> NestWhile[StringDrop[#, 2] &, IntegerString[FromDigits[Normal[sigObj["R"]], 256], 16, 64], StringTake[#, 2] === "00" &],
        "s" -> NestWhile[StringDrop[#, 2] &, IntegerString[FromDigits[Normal[sigObj["S"]], 256], 16, 64], StringTake[#, 2] === "00" &]
    ]];

    BlockchainTransaction @ Join[(First[obj]/.(<|a__, "Signed" -> False , b__|> :> <|a, "Signed" -> True , b|>)), <|"RawTransaction" -> rawTx|>]

]

PackageScope[BlockchainTransactionSendETH]
BlockchainTransactionSendETH[obj_BlockchainTransaction, network_String] := Module[
    {blockchainPath = $ethPath, requestType = "transactionSend", response},
    
    params = <|"network" -> network, "rawTx" -> obj["RawTransaction"]|>;
    response = ExplorerRequest[blockchainPath, requestType, params];

    BlockchainTransaction @ Join[First[obj], <|"TransactionID" -> StringDrop[response["txid"], 2]|>]

]

PackageScope[BlockchainAddressDataETH]
BlockchainAddressDataETH[account: (_String | _Association), prop_, network_String, maxitems_String] := Module[
    {
        blockchainPath = $ethPath, requestType = "addressData", params, response, blocks, time, from, to, response0, account0, txsCountQ, eventsCountQ, internalTxsCountQ, txsQ, eventsQ, internalTxsQ,
        properties = {"Balance", "ValueInUSDollars", "TotalTransactions", "TransactionList", "TransactionCount", "TotalEvents", "EventList", "TotalInternalTransactions", "InternalTransactionList"}
    },

    Which[
        MatchQ[prop, {} | All], {txsCountQ, eventsCountQ, internalTxsCountQ, txsQ, eventsQ, internalTxsQ} = Table["true", 6];
        ,StringQ[prop],
            If[!MemberQ[properties, prop],
                Return[Missing["NotAvailable"]],
                Switch[prop,
                    "Balance", {txsCountQ, eventsCountQ, internalTxsCountQ, txsQ, eventsQ, internalTxsQ} = Table["false", 6];
                    ,"ValueInUSDollars", {txsCountQ, eventsCountQ, internalTxsCountQ, txsQ, eventsQ, internalTxsQ} = Table["false", 6];
                    ,"TransactionCount", {txsCountQ, eventsCountQ, internalTxsCountQ, txsQ, eventsQ, internalTxsQ} = Table["false", 6];
                    ,"TotalTransactions", {eventsCountQ, internalTxsCountQ, txsQ, eventsQ, internalTxsQ} = Table["false", 5]; txsCountQ = "true";
                    ,"TotalEvents", {txsCountQ, internalTxsCountQ, txsQ, eventsQ, internalTxsQ} = Table["false", 5]; eventsCountQ = "true";
                    ,"TotalInternalTransactions", {txsCountQ, eventsCountQ, txsQ, eventsQ, internalTxsQ} = Table["false", 5]; internalTxsCountQ = "true";
                    ,"TransactionList", {txsCountQ, eventsCountQ, internalTxsCountQ, eventsQ, internalTxsQ} = Table["false", 5]; txsQ = "true";
                    ,"EventList", {txsCountQ, eventsCountQ, internalTxsCountQ, txsQ, internalTxsQ} = Table["false", 5]; eventsQ = "true";
                    ,"InternalTransactionList", {txsCountQ, eventsCountQ, internalTxsCountQ, txsQ, eventsQ} = Table["false", 5]; internalTxsQ = "true";
                ]
            ]
        ,ListQ[prop],
            With[{props = Intersection[properties, prop]},
                If[props === {},
                    Return[Table[Missing["NotAvailable"], Length[prop]]]
                    ,
                    {txsCountQ, eventsCountQ, internalTxsCountQ, txsQ, eventsQ, internalTxsQ} = Table["false", 6];
                    Switch[#,
                        "TotalTransactions", txsCountQ = "true";
                        ,"TotalEvents", eventsCountQ = "true";
                        ,"TotalInternalTransactions", internalTxsCountQ = "true";
                        ,"TransactionList", txsQ = "true";
                        ,"EventList", eventsQ = "true";
                        ,"InternalTransactionList", internalTxsQ = "true";
                    ] &/@ props
                ]
            ]
    ];

    Switch[Head[account],
        String,
            If[!validateEthereumAddress[account], (BlockchainMessage[BlockchainAddressData, "ethadd", account]; Throw[$Failed]), account0 = StringDelete[account, "0x"]];
            blocks = "all";
            time = "all";
        ,Association,
            If[!MemberQ[{"Address", "TimeInterval", "BlockNumberInterval"}, #],
                (BlockchainMessage[BlockchainAddressData, "invelem", #, "input"]; Throw[$Failed])
            ] &/@ Keys[account];
            If[!KeyExistsQ[account,"Address"], 
                (BlockchainMessage[BlockchainAddressData, "misselem", "Address", "input"]; Throw[$Failed]),
                With[{acc = account["Address"]}, If[!validateEthereumAddress[acc], (BlockchainMessage[BlockchainAddressData, "ethadd", acc]; Throw[$Failed]), account0 = StringDelete[acc, "0x"]]];
                time = With[{tmp = account["TimeInterval"]}, If[MissingQ[tmp] || tmp === All, "all",
                    If[MatchQ[tmp, _DateObject | {_DateObject, _DateObject}], tmp, (BlockchainMessage[BlockchainAddressData, "invval", tmp, "TimeInterval", "A DateObject or a list of two DateObjects"]; Throw[$Failed])]
                ]];
                blocks = With[{tmp = account["BlockNumberInterval"]}, If[MissingQ[tmp] || tmp === All, "all",
                    If[MatchQ[tmp, {_Integer | First | "First", _Integer | Last | "Last"}], tmp, (BlockchainMessage[BlockchainAddressData, "invval", tmp, "BlockNumberInterval", "A list of two integers"]; Throw[$Failed])]
                ]];
            ];
    ];

    Switch[{blocks, time},
        {"all", "all"},
            from = "all";
            to = "all";
        ,
        {_?(!MatchQ[#,"all"]&), _?(!MatchQ[#,"all"]&)},
            BlockchainMessage[BlockchainAddressData, "invtime"]; 
            Throw[$Failed];
        ,
        {"all", _?(!MatchQ[#,"all"]&)},
            Switch[Head[time],
                DateObject,
                    from = "unixtime:" <> ToString @ UnixTime[First @ dateToRange[time]] <> "000";
                    to = "unixtime:" <> ToString @ UnixTime[Last @ dateToRange[time]] <> "000";
                ,List,
                    If[Last[time] < First[time], (BlockchainMessage[BlockchainAddressData, "invint", "TimeInterval"]; Throw[$Failed])];
                    from = "unixtime:" <> ToString @ UnixTime[First @ dateToRange[First @ time]] <> "000";
                    to = "unixtime:" <> ToString @ UnixTime[First @ dateToRange[Last @ time]] <> "000";
            ];
        ,
        {_?(!MatchQ[#,"all"]&), "all"},
            If[ToString @ First @ blocks === "First", from = "1", from = ToString @ First @ blocks];
            If[ToString @ Last @ blocks === "Last", to = "-1", to = ToString @ Last @ blocks];
    ];

    params = {
        "account" -> "0x" <> ToLowerCase[account0],
        "network" -> network,
        "from" -> from,
        "to" -> to,
        "maxItems" -> maxitems,
        "txsCountQ" -> txsCountQ,
        "eventsCountQ" -> eventsCountQ,
        "internalTxsCountQ" -> internalTxsCountQ,
        "txsQ" -> txsQ,
        "eventsQ" -> eventsQ,
        "internalTxsQ" -> internalTxsQ
    };

    response = ExplorerRequest[blockchainPath, requestType, params];

    response0 = DeleteMissing[
        Association[
            "Balance" -> If[KeyExistsQ[#, "Balance"], If[#["Balance"] =!= Null, Quantity[ToExpression[#["Balance"]], "wei"], Missing["NotAvailable"]], Missing["NotAvailable"]],
            "ValueInUSDollars" -> If[from === "all" && to === "all", CurrencyConvert[Quantity[ToExpression[#["Balance"]]/10.^18, "Ethers"], "USDollars"], Missing["NotAvailable"]],
            "TransactionCount" -> If[KeyExistsQ[#, "Nonce"], If[#["Nonce"] =!= Null, #["Nonce"], Missing["NotAvailable"]], Missing["NotAvailable"]],
            "TotalTransactions" -> If[KeyExistsQ[#, "Transactions"], With[{txs = #["Transactions"]}, If[txs === 0, Missing["NotAvailable"], txs]], Missing["NotAvailable"]],
            "TotalInternalTransactions" -> If[KeyExistsQ[#, "InternalTransactions"], With[{internals = #["InternalTransactions"]}, If[internals === 0, Missing["NotAvailable"], internals]], Missing["NotAvailable"]],
            "TotalEvents" -> If[KeyExistsQ[#, "Events"], With[{events = #["Events"]}, If[events === 0, Missing["NotAvailable"], events]], Missing["NotAvailable"]],
            "TransactionList" -> If[KeyExistsQ[#, "TransactionsList"],
                    With[{txsList = #["TransactionsList"]}, If[txsList === {}, Missing["NotAvailable"], 
                        Association[
                            "TransactionID" -> StringDrop[#["Hash"], 2],
                            "BlockNumber" -> #["BlockNumber"],
                            "Time" -> FromUnixTime[UnixTime[#["DateTime"], TimeZone -> 0]],
                            "Sender" -> checksum[ToLowerCase[StringDrop[#["From"], 2]]], 
                            "Receiver" -> With[{tmp = #["To"]}, If[tmp === Null, Missing["NotAvailable"], checksum[ToLowerCase[StringDrop[tmp, 2]]]]],
                            "Amount" -> Quantity[ToExpression[#["Value"]], "wei"]
                        ] &/@ txsList]
                    ]
                    ,
                    Missing["NotAvailable"]
                ],
            "InternalTransactionList" -> If[KeyExistsQ[#, "InternalTransactionsList"],
                    With[{internalsList = #["InternalTransactionsList"]}, If[internalsList === {}, Missing["NotAvailable"], 
                        Association[
                            "CallType" -> #["Type"],
                            "TransactionID" -> StringDrop[#["TransactionHash"], 2],
                            "BlockNumber" -> #["BlockNumber"],
                            "Time" -> FromUnixTime[UnixTime[#["DateTime"], TimeZone -> 0]],
                            "Sender" -> checksum[ToLowerCase[StringDrop[#["From"], 2]]], 
                            "Receiver" -> With[{tmp = #["To"]}, If[tmp === Null, Missing["NotAvailable"], checksum[ToLowerCase[StringDrop[tmp, 2]]]]],       
                            "Amount" -> Quantity[ToExpression[#["Value"]], "wei"]
                        ] &/@ internalsList]
                    ]
                    ,
                    Missing["NotAvailable"]
                ],
            "EventList" -> If[KeyExistsQ[#, "EventsList"],
                    With[{eventsList = #["EventsList"]}, If[eventsList === {}, Missing["NotAvailable"], 
                        Association[
                            "TransactionID" -> StringDrop[#["TransactionHash"], 2],
                            "BlockNumber" -> #["BlockNumber"],
                            "Time" -> FromUnixTime[UnixTime[#["DateTime"], TimeZone -> 0]],
                            "Address" -> checksum[ToLowerCase[StringDrop[#["Address"], 2]]],      
                            "Topics" -> (StringDrop[#, 2] &/@ #["Topics"]),
                            "Data" -> With[{tmp = #["Data"]}, If[tmp === Null, Missing["NotAvailable"], StringDrop[tmp, 2]]]
                        ] &/@ eventsList]
                    ]
                    ,
                    Missing["NotAvailable"]
                ]
        ] &@ response
    ];

    If[SameQ[prop, {}] || SameQ[response0, $Failed] || SameQ[prop, All],
        If[response0 === <||>, Missing["NotAvailable"], response0],        
        Lookup[response0, prop, Missing["NotAvailable"]]
    ]

]

processContract[x_, value0_] := Module[{params, result, dataBin, abiraw, extra, constructorData, parameters, x0, file0},
    If[!MatchQ[x, (_File | _String | _Association)], BlockchainMessage[BlockchainTransaction, "invval", x, "Contract", "Either a file, a string or an association"]; Throw[$Failed]];
    Switch[Head[x], 
        File,
            If[FileExtension[x] =!= "sol", BlockchainMessage[BlockchainTransaction, "solfil", x]; Throw[$Failed]];
            file0 = FindFile[x];
            If[FailureQ[file0], BlockchainMessage[BlockchainTransaction, "nofile", x]; Throw[$Failed]]; 
            result = ExplorerRequest[$ethPath, "compile", <|"file" -> File[file0]|>];
            constructorData = "";
        ,String,
            result = ExplorerRequest[$ethPath, "compile2", <|"code" -> x|>];
            constructorData = "";
        ,Association,
            If[!MemberQ[{"Parameters", "ABI", "ByteCode", "Source", "Main", "SourceCode"}, #],
                (BlockchainMessage[BlockchainTransaction, "invelem", #, "\"Contract\""]; Throw[$Failed])
            ] &/@ Keys[x];
            parameters = With[{params = x["Parameters"]}, If[MissingQ[params], params, If[validateEthereumAddress[#], StringDelete[#, "0x"], #] &/@ If[ListQ[params], params, {params}]]];
            x0 = Delete[x,"Parameters"];
            getConstructorData = Module[{parameters = If[MissingQ[#1], {}, #1], result = #2, constructor, paramsTypes},
                constructor = Cases[ImportString[#2["abi"], "RawJSON"], <|___, "type" -> "constructor", ___|>];
                paramsTypes = If[# === {}, {}, #["type"] &/@ #["inputs"] &@@ #] &@ constructor;
                Which[
                    paramsTypes === {} && parameters =!= {}, (BlockchainMessage[BlockchainTransaction, "solnop"]; Throw[$Failed]),
                    paramsTypes =!= {} && parameters === {}, "",
                    paramsTypes === {} && parameters === {}, "",
                    paramsTypes =!= {} && parameters =!= {}, 
                        If[Length[parameters] =!= Length[paramsTypes], (BlockchainMessage[BlockchainTransaction, "solpar", Length[paramsTypes], Length[parameters]]; Throw[$Failed])];
                        If[!AllTrue[Transpose[{parameters, paramsTypes}], crossValidateSolidityInput[#[[1]], #[[2]]]&], (BlockchainMessage[BlockchainTransaction, "abienc", parameters, paramsTypes]; Throw[$Failed])];
                        With[{enc = Quiet[Check[encodeParameters[parameters, paramsTypes], $Failed]]},
                            If[FailureQ[enc],
                                BlockchainMessage[BlockchainTransaction, "abienc", parameters, paramsTypes]; Throw[$Failed],
                                enc
                            ]
                        ]
                    ]
                ]&;
            Which[
                MatchQ[SortBy[Normal[x0], Keys], {"ABI" -> _String, "ByteCode" -> _String}],
                    result = <|"bin" -> StringDelete[x0["ByteCode"], "0x"], "abi" -> x0["ABI"]|>;
                ,MatchQ[SortBy[Normal[x0], Keys], ({"Source" -> _File} | {"Main" -> _String, "Source" -> _File})],
                    If[FileExtension[x0["Source"]] =!= "sol", BlockchainMessage[BlockchainTransaction, "solfil", x]; Throw[$Failed]];
                    file0 = FindFile[x0["Source"]];
                    If[FailureQ[file0], BlockchainMessage[BlockchainTransaction, "nofile", x0["Source"]]; Throw[$Failed]];
                    params = If[KeyExistsQ[x0, "Main"], <|"file" -> File[file0], "main" -> x0["Main"]|>, <|"file" -> File[file0]|>];
                    result = ExplorerRequest[$ethPath, "compile", params];
                ,MatchQ[SortBy[Normal[x0], Keys], {"Main" -> _String, "Source" -> {_File..}}],
                    If[FileExtension[#] =!= "sol", BlockchainMessage[BlockchainTransaction, "solfil", #]; Throw[$Failed]]&/@ x0["Source"];
                    If[FailureQ[FindFile[#]], BlockchainMessage[BlockchainTransaction, "nofile", #]; Throw[$Failed]]&/@ x0["Source"];
                    file0 = File[FindFile[#]] &/@ x0["Source"];
                    If[!verifyImports[#],BlockchainMessage[BlockchainTransaction, "solimp", #]; Throw[$Failed]]&/@ file0;
                    params = <|"files" -> file0, "file" -> FileNameTake[First[file0]], "main" -> x0["Main"]|>;
                    result = ExplorerRequest[$ethPath, "compile3", params];
                ,MatchQ[SortBy[Normal[x0], Keys], ({"SourceCode" -> _String} | {"Main" -> _String, "SourceCode" -> _String})],
                    params = If[KeyExistsQ[x0, "Main"], <|"code" -> x0["SourceCode"], "main" -> x0["Main"]|>, <|"code" -> x0["SourceCode"]|>];
                    result = ExplorerRequest[$ethPath, "compile2", params];
                ,True,
                    BlockchainMessage[BlockchainTransaction, "invassoc", x, "Contract"]; Throw[$Failed]
            ];
            constructorData = getConstructorData @@ {parameters, result};
        ];
    If[KeyExistsQ[result, "warning"], BlockchainMessage[BlockchainTransaction, "solwar", result["warning"]]];
    dataBin = result["bin"] <> constructorData;
    If[!MatchQ[ImportString[result["abi"], "RawJSON"], {___, <|___,"payable" -> True, ___,"type" -> "constructor"|>, ___}] && value0 =!= 0,
        BlockchainMessage[BlockchainTransaction, "solcon"]; Throw[$Failed]
    ];
    abiraw = Cases[ImportString[result["abi"], "RawJSON"], <|___, "type" -> "function", ___|>];
    abiraw = abiraw /. (<|_, "inputs" -> in_, "name" -> nam_, "outputs" -> out_, __|> :> <|nam -> Typed[nam, Rule[Map[#["type"] &, in], Map[#["type"] &, out]]]|>);
    extra = <|"Contract" -> x, "ByteCode" -> ByteArray[FromDigits[#, 16] & /@ StringPartition[dataBin, 2]], "ABI" -> Association[abiraw]|>;
    <|"dataBin" -> dataBin, "extra" -> extra|> 
]

processWolframExpression[x_, value0_] := Module[{params, constructorData, result, dataBin, extra},
    If[Head[x] =!= Hold && Head[x] =!= Delayed, BlockchainMessage[BlockchainTransaction, "invval", x, "WolframExpression", "A Hold or a Delayed expression"]; Throw[$Failed]];
    If[value0 =!= 0, BlockchainMessage[BlockchainTransaction, "solcon"]; Throw[$Failed]];
    params = <|"file" -> File@FileNameJoin[{$SolidityTemplateDirectory, $SolidityTemplateV2Filename}]|>;
    constructorData = encodeParameters[{$WServerToken, Compress[x]}, {"string","string"}];
    result = ExplorerRequest[$ethPath, "compile", params];
    dataBin = result["bin"] <> constructorData;
    extra = <|"WolframExpression" -> x, "ByteCode" -> ByteArray[FromDigits[#, 16] & /@ StringPartition[dataBin, 2]]|>;
    <|"dataBin" -> dataBin, "extra" -> extra|>
]

processWolframExpressionReference[x_, value0_] := Module[{address, filename, filehash, params, constructorData, result, dataBin, extra},
    If[Head[x] =!= ExternalStorage`IPFSObject, BlockchainMessage[BlockchainTransaction, "invval", x, "WolframExpressionReference", "An IPFSObject"]; Throw[$Failed]];
    If[value0 =!= 0, BlockchainMessage[BlockchainTransaction, "solcon"]; Throw[$Failed]];
    address = First[x]["Address"];
    filename = With[{tmp = First[x]["FileName"]}, If[MissingQ[tmp], "", tmp]];
    filehash = With[{tmp = First[x]["FileHash"]}, If[MissingQ[tmp], "", tmp]];
    If[MissingQ[address], BlockchainMessage[BlockchainTransaction, "ipfsadd"]; Throw[$Failed]];
    params = <|"file" -> File@FileNameJoin[{$SolidityTemplateDirectory, $SolidityTemplateV1Filename}]|>;
    constructorData = encodeParameters[{$WServerToken, "IPFS", address, filename, filehash}, {"string","string","string","string","string"}];
    result = ExplorerRequest[$ethPath, "compile", params];
    dataBin = result["bin"] <> constructorData;
    extra = <|"WolframExpressionReference" -> x, "ByteCode" -> ByteArray[FromDigits[#, 16] & /@ StringPartition[dataBin, 2]]|>;
    <|"dataBin" -> dataBin, "extra" -> extra|>
]

processFunctionCall[x_] := Module[{func, inp, sender, inTypes, func0, id, paramsABI, dataBin, extra},
    If[!AssociationQ[x] && Head[x] =!= Typed, BlockchainMessage[BlockchainTransaction, "invval", x, "FunctionCall", "An association or a Typed expression"]; Throw[$Failed]];
    Switch[Head[x],
        Association,
            If[!MemberQ[{"Function", "Inputs", "Sender"}, #],
                (BlockchainMessage[BlockchainTransaction, "invelem", #, "\"FunctionCall\""]; Throw[$Failed])
            ] &/@ Keys[x];
            func = x["Function"];
            inp = x["Inputs"];
            sender = x["Sender"];
        ,Typed,
            func = x;
            inp = Missing["NotAvailable"];
            sender = Missing["NotAvailable"];
    ];
    If[!MissingQ[sender], If[!StringQ[sender], BlockchainMessage[BlockchainTransaction, "invval", sender, "Sender", "A string expression"]; Throw[$Failed], sender = StringDelete[sender, "0x"]]];
    If[MissingQ[func], 
        BlockchainMessage[BlockchainTransaction, "misselem", "Function", "\"FunctionCall\""]; Throw[$Failed],
        If[Head[func] =!= Typed, BlockchainMessage[BlockchainTransaction, "invval", func, "Function", "A Typed expression"]; Throw[$Failed],
            If[!MatchQ[func, Typed[_String, (___String | {_String...} | Rule[(_String | {_String...}), (_String | {_String...})])]],
                (BlockchainMessage[BlockchainTransaction, "invtyp", func, "Function"]; Throw[$Failed])
            ]
        ]
    ];
    inType = With[{tmp = Last[func]}, If[Head[tmp] === Rule, With[{tmp2 = Keys[tmp]}, If[ListQ[tmp2], tmp2, {tmp2}]], {}]];
    If[!MissingQ[inp], 
        If[If[ListQ[inp], Length[inp], 1] =!= Length[inType], 
            (BlockchainMessage[BlockchainTransaction, "abilen"]; Throw[$Failed])
        ];
        inp = If[validateEthereumAddress[#], StringDelete[#, "0x"], #] &/@ If[ListQ[inp], inp, {inp}];
        ,
        If[0 =!= Length[inType], (BlockchainMessage[BlockchainTransaction, "abiinp", "\"FunctionCall\""]; Throw[$Failed])]
    ];  
    If[!validateSolidityType[#], (BlockchainMessage[BlockchainTransaction, "abityp", #]; Throw[$Failed])] &/@ Flatten[{inType}];
    If[!MissingQ[inp], If[!AllTrue[Transpose[{inp, inType}], crossValidateSolidityInput[#[[1]], #[[2]]]&], 
        (BlockchainMessage[BlockchainTransaction, "abienc", inp, inType]; Throw[$Failed])
    ]];
    func0 = First[func] <> "(" <> StringJoin[If[Length[inType] > 1, Riffle[inType, ","], inType]] <> ")";    
    id = StringTake[Hash[func0, "Keccak256", "HexString"], 8];
    paramsABI = If[!MissingQ[inp], Quiet[Check[encodeParameters[inp, inType], $Failed]], ""];
    If[FailureQ[paramsABI], BlockchainMessage[BlockchainTransaction, "abienc", inp, inType]; Throw[$Failed]];
    dataBin = id <> paramsABI;
    extra = <|"FunctionCall" -> x, "ByteCode" -> ByteArray[FromDigits[#, 16] & /@ StringPartition[dataBin, 2]]|>;
    <|"dataBin" -> dataBin, "extra" -> extra, "sender" -> sender|>
]