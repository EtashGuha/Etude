Package["Blockchain`"]

$btcPath = "/btc";

PackageScope[BlockchainDataBTC]
BlockchainDataBTC[network_String] := Module[
    {blockchainPath = $btcPath, requestType = "blockchainData", response},
    params = {"network" -> network};
    response = ExplorerRequest[blockchainPath, requestType, params];

    Association[
        "Type" -> "Bitcoin",
        "Name" -> Which[network === "mainnet", "BTC.main",
                        network === "testnet", "BTC.test"
                    ],        
        "Core" -> "Bitcoin",        
        "Blocks" -> response["Blocks"],
        "LatestHash" -> response["LatestHash"],
        "MinimumFee" -> Quantity[Round[Mean[response["MinimumFee"]*10^8]], "Satoshis"]
    ]
]

PackageScope[BlockchainBlockDataBTC]
BlockchainBlockDataBTC[block:First | Last | _Integer | _String, network_String] := Module[
    {blockchainPath = $btcPath, requestType = "blockData", response, params},

    params = Which[
        ToString[block] === "First", {"blockID" -> "0"},
        ToString[block] === "Last", {"blockID" -> "-1"},
        IntegerQ[block], {"blockID" -> ToString[block]},
        hexStringQ[StringDelete[block, "0x"]] && StringLength[StringDelete[block, "0x"]] === 64, {"blockID" -> StringDelete[block, "0x"]},
        True, (BlockchainMessage[BlockchainBlockData, "invprm", block, "block hash"]; Throw[$Failed])
    ];

    AppendTo[params, "network" -> network];

    response = ExplorerRequest[blockchainPath, requestType, params];

    Association[
        "BlockHash" -> response["BlockHash"],
        "BlockNumber" -> response["BlockNumber"],        
        "TimeMined" -> TimeZoneConvert[FromUnixTime[response["TimeMined"], TimeZone -> 0], $TimeZone],
        "ByteCount" -> response["ByteCount"],        
        "Nonce" -> response["Nonce"],
        "Version" -> response["Version"],
        "Confirmations" -> response["Confirmations"],
        "PreviousBlockHash" -> Lookup[response, "PreviousBlockHash", Missing["NotAvailable"]],
        "MerkleRoot" -> response["MerkleRoot"],
        "TransactionList" -> response["TransactionList"]
    ]
]

PackageScope[BlockchainTransactionDataBTC]
BlockchainTransactionDataBTC[txid_, network_String] := Module[
    {blockchainPath = $btcPath, requestType = "transactionData", response, inputs, outputs, totalbtc, inputAmounts, params, time, blockhash, txid0},
    
    txid0 = StringDelete[txid, "0x"];

    If[hexStringQ[txid0] && StringLength[txid0] === 64,
        params = {"transactionID" -> txid0, "network" -> network},
        BlockchainMessage[BlockchainTransactionData, "invprm", txid0,"transaction ID"]; Throw[$Failed]
    ];
    response = ExplorerRequest[blockchainPath, requestType, params];

    inputs = Lookup[response,"Inputs"];
    outputs = Lookup[response,"Outputs"];
    time = Lookup[response, "Time", Missing["NotAvailable"]];
    blockhash = Lookup[response, "BlockHash", Missing["NotAvailable"]];
    If[MissingQ[blockhash], BlockchainMessage[BlockchainTransactionData, "txnadd", txid0]];

    Association[
        "TransactionID" -> response["TransactionID"],
        "BlockHash" -> blockhash,
        "BlockNumber" -> Lookup[response, "BlockNumber", Missing["NotAvailable"]],
        "Confirmations" -> Lookup[response, "Confirmations", Missing["NotAvailable"]],
        "Time" -> If[!MissingQ[time], TimeZoneConvert[FromUnixTime[time, TimeZone -> 0], $TimeZone], time],
        "LockTime" -> response["LockTime"],
        "Version" -> response["Version"],
        "Amount" -> (
            totalbtc = Total[Round[#, 1/10^8]& /@ Lookup[outputs,"value"]];
            Quantity[N[totalbtc], "Bitcoins"]
        ),
        "Fee" -> If[MatchQ[inputAmounts = response["InputAmounts"],{}],
            Quantity[0, "Satoshis"], 
            Quantity[N[Total[Round[#*10^8]& /@ inputAmounts] - totalbtc*10^8], "Satoshis"]
        ],
        "ByteCount" -> response["ByteCount"],
        "Inputs" -> DeleteCases[(Association[
            "TransactionID" -> Lookup[#, "txid"],
            "Index" -> Lookup[#, "vout"],
            "ScriptByteArray" -> Which[
                KeyExistsQ[#, "scriptSig"], ByteArray[(FromDigits[#, 16] &) /@ StringPartition[Lookup[Lookup[#, "scriptSig"], "hex"], 2]],
                KeyExistsQ[#, "coinbase"], ByteArray[(FromDigits[#, 16] &) /@ StringPartition[Lookup[#, "coinbase"], 2]],
                True, Missing["KeyAbsent"]
            ],
            "ScriptString" -> Which[
                KeyExistsQ[#, "scriptSig"], Lookup[Lookup[#, "scriptSig"], "asm"],
                KeyExistsQ[#, "coinbase"], Lookup[#, "coinbase"],
                True, Missing["KeyAbsent"]
            ],
            "SequenceNumber" -> Lookup[#, "sequence"],
            "Addresses" -> Lookup[#, "addresses"],
            "SourceConfirmations" -> Lookup[#, "sourceConfirmations"]
        ]&) /@ inputs, Missing["KeyAbsent", ___], 2],
        "Outputs" -> (Association[
            "Amount" -> Quantity[N[Lookup[#, "value"]], "Bitcoins"],
            "ScriptByteArray" -> If[
                KeyExistsQ[#, "scriptPubKey"], ByteArray[(FromDigits[#, 16] &) /@ StringPartition[Lookup[Lookup[#, "scriptPubKey"], "hex"], 2]],
                Missing["NotAvailable"]
            ],
            "ScriptString" -> If[
                KeyExistsQ[#, "scriptPubKey"], Lookup[Lookup[#, "scriptPubKey"], "asm"],
                Missing["NotAvailable"]
            ],
            "Addresses" -> If[
                KeyExistsQ[#, "scriptPubKey"], Lookup[Lookup[#, "scriptPubKey"], "addresses", Missing["NotAvailable"]],
                Missing["NotAvailable"]
            ],
            "DestinationTransaction" -> Lookup[#, "DestinationTransaction"] /. {Null -> Missing["NotAvailable"]}
        ]&) /@ outputs
    ]
]

PackageScope[BlockchainKeyEncodeBTC]
BlockchainKeyEncodeBTC[key_, format: "Address" | "WIF", network_String] := Module[{pubkey, privKey = Missing[], compressQ = True},

    Switch[Head[key], 
        String,
            If[StringLength[key] === 64, 
                pubkey = Cryptography`GenerateEllipticCurvePublicKey[key , "secp256k1", "Compressed" -> True];
                privKey = key;
            ,   
                pubkey = key
            ]
        ,PublicKey,
            If[key["CurveName"] =!= "secp256k1", BlockchainMessage[BlockchainKeyEncode, "curvpub"]; Throw[$Failed]];
            pubkey = key["PublicKeyHexString"];
        ,PrivateKey,
            If[key["CurveName"] =!= "secp256k1", BlockchainMessage[BlockchainKeyEncode, "curvpri"]; Throw[$Failed]];
            privKey = key["PrivateKeyHexString"];
            pubkey = key["PublicKeyHexString"]
    ];
    Switch[StringLength[pubkey], 66, compressQ = True, 130, compressQ = False];

    Which[
        format === "Address", FromPublicKeyToAddress[pubkey, "Network" -> network, "Type" -> "P2PKH"],
        format === "WIF" && !MissingQ[privKey], FromPrivateKeyToWIF[privKey, "Network" -> network, "Compressed" -> compressQ],
        True, BlockchainMessage[BlockchainKeyEncode, "nopriv"]; Throw[$Failed]
    ]
]

PackageScope[BlockchainTransactionCreateBTC]
BlockchainTransactionCreateBTC[tx_Association, network_String] := Module[
    {newTX, prevOuts, rawTX, messHash, sourceAmounts, fee, amount, ver, lock}, 
    
    If[!MemberQ[{"BlockchainBase", "Version", "LockTime", "Inputs", "Outputs"}, #],
        (BlockchainMessage[BlockchainTransaction, "invelem", #, "input"]; Throw[$Failed])
    ] & /@ Keys[tx];
    
    inputs = Lookup[tx, "Inputs", (BlockchainMessage[BlockchainTransaction, "misselem", "Inputs", "input"]; Throw[$Failed])];
    outputs = Lookup[tx, "Outputs", (BlockchainMessage[BlockchainTransaction, "misselem", "Outputs", "input"]; Throw[$Failed])];
    If[!MatchQ[inputs, {__Association?(KeyExistsQ[#, "TransactionID"] && KeyExistsQ[#, "Index"] &)}],
        BlockchainMessage[BlockchainTransaction, "invtxa", "Inputs"]; Throw[$Failed]
    ];
    If[!MatchQ[outputs, {__Association?(KeyExistsQ[#, "Amount"] || KeyExistsQ[#, "Address"] || KeyExistsQ[#, "Data"] || KeyExistsQ[#, "ScriptString"] &)}],
        BlockchainMessage[BlockchainTransaction, "invtxa", "Outputs"]; Throw[$Failed]
    ];

    Which[
        !MatchQ[ver = Lookup[tx, "Version", 1], _Integer?Positive], BlockchainMessage[BlockchainTransaction, "invprm", ver,"\"Version\""]; Throw[$Failed],
        !MatchQ[lock = Lookup[tx, "LockTime", 0], _Integer?(# >= 0 &)], BlockchainMessage[BlockchainTransaction, "invprm", lock,"\"LockTime\""]; Throw[$Failed]
    ];
    
    (Function[inKey, 
        If[!MemberQ[{"TransactionID", "Index", "ScriptString", "SequenceNumber", "SignatureHash"}, inKey],
            (BlockchainMessage[BlockchainTransaction, "invelist", inKey, "Inputs"]; Throw[$Failed])
        ]
    ] /@ Keys[#];
    Which[
        !MatchQ[#["TransactionID"], _String?(hexStringQ[StringDelete[#, "0x"]] && StringLength[StringDelete[#, "0x"]] === 64 &)], BlockchainMessage[BlockchainTransaction, "invprm", #["TransactionID"],"\"TransactionID\""]; Throw[$Failed],
        !MatchQ[#["Index"], _Integer?(# >= 0 &)], BlockchainMessage[BlockchainTransaction, "invprm", #["Index"],"\"Index\""]; Throw[$Failed],
        !MatchQ[Lookup[#, "ScriptString", ""], _String], BlockchainMessage[BlockchainTransaction, "invprm", #["ScriptString"],"\"ScriptString\""]; Throw[$Failed],
        !MatchQ[Lookup[#, "SequenceNumber", 4294967295], _Integer?(# >= 0 &)], BlockchainMessage[BlockchainTransaction, "invprm", #["SequenceNumber"],"\"SequenceNumber\""]; Throw[$Failed]
    ])& /@ inputs;
    
    (Function[outKey, 
        If[!MemberQ[{"Amount", "Address", "ScriptString", "Data"}, outKey],
            (BlockchainMessage[BlockchainTransaction, "invelist", outKey, "Outputs"]; Throw[$Failed])
        ]
    ] /@ Keys[#];
    Which[
        !MatchQ[Lookup[#, "ScriptString", ""], _String], BlockchainMessage[BlockchainTransaction, "invprm", #["ScriptString"],"\"ScriptString\""]; Throw[$Failed],
        !MatchQ[Lookup[#, "Address", ""], _String], BlockchainMessage[BlockchainTransaction, "invprm", #["Address"],"\"Address\""]; Throw[$Failed]
    ])& /@ outputs;

    (*Change output scripts*)
    outputs = Association[
        "Amount" -> (amount = Lookup[#, "Amount", 0];
            Switch[amount,
                _Integer?(# >= 0 &)|_Real?(# >= 0 &), Quantity[amount, "Bitcoins"],
                Quantity[_Integer?(# >= 0 &), "Bitcoins"]|Quantity[_Real?(# >= 0 &), "Bitcoins"], amount,
                Quantity[_Integer?(# >= 0 &), "Satoshis"]|Quantity[_Real?(# >= 0 && FractionalPart[#] == 0 &), "Satoshis"], N[CurrencyConvert[amount, "Bitcoins"]],
                _, BlockchainMessage[BlockchainTransaction, "invprm", amount,"\"Amount\""]; Throw[$Failed]
            ]),
        ReleaseHold[Which[
            KeyExistsQ[#, "Address"] && StringStartsQ[#["Address"], "1" | "m" | "n"], Hold[Sequence["Address" -> #["Address"], "ScriptString" -> "OP_DUP OP_HASH160 " <> AddressToPublicKeyHash[#["Address"]] <> " OP_EQUALVERIFY OP_CHECKSIG"]],
            KeyExistsQ[#, "Address"] && StringStartsQ[#["Address"], "3" | "2"], Hold[Sequence["Address" -> #["Address"], "ScriptString" -> "OP_HASH160 " <> AddressToPublicKeyHash[#["Address"]] <> " OP_EQUAL"]],
            KeyExistsQ[#, "Address"], BlockchainMessage[BlockchainTransaction, "addsup", #["Address"]]; Throw[$Failed],
            KeyExistsQ[#, "Data"] && StringQ[#["Data"]], Hold[Sequence["Data" -> #["Data"], "ScriptString" -> "OP_RETURN " <> StringJoin[IntegerString[ToCharacterCode[#["Data"]], 16, 2]]]],
            KeyExistsQ[#, "Data"], Hold[Sequence["Data" -> #["Data"], "ScriptString" -> "OP_RETURN " <> StringJoin[IntegerString[ToCharacterCode[Compress[BinarySerialize[#["Data"]], Method -> {"Version" -> 3}]], 16, 2]]]],
            True, "ScriptString" -> ToString[Lookup[#, "ScriptString", ""]]
        ]]
    ]& /@ outputs;
    
    (*Change input scripts*)
    inputs = Association[
        "TransactionID" -> StringDelete[#["TransactionID"], "0x"],
        "Index" -> #["Index"],
        "SequenceNumber" -> Lookup[#, "SequenceNumber", 4294967295],
        ReleaseHold[Which[
            MatchQ[ToString[#["SignatureHash"]], "All" | "None" | "Single" | "AllAnyoneCanPay" | "NoneAnyoneCanPay" | "SingleAnyoneCanPay"], Hold[Sequence["SignatureHash" -> ToString[#["SignatureHash"]], "ScriptString" -> Lookup[#, "ScriptString", ""]]],
            KeyExistsQ[#, "SignatureHash"], BlockchainMessage[BlockchainTransaction, "invprm", #["SignatureHash"],"\"SignatureHash\""]; Throw[$Failed],
            MatchQ[#["ScriptString"], ""] || Not[KeyExistsQ[#, "ScriptString"]], Hold[Sequence["SignatureHash" -> "All", "ScriptString" -> ""]],
            True, "ScriptString" -> Lookup[#, "ScriptString", ""]
        ]]
    ]& /@ inputs;
    newTX = Association[
        "BlockchainBase" -> {"Bitcoin", Switch[network, "mainnet", "MainNet", "testnet", "TestNet"]},
        "Signed" -> False,
        "Version" -> ver,
        "LockTime" -> lock,
        "Inputs" -> inputs,
        "Outputs" -> outputs
    ];

    (*Calculate fee*)
    sourceAmounts = Function[{txid, index}, 
        prevOuts = Catch[BlockchainTransactionDataBTC[txid, network]["Outputs"]];
        Which[
            FailureQ[prevOuts], Throw[prevOuts],
            prevOuts == Missing["NotAvailable"], BlockchainMessage[BlockchainTransaction, "txnnet", txid, {"Bitcoin", Switch[network, "mainnet", "MainNet", "testnet", "TestNet"]}]; Throw[Missing["NotAvailable"]]
        ];

        If[(index+1) > Length[prevOuts], BlockchainMessage[BlockchainTransaction, "invsti", index, txid]; Throw[$Failed], prevOuts[[index + 1]]["Amount"]]
    ] @@@ Lookup[inputs, {"TransactionID", "Index"}];
    fee = Total[QuantityMagnitude /@ sourceAmounts] - Total[QuantityMagnitude /@ Lookup[outputs, "Amount", 0.]];
    If[fee < 0, BlockchainMessage[BlockchainTransaction, "insfam"]; Throw[$Failed]];
    fee = Quantity[Round[fee*10^8], "Satoshis"];
    newTX = Insert[newTX, "Fee" -> fee, 3];

    rawTX = TransactionSerializeBTC[newTX];
    messHash = Hash[ByteArray[FromDigits[#, 16] & /@ StringPartition[rawTX, 2]], "SHA256SHA256", "HexString"];
    Append[newTX,{"MessageHash" -> messHash, "RawTransaction" -> rawTX}]
]

PackageScope[BlockchainTransactionSignBTC]
BlockchainTransactionSignBTC[obj_BlockchainTransaction, privKey: _String | _PrivateKey | List[__?(StringQ[#] || MatchQ[#, _PrivateKey] &)], network_String] := Module[
    {copyTX = obj[[1]], inputs = obj["Inputs"], privateKeysHex = If[ListQ[privKey], privKey, {privKey}], messageAndSourcePKH, signature, publicKeys, sigHash, publicKeysHash, match},
    
    privateKeysHex = If[StringQ[#],
            If[hexStringQ[StringDelete[#, "0x"]] && StringLength[StringDelete[#, "0x"]] === 64, {StringDelete[#, "0x"], True}, WifToPrivateKeyHex[#]],
            {#["PrivateKeyHexString"], #["PublicKeyHexString"]}
        ] & /@ privateKeysHex;
    publicKeys = privateKeysHex /. {List[a_String, b_?BooleanQ] :> List[a, Cryptography`GenerateEllipticCurvePublicKey[a , "secp256k1", "Compressed" -> b]]};
    publicKeysHash = {#1, #2, {HashPubKeyP2PKH[#2]}} & @@@ DeleteDuplicates[publicKeys];
    inputs = MapIndexed[
        If[Not[KeyExistsQ[#1, "SignatureHash"] && MatchQ[Lookup[#1, "ScriptString", ""], ""]],
            #1,
            sigHash = #1["SignatureHash"] /. {"All" -> "ALL", "None" -> "NONE", "Single" -> "SINGLE", "AllAnyoneCanPay" -> "ALL|ANYONECANPAY","NoneAnyoneCanPay" -> "NONE|ANYONECANPAY", "SingleAnyoneCanPay" -> "SINGLE|ANYONECANPAY"};
            messageAndSourcePKH = MessageFromTransactionBTC[copyTX, First[#2] - 1, network, sigHash];
            match = Cases[publicKeysHash, {_, _, Last[messageAndSourcePKH]}];
            If[match === {}, BlockchainMessage[BlockchainTransactionSign, "invpvk",#1["Index"],#1["TransactionID"]]; Throw[$Failed]];
            signature = GenerateDigitalSignature[MessageDigestBTC[First[messageAndSourcePKH]], First[match][[1]], Method -> <| "Type" -> "EllipticCurve", "CurveName" -> "Bitcoin", "SignatureType" -> "Deterministic", "HashingMethod" -> None|>]["DER"];
            Append[#1, "ScriptString" -> StringJoin[signature, "[", sigHash, "] ", First[match][[2]]]]
        ] &, inputs];
    copyTX["Inputs"] = inputs;
    copyTX["Signed"] = True;
    copyTX["RawTransaction"] = TransactionSerializeBTC[copyTX];
    BlockchainTransaction[copyTX]
]

PackageScope[BlockchainTransactionSendBTC]
BlockchainTransactionSendBTC[obj_BlockchainTransaction, network_String] := Module[
    {blockchainPath = $btcPath, requestType = "transactionSend", response},
    
    params = <|"network" -> network, "rawTx" -> obj["RawTransaction"]|>;
    response = ExplorerRequest[blockchainPath, requestType, params];
    BlockchainTransaction @ Join[First[obj], <|"TransactionID" -> response|>]
]

PackageScope[BlockchainAddressDataBTC]
BlockchainAddressDataBTC[address: (_String | _Association), prop_, network_String, maxitems_String] := Module[
    {blockchainPath = $btcPath, requestType = "addressData", response, blocks, time, from, to, address0, params, balance},

    Switch[Head[address],
        String,
            address0 = address;
            blocks = "all";
            time = "all";
        ,Association,
            If[!KeyExistsQ[address,"Address"], 
                (BlockchainMessage[BlockchainAddressData, "missadd"]; Throw[$Failed]),
                address0 = address["Address"];
                time = With[{tmp = address["TimeInterval"]}, If[MissingQ[tmp] || tmp === All, "all", tmp]];
                blocks = With[{tmp = address["Blocks"]}, If[MissingQ[tmp] || tmp === All, "all", tmp]];
            ];
    ];

    AddressToPublicKeyHash[address0, BlockchainAddressData];

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
                    from = "unixtime:" <> ToString @ UnixTime[First @ dateToRange[time]];
                    to = "unixtime:" <> ToString @ UnixTime[Last @ dateToRange[time]];
                ,
                List,
                    from = "unixtime:" <> ToString @ UnixTime[First @ dateToRange[First @ time]];
                    to = "unixtime:" <> ToString @ UnixTime[First @ dateToRange[Last @ time]];
            ];
        ,
        {_?(!MatchQ[#,"all"]&), "all"},
            Switch[Head[blocks],
                Integer,
                    from = ToString @ blocks;
                    to = ToString @ blocks;
                ,
                List,
                    If[First @ blocks === First, from = "0", from = ToString @ First @ blocks];
                    If[Last @ blocks === Last, to = "-1", to = ToString @ Last @ blocks];
            ];
    ];

    params = {"address" -> address0, "network" -> network, "from" -> from, "to" -> to, "maxItems" -> maxitems};
    response = ExplorerRequest[blockchainPath, requestType, params];

    balance = Quantity[Total[Apply[(Total[Round[#2, 1./10^8]] - Total[Round[#1, 1./10^8]]) &, response["Balance"], {1}]], "Bitcoins"];

    response0 = DeleteMissing[
        Association[
            "Balance" -> balance,
            "ValueInUSDollars" -> CurrencyConvert[balance, "USDollars"],
            "TotalTransactions" -> response["Transactions"] /.{0 -> Missing["NotAvailable"]},
            "TransactionsList" -> (Association[
                    "TransactionID" -> #["txid"],
                    "BlockNumber" -> #["blocknumber"],
                    "Time" -> TimeZoneConvert[FromUnixTime[#["blocktime"], TimeZone -> 0], $TimeZone],
                    "Inputs" -> (Function[x, <|"Addresses"->(x["addresses"]/.Missing[__]->Missing["NotAvailable"]),"Amount"->Quantity[Round[(x["value"]/.Missing[__]->0), 1./10^8], "Bitcoins"]|> ] /@ #["vin"]), 
                    "Outputs" -> (Function[x, <|"Addresses"->(x["addresses"]/.Missing[__]->Missing["NotAvailable"]),"Amount"->Quantity[Round[(x["value"]/.Missing[__]->0), 1./10^8], "Bitcoins"]|> ] /@ #["vout"])
                ] &/@ response["TransactionsList"]) /.{{} -> Missing["NotAvailable"]}
        ]
    ];

    If[prop === {}, response0, Lookup[response0,prop]]

]