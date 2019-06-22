Package["Blockchain`"]

(*Blockchain API server*)
PackageExport["Blockchain`$TemplateBase"];
Blockchain`$TemplateBase = Automatic;

Blockchain`$TemplateBase /: Set[HoldPattern[Blockchain`$TemplateBase], address: _String | Automatic] /; ! TrueQ[$set] := Block[{$set = True},
    setTemplateBase[address]
]

Blockchain`$TemplateBase /: SetDelayed[HoldPattern[Blockchain`$TemplateBase], address: _String | Automatic] /; ! TrueQ[$set] := Block[{$set = True},
    setTemplateBase[address]
]

setTemplateBase[address: _String | Automatic] := Module[{},
    Unprotect[Blockchain`$TemplateBase];
    Blockchain`$TemplateBase = address;
    Protect[Blockchain`$TemplateBase];
    address
]

Blockchain`$TemplateBase::invapiaddr = "Invalid template base `1`."

getTemplateBase[Automatic] := getTemplateBase["blockchain.tst1"]; (*it must be the proxy address*)

getTemplateBase[address: _String] := Switch[
    ToLowerCase[address],
    "blockchain.tst1", "http://blockchain-api-tst1.wolfram.com:8000",
    "blockchain.prd1",  "http://blockchain-api-prd1.wolfram.com:8000",
    "blockchain.proxy",  "http://blockchain-api.wolfram.com:8000",
    address, address,
    _, (Message[Blockchain`$TemplateBase::invapiaddr, address]; Throw[$Failed]);
]

getTemplateBase[address_] := (Message[Blockchain`$TemplateBase::invapiaddr, address]; Throw[$Failed]);

(*Authentication Values for CloudConnected equal to False *)
keyCloudConnectedFalse = SecuredAuthenticationKey[
             Association["ConsumerKey" -> "ConsumerKeyTest",
              "ConsumerSecret" -> "ConsumerSecretTest",
              "OAuthToken" -> "TokenTest",
              "OAuthTokenSecret" -> "TokenSecretTest",
              "OAuthVersion" -> "1.0a",
              "OAuthType" -> "OneLegged",
              "SignatureMethod" -> {"HMAC", "SHA1"},
              "TransmissionMethod" -> "Headers"]];

(* Changes the error messages to the correct function names*)
PackageScope[BlockchainMessage]
BlockchainMessage[tag_, errorcode_, params___] := With[
    {msg = MessageName[Blockchain, errorcode]},
    If[!StringQ[MessageName[tag, errorcode]], MessageName[tag, errorcode] = msg];
    Message[MessageName[tag, errorcode], params]
]

(* HTTP Requests for explorer functionalities*)
PackageScope[ExplorerRequest]
ExplorerRequest[blockchainPath_String, requestType_String, params:List[_Rule ...]|Association[_Rule ...]] := Block[
    {address = getTemplateBase[Blockchain`$TemplateBase] <> blockchainPath},

    Switch[requestType,
        "blockData",
            tag = "BlockchainBlockData";
            request = HTTPRequest[
                URLBuild[{address, "blockData"}],
                Association[
                    Method -> "GET",
                    "Query" -> params
                ]
            ]
        ,

        "blockchainData",
            tag = "BlockchainData";
            request = HTTPRequest[
                URLBuild[{address, "blockchainData"}],
                Association[
                    Method -> "GET", 
                    "Query" -> params
                ]
            ]
        ,

        "transactionData",
            tag = "BlockchainTransactionData";
            request = HTTPRequest[
                URLBuild[{address, "transactionData"}],
                Association[
                    Method -> "GET", 
                    "Query" -> params
                ]
            ]
        ,

        "estimateGas",
            tag = "BlockchainTransaction";
            request = HTTPRequest[
                URLBuild[{address, "estimateGas"}],
                Association[
                    Method -> "POST", 
                    "ContentType" -> "application/x-www-form-urlencoded",
                    "Body" -> params
                ]
            ]
        ,

        "transactionSend",
            tag = "BlockchainTransactionSubmit";
            request = HTTPRequest[
                URLBuild[{address, "transactionSubmit"}],
                Association[
                    Method -> "POST", 
                    "ContentType" -> "application/x-www-form-urlencoded",
                    "Body" -> params
                ]
            ]
        ,

        "addressData",
            tag = "BlockchainAddressData";
            request = HTTPRequest[
                URLBuild[{address, "addressData"}],
                Association[
                    Method -> "GET", 
                    "Query" -> params
                ]
            ]
        ,

         "tokenData",
            tag = "BlockchainTokenData";
            request = HTTPRequest[
                URLBuild[{address, "tokenData"}],
                Association[
                    Method -> "GET", 
                    "Query" -> params
                ]
            ]
        ,
        
        "compile",
            tag = "BlockchainTransaction";
            request = HTTPRequest[
                URLBuild[{address, "compile"}],
                Association[
                    Method -> "POST", 
                    "ContentType" -> "multipart/form-data",
                    "Body" -> params
                ]
            ]
        ,

        "compile2",
            tag = "BlockchainTransaction";
            request = HTTPRequest[
                URLBuild[{address, "compile2"}],
                Association[
                    Method -> "POST", 
                    "ContentType" -> "application/x-www-form-urlencoded",
                    "Body" -> params
                ]
            ]
        ,

        "compile3",
            tag = "BlockchainTransaction";
            request = HTTPRequest[
                URLBuild[{address, "compile3"}],
                Association[
                    Method -> "POST",
                    "ContentType" -> "multipart/form-data",
                    "Body" -> params
                ]
            ]
        ,

        "call",
            tag = "BlockchainContractValue";
            request = HTTPRequest[
                URLBuild[{address, "call"}],
                Association[
                    Method -> "GET", 
                    "Query" -> params
                ]
            ]
    ];

    If[$CloudConnected,
        result = URLRead[Internal`SignWolframHTTPRequest[request], Interactive -> False],
        result = URLRead[request, Authentication -> keyCloudConnectedFalse, Interactive -> False]
    ];

    Which[
        SameQ[result["StatusCode"], 200],
            (Developer`ReadRawJSONString[result["Body"]])["Result"] 
        ,

        MatchQ[result["StatusCode"], 401|429],
            BlockchainMessage[Symbol[tag], "errcon", result["StatusCode"], (Developer`ReadRawJSONString[result["Body"]])["Result"]];
            Throw[$Failed]
        ,

        MatchQ[result["StatusCode"], 400 | 404],
            Throw[Missing["NotAvailable"]]
        ,

         MatchQ[result["StatusCode"], 420],
            Throw[Missing["PendingToMine"]]
        ,

        MatchQ[result["StatusCode"], 527],
            BlockchainMessage[Symbol[tag], "etherr", Developer`ReadRawJSONString[result["Body"]]["Error"]];
            Throw[$Failed]
        ,

        MatchQ[result["StatusCode"], 528],
            BlockchainMessage[Symbol[tag], "btcerr", Developer`ReadRawJSONString[result["Body"]]["Error"]];
            Throw[$Failed]
        ,

        MatchQ[result["StatusCode"], 530],
            BlockchainMessage[Symbol[tag], "solerr", Developer`ReadRawJSONString[result["Body"]]["Error"]];
            Throw[$Failed]
        ,

        MatchQ[result["StatusCode"], 531],
            BlockchainMessage[Symbol[tag], "smerr", decodeParameters[StringDelete[Developer`ReadRawJSONString[result["Body"]]["Error"], "0x08c379a0"], "string"]];
            Throw[$Failed]
        ,

        MatchQ[result["StatusCode"], 540],
            BlockchainMessage[Symbol[tag], "invint", "BlockNumberInterval"];
            Throw[$Failed]
        ,

        True,
            Throw[$Failed]
    ]
]

(* Only allow supported BlockchainBases *)
PackageScope[getBlockchainBase]

getBlockchainBase[head_, Automatic] := Switch[head,
    BlockchainData, {"Bitcoin", "mainnet"},
    BlockchainBlockData, {"Bitcoin", "mainnet"},
    BlockchainTransactionData, {"Bitcoin", "mainnet"},
    BlockchainKeyEncode, {"Bitcoin", "mainnet"},
    BlockchainAddressData, {"Ethereum", "mainnet"},
    BlockchainTokenData, {"Ethereum", "mainnet"},
    _, {"Bitcoin", "mainnet"}
]

getBlockchainBase[head_, blockchain: _String | {_String} | {_String, _String}] := Switch[
	ToLowerCase[blockchain],
	"bitcoin"|{"bitcoin"}, {"Bitcoin", "mainnet"},
    "btc"|"btc.main"|{"btc"}|{"btc.main"}, {"Bitcoin", "mainnet"},
	{"bitcoin", "mainnet"}, {"Bitcoin", "mainnet"},
	{"bitcoin", "testnet"}, {"Bitcoin", "testnet"},
    "btc.test"|{"btc.test"}, {"Bitcoin", "testnet"},
	"ethereum"|{"ethereum"}, {"Ethereum", "mainnet"},
    "eth"|"eth.main"|{"eth"}|{"eth.main"}, {"Ethereum", "mainnet"},
	{"ethereum", "mainnet"}, {"Ethereum", "mainnet"},
	{"ethereum", "testnet"}, {"Ethereum", "testnet"},
    "eth.test"|{"eth.test"}, {"Ethereum", "testnet"},
    {"multichain", "wolfram"}, {"Multichain", "Wolfram"},
	_, (BlockchainMessage[head, "bbase", blockchain]; Throw[$Failed]);
]

getBlockchainBase[head_, blockchain_] := (BlockchainMessage[head, "bbase", blockchain]; Throw[$Failed]);

PackageScope[getBlockchainBaseContracts]

getBlockchainBaseContracts[head_, Automatic] = {"Ethereum", "mainnet"};

getBlockchainBaseContracts[head_, blockchain: _String | {_String} | {_String, _String}] := Switch[
    ToLowerCase[blockchain],
    "ethereum"|{"ethereum"}, {"Ethereum", "mainnet"},
    "eth"|"eth.main"|{"eth"}|{"eth.main"}, {"Ethereum", "mainnet"},
    {"ethereum", "mainnet"}, {"Ethereum", "mainnet"},
    {"ethereum", "testnet"}, {"Ethereum", "testnet"},
    "eth.test"|{"eth.test"}, {"Ethereum", "testnet"},
    _, (BlockchainMessage[head, "bbase", blockchain]; Throw[$Failed]);
]

getBlockchainBaseContracts[head_, blockchain_] := (BlockchainMessage[head, "bbase", blockchain]; Throw[$Failed]);

(*Converts single date into a range of unixtime values*)
PackageScope[dateToRange]
dateToRange[date_] := Switch[date[[2]],
   "Instant", {date, date},
   "Month", {DateObject[DateList[date]], DateObject[DateList[date]] + DateDifference[date, date + Quantity[1, "Month"]] - Quantity[1, "Seconds"]},
   _, {DateObject[DateList[date]], DateObject[DateList[date]] + Quantity[1, date[[2]]] - Quantity[1, "Seconds"]}];

(*Detects if a string is a HexString*)
PackageScope[hexStringQ]
hexStringQ[hex_] := If[StringQ[hex],And[EvenQ[StringLength[hex]], StringCases[hex, Except[Alternatives @@ Characters["0123456789abcdef"]]] === {}], False]

(*Functions for BlockchainTransactionObject*)

PackageScope[decamelize]
decamelize[str_String] := ToLowerCase[StringReplace[str, a_?LowerCaseQ ~~ b_?UpperCaseQ :> a <> " " <> b]];

PackageScope[toextrafield]
toextrafield[str_, value_] := {decamelize[str] <> ": ", value};
toextrafield[str_, value : _Integer] := {decamelize[str] <> ": ", Short[value, 0.25]};
toextrafield[str_, value : _String] := {decamelize[str] <> ": ", If[StringLength[value] >= 12, StringTake[value, 12] <> "...", value]};
toextrafield[str_, value : _Association] := {decamelize[str] <> ": ", "<|...|>"};
toextrafield[str_, value : List[_Association]] := {decamelize[str] <> ": ", "{<|...|>}"};
toextrafield[str_, value : List[_Association,__Association]] := {decamelize[str] <> ": ", "{<|...|>, ...}"};
toextrafield[str_, value : _Hold] := {decamelize[str] <> ": ", "Hold[...]"};
toextrafield[str_, value : _Delayed] := {decamelize[str] <> ": ", "Delayed[...]"};
toextrafield[str_, value : _?(Head[#] === ExternalStorage`IPFSObject &)] := {decamelize[str] <> ": ", "IPFSObject[...]"};
PackageScope[makeIcon]
makeIcon[data_, colorFunction_] := ArrayPlot[Partition[IntegerDigits[Hash[data, "SHA256"], 4, 64], 8], ImageSize -> 42, 
    ColorFunction -> colorFunction, ColorFunctionScaling -> False, Frame -> None, PlotRangePadding -> None];
    

PackageExport[Blockchain`ValidBlockchainBaseQ];

Blockchain`ValidBlockchainBaseQ[blockchainbase_] := MatchQ[Hold[blockchainbase],
	Hold[Automatic | base_String | {base_String} | {base_String, det_String}] 
        /; MatchQ[ToLowerCase[{base, det}], {}|{"ethereum"|"bitcoin"|"btc"|"btc.main"|"btc.test"|"eth"|"eth.main"|"eth.test"}|{"ethereum"|"bitcoin"|"multichain", "testnet"|"mainnet"|"wolfram"}]
];

(* WolframCloud-based request *)
$WCAPIPath = "Blockchain/api";


PackageScope[BlockchainCloudRequest]
BlockchainCloudRequest[request_String?StringQ, params : _List?ListQ : {}] := Block[
    {domain, result, resp, requestInfo, cc, tos, tag},

    tag = Switch[request,
        "getinfo", BlockchainData,
        "getblock", BlockchainBlockData,
        "gettransaction", BlockchainTransactionData,
        "getmetadatafromtransaction", BlockchainGet,
        "putdata", BlockchainPut];

    If[Not[StringQ[$CloudBase] && StringMatchQ[$CloudBase, ("https://www.test.wolframcloud.com" ~~ "" | "/") | ("https://www.wolframcloud.com" ~~ "" | "/")]],
        BlockchainMessage[tag, "invcb",$CloudBase];Throw[$Failed]
    ];

    requestInfo = If[Not[StringContainsQ[$CloudBase, "test"]],
        If[ !$CloudConnected,
            cc = CloudConnect[];
            If[!StringQ[cc],
                BlockchainMessage[tag, "notcon"];
                Throw[cc];
            ]
        ];

        tos = IntegratedServices`Private`tosApproved["Blockchain", $WolframUUID];

        If[ !tos,
            resp = IntegratedServices`CreateTOSDialogs["Blockchain", tag];
            If[!TrueQ[resp],
                If[MatchQ[resp, $Canceled],
                    BlockchainMessage[tag, "notos"];
                    Throw[resp]
                ,
                    Throw[$Failed]
                ]
            ]
        ];

        (*Need to remove ServiceCredits*)
        If[ !TrueQ[System`$ServiceCreditsAvailable > 0] && False(* TrueQ[scReq] *),
            resp = BlockchainPurchasingDialogs[tag];
            BlockchainMessage[tag, "nosc"];
            Throw[$Failed]
        ];

        Compress[<|
            "request" -> request,
            "options" -> params,
            "ClientInfo" -> IntegratedServices`Private`$clientinfo,
            "ClientCredits" -> Compress[<|
                "Quantity" -> IntegratedServices`Private`serviceCredits[$WolframUUID],
                "Timestamp" -> IntegratedServices`Private`serviceCreditTimestamp[$WolframUUID]
            |>]
        |>]
    ,
        Compress[<|
            "request" -> request,
            "options" -> params
        |>]
    ];

    result = URLRead[
        HTTPRequest[
           CloudObject["user:services-admin@wolfram.com/" <> $WCAPIPath]
           (* Replace[getBlockchainBaseValue[iBlockchainBase], Automatic :> $DefaultBlockchainBase] *),
            <|
                Method -> "POST", 
                "Body" -> <|
                    "requestInfo" -> requestInfo
                |>
            |>
        ],
        VerifySecurityCertificates -> Not[StringContainsQ[$CloudBase, "test"]]
    ];

    If[SameQ[result["StatusCode"], 200],
        result = Quiet[If[StringStartsQ[result["Body"], "<|"], ToExpression[result["Body"]], ImportString[result["Body"], "RawJSON"]]]; (*REMOVE TOEXPRESSION*)
        If[AssociationQ[result],
            If[!result["error"],
                If[Not[StringContainsQ[$CloudBase, "test"]] && NumericQ[result["info"]["AccountInformation"]["ServiceCreditsRemaining"]],
                    IntegratedServices`Private`updateAccountInfo[result["info"]]
                ];
                Uncompress[result["response"]]
            ,
                (*BlockchainMessage[$tagName, "errmsg1", Lookup[result, "msg", "Missing"]];
                Throw[$Failed, $WolframBlockchainTag]*)
                Throw[Missing["NotAvailable"]]
            ]
        ,
            BlockchainMessage[tag, "errcon0"];
            Throw[$Failed]
        ]
    ,
        BlockchainMessage[tag, "errcon0"];
        Throw[$Failed]
    ]
]