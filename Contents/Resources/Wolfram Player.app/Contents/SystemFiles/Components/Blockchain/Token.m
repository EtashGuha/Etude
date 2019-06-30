Package["Blockchain`"]

$ethPath = "/eth";

Options[BlockchainTokenData] = {BlockchainBase :> $BlockchainBase, MaxItems -> 20}

BlockchainTokenData[All, prop: (__String | {__String..} | All): {}, opt: OptionsPattern[BlockchainTokenData]] := BlockchainTokenData[<||>, prop, opt]

BlockchainTokenData[tokenInput: (_String | _Association), prop: (__String | {__String..} | All): {}, opt: OptionsPattern[BlockchainTokenData]] := Catch[Module[
	{
		blockchainPath = $ethPath, requestType = "tokenData", response, params, maxitems, time, blocks, from, to, output, name, symbol, address, standard, addressFilter, 
		senderFilter, receiverFilter, holderFilter, response0, holdersOut, holdersAPI, transfersOut, transfersAPI, sendersOut, sendersAPI, receiversOut, receiversAPI, tokenInput0
	},

	With[{blockchain = getBlockchainBase[BlockchainTokenData, OptionValue[BlockchainBase]]}, 
		If[!MatchQ[blockchain,{"Ethereum", "mainnet"}], 
			(BlockchainMessage[BlockchainTokenData, "invnet", blockchain]; Throw[$Failed])
		]
	];

	Switch[OptionValue[MaxItems],
		All, maxitems = "0",
		_Integer?(! MatchQ[#, 0] &), maxitems = ToString @ OptionValue[MaxItems],
		_String?(! MatchQ[#, "0"] &), maxitems = OptionValue[MaxItems],
		_, BlockchainMessage[BlockchainTokenData, "invopt"]; Throw[$Failed]
	];

	If[Head[tokenInput] === Association,

		If[!MemberQ[{"Name", "Symbol", "TokenAddress", "TokenStandard", "TimeInterval", "BlockNumberInterval", "Addresses", "SenderAddresses", "ReceiverAddresses", "HolderAddresses"}, #],
			(BlockchainMessage[BlockchainTokenData, "invelem", #, "input"]; Throw[$Failed])
		] &/@ Keys[tokenInput];

		output = Which[
			AnyTrue[tokenInput[#] &/@ {"TimeInterval", "BlockNumberInterval", "Addresses", "SenderAddresses", "ReceiverAddresses"}, !MissingQ[#]&], "txs",
			!MissingQ[tokenInput["HolderAddresses"]], "holderInfo",
			True, "tokenInfo"
		];

		name = With[{tmp = tokenInput["Name"]}, If[MissingQ[tmp] || tmp === All, "all", 
			If[MatchQ[tmp, _String | {_String..}], tmp, (BlockchainMessage[BlockchainTokenData, "invval", tmp, "Name", "A string expression"]; Throw[$Failed])]
		]];
		symbol = With[{tmp = tokenInput["Symbol"]}, If[MissingQ[tmp] || tmp === All, "all", 
			If[MatchQ[tmp, _String | {_String..}], tmp, (BlockchainMessage[BlockchainTokenData, "invval", tmp, "Symbol", "A string expression"]; Throw[$Failed])]
		]];
		address = With[{tmp = tokenInput["TokenAddress"]}, If[MissingQ[tmp] || tmp === All, "all",
			If[validateEthereumAddress[#], "0x" <> StringDelete[ToLowerCase[#], "0x"], (BlockchainMessage[BlockchainTokenData, "ethadd", #]; Throw[$Failed])] &/@ If[ListQ[tmp], tmp, {tmp}]			
		]];
		standard = With[{tmp = tokenInput["TokenStandard"]}, If[MissingQ[tmp] || tmp === All, "all", 
			If[MatchQ[tmp, _String | {_String..}],
				If[StringMatchQ[ToLowerCase[#], StartOfString ~~ "erc" ~~ DigitCharacter .. ~~ EndOfString], #, (BlockchainMessage[BlockchainTokenData, "invtkn", #]; Throw[$Failed])] &/@ If[ListQ[tmp], tmp, {tmp}],
				(BlockchainMessage[BlockchainTokenData, "invval", tmp, "TokenStandard", "A string expression"]; Throw[$Failed])
			]
		]];
		time = With[{tmp = tokenInput["TimeInterval"]}, If[MissingQ[tmp] || tmp === All, "all",
			If[MatchQ[tmp, _DateObject | {_DateObject, _DateObject}], tmp, (BlockchainMessage[BlockchainTokenData, "invval", tmp, "TimeInterval", "A DateObject or a list of two DateObjects"]; Throw[$Failed])]
		]];
		blocks = With[{tmp = tokenInput["BlockNumberInterval"]}, If[MissingQ[tmp] || tmp === All, "all", 
			If[MatchQ[tmp, {_Integer | First | "First", _Integer | Last | "Last"}], tmp, (BlockchainMessage[BlockchainTokenData, "invval", tmp, "BlockNumberInterval", "A list of two integers"]; Throw[$Failed])]
		]];
		addressFilter = With[{tmp = tokenInput["Addresses"]}, If[MissingQ[tmp] || tmp === All, "all", 
			If[validateEthereumAddress[#], "0x" <> StringDelete[ToLowerCase[#], "0x"], (BlockchainMessage[BlockchainTokenData, "ethadd", #]; Throw[$Failed])] &/@ If[ListQ[tmp], tmp, {tmp}]
		]];
		senderFilter = With[{tmp = tokenInput["SenderAddresses"]}, If[MissingQ[tmp] || tmp === All, "all",
			If[validateEthereumAddress[#], "0x" <> StringDelete[ToLowerCase[#], "0x"], (BlockchainMessage[BlockchainTokenData, "ethadd", #]; Throw[$Failed])] &/@ If[ListQ[tmp], tmp, {tmp}]
		]];
		receiverFilter = With[{tmp = tokenInput["ReceiverAddresses"]}, If[MissingQ[tmp] || tmp === All, "all",
			If[validateEthereumAddress[#], "0x" <> StringDelete[ToLowerCase[#], "0x"], (BlockchainMessage[BlockchainTokenData, "ethadd", #]; Throw[$Failed])] &/@ If[ListQ[tmp], tmp, {tmp}]
		]];
		holderFilter = With[{tmp = tokenInput["HolderAddresses"]}, If[MissingQ[tmp] || tmp === All, "all",
			If[validateEthereumAddress[#], "0x" <> StringDelete[ToLowerCase[#], "0x"], (BlockchainMessage[BlockchainTokenData, "ethadd", #]; Throw[$Failed])] &/@ If[ListQ[tmp], tmp, {tmp}]
		]];

		If[output === "txs" && holderFilter =!= "all",
			If[addressFilter === "all",
				addressFilter = holderFilter,
				BlockchainMessage[BlockchainTokenData, "dupfil"]
			]
		];

		Switch[{blocks, time},
			{"all", "all"},
				from = "all";
				to = "all";
			,
			{_?(!MatchQ[#,"all"]&), _?(!MatchQ[#,"all"]&)},
				BlockchainMessage[BlockchainTokenData, "invtime"]; 
				Throw[$Failed];
			,
			{"all", _?(!MatchQ[#,"all"]&)},
				Switch[Head[time],
					DateObject,
						from = "unixtime:" <> ToString @ UnixTime[First @ dateToRange[time]] <> "000";
						to = "unixtime:" <> ToString @ UnixTime[Last @ dateToRange[time]] <> "000";
					,List,
						If[Last[time] < First[time], (BlockchainMessage[BlockchainTokenData, "invint", "TimeInterval"]; Throw[$Failed])];
						from = "unixtime:" <> ToString @ UnixTime[First @ dateToRange[First @ time]] <> "000";
						to = "unixtime:" <> ToString @ UnixTime[First @ dateToRange[Last @ time]] <> "000";
				];
			,
			{_?(!MatchQ[#,"all"]&), "all"},
				If[ToString @ First @ blocks === "First", from = "1", from = ToString @ First @ blocks];
				If[ToString @ Last @ blocks === "Last", to = "-1", to = ToString @ Last @ blocks];
		];
		,
		tokenInput0 = If[validateEthereumAddress[tokenInput], "0x" <> StringDelete[tokenInput, "0x"], tokenInput];
		output = "tokenInfo";
		If[tokenInput === "ERC20", Return[BlockchainTokenData[<|"TokenStandard" -> "ERC20"|>, prop, opt]]];
		If[tokenInput === "ERC721", Return[BlockchainTokenData[<|"TokenStandard" -> "ERC721"|>, prop, opt]]];
	];

	If[MemberQ[prop, "HolderList"] || prop === "HolderList",
        holdersOut = "true";
        holdersAPI = "true";
        ,
        holdersOut = "false";
        holdersAPI = "false";
    ];
    If[MemberQ[prop, "TransferList"] || prop === "TransferList", 
        transfersOut = "true";
        transfersAPI = "true";
        ,
        transfersOut = "false";
        transfersAPI = "false";
    ];
    If[MemberQ[prop, "SenderAddressList"] || prop === "SenderAddressList", 
        sendersOut = "true";
        sendersAPI = "true";
        ,
        sendersOut = "false";
        sendersAPI = "false";
    ];
    If[MemberQ[prop, "ReceiverAddressList"] || prop === "ReceiverAddressList", 
        receiversOut = "true";
        receiversAPI = "true";
        ,
        receiversOut = "false";
        receiversAPI = "false";
    ];

	params = Join[Switch[Head[tokenInput],
		String,{
			"tokenID" -> tokenInput0
		},
		Association, Join[
			If[ListQ[name], Thread["name" -> name], {"name" -> name}],
			If[ListQ[symbol], Thread["symbol" -> symbol], {"symbol" -> symbol}],
			If[ListQ[address], Thread["address" -> address], {"address" -> address}],
			If[ListQ[standard], Thread["standard" -> standard], {"standard" -> standard}],
			{"from" -> from, "to" -> to},
			If[ListQ[addressFilter], Thread["addressFilter" -> addressFilter], {"addressFilter" -> addressFilter}],
			If[ListQ[senderFilter], Thread["senderFilter" -> senderFilter], {"senderFilter" -> senderFilter}],
			If[ListQ[receiverFilter], Thread["receiverFilter" -> receiverFilter], {"receiverFilter" -> receiverFilter}],
			If[ListQ[holderFilter], Thread["holderFilter" -> holderFilter], {"holderFilter" -> holderFilter}]
		]]
		,{
			"output" -> output,
			"maxItems" -> maxitems,
			"holdersAPI" -> holdersAPI,
        	"transfersAPI" -> transfersAPI,
        	"sendersAPI" -> sendersAPI,
        	"receiversAPI" -> receiversAPI
		}
	];

    response = ExplorerRequest[blockchainPath, requestType, params];

    response0 = Switch[output,
    	"tokenInfo",
    		DeleteMissing[Association[
    			"Name" -> #["Name"],
				"Symbol" -> #["Symbol"],
				"TokenAddress" -> checksum[ToLowerCase[StringDelete[#["Address"], "0x"]]],
				"TokenStandard" -> With[{erc = #["ERC"]}, If[ListQ[erc], "ERC" <> ToString[#] &/@ Sort[erc], "ERC" <> ToString[erc]]],
				"TotalSupply" -> ToExpression[#["TotalSupply"]],
				"TransferPrecision" -> #["Decimals"],
				"HoldersCount" -> #["HoldersCount"],
				"AddressesCount" -> #["AddressesCount"],
				"TransfersCount" -> #["TransfersCount"],
				"TransferList" -> If[KeyExistsQ[#, "Transfers"], 
					DeleteMissing[<|
						"TransactionID" -> StringDelete[#["TransactionHash"], "0x"], 
						"BlockNumber" -> #["BlockNumber"], 
						"Time" -> FromUnixTime[UnixTime[#["DateTime"], TimeZone -> 0]], 
						"Sender" -> checksum[ToLowerCase[StringDelete[#["From"], "0x"]]], 
						"Receiver" -> checksum[ToLowerCase[StringDelete[#["To"], "0x"]]], 
						"TokenID" -> With[{tmp = #["TokenID"]}, If[!MissingQ[tmp], If[MatchQ[tmp, <|"$numberDecimal" -> _|>], ToExpression[First[tmp]], ToExpression[tmp]], tmp]],
						"Amount" -> With[{tmp = #["Amount"]}, If[!MissingQ[tmp], ToExpression[First[tmp]], tmp]]
					|>] &/@ #["Transfers"]
					, Missing["NotAvailable"]
				],
				"HolderList" -> If[KeyExistsQ[#, "Holders"], (<|
					"HolderAddress" -> checksum[ToLowerCase[StringDelete[#[[1]]["Account"], "0x"]]],
					"Balance" -> If[#[[3]] === 721, #[[1]]["Balance"], ToExpression @ First @ #[[1]]["Balance"]],
					"BalanceFraction" -> If[#[[3]] === 721,
							If[ToExpression[#[[2]]] =!= 0, Quantity[N[#[[1]]["Balance"]/ToExpression[#[[2]]]]*100,"Percent"], Quantity[0,"Percent"]],
							If[ToExpression[#[[2]]] =!= 0, Quantity[N[ToExpression[First[#[[1]]["Balance"]]]/ToExpression[#[[2]]]]*100,"Percent"], Quantity[0,"Percent"]]
						]			
					|> &/@ Release[Thread[{#["Holders"],#["TotalSupply"],Hold[#["ERC"]]}]])
					, Missing["NotAvailable"]
				],
				"SenderAddressList" -> If[KeyExistsQ[#, "Senders"], checksum[ToLowerCase[StringDelete[#, "0x"]]] &/@ #["Senders"], Missing["NotAvailable"]],
				"ReceiverAddressList" -> If[KeyExistsQ[#, "Receivers"], checksum[ToLowerCase[StringDelete[#, "0x"]]] &/@ #["Receivers"], Missing["NotAvailable"]]
    		]] &/@ response
    	,"holderInfo",
			With[{single = Length[response] === 1}, 
				DeleteMissing[Association[
					"HolderAddress" -> If[single, Missing["NotAvailable"], checksum[ToLowerCase[StringDelete[#["Account"], "0x"]]]],
	    			"Name" -> #["Name"],
					"Symbol" -> #["Symbol"],
					"TokenAddress" -> checksum[ToLowerCase[StringDelete[#["Address"], "0x"]]],
					"TokenStandard" -> With[{erc = #["ERC"]}, If[ListQ[erc], "ERC" <> ToString[#] &/@ Sort[erc], "ERC" <> ToString[erc]]],
					"TransfersCount" -> #["TransfersCount"],
					"Balance" -> With[{erc = #["ERC"], bal = #["Balance"]}, If[erc === 721, bal, ToExpression @ First @ bal]],
					"HolderTokenIDs" -> #["Inventory"],
					"BalanceFraction" -> With[{erc = #["ERC"], total = #["TotalSupply"], bal = #["Balance"]}, 
						If[erc === 721,
							If[ToExpression[total] =!= 0, Quantity[N[bal/ToExpression[total]]*100,"Percent"], Quantity[0,"Percent"]],
							If[ToExpression[total] =!= 0, Quantity[N[ToExpression[First[bal]]/ToExpression[total]]*100,"Percent"], Quantity[0,"Percent"]]
						]],
					"TransferList" -> If[KeyExistsQ[#, "Transfers"], 
						DeleteMissing[<|
							"TransactionID" -> StringDelete[#["TransactionHash"], "0x"], 
							"BlockNumber" -> #["BlockNumber"], 
							"Time" -> FromUnixTime[UnixTime[#["DateTime"], TimeZone -> 0]], 
							"Sender" -> checksum[ToLowerCase[StringDelete[#["From"], "0x"]]], 
							"Receiver" -> checksum[ToLowerCase[StringDelete[#["To"], "0x"]]], 
							"TokenID" -> With[{tmp = #["TokenID"]}, If[!MissingQ[tmp], If[MatchQ[tmp, <|"$numberDecimal" -> _|>], ToExpression[First[tmp]], ToExpression[tmp]], tmp]],
							"Amount" -> With[{tmp = #["Amount"]}, If[!MissingQ[tmp], ToExpression[First[tmp]], tmp]]
						|>] &/@ #["Transfers"]
						, Missing["NotAvailable"]
					],
					"SenderAddressList" -> If[KeyExistsQ[#, "Senders"], checksum[ToLowerCase[StringDelete[#, "0x"]]] &/@ #["Senders"], Missing["NotAvailable"]],
					"ReceiverAddressList" -> If[KeyExistsQ[#, "Receivers"], checksum[ToLowerCase[StringDelete[#, "0x"]]] &/@ #["Receivers"], Missing["NotAvailable"]]
	    		]] &/@ Flatten[Lookup[response, "data"]]
	    	]
    	,"txs",
    		With[{txs = DeleteMissing[<|
    				"Name" -> #["Name"],
    				"TokenStandard" -> With[{erc = #["ERC"]}, If[ListQ[erc], "ERC" <> ToString[#] &/@ Sort[erc], "ERC" <> ToString[erc]]],
					"TransactionID" -> StringDelete[#["TransactionHash"], "0x"], 
					"BlockNumber" -> #["BlockNumber"], 
					"Time" -> FromUnixTime[UnixTime[#["DateTime"], TimeZone -> 0]], 
					"Sender" -> checksum[ToLowerCase[StringDelete[#["From"], "0x"]]], 
					"Receiver" -> checksum[ToLowerCase[StringDelete[#["To"], "0x"]]], 
					"TokenID" -> With[{tmp = #["TokenID"]}, If[!MissingQ[tmp], If[MatchQ[tmp, <|"$numberDecimal" -> _|>], ToExpression[First[tmp]], ToExpression[tmp]], tmp]],
					"Amount" -> With[{tmp = #["Amount"]}, If[!MissingQ[tmp], ToExpression[First[tmp]], tmp]]
				|>] &/@ ReverseSortBy[response, #["BlockNumber"]&]},
				If[maxitems === "0" || Length[txs] < ToExpression[maxitems], txs, Take[txs, ToExpression[maxitems]]]
			]
    ];

	If[SameQ[prop, {}] || SameQ[prop, All],
    	response0,        
    	Lookup[response0, prop, Missing["NotAvailable"]]
    ]

]]

BlockchainTokenData[arg1_, arg2 : RepeatedNull[_,1], opt : OptionsPattern[BlockchainTokenData]] := If[MatchQ[{arg2}, ({_String} | {{___String}} | {})],
       BlockchainMessage[BlockchainTokenData, "invarg", "A token specification given by a string or an association", 1]; $Failed,
       BlockchainMessage[BlockchainTokenData, "invarg", "A property", 2]; $Failed
]

BlockchainTokenData[arg___, opt : OptionsPattern[BlockchainTokenData]] := (BlockchainMessage[BlockchainTokenData, "argn", "BlockchainTokenData", Length[{arg}], "1 or 2"]; $Failed)
