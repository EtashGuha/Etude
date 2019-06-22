Package["Blockchain`"]

PackageScope[BlockchainDataWL]
BlockchainDataWL[] := Module[
    {lastBlock, res},

    lastBlock = Lookup[BlockchainCloudRequest["getblock", {"block" -> -1}], {"BlockNumber", "BlockHash"}];
    res = BlockchainCloudRequest["getinfo", {}];
    Append[res,<|"Blocks" -> First[lastBlock], "LatestHash" -> Last[lastBlock]|>]
]

PackageScope[BlockchainBlockDataWL]
BlockchainBlockDataWL[block:First | Last | _Integer | _String] := Module[
    {params},
    Which[ 
        ToString[block] === "First", params = {"block" -> 0},
        ToString[block] === "Last", params = {"block" -> -1},
        True, params = {"block" -> block}
    ];

    BlockchainCloudRequest["getblock", params]
]

PackageScope[BlockchainTransactionDataWL]
BlockchainTransactionDataWL[txid_String] := Module[
    {},

    BlockchainCloudRequest["gettransaction", {"txid" -> txid}]
]

PackageScope[BlockchainGetWL]
BlockchainGetWL[txid:(_String | {___String})] := Block[
    {response, tmp},

    response = BlockchainCloudRequest["getmetadatafromtransaction", {"tx" -> txid}];
    response = Quiet[Check[Uncompress[#, Hold], #, {Uncompress::corrupt,Uncompress::string}]] & /@ response;
    If[MatchQ[#, HoldPattern[Hold[_ByteArray]]],
        BinaryDeserialize[ReleaseHold[#]],
        #
    ] & /@ response
]

PackageScope[BlockchainPutWL]
BlockchainPutWL[data_] := Block[
    {response, tmp},

    If[TrueQ[StringLength[data] > 8000000(* $byteSizeLimit *)],
        BlockchainMessage[BlockchainPut, "sizlim"];
        Throw[$Failed]
    ];

    BlockchainCloudRequest["putdata", {"data" -> data}]
]