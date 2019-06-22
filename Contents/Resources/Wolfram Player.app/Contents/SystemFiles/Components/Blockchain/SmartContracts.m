Package["Blockchain`"]

(*PackageImport["GeneralUtilities`"]*)

$ethPath = "/eth";

$ETHGas = "0xf1004";

$SolidityTemplateDirectory = PacletManager`PacletResource["Blockchain", "Solidity_Templates"];
$SolidityTemplateV1Filename = "Template1v1.sol"; (*Template for ExternalStorageObject*)
$SolidityTemplateV2Filename = "Template2v1.sol"; (*Template for WLS*)

$WServerToken = Compress[<|"Application" -> "Wolfram BlockchainContract", "Version" -> "1.0"|>];
(*
PackageScope[BlockchainContractObject]
PackageExport[BlockchainContractObject]
BlockchainContractObject[a_Association][c_] := a[c]
BoxForm`MakeConditionalTextFormattingRule[BlockchainContractObject];

BlockchainContractObject /: MakeBoxes[f:BlockchainContractObject[a_Association], form: StandardForm | TraditionalForm] := Module[
    {icon, alwaysGrid, sometimesGrid, dd},
    icon = makeIcon[f["IconData"], 10];

    alwaysGrid = {
    BoxForm`SummaryItem[{"blockchain: ",f["Blockchain"]}],
    BoxForm`SummaryItem[{"network: ",f["Network"]}]
    };

    dd = Association @@ DeleteCases[Normal[a], ("Blockchain" -> _) | ("Network" -> _) | ("IconData" -> _)];
    dd["ApplicationBinaryInterface"] = Developer`WriteRawJSONString[dd["ApplicationBinaryInterface"], "Compact" -> True];
    sometimesGrid = (toextrafield /* BoxForm`SummaryItem) @@@ Normal[dd];

    BoxForm`ArrangeSummaryBox[BlockchainContractObject, f, icon, alwaysGrid, sometimesGrid, form, "Interpretable" -> True]
]*)


PackageScope["BlockchainContractETH"]

BlockchainContractETH[input_ExternalStorage`IPFSObject, network_String] := Module[
	{blockchainPath = $ethPath, requestType = "compile", response, filepath, params, abiraw, positions, abidata, externalStorageData, constructorData},
    filepath = FileNameJoin[{$SolidityTemplateDirectory, $SolidityTemplateV1Filename}];

    params = <|"file" -> File@filepath|>;
    response = ExplorerRequest[blockchainPath, requestType, params];
    (* ABI processing *)
    abiraw = Developer`ReadRawJSONString[response["abi"]];
    positions = Flatten[Position[abiraw[[All, "type"]], "constructor" | "event"]];
    abidata = Complement[abiraw, Part[abiraw, positions]]; (* removing: constructor and event from abi data*)

    (* Bytecode processing *)
    externalStorageData = Association @@ input;
    If[!KeyExistsQ[externalStorageData, "FileHash"], AppendTo[externalStorageData, "FileHash" -> ""]];

    constructorData = EthereumEncode[{
                    $WServerToken,
                    "IPFS",
                    externalStorageData["Address"],
                    externalStorageData["FileName"],
                    externalStorageData["FileHash"]}, 
                    Type -> {"String", "String", "String", "String", "String"}];

    (* Object *)
    
    <|"Blockchain" -> "Ethereum", 
            "Network" -> Capitalize@network, 
            "ApplicationBinaryInterface" -> abidata, 
            "ByteCode" -> response["bin"] <> constructorData,
            "IconData" -> externalStorageData["Address"] (* for ESO it will use Address to create the object's icon*)
            |>

]

BlockchainContractETH[input_String, network_String] := Module[
    {blockchainPath = $ethPath, requestType = "compile", response, filepath, filename = "", params, abiraw, positions, abidata, WLContent, constructorData},
    (* Validate Input file *)
    filename = FileBaseName[input]<>"."<>FileExtension[input];
    (* Compile *)
    filepath = FileNameJoin[{$SolidityTemplateDirectory, $SolidityTemplateV2Filename}];

    params = <|"file" -> File@filepath|>;
    response = ExplorerRequest[blockchainPath, requestType, params];
    (* ABI processing *)
    abiraw = Developer`ReadRawJSONString[response["abi"]];
    positions = Flatten[Position[abiraw[[All, "type"]], "constructor" | "event"]];
    abidata = Complement[abiraw, Part[abiraw, positions]]; (* removing: constructor and event from abi data*)

    (* Bytecode processing *)
    WLContent = Compress[Import[input, "Text"]];

    constructorData = EthereumEncode[{
                    $WServerToken,
                    filename,
                    WLContent}, 
                    Type -> {"String", "String", "String"}];

    (* Object *)
    
    <|"Blockchain" -> "Ethereum", 
            "Network" -> Capitalize@network, 
            "ApplicationBinaryInterface" -> abidata, 
            "ByteCode" -> response["bin"] <> constructorData,
            "IconData" -> filename (* for Filepath as input it will use the finename to create the object's icon*)
            |>

]

PackageScope["BlockchainContractValueETH"]

BlockchainContractValueETH[contractAddress_, functionCall_, network_] := Module[
    {blockchainPath = $ethPath, func, inp, sender, requestType = "call", func0, id, paramsABI, dataBin, inType, outType, response, params, result, version, to = StringDelete[contractAddress, "0x"]},

    If[!validateEthereumAddress[contractAddress], BlockchainMessage[BlockchainContractValue, "ethadd", contractAddress]; Throw[$Failed]];

    If[AssociationQ[functionCall] || Head[functionCall] === Typed,
        Switch[Head[functionCall],
            Association,
                If[!MemberQ[{"Function", "Inputs", "Sender"}, #],
                    (BlockchainMessage[BlockchainContractValue, "invelem", #, "input"]; Throw[$Failed])
                ] &/@ Keys[functionCall];
                func = functionCall["Function"];
                inp = functionCall["Inputs"];
                sender = functionCall["Sender"];
            ,Typed,
                func = functionCall;
                inp = Missing["NotAvailable"];
                sender = Missing["NotAvailable"];
        ];
        If[MissingQ[func], 
            BlockchainMessage[BlockchainContractValue, "misselem", "Function", "input"]; Throw[$Failed],
            If[Head[func] =!= Typed, BlockchainMessage[BlockchainContractValue, "invval", func, "Function", "A Typed expression"]; Throw[$Failed],
                If[!MatchQ[func, Typed[_String, (_String | {_String...} | Rule[(_String | {_String...}), (_String | {_String...})])]],
                    (BlockchainMessage[BlockchainContractValue, "invtyp", func, "Function"]; Throw[$Failed])
                ]
            ]
        ];
        inType = With[{tmp = Last[func]}, If[Head[tmp] === Rule, With[{tmp2 = Keys[tmp]}, If[ListQ[tmp2], tmp2, {tmp2}]], {}]];
        outType = With[{tmp = Last[func]}, If[Head[tmp] === Rule, Values[tmp], tmp]];
        If[!MissingQ[sender], If[!StringQ[sender], BlockchainMessage[BlockchainContractValue, "invval", sender, "Sender", "A string expression"]; Throw[$Failed], sender = StringDelete[sender, "0x"]]];
        If[!MissingQ[inp], 
            If[If[ListQ[inp], Length[inp], 1] =!= Length[inType], 
                (BlockchainMessage[BlockchainContractValue, "abilen"]; Throw[$Failed])
            ];
            inp = If[validateEthereumAddress[#], StringDelete[#, "0x"], #] &/@ If[ListQ[inp], inp, {inp}];
            ,
            If[0 =!= Length[inType], (BlockchainMessage[BlockchainContractValue, "abiinp", "input"]; Throw[$Failed])]
        ];  
        If[!validateSolidityType[#], (BlockchainMessage[BlockchainContractValue, "abityp", #]; Throw[$Failed])] &/@ Flatten[{inType, outType}];
        If[!MissingQ[inp], If[!AllTrue[Transpose[{inp, inType}], crossValidateSolidityInput[#[[1]], #[[2]]]&], 
            (BlockchainMessage[BlockchainContractValue, "abienc", inp, inType]; Throw[$Failed])
        ]];    
        func0 = First[func] <> "(" <> StringJoin[If[Length[inType] > 1, Riffle[inType, ","], inType]] <> ")";
        id = StringTake[Hash[func0, "Keccak256", "HexString"], 8];
        paramsABI = If[!MissingQ[inp], Quiet[Check[encodeParameters[inp, inType], $Failed]], ""];
        If[FailureQ[paramsABI], BlockchainMessage[BlockchainContractValue, "abienc", inp, inType]; Throw[$Failed]];
        dataBin = id <> paramsABI;
        params = {"network" -> network, "address" -> "0x" <> ToLowerCase[to], "data" -> "0x" <> id <> paramsABI, 
                "sender" -> With[{tmp = sender}, If[MissingQ[tmp], "0x", "0x" <> ToLowerCase[tmp]]], "block" -> "latest"
            };
        response = ExplorerRequest[blockchainPath, requestType, params];
        result = If[response === "0x", Missing["NoResult"], Quiet[Check[decodeParameters[StringDrop[response, 2], outType], $Failed]]];
        If[FailureQ[result], BlockchainMessage[BlockchainContractValue, "abidec", outType]];
        ,
        version = BlockchainContractValueETH[to, Typed["TemplateVersion", "string"], network];
        Switch[version,
            "v1",
                Switch[functionCall,
                    "Result", result = With[{add = BlockchainContractValueETH[to, <|"Function" -> Typed["ResultAddress", "string"]|>, network]}, 
                        If[add =!= "", ExternalStorage`IPFSDownload[ExternalStorage`IPFSObject[<|"Address" -> add|>], "ImportContent" -> True], add]];
                    ,"Evaluated", result = BlockchainContractValueETH[to, <|"Function" -> Typed["Evaluated", "bool"]|>, network];
                    ,All, result = <|
                            "Result" -> With[{add = BlockchainContractValueETH[to, <|"Function" -> Typed["ResultAddress", "string"]|>, network]}, 
                                If[add =!= "", ExternalStorage`IPFSDownload[ExternalStorage`IPFSObject[<|"Address" -> add|>], "ImportContent" -> True], add]], 
                            "Evaluated" -> BlockchainContractValueETH[to, <|"Function" -> Typed["Evaluated", "bool"]|>, network]
                        |>;
                    ,_, Throw[Missing["NoResult"]]
                ]
            ,"v2",
                Switch[functionCall,
                    "Result", result = With[{tmp = BlockchainContractValueETH[to, <|"Function" -> Typed["Result", "string"]|>, network]}, If[tmp =!= "", Uncompress[tmp], tmp]];
                    ,"Evaluated", result = BlockchainContractValueETH[to, <|"Function" -> Typed["Evaluated", "bool"]|>, network];
                    ,All, result = <|
                            "Result" -> With[{tmp = BlockchainContractValueETH[to, <|"Function" -> Typed["Result", "string"]|>, network]}, If[tmp =!= "", Uncompress[tmp], tmp]], 
                            "Evaluated" -> BlockchainContractValueETH[to, <|"Function" -> Typed["Evaluated", "bool"]|>, network]
                        |>;
                    ,_, Throw[Missing["NoResult"]]
                ]
            ,_, Throw[Missing["NoResult"]]
        ]
    ];

    result

]

PackageScope["ContractEncodeETH"]

ContractEncodeETH[x_, type_] := With[{x0 = If[ListQ[x], x, {x}], type0 = If[ListQ[type], type, {type}]}, encodeParameters[x0, type0]]

PackageScope["ContractDecodeETH"]

ContractDecodeETH[x_, type_] := decodeParameters[x, type]
