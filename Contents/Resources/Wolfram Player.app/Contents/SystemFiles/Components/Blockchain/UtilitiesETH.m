Package["Blockchain`"]

(**************    ETHEREUM ABI ENCODING    **************)

EthereumABI::len = "Length of data must be the same as type vector.";
EthereumABI::argx = "Called with `1` argument`2`; `3` expected.";
EthereumABI::type1 = "`1` type of input not allowed.";
EthereumABI::type2 = "`1` option value not allowed.";
EthereumABI::inv1 = "`1` data dimension does not match `2`.";
EthereumABI::inv2 = "`1` data type does not match `2`.";
EthereumABI::inv3 = "`1` is not a valid hex string.";
EthereumABI::inv4 = "Input is not a valid hex string. Try removing '0x'.";
EthereumABI::invfunc = "Function type data must contain 24 bytes.";
EthereumABI::invaddr = "Address type data must contain 20 bytes.";
EthereumABI::invbyte = "`1` type data must contain `2` bytes.";

EncodeMessageTag = ContractEncode;
DecodeMessageTag = ContractDecode;

EthereumABIMessage[tag_,errorcode_,params___]:=With[
    {msg = MessageName[EthereumABI, errorcode]},
    MessageName[tag,errorcode] = msg;
    Message[MessageName[tag, errorcode], params];
]

ValidateEncoderSingleInput[x_] := With[
    {typeList={String,Integer,Real},type=Head[x]},
    If[type===Symbol,
        If[!MemberQ[{True,False},x],
            EthereumABIMessage[EncodeMessageTag, "type1", type];
            Throw[$Failed];
        ],
        If[!MemberQ[typeList,type],
            EthereumABIMessage[EncodeMessageTag, "type1", type];
            Throw[$Failed];
        ]
    ]
]

ValidateSingleOptionValue[x_] := If[
    !(StringMatchQ[x,RegularExpression["^(?i)(integer|fixed|address|function|bool)(\\[\\d+\\])*(\\[\\])?$"]]
    ||StringMatchQ[x,RegularExpression["^(?i)(string|bytes)(\\[\\d+\\])*$"]]
    ||StringMatchQ[x, RegularExpression["^(?i)(bytes)([1-9]|[1-2][0-9]|3[0-2])(\\[\\d+\\])*(\\[\\])?$"]]),
    EthereumABIMessage[EncodeMessageTag, "type2", x];
    Throw[$Failed];
]

ValidateSingleTypeValue[x_] := If[
    !(StringMatchQ[x,RegularExpression["^(?i)(integer|fixed|address|function|bool)(\\[\\d+\\])*(\\[\\])?$"]]
    ||StringMatchQ[x,RegularExpression["^(?i)(string|bytes)(\\[\\d+\\])*$"]]
    ||StringMatchQ[x, RegularExpression["^(?i)(bytes)([1-9]|[1-2][0-9]|3[0-2])(\\[\\d+\\])*(\\[\\])?$"]]),
    EthereumABIMessage[DecodeMessageTag, "type2", x];
    Throw[$Failed];
]

ValidateDecoderData[x_] := If[StringTake[x,2] === "0x", EthereumABIMessage[DecodeMessageTag, "inv4"]; Throw[$Failed];]

CrossValidateSingleInput0[x_,type_] := Module[
    {type1, temp},
    type1 = StringJoin[StringCases[type, RegularExpression["[[:alpha:]]"]]];
    If[StringMatchQ[type1,RegularExpression["(?i)(bytes|function|address)"]],
        If[!StringQ[x],
            EthereumABIMessage[EncodeMessageTag, "inv2", x, type];
            Throw[$Failed],
            If[!StringMatchQ[x,RegularExpression["(?i)(\\d|[a-f])*"]],
                EthereumABIMessage[EncodeMessageTag, "inv3", x];
                Throw[$Failed],
                If[StringMatchQ[type,RegularExpression["(?i)(bytes([1-9]|([1-2][0-9])|(3[0-2])))"]],
                    temp = ToExpression @@ StringCases[type, NumberString];
                    If[StringLength[x] != 2*temp,
                        EthereumABIMessage[EncodeMessageTag, "invbyte", type, temp];
                        Throw[$Failed]
                    ];
                ];
                If[StringMatchQ[type1,RegularExpression["(?i)(address)"]] && (StringLength[x] != 40),
                    EthereumABIMessage[EncodeMessageTag, "invaddr"];
                    Throw[$Failed]
                ];
                If[StringMatchQ[type1,RegularExpression["(?i)(function)"]] && (StringLength[x] != 48),
                    EthereumABIMessage[EncodeMessageTag, "invfunc"];
                    Throw[$Failed]
                ];
            ]
        ]
    ];
    If[StringMatchQ[type1,RegularExpression["(?i)(string)"]],
        If[!StringQ[x],
            EthereumABIMessage[EncodeMessageTag, "inv2", x, type];
            Throw[$Failed];
        ]
    ];
    If[StringMatchQ[type1,RegularExpression["(?i)(integer)"]],
        If[!IntegerQ[x],
            EthereumABIMessage[EncodeMessageTag, "inv2", x, type];
            Throw[$Failed];
        ]
    ];
    If[StringMatchQ[type1,RegularExpression["(?i)(fixed)"]],
        If[!NumericQ[x],
            EthereumABIMessage[EncodeMessageTag, "inv2", x, type];
            Throw[$Failed];
        ]
    ];
    If[StringMatchQ[type1,RegularExpression["(?i)(bool)"]],
        If[!BooleanQ[x],
            EthereumABIMessage[EncodeMessageTag, "inv2", x, type];
            Throw[$Failed];
        ]
    ];
]

CrossValidateSingleInput[x_,type_] := Module[
    {dimen1,dimen2,type1,type2},
    If[ListQ[x],
        dimen1 = StringCases[type, RegularExpression["(\\[\\d*\\])"]];
        dimen2 = Dimensions[x];
        If[StringMatchQ[Last[dimen1], "[]"],
            If[!SameQ[FromDigits[StringTake[#, {2, -2}]] & /@ Most[dimen1],Most[dimen2]],
                EthereumABIMessage[EncodeMessageTag, "inv1", x, type];
                Throw[$Failed],
                CrossValidateSingleInput0[x,type];
            ]
            If[!SameQ[FromDigits[StringTake[#, {2, -2}]] & /@ dimen1,dimen2],
                EthereumABIMessage[EncodeMessageTag, "inv1", x, type];
                Throw[$Failed],
                CrossValidateSingleInput0[x,type];
            ]
        ],
        CrossValidateSingleInput0[x,type];
    ]
]

ValidateEncoderInputs[x___,opt: OptionsPattern[EthereumEncode]] := Module[
    {},
    If[!MatchQ[Length[{x}], 1],
        EthereumABIMessage[EncodeMessageTag, "argx", Length[{x}], "s", "1"];
        Throw[$Failed];
    ];
    If[Head[x]===List,
        ValidateEncoderSingleInput[#]&/@Flatten[x],
        ValidateEncoderSingleInput[x]
    ];
    If[OptionValue[Type]=!=Automatic,
        If[Head[x]===List,
            If[!SameQ[Length[x],Length[Flatten[{OptionValue[Type]}]]],
                EthereumABIMessage[EncodeMessageTag, "len"];
                Throw[$Failed];
            ],
            If[!SameQ[Length[OptionValue[Type]],0],
                EthereumABIMessage[EncodeMessageTag, "len"];
                Throw[$Failed];
            ]   
        ];
        If[Head[OptionValue[Type]]===List,
            ValidateSingleOptionValue[#]&/@OptionValue[Type];
            CrossValidateSingleInput[#[[1]],#[[2]]]&/@Thread[{x,OptionValue[Type]}],
            ValidateSingleOptionValue[OptionValue[Type]];
            CrossValidateSingleInput[x,OptionValue[Type]];
        ];
    ];
]

PackageScope[EthereumEncode]

SetUsage[EthereumEncode, "
EthereumEncode[input] returns a value encoded according to Ethereum's Specification.
The input can be an String, Integer, Integer List, List of List 
"
]

encodeSingleParameter[{data_?Internal`NonNegativeIntegerQ, x_?(StringMatchQ[#,___~~"int"~~___]&)}] := With[{temp = IntegerString[data, 16]}, StringPadLeft[temp, Ceiling[StringLength[temp], 64], "0"]]
encodeSingleParameter[{data_Integer?Negative, x_?(StringMatchQ[#,___~~"int"~~___]&)}] :=  With[{temp = IntegerString[16^64 + data, 16]}, StringPadLeft[temp, Ceiling[StringLength[temp], 64], "0"]]
encodeSingleParameter[{data_Integer, x_?(StringMatchQ[#,___~~"int"~~___]&)}] := With[{temp = IntegerString[data, 16]},StringPadLeft[temp, Ceiling[StringLength[temp], 64], "0"]]
encodeSingleParameter[{data_?NumericQ, x_?(StringMatchQ[#,___~~"fixed"~~__~~"x"~~__]&)}] := With[{n = FromDigits[#, 10]&@@StringCases[x, ___~~"fixed"~~__~~"x"~~n1__ -> n1]}, encodeSingleParameter[{Round[data * 10.^n], "int"}]]
encodeSingleParameter[{data_?NumericQ, x_?(StringMatchQ[#,___~~"fixed"~~___]&)}] := encodeSingleParameter[{data, "fixed192x19"}]
encodeSingleParameter[{data_?BooleanQ, _}] := If[data, StringJoin[Table["0", 63]] <> "1", StringJoin[Table["0", 64]]]
encodeSingleParameter[{data_?StringQ, "string"}] := With[{temp = StringJoin[IntegerString[ToCharacterCode[data, "ASCII"], 16]], len = StringLength[data]}, {encodeSingleParameter[{len, "uint256"}], StringPadRight[temp, Ceiling[StringLength[temp], 64], "0"]}]
encodeSingleParameter[{data_?StringQ, "bytes"}] := With[{len = StringLength[data]}, {encodeSingleParameter[{Round[len/2],"uint256"}], StringPadRight[data, Ceiling[StringLength[data], 64], "0"]}]
encodeSingleParameter[{data_?StringQ, x_?(StringMatchQ[#, "bytes"~~DigitCharacter ..]&)}] := StringPadRight[data,Ceiling[StringLength[data],64],"0"]
encodeSingleParameter[{data_?StringQ, "address"}] := With[{temp = FromDigits[data, 16]}, encodeSingleParameter[{temp, "uint160"}]]
encodeSingleParameter[{data_?StringQ, "function"}] := encodeSingleParameter[{data, "bytes24"}]
encodeSingleParameter[{data_?StringQ, _}] := With[{temp = StringJoin[IntegerString[ToCharacterCode[data, "ASCII"], 16]]}, StringPadRight[temp, Ceiling[StringLength[temp], 64], "0"]]
encodeSingleParameter[{data_?ListQ, x_?(StringMatchQ[#, __~~"["~~DigitCharacter ..~~"]"~~EndOfString]&)}] := Module[{y}, y = ToString@@StringCases[x, y1__~~"[" -> y1]; Map[encodeSingleParameter, Thread[{data, y}]]]
encodeSingleParameter[{data_?ListQ, x_?(StringMatchQ[#, __~~"[]"~~EndOfString]&)}] := Module[{y}, y = ToString@@StringCases[x, y1__~~"["->y1]; {encodeSingleParameter[{Length[data], "uint256"}], Map[encodeSingleParameter, Thread[{data, y}]]}]


arrangeDynamicParameters0[rawData_] := Module[{len,pos,temp,resul,resul2,n,temp0},
    n = Length[Cases[rawData, {_,"Dynamic"}]];
    temp0 = DeleteCases[rawData, {_,"Dynamic"}];
    len = StringLength[StringJoin[Flatten[Take[#,1]&/@temp0]]];
    len = (len+n*64)/2;
    temp = First[Cases[rawData,{_,"Dynamic"}]];
    pos = Position[rawData,temp,-1,1];
    resul = ReplacePart[rawData,pos->{encodeSingleParameter[{len,"uint256"}],""}];
    resul2 = Append[resul,Replace[temp,"Dynamic"->"NotDynamic",2]];
    resul2
]

arrangeDynamicParameters[rawData_] := With[{n = Length[Cases[rawData, {_, "Dynamic"}]]}, Nest[arrangeDynamicParameters0, rawData, n]]

PackageScope[encodeParameters]
encodeParameters[data_, types_] := Module[{types2, data2, rawDataEncoded, rawDataEncoded2, rawDataSorted, dataEncoded},
    types2 = StringMatchQ[types, (__~~"[]") | "bytes" | ("bytes["~~__) | ("string"~~___)] /. {False -> "Static", True -> "Dynamic"};
    data2 = Thread[{data, types}];
    rawDataEncoded = encodeSingleParameter[#]&/@data2;
    rawDataEncoded2 = Thread[{rawDataEncoded, types2}];
    rawDataSorted = arrangeDynamicParameters[rawDataEncoded2];
    dataEncoded = StringJoin[Flatten[Take[#, 1]&/@rawDataSorted]];
    dataEncoded
]

EthereumEncodeAux[x_List] := Module[{suffix, dimens, type, func},
  dimens = Dimensions[x];
  type = Head[Flatten[x][[1]]] /. {String -> "string", Integer -> "int", Real -> "fixed", Symbol -> "bool"};
  suffix = StringJoin[Map[StringJoin["[", ToString[#], "]"] &, dimens]];
  func = StringJoin[type, suffix]
]
EthereumEncodeAux[x_] := Head[x] /. {String -> "string", Integer -> "int", Real -> "fixed", Symbol -> "bool"}

Options[EthereumEncode] = {Type -> Automatic, EchoQ -> False}

EthereumEncode[x___, opt: OptionsPattern[EthereumEncode]] := Module[{type,echo},
    ValidateEncoderInputs[x,opt];
    type = If[OptionValue[Type] === Automatic, 
        If[MatchQ[#, _List], EthereumEncodeAux /@ #, EthereumEncodeAux[#]] &@x, 
        ToLowerCase@OptionValue[Type]
    ];
    echo = If[OptionValue[Type] === Automatic,
        echo = StringReplace[type, {"string" -> "String", "int" -> "Integer", "fixed" -> "Fixed", "bool" -> "Boolean"}],
        echo = OptionValue[Type]
    ];
    If[TrueQ@OptionValue[EchoQ], Echo[echo]];
    If[Head[type] === List, encodeParameters[x, StringJoin["f(", Riffle[type, ","], ")"]], encodeParameters[x, StringJoin["f(", type, ")"]]]
]

PackageScope[EthereumDecode]

SetUsage[EthereumDecode, "
EthereumDecode[data, type] returns a value decoded from Ethereum's specification it depends of the data type
"
]

decodeSingleParameter[{data_,x_?(StringMatchQ[#,__~~"["~~DigitCharacter ..~~"]"~~EndOfString]&)}]:=Module[{temp,n,y},
    {n,y}=Flatten[StringCases[x,y1__~~"["~~n1__~~"]"->{n1,y1}]];If[FromDigits[n]!=0,temp=Partition[Characters[data],StringLength[data]/FromDigits[n]];
    Map[decodeSingleParameter,Thread[{StringJoin[#]&/@temp,y}]],{}]
]
decodeSingleParameter[{data_,x_?(StringMatchQ[#,__~~"[]"~~EndOfString]&)}]:=Module[{temp1,temp2,y,n},
    y=ToString@@StringCases[x,y1__~~"[]"->y1];{temp1,temp2}=TakeDrop[Characters[data],64];
    n=ToString[FromDigits[StringJoin[temp1],16]];
    decodeSingleParameter[{StringJoin[temp2],y<>"["<>n<>"]"}]
]
decodeSingleParameter[{data_,x_?(StringMatchQ[#,"uint"~~___]&)}]:=FromDigits[data,16]
decodeSingleParameter[{data_,x_?(StringMatchQ[#,"int"~~___]&)}]:=Switch[First[Characters[data]],"0",FromDigits[data,16],"f",FromDigits[data,16]-16^64]
decodeSingleParameter[{data_,x_?(StringMatchQ[#,"ufixed"~~__~~"x"~~__]&)}]:=With[{n=FromDigits[#,10]&@@StringCases[x,___~~"fixed"~~__~~"x"~~n1__->n1]},decodeSingleParameter[{data,"uint"}]/10.^n]
decodeSingleParameter[{data_,x_?(StringMatchQ[#,"fixed"~~__~~"x"~~__]&)}]:=With[{n=FromDigits[#,10]&@@StringCases[x,___~~"fixed"~~__~~"x"~~n1__->n1]},decodeSingleParameter[{data,"int"}]/10.^n]
decodeSingleParameter[{data_,x_?(StringMatchQ[#,"ufixed"]&)}]:=decodeSingleParameter[{data,"ufixed192x19"}]
decodeSingleParameter[{data_,x_?(StringMatchQ[#,"fixed"]&)}]:=decodeSingleParameter[{data,"fixed192x19"}]
decodeSingleParameter[{data_,x_?(StringMatchQ[#,"bool"]&)}]:=Switch[data,"0000000000000000000000000000000000000000000000000000000000000001",True,"0000000000000000000000000000000000000000000000000000000000000000",False]
decodeSingleParameter[{data_,x_?(StringMatchQ[#,"bytes"~~__]&)}]:=Module[{n},
    n=StringCases[x,"bytes"~~n1__->n1]; 
    StringTake[data,2*FromDigits@@n]
]
decodeSingleParameter[{data_,x_?(StringMatchQ[#,"address"]&)}]:= checksum[ToLowerCase[StringPadLeft[IntegerString[decodeSingleParameter[{data,"uint160"}], 16], 40, "0"]]]
decodeSingleParameter[{data_,x_?(StringMatchQ[#,"function"]&)}]:= decodeSingleParameter[{data,"bytes24"}]
decodeSingleParameter[{data_,x_?(StringMatchQ[#,"bytes"]&)}]:=Module[{temp1,temp2,n},
    {temp1,temp2}=TakeDrop[Characters[data],64];
    n=ToString[FromDigits[StringJoin[temp1],16]];
    decodeSingleParameter[{StringJoin[temp2],"bytes"<>n}]
]
decodeSingleParameter[{data_,x_?(StringMatchQ[#,"string"]&)}]:=Module[{temp1,temp2,temp3,temp4,n},
    {temp1,temp2}=TakeDrop[Characters[data],64];
    n=FromDigits[StringJoin[temp1],16];
    temp3=StringTake[StringJoin[temp2],2*n];
    temp4=StringJoin[#]&/@Partition[Take[Characters[temp3],2*n],2];
    FromCharacterCode[FromDigits[#,16]&/@temp4 ,"ASCII"]
]

countBytes[x_?(StringMatchQ[#, (__~~"[]") | "bytes" | ("bytes["~~__) | ("string"~~___)]&) ] := {64,"Dynamic"}
countBytes[x_?(StringMatchQ[#, __~~"["~~__~~"]"]&)] := With[{n=FromDigits[#,10]&/@StringCases[x,"["~~n1__?DigitQ~~"]"->n1]},{64*(Times@@n),"Static"}]
countBytes[x_]:={64,"Static"}

PackageScope[decodeParameters]
decodeParameters[data_,types0_]:=Module[{types,temp1,temp2,temp3,temp4,temp5,temp6,tempEst,tempDyn,result0},
    (*types=StringSplit[#,","]&@@StringCases[func,"("~~x__~~")"->x];*)
    If[Head[types0]===List, types = types0, types = {types0}];
    temp1={Accumulate[#1],#2}&@(Sequence@@Transpose[countBytes[#]&/@types]);
    temp2=Thread[{{#[[1]]+1,#[[2]]}&/@Partition[Prepend[temp1[[1]],0],2,1],temp1[[2]]}];
    temp3=StringJoin[#]&/@(Take[Characters[data],#]&/@(Flatten[Take[#,1]&/@Cases[temp2,{_,"Dynamic"}],1]));
    temp4={#[[1]]+1,#[[2]]}&/@(Partition[Append[FromDigits[#,16]*2&/@temp3,Length[Characters[data]]],2,1]);
    tempEst=Thread[Cases[temp2,{x_,"Static"}]->(Flatten[Take[#,1]&/@Cases[temp2,{x_,"Static"}],1])];
    tempDyn=Thread[Cases[temp2,{_,"Dynamic"}]->temp4];
    temp5=temp2/.Join[tempEst,tempDyn];
    temp6=StringJoin[#]&/@(Take[Characters[data],#]&/@temp5);
    result0=decodeSingleParameter[#]&/@(Thread[{temp6,types}]);
    If[Length[result0]===1,First@result0,result0]
]

EthereumDecode[data_,type_]:=decodeParameters[data,type]

(**************    ETHEREUM RLP ENCODING    **************)

toBinary[x_Integer] := Which[x === 0, "",True, With[{temp = IntegerString[x, 16]}, If[OddQ[StringLength[temp]], "0" <> temp, temp]]]

encodeLength[len_Integer, offset_Integer] := Which[
    len < 56, With[{temp = IntegerString[len + offset, 16]}, If[OddQ[StringLength[temp]], "0" <> temp, temp]],
    len < 256^8, With[{BL = toBinary[len]}, With[{temp = IntegerString[StringLength[BL]/2 + offset + 55, 16]}, temp <> BL]],
    True, Echo["Input too long"]; $Failed
]

rlpEncode[x_] := Which[
    MatchQ[x, _String?(EvenQ[StringLength[#]] &)], With[{len = StringLength[x]/2}, If[len === 1 && FromDigits[x, 16] < FromDigits["80", 16], x, encodeLength[len, FromDigits["80", 16]] <> x]],
    ListQ[x], With[{temp = rlpEncode[#] & /@ x}, encodeLength[Total[StringLength[temp]]/2, FromDigits["c0", 16]] <> StringJoin[temp]]
]

PackageScope[ethRlpEncode]

ethRlpEncode[<|"TransactionCount" -> nonce_, "GasPrice" -> gasPrice_, "GasLimit" -> gasLimit_, "DestinationAddress" -> to_, "Amount" -> value_, "Data" -> data_, "ChainID" -> chainId_|>] := With[
    {aux = With[{temp = IntegerString[#, 16]}, If[temp === "0", "", If[OddQ[StringLength[temp]], "0" <> temp, temp]]] &}, 
    rlpEncode[{aux[nonce], aux[gasPrice], aux[gasLimit], If[to =!= "", StringDelete[ToLowerCase[to], "0x"], to], If[value =!= "", aux[value], value], data, aux[chainId], "", ""}]
]

ethRlpEncode[<|"TransactionCount" -> nonce_, "GasPrice" -> gasPrice_, "GasLimit" -> gasLimit_, "DestinationAddress" -> to_, "Amount" -> value_, "Data" -> data_, "v" -> v_, "r" -> r_, "s" -> s_|>] := With[
    {aux = With[{temp = IntegerString[#, 16]}, If[temp === "0", "", If[OddQ[StringLength[temp]], "0" <> temp, temp]]] &}, 
    rlpEncode[{aux[nonce], aux[gasPrice], aux[gasLimit], If[to =!= "", StringDelete[ToLowerCase[to], "0x"], to], If[value =!= "", aux[value], value], data, aux[v], r, s}]
]

decodeLength[x_String?(EvenQ[StringLength[#]] &)] := With[{len = StringLength[x]/2},
    If[len === 0, Echo["Input is null"]];
    With[{prefix = FromDigits[First[StringPartition[x, 2]], 16]},
        Which[
            prefix <= FromDigits["7f", 16], {0, 1, String},
            prefix <= FromDigits["b7", 16] && len > prefix - FromDigits["80", 16], {1, prefix - FromDigits["80", 16], String},
            prefix <= FromDigits["b7", 16] && len > prefix - FromDigits["b7", 16] && len > prefix - FromDigits["b7", 16] + FromDigits[StringTake[x, {3, 2 + 2*(prefix - FromDigits["b7", 16])}], 16], 
                {1 + prefix - FromDigits["b7", 16], FromDigits[StringTake[x, {3, 2 + 2*(prefix - FromDigits["b7", 16])}], 16], String},
            prefix <= FromDigits["f7", 16] && len > prefix - FromDigits["c0", 16], {1, prefix - FromDigits["c0", 16], List},
            prefix <= FromDigits["ff", 16] && len > prefix - FromDigits["f7", 16] && len > prefix - FromDigits["f7", 16] + FromDigits[StringTake[x, {3, 2 + 2*(prefix - FromDigits["f7", 16])}], 16], 
                {1 + prefix - FromDigits["f7", 16], FromDigits[StringTake[x, {3, 2 + 2*(prefix - FromDigits["f7", 16])}], 16], List},
            True, Echo["Input don't conform RLP encoding form"]; $Failed
        ]
    ]
]

decodeList[x_String?(EvenQ[StringLength[#]] &)] := Module[{output = {}, temp = x, temp0, from, to}, 
    While[temp =!= "",
        temp0 = decodeLength[temp];
        from = If[temp0[[1]] =!= 0, 2*temp0[[1]] - 1, 1];
        to = If[temp0[[1]] =!= 0, from + 2*temp0[[2]] + 1, 2*temp0[[2]]];
        output = Join[output, {StringTake[temp, {from, to}]}];
        temp = StringDrop[temp, to];
    ];
    output
]

rlpDecode[x_String?(EvenQ[StringLength[#]] &)] := If[x === "", Null, Module[{temp, from, to},
    temp = decodeLength[x];
    from = 1 + 2*temp[[1]];
    to = from + 2*temp[[2]] - 1;
    Switch[temp[[3]],
        String, StringTake[x, {from, to}],
        List, rlpDecode[decodeList[StringTake[x, {from, to}]]]
    ]
]]

SetAttributes[rlpDecode, Listable]

PackageScope[ethRlpEncode]
ethRlpDecode[x_String?(EvenQ[StringLength[#]] &)] := With[{dec = rlpDecode[x]}, 
    If[Length[dec] =!= 9, Echo["Invalid ETH Transaction"]; $Failed, 
    Association[
        "TransactionCount" -> FromDigits[dec[[1]], 16], 
        "GasPrice" -> FromDigits[dec[[2]], 16], 
        "GasLimit" -> FromDigits[dec[[3]], 16], 
        "DestinationAddress" -> dec[[4]], 
        "Value" -> FromDigits[dec[[5]], 16], 
        "Data" -> dec[[6]], 
        "v" -> dec[[7]], 
        "r" -> dec[[8]], 
        "s" -> dec[[9]]
    ]
]]

PackageScope[checksum]
checksum[addr_] := StringJoin[(If[FromDigits[#[[1]], 16]>7, ToUpperCase[#[[2]]], #[[2]]] & /@ Transpose[
    {StringPartition[StringTake[Hash[addr, "Keccak256", "HexString"], 40], 1], StringPartition[addr, 1]}])]

PackageScope[verifyImports]
verifyImports[x_File] := Module[{imports},
    imports = StringCases[Import[x, "Text"], Shortest["import \"" ~~ i__ ~~ "\";"] :> i];
    If[Length[imports] > 0,
        AllTrue[imports, StringMatchQ[#, "./" ~~ __?(StringFreeQ[#, ("<" | ">" | "|" | "\\" | "/" | ":" | "(" | ")" | "&" | ";" | "#" | "?" | "*" | "%" | "\"" | ".")] &) ~~ ".sol"] &],
        True
    ]
]

PackageScope[singleMainQ]
singleMainQ[x_File] := Module[{curlyBrackets = "(?:(?<curly>\\{([^\\{\\}]|(?P>curly))*\\}))"},
    Length[StringCases[Import[x, "Text"], RegularExpression["(contract|library|interface) (\\w|\\s)+(" <> curlyBrackets <> ")"]]] === 1
]

singleMainQ[x_String] := Module[{curlyBrackets = "(?:(?<curly>\\{([^\\{\\}]|(?P>curly))*\\}))"},
    Length[StringCases[x, RegularExpression["(contract|library|interface) (\\w|\\s)+(" <> curlyBrackets <> ")"]]] === 1
]

PackageScope[getConstructorParameterTypes]
getConstructorParameterTypes[x_] := Switch[Head[x],
    File, 
        With[{tmp = StringCases[Import[x, "Text"], "constructor" ~~ ("" | " ") ~~ "(" ~~ Shortest[con__] ~~ ")" :> con]}, 
            If[tmp =!= {},
                First[StringSplit[#, " "]] & /@ (StringSplit[#, ","] & @@ tmp),
                {}
            ]
        ]
    ,String,
        With[{tmp = StringCases[x, "constructor" ~~ ("" | " ") ~~ "(" ~~ Shortest[con__] ~~ ")" :> con]}, 
            If[tmp =!= {},
                First[StringSplit[#, " "]] & /@ (StringSplit[#, ","] & @@ tmp),
                {}
            ]
        ]
]

getConstructorParameterTypes[x_, main_String] := Module[{curlyBrackets = "(?:(?<curly>\\{([^\\{\\}]|(?P>curly))*\\}))", contract},
    Switch[Head[x],
        File,
            contract = StringJoin[Flatten[StringCases[#, "contract " ~~ main ~~ (" " | "{" | "\n") ~~ con__ :> con] &/@ 
                StringCases[Import[x, "Text"], RegularExpression["(contract|library|interface) (\\w|\\s)+(" <> curlyBrackets <> ")"]]
            ]];
            getConstructorParameterTypes[contract]
        ,String,
            contract = StringJoin[Flatten[StringCases[#, "contract " ~~ main ~~ (" " | "{" | "\n") ~~ con__ :> con] &/@ 
                StringCases[x, RegularExpression["(contract|library|interface) (\\w|\\s)+(" <> curlyBrackets <> ")"]]
            ]];
            getConstructorParameterTypes[contract]
    ]
]

PackageScope[validateSolidityType]
validateSolidityType[x_String] := Module[{},
    If[StringMatchQ[x, ("uint" | "int" | "address" | "fixed" | "ufixed" | "function" | "bytes" | "string" | "bool")], Return[True]];
    If[StringMatchQ[x, ("uint" | "int") ~~ Alternatives[ToString[#] & /@ Range[8, 256, 8]]], Return[True]];
    If[StringMatchQ[x, ("ufixed" | "fixed") ~~ Alternatives[ToString[#] & /@ Range[8, 256, 8]] ~~ "x" ~~ Alternatives[ToString[#] & /@ Range[80]]], Return[True]];
    If[StringMatchQ[x, "bytes" ~~ Alternatives[ToString[#] & /@ Range[32]]], Return[True]];
    With[{aux = StringCases[x, f__ ~~ "[" ~~ Alternatives[ToString[#] & /@ Range[9]] ~~ DigitCharacter .. ~~ "]" ~~ EndOfString :> f]}, If[Length[aux] =!= 0, Return[validateSolidityType[First@aux]]]];
    With[{aux = StringCases[x, f__ ~~ "[]" ~~ EndOfString :> f]}, If[Length[aux] =!= 0, If[StringMatchQ[First@aux, ("string" | "bytes")], Return[False], Return[validateSolidityType[First@aux]]]]];
    Return[False];
]

PackageScope[crossValidateSolidityInput]
crossValidateSolidityInput[data_, type_] := Module[{type0, arrayQ, arrayN},
    type0 = First[StringCases[type, StartOfString ~~ x : LetterCharacter .. :> x]];
    arrayQ = StringMatchQ[type, ___ ~~ "[" ~~ ___ ~~ "]"];
    If[arrayQ,
        If[! ListQ[data], Return[False]];
        arrayN = First[StringCases[type, "[" ~~ x___ ~~ "]" :> x]];
        If[arrayN === "",
            Return[AllTrue[Thread[{data, type0}], crossValidateSolidityInput[#[[1]], #[[2]]] &]],
            If[ToExpression[arrayN] =!= Length[data], Return[False]];
            Return[AllTrue[Thread[{data, type0}], crossValidateSolidityInput[#[[1]], #[[2]]] &]]
        ]
        ,
        Switch[type0,
            "int", Return[MatchQ[data, _?IntegerQ]],
            "uint", Return[MatchQ[data, _?Internal`NonNegativeIntegerQ]],
            "address", Return[MatchQ[data, _?(StringQ[#] && StringLength[#] === 40 && hexStringQ[ToLowerCase[#]] &)]],
            "bool", Return[MatchQ[data, _?BooleanQ]],
            "fixed", Return[MatchQ[data, _?NumericQ]],
            "ufixed", Return[MatchQ[data, _?(NumericQ[#] && NonNegative[#] &)]],
            "bytes", Return[MatchQ[data, _?(StringQ[#] && hexStringQ[ToLowerCase[#]] &)]],
            "function", Return[MatchQ[data, _?(StringQ[#] && StringLength[#] === 48 && hexStringQ[ToLowerCase[#]] &)]],
            "string", Return[MatchQ[data, _?StringQ]],_, Return[False]
        ]
    ]
]

PackageScope[validateEthereumAddress]
validateEthereumAddress[x_String?(StringLength[#] === 40 &)] := Return[hexStringQ[ToLowerCase[x]]]
validateEthereumAddress[x_String?(StringLength[#] === 42 &)] := Return[hexStringQ[ToLowerCase[StringDelete[x, "0x"]]]]
validateEthereumAddress[x: {_String..}] := Return[AllTrue[x, validateEthereumAddress]]
validateEthereumAddress[___] := Return[False]