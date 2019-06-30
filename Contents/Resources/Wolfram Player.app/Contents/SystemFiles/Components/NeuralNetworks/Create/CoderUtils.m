Package["NeuralNetworks`"]


PackageScope["ReplaceAllInsideCoders"]

ReplaceAllInsideCoders[expr_, rules_] := Scope[
	coders = DeepCases[expr, CoderP];
	Scan[System`Private`SetNoEntry[#, False]&, coders];
	result = ReplaceAll[expr, rules];
	Scan[System`Private`SetNoEntry, coders];
	result
]	


PackageScope["$currentCoderHead"]
PackageScope["$currentCoderType"]
PackageScope["$coderFormString"]

PackageScope["FailCoder"]

$coderFormString := StringForm["``[{\"``\", \[Ellipsis]}]", $currentCoderHead, $currentCoderType];
General::invcdrarg = "Invalid argument for ``: ``"; 


FailCoder[reason_] := ThrowFailure["invcdrarg", $coderFormString, fromStringForm[reason]];
FailCoder[reason_String, args__] := FailCoder @ StringForm[reason, args];


PackageScope["CoderParamTypes"]

CoderParamTypes[HoldPattern @ NetEncoder[kind_, ___]] := $EncoderData[kind, "Parameters"]
CoderParamTypes[HoldPattern @ NetDecoder[kind_, ___]] := $DecoderData[kind, "Parameters"]


PackageScope["CoderExtract"]

CoderExtract[coder_, All] := Scope[
	ktypes = CoderParamTypes[coder];
	IMap[extractAs[ktypes[#1], #2]&, CoderData[coder]]
];

getAlphaString[assoc_] := MXNetLink`CharacterEncodingAlphabet @ First @ assoc["Encoding"];

CoderExtract[_["Class", assoc_, t_], "Labels"] /; assoc["Labels"] === Automatic := Block[
	{dim = Last[TDimensions[t], None]},
	If[IntegerQ[dim], Range @ dim, Automatic]
];

CoderExtract[_["Characters", assoc_, _], "Alphabet"] :=
	Characters @ getAlphaString @ assoc;

CoderExtract[_["Characters", assoc_, _], "AlphabetString"] :=
	getAlphaString @ assoc;

CoderExtract[_["Characters", assoc_, _], "AlphabetCharacterCodes"] :=
	ToCharacterCode @ getAlphaString @ assoc;

(* see 344440, got to work around the EitherT[Automatc, ListT[...]] in Class decoder *)
extractAs[t_ /; !FreeQ[t, ListT[_, ExpressionT]], value_] := value; 
extractAs[_, value_] := FromInternalValue[value];

CoderExtract[coder_, part_] := If[StringQ[part], 
	extractAs[
		CoderParamTypes[coder][part], 
		CoderData[coder][part]
	],
	ThrowFailure["pspec1", part]
];

PackageScope["ExtractMaxIndex"]

ExtractMaxIndex[tokens_][probs_] := 
	ExtractMaxIndex[tokens, probs];

ExtractMaxIndex[tokens_, probs_] := 
	Extract[tokens, NumericArrayUtilities`PartialOrdering[Normal @ probs, -1]];
 	(* ^ normal is needed because of 361614 *)

PackageScope["ArrayMaxIndex"]

ArrayMaxIndex = Compile[{{values, _Real, 1}},
	If[Length[values] === 0, 0, First@Ordering[values, -1]],
	RuntimeAttributes -> {Listable},
	Parallelization -> True
];

PackageScope["ListMaxIndex"]

ListMaxIndex = Compile[{{values, _Real, 2}},
	Map[If[Length[#] === 0, 0, First @ Ordering[#, -1]]&, values]
];

PackageScope["MaxIndex"]

MaxIndex[input_] := First @ Ordering[input, -1]


PackageScope["DepthWrapper"]

DepthWrapper[f_][in_] := Switch[arrayDepth[in], 1, f[{in}], 2, f[in], _, $Failed];


PackageScope["StripVP"]

StripVP[ValidatedParameter[vp_]] := vp
StripVP[e_] := e;


PackageScope["ChopOrPadSequence"]

ChopOrPadSequence[_LengthVar, _] := Identity;
ChopOrPadSequence[n_Integer, code_] := PadRight[#, n, code]&;
_ChopOrPadSequence := $Unreachable;


PackageScope["ReinferCoder"]

ReinferCoder[head_, name_, assoc_, type_] := ReinferCoder[head, name, assoc, <||>, type];

ReinferCoder[head_, name_, assoc_, assocnew_, type_] := Scope[
	$currentCoderHead = head;
	$currentCoderType = name; 
	isEnc = head === NetEncoder;
	field = If[isEnc, "Output", "Input"];
	data = Lookup[If[isEnc, $EncoderData, $DecoderData], name, ThrowFailure[NetEncoder::badtype, name]];
	(* Take current options and override by new options *)
	assocnew = Join[KeySelect[assoc, !StringStartsQ[#, "$"]&], assocnew];
	(* Set default parameters *)
	assocnew = setDefaultParameters[assocnew, data];
	(* Infer internal parameters that can be infered *)
	assocnew = <|"Parameters" -> assocnew, field -> type|>;
	assocnew = DoInference[assocnew, data["InferenceRules"], List @ data["PostInferenceFunction"]];
	(* Add private parameters that cannot be infered (like $Version) *)
	assocnew["Parameters"] = Join[assoc, assocnew["Parameters"]];
	assoc = assocnew;
	res = System`Private`ConstructNoEntry[head, name, assoc["Parameters"], assoc @ field];
	If[isEnc,
		acceptsListsQ = TrueQ @ data["AcceptsLists"] @ assoc["Parameters"];
		If[acceptsListsQ === False,
			acceptsListsQ = data["AllowBypass"] @ assoc["Parameters"];
		];
		If[acceptsListsQ, System`Private`SetValid[res]];
	];
	res
];
setDefaultParameters[assoc_, data_] := Scope[
	If[Length[assoc] < Length[data["Parameters"]], 
		(* if coder needs upgrading, introduce default values of params *)
		ptypes = data["Parameters"];
		coercions = data["ParameterCoercions"];
		defaults = data["ParameterDefaults"];
		KeyDropFrom[defaults, Keys @ assoc];
		defaults = Association[#1 -> Lookup[coercions, #1, Identity][#2]& @@@ Normal[defaults]];
		assoc = Join[ptypes, defaults, assoc];
	];
	assoc
];

PackageScope["RemapCoderLengthVars"]

RemapCoderLengthVars[e_] := e;
RemapCoderLengthVars[(head:NetEncoder|NetDecoder)[name_, assoc_, type_ ? DynamicDimsQ]] := Scope[
	{assoc, type} = RemapLengthVars[{assoc, type}];
	System`Private`ConstructNoEntry[head, name, assoc, type]
];


PackageScope["ReconstructCoder"]

(* this is a little bit of an optimization, but it also sidesteps 344842 *)
ReconstructCoder[head_, name_, assoc_, type_] := Scope[
	res = System`Private`ConstructNoEntry[head, name, assoc, type];
	If[head === NetEncoder,
		data = Lookup[$EncoderData, name, ThrowFailure[NetEncoder::badtype, name]];
		acceptsListsQ = TrueQ @ data["AcceptsLists"] @ assoc;
		If[acceptsListsQ === False,
			acceptsListsQ = data["AllowBypass"] @ assoc;
		];
		If[acceptsListsQ, System`Private`SetValid[res]];
	];
	res
];


PackageScope["checkStringList"]

checkStringList[input_ ? StringVectorQ] := input;
checkStringList[_] := EncodeFail["input was not a string"];


PackageScope["OutTypeLen"]

OutTypeLen[type_, else_] := Replace[TFirstDim[type], Except[_Integer] -> else];


PackageScope["makeOneHotLookupFunction"]

makeOneHotLookupFunction[n_] := Module[{vectors = IdentityMatrix[n]}, Part[vectors, #]&];


PackageScope["countIntMinType"]

countIntMinType[counts_] := Which[
	counts < 256,
		"UnsignedInteger8",
	counts < 65536,
		"UnsignedInteger16",
	counts < 4294967296,
		"UnsignedInteger32",
	counts < 18446744073709551616,
		"UnsignedInteger64",
	True,
		Panic["Invalid number size. Too large to fit into Integer64"]
]


PackageScope["toNA"]

toNA[type_][input_] := toNA[input, type];

toNA[n_Integer, _] := N[n];
toNA[n_Real, _] := n;
toNA[e_List | e_NumericArray, type_] := UnsafeQuietCheck[toNumericArray[e, type], EncodeFail["no further information is available"]]
(* ^ even though toNA is used in Decoders, failure should never happen in decoders, 
as callers to toNA have implicitly guaranteed their input is packable *)
toNA[e_, _] := If[NumericQ[e], N[e], EncodeFail["no further information is available"]];

PackageScope["toNAList"]

toNAList[type_][e_] := toNAList[e, type];

toNAList[e_ /; VectorQ[e, MachineQ], type_] := e;
toNAList[e_List, type_] := Map[toNA[#, type]&, e];
toNAList[e_, _] := EncodeFail["no further information is available"];


PackageScope["makeReplacer"]

makeReplacer[dispatch_, {}, _] := 
	Replace[#, dispatch, {1}]&;
	
makeReplacer[dispatch_, dims_, extraRank_] := With[
	{depth = Length[dims] + 1, expectedDims = If[extraRank, ToList[_, dims, _], ToList[_, dims]] /. _LengthVar -> _,
	 dimsFunc = If[FreeQ[dims, _LengthVar], arrayDimensions, RaggedDimensions],
	 odims = dims /. lv_LengthVar :> FormatLengthVar[lv]
	},
	Replace[#, dispatch, {depth}]& /* dimsChecker[dimsFunc, expectedDims, odims]
]

dimsChecker[dimsFunc_, expectedDims_, dims_][input_] := 
	If[MatchQ[dimsFunc @ input, expectedDims], input,
		EncodeFail["input should be a list of dimensions ``", dims]];

