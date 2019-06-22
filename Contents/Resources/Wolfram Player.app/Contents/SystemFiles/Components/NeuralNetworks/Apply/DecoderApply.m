Package["NeuralNetworks`"]


PackageScope["DecoderApply"]

NetDecoder::notarray = "Input to NetDecoder should be a numeric array."
NetDecoder::invshape = "Input to NetDecoder is not ``, or a batch of these."
NetDecoder::invtype = "Input to NetDecoder contained invalid values. It should contain only ``."

DecoderApply[decoder_, input_, prop_] := Catch[iDecoderApply[decoder, input, prop], DecodeFail, DecodeFailureToMessage];

iDecoderApply[decoder_, input_, prop_] := Scope[
	If[!MachineArrayQ[input] && !MachineQ[input] && !VectorQ[input, MachineArrayQ], ThrowFailure["notarray"]];
	ddepth = DecoderDepth[decoder];
	$type = CoderType[decoder];
	ttype = TType @ $type;
	originalInput = input;
	Switch[ttype, 
		IndexIntegerT[_],
			input = Round[input];
			If[MatchQ[First[ttype], _?IntegerQ|Infinity],
				{min, max} = MinMax[input];
				If[min < 1 || max > First[ttype], ThrowFailure["invtype", TypeForm[ttype, True]]]
			],
		_,
			input = ToPackedArray @ N @ input;
	];
	ddims = DecoderDimensions[decoder];
	idims = RaggedDimensions[input];
	idepth = Length[idims];
	decoderF = getDecoderFunction[decoder, prop];
	func = Which[
		(* if decoder depth matches input depth, the input is not batched, so wrap
		the input in a list and then strip off the output list *)
		idepth === ddepth,
			checkDims[idims, ddims];
			First @* decoderF @* List @* listToNA
		,
		(* if it is one greater, the input is batched *)
		idepth === ddepth + 1,
			If[NumericArrayQ[input], 
				input = ArrayUnpack[input];
				(* ^ turn a single, batched NA into a list of NAs *)
			,
				input = Map[listToNA, input];
				(* ^ ensure that we have a list of NAs *)
			];
			checkDims[Rest @ idims, ddims];
			decoderF
		,
		(* if there is no defined decoder depth, assume input is non batched. too bad! *)
		ddepth === None,
			First @* decoderF @* List @* listToNA
		,
		True,
			ThrowFailure["invshape", TypeForm[$type]]
	];
	SwitchNumericArrayFlag[func @ input, originalInput]
];

listToNA[l_List] := toNumericArray[l];
listToNA[e_] := e;

checkDims[_, $Failed] := Null;
checkDims[idims_, ddims_] := If[!MatchQ[idims, ddims], ThrowFailure["invshape", TypeForm[$type]]];

DecoderApply[___] := $Failed;

DecodeFailureToMessage[DecodeFailure[reason_], _] := (
	Message[NetDecoder::invencin, reason]; 
	$Failed
);

PackageScope["DecodeFail"]

SetUsage @ "
DecodeFail['msg$', args$$] should be called by decoder implementations when they encounter a failure."

General::invencin = "Invalid input, ``.";

DecodeFail[msg_String, args__] := DecodeFail[StringForm[msg, args]];
DecodeFail[msg_] := Throw[DecodeFailure @ fromStringForm @ msg, DecodeFail];

PackageScope["ToDecoderFunction"]

ToDecoderFunction[dec_NetDecoder, prop_] :=
	getDecoderFunction[dec, prop];

ToDecoderFunction[SequenceT[_, dec_NetDecoder], prop_] :=
	Map[ArrayUnpack /* ToDecoderFunction[dec, prop]];

(* matteos: we don't need to switch this to Normal depending on 
   $ReturnNumericArray like with getDecoderFunction[_, None], 
   because ToDecoderFunction is only used in net evaluation, 
   (makeDecoder in IO.m), which is already Blocked by 
   SwitchNumericArrayFlag. When this returns Identity,
   the final decoding will use NDArrayGetBatchedSwitched and
   NDArrayGetUnbatchedSwitched *)
ToDecoderFunction[_, Automatic|None] := Identity;

ToDecoderFunction[dec_NetDecoder] := 
	getDecoderFunction[dec, Automatic];

(* asked for a property on a non-decoded type *)
ToDecoderFunction[_, _] := $Failed;


getDecoderFunction[_, None] := Function[If[$ReturnNumericArray, Identity, Normal][#]];

getDecoderFunction[dec_, prop_] :=
	Cached[getDecoderFunctionCached, dec, prop];


getDecoderFunctionCached[e:DecoderP[name_, assoc_, type_], Automatic] := (
	If[assoc["$Version"] === Indeterminate, legacyCoderFail[e]];
	$DecoderData[name, "ToDecoderFunction"][assoc, type]
);

getDecoderFunctionCached[e:DecoderP[name_, assoc_, type_], prop_] := 
	OnFail[
		invProp[name, assoc],
		If[assoc["$Version"] === Indeterminate, legacyCoderFail[e]];
		$DecoderData[name, "ToPropertyDecoderFunction"][assoc, prop, type]
	];

NetDecoder::invprop = "NetDecoder of type `` only supports the following properties: ``."
NetDecoder::noprops = "NetDecoder of type `` does not support properties."

invProp[type_, assoc_] := Match[
	$DecoderData[type, "AvailableProperties"],
	{} :> ThrowFailure[NetDecoder::noprops, type],
	f_Function :> ThrowFailure[NetDecoder::invprop, type, f[assoc]],
	list_List :> ThrowFailure[NetDecoder::invprop, type, list]
];


