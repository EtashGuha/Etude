Package["NeuralNetworks`"]



PackageScope["CreateDecoder"]

SetHoldAllComplete[CreateDecoder];

SetHoldAllComplete[HoldValidDecoderQ]
HoldValidDecoderQ[NetDecoder[_, _, _] ? System`Private`HoldNoEntryQ] := True;
HoldValidDecoderQ[_] := False;

CreateDecoder[NetDecoder[dec_NetDecoder ? HoldValidDecoderQ]] := dec;

CreateDecoder[NetDecoder[enc:NetEncoder[_String, _Association, _ ? ValidTypeQ]]] := 
	EncoderToDecoder[enc];

(* this is for supplying InputDepth option for character-level language models, basically *)
CreateDecoder[NetDecoder[enc:NetEncoder[_String, _Association, _ ? ValidTypeQ], opts__Rule]] := 
	EncoderToDecoder[enc, {opts}];

(* seal an existing encoder *)
CreateDecoder[coder:NetDecoder[type_String, params_Association, input_ ? ValidTypeQ]] := Scope[
	
	(* check the type even exists in this version *)
	data = Lookup[$DecoderData, type, ThrowFailure[NetDecoder::badtype, type]];

	params = UpgradeCoderParams[data["Upgraders"], params, HoldForm @ coder];

	System`Private`ConstructNoEntry[NetDecoder, type, EvalNestedCodersAndLayers @ params, input]
];

(* Upgrade from old decoders *)
CreateDecoder[NetDecoder[name_String, <|"Parameters" -> params_, "Input" -> input_|>]] := 
	System`Private`ConstructNoEntry[NetDecoder, name, params, input /. $TensorUpgradeRule];

CreateDecoder[NetDecoder[{type_String, args___}]] := iCreateDecoder[type, args];
CreateDecoder[NetDecoder[type_String]] := iCreateDecoder[type];

iCreateDecoder[type_String, args___] := Scope[
	
	$currentCoderHead = NetDecoder;
	$currentCoderType = type;

	NetDecoder::badtype = "`` is not a valid NetDecoder type.";
	data = Lookup[$DecoderData, type, ThrowFailure[NetDecoder::badtype, type]];
	
	params = ParseArguments[$coderFormString, False, data, {args}];

	assoc = <|"Parameters" -> params, "Input" -> data["Input"]|>;

	assoc = DoInference[assoc, data["InferenceRules"], List @ data["PostInferenceFunction"]];

	params = Append[assoc["Parameters"], "$Version" -> $NeuralNetworksVersionNumber];

	res = System`Private`ConstructNoEntry[
		NetDecoder, type, params, assoc["Input"]
	];

	res
];

NetDecoder::invargs = "Invalid arguments given in ``."
CreateDecoder[dec_] := ThrowFailure[NetDecoder::invargs, HoldForm @ dec];


PackageScope["EncoderToDecoder"]

EncoderToDecoder[EncoderP[kind_, assoc_, type_], extra_:{}] := Scope[
	spec = $EncoderData[kind, "EncoderToDecoder"][assoc, type];
	If[!ListQ[spec], Return[$Failed]];
	If[spec === None, None, NetDecoder @ Join[spec, extra]]
]
