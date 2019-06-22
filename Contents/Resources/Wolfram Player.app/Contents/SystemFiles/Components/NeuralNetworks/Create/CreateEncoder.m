Package["NeuralNetworks`"]



PackageScope["CreateEncoder"]

SetHoldAllComplete[CreateEncoder];

SetHoldAllComplete[HoldValidEncoderQ]
HoldValidEncoderQ[NetEncoder[_, _, _] ? System`Private`HoldNoEntryQ] := True;
HoldValidEncoderQ[_] := False;

CreateEncoder[NetEncoder[enc_NetEncoder ? HoldValidEncoderQ]] := enc;

CreateEncoder[NetEncoder[dec:NetDecoder[_String, _Association, _ ? ValidTypeQ]]] := 
	DecoderToEncoder[dec];

CreateEncoder[NetEncoder[enc:NetDecoder[_String, _Association, _ ? ValidTypeQ], opts__Rule]] := 
	DecoderToEncoder[enc, {opts}];

(* seal an existing encoder *)
CreateEncoder[coder:NetEncoder[type_String, params_Association, output_ ? ValidTypeQ]] := Scope[

	data = Lookup[$EncoderData, type, ThrowFailure[NetEncoder::badtype, type]];

	params = UpgradeCoderParams[data["Upgraders"], params, HoldForm[coder]];

	res = System`Private`ConstructNoEntry[NetEncoder, type, EvalNestedCodersAndLayers @ params, output];

	If[TrueQ @ data["AcceptsLists"] @ params, System`Private`SetValid @ res, res]
];

(* Upgrade from old encoders *)
CreateEncoder[NetEncoder[name_String, <|"Parameters" -> params_, "Output" -> output_|>]] :=
	System`Private`ConstructNoEntry[NetEncoder, name, params, output /. $TensorUpgradeRule];

CreateEncoder[NetEncoder[{type_String, args___}]] := iCreateEncoder[type, args];
CreateEncoder[NetEncoder[type_String]] := iCreateEncoder[type];

iCreateEncoder[type_String, args___] := Scope[
	
	$currentCoderHead = NetEncoder;
	$currentCoderType = type;

	NetEncoder::badtype = "`` is not a valid NetEncoder type.";
	data = Lookup[$EncoderData, type, ThrowFailure[NetEncoder::badtype, type]];
	
	params = ParseArguments[$coderFormString, False, data, {args}];

	assoc = <|"Parameters" -> params, "Output" -> data["Output"]|>;

	assoc = DoInference[assoc, data["InferenceRules"], List @ data["PostInferenceFunction"]];

	If[!FullySpecifiedTypeQ[assoc["Output"]],
		If[$DebugMode, Print[PrettyForm @ assoc]];
		NetEncoder::decnfs = "Not enough information was provided to fully specify the output of the NetEncoder.";
		ThrowFailure[NetEncoder::decnfs]
	];

	params = Append[assoc["Parameters"], "$Version" -> $NeuralNetworksVersionNumber];
	
	res = System`Private`ConstructNoEntry[
		NetEncoder, type, params, assoc["Output"]
	];

	acceptsListsQ = TrueQ @ data["AcceptsLists"] @ assoc["Parameters"];

	If[acceptsListsQ, System`Private`SetValid[res]];

	res
];

NetEncoder::invargs = "Invalid arguments given in ``."
CreateEncoder[enc_] := ThrowFailure[NetEncoder::invargs, HoldForm @ enc];


PackageScope["DecoderToEncoder"]

DecoderToEncoder[DecoderP[kind_, assoc_, type_], extra_:{}] := Scope[
	spec = $DecoderData[kind, "DecoderToEncoder"][assoc, type];
	If[!ListQ[spec], Return[$Failed]];
	If[spec === None, None, NetEncoder @ Join[spec, extra]]
]