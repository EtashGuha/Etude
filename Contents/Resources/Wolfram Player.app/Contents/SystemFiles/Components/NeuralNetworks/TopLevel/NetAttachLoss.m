Package["NeuralNetworks`"]



PackageScope["GetOutputLayers"]
PackageScope["GetOutputLayerPositions"]

GetOutputLayers[net_NetP] := 
	net @@@ GetOutputLayerPositions[net];

(* associates each output of net with the position of the layer that actually
feeds that output, no matter how deep *)
GetOutputLayerPositions[net_NetP] := 
	AssociationMap[
		traceOutput[net, NetPath[], #]&, 
		OutputNames[net]
	];

(* look at net, examine port oport of it, and recurse, keeping track of the current path via path *)
traceOutput[net_, path_, oport_] := Scope @ Switch[
	net["Type"],
	"Chain",
		last = Last @ Keys @ net["Nodes"];
		traceOutput[net["Nodes", last], Append[path, "Nodes", last], net[["Edges", -1, 2, -1]]],
	"Graph",
		outpath = Lookup[net["Edges"], NetPath["Outputs", oport]];
		node = outpath[[2]]; port = outpath[[4]];
		traceOutput[net["Nodes", node], Append[path, "Nodes", node], port],
	_,
		path
];
	

PackageScope["HasLossPortQ"]

HasLossPortQ[net_] := MemberQ[OutputNames[net], "Loss"];


PackageExport["NetAttachLoss"]

(* for trying things out we don't want to have a fully specified net *)
$enforceFullySpecified = True;
NetAttachLoss::arg1 = "First argument to NetAttachLoss should be a valid net.";
NetAttachLoss[net_, spec_:Automatic, metrics_:{}, enforce_:False] := CatchFailureAsMessage @ Scope[
	If[!ValidNetQ[net], ThrowFailure["arg1"]];
	$enforceFullySpecified = enforce;
	{lossNet, ports, wrapped} = iAttachLoss[net, spec, metrics];
	lossNet
];


PackageScope["AttachLoss"]

AttachLoss[net_, spec_, metrics_] := 
	Cached[iAttachLoss, net, spec, metrics];

iAttachLoss[net_, All, metrics_] := 
	iAttachLoss[net, 
		OutputNames[net], 
		{}
	];

iAttachLoss[net_, spec_String, metrics_] := iAttachLoss[net, {spec}, metrics];

(* %BLOCKED when support for unconnected graph nodes comes, add ability to mix specs. *)

iAttachLoss[net_, ports_List ? StringVectorQ, metrics_] := Scope[
	outputs = Outputs[net];
	General::involoss = "The net has no output port called \"``\" to use as a loss.";
	outs = LookupOr[outputs, ports, ThrowFailure["involoss", #]&];
	If[Total[Map[TRank, outs]] === 0, 
		{net, ports, False}
	,
		iAttachLoss[net, Map[# -> $Verbatim&, ports], metrics]
		(* if some ports aren't scalar, we'll need to attach summation layers to them,
		and hence need a bigger graph *)
	]
];

iAttachLoss[net_, Automatic, metrics_] := Scope[
	outputLayers = KeyDrop[GetOutputLayers[net], metrics];
	lossLayers = Map[TrueQ @ NProperty[#, "IsLoss"]&, outputLayers];

	If[HasLossPortQ[net] && Count[lossLayers, True] <= 1, 
		(* ^ if we don't check that there is no more than a single loss layer here we will be throwing away outputs *)
		Return[{net, {"Loss"}, False}]
	];
	If[And @@ lossLayers,
		(* ^ if all outputs are losses then net is already a fully set up loss network *)
		Return[{net, Keys[outputLayers], False}]
	];
	$automode = True;
	autoLossLayers = KeyValueMap[#1 -> ChooseLossLayer[#1, #2]&, outputLayers];
	iAttachLoss[net, autoLossLayers, metrics]
];

General::invlossptype = "Cannot attach a loss layer to integer-valued port ``."
General::autolossfailure = "Failed to automatically attach a loss to output port `` that is ``."

ChooseLossLayer[portName_, outputLayer_] := Scope[
	outputType = First @ outputLayer["Outputs"]; (* jeromel: "First" -> What if several ports? *)
	stripped = StripCoders[outputType];
	If[MatchQ[TType[outputType], _IndexIntegerT], ThrowFailure["invlossptype", outMsgForm[portName]]];
	res = Quiet @ Switch[
		outputLayer["Type"],
		"Softmax",
			CrossEntropyLossLayer["Index", "Input" -> stripped, "Target" -> toIndexTensor[outputType]],
		"Elementwise" /; outputLayer["Parameters", "Function"] === ValidatedParameter[LogisticSigmoid],
			(* ^ jeromel: this is %HACK that won't work on ElementwiseLayer[LogisticSigmoid[#]&] *)
			CrossEntropyLossLayer["Binary", "Input" -> stripped, "Target" -> toTargetType[outputType]],
		"NetPairEmbedding",
			ContrastiveLossLayer["Input" -> stripped],
		type_ /; $LayerData[type, "IsLoss"],
			$Verbatim,
		_,
			MeanSquaredLossLayer["Input" -> stripped, "Target" -> toTargetType[outputType]]
	];
	If[FailureQ[res], ThrowFailure["autolossfailure", portName, MsgForm[outputType]]];
	res
];

iAttachLoss[net_, lossLayer_ ? ValidNetQ, metrics_] := Scope[
	outputs = DeleteCases[OutputNames[net], Alternatives @@ metrics];
	If[Length[outputs] =!= 1,
		General::plossmultiout = "Cannot attach single loss layer `` when there are multiple outputs: ``. Use \"port\" -> losslayer to specify only a specific output be used.";
		ThrowFailure["plossmultiout", MsgForm[lossLayer], MsgForm[outputs]];
	];
	General::plossdup = "Cannot attach loss layer ``: net  already has an output port called \"Loss\".";
	If[HasLossPortQ[net], ThrowFailure["plossdup", MsgForm[lossLayer]]];	
	iAttachLoss[net, {First[outputs] -> lossLayer}, metrics]
];

iAttachLoss[net_, rule_Rule, metrics_] := iAttachLoss[net, {rule}, metrics];

iAttachLoss[net_NetP, rules:{Repeated[_Rule | _String]}, metrics_] := Scope[
	$otypes = Outputs[net]; $itypes = InputNames[net];
	$layers = <|"Net" -> net|>; 
	$outputLayers := $outputLayers = GetOutputLayers[net]; (* <- for Automatic *)
	$lossOutputs = {}; $edges = {}; $lid = 1;
	Scan[procRule, rules];
	lossNet = CatchFailure[NetGraph, toNetGraph[$layers, Flatten @ $edges, {}]];
	If[FailureQ[lossNet], 
		General::invlossnet = "Could not construct loss network for ``: ``";
		ThrowFailure["invlossnet", MsgForm[net], MsgForm[lossNet]]
	];
	{lossNet, $lossOutputs, True}
];

Clear[procRule];

procRule[portName_String] := procRule[checkOPort[portName] -> $Verbatim];

$scaling = 1.0;
procRule[portName_String -> spec_ -> Scaled[r_ ? NumericQ]] := Scope[
	$scaling = N[r]; 
	procRule[portName -> spec];
];

General::nelossport = "Can't attach a loss to a non-existent output port \"``\".";
checkOPort[portName_] := If[KeyExistsQ[$otypes, portName], portName, ThrowFailure["nelossport", portName]];

procRule[portName_String -> Automatic] := Scope[
	outLayer = Lookup[$outputLayers, checkOPort @ portName];
	$automode = True;
	procRule[portName -> ChooseLossLayer[portName, outLayer]]
];

procRule[portName_String -> Scaled[r_ ? NumericQ]] := Scope[
	$scaling = N[r];
	procRule[checkOPort[portName] -> $Verbatim]
];

procRule[portName_String -> $Verbatim] := Scope[
	src = NetPort["Net", portName]; 
	If[$scaling =!= 1.0,
		addNode[src, portName <> "Scaled", ElementwiseLayer[# * $scaling&]]];
	If[TRank[$otypes @ portName] > 0, 
		addNode[src, portName <> "Summed", SummationLayer[]]];
	AppendTo[$edges, src -> NetPort[portName]];
	AppendTo[$lossOutputs, portName];
];

SetAttributes[addNode, HoldFirst];
addNode[oldName_, newName_, layer_] := (
	$layers[newName] = NData @ layer;
	AppendTo[$edges, oldName -> newName];
	oldName = newName;
)

sowScaled[scaleName_] := 
	$layers[scaleName] = NData @ ElementwiseLayer[# * $scaling&];

procRule[portName_String -> layer_ ? ValidNetQ] := Scope[
	General::nelossport2 = "Can't attach loss `` to non-existent output port \"``\".";
	otype = Lookup[$otypes, portName, ThrowFailure["nelossport2", MsgForm[layer], portName]];
	layer = AttachLossLayer[portName, otype, NData @ layer];
	layerOutputs = Outputs[layer];
	General::nelossport3 = "Specified loss layer ``, which is being attached to port \"``\", should have exactly one output.";
	If[Length[layerOutputs] =!= 1, ThrowFailure["nelossport3", MsgForm @ layer, portName]];
	(* ^ check loss layer has only one output *)
	name = "LossNet" <> IntegerString[$lid++]; hasTarget = KeyExistsQ[layer["Inputs"], "Target"];
	AppendTo[$edges, {
		NetPort[1, portName] -> NetPort[name, "Input"], 
		If[hasTarget, NetPort[portName] -> NetPort[name, "Target"], Nothing]
	}];
	(* ^ inputs to loss layer *)
	$layers[name] = layer;
	(* ^ insert loss layer itself *)
	If[hasTarget && MemberQ[$itypes, portName], (* weird but possible edge case *)
		General::lossioclash = "Cannot attach target loss `` to the output port \"``\" as it has the same name as an input port.";
		ThrowFailure["lossioclash", layer, portName];
	];
	layerOutPort = NetPort[name, First @ Keys @ layerOutputs];
	If[TRank[First @ Outputs[layer]] > 0, 
		addNode[layerOutPort, portName <> "Summed", SummationLayer[]]];
	If[$scaling =!= 1.0,
		addNode[layerOutPort, portName <> "Scaled", ElementwiseLayer[# * $scaling&]]];
	AppendTo[$edges, layerOutPort -> NetPort[portName]];
	AppendTo[$lossOutputs, portName];
];

procRule[portName_String -> layer_] := (
	General::invlosslayer = "Cannot use `` as a loss layer for output port \"``\" as it is not a valid net layer.";
	ThrowFailure["invlosslayer", Shallow[layer], portName];
);

procRule[r_] := (
	General::invlspecel = "Loss rules element `` should be a rule mapping a port name to a loss layer.";
	ThrowFailure["invlspecel", MsgForm[r]];
)

General::invlspec = "Loss specification `` should be one of All, Automatic, a loss layer or net, \"port\", \"port\" -> spec, \"port\" -> spec -> Scaled[s], or a list of these.";
iAttachLoss[net_, spec_, _] := ThrowFailure["invlspec", spec /. n_ ? ValidNetQ :> MsgForm[n]];


toTargetType[type_] := ReplaceAll[type, dec_NetDecoder :> DecoderToEncoder[dec]];

AttachLossLayer[portName_, outputType_, customLayer_] := Scope[
	ldata = customLayer; $pname = portName; $otype = outputType;
	If[ldata["Type"] === "CrossEntropyLoss" && MatchQ[ldata["Parameters", "TargetForm"], "Index" | _EnumT],
		ldata["Parameters", "TargetForm"] = "Index"; 
		ldata["Parameters", "TargetForm"];
		targetType = toIndexTensor[outputType]; (* lower the target rank *)
	,
		targetType = toTargetType[outputType];
	];
	(* check loss layer has the right inputs *)
	inputs = Inputs[ldata];
	inputType = Lookup[inputs, "Input", plossPanic["loss layer does not have an \"Input\" port."]];
	(* check we can attach loss layer to output of net *)
	newInputType = UnifyTypes[inputType, outputType];
	If[FailureQ[newInputType], plossEdgePanic[]];
	ldata["Inputs", "Input"] = newInputType;
	(* attach the reflected output type to the target *)
	If[KeyExistsQ[inputs, "Target"],
		newTargetType = UnifyExternalTypes[inputs["Target"], targetType];
		If[!FailureQ[newTargetType],
			ldata["Inputs", "Target"] = newTargetType 
		];
	];
	(* reinfer the loss layer to make sure the new input types do their thing *)
	ldata2 = InferNData[ldata];
	(* if that fails or was incomplete, complain *)
	If[FailureQ[ldata], plossPanic[TextString @ ldata2]];
	If[$enforceFullySpecified && !ConcreteNetQ[ldata2], plossPanic @ StringForm["`` is not fully specified", UnspecifiedPathString[ldata2]]];
	ldata2
];

General::invploss1 = "Cannot attach `` loss layer `` to ``: ``";
$automode = False;
plossPanic[reason_] := ThrowFailure["invploss1", If[$automode, "automatically chosen", "provided"], MsgForm[ldata], outMsgForm[$pname], TextString @ reason];

General::invploss2 = "`` loss layer ``, which expects ``, is incompatible with ``, which produces ``."
plossEdgePanic[] := ThrowFailure["invploss2", If[$automode, "Automatically chosen", "Provided"], MsgForm[ldata], MsgForm[First @ ldata["Inputs"]], outMsgForm[$pname], MsgForm[$otype]];

toIndexTensor = MatchValues[
	TensorT[{}, _] := Automatic; (* softmax rejects size-1 arrays *)
	TensorT[dims_List, RealT] := TensorT[Most[dims], IndexIntegerT[Last[dims]]];
	TensorT[{n_}, dec_NetDecoder] := TensorT[{n}, DecoderToEncoder[dec]];
	enc_NetDecoder := DecoderToEncoder[enc];
	_ := Automatic;
];

outMsgForm[name_] := MsgForm[NetPath["Outputs", name]];



PackageScope["ToLossFunction"]

General::lfnotrealout = "Output of custom loss function should be a single real number."
General::lfnotsymbolic = "Custom loss function must support symbolic differentiation."

(* this is an experimental WIP, not used anywhere *)
ToLossFunction[f_Function, sz_] := Scope[
    vars = Array[Subscript[x, #]&, sz];
    testres = UnsafeQuietCheck[f[RandomReal[1, sz]]];
    If[!MachineRealQ[testres], ThrowFailure["lfnotrealout"]];
    res = UnsafeQuietCheck[f[vars]];
    If[FailureQ[res], ThrowFailure["lfnotsymbolic"]];
    derivative = D[res, {vars}];
    derivative = Compose[Hold, derivative] /. Subscript[x, i_] :> Part[cvar, i];
    cdfunc = Compile @@ Prepend[derivative, {{cvar, _Real, 1}}];

    {Compile @@ cdargs, Function @@ Prepend[derivative, cvar]}
];