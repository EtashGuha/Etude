Package["NeuralNetworks`"]


PackageScope["iNetReshape"]
PackageScope["wipeNet"]

$wipeDecoders = "Image"|"Boolean";

iNetReshape[net_Association, rules_List] := Scope[
	$net = net;
	rules = Normal @ Association[Inputs[net], rules];
	$net = wipeNet[$net];
	$net = $net /. DecoderP[kind:$wipeDecoders, param_, type_] :> NetDecoder[kind, ReplacePart[param, "$Dimensions" -> SizeListT[2]], AnyTensorT];
	Scan[applyPortSpec, rules];
	$net
]

DeclareMethod[wipeNet, wipeLayer, wipeContainer, wipeOperator];

wipeLayer[assoc_] := Scope[
	UnpackAssociation[$LayerData[assoc["Type"]], parameters, reshapeParams, hasDynamicPorts, inputs, outputs];
	baseParameters = KeyTake[parameters, reshapeParams];
	AppendTo[assoc["Parameters"], baseParameters];
	If[hasDynamicPorts,
		assoc["Inputs"] = Map[
			Replace[#, Except[DecoderP[$wipeDecoders, __]] :> RealTensorT]&, 
			assoc["Inputs"]
		];
		assoc["Outputs"] = Map[
			Replace[#, Except[DecoderP[$wipeDecoders, __]] :> RealTensorT]&, 
			assoc["Outputs"]
		];
		(* %HACK this is TOO flexible, we will need an API inside layers themselves to declare 
		their shapes after creation *)
	,
		assoc["Inputs"] = IMap[
			Replace[#2, Except[DecoderP[$wipeDecoders, __]] :> Lookup[inputs, #1, AnyTensorT]]&, 
			assoc["Inputs"]
		];
		assoc["Outputs"] = IMap[
			Replace[#2, Except[DecoderP[$wipeDecoders, __]] :> Lookup[outputs, #1, AnyTensorT]]&, 
			assoc["Outputs"]
		];
	];
	assoc
];

wipeContainer[assoc_] := 
	MapReplacePart[assoc, {
		"Inputs" -> wipeTypes,
		"Outputs" -> wipeTypes,
		"Nodes" -> Map[wipeNet]
	}];

wipeOperator[assoc_] := wipeLayer @ MapAtSubNets[wipeNet, assoc];

wipeTypes[assoc_] := Map[
	Replace[#, Except[DecoderP[$wipeDecoders, __]] :> AnyTensorT]&, 
	assoc
];


applyPortSpec[s_String -> t_] := Which[
	!MissingQ[$net["Inputs", s]],
		$net["Inputs", s] = ParseInputSpec[s, TypeT, t],
	!MissingQ[$net["Outputs", s]],
		$net["Outputs", s] = ParseOutputSpec[s, TypeT, t],
	True,
		ThrowFailure["netnopart", s]
];