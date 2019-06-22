Package["NeuralNetworks`"]



PackageExport["FullySpecifiedNetQ"]

SetUsage @ "
FullySpecifiedNetQ[net] gives True if a net's arrays are all initialized and its parameters, inputs, and outputs have concrete types.
The Valid flag will be recursively set on the net's assocs when this is the case to make subsequent lookups free."

DeclareMethod[FullySpecifiedNetQ, LayerFullySpecifiedQ, ContainerFullySpecifiedQ];

Clear[FullySpecifiedNetQ]; (* override default Call behavior for valid-flag fast-path *)
FullySpecifiedNetQ[net_NetP] := 
	System`Private`ValidQ[net] || 
		If[Call[net, FullySpecifiedNetQ], 
			System`Private`SetValid[net]; True, 
			False
		];

FullySpecifiedNetQ[_] := False;

ContainerFullySpecifiedQ[assoc_] := 
	And[
		AllTrue[assoc["Inputs"], FullySpecifiedTypeQ],
		AllTrue[assoc["Outputs"], FullySpecifiedTypeQ],
		AllTrue[assoc["Nodes"], FullySpecifiedNetQ],
		SharedArraysQ[assoc, RawOrRandomArrayQ]
	];

LayerFullySpecifiedQ[assoc_] := ConcreteNetQ[assoc] && InitializedNetQ[assoc];


PackageScope["RawOrRandomArrayQ"]

RawOrRandomArrayQ[e_] := NumericArrayQ[e];
RawOrRandomArrayQ[_SymbolicRandomArray] := True;



PackageExport["FindVarLengthInput"]

FindVarLengthInput[net_NetP] := Scope[
	inputs = Inputs[net];
	If[DynamicDimsQ[inputs], 
		First @ SelectFirstIndex[inputs, DynamicDimsQ],
		$Failed
	]
];


PackageExport["FindUnspecifiedPath"]

FindUnspecifiedPath[net_NetP, arrayReq_] := Scope[
	$path = NetPath[]; 
	Match[arrayReq,
		"Initialized" :> ($unspecAFunc = InitializedOrSharedArrayQ; $unspecSAFunc = RawOrRandomArrayQ),
		"Defined" :> ($unspecAFunc = InitializedOrSharedOrConcreteArrayQ; $unspecSAFunc = ConcreteArrayQ),
		None :> ($unspecAFunc = $unspecAFunc = None)
	];
	Catch[Call[net, ThrowUnspecifiedPart]; None]
];

DeclareMethod[
	ThrowUnspecifiedPart, 
	ThrowUnspecifiedPathInLayer, 
	ThrowUnspecifiedPathInContainer,
	ThrowUnspecifiedPathInOperator
];

ThrowUnspecifiedPathInLayer[layer_] := (
	ThrowUnspecifiedPathInField[ConcreteParameterQ, layer, "Parameters"];
	If[$unspecAFunc =!= None, 
		ThrowUnspecifiedPathInField[$unspecAFunc, layer, "Arrays"];
		If[SharedArraysQ[layer], 
			ThrowUnspecifiedPathInField[$unspecSAFunc, layer, "SharedArrays"]];
	];
	ThrowUnspecifiedPathInField[FullySpecifiedTypeQ, layer, "Inputs"];
	ThrowUnspecifiedPathInField[FullySpecifiedTypeQ, layer, "Outputs"];
)

ThrowUnspecifiedPathInContainer[assoc_] := (
	ScanNodes[ThrowUnspecifiedPart, assoc];
	ThrowUnspecifiedPathInField[FullySpecifiedTypeQ, assoc, "Inputs"];
	ThrowUnspecifiedPathInField[FullySpecifiedTypeQ, assoc, "Outputs"];
	If[$unspecAFunc =!= None && SharedArraysQ[assoc], 
		ThrowUnspecifiedPathInField[$unspecSAFunc, assoc, "SharedArrays"]];
);

ThrowUnspecifiedPathInOperator[assoc_] := (
	ScanSubNets[ThrowUnspecifiedPart, assoc];
	ThrowUnspecifiedPathInLayer[assoc];
);

ThrowUnspecifiedPathInField[test_, layer_, field_] := 
	KeyValueScan[
		If[!test[#2] && !StringStartsQ[#1, "$"], Throw @ Join[$path, NetPath[field, #1]]]&, 
		layer[field]
	];


PackageScope["SharedArraysQ"]

SharedArraysQ[assoc_] := KeyExistsQ[assoc, "SharedArrays"];
SharedArraysQ[assoc_, test_] := !KeyExistsQ[assoc, "SharedArrays"] || AllTrue[assoc["SharedArrays"], test];


PackageExport["NetHasTrainingBehaviorQ"]

PackageScope["HasTrainingBehaviorQ"]

DeclareMethod[HasTrainingBehaviorQ]

NetHasTrainingBehaviorQ[net_NetP] := Catch[
	HasTrainingBehaviorQ[net];
	False
]


PackageExport["ConcreteNetQ"]

SetUsage @ "
ConcreteNetQ[net] gives True if a net's parameters, inputs, and outputs have concrete types."

DeclareMethod[ConcreteNetQ, LayerConcreteQ, ContainerConcreteQ]

Clear[ConcreteNetQ]; (* override default Call behavior for valid-flag fast-path *)
ConcreteNetQ[net_NetP] := System`Private`ValidQ[net] || Call[net, ConcreteNetQ];
ConcreteNetQ[_] := False;

ContainerConcreteQ[assoc_] := And[
	AllTrue[assoc["Nodes"], ConcreteNetQ],
	SharedArraysQ[assoc, ConcreteArrayQ]
]

LayerConcreteQ[assoc_] := And[
	AllTrue[assoc["Arrays"], ConcreteOrSharedArrayQ],
	AllTrue[assoc["Inputs"], FullySpecifiedTypeQ],
	AllTrue[assoc["Outputs"], FullySpecifiedTypeQ],
	AllTrue[assoc["Parameters"], ConcreteParameterQ],
	SharedArraysQ[assoc, ConcreteArrayQ]
];
(* TODO: When we have a separate Types field, we use FullySpecifiedTypeQ there,
and for Parameters we use ConcreteParameterQ, which will check for actual integers, etc*)


PackageExport["InitializedNetQ"]

SetUsage @ "
InitializedNetQ[net] gives True if a net's parameters are all initialized to NumericArrays."

DeclareMethod[InitializedNetQ, LayerInitializedQ, ContainerInitializedQ, OperatorInitializedQ];

Clear[InitializedNetQ]; (* override default Call behavior for valid-flag fast-path *)
InitializedNetQ[net_NetP] := System`Private`ValidQ[net] || Call[net, InitializedNetQ];
InitializedNetQ[_] := False;

LayerInitializedQ[assoc_] := And[
	AllTrue[assoc["Arrays"], InitializedOrSharedArrayQ],
	SharedArraysQ[assoc, RawOrRandomArrayQ]
];

ContainerInitializedQ[assoc_] := AllTrue[assoc["Nodes"], InitializedNetQ];

OperatorInitializedQ[assoc_] := And[
	LayerInitializedQ[assoc],
	VectorQ[
		GetSubNets[assoc],
		InitializedNetQ
	]
];


PackageScope["CheckPortsForBadLengthVars"]

CheckPortsForBadLengthVars[net_] := (
	KeyValueScan[checkPortLV, Inputs[net]];
	KeyValueScan[checkPortLV, Outputs[net]];
);

General::invvardim = "Varying dimension in `` that isn't the first dimension."
checkPortLV[name_, type_] := 
	If[DynamicDimsQ[type] && (!MatchQ[type, TensorT[{_LengthVar, ___}, _]] || Count[type, _LengthVar, Infinity] > 1),
		ThrowFailure["invvardim", FmtPortName[name]];
	];


PackageScope["LiteralNetToPattern"]

General::nographpatt = "NetGraphs are not supported as patterns."
LiteralNetToPattern[_NetGraph ? ValidNetQ] := ThrowFailure["nographpatt"];

LiteralNetToPattern[nc_NetChain ? ValidNetQ] := With[
	{nodes = chainNodes[nc]},
	{chainTest = NetChainLayersEqualTo[
		nodes[[All, "Type"]],
		VisibleParamsAssoc /@ nodes
	]},
	PatternTest[Blank[NetChain], chainTest]
];

NetChainLayersEqualTo[types_, params_][net_NetChain] := Scope[
	nodes = chainNodes[net];
	Length[types] === Length[nodes] && nodes[[All, "Type"]] === types && (VisibleParamsAssoc /@ nodes === params)
];

NetChainLayersEqualTo[_, _][_] := False;

LiteralNetToPattern[sym_Symbol[data_Association, _Association] ? ValidNetQ] := With[
	{paramTest = LayerParametersEqualTo @ VisibleParamsAssoc @ data},
	PatternTest[Blank[sym], paramTest]
];

LayerParametersEqualTo[passoc_][net_] := VisibleParamsAssoc[net] === passoc;

VisibleParamsAssoc[net_NetP] := Sort @ KeySelect[net["Parameters"], StringStartsQ["$"] /* Not];


chainNodes[HoldPattern @ NetChain[assoc_Association, _]] := Values @ assoc["Nodes"];


PackageScope["UninitializedArrayQ"]

UninitializedArrayQ[None | _NumericArray | _DummyArray] := False;
UninitializedArrayQ[NetSharedArray[sname_String]] := UninitializedArrayQ[$AmbientSharedArrays[sname]];
UninitializedArrayQ[_] := True;


PackageScope["ThrowNotSpecifiedFailure"]

ThrowNotSpecifiedFailure[net_, action_, arrayReq_] := Scope[
	part = FindUnspecifiedPath[net, arrayReq];
	If[part === None, ThrowFailure["nfspecunk", action]];
	If[part[[-2]] === "Arrays", 
		ThrowFailure["nninit", action, If[arrayReq === "Initialized", "value", "or partially specified shape"], NetPathString[part]],
		If[$DebugMode,
			ThrowFailure["nfspecdebug", action, NetPathString[part], NData[net] @@ part],
			ThrowFailure["nfspec", action, NetPathString[part]]
		]
	]
];


PackageScope["UnspecifiedPathString"]

UnspecifiedPathString[net_] := NetPathString @ FindUnspecifiedPath[net, "Defined"];
