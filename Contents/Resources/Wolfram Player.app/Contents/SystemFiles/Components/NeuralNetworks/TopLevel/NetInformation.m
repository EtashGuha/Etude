Package["NeuralNetworks`"]


PackageExport["NetInformation"]

NetInformation::invnet = "First argument to NetInformation should be a valid net or list of nets."

$NetInformationFunctions = {
	"Arrays" :> getArraysAssociation[$net],
	"ArraysList" :> getArraysList[$net],
	"ArraysCount" :> getArrayCount[$net],
	"ArraysSizes" :> getArraySizes[$net, False, "Size"],
	"ArraysDimensions" :> getArrayDims[$net],

	"ArraysByteCounts" :> getArraySizes[$net, False, "Bytes"],
	"ArraysElementCounts" :> getArraySizes[$net, False, "Elems"],
	"ArraysPositionList" :> getArrayPositions[$net],

	"ArraysTotalSize" :> getArraySizes[$net, True, "Size"],
	"ArraysTotalByteCount" :> getArraySizes[$net, True, "Bytes"],
	"ArraysTotalElementCount" :> getArraySizes[$net, True, "Elems"],
	
	"SharedArraysCount" :> getSharedArrayCount[$net],

	"Layers" :> getLayersAssociation[$net],
	"LayersCount" :> getLayerCount[$net],
	"LayersGraph" :> getLayerGraph[$net],
	"LayersList" :> getLayersList[$net],
	"LayerTypeCounts" :> getLayerTypeCounts[$net],

	"MXNetNodeGraph" :> getMXNetNodeGraph[$net],
	"MXNetNodeGraphPlot" :> getMXNetNodePlot[$net],

	"InputPorts" :> NetInputs[$net],
	"OutputPorts" :> NetOutputs[$net],
	"InputPortNames" :> Keys[Inputs[$net]],
	"OutputPortNames" :> Keys[Outputs[$net]],

	"RecurrentStatesCount" :> Length[GetInteriorStates[$net]],
	"RecurrentStatesPositionList" :> Keys[GetInteriorStates[$net]],

	"InputForm" :> ToNetInputForm[$net],
	"SummaryGraphic" :> NetSummaryGraphic[$fnet, False],
	"FullSummaryGraphic" :> NetSummaryGraphic[$fnet, True],

	"TopologyHash" :> topologyHash[$net],

	"Properties" :> $NetInformationProperties
} // SortBy[First];

$NetInformationProperties = Keys[$NetInformationFunctions];

PackageScope["$NetModelInformationKeys"]
$NetModelInformationKeys = Complement[$NetInformationProperties, {"Arrays", "ArraysList", "Layers", "Properties"}];

(* ^ these are the keys accessible via second arg of NetModel. They will work on uninitialized nets *)


NetInformation::nongvar = "Cannot calculate the node graph of a net with variable-length inputs. Use NetReplacePart to make the inputs fixed length first."
NetInformation::nongspec = "Cannot calculate the node graph of a net whos"

NetInformation[net_, prop_] := CatchFailureAsMessage @ Scope[
	If[!ValidNetQ[net] && !VectorQ[net, ValidNetQ], ThrowFailure["invnet"]];
	Scan[
		If[!ValidPropertyQ[NetInformation, #, $NetInformationProperties], ThrowFailure[]]&,
		ToList[prop]
	];
	If[ListQ[net],
		iNetInformation[#, prop]& /@ net,
		iNetInformation[net, prop]
	]
]

iNetInformation[net_, prop_] := Scope[
	$fnet = net; $net = NData[net];
	Replace[prop, $NetInformationFunctions, {0, 1}]
];

$defaultProps = {
	"LayersCount", 
	"ArraysCount", "SharedArraysCount", 
	"InputPortNames", "OutputPortNames",
	"ArraysTotalElementCount", 
	"ArraysTotalSize"
};

NetInformation[net_] := CatchFailureAsMessage @ Scope[
	If[!ValidNetQ[net], ThrowFailure["invnet"]];
	$net = net;
	InformationPanel["Net information", makePropEntry /@ $defaultProps]
];

makePropEntry[prop_] := Decamel[prop] -> NetInformation[$net, prop];
makePropEntry[{prop_, f_}] := Decamel[prop] -> f[NetInformation[$net, prop]];


(*************************************)
(* Implementations of all properties *)
(*************************************)

getLayersList[net_] := Scope[
	$layers = Bag[]; 
	walkNet[net, BagPush[$layers, ConstructNet[#]]&]; 
	BagContents[$layers]
]

getLayersAssociation[net_] := Scope[
	$layers = Bag[]; 
	walkNet[net, BagPush[$layers, FromNetPath[$path] -> ConstructNet[#]]&]; 
	Association @ BagContents[$layers]
]

getLayerCount[net_] := Scope[
	$count = 0; 
	walkNet[net, $count++&]; 
	$count
]

getLayerTypeCounts[net_] := Scope[
	$types = <||>; 
	walkNet[net, KeyIncrement[$types, $TypeToSymbol[#Type]]&]; 
	$types
]


humanNetArrays[net_] := KeyMap[FromNetPath, NetArrays[net]];

getArrayPositions[net_] :=
	FromNetPath @ Keys @ NetArrays @ net;

getArraysList[net_] := 
	fromArray /@ Values @ NetArrays @ net;

getArrayDims[net_] :=
	arrayDims /@ humanNetArrays @ net;

arrayDims[ra_NumericArray] := Dimensions[ra];
arrayDims[type_] := TDimensions[type, Indeterminate];

getArraysAssociation[net_] := 
	fromArray /@ humanNetArrays @ net;

fromArray[ra_NumericArray] := ra;
fromArray[_] := Automatic;

getSharedArrayCount[net_] := 
	Count[Keys @ NetArrays[net], NetPath["SharedArrays", _]];

getArrayCount[net_] := 
	Length @ NetArrays @ net;

getArraySizes[net_, False, type_] := 
	toCounter[type] /@ arrayPCount /@ humanNetArrays[net];

getArraySizes[net_, True, type_] := 
	toCounter[type] @ Total[arrayPCount /@ NetArrays[net]];

toCounter["Bytes"] = 4 * #&;
toCounter["Elems"] = Identity;
toCounter["Size"] = toNiceByteQuantity[# * 4]&;

arrayPCount[ra_NumericArray] := Times @@ Dimensions[ra];
arrayPCount[SymbolicRandomArray[_, dims_]] := Times @@ dims;
arrayPCount[type_] := ReplaceAll[Times @@ TDimensions[type, {0}], SizeT -> Indeterminate];

toNiceByteQuantity[Indeterminate] = Indeterminate;
toNiceByteQuantity[n_] := Which[
	n < 1000  , Quantity[n, "Bytes"],
	n < 1000^2, Quantity[n/1000., "Kilobytes"],
	n < 1000^3, Quantity[n/1000.^2, "Megabytes"],
	True, Quantity[n/1000.^3, "Gigabytes"]
];


getLayerGraph[net_] := Scope[
	g = LayerDependencyGraph[net];
	edges = EdgeList[g]; 
	vertices = VertexList[g];
	{edges, vertices} = {edges, vertices} /. np_NetPath :> FromNetPath[np];
	Graph[
		vertices, edges, 
		VertexLabels -> Placed["Name", Tooltip],
		GraphLayout -> {"LayeredDigraphEmbedding", "Orientation" -> Left}
	]	
];

getMXNetNodePlot[net_] := (
	checkStaticNet[net];
	NetPlanPlot[net]
)

getMXNetNodeGraph[net_] := (
	checkStaticNet[net]; 
	MXJSONGraph[Developer`ReadRawJSONString @ ToMXJSON[net]["JSON"]]
);

checkStaticNet[net_] := (
	If[ContainsVarSequenceQ	@ Inputs[net], ThrowFailure["nongvar"]];
	If[!ConcreteNetQ[net], ThrowNotSpecifiedFailure[net, "calculate the node graph of", "Defined"]];
)

(* TODO: MOVE THIS INTO UTILS AND FIND OTHER USES FOR IT *)
DeclareMethod[walkNet, walkLayer, walkContainer, walkOperator];
walkNet[net_, f_] := Scope[$walker = f; walkNet[net]];
walkLayer[net_] := $walker[net];
walkContainer[net_] := ScanNodes[walkNet, net];
walkOperator[net_] := ScanSubNets[walkNet, net];




PackageScope["ToNetInputForm"]

ToNetInputForm[net_NetP, head_:HoldForm, maxDepth_:Infinity] := Scope[
	$depth = 2*maxDepth;
	inputs = Map[FromT, StripCoders @ Inputs @ net];
	iform = NetInputForm[net];
	If[NProperty[net, "IsMultiport"],
		AppendTo[iform, "Inputs" -> Values[inputs]],
		KeyValueScan[If[#2 =!= Automatic, AppendTo[iform, #1 -> #2]]&, inputs]
	];
	Apply[head, Compose[Hold, iform] /. HoldForm[h_] :> h]
];

DeclareMethod[NetInputForm, LayerInputForm, ContainerInputForm];

LayerInputForm[assoc_] := Scope[
	UnpackAssociation[$LayerData[assoc["Type"]], minArgCount, posArgCount, parameterDefaults, paramTypes:"Parameters"];
	UnpackAssociation[assoc, parameters];
	paramKeys = Keys[paramTypes];
	If[Length[$path] >= $depth,
		args = {Style["\[Ellipsis]", ShowStringCharacters -> False]},
		args = Table[
			key = paramKeys[[i]];
			If[StringStartsQ[key, "$"] || key === "Dimensionality",
				Nothing,
				value = Replace[parameters[key], ValidatedParameter[v_] :> FromValidatedParameter[v]];
				If[!isDefaultQ[value, parameterDefaults[key]] && value =!= paramTypes[key],
					If[AssociationQ[value] && StringQ[value["Type"]],
						value = Block[{$path = Join[$path, NetPath["Parameters", key]]}, 
							NetInputForm @ value
						];
					];
					If[i <= posArgCount, value, key -> value],
					Nothing
				]
			]
		,
			{i, Length[paramKeys]}
		];
	];
	Compose[HoldForm, NSymbol[assoc]] @@ args
];

LayerInputForm[assoc_] /; assoc["Type"] === "MX" := 
	NeuralNetworks`Layers`MX`toInputForm[assoc];

LayerInputForm[assoc_] /; assoc["Type"] === "NetBidirectional" := 
	HoldForm[NetBidirectionalOperator][
		NetInputForm /@ {
			assoc["Parameters", "ForwardNet"],
			assoc["Parameters", "BackwardNet"]
		},
		assoc["Parameters", "Aggregation"]
	];

isDefaultQ[{val_}, RepeatedInteger[val_]] := True;
isDefaultQ[a_, b_] := a === b;

ContainerInputForm[assoc_] := Scope[
	If[Length[$path] >= $depth-2, 
		UnpackAssociation[assoc, nodes, edges];
		Switch[assoc["Type"],
			"Chain", HoldForm[NetChain][Skeleton @ Length @ nodes],
			"Graph", HoldForm[NetGraph][Skeleton @ Length @ nodes, Skeleton @ Length @ edges]
		]
	,
		nodeSpec = MapAtFields["Nodes", NetInputForm, assoc]["Nodes"];
		If[DigitStringKeysQ[nodeSpec], nodeSpec = Values[nodeSpec]];
		Switch[assoc["Type"],
			"Chain", HoldForm[NetChain][nodeSpec],
			"Graph", HoldForm[NetGraph][nodeSpec, NetGraphEdges[assoc]]
		]
	]
];


PackageScope["NetSummaryGraphic"]

NetSummaryGraphic[nc_NetChain ? ValidNetQ, fullq_:False] := 
	NetSummaryGraphic[NetGraph[nc], fullq];

NetSummaryGraphic[net_NetGraph ? ValidNetQ, fullq_:False] := Scope[
	If[fullq, net = NetFlatten[net, Infinity, "Complete" -> True]];
	net = NData[net];
	LengthVarScope[net, netGraphPlot[net, True]]
];

General::nonetsumm = "Can only produce a summary graphic for a NetChain or NetGraph object."
NetSummaryGraphic[e_, _] := ThrowFailure["nonetsumm"];

topologyHash[net_] := Scope[
	net = RemoveArrays[net] /. {_LengthVar -> Null, Nullable[n_] :> n, SymbolicRandomArray[_, dims_] :> dims};
	Hash[net, Automatic, "Base36String"]
];

