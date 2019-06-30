Package["NeuralNetworks`"]


PackageScope["$SeqLenNode"]


PackageScope["GetDynamicDimensionInfo"]

SetUsage @ "
GetDynamicDimensionInfo[lvar$] is deprecated. Please use GetDynamicLengthNode and GetDynamicMaxLength.
* We're only keeping this to allow old MXLayers to keep working.
* Splitting this up allowed us to preserve reshaping where possible based on actual needs."

GetDynamicDimensionInfo[LengthVar[id_Integer]] := (
	$ReshapingIsImpossible = True;
	Lookup[{$MaxSeqLens, $SeqLenNode}, id, LVarPanic[id]]
);

GetDynamicDimensionInfo[n_Integer] := {n, None};


PackageScope["GetDynamicLengthNode"]

SetUsage @ "
GetDynamicLengthNode[lvar$] gets the MXNode[$$] that stores the vector of runtime lengths for a given dynamic dimension.
* See also GetDynamicMaxLength."

GetDynamicLengthNode[LengthVar[id_Integer]] :=
	Lookup[$SeqLenNode, id, LVarPanic[id]];

GetDynamicLengthNode[_Integer] := None;


PackageScope["GetDynamicMaxLength"]

SetUsage @ "
GetDynamicMaxLength[lvar$] gets the maximum size of a dynamic dimension for the bucket currently being written.
* Calling GetDynamicMaxLength during a Writer prevents the entire net from being reshaped. Avoid this at all costs.
* See also GetDynamicMaxLength."

GetDynamicMaxLength[LengthVar[id_]] := (
	$ReshapingIsImpossible = True;
	Lookup[$MaxSeqLens, id, LVarPanic[id]]
);

GetDynamicMaxLength[n_Integer] := n;


PackageScope["SubstituteDynamicMaxLengths"]

SetUsage @ "
SubstituteDynamicMaxLengths[expr$] replaces all LengthVars in expr$ with their maximum values for the current bucket.
* See GetDynamicMaxLength for more information."

SubstituteDynamicMaxLengths[expr_] := expr /. LengthVar[id_] :> (
	$ReshapingIsImpossible = True;
	RuleCondition @ Lookup[$MaxSeqLens, id, LVarPanic[id]]
);


LVarPanic[id_] := Scope @ LengthVarScope[
	vars = LengthVar /@ Keys[$SeqLenNode],
	Panic["MissingDynamicDimInfo", 
		"Cannot locate max length and/or length node for ``.\nInformation is available for: `` (``).", 
		LengthVar[id],
		vars,
		FormatLengthVar /@ vars
	]
];


PackageScope["ContributingNodeFilter"]

(* TODO: have ContributingNodeFilter also take requested metrics *)

ContributingNodeFilter[net_, ports_] := Scope[
	graph = PortConnectivityGraph[net];
	contributing = VertexInComponent[graph, ports];
	all = VertexList[graph];
	Association[
		Thread[all -> False],
		Thread[contributing -> True]
	]
];

findInvalidJSON[result_] := Scope[
	pos = FirstPosition[result, Except[_List | _String | _Integer | _Association | _Real], Heads -> False];
	Panic["InvalidJSON", "Non-JSON value: ``.", Extract[result, Drop[pos, -2]]];
];


PackageScope["$LastMXJSON"]
PackageScope["$LastMXData"]

$LastMXJSON = None;
$LastMXData = None;
(* ^ these are set if $DebugMode is on, or if there is an error producing the JSON or compiling
it to a symbol *)


PackageExport["ToMXJSON"]

SetUsage @ "
ToMXJSON[net$] produces an association containing JSON and other data for the given net. 
The following options can be given:
| 'MaxLength' | maximum length to use for all 'varying' dimensions |
| 'TMode' | True for training mode, False for test mode |
| 'NameRewriter' | A function to rewrite the op names |
The returned association contains the following keys:
| \"JSON\" | the JSON string |
| \"Inputs\" | an association of input port name to mxname |
| \"Outputs\" | an association of output port name to mxname |
| \"ArrayDimensions\" | an association of mxname to dimensions |
| \"Arrays\" | all the arrays |"

Options[ToMXJSON] = {
	"MaxLength" -> 8,
	"TMode" -> False,
	"NameRewriter" -> None
};

ToMXJSON[net_NetP, OptionsPattern[]] := Scope[
	UnpackOptions[maxLength, tMode, nameRewriter];
	Block[{$DebugMode = True, $nameRewriter = nameRewriter},
		meta = {All, MakeSeqIDLens[net, maxLength], tMode, {False, 0, False}, {}}; 
		plan = iToNetPlan[net, meta];
	];
	If[FailureQ[plan], Return[plan]];
	plan = First @ plan;
	UnpackAssociation[plan, inputs, outputs, inputDims, fixedArrays, weightArrays, auxilliaryArrays];
	dims = jsonDims /@ Join[inputDims, fixedArrays, weightArrays];
	Association[
		"JSON" -> $LastMXJSON,
		"Inputs" -> inputs,
		"Outputs" -> outputs,
		"ArrayDimensions" -> dims,
		"Arrays" -> Join[weightArrays, fixedArrays, auxilliaryArrays]
	]
];

jsonDims[NDSequenceArray[dims_List, _]] := dims;
jsonDims[dims_List] := dims;
jsonDims[ra_NumericArray] := Dimensions[ra];


PackageExport["ToMXSymbol"]

ToMXSymbol[args___] := Scope[
	plan = ToNetPlan[args];
	If[FailureQ[plan], Return[plan]];
	plan["Symbol"]
];


PackageExport["ToNetPlan"]
PackageExport["NetPlan"]

SetUsage @ "
ToNetPlan[net$, {outputs$, seqlens$, tmode$, devFlags$, metrics$}] produces an NetPlan[$$] that contains enough 
information to instantiate NetExecutors via ToNetExecutor when given a BatchSize and target device.
ToNetPlan[net$] targets all outputs, uses dummy seqlens if necessary and tries to include the 'ErrorRate' metric.
* outputs$ is either All or a list of output ports that should be included in the DAG.
* outputs$ can also contain NetPath[$$] for internal ports to 'tap off'. 
* seqlens$ is an association whose keys are sequence IDs and values are their corresponding lengths. 
* seqlens$ can also be an integer, that length will be used for all sequence lengths, or a list of rules \
mapping named dynamic dims to their values.
* tmode$ is True if dropout etc should be turned on.
* devFlags$ is a tuple {gpuModeQ$, dtype$, mixedPrecisionQ$}. 
	* gpuModeQ$ is True if we should try use CUDA-specific features like RNN layer.
	* dtype$ is 0 for Real32, 1 for Real64, and 2 for Real16.
	* mixedPrecisionQ$ is True if we can downcast to low precision for certain layers to make use of eg TensorCores.
* metrics$ is a list of the NetPaths that should be collected in a Metrics association.
* Certain layers produce their own metrics magically, requesting a NetPath[layerpath$$, 'magicName'] causes \
these metrics to be emitted in the corresponding Writer. 
* ToNetPlan unrolls recurrent nets when necessary, and so maximum lengths must be communicated using seqlens$.
* The default spec computes all output ports, and uses 4 as the max sequence length (for debugging).
* See the usage of NetPlan for more information.
* ToNetPlan is cached so that we can share a plan both for singleton executor and batch executor.
* Use ToDebugPlan for easier setting of the various flags."

ToNetPlan[net_, meta_] := Cached[iToNetPlan, net, meta];

(* for debugging *)
ToNetPlan[net_] := iToNetPlan[net, {All, 4, False, {False, 0, False}, {}}];

PackageScope["MakeSeqIDLens"]
MakeSeqIDLens[net_, sz_Integer] := 
	ConstantAssociation[First /@ UniqueLengthVars @ Inputs @ net, sz];

General::badseqval = "Sequence with id `` did not have a specified value.";

MakeSeqIDLens[net_, list:{Rule[_String, _Integer]..}] := Scope[
	lens = MapColumn[NameToLengthVar /* First, 1, list];
	ivars = First /@ UniqueLengthVars @ Inputs @ net;
	AssociationThread[ivars, Lookup[lens, ivars, ThrowFailure["badseqval", id]]]
];

MakeSeqIDLens[net_, list:{__Integer}] := Scope[
	ivars = First /@ UniqueLengthVars @ Inputs @ net;
	If[Length[ivars] =!= Length[list], ThrowFailure[]];
	AssociationThread[ivars, list]
];

$nameRewriter = None;

PackageExport["$ForceAttentionUnrolling"]

SetUsage @ "
$ForceAttentionUnrolling (default False) specifies whether AttentionLayer should unroll the application \
of the scoring net."

$ForceAttentionUnrolling = False;

PackageExport["$ForceRNNUnrolling"]

SetUsage @ "
$ForceRNNUnrolling (default False) specifies whether RNN layers should always unroll, instead of using \
ForEach or the built-in RNN operator."

$ForceRNNUnrolling = False;

PackageExport["$DisableRNNFusion"]

SetUsage @ "
$DisableRNNFusion (default False) specifies whether RNN layers should disable use of the built-in RNN \
operator."

$DisableRNNFusion = False;

PackageExport["$DisableReshaping"]

SetUsage @ "
$DisableReshaping (default False) prevents the executor reshaping optimization in which executors for \
different sequence lengths can sometimes by simply reshaping an initial executor."

$DisableReshaping = False;

PackageExport["$DisableMXForEach"]

SetUsage @ "
$DisableMXForEach (default False) specifies whether ToNetPlan should use the experimental MXNet foreach \
operator to avoid unrolling. SowForEach will still work when this is True, but will use unrolling instead." 

$DisableMXForEach = False;

PackageScope["$InsertFinalMXIdentity"]

SetUsage @ "
$InsertFinalMXIdentity (default True) specifies whether ToNetPlan should insert a no-op copy node that \
can be used to guarantee that the output nodes have a specific name." 

$InsertFinalMXIdentity = True;

General::nninvmixed = "WorkingPrecision -> \"Mixed\" is only supported for TargetDevice -> \"GPU\".";

PackageScope["$ReshapingIsImpossible"]

SetUsage @ "
$ReshapingIsImpossible (default False) can be set by certain operations to indicate the net being \
written cannot be reshaped to achieve different dynamic dimension lengths (i.e. is not shape-\
polymorphic). See also $DisableReshaping."

iToNetPlan[net_NetP, {outputs_, seqlens_, tmode_, {gpu_, dtype_, mixed_}, measurementPaths_}] := Timed @ Scope @ Block[
	{$DisableMXForEach = $DisableMXForEach}, 

	If[mixed && !gpu, ThrowFailure["nninvmixed"]];

	TimeLabel["Setup"];	
	If[!ConcreteNetQ[net], 
		Panic["UnspecifiedNet", "Can't compile plan for unspecified net. Unspecified part is ``", FindUnspecifiedPath[net, "Defined"]]
	];

	(* set up the dynamically scoped variables that are used to build the dag *)

	$CurrentNet = net;
	$PathToNode = <||>;				(* NetPath -> node id *)
	$PortFilter = True&;			(* to skip nodes not contributing to desired outputs *)

	$WeightArrays = <||>;			(* mangled name -> NumericArray *)
	$FixedArrays = <||>;			(* unique name to NumericArray *)
	$AuxArrays = <||>;				(* mangled name -> NumericArray *)
	$InputDims = <||>;				(* seq node name -> dims. these are only for inputs! *)

	$HiddenOutputs = <||>;			(* mapping from mxnode to mx output name *)

	$SeqLenNames = <||>;			(* seq id -> seq node name *)
	$MaxSeqLens = seqlens;			(* seq id -> max len *)

	$SeqLenNode = <||>;				(* seq ids -> node id *)
	$SeqCounter = 0;				(* counts consecutive seq ids *)

	$FlattenedNodeCache = <||>;		(* cache reshapes used for batch-flattened nodes *)
	$ZeroArrayBroadcastCache = <||>;		(* cache arrays just instantiated for their sizes *)
	$ZeroArrayCache = <||>;		(* cache arrays just instantiated for their sizes *)
	$BatchIndexArrayCache = <||>;	(* cache used to store arrays that count {0,1,2,3,..} *)

	$BatchReferenceNode = None;		(* makes SowBatchBroadcast choose the right batchsize within NetMaps etc *)

	$StateExpanded = False;			(* whether we are inside a part of the net that for which states have been expanded *)
	$UsedStates = <||>;				(* used states NetPaths -> True *)
	
	$InputStates =  $OutputStates = <||>;	(* logical state name to mx name *)
	$Inputs = $Outputs = <||>;		(* logical port name to mx name *)
	$ReshapeDims = <||>;			(* records the mxname to dim lists for easy reshaping (only when no unrolling) *)

	$TMode = tmode;					(* whether things like dropout should apply *)

	$Metrics = <||>;				(* logical layer pos to mx node *)
	$LogicalWeights = <||>;			(* NetPath to mxname *)
	$LayerArrayCache = <||>;		(* cache the weight arrays per-layer to make unrolling faster *)

	$orphanedStatePath = <||>;		(* tracks whether state was 'orphaned' by being nested in an e.g. Map operator *)
	$WithinOperatorQ = False;		(* used to decide whether to emit e.g. metrics *)
	$GPUMode = gpu;					
	$MixedPrecisionQ = mixed;
	$DType = dtype;
	$DTypeMXName = ToMXNetDataTypeName[dtype];

	$ReshapingIsImpossible = $DisableReshaping; (* if remains False at end, we need plan is shape polymorphic *)
	$EmitInternalNodesFor = <||>;	(* set to True for paths that should emit internal nodes *)

	$MeasurementPaths = measurementPaths; 	(* this is a list of NetPaths that will be stored *)

	(* for convenience of debugging, canonicalize some common shorthands into the full seq len spec *)
	If[IntegerQ[$MaxSeqLens] || ListQ[$MaxSeqLens], $MaxSeqLens = MakeSeqIDLens[net, $MaxSeqLens]];

	(* this is for when we are required by MXNet quirks to request training mode during forward pass in order to
	get gradients, but we don't actually want training mode behavior in dropout etc *)
	If[$TMode === "InferenceGradients", 
		$TMode = False; 
		$GradientsRequired = True;
	,
		$GradientsRequired = $TMode;
	];

	$DisableMXForEach = $TMode; (* foreach currently serializes Forward, introducing severe speed penalties during training *)

	(* this next section sets up the 'port filter', which lets us avoid writing parts of the net that aren't going to be useful *)
	If[outputs === All, 
		onames = Keys @ Outputs[net];
		opaths = Thread @ NetPath["Outputs", onames];
	,
		outputs = outputs /. NetPort[np_, "IntermediateArrays"] :> 
			($EmitInternalNodesFor[ToNetPath[$CurrentNet, np]] = True; Nothing);

		onames = outputs;
		(* TODO: remove the NetPath processing from here completely. The weird exception will be IntermediateArrays, but
		let's introduce a special tag symbol for that like $IntermediateArrays$[NetPath[...]]. *)
		opaths = Replace[outputs, {
			s_String :> NetPath["Outputs", s],
			NetPort[np_] :> ToNetPath[$CurrentNet, np]
		}, {1}];
		If[!MatchQ[opaths, {NetPath[__String]...}], Panic["InvalidOutputSpec", "`` is not a valid output spec.", outputs]];

		If[$EmitInternalNodesFor === <||>, 
			(* ^ the rules become a bit subtle for how we deal with filtering when we only 
			care about internal nodes, so just give up *)
			$PortFilter = ContributingNodeFilter[$CurrentNet, Join[opaths, Cases[measurementPaths, NetPath["Outputs", _]]]];
			(* if the user requested metrics, make sure we don't remove nodes that provide those metrics *)
		];
	];
	otypes = $CurrentNet @@@ opaths;

	If[measurementPaths =!= {},
		indices = Position[opaths, Alternatives @@ (measurementPaths /. 
			NetPath[a__, Alternatives @@ $aggregators] -> NetPath[a]), {1}];
			(* ^ we remove the aggregators because we want to remove the corresponding output port from the graph *)
		If[indices =!= {},
			opaths = Delete[opaths, indices];
			onames = Delete[onames, indices];
			otypes = Delete[otypes, indices];
		];
		(* ^ there  are a number of operations we don't want to do to the metrics nodes, for e.g. getFinalNode 
		 so we remove these metric nodes from the list of outputs. This is only required for NetPlanPlot, since opaths coming from ToTrainer are pre-filtered *)
	];
	
	TimeLabel["MXScan"];
	$sharedArrays = Lookup[$CurrentNet, "SharedArrays", <||>];
	istates = GetInteriorStates[$CurrentNet];
	$statePathToLogical = InvertAssociation[istates];
	(* ^ interior states exist for only those state ports that are totally unbound.
	if either the input or output or both are bound, then there is no  InputStates OR OutputStates entry.
	this could be relaxed, but it kinda makes sense for NetStateObject, which wants to track them in pairs *)
	
	(**************************************************************************)
	(******************** WRITING THE DAG FROM THE NET ************************)
	(**************************************************************************)

	(* set up datastructures to store the DAG *)
	$CurrentGraphNodes = $RootNodes = Bag[];
	$SubgraphDepth = 0;
	$RootArgIDs = Bag[];
	$HiddenOutputNodes = Bag[];
	$ZeroArrayNames = Bag[];

	(* set up the initial arg nodes for inputs and initial states *)
	KeyValueScan[sowInputNode, StripCoders @ Inputs[$CurrentNet]];
	KeyValueScan[sowStateNode, istates];

	(* recursively call the writers of the net *)
	MXScan[$CurrentNet]; 

	(* finish up by 'renaming' the outputs of the net to be stable,
	and collecting metric outputs, if any *)
	heads = Flatten @ Map[getFinalNode, opaths];
	Scan[sowMetricPath, measurementPaths];	

	$RootNodes = BagContents[$RootNodes];
	$RootArgIDs = BagContents[$RootArgIDs];
	$HiddenOutputNodes = BagContents[$HiddenOutputNodes];
	$ZeroArrayNames = BagContents[$ZeroArrayNames];

	(**************************************************************************)

	(* force hidden outputs to go at the end of the heads list *)
	If[$HiddenOutputNodes =!= {},
		heads = Join[heads, $HiddenOutputNodes];
	];
	
	If[$nameRewriter =!= None, 
		$RootNodes = MapAt[$nameRewriter, $RootNodes, {All, "name"}]
	];

	(* create the expression verssion of the JSON that will be passed to MXNet *)
	result = Association[
		"nodes" -> $RootNodes,
		"arg_nodes" -> $RootArgIDs,
		"heads" -> (List @@@ heads),
		"attrs" -> <|"mxnet_version" -> {"int", $MXNetVersion}|>
	];

	If[$debugJSON, Return[result]];

	TimeLabel["ToJSON"];

	(* turn the expression into a string *)
	jsonstr = assocToJSONString[result];

	$PlanLogger[{"ToNetPlan", $MaxSeqLens, ToCompressedBytes @ jsonstr}];

	TimeLabel["CreateSymbol"];

	(* create the MXSymbol from the JSON string *)
	symbol = CatchFailure[General, MXSymbolFromJSON[jsonstr]]; 
	If[Head[symbol] =!= MXSymbol, 
		$LastMXData ^= result;
		$LastMXJSON ^= jsonstr; 
		ThrowFailure["mxneterr"]
	];

	(* pretty sure we can skip this step, and just derive names from the ops themselves *)
	mxouts = MXSymbolOutputs[symbol];

	(* for reference, this whole 'hidden outputs' business is to get around the fact that we
	sometimes need access to derived sequence lengths, which are computed from input seq
	lengths. But those aren't real outputs in the sense of things that make it into $Outputs *)
	If[$HiddenOutputNodes =!= {},
		{hiddennames, mxouts} = TakeDrop[mxouts, -Length[$HiddenOutputNodes]];
		$HiddenOutputs = AssociationThread[$HiddenOutputNodes, hiddennames];
	];

	ScanThread[sowOutputNode, {onames, otypes, mxouts}];

	kmap = If[$nameRewriter === None, Identity, KeyMap[$nameRewriter]];

	(* this template allows us to derive new buckets from old buckets just by reshaping.
	see Layers/ShapePolymorphism.md for more info *)
	reshapeTemplate = If[$ReshapingIsImpossible || $MaxSeqLens === <||>, 
		None,
		CreateFunction[
			Hold[Association] @ Normal[$ReshapeDims] /. Map[
				i = 1; Function[id, LengthVar[id] -> Hold[Part][#2, i++]], 
				Keys[$MaxSeqLens]
			] /. BatchSize -> #1 
		]
	];

	TimeLabel["BuildAssoc"];	
	NetPlan @ Association[
		"Symbol" -> symbol, 
		"WeightArrays" -> kmap @ $WeightArrays, 
		"FixedArrays" -> $FixedArrays,
		"InputDims" -> $InputDims, 
		"ZeroArrays" -> $ZeroArrayNames,
		"AuxilliaryArrays" -> DeleteCases[kmap @ $AuxArrays, None], (* remove any nullable arrays *)
		"Inputs" -> $Inputs,
		"Outputs" -> $Outputs,
		"InputStates" -> $InputStates,
		"OutputStates" -> Map[$HiddenOutputs, $OutputStates],
		"Metrics" -> $Metrics,
		"LogicalWeights"-> $LogicalWeights,
		"ReshapeTemplate" -> reshapeTemplate,
		"NodeCount" -> Length[$RootNodes]
	]
];

SetUsage @ "
NetPlan[<|$$|>] represents an unrolled network, along with information about its weights, inputs, outputs, etc.
* NetPlans abstract the batchsize away.
* The abstract batchsize is represented as the symbol BatchSize within dimension specifications.
* Use ToNetExecutor to turn a plan into an executor.
* Names of weights and layers are mangled from their position in the original net association by riffling '.'.
* Inputs or outputs corresponding to variable-length sequences use a compound NDSequenceArray specification.
* An NetPlan[$$] object contains the following fields:
| 'Symbol' | the MXSymbol that implements the computation DAG |
| 'WeightArrays' | assoc of mx name to learned arrays (as NumericArrays) |
| 'FixedArrays' | assoc of mx name to static arrays (as NumericArrays) |
| 'AuxilliaryArrays' | assoc of mx name to aux arrays (as NumericArrays) |
| 'InputDims' | assoc of mx name to dimensions |
| 'Inputs' | mapping from input port names to mx names |
| 'Outputs' | mapping from output port names to mx names |
| 'InputStates' | mapping from state positions to mx names |
| 'OutputStates' | ditto |
| 'Metrics' | an association from NetPath to mx name |
| 'ReshapeTemplate' | for shape polymorphic nets, a function maps bucket tuple to InputDims association to feed to NetExecutorReshape |"


toOPortName[NetPath["Outputs", out_]] := out;
toOPortName[path_] := NetPort @ FromNetPath[path];

assocToJSONString[assoc_] := Module[{jsonstr},
	jsonstr = UnsafeQuietCheck @ WriteRawJSONString[assoc, "Compact" -> True];
	If[FailureQ[jsonstr], $LastMXData = assoc; findInvalidJSON[assoc]];
	jsonstr
];

compactifyJSONString[string_] :=
	StringReplace[string, ints:("[" ~~ Longest[Repeated[DigitCharacter | "," | WhitespaceCharacter | "[" | "]"]] ~~ "]") :>
		WriteRawJSONString[ReadRawJSONString @ ints, "Compact" -> True]
	];

assocToJSONString[assoc_] /; $DebugMode := Module[{jsonstr},
	$LastMXData = assoc;
	jsonstr = Quiet @ WriteRawJSONString[assoc, "Compact" -> 2];
	If[FailureQ[jsonstr], findInvalidJSON[assoc]];
	jsonstr = compactifyJSONString[jsonstr];
	$LastMXJSON = jsonstr;
	jsonstr
]

getFinalNode[path_] := Scope[
	path2 = ResolvePath[path];
	node = GetPackedNode[path2];
	If[$InsertFinalMXIdentity, 
		Block[{$path = path, $internalid = Null}, SowIdentity[node]],
		node
	]
	(* ^ this accomplishes something important, which is that it causes the output
	to have a stable name based on the output port name. this makes different unrolled nets
	have a stable output name, so we can just rebind based on that name and don't need
	to have the plan communicate a fresh mxname to use for output array rebinding each time.
	this property simplifies NetExecutorInherit. *)
];

sowStateNode[name_, path_] := Scope[
	mangled = MXManglePathWithSeq[path];
	$PathToNode[path] = SowNullNode[mangled];
	$InputDims[mangled] = ToList[BatchSize, getDims[$CurrentNet @@ path]];
	$InputStates[$statePathToLogical @ path] = mangled;
];

sowInputNode[name_, type_] := Scope[
	path = NetPath["Inputs", name];
	node = SowNullNode[mangled = MXManglePath[path]]; 
	$PathToNode[path] = node;
	If[$BatchReferenceNode === None, $BatchReferenceNode ^= node];
	Match[type,
		TensorT[{lv:LengthVar[id_], rest___}, _] :> (
			maxLen = Lookup[$MaxSeqLens, id, LVarPanic[id]];
			dims = getDims @ TensorT[{rest}];
			$InputDims[mangled] = ToList[BatchSize, maxLen, dims];
			$ReshapeDims[mangled] = ToList[BatchSize, lv, dims];
			$Inputs[name] = NDSequenceArray[mangled, getSeqLenName[id]]
		),
		(
			$InputDims[mangled] = ToList[BatchSize, getDims[type]];
			$Inputs[name]  = mangled;
		)
	]
];

getSeqLenName[seqid_] := CacheTo[$SeqLenNames, seqid, sowSeqLenNode[seqid]];

sowSeqLenNode[seqid_] := Scope[
	name = "seq_" <> IntegerString[$SeqCounter++];
	mxid = SowNullNode[name];
	BagPush[$HiddenOutputNodes, Block[{$forcedName = name, $internalid = 0}, SowIdentity[mxid]]];
	$SeqLenNode[seqid] = mxid;
	$ReshapeDims[name] = $InputDims[name] = {BatchSize};
	name
];

genBatchAggregator[agg_, keepDims_:False][node_] := SowNode[agg, node, "axis" -> 0, "keepdims" -> keepDims, "exclude" -> True]
genInstanceAggregator[agg_, keepDims_:False][node_] := SowNode[agg, SowUReshape[node, 1, -1], "axis" -> 1, "keepdims" -> keepDims]

genBatchRMS[node_] := SowSqrt @ genBatchAggregator["mean"][SowSquare @ node]
genInstanceRMS[node_] := SowSqrt @ genInstanceAggregator["mean"][SowSquare @ node]

genBatchStdDev[node_] := Scope[
	mean = genBatchAggregator["mean", True][node];
	len = genBatchAggregator["sum"][SowNode["ones_like", node]];
	SowSqrt @ SowBDivide[genBatchAggregator["sum"][SowSquare @ SowBMinus[node, mean]], SowMinusScalar[len, 1]]
]

genInstanceStdDev[node_] := Scope[
	mean = genInstanceAggregator["mean"][node];
	len = genInstanceAggregator["sum"][SowNode["ones_like", node]];
	SowSqrt @ SowDivide[genInstanceAggregator["sum"][SowSquare @ SowBMinus[node, mean]], SowMinusScalar[len, 1]]
]

genBatchNorm[ord_][node_] := SowNode["norm", SowUReshape[node, 0, -1], "ord" -> ord, "axis" -> 1]
genInstanceNorm[ord_][node_] := SowNode["norm", node, "ord" -> ord]

(* The reason that we need the complex functions above is that the pre-defined node ops tend not to aggregate over the correct dimmensions
a few of them worked but it became easier to define the versions above so that all of the aggregations shared the same code, which helps with testing and maintenance *)

getAggregator[path_] := Scope[
	aggOptions = Switch[Last @ path,
		"$L1Norm", {genInstanceNorm[1], genBatchNorm[1]},
		"$L2Norm", {genInstanceNorm[2], genBatchNorm[2]},
		"$Mean", {genInstanceAggregator["mean"], genBatchAggregator["mean"]},
		"$Total", {genInstanceAggregator["sum"], genBatchAggregator["sum"]},
		"$RootMeanSquare", {genInstanceRMS, genBatchRMS},
		"$StandardDeviation", {genInstanceStdDev, genBatchStdDev},
		"$Min", {genInstanceAggregator["min"], genBatchAggregator["min"]},
		"$Max", {genInstanceAggregator["max"], genBatchAggregator["max"]}
	];
	If[!ArrayPathQ[path], Last @ aggOptions, First @ aggOptions]
];

$aggregators = {"$L1Norm", "$L2Norm", "$Mean", "$Total", "$RootMeanSquare", "$StandardDeviation", "$Min", "$Max"};

sowMetricPath[path:NetPath[__, Alternatives @@ $aggregators]] :=
	sowMetricNode[path, getAggregator[path] @ GetPackedNode @ ResolvePath @ Most @ path, 
	If[!ArrayPathQ[path], Identity, NDNoTotalArray]];

sowMetricPath[path_] := sowMetricNode[path, GetPackedNode @ ResolvePath @ path,
	If[!ArrayPathQ[path], Identity, NDNoTotalArray]];

sowOutputNode[name_, type_, mxname_] := UseMacros[
	$Outputs[name] = Match[
		StripCoders @ type,
		VarSequenceP[id_] :> NDSequenceArray[mxname, getOutSeqLenName[id]],
		mxname
	]
];

getOutSeqLenName[seqid_] := 
	Lookup[$SeqLenNames, seqid, Lookup[$HiddenOutputs, $SeqLenNode[seqid], Panic["MissingOutSeqID"]]];

(* ^ if this output seqid is also an input seqid, then it'll have been cached  in $SeqLenNames from 
toInputSpec, so find it there.  Otherwise, it's a derived seqid, and so we need to lookup up its 
mxname via $HiddenOutputs *)

(* ------ *)

getDims[enc_NetEncoder] := getDims[CoderType[enc]];

General::mxneterr = "An error occured while compiling the net. Please contact technical support."
getDims[type_] := Scope[
	dims = TDimensions[type];
	If[VectorQ[dims, IntegerQ], dims,
		SoftPanic["getDims", "Can't determine dims of ``.", type];
		ThrowFailure["mxneterr"];
	];
	dims
];

(*****************************************************************************)
(*****************************************************************************)
(*****************************************************************************)

NetPlan[assoc_Association][arg1_, rest___] :=
	assoc[arg1][[rest]];

DefineCustomBoxes[NetPlan,
	mx:NetPlan[assoc_Association] :> MakeNetPlanBoxes[mx]
]

MakeNetPlanBoxes[plan:NetPlan[assoc_Association]] := Scope[
	UnpackAssociation[assoc, symbol, nodeCount];
	If[nodeCount < 100,
		plot = With[{symbol = symbol}, Dynamic[MXSymbolPlot[symbol], TrackedSymbols :> {}]];
		plot = Framed[plot, FrameStyle -> LightGray, ImageMargins -> {{0, 0}, {0, 5}}];
	,
		plot = Nothing;
	];
	items = makeItem[#, assoc[#]]& /@ {
		"Inputs", "Outputs", 
		"InputDims", "FixedArrays", "WeightArrays", "AuxilliaryArrays", 
		"Metrics", "LogicalWeights",
		"InputStates", "OutputStates",
		"ReshapeTemplate"
	};
	BoxForm`ArrangeSummaryBox[
		NetPlan, plan, None, items, {plot},
		StandardForm
	]	
];

fmtArrays[e_] := e /. {
	ra_NumericArray :> RuleCondition[
		With[{dims = AngleBracket @@ Dimensions[ra], type = NumericArrayType[ra]}, RawBoxes @ MakeBoxes[toNumericArray[dims, type]]]
	],
	NDArray[n_Integer] :> RawBoxes[RowBox[{"NDArray", "[", n, "]"}]]
};

PackageScope["makeItem"]
makeItem[name_, value_] := 
	BoxForm`MakeSummaryItem[{
		Pane[name <> ": ", {100, Automatic}], 
		itemGrid[fmtArrays @ value]
	}, StandardForm];

itemGrid[<||>|{}] := 
	itemPane[Style["none", Gray]];

itemGrid[a_Association] := If[Length[a] <= 5, PrettyGrid[a],
	FlipView[{
		Framed[
			itemPane @ Row[{" ", Length[a], " items ", Style["(click to see)", Gray], " "}], 
			FrameStyle -> LightGray, FrameMargins -> {{3,3},{2,2}}, ImageMargins -> {{1,0},{0,0}}],
		PrettyGrid[a]
	}]
];

itemPane[e_] := Pane[e, {Automatic,15}, Alignment -> Bottom, BaselinePosition -> Baseline];

itemGrid[a_] := itemPane[Style[a, ShowStringCharacters -> True, FontFamily -> "Source Code Pro", FontSize -> 10]];

SowNullNode[name_] := Scope[
	nodeID = BagLength[$RootNodes];
	BagPush[$RootNodes, Association[
		"op" -> "null", "name" -> name, "inputs" -> {}
	]];
	BagPush[$RootArgIDs, nodeID];
	MXNode[nodeID, 0, 0]
];


PackageScope["SowDerivedSequenceLengthNode"]

SetUsage @ "
SowDerivedSequenceLengthNode[lnode$, lvar$, func$] creates and returns a new length lnode from node$ and associates it with the given lvar$.
* func$ should be a pure function that transforms the old length into the new length via MX operations.
* This is required whenever a layer introduces a new lvar in an output shape that implicitly has some relationship to the lvar in an input shape."

SowDerivedSequenceLengthNode[lenNode_MXNode, LengthVar[newseqid_], f_] := Scope[
	prior = $SeqLenNode[newseqid];
	If[!MissingQ[prior], Return[prior]];
	(* ^ the lennode would already exist if the seqid is an input as well,
	in which case we have already verified it contains the right value via constraints *)
	oldseqid = FirstPosition[$SeqLenNode, lenNode][[1, 1]];
	oldmxid = $SeqLenNode[oldseqid];
	$internalid = 0;
	$forcedName = "seq_" <> IntegerString[$SeqCounter++];
	oldmax = $MaxSeqLens[oldseqid];
	{newmxid, newmax} = List @@ f @ slenVar[oldmxid, oldmax];
	newmxid = SowBlockGrad[newmxid];
	(* this should not be necessary, but see bugs like 344065,
	where  we have gradients coming through what should be non-diff ports of
	layers like CTC and the seq len port of sequence{most,rest,rev,last} *)
	$SeqLenNode[newseqid] = newmxid;
	$MaxSeqLens[newseqid] = newmax;
	BagPush[$HiddenOutputNodes, newmxid];
	newmxid
];

slenVar /: Subtract[slenVar[mx_, val_], n_] :=	slenVar[SowMinusScalar[mx, n], val - n];
slenVar /: Plus[slenVar[mx_, val_], n_] :=		slenVar[SowPlusScalar[mx, n], val + n];
slenVar /: Times[slenVar[mx_, val_], n_] := 	slenVar[SowTimesScalar[mx, n], val * n];
slenVar /: Divide[slenVar[mx_, val_], n_] := 	slenVar[SowDivideScalar[mx, n], val / n];
slenVar /: Floor[slenVar[mx_, val_]] := 		slenVar[SowNode["floor", mx], Floor[val]];


SetRelatedSymbolGroup[NetPlanPlot, ToDebugJSON, ToDebugPlan]

$debugPlanOptionsUsage = 
"* The following options are available:
| 'Metrics' | {} | specifies which metrics should be included. |
| 'MixedPrecisionQ' | False | specifies whether mixed precision evaluation/training can be used. |
| 'MaxSequenceLength' | 3 | specifies the seq len to use. |
| TargetDevice | 'CPU' | specifies whether the plan is to be run on a CUDA GPU. |
| 'TrainingMode' | False | specifies whether training mode should be used. |
| WorkingPrecision | 'Real32' | can be 'Real16', 'Real32' or 'Real64'. |
* The value of 'MaxSequenceLength' can be an integer (applied to all len vars), or an association \
from ID to integer, or a list of rules from named dimension (e.g. 'n') to integer. 
* The value of 'Metrics' should be a list of NetPaths corresponding to specific metrics to capture.
* 'TrainingMode' can also be 'InferenceGradients' to indicate that we are in inference mode but \
need gradients and so must set MXNet's training mode flag owing to issues like (MXNet issue 13264).
"


PackageExport["NetPlanPlot"]

SetUsage[NetPlanPlot, "
NetPlanPlot[NetPlan[$$]] shows the DAG graph of an NetPlan.
NetPlanPlot[net$] makes a plan for net$ and returns its graph.
* Options can be given for the second form to control the kind of plan that is generated.
" <> $debugPlanOptionsUsage]

Options[NetPlanPlot] = Options[ToDebugJSON] = Options[ToDebugPlan] = {
	"Metrics" -> {},
	"MixedPrecisionQ" -> False,
	"MaxSequenceLength" -> 3,
	"TrainingMode" -> False,
	"EdgeBundling" -> True,
	TargetDevice -> "CPU",
	WorkingPrecision -> "Real32"
};

NetPlanPlot::shapeinferr = "Could not infer dimensions of internal nodes. The MXNet error follows:\n``";

NetPlanPlot[plan:HoldPattern[NetPlan[assoc_]], OptionsPattern[]] := Scope[
	symbol = assoc["Symbol"];
	json = MXSymbolToJSON[symbol];
	ebundling = OptionValue["EdgeBundling"];
	mainPlot = MXSymbolPlot[
		symbol, 
		"VertexLabels" -> {Placed["ID",Above], Placed["Type", Below]}, "EdgeBundling" -> ebundling,
		"InternalDimensions" -> Replace[
			NetPlanInternalSizes[plan],
			$Failed :> formatMXError[MXGetLastError[], assoc["Symbol"]]
		]
	];
	subPlots = Cases[json,
		subgraph:KeyValuePattern[{"nodes" -> _, "arg_nodes" -> _, "node_row_ptr" -> _, "heads" -> _}] :>
			MXSymbolPlot[
				MXSymbolFromJSON[subgraph],
				"VertexLabels" -> {Placed["ID",Above], Placed["Type", Below]}, "EdgeBundling" -> ebundling
			],
		{1, Infinity}
	];
	If[subPlots === {}, mainPlot, Row @ Prepend[subPlots, mainPlot]]
];


formatMXError[str_String, sym_MXSymbol] := Scope[
	first = First @ StringSplit[str, "\n", 2];
	json = MXSymbolToJSON[sym];
	replacement = Map[id = 0; If[#name === ".", Nothing, 
		#name -> TemplateApply["\"``\" (vertex ``)", {#name, id++}]]&, json[["nodes"]]];
	StringReplace[first, replacement]
];

formatMXError[_, _] := "Couldn't obtain error";

NetPlanPlot[net_NetP, opts:OptionsPattern[]] := CatchFailureAsMessage @ Scope[

	NetPlanPlot[ToDebugPlan[net, opts], opts]
];


PackageExport["ToDebugJSON"]

SetUsage[ToDebugJSON,
"ToDebugJSON[net$] gives the JSON-like expression that represents the DAG of net.
" <> $debugPlanOptionsUsage]

$debugJSON = False;
ToDebugJSON[net_NetP, opts:OptionsPattern[]] := CatchFailureAsMessage @ Scope[
	$debugJSON = True;
	ToDebugPlan[net, opts]
];


PackageExport["ToDebugPlan"]

SetUsage[ToDebugPlan,
"ToDebugPlan[net$] gives the NetPlan that represents the DAG of net. It is like \
ToNetPlan but allows easy control of the various flags and settings by making \
them available as options. It also does not cache results.
" <> $debugPlanOptionsUsage]

ToDebugPlan[net_NetP, opts:OptionsPattern[]] := CatchFailureAsMessage @ Scope[

	UnpackOptions[maxSequenceLength, trainingMode, targetDevice, workingPrecision, mixedPrecisionQ, metrics];
	
	dtype = ParseWorkingPrecision[workingPrecision];
	gpuModeQ = Match[targetDevice, "CPU" :> False, "GPU" :> True];

	iToNetPlan[net, {All, maxSequenceLength, trainingMode, {gpuModeQ, dtype, mixedPrecisionQ}, metrics}]
]

PackageScope["$GradientsRequired"]
$GradientsRequired = False;

SetUsage @ "
$GradientsRequired is a dynamically-scoped global variable that is True if the current net being written requires gradients \
to be computed. This will never be Fale if $TMode is True, but it can be True when $TMode is false, if a user requests gradients \
when evaluating a net using NetEvaluationMode -> \"Test\" (which is the default). This variable exists to work around MXNet issues."

PackageScope["$TMode"]
$TMode = False;

SetUsage @ "
$TMode is a dynamically-scoped global variable that is True if the current net being written should have training behavior.
* Training behavior implies that regularization layers like Dropout should do their thing."

PackageScope["$GPUMode"]
$GPUMode = False;

SetUsage @ "
$GPUMode is a dynamically-scoped global variable that is True if the current net being written is targeted for an NVIDIA GPU."

PackageScope["$DType"]
$DType = 0;

SetUsage @ "
$DType is a dynamically-scoped global variable that is 2 for 'Real16' precision, \
0 for 'Real32' precision and 1 for 'Real64' precision."

PackageScope["$DTypeMXName"]
$DTypeMXName = "float32";

SetUsage @ "
$DTypeMXName is a dynamically-scoped global variable that is 'float16' for 'Real16' precision, \
'float32' for 'Real32' precision and 'float64' for 'Real64' precision."

PackageScope["$MixedPrecisionQ"]
$MixedPrecisionQ = False;

SetUsage @ "
$MixedPrecisionQ is a dynamically-scoped global variable that is True if the WorkingPrecision \
for certain layers may be lowered in order to leverage speedups such as using TensorCores."

PackageScope["SowInnerNet"]
PackageScope["$WithinOperatorQ"]

SowInnerNet[inputs_, subpath_, net_] := Block[
	{$path = Join[$path, NetPath @@ subpath],
	 $StateExpanded = $StateExpanded || $layerIsStateExpanding, 
	 $WithinOperatorQ = True, $internalid = 0,
	 $BatchReferenceNode = First[inputs, $BatchReferenceNode]
	}, 
	Block[{$path = Append[$path, "Inputs"]}, 
		KeyValueScan[
			Set[$PathToNode[Append[$path, #1]], #2]&,
			inputs
		]
	];
	MXScan[net];
	Association @ Map[
		Last[#] -> PathNode[#]&,
		OutputPaths[net]
	]
];

_SowInnerNet := $Unreachable;


PackageScope["SowNode"]

$internalid = 0;
$forcedName = None;

SetUsage @ "
SowNode['opname$', MXNode[$$]] creates a new node in the current graph, that takes the given node as input.
SowNode['opname$', {node$1, node$2, $$}] provides multiple inputs to the node.
SowNode['opname$', ispec$, 'key$1' -> value$1, 'key$2' -> value$2, $$] provides additional parameter settings.
* SowNode returns an MXNode[$$].
* To see the available operations, run ?MX`*. Note that any $ that occur in those names must be replaced with _.
* Run ?MX`opname to see detailed documentation on any given operation.
* NthOutput[MXNode[$$], n$] can be used to obtain output the n'th output of the MXNode[$$] returned by SowNode, \
if there is more than one.
* The values of parameters can be integers, lists of integers, reals, strings, booleans, None, and other JSON-like\
 data. 
* Various MX operations have shortcut functions that start with the prefix 'Sow', including Plus, Minus, \
Times, Divide, PlusScalar, TimesScalar, DivideScalar, MinusScalar, Dot, FC, Tanh, Sigmoid, Ramp, Sqrt, Softmax, \
SwapAxis, BlockGrad, and others. These can be used to make writer code clearer.
* SowNode is used within the Writer of a layer definition, or within an MXLayer."

(* TODO: fast-path
SowNode[op_, input_MXNode] := 

*)

(* SowNode sows into the $CurrentGraphNodes. This is usually $RootNodes, but if we're in a subgraph,
it is a different bag. 
*)

SowNode[op_, inputs_, params___Rule] := Scope[
	inputTuples = toInputTuple /@ PathNode[ToList @ inputs];
	$op = op; params = Association[params];
	nodeName = If[StringQ[$forcedName], $forcedName, MXManglePathWithSeq[$path]];
	If[IntegerQ[$internalid], nodeName = nodeName <> "$" <> IntegerString[$internalid++]];
	nodeID = BagLength[$CurrentGraphNodes];
	BagPush[$CurrentGraphNodes, Association @ {
		"op" -> op, "name" -> nodeName,
		If[params === <||>, Nothing, "attrs" -> Map[hyperstr, params]],
		"inputs" -> inputTuples
	}];
	MXNode[nodeID, 0, $SubgraphDepth]
];

SowNode[args___] := Panic["InvalidArgs", "Invalid args provided to SowNode: ``.", {args}];

(* these are used for internal nodes, don't have to be too fancy *)
hyperstr[e_] := Match[e,
	_String :> e,
	True :> "true",
	False :> "false",
	i_Integer :> intString[i],
	i:{(_Integer|None)...} :> writeIntList[i],
	0.0 :> "0.0",
	1.0 :> "1.0",
	-1.0 :> "-1.0",
	r_Real :> If[MachineRealQ[r] && Abs[r] < 3.402823466*^38, CDoubleString[r], ThrowFailure["invfpval"]],
	Panic["InvalidSowNode", "At position `` in net, in layer of type ``: cannot serialize `` in SowNode[``, ``, ``] for MXNetFunction of ``.", 
		$path, $CurrentNet @@ Append[$path, "Type"], e, $op, inputTuples, 
		Normal[params$], NetPathForm @ $path];
];

General::invfpval = "The net contained a floating point value that was too large to express with 32-bit machine precision."


PackageScope["SowRNNLoop"]

SowRNNLoop[func_, inputs_, initialStates_, sequenceLen_] := Scope @ Block[
	{$DisableMXForEach = $DisableMXForEach || $ForceRNNUnrolling},
	outputIndices = Range @ Length @ initialStates;
	result = SowForEach[func, sequenceLen, inputs, initialStates, outputIndices, outputIndices];	
	(* result is either a list of one MXNode (BRL and GRU), or two MXNodes (LSTM) *)
	outputs = ToMetaNode[#, sequenceLen, True]& /@ result;
	Prepend[SowMetaLast /@ outputs, First @ outputs]
];


PackageScope["SowForEach"]

SetUsage @ "
SowForEach[f$, len$, {input$1, $$, input$n}] creates a control flow graph that maps f$ over the input$i to produce a list of (transposed) outputs,\
 where f$ is effectively given a list of the elements of input$i at each timestep and should return a list of the output$i at each timestep.
SowForEach[f$, len$, {input$1, $$, input$n}, {state$1, $$, state$m}, {back$1, $$, back$m}, {out$1, $$}] also uses state$i as initial states, \
and where parts back$i of the output of f$i are the corresponding fed-back states. Only the parts out$i are returned.
SowForEach[f$, len$, <|$$|>, <|$$|>, $$] provides the inputs and states in the form of associations.
* The association form of usage can provide the output and fed back parts as string keys rather than integers.
* len$ should be the length of the input tensors on the 0th axis, either an integer or a LengthVar.
* f$ is passed the inputs$ for the two-arg form, or the inputs$ then states$ for the four-arg form.
* f$ is given a single association or a list of arguments.
* f$ should return a list or association. All these values are collected into MXNodes."

makeArgNode[name___] := Association["op" -> "null", "name" -> StringJoin[{name} /. i_Integer :> IntegerString[i]], "inputs" -> {}];

fromAssoc[list_List] := {None, list};
fromAssoc[assoc_Association] := KeysValues[assoc];
toAssoc[keys_, list_] := AssociationThread[keys, list];
toAssoc[None, list_] := list;

ClearAll[SowForEach];

Default[SowForEach, 4] = {};
Default[SowForEach, 5] = {};
Default[SowForEach, 6] = All;

atrans[a_ ? AssociationVectorQ] := AssociationTranspose[a];
atrans[a_Association] := AssociationTranspose[a];
atrans[a_] := Transpose[a];

SowForEach[func_, sequenceLen_, mappedNodes_, initialStateNodes_., feedbackParts_., outputParts_.] /; $DisableMXForEach := Scope[
	
	(* we use Substitute so that we can support LengthVar * n, as needed in AttentionLayer *)
	maxLen = SubstituteDynamicMaxLengths @ sequenceLen;

	inputs = atrans @ Map[SowUnpack[#, maxLen, 0]&, mappedNodes];
	{stateKeys, stateNodes} = fromAssoc @ initialStateNodes;
	
	outputs = ConstantArray[Null, maxLen];

	Assert[Length[initialStateNodes] == Length[feedbackParts]];
	
	MXDo[
		innerInputs = Join[inputs[[i]], toAssoc[stateKeys, stateNodes]];
		innerOutputs = If[AssociationQ[innerInputs], func @ innerInputs, func @@ innerInputs];
		Assert[ListQ[innerOutputs] || AssociationQ[innerOutputs]];
		outputs[[i]] = Part[innerOutputs, outputParts];
		{tmp, stateNodes} = fromAssoc @ Part[innerOutputs, feedbackParts];
	,
		{i, 1, maxLen}
	];

	Map[SowPack[#, True]&, atrans @ outputs]
];

SowForEach[func_, _, mappedNodes_, initialStateNodes_., feedbackParts_., outputParts_.] /; !$DisableMXForEach := Scope[

	mangled = MXManglePathWithSeq[$path];

	{inputKeys, joinedInputs} = fromAssoc @ Join[mappedNodes, initialStateNodes];

	inputTuples = toInputTuple /@ joinedInputs;
	inputCount = Length[inputTuples];
	inputRange = Range[0, inputCount - 1];

	Assert[Length[initialStateNodes] == Length[feedbackParts]];

	inputNames = IntegerString /@ inputRange;

	(* this constitutes the sub-graph data *)
	graphInputs = Bag @ inputTuples;
	graphArgIDs = Bag @ inputRange;
	graphNodes  = Bag @ Map[makeArgNode, inputNames];
	graphExterns = Bag[];
	cacheVar = Module[{var = <||>}, Hold[var]];
	
	InheritedBlock[
		{
			$SubgraphData, $SubgraphDepth, $CurrentGraphNodes,
			$BatchReferenceNode,
			$ZeroArrayBroadcastCache, $FlattenedNodeCache, $BatchIndexArrayCache, $LayerArrayCache
		},
		(* ^ inherited block will let us cheaply modify these dynamically scoped vars.
			We do not want to insert MXNode from inside the subgraph
			into caches that are visible from outside the subgraph.
		*)
		
		$CurrentGraphNodes ^= graphNodes;
		(* ^ so interior SowNodes write to our temporary DAG *)

		AppendTo[$SubgraphData, {graphInputs, graphNodes, graphArgIDs, graphExterns, cacheVar}];
		(* ^ so nested graphs can update us if they need nodes passed through *)

		$SubgraphDepth++;
		(* ^ this is always Length[$SubgraphData] *)

		innerInputs = Thread @ MXNode[inputRange, 0, $SubgraphDepth];

		If[$GradientsRequired, innerInputs = Map[SowIdentity, innerInputs]];
		(* ^ MXForEach appears to have issue with nodes that are shared, see line 281
		of subgraph_op_common.h -- this is a last-minute workaround *)

		innerOutputs = If[inputKeys === None, 
			func @@ innerInputs, 
			func @ toAssoc[inputKeys, innerInputs]
		];
		(* ^ actually call the mapped function *)

		Assert @ MatchQ[innerOutputs, {__MXNode} | _Association];
	];

	returnedNodes = Part[innerOutputs, outputParts];
	returnCount = Length[returnedNodes];

	fedBackNodes = Part[innerOutputs, feedbackParts];
	fedBackCount = Length[fedBackNodes];
	
	{outputKeys, returnedNodes} = fromAssoc @ returnedNodes;
	headTuples = Map[nodeToTuple, Join[returnedNodes, Last @ fromAssoc @ fedBackNodes]];
	(* ^ the heads list must contain first pure outputs, then pure states. 
	TODO: check that we can also return something we also fed back, e.g. headTuples can contain duplicates. *)

	graphArgIDs = BagContents[graphArgIDs];

	subgraph = <|
		"nodes" -> BagContents[graphNodes],
		"arg_nodes" -> graphArgIDs,
		"heads" -> headTuples,
		"attrs" -> <|"mxnet_version" -> {"int", $MXNetVersion}|>
	|>;
	(* ^ this is the subgraph we just captured *)

	newJson = MXSymbolToJSON @ MXSymbolFromJSON @ subgraph;
	KeyDropFrom[newJson, "attrs"];

	newNodes = newJson["nodes"];
	id = 0;
	nameIndex = Association @ Map[newNodes[[# + 1, "name"]] -> id++&, newJson["arg_nodes"]];
	(* ^ make a topo-sorted version of the subgraph, and find where it sends our original arg nodes 
	     (https://github.com/apache/incubator-mxnet/issues/12760) *)

	nodeID = BagLength[$CurrentGraphNodes];

	{dataRange, stateRange} = TakeDrop[inputRange, Length @ mappedNodes];

	BagPush[$CurrentGraphNodes, Association[
		"op" -> "_foreach", 
		"name" -> mangled,
		"attrs" -> <|
			"in_data_locs" -> findLocs[IntegerString /@ dataRange],
			"in_state_locs" -> findLocs[IntegerString /@ stateRange],
			"num_args" -> intString[BagLength[graphInputs] + 1], (* <= not clear why the +1 *)
			"num_out_data" -> intString[returnCount],
			"num_outputs" -> intString[returnCount + fedBackCount],
			"remain_locs" -> findLocs[BagContents[graphExterns]]
		|>,
		"subgraphs" -> {newJson},
		"inputs" -> BagContents[graphInputs]
	]];

	out = Thread @ MXNode[nodeID, Range[0, returnCount - 1], $SubgraphDepth];
	toAssoc[outputKeys, out]
];

findLocs[{}] := "[]";
findLocs[list_] := makeSquareList @ Lookup[nameIndex, list];

makeSquareList[list_] := StringJoin["[", Riffle[IntegerString /@ list, ", "], "]"];

$SubgraphDepth = 0;
$SubgraphData = {}

toInputTuple[MXNode[id_, out_, depth_]] /; $SubgraphDepth === depth := 
	{id, out, 0}

toInputTuple[node_MXNode] := 
	nodeToTuple @ holdCacheTo[
		$SubgraphData[[-1, 5]], node, 
		Nest[lowerNode, node, $SubgraphDepth - Last[node]]]

toInputTuple[mn_MetaNode] := toInputTuple @ mn["Batchwise"];

nodeToTuple[MXNode[id_, out_, _]] := {id, out, 0};

_toInputTuple := $Unreachable;

SetHoldRest[holdCacheTo];
holdCacheTo[Hold[var_], key_, value_] := 
	Lookup[var, key, var[key] = value];

(* lowers a node at depth d to depth d-1 *)
lowerNode[node_] := Scope[
	depth = Last[node] + 1;
	holdCacheTo[
		$SubgraphData[[depth, 5]], node,
		(* ^ first look at the depth d+1 graph's cache to see if it is already lowered *)
		Scope[
			{graphInputs, graphNodes, graphArgIDs, graphExterns, c} = $SubgraphData[[depth]];
			newName = IntegerString @ BagLength @ graphNodes;
			outerNode = nodeToTuple[node];
			innerID = BagLength[graphNodes];
			BagPush[graphInputs, outerNode];
			BagPush[graphArgIDs, innerID];
			BagPush[graphNodes, makeArgNode @ newName];
			BagPush[graphExterns, newName];
			MXNode[innerID, 0, depth]
		]
	]
];


PackageScope["GetBatchIndexArray"]

GetBatchIndexArray[] := 
	CacheTo[$BatchIndexArrayCache, $BatchReferenceNode, makeBatchIndexArray[]]

makeBatchIndexArray[] := SowPlus[
	SowNode["_arange", {}, "start" -> "0", "infer_range" -> True, "dtype" -> $DTypeMXName], 
	SowZeroArray[{}]
]


PackageScope["SowFlatten"]

SetUsage @ "SowFlatten[...]"

SowFlatten[node_] := SowFlatten[node, 1, 0];
SowFlatten[node_, depth_] := SowFlatten[node, depth, 0];

SowFlatten[node_, 0, start_] := node;
SowFlatten[node_, 1, start_] := sowFlatAt[node, start];
SowFlatten[node_, depth_, start_] := Nest[sowFlatAt[#, start]&, node, depth];

sowFlatAt[node_, at_] := 
	CacheTo[
		$FlattenedNodeCache, {node, at},
		SowNode["reshape", node, "shape" -> Join[CTable[0, at], {-3, -2}]]
	];


PackageScope["SowUnflatten"]

SetUsage @ "SowUnflatten[...]"
(* d1, d2, d3, d4, d5, d6, ..., dn
        ^   .   .   .  -----------
    start   (rank=3)
   d1, d2, d3*d4*d5, d6, ... 


ref  = ( d1, d2, d3, d4, d5, d6, ..., dn )
flat = ( d1, d2, d3*d4*d5, ... )


out = ( d1, d2, d3, d4, d5, ... )

*)

SowUnflatten[in_, ref_, depth_:1] := SowUnflatten[in, ref, depth, 0];

SowUnflatten[in_, ref_, 0, _] := in;

SowUnflatten[in_, ref_, depth_, start_] :=
	SowNode[
		"reshape_like", {in, ref}, 
		"lhs_begin" -> start, "lhs_end" -> start+1, "rhs_begin" -> start, "rhs_end" -> 1+start+depth
	];

SowUnflatten[in_, ref_, depth_, inStart_, refStart_] :=
	SowNode[
		"reshape_like", {in, ref}, 
		"lhs_begin" -> inStart, "lhs_end" -> inStart+1, "rhs_begin" -> refStart, "rhs_end" -> 1+refStart+depth
	];

PackageScope["SowFlatten1"]
PackageScope["SowUnflatten1"]
MXLayer::deprecif = "The internal function `` is deprecated. Use `` instead."
SetHoldAll[makeDeprecAlias]
makeDeprecAlias[a_, b_] := (Clear[a]; a := (Message[MXLayer::deprecif, HoldForm[a], HoldForm[b]]; b))
makeDeprecAlias[SowFlatten1, SowFlatten]
makeDeprecAlias[SowUnflatten1, SowUnflatten]

PackageScope["SowFixedArray"]
(* Used for non-batch 'ROM' arrays needed to do certain hacky things like
use spatial transformer as an affine resizer. *)

SetRelatedSymbolGroup[SowFixedArray, SowBatchBroadcast, SowZeroArray];


SetUsage @ "
SowFixedArray['name$', NumericArray[$$]] creates an MXNode that contains the given fixed array.
* This array must have truly fixed dimensions that do not depend on any dynamic dimensions.
* If the array is expensive to compute, you should cache the values used in a memoized function.
* FixedArrays must be given unique names per layer."

SowFixedArray[name_String, data_NumericArray] := Scope[
	name = StringJoin["fixedarray:", MXManglePathWithSeq @ $path, ":", name];
	$FixedArrays[name] = data;
	SowNullNode[name]
];


PackageScope["SowBatchBroadcast"]

SetUsage @ "
SowBatchBroadcast[input$] broadcasts a batch dimension onto input$.
* This is typically used to turn a non-batched node into a batched node.
* The batch dimension is the first dimension of the mxnode referenced by $CurrentBatchReference."

SowBatchBroadcast[input_, attachOne_:True] := (
	If[$BatchReferenceNode === None,  
		(* this code only kicks in if there were no input nodes to the whole net, and so we
		need a dummy batch ref *)
		$BatchReferenceNode = SowNullNode["batch_ref_node"];
		$InputDims["batch_ref_node"] = {BatchSize};
	];
	SowNode["broadcast_like", {
		If[attachOne, SowInsertDim[input, 0], input], 
		$BatchReferenceNode}, 
		"lhs_axes" -> "(0,)", "rhs_axes" -> "(0,)"
	]
);


PackageScope["SowZeroArray"]

SetUsage @ "
SowZeroArray[dims$] creates a MXNode containing a zero array of the given dimensions.
* The array has an additional batch dimension.
* The array has the correct local batch size (e.g. if it is inside a NetMapOperator)."

SowZeroArray[dims_] := 
	CacheTo[$ZeroArrayBroadcastCache, {dims, $BatchReferenceNode}, sowZeroArray1[dims]];

sowZeroArray1[dims_] := Scope[
	node = CacheTo[$ZeroArrayCache, dims, sowZeroArray2[dims]];
	SowBatchBroadcast[node, dims =!= {}]
];

sowZeroArray2[dims_] := Scope[
	name = "zeroarray" <> IntegerString[Length[$ZeroArrayCache]];
	$InputDims[name] = If[dims === {}, {1}, dims];
	BagPush[$ZeroArrayNames, name];
	SowNullNode[name]
];


PackageScope["SowSourceFixup"]

SowSourceFixup[node_, dims_:{}] := SowNode["_plus", {node, SowZeroArray[dims]}];


PackageScope["MXManglePath"]

Clear[MXManglePath];

(* TODO: Make this much more complete *)
$MXEscapes = {"/" -> "^47^", "." -> "^46^"};
$MXDeEscapes = Reverse[$MXEscapes, {2}];

MXManglePath[NetPath[p___, i_Integer]] :=
	MXManglePath[NetPath[p]] <> "_output" <> IntegerString[i];

MXManglePath[NetPath[p___]] := StringJoin[".", Riffle[StringReplace[$MXEscapes] @ {p}, "."]];

MXManglePathWithSeq[path_] := $seqPathStr <> MXManglePath[path];


PackageScope["MXUnmanglePath"]

MXUnmanglePath[s_String] := NetPath @@ StringReplace[$MXDeEscapes] @ StringSplit[s, "."];


PackageScope["SowMXConnections"]

SowMXConnections[rules_] := Scope[
	prefixed = PrefixPorts[rules];
	AssociateTo[$UsedStates, Cases[prefixed, Rule[_, state:NetPath[___, "States", _]] :> Rule[state, True]]];
	AssociateTo[$PathToNode, prefixed]
];

PackageScope["PathNode"]
PackageScope["MXNode"]

SetUsage @ "
MXNodes represent vertices of the computation graph. A particular node in the graph is described by a triplet of numbers. 
The first number is a node ID, the second is which of the outputs of that node is being used, and the third is used during optimisations - for example when fusing ops.
These are the three numbers that are printed when echoing a node."

GetPackedNode[path_] := PathNode[path] /. mn_MetaNode :> mn["Batchwise"];

(* this can recurse, as it makes doing connections easy. *)
SetAttributes[PathNode, Listable];
PathNode[n_MXNode] := n;
PathNode[n_NetPath] := PathNode @ Lookup[$PathToNode, n, Panic["NetPathNotFound", "`` not found. Available paths: ``", n, Keys[$PathToNode]]];
PathNode[e_] := e;

NodeQ[_MXNode] := True;
NodeQ[_] := False;

(* this chases NetPaths until they resolve, returning the last NetPath *)
ResolvePath[n_NetPath] := Match[
	$PathToNode[n],
	n2_NetPath :> %[n2],
	n
];

toOutputPath[name_] := Join[$path, NetPath["Outputs", name]];
toStatePath[name_] := Join[$path, NetPath["States", name]];
toInputPath[name_] := Join[$path, NetPath["Inputs", name]];


MXNode /: MakeBoxes[MXNode[i_Integer, j_Integer, k_Integer], StandardForm] := 
	FrameBox[RowBox[{i, ":", j, ":", k}], Background -> LightRed, BaseStyle -> {FontSize -> 10}, ContentPadding -> False]


PackageScope["GetInput"]
PackageScope["GetInputMetaNode"]

SetUsage @ "
GetInput['name$'] is used within MXNetFunction bodies to get the MXNode[$$] for the input to the current layer having name 'name$'.
GetInput['name$', 'type'] controls what is returned, where 'type' is one of:
| 'Batchwise' | return an MXNode in which the 0th dim is batch dim (default) |
| 'Timewise' | return an MXNode in which 0th dim is sequence dim |
| 'Packed' | return an MXNode, which may be batchwise or timewise |
| 'Unpacked' | return a list of MXNodes |
* the typical input name used by single-input layers is 'Input'.
* GetInput is used within the Writer of a layer definition, or within an MXLayer."

GetInput[name_] := GetInput[name, "Batchwise"];

GetInputMetaNode[name_] := Scope[
	path = ResolvePath @ toInputPath[name];
	Replace[$PathToNode[path], mx_MXNode :> createMetaNode[path, mx]]
];

GetInput[name_, "Batchwise"] := ReplaceAll[
	PathNode @ ResolvePath @ toInputPath[name],
	mn_MetaNode :> mn["Batchwise"]
];

GetInput[name_, form_String] := GetInputMetaNode[name][form];

createMetaNode[path_, node_] := ModuleScope[
	{lnode, maxlen} = Match[
		TFirstDim[$CurrentNet @@ path],
		LengthVar[id_] :> {$SeqLenNode[id], $MaxSeqLens[id]},
		n_Integer :> {None, n},
		$Failed :> {None, 1}
	];
	timewise = unpacked = None; batchwise = node;
	$PathToNode[path] = MetaNode[batchwise, timewise, unpacked, Evaluate @ lnode, Evaluate @ maxlen]
]

GetInput[All] := Map[GetInput, Keys[$CurrentNet @@ Append[$path, "Inputs"]]];
(* ^ used for CatenateLayer and other int-keyed input layers etc *)

PackageScope["GetInputTensor"]
PackageScope["GetInputCount"]
PackageScope["GetInputDims"]
PackageScope["GetInputRank"]
PackageScope["GetOutputDims"]
PackageScope["GetOutputRank"]

GetInputCount[] := Length[$layerInputs];

GetInputTensor[name_] := $layerInputs[name];
GetInputDims[name_] := 	TDimensions[$layerInputs[name]];
GetInputRank[name_] := 	TRank[$layerInputs[name]];
GetInputDims[All] := 	Map[TDimensions, Values[$layerInputs]];
GetInputRank[All] := 	Map[TRank, Values[$layerInputs]];

GetOutputDims[name_] := TDimensions[$layerOutputs[name]];
GetOutputRank[name_] := TRank[$layerOutputs[name]];
GetOutputDims[All] := 	Map[TDimensions, Values[$layerOutputs]];
GetOutputRank[All] := 	Map[TRank, Values[$layerOutputs]];


PackageScope["GetPreviousLayer"]

(* this is used to accomplish fusion *)

GetPreviousLayer[name_, metaQ_:False] := Scope[
	(* get the previous layer *)
	path = Join[$path, NetPath["Inputs", name]];
	While[True,
		next = Lookup[$PathToNode, path, Return[None, Block]];
		If[Head[next] =!= NetPath, Break[], path = next];
	];
	path = Drop[path, -2];
	layer = $CurrentNet @@ path;
	(* get the mxnode that fed the previous layer, which
	assumes the previous layers input is called Input *)
	node = Join[path, NetPath["Inputs", "Input"]];
	node = NestWhile[$PathToNode, node, MatchQ[_NetPath]];
	node = Replace[node, {
		If[metaQ, Nothing, mn_MetaNode :> mn["Batchwise"]], 
		_Missing :> $Failed
	}];
	{layer, node}
];


PackageScope["GetState"]

GetState[name_] := Scope[
	path = toStatePath[name];
	If[$StateExpanded,
		(* states that are expanded correspond to states of layers which are embedded within state-expanding operators,
		meaning there is no easy translation between the (many) values of the initial or final states of the layer
		as seen from inside the operator and those values as seen from outside the operator. e.g. if you map an LSTM
		there is no answer to "what was the final state of the LSTM".

		these kinds of states do not have global NDArrays representing their initial and final values, and so we must
		synthesize zero arrays from them and also mark them as orphaned so that SetState becomes a no-op. *)
		node = Lookup[$PathToNode, path, None];
		If[node === None,
			(* first time node will be None, second time we'll reuse the zero array *)
			dims = TDimensions @ $currentMXScanLayer["States", name];
			node = SowZeroArray[dims];
			$PathToNode[path] = node;
			$orphanedStatePath[path] = True;
		,
			node = PathNode[node];
		];
	,
		node = PathNode[path];
	];
	node
];


PackageScope["GetArray"]

GetArray[name_] := PathNode @ Join[$path, NetPath["Arrays", name]];


PackageScope["SowSubNet"]

SetUsage @ "
SowSubNet['name$', input$] applies a subnet of the current layer to the input. 
SowSubNet['name$', <|'key$'->in$,$$|>] applies a subnet to multiple inputs.
* The subnet must be stored as the parameter called name$ of the current net.
* SowSubNet uses SowInnerNet, which provides slightly more control for edge cases but is more verbose."

SowSubNet[subnetName_String, assoc_] :=
	SowInnerNet[
		If[AssociationQ[assoc], assoc, <|"Input" -> assoc|>], 
		NetPath["Parameters", subnetName], 
		$parameters[subnetName]
	];


PackageScope["MXDo"]

(* we have to name nodes in repeated subgraphs differently for each 
iteration, that is what MXDo does *)

SetAttributes[MXDo, HoldAll];

$seqPathStr = "";
$seqPath = {};

MXDo[expr_, ispec_] := Block[
	{$seqPath = Append[$seqPath, 0], $seqIndex = 0, $seqPathStr},
	Do[$seqPath[[-1]] = IntegerString[++$seqIndex] <> ":"; $seqPathStr = StringJoin[$seqPath]; expr, ispec]
];


PackageScope["SetOutput"]

SetUsage @ "
SetOutput['name$', MXNode[$$]] sets the output named 'name$' of the current layer to the given MXNode[$$].
* a MetaNode[$$] can also be used.
* the typical output name used by single-output layers is 'Output'.
* SetOutput is used within the Writer of a layer definition, or within an MXLayer.
"

SetOutput[name_, out_] :=
	$PathToNode[toOutputPath[name]] = checkNode @ out;

checkNode[n_MXNode | n_MetaNode] := n;
checkNode[e_] := Panic["SetOutput", "MXNetFunction tried to set an ouput to ``.", e];

PackageScope["$MeasurementPaths"]

PackageScope["ShouldSetMetricQ"]

ShouldSetMetricQ[name_] := 
	!$WithinOperatorQ && MemberQ[$MeasurementPaths, Append[$path, name]];

PackageScope["SetMetric"]

SetUsage @ "
SetMetric['name$', MXNode[$$]] sets the built-in metric named 'metric$' of the current layer to the given MXNode[$$].
* ShouldSetMetricQ['name$'] should be tested before the node is generated and SetMetric is called, to ensure \
the user actually wants this metric. Errors will occur if this is not done.
* At this level of the framework, metrics are identified purely by NetPath[$$]. Higher-level metrics are built \
on top of the collected values from these paths via various finalizers, see TrainingMetrics.m"

(* TODO: maybe metrics should live as
NetPath[..., "Metrics", "XXX"] rather than just NetPath[..., "XXX"]
*)

SetMetric[name_String, node_MXNode, f_:Identity] := 
	sowMetricNode[Append[$path, name], node, f];

SetMetric[name_String -> k_Integer, node_MXNode, f_:Identity] := 
	sowMetricNode[Join[$path, NetPath[name, k]], node, f];

(* metrics are a combination of paths for ordinary layer outputs
and special metric paths sowed by e.g. CELoss. CELoss will call SetMetric.
later, ToPlan will call sowMetricNode on the same path, the second call
should be a no-op *)
sowMetricNode[path_, node_, _] /; KeyExistsQ[$Metrics, path] := 
	Null; 

sowMetricNode[path_, node_, f_] := Block[
	{$forcedName = MXManglePath[path], $internalid = Null}, 
	blocked = SowBlockGrad @ node;
	BagPush[$HiddenOutputNodes, blocked]; 
	$PathToNode[path] = blocked;
	$Metrics[path] = f[$forcedName <> "_output"];
];


PackageScope["SetState"]

SetState[name_, spec_MXNode] := Scope[
	path = toStatePath[name];
	If[Lookup[$UsedStates, path, False],
		$PathToNode[path] = spec
	,
		If[$orphanedStatePath[path] || !KeyExistsQ[$statePathToLogical, path], Return[]];
		spec = Block[{$path = Append[path, "final"], $internalid = Null}, SowNode["BlockGrad", spec]];
		BagPush[$HiddenOutputNodes, spec];
		$OutputStates[$statePathToLogical[path]] = spec;
	];
];

SetState[___] := $Unreachable;

DeclareMethod[MXScan, MXScanLayer, MXScanContainer];

MXScanContainer[assoc_] := (
	If[!$PortFilter[$path] && !MemberQ[$PortFilter /@ InputPaths[assoc], True], Return[]];
	SowMXConnections[assoc["Edges"]]; 
	ScanNodes[MXScan, assoc];
);

MXScanLayer[assoc_] := Scope[
	If[!$PortFilter[$path], Return[]];
	type = assoc["Type"]; $currentMXScanLayer = assoc;
	UnpackAssociation[assoc, 
		type, 
		$layerOutputs:"Outputs", 
		$layerInputs:"Inputs", 
		$parameters:"Parameters", 
		$arrays:"Arrays"
	];
	UnpackAssociation[$LayerData[type], 
		auxArrays, writer, fusedRNNArray, 
		$layerIsStateExpanding:"StateExpanding", 
		$mxinfo:"MXNet"
	];
	If[EmptyQ[$arrays], 
		$arrayIDs = {} (* if layer has no arrays, nothing to do *)
	, 
		(* if we've encountered this layer before, get its array nodes *)
		arrayNodes = CacheTo[$LayerArrayCache, $path,
			(* otherwise compute them and cache them *)
			If[fusedRNNArray =!= None && canFuseRNNQ[],
				dumpFusedRNNArray @ fusedRNNArray @ $parameters,
				dumpArrays[]
			]
		];
		(* store the IDs for use in MXWriteDefault *)
		$arrayIDs = DeleteCases[None] @ Values[arrayNodes]; 
		(* and add the list of nodes to the arguments passed to the writer *)
		AssociateTo[$parameters, arrayNodes];
	];
	$internalid ^= 0; 
	If[Lookup[$EmitInternalNodesFor, $path, False],
		writeWithInternalNodes[writer, $parameters]
	,
		Catch[writer[$parameters], writerWrapper];
	]
];
(* ^ a pedantic note about the per-layer array caching above: its pointless for
layers with FusedRNNArrays as they are all recurrent and cannot be used in an operator
anyway. but maybe that will change in future *)
(* jeromel: TODO remove this restriction *)

writeWithInternalNodes[writer_, params_] := Scope[
	startNode = BagLength[$RootNodes];
	Catch[writer[params], writerWrapper];
	ids = Range[startNode, BagLength[$RootNodes] - 1];
	nodes = BagPart[$RootNodes, ids+1];
	BagPush[$HiddenOutputNodes, Thread @ MXNode[ids, 0, 0], 1];
	assoc = Association[If[#op === "null", #name -> #name, StringJoin[#name, "[", #op, "]"] -> StringJoin[#name, "_output"]] & /@ nodes];
	assoc["Graph"] = plotSubGraph[startNode, nodes];
	$Metrics[$path] ^= assoc;
];
(* ^ emit an association into $Metrics that maps the internal names into their
actaul outputs  *)

plotSubGraph[startNode_, nodes_] := Scope[
	nodes = nodes;
	sources = Union @@ nodes[[All, "inputs"]];
	inputs = Select[sources, First[#] < startNode&];
	numInputs = Length[inputs];
	offset = startNode - numInputs;
	nodes[[All, "inputs", All, 1]] -= offset;
	nodes[[All, "inputs"]] = nodes[[All, "inputs"]] /. n_Integer ? Negative :> 
		IndexOf[sources, n + offset]-1;
	nodes = Join[
		Table[<|"name" -> "", "op" -> "null", "inputs" -> {}|>, numInputs],
		nodes
	];
	json = <|"nodes" -> nodes, "arg_nodes" -> {}, "heads" -> {}|>;
	MXJSONPlot[json,
		"VertexLabels" -> {Placed["Name", Above], Placed["Type", Below]}
	]
];


PackageScope["MXWriteDefault"]
PackageScope["MXWriteDefaultAndReturn"]

MXWriteDefaultAndReturn[] := (MXWriteDefault[]; Throw[Null, writerWrapper]);

MXWriteDefault[] := MXWriteDefault[Null];

MXWriteDefault[_] := Scope[
	If[!$PortFilter[$path], Return[Nothing]];
	mxid = SowCurrentNode @ ToList[GetInput /@ Keys @ $layerInputs, $arrayIDs];
	oport = NetPath["Outputs", First @ Keys @ $layerOutputs];
	$PathToNode[Join[$path, oport]] = mxid
];


PackageScope["SowCurrentNode"]

SowCurrentNode[inputs_] := Scope[
	op = $mxinfo["Name"];
	inputIDs = toInputTuple /@ inputs;
	paramWriter = toLayerWriter[type];
	paramStrings = Association @ paramWriter[$parameters, $arrays];
	nodeName = MXManglePathWithSeq[$path];
	nodeID = BagLength[$CurrentGraphNodes];
	BagPush[$CurrentGraphNodes, Association[
		"op" -> op, "name" -> nodeName, 
		"attrs" -> paramStrings, "inputs" -> inputIDs
	]];
	MXNode[nodeID, 0, $SubgraphDepth]
];

dumpArrays[] := Block[
	{$path = Join[$path, NetPath["Arrays", Null]]},
	$parameters["FusedRNNArray"] = None;
	KeyValueMap[sowArrayNode, $arrays]
];

$fusableDropoutP = ValidatedParameter[None | <|"VariationalWeights" -> _Real|> | _Real];

(* we can only fuse if none of these conditions apply *)
canFuseRNNQ[] := Not @ Or[
	$ForceRNNUnrolling, (* did the user *want* unrolling *)
	$DisableRNNFusion, (* did user disable fusing *)
	$TMode && !MatchQ[$parameters["Dropout"], $fusableDropoutP], (* is non-fusable dropout going to be applied? *)
	MemberQ[$arrays, _NetSharedArray], (* are the arrays being shared? *)
	$GradientsRequired && !$TMode, (* do we need to work around https://github.com/apache/incubator-mxnet/issues/13264 *)
	And[ (* do we need to work around https://bugs.wolfram.com/show?number=348910 *)
		TrueQ @ Lookup[$parameters, "$CellStateConnectedQ"],
		MatchQ[Lookup[$parameters, "$SequenceLength"], LengthVar[_]]
	]
];

getVarWeightP[<|"VariationalWeights" -> p_|>] := N @ p;
getVarWeightP[p_ ? NumberQ] := N @ p;
getVarWeightP[_] := None;

(* so when we are unrolling and have to produce a new plan for each bucket, these fused arrays that are placed in the plan aren't
actaully used... *)
dumpFusedRNNArray[spec_] := Scope[

	$fusedname = StringJoin["fixedarray:", MXManglePathWithSeq @ $path, ":FusedWeights"];
	(* ^ this name must match what SowFixedArray will later chose *)

	(* collect the contents of the weight arrays into one flat array, along with the dropout mask (if needed) *)
	$fusedRNNArrayData = Bag[]; 
	$wdropout = If[$TMode, getVarWeightP @ StripVP @ $parameters["Dropout"], None];
	$weightDropoutMaskData = Bag[];
	$offset = 1; 
	Block[{$path = Join[$path, NetPath["Arrays", Null]]},
		Scan[sowFusedRNNArrayNode, spec];
	];

	(* sow the fused weights we collected *) 
	fusedArrayNode = SowFixedArray["FusedWeights", toNumericArray[BagContents @ $fusedRNNArrayData]];

	If[$wdropout =!= None,
		(* the weightDropoutMask prevents biases from being dropped out in the fused array *)
		weightDropoutMaskNode = SowFixedArray["FusedDropoutMask", toNumericArray[BagContents @ $weightDropoutMaskData]];
		weightMask = SowNode["_maximum", {SowDropConnect @ $wdropout, weightDropoutMaskNode}];
		fusedArrayNode = SowHad[fusedArrayNode, weightMask];
	];

	Append[
		Thread[Keys[$arrays] -> None],
		"FusedRNNArray" -> fusedArrayNode
	]
]

PackageExport["NDSubArray"]
sowFusedRNNArrayNode[name_String] := Scope[
	array = $arrays[name]; 
	If[Head[array] === TensorT, array = CTable[0., TDimensions @ array]];
	(* ^ for NetPlanPlot / debugging purposes *)
	If[Head[array] === SymbolicRandomArray, array = RandomVariate @@ array];
	(* ^ we are forced to realize random arrays *)
	dims = Dimensions[array];
	$path[[-1]] = name;
	$LogicalWeights[$path] = NDSubArray[$fusedname, $offset, dims];
	$offset += Times @@ dims;
	BagPush[$fusedRNNArrayData, Flatten @ Normal @ array, 1];
	If[$wdropout =!= None, BagPush[$weightDropoutMaskData, CTable[If[StringEndsQ[name, "Biases"], 1., 0.], Times @@ dims], 1]];
];

sowFusedRNNArrayNode[n_Integer] := (
	$offset += n;
	BagPush[$fusedRNNArrayData, CTable[0., n], 1];
	If[$wdropout =!= None, BagPush[$weightDropoutMaskData, CTable[0., n], 1]];
)

(* sowArrayNode writes the arrays into the parameters list so they are easy for the
writer function to access via #ArrayName *)
sowArrayNode[name_, None] := 
	name -> None;

sowArrayNode[name_, NetSharedArray[sname_]] := Scope[
	$path = NetPath["SharedArrays", Null];
	name -> Last @ sowArrayNode[sname, $sharedArrays[sname]]
]

sowArrayNode[name_, arr_] := Scope[
	$path[[-1]] = name;
	If[!$PortFilter[$path], Return[Nothing]];
	If[KeyExistsQ[$PathToNode, $path], 
		(* this caching kicks in in the case of NetOperators and shared arrays
		TODO: support aux nodes as well *)
		Return[name -> $PathToNode[$path]]
	];
	mxname = MXManglePath[$path];
	mxid = SowNullNode[mxname];
	If[MemberQ[auxArrays, name],
		$AuxArrays[mxname] = arr
	,
		$WeightArrays[mxname] = arr;
		$LogicalWeights[$path] = mxname;
	];
	$PathToNode[$path] = mxid;
	name -> mxid
];

arrDimensions[arr_NumericArray] := Dimensions[arr];
arrDimensions[SymbolicRandomArray[_, dims_]] := dims;
arrDimensions[e_] := TDimensions[e, Panic["NoDims", "No dims for array value ``.", e]];


PackageScope["toLayerWriter"]
PackageScope["toValueWriter"]

Clear[toLayerWriter];

toLayerWriter[name_] := Memoized @ makeLayerReaderWriter[name, "Writer"];

toFieldWriter[param_, key_] := With[
	{writer = toValueWriter @ LookupOr[$mxParamTypes, param]}, 
	key -> Quoted[writer[Slot[param]]]
];

IntP = SizeT | PosIntegerT | IntegerT | NaturalT;
toValueWriter[t_] := Match[t,
	ListT[_, IntP] :> writeIntList,
	SizeT|PosIntegerT :> IntegerString,
	IntegerT|NaturalT :> intString,
	ScalarT | _IntervalScalarT :> CDoubleString,
	BooleanT :> $ToBoolean,
	PoolingFunctionT :> $ToPooling,
	Defaulting[type_, _] :> %[type],
	Identity
];

intString[0] = "0";
intString[1] = "1";
intString[2] = "2";
intString[-1] = "-1";
intString[i_Integer] := If[Negative[i], "-" <> IntegerString[i], IntegerString[i]];
intString[None] := "None";
intString[_] := $Unreachable;

PackageScope["writeIntList"]

writeIntList[{0}] := "(0)";
writeIntList[{1}] := "(1)";
writeIntList[{2}] := "(2)";
writeIntList[{0, 0}] := "(0, 0)";
writeIntList[{1, 1}] := "(1, 1)";
writeIntList[{2, 2}] := "(2, 2)";
writeIntList[{a_}] := StringJoin["(", intString[a], ")"];
writeIntList[{a_, b_}] := StringJoin["(", intString[a], ", ", intString[b], ")"];
writeIntList[{a_, b_, c_}] := StringJoin["(", intString[a], ", ", intString[b], ", ",  intString[c], ")"];
writeIntList[list_] := StringRiffle[list, {"(", ", ", ")"}];

PackageScope["makeLayerReaderWriter"]
PackageScope["$mxParamTypes"]
PackageScope["$mxParamDefaults"]

makeLayerReaderWriter[name_, type_] := Scope[
	mxinfo = $LayerData[name, "MXNet"];
	$mxParamTypes = $LayerData[name, "Parameters"];
	$mxParamDefaults = $LayerData[name, "ParameterDefaults"];
	If[mxinfo === None, Return[{}&]];
	readerWriter = Select[FreeQ[$Failed]] @ KeyValueMap[
		If[type === "Writer", toFieldWriter, toFieldReader],
		Lookup[mxinfo, "Parameters", <||>]
	];
	If[mxinfo[type] =!= None,
		AppendTo[readerWriter, Quoted @@ mxinfo[type]]
	];	
	ReplaceAll[Compose[Function, readerWriter], Quoted[h_] :> h]
];

ToFrom[rules___] := {Association[rules], Association[Reverse[{rules}, 2]]};


PackageScope["$ToActivation"]
PackageScope["$FromActivation"]
PackageScope["$ToBoolean"]
PackageScope["$FromBoolean"]
PackageScope["$ToPooling"]
PackageScope["$FromPooling"]

{$ToBoolean, $FromBoolean} = ToFrom[True -> "True", False -> "False"];
$FromBoolean = Join[$FromBoolean, <|"true" -> True, "false" -> False|>];
{$ToPooling, $FromPooling} = ToFrom[Max -> "max", Mean -> "avg", Total -> "sum"];
{$ToActivation, $FromActivation} = ToFrom[
	Ramp -> "relu"
	, Tanh -> "tanh"
	, LogisticSigmoid -> "sigmoid"
	, SoftRamp -> "softrelu"
	, Erf -> "erf"
];



PackageExport["NetPlanInternalSizes"]

NetPlanInternalSizes[NetPlan[plan_]] := Scope @ Catch[
	UnpackAssociation[plan, symbol, inputDims, weightArrays, fixedArrays];
	$MXLibraryErrorHandler = Function[Throw[$Failed, NetPlanInternalSizes]];
	json = MXSymbolToJSON[symbol];
	nodes = json["nodes"];
	json["heads"] = Table[{i-1, 0, 0}, {i, Length[nodes]}];
	inputDims = inputDims /. BatchSize -> 31;
	allArgs = Join[weightArrays, fixedArrays, inputDims];
	symbol2 = MXSymbolFromJSON[json];
	inferred = MXSymbolInferShape[symbol2, Map[toDims, allArgs]];
	inferred["OutputArrays"] /. n_Integer /; Divisible[n, 31] :> RuleCondition[BatchSize * (n / 31)],
	NetPlanInternalSizes
];

toDims[list_List] := list;
toDims[ra_NumericArray] := Dimensions[ra];
toDims[SymbolicRandomArray[_, dims_]] := dims;
toDims[t_] := TDimensions[t];


PackageScope["$PlanLogger"]

$PlanLogger = Hold;

PackageScope["ApplyAsymmetricConvPoolPadding"]

ApplyAsymmetricConvPoolPadding[node_, dimensionality_, interleaving_, paddingSize_] := Scope[
	(* Workarounds for the limitations of PaddingLayer: only accepts 4D or 5D inputs, 
	  and can't pad on the first 2 axes. The workarounds are either:
	  1) Add a dummy dimension after the batch one and remove it later
	  2) Transpose from BSSSC to BCSSS and transpose back later
	  Workaround (1) is preferable for speed and has to be done in the case of one 
	  spatial dimension to make the input 4D, or when there is padding on the second 
	  axis (i.e. on the first spatial dim in the interleaved case). On the other hand,
	  when padding is on the second axis but there are 3 spatial dimensions, reshaping
	  would bring the input to mx.pad to 6D, so in this case we need to move the padded
	  dimension to somewhere else with a transposition. Placing the channel dimension
	  on the second axis is the safe way to do it of course.
	*)
	workaround = Which[
		dimensionality === 3 && interleaving === True && Total[First @ paddingSize] > 0,
			"Transpose", (* Transpose to non-interleaved shape, transpose back at the end *)
		dimensionality === 1 || (interleaving === True && Total[First @ paddingSize] > 0),
			"Reshape", (* Add a dummy dimension and remove it after padding *)
		True,
			None
	];
	Switch[workaround,
		"Transpose", node = SowTranspose[node, {0, 4, 1, 2, 3}],
		"Reshape",   node = SowInsertDim[node, 1]
	];
	mxPadSpec = writeIntList @ Switch[{interleaving, workaround},
		{False, None},
			Join[CTable[0, 4], Flatten @ paddingSize],
		{False, "Reshape"},
			Join[CTable[0, 6], Flatten @ paddingSize],
		{True, None},
			Join[{0, 0}, Flatten @ paddingSize, {0, 0}],
		{True, "Reshape"},
			Join[CTable[0, 4], Flatten @ paddingSize, {0, 0}],
		{True, "Transpose"},
			Join[CTable[0, 4], Flatten @ paddingSize]
	];
	node = SowNode["pad", node, 
		"pad_width" -> mxPadSpec,
		"mode" -> "constant",
		"constant_value" -> "0."
	];
	If[workaround === "Reshape",
		node = SowNode["reshape", node, "shape" -> {0, -3, -2}]
	];
	(* We have taken care of padding, so set it to zero in params
	   to prevent SowCurrentNode from padding again *)
	NeuralNetworks`Private`ToPlan`$parameters[["PaddingSize"]] = CTable[{0, 0}, dimensionality];

	{node, workaround}
]