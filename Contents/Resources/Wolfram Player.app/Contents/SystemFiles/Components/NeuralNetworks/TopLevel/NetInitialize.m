Package["NeuralNetworks`"]


(******************************************************************************)
PackageExport["NetDeinitialize"]

SetUsage @ "
NetDeinitialize[net$] removes all arrays from a net."

NetDeinitialize[NetP[net, meta]] := ConstructNet[RemoveArrays[net], meta];

(******************************************************************************)
(* NetInitializer: a graph based parameter initializer *)

PackageExport["NetInitialize"]
PackageExport["SymbolicRandomArray"]
PackageScope["NNConstantDist"]

SymbolicRandomArray /: Dimensions[SymbolicRandomArray[_, dims_]] := dims

Options[NetInitialize] = {
	Method -> Automatic,
	RandomSeeding -> Inherited,
	"SamplingFunction" -> RandomVariate
};

$NetInitializeMethods := $NetInitializeMethods = <|
	"Xavier" -> {Values /* Apply[XavierInitializer], <|
		"FactorType" -> {EnumT[{"In", "Out", "Mean"}], "Mean"},
		"Distribution" -> {EnumT[{"Uniform", "Normal"}], "Uniform"}
	|>},
	"Orthogonal" -> {OrthogonalInitializer&, <||>},
	"Identity" -> {
		IdentityInitializer[#Distribution]&, 
		<|"Distribution" -> {DistributionT, NormalDistribution[0,0.01]}|>
	},
	"Random" -> {DistributionInitializer[#Weights, #Biases, #Scaling]&, <|
		"Weights" -> {DistributionT, NormalDistribution[]},
		"Biases" -> {DistributionT, None},
		"Scaling" -> {DistributionT, None}
	|>}
|>;

NetInitialize::notfspec = "All parameters must be fully specified before a network can be initialized."
NetInitialize::notudist = "`` should be a valid univariate distribution."

NetInitialize[list_List, opts:OptionsPattern[]] :=
	Map[NetInitialize[#, opts]&, list];

NetInitialize[net_, None] := NetDeinitialize[net];

NetInitialize::arg2 = "Second argument should be either All or Automatic.";

(* TODO: factor ParseMethod with this handler out into its own function *)
paramHandler[key_, user_, type_, default_] := Scope[
	res = CatchFailure @ Coerce[user, type];
	If[FailureQ[res], Compose[$Failed, TypeString[type]], res]
];

NetInitialize[expr:(head_Symbol[assoc_Association, meta_]), replaceExisting:Except[_Rule]:Automatic, OptionsPattern[]] := CatchFailureAsMessage @ Scope[
	$replace = Switch[replaceExisting, 
		All, True,
		Automatic, False,
		_, ReturnFailed["arg2"]
	];
	randomSeeding = OptionValue[RandomSeeding];
	If[!MatchQ[randomSeeding, None | Automatic | Inherited | _Integer | _String],
		NetInitialize::seeding = "Value of option RandomSeeding -> `1` is not Automatic, Inherited, an integer, or a string. Inherited will be used instead.";
		Message[NetInitialize::seeding, randomSeeding];
		randomSeeding = Inherited;
	];	
	If[FullySpecifiedNetQ[expr] && !$replace && FreeQ[assoc, SymbolicRandomArray], Return[expr]];
	NetFinalCheck[assoc];
	$makeArray = makeArray;
	Match[OptionValue[Method],
		{"Custom", method_} :> (
			$makeArray = makeCustomArray;
			$customFunction = method;
		),
		method_ :> (
			$initializer = ParseMethod[method, $NetInitializeMethods, paramHandler];
		)
	];
	sfunc = Replace[OptionValue["SamplingFunction"], "Symbolic" -> SymbolicRandomArray];
	$eagerSample = sfunc =!= SymbolicRandomArray;
	$samplef = sfunc /* toRA;
	$assoc = assoc;
	$initialLayers = None;
	$graph = LayerDependencyGraph[assoc]; 
	$isVert = ConstantAssociation[VertexList[$graph], True];
	$wasinit = <||>;
	withSeeding[randomSeeding,
		Scan[updateVertex, TopologicalSort[$graph]];
		initRemainingArrays[assoc];
	];
	(* ^ this exists because operators can in theory have Arrays in them, e.g. MXLayer which counts as an operator,
	and these will not show up in the layer dependency graph *)
	System`Private`ConstructNoEntry[head, $assoc, meta]
]

(* None is to opt-in to the old behavior, because using RandomSeeding -> Inherited is
actually different from doing nothing! *)
SetHoldRest[withSeeding];
withSeeding[None, body_] := body;
withSeeding[seed_, body_] := BlockRandom[body, RandomSeeding -> seed];

NetInitialize::arg1 = "First argument to NetInitialize should be a net or list of nets."

NetInitialize[__] := RuleCondition[Message[NetInitialize::arg1]; Fail];

getVertexDepth[port_] := (
	If[$initialLayers === None, 
		$initialLayers = Complement[VertexList[$graph], EdgeList[$graph][[All, 1]]];
	];
	Min[GraphDistance[$graph, port, #]& /@ $initialLayers]
);

updateVertex[port:NetPath[pos___]] := Scope[
	If[KeyExistsQ[$wasinit, port], Return[]];
	$wasinit[port] = True;
	subnet = Part[$assoc, pos];
	arrays = subnet["Arrays"];
	arrays = DeleteCases[None] @ arrays;
	$type = subnet["Type"];
	If[!$replace, arrays = DeleteCases[_NumericArray] @ arrays];
	$port = port;
	$depth := $depth = getVertexDepth[port];
	KeyValueScan[
		updateArray[$assoc[[pos, "Arrays", #1]], #1, #2]&,
		arrays
	];
]

DeclareMethod[initRemainingArrays, Null, initContainerArrays, initOperatorArrays];

initContainerArrays[assoc_] := (
	If[Lookup[assoc, "Arrays", {}] =!= {}, updateVertex[$path]];
	ScanNodes[initRemainingArrays, assoc];
);

initOperatorArrays[assoc_] := (
	If[Lookup[assoc, "Arrays", {}] =!= {}, updateVertex[$path]];
	ScanSubNets[initRemainingArrays, assoc];
);

Clear[updateArray];

SetHoldFirst[updateArray];

updateArray[lvalue_, name_, NetSharedArray[sname_String]] := 
	updateArray[$assoc["SharedArrays", sname], name, $assoc["SharedArrays", sname]];

updateArray[lvalue_, name_, _NumericArray] /; !$replace = Null;

updateArray[lvalue_, name_, array_SymbolicRandomArray] := 
	If[$eagerSample, Set[lvalue, toRA[RandomVariate @@ array]]];

updateArray[lvalue_, name_, array_] := Set[lvalue, $makeArray[$type, name, checkBigDims @ getDims[array]]];


getDims[Nullable[t_]] := getDims[t];
getDims[arr_NumericArray] := Dimensions[arr];
getDims[SymbolicRandomArray[_, dims_]] := dims;
getDims[t_] := Replace[TDimensions[t], SizeT | $Failed | _LengthVar :> ThrowNotSpecifiedFailure[$assoc, "initialize", "Defined"], {0, Infinity}];

PackageExport["$NetInitializeMaxTensorByteCount"]
$NetInitializeMaxTensorByteCount := $RealisticSystemMemory / 10;

General::tnsrmaxsz = "Cannot create an array of dimensions `` because it would consume ``, which is ``% of total system memory."
checkBigDims[dims_] := Scope[
	sz = 4 * Apply[Times, dims];
	If[sz > $NetInitializeMaxTensorByteCount,
		ThrowFailure["tnsrmaxsz", Row[dims, "\[Times]"], 
			PositiveSIString[sz] <> "bytes", 
			Round[(sz / $RealisticSystemMemory) * 100]
		]];
	dims
];

(* this is used as needed by makeArray. *)
LookupForwardLayer[port_, n_Integer] := Scope[
	If[n < 1, Panic[]];
	If[!TrueQ[$isVert[port]], Return[None]];
	out = Complement[VertexOutComponent[$graph, port, n], VertexOutComponent[$graph, port, n - 1]];
	If[Length[out] === 0, None, Part[$assoc, Sequence @@ Last[out]]]
];

(******************************************************************************)
(* This contains the special Port/Layer level init logic. If adding a new layer
	with special init needs, add it here.
*)

spatialTransformerQ[type_] := 
(type === "Linear") && (LookupForwardLayer[$port, 1]["Type"]  === "SpatialTransformation") ||
(type === "Convolution") && (LookupForwardLayer[$port, 2]["Type"] === "SpatialTransformation")


(***************************************)

seluQ["Linear"] := Scope[
	forward = LookupForwardLayer[$port, 1];
	If[forward["Type"]  =!= "Elementwise", Return[False]];
	f = First[forward["Parameters", "Function"]];
	If[Head[f] =!= ScalarFunctionObject, Return[False]];
	ssa = f["SSA"];
	If[Length[Keys[ssa]] =!= 1, Return[False]];
	val = First[ssa];
	If[Not@ListQ[val] || (Length[val] === 0), Return[False]];
	If[First[val] === "ScaledExponentialLinearUnit",
		Return[True],
		Return[False]
	];
]

seluQ[_] := False

(***************************************)

NetInitialize::badcustominit = "Custom initializer did not return a number, tensor, or distribution."

toRA[list_List] := toNumericArray[list];
toRA[ra_NumericArray] := ra;
toRA[sa_SymbolicRandomArray] := sa;
toRA[_] := $Unreachable;

NNConstantDist /: RandomVariate[NNConstantDist[val_], dims_] := CTable[N @ val, dims];
constArr[val_, dims_] := $samplef[NNConstantDist[val], dims];

makeCustomArray[type_, name_, dim_List] := UseMacros @ 
	Match[
		$customFunction[<|
			"Type" -> $TypeToSymbol[type], "Name" -> name, "Dimensions" -> dim, 
			"Outputs" -> First[dim], "Depth" -> $depth, 
			"NextLayer" -> LookupForwardLayer[$port, 1]|>],
		0.|0 :> constArr[0, dim],
		r_ ? NumberQ :> $samplef[NormalDistribution[0, N[r]], dim],
		e_List ? MachineArrayQ :> If[arrayDimensions[e] =!= dim, Panic[], toRA @ e],
		e_ ? UnivariateDistributionQ :> $samplef[e, dim],
		ThrowFailure["badcustominit"]
	];

(* Weights: Weight ports have special methods that can depend on activation functions *)
makeArray[type_, "Weights", dim_List] := 
	checkInitRes @ Which[
		spatialTransformerQ[type],
			constArr[0, dim],
		seluQ[type],
			(* stddev = sqrt(1 / fan_in), fan_in = Last[dim] *)
			$samplef[NormalDistribution[0, Sqrt[1 / Last[dim]]], dim],
		True,	(* defualt *)
			$initializer[type, dim, LookupForwardLayer[$port, 1]]
	]

makeArray[_, "Biases", dim_List] /; setsBiasesQ[$initializer] := 
	checkInitRes @ $initializer[type -> "Biases", dim, LookupForwardLayer[$port, 1]]

makeArray[_, "Scaling", dim_List] /; setsScalingQ[$initializer] :=
	checkInitRes @ $initializer[type -> "Scaling", dim, LookupForwardLayer[$port, 1]]

checkInitRes[result_] := 
	If[validInitArrayQ[result], 
		result,
		Panic["NotNumericArray", "Initializer failed to yield a NumericArray, and instead returned ``.", result]
	];

validInitArrayQ[_NumericArray] := True;
validInitArrayQ[_SymbolicRandomArray] := True;
validInitArrayQ[_] := False;

(* Biases: All arrays named Biases are zero initialized *)

makeArray[type_, "ForgetGateBiases", dims_List] :=
	constArr[1, dims];

makeArray[type_, "Biases", dim_List] := Which[
	spatialTransformerQ[type] && (First@dim === 6),
		toRA @ {1, 0, 0, 0, 1, 0},	
	True,	
		constArr[0, dim]
]

(* Scale: All arrays named Scale are one initialized *)
makeArray[_, "Scaling", dim_List] :=
	constArr[1, dim]

(* ParametricRampLayer slope *)
makeArray[_, "Slope", dim_List] :=
	constArr[0.1, dim]

(* all other-named arrays are treated as weights if matrices otherwise zero-initialized *)
makeArray[type_, other_, dim_List] := 
	makeArray[type, If[Length[dim] == 2, "Weights", "Biases"], dim];

setsBiasesQ[_] := False;

setsScalingQ[_] := False;

(* BatchNorm Arrays: special init *)
makeArray["BatchNormalization", "MovingMean", dim_List] :=
	constArr[0, dim]

makeArray["BatchNormalization", "MovingVariance", dim_List] :=
	constArr[1, dim]

(******************************************************************************)
(* Weight Initialization Methods: Xavier, Orthogonal, Distribution, and Identity. *)

setsScalingQ[DistributionInitializer[_, _, sdist_]] := sdist =!= None;

setsBiasesQ[DistributionInitializer[_, bdist_, _]] := bdist =!= None;

DistributionInitializer[wdist_, bdist_, sdist_][_, dim_, nextLayer_:None] :=
	$samplef[wdist, dim];

DistributionInitializer[wdist_, bdist_, sdist_][_ -> "Scaling", dim_, nextLayer_:None] :=
	$samplef[sdist, dim];

DistributionInitializer[wdist_, bdist_, sdist_][_ -> "Biases", dim_, nextLayer_:None] :=
	$samplef[bdist, dim];

IdentityInitializer[dist_][type_, dim_List, nextLayer_:None] :=
	toRA @ AddNoise[IdentityTensor @@ dim, dist];

IdentityInitializer[dist_]["Convolution"|"Deconvolution", dims:{nout_, nin_, w_, h_}, nextLayer_:None] := 
	toRA @ AddNoise[IdentityConvolutionKernel @@ dims, dist];

AddNoise[arr_, dist_] := arr + RandomVariate[dist, arrayDimensions[arr]];

IdentityConvolutionKernel[nout_, nin_, w_, h_] := Scope[
	unitKernel = ToPackedArray @ System`CenterArray[1, {w, h}];
	zeroKernel = CTable[0, {w, h}];
	Table[
		If[out == in, unitKernel, zeroKernel],
		{out, nout},
		{in, nin}
	]
];

IdentityTensor[a_, b_] := Take[IdentityMatrix @ Max[a, b], a, b];
IdentityTensor[m_, rest__] := Table[IdentityTensor[rest], m];


(***********
	xavierInitializer: based on the following papers
	1. Understanding the difficulty of training deep feedforward neural networks,
		X. Glorot and Y. Bengio
	2. Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification,
		K. He et al
***********)

XavierInitializer[factorType_, distribution_][type_, dim_List, nextLayer_:None] := Scope[
	(* 1. Get number of input + output neurons *)
	fanin = Times @@ Rest[dim];
	fanout = First[dim];
	
	(* 2. The scale depends on the activation function. See He et al *)
	variance = Switch[factorType, 
		"In", 		2 / fanin,
		"Out", 		2 / fanout,
		"Mean", 	2 / (fanin + fanout)
	];
	scaleFactor = activationScaleFactor[nextLayer];
	variance *= scaleFactor;
	(* 3. Sample from distribution of given variance *)
	stddev = Sqrt[variance];
	values = Switch[distribution,
		"Normal",
			$samplef[NormalDistribution[0, stddev], dim],
		"Uniform",
			(* using StandardDeviation@UniformDistribution[{-n, n}] = n/Sqrt[3] *)
			$samplef[UniformDistribution[{-1,1} * stddev * Sqrt[3.]], dim]
	]
]

(***********
 orthogonalInitializer: based on:
	Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
		http://arxiv.org/pdf/1312.6120v3.pdf
		A.M. Saxe et al 2014
	NOTE: we follow Lasagne implementation (https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py),
		and 
***********)

OrthogonalInitializer[type_, dim_List, nextLayer_:None] := Scope[
	(* 1. Get number of input + output neurons *)
	fanin = Times @@ Rest[dim];
	fanout = First[dim];
	flatShape = {fanout, fanin};
	
	scaleFactor = activationScaleFactor[nextLayer];
	a = RandomVariate[NormalDistribution[0, 1], flatShape];
	{u, w, v} = SingularValueDecomposition[a, Min@Dimensions@a];

	(* Choose one with correct shape *)
	q = If[Dimensions[u] === flatShape, u, v];
	q = ArrayReshape[q, dim] * Sqrt[scaleFactor];

	toRA @ q
]

(***********
	 Scale factors:
		Decide on the scale depending on type of rectifier. 
		Used by orthogonalInitializer + xavierInitializer
		see http://arxiv.org/pdf/1505.00853.pdf for overview of rectifiers
***********)

(* should factor this out at some point.. *)
(* jeromel: What about "ELU", "SELU", ...? *)
reluQ[ValidatedParameter[func_]] := Scope[
	(* check for ramp. Simplest case *)
	If[func === Ramp, Return[True]];
	(* now check for string arg form of ramp *)
	If[Head[func] =!= ScalarFunctionObject, Return[False]];
	ssa = func["SSA"];

	(* for compound elementwise layers, return false *)
	If[Length[Keys[ssa]] =!= 1, Return[False]];
	val = First[ssa];
	If[Not[ListQ @ val] || (Length[val] === 0), Return[False]];
	If[First[val] === "RectifiedLinearUnit",
		Return[True],
		Return[False]
	];
]

reluQ[___] := Panic["reluQ assumes input is ValidatedParameterT[...]. Invalid input."]

(* high level scale *)
activationScaleFactor[None] := 1
activationScaleFactor[assoc_Association] := 
	iActivationScaleFactor[assoc["Type"], assoc["Parameters"]];

(* Case 1: Relu/Ramp *)
iActivationScaleFactor["Elementwise", param_] := If[reluQ @ param["Function"], 2, 1];

(* Case Default: Relu/Ramp *)
iActivationScaleFactor[other_, param_] := 1;

