Input: RealTensorT

Output: RealTensorT

AllowDynamicDimensions: True

Parameters:
	$Writer: ValidatedParameterT[validateWriterFunction]

Writer: Function[
	#Writer[[1,1]][]
]

SubNets: Function[
	NetPath["Parameters", #]& /@ Rest @ Keys[#Parameters]
]

(* ensure symbol exists before we refer to it unqualified below *)
NeuralNetworks`MXLayer;

validateWriterFunction[f_Function] := NormalizePureFunction @ f;
validateWriterFunction[_] := FailValidation[MXLayer, "third arg should be a pure function that takes no arguments."];

MakeLayerBoxes[layer:MXLayer[assoc_Association, meta_]] := Scope[
	UnpackAssociation[assoc, inputs, outputs, parameters, arrays];
	$InsertFinalMXIdentity = False;
	plot = CatchFailure[General, ToBoxes @ NetPlanPlot[layer]];
	If[FailureQ[plot],
		plot = ToBoxes @ Internal`$LastInternalFailure;
	];
	plotsec = {{FrameBox[plot, ImageMargins -> 5, FrameStyle -> None, Background -> White], "\[SpanFromLeft]"}};
	portssec = fmtSection[Join[inputs, outputs], "Ports", False];
	OptimizedArrangeSummaryBox[MXLayer, Nothing, Join[plotsec, portssec], {}]
];

MXLayer[inputs_, outputs_, params_:<||>, func_] ? System`Private`HoldEntryQ := CatchFailureAsMessage[
	makeMXLayer[inputs, outputs, params, func]
];

mx_MXLayer ? System`Private`HoldEntryQ := (
	!ArgumentCountQ[MXLayer, Length[Unevaluated[mx]], 3, 3];
	$Failed
);

SetUsage @ "
MXLayer[idims$, odims$, function$] creates a custom layer that takes a single input of dimensions idims$, \
produces a single output of dimensions odims$, and whose implementation is defined by function$.
MXLayer[<|'name$i'->dims$i,$$|>, <|'name$i'->dims$i,$$|>, function$] allows multiple inputs and outputs.
MXLayer[idims$, odims$, params$, function$] allows learnable arrays and interior layers to be defined.
* The default input and ouput are named 'Input' and 'Output' respectively.
* The function$ should use the functions GetInput and SetOutput, along with SowNode and other Sow functions, \
which are provided in the NeuralNetworks context.
* The dims$ can be lists of dimensions, and can include dynamic dimension specifications.
* If provided, params$ must be a list of associations mapping names to either array shapes, NumericArrays, or layers.
* Layers can be used within the function body using SowSubNet['name', input].
* Arrays can be obtained  within the function body using GetArray['name'].";

$singleLetterP = "Varying" | (_String ? (StringMatchQ[LetterCharacter]));

$typepatt = "Integer" | "Real" | (Restricted["Integer"|Integer, n_Integer] /; Positive[n]);
$varlenpatt = "Varying" | $singleLetterP | _LengthVar;
$tpatt = Alternatives[
	$typepatt,
	$varlenpatt,
	_Integer,
	{RepeatedNull[$varlenpatt, 1], ___Integer, RepeatedNull[$typepatt, 1]},
	(RepeatingElement|SequenceOf)[
		$typepatt | _Integer | {___Integer, RepeatedNull[$typepatt, 1]}
	]
];

procIOSpec[spec:$tpatt, kind_] := Association[kind -> parseIO[kind, spec]];
procIOSpec[spec_Association /; MatchQ[spec, KeyValuePattern[{Repeated[_String -> $tpatt]}]], kind_] := IMap[parseIO, Association @ spec];
procIOSpec[spec_, kind_] := ThrowFailure["inviospec", spec, ToLowerCase @ kind];
MessageName[MXLayer, "inviospec"] = "Invalid specification `` for ``. Specification should be a single type such as \"Real\" or {n1,n2,..}, or an association of port names to types.";

parseIO[name_, spec_] := ParseInputSpec[name, AnyTensorT, spec];

$apatt = {__Integer ? Positive};
procPSpec[key_String, array:$apatt] := $arrays[key] = TensorT[array];
procPSpec[key_String, HoldPattern @ Distributed[dims_List, const_ ? NumericQ]] := $arrays[key] = SymbolicRandomArray[NNConstantDist[const], dims];
procPSpec[key_String, HoldPattern @ Distributed[dims_List, dist_ ? UnivariateDistributionQ]] := $arrays[key] = SymbolicRandomArray[dist, dims];
procPSpec[key_String, na_NumericArray] := $arrays[key] = na;
procPSpec[key_String, net_ ? ValidNetQ] := $layers[key] = NData[net];
procPSpec[key_, val_] := ThrowFailure["invpspec"];
MessageName[MXLayer, "invpspec"] = "Invalid parameter specification (third argument). Specification should be an association of strings to learned array dimensions or interior layers.";
MessageName[MXLayer, "danglinglvars"] = "One or more interior layers operates on an unconstrained dynamic dimension (``). Please use named dynamic dimensions in interior layers that match corresponding dynamic dimensions in the input(s) to the MXLayer (``).";

makeMXLayer[inputs_, outputs_, params_, func_] := Scope[
	$arrays = $layers = <||>; KeyValueScan[procPSpec, params];
	{$layers, sarrays} = HoistSharedArrays[$layers];
	writer = validateWriterFunction[func];
	inputs = procIOSpec[inputs, "Input"];
	outputs = procIOSpec[outputs, "Output"];
	dangling = Complement[lvars1 = UniqueLengthVars[Inputs /@ $layers], lvars2 = UniqueLengthVars[inputs]];
	If[dangling =!= {}, ThrowFailure["danglinglvars", dangling, lvars2]];
	assoc = Association[
		"Type" -> "MX",
		"Inputs" -> inputs, "Outputs" -> outputs,
		"Parameters" -> Prepend[$layers, "Writer" -> ValidatedParameter[writer]],
		"Arrays" -> $arrays,
		If[sarrays =!= <||>, "SharedArrays" -> sarrays, {}]
	];
	System`Private`ConstructNoEntry[MXLayer, assoc, $StandardMetadata]
];

(* there is a special case in NetInformation.m for us *)
toInputForm[assoc_] := 
	HoldForm[MXLayer][
		toSpec @ Inputs @ assoc, 
		toSpec @ Outputs @ assoc, 
		StripVP @ assoc["Parameters", "Writer"]
	];

toSpec[<|"Input"|"Output" -> t_|>] := FromT[t];
toSpec[assoc_] := Map[FromT, assoc];

(* CO-ATTENTION LAYER USED FOR TEST.
	For more details: See "Tests/MXLayer.m" / ask jeromel *)
coattention = Function[
	input = GetInput["Input", "Batchwise"]; 
	query = GetInput["Query", "Batchwise"];
	{ilen, ilenNode} = GetDynamicDimensionInfo[First[GetInputDims["Input"]]];
	{qlen, qlenNode} = GetDynamicDimensionInfo[First[GetInputDims["Query"]]];
	L = SowNode["batch_dot", {input, SowNode["SwapAxis", query, "dim1" -> 1, "dim2" -> 2]}]; 
	L = If[qlenNode =!= None, L = SowSeqMask[SowNode["transpose", L, "axes" -> {2, 0, 1}], qlenNode, "-1e37"];
	If[ilenNode =!= None, SowNode["transpose", SowSeqMask[SowNode["transpose", L, "axes" -> {2, 1, 0}], ilenNode, "-1e37"], "axes" -> {1, 0, 2}], SowNode["transpose", L, "axes" -> {1, 2, 0}]], 
	If[ilenNode =!= None, SowNode["transpose", SowSeqMask[SowNode["transpose", L, "axes" -> {1, 0, 2}], ilenNode, "-1e37"], "axes" -> {1, 0, 2}], L]]; 
	AD = SowReshape[ SowNode["softmax", SowUReshape[L, {-1, qlen}]], {ilen, qlen}]; 
	AQ = SowReshape[ SowNode["softmax", SowUReshape[ SowNode["SwapAxis", L, "dim1" -> "1", "dim2" -> "2"], {-1, ilen}]], {qlen, ilen}]; 
	SetOutput["Output", SowNode["batch_dot", {AD, SowNode["concat", {query, SowNode["batch_dot", {AQ, input}]}, "dim" -> 2, "num_args" -> 2]}]];
];

posencoding[max_] := Function[
	input = GetInput["Input"];
	rangeIndices = SowNode["_arange", {}, "start" -> "1", "infer_range" -> True, "dtype" -> $DTypeMXName];
	(* Trick to resphape the range of indices *)
	zeros = SowNode["_zeros", {}, "shape" -> "(1,)", "dtype" -> $DTypeMXName];
	zeros = SowNode["broadcast_like", {zeros, input}, "lhs_axes" -> "(0,)", "rhs_axes" -> "(1,)"];
	rangeIndices = SowPlus[rangeIndices, zeros];
	(* Clip indices *)
	rangeIndices = SowMinScalar[rangeIndices, max];
	(* Reshape *)
	rangeIndices = SowNode["expand_dims", rangeIndices, "axis" -> 0];
	rangeIndices = SowNode["broadcast_like", {rangeIndices, input}, "lhs_axes" -> "(0,)", "rhs_axes" -> "(0,)"];
	SetOutput["Output", rangeIndices];
];

Tests: {
	{3, 3, SetOutput["Output", SowPlusScalar[GetInput["Input"], 1]] & } -> "3_FbgdxMyuGWA_b8yhu4osrcw=3.904818e+0",
	{<|"Input" -> {"n", 3}, "Query" -> {"m", 3}|>, {"n", 6}, coattention} -> "3*6_PwpKS0zFCik_NVG139vm7WU=7.586764e+0",
	{<|"Input" -> {"n", 3}, "Query" -> {5, 3}|>, {"n", 6}, coattention} -> "3*6_FaqcmbLhp/Q_BGy/Qj7RHiQ=7.890013e+0",
	{<|"Input" -> {5, 3}, "Query" -> {4, 3}|>, {5, 6}, coattention} -> "5*6_GNvmEaCe5m0_Q9VakNqbX24=1.370613e+1",
	{<|"Input" -> {5, 3}, "Query" -> {"n", 3}|>, {5, 6}, coattention} -> "5*6_ddiDtf9XCmU_Z4EmDaW5SMw=1.428408e+1",
	{<|"Input" -> {"n", 10}|>, <|"Output" -> {"n", Restricted["Integer", 3]}|>, posencoding[3]} -> "3_bXKx7ICsvVQ_Wh+ieGZp0oc=6.000000e+0"
}
