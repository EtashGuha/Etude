Input: $$InputTensor

Output: AnyTensorT

Parameters:
	$ForwardNet: NetT[<|"Input" -> $$InputTensor|>, <|"Output" -> $$ForwardTensor|>]
	$BackwardNet: NetT[<|"Input" -> $$InputTensor|>, <|"Output" -> $$BackwardTensor|>]
	$Aggregation: Defaulting[EnumT[{Catenate, Total, Mean}]]
	$$InputTensor: AnyTensorT
	$$ForwardTensor: AnyTensorT
	$$BackwardTensor: AnyTensorT
	$$SequentialOutput: ComputedType[BooleanT,
		sequentialOutputQ[First @ $$InputTensor, First @ $$ForwardTensor],
		{$$InputTensor, $$ForwardTensor}
	]

Upgraders: {
	"12.0.3" ->
		AddParam[Function["$InputTensor" -> #Inputs["Input"]]] /*
		AddParam[Function["$ForwardTensor" -> TensorT[{#Parameters["$Length"], Sequence @@ First[#Parameters["$ForwardShape"]]}, Last[#Parameters["$ForwardShape"]]]]] /*
		AddParam[Function["$BackwardTensor" -> TensorT[{#Parameters["$Length"], Sequence @@ First[#Parameters["$BackwardShape"]]}, Last[#Parameters["$BackwardShape"]]]]] /*
		AddParam[Function["$SequentialOutput" -> True]] /*
		AddParam[Function["$OutputShape" -> 
			biShape[{First @ #Parameters["$InputTensor"], First @ #Parameters["$ForwardTensor"], First @ #Parameters["$BackwardTensor"]}, #Parameters["Aggregation"]]
		]] /*
		DropParam["$Length"] /* DropParam["$ForwardShape"] /* DropParam["$BackwardShape"],
	"12.0.7" -> DropParam["$OutputShape"]
}

ExtraShapeFunctionTensors: {$$ForwardTensor, $$BackwardTensor}

ShapeFunction: Function[List @ biShape[#, $Aggregation]]

RankFunction: Function[List @ biRank[#]]

Suffix: "Operator"

StateExpanding: False

MinArgCount: 1
MaxArgCount: 3

AllowDynamicDimensions: True

SummaryFunction: Function @ Scope[
	fsub = SummaryForm[#ForwardNet];
	bsub = SummaryForm[#BackwardNet];
	HoldForm[NetBidirectionalOperator][{fsub, bsub}]
]

biRank[{i_, f_, b_}] :=
	TrySet[f, b, ShapeException["ranks of forward and backward net should be equal"]];

sequentialOutputQ[inshape_, outshape_] := Which[
	MatchQ[First[outshape], LengthVar[_]], True,
	Length[outshape] < 2, False,
	True, Length[outshape] >= Length[inshape] (* Default guess: is a sequence model if the output tensor has a higher rank than the input *)
];

biShape[{inshape_, {f0_, frest0___}, {b0_, brest0___}} , aggreg_] := Scope[
	oseq = sequentialOutputQ[inshape, {f0, frest0}];
	f1 = If[oseq, First@{frest0}, f0];
	b1 = If[oseq, First@{brest0}, b0];
	frest = If[oseq, Rest@{frest0}, {frest0}];
	brest = If[oseq, Rest@{brest0}, {brest0}];
	rest = MapThread[TrySet[#1, #2, fbinv]&, {frest, brest}];
	o1 = If[aggreg === Catenate, f1 + b1, TrySet[f1, b1, fbinv]];
	Join[{If[oseq, f0, Nothing], o1}, rest]
];

fbinv := ShapeException["shapes of forward and backward net are incompatible"];

Writer: Function @ Scope[
	input1 = GetInputMetaNode["Input"];
	input2 = SowMetaReverse[input1];
	res1 = SowSubNet["ForwardNet", input1];
	res2 = SowSubNet["BackwardNet", input2];
	If[#$SequentialOutput,
		out1 = FromMetaNode @ res1["Output"]; 
		out2 = FromMetaNode @ SowMetaReverse @ ToMetaNode[res2["Output"], First[First[#$ForwardTensor]]];
	,
		out1 = res1["Output"]; 
		out2 = res2["Output"];
	];
	out = Switch[#Aggregation,
		Catenate, SowJoin[out1, out2, If[#$SequentialOutput, 2, 1]],
		Total, SowPlus[out1, out2],
		Mean,  SowDivideScalar[SowPlus[out1, out2], 2]
	];
	SetOutput["Output", out];
]

ArgumentRewriter: rewriteArgs

 NetBidirectionalOperator::invarg1 = "First argument should be a valid net or a pair of valid nets.";
 NetBidirectionalOperator::invarg2 = "Second argument should be one of Catenate, Total, or Mean.";
checkNet[net_] := If[!ValidNetQ[net], ThrowFailure["invarg1"], net];
checkAgg[e_] := If[!MemberQ[{Catenate, Total, Mean}, e], ThrowFailure["invarg2"], e];

rewriteArgs[{net:Except[_Rule|_List], agg:Except[_Rule], opts___Rule}] := {checkNet @ net, net, checkAgg @ agg, opts};
rewriteArgs[{net:Except[_Rule|_List], opts___Rule}] := rewriteArgs[{checkNet @ net, Catenate, opts}];
rewriteArgs[{{fnet_, bnet_}, agg:Except[_Rule], opts___Rule}] := {checkNet @ fnet, checkNet @ bnet, checkAgg @ agg, opts};
rewriteArgs[{{fnet_, bnet_}, opts___Rule}] := rewriteArgs[{{fnet, bnet}, Catenate, opts}];

rewriteArgs[e_] := e;

NeuralNetworks`TestRecurrentLayer;

Tests: {
	{Hold @ TestRecurrentLayer["Input" -> {4, 2}]} -> "4*4_KcpLnt2ijuc_JwPRqC+40UQ=1.554509e+1",
	{Hold @ TestRecurrentLayer["Input" -> {"a", 2}]} -> "3*4_DNK1+VnXsSA_UcJxRqC1NeE=7.644569e+0",
	{Hold @ TestRecurrentLayer["Input" -> {4, 2}], Total} -> "4*2_ZAeWKk1QNRI_RqBJvPsfLLM=1.554509e+1",
	{Hold @ TestRecurrentLayer["Input" -> {4, 2}], Mean} -> "4*2_WgRbIFPjSHw_I8YYIA8uDP8=7.772543e+0",
	{Hold @ {BasicRecurrentLayer[1], BasicRecurrentLayer[3]}, Catenate, "Input" -> {"a", 3}} -> "3*4_FkWVNiu+FIE_O/i9giqUoFs=6.099570e+0",
	{Hold @ SequenceLastLayer[], Catenate, "Input" -> {"Varying", 2}} -> "4_U2KvkdQ7j18_DAqjMGKZJv8=1.029660e+0",
	{Hold @ SequenceLastLayer[], Catenate, "Input" -> {3, 2}} -> "4_U2KvkdQ7j18_WzLbTryXbrw=1.029660e+0",
	{Hold @ SequenceLastLayer[], Mean, "Input" -> {"Varying", 2}} -> "2_XUynSnQWEk4_VGaim3LeaiE=5.148299e-1",
	{Hold @ SequenceLastLayer[], Mean, "Input" -> {3, 2}} -> "2_XUynSnQWEk4_dt6+ogCxpCI=5.148299e-1",
	{Hold @ SequenceLastLayer[], "Input" -> {4, 3, 2}} -> "6*2_ZC4QkRbRbvc_QR2ykouB+QY=5.207198e+0",
	{Hold @ {NetChain[{LongShortTermMemoryLayer[1], SequenceLastLayer[]}], NetChain[{LongShortTermMemoryLayer[2], SequenceLastLayer[]}]}, "Input" -> {"Varying", 2}} -> "3_E6j8Dtf4ySk_WERSwhTrRyk=3.239545e-1",
	{Hold @ {NetChain[{LongShortTermMemoryLayer[2], SequenceLastLayer[]}], NetChain[{LongShortTermMemoryLayer[2], SequenceLastLayer[]}]}, Mean, "Input" -> {4, 2}} -> "2_Rth4oV1unhI_et9S4/rim+Q=4.093368e-1",
	{Hold @ {NetChain[{LongShortTermMemoryLayer[1], ConvolutionLayer[2, 2, 2, "Interleaving" -> True]}], NetChain[{LongShortTermMemoryLayer[1], ConvolutionLayer[3, 2, 2, "Interleaving" -> True]}]}, "Input" -> {4, 2}} -> "2*5_BC/l3NHwtUA_SfzrP0ZMf1E=1.279112e+1",
	{Hold @ {BasicRecurrentLayer[1], BasicRecurrentLayer[3]}, Total, "Input" -> {"a", 3}} -> "Type inconsistency in NetBidirectionalOperator: shapes of forward and backward net are incompatible.",
	(* bug 364718 *)
	{Hold @ NetBidirectionalOperator[NetGraph[{GatedRecurrentLayer[2,"Input"->{"Varying",3}], ConstantArrayLayer[]},{2->NetPort[1,"State"]}],Total], Total, "Input" -> {"Varying", 3}} -> "3*2_GcKc+/b3xY0_Uo+QnlLByVI=1.301498e+1",
	(* bug 370024 *)
	{{BasicRecurrentLayer[2, "Input" -> {"Varying", 2}], BasicRecurrentLayer[4, "Input" -> {"Varying", 2}]}, Total} -> "Type inconsistency in NetBidirectionalOperator: shapes of forward and backward net are incompatible."
}

