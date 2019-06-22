Package["NeuralNetworks`"]


(* This is where we put higher-level operations that are polymorphic over
node manifestation, e.g. that want to operate on MetaNodes *)

PackageScope["MetaNode"]

SetUsage @ "
MetaNode[batchwise$,timewise$,unpacked$,lnode$,maxlen$] represents a logical node in a MX graph, \
and can contain multiple physical manifestations. 
| batchwise$ | MXNode[$$] | physical and logical interpretation are the same |
| timewise$ | MXNode[$$] | physical and logical are 0-1 transposed |
| unpacked$ | {node$1, node$2, $$} | first (non-batch) logical dimension has been unpacked |
| lnode$ | MXNode[$$] | vector of dynamic dim lengths |
| maxlen$ | integer$ | max dynamic dim length |
* The first 3 arguments are closure variables and so can be created and cached when a certain manifestation is demanded from layer code. 
* The maxlen$ and lnode$ arguments will be None if no dynamic dim is present."

PackageScope["VarMetaQ"]
PackageScope["FixedMetaQ"]

VarMetaQ[MetaNode[_, _, _, lnode_, _]] := (lnode =!= None);
VarMetaQ[_] := $Unreachable;

FixedMetaQ[MetaNode[_, _, _, lnode_, _]] := (lnode === None);
FixedMetaQ[_] := $Unreachable;


MetaNode /: MakeBoxes[MetaNode[batchwise_, timewise_, unpacked_, lnode_, maxlen_], StandardForm] := 
	ToBoxes[HoldForm[MetaNode][batchwise, timewise, unpacked, lnode, maxlen]];

SetAttributes[MetaNode, HoldAll];

mn_MetaNode["Batchwise"] := getBatchwise[mn];
getBatchwise[MetaNode[batchwise_, timewise_, unpacked_, lnode_, maxlen_]] :=
	SetNone[batchwise, 
		If[timewise =!= None, SowTranspose01[timewise],
			SowPack[unpacked, False]]];

mn_MetaNode["Timewise"] := getTimewise[mn];
getTimewise[MetaNode[batchwise_, timewise_, unpacked_, lnode_, maxlen_]] :=
	SetNone[timewise, 
		If[batchwise =!= None, SowTranspose01[batchwise],
			SowPack[unpacked, True]]];

mn_MetaNode["Packed"] := getPacked[mn];
getPacked[MetaNode[batchwise_, timewise_, unpacked_, lnode_, maxlen_]] :=
	If[batchwise =!= None, 
		{batchwise, False},
		{SetNone[timewise, SowPack[unpacked, True]], True}
	];

mn_MetaNode["Unpacked"] := getUnpacked[mn];
getUnpacked[MetaNode[batchwise_, timewise_, unpacked_, lnode_, maxlen_]] := 
	SetNone[unpacked, 
		If[timewise =!= None,
			SowUnpack[timewise, maxlen, 0],
			SowUnpack[batchwise, maxlen, 1]
		]
	]

mn_MetaNode["LengthNode"] := Extract[mn, 4];
mn_MetaNode["MaxLength"] := Extract[mn, 5];

_MXNode[_] := $Unreachable;


PackageScope["ToMetaNode"]
PackageScope["FromMetaNode"]

(* TODO: Document ToMetaNode *)
ToMetaNode[mn_MetaNode, __] := mn; 
ToMetaNode[node_MXNode, lenref_, isTransposed_:False] := ModuleScope[
	{maxlen, lnode} = If[Head[lenref] === MetaNode,
		{lenref["MaxLength"], lenref["LengthNode"]},
		Block[{$ReshapingIsImpossible}, GetDynamicDimensionInfo @ lenref]
	];
	If[isTransposed, 
		timewise = node; batchwise = unpacked = None,
		batchwise = node; timewise = unpacked = None; 
	];
	MetaNode[batchwise, timewise, unpacked, Evaluate @ lnode, Evaluate @ maxlen]
]

FromMetaNode[node_MetaNode] := node["Batchwise"];
FromMetaNode[node_MXNode] := node;


PackageScope["SowMetaMap"]

SowMetaMap[f_, mn:MetaNode[batchwise_, timewise_, unpacked_, lnode_, maxlen_]] := ModuleScope[
	batchwise2 = timewise2 = unpacked2 = None;
	If[batchwise === None, mn["Timewise"]];
	Which[
		batchwise =!= None,	batchwise2 = SowFlatMap[f, batchwise],
		timewise =!= None,	timewise2 = SowFlatMap[f, timewise],
		unpacked =!= None,	unpacked2 = Map[f, unpacked]
	];
	MetaNode[batchwise2, timewise2, unpacked2, lnode, maxlen]
];

_SowMetaMap := $Unreachable;


PackageScope["SowFlatMap"]

SetUsage @ "
SowFlatMap[f$, node$] maps f$ over node$ by flattening the batch and first dimensions \
together and applying f$, then unflattening."

SowFlatMap[f_, mxnode_] := SowUnflatten[
	checkmxn @ f @ SowFlatten[mxnode],
	mxnode
];

_SowFlatMap := $Unreachable;

SetHoldAll[checkmxn];
checkmxn[body_] := 
	Replace[body, {
		mn_MetaNode :> mn["Batchwise"], 
		e:Except[_MXNode] :> Panic["NotMXNode", "Got `` instead of an MXNode from ``.", e, HoldForm[body]]
	}];



(* UNUSED

PackageScope["SowMetaMean"]

SowMetaMean[mn:MetaNode[batchwise_, timewise_, unpacked_, None, maxlen_, dims_]] := 
	SowNode["_DivScalar",
		Which[
			batchwise =!= None, SowSumAxis[batchwise, 1],
			timewise =!= None,  SowSumAxis[timewise, 0],
			True, SowNode["ElementWiseSum", unpacked]
		],
		"scalar" -> maxlen
	];

SowMetaMean[mn_MetaNode] := 
	SowSumAxis[SowSeqMask[mn["Timewise"], mn["LengthNode"]], 0];
	(* ^ needs divide by seq len *)

SowMetaMean[_] := $Unreachable;

*)

PackageScope["SowMetaLast"]

SowMetaLast[mn:MetaNode[batchwise_, timewise_, unpacked_, None, maxlen_]] := Which[
	unpacked =!= None, Last[unpacked],
	timewise =!= None, SowNode["SequenceLast", timewise],
	batchwise =!= None, SowNode["SequenceLast", batchwise, "axis" -> "1"]
];

SowMetaLast[mn_MetaNode] := 
	SowNode["SequenceLast", {mn["Timewise"], mn["LengthNode"]}, "use_sequence_length" -> "true"];


PackageScope["SowMetaReverse"]

(* TODO: can we use backwardsliceaxis to do a reverse on batchwise? *)

SowMetaReverse[mn:MetaNode[batchwise_, timewise_, unpacked_, lnode_, maxlen_]] := ModuleScope[
	batchwise2 = timewise2 = unpacked2 = None;
	If[lnode === None,
		Which[
			unpacked =!= None, unpacked2 = Reverse[unpacked],
			timewise =!= None, timewise2 = sowReverse[timewise],
			batchwise =!= None, timewise2 = sowReverse[timewise = SowTranspose01[batchwise]]
		],
		timewise2 = sowReverse[mn["Timewise"], lnode];
	];
	MetaNode[batchwise2, timewise2, unpacked2, lnode, maxlen]
];

sowReverse[in_] := SowNode["SequenceReverse", in, "use_sequence_length" -> "false"];
sowReverse[in_, len_] := SowNode["SequenceReverse", {in, len}, "use_sequence_length" -> "true"];


PackageScope["SowMetaDrop"]

SowMetaDrop[MetaNode[batchwise_, timewise_, unpacked_, lnode_, maxlen_], nout_Integer, leftq_] :=
	Which[
		maxlen == 1,        ThrowFailure["netseqlen"], 
		timewise =!= None,  SowTake[timewise,  If[leftq, 1, 0] + {0, nout}, 0],
		batchwise =!= None, SowTake[batchwise, If[leftq, 1, 0] + {0, nout}, 1],
		unpacked =!= None,  newUnpackedNode[If[leftq, Rest, Most] @ unpacked, None, nout]
	]

newUnpackedNode[unpacked_, lnode_, maxlen_] := ModuleScope[
	batchwise = timewise = None; 
	MetaNode[batchwise, timewise, unpacked, lnode, maxlen]
];

SowMetaDrop[mn:MetaNode[batchwise_, timewise_, unpacked_, lnode_, maxlen_], nout_LengthVar, leftq_] := ModuleScope[
	lnode2 = SowDerivedSequenceLengthNode[lnode, nout, # - 1&];
	maxlen2 = maxlen-1;
	batchwise2 = timewise2 = unpacked2 = None;
	spec = If[leftq, {1, "None"}, {0, -1}];
	Which[
		maxlen == 1,        ThrowFailure["netseqlen"], 
		batchwise =!= None,	batchwise2 = SowTake[batchwise, spec, 1],
		timewise =!= None,	timewise2 = SowTake[timewise, spec, 0],
		unpacked =!= None,	timewise2 = SowPack[If[leftq, Rest, Most] @ unpacked, True]
	];
	With[{lnode2 = lnode2, maxlen2 = maxlen2},
		MetaNode[batchwise2, timewise2, unpacked2, lnode2, maxlen2]
	]
];

SowMetaDrop[_, _, _, _] := $Unreachable;


PackageScope["MakeVariableLengthLossData"]

(* this assumes you'll do an op that produces a batch of scalars. 

if there is a dynamic dimension, we'll zero out those means ourselves 
in the batch-time-folded tensor, then sum away the time dimension.

in the end we return a batch worth of scalars *)

MakeVariableLengthLossData[inode_MetaNode ? FixedMetaQ, tnode_MetaNode] := Scope[
	{inode["Batchwise"], tnode["Batchwise"], Identity, Identity, Identity}
];
(* ^ this is a no-op, because the actual loss operation does all the summing
already *)

MakeVariableLengthLossData[inode_MetaNode, tnode_MetaNode] := ModuleScope[
	lnode = inode["LengthNode"];
	inode = SowFlatten[orig = inode["Timewise"]]; (* TODO: automatically pick timewise vs batchwise - see if both have timewise, otherwise use batchwise? *)
	tnode = SowFlatten[tnode["Timewise"]];
	{
		inode,
		tnode, 
		SowTimewiseMean[SowUnflatten[#, orig], lnode]&, (* postfMean *)
		SowTimewiseSum[SowUnflatten[#, orig], lnode]&, (* postfSum *)
		SowSwapAxis[SowSeqMask[SowUnflatten[#, orig], lnode, -1], 0, 1]& (* postfNoAgg *)
	}
];
(* ^ The postfXXX functions are post processing functions that are used in the CE, MSE, & MAE losses
Their purpose is to ensure that metric & loss calculations are handled correctly with input sizes of varying dimmensions.
MakeVariableLengthLossData will first flatten the variable length dimmension into the batch dim so that the rest of the logic can ignore it.
Then the postf functions reshape the result to restore the variable legth dimmensions & aggreagate the result appropriately.
SowTimewiseXXX functions perform the aggregation by using masking to ignore irrelevant parts of the variable length dims in a batch. 
*)


PackageScope["MeanLossImplementation"]

MeanLossImplementation[type_, dims_] := Scope[
	{input, target, postfMean, postfSum, postfNoAgg} = MakeVariableLengthLossData[
		GetInputMetaNode["Input"], GetInputMetaNode["Target"]];

	diff = SowMinus[input, target];
	loss = SowFlatMean @ SowNode[If[type === "L1", "abs", "square"], diff];

	SetOutput["Loss", postfMean @ loss];

	meanDims = Range[Count[dims, _Integer]];

	If[ShouldSetMetricQ["IOU"],
		(* Here we can assume that the final dim of both the input and target is 4*)
		zeroIndex = SowFixedArray["0", NumericArray[{0}, "Integer8"]];
		oneIndex = SowFixedArray["1", NumericArray[{1}, "Integer8"]];
		twoIndex = SowFixedArray["2", NumericArray[{2}, "Integer8"]];
		threeIndex = SowFixedArray["3", NumericArray[{3}, "Integer8"]];

		inp0 = SowTake[input, zeroIndex, -1];
		inp1 = SowTake[input, oneIndex, -1];
		inp2 = SowTake[input, twoIndex, -1];
		inp3 = SowTake[input, threeIndex, -1];

		tgt0 = SowTake[target, zeroIndex, -1];
		tgt1 = SowTake[target, oneIndex, -1];
		tgt2 = SowTake[target, twoIndex, -1];
		tgt3 = SowTake[target, threeIndex, -1];

		x1 = SowNode["_maximum", {inp0, tgt0}];
		y1 = SowNode["_maximum", {inp1, tgt1}];
		x2 = SowNode["_minimum", {inp2, tgt2}];
		y2 = SowNode["_minimum", {inp3, tgt3}];

		intersectionArea = SowHad[SowMaxScalar[SowMinus[x2, x1], 0], SowMaxScalar[SowMinus[y2, y1], 0]];

		inputArea = SowHad[SowMinus[inp2, inp0], SowMinus[inp3, inp1]];
		targetArea = SowHad[SowMinus[tgt2, tgt0], SowMinus[tgt3, tgt1]];

		iou = SowDivide[intersectionArea, SowPlusEps @ SowMinus[SowPlus[inputArea, targetArea], intersectionArea]];
		
		SetMetric["IOU", postfMean @ SowMeanAxis[iou, meanDims]];
	];

	If[ShouldSetMetricQ["GoodnessOfFit"],
		inp = SowInsertDim[input, -1]; (* b * ... * 1 *)
		tgt = SowInsertDim[target, -1];
		squaredTarget = SowSquare[tgt];
		residual = SowMinus[inp, tgt];
		squaredResidual = SowSquare[residual];
		absoluteResidual = SowNode["abs", residual];

		squaredTarget = SowMeanAxis[squaredTarget, meanDims]; (* b * 1 *)
		squaredResidual = SowMeanAxis[squaredResidual, meanDims];
		absoluteResidual = SowMeanAxis[absoluteResidual, meanDims];
		tgt = SowMeanAxis[tgt, meanDims];
 		
		concat = SowJoin[absoluteResidual, squaredResidual, squaredTarget, tgt, -1]; (* b * 4 *)
		
		SetMetric["GoodnessOfFit", postfMean @ concat];
	];	
]


PackageScope["SowMetaSoftmax"]

SowMetaSoftmax[mn_MetaNode, level_, logQ_:False] := Scope[
	logchooseF = If[logQ, SowLogSoftmax, SowSoftmax];
	lnode = mn["LengthNode"];
	(* Apply a mask if necessary, and move the dimension of interest at last *) 
	If[(lnode =!= None) && (level === 1),
		out = SowSeqMask[mn["Timewise"], lnode, "-1e37"];
		out = ToMetaNode[logchooseF[out, 0], mn, True]
	,
		(* optimization: used Packed to avoid coercing into Timewise or Batchwise with transpose *)
		{packed, transposedQ} = mn["Packed"];
		(* If Packed is actually Timewise, then WL Level spec does not correspond correctly to MXNet level *)
		If[(level === 1) && transposedQ, level = 0];
		out = ToMetaNode[logchooseF[packed, level], mn, transposedQ];
	];
	out
]