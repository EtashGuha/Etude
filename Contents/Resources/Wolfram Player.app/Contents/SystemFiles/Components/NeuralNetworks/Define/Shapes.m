Package["NeuralNetworks`"]


PackageScope["Map1"]

Map1[f_][in_] := Map[f, in]; 
(* ^ in should be {x}, but we don't bother checking *)


PackageScope["EqualityShape"]
PackageScope["EqualityRank"]

EqualityShape[shapes_List] := 
	If[MatrixQ[shapes],
		List @ Map[
			TrySetAll[#, ShapeException["all input dimensions should be equal"]]&, 
			Transpose @ shapes
		],
		ShapeException["all input ranks should be equal"]
	];

EqualityRank[ranks_List] := 
	List @ TrySetAll[ranks, ShapeException["all input ranks should be equal"]];


PackageScope["MakeRankAndShapeFunctionRules"]

MakeRankAndShapeFunctionRules[_, _, _, _, {}, _, _] := {};

MakeRankAndShapeFunctionRules[sym_, typef_, rankf_, shapef_, inputs_, outputs_, assoc_] := {
	MakeRankAndShapeFunctionRules[sym, typef, rankf,  None, inputs, outputs, assoc],
	MakeRankAndShapeFunctionRules[sym, typef, None, shapef, inputs, outputs, assoc]
}

General::nettypeinc = "Type inconsistency in ``: ``."
stringToTypeInc[sym_][e_] := If[StringQ[e], ThrowFailure[MessageName[sym, "nettypeinc"], sym, e], Join @@ e];

tensorOfRank[rank_, type_] := TensorT[ListT[rank, SizeT], type];

(* if there is no type function, we use the base types from the layer definition *)
MakeRankAndShapeFunctionRules[sym_, None, rankf_, None, inputs_, outputs_, assoc_] := With[
	{ttypes = TType /@ (assoc @@@ Join[inputs, outputs])},
	Join[inputs, outputs] -> RawComputedType[ 
		MapThread[tensorOfRank, {
			stringToTypeInc[sym] @ ForwardInverseRank[rankf, TRank[#, NaturalT]& /@ inputs, TRank[#, NaturalT]& /@ outputs],
			ttypes
		}],
		True
	]
];

(* if there is a type function, we use it to provide the types that are mapthreaded with the size lists from the
rank function to produce the final arrays *)
MakeRankAndShapeFunctionRules[sym_, typef_, rankf_, None, inputs_, outputs_, assoc_] :=
	Join[inputs, outputs] -> RawComputedType[ 
		MapThread[tensorOfRank, stringToTypeInc[sym] /@ {
			ForwardInverseRank[rankf, TRank[#, NaturalT]& /@ inputs, TRank[#, NaturalT]& /@ outputs],
			ForwardInverseType[typef, TType /@ inputs, TType /@ outputs]
		}],
		True
	];

(* we could use AtomT for ttypes here and it would also work, I think, as the rank function already provided
this information as above *)
MakeRankAndShapeFunctionRules[sym_, typef_, None, shapef_, inputs_, outputs_, assoc_] := With[
	{ttypes = If[typef === None, Map[TType, assoc @@@ Join[inputs, outputs]], None]},
	Join[inputs, outputs] -> RawComputedType[ 
		MapThread[TensorT, {
			stringToTypeInc[sym] @ ForwardInverseShape[shapef, TDimensions /@ inputs, TDimensions /@ outputs],
			If[ttypes =!= None, ttypes,
				Flatten[ForwardInverseType[typef, TType /@ inputs, TType /@ outputs], 1]
			]
		}],
		VectorQ[inputs, FixedDimTensorQ]
	]
];

FixedDimTensorQ[TensorT[_List | ListT[_Integer, _], Except[_TensorT]]] := True;
FixedDimTensorQ[_] := False;


PackageScope["ShapeException"]

ShapeException[fmt_String, args___] := Throw[StringForm[fmt, args] /. sf_StringForm :> StringFormToString[sf]];


PackageScope["HandleShapeException"]

SetHoldRest[HandleShapeException];
HandleShapeException[head_, body_] := Replace[Catch[body], s_String :> FailValidation[head, s]];


PackageScope["ValidateShape"]

(* This is because multiple layers share some of these shape functions, so we can't
hardcode the FailValidation call into the body of these functions. Hence we return a string,
and then use ValidateShape inside the ComputedType to call FailValidation with the right symbol
later.
*)

ValidateShape[symbol_, shape_] := ValidateShape[symbol, shape, Identity];
ValidateShape[symbol_, msg_String, f_] := FailValidation[symbol, msg];
ValidateShape[symbol_, shape_, f_] := If[FailureQ[shape], shape, f[shape]];


PackageScope["BroadcastShapeMerge"]

(* 
Broadcasts all lower-rank inputs up to the largest rank, calls func on all matching
dims, with three args: list of dims at level i; i itself; whether level i is the level 
given to BroadcastShapeMerge (assumed relative to the first dim of the lowest-rank tensor).
*)

BroadcastShapeMerge[idims_, level_, func_] := Scope[
	If[!ListOfListsQ[idims] || idims === {}, Return[$Failed]];
	ranks = Length /@ idims; 
	{minrank, maxrank} = MinMax[ranks];
	If[level > minrank, ShapeException["level `` should not exceed ``, the rank of lowest-rank input", level, minrank]];
	idims = PadLeft[#, maxrank, Inherited]& /@ idims;
	level = level + maxrank - minrank;
	tdims = Transpose[idims];
	IMap[func[#2, #1, #1 === level]&, tdims]
];


PackageScope["CatenateShape"]
PackageScope["CatenateRank"]

CatenateRank[ranks_, level_] := 
	If[level > Min[Cases[_Integer] @ ranks], 
		ShapeException["specified level (``) cannot exceed rank of lowest-rank input (``)", level, Min[ranks]],
		Max[ranks]
	];

CatenateShape[idims_, level_] := 
	BroadcastShapeMerge[
		idims, level, 
		If[#3, catMergeOnLevel @ #, catMergeOffLevel @ #]&
	];

catMergeOffLevel[dims_] := TrySetAll[
	dims /. Inherited -> Indeterminate,
	ShapeException["cannot catenate when off-level dimensions `` do not match", CommaForm[Cases[dims, _Integer], " and "]]
];

catMergeOnLevel[list_List /; MemberQ[list, _LengthVar]] := LengthVar[Mod[Hash[list], 2^31]]; 
(* ^ gives us a determnistic way to derive a new lengthvar from the old one(s) to represent that
there was an unknown transformation *)

catMergeOnLevel[list_List] := Total[list]
catMergeOnLevel[_] := Indeterminate;


PackageScope["PartShape"]
PackageScope["PartRank"]

PartRank[rank_, spec_List] := (rank - Count[spec, _Integer])
PartRank[rank_, _Span] := rank;
PartRank[rank_, _Integer] := rank - 1;

PartShape[{}, spec_] := ShapeException["cannot take part `` of a scalar", spec];
PartShape[indim_List, {spec_}] := Prepend[Rest[indim], iPartShape[First @ indim, spec]];
PartShape[indim_List, spec_List] := Scope[
	specLen = Length[spec];
	If[specLen > Length[indim],
		ShapeException["the number of part specifications cannot exceed the input rank"]
	];
	newShapes = MapThread[iPartShape[#1, #2]&, {indim[[1 ;; specLen]], spec}];
	Join[newShapes, indim[[(specLen + 1) ;; All]]]
];

iPartShape[n1_Integer, Span[a_, b_]] := Scope[
	a = deneg[a, n1]; b = deneg[b, n1];
	If[a > b, ShapeException["negative range ``;;`` not supported", a, b]];
	c = b - a + 1;
	c
];
iPartShape[dim_Integer, a_Integer] := (deneg[a, dim]; Nothing);
iPartShape[lv_LengthVar, spec_] := If[spec === 1;;-1, lv, ShapeException["cannot take part of a dynamic dimension"]];
iPartShape[v_Var, 1;;-1] := v;
iPartShape[_Var, _Span] := Indeterminate;
iPartShape[_Var, _Integer] := Nothing;
iPartShape[_, _] := $Unreachable;


deneg[i_, _] := i;
deneg[i_, max_Integer] := Which[
	Abs[i] > max, ShapeException["the specification `` cannot reference positions greater than ``", i, max],
	i < 0, max + i + 1, 
	True, i
];

PackageScope["OrderingShape"]

OrderingShape[{inLength_}, spec_] := Which[
	spec === All,
		{inLength},
	IntegerQ[inLength] && Abs[spec] > inLength,
		ShapeException["the specification `` cannot reference positions greater than ``", spec, inLength],
	True,
		{Abs[spec]}
];

PackageScope["ExtractShape"]

ExtractShape[{inShape_, specShape_}] := Scope[
	If[MatchQ[Last[specShape, None], _LengthVar], 
		ShapeException["the final dimension of the \"Position\" port must be fixed"]
	];
	If[Length[specShape] === 0, specShape = {1}];
	depth = Last @ specShape;
	If[depth > Length[inShape], 
		ShapeException["the final dimension `` of the \"Position\" port must not exceed the rank `` of the \"Input\" port", 
			depth, Length[inShape]]];
	{Join[Most @ specShape, Drop[inShape, depth]]}
];

PackageScope["DotShape"]
PackageScope["DotRank"]

DotRank[{}] := ShapeException["cannot take dot product of zero arrays"];
DotRank[dims_List] := Scope[
	r = Total[dims] - 2 * (Length[dims]-1);
	If[r < 0, ShapeException["inputs to dot product have insufficient rank"]];
	r
];

DotShape[{}] := ShapeException["cannot take dot product of zero arrays"];
DotShape[dims_List] := Fold[dot2, dims];

dot2[l_List, r_List] := (
	TrySet[Last[l], First[r], 
		ShapeException["cannot take dot product of arrays with shapes `` and ``", fmtDimsList[l], fmtDimsList[r]]
	];
	Join[Most[l], Rest[r]]
);

dot2[{}, _List] := ShapeException["scalar values cannot be used with Dot"];
dot2[_List, {}] := dot2[{}, {}];


PackageScope["PoolingShape"]

PoolingShape[insize_, pad_, kern_, stride_, "valid"] := 
	Floor[(insize + Total[pad, {-1}] - kern)/stride] + 1;

PoolingShape[insize_, pad_, kern_, stride_, "full"] := 
	Ceiling[(insize + Total[pad, {-1}] - kern)/stride] + 1;


PackageScope["ConvolutionShape"]

ConvolutionShape[insize_, pad_, kern_, stride_, dilation_] := 
	Floor[(insize + Total[pad, {-1}] - (dilation * (kern - 1) + 1)) / stride + 1];


PackageScope["MaybeDyn"]

MaybeDyn[other_] := other;
MaybeDyn[list_List] := 
	If[list === {} || MatchQ[First[list], _Integer | SizeT], list,
		ReplacePart[list, 1 :> NewLengthVar[]]];


PackageScope["DeconvolutionShape"]

DeconvolutionShape[insize_, pad_, kern_, stride_, dilation_] := 
	MaybeDyn[stride * (insize - 1) + dilation * (kern - 1) + 1 - Total[pad, {-1}]];


PackageScope["FlattenShape"]
PackageScope["FlattenRank"]

FlattenRank[r_, Infinity] := 1;
FlattenRank[r_, n_] := If[Abs[n] >= r, 
	ShapeException["level specification of `` is incompatible with input tensor, which has rank ``", n, r],
	r - Abs[n]
];

FlattenShape[dims_List, Infinity] := {timesDims[dims]};
FlattenShape[{a_, 1}, Infinity|1] := a;
FlattenShape[dims_List, level_] := Scope[
	{dflatten, dpreserve} = TakeDrop[dims, level + Sign[level]];
	If[level > 0, Prepend, Append][dpreserve, timesDims[dflatten]]
];

timesDims[dims_] := If[DynamicDimsQ[dims], $Failed, Times @@ dims];

PackageScope["TransposeShape"]

TransposeShape[idims_, trans_List] := Scope[
	rank = Length[idims];
	If[rank > 4, FailValidation[TransposeLayer, "only arrays up to rank 4 are currently supported"]];
	If[
		If[VectorQ[trans, IntegerQ], 
			Length[trans] > rank,
			!AllTrue[Flatten[List @@@ trans], 1 <= Abs[#] <= rank&]
		],
		ShapeException["transpose specification `` is incompatible with input array dimensions ``", 
			Replace[trans, {a_} :> a], 
			fmtDimsList @ idims];
	];
	odims = idims;
	ApplyTransposeSpec[odims, trans];
	odims
];

PackageScope["ApplyTransposeSpec"]

SetHoldFirst[ApplyTransposeSpec];
ApplyTransposeSpec[dims_, spec:{__Integer}] :=
	ComposeTo[dims[[1 ;; Length[spec]]], Part[#, Ordering[spec]]&];

ApplyTransposeSpec[dims_, spec_] :=
	Replace[spec, {
		(a_ <-> b_) :> Swap[dims[[a]], dims[[b]]],
		(a_ -> b_) :> ComposeTo[dims[[Min[a,b] ;; Max[a,b]]], Which[b > a, RotateLeft, b < a, RotateRight, True, Identity]] 
	}, {1}];


PackageScope["ReplicateShape"]

(* note: the fact that ReplicateShape (and OTHER shape functions like AggregationShop)
 returns a list of dims, rather than an actual tensor, means that we can't represent 
 partial knowledge of the output size, like the fact that the trailing or initial dimensions
 are known but the total rank is not known. refactor if this becomes a problem at some point.
*)

ReplicateShape[dims_List, spec_, level_] := 
	Internal`UnsafeQuietCheck[
		Flatten @ Insert[dims, spec /. Automatic -> Indeterminate, level], 
		ShapeException["level must be a non-zero integer between `` and ``", -Length[dims] - 1, Length[dims] + 1]
	];

PackageScope["ReshapeShape"]

ReshapeShape[inShape_, spec_] := Scope[
	outShape = spec;

	If[ContainsAny[spec, {Inherited}],
		copyPos = Flatten@Position[spec, Inherited];
		invalidCopyPos = Select[copyPos, # > Length[inShape]&];
		If[Length[invalidCopyPos] > 0, 
			FailValidation[ReshapeLayer, "can't inherit from dimensions `` given input rank of ``.", invalidCopyPos, Length[inShape]]
		];
		outShape = ReplacePart[spec, Thread[ copyPos -> inShape[[copyPos]] ]];
	];

	If[ContainsAny[spec, {Automatic}],
		If[!MatchQ[inShape, {Repeated[_Integer | LengthVar[_]]}], Return[SizeListT[]]];
		If[MatchQ[First[inShape], LengthVar[_]] && First[spec] =!= Inherited,
			FailValidation[ReshapeLayer, "in case of variable-length inputs, the first dimension specification must be Inherited in order to infer output dimensions with Automatic."];			
		];
		inferred = Apply[Times, inShape] / Apply[Times, DeleteCases[outShape, Automatic]];
		If[!IntegerQ[inferred], 
			ShapeException["could not infer dimensions in specification `` given input shape ``", spec, inShape]
		];
		outShape = ReplaceAll[outShape, Automatic -> inferred];
	];

	If[VectorQ[inShape, IntegerQ] && Apply[Times, outShape] =!= Apply[Times, inShape], 
		ShapeException["number of elements in output array must equal number of elements in input array"]
	];

	outShape
];

PackageScope["AggregationShape"]
PackageScope["AggregationRank"]

AggregationShape[{}, specs_] := ShapeException["rank of input array should be at least 1"];
AggregationShape[dims_List, specs_] := Scope[
	set = ToLevelSet[specs, Length[dims]];
	If[FailureQ[set], ReturnFailed[]];
	Delete[dims, List /@ set]
];
AggregationShape[_, _] := $Failed;

AggregationRank[irank_, specs_] := Scope[
	set = ToLevelSet[specs, irank];
	If[FailureQ[set], ReturnFailed[]];
	irank - Length[set]
];

PackageScope["ToLevelSet"]

ToLevelSet[specs_, rank_] := Scope[
	If[!FreeQ[specs, (_Integer ? Negative) | All] && !IntegerQ[rank], ReturnFailed[]];
	(* ^ we can't compute the level set because we have a spec like -1 and we don't know the input rank *)
	$rank = rank;
	specs = ToList[specs] //. {
		i_Integer ? Negative :> (rank + i + 1),
		i_Integer /; i > rank :> ShapeException["level specification `` exceeds rank of input (``)", specs, rank]
	};
	Union @ Flatten @ Map[toLevels, specs]
];

toLevels = MatchValues[
	Span[a_, All] := levelRange[a, $rank];
	Span[All, b_] := levelRange[1, b];
	Span[a_Integer, b_Integer] := levelRange[a, b];
	All := levelRange[1, $rank];
	i_Integer := {i};
];

levelRange[a_, b_] := 
	If[a <= b, Range[a, b],
		ShapeException["level specification contains empty span"]
	];


PackageScope["CheckLevelSpec"]

CheckLevelSpec[head_, spec_, allowEmpty_:False, allowedEnum_:{}, name_:"level"] := Scope[
	If[
		Or[
			!FreeQ[spec, 0], 
			spec === {} && !allowEmpty,
			If[StringQ[spec],
				!MemberQ[allowedEnum, spec],
				!MatchQ[ToList @ spec, List @ RepeatedNull[All | _Integer | Span[_Integer|All, _Integer|All]]]
			]
		],
		FailValidation[head, 
			"`` specification was not a non-zero integer, a span, All, ``or a `` of integers or spans.",
			name, 
			If[Length[allowedEnum] == 0, "", StringJoin @ Map[QuotedString[#] <> ", "&, allowedEnum]],
			If[!allowEmpty, "non-empty list", "list"]
		]
	];
	spec
];


PackageScope["CheckConvolutionOrPoolingFunction"]

(* check whether an object is a list of integers *)
intListQ[x_List] := AllTrue[x, IntegerQ];
intListQ[_] := False

CheckConvolutionOrPoolingFunction[head_, rank_, isize_, ksize_, osize_, psize_, dilation_:1] := Scope[
	isize = isize;
	totp = Total[psize, {-1}];
	If[DynamicDimsQ[isize],
		isize[[1]] = Infinity; 
		(* ^ still allows the non-dynamic dimensions to be checked. *)
	];
	(* this check should be removed when MXNet supports 3d dilation *)
	If[(rank === 3) && (Max[dilation] =!= 1), 
		FailValidation[head, "dilation values greater than 1 are not currently supported for rank-3 inputs."]
	];
	If[rank > 3 || rank < 1, FailValidation[head, "dimensionality must be either 1, 2 or 3."]];
	If[Apply[Or, Thread[isize + totp < ksize]] || Min[osize] < 1, 
		FailValidation[head, "kernel size `` cannot exceed input size `` plus padding size ``.", 
			DimsString @ ksize, DimsString @ isize, DimsString @ totp
		]
	];
	(* If any output dims are < 1, then osize is SizeListT or list of SizeT, 
	NOT an integer list. 
	Solution: if isize + ksize are known, then out size is necessarily known, as 
	padding, stride, etc always have values. Thus is osize is not a list when ksize 
	and isize are known, the osize must have value < 1. Type system should deal with 
	this better in the long run...
	*)
	If[!VectorQ[osize, IntegerQ] && intListQ[isize] && ListQ[ksize], 
		FailValidation[head, "output with non-positive dimensions was inferred."]
	];
]

PackageScope["CheckDeconvolutionFunction"]

CheckDeconvolutionFunction[rank_, isize_, ksize_, osize_, psize_] := Scope[
	If[rank > 2 || rank < 1, FailValidation[DeconvolutionLayer, "dimensionality must be either 1 or 2."]];
	If[!Apply[And, SameQ @@@ psize], FailValidation[DeconvolutionLayer, "asymmetric padding is not yet implemented."]];
	(* If any output dims are < 1, then osize is SizeListT or list of SizeT, 
	NOT an integer list. 
	Solution: if isize + ksize are known, then out size is necessarily known, as 
	padding, stride, etc always have values. Thus is osize is not a list when ksize 
	and isize are known, the osize must have value < 1. Type system should deal with 
	this better in the long run...
	*)
	If[!VectorQ[osize, IntegerQ] && intListQ[isize] && ListQ[ksize], 
		FailValidation[DeconvolutionLayer, "output with non-positive dimensions was inferred."]
	];
]

PackageScope["CheckGroupNumberFunction"]

CheckGroupNumberFunction[head_, groupN_, inputCh_, outputCh_] := Which[
	IntegerQ[inputCh] && !Divisible[inputCh, groupN],
		FailValidation[head, "number of input channels `` is not divisible by the group number ``", inputCh, groupN],
	IntegerQ[outputCh] && !Divisible[outputCh, groupN],
		FailValidation[head, "number of output channels `` is not divisible by the group number ``", outputCh, groupN],		
	True,
		Null
]

PackageScope["CheckNoDynamicChannels"]

CheckNoDynamicChannels[head_, channels_] :=
	If[DynamicDimsQ[channels],
		FailValidationWithPath[head, $path, "the number of channels cannot be dynamic. To use a dynamic spatial dimension, try using Interleaving -> True."]];


PackageScope["CheckPaddingSize"]

CheckPaddingSize[layer_, padding_, kernel_] := Scope[
	If[Or @@ Flatten[Thread/@Thread[padding >= kernel]], 
		FailValidation[layer, "padding size `` must be smaller than KernelSize ``.", DimsString @ padding, DimsString @ kernel]
	]
]
