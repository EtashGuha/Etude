Input: AnyTensorT

(* If input shape is unspecified but $Dimensions only contains integers
	we know what the output shape is, but ShapeFunction won't run. So we do it here. 
	third argument (deps) is None to prevent the CT from being evaluated unless the trigger is True, normally the CT will also be evaluated when deps are concrete, which we don't want.
	 The second argument in First[$Dimensions, Null] in the trigger is needed to avoid printing of a message in the type inference warmup phase at loading time. *)
Output: ComputedType[AnyTensorT, TensorT[StripVP @ $Dimensions, AtomT], None, VectorQ[StripVP @ $Dimensions, IntegerQ]]

TypeFunction: Identity

Parameters:
	$Dimensions: ValidatedParameterT[parseReshapeSpec]

parseReshapeSpec[spec_] := Which[
	!MatchQ[spec, {RepeatedNull[_Integer?(# > 0 &) | Automatic | Inherited]}],
		FailValidation[ReshapeLayer, "specification should be a list of either positive integers, Automatic or Inherited."],
	Count[spec, Automatic] > 1,
		FailValidation[ReshapeLayer, "only one Automatic dimension is allowed in specification."],
	True,
		spec
];

RankFunction: Map1[Length[StripVP @ $Dimensions]&]

ShapeFunction: Map1[ReshapeShape[#, StripVP @ $Dimensions]&]

AllowDynamicDimensions: True

FinalCheck: Function[
	If[MatchQ[TFirstDim[$Input], LengthVar[_]] && First[StripVP[$Dimensions]] =!= Inherited,
		FailValidation[ReshapeLayer, "first dimension specification must be set to Inherited for variable-length inputs."];
	];
]

Writer: Function @ Scope[
	lenNode = GetDynamicLengthNode @ First[GetInputDims["Input"], 0];
	dims = GetOutputDims["Output"];
	If[lenNode =!= None,
		(* At this point, when the first dim is Varying we are guaranteed 
		   that the first spec is Inherited, so just copy the max length *)
		dims = ReplacePart[dims, 1 -> 0];
	];
	in = GetInput["Input"];
	out = SowReshape[in, dims];
	SetOutput["Output", out];
]

MXNet:
	Name: "reshape"
	Aliases: {"Reshape"}
	Reader: Function @ Scope[
		shape = Replace[#["shape"], _?MissingQ -> #["target_shape"]];
		(* readIntList is not PackageScoped :( *)
		shape = ToExpression /@ StringTrim /@ StringSplit[StringTrim[shape, "(" | ")"], ","];
		If[MatchQ[#["keep_highest"], "False" | "0"], 
			FailImport["MXNet", "reshape", "can't import when keep_highest is set to False."]
		]; (* that would change the batch dimension *)
		If[Or@@Thread[shape < -1], 
			FailImport["MXNet", "reshape", "shape specifications -2, -3, -4 are not supported."]
		];
		"Dimensions" -> ReplaceAll[Rest@shape, {0 -> Inherited, -1 -> Automatic}]
	]

Tests: {
	(* Fixed dims *)
	{{1}, "Input" -> {}} -> "1_dZt2SS1lszQ_EpNq1f2dLlk=3.498157e-1",
	{{}, "Input" -> {1}} -> "_UlS0Sd7vNAo_N6dBguLAOZ8=3.498157e-1",
	{{4, 2}, "Input" -> {2, 4}} -> "4*2_Liad5oMW3eI_Vg5nFN5ntso=3.109017e+0",
	{{4, 2}, "Input" -> {2, 4, Restricted["Integer", 3]}} -> "4*2_RV1t4mh6/SU_DgkuU44R/7Y=1.800000e+1",
	{{4, 2}, "Input" -> {2, 4, Restricted["Integer", Infinity]}} -> "4*2_DvSjfzPR0W8_RTc4o7JQBu8=6.900000e+1",
	{{4, 2}, "Input" -> {2, 4, "Integer"}} -> "4*2_XSd2R1Ic3n4_NexvLbuXk00=1.031000e+3",
	{{1, 2, 3}, "Input" -> {6}} -> "1*2*3_SINkZe0REmI_A/KyG112bzs=1.911142e+0",
	{{6, Automatic}, "Input" -> {2, 3, 4}} -> "6*4_Ntg8a8+Gdco_ciPNUH4GGVQ=1.067583e+1",
	{{Inherited, 2, Automatic}, "Input" -> {2, 3, 4}} -> "2*2*6_fqadt/u77YI_LSZ7AyZ7RnI=1.067583e+1",
	{{2, 6, 2, Inherited}, "Input" -> {2, 3, 4}} -> "Validation failed for ReshapeLayer: can't inherit from dimensions {4} given input rank of 3.",
	{{2, Automatic, Automatic}, "Input" -> {2, 3, 4}} -> "Value of {2, Automatic, Automatic} given for the dimensions (first argument) was invalid: only one Automatic dimension is allowed in specification.",
	{{4, Automatic}, "Input" -> {2, 3}} -> "Type inconsistency in ReshapeLayer: could not infer dimensions in specification {4, Automatic} given input shape {2, 3}.",
	{{3, 2}, "Input" -> {2, 4}} -> "Type inconsistency in ReshapeLayer: number of elements in output array must equal number of elements in input array.",
	(* Dynamic *)
	{{Inherited, 3, 2}, "Input" -> {"Varying", 2, 3}} -> "3*3*2_GrO+e8U0hTg_Spqt6XZw7DE=7.379773e+0",
	{{Inherited, 3, Automatic, 2}, "Input" -> {"Varying", 6, 4}} -> "3*3*4*2_f5IwnBdlI8s_cEIjjnwqBMs=3.425674e+1",
	{{2, 3}, "Input" -> {"Varying", 3}} -> "Validation failed for ReshapeLayer: first dimension specification must be set to Inherited for variable-length inputs.",
	{{3, Automatic, 4}, "Input" -> {"Varying", 3}} -> "Validation failed for ReshapeLayer: in case of variable-length inputs, the first dimension specification must be Inherited in order to infer output dimensions with Automatic.",
	{{Automatic, 3, 4, 2}, "Input" -> {"Varying", 6, 4}} -> "Validation failed for ReshapeLayer: in case of variable-length inputs, the first dimension specification must be Inherited in order to infer output dimensions with Automatic."
}

Upgraders: {
	"11.3.1" -> RenameParam["$IDimensions" -> "$InputDimensions"],
	"11.3.2" -> DropAllHiddenParams,
	"11.3.6" -> MapAtParam[ValidatedParameter, "Dimensions"]
}
