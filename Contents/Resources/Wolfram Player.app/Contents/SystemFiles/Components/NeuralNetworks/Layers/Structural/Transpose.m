Input: TensorWithMinRankT[2, AtomT]

Output: AnyTensorT

Parameters:
	$Specification: ValidatedParameterT[checkTransposeSpec, 1 <-> 2]

TypeFunction: Identity

ShapeFunction: Map1[TransposeShape[#, ToList @ StripVP @ $Specification]&]

RankFunction: Identity

(* transpose is actually marginally faster than swap axis, so don't think about optimizing this! *)
MXNet: 
	Name: "transpose"
	Writer: Function[
		"axes" -> writeIntList[toMXSpec[GetInputRank["Input"], ToList @ StripVP @ #Specification]]
	]

Tests: {
	{"Input" -> {3, 3}} -> "3*3_THXxbAO5Pe0_SgDj9BedOjY=3.210989e+0",
	{"Input" -> {2, 3, 4}} -> "3*2*4_DnnyJzj6W/0_cCbq19VBOuM=1.067583e+1",
	{2 <-> 3, "Input" -> {2, 3, 4}} -> "2*4*3_cdDZVZ8yoik_XQe7JaCDO/k=1.067583e+1",
	{{1 <-> 2, 3 <-> 4}, "Input" -> {1, 2, 3, 4}} -> "2*1*4*3_JO0cFHUMQCs_N6Kq7cpYFOA=1.067583e+1",
	{3, "Input" -> {1, 2, 3, 4}} -> "3*2*1*4_dAMpV1WoWCE_FV2sz6kAHNg=1.067583e+1",

	{{2 -> 4}, "Input" -> {1, 2, 3, 4}} -> "1*3*4*2_JhgVCXRZ7CM_H6mEEchVcjA=1.067583e+1",
	{{3 -> 1}, "Input" -> {1, 2, 3, 4}} -> "3*1*2*4_DW4XD9eUr2E_TNH7Z96ee7I=1.067583e+1",
	{2 <-> 3, "Input" -> {"Varying", 5, 3}} -> "3*3*5_VnF5K9CTfac_YHec8kjYRj4=1.987316e+1",

	{{3, 1, 2}, "Input" -> {1, 2, 3, 4}} -> "2*3*1*4_D6LRSAi3OWo_NMbmZb/VoEI=1.067583e+1",

	{{3, 1, 2}, "Input" -> {1, 2, 3, 4, "Integer"}} -> "2*3*1*4_GbEkvdlmVP8_IDTFkXCBGJc=1.104000e+3",
	{{3, 1, 2}, "Input" -> {1, 2, 3, 4, Restricted["Integer", Infinity]}} -> "2*3*1*4_co6eFzBR5lU_HUs7ncTIhtQ=1.162000e+3",
	{{3, 1, 2}, "Input" -> {1, 2, 3, 4, Restricted["Integer", 3]}} -> "2*3*1*4_HTTBbZi44rU_UPSbOYOblR4=5.200000e+1",

	{1 <-> 3, "Input" -> {"Varying", 5, 3}} -> "Varying dimension in port \"Output\" that isn't the first dimension.",
	{"Input" -> 3} -> "Specification 3 is not compatible with port \"Input\", which must be an array of rank \[GreaterEqual] 2.",
	{"Input" -> "Varying"} -> "Specification Varying is not compatible with port \"Input\", which must be an array of rank \[GreaterEqual] 2.",
	{2 <-> 3, "Input" -> {"Varying", 3}} -> "Type inconsistency in TransposeLayer: transpose specification 2 <-> 3 is incompatible with input array dimensions n\[Cross]3.",
	{{3, 3, 2}, "Input" -> {1, 2, 3, 4}} -> "Value of {3, 3, 2} given for the specification (first argument) was invalid: transpose specification {3, 3, 2} is not a valid permutation on 3 dimensions."
}

AllowDynamicDimensions: True

ArgumentRewriter: rewriteArgs

rewriteArgs[{r:(Rule|TwoWayRule)[_Integer, _Integer], args___}] := {{r}, args};
rewriteArgs[e_] := e;

checkTransposeSpec[ospec_] := Scope[
	spec = ospec;
	If[IntegerQ[spec], spec = 1 <-> spec];
	If[!MatchQ[ToList @ spec, {Repeated[(TwoWayRule|Rule)[_Integer, _Integer]]} | {__Integer}],
		FailValidation[TransposeLayer, "`` is not a valid transpose specification, which should be a Rule, TwoWayRule, a list of these, or a list of integers.", ospec]];
	If[VectorQ[spec, IntegerQ],
		If[Sort[spec] =!= Range[Length[spec]],
			FailValidation[TransposeLayer, "transpose specification `` is not a valid permutation on `` dimensions.", spec, Length[spec]]];
	,
		spec = Replace[spec, {e_:Except} :> e]
	];
	spec
];

toMXSpec[rank_, rules_List] := Scope[
	indices = Range[rank];
	ApplyTransposeSpec[indices, rules];
	Prepend[indices, 0]
];

spec2way[ValidatedParameter[{a_ -> b_}]] := ValidatedParameter[a <-> b];
spec2way[v_ValidatedParameter] := v /. Rule -> TwoWayRule;

Upgraders: {
	"11.3.2" -> DropAllHiddenParams /* MapAtParam[spec2way, "Specification"]
}
