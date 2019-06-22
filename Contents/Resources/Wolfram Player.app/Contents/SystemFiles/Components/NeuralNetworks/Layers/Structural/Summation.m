Input: ChannelT[SizeT, AnyTensorT]

Output: ScalarT

(* TODO: handle integers -- propagate the type *)
(*
ShapeFunction: Function[{{}}]

TypeFunction: Function[ReplaceAll[#, IndexIntegerT[n_] :> IndexIntegerT[Infinity]]]
*)

MinArgCount: 0

Writer: Function @ Scope[
	input = GetInput["Input"];
	dims = GetInputDims["Input"];
	fdim = First[dims];
	lnode = GetDynamicLengthNode[fdim];
	If[lnode =!= None, input = SowSeqMaskBatchwise[input, lnode]];
	out = SowNode["sum", input, "axis" -> Range[Length[dims]], "keepdims" -> False];
	SetOutput["Output", out];
]

AllowDynamicDimensions: True

MXNet:
	Name: "sum"

Tests: {
	{"Input" -> 3} -> "_bt+kDbKWQgU_BwIYzpiqozg=9.048177e-1",
	{"Input" -> {3, 3}} -> "_Vd3jC2T9XxE_OyXiD9uspTI=3.210989e+0",

	{"Input" -> "n"} -> "_bt+kDbKWQgU_Jl+ATQs3ShI=9.048177e-1",
	{"Input" -> {"n", 2}} -> "_O5AD+QYYhTU_BeSZvFxxktg=1.911142e+0",
	{"Input" -> {"n", "Integer"}} -> "_chFrKwV8oBQ_X0d4/cXGuLA=1.300000e+1"
}
	
Upgraders: {
	"11.3.1" -> DropParam["Dimensions"],
	"11.3.2" -> DropAllHiddenParams /* MapAt[Map @ toChannelTensor, "Inputs"]
}

toChannelTensor[RealTensorT] := ChannelT[SizeT, RealTensorT];
toChannelTensor[e_] := e;