Input: TensorT[$$InputSize, AtomT]

Output: TensorT[$$OutputSize, AtomT]

Parameters:
	$Specification: ListT[SizeT, ListT[2, NaturalT]]
	$Padding: Defaulting[EitherT[{EnumT[{"Fixed", "Reflected"}], ScalarT}], 0.]
	$$Rank: ComputedType[NaturalT, Length @ $Specification]
	$$InputSize: ComputedType[
		SizeListT[$$Rank], 
		$$OutputSize - Total[$Specification, {2}]
	]
	$$OutputSize: ComputedType[
		SizeListT[$$Rank], 
		Total[$Specification, {2}] + $$InputSize
	]

ShapeFunction: Function[{Total[$Specification, {2}]} + #]

RankFunction: Identity

TypeFunction: Function[
	DefaultedType @ Switch[$Padding,
		n_ /; NumericQ[n] && Round[n]==n && n >= 1,
			ReplaceAll[#, IndexIntegerT[n_] :> IndexIntegerT[Round @ Max[n, $Padding]]],
		n_ /; NumericQ[n] && Round[n]==n,
			ReplaceAll[#, IndexIntegerT[n_] :> IndexIntegerT[All]],
		_String, #,
		_, {RealT}
	]
]

MinArgCount: 1
PosArgCount: 1

PostInferenceFunction: Function[
	If[Not[1 <= $$Rank <= 4], FailValidation["only an input of rank less than 5 is currently supported."]];
	If[Head[$$InputSize] === List && $Padding == "Reflected" &&
		Apply[Or, GreaterEqual@@@Transpose[{Max/@$Specification, $$InputSize}]],
			FailValidation["padding sizes must be smaller than the input size when using \"Reflected\" method."]
	];
]

toMXPaddingSpec[spec_?NumericQ] := "constant";
toMXPaddingSpec["Fixed"] := "edge";
toMXPaddingSpec["Reflected"] := "reflect";
toMXPaddingSpec[_] := $Failed;

fromMXPaddingSpec["edge"] := "Fixed";
fromMXPaddingSpec["reflect"] := "Reflected";
fromMXPaddingSpec[_] := Panic[];

(* Note: MXNet only currently supports padding the spatial dims of 2d + 3d ims. *)
Writer: Function @ Scope[

	(* parse inputs *)
	padType = toMXPaddingSpec[#Padding];
	padValue = ToString @ N @ If[NumericQ @ #Padding, #Padding, 0];

	(* 1. conform 1d + 2d array to 3d array *)
	spec = #Specification;
	id = Which[
		#$Rank === 1,
			spec = Join[{{0, 0}, {0, 0}, {0, 0}}, spec];
			SowNode["reshape", GetInput["Input"], "shape" -> Join[{0}, {1, 1}, #$InputSize]],
		#$Rank === 2,
			spec = Join[{{0, 0}, {0, 0}}, spec];
			SowNode["reshape", GetInput["Input"], "shape" -> Join[{0}, {1}, #$InputSize]],
		True,
			spec = Join[{{0, 0}}, spec];
			GetInput["Input"]
	];

	(* 2. do padding assuming no padding on channel *)
	tempSpec = spec;
	tempSpec[[2]] = {0, 0};
	id = SowNode["pad", id, 
				"pad_width" -> writeIntList[Flatten @ tempSpec],
				"mode" -> padType,
				"constant_value" -> padValue
	];

	(* 3. reshape smaller arrays back to original size *)
	If[#$Rank < 3,
		id = SowNode["reshape", id, "shape" -> Join[{0}, #$OutputSize]];
	];

	(* 4. deal with channel dim padding *)
	If[spec[[2]] =!= {0, 0},
		id = SowNode["SwapAxis", id, "dim1" -> 1, "dim2" -> 2];
		tempSpec *= 0;
		tempSpec[[3]] = spec[[2]];
		id = SowNode["pad", id,
					"pad_width" -> writeIntList[Flatten @ tempSpec],
					"mode" -> padType,
					"constant_value" -> padValue
			];
		id = SowNode["SwapAxis", id, "dim1" -> 1, "dim2" -> 2];
	];

	SetOutput["Output", id];
]

MXNet:
	Name: "pad"
	Reader: Function @ Scope[
		pad = If[#mode === "constant", #["constant_value"], fromMXPaddingSpec[#mode]];
		spec = Partition[#["pad_width"][[3;;]], 2];
		{"Padding" -> pad, "Specification" -> spec}
	]


Tests: {
	{{{1, 2}}, "Input" -> {2}, "Padding" -> "Fixed"} -> "5_O2BH5+Mv2WE_WyzUPl2tpn8=2.028764e+0",
	{{{1, 2}}, "Input" -> {2}, "Padding" -> 2.3} -> "5_cSue3BJEcuU_SKth9uZZB+Y=7.692860e+0",
{{{1, 2}}, "Input" -> {3}, "Padding" -> "Reflected"} -> "6_VCZ97/C05E8_FNmTC2kJHmE=2.140722e+0",
	{{{1, 2}, {0, 3}}, "Input" -> {2, 2}, "Padding" -> 2.3} -> "5*5_W8mlxyiv1tg_CazoKztNe4Y=4.997434e+1",
	{{{2, 0}, {1, 1}}, "Input" -> {1, 2}, "Padding" -> "Fixed"} -> "3*4_IKT5G5vbTIc_LqkZ+q4sxq4=4.757159e+0",
	{{{2, 0}, {1, 1}}, "Input" -> {3, 2}, "Padding" -> "Reflected"} -> "5*4_VXbnhPEsiWc_XU3HXY4aTgA=6.058850e+0",
	{{{0, 0}, {4, 1}, {1, 2}}, "Input" -> {1, 2, 2}, "Padding" -> "Fixed"} -> "1*7*5_WCvUqvU/mb4_ejCT2TuTbnw=1.520880e+1",
	{{{0, 0}, {4, 1}, {1, 2}, {5, 1}}, "Input" -> {1, 2, 2, 3}, "Padding" -> "Fixed"} -> "1*7*5*9_V19A93wonas_ZDAnpUNKpRE=1.554339e+2",
	{{{1, 0}, {4, 1}, {1, 2}, {5, 1}}, "Input" -> {1, 2, 2, 3}, "Padding" -> -1.2} -> "2*7*5*9_Zd7zniEgj4M_OEG2jch6+ac=7.468316e+2",
	{{{0, 0}, {0, 1}, {1, 2}, {5, 1}}, "Input" -> {1, 2, 3, 6}, "Padding" -> "Reflected"} -> "1*3*6*12_XetWwG9rAtg_Q/N6PKb4khA=1.020003e+2",

	{{{1, 2}, {0, 3}}, "Input" -> {2, 2, "Integer"}, "Padding" -> 0} -> "5*5_Bix3fy8qgHY_HglnjvW16QE=2.500000e+1",
	{{{1, 2}, {0, 3}}, "Input" -> {2, 2, Restricted["Integer", 2]}, "Padding" -> 2.3} -> "5*5_HyK0Lgno6/s_cXl8FK5JpTU=5.430000e+1",
	{{{1, 2}, {0, 3}}, "Input" -> {2, 2, Restricted["Integer", 2]}, "Padding" -> 1} -> "5*5_WXAL4gINwMk_JqD9K1hQxsc=2.700000e+1",
	{{{1, 2}, {0, 3}}, "Input" -> {2, 2, Restricted["Integer", 2]}, "Padding" -> 10} -> "5*5_OqXDiz0t2TY_K9h86m++Oos=2.160000e+2",
	{{{1, 2}, {0, 3}}, "Input" -> {2, 2, Restricted["Integer", 2]}, "Padding" -> "Reflected"} -> "Validation failed for PaddingLayer: padding sizes must be smaller than the input size when using \"Reflected\" method.",
	{{{1, 2}, {0, 3}}, "Input" -> {2, 2, Restricted["Integer", 2]}, "Padding" -> "Fixed"} -> "5*5_BsSmOax665I_WTglV5aTiPs=4.500000e+1",

	{{{0, 0}, {0, 1}, {1, 2}, {5, 1}}, "Input" -> {1, 2, 2, 3}, "Padding" -> "Reflected"} -> "Validation failed for PaddingLayer: padding sizes must be smaller than the input size when using \"Reflected\" method."
}

Upgraders: {
	"11.3.1" -> RenameParam["Rank" -> "$Rank"]
}
