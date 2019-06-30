Inputs: 
	$Input: TensorT[$$Dimensions, AtomT]
	$Target: TensorT[$$Dimensions, AtomT]

Outputs:
	$Loss: ScalarT 

Parameters:
	$$Dimensions: SizeListT[]

AllowDynamicDimensions: True

IsLoss: True

Writer: Function[
	MeanLossImplementation["L1", #$Dimensions];
]

Tests: {
	{"Input" -> 3} -> "_BpevayjPuJI_OdsqpH0qyNc=2.867776e-1",
	{"Input" -> {3, 3}} -> "_X60SkhiiT3Y_PBgdmHGIFR4=3.677752e-1",
	{"Input" -> "Scalar", "Target" -> "Scalar"} -> "_U8T8OMC1S5E_CnBwi6oyuUo=9.322849e-2",
	{"Input" -> "Varying"} -> "_BpevayjPuJI_H9tlpdKg/i0=2.867776e-1",
	{"Input" -> {"Varying", 2}} -> "_RzpwkzDVovY_T2HVyzGRH3w=2.517619e-1"
}