Inputs:
	$Input: TensorT[{$$Channels, SizeT, SizeT}, AtomT]
	$Parameters: VectorT[6]

Output: TensorT[{$$Channels}, TensorT[$Dimensions]]

Parameters:
	$Dimensions: SizeListT[2]
	$$Channels: SizeT

MinArgCount: 1

MXNet: 
	Name: "SpatialTransformer"
	Parameters:
		$Dimensions: "target_shape"
	Writer: Function[{
		"transform_type" -> "affine",
		"sampler_type" -> "bilinear"
	}]

Tests: {
	{{2, 2}, "Input" -> {3, 5, 5}} -> "3*2*2_MR4eORhIZRA_J9HnLW8Uy3Q=2.950631e+0",
	{{2, 2}, "Input" -> {3, 5, 5, Restricted["Integer", 5]}} -> "3*2*2_ZJFJwov0X3A_JfVAiYYSRcQ=2.091037e+1"
}