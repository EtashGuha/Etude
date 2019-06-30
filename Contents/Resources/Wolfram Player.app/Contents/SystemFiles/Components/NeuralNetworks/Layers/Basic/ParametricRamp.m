Input: ChannelT[$$Channels, AnyTensorT]

Output: $Input

Parameters:
	$$Channels: SizeT

Arrays:
	$Slope: ComputedType[VectorT[], VectorT[$$Channels]]

MinArgCount: 0

(* Performs channel-wise PReLU (one slope for each channel) *)
MXNet:
	Name: "LeakyReLU"
	Arrays:
		$Slope: "gamma"
	Writer: Function["act_type" -> "prelu"] 

Tests:{
	{"Input" -> {2}} -> "2_IMH4E1wGKP4_Liad5oMW3eI=7.928598e-1",
	{"Input" -> {2, 3}} -> "2*3_XNJtjBjJ1rk_Luzvstmun4A=1.911142e+0",
	{"Input" -> {2, 3, 4}} -> "2*3*4_cdwKX9/nzAo_cZaRdWoErnE=1.067583e+1",
	{"Input" -> {2, 3, 4, Restricted["Integer", 3]}} -> "2*3*4_X1FIjuzB19Y_FAuwL8KbAdI=5.200000e+1",
	{"Input" -> "Real"} -> "Specification Real is not compatible with port \"Input\", which must be an array of rank \[GreaterEqual] 1.",
	{"Slope" -> {2, 3}, "Input" -> {3}} -> "Inferred inconsistent dimensions for array \"Slope\" (a length-2 vector of real numbers versus a length-3 vector of real numbers)."
}

