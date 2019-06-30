Output: TensorT[$Dimensions, RealT]

Parameters: 
	$Dimensions: EncoderDimensionsT[]

Upgraders: {
	"11.3.9" -> Append["Dimensions" -> {}]
}

AcceptsLists: Function[
	#Dimensions =!= {}
]

ToEncoderFunction: Function[
	makeReplacer[$BooleanDispatch, #Dimensions, False] /* 
		If[ContainsQ[#Dimensions, _LengthVar], toNAList, toNA]["UnsignedInteger8"]
	(* ideal would be to return Bool Array, if such a type is available in the future *)
]

MLType: Function["Boolean"]

EncoderToDecoder: Function[
	{"Boolean", "InputDepth" -> Length[#Dimensions]}
]

TypeRandomInstance: Function[
	RandomChoice[{False,True}, TDimensions[#2]]
]

$BooleanDispatch = Dispatch[{
	True -> 1, 
	False -> 0, 
	r_Real /; 0 <= r <= 1 :> r, (* <- let pre-encoded numbers through *)
	l_ :> EncodeFail["`` was not True or False", l]
}];