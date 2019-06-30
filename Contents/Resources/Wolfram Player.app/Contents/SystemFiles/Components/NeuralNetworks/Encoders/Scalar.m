Output: TensorT[{1}]

ToEncoderFunction: Function[
	UnsafeQuietCheck[
		ArrayReshape[
			toNumericArray[#],
			{Length[#], 1}
		],
		EncodeFail["input was not a list of numeric values"]
	]&
]

TypeRandomInstance: Function[
	RandomReal[]
]

MLType: Function["Numerical"]

InputPattern: Function[_?NumberQ]

EncoderToDecoder: Function[
	{"Scalar"}
]

