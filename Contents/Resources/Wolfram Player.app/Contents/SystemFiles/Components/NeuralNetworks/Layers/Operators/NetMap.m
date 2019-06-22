Input: SequenceT[$$SequenceLength, $$InputShape]

Output: SequenceT[$$SequenceLength, $$OutputShape]

Suffix: "Operator"

Parameters:
	$Net: NetT[<|"Input" -> $$InputShape|>, <|"Output" -> $$OutputShape|>]
	$$SequenceLength: LengthVar[]
	$$InputShape: AnyTensorT
	$$OutputShape: AnyTensorT

PostInferenceFunction: Function[
	PostSet[$Input, SequenceT[$$SequenceLength, $Net["Inputs", "Input"]]];
	PostSet[$Output, SequenceT[$$SequenceLength, $Net["Outputs", "Output"]]];
]

SummaryFunction: Function[
	HoldForm[NetMapOperator] @ SummaryForm[#Net]
]

Writer: Function @ Scope[
	out = SowMetaMap[
		SowSubNet["Net", #]["Output"]&,
		GetInputMetaNode["Input"]
	];
	SetOutput["Output", out];
]

Tests: {
	{Hold @ LinearLayer[2, "Input" -> 1]} -> "3*2_TqdzaQfy9+w_fPNZLgMdZc0=5.316732e+0"
}
