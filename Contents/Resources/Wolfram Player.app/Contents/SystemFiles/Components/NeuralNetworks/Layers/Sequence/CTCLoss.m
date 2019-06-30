Inputs: 
	$Input: SequenceT[$$InputLength, VectorT[$$InputSize]]
	$Target: SequenceT[$$LabelLength, IndexIntegerT[$$AlphabetSize]]

Outputs:
	$Loss: ScalarT

Parameters:
	$$AlphabetSize: ComputedType[SizeT, 
		If[$$InputSize < 2, 
			FailValidation[
				"input must be a sequence of vectors of size greater than 1."],
			$$InputSize - 1
		]
	]
	$$InputSize: ComputedType[SizeT, $$AlphabetSize + 1]
	$$InputLength: LengthVar[]
	$$LabelLength: LengthVar[]

MinArgCount: 0

Constraints: Function[
	SowConstraint[$$InputLength >= $$LabelLength];
]

Writer: Function @ Scope[
	ilenNode = GetDynamicLengthNode[#$InputLength];
	tlenNode = GetDynamicLengthNode[#$LabelLength];

	useInLen = If[ilenNode === None, False, True];
	useLabLen = If[tlenNode === None, False, True];

	input = GetInput["Input", "Timewise"];
	target = GetInput["Target", "Batchwise"];

	(* zero indexing for mxnet *)
	target = SowMinusScalar[target, 1];

	(* In analogy with CrossEntropy: CTC should act on probability 
		distribution. BUT: mxnet ctc acts on pre-softmax activations.
		Use property softmax(x) = softmax(log(softmax(x))) for 
		correct 'standalone-layer' behaviour, remove during fusion. *)
	input = SowSafeLog[input];
	(* CTC loss can have between 2 and 4 input args. Decide which *)

	args = {input, SowBlockGrad @ target};
	If[useInLen, AppendTo[args, ilenNode]];
	If[useLabLen, AppendTo[args, tlenNode]];

	(* CTC loss does not support any other type but Real32. Remove when it does *)
	args = SowCast[args, $DType, "Real32"];

	loss = SowNode["_contrib_CTCLoss", args,
		"use_label_lengths" -> useLabLen,
		"use_data_lengths" -> useInLen,
		"blank_label" -> "last"
	];

	SetOutput["Loss", loss];
]

IsLoss: True

Tests: {
	{"Input" -> {20, 2}, "Target" -> {4}} -> "_TA0A6ovgTJk_Oxi9npgd29c=1.384940e+0",
	{"Input" -> {20, 2}, "Target" -> {"Varying"}} -> "_I/dT+CqR6ks_MKxmAdrQtDA=2.393548e+0",
	{"Input" -> {"z", 2}, "Target" -> {"x"}} -> "_M9tQG6cTNpM_TESDtStQ2MU=8.680964e+0",
	{"Input" -> {"z", 2}, "Target" -> {2}} -> "_VKxJsm3TVOI_EoLn/EMLQo0=3.575652e+1"
}
