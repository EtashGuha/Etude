Package["NeuralNetworks`"]


PackageScope["JITInferTrainNet"]
PackageScope["$LastJITFailure"]

(* JITInferTrainNet is used to fill in missing dimensions and encoders/decoders based on the training data.
To do this, it has to have some built-in knowledge of how the loss spec *would* be interpreted in the context of 
a fully-specified version of the net the user provided. so it's a bit of a messy business. but JIT inference is
a nice feature to have because you can use to train e.g. a generic form of LeNet on many different tasks and leave
it up to NetTrain to match up the port dimensions etc. *)

$pleaseSpecMsg = " Please specify shapes for the input and/or output ports of the net before calling NetTrain."

NetTrain::nettinf1 = "Could not automatically infer types for input and outputs of partially specified net ``." <> $pleaseSpecMsg;
JITInferTrainNet[NetP[head, assoc, meta], lossSpec_, data_, metricPorts_] := Timed @ CatchFailureAsMessage[NetTrain, Scope[
	inputs = Inputs[assoc]; outputs = Outputs[assoc];
	unreflectedPorts = Flatten @ findUnreflectedPorts @ ToList[lossSpec];
	KeyDropFrom[outputs, unreflectedPorts];
	types = KeyDrop[Join[inputs, outputs], metricPorts]; 
	$port = None; $assoc = assoc; 
	$softmaxHint = !FreeQ[assoc, <|"Type" -> "Softmax", ___|>];
	ParseTrainingData[data, types, 1, inferNetTypes];
	(* here we subvert the normal intention of ParseTrainingData, which would normally
	return a generator. instead we use it as a dispathc mechanism to record what types 
	the actual inputs have to help us fill in missing types in the net. *)
	net2 = CatchFailure[General, ConstructWithInference[head, $assoc, meta]];
	If[FailureQ[net2], 
		$LastJITFailure ^= net2;
		ThrowFailure["nettinf1", MsgForm[assoc]];
	];
	net2
]];

findUnreflectedPorts = MatchValues[
	_ := {};
	str_String := str;
	list_List := Map[%, list];
	key_ -> (Automatic | _Scaled) := %[key];
	key_ -> (net_ ? ValidNetQ) := If[Length[Inputs[net]] === 1, %[key], {}];
	key_ -> other_ -> _Scaled := %[key -> other];
];

inferNetTypes[data_Association, itypes_Association, _] := Scope[
	$preserveCoders = True;
	KeyValueScan[
		Function[{key, column},
			t = itypes[$port = key];
			$isInput = KeyExistsQ[inputs, key];
			(* this is so that ambigious cases, like pure integers, can be resolved:
			if the actual output is a vector, the integers MUST be class labels *)
			$intPossible = Or[
				MatchQ[TType[t], _IndexIntegerT],
				(* ^ AtomT: real by default (otherwise there is the risk that the net will accept only integers if data consist only of integers) *)
				$softmaxHint && !$isInput && TRank[t] =!= 0
				(* ^ %HACK for softmax *)
			];
			$rankHint = TRank[t];
			$dimsHint = TDimensions[t];
			inft = getTypeOrCoder[column];
			If[!$isInput, inft = toDecoders[inft]];
			res = bridge[t, inft];
			If[$isInput,
				$assoc["Inputs", key] = res,
				$assoc["Outputs", key] = toDecoders[res]
			]
		],
		data
	]
]

toDecoders[e_] := ReplaceAll[e, enc_NetEncoder :> EncoderToDecoder[enc]];
	
General::nettinf2 = "Could not automatically find way to encode training data, which consists of ``, to be compatible with port \"``\", which expects ``." <> $pleaseSpecMsg;
bridge[VectorT[1|SizeT, AtomT|RealT] | TensorT[SizeListT[1], AtomT|RealT], ScalarT] := NetEncoder["Scalar"];
bridge[TensorT[{n_, 1|SizeT}, AtomT|RealT], TensorT[{SizeT | LengthVar[-1]}, RealT]] := TensorT[{n}, NetEncoder["Scalar"]];

(* jeromel: %TODO generalize bridges?
SetAttributes[bridge, Orderless];
bridge[dst_TensorT, src_TensorT] /; And[
	MatchQ[dimsWithout1[dst], dimsWithout1[src]],
	scalarTensorQ[dst],
	scalarTensorQ[src],
	FullySpecifiedTypeQ[DefaultedType[dst]]
] := With[
	{dims = TDimensions[dst]},
	If[dims === {}, NetEncoder["Scalar"], TensorT[dims, NetEncoder["Scalar"]]]
];
(* Remove dummy dimension of ones and replace undeterminate sizes by blanks*)
dimsWithout1[t_TensorT] := ReplaceAll[
	DeleteCases[
		TDimensions[t, {SizeT}],
		1
	],
	SizeT -> _
];
dimsWithout1[ScalarT|RealT|AtomT|_IndexIntegerT] := {};
scalarTensorQ[t_] := MatchQ[TType[t], AtomT|RealT];
*)

bridge[dst_, f_Failure] := 
	If[FullySpecifiedTypeQ[dst], dst, ThrowRawFailure[f]];

(* %HACK on the assumption this will become a CrossEntropyLoss["index"] layer *)
bridge[RealTensorT|AnyTensorT, IndexIntegerT[n_Integer]] /; $softmaxHint := TensorT[{n}]; 

bridge[dst_, src_] := Scope[
	res = UnifyTypes[dst, src]; 
	If[FailureQ[res],
		If[FullySpecifiedTypeQ[DefaultedType[dst]] || FullySpecifiedTypeQ[DefaultedType[dst /. _NetDecoder :> RealT]], 
			(* ^ the second condition is to work around that NetDecoder["Class"] *behaves* as if
			it is fully spec'd even though it has the extra SizeListT[] in it; it works on all input sizes *)
			res = dst,
			ThrowFailure["nettinf2", TypeString[src, True], $port, TypeString[dst, True]]
		];
	];
	res
];

nominalListQ[list_List] :=
	VectorQ[list, SymbolQ] || StringVectorQ[list] && CountDistinct[list] <= Ceiling[Power[Length[list], 0.55], 2];

rangeQ[list_] := Scope[u = Union[list]; (DeleteDuplicates @ Differences @ u) === {1}];

getTypeOrCoder[list_List] := CatchFailure[NetTrain, Scope @ Which[
	$intPossible && VectorQ[list, VectorQ[#, PositiveIntegerQ]&],
		TensorT[{NewLengthVar[]}, IndexIntegerT[Max[list]]],
	$intPossible && VectorQ[list, IntegerQ] && rangeQ[list],
		If[Min[list] === 1, IndexIntegerT[Max[list]], NetEncoder[{"Class", Union[list], "UnitVector"}]],
	(* ^ todo: add cases for where we know the array TYPE is IntegerT, as well, e.g. EmbeddingLayer but unkonwn input rank *)
	MachineArrayQ[list],
		TensorT @ Rest @ machineArrayDimensions[list],
	VectorQ[list, MachineArrayQ] && (SameQ @@ (dims = Map[machineArrayDimensions /* Rest, list])),
		TensorT[Prepend[First[dims], LengthVar[-1]]],
	VectorQ[list, BooleanQ],
		NetEncoder["Boolean"],
	nominalListQ[list],
		NetEncoder[{"Class", Union @ list}],
	VectorQ[list, ImageQ],
		dims= getCommonValue[ImageDimensions, list];
		cspace = getCommonValue[ImageColorSpace, list];
		Replace[$Failed :> infpanic[]] @ Quiet @ Check[
			NetEncoder[{"Image", dims, cspace}],
			$Failed
		],
	True,
		infpanic[]
]];

getTypeOrCoder[r_Real | r_Rational | r_Integer] := ScalarT;

getTypeOrCoder[_] := infpanic[];

getCommonValue[f_, vals_] := Scope[
	res = Map[f, vals];
	If[SameQ @@ res, First[res], infpanic[]]
];

General::nettinf3 = "Could not automatically infer type to use for ``." <> $pleaseSpecMsg
infpanic[] := ThrowFailure["nettinf3", PartForm[$port, "input"]];
