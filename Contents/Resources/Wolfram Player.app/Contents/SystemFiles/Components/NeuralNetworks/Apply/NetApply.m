Package["NeuralNetworks`"]


PackageScope["NetApply"]

General::inpmiss = "Required input slot `` was not provided.";
General::inpunk = "Unknown input `` provided.";
General::nninit = "Cannot `` net: unspecified `` for ``.";
General::nfspec = "Cannot `` net: `` is not fully specified."
General::nfspecdebug = "Cannot `` net: `` is not fully specified."
General::nfspecunk = "Cannot `` net: net is not fully specified."

PackageScope["$EvaluationContext"]
PackageScope["$EvaluationDataType"]
PackageScope["$EvaluationTrainingMode"]
PackageScope["$EvaluationMixedPrecisionQ"]

PackageScope["$NPGData"]
PackageScope["$RSData"]
PackageScope["$RSData2"]

(* ^ these store data taken from the input association corresponding to 
NetPortGradient["foo"] -> ... or 
NetPort[All,"States"] -> <|...|>.
*)

$EvaluationContext = 1;
$EvaluationTrainingMode = False;
$EvaluationBatchSize = True; 
$EvaluationDataType = 0;
$EvaluationMixedPrecisionQ = False;

NetApply[arg__, NetEvaluationMode -> tspec_] := Scope[
	$EvaluationTrainingMode = ParseEvalMode[tspec];
	NetApply[arg]
];

NetApply[arg__, WorkingPrecision -> wprec_] := Block[
	{
		$EvaluationDataType = ParseWorkingPrecision[wprec],
		$EvaluationMixedPrecisionQ = If[wprec === "Mixed", True, False]
	},
	NetApply[arg]
];

NetApply[arg__, "ProgressCallback" -> f_] := Block[
	{$BatchHook := f @ N[Part[$BatchProgress, 1] / Part[$BatchProgress, 2]]},
	NetApply[arg]
];

NetApply[arg__, "ProgressReporting" -> rep_] := Block[
	{$BatchProgress = {0, 1}},
	If[TrueQ[rep],
		WithInteractiveProgressBar[NetApply[arg], $BatchProgress],
		NetApply[arg]
	]
];

seedToMXSeed = MatchValues[
	i_Integer := i;
	Automatic := BlockRandom[RandomInteger[2^31], RandomSeeding -> Automatic];
	Inherited := RandomInteger[2^31];
	other_ := Hash[other]
];

NetApply[arg__, RandomSeeding -> seed_] := (
	MXSeedRandom @ seedToMXSeed @ seed;
	NetApply[arg]
);

NetApply[arg__, TargetDevice -> device_] := Block[
	{$EvaluationContext = ParseTargetDevice[device]},
	NetApply[arg]
];

NetApply[arg__, BatchSize -> batchSize_] := Block[
	{$EvaluationBatchSize = batchSize},
	If[!PositiveIntegerQ[batchSize], ThrowFailure["optpi", BatchSize, batchSize]];
	NetApply[arg]
];

NetApply[net_] := NetApply[net, <||>];

$AllStatesPort = NetPort[All, "States"];

NetApply[inet_, idata_, prop_:Automatic] := Block[
	{
		inputs, numInputs, 
		data = idata, net = inet, splitq, $batchsize, 
		func, $RSData2, $NPGData
	},
	inputs = Inputs[net]; 
	numInputs = Length[inputs];

	If[AssociationQ[data],
		(* check that the right number of slots were provided *)	
		If[numInputs =!= Length[data], 
			If[KeyExistsQ[data, $AllStatesPort],
				$RSData2 = data[$AllStatesPort];
				KeyDropFrom[data, $AllStatesPort];
			];
			If[!FreeQ[prop, NetPortGradient],
				$NPGData = KeySelect[data, MatchQ[NetPortGradient[_String]]];
				KeyDropFrom[data, Keys[$NPGData]];
				(* its ok to have extra keys if they are for out grads *)
			];
			If[numInputs =!= Length[data],
				ThrowFailure["invargc", Length[data], numInputs]];
		];
		(* get the data in canonical order *)
		data = LookupOr[data, Keys[inputs], ThrowFailure["inpmiss", #]&];
	,
		If[numInputs =!= 1,
			Which[
				DigitStringKeysQ[inputs] && ListQ[data] && Length[data] === numInputs,
					Null, (* CatenateLayer etc *)
				numInputs === 2 && (RuleVectorQ[data] || RuleQ[data]),
					data = {Keys[data], Values[data]},
				True,
					ThrowFailure["invargc", 1, numInputs];
			];
		,
			data = {data};
		]
	];

	Which[
		FullySpecifiedNetQ[net], 
			(* it's ready to go *)
			$batchsize = Automatic
		, 
		InitializedNetQ[net], 
			(* arrays present, but we need to infer shape based on the input *)
			{net, $batchsize, splitq} = JITInferNet[net, data];
			If[splitq, data = First[data]];
			inputs = Inputs[net];
		,
		True, 
			(* net is not ready *)
			ThrowNotSpecifiedFailure[net, "evaluate", "Initialized"];
	];
	
	(* force $batchsize to be resolved if it wasn't resolved by JITInferNet because
	the net was already specified *)
	If[$batchsize === Automatic, 
		$batchsize = GuessBatchSize[First[inputs, Null], First[data, Null]];
	];

	$batchsize = If[$batchsize =!= None, $EvaluationBatchSize, False];
	func = ToNetEvaluator[net, {
		$batchsize, $EvaluationTrainingMode, 
		{$EvaluationContext, $EvaluationDataType, $EvaluationMixedPrecisionQ},
		If[FastValueQ[$RSData], PleaseSetRSData, Identity] @ prop
	}];

	SwitchNumericArrayFlag[ 
		Catch[func @@ data, EncodeFail],
		data
	]
];

NetApply[net_, len___] := ThrowFailure["invargc", Length[{len}], 1];

General::invargc = "`` inputs provided, `` were expected."
checkInputCount[netn_, datan_] := If[netn =!= datan, 
	ThrowFailure["invargc", datan, netn]
];

PackageScope["NetApplyFast"]

NetApplyFast[net_, data_] := 
	Replace[$FastPathCache["get"[net, Null]], Null :> ToFastPathFunction[net]][net, data];


PackageScope["ToFastPathFunction"]

ToFastPathFunction[net_] := ModuleScope[
	Scan[$FastPathCache["remove"[First[#]]]&, $FastPathCache["listTable"[]]];
	If[!FullySpecifiedNetQ[net], Return[NetApply, Module]];
	(* ^ if we need to do JIT specification, caching is pointless due to instance evaporation *)
	inputs = Inputs[net];
	If[Length[inputs] =!= 1,
		func = NetApply,
		func = With[
			{inputChecker = fastChecker @ First @ inputs, 
			 evalFunc = ToNetEvaluator[net, {False, False, {1, 0, False}, Automatic}]},
			Function[
				If[inputChecker, Catch[SwitchNumericArrayFlag[evalFunc[#2], #2], EncodeFail], NetApply[#1, #2]]
			] /. Hold[h_] :> h
		];
	];

	$FastPathCache["put"[net, Null, func]];
	func
];

fastChecker[TensorT[{d_Integer}]] := Hold @ And[VectorQ[#2, NumberQ] || (NumericArrayQ[#2] && ArrayDepth[#2] == 1), Length[#2] === d]
fastChecker[TensorT[dims:{__Integer}]] := Hold[machineArrayDimensions[#2] === dims];
fastChecker[EncoderP["Scalar"]] := Hold @ NumberQ[#2];
fastChecker[EncoderP["Boolean"]] := Hold @ BooleanQ[#2];
fastChecker[EncoderP["Image"]] := Hold @ Image`ValidImageQ[#2];
fastChecker[EncoderP["Characters"]] := Hold @ StringQ[#2];
fastChecker[EncoderP["Tokens"]] := Hold @ StringQ[#2];
fastChecker[v:VarSequenceP[]] := With[
	{dimPatt = ReplacePart[TDimensions[v], 1 -> _]}, 
	Hold[MatchQ[machineArrayDimensions[#2], dimPatt]]];

fastChecker[_] := Return[NetApply, Module];



PackageScope["GuessBatchSize"]

GuessBatchSize[_, {}] := ThrowFailure["netemptin"];

GuessBatchSize[Null, Null] := None; (* for graphs with no inputs *)

GuessBatchSize[enc_NetEncoder, data_] := 
	If[ListQ[data],
		If[AcceptsListsQ[enc],
			If[FailureQ[Quiet[enc @ First @ data]],
				None,
				Length[data]
			],
			Length[data]
		],
		None
	];

GuessBatchSize[SequenceT[_, enc_NetEncoder], data_] :=
	If[!ListQ[data], None,
		If[ListOfListsQ[data],
			If[AcceptsListsQ[enc], 
				Replace[arrayDimensions[data], {
					{n_, _} :> n,
					_ :> None
				}]
			,
				Length[data]
			],
			None
		]
	];

GuessBatchSize[t_TensorT, data_] := 
	If[!ListQ[data] && !NumericArrayQ[data], 
		None,
		tr = TRank[t];
		If[tr === 0, 
			Length[data],
				dr = maxRank[data];
				Which[
					dr == tr, None, 
					dr == tr + 1, Length[data],
					True, None
				]
		]
	];

GuessBatchSize[ListT[_, t_] | {t_, ___}, data_] := 
	If[!ListQ[data] || data === {}, 
		None, (* <- will fail later *)
		GuessBatchSize[t, First[data]]
	];

(* other types *)
GuessBatchSize[_, data_] := If[ListQ[data], Length[data], None];

(* TODO: Move this into kernel *)

maxRank[{}] := 0;
maxRank[data_ ? NumericArrayQ] := ArrayDepth[data];
maxRank[data_ ? PackedArrayQ] := ArrayDepth[data];
maxRank[data_List] := 1 + maxRank[First @ data];
maxRank[_] := 0;

PackageScope["JITInferNet"]

$batchsize = Automatic;
$seqsizes = <||>;

JITInferNet[net:head_[assoc_Association, _], data_] := Scope[
	inputs = Inputs[assoc];
	dlen = Length[data];

	(* multiport layers that haven't been expanded get expanded now *)
	If[MatchQ[inputs, <|$Multiport -> _|>] && dlen === 1,
		multi = True; data = First[data]; dlen = Length[data];
		SetMultiport[inputs, dlen];
	,
		multi = False;
	];

	(* if all the net's existing input types were already fully specified, adding the
	dimensions of the actual input data will add nothing, so bail now. *)
	If[AllTrue[inputs, FullySpecifiedTypeQ], 
		ThrowNotSpecifiedFailure[assoc, "evaluate", "Initialized"];
	];
	(* TODO: this prevents you from doing a property on a partially specified net *)
	checkInputCount[Length[inputs], dlen];
	{inputs, batchsize} = JITInferInputTypes[data, inputs];
	assoc["Inputs"] = inputs;
	net2 = CatchFailure[Inherited, ConstructWithInference[head, assoc]];
	If[FailureQ[net2], 
		If[inferenceFailureQ[net2],
			net2 = net, (* punt to reporting the unspecified part *)
			ThrowRawFailure[net2] 
		];
	];
	If[!FullySpecifiedNetQ[net2], 
		If[$DebugMode, Print[PrettyForm[NData @ net2]]];
		ThrowNotSpecifiedFailure[net2, "evaluate", "Initialized"]
	];
	{net2, batchsize, multi}
]; 

inferenceFailureQ[Failure[_, <|"MessageTemplate" :> MessageName[_, "tyinc"|"tyfail"|"tyufail"], _|>]] := True;
inferenceFailureQ[_] := False;

PackageScope["JITInferInputTypes"]

General::netemptin = "Empty inputs are not permitted."
dFirst[a_] := First[a, ThrowFailure["netemptin"]];

JITInferInputTypes[data_, types_] := Scope[
	{keys, types} = KeysValues[types];
	unbatched = CatchFailure[Inherited, inferUnbatchedInputs[data, types, keys]];
	If[FailureQ[unbatched],
		batched = inferBatchedInputs[data, types];
		If[FailureQ[batched], ThrowRawFailure[unbatched]];
		{AssociationThread[keys, batched], Length @ dFirst @ data}
	,
		{AssociationThread[keys, unbatched], None}
	]
];	

inferUnbatchedInputs[data_, types_, keys_] := 
	MapThread[
		Replace[infer[#1, #2], $Failed :> InputErrorHandler[#3, #2][#1]]&,
		{data, types, keys}
	];

checkZero[{0}] := ThrowFailure["netemptin"];
checkZero[e_] := e;

(* Assume the user intended batched evaluation
   when feeding a list of NumericArray, so fail 
   for the unbatched test *)
infer[{_NumericArray..}, _] := $Failed

infer[d_, t_TensorT] := Scope[
	If[FailureQ[dims = checkZero @ machineArrayDimensions[d]], ReturnFailed[]];
	st = DefaultedType @ TType[t];
	UnifyTypes[TensorT[dims, st], t]
];

infer[d_, enc_NetEncoder] := 
	If[ListQ[d] && !AcceptsListsQ[enc], $Failed, enc];

infer[d_, SequenceT[_LengthVar, type_]] := Scope[
	If[!ListQ[d] || d === {}, ReturnFailed[]];
	type = TensorT[{}, type];	
	If[FailureQ[type = infer[dFirst[d], type]], ReturnFailed[]];
	If[!AllSameBy[d, Dimensions], ReturnFailed[]];
	SequenceT[NewLengthVar[], type]
];

infer[d_, EitherT[alts_]] := Scope[
	Do[If[!FailureQ[type = infer[d, alt]], Return[type, Block]], {alt, alts}];
	$Failed
];

infer[ds_, ListT[n_, t_]] := Scope[
	If[!ListQ[ds] || ds === {}, ReturnFailed[]];
	If[IntegerQ[n] && Length[ds] != n, ReturnFailed[]];
	Do[If[FailureQ[t = infer[d, t]], ReturnFailed[]], {d, ds}];
	Table[t, Length[ds]]
];

General::netcni = "Could not infer unspecified parameters from the given inputs."

(* this won't come up very often, maily layers that use a SwitchedType without
a fallback. totally heuristic, open to juggling the order here. *)
infer[d_, TypeT] := Which[
	RealQ[d], ScalarT,
	PositiveMachineIntegerQ[d], IndexIntegerT[Infinity],
	MachineIntegerQ[d], IndexIntegerT[All],
	NumberQ[d], ScalarT,
	MachineIntegerArrayQ[d] && Min[d] > 0, TensorT[arrayDimensions[d], IndexIntegerT[Infinity]],
	MachineIntegerArrayQ[d], TensorT[arrayDimensions[d], IndexIntegerT[All]],
	MachineArrayQ[d], TensorT[arrayDimensions[d]],
	True, ThrowFailure["netcni"]
];

infer[d_, t_] := If[TestType[d, t], t, $Failed];

inferBatchedInputs[data_, types_] := Scope[
	If[!ListOfListsQ[data], ReturnFailed[]];
	If[!AllSameBy[data, Length], ReturnFailed[]];
	MapThread[
		Replace[inferBatch[#1, #2], $Failed :> ReturnFailed[]]&,
		{data, types}
	]
];

inferBatch[d_, t_TensorT] := Scope[
	If[FailureQ[dims = checkZero @ machineArrayDimensions[d]], ReturnFailed[]];
	UnifyTypes[TensorT[Rest @ dims, DefaultedType @ TType[t]], t]
];

inferBatch[d_, SequenceT[len_LengthVar, type_]] := Scope[
	If[!ListQ[d] || d === {} || !ListOfListsQ[d], ReturnFailed[]];
	If[Apply[SameQ, lens = Length /@ d], len = dFirst[lens]];
	If[Min[lens] === 0, ReturnFailed[]];
	type = TensorT[{}, DefaultedType @ type];
	If[FailureQ[type = infer[d[[1,1]], type]], ReturnFailed[]];
	SequenceT[len, type]
];

inferBatch[d_, t_] := infer[dFirst[d], t];