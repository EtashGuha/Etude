Package["NeuralNetworks`"]


PackageScope["EncoderApply"]

Clear[EncoderApply, iEncoderApply];

EncoderApply[encoder_, input_] := SwitchNumericArrayFlag[
	encoderOutputGetSwitched @ iEncoderApply[encoder, input], 
	input
]

encoderOutputGetSwitched /; $ReturnNumericArray := Identity
encoderOutputGetSwitched /; !$ReturnNumericArray := Normal

(* Ambiguous case for encoders that accept lists. we can't tell if a 
   list input is batched or not. first we assume it is unbatched, so
   we wrap an extra list over it, if that fails, we'll try again via 
   fallthrough (RuleCondition[Fail]) *)
iEncoderApply[encoder_ ? AcceptsListsQ, input_List] := RuleCondition @ Catch[
	First @ Cached[getEncoderFunction, encoder] @ List @ input, 
	EncodeFail, EncodeFailureToFail
]

(* Batched case + bypass, encoded by either a single NumericArray or a packed array. 
   This is slightly hacky: it should trigger the bypasser if the encoder is 
   bypassable, but that means we can't wrap the NA in a list as above because 
   it will fail the bypass checker. So this codepath is just for bypass purposes, 
   the encoder function should never actually get applied *)
iEncoderApply[encoder_, input_] /; And[
	NumericArrayQ[input] || PackedArrayQ[input],
	ArrayDepth[input] === TRank[CoderType @ encoder] + 1
	] := RuleCondition @ Catch[
		Cached[getEncoderFunction, encoder] @ input,
		EncodeFail, EncodeFailureToFail
	]

(* Ordinary batched case *)
iEncoderApply[encoder_, input_List] /; MatchQ[input, {RepeatedNull[EncoderInputPattern[encoder]]}] := Catch[
	ArrayUnpack @ Cached[getEncoderFunction, encoder] @ input,
	EncodeFail, EncodeFailureToMessage
]

(* Single input (unbatched) case *)
iEncoderApply[encoder_, input_] := Catch[
	First @ Cached[getEncoderFunction, encoder] @ List @ input, 
	EncodeFail, EncodeFailureToMessage
]

EncodeFailureToMessage[EncodeFailure[reason_], _] := (
	Message[NetEncoder::invencin, reason]; 
	$Failed
);

EncodeFailureToFail[_, _] := Fail;


PackageScope["CatchEncodeFailures"]

SetUsage @ "
CatchEncodeFailures[f$] is a wrapper around an encoder function that merely catches \
any calls to EncodeFail that happen when it is applied, returning them as EncodeFailure[$$] \
expressions."

CatchEncodeFailures[f_][in_] := 
	Catch[f @ in, EncodeFail]


PackageScope["PropogateEncodeFailures"]

SetUsage @ "
PropogateEncodeFailures[f$, name$] is a wrapper around an encoder function that catches \
any calls to EncodeFail that happen when it is applied, turning them into ThrowFailure \
calls that issue an apporpriate message and abort the computation."

PropogateEncodeFailures[f_, name_][in_] := 
	Catch[f @ in, EncodeFail, propogateEncodeFailure[name]];

propogateEncodeFailure[name_][failure_, _] := InputErrorHandler[name, Null][failure];


PackageScope["ToEncoderFunction"]

ClearAll[ToEncoderFunction]

SetUsage @ "
ToEncoderFunction[type$, isBatched$] returns a function that encodes inputs of the given type.
* isBatched indicates whether the function should expect a list of inputs or a single input.
* Note that underlying encoder functions always work in batch mode."

ToEncoderFunction[type_] := 
	ToEncoderFunction[type, True]

ToEncoderFunction[TensorT[{___LengthVar, ___Integer}, RealT|AtomT], True|False] :=
	Identity;

ToEncoderFunction[type_, False] := CatchEncodeFailures[
	List /* Cached[getEncoderFunction, type] /* First
]

ToEncoderFunction[type_, True] := 
	CatchEncodeFailures @ Cached[getEncoderFunction, type];

(* %HACK this is a temporary measure until we figure out how to incorporate the various cases
into a coherent type system that also works with TensorT. for now, this is motivated by
ExtractLayer which can take negative or positive integers. *)

getEncoderFunction[t:TensorT[dims_List, IndexIntegerT[max_]]] :=
With[{
	rank = 1 + Length[dims],
	failstr = StringForm["input is not ``", TypeString @ t],
	raggedArrayQ = Function[{in, rank, test}, VectorQ[in, ArrayQ[#, rank-1, test]&] || ArrayQ[in]]
	},
With[{
	listArrayQ = If[DynamicDimsQ @ dims, raggedArrayQ, ArrayQ],
	narrayIntegerCore = Function[{a, rank, test},
		Or[
			VectorQ[a, NumericArrayQ[#] && ArrayDepth[#] == rank - 1 && test[#]&] && Or[
				VectorQ[a, MatchQ[NumericArrayType[#], __~~"Integer"~~__]&],
				With[{n = Normal[a]}, Round[n] == n]
			],
			NumericArrayQ[a] && ArrayDepth[a] == rank && test[a] && Or[
				MatchQ[NumericArrayType[a], __~~"Integer"~~__],
				With[{n = Normal[a]}, Round[n] == n]
			]
		]
	]
	},
With[{
	narrayIntegerQ = Function[{a, rank}, narrayIntegerCore[a, rank, True&]],
	narrayPositiveIntegerQ = Function[{a, rank}, narrayIntegerCore[a, rank, Min[#] > 0&]]
	},
	Switch[max,
		_Integer,
			If[narrayPositiveIntegerQ[#, rank] || listArrayQ[#, rank, PositiveMachineIntegerQ] && Max[#] <= max, #, EncodeFail @ failstr]&,
		Infinity,
			If[narrayPositiveIntegerQ[#, rank] || listArrayQ[#, rank, PositiveMachineIntegerQ], #, EncodeFail @ failstr]&,
		_,
			If[narrayIntegerQ[#, rank] || listArrayQ[#, rank, MachineIntegerQ], #, EncodeFail @ failstr]&
	]
]]];

(* TODO: remove this in favor of having all encoders support an InputDepth parameter.
alternatively, make InputDepth a feature of NetEncoder itself and have it uniformly supported via this mechanism.
third way would be to have a super-encoder that wraps other encoders with an input depth *)
getEncoderFunction[type:TensorT[{n_}, enc_NetEncoder]] := 
	MakeBypasser[
		BatchSeqRecursiveEncoder[
			Cached[getEncoderFunction, enc] /* toArray,
			If[IntegerQ[n], n, Automatic]
		],
		StripCoders @ type
	]

toArray[list:{__NumericArray}] := ArrayCatenate[list];
toArray[list_List] := toNumericArray[list];
toArray[na_NumericArray] := na;
_toArray := $Unreachable;

getEncoderFunction[e: HoldPattern @ NetEncoder[type_, assoc_Association, otype_]] := Scope[
	If[assoc["$Version"] === Indeterminate, legacyCoderFail[e]];
	func = $EncoderData[type, "ToEncoderFunction"][assoc, otype];
	If[TrueQ @ $EncoderData[type, "AllowBypass"][assoc],
		func = MakeBypasser[func, otype]
	];
	func
]

PackageScope["EncodeFailure"]

SetUsage @ "
EncodeFailure['reason$'] represents the failure of the encoder on an input. 
* EncodeFailure[$$] is produced by encoder functions if they are in 'return' mode.
* EncodeFailure[$$] is intercepted by InputErrorHandler and used to report the failure \
in the context of the a particular port and batch permutation."



PackageScope["EncodeFail"]

SetUsage @ "
EncodeFail['msg$', args$$] should be called by encoder implementations when they encounter a failure."

General::invencin = "Invalid input, ``.";

EncodeFail[msg_String, args__] := EncodeFail[StringForm[msg, args]];
EncodeFail[msg_] := Throw[EncodeFailure @ fromStringForm @ msg, EncodeFail];


PackageScope["BatchSeqRecursiveEncoder"]

BatchSeqRecursiveEncoder[f_, n_Integer][in_] :=
	If[in === {} || !ListOfListsQ[in] || !MatchQ[arrayDimensions[in, 2], {_, n}],
		EncodeFail["input was not a sequence of length ``", n],
		Map[f, in]
	];

BatchSeqRecursiveEncoder[f_, Automatic][in_] :=
	If[in === {} || !ListOfListsQ[in] || (Min[Length /@ in] === 0),
		EncodeFail["input was not a non-empty sequence"],
		Map[f, in]
	];


PackageScope["MakeBypasser"]

MakeBypasser[encoderf_, otype_] := Scope[
	dims = TDimensions[otype];
	itype = TType[otype];
	If[dims === {},
		test = And[MachineArrayQ[#], arrayDepth[#] == 1]&;
	,
		With[dimsp = Prepend[dims, _] /. _LengthVar -> _,
		test = And[
			NumericArrayQ[#] || PackedArrayQ[#] || 
				VectorQ[#, NumericArrayQ] || VectorQ[#, PackedArrayQ], 
				MatchQ[arrayDimensions[#], dimsp]
			]&
		];
	];
	If[MatchQ[itype, IndexIntegerT[_?IntegerQ|Infinity|All]],
		With[max = Replace[First @ itype, SizeT -> Infinity],
		test = Insert[test, Unevaluated[checkMinMax[{1, max}, arrayMinMax[#]]], {1, -1}]]
	];
	Bypasser[encoderf, test]
]

checkMinMax[{a1_, b1_}, {a2_, b2_}] := a2 >= a1 && b2 <= b1;


PackageScope["Bypasser"]

Bypasser[func_, test_][data_] := If[TrueQ @ test[data], data, func[data]]
