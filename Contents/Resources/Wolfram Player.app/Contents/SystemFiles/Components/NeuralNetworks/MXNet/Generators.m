Package["NeuralNetworks`"]



PackageExport["$SequenceBucketingPartitions"]


PackageScope["ConstructTrainingGenerator"]

SetUsage @ "
ConstructTrainingGenerator[<|name$1 -> data$1,$$|>,<|name$1 -> type$1,$$|>,batchsize$] creates a function that produces batches.
* The function maps n$ \[Function] n$th batch.
* The underlying data$i should all be of the same length.
* Each returned batch is association from the name$i to the batch values.
* The order of data is randomized.
* The function is built out of various type-specific iterators from MXNetLink`Generators`.
* If any of the type$i are variable length sequences, the batches will be grouped by length."

ConstructTrainingGenerator[data_, types_, batchsize_] := 
	constructGroupGenerator[
		data, types, batchsize, 
		MakeBucketedRandomizedBatchPermutation[#1, #2, $SequenceBucketingPartitions]&, 
		MakeRandomizedBatchPermutation,
		"Training example"
	];


PackageScope["ConstructValidationGenerator"]

SetUsage @ "
ConstructValidationGenerator[<|name$1 -> data$1,$$|>,<|name$1 -> type$1,$$|>,batchsize$] creates a function that produces batches.
* The function maps n$ \[Function] n$th batch.
* The underlying data$i should all be of the same length.
* Each returned batch is association from the name$i to the batch values.
* The function is built out of various type-specific iterators from MXNetLink`Generators`.
* If any of the type$i are variable length sequences, the batches will be grouped by length."

ConstructValidationGenerator[data_, types_, batchsize_] := Scope[
	If[AssociationQ[data],
		n = Length @ First @ data;
		If[n < batchsize, data = Map[padUpTo[batchsize], data]];
	];
	(* ^ handle case where batchsize is larger than validation set size (the excess won't contribute
	to the loss) *)
	constructGroupGenerator[
		data, types, batchsize,
		MakeBucketedBatchPermutation, None&,
		"Validation example"
	]
];

padUpTo[n_][data_List | data_NumericArray] := Join[CTable[First[data], n - Length[data]], data];

constructGroupGenerator[data_, types_, batchsize_, varbatcher_, nonvarbatcher_, name_] := Scope[
	$batchsize = batchsize;
	n = Length @ First @ data;
	names = Keys[types];
	data = Lookup[data, names];
	types = Lookup[types, names];
	$PermutationLogger[StringSplit[name, " "][[1]]];
	If[bucketed = ContainsVarSequenceQ[types],
		$seqpositions = $seqnames = <||>;
		$seqlens = {};
		data = MapThread[preencodeSequences, {names, data, types}];
		{$perm, buckets} = varbatcher[$seqlens, $batchsize];
		$LastGeneratorSequenceStatistics ^= Insert[MinMax[#], N[Mean[#]], 2]& /@ $seqlens;
		If[$PermutationLogger =!= Hold,
			$PermutationLogger[{"SeqLenQuartiles", N[Quartiles[#]]& /@ $seqlens}];
			$PermutationLogger[{"SeqLenMinMeanMax", $LastGeneratorSequenceStatistics}];
			$PermutationLogger[{"SeqLensCompressed", ToCompressedBytes[Sort /@ $seqlens]}];
		];
	,
		$perm = nonvarbatcher[n, $batchsize];
	];
	{batchCount, excess} = BatchCountExcess[n, batchsize];
	$LastGeneratorPermutation ^= $perm;
	$PermutationLogger[{"PermutationCompressed", ToCompressedBytes[$perm]}];
	generator = Association @ MapThread[constructGenerator, {names, data, types}];
	If[$PrecompCache =!= <||>, $PrecompCache ^= KeyTake[$PrecompCache, $seenPrecompHashes]];
	(* ^ if the cache is occupied, flush it of stale entries (where stale means entries not
	from the current net). This prevents the cache growing without bound. *)
	gen = If[bucketed, 
		BucketedGroupGenerator[generator, buckets],
		GroupGenerator[generator]
	];
	WrapGeneratorContext[gen, name, batchsize, $perm]
];


PackageScope["$PermutationLogger"]

PackageScope["WithGeneratorContext"]

PackageScope["$LastGeneratorSequenceStatistics"]
PackageScope["$LastGeneratorPermutation"]
PackageScope["$LastGeneratorIndices"]
PackageScope["$LastGeneratorName"]
PackageScope["$LastGeneratorData"]

$LastGeneratorPermutation = None;
$LastGeneratorIndices = None;
$LastGeneratorName = None;
$LastGeneratorData = None;

(* We must use permanent globals because constraint checking happens outside the generator call *)

WrapGeneratorContext[gen_, name_, batchsize_, perm_] := Module[{perm2 = perm, result = None}, 
	$LastGeneratorPermutation = perm;
	Function[n, 
		$LastGeneratorName = name;
		$LastGeneratorIndices = If[!ListQ[perm2], n, perm2[[n]]];
		$LastGeneratorData = result;
		result = gen[n]
	]
];

(* deal with some issues that prevent reproducible hashes 
TODO: factor this out of MX and NN into GU *)
funcHash[encf_] := Hash[encf /. {
	d_Dispatch :> RuleCondition @ Hash @ Normal @ d, 
	sym_Symbol ? System`Private`HasImmediateValueQ /; MemberQ[Attributes[sym], Temporary] :> RuleCondition @ Hash @ sym
}];

preencodeSequences[name_, list_, enc_NetEncoder ? SequenceCoderQ] := Scope[
	hash = BitXor[Hash[list], Hash[funcHash @ enc]];
	AppendTo[$seenPrecompHashes, hash];
	
	If[KeyExistsQ[$PrecompCache, hash], 
		{encoded, lengths} = $PrecompCache[hash];
		Goto[FoundInCache];
	];
	
	lengthf = $EncoderData[CoderName[enc], "SequenceLengthFunction"];
	If[lengthf =!= None,
		(* if the encoder provides a function to get the lengths directly, do that,
		avoiding an expensive encoding step *)
		lengthf = With[{data = CoderData[enc], func = lengthf}, ToPackedArray[func[data, #]]&];
		lengths = batchApplyEnc[name, lengthf, list];
		encoded = None;
	,
		(* otherwise, try pre-encode everything. if we run out of memory we'll instead 
		drop the encoded arrays and just store the lengths on their own *)
		encf = ToEncoderFunction[enc, True];
		{encoded, lengths} = batchApplyEncWithLen[name, encf, list];
		(* if the results fit in memory, batchApplyEncWithLen will produce Preencoded[...],
		otherwise it will just compute then lengths and return None *)
	];

	$PrecompCache[hash] = {encoded, lengths};

	Label[FoundInCache];

	insertLengths[name, lengths, GetLengthVarID[enc]];
	If[encoded === None, list, Preencoded @ encoded]
];

preencodeSequences[name_, list_, enc:SequenceT[_LengthVar, _NetEncoder]] := (
	insertLengths[name, getSeqLengths @ list, GetLengthVarID[enc]];
	list
);

preencodeSequences[name_, list_, t:TensorT[{LengthVar[id_], rest___}, _]] := Scope[
	dims = Rest @ TDimensions @ t; 
	If[NumericArrayQ[list],
		If[!MatchQ[arrayDimensions @ list, Join[{_, _}, dims]], throwInvalidInDim[name, t]],
		list = Map[checkArray[name, dims, t, #]&, list];
	];
	insertLengths[name, getSeqLengths @ list, id];
	Preencoded[list]
];

insertLengths[name_, lens_, id_] := Scope[
	lastpos = $seqpositions[id];
	If[IntegerQ[lastpos],
		If[lens =!= $seqlens[[lastpos]],
			ThrowFailure["incseqlen", name, $seqnames[id]]
		],
		AppendTo[$seqlens, lens];
		$seqpositions[id] ^= Length[$seqlens];
		$seqnames[id] ^= name;
	];
];

preencodeSequences[name_, list_, t_] := list;

constructGenerator[name_, Preencoded[list_], _] := 
	name -> ListGenerator[list, $batchsize, $perm];

constructGenerator[name_, list_, ScalarT] := 
	name -> NumericArrayGenerator[checkArray[name, {}, ScalarT, list], $batchsize, $perm];

(* scalar encoder doesn't actually need to be applied, since NDArraySet supports an extra singleton dimension,
but we must check its not already preencodd so checkArray doesn't fail *)
notScalarPreencoded[list_, dims_] := 
	Not[(NumericArrayQ[list] || VectorQ[list, NumericArrayQ]) && MatchQ[arrayDimensions[list], {_, Sequence @@ dims, 1}]];

constructGenerator[name_, list_, EncoderP["Scalar"]] /; notScalarPreencoded[list, {}] := 
	name -> NumericArrayGenerator[checkArray[name, {}, ScalarT, list], $batchsize, $perm];

constructGenerator[name_, list_, TensorT[{i_Integer}, EncoderP["Scalar"]]] /; notScalarPreencoded[list, {i}] :=
	name -> NumericArrayGenerator[checkArray[name, {i}, TensorT[{i}, RealT], list], $batchsize, $perm];

(* the general generator constructor *)

constructGenerator[name_, list_, type_] := Scope[
	encf = ToEncoderFunction[type, True];
	If[(NumericArrayQ[list] || VectorQ[list, NumericArrayQ]) && MatchQ[encf, CatchEncodeFailures[Bypasser[_, _]]],
		If[TrueQ @ encf[[1, 2]] @ list, encf = Identity]];
	(* ^ if the entire input is a NumericArray of the right dimensions, it's the bypass case, and we can skip encoding *)
	name -> If[encf === Identity, 
		NumericArrayGenerator[checkArray[name, TDimensions[type], type, list], $batchsize, $perm],
		toEncodedGenerator[type, name, list, encf]
	]
];

Clear[toEncodedGenerator];

(* we must blacklist scalar inputs because they don't play nice with
pre-encoded generators, as the final '1' dimension can't be elided in
that case.
we must also blacklist sequences (i.e. of encoders), as the windowed
NDArray needs to be transposed into the target NDArray, and that isn't
hooked up (yet) *)
toEncodedGenerator[HoldPattern[NetEncoder][_, _, ScalarT] | ScalarT | VarSequenceP[], _, list_, encf_] :=
	EncodedGenerator[list, encf, $batchsize, $perm];


PackageScope["$seenPrecompHashes"]

$encoderPrecomputationSettings = Association[
	"Cache" :> Hold[$PrecompCache, $seenPrecompHashes],
	"MaxMemory" :> $MaxPrecomputationMemory, 
	"MaxTime" :> $MaxPrecomputationTime
];

toEncodedGenerator[t_ /; !SequenceCoderQ[t], name_, list_, encf_] /; TrueQ[$EnablePreEncoding] := Scope[
	res = PrecomputedEncodedGenerator[
		list, encf, $batchsize, $perm, $encoderPrecomputationSettings, 
		PreencodedInputErrorHandler[name, t, #1, $perm[[#2]], list, encf]&
	];
	If[FailureQ[res] || res === $Aborted[], Abort[]];
	res
]

toEncodedGenerator[t_, name_, list_, encf_] := 
	EncodedGenerator[list, encf, $batchsize, $perm];


PackageScope["ClearPrecompCache"]
PackageScope["batchApplyEnc"]
PackageScope["$PrecompCache"]

ClearPrecompCache[] := (
	$PrecompCache = <||>;
	$seenPrecompHashes = {};
);

ClearPrecompCache[];

batchApplyEnc[name_, f_, list_] := Scope[
	len = Length[list];
	If[Length[list] < 256,
		res = f @ list;
		If[Head[res] === EncodeFailure, 
			PreencodedInputErrorHandler[name, Null, res, Range @ len, list, f]];
		Return @ res
	];	
	bag = Bag[];
	withEncoderProgress[
		Do[
			# @ N[n / len];
			res = f @ Take[list, {n, UpTo[n+127]}];
			If[Head[res] === EncodeFailure, 
				PreencodedInputErrorHandler[name, Null, res, Range[n, Min[n+127, len]], list, f]];
			BagPush[bag, Normal @ res, 1],
			{n, 1, len, 128}
		]&,
		name, len
	];
	BagContents[bag]
];

batchApplyEncWithLen[name_, f_, list_] := Scope[
	len = Length[list];
	If[Length[list] < 256,
		res = f @ list;
		If[Head[res] === EncodeFailure, 
			PreencodedInputErrorHandler[name, Null, res, Range @ len, list, f]];
		Return @ {res, getSeqLengths @ res}
	];
	resultBag = If[$EnablePreEncoding, Bag[], Null];
	lengthBag = Bag[];
	size = 0; limit = $encoderPrecomputationSettings["MaxMemory"];
	withEncoderProgress[
		Do[
			#[N[n / len]];
			in = Take[list, {n, UpTo[n+127]}];
			res = f @ in;
			If[Head[res] === EncodeFailure, 
				PreencodedInputErrorHandler[name, Null, res, Range[n, Min[n+127, len]], list, f]];
			size += ByteCount[res];
			If[size > limit, resultBag = Null];
			BagPush[resultBag, Normal @ res, 1];
			BagPush[lengthBag, getSeqLengths @ res, 1];
		,
			{n, 1, len, 128}
		]&,
		name, len
	];
	{
		If[resultBag === Null, None, BagContents[resultBag]],
		BagContents[lengthBag]
	}
]

(* similar code exists in MXNetLink *)
withEncoderProgress[f_, name_, len_] :=
	ComputeWithProgress[f,
		<|
			"Static" -> "Processing data for \"`port`\" port"
			(*, "Details" -> "input `input` of `total`"*) (* jeromel: this detail is more confusing than informative *)
		|>,
		"PrintEllipses" -> True,
		"StaticData" -> <|"port" -> name, "total" -> Length[list]|> (* jeromel: Length[list] here is wrong in some cases *)
	];

PackageScope["$EnablePreEncoding"]
PackageScope["$MaxPrecomputationMemory"]
PackageScope["$MaxPrecomputationTime"]

$EnablePreEncoding = True;
$MaxPrecomputationMemory := ($RealisticSystemMemory / 8);
$MaxPrecomputationTime = 60.0;

numericArrayOrPackedArrayListQ[list_] :=
	NumericArrayQ[list] || VectorQ[list, NumericArrayQ] || PackedArrayQ[list] || VectorQ[list, PackedArrayQ] ||
	VectorQ[list, NumericArrayQ[#] || PackedArrayQ[#]&];

checkArray[name_, dims_, type_, list_] := Scope[
	If[Or[
		!numericArrayOrPackedArrayListQ[list] && !numericArrayOrPackedArrayListQ[list = ToPackedArray @ N @ list] && 
		!(ListQ[list] && numericArrayOrPackedArrayListQ[list = Map[ToPackedArray, list]]),
		Rest[arrayDimensions[list]] =!= dims], 
		findInvalidBatchDims[name, dims, type, list]
	];
	list
];

(* also used by evaluator, so needs to be General *)
General::invindim3 = "Data provided to port \"``\" should be ``, but was ``.";
General::invindim2 = "Data provided to port \"``\" should be ``, but element #`` was ``.";
findInvalidBatchDims[name_, dims_, type_, list_] := Scope[
	tstring = TypeString @ ListT[SizeT, type];
	If[MachineArrayQ[list] || AtomQ[list],
		ThrowFailure["invindim3", name, tstring, describeInvalid @ list]];
	If[ListQ[list] && dims =!= {},
		Do[
			If[!numericArrayOrPackedArrayListQ[ToPackedArray @ list[[i]]],
				ThrowFailure["invindim2", name, tstring, i, describeInvalid @ list[[i]]]],
			{i, Length @ list}
		]
	];
	throwInvalidInDim[name, type]
];

describeInvalid[_] := "a non-numeric value";
describeInvalid[_Association] :=  "an association";
describeInvalid[s_Symbol] := s;
describeInvalid[e_ ? NumberQ] := "a number";
describeInvalid[_String] := "a string";
describeInvalid[e_List | e_NumericArray] := Scope[
	dims = machineArrayDimensions @ e; 
	If[FailureQ[dims], "a list of non-numeric values", TypeString @ TensorT @ dims]
]


(* more exotic group generators below *)

PackageScope["MakeBucketWrapperFunction"]

(* This takes an assoc of name->type and produces a function that can be applied
to a data assoc of name->value and wraps it in a Bucket[...]. This is used so that
the output of custom generators can be fed to bucketed executors, that need the 
Bucket wrapper to decide which individual executor to use to process that input. *)

(* %FACTOR *)
MakeBucketWrapperFunction[inputKeys_, varIDs_] := Scope[
	skip = {};
	conds = Cases[PositionIndex[varIDs], list:{_, rest__} :> (
		AppendTo[skip, {rest}];
		Equal @@ Map[Hold[mapLength @ #]&, list]
	)];
	If[conds === {}, checker = Identity,
		checker = CreateFunction @ {
			Hold[If][And @@ conds, Slot[], 
				Hold[ThrowFailure["invseqleneq"]]] /. 
				i_Integer :> RuleCondition @ Slot @ inputKeys[[i]]
		};
		inputKeys = Delete[inputKeys, Map[List] @ Flatten @ skip];
	];
	bucketer = With[
		{slots = Thread @ Slot[inputKeys]},
		Function @ Bucket[#, Map[
			ReplaceAll[mapLength @ #, 0 :> ThrowFailure["invseqgenlen"]]&,
			slots
		]]
	];
	checker /* bucketer
];
mapLength[l_List] := Map[Length, l];
mapLength[l_NumericArray] /; ArrayDepth[l] >= 2 := With[
	{dimensions = Dimensions[l]},
	Table[dimensions[[2]], First[dimensions]]
];
mapLength[_] := $Unreachable;


General::invseqleneq = "Generator produced sequences of different length that should have the same length."
NetTrain::invseqgenlen = "Generator function produced a sequence of length zero."

PackageScope["CustomGenerator"]
constructGroupGenerator[CustomGenerator[f_, info_, bucketf_], types_, batchsize_, _, _, name_] := Scope[
	validator = ToBatchTypeTest[types, batchsize, False];
	encoders = Map[ToEncoderFunction[#, True]&, types];
	If[MatchQ[Values[encoders], {Identity..}], encoders = None];
	WrapGeneratorContext[
		ToCustomGeneratorFunction[f, info, bucketf, validator, batchsize, encoders],
		name, batchsize, Automatic
	]
];

NetTrain::invgenunsp = "Generator function did not produce data of the correct type.";

ToCustomGeneratorFunction[f_, info_, bucketf_, validator_, batchsize_, encoders_] := Function @ Block[{res},
	checkValid[validator, res = TryPack @ f[info]];
	bucketf @ MapIndexed[encoders[[First @ #2]][#1]&, res]
];

ToCustomGeneratorFunction[f_, info_, bucketf_, validator_, batchsize_, None] := Function @ Block[{res},
	checkValid[validator, res = TryPack @ f[info]];
	bucketf @ res
];

(* ^ technically TryPack will rewrite lists of rules to a rule of lists, but the f that we are 
passed here is actually always an assoc because of checkAndWrapGeneratorOutput in NetTrain.m *)

PackageScope["$LastInvalidGeneratorOutput"]
PackageScope["$LastGeneratorOutputValidator"]

$LastInvalidGeneratorOutput = None;
$LastGeneratorOutputValidator = None;

checkValid[validator_, res_] :=
	If[!TrueQ[validator[res]],
		$LastInvalidGeneratorOutput = res;
		$LastGeneratorOutputValidator = validator;
		ThrowFailure["invgenunsp"]
	];

constructGroupGenerator[HDF5TrainingData[_, len_, names_, colInfo_], types_, batchsize_, _, _, _] := Scope[
	If[ContainsVarSequenceQ[types], 
		ThrowFailure["novseqgen"];
	];
	If[Sort[names] =!= Sort[Keys[types]], 
		NetTrain::invtrainkeys = "Training data should consist of keys ``.";
		ThrowFailure["invtrainkeys", Keys[types]];
	];
	$batchsize = batchsize; $length = len;
	$LastGeneratorPermutation ^= None;
	readers = MapThread[makeHDF5Reader, {names, colInfo, Lookup[types, names]}];
	CreateFunction[
		Hold[AssociationThread][names, readers]
	]
];

innerType[enc_NetEncoder] := CoderType[enc];
innerType[t_] := t;

makeHDF5Reader[name_, {ds_, fs_, Hold[ms_], dims_}, type_] := Scope[
	tdims = TDimensions[innerType @ type];
	NetTrain::invh5net = "Net is not suitable for training on H5 data.";
	If[FailureQ[tdims], ThrowFailure["invh5net"]];
	If[tdims =!= dims, throwInvalidInDim[name, type]];
	rank = Length[dims] + 1;
	batchDims = Prepend[$batchsize] @ dims;
	ms = HDF5Tools`h5screatesimplen[rank, batchDims];
	offset = CTable[0, rank]; 
	offset[[1]] = Hold[Min][(Slot[1]-1) * $batchsize, $length - $batchsize];
	Hold[PreemptProtect] @ Hold[CompoundExpression][
		Hold[HDF5Tools`h5sselecthyperslab][fs, HDF5Tools`H5SSELECTSET, offset, {}, batchDims, {}],
		Hold[HDF5Tools`h5dreadtensorreal][ds, ms, fs, HDF5Tools`H5PDEFAULT]
	]
]

(* also used by evaluator, so needs to be General *)
General::invindim = "Data provided to port \"``\" should be ``.";
throwInvalidInDim[name_, type_] := ThrowFailure["invindim", name, TypeString @ ListT[SizeT, type]];

(******************************************************************************)
PackageExport["GroupGenerator"]

SetUsage @ "
GroupIterater[iters$] yields a generator that yields batches from \
several sub-generators."

GroupGenerator[iters_List] := ApplyThrough[iters];

GroupGenerator[<|key1_ -> gen1_, key2_ -> gen2_|>] := 
	Function[AssociationThread[{key1, key2}, {gen1[#], gen2[#]}]];

GroupGenerator[assoc_Association] := ApplyThrough[assoc];

(******************************************************************************)
PackageScope["BucketedGroupGenerator"]

(* TODO: maybe move this into MXNetLink? *)

SetUsage @ "
BucketedGroupGenerator[<|key$1->gen$1,$$|>,bucketkeys$] yields a generator that yields batches from \
several sub-generators, returning them as Bucket[<|key$1->data$1,$$|>,bucketkey$].
* calling generator$[n$] uses the n$th key from bucketkeys$."

BucketedGroupGenerator[<|key1_ -> gen1_, key2_ -> gen2_|>, ibuckets_] := Module[
	{buckets = ibuckets},
	Function[Bucket[AssociationThread[{key1, key2}, {gen1[#], gen2[#]}], buckets[[#]]]]
];

BucketedGroupGenerator[assoc_Association, ibuckets_] := Module[
	{buckets = ibuckets},
	Function[Bucket[ApplyThrough[assoc, #], buckets[[#]]]]
];


PackageScope["ConstructBucketedGeneratorFactory"]

ConstructBucketedGeneratorFactory[inputs_, batchsize_] := Scope[
	{lenCode, lens, valCode} = BodyFunctionReplaceAll[
		makeInputProcessingCode[makeGen, inputs, True],
		{$TempData -> TempVar, $InputData -> Slot}
	];
	names = Keys @ inputs;
	AppendTo[lenCode, lens];
	body = Hold[
		Eval[lenCode],
		TempVar[buckets] = Eval[lens];
		{TempVar[n], TempVar[excess]} = BatchCountExcess[Length[#1], $batchsize];
		{TempVar[$perm], buckets} ^= MakeBucketedBatchPermutation[buckets, $batchsize];
		List[
			WrapGeneratorContext[BucketedGroupGenerator[
				AssociationThread[Eval @ names, Eval @ valCode], 
				buckets
			], "Input", batchsize, $perm], 
			Extract[buckets, Ordering[buckets, -1]],
			Ordering[Drop[Flatten @ $perm, excess]] + excess,
			n
		]
	];
	ninputs = Length[inputs];
	If[ninputs > 1, PrependTo[body, Hold[doBatchTests] @ Thread @ Slot @ Range @ ninputs]];
	BodyFunctionReplaceAll[CreateFunction[body], $batchsize -> batchsize]
]


(* we are no longer calling
checkArray[name, Eval[TDimensions[type]], type, Slot[i]], 
here... can't we check this stuff above somehow? *)

Clear[makeGen];
makeGen[name_, Identity, data_, type_ ? VarSequenceQ, _] :=
	Hold @ ListGenerator[data, $batchsize, $perm]

makeGen[name_, Identity, data_, type_, _] :=
	Hold @ NumericArrayGenerator[data, $batchsize, $perm];

makeGen[name_, encf_, data_, _, _] :=
	Hold @ EncodedGenerator[data, encf, $batchsize, $perm]
