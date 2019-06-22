Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["ListGenerator"]

SetUsage @ "
ListGenerator[list$, batchSize$, permutation$] creates an \
generator that yields batches from a list, each of size batchSize$, \
where the order is given by permutation$, which should be a list of \
integers whose length is divided by batchSize$."

ListGenerator[list_, batchSize_, perm_] := 
	EncodedGenerator[list, Identity, batchSize, perm]

(******************************************************************************)

PackageExport["EncodedGenerator"]

SetUsage @ "
EncodedGenerator[list$, encoder$, batchSize$] creates a generator that \
yields batches from a list, applying encoder$ to each batch. If batchSize$ \
does not divide list$, the first batch will be padded at the beginning.
EncodedGenerator[list$, encoder$, batchSize$, permutation$] creates \
a generator that applies encoder$ to batches from list$ under the \
given permutation$."

EncodedGenerator[list_, encoder_, batchSize_, perm_] := ModuleScope[
	permutation = normalizePerm[perm, list, batchSize];
	source = list; 
	Function[n,
		encoder @ Part[source, Part[permutation, n]]
	]
]

(******************************************************************************)

PackageExport["PrecomputedEncodedGenerator"]

SetUsage @ "
PrecomputedEncodedGenerator[array$, enc$, batchSize$, perm$, settings$] is like \
EncodedGenerator but will decide whether to pre-encode the array into a \
CPU-hosted NDArray based on memory and time constraints.
* The settings should be given in an association, and can include the following keys:
| 'Cache' | either None or a cache structure |
| 'MaxMemory' | if precomputation will consume more than this, don't precompute |
| 'MaxTime' | as above |
| 'Description' | string to use in progress reporting |
| 'ReportProgress' | form of progress reporting to use |
If not None, 'Cache' should be set to Hold[assoc$, list$], where assoc$ is an association \
acting as the cache, and list$ will have the keys appended to it."

PackageExport["$DefaultPrecomputationSettings"]

$DefaultPrecomputationSettings = <|
	"MaxMemory" -> $SystemMemory/8,
	"MaxTime" -> Infinity,
	"ReportingForm" -> Automatic,
	"PortName" -> "Input",
	"ReportingText" -> <|
		"Static" -> "Processing data for \"`port`\" port", 
		"Details" -> "input `input` of `total`"
	|>
|>

PrecomputedEncodedGenerator::fail = "Generator failed on element ``."
GenFail[i_] := Message[PrecomputedEncodedGenerator::fail, i];

PrecomputedEncodedGenerator[list_, enc_, batchSize_, perm_, settings_] := 
	PrecomputedEncodedGenerator[list, enc, batchSize, perm, settings, GenFail];

PrecomputedEncodedGenerator[list_, enc_, batchSize_, perm_, settings_, failf_] /; Lookup[settings, "Cache", None] =!= None :=
	Replace[
		settings["Cache"], 
		Hold[cacheSymbol_Symbol, seenSymbol_Symbol] :> Block[{hash},
			hash = BitXor[Hash[list], Hash[perm], funcHash[enc], Hash[batchSize], $DefaultDataTypeCode];
			AppendTo[seenSymbol, hash];
			CacheTo[cacheSymbol, hash,
				PrecomputedEncodedGenerator[list, enc, batchSize, perm, KeyDrop[settings, "Cache"], failf]
			]
		]
	]

(* deal with some issues that prevent reproducible hashes *)
funcHash[encf_] := Hash[encf /. {
	d_Dispatch :> RuleCondition @ Hash @ Normal @ d, 
	sym_Symbol ? System`Private`HasImmediateValueQ /; MemberQ[Attributes[sym], Temporary] :> RuleCondition @ Hash @ sym
}];

PrecomputedEncodedGenerator[list_, enc_, batchSize_, perm_, settings_, failf_] := ModuleScope[
	UnpackAssociation[settings, maxMemory, maxTime, description, 
		portName, reportingForm, reportingText, 
		"DefaultFunction" -> $DefaultPrecomputationSettings];
	permutation = normalizePerm[perm, list, batchSize];		
	gen = EncodedGenerator[list, enc, batchSize, permutation];
	{time, batch} = AbsoluteTiming[gen[1]];
	batches = Length[list] / batchSize; 
	time *= batches;
	(* if we would exceed the mem limit, don't precompute *)
	If[(ByteCount[batch] * batches > maxMemory) || (time > maxTime),
		(* but if the first batch was expensive to compute, 
		reuse it in the returned generator *)
		Return @ If[time < 0.01, gen, 
			Function[n, If[n == 1, batch, gen[n]]]
		]
	];
	ComputeWithProgress[
		makePrecomputedGenerator[gen, batch, Length[permutation], #, failf]&,
		reportingText,
		"ReportingForm" -> reportingForm,
		"TimeEstimate" -> time,
		"PrintEllipses" -> True,
		"StaticData" -> <|"port" -> portName, "total" -> Length[list]|>
	]
]

makePrecomputedGenerator[generator_, firstBatch_, numBatches_, progressCallback_, failf_] := Scope[
	dims = arrayDimensions[firstBatch];
	batchSize = First[dims];
	PrependTo[dims, numBatches];
	tempArray = CreateConstantNumericArray[dims, 0];
	tryCopyArrayToArray[firstBatch, tempArray, 1, failf];
	callbackInfo = <|"progress" :> N[i/numBatches], "input" :> (i * batchSize)|>;
	Do[
		progressCallback[callbackInfo];
		tryCopyArrayToArray[generator[i], tempArray, i, failf],
		{i, 2, numBatches}
	];
	ArraySliceIterator[tempArray]
]

tryCopyArrayToArray[src_, dst_, i_, failf_] := 
	If[copyArrayToArray[src, dst, i] =!= Null, failf[src, i]];

copyArrayToArray[from_, to_, i_] :=
	mxlCopyArrayToArray[from, to, 0, i, False];

copyArrayToArray[from_ /; VectorQ[from, NumericArrayQ], to_, i_] :=
	mxlCopyArrayToArray[joinArrays @ from, to, 0, i, False];

joinArrays[arrays_] := ArrayReshape[Join @@ arrays, Prepend[arrayDimensions @ First @ arrays, Length[arrays]]];


(******************************************************************************)

PackageExport["ArraySliceIterator"]

ArraySliceIterator[source_] := ModuleScope[
	Assert[PackedOrNumericArrayQ[source]];
	dims = arrayDimensions[source];
	buffer = CreateConstantNumericArray[Rest @ dims, 0.0];
	Function[
		mxlCopyArrayToArray[source, buffer, #, 0, False];
		buffer
	]
]

(******************************************************************************)

PackageExport["NumericArrayGenerator"]

SetUsage @ "
NumericArrayGenerator[array$, batchSize$, permutation$] is like ListGenerator but 
is optimized assuming array$ is a packed or numeric array, or a list of these. 
The batches are actually copied into a once-allocated numeric array which is 
continually overwritten."

NumericArrayGenerator[array_, batchSize_, perm_] := ModuleScope[
	dims = arrayDimensions[array];
	dims[[1]] = batchSize;
	buffer = CreateConstantNumericArray[dims, 0.];
	permutation = normalizePerm[perm, array, batchSize]; 
	source = If[(Times @@ dims) < 1*^6, Developer`ToPackedArray @ array, array];
	(* ^ we don't mind making a copy if there aren't too many elements *)
	If[PackedOrNumericArrayQ[source],
		Function[n,
			(*mxlCall[mxlCopyArrayToArrayPermuted, source, buffer, permutation[[n]]];*)
			mxlCopyArrayToArrayPermuted[source, buffer, permutation[[n]]];
			buffer
		],
		Function[n,
			copyListOfArraysToArray[Part[source, permutation[[n]]], buffer];
			buffer
		]
	]
]

copyListOfArraysToArray[arrays_, target_] :=
	Do[
		mxlCopyArrayToArray[arrays[[i]], target, 0, i, False],
		{i, Length[arrays]}
	]

(* this lets user pass None, and we'll create an identity permutation.
mostly just for debugging *)

normalizePerm[list_, _, _] := 
	list

normalizePerm[None, length_, batchSize_] := 
	MakeBatchPermutation[Length @ length, batchSize]

