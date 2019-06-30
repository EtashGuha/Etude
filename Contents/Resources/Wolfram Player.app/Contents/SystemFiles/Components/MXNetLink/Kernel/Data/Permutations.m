Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(* TODO: DON'T WE NEED TO ADD THE PADDING AT THE BEGINNING, RATHER THAN THE END???? 
E.G> LOOK IN TRAINER DO VALIDATION, WHERE IT DROPS THE EXCESS FROM THE BEGINNING *)

PackageExport["MakeBatchPermutation"]

MakeBatchPermutation[n_, batchSize_] := 
	Partition[
		makeRangePermutation[n, batchSize], 
		batchSize
	]

(******************************************************************************)

PackageExport["MakeRandomizedBatchPermutation"]

SetUsage @ "
MakeRandomizedBatchPermutation[n$, batchSize$] creates a random \
permutation array, an array of integers taken from [1,n]. The length \
of this array is Ceiling[n$, batchSize$]. If batchSize$ divides n$, \
the permutation will sample each integer once, otherwise, the extra \
padding will be filled with random integers <= n."

MakeRandomizedBatchPermutation[n_, batchSize_] := 
	Partition[
		RandomSample @ makeRangePermutation[n, batchSize], 
		batchSize
	]

makeRangePermutation[n_, batchSize_] := Scope[
	padn = Ceiling[n, batchSize];
	excess = padn - n;
	range = Range[1 - excess, padn - excess];
	If[excess > 0, range[[ ;; excess]] = RandomInteger[{1,n}, excess]];
	range
]

(******************************************************************************)

PackageExport["MakeBucketedRandomizedBatchPermutation"]

SetUsage @ "
MakeBucketedRandomizedBatchPermutation[buckets$, batchSize$] creates a random \
permutation array as in MakeRandomBatchPermutation where n$ is taken as \
Length[buckets$]. However, the result has the property that the individual \
batches, which are successive spans of length batchSize$, are internally \
sorted with respect to buckets$. In other words, each subpermutation perm$i \
has the property that buckets$[[perm$i]] is sorted. This is used when \
dealing with sequence inputs and ensures that a batch consists of \
similar-length sequences and those sequences are in sorted order, which \
is needed for the cuDNN RNN implementation. The return value consists of \
{permutation$, buckets}, where buckets$ are the maximum buckets per-batch.
MakeBucketedRandomizedBatchPermutation[buckets$, batchSize$, nbins$] specifies \
that the buckets should be put into nbins$ classes before being sorted, \
and have sorting occur between classes, which maintains more diversity then \
if the buckets are directly sorted. 
* The buckets are a tuple of n$ identical-length lists, where n$ is the number \
of independent sequences.
* nbins$ = 1 essentially prevents sorting and allows for true SGD, but is slow \
because the average bucket tends to the max seq length, and so bucketing becomes \
pointless.
* nbins$ = None fully sorts buckets and is fastest, but increases the variance \
of the gradient estimate as elements in the batch can be strongly correlated \
with each-other through their sequence lengths.
"

MakeBucketedRandomizedBatchPermutation[ubuckets_, batchSize_, nbins_:None] := Scope[
	buckets = Transpose[ubuckets];
	n = Length[buckets];
	perm = Flatten @ MakeRandomizedBatchPermutation[n, batchSize];
	(* make an initial permutation to ensure that the subsequent sorting breaks ties randomly *)
	sortOrder = Ordering[buckets[[perm]]];
	If[IntegerQ[nbins] && n > 1,
		sortOrder = Ordering[sortOrder];
		sortOrder = Ceiling[sortOrder, (n-1.) / nbins]; 
		(* the n-1 ensures the biggest bucket always comes last *)
		sortOrder  = Ordering[sortOrder];
		(* if specified, bin the buckets before sorting, so that they are more softly randomized *)
	];
	order = Partition[sortOrder, batchSize];
	(* create a sort order by bucket, and batchify the order so we can randomize the batches *)
	batchorder = RandomSample[Range[Length[perm] / batchSize]];
	(* randomize the order of the batches, each of which is sorted internally *)
	maxPos = First @ Ordering[batchorder, -1];
	batchorder[[{1,maxPos}]] = batchorder[[{maxPos, 1}]];
	(* ensure the biggest batch comes first (for priming purposes) *)
	perm = perm[[Join @@ order[[batchorder]]]];
	(* join them back up again *)
	toPermAndBuckets[perm, buckets, ubuckets, batchSize]
	(* also create the per-batch maxima *)
];

(******************************************************************************)

PackageExport["MakeBucketedBatchPermutation"]

SetUsage @ "
MakeBucketedBatchPermutation[buckets$, batchSize$] returns a \
permutation that sorts the buckets.
* The buckets are a tuple of n$ identical-length lists, where n$ 
is the number of independent sequences.
* If n$ does not divide batchSize$, the permutation will be padded \
on the beginning with the index of the smallest bucket$.
* A list of the per-batch buckets are also returned.
"

(* we pad with the first element to ensure that the shortest sequences
are the ones that are padded, rather than the longest *)

MakeBucketedBatchPermutation[ubuckets_, batchSize_] := Scope[
	buckets = Transpose[ubuckets];
	perm = Ordering[buckets];
	{bmax, excess} = BatchCountExcess[Length[buckets], batchSize];
	If[excess =!= 0,
		perm = Join[ConstantArray[First[perm], excess], perm];
	];
	toPermAndBuckets[perm, buckets, ubuckets, batchSize]
];

(******************************************************************************)

PackageExport["BatchCountExcess"]

BatchCountExcess[n_, batchSize_] := Scope[
	padn = Ceiling[n, batchSize];
	{padn / batchSize, padn - n}
];

toPermAndBuckets[perm_, buckets_, ubuckets_, batchSize_] := 
	Scope @ List[
		permPart = Partition[perm, batchSize],
		Transpose @ Map[partOperator[permPart], ubuckets]
	]

partOperator[perm_][list_] := Part[list, #]& /@ perm;
