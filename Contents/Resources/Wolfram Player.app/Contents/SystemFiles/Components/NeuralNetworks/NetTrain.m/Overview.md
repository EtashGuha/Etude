# How Training Works

## Pre-encoding

When processing a particular input column, we will detect if it is feasible to perform "pre-encoding", in which we process the column once ahead of time, applying the encoder etc, so that we can then reuse this pre-encoded form again and again during the rounds of training. We only bother to do this under the following circumstances:

1. The max memory taken to store the encoded data will be less than 1/8 of `$SystemMemory` (technically `$RealisticSystemMemory` which takes into account cloud limits).
2. The maximum time to perform the encoding is less than 60 seconds and more than 0.2 seconds.

Then, we cache the results of the encoding in a small cache based on the hash of the input data. This means that if a user tries training again and again on the same data by calling `NetTrain` multiple times, we'll only preencode it once. Every time NetTrain is called, we will remove cache entries that aren't used.

The main encoding loop occurs in `PrecomputedEncodedGenerator` in MXNetLink. 

Pre-encoding can be disabled via `NetTrain[..., "InternalOptions" -> {"DisablePreEncoding" -> True}]`.

## Sequence length measurements

Regardless of whether we do the pre-encoding mentioned above, we *have* to run certian NetEncoders on the full column when the encoder is producing variable-length arrays. This is so that we can store the sequence length (per `LengthVar`) for each input and then compute a sorting permutation for the purposes of bucketing.

This is expensive, so there is an optional API that NetEncoders can supply for computing just the length on an input without having to compute the data itself, which would otherwise be thrown away anyway. This API is provided via the `SequenceLengthFunction` encoder field.

The way sequence length measurement happens is that `preencodeSequences` is mapped over all input columns and detects var-length columns to preencode/measure. If it chooses to preencode them it will wrap the result in `Preencoded[...]` so that the later call to `constructGenerator` will treat the arrays as-is. Measured lengths go into an `$seqlens` association that will be fed to e.g. `MakeBucketedRandomizedBatchPermutation`. 

These results are cached in the same way as ordinary pre-encoding. 

## Bucketing

1. First encoding all inputs, so that we have access to the sequence lengths. We might or might not throw out the actual encoded data, depending on whether it can fit in memory (see above)
2. Sort the inputs by length.
3. Partition the sorted inputs into batches of size `BatchSize`.
4. Create a generator that produces these batches in sequence, wrapped in `Bucket[data, lens]`. There can be multiple lens if there are multiple independent LengthVars (e.g. for translation, there might be Source and Destination inputs of independent lengths).

## BatchSize choosing

For training, the basic strategy is to model the memory requirements of the net as a function of batchsize, and to choose the biggest batchsize (within a sensible range) that doesn't exist a memory target.

For evaluation, we add an additional heuristic that the speedup from large batchsizes is minimal for e.g. big convnets. Capping the max additional memory used by increasing the batch size is a way to avoid choosing massive batch sizes for the kind sof nets that don't benefit from it. 

Variable-length sequences we just always pick a batch size of 16, because we don't know the sequence length statistics yet and so can't do anything more sensible. 

The code comments in `MXNet/BatchSize.m` go into more detail.

## Creating a loss net

TODO

## Interpeting learning rates

TODO

## Training Properties

TODO

## Future changes

TODO