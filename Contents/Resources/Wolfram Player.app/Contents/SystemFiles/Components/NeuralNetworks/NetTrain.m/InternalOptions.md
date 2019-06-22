# NetTrain Internal Options

You can provide internal options to `NetTrain` using the `"InternalOptions" -> {"key" -> value, "key" -> value, ...}`. 

The following tables contains the allowed values of "InternalOptions" and what they do:

| Option | Meaning |
| ----- | ----- |
| SequenceBucketingPartitions | trades off efficiency of bucketing against bias in the gradients |
| UpdatesPerBatch | calculate multiple forward+backward passes per data batch |
| DisablePreEncoding | disables pre-encoding of inputs that have associated `NetEncoder`s |
| ForceSynchronization | whether to call NDArrayWaitForAll before obtaining losses |
| DisableErrorRate | disable reporting of error rate for CELoss |
| SelectLowestValidationError | whether to use error rate for picking best net when cross-validating (versus just loss) | 
| PlotLossQuartiles | whether to show shaded region for loss distribution |
| LowPrecisionGradientRescaling | multiplier to apply to loss for 16-bit training| 
| WorkingPrecision | "Real64", "Real32" or "Real16" (experimental) |

## Reporting options

These options turn on new fields. When using panel-reporting (the default when you have a front-end), you will see new entries in the panel. For print-reporting (the default when you are running a command-line kernel), you will see new columns in the printed table.

| Option | Meaning |
| ----- | ----- |
| ReportTimings | whether to show timings of various phases of training loop |
| ReportMemoryUsage | whether to show memory used by system, kernel, and GPUs |
| MemoryReportingInterval | how frequently to update memory estimates (2 seconds default) |

The timings that are reported include the following:

| Name | Meaning |
| ----- | ----- |
| batch | time taken for entire batch |
| sync | time waiting for MXNet to compute loss |
| gen | time for generator to produce batch of data |
| exec | time to send data to MXNet |
| coll | time to collect stats (including loss, therefore sync) |
| rep | time to do reporting/update loss plots |
| calls | time for user callbacks |

Three times are shown for e.g. of these, which are the most recent value, and the minimum and maximum ever encountered. Means are not reported currently.

These timings are also available from callback functions (e.g. TrainingProgressFunction) via the key `"$InternalTimings"` (the keys have more verbose names, obviously).

## Logging options

The following internal options can be be used to log stuff. 

| Option | Things logged | 
| ----- | ----- | ----- |
| PlanLogging | calls to `ToNetPlan`: logs the JSON string (compressed) | 
| ExecutorLogging | `NetExecutorCreate`: logs the batched dims, device + precision, and max memory <br> `NetExecutorInherit`: logs the batched dims, device + precision <br> `NetExecutorReshape`: logs the target dims, ddevice + precision <br> `NetExecutorReshapeBatchsize`: logs the target batchsize, device + precision |
| BucketingLogging | `NewReshapedBucket`: logs the bucket key and template value used to create a fast bucket via reshaping <br> `NewMasterBucket`, `NewBucket`: logs the bucket key to make the [initial master] bucket <br> `ClearBuckets`: logs the clearing of all previous buckets along with the new and old max bucket keys |
| MXNetLogging | all calls to LibraryLink functions defined in MXNetLink are logged (highly verbose) |
| MemoryUsageLogging | periodically logs the current memory usage, see MemoryUsageReporting below |
| TimingLogging | logs timings of different stages of training loop |
| PermutationLogging | `SeqLenQuartiles`, `SeqLenMinMeanMax`, `SeqLensCompressed`: logs info on the sequence lengths (one list per LengthVar) <br> `PermutationCompressed`: logs the permutation of the training examples before batching |
| GradientLogging | logs an association containing magnitudes of gradients as well as magnitudes of actual updates made, one entry per weight array | 
| GradientNormFunction | how to compute the magnitude of gradients+updates for above; default is L-infinity norm | 
| VerboseLogging | applies the given setting to Plan, Executor, Bucketing, MemoryUsage, Permutation, and Timing loggers |

The logging options all use the same syntax, e.g. `"PlanLogging" -> File[..]`. The following values can be given as the RHS:

| Setting | Behavior |
| ----- | ----- |
| `False` | log is disabled (default) |
| `True` | just `Print` the log entries as they happen |
| `Bag[..]` | append the log entries to the bag |
| `File[..]` | print the entries to a `"*.wl"` file using `PutAppend` |
| `OutputStream[..]` | write the log entries to the output stream |
| `"StandardError"` | write the log entries to standard error of command-line kernel

All entries are written/stored as `{time, abs batch} -> value`, where time is the seconds since the kernel started and abs batch is the absolute batch number, e.g. how many batches have been processed since training began.



