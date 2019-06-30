# Evaluation 

This file mainly describes how evauation of nets is done, rather than training, though there is some overlap in this functionality at the level of `NetExecutor`.

## Phases of evaluation

Evaluation SubValues are attached to all layer heads (and NetChain, NetGraph) via SetupGenericDispatch. These ensure that e.g. `NetChain[...][input, (opts)]` will call `NetApply[net, input, (opts)]`, which does the following things:

1. checks the number of inputs the net expects, compares to the number of inputs given by user
2. canonicalizes user inputs into a list of values in the right order
3. decides if the input is batched or not, and what the batch length is
4. looks up an *evaluator* for the combination of net, target device, and batchiness
5. feeds the inputs to the evaluator and returns the result

## Evaluator

The *evaluator* is created and cached against the net. Its job is to be applied to the list of inputs and produce a result. Evaluators are created with `ToNetEvaluator` in `ToEvaluator.m`.

The first step of creating an evaluator is creating a `NetPlan`. The plan represents the mapping between the WL net and its realization as a concrete MXNet graph (technically a MXSymbol). This plan is agnostic to batch size (although it can change based on device). The plan is cached against the net in an expression table.

Several kinds of evaluator can be produced depending on batchsize (singleton vs batched) and net type (bucketed vs not bucketed). 

The evaluator is a function that applies encoders as appropriate, calls `NetExecutorForward` (and optionally `NetExecutorBackward` if gradients are required), and extracts the outputs from the net. (This final step has depends on which property is being computed, see the Properties section below).

The evaluator contains a `NetExecutor`, which encapsulates the underlying MXNet objects. For bucketed evaluators, this executor is a bucketed executor, which abstracts over multiple concrete executors. 

Batched bucketed evaluators will also try to re-order their inputs to make the batches efficient. 

Batched evaluators will check if the number of inputs is greater than the underlying executors batch size, and if so, split the computation up into batches and collect them in bags. 

## Executors

NetExecutors manage the underlying MXNet structures that compute nets. They ultimately represents the binding between the logical, high-level WL representation of nets, which has a hierarchical structure based on containers and layers and then individual arrays, as well as concepts such as dynamic-length arrays, and the MXNet representation of these concepts. 

Other than creating fresh executors, it is necessary to resize existing executors when the desired batch size is different to that initially requested (e.g. for the final batch when the length of the input data doesn't divide the chosen batch size). It is also extremely important to use fast 'reshaping' when dealing with dynamic dimensions (this is possible for certain nets that do not require unrolling), because it makes it very fast to derive executors for new bucket sizes.

The low-level interface with MXNetLink is contained in `NetExecutor.m`, this is also where the reshaping code lives. This code has been evolved from some now-dead code in MXNetLink that also wraps the underlying executor handles. That code can be eliminated or at least simplified to drop all the features it acquired before the functionality migrated into the NN paclet.

A higher-level construction function that takes `NetPlans` and turns them into executors is `ToNetExecutor`, and lives in `ToExecutor.m`. This file also contains the abstraction that introduces bucketing via `ToBucketedNetExecutor` -- a bucketed executor contains executors and requires an explicit lookup step, which is done by the Trainers and Evaluators as appropriate.

A `MultiExecutor` represents an executor spread over multiple devices, achieved by partitioning the batch. This is intended to have the same API as a normal `Executor` and work transparently. This code lives in `MultiExecutor.m`.

## Properties

When a property is requested during evaluation using the syntax `net[data, prop]`, that property must be parsed and turned into code that actually computes that property. This affects the evaluator in two ways: 1) it determines what outputs and/or gradients of the net need be computed, which will require a different executor 2) it introduces different code to compute the final result. 

This functionality is encapsulated by a function called `ParseNetProperty`, which takes a property spec and returns an `OutputDescriptor` object that contains the code needed to compute the property given a dynamically-scoped variable called `$ExecutorArrays` that should contain the arrays. `ParseNetProperty` also returns info on what outputs etc the executor should compute.

This code lives in `IO.m`.

## Fast Path

There is a special subvalue that will run if a layer or net is eligible (namely has a single input port). This will bypass the normal machinery in NetApply. Its goal is to allow simple invocations of a net to avoid the overhead associated with support for more general cases.

The special subvalue only matches net applications of one argument, and will redirect to the function `NetApplyFast`. `NetApplyFast` will cache the *FastPathFunction* in an ExpressionTable (used so that when the net is garbage collected the function is too). 

The body of the FastPathFunction has a simple check in it that verifies the input is of the right type (e.g. image, audio etc) and then looks up the unbatched executor and feeds it the input. If the check fails the fast path function will instead call the slower `NetApply` function as usual.

## Batch Size

Unless the option `BatchSize -> n` option is provided when evaluation a net on a batch of inputs, we must choose a batch size automatically.

This is done in `BatchSize.m`. The current strategy is cobbled together from the information we have available to us, see the file for more detailed comments. 

Essentially, for bucketed nets, we always pick a fixed value, because memory usage is an unpredictable function of typical sequence length, which we don't know in advance. For un-bucketed nets, we can be a little smarter, and pick something that is likely to fit in the available device memory and is also large enough to take advantage of economies of scale. 

## Caching

This recaps where caching happens for the purposes of evaluation:

| object | created by | cached by creator | called by | cached against |
| --- | --- | --- | --- | --- |
| evaluator | `ToNetEvaluator` | Y | `NetApply` | batchiness, device, evaluation mode, property spec |
| `NetPlan` | `ToNetPlan` | Y | `ToNetEvaluator` | as above |
| batch size | `ChooseEvaluationBatchSize` | Y | `ToNetEvaluator` | plan, device, precision | 
| available mem | `GetGPUInformation` | Y | `ChooseEvaluationBatchSize` | device |
| constraints | `NetConstraintData` | N | `ToBucketedNetExecutor` | net | 
| reshaped executor | `NetExecutorReshape` | N | batch evaluator | executor, batch size |
| `NDArray` | - | - | `ToNetExecutor` | `NumericArray` |

Note that `ToNetEvaluator` cache is a singleton cache: it has a size of 1.
