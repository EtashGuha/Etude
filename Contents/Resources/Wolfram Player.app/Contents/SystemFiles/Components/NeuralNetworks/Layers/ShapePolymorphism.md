# Shape Polymorphism

The basic idea behind shape polymorphism is to speed up bucketing.

Bucketing occurs whenever we have dynamic dimensions that can take on any values. To handle this, we create an `NetExecutor` for each bucket. The slowest way of creating a new executor is to create a new `NetPlan` from the original net. This is required when the actual computation graph must be different for each size of input. For example, an unrolled RNN has a different graph if the sequence length is different, and so `ToNetPlan` must be called with different settings for the maximum value of the dynamic dimensions (the `LengthVar`s).

But if we have a shape polymorphic executor (created with any initial value for the dynamic dimensions), we can simply "reshape" it with new values for the dynamic dimensions and ordinary MXNet shape inference is enough to create a new executor that is capable of computing the longer or shorter sequence.

For shape polymorphism to be possible, we must have that the underlying computation graph (for example, as represented by the JSON) is independent of the array sizes, and that specializing it to a specific size via `MXSymbolInferShape` will always succeed. This requires that the values of any dynamic dimensions are not hard-coded into the graph (note that it is the *maximum* value of a dynamic dimension that would be hardcoded, there is always explicit handling of padding).

The most common way that this hardcoding can happen is if the writer looks up the maximum value of a dynamic dimension. This immediately rules out polymorphism, because it implies that some property of the produced plan and subsequent executor depends on the maximum length. For this reason, calling `GetDynamicMaxLength` or `SubstituteDynamicMaxLengths` (or the legacy function `GetDynamicDimensionInfo`) will disable polymorphism (by setting the variable `$ReshapingIsImpossible` which is stored in the resulting plan).

We specially introduced features into MXNet like `broadcast_like`, `reshape_like`, and shape-inference in `arange` to make polymorphism possible in common cases.


