# Writers

This section gives details on the API that is available to writers, and gives hints on how best to write and debug writers.

## What a Writer does

A Writer is a function defined in a layer definition file (see `Neuralnetworks/Layers/Layers.md`). It is a function that actually implements a layer in terms of MXNet primitives.

Writers work by obtaining *MXNodes* that represent vertices of the computation graph and applying functions to them to produce new MXNodes. There are specific "high-level" functions like `SowPlus`, `SowMinus`, `SowFC`, etc. that operate on MXNodes, which should be used when possible, but you can also use `SowNode` when such a function is not available. Run ``?MX`*`` of the operators available in MXNet. Note that the operator names returned by ``?MX`*`` can differ slightly from the actual names in MXNet. For example, `batch$dot` is actually `batch_dot`. As a result, you might need to look at the MXNet documentation to figure out the correct name to pass to `SowNode`.

In addition to MXNodes, there are also special objects called *MetaNodes* that are also available. These offer an abstraction on top of ordinary MXNodes that can be useful when dealing with dynamic dimensions. 

## Basic functions

The following table lists functions that can be used from inside a Writer to access or set the inputs, outputs, and states of the layer being written:

| Function | Purpose | 
| ---- | ---- |
| GetInput["name"] | obtain the MXNode corresponding to the input of the current layer |
| GetInput["name", "kind"] | obtain a particular representation of MXNode ("Batchwise", "Timewise") |
| GetInputMetaNode["name"] | obtain an MetaNode |
| Get{In,Out}putDims["name"]| get the dimensions of the given input or output |
| Get{In,Out}putRank["name"] | get the rank of the given input or output |
| GetState["name"] | get the initial value of a recurrent state |
| SetOutput["name", node] | set the output of the layer to be an MXNode or MetaNode |
| SetState["name", node] | set the final value of a recurrent state |

All of these functions are documented, so you cna use e.g. `?GetInput` to see more information about one of them.

If it turns out that the full writer is not necessary for a specific setting of the parameters of a layer, and a simple MXNet translation is possible, the function `MXWriteDefaultAndReturn` can be called. `SowCurrentNode[inputs]` can also be called if the inputs need to be first preprocessed and then the simple translation used.

## NetPlanPlot

`NetPlanPlot` allows one to take a net and plot a graph of the MX computation graph that implements it. 

Write `NetPlanPlot[net]` to see a plot of the MX operations that implement a net. Any dynamic dimensions that are present will use a value of 2, though you can specify other values (see `?NetPlanPlot`).

There are several kinds of bugs that can occur in a Writer. The most basic case is that your Writer fails to produce valid nodes, e.g. because of a typo. This will cause an internal failure. The `LIF` shortcut should usually make it obvious where the problem occurred (if you haven't already, define ``LIF := Internal`$LastInternalFailure`` in your `init.m`).

If your Writer *does* produce a valid graph (technically an `MXSymbol`), it can still have type issues. For example, you might apply an operation to the wrong rank or dimensions of tensor, which will generate an MXNet error when InferShape is called. `NetPlanPlot` will still show the graph in this case, but it will have a large red error message that describes the error and locates the particular node where the error occurred. 

If all goes well, however, you will be able to see the graph along with the dimensions of all arrays that are present on the edges. Mousing over a node will show the parameters that were specified when that node was Sown, and mousing over an edge will show its dimensions. This is an invaluable tool for understanding existing and new layers.

If there are are foreach operators in your graph, their contents will be displayed next to the main graph.

It should happen rarely (becuase unrolling is not the norm), but you can use the option "EdgeBundling" -> False if an unrolled graph is hard to read.

## ToDebugPlan

`ToDebugPlan` gives on an easy way of getting a full `NetPlan` object for a net while providing the arguments as options that default to normal values. `ToNetPlan` is not as user friendly to use because it takes all those arguments as a giant tuple (as this is needed for caching).

## The IntermediateArrays property

This is a hidden property that is available for any net. It allows you to "X-ray" a layer of a normal net in-vitro, and see the live values that are passing through that layer:

```
net = NetInitialize @ NetChain[{2, ElementwiseLayer[(# + 1)/2 &], 3}, "Input" -> 2];
net[{1, 2}, NetPort[{2}, "IntermediateArrays"]]
```

Here, {2} refers to the second layer. This property will return an association between the named nodes that make up the second layer, and their actual values for this evaluation. A final "Graph" key shows a labeled graph to help you interpret the named nodes and how they connect up.

## MetaNodes

To first approximation, MetaNodes are used to handle dynamic dimensions. They represent an "logical" tensor node in the graph, e.g. a node that may have a physical representation that is different from its logical representation. For example, while all arrays conceptually have the batch dimension first, RNNs often operate on a transposed version of this in which the time dimension is first and the batch dimension second. MetaNodes abstract this detail: they have accessors that allow you to obtain either representation, where the representation is computed on-demand and then cached. Additionally, MetaNodes carry with them information about the dynamic dimension (if any), for example, they carry the node that contains the vector of lengths of the dynamic dimension for the batch.

MetaNodes have their own API. See `?MetaNode` and `?SowMeta*`` for more information.

## Shape Polymorphism

Shape polymorphism is the main property that guarentees low-latency operation. If a single layer is not shape poylmorphic, the entire net will have to be unrolled for each size of input array. See the `Layers/ShapePolymorphism.md` file for more info.