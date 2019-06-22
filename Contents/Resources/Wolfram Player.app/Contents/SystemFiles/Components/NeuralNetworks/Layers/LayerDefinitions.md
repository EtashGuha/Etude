# Layer Definitions

## Adding a new layer

See the `Notes/AddingYourFirstLayer.md` for the basic outline. In this document, read the sections about the Core Fields, the Writer field, and the MXNet field, and the Tests field to get started. 

## Introduction

Layer definitions use a custom format. This document describes the format, the purpose of various fields, and other topics you'll need to get to know to define new layers and maintain existing layers.

## Directory Structure

The directory structure is purely conventional. It doesn't make a difference where exactly you put layer definition files. But here's the basic idea:

| Directory | Purpose |
| ---- | ---- |
| Basic | common layers |
| Experimental | internal layers that are being prototyped before inclusion |
| Loss | layers that compute losses |
| Operators | layers that contain other layers |
| Placeholder | represent foreign layers during import, prior to replacement |
| Sequence | layers that operate on sequences |
| Spatial | convolution and related 1D/2D/3D layers |
| Training | layers for regularization etc |

## Definition File Format

There is one definition file per layer. The name of the layer is used to produce the corresponding symbol in a pre-defined way (usually by adding the word Layer to the end of the file name), so for example the file `"Linear.m"` will be used to populate a layer called `LinearLayer`. If there is a system symbol with this name, then that will be used, otherwise an "internal" layer is produced under the ```NeuralNetworks` ``` context.

The file format shared by layers, encoders, and decoders looks like this:

```
FieldName1: value

FieldName2:
	SubFieldName1: value
	SubFieldName2: value

arbitraryStatement[];
```

Arbitrary statements are allowed, but must end in a `;`. Nested fields and values are turned into nested associations in the obvious way, so for this file we'd have:

```
<|
	"FieldName1" -> value,
	"FieldName2" -> <|
		"SubFieldName1" -> value,
		"SubFieldName2" -> value
	|>
|>

```

These definitions associations are imported via `ReadDefinitionFile`, then passed to `DefineLayer`, `DefineEncoder`, `DefineDecoder`. You could build the association yourself to pass to these layers, but there's no good reason to do that. Just use the funky syntax!

A note about contexts: definition files are loaded each in a unique context, via the second arg of `ReadDefinitionFile`. This makes sure that function definitions, variables, etc in the definition don't conflict. This context is based on the name of the file itself.

When a value is a pure functions its extremely common to structure them like this:

```
FieldName: Function[
	var1 = foo[#1];
	var2 = bar[#2];
	var1 + var2 
]
```

Note that as a special, magic courtesy, the definitions `x = y` you perform in a pure function will automatically be localized with a `Block`, so this function will turn into:

```
Function @ Block[{var1, var2},
	var1 = foo[#1];
	var2 = bar[#2];
	var1 + var2 
]
```

#### Note: Loading

When a definition is loaded via `ReadDefinitionFile`, it is then fed to `DefineLayer`, which will do check that the definition is well-formed (has all the necessary fields, no extra fields, and the values of these fields is correct). Errors will be reported if a problem is found. If everything is good, the data will be inserted into `$LayerData` under a key based no the name of the file.

The definition is actually represented as a type that can be found in `DefineLayer.m`, and it looks something like this:

```
LayerDefinitionT = ObjectDefinitionT[
	...
	"Inputs" -> 					AssocT[StringT, TypeExpressionT],
	"Outputs" -> 					AssocT[StringT, TypeExpressionT],
	"Parameters" -> 				AssocT[StringT, TypeExpressionT],
	...
	"ShapeFunction" -> 				FunctionT,
	"RankFunction" -> 				FunctionT,
	...	
	"Tests" -> 						ListT @ RuleT[ExpressionT, StringT],
	...
	"AllowDynamicDimensions" -> 	Defaulting[BooleanT, Automatic],		
	...	
]
```

Don't worry about what these mean, but know that this makes it easy to define new fields and ensure they are correctly used.

To browse the actual data in `$LayerData`, you can use `LayerInfo["layername"]` to print out a nice-looking table of all the fields. This data represents everything there is to know about the layer and everything that determines how it behaves. 

## Core Fields

The core fields define the basics of the layer, like what inputs and outputs it has, the user-visible parameters, and internal parameters (which exist for book-keeping purposes). We call each individual input, output, parameter, or array an **element**, and it can be addressed by a `NetPath`, but more on that later.

The core fields are *Inputs*, *Outputs*, *Parameters*, and *Arrays*. The last two are optional (very simple layers may not need them), but the first two must always be provided.

Let's look at an example for the *HingeLoss* layer (try it out using ```NeuralNetworks`HingeLossLayer[]```):

```
Inputs: 
	$Input: TensorT[$$Dimensions]
	$Target: TensorT[$$Dimensions]

Outputs:
	$Loss: ScalarT 

Parameters:
	$Margin: Defaulting[ScalarT, 0.5]
	$Norm: Defaulting[EnumT[{"L1","L2"}], "L2"]
	$$Dimensions: SizeListT[]
```
First, note that the keys for *Inputs*, *Outputs*, and *Parameters* all start with `$`. This has a special meaning, and only applies to elements of this core fields. We'll elaborate on this later.

Next, we can see that the layer has two inputs called *Input* and *Target*, that are both arrays with identical dimensions. These dimensions are stored as a hidden parameter.

Additionally, the layer has one output called Loss that is a scalar.

Lastly, it has three parameters, two visible and one hidden (the extra `$` represents that it is hidden). The visible parameters have default values for when the user doesn't provide a value.

#### Note: NetPath references

There is one piece of magic to mention here: if you use a symbol like `$Variable` as a key, such as `$Target` here, the key actually used will be the string `"Target"`, but the symbol `$Target` will be replaced with `NetPath["Inputs", "Target"]` in all other locations. These `NetPath` expressions are basically references to elements of the current layer. They are treated specially by type inference, and also will evaluate to their values when used inside various pure functions.

But they're not pleasant to type, so the `$Variable` syntax is a convenient way of expressing them from inside a definition file. 

To make this mechanism more explicit, let's look at what the above definition actually turned into:

```
<|
	"Inputs" -> <|
		"Input" -> TensorT[NetPath["Parameters", "$Dimensions"], RealT],
		"Target" -> TensorT[NetPath["Parameters", "$Dimensions"], RealT]
	|>,
	"Outputs" -> <|
		"Loss" -> TensorT[{}, RealT]
	|>,
	"Parameters" -> <|
		"Margin" -> Defaulting[TensorT[{}, RealT], 0.5],
		"Norm" -> Defaulting[EnumT[{"L1", "L2"}], "L2"],
		"$Dimensions" -> ListT[NaturalT, SizeT]
	|>,
	...
|>
```

## Inference Rules

When you express the type of a net **element** (an input, output, parameter, or array) to be made up of the values of other elements, what actually happens is that an **inference rule** is introduced to connect the element with the other elements it depends on.  

These inference rules can be seen using e.g. `LayerInfo["HingeLoss", "InferenceRules"]`.

These rules are like little programs that run whenever their inputs (the dependent elements) change. That program should produce a new output (the type of the original element). The type inference engine will run these little programs again and again until a fixed point is reached.

This mechanism will be described in more detail in `Types/Inference.md`.

## Optional Fields

In addition to the core fields, the following fields can also be provided:

| Key | Meaning |
| --- | --- |
| **MXNet** | data to establish simple translations to MXNet without using a Writer |
| **Writer** | function to compile more complex layers to MXNet operations |
| **Tests** | list of layer inputs and their required outputs |
| **ShapeFunction** | function mapping dims of all input ports to dims of all output ports |
| **RankFunction** | function mapping ranks of all input ports to ranks of all output ports |
| **TypeFunction** | function mapping types of all input ports to types of all output ports |
| **ExtraShapeFunctionTensors** | list of additional parameters (array, etc.) passed to the shape function |
| **PostInferenceFunction** | function to run after inference involving this layer is complete |
| **PostConstructionFunction** | function to run after this layer is first constructed |
| **FinalCheck** | function to run before the net is evaluated |
| **ReshapeParams** | a list of internal parameters that should be wiped when a net is reshaped |
| **ArgumentRewriter** | how to rewrite the args to the function before doing the standard argument processing |
| **Constraints** | a function to run to generate constraint relationships between inputs and outputs |
| **StateExpanding** | whether this operator will change the state size of interior nets. usually operators do. |
| **Immutable** | whether parameters of this layer can be changed after it is created |
| **HasDynamicPorts** | whether this layer has ports that change at construction time |

You *must* provide one of `Writer` or `MXNet` in order for the layer to be actually usable. All the other fields represent features that may or may not be important, in descending order of use-frequency. Also, some of these are set automatically and are referenced here for completeness.

---

### MXNet

#### Motivation

For a layer to actually be usable, it must be turned into the native operations of the backend, which for us is MXNet. For simple layers, we can just define a one-to-one translation between the WL layer and its arrays and properties and the MXNet primitves.

### How to use

The MXNet section is a set of sub-properties that define the mapping of the layer to MXNet (and back). It is only used for simple mappings for where there is a single MXNet layer that implements the WL layer. More complex mappings should depend on a custom Writer (and are typically export-only).

| Sub-property | Meaning |
| ---- | ---- |
| Name | Name of the MX op |
| Parameters | Section mapping WL layer parameter to MX field name |
| Arrays | Section mapping WL layer array name to MX array name |
| Reader | Function that takes the MX op JSON and returns list of arguments to create a WL layer | 
| Writer | Function that takes the WL layer association and yields a list of MX op JSON parameters |
| Aliases | List of other MX op names that should also import to this layer |

The reader and writer functions are optional but allow parameters to be translated to and from the underlying MXNet attributes dictionary.

---

### Writer

#### Motivation

For more complex layers that do not have a single MXNet operation that implements them, or where the operation varies depending on the settings of the layer, we must use instead a Writer function that has full programmatic control over the translation process.

#### How to use

Writers work by obtaining *MXNodes* that represent vertices of the computation graph and applying functions to them to produce new MXNodes. There are specific "high-level" functions like `SowPlus`, `SowMinus`, `SowFC`, etc. that operate on MXNodes, which should be used when possible, but you can also use `SowNode` when such a function is not available. Run `?MX\`*` of the operators available in MXNet.

In addition to MXNodes, there are also special objects called *MetaNodes* that are also available. These offer an abstraction on top of ordinary MXNodes that can be useful when dealing with dynamic dimensions. 

The following table lists functions that can be used from inside a Writer to access or set the inputs, outputs, and states of the layer being written:
| Function | Purpose | 
| ---- | ---- |
| GetInput["name"] | obtain the MXNode corresponding to the input of the current layer |
| GetInput["name", "kind"] | obtain a particular representation of MXNode ("Batchwise", "Timewise") |
| GetInputMetaNode["name"] | obtain a MetaNode |
| Get{In,Out}putDims["name"]| get the dimensions of the given input or output |
| Get{In,Out}putRank["name"] | get the rank of the given input or output |
| GetState["name"] | get the initial value of a recurrent state |
| SetOutput["name", node] | set the output of the layer to be an MXNode or MetaNode |
| SetState["name", node] | set the final value of a recurrent state |

If it turns out that the full writer is not necessary for a specific setting of the parameters of a layer, and a simple MXNet translation is possible, the function `MXWriteDefaultAndReturn` can be called. `SowCurrentNode[inputs]` can also be called if the inputs need to be first preprocessed and then the simple translation used.

For more information about writers, see `NeuralNetworks/Layers/Writers.md`, which also contains some helpful tips about how to debug and prototype writers.

---

### Tests

#### Motivation

Layer tests ensure that layer functionality keeps works even when refactors or framework changes occur. They also ensure that GPU behavior is suitably close to CPU behavior.

#### How to use

This section is a list of test cases:

```
Tests: {
	{arg1, ..., opt1, ...} -> "dims_hash1_hash2_norm",
	...
}
```

These can be created with `CreateLayerTests` and run with `RunLayerTests`. 

Layer tests take the view that once layer functionality has been verified to *work*, it should not change thereafter. To achieve that, we do not store concrete examples of provided inputs and correct outputs. Rather, we use pseudorandomness to generate random but deterministic input arrays of the desired shapes, and then we verify that the output arrays have the correct properties. These properties include the dimension of the tensor, a numeric hash that summarizes the contents of the array (but is still insensitive to small magnitude variations), and the norm of the tensor.

Therefore, when you create tests with `CreateLayerTests` you should verify that the input/output example you see is correct before you paste the new test case into the definition file.

See `?CreateLayerTests` for more detailed information, but the basic recipe is to create layer tests that exercise all the different functionality of a layer. Each call will produce a test case that can be pasted into the `Tests` section of the layer definition file.

When you run layer tests with `RunLayerTests`, all the tests that were found during the most recent loading even will be run in sequence. Any failures will be reported so they can be run from your notebook in isolation.

If anything about a layer or the testing system changes and a large number of tests start failing, you can run `UpdateLayerTests` to update all the tests to their current behavior, which is assumed to be correct.

One important property verified by layer tests (where applicable) is that a batch of variable-length inputs behaves identically to those same inputs evaluated individually. This will cause several different buckets to be created, and will ensure that padding is being correctly ignored etc. It will also ensure that shape polymorphism is being handled correctly. See the `NeuralNetworks/Layers/ShapePolymorphism.md` file for more info.

---

### ShapeFunction, RankFunction, TypeFunction

#### Motivation

These exist as an alternative and improvement on "old style" type inference. The old style of inference is accomplished by using hidden parameters to "connect" inputs, outputs, and parameters, as well as `ComputedType` expressions when more complex situations are encountered. 

The problem with the old style of inference occurs with more complicated cases that require use of `ComputedType` in multiple directions. To ensure that type inference works "sideways", "forwards", and "backwards", you would have to write a `ComputedType` expression for every possible direction that type inference can flow, e.g. inferring input A from input B and output X, inferring input B from input A and output X, etc. 

To avoid having to do this kind of tedious task, **ShapeFunction**, **RankFunction** and **TypeFunction** allow you to write more algebraic kinds of expressions that involve variables that may be concrete or may be symbolic. As long as your function works "generically", a single function can be used to produce inference that flows in every possible direction. 

A good illustration is `TransposeLayer`: if the input dims are `{_, 2, 3}`, and the outputs are `{_, 1, _}`, then a *ShapeFunction* that just applies the correct permutation to the input dimensions can be used to correctly deduce that the input dims are `{1, 2, 3}` and the output dims are `{2, 1, 3}`. Even shape functions that add dimensions (see `CatenateLayer`) can be correctly "reversed" by algebraic manipulation.

#### How to use

Define a **ShapeFunction** to operate on a list of lists of dimensions (one list for each input port, in order), and return a list of lists of dimensions (one list for each output port, in order). Ensure that you don't *look* at any particular value. You can only add, subtract, or shuffle them around. You can call `FailShape` if an inconsistency is found. 

Define a **RankFunction** to operate on a list of ranks (one rank for each input port, in order), and return a list of ranks (one rank for each output port, in order). Again, ensure you don't look at any particular value, and only add, subtract, etc.

Define a **TypeFunction** to operate on a list of types (one type for each input port, in order), and return a list of types (one rank for each output port, in order). You know the drill: mostly just copy the input type to the output type, or return a fixed value for the output. 

Put your shape, rank, and type functions in `Shapes.m`, and test them individually in the `Internal/Shapes.m` test file.

---

### PostInferenceFunction, PostConstructionFunction, and FinalCheck

#### Motivation

Sometimes, there are special kinds of inference that can only be done after all other inference is complete, or that are too hard to express as a `ComputedType` or `ShapeFunction` etc. This can also be need for *validation*: checking that various settings are consistent with one another or satisfy a particular restriction. There are three mechanisms designed for this:

The **PostInferenceFunction** will run after every net has been fully inferred (it will also run after a layer has been constructed). This function has an opportunity to examine the values of all layer elements, and then, if necessary, change elements to have new values using `PostSet`. Optionally, it can signal that an another entire 'sweep' of inference should occur by calling `RestartInference[]`. 

The **PostConstructionFunction** is like **PostInferenceFunction**, but it runs once, only after a layer is constructed. It will never be run again.

**FinalCheck** is again run only once, just before a net is compiled and first used (trained, or evaluated on inputs).

#### How to use

For checking of constraints:

Use **PostInferenceFunction** when a constraint could be violated as more information becomes available, and once it is violated, there is no way the introduction of yet more information could solve the violation. For example, a `ConvolutionLayer` with a kernel size of 5 cannot have inputs less than size 5, and so a PostInferenceFunction is used there to check that condition, and call the `FailValidation` function to report this violation.

Use **PostConstructionFunction** when a layer could be constructed in an inconsistent way, and this inconsistency will only ever show up during construction and so only needs to be checked for once. Here, call `FailConstruction` to report the error. An example would be `ResizeLayer`, which must have a spec of `Scaled[n]` when `Method -> "Nearest"`. 

Use **FinalCheck** when checking for a violation that could otherwise go later. The only safe time to report an actual error is when the user attempts to use the net. For example, layers that do not support dynamic dimensions must only check for the presence of dynamic dimensions when the net is used, as dynamic dimensions can go away when a user provides more information like a fixed input size.

---

### ReshapeParams

#### Motivation

When an input of a net is changed via `NetReplacePart[net, "Input" -> size]`, it can be necessary to throw away old elements inside the net to make it possible for the new type information to propagate through the net without encountering and clashing with parameters that relate to the old input size. This is accomplished in `NetReshape.m`. 

The set of parameters for a layer that should be wiped is given by the **ReshapeParams** field. The parameters listed in this field will be reset to their initial definition defaults when a reshape occurs. 

#### How to use

This field is set automatically to be *all* hidden parameters (parameters starting with `$`). But it might need to be overriden in rare cases. 

---

### ArgumentRewriter

#### Motivation

TODO

#### How to use

TODO

---

### Constraints

#### Motivation

Certain layers will not work on all input sizes, they place constraints on the dimensions of their inputs. Normally one would establish these with `PostInferenceFunction`, but if the layer is also to work on an array of dynamic size, the constraint needs to be verified at runtime, and reported sensibly to the user.

The **Constraints** field defines a function that will be run to express the constraints that apply to the inputs, outputs, and parameters of the net.

Before a net is run, the constraint system will collect all constraints from a net, relate them to the sizes of the input dimensions, and solve systems of equations to establish fast checks that can be applied during bucket creation to ensure that the constraints are not violated.

Call `NetConstraintData` on a net to see the underlying machinery.

#### How to use

From within the **Constraints** function, call `SowConstraint` to create a constraints that relates parameters/inputs/outputs to each other. For example, here is the constraint function of `SequenceMost`:

```
Constraints: Function[
	SowConstraint[$$Length > 1];
	SowConstraint[$$OutputLength == $$Length - 1]
]
```

---

### StateExpanding

#### Motivation

State ports are special ports that exist on layers like `LongShortTermMemoryLayer`, and that if left unconnected will implicitly be available for setting and getting individually by the user when running the net. They act as both inputs and outputs. `NetStateObject` explicitly manages values for these state ports.

Unfortunately, if an operator layer like `NetMapOperator` contains a net that has state ports, these state ports would need to inherit an extra dimension that corresponds to the length of the array being mapped over. Similar stories apply for other kinds of operator. It is possible to support this feature, in theory, but as its complex and not particularly useful, it is currently forbidden. 

The **StateExpanding** field will ensure that the framework suppresses state ports in interior nets.

#### How to use

This is set automatically for operator-type layers. You should set this to false if writing an operator that will not affect the size of the interior state ports, e.g. NetBidirectionalOperator.

---

### Immutable

#### Motivation

In general, you can replace any parts of a net using `NetReplacePart`. Sometimes, however, this will result in a broken net. An example would be a `NetMapOperator`, whose very ports depends on the ports of the mapped net. Hence, replacing the mapped net will not always make sense, and so needs to be forbidden.

#### How to use

Set the **Immutable** field to True if you wish to forbid the user making changes to the parameters etc of a layer after it has been created. It is automatically set to true for operator layers.

---

### HasDynamicPorts

#### Motivation

Some layers do not have a fixed set of ports in advance, and adapt their number of ports depending on how they are constructed/used. 

If these nets have shape functions (see `ShapeFunction` etc above), the inference rules created by the shape function must be created at runtime rather than at definition time, as the rules themselves will not be known in advance. This is slower, of course. 

**HasDynamicPorts** is the field that specifies whether a ShapeFunction can produce a static set of inference rules that are baked into the layer definition, or whether a runtime set of rules must be produced once the ports are known for a specific instance of a layer.

#### How to use

**HasDynamicPorts** will be automatically set for "multiport" layers like `ThreadingLayer` and `TotalLayer` because they use the `Input: $Multiport` idiom. However, the field should be manually set to true for operators whose ports adapt to their contained net, such as `NetMapOperator`. 

