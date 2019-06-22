Package["NeuralNetworks`"]



PackageScope["CreateLayerTest"]

SetUsage @ "
CreateLayerTest[layer$, arg$1, $$, arg$n, opt$1, $$, opt$n] returns a layer test \
given a net layer layer$, the positional arguments to the layer arg$1, $$ ,arg$n and \
any optional arguments opt$1, $$, opt$2. 
CreateLayerTest[layer[arg$1, $$, opt$1, $$]] is a synonym for the above.
* It is a requirement the provided arguments determine the input and output shapes of the layer. 
* The resulting test is automatically copied to the clip board, and can be pasted \
into the tests section of a layer definition file.
* The RHS of the entry generated for the tests section is a string formed from the following \
elements, joined by underscores:
	* The shape of the output, as a list of dimensions.
	* A hash of the output on a single randomly generated example.
	* A hash of the output on a batch of randomly generated examples.
	* A numeric value that is the L1 norm of the single example. 
* The numeric hashes have a bit of 'slop', so array elements are rounded prior to hashing.
* The output of CreateLayerTest is {entry$, singleViz$, batchViz$}:
	* entry$ is the value copied to the clipboard.
	* singleViz$ is a graphical representation of the input and output of the layer on a single example.
	* batchViz$ is the same, but for a batch of examples.
* The representation lays out high dimensional arrays in a flat grid. High dimensional arrays \
are represented as follows:
	* The last two axes become rows and columns.
	* The first axis uses a dark separator, and successive axes use lighter and lighter separators.
	* Entry colors are random but unique to each value, making it easy to match values between input and output."

General::typenotspec = "In test of ``, port `` has a type that is not fully specified:\n``"
General::laytestevalfailed = "Layer `` failed when evaluated on `` input with dimensions ``. 
Layer InputForm available as $LastTestLayerInputForm. Layer available as $LastTestLayer. Input available as $LastTestInput and $LastTestBatchInput."
General::laytestinconsistent = "Layer `1` had numerical inconsistency between `5` and `6` evaluations with TargetDevice -> `2` (`3` > `4`)."
General::laytesttrain = "Layer `` produce the same outputs with NetEvaluationMode -> \"Train\" (and TargetDevice -> ``), which is unexpected."
General::probreshapefailure = "Layer `` evaluated on entire batch but failed to subsequently evaluate on batch element #`` which has dimensions ``, this indicates the Writer was not shape-polymorphic (e.g. contained a hard-coded sequence length) but failed to set $ReshapingIsImpossible = True. See $LastInternalfailure for more clues."
General::laytestshape = "Layer outputs `` but `` was expected:  ``"
General::laytestnatype = "Layer outputs a NumericArray of type `` but `` was expected."
General::laytestreshape = "ReshapeParams maybe be wrong. NetReplacePart[`1`, `2`] is not working properly."

SetHoldAll[CreateLayerTest];

CreateLayerTest[head_Symbol[args___]] := CreateLayerTest[head, args];

CreateLayerTest[head_, args___] := CatchFailure @ Scope[
	{res, v1, v2} = getTestOutput[head, args];
	istring = With[{res2 = res}, ToString[Unevaluated[{args} -> res2], InputForm]];
	CopyToClipboard[istring];
	{{args} -> res, nform /@ v1, nform /@ v2}
];

nform[assoc_Association] := Map[nform, assoc];
nform[l_List /; MachineArrayQ[l] || VectorQ[Flatten[l], NumberQ]] := ColorArrayForm[l];
nform[e_] := e;

realNATypeQ[x_NumericArray] := realNATypeQ[NumericArrayType[x]]
realNATypeQ[x_String] := StringStartsQ[x, "Real"] (* ensure compatibility with future types *)

getTestOutput[head_, args___] := Scope[
	tolerance = If[MatchQ[$LayerTestDevice, "GPU"|{"GPU", _}], 1.*^-5, 1.*^-6];
	$head = head;
	GeneralUtilities`PackageScope`$DisableCatchFailureAsMessage = True;
	$LastTestLayerInputForm ^= With[{h = head}, HoldForm[h[args]]];
	$LastTestLayer ^= layer = createInitializedLayer[head, args];
	If[FailureQ[layer], 
		res = tostr[layer, False];
		Return @ {res, None, None}
	];
	inputs = Inputs[layer];
	{in, inBatch} = createRandomInputData[inputs];
	batchSize = 4;
	Label[batchHadShortElem];
	$LastTestInput ^= in; $LastTestBatchInput ^= inBatch;
	out = layer[in, TargetDevice -> $LayerTestDevice, WorkingPrecision -> $LayerTestType];
	If[FailureQ[out], Return @ {tostr[out, False], None, None}];

	outNAType = If[NumericArrayQ[out], NumericArrayType[out], None]; (* change if TypedScalar arrives! *)
	(* only deal with real types. Integer types, eg OrderingLayer, should not be affected by this check *)
	If[realNATypeQ[outNAType],
		expectedNAType = Replace[$LayerTestType, "Mixed" -> "Real32"];
		If[(outNAType =!= expectedNAType) && (outNAType =!= None),
			ThrowFailure["laytestnatype", outNAType, expectedNAType];
		];
	];

	out = Normal[out];
	otype = First @ Outputs @ layer;
	If[!ToLiteralTypeTest[otype][out],
		ThrowFailure["laytestshape", TypeString @ TensorT @ machineArrayDimensions @ out, TypeString[otype], $LastTestLayerInputForm];
	];
	outBatch = layer[inBatch, TargetDevice -> $LayerTestDevice, WorkingPrecision -> $LayerTestType];
	If[FailureQ[outBatch],
		(* The following line is here to handle the cases of SequenceMostLayer and SequenceRestLayer which both require sequences of at least length 2.
			It turns out that the 3rd example of the batchs (of 4 elements) has length 1 (too short for these layers),
			which explains the Take[...., 2] to limit to a batch of valid inputs for these layers.*)
		If[outBatch[["MessageName"]] === "netseqlen", inBatch = Take[inBatch, 2]; batchSize = Length @ inBatch; Goto[batchHadShortElem]];
		ThrowFailure["laytestevalfailed", $LastTestLayerInputForm, "batch", cleverDims[inBatch]];
	];
	outBatch = Normal[outBatch];

	(* Check consistency between NetEvaluationMode -> "Test" and NetEvaluationMode -> "Train" *)
	trainingLayerQ = MatchQ[head, BatchNormalizationLayer|ImageAugmentationLayer|DropoutLayer];

	If[(* TODO FIXME *) Or[
		MatchQ[head, BatchNormalizationLayer|ImageAugmentationLayer] && $LayerTestType === "Real64",
		head === NetFoldOperator && $ForceRNNUnrolling
		],
		Goto[skipTrainConsistencyCheck]
	];
	outBatch2 = layer[inBatch,
		NetEvaluationMode -> "Train",
		TargetDevice -> $LayerTestDevice, WorkingPrecision -> $LayerTestType
	];
	If[FailureQ[outBatch2],
		ThrowFailure["laytestevalfailed", $LastTestLayerInputForm, "batch with Train EvaluationMode", cleverDims[inBatch]];
	];
	If[head === ConstantArrayLayer, Goto[skipBatchConsistencyCheck]];
	diff = numericDifference[Normal[outBatch2], outBatch];
	Which[trainingLayerQ,
			If[diff < tolerance,
				ThrowFailure["laytesttrain", $LastTestLayerInputForm, $LayerTestDevice]
			],
		diff > tolerance,
			ThrowFailure["laytestinconsistent", $LastTestLayerInputForm, $LayerTestDevice, diff, tolerance, "Train", "Test"];
	];
	Label[skipTrainConsistencyCheck];

	(* Check consistency between singleton and batch evaluation *)
	If[head === ConstantArrayLayer, Goto[skipBatchConsistencyCheck]];
	If[!ToBatchTypeTest[<|"Input" -> otype|>, batchSize][<|"Input" -> outBatch|>],
		ThrowFailure["laytestshape", TypeString @ TensorT @ machineArrayDimensions @ outBatch,
			StringForm["a batch of `` ``", batchSize, TypeString[otype, True]], $LastTestLayerInputForm];
	];
	Do[
		bin = getNth[inBatch, i];
		bout = layer[bin, TargetDevice -> $LayerTestDevice, WorkingPrecision -> $LayerTestType];
		If[FailureQ[bout],
			ThrowFailure["probreshapefailure", $LastTestLayerInputForm, i, If[AssociationQ[bin], Map[Dimensions], Dimensions] @ bin]];
		bout = Normal[bout];
		If[(diff = numericDifference[bout, getNth[outBatch, i]]) > tolerance && head =!= ConstantArrayLayer,
			ThrowFailure["laytestinconsistent", $LastTestLayerInputForm, $LayerTestDevice, diff, tolerance, "batch", "singleton"];
		],
		{i, batchSize}
	];
	Label[skipBatchConsistencyCheck];

	(* Check that reshaping works (could be that ReshapeParams is not well defined *)
	If[head === NeuralNetworks`MXLayer, Goto[skipReshapeTests]];
	Do[
		resizedlayer = Quiet @ NetReplacePart[layer, port];
		If[normalizeLayer[resizedlayer] =!= normalizeLayer[layer],
			ThrowFailure["laytestreshape", $LastTestLayerInputForm, port];
		],
		{port,
			Apply[Join] @ Map[
				Normal @ NetInformation[layer, #]&,
				{"InputPorts", "OutputPorts"}
			]
		}
	];
	Label[skipReshapeTests];
	res = StringJoin[tostr[out, False], "_", tostr[outBatch, True], "=", SciString[tototal[out], 10]];
	$LastTest ^= {res, in -> out, inBatch -> outBatch}
];

cleverDims[dims_Association] := Map[cleverDims, dims];
cleverDims[s_String] := String;
cleverDims[l_List /; !PackedArrayQ[l] && Length[l] <= 3] := Map[cleverDims, l];
cleverDims[e_List] := Dimensions[e];

(* jeromel: copy-paste from normalizeNet in Test/Utils.init.m *)
normalizeLayer[layer_] := Block[{data, lenAssoc, lenAssocIndex, newLengthVar, replacements},
	data = NData[layer];
	lenAssoc = <||>; lenAssocIndex = 0;
	replacements = {
		ListT[n_Integer, SizeT] :> RuleCondition @ Table[SizeT, n],
		TensorT[{}, i_IndexIntegerT] :> i,
		AtomT :> RealT,
		(* assign arbitrary ordered indices to LengthVar variables, instead of the original random ones *)
		lv_LengthVar :> RuleCondition @ CacheTo[lenAssoc, lv, LengthVar[lenAssocIndex++]],
		Nullable[i_] :> i,
		a_Association :> Map[ReplaceAll[replacements], KeySort[a]]
	};
	data /. replacements
];


PackageScope["$LastTest"]
PackageScope["$LastTestLayer"]
PackageScope["$LastTestLayerInputForm"]
PackageScope["$LastTestInput"]
PackageScope["$LastTestBatchInput"]
PackageScope["$TestHashFunction"]

$TestHashFunction = RoundRelative /* Base64Hash;

$LastTestLayerInputForm = $LastTest = $LastTestLayer = $LastTestInput = $LastTestBatchInput = None;

numericDifference[a_, b_] := If[Dimensions[a] =!= Dimensions[b], Infinity, Max[Abs[a - b]]];

getNth[assoc_Association, n_] := assoc[[All, n]];
getNth[list_List, n_] := list[[n]];

tostr[f_Failure, _] := TextString[f];
tostr[e_, batchq_] := If[batchq, "", StringRiffle[Dimensions[e],"*"] <> "_"] <> $TestHashFunction[e];

tototal[f_Failure] := None;
tototal[e_] := Total[Abs[List[e]], Infinity];


PackageScope["RoundRelative"]

SetAttributes[RoundRelative, Listable];
RoundRelative[n_ ? NumericQ] := 
	If[Abs[n] < 10^-6, 0., Block[{man,exp}, 
		{man,exp} = MantissaExponent[N[n]];
		Round[man, .01] * 10^exp]
	];
RoundRelative[n_] := n;

(* 
a subtle point about tests for layers that use SowSeqMask to deal with dynamic dims: 
the batch/non-batch equality checking takes place on the first element of the random 
synthesized data. we are lucky in that for sequences this element is in most cases 
shorter than at least one of the others (look at Softmax["Input" -> {"Varying", 4}]), 
and so we know that the batched case involves applying a nontrivial mask that affects 
this first batch element, and the unbatched case does because a batchsize of 1 implies 
the masking operation is a no-op because there is no raggedness, and so we're 
indirectly  testing that the code path that uses a mask isn't changing the semantics 
of the layer.

just something to be aware of, and to manually make sure of by visual inspection when
creating layer tests for such layers.
*)

$longSequenceRules := $longSequenceRules = {
	NameToLengthVar["x"] -> lvs[10, 8, 9, 7],
	NameToLengthVar["y"] -> lvs[30,28,29,27],
	NameToLengthVar["z"] -> lvs[70,68,69,67]
};

createRandomInputData[<|"Input" -> t:TensorT[_List, RealT]|>] := Scope[
	t = t /. $longSequenceRules;
	SeedRandom[54321];
	val0 = rand0[t];
	val1 = rand1[t];
	val2 = rand2[t];
	val3 = rand3[t];
	{val0, {val0, val1, val2, val3}}
];

createRandomInputData[types_] := Scope @ BlockRandom[ 
	types = types /. $longSequenceRules;
	KeyValueScan[
		If[!FullySpecifiedTypeQ[#2], ThrowFailure["typenotspec", $head, #1, #2]]&,
		types /. lvs[a_, ___] :> a
	];
	SeedRandom[54321];
	val0 = Map[rand0, types];
	val1 = Map[rand1, types];
	val2 = Map[rand2, types];
	val3 = Map[rand3, types];
	single = val0; batch = AssociationTranspose[{val0, val1, val2, val3}];
	res = {single, batch};
	If[Length[types] === 1, First /@ res, res]
];

(* ensure ragged sequence lengths in the batch *)
rand0[t_] := TypeRandomInstance[t /. _LengthVar -> 3 /. v_lvs :> RuleCondition[v[[1]]]];
rand1[t_] := TypeRandomInstance[t /. _LengthVar -> 4 /. v_lvs :> RuleCondition[v[[2]]]];
rand2[t_] := TypeRandomInstance[t /. _LengthVar -> 1 /. v_lvs :> RuleCondition[v[[3]]]];
rand3[t_] := TypeRandomInstance[t /. _LengthVar -> 2 /. v_lvs :> RuleCondition[v[[4]]]];

createInitializedLayer[head_, args___] := Scope @ BlockRandom[
	SeedRandom[12321];
	args2 = MapSequence[ReleaseHold, args];
	net = head[args2];
	If[FailureQ[net], Return[net]];
	SeedRandom[12345];
	NetInitialize[net, Method -> {"Random", "Weights" -> 1., "Biases" -> 1.}]
];

PackageExport["RunLayerTestsSQA"]

Clear[RunLayerTests];

$LayerTestDevice = "CPU";
$LayerTestType = "Real32";

Options[RunLayerTests] = Options[RunLayerTestsSQA] = {
	TargetDevice -> "CPU", WorkingPrecision -> "Real32",
	"Print" -> False
};

failFunc = Function[
	Print[$LastTestLayerInputForm, " test failed: ", Inactive[Unequal][paster @ #1, #2]]
];

General::laytesthashneq = "Layer `` when run on $LastTestInput and $LastTestBatchInput produced `` instead of ``.
Layer InputForm available as $LastTestLayerInputForm. Layer available as $LastTestLayer."

throwFailFunc = Function[
	ThrowFailure["laytesthashneq", $LastTestLayerInputForm, #1, #2];
];

RunLayerTestsSQA[opts:OptionsPattern[]] := Scope[
	failFunc = throwFailFunc;
	RunLayerTests[opts]
];

PackageExport["RunLayerTests"]

SetUsage @ "
RunLayerTests[] runs all layer tests, returning the total of number tests run.
RunLayerTests[layer$] runs all layer test associated with the layer$.
RunLayerTests[{layer$1, $$, layer$n}] runs all layer test associated with a list layers.
The option TargetDevice can be used to set the target device the tests are run on."

RunLayerTests[opts:OptionsPattern[]] := RunLayerTests[$NetHeads, opts];

RunLayerTests::hardfails = "The following layers encountered hard failures: ``.";

RunLayerTests[heads_List | heads_Symbol, OptionsPattern[]] := CatchFailureAsMessage @ Scope[
	$testCount = 0;
	$lastFailRules ^= <||>;
	$hardFailures = {};
	$shouldPrint = OptionValue["Print"];
	$LayerTestDevice = OptionValue[TargetDevice];
	$LayerTestType = OptionValue[WorkingPrecision];
	Scan[iRunTest, ToList[heads]];
	If[$hardFailures =!= {}, Message[RunLayerTests::hardfails, $hardFailures]];
	$testCount
];

RunLayerTests::failskip = "A hard failure was encountered for ``, skipping remaining tests for that layer.";
iRunTest[sym_Symbol] := Scope[
	tests = $LayerData[$SymbolToType[sym], "Tests"];
	$testHead = sym;
	If[FailureQ @ CatchFailureAsMessage[RunLayerTests, Scan[runTest, tests]],
		Message[RunLayerTests::failskip, sym];
		AppendTo[$hardFailures, sym];
	];
];

toHashAndTotal[str_String] := 
	If[StringFreeQ[str, "="], 
		{str, 0}, (* <- hasn't been upgraded yet *)
		MapAt[Internal`StringToDouble, 2] @ StringSplit[str, "="]
	];

runTest[{args___} -> result_] := Scope[
	If[$shouldPrint, Print[$testHead, args]];
	$MessageList = {};
	(* matteos: the Catch here is to handle generic exceptions which
	   can be possibly thrown by layers (currently ShapeException).
	   Without this, we may not be aware of those cases. Labeled Throws
	   will pass through this. *)
	out = Catch @ First @ getTestOutput[$testHead, args];
	If[$MessageList =!= {}, 
		Print @ With[h = $testHead, Defer[h[args]]];
		ThrowRawFailure[$Failed]];
	{outHash, outTotal} = toHashAndTotal @ out;
	{resHash, resTotal} = toHashAndTotal @ result;
	diff = Abs[outTotal - resTotal];

	(* Hashes are super fragile and only work for CPU with Real32. Turn off for all other cases *)
	failedHash = If[($LayerTestType == "Real32") && ($LayerTestDevice == "CPU"),
		outHash =!= resHash,
		False
	];

	(* check numeric differences *)
	failedND = (diff / ($MachineEpsilon + outTotal)) > $LayerTestTotalThreshold;

	If[failedHash || failedND,
		file = $LayerData[$SymbolToType[$testHead], "SourceFile"];
		If[!KeyExistsQ[$lastFailRules, file], $lastFailRules[file] = {}];
		AppendTo[$lastFailRules[file], result -> out];
		failFunc[out, result]
	];
	$testCount++
];


PackageScope["$LayerTestTotalThreshold"]
$LayerTestTotalThreshold = 0.01;

paster[e_] := MouseAppearance[
	EventHandler[e, "MouseClicked" :> CopyToClipboard[ToString[e, InputForm]]],
	"LinkHand"
];


PackageScope["UpdateLayerTests"]

$lastFailRules = <||>

UpdateLayerTests[] := (
	KeyValueScan[updateFile, $lastFailRules];
	$lastFailRules = <||>;
);

UpdateLayerTests::failupdate = "Failed to update `` tests in `` because there are some collisions."

updateFile[file_, rules_] := Scope[
	safeRules = Map[TemplateApply["\"`1`\"", stringEscape @ First[#]] -> TemplateApply["\"`1`\"", stringEscape @ Last[#]]&, rules];
	If[Length[Association[safeRules]] =!= Length[DeleteDuplicates @ Map[List, safeRules]],
		Message[UpdateLayerTests::failupdate, Length[safeRules], file];
		Return[];
	];
	str = FileString[file];
	str2 = StringReplace[str, safeRules];
	If[str =!= str2, 
		Print["Updating ", Length[safeRules], " tests in ", file];
		WriteString[file, str2];
		Close[file];
	,
		Print["Couldn't find targets in ", file];
	];

];

stringEscape[str_String] := StringReplace[str, {"\"" -> "\\\"", "\\" -> "\\\\"}];