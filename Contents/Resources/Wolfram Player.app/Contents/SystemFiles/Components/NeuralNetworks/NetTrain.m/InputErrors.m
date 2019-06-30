Package["NeuralNetworks`"]


$inputErrorsSetup := Quoted[
	$trainingGeneratorIsCustom = False; 
	$batchWasSkipped = <||>; 
	$skippedExamples = Bag[];
	$skippedExampleCount = 0;
];

NetTrain::skipbatch = "Batch #`` will be skipped, because one or or more inputs provided to port \"``\" was invalid: ``. This batch will be ignored in subsequent training rounds. More information can be obtained via the \"SkippedTrainingData\" property."
NetTrain::skipbatchgen = "Batch #`` will be skipped, because one or or more inputs provided to port \"``\" was invalid: ``.";

NetTrainInputErrorHandler[name_, types_][data_] := Scope[
	If[$batch === 0, Return[]]; (* we are being timed; training hasn't started yet *)
	If[MatchQ[data, _EncodeFailure],
		reason = First @ data
	,
		type = TypeString @ types[name];
		dims = machineArrayDimensions[input];
		reason = If[ListQ[dims] && Length[dims] > 0, 
			StringForm["the training data was ``, but should have been ``", TypeString @ TensorT @ dims, type],
			StringForm["the training data was not ``", type]
		]
	];
	Which[
		$trainingGeneratorIsCustom,
			Message[NetTrain::skipbatchgen, $batch, name, reason],
		!KeyExistsQ[$batchWasSkipped, $currIndex],
			BagPush[$skippedExamples, $thisBatchPermutation, 1];
			Message[NetTrain::skipbatch, $batch, name, reason];
			$batchWasSkipped[$currIndex] ^= True,
		True,
			Null
	];
	$skippedExampleCount += batchSize;
];


PackageScope["PreencodedInputErrorHandler"]

(* PreencodedInputErrorHandler is called by the sequence preencoding generator in Generators.m, or the preencoding 
functions in MXNetLink (mediated through a function that looks up the permutation indices). It has no choice but
to fail, but it will try to provide the location of the example that did fail.
*)

General::encgenfail1 = "Could not encode input number `` for port \"``\": ``. Please check the example.";
General::encgenfail2 = "Could not encode one or more inputs for port \"``\": ``. The invalid inputs had indices ``.";

PreencodedInputErrorHandler[name_, type_, result_, indices_, list_, encf_] := Scope[
	Do[
		Replace[
			encf @ Part[list, List @ index],
			EncodeFailure[reason_] :> ThrowFailure["encgenfail1", index, name, reason]
		]
	,
		{index, indices}
	];
	ThrowFailure["encgenfail2", name, failureForm[result, type], indices];
]

failureForm[EncodeFailure[reason_], type_] := 
	reason;

failureForm[data_, type_] := 
	StringForm[
		"supplied data was ``, but expected ``", 
		TypeString @ TensorT @ machineArrayDimensions @ data, 
		TypeString @ type
	];


PackageScope["InputErrorHandler"]

General::invindata1 = "Data supplied to `` was ``, but expected ``.";
General::invindata2 = "Data supplied to `` was not `` (or a list of these).";
General::invindata3 = "Data supplied to `` could not be encoded; ``.";

(* TODO: split this into batched and non-batched versions, the batched version
should search in the input data to find the element that didn't match and report it.
also deal with sequences etc. *)
InputErrorHandler[name_, type_][input_] := Scope[
	portName = FmtPortName @ name;
	type = TypeString @ type;
	Replace[input, EncodeFailure[reason_] :> ThrowFailure["invindata3", portName, reason]];
	dims = machineArrayDimensions[input];
	If[FailureQ[dims], 
		ThrowFailure["invindata2", portName, type],
		ThrowFailure["invindata1", portName, TypeString @ TensorT[dims], type]
	]
]