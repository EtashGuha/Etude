Package["NeuralNetworks`"]


(* Note: this is substituted into the definition of NetTrain using a MacroEvaluate *)
$validationSetup := Quoted[

	If[Length[validationSet] == 2 && MatchQ[validationSet, {_, "Interval" -> _}],
		{validationSet, validationInterval} = validationSet;
		validationInterval = {validationInterval};
	,
		validationInterval = {"Interval" -> 1}; (* default interval is 1 round *)
	];

	If[MatchQ[validationSet, Automatic] && MatchQ[data, _String], validationSet = data];
	
	$validationTimes = Internal`Bag[];
	$computingValidation = False;
	Switch[validationSet
	,	None,
		$doValidation = False;

	,	Scaled[_ ? NumericQ],
		(* for this option, we re-use the training generator, reserving some fraction 
		of the batches at the end to be devoted to validation. *)
		$doValidation = True;
		$scaledValidation = True;
		vfraction = N @ First[validationSet];
		NetTrain::invvsfrac = "Setting ValidationSet->Scaled[n] should use n between 0 and 1 such that training data retains at least one batch.";
		If[!(0 < vfraction < 1), ThrowFailure["invvsfrac"]];
		newBPR = Floor[$batchesPerRound * (1 - vfraction)];
		If[newBPR < 1 || newBPR == $batchesPerRound, ThrowFailure["invvsfrac"]];
		$validationLength = ($batchesPerRound - newBPR) * batchSize;
		$batchesPerRound = newBPR; $trainingLength = newBPR * batchSize; 
		$validationGenerator = Function[n, $trainingGenerator[n + $batchesPerRound]];
	,	_,
		$doValidation = True;
		$scaledValidation = False;
		validationSet = TryPack[validationSet];
		{$validationGenerator, $validationLength} = ParseTrainingData[validationSet, inputs, batchSize, ConstructValidationGenerator];
	];

	If[$doValidation,
		(* not saving to disk can speed things up considerably *)
		$useMemoryCheckpoints = NetInformation[$net, "ArraysTotalByteCount"] < MemoryAvailable[]/8;
		If[$useMemoryCheckpoints, $memoryCheckpointVar = Module[{var}, Hold[var]]];
	];		

	$batchperm = Range[$batchesPerRound];

	If[$doValidation,
		$validationDir = CreateDirectory[];
		AppendTo[$CleanupQueue, Hold[DeleteDirectory][$validationDir, DeleteContents -> True]];
		$validationPeriodicCallback = makePeriodicFunction[ValidationSet, validationFunction, validationInterval];
	,
		$validationPeriodicCallback = Hold;
	];
];

validationFunction[] := UseMacros @ Block[{vloss, vmetrics},
	If[$doPrinting && $lastValidationTime > 5.0, 
		Print["Computing validation loss."];
	];
	handleEvent["ValidationStarted"];
	$CoreLoopLogger["ValidationStarted"];
	$computingValidation = True;
	$lastValidationTime = First @ AbsoluteTiming[
		{vloss, vmetrics} = TrainerDoValidation[$trainer, $validationGenerator, $validationLength];
		$validationLoss = First @ Values @ vloss;
		pushAssocToBags[vmetrics, $validationMeasurementsBags];
		BagPush[$validationTimes, N[$absoluteBatch / $batchesPerRound]];
		$validationMeasurements = vmetrics;
	];
	$validationTimeComp = $lastValidationTime;
	$computingValidation = False;
	If[$scaledValidation,
		(* if we are doing Scaled, the $nextDataBatch might have been corrupted by reusing the generator,
		so we have to regenerate it here *)
		$nextDataBatch = $trainingGenerator[$nextIndex];
	];
	If[$betterCriterion[], 
		handleEvent["NetValidationImprovement"];
		$bestValidationRound = $round;
		If[$useMemoryCheckpoints, 
			TrainerSaveCheckpoint[$trainer, $memoryCheckpointVar];
			$bestValidationFile = $memoryCheckpointVar;
		,
			Block[{newFile, oldFile},
				newFile = FileNameJoin[{$validationDir, IntegerString[$validationFileCount++]}];
				oldFile = $bestValidationFile;
				TrainerSaveCheckpoint[$trainer, newFile];
				$bestValidationFile = newFile;
				If[StringQ[oldFile], Quiet @ DeleteFile[oldFile]];
			]
		];
	];
	handleEvent["ValidationComplete"];
	$CoreLoopLogger["ValidationComplete"];
];
