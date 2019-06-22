Package["NeuralNetworks`"]


DefineMacro[checkStop, checkStop[] := Quoted @ If[$shouldStop, Goto[Done]]];
DefineMacro[timeInto, timeInto[var_Symbol, body_] := Quoted @ updateTimings[var, First @ AbsoluteTiming @ body]];


(* Note: this is substituted into the definition of NetTrain using a MacroEvaluate *)
$coreTrainingLoop = Quoted[

	ScopeVariable[
		$roundStartTime, 
		$roundMeasurementsTotals, 
		$batchStartTime,
		$batchTime,
		$rawBatchLosses,
		$rawBatchMetrics
	];

	$inputsPerRound = $batchesPerRound * batchSize;

	$startOfRound;
	$coreTrainingLoopInner;
];


$coreTrainingLoopInner := UseMacros[
	(* conceptually, this is the loop, though its done using a Goto because
	Do compiles stuff *)
	$CoreLoopLogger["StartOfLoop"];

	Label[StartOfBatch];
		$startOfBatch; checkStop[];

		$initiateTrainerUpdate; 
		
		$fetchDataFromGenerator; 
		checkStop[];

		$obtainLossFromTrainer;

		$calculateProgressAndTimings;

		$isEndOfRound = Divisible[$absoluteBatch, $batchesPerRound];
		If[$isEndOfRound, $summarizeRound];

		$callPeriodicFunctions; 
		checkStop[];

		If[$absoluteBatch == $maxBatches, $shouldStop = True; $reasonTrainingStopped = "MaxTrainingBatches"];
		If[$isEndOfRound, 
			$endOfBatchAndRound,
			$endOfBatch;
		];

		checkStop[];
	Goto[StartOfBatch];

	Label[Done];
	$CoreLoopLogger["TrainerFinalize"];
	TrainerFinalize[$trainer];
	$CoreLoopLogger["EndOfLoop"];
];	


$startOfRound := (
	$batch = 1;
	$roundStartTime = $now;
	handleEvent["RoundStarted"];
);	


(* we want the round loss to be calculated in time for the periodic functions
and checkpointing functions to have access to it *)
$summarizeRound := (
	
	(* search NetTrain.m/ files for addRoundCollector JoinTo[$roundCollector, ...] for more info.
	at a minimum this includes code that divides the accumulated loss for this
	round by the number of batches in the round to give the round loss, and stores
	this in the history bags *)

	List @@ $roundCollectors;
)


$endOfBatchAndRound := (
	handleEvent["RoundComplete"];
	(* update this in time for the callbacks for first batch of next round *)
	$progressFraction = $rawProgressFraction;
	If[$round >= maxTrainingRounds, 
		$shouldStop = True;
		$reasonTrainingStopped = "MaxTrainingRounds";
	,
		$round++;
		$absoluteBatch++;
		$startOfRound
	]
);


$startOfBatch := (
	$CoreLoopLogger["StartOfBatch"];
	$batchStartTime = $now;
	$currIndex = $nextIndex; 
	$validationTimeComp = 0;
	$roundMeasurementsTotals = 0;
	If[$batchStartTime > $endTime, $reasonTrainingStopped = "TimeGoal"; $shouldStop = True];
);


$endOfBatch := (
	$batch++;
	$absoluteBatch++;
);


$initiateTrainerUpdate := UseMacros[
	$CoreLoopLogger["TrainerUpdate"];
	If[$GradientLogger =!= Hold, $wlast = $wcurr];
	timeInto[$executorTimings,
		TrainerUpdate[$trainer, $nextDataBatch]];
	$GradientLogger[$gradientAndUpdateMagnitudes];
];	


$fetchDataFromGenerator := UseMacros[
	$CoreLoopLogger["FetchDataFromGenerator"];
	If[$absoluteBatch == $maxBatches, Goto[skipPrefetch]];
	If[$batch === $batchesPerRound && $batchesPerRound > 4, 
		(* at the end of the training round, randomize the visit order of batches *)
		$batchperm = RandomSample[$batchperm];
	];
	(* obtain the next data batch while the net is still busy with the current batch *)
	timeInto[$generatorTimings,
		$nextIndex = $batchperm[[Mod[$batch + 2, $batchesPerRound, 1]]];
		$nextDataBatch = $trainingGenerator @ $nextIndex;
	];
	Label[skipPrefetch];
];


$obtainLossFromTrainer := UseMacros[
	$CoreLoopLogger["ObtainingLoss"];
	timeInto[$syncTimings, 
		If[$forceSync, NDArrayWaitForAll[]];
		$rawBatchLosses = Mean /@ TrainerCurrentLoss[$trainer];
		$rawBatchLosses["TotalLoss"] = Total @ $rawBatchLosses;
		(* ^ get an association of losses, but meaned across the batch dim *)
		If[$collectPerExampleLosses, recordCurrentExampleLosses[]];
		$rawBatchMetrics = TrainerCurrentMetrics[$trainer];
	];
	$CoreLoopLogger["BatchCollectors"];
	timeInto[$collectorTimings, 
		List @@ $batchCollectors
	];
	(* ^ search NetTrain.m for calls to addBatchCollector and JoinTo[$batchCollectors, ...] for more info.
	at a minimum this includes code that obtains the loss from the net and adds it
	to the history bags *)
];


$calculateProgressAndTimings := (
	$batchTime = $now - $batchStartTime - $validationTimeComp;
	(* ^ if a validation happened, don't count this against the batch time *)
	If[$batchTime == 0, (* < see 321439 *)
		$batchTime = $absoluteBatch / ($now - $startTime + $MachineEpsilon)
		,
		$rawBatchRate = 1.0 / $batchTime;
		$batchRate = If[$batchRate === 0, $rawBatchRate, 0.9 * $batchRate + 0.1 * $rawBatchRate];
	];
	updateTimings[$batchTimings, $batchTime];

	$progressFraction = $rawProgressFraction;
);


$callPeriodicFunctions := UseMacros[
	
	$CoreLoopLogger["CallPeriodicFunctions"];
	If[$isEndOfRound, $virtualRound++];
	(* ^ the only reason that $virtualRound exists and is different from $round is that we 
	need to bump it just before the functions that should actually trigger at the end of the
	round, so that $clock will increment and cause them to run. but we don't want them to 
	observe the round to be 1 greater than they expect. *)

	$validationPeriodicCallback[$clock];
	$checkpointPeriodicCallback[$clock];
	$earlyStoppingPeriodicCallback[$clock]; 

	(* we call the TP function after validation and checkpointing to ensure that
	the TPF gets access to the most recent validation numbers, and checkpoint files. if they
	are all sync'd to happen once every round (the default), we want this to happen *)
	timeInto[$callbackTimings, 
		$progressPeriodicCallback[$clock, $livePropertyData]];

	timeInto[$reportingTimings,
		$memoryUsagePeriodicCallback[$clock];
		$reportingPeriodicCallback[$clock]];	

	$timingPeriodicCallback[$clock];	
];

