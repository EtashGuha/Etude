Package["NeuralNetworks`"]


PackageExport["NetTrainResultsObject"]

$resultsObjectKeys = Sort @ {
	"BatchLossList", 
	"RoundLoss", "RoundLossList",
	
	"ValidationLoss", "ValidationLossList", 
	
	"ValidationMeasurements", "ValidationMeasurementsLists",
	"RoundMeasurements", "RoundMeasurementsLists",
	"BatchMeasurements", "BatchMeasurementsLists",

	"CheckpointingFiles",
	"InitialLearningRate", "FinalLearningRate",
	"InternalVersionNumber", 
	"LossPlot", "FinalPlots",
		
	"BatchesPerRound", "BatchSize", "TotalBatches", "TotalRounds", "TotalTrainingTime", "TargetDevice",
	"ExamplesProcessed", "TrainingExamples", "ValidationExamples", "MeanBatchesPerSecond", "MeanExamplesPerSecond",
	"SkippedTrainingData",

	"TrainedNet", "TrainingNet",
	
	"NetTrainInputForm", "OptimizationMethod",
	"ReasonTrainingStopped", 
	"BestValidationRound",
	"WeightsLearningRateMultipliers", 

	"ValidationPositions", "RoundPositions",

	"$MeasurementsInfo"
};

makeNetTrainResultsObject[] := NetTrainResultsObject @ 
	AssociationThread[
		Append[$resultsObjectKeys, "Properties"],
		Append[Lookup[$returnSpecFunctions, $resultsObjectKeys], Most @ $resultsObjectKeys]
	];

RunInitializationCode[

	DefineCustomBoxes[NetTrainResultsObject,
		NetTrainResultsObject[assoc_Association /; VersionOrder[
			Lookup[assoc, "InternalVersionNumber", Lookup[assoc, "VersionNumber"]], 
			$NeuralNetworksVersionNumber] > -1] :> 
			MakeNetTrainResultsObjectBoxes[assoc]
	];

	Format[NetTrainResultsObject[assoc_Association], OutputForm] := 
		"NetTrainResultsObject[<>]"
]

ntroSubTable[list_List] := Scope[
	row = Replace[list, {e_List :> Row[e, " "], Null -> Nothing}, {1}];
	Row[row, ","]
]

ntroEntry[k_ -> list_List] := k -> ntroSubTable[list];
ntroEntry[e_] := e;

MakeNetTrainResultsObjectBoxes[assoc_] := Scope[
	UnpackAssociation[assoc,
		totalRounds, totalBatches, totalTrainingTime, 
		trainingExamples, validationExamples, examplesProcessed, meanExamplesPerSecond,
		optimizationMethod, batchSize, targetDevice,
		roundMeasurements, validationMeasurements,
		finalPlots, measurementsInfo:"$MeasurementsInfo",
		skippedTrainingData
	];
	plotNames = Table[
		Apply[measurementTooltip] @ Function[{#, #LongName}] @ SelectFirst[measurementsInfo, #Key === key&],
		{key, Keys @ finalPlots}
	];
	details = ntroEntry /@ {
		"summary" -> {
			If[IntegerQ @ trainingSetSize, {"examples:", trainingSetSize}],
			{"batches:", totalBatches},
			{"rounds:", totalRounds},
			{"time:", StringDelete[timeUnitStr @ totalTrainingTime, " "]},
			{"examples/s:", rateForm @ meanExamplesPerSecond}
		},
		"data" -> {
			If[IntegerQ @ trainingExamples, {"training examples:", trainingExamples}],
			If[IntegerQ @ validationExamples, {"validation examples:", validationExamples}],
			{"processed examples:", examplesProcessed}, {"skipped examples:", skippedTrainingData["SkippedExamples"]}
		},
		"method" -> {
			{optimizationMethod, "optimizer"},
			{"batch size", batchSize},
			targetDevice
		},
		"round" -> ntroMeasure[roundMeasurements, measurementsInfo],
		If[IntegerQ @ validationExamples,
		"validation" -> ntroMeasure[validationMeasurements, measurementsInfo], Nothing],
		Center -> makeStaticPlotFlipper[plotNames, Map[stripLegend, Values @ finalPlots]],
		Center -> If[IntegerQ @ validationExamples, RawBoxes[$legend], ""]
	}; 
	ToBoxes @ First @ First @ InformationPanel[
		"NetTrain Results", details,
		LineBreakWithin -> True,
		ColumnWidths -> {8, 26}
	]
];

(* this ugly mechanism exits because FinalPlots are formatted for users
to do other stuff with, and so are stripped of their dynamic features, but
NTRO actually needs to reconstruct these for nice feeling *)

stripLegend[Legended[e_, ___]] := convertColumn @ e;
stripLegend[e_] := e;

convertColumn[Column[{a_, b_}, ___]] := clickFlip[a, b];
convertColumn[e_] := e;


Clear[makeStaticPlotFlipper];
makeStaticPlotFlipper[labels_, {plot_}] := plot;
makeStaticPlotFlipper[labels_, plots_List] := 
	DynamicModule[
		{index = 1, plotList = stripLegend /@ plots},
		Column[{

			RawBoxes @ makeLeftRightClickerBoxes[index, labels],
			Dynamic[plotList[[index]], TrackedSymbols :> {index}]
		}, Alignment -> Center]
	]

ntroMeasure[data_, info_] := ntroMeasureRow[data, #]& /@ info;

ntroMeasureRow[_, assoc_] /; assoc["TextFormat"] === None := Null
ntroMeasureRow[data_, assoc_] := Scope[
	key = assoc["Key"];
	textFormat = Replace[assoc["TextFormat"], {errForm -> percentForm, f_ :> f /* lossForm}];
	preFormat = applyPre @ assoc["PreFormat"];
	{measurementTooltip[assoc, assoc["ShortName"] <> ":"], textFormat @ preFormat @ Lookup[data, key, None]}
];

timeUnitStr[n_Real] :=
	TextString @ NumberForm[#, 2]& @ Which[
		n > 60*60, Quantity[n / 60 / 60, "Hours"],
		n > 60, Quantity[n / 60, "Minutes"],
		n > 10, Quantity[Ceiling @ n, "Seconds"],
		True, Quantity[n, "Seconds"]
	] // StringReplace[". " -> " "];

NetTrainResultsObject /: Normal[NetTrainResultsObject[assoc_Association]] := assoc;

NetTrainResultsObject /: Part[NetTrainResultsObject[assoc_Association], key:(_String | {__String} | All), rest___] := 
	Part[assoc, Replace[key, $resultsObjectPropertyAliases], rest];
	(* ^ see Properties.m for the definition of $resultsObjectPropertyAliases *)

NetTrainResultsObject[assoc_Association][All, rest___] := 
	Part[KeyDrop[assoc, "$MeasurementsInfo"], Sequence @@ ReplaceAll[{rest}, $resultsObjectPropertyAliases]];

NetTrainResultsObject[assoc_Association][key_ /; StringQ[key] || StringVectorQ[key], rest___] := 
	Part[Lookup[assoc, ReplaceAll[key, $resultsObjectPropertyAliases]], rest];
