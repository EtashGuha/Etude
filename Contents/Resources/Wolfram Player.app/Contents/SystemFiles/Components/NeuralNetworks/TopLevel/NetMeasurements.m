Package["NeuralNetworks`"]


PackageExport["NetMeasurements"]

Clear[NetMeasurements]

Options[NetMeasurements] = {
	BatchSize -> Automatic,
	TargetDevice -> "CPU",
	WorkingPrecision -> Automatic,
	NetEvaluationMode -> "Test",
	LossFunction -> Automatic
};

(* TODO: add ProgressReporting option here, when available *)

NetMeasurements::invnet = "First argument to NetMeasurements should be a fully specified net.";


PackageScope["ClearNetMeasurementsCache"]

ClearNetMeasurementsCache[] := (
	$NetMeasurementsCache = <||>;
);

$NetMeasurementsCache = <||>;

NetMeasurements[net_, data_, metrics_, opts:OptionsPattern[]] := CatchFailureAsMessage @ Scope[

	GeneralUtilities`$ComputeWithProgressEnabled = !TrueQ[$NNTestingMode];
	$LastGeneratorData = None;

	metrics = If[ListQ[metrics], removeList = False; metrics, removeList = True; {metrics}];
	netInformationData = AssociationThread[Range[Length[metrics]], metrics] // Select[MemberQ[$NetModelInformationKeys, #] &];
	netInformationData = netInformationData // Map[NetInformation[net, #]&] // Normal;
	metrics = DeleteCases[metrics, Alternatives @@ $NetModelInformationKeys];
	(* ^ if any NetInformation properties are passed to NetMeasurements then call NetInformation to get the appropriate info
	and remove these properties from the list of metrics. 
	We want to have an association from pos -> property so that we can return the requested properties & measurements in the correct order.
	removeList is there to make sure that if the user requested a list of measurements we return a list otherwise we return a scalar.
	*)

	If[metrics === {}, Return[Values @ netInformationData // If[removeList, First, Identity]]];
	
	If[!ValidNetQ[net], ThrowFailure["invnet"]];	

	If[!InitializedNetQ[net], ThrowNotSpecifiedFailure[net, "measure", "Initialized"]];

	net = NData[net];

	UnpackOptions[batchSize, workingPrecision, targetDevice, netEvaluationMode, lossFunction];

	If[!PositiveMachineIntegerQ[batchSize] && batchSize =!= Automatic,
		ThrowFailure["optpi", BatchSize, batchSize];
	];

	mixedPrecisionQ = If[workingPrecision === "Mixed", workingPrecision = "Real32"; True, False];
	dataType = $DefaultDataTypeCode = ParseWorkingPrecision[workingPrecision];
	context = $DefaultContext = ParseMultiContext[targetDevice];
	tmode = ParseEvalMode[netEvaluationMode];

	SetAutomatic[batchSize, 32];

	metricPorts = GetUserMeasurementPorts[OutputNames[net], metrics];

	{net, ports, hasPrefix} = AttachLoss[net, lossFunction, metricPorts]; 

	measurementsInfo = ParseMeasurementSpecs[net, metrics, hasPrefix];
	
	measurementPaths = measurementsInfo[[All, "Path"]];

	inputs = Inputs[net];
	data = TryPack[data];

	dataFailure = CatchFailure[NetTrain, 
		{generator, length} = ParseTrainingData[data, inputs, batchSize, ConstructValidationGenerator];
	];
	If[FailureQ[dataFailure], ReissueMessage[dataFailure, NetMeasurements]; ReturnFailed[]];
	(* ^ TODO: clean this up by making the message names more specific so that they can
	live on General instead of just on NetTrain *)

	cacheKey = {NetUUID[net], Hash[data], Hash[{opts}]};
	cacheVar = CacheTo[$NetMeasurementsCache, cacheKey, Module[{var = <||>}, Hold[var]]];

	measurementPaths = Complement[measurementPaths, Keys @ ReleaseHold @ cacheVar];
	(* ^ avoid asking execturo to perform measurements we already have taken *)

	If[measurementPaths === {}, Goto[SkipMakingExecutor]];

	bucketed = ContainsVarSequenceQ[inputs];
	If[!bucketed,
		plan = ToNetPlan[net, {All, <||>, tmode, {Min[context] =!= 1, dataType, mixedPrecisionQ}, measurementPaths}];
		executor = ToNetExecutor[plan, batchSize, "Context" -> $DefaultContext, "ArrayCaching" -> False];
	,
		executor = ToBucketedNetExecutor[net, 
			{All, {context, dataType, mixedPrecisionQ}, batchSize, tmode, measurementPaths, {<||>, <||>, False, 1}, Automatic}
		];
	];

	Label[SkipMakingExecutor];

	{losses, measurements} = ComputeWithProgress[
		ExecutorMeasure[executor, generator, length, batchSize, {}, measurementsInfo, #, cacheVar]&,
		"Collecting measurement data"
	];

	measurements = Values @ measurements;

	Do[measurements = Insert[measurements, Last @ info, First @ info], {info, netInformationData}];
	(* ^ make sure that the measurements are returned in the requested order *)

	If[removeList, First @ measurements, measurements]	
];

DeclareArgumentCount[NetMeasurements, 3]