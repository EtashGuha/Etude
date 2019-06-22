Package["NeuralNetworks`"]


(* Note: this is substituted into the definition of NetTrain using a MacroEvaluate *)
$internalOptionsSetup := Quoted[

	$updatesPerBatch = 1; $PlanLogger = $ExecutorLogger = $BucketingLogger = 
		$MXNetLogger = $MemoryUsageLogger = $GradientLogger = $PermutationLogger = 
		$TimingLogger = $CoreLoopLogger = $TrainerLogger = Hold;

	$reportGPUMemoryUsage = $reportTimings = $reportMemoryUsage = $forceSync = False;

	$DisableValidationExecutor = False;

	$memoryReportingInterval = 2;

	$LowPrecisionGradientRescaling = 5; 

	$SequenceBucketingPartitions = 8;

	$gradientNormFunction = Abs /* Max;

	parseInternalOptions[internalOptions, $internalOptionsHandlers];

	If[$GradientLogger =!= Hold,
		$wcurr := Map[NDArrayGetNoDiverge, Values @ $trainerArrays]; 
		$wlast = None;
		$gradientAndUpdateMagnitudes := <|
			"GradientMagnitudes" -> ToPackedArray @ Map[NDArrayGetNoDiverge /* $gradientNormFunction, Values @ $trainerGradients],
			"UpdateMagnitudes" -> ToPackedArray @ MapThread[$gradientNormFunction[#1 - #2]&, {$wcurr, $wlast}]
		|>;
	];	
];


parseInternalOptions[{}, _] := Null;

NetTrain::badintopt = "`` is not a valid internal option, valid options are: ``.";
parseInternalOptions[opts_, handlers_] := (
	If[!ListQ[opts], ThrowFailure["badintopt", opts, Keys[handlers]]];
	Do[
		If[!RuleQ[opt], ThrowFailure["badintopt", opt, Keys[handlers]]];
		Lookup[handlers, First[opt], ThrowFailure["badintopt", opt, Keys[handlers]]] @ Last[opt]
	, 
		{opt, opts}
	];
);

$verboseLogging = {
	"PlanLogging", "ExecutorLogging", "BucketingLogging",
	"MemoryUsageLogging", "PermutationLogging",
	"TimingLogging"
};


$internalOptionsHandlers = {
	"SequenceBucketingPartitions" -> Function[
		If[!IntegerQ[#] && # =!= None, ThrowFailure[]];
		$SequenceBucketingPartitions = #;
	],
	"UpdatesPerBatch" -> Function[
		If[!IntegerQ[#] || # < 0, ThrowFailure[]];
		$updatesPerBatch = #
	],
	"DisableValidationExecutor" -> Function[$DisableValidationExecutor = TrueQ[#]],
	"DisablePreEncoding" -> Function[$EnablePreEncoding = !TrueQ[#]],
	"PlanLogging" -> Function[$PlanLogger = ToLogger[#]],
	"ExecutorLogging" -> Function[$ExecutorLogger = ToLogger[#]],
	"BucketingLogging" -> Function[$BucketingLogger = ToLogger[#]],
	"MXNetLogging" -> Function[$MXNetLogger = ToLogger[#]],
	"MemoryUsageLogging" -> Function[$MemoryUsageLogger = ToLogger[#]],
	"CoreLoopLogging" -> Function[$CoreLoopLogger = ToLogger[#]],
	"TrainerLogging" -> Function[$TrainerLogger = ToLogger[#]],
	"VerboseLogging" -> Function[value, Map[Lookup[$internalOptionsHandlers, #][value]&, $verboseLogging]],
	"ReportMemoryUsage" -> Function[
		$reportMemoryUsage = TrueQ[#];
		$reportGPUMemoryUsage = Max[$DefaultContext] > 1;
		If[$reportMemoryUsage, $lastMemoryUsageInfoString = MemoryUsageInfoString[]] (* library is slow to load *)
	],
	"ReportTimings" -> Function[$reportTimings = TrueQ[#]],
	"MemoryReportingInterval" -> Function[
		$memoryReportingInterval = Which[NumberQ[#], N[#], QuantityQ[#], #, True, 2]
	],
	"ForceSynchronization" -> Function[$forceSync = TrueQ[#]],
	"TimingLogging" -> Function[$TimingLogger = ToLogger[#]],
	"PermutationLogging" -> Function[$PermutationLogger = ToLogger[#]],
	"GradientLogging" -> Function[$GradientLogger = ToLogger[#]],
	"GradientNormFunction" -> Function[$gradientNormFunction = #],
	"LowPrecisionGradientRescaling" -> Function[$LowPrecisionGradientRescaling = #]
};
