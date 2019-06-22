BeginPackage["LibraryLink`Examples`"]

CreateTimeDelayBackgroundTask::usage = "CreateTimeDelayBackgroundTask[delay, eventData, eventHandler]"

CreateRandomIntegerDataSourceBackgroundTask::usage = "CreateRandomIntegerDataSourceBackgroundTask[{imin, imax}, {n1, n2, ...}, eventHandler] creates an AsynchronousTask that generates an event with an n1 x n2 x ... array of pseudorandom integers in the range {imin, imax}."
(*
CreateRandomRealDataSourceBackgroundTask::usage = "CreateRandomRealDataSourceBackgroundTask[{imin, imax}, {n1, n2, ...}, eventHandler] creates an AsynchronousTask that generates an event with an n1 x n2 x ... array of pseudorandom reals in the range {imin, imax}."

CreateRandomComplexDataSourceBackgroundTask::usage = "CreateRandomComplexDataSourceBackgroundTask[{zmin, zmax}, {n1, n2, ...}, eventHandler] creates an AsynchronousTask that generates an event with an n1 x n2 x ... array of pseudorandom complex numbers in the rectangle with corners given by the complex numbers zmin and zmax."
*)

(* LibraryFunctions that start one-shot asynchronous tasks *)
startBackgroundTaskReturningInt
startBackgroundTaskReturningReal
startBackgroundTaskReturningComplex
startBackgroundTaskReturningMTensor

(* LibraryFunctions that start repeating asynchronous tasks *)
startRepeatingBackgroundTaskReturningInt
startRepeatingBackgroundTaskReturningReal
startRepeatingBackgroundTaskReturningComplex
startRepeatingBackgroundTaskReturningMTensor

(* LibraryFunctions for testing asynchronous tasks that start their own threads *)
startPrimaryBackgroundTaskReturningInt
startSecondaryBackgroundTaskReturningInt

(* LibraryFunctions for timing asynchronous task event generation *)
startTimingBackgroundTaskReturningInt
startTimingBackgroundTaskReturningReal

Begin["`Private`"]

CreateTimeDelayBackgroundTask[delayMillis_Integer, eventData_Integer, eventHandler_] := 
	Internal`CreateAsynchronousTask[startBackgroundTaskReturningInt, 
		{delayMillis, eventData}, eventHandler]

CreateTimeDelayBackgroundTask[delayMillis_Integer, eventData_Real, eventHandler_] := 
	Internal`CreateAsynchronousTask[startBackgroundTaskReturningReal, 
		{delayMillis, eventData}, eventHandler]

CreateTimeDelayBackgroundTask[delayMillis_Integer, eventData_Complex, eventHandler_] := 
	Internal`CreateAsynchronousTask[startBackgroundTaskReturningComplex, 
		{delayMillis, eventData}, eventHandler]

CreateTimeDelayBackgroundTask[delayMillis_Integer, eventData_List, eventHandler_] := 
	Internal`CreateAsynchronousTask[startBackgroundTaskReturningMTensor, 
		{delayMillis, eventData}, eventHandler]

Options[CreateRandomIntegerDataSourceBackgroundTask] = 
Options[CreateRandomRealDataSourceBackgroundTask] = 
Options[CreateRandomComplexDataSourceBackgroundTask] = 
	{UpdateInterval -> Automatic};

MTypeInteger=2
MTypeReal=3
MTypeComplex=4

CreateRandomIntegerDataSourceBackgroundTask[range:{imin_Integer, imax_Integer}, 
	dims_List, eventHandler_, OptionsPattern[]] := 
	With[{periodMillis = OptionValue[UpdateInterval] /. {Automatic -> 50, 
		t_ :> Round[1000*t]}}, 
		Internal`CreateAsynchronousTask[
			startRepeatingBackgroundTaskReturningIntegerMTensor, 
			{periodMillis, MTypeInteger, imin, imax, dims}, 
			eventHandler]
	]

CreateRandomRealDataSourceBackgroundTask[range:{rmin_Real, rmax_Real}, 
	dims_List, eventHandler_, OptionsPattern[]] := 
	With[{periodMillis = OptionValue[UpdateInterval] /. {Automatic -> 50, 
		t_ :> Round[1000*t]}}, 
		Internal`CreateAsynchronousTask[
			startRepeatingBackgroundTaskReturningRealMTensor, 
			{periodMillis, MTypeReal, rmin, rmax, dims}, 
			eventHandler]
	]

CreateRandomComplexDataSourceBackgroundTask[range:{zmin_Complex, zmax_Complex}, 
	dims_List, eventHandler_, OptionsPattern[]] := 
	With[{periodMillis = OptionValue[UpdateInterval] /. {Automatic -> 50, 
		t_ :> Round[1000*t]}}, 
		Internal`CreateAsynchronousTask[
			startRepeatingBackgroundTaskReturningComplexMTensor, 
			{periodMillis, MTypeComplex, zmin, zmax, dims}, 
			eventHandler]
	]

CreateContinuousDataSourceBackgroundTask[periodMillis_Integer, eventData_List, eventHandler_] := 
	Internal`CreateAsynchronousTask[startRepeatingBackgroundTaskReturningMTensor, 
		{periodMillis, eventData}, eventHandler]

LoadLibraries[] := 
With[{
	checkLoad = Function[{libname, function, argTypes, returnType},
		With[{
			lib =
			FileNameJoin[{
				DirectoryName[System`Private`$InputFileName],
				"LibraryResources",
				$SystemID,
				libname
			}]},
			LibraryFunctionLoad[lib, function, argTypes, returnType] /. {
				lf_LibraryFunction :> lf,
				other_ :> (
					AppendTo[$LoadErrors, {function, lib, other}]; 
					$Failed)
			}
		]
	]},

	$LoadErrors = {};

	startBackgroundTaskReturningInt = checkLoad["async-tasks-oneshot", 
		"start_int_background_task", {Integer, Integer}, Integer];
	startBackgroundTaskReturningReal = checkLoad["async-tasks-oneshot",
		"start_real_background_task", {Integer, Real}, Integer];
	startBackgroundTaskReturningComplex = checkLoad["async-tasks-oneshot",
		"start_complex_background_task", {Integer, Complex}, Integer];
	startBackgroundTaskReturningMTensor = checkLoad["async-tasks-oneshot",
		"start_mtensor_background_task", {Integer, {_, _, "Manual" }}, Integer];

	startRepeatingBackgroundTaskReturningInt = checkLoad["async-tasks-repeating",
		"start_int_repeating_background_task", {Integer, Integer}, Integer];
	startRepeatingBackgroundTaskReturningReal = checkLoad["async-tasks-repeating",
		"start_real_repeating_background_task", {Integer, Real}, Integer];
	startRepeatingBackgroundTaskReturningComplex = checkLoad["async-tasks-repeating",
		"start_complex_repeating_background_task", {Integer, Complex}, Integer];

	startRepeatingBackgroundTaskReturningIntegerMTensor = 
		checkLoad["async-tasks-repeating",
		"start_mtensor_repeating_background_task", 
		{(*period*) Integer, (*elt type*)Integer, 
			(*imin*) Integer, (*imax*)Integer, (*dims*){Integer, 1}},
		Integer];

	startRepeatingBackgroundTaskReturningRealMTensor = 
		Quiet[checkLoad["async-tasks-repeating",
			"start_mtensor_repeating_background_task", 
			{(*period*) Integer, (*elt type*)Integer, 
				(*rmin*) Real, (*rmax*)Real, (*dims*){Integer, 1}},
			Integer],
			LibraryFunction::overload];

	startRepeatingBackgroundTaskReturningComplexMTensor = 
		Quiet[checkLoad["async-tasks-repeating",
			"start_mtensor_repeating_background_task", 
			{(*period*) Integer, (*elt type*)Integer, 
				(*zmin*) Complex, (*zmax*)Complex, (*dims*){Integer, 1}},
			Integer],
			LibraryFunction::overload];

	startPrimaryBackgroundTaskReturningInt = checkLoad["async-tasks-without-thread",
		"start_primary_int_background_task", {Integer, Integer}, Integer];
	startSecondaryBackgroundTaskReturningInt = checkLoad["async-tasks-without-thread",
		"start_secondary_int_background_task", {Integer}, Integer];

	startTimingBackgroundTaskReturningInt = checkLoad["async-tasks-timing",
		"start_int_timing_background_task", {Integer,Integer}, Integer];
	startTimingBackgroundTaskReturningReal = checkLoad["async-tasks-timing",
		"start_real_timing_background_task", {Integer,Real}, Integer];

]

If[$Loaded === Unevaluated[$Loaded],
	LoadLibraries[];
	$Loaded = True;
]

End[]
EndPackage[]
