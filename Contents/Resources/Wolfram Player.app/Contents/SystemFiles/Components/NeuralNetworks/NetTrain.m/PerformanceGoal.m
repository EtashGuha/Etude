Package["NeuralNetworks`"]

(* See MXNets https://github.com/apache/incubator-mxnet/blob/master/docs/faq/env_var.md 
	for more info about available environment variables 
*)

(* Note: this is substituted into the definition of NetTrain using a MacroEvaluate *)
$applyPerformanceGoal := Quoted[

	performanceGoal2 = If[ListQ[performanceGoal], performanceGoal, {performanceGoal}];
	If[!SubsetQ[{"TrainingSpeed", "TrainingMemory", Automatic}, performanceGoal2],
		NetTrain::perfgoal = "Value of option PerformanceGoal -> `1` is not \"TrainingSpeed\", \"TrainingMemory\", Automatic or a list of these.";
		ThrowFailure["perfgoal", performanceGoal] 
	];

	(* Environment Variables *)
	mxEnvVar = <||>;

	(* MXNET_BACKWARD_DO_MIRROR=1 will save 30%~50% of device memory, 
		but retains about 95% of running speed. *)

	(* NOTE: disabled for 12 due to https://github.com/apache/incubator-mxnet/issues/13592
		Reenable it when fixed.
	*)
(* 	If[MemberQ[performanceGoal2, "TrainingMemory"],
		mxEnvVar["MXNET_BACKWARD_DO_MIRROR"] = "1";
	]; *)

	(* Value of 1 chooses the best algo in a limited workspace (default)
  	Value of 2 chooses the fastest algo whose memory requirements may be larger 
  		than the default workspace threshold. Typically for convolutions. *)
	If[MemberQ[performanceGoal2, "TrainingSpeed"],
		mxEnvVar["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "2";
	];

	SetEnvironment @ Normal[mxEnvVar];
	AppendTo[$CleanupQueue, Hold[SetEnvironment][Thread[Keys[mxEnvVar] -> None]]];
]