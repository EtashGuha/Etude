Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

MXExecutorData /: Normal[MXExecutorData[data_]] := data

MXExecutorData[data_][query__String] := data[query]

(******************************************************************************)

PackageExport["MXExecutorRequiredMemory"]

SetUsage @ "
MXExecutorRequiredMemory[MXExecutor[$$]] returns the maximum memory required to evaluate \
the executor MXExecutor[$$].
* MXExecutorRequiredMemory[MXExecutorData[$$]] is equivalent to MXExecutorRequiredMemory[MXExecutor[$$]]. 
";

MXExecutorRequiredMemory[executor_MXExecutor] := Scope[
	printed = MXExecutorPrint @ executor;
	If[FailureQ[printed], ReturnFailed[]];
	mem = StringCases[printed, 
		___ ~~ "Total " ~~ x : NumberString ~~ " MB allocated" ~~ ___ :> ToExpression[x]
	];
	If[Length[mem] === 0, ReturnFailed[]];
	First@mem
]

MXExecutorRequiredMemory[data_MXExecutorData] := 
	MXExecutorRequiredMemory[data["Executor"]]

(******************************************************************************)

PackageExport["MXExecutorMemoryInformation"]

MXExecutorMemoryInformation[exec:MXExecutorData[data_]] := Scope[
	sizes = AssociationMap[
		Total[Map[NDArrayByteCount, data[#]]]&,
		{"ArgumentArrays", "GradientArrays", "AuxilliaryArrays", "OutputArrays"}
	];
	sizes["InternalArrays"] = MXExecutorRequiredMemory[exec] * 2^20;
	sizes
];

(******************************************************************************)

PackageExport["MXExecutorPrint"]

SetUsage @ "
MXExecutorPrint[MXExecutor[$$]] returns a string containing the execution plan.
MXExecutorPrint[MXExecutorData[$$]] is equivalent to MXExecutorPrint[MXExecutor[$$]].";

mxlDeclare[mxlMXExecutorPrint, {"Integer"}, "String"];

MXExecutorPrint[executor_MXExecutor] := 
	CatchFailure @ mxlCall[mxlMXExecutorPrint, MLEID @ executor]

MXExecutorPrint[executor_MXExecutorData] := 
	MXExecutorPrint[executor["Executor"]]

