Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["MXExecutorForward"]

SetUsage @ "
MXExecutorForward[MXExecutor[$$], trainMode$] does a forward pass, modifying the output NDArrays of the computation graph.
MXExecutorForward[MXExecutorData[$$]] is equivalent to MXExecutorForward[MXExecutor[$$]]. trainMode$ is True \
if intermediate results in the forward pass need to be memorized, and False if not.";
		
mxlDeclare[mxlMXExecutorForward, {"Integer", "Boolean"}];

MXExecutorForward[executor_MXExecutor, trainMode_:False] :=
	mxlCall[mxlMXExecutorForward, MLEID @ executor, TrueQ @ trainMode];

MXExecutorForward[data_MXExecutorData, trainMode_:False] := 
	MXExecutorForward[data["Executor"], trainMode];

MXExecutorForward[__] := $Unreachable;

(******************************************************************************)

PackageExport["MXExecutorBackward"]

SetUsage @ "
MXExecutorBackward[MXExecutor[$$]] does a backward pass to get the gradients. It assumes no \
output gradient arrays need to be specified to obtain the gradients (ie. MXNet loss functions are used \
at the end of networks), and that MXExecutorForward[MXExecutor[$$], True] has already been evaluated.
MXExecutorBackward[MXExecutor[$$], outputGrads$] does a backward pass with a list of explicit \
output gradients outputGrads$ in the same order as they appear in the executor.
MXExecutorBackward[MXExecutorData[$$]] is equivalent to MXExecutorBackward[MXExecutor[$$]].";

mxlDeclare[mxlMXExecutorBackward, {"Integer", "IntegerVector"}];

MXExecutorBackward[executor_MXExecutor, outGrads_] := 
	mxlCall[
		mxlMXExecutorBackward, 
		MLEID @ executor, 
		Which[
			outGrads === None, {},
			ListQ[outGrads], Map[MLEID, outGrads],
			NDArrayQ[outGrads], {MLEID @ outGrads},
			True, Panic["InvalidOutGrads"]
		]
	];

MXExecutorBackward[data_MXExecutorData, outGrads_] := 
	MXExecutorBackward[data["Executor"], outGrads]

MXExecutorBackward[executor_MXExecutor] := MXExecutorBackward[executor, {}]
MXExecutorBackward[executor_MXExecutorData] := MXExecutorBackward[executor, {}]

MXExecutorBackward[___] := $Unreachable;
