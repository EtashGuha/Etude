BeginPackage["Compile`Core`Optimization`OptimizeBasicBlocks`"]

OptimizeBasicBlocksPass

Begin["`Private`"] 

Needs["CompileUtilities`Debug`Logger`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`Optimization`FuseBasicBlocks`"]
Needs["Compile`Core`Optimization`JumpThreading`"]
Needs["CompileUtilities`Callback`"]


run[fm_, opts_] := fm (**< This pass does nothing, the sequence of required passes do the work *)

(**********************************************************)
(**********************************************************)
(**********************************************************)

RegisterCallback["RegisterPass", Function[{st},
logger = CreateLogger["OptimizeBasicBlocks", "TRACE"];

info = CreatePassInformation[
	"OptimizeBasicBlocks",
	"Optimizes the basic blocks by deleting and removing redundancies.",
	""
];

OptimizeBasicBlocksPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
		  JumpThreadingPass
		, FuseBasicBlocksPass
		, JumpThreadingPass (**< TODO: Think about if we get any real benifit from call it again. *) 
	}
|>];

RegisterPass[OptimizeBasicBlocksPass]
]]

End[] 


EndPackage[]
