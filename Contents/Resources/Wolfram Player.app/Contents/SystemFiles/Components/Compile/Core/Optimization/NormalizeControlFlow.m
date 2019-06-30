BeginPackage["Compile`Core`Optimization`NormalizeControlFlow`"]

NormalizeControlFlowPass;

Begin["`Private`"] 

Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`Transform`TopologicalOrderRenumber`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Debug`Logger`"]
Needs["CompileUtilities`Callback`"]




run[fm_, opts_] := 0 (**< This pass does nothing, the sequence of required passes do the work *)

(**********************************************************)
(**********************************************************)
(**********************************************************)


RegisterCallback["RegisterPass", Function[{st},
logger = CreateLogger["NormalizeControlFlow", "TRACE"];

info = CreatePassInformation[
	"NormalizeControlFlow",
	"Normalizes the CFG so that basic blocks are indexed in topological order.",
	""
];

NormalizeControlFlowPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
		  PostTopologicalOrderRenumberPass
		(*
		, ImmediatePostDominatorPass
		, TopologicalOrderRenumberPass
		*)
	}
|>];

RegisterPass[NormalizeControlFlowPass]
]]

End[] 

EndPackage[]
