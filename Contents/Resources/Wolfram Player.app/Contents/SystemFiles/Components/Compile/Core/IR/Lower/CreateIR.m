BeginPackage["Compile`Core`IR`Lower`CreateIR`"]

CreateIR;
CreateIRPass;
$OptimizationPasses;

Begin["`Private`"] 

Needs["Compile`Core`IR`Lower`MExpr`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`Optimization`DeadBranchElimination`"]
Needs["Compile`Core`Optimization`DeadCodeElimination`"]
Needs["Compile`Core`Optimization`EvaluateExpression`"]
Needs["Compile`Core`Optimization`FuseBasicBlocks`"]
Needs["Compile`Core`Debug`InsertDebugDeclarePass`"]
Needs["Compile`Core`Optimization`JumpThreading`"]
Needs["Compile`Core`Analysis`Properties`LastBasicBlock`"]
Needs["Compile`Core`Analysis`Function`FunctionInlineInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`Optimization`ConstantStackAllocation`"]
Needs["Compile`Core`Optimization`RemoveRedundantStackAllocate`"]
Needs["Compile`Core`Transform`ResolveFunctionCall`"]
Needs["Compile`Core`PassManager`PassRunner`"]
Needs["Compile`Core`Transform`TopologicalOrderRenumber`"]
Needs["Compile`Core`Optimization`CoerceCast`"]
Needs["Compile`Core`Optimization`ConstantPropagation`"]
Needs["Compile`Core`Optimization`CopyPropagation`"]
Needs["Compile`Core`Optimization`PhiElimination`"]
Needs["Compile`Core`Transform`ResolveTypes`"]
Needs["Compile`Core`Transform`ResolveExternalDeclarations`"]
Needs["Compile`Core`Transform`BasicBlockPhiReorder`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`Transform`CreateList`"]
Needs["Compile`Core`Transform`MoveMutability`"]
Needs["Compile`Core`Transform`ProcessMutability`"]
Needs["Compile`TypeSystem`Inference`InferencePass`"]
Needs["Compile`Core`PassManager`CompiledProgramPass`"]
Needs["Compile`Core`IR`CompiledProgram`"]
Needs["Compile`Core`Analysis`Properties`ClosureVariablesProvided`"]
Needs["CompileUtilities`Error`Exceptions`"]
Needs["Compile`Core`Analysis`Properties`HasClosure`"]
Needs["Compile`Core`Transform`Closure`ResolveClosureVariableType`"]
Needs["Compile`Core`Transform`Closure`ResolveLambdaClosure`"]
Needs["Compile`Core`Lint`IR`"]
Needs["Compile`Core`Lint`Types`"]


$OptimizationPasses[None] := {
}

(*
  
*)
$OptimizationPasses[0] := {
	$OptimizationPasses[None]
	, ResolveExternalDeclarationsPass
	, ResolveTypesPass
	, HasClosurePass
	, ResolveClosureVariableTypePass
	, LintTypesPass
	, TopologicalOrderRenumberPass
	, CreateListPass
	, DeadCodeEliminationPass
	, InferencePass
	, BasicBlockPhiReorderPass
	, ProcessMutabilityPass
	, MoveMutabilityPass
	, ResolveFunctionCallPass
	, FuseBasicBlocksPass
	, PhiEliminationPass
	, CopyPropagationPass
	, ResolveLambdaClosurePass
	, LastBasicBlockPass
	, ConstantStackAllocationPass
	, FunctionInlineInformationPass

	(*, LiveVariablesPass*)
	, LintIRPass
}

$OptimizationPasses[1] := {
	$OptimizationPasses[0]
	, DeadBranchEliminationPass
	, JumpThreadingPass
	, FuseBasicBlocksPass
	, LintIRPass
}

$OptimizationPasses[2] := {
	$OptimizationPasses[1]
	, RemoveRedundantStackAllocatePass
	, CopyPropagationPass
	, ConstantPropagationPass
	, EvaluateExpressionPass
	, DeadCodeEliminationPass
	, DeadBranchEliminationPass
	, ConstantPropagationPass
	, DeadCodeEliminationPass
	, CoerceCastPass
	, CopyPropagationPass
	, ConstantPropagationPass
	, DeadCodeEliminationPass
	, DeadBranchEliminationPass
	, LintIRPass
}

$OptimizationPasses[3] := {
	$OptimizationPasses[2],
	$OptimizationPasses[2]
}

$DefaultOptimizationLevel = 1
Options[CreateIR] := Options[CreateIR] = {
	"OptimizationLevel" -> $DefaultOptimizationLevel,
	"EnvironmentOptions" -> <||>
}

CreateIR[cp_?CompiledProgramQ, opts:OptionsPattern[]] :=
	CreateIR[cp, <| opts |>]
	
CreateIR[cp_?CompiledProgramQ, opts_?AssociationQ] :=
	Module[{pm, passes, envOptions},
		envOptions = Lookup[opts, "EnvironmentOptions", <||>];
		passes = getOptimizationLevel[opts];

		pm = RunPass[ MExprLowerPass, cp, opts];		

		(* Add the Lint IR pass regardless of the optimization level.
		 * The printout can be silenced using Quiet.
		 *)
		RunPasses[
			{
				ClosureVariablesProvidedPass,
				LintIRPass
			},
			pm,
			opts
		];

		If[TrueQ[Lookup[opts, "LLVMDebug", False]],
			RunPass[InsertDebugDeclarePass, pm, opts];
		];

		pm["setProperty", "environmentOptions" -> envOptions];
		RunPasses[passes, pm, <|
				"PassOptions" -> Lookup[ opts, "PassOptions", {}],
				"PassLogger" -> Lookup[ opts, "PassLogger", Automatic]|>]
	]

CreateIR[args___] :=
    ThrowException[{"Invalid call to CreateIR.", {args}}]

getOptimizationLevel[opts_?AssociationQ] :=
	With[{level = Lookup[opts, "OptimizationLevel", $DefaultOptimizationLevel]},
		If[ListQ[level],
			level,
			$OptimizationPasses[level]
		]
	]

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"CreateIR",
	"Converts the MExpr AST into Wolfram IR."
];

CreateIRPass = CreateCompiledProgramPass[<|
	"information" -> info,
	"runPass" -> CreateIR
|>];

RegisterPass[CreateIRPass]
]]

End[] 

EndPackage[]
