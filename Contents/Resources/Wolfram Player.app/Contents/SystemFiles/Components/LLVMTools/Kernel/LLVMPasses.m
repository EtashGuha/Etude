BeginPackage["LLVMTools`LLVMPasses`"]



Begin["`Private`"]

Needs["LLVMLink`"]
Needs["LLVMTools`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)


LLVMRunPasses::unknownpass = "Unknown pass '`1`'";

LLVMRunPasses[LLVMModule[moduleRef_], passes:{_LibraryFunction...}] := Module[{passRef, res},
	passRef = LLVMLibraryFunction["LLVMCreatePassManager"][];
	Do[
		(* TODO: Perhaps define LLVMLibraryFunction[x_] = Null, or some error reporting code? *)
		If[pass[passRef] =!= Null,
			Message[LLVMRunPasses::unknownpass, pass]
		],
		{pass, passes}
	];

	res = LLVMLibraryFunction["LLVMRunPassManager"][passRef, moduleRef];
	LLVMLibraryFunction["LLVMDisposePassManager"][passRef];

	(* Returns True if any passes modified the module, False otherwise *)
	res == 1
];

LLVMRunInstructionProfilingPass[LLVMModule[moduleRef_]] := Module[{passManagerRef, res},
	(* Create a pass manager *)
	passManagerRef = LLVMLibraryFunction["LLVMCreatePassManager"][];
	(* Add the InstrProf pass to the pass manager *)
	LLVMLibraryFunction["LLVMWLAddInstrProfPass"][passManagerRef];
	(* Run the pass manager on the moduel *)
	res = LLVMLibraryFunction["LLVMRunPassManager"][passManagerRef, moduleRef];
	LLVMLibraryFunction["LLVMDisposePassManager"][passManagerRef];

	(* Returns True if any passes modified the module, False otherwise *)
	res == 1
];

LLVMRunPassManagerOptPasses[LLVMModule[moduleRef_], optLevel_] := Module[{builderRef, managerRef, res},
	builderRef = LLVMLibraryFunction["LLVMPassManagerBuilderCreate"][];
	LLVMLibraryFunction["LLVMPassManagerBuilderSetOptLevel"][builderRef, optLevel];

	managerRef = LLVMLibraryFunction["LLVMCreatePassManager"][];
	LLVMLibraryFunction["LLVMPassManagerBuilderPopulateModulePassManager"][builderRef, managerRef];

	res = LLVMLibraryFunction["LLVMRunPassManager"][managerRef, moduleRef];

	LLVMLibraryFunction["LLVMDisposePassManager"][managerRef];
	LLVMLibraryFunction["LLVMPassManagerBuilderDispose"][builderRef];

	(* Returns True if any passes modified the module, False otherwise *)
	res == 1
];

LLVMRunOptimizationPasses[mod_LLVMModule, OptionsPattern[]] := Module[{},
	Replace[OptionValue["OptimizationLevel"], {
		"ClangOptimization"[level_Integer] :> LLVMRunPasses[mod, LLVMOptimizationPasses[level]]
	}];
]
LLVMRunOptimizationPasses[args___] :=
    ThrowException[{"Unrecognized call to LLVMRunOptimizationPasses ", {args}}]

(* An arbitrary default, None or 3 could be better options. *)
Options[LLVMRunOptimizationPasses] = {"OptimizationLevel" -> "ClangOptimization"[2]}

LLVMOptimizationPasses[0] = {};
LLVMOptimizationPasses[1] := LLVMOptimizationPasses[1] = Join[
	LLVMOptimizationPasses[0],
	{
		LLVMLibraryFunction["LLVMAddAggressiveDCEPass"],
		LLVMLibraryFunction["LLVMAddAlignmentFromAssumptionsPass"],
		LLVMLibraryFunction["LLVMAddAlwaysInlinerPass"],
		LLVMLibraryFunction["LLVMAddBasicAliasAnalysisPass"],
		LLVMLibraryFunction["LLVMAddBitTrackingDCEPass"],
		LLVMLibraryFunction["LLVMAddCorrelatedValuePropagationPass"],
		LLVMLibraryFunction["LLVMAddDeadArgEliminationPass"],
		LLVMLibraryFunction["LLVMAddDeadStoreEliminationPass"],
		LLVMLibraryFunction["LLVMAddEarlyCSEPass"],
		LLVMLibraryFunction["LLVMAddFunctionAttrsPass"],
		LLVMLibraryFunction["LLVMAddGlobalOptimizerPass"],
		LLVMLibraryFunction["LLVMAddIndVarSimplifyPass"],
		LLVMLibraryFunction["LLVMAddInstructionCombiningPass"],
		LLVMLibraryFunction["LLVMAddIPConstantPropagationPass"],
		LLVMLibraryFunction["LLVMAddJumpThreadingPass"],
		LLVMLibraryFunction["LLVMAddLoopDeletionPass"],
		LLVMLibraryFunction["LLVMAddLoopIdiomPass"],
		LLVMLibraryFunction["LLVMAddLoopRotatePass"],
		LLVMLibraryFunction["LLVMAddLoopUnrollPass"],
		LLVMLibraryFunction["LLVMAddLoopUnswitchPass"],
		LLVMLibraryFunction["LLVMAddLoopVectorizePass"],
		LLVMLibraryFunction["LLVMAddPromoteMemoryToRegisterPass"],
		LLVMLibraryFunction["LLVMAddMemCpyOptPass"],
		LLVMLibraryFunction["LLVMAddPruneEHPass"],
		LLVMLibraryFunction["LLVMAddReassociatePass"],
		LLVMLibraryFunction["LLVMAddSCCPPass"],
		LLVMLibraryFunction["LLVMAddScopedNoAliasAAPass"],
		LLVMLibraryFunction["LLVMAddCFGSimplificationPass"],
		LLVMLibraryFunction["LLVMAddScalarReplAggregatesPass"],
		LLVMLibraryFunction["LLVMAddScalarReplAggregatesPassSSA"],
		LLVMLibraryFunction["LLVMAddStripDeadPrototypesPass"],
		LLVMLibraryFunction["LLVMAddTailCallEliminationPass"],
		LLVMLibraryFunction["LLVMAddTypeBasedAliasAnalysisPass"],
		LLVMLibraryFunction["LLVMAddVerifierPass"],
		LLVMLibraryFunction["LLVMAddLICMPass"] (* Suggested by https://llvm.org/docs/Frontend/PerformanceTips.html#id15 *)
	}
];

LLVMOptimizationPasses[2] := LLVMOptimizationPasses[2] = Join[
	LLVMOptimizationPasses[1],
		{
			LLVMLibraryFunction["LLVMAddConstantMergePass"],
			LLVMLibraryFunction["LLVMAddGlobalDCEPass"],
			LLVMLibraryFunction["LLVMAddGVNPass"],
			LLVMLibraryFunction["LLVMAddFunctionInliningPass"],
			LLVMLibraryFunction["LLVMAddMergedLoadStoreMotionPass"],
			LLVMLibraryFunction["LLVMAddSLPVectorizePass"]
		}
];

LLVMOptimizationPasses[3] := LLVMOptimizationPasses[3] = Join[
	LLVMOptimizationPasses[2],
	{
		LLVMLibraryFunction["LLVMAddArgumentPromotionPass"]
	}
];

LLVMOptimizationPasses["s"] := LLVMOptimizationPasses["s"] = LLVMOptimizationPasses[2];
LLVMOptimizationPasses["z"] := LLVMOptimizationPasses["z"] = Select[LLVMOptimizationPasses[2], # != LLVMLibraryFunction["LLVMAddSLPVectorizePass"]&];


End[]

EndPackage[]
