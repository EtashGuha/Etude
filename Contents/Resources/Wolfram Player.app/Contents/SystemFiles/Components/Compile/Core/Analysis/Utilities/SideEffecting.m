
BeginPackage["Compile`Core`Analysis`Utilities`SideEffecting`"]

SideEffectingAnalysisPass;

Begin["`Private`"] 

Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`IR`Instruction`InertInstruction`"]
Needs["Compile`Core`IR`Instruction`LoadGlobalInstruction`"]
Needs["CompileAST`Class`Symbol`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`Transform`TopologicalOrderRenumber`"]
Needs["Compile`Core`Analysis`DataFlow`Def`"]
Needs["Compile`Utilities`Language`SideEffecting`"]
Needs["Compile`Core`IR`Instruction`CallInstruction`"]
Needs["CompileUtilities`Callback`"]



initialize[fm_, opts_] := (
	fm["removeProperty", "sideEffecting"];
	fm["scanBasicBlocks",
		Function[{bb},
			bb["removeProperty", "sideEffecting"]
		]
	];
)

runFM[fm_, opts_] :=
	fm["reversePostOrderScan", runBB[fm, #]&]
runBB[fm_, bb_] :=
	Which[
		bb["hasProperty", "sideEffecting"],
			Nothing,
		AnyTrue[bb["getParents"], #["hasProperty", "sideEffecting"]&],
			bb["setProperty", "sideEffecting" -> True],
		True,
			bb["scanInstructions", runInst[fm, bb, #]&]
	]
runInst[fm_, bb_, inst_] :=
	Module[{var, def, src},
		var = Which[
			bb["hasProperty", "sideEffecting"],
				$Failed,
			CallInstructionQ[inst],
				inst["function"],
			InertInstructionQ[inst],
				inst["head"],
			True,
				$Failed
		];
		If[!FailureQ[var],
			def = var["def"];
			If[LoadGlobalInstructionQ[def],
				src = def["source"];
				If[sideEffectingQ[src],
					fm["setProperty", "sideEffecting" -> True];
					bb["setProperty", "sideEffecting" -> True]
				]
			]
		]
	]
sideEffectingQ[src_?MExprSymbolQ] :=
	sideEffectingQ[src["fullName"]]
sideEffectingQ[src_?StringQ] :=
	MemberQ[$SystemSideEffectNames, StringTrim[src, "System`"]]
	
(**********************************************************)
(**********************************************************)
(**********************************************************)




RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"SideEffectingAnalysis",
	"Sets a sideEffecting property to a function module if invoking it " <>
	"causes a side effect."
];

SideEffectingAnalysisPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> runFM,
	"initializePass" -> initialize,
	"requires" -> {
		TopologicalOrderRenumberPass,
		DefPass
	}
|>];

RegisterPass[SideEffectingAnalysisPass]
]]



End[] 

EndPackage[]
