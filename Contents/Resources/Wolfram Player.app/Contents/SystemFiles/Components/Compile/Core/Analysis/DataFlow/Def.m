BeginPackage["Compile`Core`Analysis`DataFlow`Def`"]

DefPass;

Begin["`Private`"] 

Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`Transform`TopologicalOrderRenumber`"]
Needs["CompileUtilities`Callback`"]



clearDef[var_] :=
	var["clearDef"]

initialize[fm_, opts_] :=
	CreateInstructionVisitor[
		<|
			"visitInstruction" -> Function[{st, inst},
			    If[inst["definesVariableQ"],
			    		clearDef[inst["definedVariable"]]
			    ];
			    clearDef /@ inst["definedVariable"]
			]
		|>,
		fm
	]

run[fm_, opts_] :=
	CreateInstructionVisitor[
		<|
			"visitInstruction" -> Function[{st, inst},
				If[inst["definesVariableQ"],
			    		inst["definedVariable"]["setDef", inst]
			    ]
			]
		|>,
		fm
	]


(**********************************************************)
(**********************************************************)
(**********************************************************)



RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"Def",
	"For each program variable, the pass will compute the set of instructions that may define the variable as well as the ones that may use it.",
	"This pass compute the def chain of a variable. The input is in SSA form, so the def set is one element. " <>
	"If the variable is global, then the def set is empty."
];

DefPass = CreateFunctionModulePass[<|
	"information" -> info,
	"initializePass" -> initialize,
	"runPass" -> run,
	"requires" -> {
		TopologicalOrderRenumberPass
	},
	"traversalOrder" -> "reversePostOrder",
	"passClass" -> "Analysis"
|>];

RegisterPass[DefPass]
]]


End[] 

EndPackage[]
