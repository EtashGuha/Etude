BeginPackage["Compile`Core`Analysis`DataFlow`Use`"]

UsePass;

Begin["`Private`"] 

Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`IR`Instruction`PhiInstruction`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`Transform`TopologicalOrderRenumber`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Instruction`BranchInstruction`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`Analysis`Dominator`DominatorPass`"]


clearUses[var_] :=
	var["clearUses"]


initialize[fm_, opts_] :=
	CreateInstructionVisitor[
		<|
			"visitInstruction" -> 
			Function[{st, inst},
			    If[inst["definesVariableQ"],
			    		clearUses[inst["definedVariable"]]
			    ];
			    clearUses /@ inst["usedVariables"];
			    Which[ 
			    	PhiInstructionQ[inst],
			    		Map[ #["clearUses"]&, inst["getSourceBasicBlocks"]],
			    	BranchInstructionQ[inst],
			    		If[inst["condition"] =!= None,
			    			inst["condition"]["clearUses"]
			    		];
			    		Map[ #["clearUses"]&, inst["operands"]],
			    	True,
			    		Null];
			]
		|>,
		fm
	]


run[fm_, opts_] :=
	CreateInstructionVisitor[
		<|
			"visitInstruction" -> 
				Function[{st, inst},
					If[ inst["hasOperands"],
						#["addUse", inst]& /@ Select[ inst["operands"], ConstantValueQ]
					];
					#["addUse", inst]& /@ inst["usedVariables"];
			    	Which[ 
			    		PhiInstructionQ[inst],
			    			Map[ #["addUse", inst]&, inst["getSourceBasicBlocks"]],
			    		BranchInstructionQ[inst],
                        	If[inst["condition"] =!= None,
                        		inst["condition"]["addUse", inst]
                        	];
			    			Map[ #["addUse", inst]&, inst["operands"]],
			    		True,
			    			Null
			    	];				
				]
		|>,
		fm
	]

(**********************************************************)
(**********************************************************)
(**********************************************************)



RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"Use",
	"For each program variable, the pass will compute the set of instructions that may define the variable as well as the ones that may use it.",
	"This pass compute the use chain of a variable. The input is assumed to be in SSA form"
];

UsePass = CreateFunctionModulePass[<|
	"information" -> info,
	"initializePass" -> initialize,
	"runPass" -> run,
	"requires" -> {
		TopologicalOrderRenumberPass,
		DominatorPass
	},
	"traversalOrder" -> "reversePostOrder",
	"passClass" -> "Analysis"
|>];

RegisterPass[UsePass]
]]

End[] 

EndPackage[]
