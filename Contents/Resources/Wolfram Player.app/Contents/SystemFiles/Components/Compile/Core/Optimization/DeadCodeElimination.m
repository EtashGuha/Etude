BeginPackage["Compile`Core`Optimization`DeadCodeElimination`"]
DeadCodeEliminationPass;

Begin["`Private`"] 

Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`IR`Instruction`InertInstruction`"]
Needs["Compile`Core`IR`Instruction`LoadArgumentInstruction`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`IR`Instruction`StackAllocateInstruction`"]
Needs["Compile`Core`IR`Instruction`StoreInstruction`"]
Needs["Compile`Core`Analysis`DataFlow`Use`"]
Needs["Compile`Core`IR`Instruction`CallInstruction`"]
Needs["CompileUtilities`Callback`"]



(** We do a VariableQ check so that we can make this code also work for registers
  * in the case of registers, we want to just skip the instruction
  *)

removeDeadLoads[fm_] :=
	Module[{toRemove},
		toRemove = Internal`Bag[];
		fm["scanInstructions",
			Function[{inst},
				If[!LoadArgumentInstructionQ[inst] &&
				   inst["definesVariableQ"] &&
				   VariableQ[inst["definedVariable"]] &&
				   !targetIsClosureCaptured[inst] &&
				   FreeQ[inst["getProperty", "live[out]"], inst["target"]],
					Internal`StuffBag[toRemove, inst]
				]
			]
		];
		#["unlink"]& /@ Internal`BagPart[toRemove, All]
	]


(*
 This is a CallInstruction or a InertInstruction. 
 Return True if we think we cannot remove the Instruction.
 Perhaps this should look at the function to see if it has side-effects.
*)
sideEffectCallInstructionQ[ inst_] :=
	CallInstructionQ[inst] || InertInstructionQ[inst]

(*
 A variable is captured if it is escapes the current function module scope.
 This is a property that is set when lowering a function module.
*)
targetIsClosureCaptured[inst_] :=
    inst["definedVariable"]["getProperty", "isCapturedVariable", False]

(*
  TODO maybe we should remove all CallInstructions here.
*)
iRun[fm_, opts_] :=
	Module[{changed = False, toRemove, uses},
		(* removeDeadLoads[fm]; *)
		toRemove = Internal`Bag[];
		CreateInstructionVisitor[
			<|
				"visitInstruction" -> Function[{st, inst},

					If[!LoadArgumentInstructionQ[inst] &&
					   inst["definesVariableQ"] &&
					   VariableQ[inst["definedVariable"]] &&
					   !sideEffectCallInstructionQ[ inst] &&
					   !targetIsClosureCaptured[inst],
						If[inst["definedVariable"]["uses"] === {},
							Internal`StuffBag[toRemove, inst];
							changed = True
						]
					];
					(*
					  If this is a StackAllocateInstruction and its only uses are in 
					  StoreInstruction then the StackAllocate and the Stores can all 
					  be deleted.
					*)
					If[changed === False && StackAllocateInstructionQ[inst],
						(* InsertDebugDeclarePass creates StackAllocate's which are meant to be read by the debugger; keep those. *)
						If[!TrueQ[inst["getProperty", "DebuggerReadable", False]],
							uses = inst["target"]["uses"];
							If[ Length[uses] === 0 ||  Length[uses] === 1 && StoreInstructionQ[First[uses]] || AllTrue[uses, StoreInstructionQ],
								Scan[Internal`StuffBag[toRemove, #]&, uses];
								Internal`StuffBag[toRemove, inst];
								changed = True]];
						];
				]
			|>,
			fm
		];
		#["unlink"]& /@ Internal`BagPart[toRemove, All];
		changed
	]

run[fm_, opts_] :=
	Module[{changed = True},
		While[changed,
			changed = iRun[fm, opts]
		];
		fm
	]


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"DeadCodeElimination",
	"Deletes instructions that are defined but not used."
];

DeadCodeEliminationPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
	    UsePass(*,
		LiveVariablesPass*)
	}
|>];

RegisterPass[DeadCodeEliminationPass]
]]

End[] 

EndPackage[]
