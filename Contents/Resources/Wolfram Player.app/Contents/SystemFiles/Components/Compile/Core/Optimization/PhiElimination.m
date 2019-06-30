
BeginPackage["Compile`Core`Optimization`PhiElimination`"]

(* 

Try to eliminate PhiInstructions.  This is similar to the code in SSA Builder 
tryRemoveTrivialPhi.

At present it just removes Phis that look like

v1 = phi( v0, v2)

...

v2 = v1

since the source of the copy is equal to the target of the phi we can eliminate this branch.
The code just replaces this with 

v1 = v0

More functionality would be to use some of the techniques in tryRemoveTrivialPhi.



In addition similar functionality is found in

   https://github.com/golang/go/blob/master/src/cmd/compile/internal/ssa/phielim.go    
*)

PhiEliminationPass;

Begin["`Private`"] 

Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`PassManager`PassRunner`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`Optimization`EvaluateExpression`"]
Needs["Compile`Core`Optimization`CopyPropagation`"]
Needs["Compile`Core`Optimization`CoerceCast`"]
Needs["Compile`Core`IR`Instruction`BranchInstruction`"]
Needs["Compile`Core`IR`Instruction`CopyInstruction`"]
Needs["Compile`Core`IR`Variable`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Reference`"]


processDef[state_, var_, srcVar_, inst_?CopyInstructionQ] :=
	If[ !inst["source"]["sameQ", var],
		state["toKeep"]["appendTo",srcVar]
	]

processDef[state_, var_, srcVar_, inst_] :=
	state["toKeep"]["appendTo",srcVar];

processSource[state_, var_, src_] :=
	Module[{},
		If[VariableQ[src],
			processDef[state, var, src, src["def"]]
			,
			state["toKeep"]["appendTo",src]];
	]

processPhi[state_, inst_] :=
	Module[{trgt = inst["target"], srcVar, newInst},
		state["toKeep"]["set",{}];
		Scan[processSource[state, trgt, #]&, inst["getSourceVariables"]];
		If[ state["toKeep"]["length"] =!= 1,
			Return[]];
		srcVar = state["toKeep"]["getPart", 1];
		newInst = CreateCopyInstruction[trgt, srcVar, inst["mexpr"]];
		newInst["cloneProperties", inst]; 
		newInst["moveBefore", inst];
		inst["unlink"];
		trgt["setDef", newInst];
		state["changed"]["set", True];
	]

createState[] :=
	<|"toKeep" -> CreateReference[{}], "changed" -> CreateReference[False]|>

run[fm_, opts_] :=
	Module[{state, changed = True},
		While[changed === True,
			state = createState[];
			CreateInstructionVisitor[
				state,
				<|
					"visitPhiInstruction" -> processPhi
				|>,
				fm,
				"IgnoreRequiredInstructions" -> True
			];
			changed = state["changed"]["get"];
			];
		fm
	]

	
(**********************************************************)
(**********************************************************)
(**********************************************************)


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"PhiElimination",
	"Eliminates Phi Instructions that have a loop back to original value."
];

PhiEliminationPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
	}
|>];

RegisterPass[PhiEliminationPass]
]]


End[] 

EndPackage[]
