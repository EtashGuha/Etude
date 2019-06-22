BeginPackage["Compile`Core`Optimization`ConstantStackAllocation`"]
ConstantStackAllocationPass;

Begin["`Private`"] 

Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`PassManager`PassRunner`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Instruction`StackAllocateInstruction`"]
Needs["Compile`Core`IR`Instruction`LoadArgumentInstruction`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`Optimization`ConstantPropagation`"]



visitStack[state_, inst_] :=
	Module[{size = inst["size"]},
		If[ConstantValueQ[size] && inst["basicBlock"]["id"] =!= state["firstBB"]["id"],
			state["constantStacks"]["appendTo", inst];
		]
	]



run[fm_, opts_] :=
	Module[{state = <|"constantStacks" -> CreateReference[{}], 
					"firstBB" -> fm["firstBasicBlock"] |>, 
				list, lastInst},
		CreateInstructionVisitor[
			state,
			<| "visitStackAllocateInstruction" -> visitStack|>,
			fm ,
			"IgnoreRequiredInstructions" -> True];
		list = state["constantStacks"]["get"];
		If[ Length[list] > 0,
			lastInst = fm["firstBasicBlock"]["firstInstruction"];
			While[ StackAllocateInstructionQ[lastInst["next"]] || LoadArgumentInstructionQ[lastInst["next"]],
				lastInst = lastInst["next"]];
			Scan[
				(
					#["moveAfter", lastInst];
					lastInst = #;
				)&, list];
				];
		fm
	]


	
(**********************************************************)
(**********************************************************)
(**********************************************************)


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"ConstantStackAllocation",
	"Moves StackAllocate instructions which have a constant size to the top of the function."
];

ConstantStackAllocationPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
		ConstantPropagationPass
	}
|>];

RegisterPass[ConstantStackAllocationPass]
]]


End[] 

EndPackage[]
