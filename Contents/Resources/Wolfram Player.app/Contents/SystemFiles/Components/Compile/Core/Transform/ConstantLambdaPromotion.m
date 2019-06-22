BeginPackage["Compile`Core`Transform`ConstantLambdaPromotion`"]


ConstantLambdaPromotionPass

Begin["`Private`"] 

Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`Optimization`CopyPropagation`"]
Needs["Compile`Core`Optimization`ConstantPropagation`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Instruction`CopyInstruction`"]


visitLambda[data_, inst_] :=
	Module[{
		newInst,
		func = inst["source"]
	},
		If[!ConstantValueQ[func],
			Return[];
		];
		newInst = CreateCopyInstruction[inst["target"], func, inst["mexpr"]];
		newInst["cloneProperties", inst]; 
		newInst["moveBefore", inst];
		inst["unlink"]
	];
	
run[fm_, opts_] :=
	Module[{
		visitor
	},
		visitor = CreateInstructionVisitor[
			<|
				"visitLambdaInstruction" -> visitLambda
			|>,			
			fm,
			"IgnoreRequiredInstructions" -> True
		];
		fm
	];
	
RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"ConstantLambdaPromotion",
		"The pass removes lambda instructions if they are known to be constant."
];

ConstantLambdaPromotionPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
		CopyPropagationPass,
		ConstantPropagationPass
	}
|>];

RegisterPass[ConstantLambdaPromotionPass]
]]

End[] 

EndPackage[]
