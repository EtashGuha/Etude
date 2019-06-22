BeginPackage["Compile`Core`Transform`InertFunctionPromotion`"]


InertFunctionPromotionPass

Begin["`Private`"] 

Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`Optimization`CopyPropagation`"]
Needs["Compile`Core`Optimization`ConstantPropagation`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Instruction`CallInstruction`"]


knownLocalFunctionQ[fm_, func_] :=
	Module[{pm, name, fms},
		pm = fm["programModule"];
		name = func["value"];
		fms = pm["functionModules"]["get"];
		TrueQ[AnyTrue[fms, #["name"] === name&]]
	];
	
visitInert[data_, inst_] :=
	Module[{
		newInst,
		fm = data["fm"],
		func = inst["head"]
	},
		If[!ConstantValueQ[func],
			Return[];
		];
		If[!knownLocalFunctionQ[fm, func],
			Return[]
		];
		
		newInst = CreateCallInstruction[inst["target"], func, inst["arguments"], inst["mexpr"]];
		newInst["cloneProperties", inst]; 
		newInst["moveBefore", inst];
		inst["unlink"]
	];
	
run[fm_, opts_] :=
	Module[{
		state, visitor
	},
		state = <| "fm" -> fm |>;
		visitor = CreateInstructionVisitor[
			state, 
			<|
				"visitInertInstruction" -> visitInert
			|>,			
			fm,
			"IgnoreRequiredInstructions" -> True
		];
		fm
	];
	
RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"InertFunctionPromotion",
		"The pass promotes inert functions into calls if the head is a known function."
];

InertFunctionPromotionPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
		CopyPropagationPass,
		ConstantPropagationPass
	}
|>];

RegisterPass[InertFunctionPromotionPass]
]]

End[] 

EndPackage[]
