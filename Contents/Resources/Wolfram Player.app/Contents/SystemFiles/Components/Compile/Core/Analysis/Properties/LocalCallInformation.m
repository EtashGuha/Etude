
BeginPackage["Compile`Core`Analysis`Properties`LocalCallInformation`"]

LocalCallInformationPass

Begin["`Private`"]

Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`IR`ConstantValue`"]


getLocalFunction[fm_, func_] :=
	Module[{pm, name, fms},
		pm = fm["programModule"];
		name = func["value"];
		fms = pm["functionModules"]["get"];
		SelectFirst[fms, #["name"] === name&]
	];


visitCallInvoke[data_, inst_] :=
	Module[{
		localFM,
		fm = data["fm"],
		func = inst["function"]
	},
		If[!ConstantValueQ[func],
			Return[];
		];
		localFM = getLocalFunction[fm, func];
		If[MissingQ[localFM],
			Return[]
		];
		func["setProperty", "localFunctionModule" -> localFM];
		inst["setProperty", "localFunctionModuleCall" -> localFM];
	];
	
run[fm_, opts_] :=
	Module[{
		state, visitor
	},
		state = <| "fm" -> fm |>;
		visitor = CreateInstructionVisitor[
			state, 
			<|
				"visitCallInstruction" -> visitCallInvoke,
				"visitInvokeInstruction" -> visitCallInvoke
			|>,			
			fm,
			"IgnoreRequiredInstructions" -> True
		];
		fm
	];
	

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"LocalCallInformation",
		"If a call is to a function that is within the program module, then this pass adds properties " <>
		"to the call linking it to the function module. This analysis is used by other passes such as " <>
		"the pack closure environment"
];

LocalCallInformationPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"passClass" -> "Analysis"
|>];

RegisterPass[LocalCallInformationPass]
]]


End[]

EndPackage[]
