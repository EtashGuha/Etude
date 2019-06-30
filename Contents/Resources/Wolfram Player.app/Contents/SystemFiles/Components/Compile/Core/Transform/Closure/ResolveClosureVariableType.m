BeginPackage["Compile`Core`Transform`Closure`ResolveClosureVariableType`"]

ResolveClosureVariableTypePass

Begin["`Private`"] 

Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`Transform`Closure`Utilities`"]
Needs["TypeFramework`TypeObjects`TypeVariable`"]


run[fm_, opts_] :=
	Module[{ visitor, tyEnv, pm = fm["programModule"]},
		tyEnv = pm["typeEnvironment"];
		visitor =  CreateInstructionVisitor[
			<|
				"visitCallInstruction" -> Function[{st, inst},
					Module[{
						closureVar, closureVarType
					},
						If[inst["getProperty", "isClosureLoad", False] === False,
							Return[]
					    ];
						If[!LoadClosureVariableCallQ[inst["function"]],
							Return[]
						];
						closureVar = inst["getProperty", "closureVariable"];
						If[MissingQ[closureVar],
							Return[]
						];
						closureVarType = closureVar["type"];
						If[ closureVarType === Undefined,
							closureVarType = CreateTypeVariable[closureVar["name"]];
							closureVar["setType", closureVarType]];
						inst["target"]["setType", closureVarType];
						inst["function"]["setType", tyEnv["resolve", TypeSpecifier[{} -> closureVarType]]]
					]
				]
			|>,
			"IgnoreRequiredInstructions" -> True
		];
		visitor["traverse", fm];
	    fm
	]


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"ResolveClosureVariableType",
	"The pass resolves all types within a program module --- transforming TypeSpecifier[<<string>>] to the type object." <>
	"It uses the type environment in the program module to perform the type resolution."
];

ResolveClosureVariableTypePass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run
|>];

RegisterPass[ResolveClosureVariableTypePass]
]]

End[]
	
EndPackage[]
