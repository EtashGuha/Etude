BeginPackage["Compile`Core`Transform`ResolveTypes`"]


ResolveTypesPass

ResolveTypeInstruction

Begin["`Private`"] 

Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`ProgramModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Variable`"]
Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`IR`ProgramModule`"]
Needs["TypeFramework`"] (* for TypeObjectQ *)

ClearAll[needsResolve]
needsResolve[_?TypeObjectQ] := False
needsResolve[_Type] := True
needsResolve[_TypeSpecifier] := True
needsResolve[_?StringQ] := True
needsResolve[_?StringQ[___]] := True
needsResolve[___] := False

ClearAll[wrapType]
wrapType[t_TypeSpecifier] := t
wrapType[t_] := TypeSpecifier[t]
wrapType[Type[t_]] := TypeSpecifier[t]

resolve[tyEnv_, tyVarMap_, inst_, var_] :=
	Module[{
		ty = var["type"]
	},
		If[!needsResolve[ty],
			Return[]
		];
		ty = wrapType[ty];
		ty = tyEnv["resolveWithVariables", ty, tyVarMap];
		var["setType", ty] 
	];
	
visitInstruction[state_, inst_] :=
	With[{
		tyEnv = state["tyEnv"],
		tyVarMap = state["tyVarMap"]
	},
		If[inst["hasOperator"],
	    	resolve[tyEnv, tyVarMap, inst, inst["operator"]]
	    ];
		If[inst["definesVariableQ"],
	    	resolve[tyEnv, tyVarMap, inst, inst["definedVariable"]]
	    ];
		If[ inst["hasOperands"],
	    	resolve[tyEnv, tyVarMap, inst, #]& /@ Select[inst["operands"], ConstantValueQ[#] || VariableQ[#]&];
		]
	]

run[fm_?FunctionModuleQ, opts_] :=
	Module[{ visitor, state, tyEnv, pm = fm["programModule"], tyVarMap},
		tyEnv = pm["typeEnvironment"];
		tyVarMap = Lookup[opts, "tyVarMap", CreateReference[<||>]];
		state = <|
			"tyEnv" -> tyEnv,
			"tyVarMap" -> tyVarMap
		|>;
		If[fm["type"] =!= Undefined,
			fm["setType", tyEnv["resolveWithVariables", fm["type"], tyVarMap]]
		];
		visitor = CreateInstructionVisitor[
			state,
			<|
				"visitInstruction" -> visitInstruction
			|>,
			"IgnoreRequiredInstructions" -> True
		];
		visitor["traverse", fm];
	    fm
	];
	
run[pm_?ProgramModuleQ, opts_] :=
	With[{
		tyEnv = pm["typeEnvironment"],
		tyVarMap = CreateReference[<||>]
	},
		Do[
			tyDecl["resolveWithVariables", tyEnv, tyVarMap],
			{tyDecl, pm["typeDeclarations"]["get"]}
		];
		pm["scanFunctionModules",
			Function[{fm},
	        	run[ fm, Append[opts, "tyVarMap" -> tyVarMap]]
	        ]
	    ];
	    pm
	];


ResolveTypeInstruction[ pm_, inst_] :=
	Module[{ state, tyEnv, tyVarMap},
		tyEnv = pm["typeEnvironment"];
		tyVarMap = CreateReference[<||>];
		state = <|
			"tyEnv" -> tyEnv,
			"tyVarMap" -> tyVarMap
		|>;
		visitInstruction[state, inst];
	];


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"ResolveTypes",
	"The pass resolves all types within a program module --- transforming Type[<<string>>] to the type object." <>
	"It uses the type environment in the program module to perform the type resolution."
];

ResolveTypesPass = CreateProgramModulePass[<|
	"information" -> info,
	"runPass" -> run
|>];

RegisterPass[ResolveTypesPass]
]]

End[]
	
EndPackage[]
