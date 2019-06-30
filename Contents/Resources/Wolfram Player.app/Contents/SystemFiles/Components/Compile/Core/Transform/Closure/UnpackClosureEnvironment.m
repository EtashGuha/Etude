BeginPackage["Compile`Core`Transform`Closure`UnpackClosureEnvironment`"]

UnpackClosureEnvironmentPass

Begin["`Private`"] 

Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`Transform`Closure`Utilities`"]
Needs["CompileAST`Class`Literal`"]
Needs["TypeFramework`TypeObjects`TypeConstructor`"]
Needs["TypeFramework`TypeObjects`TypeApplication`"]
Needs["TypeFramework`TypeObjects`TypeArrow`"]
Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`IR`Instruction`CallInstruction`"]
Needs["Compile`Core`IR`Instruction`GetElementInstruction`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`IR`Instruction`LoadArgumentInstruction`"]
Needs["Compile`Core`IR`Instruction`StackAllocateInstruction`"]
Needs["Compile`Core`IR`Instruction`StoreInstruction`"]
Needs["CompileAST`Create`Construct`"]



getIndex[tyEnv_, ty_?TypeApplicationQ, name_] :=
	getIndex[tyEnv, ty["type"], name]
getIndex[tyEnv_, ty_?TypeConstructorQ, name_] :=
	With[{
		metadata = ty["getProperty", "metadata", <||>]
	},
		Lookup[Lookup[metadata, "Fields", <||>], name]
	]
	
getIndex[tyEnv_, ty_TypeSpecifier, name_] :=
	getIndex[tyEnv, tyEnv["resolve", ty], name]

getIndex[tyEnv_, ty_Type, name_] :=
	getIndex[tyEnv, tyEnv["resolve", ty], name]

visitCallInvoke[data_, inst_] :=
	Module[{
		fm = data["fm"],
		func = inst["function"],
		envVar,
		envTy = data["envTy"],
		newInst,
		capturedVarName,
		capturedVarIndex,
		tyEnv = data["typeEnvironment"]
	},
		If[!LoadClosureVariableCallQ[func],
			Return[]
		];
		data["changed"]["set", True];

		envVar = data["envVar"];
		capturedVarName = inst["target"]["getProperty", "aliasesVariable"]["name"];
		capturedVarIndex = getIndex[tyEnv, envTy, capturedVarName];

		If[MissingQ[capturedVarIndex],
			ThrowException["Could not get index of captured variable: " <> ToString[capturedVarName]];
		];

		newInst = CreateGetElementInstruction[
			inst["target"],
			envVar,
			{
				CreateConstantValue[capturedVarName]
			},
			GetClosureEnvironmentFunction[fm, capturedVarIndex, envTy],
			inst["mexpr"]
		];
		newInst["cloneProperties", inst];
		newInst["moveAfter", inst];
		inst["unlink"]
	];

visitLoadArgument[data_, inst_] :=
	With[{
		idx = inst["index"]
	},
		If[idx === None,
			Return[]
		];
		If[MExprLiteralQ[idx] && IntegerQ[idx["data"]],
			With[{
				nextIdx = idx["data"] + 1
			},
				inst["setIndex", CreateMExprLiteral[nextIdx]]
			];
			Return[]
		];
		If[ConstantValueQ[idx] && IntegerQ[idx["value"]],
			idx["setValue", idx["value"] + 1];
			Return[]
		];
		If[!IntegerQ[idx],
			Return[]
		];
		inst["setIndex", idx + 1]
	];

run[fm_, opts_] :=
	Module[{
		state,
		pm,
		tyEnv,
		envVarName,
		loadEnvInst,
		firstBB,
		envTy,
		envRefTy,
		envVarRefName,
		envVarRef,
		funTy
	},
		If[!CapturesVariablesQ[fm],
			Return[fm]
		];
		pm = fm["programModule"];
		tyEnv = pm["typeEnvironment"];
		envVarName = ClosureEnvironmentVariableName[fm] <> "$" <> ToString[fm["id"]];
		envVarRefName = envVarName <> "$$ref";
		envVarRef = CreateVariable[envVarRefName];
		(* Create the environment variable type *)
		envTy = ResolveEnvironmentVariableType[tyEnv, fm, CapturedVariables[fm]];
		envRefTy = tyEnv["resolve", TypeSpecifier["Handle"[envTy]]];
		envVarRef["setType", envRefTy];
		(* Visit instructions *)
		state = <|
			"envVar" -> envVarRef,
			"envTy" -> envTy,
			"fm" -> fm,
			"typeEnvironment" -> tyEnv,
			"changed" -> CreateReference[False]
		|>;
		CreateInstructionVisitor[
			state,
			<|
				"visitLoadArgumentInstruction" -> visitLoadArgument,
				"visitCallInstruction" -> visitCallInvoke,
				"visitInvokeInstruction" -> visitCallInvoke
			|>,			
			fm,
			"IgnoreRequiredInstructions" -> True
		];
		If[state["changed"]["get"] === False,
			Return[fm]
		];
		
		(* Add a load environment instruction at the beginning of the basic block *)
		firstBB = fm["firstBasicBlock"];
		loadEnvInst = CreateLoadArgumentInstruction[envVarRef, CreateMExprLiteral[1], None, None, True];
		loadEnvInst["moveBefore", firstBB["firstNonLabelInstruction"]];
		
		fm["setArguments", Prepend[fm["arguments"], envVarRef]];
		(*
		 Fix the function type.
		*)
		funTy = GetClosureFunctionType[tyEnv, fm, envRefTy];
		fm["setType", funTy];
		fm
	];


	
RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"UnpackClosureEnvironment",
	"The pass unpacks the closure environment variables into local variables."
];

UnpackClosureEnvironmentPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run
|>];

RegisterPass[UnpackClosureEnvironmentPass]
]]

End[]
	
EndPackage[]
