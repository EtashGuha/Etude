BeginPackage["Compile`Core`Transform`Closure`PassClosureArguments`"]

PassClosureArgumentsPass

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
Needs["Compile`Core`IR`Instruction`InvokeInstruction`"]
Needs["Compile`Core`IR`Instruction`GetElementInstruction`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`IR`Instruction`LoadArgumentInstruction`"]
Needs["Compile`Core`IR`Instruction`StackAllocateInstruction`"]
Needs["Compile`Core`IR`Instruction`StoreInstruction`"]
Needs["CompileAST`Create`Construct`"]



(*
  inst is a CallInstruction where the function is actually a ClosureTuple.
  Call[ closure, {arg1, arg2, ...}]
  
  We should change the code to 
     env = closure[1];
     fun = closure[2];
     
     Call[ fun, {env, arg1, arg2, ...}]
*)

fixCall[state_, var_, inst_] :=
	Module[{newInst, envVar, funVar},
		newInst = CreateGetElementInstruction[
			"env", var, { CreateConstantValue[0]},CreateConstantValue[Native`ClosureEnvironment],inst["mexpr"]];
		newInst["moveBefore", inst];
		envVar = newInst["target"];
		newInst = CreateGetElementInstruction[
			"env", var, { CreateConstantValue[1]},CreateConstantValue[Native`ClosureFunction],inst["mexpr"]];
		newInst["moveBefore", inst];
		funVar = newInst["target"];
		
		inst["setArguments", Prepend[inst["arguments"], envVar]];
		inst["setFunction", funVar];
	]

processInstruction[state_, var_, inst_?CallInstructionQ] :=
	Module[{fun = inst["function"]},
		If[ VariableQ[fun] && fun["id"] === var["id"],
			fixCall[state, var, inst]];
	]

processInstruction[state_, var_, inst_?InvokeInstructionQ] :=
	Module[{fun = inst["function"]},
		If[ VariableQ[fun] && fun["id"] === var["id"],
			fixCall[state, var, inst]];
	]



processInstruction[state_, var_, l_List] :=
	Scan[processInstruction[state, var, #]&, l]

processInstruction[state_, var_, inst_] :=
	ThrowException[{"PassClosureArguments cannot follow chain."}]

processClosureArgument[state_, index_] :=
	Module[{loadInst, var, uses, argTy},
		loadInst = state["loadMap"]["lookup", index, Null];
		If[ loadInst === Null,
			ThrowException[{"PassClosureArguments cannot find LoadArgumentInstruction."}]];
		argTy = Part[state["functionModule"]["type"]["arguments"], index];
		var = loadInst["target"];
		var["setType", argTy];
		uses = var["uses"];
		processInstruction[state, var, uses];
	]


visitLoadArgument[state_, inst_] :=
	Module[{index},
		index = inst["index"]["data"];
		state["loadMap"]["associateTo", index -> inst];
	]



run[fm_, opts_] :=
	Module[{
		pm, args, state
	},
		args = fm["getProperty", "closureArguments", {}];
		If[Length[args] === 0,
			Return[fm]
		];
		pm = fm["programModule"];
		state = <|"functionModule" -> fm, "programModule" -> pm, "loadMap" -> CreateReference[<||>]|>;
		CreateInstructionVisitor[
			state,
			<|
				"visitLoadArgumentInstruction" -> visitLoadArgument
			|>,			
			fm,
			"IgnoreRequiredInstructions" -> True
		];
		Scan[ processClosureArgument[state, #]&, args];
		fm
	];


	
RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"PassClosureArguments",
	"The pass works with function that are called with closure arguments to 
	unpack the closure. "
];

PassClosureArgumentsPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run
|>];

RegisterPass[PassClosureArgumentsPass]
]]

End[]
	
EndPackage[]
