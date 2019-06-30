BeginPackage["Compile`Core`Transform`Closure`PackClosureEnvironment`"]

PackClosureEnvironmentPass

Begin["`Private`"] 

Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`Transform`Closure`Utilities`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`Transform`ConstantLambdaPromotion`"]
Needs["Compile`Core`Transform`InertFunctionPromotion`"]
Needs["Compile`Core`Analysis`Properties`LocalCallInformation`"]
Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`Transform`Closure`DeclareClosureEnvironmentType`"]
Needs["Compile`Core`IR`Instruction`LoadInstruction`"]
Needs["Compile`Core`IR`Instruction`SetElementInstruction`"]
Needs["Compile`Core`IR`Instruction`StackAllocateInstruction`"]
Needs["Compile`Core`IR`Instruction`CallInstruction`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`Transform`ResolveFunctionCall`"]
Needs["Compile`TypeSystem`Inference`InferencePass`"]
Needs["Compile`Core`Transform`ResolveTypes`"]
Needs["TypeFramework`TypeObjects`TypeArrow`"]

localCallQ[inst_] :=
	inst["hasProperty", "localFunctionModuleCall"]


getClosureBinding[ closureBinding_, var_] :=
	Module[ {newVar = closureBinding["lookup", var["id"]]},
		If[ !VariableQ[newVar],
			ThrowException[CompilerException[{"Cannot find binding for variable in closure binding environment", var}]]];
		newVar
	]

preparePackEnvironment[data_, inst_, calledFM_] :=
	Module[{
		fm, pm, tyEnv, 
		envTy, stackAllocInst,
		capturedVars, setElementInst, prevInst,
		envVarRefName, envRefVar, ii = 1,
		closureBinding = Null
	},
		fm = data["fm"];
		pm = fm["programModule"];
		tyEnv = pm["typeEnvironment"];
		capturedVars = CapturedVariables[calledFM];
		closureBinding = fm["getProperty", "closureVariablesBinding", Null];
		If[ closureBinding === Null,
			ThrowException[CompilerException[{"closureVariablesBinding property is not set", fm}]]];
		
		(* Create the environment variable type *)
		envTy = ResolveEnvironmentVariableType[tyEnv, calledFM, capturedVars];
		(*
		 * Create the target variable that will hold the environment
		 * We add the id of the instruction to make sure that we are not
		 * introducing duplicate variables
		*)
		envVarRefName = ClosureEnvironmentVariableName[calledFM] <> "$" <> ToString[inst["id"]] <> "$ref";
		envRefVar = CreateVariable[envVarRefName];
		envRefVar["setType", TypeSpecifier["Handle"[envTy]]];
		(* Store the environment on the stack *)
		stackAllocInst = CreateStackAllocateInstruction[envRefVar, CreateTypedConstant[tyEnv, 1, "MachineInteger"], None, inst["mexpr"]];
		stackAllocInst["moveBefore", inst];
		prevInst = stackAllocInst;
		(* Insert the captured variables in each slot in the environment data structure *)
		Map[
			Function[{var},
				Module[{remapVar},
					remapVar = getClosureBinding[ closureBinding, var];
					setElementInst = CreateSetElementInstruction[
						envRefVar,
						{
							CreateConstantValue[var["name"]]
						},
						remapVar,
						SetClosureEnvironmentFunction[calledFM, ii, envTy],
						inst["mexpr"]
					];
					setElementInst["moveAfter", prevInst];
					ii++;
					prevInst = setElementInst
			]]
			,
			capturedVars
		];
		envRefVar
	];



getCapturesVariableFunction[data_, arg_?ConstantValueQ] :=
	Module[{pm, val = arg["value"], fmArg},
		pm = data["fm"]["programModule"];
		fmArg = pm["getFunctionModule", val];
		If[FunctionModuleQ[fmArg] && CapturesVariablesQ[fmArg],
			fmArg, Null]
	]

getCapturesVariableFunction[data_, arg_] :=
	Null



(*
  Support for passing closure consuming functions as an argument is somewhat limited.
  At present it is only provided for the first argument going to an externally linked
  function.  This is useful for interfacing with external code.
*)
getClosureArgument[data_, inst_] :=
	Module[ {args, closureFM},
		args = inst["arguments"];
		If[ Length[args] === 0,
			Return[ Null]];
		closureFM = getCapturesVariableFunction[ data, First[args]];
		Scan[
			If[	getCapturesVariableFunction[data,#] =!= Null,
				ThrowException[{"Closure functions are only supported for the first argument.", #}]]&, Rest[args]];
		closureFM
	]


(*
  func has a closure argument as an argument.  But this is only supported for
  external functions which have a ClosureForward property set.  Return the 
  name of the closure forward or Null.
*)
getClosureForward[ data_, func_?ConstantValueQ] :=
	Module[{name = func["value"], pm, decl},
		pm = data["fm"]["programModule"];
		decl = pm["externalDeclarations"]["lookupFunction", name];
		If[!AssociationQ[decl],
			Return[Null]];
		Lookup[decl, "ClosureForward", Null]
	]

getClosureForward[ data_, func_] :=
	Null


(*
  closureFM is an argument to the call instruction inst.
  if the call cannot take a closure then exit.
  Otherwise setup the pack environment,  cast to VoidHandle and 
  add as an argument.
*)
processClosureArgument[data_, inst_, closureFM_] :=
	Module[{func = inst["function"], extClosure, envVar,castInst,funCast, pm,args},
		extClosure = getClosureForward[data,func];
		If[ extClosure === Null,
			ThrowException[{"Cannot pass a closure function as an argument", closureFM["name"]}]];
		(*
		  Now prepare the ClosureEnvironment with preparePackEnvironment. Cast the 
		  env and the function being called to VoidHandle,  needed for the external function.
		*)
		envVar = preparePackEnvironment[data, inst, closureFM];
		pm = data["fm"]["programModule"];
		args = inst["arguments"];
		funCast = CreateBitCastTo[pm, First[args], "VoidHandle"];
		funCast["moveBefore", inst];
		castInst = CreateBitCastTo[pm, envVar, "VoidHandle"];
		castInst["moveBefore", inst];
		args = Join[{castInst["target"], funCast["target"]}, Drop[args,1]];
		inst["setArguments", args];
		(*
		  Add the extClosure to the externalDeclarations.
		*)
		pm["externalDeclarations"]["lookupUpdateFunction", pm["typeEnvironment"], extClosure];
		func = CreateConstantValue[extClosure];
		inst["setFunction", func]; 
	]



(*
  calledFM is a closure function passed as an argument at argIndex of inst.
  create a ClosureTuple that contains the closure environment and the function.
  Then modify the inst to take the ClosureTuple.
*)
insertClosureTuple[data_, inst_, calledFM_, closureFM_, argIndex_] :=
	Module[{envVar, fun, newInst, typeRes, funTy, funArg},
		envVar = preparePackEnvironment[data, inst, closureFM];
		funTy = GetClosureFunctionType[data["typeEnvironment"], closureFM, envVar["type"]];
		funArg = inst["getArgument", argIndex];
		funArg["setType", funTy];
		fun = CreateConstantValue[ Native`CreateTuple];
		newInst = CreateCallInstruction["closure", fun, {envVar, funArg}];
		newInst["moveBefore", inst];
		ResolveTypeInstruction[ data["programModule"], newInst];
		typeRes = InferTypeInstruction[ data["programModule"], newInst];
		If[ !typeRes, 
				ThrowException[{"Cannot infer type for closure tuple."}]];
		ResolveFunctionCallInstructionWithInferencing[ data["programModule"], newInst];
		inst["setArgument", argIndex, newInst["target"]];
		(*
		  Fix the calledFM function type
		*)
		Module[{args},
			funTy = calledFM["type"];
			If[ !TypeArrowQ[funTy],
				ThrowException[{"Unexpected function type", calledFM, funTy}]];
			args = funTy["arguments"];
			args = ReplacePart[args, argIndex -> newInst["target"]["type"]];
			funTy = data["typeEnvironment"]["resolve", TypeSpecifier[ args -> funTy["result"]]];
			calledFM["setType", funTy];
		]
		
	]



(*
    fm for inst (a call inst) provides closure variables.
    Look at function,  if this consumes closure varaibles treat specially.
    Then look at arguments if any of these consume closure variables treat specially.
    Otherwise do nothing.
*)
visitCallInvoke[data_, inst_] :=
	Module[{
		func, calledFM, envVar, closureFM, args, prop
	},
		func = inst["function"];
		Which[
			localCallQ[inst],
				calledFM = inst["getProperty", "localFunctionModuleCall"];
				args = inst["arguments"];
				Do[
					closureFM = getCapturesVariableFunction[ data, Part[args, i]];
					If[FunctionModuleQ[closureFM],
						insertClosureTuple[data, inst, calledFM, closureFM, i];
						prop = calledFM["getProperty", "closureArguments", {}];
						prop = Append[prop, i];
						calledFM["setProperty", "closureArguments" -> prop]], 
					{i, Length[args]}];
				If[CapturesVariablesQ[calledFM],
					envVar = preparePackEnvironment[data, inst, calledFM];
					(* The call now calls the closure function with the constructed environment variable *) 
					inst["setArguments", Prepend[inst["arguments"], envVar]];
					(*
					 If the calledFM also has closure arguments these are now shifted on, so fix.
					*)
					If[ calledFM["hasProperty", "closureArguments"],
						calledFM["setProperty", "closureArguments" -> 1 + calledFM["getProperty", "closureArguments"]]]
					];

			,
			closureFM = getClosureArgument[data, inst];
			FunctionModuleQ[closureFM],
				processClosureArgument[data, inst, closureFM]
			,
			True,
				Null];
	];

	
run[fm_, opts_] :=
	Module[{
		state, visitor, pm
	},
		If[Length[fm["getProperty", "closureVariablesProvided", {}]] === 0,
		   Return[]
		];
		pm = fm["programModule"];
		state = <| "fm" -> fm , "programModule" -> pm, 
					"typeEnvironment" -> pm["typeEnvironment"]|>;
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
	"PackClosureEnvironment",
	"The pass packs the closure environment variables into the environment data structure."
];

PackClosureEnvironmentPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
		ConstantLambdaPromotionPass,
		InertFunctionPromotionPass,
		LocalCallInformationPass,
		DeclareClosureEnvironmentTypePass
	}
|>];

RegisterPass[PackClosureEnvironmentPass]
]]


End[]
	
EndPackage[]
