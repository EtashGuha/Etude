BeginPackage["Compile`AST`Macro`Builtin`Typed`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileAST`Create`Construct`"]
Needs["CompileAST`Export`FromMExpr`"]


setupMacros[st_] :=
	Module[ {env = st["macroEnvironment"]},
		RegisterMacro[env, Typed,
			Typed[Typed[a_, t_], t_] -> Typed[a, t]
		];
		RegisterMacro[env, Compile`AssertType,
			Compile`AssertType[a_, t_] -> Native`AssertType[a, Typed[0, t]]
		];
		RegisterMacro[env, Compile`AssertTypeApplication,
			Compile`AssertTypeApplication[a_, t_] -> Native`AssertTypeApplication[a, Typed[0, t]]
		];
		RegisterMacro[env, Compile`Utilities`TypedKernelFunction,
				Compile`Utilities`TypedKernelFunction[body_,ty_] -> 
					Compile`Internal`MacroEvaluate[expandKernelFunction[body, ty]]
		];
		RegisterMacro[env, Compile`Utilities`TypedKernelFunctionCall,
			Compile`Utilities`TypedKernelFunctionCall[body_,ty_][ args___] -> 
				Compile`Internal`MacroEvaluateHeld[expandKernelFunctionCall[body, ty, {args}]]
		]
	];


makeUnique[] :=
	Unique["CompileInternal`var"]

getFunArgs[ args_List -> res_] :=
	args

getFunArgs[ ty_] :=
	ThrowException[ {"The type specification for KernelFunction is not a function specification.", ty}]

getFunRes[  args_List -> res_] :=
	res

getFunRes[ ty_] :=
	ThrowException[{"The type specification for KernelFunction is not a function specification.", ty}]

isFunctionType[ args_List -> res_] :=
	True

isFunctionType[ _] :=
	False

createMExpr[ h_, arg_] :=
	createMExpr[ h, {arg}]

createMExpr[ h_, args_List] :=
	With[ {args1 = DeleteCases[Flatten[args],{}]},
		CreateMExprNormal[ h, args1]
	]

(*
  ty is a function type,  we need to make a function that takes the 
  specified types,  converts them into Expressions,  calls the function
  and then converts back into the output type.
*)
createResult[fun_, funTy_?isFunctionType] :=
	Module[ {argTys, resTy, nArgsList, nArgs, nBody, nFun},
		argTys = getFunArgs[ funTy];
		resTy = getFunRes[ funTy];
		nArgsList = Map[ Typed[makeUnique[], #]&, argTys];
		nArgs = Map[ First, nArgsList];
		nBody = createBody[fun, nArgs, resTy];
		nArgsList = createMExpr[List, nArgsList];
		nFun = createMExpr[ Function, {nArgsList, nBody}];
		nFun
	]

createResult[ var_, ty_] :=
	createMExpr[Compile`Cast, {var, TypeSpecifier[ty]}]


(*
 Assign the local variable to the global symbol via the evaluator.
 This sets up the closure data.
*)
assignVar[ var_] :=
	Module[{varGlobal, setArg},
		varGlobal = createMExpr[ Native`ConstantExpression, var];
		setArg = createMExpr[ Native`ConstantExpression, Set];
		createMExpr[Native`CreateEvaluateExpression, {setArg, varGlobal, var}]
	]

createBody[ fun_, args_, resTy_] :=
	Module[{exprFun, exprArgs, evalBody, var},
		If[ fun["symbolQ"],
			exprFun = createMExpr[ Native`LookupSymbol, fun["fullName"]]
			,
			exprFun = createMExpr[ Compile`ConstantValue, {}];
			exprFun["setProperty", "constantValueArgument" -> fun];
			exprFun = createMExpr[ Typed, {exprFun, TypeSpecifier["Expression"]}];
			exprFun];
		exprArgs = Map[ createMExpr[Compile`Cast, {#, TypeSpecifier["Expression"]}]&, args];
		evalBody = createMExpr[Native`CreateEvaluateExpression, Prepend[exprArgs, exprFun]];
		var = CreateMExpr @@ {makeUnique[]};
		createMExpr[
			Module,
			{
			createMExpr[List, var]
			,
			createMExpr[CompoundExpression,
				{
				createMExpr[ Set, {var, evalBody}]
				,
				If[isFunctionType[resTy],
					assignVar[var],
					{}]
				,
				createResult[var, resTy]
				}]
			}]
	]

(*
  Expand KernelFunction with inline code.
*)
expandKernelFunctionCall[ fun_, ty_, argsIn_] :=
	Module[ {funTy, argTys, resTy, args},
		funTy = ReleaseHold[ FromMExpr[ty]];
		argTys = getFunArgs[ funTy];
		resTy = getFunRes[ funTy];
		args = argsIn["arguments"];
		If[ Length[argTys] =!= Length[args],
			ThrowException[{"The number of type arguments does not match the number of called arguments.", ty, args}]];
		createBody[fun, args, resTy]	
	]

(*
  Expand KernelFunction into a separate function.
*)
expandKernelFunction[ fun_, ty_] :=
	Module[ {funTy},
		funTy = ReleaseHold[ FromMExpr[ty]];
		If[ isFunctionType[ getFunRes[ funTy]],
			ThrowException[{"A KernelFunction function that returns a function is not currently supported.", ReleaseHold[FromMExpr[fun]], funTy}]];
		createResult[fun, funTy]
	]


RegisterCallback["SetupMacros", setupMacros]

End[]
EndPackage[]
