
BeginPackage["Compile`TypeSystem`FunctionData`KernelFunction`"]

MakeKernelFunction

AddKernelFunction


Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]
Needs["Compile`"]
Needs["CompileUtilities`Error`Exceptions`"]
Needs["CompileAST`Create`Construct`"]
Needs["CompileAST`Export`FromMExpr`"]

createUniqueSymbol[base_] :=
	createSymbol[Unique[base]]

createSymbol[sym_] :=
	CreateMExprSymbol[sym]

createMExprNormal[h_, args_] :=
	CreateMExprNormal[h, args]


MakeKernelFunction[ sym_, argTys_List -> resTy_] :=
	Module[{args = Table[createUniqueSymbol["arg"],{Length[argTys]}],
			eVars = Table[createUniqueSymbol["e"],{Length[argTys]}], 
			res = createUniqueSymbol["ef"],
			modList, bArgs, b1, b2, body, mod, fun},
		modList = Map[createMExprNormal[ Compile`Cast, {#, TypeSpecifier["Expression"]}]&, args];
		modList = MapThread[ createMExprNormal[Set, {##}]&, {eVars, modList}];
		modList = Append[modList, res];
		modList = createMExprNormal[List, modList];
		bArgs = Prepend[eVars, Native`ConstantExpression[sym]];
		b1 = createMExprNormal[ Native`CreateEvaluateExpression, bArgs];
		b1 = createMExprNormal[ Set, {res, b1}];
		b2 = createMExprNormal[ Compile`Cast, {res, TypeSpecifier[resTy]}];
		body = createMExprNormal[ CompoundExpression, {b1, b2}];
		mod = createMExprNormal[ Module, {modList, body}];
		args = createMExprNormal[List, args];
		fun = createMExprNormal[Function, {args, mod}];
		ReleaseHold[ FromMExpr[ fun]]
	]

AddKernelFunction[ env_, sym_, ty_] :=
	Module[ {fun = MakeKernelFunction[sym, ty]},
		If[ MatchQ[fun, _Function],
			env["declareFunction", sym, Typed[ty]@fun]]
	]



End[]

EndPackage[]
