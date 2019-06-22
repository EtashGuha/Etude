BeginPackage["Compile`AST`Macro`Builtin`List`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]
Needs["CompileAST`Create`Construct`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)

setupMacros[st_] :=
	Module[ {env = st["macroEnvironment"]},

		RegisterMacro[env, First,
			First[a_] -> Part[a, 1]
		];
		
		RegisterMacro[env, Last,
			Last[a_] -> Part[a, -1]
		];
		
		RegisterMacro[env, Rest,
			Rest[a_] -> Drop[a, 1]
		];
		
		RegisterMacro[env, Most,
			Most[a_] -> Drop[a, -1]
		];
		
		
		
		
		
(*		
  These all need improved Closure support. Coming soon.
		RegisterMacro[env, MemberQ,
			MemberQ[lst_, elem_] ->
				AnyTrue[lst, # === elem&]
		];
		RegisterMacro[env, FreeQ,
			FreeQ[lst_, elem_] ->
				AllTrue[lst, # =!= elem&]
		];
		RegisterMacro[env, Count,
			Count[lst_, elem_] ->
				Fold[ If[#2 === elem, #1+1,#1]&, 0, lst]
		];
		RegisterMacro[env, Cases,
			Cases[lst_, elem_] ->
				Select[list, # === elem&]
		];
		RegisterMacro[env, DeleteCases,
			DeleteCases[lst_, elem_] ->
				Select[list, # =!= elem&]
		];
*)	

		RegisterMacro[env, ConstantArray,
			ConstantArray[a_, n:Except[_List]] ->
				Module[ {elem = a},
					Compile`AssertType[n, "MachineInteger"];
					Table[elem, {n}]]
					,
			ConstantArray[f_, nList_List] ->
				Compile`Internal`MacroEvaluate[ makeConstantArray[f, nList]]
		];

		RegisterMacro[env, Array,
			Array[f_, n:Except[_List]] ->
				Module[ {},
					Compile`AssertType[n, "MachineInteger"];
					Table[f[i], {i,n}]],
			Array[f_, n:Except[_List], r:Except[_List]] ->
				Module[{},
					Compile`AssertType[n, "MachineInteger"];
					Compile`AssertType[r, "MachineInteger"];
					Table[f[i], {i,r,r+(n-1)}]
				],
			Array[f_, nList_List] ->
				Compile`Internal`MacroEvaluate[ makeTable[f, nList]],
			Array[f_, nList_List, rList_List] ->
				Compile`Internal`MacroEvaluate[ makeTable[f, nList, rList]]

		];
		
        RegisterMacro[env, Norm,
            Norm[lst_] ->
                Sqrt[Total[Map[#^2&, lst]]],
            Norm[lst_, p_] ->
                Power[Total[Map[#^p&, lst]], 1/p]
        ];

]

buildLiteral[ x_] :=
	CreateMExprLiteral[x]
	
buildSymbol[ x_] :=
	CreateMExprSymbol[x]
	
buildMExpr[ h_, args_] :=
	CreateMExpr[h, args]

checkTypes[ argsIn_] :=
	Module[{args},
		args = Map[ buildMExpr[ Compile`AssertType, {#, "MachineInteger"}]&, argsIn];
		buildMExpr[ CompoundExpression, args]
	]


makeConstantArray[ elem_, nList_] :=
	Module[{
			var = buildSymbol[Unique["x"]],
			iters = Map[ {#}&, nList["arguments"]], 
			fApp, table, checks, args, body
		},
		iters = Map[ buildMExpr[List, #]&, iters];
		fApp = Prepend[ iters, var];
		table = buildMExpr[Table, fApp];
		checks = checkTypes[nList["arguments"]];
		body = buildMExpr[CompoundExpression, Append[checks["arguments"], table]];
		args = buildMExpr[ Set, {var, elem}];
		args = buildMExpr[ List, {args}];
		buildMExpr[ Module, {args, body}]
	]


makeTable[ f_, nList_] :=
	Module[{
			iters = Map[ {buildSymbol[Unique["x"]], #}&, nList["arguments"]], 
			fApp, table, checks
		},
		fApp = buildMExpr[f,iters[[All, 1]]];
		iters = Map[ buildMExpr[List, #]&, iters];
		fApp = Prepend[ iters, fApp];
		table = buildMExpr[Table, fApp];
		checks = checkTypes[nList["arguments"]];
		buildMExpr[CompoundExpression, Append[checks["arguments"], table]]
	]


buildShift[r_, n_] :=
	buildMExpr[Plus, {r, n, buildLiteral[-1]}]

(*
  iters are {i,r,r+(n-1)}
*)
makeTable[ f_, nList_, rList_] :=
	Module[{
		nArgs = nList["arguments"], rArgs = rList["arguments"]
	},
		If[ Length[nArgs] =!= Length[rArgs],
			ThrowException[CompilerException["Array arguments do not have equal length.", 
						ReleaseHold[nList["toExpression"]], ReleaseHold[rList["toExpression"]]]]];
	Module[{
			iters = MapThread[ {buildSymbol[Unique["x"]], #1, buildShift[#1, #2]}&, {rArgs, nArgs}], 
			fApp, table, checks
		},
		fApp = buildMExpr[f,iters[[All, 1]]];
		iters = Map[ buildMExpr[List, #]&, iters];
		fApp = Prepend[ iters, fApp];
		table = buildMExpr[Table, fApp];
		checks = checkTypes[Join[nArgs, rArgs]];
		buildMExpr[CompoundExpression, Append[checks["arguments"], table]]
	]]


RegisterCallback["SetupMacros", setupMacros]


End[]
EndPackage[]
