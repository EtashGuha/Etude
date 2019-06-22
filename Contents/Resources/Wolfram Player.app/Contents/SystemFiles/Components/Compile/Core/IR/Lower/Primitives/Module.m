BeginPackage["Compile`Core`IR`Lower`Primitives`Module`"]

Begin["`Private`"]

Needs["CompileAST`Create`Construct`"]
Needs["CompileAST`Class`Symbol`"]
Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["Compile`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileUtilities`Asserter`Assert`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)


createScopedVariable[state_, mexpr_, inSetQ_, writableQ_, opts_] :=
	Module[{builder, var},
		Which[
			MExprSymbolQ[mexpr],
				builder = state["builder"];
				var = state["createFreshVariable", mexpr];

				(* TODO Question: "variableDeclaration" -> True was previously only set if the
				   variable was a reference (so only set in non-SSA mode), but it seems like
				   it ought to be set now. What is the reason it was only set in non-SSA mode
				   before? *)
				(*inst["setProperty", "variableDeclaration" -> True];*)
				builder["symbolBuilder"]["add", mexpr["lexicalName"], var, <|"Writable" -> writableQ|>];
				var,
			state["isTypedMarkup", mexpr, opts],
				(*
				 Typed scoped symbols have to be initialized.
				 If types had a default value then we could consider using this here.
				*)
				If[ !inSetQ,
					ThrowException[LanguageException[{"Scoped variables which are typed must be initialized.", mexpr["toString"]}]]];
				var = createScopedVariable[state, mexpr["part", 1], inSetQ, writableQ, opts];
				var["setType", state["parseTypedMarkup", mexpr, opts]["type"]];
				var,
			mexpr["hasHead", Set],
				createScopedVariable[state, mexpr["part", 1], True, writableQ, opts],
			True,
				AssertThat["A Module variable must either be a symbol.",
					mexpr["toString"]]["fails"];
				False
		]
	]

createScopedVariable[args___] := ThrowException[ {"Unexpected arguments to createScopedVariable.", {args}}] 

createScopedVariables[state_, vars_, writableQ_, opts_] :=
	Map[createScopedVariable[state, #, False, writableQ, opts]&, vars["arguments"]];

assignScopedVariables[state_, lhss_, rhss_, opts_] :=
	MapThread[assignScopedVariable[state, #1, #2, opts]&, {lhss, rhss}]



uninitializedExpr := uninitializedExpr = CreateMExprSymbol[Compile`Uninitialized]
	
assignScopedVariable[state_, lhs_, rhs_, opts_] :=
	Module[{builder, inst},
		builder = state["builder"];
 		inst = builder["createCopyInstruction", lhs, rhs, lhs["mexpr"]];
 		builder["symbolBuilder"]["updateVariable", lhs["mexpr"]["lexicalName"], inst];
  		inst["setProperty", "variableWrite" -> True];
  		inst["target"]
  	]
	
processesScopedVariablesRHSs[state_, vars_, nosetQ_, opts_] :=
	Module[{rhss},
		rhss = Map[processesScopedVariablesRHS[state, #, nosetQ, opts]&, vars["arguments"]];
		rhss
	]


(*
  If Types had a default value we could consider using this 
  in place of Uninitialized.
*)		
processesScopedVariablesRHS[state_, mexpr_, False, opts_] := (
	If[mexpr["hasHead", Set],
		state["lower", mexpr["part", 2], opts],
		ThrowException[LanguageException[{"Variable in local variable specification requires a value.", mexpr["toString"]}]]
	]
)

processesScopedVariablesRHS[state_, mexpr_, True, opts_] := (
	If[mexpr["hasHead", Set],
		state["lower", mexpr["part", 2], opts],
		state["lower", uninitializedExpr, opts]
	]
)


lowerModule[state_, mexpr_, opts_] :=
	With[{args = mexpr["part", 1],
		  body = mexpr["part", 2]},
		(* The RHS of scoped variables need to be processed before
		 * we add the scoped variables to the environment. This makes
		 * code such as Module[{x = x}, x] pickup the right scope
		 *)
	    With[{rhss = processesScopedVariablesRHSs[state, args, True, opts],
	          lhss = createScopedVariables[state, args, True, opts]},
	        assignScopedVariables[state, lhss, rhss, opts]; 
		    state["lower", body, opts]
	    ]
	]

lowerBlock[state_, mexpr_, opts_] :=
	With[{args = mexpr["part", 1],
		  body = mexpr["part", 2]},
		(* The RHS of scoped variables need to be processed before
		 * we add the scoped variables to the environment. This makes
		 * code such as Module[{x = x}, x] pickup the right scope
		 *)
	    With[{rhss = processesScopedVariablesRHSs[state, args, True, opts],
	          lhss = createScopedVariables[state, args, True, opts]},
	        assignScopedVariables[state, lhss, rhss, opts]; 
		    state["lower", body, opts]
	    ]
	]

lowerWith[state_, mexpr_, opts_] :=
	With[{args = mexpr["part", 1],
		  body = mexpr["part", 2]},
		(* The RHS of scoped variables need to be processed before
		 * we add the scoped variables to the environment. This makes
		 * code such as Module[{x = x}, x] pickup the right scope
		 *)
	    With[{rhss = processesScopedVariablesRHSs[state, args, False, opts],
	          lhss = createScopedVariables[state, args, False, opts]},
	        assignScopedVariables[state, lhss, rhss, opts]; 
		    state["lower", body, opts]
	    ]
	]

(* 
 *
 * Not currently supported,  could be.
 *)
lowerLocalDecl[state_, mexpr_, opts_] := 
	ThrowException[ {"Native`DeclareVariable is not supported.", {mexpr}}]

RegisterCallback["RegisterPrimitive", Function[{st},
	RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Module], lowerModule];
	RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[With], lowerWith];
	RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Block], lowerBlock];
	RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Native`DeclareVariable], lowerLocalDecl]
]]

End[]

EndPackage[]
