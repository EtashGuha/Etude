BeginPackage["Compile`Core`IR`Lower`Primitives`Set`"]

Begin["`Private`"]

Needs["CompileAST`Class`Symbol`"]
Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`"]
Needs["CompileUtilities`Callback`"]


(*
 allow calls such as a = Native`MutabilityClone[a],  which are only inserted by 
 part assignment
*)

cloneCallQ[ expr_] :=
	Module[{},
		expr["normalQ"] &&
		expr["length"] === 1 && 
		expr["head"]["symbolQ"] &&
		expr["head"]["fullName"] === "Native`MutabilityClone"
	]

isDeclareVariable[state_, mexpr_, opts_] :=
    With[{hd = mexpr["head"]},
        hd["symbolQ"] && 
        hd["fullName"] === "Native`DeclareVariable" && 
        (mexpr["length"] === 1 || mexpr["length"] === 2)
    ]
isDecoratedLHS[state_, mexpr_, opts_] :=
	state["isTypedMarkup", mexpr, opts] || isDeclareVariable[state, mexpr, opts]


stripTyped[mexpr_] :=
	If[mexpr["normalQ"] && mexpr["hasHead", Typed],
	   mexpr["part", 1],
	   mexpr
	]
	
getSymbol[state_, mexpr_, opts_] :=
	Which[
		isDeclareVariable[state, mexpr, opts],
			stripTyped[mexpr["part", 1]],
		state["isTypedMarkup", mexpr, opts],
			stripTyped[mexpr],
		True,
			mexpr
	]

(*
  Note that Set[ Part[ sym, args], val] is handled by the Macro system.
*)
lower[state_, mexpr_, opts_] :=
	Module[{builder, lhsExpr, rhsExpr, lhsSymbol, rhs, trgt, inst, ef},
		If[mexpr["length"] =!= 2,
			ThrowException[LanguageException[{Set, "argcount", mexpr["toString"]}]]
		];
		builder = state["builder"];
        lhsExpr = mexpr["part", 1];
        rhsExpr = mexpr["part", 2];
        lhsSymbol = getSymbol[state, lhsExpr, opts];
		Which[
			MExprSymbolQ[lhsSymbol],
				If[isDeclareVariable[state, lhsExpr, opts],
		        	ThrowException[ {"Native`DeclareVariable is not supported.", {mexpr}}]
				];
				If[ !builder["symbolBuilder"]["isWritable", lhsSymbol["lexicalName"]] && !cloneCallQ[rhsExpr],
					ThrowException[LanguageException[{Set, "nowrite", mexpr["toString"]}]]
				];
				Which[
		        	state["isTypedMarkup", lhsExpr, opts],
						ThrowException[LanguageException[{"A typed variable can only be given when a variable is declared.", mexpr["toString"]}]],
		        	True,
						trgt = state["createFreshVariable"]
		        ];
				builder["symbolBuilder"]["pushAssignAlias", lhsSymbol["lexicalName"], trgt];
				rhs = state["lower", rhsExpr, opts];
				builder["symbolBuilder"]["popAssignAlias", lhsSymbol["lexicalName"]];
				inst = builder["createCopyInstruction", trgt, rhs, mexpr];
				inst["setProperty", "variableWrite" -> True];
				trgt["setProperty", "variableValue" -> lhsSymbol["lexicalName"]];
				builder["symbolBuilder"]["updateVariable", lhsSymbol["lexicalName"], inst];
				ef = rhs,
			True,
				ThrowException[LanguageException[{Set, "lhs", mexpr["toString"]}]]
		];
		ef
	]

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Set], lower]
]]

End[]

EndPackage[]
