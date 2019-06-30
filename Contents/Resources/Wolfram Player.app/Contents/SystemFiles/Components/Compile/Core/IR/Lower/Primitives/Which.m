BeginPackage["Compile`Core`IR`Lower`Primitives`Which`"]

Begin["`Private`"]

Needs["Compile`Core`IR`ConstantValue`"]
Needs["CompileAST`Create`Construct`"]
Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Reference`"]

(*

  Which[
     c1, e1,
     c2, e2,
     ...
     True, last
     ]
     
  Creates BasicBlocks

  [branch c1,  b1, cb2]
  
b1:
  [e1]
  [branch endWhich]

cb2:
  [branch c2,  b2, cb3]
  
b2:
  [e2]
  [branch endWhich]
...


true:
  [last]
  [branch endWhich]
  
endWhich

*)


isAtomValue[ mexpr_, val_] :=
	(mexpr["symbolQ"] && mexpr["data"] === HoldComplete[val] ||
	 mexpr["literalQ"] && mexpr["data"] === val)

(*
  Prepare branches for the Which.

*)
prepareBranches[ state_, mexpr_, opts_] :=
	Module[ {len, args, conds, exprs, lastCond},
		args = mexpr["arguments"];
		len = Length[args];
		If[ len < 2,
			ThrowException[LanguageException[{"Which is expected to have 2 or more arguments ", mexpr["toString"]}]]
		];
		If[ OddQ[len],
			ThrowException[LanguageException[{"Which called with `1` arguments.", len}]]];
		args = Partition[args, 2];
		args = Transpose[args];
		conds = First[args];
		exprs = Last[args];
		
		(*
		 Now look at the last condition.  If this is not True, we need to make everything Void
		 and add a final case that is True/Void
		*)
		lastCond = Last[conds];
		If[ !isAtomValue[lastCond, True],
			conds = Append[conds, CreateMExprSymbol[True]];
			exprs = Map[CreateMExpr[ CompoundExpression, {#, CreateMExprSymbol[Compile`Void]}]&,exprs];
			exprs = Append[exprs, CreateMExprSymbol[Compile`Void]]];
		(*
		    Maybe prune any conditions that are True/False, except for the last True
		*)
		(*
		  Return the prepared conditions
		*)
		Transpose[{conds, exprs}]
	]


createCode[state_, opts_, data_, {cond_, expr_}] :=
	Module[{condVar, exprVar, currBB, exprBB, nextBB, builder = state["builder"], fmBuilder, lastBB},
		fmBuilder = builder["currentFunctionModuleBuilder"];
		
		condVar = state["lower", cond, opts];
		currBB = builder["currentBasicBlock"];
		exprBB = builder["addBasicBlock", "codeWhich", expr];
		nextBB = builder["addBasicBlock", "testWhich", expr];
		builder["setCurrentBasicBlock", currBB];
		builder["createBranchInstruction", {exprBB, nextBB}, condVar, cond];
		currBB["addChild", exprBB];
		builder["sealBasicBlock", exprBB];
		currBB["addChild", nextBB];
		builder["sealBasicBlock", nextBB];
		builder["setCurrentBasicBlock", exprBB];
		exprVar = state["lower", expr, opts];
		exprBB = builder["currentBasicBlock"];
        If[ fmBuilder["returnMode"],
        	fmBuilder["setReturnMode", False];
        	lastBB = fmBuilder["lastBasicBlock"];	
        	builder["createBranchInstruction", lastBB, expr];
        	exprBB["addChild", lastBB];
        	,
        	data["nonReturn"]["set", True];
			builder["createBranchInstruction", data["endBB"], expr];
			exprBB["addChild", data["endBB"]];
			data["exprVars"]["appendTo", {exprBB, exprVar}]];
		builder["setCurrentBasicBlock", nextBB];
		data
	]


ClearAll[lower]
lower[state_, mexpr_, opts_] :=
     Module[{builder, code, fmBuilder, initBB, endBB, data, resVar, exprVars},
     	
		code = prepareBranches[ state, mexpr, opts];
		builder = state["builder"];
		fmBuilder = builder["currentFunctionModuleBuilder"];
		initBB = builder["currentBasicBlock"];
		resVar = state["createFreshVariable", mexpr];
		endBB = builder["addBasicBlock", "endWhich", mexpr];
		builder["setCurrentBasicBlock", initBB];		
		data = <|"exprVars" -> CreateReference[{}],"endBB" -> endBB, "nonReturn" -> CreateReference[False]|>;
		
		Scan[ createCode[state, opts, data, #1]&, code];		
		
		exprVars = data["exprVars"]["get"];
		If[ TrueQ[data["nonReturn"]["get"]],
			builder["createBranchInstruction", endBB];
			builder["currentBasicBlock"]["addChild", endBB];
			builder["setCurrentBasicBlock", endBB];			
			builder["createPhiInstruction", resVar, exprVars, mexpr];
			builder["sealBasicBlock", endBB];
			,
			If[ Length[exprVars] =!= 0 || endBB["parents"]["length"] =!= 0,
				ThrowException[LanguageException[{"Unexpected state compiling Which."}]]];
			fmBuilder["setReturnMode", True];
			resVar = Null];
		resVar
    ]

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Which], lower]
]]


End[]

EndPackage[]
