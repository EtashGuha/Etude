BeginPackage["Compile`AST`Transform`Beta`"]

MExprBetaReduce; 

Begin["`Private`"] 

Needs["Compile`AST`Transform`Alpha`"]
Needs["CompileAST`Create`Construct`"]
Needs["CompileAST`Class`Normal`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)


ClearAll[MExprBetaReduce]
MExprBetaReduce[args___] :=
	reduce[args]

ClearAll[reduce]
reduce[fun_?MExprNormalQ, arg_?MExprNormalQ] :=
	reduce[fun, arg["arguments"]]
	
(*
 Called from the Macro system for Function when we have Function[ vars, body][args]
 Because of the ElaborateFunctionSlots pass,  this will never see slot type functions.
 This is going to fail for slot functions with unused arguments,  eg 
   #1+1&[1,2]
*)
	
reduce[fun_?MExprNormalQ, args0_?ListQ] :=
	Module[{boundVars, body, vars, len, args = args0},

		If[ !fun["hasHead", Function] || fun["length"] =!= 2,
			ThrowException[{"Input should be a function with parameters and a body", fun}]];
		vars = fun["part",1];
		len = If[vars["hasHead", List], vars["length"], 1];
		If[ len =!= Length[args], 
			ThrowException[{"A function is being invoked where the number of arguments doesn't match the number of parameters", fun, args}]];
		boundVars = fun["part", 1]["arguments"];
		body = fun["part", 2]["clone"];
		args = MapThread[addTyped, {boundVars, args}];
		boundVars = stripTyped /@ boundVars;
		Do[
			body = MExprSubstitute[body, replacement],
			{replacement, Thread[Rule[boundVars, args]]}
		];
		body
	]

addTyped[bound_, arg_] :=
	If[MExprNormalQ[bound] && bound["hasHead", Typed],
	   With[{
	      hd = CreateMExprSymbol[Typed],
	      args = {stripTyped[arg], bound["part", 2]}
	   },
	      CreateMExprNormal[hd, args]
	   ],
	   arg
	]
	
stripTyped[mexpr_] :=
	If[MExprNormalQ[mexpr] && mexpr["hasHead", Typed],
	   mexpr["part", 1],
	   mexpr
	]

End[]

EndPackage[]
