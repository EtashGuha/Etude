BeginPackage["Compile`Core`IR`Lower`Primitives`Part`"]

Begin["`Private`"]

Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileUtilities`Callback`"]



(*
TODO,  this is all obsolete and can be removed.


 TODO think about multi-arg Part instructions.
 
 Set[ Part[ expr, args], rhs]
 
*)
lowerSetPart[state_, mexpr_, opts_] :=
	Module[{builder, trgt, offset, rhs, structureExpr, targetExpr, offsetExpr},
		builder = state["builder"];
		If[mexpr["hasHead", Compile`Internal`UnsafeSetPart] || mexpr["hasHead", Primitive`SetPart],
			(* The form of UnsafeSetPart is 
			 * UnsafeSetPart[lst, index, val]
			 *)
			rhs = state["lower", mexpr["part", 3], opts];
			targetExpr = mexpr["part", 1];
			offsetExpr = mexpr["part", 2]
			,
			(* Else *)
			rhs = state["lower", mexpr["part", 2], opts];
			structureExpr = mexpr["part", 1];
			targetExpr = structureExpr["part", 1];
			offsetExpr = structureExpr["part", 2]
		];
		offset = state["lower", offsetExpr, opts];
		trgt = state["lower", targetExpr, opts];
		builder["createSetElementInstruction", trgt, offset, rhs, mexpr];
		trgt
	]
	
(* In the future lowerSet part may introduce a tensor copy
 * but a get part would not
 *)

(*
 TODO figure out error for Part of 0 args.  Happens later.
*)
lowerGetPart[state_, mexpr_, opts_] :=
	Module[{builder, args, trgt},
	    args = state["lower", #, opts]& /@ mexpr["arguments"];
	    builder = state["builder"];
		If[ Length[args] === 1, 
			First[args],
			trgt = state["createFreshVariable", mexpr];
			builder["createGetElementInstruction", trgt, First[args], Rest[args], mexpr];
			trgt
		]
	]


RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Primitive`Part], lowerGetPart]
]]

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Primitive`SetPart], lowerSetPart]
]]

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive["UnsafeSetPart"], lowerSetPart]
]]

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive["UnsafeGetPart"], lowerGetPart]
]]

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive["GetPart"], lowerGetPart]
]]

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive["SetPart"], lowerSetPart]
]]

End[]

EndPackage[]
