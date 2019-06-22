BeginPackage["Compile`Core`IR`Lower`Primitives`Structural`"]

Begin["`Private`"]

Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileUtilities`Callback`"]




(*
  Could also check argument lengths etc...
*)

lower[state_, mexpr_, opts_] :=
	Module[{args, fun, builder, trgt, inst},
	    args = state["lower", #, opts]& /@ mexpr["arguments"];
		builder = state["builder"];
		fun = CreateConstantValue[mexpr["head"]];
		trgt = state["createFreshVariable", mexpr];
		inst = builder["createCallInstruction",
			trgt,
			fun,
			args,
			mexpr
		];
		trgt
	]

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Take], lower]
]]

End[]

EndPackage[]
