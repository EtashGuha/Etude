BeginPackage["Compile`Core`IR`Lower`Primitives`Inert`"]

Begin["`Private`"]

Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileUtilities`Callback`"]



lower[state_, mexpr_, opts_] :=
	Module[{args, fun, builder, trgt, inst},
		builder = state["builder"];
	    args = state["lower", #, opts]& /@ mexpr["arguments"];
		fun = state["lower", mexpr["head"], opts];
		trgt = state["createFreshVariable", mexpr];
		inst = builder["createInertInstruction",
			trgt,
			fun,
			args,
			mexpr
		];
		trgt
	]
	

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive["Inert"], lower]
]]

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Association], lower]
]]

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Rational], lower]
]]

End[]

EndPackage[]
