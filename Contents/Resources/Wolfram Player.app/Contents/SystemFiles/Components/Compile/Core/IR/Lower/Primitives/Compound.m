BeginPackage["Compile`Core`IR`Lower`Primitives`Compound`"]

Begin["`Private`"]

Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileUtilities`Callback`"]



lower[state_, mexpr_, opts_] :=
	Module[{trgt = Null},
		Do[
			trgt = state["lower", arg, opts],
			{arg, mexpr["arguments"]}
		];
		trgt
	]


RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[CompoundExpression], lower]
]]


End[]

EndPackage[]
