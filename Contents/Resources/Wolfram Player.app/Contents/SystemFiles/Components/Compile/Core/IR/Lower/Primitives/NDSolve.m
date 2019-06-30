BeginPackage["Compile`Core`IR`Lower`Primitives`NDSolve`"]

Begin["`Private`"]

Needs["CompileAST`Export`FromMExpr`"]
Needs["CompileAST`Create`Construct`"]
Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileUtilities`Callback`"]


		
lower[state_, mexpr_, opts_] :=
	With[{args = ReleaseHold[FromMExpr[#]]& /@ mexpr["arguments"]},
		With[{fun = First[Apply[NDSolve`ProcessEquations, args]]["NumericalFunction"][[1, 1]]},
			state["lower", CreateMExpr[fun], opts]
		]
	]

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[NDSolve], lower]
]]


End[]

EndPackage[]
