BeginPackage["Compile`Core`IR`Lower`Primitives`Error`"]

Begin["`Private`"]

Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Error`Exceptions`"]
Needs["CompileAST`Export`FromMExpr`"]


(*
  Could also check argument lengths etc...
*)

lower[state_, mexpr_, opts_] :=
	Module[{},
	    ThrowException[ReleaseHold[FromMExpr[mexpr]]]
	]

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Compile`Error], lower]
]]

End[]

EndPackage[]
