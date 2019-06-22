BeginPackage["Compile`Core`IR`Lower`Primitives`Return`"]

Begin["`Private`"]

Needs["Compile`Core`IR`ConstantValue`"]
Needs["CompileAST`Create`Construct`"]
Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`"]
Needs["CompileUtilities`Callback`"]




nullExpr := nullExpr = CreateMExpr[Null]


lower[state_, mexpr_, opts_] := 
	Module[ {val, builder = state["builder"]},
		Which[
			mexpr["length"] === 0,
				val = CreateConstantValue[Compile`Void],
			mexpr["length"] === 1,
				val = state["lower", mexpr["part", 1], opts],
			True,
				ThrowException[LanguageException[{"Return is expected to have 0 or 1 arguments", mexpr["toString"]}]]
		];
		builder["currentFunctionModuleBuilder"]["addReturn", builder["currentBasicBlock"], val];
		builder["currentFunctionModuleBuilder"]["setReturnMode", True];
		Null
	]	

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Return], lower]
]]

End[]

EndPackage[]
