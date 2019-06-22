BeginPackage["Compile`Core`IR`Lower`Primitives`FunctionVariable`"]

AddFunctionVariable

Begin["`Private`"]

Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)


lower[state_, mexpr_, opts_] :=
	Module[{builder, trgt, fun},
		If[ !mexpr["symbolQ"],
			ThrowException[{"The argument should be a symbol ", mexpr}]
		];
		fun = mexpr["symbol"];
		builder = state["builder"];
		trgt = state["createFreshVariable"];
		builder["createLambdaInstruction", trgt, fun, mexpr];
		trgt
	]

AddFunctionVariable[sym_] :=
	RegisterLanguagePrimitiveLowering[CreateSystemPrimitiveAtom[sym], lower]

End[]

EndPackage[]
