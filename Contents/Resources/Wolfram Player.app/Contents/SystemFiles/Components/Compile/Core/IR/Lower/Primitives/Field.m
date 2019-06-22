BeginPackage["Compile`Core`IR`Lower`Primitives`Field`"]


Begin["`Private`"]

Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`"]
Needs["CompileUtilities`Callback`"]



fieldName[state_, mexpr_, opts_] :=
	Which[
		mexpr["length"] =!= 2,
			ThrowException[{"Invalid number of argument in " <> mexpr["toString"]}]
		,
		mexpr["part", 2]["hasHead", String],
			CreateConstantValue[mexpr["part", 2]],
		mexpr["length"] === 2,
			state["lower", mexpr["part", 2], opts]
	]
	
lowerGetField[state_, mexpr_, opts_] :=
	Module[{builder, trgt, src, offset},
		builder = state["builder"];
		trgt = state["createFreshVariable", mexpr];
		src = state["lower", mexpr["part", 1], opts];
		offset = fieldName[state, mexpr];
		builder["createGetFieldInstruction", trgt, src, offset, mexpr];
		trgt 
	]

lowerSetField[state_, mexpr_, opts_] :=
	Module[{builder, trgt, field, rhs, structureExpr, targetExpr, fieldExpr},
		builder = state["builder"];
		rhs = state["lower", mexpr["part", 2]];
		structureExpr = mexpr["part", 1];
		targetExpr = structureExpr["part", 1];
		fieldExpr = structureExpr["part", 2];
		trgt = state["lower", targetExpr, opts];
		field = state["lower", fieldExpr, opts];
		builder["createSetFieldInstruction", trgt, field, rhs, mexpr];
		trgt
	]
	

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Field], lowerGetField]
]]

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive["GetField"], lowerGetField]
]]

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive["SetField"], lowerSetField]
]]


End[]

EndPackage[]
