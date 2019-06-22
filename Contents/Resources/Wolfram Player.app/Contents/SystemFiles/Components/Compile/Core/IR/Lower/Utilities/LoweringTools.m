
BeginPackage["Compile`Core`IR`Lower`Utilities`LoweringTools`"]

LowerGeneral
AddLowerGeneral

AddLowerGeneralAtom

Begin["`Private`"] 

Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileAST`Class`Normal`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`Core`IR`Lower`Primitives`Atom`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileAST`Class`Symbol`"]

AddLowerGeneral[fun_] :=
	Module[ {prim},
		prim = CreateSystemPrimitive[fun];
		If[!LanguagePrimitiveQ[prim],
			ThrowException[{"Could not create system primitive for: ", fun}]
		];
		If[!KeyExistsQ[ $LanguagePrimitiveLoweringRegistry, prim[ "fullName"]],
			RegisterLanguagePrimitiveLowering[prim, LowerGeneral]
		]
	]


LowerGeneral[state_, mexpr_] :=
	LowerGeneral[state, mexpr, <||>]
LowerGeneral[state_, mexpr_, opts_] :=
	Module[{inst},
	    inst = lowerWorker[state, mexpr, opts];
		inst["target"]
	]

(*
 If the head is compound, then process it into a non mexpr form, so the
 ConstantValue can be created.
*)
lowerWorker[state_, mexpr_, opts_] :=
	Module[{args, fun, builder, trgt, inst, head},
	    args = state["lower", #, opts]& /@ mexpr["arguments"];
		builder = state["builder"];
		head = mexpr["head"];
		If[ MExprNormalQ[head],
			head = getNormalForm[head]
		];
		fun = CreateConstantValue[head];
		trgt = state["createFreshVariable", mexpr];
		inst = builder["createCallInstruction",
			trgt,
			fun,
			args,
			mexpr
		];
		inst
	]


getAtom[ mexpr_] :=
	Which[
		mexpr["symbolQ"],
			ReleaseHold[mexpr["data"]],
		mexpr["atomQ"],
			mexpr["data"],
		True,
			ThrowException[CompilerException[{"Cannot create form for constant ", mexpr}]]
	]

(*
 The non-mexpr form is head[ args]  where head and args are atomic themselves.
 If more complex forms are needed I don't think this should go through LowerGeneral.
*)
getNormalForm[ mexpr_] :=
	Module[ {hd, args},
		hd = getAtom[mexpr["head"]];
		args = Map[ getAtom, mexpr["arguments"]];
		hd @@ args
	]



AddLowerGeneralAtom[ sym_] :=
	RegisterLowerConstantAtom[sym]


lowerFunctionCall[state_, mexpr_, opts_] :=
	Module[{args, fun, builder, trgt, inst, name, head},
	    args = state["lower", #, opts]& /@ mexpr["arguments"];
		builder = state["builder"];
		head = mexpr["head"];
		If[ !MExprSymbolQ[head],
			ThrowException[{"Unexpected form for FunctionCall", mexpr}]];
		name = state["processFunctionName", mexpr["head"]];
		fun = CreateConstantValue[name];
		trgt = state["createFreshVariable", mexpr];
		inst = builder["createCallInstruction",
			trgt,
			fun,
			args,
			mexpr
		];
		inst["target"]
	]



RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive["FunctionCall"], lowerFunctionCall]
]]



End[] 

EndPackage[]
