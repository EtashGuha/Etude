BeginPackage["Compile`Core`IR`Lower`Primitives`StackAllocate`"]

Begin["`Private`"]

Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileAST`Export`FromMExpr`"]
Needs["TypeFramework`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`"]
Needs["CompileUtilities`Callback`"]




lowerLoad[state_, mexpr_, opts_] :=
	Module[{src, builder, trgt, inst},
		If[mexpr["length"] =!= 1,
			ThrowException[LanguageException[{Primitive`Load, "argcount", mexpr["toString"]}]]
		];
	    src = state["lower", mexpr["part",1], opts];
		builder = state["builder"];
		trgt = state["createFreshVariable", mexpr];
		inst = builder["createLoadInstruction", trgt, src, CreateConstantValue[Native`Load], mexpr];
		trgt
	]
	
lowerStore[state_, mexpr_, opts_] :=
	Module[{trgt, val, builder, inst},
		If[mexpr["length"] =!= 2,
			ThrowException[LanguageException[{Primitive`Store, "argcount", mexpr["toString"]}]]
		];
	    trgt = state["lower", mexpr["part",1], opts];
	    val = state["lower", mexpr["part",2], opts];
		builder = state["builder"];
		inst = builder["createStoreInstruction", val, trgt,  CreateConstantValue[Native`Store], mexpr];
		CreateVoidValue[]
	]


isField[ mexpr_] :=
	mexpr["normalQ"] &&
	With[{
		h = mexpr["head"]	
	},
		h["symbolQ"] &&
		h["fullName"] === "Native`Field" &&
		mexpr["length"] === 1 &&
		mexpr["part", 1]["literalQ"]
	]

getTypeVariable[ base_, offset_, trgt_] :=
	TypeVariable[StringJoin[base, ToString[offset["id"]], ToString[trgt["id"]]]]

lowerSetElement[state_, mexpr_, opts_] :=
	Module[{trgt, offsetExpr, offsetVal, offset, operator, src, builder,inst, resTy, tyVar1, tyVar2},
		If[mexpr["length"] =!= 3,
			ThrowException[LanguageException[{Native`SetElement, "argcount", mexpr["toString"]}]]
		];
	    trgt = state["lower", mexpr["part",1], opts];
	    offsetExpr = mexpr["part",2];
	    src = state["lower", mexpr["part",3], opts];
		operator = CreateConstantValue[Native`SetElement];
	    If[ isField[offsetExpr],
	    	offsetVal = offsetExpr["part", 1]["data"];
	    	If[ !IntegerQ[offsetVal],
	    		ThrowException[LanguageException[{"The field ", offsetVal, " in ", mexpr["toString"], " for Native`SetElement is not an integer"}]]
	    	];
	    	offset = CreateConstantValue[offsetVal];
	    	tyVar1 = getTypeVariable[ "a$", offset, trgt];
	    	tyVar2 = getTypeVariable[ "b$", offset, trgt];
	    	resTy = TypeSpecifier[ {"Handle"[tyVar1], tyVar2, TypeProjection[tyVar1, offsetVal]} -> "Void"];
	    	operator["setType", resTy];
	    	,
	    	offset = state["lower", mexpr["part",2], opts]
	    ];
  		builder = state["builder"];
		inst = builder["createSetElementInstruction", trgt, offset, src, operator, mexpr];
		CreateVoidValue[]
	]
	

	
lowerGetElement[state_, mexpr_, opts_] :=
	Module[{trgt, src, offsetExpr, offsetVal, offset, operator, builder, inst, resTy, tyVar1, tyVar2},
		If[mexpr["length"] =!= 2,
			ThrowException[LanguageException[{Native`GetElement, "argcount", mexpr["toString"]}]]
		];
		trgt = state["createFreshVariable", mexpr];
	    src = state["lower", mexpr["part", 1], opts];
	    offsetExpr = mexpr["part",2];
	    operator = CreateConstantValue[Native`GetElement];
	    If[ isField[offsetExpr],
	    	offsetVal = offsetExpr["part", 1]["data"];
	    	If[ !IntegerQ[offsetVal],
	    		ThrowException[LanguageException[{Native`GetElement, "field", mexpr["toString"]}]]
	    	];
	    	offset = CreateConstantValue[offsetVal];
			tyVar1 = getTypeVariable[ "a$", offset, trgt];
	    	tyVar2 = getTypeVariable[ "b$", offset, trgt];
	    	resTy  = TypeSpecifier[ {"Handle"[tyVar1], tyVar2} -> TypeProjection[tyVar1, offsetVal]];
	    	operator["setType", resTy];
	    	,
	    	offset = state["lower", mexpr["part",2]]
	    ];
		builder = state["builder"];
		inst = builder["createGetElementInstruction", trgt, src, offset, operator, mexpr];
		trgt
	]

	
lowerGetField[state_, mexpr_, opts_] :=
	Module[{trgt, src, offsetExpr, offsetVal, offset, operator, builder, inst, resTy, tyVar1, tyVar2},
		If[mexpr["length"] =!= 2,
			ThrowException[LanguageException[{Native`GetField, "argcount", mexpr["toString"]}]]
		];
		trgt = state["createFreshVariable", mexpr];
	    src = state["lower", mexpr["part", 1], opts];
	    offsetExpr = mexpr["part",2];
	    operator = CreateConstantValue[Native`GetField];
	   	offsetVal = offsetExpr["data"];
    	If[ !IntegerQ[offsetVal],
    		ThrowException[LanguageException[{Native`GetField, "field", mexpr["toString"]}]]
    	];
    	offset = CreateConstantValue[offsetVal];
    	tyVar1 = TypeVariable[StringJoin["a$", ToString[offset["id"]]]];
    	tyVar2 = TypeVariable[StringJoin["b$", ToString[offset["id"]]]];
    	resTy  = TypeSpecifier[ {tyVar1, tyVar2} -> TypeProjection[tyVar1, offsetVal]];
    	operator["setType", resTy];
		builder = state["builder"];
		inst = builder["createGetElementInstruction", trgt, src, offset, operator, mexpr];
		trgt
	]


(*
 Only allocates a single space and give a Handle generic type.
*)
lowerCreateHandle[state_, mexpr_, opts_] :=
	Module[{builder, trgt, inst, handTy},
		If[mexpr["length"] =!= 0,
			ThrowException[LanguageException[{Native`Handle, "argcount", mexpr["toString"]}]]
		];
		builder = state["builder"];
		trgt = state["createFreshVariable", mexpr];
		inst = builder["createStackAllocateInstruction", trgt, CreateConstantValue[1], CreateConstantValue[Native`StackAllocate], mexpr];
		handTy = TypeSpecifier["Handle"[ makeVariable["a", mexpr]]];
		trgt["setType", handTy];
		trgt
	]


makeVariable[ base_, mexpr_] :=
	TypeVariable[base <> ToString[mexpr["id"]]]

(*
 Only allocates a single space and give a Handle generic type.
*)
lowerCreateTupleHandle[state_, mexpr_, opts_] :=
	Module[{trgt, ty, var1 = makeVariable["a1", mexpr], var2 = makeVariable["a2", mexpr]},
		If[mexpr["length"] =!= 0,
			ThrowException[LanguageException[{Native`CreateTupleHandle, "argcount", mexpr["toString"]}]]
		];
		trgt = lowerCreateHandle[state, mexpr, opts];
		ty = TypeSpecifier["Handle"[ "Tuple"[var1, var2]]];
		trgt["setType", ty];
		trgt
	]
	
lower[state_, mexpr_, name_, opts_] :=
	Module[{size, builder, trgt, inst},
		If[mexpr["length"] =!= 1,
			ThrowException[LanguageException[{name, "argcount", mexpr["toString"]}]]
		];
	    size = state["lower", mexpr["part",1], opts];
		builder = state["builder"];
		trgt = state["createFreshVariable", mexpr];
		inst = builder["createStackAllocateInstruction", trgt, size, CreateConstantValue[Native`StackAllocate], mexpr];
		inst
	]

(*
 Takes a size argument and gives a CArray generic type.
*)
lowerStackArray[state_, mexpr_, opts_] :=
	Module[ {inst, trgt, ty},
		inst = lower[state, mexpr, Native`StackArray, opts];
		trgt = inst["target"];
		ty = TypeSpecifier["CArray"[ TypeVariable["a$" <> ToString[trgt["id"]] ]]];
		trgt["setType", ty];
		trgt
	]


lowerStackAllocateObject[state_, mexpr_, opts_] :=
	Module[{builder, tyArg, tyObj, kind, trgt, varArgs, inst, objTy},
		If[mexpr["length"] =!= 1,
			ThrowException[LanguageException[{Native`StackAllocateObject, "argcount", mexpr["toString"]}]]
		];
		builder = state["builder"];
		tyArg = ReleaseHold[ FromMExpr[mexpr["part", 1]]];
		tyObj = state["typeEnvironment"]["resolve", TypeSpecifier[tyArg]]; (* We need to figure out the kind here, so resolve is needed *)
		kind = tyObj["kind"];
		If[!kind["isFunction"],
			ThrowException[LanguageException[{Native`StackAllocateObject, "typeKind", mexpr["toString"]}]]
		];
		varArgs = Table[TypeVariable[ "varField$" <> ToString[ii] <> ToString[mexpr["id"]]], {ii, Length[kind["arguments"]]}]; 
		trgt = state["createFreshVariable", mexpr];
		inst = builder["createStackAllocateInstruction", trgt, CreateConstantValue[1], CreateConstantValue[Native`StackAllocateObject], mexpr];
		objTy = TypeSpecifier["Handle"[Apply[tyArg, varArgs]]];
		trgt["setType", objTy];
		trgt
	]


(*
 Takes a size argument and gives a CArray generic type.
*)
lowerField[state_, mexpr_, opts_] :=
	Module[ {val, exprVal, res, ty},
		If[mexpr["length"] =!= 1,
			ThrowException[LanguageException[{Native`Field, "argcount", mexpr["toString"]}]]
		];
		val = mexpr["part",1];
		If[!val["literalQ"],
			ThrowException[LanguageException[{Native`Field, "literal", mexpr["toString"]}]]
		];
		exprVal = val["data"];
		If[ !IntegerQ[exprVal],
			ThrowException[LanguageException[{Native`Field, "literal", mexpr["toString"]}]]
		];
		res = state["lower", val, opts];
		ty = TypeSpecifier[TypeLiteral[exprVal, "MachineInteger"]];
		res["setType", ty];
		res
	]

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Native`StackAllocateObject], lowerStackAllocateObject]
]]


(*RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Native`Field], lowerField]
*)

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Native`StackArray], lowerStackArray]
]]

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Native`SetElement], lowerSetElement]
]]

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Native`GetElement], lowerGetElement]
]]

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Native`Handle], lowerCreateHandle]
]]

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Native`Load], lowerLoad]
]]
RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Native`Store], lowerStore]
]]

RegisterCallback["RegisterPrimitive", 
	Function[{st},

		RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Native`CreateTupleHandle], 
					lowerCreateTupleHandle];

		RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Native`GetField], 
					lowerGetField];
	]]


End[]

EndPackage[]
