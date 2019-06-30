BeginPackage["Compile`Core`IR`Lower`Primitives`Typed`"]

Begin["`Private`"]

Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`"]
Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"] (* For TypeVariable *)
Needs["CompileUtilities`Asserter`Assert`"]
Needs["CompileAST`Create`Construct`"]

	
lowerTyped[state_, mexpr_, opts_] :=
	Module[ {arg, ty,varAndty, ef},
		If[ mexpr["length"] =!= 2,
			ThrowException[LanguageException[{"Typed is expected to have 2 arguments ", mexpr["toString"]}]]
		];
		arg = mexpr["part", 1];
		ty =  mexpr["part", 2];
		varAndty = state["parseTypeMarkup", ty, opts];
		ty = varAndty["type"];
		ef = state["lower", arg];
		ef["setType", ty];
		ef
	]

(*
  Process TypeJoin inside of Cast.  This needs to use parseTypeMarkupLower on 
  each argument because these get lowered.
*)	
lowerTypeJoin[ state_, mexpr_, opts_] :=
	Module[{
			builder, trgt,
			src1, src1VarTy, src1Ty,
			src2, src2VarTy, src2Ty,
			fun, inst
	},
	    AssertThat["A TypeJoin has two arguments.", mexpr["length"]
			]["named", mexpr
			]["isEqualTo", 2
		];
		builder = state["builder"];
		trgt = state["createFreshVariable", mexpr];
		
		src1VarTy = state["parseTypeMarkupLower", mexpr["part", 1], opts];
		src1Ty = src1VarTy["type"];
		src1 = src1VarTy["variable"];
		If[MissingQ[src1],
			src1 = state["lower", mexpr["part", 1], opts];
			src1["setType", src1Ty];
		];
		
		src2VarTy = state["parseTypeMarkupLower", mexpr["part", 2], opts];
		src2Ty = src2VarTy["type"];
		src2 = src2VarTy["variable"];
		If[MissingQ[src1],
			src2 = state["lower", mexpr["part", 2], opts];
			src2["setType", src2Ty];
		];
		
		fun = CreateConstantValue[Compile`TypeJoin];
		inst = builder["createCallInstruction",
			trgt,
			fun,
			{src1, src2},
			mexpr
		];
	    If[trgt["type"] === Undefined,
	    	trgt["setType", TypeSpecifier[TypeVariable[ToString[trgt["id"]]]]]
	    ];
		trgt
	] 

lowerCast[ state_, mexpr_, opts_] :=
	lowerCastBase[state, mexpr, Compile`Cast, opts]

lowerConvertArray[ state_, mexpr_, opts_] :=
    lowerCastBase[state, mexpr, Compile`ConvertArray, opts]

lowerCastElements[ state_, mexpr_, opts_] :=
	lowerCastBase[state, mexpr, Compile`CastElements, opts]
	
lowerBitCast[ state_, mexpr_, opts_] :=
	lowerCastBase[state, mexpr, Native`BitCast, opts]
	
lowerReinterpretCast[ state_, mexpr_, opts_] :=
    lowerCastBase[state, mexpr, Native`ReinterpretCast, opts]

lowerCastBase[state_, mexpr_, head_, opts_] :=
	Module[ {src1, src2, src2Ty, fun, trgt, builder, inst},
		builder = state["builder"];
		If[ mexpr["length"] =!= 2,
			ThrowException[LanguageException[{ToString[head] <> " is expected to have 2 arguments ", mexpr["toString"]}]]
		];
		trgt = state["createFreshVariable", mexpr];
		src1 = state["lower", mexpr["part", 1], opts];
		src2Ty = state["parseTypeMarkup", mexpr["part", 2], opts];
		src2 = CreateConstantValue[CreateMExpr[Undefined]];
		src2["setType", src2Ty["type"]];
		fun = CreateConstantValue[head];
		inst = builder["createCallInstruction",
			trgt,
			fun,
			{src1, src2},
			mexpr
		];
		trgt
	]

	

RegisterCallback["RegisterPrimitive", 
    Function[{st},
        RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Typed], lowerTyped];
        RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Compile`TypeJoin], lowerTypeJoin];
        RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Compile`Cast], lowerCast];
        RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Compile`ConvertArray], lowerConvertArray];
        RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Compile`CastElements], lowerCastElements];
        RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Native`BitCast], lowerBitCast];
        RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Native`ReinterpretCast], lowerReinterpretCast];
]]

End[]

EndPackage[]
