BeginPackage["Compile`Core`IR`Lower`Primitives`Atom`"]

RegisterLowerConstantAtom

Begin["`Private`"]

Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`"]
Needs["CompileUtilities`Error`Exceptions`"]
Needs["CompileAST`Export`FromMExpr`"]

lower[state_, mexpr_, opts_] := 
	CreateConstantValue[mexpr]

lowerConstant[state_, mexpr_, opts_] := 
	Module[ {},
		If[ mexpr["length"] =!= 1,
			ThrowException[LanguageException[{"ConstantValue is expected to have 1 argument", mexpr}]]
		];
		CreateConstantValue[mexpr["part", 1]]
	]


isString[ mexpr_] :=
	mexpr["literalQ"] && mexpr["head"]["symbolQ"] && mexpr["head"]["data"] === HoldComplete[String]


lowerConstantExpression[state_, mexpr_, opts_] :=
	Module[{builder, arg, data, trgt, cons, ty, inst},
	    builder = state["builder"];
		If[ mexpr["length"] =!= 1,
			ThrowException[LanguageException[{"ConstantExpression is expected to have one argument ", mexpr["toString"]}]]
		];
		arg = mexpr["part", 1];
		data = Which[
			arg["symbolQ"], Primitive`GlobalSymbol[arg["fullName"]],
			arg["literalQ"], arg["data"],
			True, FromMExpr[arg]
		];
		cons = CreateConstantValue[data];
		ty = TypeSpecifier["Expression"];
		cons["setType", ty];
		trgt = state["createFreshVariable", mexpr];
		inst = builder["createCopyInstruction", trgt, cons, mexpr];
	    trgt
	]



(*
  Lower Native`Global[ name], which just lowers to a constant value.  
  Code in ConstantValue makes sure the global is created correctly.
*)
lowerGlobal[state_, mexpr_, opts_] := 
	Module[ {},
		If[ mexpr["length"] =!= 1,
			ThrowException[LanguageException[{"Global is expected to have 1 argument", mexpr}]]
		];
		CreateConstantValue[mexpr]
	]




RegisterCallback["RegisterPrimitive", Function[{st},
	With[{cons = {
		True,
		False,
		Catalan,
		Degree,
		E,
		EulerGamma, 
		Glaisher,
		GoldenAngle,
		GoldenRatio,
		Khinchin,
		MachinePrecision,
		Pi
	}},
		Scan[ RegisterLanguagePrimitiveLowering[CreateSystemPrimitiveAtom[#], lower]&, cons]
	];
	RegisterLanguagePrimitiveLowering[CreateSystemPrimitiveLiteral[], lower];
	RegisterLanguagePrimitiveLowering[CreateSystemPrimitiveAtom[Undefined], lower];
	RegisterLanguagePrimitiveLowering[CreateSystemPrimitiveAtom[Compile`Uninitialized], lower];
	RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Compile`ConstantValue], lowerConstant];
	RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Native`ConstantExpression], lowerConstantExpression];
	RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Native`Global], lowerGlobal];
]]


RegisterLowerConstantAtom[ atom_] :=
Module[{prim},
	prim = CreateSystemPrimitiveAtom[atom];
	If[!LanguagePrimitiveQ[prim],
		ThrowException[{"Could not create system primitive for: ", atom}]
	];
	If[!KeyExistsQ[ $LanguagePrimitiveLoweringRegistry, prim[ "fullName"]],
		RegisterLanguagePrimitiveLowering[prim, lower]
	]
]

End[]

EndPackage[]
