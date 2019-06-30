BeginPackage["Compile`Core`IR`Lower`Primitives`SizeOf`"]

Begin["`Private`"]


Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Callback`"]
Needs["CompileAST`Create`Construct`"]

(*
  This needs to use parseTypeMarkupLower on the argument because it gets lowered 
  without a need for a wrapper such as TypeOf.
*)	
lower[state_, mexpr_, opts_] :=
	Module[{body, arg, trgt, builder, inst},
		If[mexpr["length"] =!= 1,
			ThrowException[{"Invalid number of argument in " <> mexpr["toString"]}]
		];
		body = state["parseTypeMarkupLower", mexpr["part", 1], opts];
		arg = If[MissingQ[body["variable"]],
			arg = CreateConstantValue[CreateMExpr[Undefined]];
			arg["setType", body["type"]];
			arg,
			body["variable"]
		];
		
		builder = state["builder"];
		trgt = state["createFreshVariable", mexpr];
		inst = builder["createCallInstruction",
			trgt,
			CreateConstantValue[Native`SizeOf],
			{arg},
			mexpr
		];
		trgt
	];


RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Native`SizeOf], lower]
]]

End[]

EndPackage[]
