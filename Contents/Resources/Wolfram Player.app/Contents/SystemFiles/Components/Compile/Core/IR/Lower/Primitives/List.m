BeginPackage["Compile`Core`IR`Lower`Primitives`List`"]

Begin["`Private`"]

Needs["Compile`Core`IR`ConstantValue`"]
Needs["CompileAST`Class`Literal`"]
Needs["CompileAST`Class`Normal`"]
Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Callback`"]


ConstantMTensorQ[ mexpr_] :=
	Module[ {d},
		If[ mexpr["isList"],
			d = GetTensorData[mexpr];
			ListQ[d],
			False
		]
	]

GetTensorData[mexpr_?MExprNormalQ] :=
    Module[ {d1, d2, i, err = False},
        If[ ! mexpr["isList"],
            Return[Null]];
        If[ mexpr["length"] === 0,
            Return[Null]];
        d1 = GetTensorData[mexpr["part", 1]];
        If[ d1 === Null,
            Return[Null]];
        Do[
         d2 = GetTensorData[mexpr["part", i]];
         If[ d1 =!= d2,
             err = True;
             Break[]],
         {i, 2, mexpr["length"]}];
        If[ err,
            Null,
            Prepend[d1, mexpr["length"]]]
    ]

GetTensorData[mexpr_] :=
    Module[ {h},
        If[ ! MExprLiteralQ[mexpr],
            Return[Null]];
        h = mexpr["getHead"];
        If[ MemberQ[{Integer, Real, Complex}, h],
            {h},
            Null]
    ]

lowerListCreate[state_, mexpr_, opts_] :=
	Module[{inst},
	    inst = lowerWorker[state, mexpr, opts];
		If[ ConstantMTensorQ[mexpr],
			inst["setProperty", "constantMTensor" -> True]
		];
		inst["target"]
	]

lowerGeneral[state_, mexpr_, opts_] :=
	Module[{inst},
	    inst = lowerWorker[state, mexpr, opts];
		inst["target"]
	]


lowerWorker[state_, mexpr_, opts_] :=
	Module[{args, fun, builder, trgt, inst},
	    args = state["lower", #, opts]& /@ mexpr["arguments"];
		builder = state["builder"];
		fun = CreateConstantValue[mexpr["head"]];
		trgt = state["createFreshVariable", mexpr];
		inst = builder["createCallInstruction",
			trgt,
			fun,
			args,
			mexpr
		];
		inst
	]

lowerFunctionCall[len_, txt_][state_, mexpr_, opts_] :=
	Module[{},
		If[ mexpr["length"] =!= len,
			ThrowException[{txt, mexpr}]
		];
		lowerGeneral[state, mexpr, opts]		
	]


RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[List], lowerListCreate]
]]

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Primitive`TensorRank], lowerGeneral]
]]

RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[
			Native`IteratorCount], lowerFunctionCall[3,"Malformed IteratorCount"]]
]]

End[]

EndPackage[]
