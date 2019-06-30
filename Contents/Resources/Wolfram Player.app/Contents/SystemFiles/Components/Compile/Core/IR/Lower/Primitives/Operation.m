BeginPackage["Compile`Core`IR`Lower`Primitives`Operation`"]

AddBinaryFunction

AddUnaryFunction

Begin["`Private`"]

Needs["Compile`Core`IR`ConstantValue`"]
Needs["CompileAST`Create`Construct`"]
Needs["CompileAST`Class`Symbol`"]
Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["Compile`Utilities`Language`Attributes`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`"]
Needs["CompileUtilities`Callback`"]



sortOrderlessArgs[{arg_}] :=
	arg
sortOrderlessArgs[args_] :=
	SortBy[
	    args,
	    Function[{elem},
	        If[ConstantValueQ[elem],
	            0,
	            1
	        ]
	    ]
	]

createBinaryInstruction[builder_, trgt_, op_, arg1_, arg2_, mexpr_] :=
	builder["createBinaryInstruction",
			trgt,
			CreateConstantValue[op],
			{arg1, arg2},
			mexpr
	]

createCompareInstruction[builder_, trgt_, op_, arg1_, arg2_, mexpr_] :=
	builder["createCompareInstruction",
			trgt,
			CreateConstantValue[op],
			{arg1, arg2},
			mexpr
	]

eval[op_, args_] :=
	Apply[ op, args]


lower2[state_, op_, {arg1In_, arg2In_}, mexpr_, insFun_, opts_] :=
	Module[{builder, trgt, inst, arg1, arg2},
		builder = state["builder"];
		arg1 = state["lower", arg1In, opts];
		arg2 = state["lower", arg2In, opts];
		trgt = state["createFreshVariable", mexpr];
		inst = If[ConstantValueQ[arg1] && ConstantValueQ[arg2],
			With[{
				constant = eval[op, {arg1["value"], arg2["value"]}]
			},
			With[{
				constantValue = CreateConstantValue[constant]
			},
				inst = builder["createCopyInstruction",
					trgt,
					constantValue,
					mexpr
				];
				trgt= constantValue;
				inst
			]],
			insFun[builder,
					trgt,
					op,
					arg1, arg2,
					mexpr
			]
		];
		trgt
	]

orderlessQ[mexpr_] :=
	With[{hd = mexpr["head"]},
		If[MExprSymbolQ[hd],
			With[{attrs = Lookup[$SystemAttributes, hd["fullName"], {}]},
			    MemberQ[attrs, Orderless]
			],
			False
		]
	]

lowerNary[ state_, mexpr_, opts_] :=
	Module[ {args},
		args = mexpr["arguments"];
	    If[orderlessQ[mexpr],
	    	args = sortOrderlessArgs[args]  
	    ];
	    Fold[lower2[state, mexpr["getHead"], {#1, #2}, mexpr, createBinaryInstruction, opts]&, args]
	]	
	
lowerUnaryBinary[ state_, mexpr_, opts_] :=
	Module[ {},
		Switch[Length[mexpr["arguments"]],
			1,
				lowerUnary[state, mexpr, opts],
			2,
				lowerBinary[state, mexpr, opts],
			_,
				ThrowException[LanguageException[{"Expression is expected to have 1 or 2 arguments ", mexpr["toString"]}]]
		]
	]	
	
lowerBinary[ state_, mexpr_, opts_] :=
	Module[ {args},
		args = mexpr["arguments"];
	    If[orderlessQ[mexpr],
	    	args = sortOrderlessArgs[args]  
	    ];
		If[ Length[ args] =!= 2,
			ThrowException[LanguageException[{"Binary instructions are expected to have 2 arguments ", mexpr["toString"]}]]
		];	
		lower2[state, mexpr["getHead"], {args[[1]], args[[2]]}, mexpr, createBinaryInstruction, opts]
	]	
	
lowerCompare[ state_, mexpr_, opts_] :=
	Module[ {args},
		args = mexpr["arguments"];
		If[ Length[ args] =!= 2,
			ThrowException[LanguageException[{"Compare instructions are expected to have 2 arguments ", mexpr["toString"]}]]
		];	
		lower2[state, mexpr["getHead"], {args[[1]], args[[2]]}, mexpr, createCompareInstruction, opts]
	]	
	

lowerUnary[state_, mexpr_, opts_]  :=
	Module[{args, op, builder, trgt, inst, arg},
		If[ Length[ mexpr["arguments"]] =!= 1,
			ThrowException[LanguageException[{"Unary instructions are expected to have 1 argument ", mexpr["toString"]}]]
		];
	    args = state["lower", #, opts]& /@ mexpr["arguments"];
		builder = state["builder"];
		op = mexpr["getHead"];
		arg = First[args];
		trgt = state["createFreshVariable"];
		inst = If[ConstantValueQ[arg],
			With[{
				constant = eval[op, {arg["value"]}]
			},
			With[{
				constantValue = CreateConstantValue[constant]
			},
				inst = builder["createCopyInstruction",
					trgt,
					constantValue,
					mexpr
				];
				trgt= constantValue;
				inst
			]],
			builder["createUnaryInstruction",
					trgt,
					CreateConstantValue[op],
					arg,
					mexpr
			]
		];
		trgt
	]

RegisterCallback["RegisterPrimitive", Function[{st},
Do[
	RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[op], lowerBinary],
	{
		op,
		{
			Subtract,
			Divide,
			Mod,
			Quotient,
			Power,
			BitAnd,
			BitOr,
			BitXor,
			BitShiftLeft,
			BitShiftRight,
			Chop,
			"AbsErr",
			"RelErr",
			"MaxAbs",
			"IntExp2",
			"IntLen2",
			"Unitize2"
		}
	}
]
]]

RegisterCallback["RegisterPrimitive", Function[{st},
Do[
	RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[op], lowerNary],
	{
		op,
		{
			Plus,
			Times
		}
	}
]
]]

RegisterCallback["RegisterPrimitive", Function[{st},
Do[
	RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[op], lowerUnary];
	,
	{
		op,
		{
			Sin,
			Cos,
			Tan,
			Csc,
			Sec,
			Cot,
			Sinh,
			Cosh,
			Tanh,
			Csch,
			Sech,
			Coth,
			ArcSin,
			ArcCos,
			"ArcTan_1",
			ArcCsc,
			ArcSec,
			ArcCot,
			ArcSinh,
			ArcCosh,
			ArcTanh,
			ArcCsch,
			ArcSech,
			ArcCoth,
			"AbsSquare",
			Exp,
			"Log_1",
			Log2,
			Log10,
			Abs,
			Arg,
			Conjugate,
			Im,
			Re,
			Minus,
			Sign,
			IntegerPart,
			EvenQ,
			OddQ,
			(*Square,*)
			Sqrt,
			CubeRoot,
			Internal`ReciprocalSqrt,
			"Reciprocal",
			"BitOp1Arg",
			BitNot,
			BitLength,
			"IntExp1",
			UnitStep,
			Sinc,
			Fibonacci,
			LucasL,
			Gudermannian,
			InverseGudermannian,
			Haversine,
			InverseHaversine,
			Erfc,
			Erf,
			Gamma,
			LogGamma,
			Unitize,
			"Mod1",
			Not
		}
	}
]
]]

RegisterCallback["RegisterPrimitive", Function[{st},
Do[
	RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[op], lowerUnaryBinary],
	{
		op,
		{
			Log,
			ArcTan
		}
	}
]
]]

RegisterCallback["RegisterPrimitive", Function[{st},
Do[
	RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[op], lowerCompare],
	{
		op,
		{
			SameQ,
			UnsameQ,
			Less,
			LessEqual,
			Equal,
			GreaterEqual,
			Greater,
			Unequal
		}
	}
]
]]


AddUnaryFunction[sym_] :=
	RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[sym], lowerUnary]

AddBinaryFunction[ sym_] :=
	RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[sym], lowerBinary]

End[]

EndPackage[]
