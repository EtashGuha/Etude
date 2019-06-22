BeginPackage["Compile`Core`IR`Lower`Primitives`Function`"]

Begin["`Private`"]

Needs["Compile`Core`IR`ConstantValue`"]
Needs["CompileAST`Create`Construct`"]
Needs["CompileAST`Class`Symbol`"]
Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["Compile`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileUtilities`Asserter`Assert`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`Error`Exceptions`"]



createScopedVariable[state_, idx_, mexpr_, opts_] :=
	Module[{builder, var, inst},
		Which[
			MExprSymbolQ[mexpr],
				builder = state["builder"];
				var = state["createFreshVariable", mexpr];
				inst = builder["createLoadArgumentInstruction", var, CreateMExprLiteral[idx], mexpr, mexpr, False];
				builder["symbolBuilder"]["add", mexpr["lexicalName"], var, inst, <|"Writable" -> False|>];
				inst["setProperty", "variableDeclaration" -> True];
				builder["addArgument", var];
				var,
			state["isTypedMarkup", mexpr, opts],
				var = createScopedVariable[state, idx, mexpr["part", 1], opts];
				var["setType", state["parseTypedMarkup", mexpr, opts]["type"]];
				var,
			True,
				AssertThat["A Function variable must either be a symbol or a type annotated symbol.",
					  mexpr["toString"]
					]["fails"
				];
				False
		]
	]

createScopedVariables[state_, vars_, opts_] :=
	MapIndexed[createScopedVariable[state, First[#2], #1, opts]&, vars["arguments"]]


lower[state_, mexpr_, opts_] :=
	Module[{
		  builder = state["builder"],
		  args = mexpr["part", 1],
		  body = mexpr["part", 2],
		  prevFm = state["builder"]["currentFunctionModuleBuilder"],
		  capturedVars,
		  baseVars = CreateReference[{}],
		  name,
		  res,
		  bodyTy = Null,
		  e
		},
		AssertThat["Function is expected to have only 2 arguments", mexpr["length"]
	    		]["named", "arguments"
	    		]["isEqualTo", 2
	    	];
		name = If[prevFm === Undefined,
			state["mainName"],
			state["mainName"] <> "_" <> "Function$" <> ToString[Length[builder["getFunctionModules"]]]
		];
		builder["createFunctionModule", name, mexpr];
		
	    createScopedVariables[state, args, opts];
	    
        If[mexpr["getProperty", "closureVariablesConsumed"] =!= {},
            builder["currentFunctionModuleBuilder"]["setProperty",
                        "capturedScopedMExprNames" -> (#["lexicalName"]& /@ mexpr["getProperty", "closureVariablesConsumed"])
            ];
            capturedVars = CreateReference[{}];
            (*
             * For each closure variable, we create a call instruction that loads it using
             * Native`LoadClosureVariable the instruction has the property isClosureLoad,
             * a closure variable property of the source closure variable, and the target
             * variable has an aliasesVariable property to the closure variable.
             * Closure variables are not writable.
             *)
            Do[
	            If[builder["symbolBuilder"]["exists", e["lexicalName"]],
	                Module[{src, trgt, inst},
	                	src = builder["symbolBuilder"]["peekAssignAlias", e["lexicalName"]];
	                	If[ src === Null,
		                	src = builder["symbolBuilder"]["getVariable", e["lexicalName"]];
		                	builder["symbolBuilder"]["pushAssignAlias", e["lexicalName"], src];
		                	baseVars["appendTo", e["lexicalName"]]
		                ];
		                trgt = state["createFreshVariable", e];
		                inst = builder["createCallInstruction", trgt, CreateConstantValue[Native`LoadClosureVariable], {}, e];
		                inst["setProperty", "isClosureLoad" -> True];
                        inst["setProperty", "closureVariable" -> src];
                        src["setProperty", "isCapturedVariable" -> True];
                        src["setProperty", "capturedByVariables" -> Append[src["getProperty", "capturedByVariables", {}], trgt]];
                        trgt["setProperty", "isClosureVariable" -> True];
                        trgt["setProperty", "aliasesVariable" -> src];
		                builder["symbolBuilder"]["add", e["lexicalName"], trgt, inst, <|"Writable" -> False|>];
		                capturedVars["appendTo", src];
	                ]
	            ],
	            {e, DeleteDuplicatesBy[mexpr["getProperty", "closureVariablesConsumed"], #["lexicalName"]&]}
	        ];
            builder["currentFunctionModuleBuilder"]["setProperty", "closureVariablesConsumed" -> capturedVars]
        ]; 

		builder["currentFunctionModuleBuilder"]["setBodyType", bodyTy];

		res = state["lower", body, opts];
		
		Scan[builder["symbolBuilder"]["popAssignAlias", #]&, baseVars["get"]];
		
		builder["currentFunctionModuleBuilder"]["finish", state, res];
		builder["currentFunctionModuleBuilder"]["setResult", res];
		
		builder["currentFunctionModuleBuilder"]["removeProperty", "capturedScopedMExprNames"];
		builder["currentFunctionModuleBuilder"]["setProperty", "entryQ" -> name === state["mainName"]];

		If[prevFm =!= Undefined,
			(*
			 ResolveFunctionCall also sets the localFunction for any global variables which 
			 turn into functions.
			*)
			builder["currentFunctionModuleBuilder"]["setProperty", "localFunction"->True];
			builder["setCurrentFunctionModuleBuilder", prevFm];
	        With[{
	        	nameStr = name
	        },
	        	res = state["createFreshVariable", CreateMExpr[nameStr]]
	        ];
			builder["createLambdaInstruction", res, CreateConstantValue[name], mexpr];
			,
			builder["currentFunctionModuleBuilder"]["setProperty", "localFunction"->False];
		];
		res
	]



lowerLocal[state_, mexpr_, opts_] :=
	Module[{
		  builder = state["builder"],
		  h = mexpr["head"],
		  funName, args, trgt, fun, inst
		},
		If[ h["length"] =!= 1 || h["part",1]["getHead"] =!= String,
			ThrowException[{"Unexpected form for Compile`LocalFunction", mexpr}]
		];
		funName = h["part",1];
	    args = state["lower", #, opts]& /@ mexpr["arguments"];
		fun = CreateConstantValue[funName];
		trgt = state["createFreshVariable", mexpr];
		inst = builder["createCallInstruction",
			trgt,
			fun,
			args,
			mexpr
		];
		trgt
	]

RegisterCallback["RegisterPrimitive", 
	Function[{st},
		RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Function], lower];
		RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Compile`LocalFunction[]], lowerLocal];
	]
]


End[]

EndPackage[]
