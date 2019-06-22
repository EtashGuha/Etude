
BeginPackage["Compile`TypeSystem`Bootstrap`Random`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]
Needs["CompileAST`Create`Construct`"]

"StaticAnalysisIgnore"[

setupTypes[st_] :=
	Module[{env = st["typeEnvironment"],
			inline = MetaData[<|"Inline" -> "Hint"|>],
			llvmLinkage = MetaData[<|"Linkage" -> "Runtime"|>]},

		env["declareFunction", Native`PrimitiveFunction["Runtime_UniformRandomMIntegers"], 
			llvmLinkage@TypeSpecifier[{"CArray"["MachineInteger"], "MachineInteger", "MachineInteger", "MachineInteger"} -> "Integer32"]];

		env["declareFunction", Native`PrimitiveFunction["Runtime_UniformRandomMReals"], 
			llvmLinkage@TypeSpecifier[{"CArray"["Real64"], "MachineInteger", "Real64", "Real64"} -> "Integer32"]];

		env["declareFunction", Native`PrimitiveFunction["Runtime_UniformRandomMComplexes"], 
			llvmLinkage@TypeSpecifier[{"VoidHandle", "MachineInteger", "Real64", "Real64", "Real64", "Real64"} -> "Integer32"]];

		env["declareFunction", Native`RandomRuntime, 
			inline@Typed[{"CArray"["MachineInteger"], "MachineInteger", "MachineInteger", "MachineInteger"} -> "Integer32"
			]@Function[{arr, len, min, max},
				Native`PrimitiveFunction["Runtime_UniformRandomMIntegers"][arr, len, min, max]
			]];

		env["declareFunction", Native`RandomRuntime, 
			inline@Typed[{"CArray"["Real64"], "MachineInteger", "Real64", "Real64"} -> "Integer32"
			]@Function[{arr, len, min, max},
				Native`PrimitiveFunction["Runtime_UniformRandomMReals"][arr, len, min, max]
			]];

		env["declareFunction", Native`RandomRuntime, 
			inline@Typed[{"CArray"["ComplexReal64"], "MachineInteger", "ComplexReal64", "ComplexReal64"} -> "Integer32"
			]@Function[{arr, len, min, max},
				Module[{voidArg},
					voidArg = Native`BitCast[ arr, TypeSpecifier[ "VoidHandle"]];
					Native`PrimitiveFunction["Runtime_UniformRandomMComplexes"][voidArg, len,Re[ min], Im[min], Re[ max], Im[max]]
				]				
			]];


		env["declareFunction", Native`Random, 
			Typed[TypeForAll[{"base"}, {Element["base", "BasePackable"]}, {"base", "base"} -> "base"]
			]@Function[{min, max},
				Module[{arr = Native`StackArray[1], err},
					err = Native`RandomRuntime[arr, 1, min, max];
					If[err =!= Typed[0, "Integer32"],
						Native`ThrowWolframException[Typed[Native`ErrorCode["RandomNumberError"], "Integer32"]]];
					arr[[0]]
				]
			]];

		env["declareFunction", Native`Random, 
			Typed[TypeForAll[ {"base", "rank"}, {Element["base", "BasePackable"]}, 
					{"base", "base", "rank", "PackedArray"["MachineInteger", 1]} -> 
							"PackedArray"["base", "rank"]]
			]@Function[{min, max, rank, dimsArray},
				Module[{len, base, paOut, data, err},
					base = Native`MTensorElementType[max];
					len = Length[dimsArray];
					data = Native`ArrayData[dimsArray];
					paOut = Native`CreatePackedArray[ base, len, data];
					data = Native`ArrayData[paOut];
					len = Native`MTensorNumberOfElements[paOut];
					err = Native`RandomRuntime[data, len, min, max];
					If[err =!= Typed[0, "Integer32"],
						Native`ThrowWolframException[Typed[Native`ErrorCode["RandomNumberError"], "Integer32"]]];
					paOut
				]
			]];

		env["declareFunction", Native`PrimitiveFunction["Runtime_SeedRandom"], 
			llvmLinkage@TypeSpecifier[{"MachineInteger", "MachineInteger"} -> "Integer32"]];

		env["declareFunction", SeedRandom, 
			Typed[ {} -> "Void"
			]@Function[{},
				Module[{err},
					err = Native`PrimitiveFunction["Runtime_SeedRandom"][0,0];
					If[err =!= Typed[0, "Integer32"],
						Native`ThrowWolframException[Typed[Native`ErrorCode["RandomNumberError"], "Integer32"]]];
				]
			]];

		env["declareFunction", SeedRandom, 
			Typed[ {"MachineInteger"} -> "Void"
			]@Function[{seed},
				Module[{err},
					err = Native`PrimitiveFunction["Runtime_SeedRandom"][seed,0];
					If[err =!= Typed[0, "Integer32"],
						Native`ThrowWolframException[Typed[Native`ErrorCode["RandomNumberError"], "Integer32"]]];
				]
			]];

		env["declareType", TypeConstructor["struct.RandomGenerator_struct"]];
		env["declareType", TypeAlias["RandomGenerator", "Handle"["struct.RandomGenerator_struct"]]];

		env["declareFunction", Native`PrimitiveFunction["Runtime_GetCurrentRandomGenerator"], 
			llvmLinkage@TypeSpecifier[{} -> "RandomGenerator"]];

		env["declareFunction", Native`PrimitiveFunction["Runtime_RandomGenerator_getBitFunction"], 
			llvmLinkage@TypeSpecifier[{"RandomGenerator"} -> {"MachineInteger", "RandomGenerator"} -> "MachineInteger"]];

		env["declareFunction", Native`CurrentRandomGenerator, 
			Typed[{} -> "RandomGenerator"
			]@Function[{},
				Native`PrimitiveFunction["Runtime_GetCurrentRandomGenerator"][]
			]];

		env["declareFunction", Native`RandomGeneratorBitFunction, 
			Typed[{"RandomGenerator"} -> {"MachineInteger", "RandomGenerator"} -> "MachineInteger"
			]@Function[{gen},
				Native`PrimitiveFunction["Runtime_RandomGenerator_getBitFunction"][gen]	
			]];


	]

] (* StaticAnalysisIgnore *)


setupMacros[st_] :=
	Module[ {env = st["macroEnvironment"]},
		RegisterMacro[env, RandomInteger,
			RandomInteger[] -> Compile`Internal`MacroEvaluate[ makeRandom[0,1, "MachineInteger"]],
			RandomInteger[max:Except[_List]] -> Compile`Internal`MacroEvaluate[ makeRandom[0,max, "MachineInteger"]],
			RandomInteger[{min_, max_}] -> Compile`Internal`MacroEvaluate[ makeRandom[min, max, "MachineInteger"]],
			RandomInteger[max:Except[_List], dims_] -> RandomInteger[{0,max}, dims],
			RandomInteger[{min_, max_}, dims_] -> Compile`Internal`MacroEvaluate[ makeRandom[min, max, dims, "MachineInteger"]]
		];
		RegisterMacro[env, RandomReal,
			RandomReal[] -> Compile`Internal`MacroEvaluate[ makeRandom[0.,1., "Real64"]],
			RandomReal[max:Except[_List]] -> Compile`Internal`MacroEvaluate[ makeRandom[0.,max, "Real64"]],
			RandomReal[{min_, max_}] -> Compile`Internal`MacroEvaluate[ makeRandom[min, max, "Real64"]],
			RandomReal[max:Except[_List], dims_] -> RandomReal[{0.,max}, dims],
			RandomReal[{min_, max_}, dims_] -> Compile`Internal`MacroEvaluate[ makeRandom[min, max, dims, "Real64"]]
		];
		RegisterMacro[env, RandomComplex,
			RandomComplex[] -> Compile`Internal`MacroEvaluate[ makeRandom[0. + 0. I,1. + 1. I, "ComplexReal64"]],
			RandomComplex[max:Except[_List]] -> Compile`Internal`MacroEvaluate[ makeRandom[0. + 0. I,max, "ComplexReal64"]],
			RandomComplex[{min_, max_}] -> Compile`Internal`MacroEvaluate[ makeRandom[min, max, "ComplexReal64"]],
			RandomComplex[max:Except[_List], dims_] -> RandomComplex[{0. + 0. I, max}, dims],
			RandomComplex[{min_, max_}, dims_] -> Compile`Internal`MacroEvaluate[ makeRandom[min, max, dims, "ComplexReal64"]]
		];

	]


makeCast[ arg_, ty_] :=
	Module[{tmp},
		tmp = CreateMExpr[ TypeSpecifier, {ty}];
		With[{a1 = {arg, tmp}},
			CreateMExpr[ Compile`Cast, a1]]
	]

makeRandom[minIn_, maxIn_, ty_] :=
	Module[ {
			min = makeCast[minIn,ty],
			max = makeCast[maxIn,ty],
			args
		},
		args = {min, max};
		With[{a1 = args},
			CreateMExpr[ Native`Random, a1]]
	]

makeRandom[minIn_, maxIn_, dimsIn_, ty_] :=
	Module[ {
			min = makeCast[minIn,ty],
			max = makeCast[maxIn,ty],		
			dims = 
				If[dimsIn["head"]["symbolQ"] && dimsIn["head"]["fullName"] === "System`List", 
							dimsIn, CreateMExpr[ List, {dimsIn}]],
			len, args
		},
		len = dims["length"];
		(*
		  If len is zero return the scalar.
		*)
		args = If[ len === 0, {min, max}, {min, max, Typed[len, len], dims}];
		With[{a1 = args},
			CreateMExpr[ Native`Random, a1]]
	]


RegisterCallback["SetupMacros", setupMacros]
RegisterCallback["SetupTypeSystem", setupTypes]


End[]

EndPackage[]
