
BeginPackage["Compile`TypeSystem`Bootstrap`TakeDrop`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]
Needs["CompileAST`Create`Construct`"]

"StaticAnalysisIgnore"[

setupTypes[st_] :=
	Module[{env = st["typeEnvironment"],
			inline = MetaData[<|"Inline" -> "Hint"|>]},

		env["declareFunction", Native`PrimitiveFunction["Runtime_MTensor_TakeDrop"], 
						MetaData[
							<|"Linkage" -> "Runtime"|>
							]@TypeSpecifier[ 
								{"Handle"["MTensor"], "MTensor", "MachineInteger", "CArray"["VoidHandle"], "MBool"} -> "ErrorCode"]];


		env["declareFunction", Native`TakeDropAdd, 
			Typed[
				{"CArray"["VoidHandle"], "MachineInteger", "MachineInteger"} -> "CArray"["VoidHandle"]
			]@Function[{specsIn, index, arg}, 
				Module[{specs = specsIn, val},
					val = Native`BitCast[ 0, "VoidHandle"];
					specs[[index]] = val;
					val = Native`BitCast[ arg, "VoidHandle"];
					specs[[index+1]] = val;
					specs
				]
			]
		];

		env["declareFunction", Native`TakeDropAdd, 
			Typed[
				{"CArray"["VoidHandle"], "MachineInteger", "PackedArray"["MachineInteger", 1]} -> "CArray"["VoidHandle"]
			]@Function[{specsIn, index, arg}, 
				Module[{specs = specsIn, val},
					val = Native`BitCast[ 1, "VoidHandle"];
					specs[[index]] = val;
					val = Native`BitCast[ arg, "VoidHandle"];
					specs[[index+1]] = val;
					specs
				]
			]
		];
		
		env["declareFunction", Native`TakeDropFinalize, 
			Typed[
				TypeForAll[ {"a", "b"},  {"PackedArray"["a", "b"], "CArray"["VoidHandle"], "MachineInteger", "MBool"} -> "PackedArray"["a", "b"]]
			]@Function[{array, specs, len, takeQ}, 
				Module[{hand = Native`Handle[], hand1, err},
					hand1 = Native`BitCast[ hand, "Handle"["MTensor"]];
					Native`Store[hand1, Compile`NullReference];
					err = Native`PrimitiveFunction["Runtime_MTensor_TakeDrop"][hand1, Native`BitCast[array, "MTensor"], len, specs, takeQ];
					If[err =!= Typed[0, "Integer32"],
						Native`ThrowWolframException[Typed[Native`ErrorCode["TakeError"], "Integer32"]]];
					Native`Load[hand]
				]
			]
		];

	]

] (* StaticAnalysisIgnore *)


setupMacros[st_] :=
	Module[ {env = st["macroEnvironment"]},

		Scan[
			With[ {fun = First[#], bool = Last[#]},
				RegisterMacro[env, fun,
					fun[a_, inds__] -> 
						 Native`TakeDropFinalize[
						 	a,
							Native`TakeDropAdd[ 
								Native`TakeDropInitialize[ {inds}],
								0,
								inds
							],
							Compile`Internal`MacroEvaluate[ getLength[{inds}]],
							bool
						]
				]]&, {{Take, Native`MBoolTrue}, {Drop, Native`MBoolFalse}}];
	
		RegisterMacro[env, Native`TakeDropAdd,
			Native`TakeDropAdd[array_, indNum_, ind1_, indr__] -> 
					Native`TakeDropAdd[ Native`TakeDropAdd[ array, indNum, ind1], indNum+2, indr]
		];

		RegisterMacro[env, Native`TakeDropInitialize,
			Native`TakeDropInitialize[args_] -> 
					Compile`Internal`MacroEvaluate[ initializeTakeDrop[args]]
		];

	];


initializeTakeDrop[ args_] :=
		With[{arg = CreateMExprLiteral[ 2 args["length"]]},
			CreateMExpr[ Native`StackArray, {arg}]
		]

getLength[ args_] :=
	CreateMExprLiteral[ args["length"]]



RegisterCallback["SetupMacros", setupMacros]
RegisterCallback["SetupTypeSystem", setupTypes]


End[]

EndPackage[]
