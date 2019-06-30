
BeginPackage["Compile`TypeSystem`Bootstrap`Unchecked`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]


	
"StaticAnalysisIgnore"[

setupTypes[st_] :=
With[{
	env = st["typeEnvironment"],
	inline = MetaData[<|"Inline" -> "Hint"|>],
	llvmLinkage = MetaData[<|"Linkage" -> "LLVMCompileTools"|>]
},

	addLLVMFunction[origName_, targetName_, ty_] :=
		(
			env["declareFunction", origName,  MetaData[<|"Redirect" -> Native`PrimitiveFunction[targetName]|>]@ty];
			env["declareFunction", Native`PrimitiveFunction[targetName], llvmLinkage@ty];
		);
		

	Do[
		With[{
			op = op0,	
			nameWork = "UncheckedWork" <> ToString[op0],
			name = "Unchecked" <> ToString[op0]
		},
			env["declareFunction", Native`Unchecked[op],
				inline@
				Typed[TypeForAll[ {"a","b"},
					{Element["a", "Number"], Element["b", "Number"]}, 
					{"a", "b"} -> "TypeJoin"["a", "b"]
				]]@
				Function[{x, y},
					Module[ {
						arg1 = Compile`Cast[x, Compile`TypeJoin[x, y]],
						arg2 = Compile`Cast[y, Compile`TypeJoin[x, y]]},
						Native`PrimitiveFunction[nameWork][arg1, arg2]
				]]
			];
						
			env["declareFunction", Native`PrimitiveFunction[nameWork],
				inline@
				Typed[TypeForAll[{"a"},
					{Element["a", "Real"]},
					{"a", "a"} -> "a"
				]]@
				Function[{x,y},
					Native`PrimitiveFunction[name][x, y]
				]
			];
			
			env["declareFunction", Native`PrimitiveFunction[nameWork],
				inline@
				Typed[TypeForAll[ {"a"},
					{"Complex"["a"], "Complex"["a"]} -> "Complex"["a"]
				]]@
				Function[{x,y},
					op[x, y]
				]
			];
			
			addLLVMFunction[
				Native`Unchecked[op],
				name,
				TypeSpecifier[TypeForAll[{"a"},
					{Element["a", "Real"]},
					{"a", "a"} -> "a"
				]]
			];

		];
		,
		{op0, {Plus, Times, Subtract, Divide}}
	];
	
	env["declareFunction", Native`Unchecked[Minus],
		inline@
		Typed[TypeForAll[ {"a"},
			{Element["a", "Number"]}, 
			{"a"} -> "a"
		]]@
		Function[{x},
			Native`PrimitiveFunction["UncheckedSubtract"][Compile`Cast[0, Compile`TypeOf[x]], x]
		]
	];
	
	env["declareFunction", Native`Unchecked[Floor],
		inline@
		Typed[TypeForAll[ {"a"},
			{Element["a", "Integral"]}, 
			{"a"} -> "a"
		]]@
		Function[{x},
			x
		]
	];
	
	env["declareFunction", Native`Unchecked[Ceiling],
		inline@
		Typed[TypeForAll[ {"a"},
			{Element["a", "Integral"]}, 
			{"a"} -> "a"
		]]@
		Function[{x},
			x
		]
	];
	
	Do[
		With[{
			op = op0,
			name = "Unchecked" <> ToString[op0]
		},	
			env["declareFunction", Native`Unchecked[op],
				inline@
				Typed[TypeForAll[ {"a"},
					{Element["a", "RealFloatingPoint"]}, 
					{"a"} -> "a"
				]]@
				Function[{x},
					Native`PrimitiveFunction[name][x]
				]
			];
			
			addLLVMFunction[
				Native`Unchecked[op],
				name,
				TypeSpecifier[TypeForAll[{"a"},
					{Element["a", "RealFloatingPoint"]},
					{"a"} -> "a"
				]]
			];

			If[op === Abs,
				env["declareFunction", Native`Unchecked[op],
					inline@
					Typed[TypeForAll[ {"a"},
						{Element["a", "RealFloatingPoint"]}, 
						{"Complex"["a"]} -> "a"
					]]@
					Function[{x},
						op[x]
					]
				]
				,	
				env["declareFunction", Native`Unchecked[op],
					inline@
					Typed[TypeForAll[ {"a"},
						{Element["a", "RealFloatingPoint"]}, 
						{"Complex"["a"]} -> "Complex"["a"]
					]]@
					Function[{x},
						op[x]
					]
				]
			];
		];
		,
		{op0, {Sqrt, Exp, Cos, Sin, Log, Log2, Log10, Abs}}
	];
	
	
	Do[
		With[{
			op = op0,
			name = "Unchecked" <> ToString[op0]
		},		
			env["declareFunction", Native`Unchecked[op],
				inline@
				Typed[TypeForAll[ {"a"},
					{Element["a", "FloatingPoint"]}, 
					{"a"} -> "MachineInteger"
				]]@
				Function[{x},
					Native`PrimitiveFunction[name][x]
				]
			];
			
			addLLVMFunction[
				Native`Unchecked[op],
				name,
				TypeSpecifier[TypeForAll[{"a"},
					{Element["a", "FloatingPoint"]},
					{"a"} -> "MachineInteger"
				]]
			]
		];
		,
		{op0, {Floor, Ceiling}}
	];
	
	
	env["declareFunction", Native`Unchecked[Abs],
		inline@
		Typed[TypeForAll[ {"a"},
			{Element["a", "Integral"]}, 
			{"a"} -> "a"
		]]@
		Function[{x},
			Native`UncheckedBlock@
			If[x < 0,
				-x,
				x
			]
		]
	];
	
	env["declareFunction", Native`Unchecked[Tan],
		inline@
		Typed[TypeForAll[ {"a"},
			{Element["a", "FloatingPoint"]}, 
			{"a"} -> "a"
		]]@
		Function[{x},
			Native`UncheckedBlock[Sin[x] / Cos[x]]
		]
	];

(*
  Power
*)
	env["declareFunction", Native`Unchecked[Power],
		inline@
		Typed[TypeForAll[ {"a","b"},
			{Element["a", "Integral"], Element["b", "Integral"]}, 
			{"a", "b"} -> "TypeJoin"["a", "b"]
		]]@
		Function[{x, y},
			Module[ {
				arg1 = Compile`Cast[x, Compile`TypeJoin[x, y]],
				arg2 = Compile`Cast[y, Compile`TypeJoin[x, y]]},
				Power[arg1, arg2]
		]]
	];

	env["declareFunction", Native`Unchecked[Power],
		inline@
		Typed[TypeForAll[ {"a"},
			{Element["a", "RealFloatingPoint"]}, 
			{"a", "Integer32"} -> "a"
		]]@
		Function[{x, y},
			Native`PrimitiveFunction["UncheckedPowerFI"][x, y]]
	];

	env["declareFunction", Native`Unchecked[Power],
		inline@
		Typed[TypeForAll[ {"a","b"},
			{Element["a", "Real"], Element["b", "RealFloatingPoint"]}, 
			{"a", "b"} -> "TypeJoin"["a", "b"]
		]]@
		Function[{x, y},
			Module[ {
				arg1 = Compile`Cast[x, Compile`TypeJoin[x, y]],
				arg2 = Compile`Cast[y, Compile`TypeJoin[x, y]]},
				Native`PrimitiveFunction["UncheckedPower"][arg1, arg2]
		]]
	];
	
	env["declareFunction", Native`PrimitiveFunction["UncheckedPower"], 
			llvmLinkage@TypeSpecifier[TypeForAll[{"a"},
				{Element["a", "RealFloatingPoint"]},
				{"a", "a"} -> "a"
			]]];

	env["declareFunction", Native`PrimitiveFunction["UncheckedPowerFI"], 
			llvmLinkage@TypeSpecifier[TypeForAll[{"a"},
				{Element["a", "RealFloatingPoint"]},
				{"a", "Integer32"} -> "a"
			]]];

	env["declareFunction", Native`Unchecked[Power],
		inline@
		Typed[TypeForAll[ {"a","b"},
			{Element["b", "Real"]},
			{"Complex"["a"], "b"} -> "TypeJoin"["Complex"["a"],"Complex"["b"]]
		]]@
		Function[{x, y},
			Module[ {
				arg1 = Compile`Cast[x, Compile`TypeJoin[x, y]],
				arg2 = Compile`Cast[y, Compile`TypeJoin[x, y]]},
				Power[arg1, arg2]
		]]
	];

	env["declareFunction", Native`Unchecked[Power],
		inline@
		Typed[TypeForAll[ {"a","b"},
			{Element["a", "Real"]},
			{"a", "Complex"["b"]} -> "TypeJoin"["Complex"["a"],"Complex"["b"]]
		]]@
		Function[{x, y},
			Module[ {
				arg1 = Compile`Cast[x, Compile`TypeJoin[x, y]],
				arg2 = Compile`Cast[y, Compile`TypeJoin[x, y]]},
				Power[arg1, arg2]
		]]
	];

	env["declareFunction", Native`Unchecked[Power],
		inline@
		Typed[TypeForAll[ {"a","b"},
			{"Complex"["a"], "Complex"["b"]} -> "TypeJoin"["Complex"["a"],"Complex"["b"]]
		]]@
		Function[{x, y},
			Module[ {
				arg1 = Compile`Cast[x, Compile`TypeJoin[x, y]],
				arg2 = Compile`Cast[y, Compile`TypeJoin[x, y]]},
				Power[arg1, arg2]
		]]
	];
		

	Do[
		With[{
			op = op0,
			name = "Unchecked" <> ToString[op0]
		},
			env["declareFunction", Native`Unchecked[op],
				inline@
				Typed[TypeForAll[ {"a","b"},
					{Element["a", "FloatingPoint"], Element["b", "FloatingPoint"]}, 
					{"a", "b"} -> "TypeJoin"["a", "b"]
				]]@
				Function[{x, y},
					Module[ {
						arg1 = Compile`Cast[x, Compile`TypeJoin[x, y]],
						arg2 = Compile`Cast[y, Compile`TypeJoin[x, y]]},
						Native`PrimitiveFunction[name][arg1, arg2]
				]]
			];
					
			addLLVMFunction[
				Native`Unchecked[op],
				name,
				TypeSpecifier[TypeForAll[{"a"},
					{Element["a", "FloatingPoint"]},
					{"a", "a"} -> "a"
				]]
			]
		];
		,
		{op0, {Min, Max, "FusedMultiplyAdd"}}
	];


	Do[
		With[{
			op = op0,
			name = ToString[op0]
		},
			env["declareFunction", Native`Unchecked[op],
				inline@
				Typed[TypeForAll[ {"a","b"},
					{Element["a", "Integral"], Element["b", "Integral"]}, 
					{"a", "b"} -> "TypeJoin"["a", "b"]
				]]@
				Function[{x, y},
					Module[ {
						arg1 = Compile`Cast[x, Compile`TypeJoin[x, y]],
						arg2 = Compile`Cast[y, Compile`TypeJoin[x, y]]},
						Native`PrimitiveFunction[name][arg1, arg2]
				]]
			];
		];
		,
		{op0, {BitAnd, BitOr, BitXor, BitShiftLeft, BitShiftRight}}
	];
	
	
	env["declareFunction", Native`Unchecked[BitLength],
		inline@
		Typed[TypeForAll[ {"a"},
			{Element["a", "Integral"]}, 
			{"a"} -> "MachineInteger"
		]]@
		Function[{x},
			Native`PrimitiveFunction["UncheckedBitLength"][x]
		]
	];
	env["declareFunction", Native`PrimitiveFunction["UncheckedBitLength"],
		llvmLinkage@
		TypeSpecifier[TypeForAll[ {"a"},
			{Element["a", "Integral"]}, 
			{"a"} -> "MachineInteger"
		]]
	];
	
]
] (* StaticAnalysisIgnore *)


RegisterCallback["SetupTypeSystem", setupTypes]


End[]

EndPackage[]
