
BeginPackage["Compile`TypeSystem`Bootstrap`Traits`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]


	
"StaticAnalysisIgnore"[

setupTypes[st_] :=
With[{
	env = st["typeEnvironment"],
	inline = MetaData[<|"Inline" -> "Hint"|>],
	$signedIntSizes = <|
		"Integer8" -> 8,
		"Integer16" -> 16,
		"Integer32" -> 32,
		"Integer64" -> 64
	|>,
	$unsignedIntSizes = <|
		"UnsignedInteger8" -> 8,
		"UnsignedInteger16" -> 16,
		"UnsignedInteger32" -> 32,
		"UnsignedInteger64" -> 64
	|>
},

	env["declareFunction", Native`Traits`MinValue,
		inline@
		Typed[TypeForAll[{"a"}, {Element["a", "UnsignedInteger"]}, {"a"} -> "a"]]@
		Function[{x},
			Typed[0, Compile`TypeOf[x]]
		]
	];
	Do[
		With[{
			intType = intType0,
			maxValue = 2^($signedIntSizes[intType0] - 1),
			minValue = -(2^($signedIntSizes[intType0]-1)-1)
		},
			env["declareFunction", Native`Traits`MinValue,
				inline@
				Typed[{intType} -> intType]@
				Function[{x},
					Typed[minValue, intType]
				]
			];
			
			env["declareFunction", Native`Traits`MaxValue,
				inline@
				Typed[{intType} -> intType]@
				Function[{x},
					Typed[maxValue, intType]
				]
			]
		],
		{intType0, Keys[$signedIntSizes]}
	];
	
	Do[
		With[{
			intType = intType0,
			maxValue = 2^$signedIntSizes[intType0]
		},
			env["declareFunction", Native`Traits`MaxValue,
				inline@
				Typed[{intType} -> intType]@
				Function[{x},
					Typed[maxValue, intType]
				]
			]
		],
		{intType0, Keys[$unsignedIntSizes]}
	];
	

	env["declareFunction", Native`Traits`UnsignedIntegerQ,
		inline@
		Typed[TypeForAll[{"a"}, {Element["a", "UnsignedInteger"]}, {"a"} -> "Boolean"]]@
		Function[{x},
			True
		]
	];
	
	env["declareFunction", Native`Traits`UnsignedIntegerQ,
		inline@
		Typed[TypeForAll[{"a"}, {"a"} -> "Boolean"]]@
		Function[{x},
			False
		]
	];
	
	env["declareFunction", Native`Traits`SignedIntegerQ,
		inline@
		Typed[TypeForAll[{"a"}, {Element["a", "SignedInteger"]}, {"a"} -> "Boolean"]]@
		Function[{x},
			True
		]
	];
	
	env["declareFunction", Native`Traits`SignedIntegerQ,
		inline@
		Typed[TypeForAll[{"a"}, {"a"} -> "Boolean"]]@
		Function[{x},
			False
		]
	];
	
	env["declareFunction", Native`Traits`IntegerQ,
		inline@
		Typed[TypeForAll[{"a"}, {Element["a", "Integer"]}, {"a"} -> "Boolean"]]@
		Function[{x},
			True
		]
	];
	
	env["declareFunction", Native`Traits`IntegerQ,
		inline@
		Typed[TypeForAll[{"a"}, {"a"} -> "Boolean"]]@
		Function[{x},
			False
		]
	];
	
	env["declareFunction", Native`Traits`RealQ,
		inline@
		Typed[TypeForAll[{"a"}, {Element["a", "FloatingPoint"]}, {"a"} -> "Boolean"]]@
		Function[{x},
			True
		]
	];
	
	env["declareFunction", Native`Traits`RealQ,
		inline@
		Typed[TypeForAll[{"a"}, {"a"} -> "Boolean"]]@
		Function[{x},
			False
		]
	];

]
] (* StaticAnalysisIgnore *)


RegisterCallback["SetupTypeSystem", setupTypes]


End[]

EndPackage[]
