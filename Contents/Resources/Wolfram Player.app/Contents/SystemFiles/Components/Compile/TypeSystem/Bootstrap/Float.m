
BeginPackage["Compile`TypeSystem`Bootstrap`Float`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]


	
"StaticAnalysisIgnore"[

setupTypes[st_] :=
With[{
	env = st["typeEnvironment"],
	inline = MetaData[<|"Inline" -> "Hint"|>]
},

    env["declareFunction", Native`NaN,
        inline@
        Typed[{} -> "Real16"]@
        Function[{},
            Native`BitCast[Typed[16^^7C01, "UnsignedInteger32"], "Real16"]
        ]
    ];
    env["declareFunction", Native`NaN,
        inline@
        Typed[{} -> "Real32"]@
        Function[{},
            Native`BitCast[Typed[16^^7f800001, "UnsignedInteger32"], "Real32"]
        ]
    ];
    env["declareFunction", Native`NaN,
        inline@
        Typed[{} -> "Real64"]@
        Function[{},
            Native`BitCast[Typed[16^^7FF0000000000001, "UnsignedInteger64"], "Real64"]
        ]
    ];
    
    
    env["declareFunction", Native`Infinity,
        inline@
        Typed[{} -> "Real16"]@
        Function[{},
            Native`BitCast[Typed[16^^7C00, "UnsignedInteger32"], "Real16"]
        ]
    ];
    env["declareFunction", Native`Infinity,
        inline@
        Typed[{} -> "Real32"]@
        Function[{},
            Native`BitCast[Typed[16^^7f800000, "UnsignedInteger32"], "Real32"]
        ]
    ];
    env["declareFunction", Native`Infinity,
        inline@
        Typed[{} -> "Real64"]@
        Function[{},
            Native`BitCast[Typed[16^^7FF0000000000000, "UnsignedInteger64"], "Real64"]
        ]
    ];
    
	env["declareFunction", Native`NaNQ,
		inline@
		Typed[TypeForAll[{"a"}, {Element["a", "FloatingPoint"]}, {"a"} -> "Boolean"]]@
		Function[{x},
			(* IEEE 754 says that only NaNs satisfy f != f *)
			Native`UncheckedBlock[x =!= x]
		]
	];
	
	env["declareFunction", Native`FiniteQ,
		inline@
		Typed[TypeForAll[{"a"}, {Element["a", "FloatingPoint"]}, {"a"} -> "Boolean"]]@
		Function[{x},
			Native`UncheckedBlock[
				!Native`NaNQ[x - x]
			]
		]
	];
	
	env["declareFunction", Native`InfiniteQ,
		inline@
		Typed[TypeForAll[{"a"}, {Element["a", "FloatingPoint"]}, {"a"} -> "Boolean"]]@
		Function[{f},
			Native`UncheckedBlock[
				!Native`NaNQ[f] && !Native`FiniteQ[f]
			]
		]
	];
]
] (* StaticAnalysisIgnore *)


RegisterCallback["SetupTypeSystem", setupTypes]


End[]

EndPackage[]
