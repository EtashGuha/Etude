
BeginPackage["Compile`TypeSystem`Bootstrap`ReinterpretCast`"]

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
    
              
        env["declareFunction", Native`PrimitiveFunction["ReinterpretCast"], 
            llvmLinkage@
            TypeSpecifier[TypeForAll[ {"a", "b"}, {"a", "b"} -> "b"]]
        ];
        
        env["declareFunction", Native`ReinterpretCast,
	        inline@
	        Typed[
	        	TypeForAll[
	        		{"a", "b"},
	        		{Element["a", "Real"], Element["b", "Real"]},
	        		{"a", "b"} -> "b"
	        	]
	        ]@
	        Function[{a, b}, 
                Native`PrimitiveFunction["ReinterpretCast"][a, b]
            ]
        ];

        env["declareFunction", Native`ReinterpretCast,
	        inline@
	        Typed[
	        	TypeForAll[
	        		{"a", "b"},
	        		{Element["a", "Real"], Element["b", "Real"]},
	        		{"a", "Complex"["b"]} -> "Complex"["b"]
	        	]
	        ]@
	        Function[{a, b}, 
                Native`PrimitiveFunction["ReinterpretCast"][a, b]
            ]
        ];

       env["declareFunction", Native`ReinterpretCast,
	        inline@
	        Typed[
	        	TypeForAll[
	        		{"a", "b"},
	        		{Element["a", "Real"], Element["b", "Real"]},
	        		{"Complex"["a"], "Complex"["b"]} -> "Complex"["b"]
	        	]
	        ]@
	        Function[{a, b}, 
                Native`PrimitiveFunction["ReinterpretCast"][a, b]
            ]
        ];

        (* Casting to the same type is a no-op. *)
        env["declareFunction", Native`ReinterpretCast,
            inline@
            Typed[
                TypeForAll[
                    {"a"},
                    {Element["a", "Number"]},
                    {"a", "a"} -> "a"
                ]
            ]@
            Function[{a, b}, 
                a
            ]
        ];
        
    ]
    
] (* StaticAnalysisIgnore *)


RegisterCallback["SetupTypeSystem", setupTypes]


End[]

EndPackage[]
