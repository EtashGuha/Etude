BeginPackage["Compile`TypeSystem`Bootstrap`Utilities`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]


"StaticAnalysisIgnore"[

setupTypes[st_] :=
	With[ {
	    env = st["typeEnvironment"],
        alwaysInline = MetaData[<|"Inline" -> "Always"|>],
	    inline = MetaData[<|"Inline" -> "Hint"|>],
	    llvmCompileToolsLinkage = MetaData[<|"Linkage" -> "LLVMCompileTools"|>]
    },
		
		env["declareFunction", Native`StackAllocate, TypeSpecifier[TypeForAll[ {"a", "b"}, {"MachineInteger"} -> "a"["b"]]]];
		env["declareFunction", Native`CreateHandle, TypeSpecifier[TypeForAll[ {"a"}, {"MachineInteger"} -> "Handle"["a"]]]];
		
		env["declareFunction", Native`StackAllocateObject, TypeSpecifier[TypeForAll[ {"a", "b"}, {"MachineInteger"} -> "a"["b"]]]];
		env["declareFunction", Native`StackAllocateObject, TypeSpecifier[TypeForAll[ {"a", "b", "c"}, {"MachineInteger"} -> "a"["b", "c"]]]];
		env["declareFunction", Native`StackAllocateObject, TypeSpecifier[TypeForAll[ {"a", "b", "c", "d"}, {"MachineInteger"} -> "a"["b", "c", "d"]]]];
		env["declareFunction", Native`StackAllocateObject, TypeSpecifier[TypeForAll[ {"a", "b", "c", "d", "e"}, {"MachineInteger"} -> "a"["b", "c", "d", "e"]]]];
		env["declareFunction", Native`StackAllocateObject, TypeSpecifier[TypeForAll[ {"a", "b", "c", "d", "e", "f"}, {"MachineInteger"} -> "a"["b", "c", "d", "e", "f"]]]];
		env["declareFunction", Native`StackAllocateObject, TypeSpecifier[TypeForAll[ {"a", "b", "c", "d", "e", "f", "g"}, {"MachineInteger"} -> "a"["b", "c", "d", "e", "f", "g"]]]];
		
		
		env["declareFunction", Native`Load, TypeSpecifier[TypeForAll[ {"a", "b"}, {"a"["b"]} -> "b"]]];
		env["declareFunction", Native`Store, TypeSpecifier[TypeForAll[ {"a", "b"}, {"a"["b"], "b"} -> "Void"]]];
		
		env["declareType", AbstractType["Indexable", {"a"},
            {
                Native`GetElement -> MetaData[<|"Builtin" -> True|>]@TypeSpecifier[TypeForAll[{"b"}, {"a"["b"], "MachineInteger"} -> "b"]],
                Native`SetElement -> MetaData[<|"Builtin" -> True|>]@TypeSpecifier[TypeForAll[{"b"}, {"a"["b"], "MachineInteger", "b"} -> "Void"]]
            }
        ]];

		env["declareType", AbstractType["Iteratorable", {}]];
		env["declareType", TypeInstance["Iteratorable", "MachineInteger"]];
		env["declareType", TypeInstance["Iteratorable", "Real64"]];


		env["declareType", TypeConstructor["MIteratorBase", {"*"} -> "*"]];

		env["declareType", TypeConstructor["MIterator", {"*"} -> "*"]];
	
		env["declareType", TypeAlias["MIterator"["a"], 
								       "Handle"["MIteratorBase"["a"]], 
  										"VariableAlias" -> True]];
		
		env["declareType", 
			MetaData[<|"Fields" -> <|"f1" -> 1, "f2" -> 2|>|>
      			]@TypeConstructor["MIteratorWork", {"*", "*", "*", "*"} -> "*"]];
		
		env["declareType", TypeAlias["MIteratorBase"["a"], 
								"MIteratorWork"[ "a", "a", "a", "MachineInteger"], 
  										"VariableAlias" -> True]];
		
		env["declareFunction", Native`CreateIterator,
			alwaysInline@
			Typed[
				TypeForAll[ {"a", "b","c"}, 
						{Element["a", "Iteratorable"], Element["b", "Iteratorable"], Element["c", "Iteratorable"]}, 
							{"a", "b", "c"} -> "MIterator"[ "TypeJoin"["a", "c"]]]
			]@Function[{startIn, endIn, incrementIn},
				Module[{obj, num, start, end, increment},
					start = Compile`Cast[startIn, Compile`TypeJoin[startIn, incrementIn]];
					increment = Compile`Cast[incrementIn, Compile`TypeJoin[startIn, incrementIn]];
					end = Native`IteratorConvertEnd[ endIn, start];
					obj = Native`StackAllocateObject["MIteratorBase"];
					num = Native`UncheckedBlock[Native`IteratorCount[start, end, increment]];
					obj[[Native`Field[0]]] = start;
					obj[[Native`Field[1]]] = end;
					obj[[Native`Field[2]]] = increment;
					obj[[Native`Field[3]]] = num;
					obj
				]
			]];

			env["declareFunction", Native`IteratorConvertEnd,
							inline@Typed[
								TypeForAll[{"a"}, {"a", "MachineInteger"} -> "MachineInteger"]
							]@Function[ {arg1, arg2}, Floor[arg1]]];
							
			env["declareFunction", Native`IteratorConvertEnd,
							inline@Typed[
								TypeForAll[{"a"}, {"a", "Real64"} -> "Real64"]
							]@Function[ {arg1, arg2}, Compile`Cast[arg1, TypeSpecifier["Real64"]]]];
							
            env["declareFunction", Native`IteratorCount,
                (*inline@*)
                Typed[TypeSpecifier[{"MachineInteger", "MachineInteger", "MachineInteger"} -> "MachineInteger"]]@
                Function[{low, high, incr},
                    Module[{diff = high-low, reps},
                        If[incr == 0,
                            Native`ThrowWolframException[Typed[Native`ErrorCode["DimensionError"], "Integer32"]]
                        ];
                        reps = Floor[Compile`Cast[diff, "Real64"] / Compile`Cast[incr, "Real64"]];
                        If[reps == 0,
                            If[incr > 0,
                                If[diff < 0,
                                    reps = -1;
                                ],
                                If[diff > 0,
                                    reps = -1;
                                ]
                            ];
                        ];
                        reps+1
                    ]
                ]
            ];
            
            
			env["declareFunction", Native`IteratorCount,
				inline@Typed[
					{"Real64", "Real64", "Real64"} -> "MachineInteger"
				]@Function[ {arg1, arg2, arg3}, 
					Native`PrimitiveFunction["Runtime_IteratorCount_R_R_R_I"][arg1,arg2,arg3]+1
				]];
	
			env["declareFunction", Native`PrimitiveFunction["Runtime_IteratorCount_I_I_I_I"], 
				MetaData[
					<|"Linkage" -> "Runtime"|>
				]@TypeSpecifier[{"MachineInteger", "MachineInteger", "MachineInteger"} -> "MachineInteger"]];

			env["declareFunction", Native`PrimitiveFunction["Runtime_IteratorCount_R_R_R_I"], 
				MetaData[
					<|"Linkage" -> "Runtime"|>
				]@TypeSpecifier[{"Real64", "Real64", "Real64"} -> "MachineInteger"]];

						
			env["declareFunction", Native`IteratorStart,
				inline@Typed[
					TypeForAll[{"a"}, {"MIterator"["a"]} -> "a"]
				]@Function[ {iter}, iter[[Native`Field[0]]]]];

			env["declareFunction", Native`IteratorEnd,
				inline@Typed[
					TypeForAll[{"a"}, {"MIterator"["a"]} -> "a"]
				]@Function[ {iter}, iter[[Native`Field[1]]]]];

			env["declareFunction", Native`IteratorIncrement,
				inline@Typed[
					TypeForAll[{"a"}, {"MIterator"["a"]} -> "a"]
				]@Function[ {iter}, iter[[Native`Field[2]]]]];

			(*
			  TODO,  probably add some type of floor
			*)
			env["declareFunction", Native`IteratorLength,
				inline@Typed[
					TypeForAll[{"a"}, {"MIterator"["a"]} -> "MachineInteger"]
				]@Function[ {iter}, iter[[Native`Field[3]]]]];

                
			env["declareFunction", Native`IteratorValue,
				inline@Typed[
					TypeForAll[{"a"}, {"MIterator"["a"], "MachineInteger"} -> "a"]
				]@Function[ {iter, pos},
						Module[ {start, inc},
							start = iter[[Native`Field[0]]];
							inc = iter[[Native`Field[2]]];
							Native`UncheckedBlock[start+inc*pos]
						]
				]];
				
            env["declareFunction", Native`IteratorValue,
                inline@Typed[
                     {"MachineInteger", "MachineInteger", "MachineInteger"} -> "MachineInteger"
                ]@Function[ {start, inc, pos},
                     Native`UncheckedBlock[start+inc*pos]
                ]];
                
            env["declareFunction", Native`IteratorValue,
                inline@Typed[
                     {"Real64", "Real64", "MachineInteger"} -> "Real64"
                ]@Function[ {start, inc, pos},
                     Native`UncheckedBlock[start+inc*pos]
                ]];

			env["declareFunction", Native`MutabilityClone,
				inline@
				Typed[
						TypeForAll[ {"a"}, {"CArray"["a"]} -> "CArray"["a"]]
					]@Function[{arr},
						arr
					]]; 


			env["declareFunction", Native`CArrayToString,
				Typed[
					TypeForAll[{"a"}, {"CArray"["a"], "MachineInteger"} -> "String"]
				]@Function[ {array, len},
					Module[ {str = "{"},
						str = str <> ToString[ array[[0]]];
						Do[
							str = str <> ",";
							str = str <> ToString[array[[i]]];
							,
							{i,1,len-1}];
						str <> "}"
					]
				]];
			
			
			
			(*
			 Copy len elements from src to dest.   
			 Can use unchecked because if this is going to overflow we are running out of memory.
			*)
			env["declareFunction", Native`CopyTo,
				inline@Typed[
					TypeForAll[{"a"}, {"CArray"["a"], "CArray"["a"], "MachineInteger"} -> "Void"]
				]@Function[ {src, dest, len},
					Module[{destC, srcC, nbytes},
						Native`UncheckedBlock[
							nbytes = len*Native`SizeOf[Native`GetElement[src,0]];
							destC = Native`BitCast[dest, "VoidHandle"];
							srcC = Native`BitCast[src, "VoidHandle"];
							Native`MemCpy[destC, srcC, nbytes]];
					]
				]];


			(*
			 Copy len elements from src to dest,  starting at start in dest
			*)
			env["declareFunction", Native`CopyTo,
				inline@Typed[
					TypeForAll[{"a"}, {"CArray"["a"], "CArray"["a"], "MachineInteger", "MachineInteger"} -> "Void"]
				]@Function[ {src, dest, start, len},
					Module[ {destFix = Native`AddressShift[dest, start]},
						Native`CopyTo[src, destFix, len]
					]
				]];

			(*
			 Copy len elements, from src to destSrc,  starting at startSrc in src and startDest in dest
			*)
			env["declareFunction", Native`CopyTo,
				inline@Typed[
					TypeForAll[{"a"}, {"CArray"["a"], "MachineInteger", "CArray"["a"], "MachineInteger", "MachineInteger"} -> "Void"]
				]@Function[ {src, startSrc, dest, startDest, len},
					Module[ {srcFix = Native`AddressShift[src, startSrc], destFix = Native`AddressShift[dest, startDest]},
						Native`CopyTo[srcFix, destFix, len]
					]
				]];

		env["declareFunction", Native`SizeOf,
				inline@
				Typed[TypeForAll[{"arg1"}, {"arg1"} -> "MachineInteger"]]@
				Function[{arg},
					Native`SizeOf[arg]
				]
			];
			
		env["declareFunction", Compile`ResultOf,
				MetaData[<|"Class" -> "Erasure"|>
				]@TypeSpecifier[
					TypeForAll[{"res"}, {{} -> "res"} -> "res"]
					]
			];

		env["declareFunction", Compile`ConstantZero,
				inline@Typed[
					TypeForAll[{"a"}, {"a"} -> "a"]
					] @Function[{arg},
						Compile`ResultOf[Compile`ConstantZero, arg]
					]
			];
			
		env["declareFunction", Compile`ResultOf,
				MetaData[<|"Class" -> "Erasure"|>
				]@TypeSpecifier[
					TypeForAll[{"arg1", "res"}, {{"arg1"} -> "res", "arg1"} -> "res"]
					]
			];
	
		env["declareFunction", Compile`ResultOf,
				MetaData[<|"Class" -> "Erasure"|>
				]@TypeSpecifier[
					TypeForAll[{"arg1", "arg2", "res"}, {{"arg1", "arg2"} -> "res", "arg1", "arg2"} -> "res"]
					]
			];
		
		env["declareFunction", Compile`ResultOf,
				MetaData[<|"Class" -> "Erasure"|>
				]@TypeSpecifier[
					TypeForAll[{"arg1", "arg2", "arg3", "res"}, {{"arg1", "arg2", "arg3"} -> "res", "arg1", "arg2", "arg3"} -> "res"]
					]
			];
		
		env["declareFunction", Compile`ResultOf,
				MetaData[<|"Class" -> "Erasure"|>
				]@TypeSpecifier[
					TypeForAll[{"arg1", "arg2", "arg3", "arg4", "res"}, {{"arg1", "arg2", "arg3", "arg4"} -> "res", "arg1", "arg2", "arg3", "arg4"} -> "res"]
					]
			];

        env["declareFunction", Native`LoadClosureVariable,
                MetaData[
                    <|"Linkage" -> "External"|>
                ]@TypeSpecifier[
                    TypeForAll[{"a"}, {} -> "a"]
                ]
        ];

		env["declareType", 
			MetaData[<|"Fields" -> <|"f1" -> 1, "f2" -> 2|>|>
      			]@TypeConstructor["Structure2", {"*", "*"} -> "*"]];

		env["declareType", 
			MetaData[<|"Fields" -> <|"f1" -> 1, "f2" -> 2|>|>
      			]@TypeConstructor["Tuple", {"*", "*"} -> "*"]];


		env["declareFunction", Native`CreateTuple, 
			Typed[TypeFramework`TypeForAll[{"a", "b"}, {"a", "b"} -> "Tuple"["a", "b"]]
			]@Function[{arg1, arg2},
		    		Module[ {obj},
		        		obj = Native`CreateTupleHandle[];
		        		obj[[Native`Field[0]]] = arg1;
		        		obj[[Native`Field[1]]] = arg2;
		        		Native`Load[obj]
		    		]
			]];

		env["declareFunction", Native`FunctionApply, 
			MetaData[ <|"Inline" -> "Never"|>
			]@Typed[TypeFramework`TypeForAll[{"a", "b"}, {{"a"} -> "b", "a"} -> "b"]
			]@Function[{arg1, arg2},
		    		arg1[arg2]
			]];

		env["declareFunction", Native`AssertType,
				MetaData[<|"Class" -> "Erasure"|>
				]@Typed[
					TypeForAll[{"a"}, {"a", "a"} -> "Void"]
					] @Function[{arg1, arg2},
						Null
					]
			];
	
		env["declareFunction", Native`AssertTypeApplication,
				MetaData[<|"Class" -> "Erasure"|>
				]@Typed[
					TypeForAll[{"app", "arg1", "arg2"}, {"app"["arg1", "arg2"], "app"} -> "Void"]
					] @Function[{arg1, arg2},
						Null
					]
			];
		
		env["declareFunction", Native`PrimitiveFunction["AddressShift"], 
				llvmCompileToolsLinkage@
				TypeSpecifier[{"VoidHandle","Integer32"} -> "VoidHandle"]];
				
		env["declareFunction", Native`PrimitiveFunction["AddressShift"], 
				llvmCompileToolsLinkage@
				TypeSpecifier[{"VoidHandle","Integer64"} -> "VoidHandle"]];


		env["declareFunction", Native`AddressShift, 
			inline@
			Typed[TypeForAll[{"a"},
						{"CArray"["a"], "MachineInteger"} -> "CArray"["a"]]
					] @Function[{src, len},
						Module[ {voidArg = Native`BitCast[src, "VoidHandle"], lenBytes, res},
							Native`UncheckedBlock[
							lenBytes = len*Native`SizeOf[Native`GetElement[src,0]];
							res = Native`PrimitiveFunction["AddressShift"][voidArg, lenBytes];
							Native`BitCast[res, Compile`TypeOf[src]]]
						]
					]
			];


				
		env["declareFunction", Native`PrimitiveFunction["memcpyIntrinsicAligned32"], 
				llvmCompileToolsLinkage@
				TypeSpecifier[{"VoidHandle", "VoidHandle", "Integer32"} -> "Void"]];

		env["declareFunction", Native`PrimitiveFunction["memcpyIntrinsicAligned64"], 
				llvmCompileToolsLinkage@
				TypeSpecifier[{"VoidHandle", "VoidHandle", "Integer64"} -> "Void"]];
			
		env["declareFunction", Native`MemCpy,
				inline@Typed[
					{"VoidHandle", "VoidHandle", "Integer32"} -> "Void"
					] @Function[{arg1, arg2, arg3},
						Native`PrimitiveFunction["memcpyIntrinsicAligned32"][arg1, arg2, arg3]
					]
			];

		env["declareFunction", Native`MemCpy,
				inline@Typed[
					{"VoidHandle", "VoidHandle", "Integer64"} -> "Void"
					] @Function[{arg1, arg2, arg3},
						Native`PrimitiveFunction["memcpyIntrinsicAligned64"][arg1, arg2, arg3]
					]
			];
			
        env["declareFunction", Native`ReadCycleCounter,
                inline@
                Typed[{} -> "UnsignedInteger64"]@
                Function[{}, Native`PrimitiveFunction["ReadCycleCounter"][]]
        ];
        env["declareFunction", Native`PrimitiveFunction["ReadCycleCounter"],
                llvmCompileToolsLinkage@
                TypeSpecifier[{} -> "UnsignedInteger64"]
        ];

        Do[
	        env["declareFunction", Native`PrimitiveFunction["Expect"], 
	                llvmCompileToolsLinkage@
	                TypeSpecifier[{ty, ty} -> ty]
	        ],
	        {ty, {
	            "Boolean",
	            "Integer8",
	            "UnsignedInteger8",
	            "Integer16",
	            "UnsignedInteger16",
	            "Integer32",
	            "UnsignedInteger32",
	            "Integer64",
	            "UnsignedInteger64"
	        }}
        ];

	]
] (* StaticAnalysisIgnore *)


RegisterCallback["SetupTypeSystem", setupTypes]


End[]

EndPackage[]

