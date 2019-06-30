
BeginPackage["Compile`TypeSystem`Bootstrap`Basics`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]


resolvePointerSize[ tyEnv_] :=
    Module[{envSize = tyEnv["getProperty", "MachineIntegerSize", Null]},
        envSize
    ]
    
"StaticAnalysisIgnore"[

setupTypes[st_] :=
	With[{env = st["typeEnvironment"],
	      inline = MetaData[<|"Inline" -> "Hint"|>],
	      inlineAlways = MetaData[<|"Inline" -> "Always"|>]},
		env["declareType", TypeConstructor["Void"]];
		env["declareType", TypeConstructor["Error"]];
		env["declareType", TypeConstructor["Undefined", "Top" -> True, "ShortName" -> "\[CapitalOSlash]"]];
		env["declareType", TypeConstructor["Uninitialized"]];
		env["declareType", TypeConstructor["Empty", "Bottom" -> True]];
		env["declareType", TypeConstructor["Anything", "Top" -> True]];
		
		env["declareType", TypeAlias["ErrorCode", "Integer32"]];
		env["declareType", TypeAlias["MBool", "Integer32"]];

		env["declareType", TypeConstructor["Boolean", "Implements" -> "Equal"]];
		env["declareType", AbstractType["Equal", {"a"},
            {
                SameQ -> TypeSpecifier[{"a", "a"} -> "Boolean"],
                UnsameQ -> TypeSpecifier[{"a", "a"} -> "Boolean"],
                Equal -> TypeSpecifier[{"a", "a"} -> "Boolean"],
                Unequal -> TypeSpecifier[{"a", "a"} -> "Boolean"]
            }
        ]];

		env["declareFunction", SameQ, 
			inline@Typed[
				 {"Boolean", "Boolean"} -> "Boolean"
			]@Function[{x, y},
 				Native`PrimitiveFunction["binary_sameq"][x,y]
				]];

		env["declareFunction", Equal, 
			inline@Typed[
				 {"Boolean", "Boolean"} -> "Boolean"
			]@Function[{x, y},
 				Native`PrimitiveFunction["binary_equal"][x,y]
				]];

		env["declareFunction", SameQ, 
			inline@Typed[TypeForAll[ {"a", "b"}, 
					{TypePredicate[ TypeEvaluate[Unequal, {"a", "b"}], TrueQ]}, {"a", "b"} -> "Boolean"]
			]@Function[{x, y},
 				False
				]];

		env["declareFunction", TrueQ, 
			inline@Typed[
				 {"Boolean"} -> "Boolean"
			]@Function[{x},
 				x
				]];

		env["declareFunction", Not, Typed[
					{"Boolean"} -> "Boolean"
					]@Function[{arg}, Native`PrimitiveFunction["not_Boolean"][arg]]];
					
		env["declareFunction", Native`PrimitiveFunction["not_Boolean"], 
				MetaData[<|"Linkage" -> "LLVMCompileTools"|>]@TypeSpecifier[{"Boolean"} -> "Boolean"]];

		env["declareType", TypeConstructor["Handle", {"*"} -> "*", "ByteCount" -> ($SystemWordLength/8)]];
		env["declareType", TypeConstructor["TypeJoin", {"*", "*"} -> "*"]];
		env["declareFunction", Compile`Uninitialized, TypeSpecifier["Uninitialized"]];
		env["declareFunction", Undefined, TypeSpecifier["Undefined"]];

		env["declareAtom", Compile`NullReference, TypeSpecifier[TypeForAll[{"a"}, "a"]]];
		env["declareAtom", Compile`Void, TypeSpecifier["Void"]];
		
		(*
		   Making Null void means you can't assign to a function that returns Null 
		   which seems sensible.
		*)
		env["declareAtom", Null, TypeSpecifier["Void"]];
		
		env["declareFunction", Identity, MetaData[<|"ArgumentAlias"->True|>]@Typed[TypeForAll[ {"a"}, {"a"} -> "a"]
			]@Function[arg, arg]];

		env["declareFunction", Compile`TypeJoin,
		    MetaData[<|"Class" -> "Erasure"|>]@TypeSpecifier[TypeForAll[ {"a"}, {"a", "a"} -> "a"]]];
		
		env["declareFunction", Compile`Cast, 
			inlineAlways@
			Typed[Function[{arg1, arg2}, arg1], TypeForAll[ {"a"}, {"a", "a"} -> "a"]]];

		env["declareFunction", Native`BitCast, 
			inlineAlways@
			Typed[TypeForAll[ {"a", "b"}, {"a", "b"} -> "b"]]@
			Function[{arg1, arg2}, Native`PrimitiveFunction["BitCast"][arg1]]];

		env["declareFunction", Native`PrimitiveFunction["BitCast"], 
			MetaData[<|"Linkage" -> "LLVMInternal"|>
				]@TypeSpecifier[TypeForAll[ {"a", "b"}, {"a"} -> "b"]]];

		(*
		   Could this be done with "Handle"["Anything"] ?
		*)
		env["declareType", TypeConstructor["VoidHandle", "ShortName" -> "VH", "ByteCount" -> resolvePointerSize[env]]];
		env["declareFunction", Native`CreateVoidHandle,
			inline@
			Typed[
				TypeForAll[ {"a"}, {"a"} -> "VoidHandle"]
			]@Function[{arg},
				Module[{hand},
					hand = Native`Handle[];
					Native`Store[hand, arg];
					Native`BitCast[ hand, TypeSpecifier[ "VoidHandle"]]
				]
			]];


		env["declareFunction", Native`PrimitiveFunction["throwWolframException"], 
			MetaData[<|"Linkage" -> "Runtime", "Throws" -> True|>
				]@TypeSpecifier[{"Integer32"} -> "Void"]];
				
		env["declareFunction", Native`ThrowWolframException,
			inline@Typed[
				{"Integer32"} -> "Void"
			]@Function[{arg},
				Native`PrimitiveFunction["throwWolframException"][arg]
			]];
			
		env["declareFunction", Native`PrimitiveFunction["SetJumpStack_Push"], 
			MetaData[<|"Linkage" -> "Runtime"|>
				]@TypeSpecifier[{} -> "Integer64"]];

		env["declareFunction", Native`PrimitiveFunction["SetJumpStack_Pop"], 
			MetaData[<|"Linkage" -> "Runtime"|>
				]@TypeSpecifier[{} -> "Void"]];


			env["declareFunction", Native`PrimitiveFunction["getAbortWatchHandle"], 
				MetaData[<|"Linkage" -> "Runtime"|>
				]@TypeSpecifier[{} -> "Handle"["Integer32"]]];

			env["declareFunction", Native`PrimitiveFunction["checkAbortWatch"], 
			MetaData[<|"Linkage" -> "Runtime"|>
				]@TypeSpecifier[{} -> "Integer32"]];
		
			env["declareFunction", Native`PrimitiveFunction["checkAbortWatchThrow"], 
				MetaData[<|"Linkage" -> "Runtime"|>
				]@TypeSpecifier[{} -> "Integer32"]];

			env["declareFunction", Native`GetCheckAbort, 
				inline@Typed[
					{} -> "Integer32"
				]@Function[{},
					Module[{hand},
						hand = Native`Global["abortWatchHandle"];
						If[ Native`Load[hand] =!= Typed[0, "Integer32"],
							Native`PrimitiveFunction["checkAbortWatchThrow"][],
							Typed[0, "Integer32"]]
					]
			]];
			
		env["declareAtom", Native`Global["abortWatchHandle"], TypeSpecifier["Handle"["Integer32"]]];


	]
] (* StaticAnalysisIgnore *)

RegisterCallback["SetupTypeSystem", setupTypes]


setupMacros[st_] :=
	Module[ {env = st["macroEnvironment"]},
		RegisterMacro[env, Native`MBoolTrue,
			Native`MBoolTrue -> Typed[1, "MBool"]
		];
		RegisterMacro[env, Native`MBoolFalse,
			Native`MBoolFalse -> Typed[0, "MBool"]
		];

	]

RegisterCallback["SetupMacros", setupMacros]

End[]

EndPackage[]
