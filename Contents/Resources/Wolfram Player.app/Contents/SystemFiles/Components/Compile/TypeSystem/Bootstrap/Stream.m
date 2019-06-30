
BeginPackage["Compile`TypeSystem`Bootstrap`Stream`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]

(*
   Support for Streams.
   
   Initially PackedArray streams,  used by Fold
   
*)

"StaticAnalysisIgnore"[

setup[st_] :=
	Module[{env = st["typeEnvironment"],
	        inline = MetaData[<|"Inline" -> "Hint"|>]},

		(*
		    Readable streams.  These are fixed size streams that return a constant type.
		    You could build on top of them to combine eg Byte into Integer etc...
		*)
		
		(*
		 This declares the base type for the stream, it has two parameters. 
		 The type used to provide a source and the type that is read.
		*)
		env["declareType", TypeConstructor["DataReadStream", {"*", "*"} -> "*"]];

		(*
		 This declares the type of implementation of the stream. 
		 It is declared as a  struct (in the strange way that the type system provides). 
		 It has two extra fields which are used for holding the position and the length.
		*)
		env["declareType", 
			MetaData[<|"Fields" -> <|"f1" -> 1, "f2" -> 2|>|>
      			]@TypeConstructor["DataReadStreamBase", {"*", "*", "*", "*"} -> "*"]];
		
		(*
		 This declares a parametrized alias from the base type to the implementation. 
		 It fills in the types for the position and length.
		*)
		env["declareType", TypeAlias["DataReadStream"["a", "b"],
								"Handle"["DataReadStreamBase"[ "a", "b", "MachineInteger", "MachineInteger"]], 
  										"VariableAlias" -> True]];
				
		env["declareFunction", Native`StreamHasMoreData,
			inline@Typed[
				TypeForAll[ {"a", "b"}, {"DataReadStream"["a", "b"]} -> "Boolean"]
			]@Function[{streamObj},
					Module[{pos, len},
						pos = streamObj[[Native`Field[2]]];
						len = streamObj[[Native`Field[3]]];
						pos < len
					]]
				];

		(*
		    Writable streams.  These are fixed size streams that take a constant type.
		    You could build on top of them to combine eg Integer into Byte.  They 
		    are fixed size,  so there is a limit on what you can write in.
		*)

		(*
		 This declares the base type for the stream, it has two parameters. 
		 The type that is written and the type used to provide the result.
		*)
		env["declareType", TypeConstructor["DataWriteStream", {"*", "*"} -> "*"]];

		(*
		 This declares the type of implementation of the stream. 
		 It is declared as a  struct (in the strange way that the type system provides). 
		 It has two extra fields which are used for holding the position and the length.
		*)
		env["declareType", 
			MetaData[<|"Fields" -> <|"f1" -> 1, "f2" -> 2|>|>
      			]@TypeConstructor["DataWriteStreamBase", {"*", "*", "*", "*"} -> "*"]];
		
		(*
		 This declares a parametrized alias from the base type to the implementation. 
		 It fills in the types for the position and length.
		*)
		env["declareType", TypeAlias["DataWriteStream"["a", "b"],
								"Handle"["DataWriteStreamBase"[ "a", "b", "MachineInteger", "MachineInteger"]], 
  										"VariableAlias" -> True]];
				
		env["declareFunction", Native`StreamHasMoreSpace,
			inline@Typed[
				TypeForAll[ {"a", "b"}, {"DataReadStream"["a", "b"]} -> "Boolean"]
			]@Function[{streamObj},
					Module[{pos, len},
						pos = streamObj[[Native`Field[2]]];
						len = streamObj[[Native`Field[3]]];
						pos < len
					]]
				];

		env["declareFunction", Native`StreamHasMoreSpace,
			inline@Typed[
				TypeForAll[ {"a", "b"}, {"DataWriteStream"["a", "b"]} -> "Boolean"]
			]@Function[{streamObj},
					Module[{pos, len},
						pos = streamObj[[Native`Field[2]]];
						len = streamObj[[Native`Field[3]]];
						pos < len
					]]
				];

		env["declareFunction", Native`StreamGetResult,
			inline@Typed[
				TypeForAll[ {"a", "b"}, {"DataWriteStream"["a", "b"]} -> "a"]
			]@Function[{streamObj},
					streamObj[[Native`Field[0]]]
					]
				];


		env["declareFunction", Fold,
			Typed[
				TypeForAll[ {"elem", "accumulator", "cons"}, 
					{{"accumulator", "elem"} -> "accumulator", "accumulator", "DataReadStream"["cons", "elem"]} -> "accumulator"]
			]@Function[{fun, init, stream},
				Module[ {res, elem},
					res = init;
					While[Native`StreamHasMoreData[stream],
						elem = Native`StreamGetNextData[stream];
						res = fun[res, elem]
  					];
 				res
 			]]
		];

		env["declareFunction", Fold,
			Typed[
				TypeForAll[ {"elem", "cons"}, 
					{{"elem", "elem"} -> "elem", "DataReadStream"["cons", "elem"]} -> "elem"]
			]@Function[{fun, stream},
				Module[ {init},
					init = Native`StreamGetNextData[stream];
					Fold[fun, init, stream]
 			]]
		];


	]

] (* StaticAnalysisIgnore *)



RegisterCallback["SetupTypeSystem", setup]




End[]

EndPackage[]
