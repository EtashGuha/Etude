
BeginPackage["Compile`TypeSystem`Bootstrap`PackedArrayStructural`"]

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
	        inline = MetaData[<|"Inline" -> "Hint"|>],
	        alwaysInline = MetaData[<|"Inline" -> "Always"|>],
	        llvmLinkage = MetaData[<|"Linkage" -> "LLVMCompileTools"|>]},

		(*
		  Create a data read stream for rank 1 PackedArrays,  could be optimized to work on raw data.
		*)
		env["declareFunction", Native`CreateDataReadStream,
			inline@Typed[
				TypeForAll[ {"a"}, {"PackedArray"["a", 1]} -> "DataReadStream"["PackedArray"["a", 1], "a"]]
			]@Function[{array},
					Module[{streamObj, len},
						streamObj = Native`StackAllocateObject["DataReadStreamBase"];
						len = Length[array];
						streamObj[[Native`Field[0]]] = array;
						streamObj[[Native`Field[2]]] = 0;
						streamObj[[Native`Field[3]]] = len;
						streamObj
					]]
				];
				
		(*
		  Create a data read stream for rank > 1 PackedArrays.
		*)
		env["declareFunction", Native`CreateDataReadStream,
			inline@Typed[
				TypeForAll[ {"a", "b"}, {TypePredicate[ TypeEvaluate[Greater, {"b", 1}], TrueQ]}, {"PackedArray"["a", "b"]} -> 
						"DataReadStream"["PackedArray"["a", "b"], "PackedArray"["a", TypeEvaluate[Plus, {"b", -1}]]]]
			]@Function[{array},
					Module[{streamObj, len},
						streamObj = Native`StackAllocateObject["DataReadStreamBase"];
						len = Length[array];
						streamObj[[Native`Field[0]]] = array;
						streamObj[[Native`Field[2]]] = 0;
						streamObj[[Native`Field[3]]] = len;
						streamObj
					]]
				];
				

	   (*
	     Implement GetNextData method,  this could be specialized to work on raw data, esp for rank 1.
	   *)
		env["declareFunction", Native`StreamGetNextData,
			inline@Typed[
				TypeForAll[ {"a", "b", "c"}, {"DataReadStream"["PackedArray"["a","b"], "c"]} -> "c"]
			]@Function[{streamObj},
					Native`UncheckedBlock@
					Module[{pos, len, array, elem},
						pos = streamObj[[Native`Field[2]]];
						len = streamObj[[Native`Field[3]]];
						pos = pos+1;
						If[ pos > len,
							Native`ThrowWolframException[Typed[Native`ErrorCode["StreamDataNotAvailable"], "Integer32"]]];
						array = streamObj[[Native`Field[0]]];
						elem = Part[array,pos];
						streamObj[[Native`Field[2]]] = pos;
						elem
					]]
				];

		env["declareFunction", Native`CreatePackedArrayDataWriteStream,
			inline@Typed[
				TypeForAll[ {"a"}, {Element["a", "BasePackable"]}, {"MachineInteger", "a"} -> "DataWriteStream"["PackedArray"["a", 1], "a"]]
			]@Function[{len, elem},
					Native`UncheckedBlock@
					Module[{rank, base, dims, array, streamObj, pos},
                        If[ len <= 0,
                            Native`ThrowWolframException[Typed[Native`ErrorCode["DimensionError"], "Integer32"]]
                        ];
						rank = ArrayDepth[elem]+1;
						dims = Native`StackArray[rank];
						dims[[0]] = len;
						base = Native`MTensorElementType[elem];
						array = Native`CreatePackedArray[ base, rank, dims];
						pos = 1;
						Part[array, pos] = elem;
						streamObj = Native`StackAllocateObject["DataWriteStreamBase"];
						streamObj[[Native`Field[0]]] = array;
						streamObj[[Native`Field[2]]] = pos;
						streamObj[[Native`Field[3]]] = len;
						streamObj
					]]
				];

		env["declareFunction", Native`CreatePackedArrayDataWriteStream,
			alwaysInline@
			Typed[
				TypeForAll[ {"a", "b"}, {"MachineInteger", "PackedArray"["a", "b"]} -> "DataWriteStream"["PackedArray"["a", TypeEvaluate[Plus, {"b", 1}]], "PackedArray"["a", "b"]]]
			]@Function[{len, elem},
					Native`UncheckedBlock@
					Module[{dimsElem, rankElem, rank, base, dims, array, streamObj, pos},
                        If[ len <= 0,
                            Native`ThrowWolframException[Typed[Native`ErrorCode["DimensionError"], "Integer32"]]
                        ];
						rankElem = ArrayDepth[elem];
						rank = rankElem + 1;
						dims = Native`StackArray[rank];
						dims[[0]] = len;
						dimsElem = Native`ArrayDimensions[elem];
						Native`CopyTo[dimsElem, dims, 1, rankElem];
						base = Native`MTensorElementType[elem];
						array = Native`CreatePackedArray[ base, rank, dims];
						pos = 1;
						Part[array, pos] = elem;
						streamObj = Native`StackAllocateObject["DataWriteStreamBase"];
						streamObj[[Native`Field[0]]] = array;
						streamObj[[Native`Field[2]]] = pos;
						streamObj[[Native`Field[3]]] = len;
						streamObj
					]]
				];

				
        With[{
            f = Function[{streamObj, elem},
                    Native`UncheckedBlock@
                    Module[{pos, len, array},
                        pos = streamObj[[Native`Field[2]]];
                        len = streamObj[[Native`Field[3]]];
                        pos = pos+1;
                        If[ pos > len,
                            Native`ThrowWolframException[Typed[Native`ErrorCode["StreamSpaceNotAvailable"], "Integer32"]]];
                        array = streamObj[[Native`Field[0]]];
                        Part[array,pos] = elem;
                        streamObj[[Native`Field[2]]] = pos;                     
                    ]]
        },
			env["declareFunction", Native`StreamPutNextData,
				inline@Typed[
					TypeForAll[ {"a", "b"}, {Element["b", "BasePackable"]}, 
								{"DataWriteStream"["PackedArray"["a", 1], "b"], "b"} -> "Void"]
				]@f
			];
	
			env["declareFunction", Native`StreamPutNextData,
				inline@Typed[
					TypeForAll[ {"a", "b"}, 
						{"DataWriteStream"["PackedArray"["a", TypeEvaluate[Plus, {"b", 1}]], "PackedArray"["a", "b"]], 
							"PackedArray"["a", "b"]} -> "Void"]
				]@f
			];
	   ];
	       
	       
        With[{
            f = Function[{streamObj, elem},
                    Native`UncheckedBlock@
                    Module[{pos, array},
                        pos = streamObj[[Native`Field[2]]];
                        pos = pos+1;
                        array = streamObj[[Native`Field[0]]];
                        Part[array,pos] = elem;
                        streamObj[[Native`Field[2]]] = pos;                     
                    ]]
        },
            env["declareFunction", Native`UncheckedStreamPutNextData,
                inline@Typed[
                    TypeForAll[ {"a", "b"}, {Element["b", "BasePackable"]}, 
                                {"DataWriteStream"["PackedArray"["a", 1], "b"], "b"} -> "Void"]
                ]@f
            ];
    
            env["declareFunction", Native`UncheckedStreamPutNextData,
                inline@Typed[
                    TypeForAll[ {"a", "b"}, 
                        {"DataWriteStream"["PackedArray"["a", TypeEvaluate[Plus, {"b", 1}]], "PackedArray"["a", "b"]], 
                            "PackedArray"["a", "b"]} -> "Void"]
                ]@f
            ];
       ];


		Module[{
			func = Function[{streamObj, len},
				Module[{
					spaceLen = streamObj[[Native`Field[3]]],
					paGen = streamObj[[Native`Field[0]]], 
					base, rank, dimsNew, dimsGen, paNew,
					dataGen, dataNew, numElem
				},
					If[spaceLen === len,
						Return[paGen]];
					(*
					  Truncate to len
					*)
					base = Native`MTensorElementType[paGen];
					rank = ArrayDepth[paGen];
					dimsNew = Native`StackArray[rank];
					dimsGen = Native`ArrayDimensions[paGen];
					Native`SetElement[dimsNew, 0, len];
					Native`CopyTo[dimsGen, 1, dimsNew, 1, rank-1];
					paNew = Native`CreatePackedArray[ base, rank, dimsNew]; 
					dataGen = Native`ArrayData[paGen];
					dataNew = Native`ArrayData[paNew];
					numElem = Native`MTensorNumberOfElements[paNew];
					Native`CopyTo[dataGen, dataNew, numElem];
					paNew
					]
				]
		},
		env["declareFunction", Native`StreamGetResult,
			inline@Typed[
				TypeForAll[ {"base"}, 
					{"DataWriteStream"["PackedArray"["base", 1], "base"], "MachineInteger"} -> 
							"PackedArray"["base", 1]]
			]@func];
			
		env["declareFunction", Native`StreamGetResult,
			inline@Typed[
				TypeForAll[ {"base", "rank"}, 
					{"DataWriteStream"["PackedArray"["base", "rank"], 
							"PackedArray"["base", TypeEvaluate[Plus, {"rank", -1}]]], 
						"MachineInteger"} -> 
							"PackedArray"["base", "rank"]]
			]@func];
		];


		(*
		  Could these two implementations of Fold be merged?  The problem 
		  is matching the second argument of the function to the element of the packed array.
		  
		  If we could have a way to say that "PackedArray"[ "a", 0] is compatible with "a"
		  we would be in business.
		*)
		env["declareFunction", Fold,
			Typed[
				TypeForAll[ {"a", "b"}, {{"a", "b"} -> "a", "a", "PackedArray"["b",1]} -> "a"]
			]@Function[{fun, init, array},
				Module[ {res = init, stream = Native`CreateDataReadStream[array], elem},
					While[Native`StreamHasMoreData[stream],
						elem = Native`StreamGetNextData[stream];
						res = fun[res, elem]
  					];
 				res
 			]]
		];

		env["declareFunction", Fold,
			Typed[
				TypeForAll[ {"a", "b", "c"}, {TypePredicate[ TypeEvaluate[Greater, {"c", 0}], TrueQ]}, 
						{{"a", "PackedArray"["b","c"]} -> "a", "a", 
								"PackedArray"["b",TypeEvaluate[Plus, {"c", 1}]]} -> "a"]
			]@Function[{fun, init, array},
				Module[ {res = init, stream = Native`CreateDataReadStream[array], elem},
					While[Native`StreamHasMoreData[stream],
						elem = Native`StreamGetNextData[stream];
						res = fun[res, elem]
  					];
 				res
 			]]
		];


		(*
		  Assumes you cannot have zero length packed arrays.
		*)
		env["declareFunction", Fold,
			Typed[
				TypeForAll[ {"a"}, {{"a", "a"} -> "a", "PackedArray"["a",1]} -> "a"]
			]@Function[{fun, array},
				Module[ {res, stream = Native`CreateDataReadStream[array], elem},
					res = Native`StreamGetNextData[stream];
					While[Native`StreamHasMoreData[stream],
						elem = Native`StreamGetNextData[stream];
						res = fun[res, elem]
  					];
 				res
 			]]
		];

		env["declareFunction", Fold,
			Typed[
				TypeForAll[ {"a", "b"}, {TypePredicate[ TypeEvaluate[Greater, {"b", 0}], TrueQ]}, 
						{{"PackedArray"["a","b"], "PackedArray"["a","b"]} -> "PackedArray"["a","b"],  
								"PackedArray"["a",TypeEvaluate[Plus, {"b", 1}]]} -> "PackedArray"["a","b"]]
			]@Function[{fun, array},
				Module[ {res, stream = Native`CreateDataReadStream[array], elem},
					res = Native`StreamGetNextData[stream];
					While[Native`StreamHasMoreData[stream],
						elem = Native`StreamGetNextData[stream];
						res = fun[res, elem]
  					];
 				res
 			]]
		];
		
		Module[ {
            func = Function[{fun, init, array},
                Module[ {loop = True, res = init, funRes, stream = Native`CreateDataReadStream[array], elem},
                    While[loop && Native`StreamHasMoreData[stream],
                        elem = Native`StreamGetNextData[stream];
                        funRes = fun[res, elem];
                        res = Native`GetField[funRes, 0];
                        loop = Native`GetField[funRes, 1];
                      ];
                    res
                ]]    
        },
            env["declareFunction", Compile`FoldVeto,
                Typed[
                    TypeForAll[ {"a", "b"}, {{"a", "b"} -> "Tuple"["a", "Boolean"], "a", "PackedArray"["b",1]} -> "a"]
                ]@func
            ];
        ];
		
		env["declareFunction", Scan,
			Typed[
				TypeForAll[ {"a", "b"}, {{"a"} -> "b", "PackedArray"["a",1]} -> "Void"]
			]@Function[{fun, array},
				Module[ {stream = Native`CreateDataReadStream[array], elem},
					While[Native`StreamHasMoreData[stream],
						elem = Native`StreamGetNextData[stream];
						fun[elem]
  					];
 			]]
		];

		env["declareFunction", Scan,
			Typed[
				TypeForAll[ {"a", "b", "c"},{TypePredicate[ TypeEvaluate[Greater, {"b", 0}], TrueQ]},
									 {{"PackedArray"["a","b"]} -> "c", "PackedArray"["a",TypeEvaluate[Plus, {"b", 1}]]} -> "Void"]
			]@Function[{fun, array},
				Module[ {stream = Native`CreateDataReadStream[array], elem},
					While[Native`StreamHasMoreData[stream],
						elem = Native`StreamGetNextData[stream];
						fun[elem]
  					];
 			]]
		];


		With[ {
			func =
				Function[{fun, array},
					Module[ {len, elem, inStream = Native`CreateDataReadStream[array], outStream},
						len = Length[array];
						elem = Native`StreamGetNextData[inStream];
						elem = fun[elem];
						outStream = Native`CreatePackedArrayDataWriteStream[len, elem];
						While[Native`StreamHasMoreData[inStream],
							elem = Native`StreamGetNextData[inStream];
							elem = fun[elem];
							Native`StreamPutNextData[outStream, elem]
	  					];
	 				Native`StreamGetResult[outStream]
	 			]]	
		},

			env["declareFunction", Map,
				Typed[
					TypeForAll[ {"a", "b", "c"}, 
						{Element["a", "BasePackable"], Element["b", "BasePackable"]}, 
								{{"a"} -> "b", "PackedArray"["a", 1]} -> 
											"PackedArray"["b", 1]]
				]@func
			];

			env["declareFunction", Map,
				Typed[
					TypeForAll[ {"a", "b", "c"},
						{Element["a", "BasePackable"], TypePredicate[ TypeEvaluate[Greater, {"c", 0}], TrueQ]},  
								{{"a"} -> "PackedArray"["b", "c"], "PackedArray"["a", 1]} -> 
																	"PackedArray"["b", TypeEvaluate[Plus, {"c", 1}]]]
				]@func
			];

			env["declareFunction", Map,
				Typed[
					TypeForAll[ {"a", "b", "c"},
						{TypePredicate[ TypeEvaluate[Greater, {"b", 0}], TrueQ], Element["c", "BasePackable"]},  
								{{"PackedArray"["a", "b"]} -> "c", "PackedArray"["a", TypeEvaluate[Plus, {"b", 1}]]} -> 
																	"PackedArray"["c", 1]]
				]@func
			];

			env["declareFunction", Map,
				Typed[
					TypeForAll[ {"a", "b", "c", "d"},
					{TypePredicate[ TypeEvaluate[Greater, {"b", 0}], TrueQ], TypePredicate[ TypeEvaluate[Greater, {"d", 0}], TrueQ]},
								{{"PackedArray"["a", "b"]} -> "PackedArray"["c", "d"], "PackedArray"["a", TypeEvaluate[Plus, {"b", 1}]]} -> 
																	"PackedArray"["c", TypeEvaluate[Plus, {"d", 1}]]]
				]@func
			];
		];


		With[ {
			func =
				Function[{fun, array},
					Native`UncheckedBlock@
					Module[ {len, elem, inStream = Native`CreateDataReadStream[array], outStream, pos = 1, list},
						len = Length[array];
						elem = Native`StreamGetNextData[inStream];
						list = {pos};
						elem = fun[elem, list];
						outStream = Native`CreatePackedArrayDataWriteStream[len, elem];
						While[Native`StreamHasMoreData[inStream],
							pos = pos + 1;
							list = {pos};
							elem = Native`StreamGetNextData[inStream];
							elem = fun[elem, list];
							Native`StreamPutNextData[outStream, elem]
	  					];
	 				Native`StreamGetResult[outStream]
	 			]]	
		},

			env["declareFunction", MapIndexed,
				Typed[
					TypeForAll[ {"a", "b", "c"}, 
						{Element["a", "BasePackable"], Element["b", "BasePackable"]}, 
								{{"a", "PackedArray"["MachineInteger", 1]} -> "b", "PackedArray"["a", 1]} -> 
											"PackedArray"["b", 1]]
				]@func
			];

			env["declareFunction", MapIndexed,
				Typed[
					TypeForAll[ {"a", "b", "c"},
						{Element["a", "BasePackable"], TypePredicate[ TypeEvaluate[Greater, {"c", 0}], TrueQ]},  
								{{"a", "PackedArray"["MachineInteger", 1]} -> "PackedArray"["b", "c"], "PackedArray"["a", 1]} -> 
																	"PackedArray"["b", TypeEvaluate[Plus, {"c", 1}]]]
				]@func
			];

			env["declareFunction", MapIndexed,
				Typed[
					TypeForAll[ {"a", "b", "c"},
						{TypePredicate[ TypeEvaluate[Greater, {"b", 0}], TrueQ], Element["c", "BasePackable"]},  
								{{"PackedArray"["a", "b"], "PackedArray"["MachineInteger", 1]} -> "c", "PackedArray"["a", TypeEvaluate[Plus, {"b", 1}]]} -> 
																	"PackedArray"["c", 1]]
				]@func
			];

			env["declareFunction", MapIndexed,
				Typed[
					TypeForAll[ {"a", "b", "c", "d"},
					{TypePredicate[ TypeEvaluate[Greater, {"b", 0}], TrueQ], TypePredicate[ TypeEvaluate[Greater, {"d", 0}], TrueQ]},
								{{"PackedArray"["a", "b"], "PackedArray"["MachineInteger", 1]} -> "PackedArray"["c", "d"], "PackedArray"["a", TypeEvaluate[Plus, {"b", 1}]]} -> 
																	"PackedArray"["c", TypeEvaluate[Plus, {"d", 1}]]]
				]@func
			];
		];


		env["declareFunction", Total,
			Typed[
				TypeForAll[ {"a"}, {"PackedArray"["a",1]} -> "a"]
			]@Function[{array},
				Fold[Plus, array]]
		];

		env["declareFunction", Total,
			Typed[
				TypeForAll[ {"a", "b"}, {TypePredicate[ TypeEvaluate[Greater, {"b", 1}], TrueQ]}, 
							{"PackedArray"["a","b"]} ->"PackedArray"["a",TypeEvaluate[Plus, {"b", -1}]]]
			]@Function[{array},
				Fold[Plus, array]]
		];

		Module[ {
			func =
				Function[{fun, init, array},
					Module[ {res = init, len, elem, inStream = Native`CreateDataReadStream[array], outStream},
						len = Length[array] + 1;
						outStream = Native`CreatePackedArrayDataWriteStream[len, init];
						While[Native`StreamHasMoreData[inStream],
							elem = Native`StreamGetNextData[inStream];
							res = fun[res, elem];
							Native`StreamPutNextData[outStream, res]
	  					];
	 				Native`StreamGetResult[outStream]
	 			]]	
		},

		env["declareFunction", FoldList,
			Typed[
				TypeForAll[ {"a", "b"},{Element["a", "BasePackable"]}, {{"a", "b"} -> "a", "a", "PackedArray"["b",1]} -> 
								"PackedArray"["a", 1]]
			]@func];
		
		env["declareFunction", FoldList,
			Typed[
				TypeForAll[ {"a", "b", "c"}, {{"PackedArray"["a","b"], "c"} -> "PackedArray"["a","b"], "PackedArray"["a","b"], "PackedArray"["c",1]} -> 
								"PackedArray"["a",TypeEvaluate[Plus, {"b", 1}]]]
			]@func];

		env["declareFunction", FoldList,
			Typed[
				TypeForAll[ {"a", "b", "c"}, {Element["a", "BasePackable"], TypePredicate[ TypeEvaluate[Greater, {"c", 0}], TrueQ]}, 
								{{"a", "PackedArray"["b","c"]} -> "a", "a", "PackedArray"["b",TypeEvaluate[Plus, {"c", 1}]]} -> 
								"PackedArray"["a", 1]]
			]@func];
		
		env["declareFunction", FoldList,
			Typed[
				TypeForAll[ {"a", "b", "c", "d"}, {TypePredicate[ TypeEvaluate[Greater, {"d", 0}], TrueQ]},
						{{"PackedArray"["a","b"], "PackedArray"["c","d"]} -> "PackedArray"["a","b"], 
								"PackedArray"["a","b"], "PackedArray"["c",TypeEvaluate[Plus, {"d", 1}]]} -> 
						"PackedArray"["a",TypeEvaluate[Plus, {"b", 1}]]]
			]@func];
		
		];

		Module[ {
			func =
				Function[{fun, array},
					Module[ {res, len, elem, inStream = Native`CreateDataReadStream[array], outStream},
						res = Native`StreamGetNextData[inStream];
						len = Length[array];
						outStream = Native`CreatePackedArrayDataWriteStream[len, res];
						While[Native`StreamHasMoreData[inStream],
							elem = Native`StreamGetNextData[inStream];
							res = fun[res, elem];
							Native`StreamPutNextData[outStream, res]
	  					];
	 				Native`StreamGetResult[outStream]
	 			]]	
		},

		env["declareFunction", FoldList,
			Typed[
				TypeForAll[ {"a", "b"},{Element["a", "BasePackable"]}, {{"a", "a"} -> "a", "PackedArray"["a",1]} -> 
								"PackedArray"["a", 1]]
			]@func];
		
(*
  PA[1] -> PA and PA[2] -> scalar don't exist
*)

		
		env["declareFunction", FoldList,
			Typed[
				TypeForAll[ {"a", "b"}, {TypePredicate[ TypeEvaluate[Greater, {"b", 0}], TrueQ]},
						{{"PackedArray"["a","b"], "PackedArray"["a","b"]} -> "PackedArray"["a","b"], 
								"PackedArray"["a",TypeEvaluate[Plus, {"b", 1}]]} -> 
						"PackedArray"["a",TypeEvaluate[Plus, {"b", 1}]]]
			]@func];
		
		];

		env["declareFunction", Nest,
			Typed[
				TypeForAll[ {"a", "b"},
						{{"a"} -> "a", "a", "MachineInteger"} -> "a"]
								
			]@Function[{fun, init, cntIn},
				Native`UncheckedBlock@
				Module[{cnt = cntIn, elem = init},
					While[cnt > 0,
						cnt = cnt - 1;
						elem = fun[elem];
					];
					elem
				]
			]
		];

		Module[ {
			func =
				Function[{fun, init, cntIn},
					Native`UncheckedBlock@
					Module[ {elem = init, outStream, cnt = cntIn},
						outStream = Native`CreatePackedArrayDataWriteStream[cntIn+1, init];
						While[cnt > 0,
							cnt = cnt - 1;
							elem = fun[elem];
							Native`StreamPutNextData[outStream, elem]
	  					];
	 				Native`StreamGetResult[outStream]
	 			]]
			},
		env["declareFunction", NestList,
			Typed[
				TypeForAll[ {"a", "b"}, {Element["a", "BasePackable"]},
						{{"a"} -> "a", "a", "MachineInteger"} -> "PackedArray"["a", 1]]
								
			]@func
		];
		env["declareFunction", NestList,
			Typed[
				TypeForAll[ {"a", "b"},
						{{"PackedArray"["a", "b"]} -> "PackedArray"["a", "b"], "PackedArray"["a", "b"], "MachineInteger"} -> "PackedArray"["a", TypeEvaluate[Plus, {"b", 1}]]]
								
			]@func
		]];
		
(*
  Create a CArray read stream from a PackedArray or NumericArray.
*)		
		env["declareFunction", Native`CreateElementReadStream,
			inline@Typed[
				TypeForAll[ {"elem", "rank", "container"},{Element["container", "ArrayContainer"]}, {"container"["elem", "rank"]} -> "DataReadStream"["CArray"["elem"], "elem"]]
			]@Function[{array},
					Module[{streamObj, len},
						streamObj = Native`StackAllocateObject["DataReadStreamBase"];
						len = Native`ArrayNumberOfElements[array];
						streamObj[[Native`Field[0]]] = Native`ArrayData[array];
						streamObj[[Native`Field[2]]] = 0;
						streamObj[[Native`Field[3]]] = len;
						streamObj
					]]
				];

(*
  Implement GetNextData from the StreamData.
*)
		env["declareFunction", Native`StreamGetNextData,
			Typed[
				TypeForAll[ {"elem"}, {"DataReadStream"["CArray"["elem"], "elem"]} -> "elem"]
			]@Function[{streamObj},
					Native`UncheckedBlock@
					Module[{pos, len, array, elem},
						pos = streamObj[[Native`Field[2]]];
						len = streamObj[[Native`Field[3]]];
						If[ pos >= len,
							Native`ThrowWolframException[Typed[Native`ErrorCode["StreamDataNotAvailable"], "Integer32"]]];
						array = streamObj[[Native`Field[0]]];
						elem = Part[array,pos];
						streamObj[[Native`Field[2]]] = pos + 1;
						elem
					]]
				];

	]

] (* StaticAnalysisIgnore *)



RegisterCallback["SetupTypeSystem", setup]


RegisterCallback["InstantiateFunctions", createFunctions]

createFunctions[ state_] :=
		Module[{baseList, arrList},
			baseList = {"MachineInteger", "Real64", "Complex"["Real64"]};
			arrList = Flatten[ Table[ {"PackedArray"["MachineInteger", i], "PackedArray"["Real64", i], "PackedArray"["Complex"["Real64"], i]}, {i, 1, 3}]];
			Print["Instantiate PackedArray Structural"];

			Print[ "Map"];
			Scan[
				Function[ {tys},
					Print["Map - ", tys];
					With[ {ty1 = tys[[1]], ty2 = tys[[2]]},
				   		state["create"][ Function[ {Typed[ arg1, {ty1} -> ty2], Typed[arg2, "PackedArray"[ty1, 1]]}, 
				   			Map[ arg1, arg2]]];
					];
					With[ {ty1 = tys[[1]], ty2 = tys[[2]]},
				   		state["create"][ Function[ {Typed[ arg1, {ty1} -> "PackedArray"[ty2, 1]], Typed[arg2, "PackedArray"[ty1, 1]]}, 
				   			Map[ arg1, arg2]]];
					];
					With[ {ty1 = tys[[1]], ty2 = tys[[2]]},
				   		state["create"][ Function[ {Typed[ arg1, {"PackedArray"[ty1, 1]} -> ty2], Typed[arg2, "PackedArray"[ty1, 2]]}, 
				   			Map[ arg1, arg2]]];
					];
					With[ {ty1 = tys[[1]], ty2 = tys[[2]]},
				   		state["create"][ Function[ {Typed[ arg1, {"PackedArray"[ty1, 1]} -> "PackedArray"[ty2, 1]], Typed[arg2, "PackedArray"[ty1, 2]]}, 
				   			Map[ arg1, arg2]]];
					];

				]
				   ,
				   Flatten[Outer[ List, baseList, baseList], 1]];

		]

End[]

EndPackage[]
