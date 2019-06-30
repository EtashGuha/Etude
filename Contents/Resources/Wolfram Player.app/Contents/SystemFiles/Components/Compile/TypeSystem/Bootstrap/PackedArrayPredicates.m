
BeginPackage["Compile`TypeSystem`Bootstrap`PackedArrayPredicates`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]

(*
   Support for Streams.
   
   Initially PackedArray streams,  used by Fold
   
*)

"StaticAnalysisIgnore"[

setup[st_] :=
	Module[{env = st["typeEnvironment"]},


		Module[ {
            func = Function[{fun, work, init, array},
                Module[ {loop = True, res = init, funRes, stream = Native`CreateDataReadStream[array], elem},
                    While[loop && Native`StreamHasMoreData[stream],
                        elem = Native`StreamGetNextData[stream];
                        funRes = fun[work, res, elem];
                        res = Native`GetField[funRes, 0];
                        loop = Native`GetField[funRes, 1];
                      ];
                    res
                ]]    
        },
            env["declareFunction", Compile`FoldVetoPredicate,
                Typed[
                    TypeForAll[ {"acc", "base"}, 
                    		{{ {"base"} -> "Boolean","acc", "base"} -> "Tuple"["acc", "Boolean"], 
                    		 {"base"} -> "Boolean", "acc", 
                    		 "PackedArray"["base",1]} -> "acc"]
                ]@func
            ];
            
            env["declareFunction", Compile`FoldVetoPredicate,
                Typed[
                    TypeForAll[ {"acc", "base", "rank"}, 
                     		{{ {"PackedArray"["base","rank"]} -> "Boolean","acc", "PackedArray"["base","rank"]} -> "Tuple"["acc", "Boolean"], 
                    		 {"PackedArray"["base","rank"]} -> "Boolean", "acc", 
                    		 "PackedArray"["base",TypeEvaluate[Plus, {"rank", 1}]]} -> "acc"]
                ]@func
            ];
           
         ];

     
		Module[ {
			(*
			   For the Tuple,  the first value is what is returned, and the second says whether to continue or not.
			*)
            allTrueFunc = Function[{arr, pred},
 						Module[{f = Function[{pred1, test, elem}, 
     									If[pred1[elem], 
     										Native`CreateTuple[test, True],   (* Continue *)
      										Native`CreateTuple[False, False]]]},
                 			Compile`FoldVetoPredicate[f, pred, True, arr]]],
                 			
            anyTrueFunc = Function[{arr, pred},
 						Module[{f = Function[{pred1, test, elem}, 
     									If[pred1[elem], 
     										Native`CreateTuple[True, False],   
      										Native`CreateTuple[test, True]]]}, (* Continue *)
                 			Compile`FoldVetoPredicate[f, pred, False, arr]]],
 
            noneTrueFunc = Function[{arr, pred},
 						Module[{f = Function[{pred1, test, elem}, 
     									If[pred1[elem], 
     										Native`CreateTuple[False, False],   
      										Native`CreateTuple[test, True]]]}, (* Continue *)
                 			Compile`FoldVetoPredicate[f, pred, True, arr]]],
          
            ty1 = Typed[ TypeForAll[ {"base"}, {"PackedArray"["base",1], {"base"} -> "Boolean"} -> "Boolean"]],
            ty2 = Typed[TypeForAll[ {"base", "rank"}, 
                    		{TypePredicate[ TypeEvaluate[Greater, {"rank", 0}], TrueQ]},
                    		{"PackedArray"["base",TypeEvaluate[Plus, {"rank", 1}]], 
                    			{"PackedArray"["base","rank"]} -> "Boolean"} -> "Boolean"]
                ] 
        },
            
            env["declareFunction", AllTrue, ty1@allTrueFunc];
            env["declareFunction", AllTrue, ty2@allTrueFunc];

            env["declareFunction", AnyTrue, ty1@anyTrueFunc];
            env["declareFunction", AnyTrue, ty2@anyTrueFunc];

            env["declareFunction", NoneTrue, ty1@noneTrueFunc];
            env["declareFunction", NoneTrue, ty2@noneTrueFunc];

       ];

		Module[ {
			func =
				Function[{array, fun},
					Module[ {len, elem, use, inStream = Native`CreateDataReadStream[array], outStream},
						len = Length[array];
						elem = Native`StreamGetNextData[inStream];
						use = fun[elem];
						While[
							!use && Native`StreamHasMoreData[inStream],
							len = len - 1;
							elem = Native`StreamGetNextData[inStream];
							use = fun[elem]];
						If[!use,
							Native`ThrowWolframException[Typed[Native`ErrorCode["SelectEmptyTensor"], "Integer32"]]];
						outStream = Native`CreatePackedArrayDataWriteStream[len, elem];
						While[Native`StreamHasMoreData[inStream],
							elem = Native`StreamGetNextData[inStream];
							use = fun[elem];
							If[use,
								Native`StreamPutNextData[outStream, elem]
								,
								len = len -1]
	  					];
	 				Native`StreamGetResult[outStream, len]
	 			]]	
		},

			env["declareFunction", Select,
				Typed[
					TypeForAll[ {"base"}, 
								{"PackedArray"["base", 1], {"base"} -> "Boolean"} -> 
																"PackedArray"["base", 1]]
				]@func
			];

			env["declareFunction", Select,
				Typed[
					TypeForAll[ {"base", "rank"}, 
						   {TypePredicate[ TypeEvaluate[Greater, {"rank", 1}], TrueQ]},
								{"PackedArray"["base", "rank"], 
									{"PackedArray"["base",TypeEvaluate[Plus, {"rank", -1}]]} -> "Boolean"} -> 
																"PackedArray"["base", "rank"]]
				]@func
			];

		];

		env["declareFunction", SameQ, 
			Typed[
				TypeForAll[ {"elem", "rank", "cont"}, {Element["cont", "ArrayContainer"]}, 
						{"cont"["elem", "rank"], "cont"["elem", "rank"]} -> "Boolean"]
			]@Function[{arg1, arg2},
				Module[{len1 = Native`ArrayNumberOfElements[arg1], len2 = Native`ArrayNumberOfElements[arg2], d1, d2},
					If[
						Compile`EagerOr[ 
							ArrayDepth[arg1] =!= ArrayDepth[arg2], len1 =!= len2],
						Return[False]];
					d1 = Native`ArrayData[arg1];
					d2 = Native`ArrayData[arg2];
					Native`UncheckedBlock[
						Do[
							If[d1[[i]] =!= d2[[i]],
								Return[False]];
							,{i, 0, len1-1}]
					];
					True
				]
			]];

		env["declareFunction", Equal, 
			Typed[
				TypeForAll[ {"elem1", "elem2", "rank1", "rank2"}, 
						{"PackedArray"["elem1", "rank1"], "PackedArray"["elem2", "rank2"]} -> "Boolean"]
			]@Function[{arg1, arg2},
				Module[{len1 = Native`ArrayNumberOfElements[arg1], len2 = Native`ArrayNumberOfElements[arg2], d1, d2},
					If[
						Compile`EagerOr[ 
							ArrayDepth[arg1] =!= ArrayDepth[arg2], len1 =!= len2],
						Return[False]];
					d1 = Native`ArrayData[arg1];
					d2 = Native`ArrayData[arg2];
					Native`UncheckedBlock[
						Do[
							If[d1[[i]] != d2[[i]],
								Return[False]];
							,{i, 0, len1-1}]
					];
					True
				]
			]];

(*
  These two would be better with a predicate that said other was not a PackedArray.
*)
		env["declareFunction", Equal, 
			Typed[
				TypeForAll[ {"elem", "rank", "other"}, 
						{Element["other", "Number"]},
						{"PackedArray"["elem", "rank"], "other"} -> "Boolean"]
			]@Function[{arg1, arg2},
				False
			]];

		env["declareFunction", Equal, 
			Typed[
				TypeForAll[ {"elem", "rank", "other"}, 
						{Element["other", "Number"]},
						{"other", "PackedArray"["elem", "rank"]} -> "Boolean"]
			]@Function[{arg1, arg2},
				False
			]];

	]

] (* StaticAnalysisIgnore *)



RegisterCallback["SetupTypeSystem", setup]

RegisterCallback["InstantiateFunctions", createFunctions]

createFunctions[ state_] :=
		Module[{},
			Print["Instantiate PackedArray Predicate"];
			
			Print["Select"];
			
			state["create"][ Function[{Typed[arg1, "PackedArray"["MachineInteger", 1]]}, 
						Select[arg1, # > 0 &]]];
						
			state["create"][ Function[{Typed[arg1, "PackedArray"["MachineInteger", 2]]}, 
						Select[arg1, First[#] > 0 &]]];
		]

End[]

EndPackage[]
