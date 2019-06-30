
BeginPackage["Compile`TypeSystem`Bootstrap`NumericArray`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)

(*
   Support for NumericArray.
   
   NumericArray is the WL expression, and MNumericArray is the corresponding RTL C-language type.
   The NumericArrayPackable type is the base type of all element types that are allowed in NumericArray.
*)

"StaticAnalysisIgnore"[

setupTypes[st_] :=
	Module[{env = st["typeEnvironment"],
	        inline = MetaData[<|"Inline" -> "Hint"|>],
	        llvmLinkage = MetaData[<|"Linkage" -> "LLVMCompileTools"|>], numericArrayElementTypes, typesAndValues},

		env["declareType", TypeConstructor["NumericArray", {"*", "*"} -> "*", "Implements" -> "ArrayContainer"]];
		
		env["declareType", TypeInstance["StringSerializable", {"a", "b"}, "NumericArray"["a", "b"]]];

		env["declareType", TypeConstructor["MNumericArray"]];
		
		numericArrayElementTypes = {
            "Integer8",
            "UnsignedInteger8",
            "Integer16",
            "UnsignedInteger16",
            "Integer32",
            "UnsignedInteger32",
            "Integer64",
            "UnsignedInteger64",
            "Real32",
            "Real64",
            "ComplexReal32",
            "ComplexReal64"
		};
		
        env["declareType", AbstractType["NumericArrayPackable", {}]];
		Scan[
            env["declareType", TypeInstance["NumericArrayPackable", #]]&,
            numericArrayElementTypes
		];


		env["declareFunction", Native`ArrayDimensions, 
				inline@Typed[
					TypeForAll[ {"a", "b"}, {"NumericArray"["a", "b"]} -> "CArray"["MachineInteger"]]
					]@Function[{arg}, Native`PrimitiveFunction["AddGetMNumericArrayDimensions"][arg]]];
		
		env["declareFunction", Native`PrimitiveFunction["AddGetMNumericArrayDimensions"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[TypeForAll[ {"a", "b"}, {"NumericArray"["a", "b"]} -> "CArray"["MachineInteger"]]]];


		env["declareFunction", ArrayDepth,
					inline@Typed[
						TypeForAll[ {"a"}, {Element["a", "NumericArrayPackable"]}, {"a"} -> "MachineInteger"]
					]@Function[{pa},
						0
					]];


		env["declareFunction", Native`ArrayNumberOfElements, 
				inline@Typed[
					TypeForAll[{"a", "b"}, {"NumericArray"["a", "b"]} -> "MachineInteger"]
					]@Function[{arg}, Native`PrimitiveFunction["MNumericArrayNumberOfElements"][arg]]];

		env["declareFunction", Native`PrimitiveFunction["MNumericArrayNumberOfElements"], 
						llvmLinkage@TypeSpecifier[TypeForAll[{"a", "b"}, {"NumericArray"["a", "b"]} -> "MachineInteger"]]];


		env["declareFunction", Native`ArrayData, 
				inline@Typed[
					TypeForAll[{"a", "b"}, {"NumericArray"["a", "b"]} -> "CArray"["a"]]
					]@Function[{arg},
						Module[ {res},
							res = Native`PrimitiveFunction["MNumericArrayData"][arg];
							Native`BitCast[res, Compile`TypeOf[ Compile`ResultOf[Native`ArrayData, arg]]]
						]
					]
			];

		env["declareFunction", Native`PrimitiveFunction["MNumericArrayData"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[TypeForAll[ {"a", "b"}, {"NumericArray"["a", "b"]} -> "VoidHandle"]]];



		env["declareFunction", Length,
					Typed[
						TypeForAll[ {"a", "b"}, {"NumericArray"["a", "b"]} -> "MachineInteger"]
					]@Function[{pa},
						Native`GetElement[ Native`ArrayDimensions[pa], 0]
					]];


		env["declareFunction", Native`GetArrayElement, 
			Typed[
				TypeForAll[{"a", "b"},
					{TypePredicate[ TypeEvaluate[Greater, {"b", 1}], TrueQ]},
					{"NumericArray"["a", "b"], "MachineInteger"} -> "NumericArray"["a", TypeEvaluate[Plus, {"b", -1}]]
				]
			]@Function[{paIn, index}, 
				Module[{dataIn, base, rankIn, dimsIn, rank, dims, pa, len, pos, data},
					base = Native`MNumericArrayElementType[paIn];
					rankIn = ArrayDepth[paIn];
					dimsIn = Native`ArrayDimensions[paIn];
					rank = Native`UncheckedBlock@(rankIn-1);
					dims = Native`AddressShift[dimsIn, 1];
					pa = Native`CreateNumericArray[ base, rank, dims];
					len = Native`ArrayNumberOfElements[pa];
					pos = Native`UncheckedBlock@(index*len);
					dataIn = Native`ArrayData[paIn];
					data = Native`ArrayData[pa];
					Native`CopyTo[dataIn, pos, data, 0, len];
					pa
				]
			]
		];


		env["declareFunction", Native`PartViewFinalizeGetWork, 
			inline@Typed[
				TypeForAll[ {"a", "b", "c"}, 
						{"ArrayPartView"["NumericArray"["a", "b"], "NumericArray"["a", "c"]]} -> "NumericArray"["a", "c"]]
			]@Function[{partProc},
					Module[{inpPA, partTypes, partSpecs, rankOut, len, success, res, res1},
						inpPA = Native`BitCast[ partProc[[Native`Field[0]]], TypeSpecifier["MNumericArray"]];
						len = partProc[[Native`Field[2]]];
						rankOut = partProc[[Native`Field[3]]];
						partTypes = partProc[[Native`Field[4]]];
						partSpecs = partProc[[Native`Field[5]]];					
						res = Native`Handle[];
						res1 = Native`BitCast[ res, TypeSpecifier[ "Handle"["MNumericArray"]]];
						success = Native`PrimitiveFunction["MNumericArray_getParts"][
							Compile`NullReference, res1, inpPA, rankOut, len, partTypes, partSpecs];
						If[success =!= Typed[0, "Integer32"],
							Native`ThrowWolframException[Typed[Native`ErrorCode["ArrayPartError"], "Integer32"]]];
						Native`Load[ res]
					]]
				];

		env["declareFunction", Native`PrimitiveFunction["MNumericArray_getParts"], 
						MetaData[
							<|"Linkage" -> "Runtime"|>
							]@TypeSpecifier[ 
								{"MNumericArray", "Handle"["MNumericArray"], "MNumericArray", "MachineInteger", "MachineInteger", "CArray"["Integer32"], "CArray"["VoidHandle"]} -> "Integer32"]];


        (**  MNumericArrayElementType for all NumericArray types. Returns the int constant that defines the MNumericArray element type, per mdata_array_types.h in theRTL.  **)
        
        typesAndValues = {
            {"Integer8",          1 (* BIT8_TYPE *)},
            {"UnsignedInteger8",  2 (* UBIT8_TYPE *)},
            {"Integer16",         3 (* BIT16_TYPE *)},
            {"UnsignedInteger16", 4 (* UBIT16_TYPE *)},
            {"Integer32",         5 (* BIT32_TYPE *)},
            {"UnsignedInteger32", 6 (* UBIT32_TYPE *)},
            {"Integer64",         7 (* BIT64_TYPE *)},
            {"UnsignedInteger64", 8 (* UBIT64_TYPE *)},
            {"Real32",            9 (* REAL32_TYPE *)},
            {"Real64",            10 (* REAL64_TYPE *)},
            {"ComplexReal32",     11 (* COMPLEX_REAL32_TYPE *)},
            {"ComplexReal64",     12 (* COMPLEX_REAL64_TYPE *)}
        };
        
        Function[{typeName, intValue},
            env["declareFunction", Native`MNumericArrayElementType,
                inline @ Typed[
                    {typeName} -> "Integer32"
                ] @ Function[{arg}, Typed[intValue, "Integer32"]]
            ];
            env["declareFunction", Native`MNumericArrayElementType,
                inline @ Typed[
                    TypeForAll[{"a"}, {"NumericArray"[typeName, "a"]} -> "Integer32"]
                ] @ Function[{arg}, Typed[intValue, "Integer32"]]
            ]                
        ] @@@ typesAndValues;
        		
        (**  Native`CreateNumericArray  **)
        
		env["declareFunction", Native`CreateNumericArray,
		    inline @ Typed[
		        TypeForAll[{"a", "b"}, {"Integer32", "MachineInteger", "CArray"["MachineInteger"]} -> "NumericArray"["a", "b"]]
		    ] @ Function[{naType, naRank, naDims},
		            Module[{res}, 
		                res = Native`PrimitiveFunction["Runtime_CreateNumericArray"][naType, naRank, naDims];
		                Native`BitCast[res, Compile`TypeOf[Compile`ResultOf[Native`CreateNumericArray, naType, naRank, naDims]]
		            ]
		        ]
		    ]
		];
		
		
		env["declareFunction", Native`PrimitiveFunction["Runtime_CreateNumericArray"],
		    MetaData[<|"Linkage" -> "Runtime"|>] @
		       TypeSpecifier[{"Integer32", "MachineInteger", "CArray"["MachineInteger"]} -> "MNumericArray"]
		];
 
 
 
        (**  Compile`ArrayElementType returns a string giving the element type for either NumericArray or PackedArray. e.g. "Integer64", "Real64", etc.  **)
        Scan[
            env["declareFunction", Compile`ArrayElementType,
                inline @ Typed[
                    TypeForAll[{"container", "rank"}, {Element["container", "ArrayContainer"]}, {"container"[#, "rank"]} -> "String"]
                ] @ Function[{array}, #]
            ]&,
            numericArrayElementTypes
        ];
                
        
        (* Compile`ConvertArray and Compile`CastElements. *)
 
        Scan[addNumericArrayCastElements[env, #]&, numericArrayElementTypes];
 
        (* Converting either NA or PA to NA. *)
        env["declareFunction", Compile`ConvertArray,
            inline@
            Typed[
                TypeForAll[{"a", "b", "containerIn", "rank"}, {Element["containerIn", "ArrayContainer"], Element["b", "NumericArrayPackable"]},
                                    {"containerIn"["a", "rank"], "NumericArray"["b", "rank"]} -> "NumericArray"["b", "rank"]]
            ] @ Function[{arrayIn, naOutType},
                    Module[ {base, rank, dims, naOut, dataIn, dataOut, sizeIn, sizeOut},
                        rank = ArrayDepth[arrayIn];
                        dims = Native`ArrayDimensions[arrayIn];
                        base = Native`MNumericArrayElementType[naOutType];
                        naOut = Native`CreateNumericArray[base, rank, dims];
                        dataIn = Native`ArrayData[arrayIn];
                        dataOut = Native`ArrayData[naOut];
                        sizeIn = Native`ArrayNumberOfElements[arrayIn];
                        sizeOut = Native`ArrayNumberOfElements[naOut];
                        Native`CopyArrayElementsWork[ dataIn, sizeIn, 0, dataOut, sizeOut, 0, sizeIn];                     
                        naOut   
                    ]
               ]
        ];
        
        (* Converting either NA or PA to PA. *)
        env["declareFunction", Compile`ConvertArray,
            inline@
            Typed[
                TypeForAll[{"a", "b", "containerIn", "rank"}, {Element["containerIn", "ArrayContainer"], Element["b", "BasePackable"]}, 
                                    {"containerIn"["a", "rank"], "PackedArray"["b", "rank"]} -> "PackedArray"["b", "rank"]]
            ] @ Function[{arrayIn, paOutType},
                    Module[ {base, rank, dims, paOut, dataIn, dataOut, sizeIn, sizeOut},
                        rank = ArrayDepth[arrayIn];
                        dims = Native`ArrayDimensions[arrayIn];
                        base = Native`MTensorElementType[paOutType];
                        paOut = Native`CreatePackedArray[base, rank, dims];
						dataIn = Native`ArrayData[arrayIn];
                        dataOut = Native`ArrayData[paOut];
                        sizeIn = Native`ArrayNumberOfElements[arrayIn];
                        sizeOut = Native`ArrayNumberOfElements[paOut];
                       	Native`CopyArrayElementsWork[ dataIn, sizeIn, 0, dataOut, sizeOut, 0, sizeIn]; 
                        paOut
                   ]
               ]
        ];
 
        (* Make CastElements a no-op when the desired type is the same as the array's actual element type. *)               
        env["declareFunction", Compile`CastElements,
            inline @ Typed[
                TypeForAll[{"a", "container", "rank"}, {Element["container", "ArrayContainer"]}, 
                                    {"container"["a", "rank"], "a"} -> "container"["a", "rank"]]
            ] @ Function[{array, targetElemType}, array]
        ]
    ];


] (* StaticAnalysisIgnore *)


"StaticAnalysisIgnore"[

(* All the "declareFunction" calls that need a separate definition for each type supported by NumericArray go here *)
addNumericArrayCastElements[env_, resultElementType_] :=
    With[{inline = MetaData[<|"Inline" -> "Hint"|>]},
        (* Copies elements from one array to another, with casting as appropriate. Works for either NumericArray or PackedArray, of any depth.
           Copies from specified starting position in input array to specified starting position in output array. Indices are zero-based.
           Same signature as Java's System.arraycopy().
        *)
        env["declareFunction", Native`CopyArrayElements,
            inline@
            Typed[
                TypeForAll[{"a", "b", "containerIn", "containerOut", "rank"}, {Element["containerIn", "ArrayContainer"], Element["containerOut", "ArrayContainer"]}, 
                                    {"containerIn"["a", "rank"], "MachineInteger", "containerOut"[resultElementType, "rank"], "MachineInteger", "MachineInteger"} -> "Void"]
            ] @ Function[ {arrayIn, inPos, arrayOut, outPos, count},
                    Module[{dataIn, dataOut, sizeIn, sizeOut},
                        dataIn = Native`ArrayData[arrayIn];
                        dataOut = Native`ArrayData[arrayOut];
                        sizeIn = Native`ArrayNumberOfElements[arrayIn];
                        sizeOut = Native`ArrayNumberOfElements[arrayOut];
                        Native`CopyArrayElementsWork[ dataIn, sizeIn, inPos, dataOut, sizeOut, outPos, count]                        
                    ]
                ]
        ];

        env["declareFunction", Native`CopyArrayElementsWork,
            MetaData[<||>] @ Typed[
                TypeForAll[{"a", "b"},
                                    {"CArray"["a"], "MachineInteger", "MachineInteger", "CArray"[resultElementType], "MachineInteger", "MachineInteger", "MachineInteger"} -> "Void"]
            ] @ Function[ {dataIn, inSize, inPos, dataOut, outSize, outPos, count},
                    Module[{elem},
                        If[inPos < 0 || outPos < 0 || count < 0,
                            Native`ThrowWolframException[Typed[Native`ErrorCode["ArrayPartError"], "Integer32"]]
                        ];
                        If[inPos + count > inSize || outPos + count > outSize,
                            Native`ThrowWolframException[Typed[Native`ErrorCode["ArrayPartError"], "Integer32"]]
                        ];
                        Do[
                            Native`UncheckedBlock[
                            	elem = Native`ReinterpretCast[Native`GetElement[dataIn, inPos + i], resultElementType];
                            	Native`SetElement[dataOut, outPos + i, elem]];
                            ,
                            {i, 0, count-1}
                        ]
                        
                    ]
                ]
        ];

        env["declareFunction", Compile`CastElements,
            inline@
            Typed[
                TypeForAll[ {"t", "rank"}, {"NumericArray"["t", "rank"], resultElementType} -> "NumericArray"[resultElementType, "rank"]]
            ] @ Function[{na, targetElemType},
                    Module[ {base, rank, dims, naOut, dataIn, dataOut, sizeIn, sizeOut},
                        rank = ArrayDepth[na];
                        dims = Native`ArrayDimensions[na];
                        base = Native`MNumericArrayElementType[targetElemType];
                        naOut = Native`CreateNumericArray[base, rank, dims];
						dataIn = Native`ArrayData[na];
                        dataOut = Native`ArrayData[naOut];
                        sizeIn = Native`ArrayNumberOfElements[na];
                        sizeOut = Native`ArrayNumberOfElements[naOut];
                        Native`CopyArrayElementsWork[ dataIn, sizeIn, 0, dataOut, sizeOut, 0, sizeIn]   ;
                        naOut
                    ]
               ]
        ]
    ];
    

] (* StaticAnalysisIgnore *)



RegisterCallback["SetupTypeSystem", setupTypes]

RegisterCallback["InstantiateFunctions", createFunctions]

createFunctions[ state_] :=
	Module[{baseList},
		baseList = {"Integer32", "Integer64", "Real32", "Real64"};
		Print["Instantiate NumericArray"];

		Scan[
			Function[ty1,
				Scan[
					Function[ty2,
				
						state["create"][ Function[ {Typed[ arg1, "NumericArray"[ty1, 1]]}, 
						   			Compile`ConvertArray[arg1, "NumericArray"[ty2, 1]]
						   ]];
						state["create"][ Function[ {Typed[ arg1, "NumericArray"[ty1, 1]]}, 
						   			Compile`CastElements[arg1, ty2]
						   ]];
				], baseList];
			], baseList];

	]

End[]

EndPackage[]
