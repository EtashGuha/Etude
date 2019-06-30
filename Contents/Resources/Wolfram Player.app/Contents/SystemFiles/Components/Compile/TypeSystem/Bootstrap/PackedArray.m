
BeginPackage["Compile`TypeSystem`Bootstrap`PackedArray`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]

(*
   Support for PackedArrays.
   
   There is a type PackedArray which takes two args.  The base type and an integral literal which gives the rank!
   The BasePackable abstract class lists the base types which can go into a PackedArray.
   
   There is also an MTensor type which is the RTL implementation,  this doesn't distinguish based on base and rank.
*)

"StaticAnalysisIgnore"[

setupTypes[st_] :=
	With[{
		env = st["typeEnvironment"],
		inline = MetaData[<|"Inline" -> "Hint"|>],
		llvmLinkage = MetaData[<|"Linkage" -> "LLVMCompileTools"|>]
	},

		env["declareType", AbstractType["ArrayContainer", {}]];

		env["declareType", TypeConstructor["PackedArray", {"*", "*"} -> "*", "Implements" -> "ArrayContainer"]];
		
		env["declareType", TypeInstance["StringSerializable", {"a", "b"}, "PackedArray"["a", "b"]]];

		env["declareType", TypeConstructor["MTensor"]];
		
		
		env["declareType", AbstractType["BasePackable", {}]];
		env["declareType", TypeInstance["BasePackable", "MachineInteger"]];
		env["declareType", TypeInstance["BasePackable", "Real64"]];
		env["declareType", TypeInstance["BasePackable", "Complex"["Real64"]]];
(*
  Native`ArrayDimensions could also be written with a cast to an MTensor, this would be 
  better because external linked functions like AddGetMTensorDimensions should not be polymorphic.
*)
		env["declareFunction", Native`ArrayDimensions, 
				inline@Typed[
					TypeForAll[ {"a", "b"}, {"PackedArray"["a", "b"]} -> "CArray"["MachineInteger"]]
					]@Function[{arg}, Native`PrimitiveFunction["AddGetMTensorDimensions"][arg]]];
		
		env["declareFunction", Native`PrimitiveFunction["AddGetMTensorDimensions"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[TypeForAll[ {"a", "b"}, {"PackedArray"["a", "b"]} -> "CArray"["MachineInteger"]]]];

	
	(*
	  TODO,  tensor rank could return a constant at compile time
	*)
		env["declareFunction", ArrayDepth,
					inline@Typed[
						TypeForAll[ {"a"}, {Element["a", "BasePackable"]}, {"a"} -> "MachineInteger"]
					]@Function[{pa},
						0
					]];


		env["declareFunction", ArrayDepth,
					inline@Typed[
						TypeForAll[ {"container", "a"}, {Element["container", "ArrayContainer"]}, {"container"["a", 0]} -> "MachineInteger"]
					]@Function[{pa},
						0
					]];

		env["declareFunction", ArrayDepth,
					inline@Typed[
						TypeForAll[ {"container", "a"}, {Element["container", "ArrayContainer"]}, {"container"["a", 1]} -> "MachineInteger"]
					]@Function[{pa},
						1
					]];

		env["declareFunction", ArrayDepth,
					inline@Typed[
						TypeForAll[ {"container", "a"}, {Element["container", "ArrayContainer"]}, {"container"["a", 2]} -> "MachineInteger"]
					]@Function[{pa},
						2
					]];

		env["declareFunction", ArrayDepth,
					inline@Typed[
						TypeForAll[ {"container", "a"}, {Element["container", "ArrayContainer"]}, {"container"["a", 3]} -> "MachineInteger"]
					]@Function[{pa},
						3
					]];
					
		env["declareFunction", ArrayDepth,
					inline@Typed[
						TypeForAll[ {"container", "a"}, {Element["container", "ArrayContainer"]}, {"container"["a", 4]} -> "MachineInteger"]
					]@Function[{pa},
						4
					]];

		env["declareFunction", TensorRank,
					inline@Typed[
						TypeForAll[ {"a"}, {Element["a", "BasePackable"]}, {"a"} -> "MachineInteger"]
					]@Function[{pa},
						0
					]];


		env["declareFunction", TensorRank,
					inline@Typed[
						TypeForAll[ {"a", "b"}, {"PackedArray"["a", "b"]} -> "MachineInteger"]
					]@Function[{pa},
						ArrayDepth[pa]
					]];
					
		(*
		   Implement ArrayDepth generically, this could be done with the field,  ie not a constant
		*)

		(*
		  Should switch to Native`ArrayNumberOfElements.
		*)
		env["declareFunction", Native`ArrayNumberOfElements, 
				inline@Typed[
					TypeForAll[{"elem"}, {Element["elem", "BasePackable"]}, {"elem"} -> "MachineInteger"]
					]@Function[{arg}, 1]];
					
		env["declareFunction", Native`ArrayNumberOfElements, 
				inline@Typed[
					TypeForAll[{"a", "b"}, {"PackedArray"["a", "b"]} -> "MachineInteger"]
					]@Function[{arg}, Native`PrimitiveFunction["MTensorNumberOfElements"][arg]]];

		env["declareFunction", Native`MTensorNumberOfElements, 
				inline@Typed[
					TypeForAll[{"a", "b"}, {"PackedArray"["a", "b"]} -> "MachineInteger"]
					]@Function[{arg}, Native`PrimitiveFunction["MTensorNumberOfElements"][arg]]];

		env["declareFunction", Native`PrimitiveFunction["MTensorNumberOfElements"], 
						llvmLinkage@TypeSpecifier[TypeForAll[{"a", "b"}, {"PackedArray"["a", "b"]} -> "MachineInteger"]]];


		env["declareFunction", Native`ArrayData, 
				inline@Typed[
					TypeForAll[{"a", "b"}, {"PackedArray"["a", "b"]} -> "CArray"["a"]]
					]@Function[{arg},
						Module[ {res},
							res = Native`PrimitiveFunction["GetMTensorData"][arg];
							Native`BitCast[res, Compile`TypeOf[ Compile`ResultOf[Native`ArrayData, arg]]]
						]
					]
			];

		env["declareFunction", Native`PrimitiveFunction["GetMTensorData"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[TypeForAll[ {"a", "b"}, {"PackedArray"["a", "b"]} -> "VoidHandle"]]];

		
		env["declareFunction", Native`CreatePackedArray, 
			inline@Typed[
				TypeForAll[ {"a", "b"}, {"Integer32", "MachineInteger", "CArray"["MachineInteger"]} -> "PackedArray"["a", "b"]]
				]@Function[{paType, paRank, paDims},
					Module[ {res},
						res = Native`PrimitiveFunction["CreatePackedArray"][paType, paRank, paDims];
						Native`BitCast[res, Compile`TypeOf[Compile`ResultOf[ Native`CreatePackedArray, paType, paRank, paDims]]]
					]	
				]];

		env["declareFunction", Native`CreatePackedArray, 
			inline@Typed[
				TypeForAll[ {"a"}, {Element["a", "BasePackable"]},{"MachineInteger", "a"} -> "PackedArray"["a", 1]]
				]@Function[{len, elem},
					Module[ {paType, dims, res},
						paType = Native`MTensorElementType[elem];
						dims = Native`StackArray[1];
						dims[[0]] = len;
						res = Native`PrimitiveFunction["CreatePackedArray"][paType, 1, dims];
						Native`BitCast[res, Compile`TypeOf[Compile`ResultOf[ Native`CreatePackedArray, len, elem]]]
					]	
				]];

		env["declareFunction", Native`CreatePackedArray, 
			Typed[
				TypeForAll[ {"elem", "rank"},
							{"MachineInteger", "PackedArray"["elem", "rank"]} -> "PackedArray"["elem", TypeEvaluate[Plus, {"rank", 1}]]]
				]@Function[{len, elem},
					Module[ {rankElem, rank, paType, dims, dimsElem, res},
						rankElem = ArrayDepth[elem];
						rank = rankElem + 1;
						dims = Native`StackArray[rank];
						dims[[0]] = len;
						dimsElem = Native`ArrayDimensions[elem];
						Native`CopyTo[dimsElem, dims, 1, rankElem];
						paType = Native`MTensorElementType[elem];
						res = Native`PrimitiveFunction["CreatePackedArray"][paType, rank, dims];
						Native`BitCast[res, Compile`TypeOf[Compile`ResultOf[ Native`CreatePackedArray, len, elem]]]
					]	
				]];

		env["declareFunction", Native`CreatePackedArray, 
			inline@Typed[
				TypeForAll[ {"elem", "outerRank"}, {Element["elem", "BasePackable"]},{"MachineInteger", "CArray"["MachineInteger"],  "outerRank", "elem"} -> "PackedArray"["elem", "outerRank"]]
				]@Function[{outerRank,  outerDims, outerRankTy, elem},
					Module[ {paType, res},
						paType = Native`MTensorElementType[elem];
						res = Native`PrimitiveFunction["CreatePackedArray"][paType, outerRank, outerDims];
						Native`BitCast[res, Compile`TypeOf[Compile`ResultOf[ Native`CreatePackedArray, outerRank, outerDims, outerRankTy, elem]]]
					]	
				]];

		env["declareFunction", Native`CreatePackedArray, 
			inline@Typed[
				TypeForAll[ {"elem", "outerRank", "elemRank"}, {Element["elem", "BasePackable"]},
						{"MachineInteger", "CArray"["MachineInteger"],  "outerRank", "PackedArray"["elem", "elemRank"]} -> 
								"PackedArray"["elem", TypeEvaluate[Plus, {"outerRank", "elemRank"}]]]
				]@Function[{outerRank,  outerDims, outerRankTy, elem},
					Module[ {paType, elemRank, elemDims, rank, dims, res},
						elemRank = ArrayDepth[elem];
						rank = elemRank + outerRank;
						elemDims = Native`ArrayDimensions[elem];
						dims = Native`StackArray[rank];
						Native`CopyTo[outerDims, dims, outerRank];
						Native`CopyTo[elemDims, 0, dims, outerRank, elemRank];						
						paType = Native`MTensorElementType[elem];
						res = Native`PrimitiveFunction["CreatePackedArray"][paType, rank, dims];
						Native`BitCast[res, Compile`TypeOf[Compile`ResultOf[ Native`CreatePackedArray, outerRank, outerDims, outerRankTy, elem]]]
					]	
				]];


		env["declareFunction", Native`PrimitiveFunction["CreatePackedArray"], 
						MetaData[
							<|"Linkage" -> "Runtime"|>
							]@TypeSpecifier[  
								{"Integer32", "MachineInteger", "CArray"["MachineInteger"]} -> "MTensor"]];
		
		
		env["declareFunction", Native`MTensorElementType, 
			inline@Typed[
				     {"MachineInteger"} -> "Integer32"
				 ]@Function[ {arg}, Typed[2, "Integer32"]]];
		
		env["declareFunction", Native`MTensorElementType, 
			inline@Typed[
				 TypeForAll[ {"a"}, {"PackedArray"["MachineInteger", "a"]} -> "Integer32"]
				 ]@Function[ {arg}, Typed[2, "Integer32"]]];
		
		env["declareFunction", Native`MTensorElementType, 
			inline@Typed[
				     {"Real64"} -> "Integer32"
				 ]@Function[ {arg}, Typed[3, "Integer32"]]];
		
		env["declareFunction", Native`MTensorElementType, 
			inline@Typed[
				 TypeForAll[ {"a"}, {"PackedArray"["Real64", "a"]} -> "Integer32"]
				 ]@Function[ {arg}, Typed[3, "Integer32"]]];

		env["declareFunction", Native`MTensorElementType, 
			inline@Typed[
				     {"Complex"["Real64"]} -> "Integer32"
				 ]@Function[ {arg}, Typed[4, "Integer32"]]];
		
		env["declareFunction", Native`MTensorElementType, 
			inline@Typed[
				 TypeForAll[ {"a"}, {"PackedArray"["Complex"["Real64"], "a"]} -> "Integer32"]
				 ]@Function[ {arg}, Typed[4, "Integer32"]]];

		env["declareFunction", Native`MakeZeroRankTensor, 
			inline@Typed[
				 TypeForAll[ {"a"},{Element["a", "Integral"]},  {"a"} -> "PackedArray"["MachineInteger", 0]]
				 ]@Function[ {argIn},
				 	Module[{arg, base, paNew, data},
				 		arg = Compile`Cast[argIn, "MachineInteger"];
				 		base = Native`MTensorElementType[arg];
				 		paNew = Native`CreatePackedArray[ base, 0, Compile`NullReference];
				 		data = Native`ArrayData[paNew];
						data[[0]] = arg;
				 		paNew
				 	]
				 ]];
				 
		env["declareFunction", Native`MakeZeroRankTensor, 
			inline@Typed[
				 TypeForAll[ {"a"},{Element["a", "RealFloatingPoint"]},  {"a"} -> "PackedArray"["Real64", 0]]
				 ]@Function[ {argIn},
				 	Module[{arg, base, paNew, data},
				 		arg = Compile`Cast[argIn, "Real64"];
				 		base = Native`MTensorElementType[arg];
				 		paNew = Native`CreatePackedArray[ base, 0, Compile`NullReference];
				 		data = Native`ArrayData[paNew];
						data[[0]] = arg;
				 		paNew
				 	]
				 ]];

		env["declareFunction", Native`MakeZeroRankTensor, 
			inline@Typed[
				 TypeForAll[ {"a"},{Element["a", "RealFloatingPoint"]},  {"Complex"["a"]} -> "PackedArray"["ComplexReal64", 0]]
				 ]@Function[ {argIn},
				 	Module[{arg, base, paNew, data},
				 		arg = Compile`Cast[argIn, "ComplexReal64"];
				 		base = Native`MTensorElementType[arg];
				 		paNew = Native`CreatePackedArray[ base, 0, Compile`NullReference];
				 		data = Native`ArrayData[paNew];
						data[[0]] = arg;
				 		paNew
				 	]
				 ]];

		env["declareFunction", Length,
					Typed[
						TypeForAll[ {"a", "b"}, {"PackedArray"["a", "b"]} -> "MachineInteger"]
					]@Function[{pa},
						Native`GetElement[ Native`ArrayDimensions[pa], 0]
					]];
					
		env["declareFunction", Native`PrimitiveFunction["PackedArray`ElementType"],
		MetaData[<|"Class" -> "Erasure"|>]@
			TypeSpecifier[
				TypeForAll[{"a", "b"}, {"PackedArray"["a", "b"]} -> "a"]
			]
		];

		env["declareFunction", ByteCount,
			Typed[
				TypeForAll[ {"a", "b"}, {"PackedArray"["a", "b"]} -> "MachineInteger"]
			]@Function[{pa},
				Native`MTensorNumberOfElements[pa] * Native`SizeOf[Native`PrimitiveFunction["PackedArray`ElementType"][pa]]
			]
		];

(*
  Copy all data from in to out, starting at start.
  The expectation is that all of the elements are setup
  correctly.
*)
		env["declareFunction", Native`PackedArrayDataCopyTo,
				inline@Typed[
					TypeForAll[{"a", "b", "c"}, {"PackedArray"["a", "b"], "PackedArray"["a", "c"], "MachineInteger"} -> "Void"]
				]@Function[ {in, out, start},
					Module[ {dataIn, dataOut},
						dataIn = Native`ArrayData[in];
						dataOut = Native`ArrayData[out];
						Native`CopyTo[dataIn, dataOut, start, Native`MTensorNumberOfElements[in]];
					]
				]];

		env["declareFunction", Native`PackedArrayDataCopyTo,
				inline@Typed[
					TypeForAll[{"a", "b"}, {"a", "PackedArray"["a", "b"], "MachineInteger"} -> "Void"]
				]@Function[ {in, out, start},
					Module[ {dataOut},
						dataOut = Native`ArrayData[out];
						dataOut[[start]] = in;
					]
				]];

		env["declareFunction", Native`CheckPackedArrayDimensions,
				Typed[
					TypeForAll[{"a", "b"}, {"PackedArray"["a", "b"], "PackedArray"["a", "b"], "MachineInteger"} -> "Void"]
				]@Function[ {pa1, pa2, start},
					Module[ {dims1, dims2, rank},
						dims1 = Native`ArrayDimensions[pa1];
						dims2 = Native`ArrayDimensions[pa2];
						rank = ArrayDepth[pa1];
						(*
						  The rank must match because the type does.
						*)
						If[ rank =!= ArrayDepth[pa2],
							Print["Rank does not match"]];
						Do[
							If[dims1[[i]] =!= dims2[[i]],
								Native`ThrowWolframException[Typed[Native`ErrorCode["DimensionError"], "Integer32"]]]
								, {i,start,rank-1}]
					]
				]];
				
		env["declareFunction", Native`CheckPackedArrayDimensions,
				inline@Typed[
					TypeForAll[{"a", "b"}, {"PackedArray"["a", "b"], "PackedArray"["a", "b"]} -> "Void"]
				]@Function[ {pa1, pa2},
					Native`CheckPackedArrayDimensions[pa1, pa2, 0]
				]];

(*
 Check len values in arr1 starting at start1 with arr2 starting at start2.
*)				
		env["declareFunction", Native`CheckDimensions,
				inline@Typed[
					{"CArray"["MachineInteger"], "MachineInteger", "CArray"["MachineInteger"], "MachineInteger", "MachineInteger"} -> "Void"
				]@Function[ {arr1, start1, arr2, start2, len},
					Do[
						If[arr1[[i+start1]] =!= arr2[[i+start2]],
							Native`ThrowWolframException[Typed[Native`ErrorCode["DimensionError"], "Integer32"]]]
							, {i,0,len-1}]
				]];
				
				
		env["declareFunction", ToString,
				Typed[
					TypeForAll[{"a", "b"}, {"PackedArray"["a", "b"]} -> "String"]
				]@Function[ {pa},
					Module[ {rank = ArrayDepth[pa]},
						"PackedArray[" <> 
							ToString[rank] <> 
							"," <>
							Native`CArrayToString[Native`ArrayDimensions[pa], rank] <>
							"," <>
							Native`CArrayToString[Native`ArrayData[pa], Native`MTensorNumberOfElements[pa]] <>
							"]"
					]
				]];
	
	
			(*
			  Range implementation,  fix up for real arguments.
			*)
			env["declareFunction", Native`Range, 
			Typed[
				TypeForAll[ {"a", "b"},{Element["a", "Integral"], Element["b", "Integral"]}, {"a", "b"} -> "PackedArray"["TypeJoin"["MachineInteger", "b"], 1]]
			]@Function[{start, end},
					Module[{len, base, dims, pa, data, val},
						len = Compile`Cast[ end-start+1, TypeSpecifier["MachineInteger"]];
						base = Native`MTensorElementType[1];
						dims = Native`StackArray[1];
						Native`SetElement[dims, 0, len];
						pa = Native`CreatePackedArray[ base, 1, dims];
						data = Native`ArrayData[pa];
						val = Compile`Cast[start, TypeSpecifier["MachineInteger"]];
						Do[
							Native`SetElement[data,i,val];
							val = val+1, {i, 0, len-1}];
						pa
					]
				]];

			env["declareFunction", Native`Range, 
			Typed[
				TypeForAll[ {"a", "b", "c"},{Element["a", "Integral"], Element["b", "Integral"], Element["c", "Integral"]}, {"a", "b", "c"} -> "PackedArray"["MachineInteger", 1]]
			]@Function[{startIn, endIn, incIn},
					Module[{start, end, inc, len, base, dims, pa, data, val},
						start = Compile`Cast[ startIn, TypeSpecifier["MachineInteger"]];
						end = Compile`Cast[ endIn, TypeSpecifier["MachineInteger"]];
						inc = Compile`Cast[ incIn, TypeSpecifier["MachineInteger"]];
						len = Native`PrimitiveFunction["Runtime_IteratorCount_I_I_I_I"][start, end, inc]+1;
						base = Native`MTensorElementType[1];
						dims = Native`StackArray[1];
						Native`SetElement[dims, 0, len];
						pa = Native`CreatePackedArray[ base, 1, dims];
						data = Native`ArrayData[pa];
						Do[
							val = start+inc*i;
							Native`SetElement[data,i,val];
							, {i, 0, len-1}];
						pa
					]
				]];

			env["declareFunction", Native`Clone,
					inline@Typed[
						TypeForAll[ {"a", "b"}, {"PackedArray"["a", "b"]} -> "PackedArray"["a", "b"]]
					]@Function[{pa},
						Module[ {base, rank, dims, paOut, dataIn, dataOut, len},
							base = Native`MTensorElementType[pa];
							rank = ArrayDepth[pa];
							dims = Native`ArrayDimensions[pa];
							paOut = Native`CreatePackedArray[base, rank, dims];
							dataIn = Native`ArrayData[pa];
							dataOut = Native`ArrayData[paOut];
							len = Native`MTensorNumberOfElements[pa];
							Native`CopyTo[dataIn, dataOut, len];
							paOut	
						]
					]]; 

			env["declareFunction", Native`MutabilityClone,
					Typed[
						TypeForAll[ {"a", "b"}, {"PackedArray"["a", "b"]} -> "PackedArray"["a", "b"]]
					]@Function[{pa},
						Native`Clone[pa]
					]]; 

			env["declareFunction", Native`Copy,
					Typed[
						TypeForAll[ {"a", "b"}, {"MTensor", "PackedArray"["a", "b"]} -> "PackedArray"["a", "b"]]
					]@Function[{paOld, pa},
						Module[ {base, rank, dims, paOut, dataIn, dataOut, len},
							base = Native`MTensorElementType[pa];
							rank = ArrayDepth[pa];
							dims = Native`ArrayDimensions[pa];
							paOut = Native`CreatePackedArray[ base, rank, dims];
							dataIn = Native`ArrayData[pa];
							dataOut = Native`ArrayData[paOut];
							len = Native`MTensorNumberOfElements[pa];
							Native`CopyTo[dataIn, dataOut, len];
							paOut	
						]
					]]; 

			env["declareFunction", Native`CopyMTensorWithElem,
					Typed[
						TypeForAll[ {"a"}, {"PackedArray"["a", 1], "a"} -> "PackedArray"["a", 1]]
					]@Function[{pa, elem},
						Module[ {base, rank, dims, paOut, dataOut},
							base = Native`MTensorElementType[pa];
							rank = ArrayDepth[pa];
							dims = Native`ArrayDimensions[pa];
							paOut = Native`CreatePackedArray[ base, rank, dims];
							dataOut = Native`ArrayData[paOut];
							Native`SetElement[dataOut,0,elem];
							paOut	
						]
					]]; 
					
					
			env["declareFunction", Native`CopyMTensorWithElem,
					Typed[
						TypeForAll[ {"a", "b"}, {"PackedArray"["a", "b"], "PackedArray"["a", TypeEvaluate[Plus, {"b", -1}]]} -> "PackedArray"["a", "b"]]
					]@Function[{pa, elem},
						Module[ {base, rank, dims, len, dimsElem, paOut, dataOut, dataElem, numElem},
							base = Native`MTensorElementType[pa];
							rank = ArrayDepth[pa];
							dims = Native`StackArray[rank];
							len = Length[pa];
							dimsElem = Native`ArrayDimensions[elem];
							Native`SetElement[dims, 0, len];
							Native`CopyTo[dimsElem, dims, 1, len-1];
							paOut = Native`CreatePackedArray[ base, rank, dims];
							dataOut = Native`ArrayData[paOut];
							dataElem = Native`ArrayData[elem];
							numElem = Native`MTensorNumberOfElements[elem];
							Native`CopyTo[dataElem, dataOut, numElem];
							paOut	
						]
					]]; 

			Module[{tys = {"MachineInteger","Real64", "Complex"["Real64"]}, combs},
				combs = Tuples[ tys, 2];
				Scan[ addTypeJoinBase[ env, #]&, combs];
				Scan[ addPackedArrayCastElements[ env, #]&, tys];
			];
			
			env["declareFunction", Native`AppPrePend,
					inline@Typed[
						TypeForAll[ {"a"}, {"PackedArray"["a", 1], "a", "Boolean"} -> "PackedArray"["a", 1]]
					]@Function[{paIn, elem, appendQ},
						Module[ {base, rank, dims, len, dataIn, paOut, dataOut, numElem},
							base = Native`MTensorElementType[paIn];
							rank = TensorRank[paIn];
							dims = Native`StackArray[rank];
							len = Length[paIn];
							dims[[0]] = len + 1;
							paOut = Native`CreatePackedArray[ base, rank, dims];
							dataOut = Native`ArrayData[paOut];
							dataIn = Native`ArrayData[paIn];
							numElem = Native`MTensorNumberOfElements[paIn];
							If[appendQ,
								Native`CopyTo[dataIn, dataOut, 0, numElem];
								Native`SetElement[dataOut, numElem, elem],
								Native`CopyTo[dataIn, dataOut, 1, numElem];
								Native`SetElement[dataOut, 0, elem]];
							paOut	
						]
					]]; 
			
			env["declareFunction", Native`AppPrePend,
				inline@Typed[
					TypeForAll[ {"a", "b"}, {"PackedArray"["a", "b"], "PackedArray"["a", TypeEvaluate[Plus, {"b", -1}]], "Boolean"} -> "PackedArray"["a", "b"]]
				]@Function[{paIn, elem, appendQ},
					Module[ {base, rank, dimsIn, dimsOut, len, dataIn, paOut, dataOut, numIn, numElem, dimsElem, dataElem},
						base = Native`MTensorElementType[paIn];
						rank = TensorRank[paIn];
						dimsIn = Native`ArrayDimensions[paIn];
						dimsOut = Native`StackArray[rank];
						len = dimsIn[[0]];
						dimsOut[[0]] = len + 1;
						Native`CopyTo[dimsIn, dimsOut, 1, len-1];
						paOut = Native`CreatePackedArray[ base, rank, dimsOut];
						dataOut = Native`ArrayData[paOut];
						dataIn = Native`ArrayData[paIn];
						numIn = Native`MTensorNumberOfElements[paIn];
						numElem = Native`MTensorNumberOfElements[elem];
						dataElem = Native`ArrayData[elem];
						dimsElem = Native`ArrayDimensions[elem];
						Native`CheckDimensions[dimsIn, 1, dimsElem, 0, rank-1];
						If[appendQ,
							Native`CopyTo[dataIn, dataOut, 0, numIn];
							Native`CopyTo[dataElem, dataOut, numIn, numElem],
							Native`CopyTo[dataElem, dataOut, 0, numElem];
							Native`CopyTo[dataIn, dataOut, numElem, numIn]];
						paOut	
					]
				]
			]; 

			env["declareFunction", Append,
				Typed[
					TypeForAll[ {"a"},{Element["a", "BasePackable"]}, {"PackedArray"["a", 1], "a"} -> "PackedArray"["a", 1]]
				]@Function[ {arr, elem},
					Native`AppPrePend[arr, elem, True]
				]
			];
					
			env["declareFunction", Append,
				Typed[
					TypeForAll[ {"a", "b"}, {"PackedArray"["a", "b"], "PackedArray"["a", TypeEvaluate[Plus, {"b", -1}]]} -> "PackedArray"["a", "b"]]
				]@Function[ {arr, elem},
					Native`AppPrePend[arr, elem, True]
				]
			];

			env["declareFunction", Prepend,
				Typed[
					TypeForAll[ {"a"},{Element["a", "BasePackable"]},{"PackedArray"["a", 1], "a"} -> "PackedArray"["a", 1]]
				]@Function[ {arr, elem},
					Native`AppPrePend[arr, elem, False]
				]
			];
					
			env["declareFunction", Prepend,
				Typed[
					TypeForAll[ {"a", "b"}, {"PackedArray"["a", "b"], "PackedArray"["a", TypeEvaluate[Plus, {"b", -1}]]} -> "PackedArray"["a", "b"]]
				]@Function[ {arr, elem},
					Native`AppPrePend[arr, elem, False]
				]
			];

			
		]

] (* StaticAnalysisIgnore *)


"StaticAnalysisIgnore"[

addPackedArrayCastElements[env_, t2_] :=
	Module[{},
	    
        env["declareFunction", Compile`CastElements,
            MetaData[<|"Inline" -> "Always", "ArgumentAlias"->True|>
            ] @ Typed[
                TypeForAll[ {"a"}, {"PackedArray"[t2, "a"], t2} -> "PackedArray"[t2, "a"]]
            ] @ Function[{pa, trgt},
                pa
            ]
        ]; 
        
		env["declareFunction", Compile`CastElements,
			Typed[
				TypeForAll[ {"t1", "a"}, {"PackedArray"["t1", "a"], t2} -> "PackedArray"[t2, "a"]]
			] @ Function[{pa, trgt},
				Module[ {base, rank, dims, paOut, dataIn, dataOut, sizeIn, sizeOut},
					rank = ArrayDepth[pa];
					dims = Native`ArrayDimensions[pa];
					base = Native`MTensorElementType[trgt];
					paOut = Native`CreatePackedArray[ base, rank, dims];
					dataIn = Native`ArrayData[pa];
					dataOut = Native`ArrayData[paOut];
					sizeIn = Native`ArrayNumberOfElements[pa];
					sizeOut = Native`ArrayNumberOfElements[paOut];
					Native`CopyArrayElementsWork[dataIn, sizeIn, 0, dataOut, sizeOut, 0, sizeIn];
					paOut	
				]
			]]; 
	]

] (* StaticAnalysisIgnore *)



(*
  Add rules for TypeJoinBase,  this is used for conversions between PackedArray.
  Ie  Native`TypeJoinBase[ NA[ Int, 2], NA[ Complex, 2]] -> Complex but it doesn't need to care about the rank.
  Don't know if we need add a TypeAlias here,  we don't have BaseTypeJoin on the RHS of a type rule.
  I wonder if this could be done with the underlying 
*)
addTypeJoinBase[ env_, {t1_, t2_}] :=
	Module[ {ty},
		ty = TypeSpecifier[TypeForAll[ {"a", "b"}, {"PackedArray"[t1, "a"], "PackedArray"[t2, "b"]} -> "TypeJoin"[t1, t2]]];
		env["declareFunction", Native`TypeJoinBase, MetaData[<|"Class" -> "Erasure"|>]@ty];
		ty = TypeSpecifier[TypeForAll[ {"a", "b"}, {"PackedArray"[t2, "a"], "PackedArray"[t1, "b"]} -> "TypeJoin"[t1, t2]]];
		env["declareFunction", Native`TypeJoinBase, MetaData[<|"Class" -> "Erasure"|>]@ty];
	]

(*
 Only add one rule here.
*)
addTypeJoinBase[ env_, {t_, t_}] :=
	Module[ {ty},
		ty = TypeSpecifier[TypeForAll[ {"a", "b"}, {"PackedArray"[t, "a"], "PackedArray"[t, "b"]} -> t]];
		env["declareFunction", Native`TypeJoinBase, MetaData[<|"Class" -> "Erasure"|>]@ty];
	]





addLLVMFunction[env_, origName_, targetName_, ty_] :=
	(
		env["declareFunction", origName,  MetaData[<|"Redirect" -> Native`PrimitiveFunction[targetName]|>]@ty];
		env["declareFunction", Native`PrimitiveFunction[targetName], MetaData[<|"Linkage" -> "LLVMCompileTools"|>]@ty];
	)

RegisterCallback["SetupTypeSystem", setupTypes]

RegisterCallback["InstantiateFunctions", createFunctions]

	createFunctions[ state_] :=
		Module[{baseList, arrList},
			baseList = {"MachineInteger", "Real64", "Complex"["Real64"]};
			arrList = Flatten[ Table[ {"PackedArray"["MachineInteger", i], "PackedArray"["Real64", i], "PackedArray"["Complex"["Real64"], i]}, {i, 1, 3}]];
			Print["Instantiate PackedArray"];

			Print[ "Part"];
			Scan[
				Function[ {ty},
					Print["Part - ", ty];
				   state["create"][ Function[ {Typed[ arg1, ty], Typed[arg2, "MachineInteger"]}, Part[arg1, arg2]]];
				   state["create"][ Function[ {Typed[ arg1, ty], Typed[arg2, "Span"]}, Part[arg1, arg2]]];
				   state["create"][ Function[ {Typed[ arg1, ty]}, Length[arg1]]];
				   state["create"][ Function[ {Typed[ arg1, ty]}, ArrayDepth[arg1]]];
				]
				   ,
				   arrList];
				   
			Print[ "Part1"];
			Scan[
				Function[ {ty},
					Print["Part1 - ", ty];
				   state["create"][ Function[ {Typed[ arg1, ty], Typed[arg2, "MachineInteger"], Typed[arg3, "MachineInteger"]}, Part[arg1, arg2, arg3]]];
				]
				   ,
				   Select[ arrList, (#[[2]] > 1) &]];
			Print[ "Part2"];
			Scan[
				Function[ {ty},
					Print["Part2 - ", ty];
				   state["create"][ Function[ {Typed[ arg1, ty], Typed[arg2, "MachineInteger"], Typed[arg3, "MachineInteger"], Typed[arg4, "MachineInteger"]}, Part[arg1, arg2, arg3, arg4]]];
				]
				   ,
				   Select[ arrList, (#[[2]] > 2) &]];
			
			Print[ "Set Part"];
			Scan[
				Function[ {ty},
					Print["Set Part - ", ty];
					With[ {valTy = Part[ty,1]},
				   state["create"][ Function[ {Typed[ arg1, ty], Typed[arg2, "MachineInteger"], Typed[arg3, valTy]}, 
				   			Module[ {arr = arg1},
				   				Part[arr, arg2] = arg3;
				   				arr
				   			]
				   ]]];
				]
				   ,
				   Select[ arrList, (#[[2]] === 1) &]];
	
			Scan[
				Function[ {ty},
					Print["Set Part - ", ty];
					With[ {valTy = Part[ty,1]},
				   state["create"][ Function[ {Typed[ arg1, ty], Typed[arg2, "MachineInteger"], Typed[arg3, "MachineInteger"], Typed[arg4, valTy]}, 
				   			Module[ {arr = arg1},
				   				Part[arr, arg2, arg3] = arg4;
				   				arr
				   			]
				   ]]];
				]
				   ,
				   Select[ arrList, (#[[2]] === 2) &]];
		   
				   
			Print[ "List"];
			Scan[
				Function[ {ty},
					Print["List - ", ty];
				   state["create"][ Function[ {Typed[ arg1, ty]}, 
				   			List[arg1]]];
				   state["create"][ Function[ {Typed[ arg1, ty], Typed[ arg2, ty]}, 
				   			List[arg1, arg2]]];
				]
				   ,
				   arrList];
			Scan[
				Function[ {ty},
					Print["List - ", ty];
				   state["create"][ Function[ {Typed[ arg1, ty], Typed[ arg2, ty], Typed[ arg3, ty]}, 
				   			List[arg1, arg2, arg3]]];
				   state["create"][ Function[ {Typed[ arg1, ty], Typed[ arg2, ty], Typed[ arg3, ty], Typed[ arg4, ty]}, 
				   			List[arg1, arg2, arg3, arg4]]];
				]
				   ,
				   arrList];
				   
			Print[ "Join"];
			Scan[
				Function[ {ty},
					Print["Join - ", ty];
					   state["create"][ Function[ {Typed[ arg1, ty], Typed[ arg2, ty]}, 
				   			Join[arg1, arg2]]];
				   state["create"][ Function[ {Typed[ arg1, ty], Typed[ arg2, ty], Typed[ arg3, ty]}, 
				   			Join[arg1, arg2, arg3]]];
				   state["create"][ Function[ {Typed[ arg1, ty], Typed[ arg2, ty], Typed[ arg3, ty], Typed[ arg4, ty]}, 
				   			Join[arg1, arg2, arg3, arg4]]];
				]
				   ,
				   arrList];

			Print[ "CopyMTensorWithElem"];
			Scan[
				Function[ {ty},
					Print["CopyMTensorWithElem - ", ty];
				   state["create"][ Function[ {Typed[ arg1, "PackedArray"[ty, 1]], Typed[ arg2, ty]}, 
				   			Native`CopyMTensorWithElem[arg1, arg2]]];
					]
				   ,
				   baseList];

(*
 Temporarily disable Complex ToString instantiation.
*)
			Print[ "ToString"];
			Scan[
				Function[ {ty},
					Print["ToString - ", ty];
				   state["create"][ Function[ {Typed[ arg1, ty]}, 
				   			ToString[arg1]]];
					]
				   ,
				   DeleteCases[arrList, "PackedArray"["Complex"[_], _]]];


		]





"StaticAnalysisIgnore"[

setupPartFunctions[st_] :=
	Module[ {env = st["typeEnvironment"], inline = MetaData[<|"Inline" -> "Hint"|>]},

	
		env["declareType", TypeConstructor["StructTest2", {"*"} -> "*"]];
		env["declareType", TypeAlias["StructTest2"["a"], 
			"StructTest3"[ "a", "MachineInteger"], 
  			"VariableAlias" -> True]
  		];

		env["declareType", 
			MetaData[<|"Fields" -> <|"f1" -> 1, "f2" -> 2|>|>]@
			TypeConstructor["StructTest3", {"*", "*"} -> "*"]
		];


		env["declareFunction", Native`PartUnaryIndex, 
			inline@
			Typed[
				TypeForAll[ {"a", "b", "rank"}, {"a"["b", "rank"], "MachineInteger"} -> "MachineInteger"]
			]@Function[{pa, indexIn}, 
				Module[{len = Length[pa], index = indexIn},
					If[index < 0,
						index = len+index+1];
					If[index < 1 || index > len,
						Native`ThrowWolframException[Typed[Native`ErrorCode["ArrayPartError"], "Integer32"]]];
					index-1
				]
			]];
					
		env["declareFunction", Native`PartBinaryIndex, 
			inline@
			Typed[
				TypeForAll[ {"a", "b", "rank"}, {TypePredicate[ TypeEvaluate[Greater, {"rank", 1}], TrueQ]}, {"a"["b", "rank"], "MachineInteger", "MachineInteger"} -> "MachineInteger"]
			]@Function[{pa, indexIn0, indexIn1},
				Module[{
						index0 = indexIn0, index1 = indexIn1,
						len0 = Native`ArrayDimensions[pa][[0]], 
						len1 = Native`ArrayDimensions[pa][[1]], index},
					If[index0 < 0,
						index0 = len0+index0+1];
					If[index1 < 0,
						index1 = len1+index1+1];
					If[index1 < 1 || index1 > len1,
						Native`ThrowWolframException[Typed[Native`ErrorCode["ArrayPartError"], "Integer32"]]];
					If[index0 < 1 || index0 > len0,
						Native`ThrowWolframException[Typed[Native`ErrorCode["ArrayPartError"], "Integer32"]]];
					index = len1*(index0-1)+(index1-1);
					index
				]
			]];
					
		env["declareFunction", Native`GetPartUnary, 
			inline@Typed[
				TypeForAll[ {"container", "b"}, {"container"["b", 1], "MachineInteger"} -> "b"]
			]@Function[{pa, indexIn}, 
				Module[{index = Native`PartUnaryIndex[pa, indexIn]},
					Native`GetArrayElement[pa, index]
				]
			]
		];
		
		(*
		  TODO,  use Native`GetElement instead of Native`GetArrayElement 
		*)
		env["declareFunction", Native`GetArrayElement, 
			inline@Typed[
				TypeForAll[ {"container", "b"}, {"container"["b", 1], "MachineInteger"} -> "b"]
			]@Function[{pa, index}, 
				Module[{data = Native`ArrayData[pa]},
					Native`GetElement[data, index]
				]
			]
		];
		
							
		env["declareFunction", Native`GetPartUnary, 
			Typed[
				TypeForAll[{"a", "b", "c"},
					{TypePredicate[ TypeEvaluate[Greater, {"b", 1}], TrueQ]},
					{"c"["a", "b"], "MachineInteger"} -> "c"["a", TypeEvaluate[Plus, {"b", -1}]]
				]
			]@Function[{pa, indexIn},
				Module[ {index = Native`PartUnaryIndex[pa, indexIn]},
					Native`GetArrayElement[ pa, index]
				]
			]
		];

		env["declareFunction", Native`GetArrayElement, 
			Typed[
				TypeForAll[{"a", "b", "c"},
					{TypePredicate[ TypeEvaluate[Greater, {"b", 1}], TrueQ]},
					{"PackedArray"["a", "b"], "MachineInteger"} -> "PackedArray"["a", TypeEvaluate[Plus, {"b", -1}]]
				]
			]@Function[{paIn, index}, 
				Module[{dataIn, base, rankIn, dimsIn, rank, dims, pa, len, pos, data},
					base = Native`MTensorElementType[paIn];
					rankIn = TensorRank[paIn];
					dimsIn = Native`ArrayDimensions[paIn];
					rank = Native`UncheckedBlock@(rankIn-1);
					dims = Native`AddressShift[dimsIn, 1];
					pa = Native`CreatePackedArray[ base, rank, dims];
					len = Native`ArrayNumberOfElements[pa];
					pos = Native`UncheckedBlock@(index*len);
					dataIn = Native`ArrayData[paIn];
					data = Native`ArrayData[pa];
					Native`CopyTo[dataIn, pos, data, 0, len];
					pa
				]
			]
		];

		env["declareFunction", Native`GetPartUnary, 
			inline@Typed[
				TypeForAll[ {"container", "elem", "rank", "spec"}, 
					{TypePredicate[ TypeEvaluate[Unequal, {"spec", "MachineInteger"}], TrueQ]},
					{"container"["elem", "rank"], "spec"} -> "container"["elem", "rank"]]
			]@Function[{pa, spec}, 
				Native`GetPartNary[ pa, spec]
			]
		];


		env["declareFunction", Native`Unchecked[Native`GetPartUnary], 
			inline@Typed[
				TypeForAll[ {"a", "b"}, {"a"["b", 1], "MachineInteger"} -> "b"]
			]@Function[{pa, index}, 
				Native`UncheckedBlock@
					Native`GetArrayElement[pa, index-1]
			]
		];

		env["declareFunction", Native`Unchecked[Native`GetPartUnary], 
			Typed[
				TypeForAll[{"a", "b", "c"},
					{TypePredicate[ TypeEvaluate[Greater, {"b", 1}], TrueQ]},
					{"c"["a", "b"], "MachineInteger"} -> "c"["a", TypeEvaluate[Plus, {"b", -1}]]
				]
			]@Function[{pa, index}, 
				Native`UncheckedBlock@
					Native`GetArrayElement[ pa, index-1]
			]
		];


		env["declareFunction", Native`GetPartBinary, 
			Typed[
				TypeForAll[ {"a", "b"},
					{"a"["b", 2], "MachineInteger", "MachineInteger"} -> "b"
				]
			]@Function[{pa, index0, index1},
				Module[{data = Native`ArrayData[pa],index},
					index = Native`PartBinaryIndex[pa, index0, index1];
					Native`GetElement[data, index]
				]
			]];

		env["declareFunction", Native`ArrayElementIndex, 
			inline@
			Typed[
				TypeForAll[ {"a", "b", "rank"},{TypePredicate[ TypeEvaluate[Greater, {"b", 1}], TrueQ]}, {"a"["b", "rank"], "MachineInteger", "MachineInteger"} -> "MachineInteger"]
			]@Function[{pa, index0, index1},
				Module[{len1 = Native`ArrayDimensions[pa][[1]]},
					len1*index0 + index1
				]
			]];


		env["declareFunction", Native`GetArrayElement, 
			Typed[
				TypeForAll[ {"container", "b"}, {"container"["b", 2], "MachineInteger", "MachineInteger"} -> "b"]
			]@Function[{pa, index0, index1}, 
				Module[{data = Native`ArrayData[pa], pos},
					pos = Native`ArrayElementIndex[pa, index0, index1];
					Native`GetElement[data, pos]
				]
			]
		];

						
		env["declareFunction", Native`GetPartBinary, 
			Typed[
				TypeForAll[ {"a", "b", "c"}, {TypePredicate[ TypeEvaluate[Greater, {"b", 2}], TrueQ]}, {"c"["a", "b"], "MachineInteger", "MachineInteger"} -> "c"["a", TypeEvaluate[Plus, {"b", -2}]]]
			]@Function[{pa, index1, index2}, 
				Native`GetPartNary[ pa, index1, index2]
			]];

(*
  Might need a constraint that the rank is > 2.
*)
		env["declareFunction", Native`GetPartBinary, 
			inline@Typed[
				TypeForAll[ {"container", "elem", "rank", "spec1",  "spec2"}, 
					{TypePredicate[ TypeEvaluate[Unequal, {"spec1", "MachineInteger"}], TrueQ],
					 TypePredicate[ TypeEvaluate[Unequal, {"spec2", "MachineInteger"}], TrueQ]},
					{"container"["elem", "rank"], "spec1", "spec2"} -> "container"["elem", "rank"]]
			]@Function[{pa, index1, index2}, 
				Native`GetPartNary[ pa, index1, index2]
			]
		];

		env["declareFunction", Native`GetPartBinary, 
			inline@Typed[
				TypeForAll[ {"container", "elem", "rank", "spec"}, 
					{TypePredicate[ TypeEvaluate[Greater, {"rank", 1}], TrueQ],
					TypePredicate[ TypeEvaluate[Unequal, {"spec", "MachineInteger"}], TrueQ]},
					{"container"["elem", "rank"], "spec", "MachineInteger"} -> "container"["elem", TypeEvaluate[Plus, {"rank", -1}]]]
			]@Function[{pa, index1, index2}, 
				Native`GetPartNary[ pa, index1, index2]
			]
		];

		env["declareFunction", Native`GetPartBinary, 
			inline@Typed[
				TypeForAll[ {"container", "elem", "rank", "spec"}, 
					{TypePredicate[ TypeEvaluate[Greater, {"rank", 1}], TrueQ],
					TypePredicate[ TypeEvaluate[Unequal, {"spec", "MachineInteger"}], TrueQ]},
					{"container"["elem", "rank"], "MachineInteger", "spec"} -> "container"["elem", TypeEvaluate[Plus, {"rank", -1}]]]
			]@Function[{pa, index1, index2}, 
				Native`GetPartNary[ pa, index1, index2]
			]
		];


		env["declareType", TypeConstructor["ArrayPartViewBase", {"*", "*"} -> "*"]];

		env["declareType", TypeConstructor["ArrayPartView", {"*", "*"} -> "*"]];
	
		env["declareType", TypeAlias["ArrayPartView"["a", "b"], 
								       "Handle"["ArrayPartViewBase"["a", "b"]], 
  										"VariableAlias" -> True]];

		env["declareType", 
			MetaData[<|"Fields" -> <|"f1" -> 1, "f2" -> 2|>|>]@
			TypeConstructor["ArrayPartViewWork", {"*", "*", "*", "*", "*", "*"} -> "*"]];
		
		env["declareType", TypeAlias["ArrayPartViewBase"["a", "b"], 
								"ArrayPartViewWork"[ "a", "b", 
										"MachineInteger", "MachineInteger", "CArray"["Integer32"], "CArray"["VoidHandle"]], 
  										"VariableAlias" -> True]];
		
		env["declareFunction", Native`PartViewCopy,
			inline@Typed[
				TypeForAll[ {"base", "rankInit", "rankIn", "rankOut", "container"}, 
						{"ArrayPartView"["container"["base", "rankInit"],"container"["base", "rankIn"]]} -> 
							"ArrayPartView"["container"["base", "rankInit"],"container"["base", "rankOut"]]]
			]@Function[{objIn},
				Module[ {objOut},
					objOut = Native`StackAllocateObject["ArrayPartViewBase"];
					objOut[[Native`Field[0]]] = objIn[[Native`Field[0]]];
					objOut[[Native`Field[2]]] = objIn[[Native`Field[2]]];
					objOut[[Native`Field[3]]] = objIn[[Native`Field[3]]];
					objOut[[Native`Field[4]]] = objIn[[Native`Field[4]]];
					objOut[[Native`Field[5]]] = objIn[[Native`Field[5]]];
					objOut
				]
			]];

		
		(*
		  Error if indNum is greater than the rank.  Should really be a compile error.
		*)
		env["declareFunction", Native`CreatePartView,
			inline@Typed[
				TypeForAll[ {"a", "b", "c"}, {"c"["a", "b"], "MachineInteger"} -> "ArrayPartView"["c"["a", "b"],"c"["a", "b"]]]
			]@Function[{array, indNum},
					Module[{partProcessor, partTypes,partSpecs},
						partProcessor = Native`StackAllocateObject["ArrayPartViewBase"];
						partTypes = Native`StackArray[indNum];
						partSpecs = Native`StackArray[indNum];
						partProcessor[[Native`Field[0]]] = array;
						partProcessor[[Native`Field[2]]] = indNum;
						partProcessor[[Native`Field[3]]] = ArrayDepth[array];
						partProcessor[[Native`Field[4]]] = partTypes;
						partProcessor[[Native`Field[5]]] = partSpecs;
						partProcessor
					]]
				];

			
		env["declareFunction", Native`PartViewAdd, 
			inline@Typed[
				TypeForAll[ {"a", "b", "c", "d"},
					{TypePredicate[ TypeEvaluate[Greater, {"c", 0}], TrueQ]},
					{"ArrayPartView"["d"["a", "b"], "d"["a", "c"]], "MachineInteger", "MachineInteger"} -> 
						"ArrayPartView"["d"["a", "b"], "d"["a", TypeEvaluate[Plus, {"c", -1}]]]]
			]@Function[{partProcessorIn, index, partSpec},
					Module[{partProcessor, partTypes, partSpecs, handle},
						partProcessor = Native`PartViewCopy[partProcessorIn];
						partProcessor[[Native`Field[3]]] = partProcessor[[Native`Field[3]]]-1;
						partTypes = partProcessor[[Native`Field[4]]];
						partTypes[[index]] = Typed[0, "Integer32"];
						partSpecs = partProcessor[[Native`Field[5]]];
						handle = Native`CreateVoidHandle[partSpec];
						partSpecs[[index]] = handle;
						partProcessor
					]]
				];

		env["declareFunction", Native`PartViewAdd, 
			inline@Typed[
				TypeForAll[ {"a", "b", "c", "d"},
					{TypePredicate[ TypeEvaluate[Greater, {"c", 0}], TrueQ]},
					{"ArrayPartView"["d"["a", "b"], "d"["a", "c"]], "MachineInteger", "Span"} -> 
						"ArrayPartView"["d"["a", "b"], "d"["a", "c"]]]
			]@Function[{partProcessorIn, index, partSpec},
					Module[{partProcessor, partTypes, partSpecs, handle},
						partProcessor = Native`PartViewCopy[partProcessorIn];
						partProcessor[[Native`Field[3]]] = partProcessor[[Native`Field[3]]];
						partTypes = partProcessor[[Native`Field[4]]];
						partTypes[[index]] = Typed[3, "Integer32"];
						partSpecs = partProcessor[[Native`Field[5]]];
						handle = Native`CreateVoidHandle[partSpec];
						partSpecs[[index]] = handle;
						partProcessor
					]]
				];


		env["declareFunction", Native`PartViewFinalizeGet, 
			inline@Typed[
				TypeForAll[ {"a", "b", "c", "d"}, 
						{TypePredicate[ TypeEvaluate[Greater, {"c", 0}], TrueQ]},
						{"ArrayPartView"["d"["a", "b"], "d"["a", "c"]]} -> "d"["a", "c"]]
			]@Function[{partProc},
					Native`PartViewFinalizeGetWork[partProc]]
				];

		env["declareFunction", Native`PartViewFinalizeGet, 
			inline@Typed[
				TypeForAll[ {"a", "b", "c"}, 
						{"ArrayPartView"["c"["a", "b"], "c"["a", 0]]} ->  "a"]
			]@Function[{partProc},
					Module[{zeroNA},
						zeroNA = Native`PartViewFinalizeGetWork[ partProc];
						Native`ArrayData[zeroNA][[0]]
						]]
				];


		env["declareFunction", Native`PartViewFinalizeGetWork, 
			inline@Typed[
				TypeForAll[ {"a", "b", "c"}, 
						{"ArrayPartView"["PackedArray"["a", "b"], "PackedArray"["a", "c"]]} -> "PackedArray"["a", "c"]]
			]@Function[{partProc},
					Module[{inpPA, partTypes, partSpecs, rankOut, len, success, res, res1},
						inpPA = Native`BitCast[ partProc[[Native`Field[0]]], TypeSpecifier["MTensor"]];
						len = partProc[[Native`Field[2]]];
						rankOut = partProc[[Native`Field[3]]];
						partTypes = partProc[[Native`Field[4]]];
						partSpecs = partProc[[Native`Field[5]]];					
						res = Native`Handle[];
						res1 = Native`BitCast[ res, TypeSpecifier[ "Handle"["MTensor"]]];
						success = Native`PrimitiveFunction["MTensor_getParts"][
							Compile`NullReference, res1, inpPA, rankOut, len, partTypes, partSpecs];
						If[success =!= Typed[0, "Integer32"],
							Native`ThrowWolframException[Typed[Native`ErrorCode["ArrayPartError"], "Integer32"]]];
						Native`Load[ res]
					]]
				];

		env["declareFunction", Native`PrimitiveFunction["MTensor_getParts"], 
						MetaData[
							<|"Linkage" -> "Runtime"|>
							]@TypeSpecifier[ 
								{"MTensor", "Handle"["MTensor"], "MTensor", "MachineInteger", "MachineInteger", "CArray"["Integer32"], "CArray"["VoidHandle"]} -> "Integer32"]];


		env["declareFunction", Native`SetPartUnary, 
			inline@Typed[
			     TypeForAll[ {"elem"}, {"PackedArray"["elem", 1], "MachineInteger", "elem"} -> "Void"]
			]@Function[{pa, indexIn, val}, 
				Module[{index},
					index = Native`PartUnaryIndex[pa, indexIn];
					Native`SetArrayElementUnary[pa, index, val];
				]
			]
		];
		
		env["declareFunction", Native`SetArrayElementUnary, 
			inline@Typed[
				TypeForAll[ {"elem"}, {"PackedArray"["elem", 1], "MachineInteger", "elem"} -> "Void"]
			]@Function[{pa, index, val}, 
				Module[{data = Native`ArrayData[pa]},
					Native`SetElement[data, index, val];
				]
			]
		];
						
		env["declareFunction", Native`SetPartUnary, 
						Typed[
							TypeForAll[ {"elem", "rank"}, {TypePredicate[ TypeEvaluate[Greater, {"rank", 1}], TrueQ]}, 
									{"PackedArray"["elem", "rank"], "MachineInteger", "PackedArray"["elem", TypeEvaluate[Plus, {"rank", -1}]]} -> 
																"Void"]
							]@Function[{pa, indexIn, val}, 
								Module[{index},
									index = Native`PartUnaryIndex[pa, indexIn];
									Native`SetArrayElementUnary[pa, index, val];
								]
							]
			];

		env["declareFunction", Native`SetArrayElementUnary, 
			Typed[
				TypeForAll[ {"elem", "rank"}, {TypePredicate[ TypeEvaluate[Greater, {"rank", 1}], TrueQ]}, 
					{"PackedArray"["elem", "rank"], "MachineInteger", "PackedArray"["elem", TypeEvaluate[Plus, {"rank", -1}]]} -> 
					"Void"]
			]@Function[{pa, index, val}, 
				Module[{lenPart, lenElem, dataElem, pos, data},
					lenPart = Quotient[ Native`ArrayNumberOfElements[pa], Length[pa]];
					lenElem = Native`ArrayNumberOfElements[val];
					If[ lenPart =!= lenElem,
						Native`ThrowWolframException[Typed[Native`ErrorCode["ArrayPartError"], "Integer32"]]];
					pos = Native`UncheckedBlock@(index*lenElem);
					dataElem = Native`ArrayData[val];
					data = Native`ArrayData[pa];
					Native`CopyTo[dataElem, data, pos, lenElem];
				]
			]
		];

		env["declareFunction", Native`SetPartUnary, 
			inline@Typed[
				TypeForAll[ {"container", "elem", "rank", "spec", "value"}, 
					{TypePredicate[ TypeEvaluate[Unequal, {"spec", "MachineInteger"}], TrueQ]},
					{"container"["elem", "rank"], "spec", "value"} -> "Void"]
			]@Function[{pa, spec, elem}, 
				Native`SetPartNary[ pa, spec, elem]
			]
		];


		env["declareFunction", Native`Unchecked[Native`SetPartUnary], 
			inline@Typed[
			     TypeForAll[ {"a"}, {"PackedArray"["a", 1], "MachineInteger", "a"} -> "Void"]
			]@Function[{pa, index, val}, 
				Native`UncheckedBlock@
				Module[{data = Native`ArrayData[pa]},
					data[[index-1]] = val;
				]
			]
		];
							
		env["declareFunction", Native`Unchecked[Native`SetPartUnary], 
						Typed[
							TypeForAll[ {"a", "b", "c"}, {TypePredicate[ TypeEvaluate[Greater, {"b", 1}], TrueQ]}, {"PackedArray"["a", "b"], "MachineInteger", "PackedArray"["a", TypeEvaluate[Plus, {"b", -1}]]} -> 
																"Void"]
							]@Function[{pa, index, elem}, 
								Native`SetPartNary[ pa, index, elem];
							]];

		env["declareFunction", Native`SetPartBinary, 
						Typed[
							TypeForAll[ {"a"}, {"PackedArray"["a", 2], "MachineInteger", "MachineInteger", "a"} -> "Void"]
							]@Function[{pa, index0, index1, val}, 
								Module[{data = Native`ArrayData[pa], index},
									index = Native`PartBinaryIndex[pa, index0, index1];
									Native`SetElement[data, index, val];
								]
							]];

		env["declareFunction", Native`SetArrayElementBinary, 
			Typed[
				TypeForAll[ {"elem"},{Element["elem", "BasePackable"]}, {"PackedArray"["elem", 2], "MachineInteger", "elem"} -> "Void"]
			]@Function[{pa, index, val}, 
				Module[{data = Native`ArrayData[pa]},
					Native`SetElement[data, index, val];
				]
			]
		];
						
		env["declareFunction", Native`SetPartBinary, 
			Typed[
				TypeForAll[ {"a", "b"}, {TypePredicate[ TypeEvaluate[Greater, {"b", 2}], TrueQ]}, {"PackedArray"["a", "b"], "MachineInteger", "MachineInteger", "PackedArray"["a", TypeEvaluate[Plus, {"b", -2}]]} -> "Void"]
			]@Function[{pa, index1, index2, elem},
				Module[{pos},
					pos = Native`PartBinaryIndex[pa, index1, index2];
					Native`SetArrayElementBinary[pa, pos, elem];
				]
			]
		];

		env["declareFunction", Native`SetArrayElementBinary, 
			Typed[
				TypeForAll[ {"elem", "rank"}, {TypePredicate[ TypeEvaluate[Greater, {"rank", 2}], TrueQ]}, 
					{"PackedArray"["elem", "rank"], "MachineInteger", "PackedArray"["elem", TypeEvaluate[Plus, {"rank", -2}]]} -> 
					"Void"]
			]@Function[{pa, index, val}, 
				Module[{dims = Native`ArrayDimensions[pa], lenPart, lenElem, dataElem, pos, data},
					lenPart = Quotient[ Native`ArrayNumberOfElements[pa], dims[[0]]*dims[[1]]];
					lenElem = Native`ArrayNumberOfElements[val];
					If[ lenPart =!= lenElem,
						Native`ThrowWolframException[Typed[Native`ErrorCode["ArrayPartError"], "Integer32"]]];
					pos = Native`UncheckedBlock@(index*lenElem);
					dataElem = Native`ArrayData[val];
					data = Native`ArrayData[pa];
					Native`CopyTo[dataElem, data, pos, lenElem];
				]
			]
		];

		env["declareFunction", Native`SetArrayElementNary, 
			Typed[
				TypeForAll[ {"elem", "rank"},  
					{"PackedArray"["elem", "rank"], "MachineInteger", "MachineInteger", "elem"} -> 
					"Void"]
			]@Function[{pa, lenHole, index, val}, 
				Module[{pos, data},
					pos = index;
					data = Native`ArrayData[pa];
					Native`SetElement[data, pos, val];
				]
			]
		];

		env["declareFunction", Native`SetArrayElementNary, 
			Typed[
				TypeForAll[ {"elem", "rank", "rankElem"}, {TypePredicate[ TypeEvaluate[Greater, {"rank", "rankElem"}], TrueQ]}, 
					{"PackedArray"["elem", "rank"], "MachineInteger", "MachineInteger", "PackedArray"["elem", "rankElem"]} -> 
					"Void"]
			]@Function[{pa, lenHole, index, elem}, 
				Module[{lenElem, dataElem, pos, data},
					lenElem = Native`ArrayNumberOfElements[elem];
					If[ lenHole =!= lenElem,
						Native`ThrowWolframException[Typed[Native`ErrorCode["ArrayPartError"], "Integer32"]]];
					pos = Native`UncheckedBlock@(index*lenElem);
					dataElem = Native`ArrayData[elem];
					data = Native`ArrayData[pa];
					Native`CopyTo[dataElem, data, pos, lenElem];
				]
			]
		];

		env["declareFunction", Native`SetPartBinary, 
			inline@Typed[
				TypeForAll[ {"container", "elem", "rank", "spec1", "value"}, 
					{TypePredicate[ TypeEvaluate[Unequal, {"spec1", "MachineInteger"}], TrueQ]},
					{"container"["elem", "rank"], "MachineInteger", "spec1", "value"} -> "Void"]
			]@Function[{pa, spec1, spec2, elem}, 
				Native`SetPartNary[ pa, spec1, spec2, elem]
			]
		];

		env["declareFunction", Native`SetPartBinary, 
			inline@Typed[
				TypeForAll[ {"container", "elem", "rank", "spec1", "value"}, 
					{TypePredicate[ TypeEvaluate[Unequal, {"spec1", "MachineInteger"}], TrueQ]},
					{"container"["elem", "rank"], "spec1", "MachineInteger", "value"} -> "Void"]
			]@Function[{pa, spec1, spec2, elem}, 
				Native`SetPartNary[ pa, spec1, spec2, elem]
			]
		];

		env["declareFunction", Native`SetPartBinary, 
			inline@Typed[
				TypeForAll[ {"container", "elem", "rank", "spec1", "spec2", "value"}, 
					{TypePredicate[ TypeEvaluate[Unequal, {"spec1", "MachineInteger"}], TrueQ],
					 TypePredicate[ TypeEvaluate[Unequal, {"spec2", "MachineInteger"}], TrueQ]},
					{"container"["elem", "rank"], "spec1", "spec2", "value"} -> "Void"]
			]@Function[{pa, spec1, spec2, elem}, 
				Native`SetPartNary[ pa, spec1, spec2, elem]
			]
		];

		env["declareFunction", Native`PartViewFinalizeSet, 
			inline@Typed[
				TypeForAll[ {"a", "b", "c"},
						{"ArrayPartView"["PackedArray"["a", "b"], "PackedArray"["a", "c"]], "PackedArray"["a", "c"]} -> "Void"]
			]@Function[{partProc, elem},
					Module[{inpPA, elemTen, partTypes, partSpecs, rankOut, len, success},
						inpPA = Native`BitCast[ partProc[[Native`Field[0]]], TypeSpecifier["MTensor"]];
						len = partProc[[Native`Field[2]]];
						rankOut = partProc[[Native`Field[3]]];
						partTypes = partProc[[Native`Field[4]]];
						partSpecs = partProc[[Native`Field[5]]];
						elemTen = Native`BitCast[ elem, TypeSpecifier["MTensor"]];				
						success = Native`PrimitiveFunction["MTensor_setParts"][
							inpPA, elemTen, rankOut, len, partTypes, partSpecs];
						If[success =!= Typed[0, "Integer32"],
							Native`ThrowWolframException[Typed[Native`ErrorCode["ArrayPartError"], "Integer32"]]];
					]
				]];

		env["declareFunction", Native`PartViewFinalizeSet, 
			inline@Typed[
				TypeForAll[ {"a", "b"},
						{"ArrayPartView"["PackedArray"["a", "b"], "PackedArray"["a", 0]], "a"} -> "Void"]
			]@Function[{partProc, elem},
					Module[{zeroPA, data},
						zeroPA = Native`CreatePackedArray[Native`MTensorElementType[elem], 0, Compile`NullReference];
						data = Native`ArrayData[zeroPA];
						data[[0]] = elem;
						Native`PartViewFinalizeSet[partProc, zeroPA];
					]
				]];

		env["declareFunction", Native`PrimitiveFunction["MTensor_setParts"], 
						MetaData[
							<|"Linkage" -> "Runtime"|>
							]@TypeSpecifier[ 
								{"MTensor", "MTensor", "MachineInteger", "MachineInteger", "CArray"["Integer32"], "CArray"["VoidHandle"]} -> "Integer32"]];


	]

] (* StaticAnalysisIgnore *)




RegisterCallback["SetupTypeSystem", setupPartFunctions]


"StaticAnalysisIgnore"[

setupListFunctions[st_] :=
	With[ {env = st["typeEnvironment"], inline = MetaData[<|"Inline" -> "Hint"|>]},
	

		env["declareFunction", Native`ListUnary, 
			Typed[
				TypeForAll[ {"a"},{Element["a", "BasePackable"]}, {"a"} -> "PackedArray"["a", 1]]
			]@Function[{arg1}, 
				Module[{base, dims, pa, paData},
					base = Native`MTensorElementType[arg1];
					dims = Native`StackArray[1];
					dims[[0]] = 1;
					pa = Native`CreatePackedArray[ base, 1, dims];
					paData = Native`ArrayData[ pa];
					paData[[0]] = arg1;
					pa
				]
			]
		];

		env["declareFunction", Native`ListUnary, 
				Typed[
					TypeForAll[ {"a", "b"}, {"PackedArray"["a", "b"]} -> "PackedArray"["a", TypeEvaluate[Plus, {"b", 1}]]]
					]@Function[{arg1}, 
						Module[{base, rankArg, rank, dims, pa},
							base = Native`MTensorElementType[arg1];
							rankArg = ArrayDepth[arg1];
							rank = rankArg+1;
							dims = Native`StackArray[rank];
							dims[[0]] = 1;
							Native`CopyTo[Native`ArrayDimensions[arg1], dims, 1, rankArg];
							pa = Native`CreatePackedArray[ base, rank, dims];
							Native`PackedArrayDataCopyTo[arg1, pa, 0];
							pa
						]
			]];

		env["declareFunction", Native`ListBinary, 
				Typed[
					TypeForAll[ {"a"},{Element["a", "BasePackable"]}, {"a", "a"} -> "PackedArray"["a", 1]]
					]@Function[{arg1, arg2}, 
						Module[{base, dims, pa, paData},
							base = Native`MTensorElementType[arg1];
							dims = Native`StackArray[1];
							dims[[0]] = 2;
							pa = Native`CreatePackedArray[ base, 1, dims];
							paData = Native`ArrayData[pa];
							paData[[0]] = arg1;
							paData[[1]] = arg2;
							pa
						]
			]];

		env["declareFunction", Native`ListBinary, 
				Typed[
					TypeForAll[ {"a", "b"}, {"PackedArray"["a", "b"], "PackedArray"["a", "b"]} -> "PackedArray"["a", TypeEvaluate[Plus, {"b", 1}]]]
					]@Function[{arg1, arg2}, 
						Module[{base, rankArg, rank, dims, pa, len},
							Native`CheckPackedArrayDimensions[arg1, arg2];
							base = Native`MTensorElementType[arg1];
							rankArg = ArrayDepth[arg1];
							rank = rankArg+1;
							dims = Native`StackArray[rank];
							dims[[0]] = 2;
							Native`CopyTo[Native`ArrayDimensions[arg1], dims, 1, rankArg];
							pa = Native`CreatePackedArray[ base, rank, dims];
							len = Native`MTensorNumberOfElements[arg1];
							Native`PackedArrayDataCopyTo[arg1, pa, 0];
							Native`PackedArrayDataCopyTo[arg2, pa, len];
							pa
						]
			]];


		env["declareType", 
			MetaData[<|"Fields" -> <|"f1" -> 1, "f2" -> 2|>|>
      			]@TypeConstructor["ArrayListViewBase", {"*", "*"} -> "*"]];

		env["declareType", TypeConstructor["ArrayListView", {"*", "*"} -> "*"]];
	
		env["declareType", TypeAlias["ArrayListView"["a", "b"], 
								       "Handle"["ArrayListViewBase"["a", "b"]], 
  										"VariableAlias" -> True]];

		env["declareFunction", Native`CreateListView,
			inline@Typed[
				TypeForAll[ {"a"}, {"MachineInteger"} -> "ArrayListView"["MachineInteger","CArray"["a"]]]
			]@Function[{len},
					Module[{listView, elemArray},
						listView = Native`StackAllocateObject["ArrayListViewBase"];
						elemArray = Native`StackArray[len];
						listView[[Native`Field[0]]] = len;
						listView[[Native`Field[1]]] = elemArray;
						listView
					]]
				];

		env["declareFunction", Native`ListViewAdd, 
			inline@Typed[
				TypeForAll[ {"a"},
					{Element["a", "BasePackable"]}, 
					{"ArrayListView"["MachineInteger", "CArray"["a"]], "MachineInteger", "a"} -> "ArrayListView"["MachineInteger", "CArray"["a"]]]
			]@Function[{listView, index, elem},
					Module[{elemArray},
						elemArray = listView[[Native`Field[1]]];
						elemArray[[index]] = elem;
						listView
					]]
				];

		env["declareFunction", Native`ListViewAdd, 
			inline@Typed[
				TypeForAll[ {"a", "b"},
					{Element["a", "BasePackable"]}, 
					{"ArrayListView"["MachineInteger", "CArray"["PackedArray"["a", "b"]]], "MachineInteger", "PackedArray"["a", "b"]} -> 
									"ArrayListView"["MachineInteger", "CArray"["PackedArray"["a", "b"]]]]
			]@Function[{listView, index, elem},
					Module[{elemArray},
						elemArray = listView[[Native`Field[1]]];
						elemArray[[index]] = elem;
						listView
					]]
				];

		env["declareFunction", Native`ListViewFinalize, 
			inline@Typed[
				TypeForAll[ {"a"},
					{Element["a", "BasePackable"]},
					{"ArrayListView"["MachineInteger", "CArray"["a"]]} -> "PackedArray"["a", 1]]
			]@Function[{listView},
					Module[{len, elemArray, elem, base, dims, rank, pa, i, paData},
						len = listView[[Native`Field[0]]];
						elemArray = listView[[Native`Field[1]]];
						elem = elemArray[[0]];
						base = Native`MTensorElementType[elem];
						rank = 1;
						dims = Native`StackArray[rank];
						dims[[0]] = len;
						pa = Native`CreatePackedArray[ base, rank, dims];
						paData = Native`ArrayData[pa];
						Native`CopyTo[elemArray, paData, len];
						pa
					]]
				];

		env["declareFunction", Native`ListViewFinalize, 
			inline@Typed[
				TypeForAll[ {"a", "b"},
					{"ArrayListView"["MachineInteger", "CArray"["PackedArray"["a", "b"]]]} -> "PackedArray"["a", TypeEvaluate[Plus, {"b", 1}]]]
			]@Function[{listView},
					Module[{len, elemLen, elemArray, elem, elem1, base, dimsElem, dimsOut, rankElem, rank, pa, i, paData, elemData},
						len = listView[[Native`Field[0]]];
						elemArray = listView[[Native`Field[1]]];
						elem1 = elemArray[[0]];
						base = Native`MTensorElementType[elem1];
						rankElem = ArrayDepth[elem1];
						Native`UncheckedBlock[rank = rankElem + 1];
						dimsOut = Native`StackArray[rank];
						dimsElem = Native`ArrayDimensions[elem1];
						dimsOut[[0]] = len;
						Native`CopyTo[ dimsElem, dimsOut, 1, rankElem];
						pa = Native`CreatePackedArray[ base, rank, dimsOut];
						elemLen = Native`MTensorNumberOfElements[elem1];
						paData = Native`ArrayData[pa];
						Do[
							Native`UncheckedBlock[elem = elemArray[[i]]];
							Native`CheckPackedArrayDimensions[elem1, elem];
							Native`PackedArrayDataCopyTo[elem, pa, elemLen*i],
							{i,0,len-1}];
						pa
					]]
				];


		env["declareFunction", Native`JoinBinary, 
				inline@Typed[
					TypeForAll[ {"a", "b"}, {"PackedArray"["a", "b"], "PackedArray"["a", "b"]} -> "PackedArray"["a", "b"]]
					]@Function[{arg1, arg2}, 
						Module[{base, rankArg, rank, dims, pa, len},
							Native`CheckPackedArrayDimensions[arg1, arg2, 1];
							base = Native`MTensorElementType[arg1];
							rankArg = ArrayDepth[arg1];
							rank = rankArg;
							dims = Native`StackArray[rank];
							Native`CopyTo[Native`ArrayDimensions[arg1], dims, rankArg];
							dims[[0]] = Length[arg1] + Length[arg2];
							pa = Native`CreatePackedArray[ base, rank, dims];
							len = Native`MTensorNumberOfElements[arg1];
							Native`PackedArrayDataCopyTo[arg1, pa, 0];
							Native`PackedArrayDataCopyTo[arg2, pa, len];
							pa
						]
			]];

		env["declareFunction", Native`JoinViewFinalize, 
			inline@Typed[
				TypeForAll[ {"a", "b"},
					{"ArrayListView"["MachineInteger", "CArray"["PackedArray"["a", "b"]]]} -> "PackedArray"["a", "b"]]
			]@Function[{listView},
					Module[{len, outLen, elemArray, elem, elem1, base, dimsElem, dimsOut, rankElem, rank, pa, i, pos},
						len = listView[[Native`Field[0]]];
						elemArray = listView[[Native`Field[1]]];
						elem1 = elemArray[[0]];
						base = Native`MTensorElementType[elem1];
						rankElem = ArrayDepth[elem1];
						rank = rankElem;
						dimsOut = Native`StackArray[rank];
						dimsElem = Native`ArrayDimensions[elem1];
						Native`CopyTo[ dimsElem, dimsOut, rankElem];
						outLen = 0;
						Do[
							elem = elemArray[[i]];
							outLen = outLen + Length[elem]
							,
							{i,0,len-1}];
						dimsOut[[0]] = outLen;
						pa = Native`CreatePackedArray[ base, rank, dimsOut];
						pos = 0;
						Do[
							elem = elemArray[[i]];
							Native`CheckPackedArrayDimensions[elem1, elem, 1];
							Native`PackedArrayDataCopyTo[elem, pa, pos];
							pos = pos + Native`MTensorNumberOfElements[elem],
							{i,0,len-1}];
						pa
					]]
				];


	]

] (* StaticAnalysisIgnore *)


RegisterCallback["SetupTypeSystem", setupListFunctions]




"StaticAnalysisIgnore"[

setupMemoryManagement[st_] :=
	Module[ {env = st["typeEnvironment"], inline = MetaData[<|"Inline" -> "Hint"|>]},

		env["declareAtom", Native`EternalMTensor, TypeSpecifier["MTensor"]];

		env["declareFunction", Native`GetEternalPackedArray,
			inline@
			Typed[TypeForAll[{"a"}, {} -> "a"]
			]@Function[{},
				Native`BitCast[ Native`EternalMTensor, Compile`TypeOf[ Compile`ResultOf[Native`GetEternalPackedArray]]]
			]];

		env["declareFunction", Native`MemoryRelease, 
			inline@
			Typed[
				{"CArray"["MTensor"], "MachineInteger"} -> "Void"
			]@Function[{stack, pos},
				Module[ {ten, refcnt, cond},
					ten = stack[[pos]];
					refcnt = Native`PrimitiveFunction["MTensorRefCountDecrement"][ten];
					cond = refcnt === Typed[1, "UnsignedInteger64"];
					cond = Native`PrimitiveFunction["Expect"][cond, False];
					If[cond,
						Native`PrimitiveFunction["FreeMTensor"][ten]
					]
				]
			]];

		env["declareFunction", Native`MemoryAcquire, 
			inline@
			Typed[
				TypeForAll[ {"a", "b"}, {"PackedArray"["a", "b"]} -> "Void"]
			]@Function[{arr},
				Module[ {ten},
					ten = Native`BitCast[arr, TypeSpecifier["MTensor"]];
					Native`PrimitiveFunction["MTensorRefCountIncrement"][ten];
				]
			]];
							
		env["declareFunction", Native`MemoryStore, 
			inline@
			Typed[
				TypeForAll[ {"a", "b"}, {"CArray"["MTensor"], "MachineInteger", "PackedArray"["a", "b"]} -> "Void"]
			]@Function[{stack, pos, arr},
				Module[ {ten},
					ten = Native`BitCast[arr, TypeSpecifier["MTensor"]];
					stack[[pos]] = ten;
				]
			]];

		env["declareFunction", Native`MemoryStore, 
			inline@
			Typed[
				TypeForAll[ {"a", "b"}, {"CArray"["MTensor"], "MachineInteger", "MTensor"} -> "Void"]
			]@Function[{stack, pos, ten},
				stack[[pos]] = ten;
			]];


		env["declareFunction", Native`PrimitiveFunction["FreeMTensor"], 
				MetaData[
					<|"Linkage" -> "Runtime"|>
				]@TypeSpecifier[{"MTensor"} -> "Void"]];

		env["declareFunction", Native`PrimitiveFunction["MTensorRefCountIncrement"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[{"MTensor"} -> "UnsignedInteger64"]];

		env["declareFunction", Native`PrimitiveFunction["MTensorRefCountDecrement"], 
				MetaData[
					<|"Linkage" -> "LLVMCompileTools"|>
				]@TypeSpecifier[{"MTensor"} -> "UnsignedInteger64"]];



	]

] (* StaticAnalysisIgnore *)



RegisterCallback["SetupTypeSystem", setupMemoryManagement]




End[]

EndPackage[]
