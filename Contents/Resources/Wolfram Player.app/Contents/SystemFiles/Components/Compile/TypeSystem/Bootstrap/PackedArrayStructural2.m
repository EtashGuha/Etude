
BeginPackage["Compile`TypeSystem`Bootstrap`PackedArrayStructural2`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]
Needs["CompileAST`Create`Construct`"]

"StaticAnalysisIgnore"[

setupTypes[st_] :=
	Module[{env = st["typeEnvironment"],
			inline = MetaData[<|"Inline" -> "Hint"|>]},

		env["declareFunction", Native`PrimitiveFunction["Runtime_MTensor_Partition"], 
						MetaData[
							<|"Linkage" -> "Runtime"|>
							]@TypeSpecifier[ 
								{"Handle"["MTensor"], "MTensor", "MTensor", "MTensor"} -> "ErrorCode"]];

(*
  This would be quite easy to implement directly.
*)
		env["declareFunction", Partition,
			inline@Typed[
				TypeForAll[ {"elem", "rank"}, {"PackedArray"["elem", "rank"], "MachineInteger"} -> 
								"PackedArray"["elem", TypeEvaluate[Plus, {"rank", 1}]]]
			]@Function[{array, len},
				Module[{hand = Native`Handle[], hand1, err, lenTen},
					lenTen = {len};
					hand1 = Native`BitCast[ hand, "Handle"["MTensor"]];
					Native`Store[hand1, Compile`NullReference];
					err = Native`PrimitiveFunction["Runtime_MTensor_Partition"][hand1, 
								Native`BitCast[array, "MTensor"], Native`BitCast[lenTen, "MTensor"], Native`BitCast[lenTen, "MTensor"]];
					If[err =!= Typed[0, "Integer32"],
						Native`ThrowWolframException[Typed[Native`ErrorCode["PartitionError"], "Integer32"]]];
					Native`Load[hand]
				]
			]];

		env["declareFunction", Partition,
			inline@Typed[
				TypeForAll[ {"elem", "rank"}, {"PackedArray"["elem", "rank"], "MachineInteger", "MachineInteger"} -> 
								"PackedArray"["elem", TypeEvaluate[Plus, {"rank", 1}]]]
			]@Function[{array, len, off},
				Module[{hand = Native`Handle[], hand1, err, lenTen, offTen},
					lenTen = {len};
					offTen = {off};
					hand1 = Native`BitCast[ hand, "Handle"["MTensor"]];
					Native`Store[hand1, Compile`NullReference];
					err = Native`PrimitiveFunction["Runtime_MTensor_Partition"][hand1, 
								Native`BitCast[array, "MTensor"], Native`BitCast[lenTen, "MTensor"], Native`BitCast[offTen, "MTensor"]];
					If[err =!= Typed[0, "Integer32"],
						Native`ThrowWolframException[Typed[Native`ErrorCode["PartitionError"], "Integer32"]]];
					Native`Load[hand]
				]
			]];


		env["declareFunction", Native`PrimitiveFunction["Runtime_MTensor_Reverse"], 
						MetaData[
							<|"Linkage" -> "Runtime"|>
							]@TypeSpecifier[ 
								{"Handle"["MTensor"], "MTensor"} -> "ErrorCode"]];

		env["declareFunction", Reverse,
			inline@Typed[
				TypeForAll[ {"elem", "rank"}, {"PackedArray"["elem", "rank"]} ->  "PackedArray"["elem", "rank"]]
			]@Function[{array},
				Module[{hand = Native`Handle[], hand1, err},
					hand1 = Native`BitCast[ hand, "Handle"["MTensor"]];
					Native`Store[hand1, Compile`NullReference];
					err = Native`PrimitiveFunction["Runtime_MTensor_Reverse"][hand1, Native`BitCast[array, "MTensor"]];
					If[err =!= Typed[0, "Integer32"],
						Native`ThrowWolframException[Typed[Native`ErrorCode["ReverseError"], "Integer32"]]];
					Native`Load[hand]
				]
			]];

		env["declareFunction", Native`PrimitiveFunction["Runtime_MTensor_Flatten"], 
						MetaData[
							<|"Linkage" -> "Runtime"|>
							]@TypeSpecifier[ 
								{"Handle"["MTensor"], "MTensor", "MachineInteger"} -> "ErrorCode"]];

		env["declareFunction", Flatten,
			inline@Typed[
				TypeForAll[ {"elem"}, {"PackedArray"["elem", 1]} ->  "PackedArray"["elem", 1]]
			]@Function[{array},
				array
			]];


		env["declareFunction", Flatten,
			inline@Typed[
				TypeForAll[ {"elem", "rank"}, 
						{TypePredicate[ TypeEvaluate[Greater, {"rank", 1}], TrueQ]}, 
							{"PackedArray"["elem", "rank"]} ->  "PackedArray"["elem", 1]]
			]@Function[{array},
				Module[{hand = Native`Handle[], hand1, err, n},
					n = ArrayDepth[array]-1;
					hand1 = Native`BitCast[ hand, "Handle"["MTensor"]];
					Native`Store[hand1, Compile`NullReference];
					err = Native`PrimitiveFunction["Runtime_MTensor_Flatten"][hand1, Native`BitCast[array, "MTensor"], n];
					If[err =!= Typed[0, "Integer32"],
						Native`ThrowWolframException[Typed[Native`ErrorCode["FlattenError"], "Integer32"]]];
					Native`Load[hand]
				]
			]];


		env["declareFunction", Native`PrimitiveFunction["Runtime_MTensor_Rotate"], 
						MetaData[
							<|"Linkage" -> "Runtime"|>
							]@TypeSpecifier[ 
								{"Handle"["MTensor"], "MTensor", "MachineInteger", "CArray"["MachineInteger"]} -> "ErrorCode"]];

		env["declareFunction", RotateLeft,
			inline@Typed[
				TypeForAll[ {"elem", "rank"}, 
							{"PackedArray"["elem", "rank"], "MachineInteger"} ->  "PackedArray"["elem", "rank"]]
			]@Function[{array, rot},
				Module[{hand = Native`Handle[], hand1, err, rots = Native`StackArray[1]},
					rots[[0]] = rot;
					hand1 = Native`BitCast[ hand, "Handle"["MTensor"]];
					Native`Store[hand1, Compile`NullReference];
					err = Native`PrimitiveFunction["Runtime_MTensor_Rotate"][hand1, Native`BitCast[array, "MTensor"], 1, rots];
					If[err =!= Typed[0, "Integer32"],
						Native`ThrowWolframException[Typed[Native`ErrorCode["RotateError"], "Integer32"]]];
					Native`Load[hand]
				]
			]];

		env["declareFunction", RotateLeft,
			inline@Typed[
				TypeForAll[ {"elem", "rank"}, 
							{"PackedArray"["elem", "rank"], "PackedArray"["elem", 1]} ->  "PackedArray"["elem", "rank"]]
			]@Function[{array, rotArr},
				Module[{hand = Native`Handle[], hand1, err, rotLen, rotData = Native`ArrayData[rotArr]},
					rotLen = Native`ArrayDimensions[rotArr][[0]];
					hand1 = Native`BitCast[ hand, "Handle"["MTensor"]];
					Native`Store[hand1, Compile`NullReference];
					err = Native`PrimitiveFunction["Runtime_MTensor_Rotate"][hand1, Native`BitCast[array, "MTensor"], rotLen, rotData];
					If[err =!= Typed[0, "Integer32"],
						Native`ThrowWolframException[Typed[Native`ErrorCode["RotateError"], "Integer32"]]];
					Native`Load[hand]
				]
			]];

		env["declareFunction", RotateRight,
			inline@Typed[
				TypeForAll[ {"elem", "rank"}, 
							{"PackedArray"["elem", "rank"], "MachineInteger"} ->  "PackedArray"["elem", "rank"]]
			]@Function[{array, rot},
				Module[{hand = Native`Handle[], hand1, err, rots = Native`StackArray[1]},
					rots[[0]] = -rot;
					hand1 = Native`BitCast[ hand, "Handle"["MTensor"]];
					Native`Store[hand1, Compile`NullReference];
					err = Native`PrimitiveFunction["Runtime_MTensor_Rotate"][hand1, Native`BitCast[array, "MTensor"], 1, rots];
					If[err =!= Typed[0, "Integer32"],
						Native`ThrowWolframException[Typed[Native`ErrorCode["RotateError"], "Integer32"]]];
					Native`Load[hand]
				]
			]];

		env["declareFunction", RotateRight,
			inline@Typed[
				TypeForAll[ {"elem", "rank"}, 
							{"PackedArray"["elem", "rank"], "PackedArray"["elem", 1]} ->  "PackedArray"["elem", "rank"]]
			]@Function[{array, rotArr},
				Module[{hand = Native`Handle[], hand1, err, rotLen, rotData = Native`ArrayData[-rotArr]},
					rotLen = Native`ArrayDimensions[rotArr][[0]];
					hand1 = Native`BitCast[ hand, "Handle"["MTensor"]];
					Native`Store[hand1, Compile`NullReference];
					err = Native`PrimitiveFunction["Runtime_MTensor_Rotate"][hand1, Native`BitCast[array, "MTensor"], rotLen, rotData];
					If[err =!= Typed[0, "Integer32"],
						Native`ThrowWolframException[Typed[Native`ErrorCode["RotateError"], "Integer32"]]];
					Native`Load[hand]
				]
			]];

		env["declareFunction", Native`PrimitiveFunction["Runtime_MTensor_12TransposeConjugate"], 
						MetaData[
							<|"Linkage" -> "Runtime"|>
							]@TypeSpecifier[ 
								{"Handle"["MTensor"], "MTensor", "Integer32"} -> "ErrorCode"]];

		env["declareFunction", Transpose,
			inline@Typed[
				TypeForAll[ {"elem", "rank"}, 
							{TypePredicate[ TypeEvaluate[Greater, {"rank", 1}], TrueQ]}, 
							{"PackedArray"["elem", "rank"]} ->  "PackedArray"["elem", "rank"]]
			]@Function[{array},
				Module[{hand = Native`Handle[], hand1, err},
					hand1 = Native`BitCast[ hand, "Handle"["MTensor"]];
					Native`Store[hand1, Compile`NullReference];
					err = Native`PrimitiveFunction["Runtime_MTensor_12TransposeConjugate"][hand1, Native`BitCast[array, "MTensor"], Typed[0, "Integer32"]];
					If[err =!= Typed[0, "Integer32"],
						Native`ThrowWolframException[Typed[Native`ErrorCode["TransposeError"], "Integer32"]]];
					Native`Load[hand]
				]
			]];

		env["declareFunction", ConjugateTranspose,
			inline@Typed[
				TypeForAll[ {"elem", "rank"}, 
							{TypePredicate[ TypeEvaluate[Greater, {"rank", 1}], TrueQ]}, 
							{"PackedArray"["elem", "rank"]} ->  "PackedArray"["elem", "rank"]]
			]@Function[{array},
				Module[{hand = Native`Handle[], hand1, err},
					hand1 = Native`BitCast[ hand, "Handle"["MTensor"]];
					Native`Store[hand1, Compile`NullReference];
					err = Native`PrimitiveFunction["Runtime_MTensor_12TransposeConjugate"][hand1, Native`BitCast[array, "MTensor"], Typed[1, "Integer32"]];
					If[err =!= Typed[0, "Integer32"],
						Native`ThrowWolframException[Typed[Native`ErrorCode["TransposeError"], "Integer32"]]];
					Native`Load[hand]
				]
			]];


		env["declareFunction", Native`PrimitiveFunction["Runtime_MTensor_Sort"], 
						MetaData[
							<|"Linkage" -> "Runtime"|>
							]@TypeSpecifier[ 
								{"Handle"["MTensor"], "MTensor"} -> "ErrorCode"]];


		env["declareFunction", Sort,
			inline@Typed[
				TypeForAll[ {"elem", "rank"}, 
							{"PackedArray"["elem", "rank"]} ->  "PackedArray"["elem", "rank"]]
			]@Function[{array},
				Module[{hand = Native`Handle[], hand1, err},
					hand1 = Native`BitCast[ hand, "Handle"["MTensor"]];
					Native`Store[hand1, Compile`NullReference];
					err = Native`PrimitiveFunction["Runtime_MTensor_Sort"][hand1, Native`BitCast[array, "MTensor"]];
					If[err =!= Typed[0, "Integer32"],
						Native`ThrowWolframException[Typed[Native`ErrorCode["SortError"], "Integer32"]]];
					Native`Load[hand]
				]
			]];

		env["declareFunction", Native`PrimitiveFunction["Runtime_MTensor_computeSortPermutation"], 
						MetaData[
							<|"Linkage" -> "Runtime"|>
							]@TypeSpecifier[ 
								{"MTensor", "MTensor"} -> "MBool"]];


		env["declareFunction", Ordering,
			inline@Typed[
				TypeForAll[ {"elem", "rank"}, 
							{"PackedArray"["elem", "rank"]} ->  "PackedArray"["MachineInteger", 1]]
			]@Function[{array},
				Module[{ordering, sorted},
					ordering = Native`CreatePackedArray[Length[array], Typed[0, "MachineInteger"]];
					sorted = Native`PrimitiveFunction["Runtime_MTensor_computeSortPermutation"][Native`BitCast[ordering, "MTensor"], Native`BitCast[array, "MTensor"]];
					ordering
				]
			]];

		env["declareFunction", Dimensions,
			inline@Typed[
				TypeForAll[ {"elem", "rank"}, 
							{"PackedArray"["elem", "rank"]} ->  "PackedArray"["MachineInteger", 1]]
			]@Function[{array},
				Module[{dims = Native`ArrayDimensions[array], rank = TensorRank[array], paOut, data},
					paOut = Native`CreatePackedArray[rank, Typed[0, "MachineInteger"]];
					data = Native`ArrayData[paOut];
					Native`CopyTo[dims, data, rank];
					paOut
				]
			]];


	]

] (* StaticAnalysisIgnore *)

setupMacros[st_] :=
	Module[ {env = st["macroEnvironment"]},

		RegisterMacro[env, RotateLeft,
			RotateLeft[array_] -> 
					RotateLeft[ array, 1]
		];
		
		RegisterMacro[env, RotateRight,
			RotateRight[array_] -> 
					RotateRight[ array, 1]
		];


	];



RegisterCallback["SetupTypeSystem", setupTypes]
RegisterCallback["SetupMacros", setupMacros]

End[]

EndPackage[]
