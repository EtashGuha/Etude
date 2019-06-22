
BeginPackage["Compile`TypeSystem`Bootstrap`PackedArrayMath`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]



(*
  These should be in some utility
*)
getBase[ "MachineInteger"] :=
	Typed[ 2, "Integer32"]

getBase[ "Real64"] :=
	Typed[ 3, "Integer32"]

getBase[ "Complex"["Real64"]] :=
	Typed[ 4, "Integer32"]



(*
  Unary Math Functions
*)
$unaryFunData =
	{
		{1, Sin, "RealComplex"},
		{2, Cos, "RealComplex"},
		{3, Tan, "RealComplex"},

		{5, Csc, "RealComplex"},
		{6, Sec, "RealComplex"},
		{7, Cot, "RealComplex"},
		
		{9, Sinh, "RealComplex"},
		{10, Cosh, "RealComplex"},
		{11, Tanh, "RealComplex"},

		{13, Csch, "RealComplex"},
		{14, Sech, "RealComplex"},
		{15, Coth, "RealComplex"},

		{17, ArcSin, "RealComplex"},
		{18, ArcCos, "RealComplex"},
		{19, ArcTan, "RealComplex"},

		{21, ArcCsc, "RealComplex"},
		{22, ArcSec, "RealComplex"},
		{23, ArcCot, "RealComplex"},
		
		{25, ArcSinh, "RealComplex"},
		{26, ArcCosh, "RealComplex"},
		{27, ArcTanh, "RealComplex"},

		{29, ArcCsch, "RealComplex"},
		{30, ArcSech, "RealComplex"},
		{31, ArcCoth, "RealComplex"},
		
		{32, Exp, "RealComplex"},
		{33, Internal`Expm1, "RealComplex"},

		{34, Log, "RealComplex"},
		{35, Internal`Log1p, "RealComplex"},
		
		{36, Log2, "RealComplex"},
		{37, Log10, "RealComplex"},
		
		{38, Abs, "IntegerRealComplexToReal"},
		{39, Arg, "IntegerRealToIntegerComplexToReal"},
		{40, Conjugate, "IntegerRealComplex"},
		{41, Im, "IntegerRealComplexToReal"},
		{42, Re, "IntegerRealComplexToReal"},
		{43, Minus, "IntegerRealComplex"},
		
		{44, Sign, "IntegerRealToIntegerComplexToComplex"},
		{49, Round, "IntegerRealToInteger"},
		{50, Floor, "IntegerRealToInteger"},
		{51, Ceiling, "IntegerRealToInteger"},
		{52, FractionalPart, "IntegerRealComplex"},
		{53, IntegerPart, "IntegerRealToInteger"},

		(*{56, Square, "IntegerRealComplex"},*)
		{57, Sqrt, "RealComplex"},
		{58, CubeRoot, "Real"},
		{59, Internal`ReciprocalSqrt, "RealComplex"},
		(*{60, "Reciprocal", "RealComplex"},*)
		(*{61, "BitOp1Arg", "RealComplex"},*)

		{62, BitNot, "Integer"},
		{63, BitLength, "Integer"},
		(*{64, "IntExp1", "RealComplex"},*)
		(*{65, "IntExp2", "RealComplex"},*)
		{66, UnitStep, "IntegerRealToInteger"},
		{67, Sinc, "RealComplex"},
		{68, Fibonacci, "IntegerRealComplex"},
		{69, LucasL, "IntegerRealComplex"},
		{70, Gudermannian, "RealComplex"},
		{71, InverseGudermannian, "RealComplex"},
		{72, Haversine, "RealComplex"},
		{73, InverseHaversine, "RealComplex"},
		{74, Erfc, "Real"},
		{75, Erf, "Real"},
		{76, Gamma, "IntegerReal"},
		{77, LogGamma, "Real"},
		{78, Unitize, "IntegerRealComplexToInteger"},
		(*{79, "Mod1", "RealComplex"},*)

		(*{80, "Logistic", "RealComplex"},*)
		{81, Ramp, "IntegerReal"}(*,
		{82, "AbsSquare", "RealComplex"}*)};



initializeUnaryPackedArrayMath[st_] :=
	Module[ {env = st["typeEnvironment"], unaryRealFuns, unaryRealComplexFuns, unaryIntegerRealComplexFuns, unaryIntegerRealFuns, unaryToInteger, unaryFuns},

		unaryRealFuns = Cases[ $unaryFunData, {_,_,"Real"}];
		Scan[
			Function[ {ty},
				Scan[
					Function[{data},
		   				setupUnaryMathFunctionID[ env, ty, data]], unaryRealFuns]],
		   		{"Real64"}];

		unaryRealComplexFuns = Cases[ $unaryFunData, {_,_,"RealComplex"}];
		Scan[
			Function[ {ty},
				Scan[
					Function[{data},
		   				setupUnaryMathFunctionID[ env, ty, data]], unaryRealComplexFuns]],
		   		{"Real64", "Complex"["Real64"]}];

		unaryIntegerRealComplexFuns = Cases[ $unaryFunData, {_,_,"IntegerRealComplex"}];
		Scan[
			Function[ {ty},
				Scan[
					Function[{data},
		   				setupUnaryMathFunctionID[ env, ty, data]], unaryIntegerRealComplexFuns]],
		   		{"MachineInteger", "Real64", "Complex"["Real64"]}];

		unaryIntegerRealFuns = Cases[ $unaryFunData, {_,_,"IntegerReal"}];
		Scan[
			Function[ {ty},
				Scan[
					Function[{data},
		   				setupUnaryMathFunctionID[ env, ty, data]], unaryIntegerRealFuns]],
		   		{"MachineInteger", "Real64"}];

		unaryToInteger = Cases[ $unaryFunData, {_,_,"Integer"}];
		Scan[
			Function[ {ty},
				Scan[
					Function[{data},
		   				setupUnaryMathFunction[ env, ty, "MachineInteger", data]], unaryToInteger]],
		   		{"MachineInteger"}];

		unaryToInteger = Cases[ $unaryFunData, {_,_,"IntegerRealToInteger"}];
		Scan[
			Function[ {ty},
				Scan[
					Function[{data},
		   				setupUnaryMathFunction[ env, ty, "MachineInteger", data]], unaryToInteger]],
		   		{"MachineInteger", "Real64"}];

		unaryToInteger = Cases[ $unaryFunData, {_,_,"IntegerRealComplexToInteger"}];
		Scan[
			Function[ {ty},
				Scan[
					Function[{data},
		   				setupUnaryMathFunction[ env, ty, "MachineInteger", data]], unaryToInteger]],
		   		{"MachineInteger", "Real64", "Complex"["Real64"]}];

		unaryToInteger = Cases[ $unaryFunData, {_,_,"IntegerRealToIntegerComplexToReal"}];
		Scan[
			Function[{data},
		   				setupUnaryMathFunction[ env, "MachineInteger", "MachineInteger", data]], unaryToInteger];
		Scan[
			Function[{data},
		   				setupUnaryMathFunction[ env, "Real64", "MachineInteger", data]], unaryToInteger];
		Scan[
			Function[{data},
		   				setupUnaryMathFunction[ env, "Complex"["Real64"], "Real64", data]], unaryToInteger];


		unaryToInteger = Cases[ $unaryFunData, {_,_,"IntegerRealToIntegerComplexToComplex"}];
		Scan[
			Function[{data},
		   				setupUnaryMathFunction[ env, "MachineInteger", "MachineInteger", data]], unaryToInteger];
		Scan[
			Function[{data},
		   				setupUnaryMathFunction[ env, "Real64", "MachineInteger", data]], unaryToInteger];
		Scan[
			Function[{data},
		   				setupUnaryMathFunction[ env, "Complex"["Real64"], "Complex"["Real64"], data]], unaryToInteger];


		unaryFuns = Cases[ $unaryFunData, {_,_,"IntegerRealComplexToReal"}];
		Scan[
			Function[{data},
		   				setupUnaryMathFunction[ env, "MachineInteger", "MachineInteger", data]], unaryFuns];
		Scan[
			Function[{data},
		   				setupUnaryMathFunction[ env, "Real64", "Real64", data]], unaryFuns];
		Scan[
			Function[{data},
		   				setupUnaryMathFunction[ env, "Complex"["Real64"], "Real64", data]], unaryFuns];


		env["declareFunction", Native`PrimitiveFunction["MTensorMathUnary"], 
						MetaData[
							<|"Linkage" -> "Runtime"|>
							]@TypeSpecifier[ 
								{"Handle"["MTensor"], "MTensor", "Integer32", "Integer32"} -> "Integer32"]];
		]




setupUnaryMathFunctionID[ env_, baseTy_, {index_, fun_, cl_}] :=
	setupUnaryMathFunction[env, baseTy, baseTy, {index, fun, cl}] 

setupUnaryMathFunction[ env_, inBaseTy_, outBaseTy_, {index_, fun_, _}] :=
	Module[ {base = getBase[ outBaseTy]},
		setupUnaryMathFunction1[env, inBaseTy, outBaseTy, base, Typed[index, "Integer32"], fun]
	]
	
	
"StaticAnalysisIgnore"[

setupUnaryMathFunction1[ env_, inBaseTy_, outBaseTy_, baseOut_, index_, fun_] :=
	env["declareFunction", fun,
			Typed[
				TypeForAll[ {"a"}, {"PackedArray"[inBaseTy, "a"]} -> "PackedArray"[outBaseTy, "a"]]
					]@Function[{arg},
						Module[ {tenIn, tenOut, resHand = Native`Handle[], success},
							tenIn = Native`BitCast[ arg, TypeSpecifier[ "MTensor"]];
							Native`Store[resHand, Compile`NullReference];
							success = Native`PrimitiveFunction["MTensorMathUnary"][resHand, tenIn, index, baseOut];
							If[success =!= Typed[0, "Integer32"],
								Native`ThrowWolframException[Typed[Native`ErrorCode["MathTensorError"], "Integer32"]]];
							tenOut = Native`Load[resHand];
							Native`BitCast[tenOut, Compile`TypeOf[ Compile`ResultOf[fun, arg]]]
						]
					]];

] (* StaticAnalysisIgnore *)



(*
  Binary Math Functions
*)
$binaryFunData =
	{
		{257, Plus, "IntegerRealComplex"},
		{258, Subtract, "IntegerRealComplex"},
		{259, Times, "IntegerRealComplex"},
		{260, Divide, "RealComplex"},
		{261, Mod, "IntegerRealComplex"},
		{262, Quotient, "IntegerRealToInteger"},
		{263, Power, "IntegerRealComplex"},
		{264, Log2, "IntegerRealComplex"},
		{265, ArcTan, "IntegerRealComplex"},
		{266, BitAnd, "Integer"},
		{267, BitOr, "Integer"},
		{268, BitXor, "Integer"},
		(*{269, Chop, "IntegerRealComplex"},
		{270, AbsErr, "IntegerRealComplex"},
		{271, RelErr, "IntegerRealComplex"},
		{272, MaxAbs, "IntegerRealComplex"},
		{273, IntExp2, "IntegerRealComplex"},
		{274, IntLen2, "IntegerRealComplex"},*)
		{275, BitShiftLeft, "Integer"},
		{275, BitShiftRight, "Integer"} (*,
		{276, Unitize2, "IntegerRealComplex"}*)
	};



"StaticAnalysisIgnore"[

initializeBinaryPackedArrayMath[st_] :=
	Module[ {env = st["typeEnvironment"], binaryFuns},

		binaryFuns = Cases[ $binaryFunData, {_,_,"RealComplex"}];
		Scan[
			Function[ {data},
				setupBinaryMathFunction[ env, data]], binaryFuns];
		
		binaryFuns = Cases[ $binaryFunData, {_,_, "IntegerRealComplex"}];
		Scan[
			Function[ {data},
				setupBinaryMathFunction[ env, data]], binaryFuns];
		
		env["declareFunction", Native`PrimitiveFunction["MTensorMathBinary"], 
						MetaData[
							<|"Linkage" -> "Runtime"|>
							]@TypeSpecifier[ 
								{"Handle"["MTensor"], "MTensor", "MTensor","Integer32", "Integer32"} -> "Integer32"]];

	]

] (* StaticAnalysisIgnore *)


(*
  Give these float versions a different type.
*)
setupBinaryMathFunction[ env_, {indexIn_, fun_, "RealComplex"}] :=
	Module[ {index},
		index = Typed[indexIn, "Integer32"];
		setupBinaryMathFunctionScalarTensorFloat[env, index, fun];
		setupBinaryMathFunctionTensorScalarFloat[env, index, fun];
		setupBinaryMathFunctionTensorTensorFloat[env, index, fun];
	]
	
setupBinaryMathFunction[ env_, {indexIn_, fun_, _}] :=
	Module[ {index},
		index = Typed[indexIn, "Integer32"];
		setupBinaryMathFunctionScalarTensor[env, index, fun];
		setupBinaryMathFunctionTensorScalar[env, index, fun];
		setupBinaryMathFunctionTensorTensor[env, index, fun];
	]
	

"StaticAnalysisIgnore"[

setupBinaryMathFunctionScalarTensor[ env_, index_, fun_] :=
	env["declareFunction", fun,
		MetaData[<|"Inline" -> "Hint"|>
			]@Typed[
				TypeForAll[ {"a", "b", "c"}, {Element["a", "Number"]}, {"a", "PackedArray"["b", "c"]} -> 
							"PackedArray"["TypeJoin"["a", "b"], "c"]]
					]@Function[{scalar, pa2},
						Module[ {pa1},
							pa1 = Native`MakeZeroRankTensor[scalar];
							fun[ pa1, pa2]
						]
					]];

setupBinaryMathFunctionScalarTensorFloat[ env_, index_, fun_] :=
	(
	env["declareFunction", fun,
		MetaData[<|"Inline" -> "Hint"|>
			]@Typed[
				TypeForAll[ {"a", "b", "c"}, {Element["a", "Number"]}, {"a", "PackedArray"["b", "c"]} -> 
							"PackedArray"["TypeJoin"["a", "b"], "c"]]
					]@Function[{scalar, pa2},
						Module[ {pa1},
							pa1 = Native`MakeZeroRankTensor[scalar];
							fun[ pa1, pa2]
						]
					]];

(*
  This doesn't reach the Error,  it fails out with multiple alternatives.  It should
  have a better error, but at least it doesn't pass inferencing.
*)
	env["declareFunction", fun,
		TypeSpecifier[TypeForAll[ {"a", "b", "c"}, {Element["a", "BasePackable"], Element["a", "Integral"], Element["b", "Integral"]}, 
						{"a", "PackedArray"["b", "c"]} -> 
							"Error"]]
		];
	)

] (* StaticAnalysisIgnore *)



"StaticAnalysisIgnore"[

setupBinaryMathFunctionTensorScalar[ env_, index_, fun_] :=
	env["declareFunction", fun,
		MetaData[<|"Inline" -> "Hint"|>
			]@Typed[
				TypeForAll[ {"a", "b", "c"}, {Element["c", "Number"]}, {"PackedArray"["a", "b"], "c"} -> 
										"PackedArray"["TypeJoin"["a", "c"], "b"]]
					]@Function[{pa1, scalar},
						Module[ {pa2},
							pa2 = Native`MakeZeroRankTensor[scalar];
							fun[ pa1, pa2]
						]
					]];
					
setupBinaryMathFunctionTensorScalarFloat[ env_, index_, fun_] :=
	(
	env["declareFunction", fun,
		MetaData[<|"Inline" -> "Hint"|>
			]@Typed[
				TypeForAll[ {"a", "b", "c"}, {Element["c", "Number"]}, {"PackedArray"["a", "b"], "c"} -> 
										"PackedArray"["TypeJoin"["a", "c"], "b"]]
					]@Function[{pa1, scalar},
						Module[ {pa2},
							pa2 = Native`MakeZeroRankTensor[scalar];
							fun[ pa1, pa2]
						]
					]];

(*
  This doesn't reach the Error,  it fails out with multiple alternatives.  It should
  have a better error, but at least it doesn't pass inferencing.
*)
	env["declareFunction", fun,
		TypeSpecifier[TypeForAll[ {"a", "b", "c"}, {Element["c", "BasePackable"], Element["a", "Integral"], Element["c", "Integral"]}, {"PackedArray"["a", "b"], "c"} -> 
										"Error"]
					]];
	)

] (* StaticAnalysisIgnore *)


"StaticAnalysisIgnore"[

setupBinaryMathFunctionTensorTensor[ env_, index_, fun_] :=
	env["declareFunction", fun,
					Typed[
						TypeForAll[ {"a", "b", "c", "d"},  {"PackedArray"["a", "b"], "PackedArray"["c", "d"]} -> 
										"PackedArray"["TypeJoin"["a", "c"], TypeEvaluate[Max, {"b", "d"}]]]
					]@Function[{pa1, pa2},
						Module[ {ten1, ten2, resHand = Native`Handle[], tenOut, baseOut1, baseOut2, baseOut, success},
							ten1 = Native`BitCast[ pa1, TypeSpecifier[ "MTensor"]];
							ten2 = Native`BitCast[ pa2, TypeSpecifier[ "MTensor"]];
							baseOut1 = Native`MTensorElementType[pa1];
							baseOut2 = Native`MTensorElementType[pa2];
							baseOut =
							 	If[baseOut1 > baseOut2, baseOut1, baseOut2];	
							Native`Store[resHand, Compile`NullReference];
							success = Native`PrimitiveFunction["MTensorMathBinary"][resHand, ten1, ten2, index, baseOut];
							If[success =!= Typed[0, "Integer32"],
								Native`ThrowWolframException[Typed[Native`ErrorCode["ArrayPartError"], "Integer32"]]];
							tenOut = Native`Load[resHand];
							Native`BitCast[tenOut, Compile`TypeOf[ Compile`ResultOf[fun, pa1, pa2]]]
						]
					]];
					
setupBinaryMathFunctionTensorTensorFloat[ env_, index_, fun_] :=
	(
	env["declareFunction", fun,
					Typed[
						TypeForAll[ {"a", "b", "c", "d"},  {"PackedArray"["a", "b"], "PackedArray"["c", "d"]} -> 
										"PackedArray"["TypeJoin"["a", "c"], TypeEvaluate[Max, {"b", "d"}]]]
					]@Function[{pa1, pa2},
						Module[ {ten1, ten2, resHand = Native`Handle[], tenOut, baseOut1, baseOut2, baseOut, success},
							ten1 = Native`BitCast[ pa1, TypeSpecifier[ "MTensor"]];
							ten2 = Native`BitCast[ pa2, TypeSpecifier[ "MTensor"]];
							baseOut1 = Native`MTensorElementType[pa1];
							baseOut2 = Native`MTensorElementType[pa2];
							baseOut =
							 	If[baseOut1 > baseOut2, baseOut1, baseOut2];	
							Native`Store[resHand, Compile`NullReference];
							success = Native`PrimitiveFunction["MTensorMathBinary"][resHand, ten1, ten2, index, baseOut];
							If[success =!= Typed[0, "Integer32"],
								Native`ThrowWolframException[Typed[Native`ErrorCode["ArrayPartError"], "Integer32"]]];
							tenOut = Native`Load[resHand];
							Native`BitCast[tenOut, Compile`TypeOf[ Compile`ResultOf[fun, pa1, pa2]]]
						]
					]];
(*
  This doesn't reach the Error,  it fails out with multiple alternatives.  It should
  have a better error, but at least it doesn't pass inferencing.
*)
	env["declareFunction", fun,
			TypeSpecifier[TypeForAll[ {"a", "b", "c", "d"},{Element["a", "Integral"], Element["c", "Integral"]},  
							{"PackedArray"["a", "b"], "PackedArray"["c", "d"]} -> "Error"]]];
	)


] (* StaticAnalysisIgnore *)


(*
  Dot
*)

"StaticAnalysisIgnore"[

initializePackedArrayDot[st_] :=
	Module[ {env = st["typeEnvironment"]},	
		
		env["declareFunction", Dot,
					Typed[
						TypeForAll[ {"a", "b"},  {"PackedArray"["a", 1], "PackedArray"["b", 1]} -> 
										"TypeJoin"["a", "b"]]
					]@Function[{paIn1, paIn2},
						Module[ {pa1, pa2, len1, len2, ef = Native`Handle[], success},
							pa1 = Compile`CastElements[paIn1, Compile`TypeOf[Native`TypeJoinBase[paIn1, paIn2]]];
							pa2 = Compile`CastElements[paIn2, Compile`TypeOf[Native`TypeJoinBase[paIn1, paIn2]]];
							len1 = Length[pa1];
							len2 = Length[pa2];
							If[len1 =!= len2,
								Native`ThrowWolframException[Typed[Native`ErrorCode["DotTensorLength"], "Integer32"]]];
							success = PackedArray`VectorDot[ ef, Native`ArrayData[pa1], 1, Native`ArrayData[pa2], 1, len1,Typed[ 0, "Integer32"]];
							If[success =!= Typed[0, "Integer32"],
								Native`ThrowWolframException[Typed[Native`ErrorCode["DotTensorError"], "Integer32"]]];
							Native`Load[ef]
						]
					]];
		declareVectorDot[ env, "VectorDot_I", "MachineInteger"];
		declareVectorDot[ env, "VectorDot_R64", "Real64"];
		declareVectorDotComplex[ env, "VectorDot_CR64", "Complex"[ "Real64"]];		
				
		declareGeneralDot[env];
	]

] (* StaticAnalysisIgnore *)



"StaticAnalysisIgnore"[

declareGeneralDot[env_] :=
	Module[ {func},
		(*
		  We need to specify anything other than rank 1 for both args.  We don't have an Or, so we 
		  specify two functions.  One has arg1 has rank greater than 1 and arg2 has rank anything. 
		  The other has arg1 with rank of 1 and arg2 being greater than 1.
		*)
		func = Function[{paIn1, paIn2},
						Module[ {pa1, pa2, mten1, mten2, rank1, ef = Native`Handle[], success},
							pa1 = Compile`CastElements[paIn1, Compile`TypeOf[Native`TypeJoinBase[paIn1, paIn2]]];
							pa2 = Compile`CastElements[paIn2, Compile`TypeOf[Native`TypeJoinBase[paIn1, paIn2]]];
							rank1 = ArrayDepth[pa1];
							mten1 = Native`BitCast[ pa1, TypeSpecifier[ "MTensor"]];
							mten2 = Native`BitCast[ pa2, TypeSpecifier[ "MTensor"]];
							Native`Store[ef, Compile`NullReference];
							success = PackedArray`GeneralDot[ ef, mten1, mten2, Typed[ 0, "Integer32"]];
							If[success =!= Typed[0, "Integer32"],
								Native`ThrowWolframException[Typed[Native`ErrorCode["DotTensorError"], "Integer32"]]];
							Native`BitCast[ Native`Load[ef], Compile`TypeOf[ Compile`ResultOf[Dot, paIn1, paIn2]]]
						]
					];

		env["declareFunction", Dot,
					Typed[
						TypeForAll[ {"a", "b", "c", "d"}, {TypePredicate[ TypeEvaluate[Greater, {"c", 1}], TrueQ]}, {"PackedArray"["a", "c"], "PackedArray"["b", "d"]} -> 
										"PackedArray"["TypeJoin"["a", "b"], TypeEvaluate[Plus, {"c", "d", -2}]]]
					]@func];
					
		env["declareFunction", Dot,
					Typed[
						TypeForAll[ {"a", "b", "d"}, {TypePredicate[ TypeEvaluate[Greater, {"d", 1}], TrueQ]}, {"PackedArray"["a", 1], "PackedArray"["b", "d"]} -> 
										"PackedArray"["TypeJoin"["a", "b"], TypeEvaluate[Plus, {"d", -1}]]]
					]@func];
		
		env["declareFunction", PackedArray`GeneralDot,
			MetaData[<|"Inline" -> "Hint"|>
				]@Typed[
						TypeSpecifier[ 
							{"Handle"["MTensor"], "MTensor", "MTensor", "Integer32"} -> "Integer32"]
					]@Function[{res, ten1, ten2, flags},
						Native`PrimitiveFunction["MTensorGeneralDot"][ res, ten1, ten2, flags]
					]];
					
		env["declareFunction", Native`PrimitiveFunction["MTensorGeneralDot"], 
				MetaData[
					<|"Linkage" -> "Runtime"|>
					]@TypeSpecifier[ 
						{"Handle"["MTensor"], "MTensor", "MTensor", "Integer32"} -> "Integer32"]];
	]

] (* StaticAnalysisIgnore *)



"StaticAnalysisIgnore"[

declareVectorDot[ env_, fun_, baseTy_] :=
	Module[{},
		env["declareFunction", PackedArray`VectorDot,
			MetaData[<|"Inline" -> "Hint"|>
				]@Typed[
						TypeSpecifier[ 
							{"Handle"[baseTy], "CArray"[baseTy], "MachineInteger", 
									"CArray"[baseTy], "MachineInteger", "MachineInteger", "Integer32"} -> "Integer32"]
					]@Function[{res, arr1, st1, arr2, st2, len, flags},
						Native`PrimitiveFunction[fun][ res, arr1, st1, arr2, st2, len, flags]
					]];
		env["declareFunction", Native`PrimitiveFunction[fun], 
				MetaData[
					<|"Linkage" -> "Runtime"|>
					]@TypeSpecifier[ 
						{"Handle"[baseTy], "CArray"[baseTy], "MachineInteger", 
							"CArray"[baseTy], "MachineInteger", "MachineInteger", "Integer32"} -> "Integer32"]];
	];

(*
 We need to use a void* signature for complex because of problems working directly 
 with the mcomplex type in the RTL, even though it is compatible.
*)	
declareVectorDotComplex[ env_, fun_, baseTy_] :=
	Module[{},
		env["declareFunction", PackedArray`VectorDot,
			MetaData[<|"Inline" -> "Hint"|>
				]@Typed[
						TypeSpecifier[ 
							{"Handle"[baseTy], "CArray"[baseTy], "MachineInteger", 
									"CArray"[baseTy], "MachineInteger", "MachineInteger", "Integer32"} -> "Integer32"]
					]@Function[{res, arr1, st1, arr2, st2, len, flags},
						Module[ {res1, voidR, void1, void2},
							voidR = Native`BitCast[res, TypeSpecifier["VoidHandle"]]; 
							void1 = Native`BitCast[arr1, TypeSpecifier["VoidHandle"]]; 
							void2 = Native`BitCast[arr2, TypeSpecifier["VoidHandle"]]; 
							res1 = Native`PrimitiveFunction[fun][ voidR, void1, st1, void2, st2, len, flags];
							res1
						]
					]];
		env["declareFunction", Native`PrimitiveFunction[fun], 
				MetaData[
					<|"Linkage" -> "Runtime"|>
					]@TypeSpecifier[ 
						{"VoidHandle", "VoidHandle", "MachineInteger", 
							"VoidHandle", "MachineInteger", "MachineInteger", "Integer32"} -> "Integer32"]];
	];


] (* StaticAnalysisIgnore *)



"StaticAnalysisIgnore"[

initializePackedArrayN[st_] :=
	Module[ {env = st["typeEnvironment"]},	
		
		env["declareFunction", N,
					Typed[
						TypeForAll[ {"a"},  {"PackedArray"["MachineInteger", "a"]} -> "PackedArray"["Real64", "a"]]
					]@Function[{paIn},
						Module[ {},
							Compile`CastElements[paIn, TypeSpecifier["Real64"]]
						]
					]];

        env["declareFunction",
            N, 
      		MetaData[<|"Inline" -> "Hint"|>
				]@Typed[TypeForAll[{"a"}, {"PackedArray"["Real64", "a"]} -> "PackedArray"["Real64", "a"]]
      		]@Function[{arg}, arg]];
 
         env["declareFunction",
            N, 
      		MetaData[<|"Inline" -> "Hint"|>
				]@Typed[TypeForAll[{"a"}, {"PackedArray"["Complex"["Real64"], "a"]} -> "PackedArray"["Complex"["Real64"], "a"]]
      		]@Function[{arg}, arg]];
	]

] (* StaticAnalysisIgnore *)



RegisterCallback["SetupTypeSystem", initializeUnaryPackedArrayMath]
RegisterCallback["SetupTypeSystem", initializeBinaryPackedArrayMath]
RegisterCallback["SetupTypeSystem", initializePackedArrayDot]
RegisterCallback["SetupTypeSystem", initializePackedArrayN]



RegisterCallback["InstantiateFunctions", createFunctions]

createFunctions[ state_] :=
	Module[{arrList, mixed},
		arrList = Flatten[ Table[ {"PackedArray"["MachineInteger", i], "PackedArray"["Real64", i], "PackedArray"["Complex"["Real64"], i]}, {i, 1, 2}]];
		mixed = Tuples[arrList, 2];
		Print["Instantiate PackedArray for math functions"];

		Print[ "Plus/Times"];
		Scan[
			Function[ {ty},
				Print["Plus/Times - ", ty];
			   state["create"][ Function[ {Typed[ arg1, ty], Typed[ arg2, "MachineInteger"]}, arg1 + arg2]];
			   state["create"][ Function[ {Typed[ arg1, ty], Typed[ arg2, "MachineInteger"]}, arg2 + arg1]];
			   state["create"][ Function[ {Typed[ arg1, ty], Typed[ arg2, ty]}, arg1 + arg2]];
			   state["create"][ Function[ {Typed[ arg1, ty], Typed[ arg2, ty]}, arg1 * arg2]];
			]
			,
			arrList];

		Print[ "Dot"];
		Scan[
			Function[ {ty},
				Print["Dot - ", ty];
				With[ { ty1 = First[ty], ty2 = Last[ty]},
					state["create"][ Function[ {Typed[ arg1, ty1], Typed[ arg2, ty2]}, arg1.arg2]]];
			]
			,
			mixed];

	]

End[]

EndPackage[]
