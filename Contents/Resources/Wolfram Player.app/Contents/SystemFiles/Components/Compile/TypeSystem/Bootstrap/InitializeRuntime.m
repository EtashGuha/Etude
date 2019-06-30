
BeginPackage["Compile`TypeSystem`Bootstrap`InitializeRuntime`"]

Begin["`Private`"]

Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)

initializeRuntime[st_] :=
	Module[ {env = st["typeEnvironment"], realList, unaryRealFuns, binaryRealFuns, unaryFuns, binaryFuns, 
				signedList, unsignedList, integerList, complexList, compareFuns, equalFuns, integerWork, funCreated,
				unaryIntegerFuns, unaryRealToIntegerFuns, unaryRealOnly, unaryNotComplexFuns, unaryComplexToBase, binaryIntegerFuns, binaryRealToIntegerFuns},
		realList = {"Real32", "Real64"};
		signedList = { "Integer8", "Integer16", "Integer32", "Integer64"};
		unsignedList = {"Bit", "UnsignedInteger8", "UnsignedInteger16", "UnsignedInteger32","UnsignedInteger64"};
		integerList = Join[ signedList, unsignedList];
		integerList = {"Bit", 
			"Integer8", "UnsignedInteger8", "Integer16", "UnsignedInteger16", "Integer32", "UnsignedInteger32",
			"Integer64", "UnsignedInteger64"};
		complexList = {"Complex"["Real32"], "Complex"["Real64"]};

		unaryRealFuns = {
			"unary_sin", "unary_cos", "unary_tan", 
			"unary_sec", "unary_csc", "unary_cot",
			"unary_asin", "unary_acos", "unary_atan", 
			"unary_asec", "unary_acsc", "unary_acot",
			"unary_sinh", "unary_cosh", "unary_tanh",
			"unary_sech", "unary_csch", "unary_coth",
			"unary_asinh", "unary_acosh", "unary_atanh", 
			"unary_asech", "unary_acsch", "unary_acoth",
			"unary_exp", "unary_expm1", 
			"unary_log", "unary_log1p", "unary_log2", "unary_log10", 
			"unary_sqrt", "unary_rsqrt", 
			"unary_sinc", 
			"unary_gudermannian", "unary_inversegudermannian", 
			"unary_haversine", "unary_inversehaversine"
		};
		binaryRealFuns = {
			"binary_plus", "binary_times", "binary_subtract", "binary_divide", "binary_log",
		 	"binary_atan2"
		};
		binaryFuns = {
			"binary_plus", "binary_times", "binary_subtract", "binary_mod", "binary_pow"
		};
		unaryFuns = {
			"unary_minus", "unary_abs", "unary_fracpart", "unary_intpart", "unary_fibonacci", "unary_lucasl", "unary_gamma", "unary_sign"
		};
		unaryIntegerFuns = {
			"unary_floor", "unary_ceiling", "unary_round",  "unary_nneg", "unary_bit_length", "unary_arg"
		};
		unaryNotComplexFuns = {
			"unary_ramp", "unary_gamma"
		};
		unaryRealToIntegerFuns = {
			"unary_floor", "unary_ceiling", "unary_round", "unary_nneg", "unary_sign", "unary_arg", "unary_intpart"
		};
		unaryRealOnly = {
			"unary_cbrt", "unary_erf", "unary_erfc", "unary_loggamma"
		};
		unaryComplexToBase = {
			"unary_abs", "unary_arg"
		};
		binaryIntegerFuns = {
			"binary_quotient"
		};
		binaryRealToIntegerFuns = {
			"binary_quotient"
		};
		equalFuns = {
			 "binary_sameq", "binary_equal"
		};
		compareFuns = Join[{
			"binary_less", "binary_lessequal", "binary_greater", "binary_greaterequal"
		}, equalFuns];
		
		(* Union together the funs and realFuns, and scan together to prevent declaring
		multiple functions with same name *)
		Scan[
			Scan[
				Function[{fun},
					addRuntimeFunctionA[env, fun, #, 1]],
					Union[ unaryRealFuns, unaryFuns, unaryRealOnly, unaryNotComplexFuns]]&, realList];
		
		(* binary_pow is handled specially later *)
		Scan[
			Scan[
				Function[{fun},
					addRuntimeFunctionA[env, fun, #, 2]],
					Complement[binaryRealFuns ~Union~ binaryFuns, {"binary_pow"}]]&, realList];
		
		Scan[
			Scan[
				Function[{fun},
					addRuntimeFunctionB[env, fun, {#}, "MachineInteger"]],
					unaryRealToIntegerFuns]&, realList];

		Scan[
			Scan[
				Function[{fun},
					addRuntimeFunctionA[env, fun, #, 1]],
					Union[unaryIntegerFuns, unaryFuns, unaryNotComplexFuns]]&, integerList];
		
		Scan[
			Scan[
				Function[{fun},
					addRuntimeFunctionATest[st, env, fun, #, 2]],
					binaryIntegerFuns ~Union~ binaryFuns]&, integerList];

		Scan[
			Scan[
				Function[{fun},
					addRuntimeFunctionB[env, fun, {#, #}, "MachineInteger"]],
					binaryRealToIntegerFuns]&, realList];

		Scan[
			Scan[
				Function[{ty},
					addRuntimeFunctionB[env, "binary_pow", {#, ty}, #]],
					Join[signedList, unsignedList, realList]]&, realList];

		Scan[
			Scan[
				Function[{fun},
					addCompareFunction[env, fun, #, "SignedInteger"]],
					compareFuns]&, signedList];

		Scan[
			Scan[
				Function[{fun},
					addCompareFunction[env, fun, #, "UnsignedInteger"]],
					compareFuns]&, unsignedList];

		Scan[
			Scan[
				Function[{fun},
					addCompareFunction[env, fun, #, "Real"]],
					compareFuns]&, realList];

		Scan[
			Function[{fun},
				addCompareFunction[env, fun, "Boolean", "Boolean"]],
					equalFuns];
		
		
		(*
		 Add TypeJoin Real32, Real64 -> Real64
		*)
		addTypeJoin[ env, "Real32", "Real64","Real64"];
		funCreated = createRuntimeName["cast", {"Real32", "Real64"}];
		addCast[ env, "Real32", "Real64", funCreated];

		(*
		 Add TypeJoin t, Real -> Real
		*)
		Scan[ 
			(
			addTypeJoin[ env, #, "Real32","Real32"];
			funCreated = createRuntimeName["cast", {#, "Real32"}];
			addCast[ env, #, "Real32", funCreated];
			
			addTypeJoin[ env, #, "Real64","Real64"];
			funCreated = createRuntimeName["cast", {#, "Real64"}];
			addCast[ env, #, "Real64", funCreated];
			)&,
			Join[signedList, unsignedList]];
		
		
		(*
		 Add TypeJoin for integers
		*)
		integerWork = Join[signedList, unsignedList];
		integerWork = addTypeJoinCross[ env, integerWork, "UnsignedInteger64"];
		integerWork = addTypeJoinCross[ env, integerWork, "Integer64"];
		integerWork = addTypeJoinCross[ env, integerWork, "UnsignedInteger32"];
		integerWork = addTypeJoinCross[ env, integerWork, "Integer32"];
		integerWork = addTypeJoinCross[ env, integerWork, "UnsignedInteger16"];
		integerWork = addTypeJoinCross[ env, integerWork, "Integer16"];
		integerWork = addTypeJoinCross[ env, integerWork, "UnsignedInteger8"];
		integerWork = addTypeJoinCross[ env, integerWork, "Integer8"];
		
		(*
		 Add Safe Integer Cast
		*)
		
		integerWork = Join[signedList, unsignedList];
		Scan[
			Function[ t1,
				Scan[
					Function[ t2,
						Module[{
							name = createRuntimeName["integer_safe_cast", {t1, t2}],
							ty = TypeSpecifier[{t1} -> t2]},
							env["declareFunction", Native`PrimitiveFunction[name], 
								MetaData[<|"Linkage" -> "Runtime", "Throws"->True|>]@ty];		
							]],
					integerWork]],
				integerWork];
		
		(*
		  Add TypeJoin[ t,t] = t
		*)
		Scan[ 
			addTypeJoin[ env, #, #, #]&,
			Join[signedList, unsignedList, realList]];
		
		(*
		Support for Rational
		*)
		Scan[
			addTypeJoin[ env, #, "Rational"["Integer32"], #]&,
			realList];
		addTypeJoin[ env, "Integer32", "Rational"["Integer32"], "Rational"["Integer32"]];
		addCast[ env, "Rational"["Integer32"], "Real64", "RationalCast"];

		Scan[
			addTypeJoin[ env, #, "Rational"["Integer64"], #]&,
			realList];
		addTypeJoin[ env, "Integer64", "Rational"["Integer64"], "Rational"["Integer64"]];
		addCast[ env, "Rational"["Integer64"], "Real64", "RationalCast"];



		Scan[
			addRuntimeFunction[env, "to_string", {#}, "CArray"["UnsignedInteger8"]]&
			,
			Join[signedList, unsignedList, realList]];
		
		(*
		Support for Complex,  casts to Complex[ Real64] are implemented by
		casting to Real64 and then to the complex.
		*)
		Scan[
			(addTypeJoin[env, #, "Complex"["Real64"], "Complex"["Real64"]])&,
			complexList];		
		addTypeJoin[env, "Complex"["Real32"], "Complex"["Real32"], "Complex"["Real32"]];
			
		Scan[
			(addTypeJoin[env, #, "Complex"["Real64"], "Complex"["Real64"]];
			addComplexReal64Cast[env,#])&,
			Join[signedList, unsignedList, realList]];
			
		Module[ {fun = createRuntimeName["cast", {"Real64", "Complex"["Real64"]}]},	
			addCast[ env, "Real64", "Complex"["Real64"], fun]];
		
		addTypeJoinComplex[env, "Real64", "Complex"["Integer32"], "Real64"];
		addTypeJoinComplex[env, "Complex"["Real64"], "Complex"["Integer32"], "Real64"];
		addTypeJoinComplex[env, "Integer32", "Complex"["Integer32"], "Integer32"];
		addTypeJoinComplex[env, "Real64", "Complex"["Rational"["Integer32"]], "Real64"];
		addTypeJoinComplex[env, "Complex"["Real64"], "Rational"["Integer32"], "Real64"];
		addTypeJoinComplex[env, "Complex"["Real64"], "Complex"["Rational"["Integer32"]], "Real64"];

		addTypeJoinComplex[env, "Real64", "Complex"["Integer64"], "Real64"];
		addTypeJoinComplex[env, "Complex"["Real64"], "Complex"["Integer64"], "Real64"];
		addTypeJoinComplex[env, "Integer64", "Complex"["Integer64"], "Integer64"];
		addTypeJoinComplex[env, "Real64", "Complex"["Rational"["Integer64"]], "Real64"];
		addTypeJoinComplex[env, "Complex"["Real64"], "Rational"["Integer64"], "Real64"];
		addTypeJoinComplex[env, "Complex"["Real64"], "Complex"["Rational"["Integer64"]], "Real64"];
		
		
		Scan[
			Scan[
				Function[{fun},
					addRuntimeFunctionA[env, fun, #, 2]],
					Complement[binaryRealFuns ~Union~ binaryFuns, {"binary_pow"}]]&, complexList];
					
		Scan[
			Scan[
				Function[{ty},
					addRuntimeFunctionBComplex[env, "binary_pow", {#, ty}, #]],
					Join[signedList, unsignedList, realList, complexList]]&, complexList];
					
		Scan[
			Scan[
				Function[{fun},
					addRuntimeFunctionA[env, fun, #, 1]],
					Union[ unaryRealFuns, unaryFuns]]&, complexList];

		Scan[
			Scan[
				Function[{fun},
					addRuntimeFunctionC[env, fun, #, 1, First[#]]],
					unaryComplexToBase]&, complexList];

					
	]

addTypeJoinComplex[ env_, t1_, t2_, tf_] :=
	Module[ {},
		env["declareType", TypeAlias["TypeJoin"[t1, t2], "Complex"[tf]]];
		env["declareType", TypeAlias["TypeJoin"[t2, t1], "Complex"[tf]]];
	]

addTypeJoinCross[ env_, workIn_, common_] :=
	Module[ {work = DeleteCases[workIn, common], fun},
		
		Scan[ 
			(
			fun = createRuntimeName["cast", {#, common}];
			addCast[ env, #, common, fun];
			addTypeJoin[ env, #, common,common];
			)&,
			work];
		work
	]



(*
  Maybe these should be in the runtime?  Or a runtime for inlining.
*)
addCompareFunction[env_, baseName_, arg_, markup_] :=
	Module[ {ty, newName},
		ty = TypeSpecifier[ {arg, arg} -> "Boolean"];
		newName = baseName <> "_" <> markup;
		env["declareFunction", Native`PrimitiveFunction[newName], MetaData[<|"Linkage" -> "LLVMCompareFunction"|>]@ty];
		env["declareFunction", Native`PrimitiveFunction[baseName], MetaData[<|"Redirect" -> Native`PrimitiveFunction[newName]|>]@ty];
	]

addRuntimeFunctionA[ env_, baseName_, arg_, len_] :=
	Module[ {args, runtimeName, ty},
		args = Table[arg, {len}];
		runtimeName = createRuntimeName[ "checked_" <> baseName, args];
		ty = TypeSpecifier[args -> arg];
		env["declareFunction", Native`PrimitiveFunction[runtimeName], MetaData[<|"Linkage" -> "Runtime", "Throws"->True|>]@ty];
		env["declareFunction", Native`PrimitiveFunction[baseName], MetaData[<|"Redirect" -> Native`PrimitiveFunction[runtimeName]|>]@ty];
	]


addRuntimeFunctionA[ env_, baseName_, arg:"Complex"[elem:"Real32"|"Real64"], len:1] :=
	Module[ {args, args1, runtimeName, ty1, ty2},
		args = Table[arg, {len}];
		args1 = Table[ "Real64", {2*len}];
		runtimeName = createRuntimeName[ "checked_" <> baseName, args];
		ty1 = TypeSpecifier[args -> arg];
		ty2 = TypeSpecifier[Join[ {"Handle"[ elem], "Handle"[ elem]}, args1] -> "Void"];
		env["declareFunction", Native`PrimitiveFunction[runtimeName], MetaData[<|"Linkage" -> "Runtime", "Throws"->True|>]@ty2];
		With[ {
			runtimeFun = runtimeName
		},
			env["declareFunction", Native`PrimitiveFunction[baseName], 
				Typed[ty1
				]@Function[ {arg1},
					Module[ {
						re1 = Re[arg1],
						im1 = Im[arg1],
						handRe = Native`Handle[],
						handIm = Native`Handle[],
						ef
					},
						Native`PrimitiveFunction[runtimeFun][handRe, handIm, re1, im1];
						ef = Complex[ Native`Load[handRe], Native`Load[handIm]];
						ef
				]]]
		];
	]


addRuntimeFunctionA[ env_, baseName_, arg:"Complex"[elem:"Real32"|"Real64"], len:2] :=
	Module[ {args, args1, runtimeName, ty1, ty2},
		args = Table[arg, {len}];
		args1 = Table[ "Real64", {2*len}];
		runtimeName = createRuntimeName[ "checked_" <> baseName, args];
		ty1 = TypeSpecifier[args -> arg];
		ty2 = TypeSpecifier[Join[ {"Handle"[ elem], "Handle"[ elem]}, args1] -> "Void"];
		env["declareFunction", Native`PrimitiveFunction[runtimeName], MetaData[<|"Linkage" -> "Runtime", "Throws"->True|>]@ty2];
		With[ {
			runtimeFun = runtimeName
		},
			env["declareFunction", Native`PrimitiveFunction[baseName], 
				Typed[ty1
				]@Function[ {arg1, arg2},
					Module[ {
						re1 = Re[arg1],
						im1 = Im[arg1],
						re2 = Re[arg2],
						im2 = Im[arg2],
						handRe = Native`Handle[],
						handIm = Native`Handle[],
						ef
					},
						Native`PrimitiveFunction[runtimeFun][handRe, handIm, re1, im1, re2, im2];
						ef = Complex[ Native`Load[handRe], Native`Load[handIm]];
						ef
				]]]
		];
	]

(*
  Check the "integerIntrinsics" setting along with the function and type to decide whether 
  to use integer intrinsics for plus/subtract/times on integer 16,32,64 signed and unsigned
*)
addRuntimeFunctionATest[st_, env_, fun_, ty_, len_] :=
	Module[ {intrinsics = Lookup[st, "integerIntrinsics", False]},
		If[ TrueQ[intrinsics] && testIntrinsicFunction[fun] && testIntrinsicType[ty],
			addRuntimeFunctionIntrinsic[env, fun, ty, len],
			addRuntimeFunctionA[env, fun, ty, len]]
	]


testIntrinsicFunction[fun_] :=
	MemberQ[ {"binary_plus", "binary_subtract", "binary_times"}, fun]

testIntrinsicType[ty_] :=
	MemberQ[ {"Integer16", "Integer32", "Integer64", "UnsignedInteger16", "UnsignedInteger32", "UnsignedInteger64"}, ty]


$signedIntrinsics =
<|
	"binary_plus" -> Native`SignedPlusWithOverflow,
	"binary_subtract" -> Native`SignedSubtractWithOverflow,
	"binary_times" -> Native`SignedTimesWithOverflow
|>

$unsignedIntrinsics =
<|
	"binary_plus" -> Native`UnsignedPlusWithOverflow,
	"binary_subtract" -> Native`UnsignedSubtractWithOverflow,
	"binary_times" -> Native`UnsignedTimesWithOverflow
|>

addRuntimeFunctionIntrinsic[env_, baseName_, arg_, 2] :=
	Module[ {ty, targetFun},
		targetFun = If[ MemberQ[ {"Integer16", "Integer32", "Integer64"}, arg], 
						Lookup[$signedIntrinsics, baseName, Null],
						Lookup[$unsignedIntrinsics, baseName, Null]];
		If[targetFun === Null,
			ThrowException[{"Cannot find target", baseName}]];
		ty = TypeSpecifier[{arg, arg} -> arg];
		With[{tF = targetFun},
			env["declareFunction", Native`PrimitiveFunction[baseName], 
				MetaData[<|"Inline" -> "Hint"|>]@Typed[ty]@Function[{x1,x2}, tF[x1,x2]]]];
	]


addRuntimeFunctionB[ env_, baseName_, args_, out_] :=
	Module[ {runtimeName, ty},
		runtimeName = createRuntimeName[ "checked_" <> baseName, args];
		ty = TypeSpecifier[args -> out];
		env["declareFunction", Native`PrimitiveFunction[runtimeName], MetaData[<|"Linkage" -> "Runtime", "Throws"->True|>]@ty];
		env["declareFunction", Native`PrimitiveFunction[baseName], MetaData[<|"Redirect" -> Native`PrimitiveFunction[runtimeName]|>]@ty];
	]



prepareType[ "Complex"[elem_]] :=
	{elem, elem}

prepareType[ elem_] :=
	elem

addRuntimeFunctionBComplex[ env_, baseName_, args_, "Complex"[outElem_]] :=
	Module[ {runtimeName, ty1, tyArgs, func, ty2},
		runtimeName = createRuntimeName[ "checked_" <> baseName, args];
		ty1 = TypeSpecifier[args -> "Complex"[outElem]];
		tyArgs = Flatten[ Map[ prepareType, args]];
		tyArgs = Join[{"Handle"[outElem], "Handle"[outElem]}, tyArgs];
		ty2 = TypeSpecifier[tyArgs -> "Void"];
		Which[
			MatchQ[args, {"Complex"[_], "Complex"[_]}],
				func =
					With[ {runtimeFun = runtimeName},
					Function[ {arg1, arg2},
						Module[ {
							re1 = Re[arg1],
							im1 = Im[arg1],
							re2 = Re[arg2],
							im2 = Im[arg2],
							handRe = Native`Handle[],
							handIm = Native`Handle[],
							ef
						},
							Native`PrimitiveFunction[runtimeFun][handRe, handIm, re1, im1, re2, im2];
							ef = Complex[ Native`Load[handRe], Native`Load[handIm]];
							ef
					]]];
			,
			MatchQ[args, {"Complex"[_], _}],
				func =
					With[ {runtimeFun = runtimeName},Function[ {arg1, arg2},
						Module[ {
							re1 = Re[arg1],
							im1 = Im[arg1],
							handRe = Native`Handle[],
							handIm = Native`Handle[],
							ef
						},
							Native`PrimitiveFunction[runtimeFun][handRe, handIm, re1, im1, arg2];
							ef = Complex[ Native`Load[handRe], Native`Load[handIm]];
							ef
					]]];
			,
			MatchQ[args, {_, "Complex"[_]}],
				func =
					With[ {runtimeFun = runtimeName},
						Function[ {arg1, arg2},
						Module[ {
							re2 = Re[arg2],
							im2 = Im[arg2],
							handRe = Native`Handle[],
							handIm = Native`Handle[],
							ef
						},
							Native`PrimitiveFunction[runtimeFun][handRe, handIm, arg1, re2, im2];
							ef = Complex[ Native`Load[handRe], Native`Load[handIm]];
							ef
					]]];
			,
			True,
				ThrowException[{"Unexpected Complex form", baseName}]];
		env["declareFunction", Native`PrimitiveFunction[baseName], Typed[ty1]@func];
		env["declareFunction", Native`PrimitiveFunction[runtimeName], MetaData[<|"Linkage" -> "Runtime", "Throws"->True|>]@ty2];
	]

addRuntimeFunctionC[ env_, baseName_, arg:"Complex"[elem:"Real32"|"Real64"], len:1, out_] :=
	Module[ {args, args1, runtimeName, ty1, ty2},
		args = Table[arg, {len}];
		args1 = Table[ elem, {2*len}];
		runtimeName = createRuntimeName[ "checked_" <> baseName, args];
		ty1 = TypeSpecifier[args -> out];
		ty2 = TypeSpecifier[args1 -> out];
		env["declareFunction", Native`PrimitiveFunction[runtimeName], MetaData[<|"Linkage" -> "Runtime", "Throws"->True|>]@ty2];
		With[ {
			runtimeFun = runtimeName
		},
			env["declareFunction", Native`PrimitiveFunction[baseName], 
				Typed[ty1
				]@Function[ {arg1},
					Module[ {
						re1 = Re[arg1],
						im1 = Im[arg1]
					},
						Native`PrimitiveFunction[runtimeFun][re1, im1]
				]]]
		];
	]

addRuntimeFunction[ env_, baseName_, args_, out_] :=
	Module[ {runtimeName, ty},
		runtimeName = createRuntimeName[ baseName, args];
		ty = TypeSpecifier[args -> out];
		env["declareFunction", Native`PrimitiveFunction[runtimeName], MetaData[<|"Linkage" -> "Runtime"|>]@ty];
		env["declareFunction", Native`PrimitiveFunction[baseName], MetaData[<|"Redirect" -> Native`PrimitiveFunction[runtimeName]|>]@ty];
	]


convertTypeName[ ty_String] :=
	ty

convertTypeName[ ty1_String[ty2_String]] :=
	ty1 <> ty2

createRuntimeName[ name_, argsIn_List] :=
	Module[{args = Map[convertTypeName[#]&, argsIn]},
		StringJoin[Riffle[ Prepend[args, name], "_"]]
	]


(*
  Implementation of TypeJoin
  
  We have things like
  
	Typed[
		Function[{x, y},
			Module[ {
				z = Native`Handle[],
				arg1 = Compile`Cast[x, Compile`TypeJoin[x, y]],
				arg2 = Compile`Cast[y, Compile`TypeJoin[x, y]]},
				Native`PrimitiveFunction["binary_plus"][z, arg1, arg2];
				Native`Load[z]
		]],
		{"a", "b"} -> "TypeJoin"["a", "b"]]
		
	The function Compile`TypeJoin in the Cast is implemented with the declareFunction
	The "TypeJoin" type signature is implemented with a TypeAlias

*)

(*
Both {t1, t2} and {t2, t1} are handled here
*)
addTypeJoin[ env_, t1_, t2_, tf_] :=
	Module[ {ty},
		env["declareType", TypeAlias["TypeJoin"[t1, t2], tf]];
		env["declareType", TypeAlias["TypeJoin"[t2, t1], tf]];
		ty = TypeSpecifier[ {t1, t2} -> tf];
		env["declareFunction", Compile`TypeJoin, MetaData[<|"Class" -> "Erasure"|>]@ty];
		ty = TypeSpecifier[ {t2, t1} -> tf];
		env["declareFunction", Compile`TypeJoin, MetaData[<|"Class" -> "Erasure"|>]@ty];
	]

(*
The case when all types are the same, e.g., TypeJoin[Real64, Real64] -> Real64
No need to join both {t1, t2} and {t2, t1}
*)
addTypeJoin[ env_, t_, t_, t_] :=
	Module[ {ty},
		env["declareType", TypeAlias["TypeJoin"[t, t], t]];
		ty = TypeSpecifier[ {t, t} -> t];
		env["declareFunction", Compile`TypeJoin, MetaData[<|"Class" -> "Erasure"|>]@ty];
	]



"StaticAnalysisIgnore"[

addCast[ env_, t1_, t2_, fun_] :=
	Module[ {ty},
		ty = TypeSpecifier[ {t1, t2} -> t2];
		env["declareFunction", Compile`Cast, 
				MetaData[<|"Inline" -> "Hint"|>]@Typed[Function[{arg1, arg2}, 
					Native`PrimitiveFunction[fun][arg1]], ty]];
		ty = TypeSpecifier[ {t1} -> t2];
		env["declareFunction", Native`PrimitiveFunction[fun], MetaData[<|"Linkage" -> "LLVMCompileTools"|>]@ty];
	]

] (* StaticAnalysisIgnore *)



(*
  Cast to ComplexReal64 going via Real64.
*)

"StaticAnalysisIgnore"[

addComplexReal64Cast[ env_, t1_] :=
	Module[ {ty},
		ty = TypeSpecifier[ {t1, "Complex"["Real64"]} -> "Complex"["Real64"]];
		env["declareFunction", Compile`Cast, 
				MetaData[<|"Inline" -> "Hint"|>
				]@Typed[
					Function[{arg1, arg2},
						Compile`Cast[ Compile`Cast[arg1, TypeSpecifier["Real64"]], TypeSpecifier["Complex"["Real64"]]]
						], ty]];
	]

] (* StaticAnalysisIgnore *)



RegisterCallback["SetupTypeSystem", initializeRuntime]
RegisterCallback["SetupTypeSystem", initializeIntrinsic]

initializeIntrinsic[st_] :=
	With[{
		env = st["typeEnvironment"], 
		inline = MetaData[<|"Inline" -> "Hint"|>],
		llvmLinkage = MetaData[<|"Linkage" -> "LLVMCompileTools"|>]
	},
		Module[{list1, list2, list},
			list1 = Outer[Append, 
					{{Native`SignedPlusWithOverflow, "SignedPlusWithOverflowIntrinsic"}, 
					 {Native`SignedSubtractWithOverflow, "SignedSubtractWithOverflowIntrinsic"},
					 {Native`SignedTimesWithOverflow, "SignedTimesWithOverflowIntrinsic"}}, 
					 {"Integer16", "Integer32", "Integer64"}, 1];
			list2 = Outer[Append, 
					{{Native`UnsignedPlusWithOverflow, "UnsignedPlusWithOverflowIntrinsic"}, 
					 {Native`UnsignedSubtractWithOverflow, "UnsignedSubtractWithOverflowIntrinsic"},
					 {Native`UnsignedTimesWithOverflow, "UnsignedTimesWithOverflowIntrinsic"}}, 
					 {"UnsignedInteger16", "UnsignedInteger32", "UnsignedInteger64"}, 1];
			list = Join[ Flatten[list1,1], Flatten[list2,1]];
			
			Scan[
				With[{funName = Part[#,1], primName = Part[#,2], ty = Part[#,3]},
					env["declareFunction", Native`PrimitiveFunction[primName], 
						llvmLinkage@TypeSpecifier[{ty, ty} -> "Structure2"[ty, "Boolean"]]
					];
								
					env["declareFunction", funName,
						inline@Typed[{ty, ty} -> ty]@
						Function[{arg1, arg2},
							Module[ {res},
								res = Native`PrimitiveFunction[primName][arg1, arg2];
								If[Native`GetField[res,1],
									Native`ThrowWolframException[Typed[1, "Integer32"]]
								];
								Native`GetField[res,0]
							]
						]
					];
	
				]&,
				list
			]
		]
	]


End[]

EndPackage[]
