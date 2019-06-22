(* :Title: ArgumentTests.m *)

(* :Author:
        Todd Gayley
        tgayley@wolfram.com
*)

(* :Package Version: 4.9 *)

(* :Mathematica Version: 4.0 *)
		     
(* :Copyright: J/Link source code (c) 1999-2019, Wolfram Research, Inc. All rights reserved.

   Use is governed by the terms of the J/Link license agreement, which can be found at
   www.wolfram.com/solutions/mathlink/jlink.
*)

(* :Discussion:
   Predicates used for argument pattern tests when creating function defs for calls into Java.
	
   This file is a component of the J/Link Mathematica source code.
   It is not a public API, and should never be loaded directly by users or programmers.

   J/Link uses a special system wherein one package context (JLink`) has its implementation
   split among a number of .m files. Each component file has its own private context, and also
   potentially introduces public symbols (in the JLink` context) and so-called "package" symbols,
   where the term "package" comes from Java terminology, referring to symbols that are visible
   everywhere within the implementation of J/Link, but not to clients.
*)


AllowRaggedArrays::usage =
"AllowRaggedArrays[True] lets you pass ragged (i.e., non-rectangular) arrays to Java. For example, a method that takes int[][] could be passed {{1,2},{3}}. AllowRaggedArrays[True] can drastically reduce the speed with which large arrays are passed back and forth between the Wolfram Language and Java. Call AllowRaggedArrays[False] to restore the default behavior."

$RelaxedTypeChecking::usage =
"$RelaxedTypeChecking is a flag that can be set to True to speed up the validation performed in the Wolfram Language (via pattern tests) on arrays of data being sent as arguments to Java calls. For a very large matrix, it can be expensive to test that it is, say, a rectangular matrix of integers before it is sent to Java. The speed is gained by making the tests much less strict, so you must make sure that you pass methods exactly the arguments they expect. The default value is False. You can set and reset the value whenever you want; a typical use would be Block[{$RelaxedTypeChecking = True}, callToJava[largeMatrix]]."


Begin["`Package`"]

(* The isXXX functions are predicates used in function definitions for Java methods created during LoadJavaClass.

   These could be coalesced by parameterizing them like isRealArray[#, n]&. There is less motivation to do this now that
   they are not public.
   The comments below are the usage messages these symbols had when they were public.
*)
isRealList
(* "isRealList is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a list of real numbers, a reference to a 1-D Java array of real numbers, or Null." *)
isRealArray2
(* "isRealArray2 is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a depth 2 list of real numbers (e.g. {{1., 2.}, {3., 4.}}), a reference to a 2-D Java array of real numbers, or Null. The array must be rectangular (i.e., satisfying MatrixQ) unless you have called AllowRaggedArrays[True]." *)
isRealArray3
(* "isRealArray3 is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a depth 3 list of real numbers (e.g. {{{1., 2.}, {3., 4.}}, {{5., 6.}, {7., 8.}}}), a reference to a 3-D Java array of real numbers, or Null. The array must be rectangular unless you have called AllowRaggedArrays[True]." *)
isNumberList
(* "isNumberList is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a list of mixed real numbers or integers, a reference to a 1-D Java array of integers or real numbers, or Null." *)
isNumberArray2
(* "isNumberArray2 is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a depth 2 list of real numbers or integers or both (e.g. {{1, 2}, {3., 4}}), a reference to a 2-D Java array of integers or real numbers, or Null. The array must be rectangular (i.e., satisfying MatrixQ) unless you have called AllowRaggedArrays[True]." *)
isNumberArray3
(* "isNumberArray3 is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a depth 3 list of real numbers or integers or both (e.g. {{{1, 2}, {3., 4}}, {{5, 6}, {7, 8}}}), a reference to a 3-D Java array of integers or real numbers, or Null. The array must be rectangular unless you have called AllowRaggedArrays[True]." *)
isIntegerList
(* "isIntegerList is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a list of integers, a reference to a 1-D Java array of integer type, or Null." *)
isIntegerArray2
(* "isIntegerArray2 is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a depth 2 list of integers (e.g. {{1, 2}, {3, 4}}), a reference to a 2-D Java array of integer type, or Null. The array must be rectangular (i.e., satisfying MatrixQ) unless you have called AllowRaggedArrays[True]." *)
isIntegerArray3
(* "isIntegerArray3 is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a depth 3 list of integers (e.g. {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}), a reference to a 3-D Java array of integer type, or Null. The array must be rectangular unless you have called AllowRaggedArrays[True]." *)
isString
(* "isString is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a string, a reference to a Java string object, or Null." *)
isStringList
(* "isStringList is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a list of strings, a reference to a 1-D Java array of strings, or Null." *)
isStringArray2
(* "isStringArray2 is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a depth 2 list of strings (e.g. {{\"a\", \"b\"}, {\"c\", \"d\"}}), a reference to a 2-D Java array of strings, or Null. The array need not be rectangular (that is, it need not satisfy MatrixQ)." *)
isStringArray3
(* "isStringArray3 is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a depth 3 list of strings (e.g. {{{\"a\", \"b\"}, {\"c\", \"d\"}}, {{\"e\", \"f\"}, {\"g\", \"h\"}}}), a reference to a 3-D Java array of strings, or Null. The array need not be rectangular." *)
isTrueFalseList
(* "isTrueFalseList is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a list of True or False values, a reference to a 1-D Java array of booleans, or Null." *)
isTrueFalseArray2
(* "isTrueFalseArray2 is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a depth 2 list of True or False values (e.g. {{True, False}, {False, True}}), a reference to a 2-D Java array of booleans, or Null. The array need not be rectangular (that is, it need not satisfy MatrixQ)." *)
isTrueFalseArray3
(* "isTrueFalseArray3 is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a depth 3 list of True or False values (e.g. {{{True, False}, {False, True}}, {{True, False}, {False, True}}}), a reference to a 3-D Java array of booleans, or Null. The array need not be rectangular." *)
isComplex
(* "isComplex is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument has head Complex, or is a reference to a Java object of the class designated with SetComplexClass, or is Null." *)
isComplexList
(* "isComplexList is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a list of complex numbers, a reference to a 1-D Java array of objects of the class designated with SetComplexClass, or Null." *)
isComplexArray2
(* "isComplexArray2 is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a depth 2 list of complex numbers (e.g. {{1 + I, 2 + I}, {3 + I, 4 + I}}), a reference to a 2-D Java array of objects of the class designated with SetComplexClass, or Null. The array need not be rectangular (that is, it need not satisfy MatrixQ)." *)
isComplexArray3
(* "isComplexArray3 is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a depth 3 list of complex numbers (e.g. {{{1 + I, 2 + I}, {3 + I, 4 + I}}, {{5 + I, 6 + I}, {7 + I, 8 + I}}}), a reference to a 3-D Java array of objects of the class designated with SetComplexClass, or Null. The array need not be rectangular." *)
isObjectList
(* "isObjectList is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a list of java objects, a reference to a 1-D Java array of objects, or Null." *)
isObjectArray2
(* "isObjectArray2 is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a depth 2 list of java objects, a reference to a 2-D Java array of objects, or Null. The array need not be rectangular (that is, it need not satisfy MatrixQ)." *)
isObjectArray3
(* "isObjectArray3 is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a depth 3 list of java objects, a reference to a 3-D Java array of objects, or Null. The array need not be rectangular." *)
isBigInteger
(* "isBigInteger is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is an integer, a reference to a Java BigInteger object, or Null." *)
isBigIntegerList
(* "isBigIntegerList is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a list of integers, a reference to a 1-D Java array of BigInteger, or Null." *)
isBigIntegerArray2
(* "isBigIntegerArray2 is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a depth 2 list of integers (e.g. {{1, 2}, {3, 4}}), a reference to a 2-D Java array of BigInteger, or Null. The array need not be rectangular (that is, it need not satisfy MatrixQ)." *)
isBigIntegerArray3
(* "isBigIntegerArray3 is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a depth 3 list of integers (e.g. {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}), a reference to a 3-D Java array of BigInteger, or Null. The array need not be rectangular." *)
isBigDecimal
(* "isBigDecimal is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is an integer, real number, a reference to a Java BigDecimal object, or Null." *)
isBigDecimalList
(* "isBigDecimalList is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a list of integers or reals or both, a reference to a 1-D Java array of BigDecimal, or Null." *)
isBigDecimalArray2
(* "isBigDecimalArray2 is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a depth 2 list of integers or reals or both (e.g. {{1, 2}, {3., 4.}}), a reference to a 2-D Java array of BigDecimal, or Null. The array need not be rectangular (that is, it need not satisfy MatrixQ)." *)
isBigDecimalArray3
(* "isBigDecimalArray3 is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a depth 3 list of integers or reals or both (e.g. {{{1, 2}, {3., 4.}}, {{5, 6}, {7, 8}}}), a reference to a 3-D Java array of BigDecimal, or Null. The array need not be rectangular." *)
isExprList
(* "isExprList is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a list of anything, or a reference to a 1-D Java array of Expr, or Null." *)
isExprArray2
(* "isExprArray2 is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a list of lists, a reference to a 2-D Java array of Expr, or Null. The array need not be rectangular (that is, it need not satisfy MatrixQ)." *)
isExprArray3
(* "isExprArray3 is a predicate used in function definitions for Java methods created during LoadJavaClass. It returns True if its argument is a list of lists down to at least 3 levels deep, a reference to a 3-D Java array of Expr, or Null. The array need not be rectangular." *)
isJavaIntegerArray
(* "isJavaIntegerArray[expr, depth] returns True if expr is Null or a reference to a java object that is an array of integer type of the specified depth, False otherwise." *)
isJavaRealArray
(* "isJavaRealArray[expr, depth] returns True if expr is Null or a reference to a java object that is an array of floating-point type of the specified depth, False otherwise." *)
isJavaStringArray
(* "isJavaStringArray[expr, depth] returns True if expr is Null or a reference to a java object that is an array of strings of the specified depth, False otherwise." *)
isJavaBooleanArray
(* "isJavaBooleanArray[expr, depth] returns True if expr is Null or a reference to a java object that is an array of booleans of the specified depth, False otherwise." *)
isJavaObjectArray
(* "isJavaObjectArray[expr, depth] returns True if expr is Null or a reference to a java object that is an array of objects of the specified depth, False otherwise." *)
isJavaExprArray
(* "isJavaExprArray[expr, depth] returns True if expr is Null or a reference to a java object that is an array of Expr objects of the specified depth, False otherwise." *)
isJavaComplexArray
(* "isJavaComplexArray[expr, depth] returns True if expr is Null or a reference to a java object that is an array of the specified depth of the class specified by SetComplexClass, False otherwise." *)
isJavaBigDecimalArray
(* "isJavaBigDecimalArray[expr, depth] returns True if expr is Null or a reference to a java object that is an array of BigDecimal of the specified depth, False otherwise." *)
isJavaBigIntegerArray
(* "isJavaBigIntegerArray[expr, depth] returns True if expr is Null or a reference to a java object that is an array of BigInteger of the specified depth, False otherwise." *)

$allowRaggedArrays

End[]  (* `Package` *)


(* Current context will be JLink`. *)

Begin["`ArgumentTests`Private`"]

(* The $RelaxedTypeChecking state is a global shared by all running JVMs. *)

If[!ValueQ[$RelaxedTypeChecking],
	$RelaxedTypeChecking = False
]

(*
	The AllowRaggedArrays state is deemed too little-used to try to accommodate different
	states for different JVMs. Calling it for one JVM will allow ragged arrays to be passed
	to any JVM, which will cause errors because those JVMs have not had jAllowRaggedArrays
	called on them. We choose to ignore this obscure bug, as J/Link will most likely
	switch to the .NET/Link technique of not validating args on the Mathematica side anyway.
*)

AllowRaggedArrays[allow:(True | False)] := AllowRaggedArrays[GetJVM[InstallJava[]], allow]

AllowRaggedArrays[jvm_JVM, allow:(True | False)] := (jAllowRaggedArrays[jvm, allow]; $allowRaggedArrays = allow;)

If[!ValueQ[$allowRaggedArrays],
	$allowRaggedArrays = False
]


(* The logic of the Integer, Real, and Number forms of these tests hinges on the fact that MatrixQ[x, NumberQ] is very
   fast. There is no need to avoid that test even with $RelaxedTypeChecking. Thus, the tests get a bit complicated
   as a result of the desire to keep this extra type checking in.
*)

isIntegerList[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		ListQ[x],
	(* else *)
		MatchQ[x, {___Integer}]
	] || TrueQ[isJavaIntegerArray[x, 1]]

isIntegerArray2[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		($allowRaggedArrays && ListQ[x]) || MatrixQ[x, NumberQ],
	(* else *)
		($allowRaggedArrays || MatrixQ[x]) && MatchQ[x, {{___Integer}..}] 
	] || TrueQ[isJavaIntegerArray[x, 2]]
	
isIntegerArray3[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		ListQ[x],
	(* else *)
		($allowRaggedArrays || ArrayDepth[x] == 3) && VectorQ[x, MatchQ[#, {{___Integer}..}]&]
	] || TrueQ[isJavaIntegerArray[x, 3]]
	
isRealList[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		ListQ[x],
	(* else *)
		MatchQ[x, {___Real}]
	] || TrueQ[isJavaRealArray[x, 1]]

isRealArray2[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		($allowRaggedArrays && ListQ[x]) || MatrixQ[x, NumberQ],
	(* else *)
		($allowRaggedArrays || MatrixQ[x]) && MatchQ[x, {{___Real}..}] 
	] || TrueQ[isJavaRealArray[x, 2]]
	
isRealArray3[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		ListQ[x],
	(* else *)
		($allowRaggedArrays || ArrayDepth[x] == 3) && VectorQ[x, MatchQ[#, {{___Real}..}]&]
	] || TrueQ[isJavaRealArray[x, 3]]

isNumberList[x_] :=
	VectorQ[x, NumberQ] || TrueQ[isJavaRealArray[x, 1]] || TrueQ[isJavaIntegerArray[x, 1]]

isNumberArray2[x_] :=
	MatrixQ[x, NumberQ] ||
	$allowRaggedArrays && If[TrueQ[$RelaxedTypeChecking], ListQ[x], MatchQ[x, {{(_Real | _Integer)...}..}]] ||
	TrueQ[isJavaRealArray[x, 2]] || TrueQ[isJavaIntegerArray[x, 2]]

isNumberArray3[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		ListQ[x],
	(* else *)
		($allowRaggedArrays && Length[x] > 0 && VectorQ[x, MatchQ[#, {{(_Real | _Integer)...}..}]&]) || (ArrayDepth[x] == 3 && VectorQ[x, MatrixQ[#, NumberQ]&])
	] || TrueQ[isJavaRealArray[x, 3]] || TrueQ[isJavaIntegerArray[x, 3]]

isComplex[x_] :=
	MatchQ[x, _Complex | Null] || JavaObjectQ[x] && GetClass[x] === GetComplexClass[]

isComplexList[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		ListQ[x],
	(* else *)
		MatchQ[x, {___Complex}]
	] || TrueQ[isJavaComplexArray[x, 1]]

isComplexArray2[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		ListQ[x],
	(* else *)
		MatchQ[x, {{___Complex}..}]
	] || TrueQ[isJavaComplexArray[x, 2]]

isComplexArray3[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		ListQ[x],
	(* else *)
		Length[x] > 0 && VectorQ[x, MatchQ[#, {{___Complex}..}]&]
	] || TrueQ[isJavaComplexArray[x, 3]]

isString[x_] :=
	MatchQ[x, _String | Null] || JavaObjectQ[x] && ClassName[x] === "java.lang.String"

isStringList[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		ListQ[x],
	(* else *)
		MatchQ[x, {___String}]
	] || TrueQ[isJavaStringArray[x, 1]]

isStringArray2[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		ListQ[x],
	(* else *)
		MatchQ[x, {{___String}..}]
	] || TrueQ[isJavaStringArray[x, 2]]

isStringArray3[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		ListQ[x],
	(* else *)
		Length[x] > 0 && VectorQ[x, MatchQ[#, {{___String}..}]&]
	] || TrueQ[isJavaStringArray[x, 3]]

isTrueFalseList[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		ListQ[x],
	(* else *)
		MatchQ[x, {(True|False)...}]
	] || TrueQ[isJavaBooleanArray[x, 1]]

isTrueFalseArray2[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		ListQ[x],
	(* else *)
		MatchQ[x, {{(True|False)...}..}]
	] || TrueQ[isJavaBooleanArray[x, 2]]

isTrueFalseArray3[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		ListQ[x],
	(* else *)
		Length[x] > 0 && VectorQ[x, MatchQ[#, {{(True|False)...}..}]&]
	] || TrueQ[isJavaBooleanArray[x, 3]]

isExprList[x_] :=
	ListQ[x] || TrueQ[isJavaExprArray[x, 1]]

isExprArray2[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		ListQ[x],
	(* else *)
		MatchQ[x, {__List}]
	] || TrueQ[isJavaExprArray[x, 2]]

isExprArray3[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		ListQ[x],
	(* else *)
		MatchQ[x, {{__List}..}]
	] || TrueQ[isJavaExprArray[x, 3]]

isBigInteger[x_] :=
	MatchQ[x, _Integer | Null] || JavaObjectQ[x] && (ClassName[x] === "java.math.BigInteger" || InstanceOf[x, "java.math.BigInteger"])

isBigIntegerList[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		ListQ[x],
	(* else *)
		MatchQ[x, {___Integer}]
	] || TrueQ[isJavaBigIntegerArray[x, 1]]

isBigIntegerArray2[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		ListQ[x],
	(* else *)
		MatchQ[x, {{___Integer}..}]
	] || TrueQ[isJavaBigIntegerArray[x, 2]]

isBigIntegerArray3[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		ListQ[x],
	(* else *)
		Length[x] > 0 && VectorQ[x, MatchQ[#, {{___Integer}..}]&]
	] || TrueQ[isJavaBigIntegerArray[x, 3]]

isBigDecimal[x_] :=
	MatchQ[x, _Real | Null] || JavaObjectQ[x] && (ClassName[x] === "java.math.BigDecimal" || InstanceOf[x, "java.math.BigDecimal"])

isBigDecimalList[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		ListQ[x],
	(* else *)
		MatchQ[x, {___Real}]
	] || TrueQ[isJavaBigDecimalArray[x, 1]]

isBigDecimalArray2[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		ListQ[x],
	(* else *)
		MatchQ[x, {{___Real}..}]
	] || TrueQ[isJavaBigDecimalArray[x, 2]]

isBigDecimalArray3[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		ListQ[x],
	(* else *)
		Length[x] > 0 && VectorQ[x, MatchQ[#, {{___Real}..}]&]
	] || TrueQ[isJavaBigDecimalArray[x, 3]]

isObjectList[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		ListQ[x],
	(* else *)
		MatchQ[x, {___?JavaObjectQ}]
	] || TrueQ[isJavaObjectArray[x, 1]]

isObjectArray2[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		ListQ[x],
	(* else *)
		MatchQ[x, {{___?JavaObjectQ}..}]
	] || TrueQ[isJavaObjectArray[x, 2]]

isObjectArray3[x_] :=
	If[TrueQ[$RelaxedTypeChecking],
		ListQ[x],
	(* else *)
		Length[x] > 0 && VectorQ[x, MatchQ[#, {{___?JavaObjectQ}..}]&]
	] || TrueQ[isJavaObjectArray[x, 3]]

(* Null matches any Java object. *)
isJavaIntegerArray[Null, _]    = True
isJavaRealArray[Null, _]       = True
isJavaBooleanArray[Null, _]    = True
isJavaComplexArray[Null, _]    = True
isJavaStringArray[Null, _]     = True
isJavaBigIntegerArray[Null, _] = True
isJavaBigDecimalArray[Null, _] = True
isJavaExprArray[Null, _]       = True
isJavaObjectArray[Null, _]     = True


End[]
