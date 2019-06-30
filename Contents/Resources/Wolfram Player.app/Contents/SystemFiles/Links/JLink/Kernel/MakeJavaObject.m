(* :Title: MakeJavaObject.m *)

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
   MakeJavaObject and MakeJavaExpr.
	
   This file is a component of the J/Link Mathematica source code.
   It is not a public API, and should never be loaded directly by users or programmers.

   J/Link uses a special system wherein one package context (JLink`) has its implementation
   split among a number of .m files. Each component file has its own private context, and also
   potentially introduces public symbols (in the JLink` context) and so-called "package" symbols,
   where the term "package" comes from Java terminology, referring to symbols that are visible
   everywhere within the implementation of J/Link, but not to clients.
*)



MakeJavaObject::usage =
"MakeJavaObject[expr] constructs a new Java object whose \"value\" is expr. The expression must be an integer (in which case the created object is of type java.lang.Integer), real (java.lang.Double), String (java.lang.String), or True/False (java.lang.Boolean). It can also be a list or matrix of numbers, Strings, or True/False, in which case the returned object is a Java array of the corresponding primitive type (for example, MakeJavaObject[{1,2,3}] would create an int[] with these values). MakeJavaObject is a shorthand for calling JavaNew. It is typically used when you need to call a Java method that is typed to take Object, and you want to pass it a numeric, String, or array value. You use MakeJavaObject to manually convert such arguments into Java objects before passing them to the method."

MakeJavaExpr::usage =
"MakeJavaExpr[expr] constructs a new Java object of the J/Link Expr class that represents the Wolfram Language expression expr."


Begin["`Package`"]
(* No Package-level exports, but Begin/End are needed by tools. *)
End[]


(* Current context will be JLink`. *)

Begin["`MakeJavaObject`Private`"]


MakeJavaObject::arg =
"MakeJavaObject cannot convert `1` to a Java object. It does not operate on arguments of that type."
MakeJavaObject::empty =
"MakeJavaObject cannot operate on `1` because it has no elements from which to extract type information. Use JavaNew to create empty arrays."

MakeJavaObject[i_?Developer`MachineIntegerQ] :=
    If[-2147483648 <= i <= 2147483647, JavaNew["java.lang.Integer", i], JavaNew["java.lang.Long", i]]
MakeJavaObject[i_Integer] := JavaNew["java.math.BigInteger", ToString[i, FormatType->InputForm, NumberMarks->False]]
MakeJavaObject[jvm_JVM, i_?Developer`MachineIntegerQ] := 
    If[-2147483648 <= i <= 2147483647, JavaNew[jvm, "java.lang.Integer", i], JavaNew[jvm, "java.lang.Long", i]]
MakeJavaObject[jvm_JVM, i_Integer] := JavaNew[jvm, "java.math.BigInteger", ToString[i, FormatType->InputForm, NumberMarks->False]]

MakeJavaObject[x_?Developer`MachineRealQ] := JavaNew["java.lang.Double",x]
MakeJavaObject[x_Real] :=
	(
		LoadJavaClass["com.wolfram.jlink.Utils"];
		ReturnAsJavaObject[com`wolfram`jlink`Utils`bigDecimalFromString[ToString[x, FormatType->InputForm, NumberMarks->False]]]
	)
MakeJavaObject[jvm_JVM, x_?Developer`MachineRealQ] := JavaNew[jvm, "java.lang.Double", x]
MakeJavaObject[jvm_JVM, x_Real] :=
	UseJVM[jvm,
		LoadJavaClass["com.wolfram.jlink.Utils"];
		ReturnAsJavaObject[com`wolfram`jlink`Utils`bigDecimalFromString[ToString[x, FormatType->InputForm, NumberMarks->False]]]
	]

MakeJavaObject[s_String] := JavaNew["java.lang.String", s]
MakeJavaObject[jvm_JVM, s_String] := JavaNew[jvm, "java.lang.String", s]

MakeJavaObject[t:(True | False)] := JavaNew["java.lang.Boolean", t]
MakeJavaObject[jvm_JVM, t:(True | False)] := JavaNew[jvm, "java.lang.Boolean", t]

(* Note that we load the ObjectMaker class with no type checking, and then manually test the argument against various
   array types. This lets us test more efficiently, and prevents the user from
   seeing cryptic argtype errors reported from the private implementation functions (e.g., ObjectMaker`makeIntArray).
*)

MakeJavaObject[a_List] := MakeJavaObject[getDefaultJVM[], a]

MakeJavaObject[jvm_JVM, a_List] :=
	Block[{$RelaxedTypeChecking = False, result},
		UseJVM[jvm,
			(* Just in case user has $RelaxedTypeChecking set to True, we turn it off. Things would break badly if it were True. *)
			LoadJavaClass["com.wolfram.jlink.ObjectMaker", StaticsVisible->False, UseTypeChecking->False];
			result = 
				Which[
					VectorQ[a],
						mjo1[a],
					MatrixQ[a] || $allowRaggedArrays && VectorQ[a, VectorQ],
						mjo2[a],
					ArrayDepth[a] == 3 && VectorQ[a, MatrixQ] || $allowRaggedArrays && VectorQ[a, VectorQ[#, VectorQ]&],
						mjo3[a],
					True,
						Message[MakeJavaObject::arg, a]
				];
			If[JavaObjectQ[result] && result =!= Null, result, $Failed]
		]
	]

MakeJavaObject[obj_?JavaObjectQ] := obj
MakeJavaObject[jvm_JVM, obj_?JavaObjectQ] := obj

MakeJavaObject[a_] := (Message[MakeJavaObject::arg, a]; $Failed)
MakeJavaObject[jvm_JVM, a_] := (Message[MakeJavaObject::arg, a]; $Failed)


SetAttributes[MakeJavaExpr, {HoldAllComplete}];

MakeJavaExpr[e_] := With[{jvm = getDefaultJVM[]}, MakeJavaExpr[jvm, e]]

(* This def lets jvm arg eval; needed because MakeJavaExpr is HoldAllcomplete. *)
MakeJavaExpr[jvmSpec_, e_] := With[{jvm = jvmSpec}, MakeJavaExpr[jvm, e]]

MakeJavaExpr[Null, e_] := (Message[Java::init]; $Failed)

MakeJavaExpr[jvm_JVM, e_] :=
	JavaBlock[
		UseJVM[jvm,
			LoadJavaClass["com.wolfram.jlink.ObjectMaker", StaticsVisible->False, UseTypeChecking->False];
			Switch[HoldComplete[e],
				HoldComplete[_Sequence],
					(* Keep the Sequence head in the Expr. *)
					ReturnAsJavaObject[com`wolfram`jlink`ObjectMaker`makeExpr[HoldComplete[e]]@part[1]],
				HoldComplete[_Unevaluated],
					(* Strip off the Unevaluated head, leaving behind the rest (unevaluated, of course). *)
					ReturnAsJavaObject[com`wolfram`jlink`ObjectMaker`makeExpr[HoldComplete[e]]@part[{1,1}]],
				_,
					ReturnAsJavaObject[com`wolfram`jlink`ObjectMaker`makeExpr[e]]
			]
		]
	]


mjo1[a_] :=
	Which[
		isIntegerList[a],
			ReturnAsJavaObject[com`wolfram`jlink`ObjectMaker`makeIntArray[a]],
		isNumberList[a],
			ReturnAsJavaObject[com`wolfram`jlink`ObjectMaker`makeDoubleArray[a]],
		isStringList[a],
			ReturnAsJavaObject[com`wolfram`jlink`ObjectMaker`makeStringArray[a]],
		isTrueFalseList[a],
			ReturnAsJavaObject[com`wolfram`jlink`ObjectMaker`makeBooleanArray[a]],
		isObjectList[a],
			ReturnAsJavaObject[com`wolfram`jlink`ObjectMaker`makeObjectArray[a]],
		a === {},
			Message[MakeJavaObject::empty, a],
		True,
			Message[MakeJavaObject::arg, a]
	]

mjo2[a_] :=
	Which[
		isIntegerArray2[a],
			ReturnAsJavaObject[com`wolfram`jlink`ObjectMaker`makeIntArray2[a]],
		isNumberArray2[a],
			ReturnAsJavaObject[com`wolfram`jlink`ObjectMaker`makeDoubleArray2[a]],
		isStringArray2[a],
			ReturnAsJavaObject[com`wolfram`jlink`ObjectMaker`makeStringArray2[a]],
		isTrueFalseArray2[a],
			ReturnAsJavaObject[com`wolfram`jlink`ObjectMaker`makeBooleanArray2[a]],
		isObjectArray2[a],
			ReturnAsJavaObject[com`wolfram`jlink`ObjectMaker`makeObjectArray2[a]],
		MatchQ[a, {{}..}],
			Message[MakeJavaObject::empty, a],
		True,
			Message[MakeJavaObject::arg, a]
	]

mjo3[a_] :=
	Which[
		isIntegerArray3[a],
			ReturnAsJavaObject[com`wolfram`jlink`ObjectMaker`makeIntArray3[a]],
		isNumberArray3[a],
			ReturnAsJavaObject[com`wolfram`jlink`ObjectMaker`makeDoubleArray3[a]],
		isStringArray3[a],
			ReturnAsJavaObject[com`wolfram`jlink`ObjectMaker`makeStringArray3[a]],
		isTrueFalseArray3[a],
			ReturnAsJavaObject[com`wolfram`jlink`ObjectMaker`makeBooleanArray3[a]],
		isObjectArray3[a],
			ReturnAsJavaObject[com`wolfram`jlink`ObjectMaker`makeObjectArray3[a]],
		MatchQ[a, {{{}..}..}],
			Message[MakeJavaObject::empty, a],
		True,
			Message[MakeJavaObject::arg, a]
	]


End[]
