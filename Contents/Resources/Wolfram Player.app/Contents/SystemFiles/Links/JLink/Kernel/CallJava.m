(* :Title: CallJava.m *)

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
   Functions related to loading classes and calling from Mathematica into Java.

   This file is a component of the J/Link Mathematica source code.
   It is not a public API, and should never be loaded directly by users or programmers.

   J/Link uses a special system wherein one package context (JLink`) has its implementation
   split among a number of .m files. Each component file has its own private context, and also
   potentially introduces public symbols (in the JLink` context) and so-called "package" symbols,
   where the term "package" comes from Java terminology, referring to symbols that are visible
   everywhere within the implementation of J/Link, but not to clients.
*)


JavaObject::usage =
"JavaObject is used to denote an expression that refers to an object residing in Java."

JavaObjectQ::usage =
"JavaObjectQ[expr] gives True if expr is a reference to a Java object or Null, and gives False otherwise."

JavaClass::usage =
"JavaClass[\"classname\", n] represents a Java class with the specified name. The second argument is an integer index that is not relevant to users. JavaClass expressions cannot be typed in by the user; they are created by LoadJavaClass."

LoadClass::usage =
"LoadClass is deprecated. The new name is LoadJavaClass."

LoadJavaClass::usage =
"LoadJavaClass[\"classname\"] loads the specified class into Java and sets up definitions so that it can be used from the Wolfram Language. You must specify the fully-qualified class name, for example \"java.awt.Frame\". It is safe to call LoadJavaClass multiple times on the same class; it simply returns right away without doing anything."

JavaNew::usage =
"JavaNew[javaclass, args] constructs a Java object of the specified JavaClass, passing the args to its constructor. You can also specify the class name instead of a JavaClass: JavaNew[\"classname\", args]. With this second form, the class will be loaded first if necessary."

MethodFunction::usage =
"MethodFunction is no longer supported. Speed improvements in J/Link have made its benefits minimal."

FieldFunction::usage =
"FieldFunction is no longer supported. Speed improvements in J/Link have made its benefits minimal."

SetField::usage =
"SetField[obj@field, val] sets a value of an object field. It is an alternative syntax to obj@field = val. For static methods, use SetField[staticfield, val] (compared to staticfield = val)."

StaticsVisible::usage =
"StaticsVisible is an option to LoadJavaClass (in J/Link) and LoadNETType (in .NET/Link) that specifies whether the class-specific context in which static method and field definitions are created should be placed on $ContextPath. If you load a class named \"com.foo.Bar\" containing a static method named baz, with the option StaticsVisible->True you could call the method simply with baz[args]. With StaticsVisible->False, you would have to write Bar`baz[args]. The default setting is False, to eliminate the possibility of name conflicts leading to shadowing problems."

AllowShortContext::usage =
"AllowShortContext is an option to LoadJavaClass (in J/Link) and LoadNETType (in .NET/Link) that specifies whether to create a class-specific context in \"short\" form. If you load a class named \"com.foo.Bar\" containing a static method named baz, with the option AllowShortContext->True you could call the method with Bar`baz[args]. With AllowShortContext->False, you would have to write com`foo`Bar`baz[args]. AllowShortContext->False allows you to avoid conflicts with other contexts in the system. The default is True."

UseTypeChecking::usage =
"UseTypeChecking is an option to LoadJavaClass that specifies whether to include or omit the type checking performed in the Wolfram Language on arguments to Java calls. The default is UseTypeChecking->True. UseTypeChecking is largely superseded by the flag $RelaxedTypeChecking. If you use UseTypeChecking->False, you must make sure you pass methods exactly the argument types they expect. Also note that J/Link will not be able to correctly differentiate between multiple definitions of the same method that take the same number of arguments. Most users will have no use for this option."

Val::usage =
"Val is deprecated. The new name is JavaObjectToExpression."

JavaObjectToExpression::usage =
"JavaObjectToExpression[javaobject] converts the specified Java object reference into its value as a \"native\" Wolfram Language expression. Normally, all Java objects that have a meaningful \"by value\" representation in the Wolfram Language are returned by value to the Wolfram Language automatically. Such objects include strings, arrays (these become lists), and instances of so-called wrapper classes like java.lang.Integer. However, you can get a reference form of one of these types if you explicitly call JavaNew or use the ReturnAsJavaObject function. In such cases, you can use JavaObjectToExpression to retrieve the value. JavaObjectToExpression has no effect on objects that have no meaningful \"value\" representation in the Wolfram Language."

ByRef::usage =
"ByRef is deprecated. The new name is ReturnAsJavaObject."

ReturnAsJavaObject::usage =
"ReturnAsJavaObject[expr] causes a Java method call or field access during the evaluation of expr to return its result as an object reference (that is, a JavaObject expression), not a value. Most Java objects are returned as references normally, but those that have a meaningful Wolfram Language representation are returned \"by value\". Such objects include strings, arrays, and the so-called wrapper classes like java.lang.Integer. ReturnAsJavaObject overrides the normal behavior and forces any object returned to the Wolfram Language to be sent only as a reference. It is typically used to avoid needlessly sending large arrays of numbers back and forth between Java and the Wolfram Language. You can use ReturnAsJavaObject to cause only a reference to be sent, then use the JavaObjectToExpression function at the end if the final value is needed."


Begin["`Package`"]

jlinkExternalCall

(* Used in jlinkExternalCall to direct output to the appropriate link (JavaLink[] or JavaUILink[]). *)
getActiveJavaLink

$inPreemptiveCallToJava
$externalCallLinks

(* These 4 are called directly from Java. *)
loadClassAndCreateInstanceDefs
createInstanceDefs
loadClassFromJava
issueNoDefaultCtorMessage

(* Exported so that defs created as classes/objects are loaded can be cleared when Java is restarted. *)
clearObjectDefs
callAllUnloadClassMethods

(* Various methods for converting among class/name/index. *)

jvmFromInstance

classFromInstance
classFromID

classNameFromClass
classIDFromClass
jvmsFromClass
parentClassIDFromClass

End[]   (* `Package` *)


(* Current context will be JLink`. *)

Begin["`CallJava`Private`"]


(****************************************  Java constants  *********************************************)

(* Must remain in sync with Java code. *)
TYPEBOOLEAN		= -1
TYPEBYTE		= -2
TYPECHAR		= -3
TYPESHORT		= -4
TYPEINT			= -5
TYPELONG		= -6
TYPEFLOAT		= -7
TYPEDOUBLE		= -8
TYPESTRING		= -9
TYPEBIGINTEGER	= -10
TYPEBIGDECIMAL	= -11
TYPEEXPR		= -12
TYPECOMPLEX		= -13
(* Every type that has a byval representation in Mathematica must be > TYPEOBJECT
   (required only for createArgPairs). This does not apply to TYPE_DOUBLEORINT and
   TYPE_FLOATORINT, which are used in a very limited way.
*)
TYPEOBJECT		= -14
TYPEFLOATORINT	= -15
TYPEDOUBLEORINT	= -16
TYPEARRAY1		= -17
TYPEARRAY2		= TYPEARRAY1 + TYPEARRAY1
TYPEARRAY3		= TYPEARRAY2 + TYPEARRAY1
TYPEBAD			= -10000


(*********************************  Raw JavaObjects stuff  ****************************************)

(* TODO: HoldAllComplete attr for raw objects, also chain rule for @ precedence change. *)

(********

THIS SECTION NOT YET FIXED UP FOR MULTIPLE VMs.

*********)

JavaObjectQ[_System`Java`JavaObjectInstance] = True

Unprotect[System`Java`JavaObjectInstance]

obj_System`Java`JavaObjectInstance[meth_[args___]] := javaMethod[obj, meth, args]
obj_System`Java`JavaObjectInstance[field_Symbol] := javaField[obj, field];

Protect[System`Java`JavaObjectInstance]


rawObjectComingFromJavaQ = IntegerQ


loadClassAndCreateInstanceDefs[clsName_String, obj_?rawObjectComingFromJavaQ] :=
	Module[{cls},
		cls = loadClassFromJava[clsName, obj];
		If[Head[cls] === JavaClass,
			createInstanceDefs[classIDFromClass[cls], obj],
		(* else *)
			$Failed
		]
	]

(* TODO: Note use of Null on rhs here. I haven't dealt with LoadJavaClass signature yet. *)
loadClassFromJava[vmName_String, clsName_String, obj_?rawObjectComingFromJavaQ] := LoadJavaClass[GetJVM[vmName], clsName, Null, StaticsVisible->False]

(* TODO: Change to use classFromInstance *)
classIDFromInstance[obj_System`Java`JavaObjectInstance] := System`Java`GetClassIndex[obj]


(* Will want to redo the complex tables classNameFromInstance, classIDFromInstance, etc., etc.
   That should be moved into the kernel. For now, classIDFromInstance is redone.
*)

createInstanceDefs[vmName_String, classID_Integer, obj_?rawObjectComingFromJavaQ] :=
	Block[{clsName},
		clsName = classNameFromClass[classFromID[classID]];
		JAssert[clsName =!= $Failed];
		(* TODO: No work with array types, and setting of isJavaStringArray[obj, arrayDepth] = True and others.
		   That will be differnt now, as it is cheap to store all that info in the kernel JavaObject struct
		   and have kernel functions that return it.

		   Respect vmName arg.
		*)
		System`Java`CreateJavaObject[clsName, classID, obj]
	]


(******************************************  callJava  ***********************************************)

(* callJava is a wrapper function that exists to interpose the ReturnAsJavaObject logic before calling jCallJava.
   It also serves as a handy place to catch cases where a call into Java is being made (probably only
   get this far on a static call) but Java is not running (because UninstallJava was called).

   All definitions in this file are made in terms of callJava.
   If $byRef is True, we alter the args to force a "by ref" return before sending them along to jCallJava.
*)

callJava[jvm_JVM, {a__, 1}, b___] /; $byRef := jCallJava[jvm, {a, 0}, b]
callJava[jvm_JVM, c_, d___]                 := jCallJava[jvm, c, d]
callJava[Null, ___]                         := (Message[Java::init]; $Failed) (* No default JVM available *)


(******************************************  JavaNew  ************************************************)

JavaNew::argx0 = "There is no constructor for class `1` that takes zero arguments."
JavaNew::argx1 = "Incorrect number or type of arguments to constructor for class `1`. The argument was `2`."
JavaNew::argx = "Incorrect number or type of arguments to constructor for class `1`. The arguments, shown here in a list, were `2`."
JavaNew::fail = "Error calling constructor for class `1`."
JavaNew::intf = "Cannot create object of type `1`, because it is an interface, not a class."
JavaNew::invcls =
"The JavaClass expression `1` is not valid in the current Java session. The Java runtime may have been quit and restarted since this expression was defined."

JavaNew[clsName_String, args___] :=
	With[{jvm = getDefaultJVM[]},
		If[Head[jvm] === JVM,
			JavaNew[jvm, clsName, args],
		(* else *)
			Message[Java::init];
			$Failed
		]
	]

JavaNew[jvm_JVM, clsName_String, args___] :=
	Module[{cls = LoadJavaClass[jvm, clsName]},
		If[Head[cls] === JavaClass,
			JavaNew[jvm, cls, args],
		(* else *)
			(* LoadJavaClass will have already issued a message if it failed. *)
			$Failed
		]
	]

JavaNew[cls_JavaClass, args___] :=
	With[{jvm = getDefaultJVM[]},
		If[Head[jvm] === JVM,
			JavaNew[jvm, cls, args],
		(* else *)
			Message[Java::init];
			$Failed
		]
	]

JavaNew[jvm_JVM, cls_JavaClass, args___] :=
	Module[{res},
		If[isLoadedClass[cls],
			If[interfaceQFromClass[cls],
				Message[JavaNew::intf, classNameFromClass[cls]];
				Return[$Failed]
			];
			res = javaNew[jvm, classIDFromClass[cls], args];
			If[JavaObjectQ[res],
				res,
			(* else *)
				If[Head[res] === javaNew,
					Switch[Length[{args}],
						0,
							Message[JavaNew::argx0, classNameFromClass[cls]],
						1,
							Message[JavaNew::argx1, classNameFromClass[cls], args],
						_,
							Message[JavaNew::argx, classNameFromClass[cls], {args}]
					],
				(* else *)
					Message[JavaNew::fail, classNameFromClass[cls]]
				];
				$Failed
			],
		(* else *)
			Message[JavaNew::invcls, cls];
			$Failed
		]
	]


(* JavaNew calls javaNew, which only has defs created for it during LoadJavaClass.
   All these defs do nothing but call javaConstructor:

	   javaNew[jvm, id_Integer, argPats...] := javaConstructor[jvm, id, {1,1,1}, argtype, arg1, argtpe, arg2,...]

    Thus, javaNew exists only for pattern-matching. It's a bottleneck that ensures that only legit arg sequences
    make it through to javaConstructor, which calls Java directly.

    NOTE: Now that I have changed javaConstructor so that it does nothing but call callJava, I can get rid of
    javaConstructor completely, making defs for javaNew call callJava...
*)

javaConstructor[jvm_JVM, classID_Integer, indices_List, argTypesAndVals___] :=
	callJava[jvm, {classID, 1, Null, indices, 0}, Length[{argTypesAndVals}]/2, argTypesAndVals]



(********************************************  SetField  ****************************************************)

(* Note that ALL field setting, even statics, is routed through setField. The top layer of providing a
   means to use Set notation is just syntactic sugar.
*)

(* NOTE that because SetField and setField have to Hold the sym or sym[field] argument and
   not the others, it has to reverse the usual order and have the JVM argument be 2nd, not first.
   Then the funcs can remain HoldFirst.
*)

SetAttributes[SetField, HoldFirst]
SetField[sym_[field_Symbol], val_] := With[{obj = sym}, setField[obj[field], val]]
SetField[staticField_Symbol, val_] := setField[staticField, getDefaultJVM[], val]
SetField[staticField_Symbol, jvm_JVM, val_] := setField[staticField, jvm, val]

SetAttributes[setField, HoldFirst]
(* These are the only rules for setField, fall-throughs that issue a message. Other uses are via UpValues on objects or static symbols. *)
setField[obj_[_], _] := (Message[Java::obj, obj]; $Failed)
setField[sym_Symbol, _] := (Message[Java::flds, HoldForm[sym]]; $Failed)
setField[sym_Symbol, Null, _] := (Message[Java::init]; $Failed)  (* Bad JVM returned by getDefaultJVM[] *)
setField[sym_Symbol, jvm_JVM, _] := (Message[Java::flds, HoldForm[sym]]; $Failed)


(********************************  Instance methods and fields  **************************************)

Java::obj = "Attempting to use invalid Java object: `1`."
Java::flds = "Attempting to use invalid Java static field: `1`."

Java::argx0 = "Method named `1` defined in class `2` does not take zero arguments."
Java::argx1 = "Method named `1` defined in class `2` was called with an incorrect number or type of arguments. The argument was `3`."
Java::argx = "Method named `1` defined in class `2` was called with an incorrect number or type of arguments. The arguments, shown here in a list, were `3`."
Java::argxs0 = "The static method `1` does not take zero arguments."
Java::argxs1 = "The static method `1` was called with an incorrect number or type of arguments. The argument was `2`."
Java::argxs = "The static method `1` was called with an incorrect number or type of arguments. The arguments, shown here in a list, were `2`."
Java::nometh = "No method named `1` defined in class `2`."
Java::nometh$ = "No method named `1` defined in class `2`. The method name you specified might have had a conflict with a local variable name in a Module."
Java::fldx = "Attempting to set field named `1` defined in class `2` to an incorrect type of value: `3`."
Java::fldxs = "Attempting to set static field `1` to an incorrect type of value: `2`."
Java::nofld = "No field named `1` defined in class `2`."
Java::nofld$ = "No field named `1` defined in class `2`. The field name you specified might have had a conflict with a local variable name in a Module."
(* This one doesn't really belong here, but it has nowhere else to go. It is issued by MathListener classes
   when user tries to call setHandler[] with an event type that is not supported.
*)
Java::nohndlr = "No event handler method named `1` in `2`."
JavaObject::bad = "The method or field named `1` was called on an invalid Java object."

(* These functions, javaMethod and javaField, are only used during calls that involve an object, static or not.
    That is, the calls that don't go through these functions are static methods and fields that are called by
    name, without an object. These functions allow the "scoping" of names by an object's class, necessary for the
    obj@method[args] syntax to work. You can think of javaMethod and javaField as the bridge between the user-level
    syntax for calling methods/fields, and the internal defs set up for each method/field during LoadJavaClass.

    Every 'obj@' call gets through to javaMethod and javaField--there is no pattern-matching that has to be satisfied.
    Here is where I want to catch the errors:
   		- No method/field with that name at all
   		- Bad arg count or type (obj@meth[] was called and it returned unevaluated)

   	We separate the process of constructing a function that knows how to access a given method or field from the process
   	of calling that function on an instance and some args. That way, the optimizations MethodFunction and FieldFunction
   	can perform the first step ahead of time. The instanceFunc function creates the function that will act on an instance.
   	javaField and javaMethod just apply the function that instanceFunction returns to an instance and args.
*)

Attributes[javaMethod] = {HoldAllComplete}
Attributes[javaField] = {HoldAllComplete}
Attributes[instanceFunc] = {HoldAllComplete}

(* TODO: At one point, I wanted to search to see if methName was a defined method for this class, and if not,
    then let the method call evaluate and see if it was a legal method afterwards. This would counteract the
    potentially confusing effects of the HoldAll attribute. Now I'm not sure if this is necessary--could force
    users to Evaluate...
*)
javaMethod[instance_, methName_Symbol, args___] :=
	lookupInstanceFunc[classFromInstance[instance], HoldComplete[methName], True] [instance, args]

(* For 'gets' *)
javaField[instance_Symbol, fieldName_Symbol] :=
	lookupInstanceFunc[classFromInstance[instance], HoldComplete[fieldName], False] [instance]

(* For 'sets' *)
javaField[instance_Symbol, fieldName_Symbol, val_] :=
	lookupInstanceFunc[classFromInstance[instance], HoldComplete[fieldName], False] [instance, val]

(* Calling a method on an object reference and referring to a field are almost identical in implementation. instanceFunc
   implements this shared functionality. It returns a _function_ that takes an instance and args and calls the field or method.
*)
instanceFunc[cls_, name_Symbol, isMethod_] :=
	Module[{nameStr, sym, resultFunc, wasOn1, wasOn2},
		{wasOn1, wasOn2} = (Head[#] =!= $Off &) /@ {General::spell, General::spell1};
		Off[General::spell, General::spell1];
		(* Find name as symbol in its private context. *)
		nameStr = SymbolName[Unevaluated[name]];
		(* nameStr is now the name in short form, without any context header. *)
		sym = findName[nameStr, cls, isMethod];
		If[sym === $Failed,
			(* findName will already have issued an appropriate message. *)
			resultFunc = $Failed&,
		(* else *)
			(* Call a def for the symbol in its class-specific `JPrivate context. These defs were created during LoadJavaClass.
			   Note that we don't have to send in the class id, as that is known by the definition made in the class's context.
			*)
			If[isMethod,
				With[{sym = sym},
					resultFunc = checkMethodResult[sym[#1, ##2], cls]&
				],
			(* else *)
				(* is field *)
				With[{sym = sym},
					resultFunc = checkFieldResult[sym[fieldTag, #1, ##2], cls]&
				]
			]
		];
		If[wasOn1, On[General::spell]];
		If[wasOn2, On[General::spell1]];
		resultFunc
	]


(* Caching. *)

lookupInstanceFunc[cls_, heldName_, isMethod_] :=
	If[Head[#] === Function,
		#,
	(* else *)
		cacheInstanceFunc[cls, heldName, isMethod]
	]& @ $instanceFuncCache[cls, heldName, isMethod]


cacheInstanceFunc[cls_, heldName:HoldComplete[name_], isMethod_] :=
	Block[{f = instanceFunc[cls, name, isMethod]},
		If[f =!= ($Failed&), $instanceFuncCache[cls, heldName, isMethod] = f];
		f
	]

Internal`SetValueNoTrack[$instanceFuncCache, True]


(* These are funcs used in the functions created by MethodFunction and FieldFunction. They wrap the call and perform error-checking. *)

SetAttributes[checkMethodResult, HoldFirst]
SetAttributes[checkFieldResult, HoldFirst]

checkMethodResult[methodCall_, cls_] :=
	Block[{sym = Head[Unevaluated[methodCall]], result, nameStr},
		result = methodCall; (* This is where the actual call occurs. *)
		If[Head[result] === sym,
			(* Didn't match any pattern for that method--bad arg count or type. Note that the message
			   is issued for the original clsID, which is the class of the instance, not actualClsID,
			   which is the class the method is declared in.
			*)
			nameStr = StringDrop[ToString[sym], Last[{0} ~Join~ Flatten[StringPosition[ToString[sym], "`"]]]];
			Switch[Length[result] - 1,
				0,
					Message[Java::argx0, nameStr, classNameFromClass[cls]],
				1,
					Message[Java::argx1, nameStr, classNameFromClass[cls], Last[result]],
				_,
					Message[Java::argx, nameStr, classNameFromClass[cls], Rest[List @@ result]]
			];
			$Failed,
		(* else *)
			result
		]
	]

checkFieldResult[fieldCall_, cls_] :=
	Block[{sym = Head[Unevaluated[fieldCall]], result, nameStr},
		result = fieldCall; (* This is where the actual call occurs. *)
		If[Head[result] === sym,
			(* Didn't match any pattern for that field--bad arg type. (Was a 'set' call) *)
			nameStr = StringDrop[ToString[sym], Last[{0} ~Join~ Flatten[StringPosition[ToString[sym], "`"]]]];
			Message[Java::fldx, nameStr, classNameFromClass[cls], Last[result]];
			$Failed,
		(* else *)
			result
		]
	]


(* Finds method or field name as symbol in its private context, recursively checking superclasses.
   name is the short (no context) name. Criterion for finding the symbol is that it have DownValues defined for it.
   Block used for speed only.
*)
findName[name_String, cls_, isMethod_] :=
	Block[{ctxt, sym, pctxt, parentClsID, looksLikeModuleVar, curCls, curClsID = classIDFromClass[cls]},
		pctxt = "JPrivate`" <> name;
		While[True,
			curCls = classFromID[curClsID];
			ctxt = contextFromClass[curCls];
			If[Head[ctxt] =!= String,
				Message[JavaObject::bad, name];
				Return[$Failed]
			];
			sym = ToExpression[ctxt <> pctxt];
			If[DownValues[Evaluate[sym]] =!= {},
				Return[sym]
			];
			parentClsID = parentClassIDFromClass[curCls];
			If[parentClsID === Null,
				(* We've reached the Object class, so no method of that name is even defined in Mathematica.
				   Note that the message is issued for the original class (this is the class of the instance
				   on which the method is being invoked).
				*)
				looksLikeModuleVar = MatchQ[Characters[name], {__, "$", ___?DigitQ}];
				With[{tag = If[isMethod, "nometh", "nofld"]  <> If[looksLikeModuleVar, "$", ""]},
					Message[MessageName[Java, tag], name, classNameFromClass[cls]]
				];
				Return[$Failed]
			];
			curClsID = parentClsID
		]
	]


(**********************************************  LoadJavaClass  ***************************************************)

LoadClass = LoadJavaClass  (* LoadClass is deprecated. *)

LoadJavaClass::fail = "Java failed to load class `1`."
LoadJavaClass::ambig = "Ambiguous function in class `1`. `2` has multiple definitions: `3`."
LoadJavaClass::ambigctor = "Ambiguous constructor in class `1`. Multiple definitions: `2`."

Options[LoadJavaClass] = {StaticsVisible->False, AllowShortContext->True, UseTypeChecking->True}

(* Here is the structure of what Java side sends when a class is loaded:
 	{className_String, classID_Integer, ctors_List, methods_List, fields_List}
  where:
    ctors is:   {id_Integer, declaration_String, paramTypes___Integer}
    methods is: {id_Integer, isStatic_True|False, declaration_String, retType_String, name_String, paramTypes___Integer}
    fields is:  {id_Integer, isInherited_True|False, isStatic_True|False, type_String, name_String, type_Integer}
*)

(* New in v2.0, LoadJavaClass lets you supply a second argument that is an object reference whose ClassLoader will be used to load
   the named class. This feature is used by calls to putReference() in Java, when they need to call LoadJavaClass to load the
   classes of the objects they are sending to Mathematica. You can use the second argument from Mathematica also, however.
   The only reason for doing this would be if you needed to ensure that a class was loaded by some special ClassLoader.
   The actual circumstances where you would want to do this, and where the two-argument form of LoadJavaClass would be the best
   way, are too obscure to even go into here.
*)

LoadJavaClass[c:{__String}, objSupplyingClassLoader_Symbol:Null, isBeingLoadedAsComplexClass:(True | False):False, opts___?OptionQ] :=
	LoadJavaClass[#, objSupplyingClassLoader, isBeingLoadedAsComplexClass, opts]& /@ c

LoadJavaClass[jvm_JVM, c:{__String}, objSupplyingClassLoader_Symbol:Null, isBeingLoadedAsComplexClass:(True | False):False, opts___?OptionQ] :=
	LoadJavaClass[jvm, #, objSupplyingClassLoader, isBeingLoadedAsComplexClass, opts]& /@ c

LoadJavaClass[c1_String, c2__String, objSupplyingClassLoader_Symbol:Null, isBeingLoadedAsComplexClass:(True | False):False, opts___?OptionQ] :=
	LoadJavaClass[{c1, c2}, objSupplyingClassLoader, isBeingLoadedAsComplexClass, opts]

LoadJavaClass[jvm_JVM, c1_String, c2__String, objSupplyingClassLoader_Symbol:Null, isBeingLoadedAsComplexClass:(True | False):False, opts___?OptionQ] :=
	LoadJavaClass[jvm, {c1, c2}, objSupplyingClassLoader, isBeingLoadedAsComplexClass, opts]

LoadJavaClass[className_String, objSupplyingClassLoader_Symbol:Null, isBeingLoadedAsComplexClass:(True | False):False, opts___?OptionQ] :=
	LoadJavaClass[getDefaultJVM[], className, objSupplyingClassLoader, isBeingLoadedAsComplexClass, opts]

LoadJavaClass[jvm:(_JVM | Null), className_String, objSupplyingClassLoader_Symbol:Null, isBeingLoadedAsComplexClass:(True | False):False, opts___?OptionQ] :=
	 executionProtect[
	     (* Preempt/AbortProtect because this deals with global data and we want to prevent reentrancy.
	        Also, loading Java classes is not an unlikely thing to have happen on a preemptive link.
	     *)
	    Block[{lc, classID, legalClassName, returnedClassName, parentClass, isInterface, alreadyLoaded,
				staticsVisible, allowShortContext, useTypeChecking, ctorRecs, methRecs, fieldRecs,
				classContext, shortClassContext, usingShortContext, wasOn1, wasOn2, wasOn3, cls},
			(* Block for speed only. *)

			If[!checkJVM[jvm],   (* Issues messages. *)
				Return[$Failed]
			];

			alreadyLoaded = classFromName[className];
			(* Only bail out if this class has been loaded in this JVM. Be sure to keep classID
			   if already loaded into different JVM.
			*)
			If[Head[alreadyLoaded] === JavaClass,
				If[MemberQ[jvmsFromClass[alreadyLoaded], jvm],
					Return[alreadyLoaded]
				];
				classID = classIDFromClass[alreadyLoaded];
				(* Clear and re-make the class defs in case the class has changed since last loaded. *)
				clearOutClassContext[jvm, alreadyLoaded],
			(* else *)
				classID = $clsID++;
			];

			If[!TrueQ[$isNestedLoad],
				jAddToClassPath[jvm, autoClassPath[], True, False];
				(* For compatibility with J/Link 1.1. *)
				jAddToClassPath[jvm, $ExtraClassPath, True, False]
			];

			(* $isNestedLoad's sole purpose is to avoid repeatedly calling AddToClassPath in the above lines as parent
			   classes are recursively loaded. AddToClassPath is comparatively expensive, and autoClassPath and
			   $ExtraClassPath won't change during the loading of a single class (and all its parent classes).
			*)
			Block[{$isNestedLoad = True},
				lc = jLoadClass[jvm, classID, className, objSupplyingClassLoader, isBeingLoadedAsComplexClass]
			];

			(* This pattern is {"classname", parentCls_JavaClass, isInterface, {ctors}, {methods}, {fields}} *)
			(* Note that the classname we store is the one returned from loadclass, not the name originally
			    specified by user. The one returned from Java is the fully qualified name.
			*)
			If[!MatchQ[lc, {_String, Null | _JavaClass, True | False, _List, _List, _List}],
				Message[LoadJavaClass::fail, className];
				Return[$Failed]
			];

			{returnedClassName, parentClass, isInterface, ctorRecs, methRecs, fieldRecs} = lc;

			{staticsVisible, allowShortContext, useTypeChecking} =
				{StaticsVisible, AllowShortContext, UseTypeChecking} /. Flatten[{opts}] /. Options[LoadJavaClass];

			usingShortContext = TrueQ[allowShortContext];

			cls = storeClass[classID,
					   		returnedClassName,
					   		jvm,
					   		If[# === Null, Null, classIDFromClass[#]]& @ parentClass,
					   		isInterface,
					   		usingShortContext
				  ];

			{wasOn1, wasOn2, wasOn3} = (Head[#] =!= $Off &) /@ {General::shdw, General::spell, General::spell1};
			Off[General::shdw];
			Off[General::spell];
			Off[General::spell1];

			classContext = contextFromClass[cls];
			(* shortClassContext is for statics. If classContext is java`lang`Foo`, shortClassContext is Foo`.
			    Want to make statics available by using just cass name as context, not full package path.
			*)
			If[usingShortContext,
				shortClassContext = shortClassContextFromClassContext[classContext],
			(* else *)
				shortClassContext = classContext
			];

			createConstructorStubs[classID, ctorRecs, TrueQ[useTypeChecking]];
			createMethodStubs[classID, classContext, shortClassContext, methRecs, TrueQ[useTypeChecking]];
			createFieldStubs[classID, classContext, shortClassContext, fieldRecs, TrueQ[useTypeChecking]];

			(* Call the class's onLoadClass method, if it has one. *)
			Begin[shortClassContext];
			callOnLoadClassMethod[jvm, classID];
			End[];

			(* This occurs after caling the onLoadClass method for a reason, although it's not necessarily a good one.
			    Don't want behavior of code executed by onLoadClass to be dependent on StaticsVisible setting. Force that code
			    to always use full context names for statics it wants to refer to.
			*)
			If[staticsVisible,
				BeginPackage[classContext];
				EndPackage[]
			];

			If[wasOn1, On[General::shdw]];
			If[wasOn2, On[General::spell]];
			If[wasOn3, On[General::spell1]];

			cls
		]
	]


(* Class IDs are generated in Mathematica and are unique across all JVMs. *)

If[!IntegerQ[$clsID], $clsID = 0]

Internal`SetValueNoTrack[$clsID, True]


(***************************************  Constructor Handling  *********************************************)

createConstructorStubs[classID_Integer, ctors_List, useTypeChecking_] :=
	Module[{ctorsWithoutObjects, ctorsWithObjects, ctorList, ctorListForNonObjectDups,
			ctorListForObjectDups, indices, nonDups, dups, widest},
		(* Rely on the fact that argtype constants are negative numbers, and the only such numbers in each record. *)
		If[!FreeQ[ctors, x_Integer /; x <= TYPEBAD],
			(* TODO: Flesh out this error. Needs to report as problem in ctor. *)
			Message[LoadJavaClass::badtype];
			Return[]
		];
		(* Drop declaration; strip down to {id, argTypes...} *)
		ctorList = Delete[#, {2}]& /@ ctors;

		(* If not ctors defined, add the no-arg ctor. *)
		If[ctorList === {}, ctorList = {{0}}];

		ctorsWithoutObjects = Select[ctorList, FreeQ[#, TYPEOBJECT]&];
		ctorsWithObjects = Select[ctorList, !FreeQ[#, TYPEOBJECT]&];

		ctorListForNonObjectDups = Reverse /@ (ctorsWithoutObjects /. (x_Integer /; x < 0) :> argTypeToPattern[x]);
		ctorListForNonObjectDups = Split[Sort @ ctorListForNonObjectDups, Drop[#1, -1] === Drop[#2, -1] &];
		(* At this point, ctorListForNonObjectDups looks like:
		                           vv  Identical patterns are grouped  vv
			{{{argPat..., index}}, {{argPat..., index},{argPat..., index}}, {{argPat..., index}}}
		*)
		DebugPrint["ctorListForNonObjectDups = ", ctorListForNonObjectDups, Trigger:>$DebugDups];
		ctorListForNonObjectDups = Map[Last, ctorListForNonObjectDups, {2}];
		(* Now just {{index}, {index, index}, {index}, ...} *)

		nonDups = Select[ctorListForNonObjectDups, Length[#] == 1&];
		DebugPrint["nonDups = ", nonDups, Trigger:>$DebugDups];
		dups = Select[ctorListForNonObjectDups, Length[#] > 1&];
		DebugPrint["dups = ", dups, Trigger:>$DebugDups];
		(* The First below is just to strip outer list braces (the Select always returns one element) *)
		widest = pickWidest /@ (dups /. i_Integer :> First @ Select[ctorsWithoutObjects, First[#] == i&]);
		DebugPrint["widest = ", widest, Trigger:>$DebugDups];
		indices = First /@ Join[nonDups, widest];
		DebugPrint[indices, Trigger:>$DebugDups];
		(* These are the indices for which defs will be created (for ctors that don't have object args). *)
		(*indices = Last /@ Last /@ ctorListForNonObjectDups;*)
		Scan[createCtorDef[classID, useTypeChecking, {#}]&, Select[ctors, MemberQ[indices, First[#]]&]];

		ctorListForObjectDups = Reverse /@ (ctorsWithObjects /. (x_Integer /; x < 0) :> argTypeToPattern[x]);
		ctorListForObjectDups = Split[Sort @ ctorListForObjectDups, Drop[#1, -1] === Drop[#2, -1] &];
		(* At this point, ctorListForObjectDups looks like:
		                           vv  Identical patterns are grouped  vv
			{{{argPat..., index}}, {{argPat..., index},{argPat..., index}}, {{argPat..., index}}}
		*)
		indices = ctorListForObjectDups /. {__, n_Integer} :> n;
		(* indices looks like:  {{1}, {2,3,4}, {5}, {6}} *)
		(* The First below is just to strip outer list braces (the Select always returns one element) *)
		Scan[createCtorDef[classID, useTypeChecking, #]&, indices /. n_Integer :> First @ Select[ctors, First[#] == n &]];
	]

(* ctors looks like: {{index_Integer, declaration_String, paramTypes___Integer}...}.
    Only for object-containing dups will the list of ctors have > 1 element.
*)
createCtorDef[classID_Integer, useTypeChecking_, ctors_List] :=
	Module[{argTypes, argNames, argPats, indices, patternFunc},
		JAssert @ MatchQ[ctors, {{_Integer, _String, ___Integer}..}];
		(* useTypeChecking == False means don't create Mathematica patterns for arg matching on LHS of definitions; just use _. *)
		patternFunc = If[useTypeChecking, argTypeToPattern, Blank[]&];
		If[Length[ctors] == 1,
			(* all cases except object-containing ctors with multiple identical arg sequences *)
			argTypes = Drop[First @ ctors, 2],
		(* else *)
			(* Object-containing ctors with multiple identical arg sequences. Need to create argTypes that has the
			    largest of all types in each slot.
			*)
			argTypes = Apply[Min, Drop[Transpose[ctors], 2], {1}]
		];
		argNames = Take[$argNames, Length[argTypes]];
		argPats = MapThread[Pattern, {argNames, patternFunc /@ argTypes}];
		indices = Sort[First /@ ctors];
		(javaNew[jvm_, classID, Sequence @@ argPats] := javaConstructor[jvm, classID, ##])& @@
			Join[{indices}, createArgPairs[argTypes, argNames]];
	]

(* dups looks like {{index_Integer, paramTypes___Integer}, {index_Integer, paramTypes___Integer}, ...} *)
pickWidest[dups_List] :=
	Module[{sums},
		JAssert @ MatchQ[dups, {{__Integer}..}];
		(* Logic here is to pick the version with largest args by using sum of type constants as measure.
		    Somewhat arbitrary, as type constants don't measure 'sizeof' accurately, but at least they are in proper order.
		*)
		sums = (Plus @@ Drop[#, 1])& /@ dups;
		dups[[ First @ Flatten @ Position[sums, Min[sums]] ]]
	]


(*********************************************  Method Handling  ************************************************)

createMethodStubs[classID_Integer, classContext_String, shortClassContext_String, meths_List, useTypeChecking_] :=
	Module[{methNames, methsWithoutObjects, methsWithObjects,
	        methListForNonObjectDups, methListForObjectDups, dups, nonDups, indices},
	        				(*  index,     decl,   isStatic,   name,   paramTypes *)
	    JAssert[MatchQ[meths, {{_Integer, _String, True|False, _String, ___Integer}...}]];
		(* The following works because argtype constants are negative numbers, and the only such numbers in each record. *)
		If[!FreeQ[meths, x_Integer /; x <= TYPEBAD],
			(* TODO: Flesh out this error. Needs to report as problem in method. *)
			Message[LoadJavaClass::badtype];
			Return[]
		];

		methsWithoutObjects = Select[meths, FreeQ[#, TYPEOBJECT]&];
		methsWithObjects = Select[meths, !FreeQ[#, TYPEOBJECT]&];

		methListForNonObjectDups = Reverse /@ (methsWithoutObjects /. (x_Integer /; x < 0) :> argTypeToPattern[x]);
		methListForNonObjectDups = Split[Sort @ methListForNonObjectDups, Drop[#1, -3] === Drop[#2, -3] &];
		(* At this point, methListForNonObjectDups looks like:
		               vvvv  Identical name and patterns are grouped
			{{rec}, {rec, rec}, {rec}}
			where rec is {argPat..., name, decl, static, index}
		*)
		methListForNonObjectDups = Map[Last, methListForNonObjectDups, {2}];

		nonDups = Select[methListForNonObjectDups, Length[#] == 1&];
		DebugPrint["nonDups = ", nonDups, Trigger:>$DebugDups];
		dups = Select[methListForNonObjectDups, Length[#] > 1&];
		DebugPrint["dups = ", dups, Trigger:>$DebugDups];
		(* The First below is just to strip outer list braces (the Select always returns one element). Drop[#, {2,4}]
		    is to strip down to just index and paramTypes.
		*)
		widest = pickWidest /@ (dups /. i_Integer :> Drop[First @ Select[methsWithoutObjects, First[#] == i&], {2,4}]);
		DebugPrint["widest = ", widest, Trigger:>$DebugDups];
		(* These are the indices for which defs will be created (for meths that don't have object args). *)
		indices = First /@ Join[nonDups, widest];
		DebugPrint[indices, Trigger:>$DebugDups];

		Scan[createMethodDef[classID, classContext, shortClassContext, useTypeChecking, {#}]&,
			 Select[meths, MemberQ[indices, First[#]]&]
		];

		methListForObjectDups = Reverse /@ (methsWithObjects /. (x_Integer /; x < 0) :> argTypeToPattern[x]);
		methListForObjectDups = Split[Sort @ methListForObjectDups, Drop[#1, -3] === Drop[#2, -3] &];
		(* At this point, methListForObjectDups looks like:
		               vvvv  Identical name and patterns are grouped
			{{rec}, {rec, rec}, {rec}}
			where rec is {argPat..., name, decl, static, index}
		*)
		indices = methListForObjectDups /. {__, n_Integer} :> n;
		(* indices looks like:  {{1}, {2,3,4}, {5}, {6}} *)
		(* The First below is just to strip outer list braces (the Select always returns one element) *)
		Scan[createMethodDef[classID, classContext, shortClassContext, useTypeChecking, #]&,
			 indices /. n_Integer :> First @ Select[meths, First[#] == n &]
		]
	]


(* meths looks like:
	{{index_Integer, declaration_String, isStatic_True|False, name_String, paramTypes___Integer}..}
*)
createMethodDef[classID_Integer, ctxt_String, shortCtxt_String, useTypeChecking_, meths_List] :=
	Module[{methName, argTypes, argNames, argPats, indices, atLeastOneStatic, sym, patternFunc, shortSym},
		JAssert @ MatchQ[meths, {{_Integer, _String, True|False, _String, ___Integer}...}];
		(* useTypeChecking == False means don't create Mathematica patterns for arg matching on LHS of definitions; just use _. *)
		patternFunc = If[useTypeChecking, argTypeToPattern, Blank[]&];
		methName = toLegalName @ meths[[1, 4]];
		If[Length[meths] == 1,
			(* all cases except object-containing meths with multiple identical arg sequences *)
			argTypes = Drop[First @ meths, 4],
		(* else *)
			(* Object-containing meths with multiple identical arg sequences. Need to create argTypes that has the
			    largest of all types in each slot.
			*)
			argTypes = Apply[Min, Drop[Transpose[meths], 4], {1}]
		];
		argNames = Take[$argNames, Length[argTypes]];
		argPats = MapThread[Pattern, {argNames, patternFunc /@ argTypes}];
		indices = Sort[First /@ meths];
		atLeastOneStatic = Or @@ (#[[3]]& /@ meths);
		With[{indices = indices, argc = Length[argTypes], argPairs = createArgPairs[argTypes, argNames]},
			ToExpression[ctxt <> "JPrivate`" <> methName][obj_, Sequence @@ argPats] :=
					callJava[jvmFromInstance[obj], {classID, 2, obj, indices, 1}, argc, Sequence @@ argPairs];
			If[atLeastOneStatic,
				With[{sym = Symbol[ctxt <> methName]},
					sym[jvm_JVM, Sequence @@ argPats] := callJava[jvm, {classID, 2, Null, indices, 1}, argc, Sequence @@ argPairs];
					sym[Sequence @@ argPats] := callJava[getDefaultJVM[], {classID, 2, Null, indices, 1}, argc, Sequence @@ argPairs];
					sym[args___] := (
						Switch[Length[{args}],
							0,
								Message[Java::argxs0, HoldForm[sym]],
							1,
								Message[Java::argxs1, HoldForm[sym], args],
							_,
								Message[Java::argxs, HoldForm[sym], {args}]
						];
						$Failed);
					(* Downvalues of isJavaStaticSymbol are used to record which symbols in a context hav been given defs.
					   This is used only in clearOutClassContext, to avoid clearing non-Java symbols in case the same
					   context name is being used by a Mathematica package. No need to do this for the shortCtxt symbols,
					   as they do not need to be cleared when the class is unloaded. They just point to their deep-context
					   counterparts, which will get cleared.
					*)
					isJavaStaticSymbol[ctxt <> methName] = True;
					isJavaStaticSymbol[shortCtxt <> methName] = True;
					(* Now make def also available in "short" class context. *)
					If[shortCtxt != ctxt,
						(* If a def already exists mapping the short context sym to the long context sym,
						   we don't want to do this again (this leads to wacky defs like a`b`c = a`b`c).
						   This can only happen on reload of a class into a second JVM. This test avoids
						   making defs a second time.
						*)
						shortSym = ToHeldExpression[shortCtxt <> methName];
						If[Not[ValueQ @@ shortSym],
							Evaluate[ReleaseHold[shortSym]] = sym
						]
					]
				]
			]
		];
	]


(******************************************  Field Handling  *************************************************)

Java::setfield = "Trying to set field to illegal value."

(* fields is:  {{index_Integer, isStatic_True|False, type_String, name_String, type_Integer}...}
*)
createFieldStubs[classID_Integer, classContext_String, shortClassContext_String, fields_List, useTypeChecking_] :=
	Module[{fieldNames},
		JAssert @ MatchQ[fields, {{_Integer, True|False, _String, _String, _Integer}...}];
		(* Bit of a hack follows; rely on the fact that argtype constants are negative numbers,
		   and the only such numbers in each record.
		*)
		If[!FreeQ[fields, x_Integer /; x <= TYPEBAD],
			(* Flesh out this error. Needs to report as problem in field. *)
			Message[LoadJavaClass::badtype];
			Return[]
		];
		Scan[createFieldDef[classID, classContext, shortClassContext, useTypeChecking, #]&, fields];
	]


(* field looks like:
	{index_Integer, isStatic_True|False, type_String, name_String, type_Integer}
*)
createFieldDef[classID_Integer, ctxt_String, shortCtxt_String, useTypeChecking_, field_List] :=
	Module[{fieldName, index, argName, argType, patternFunc},
		JAssert @ MatchQ[field, {_Integer, True|False, _String, _String, _Integer}];
		(* useTypeChecking == False means don't create Mathematica patterns for arg matching on LHS of definitions; just use _. *)
		patternFunc = If[useTypeChecking, argTypeToPattern, Blank[]&];
		fieldName = toLegalName @ field[[4]];
		index = field[[1]];
		argName = First[$argNames];
		argType = field[[5]];
		With[{index = index, privSym = ToExpression[ctxt <> "JPrivate`" <> fieldName],
				argName = argName, argType = argType, pat = Pattern @@ {argName, patternFunc[argType]}},
			(* The Private` symbol fieldTag is just a placeholder to distinguish field accesses from
			   no-arg method calls of the same name.
			*)
			privSym[fieldTag, instance_Symbol] := callJava[jvmFromInstance[instance], {classID, 3, instance, {index}, 1}, 0];
			privSym[fieldTag, instance_Symbol, pat] := callJava[jvmFromInstance[instance], {classID, 3, instance, {index}, 1}, 1, argType, argName];
			If[field[[2]] === True,
		 		(* is static *)
		 		(* Note there is a very small problem with my scheme for accessing fields in the static case.
		 		   I want the user to be able to get the value with java`awt`Button`foo, but that requires
		 		   assigning an ownvalue to java`awt`Button`foo, which will cause a problem with the func
		 		   def for that symbol that would be created if a static method had the same name as a static field.
		 		   Bail on this for now. Maybe it's OK to document some hack for this unlikely circumstance...
		 		   Note it's only a problem for getting, not setting, and you could avoid even that by using
		 		   the object syntax for calling a static method.
		 		*)
		 		makeStaticFieldDefs[ToHeldExpression[ctxt <> fieldName], ctxt <> fieldName, classID, index, argType, pat];
		 		If[shortCtxt != ctxt,
		 			(* Because we set UpValues for Set calls, it is not enough to just define the shortCtxt field symbols to be the
		 			   deep context symbols, as is done with methods. Instead, we must explicitly make definitions for the shortCtxt ones.
		 			*)
		 			makeStaticFieldDefs[ToHeldExpression[shortCtxt <> fieldName], shortCtxt <> fieldName, classID, index, argType, pat]
		 		]
			]
		];
	]

	makeStaticFieldDefs[Hold[sym_], symStr_, classID_, index_, argType_, pat_] :=
		(* The !ValueQ test prevents this from being called twice on a symbol. This will happen if you load two classes
		   with the same short context. Calling it twice can cause all sorts of bad behavior.
		*)
		If[!ValueQ[sym],
			(* For consistency we route everything through setField for statics even though Set method has no drawbacks
			   for statics. If I wanted to completely remove all reliance on Set for statics I could just remove the
			   following one line. There is no reason to want to do that, though.
			*)
			sym /: Set[sym, val_] := setField[sym, getDefaultJVM[], val];
			sym /: setField[sym, jvm_JVM, val_] :=
						If[MatchQ[val, pat],
							callJava[jvm, {classID, 3, Null, {index}, 1}, 1, argType, val];
							val,
						(* else *)
							Message[Java::fldxs, HoldForm[sym], val];
							$Failed
						];
			(* Must make this def last. *)
			sym := callJava[getDefaultJVM[], {classID, 3, Null, {index}, 1}, 0];
			(* Downvalues of isJavaStaticSymbol are used to record which symbols in a context have been given defs.
				This is used only in clearOutClassContext, to avoid clearing non-Java symbols in case the same
				context name is being used by a Mathematica package.
			*)
			isJavaStaticSymbol[symStr] = True;
		]


(*****************************************   Returning Refs   **********************************************)

(* Note that ReturnAsJavaObject sets up an "environment" where all calls return by ref. This means that
   ReturnAsJavaObject[foo[obj@method[]]] will work, but be careful if there are more Java calls embedded in the
   expression, as in ReturnAsJavaObject[obj@method[SomeClass`FOO]], as these deeper calls will also return by ref.
*)

ByRef = ReturnAsJavaObject   (* ByRef is deprecated. *)

SetAttributes[ReturnAsJavaObject, HoldAll]

ReturnAsJavaObject[x_] := Block[{$byRef = True}, x]


Val = JavaObjectToExpression   (* Val is deprecated. *)

JavaObjectToExpression[x_?JavaObjectQ] := jVal[jvmFromInstance[x], x]

JavaObjectToExpression[x_] := x   (* Perhaps this should issue a message? *)


(******************************************  ExternalCall fix  *********************************************)

If[!ValueQ[$externalCallLinks], $externalCallLinks = {}; $inPreemptiveCallToJava = {}]

Internal`SetValueNoTrack[$externalCallLinks, True]
Internal`SetValueNoTrack[$inPreemptiveCallToJava, True]
Internal`SetValueNoTrack[inPreemptiveCallFromJava, True]

(* Calls to external functions via the standard Install mechanism are accomplished by the function ExternalCall.
   Unfortunately, ExternalCall is deficient in its handling of aborts. It also needs to dynamically choose
   the link to use to communicate with Java. Thus, I need my own version, jlinkExternalCall.

   This function differs functionally from ExternalCall in that it wraps the write-read pair in AbortProtect, so that you
   cannot leave the link in an "off-by-one" state by aborting between the write and read. This AbortProtect does not
   prevent the necessary behavior that user aborts fired while the kernel is blocking in LinkRead are sent to Java
   as MLAbortMessages. In other words, the ability to abort Java computations is not affected. The AbortProtect does
   prevent the behavior of being able to do a "hard" abort via the two-step combination of
   "Interrupt Evaluation/Abort Command Being Evaluated". This procedure causes Mathematica to treat the
   abort like any other abort and ignore that it is in LinkRead. This is not very useful, though, since the link will
   probably be out of sync because the result is not read. The correct way to handle this is to select "Kill linked program"
   in the Interrupt dialog box, not "Abort Command Being Evaluated". This causes Java to quit.

   The code itself is quite different from ExternalCall. Gone is the need for ExternalAnswer and the silly recursive way
   in which that was implemented.

   The link that will be passed in here is the one given by getActiveJavaLink[].

   This function also rejects preemptive calls into Java when they are unsafe, which is the case
   when they are preemptive, reentrant (call into Java is already occurring), but not originating on the UI link.

   TODO: Code for AbortProtect suggests that it is not interruptible by preemptive evals. Doesn't that
   mean that (at least until that gets changed) there is no need for all the reentrancy control in this func?
*)

jlinkExternalCall[jvm_JVM, packet_CallPacket] :=
	Block[{$CurrentLink, pkt = packet, res, sequenceResult, isPreemptive = TrueQ[MathLink`IsPreemptive[]],
				$externalCallLinks = $externalCallLinks, $inPreemptiveCallToJava = $inPreemptiveCallToJava},
		If[Head[finishInstall[jvm]] =!= JVM, Return[$Failed]];
		$CurrentLink = getActiveJavaLink[jvm];
		AbortProtect[
			(* Reject as unsafe calls that are: preemptive, reentrant on the same link,
			   and not just callbacks from a preemptive comp that began in Java or callbacks
			   during a preemptive comp that went out to Java.
			*)
			If[isPreemptive && MemberQ[$externalCallLinks, $CurrentLink] &&
						!inPreemptiveCallFromJava[jvm] && !MemberQ[$inPreemptiveCallToJava, jvm],
				Message[Java::preemptive];
				$Failed,
			(* else *)
				AppendTo[$externalCallLinks, $CurrentLink];
				If[isPreemptive, AppendTo[$inPreemptiveCallToJava, jvm]];
				While[True,
					If[LinkWrite[$CurrentLink, pkt] === $Failed, Return[$Failed]];
					If[$CurrentLink === $InternalLink, Java`DispatchToJava[$CurrentLink]];
					res = LinkRead[$CurrentLink, HoldComplete];
					Switch[res,
						HoldComplete[EvaluatePacket[_]],
							(* Re-enable aborts during the computation in Mathematica of EvaluatePacket contents, but have
							   them just cause $Aborted to be returned to Java, not call Abort[].
							*)
							pkt = ReturnPacket[CheckAbort[res[[1,1]], $Aborted]],
						HoldComplete[ReturnPacket[_]],
							Return[res[[1,1]]],
                        HoldComplete[_Sequence],
                            (* Exceptionally unlikely case--we are getting a Sequence[...] expr from Java, as in 
                               JavaObjectToExpression[MakeJavaExpr[Sequence[1,2,3]]]. There seems to be no way to
                               use Return with such an animal, so we need ugly special handling.
                            *)
                            sequenceResult = res[[1]];
                            Break[],
                        HoldComplete[_],
                            Return[res[[1]]],
						_,
							Return[res]
					]
				]
			]
		];
		(* The only way to get here is if we got a Sequence[...] expr returned from Java. *)
		sequenceResult
	]


(* This gives the link that will be used for any given call to Java. Note that preemptive calls
   to Java will never use the UI link unless they are just callbacks during a preemptive comp
   that began in Java. Note also that JavaUILink[] will never be returned unless it is safe to use it.
*)
getActiveJavaLink[jvm_JVM] :=
	Block[{isPreemptive = TrueQ[MathLink`IsPreemptive[]]},
		Which[
			!isPreemptive && $ParentLink === JavaUILink[jvm] && $ParentLink =!= Null ||
					isPreemptive && inPreemptiveCallFromJava[jvm],
				JavaUILink[jvm],
			isPreemptive && (MemberQ[$externalCallLinks, javaPreemptiveLink[jvm]] ||
					(MemberQ[$externalCallLinks, JavaLink[jvm]] && !MemberQ[$inPreemptiveCallToJava, jvm])),
				javaPreemptiveLink[jvm],
			True,
				JavaLink[jvm]
		]
	]


(***************************************  Creating instance defs  ***************************************)

(* Called only from Java, for returning objects to Mathematica whose classes have not been loaded by user. *)
loadClassAndCreateInstanceDefs[vmName_String, clsName_String, obj_Symbol] :=
	Module[{cls},
		cls = loadClassFromJava[vmName, clsName, obj];
		If[Head[cls] === JavaClass,
			createInstanceDefs[vmName, classIDFromClass[cls], obj],
		(* else *)
			$Failed
		]
	]


(* Called from Java whenever classes need to be loaded by Java code. This is currently in two circumstances: loading
   parent classes of a class the user has manually loaded using LoadJavaClass or JavaNew; or classes loaded because an object
   of their type is being returned from Java.
   Note that if you want to have a class loaded with your own settings for the options of LoadJavaClass, then you had better
   load it yourself, before it is autoloaded for you.
*)
loadClassFromJava[vmName_String, clsName_String, obj_Symbol] := LoadJavaClass[GetJVM[vmName], clsName, obj, StaticsVisible->False]
loadClassFromJava[_, Null, _] = Null  (* For superclass of java.lang.Object. *)


(* These have just the following definitions. Defs are never added (instead, upvalues are placed on the JavaObjectN symbols). *)
JavaObjectQ[_] = False
JavaObjectQ[Null] = True
classFromInstance[_] = $Failed
jvmFromInstance[Null] := getDefaultJVM[]

Internal`SetValueNoTrack[JavaObjectQ, True];
Internal`SetValueNoTrack[classFromInstance, True];
Internal`SetValueNoTrack[jvmFromInstance, True];

(* This function cannot be changed without also changing unloadClass. *)

createInstanceDefs[vmName_String, classID_Integer, obj_Symbol] :=
	executionProtect[  (* Probably only needs to be AbortProtect. *)
	    Block[{cls, clsName, arrayDepth = 0, arrayType, nameLen, complexClass, jvm},
			(* Block for speed only. *)
			jvm = GetJVM[vmName];
			cls = classFromID[classID];
			Internal`SetValueNoTrack[obj, True];
			SetAttributes[obj, {HoldAllComplete}];
			addToJavaBlock[obj];
			clsName = classNameFromClass[cls];
			JAssert[clsName =!= $Failed];
			nameLen = StringLength[clsName];
			arrayDepth = Which[nameLen >= 3 && StringTake[clsName, 3] === "[[[",
								 3,
							   nameLen >= 2 && StringTake[clsName, 2] === "[[",
								 2,
							   nameLen >= 1 && StringTake[clsName, 1] === "[",
							     1,
							   True,
							     0
						 ];
			arrayType = If[arrayDepth === 0, Null, StringTake[clsName, {arrayDepth + 1}]];
			(* This defeats normal precedence for @ operator. Needed for chaining: obj@meth1[]@meth2[]. *)
			obj[(meth:_[___])[args___]] := obj[meth][args];
			obj[meth_[args___]] := javaMethod[obj, meth, args];
			obj[field_Symbol] := javaField[obj, field];
			obj /: setField[obj[field_Symbol], val_] := If[# === $Failed, $Failed, val]& [javaField[obj, field, val]];
            (* Set JavaObjectQ after the setField defs above; see bug 279943 *)
            JavaObjectQ[obj] ^= True;
            classFromInstance[obj] ^= cls;
            jvmFromInstance[obj] ^= jvm;
			(* New in M version 8. This allows us to avoid setting rules on Set. See comments elsewhere on VetoableValueChange system. *)
			Internal`ValueChangeVeto[obj, True];
			If[arrayType =!= Null,
				Switch[arrayType,
					"B" | "C" | "S" | "I" | "J",
						obj /: isJavaIntegerArray[obj, arrayDepth] = True,
					"F" | "D",
						obj /: isJavaRealArray[obj, arrayDepth] = True,
					"Z",
						obj /: isJavaBooleanArray[obj, arrayDepth] = True,
					"L",
						obj /: isJavaObjectArray[obj, arrayDepth] = True;
						Which[StringMatchQ[clsName, "*java.lang.String*"],
								 obj /: isJavaStringArray[obj, arrayDepth] = True,
							  StringMatchQ[clsName, "*java.math.BigDecimal*"],
								 obj /: isJavaBigDecimalArray[obj, arrayDepth] = True,
							  StringMatchQ[clsName, "*java.math.BigInteger*"],
								 obj /: isJavaBigIntegerArray[obj, arrayDepth] = True,
							  StringMatchQ[clsName, "*com.wolfram.jlink.Expr*"],
								 obj /: isJavaExprArray[obj, arrayDepth] = True,
							  complexClassName = classNameFromClass[GetComplexClass[]];
							  StringQ[complexClassName] && StringMatchQ[clsName, "*" <> complexClassName <> "*"],
								 obj /: isJavaComplexArray[obj, arrayDepth] = True
						]
				]
			];
			Format[obj, OutputForm] = Format[obj, TextForm] = "<<JavaObject[" <> clsName <> "]>>";
			(obj /: MakeBoxes[obj, fmt_] = InterpretationBox[RowBox[{"\[LeftGuillemet]", RowBox[{"JavaObject", "[", #, "]"}], "\[RightGuillemet]"}], obj])& [clsName];
			obj
		]
	]



(* Called during Un/InstallJava to wipe out defs created as objects are created. *)
clearObjectDefs[jvm_JVM] := ClearAll[Evaluate["JLink`Objects`" <> nameFromJVM[jvm] <> "`*"]]


(****************************************  Method call-building utils  ************************************)

(* This creates parameter sequence on rhs of ctor/method defs. *)
createArgPairs[argTypes_, argNames_] := Flatten[Transpose[{argTypes, argNames}]]

(* This creates the patterns on the lhs of definitions. *)
argTypeToPattern[n_Integer] :=
	Switch[n,
		TYPEBOOLEAN,
			Return[True | False],
		TYPEBYTE | TYPECHAR | TYPESHORT | TYPEINT | TYPELONG,
			Return[_Integer],
		TYPEFLOAT | TYPEDOUBLE,
			Return[_Real],
		TYPECOMPLEX,
			Return[_?isComplex],
		TYPESTRING,
			Return[_?isString],
		TYPEEXPR,
			Return[_],
		TYPEOBJECT,
			Return[_?JavaObjectQ],
		TYPEBIGINTEGER,
			Return[_?isBigInteger],
		TYPEBIGDECIMAL,
			Return[_?isBigDecimal],
		TYPEDOUBLEORINT | TYPEFLOATORINT,
			Return[_?NumberQ],
		TYPEBOOLEAN + TYPEARRAY1,
			Return[_?isTrueFalseList],
		TYPEBYTE + TYPEARRAY1 | TYPECHAR + TYPEARRAY1 | TYPESHORT + TYPEARRAY1 |
				TYPEINT + TYPEARRAY1 | TYPELONG + TYPEARRAY1,
			Return[_?isIntegerList],
		TYPEFLOAT + TYPEARRAY1 | TYPEDOUBLE + TYPEARRAY1,
			Return[_?isRealList],
		TYPECOMPLEX + TYPEARRAY1,
			Return[_?isComplexList],
		TYPESTRING + TYPEARRAY1,
			Return[_?isStringList],
		TYPEOBJECT + TYPEARRAY1,
			Return[_?isObjectList],
		TYPEBIGINTEGER + TYPEARRAY1,
			Return[_?isBigIntegerList],
		TYPEBIGDECIMAL + TYPEARRAY1,
			Return[_?isBigDecimalList],
		TYPEEXPR + TYPEARRAY1,
			Return[_?isExprList],
		TYPEDOUBLEORINT + TYPEARRAY1 | TYPEFLOATORINT + TYPEARRAY1,
			Return[_?isNumberList],
		TYPEBOOLEAN + TYPEARRAY2,
			Return[_?isTrueFalseArray2],
		TYPEBYTE + TYPEARRAY2 | TYPECHAR + TYPEARRAY2 | TYPESHORT + TYPEARRAY2 |
				TYPEINT + TYPEARRAY2 | TYPELONG + TYPEARRAY2,
			Return[_?isIntegerArray2],
		TYPEFLOAT + TYPEARRAY2 | TYPEDOUBLE + TYPEARRAY2,
			Return[_?isRealArray2],
		TYPECOMPLEX + TYPEARRAY2,
			Return[_?isComplexArray2],
		TYPESTRING + TYPEARRAY2,
			Return[_?isStringArray2],
		TYPEDOUBLEORINT + TYPEARRAY2 | TYPEFLOATORINT + TYPEARRAY2,
			Return[_?isNumberArray2],
		TYPEOBJECT + TYPEARRAY2,
			Return[_?isObjectArray2],
		TYPEBIGINTEGER + TYPEARRAY2,
			Return[_?isBigIntegerArray2],
		TYPEBIGDECIMAL + TYPEARRAY2,
			Return[_?isBigDecimalArray2],
		TYPEEXPR + TYPEARRAY2,
			Return[_?isExprArray2],
		TYPEBOOLEAN + TYPEARRAY3,
			Return[_?isTrueFalseArray3],
		TYPEBYTE + TYPEARRAY3 | TYPECHAR + TYPEARRAY3 | TYPESHORT + TYPEARRAY3 |
				TYPEINT + TYPEARRAY3 | TYPELONG + TYPEARRAY3,
			Return[_?isIntegerArray3],
		TYPEFLOAT + TYPEARRAY3 | TYPEDOUBLE + TYPEARRAY3,
			Return[_?isRealArray3],
		TYPECOMPLEX + TYPEARRAY3,
			Return[_?isComplexArray3],
		TYPESTRING + TYPEARRAY3,
			Return[_?isStringArray3],
		TYPEDOUBLEORINT + TYPEARRAY3 | TYPEFLOATORINT + TYPEARRAY3,
			Return[_?isNumberArray3],
		TYPEOBJECT + TYPEARRAY3,
			Return[_?isObjectArray3],
		TYPEBIGINTEGER + TYPEARRAY3,
			Return[_?isBigIntegerArray3],
		TYPEBIGDECIMAL + TYPEARRAY3,
			Return[_?isBigDecimalArray3],
		TYPEEXPR + TYPEARRAY3,
			Return[_?isExprArray3],
		_,
			_List   (* For 4-deep and deeper arrays, just use a minimal pattern. *)
	]

(* Equivs looks like:
     when called for constructor:  {{argPat..., declaration, index}..}
     when called for method:       {{argPat..., name, decl, static, index}..}
*)
warnForDups[class_String, equivs_List] :=
	Module[{name, decs},
		If[Length[equivs] > 1,
			If[Head @ equivs[[1,-2]] === String,
				(* is a constructor *)
				decs = #[[-2]]& /@ equivs;
				Message[LoadJavaClass::ambigctor, class, decs],
			(* else *)
				(* is a method *)
				name = equivs[[1,-4]];
				decs = #[[-3]]& /@ equivs;
				Message[LoadJavaClass::ambig, class, name, decs]
			]
		]
	]

toLegalName[s_String] := StringReplace[s, "_" -> "U"]


(* As an optimization, create ahead of time a list of arg symbols used in Java function definitions. This is a surprisingly
   expensive operation to do repeatedly. Note the arbitrary max of 40 args per method.
*)
$argNames = Table[ToExpression["arg$" <> ToString[n]], {n, 1, 200}];



(******************************************  Class-related utils  *****************************************)

(*
	There are three ways to refer to a Java class after it has been loaded:

	(1) By its fully-qualified name as a string  "com.foo.MyClass"
	(2) By its class id, which is an key into the collection of ClassRecords held in Java.
		This is the only way classes are known to Java. You must always convert to this id
		when crossing over into Java. This method is not visible to users.
	(3) By a JavaClass expr, which is what it returned by LoadJavaClass. A JavaClass just encapsulates
		the string form with the id form: JavaClass["com.foo.MyClass", 17]

	Want users to be able to refer to classes as strings or as JavaClass exprs (id form is
	not documented), although JavaClass is the preferred way. We have functions to convert among
	these representations.
*)


JavaClass::notfound = "No class named `1` has been loaded."


Format[JavaClass[name_, rest__], OutputForm] := "JavaClass[" <> name <> ", <>]"
Format[JavaClass[name_, rest__], TextForm] := "JavaClass[" <> name <> ", <>]"
JavaClass /: MakeBoxes[JavaClass[name_, rest__], fmt_] :=
	InterpretationBox[RowBox[{"JavaClass", "[", name, ",", "<>", "]"}], JavaClass[name, rest]]


(*********
		Next set are the only funcs that know anything about he structure of JavaClass:
            JavaClass[name_String, id_Integer, jvms_List, ...]
        and how class info is stored in Mathematica (as DownValues of loadedClasses).
*********)

Internal`SetValueNoTrack[loadedClasses, True]

storeClass[id_Integer, className_String, jvm_JVM, parentClsID:(Null | _Integer),
				isInterface:(True | False), usesShortContext:(True | False)] :=
	Module[{cls},
		cls = classFromName[className];
		If[Head[cls] === JavaClass,
			(* Insert jvm without checking if it's already there because this function will
               not be called, if it was there.
            *)
            cls = Insert[cls, jvm, {3, -1}],
		(* else *)
			cls = JavaClass[className, id, {jvm}, parentClsID, toContextName[toLegalName[className]], isInterface, usesShortContext]
		];
		loadedClasses[id] = cls
	]

classNameFromClass[cls_JavaClass] := cls[[1]]
classIDFromClass[cls_JavaClass] := cls[[2]]
jvmsFromClass[cls_JavaClass] := cls[[3]]
parentClassIDFromClass[cls_JavaClass] := cls[[4]]
contextFromClass[cls_JavaClass] := cls[[5]]
interfaceQFromClass[cls_JavaClass] := cls[[6]]
useShortContextFromClass[cls_JavaClass] := cls[[7]]


(* Does not take a JVM arg. *)
classFromName[cls_String] :=
	Module[{classes, legalClassName},
		legalClassName = toLegalName[cls];
		classes = Select[DownValues[loadedClasses], classNameFromClass[Last[#]] === legalClassName &];
		If[classes === {},
			$Failed,
		(* else *)
			JAssert[Length[classes] === 1];
			First[classes] [[2]]
		]
	]

(* Need this function to detect if a user is using a JavaClass expression that refers to an unloaded class.
   This could happen if the Java runtime is quit and restarted.
*)
isLoadedClass[jc_JavaClass] := ValueQ[loadedClasses[classIDFromClass[jc]]]

classFromID[i_Integer] := loadedClasses[i]


allLoadedClasses[] := Last /@ DownValues[loadedClasses]

destroyLoadedClassInfo[] := Clear[loadedClasses]


(******************************************)

(* TODO: These should Block some symbols that would be useful (e.g. $ClassContext). *)

callOnLoadClassMethod[jvm_JVM, i_Integer] := jOnLoadClass[jvm, i]
callOnUnloadClassMethod[jvm_JVM, i_Integer] := jOnUnloadClass[jvm, i]

(* TODO: NOT FUNCTIONING. find classes only loaded into one JVM--the dying one. *)
callAllUnloadClassMethods[jvm_JVM] := callOnUnloadClassMethod[jvm, #]& /@ classIDFromClass /@ allLoadedClasses[jvm]

(* unused. *)
clearAllClassContexts[jvm_JVM] := Scan[clearOutClassContext[jvm, #]&, allLoadedClasses[]]

(* This one is used only when a class is being reloaded. *)
clearOutClassContext[jvm_JVM, jc_JavaClass] :=
	Module[{ctxt = contextFromClass[jc], shortCtxt, nms, javaNames},
		JAssert[Head[ctxt] === String];
		(* Must get rid of $ContextPath or symbols in visible ccontexts will have their names returned without
		   the context prefix.
		*)
		nms = Block[{$ContextPath}, Names[ctxt <> "*"]];
		(* Downvalues of isJavaStaticSymbol are used to record which symbols in a context have been given defs.
		   We use this to avoid clearing non-Java symbols in case the same context name is being used by a
		   Mathematica package. We only need be concerned with statics, since non-statics are only created
		   in the special JPrivate` context, which we can be sure will have no conflicts.
		*)
		javaNames = Select[nms, isJavaStaticSymbol];
		(* ClearAll["sym1", "sym2", ...] is vastly more expensive than ClearAll[sym1, sym2, ...] or ClearAll["ctxt`*"],
		   so avoid the first method at all costs. We only do selective clearing when it is necessary (this will likely
		   be only in cases where a read-in context has the same name as a Java-created one), and when we do, do it
		   for symbol names rather than strings.
		*)
		If[javaNames =!= {},
			If[Length[nms] == Length[javaNames],
				ClearAll @@ {ctxt <> "*"},
			(* else *)
				(* The Unevaluated wrapped around each symbol does not interfere with ClearAll. *)
				ClearAll @@ (ToExpression[#, InputForm, Unevaluated]& /@ javaNames)
			]
		];
		If[Names[#] =!= {}, ClearAll[#]]& [ctxt <> "JPrivate`*"];
		If[usesShortContextFromClass[jc],
			shortCtxt = shortClassContextFromClassContext[ctxt];
			nms = Block[{$ContextPath}, Names[shortCtxt <> "*"]];
			If[shortCtxt == "System`",
				(* The trick above with $ContextPath doesn't work for the System` context. We need to explicitly force the
				   name strings to begin with "System`", as that's how they are recorded in isJavaStaticSymbol.
				*)
				nms = ("System`" <> #)& /@ nms
			];
			javaNames = Select[nms, isJavaStaticSymbol];
			If[javaNames =!= {},
				If[Length[nms] == Length[javaNames],
					ClearAll @@ {shortCtxt <> "*"},
				(* else *)
					ClearAll @@ (ToExpression[#, InputForm, Unevaluated]& /@ javaNames)
				]
			];
		]
	]


shortClassContextFromClassContext[ctxt_String] :=
	StringDrop[ctxt, If[Length[#] > 1, #[[-2]], 0]]& @ Union[Flatten[StringPosition[ctxt, "`"]]]

toContextName[clsName_String] :=
	If[StringTake[clsName, 1] === "[", "java`lang`Array`", StringReplace[clsName, "." -> "`"] <> "`"]


End[]