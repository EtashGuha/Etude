(* :Title: Reflection.m *)

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
   Functions for retrieving info about ctors/methods/fields.
	
   This file is a component of the J/Link Mathematica source code.
   It is not a public API, and should never be loaded directly by users or programmers.

   J/Link uses a special system wherein one package context (JLink`) has its implementation
   split among a number of .m files. Each component file has its own private context, and also
   potentially introduces public symbols (in the JLink` context) and so-called "package" symbols,
   where the term "package" comes from Java terminology, referring to symbols that are visible
   everywhere within the implementation of J/Link, but not to clients.
*)



Constructors::usage =
"Constructors[javaclass] returns a list (in TableForm) of the Java declarations for all constructors of the specified class. You can also specify a class by its name or an object of that class, as in Constructors[\"java.net.URL\"] or Constructors[javaobject]."

Methods::usage =
"Methods[javaclass] returns a list (in TableForm) of the Java declarations for all methods of the specified class. You can also specify a class by its name or an object of that class, as in Methods[\"java.net.URL\"] or Methods[javaobject]. To make them easier to read, the declarations have had removed some keywords that are not very relevant to their use from the Wolfram Language. These keywords are final, synchronized, and native. The public keyword is also removed, as the methods are always public."

Fields::usage =
"Fields[javaclass] returns a list (in TableForm) of the Java declarations for all fields of the specified class. You can also specify a class by its name or an object of that class, as in Fields[\"java.net.URL\"] or Fields[javaobject]. To make them easier to read, the declarations have had the keywords transient and volatile removed. The public keyword is also removed, as the fields are always public."

Inherited::usage =
"Inherited is an option to Methods and Fields (in J/Link) and NETTypeInfo (in .NET/Link). The default, Inherited->True, means include information about members inherited from superclasses."


Begin["`Package`"]
(* No Package-level exports, but Begin/End are needed by tools. *)
End[]


(* Current context will be JLink`. *)

Begin["`Reflection`Private`"]


Constructors::null = "Object is null."
Methods::null = "Object is null."
Fields::null = "Object is null."

Options[Methods] = Options[Fields] = {Inherited->True, IgnoreCase->False}

Constructors[Null] := Message[Constructors::null]

Constructors[obj_?JavaObjectQ, opts___?OptionQ] :=
	Constructors[jvmFromInstance[obj], classFromInstance[obj], opts]

Constructors[clsName_String, opts___?OptionQ] :=
	Constructors[GetJVM[InstallJava[]], clsName, opts]

Constructors[jvm_JVM, clsName_String, opts___?OptionQ] := 
	Module[{cls},
		cls = LoadJavaClass[jvm, clsName];
		If[Head[cls] === JavaClass,
			Constructors[jvm, cls, opts],
		(* else *)
			{}
		]
	]

Constructors[cls_JavaClass, opts___?OptionQ] := Constructors[getDefaultJVM[], cls, opts]
	
Constructors[jvm_JVM, cls_JavaClass, opts___?OptionQ] :=
	Module[{},
		TableForm[shortenNames[#, False]& /@ ctors[jvm, cls]]
	]


Methods[Null, ___] := Message[Methods::null]

Methods[obj_?JavaObjectQ, pat_String:"*", opts___?OptionQ] :=
	Methods[jvmFromInstance[obj], classFromInstance[obj], pat, opts]

Methods[clsName_String, pat_String:"*", opts___?OptionQ] :=
	Methods[GetJVM[InstallJava[]], clsName, pat, opts]

Methods[jvm_JVM, clsName_String, pat_String:"*", opts___?OptionQ] := 
	Module[{cls},
		cls = LoadJavaClass[jvm, clsName];
		If[Head[cls] === JavaClass,
			Methods[jvm, cls, pat, opts],
		(* else *)
			{}
		]
	]

Methods[cls_JavaClass, pat_String:"*", opts___?OptionQ] := Methods[getDefaultJVM[], cls, pat, opts]
	
Methods[jvm_JVM, cls_JavaClass, pat_String:"*", opts___?OptionQ] :=
	Module[{inherit, ignoreCase},
		{inherit, ignoreCase} = {Inherited, IgnoreCase} /. Flatten[{opts}] /. Options[Methods];
		Select[shortenNames[#, False]& /@ Union @ Sort @ methods[jvm, cls, inherit],
			   StringMatchQ[extractName[#, False], pat, IgnoreCase -> ignoreCase]&
		] // sortOnName // TableForm
	]


Fields[Null, ___] := Message[Fields::null]

Fields[obj_?JavaObjectQ, pat_String:"*", opts___?OptionQ] :=
	Fields[jvmFromInstance[obj], classFromInstance[obj], pat, opts]

Fields[clsName_String, pat_String:"*", opts___?OptionQ] := 
	Fields[GetJVM[InstallJava[]], clsName, pat, opts]

Fields[jvm_JVM, clsName_String, pat_String:"*", opts___?OptionQ] := 
	Module[{cls},
		cls = LoadJavaClass[jvm, clsName];
		If[Head[cls] === JavaClass,
			Fields[jvm, cls, pat, opts],
		(* else *)
			{}
		]
	]

Fields[cls_JavaClass, pat_String:"*", opts___?OptionQ] := Fields[getDefaultJVM[], cls, pat, opts]

Fields[jvm_JVM, cls_JavaClass, pat_String:"*", opts___?OptionQ] :=
	Module[{inherit, ignoreCase},
		{inherit, ignoreCase} = {Inherited, IgnoreCase} /. Flatten[{opts}] /. Options[Fields];
		Select[shortenNames[#, True]& /@ Union @ Sort @ fields[jvm, cls, inherit],
			   StringMatchQ[extractName[#, True], pat, IgnoreCase -> ignoreCase]&
		] // sortOnName // TableForm
	]


extractName[decl_String, isField:(True | False)] :=
	Module[{d = decl, spacePositions, spacePos, parenPos},
		spacePositions = Union @ Flatten @ StringPosition[d, " "];
		If[isField,
			StringDrop[d, Last[spacePositions]],
		(* else *)
			parenPos = First @ Flatten @ StringPosition[d, "("];
			spacePos = Last @ Select[spacePositions, # < parenPos &];
			StringTake[d, {spacePos + 1, parenPos - 1}]
		]		
	]
	

ctors[jvm_JVM, cls_JavaClass] := jReflect[jvm, classIDFromClass[cls], 1, True]
methods[jvm_JVM, cls_JavaClass, includeInherited:True|False] := jReflect[jvm, classIDFromClass[cls], 2, includeInherited]
fields[jvm_JVM, cls_JavaClass, includeInherited:True|False] := jReflect[jvm, classIDFromClass[cls], 3, includeInherited]


shortenNames[s_String, isField:(True | False)] :=
	Module[{str = s, parenPos, periodPos, spacePos, isStatic, isFinal},
		(* Array ctors have faked names that are empty strings. Bail on these. *)
		If[StringLength[s] == 0, Return[s]];
		(* Drop "public" prefix, if it is there. Cannot change the order of next trimmings, as this is the
		   canonical order in which these identifiers appear.
		*)
		If[StringLength[str] >= 7 && StringTake[str, 7] === "public ",
			str = StringDrop[str, 7]
		];
		If[StringLength[str] >= 7 && StringTake[str, 7] === "static ",
			isStatic = True;
			str = StringDrop[str, 7]
		];	
		If[StringLength[str] >= 6 && StringTake[str, 6] === "final ",
			isFinal = True;
			str = StringDrop[str, 6]
		];
		If[StringLength[str] >= 7 && StringTake[str, 7] === "native ",
			str = StringDrop[str, 7]
		];
		If[StringLength[str] >= 13 && StringTake[str, 13] === "synchronized ",
			str = StringDrop[str, 13]
		];
		If[StringLength[str] >= 10 && StringTake[str, 10] === "transient ",
			str = StringDrop[str, 10]
		];
		If[StringLength[str] >= 9 && StringTake[str, 9] === "volatile ",
			str = StringDrop[str, 9]
		];
		(* Put back 'final' if we trimmed it earlier (but only for fields). *)
		If[isField && isFinal, str = "final " <> str];
		(* Put back 'static' if we trimmed it earlier. *)
		If[isStatic, str = "static " <> str];
		(* Get rid of any "java.lang." but not "java.lang.reflect." *)
		str = StringReplace[str, "java.lang.reflect" -> "JAVALANGREFLECT"];
		str = StringJoin @@ (Characters[str] //. {a___, "j","a","v","a",".","l","a","n","g",".", b___} :> {a, b});
		str = StringReplace[str, "JAVALANGREFLECT" -> "java.lang.reflect"];
		(* The next gyrations are not needed for fields, partly because of the different way in which their
		   declarations are returned from the Java code.
		*)
		If[!isField,
			(* Add spaces after commas for readability. *)
			str = StringJoin @@ Flatten[Characters[str] /. "," -> {",", " "}];
			parenPos = First @ Flatten @ StringPosition[str, "(", 1];
			(* Get last period before '('. *)
			periodPos = Select[Union @ Flatten @ StringPosition[str, "."], # < parenPos &];
			If[Length[periodPos] =!= 0,
				periodPos = Last[periodPos];
				spacePos = Select[Union @ Flatten @ StringPosition[str, " "], # < periodPos &];
				(* Fake the spacePos if there is no space before period (this will be the case for ctors, and only for ctors). *)
				spacePos = If[Length[spacePos] === 0, 0, Last[spacePos]];
				(* Delete everything between first space before '(' and last period before '(', including the period. *)
				str = StringDrop[str, {spacePos + 1, periodPos}]
			]
		];
		str
	]


sortOnName[x_List] := Sort[x, OrderedQ[{dropPrefixes[#1], dropPrefixes[#2]}]&]

dropPrefixes[s_String] :=
	s //
		If[StringMatchQ[#, "static *"], StringDrop[#, 7], #]& //
			If[StringMatchQ[#, "final *"], StringDrop[#, 6], #]& //
				StringDrop[#, First[Flatten[StringPosition[#, " "]]]]&


End[]
