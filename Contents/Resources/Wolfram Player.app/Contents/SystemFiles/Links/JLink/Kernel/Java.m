(* :Title: Java.m *)

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
   Declarations of all the "j functions", which are the special calls into Java defined in
   Install.java. These must be declared in the `Package` context, but their defs are
   embedded in Install.java. Any new j functions added to Install.java must also be named here.
	
   This file is a component of the J/Link Mathematica source code.
   It is not a public API, and should never be loaded directly by users or programmers.

   J/Link uses a special system wherein one package context (JLink`) has its implementation
   split among a number of .m files. Each component file has its own private context, and also
   potentially introduces public symbols (in the JLink` context) and so-called "package" symbols,
   where the term "package" comes from Java terminology, referring to symbols that are visible
   everywhere within the implementation of J/Link, but not to clients.
*)


Begin["`Package`"]

jCallJava
jLoadClass
jThrow
jReleaseObject
jReleaseAllObjects
jVal
jOnLoadClass
jOnUnloadClass
jSetComplex
jGetComplex
jReflect
jShow
jSameQ
jInstanceOf
jGetException
jSetException
jAllowRaggedArrays
jConnectToFEServer
jDisconnectToFEServer
jPeekClasses
jPeekObjects
jClassPath
jAddToClassPath
jSetUserDir
jUIThreadWaitingQ
jAllowUIComputations
jYieldTime
jGetConsole
jExtraLinks
jGetWindowID
jAddTitleChangeListener
jSetVMName

End[]  (* `Package` *)


(* Current context will be JLink`. *)

Begin["`Java`Private`"]

jCallJava[jvm_JVM, indices_List, argCount_Integer, args___] :=
	jlinkExternalCall[jvm, CallPacket[1, {indices, argCount, args}]]
	
jLoadClass[jvm_JVM, classID_Integer, class_String, objSupplyingClassLoader_Symbol, isComplexClass_] :=
	jlinkExternalCall[jvm, CallPacket[2, {classID, class, objSupplyingClassLoader, isComplexClass}]]
	
jThrow[jvm_JVM, exc_, msg_String] :=
	jlinkExternalCall[jvm, CallPacket[3, {exc, msg}]]
	
jReleaseObject[jvm_JVM, instances:{__Symbol}] := 
	jlinkExternalCall[jvm, CallPacket[4, {instances}]]
	
jVal[jvm_JVM, inst_Symbol] :=
	jlinkExternalCall[jvm, CallPacket[5, {inst}]]
	
jOnLoadClass[jvm_JVM, classID_Integer] :=
	jlinkExternalCall[jvm, CallPacket[6, {classID}]]
	
jOnUnloadClass[jvm_JVM, classID_Integer] :=
	jlinkExternalCall[jvm, CallPacket[7, {classID}]]
	
jSetComplex[jvm_JVM, classID_Integer] :=
	jlinkExternalCall[jvm, CallPacket[8, {classID}]]
	
jReflect[jvm_JVM, classID_Integer, type_Integer, inherited:(True | False)] :=
	jlinkExternalCall[jvm, CallPacket[9, {classID, type, inherited}]]
	
jShow[jvm_JVM, wnd_Symbol] :=
	jlinkExternalCall[jvm, CallPacket[10, {wnd}]]
	
jSameQ[jvm_JVM, obj1_Symbol, obj2_Symbol] :=
	jlinkExternalCall[jvm, CallPacket[11, {obj1, obj2}]]
	
jInstanceOf[jvm_JVM, obj_Symbol, className_String] :=
	jlinkExternalCall[jvm, CallPacket[12, {obj, className}]]
	
jAllowRaggedArrays[jvm_JVM, allow:(True | False)] :=
	jlinkExternalCall[jvm, CallPacket[13, {allow}]]
	
jGetException[jvm_JVM] :=
	jlinkExternalCall[jvm, CallPacket[14, {}]]
	
jConnectToFEServer[jvm_JVM, linkName_String, protocol_String] :=
	jlinkExternalCall[jvm, CallPacket[15, {linkName, protocol}]]
	
jDisconnectToFEServer[jvm_JVM] :=
	jlinkExternalCall[jvm, CallPacket[16, {}]]
	
jPeekClasses[jvm_JVM] :=
	jlinkExternalCall[jvm, CallPacket[17, {}]]
	
jPeekObjects[jvm_JVM] :=
	jlinkExternalCall[jvm, CallPacket[18, {}]]
	
jClassPath[jvm_JVM] :=
	jlinkExternalCall[jvm, CallPacket[19, {}]]
	
jAddToClassPath[jvm_JVM, dirs:{__String}, searchForJars:(True | False), prepend:(True | False)] :=
	jlinkExternalCall[jvm, CallPacket[20, {dirs, searchForJars, prepend}]]
	
jSetUserDir[jvm_JVM, dir_String] :=
	jlinkExternalCall[jvm, CallPacket[21, {dir}]]
	
jAllowUIComputations[jvm_JVM, allow:(True | False), enteringModal:(True | False):False] :=
	jlinkExternalCall[jvm, CallPacket[22, {allow, enteringModal}]]
	
jUIThreadWaitingQ[jvm_JVM] :=
	jlinkExternalCall[jvm, CallPacket[23, {}]]
	
jYieldTime[jvm_JVM, millis_Integer] :=
	jlinkExternalCall[jvm, CallPacket[24, {millis}]]
	
jGetConsole[jvm_JVM] :=
	jlinkExternalCall[jvm, CallPacket[25, {}]]
	
jExtraLinks[jvm_JVM, uiName_String, preName_String, prot_String, linkSnooperCmdLine:_String:""] :=
	jlinkExternalCall[jvm, CallPacket[26, {uiName, preName, prot, linkSnooperCmdLine}]]
	
jGetWindowID[jvm_JVM, obj_Symbol] :=
	jlinkExternalCall[jvm, CallPacket[27, {obj}]]
	
jAddTitleChangeListener[jvm_JVM, obj_Symbol, func_String] :=
	jlinkExternalCall[jvm, CallPacket[28, {obj, func}]]
	
jSetVMName[jvm_JVM, name_String] :=
	jlinkExternalCall[jvm, CallPacket[29, {name}]]

jSetException[jvm_JVM, obj_Symbol] :=
	jlinkExternalCall[jvm, CallPacket[30, {obj}]]
	

End[]

