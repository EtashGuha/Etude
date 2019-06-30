(* :Title: Misc.m *)

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
   Miscellaneous functions.

   This file is a component of the J/Link Mathematica source code.
   It is not a public API, and should never be loaded directly by users or programmers.

   J/Link uses a special system wherein one package context (JLink`) has its implementation
   split among a number of .m files. Each component file has its own private context, and also
   potentially introduces public symbols (in the JLink` context) and so-called "package" symbols,
   where the term "package" comes from Java terminology, referring to symbols that are visible
   everywhere within the implementation of J/Link, but not to clients.
*)



JavaShow::usage =
"JavaShow[windowObj] causes the specified Java window to be brought to the foreground, so that it appears in front of notebook windows. Use this function in place of standard Java window methods like setVisible (or its deprecated equivalent, show) and toFront. The effects those methods have are unfortunately not identical on different Java virtual machines and operating systems. JavaShow performs the required steps in VM-specific ways, so programs that use it will work identically on different configurations. The argument must be an object of a class that can represent a top-level window (i.e., an instance of the java.awt.Window class or a subclass)."

DoModal::usage =
"DoModal[] does not return until the Java side sends an expression of the form EvaluatePacket[EndModal[___]]. Use DoModal to have a program wait until some Java user-interface action signals that it can proceed. Typically, this would be used to implement a modal dialog box that needs to return a result to Mathematica when it is dismissed. DoModal returns the EndModal[args] expression."

EndModal::usage =
"EndModal is the head of an expression sent by Java to signal the end of a DoModal[] loop."

ServiceJava::usage =
"ServiceJava[] allows a single computation originating from Java to proceed. ServiceJava joins DoModal and ShareKernel as the means to allow computations to proceed that originate from Java. \"Originate\" from Java means that the call to Mathematica is in response to a user action in Java (like pressing a button), as opposed to a callback into Mathematica from a call to Java that originated from Mathematica. Use ServiceJava when you want a running Mathematica program to periodically allow computations from the Java user interface. ServiceJava, like ShareKernel, is unnecessary in Mathematica 5.1 and later because the kernel is always ready for computations that originate in Java, even when busy with another computation."

AppletViewer::usage =
"AppletViewer[class, params] displays a window with an applet of the named class running in it. The class can be given as a string or a JavaClass expression. The params argument is a list of strings, each specifying a name=value pair supplying applet properties that would normally appear in an HTML file in <PARAM> tags or other applet attributes (like WIDTH and HEIGHT attributes). A typical specification might be {\"WIDTH=400\", \"HEIGHT=400\"}."

PeekObjects::usage =
"PeekObjects is deprecated. It has been replaced by LoadedJavaObjects."

LoadedJavaObjects::usage =
"LoadedJavaObjects[] returns a list of the Java objects that have been sent to the Wolfram Language (and not yet released with ReleaseJavaObject). It is intended to be used only as a debugging aid."

PeekClasses::usage =
"PeekClasses is deprecated. It has been replaced by LoadedJavaClasses."

LoadedJavaClasses::usage =
"LoadedJavaClasses[] returns a list of the classes currently loaded into Java by the Wolfram Language. It is intended to be used only as a debugging aid."

SetInternetProxy::usage =
"SetInternetProxy[\"host\", port] sets proxy information in your Java session for accessing the Internet. If you use a proxy for accessing the Internet, you may need to call this function to enable Java code to use the Internet. Consult your network administrator for proxy settings. A typical example would look like SetInternetProxy[\"proxy.mycompany.com\", 8080]."

ShowJavaConsole::usage =
"ShowJavaConsole[] displays the Java console window and begins capturing output sent to the Java System.out and System.err streams. Anything written to these streams before ShowJavaConsole is first called will not appear, but subsequent output will be captured even if the window is closed and reopened. Only the most recent one thousand lines of output are displayed. ShowJavaConsole[\"stdout\"] captures only System.out, and ShowJavaConsole[\"stderr\"] captures only System.err."

InstanceOf::usage =
"InstanceOf[javaobject, javaclass] gives True if javaobject is an instance of the class or interface javaclass, or a subclass. Otherwise, it returns False. It mimics the behavior of the Java language's instanceof operator. The javaclass argument can be either the fully-qualified class name as a string, or a JavaClass expression."

SameObjectQ::usage =
"SameObjectQ[javaobject1, javaobject2] returns True if and only if the JavaObject expressions javaobject1 and javaobject2 refer to the same Java object. In other words, it behaves like the Java language's == operator as applied to object references."

JavaThrow::usage =
"JavaThrow[\"exceptionClassName\", \"detailMsg\"] causes the specified exception to be thrown in the Java thread that called the Wolfram Language program in which JavaThrow occurred. You can also call JavaThrow with an Exception object created by a call to JavaNew, instead of a string giving the name of the exception class. If you use the string form, the second argument is an optional detail message for the exception. It is up to the user to ensure that the exception being created has an appropriate constructor. JavaThrow is a specialized function that few programmers will have any use for."

GetClassPath::usage =
"GetClassPath[] is deprecated. Use JavaClassPath[] instead."

JavaClassPath::usage =
"JavaClassPath[] returns the class search path in use by the Java runtime. This includes classes specified via the CLASSPATH environment variable (if any), directories and files added by the user with AddToClassPath, and those directories automatically searched by J/Link. The result is a list of strings specifying directories and .jar or .zip files, in the same order that they will be searched for classes by the Java runtime."

AddToClassPath::usage =
"AddToClassPath[\"location1\", \"location2\", ...] adds the specified full paths to directories and jar or zip files to the J/Link class search path. If you specify a directory, it will be searched automatically for .jar and .zip files, in addition to appropriately nested .class files, so you do not have to name .jar or .zip files explicitly."

$ExtraClassPath::usage =
"$ExtraClassPath is deprecated. Use the function AddToClassPath instead."

SetComplexClass::usage =
"SetComplexClass[\"classname\"] specifies the Java class to use for complex numbers sent from, and returned to, the Wolfram Language."

GetComplexClass::usage =
"GetComplexClass[] returns the Java class used for complex numbers sent from, and returned to, the Wolfram Language."

ParentClass::usage =
"ParentClass[javaclass] returns the JavaClass expression representing the parent class of javaclass. You can also specify an object of a class, as in ParentClass[javaobject]."

ClassName::usage =
"ClassName[javaclass] returns, as a string, the fully-qualified name of the Java class represented by javaclass. You can also specify an object of the class, as in ClassName[javaobject]."

GetClass::usage =
"GetClass[javaobject] returns the JavaClass that identifies the object's class."

ImplementJavaInterface::usage =
"ImplementJavaInterface[interfaces, handlerMappings] uses the Dynamic Proxy facility of Java to create a new Java class and return an object of that class that implements the named interface or list of interfaces by calling back into the Wolfram Language. In short, it lets you create a Java object that implements a given Java interface entirely in Mathematica code. The handlerMappings argument is a list of rules that specify the name of the Java method and the name of the method that will be called in Mathematica to implement the body of that method, as in \"intfMeth1\"->\"mathHandlerMethod1\". The arguments passed to the Mathematica method will be exactly the arguments originally passed to the Java method."

JavaWindowToFront::usage =
"JavaWindowToFront is an internal symbol."


Begin["`Package`"]

$currentClassPath

clearComplexClass
osIsWindows
osIsClassicMac
osIsMacOSX

isPreemptiveKernel
hasFrontEnd
hasServiceFrontEnd
hasLocalFrontEnd

executionProtect

deleteDuplicates

(* Exposed for the benefit of .NET/Link. Not used outside this file by J/Link. *)
registerWindow
unregisterWindow
windowID
unregisterAllWindows
windowRegisteredQ
getNextWindowID

(* Called from Java. *)
titleChangedFunc

(* Experimental utility func. *)
asyncEvaluate

(* Find the class file for a given class *)
locateClass

contextIndependentOptions

End[]  (* `Package` *)


(* Current context will be JLink`. *)

Begin["`Misc`Private`"]



(************************************  Funcs that call Java directly  **************************************)

SameObjectQ::obj = "At least one argument to SameObjectQ was not a valid reference to a Java object or Null."

(* Wrap in TrueQ to ensure T/F result even on exception. *)
SameObjectQ[obj1_?JavaObjectQ, obj2_?JavaObjectQ] :=
	Module[{jvm1 = jvmFromInstance[obj1], jvm2 = jvmFromInstance[obj2]},
		TrueQ[jvm1 === jvm2 && jSameQ[jvm1, obj1, obj2]]
	]

SameObjectQ[_, _] := (Message[SameObjectQ::obj]; False)


SetComplexClass[clsName_String] := SetComplexClass[GetJVM[InstallJava[]], clsName]

SetComplexClass[jvm_JVM, clsName_String] :=
	Module[{cls = LoadJavaClass[jvm, clsName, Null, True]},
		If[Head[cls] === JavaClass,
			SetComplexClass[jvm, cls],
		(* else *)
			$Failed
		]
	]

(* It is too expensive to call into Java to get the complex class, so I have to store it in
   Mathematica as well. This means that user must not let the values stored in Java and Mathematica
   get out of sync. The only way to do that would be to manually call setComplexClass() in Java.
*)

SetComplexClass[jc_JavaClass] := SetComplexClass[getDefaultJVM[], jc]

SetComplexClass[jvm_JVM, jc_JavaClass] :=
	If[checkJVM[jvm],
		If[jSetComplex[jvm, classIDFromClass[jc]] =!= Null,
			$complexClass =.;
			$Failed,
		(* else *)
			$complexClass = jc;
		],
	(* else *)
		$Failed
	]

GetComplexClass[] := If[ValueQ[$complexClass], $complexClass, Null]

clearComplexClass[] := Clear[$complexClass]

Internal`SetValueNoTrack[$complexClass, True]


GetClassPath = JavaClassPath  (* GetClassPath is deprecated. *)

JavaClassPath[] := JavaClassPath[GetJVM[InstallJava[]]]

JavaClassPath[jvm_JVM] :=
	If[checkJVM[jvm],
		Module[{res = jClassPath[jvm]},
			If[osIsWindows[],
				(* On Windows, files and dirs come back looking like /C:/path/to/dir/. Fix these up to avoid confusing users, and
				   to put entries into a format Mathematica functions can understand.
				*)
				res = If[StringMatchQ[#, "/*:*"], StringDrop[StringReplace[#, "/" -> "\\"], 1], #]& /@ res
			];
			res
		],
	(* else *)
		$Failed
	]


(* $currentClassPath allows us to keep track of additions to the classpath by AddToClassPath[]
   so they can be restored if Java is restarted within a kernel session.
*)
If[!ValueQ[$currentClassPath], $currentClassPath = {}]

Options[AddToClassPath] = {Prepend->False}

AddToClassPath[locs:(_String | _File).., opts:OptionsPattern[]] := AddToClassPath[GetJVM[InstallJava[]], {locs}, opts]
AddToClassPath[locs:{(_String | _File)...}, opts:OptionsPattern[]] := AddToClassPath[GetJVM[InstallJava[]], locs, opts]
AddToClassPath[jvm_JVM, locs:(_String | _File).., opts:OptionsPattern[]] := AddToClassPath[jvm, {locs}, opts]
AddToClassPath[jvm_JVM, locs:{(_String | _File)...}, OptionsPattern[]] :=
    If[checkJVM[jvm],
        jAddToClassPath[jvm, expandTilde /@ (locs /. File[s_] :> s), True, TrueQ[OptionValue[Prepend]]];
        $currentClassPath = JavaClassPath[jvm],
    (* else *)
        $Failed
    ]


(* $ExtraClassPath is deprecated, but still works. *)
If[!MatchQ[$ExtraClassPath, {___String}],
	$ExtraClassPath = {}
]

(* Java does not understand the ~ char in file paths, so help the user out by expanding it before the paths are sent
   to Java. Note that we could call ExpandFileName on all paths, and thus have Mathematica resolve partial pathnames
   and . and .. relative to its current directory. When Java gets these it resolves them relative to user.home, so it
   would potentially break user code to change it now. But having Mathematica resolve them seems clearly the correct behavior.
*)
expandTilde[s_String] := If[StringStartsQ[s, "~"], ExpandFileName[s], s]


(* Call with either a Java exception object or the string name of the exception class you want to throw. In the first case, supply
   no second arg (it will be ignored); in the second case, the second arg is an optional detail message for the exception.
   It is up to the user to ensure that the exception being created has an appropriate zero- or one-arg constructor.
   This function is still experimental.
*)
JavaThrow[exc_?JavaObjectQ] := jThrow[jvmFromInstance[exc], exc, ""]
JavaThrow[exc_String, msg_String:""] := jThrow[getDefaultJVM[], exc, msg]


PeekClasses = LoadedJavaClasses
PeekObjects = LoadedJavaObjects

LoadedJavaClasses[] := LoadedJavaClasses[getDefaultJVM[]]

LoadedJavaClasses[jvm:(_JVM | Null)] :=
	If[checkJVM[jvm],
		classFromID /@ jPeekClasses[jvm],
	(* else *)
		$Failed
	]

LoadedJavaObjects[] := LoadedJavaObjects[getDefaultJVM[]]

LoadedJavaObjects[jvm:(_JVM | Null)] :=
	If[checkJVM[jvm],
		jPeekObjects[jvm],
	(* else *)
		$Failed
	]



(*********************************************  Public utility funcs  ***********************************************)

SetAttributes[ClassName, Listable]

ClassName[Null] = Null   (* Should this issue a message? *)

ClassName[cls_JavaClass] := classNameFromClass[cls]

ClassName[obj_?JavaObjectQ] := classNameFromClass[classFromInstance[obj]]


SetAttributes[ParentClass, Listable]

ParentClass[Null] = Null   (* Should this issue a message? *)

ParentClass[obj_?JavaObjectQ] := ParentClass[GetClass[obj]]

ParentClass[cls_String] := ParentClass[LoadJavaClass[cls]]

ParentClass[cls_JavaClass] :=
	If[# === Null,
		(* cls was java.lang.Object *)
		Null,
	(* else *)
		classFromID[#]
	]& [parentClassIDFromClass[cls]]


SetAttributes[GetClass, Listable]

GetClass[Null] = Null   (* Should this issue a message? *)

GetClass[obj_?JavaObjectQ] := classFromInstance[obj]


InstanceOf::obj = "`1` is not a valid Java or .NET object reference."
InstanceOf::cls = "Invalid class specification `1`."

InstanceOf[Null, _String | _JavaClass] = False
InstanceOf[obj_?JavaObjectQ, cls_JavaClass] := InstanceOf[obj, classNameFromClass[cls]]
InstanceOf[obj_?JavaObjectQ, cls_String] := TrueQ[jInstanceOf[jvmFromInstance[obj], obj, cls]]
InstanceOf[obj_, cls_] :=
    Module[{isJava, isNET},
        isJava = JavaObjectQ[obj];
        isNET = NETLink`NETObjectQ[obj];
        If[!isJava && !isNET,
            Message[InstanceOf::obj, obj],
        (* else *)
            Message[InstanceOf::cls, cls]
        ];
        False
    ]


(********************************************  ImplementJavaInterface  **********************************************)

ImplementJavaInterface::jdk13 =
"The ImplementJavaInterface function requires JDK 1.3 or later. Your Java version does not appear to be modern enough."

ImplementJavaInterface[intfs:{__String}, handlerMappings__Rule] := ImplementJavaInterface[GetJVM[InstallJava[]], intfs, {handlerMappings}]
ImplementJavaInterface[intf_String, handlerMappings__Rule] := ImplementJavaInterface[GetJVM[InstallJava[]], {intf},{handlerMappings}]
ImplementJavaInterface[intf_String, handlerMappings:{__Rule}] := ImplementJavaInterface[GetJVM[InstallJava[]], {intf}, handlerMappings]
ImplementJavaInterface[jvm_JVM, intfs:{__String}, handlerMappings__Rule] := ImplementJavaInterface[jvm, intfs, {handlerMappings}]
ImplementJavaInterface[jvm_JVM, intf_String, handlerMappings__Rule] := ImplementJavaInterface[jvm, {intf},{handlerMappings}]
ImplementJavaInterface[jvm_JVM, intf_String, handlerMappings:{__Rule}] := ImplementJavaInterface[jvm, {intf}, handlerMappings]

ImplementJavaInterface[jvm_JVM, intfs:{__String}, handlerMappings:{__Rule}] :=
	Module[{invHandler, interfaces, proxy},
		UseJVM[jvm,
			If[LoadJavaClass["java.lang.reflect.Proxy"] === $Failed,
				Message["ImplementJavaInterface::jdk13"];
				Return[$Failed]
			];
			LoadJavaClass["com.wolfram.jlink.JLinkClassLoader"];
			JavaBlock[
				interfaces = Symbol["com`wolfram`jlink`JLinkClassLoader`classFromName"] /@ intfs;
				If[!(And @@ (JavaObjectQ /@ interfaces)),
					(* Error messages will already have been issued by classFromName(). *)
					Return[$Failed]
				];
				invHandler = JavaNew["com.wolfram.jlink.MathInvocationHandler", handlerMappings /. Rule->List];
				Proxy`newProxyInstance[interfaces[[1]]@getClassLoader[], interfaces, invHandler]
			]
		]
	]


(*************************************  SetInternetProxy  ************************************************)

(* J/Link's SetInternetProxy is deprecated in favor of the one in PacletManager`. But we still need
   to support it for users who are doing their own "raw" J/Link programming with java.net calls
   to access the internet. After we perform the original J/Link steps (setting various system props),
   we go on to call the PacletManager` version, which configures  the Apache HttpClient library
   we use internally for built-in internet functionality.
*)

SetInternetProxy[host_String, port_Integer] := SetInternetProxy[GetJVM[InstallJava[]], host, port]

SetInternetProxy[jvm_JVM, host_String, port_Integer] :=
	JavaBlock[
		LoadJavaClass[jvm, "java.lang.System"];
		java`lang`System`getProperties[]@put[MakeJavaObject[jvm, "proxySet"], MakeJavaObject[jvm, "true"]];
		java`lang`System`getProperties[]@put[MakeJavaObject[jvm, "proxyHost"], MakeJavaObject[jvm, host]];
		java`lang`System`getProperties[]@put[MakeJavaObject[jvm, "proxyPort"], MakeJavaObject[jvm, ToString[port]]];
		PacletManager`SetInternetProxy["HTTP", {host, port}];
	]

SetInternetProxy[protocol_String, {host_String, port_}] := PacletManager`SetInternetProxy[protocol, {host, port}];


(********************************************  AppletViewer  ************************************************)

AppletViewer[cls:(_String | _JavaClass)] := AppletViewer[cls, {}]

AppletViewer[cls:(_String | _JavaClass), params:{___String}] := AppletViewer[GetJVM[InstallJava[]], cls, params]

AppletViewer[jvm_JVM, cls:(_String | _JavaClass), params:{___String}] :=
	JavaBlock[
		AppletViewer[JavaNew[jvm, cls], params]
	]

AppletViewer[applet_?JavaObjectQ] := AppletViewer[applet, {}]

AppletViewer[applet_?JavaObjectQ, params:{___String}] :=
	JavaBlock[
		JavaNew[jvmFromInstance[applet], "com.wolfram.jlink.MathAppletFrame", applet, params] // JavaShow;
	]


(************************************************  DoModal  *****************************************************)

(* DoModal enters a loop that waits until the Java side sends back
		EvaluatePacket[EndModal[...args for event listener...]]
   The loop it runs is capable of servicing normal EvaluatePacket requests. It is really a modified
   version of the ExternalAnswer function that handles reads during a normal call into Java.
*)

DoModal::preempt = "DoModal cannot execute when another call to Java is currently in progress."

DoModal[] := DoModal[getDefaultJVM[]]

DoModal[jvm_JVM] := doModal[getActiveJavaLink[jvm], jvm]

(* This sig for .NET/Link. *)
DoModal[link_LinkObject] := doModal[link, Null]

(* EndModal should not be called with arguments, as they will be ignored, but I want to make sure that it
   ends the loop even if users call it incorrectly.
*)
EndModal[___] := (endModal = True;)

doModal[link_LinkObject, jvm_] :=
	AbortProtect[
		Block[{result = Null, endModal = False, e},
			If[link === javaPreemptiveLink[jvm],
				(* For various reasons, DoModal is not allowed on the preemptive link. For one thing,
				   we don't want to hang up the whole preemption system while a modal Java dialog is up.
				*)
				Message[DoModal::preempt];
				Return[$Failed]
			];
			If[link === JavaLink[jvm] || link === JavaUILink[jvm], jAllowUIComputations[jvm, True, True]];
	    	While[!TrueQ[endModal],
				e = LinkReadHeld[link];
				JAssert[MatchQ[e, Hold[EvaluatePacket[_]]]];
				(* This 2+2 eval is a hack to absorb any aborts that might be pending here,
				   so they do not interfere with a call to EndModal[] that might occur during
				   the evaluation of e.
				*)
				CheckAbort[2+2, $Aborted];
				result = CheckAbort[e[[1,1]], $Aborted];
				If[LinkWrite[link, ReturnPacket[result]] === $Failed,
					Return[$Failed]
				]
			];
			If[link === JavaLink[jvm], jAllowUIComputations[jvm, False, False]];
			result
		]
	]


(* This allows a single computation originating from the Java UI thread (or any other non-Reader thread)
   to proceed.
*)
ServiceJava[] := ServiceJava[getDefaultJVM[]]

ServiceJava[jvm_JVM] :=
	If[isPreemptiveKernel[] && Head[JavaUILink[jvm]] === LinkObject,
		(* A no-op for preemptive kernel, because comps come in on the UI link. Because
		   we have taken out a read/write to Java (in myLinkReadyQ), we have lost some yielding
		   to Java, and programs that use ServiceJava probably keep the kernel busy to the detriment
		   of the Java UI thread. Therefore we Pause here for a tiny bit in an attempt to make
		   legacy programs written with ServiceJava behave as responsively as they used to.
		   Programs targeted at 5.1 and later have no need to call ServiceJava, and they can Pause
		   manually if desired to maintain smoothness in Java, say for an animation.
		*)
		Pause[.01],
	(* else *)
		AbortProtect[
			Module[{e, result, jl = JavaLink[jvm]},
				If[myLinkReadyQ[jl],
					jAllowUIComputations[jvm, True, False];
					e = LinkReadHeld[jl];
					JAssert[MatchQ[e, Hold[EvaluatePacket[_]]]];
					result = CheckAbort[e[[1,1]], $Aborted];
					LinkWrite[jl, ReturnPacket[result]]
				]
			]
		]
	]


(*******************************************  JavaShow  ***********************************************)

JavaShow::wnd = "The object passed to JavaShow must be a subclass of java.awt.Window."

JavaShow[windowObj_?JavaObjectQ] :=
	JavaBlock[
		Module[{id, title, jvm},
			If[!InstanceOf[windowObj, "java.awt.Window"],
				Message[JavaShow::wnd];
				Return[$Failed]
			];
			jvm = jvmFromInstance[windowObj];
			jShow[jvm, windowObj];
			If[!windowRegisteredQ[windowObj] && hasLocalFrontEnd[] && hasServiceFrontEnd[],
				id = If[osIsWindows[], jGetWindowID[jvm, windowObj], getNextWindowID[]];
				If[!TrueQ[id > 0],
					(* Failure in JAWT code to get HWND. Fail silently. *)
					Return[]
				];
				If[InstanceOf[windowObj, "java.awt.Frame"],
					title = windowObj@getTitle[]
				];
				If[SymbolQ[title] || title == "", title = "Java Window"];
				registerWindow[windowObj, title, "Java", id, jvm];
				windowObj@addWindowListener[JavaNew[jvm, "com.wolfram.jlink.MathWindowListener", {{"windowClosing", ToString[windowClosingFunc]}}]];
				jAddTitleChangeListener[jvm, windowObj, ToString[titleChangedFunc]];
				KeepJavaObject[windowObj, Manual]
			];
		]
	]


windowClosingFunc[evt_] :=
	JavaBlock[
		unregisterWindow[evt@getWindow[]];
		ReleaseJavaObject[evt];
	]
	
	
titleChangedFunc[windowObj_, newTitle_] :=
	Module[{windowId = windowID[windowObj], jvm = windowJVM[windowObj]},
	    If[windowId > 0,
	        unregisterWindow[windowObj];
	        registerWindow[windowObj, newTitle, "Java", windowId, jvm]
	    ];
	]


(* Called by FE when user selects Java window from Window menu. To keep the FE responsive,
   this is written so as not to block. Otherwise, it would hang the FE if the user selected
   a Java window from the FE Window menu while a long Java computation was going on
   (calls on JavaLink[] cannot preempt each other, so the call to jShow[] would block
   until the previous Java computation was finished).
*)
JavaWindowToFront[id_Integer] :=
	asyncEvaluate[
		Module[{windowList},
			windowList = Cases[registeredWindows[], {obj_, id, title_, "Java", ___} -> obj];
			If[Length[windowList] > 0,
				JavaShow[First[windowList]]
			]
		]
	]


(* The following funcs deal with registering windows with the FE so they
   appear in its Windows menu. These funcs are used by .NET/Link also.
*)

(* Returns False if window is already registered. *)
registerWindow[obj_, title_String, type_String, id_Integer, extra___] :=
	If[!windowRegisteredQ[obj],
        MathLink`CallFrontEnd[FrontEnd`AddMenuCommands["MenuListWindows", {MenuItem[title, KernelExecute[JavaWindowToFront[id]], MenuEvaluator -> Automatic]}]];
		AppendTo[$registeredWindows, {obj, id, title, type, extra}];
		True,
	(* else *)
		(* Already registered. *)
		False
	]

unregisterWindow[obj_] :=
	Module[{windowRec},
		windowRec = Cases[$registeredWindows, {obj, __}];
		If[Length[windowRec] > 0,
			(* Should always have length 1. *)
			windowRec = First[windowRec];
			With[{id = windowRec[[2]], title = windowRec[[3]]},
			    If[hasServiceFrontEnd[],
				    MathLink`CallFrontEnd[FrontEnd`RemoveMenuCommands["MenuListWindows", {MenuItem[title, KernelExecute[JavaWindowToFront[id]], MenuEvaluator -> Automatic]}]]
			    ]
			];
			$registeredWindows = DeleteCases[$registeredWindows, {obj, __}]
		]
	]

(* Type will be "Java" or ".NET". This func is called in UninstallJava/NET. *)
unregisterAllWindows[type_String, extra___] :=
	unregisterWindow /@ First /@ Cases[registeredWindows[], {_, _, _, type, extra}]

windowRegisteredQ[obj_] := MemberQ[First /@ registeredWindows[], obj]

windowID[obj_] := If[windowRegisteredQ[obj], First[Cases[registeredWindows[], {obj, id_, __} -> id]], -1]

windowTitle[obj_] := If[windowRegisteredQ[obj], First[Cases[registeredWindows[], {obj, id_, title_, __} -> title]], ""]

windowJVM[obj_] := If[windowRegisteredQ[obj], First[Cases[registeredWindows[], {obj, id_, title_, "Java", jvm_, ___} -> jvm]], Null]

getNextWindowID[] := $nextWindowID++

registeredWindows[] := $registeredWindows

If[!ValueQ[$nextWindowID], $nextWindowID = 1]
If[!ValueQ[$registeredWindows], $registeredWindows = {}]

Internal`SetValueNoTrack[$nextWindowID, True]
Internal`SetValueNoTrack[$registeredWindows, True]

(**********************************  locateClass  **************************************)

(* A simple utility function that gives the location on disk of the class file that a given class is coming from. *)

locateClass[obj_?JavaObject] := JavaBlock[obj@getClass[]@getProtectionDomain[]@getCodeSource[]getLocation[]@toString[]]

locateClass[class_String] := 
    JavaBlock[
        Module[{cls},
            LoadJavaClass["com.wolfram.jlink.JLinkClassLoader"];
            cls = JLinkClassLoader`classFromName[class];
            If[cls =!= Null && JavaObjectQ[cls],
                cls@getProtectionDomain[]@getCodeSource[]@getLocation[]@toString[],
            (* else *)
                $Failed
            ]
        ]
    ]


(**********************************  asyncEvaluate  **************************************)

(* Evaluates an expr at a later time. The expr is evaluated non-preemptively and treated like input
   arriving on a non-preemptive sharing link.
*)

SetAttributes[asyncEvaluate, HoldFirst]

asyncEvaluate[e_] :=
	With[{loop = LinkOpen[LinkMode->Loopback]},
		LinkWriteHeld[loop, Hold[e]];
        LinkWriteHeld[loop, Hold[MathLink`RemoveSharingLink[loop]]];
        LinkWriteHeld[loop, Hold[LinkClose[loop]]];
		MathLink`AddSharingLink[loop];
	]


(*******************************************  ShowJavaConsole  ***********************************************)

ShowJavaConsole[] := ShowJavaConsole[GetJVM[InstallJava[]], "stdout", "stderr"]

ShowJavaConsole[None] := ShowJavaConsole[GetJVM[InstallJava[]], "none"]

ShowJavaConsole[jvm_JVM] := ShowJavaConsole[jvm, "stdout", "stderr"]

ShowJavaConsole[strms__String] := ShowJavaConsole[GetJVM[InstallJava[]], strms]

ShowJavaConsole[jvm_JVM, strms__String] :=
	If[checkJVM[jvm],
		JavaBlock[
			Module[{console},
				console = jGetConsole[jvm];
				If[console@isFirstTime[],
					console@setSize[450, 400];
					console@setLocation[100, 100];
					console@setFirstTime[False]
				];
				(* Plus should really be BitOr, but I don't want to introduce an unnecessary M-- 3.x incompatibility. *)
				console@setCapture[Plus @@ (Union[{strms}] /.
					{"stdout" -> Symbol["com`wolfram`jlink`ui`ConsoleWindow`STDOUT"],
					 "stderr" -> Symbol["com`wolfram`jlink`ui`ConsoleWindow`STDERR"],
					 "none" -> Symbol["com`wolfram`jlink`ui`ConsoleWindow`NONE"]})
				];
				JavaShow[console];
				console
			]
		],
	(* else *)
		$Failed
	]


(*****************************************  Platform tests  *******************************************)

osIsWindows[] = StringMatchQ[$System, "*Windows*"]

osIsClassicMac[] = StringMatchQ[$System, "*Macintosh*"]

osIsMacOSX[] = StringMatchQ[$System, "*Mac*X*"]

isPreemptiveKernel[] = $VersionNumber >= 5.1

hasFrontEnd[] := Head[$FrontEnd] === FrontEndObject

hasServiceFrontEnd[] := Head[MathLink`$ServiceLink] === LinkObject

hasLocalFrontEnd[] := hasFrontEnd[] && MathLink`CallFrontEnd[FrontEnd`Value["$MachineID"]] === $MachineID

(**************************************  Misc  ****************************************)


SetAttributes[executionProtect, {HoldFirst}]

executionProtect[e_] := AbortProtect[PreemptProtect[e]]


If[$VersionNumber >= 6.1,
    deleteDuplicates = DeleteDuplicates,
(* else *)
    deleteDuplicates[lis_List] := Tally[lis][[All, 1]]
]


(***********************************  contextIndependentOptions  **************************************)

(* This processes the names of options in a context-independent way.
*)
contextIndependentOptions[optName_Symbol, opts_List, defaults_List] :=
    First[ contextIndependentOptions[{optName}, opts, defaults] ]

contextIndependentOptions[optNames_List, opts_List, defaults_List] :=
    Module[{optNameStrings, stringifiedOptionSettings, stringifiedOptionDefaults},
        optNameStrings = (# /. x_Symbol :> SymbolName[x])& /@ optNames;
        stringifiedOptionSettings = MapAt[(# /. x_Symbol :> SymbolName[x])&, #, {1}]& /@ Flatten[{opts}];
        stringifiedOptionDefaults = MapAt[(# /. x_Symbol :> SymbolName[x])&, #, {1}]& /@ Flatten[{defaults}];
        optNameStrings /. stringifiedOptionSettings /. stringifiedOptionDefaults
    ]


End[]