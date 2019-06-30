(* :Title: JVMs.m *)

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
   Functions for managing multiple JVMs.

   This file is a component of the J/Link Mathematica source code.
   It is not a public API, and should never be loaded directly by users or programmers.

   J/Link uses a special system wherein one package context (JLink`) has its implementation
   split among a number of .m files. Each component file has its own private context, and also
   potentially introduces public symbols (in the JLink` context) and so-called "package" symbols,
   where the term "package" comes from Java terminology, referring to symbols that are visible
   everywhere within the implementation of J/Link, but not to clients.
*)

UseJVM::usage = "UseJVM[jvm, body] acts like a wrapper that causes all J/Link calls in it s body to use the specified JVM as the default Java runtime. UseJVM will only be used by advanced programmers who want to have more than one Java runtime installed into the Wolfram Language."

GetJVM::usage = "GetJVM[link] returns the JVM expression that corresponds to the given link, which was returned from InstallJava[]. GetJVM will only be used by advanced programmers who want to have more than one Java runtime installed into the Wolfram Language."

JVM::usage = "JVM is the head of an expression that identifies a particular Java runtime installed into the current Wolfram Language session via InstallJava[]."


Begin["`Package`"]

setDefaultJVM
getDefaultJVM

checkJVM

addJVM
removeJVM

nameFromJVM
javaLinkFromJVM
javaUILinkFromJVM
javaPreemptiveLinkFromJVM
initFileFromJVM
launchTimeFromJVM
installFinishedFromJVM
useExtraLinksFromJVM

setJVMInstallFinished
setJVMExtraLinks

createJVMName

End[]  (* `Package` *)


(* Current context will be JLink`. *)

Begin["`JVMs`Private`"]


JVM::uninst = "The Java runtime specified by `1` is no longer valid. It was previously shut down by a call to UninstallJava[]."

(* checkJVM is a utility function called by functions that checks the validity of the
   supplied JVM argument. It takes care of issuing a message.
*)

checkJVM[Null] := (Message[Java::init]; False)

checkJVM[jvm_JVM] :=
	If[StringQ[jvm["Name"]],
		True,
	(* else *)
		(* This JVM was removed via UninstallJava[]. *)
		Message[JVM::uninst, jvm];
		False
	]


(*************************  UseJVM/GetJVM  ****************************)

(* UseJVM is the main way to specify a JVM in J/Link Mathematica functions that
   do not operate on objects, and thus do not have an implied JVM. Examples
   include JavaNew, LoadJavaClass, ShowJavaConsole, MakeJavaObject, etc.
   UseJVM acts as a wrapper, much like Block. Within the body, all Mathematica
   functions that need to acquire a JVM from "thin air", as well as static
   Java methods, will use the given JVM.

   UseJVM is safe from preemption, meaning that preemptive code that interrupts
   UseJVM will use the normal default JVM unless it also calls UseJVM.
*)

SetAttributes[UseJVM, {HoldRest}]

UseJVM[jvm_JVM, expr_] :=
	If[checkJVM[jvm],
		If[TrueQ[MathLink`IsPreemptive[]],
			Block[{$defaultPreemptiveJVM = jvm}, expr],
		(* else *)
			Block[{$defaultJVM = jvm}, expr]
		],
	(* else *)
		$Failed
	]


(* GetJVM is the function that programmers use to acquire a JVM expression to be
   used in UseJVM or directly in J/Link Mathematica functions. The standard idiom
   is to call GetJVM on the result of InstallJava and store that value for your
   later use.
*)

(* link argument must be the JavaLink[] or JavaUILink[] for the desired JVM. *)
GetJVM[link_LinkObject] :=
	Module[{names},
		names = Cases[SubValues[JVM], (Verbatim[HoldPattern][JVM[name_]["JavaLink" | "JavaUILink"]] :> link) :> name];
		If[Length[names] == 1,
			JVM[First[names]],
		(* else *)
			(* Don't yet know if this will always be an error. *)
			Null
		]
	]

(* This needs to be fast, as it is called during createInstanceDefs[]. Don't worry about error checking,
   as we don't even document this signature (only documented for LinkObject arg). Only Java
   code that calls back to Mathematica in the internals of J/Link needs to deal with JVMs by name.
*)
GetJVM[name_String] := JVM[name]


(**********************************  End Public  *********************************)

Internal`SetValueNoTrack[$defaultJVM, True]
Internal`SetValueNoTrack[$defaultPreemptiveJVM, True]
Internal`SetValueNoTrack[$jvmIndex, True]
Internal`SetValueNoTrack[JVM, True]


setDefaultJVM[jvm_] := $defaultJVM = $defaultPreemptiveJVM = jvm

getDefaultJVM[] :=
	If[TrueQ[MathLink`IsPreemptive[]],
		$defaultPreemptiveJVM,
	(* else *)
		$defaultJVM
	]

If[!ValueQ[$defaultJVM], $defaultJVM = Null]
If[!ValueQ[$defaultPreemptiveJVM], $defaultPreemptiveJVM = Null]


If[!ValueQ[$jvmIndex], $jvmIndex = 1]

createJVMName[] := "vm" <> ToString[$jvmIndex++]


addJVM[name_String, link_LinkObject, ui_, pre_, initFile_, launchTime_, installFinished_, useExtraLinks_] :=
	executionProtect[
		JVM[name]["Name"] = name;
		JVM[name]["JavaLink"] = link;
		JVM[name]["JavaUILink"] = ui;
		JVM[name]["JavaPreemptiveLink"] = pre;
		JVM[name]["InitFile"] = initFile;
		JVM[name]["LaunchTime"] = launchTime;
		JVM[name]["InstallFinished"] = installFinished;
		JVM[name]["UseExtraLinks"] = useExtraLinks;
		JVM[name]
	]

removeJVM[jvm_JVM] :=
	executionProtect[
		(* Quiet because this can be called in situations when these rules have
		   not been set up yet, and that spits a bunch of ugly Unset warnings.
		*)
		Quiet[
			With[{name = jvm["Name"]},
				JVM[name]["Name"] =.;
				JVM[name]["JavaLink"] =.;
				JVM[name]["JavaUILink"] =.;
				JVM[name]["JavaPreemptiveLink"] =.;
				JVM[name]["InitFile"] =.;
				JVM[name]["LaunchTime"] =.;
				JVM[name]["InstallFinished"] =.;
				JVM[name]["UseExtraLinks"] =.
			];
			If[jvm === $defaultJVM, $defaultJVM = Null];
            If[jvm === $defaultPreemptiveJVM, $defaultPreemptiveJVM = Null];
		]
	]


(********  Accessors to extract properties from a JVM[] expression.  *********)

nameFromJVM[jvm_JVM] := jvm["Name"]
nameFromJVM[_] = Null

javaLinkFromJVM[jvm_JVM] := jvm["JavaLink"]
javaLinkFromJVM[_] = Null

javaUILinkFromJVM[jvm_JVM] := jvm["JavaUILink"]
javaUILinkFromJVM[_] = Null

javaPreemptiveLinkFromJVM[jvm_JVM] := jvm["JavaPreemptiveLink"]
javaPreemptiveLinkFromJVM[_] = Null

initFileFromJVM[jvm_JVM] := jvm["InitFile"]

launchTimeFromJVM[jvm_JVM] := jvm["LaunchTime"]

installFinishedFromJVM[jvm_JVM] := jvm["InstallFinished"]

useExtraLinksFromJVM[jvm_JVM] := jvm["UseExtraLinks"]


(********  Setters to modify properties of a JVM[] expression.  *********)

setJVMInstallFinished[jvm_JVM] :=
	jvm["InstallFinished"] = True

setJVMExtraLinks[jvm_JVM, uilink_, prelink_] :=
	(
		jvm["JavaUILink"] = uilink;
		jvm["JavaPreemptiveLink"] = prelink;
	)



End[]
