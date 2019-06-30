(* :Title: Debug.m *)

(* :Author:
        Todd Gayley
        tgayley@wolfram.com
*)

(* :Package Version: 4.9 *)

(* :Mathematica Version: 4.0 *)
		     
(* :Copyright: J/Link source code (c) 1999-2019 Wolfram Research, Inc. All rights reserved.

   Use is governed by the terms of the J/Link license agreement, which can be found at
   www.wolfram.com/solutions/mathlink/jlink.
*)

(* :Discussion:
   Debugging utilities used within J/Link.
	
   This file is a component of the J/Link Mathematica source code.
   It is not a public API, and should never be loaded directly by users or programmers.

   J/Link uses a special system wherein one package context (JLink`) has its implementation
   split among a number of .m files. Each component file has its own private context, and also
   potentially introduces public symbols (in the JLink` context) and so-called "package" symbols,
   where the term "package" comes from Java terminology, referring to symbols that are visible
   everywhere within the implementation of J/Link, but not to clients.
*)



Begin["`Package`"]

$Debug
$DebugLevel
$DebugExceptions
$DebugDups
$DebugCommandLine
DebugPrint
JAssert
JAssertAction
$JAssertAction
Verify
VerifyAction
$VerifyAction
Trigger

End[]  (* `Package` *)


(* Current context will be JLink`. *)

Begin["`Debug`Private`"]

$Debug = False
$DebugLevel = 0
$DebugExceptions = False
$DebugDups = False
If[!ValueQ[$DebugCommandLine], $DebugCommandLine = False]
SetAttributes[DebugPrint, HoldAll]
DebugPrint[e__, Trigger :> tst_] := Print[e] /; ($Debug && tst)
DebugPrint[e__, Trigger :> tst_] = Null
DebugPrint[lev_Integer, e__] := Print[e] /; ($Debug && $DebugLevel >= lev)
DebugPrint[lev_Integer, e__] = Null
DebugPrint[e__] := Print[e] /; $Debug
DebugPrint[e__] = Null
SetAttributes[JAssert, HoldAll]
Options[JAssert] = {JAssertAction :> $JAssertAction}
JAssert[exprs__, action___?OptionQ] :=
	StackInhibit[
		Scan[Function[e, If[e =!= True, (JAssertAction /. {action} /. Options[JAssert])[e]], HoldAll], Hold[exprs]]
	] /; $Debug
JAssert[___] = Null
$JAssertAction = Function[e, Print["Assertion failed: ", HoldForm[e]]; (* Dialog[] *), HoldAll]
SetAttributes[Verify, HoldAll]
Options[Verify] = {VerifyAction :> $VerifyAction}
Verify[expr_, test_, action___?OptionQ] :=
	StackInhibit[
		(If[test[#] =!= True, (VerifyAction /. {action} /. Options[Verify])[expr, test]]; #)& [expr]
	] /; $Debug
Verify[expr_, __] := expr
$VerifyAction = Function[{e, t},
					   Print["Verify failed. Test was: ", t, "\nExpression was: ", HoldForm[e]]; Dialog[],
					   HoldAll
			   ]

End[]
