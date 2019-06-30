(* ::Package:: *)

(* ::Title:: *)
(*AppleScript*)

(* ::Section:: *)
(*Annotations*)

(* :Title: AppleScript.m *)

(* :Author:
        Brenton Bostick
        brenton@wolfram.com
*)

(* :Package Version: 1.2 *)

(* :Mathematica Version: 7.0 *)

(* :Copyright: RobotTools source code (c) 2005-2008 Wolfram Research, Inc. All rights reserved. *)

(* :Discussion:
   AppleScript functionality.
   This file uses classes described in
   http://developer.apple.com/reference/Cocoa/idxJava-date.html
   Even though the Java classes have been deprecated, I'm going to hold on to them as long as they are available. AppleScript through
   Java is very convenient!
*)

(* ::Section:: *)
(*Information*)

`Information`CVS`$AppleScriptId = "$Id: AppleScript.m,v 1.21 2009/09/08 17:09:25 brenton Exp $"

(* ::Section:: *)
(*Public*)

AppleScriptExecute::usage =
"AppleScriptExecute[\"script\"] executes the given AppleScript and returns the result.
AppleScriptExecute[nb, \"script\"] makes sure that nb is selected, and then executes script."

InstallAppleScript::usage =
"InstallAppleScript[] initializes AppleScript specific components."

(* ::Section:: *)
(*Package*)

Begin["`Package`"]

appleScriptExecute

clickMenuItemScript

End[] (*`Package`*)

(* ::Section:: *)
(*Private*)

(* ::Subsection:: *)
(*Begin*)

Begin["`AppleScript`Private`"]

(* ::Subsection:: *)
(*Messages*)

AppleScriptExecute::notmac =
"AppleScript cannot be run on a non-Macintosh machine: `1`"

AppleScriptExecute::errors =
"AppleScript error(s) occurred: `1`"

InstallAppleScript::uin =
"UI elements are not enabled. \
Under System Preferences, in the System Row, click on Universal Access. \
Make sure that the \"Enable access for assistive devices\" checkbox at the bottom is checked."

InstallAppleScript::uie =
"An unhandled error occurred. The result was: `1`"

(* ::Subsection:: *)
(*InstallAppleScript*)

checkIfUIElementsEnabled[] :=
	Module[{script, uiElementsEnabled},
		script = "tell application \"System Events\" to UI elements enabled";
		uiElementsEnabled = appleScriptExecute[script];
		Switch[uiElementsEnabled,
			"true",
			Null
			,
			"false",
			Message[InstallAppleScript::uin];
			$Failed
			,
			_,
			Message[InstallAppleScript::uie, uiElementsEnabled];
			$Failed
		]
	]

Unprotect[InstallAppleScript]

InstallAppleScript[] /; $InterfaceEnvironment === "Macintosh" :=
	Catch[
		throwIfMessage[
			checkIfUIElementsEnabled[]
		]
	]

SetAttributes[InstallAppleScript, ReadProtected]

Protect[InstallAppleScript]

(* ::Subsection:: *)
(*clickMenuItemScript*)

(* clickMenuItemScript returns an AppleScript string for clicking on a menu item *)
clickMenuItemScript[process:stringPat:"Mathematica", path:menuPathPat] :=
	Module[{script},
		script =
			StringJoin[
				"tell application \"System Events\" to tell process \"" <> process <> "\" to tell menu bar 1 to ",
				MapIndexed[
					Which[
						#2[[1]] == 1,
						"tell menu bar item \"" <> #1 <> "\" to tell menu \"" <> #1 <> "\" to "
						,
						#2[[1]] == Length[path],
						"click menu item \"" <> #1 <> "\""
						,
						True,
						"tell menu item \"" <> #1 <> "\" to tell menu \"" <> #1 <> "\" to "
					]&
					,
					path
				]
			];
		(* the return is to suppress any output from the tell command *)
		script = script <> " \n return";
		script
	]


(* ::Subsection:: *)
(*AppleScriptExecute*)

Unprotect[AppleScriptExecute]

AppleScriptExecute[nb:focusedNotebookPat:FocusedNotebook[], script:stringPat] /; $InterfaceEnvironment == "Macintosh" :=
	Module[{buffer},
		Catch[
			(* step 1: set variables and validate user input *)
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							Sow[appleScriptExecute[script]]
						]
				]
			];
			(* step 3: execute buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer]
			]
		]
	]

SetAttributes[AppleScriptExecute, ReadProtected]

AppleScriptExecute[arg_] :=
	(Message[AppleScriptExecute::string, 1, HoldForm[AppleScriptExecute[arg]]]; $Failed)

AppleScriptExecute[args___] :=
	(ArgumentCountQ[AppleScriptExecute, System`FEDump`NonOptionArgCount[{args}], 1, 2]; $Failed)

SyntaxInformation[AppleScriptExecute] =
	{"ArgumentsPattern" -> {_., _}}

Protect[AppleScriptExecute]

(* ::Subsection:: *)
(*Low-Level Functions*)

(*
currently, any scripts must suppress output, say by using return at the end of the script
this is to allow errors to be detected
*)
appleScriptExecute[script:stringPat] :=
	Module[{im},
		(*
		currently have to ignore what osascript returns, but in the future there
		will be better error handling
		*)
		im = Import["!osascript -e '" <> script <> "' -s o 2>&1", "Lines"];
		If[im == {},
			Null
			,
			Message[AppleScriptExecute::errors, im];
			$Failed
		]
	]

(* ::Subsection:: *)
(*End*)

End[] (*`AppleScript`Private`*)
