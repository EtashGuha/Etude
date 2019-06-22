(* ::Package:: *)

(* ::Title:: *)
(*RobotExecute*)

(* ::Section:: *)
(*Annotations*)

(* :Title: RobotExecute.m *)

(* :Author:
        Brenton Bostick
        brenton@wolfram.com
*)

(* :Package Version: 1.2 *)

(* :Mathematica Version: 7.0 *)

(* :Copyright: RobotTools source code (c) 2005-2008 Wolfram Research, Inc. All rights reserved. *)

(* :Discussion:
   Implementation of RobotExecute functionality.
*)

(* ::Section:: *)
(*Information*)

`Information`CVS`$RobotExecuteId = "$Id: RobotExecute.m,v 1.19 2012/01/29 16:45:14 brenton Exp $"

(* ::Section:: *)
(*Public*)

(* ::Section:: *)
(*Package*)

Begin["`Package`"]

CalledByItemActivate

CallInstallRobotTools

reapHeldList

reapString

RobotBlock

RobotExecute

throwIfMessage

End[] (*`Package`*)

(* ::Section:: *)
(*Private*)

(* ::Subsection:: *)
(*Begin*)

Begin["`RobotExecute`Private`"]

(* ::Subsection:: *)
(*Messages*)

RobotExecute::badcmds =
"Commands given to RobotExecute did not execute: `1`"

(* ::Subsection:: *)
(*throwIfMessage*)

Attributes[throwIfMessage] = {HoldAll}

throwIfMessage[expr_] :=
	Check[
		expr
		,
		Throw[$Failed]
	]

(* ::Subsection:: *)
(*reap*)

(* reapHeldList is just a convenience function that gets rid of the [[2, 1]] *)

Attributes[reapHeldList] = {HoldFirst}

(* the DeleteCases[..., Null] and Sow[Null] is to properly handle whenever NOTHING gets sown and Reap[expr][[2]] returns {} *)
reapHeldList[expr_] :=
	DeleteCases[
		Reap[
			Sow[Null];
			expr
			,
			None
			,
			Hold[#2]&
		][[2, 1]]
		,
		Null
	]


Attributes[reapString] = {HoldFirst}

(* the Sow[""] below is similar to the Sow[Null] above *)
reapString[expr_] :=
	StringJoin[
		Reap[
			Sow[""];
			expr
			,
			None
			,
			#2&
		][[2, 1]]
	]

(* ::Subsection:: *)
(*RobotBlock*)

Unprotect[RobotBlock]

(*
the symbols listed in RobotBlock are either high-level functions that evaluate low-level functions as side-effects, or low-level
functions that call robot methods as side-effects
*)

Attributes[RobotBlock] = {HoldFirst}

RobotBlock[expr_] :=
	Block[{
		(* high-level Null-returning functions with side-effects *)
		AppleScriptExecute, Delay, KeyPress, KeyRelease, KeyType, MouseClick, MouseDoubleClick, MouseDrag,
		MouseMove, MousePress, MouseRelease, MouseSingleClick, MouseTripleClick, MouseWheel,
		SetAutoDelay, SetAutoWaitForIdle, SetFocusedNotebook, WaitForIdle,
		(* high-level impure functions with no side-effects *)
		BoxInformation, GetAutoDelay, GetBoxRectangles, GetPixelColor, GetSelectionBoundingBoxes, GetWindowRectangle,
		IsAutoWaitForIdle, NotebookImage, RasterizeNotebook, RasterizeScreenShot, RasterizeSelection, ScreenShot,
		(* low-level referentially transparent functions, so keep them unBlocked.
		However, note that keyType and mouseClick return expressions that are not referentially transparent,
		but that are still blocked here *)
		(* clickMenuItemScript, keyType, mouseClick, *)
		(* low-level Null-returning functions with side-effects *)
		appleScriptExecute, delay, iaMouse, keyPress, keyRelease, mouseMove, mousePress, mouseRelease,
		mouseWheel, setAutoDelay, setAutoWaitForIdle, setFocusedNotebook, waitForIdle,
		(* low-level impure functions with no side-effects *)
		boxInformation, getAutoDelay, getBoxRectangles, getPixelColor, getSelectionBoundingBoxes, getWindowRectangle,
		isAutoWaitForIdle, mousePosition, notebookImage, rasterizeScreenShot, screenShot}
		,
		expr
	]

SetAttributes[RobotBlock, ReadProtected]

Protect[RobotBlock]

(* ::Subsection:: *)
(*RobotExecute*)

Unprotect[RobotExecute]

Options[RobotExecute] = {CallInstallRobotTools -> True, CallSetFocusedNotebook -> True, CalledByItemActivate -> False}

RobotExecute[nb:focusedNotebookPat:FocusedNotebook[], buffer:blankPat, OptionsPattern[]] :=
	Module[{callInstallRobotTools, callSetFocusedNotebook, calledByItemActivate},
		Catch[
			throwIfMessage[
				{callInstallRobotTools, callSetFocusedNotebook, calledByItemActivate} = OptionValue[{CallInstallRobotTools, CallSetFocusedNotebook, CalledByItemActivate}]
			];
			throwIfMessage[
				If[callInstallRobotTools,
					InstallRobotTools[];
					setAutoDelay[$AutoDelay];
					setAutoWaitForIdle[$AutoWaitForIdle]
				]
			];
			throwIfMessage[
				If[callSetFocusedNotebook,
					(*
					restore this code if there is funkiness with SetSelectedNotebook on X
					If[calledByItemActivate && $InterfaceEnvironment == "X",
						SetFocusedNotebook[nb]
						,
						SetSelectedNotebook[nb]
					]*)
					SetSelectedNotebook[nb]
				]
			];
			throwIfMessage[
				CompoundExpression @@ ReleaseHold[buffer]
			]
		]
	]

SetAttributes[RobotExecute, ReadProtected]

RobotExecute[args___] :=
	(ArgumentCountQ[RobotExecute, System`FEDump`NonOptionArgCount[{args}], 1, 2]; $Failed)

SyntaxInformation[RobotExecute] = {"ArgumentsPattern" -> {_., _, OptionsPattern[]}}

Protect[RobotExecute]

(* ::Subsection:: *)
(*End*)

End[] (*`RobotExecute`Private`*)
