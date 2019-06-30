(* ::Package:: *)

(* ::Title:: *)
(*Delay*)

(* ::Section:: *)
(*Annotations*)

(* :Title: Delay.m *)

(* :Author:
        Brenton Bostick
        brenton@wolfram.com
*)

(* :Package Version: 1.2 *)

(* :Mathematica Version: 7.0 *)

(* :Copyright: RobotTools source code (c) 2005-2008 Wolfram Research, Inc. All rights reserved. *)

(* :Discussion:
   Delay-related functionality.
*)

(* ::Section:: *)
(*Information*)

`Information`CVS`$DelayId = "$Id: Delay.m,v 1.19 2008/07/08 14:03:51 brenton Exp $"

(* ::Section:: *)
(*Public*)

Delay::usage =
"Delay[d] delays the robot by d seconds."

GetAutoDelay::usage =
"GetAutoDelay[] gets the auto delay of the robot."

IsAutoWaitForIdle::usage =
"IsAutoWaitForIdle[] returns whether the robot automatically invokes waitForIdle after generating an event."

SetAutoDelay::usage =
"SetAutoDelay[d] sets the auto delay of the robot to be d seconds."

SetAutoWaitForIdle::usage =
"SetAutoWaitForIdle[b] tells the robot whether to call waitForIdle after each event."

WaitForIdle::usage =
"WaitForIdle[] makes the robot wait until all events are done being processed."

$AutoDelay::usage =
"$AutoDelay is the delay between successive robot events."

$AutoWaitForIdle::usage =
"$AutoWaitForIdle determines whether the robot calls waitForIdle after each event."

(* ::Section:: *)
(*Package*)

Begin["`Package`"]

delay

getAutoDelay

isAutoWaitForIdle

setAutoDelay

setAutoWaitForIdle

waitForIdle

End[] (*`Package`*)

(* ::Section:: *)
(*Private*)

(* ::Subsection:: *)
(*Begin*)

Begin["`Delay`Private`"]

(* ::Subsection:: *)
(*Delay*)

Unprotect[Delay]

Delay[nb:focusedNotebookPat:FocusedNotebook[], d:numericPat] :=
	Module[{vDelay, buffer},
		Catch[
			(* step 1: set variables and validate user input *)
			throwIfMessage[
				vDelay = validateDelay[Delay, d]
			];
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							Sow[delay[vDelay]]
						]
				]
			];
			(* step 3: execute buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer]
			]
		]
	]

SetAttributes[Delay, ReadProtected]

Delay[arg_] :=
	(Message[Delay::intnm, HoldForm[Delay[arg]], 1]; $Failed)

Delay[args___] :=
	(ArgumentCountQ[Delay, System`FEDump`NonOptionArgCount[{args}], 1, 1]; $Failed)

SyntaxInformation[Delay] = {"ArgumentsPattern" -> {_}}

Protect[Delay]

(* ::Subsection:: *)
(*GetAutoDelay*)

Unprotect[GetAutoDelay]

GetAutoDelay[nb:focusedNotebookPat:FocusedNotebook[]] :=
	Module[{buffer},
		Catch[
			(* step 1: set variables and validate user input *)
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							Sow[getAutoDelay[]]
						]
				]
			];
			(* step 3: execute buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer]
			]
		]
	]

SetAttributes[GetAutoDelay, ReadProtected]

GetAutoDelay[args___] :=
	(ArgumentCountQ[GetAutoDelay, System`FEDump`NonOptionArgCount[{args}], 0, 0]; $Failed)

SyntaxInformation[GetAutoDelay] = {"ArgumentsPattern" -> {}}

Protect[GetAutoDelay]

(* ::Subsection:: *)
(*IsAutoWaitForIdle*)

Unprotect[IsAutoWaitForIdle]

IsAutoWaitForIdle[nb:focusedNotebookPat:FocusedNotebook[]] :=
	Module[{buffer},
		Catch[
			(* step 1: set variables and validate user input *)
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							Sow[isAutoWaitForIdle[]]
						]
				]
			];
			(* step 3: execute buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer]
			]
		]
	]

SetAttributes[IsAutoWaitForIdle, ReadProtected]

IsAutoWaitForIdle[args___] :=
	(ArgumentCountQ[IsAutoWaitForIdle, System`FEDump`NonOptionArgCount[{args}], 0, 0]; $Failed)

SyntaxInformation[IsAutoWaitForIdle] = {"ArgumentsPattern" -> {}}

Protect[IsAutoWaitForIdle]

(* ::Subsection:: *)
(*SetAutoDelay*)

Unprotect[SetAutoDelay]

SetAutoDelay[nb:focusedNotebookPat:FocusedNotebook[], delay:numericPat] :=
	Module[{vDelay, buffer},
		Catch[
			(* step 1: set variables and validate user input *)
			throwIfMessage[
				vDelay = validateDelay[SetAutoDelay, delay]
			];
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							Sow[setAutoDelay[vDelay]]
						]
				]
			];
			(* step 3: execute buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer]
			]
		]
	]

SetAttributes[SetAutoDelay, ReadProtected]

SetAutoDelay[arg_] :=
	(Message[SetAutoDelay::intnm, HoldForm[SetAutoDelay[arg]], 1]; $Failed)

SetAutoDelay[args___] :=
	(ArgumentCountQ[SetAutoDelay, System`FEDump`NonOptionArgCount[{args}], 1, 1]; $Failed)

SyntaxInformation[SetAutoDelay] = {"ArgumentsPattern" -> {_}}

Protect[SetAutoDelay]

(* ::Subsection:: *)
(*SetAutoWaitForIdle*)

Unprotect[SetAutoWaitForIdle]

SetAutoWaitForIdle[nb:focusedNotebookPat:FocusedNotebook[], wait:boolPat] :=
	Module[{buffer},
		Catch[
			(* step 1: set variables and validate user input *)
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							Sow[setAutoWaitForIdle[wait]]
						]
				]
			];
			(* step 3: execute buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer]
			]
		]
	]

SetAttributes[SetAutoWaitForIdle, ReadProtected]

SetAutoWaitForIdle[arg_] :=
	(Message[SetAutoWaitForIdle::bool, HoldForm[SetAutoWaitForIdle[arg]], 1]; $Failed)

SetAutoWaitForIdle[args___] :=
	(ArgumentCountQ[SetAutoWaitForIdle, System`FEDump`NonOptionArgCount[{args}], 1, 1]; $Failed)

SyntaxInformation[SetAutoWaitForIdle] = {"ArgumentsPattern" -> {_}}

Protect[SetAutoWaitForIdle]

(* ::Subsection:: *)
(*WaitForIdle*)

Unprotect[WaitForIdle]

WaitForIdle[nb:focusedNotebookPat:FocusedNotebook[]] :=
	Module[{buffer},
		Catch[
			(* step 1: set variables and validate user input *)
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							Sow[waitForIdle[]]
						]
				]
			];
			(* step 3: execute buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer]
			]
		]
	]

SetAttributes[WaitForIdle, ReadProtected]

WaitForIdle[args___] :=
	(ArgumentCountQ[WaitForIdle, System`FEDump`NonOptionArgCount[{args}], 0, 0]; $Failed)

SyntaxInformation[WaitForIdle] = {"ArgumentsPattern" -> {}}

Protect[WaitForIdle]

(* ::Subsection:: *)
(*$AutoDelay*)

(* TODO: find documentation on delays on various systems *)

$AutoDelay =
	0.01

(* ::Subsection:: *)
(*$AutoWaitForIdle*)

$AutoWaitForIdle =
	True

(* ::Subsection:: *)
(*Low-Level functions*)

delay[d:numericPat] :=
	If[JavaObjectQ[$Robot], $Robot@delay[Round[1000 * d]]]

getAutoDelay[] :=
	If[JavaObjectQ[$Robot], $Robot@getAutoDelay[] / 1000.0]

isAutoWaitForIdle[] :=
	If[JavaObjectQ[$Robot], $Robot@isAutoWaitForIdle[]]

setAutoDelay[d:numericPat] :=
	If[JavaObjectQ[$Robot], $Robot@setAutoDelay[Round[1000 * d]]]

setAutoWaitForIdle[b:boolPat] :=
	If[JavaObjectQ[$Robot], $Robot@setAutoWaitForIdle[b]]

waitForIdle[] :=
	If[JavaObjectQ[$Robot], $Robot@waitForIdle[]]

(* ::Subsection:: *)
(*Validation*)

validateDelay[head:symbolPat, delay:blankPat] :=
	Which[
		0 <= delay <= 60,
		delay
		,
		True,
		Message[head::d60, delay];
		$Failed
	]

(* ::Subsection:: *)
(*End*)

End[] (*`Delay`Private`*)
