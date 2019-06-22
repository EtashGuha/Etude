(* ::Package:: *)

(* ::Title:: *)
(*FrontEnd*)

(* ::Section:: *)
(*Annotations*)

(* :Title: FrontEnd.m *)

(* :Author:
        Brenton Bostick
        brenton@wolfram.com
*)

(* :Package Version: 1.2 *)

(* :Mathematica Version: 7.0 *)

(* :Copyright: RobotTools source code (c) 2005-2008 Wolfram Research, Inc. All rights reserved. *)

(* :Discussion:
	Wrapper functions for front end packet calls.
	
*)

(* ::Section:: *)
(*Information*)

`Information`CVS`$FrontEndId = "$Id: FrontEnd.m,v 1.28 2010/01/19 17:15:50 brenton Exp $"

(* ::Section:: *)
(*Public*)

AllCells::usage =
"AllCells is an option for front end box functions that determines if all cells of a notebook should be considered,\ 
or only the current selection."

BasicBox::usage =
"BasicBox is a low-level box construct."

BoxInformation::usage =
"BoxInformation[nb] returns a nested structure of information for each box in nb.
The format for each box is {box, rect, {child1, child2, ...}}.
Each child returns a similar structure.
BoxInformation[nb, allCells] returns information for all cells if allCells is True, and returns \
information for the current selection if allCells is False."

DummyPrimitiveBox::usage =
"DummyPrimitiveBox is a low-level box construct."

(*FocusedNotebook[] is used to flag whether functions were called with the optional notebook object first
argument. FocusedNotebook[] is the default value. In RobotExecute, it is checked to see whether or not
SetFocusedNotebook should be called before commands are executed. *)
FocusedNotebook::usage =
"FocusedNotebook is an internal RobotTools symbol."

GetBoxRectangles::usage =
"GetBoxRectangles[nb, box] returns a list of bounding rectangles for all instances of box in nb."

GetSelectionBoundingBoxes::usage =
"GetSelectionBoundingBoxes[nb] returns the bounding boxes of the current MathEdit selection in nb."

GetWindowRectangle::usage =
"GetWindowRectangle[nb] returns the rectangle of the client area of the notebook nb."

InlineGraphicBox::usage =
"InlineGraphicBox is a low-level box construct."

LineWrapBox::usage =
"LineWrapBox is a low-level box construct."

NotebookImage::usage =
"NotebookImage[nb] returns an image of the notebook nb."

NumberBox::usage =
"NumberBox is a low-level box construct."

PrunedBoxes::usage =
"PrunedBoxes is an option for front end box functions that allows certain boxes to be pruned."

SetFocusedNotebook::usage =
"SetFocusedNotebook[nb] gives keyboard input focus to nb."

VerticalSpanBox::usage =
"VerticalSpanBox is a low-level box construct."

(* ::Section:: *)
(*Package*)

Begin["`Package`"]

blockBoxInformation

blockInputNotebook

boxInformation

getBoxRectangles

getSelectionBoundingBoxes

getWindowRectangle

mousePosition

notebookImage

resolveFocusedNotebook

setFocusedNotebook

(*validateFocusedNotebook*)

$focusedNotebook

End[] (*`Package`*)

(* ::Section:: *)
(*Private*)

(* ::Subsection:: *)
(*Begin*)

Begin["`FrontEnd`Private`"]

(* ::Subsection:: *)
(*Messages*)

General::focused =
"SetFocusedNotebook needs to be called before using this function."

GetSelectionBoundingBoxes::noselection =
"GetSelectionBoundingBoxes was called on a notebook with no selection."

GetSelectionBoundingBoxes::morethan1cell =
"GetSelectionBoundingBoxes was called on a notebook with more than 1 cell selected."

GetSelectionBoundingBoxes::cellbrkt =
"GetSelectionBoundingBoxes was called on a notebook with a cell bracket selected."

GetSelectionBoundingBoxes::notbox =
"GetSelectionBoundingBoxes was called on a notebook with a non-box selection."

(* ::Subsection:: *)
(*blockInputNotebook*)

(*
blockInputNotebook is useful for keeping multiple evaluations of InputNotebook[] from having to call the front end.
blockInputNotebook should be used in any top-level function that calls InputNotebook more than once,
or calls other functions that call InputNotebook[] more than once.
*)

Attributes[blockInputNotebook] = {HoldFirst}

(* blockInputNotebook is a run-time front end function *)
blockInputNotebook[expr_] :=
	Module[{inputNotebook},
		inputNotebook := MathLink`CallFrontEnd[FrontEnd`InputNotebook[]];
		Block[{InputNotebook},
			InputNotebook[] :=
				InputNotebook[] =
				inputNotebook;
			expr
		]
	]

(* ::Subsection:: *)
(*blockBoxInformation*)

(*
blockBoxInformation is useful for keeping multiple evaluations of boxInformation[] from having to call the front end.
blockBoxInformation should be used in any top-level function that calls boxInformation more than once,
or calls other functions that call boxInformation more than once.
*)

Attributes[blockBoxInformation] = {HoldRest}

blockBoxInformation[{nb:nbobjPat, allCells:boolPat, prunedBoxes:stringListPat}, expr_] :=
	Module[{cachedBoxInformation = boxInformation[nb, allCells, prunedBoxes]},
		Block[{boxInformation},
			boxInformation[nb, allCells, prunedBoxes] = cachedBoxInformation;
			expr
		]
	]

(* ::Subsection:: *)
(*resolveFocusedNotebook*)

(* resolveFocusedNotebook is a run-time front end function *)
(* use InputNotebook[] so that blockInputNotebook works *)
resolveFocusedNotebook[nb:focusedNotebookPat] :=
	nb /. HoldPattern[FocusedNotebook[]] :> InputNotebook[]

(* ::Subsection:: *)
(*BoxInformation*)

Unprotect[BoxInformation]

Options[BoxInformation] = {AllCells -> True, PrunedBoxes -> Automatic}

BoxInformation[nb:focusedNotebookPat:FocusedNotebook[], OptionsPattern[]] :=
	Module[{allCells, prunedBoxes, resolvedNB, vAllCells, vPrunedBoxes, buffer},
		Catch[
			(* step 1: set variables and validate user input *)
			throwIfMessage[
				{allCells, prunedBoxes} = OptionValue[{AllCells, PrunedBoxes}];
				resolvedNB = resolveFocusedNotebook[nb];
				vAllCells = validateAllCells[BoxInformation, allCells];
				vPrunedBoxes = validatePrunedBoxes[BoxInformation, prunedBoxes]
			];
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							Sow[boxInformation[resolvedNB, vAllCells, vPrunedBoxes]]
						]
				]
			];
			(* step 3: execute buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer, CallInstallRobotTools -> False, CallSetFocusedNotebook -> False]
			]
		]
	]

SetAttributes[BoxInformation, ReadProtected]

BoxInformation[arg_] :=
	(Message[BoxInformation::nbobj, 1, HoldForm[BoxInformation[arg]]]; $Failed)

BoxInformation[args___] :=
	(ArgumentCountQ[BoxInformation, System`FEDump`NonOptionArgCount[{args}], 0, 2]; $Failed)

SyntaxInformation[BoxInformation] = {"ArgumentsPattern" -> {_., OptionsPattern[]}}

Protect[BoxInformation]

(* ::Subsection:: *)
(*GetBoxRectangles*)

Unprotect[GetBoxRectangles]

Options[GetBoxRectangles] = {AllCells -> True, PrunedBoxes -> Automatic}

GetBoxRectangles[nb:focusedNotebookPat:FocusedNotebook[], box:blankPat(*boxPat*), OptionsPattern[]] :=
	Module[{allCells, prunedBoxes, vAllCells, vPrunedBoxes, resolvedNB, buffer},
		Catch[
			(* step 1: set variables and validate user input *)
			throwIfMessage[
				{allCells, prunedBoxes} = OptionValue[{AllCells, PrunedBoxes}];
				resolvedNB = resolveFocusedNotebook[nb];
				vAllCells = validateAllCells[GetBoxRectangles, allCells];
				vPrunedBoxes = validatePrunedBoxes[GetBoxRectangles, prunedBoxes]
			];
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							Sow[getBoxRectangles[resolvedNB, box, vAllCells, vPrunedBoxes]]
						]
				]
			];
			(* step 3: execute buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer, CallInstallRobotTools -> False, CallSetFocusedNotebook -> False]
			]
		]
	]

SetAttributes[GetBoxRectangles, ReadProtected]

GetBoxRectangles[args___] :=
	(ArgumentCountQ[GetBoxRectangles, System`FEDump`NonOptionArgCount[{args}], 1, 2]; $Failed)

SyntaxInformation[GetBoxRectangles] = {"ArgumentsPattern" -> {_., _, OptionsPattern[]}}

Protect[GetBoxRectangles]

getBoxRectangles[nb:nbobjPat, box:boxPat, allCells:boolPat, prunedBoxes:stringListPat] :=
	Cases[boxInformation[nb, allCells, prunedBoxes], {box, rect_, _} :> rect, Infinity]

(* ::Subsection:: *)
(*GetSelectionBoundingBoxes*)

Unprotect[GetSelectionBoundingBoxes]

GetSelectionBoundingBoxes[nb:focusedNotebookPat:FocusedNotebook[]] :=
	Module[{resolvedNB, buffer},
		Catch[
			(* step 1: set variables and validate user input *)
			throwIfMessage[
				resolvedNB = resolveFocusedNotebook[nb]
			];
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							Sow[getSelectionBoundingBoxes[resolvedNB]]
						]
				]
			];
			(* step 3: execute buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer, CallInstallRobotTools -> False, CallSetFocusedNotebook -> False]
			]
		]
	]

SetAttributes[GetSelectionBoundingBoxes, ReadProtected]

GetSelectionBoundingBoxes[arg_] :=
	(Message[GetSelectionBoundingBoxes::nbobj, 1, HoldForm[GetSelectionBoundingBoxes[arg]]]; $Failed)

GetSelectionBoundingBoxes[args___] :=
	(ArgumentCountQ[GetSelectionBoundingBoxes, System`FEDump`NonOptionArgCount[{args}], 0, 1]; $Failed)

SyntaxInformation[GetSelectionBoundingBoxes] = {"ArgumentsPattern" -> {_.}}

Protect[GetSelectionBoundingBoxes]

(* ::Subsection:: *)
(*GetWindowRectangle*)

Unprotect[GetWindowRectangle]

GetWindowRectangle[nb:focusedNotebookPat:FocusedNotebook[]] :=
	Module[{resolvedNB, buffer},
		Catch[
			(* step 1: set variables and validate user input *)
			throwIfMessage[
				resolvedNB = resolveFocusedNotebook[nb]
			];
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							Sow[getWindowRectangle[resolvedNB]]
						]
				]
			];
			(* step 3: execute buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer, CallInstallRobotTools -> False, CallSetFocusedNotebook -> False]
			]
		]
	]

SetAttributes[GetWindowRectangle, ReadProtected]

GetWindowRectangle[arg_] :=
	(Message[GetWindowRectangle::nbobj, 1, HoldForm[GetWindowRectangle[arg]]]; $Failed)

GetWindowRectangle[args___] :=
	(ArgumentCountQ[GetWindowRectangle, System`FEDump`NonOptionArgCount[{args}], 0, 1]; $Failed)

SyntaxInformation[GetWindowRectangle] = {"ArgumentsPattern" -> {_.}}

Protect[GetWindowRectangle]

(* ::Subsection:: *)
(*NotebookImage*)

Unprotect[NotebookImage]

NotebookImage[nb:focusedNotebookPat:FocusedNotebook[]] :=
	Module[{resolvedNB, buffer},
		Catch[
			(* step 1: set variables and validate user input *)
			throwIfMessage[
				resolvedNB = resolveFocusedNotebook[nb]
			];
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							Sow[notebookImage[resolvedNB]]
						]
				]
			];
			(* step 3: execute buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer, CallInstallRobotTools -> False, CallSetFocusedNotebook -> False]
			]
		]
	]

SetAttributes[NotebookImage, ReadProtected]

NotebookImage[arg_] :=
	(Message[NotebookImage::nbobj, 1, HoldForm[NotebookImage[arg]]]; $Failed)

NotebookImage[args___] :=
	(ArgumentCountQ[NotebookImage, System`FEDump`NonOptionArgCount[{args}], 0, 1]; $Failed)

SyntaxInformation[NotebookImage] = {"ArgumentsPattern" -> {_.}}

Protect[NotebookImage]

(* ::Subsection:: *)
(*SetFocusedNotebook[]*)

Unprotect[SetFocusedNotebook]

(* SetFocusedNotebook is a run-time front end function *)
(* SetFocusedNotebook[] has to exist because SetSelectedNotebook[] is broken on X *)
(* calling SetFocusedNotebook[] consecutively on the same notebook will click the mouse again. *)
SetFocusedNotebook[nb:focusedNotebookPat:FocusedNotebook[]] :=
	Module[{buffer},
		Catch[
			(* step 1: set variables and validate user input *)
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							Sow[setFocusedNotebook[nb]]
						]
				]
			];
			(* step 3: execute buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer, CallSetFocusedNotebook -> False]
			]
		]
	]

SetAttributes[SetFocusedNotebook, ReadProtected]

Protect[SetFocusedNotebook]

(* ::Subsection:: *)
(*Validation*)

validateAllCells[head:symbolPat, allCells_] :=
	Which[
		MatchQ[allCells, boolPat],
		allCells
		,
		True,
		Message[head::optvg, AllCells, allCells, "True or False"];
		$Failed
	]

validatePrunedBoxes[head:symbolPat, prunedBoxes_] :=
	Which[
		MatchQ[prunedBoxes, boxListPat],
		ToString /@ prunedBoxes
		,
		MatchQ[prunedBoxes, Automatic],
		{"DummyPrimitiveBox"}
		,
		True,
		Message[head::optvg, PrunedBoxes, prunedBoxes, "a list of boxes or Automatic"];
		$Failed
	]

(*validateFocusedNotebook[head:symbolPat, nb:focusedNotebookPat] :=
	Which[
		(* Windows and Macintosh are always OK *)
		(*$InterfaceEnvironment =!= "X",
		Null
		,*)
		(* if SetFocusedNotebook[] was never called and if a function like ItemActivate was called without the nb argument,
		then don't give a message *)
		!ValueQ[$focusedNotebook] && MatchQ[nb, HoldPattern[FocusedNotebook[]]],
		Null
		,
		(* if nb matches the currently focused notebook or the literal FocusedNotebook[], then don't give a message *)
		MatchQ[nb, $focusedNotebook | HoldPattern[FocusedNotebook[]]],
		Null
		,
		(* everything else gives a message *)
		True,
		Message[head::focused];
		$Failed
	]*)

(* ::Subsection:: *)
(*Low-Level Functions*)

(* blockInformation is a run-time front end function *)
boxInformation[nb:nbobjPat, allCells:boolPat, prunedBoxes:stringListPat] :=
	Module[{info, winPos, cells},
		info = MathLink`CallFrontEnd[FrontEnd`UndocumentedBoxInformationPacket[nb, allCells]];
		If[info === $Failed,
			{}
			,
			Internal`WithLocalSettings[
				(*
				TODO: Remove Off[]/On[] when UndocumentedBoxInformationPacket is really really really fixed
				The Begin[]/End[] is for the ToExpression[] below that introduces the weird box symbols
				Maybe technically should be "FrontEnd`", but "RobotTools`" also makes sense and
				allows old code to still work.
				*)
				Off[Infinity::indet];
				Begin["RobotTools`"]
				,
				winPos = (FE`WindowRectangle /. First[info])[[1]];
				cells = ReplaceAll[Rest[info], FE`CellWrapper -> Identity];
				(* remove the goofy children of LocatorBox *)
				cells =
					Replace[cells,
						{FE`BoxType -> "LocatorBox", args__, FE`Children -> {___}} :>
							{FE`BoxType -> "LocatorBox", args, FE`Children -> {}}
					];
				cells = Replace[cells, {FE`BoxType -> Alternatives @@ prunedBoxes, (*args*)__} :> Sequence[], Infinity];
				cells =
					ReplaceRepeated[cells,
						{FE`BoxType -> type_, FE`Position -> pos:{_, _}, FE`BoundingRectangle -> rect:{_, _}, __, FE`Children -> children:{___}} :>
							{ToExpression[type], Transpose[winPos + pos + Transpose[rect]], children}
					];
				cells
				,
				End[];
				On[Infinity::indet]
			]
		]
	]

(* getSelectionBoundingBoxes is a run-time front end function *)
getSelectionBoundingBoxes[nb:nbobjPat] :=
	Module[{info},
		info = MathLink`CallFrontEnd[FrontEnd`CellInformation[nb]];
		Which[
			info === $Failed,
			Message[GetSelectionBoundingBoxes::noselection];
			$Failed
			,
			Length[info] > 1,
			Message[GetSelectionBoundingBoxes::morethan1cell];
			$Failed
			,
			("CursorPosition" /. info[[1]]) === "CellBracket",
			Message[GetSelectionBoundingBoxes::cellbrkt];
			$Failed
			,
			("ContentData" /. info[[1]]) =!= BoxData,
			Message[GetSelectionBoundingBoxes::notbox];
			$Failed
			,
			True,
			MathLink`CallFrontEnd[FrontEnd`GetSelectionBoundingBoxes[nb]]
		]
	]

(*
Ideally, GetWindowRectangle would have it's own packet.
Currently, it uses UndocumentedBoxInformationPacket and gets a lot of info back that it never uses.
Send in False for the 2nd arg in a futile attempt to make everything a little quicker, if there happens to be a selection.

Because getWindowRectangle does not call boxInformation, then blockBoxInformation will not work on it.
Whenever GetWindowRectangle gets its own packet, then all of this will be solved.
*)

(* getWindowRectangle is a run-time front end function *)
getWindowRectangle[nb:nbobjPat] :=
	Module[{info},
		info = MathLink`CallFrontEnd[FrontEnd`UndocumentedBoxInformationPacket[nb, False]];
		FE`WindowRectangle /. First[info]
	]

(* notebookImage is a run-time front end function *)
notebookImage[nb:nbobjPat] :=
	MathLink`CallFrontEnd[FrontEnd`NotebookImage[nb]]

(* mousePosition is a run-time front end function *)
mousePosition[] :=
	MathLink`CallFrontEnd[FrontEnd`Value[FrontEnd`MousePosition["ScreenAbsolute"], False]]

(* setFocusedNotebook is a run-time front end function *)
setFocusedNotebook[nb:focusedNotebookPat] :=
	Module[{pos, rect},
		If[!MatchQ[nb, HoldPattern[FocusedNotebook[]]],
			MathLink`CallFrontEnd[FrontEnd`SetSelectedNotebook[nb]];
			(* get original mouse coordinates *)
			pos = mousePosition[];
			rect = getWindowRectangle[nb];
			mouseMove[rect[[1]]];
			mouseClick["Button1"];
			(* move back to original mouse coordinates *)
			mouseMove[pos];
			(* $focusedNotebook is a global variable that keeps track of the last time that
			SetFocusedNotebook[] was called.
			This is then checked in ItemActivate on X to issue a message if SetFocusedNotebook[] was not called before
			ItemActivate.
			*)
			$focusedNotebook = nb
		]
	]

(* ::Subsection:: *)
(*End*)

End[] (*`FrontEnd`Private`*)
