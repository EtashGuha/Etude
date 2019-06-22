(* ::Package:: *)

(* ::Title:: *)
(*Mouse*)

(* ::Section:: *)
(*Annotations*)

(* :Title: Mouse.m *)

(* :Author:
        Brenton Bostick
        brenton@wolfram.com
*)

(* :Package Version: 1.2 *)

(* :Mathematica Version: 7.0 *)

(* :Copyright: RobotTools source code (c) 2005-2008 Wolfram Research, Inc. All rights reserved. *)

(* :Discussion:
	Implementation of mouse-related functionality.
	MousePress, MouseRelease, and MouseWheel are not like other mouse functions.
	They take a button or wheel as an argument and they do not take coordinates.
	They are lower-level functions and should be used sparingly.
	
	The
	Catch[
		throwIfMessage[
		
		];
		throwIfMessage[
		
		];
	]
	idiom is an approximation to a proper kernel message handling mechanism.
	This allows the function to return $Failed if a message is generated at any point before actual execution of the robot commands.
	Since errant behavior in mouse moving and clicking could be disastrous to user's machine, the slightest indication that anything
	is wrong (i.e., a message) tells the function to return before executing the commands.
	
	On Windows, Using CurrentValue["MouseButtons"]:
	Java button 1 - Mathematica button 1 - left mouse button
	Java button 2 - Mathematica button 3 - mouse wheel button
	Java button 3 - Mathematica button 2 - right mouse button
*)

(* ::Section:: *)
(*TODO*)

(*
TODO: SpeedFunction option for each point? MapIndexed 

TODO: syntax for MouseClicks that specifies how to click 

TODO: MouseSteps-> something Equidistant? Any similar options in plotting options, contour, desnity, etc.? 

TODO: mention somewhere that mouse buttons can be bitwise-ored together

TODO: update MousePress and MouseRelease to take a single point and the Button option, with the button arg trumping the option

TODO: handle Mouse*Click options better
*)

(* ::Section:: *)
(*Information*)

`Information`CVS`$MouseId = "$Id: Mouse.m,v 1.45 2012/01/11 18:00:42 brenton Exp $"

(* ::Section:: *)
(*Public*)

Clicks::usage =
"Clicks is an option to mouse functions.
Clicks specifies the number of clicks."

MouseButton::usage =
"MouseButton is deprecated. Use Button."

MouseClick::usage =
"MouseClick[] clicks the first mouse button.
MouseClick[{x1, y1}] moves the mouse pointer to {x1, y1}, then clicks the first mouse button.
MouseClick[{pt1, pt2, ...}] moves the mouse pointer to each point, and clicks the first mouse button after each movement.
MouseClick[nb, pts] selects nb first."

MouseClicks::usage =
"MouseClicks is deprecated. Use Clicks."

MouseDoubleClick::usage =
"MouseDoubleClick[] double clicks the first mouse button."

MouseDrag::usage =
"MouseDrag[{x1, y1}, {x2, y2}] moves the mouse pointer to {x1, y1}, presses the first mouse button, moves the mouse pointer to {x2, y2}, then releases the first mouse button.
MouseDrag[{pt1, pt2, ...}] moves the mouse pointer to pt1, presses the first mouse button, moves the mouse pointer to the rest of the points, then releases the first mouse button.
MouseDrag[nb, pts] selects nb first."

MouseMove::usage =
"MouseMove[{x1, y1}] moves the mouse pointer to the screen coordinates {x1, y1}.
MouseMove[{pt1, pt2, ...}] moves the mouse pointer to each point in succession.
MouseMove[nb, pts] selects nb first."

(*MousePoints::usage =
"MousePoints[nb, pts] returns the screen coordinates of pts after interpolation and scaling."*)

MousePress::usage =
"MousePress[btn] presses mouse button btn.
MousePress[nb, btn] selects nb first."

MouseRelease::usage =
"MouseRelease[btn] releases mouse button btn.
MouseRelease[nb, btn] selects nb first."

MouseSingleClick::usage =
"MouseSingleClick is equivalent to MouseClick."

MouseSteps::usage =
"MouseSteps is deprecated. Use Steps."

MouseTripleClick::usage =
"MouseTripleClick[] triple clicks the first mouse button."

MouseWheel::usage =
"MouseWheel[amount] scrolls the mouse scroll wheel a given amount."

Steps::usage =
"Steps is an option for mouse functions.
Steps specifies the number of steps to take between points."

$MouseClickDelay::usage =
"$MouseClickDelay is the delay before the mouse is clicked."

$MouseMoveDelay::usage =
"$MouseMoveDelay is the delay before the mouse is moved."

$MousePressDelay::usage =
"$MousePressDelay is the delay before a mouse button is pressed."

$MouseReleaseDelay::usage =
"$MouseReleaseDelay is the delay before a mouse button is released."

$MouseWheelDelay::usage =
"$MouseWheelDelay is the delay before the mouse wheel is turned."

CallSetFocusedNotebook

(* ::Section:: *)
(*Package*)

Begin["`Package`"]

mouseClick

mouseMove

mousePress

mouseRelease

mouseWheel

sowButton

End[] (*`Package`*)

(* ::Section:: *)
(*Private*)

(* ::Subsection:: *)
(*Begin*)

Begin["`Mouse`Private`"]

(* ::Subsection:: *)
(*interpolationPointsFunction*)

(*
If order is Automatic, then resolveInterpolationOrder returns 0 if there is only 1 point or 1 if there are more than 1 point.
This matches what a user would typically want, linear interpolation.
If order is an integer, then it is just returned untouched.
*)
resolveInterpolationOrder[points:numericPointListPat, order:orderPat] :=
	If[order === Automatic, Min[Length[points] - 1, 1], order]

(*
interpolationPointsFunction returns a function that interpolates through points with the specified order
fun = interpolationPointsFunction[points, order];
fun[1] is the first point
fun[1.5] is between the first and second points, etc.
*)
interpolationPointsFunction[points:numericPointListPat, order:orderPat] :=
	Interpolation[MapIndexed[{#2, #1}&, points], InterpolationOrder -> resolveInterpolationOrder[points, order]]

(* ::Subsection:: *)
(*sowButton*)

Attributes[sowButton] = {HoldRest}

sowButton[button:buttonPat, expr_] :=
	(
		Sow[mousePress[button]];
		expr;
		Sow[mouseRelease[button]]
	)

(* ::Subsection:: *)
(*Mouse*)

(*
for backward-compatibility, in retrospect, the Mouse* prefixes aren't needed
*)
MouseButton =
	Button

MouseClicks =
	Clicks

MouseSteps =
	Steps

(*
Mouse exists as a symbol only to carry the options for all of the mouse functions, as a convenience
*)
Options[Mouse] =
	{
		Modifiers -> {},
		Button -> "Button1",
		Clicks -> 1,
		Steps -> 10,
		InterpolationOrder -> Automatic,
		Scaling -> None,
		CallSetFocusedNotebook -> True
	}

(* ::Subsection:: *)
(*MousePress*)

Unprotect[MousePress]

(* filter the options from Mouse *)
Options[MousePress] = FilterRules[Options[Mouse], {Button, Scaling, CallSetFocusedNotebook}]

(*
This is kept for backward-compatibility
if the button arg is specified, then just ignore all options
*)
MousePress[nb:focusedNotebookPat:FocusedNotebook[], button:buttonPat:"Button1", OptionsPattern[]] :=
	Module[{buffer, callSetFocusedNotebook},
		Catch[
			(* step 1: set variables and validate user input *)
			throwIfMessage[
				{callSetFocusedNotebook} = OptionValue[{CallSetFocusedNotebook}];
			];
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							Sow[mousePress[button]]
						]
				]
			];
			(* step 3: execute buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer, CallSetFocusedNotebook -> callSetFocusedNotebook]
			]
		]
	]

(*
TODO: validatePoints will handle resolving the menu notebook, don't do it before if not necessary
*)

MousePress[nb:focusedNotebookPat:FocusedNotebook[], point:pointPat, OptionsPattern[]] :=
	Module[{button, scaling, (*resolvedNB,*) vScaling, vPoint, buffer, callSetFocusedNotebook},
		Catch[
			(* step 1: set variables and validate user input *)
			throwIfMessage[
				{button, scaling, callSetFocusedNotebook} = OptionValue[{Button, Scaling, CallSetFocusedNotebook}];
				vScaling = validateScaling[MousePress, scaling];
				{vPoint} = validatePoints[MousePress, nb, {point}, vScaling]
			];
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							Sow[mouseMove[vPoint]];
							Sow[mousePress[button]]
						]
				]
			];
			(* step 3: execute buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer, CallSetFocusedNotebook -> callSetFocusedNotebook]
			]
		]
	]

SetAttributes[MousePress, ReadProtected]

MousePress[arg_] :=
	(Message[MousePress::pnt, 1, HoldForm[MousePress[arg]]]; $Failed)

MousePress[arg1:nbobjPat, args__] :=
	(Message[MousePress::pnt, 2, HoldForm[MousePress[arg1, args]]]; $Failed)

MousePress[arg1_, arg2:pointPat] :=
	(Message[MousePress::nbobj, 1, HoldForm[MousePress[arg1, arg2]]]; $Failed)

MousePress[args___] :=
	(ArgumentCountQ[MousePress, System`FEDump`NonOptionArgCount[{args}], 0, 2]; $Failed)

SyntaxInformation[MousePress] = {"ArgumentsPattern" -> {_., _., OptionsPattern[]}}

Protect[MousePress]

(* ::Subsection:: *)
(*MouseRelease*)

Unprotect[MouseRelease]

(* filter the options from Mouse *)
Options[MouseRelease] = FilterRules[Options[Mouse], {Button, CallSetFocusedNotebook}]

(*
This is kept for backward-compatibility
if the button arg is specified, then just ignore all options
*)
MouseRelease[nb:focusedNotebookPat:FocusedNotebook[], button:buttonPat:"Button1", OptionsPattern[]] :=
	Module[{buffer, callSetFocusedNotebook},
		Catch[
			(* step 1: set variables and validate user input *)
			throwIfMessage[
				{callSetFocusedNotebook} = OptionValue[{CallSetFocusedNotebook}];
			];
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							Sow[mouseRelease[button]]
						]
				]
			];
			(* step 3: execute buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer, CallSetFocusedNotebook -> callSetFocusedNotebook]
			]
		]
	]

MouseRelease[nb:focusedNotebookPat:FocusedNotebook[], point:pointPat, OptionsPattern[]] :=
	Module[{button, scaling, (*resolvedNB,*) vScaling, vPoint, buffer, callSetFocusedNotebook},
		Catch[
			(* step 1: set variables and validate user input *)
			throwIfMessage[
				{button, scaling, callSetFocusedNotebook} = OptionValue[{Button, Scaling, CallSetFocusedNotebook}];
				vScaling = validateScaling[MouseRelease, scaling];
				{vPoint} = validatePoints[MouseRelease, nb, {point}, vScaling]
			];
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							Sow[mouseMove[vPoint]];
							Sow[mouseRelease[button]]
						]
				]
			];
			(* step 3: execute buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer, CallSetFocusedNotebook -> callSetFocusedNotebook]
			]
		]
	]

SetAttributes[MouseRelease, ReadProtected]

MouseRelease[arg_] :=
	(Message[MouseRelease::pnt, 1, HoldForm[MouseRelease[arg]]]; $Failed)

MouseRelease[arg1:nbobjPat, args__] :=
	(Message[MouseRelease::pnt, 2, HoldForm[MouseRelease[arg1, args]]]; $Failed)

MouseRelease[arg1_, arg2:pointPat] :=
	(Message[MouseRelease::nbobj, 1, HoldForm[MouseRelease[arg1, arg2]]]; $Failed)

MouseRelease[args___] :=
	(ArgumentCountQ[MouseRelease, System`FEDump`NonOptionArgCount[{args}], 0, 2]; $Failed)

SyntaxInformation[MouseRelease] = {"ArgumentsPattern" -> {_., _., OptionsPattern[]}}

Protect[MouseRelease]

(* ::Subsection:: *)
(*MouseWheel*)

Unprotect[MouseWheel]

Options[MouseWheel] = FilterRules[Options[Mouse], {CallSetFocusedNotebook}]

MouseWheel[nb:focusedNotebookPat:FocusedNotebook[], amount:intPat:0, OptionsPattern[]] :=
	Module[{buffer, callSetFocusedNotebook},
		Catch[
			(* step 1: set variables and validate user input *)
			throwIfMessage[
				{callSetFocusedNotebook} = OptionValue[{CallSetFocusedNotebook}];
			];
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							Sow[mouseWheel[amount]]
						]
				]
			];
			(* step 3: execute buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer, CallSetFocusedNotebook -> callSetFocusedNotebook]
			]
		]
	]

SetAttributes[MouseWheel, ReadProtected]

MouseWheel[arg_] :=
	(Message[MouseWheel::int, HoldForm[MouseWheel[arg]], 1]; $Failed)

MouseWheel[arg1:nbobjPat, args__] :=
	(Message[MouseWheel::int, HoldForm[MouseWheel[arg1, args]], 2]; $Failed)

MouseWheel[arg1_, arg2:wholePat] :=
	(Message[MouseWheel::nbobj, 1, HoldForm[MouseWheel[arg1, arg2]]]; $Failed)

MouseWheel[args___] :=
	(ArgumentCountQ[MouseWheel, System`FEDump`NonOptionArgCount[{args}], 0, 2]; $Failed)

SyntaxInformation[MouseWheel] = {"ArgumentsPattern" -> {_., _., OptionsPattern[]}}

Protect[MouseWheel]

(* ::Subsection:: *)
(*MouseMove*)

Unprotect[MouseMove]

(* filter the options from Mouse *)
Options[MouseMove] = FilterRules[Options[Mouse], {Modifiers, Steps, InterpolationOrder, Scaling, CallSetFocusedNotebook}]

(*
if a point list like {Center, Top} is given, this is ambiguous, and treat them like two points, since it is easy to get the one point
behavior with {{Center, Top}}
*)
MouseMove[nb:focusedNotebookPat:FocusedNotebook[], pts:{posPat, posPat}, opts:OptionsPattern[]] :=
	MouseMove[nb, resolvePositions[pts], opts]

(* if a single point is given, then wrap it in a list *)
MouseMove[nb:focusedNotebookPat:FocusedNotebook[], pt:pointPat, opts:OptionsPattern[]] :=
	MouseMove[nb, {pt}, opts]

MouseMove[nb:focusedNotebookPat:FocusedNotebook[], pts:pointListPat, OptionsPattern[]] :=
	Module[{scaling, modifiers, steps, order, (*resolvedNB,*) vPoints, vScaling, vModifiers, vSteps, vOrder, buffer, fun, callSetFocusedNotebook},
		Catch[
			(*
			step 1: set variables and validate user input
			All option values are assigned to temporary variables
			
			validate* functions are for options. If a validate* function doesn't like a value, it gives a message, with the message
			name beginning with whatever top-level symbol was given to the validate* function.
			
			vMenuNotebook is the user-input notebook, which is possibly the default MenuNotebook[], useful for giving to RobotExecute
			(so that RobotExecute knows whether to select a notebook, based on MenuNotebook[] vs. real notebook)
			vNotebook is the resolved notebook, useful for giving to boxInformation
			(since a real notebook is required)			
			*)
			throwIfMessage[
				{scaling, modifiers, steps, order, callSetFocusedNotebook} = OptionValue[{Scaling, Modifiers, Steps, InterpolationOrder, CallSetFocusedNotebook}];
				vScaling = validateScaling[MouseMove, scaling];
				vModifiers = validateModifiers[MouseMove, modifiers];
				vSteps = validateSteps[MouseMove, steps];
				vOrder = validateInterpolationOrder[MouseMove, order];
				vPoints = validatePoints[MouseMove, nb, pts, vScaling]
			];
			(*
			step 2: construct buffer
			a buffer needs to be constructed inside a RobotBlock to allow all of the medium-level functions to evaluate.
			(e.g., keyType and mouseClick)
			*)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							sowKeyModifiers[vModifiers,
								If[vPoints != {},
									fun = interpolationPointsFunction[vPoints, vOrder];
									Do[
										Sow[mouseMove[fun[i]]]
										,
										{i, 1, Length[vPoints], 1/vSteps}
									]
								]
							]
						]
				]
			];
			(* step 3: execute the buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer, CallSetFocusedNotebook -> callSetFocusedNotebook]
			]
		]
	]

SetAttributes[MouseMove, ReadProtected]

MouseMove[arg_] :=
	(Message[MouseMove::pntl, 1, HoldForm[MouseMove[arg]]]; $Failed)

MouseMove[arg1:nbobjPat, args__] :=
	(Message[MouseMove::pntl, 2, HoldForm[MouseMove[arg1, args]]]; $Failed)

MouseMove[arg1_, arg2:pointPat|pointListPat] :=
	(Message[MouseMove::nbobj, 1, HoldForm[MouseMove[arg1, arg2]]]; $Failed)

MouseMove[args___] :=
	(ArgumentCountQ[MouseMove, System`FEDump`NonOptionArgCount[{args}], 1, 2]; $Failed)

SyntaxInformation[MouseMove] = {"ArgumentsPattern" -> {_., _, OptionsPattern[]}}

Protect[MouseMove]

(* ::Subsection:: *)
(*MouseClick*)

Unprotect[MouseClick]

(* filter the options from Mouse *)
Options[MouseClick] = FilterRules[Options[Mouse], {Modifiers, Button, Clicks, Steps, InterpolationOrder, Scaling, CallSetFocusedNotebook}]

(* override the default, since the user probably only wants to click on the points specified *)
SetOptions[MouseClick, Steps -> 1]

(*
if a point list like {Center, Top} is given, this is ambiguous, and treat them like two points, since it is easy to get the one point
behavior with {{Center, Top}}
*)
MouseClick[nb:focusedNotebookPat:FocusedNotebook[], pts:{posPat, posPat}, opts:OptionsPattern[]] :=
	MouseClick[nb, resolvePositions[pts], opts]

(* if a single point is given, then wrap it in a list *)
MouseClick[nb:focusedNotebookPat:FocusedNotebook[], pt:pointPat, opts:OptionsPattern[]] :=
	MouseClick[nb, {pt}, opts]

MouseClick[nb:focusedNotebookPat:FocusedNotebook[], pts:pointListPat:{}, OptionsPattern[]] :=
	Module[{scaling, modifiers, steps, order, button, clicks, (*resolvedNB,*) vPoints, vScaling, vModifiers, vSteps, vOrder, vButton, vClicks, buffer, clickTable, fun, iPts, callSetFocusedNotebook},
		Catch[
			(* step 1: set variables and validate user input *)
			throwIfMessage[
				{scaling, modifiers, steps, order, button, clicks, callSetFocusedNotebook} = OptionValue[{Scaling, Modifiers, Steps, InterpolationOrder, Button, Clicks, CallSetFocusedNotebook}];
				vScaling = validateScaling[MouseClick, scaling];
				vModifiers = validateModifiers[MouseClick, modifiers];
				vSteps = validateSteps[MouseClick, steps];
				vOrder = validateInterpolationOrder[MouseClick, order];
				vButton = validateButton[MouseClick, button];
				vClicks = validateClicks[MouseClick, clicks];
				clickTable = Table[vButton, {vClicks}];
				vPoints = validatePoints[MouseClick, nb, pts, vScaling]
			];
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer = 
						reapHeldList[
							sowKeyModifiers[vModifiers,
								Switch[Length[vPoints],
									0,
									Map[Sow, mouseClick /@ clickTable, {2}]
									,
									_,
									fun = interpolationPointsFunction[vPoints, vOrder];
									iPts = Table[fun[i], {i, 1, Length[vPoints], 1/vSteps}];
									Do[
										Sow[mouseMove[iPts[[i]]]];
										Map[Sow, mouseClick /@ clickTable, {2}]
										,
										{i, Length[iPts]}				
									]
								]
							]
						]
				]
			];
			(* step 3: execute the buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer, CallSetFocusedNotebook -> callSetFocusedNotebook]
			]
		]
	]

SetAttributes[MouseClick, ReadProtected]

MouseClick[arg_] :=
	(Message[MouseClick::pntl, 1, HoldForm[MouseClick[arg]]]; $Failed)

MouseClick[arg1:nbobjPat, args__] :=
	(Message[MouseClick::pntl, 2, HoldForm[MouseClick[arg1, args]]]; $Failed)

MouseClick[arg1_, arg2:pointPat|pointListPat] :=
	(Message[MouseClick::nbobj, 1, HoldForm[MouseClick[arg1, arg2]]]; $Failed)

MouseClick[args___] :=
	(ArgumentCountQ[MouseClick, System`FEDump`NonOptionArgCount[{args}], 0, 2]; $Failed)

SyntaxInformation[MouseClick] = {"ArgumentsPattern" -> {_., _., OptionsPattern[]}}

Protect[MouseClick]

(* ::Subsection:: *)
(*MouseSingleClick*)

Unprotect[MouseSingleClick]

Options[MouseSingleClick] = FilterRules[Options[Mouse], {Modifiers, Button, Steps, InterpolationOrder, Scaling, CallSetFocusedNotebook}]

MouseSingleClick[nb:focusedNotebookPat:FocusedNotebook[], pts:{posPat, posPat}, opts:OptionsPattern[]] :=
	MouseSingleClick[nb, resolvePositions[pts], opts]

MouseSingleClick[nb:focusedNotebookPat:FocusedNotebook[], pt:pointPat, opts:OptionsPattern[]] :=
	MouseSingleClick[nb, {pt}, opts]

MouseSingleClick[nb:focusedNotebookPat:FocusedNotebook[], pts:pointListPat:{}, opts:OptionsPattern[]] :=
	MouseClick[nb, pts, Clicks -> 1, opts]

SetAttributes[MouseSingleClick, ReadProtected]

MouseSingleClick[arg_] :=
	(Message[MouseSingleClick::pntl, 1, HoldForm[MouseSingleClick[arg]]]; $Failed)

MouseSingleClick[arg1:nbobjPat, args__] :=
	(Message[MouseSingleClick::pntl, 2, HoldForm[MouseSingleClick[arg1, args]]]; $Failed)

MouseSingleClick[arg1_, arg2:pointPat|pointListPat] :=
	(Message[MouseSingleClick::nbobj, 1, HoldForm[MouseSingleClick[arg1, arg2]]]; $Failed)

MouseSingleClick[args___] :=
	(ArgumentCountQ[MouseSingleClick, System`FEDump`NonOptionArgCount[{args}], 0, 2]; $Failed)

SyntaxInformation[MouseSingleClick] = {"ArgumentsPattern" -> {_., _., OptionsPattern[]}}

Protect[MouseSingleClick]

(* ::Subsection:: *)
(*MouseDoubleClick*)

Unprotect[MouseDoubleClick]

Options[MouseDoubleClick] = FilterRules[Options[Mouse], {Modifiers, Button, Steps, InterpolationOrder, Scaling, CallSetFocusedNotebook}]

MouseDoubleClick[nb:focusedNotebookPat:FocusedNotebook[], pts:{posPat, posPat}, opts:OptionsPattern[]] :=
	MouseDoubleClick[nb, resolvePositions[pts], opts]

MouseDoubleClick[nb:focusedNotebookPat:FocusedNotebook[], pt:pointPat, opts:OptionsPattern[]] :=
	MouseDoubleClick[nb, {pt}, opts]

MouseDoubleClick[nb:focusedNotebookPat:FocusedNotebook[], pts:pointListPat:{}, opts:OptionsPattern[]] :=
	MouseClick[nb, pts, Clicks -> 2, opts]

SetAttributes[MouseDoubleClick, ReadProtected]

MouseDoubleClick[arg_] :=
	(Message[MouseDoubleClick::pntl, 1, HoldForm[MouseDoubleClick[arg]]]; $Failed)

MouseDoubleClick[arg1:nbobjPat, args__] :=
	(Message[MouseDoubleClick::pntl, 2, HoldForm[MouseDoubleClick[arg1, args]]]; $Failed)

MouseDoubleClick[arg1_, arg2:pointPat|pointListPat] :=
	(Message[MouseDoubleClick::nbobj, 1, HoldForm[MouseDoubleClick[arg1, arg2]]]; $Failed)

MouseDoubleClick[args___] :=
	(ArgumentCountQ[MouseDoubleClick, System`FEDump`NonOptionArgCount[{args}], 0, 2]; $Failed)

SyntaxInformation[MouseDoubleClick] = {"ArgumentsPattern" -> {_., _., OptionsPattern[]}}

Protect[MouseDoubleClick]

(* ::Subsection:: *)
(*MouseTripleClick*)

Unprotect[MouseTripleClick]

Options[MouseTripleClick] = FilterRules[Options[Mouse], {Modifiers, Button, Steps, InterpolationOrder, Scaling, CallSetFocusedNotebook}]

MouseTripleClick[nb:focusedNotebookPat:FocusedNotebook[], pts:{posPat, posPat}, opts:OptionsPattern[]] :=
	MouseTripleClick[nb, resolvePositions[pts], opts]

MouseTripleClick[nb:focusedNotebookPat:FocusedNotebook[], pt:pointPat, opts:OptionsPattern[]] :=
	MouseTripleClick[nb, {pt}, opts]

MouseTripleClick[nb:focusedNotebookPat:FocusedNotebook[], pts:pointListPat:{}, opts:OptionsPattern[]] :=
	MouseClick[nb, pts, Clicks -> 3, opts]

SetAttributes[MouseTripleClick, ReadProtected]

MouseTripleClick[arg_] :=
	(Message[MouseTripleClick::pntl, 1, HoldForm[MouseTripleClick[arg]]]; $Failed)

MouseTripleClick[arg1:nbobjPat, args__] :=
	(Message[MouseTripleClick::pntl, 2, HoldForm[MouseTripleClick[arg1, args]]]; $Failed)

MouseTripleClick[arg1_, arg2:pointPat|pointListPat] :=
	(Message[MouseTripleClick::nbobj, 1, HoldForm[MouseTripleClick[arg1, arg2]]]; $Failed)

MouseTripleClick[args___] :=
	(ArgumentCountQ[MouseTripleClick, System`FEDump`NonOptionArgCount[{args}], 0, 2]; $Failed)

SyntaxInformation[MouseTripleClick] = {"ArgumentsPattern" -> {_., _., OptionsPattern[]}}

Protect[MouseTripleClick]

(* ::Subsection:: *)
(*MouseDrag*)

Unprotect[MouseDrag]

Options[MouseDrag] = FilterRules[Options[Mouse], {Modifiers, Button, Steps, InterpolationOrder, Scaling, CallSetFocusedNotebook}]

MouseDrag[nb:focusedNotebookPat:FocusedNotebook[], pts:{posPat, posPat}, opts:OptionsPattern[]] :=
	MouseDrag[nb, resolvePositions[pts], opts]

MouseDrag[nb:focusedNotebookPat:FocusedNotebook[], pt:pointPat, opts:OptionsPattern[]] :=
	MouseDrag[nb, {pt}, opts]

MouseDrag[nb:focusedNotebookPat:FocusedNotebook[], pts:pointListPat, OptionsPattern[]] :=
	Module[{scaling, modifiers, steps, order, button, (*resolvedNB,*) vPoints, vScaling, vModifiers, vSteps, vOrder, vButton, buffer, fun, callSetFocusedNotebook},
		Catch[
			(* step 1: set variables and validate user input *)
			throwIfMessage[
				{scaling, modifiers, steps, order, button, callSetFocusedNotebook} = OptionValue[{Scaling, Modifiers, Steps, InterpolationOrder, Button, CallSetFocusedNotebook}];
				vScaling = validateScaling[MouseDrag, scaling];
				vModifiers = validateModifiers[MouseDrag, modifiers];
				vSteps = validateSteps[MouseDrag, steps];
				vOrder = validateInterpolationOrder[MouseDrag, order];
				vButton = validateButton[MouseDrag, button];
				vPoints = validatePoints[MouseDrag, nb, pts, vScaling]
			];
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							sowKeyModifiers[vModifiers,
								Switch[Length[vPoints],
									0,
									(* who would call MouseDrag[]? *)
									sowButton[vButton, Null]
									,
									1,
									(* who would call MouseDrag[{x, y}]? *)
									Sow[mouseMove[First[vPoints]]];
									sowButton[vButton, Null]
									,
									_,
									fun = interpolationPointsFunction[vPoints, vOrder];
									Sow[mouseMove[fun[1]]];
									sowButton[vButton,
										Do[
											Sow[mouseMove[fun[i]]]
											,
											(* remember, the first step isn't 2, it's 1+1/vSteps *)
											{i, 1+1/vSteps, Length[vPoints], 1/vSteps}
										]
									]
								]
							]
						]
				]
			];
			(* step 3: execute the buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer, CallSetFocusedNotebook -> callSetFocusedNotebook]
			]
		]
	]

SetAttributes[MouseDrag, ReadProtected]

MouseDrag[arg_] :=
	(Message[MouseDrag::pntl, 1, HoldForm[MouseDrag[arg]]]; $Failed)

MouseDrag[arg1:nbobjPat, args_] :=
	(Message[MouseDrag::pntl, 2, HoldForm[MouseDrag[arg1, args]]]; $Failed)

MouseDrag[arg1_, arg2:pointPat|pointListPat] :=
	(Message[MouseDrag::nbobj, 1, HoldForm[MouseDrag[arg1, arg2]]]; $Failed)

MouseDrag[args___] :=
	(ArgumentCountQ[MouseDrag, System`FEDump`NonOptionArgCount[{args}], 1, 2]; $Failed)

SyntaxInformation[MouseDrag] = {"ArgumentsPattern" -> {_., _, OptionsPattern[]}}

Protect[MouseDrag]

(* ::Subsection:: *)
(*$MouseClickDelay*)

$MouseClickDelay = 0

(* ::Subsection:: *)
(*$MouseDragDelay*)

$MouseDragDelay = 0.005

(* ::Subsection:: *)
(*$MouseMoveDelay*)

$MouseMoveDelay = 0.005
		
(* ::Subsection:: *)
(*$MousePressDelay*)

$MousePressDelay = 0

(* ::Subsection:: *)
(*$MouseReleaseDelay*)

$MouseReleaseDelay = 0
		
(* ::Subsection:: *)
(*$MouseWheelDelay*)

$MouseWheelDelay = 0.1

(* ::Subsection:: *)
(*Low-Level Functions*)

mouseMove[{x:numericPat, y:numericPat}] :=
	(
		delay[$MouseMoveDelay];
		$Robot@mouseMove[Round[x], Round[y]]
	)

(* mouseClick is referentially transparent *)
mouseClick[button:buttonPat] :=
	{delay[$MouseClickDelay], mousePress[button], mouseRelease[button]}

mousePress[button:buttonPat] :=
	(
		delay[$MousePressDelay];
		$Robot@mousePress[button /. $javaButtons]
	)

mouseRelease[button:buttonPat] :=
	(
		delay[$MouseReleaseDelay];
		$Robot@mouseRelease[button /. $javaButtons]
	)

mouseWheel[wheel:intPat] :=
	(
		delay[$MouseWheelDelay];
		$Robot@mouseWheel[wheel]
	)

(* ::Subsection:: *)
(*Validation*)

validateSteps[head:symbolPat, steps_] :=
	Which[
		MatchQ[steps, naturalPat],
		steps
		,
		True,
		Message[head::optvg, Steps, steps, "a non-negative integer"];
		$Failed
	]

validateInterpolationOrder[head:symbolPat, order_] :=
	Which[
		MatchQ[order, orderPat],
		order
		,
		True,
		Message[head::optvg, InterpolationOrder, order, "a non-negative integer or Automatic"];
		$Failed
	]

validateButton[head:symbolPat, button_] :=
	Which[
		MatchQ[button, buttonPat],
		button
		,
		True,
		Message[head::optvg, Button, button, "a button"];
		$Failed
	]

validateClicks[(*heads:*)symbolPat, clicks_] :=
	Which[
		MatchQ[clicks, wholePat],
		clicks
		,
		True,
		Message[head::optvg, Clicks, clicks, "a non-negative integer"];
		$Failed
	]

(* ::Subsection:: *)
(*End*)

End[] (*`Mouse`Private`*)
