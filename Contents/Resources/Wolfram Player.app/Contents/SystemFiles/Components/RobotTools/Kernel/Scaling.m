(* ::Package:: *)

(* ::Title:: *)
(*Scaling*)

(* ::Section:: *)
(*Annotations*)

(* :Title: Scaling.m *)

(* :Author:
        Brenton Bostick
        brenton@wolfram.com
*)

(* :Package Version: 1.2 *)

(* :Mathematica Version: 7.0 *)

(* :Copyright: RobotTools source code (c) 2005-2008 Wolfram Research, Inc. All rights reserved. *)

(* :Discussion:
   Implementation of Scaling-related functionality.
   
   The Scaled construct is used in RobotTools a little differently than it is used in graphics.
   
   Scaled[{0.5, 0.5}, "Screen"] represents the center of the screen
   {Rescale[0.5, {0, 1}, {0, maxx}], Rescale[0.5, {0, 1}, {0, maxy}]} is the equivalent
*)

(* ::Section:: *)
(*Information*)

`Information`CVS`$ScalingId = "$Id: Scaling.m,v 1.20 2008/04/07 19:23:55 pratikd Exp $"

(* ::Section:: *)
(*Public*)

BottomLeft::usage =
"BottomLeft is a symbol that represents the bottom-left corner for purposes of alignment and positioning."

BottomRight::usage =
"BottomRight is a symbol that represents the bottom-right corner for purposes of alignment and positioning."

RescaleRectangle::usage =
"RescaleRectangle is deprecated."

Scaling::usage =
"Scaling is an option for mouse and screen shot functions.
Scaling specifies the coordinate system of the coordinates."

TopLeft::usage =
"TopLeft is a symbol that represents the top-left corner for purposes of alignment and positioning."

TopRight::usage =
"TopRight is a symbol that represents the top-right corner for purposes of alignment and positioning."

(* ::Section:: *)
(*Package*)

Begin["`Package`"]

validatePoints

resolvePositions

validateScaling

End[] (*`Package`*)

(* ::Section:: *)
(*Private*)

(* ::Subsection:: *)
(*Begin*)

Begin["`Scaling`Private`"]

(* ::Subsection:: *)
(* Messages *)

General::offsc =
"The coordinates are outside of the viewing region: `1`."

Scaling::error =
"An invalid scaling pattern was given: `1`."

(* ::Subsection:: *)
(*RescaleRectangle*)

RescaleRectangle[rect:rectPat, s:numericPat] :=
	RescaleRectangle[rect, {s, s}]

RescaleRectangle[rect:rectPat, scale:pointPat] :=
	(
		Message[RescaleRectangle::usage];
		MapThread[Rescale[#1, {0, 1}, #2]&, {scale, Transpose[rect]}]
	)

RescaleRectangle[args___] :=
	(ArgumentCountQ[RescaleRectangle, System`FEDump`NonOptionArgCount[{args}], 2, 2]; Null /; False)

SyntaxInformation[RescaleRectangle] = {"ArgumentsPattern" -> {_, _}}

(* ::Subsection:: *)
(*Scaling Functions*)

(* TODO: properly retain information about each screen and don't blindly treat union of all screen rectangles as contiguous *)

screensIntervals :=
	Thread[Interval @@ ("FullScreenArea" /. Lookup[SystemInformation["Devices"],"ScreenInformation"])]

screensRect :=
	Transpose[#-{0, 1}& /@ Flatten /@ List @@@ screensIntervals]

(* TODO: give messages when notebook is not valid or specified box is not present *)

(*
even though nb:focusedNotebook goes against convention here, it is valid.
FocusedNotebook[] will only ever get passed in when there is no front end interaction
*)
resolveCoordinates[head:symbolPat, nb:focusedNotebookPat, Scaled[point:numericPointPat, scaling:scalingPat]] :=
	Module[{range, scale, nPt},
		{range, scale} =
			Transpose /@ Switch[scaling,
				None,
				{screensRect, screensRect}
				,
				"Screen",
				{{{0, 0}, {1, 1}}, screensRect}
				,
				"Notebook",
				{{{0, 0}, {1, 1}}, getWindowRectangle[nb]}
				,
				"Selection",
				{{{0, 0}, {1, 1}}, getSelectionBoundingBoxes[nb][[1, {1, 3}]]}
				,
				nbobjPat,
				{{{0, 0}, {1, 1}}, getWindowRectangle[scaling]}
				,
				{nbobjPat},
				{{{0, 0}, {1, 1}}, getWindowRectangle[scaling[[1]]]}
				,
				boxPat,
				{{{0, 0}, {1, 1}}, getBoxRectangles[nb, scaling, True, {}][[1]]}
				,
				{boxPat},
				{{{0, 0}, {1, 1}}, getBoxRectangles[nb, scaling[[1]], True, {}][[1]]}
				,
				{boxPat, naturalPat},
				{{{0, 0}, {1, 1}}, getBoxRectangles[nb, scaling[[1]], True, {}][[scaling[[2]]]]}
				,
				{nbobjPat, boxPat},
				{{{0, 0}, {1, 1}}, getBoxRectangles[scaling[[1]], scaling[[2]], True, {}][[1]]}
				,
				{nbobjPat, boxPat, naturalPat},
				{{{0, 0}, {1, 1}}, getBoxRectangles[scaling[[1]], scaling[[2]], True, {}][[scaling[[3]]]]}
				,
				{nbobjPat, Cell},
				{{{0, 0}, {1, 1}}, getBoxRectangles[scaling[[1]], BoxData, True, {}][[1]]}
				,
				{nbobjPat, Cell, naturalPat},
				{{{0, 0}, {1, 1}}, getBoxRectangles[scaling[[1]], BoxData, True, {}][[scaling[[3]]]]}
				,
				{nbobjPat, "Selection"},
				{{{0, 0}, {1, 1}}, getSelectionBoundingBoxes[scaling[[1]]][[1, {1, 3}]]}
				,
				{{numericPat, numericPat}, {numericPat, numericPat}},
				{{{0, 0}, {1, 1}}, scaling}
				,
				_,
				Message[Scaling::error, scaling];
				$Failed
			];
		nPt = RescalingTransform[range, scale][point];
		(* test if coordinates are on the screen *)
		If[Thread[Unevaluated[And @@ IntervalMemberQ[screensIntervals, nPt]]],
			nPt
			,
			Message[head::offsc, nPt];
			$Failed
		]
	]

resolvePositions[Scaled[s_, scaling:scalingPat]] :=
	Scaled[resolvePositions[s], scaling]

resolvePositions[pt:pointPat] :=
	ReplaceRepeated[pt, {
		Center -> {1/2, 1/2},
		Left -> {0, 1/2},
		Right -> {1, 1/2},
		Top -> {1/2, 0},
		TopLeft -> {0, 0},
		TopRight -> {1, 0},
		Bottom -> {1/2, 1},
		BottomLeft -> {0, 1},
		BottomRight -> {1, 1},
		{Center, y_} :> {1/2, y},
		{Left, y_} :> {0, y},
		{Right, y_} :> {1, y},
		{x_, Center} :> {x, 1/2},
		{x_, Top} :> {x, 0},
		{x_, Bottom} :> {x, 1}
	}]

(*resolveAll[All] =
	{{0, 0}, {1, 1}}

resolveAll[pts_] :=
	pts*)

(*resolveRange[range:scalingPat] :=
	If[range === None, screenRect, {{0, 0}, {1, 1}}]*)

(* TODO: validate points, make sure that nothing like {Top, 0.8} or {0.4, Left} is given *)

(* TODO: give messages if Scaled has no 2nd arg *)

(* TODO: allow nested Scaled[]s, maybe using Inherited? *)

(*resolveCoordinates[nb:nbobjPat, Scaled[point:numericPointPat, scaling:scalingPat]] :=
	RescalingTransform[resolveScaling[nb, scaling], (*placeholder to silence Workbench*)Sequence[]][point]*)

(* the default of None for scaling indicates that coordinates are in screen coordinates *)
validatePoints[head:symbolPat, nb:focusedNotebookPat, pts:pointListPat, scaling:scalingPat:None] :=
	Module[{resolvedNB, frontEndInteractionFlag = False, ptsScaling, nPts},
		(* determine overall scaling *)
		ptsScaling = If[Head[pts] === Scaled, pts[[2]], scaling];
		(* convert appropriate nPts to {{0, 0}, {1, 1}}, if possible *)
		nPts = Replace[pts, All -> {{0, 0}, {1, 1}}];
		(* make sure all points are wrapped in Scaled, there's probably a better way to do this... *)
		nPts = If[Head[#] =!= Scaled, Scaled[#, ptsScaling], #]& /@ If[Head[nPts] === Scaled, nPts[[1]], nPts];
		(* convert symbolic positions, resolvePositions does not touch Scaled *)
		nPts = resolvePositions /@ nPts;
		(*
		resolve scaling
		go through each point in nPts and see if it needs to call the front end
		The thinking here is that if a scaling has been specified that needs to use the front end, then it is ok to also call the front end to
		figure out InputNotebook[]. If the front end does not need to be called, then there is no point in calling InputNotebook[].
		
		The Scan statement below tries to catch all possible patterns of scalings.
		It could be tidied up to group all front-end-needing patterns together, and everything else together, but in the future there may be
		some kind of distinction between different packets that are needed, so for right now just keep them separate.
		*)
		Scan[
			Switch[#[[2]],
				None,
				Null
				,
				"Screen",
				Null
				,
				"Notebook",
				frontEndInteractionFlag = True
				,
				"Selection",
				frontEndInteractionFlag = True
				,
				nbobjPat | {nbobjPat},
				frontEndInteractionFlag = True
				,
				boxPat | {boxPat} | {{boxPat}} | {boxPat, naturalPat} | {{boxPat, naturalPat}} | {nbobjPat, boxPat} | {nbobjPat, {boxPat}} |
				{nbobjPat, boxPat, naturalPat} | {nbobjPat, {boxPat, naturalPat}},
				frontEndInteractionFlag = True
				,
				{nbobjPat, "Selection"},
				frontEndInteractionFlag = True
				,
				{numericPointPat, numericPointPat},
				Null
			]&
			,
			nPts
		];
		
		(* this guarantees that InputNotebook[] is only evaluated if the scalings need the front end, and a literal FocusedNotebook[] is given *)
		resolvedNB = If[frontEndInteractionFlag, resolveFocusedNotebook[nb], nb];
		
		(* this guarantees that UndocumentedBoxInformationPacket is sent only once, and that is only if the scalings need the front end *)
		nPts =
			If[frontEndInteractionFlag,
				blockBoxInformation[{resolvedNB, True, {}},
					(* each point has a Scaled wrapped around it *)
					(*nPts = If[#[[2]] === None, #[[1]], f[resolveScaling[nb, #[[2]]], #[[1]], {{0, 0}, {1, 1}}]]& /@ nPts*)
					resolveCoordinates[head, resolvedNB, #]& /@ nPts
				]
				,
				resolveCoordinates[head, resolvedNB, #]& /@ nPts
			];
		nPts
	]

(* ::Subsection:: *)
(*Validation*)

validateScaling[head:symbolPat, scaling_] :=
	Which[
		MatchQ[scaling, scalingPat],
		scaling
		,
		True,
		Message[head::optvg, Scaling, scaling, "a valid scaling expression"];
		$Failed
	]

(* ::Subsection:: *)
(*End*)

End[] (*`ScreenShot`Private`*)
