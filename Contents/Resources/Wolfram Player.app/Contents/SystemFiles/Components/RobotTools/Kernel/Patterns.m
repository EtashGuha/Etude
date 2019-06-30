(* ::Package:: *)

(* ::Title:: *)
(*Patterns*)

(* ::Section:: *)
(*Annotations*)

(* :Title: Patterns.m *)

(* :Author:
        Brenton Bostick
        brenton@wolfram.com
*)

(* :Package Version: 1.2 *)

(* :Mathematica Version: 7.0 *)

(* :Copyright: RobotTools source code (c) 2005-2008 Wolfram Research, Inc. All rights reserved. *)

(* :Discussion:
   Several pattern definitions that are used throughout the package.
   Most functions definitions in RobotTools will have patterns like foo[bar:barPat].
   Even though it may look silly to see foo[bar:stringPat], if this is missing, this helps to warn the developer that a function
   may not have the correct pattern. Several patterns build on simpler patterns, so for example, this guarantess that mal-formed
   points are not passed to mouse functions.
*)

(* ::Section:: *)
(*Information*)

`Information`CVS`$PatternsId = "$Id: Patterns.m,v 1.23 2010/10/21 20:45:57 brenton Exp $"

(* ::Section:: *)
(*Package*)

Begin["`Package`"]

blankPat

boolPat

boxPat

boxListPat

buttonPat

focusedNotebookPat

gridPat

heldListPat

intPat

intListPat

iteratedStringPat

listPat

keyCmdsPat

keyModifierPat

keyModifierListPat

menuHeadPat

menuItemNamePat

menuKeyPat

menuKeyModifierPat

menuKeyModifierListPat

menuMethodPat

menuPathPat

metaCharPat

modifiedCharPat

modifierListPat

naturalPat

nbobjPat

nestringPat

nestringListPat

numericPat

numericPointPat

numericPointListPat

numericRectPat

orderPat

pointPat

pointListPat

posPat

prunedBoxesPat

rectPat

scalingPat

stringPat

stringListPat

symbolPat

wholePat

End[] (*`Package`*)

Begin["`Patterns`Private`"]

blankPat =
	_

boolPat =
	True | False

listPat =
	_List

heldListPat =
	Hold[{___}]

gridPat =
	{{__}..}

stringPat =
	_String

NonemptyStringQ[s_] := StringQ[s] && StringLength[s] > 0

(*
nestringPat used to be defined as
s_String /; StringLength[s] > 0
but since nestringPat is used within larger patterns,
the s stays binded throughout the entire pattern.
So, in a sequence of nestringPat, all of them had to be the same string.
Cannot use named patterns in here!
*)
nestringPat =
	_?NonemptyStringQ

stringListPat =
	{stringPat...}

nestringListPat =
	{nestringPat ...}

modifiedCharPat =
	"\[LeftModified]" | "\[RightModified]"

metaCharPat =
	"\[LeftModified]" | "\[RightModified]" | "\[KeyBar]"

intPat =
	_Integer

intListPat =
	{intPat...}

naturalPat =
	intPat?Positive

wholePat =
	intPat?NonNegative

symbolPat =
	_Symbol

boxPat =
	RobotTools`BasicBox |
	RobotTools`DummyPrimitiveBox |
	RobotTools`InlineGraphicBox |
	RobotTools`LineWrapBox |
	RobotTools`NumberBox |
	RobotTools`VerticalSpanBox |
	List |
	BoxData |
	ActionMenuBox | AdjustmentBox | AnimatorBox | Arrow3DBox | ArrowBox | BezierCurve3DBox | BezierCurveBox |
	BSplineCurve3DBox| BSplineCurveBox | ButtonBox | CheckboxBox | CircleBox | ColorSetterBox | ConeBox |
	CounterBox | CuboidBox | CylinderBox | DiskBox | DynamicBox | DynamicModuleBox | DynamicWrapperBox |
	ErrorBox | FormBox | FractionBox | FrameBox | GeometricTransformation3DBox | GeometricTransformationBox |
	Graphics3DBox | GraphicsBox | GraphicsComplex3DBox | GraphicsComplexBox | GraphicsGridBox |
	GraphicsGroup3DBox | GraphicsGroupBox | GridBox | InputFieldBox | Inset3DBox | InsetBox |
	InterpretationBox | ItemBox | Line3DBox | LineBox | LocatorBox | LocatorPaneBox | OpenerBox |
	OptionValueBox | OverscriptBox | PaneBox | PanelBox | PaneSelectorBox | Point3DBox | PointBox |
	Polygon3DBox | PolygonBox | PopupMenuBox | ProgressIndicatorBox | RadicalBox | RadioButtonBox | RasterBox |
	RectangleBox | RotationBox | RowBox | SetterBox | Slider2DBox | SliderBox | SphereBox | SqrtBox | StyleBox |
	SubscriptBox | SubsuperscriptBox | SuperscriptBox | TabViewBox | TagBox | TemplateBox | Text3DBox |
	TextBox | TogglerBox | TooltipBox | TubeBox | UnderoverscriptBox | UnderscriptBox | ValueBox
	
boxListPat =
	{boxPat...}

prunedBoxesPat =
	boxListPat | Automatic

buttonPat =
	"Button1" | "Button2" | "Button3" |
	(* these are deprecated *)
	"MouseButton1" | "MouseButton2" | "MouseButton3"

numericPat =
	_?NumericQ

posPat =
	Bottom | RobotTools`BottomLeft | RobotTools`BottomRight | Center | Left | Right | Top | RobotTools`TopLeft | RobotTools`TopRight

menuHeadPat =
	MenuItem |
	Menu |
	FrontEnd`HelpMenu

menuItemNamePat =
	stringPat |
	(* these come from GetPopupList["MenuListNotebookEvaluators"] *)
	Default | None | Automatic

menuKeyPat =
	_FrontEnd`MenuKey

menuKeyModifierPat =
	"Command" | "Control" | "Shift" | "Option"

menuMethodPat =
	stringPat | (*stringListPat |*)
	Automatic

nbobjPat =
	_NotebookObject

focusedNotebookPat =
	nbobjPat |
	RobotTools`FocusedNotebook[]

menuPathPat =
	nestringListPat

keyModifierPat =
	"\[AltKey]" | "\\[AltGrKey]" | "\[CommandKey]" | "\[ControlKey]" | "\[OptionKey]" | "\[ShiftKey]" |
	(* http://support.microsoft.com/default.aspx?scid=kb;en-us;301583 *)
	"\\[WindowsKey]"

keyModifierListPat =
	{keyModifierPat...}

orderPat =
	wholePat |
	Automatic


menuKeyModifierListPat =
	{menuKeyModifierPat...}

modifierListPat =
	{(menuKeyModifierPat | keyModifierPat)...}


iteratedStringPat =
	{stringPat, intPat} | RepeatedString[stringPat, intPat]

keyCmdsPat =
	{(stringPat | iteratedStringPat)...}


numericPointPat =
	{numericPat, numericPat}

numericPointListPat =
	{numericPointPat...}

scalingPat =
	None |
	"Screen" | "Notebook" | "Selection" | (* TODO: all valid values for MousePosition[] *)
	nbobjPat | {nbobjPat} |
	boxPat | {boxPat} | {{boxPat}} |
	{boxPat, naturalPat} | {{boxPat, naturalPat}} |
	{nbobjPat, boxPat} | {nbobjPat, {boxPat}} | {nbobjPat, boxPat, naturalPat} | {nbobjPat, {boxPat, naturalPat}} |
	(*
	TODO: some kind of thumb syntax for sliders
	{nbobjPat, {boxPat, naturalPat}, } |
	*)
	{nbobjPat, "Selection"} |
	(* don't use rectPat or pointPat, since they both refer to scalingPat *)
	{numericPointPat, numericPointPat}

pointPat =
	numericPointPat |
	Scaled[numericPointPat, scalingPat] |
	posPat |
	Scaled[posPat, scalingPat] |
	{posPat, posPat} |
	Scaled[{posPat, posPat}, scalingPat]

pointListPat =
	{pointPat...} |
	Scaled[{pointPat...}, scalingPat] |
	All |
	Scaled[All, scalingPat]

numericRectPat =
	{numericPointPat, numericPointPat}

rectPat =
	{pointPat, pointPat} |
	Scaled[{pointPat, pointPat}, scalingPat] |
	All |
	Scaled[All, scalingPat]

End[] (*`Patterns`Private`*)
