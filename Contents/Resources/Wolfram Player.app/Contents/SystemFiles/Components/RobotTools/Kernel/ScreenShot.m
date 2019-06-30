(* ::Package:: *)

(* ::Title:: *)
(*ScreenShot*)

(* ::Section:: *)
(*Annotations*)

(* :Title: ScreenShot.m *)

(* :Author:
        Brenton Bostick
        brenton@wolfram.com
*)

(* :Package Version: 1.2 *)

(* :Mathematica Version: 7.0 *)

(* :Copyright: RobotTools source code (c) 2005-2008 Wolfram Research, Inc. All rights reserved. *)

(* :Discussion:
   Implementation of screenshot-related functionality.
*)

(* ::Section:: *)
(*Information*)

`Information`CVS`$ScreenShotId = "$Id: ScreenShot.m,v 1.29 2008/07/08 14:03:52 brenton Exp $"

(* ::Section:: *)
(*Public*)

GetPixelColor::usage =
"GetPixelColor[point] returns the color of the pixel at the point given."

RasterizeNotebook::usage =
"RasterizeNotebook[nb] returns a raster of the client area of nb."

RasterizeScreenShot::usage =
"RasterizeScreenShot[rect] returns a raster of the specified portion of the screen."

RasterizeSelection::usage =
"RasterizeSelection[nb] returns a raster of the current MathEdit selection in the notebook object nb."

ScreenShot::usage =
"ScreenShot[rect] returns an array of RGB triples representing the specified area of the screen."

(* ::Section:: *)
(*Package*)

Begin["`Package`"]

getPixelColor

rasterizeScreenShot

screenShot

nativeScreenShot

End[] (*`Package`*)

(* ::Section:: *)
(*Private*)

(* ::Subsection:: *)
(*Begin*)

Begin["`ScreenShot`Private`"]

(* ::Subsection:: *)
(*ScreenShot*)

Unprotect[ScreenShot]

Options[ScreenShot] = {Scaling -> None}

ScreenShot[nb:focusedNotebookPat:FocusedNotebook[], rect:rectPat, OptionsPattern[]] :=
	Module[{scaling, vScaling, vRect, buffer},
		Catch[
			(* step 1: set variables and validate user input *)
			throwIfMessage[
				scaling = OptionValue[Scaling];
				vScaling = validateScaling[ScreenShot, scaling];
				vRect = validatePoints[ScreenShot, nb, rect, vScaling]
			];
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							Sow[screenShot[vRect]]
						]
				]
			];
			(* step 3: execute buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer]
			]
		]
	]

SetAttributes[ScreenShot, ReadProtected]

ScreenShot[args___] :=
	(ArgumentCountQ[ScreenShot, System`FEDump`NonOptionArgCount[{args}], 1, 3]; $Failed)

SyntaxInformation[ScreenShot] = {"ArgumentsPattern" -> {_., _, OptionsPattern[]}}

Protect[ScreenShot]

(* ::Subsection:: *)
(*rasterizeScreenShot*)

(* TODO: handle Graphics options better *)

rasterizeScreenShot[rect:numericRectPat, opts:OptionsPattern[]] :=
	Module[{normalRect, transposedNormalRect},
		normalRect = {{0, 0}, First[Differences[rect]]};
		transposedNormalRect = Transpose[normalRect];
		Graphics[
			Raster[screenShot[rect], normalRect, {0, 255}],
			opts,
			PlotRegion -> {{0, 1}, {0, 1}},
			PlotRange -> transposedNormalRect,
			ImageSize -> transposedNormalRect
		]
	]

(* ::Subsection:: *)
(*RasterizeScreenShot*)

Unprotect[RasterizeScreenShot]

Options[RasterizeScreenShot] = {Scaling -> None}

RasterizeScreenShot[nb:focusedNotebookPat:FocusedNotebook[], rect:rectPat, opts:OptionsPattern[]] :=
	Module[{scaling, (*resolvedNB,*) vScaling, vRect, graphicsOpts, buffer},
		Catch[
			(* step 1: set variables and validate user input *)
			throwIfMessage[
				scaling = OptionValue[Scaling];
				graphicsOpts = FilterRules[Flatten[{opts}], Options[Graphics]];
				vScaling = validateScaling[RasterizeScreenShot, scaling];
				vRect = validatePoints[RasterizeScreenShot, nb, rect, vScaling]
			];
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							Sow[rasterizeScreenShot[vRect, graphicsOpts]]
						]
				]
			];
			(* step 3: execute buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer]
			]
		]
	]

SetAttributes[RasterizeScreenShot, ReadProtected]

RasterizeScreenShot[args___] :=
	(ArgumentCountQ[RasterizeScreenShot, System`FEDump`NonOptionArgCount[{args}], 1, 3]; $Failed)

SyntaxInformation[RasterizeScreenShot] = {"ArgumentsPattern" -> {_., _, OptionsPattern[]}}

Protect[RasterizeScreenShot]

(* ::Subsection:: *)
(*RasterizeSelection*)

Unprotect[RasterizeSelection]

RasterizeSelection[nb:focusedNotebookPat:FocusedNotebook[], opts:OptionsPattern[]] :=
	Module[{coors, resolvedNB, buffer},
		Catch[
			(* step 1: set variables and validate user input *)
			throwIfMessage[
				resolvedNB = resolveFocusedNotebook[nb];
				coors = getSelectionBoundingBoxes[resolvedNB]
			];
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							Sow[rasterizeScreenShot[coors[[1, {1, 3}]], opts]]
						]
				]
			];
			(* step 3: execute buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer]
			]
		]
	]

SetAttributes[RasterizeSelection, ReadProtected]

RasterizeSelection[args___] :=
	(ArgumentCountQ[RasterizeSelection, System`FEDump`NonOptionArgCount[{args}], 0, 1]; $Failed)

SyntaxInformation[RasterizeSelection] = {"ArgumentsPattern" -> {_.}}

Protect[RasterizeSelection]

(* ::Subsection:: *)
(*RasterizeNotebook*)

Unprotect[RasterizeNotebook]

RasterizeNotebook[nb:focusedNotebookPat:FocusedNotebook[], opts:OptionsPattern[]] :=
	Module[{rect, resolvedNB, buffer},
		Catch[
			(* step 1: set variables and validate user input *)
			throwIfMessage[
				resolvedNB = resolveFocusedNotebook[nb];
				rect = getWindowRectangle[resolvedNB]
			];
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							Sow[rasterizeScreenShot[rect, opts]]
						]
				]
			];
			(* step 3: execute buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer]
			]
		]
	]

SetAttributes[RasterizeNotebook, ReadProtected]

RasterizeNotebook[args___] :=
	(ArgumentCountQ[RasterizeNotebook, System`FEDump`NonOptionArgCount[{args}], 0, 1]; $Failed)

SyntaxInformation[RasterizeNotebook] = {"ArgumentsPattern" -> {_.}}

Protect[RasterizeNotebook]

(* ::Subsection:: *)
(*GetPixelColor*)

Unprotect[GetPixelColor]

Options[GetPixelColor] = {Scaling -> None}

GetPixelColor[nb:focusedNotebookPat:FocusedNotebook[], pt:pointPat, OptionsPattern[]] :=
	Module[{scaling, (*resolvedNB,*) vScaling, vPoint, buffer},
		Catch[
			(* step 1: set variables and validate user input *)
			throwIfMessage[
				scaling = OptionValue[Scaling];
				vScaling = validateScaling[GetPixelColor, scaling];
				{vPoint} = validatePoints[GetPixelColor, nb, {pt}, vScaling]
			];
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							Sow[getPixelColor[vPoint]]
						]
				]
			];
			(* step 3: execute buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer]
			]
		]
	]

SetAttributes[GetPixelColor, ReadProtected]

GetPixelColor[args___] :=
	(ArgumentCountQ[GetPixelColor, System`FEDump`NonOptionArgCount[{args}], 1, 2]; $Failed)

SyntaxInformation[GetPixelColor] = {"ArgumentsPattern" -> {_., _, OptionsPattern[]}}

Protect[GetPixelColor]

(* ::Subsection:: *)
(*Low-Level Functions*)

getPixelColor[{x:numericPat, y:numericPat}] :=
	Module[{color, k, r, g, b, a},
		JavaBlock[
			color = $Robot@getPixelColor[x, y];
			k = color@getRGB[];
			{b, g, r, a} = Mod[NestList[Quotient[#, 256]&, k, 3], 256]/255
		];
		RGBColor[r, g, b, a]
	]


getOffsets[] := Block[{intervals, x,y},
	intervals = RobotTools`Scaling`Private`screensIntervals;
	x = -1*intervals[[1, 1, 1]];
	y = intervals[[2, 1, 2]];
	{x, y}
]

screenShot[{{x1:numericPat, y1:numericPat}, {x2:numericPat, y2:numericPat}}] :=
	Block[{width = Round[x2 - x1], height = Round[y2 - y1], rect, bufferedImage,
				integerRaster, pixels, offsetX, offsetY, newX, newY},
		{offsetX, offsetY} = getOffsets[];
		newX = Round[x1-offsetX];
		newY = Round[offsetY-y1-height];
		JavaBlock[
			pixels = $Robot@createScreenCapture[
				JavaNew["java.awt.Rectangle", newX, newY, width, height]
			]@getData[]@getPixels[
				0, 
				0, 
				width, 
				height, 
				JavaNew["[I", 3 width height]
			];
			Reverse[ArrayReshape[pixels,{height,width,3}]]
		]
	]

(*native screenshot will attempt to take a "native" resolution screenshot, i.e. with full pixels*)
(*instead of the scaled one that createScreenCapture normally does*)
(*this requires Java 9 to work, so in this function we go to check if Java 9 is running*)
(*if not, then we will reinstall java as java 9*)
nativeScreenShot[{{x1:numericPat, y1:numericPat}, {x2:numericPat, y2:numericPat}}] :=
	Block[
		{width = Round[x2 - x1], height = Round[y2 - y1], bufferedImage, pixels, obj,
				imgs, res, graphicsBuffer, offsetX, offsetY, newX, newY},
		{offsetX, offsetY} = getOffsets[];
		newX = Round[x1-offsetX];
		newY = Round[offsetY-y1-height];
		If[RobotTools`HavaJava9,
			(*THEN*)
			(*we have java 9, so run the new createMultiResolutionScreenCapture method to get a native function*)
			JavaBlock[
				rect = JavaNew["java.awt.Rectangle", newX, newY, width, height];
				imgs = $Robot@createMultiResolutionScreenCapture[rect]@getResolutionVariants[];
				imageObj = If[imgs@size[] == 2,
					imgs@get[1],
					imgs@get[0]
				];
				(*intialize a new buffered image which will store the bytes for us*)
				bufferedImage = JavaNew[
					"java.awt.image.BufferedImage", 
					imageObj@getWidth[],
					imageObj@getHeight[],
					BufferedImage`TYPEUINTURGB
				];
				(*now make a graphics buffer for the buffered image*)
				graphicsBuffer = bufferedImage@createGraphics[];
				(*draw the real image onto the buffered image's graphics object to copy the pixels over*)
				graphicsBuffer@drawImage[imageObj, 0, 0, Null];
				(*now get the raw pixels from the buffered image*)
				pixels = bufferedImage@getData[]@getPixels[
					0,
					0,
					imageObj@getWidth[],
					imageObj@getHeight[],
					JavaNew["[I", 3*imageObj@getWidth[]*imageObj@getHeight[]]
				];
				(*reshape the pixels to make the image in the right order*)
				res = Reverse[ArrayReshape[pixels, {imageObj@getHeight[], imageObj@getWidth[], 3}]];
				(*dispose of the graphics object*)
				graphicsBuffer@dispose[];
				(*return the shaped array*)
				res
			],
			(*ELSE*)
			(*we don't have Java 9 to run this, so just do it normally*)
			screenShot[{{x1,y1},{x2,y2}}]
		]
	]




(* ::Subsection:: *)
(*End*)

End[] (*`ScreenShot`Private`*)
