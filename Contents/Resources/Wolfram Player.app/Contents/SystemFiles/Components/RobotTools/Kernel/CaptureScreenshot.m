
BeginPackage["RobotTools`CaptureScreenshot`",
	{
		"PacletManager`",
		"RobotTools`"
	}
]

If[!$CloudEvaluation,
	InstallRobotTools[]
]


(*messages*)
System`CurrentNotebookImage::canvasNBClipping = "Part of the notebook `1` is off the full screen canvas on the `2` side and may be clipped.";
System`CurrentNotebookImage::invalidNB = "The notebook `1` is invalid, CurrentNotebookImage only works on valid NotebookObjects.";
System`CurrentNotebookImage::invalid = "The arguments `1` are invalid, CurrentNotebookImage only works on valid NotebookObjects.";
System`CurrentNotebookImage::monitorClipping = "Part of the notebook `1` is off the monitor canvas on the `2` side and may be clipped.";

System`CurrentScreenImage::canvasRectClipping = "Part of the rectangle `1` is off the full screen canvas on the `2` side and may be clipped.";
System`CurrentScreenImage::rect="The rectangle `1` is invalid - rectangles must be of the form {{xmin,ymin},{xmax,ymax}}.";
System`CurrentScreenImage::monitorNum="The monitor `1` is invalid - valid monitors are `2`.";
System`CurrentScreenImage::invalid = "The argument `1` is invalid, CurrentScreenImage accepts no arguments, an integer representing the monitor, or a rectangle of the screen position to capture.";


Begin["`Private`"]

(*this is necessary to prevent recursive autoloading of the System` symbols that the paclet manager sets up for us*)
(Unprotect[#];
ClearAttributes[#,{Stub,Protected,ReadProtected}];
Clear[#];)&/@{
	"System`CurrentNotebookImage",
	"System`CurrentScreenImage"
}

(*check if we need to figure out the border widths on windows*)
If[StringMatchQ[$OperatingSystem,___~~"Windows"~~___],
	(
		(*use netlink to get this information*)
		Quiet[Needs["NETLink`"]];
		NETLink`InstallNET[];
		NETLink`LoadNETType["System.Windows.Forms.SystemInformation"];
		$WindowsFrameBorderWidth = System`Windows`Forms`SystemInformation`FrameBorderSize@Width[];
		$WindowsFrameBorderHeight = System`Windows`Forms`SystemInformation`FrameBorderSize@Height[];
	)
];

(*this gets an association with all the relevant position information of a notebook*)
getNotebookPositionInfoAssoc[notebook_NotebookObject]/;MemberQ[Notebooks[],notebook]:=Block[{},
	(
		Join[
			(*we use the first element of "ScreenInformation" as the primary monitor*)
			AssociationThread[{"xStart", "xEnd", "yStart", "yEnd"}, Join @@ (Transpose@First@getScreenRegions[$screenInfoCache])],
			AssociationThread[{"xSize", "ySize"}, AbsoluteCurrentValue[notebook,WindowSize]],
			(*if we're on windows, remove off the extra pixel margins - see http://stackoverflow.com/questions/18236364/net-systeminformation-primarymonitormaximizedwindowsize-shows-larger-than-prima*)
			If[StringMatchQ[$OperatingSystem,___~~"Windows"~~___],
				MapAt[
					# + $WindowsFrameBorderWidth&,
					MapAt[
						# + $WindowsFrameBorderHeight&,
						#,
						{{"bottomMargin"},{"topMargin"}}
					],
					{{"leftMargin"},{"rightMargin"}}
				],
				#
			]&@If[ListQ[#],
				AssociationThread[{"leftMargin", "rightMargin", "bottomMargin", "topMargin"}, Join@@#],
				(*in the case where the window is maximized or the margins are all exactly the same, WindowMargins will be a single number*)
				AssociationThread[{"leftMargin", "rightMargin", "bottomMargin", "topMargin"},Table[#,4]]
			]&@AbsoluteCurrentValue[notebook,WindowMargins]
		]
	)
];

(*this method calculates an association of the OS margins on a particular monitor from the center of a window*)
calcMarginAssoc[{{xLeft_,yTop_},{xRight_,yBottom_}}]:=Block[
	{
		xCenter,
		yCenter,
		allMonitorPositions,
		monitorPosition
	},
	(
		(*allMonitorPositions= (Lookup[#, "FullScreenArea"]) & /@ $screenInfoCache;*)
		allMonitorPositions = Transpose/@getScreenRegions[$screenInfoCache];
		(*calculate first the center of this notebook, and which monitor this window is *primarily* on, note that part of it could also be*)
		(*on another monitor, but we just arbitrarily choose the monitor this primarily shows up on as the one to use for system margin*)
		(*calculation*)
		xCenter = xLeft + Round[EuclideanDistance[xRight,xLeft]/2];
		yCenter = yBottom + Round[EuclideanDistance[yTop,yBottom]/2];

		(*now see which of the FullScreenArea's this point belongs to - note just need to find the first one*)
		monitorPosition = LengthWhile[allMonitorPositions, Not[(IntervalMemberQ[Interval[First[#]], xCenter] && IntervalMemberQ[Interval[Last[#]], yCenter])] &] + 1;
		(*now that we know the monitor positions of this, so calculate the system margins for this monitor*)
		Append[
			calcMarginAssocForMonitorNumber[monitorPosition],
			"Monitor"->monitorPosition
		]
	)
];
 
(*we have to take the absolute value of these because it could be negative, but because it's always done for the same monitor (the one specified)*)
(*it will always be valid if we take the absolute value of it*)
calcMarginAssocForMonitorNumber[num_?IntegerQ]:=Abs/@
	AssociationThread[
		{"systemLeftMargin","systemRightMargin","systemTopMargin","systemBottomMargin"},
		(*this calculates the difference between the full screen area and the screen area for the specified monitor with the order of dimensions as specified above*)
		Join@@Transpose[Subtract@@(Transpose@Lookup[$screenInfoCache[[num]],#]&/@{"ScreenArea","FullScreenArea"})]
	]
 
(*this turns the position info assoc from the above function into a simple rectangle representing usable notebook position for CaptureScreenshot*)
getNotebookPosition[notebook_NotebookObject]/;MemberQ[Notebooks[],notebook]:=Block[
	{
		infoAssoc = getNotebookPositionInfoAssoc[notebook],
		systemMargin,
		xLeft, xRight, yTop, yBottom,
		xMinCanvas, yMinCanvas, xMaxCanvas, yMaxCanvas,
		xMinMonitor, yMinMonitor, xMaxMonitor, yMaxMonitor,
		xLeftClipped, xRightClipped, yTopClipped, yBottomClipped
	},
	(	
		(*just subtract off from the x and y ends the right and bottom margins*)
		(*and add on the left and top margins to the x and y starts*)
		xRight = infoAssoc["xEnd"] - infoAssoc["rightMargin"];
		xLeft = infoAssoc["xStart"] + infoAssoc["leftMargin"];
		yTop = infoAssoc["yEnd"] - infoAssoc["topMargin"];
		yBottom = infoAssoc["yStart"] + infoAssoc["bottomMargin"];

		(*now calculate handle the system margins, these are based on the primary monitor*)
		systemMargin = calcMarginAssocForMonitorNumber[1];

		xRight -= systemMargin["systemRightMargin"];
		xLeft += systemMargin["systemLeftMargin"];
		yTop -= systemMargin["systemTopMargin"];
		yBottom += systemMargin["systemBottomMargin"];


		(*find which monitor we are on from the rectangle*)
		systemMargin = calcMarginAssoc[{{xLeft,yTop},{xRight,yBottom}}];

		{{xMinCanvas,yMinCanvas},{xMaxCanvas,yMaxCanvas}} = calculateFullMonitorCanvas[];

		(*before returning the positions, check to make sure that they are within the full canvas*)
		(*clipping them to the canvas if they would otherwise be off of the canvas*)
		If[xLeft < xMinCanvas,
			Message[System`CurrentNotebookImage::canvasNBClipping,notebook,"left"];
			xLeft = xMinCanvas
		];
		If[yTop < yMinCanvas,
			Message[System`CurrentNotebookImage::canvasNBClipping,notebook,"top"];
			yTop = yMinCanvas
		];
		If[xRight > xMaxCanvas,
			Message[System`CurrentNotebookImage::canvasNBClipping,notebook,"right"];
			xRight = xMaxCanvas
		];
		If[yBottom > yMaxCanvas,
			Message[System`CurrentNotebookImage::canvasNBClipping,notebook,"bottom"];
			yBottom = yMaxCanvas
		];
		
		(*we should also make sure that the notebook is fully within the current monitor's screen*)
		(*if it's not, then we should issue a message that the screenshot may also be clipped off, and return the clipped coords don't clip it, as it could be fine on say Windows, but not on say Mac*)
		{{xLeftClipped,yBottomClipped},{xRightClipped,yTopClipped}} = {{xLeft,yBottom},{xRight,yTop}};
		{{xMinMonitor,yMinMonitor},{xMaxMonitor,yMaxMonitor}} = getScreenRegions[$screenInfoCache][[systemMargin["Monitor"]]];

		If[xLeft < xMinMonitor,
			Message[System`CurrentNotebookImage::monitorClipping,notebook,"left"];
			xLeftClipped = xMinMonitor;
		];
		If[yBottom < yMinMonitor,
			Message[System`CurrentNotebookImage::monitorClipping,notebook,"bottom"];
			yBottomClipped = yMinMonitor;
		];
		If[xRight > xMaxMonitor,
			Message[System`CurrentNotebookImage::monitorClipping,notebook,"right"];
			xRightClipped = xMaxMonitor;
		];
		If[yTop > yMaxMonitor,
			Message[System`CurrentNotebookImage::monitorClipping,notebook,"top"];
			yTopClipped = yMaxMonitor;
		];
		
		(*finally return the coordinates*)
		<|"AbsoluteCoords"->{{xLeft,yBottom},{xRight,yTop}},"ClippedCoords"->{{xLeftClipped,yBottomClipped},{xRightClipped,yTopClipped}},"MonitorNumber"->systemMargin["Monitor"]|>
	)
];
 
 
(*helper function to find all actual visible notebooks we can take a screenshot of*)
allActiveNotebooks[]:=Select[Function[{nb}, And @@ (#[AbsoluteCurrentValue[nb,Visible]] & /@ {Not@*TrueQ, MissingQ})]]@Notebooks[]


(*add's up all of the monitor's positions to make the full canvas across all monitors and returns that rectangle*)
calculateFullMonitorCanvas[]:=Block[
	{
		screenPositions = (Transpose[#["FullScreenArea"]] & /@ Association /@ $screenInfoCache),
		leftX,
		rightX,
		bottomY,
		topY
	},
	(*it would be nice if there was a simpler way to calculate this, but unfortunately it doesn't appear as if there is...*)
	(*this combines all of the disparate FullScreenArea results into a giant canvas across all monitors that RobotTools can use*)
	leftX = First@First@First@MinimalBy[First@*First]@screenPositions;
	topY = Last@First@First@MinimalBy[Last@*First]@screenPositions;
	bottomY = Last@Last@First@MaximalBy[Last@*Last]@screenPositions;
	rightX = First@Last@First@MaximalBy[First@*Last]@screenPositions;

	(* Convert coordinates so the origin is in the bottom left. *)
	rightX = rightX-leftX;
	leftX = 0;
	topY = bottomY - topY;
	bottomY = 0;
	{{leftX, bottomY}, {rightX, topY}}
]

$img = Null;

(*only save the cache if we have a front end available*)
If[$FrontEnd =!= Null,
	$screenInfoCache = SystemInformation["Devices","ScreenInformation"]
]

Options[System`CurrentNotebookImage] = {Background->Black,Method->Image,Head->Image,ImageResolution->Automatic};

Options[System`CurrentScreenImage] = {Background->Black,Method->Image,Head->Image,ImageResolution->Automatic};

System`CurrentScreenImage[args___,opts:OptionsPattern[]]:=Which[
	$FrontEnd === Null, (*no front end available*)
	(
		Message[FrontEndObject::notavail];
		$Failed
	),
	True, (*default case*)
	Refresh[$img = iScreenCapture[args,opts],TrackedSymbols:>{$img}, UpdateInterval->0]
]


System`CurrentNotebookImage[args___,opts:OptionsPattern[]]:=Which[
	$FrontEnd === Null, (*no front end available*)
	(
		Message[FrontEndObject::notavail];
		$Failed
	),
	True, (*default case*)
	Refresh[$img = iNotebookCapture[args,opts],TrackedSymbols:>{$img}, UpdateInterval->0]
]
 
(*no arg returns a screenshot across all monitors*)
iScreenCapture[opts:OptionsPattern[]] := Block[
	{
		fixedOpts = FilterRules[Flatten[{opts},{1}],Except[ImageResolution]]
	},
	(
		$screenInfoCache = SystemInformation["Devices","ScreenInformation"];
		If[fixedOpts == {},
			(*THEN*)
			(*can just straight create the image of the entire screen, note that this will fill in black pixels where there is nothing*)
			$img = ImageReflect[Image[getScreenshot[calculateFullMonitorCanvas[],OptionValue[ImageResolution]],"Byte"]],
			(*ELSE*)
			(*need to do more complex handling of the screenshot using Graphics options with Raster*)
			(
				With[
					{
						canvasRange = Transpose[calculateFullMonitorCanvas[]],
						screenCoordinates = getScreenRegions[$screenInfoCache]
					},
					(
						$img = Image@Graphics[
							(*first get all the rasters with appropriate location information*)
							Raster[
								getScreenshot[#,OptionValue[ImageResolution]],
								#,
								{0,255}] & /@ screenCoordinates,
							(*now apply all the user's options for the Graphics*)
							Sequence@@FilterRules[fixedOpts,Options[Graphics]],
							(*finally apply the hard coded plot region range and image size*)
							PlotRegion -> {{0, 1}, {0, 1}},
							PlotRange -> canvasRange,
							ImageSize -> EuclideanDistance @@@ canvasRange
						]
					)
				]
			)
		]
	)
];

(*this takes a rectangle of the type used by Raster for the pixel filling, and turns it into a rectangle describing the same area for Graphics to be placed appropriately*)
(*there's probably a simpler way to do this without so much weird transformations, but oh well this works*)
monitorRectangleToGraphicsRectangle[rect_,height_] := Reverse/@Partition[RotateRight[Join @@ MapAt[height - # &, rect, {All, 2}]], 2];

ptInsideRectangle[{xPt_,yPt_},{{xMinRect_,yTopRect_},{xMaxRect_,yBottomRect_}}]:=And[xPt >= xMinRect, yPt >= yTopRect, xPt <= xMaxRect, yPt <= yBottomRect] 

(*notebook arg captures a screenshot of the specific notebook*)
Options[iNotebookCapture] = Options[System`CurrentNotebookImage];
Options[iScreenCapture] = Options[System`CurrentNotebookImage];
iNotebookCapture[nb_NotebookObject,opts:OptionsPattern[]] /; MemberQ[Notebooks[], nb] := Block[
	{
		currentNb = InputNotebook[],
		rectAssoc = getNotebookPosition[nb],
		fixedOpts = FilterRules[Flatten[{opts},{1}],Except[ImageResolution]],
		method,
		head,
		background
	},
	$screenInfoCache = SystemInformation["Devices","ScreenInformation"];
	method = OptionValue[Method];
	head = OptionValue[Head];
	background = OptionValue[Background];
	(*switch to the requested notebook*)
	SetSelectedNotebook[nb];
	(*check the method to use*)
	Which[
		method === Image,
		(
			$img = If[head === Image,
				Identity,
				Graphics[Raster[ImageData[ImageReflect@#]],PlotRange->Transpose[{{0,0},ImageDimensions[#]}]]&
			]@ (
				If[rectAssoc["AbsoluteCoords"] =!= rectAssoc["ClippedCoords"] && (background =!= None || N@List@@ColorConvert[background,"RGB"] =!= N@List@@ColorConvert[Black, "RGB"]),
					(*THEN*)
					(*need to do some special handling for the background by capturing just the clipped area and computing the background to compose as the background color*)
					(
						background = List@@ColorConvert[background,"RGB"];
						ImageApplyIndexed[
							Function[{pixel,pos},
								(*pos is the index in the image, which we need to increase by the upper left coordinates of this window, so add that to the position so*)
								(*that the comparison is still done using global monitor canvas coordinates*)
								If[ptInsideRectangle[Reverse[pos]+rectAssoc["AbsoluteCoords"][[1]],rectAssoc["ClippedCoords"]],
									(*THEN*)
									(*it's within the area we want, return whatever it is*)
									pixel,
									(*ELSE*)
									(*it's in the background area, so reset it to the background option*)
									background
						 		]
							],
							ImageReflect@Image[getScreenshot[rectAssoc["AbsoluteCoords"],OptionValue[ImageResolution]],"Byte"]
						]
					),
					(*ELSE*)
					(*no special handling needed, can just straight take the image of the coords*)
					ImageReflect[Image[getScreenshot[rectAssoc["AbsoluteCoords"],OptionValue[ImageResolution]],"Byte"]]
				]
			)
		),
		method === Graphics,
		(
			$img = head@Graphics[
				(*first get all the rasters with appropriate location information*)
				Raster[
					getScreenshot[rectAssoc["ClippedCoords"],OptionValue[ImageResolution]],
					rectAssoc["ClippedCoords"],
					{0,255}
				],
				(*now apply all the user's options for the Graphics*)
				Sequence@@FilterRules[fixedOpts,Options[Graphics]],
				(*finally apply the hard coded plot region range and image size*)
				PlotRegion -> {{0, 1}, {0, 1}},
				PlotRange -> Transpose[rectAssoc["ClippedCoords"]],
				ImageSize -> EuclideanDistance@@@Transpose[rectAssoc["AbsoluteCoords"]]
			]
		)
	];
	(*now switch back to the user's input notebook*)
	SetSelectedNotebook[currentNb];
	(*return the image*)
	$img
];

iNotebookCapture[nb_NotebookObject] /; !MemberQ[Notebooks[],nb]:= (
	Message[System`CurrentNotebookImage::invalidNB,nb];
	$Failed
)

iNotebookCapture[opts:OptionsPattern[]]:=iNotebookCapture[EvaluationNotebook[],opts]

iNotebookCapture[any___]:=(
	Message[System`CurrentNotebookImage::invalid,{any}];
	$Failed
)

(*rectangle arg just confirms that the rectangle is within the full screen canvas and then returns that *)
iScreenCapture[{{xLeft_?IntegerQ,yBottom_?IntegerQ},{xRight_?IntegerQ,yTop_?IntegerQ}},OptionsPattern[]]:=Block[
	{
		xMin,xMax,yMin,yMax,temp
	},
	(
		$screenInfoCache = SystemInformation["Devices","ScreenInformation"];
		{{xMin,yMin},{xMax,yMax}} = calculateFullMonitorCanvas[];
		If[xLeft < xRight && yBottom < yTop,
			(*THEN*)
			(*the rectangle is at least a valid rectangle - check to make sure it's within the screen*)
			If[IntervalMemberQ[Interval[{xMin,xMax}],Interval[{xLeft,xRight}]]&&IntervalMemberQ[Interval[{yMin,yMax}],Interval[{yTop,yBottom}]],
				(*THEN*)
				(*it's within the canvas - take the picture*)
				$img = ImageReflect[Image[getScreenshot[{{xLeft,yBottom},{xRight,yTop}},OptionValue[ImageResolution]],"Byte"]],
				(*ELSE*)
				(*outside of the canvas - raise message and fail*)
				(
					If[xLeft < xMin,
						Message[System`CurrentScreenImage::canvasRectClipping,{{xLeft,yBottom},{xRight,yTop}},"left"]
					];
					If[xRight > xMax,
						Message[System`CurrentScreenImage::canvasRectClipping,{{xLeft,yBottom},{xRight,yTop}},"right"]
					];
					If[yBottom < yMin,
						Message[System`CurrentScreenImage::canvasRectClipping,{{xLeft,yBottom},{xRight,yTop}},"bottom"]
					];
					If[yTop > yMax,
						Message[System`CurrentScreenImage::canvasRectClipping,{{xLeft,yBottom},{xRight,yTop}},"top"]
					];
					
					$Failed
				)
			],
			(*ELSE*)
			(*invalid rectangle - raise message and fail*)
			(
				Message[System`CurrentScreenImage::rect,{{xLeft,yTop},{xRight,yBottom}}];
				$Failed
			)
		]
	)
]

iScreenCapture[monitorNumber_?IntegerQ,OptionsPattern[]] := Block[{region},
	(
		$screenInfoCache = SystemInformation["Devices","ScreenInformation"];
		If[monitorNumber >= 1 && monitorNumber <= Length[$screenInfoCache],
			(*THEN*)
			(*get the full screen area for that monitor and capture it*)
			$img = ImageReflect[
				Image[
					getScreenshot[
						getScreenRegions[$screenInfoCache][[monitorNumber]],
						OptionValue[ImageResolution]
					],
					"Byte"
				]
			],
			(*ELSE*)
			(*invalid number - issue message and fail*)
			(
				Message[System`CurrentScreenImage::monitorNum,monitorNumber,Range[Length[$screenInfoCache]]];
				$Failed
			)
		]
	)
]
iScreenCapture[any___]:=(
	Message[System`CurrentScreenImage::invalid,any];
	$Failed
)


(*helper function which calls either the full native screen version or the normal one based on the option*)
getScreenshot[loc_,resOpt_] := If[resOpt===Full,
	RobotTools`Package`nativeScreenShot[loc],
	RobotTools`Package`screenShot[loc]
]

(*reprotect all the System` symbols again*)
(
	SetAttributes[#,{ReadProtected}];
	Protect[#]
)&/@{
	"System`CurrentNotebookImage",
	"System`CurrentScreenImage"
}

getScreenRegions[infoCache_] := Block[{leftX, topY, bottomY, rightX, screenPositions, allMonitorPositions, conversion, convertedPositions},

		screenPositions = (Transpose[#["FullScreenArea"]] & /@ Association /@ infoCache);
		leftX = First@First@First@MinimalBy[First@*First]@screenPositions;
		topY = Last@First@First@MinimalBy[Last@*First]@screenPositions;
		bottomY = Last@Last@First@MaximalBy[Last@*Last]@screenPositions;
		rightX = First@Last@First@MaximalBy[First@*Last]@screenPositions;

		allMonitorPositions= (Lookup[#, "FullScreenArea"]) & /@ infoCache;
		conversion = (
			{{x1,x2},{y1,y2}} = #;
			x1 -= leftX;
			x2 -= leftX;
			swap = y1;
			y1 = bottomY - y2;
			y2 = bottomY - swap;
			{{x1, y1},{x2, y2}}
		)&;

		convertedPositions = conversion /@ allMonitorPositions
]

End[]

EndPackage[]

