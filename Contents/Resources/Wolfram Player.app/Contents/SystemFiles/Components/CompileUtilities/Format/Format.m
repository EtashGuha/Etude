BeginPackage["CompileUtilities`Format`"]

CreateBoxIcon
$FormatingGraphicsOptions
LazyFormat

(*
This exists mainly to strip the type from a TypeLiteral, e.g., 1:Integer64 is printed as 1
*)
PrettyFormat


(*
TemporaryInformation and InformationPanel were copied from GeneralUtilities
These are the only functions used from GeneralUtilities and we don't want to pay for all
of the code that gets loaded by GeneralUtilities
*)
CompileTemporaryInformation
CompileInformationPanel


Begin["`Private`"]

Needs["CompileUtilities`Reference`"]

CreateBoxIcon[text_] :=
	With[{sz = CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]},
		Graphics[
			Text[Style[
				text,
				GrayLevel[0.7],
				Bold, 
	   			sz
			]],
			Background -> GrayLevel[0.93],
			Axes -> False, 
		 	AspectRatio -> 1,
		 	ImageSize -> {
		 		Automatic,
		 		Dynamic[sz*(CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]), ImageSizeCache -> {27., {0., 10.}}]
			},
			Frame -> True, 
		 	FrameTicks -> None, 
		 	FrameStyle -> Directive[Thickness[Tiny], GrayLevel[0.55]]
		]
	]

$FormatingGraphicsOptions = Sequence[
	Background -> GrayLevel[0.93],
	Axes -> False, 
 	AspectRatio -> 1,
 	ImageSize -> {
 		Automatic,
 		Dynamic[3.5*(CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]), ImageSizeCache -> {27., {0., 10.}}]
	},
	Frame -> True, 
 	FrameTicks -> None, 
 	FrameStyle -> Directive[Thickness[Tiny], GrayLevel[0.55]]
]




(*
This exists mainly to strip the type from a TypeLiteral, e.g., 1:Integer64 is printed as 1
*)
PrettyFormat[Type[t_]] := PrettyFormat[t]
PrettyFormat[TypeSpecifier[t_]] := PrettyFormat[t]
PrettyFormat[Rule[params_, ret_]] := ToString[PrettyFormat[params]]<>" \[Rule] "<>ToString[PrettyFormat[ret]]
PrettyFormat[l_List] := ToString[PrettyFormat /@ l]

PrettyFormat[ "Complex"["Real64"]] := "Complex128"
PrettyFormat[ "Complex"["Real32"]] := "Complex64"

PrettyFormat[app_String[args__]] := ToString[("TypeSpecifier[" <> PrettyFormat[app] <>"]") @@ (PrettyFormat /@ {args})]
PrettyFormat[s_String] := s
PrettyFormat[i_Integer] := ToString[i]


(*
  This formats into a Dynamic, but keeps the data in a symbol in the Kernel
  because some of the data being formatted is not serializable in the FrontEnd.
  The $localLazyVars is the only holder of the data.  There is a simple memory 
  cleaner that shrinks the list when it reaches a certain size.
  
  If the Kernel is restarted or the data has been purged, then Undefined is printed.
*)

$localLazyVars = CreateReference[{}]
(* max size *)
$lengthLimit = 100;
(* size to shrink to *)
$shrinkSize = 50;

LazyFormat[predicate_, arg_] :=
	With[ {tmp = Unique["printLocal", Temporary]},
		If[ $localLazyVars["length"] > $lengthLimit,
			Do[ $localLazyVars["popFront"], {$localLazyVars["length"]-$shrinkSize}]];
		$localLazyVars["appendTo", Hold[tmp]];
		tmp = arg;
		Dynamic[If[TrueQ[predicate[tmp]], tmp, "Undefined"]]
	]








CompileTemporaryInformation[info_] := (
	DisplayTemporary[None];
	DisplayTemporary @ Framed[
		Style[info,
			FontFamily -> "Verdana", FontSize -> 11,
		FontColor -> RGBColor[0.2, 0.4, 0.6]],
		FrameMargins -> {{16, 16}, {8, 8}},
		FrameStyle -> RGBColor[0.2, 0.4, 0.6],
		Background -> RGBColor[0.96, 0.98, 1.]
	];
);


Options[CompileInformationPanel] = {
	UpdateInterval -> None,
	TrackedSymbols :> {},
	ColumnWidths -> {Automatic, Automatic},
	Background -> RGBColor[0.9802, 0.9802, 0.9802],
	ColumnsEqual -> False
};

CompileInformationPanel[titlestring_, data_, opts:OptionsPattern[]] := Module[
	{grid, title, tracked, gridoptions, background, interval},
	background = OptionValue[Background];
	title = Item[
		Framed[
			Style[titlestring, Bold, 12, "SuggestionsBarText"],
			FrameMargins -> {{10, 5}, {-5, 5}},
			FrameStyle -> None
		],
		ItemSize -> {Automatic,1},
		Alignment -> {Left, Bottom},
		FrameStyle -> Opacity[0.1],
		Background -> Darker[background, 0.05],
		Frame -> {{False, False}, {True, False}}
	];
	If[titlestring === None, title = Nothing];
	tracked = First @ FilterRules[Join[{opts}, Options[CompileInformationPanel]], {TrackedSymbols}];
	(* ^ we don't want to let the rhs of :> evaluate, so we need the full rule *)
	gridoptions = {
		Dividers -> {
			{False, {Opacity[0.15]}, False}, 
			{}
		},
		ColumnWidths -> OptionValue[ColumnWidths],
		ColumnsEqual -> OptionValue[ColumnsEqual],
		ColumnAlignments -> {Right, Left},
		ColumnSpacings -> {1.6, 2.5},
		RowSpacings -> 2
	};
	interval = OptionValue[UpdateInterval];
	grid = If[interval =!= None, 
		DynamicModule[{data2, gridoptions2, interval2},
			(* ^ not sure about this solution, but it seems that the values of gridoptions etc. are not cleared when using a DynamicModule.
			*)
			data2 = data;
			gridoptions2 = gridoptions;
			interval2 = interval;
			Dynamic[
				makeGrid[data2, gridoptions2], 
				tracked,
				UpdateInterval -> interval2
			]
		]
		,
		makeGrid[data, gridoptions]
	];
	grid = Item[
		Deploy@Framed[grid, FrameMargins -> {{10, 10}, {10, 5}}, FrameStyle -> None],
		BaseStyle -> {
			FontWeight -> "Light", 
			FontSize -> 12,
			FontFamily -> CurrentValue["PanelFontFamily"], 
			NumberMarks -> False, 
			Deployed -> False
		},
		Alignment -> Center
	];
	Deploy@Style[Framed[
		Column[{title, grid},
			ColumnWidths -> Automatic,
			ColumnAlignments -> Left,
			RowLines -> False,
			RowSpacings -> {3, 1},
			StripOnInput -> True
		],
		Background -> background,
		FrameMargins -> {{0, 0}, {0, 0}},
		FrameStyle -> LightGray,
		RoundingRadius -> 5
	], LineBreakWithin -> False]
];

makeGrid[data_, opts:OptionsPattern[]] := Grid[
	If[!MatchQ[#2, _Missing],
		If[MatchQ[#1, None | Left | Right | Center],
			{Item[#2, Alignment -> #1], SpanFromLeft}
			,
			{keystyle[#1], #2}
		]
		,
		Nothing
	]& @@@ data,
	opts
];
keystyle[x_] := Row[{Spacer[5], Style[x, GrayLevel[0.4]]}];






End[]

EndPackage[]
