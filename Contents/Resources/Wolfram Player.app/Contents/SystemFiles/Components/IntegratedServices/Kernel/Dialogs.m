(* ::Package:: *)

(* ::Title:: *)
(*CreateIntegratedServiceDialog*)


(* ::Section::Closed:: *)
(*Notes (larrya, 8/26/2016)*)


(* ::Text:: *)
(*IntegratedService`Private`CreateIntegratedServiceDialog is a function that generates an IntegratedServiceDialog. There two DownValues for the function:*)


(* ::ItemNumbered::Closed:: *)
(*A five input argument variant whose required arguments are:*)


(* ::SubitemNumbered::Closed:: *)
(*serviceID\[LongDash]The service's identity. The id can be the service's logo image, company name (i.e., as text), or an expr (e.g., a Grid) that combines the image and the name.*)


(* ::SubitemNumbered:: *)
(*descriptionTxt\[LongDash]Text that describes the purpose of the dialog. The text could be a string or an expr (e.g., a Row) if you wanted to include inline styles or hyperlinks.*)


(* ::SubitemNumbered:: *)
(*actionBtnLabel\[LongDash]The text that appears on the action button.*)


(* ::SubitemNumbered:: *)
(*actionBtnFunction\[LongDash]The function or operation of the action button. A function that connects or interacts with the IntegratedService. *)


(* ::SubitemNumbered:: *)
(*learnFunction\[LongDash]The function associated with the "Learn More" button. If it turns out that the button is to be used as a hyperlink, exclusively, this argument could be a simple url or uri.*)


(* ::ItemNumbered:: *)
(*A four input argument variant is used if a serviceID is not required or is unavailable. In that case the genericHeader, "Integrated Service", is displayed singularly. The four arguments are identical to args 2\[Dash]5 in the 5-arg variant, but without the first argument.*)


(* ::Subsection::Closed:: *)
(*Updated (larrya, 1/27/2017)*)


(* ::Text:: *)
(*Three new options to CreateIntegratedServicesDialog have been added: "ShowLearnButton", "GenericTitle", and "GenericTitleBelow". Note the option defaults are "ShowLearnButton"->True and "GenericTitle"->None (*this value can be either a string or None. If None, the generic title is hidden.*), "GenericTitleBelow"->False (*If False, the generic title is inline with the service title and a vertical separator is placed between them. If True, the generic title is stacked vertically below the service title*). Feel free to change the default values.*)


(* ::Text:: *)
(*The last 2 arguments of CreateIntegratedServicesDialog must now be Function expressions, that is Head[#] === Function&*)


(* ::Section::Closed:: *)
(*Header*)


BeginPackage["IntegratedServices`"];
Begin["`Private`"];


(* ::Section:: *)
(*Utilities*)


(* ::Subsection::Closed:: *)
(*Global symbols*)


(* ::Subsubsection::Closed:: *)
(*Fonts*)


$basefontfamily = "Helvetica";
$basefontweight = "Regular";


(* ::Subsubsection::Closed:: *)
(*Font Size*)


$titlefontsize = 14;
$dividerfontsize = 30;

$textfontsize = 12;


(* ::Subsubsection::Closed:: *)
(*Colors*)


$basefontcolor = RGBColor[0.3, 0.3, 0.3];

$linkcolordefault = RGBColor[0., 0.5450980392156862, 0.796078431372549];
$linkcolorhover = RGBColor[0., 0.4588235294117647, 0.6588235294117647];

$defaultbuttoncolordesktop = RGBColor[0.9568627450980391, 0.9725490196078431, 1.];
$defaultbuttonbackgrounddesktop = RGBColor[0.11372549019607843, 0.5490196078431373, 0.9882352941176471];

$servicetitlecolor = RGBColor[0.6509803921568628, 0.6509803921568628, 0.6509803921568628];
$servicetitlecolor2 = RGBColor[0.3, 0.3, 0.3];
$titledividercolor = RGBColor[0.8156862745098039, 0.8156862745098039, 0.8156862745098039];


(* ::Subsubsection::Closed:: *)
(*Labels*)


$learnmorelabel = "Learn More";

$windowtitle = "Wolfram Integrated Services";


(* ::Subsubsection::Closed:: *)
(*Heading Divider*)


$generictitle = $windowtitle;
$dividerimg = Graphics[{$titledividercolor, AbsoluteThickness[1.0],
  Line[{{0, 0}, {0, 1}}]}, PlotRangePadding -> None,
 ImagePadding -> None, AspectRatio -> 17, ImageSize -> {2, Automatic}];


(* ::Subsection::Closed:: *)
(*Text Style*)


baseStyle[text_, opts:OptionsPattern[Style]] := Style[text,
	FontFamily -> $basefontfamily,
	FontWeight -> $basefontweight,
	AutoSpacing -> False,
	LineIndent -> 0,
	LineSpacing -> {1, 2}(*{1, 0}*),
	Hyphenation -> False,
	opts
]


headingStyle[text_, opts:OptionsPattern[]] := baseStyle[text, FontSize -> $titlefontsize, opts]


serviceHeadingStyle[text_, opts___] := headingStyle[text,  FontSize -> $textfontsize,FontColor -> $servicetitlecolor2, opts]


descriptionStyle[text_] := baseStyle[text, FontSize -> $textfontsize, FontColor -> $basefontcolor]


btnLabelStyle[text_, fntcolr_:Automatic, opts:OptionsPattern[]] := baseStyle[text, LineBreakWithin -> False, FontSize -> $textfontsize, FontColor -> fntcolr, opts]


(* ::Subsection::Closed:: *)
(*Dialog Header*)


genericHeader[genericTitle_] := baseStyle[genericTitle, FontSize -> $titlefontsize, FontColor -> $basefontcolor];
genericHeader[] := genericHeader[$generictitle];


headerGrid[serviceID_, None, _, opts:OptionsPattern[]] := serviceHeadingStyle[serviceID];

headerGrid[serviceID_, genericTitle_, True, opts:OptionsPattern[Grid]] := Grid[
	{
		{
			serviceHeadingStyle[serviceID]
		},
		{
			genericHeader[genericTitle]
		}
	},
	Alignment -> {Left, Center(*Baseline*)},
	Spacings -> {0, 2.5},
	opts
]

headerGrid[serviceID_, genericTitle_, _, opts:OptionsPattern[Grid]] := Grid[
	{
		{
			serviceHeadingStyle[serviceID](*,
			$dividerimg*),
			genericHeader[genericTitle]
		}
	},
	Alignment -> {Left, Center},
	Spacings -> {1,0},
	opts
];



(* ::Subsection::Closed:: *)
(*Buttons*)


SetAttributes[actionChoiceBtns, HoldRest];
actionChoiceBtns[lbls:{actionLbl_, cancelLbl_}, funcs:{actionFunc_, cancelFunc_}, choicebuttonopts:{{OptionsPattern[]},{OptionsPattern[]}}, gopts:OptionsPattern[]] := ChoiceButtons[lbls, funcs, choicebuttonopts, gopts]

actionChoiceBtns[lbls:{actionLbl_, cancelLbl_}, funcs:{actionFunc_, cancelFunc_}, choicebuttonopts:{{actbtnopts:OptionsPattern[]},{cancelbtnopts:OptionsPattern[]}}, gopts:OptionsPattern[]] := Grid[{{
	CancelButton[cancelLbl, cancelFunc, cancelbtnopts],
	DefaultButton[actionLbl, actionFunc, actbtnopts]
}}, gopts] /; $CloudEvaluateQ

actionChoiceBtns[actionLbl_, actionFunc_, choicebuttonopts:{{defaultBtnOpts:OptionsPattern[]}, {cancelBtnOpts:OptionsPattern[]}}, gopts:OptionsPattern[]] := actionChoiceBtns[
	{
		Pane[If[$OperatingSystem === "MacOSX",
				btnLabelStyle[actionLbl, White],
				btnLabelStyle[actionLbl]],
			{Automatic, Full}, Alignment -> {Center, Center}
		],
		Pane[btnLabelStyle["Cancel"],
			{Automatic, Full},
			Alignment -> {Center, Center}
		]
	},
	{actionFunc, DialogReturn[$Canceled]},
	{
		{
			defaultBtnOpts,
			ImageSize -> Dynamic[CurrentValue["DefaultButtonSize"]],
			FrameMargins -> If[$OperatingSystem === "MacOSX", {{10, 10}, {Automatic, Automatic}}, Automatic]
		},
		{
			cancelBtnOpts,
			ImageSize -> Dynamic[CurrentValue["DefaultButtonSize"]],
			FrameMargins -> If[$OperatingSystem === "MacOSX", {{10, 10}, {Automatic, Automatic}}, Automatic]
		}
	},
	gopts,
	BaselinePosition -> Baseline
];

actionChoiceBtns[actionLbl_, actionFunc_, gopts:OptionsPattern[]] := actionChoiceBtns[actionLbl, actionFunc,
	{{}, {}},
	gopts
]


SetAttributes[learnBtn, HoldRest];
learnBtn[label_, function_, opts:OptionsPattern[]] := Button[
	Pane[btnLabelStyle[label],
			{Automatic, Full},
			Alignment -> {Center, Center}
	], function,
	opts,
	ImageSize -> Dynamic[CurrentValue["DefaultButtonSize"]],
	FrameMargins -> If[$OperatingSystem === "MacOSX", {{10, 10}, {Automatic, Automatic}}, Automatic]
]

learnBtn[label_, function_, opts:OptionsPattern[]] := Button[btnLabelStyle[label], function] /; $CloudEvaluateQ


(* ::Subsection:: *)
(*Layout*)


SetAttributes[dialogLayout, HoldRest];

dialogLayout[serviceID_, description_, actionBtnLabel_, actionBtnFunction_, learnFunction_, showLearnButton_, genericTitle_, placeGenericTitleBelow_, gridOpts:OptionsPattern[Grid]] := Grid[
	{
		{
			(*headerGrid[serviceID, genericTitle, placeGenericTitleBelow]*)
			serviceID
		},
		{
(*MOD--larrya*)			Pane[descriptionStyle[description], ImageSize -> {Full, All}, FrameMargins -> If[$OperatingSystem === "MacOSX", {{5, 0}, {12, 24}}, {{4, 0}, {12, 24}}](*{{5, 0}, {12, 24}}*)](*Pane is used to add fixed vertical space above and below via FrameMargins.*)
		},
		{
			Grid[
				{
					{
						Framed[If[showLearnButton, learnBtn[$learnmorelabel, learnFunction], ""], FrameStyle -> None, FrameMargins -> None],
						Framed[Style[actionChoiceBtns[actionBtnLabel, actionBtnFunction,{{FrameMargins->{{10, 10}, {Automatic, Automatic}}}, {FrameMargins->{{10, 10}, {Automatic, Automatic}}}}], LineBreakWithin -> False], FrameStyle -> None, FrameMargins -> None, ImageSize -> Full, Alignment -> Right](*Pane is used generate horizontal space that auto-adjusts to WindowWidth and button widths.*)
					}
				},
				ItemSize -> {Automatic, Automatic}
			]
		}
	},
	ItemSize -> {Automatic, Automatic},
	Alignment -> Left,
	Spacings -> {Automatic, 0},
	gridOpts
]


(* ::Section:: *)
(*CreateIntegratedServicesDialog*)


SetAttributes[CreateIntegratedServicesDialog, HoldAllComplete];

Options[CreateIntegratedServicesDialog] = {"ShowLearnButton" -> True, "GenericTitle" -> None, "GenericTitleBelow" -> False};

CreateIntegratedServicesDialog[serviceID_, descriptionTxt_, actionBtnLabel_, actionBtnFunction_Function, learnFunction_Function, opts:OptionsPattern[]] := With[
	{
		linkcolordefault = $linkcolordefault,
		linkcolorhover = $linkcolorhover,
		showLearnBtn = "ShowLearnButton" /. {opts} /. Options[CreateIntegratedServicesDialog],
		genericTitle = "GenericTitle" /. {opts} /. Options[CreateIntegratedServicesDialog],
		genericTitleBelow = "GenericTitleBelow" /. {opts} /. Options[CreateIntegratedServicesDialog]
	},
  DialogInput[
	{
		Pane[
			dialogLayout[serviceID, descriptionTxt, actionBtnLabel, actionBtnFunction, learnFunction, showLearnBtn, genericTitle, genericTitleBelow],
			ImageSize -> {Full, All},
			FrameMargins -> {{15, 15}, {10, 10}}
		]
	},
	StyleDefinitions -> Notebook[{
		Cell[StyleData[StyleDefinitions -> "Dialog.nb"]],
		Cell[StyleData["Hyperlink"],
			FontColor -> linkcolordefault],
		Cell[StyleData["HyperlinkActive"],
			FontColor -> linkcolorhover]
	}],
	Background -> RGBColor[1, 1, 1],
	CellContext -> Cell,
	WindowMargins -> Automatic,
	WindowSize -> {500, All},
	Deployed -> True,
	WindowTitle -> $windowtitle
]
];

(* ::Section::Closed:: *)
(*Footer*)

End[];
EndPackage[];
