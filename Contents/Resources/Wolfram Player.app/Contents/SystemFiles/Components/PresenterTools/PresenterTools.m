(* ::Package:: *)

(* :Title: PresenterTools *)

(* :Context: PresenterTools` *)

(* :Author:
		Andrew Hunt
		andyh@wolfram.com
		Jay Warendorff
		jayw@wolfram.com
*)

(* :Package Version: @presentertoolsversion@ *)

(* :Mathematica Version: 11.3 *)

(* :Discussion: Slideshow / Presentation tools *)

(* :Keywords:  *)


(*******************************************************************************

PresenterTools

FUNCTIONS

CreatePresentation[...]
-- GUI: Launch dialog for creating PresenterTools notebook
-- Function:
	CreatePresentation[
		"ThemeName" -> String ("Default", "Garnet", "Facet", ...),
		"ColorSet" -> {"Display1" -> RGBColor[...], ...},
		"FontSet" -> {"Display1" -> String, ...}
		...
	]


*******************************************************************************)



(*********************************** PUBLIC ***********************************)

BeginPackage["PresenterTools`", {"PresenterTools`Styles`"}]
(* Exported symbols added here with SymbolName::usage *)


CreatePresentation::usage = "CreatePresentation[], CreatePresentation[content, name, options] creates a presentation notebook with content that uses theme name with various options.";

(* theme functions *)
LoadThemeDefinitions::usage = "LoadThemeDefinitions[file] loads defined themes from file.";
GetThemeInformation::usage = "GetThemeInformation[theme] returns an association of information about theme.";
SetThemeInformation::usage = "SetThemeInformation[theme, assoc] sets information from assoc to theme.";
GetPresentationInformation::usage = "";

GetThemeNames::usage = "GetThemeNames[] provides the list of available themes.";
(* list of currently defined themes *)
$ThemeNames = {};

GetLiteralizedNotebook::usage = "";
GetCellInformation::usage = "";

(* dialogs *)
CreatePresentationDialog::usage = "CreatePresentationDialog[] initiates a dialog to choose a presentation notebook.";
ThemeOptionsDialog::usage = "ThemeOptionsDialog[notebook] initiates a dialog to change theme of selected notebook.";
UpdateThemeNotebook::usage = "";
UpdateStyleDefinitionsDialog::usage = "UpdateStyleDefinitionsDialog[notebook] initiates a dialog to choose which existing options to apply to notebook's stylesheet.";
SetPresenterControlsDialog::usage = "SetPresenterControlsDialog[notebook] initiates a dialog to choose which slideshow keyboard shortcuts to apply to the notebook.";
AutomaticSlideBreakDialog::usage = "AutomaticSlideBreakDialog[notebook] initiates a dialog to set which cell styles have automatic slide (page) breaks.";
(*GetStylesheetFromNotebookDialog::usage = "";*)
GetPresenterStylesheet::usage = "";

SlideNavigationPopupMenu::usage = "";


(* misc *)
ToggleNavigationBar::usage = "ToggleNavigationBar[notebook] toggles whether the slide navigation bar within 'Slideshow Presentation' ScreenStyleEnvironment should appear.";


(* *)
SetSelectionOptions::usage = "SetSelectionOptions[notebook, option] sets Cell option and tracks usage.";

OutlinePaletteNavigationBar::usage = "";
SlideshowNavigationBar::usage = "";



(**********
 TODO: Move most public functions below to private
 *********)

SetEnvironmentTaggingRules::usage = "SetEnvironmentTaggingRules[ScreenStyleEnvironment] set PresenterTools TaggingRules, if not present.";

CreateColorThemeDialog::usage = "";

ThumbnailMagnification::usage = "ThumbnailMagnification is the magnification value of the thumbnails within the thumbnail palette.";

SwitchToSlideShowWorking::usage = "";
SwitchToSlideShowPresentation::usage = "";
(* SwitchToPresentation::usage = ""; *)

GetPreviousWindowSize::usage = "";
RemoveSlideBreak::usage = "";

(* Stylesheets and Themes *)
CreateCellStyleDefinitionString::usage = "";

CreateStyleHintList::usage = "";

CreateNotebookThumbnail::usage = "CreateNotebookThumbnail[nb] returns a thumbnail Graphics of the nb expression.";
CreateThumbnailList::usage = "";

(* UserStyleOptionUpdateList = {}; *)
UserModificationsQ::usage = "";
SelectionCellStyleModifiedQ::usage = "";
DefaultOptionValueQ::usage = "";

(* BEGIN TOOLBAR *)

InsertSlideBreak::usage = "";

GetFontFamilyOfSelection::usage = "GetFontFamilyOfSelection[notebook] returns FontFamily of selection.";
GetCellStyleOfSelection::usage = "GetCellStyleOfSelection[notebook] returns Cell style of selection.";
GetStyleOfSelectionMenu::usage = "";

InsertOrModifyStyledCell::usage = "InsertOrModifyStyledCell[notebook, style] inserts a new cell of the specified style into notebook.";
InsertStyledCell::usage = "";
SetCellStyleMenu::usage ="";
SlideBreakRefreshQ::usage = "";
AddSlideBreakCells::usage = "";
RemoveSlideBreakCells::usage = "";

SetFontFamily::usage = "";
SetFontFamilyMenu::usage = "";
SetFontFamilyMenuList::usage = "";

SetFontSizeIncrease::usage = "";
SetFontSizeDecrease::usage = "";

SetFontWeight::usage = "";
SetFontSlant::usage = "";
SetFontUnderline::usage = "";
SetFontColor::usage = "";

CellAlignmentMenu::usage = "";
SetTextAlignment::usage = "";
(*
NudgeUp::usage = "";
NudgeDown::usage = "";
NudgeLeft::usage = "";
NudgeRight::usage = "";
*)
ClearCellFormatting::usage = "";

SetLockScaling::usage = "";

LockGraphicCellSizeQ::usage = "";
GetImageDimensions::usage = "";


CreatePresenterNotesPalette::usage = "CreatePresenterNotesPalette[nb] creates the Side Notes palette for nb.";

CreateOutlinePalette::usage = "CreateOutlinePalette[nb] creates the outline palette for nb.";
OpenPresenterPalette::usage = "OpenPresenterPalette[nb] opens the outline palette for nb.";

GetPresenterNotesCells::usage = "";
CreateSlideThumbnailCell::usage = "";
GetSystemInformation::usage = "";
NumberOfDisplays::usage = "";
MultipleDisplaysQ::usage = "MultipleDisplaysQ[] returns True is there are more than one display monitors.";

NotebookWindowSizeMenu::usage = "";

WindowElementsMenu::usage = "";

PresentNotebook::usage = "";
AdditionalViewsMenu::usage = "";

(*UpdateChangesToStyleSheet::usage = ""; *)

EndPresentation::usage = "EndPresentation[notebook] ends the presentation and returns user to 'Slideshow Working' environment";
NotebookToggleFullScreenWrapper::usage = "";

(* END TOOLBAR *)

GetThemeStyleHints::usage = "";
GetThemeFontStyleHints::usage = "";

GetPresenterToolsStyleHints::usage = "";

$UserStylesheetDirectory::usage = "Path to $UserBaseDirectory/SystemFiles/FrontEnd/Stylesheets";

AllContentsSelectedQ::usage = "";
CellBracketSelectedQ::usage = "";
CellContentsSelectedQ::usage = "";
CursorInCellNoSelectionQ::usage = "";

SlideBreakExpressionPanel::usage = "";

(*LoadPresenterToolsMenus::usage = "";*)

GetThemeNameFromFilename::usage = "";
GetThemeFilenameFromName::usage = "";

GetThemeColorMenu::usage = "GetThemeColorMenu[name] creates color menu from GetThemeInformation[name, \"ColorSet\"]";
GetThemeFontMenu::usage = "GetThemeFontMenu[name] creates color menu from GetThemeInformation[name, \"FontSet\"]";

GetColorStyleThemeList::usage = "";
GetFontStyleThemeList::usage = "";

ChangeNotebookStyleSheet::usage = "";
GetUserModifications::usage = "";
UserModificationCellStyles::usage = "";

SetStyleUpdateList::usage = "";
UpdateNotebookStyleHints::usage = "";

GetSlides::usage = "GetSlides[nb] returns a list of slides found within nb";

(* MESSAGES *)
SetHorizontalMargin::val = "`1`";
SetHorizontalMargin::argx = "`1`";

(* DIALOGS *)
SelectACellMessageDialog::usage = "";

SlideShowAuthorToolbar::sel = "Can not determine selection.";

PresenterTools`StyleForInsertNewCell;
CollatePresenterNotes;

$PresenterToolsImagesDirectory::usage = "Path to Images directory.";
$PresenterToolsBitmapsDirectory::usage = "Path to bitmaps directory.";

$OutlineThumbnailMagnification = 0.8;

getCellStyle::usage = "";
TestList::usage = "";

(*
$PresenterNotesPalette = $Failed;
$PresenterOutlinePalette = $Failed;
*)

System`BackgroundAppearance;
System`WholeCellGroupOpener;
System`ReturnCreatesNewCell;
System`GroupOpenerInsideFrame;
System`GeneratedCellStyles;
System`CellInsertionPointCell;
System`BlinkingCellInsertionPoint;
System`CellInsertionPointColor;
System`TrackCellChangeTimes;
System`LineColor;
System`FrontFaceColor;
System`GraphicsColor;
System`BackFaceColor;



(********************************** PRIVATE ***********************************)

Begin["`Private`"]
(* Implementation of the package *)

(* Print full messages *)
$MessagePrePrint = .;


(* Update Paclet on the main link not the Preemptive link, in order to avoid timeout during update.
   Fix for bug: https://bugs.wolfram.com/show?number=351362
   *)
updatesAvailable = PacletManager`PacletCheckUpdate["PresenterTools"];
If[MatchQ[updatesAvailable, {__PacletManager`Paclet}],
    RunScheduledTask[
    	Quiet[
    		PacletUpdate["PresenterTools", "UpdateSites" -> False, "Asynchronous" -> True]
    	], {20}, "AutoRemove" -> True
    ]
];


(******************************** INFORMATION *********************************)

PresenterTools`Information`$Version = "PresenterTools Version 1.6.0.20190326";
PresenterTools`Information`$VersionNumber = ToExpression["1.6"];
PresenterTools`Information`$ReleaseNumber = ToExpression["0"];
PresenterTools`Information`$MinorReleaseNumber = ToExpression["20190326"];


(***************************** GLOBAL VARIABLES *******************************)

(* By default, Debug is False *)
$DebugQ = False;

(* Set base directory *)
$PresenterToolsDirectory = DirectoryName[ System`Private`$InputFileName ];
$PresenterToolsDataDirectory = FileNameJoin[{$PresenterToolsDirectory, "Data"}];
$PresenterToolsImagesDirectory = FileNameJoin[{$PresenterToolsDirectory, "Data", "Images"}];
$PresenterToolsSystemResourcesDirectory = FileNameJoin[{$PresenterToolsDirectory, "FrontEnd", "SystemResources"}];
$PresenterToolsBitmapsDirectory = FileNameJoin[{$PresenterToolsSystemResourcesDirectory, "Bitmaps", "PresenterTools"}];

$PresenterToolsDefaultThemeDefinitions = FileNameJoin[{$PresenterToolsDirectory, "FrontEnd", "SystemResources", "DefaultTheme.m"}];
$PresenterToolsThemeDefinitions = FileNameJoin[{$PresenterToolsSystemResourcesDirectory, "Themes.m"}];




(* Style Environment names *)
(* "PresenterTools Presentation (Preparation)" *)
$PresenterToolsPreparationName = "Slideshow Working";
(* "PresenterTools Presentation" *)
$PresenterToolsPresentationName = "Slideshow Presentation";
(* "Continuous Scrolling Presentation" *)
$PresenterToolsContinuousScrollName = "Scrolling Presentation";

(* Cell options currenly supported via toolbar *)
$ToolbarSupportedOptions = {
	FontSize,
	FontWeight,
	FontSlant,
	FrontEnd`FontVariationsUnderline,
	FontColor,
	CellMargins,
	TextAlignment
	};

PresenterTools`StyleForInsertNewCell = "Title";

$StylesheetThemeCellStyles = {"Title", "Subtitle", "Section", "Subsection", "Subsubsection", "Text", "SmallText", "Item", "ItemNumbered", "Input", "Code", "Output", "Message", "Program", "Graphics"};
$StylesheetThemeCellOptions = {FontFamily, FontSize, FontColor, FontWeight, FontSlant, (* CellDingbat, *) CellFrame, CellFrameColor, CellFrameMargins, CellMargins, TextAlignment, Background};

(* Set $SlideBreakStyles to {} for 11.3.0 *)
$SlideBreakStyles = {}; (* {"Title", "Chapter", "Section"}; *)


(******************************** FUNCTIONS ***********************************)



(* dialog interface for CreatePresentation *)
CreatePresentationDialog[]:=
	FrontEndExecute[{FrontEnd`NotebookOpen[FrontEnd`FindFileOnPath["PresenterNotebookChooser.nb", "PalettePath"]]}];

(**************************************

 CreatePresentation
	Main user level function to create a new slideshow notebook.

	in: Nothing, String theme name and (optional) List of attributes

		CreatePresentation[
			"Theme" -> String ("Default" | "Garnet" | ...),
			"FontSet" -> String ("Default" | "Garnet" | ...),
			"ColorSet" ->  String ("Default" | "Garnet" | ...),
			...
			]
	out: Notebook

***************************************)

Options[CreatePresentation] = {
	"Theme" -> "Default",
	"FontSet" -> Automatic,
	"ColorSet" -> Automatic,
	"BackgroundImages" -> None,
	"SampleContent" -> False,
	"StyleDefinitions" -> Automatic, (*FrontEnd`FileName[{"PresenterTools"}, "Default.nb", CharacterEncoding -> "UTF-8"], *)
	"SlideBreakStyles" -> {}
};

(* no theme or content -> "Default" theme with no content *)
(* backwards compatibility with v1/v1.5 *)
CreatePresentation[theme_String, opts:OptionsPattern[]]:=
Module[{},
	If[MemberQ[PresenterTools`GetThemeNames[], theme],
		iCreatePresentation[{}, "Theme" -> theme, opts],
	(* ELSE *)
		Message[CreatePresentation::theme, theme]
	]
];

CreatePresentation[opts:OptionsPattern[]]:= iCreatePresentation[{}, opts];
CreatePresentation[content_List, theme_String, opts:OptionsPattern[]]:= CreatePresentation[content, "Theme" -> theme, opts];

(* MAIN *)
CreatePresentation[content_List, opts:OptionsPattern[]]:=
Module[{theme},
	theme = OptionValue["Theme"];
	If[MemberQ[PresenterTools`GetThemeNames[], theme],
		iCreatePresentation[content, (*theme,*) opts],
	(* ELSE *)
		Message[CreatePresentation::theme, theme]
	]
];
CreatePresentation[a__]:= Message[CreatePresentation::argx, {a}];
CreatePresentation::argx = "`1` is not recognized by CreatePresentation.";
CreatePresentation::theme = "Nothing known about theme `1`";
CreatePresentation::info = "`1`";

(* iCreatePresentation
   Main function that creates the presenter notebook expression
   *)
iCreatePresentation[content_List, opts:OptionsPattern[{CreatePresentation, System`Notebook}]]:=
Module[{nbPrototype, fontset, colorset, themeDisplayName, themeStylesheet, theme,
	baseStyleHints, userStyleHints, titleBackgroundQ, titlePrototype, backgroundName, optionalChangesQ, styleDefinitions,
	fonts, colors, sampleContentQ, notebookContent, nbOptions, slideBreakStyles},

	(* "Theme" needs to be a String ("Default" | "Garnet" | "Aqua" ...)
	   "FontSet" and "ColorSet" are either :
	    1) Automatic (which is the default fonts/colors of "Theme"
		2) Strings ("Default" | "Garnet" | "Aqua" ...) or
	    3) Lists of rules ("Display1" -> "Helvetica" | RGBColor ...)
	   *)
	(* Default theme *)
	theme = OptionValue["Theme"];
	If[Not@MemberQ[GetThemeNames[], theme], theme = "Default"];
	themeDisplayName = theme;

	fonts = OptionValue["FontSet"];
	colors = OptionValue["ColorSet"];

	(* did user make any optional changes to theme (font, color, etc.) or straight theme?
	   If optional changes then need an embeded stylesheet *)
	optionalChangesQ = Or[(theme =!= fonts) && (fonts =!= Automatic), (theme =!= colors) && (colors =!= Automatic)];

	(* adjust 'fonts' and 'colors' if Automatic *)
	(* if fonts/colors are strings, set 'fontset' and 'colorset' *)
	Which[
		fonts === Automatic,
			fonts = theme;
			fontset = GetThemeInformation[theme, "FontSet"],
		StringQ[fonts],
			fontset = GetThemeInformation[fonts, "FontSet"],
		ListQ[fonts],
			fontset = fonts;
			fonts = theme,
		True,
			fonts = theme;
			fontset = GetThemeInformation[theme, "FontSet"]
	];
	Which[
		colors === Automatic,
			colors = theme;
			colorset = GetThemeInformation[theme, "ColorSet"],
		StringQ[colors],
			colorset = GetThemeInformation[colors, "ColorSet"],
		ListQ[colors],
			colorset = colors;
			colors = theme,
		True,
			colors = theme;
			colorset = GetThemeInformation[theme, "ColorSet"]
	];

	slideBreakStyles = OptionValue["SlideBreakStyles"];
	slideBreakStyles = If[MatchQ[slideBreakStyles, {}] || MatchQ[slideBreakStyles, {__?StringQ}], slideBreakStyles, {}];

	sampleContentQ = TrueQ[OptionValue["SampleContent"]];

	nbOptions = DeleteCases[{opts}, (a:Alternatives["StyleDefinitions", "FontSet", "ColorSet", "BackgroundImages", "SampleContent", "Name"] -> _), Infinity];

	(************************
	   StyleHints
	   Merge new user FontSet|ColorSet values with those from base Theme's FontSet|ColorSet
	   *)
	fontset = MergeSets[GetThemeInformation[theme, "FontSet"], fontset];
	If[MatchQ[fontset, $Failed], fontset = GetThemeInformation[theme, "FontSet"]];
	colorset = MergeSets[GetThemeInformation[theme, "ColorSet"], colorset];
	If[MatchQ[colorset, $Failed], colorset = GetThemeInformation[theme, "ColorSet"]];

	(* TODO: "BackgroundImages" routine needs error checking
	   AND the Key probably should be designed differently
	   *)
	titleBackgroundQ = TitleBackgroundQ[themeDisplayName];

	(* Check if theme fonts and colors are from the same theme as stylesheet
	   If so, then we don't need to override StyleHints within stylesheet
	   *)
	If[optionalChangesQ,
		(* add user provided FontSet and ColorSet StyleHints *)
		(*baseStyleHints = If[Head[baseStyleHints] === Association, Normal[baseStyleHints], {}]; *)
		baseStyleHints = {"SlideBreakStyles" -> slideBreakStyles (*$SlideBreakStyles*) };

		userStyleHints = ResetOptions[baseStyleHints, {"ColorSet" -> colorset, "FontSet" -> fontset}];
		userStyleHints = Join[{
							System`ParentList,
							"CodeFont" -> CurrentValue[{System`StyleHints, "CodeFont"}]
							}, userStyleHints];

		nbPrototype = Cell[StyleData["Notebook"], System`StyleHints -> userStyleHints],
	(* ELSE *)
		nbPrototype = {}
	];

	(*  Title Background 9-Patch images
		Title prototype is for background 9patch images.
		If theme color (background) =!= theme then we need to call a
		different background then the one defined within the stylesheet
	*)
	titlePrototype =
		If[(titleBackgroundQ === True) && (theme =!= colors),
			(
				backgroundName =
					StringJoin["Background-", theme, "-", colors, ".png"];
				With[{background = backgroundName},
					Cell[StyleData["FirstSlide"],
						PrivateCellOptions->{
							"PagewiseNotebookBaseStyle" -> {
								BackgroundAppearance -> FrontEnd`FileName[{"PresenterTools"}, background]
							}
						}
					]
				]
			),
		(* ELSE *)
			{}
		];

	themeStylesheet =
		If[OptionValue["StyleDefinitions"] === Automatic,
			PresenterTools`GetThemeInformation[themeDisplayName, "StyleDefinitions"],
		(* ELSE *)
			OptionValue["StyleDefinitions"]
		];

	styleDefinitions =
		If[(nbPrototype === titlePrototype === {}),
			(* no need to have a Notebook as the RHS of StyleDefinitions *)
			themeStylesheet,
		(* ELSE *)
			(* embedded Notebook needed for StyleDefinitions
			   since changes were made to nbPrototype and/or titlePrototype
			*)
			With[{styleSheet = Cell[StyleData[StyleDefinitions -> themeStylesheet ]],
				notebookStyleData = nbPrototype, titleStyleData = titlePrototype},
				Notebook[
					Flatten[{
						styleSheet,
						notebookStyleData,
						titleStyleData
					}], Visible -> False, StyleDefinitions -> "PrivateStylesheetFormatting.nb"
				]
			]
		];


	(************************
	   Notebook Content
	   check notebook content and "SampleContent" option
	*)
	If[(content === {}) && sampleContentQ,
		slideBreakStyles = {"Title", "Chapter", "Section"};
		(* use sample content *)
		If[MemberQ[{"ChineseSimplified", "ChineseTraditional", "Japanese", "Spanish"}, $Language],
			notebookContent = Get[FileNameJoin[{$PresenterToolsSystemResourcesDirectory, $Language, "SlideShowChooserTemplate.nb"}]],
		(* ELSE *)
			notebookContent = Get[FileNameJoin[{$PresenterToolsSystemResourcesDirectory, "SlideShowChooserTemplate.nb"}]]
		];
		If[Head[notebookContent] === Notebook,
			notebookContent = First[notebookContent];
		,
			notebookContent = Cell[CellGroupData[{ Cell["Presentation title", "Title"], Cell["Section heading", "Section"] }]];
		],
	(* ELSE *)
		(* user provided content or "SampleContent" === False *)
		If[content =!= {},
			notebookContent = Join[{ Cell["", "SlideShowNavigationBar", "FirstSlide", CellTags -> "SlideShowHeader"] }, content];
(*			slideBreakStyles = $SlideBreakStyles *),
		(* ELSE *)
			notebookContent = { Cell["", "SlideShowNavigationBar", "FirstSlide", CellTags -> "SlideShowHeader"] };
(*			slideBreakStyles = $SlideBreakStyles *)
		]
	];

	(************************
	   Return notebook
	*)
	With[{fontsname = fonts, colorsname = colors,
		styleDefs = styleDefinitions, options = nbOptions, slideStyles = slideBreakStyles},
		Notebook[
			Flatten[{notebookContent }],
			Sequence@@options,
			StyleDefinitions -> styleDefs,
			TaggingRules -> {"PresenterSettings" -> {
				"ShowToolbar" -> True,
				"ShowNavigationBar" -> False,
				"SlideBreakStyles" -> slideStyles,
				"ShowSlideBreaks" -> True,
				"Theme" -> {
					"ThemeName" -> themeDisplayName,
					"FontSetName" -> fontsname,
					"ColorSetName" -> colorsname
					(*"Name" -> themeDisplayName,*)
					(*
					"FontSet" -> fontSet,
					"ColorSet" -> colorSet
					*)
					}
				}},
			ScreenStyleEnvironment -> "Slideshow Working"
		]
	]
];




GetPresenterStylesheet[file_ /; FileType[file] === File]:=
Module[{styles, nbObj},

	nbObj = NotebookOpen[file, Visible -> False];
	styles = CurrentValue[nbObj, StyleDefinitions];
	NotebookClose[nbObj];

	Which[
		(* Not a PresenterTools stylesheet call *)
		(Head[styles] === Notebook) && Not@FreeQ[Cases[styles, Cell[StyleData[StyleDefinitions -> defs_]] :> defs, Infinity], "PresenterTools"],
			styles,
		(Head[styles] === FrontEnd`FileName) && Not@FreeQ[styles, "PresenterTools"],
			styles,
		True,
			$Failed
	]
];
GetPresenterStylesheet[a___]:= (Message[GetPresenterStylesheet::argx, {a}]; $Failed);
GetPresenterStylesheet::argx = "`1`";
GetPresenterStylesheet::info = "`1`";





(*******************************************************

 SetEnvironmentTaggingRules

	Only used if "PresenterSettings" TaggingRules === Inherited / do not exist

	in: Notebookbject
	out: None

*******************************************************)

SetEnvironmentTaggingRules[env_, nbObj_]:=
Module[{rules, currentRules},

	currentRules = CurrentValue[nbObj, {TaggingRules, "PresenterSettings"}];

	If[currentRules === Inherited,

		Switch[env,
			"Working",
				rules = {
					"ShowToolbar" -> True,
					"SlideBreakStyles" -> $SlideBreakStyles,
					"ShowSlideBreaks" -> True
				},
			"Slideshow Working",
				rules = {
					"ShowToolbar" -> True,
					"SlideBreakStyles" -> $SlideBreakStyles,
					"ShowSlideBreaks" -> True
				},
			"Scrolling Presentation",
				rules = {
					"ShowToolbar" -> False,
					"SlideBreakStyles" -> $SlideBreakStyles,
					"ShowSlideBreaks" -> False
				},
			"Slideshow Presentation",
				rules = {
					"ShowToolbar" -> False,
					"SlideBreakStyles" -> $SlideBreakStyles,
					"ShowSlideBreaks" -> False
				},
			True,
				rules = {
					"ShowToolbar" -> True,
					"SlideBreakStyles" -> $SlideBreakStyles,
					"ShowSlideBreaks" -> True
				}
		];

		CurrentValue[nbObj, {TaggingRules, "PresenterSettings"}] = rules,

	(* ELSE *)
		Null
	]
];
SetEnvironmentTaggingRules[a___]:= Message[SetEnvironmentTaggingRules::argx, {a}];
SetEnvironmentTaggingRules::argx = "`1`";
SetEnvironmentTaggingRules::info = "`1`";






(***********************************************************************

THEMES and THEMING

***********************************************************************)
(*

A 'theme' is a modular stylesheet.
The theme is an association consisting of cell styles and defines a specific list of option values:
*)

GetThemeInformation[name_String /; MemberQ[GetThemeNames[], name], attrib_String]:=
Module[{theme},
	theme = GetThemeInformation[name];

	If[MemberQ[Keys[theme], attrib],
		theme[attrib],
	(* ELSE *)
		Message[GetThemeInformation::info, StringJoin[attrib, " not found within theme ", name]];
		theme[attrib]
	]
];

GetThemeInformation[a_String, ___]:= (Message[GetThemeInformation::notdef, a]; $Failed)
GetThemeInformation[a___]:= (Message[GetThemeInformation::argx, {a}]; $Failed)
GetThemeInformation::argx = "`1`";
GetThemeInformation::info = "`1`";
GetThemeInformation::notdef = "Theme `1` not defined";



GetPresentationInformation[nbObj_]:=
Module[{},
	CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "Theme"}, "Default"]
]



(*******************************************************************************

	THEMES

*******************************************************************************)



(*******************************************************

 DefaultThemeInformation

	variable containing Default theme values

*******************************************************)

$defaultThemeInformation =
<|
	"FileName" -> "Default.nb",
	"StyleDefinitions" -> FrontEnd`FileName[{"PresenterTools"}, "Default.nb", CharacterEncoding -> "UTF-8"],
	"SlideBreakStyles" -> $SlideBreakStyles,
	"BackgroundImages" -> <|"Title" -> False|>,
	"FontSet" -> {
		"Display1"->"Source Sans Pro",
		"Display2"->"Source Sans Pro",
		"Display3"->"Source Sans Pro",
		"Text1"->"Source Sans Pro",
		"Text2"->"Source Sans Pro",
		"Text3"->"Source Sans Pro",
		"Dingbat"->"Source Sans Pro",
		"Default"->"Source Sans Pro"
	},
	"ColorSet" -> {
		(* "DD1100" *)
		"Display1" -> RGBColor[0.866667, 0.0666667, 0],
		(* "F77700" *)
		"Display2" -> RGBColor[0.968627, 0.466667, 0],
		(* "FD6053" *)
		"Display3" -> RGBColor[0.992157, 0.376471, 0.32549],
		(* "636363" *)
		"Text1" -> GrayLevel[0.388235],
		"Text2" -> GrayLevel[1],
		"Text3" -> GrayLevel[0],
		"Text1Reverse" -> GrayLevel[0.75],
		"Text2Reverse" -> GrayLevel[0.75],
		"Text3Reverse" -> GrayLevel[0.75],
		(* "DD1100" *)
		"Dingbat" -> RGBColor[0.866667, 0.0666667, 0],
		(* "636363" *)
		"Default" -> GrayLevel[0.388235],
		"Background" -> GrayLevel[1] (*RGBColor[0.866667, 0.0666667, 0]*)
	}
|>;



(*******************************************************

 SetThemeInformation

	in: String theme name, attributes
	out: None

*******************************************************)

SetThemeInformation[name_String, assoc_Association]:=
Module[{theme, keys},

	If[!MemberQ[PresenterTools`$ThemeNames, name],
		(* theme is not part of known themes list, so add it *)
		AppendTo[PresenterTools`$ThemeNames, name];
		(* all themes are based on "Default" *)
		theme = $defaultThemeInformation,
	(* ELSE *)
		theme = GetThemeInformation[name];
	];

	(* apply user's values from assoc *)
	keys = Keys[assoc];
	(theme[#] = assoc[#]) & /@ keys;

	(* define new theme information *)
	PresenterTools`GetThemeInformation[name] := theme;

	theme
];
SetThemeInformation[a___]:= Message[SetThemeInformation::argx, {a}];
SetThemeInformation::argx = "`1`";
SetThemeInformation::info = "`1`";





(*******************************************************

 LoadThemeDefinitions
 loads defined themes from file

	in: path to file
	out: None

*******************************************************)

LoadThemeDefinitions[path_String:$PresenterToolsThemeDefinitions]:=
Module[{},

	(* confirm that 'Default' theme is loaded *)
	If[!MemberQ[GetThemeNames[], "Default"],
		SetThemeInformation["Default", $defaultThemeInformation]
	];

	(* filepath to load isn't a String *)
	If[Head[path] =!= String,
		Message[LoadThemeDefinitions::chtype, path],
	(* ELSE *)
		(* is path a file? *)
		If[FileType[path] === File,
			(* does Get[path] succeed? *)
			If[Get[path] === $Failed,
				Message[LoadThemeDefinitions::info, "$Failed"]
			],
		(* ELSE *)
			Message[FileType::fstr, path]
		]
	]
];
LoadThemeDefinitions[a___]:= Message[LoadThemeDefinitions::argx, {a}];
LoadThemeDefinitions::argx = "`1`";
LoadThemeDefinitions::chtype = "Expected String, found `1`";
LoadThemeDefinitions::info = "Expected Null, received: `1`";














(* Themes from Design *)
GetThemeStyleHints[theme_String, set_String:"All"]:=
Module[{results},

	With[{name = theme},
		results =
			Switch[set,
				"FontSet",
					GetThemeInformation[name, "FontSet"],
				"ColorSet",
					GetThemeInformation[name, "ColorSet"],
				_,
					Join[
						GetThemeInformation[name, "FontSet"],
						GetThemeInformation[name, "ColorSet"],
						GetThemeInformation[name, "SlideBreakStyles"]
					]
			];

		results

	]
]




(*******************************************************

 TitleBackgroundQ

	in: xxx
	out: True | False

*******************************************************)

TitleBackgroundQ[themeName_String]:=
Module[{titleBackgroundQ},
	(* TODO: "BackgroundImages" routine needs error checking
	   AND the Key probably should be designed differently
	   *)
	titleBackgroundQ = PresenterTools`GetThemeInformation[themeName, "BackgroundImages"];
	titleBackgroundQ =
		If[Head[titleBackgroundQ] === Association,
			If[TrueQ[titleBackgroundQ["Title"]],
				True,
				(* ELSE *)
				False
			],
		(* ELSE *)
			False
		];
	(* check to make sure (for now) *)
	If[BooleanQ[titleBackgroundQ],
		titleBackgroundQ,
	(* ELSE *)
		False
	]
];
TitleBackgroundQ[a___]:= Message[TitleBackgroundQ::argx, {a}];
TitleBackgroundQ::argx = "`1`";
TitleBackgroundQ::info = "`1`";







(*******************************************************

 GetThemeFontStyleHints

	in: xxx
	out: yyy

*******************************************************)

(* if name not found, use "Default" *)
(*
GetThemeFontStyleHints[theme_String]:= GetThemeFontStyleHints["Default"];

GetThemeFontStyleHints[a___]:= Message[GetThemeFontStyleHints::argx, {a}];
GetThemeFontStyleHints::argx = "`1`";
GetThemeFontStyleHints::info = "`1`";
*)










(****************************************************************************

	MENUS

******************************************************************************)




(**************************************

 SetFontFamilyMenuList
	Row of style editing buttons on the "Style" tab
	in: None
	out: Grid

***************************************)

SetFontFamilyMenuList[]:=
Module[{},
	Flatten[{
		With[{fontList = FE`Evaluate[FEPrivate`GetPopupList["MenuListFonts"]]},
			RuleDelayed[ Style[#[[1]], FontFamily -> #[[1]]],
				PresenterTools`SetSelectionOptions["FontFamily", #[[2]], InputNotebook[]] ]& /@ fontList
		]
	}]
]
SetFontFamilyMenuList[a__] := Message[SetFontFamilyMenuList::argx, {a}];
SetFontFamilyMenuList::argx = "Too many arguments: `1`";
SetFontFamilyMenuList::info = "`1`";




(**************************************

 SetCellStyleMenu
	Cell style menu
	in: None
	out: Grid

***************************************)
(* TODO: Apply FontFamily from font menu choice *)
Options[SetCellStyleMenu] = {
	"ModifierString" -> "(plus slide break)",
	"StylesToAppend" -> {"SideNote", "SideCode"}
}
SetCellStyleMenu[nbObj_, opts:OptionsPattern[]]:=
Module[{styleList, slideBreakStyles, stylesToAppend, modifierString, modifierStringLength,
	longestSlideBreakLine, longestStyleLine, slideBreakStyleWidth, res, result},

	modifierString = OptionValue["ModifierString"];
	stylesToAppend = OptionValue["StylesToAppend"];

	(* Styles with auto slide breaks need to be marked in style menu,
	   "ModifierString" option is the string to append to style name
	*)
	modifierStringLength = StringLength[modifierString];

	styleList = Part[Rest[FE`Evaluate[FEPrivate`GetPopupList[nbObj, "MenuListStyles"]]], All, 1];
	styleList = Select[styleList, FreeQ[stylesToAppend, #] &, Infinity];

	(* list of automatic slidebreak styles *)
	slideBreakStyles = CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "SlideBreakStyles"}, {}];

	longestSlideBreakLine = Max[StringLength /@ slideBreakStyles] + StringLength[modifierString] + 2;
	longestStyleLine = Max[StringLength /@ styleList];

	slideBreakStyleWidth =
		If[longestSlideBreakLine > longestStyleLine,
			longestSlideBreakLine - modifierStringLength,
		(* ELSE *)
			longestStyleLine - modifierString
		];

	styleList =
		Join[styleList, {System`Delimiter}, stylesToAppend];

	res=
	Flatten[{
		With[{styles = styleList, slidebreaks = slideBreakStyles, imageSize = slideBreakStyleWidth*10, modifier = modifierString},
			With[{
				name = #
				},
				Which[
					(* style if member of list of automatic slidebreak styles *)
					MemberQ[slidebreaks, name],
						{
						(* style *)
						RuleDelayed[name, PresenterTools`InsertOrModifyStyledCell[nbObj, name]],
						(* style  with SlideShowNavigationBar cell written above *)
						RuleDelayed[
							Row[{Pane[name, ImageSize -> imageSize], modifier}, Alignment -> {{Left, Right}, Automatic}],
							(
								result = PresenterTools`InsertOrModifyStyledCell[nbObj, name];
								SelectionMove[nbObj, Before, Cell];
								If[name === "Title",
								FrontEndExecute[{FrontEnd`NotebookWrite[nbObj,
									   Cell["", "SlideShowNavigationBar", "FirstSlide", CellTags -> "SlideShowHeader"], After]}],
								(* ELSE *)
									FrontEndExecute[{FrontEnd`NotebookWrite[nbObj,
									   Cell["", "SlideShowNavigationBar", CellTags -> "SlideShowHeader"], After]}]
								];
								If[result["action"] === "insert",
									SelectionMove[nbObj, Next, Cell];
									SelectionMove[nbObj, All, CellContents],
								(* ELSE *)
									SelectionMove[nbObj, Next, Cell]
								]
							)
						]
						},
					name =!= Delimiter,
						RuleDelayed[name, PresenterTools`InsertOrModifyStyledCell[nbObj, name]],
					True, name
				]] & /@ styles
		]
	}];

	res
];
(*
SetCellStyleMenu[a__] := Message[SetCellStyleMenu::argx, {a}];
SetCellStyleMenu::argx = "`1`";
SetCellStyleMenu::info = "`1`";
*)




(*

SetOptions[InputNotebook[], NotebookEventActions -> {
   {"MenuCommand", "ScreenStyleEnvironment"} :> inc++,
   {"MenuCommand", {"ScreenStyleEnvironment", "SlideShow"}} :> inc++,
   {"MenuCommand", "Title"} :> inc++,
   {"MenuCommand", "PlainFont"} :> inc++,
   {"MenuCommand", "FontWeight"} :> inc++,
   {"MenuCommand", "FontVariationsUnderline"} :> inc++,

   {"MenuCommand", "FontSize"} :> inc++
   }]
*)

(*******************************************************************************

 SetSelectionOptions
	Set/Remove option on a selection

	in: Option with new value and NotebookObject
	out: Change option via FE

	notes: If no selection, pop 'select a cell' dialog

*******************************************************************************)
SetSelectionOptions[option_String, nbObj_NotebookObject, opts___?OptionQ]:=
	SetSelectionOptions[option, None, nbObj, opts];

SetSelectionOptions[option_String, value_, nbObj_NotebookObject, opts___?OptionQ]:=
Module[{cellInfo, selection, tag, lnkre},

	(* gather information once, then pass around *)
	selection = SelectedCells[nbObj];

	cellInfo = GetCellInformation[nbObj];

	(* GetCellInformation returns a list when multiple cells are selected and
		returns an Association otherwise.
	*)
	If[MatchQ[cellInfo, {__Association}] || (ListQ[cellInfo] && (Length[cellInfo] > 1)),
		(* multi-cell selection *)
		tag = StringJoin @@ (ToString /@ Date[]);
		FrontEndExecute[FrontEnd`SelectionAddCellTags[nbObj, tag]];
		SelectionMove[nbObj, Before, Cell, AutoScroll->False];
		While[NotebookFind[nbObj, tag, Next, CellTags, AutoScroll->False] =!= $Failed,
			iSetSelectionOptions[option, value, Developer`CellInformation[SelectedCells[nbObj][[1]]], nbObj, opts]];
		NotebookFind[nbObj, tag, All, CellTags, AutoScroll->False];
		FrontEndExecute[FrontEnd`SelectionRemoveCellTags[nbObj, tag]]
		,
	(* ELSE *)
		(* not a multi-cell selection
		   selection is either:
		      a) single cell bracket
		      b) within a cell
		      c) no selection
		   *)
		(* If $Failed, then there is no selection *)
		If[MatchQ[cellInfo, $Failed],
			(* no selection, so popup dialog *)
			cellInfo = None;
			SelectACellMessageDialog[],
		(* ELSE *)
			(* single selection *)
			(* Check CellInformation *)
(*
			While[(LinkWrite[$ParentLink, FrontEnd`CellInformation[nbObj]]; lnkre = LinkRead[$ParentLink]);
				(lnkre =!= $Failed && Not[MemberQ["CursorPosition" /. lnkre, "CellBracket"]]),
			  FrontEndExecute[FrontEnd`SelectionMove[nbObj, All, Cell, AutoScroll -> False]]];
			iSetSelectionOptions[option, value, Developer`CellInformation[SelectedCells[nbObj][[1]]], nbObj, opts]
*)
			iSetSelectionOptions[option, value, cellInfo, nbObj, opts]

		]
	]

];
SetSelectionOptions::info = "`1`";
SetSelectionOptions::argx = "Incorrect arguments: `1`";

iSetSelectionOptions[a___]:= Message[SetSelectionOptions::argx, {a}];
iSetSelectionOptions::info = "`1`";
iSetSelectionOptions::argx = "Incorrect arguments: `1`";



(*******************************************************

 iSetSelectionOptions
	Handles setting various cell option modifications

	in: type, value, CellInformation, NotebookObject
	out: None

*******************************************************)

(* multi-cell selection *)
(* Not used for types "FontSize" and "Nudge" *)

(* Map is not always the best here, should fix *)
(* support both CellInformation as List and Association *)
(*iSetSelectionOptions[type_String, value_, cellInfo:{__List}, nbObj_NotebookObject, opts___?OptionQ]:=
	iSetSelectionOptions[type, value, #, nbObj, opts] & /@ cellInfo;
iSetSelectionOptions[type_String, value_, cellInfo:{__Association}, nbObj_NotebookObject, opts___?OptionQ]:=
	iSetSelectionOptions[type, value, #, nbObj, opts] & /@ cellInfo;*)



(***************************************

	FontFamily

	Called from SetSelectionOptions["FontFamily", ...]

***************************************)
(* single-cell selection *)
iSetSelectionOptions["FontFamily", fontFamily_, cellInfo_ /; CursorInCellNoSelectionQ[cellInfo], nbObj_NotebookObject]:= Null;

iSetSelectionOptions["FontFamily", fontFamily_, cellInfo_, nbObj_NotebookObject]:=
Module[{style, position, log = False, defaultValueQ},

	If[Head[cellInfo] === Association,
		style = cellInfo["Style"];
		position = cellInfo["CursorPosition"],
	(* ELSE *)
		style = ("Style" /. cellInfo);
		position = ("CursorPosition" /. cellInfo);
	];

	Which[
		position === "CellBracket",
			CurrentValue[NotebookSelection[nbObj], FontFamily] = fontFamily;
			log = True,
		(* If CursorPosition is a List AND first element is 0, then it's possible
		   that all the cell contents are selected. If so, the select CellBracket
		   and apply options at the cell level instead
		   *)
		(Head[position] === List && First[position] === 0),
			(* if selection is all contents, then select the CellBracket instead *)
			If[AllContentsSelectedQ[nbObj],
				SelectCellBracket[nbObj, cellInfo];
				CurrentValue[NotebookSelection[nbObj], FontFamily] = fontFamily;
				log = True,
			(* ELSE *)
				(* subset of cell contents is selected *)
				CurrentValue[NotebookSelection[nbObj], FontFamily] = fontFamily;
				log = False
			],
		(* fall through *)
		True,
			CurrentValue[NotebookSelection[nbObj], FontFamily] = fontFamily;
			log = False
	];

	defaultValueQ = DefaultOptionValueQ[nbObj, style, FontFamily];

	If[defaultValueQ,
		CurrentValue[NotebookSelection[nbObj], FontFamily] = Inherited;
		SetStyleUpdateList[nbObj,
				PresenterTools`Styles`RemoveStyleOptionsList[nbObj, {style, FontFamily}]],
		(* ELSE *)
		(* log change for addition to embedded stylesheet *)
		If[log,	With[{s = First@Flatten[{style}]},
			(* only modify first cell style *)
			PresenterTools`Styles`SetStyleOptionsList[nbObj, {s, FontFamily -> fontFamily}]]
		]
	]
];




(***************************************

	FontWeight

	Called from SetSelectionOptions["FontWeight", ...]

***************************************)
(* multi-cell selection *)
(*iSetSelectionOptions["FontWeight", value_, cellInfo:{__List}, nbObj_NotebookObject]:=
	iSetSelectionOptions["FontWeight", value, #, nbObj] & /@ cellInfo;*)

(* single-cell selection *)
iSetSelectionOptions["FontWeight", value_, cellInfo_ /; CursorInCellNoSelectionQ[cellInfo], nbObj_NotebookObject]:= Null;

iSetSelectionOptions["FontWeight", value_, cellInfo_, nbObj_NotebookObject]:=
Module[{style, position, val, log = False, defaultValueQ},

	If[Head[cellInfo] === Association,
		style = cellInfo["Style"];
		position = cellInfo["CursorPosition"],
	(* ELSE *)
		style = ("Style" /. cellInfo);
		position = ("CursorPosition" /. cellInfo);
	];

	Which[
		position === "CellBracket",
			log = True,
		(* If CursorPosition is a List AND first element is 0, then it's possible
		   that all the cell contents are selected. If so, the select CellBracket
		   and apply options at the cell level instead
		   *)
		(Head[position] === List && First[position] === 0),
			(* if selection is all contents, then select the CellBracket instead *)
			If[AllContentsSelectedQ[nbObj],
				SelectCellBracket[nbObj, cellInfo];
				log = True,
			(* ELSE *)
				log = False
			],
		(* fall through *)
		True,
			log = False
	];

	(* set option on selection *)
	If[CurrentValue[NotebookSelection[nbObj], FontWeight] === "Bold",
		CurrentValue[NotebookSelection[nbObj], FontWeight] = "Plain";
		val = "Plain",
	(* ELSE *)
		CurrentValue[NotebookSelection[nbObj], FontWeight] = "Bold";
		val = "Bold"
	];

	defaultValueQ = DefaultOptionValueQ[nbObj, style, FontWeight];

	If[defaultValueQ,
		CurrentValue[NotebookSelection[nbObj], FontWeight] = Inherited;
			SetStyleUpdateList[nbObj,
				PresenterTools`Styles`RemoveStyleOptionsList[nbObj, {style, FontWeight}]],
		(* ELSE *)
		(* log change for addition to embedded stylesheet *)
		If[log, With[{s = First@Flatten[{style}], rhs = val},
			(* only modify first cell style *)
			PresenterTools`Styles`SetStyleOptionsList[nbObj, {s, FontWeight -> rhs}]]
		]
	]
];




(***************************************

	FontSlant

	Called from SetSelectionOptions["FontSlant", ...]

***************************************)
(* multi-cell selection *)
(*iSetSelectionOptions["FontSlant", value_, cellInfo:{__List}, nbObj_NotebookObject]:=
	iSetSelectionOptions["FontSlant", value, #, nbObj] & /@ cellInfo;*)

(* single-cell selection *)
iSetSelectionOptions["FontSlant", value_, cellInfo_ /; CursorInCellNoSelectionQ[cellInfo], nbObj_NotebookObject]:= Null;

iSetSelectionOptions["FontSlant", value_, cellInfo_, nbObj_NotebookObject]:=
Module[{style, position, val, log = False, defaultValueQ},

	If[Head[cellInfo] === Association,
		style = cellInfo["Style"];
		position = cellInfo["CursorPosition"],
	(* ELSE *)
		style = ("Style" /. cellInfo);
		position = ("CursorPosition" /. cellInfo);
	];

	Which[
		position === "CellBracket",
			log = True,
		(* If CursorPosition is a List AND first element is 0, then it's possible
		   that all the cell contents are selected. If so, the select CellBracket
		   and apply options at the cell level instead
		   *)
		(Head[position] === List && First[position] === 0),
			(* if selection is all contents, then select the CellBracket instead *)
			If[AllContentsSelectedQ[nbObj],
				SelectCellBracket[nbObj, cellInfo];
				log = True,
			(* ELSE *)
				log = False
			],
		(* fall through *)
		True,
			log = False
	];

	(* set option on selection *)
	If[CurrentValue[NotebookSelection[nbObj], FontSlant] === "Italic",
		CurrentValue[NotebookSelection[nbObj], FontSlant] = "Plain";
		val = "Plain",
	(* ELSE *)
		CurrentValue[NotebookSelection[nbObj], FontSlant] = "Italic";
		val = "Italic";
	];

	defaultValueQ = DefaultOptionValueQ[nbObj, style, FontSlant];

	If[defaultValueQ,
		CurrentValue[NotebookSelection[nbObj], FontSlant] = Inherited;
		SetStyleUpdateList[nbObj,
				PresenterTools`Styles`RemoveStyleOptionsList[nbObj, {style, FontSlant}]],
		(* ELSE *)
		(* log change for addition to embedded stylesheet *)
		If[log, With[{s = First@Flatten[{style}], rhs = val},
			(* only modify first cell style *)
			PresenterTools`Styles`SetStyleOptionsList[nbObj, {s, FontSlant -> rhs}]]
		]
	]
]




(***************************************

	FontUnderline

	Called from SetSelectionOptions["FontUnderline", ...]

***************************************)
(* multi-cell selection *)
(*iSetSelectionOptions["FontUnderline", value_, cellInfo:{__List}, nbObj_NotebookObject]:=
	iSetSelectionOptions["FontUnderline", value, #, nbObj] & /@ cellInfo;*)

(* single-cell selection *)
iSetSelectionOptions["FontUnderline", value_, cellInfo_ /; CursorInCellNoSelectionQ[cellInfo], nbObj_NotebookObject]:= Null;

iSetSelectionOptions["FontUnderline", value_, cellInfo_, nbObj_NotebookObject]:=
Module[{style, position, val, log = False, defaultValueQ},

	If[Head[cellInfo] === Association,
		style = cellInfo["Style"];
		position = cellInfo["CursorPosition"],
	(* ELSE *)
		style = ("Style" /. cellInfo);
		position = ("CursorPosition" /. cellInfo);
	];

	Which[
		position === "CellBracket",
			log = True,
		(* If CursorPosition is a List AND first element is 0, then it's possible
		   that all the cell contents are selected. If so, the select CellBracket
		   and apply options at the cell level instead
		   *)
		(Head[position] === List && First[position] === 0),
			(* if selection is all contents, then select the CellBracket instead *)
			If[AllContentsSelectedQ[nbObj],
				SelectCellBracket[nbObj, cellInfo];
				log = True,
			(* ELSE *)
				log = False
			],
		(* fall through *)
		True,
			log = False
	];

	(* set option on selection *)
	If[CurrentValue[NotebookSelection[nbObj], FrontEnd`FontVariationsUnderline] === True,
		CurrentValue[NotebookSelection[nbObj], FrontEnd`FontVariationsUnderline] = False;
		val = False,
	(* ELSE *)
		CurrentValue[NotebookSelection[nbObj], FrontEnd`FontVariationsUnderline] = True;
		val = True;
	];

	defaultValueQ = DefaultOptionValueQ[nbObj, style, FrontEnd`FontVariationsUnderline];

	If[defaultValueQ,
		CurrentValue[NotebookSelection[nbObj], FrontEnd`FontVariationsUnderline] = Inherited;
		SetStyleUpdateList[nbObj,
				PresenterTools`Styles`RemoveStyleOptionsList[nbObj, {style, FrontEnd`FontVariationsUnderline}]],
		(* ELSE *)
		(* log change for addition to embedded stylesheet *)
		If[log,
			With[{s = First@Flatten[{style}], rhs = val},
				(* only modify first cell style *)
(*				PresenterTools`Styles`SetStyleOptionsList[nbObj, {s, FontVariations->{"Underline"->rhs}}]*)
				PresenterTools`Styles`SetStyleOptionsList[nbObj, {s, FrontEnd`FontVariationsUnderline->rhs}]
			]
		];
	]
];




(***************************************

	FontColor

	Called from SetSelectionOptions["FontColor", ...]

***************************************)
(* multi-cell selection *)
(*iSetSelectionOptions["FontColor", value_, cellInfo:{__List}, nbObj_NotebookObject]:=
	iSetSelectionOptions["FontColor", value, #, nbObj] & /@ cellInfo;*)

(* single-cell selection *)
iSetSelectionOptions["FontColor", value_, cellInfo_ /; CursorInCellNoSelectionQ[cellInfo], nbObj_NotebookObject]:= Null;

iSetSelectionOptions["FontColor", value_, cellInfo_, nbObj_NotebookObject]:=
Module[{style, currentColor, newColor, position, log = False, defaultValueQ},

	If[Head[cellInfo] === Association,
		style = cellInfo["Style"];
		position = cellInfo["CursorPosition"],
	(* ELSE *)
		style = ("Style" /. cellInfo);
		position = ("CursorPosition" /. cellInfo);
	];

	Which[
		position === "CellBracket",
			log = True,
		(* If CursorPosition is a List AND first element is 0, then it's possible
		   that all the cell contents are selected. If so, the select CellBracket
		   and apply options at the cell level instead
		   *)
		(Head[position] === List && First[position] === 0),
			(* if selection is all contents, then select the CellBracket instead *)
			If[AllContentsSelectedQ[nbObj],
				SelectCellBracket[nbObj, cellInfo];
				log = True,
			(* ELSE *)
				log = False
			],
		(* fall through *)
		True,
			log = False
	];

	(* save current color, before applying color change *)
	currentColor = CurrentValue[NotebookSelection[nbObj], FontColor];

	Which[
		(* FontColor -> Theme color 1 *)
		value === "Display1",
			SetOptions[NotebookSelection[nbObj], FontColor :> Dynamic[CurrentValue[{System`StyleHints, "ColorSet", "Display1"}]] ];
			newColor = Dynamic[CurrentValue[{System`StyleHints, "ColorSet", "Display1"}]],
		(* FontColor -> Theme color 2 *)
		value === "Display2",
			SetOptions[NotebookSelection[nbObj], FontColor :> Dynamic[CurrentValue[{System`StyleHints, "ColorSet", "Display2"}]] ];
			newColor = Dynamic[CurrentValue[{System`StyleHints, "ColorSet", "Display2"}]],
		(* FontColor -> Theme color 3 *)
		value === "Display3",
			SetOptions[NotebookSelection[nbObj], FontColor :> Dynamic[CurrentValue[{System`StyleHints, "ColorSet", "Display3"}]] ];
			newColor = Dynamic[CurrentValue[{System`StyleHints, "ColorSet", "Display3"}]],
		(* FontColor -> Black *)
		value === "Black",
			CurrentValue[NotebookSelection[nbObj], FontColor] = GrayLevel[0];
			newColor = GrayLevel[0],
		(* FontColor -> Gray *)
		value === "Gray",
			CurrentValue[NotebookSelection[nbObj], FontColor] = GrayLevel[0.5];
			newColor = GrayLevel[0.5],
		(* FontColor -> White *)
		value === "White",
			CurrentValue[NotebookSelection[nbObj], FontColor] = GrayLevel[1];
			newColor = GrayLevel[1],
		(* Palette *)
		True,
			(* no value, so present FE FontColorDialog *)
			FrontEndExecute[{FrontEndToken[nbObj, "FontColorDialog", Automatic]}];
			(* new color = color of selection *)
			newColor = CurrentValue[NotebookSelection[nbObj], FontColor];
	];

	defaultValueQ = DefaultOptionValueQ[nbObj, style, FontColor];

	If[defaultValueQ,
		CurrentValue[NotebookSelection[nbObj], FontColor] = Inherited;
		SetStyleUpdateList[nbObj,
				PresenterTools`Styles`RemoveStyleOptionsList[nbObj, {style, FontColor}]],
		(* ELSE *)

		CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "LastColorUsed"}] = newColor;
		(* log change for addition to embedded stylesheet *)
		If[log,	With[{s = First@Flatten[{style}], color = newColor},
					(* only modify first cell style *)
					PresenterTools`Styles`SetStyleOptionsList[nbObj, {s, RuleDelayed[FontColor, color ]}]
					]
		]
	]
	(* do nothing if new color = old color *)
(*
	If[InputForm@Setting[newColor] =!= InputForm@Setting[currentColor],
		CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "LastColorUsed"}] = newColor;
		(* log change for addition to embedded stylesheet *)
		If[log,	With[{s = style, color = newColor},
					PresenterTools`Styles`SetStyleOptionsList[nbObj, {s, Rule[FontColor, color ]}, stylesheet]
					]
		],
	(* ELSE *)
		Return[]
	]
*)
];




(***************************************

	TextAlignment

	Called from SetSelectionOptions["TextAlignment", ...]

***************************************)
(* multi-cell selection *)
(*iSetSelectionOptions["TextAlignment", value_, cellInfo:{__List}, nbObj_NotebookObject]:=
	iSetSelectionOptions["TextAlignment", value, #, nbObj] & /@ cellInfo;*)

(* single-cell selection *)
iSetSelectionOptions["TextAlignment", value_, cellInfo_, nbObj_NotebookObject]:=
Module[{style, textAlignment, defaultValueQ},

	If[Head[cellInfo] === Association,
		style = cellInfo["Style"],
	(* ELSE *)
		style = ("Style" /. cellInfo);
	];

	SelectCellBracket[nbObj, cellInfo];
	(*  *)
	Which[
		(* TextAlignment -> Left *)
		value === "Left",
			textAlignment = Left,
		(* TextAlignment -> Right *)
		value === "Right",
			textAlignment = Right,
		 True,
			textAlignment = Center
	];

	(* CellBracket selected *)
	SetOptions[NotebookSelection[nbObj], TextAlignment -> textAlignment];

	defaultValueQ = DefaultOptionValueQ[nbObj, style, TextAlignment];

	If[defaultValueQ,
		CurrentValue[NotebookSelection[nbObj], TextAlignment] = Inherited;
		SetStyleUpdateList[nbObj,
				PresenterTools`Styles`RemoveStyleOptionsList[nbObj, {style, TextAlignment}]],
		(* ELSE *)
		(* log change for addition to embedded stylesheet *)
		With[{s = First@Flatten[{style}], align = textAlignment},
			(* only modify first cell style *)
			PresenterTools`Styles`SetStyleOptionsList[nbObj, {s, TextAlignment -> align}]]
	]
]




(**************************************

 FontSize
	Increase / decrease FontSize of selection,
	as a function of WindowSize
	Called from SetSelectionOptions["FontSize", ...]

	in: Notebook

***************************************)
(* multi-cell selection *)
(* support CellInformation as List and Association *)
(*iSetSelectionOptions["FontSize", value_, cellInfo:{__List}, nbObj_NotebookObject]:=
	iSetSelectionOptions["FontSize", value, #, nbObj] & /@ cellInfo;
iSetSelectionOptions["FontSize", value_, cellInfo:{__Association}, nbObj_NotebookObject]:=
	iSetSelectionOptions["FontSize", value, #, nbObj] & /@ cellInfo;*)

(* single-cell selection *)
iSetSelectionOptions["FontSize", value_, cellInfo_ /; CursorInCellNoSelectionQ[cellInfo], nbObj_NotebookObject]:= Null;

iSetSelectionOptions["FontSize", value_, cellInfo_, nbObj_NotebookObject]:=
Module[{style, position, log = False, width, height, currentSize, smidge = 0.005, percent, windowsize = AbsoluteCurrentValue[nbObj, {WindowSize}], defaultValueQ },

	(* Get Screen height and width *)
	{width, height} = windowsize;
	(* CellInformation *)
	If[Head[cellInfo] === Association,
		style = cellInfo["Style"];
		position = cellInfo["CursorPosition"],
	(* ELSE *)
		style = ("Style" /. cellInfo);
		position = ("CursorPosition" /. cellInfo);
	];

	Which[
		position === "CellBracket",
			log = True,
		(* If CursorPosition is a List AND first element is 0, then it's possible
		   that all the cell contents are selected. If so, the select CellBracket
		   and apply options at the cell level instead
		   *)
		(Head[position] === List && First[position] === 0),
			(* if selection is all contents, then select the CellBracket instead *)
			If[AllContentsSelectedQ[nbObj],
				SelectCellBracket[nbObj, cellInfo];
				log = True,
			(* ELSE *)
				log = False
			],
		(* fall through *)
		True,
			log = False
	];
	(* Change FontSize *)
	currentSize = AbsoluteCurrentValue[NotebookSelection[nbObj], FontSize];

	(*
		Avoid FontSize getting to small.
		Minimun could be a number, but using '8' for now.
	*)
	If[(currentSize < 8) && (value < 0), log = False ];

(*
	percent =
		If[value <= 0,
			N[(currentSize / width) - smidge, 4],
		(* ELSE *)
			(* Increase *)
			N[(currentSize / width) + smidge, 4]
		];
*)
		smidge =
			Which[
				(* -3 *)
				(value <= -3), smidge*-5,
				(* -2 *)
				(-3 < value <= -2), smidge*-3,
				(* -1 *)
				(-2 < value <= -1), smidge*-1,
				(*  0 *)
				(-1 < value < 1), smidge*0,
				(* +1 *)
				(2 > value >= 1), smidge*1,
				(* +2 *)
				(3 > value >= 2), smidge*3,
				(* +3 *)
				(value >= 3), smidge*5,
				(* do nothing *)
				True, smidge*0
			];

	percent = N[(currentSize / width) + smidge, 4];

	(* FontSize can't be toooo small... *)
	If[percent < 0.01, percent = 0.01];

	With[{per = percent},
		SetOptions[NotebookSelection[nbObj],
			FontSize :> Dynamic[(per * FrontEnd`AbsoluteCurrentValue[{WindowSize, 1}])]]
	];

	defaultValueQ = DefaultOptionValueQ[nbObj, style, FontSize];

	If[defaultValueQ,
		CurrentValue[NotebookSelection[nbObj], FontSize] = Inherited;
		SetStyleUpdateList[nbObj,
				PresenterTools`Styles`RemoveStyleOptionsList[nbObj, {style, FontSize}]],
		(* ELSE *)

		If[log,
			(* log change for addition to embedded stylesheet *)
			With[{s = First@Flatten[{style}], per = percent, size = OptionValue[Options[NotebookSelection@nbObj], FontSize]},
				(* only modify first cell style *)
				PresenterTools`Styles`SetStyleOptionsList[nbObj, {s, FontSize :> Dynamic[(per * FrontEnd`AbsoluteCurrentValue[{WindowSize, 1}])] }]
				]
		]
	]
];





(****************************)
(* FontSizeReset			*)
(****************************)
FontSizeReset[nbObj_NotebookObject, windowsize_List] :=
Module[{cellInfo, width, height, currentSize},
	(* Get Screen height and width *)
	{width, height} = windowsize;
	(* Check CellInformation *)
	cellInfo = GetCellInformation[nbObj];
	If[TrueQ[cellInfo == $Failed],
		(* Selection is between Cells *)
		SelectACellMessageDialog[]
	,
		(* Check CellBracket selected *)
		If[MatchQ[cellInfo, {{__, "CursorPosition" -> "CellBracket", __}}],
			(* CellBracket selected *)
			(* Change FontSize *)
			currentSize = AbsoluteCurrentValue[NotebookSelection[nbObj], FontSize];
			percent = N[(currentSize / width) + smidge, 4];
			SetOptions[NotebookSelection[nbObj], FontSize -> Inherited]
		,
			(* CellBracket NOT selected *)
			(* Change FontSize *)
			currentSize = AbsoluteCurrentValue[NotebookSelection[nbObj], FontSize];
			percent = N[(currentSize / width) + smidge, 4];
			SetOptions[NotebookSelection[nbObj], FontSize -> Inherited]
		]
	]
];




(***************************************

	Nudge Left/Right/Up/Down

	Called from SetSelectionOptions["Nudge", "Left|Right|Up|Down"...]

***************************************)
(* multi-cell selection *)
(* support CellInformation as List and Association *)
iSetSelectionOptions["Nudge", value_String, cellInfo:{__List}, nbObj_NotebookObject, opts___?OptionQ]:=
	iSetSelectionOptions["Nudge", value, #, nbObj, opts] & /@ Take[cellInfo, 1];
iSetSelectionOptions["Nudge", value_String, cellInfo:{__Association}, nbObj_NotebookObject, opts___?OptionQ]:=
	iSetSelectionOptions["Nudge", value, #, nbObj, opts] & /@ Take[cellInfo, 1];

(* single-cell selection *)
iSetSelectionOptions["Nudge", val_String, cellInfo_, nbObj_NotebookObject, opts___?OptionQ]:=
Module[{style, cellMargins, heldMargins, log = True, optionValue, userOptionsList, styleCellMargins, originator},

	originator = "Originator" /. {opts} /. {"Originator" -> "Notebook"};

	If[Head[cellInfo] === Association,
		style = cellInfo["Style"],
	(* ELSE *)
		style = ("Style" /. cellInfo);
	];

	SelectCellBracket[nbObj, cellInfo];
	userOptionsList = GetUserModifications[nbObj];

	styleCellMargins = PresenterTools`Styles`GetUserCellStyleOptionValue[userOptionsList, style, CellMargins];

	(*  *)
	Which[
		(********************
			Nudge -> Left
		********************)
		StringMatchQ[val, "Left"],
			heldMargins =
				SetHorizontalCellMargin["NudgeLeft",
					AbsoluteCurrentValue[NotebookSelection[nbObj], CellMargins],
					nbObj,
					AbsoluteCurrentValue[nbObj, {WindowSize}],
					AbsoluteCurrentValue[NotebookSelection[nbObj], {TextAlignment}]
				];
			optionValue = {-1, 0};
			cellMargins = ReleaseHold @ heldMargins,
		(********************
			Nudge -> Right
		********************)
		StringMatchQ[val, "Right"],
			heldMargins =
				SetHorizontalCellMargin["NudgeRight",
					AbsoluteCurrentValue[NotebookSelection[nbObj], CellMargins],
					nbObj,
					AbsoluteCurrentValue[nbObj, {WindowSize}],
					AbsoluteCurrentValue[NotebookSelection[nbObj], {TextAlignment}]
				];
			optionValue = {1, 0};
			cellMargins = ReleaseHold @ heldMargins,
		(********************
			Nudge -> Up
		********************)
		StringMatchQ[val, "Up"],
			heldMargins =
				SetVerticalCellMargin["NudgeUp",
					AbsoluteCurrentValue[NotebookSelection[nbObj], CellMargins],
					nbObj,
					AbsoluteCurrentValue[nbObj, {WindowSize}]
				];
			optionValue = {0, 1};
			cellMargins = ReleaseHold @ heldMargins,
		(********************
			Nudge -> Down
		********************)
		StringMatchQ[val, "Down"],
			heldMargins =
				SetVerticalCellMargin["NudgeDown",
					AbsoluteCurrentValue[NotebookSelection[nbObj], CellMargins],
					nbObj,
					AbsoluteCurrentValue[nbObj, {WindowSize}]
				];
			optionValue = {0, -1};
			cellMargins = ReleaseHold @ heldMargins,
		(********************
			Fall through
		********************)
		True,
			Message[SetSelectionOptions::argx, "Incorrect nudge direction: " <> ToString[val]];
			log = False;
	];

(*	cellMargins = If[(originator === "Palette"), cellMargins, cellMargins /. _NotebookObject :> System`EvaluationNotebook[]]; *)
	cellMargins = cellMargins /. FrontEnd`AbsoluteCurrentValue[_NotebookObject, b___] :> FrontEnd`AbsoluteCurrentValue[b];
	cellMargins = cellMargins /. AbsoluteCurrentValue[_NotebookObject, b___] :> FrontEnd`AbsoluteCurrentValue[b];

		If[log,
			With[{margins = cellMargins},
				SetOptions[NotebookSelection[nbObj], CellMargins :> Dynamic[margins]]
			];

		defaultValueQ = DefaultOptionValueQ[nbObj, style, CellMargins];

		If[defaultValueQ,
			CurrentValue[NotebookSelection[nbObj], CellMargins] = Inherited;
			SetStyleUpdateList[nbObj,
				PresenterTools`Styles`RemoveStyleOptionsList[nbObj, {style, CellMargins}]],
			(* ELSE *)
			(* log change for addition to embedded stylesheet *)
			With[{sty = First@Flatten[{style}], value = heldMargins, adjValue = optionValue, margins = cellMargins},
				(* only modify first cell style *)
				PresenterTools`Styles`SetStyleOptionsList[nbObj, {sty, CellMargins :> (*adjValue*) Dynamic[margins]}]
			]
		]
	]
]





(****************************************************************************

 SetVerticalCellMargin
	Used by Nudge Up/Down

	in: current cell's CellMargins and nb WindowSize
	out: adjusted CellMargins

******************************************************************************)

SetVerticalCellMargin[directive_String, 0, nbObj_, windowsize_List, amount_:0] :=
	SetVerticalCellMargin[directive, {{0, 0}, {0, 0}}, nbObj, windowsize, amount];
(*
SetVerticalCellMargin[directive_String, Dynamic[{{l_, r_}, {b_, t_}}], windowsize_List, override_:0] :=
	SetVerticalCellMargin[directive, {{l, r}, {b, t}}, windowsize, override];
*)
SetVerticalCellMargin[directive_String, {{l_, r_}, {b_, t_}}, nbObj_, windowsize_List, override_:0] :=
Module[{w, h, n, percentage},
	{w, h} = windowsize;

	percentage = N[t/h, 4];

	If[override =!= 0,
		n = override,
	(* ELSE *)
		n = Switch[directive,
			"FullUp", 0.01,
			"MediumUp", 0.2,
			"NudgeUp", percentage - 0.025,
			"Center", 0.35,
			"NudgeDown", percentage + 0.025,
			"MediumDown", 0.55,
			"FullDown", 0.75,
			_, 0.35
		]
	];

	n = N[n, 4];
	(* Prohibit cells from being nudged too far up/down *)
(*	n = If[0 <= n <= 0.76, n, percentage];*)
(*	n = If[0 <= n, n, percentage]; *)

	(* Return modified CellMargins *)
	With[{scale = n, bottom = b, left = l, right = r, height = h, width = w},
		Hold[
			{
				{
					N[left/width, 4]*FrontEnd`AbsoluteCurrentValue[nbObj, {WindowSize, 1}],
					N[right/width, 4]*FrontEnd`AbsoluteCurrentValue[nbObj, {WindowSize, 1}]
				},
				{
					N[bottom/height, 4]*FrontEnd`AbsoluteCurrentValue[nbObj, {WindowSize, 2}],
					scale*FrontEnd`AbsoluteCurrentValue[nbObj, {WindowSize, 2}]
				}
			}
		]]
];
(* 'Reset' set CellMargins to Inherited *)
SetVerticalCellMargin["Reset", {{l_, r_}, {b_, t_}}, __] := {{l, r}, {Inherited, Inherited}};




(****************************************************************************

 SetHorizontalCellMargin
	Used by Nudge Left/Right

	in: current cell's CellMargins and nb WindowSize
	out: adjusted CellMargins

******************************************************************************)

(*
	If TextAlignment -> Left then
		* NudgeLeft decreases the Left CellMargin
		* NudgeRight increases the Left CellMargin
*)
(************************

	NudgeLeft

************************)
(*
NudgeRight, {{92.16, 92.16}, {21.44, 129.6}}, NotebookObject[<<Untitled-1.nb>>], {1152, 720}, Left
*)
SetHorizontalCellMargin["NudgeLeft", 0, nbObj_, windowsize_List, alignment_] :=
	SetHorizontalCellMargin["NudgeLeft", {{0, 0}, {0, 0}}, nbObj, windowsize, alignment, 0.05];

SetHorizontalCellMargin["NudgeLeft", 0, nbObj_, windowsize_List, alignment_, nudge_:0.05] :=
	SetHorizontalCellMargin["NudgeLeft", {{0, 0}, {0, 0}}, nbObj, windowsize, alignment, nudge];

SetHorizontalCellMargin["NudgeLeft", {{0, 0}, {0, 0}}, nbObj_, windowsize_List, alignment_] :=
	SetHorizontalCellMargin["NudgeLeft", {{0, 0}, {0, 0}}, nbObj, windowsize, alignment, 0.05];

SetHorizontalCellMargin["NudgeLeft", {{l_, r_}, {b_, t_}}, nbObj_, windowsize_List, alignment_, nudgeValue_:0.05] :=
Module[{h, w, leftPercentage, rightPercentage, textAlignment, nudge = nudgeValue},
	(* Get Screen height and width *)
	{w, h} = windowsize;
(*	 Message[SetHorizontalMargin::val, {w, h}];*)
	nudge =
		If[NumberQ[nudgeValue],
			nudgeValue,
		(* ELSE *)
			Message[SetHorizontalMargin::val, "Nudge value needs to be a number: ", nudgeValue];
			0.05
		];

	(* Change TextAlignment to a number *)
	If[ !NumberQ[alignment],
		textAlignment = Switch[alignment, Left, -1, Right, 1, Center, 0, _, alignment]];
	(* Message[SetHorizontalMargin::val, ToString[textAlignment]]; *)
	Which[
		(* TextAlignment -> Left *)
		textAlignment < 0,
			(
				leftPercentage = N[l/w, 3];
				leftPercentage = N[leftPercentage - nudge, 4];
				rightPercentage = N[r/w, 4];
				(* Prohibit cells from being nudged too far left/right *)
					 leftPercentage = If[0 <= leftPercentage <= 0.76, leftPercentage, 0];
			),
			(* TextAlignment -> Right *)
		textAlignment > 0,
			(
				leftPercentage = N[l/w, 3];
				rightPercentage = N[r/w, 4];
				rightPercentage = N[rightPercentage + nudge, 4];
				(* Prohibit cells from being nudged too far left/right *)
					 rightPercentage = If[0 <= rightPercentage <= 0.76, rightPercentage, 0];
			),
		True,
			(
					leftPercentage = N[l/w, 3];
					rightPercentage = N[r/w, 4];
					leftPercentage = 0;
					rightPercentage = rightPercentage + nudge;
			)
	];
		(* Return modified CellMargins *)
	With[{left = leftPercentage, right = rightPercentage, bottom = b, top = t, width = w, height = h},
		Hold[
				{
					{left*FrontEnd`AbsoluteCurrentValue[nbObj, {WindowSize, 1}], right*FrontEnd`AbsoluteCurrentValue[nbObj, {WindowSize, 1}]},
				{N[bottom/height, 4]*FrontEnd`AbsoluteCurrentValue[nbObj, {WindowSize, 2}], N[top/height, 4]*FrontEnd`AbsoluteCurrentValue[nbObj, {WindowSize, 2}]
				}}
			]]
];




(***********************

	NudgeRight

************************)
SetHorizontalCellMargin["NudgeRight", 0, nbObj_, windowsize_List, alignment_] :=
	SetHorizontalCellMargin["NudgeRight", {{0, 0}, {0, 0}}, nbObj, windowsize, alignment, 0.05];

SetHorizontalCellMargin["NudgeRight", 0, nbObj_, windowsize_List, alignment_, nudge_:0.05] :=
	SetHorizontalCellMargin["NudgeRight", {{0, 0}, {0, 0}}, nbObj, windowsize, alignment, nudge];

SetHorizontalCellMargin["NudgeRight", {{0, 0}, {0, 0}}, nbObj_, windowsize_List, alignment_] :=
	SetHorizontalCellMargin["NudgeRight", {{0, 0}, {0, 0}}, nbObj, windowsize, alignment, 0.05];

SetHorizontalCellMargin["NudgeRight", {{l_, r_}, {b_, t_}}, nbObj_, windowsize_List, alignment_, nudgeValue_:0.05] :=
Module[{h, w, leftPercentage, rightPercentage, textAlignment, nudge = nudgeValue},
	(* Get Screen height and width *)
	{w, h} = windowsize;
(*	 Message[SetHorizontalMargin::val, {w, h}];*)
	nudge =
		If[NumberQ[nudgeValue],
			nudgeValue,
		(* ELSE *)
			Message[SetHorizontalMargin::val, "Nudge value needs to be a number: ", nudgeValue];
			0.05
		];

	(* Change TextAlignment to a number *)
	(* Left -> -1, Center -> 0, Right -> 1*)
	If[ !NumberQ[alignment],
		textAlignment = Switch[alignment, Left, -1, Right, 1, Center, 0, _, alignment]];
	(* Message[SetHorizontalMargin::val, ToString[textAlignment]]; *)
	Which[
		(* TextAlignment -> Left *)
		textAlignment < 0,
			(
				leftPercentage = N[l/w, 3];
				leftPercentage = N[leftPercentage + nudge, 4];
				rightPercentage = N[r/w, 4];
				(* Prohibit cells from being nudged too far left/right *)
				leftPercentage = If[0 <= leftPercentage <= 0.76, leftPercentage, 0];
			),
			(* TextAlignment -> Right *)
		textAlignment > 0,
			(
				leftPercentage = N[l/w, 3];
				rightPercentage = N[r/w, 4];
				rightPercentage = N[rightPercentage - nudge, 4];
				(* Prohibit cells from being nudged too far left/right *)
				rightPercentage = If[0 <= rightPercentage <= 0.76, rightPercentage, 0];
			),
		True,
			(
				leftPercentage = N[l/w, 3];
				rightPercentage = N[r/w, 4];
				rightPercentage = 0;
				leftPercentage = leftPercentage + nudge;
			)
	];
	With[{left = leftPercentage, right = rightPercentage, width = w, height = h},
		Hold[
				{
					{left*FrontEnd`AbsoluteCurrentValue[nbObj, {WindowSize, 1}], right*FrontEnd`AbsoluteCurrentValue[nbObj, {WindowSize, 1}]},
					{N[b/height, 4]*FrontEnd`AbsoluteCurrentValue[nbObj, {WindowSize, 2}], N[t/height, 4]*FrontEnd`AbsoluteCurrentValue[nbObj, {WindowSize, 2}]
				}}
			]
		]
];



(************************

	Reset

************************)
(* 'Reset' set CellMargins to Inherited *)
SetHorizontalCellMargin["Reset", {{l_, r_}, {b_, t_}}, nbObj_, windowsize_List, textAlignment_] :=
Module[{h, w},
	{w, h} = windowsize;
	(* Return modified CellMargins *)
	With[{bottom = b, top = t, width = w, height = h},
		Hold[{
			{Inherited, Inherited},
			{N[bottom/height, 4]*FrontEnd`AbsoluteCurrentValue[nbObj, {WindowSize, 2}], N[top/height, 4]*FrontEnd`AbsoluteCurrentValue[nbObj, {WindowSize, 2}]
			}}]
		]
];
SetHorizontalCellMargin[a___] := (
	Message[SetHorizontalMargin::argx, ToString[{a}]];
	Hold[{{Inherited, Inherited}, {Inherited, Inherited}}]
);





















(*
SlideShowAuthoring`Private`slideShowDockHeight = 172;
PresenterTools`Private`slideShowDockHeight = 172;
*)


(**************************************

 SetFontSizeInputField

	in: None
	out: Grid

***************************************)
(* NOT USED *)
SetFontSizeInputField[]:=
Module[{},
	InputField[Dynamic[fontSize], Number,
		FieldHint -> "12",
		ImageSize -> {30, Automatic},
		Alignment -> {Right, Center}
	]
];




(**************************************

 NudgingValueInputField

	in: None
	out: Pane

***************************************)

NudgingValueInputField[]:=
Module[{},
	InputField[Dynamic[nudgingValue], Number,
		FieldHint -> "12",
		ImageSize -> {30, Automatic},
		Alignment -> {Right, Center}
	]

];




(**************************************

 NudgeResetButton

	in: None
	out: Pane

***************************************)
(*
NudgeResetButton[nbObj_NotebookObject]:=
Module[{bb},
	bb =
		Button["X",
			If[GetSelection[nbObj] =!= $Failed,
				SetOptions[NotebookSelection[nbObj],
					CellMargins -> {{Inherited, Inherited}, {Inherited, Inherited}}]
			,
				Logger["CellInformation: $Failed", "DEBUG"];
				Message[GetSelection::val, ToString[GetSelection[nbObj]]]
			],
		Appearance -> None];

	TemplateButtonWithHover[bb, "Clear cell margins"]
];

*)



(**************************************

 ClearCellFormatting

	Removes cell level options that have been added by PresenterTools

	in: NotebookObject
	out: None

***************************************)

ClearCellFormatting[]:= ClearCellFormatting[GetCellInformation[InputNotebook[]], InputNotebook[]];

ClearCellFormatting[nbObj_NotebookObject]:= ClearCellFormatting[GetCellInformation[nbObj], nbObj]

ClearCellFormatting[$Failed, ___]:= Null;

(* multi-cell selection *)
(* support CellInformation as List and Association *)
ClearCellFormatting[cellInfo:{__List}, nbObj_NotebookObject]:=
	ClearCellFormatting[#, nbObj] & /@ cellInfo
ClearCellFormatting[cellInfo:{__Association}, nbObj_NotebookObject]:=
	ClearCellFormatting[#, nbObj] & /@ cellInfo

(* single-cell selection *)
ClearCellFormatting[cellInfo_, nbObj_NotebookObject]:=
Module[{style, position, clear, optionList},

	If[Head[cellInfo] === Association,
		style = cellInfo["Style"];
		position = cellInfo["CursorPosition"],
	(* ELSE *)
		style = ("Style" /. cellInfo);
		position = ("CursorPosition" /. cellInfo);
	];

	Logger[{position, style}];

	(* cursor within cell, but no selection *)
	If[CursorInCellNoSelectionQ[cellInfo], Return[]];

	Which[
		position === "CellBracket",
			clear = True,
		(* If CursorPosition is a List AND first element is 0, then it's possible
		   that all the cell contents are selected. If so, the select CellBracket
		   and apply options at the cell level instead
		   *)
		(Head[position] === List && First[position] === 0),
			(* if selection is all contents, then select the CellBracket instead *)
			If[AllContentsSelectedQ[nbObj],
				SelectCellBracket[nbObj, cellInfo];
				clear = True,
			(* ELSE *)
				clear = False
			],
		(* fall through *)
		True,
			(* fall through - > do nothing *)
			clear = False
	];

	If[clear === True,

		(* remove cell level options from user's cell *)
		SetOptions[NotebookSelection[nbObj],
			FontSize -> Inherited,
			FontWeight -> Inherited,
			FontSlant -> Inherited,
			FrontEnd`FontVariationsUnderline -> Inherited,
			FontColor -> Inherited,
			CellMargins -> Inherited,
			TextAlignment -> Inherited
			];
		FrontEndTokenExecute[nbObj, "ClearCellOptions"];

		(* remove options from GetUserModifications *)
		optionList = GetUserModifications[nbObj];
		optionList = DeleteStyleOptionsUpdate[nbObj, optionList, style,
						{FontSize, FontWeight, FontSlant, FrontEnd`FontVariationsUnderline, FontColor, CellMargins, TextAlignment}];
		SetStyleUpdateList[nbObj, optionList],

	(* ELSE *)
		SelectACellMessageDialog[]
	]
];
ClearCellFormatting[a__] := Message[ClearCellFormatting::argx, {a}];
ClearCellFormatting::argx = "Argument should be a NotebookObject: `1`";
ClearCellFormatting::info = "`1`";











(**************************************

 InsertSlideBreak
	Insert new slide

	New slide insertion logic:
	http://files2.wolfram.com/ux/2016/Cloud&Desktop/Mathematica/SlideDeck/2.1/#g=1&p=inserting_slide_break
	a) If cursor is between cells, the break happens at that position.
	b) If a cell bracket is selected, the break occurs right before the selected cell.
	c) If content is selected or if the text cursor is in a cell, the break occurs before the selection or cursor location.

***************************************)

InsertSlideBreak[opts___?OptionQ]:= InsertSlideBreak[InputNotebook[], opts];

(* Check CellInformation *)
InsertSlideBreak[nbObj_NotebookObject, opts___?OptionQ]:=
	InsertSlideBreak[nbObj, GetCellInformation[nbObj], opts];

(* no selection; write slide break cell *)
InsertSlideBreak[nbObj_NotebookObject, $Failed, opts___?OptionQ]:=
Module[{},
	SelectionMove[nbObj, After, Cell];
	InsertSlideBreakCell[nbObj, opts]
];

(* multiple cells selected; write slide break after selection *)
InsertSlideBreak[nbObj_NotebookObject, {cellInformation__List}, opts___?OptionQ]:=
Module[{},
	SelectionMove[nbObj, After, Cell];
	InsertSlideBreakCell[nbObj, opts]
];
(* multiple cells selected; write slide break after selection *)
InsertSlideBreak[nbObj_NotebookObject, {cellInformation__Association}, opts___?OptionQ]:=
Module[{},
	SelectionMove[nbObj, After, Cell];
	InsertSlideBreakCell[nbObj, opts]
];

InsertSlideBreak[nbObj_NotebookObject, cellInformation_Association, opts___?OptionQ]:=
Module[{cellInfo = cellInformation, cursorPosition, contentDataForm, cellStyle, slideBreakStyles},

	(* Selection is NOT between Cells... *)
	(* set some variables to use in query *)
	cursorPosition = cellInfo["CursorPosition"];
	contentDataForm = cellInfo["ContentDataForm"];
	(* possible multiple cell styles *)
	cellStyle = cellInfo["Style"];

	Which[
		(* CellBracket selected, add break above cell *)
		cursorPosition === "CellBracket",
			SelectionMove[nbObj, Before, Cell];
			InsertSlideBreakCell[nbObj, opts],
		(* Cursor within unformatted cell expression, add break above cell *)
		contentDataForm === System`CellExpression,
			SelectionMove[nbObj, Before, Cell];
			InsertSlideBreakCell[nbObj, opts],
		(* If cursor within cell expression, split cell at cursor *)
		(* If no selection, split cell and add break between cells *)
		CursorInCellNoSelectionQ[cellInfo],
(*					SelectionMove[nbObj, Before, Cell]; *)
			SplitCellAtCursor[nbObj];
			SelectionMove[nbObj, Before, Cell];
			(* cellStyle should be String due to multi-cell catch above *)
			If[Not@MemberQ[
				CurrentValue[InputNotebook[], {TaggingRules, "PresenterSettings", "SlideBreakStyles"}, {}],
				cellStyle],
				InsertSlideBreakCell[nbObj, opts]
			],
		(* There is a selection, move cursor to beginning of selection and split cell and add break between cells *)
		CellContentsSelectedQ[cellInfo],
			SelectionMove[nbObj, Before, Character];
			SplitCellAtCursor[nbObj];
			SelectionMove[nbObj, Before, Cell];
			(* cellStyle should be String due to multi-cell catch above *)
			If[Not@MemberQ[
				CurrentValue[InputNotebook[], {TaggingRules, "PresenterSettings", "SlideBreakStyles"}, {}],
				cellStyle],
				InsertSlideBreakCell[nbObj, opts]
			],
		True,
			(* TODO: Clean up fail case *)
			SelectionMove[nbObj, After, Notebook];
			InsertSlideBreakCell[nbObj, opts]
(*			MessageDialog["Failed to determine where to add slide break"]*)
	]
];
InsertSlideBreak[nbObj_NotebookObject, {cellInformation_List}, opts___?OptionQ]:=
Module[{cellInfo = cellInformation, cursorPosition, contentDataForm, cellStyle, slideBreakStyles},

	(
		(* Where is cursor? *)
		If[MatchQ[cellInfo, $Failed],
			(* Selection is between Cells *)
			InsertSlideBreakCell[nbObj, opts],
		(* ELSE *)
			(* Selection is NOT between Cells... *)
			(* set some variables to use in query *)
(*			cursorPosition = cellInfo["CursorPosition"];
			contentDataForm = cellInfo["ContentDataForm"];
			(* possible multiple cell styles *)
			cellStyle = cellInfo["Style"];
*)
			cursorPosition = ("CursorPosition" /. cellInfo);
			contentDataForm = ("ContentDataForm" /. cellInfo);
			(* possible multiple cell styles *)
			cellStyle = ("Style" /. cellInfo);

			Which[
				(* CellBracket selected, add break above cell *)
				cursorPosition === "CellBracket",
					SelectionMove[nbObj, Before, Cell];
					InsertSlideBreakCell[nbObj, opts],
				(* Cursor within unformatted cell expression, add break above cell *)
				contentDataForm === System`CellExpression,
					SelectionMove[nbObj, Before, Cell];
					InsertSlideBreakCell[nbObj, opts],
				(* If cursor within cell expression, split cell at cursor *)
				(* If no selection, split cell and add break between cells *)
				CursorInCellNoSelectionQ[cellInfo],
(*					SelectionMove[nbObj, Before, Cell]; *)
					SplitCellAtCursor[nbObj];
					SelectionMove[nbObj, Before, Cell];
					(* cellStyle should be String due to multi-cell catch above *)
					If[Not@MemberQ[
						CurrentValue[InputNotebook[], {TaggingRules, "PresenterSettings", "SlideBreakStyles"}, {}],
						cellStyle],
						InsertSlideBreakCell[nbObj, opts]
					],
				(* There is a selection, move cursor to beginning of selection and split cell and add break between cells *)
				CellContentsSelectedQ[cellInfo],
					SelectionMove[nbObj, Before, Character];
					SplitCellAtCursor[nbObj];
					SelectionMove[nbObj, Before, Cell];
					(* cellStyle should be String due to multi-cell catch above *)
					If[Not@MemberQ[
						CurrentValue[InputNotebook[], {TaggingRules, "PresenterSettings", "SlideBreakStyles"}, {}],
						cellStyle],
						InsertSlideBreakCell[nbObj, opts]
					],
				True,
					(* TODO: Clean up fail case *)
					MessageDialog["Failed to determine where to add slide break"]
			]
		]
	)
];
InsertSlideBreak[a__] := Message[InsertSlideBreak::argx, {a}];
InsertSlideBreak::argx = "Argument should be a NotebookObject: `1`";
InsertSlideBreak::info = "`1`";




(**************************************

 InsertSlideBreak
	Insert new slide

	in: None
	out: None

***************************************)
Options[InsertSlideBreakCell] = {
	"IncludeBackground" -> False
}
InsertSlideBreakCell[opts___?OptionQ]:= InsertSlideBreakCell[InputNotebook[], opts];
InsertSlideBreakCell[nbObj_NotebookObject, opts___?OptionQ]:=
Module[{firstQ},
	firstQ = "IncludeBackground" /. {opts} /. Options[InsertSlideBreakCell];
	If[MatchQ[firstQ, True],
		FrontEndExecute[{
			FrontEnd`NotebookWrite[nbObj,
				Cell["", "SlideShowNavigationBar", "FirstSlide", CellTags->"SlideShowHeader"], After]
		}],
	(* ELSE *)
		FrontEndExecute[{
			FrontEnd`NotebookWrite[nbObj,
				Cell["", "SlideShowNavigationBar", CellTags->"SlideShowHeader"], After]
		}]
	]
];
InsertSlideBreakCell[a__] := Message[InsertSlideBreakCell::argx, {a}];
InsertSlideBreakCell::argx = "Argument should be a NotebookObject: `1`";
InsertSlideBreakCell::info = "`1`";




(**************************************

 SplitCellAtCursor
	Split cells

	in: None
	out: None

***************************************)

SplitCellAtCursor[]:= SplitCellAtCursor[InputNotebook[]];
SplitCellAtCursor[nbObj_NotebookObject]:=
Module[{},
	FrontEndExecute[FrontEndToken[nbObj, "CellSplit"]]
];
SplitCellAtCursor[a__] := Message[SplitCellAtCursor::argx, {a}];
SplitCellAtCursor::argx = "Argument should be a NotebookObject: `1`";
SplitCellAtCursor::info = "`1`";





(*******************************************************

 RemoveSlideBreak

	To 'remove slide break' is to set PageBreakAbove -> False to Cell(s)

	in: NotebookObject
	out: None

*******************************************************)

RemoveSlideBreak[nbObj_NotebookObject, str_String:"Single"]:=
Module[{cells, counters},
	cells = Cells@NotebookSelection[nbObj];

	CurrentValue[cells, PageBreakAbove] = False;

	(
		(* adjust CounterIncrements if the cell style(s) include "SlideShowNavigationBar" *)
		counters = CurrentValue[#, CounterIncrements];
		If[TrueQ[Length[counters] > 0],
			counters = DeleteCases[counters, "SlideShowNavigationBar", Infinity];
			CurrentValue[#, CounterIncrements] = counters;
		]
	) & /@ cells
];

(*
	Not used
*)
RemoveSlideBreak[nbObj_NotebookObject, "All"]:=
Module[{style},

	style = cellInfo["Style"];
	Message[RemoveSlideBreak::info, "Remove slide breaks above " <> ToString[style]];
(* TODO Add 'All' option to RemoveSlideBreaks *)
	CurrentValue[NotebookSelection[nbObj], PageBreakAbove] = False
];

RemoveSlideBreak::argx = "`1`";
RemoveSlideBreak::info = "`1`";





(**************************************

 SetLockScaling

	in: None
	out: expression

***************************************)
(* TODO: SetLockScaling needs test for already True/False *)

(* Remove ImageSize option *)
SetLockScaling[nbObj_NotebookObject, False]:=
Module[{origCell, newCell, nbsel, graphicsWidth, windowWidth},

	If[MemberQ[{GraphicsBox, Graphics3DBox}, Head[CurrentValue[nbObj, "SelectionData"]] ], (* FIXME: Ian? *)

(* lou *)
		nbsel = NotebookSelection[nbObj];
		graphicsWidth = AbsoluteCurrentValue[nbsel, {GraphicsBoxOptions, ImageSize, 1}];
		windowWidth = AbsoluteCurrentValue[nbObj, {WindowSize, 1}];
		With[{percent = graphicsWidth / windowWidth},
			If[NumericQ[percent],
				(* 2018-06-27 andy *)
				(* adding Dynamic 2nd arg of None in order to disable graphic resizing.
				   http://bugs.wolfram.com/show?number=353888
				   "Resizing image after selecting 'Scale with Screen Size' produces 'Tag Times is Protected' errors"
				   *)
				CurrentValue[nbsel, {GraphicsBoxOptions, ImageSize}] =
					Dynamic[percent * AbsoluteCurrentValue[{WindowSize, 1}], None];
					SelectCellBracket[nbObj];
					origCell = NotebookRead[nbObj];
					newCell = PresenterTools`Styles`RemoveCellStyle[origCell, "ImageSizeLock"];
					NotebookWrite[nbObj, newCell, All],

			(* else -- bug 337779 *)
				Beep[];
				$Failed
			]
		],

	(* ELSE *)
		Message[SetLockScaling::info, "Graphic not selected"]
	]
];

(* Force specific ImageSize option *)
SetLockScaling[nbObj_NotebookObject, True]:=
Module[{origCell, newCell, nbsel = NotebookSelection[InputNotebook[]], graphicsWidth},

	If[MemberQ[{GraphicsBox, Graphics3DBox}, Head[CurrentValue[nbObj, "SelectionData"]] ], (* FIXME: Ian? *)

(* lou *)
		graphicsWidth = AbsoluteCurrentValue[nbsel, {GraphicsBoxOptions, ImageSize, 1}];
		If[NumericQ[graphicsWidth],
			CurrentValue[nbsel, {GraphicsBoxOptions, ImageSize}] = graphicsWidth;
			SelectCellBracket[nbObj];
			origCell = NotebookRead[nbObj];
			newCell = PresenterTools`Styles`AddCellStyle[origCell, "ImageSizeLock"];
			NotebookWrite[nbObj, newCell, All],
		(* else -- 337779 *)
			Message[SetLockScaling::info, "Graphic width needs to be numeric, not "<> ToString[graphicsWidth] ];
			$Failed
		],

	(* ELSE *)
		Message[SetLockScaling::info, "Graphic not selected"]
	]
];
SetLockScaling::info = "`1`";




(*******************************************************

 GetImageDimensions

	in: xxx
	out: yyy

*******************************************************)

GetImageDimensions[val_, graphic_]:=
Module[{img = graphic},

	img = img /. {___, t : _[_, _BoxForm`ImageTag, ___], ___} :> t;
	img = ToExpression[img];

	ImageDimensions[img]
]
GetImageDimensions::argx = "`1`";
GetImageDimensions::info = "`1`";






(*******************************************************

 SetCellLevelTaggingRule

	in: NotebookObject, tag name and value
	out: None

*******************************************************)

SetCellLevelTaggingRule[nbObj_NotebookObject, name_ -> value_] :=
Module[{tags = Cases[NotebookRead[nbObj], t:Rule[TaggingRules, _] :> t]},

	If[MemberQ[{{}, Inherited}, tags],
		SetOptions[NotebookSelection[nbObj], TaggingRules -> {name -> value}],
	(* ELSE *)
		If[CurrentValue[NotebookSelection[nbObj], {TaggingRules, name}] === value,
			SetOptions[NotebookSelection[nbObj],
				TaggingRules -> DeleteCases[Part[tags, 1, 2], Rule[name, _]]],
		(* ELSE *)
			SetOptions[NotebookSelection[nbObj],
				TaggingRules -> Append[DeleteCases[Part[tags, 1, 2], Rule[name, _]], name -> value]]
		]
	]
];

SetCellLevelTaggingRule[Cell[con__, (Rule|RuleDelayed)[TaggingRules, t_], other___], name_ -> value_]:=
Module[{},
	Cell[con, other, TaggingRules -> Flatten[{DeleteCases[t, Rule[name, _]], name -> value}, 1]]
]
SetCellLevelTaggingRule[Cell[con__, other___], name_ -> value_]:=
Module[{},
	Cell[con, other, TaggingRules -> {name -> value}]
]

SetCellLevelTaggingRule[a___]:= Message[SetCellLevelTaggingRule::argx, {a}];
SetCellLevelTaggingRule::argx = "`1`";
SetCellLevelTaggingRule::info = "`1`";








(*******************************************************

 GetUserModifications

	in: NotebookObject
	out: List of user's style option modifications to (possibly) make to stylesheet

*******************************************************)

GetUserModifications[]:= GetUserModifications[InputNotebook[]];
GetUserModifications[nbObj_NotebookObject]:=
Module[{list, result},
	Logger["GetUserModifications:"];
	Logger[nbObj];
	list = CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "UserModifications"}, {}];

	Which[
		Head[list] === List,
			result = list,
		True,
			Message[GetUserModifications::info, list];
			result = {};
	];

	result
];
GetUserModifications[a___]:= Message[GetUserModifications::argx, {a}];
GetUserModifications::argx = "`1`";
GetUserModifications::info = "`1`";




(*******************************************************

 UserModificationCellStyles

	in: xxx
	out: yyy

*******************************************************)

UserModificationCellStyles[nbObj_]:=
Module[{},

	CurrentValue[nbObj,
		{TaggingRules, "PresenterSettings", "UserModifications"}, {}] /. {sty_String, r__?OptionQ} :> sty
];
UserModificationCellStyles[a___]:= Message[UserModificationCellStyles::argx, {a}];
UserModificationCellStyles::argx = "`1`";
UserModificationCellStyles::info = "`1`";







(*******************************************************

 SetStyleUpdateList

	in: nb object and list of user's style option modifications
	out: None

*******************************************************)

SetStyleUpdateList[nbObj_NotebookObject, list_List]:=
Module[{},
	Logger["SetStyleUpdateList:"];
	Logger[nbObj];
	Logger[list];
	(* add user's style modifications to notebook tagging rules *)
	CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "UserModifications"}] = list;

	(* set notebook tagging rule to whether the list is empty (possible speed update) *)
	(* set asset colors/etc. in toolbars *)
(*
	If[Length[list] > 0,
		SetNotebookContainsStyleUpdates[nbObj, True],
	(* ELSE *)
		SetNotebookContainsStyleUpdates[nbObj, False];
	];
*)
]
SetStyleUpdateList[a___]:= (Message[SetStyleUpdateList::argx, {a}]; $Failed);

SetStyleUpdateList::argx = "`1`";
SetStyleUpdateList::info = "`1`";




(*******************************************************

 DeleteStyleOptionsUpdate

	in: Notebook object, list of user modifications, Style to be modified
	out: revised list of modifications

*******************************************************)

DeleteStyleOptionsUpdate[nbObj_NotebookObject, list_List, sty_, options_List]:=
Module[{userOptions = list},

	(* TODO: need better check  *)
	If[MemberQ[options, FrontEnd`FontVariationsUnderline],
		userOptions = userOptions /. {sty, Rule[FontVariations, {Rule["Underline", _]}] } :> Nothing
	];
	(* remove named options from usermodification list *)
	DeleteCases[userOptions, {sty, (Rule|RuleDelayed)[Alternatives@@options, _]}, Infinity]
];
DeleteStyleOptionsUpdate[a___]:= Message[DeleteStyleOptionsUpdate::argx, {a}];
DeleteStyleOptionsUpdate::argx = "`1`";
DeleteStyleOptionsUpdate::info = "`1`";




(*******************************************************

 UserModificationsQ

	in: Notebook object
	out: True | False

*******************************************************)

UserModificationsQ[]:= UserModificationsQ[InputNotebook[]];
UserModificationsQ[nbObj_NotebookObject]:=
Module[{update},
	Logger["UserModificationsQ:"];
	Logger[nbObj];

	update = CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "UserModificationsQ"}, False];

	If[!MemberQ[{True, False}, update],
		update = TrueQ[Length[GetUserModifications[nbObj]] > 0]
	];

	update
]
UserModificationsQ[a___]:= Message[UserModificationsQ::argx, {a}];
UserModificationsQ::argx = "`1`";
UserModificationsQ::info = "`1`";



(*******************************************************

 SelectionCellStyleModifiedQ

	in: xxx
	out: yyy

*******************************************************)

SelectionCellStyleModifiedQ[nbObj_NotebookObject]:=
Module[{cell, cellstyle, changes, ret},
 cell = Cells[NotebookSelection[nbObj]];
If[!MatchQ[cell, {_CellObject}], Return[False]];
  cell = First[cell];
  cellstyle = CurrentValue[cell, "CellStyleName"];
  If[!StringQ[cellstyle], Return[False]];
   changes =
    With[{c = cell, sty = cellstyle},
     FE`Evaluate[
      FEPrivate`AbsoluteCellStyleComplement[c,
       FEPrivate`AbsoluteResolveCellStyle[c, sty]]]];
   ret = IntersectingQ[
    Keys[changes], PresenterTools`Private`$ToolbarSupportedOptions];
	ret

]
SelectionCellStyleModifiedQ[a___]:= Message[SelectionCellStyleModifiedQ::argx, {a}];
SelectionCellStyleModifiedQ::argx = "`1`";
SelectionCellStyleModifiedQ::info = "`1`";



(*******************************************************

 DefaultOptionValueQ

	Is the given option value within NotebookSelection the
	default value or does it differ from the current stylesheet

	in: NotebookObject, cellstyle, cell option
	out: True | False

*******************************************************)

DefaultOptionValueQ[nbObj_, cellStyle_, option_] :=
Module[{},
(*
	With[{cellObj = NotebookSelection[nbObj], style = cellStyle},
		FE`Evaluate[FEPrivate`CellStyleComplement[cellObj, FEPrivate`ResolveCellStyle[cellObj, style]]]
*)
	SameQ[AbsoluteCurrentValue[NotebookSelection[nbObj], option],
		Setting @ AbsoluteCurrentValue[nbObj, {StyleDefinitions, cellStyle, option}]
	]
]
DefaultOptionValueQ[a___]:= Message[DefaultOptionValueQ::argx, {a}];
DefaultOptionValueQ::argx = "`1`";
DefaultOptionValueQ::info = "`1`";



(*******************************************************

 SetNotebookContainsStyleUpdates

 Set if notebook contains user style modifications
	in: Notebook object and True|False
	out: None

*******************************************************)

SetNotebookContainsStyleUpdates[nbObj_NotebookObject, True]:= (
	CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "UserModificationsQ"}] = True;
	PresenterTools`$ButtonColorUpdateStylesheet = GrayLevel[1]; (*RGBColor[0.782467, 0.158007, 0.178073];*)
	PresenterTools`$FrameColorUpdateStylesheet = GrayLevel[0.7]; (*RGBColor[0.782467, 0.158007, 0.178073];*)
)
SetNotebookContainsStyleUpdates[nbObj_NotebookObject, False]:= (
	CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "UserModificationsQ"}] = False;
	PresenterTools`$ButtonColorUpdateStylesheet = GrayLevel[0.95];
	PresenterTools`$FrameColorUpdateStylesheet = GrayLevel[0.95];
)
SetNotebookContainsStyleUpdates[a___]:= Message[SetNotebookContainsStyleUpdates::argx, {a}];
SetNotebookContainsStyleUpdates::argx = "`1`";
SetNotebookContainsStyleUpdates::info = "`1`";







(*******************************************************

 GetLiteralizedNotebook

	in: xxx
	out: yyy

*******************************************************)

GetLiteralizedNotebook[nbObj_, windowSize_:{800,600}]:=
Module[{expr, styleHintList, styleSheet, nbExpr, content, options},

(*	windowSize = AbsoluteCurrentValue[nbObj, WindowSize];*)
	expr = PresenterTools`Styles`GetStylesheet[nbObj];
	styleHintList = Union[Cases[expr, Dynamic[CurrentValue[{System`StyleHints, cat_String, name_String}]] :>
								{cat, name, Dynamic[CurrentValue[nbObj, {System`StyleHints, cat, name}]]}, Infinity]];
	styleSheet = Fold[replaceStyleHints, expr, styleHintList];
	styleSheet = replaceOptionValues[styleSheet, "WindowSize", windowSize];
	styleSheet = replaceOptionValues[styleSheet, "DockedCells", FEPrivate`FrontEndResource["FEExpressions", "SlideshowToolbar"]];

	nbExpr = NotebookGet[nbObj];
	content = First[List@@nbExpr];
	options = Rest[List@@nbExpr];

	With[{con = content, opts = options},
		Notebook[con, StyleDefinitions -> styleSheet, ScreenStyleEnvironment -> "Slideshow Presentation", opts] ]

];
(*
GetLiteralizedNotebook[a___]:= Message[GetLiteralizedNotebook::argx, {a}];
GetLiteralizedNotebook::argx = "`1`";
GetLiteralizedNotebook::info = "`1`";
*)



replaceStyleHints[nbexpr_, {cat_, name_, value_}] :=
Module[{pos, expr = nbexpr},
  pos = Position[expr, Dynamic[CurrentValue[{System`StyleHints, cat, name}]], Infinity];
  expr = ReplacePart[expr, pos -> Setting[value]];
  pos = Position[expr, CurrentValue[{System`StyleHints, cat, name}], Infinity];
  expr = ReplacePart[expr, pos -> Setting[value]]
  ];

replaceOptionValues[nbexpr_, "WindowSize", {height_, width_}] :=
Module[{pos, expr = nbexpr},
  pos = Position[expr, FrontEnd`AbsoluteCurrentValue[{WindowSize, 1}],
     Infinity];
  expr = ReplacePart[expr, pos -> height];
  pos = Position[expr, FrontEnd`AbsoluteCurrentValue[{WindowSize, 2}],
     Infinity];
  expr = ReplacePart[expr, pos -> width]
  ];

replaceOptionValues[nbexpr_, "DockedCells", cells_] :=
Module[{pos, expr = nbexpr, cellsPos, screenEnvironment},
  pos = Position[expr,
    Cell[StyleData[All, "Slideshow Presentation", ___], ___],
    Infinity];
  screenEnvironment = First@Extract[expr, pos];
  cellsPos =
   Position[screenEnvironment, Rule[System`DockedCells, _],
    Infinity];
  screenEnvironment = ReplacePart[screenEnvironment, pos -> cells];
  ReplacePart[expr, pos -> screenEnvironment]
  ];











(**************************************

 SetPresenterControlsDialog
	Dialog for user to choose which slideshow keyboard shortcuts to apply to the notebook.
	in: None
	out: Association

***************************************)

SetPresenterControlsDialog[]:= SetPresenterControlsDialog[InputNotebook[]];
SetPresenterControlsDialog[nbObj_, opts___?OptionQ]:=
Module[{shortcuts, currentShortcuts, possibleShortcuts, shortcutMenuNames, defaultShortcuts, shortcutsAssoc},
(*
DynamicModule[{keyControls, (*firstSlide, previousSlide, nextSlide, lastSlide, exitPresentation,*)
	currentShortcuts, possibleShortcuts, shortcutMenuNames, defaultShortcuts, shortcutsAssoc},
*)
	(* list of possible shortcuts *)
	possibleShortcuts = GetPossibleShortcutAssociation[];

	(* default Mathematica settings *)
	shortcuts =
	defaultShortcuts =
		<|
			"FirstSlide" -> "Home", (*"HomeKeyDown",*)
			"PreviousSlide" -> "Left Arrow", (*"Page Up", "PageUpKeyDown",*)
			"NextSlide" -> "Right Arrow", (*"Page Down", "PageDownKeyDown",*)
			"LastSlide" -> "End", (*"EndKeyDown",*)
			"ExitPresentation" -> "Not Assigned"
		|>;

	(* Get user assigned slide keyboard shortcuts, if any *)
	(* e.g.
		"FirstSlide" -> "HomeKeyDown",
		"PreviousSlide" -> "PageUpKeyDown",
		"NextSlide" -> "PageDownKeyDown",
		"LastSlide" -> "EndKeyDown",
		"ExitPresentation" -> "Not Assigned"
	*)
	currentShortcuts = GetCurrentShortcuts[nbObj];

	If[currentShortcuts =!= {},
		(shortcuts[#] = GetKeyNameFromKeyboardShortcut@currentShortcuts[#]) & /@ Keys[currentShortcuts]
	];

	With[{
			keys = Keys[possibleShortcuts],
			keyCommands= possibleShortcuts,
			next = shortcuts["NextSlide"],
			prev = shortcuts["PreviousSlide"],
			first = shortcuts["FirstSlide"],
			last = shortcuts["LastSlide"],
			exit = shortcuts["ExitPresentation"],
			default = defaultShortcuts,
			fm = {{10, 10}, {2, 2}},
			no = nbObj
		},

(*		keyControls =*)
			CreateDialog[
			DynamicModule[{
				firstSlide = first,
				previousSlide = prev,
				nextSlide = next,
				lastSlide = last,
				exitPresentation = exit
				},

				Grid[{
					{
						Pane[
							Style[Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "SpecifyKeysCommandsLabel"], FontSize->12, FontFamily :> CurrentValue["PanelFontFamily"]],
							ImageMargins -> {{10, 20}, {20, 10}}
						],
						SpanFromLeft
					},
					(* ROW ONE *)
					{
						Panel[Grid[{{
							Style[Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "FirstSlideMenuItem"], 12],
								PopupMenu[
									Dynamic[firstSlide],
									keys,
									"",
									Framed[
										Pane[
											Grid[{{
												Style[
													Dynamic[firstSlide],
													12, FontFamily :> CurrentValue["PanelFontFamily"]
												],
												Item[
													Style["\[FilledDownTriangle]", FontColor -> GrayLevel[0.6], FontSize -> 12, FontWeight -> "Plain"],
													Alignment -> {Right, Baseline}
													]
											}}, Alignment -> {Left, Baseline}, Spacings -> {0.5, 0}, ItemSize -> {{Scaled[0.75], Scaled[0.2]}}],
											{127, 16}, Alignment -> {Left, Center}],
										FrameMargins -> {{10, 0}, {2, 0}}, FrameStyle -> GrayLevel[.75], Background -> GrayLevel[1], RoundingRadius -> 5
									], Appearance -> None
								]
							}, {
							Style[Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "PreviousSlideMenuItem"], 12],
								PopupMenu[
									Dynamic[previousSlide],
									keys,
									"",
									Framed[
										Pane[
											Grid[{{
												Style[
													Dynamic[previousSlide],
													12, FontFamily :> CurrentValue["PanelFontFamily"]
												],
												Item[
													Style["\[FilledDownTriangle]", FontColor -> GrayLevel[0.6], FontSize -> 12, FontWeight -> "Plain"],
													Alignment -> {Right, Baseline}
													]
											}}, Alignment -> {Left, Baseline}, Spacings -> {0.5, 0}, ItemSize -> {{Scaled[0.75], Scaled[0.2]}}],
											{127, 16}, Alignment -> {Left, Center}],
										FrameMargins -> {{10, 0}, {2, 0}}, FrameStyle -> GrayLevel[.75], Background -> GrayLevel[1], RoundingRadius -> 5
									], Appearance -> None
								]
							}, {
							Style[Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "NextSlideMenuItem"], 12],
								PopupMenu[
									Dynamic[nextSlide],
									keys,
									"",
									Framed[
										Pane[
											Grid[{{
												Style[
													Dynamic[nextSlide],
													12, FontFamily :> CurrentValue["PanelFontFamily"]
												],
												Item[
													Style["\[FilledDownTriangle]", FontColor -> GrayLevel[0.6], FontSize -> 12, FontWeight -> "Plain"],
													Alignment -> {Right, Baseline}
													]
											}}, Alignment -> {Left, Baseline}, Spacings -> {0.5, 0}, ItemSize -> {{Scaled[0.75], Scaled[0.2]}}],
											{127, 16}, Alignment -> {Left, Center}],
										FrameMargins -> {{10, 0}, {2, 0}}, FrameStyle -> GrayLevel[.75], Background -> GrayLevel[1], RoundingRadius -> 5
									], Appearance -> None
								]
							}, {
							Style[Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "LastSlideMenuItem"], 12],
								PopupMenu[
									Dynamic[lastSlide],
									keys,
									"",
									Framed[
										Pane[
											Grid[{{
												Style[
													Dynamic[lastSlide],
													12, FontFamily :> CurrentValue["PanelFontFamily"]
												],
												Item[
													Style["\[FilledDownTriangle]", FontColor -> GrayLevel[0.6], FontSize -> 12, FontWeight -> "Plain"],
													Alignment -> {Right, Baseline}
													]
											}}, Alignment -> {Left, Baseline}, Spacings -> {0.5, 0}, ItemSize -> {{Scaled[0.75], Scaled[0.2]}}],
											{127, 16}, Alignment -> {Left, Center}],
										FrameMargins -> {{10, 0}, {2, 0}}, FrameStyle -> GrayLevel[.75], Background -> GrayLevel[1], RoundingRadius -> 5
									], Appearance -> None
								]
							}, {
							Style[Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "ExitPresentationLabel"], 12],
								PopupMenu[
									Dynamic[exitPresentation],
									keys,
									"",
									Framed[
										Pane[
											Grid[{{
												Style[
													Dynamic[exitPresentation],
													12, FontFamily :> CurrentValue["PanelFontFamily"]
												],
												Item[
													Style["\[FilledDownTriangle]", FontColor -> GrayLevel[0.6], FontSize -> 12, FontWeight -> "Plain"],
													Alignment -> {Right, Baseline}
													]
											}}, Alignment -> {Left, Baseline}, Spacings -> {0.5, 0}, ItemSize -> {{Scaled[0.75], Scaled[0.2]}}],
											{127, 16}, Alignment -> {Left, Center}
										],
										FrameMargins -> {{10, 0}, {2, 0}}, FrameStyle -> GrayLevel[.75], Background -> GrayLevel[1], RoundingRadius -> 5
									], Appearance -> None
								]
							}
						}, Alignment->{{Right, Left}}, Spacings -> {1, 1}], ImageMargins -> {{30, 30}, {20, 0}},
							Background -> GrayLevel[0.95], ImageSize -> Full
						],
						SpanFromLeft
					},
					(* ROW TWO *)
					{
(*
						Item[
							Grid[{{
*)
								Item[
									Button[
										Style[
											Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "UseDefaultKeysButtonText"]],
										(
											nextSlide = default["NextSlide"];
											previousSlide = default["PreviousSlide"];
											firstSlide = default["FirstSlide"];
											lastSlide = default["LastSlide"];
											exitPresentation = default["ExitPresentation"];
										), ImageSize -> Dynamic[CurrentValue["DefaultButtonSize"]]],
									Alignment -> Left
								],
								Item[
									ChoiceButtons[{
											Style[
												Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "OKButtonText"],
													FontColor :> FEPrivate`If[FEPrivate`$OperatingSystem === "MacOSX", GrayLevel[1], Automatic]],
												Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "CancelButtonText"]
										},
										{
											With[{
												ns = keyCommands[nextSlide],
												ps = keyCommands[previousSlide],
												fs = keyCommands[firstSlide],
												ls = keyCommands[lastSlide],
												ep = keyCommands[exitPresentation]
												},

											(*CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "PresenterControlsPalette", "PaletteNumber"}] = 0;*)
											CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "Dialogs", "PresenterControlsPalette"}] = 0;
											DialogReturn[
												(* Set keyboard shortcuts in embedded stylesheet *)
												SetKeyboardShortcuts[nbObj,
													<|
														"NextSlide" -> ns,
														"PreviousSlide"-> ps,
														"FirstSlide" -> fs,
														"LastSlide" -> ls,
														"ExitPresentation" -> ep
													|>
												]
											]
											],
											(
											(*CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "PresenterControlsPalette", "PaletteNumber"}] = 0;*)
											CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "Dialogs", "PresenterControlsPalette"}] = 0;
											DialogReturn[$Canceled]
											)
										}
									],
									Alignment -> Right
								]
(*
							}}, Alignment -> {Left, Center}, Spacings -> {0.5, 0}], ImageMargins -> {{5, 5}, {0, 0}},
								Background -> GrayLevel[0.95]
							]
*)
					},
					(* ROW THREE *)
					{
						" ", SpanFromLeft
					}
					}, ImageSize -> Full, Alignment -> {Center, {Top}}, (*Background -> GrayLevel[0.9], *) Spacings -> {1, 0},
						Frame -> None
				],
			(*	], *)
				Initialization :> (
					SetSelectedNotebook[EvaluationNotebook[]]
				)
			],
				Background -> GrayLevel[0.95],
				FrameMargins -> {{10, 10}, {10, 10}},
				Modal -> False,
				NotebookEventActions -> {
					"ReturnKeyDown" :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
					"EscapeKeyDown" :> FE`Evaluate[FEPrivate`FindAndClickCancelButton[]],
					"WindowClose" :> (
								FE`Evaluate[FEPrivate`FindAndClickCancelButton[]];
								(*CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "PresenterControlsPalette", "PaletteNumber"}] = 0;*)
								CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "Dialogs", "PresenterControlsPalette"}] = 0
								)
					}
(*
					,
				NotebookDynamicExpression :> (
					If[Not[MemberQ[Notebooks[], nbObj]], NotebookClose[EvaluationNotebook[]]]
				)
*)
			]
		]

];
SetPresenterControlsDialog::info = "`1`";




(*******************************************************

 ApplySetPresenterControlsDialogResults

 Modify stylesheet to apply the result from SetPresenterControlsDialog

	in: notebook, Association of which cell styles are added/removed
	out: apply new stylesheet to notebook

*******************************************************)

ApplySetPresenterControlsDialogResults[nbObj_]:=
Module[{},

	foo
];
(*
ApplySetPresenterControlsDialogResults[a___]:= Message[ApplySetPresenterControlsDialogResults::argx, {a}];
ApplySetPresenterControlsDialogResults::argx = "`1`";
ApplySetPresenterControlsDialogResults::info = "`1`";
*)





(*******************************************************

 GetCurrentShortcuts

	in: Notebook (not stylesheet)
	out: Any NotebookEventActions slide shortcuts

*******************************************************)

GetCurrentShortcuts[nbObj_]:=
Module[{currentStylesheet, pos, res = {}, styleEnvCell},

	currentStylesheet = PresenterTools`Styles`GetStylesheet[nbObj];

	If[Head[currentStylesheet] === Notebook,
		pos = Position[currentStylesheet, Cell[StyleData["Notebook", "Slideshow Presentation"], ___]];

		If[pos =!= {},
			styleEnvCell = Extract[currentStylesheet, pos];

			If[!FreeQ[styleEnvCell, NotebookEventActions],
				(* found existings events *)
				res =
					Association@@
					Flatten[{
						Cases[styleEnvCell, RuleDelayed[k_, expr_ /; Not@FreeQ[Hold[expr], "ScrollPageFirst"]] :> Rule["FirstSlide", k], Infinity],
						Cases[styleEnvCell, RuleDelayed[k_, expr_ /; Not@FreeQ[Hold[expr], "ScrollPagePrevious"]] :> Rule["PreviousSlide", k], Infinity],
						Cases[styleEnvCell, RuleDelayed[k_, expr_ /; Not@FreeQ[Hold[expr], "ScrollPageNext"]] :> Rule["NextSlide", k], Infinity],
						Cases[styleEnvCell, RuleDelayed[k_, expr_ /; Not@FreeQ[Hold[expr],"ScrollPageLast"]] :> Rule["LastSlide", k], Infinity],
						Cases[styleEnvCell, RuleDelayed[k_, expr_ /; Not@FreeQ[Hold[expr], PresenterTools`EndPresentation]] :> Rule["ExitPresentation", k], Infinity, Heads -> True]
					}],
			(* ELSE *)
				(* return empty list *)
				res = {}
			]
		]
	];

	res
];
GetCurrentShortcuts[a___]:= Message[GetCurrentShortcuts::argx, {a}];
GetCurrentShortcuts::argx = "`1`";
GetCurrentShortcuts::info = "`1`";







(*******************************************************

 SetKeyboardShortcuts

	in: xxx
	out: yyy

*******************************************************)

SetKeyboardShortcuts[nbObj_NotebookObject, keyControls_Association]:=
Module[{currentStylesheet, newStylesheet, pos, environmentExpr},

	currentStylesheet = CurrentValue[nbObj, StyleDefinitions];

	With[{
		ns = keyControls["NextSlide"],
		fs = keyControls["FirstSlide"],
		ps = keyControls["PreviousSlide"],
		ls = keyControls["LastSlide"],
		exit = keyControls["ExitPresentation"]
		},

	If[Head[currentStylesheet] === Notebook,

		(* check for environment "Slideshow Presentation *)
		pos = Position[currentStylesheet, Cell[StyleData["Notebook", "Slideshow Presentation"], ___]];

		If[pos === {},
			(* Embedded stylesheet is a notebook, but no screenstyle environment *)
			newStylesheet =
				AddCellsToEndOfNotebook[currentStylesheet, {
					Cell[StyleData["Notebook", "Slideshow Presentation"],
						NotebookEventActions -> {
							System`ParentList,
							fs :> FrontEndTokenExecute[nbObj, "ScrollPageFirst"],
							ps :> FrontEndTokenExecute[nbObj, "ScrollPagePrevious"],
							ns :> FrontEndTokenExecute[nbObj, "ScrollPageNext"],
							ls :> FrontEndTokenExecute[nbObj, "ScrollPageLast"],
							exit :> PresenterTools`EndPresentation[nbObj, "KeyboardShortcut"]
						}
					]
				}],
		(* ELSE *)
			(* Embedded stylesheet is a notebook, and has a screenstyle environment *)
			environmentExpr = Extract[currentStylesheet, First@pos];

				environmentExpr =
					ResetOptions[environmentExpr,
							NotebookEventActions -> {
								System`ParentList,
								fs :> FrontEndTokenExecute[nbObj, "ScrollPageFirst"],
								ps :> FrontEndTokenExecute[nbObj, "ScrollPagePrevious"],
								ns :> FrontEndTokenExecute[nbObj, "ScrollPageNext"],
								ls :> FrontEndTokenExecute[nbObj, "ScrollPageLast"],
								exit :> PresenterTools`EndPresentation[nbObj, "KeyboardShortcut"]
							}];
				newStylesheet = ReplacePart[currentStylesheet, pos -> environmentExpr];
		],
	(* ELSE *)
		(* No embeddeed stylesheet *)

		newStylesheet =
			PresenterTools`Styles`CreatePrivateStylesheet[{
				Cell[StyleData[StyleDefinitions -> currentStylesheet]],
				Cell[StyleData["Notebook", "Slideshow Presentation"],
					NotebookEventActions -> {
							System`ParentList,
							fs :> FrontEndTokenExecute[nbObj, "ScrollPageFirst"],
							ps :> FrontEndTokenExecute[nbObj, "ScrollPagePrevious"],
							ns :> FrontEndTokenExecute[nbObj, "ScrollPageNext"],
							ls :> FrontEndTokenExecute[nbObj, "ScrollPageLast"],
							exit :> PresenterTools`EndPresentation[nbObj, "KeyboardShortcut"]
					}
				]
			}]
		]
	];

	SetOptions[nbObj, StyleDefinitions -> newStylesheet]
];
SetKeyboardShortcuts::argx = "`1`";
SetKeyboardShortcuts::info = "`1`";





(*******************************************************

 GetPossibleShortcutAssociation

	in: None
	out: List of possible keyboard shortcuts
		"DownArrowKeyDown" :> "Down Arrow",
		{"KeyDown", " "} :> "Space",


*******************************************************)

GetPossibleShortcutAssociation[opts___?OptionQ]:=
Module[{},

	(* list from blee *)
(*
	{
		"RightArrowKeyDown" :> "Right Arrow",
		"LeftArrowKeyDown" :> "Left Arrow",
		"UpArrowKeyDown" :> "Up Arrow",
		"DownArrowKeyDown" :> "Down Arrow",
		{"KeyDown", " "} :> "Space",
		"ReturnKeyDown" :> "Enter",
		"PageUpKeyDown" :> "Page Up",
		"PageDownKeyDown" :> "Page Down",
		"HomeKeyDown" :> "Home",
		"EndKeyDown" :> "End",
		{"KeyDown", "\t"} :> "Tab",
		"EscapeKeyDown" :> "Esc",
		{"KeyDown", "["} :> "[",
		{"KeyDown", "]"} :> "]",
		{"KeyDown", "-"} :> "-",
		{"KeyDown", "="} :> "=",
		{"KeyDown", "."} :> ".",
		{"KeyDown", ","} :> ",",
		{"KeyDown", "/"} :> "/",
		Delimiter,
		"NotAssigned" :> "NotAssigned"
	}
*)
	<|
		"Right Arrow" -> "RightArrowKeyDown",
		"Left Arrow" -> "LeftArrowKeyDown",
		"Up Arrow" -> "UpArrowKeyDown",
		"Down Arrow" -> "DownArrowKeyDown",
		"Space" -> {"KeyDown", " "},
		"Enter" -> "ReturnKeyDown",
		"Page Up" -> "PageUpKeyDown",
		"Page Down" -> "PageDownKeyDown",
		"Home" -> "HomeKeyDown",
		"End" -> "EndKeyDown",
		"Tab" -> {"KeyDown", "\t"},
		"Esc" -> "EscapeKeyDown",
		"[" -> {"KeyDown", "["},
		"]" -> {"KeyDown", "]"},
		"-" -> {"KeyDown", "-"},
		"=" -> {"KeyDown", "="},
		"." -> {"KeyDown", "."},
		"," -> {"KeyDown", ","},
		"/" -> {"KeyDown", "/"},
		"Not Assigned" -> "NotAssigned"
	|>
];
GetPossibleShortcutAssociation[a___]:= Message[GetPossibleShortcutAssociation::argx, {a}];
GetPossibleShortcutAssociation::argx = "`1`";
GetPossibleShortcutAssociation::info = "`1`";




GetKeyboardShortcutFromKeyName[str_]:=
Module[{keys},

keys = <|
		"Right Arrow" -> "RightArrowKeyDown",
		"Left Arrow" -> "LeftArrowKeyDown",
		"Up Arrow" -> "UpArrowKeyDown",
		"Down Arrow" -> "DownArrowKeyDown",
		"Space" -> {"KeyDown", " "},
		"Enter" -> "ReturnKeyDown",
		"Page Up" -> "PageUpKeyDown",
		"Page Down" -> "PageDownKeyDown",
		"Home" -> "HomeKeyDown",
		"End" -> "EndKeyDown",
		"Tab" -> {"KeyDown", "\t"},
		"Esc" -> "EscapeKeyDown",
		"[" -> {"KeyDown", "["},
		"]" -> {"KeyDown", "]"},
		"-" -> {"KeyDown", "-"},
		"=" -> {"KeyDown", "="},
		"." -> {"KeyDown", "."},
		"," -> {"KeyDown", ","},
		"/" -> {"KeyDown", "/"},
		"Not Assigned" -> "NotAssigned"
	|>;

	keys[str]
]




GetKeyNameFromKeyboardShortcut[shortcut_]:=
Module[{keys},

keys = <|
		"RightArrowKeyDown" -> "Right Arrow",
		"LeftArrowKeyDown" -> "Left Arrow",
		"UpArrowKeyDown" -> "Up Arrow",
		"DownArrowKeyDown" -> "Down Arrow",
		{"KeyDown", " "} -> "Space",
		"ReturnKeyDown" -> "Enter",
		"PageUpKeyDown" -> "Page Up",
		"PageDownKeyDown" -> "Page Down",
		"HomeKeyDown" -> "Home",
		"EndKeyDown" -> "End",
		{"KeyDown", "\t"} -> "Tab",
		"EscapeKeyDown" -> "Esc",
		{"KeyDown", "["} -> "[",
		{"KeyDown", "]"} -> "]",
		{"KeyDown", "-"} -> "-",
		{"KeyDown", "="} -> "=",
		{"KeyDown", "."} -> ".",
		{"KeyDown", ","} -> ",",
		{"KeyDown", "/"} -> "/",
		"NotAssigned" -> "Not Assigned"
	|>;

	keys[shortcut]
]







(**************************************

 DeleteSlideButton
	Button to delete current slide
	in: None
	out: expression

***************************************)

DeleteSlideButton[]:=
Module[{},
	With[{icon = bitmapResource["DeleteSlide"]},
  MouseAppearance[
   Button[Grid[{{icon,
	   Style["Delete Slide", 12, FontFamily :> CurrentValue["PanelFontFamily"]]}},
	 Alignment -> {Left, Center}, Spacings -> {0.5, 0}],
		 (
			If[CurrentValue[EvaluationNotebook[], ScreenStyleEnvironment] === "SlideShow",
				SelectionMove[EvaluationNotebook[], All, Notebook];
				SetOptions[NotebookSelection[EvaluationNotebook[]], Editable -> True, Deletable -> True];
				NotebookDelete[EvaluationNotebook[]];
			,
				Beep[]
			]
		),
	Appearance -> None], "LinkHand"
	]
	]
];




GetStyleOfSelectionMenu[nbObj_, style_]:=
Module[{},

	ActionMenu[
	 Pane[Grid[{{Pane[
	      Style[style, 12,
	       FontFamily :> CurrentValue["PanelFontFamily"]],
	      ImageSizeAction -> "Clip", ImageSize -> {110, 16}],
	     Item[Dynamic@
	       RawBoxes@
	        FEPrivate`ImportImage[
	         FrontEnd`FileName[{"PresenterTools"}, "TriangleDown.png"]],
	      Alignment -> {Right, Center}]}}, Alignment -> {Left, Center},
	   Spacings -> {0.5, 0},
	   ItemSize -> {{Scaled[0.75], Scaled[0.2]}}], {100, 20},
	  Alignment -> {Left, Center}],
	 PresenterTools`SetCellStyleMenu[EvaluationNotebook[]],
	 Appearance -> None]

 ]

(**************************************

 PresentNotebook
	Button to change mode to Presentation
	in: None
	out: expression

***************************************)

PresentNotebook[]:= PresentNotebook[InputNotebook[]];
PresentNotebook[nbObj_]:=
Module[{},
		(
(*			Message[PresentNotebook::mesg, ToString[AbsoluteCurrentValue[nbObj, WindowSize]]]; *)
		Set[CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "WindowSize"}], AbsoluteCurrentValue[nbObj, WindowSize]];
		Set[CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "WindowMargins"}], AbsoluteCurrentValue[nbObj, WindowMargins]];
		Set[CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "ShowToolbar"}], False];
(*		Set[CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "PresenterToolsSlideshowPresentationToolbar"}], True]; *)
(*		Set[CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "PresenterToolsPresentationToolbar"}], False]; *)
		Set[CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "ShowSlideBreaks"}], False];
		SetOptions[nbObj, {ScreenStyleEnvironment -> "Slideshow Presentation"}];

(*		Set[CurrentValue[nbObj, WindowFrameElements], DeleteCases[CurrentValue[nbObj, WindowFrameElements], "ResizeArea"]];*)
		PresenterTools`NotebookToggleFullScreenWrapper[nbObj]
(*		SetOptions[nbObj, {WindowSize -> {Full, Full}}] *)
		)
];



(**************************************

 AdditionalViewsMenu
	Button to change notebook to different views

***************************************)

AdditionalViewsMenu[nbObj_]:=
Module[{},
	(* TODO: Hook up screen environments *)
	ActionMenu[
		Panel[Dynamic@RawBoxes@FEPrivate`FrontEndResource["PresenterToolBitmaps", "ViewsMenu"],
			FrameMargins -> 0, ImageMargins -> 0, Background -> GrayLevel[.99],
			ImageSize -> {15, 55}, Alignment -> Center],
		{
			"Working" :> (CurrentValue[nbObj, ScreenStyleEnvironment] = "Working"),
			"Slideshow Working" :> (CurrentValue[nbObj, ScreenStyleEnvironment] = "Slideshow Working"),
			Delimiter,
			"Slideshow Presentation" :> (CurrentValue[nbObj, ScreenStyleEnvironment] = "Slideshow Presentation"),
			"Scrolling Presentation" :> (CurrentValue[nbObj, ScreenStyleEnvironment] = "Scrolling Presentation"),
			Delimiter,
			"Printout" :> (CurrentValue[nbObj, ScreenStyleEnvironment] = "Printout")
		}, Appearance -> None, AutoAction -> False
	]
];
AdditionalViewsMenu::mesg = "`1`";





UserThemesQ[]:=
	If[Total@FileNames["*.nb", FileNameJoin[{$UserBaseDirectory, "SystemFiles", "FrontEnd", "Stylesheets"}]] > 0, True, False];

GetUserThemes[]:=
Module[{list, userThemes},
	list = FileNames["*.nb", FileNameJoin[{$UserBaseDirectory, "SystemFiles", "FrontEnd", "Stylesheets"}]];
	userThemes = If[Length[list] > 0,
		Flatten[{Delimiter,
		With[{ss = FileBaseName[#]<>".nb"},
			RuleDelayed[
			FileBaseName[ss],
			ChangeNotebookStyleSheet[InputNotebook[],
				FrontEnd`FileName[{$UserBaseDirectory, "SystemFiles", "FrontEnd", "Stylesheets"}, ss]]
(*			FrontEnd`FindFileOnPath[FileBaseName[ss]<>".nb", "StyleSheetPath"] *)
			]] & /@ list}], {}]
];




(**************************************

 AutomaticSlideBreakDialog
	Dialog for setting automatic slide (page) breaks above selected cell
	in: None
	out: Dialog

***************************************)
Options[AutomaticSlideBreakDialog] = {
	"WindowTitle" -> ""
};
AutomaticSlideBreakDialog[]:= AutomaticSlideBreakDialog[InputNotebook[]];
AutomaticSlideBreakDialog[nbObj_, opts___?OptionQ]:=
Module[{slideBreakStyles, origBreaks, result, menuStyles, styleList,
	styleSheet, windowTitle},

	styleSheet = CurrentValue[nbObj, StyleDefinitions];

	(* Get Notebook's slide break values *)
	slideBreakStyles = origBreaks = CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "SlideBreakStyles"}, {}];

	(* If slidebreaks are not defined in TaggingRules, set TaggingRules *)
	If[slideBreakStyles === Inherited,
		(* Set slide break values *)
		origBreaks = {};
		slideBreakStyles = $SlideBreakStyles;
		CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "SlideBreakStyles"}] = slideBreakStyles;
	,
		slideBreakStyles = If[TrueQ[Length[slideBreakStyles] > 0], slideBreakStyles, {}];
	];

	(* style list from stylesheet menu *)
	styleList = Rest[FE`Evaluate[FEPrivate`GetPopupList["MenuListStyles"]]] /. Rule[s_, _] :> s;
(*	styleList = DeleteCases[styleList, Alternatives@@Join[slideBreakStyles, {All, "SideNote"}]];*)
	styleList = Select[styleList, FreeQ[Join[slideBreakStyles, {"SideNote", "SideCode"}], #] &, Infinity];
	(* first styles in list need to be $SlideBreakStyles *)
	menuStyles = Join[slideBreakStyles, styleList];

	With[{nbo = nbObj, initialSlideBreakStyles = slideBreakStyles, styleMenuList = menuStyles,
		styles = styleSheet},

	(* parts of dialog from pre M- 11.2 distribution, author = ? *)
		CreateDialog[
			DynamicModule[{$CellContext`x = initialSlideBreakStyles, $CellContext`i = 1,
							$CellContext`list, addStyles, removeStyles},
			Pane[
			 Grid[{{
				Style[Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "AutomaticSlideBreakDialogText"],
	   				13, FontFamily :> CurrentValue["PanelFontFamily"]],
				SpanFromLeft
				}, {
			   Grid[{{
			   		Style[Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "AutomaticSlideBreakDialogLabelCell"], 12,
			   			FontFamily :> CurrentValue["PanelFontFamily"], FontColor->GrayLevel[0.4], FontVariations -> {"CapsType" -> "AllCaps"}],
			   		Style[Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "AutomaticSlideBreakDialogLabelPreview"], 12,
			   			FontFamily :> CurrentValue["PanelFontFamily"], FontColor->GrayLevel[0.5], FontVariations -> {"CapsType" -> "AllCaps"}]
			   		}, {
					Framed[
						Pane[
							Grid[
								MapIndexed[{
									EventHandler[
										Pane[
											Row[{
												Function[{$CellContext`y},
													Checkbox[
														Dynamic[
															MemberQ[$CellContext`x, $CellContext`y],
															BoxForm`TogglerBarFunction[$CellContext`x, $CellContext`y]&
															], ImageMargins -> {{5, 5}, {0, 0}}
													]][#],
												Style[#, 12, FontFamily :> CurrentValue["PanelFontFamily"], FontColor -> GrayLevel[0.4]]
												}, StripOnInput -> True,
												Alignment -> {Left, Center}
											],
											ImageSize -> {Full, Automatic}
										], "MouseClicked" :> ($CellContext`i = First[#2]),
											PassEventsDown -> True
									] }&, $CellContext`list = styleMenuList
								],
								ItemSize -> Fit, (*Background -> GrayLevel[1],*)

								Background -> {None, {{{GrayLevel[1] (*None, GrayLevel[0.875]*)}}, Dynamic[$CellContext`i] -> GrayLevel[0.95]}},
								Alignment -> {Left, Center}
							], {230, 170},
						Scrollbars -> {False, True}, AppearanceElements -> None
						], FrameMargins -> 0, FrameStyle -> GrayLevel[0.75], ImageMargins -> 0
					],
					(*
						Right sidebar: image of selected cell in left sidebar
					*)
					Framed[
						Dynamic[
							With[{slideText = Part[$CellContext`list, $CellContext`i],
								  slideCellStyle = Part[$CellContext`list, $CellContext`i]},
								Rasterize[
									Notebook[{
										(*Cell["", "SlideShowNavigationBar"],*)
										Cell[BoxData[ToBoxes[
											Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, "slideBreakImage.png"]]]
											], CellMargins -> 0, Magnification -> 0.7],
										Cell[""],
										Cell[slideText, slideCellStyle, PageWidth -> 250],
										Cell[""]
										}, (*ScreenStyleEnvironment -> "Slideshow Working", Magnification -> 0.9,*)
										WindowSize -> {350, 300},
										StyleDefinitions -> styles
									], ImageSize -> 300
								]
							]
						], Alignment -> {Left, Top}, ImageSize -> {250, 170}, Background -> White,
						FrameMargins -> {{0, 0}, {0, 10}}, FrameStyle -> GrayLevel[0.75], ImageMargins -> 0
					]
				}}, Alignment -> {Left, Top}, Spacings -> {1, 1}
				],
				SpanFromLeft
				}, {
				Item[
					Button[
						Framed[
							Style["?", 14, FontColor -> RGBColor[0.0392157, 0.376471, 1]],
							RoundingRadius -> 11, ImageSize -> {22, 22}, Background -> Dynamic[If[Dynamic[FEPrivate`And[CurrentValue["MouseOver"], CurrentValue["NotebookSelected"]]], GrayLevel[0.95], GrayLevel[1]]], Alignment -> Center, FrameStyle -> GrayLevel[0.75]
						],
						Inherited,
						BaseStyle -> "Link", ButtonData -> "paclet:workflow/AdjustSlideBreakDefaultsInAPresentation", ButtonNote -> "paclet:workflowguide/CreatingDocumentsAndPresentations",
						Appearance -> None, Appearance -> None
					],
					Alignment -> {Left, Center}
				],
				Item[
						ChoiceButtons[{
							Style[Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "AutomaticSlideBreakDialogButtonText"],
								FontColor :> FEPrivate`If[FEPrivate`$OperatingSystem === "MacOSX", GrayLevel[1], Automatic]],
							Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "CancelButtonText"]
							}, {
							(
								(*CurrentValue[nbo, {TaggingRules, "PresenterSettings", "SlideBreakDefaults", "PaletteNumber"}] = 0;*)
								CurrentValue[nbo, {TaggingRules, "PresenterSettings", "Dialogs", "SlideBreakDefaults"}] = 0;
								CurrentValue[nbo, {TaggingRules, "PresenterSettings", "SlideBreakStyles"}] = $CellContext`x;
								removeStyles = Complement[initialSlideBreakStyles, $CellContext`x];
								addStyles = Complement[$CellContext`x, initialSlideBreakStyles];
								addStyles = If[addStyles === {}, $CellContext`x, addStyles];
								DialogReturn[
									ApplyAutomaticSlideBreakDialogResults[nbo, <| "Stylesheet" -> styles, "StylesAdded" -> addStyles, "StylesRemoved" -> removeStyles, "AllStyles" -> $CellContext`x |>]
									]
								),(*, Method -> "Queued"*)
								(
									(*CurrentValue[nbo, {TaggingRules, "PresenterSettings", "SlideBreakDefaults", "PaletteNumber"}] = 0;*)
									CurrentValue[nbo, {TaggingRules, "PresenterSettings", "Dialogs", "SlideBreakDefaults"}] = 0;
									DialogReturn[$Canceled]
								)
							}
						],
					Alignment -> Right
					]
				}}, Spacings -> {0, {1.5, 2, 1.5}}, Alignment -> {Left, Top}

				], ImageMargins -> 15
				]
				],
			CellContext -> Notebook,
			(*DynamicUpdating -> True,*)
			WindowTitle -> Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "OpenAutomaticSlideBreakDialog"],
			Background -> GrayLevel[0.95],
			NotebookEventActions -> {
				"ReturnKeyDown" :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
				"EscapeKeyDown" :> FE`Evaluate[FEPrivate`FindAndClickCancelButton[]],
				"WindowClose" :> (FE`Evaluate[FEPrivate`FindAndClickCancelButton[]];
									(*CurrentValue[nbo, {TaggingRules, "PresenterSettings", "SlideBreakDefaults", "PaletteNumber"}] = 0;*)
									CurrentValue[nbo, {TaggingRules, "PresenterSettings", "Dialogs", "SlideBreakDefaults"}] = 0;
									)
				}
(*
				,
			NotebookDynamicExpression :> (
				If[Not[MemberQ[Notebooks[], no]], NotebookClose[EvaluationNotebook[]]]
			)
*)
	]
]
];
(*
AutomaticSlideBreakDialog[a__]:= Message[AutomaticSlideBreakDialog::argx, a];
AutomaticSlideBreakDialog::argx = "Argument should be a NotebookObject: `1`";
AutomaticSlideBreakDialog::info = "`1`";
*)



(*******************************************************

 ApplyAutomaticSlideBreakDialogResults

 Modify stylesheet to apply the result from AutomaticSlideBreakDialog
 (add / remove cell styles used for slide breaks)

	ApplyAutomaticSlideBreakDialogResults[
		nbObject,
		<|
			"StylesAdded" -> addStyles,
			"StylesRemoved" -> removeStyles
		|>
	]

	in: notebook, Association of which cell styles are added/removed
	out: apply new stylesheet to notebook

*******************************************************)

ApplyAutomaticSlideBreakDialogResults[nbObj_, result_, opts___?OptionQ]:=
Module[{stylesAdded, stylesRemoved, styleSheet, margins, stylesTotal, updatedStylesheet, menuValue,
	addedObjects, removedObjects, allSlideBreakStyles},

	allSlideBreakStyles = result["AllStyles"];
	stylesAdded = result["StylesAdded"];

	If[stylesAdded =!= {},
		AddSlideBreakCells[nbObj, "CellStyles" -> stylesAdded]
	];

	stylesRemoved = result["StylesRemoved"];

	If[stylesRemoved =!= {},
		RemoveSlideBreakCells[nbObj, "CellStyles" -> stylesRemoved]
	];

(*	Code below was commented out when back-tracked on using AttachedCells as Slide Breaks *)
(*
	styleSheet = result["Stylesheet"];

	stylesAdded =
		(
			menuValue = CurrentValue[nbObj, {StyleDefinitions, #, MenuSortingValue}];
			margins = Setting[PresenterTools`Styles`GetStyleOptionValue[styleSheet, #, CellMargins, "Slideshow Working"]];
			margins = margins /. {
						FrontEnd`AbsoluteCurrentValue[{WindowSize, 1}] :> CurrentValue[nbObj, {WindowSize, 1}],
						FrontEnd`AbsoluteCurrentValue[{WindowSize, 2}] :> CurrentValue[nbObj, {WindowSize, 2}]
						};

			margins = SetVerticalCellMargin["NudgeUp", margins, nbObj, AbsoluteCurrentValue[nbObj, {WindowSize}], 0.2];
			{
				{#, "Working", None, Flatten@{
							CreateSlideBreakCell[#],
							PageBreakAbove -> True,
							CounterIncrements->{#,"SlideShowNavigationBar"},
							CellMargins -> margins,
							System`CellElementSpacings->{"ClosedGroupTopMargin"->4},
(*							System`CellGroupingRules->{"SectionGrouping", -5},*)
							System`MenuSortingValue->Inherited
					}},
				{#, "Slideshow Working", None, {
							CellMargins -> margins,
							System`MenuSortingValue->Inherited,
							System`CellGroupingRules->{"SectionGrouping", -5},
							CellElementSpacings->{"ClosedGroupTopMargin"->150}
					}},
				{#, "Slideshow Presentation", StyleDefinitions -> StyleData[#, "Slideshow Working"], {
							System`MenuSortingValue->Inherited
					}}
			}
		) & /@ result["StylesAdded"];

	stylesRemoved =
		(
			cellGrouingRules = CellGroupingRules /. FE`Evaluate@FEPrivate`AbsoluteResolveCellStyle[nbObj, #, "Working", "IncludeEnvironment" -> True];
			{
				{#, "Working", None, Flatten@{
						RuleDelayed[CellDynamicExpression, Null],
						CellElementSpacings->{"ClosedGroupTopMargin"->4},
						If[MemberQ[{"Title", "Chapter", "Section"}, #],
							PageBreakAbove -> False,
						(* ELSE *)
							PageBreakAbove -> Inherited
						],
						CounterIncrements -> Inherited,
						CellMargins -> Inherited,
						System`CellElementSpacings -> {"ClosedGroupTopMargin" -> 4},
(*						System`CellGroupingRules -> Inherited,*)
						System`MenuSortingValue -> Inherited
				}},
				{#, "Slideshow Working", None, {
						CellMargins -> Inherited,
						CellElementSpacings -> {"ClosedGroupTopMargin" -> Automatic},
						System`MenuSortingValue -> Inherited,
						(* TODO: Not all cell styles use Section Grouping... *)
						System`CellGroupingRules -> cellGrouingRules
(*						System`CellGroupingRules -> Inherited*)
				}}
			}
		) & /@ result["StylesRemoved"];

	stylesTotal = Join[Flatten[stylesAdded, 1], Flatten[stylesRemoved, 1]];

	updatedStylesheet = PresenterTools`Styles`UpdateStylesheet[stylesTotal, styleSheet];

 	SetOptions[nbObj, StyleDefinitions -> updatedStylesheet]
*)
];
(*
ApplyAutomaticSlideBreakDialogResults[a___]:= Message[ApplyAutomaticSlideBreakDialogResults::argx, {a}];
ApplyAutomaticSlideBreakDialogResults::argx = "`1`";
ApplyAutomaticSlideBreakDialogResults::info = "`1`";
*)





(*******************************************************

 SlideBreakRefreshQ

	in: xxx
	out: yyy

*******************************************************)

SlideBreakRefreshQ[nbObj_ /; MatchQ[Cells[nbObj], {}] ]:= False;

SlideBreakRefreshQ[nbObj_]:=
Module[{cs, cells = Cells[nbObj], stylepat},

	If[Not@SlideBreakFirstCellQ[cells],
		True,
		(* ELSE *)
		stylepat = Alternatives @@ CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "SlideBreakStyles"}, {}];
		MemberQ[Partition[CurrentValue[cells, "CellStyleName"], 2, 1], {Except["SlideShowNavigationBar"], stylepat}]
	]
];
SlideBreakRefreshQ[a___]:= Message[SlideBreakRefreshQ::argx, {a}];
SlideBreakRefreshQ::argx = "`1`";
SlideBreakRefreshQ::info = "`1`";


SlideBreakFirstCellQ[{}]:= False;

SlideBreakFirstCellQ[{firstobj_, rest___}]:=
	MemberQ[CurrentValue[firstobj, CellStyle], "SlideShowNavigationBar"];



(* NOT USED *)
FirstSlideRefreshQ[nbObj_]:=
Module[{cs, cells = Cells[nbObj], pagewiseNotebookBaseStyle},
	pagewiseNotebookBaseStyle =
		CurrentValue[EvaluationNotebook[], {StyleDefinitions, "FirstSlide", System`PrivateCellOptionsPagewiseNotebookBaseStyle}];

	If[(pagewiseNotebookBaseStyle === {}),
		cs =
		  Cases[cells, x_ /; MemberQ[
		      CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "SlideBreakStyles"}, {}],
		      "Title"] && (x === cells[[1]] || (CurrentValue[
		          cells[[Position[cells, x][[1, 1]] - 1]], "CellStyleName"] =!=
		          "SlideShowNavigationBar"))];
		 cs =!= {}
	]
];
FirstSlideRefreshQ[a___]:= Message[FirstSlideRefreshQ::argx, {a}];
FirstSlideRefreshQ::argx = "`1`";
FirstSlideRefreshQ::info = "`1`";






(*******************************************************

 AddSlideBreakCells

	in: Notebook
	out: Notebook

*******************************************************)
Options[AddSlideBreakCells] = {
	"CellStyles" -> {},
	"SlideBreakCell" -> Cell["", "SlideShowNavigationBar", CellTags -> "SlideShowHeader"]
};
AddSlideBreakCells[nbObj_, OptionsPattern[]] :=
Module[{cells = Cells[nbObj], cellstyles, cs, slideBreakCell, styleList, stylepat
	},
	styleList = OptionValue["CellStyles"];
	slideBreakCell = OptionValue["SlideBreakCell"];

	(* FirstSlide test for top of notebook *)
	If[(cells === {}) || Not@MemberQ[CurrentValue[First[cells], CellStyle], "SlideShowNavigationBar"],
		SelectionMove[nbObj, Before, Notebook, AutoScroll -> False];
		NotebookWrite[nbObj,
			Cell["", "SlideShowNavigationBar", "FirstSlide", CellTags -> "SlideShowHeader"], All, AutoScroll -> False];
		cells = Cells[nbObj]
	];

	If[styleList === {}, Return[]];

	(* Add NavigationBarCell above all SlideBreakCells *)

	stylepat = Alternatives @@ styleList;
	cellstyles = Thread[cells -> CurrentValue[cells, "CellStyleName"]];
	cellstyles = Partition[cellstyles, 2, 1];
	cs = Cases[cellstyles, {obj1_ -> Except["SlideShowNavigationBar"], obj2_ -> stylepat} :> obj2];

	If[cs =!= {},
		(SelectionMove[#, All, Cell, AutoScroll -> False];
		SelectionMove[#, Before, Cell, AutoScroll -> False];
		NotebookWrite[nbObj, slideBreakCell, All, AutoScroll -> False];
		) & /@ cs;
		SelectionMove[nbObj, After, Cell, AutoScroll -> False];
	]
]
AddSlideBreakCells[a___]:= Message[AddSlideBreakCells::argx, {a}];
AddSlideBreakCells::argx = "`1`";
AddSlideBreakCells::info = "`1`";




(*******************************************************

 RemoveSlideBreakCells

	in: xxx
	out: yyy

*******************************************************)
Options[RemoveSlideBreakCells] = {
	"CellStyles" -> {},
	"SlideBreakStyle" -> "SlideShowNavigationBar"
};

RemoveSlideBreakCells[nbObj_, OptionsPattern[]] :=
Module[{cells, slideBreakStyle, styleList, stylePat, cellStyles, cs},
	styleList = OptionValue["CellStyles"];
	If[styleList === {}, Return[]];
	slideBreakStyle = OptionValue["SlideBreakStyle"];

	cells = Cells[nbObj];
	cellStyles = CurrentValue[cells, "CellStyleName"];
	If[!MemberQ[cellStyles, slideBreakStyle], Return[]];

	(* FIXME: Not sure I captured all the cases of the original implementation. *)

	stylePat = Alternatives @@ styleList;
	cs = Cases[Partition[Thread[cells -> cellStyles], 2, 1], {obj_ -> slideBreakStyle, _ -> stylePat} :> obj];
	NotebookDelete[cs]
]
RemoveSlideBreakCells[a___]:= Message[RemoveSlideBreakCells::argx, {a}];
RemoveSlideBreakCells::argx = "`1`";
RemoveSlideBreakCells::info = "`1`";





(*******************************************************

 CreateSlideBreakCell

	Criteria for slide break to appear:

	TrueQ[(
		And[
			CurrentValue[{TaggingRules, "PresenterSettings", "ShowSlideBreaks"}],
			MemberQ[CurrentValue[{TaggingRules, "PresenterSettings", "SlideBreakStyles"}], cellStyle],
			CurrentValue[ParentCell @ EvaluationCell[], PageBreakAbove] === True
		]
	)]

	in: cellStyle
	out: CellDynamicExpression

*******************************************************)

CreateSlideBreakCell[style_String]:=
With[{cellStyle = style}, {
	CellDynamicExpression:>
   FrontEndExecute[
    FrontEnd`AttachCell[
     EvaluationCell[],
     Cell[
      BoxData[
       ToBoxes[
        Dynamic@Refresh[
         If[
          TrueQ[
           And[
            CurrentValue[EvaluationNotebook[], ScreenStyleEnvironment] === "Slideshow Working",
            MemberQ[CurrentValue[{TaggingRules, "PresenterSettings", "SlideBreakStyles"}, {}], cellStyle],
            CurrentValue[ParentCell[EvaluationCell[]], PageBreakAbove] === True]
            ],
          Dynamic[RawBoxes[FEPrivate`FrontEndResource["PresenterToolExpressions","AutomaticSlideBreakCell"]]],
	Pane[Style["", FontSize -> 1], ImageSize -> 1, FrameMargins -> 0, ImageMargins -> 0]], None]]], "SlideBreakAttachCell"], {
      Offset[{15, 50}, 0], {"CellBracket", Top}}, {Right, Bottom}]]
}];
(* { Offset[{25, 70}, 0], {"CellBracket", Top}} *)
CreateSlideBreakCell::argx = "`1`";
CreateSlideBreakCell::info = "`1`";




(*******************************************************

 EndPresentation
	Takes one to 'Slideshow Working' environment and
	sets various options depending on starting location

	in: NotebookObject, location
	out: None

*******************************************************)

EndPresentation[nbObj_]:=
Module[{},
	SetOptions[nbObj, ScreenStyleEnvironment -> "Slideshow Working"];
	CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "ShowToolbar"}] = True;
	CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "ShowSlideBreaks"}] = True;
];

(* Two argument versions *)

(* location: Menu within NavigationBar in Slideshow Presentation environment *)
EndPresentation[nbObj_,"NavBar"]:=
Module[{},
	EndPresentation[nbObj];
];
(* location: Menu within NavigationBar in Slideshow Presentation environment (Full screen) *)
EndPresentation[nbObj_,"NavBarFullScreen"]:=
Module[{},
(*	FrontEndExecute[FrontEnd`Value[FEPrivate`NotebookToggleFullScreen[nbObj]]]; *)
	CurrentValue[nbObj, WindowSize] = PresenterTools`GetPreviousWindowSize[nbObj];
	CurrentValue[nbObj, WindowMargins] = CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "WindowMargins"}, Automatic];
	EndPresentation[nbObj];
];

EndPresentation[nbObj_,"MenuSlideshowPresentation"]:=
Module[{},
	EndPresentation[nbObj];
];
EndPresentation[nbObj_,"MenuSlideshowPresentationFullScreen"]:=
Module[{},
	CurrentValue[nbObj, WindowSize] = PresenterTools`GetPreviousWindowSize[nbObj];
	CurrentValue[nbObj, WindowMargins] = CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "WindowMargins"}, Automatic];
	EndPresentation[nbObj];
];


EndPresentation[nbObj_,"MenuScrollingPresentation"]:=
Module[{},
	EndPresentation[nbObj];
];
EndPresentation[nbObj_,"MenuScrollingPresentationFullScreen"]:=
Module[{},
	CurrentValue[nbObj, WindowSize] = PresenterTools`GetPreviousWindowSize[nbObj];
	CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "ShowToolbar"}] = True;
	SetOptions[nbObj, ScreenStyleEnvironment -> "Slideshow Working"];
];

EndPresentation[nbObj_,"KeyboardShortcut"]:=
Module[{},
	If[CurrentValue[nbObj, WindowSize] === {Full, Full},
		CurrentValue[nbObj, WindowSize] = PresenterTools`GetPreviousWindowSize[nbObj];
		CurrentValue[nbObj, WindowMargins] = CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "WindowMargins"}, Automatic];
		EndPresentation[nbObj],
	(* ELSE*)
		EndPresentation[nbObj]
	]
];

EndPresentation[nbObj_, location___]:=
Module[{},

	FrontEndExecute[FrontEnd`Value[FEPrivate`NotebookToggleFullScreen[nbObj]]]

];
EndPresentation::argx = "`1`";
EndPresentation::info = "`1`";



(*******************************************************

 NotebookToggleFullScreenWrapper
	Wrapper for FE's NotebookToggleFullScreen[]
	in: None
	out: None

*******************************************************)

NotebookToggleFullScreenWrapper[nbObj_]:=
Module[{},

	FrontEndExecute[FrontEnd`Value[FEPrivate`NotebookToggleFullScreen[nbObj]]]
(*
	Which[
		FEPrivate`MemberQ[{Full, {Full}, {Full,Full}}, FrontEnd`CurrentValue[nbObj, WindowSize]],
			FrontEndExecute[FrontEnd`Value[FEPrivate`NotebookToggleFullScreen[nbObj]]],
		True,
			FrontEndExecute[FrontEnd`Value[FEPrivate`NotebookToggleFullScreen[nbObj]]]
	]
*)
];
(*NotebookToggleFullScreenWrapper[a___]:= Message[NotebookToggleFullScreenWrapper::argx, {a}];*)
NotebookToggleFullScreenWrapper::argx = "`1`";
NotebookToggleFullScreenWrapper::info = "`1`";















(**************************************

 ExportThemeButton
	Save theme as style sheet
	in: None
	out: expression

***************************************)

ExportThemeButton[]:=
Module[{styleSheetExpr, nameOfNewStylesheet},
	With[{icon = icon["SaveTheme"]},
		MouseAppearance[Tooltip[
			Button[
				Pane[
					Column[{
						icon,
						Style["Save\nTheme As...", 12, FontFamily :> CurrentValue["PanelFontFamily"], TextAlignment -> Center]
						}, Spacings -> {0, .5, 0}, Alignment -> {Center, Bottom}
					], Alignment -> {Center, Center}, ImageSize -> {Automatic, 80}
				],
				(* *)
				styleSheetExpr = CurrentValue[EvaluationNotebook[], StyleDefinitions];

				If[Head[styleSheetExpr] === Notebook,
					(
						nameOfNewStylesheet =
							CreateDialog[
								DynamicModule[{themeFilename = None, $errortext = ""},
									Pane[
										Grid[{{
											Style["Stylesheet name:", 14, FontFamily :> CurrentValue["PanelFontFamily"]],
											Style[Dynamic[$errortext], 14, Red, FontFamily :> CurrentValue["PanelFontFamily"]]
											}, {
											Style["FileNameJoin[{$UserBaseDirectory, \"SystemFiles\", \"FrontEnd\", \"StyleSheets\"}]", 11, GrayLevel[0.5], FontFamily :> CurrentValue["PanelFontFamily"]], SpanFromLeft
											}, {
											InputField[Dynamic[themeFilename], String, ImageSize -> 250, BaseStyle -> {"ControlStyle"}, FieldHint -> "MyStylesheetName"], SpanFromLeft
											}, {
											Item[Row[{
												CancelButton[], Spacer[10],
												DefaultButton[
													Style["Save", FontColor :> FEPrivate`If[FEPrivate`$OperatingSystem === "MacOSX", GrayLevel[1], Automatic]],
													If[
														(MatchQ[None, Setting@Dynamic[themeFilename]]),
														$errortext = "Invalid name.",
													(* ELSE *)
														DialogReturn[
															ExportTheme[
																FileNameJoin[{$UserBaseDirectory, "SystemFiles", "FrontEnd", "StyleSheets", Setting@Dynamic[themeFilename]}],
																styleSheetExpr
																]
														]
													],
													ImageSize -> Automatic]
												}], Alignment -> Right], SpanFromLeft
											}},
										Alignment -> {Left, Center}, Spacings -> {1, {1, 1, 1.5}}, Frame -> None],
										ImageMargins -> 10
									]
								],
								NotebookEventActions -> {
									"ReturnKeyDown" :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
									"EscapeKeyDown" :> FE`Evaluate[FEPrivate`FindAndClickCancelButton[]],
									"WindowClose" :> FE`Evaluate[FEPrivate`FindAndClickCancelButton[]]
									}
							]
(*							If[TrueQ[nameOfNewStylesheet =!= $Canceled],
								nameOfNewStylesheet = If[StringMatchQ[nameOfNewStylesheet, "*.nb"], nameOfNewStylesheet, nameOfNewStylesheet <> ".nb"];
								Export[ FileNameJoin[{$UserBaseDirectory, "SystemFiles", "FrontEnd", "StyleSheets", nameOfNewStylesheet}], styleSheetExpr]
							]
*)					)
				,
					MessageDialog["No changes have been made to stylesheet."]

				], Appearance -> None(*, Method -> "Queued"*)
		], "Save out current embedded stylesheet.", TooltipDelay -> .25], "LinkHand"
		]
	]
];
ExportThemeButton::argx = "`1`";




(*******************************************************

 ExportTheme

	in: xxx
	out: yyy

*******************************************************)

ExportTheme[name_String, nbObj_NotebookObject]:= ExportTheme[name, NotebookGet[nbObj]];

ExportTheme[name_String, nbExpr_Notebook]:=
Module[{styleSheetExpr = nbExpr},

	If[FileType[name] =!= None,
		styleSheetExpr = styleSheetExpr /. Visible -> True;
		Export[ FileNameJoin[{$UserBaseDirectory, "SystemFiles", "FrontEnd", "StyleSheets", name}], nbExpr],
	(* ELSE *)
		Message[ExportTheme::info, "File already exists."]
	]
];
ExportTheme[a___]:= Message[ExportTheme::argx, {a}];
ExportTheme::argx = "`1`";
ExportTheme::info = "`1`";







(*******************************************************

 ChangeNotebookStyleSheet

	in: xxx
	out: yyy

*******************************************************)

ChangeNotebookStyleSheet[nbObj_, newStyleSheet_]:=
Module[{stylesheet, res},

	stylesheet = CurrentValue[nbObj, StyleDefinitions];

	res = iChangeNotebookStyleSheet[nbObj, stylesheet, newStyleSheet]
];
ChangeNotebookStyleSheet[a___]:= Message[ChangeNotebookStyleSheet::argx, {a}];
ChangeNotebookStyleSheet::argx = "`1`";
ChangeNotebookStyleSheet::info = "`1`";



iChangeNotebookStyleSheet[nbObj_, stylesheet_String, styleDefinitions_]:=
	SetOptions[nbObj, StyleDefinitions -> styleDefinitions]

iChangeNotebookStyleSheet[nbObj_, stylesheet_FrontEnd`FileName, styleDefinitions_]:=
	SetOptions[nbObj, StyleDefinitions -> styleDefinitions]

iChangeNotebookStyleSheet[nbObj_, stylesheet_Notebook, styleDefinitions_]:=
Module[{newStylesheet},
(*	Message[ChangeNotebookStyleSheet::info, {styleDefinitions, stylesheet}];*)

	newStylesheet = stylesheet /. Cell[StyleData[StyleDefinitions -> _, o___]] ->
										Cell[StyleData[StyleDefinitions -> styleDefinitions, o]];
		SetOptions[nbObj, StyleDefinitions -> newStylesheet]
];




(*******************************************************

 UpdateNotebookStyleHints

	in: xxx
	out: yyy

*******************************************************)
Options[UpdateNotebookStyleHints] = {
	"FontSet" -> "Default",
	"ColorSet" -> "Default"
}
UpdateNotebookStyleHints[nbObj_NotebookObject, opts___?OptionQ]:=
Module[{stylesheet, definitions, currentStyleHints, fontSet, colorSet, updatedOptions},

	stylesheet = CurrentValue[nbObj, StyleDefinitions];
	currentStyleHints = CurrentValue[nbObj, {StyleDefinitions, "Notebook", System`StyleHints}];

	fontSet = "FontSet" /. {opts} /. Options[UpdateNotebookStyleHints];
	fontSet = fontSet /. {name_String} :> PresenterTools`GetThemeInformation[name, "FontSet"];

	colorSet = "ColorSet" /. {opts} /. Options[UpdateNotebookStyleHints];
	colorSet = colorSet /. {name_String} :> PresenterTools`GetThemeInformation[name, "ColorSet"];

	updatedOptions = ResetOptions[currentStyleHints, {"FontSet" -> fontSet, "ColorSet" -> colorSet}];

(*
Message[UpdateNotebookStyleHints::info, newHints];
Message[UpdateNotebookStyleHints::info, currentStyleHints];
*)
	definitions =
		Switch[Head[stylesheet],
			String,
				PresenterTools`Styles`UpdateStylesheet[{
					{"Notebook", "Working", None, Flatten@{MenuSortingValue -> None, System`StyleHints -> updatedOptions}}
					}, stylesheet],
			FrontEnd`FileName,
				PresenterTools`Styles`UpdateStylesheet[{
					{"Notebook", "Working", None, Flatten@{MenuSortingValue -> None, System`StyleHints -> updatedOptions}}
					}, stylesheet],
			Notebook,
				PresenterTools`Styles`UpdateStylesheet[{
					{"Notebook", "Working", None, Flatten@{MenuSortingValue -> None, System`StyleHints -> updatedOptions}}
					}, stylesheet],
			_,
				$Failed
		];

	SetOptions[nbObj, StyleDefinitions -> definitions]



];
UpdateNotebookStyleHints[a___]:= Message[UpdateNotebookStyleHints::argx, {a}];
UpdateNotebookStyleHints::argx = "`1`";
UpdateNotebookStyleHints::info = "`1`";










(**************************************

 SwitchToSlideShowPresentationButton

	in: None
	out: expression

***************************************)

SwitchToSlideShowPresentationButton[]:=
Module[{},
	With[{icon = bitmapResource["SlideView"]},
	MouseAppearance[Tooltip[
		PaneSelector[{
			True -> Button[
				Panel[icon, Alignment -> {Center, Center}, ImageMargins -> 0, FrameMargins -> 3, (* ImageSize -> {75, 70},*) Appearance -> None,
					Background -> GrayLevel[1]],
				SwitchToSlideShowPresentation[InputNotebook[]],
			Appearance -> None],
			False -> Button[
				Panel[icon, Alignment -> {Center, Center}, ImageMargins -> 0, FrameMargins -> 3, (* ImageSize -> {75, 70},*) Appearance -> None,
					Background -> GrayLevel[.95]],
				SwitchToSlideShowPresentation[InputNotebook[]],
			Appearance -> None]
		}, Dynamic[CurrentValue["MouseOver"]]],
	"Switch to Slide Presentation", TooltipDelay -> .25], "LinkHand"
	]
	]
];


SwitchToSlideShowPresentation[nbObj_NotebookObject]:=
(
	Set[CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "ShowToolbar"}], False];
	SetOptions[nbObj, { ScreenStyleEnvironment -> "Slideshow Presentation"}];
);




(**************************************

 SwitchToSlideShowWorkingButton

	in: None
	out: expression

***************************************)

SwitchToSlideShowWorkingButton[]:=
Module[{},
	With[{icon = bitmapResource["ScrollingView"]},
	MouseAppearance[Tooltip[
	Button[
		MouseAppearance[
			Panel[
				Column[{icon,
					Style["Slideshow\nWorking", 12, FontFamily :> CurrentValue["PanelFontFamily"], TextAlignment -> Center,
						FontColor -> GrayLevel[0]]
				}, Spacings -> {0, 1, 0}, Alignment -> {Center, Bottom}], Alignment -> {Center, Center}, ImageSize -> {75, 70}, Appearance -> None,
					Background -> GrayLevel[.975]], "LinkHand"
		], SwitchToSlideShowWorking[InputNotebook[]],
	Appearance -> None],
	"Slide show working", TooltipDelay -> .25], "LinkHand"
	]
	]
];


SwitchToSlideShowWorking[nbObj_NotebookObject]:=
(
(*	Set[CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "PresenterToolsPresentationToolbar"}], False]; *)
	Set[CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "ShowToolbar"}], True];
	SetOptions[nbObj, { ScreenStyleEnvironment -> "Slideshow Working"}];
)



(**************************************

 SwitchToPresentationButton

	in: None
	out: expression

***************************************)
(*
SwitchToPresentationButton[]:=
Module[{},
	With[{icon = bitmapResource["ScrollingView"]},
	MouseAppearance[Tooltip[
		PaneSelector[{
			True -> Button[
						Panel[icon, Alignment -> {Center, Center}, ImageMargins -> 0, FrameMargins -> 3, (*ImageSize -> {75, 70},*) Appearance -> None,
							Background -> GrayLevel[1]],
						SwitchToPresentation[InputNotebook[]], Appearance -> None],
			False -> Button[
						Panel[icon, Alignment -> {Center, Center}, ImageMargins -> 0, FrameMargins -> 3, (*ImageSize -> {75, 70},*) Appearance -> None,
							Background -> GrayLevel[.95]],
						SwitchToPresentation[InputNotebook[]], Appearance -> None]
		}, Dynamic[CurrentValue["MouseOver"]]],
	"Switch to Scrolling Presentation", TooltipDelay -> .25], "LinkHand"
	]
	]
];

SwitchToPresentation[nbObj_NotebookObject]:=
(
(*	Set[CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "PresenterToolsPresentationToolbar"}], True]; *)
	SetOptions[nbObj, { ScreenStyleEnvironment -> "Presentation"}];
)
*)


(**************************************

 WindowElementsMenu

	in: None
	out: expression

***************************************)

(*WindowElementsMenu[]:= WindowElementsMenu[InputNotebook[]]; *)
WindowElementsMenu[nbObj_, icon_]:=
Module[{},

	With[{image = icon},
	DynamicModule[{
		list = CurrentValue[nbObj, WindowElements],
		choices = {"StatusArea", (*"MemoryMonitor",*) "MagnificationPopUp", "HorizontalScrollBar", "VerticalScrollBar", "ResizeArea", "MenuBar"}},

		ActionMenu[
			image,
			{
				Sequence@@((Row[{Style["\[Checkmark] ", ShowContents -> Dynamic[MemberQ[list, #]]], #}] :>
					(
						If[MemberQ[list, #],
							list = DeleteCases[list, #],
						(* ELSE *)
							AppendTo[list, #]
						];
						CurrentValue[nbObj, WindowElements] = list
					)
				) & /@ choices),
				Delimiter,
				"Show All" :> (
					Set[CurrentValue[nbObj, WindowElements], Inherited];
					Set[CurrentValue[nbObj, WindowFrameElements], Inherited]
					),
				"Hide All" :> (
					Set[CurrentValue[nbObj, WindowElements], {}];
					(*Set[CurrentValue[nbObj, WindowFrameElements],
						DeleteCases[CurrentValue[InputNotebook[], WindowFrameElements], "ResizeArea"] ]*)
					)
			}, Appearance -> None
		]
	]]
];

(*
WindowElementsMenu[]:= WindowElementsMenu[InputNotebook[]];
WindowElementsMenu[nbObj_]:=
Module[{},
	With[{choices = CurrentValue[nbObj, WindowElements], list = CurrentValue[nbObj, WindowElements]},
	{
				"All On" :> Set[CurrentValue[nbObj, WindowElements], Inherited],
				"All Off" :> Set[CurrentValue[nbObj, WindowElements], {}],
				Delimiter,
				Sequence@@((Row[{Style["+ ", ShowContents -> Dynamic[MemberQ[CurrentValue[nbObj, WindowElements], #]]], #}] :>
					(
						If[MemberQ[CurrentValue[nbObj, WindowElements], #],
							list = DeleteCases[CurrentValue[nbObj, WindowElements], #]
						,
							AppendTo[CurrentValue[nbObj, WindowElements], #]
						];
						CurrentValue[nbObj, WindowElements] = CurrentValue[nbObj, WindowElements]
					)
				) & /@ choices)
			}
	]
];
*)

(**************************************

 NotebookWindowSizeMenu

	in: Notebook
	out: None

***************************************)
Options[NotebookWindowSizeMenu] = {
	"ToolbarHeight" -> 80
}
NotebookWindowSizeMenu[]:= NotebookWindowSizeMenu[InputNotebook[]];
NotebookWindowSizeMenu[nbObj_, opts:OptionsPattern[]]:=
Module[{height},

	height = OptionValue["ToolbarHeight"];

	With[{toolbar = height},

		{
			"800 \[Times] 600 (4:3)" :> (CurrentValue[nbObj, WindowSize] = {800, 600 + toolbar};
							CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "WindowSize"}] = {800, 600 + toolbar};),
			"1024 \[Times] 768 (4:3)" :> (CurrentValue[nbObj, WindowSize] = {1024, 768 + toolbar};
							CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "WindowSize"}] = {1024, 768 + toolbar};),
			"1280 \[Times] 960 (4:3)" :> (CurrentValue[nbObj, WindowSize] = {1280, 960 + toolbar};
							CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "WindowSize"}] = {1280, 960 + toolbar};),
			"1366 \[Times] 768 (16:9)" :> (CurrentValue[nbObj, WindowSize] = {1366, 768 + toolbar};
							CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "WindowSize"}] = {1366, 768 + toolbar};),
			"1400 \[Times] 1050 (4:3)" :> (CurrentValue[nbObj, WindowSize] = {1400, 1050 + toolbar};
							CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "WindowSize"}] = {1400, 1050 + toolbar};),
			"1600 \[Times] 900 (16:9)" :> (CurrentValue[nbObj, WindowSize] = {1600, 900 + toolbar};
							CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "WindowSize"}] = {1600, 900 + toolbar};),
			"1680 \[Times] 1050 (16:10)" :> (CurrentValue[nbObj, WindowSize] = {1680, 1050 + toolbar};
							CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "WindowSize"}] = {1680, 1050 + toolbar};),
			"1920 \[Times] 1080 (16:9)" :> (CurrentValue[nbObj, WindowSize] = {1920, 1080 + toolbar};
							CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "WindowSize"}] = {1920, 1080 + toolbar};),
			"2560 \[Times] 1440 (16:9)" :> (CurrentValue[nbObj, WindowSize] = {2560, 1440 + toolbar};
							CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "WindowSize"}] = {2560, 1440 + toolbar};)
		}
	]

];

NotebookWindowSizeMenu[a__] := Message[NotebookWindowSizeMenu::argx, {a}];
NotebookWindowSizeMenu::argx = "Argument should be a NotebookObject: `1`";
NotebookWindowSizeMenu::info = "`1`";









CombineStyleDataOptions[Cell[StyleData[sty_String /; MemberQ[$StylesheetThemeCellStyles, sty]], opts___?OptionQ]] :=
Module[{cellStyles, cellOptions, newOptions},
	newOptions =
		Join[
			GetBaseCellStyleOptions[sty],
			DeleteCases[{opts}, Rule[o_ /; MemberQ[$StylesheetThemeCellOptions, o], _], Infinity]
		];
	Cell[StyleData[sty], Sequence @@ newOptions]
];
CombineStyleDataOptions[a___] := a


GetBaseCellStyleOptions[styleName_String] :=
Module[{},
(*
	(# :> CurrentValue[{StyleHints, styleName, #}] & /@ $StylesheetThemeCellOptions)
*)
With[{num = 0.0008},
	Flatten[{
		If[StringMatchQ[styleName, "Input" | "Output" | "Code" | "Message"],
			{}
		,
			FontFamily :> Dynamic[CurrentValue[{StyleHints, styleName, FontFamily}] ]
		],
		FontSize :> Dynamic[num FrontEnd`AbsoluteCurrentValue[{WindowSize, 1}] CurrentValue[{StyleHints, styleName, FontSize}] ],
		FontColor :> Dynamic[CurrentValue[{StyleHints, styleName, FontColor}] ],
		FontWeight :> Dynamic[CurrentValue[{StyleHints, styleName, FontWeight}] ],
		FontSlant :> Dynamic[CurrentValue[{StyleHints, styleName, FontSlant}] ],
(*
		CellDingbat :> Dynamic[FEPrivate`If[CurrentValue[{StyleHints, styleName, CellDingbat}] === None, None, Dynamic[CurrentValue[{StyleHints, styleName, CellDingbat}]]] ],
*)
		CellFrame :> Dynamic[CurrentValue[{StyleHints, styleName, CellFrame}] ],
		CellFrameColor :> Dynamic[CurrentValue[{StyleHints, styleName, CellFrameColor}] ],
		CellFrameMargins :> Dynamic[CurrentValue[{StyleHints, styleName, CellFrameMargins}] ],
		CellMargins :> Dynamic[CurrentValue[{StyleHints, styleName, CellMargins}] ],
		TextAlignment :> Dynamic[CurrentValue[{StyleHints, styleName, TextAlignment}] ],
		Background :> Dynamic[CurrentValue[{StyleHints, styleName, Background}] ]
	}]
]
]










stylesheetThumbnail[img_, rolloverImg_, _]:=
	Tooltip[thumbnail[img], thumbnail[rolloverImg], TooltipStyle -> {Background -> White, CellFrame -> 0}, TooltipDelay -> 0.35]
thumbnail[id_] :=
  RawBoxes[DynamicBox[FEPrivate`FrontEndResource["LocalizedBitmaps", id]]];






(*******************************************************

 ThemeOptionsDialog

	in: PresenterTools`GetThemeNames[]
		{"Default", "Garnet", "Facet", "Carbon", "Aqua", "Sky"}
	out: Notebook

*******************************************************)

Options[ThemeOptionsDialog] = {
	"ShowPopulateNotebookButton" -> False,
	"Title" -> "Theme options",
	"WindowTitle" -> None,
	"ThemeLimit" -> 6
};
ThemeOptionsDialog[nbObj_NotebookObject, opts___?OptionQ]:= ThemeOptionsDialog[nbObj, PresenterTools`GetThemeNames[], opts];

ThemeOptionsDialog[nbObj_, themeNames_List, OptionsPattern[]]:=
Module[{themes, themeList = themeNames, themesInfo, themeFontsMenu, themeColorsMenu, title, populateButtonQ, currentTheme, currentFontSet, currentColorSet, stylesheet, themeLimit},

	title = OptionValue["Title"];
	populateButtonQ = OptionValue["ShowPopulateNotebookButton"];

	themeLimit = OptionValue["ThemeLimit"];
	If[Length[themeList] > themeLimit,
		themeList = Take[themeList, themeLimit];
	];

	currentTheme = CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "Theme", "Name"}, "Default"];
(*	If[Not@MemberQ[themeList, currentTheme],
		currentTheme = "Default"
	];*)
	currentFontSet = CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "Theme", "FontSetName"}, "Default"];
(*	If[Not@MemberQ[themeList, currentFontSet],
		currentFontSet = "Default"
	];*)
	currentColorSet = CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "Theme", "ColorSetName"}, "Default"];
(*	If[Not@MemberQ[themeList, currentColorSet],
		currentColorSet = "Default"
	];*)

	(* gather information about each of themes *)
	themesInfo = Rule[#, PresenterTools`GetThemeInformation[#]] & /@ themeList;
	themes = Association@@ themesInfo;

	themeColorsMenu = PresenterTools`GetThemeColorMenu[#, themes[][[#]][["ColorSet"]]] & /@ themeList;
	themeColorsMenu = Association@@ themeColorsMenu;
	themeFontsMenu = PresenterTools`GetThemeFontMenu[#, themes[][[#]][["FontSet"]]] & /@ themeList;
	themeFontsMenu = Association@@ themeFontsMenu;

	With[{
		allThemes = themes,
		detailCount = 6, (* TODO: detail count should be defined per theme information *)
		themeDisplayNames = themeList,
		fontsets = themeFontsMenu,
		colorsets = themeColorsMenu,
(*		data = themeData,*)
		themeSize = {170, 96},
		detailSize = {284, 163},

		currenttheme = currentTheme,
		currentfontset = currentFontSet,
		currentcolorset = currentColorSet,
		windowTitle = title,
		ShowPopulateNotebookButtonQ = populateButtonQ,
		$CellContext`nb$ = nbObj
		},


		CreateDialog[{
			Cell[BoxData[ToBoxes@

			DynamicModule[{themeName = currenttheme, previoustheme = currenttheme, detailNumber = 1, numberOfDetails = detailCount,
				num, fontset = currentfontset, colorset = currentcolorset},

				Panel[
					Grid[{{
						Item[
					 		Style[Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "AutomaticSlideBreakDialogText"],
					   			13, FontFamily :> CurrentValue["PanelFontFamily"]],
			 				Alignment -> Left
			 			],
						SpanFromLeft,
						SpanFromLeft
						}, {
						Item[ Style[Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "ThemeChooserThemeLabel"],
							FontSize -> 16, FontFamily :> CurrentValue["PanelFontFamily"], FontColor -> RGBColor[0.537254, 0.537254, 0.537254]], Alignment -> Left],
						SpanFromLeft
						}, {
						(***************************

							Main theme image
							click image to select theme

						****************************)
						Grid[{
						Sequence@@Partition[
							With[{val = #,
								themeImage1 = StringJoin[ #, "-title-1", ".png"],
								themeImage0 = StringJoin[ #, "-title-0", ".png"]
								},
								MouseAppearance[
								EventHandler[
									Tooltip[
										Framed[
											(* show 'checkmarked' image for selected theme *)
											Dynamic@
											If[themeName === val,
												RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, themeImage1]],
											(* ELSE *)
												RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, themeImage0]]
											],
											(* add border to checkmarked image *)
											FrameStyle ->
												Dynamic[
													If[themeName === val, {Thickness[4], RGBColor[0.6665, 0.9175, 0.992]}, {Thickness[4], GrayLevel[0.95]} ]
													],
											FrameMargins -> 0,
											ImageSize -> {Automatic, 96},
											Background -> None,
											ContentPadding -> False
										],
											StringJoin["Theme: ", Setting[val]],
											TooltipDelay -> .25
										],
										{"MouseDown" :>
											If[CurrentValue["MouseClickCount"] == 2,
												themeName = val;
												FEPrivate`FindAndClickDefaultButton[],
											(* ELSE *)
												themeName = val;
											]
										}
(*									Appearance -> None,
									(*ImageSize -> themeSize, *)
									ContentPadding -> False,*)
								], "LinkHand"]
							]& /@ themeDisplayNames, (*Range[numberOfThemes], *)
							3
							]},
							FrameStyle -> GrayLevel[.6],
							Background -> None,
							FrameMargins -> {{15, 15}, {15, 15}},
							ImageMargins -> 0,
							Alignment -> {Left, Top}
						],
						SpanFromLeft
						 }, {
							(***************************

								Detail images

							****************************)
						(*Grid[{{*)
							(*******************************
							optional font / color selectors
							********************************)
						Item[
							Style[Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "ThemeChooserFontSetLabel"],
								14, FontFamily :> CurrentValue["ControlsFontFamily"], FontColor -> RGBColor[0.537254, 0.537254, 0.537254]], Alignment -> Left],
						Item[
							Style[Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "ThemeChooserColorSetLabel"],
							14, FontFamily :> CurrentValue["ControlsFontFamily"], FontColor -> RGBColor[0.537254, 0.537254, 0.537254]], Alignment -> Left],
						SpanFromLeft
						}, {
						PopupMenu[Dynamic[fontset], fontsets, ImageSize -> {350, Automatic}],
						PopupMenu[Dynamic[colorset], colorsets, ImageSize -> {180, Automatic} ]
						}, {
						Dynamic@Refresh[
							If[ShowPopulateNotebookButtonQ,
								Item[
									Pane[
							Grid[{{
										Checkbox[Dynamic[content], BaselinePosition -> Axis],
										Style[Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "ThemeChooserIncludeSampleContent"], FontSize -> 13, FontFamily :> CurrentValue["ControlsFontFamily"], FontColor -> RGBColor[0.39215, 0.39215, 0.39215]]
									}}, Spacings -> 0.5, Alignment -> {Left, Center}], ImageMargins -> {{0, 0}, {8, 0}}], Alignment -> {Left, Center}],
							(* ELSE *)
								" "
							], None],
							SpanFromLeft
						}, {
						OpenerView[{
							Pane[
							Style[Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "ThemeChooserPreviewLabel"],
									FontSize -> 14, FontFamily :> CurrentValue["ControlsFontFamily"], FontColor -> RGBColor[0.537254, 0.537254, 0.537254]],
							ImageSize -> Full, ImageMargins -> 0, FrameMargins -> 0, ContentPadding -> False
							],
						(****)
							Pane[
								Grid[{{
									Row[{
										Row[{
											Button[
												Panel[Style["\[FirstPage]", FontSize -> 24, FontColor -> GrayLevel[0.5]], Appearance -> None, ImageSize -> {15, 15}, Alignment -> Center,
													Background ->
													Dynamic[
														If[CurrentValue["MouseOver"], GrayLevel[0.95], GrayLevel[1] ]
													]],
												detailNumber = 1, Appearance -> {"Default" -> None, "Pressed" -> None}, Tooltip -> "First Slide", TooltipDelay -> .25],
											Button[
												Panel[Style["\[LeftPointer]", FontSize -> 24, FontColor -> GrayLevel[0.5]], Appearance -> None, ImageSize -> {15, 15}, Alignment -> Center,
													Background ->
													Dynamic[
														If[CurrentValue["MouseOver"], GrayLevel[0.95], GrayLevel[1] ]
													]],
												detailNumber--; If[detailNumber < 1, detailNumber = 1], Appearance -> {"Default" -> None, "Pressed" -> None}, Tooltip -> "Previous Slide", TooltipDelay -> .25],
											Button[
												Panel[Style["\[RightPointer]", FontSize -> 24, FontColor -> GrayLevel[0.5]], Appearance -> None, ImageSize -> {15, 15}, Alignment -> Center,
													Background ->
													Dynamic[
														If[CurrentValue["MouseOver"], GrayLevel[0.95], GrayLevel[1] ]
													]],
												detailNumber++; If[detailNumber > numberOfDetails, detailNumber = numberOfDetails], Appearance -> {"Default" -> None, "Pressed" -> None}, Tooltip -> "Next Slide", TooltipDelay -> .25],
											Button[
												Panel[Style["\[LastPage]", FontSize -> 24, FontColor -> GrayLevel[0.5]], Appearance -> None, ImageSize -> {15, 15}, Alignment -> Center,
													Background ->
													Dynamic[
														If[CurrentValue["MouseOver"], GrayLevel[0.95], GrayLevel[1] ]
													]],
												detailNumber = numberOfDetails, Appearance -> {"Default" -> None, "Pressed" -> None}, Tooltip -> "Last Slide", TooltipDelay -> .25]
										}]
								}, Spacer[180], Alignment -> {Left, Baseline}]
								}, {
								Button[
									MouseAppearance[
										Dynamic@
											With[{themeImage =
												StringJoin[
													ToString[Setting[themeName]],
													"-detail-",
													ToString[Setting[fontset]],
													"-",
													ToString[Setting[colorset]],
													"-",
													ToString[Setting[detailNumber]],
													".png"
													]
												},
												Framed[
													Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, themeImage]],
													ImageSize -> {Automatic, 162},
													Appearance -> None,
													ImageMargins -> 0,
													FrameMargins -> 0,
													FrameStyle -> GrayLevel[.6],
													Background -> None
													(*Background -> GrayLevel[.6]*)
													]
											], "LinkHand"
									],
									If[GreaterEqual[detailNumber, numberOfDetails], detailNumber = 1, detailNumber++],
									FrameMargins -> 0,
									ImageMargins -> 0,
									Appearance -> None,
									ContentPadding -> False,
									Tooltip ->
										ToBoxes@
											Pane[
											 Grid[{{
											    "Theme: ", Dynamic[themeName]
											    }, {
											    "FontSet: ", Dynamic[fontset]
											    }, {
											    "ColorSet: ", Dynamic[colorset]
											    }}, Alignment -> Left]
											 ],
									TooltipDelay -> .25
								]
							}}, Alignment -> {Center, Top}, Spacings -> 0, Frame-> None
							], ImageSize -> Full, Alignment -> Center,
								FrameMargins -> 0, ImageMargins -> 0, Spacings -> 0
							]
						},
						Dynamic[CurrentValue[System`$FrontEnd, {
							System`PrivateFrontEndOptions, "InterfaceSettings", "PresenterTools", "ThemeOptionsPalette", "ShowPreview"}, True]],
							FrameMargins -> 0, ImageMargins -> 0
						],
					SpanFromLeft
(*					}}, Alignment -> {Center, Top}, Spacings -> {1.5, {0, 1, 0, 0}}, Frame -> None],
						SpanFromLeft,
						SpanFromLeft*)
					}, {
						Item[
						(***************************************************
						   Create presentation based on user's theme choices
						   and return presentation to user
						   *)
						   ChoiceButtons[{
								Style[Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "ThemeOptionsApply"],
									FontColor :> FEPrivate`If[FEPrivate`$OperatingSystem === "MacOSX", GrayLevel[1], Automatic]],
								Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "CancelButtonText"]
						   },
							{
								With[{themename = themeName, syledef = allThemes[[Key[themeName], Key["StyleDefinitions"]]],
									fonts = fontset, colors = colorset,
									backgroundName = StringJoin["Background-", themeName, "-", colorset, ".png"]
									},
										(

											If[themename =!= previoustheme,
												ChangeNotebookStyleSheet[nbObj, syledef];
											];
											UpdateNotebookStyleHints[nbObj, {
												Rule["FontSet", fonts],
												Rule["ColorSet", colors]
												}];

											(* returnValue = list of options to add to style sheet *)

											If[True, (*TitleBackgroundQ[themename] === True,*)

												stylesheet =
													PresenterTools`Styles`UpdateStylesheet[
														{{
															"FirstSlide",
															"Working",
															None,
															{
																PrivateCellOptions->{
																	"PagewiseNotebookBaseStyle" -> {
																		BackgroundAppearance -> FrontEnd`FileName[{"PresenterTools"}, backgroundName]
																	}
																}
															}
														}},
														CurrentValue[nbObj, StyleDefinitions]
													],
											(* ELSE *)
												stylesheet =
													PresenterTools`Styles`UpdateStylesheet[
														{{
															"FirstSlide",
															"Working",
															None,
															{
																PrivateCellOptions->{
																	"PagewiseNotebookBaseStyle" -> {}
																}
															}
														}},
														CurrentValue[nbObj, StyleDefinitions]
													]
													];

											If[Head[stylesheet] === Notebook,
													SetOptions[nbObj, StyleDefinitions -> stylesheet]
											];
											(*CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "ThemeOptionsPalette", "PaletteNumber"}] = 0;*)
											CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "Dialogs", "ThemeOptions"}] = 0;
											CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "Theme", "Name"}] = themename;
											CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "Theme", "FontSetName"}] = fonts;
											CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "Theme", "ColorSetName"}] = colors;

											NotebookClose[ButtonNotebook[]];
										)
									],
								(*CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "ThemeOptionsPalette", "PaletteNumber"}] = 0;*)
								CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "Dialogs", "ThemeOptions"}] = 0;
								DialogReturn[$Canceled]
							  }(*, Appearance -> "Pressed"*)
							], Alignment -> Right],
						SpanFromLeft
					}}, Alignment -> Center, Spacings -> {1, {4, 4, 1.5, 2.25, 0.75, 1, 2, 1, 4, {1}}},
						FrameStyle -> RGBColor[0.898039, 0.898039, 0.898039], Dividers -> {False, {2 -> True, 7 -> True, 8 -> True}}
				], Background -> GrayLevel[1],
					FrameMargins -> {{5, 0}, {30, 30}},
					ImageMargins -> 0, (*{{20, 20}, {20, 20}}*)
					ContentPadding -> False,
					Appearance -> None
				],
			Initialization :> (
				Needs["PresenterTools`"];
			),
			Deinitialization :> (
				CurrentValue[System`$FrontEnd, {
					System`PrivateFrontEndOptions, "InterfaceSettings", "PresenterTools", "ThemeOptionsPalette", "WindowMargins"}] = CurrentValue[EvaluationNotebook[], WindowMargins]
			)
			]],

			CellMargins -> 0,
			ShowCellBracket -> False,
			TextAlignment -> Center,
			CellFrame -> False,
			Background -> GrayLevel[1]
			]
		},
		WindowSize-> {600, All},
		WindowFrame -> "Palette",
		WindowFrameElements -> {"CloseBox", "ZoomBox", "MinimizeBox"},
		WindowElements -> {},
		WindowMargins -> Automatic,
		WindowTitle -> Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "ThemeOptionsMenuItem"],
		Background -> GrayLevel[1],
(*		Evaluator -> "System", *)
		Saveable -> False,
		Deployed -> True,
		Editable -> False,
		ShowCellBracket -> False,
		NotebookEventActions->{
			"ReturnKeyDown" :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
			"EscapeKeyDown" :> FE`Evaluate[FEPrivate`FindAndClickCancelButton[]],
			"WindowClose" :> (FE`Evaluate[FEPrivate`FindAndClickCancelButton[]];
								(*CurrentValue[EvaluationNotebook[], {TaggingRules, "PresenterSettings", "ThemeOptionsPalette", "PaletteNumber"}] = 0;*)
								CurrentValue[EvaluationNotebook[], {TaggingRules, "PresenterSettings", "Dialogs", "ThemeOptions"}] = 0;
								)
				}
(*
				,
			NotebookDynamicExpression :> (
				Message[foo::info, $CellContext`nb$];
				If[Not[MemberQ[Notebooks[], $CellContext`nb$]],
					Message[ThemeOptionsDialog::info, $CellContext`nb$];
					(* NotebookClose[EvaluationNotebook[]] *),
				(* ELSE *)
					Message[bar::arg, $CellContext`nb$];
					Message[bar::info, $CellContext`nb$];
					]
			)
*)
		]
	]


];
ThemeOptionsDialog[a___]:= Message[ThemeOptionsDialog::argx, {a}];
ThemeOptionsDialog::argx = "`1`";
ThemeOptionsDialog::info = "`1`";




(*******************************************************

 UpdateThemeNotebook

	in: xxx
	out: yyy

*******************************************************)

UpdateThemeNotebook[nbObj_, themename_String, fonts_String, colors_String] :=
Module[{previoustheme, stylesheet, backgroundName, syledef},

	previoustheme = CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "Theme", "Name"}, "Default"];
	backgroundName = StringJoin["Background-", themename, "-", colors, ".png"];
	syledef = PresenterTools`GetThemeInformation[themename, "StyleDefinitions"];

	If[themename =!= previoustheme, ChangeNotebookStyleSheet[nbObj, syledef]];

	UpdateNotebookStyleHints[nbObj, {Rule["FontSet", fonts], Rule["ColorSet", colors]}];
	(*returnValue=list of options to add to style sheet*)
	If[TitleBackgroundQ[themename] === True,
		With[{bkgndname = backgroundName},
			stylesheet = PresenterTools`Styles`UpdateStylesheet[{
				{"FirstSlide", "Working", None, {PrivateCellOptions -> {"PagewiseNotebookBaseStyle" -> {BackgroundAppearance ->FrontEnd`FileName[{"PresenterTools"}, bkgndname]}}}}}, CurrentValue[nbObj, StyleDefinitions]]
		];
		SetOptions[nbObj, StyleDefinitions -> stylesheet],
	(*ELSE*)
		stylesheet = PresenterTools`Styles`UpdateStylesheet[{{"FirstSlide", "Working", None, {PrivateCellOptions -> {"PagewiseNotebookBaseStyle" -> {}}}}}, CurrentValue[nbObj, StyleDefinitions]];
		SetOptions[nbObj, StyleDefinitions -> stylesheet]
	];
	(*CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "ThemeOptionsPalette", "PaletteNumber"}] = 0;*)
	CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "Dialogs", "ThemeOptions"}] = 0;
	CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "Theme", "Name"}] = themename;
	CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "Theme", "FontSetName"}] = fonts;
	CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "Theme", "ColorSetName"}] = colors;
];
UpdateThemeNotebook[a___]:= Message[UpdateThemeNotebook::argx, {a}];
UpdateThemeNotebook::argx = "`1`";
UpdateThemeNotebook::info = "`1`";
















(**************************************

 UpdateStyleDefinitionsDialog
	Dialog for user to choose which options to apply to stylesheet.
	in: Options
	out: DialogInput

***************************************)

UpdateStyleDefinitionsDialog[list_List]:= UpdateStyleDefinitionsDialog[InputNotebook[], list];

(* per sw, only display changes to the "currently selected
   cell style", instead of displaying all changes for every
   cell style that contains a change.
   *)
UpdateStyleDefinitionsDialog[nbObj_NotebookObject, userList:{__?OptionQ}, cellStyle_:None]:=
Module[{returnValue, styleList (*, optionList, numberOfChanges, optionsWCheckbox, displayOptions*)},

	styleList = Merge[userList, Flatten];
	styleList = {{cellStyle, styleList[cellStyle]}};

	styleList = styleList /. {sty_String, opts_List} :> ({sty, #} & /@ opts);
	styleList = Flatten[styleList, 1];

With[{list = styleList, $CellContext`nb$ = nbObj, style = cellStyle},

	returnValue =
	CreateDialog[
		DynamicModule[{optionList = list, numberOfChanges, optionsWCheckbox, displayOptions},

			numberOfChanges = Length[optionList];

			(* Add checkboxes to select which options are added to stylesheet *)
			(*( $CellContext`optionsToApply[#]:= True) & /@ Range[Length[optionList]];*)
			optionsWCheckbox =
				With[{val = #, option = Part[optionList, #, 2]},
					(
						PresenterTools`Private`optionsToApply[val]:= True;
						Flatten[{
							Style[
								Checkbox[Dynamic[PresenterTools`Private`optionsToApply[val]]], CellTags -> "label"], option
							}]
					)] & /@ Range[numberOfChanges];
(*
				PrependTo[optionsWCheckbox, {
							Style["Add option", FontSize->Inherited, FontWeight->Bold, FontFamily :> CurrentValue["PanelFontFamily"], CellTags -> "label"],
							Style["Style", FontSize->Inherited, FontWeight->Bold, FontFamily :> CurrentValue["PanelFontFamily"], CellTags -> "label"],
							Style["Option", FontSize->Inherited, FontWeight->Bold, FontFamily :> CurrentValue["PanelFontFamily"], CellTags -> "label"]
						}];
*)
			(* modify some RHS to avoid displaying raw calculations *)
			displayOptions = optionsWCheckbox;
			displayOptions = displayOptions /. (Rule|RuleDelayed)[CellMargins, _] :> "CellMargins \[Rule] Dynamic[...]";
			displayOptions = displayOptions /. (Rule|RuleDelayed)[FontSize, _] :> "FontSize \[Rule] Dynamic[...]";
			displayOptions = displayOptions /. (Rule|RuleDelayed)[FontVariationsUnderline, ul_] :> Row[{"Underline -> ", ul}];
			displayOptions = displayOptions /. (Rule|RuleDelayed)[FontColor, fc_] :> (FontColor -> Setting[fc]);

			Panel[
				Grid[{
					{
						Pane[
							Style[
(*							Row[{Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "UpdateAllStylesWithLabel"]}],*)
							Row[{
								Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "UpdateAllStylesLabel"],
								Style[style, Bold],
								Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "UpdateCellsWithLabel"]
								}],
								"Text",
								FontSize->13,
								FontFamily :> CurrentValue["PanelFontFamily"]
							], ImageSize -> {400, Automatic}, Alignment -> Left],
						SpanFromLeft

					},
					(* ROW ONE *)

					{
						(* Remove CellMargins due to the enormous math RHS *)
						Framed[
							Panel[
								Grid[displayOptions, Alignment->Left, Spacings -> {1.5, {0.5, 0.25}}],
								ImageMargins -> 0,
								FrameMargins -> 10,
								ImageSize -> {400, Automatic},
								Appearance -> None,
								Background -> GrayLevel[1]
							],
							ContentPadding -> False,
							FrameMargins -> 3,
							ImageMargins -> 0,
							FrameStyle -> GrayLevel[0.85]
						],
						SpanFromLeft
					},

					(* ROW TWO *)
					{
						" ",
(*						Button[Pane["Do Not Update"], DialogReturn[(*Options to add to stylesheet*)None], Appearance -> "DialogBox"],*)
						Item[
							ChoiceButtons[{
								Style[
									Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "Update"],
									FontColor :> FEPrivate`If[FEPrivate`$OperatingSystem === "MacOSX", GrayLevel[1], Automatic]],
									Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "CancelButtonText"]
								}, {
									(
										DialogReturn[
											(*Options to add to stylesheet*)
											ApplyUpdateStylesDialogResults[$CellContext`nb$, {
															PresenterTools`Private`optionsToApply[#] & /@ Range[numberOfChanges], optionList}, {list, optionList}]
										]
									),
									DialogReturn[$Canceled]
								}
							], Alignment -> Right
						]
					}
					(* ROW THREE *)
					}, Alignment -> {Left, {Top}}, Spacings -> {1, {2, 1, 2}}, Frame -> None
				],
				FrameMargins -> {{20, 20}, {10, 20}},
				Background -> GrayLevel[.9]
				],
				Initialization :> (
					SetSelectedNotebook[EvaluationNotebook[]]
				)
			],
			WindowFloating -> True,
			Modal -> False,
			NotebookEventActions -> {
				"ReturnKeyDown" :> FE`Evaluate[FEPrivate`FindAndClickDefaultButton[]],
				"EscapeKeyDown" :> FE`Evaluate[FEPrivate`FindAndClickCancelButton[]],
				"WindowClose" :> FE`Evaluate[FEPrivate`FindAndClickCancelButton[]]
			}
		]
	]
]


ApplyUpdateStylesDialogResults[nbObj_, $Canceled, {originalList_, appliedList_}, opts___]:= $Canceled;

ApplyUpdateStylesDialogResults[nbObj_, return_, {originalList_, appliedList_}, opts___]:=
Module[{styleNames, styleEnvSlide, styleEnvScroll, optionChanges, changes, returnValue},

	(* user canceled dialog: leave StyleUpdateList intact *)
	If[MatchQ[return, $Canceled], Return[$Canceled]];

	(* user chooses to leave formatting in cell (unchecked all options) and not to update stylesheet: reset StyleUpdateList *)
	If[MatchQ[return, None],
		CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "UserModifications"}] = {};
		Return[None]
	];

	changes = Transpose[return];
	returnValue = Cases[changes, {True, o_} :> o, Infinity];

	(* user chooses to update stylesheet with option changes: *)
	(* reset StyleUpdateList and mark cells that contain options that user doesn't want updated in stylesheet *)
(* TODO: UpdateStylesDialog: mark (add) unchanged cells in tagging rules *)
(* TODO: UpdateStylesDialog: use Associations to potentially avoid hackery *)
	(* remove lables from dialog *)
	returnValue = returnValue /. Style[__, CellTags -> "label"] :> Nothing;
	returnValue = returnValue /. {a___, {}, b___} :> {a, b};
(*	returnValue = List[First[#], "Slideshow Working", None, Rest[#]]& /@ returnValue;*)
	returnValue = List[First[#], "Working", None, Rest[#]]& /@ returnValue;

	(* not all option changes should be applied to "Working" environment.
		Some option changes should only be applied to "Slideshow Working". (note that both
		"Slideshow Presentation" and "Scrolling Presentation" inherit from "Scrolling Working")
	    Current exceptions are those that would strongly effect display in Working/Printout/etc.
	    environments, such as FontSize and Cell Margins
	   *)
    returnValue = returnValue /. {s_, "Working", v_, r : {(Rule | RuleDelayed)[opt_ /; MemberQ[{CellMargins, FontSize}, opt], ___], o___}, rest___} :>
    					{s, "Slideshow Working", v, r, rest};

	styleNames = Union @ Cases[returnValue, {sty_String, _String, __} :> sty, Infinity];

	(* add environments if not already present *)
	styleEnvWorkSlide = {#, "Slideshow Working", None, (*StyleDefinitions -> StyleData[#, "Slideshow Working"],*) {} } & /@ styleNames;
	styleEnvSlide = {#, "Slideshow Presentation", StyleDefinitions -> StyleData[#, "Slideshow Working"], {} } & /@ styleNames;
	styleEnvScroll = {#, "Scrolling Presentation", StyleDefinitions -> StyleData[#, "Slideshow Working"], {} } & /@ styleNames;
	returnValue = Join[returnValue, styleEnvWorkSlide, styleEnvSlide, styleEnvScroll];

	(* returnValue = list of options to add to style sheet *)
	With[{stylesheet = PresenterTools`Styles`UpdateStylesheet[returnValue, CurrentValue[nbObj, StyleDefinitions]] },
		SetOptions[nbObj, StyleDefinitions -> stylesheet]
	];

	(* RESET StyleOptionUpdate list *)
(*	CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "UserModifications"}] = {}; *)
	CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "UserModifications"}] = Complement[originalList, appliedList];
(*	SetStyleUpdateList[nbObj, Complement[originalList, appliedList]];*)

	(* remove options on cells which have been added to stylesheet *)
	(* TODO: UpdateStylesDialog: rewrite for clarity and efficiency *)
	optionChanges = Cases[returnValue, {sty_String, _String, None, o_List, ___} :> {sty, o}];
	optionChanges = optionChanges /. {sty_String, {Rule[opt_, val_]}} :> {sty, {Rule[opt, Inherited]}};
	optionChanges = optionChanges /. {sty_String, {RuleDelayed[opt_, val_]}} :> {sty, {RuleDelayed[opt, Inherited]}};

	Block[{cells, options},
		cells = Cells[nbObj, CellStyle -> #];
		options = Union@Flatten[Cases[optionChanges, {#, o_} :> o]];
		SetOptions[#, options] & /@ cells
	] & /@ styleNames;

];
UpdateStyleDefinitionsDialog::list = "OptionList: `1`";







(*******************************************************

 GetThemeNames

	in: None
	out: List

*******************************************************)

GetThemeNames[]:=
	PresenterTools`$ThemeNames;





(*******************************************************

 GetThemeColorMenu

	in: String ("Default", "Garnet", "Facet", ...)
	out: Row colors

*******************************************************)

GetThemeColorMenu[name_String]:= GetThemeColorMenu[name, PresenterTools`GetThemeInformation[name, "ColorSet"]];

GetThemeColorMenu[name_String, colors_?OptionQ]:=
Module[{},

	Rule[name,
		Row @
			Riffle[
				Take[ Flatten[{colors /. Rule[_, co_] :> Panel[" ", Background -> co, ImageSize -> {31, 15}]}], 3],
				Spacer[1]
			]
	]
];
GetThemeColorMenu[a___]:= Message[GetThemeColorMenu::argx, {a}];
GetThemeColorMenu::argx = "`1`";
GetThemeColorMenu::info = "`1`";



(*******************************************************

 GetThemeFontsMenu

	in: String ("Default", "Garnet", "Facet", ...)
	out: Row of Style[]s

*******************************************************)

GetThemeFontMenu[name_String]:= GetThemeFontMenu[name, PresenterTools`GetThemeInformation[name, "FontSet"]];

GetThemeFontMenu[name_String, fonts_?OptionQ]:=
Module[{},

	Rule[name,
		Row @
			Riffle[
				Take[ Flatten[{fonts /. Rule[_, f_] :> Style[f, 13, FontFamily -> f]}], 2],
				Style[", ", FontFamily -> ".SF NS Text"]]
			]
];
GetThemeFontMenu[a___]:= Message[GetThemeFontMenu::argx, {a}];
GetThemeFontMenu::argx = "`1`";
GetThemeFontMenu::info = "`1`";





(**************************************

 CreateThemeThumbnailButton
	Generate thumbnail of theme
	in: Name of theme
	out: Button

***************************************)

CreateThemeThumbnailButton[theme_]:=
Button[
	Panel[
		MakeNotebookThumbnailImage["Default.nb"],
		FrameMargins -> 10, Background -> GrayLevel[.3]],
	mainTheme=theme;SetBorderHightlight[theme];, Appearance -> None
];


FlattenCellGroups[nb_NotebookObject?NotebookOpenQ] := FlattenCellGroups[NotebookGet@nb]
FlattenCellGroups[Notebook[cells_, opts___]] := Notebook[FlattenCellGroups[cells], opts]
FlattenCellGroups[{c__Cell}] := Flatten[FlattenCellGroups /@ {c}]
FlattenCellGroups[Cell[CellGroupData[{c__Cell}, ___], ___]] := FlattenCellGroups /@ {c}
FlattenCellGroups[x_] := x




(**************************************

 MakeNotebookImageFromStyleDefinitions
	Generate thumbnail of theme
	in: StyleDefinitions of theme
	out: Raster

***************************************)

MakeNotebookImageFromStyleDefinitions[styleDefinitions_, nbopts___?OptionQ]:=
Module[{nbObj, image, opts},
	opts = Join[{nbopts}, {CellLabelAutoDelete -> False, ShowCellLabel -> True, WindowSize -> {300, 275}, ScreenStyleEnvironment -> "SlideShow"}];
	nbObj = NotebookPut[
		Notebook[{
		   Cell["Title", "Title"],
		   Cell["Section", "Section"],
		   Cell[BoxData[RowBox[{"\[Integral]", RowBox[{RowBox[{"Sqrt", "[", RowBox[{"x", "+", RowBox[{"Sqrt", "[", "x", "]"}]}], "]"}], RowBox[{"\[DifferentialD]", "x"}]}]}]], "Input", CellLabel -> "In[1]:="]
		}, StyleDefinitions -> styleDefinitions, Sequence@@opts]];
	image=Rasterize[NotebookGet[nbObj]];
	NotebookClose[nbObj];
	image
]




(**************************************

 CreateColorThemeDialog
	Create color theme dialog
	in: None
	out: Association of colors

***************************************)

CreateColorThemeDialog[]:=
Module[{},

DialogInput[DynamicModule[{colorSet=""},

	Pane[
		Grid[{
			{
				Item[Pane[Style["Color set name:", FontSize -> 13, Bold, FontFamily :> CurrentValue["PanelFontFamily"]], Alignment -> Center, Appearence -> None], Alignment -> Left],
				SpanFromLeft
			},
			{
				InputField[Dynamic[colorSet], String, ImageSize -> Full],
				SpanFromLeft,
				SpanFromLeft
			},
			{
				Pane[Style["Type", FontSize -> 13, Bold, FontFamily :> CurrentValue["PanelFontFamily"]], Alignment -> Center, ImageSize -> 70, Appearence -> None],
				Pane[Style["Color", FontSize -> 13, Bold, FontFamily :> CurrentValue["PanelFontFamily"]], Alignment -> Center, ImageSize -> 70, Appearence -> None],
				Pane[" ", ImageSize -> 60, Appearence -> None]
			},
			{
				Style["Heading", FontSize -> 12, FontFamily :> CurrentValue["PanelFontFamily"]],
				ColorSetter[RGBColor[0.868147, 0.32929, 0.11931], ImageSize -> {50, 25}],
				Tooltip[Button[ Style["x", FontSize -> 14, FontColor -> GrayLevel[0.6], FontFamily :> CurrentValue["PanelFontFamily"]], Appearance -> None], "Delete"]
			},
			{
				Style["Text", FontSize -> 12, FontFamily :> CurrentValue["PanelFontFamily"]],
				ColorSetter[RGBColor[0.211414, 0.553719, 0.920211], ImageSize -> {50, 25}],
				Tooltip[Button[ Style["x", FontSize -> 14, FontColor -> GrayLevel[0.6], FontFamily :> CurrentValue["PanelFontFamily"]], Appearance -> None], "Delete"]
			},
			{
				Style["Code", FontSize -> 12, FontFamily :> CurrentValue["PanelFontFamily"]],
				ColorSetter[RGBColor[0.180255, 0.542229, 0.246693], ImageSize -> {50, 25}],
				Tooltip[Button[ Style["x", FontSize -> 14, FontColor -> GrayLevel[0.6], FontFamily :> CurrentValue["PanelFontFamily"]], Appearance -> None], "Delete"]
			}, (*
			{
				Item[
					PaneSelector[{True ->
					Button[
						Panel["Add Color", Background -> GrayLevel[.9], ImageSize -> {210, 30}, Alignment -> Center],
						MathLink`CallFrontEnd[
						 FrontEnd`BoxReferenceFind[
						  FE`BoxReference[EvaluationNotebook[], {{"AddColorButton"}},
						   FE`BoxOffset -> {FE`BoxChild[1]}]]];
						FrontEndExecute[{FrontEnd`FrontEndToken[EvaluationNotebook[], "MovePrevious"]}];
						FrontEndExecute[{FrontEnd`FrontEndToken[EvaluationNotebook[], "NewRow"]}],
						Appearance -> None, BoxID -> "AddColorButton"],
						False ->
						Button[
						Panel["Add Color", Background -> GrayLevel[.8], ImageSize -> {210, 30}, Alignment -> Center],
						MathLink`CallFrontEnd[
						 FrontEnd`BoxReferenceFind[
						  FE`BoxReference[EvaluationNotebook[], {{"AddColorButton"}},
						   FE`BoxOffset -> {FE`BoxChild[1]}]]];
						FrontEndExecute[{FrontEnd`FrontEndToken[EvaluationNotebook[], "MovePrevious"]}];
						FrontEndExecute[{FrontEnd`FrontEndToken[EvaluationNotebook[], "NewRow"]}],
						Appearance -> None, BoxID -> "AddColorButton"]}, Dynamic[CurrentValue["MouseOver"]] ],
					Alignment -> {Left, Bottom}
				],
				SpanFromLeft,
				SpanFromLeft
			},*)
			{
				Item[
					Pane[
						Grid[{{
							CancelButton[],
							DefaultButton["Create",
								DialogReturn[
									(* Association uses three theme Keys, {main, font, color} *)
									Association["ColorSetName" -> colorSet]
								]
							]
						}}, Alignment -> {Left, Center}, Spacings -> {1, 0}], ImageMargins -> {{0, 0}, {5, 20}}],
					Alignment -> Right, Background -> GrayLevel[1]
				],
				SpanFromLeft,
				SpanFromLeft
			}
			}, Frame -> All,  FrameStyle -> Directive[GrayLevel[1]], (* Dividers -> {True, False}, *)
						Spacings -> {{ Automatic}, {2, -2 -> 3}}, Alignment -> {Center, Center},
						Background -> GrayLevel[1]],
		ImageSize -> 225, FrameMargins -> {{0, 0}, {10, 0}}
	]], WindowTitle -> "New Color Set"
]];




(*******************************************************

 ToggleNavigationBar
 toggles shide/hide within Slideshow Presentation' ScreenStyleEnvironment

	in: notebook object
	out: None

*******************************************************)

ToggleNavigationBar[nbObj_]:=
Module[{navQ},

	navQ = CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "ShowNavigationBar"}, False];

	If[TrueQ[navQ],
		Set[CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "ShowNavigationBar"}], False],
	(* ELSE *)
		Set[CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "ShowNavigationBar"}], True]
	]
]
ToggleNavigationBar[a___]:= Message[ToggleNavigationBar::argx, {a}];
ToggleNavigationBar::argx = "`1`";
ToggleNavigationBar::info = "`1`";





(*******************************************************

 SlideNavigationPopupMenu

	in: xxx
	out: yyy

*******************************************************)

SlideNavigationPopupMenu[nbObj_]:=
Module[{},

With[{cells =
   Cells[nbObj,
    CellStyle -> Alternatives @@ Flatten@Join[{"SlideShowNavigationBar"},
       CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "SlideBreakStyles"}, {}]]]
  },
	 PopupMenu[Dynamic[0,
	   With[{$CellContext`nb$ = nbObj},
	     (
	      SelectionMove[$CellContext`nb$, Before, Notebook,
	       AutoScroll -> False];
	      (*taken from pre-11.2 slide navigator popup*)

	      SelectionMove[#, All, Cell, AutoScroll -> False];
	      SelectionMove[$CellContext`nb$, After, Cell,
	       AutoScroll -> False];
	      NotebookWrite[$CellContext`nb$,
	       Cell["", Deletable -> True, ShowCellBracket -> False], All];
	      NotebookDelete[$CellContext`nb$];
	      SelectionMove[$CellContext`nb$, Next, Cell];
	      SelectionMove[$CellContext`nb$, Before, Cell];
	      )
	     ] &
	   ], Rule[Part[cells, #], #] & /@ Range[Length@cells]
	  ]
	 ]
];
SlideNavigationPopupMenu[a___]:= Message[SlideNavigationPopupMenu::argx, {a}];
SlideNavigationPopupMenu::argx = "`1`";
SlideNavigationPopupMenu::info = "`1`";







(**************************************

 CreateNotebookScreenshot
	Create image of notebook
	in: Notebook / NotebookObject
	out: Graphics

***************************************)

CreateNotebookScreenshot[nbExpr_Notebook] :=
Module[{},
	Replace[nbExpr, a : Except[(_Graphics | _Graphics3D | _Graph | _Image)] :> Rasterize[a]]
];

CreateNotebookScreenshot[nb_Notebook, "GUIKit"] :=
Module[{nbObj, res},
	nbObj = NotebookPut@nb;
	res = CreateNotebookScreenshot[nbObj];
	NotebookClose[nbObj];
	res
];
CreateNotebookScreenshot[nb_NotebookObject, "GUIKit"] :=
Module[{left, top, size, opts = AbsoluteOptions@nb},
	{left, top} = WindowMargins /. opts // Diagonal;
	size = WindowSize /. opts;
	Needs["GUIKit`"];
	GUIKit`GUIScreenShot[{
		{left + 10, left + size[[1]] - 10},
		{top + 75, top + size[[2]] + 30}
	}]
];


CreateThumbnailList[nbObj_NotebookObject]:=
Module[{},

	SetOptions[nbObj, ScreenStyleEnvironment -> "Slideshow Working"];
	SelectionMove[nbObj, All, Notebook];
	FrontEndTokenExecute[nbObj, "SelectionConvert", "Bitmap"];
	SetOptions[nbObj, ScreenStyleEnvironment -> "Working"];
	CreateThumbnailList[NotebookGet[nbObj], CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "SlideBreakStyles"}, {}] ]
];
CreateThumbnailList[nbExpr_Notebook, slideBreakStyles_List] :=
Module[{expr = nbExpr, opts, cellList},
	expr = First[nbExpr];
	opts = Rest[List @@ nbExpr];
	expr = FlattenCellGroups[expr];
	(* how many cells? *)
	(* Print[Length[expr]]; *)
(*	cellList = SplitBy[expr, Part[#, 2] === "SlideShowNavigationBar" &]; *)
	cellList = SplitBy[expr, MemberQ[slideBreakStyles, Part[#, 2]] &];
	cellList = DeleteCases[cellList, {Cell[_, "SlideShowNavigationBar", ___]}];

	CreateNotebookThumbnail[Notebook[#, Background -> GrayLevel[0.8], Sequence @@ opts]] & /@ cellList
]

(********************************************)
iSlideBreakCellQ[Cell[_, sty_String /; StringMatchQ[sty, Alternatives@@{"Title", "Section", "SlideShowNavigationBar"}], ___]] := True;
iSlideBreakCellQ[___] := False

iCombineCellListToSlides[{{o:Cell[_, sty_String/; StringMatchQ[sty, Alternatives@@{"Title", "Section", "SlideShowNavigationBar"}], ___]}, t:{Cell[_, sty_String /; ! StringMatchQ[sty, Alternatives@@{"Title", "Section", "SlideShowNavigationBar"}], ___], ___}, r___}] :=
Module[{},
	AppendTo[PresenterTools`Private`iCellGroupList, Cell[CellGroupData[Flatten[{o, t}], Open]]];
	iCombineCellListToSlides[{r}]
];
iCombineCellListToSlides[{}] := Nothing;

CreateThumbnailList[nbObj_NotebookObject, "Test"] :=
Module[{cells = Cells[nbObj], dummyCells, opts},
	opts = Options[nbObj];
Print[Length[cells]];
	dummyCells = NotebookRead[cells] /. Cell[_, sty_String, ___] :> Cell[sty, sty];
	test = SplitBy[dummyCells, iSlideBreakCellQ[#] &];
Print[test];
	iCombineCellListToSlides @ test;
Print[Length@PresenterTools`Private`iCellGroupList];
	CreateNotebookThumbnail[Notebook[{#}, Background -> GrayLevel[0.8], Sequence @@ opts]] & /@ PresenterTools`Private`iCellGroupList
]

(********************************************)






(**************************************

 CreateNotebookThumbnail
	Create thumbnail of notebook
	Calls CreateNotebookScreenshot

	in: Notebook
	out: Graphics

***************************************)

CreateNotebookThumbnail[expr_Notebook, num_:100, color_:GrayLevel[1], options___?OptionQ]:=
Module[{res, opts = options},

	opts = If[TrueQ[Length[{opts}] > 0], {opts}, {FrameStyle -> GrayLevel[.85]}];
	res = CreateNotebookScreenshot[expr];
	Framed[
		ImagePad[ImageCrop[ImageResize[res, num], {num, num*0.7}], 3, color],
		Sequence@@opts
	]
]




(*******************************************************

 GetNotebookThumbnail

	in: xxx
	out: yyy

*******************************************************)
Options[GetNotebookThumbnail] = {
	"BackgroundColor" -> GrayLevel[1]
}

GetNotebookThumbnail[list_List, imageSize_: {100, 70}, opts___?OptionQ]:= GetNotebookThumbnail[#, imageSize, opts]& /@ list;
(*
GetNotebookThumbnail[notebookExpr_Notebook /; FreeQ[notebookExpr, Cell[_, "SlideShowNavigationBar", "FirstSlide", ___]], imageSize_: {100, 70}] :=
	AlternateGetNotebookThumbnail[notebookExpr, imageSize];
*)
GetNotebookThumbnail[notebookExpr_Notebook, imageSize_: {100, 70}, opts___?OptionQ] :=
Module[{img, nbExpr = notebookExpr, desiredWidth, desiredHeight, backgroundColor},

	desiredWidth = Part[imageSize, 1];
	desiredHeight = Part[imageSize, 2];
	backgroundColor = "BackgroundColor" /. {opts} /. Options[GetNotebookThumbnail];

	nbExpr = MakeThumbnailNotebook[nbExpr(*, "WindowSize" -> imageSize*)];

	img = FrontEndExecute[FrontEnd`ExportPacket[nbExpr, "PNG"]][[1]];

	If[MatchQ[img, $Failed],
		img = Rasterize["", Background -> backgroundColor, ImageSize -> imageSize],
	(* ELSE *)
		img = ImportString[img, "PNG"];

	If[(Head[img] === Image),
			img = ImageResize[img, desiredWidth];
			If[Part[ImageDimensions[img], 2] < desiredHeight,
				img = ImagePad[img, {{0, 0}, {desiredHeight, 0}}, backgroundColor];
				img = ImageCrop[img, imageSize, Bottom, Padding -> Automatic],
		(* ELSE *)
				img = ImageCrop[img, imageSize, Bottom, Padding -> Automatic]
		]
	]
	];

	img
]


AlternateGetNotebookThumbnail[slide_, pnb_, outline_, imageSize_: {100, 70}] :=
	Module[{outlineMargins, outlineWindowSize, positionThumbnailQ, thumbnailmargins, leftmargin, bottommargin, nb, img},
		(* Load Java early so that when NotebookPut is called below, CurrentNotebookImage is ready  *)
		CurrentNotebookImage[pnb];
		With[{pnb1 = pnb},
			outlineMargins = AbsoluteCurrentValue[outline, WindowMargins];
			outlineWindowSize = CurrentValue[outline, WindowSize];
			positionThumbnailQ = AllTrue[Flatten@outlineMargins, NonNegative] && AllTrue[Transpose[{outlineWindowSize, {200, 140}}], #[[1]] > #[[2]] &];
			thumbnailmargins = If[positionThumbnailQ,
				{leftmargin, bottommargin} = (outlineMargins /. {{l_, _}, {b_, _}} :> {l + Round[outlineWindowSize[[1]]/2] - 100, b + Round[outlineWindowSize[[2]]/2] - 70});
				{{leftmargin, Automatic}, {bottommargin, Automatic}},
				{{Automatic, Automatic}, {Automatic, Automatic}}];
			nb = NotebookPut[((slide /. Cell[con_, "SlideShowNavigationBar", stys___, opts___?OptionQ] :> Cell[con, "SlideShowNavigationBar", stys, PageBreakAbove -> False,
				opts]) /. Notebook[a_, b_List] :> Join[DeleteCases[Notebook[a, ScreenStyleEnvironment -> "Slideshow Presentation",
					StyleDefinitions -> CurrentValue[pnb1, StyleDefinitions]], _[WindowSize | WindowMargins, _]], Notebook[WindowSize -> {200, 140}, WindowFrame -> "Frameless",
						WindowFrameElements -> {}, WindowElements -> {}, WindowMargins -> thumbnailmargins, WindowFloating -> True, BlinkingCellInsertionPoint -> False, CellInsertionPointCell -> None,
CellInsertionPointColor -> None]])];
			Pause[.2];
			SetSelectedNotebook[nb];
			img = CurrentNotebookImage[nb];
			NotebookClose[nb];

			ImageCrop[ImageResize[img, imageSize], imageSize, Top, Padding -> Automatic]
		]
]

(* two argument form which doesn't not need Outline Palette or original notebook *)
AlternateGetNotebookThumbnail[slide_, imageSize_: {100, 70}] :=
Module[{nb, img},
	(* Load Java early so that when NotebookPut is called below, CurrentNotebookImage is ready  *)
	CurrentNotebookImage[];

	nb = NotebookPut[((slide /. Cell[con_, "SlideShowNavigationBar", stys___, opts___?OptionQ] :> Cell[con, "SlideShowNavigationBar", stys, PageBreakAbove -> False,
			opts]) /. Notebook[a_, b_List] :> Join[DeleteCases[Notebook[a, ScreenStyleEnvironment -> "Slideshow Presentation",
				StyleDefinitions -> CurrentValue[pnb1, StyleDefinitions]], _[WindowSize | WindowMargins, _]], Notebook[WindowSize -> {200, 140}, WindowFrame -> "Frameless",
						WindowFrameElements -> {}, WindowElements -> {}, WindowMargins -> {{Automatic, Automatic}, {Automatic, Automatic}}, WindowFloating -> True, BlinkingCellInsertionPoint -> False, CellInsertionPointCell -> None,
CellInsertionPointColor -> None]])];
	Pause[.2];
	SetSelectedNotebook[nb];
	img = CurrentNotebookImage[nb];
	NotebookClose[nb];

	ImageCrop[ImageResize[img, imageSize], imageSize, Top, Padding -> Automatic]

]



(*******************************************************

 MakeThumbnailNotebook

	in: Notebook
	out: Notebook

*******************************************************)

Options[MakeThumbnailNotebook] = {
(*	"WindowSize" -> {200, 150} *)
	"WindowSize" -> {400, 280}
}
MakeThumbnailNotebook[Notebook[notebookExpr_, notebookOpts___?OptionQ], opts___?OptionQ] :=
Module[{nbExpr = notebookExpr, size, nbOpts = {notebookOpts}},

 	nbExpr = nbExpr /. Cell[con_, sty_, o___] :> Cell[con, sty, PageBreakAbove -> False, o];
	nbOpts = nbOpts /. {a___, Rule[TaggingRules, _], b___} :> {a, b};

	size = "WindowSize" /. {opts} /. Options[MakeThumbnailNotebook];

	With[{options = Join[{WindowSize -> size, ScrollingOptions -> {"PagewiseDisplay" -> False}, ScreenStyleEnvironment -> "Slideshow Presentation"}, nbOpts]},
		Notebook[
			Flatten@{
				Cell["", CellOpen -> False, CellElementSpacings -> {"ClosedCellHeight" -> 0}, CellMargins -> {{0, 0}, {0, 0}}],
				nbExpr,
				Cell["", CellOpen -> False, CellElementSpacings -> {"ClosedCellHeight" -> 0}, CellMargins -> {{0, 0}, {0, 0}}]
				},
			Sequence@@options
		]
	]
]






(****************************

 GetPreviousWindowSize

****************************)

GetPreviousWindowSize[nbObj_] :=
Module[{width, height},
	width = CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "WindowSize"}];
	height = CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "WindowSize"}];
	width = If[TrueQ[width === Inherited], Automatic, First[width]];
	height = If[TrueQ[height === Inherited], Automatic, If[Length[height] > 1, Part[height, 2], First[height]]];
	{width, height}
];




(********************************)
(* SaveNotebookWithOutToolbar   *)
(********************************)
SaveNotebookWithOutToolbar[nb_NotebookObject]:=
Module[{expr, ssExpr, ssPath},
	If[TrueQ[(# =!= Null) && (# =!= $Canceled)],
		ssPath = FileNameJoin[{$InstallationDirectory, "SystemFiles", "FrontEnd", "StyleSheets", "Utility", "PresenterTools.nb"}];
		If[TrueQ[FileType[ssPath] == File],
			ssExpr = Import[ssPath];
			expr = NotebookGet[nb];
(*				Message[SaveNotebookWithOutToolbar::argx,{Head@ssExpr, Head@expr}]; *)
			If[TrueQ[Head[ssExpr] == Notebook],
				With[{stylesheet = ssExpr},
(*				Message[SaveNotebookWithOutToolbar::argx,{Head@stylesheet}]; *)
					expr = expr /. Notebook[c_, opts___] :>
						Notebook[c, StyleDefinitions -> stylesheet, DockedCells -> {FEPrivate`FrontEndResource["FEExpressions", "SlideshowToolbar"]},
							Sequence@@DeleteCases[{opts}, StyleDefinitions -> _] ];
				];
				Export[#, expr, "NB"] ] ]
	, Abort[]
	] &[System`SystemDialogInput["FileSave",
		{$UserDocumentsDirectory, {"Notebook (*.nb)" -> {"*.nb"}}}]
	]
];






(**************************************

  CreateOutlinePalette
	Create palette of clickable thumbnails
	in:  None
	out: Palette notebook

***************************************)

PresenterTools`Private`ImageButtonWithBorderAndNumber[i_, im_, nb_, cells_] :=
	With[{nb1 = nb, slideNumber = i, cellObj = Part[cells, i], id = "Thumbnail" <> ToString@i},
		MouseAppearance[Button[Grid[{{Framed[Pane[im, ImageSize -> Dynamic[CurrentValue[EvaluationNotebook[],
													{TaggingRules, "PresenterSettings", "OutlinePalette", "ThumbnailMagnification"}, .8]*{100, 70}],
								ImageSizeAction -> "ResizeToFit", Alignment -> {Left, Top}, ContentPadding -> False],
							ImageMargins -> {{0, 10}, {0, 0}},
							FrameMargins -> 0,
							FrameStyle -> Dynamic[If[CurrentValue[NotebookSelection@nb1, {"CounterValue", "SlideShowNavigationBar"}] === slideNumber,
											{Thickness[2], RGBColor[0.6665, 0.8355, 0.992]},
											(*ELSE*){Thickness[2], GrayLevel[0.85]}]], Background -> GrayLevel[1]]
											},{
						Style[ToString[slideNumber],
							FontColor -> GrayLevel[0.35],
							FontFamily :> CurrentValue["PanelFontFamily"],
							FontSize -> Dynamic[CurrentValue[EvaluationNotebook[], {TaggingRules, "PresenterSettings", "OutlinePalette", "ThumbnailMagnification"}, .8]*20]]}},
						Alignment -> Center, Spacings -> {0, 0.5}],
					(SetSelectedNotebook[nb1];
					SelectionMove[nb1, Before, Notebook, AutoScroll -> False];
					SelectionMove[cellObj, All, Cell, AutoScroll -> False];
					SelectionMove[nb1, After, Cell, AutoScroll -> False];
					NotebookWrite[nb1, Cell["", Deletable -> True, ShowCellBracket -> False], All];
					NotebookDelete[nb1];
					SelectionMove[nb1, Next, Cell];
					SelectionMove[nb1, Before, Cell];
					MathLink`CallFrontEnd[FrontEnd`BoxReferenceFind[FE`BoxReference[ButtonNotebook[], {{id}}]]]),
					Evaluator -> Automatic,
					Method -> "Preemptive",
					BoxID -> "Thumbnail" <> ToString@i,
					Appearance -> {"Default" -> None, "Pressed" -> None}], "LinkHand"]
					];

Options[OutlinePaletteNavigationBar] = {"MoveToCell" -> False};

OutlinePaletteNavigationBar[nbObj1_, opts___] :=
	With[{nbObj = nbObj1, mtc = ("MoveToCell" /. {opts} /. Options[OutlinePaletteNavigationBar])},
	Cell[BoxData[ToBoxes[Grid[{{Style["", FontSize -> 9],
		(*general navigation*)
		Grid[{{Pane[Grid[{{Button[Dynamic[RawBoxes[FEPrivate`FrontEndResource["PresenterToolExpressions", "FirstSlideButton"]]],
					Module[{len},
						If[(len = Length@Cells[nbObj, CellStyle -> "SlideShowNavigationBar"]) > 0,
						SelectionMove[nbObj, Before, Notebook];
						NotebookFind[nbObj, "SlideShowNavigationBar", Next, CellStyle];
							If[len > 1, SelectionMove[nbObj, After, Cell]];
							MathLink`CallFrontEnd[FrontEnd`BoxReferenceFind[FE`BoxReference[ButtonNotebook[], {{"Thumbnail1"}}]]];
							SelectionMove[ButtonNotebook[], Next, Cell];
							If[TrueQ@mtc, SelectionMove[ButtonNotebook[], Before, Cell]]]],
					Appearance -> {"Default" -> None, "Pressed" -> None},
					Tooltip -> Dynamic[FEPrivate`FrontEndResource["SlideshowToolbarText", "TooltipFirstSlide"]], TooltipDelay -> 0.25],
				Button[Dynamic[RawBoxes[FEPrivate`FrontEndResource["PresenterToolExpressions", "PreviousSlideButton"]]],
					Module[{cv, cv2, n},
						If[Length@Cells[nbObj, CellStyle -> "SlideShowNavigationBar"] > 0,
							cv = CurrentValue[NotebookSelection[nbObj], {"CounterValue", "SlideShowNavigationBar"}];
							NotebookFind[nbObj, "SlideShowNavigationBar", Previous, CellStyle];
                					cv2 = CurrentValue[NotebookSelection[nbObj ], {"CounterValue", "SlideShowNavigationBar"}];
							If[cv2 === cv && cv =!= 1, NotebookFind[nbObj, "SlideShowNavigationBar", Previous, CellStyle]];
							n = If[cv === 1, 1, CurrentValue[NotebookSelection[nbObj ], {"CounterValue", "SlideShowNavigationBar"}]];
							MathLink`CallFrontEnd[FrontEnd`BoxReferenceFind[FE`BoxReference[ButtonNotebook[], {{"Thumbnail" <> ToString@n}}]]];
							SelectionMove[ButtonNotebook[], Next, Cell];
                  					If[TrueQ@mtc, SelectionMove[ButtonNotebook[], Before, Cell]]]],
					Appearance -> {"Default" -> None, "Pressed" -> None},
					Tooltip -> Dynamic[FEPrivate`FrontEndResource["SlideshowToolbarText", "TooltipPreviousSlide"]], TooltipDelay -> 0.25],
				Button[Dynamic[RawBoxes[FEPrivate`FrontEndResource["PresenterToolExpressions", "NextSlideButton"]]],
					Module[{len, cv, n},
						If[(len = Length@Cells[nbObj, CellStyle -> "SlideShowNavigationBar"]) > 0,
							cv = CurrentValue[NotebookSelection[nbObj], {"CounterValue", "SlideShowNavigationBar"}];
                  					n = If[cv =!= len,
                  						NotebookFind[nbObj, "SlideShowNavigationBar", Next, CellStyle];
                  						If[CurrentValue[NotebookSelection[nbObj], {"CounterValue", "SlideShowNavigationBar"}] === cv,
									NotebookFind[nbObj, "SlideShowNavigationBar", Next, CellStyle]];
                  						SelectionMove[nbObj, After, Cell]; cv + 1, cv];
							MathLink`CallFrontEnd[FrontEnd`BoxReferenceFind[FE`BoxReference[ButtonNotebook[], {{"Thumbnail" <> ToString@n}}]]];
							SelectionMove[ButtonNotebook[], Next, Cell];
							If[TrueQ@mtc, SelectionMove[ButtonNotebook[], Before, Cell]]]],
					Appearance -> {"Default" -> None, "Pressed" -> None},
					Tooltip -> Dynamic[FEPrivate`FrontEndResource["SlideshowToolbarText", "TooltipNextSlide"]], TooltipDelay -> 0.25],
				Button[Dynamic[RawBoxes[FEPrivate`FrontEndResource["PresenterToolExpressions", "LastSlideButton"]]],
					Module[{len},
						If[(len = Length@Cells[nbObj, CellStyle -> "SlideShowNavigationBar"]) > 0,
							SelectionMove[nbObj, After, Notebook];
                  					NotebookFind[nbObj, "SlideShowNavigationBar", Previous, CellStyle];
							MathLink`CallFrontEnd[FrontEnd`BoxReferenceFind[FE`BoxReference[ButtonNotebook[], {{"Thumbnail" <> ToString@len}}]]];
							SelectionMove[ButtonNotebook[], Next, Cell];
							If[TrueQ@mtc, SelectionMove[ButtonNotebook[], Before, Cell]]]],
					Appearance -> {"Default" -> None, "Pressed" -> None},
					Tooltip -> Dynamic[FEPrivate`FrontEndResource["SlideshowToolbarText", "TooltipLastSlide"]], TooltipDelay -> 0.25]}},
				Alignment -> Left,
				Background -> RGBColor[0.2, 0.2, 0.2],
				ItemSize -> Full,
				Spacings -> {{0, {0.5}, 0.}, {0., {0.2}, 0.}}],
			ContentPadding -> False,
			FrameMargins -> {{0, 0.}, {0, 0}},
			ImageMargins -> {{0, 0}, {0, 0}}]}},
		Background -> RGBColor[0.2, 0.2, 0.2],
		Spacings -> {0, 0},
		ItemSize -> Automatic],
		(*hamburglar*)Style["", FontSize -> 9]
		(*end of row*)}}, Frame -> {All, False},
				FrameStyle -> Directive[RGBColor[0.2, 0.2, 0.2]],
				Background -> RGBColor[0.2, 0.2, 0.2],
				Alignment -> {{Left, Center, Right}},
				ItemSize -> {{Scaled[0.1035], Scaled[0.8], Scaled[0.1]}},
				Spacings -> {{0.35, {0.}, 0.}, {0.1, {0.2}, 0.2}}]]],
		FontSize -> 9,
		Background -> RGBColor[0.2, 0.2, 0.2],
		Magnification -> 1, CellFrame -> False,
		CellMargins -> {{-1, -1}, {0, -1}},
		CellFrameMargins -> {{5, 5}, {0, 0}},
		CellFrameColor -> RGBColor[0.2, 0.2, 0.2]]];

FirstPositionOFASlideBreakCell[nb_] := (If[MatchQ[#, {{_Integer}}], #[[1, 1]], {}] &@Position[Cells[nb],
							_?(MatchQ[Developer`CellInformation@#, {"Style" -> ({"SlideShowNavigationBar", "FirstSlide"} | "SlideShowNavigationBar"), __}] &), 1, 1])

Options[CreateOutlinePalette] = {
	"WindowTitle" -> None
};
CreateOutlinePalette[opts___?OptionQ]:= CreateOutlinePalette[InputNotebook[], opts];
CreateOutlinePalette[nb_, opts___?OptionQ] :=
	(If[IntegerQ@# && # > 1,
		SelectionMove[nb, Before, Notebook]; NotebookWrite[nb, Cell["", "SlideShowNavigationBar", "FirstSlide", CellTags->"SlideShowHeader"]]] &[FirstPositionOFASlideBreakCell[nb]];
	NotebookPut@With[{nb1 = nb, imageSize = {100, 70}, backgroundColor = If[ColorQ[AbsoluteCurrentValue[nb, Background]], AbsoluteCurrentValue[nb, Background], GrayLevel[1]]},
		Notebook[{Cell[BoxData@ToBoxes@Pane@Row[If[Length@Cells[nb1, CellStyle -> "SlideShowNavigationBar"] === 0,
							{Framed[Pane[Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, "GenericSlideThumbnail.png"]],
									ImageSize -> Dynamic[CurrentValue[EvaluationNotebook[],
									{TaggingRules, "PresenterSettings", "OutlinePalette", "ThumbnailMagnification"}, .8]*{100, 70}], ImageSizeAction -> "ResizeToFit",
									Alignment -> {Left, Top}, ContentPadding -> False], ImageMargins -> {{0, 0}, {0, 0}}, FrameMargins -> 0,
									FrameStyle -> {Thickness[2], GrayLevel[0.85]}, Background -> GrayLevel[1]]},
							Table[PresenterTools`Private`ImageButtonWithBorderAndNumber[i,
									Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, "GenericSlideThumbnail.png"]], nb1,
										Cells[nb1, CellStyle -> "SlideShowNavigationBar"]],
								{i, Length@Cells[nb1, CellStyle -> "SlideShowNavigationBar"]}]]],
				"Output",
				CellMargins -> {{30, 0}, {20, 20}},
				TextAlignment -> Center,
				LineIndent -> 0,
				LinebreakAdjustments -> {1., 10, 0, 0, 10},
				ShowStringCharacters -> False,
				ShowCellBracket -> False,
				CellTags -> "ThumbnailArray"]},
	DockedCells -> {PresenterTools`OutlinePaletteNavigationBar[nb1],
		Cell[BoxData[ToBoxes@DynamicModule[{},Pane[Grid[{{Pane[Row[{Style[Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "ThumbnailSize"],
									FontSize -> 11, FontFamily :> CurrentValue["PanelFontFamily"]],
						Slider[Dynamic[CurrentValue[EvaluationNotebook[], {TaggingRules, "PresenterSettings", "OutlinePalette", "ThumbnailMagnification"}, .8]], {.4, 2, .1},
							Appearance -> Small, ImageSize -> 100]},
						Spacer[1.5]], Appearance -> None, ImageSize -> Full, ImageMargins -> 0, FrameMargins -> 3],

						MouseAppearance[Button[Mouseover[Panel[Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, "RefreshThumbnails.png"]],
									Background -> GrayLevel[0.95], Appearance -> None],
										Panel[Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, "RefreshThumbnails.png"]],
											Background -> GrayLevel[0.99], Appearance -> None]],

						(If[IntegerQ@# && # > 1,
								SelectionMove[nb1, Before, Notebook];
					NotebookWrite[nb1, Cell["", "SlideShowNavigationBar", "FirstSlide", CellTags->"SlideShowHeader"]];
		CurrentValue[ButtonNotebook[], {TaggingRules, "PresenterSettings", "OutlineThumbnails"}] = Table[ToString@i -> "Generic", {i, Length@Cells[nb1, CellStyle -> "SlideShowNavigationBar"]}];
					CurrentValue[nb1, {TaggingRules, "PresenterSettings", "OutlinePalette", "Thumbnails"}] = Inherited]&[PresenterTools`Private`FirstPositionOFASlideBreakCell[nb1]]);
									Module[{thumbnail, dialogOutlineThumbnailsTaggingRule, dialogImageCount, controlsDockedCell, gt,
											slideCells = Cells[nb1, CellStyle -> "SlideShowNavigationBar"], slideCount, bn = ButtonNotebook[],
											thumbnailRules, t, UseStandardMethod = MatchQ[CurrentValue[nb1, {StyleDefinitions, "FirstSlide",
																	"PrivateCellOptionsPagewiseNotebookBaseStyle"}], {_ -> None}]},
			If[slideCells === {},

					CurrentValue[bn, {TaggingRules, "PresenterSettings", "OutlineThumbnails"}] = {"1" -> "Generic"};
				CurrentValue[nb1, {TaggingRules, "PresenterSettings", "OutlinePalette", "Thumbnails"}] = {};
					NotebookFind[bn, "ThumbnailArray", All, CellTags];
					NotebookWrite[bn,
						Cell[BoxData@ToBoxes@Pane@Row[{Framed[Pane[Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, "GenericSlideThumbnail.png"]],
						ImageSize -> Dynamic[CurrentValue[EvaluationNotebook[], {TaggingRules, "PresenterSettings","OutlinePalette", "ThumbnailMagnification"}, .8]*{100, 70}],
											ImageSizeAction -> "ResizeToFit",
											Alignment -> {Left, Top},
											ContentPadding -> False],
											ImageMargins -> {{0, 0}, {0, 0}},
											FrameMargins -> 0,
											FrameStyle -> {Thickness[2], GrayLevel[0.85]},
											Background -> GrayLevel[1]]}],
							"Output",
							CellMargins -> {{30, 0}, {20, 20}},
							TextAlignment -> Center,
							LineIndent -> 0,
							LinebreakAdjustments -> {1., 10, 0, 0, 10},
							ShowStringCharacters -> False,
							ShowCellBracket -> False,
							CellTags -> "ThumbnailArray"],
						All];
						$CellContext`x = False,

			If[MatchQ[CurrentValue[bn, {TaggingRules, "PresenterSettings", "OutlineThumbnails"}], {_ -> "Thumbnail", ___} | {} | Inherited],

					$CellContext`r = Table[0, Length@slideCells];
					$CellContext`$showloading = True;
				Pause[.5];
				$CellContext`$slides = PresenterTools`GetSlides[nb1, "DeleteNavigationBarWithFirstSlideStyle" -> UseStandardMethod];
					$CellContext`$showloading = False;
					$CellContext`$showprogress = True;
				$CellContext`$slideCount = Length[$CellContext`$slides];
					$CellContext`n = 0;
				Do[If[TrueQ@$CellContext`StopThumbnailGeneration,
						CurrentValue[bn, {TaggingRules, "ResumeNumber"}] = i;
						Break[]];
					t = If[FreeQ[$CellContext`$slides[[i]], Cell[_, "SlideShowNavigationBar", ___]],
							PresenterTools`Private`GetNotebookThumbnail[$CellContext`$slides[[i]], imageSize, "BackgroundColor" -> backgroundColor],
							AlternateGetNotebookThumbnail[$CellContext`$slides[[i]], nb1, bn]];
						Pause[.02];
						$CellContext`n = i;
					$CellContext`r[[i]] = t, {i, 1, $CellContext`$slideCount}];
				If[Not@TrueQ@$CellContext`StopThumbnailGeneration,
				CurrentValue[bn, {TaggingRules, "PresenterSettings", "OutlineThumbnails"}] = Table[ToString@i -> "Thumbnail", {i, $CellContext`$slideCount}];
				CurrentValue[nb1, {TaggingRules, "PresenterSettings", "OutlinePalette", "Thumbnails"}] = Table[ToString@i -> $CellContext`r[[i]], {i, $CellContext`$slideCount}];
					Pause[.02];
					$CellContext`$showprogress = False;
				NotebookFind[bn, "ThumbnailArray", All, CellTags];
				NotebookWrite[bn,
							Cell[BoxData@ToBoxes@Pane@Row[If[Length@Cells[nb1, CellStyle -> "SlideShowNavigationBar"] === 0,
									{Framed[Pane[Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, "GenericSlideThumbnail.png"]],
											ImageSize -> Dynamic[CurrentValue[EvaluationNotebook[],
												{TaggingRules, "PresenterSettings", "OutlinePalette", "ThumbnailMagnification"}, .8]*{100, 70}],
											ImageSizeAction -> "ResizeToFit", Alignment -> {Left, Top}, ContentPadding -> False],
										ImageMargins -> {{0, 0}, {0, 0}}, FrameMargins -> 0, FrameStyle -> {Thickness[2], GrayLevel[0.85]}, Background -> GrayLevel[1]]},
									Table[PresenterTools`Private`ImageButtonWithBorderAndNumber[i, $CellContext`r[[i]], nb1, Cells[nb1, CellStyle -> "SlideShowNavigationBar"]],
										{i, Length@Cells[nb1, CellStyle -> "SlideShowNavigationBar"]}]]],
												"Output",
												CellMargins -> {{30, 0}, {20, 20}},
												TextAlignment -> Center,
												LineIndent -> 0,
												LinebreakAdjustments -> {1., 10, 0, 0, 10},
												ShowStringCharacters -> False, ShowCellBracket -> False,
												CellTags -> "ThumbnailArray"],
					All]],

				$CellContext`$slideCount = Length@Cells[nb1, CellStyle -> "SlideShowNavigationBar"];
				(*CurrentValue[nb1, {TaggingRules, "PresenterSettings", "OutlinePalette", "Thumbnails"}] = Table[ToString@i -> "Generic", {i, $CellContext`$slideCount}];*)
				NotebookFind[bn, "ThumbnailArray", All, CellTags];
				NotebookWrite[bn,
						Cell[BoxData@ToBoxes@Pane@Row[Table[PresenterTools`Private`ImageButtonWithBorderAndNumber[i,
										Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, "GenericSlideThumbnail.png"]], nb1,
																		Cells[nb1, CellStyle -> "SlideShowNavigationBar"]],
											{i, $CellContext`$slideCount}]],
												"Output",
												CellMargins -> {{30, 0}, {20, 20}},
												TextAlignment -> Center,
												LineIndent -> 0,
												LinebreakAdjustments -> {1., 10, 0, 0, 10},
												ShowStringCharacters -> False, ShowCellBracket -> False,
												CellTags -> "ThumbnailArray"],
				All]]]],

									Tooltip -> Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "OutlinePaletteRefreshTooltip"], TooltipDelay -> .25, Appearance -> {"Default" -> None, "Pressed" -> None}, Method -> "Queued"],
								"LinkHand"]},

						{Grid[{{DynamicWrapper[Checkbox[Dynamic[$CellContext`x(*, (If[TrueQ@$CellContext`$cancel,
														$CellContext`x = False; Clear@$CellContext`$cancel,
														$CellContext`x = Not@$CellContext`x]&)*)]],
							If[Not@MemberQ[Notebooks[], nb1],

								NotebookClose[EvaluationNotebook[]],


							If[Cells[nb1, CellStyle -> "SlideShowNavigationBar"] === {},

								If[TrueQ@$CellContext`x,

									Module[{pnbThumbnailTaggingRule = CurrentValue[nb1, {TaggingRules, "PresenterSettings", "OutlinePalette", "Thumbnails"}],
										thumbnail},
								If[(* Thumbnails are not present in tagging rules of presenter notebook. *)
										(pnbThumbnailTaggingRule === Inherited || pnbThumbnailTaggingRule === {}),

								thumbnail = PresenterTools`Private`GetNotebookThumbnail[Notebook[NotebookGet[nb1][[1]], ScreenStyleEnvironment -> "Slideshow Working",
												StyleDefinitions -> FrontEnd`FileName[{"PresenterTools"}, "Default.nb", CharacterEncoding -> "UTF-8"]], imageSize, "BackgroundColor" -> backgroundColor];
								CurrentValue[EvaluationNotebook[], {TaggingRules, "PresenterSettings", "OutlineThumbnails"}] = {"1" -> "Thumbnail"};
								CurrentValue[nb1, {TaggingRules, "PresenterSettings", "OutlinePalette", "Thumbnails"}] = {"1" -> thumbnail};
								NotebookFind[EvaluationNotebook[], "ThumbnailArray", All, CellTags];
								NotebookWrite[EvaluationNotebook[],
										Cell[BoxData@ToBoxes@Pane@Row[{Framed[Pane[thumbnail,
						ImageSize -> Dynamic[CurrentValue[EvaluationNotebook[], {TaggingRules, "PresenterSettings","OutlinePalette", "ThumbnailMagnification"}, .8]*{100, 70}],
																ImageSizeAction -> "ResizeToFit",
																Alignment -> {Left, Top},
																ContentPadding -> False],
																ImageMargins -> {{0, 0}, {0, 0}},
																FrameMargins -> 0,
																FrameStyle -> {Thickness[2], GrayLevel[0.85]},
																Background -> GrayLevel[1]]}],
											"Output",
											CellMargins -> {{30, 0}, {20, 20}},
											TextAlignment -> Center,
											LineIndent -> 0,
											LinebreakAdjustments -> {1., 10, 0, 0, 10},
											ShowStringCharacters -> False,
											ShowCellBracket -> False,
											CellTags -> "ThumbnailArray"],
										All],

								thumbnail = CurrentValue[nb1, {TaggingRules, "PresenterSettings", "OutlinePalette", "Thumbnails", "1"}];
								CurrentValue[nb1, {TaggingRules, "PresenterSettings", "OutlinePalette", "Thumbnails"}] = {"1" -> thumbnail};
								CurrentValue[EvaluationNotebook[], {TaggingRules, "PresenterSettings", "OutlineThumbnails"}] = {"1" -> "Thumbnail"};
								NotebookFind[EvaluationNotebook[], "ThumbnailArray", All, CellTags];
								NotebookWrite[EvaluationNotebook[],
										Cell[BoxData@ToBoxes@Pane@Row[{Framed[Pane[thumbnail,
						ImageSize -> Dynamic[CurrentValue[EvaluationNotebook[], {TaggingRules, "PresenterSettings","OutlinePalette", "ThumbnailMagnification"}, .8]*{100, 70}],
																ImageSizeAction -> "ResizeToFit",
																Alignment -> {Left, Top},
																ContentPadding -> False],
																ImageMargins -> {{0, 0}, {0, 0}},
																FrameMargins -> 0,
																FrameStyle -> {Thickness[2], GrayLevel[0.85]},
																Background -> GrayLevel[1]]}],
												"Output",
												CellMargins -> {{30, 0}, {20, 20}},
												TextAlignment -> Center,
												LineIndent -> 0,
												LinebreakAdjustments -> {1., 10, 0, 0, 10},
												ShowStringCharacters -> False,
												ShowCellBracket -> False,
												CellTags -> "ThumbnailArray"],
											All]]],

								CurrentValue[EvaluationNotebook[], {TaggingRules, "PresenterSettings", "OutlineThumbnails"}] = {"1" -> "Generic"};
								NotebookFind[EvaluationNotebook[], "ThumbnailArray", All, CellTags];
								NotebookWrite[EvaluationNotebook[],
										Cell[BoxData@ToBoxes@Pane@Row[{Framed[Pane[Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"},
																					"GenericSlideThumbnail.png"]],
												ImageSize -> Dynamic[CurrentValue[EvaluationNotebook[], {TaggingRules, "PresenterSettings","OutlinePalette",
																				"ThumbnailMagnification"}, .8]*{100, 70}],
												ImageSizeAction -> "ResizeToFit",
												Alignment -> {Left, Top},
												ContentPadding -> False],
												ImageMargins -> {{0, 0}, {0, 0}},
												FrameMargins -> 0,
												FrameStyle -> {Thickness[2], GrayLevel[0.85]},
												Background -> GrayLevel[1]]}],
											"Output",
											CellMargins -> {{30, 0}, {20, 20}},
											TextAlignment -> Center,
											LineIndent -> 0,
											LinebreakAdjustments -> {1., 10, 0, 0, 10},
											ShowStringCharacters -> False,
											ShowCellBracket -> False,
											CellTags -> "ThumbnailArray"],
										All]],

								If[TrueQ@$CellContext`x,

								(If[IntegerQ@# && # > 1,
									SelectionMove[nb1, Before, Notebook];
									NotebookWrite[nb1, Cell["", "SlideShowNavigationBar", "FirstSlide", CellTags -> "SlideShowHeader"]];
	CurrentValue[EvaluationNotebook[], {TaggingRules, "PresenterSettings", "OutlineThumbnails"}] = Table[ToString@i -> "Generic", {i, Length@Cells[nb1, CellStyle -> "SlideShowNavigationBar"]}];
		CurrentValue[nb1, {TaggingRules, "PresenterSettings", "OutlinePalette", "Thumbnails"}] = Inherited]&[PresenterTools`Private`FirstPositionOFASlideBreakCell[nb1]]);

									Module[{slideCells = Cells[nb1, CellStyle -> "SlideShowNavigationBar"],
										pnbThumbnailTaggingRule = CurrentValue[nb1, {TaggingRules, "PresenterSettings", "OutlinePalette", "Thumbnails"}],
										palnbThumbnails, controlsDockedCell, slideCount, t, UseStandardMethod = MatchQ[CurrentValue[nb1, {StyleDefinitions,
															"FirstSlide", "PrivateCellOptionsPagewiseNotebookBaseStyle"}], {_ -> None}]},
If[MatchQ[CurrentValue[EvaluationNotebook[], {TaggingRules, "PresenterSettings", "OutlineThumbnails"}], Alternatives[{(_ -> "Generic")..}, Inherited, {}]],
	If[(* Thumbnails are not present in tagging rules of presenter notebook. *)
											(pnbThumbnailTaggingRule === Inherited || pnbThumbnailTaggingRule === {}),
											(* Generate thumbnails dispaying them in the palette and putting them into the tagging rules of the presenter notebook *)
  		$CellContext`r = Table[0, Length@slideCells];
		$CellContext`$showloading = True;
		Pause[.5];
		$CellContext`$slides = PresenterTools`GetSlides[nb1, "DeleteNavigationBarWithFirstSlideStyle" -> UseStandardMethod];
		$CellContext`$showloading = False;
		$CellContext`$showprogress = True;
		$CellContext`$slideCount = Length[$CellContext`$slides];
		$CellContext`n = 0;
		Do[If[TrueQ@$CellContext`StopThumbnailGeneration,
				CurrentValue[EvaluationNotebook[], {TaggingRules, "ResumeNumber"}] = i;
				$CellContext`x = False;
				(*CurrentValue[EvaluationNotebook[], {TaggingRules, "PresenterSettings", "OutlineThumbnails"}] = Table[ToString@i -> "Generic", {i, Length@slideCells}];*)
				Break[]];
			t = If[FreeQ[$CellContext`$slides[[i]], Cell[_, "SlideShowNavigationBar", ___]],
				PresenterTools`Private`GetNotebookThumbnail[$CellContext`$slides[[i]], imageSize, "BackgroundColor" -> backgroundColor],
				AlternateGetNotebookThumbnail[$CellContext`$slides[[i]], nb1, EvaluationNotebook[]]];
			Pause[.02];
			$CellContext`n = i;
			$CellContext`r[[i]] = t, {i, 1, $CellContext`$slideCount}];
		If[Not@TrueQ@$CellContext`StopThumbnailGeneration,
		CurrentValue[EvaluationNotebook[], {TaggingRules, "PresenterSettings", "OutlineThumbnails"}] = Table[ToString@i -> "Thumbnail", {i, $CellContext`$slideCount}];
		CurrentValue[nb1, {TaggingRules, "PresenterSettings", "OutlinePalette", "Thumbnails"}] = Table[ToString@i -> $CellContext`r[[i]], {i, $CellContext`$slideCount}];
		Pause[.02];
		$CellContext`$showprogress = False;
		NotebookFind[EvaluationNotebook[], "ThumbnailArray", All, CellTags];
		NotebookWrite[EvaluationNotebook[],
				Cell[BoxData@ToBoxes@Pane@Row[If[Length@Cells[nb1, CellStyle -> "SlideShowNavigationBar"] === 0,
						{Framed[Pane[Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, "GenericSlideThumbnail.png"]],
								ImageSize -> Dynamic[CurrentValue[EvaluationNotebook[],
									{TaggingRules, "PresenterSettings", "OutlinePalette", "ThumbnailMagnification"}, .8]*{100, 70}],
								ImageSizeAction -> "ResizeToFit", Alignment -> {Left, Top}, ContentPadding -> False],
							ImageMargins -> {{0, 0}, {0, 0}}, FrameMargins -> 0, FrameStyle -> {Thickness[2], GrayLevel[0.85]}, Background -> GrayLevel[1]]},
						Table[PresenterTools`Private`ImageButtonWithBorderAndNumber[i, $CellContext`r[[i]], nb1, Cells[nb1, CellStyle -> "SlideShowNavigationBar"]],
							{i, Length@Cells[nb1, CellStyle -> "SlideShowNavigationBar"]}]]],
								"Output",
								CellMargins -> {{30, 0}, {20, 20}},
								TextAlignment -> Center,
								LineIndent -> 0,
								LinebreakAdjustments -> {1., 10, 0, 0, 10},
								ShowStringCharacters -> False, ShowCellBracket -> False,
								CellTags -> "ThumbnailArray"],
				All];
			(*CurrentValue[EvaluationNotebook[], {TaggingRules, "PresenterSettings", "OutlineThumbnails"}] = Table[ToString@i -> "Thumbnail", {i, Length@slideCells}]*)],

											(* Thumbnails have been generated and inserted into presenter notebook. *)
		NotebookFind[EvaluationNotebook[], "ThumbnailArray", All, CellTags];
		NotebookWrite[EvaluationNotebook[],
				Cell[BoxData@ToBoxes@Pane@Row[If[Length@Cells[nb1, CellStyle -> "SlideShowNavigationBar"] === 0,
								{Framed[Pane[Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, "GenericSlideThumbnail.png"]],
										ImageSize -> Dynamic[CurrentValue[EvaluationNotebook[],
											{TaggingRules, "PresenterSettings", "OutlinePalette", "ThumbnailMagnification"}, .8]*{100, 70}],
										ImageSizeAction -> "ResizeToFit", Alignment -> {Left, Top}, ContentPadding -> False],
									ImageMargins -> {{0, 0}, {0, 0}}, FrameMargins -> 0, FrameStyle -> {Thickness[2], GrayLevel[0.85]}, Background -> GrayLevel[1]]},
								Table[PresenterTools`Private`ImageButtonWithBorderAndNumber[i,
													CurrentValue[nb1, {TaggingRules, "PresenterSettings", "OutlinePalette", "Thumbnails", ToString@i}], nb1,
													Cells[nb1, CellStyle -> "SlideShowNavigationBar"]],
									{i, Min[Length@Cells[nb1, CellStyle -> "SlideShowNavigationBar"], Length@CurrentValue[nb1, {TaggingRules, "PresenterSettings", "OutlinePalette", "Thumbnails"}]]}]]],
										"Output",
										CellMargins -> {{30, 0}, {20, 20}},
										TextAlignment -> Center,
										LineIndent -> 0,
										LinebreakAdjustments -> {1., 10, 0, 0, 10},
										ShowStringCharacters -> False, ShowCellBracket -> False,
										CellTags -> "ThumbnailArray"],
				All];
		CurrentValue[EvaluationNotebook[], {TaggingRules, "PresenterSettings", "OutlineThumbnails"}] = If[Cells[nb1, CellStyle -> "SlideShowNavigationBar"] === {},
																			{},
																			Table[ToString@i -> "Thumbnail",
					{i, Min[Length@Cells[nb1, CellStyle -> "SlideShowNavigationBar"], Length@CurrentValue[nb1, {TaggingRules, "PresenterSettings", "OutlinePalette", "Thumbnails"}]]}]]]]],

	If[MatchQ[CurrentValue[EvaluationNotebook[], {TaggingRules, "PresenterSettings", "OutlineThumbnails"}], {(_ -> "Thumbnail") ..}],

									Module[{slideCells = Cells[nb1, CellStyle -> "SlideShowNavigationBar"], palnbThumbnails},
			NotebookFind[EvaluationNotebook[], "ThumbnailArray", All, CellTags];
			NotebookWrite[EvaluationNotebook[],
					Cell[BoxData@ToBoxes@Pane@Row[Table[PresenterTools`Private`ImageButtonWithBorderAndNumber[i,
							Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, "GenericSlideThumbnail.png"]], nb1,
							Cells[nb1, CellStyle -> "SlideShowNavigationBar"]], {i, Length@Cells[nb1, CellStyle -> "SlideShowNavigationBar"]}]],
						"Output",
						CellMargins -> {{30, 0}, {20, 20}},
						TextAlignment -> Center, LineIndent -> 0,
						LinebreakAdjustments -> {1., 10, 0, 0, 10},
						ShowStringCharacters -> False,
						ShowCellBracket -> False,
						CellTags -> "ThumbnailArray"],
					All];
			CurrentValue[EvaluationNotebook[], {TaggingRules, "PresenterSettings", "OutlineThumbnails"}] = Table[ToString@i -> "Generic",
												{i, Length@CurrentValue[EvaluationNotebook[], {TaggingRules, "PresenterSettings", "OutlineThumbnails"}]}]]]]]],
								SynchronousUpdating -> False],

							Style[Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "OutlinePaletteShowThumbnailsLabel"], FontSize -> 11,
								FontFamily :> CurrentValue["ControlsFontFamily"]]}}, Spacings -> 0.5, Alignment -> {Right, Center}], SpanFromAbove}},
						ItemSize -> {{Scaled[0.9], Scaled[0.1]}}, Spacings -> 0, Alignment -> {{Left, {Right}}, Center}],
					BaseStyle -> {Background -> GrayLevel[0.96]},
					FrameMargins -> {{6, 6}, {0, 4}},
					Appearance -> None],Initialization :> {$CellContext`x = False}]],
			CellContext -> Notebook,
			CellFrameColor -> GrayLevel[0.75],
			CellFrame -> {{0, 0}, {1, 0}},
			CellFrameMargins -> {{0, 0}, {0, 0}},
			Background -> GrayLevel[0.96], CellMargins -> 0],
	Cell[BoxData[ToBoxes@DynamicModule[{}, Dynamic[Style[Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "Loading"], FontSize -> 12, FontFamily -> "Source Sans Pro"]],
						Initialization :> {$CellContext`$showloading = False}]],
		"Text",
		CellOpen -> Dynamic[$CellContext`$showloading],
		CellContext -> Notebook,
		Background -> None,
		CellMargins -> 0,
		CellFrame -> 1,
		CellFrameColor -> GrayLevel[0.96],
		CellFrameMargins -> {{10, 0}, {4, 4}}],
	Cell[BoxData[ToBoxes@DynamicModule[{}, Style[Dynamic@Which[TrueQ@$CellContext`StopThumbnailGeneration, Row[{Button["Resume",
					(If[IntegerQ@# && # > 1,
					SelectionMove[nb1, Before, Notebook];
					NotebookWrite[nb1, Cell["", "SlideShowNavigationBar", "FirstSlide", CellTags -> "SlideShowHeader"]];
	CurrentValue[ButtonNotebook[], {TaggingRules, "PresenterSettings", "OutlineThumbnails"}] = Table[ToString@i -> "Generic", {i, Length@Cells[nb1, CellStyle -> "SlideShowNavigationBar"]}];
		CurrentValue[nb1, {TaggingRules, "PresenterSettings", "OutlinePalette", "Thumbnails"}] = Inherited]&[PresenterTools`Private`FirstPositionOFASlideBreakCell[nb1]]);
				Module[{slideCells = Cells[nb1, CellStyle -> "SlideShowNavigationBar"], t, UseStandardMethod = MatchQ[CurrentValue[nb1, {StyleDefinitions,
															"FirstSlide", "PrivateCellOptionsPagewiseNotebookBaseStyle"}], {_ -> None}]},
						$CellContext`StopThumbnailGeneration = False;
						$CellContext`x = True;
						$CellContext`n = 0;
						$CellContext`r = Table[0, Length@slideCells];
						$CellContext`$showloading = True;
						Pause[.5];
						$CellContext`$slides = PresenterTools`GetSlides[nb1, "DeleteNavigationBarWithFirstSlideStyle" -> UseStandardMethod];
						$CellContext`$showloading = False;
						$CellContext`$showprogress = True;
						$CellContext`$slideCount = Length[$CellContext`$slides];
						Do[If[TrueQ@$CellContext`StopThumbnailGeneration,
							CurrentValue[EvaluationNotebook[], {TaggingRules, "ResumeNumber"}] = i;
							Break[]];
							t = If[FreeQ[$CellContext`$slides[[i]], Cell[_, "SlideShowNavigationBar", ___]],
								PresenterTools`Private`GetNotebookThumbnail[$CellContext`$slides[[i]], imageSize, "BackgroundColor" -> backgroundColor],
								AlternateGetNotebookThumbnail[$CellContext`$slides[[i]], nb1, ButtonNotebook[]]];
							Pause[.02];
							$CellContext`n = i;
							$CellContext`r[[i]] = t, {i, 1, $CellContext`$slideCount}];
						If[Not@TrueQ@$CellContext`StopThumbnailGeneration,
						CurrentValue[EvaluationNotebook[], {TaggingRules, "PresenterSettings", "OutlineThumbnails"}] = Table[ToString@i -> "Thumbnail",
																			{i, $CellContext`$slideCount}];
						CurrentValue[nb1, {TaggingRules, "PresenterSettings", "OutlinePalette", "Thumbnails"}] = Table[ToString@i -> $CellContext`r[[i]],
																			{i, $CellContext`$slideCount}];
						Pause[.02];
						$CellContext`$showprogress = False;
						NotebookFind[EvaluationNotebook[], "ThumbnailArray", All, CellTags];
						NotebookWrite[EvaluationNotebook[],
								Cell[BoxData@ToBoxes@Pane@Row[If[Length@Cells[nb1, CellStyle -> "SlideShowNavigationBar"] === 0,
										{Framed[Pane[Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, "GenericSlideThumbnail.png"]],
											ImageSize -> Dynamic[CurrentValue[EvaluationNotebook[],
													{TaggingRules, "PresenterSettings", "OutlinePalette", "ThumbnailMagnification"}, .8]*{100, 70}],
											ImageSizeAction -> "ResizeToFit", Alignment -> {Left, Top}, ContentPadding -> False],
											ImageMargins -> {{0, 0}, {0, 0}}, FrameMargins -> 0, FrameStyle -> {Thickness[2], GrayLevel[0.85]},
											Background -> GrayLevel[1]]},
										Table[PresenterTools`Private`ImageButtonWithBorderAndNumber[i, $CellContext`r[[i]], nb1,
																			Cells[nb1, CellStyle -> "SlideShowNavigationBar"]],
							{i, Length@Cells[nb1, CellStyle -> "SlideShowNavigationBar"]}]]],
									"Output",
									CellMargins -> {{30, 0}, {20, 20}},
									TextAlignment -> Center,
									LineIndent -> 0,
									LinebreakAdjustments -> {1., 10, 0, 0, 10},
									ShowStringCharacters -> False, ShowCellBracket -> False,
									CellTags -> "ThumbnailArray"],
									All]]], ImageSize -> {70, Automatic}, Method -> "Queued"],
											Button["Cancel", $CellContext`$cancel = True; $CellContext`$showprogress = False;
											$CellContext`StopThumbnailGeneration = False; $CellContext`x = False]}],
										Not@TrueQ@$CellContext`StopThumbnailGeneration,
										Row[{Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "Loading"], Dynamic[$CellContext`n], "/", Dynamic@$CellContext`$slideCount, Spacer[6],
						ProgressIndicator[Dynamic[$CellContext`n], {0, Dynamic@If[Not@IntegerQ@$CellContext`$slideCount, 0, $CellContext`$slideCount]}, ImageSize -> {110, 16}],
								Spacer[6], Button["Stop", $CellContext`StopThumbnailGeneration = True, ImageSize -> {70, Automatic}]}],
										True,
										Button["\[FilledSmallSquare]", Null, ImageSize -> {70, Automatic}]], 12,
									FontFamily :> CurrentValue["PanelFontFamily"]],
					Initialization :> {$CellContext`$showprogress = False; $CellContext`$slideCount = 0}]], "DockedCell",
		CellOpen -> Dynamic[$CellContext`$showprogress],
		CellContext -> Notebook,
		Background -> None,
		CellFrame -> 1,
		CellFrameColor -> GrayLevel[0.96],
		CellMargins -> 0,
		CellFrameMargins -> {{10, 0}, {4, 4}}]},

    	WindowSize -> {300, 500},
    	WindowTitle -> Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "OutlinePaletteWindowTitle"],
    	WindowElements -> {"VerticalScrollBar", "StatusArea"},
    	WindowFrameElements -> {"CloseBox", "ResizeArea"},
    	WindowMargins -> {{Automatic, 0}, {0, 0}},
    	AutoMultiplicationSymbol -> False,
    	Background -> GrayLevel[0.96],
    	ClosingAutoSave -> False,
    	Saveable -> False,
    	System`BlinkingCellInsertionPoint -> False,
    	System`CellInsertionPointCell -> None,
    	StyleDefinitions -> "Palette.nb",
    	TaggingRules -> {"PresenterSettings" -> {"OutlineThumbnails" -> Table[ToString@i -> "Generic", {i, Length@Cells[nb1, CellStyle -> "SlideShowNavigationBar"]}]}},
    	CellContext -> Notebook]]);




(**************************************

  CreatePresenterNotesPalette
	Create palette of clickable thumbnails
	in:  None
	out: Palette notebook

***************************************)
Options[CreatePresenterNotesPalette] = {
	"Background" -> GrayLevel[1],
	"BackgroundHover" -> RGBColor[0.91, 0.9725, 0.992],
	"Borders" -> GrayLevel[0.75],
	"WindowTitle" -> None
}
CreatePresenterNotesPalette[]:= CreatePresenterNotesPalette[InputNotebook[]];
(* first check if notebook contains SideNotes *)
CreatePresenterNotesPalette[nbObj_, opts:OptionsPattern[]]:=
Module[{presenterNotesCells, numberOfDisplays, systemInformation, backgroundColor, windowTitle,
	borderColor, backgroundHoverColor, dockedCell, windowMargins, stylesheet, paletteSize, noNotesFoundCell, palette},

(*
If[(* No SideNotes found within notebook *)
	Cells[nbObj, CellStyle -> Flatten[{"SlideShowNavigationBar", CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "SlideBreakStyles"}, {}], "SlideShowNavigationBar", "SideNote",
																						"SideCode"}]] === {},

	MessageDialog[Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "NoSideNotesPresentDialog"]],
*)
	backgroundColor = OptionValue["Background"];
	backgroundHoverColor = OptionValue["BackgroundHover"];
	borderColor = OptionValue["Borders"];
(*
		If[windowTitle === None,
			Dynamic[FEPrivate`FrontEndResource["PresenterToolStrings", "SideNotesPaletteWindowTitle"]],
		(* ELSE *)
			StringJoin[Dynamic[FEPrivate`FrontEndResource["PresenterToolStrings", "SideNotesPaletteWindowTitle"]], ": ", windowTitle]
		];
*)
	paletteSize = {300, 500};
	PresenterNotes`Private`slideNumber = 0;

	If[IntegerQ@# && # > 1,
		SelectionMove[nbObj, Before, Notebook];
		NotebookWrite[nbObj, Cell["", "SlideShowNavigationBar", "FirstSlide", CellTags -> "SlideShowHeader"]]]&[PresenterTools`Private`FirstPositionOFASlideBreakCell[nbObj]];

	presenterNotesCells = GetPresenterNotesCells[nbObj];

	(* use specific stlyes from stylesheet *)

	dockedCell =
	With[{$CellContext`nb$ = nbObj, border = borderColor, background = backgroundColor, backgroundHover = backgroundHoverColor},
		{
			PresenterTools`OutlinePaletteNavigationBar[$CellContext`nb$, "MoveToCell" -> True],
			Cell[BoxData[ToBoxes@
				Panel[
					Grid[{{
						Style[
							(* Insert Presenter Note *)
							(* Don't display button if in "Slideshow Presentation" *)
							Button[
								Mouseover[
									Panel[
										Row[{
											Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, "InsertPresenterNote.png"]],
											Style[Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "AddSideNotesLabel"], 11, FontFamily :> CurrentValue["PanelFontFamily"]]
										}, Spacer[5], Alignment -> {Left, Center}], Appearance -> None,
											Background -> background, ImageSize -> Full, FrameMargins -> 4
									],
									Panel[
										Row[{
										Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, "InsertPresenterNote.png"]],
										Style[Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "AddSideNotesLabel"], 11, FontFamily :> CurrentValue["PanelFontFamily"]]
										}, Spacer[5], Alignment -> {Left, Center}], Appearance -> None,
										Background -> backgroundHover, ImageSize -> Full, FrameMargins -> 4
									]
								],
								(
									SetSelectedNotebook[$CellContext`nb$];
									(* where to write new cell *)
									If[CellInsertionPointQ[$CellContext`nb$],
										If[
											SelectionMove[$CellContext`nb$, Previous, Cell, AutoScroll -> False];
											If[SelectedCells[$CellContext`nb$] === {},
												SelectionMove[$CellContext`nb$, All, Notebook, AutoScroll -> False];
												SelectionMove[$CellContext`nb$, After, Cell];
												InsertStyledCell["SideNote", $CellContext`nb$],
											(* ELSE *)
												SelectionMove[$CellContext`nb$, After, Cell, AutoScroll -> False];
												InsertStyledCell["SideNote", $CellContext`nb$]
											],
										(* ELSE *)
											SelectionMove[$CellContext`nb$, All, Notebook, AutoScroll -> False];
											SelectionMove[$CellContext`nb$, After, Cell];
											InsertStyledCell["SideNote", $CellContext`nb$]
										],
									(* ELSE *)
										If[CellBracketSelectedQ[$CellContext`nb$],
											InsertStyledCell["SideNote", $CellContext`nb$],
										(* ELSE *)
											MoveSelectionToBetweenCells[$CellContext`nb$];
											InsertStyledCell["SideNote", $CellContext`nb$]
										];
									]
								),
								Appearance -> {"Default" -> None, "Pressed" -> None}
							]
							(* Don't display button if in "Slideshow Presentation" *)
,
							ShowContents ->
								Dynamic[If[CurrentValue[$CellContext`nb$, ScreenStyleEnvironment] === "Slideshow Presentation", False, True]]

						],
						MouseAppearance[
						Button[
							Mouseover[
								Panel[
									Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, "RefreshPresenterNotes.png"]],
									Background -> background,
									Appearance -> None
								],
								Panel[
									Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, "RefreshPresenterNotes.png"]],
									Background -> backgroundHover,
									Appearance -> None
								]
							],
							Module[{notesCellList, bn = ButtonNotebook[],
									loadingText = Pane[Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "Loading"], ImageSize -> {Automatic, 50}, Alignment -> Bottom]
									},

								If[IntegerQ@# && # > 1,
									SelectionMove[$CellContext`nb$, Before, Notebook];
		NotebookWrite[$CellContext`nb$, Cell["", "SlideShowNavigationBar", "FirstSlide", CellTags -> "SlideShowHeader"]]]&[PresenterTools`Private`FirstPositionOFASlideBreakCell[$CellContext`nb$]];

								PresenterNotes`Private`slideNumber = 0;
								(* write notes into notebook *)
								SetOptions[bn, Deployed -> False, Editable -> True];
								(*CurrentValue[bn, WindowClickSelect] = True;*)
								SelectionMove[bn, All, Notebook, AutoScroll -> False];
								FrontEndTokenExecute[bn, "Clear"];
								NotebookWrite[bn,
										Cell[BoxData@ToBoxes@Grid[{{Style[loadingText, "FontFamily" -> "Source Sans Pro", FontSize -> 14],
														Spacer[25]},
													{ProgressIndicator[Appearance -> "Necklace", ImageSize -> 30], Spacer[25]}}], "Output",
											TextAlignment -> Center, CellTags -> "Loader"], All];
								notesCellList = PresenterTools`GetPresenterNotesCells[$CellContext`nb$];
								NotebookFind[bn, "Loader", All, CellTags];
								If[MatchQ[notesCellList, {}],
									NotebookDelete[bn],
								(* ELSE *)
									NotebookWrite[bn, notesCellList, None, AutoScroll -> False];
								];
								SetOptions[bn, WindowClickSelect -> Inherited, Deployed -> Inherited, Editable -> Inherited];
								SelectionMove[bn, Before, Notebook, AutoScroll -> False]
							],
							Tooltip -> Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "SideNotesRefreshTooltip"], TooltipDelay -> .25,
							Appearance -> {"Default" -> None, "Pressed" -> None}, Method -> "Queued"
						], "LinkHand"
						]
						}},
						ItemSize -> {{Scaled[0.9], Scaled[0.1]}},
						Spacings -> 0,
						Alignment -> {{Left, {Right}}, Center}
						],
						Background->background,
						FrameMargins -> {{6, 6}, {3, 3}},
						ImageMargins -> 0
					]
				],
					CellFrame -> {{0.01, 0.01}, {1, 0.01}},
					CellFrameMargins->0,
					CellFrameColor->border,
					Background->background,
					CellMargins -> {{0, 0}, {0, 0}}
			]
		}
	];


	(* support multiple monitors *)
	systemInformation = GetSystemInformation[];
	numberOfDisplays = NumberOfDisplays[systemInformation];
	(* determine palette WindowMargins for centering palette on 2nd screen, when available *)
	windowMargins =
		If[numberOfDisplays > 1,
			MultipleDisplayMargins[systemInformation, paletteSize, 2],
		(* ELSE *)
			{{Automatic, 0}, {0, 0}}
		];

	(* Create palette *)
	palette = NotebookPut@With[{margins = windowMargins, ss = stylesheet,
			loadingText = Pane[Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "Loading"], ImageSize -> {Automatic, 50}, Alignment -> Bottom],
			dockedcell = dockedCell, windowSize = paletteSize, spacerCell = BottomSpacerButtonCell[10],
			background = backgroundColor, border = borderColor, $CellContext`nb$ = nbObj },
			Notebook[{
				Cell[BoxData@ToBoxes@Grid[{{Style[loadingText, "FontFamily" -> "Source Sans Pro", FontSize -> 14],
							Spacer[25]},
							{ProgressIndicator[Appearance -> "Necklace", ImageSize -> 30], Spacer[25]}}], "Output",
					TextAlignment -> Center, CellTags -> "Loader"],
				spacerCell
			},
			WindowSize -> windowSize,
			WindowMargins -> margins,
			WindowTitle -> Dynamic[FEPrivate`FrontEndResource["PresenterToolStrings", "SideNotesPaletteWindowTitle"]],
			WindowElements -> {"VerticalScrollBar", "StatusArea", "MagnificationPopUp"},
			WindowFrameElements -> {"CloseBox", "ResizeArea"},
			ClosingAutoSave -> False,
			Saveable -> False,
			Deployed -> False,
			WindowClickSelect -> False,
(*
			WindowFloating -> True,
			WindowFrame -> "Palette",
			WindowToolbars -> {},
			WindowClickSelect -> True,
			Editable -> False,
			ScrollingOptions -> {"VerticalScrollRange" -> Fit},
			Editable -> False,
			ShowCellBracket -> False,
*)
			ScreenStyleEnvironment -> "Palettes",
			AutoMultiplicationSymbol -> False,
			Background -> background,
			System`BlinkingCellInsertionPoint -> False,
			System`CellInsertionPointCell -> None,
			System`CellInsertionPointColor -> background,
			NotebookEventActions -> {
				"WindowClose" :> (
							FE`Evaluate[FEPrivate`FindAndClickCancelButton[]];
							(*CurrentValue[$CellContext`nb$, {TaggingRules, "PresenterSettings", "SideNotesPalette", "PaletteNumber"}] = 0;*)
							CurrentValue[$CellContext`nb$, {TaggingRules, "PresenterSettings", "Dialogs", "SideNotes"}] = 0;
							)
				},
(*
			NotebookDynamicExpression :> (
				If[Not[MemberQ[Notebooks[], $CellContext`nb$]], NotebookClose[EvaluationNotebook[]]]
			),
*)
			DockedCells -> dockedcell,
			StyleDefinitions -> (*ss*)

Notebook[{
	Cell[StyleData[StyleDefinitions -> "Palette.nb"]],

Cell[StyleData["PresenterNotesCodePalette", StyleDefinitions -> StyleData["Input"]],
 CellMargins->{{10, 5}, {6, 15}},
 CellFrame->{{2, 0}, {0, 0}},
 CellFrameMargins->{{15, 15}, {0, 0}},
 CellFrameColor->GrayLevel[0.85],
 System`GeneratedCellStyles->{"Output"->"SideCodeOutput"},
 MenuSortingValue->None,
 ShowCellBracket -> False,
 FontSize->12,
 FontColor->GrayLevel[0.4],
 Selectable->True,
 Background->GrayLevel[0.98]],

Cell[StyleData["PresenterNotesCodeOutputPalette", StyleDefinitions -> StyleData["Output"]],
 CellMargins->{{10, 5}, {12, 6}},
 CellFrame->{{2, 0}, {0, 0}},
 CellFrameMargins->{{15, 15}, {0, 0}},
 CellFrameColor->GrayLevel[0.85],
 MenuSortingValue->None,
 FontSize->12,
 FontColor->GrayLevel[0.4],
 ShowCellBracket -> False,
 Selectable->True,
 Background->background],

Cell[StyleData["PresenterNotesTextPalette"],
 CellMargins->{{10, 5}, {10, 14}},
 CellFrame->{{2, 0}, {0, 0}},
 CellFrameMargins->{{15, 15}, {0, 0}},
 CellFrameColor->GrayLevel[0.85],
 LineSpacing->{1, 3},
 MenuSortingValue->None,
 FontFamily:>CurrentValue["PanelFontFamily"],
 FontSize->12,
 FontColor->GrayLevel[0.5],
 ShowCellBracket -> False,
 Selectable->True,
 Background->background
],

Cell[StyleData["SlideHeadingPalette"],
 CellMargins->{{10, 5}, {10, 10}},
 CellFrame->{{2, 0}, {0, 0}},
 CellFrameMargins->{{15, 15}, {8, 8}},
 CellFrameColor->GrayLevel[0.85],
(* System`CellElementSpacings->{"ClosedGroupTopMargin"->30}, *)
 CellGroupingRules->{"SectionGrouping", -5},
 MenuSortingValue->None,
 FontFamily:>CurrentValue["PanelFontFamily"],
 FontSize->12,
 FontColor->GrayLevel[0.5],
 CellElementSpacings->{"CellMinHeight"->1},
 ShowCellBracket -> False,
 Selectable->True,
 Background->background,
 CellFrameLabels -> {{None, None}, {None, None}}],

Cell[StyleData["SlideBreaksPalette"],
 CellMargins->{{10, 5}, {9, 12}},
 CellFrame->{{2, 0}, {0, 0}},
 CellFrameMargins->{{15, 15}, {8, 8}},
 CellFrameColor->GrayLevel[0.85],
(* System`CellElementSpacings->{"ClosedGroupTopMargin"->30},*)
 CellGroupingRules->{"SectionGrouping", -5},
 MenuSortingValue->None,
 FontFamily:>CurrentValue["PanelFontFamily"],
 FontSize->12,
 FontColor->GrayLevel[0.4],
 FontWeight->Bold,
 CellElementSpacings->{"CellMinHeight"->1},
 ShowCellBracket -> False,
 Selectable->True,
 Background->background,
 CellFrameLabels -> {{None, None}, {None, None}}]},
StyleDefinitions->"StylesheetFormatting.nb",
System`TrackCellChangeTimes -> False
]

		]

];
	(* gather PresenterNotes from notebook
  	   and format the cells for display within
  	   palette
  	   *)
(*	presenterNotesCells = GetPresenterNotesCells[nbObj];*)
	NotebookFind[palette, "Loader", All, CellTags];

	If[MatchQ[presenterNotesCells, {}],
		NotebookDelete[palette],
	(* ELSE *)
		NotebookWrite[palette, presenterNotesCells, All, AutoScroll -> False]
	];

	palette
];

CreatePresenterNotesPalette[a__] := Message[CreatePresenterNotesPalette::argx, {a}];
CreatePresenterNotesPalette::argx = "Argument should be a NotebookObject: `1`";
CreatePresenterNotesPalette::info = "`1`";






(*******************************************************

 OpenPresenterPalette

	in: NotebookObject, palette name
	out: NotebookPut @ palette

*******************************************************)

Options[OpenPresenterPalette] = {
	"UseNotebookPut" -> False
}

OpenPresenterPalette[nbObj_, type_String, opts___?OptionQ]:=
Module[{useNotebookPut = ("UseNotebookPut" /. {opts} /. Options[OpenPresenterPalette]),
	notebookNumber, paletteObject, paletteExpression, notebooks, allNotebookNumbers, loggedNumber, pos, paletteTag, windowTitle},

	notebooks = Notebooks[];
	allNotebookNumbers = Part[#, 2]& /@ notebooks;
	notebookNumber = Part[nbObj, 2];
(*	loggedNumber = CurrentValue[nbObj, {TaggingRules, "PresenterSettings", type, "PaletteNumber"}]; *)
	loggedNumber = CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "Dialogs", type}];
	pos = Position[allNotebookNumbers, loggedNumber];
	windowTitle = "WindowTitle" /. NotebookInformation[nbObj];

	If[Head[notebookNumber] === Integer,
		If[(loggedNumber === Inherited) || (loggedNumber === 0),
			paletteExpression = GetPaletteNotebook[nbObj, type, "WindowTitle" -> windowTitle];
			Which[
				Head[paletteExpression] === Notebook,
					paletteObject = NotebookPut@paletteExpression,
				Head[paletteExpression] === NotebookObject,
					paletteObject = paletteExpression,
				True, Return[]
			];
			CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "Dialogs", type}] = Part[paletteObject, 2];
			(*CurrentValue[nbObj, {TaggingRules, "PresenterSettings", type, "PaletteNumber"}] = Part[paletteObject, 2]*),
			If[pos =!= {},
				SetSelectedNotebook[First@Part[Notebooks[], Flatten@pos]],
			(* ELSE *)
				paletteExpression = GetPaletteNotebook[nbObj, type, "WindowTitle" -> windowTitle];
				If[paletteExpression =!= $Failed,
                	paletteObject = If[TrueQ@useNotebookPut, NotebookPut@paletteExpression, paletteExpression];
					CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "Dialogs", type}] = Part[paletteObject, 2];
					(*CurrentValue[nbObj, {TaggingRules, "PresenterSettings", type, "PaletteNumber"}] = Part[paletteObject, 2]*),
				(* ELSE *)
					Return[]
				]
			]
		],
	(* ELSE *)
		Message[OpenPresenterPalette::info, notebookNumber];
	]
];
OpenPresenterPalette[a___]:= Message[OpenPresenterPalette::argx, {a}];
OpenPresenterPalette::argx = "`1`";
OpenPresenterPalette::info = "Info `1`";



GetPaletteNotebook[nbObj_, "AddNewStylePalette", opts___?OptionQ]:=
	PresenterTools`Styles`NewCellStyleDialog[nbObj, opts];

GetPaletteNotebook[nbObj_, "OutlinePalette", opts___?OptionQ]:=
	CreateOutlinePalette[nbObj, opts];

GetPaletteNotebook[nbObj_, "PresenterControlsPalette", opts___?OptionQ]:=
	SetPresenterControlsDialog[nbObj, opts];

GetPaletteNotebook[nbObj_, "SideNotesPalette", opts___?OptionQ]:=
	CreatePresenterNotesPalette[nbObj, opts];
(*
Module[{},
	If[ContainsSideNotesQ[nbObj],
		CreatePresenterNotesPalette[nbObj, opts],
	(* ELSE *)
		MessageDialog[
			Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "NoSideNotesPresentDialog"] ];
		$Failed
	]
]
*)
GetPaletteNotebook[nbObj_, "SlideBreakDefaults", opts___?OptionQ]:=
	AutomaticSlideBreakDialog[nbObj, opts];

GetPaletteNotebook[nbObj_, "ThemeOptionsPalette", opts___?OptionQ]:=
	ThemeOptionsDialog[nbObj, opts];

(*
GetPaletteNotebook[nbObj_, "GetStylesheetFromNotebook", opts___?OptionQ]:=
	GetStylesheetFromNotebookDialog[nbObj, opts];
*)



(*******************************************************

 ContainsSideNotesQ

	in: xxx
	out: yyy

*******************************************************)

ContainsSideNotesQ[nbObj_]:=
Module[{cells},

	cells = Cells[nbObj, CellStyle -> {"SideNote", "SideCode", "SideCodeOutput"}];

	If[(Head[cells] === List) && (Length[cells] > 0),
		True,
	(* ELSE *)
		False
	]

];
ContainsSideNotesQ[a___]:= Message[ContainsSideNotesQ::argx, {a}];
ContainsSideNotesQ::argx = "`1`";
ContainsSideNotesQ::info = "`1`";




(*******************************************************

 CloseChildNotebookDynamicWrapper

	in: expression
	out: DynamicWrapper[...]

*******************************************************)

CloseChildNotebookDynamicWrapper[nbObj_, con_]:=
Module[{},
	With[{content = con, no = nbObj},
	 	(* Close dialog when parent notebook is closed *)
	 	DynamicWrapper[
   			content,
	 		If[Not@MemberQ[Notebooks[], no],
	 			NotebookClose[EvaluationNotebook[]],
	 		(* ELSE *)
	 			content
	 		], SynchronousUpdating -> False
	 	]
	]
];
CloseChildNotebookDynamicWrapper[a___]:= Message[CloseChildNotebookDynamicWrapper::argx, {a}];
CloseChildNotebookDynamicWrapper::argx = "`1`";
CloseChildNotebookDynamicWrapper::info = "`1`";




(*******************************************************

 GetPresenterNotesCells

	in: NotebookObject
	out: List of Cells

*******************************************************)
(*
GetPresenterNotesCells[nbObj_NotebookObject /; ContainsSideNotesQ[nbObj]]:=
*)
GetPresenterNotesCells[nbObj_NotebookObject, opts___?OptionQ]:=
Module[{presenterNotesCells, slideBreakStyles, stylesheet, slideBreakHeadingStyles},

	slideBreakStyles = {"SlideShowNavigationBar"};
	slideBreakHeadingStyles = CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "SlideBreakStyles"}, {}];
	stylesheet = PresenterTools`Styles`GetStylesheet[nbObj];

	(* gather presenternotes cells *)
	presenterNotesCells = Cells[nbObj, CellStyle -> Flatten[{slideBreakStyles, slideBreakHeadingStyles, "SlideShowNavigationBar",
														"SideNote", "SideCode" }]];
	presenterNotesCells = Flatten[{NotebookRead /@ presenterNotesCells}];

	(* change cell styles for use in notes palette *)
	presenterNotesCells = presenterNotesCells /. ce:Cell[a_, sty_String /; MemberQ[slideBreakHeadingStyles, sty], o___] :>
								Cell[a, "SlideHeadingPalette",
												FontColor -> Dynamic[
													If[CurrentValue[NotebookSelection[nbObj], {"CounterValue", "SlideShowNavigationBar"}] === CurrentValue[{"CounterValue", "SlideHeading"}],
														GrayLevel[0],
													(* ELSE *)
														GrayLevel[0.8]
													]
												],
												CellFrameColor -> Dynamic[
													If[CurrentValue[NotebookSelection[nbObj], {"CounterValue", "SlideShowNavigationBar"}] === CurrentValue[{"CounterValue", "SlideHeading"}],
														RGBColor[0.6665, 0.8355, 0.992],
													(* ELSE *)
														GrayLevel[0.8]
													]
												], o];
	presenterNotesCells = presenterNotesCells /. ce:Cell[a_, "SlideShowNavigationBar", o___] :>
								ConvertPresenterNoteSlideBreaks[ce, nbObj];
	presenterNotesCells = presenterNotesCells /. Cell[con_, "SideNote", o___] :>
								Cell[con, "PresenterNotesTextPalette", Selectable -> True,
												FontColor -> Dynamic[
													If[CurrentValue[NotebookSelection[nbObj], {"CounterValue", "SlideShowNavigationBar"}] === CurrentValue[{"CounterValue", "SlideHeading"}],
														GrayLevel[0],
													(* ELSE *)
														GrayLevel[0.8]
													]
												],
												CellFrameColor -> Dynamic[
													If[CurrentValue[NotebookSelection[nbObj], {"CounterValue", "SlideShowNavigationBar"}] === CurrentValue[{"CounterValue", "SlideHeading"}],
														RGBColor[0.6665, 0.8355, 0.992],
													(* ELSE *)
														GrayLevel[0.8]
													]
												], o];
	presenterNotesCells = presenterNotesCells /. Cell[con_, "SideCode", o___] :>
								Cell[con, "PresenterNotesCodePalette",
												CellFrameColor -> Dynamic[
													If[CurrentValue[NotebookSelection[nbObj], {"CounterValue", "SlideShowNavigationBar"}] === CurrentValue[{"CounterValue", "SlideHeading"}],
														RGBColor[0.6665, 0.8355, 0.992],
													(* ELSE *)
														GrayLevel[0.9]
													]
												], o];
	presenterNotesCells = presenterNotesCells /. Cell[con_, "SideCodeOutput", o___] :>
								Cell[con, "PresenterNotesCodeOutputPalette", o];
	presenterNotesCells = presenterNotesCells /. ce:Cell[a_, "PresenterNotesCodePalette", o___] :>
								CreatePresenterCodeLinkCell[ce, nbObj, stylesheet];

	(* return result *)
	Flatten[{presenterNotesCells}]

];
GetPresenterNotesCells[_]:= {};
(*GetPresenterNotesCells[a___]:= Message[GetPresenterNotesCells::argx, {a}];*)
GetPresenterNotesCells::argx = "`1`";
GetPresenterNotesCells::info = "`1`";





PresenterNotes`Private`slideNumber = 0;
ConvertPresenterNoteSlideBreaks[Cell[con_, sty_String, o___], nbObj_]:=
Module[{cells, allCells, breakPos, slideBreakMarker},
	PresenterNotes`Private`slideNumber = PresenterNotes`Private`slideNumber + 1;

	(* list of all cells in notebook *)
	allCells = Cells[nbObj];

	(* list of all slidebreak cells in notebook *)
	cells = Cells[nbObj, CellStyle -> "SlideShowNavigationBar"];

	(* slide break positions *)
	breakPos = Position[allCells, #] & /@ cells;

(*	count = Cells[nbObj, CellStyle -> Alternatives @@ Join[{"SlideShowNavigationBar"}, CurrentValue[nbObj, {TaggingRules,"PresenterSettings", "SlideBreakStyles"}]]];*)

	With[{
		slide = PresenterNotes`Private`slideNumber,
		slideLabel = Row[{
						(* 'SLIDE' *)
						Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "SLIDE"],
						ToString[PresenterNotes`Private`slideNumber]
					}, Spacer[2]],
		tooltip = StringJoin["Go to slide ", ToString[PresenterNotes`Private`slideNumber]],
		cellObj = Part[cells, PresenterNotes`Private`slideNumber],
		nbobj = nbObj
		},
		slideBreakMarker =
			Cell[BoxData[
				ToBoxes@
					MouseAppearance[
						(* GOTO SLIDE BUTTON *)
						Button[
							Grid[{{
								Style["",
									FontSize -> 2,
									ShowStringCharacters -> False,
									CellElementSpacings->{"CellMinHeight"->1}],
								Pane[
									Style[slideLabel,
										FontFamily:>CurrentValue["PanelFontFamily"],
										FontSize->11,
										FontTracking->1,
										FontColor->GrayLevel[0]
										],
									ImageSize -> {Full, 12}, Alignment -> Center
								],
								Item[
									Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, "JumpToSlide.png"]],
									Alignment -> Right
									]
								}},
								ItemSize -> {{Scaled[.32], Scaled[0.32], Scaled[0.32]}},
								Alignment -> {{Left, Center, Right}, Center}(*,
								Frame -> All*)
							],
							(
								(* taken from pre-11.2 slide navigator popup *)
								SetSelectedNotebook[nbObj];
								SelectionMove[cellObj, All, Cell, AutoScroll -> False];
								SelectionMove[nbObj, After, Cell, AutoScroll -> False];
								NotebookWrite[nbObj, Cell["", Deletable -> True, ShowCellBracket -> False], All];
								NotebookDelete[nbObj];
								SelectionMove[nbObj, Next, Cell];
								SelectionMove[nbObj, Before, Cell, AutoScroll -> False];
							),
							Appearance -> {"Default" -> None, "Pressed" -> None},
							Tooltip-> tooltip,
							TooltipDelay -> .25,
							BoxID -> "Thumbnail" <> ToString[PresenterNotes`Private`slideNumber]
						], "LinkHand"
					]
				],
				CellGroupingRules->{"SectionGrouping", -5},
				CellMargins->{{0, 0}, {6, 20}},
				CellFrameMargins -> {{Inherited, Inherited}, {0, 0}},
				CellElementSpacings->{"CellMinHeight"->1},
				CellTags -> "SlideBreakCell",
				CounterIncrements->"SlideHeading",
				Background -> Dynamic[
					If[CurrentValue[NotebookSelection[nbobj], {"CounterValue", "SlideShowNavigationBar"}] === slide,
						RGBColor[0.6665, 0.8355, 0.992],
					(* ELSE *)
						GrayLevel[0.9]
					]
				]
			]
	];

	With[{slideMarker = slideBreakMarker},
		{
			slideMarker
		}
	]
];




(*******************************************************

 MultipleDisplayMargins

	in: Result of
	out: yyy

*******************************************************)

MultipleDisplayMargins[{monitors__List}, notebook : {w_, h_} : {0, 0}, display_Integer:2] :=
Module[{x1, x2, y1, y2, width, height, monitorData, screenArea, mainMonitorData, mainScreenArea},
	(* monitor1:{opt1s__?OptionQ}, monitor2:{opt2s__?OptionQ}, _ *)
	mainMonitorData = Association@@First[{monitors}];
	monitorData = Association@@Part[{monitors}, display];

	mainScreenArea = mainMonitorData["FullScreenArea"];
	screenArea = monitorData["FullScreenArea"];
(*
	{x1, x2, y1, y2} = Flatten@Take[screenArea, 4];
	width = Ceiling@N[((x1 + x2) - w)/2];
	height = Ceiling@N[((y1 + y2) - h)/2];
	{width, height}
*)
	PositionNotebookOnMultipleDisplays[mainScreenArea, screenArea, notebook]

]

PositionNotebookOnMultipleDisplays[
	{h1 : {l1_, r1_}, v1 : {b1_, t1_}},
	{h2 : {l2_, r2_}, v2 : {b2_, t2_}},
	notebook : {w_, h_} : {0, 0}
	] :=
Module[{xpos, ypos},

	xpos = Ceiling@N[((l2 + r2) - w)/2];
	ypos = Ceiling@N[((b2 + t2) - h)/2];

	Which[
		h1 === h2,
			Which[
				b1 === t2,
					(* Top *)
					{{xpos, Automatic}, {Automatic, ypos}},
				t1 === b2,
					(* bottom *)
					{{xpos, Automatic}, {Automatic, ypos}},
				True,
					Automatic
			],
		v1 === v2,
			Which[
				l1 === r2,
					(* Left *)
					{{xpos, Automatic}, {Automatic, ypos}},
				r1 === l2,
					(* Right *)
					{{xpos, Automatic}, {Automatic, ypos}},
				True,
					Automatic
			],
		True,
			Automatic
	]

];
MultipleDisplayMargins[a___]:= Message[MultipleDisplayMargins::argx, {a}];
MultipleDisplayMargins::argx = "`1`";
MultipleDisplayMargins::info = "`1`";




(*******************************************************

 CreatePresenterCodeLinkCell

	in: xxx
	out: yyy

*******************************************************)

CreatePresenterCodeLinkCell[ce:Cell[con_, sty_String, o___], nbObj_, stylesheet_Notebook]:=
Module[{image, code, button, style = sty, evaluatable},

	evaluatable = If[style === None, False, CurrentValue[nbObj, {StyleDefinitions, style, Evaluatable}]];

	With[{tag = StringJoin @@ (ToString /@ Date[])},
	image =
		EventHandler[
			MouseAppearance[
				Framed[
				Tooltip[
				RemoveBackground@
					Rasterize[
						Notebook[{
						Cell[con, style, FontColor -> GrayLevel[0], Magnification -> 1,  FontSize -> 12, CellFrameMargins -> 0, CellMargins -> 0, o]
						}, StyleDefinitions -> stylesheet, WindowSize -> {260, Automatic}]
						], Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "SideNotesPasteIntoNotebookTooltip"], TooltipDelay -> 0.25],
				RoundingRadius -> 3, FrameStyle -> GrayLevel[0.85],
				FrameMargins -> 8, ImageSize -> Full,
				Background -> Dynamic[If[CurrentValue["MouseOver"],
					RGBColor[0.91, 0.9725, 0.992], GrayLevel[0.975]]]], "LinkHand"],

				{
					{"MouseClicked", 1} :>
				(
					nbObj1 = (nbObj /. FrontEndObject[_] -> $FrontEnd);
					SetSelectedNotebook[nbObj1];
				If[Cells[nbObj1, CellTags -> tag] === {},
					(* where to write new cell *)
					Module[{lnkre},
						Which[(* One or more cell brackets selected. *)
							MatchQ[Developer`CellInformation[nbObj1], {{__, "CursorPosition" -> "CellBracket", __}, ___}],
							SelectionMove[nbObj1, After, Cell];
							NotebookWrite[nbObj1, Cell[con, "Input", o, CellTags -> tag], All],
							(* Cursor inside a cell, but not at cell bracket *)
							SelectedCells[nbObj1] =!= {} && Not@MatchQ[Developer`CellInformation[nbObj1], {{__, "CursorPosition" -> "CellBracket", __}, ___}],
							While[(LinkWrite[$ParentLink, FrontEnd`CellInformation[nbObj1]];
								lnkre = LinkRead[$ParentLink]);
								(lnkre =!= $Failed && Not[MemberQ["CursorPosition" /. lnkre, "CellBracket"]]),
								FrontEndExecute[FrontEnd`SelectionMove[nbObj1, All, Cell, AutoScroll -> False]]];
							SelectionMove[nbObj1, After, Cell];
							NotebookWrite[nbObj1, Cell[con, "Input", o, CellTags -> tag], All],
							Cells[nbObj1] === {},
							NotebookWrite[nbObj1, Cell[con, "Input", o, CellTags -> tag], All],
							Cells[nbObj1, CellStyle -> "SlideShowNavigationBar"] === {},
							SelectionMove[nbObj1, After, Notebook];
							NotebookWrite[nbObj1, Cell[con, "Input", o, CellTags -> tag], All],
							True,
							SelectionMove[nbObj1, Next, Cell, AutoScroll -> False];
							If[(* The cursor is at the bottom of the notebook. *)
 								SelectedCells[nbObj1] === {},
 								NotebookWrite[nbObj1, Cell[con, "Input", o, CellTags -> tag], All],
 								If[MatchQ[Developer`CellInformation[SelectedCells[nbObj1][[1]]],
 														{"Style" -> {"SlideShowNavigationBar", "FirstSlide"} | "SlideShowNavigationBar", __}],
									SelectionMove[nbObj1, Before, Cell, AutoScroll -> False];
									NotebookWrite[nbObj1, Cell[con, "Input", o, CellTags -> tag], All],
									If[NotebookFind[nbObj1, "SlideShowNavigationBar", Next, CellStyle] === $Failed,
										SelectionMove[nbObj1, After, Notebook];
										NotebookWrite[nbObj1, Cell[con, "Input", o, CellTags -> tag], All],
										SelectionMove[nbObj1, Before, Cell];
										NotebookWrite[nbObj1, Cell[con, "Input", o, CellTags -> tag], All]]]]]];
					If[CurrentValue["ShiftKey"],
						SelectionMove[nbObj, Previous, Cell];
						FrontEndExecute[{FrontEndToken[nbObj, "SelectionEvaluate"]}]
					],
					NotebookFind[nbObj1, tag, All, CellTags];
					SelectionEvaluateCreateCell[nbObj1]]
				),
					{"MouseClicked", 2} :>
						(
							SetSelectedNotebook[nbObj];
							(* where to write new cell *)
							If[CellInsertionPointQ[nbObj],
								(* check to see if the insertion point is at the top of the notebook.
								   ( select the previous cell, if selection === {} you're at the top of the notebook )
									If selection =!= {} then move after the selection cell and write the cell.
									If selection === {} then write the cell.
								*)
								SelectionMove[nbObj, Previous, Cell, AutoScroll -> False];
								If[SelectedCells[nbObj] === {},
									FrontEndTokenExecute[nbObj, "SelectAll"];
(*									SelectionMove[nbObj, All, Notebook, AutoScroll -> False];*)
									SelectionMove[nbObj, After, Cell];
									NotebookWrite[nbObj, Cell[con, "Input", o], All],
								(* ELSE *)
									SelectionMove[nbObj, After, Cell, AutoScroll -> False];
									NotebookWrite[nbObj, Cell[con, "Input", o], All]
								],
							(* ELSE *)
								If[CellBracketSelectedQ[nbObj],
									NotebookWrite[nbObj, Cell[con, "Input", o], All],
								(* ELSE *)
									MoveSelectionToBetweenCells[nbObj];
									NotebookWrite[nbObj, Cell[con, "Input", o], All]
								]
							];
							FrontEndExecute[{FrontEndToken[nbObj, "SelectionEvaluate"]}]
						)
				}
		]];

	Cell[BoxData[ToBoxes@image], sty,
		Background -> None,
		CellFrameColor -> Dynamic[
			If[CurrentValue[NotebookSelection[nbObj], {"CounterValue", "SlideShowNavigationBar"}] === CurrentValue[{"CounterValue", "SlideHeading"}],
				RGBColor[0.6665, 0.8355, 0.992],
			(* ELSE *)
				GrayLevel[0.8]
			]
		],
		(*
		CellFrameLabels -> {{None,
			Cell[BoxData[ToBoxes@
						MouseAppearance[
							Button[
								Style[
									Mouseover[
										Panel[
											Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, "SideNotePaste.png"]],
											Background -> GrayLevel[.99], FrameMargins -> 4,
											Appearance -> None
										],
										Panel[
											Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, "SideNotePaste.png"]],
											Background -> RGBColor[0.91, 0.9725, 0.992], FrameMargins -> 4,
											Appearance -> None
										]
									], Selectable -> False
								],
								(* Paste and Evaluate *)
								(
									SetSelectedNotebook[nbObj];
									(* where to write new cell *)
									If[CellInsertionPointQ[nbObj],
										(* check to see if the insertion point is at the top of the notebook.
										   ( select the previous cell, if selection === {} you're at the top of the notebook )
											If selection =!= {} then move after the selection cell and write the cell.
											If selection === {} then write the cell.
										*)
										SelectionMove[nbObj, Previous, Cell, AutoScroll -> False];
										If[SelectedCells[nbObj] === {},
											SelectionMove[nbObj, All, Notebook, AutoScroll -> False];
											SelectionMove[nbObj, After, Cell];
											NotebookWrite[nbObj, Cell[con, "Input", o], All],
										(* ELSE *)
											SelectionMove[nbObj, After, Cell, AutoScroll -> False];
											NotebookWrite[nbObj, Cell[con, "Input", o], All]
										],
									(* ELSE *)
										If[CellBracketSelectedQ[nbObj],
											NotebookWrite[nbObj, Cell[con, "Input", o], All],
										(* ELSE *)
											MoveSelectionToBetweenCells[nbObj];
											NotebookWrite[nbObj, Cell[con, "Input", o], All]
										];
									];
									FrontEndExecute[{FrontEndToken[nbObj, "SelectionEvaluate"]}]
								),
								Appearance -> {"Default" -> None, "Pressed" -> None},
							 	Tooltip -> FEPrivate`FrontEndResource["PresenterToolStrings", "SideNotesPasteAndEvaluateTooltip"],
							 	TooltipDelay -> 0.25
							], "LinkHand"
						]
					],
				Background -> None,
				CellMargins -> 0
				]
			}, {None, None}},
			Background -> Dynamic[
				If[CurrentValue["MouseOver"], RGBColor[0.91, 0.9725, 0.992], GrayLevel[0.975]]
			],
			CellFrameMargins -> 4,
			CellFrame -> 2,
*)
		o]
];
CreatePresenterCodeLinkCell[a___]:= Message[CreatePresenterCodeLinkCell::argx, {a}];
CreatePresenterCodeLinkCell::argx = "`1`";
CreatePresenterCodeLinkCell::info = "`1`";





(*******************************************************

 NumberOfDisplays

	in: None
	out: Integer

*******************************************************)

NumberOfDisplays[]:= NumberOfDisplays[GetSystemInformation[]];

NumberOfDisplays[screenInformation_?OptionQ]:=
Module[{len},
	(* how many displays listed? *)
	len = Length[screenInformation];
	(* test for integer number of screens *)
	ret =
		If[IntegerQ[len],
			len,
		(* ELSE *)
			1
		];

	ret
]
NumberOfDisplays[a__]:= Message[NumberOfDisplays::argx, {a}];
NumberOfDisplays::argx = "`1`";
NumberOfDisplays::info = "`1`";





(*******************************************************

 MultipleDisplaysQ

	in: None
	out: True | False

*******************************************************)

MultipleDisplaysQ[]:= MultipleDisplaysQ[GetSystemInformation[]];

MultipleDisplaysQ[screenInformation_?OptionQ]:=
Module[{len},
	(* how many displays listed? *)
	len = Length[screenInformation];

	(* test for integer number of screens *)
	If[IntegerQ[len],
		(len > 1),
	(* ELSE *)
		False
	]

];
MultipleDisplaysQ[a___]:= Message[MultipleDisplaysQ::argx, {a}];
MultipleDisplaysQ::argx = "`1`";
MultipleDisplaysQ::info = "`1`";






(*******************************************************

 GetSystemInformation

	in: xxx
	out: yyy

*******************************************************)

GetSystemInformation[]:=
Module[{screenInformation},
	(* systemInformation *)
	screenInformation = SystemInformation["Devices"];
	screenInformation = OptionValue[screenInformation, "ScreenInformation"]
];
GetSystemInformation[a__]:= Message[GetSystemInformation::argx, {a}];
GetSystemInformation::argx = "`1`";
GetSystemInformation::info = "`1`";





(*******************************************************

 BottomSpacerButtonCell

	in: xxx
	out: yyy

*******************************************************)

BottomSpacerButtonCell[]:= BottomSpacerButtonCell[20];
BottomSpacerButtonCell[space_, cellOpts___?OptionQ]:=
Module[{res, opts},
	opts = {CellMargins->{{0, 0}, {0, 0}},
	CellBracketOptions->{"OverlapContent"->True},
	Selectable->False,
	ShowCellBracket->False,
	CellFrameMargins->{{0, 0}, {0, 0}},
	CellTags -> "ButtonSpacer"
	};

	res=
		Cell[BoxData[
			 ButtonBox[
			  TagBox[GridBox[{
			     {
			      PaneBox[" ",
			      ImageMargins->space,
			       ImageSize->Full]}
			    },
			    AutoDelete->False,
			    GridBoxItemSize->{"Columns" -> {{Automatic}}, "Rows" -> {{Automatic}}}],
			   "Grid"],
			  ButtonFunction:>SelectionMove[
			    ButtonNotebook[], Before, ButtonCell, AutoScroll -> False],
			  Evaluator->Automatic,
			  Method->"Preemptive",Appearance->{"Default" -> None, "Pressed" -> None}]], "Text",
			Sequence@@opts];
	ResetOptions[res, {cellOpts}]
];
BottomSpacerButtonCell[a___]:= Message[BottomSpacerButtonCell::argx, {a}];
BottomSpacerButtonCell::argx = "`1`";
BottomSpacerButtonCell::info = "`1`";







(*********************************************

   CELL INFORMATION / STYLE ROUTINES

*********************************************)








(**************************************

  GetCellInformation
	Call Developer`CellInformation

	in:  NotebookObject (defaults to InputNotebook[])
	out: Association or List of Associations

***************************************)

GetCellInformation[]:= GetCellInformation[InputNotebook[]];

GetCellInformation[nbObj_]:=
	GetCellInformation[nbObj, Developer`CellInformation[nbObj] ];

(* Return $Failed if cursor is between cells *)
GetCellInformation[nbObj_, $Failed]:= $Failed;

GetCellInformation[nbObj_, cellInfo_List]:=
Module[{},
(*
	cellInfo
*)
		(* selection is NOT between Cells so return Association *)
		If[Length[cellInfo] > 1,
			(Association@@#) & /@ cellInfo,
		(* ELSE *)
			Association@@cellInfo
		]
];
(*
multipleCellBracketsSelected[x_]:=
 MatchQ[x,{{"Style"->_,__},{"Style"->_,__},___}]
*)


MoveSelectionToBetweenCells[]:= MoveSelectionToBetweenCells[InputNotebook[]];
MoveSelectionToBetweenCells[nbObj_NotebookObject]:=
Module[{cellInfo},
	(* Check CellInformation *)
	cellInfo = GetCellInformation[nbObj];
	If[cellInfo =!= $Failed,
		(* Selection is NOT between Cells *)
		SelectionMove[nbObj, After, Cell];
	]
];
MoveSelectionToBetweenCells[a___]:= Message[MoveSelectionToBetweenCells::argx, {a}];
MoveSelectionToBetweenCells::argx = "Argument should be an NotebookObject: `1`";



CellInsertionPointQ[]:= CellInsertionPointQ[InputNotebook[]];
CellInsertionPointQ[nbObj_NotebookObject]:=
Module[{cellInfo},
	(* Check CellInformation *)
	cellInfo = GetCellInformation[nbObj];
	If[MatchQ[cellInfo, $Failed],
		True
	,
		False
	]
];





(*******************************************************

 GetFontFamilyOfSelection

	If no selction, return FontFamily of Notebook's DefaultNewCellStyle

	in: NotebookObject
	out: String FontFamily

*******************************************************)

GetFontFamilyOfSelection[]:= GetFontFamilyOfSelection[InputNotebook[], Developer`CellInformation[InputNotebook[]]];

GetFontFamilyOfSelection[$Failed]:= CurrentValue[FontFamily];

GetFontFamilyOfSelection[nbObj_]:= GetFontFamilyOfSelection[nbObj, Developer`CellInformation[nbObj]];

GetFontFamilyOfSelection[nbObj_, cellInfo_]:=
Module[{},

	If[cellInfo =!= $Failed,
		(* Selection is NOT between Cells *)
		(* get FontFamily of selection *)
		CurrentValue[NotebookSelection[nbObj], FontFamily],
	(* ELSE *)
		(* fall through case, use FontFamily of the Default New Cell *)
		(*CurrentValue[nbObj, FontFamily]*)
		CurrentValue[nbObj, {StyleDefinitions, CurrentValue[nbObj, DefaultNewCellStyle], FontFamily}]
	]
];
GetFontFamilyOfSelection[a___]:= Message[GetFontFamilyOfSelection::argx, {a}];
GetFontFamilyOfSelection::argx = "`1`";
GetFontFamilyOfSelection::info = "`1`";





(*******************************************************

 GetCellStyleOfSelection

	If no selction, return Notebook's DefaultNewCellStyle

	in: NotebookObject
	out: String cell style

*******************************************************)
(*GetCellStyleOfSelection[]:= GetCellStyleOfSelection[InputNotebook[], Developer`CellInformation[InputNotebook[]]];*)

GetCellStyleOfSelection[$Failed]:= CurrentValue[DefaultNewCellStyle];
GetCellStyleOfSelection[nbObj_NotebookObject, $Failed]:= CurrentValue[nbObj, DefaultNewCellStyle];

GetCellStyleOfSelection[nbObj_NotebookObject]:= GetCellStyleOfSelection[nbObj, GetCellInformation[nbObj]];

GetCellStyleOfSelection[nbObj_, cellInformation_Association]:=
Module[{cellStyle, cellInfo = cellInformation},

	cellStyle = cellInfo["Style"];

	If[Length[cellStyle] > 0,
		First[cellStyle],
	(* ELSE *)
		cellStyle
	]
];

GetCellStyleOfSelection[nbObj_, cellInformation_List]:=
Module[{cellStyle, cellInfo = cellInformation},

	(* Selection is NOT between Cells *)
	(* get name of style(s) *)
	cellInfo = First[cellInfo];
	cellStyle = {"Style"} /. cellInfo;

	cellStyle = If[Length[cellStyle] > 0, First[cellStyle], cellStyle ];

	If[Head[cellStyle] === List,
		(* display all or first style? *)
		(* StringJoin@@Riffle[cellStyle, ", "],*)
		First[cellStyle],
	(* ELSE *)
		cellStyle
	]

]
GetCellStyleOfSelection::argx = "`1`";
GetCellStyleOfSelection::info = "`1`";




getCellStyle[nbObj_, info_]:=
Module[{cellInfo = info, cellStyle},

	(* Selection is NOT between Cells *)
	(* get name of style(s) *)
	cellInfo = If[Length[cellInfo] > 0, First[cellInfo], cellInfo ];
	cellStyle = {"Style"} /. cellInfo;

	cellStyle = If[Length[cellStyle] > 0, First[cellStyle], cellStyle ];

	If[Head[cellStyle] === List,
		(* display all or first style? *)
		(* StringJoin@@Riffle[cellStyle, ", "],*)
		First[cellStyle],
	(* ELSE *)
		cellStyle
	]
];



(*******************************************************

 TestList

	in: xxx
	out: yyy

*******************************************************)

TestList[list_]:=
Module[{},

	RuleDelayed[#, #] & /@ list;
	{"Text" :> "Text"}
];
TestList[a___]:= Message[TestList::argx, {a}];
TestList::argx = "`1`";
TestList::info = "`1`";




(* single cell case; cellInfo is an Association*)
ChangeCellStyleOfSelection[nbObj_, cellInfo_Association, sty_, opts___?OptionQ]:=
Module[{cells, optionChanges = {opts}},

	cells = SelectedCells[nbObj];

	SelectCellBracket[nbObj, cellInfo];
	If[Length[optionChanges] > 1,
		FrontEndTokenExecute[nbObj, "Style", sty];
		SetOptions[nbObj, opts],
	(* ELSE *)
		FrontEndTokenExecute[nbObj, "Style", sty]
	]
];
ChangeCellStyleOfSelection[nbObj_, cellInfo:{__Association}, sty_, opts___?OptionQ]:=
Module[{cells, optionChanges = {opts}},

	cells = SelectedCells[nbObj];

	If[Length[optionChanges] > 1,
		FrontEndTokenExecute[nbObj, "Style", sty];
		SetOptions[nbObj, opts],
	(* ELSE *)
		FrontEndTokenExecute[nbObj, "Style", sty]
	]
(*
	Which[
		Head[cellInfo] === Association,
	 		SelectCellBracket[nbObj, cellInfo];
			FrontEndTokenExecute[nbObj, "Style", sty],
		Length[cellInfo] > 1,
			If[Length[optionChanges] > 1,
				(
					FrontEndTokenExecute[nbObj, "Style", sty];
					SetOptions[#, opts]
				) & /@ cells,
			(* ELSE *)
				FrontEndTokenExecute[nbObj, "Style", sty]
			],
		True, Null
	]
*)
];






CellBracketSelectedQ[nbObj_CellObject]:= CellBracketSelectedQ[nbObj, GetCellInformation[nbObj]];
CellBracketSelectedQ[nbObj_CellObject, cellInfo_List]:= CellBracketSelectedQ[cellInfo];
CellBracketSelectedQ[nbObj_CellObject, cellInfo_Association]:= CellBracketSelectedQ[cellInfo];
CellBracketSelectedQ[nbObj_CellObject, $Failed]:= False;

CellBracketSelectedQ[{nbObj_CellObject}]:= CellBracketSelectedQ[nbObj, GetCellInformation[nbObj]];
CellBracketSelectedQ[{nbObj_CellObject}, cellInfo_List]:= CellBracketSelectedQ[cellInfo];
CellBracketSelectedQ[{nbObj_CellObject}, cellInfo_Association]:= CellBracketSelectedQ[cellInfo];
CellBracketSelectedQ[{nbObj_CellObject}, $Failed]:= False;

CellBracketSelectedQ[{nbObj__CellObject}]:= True;
CellBracketSelectedQ[{nbObj__CellObject}, cellInfo_List]:= True;
CellBracketSelectedQ[{nbObj__CellObject}, cellInfo_Association]:= True;
CellBracketSelectedQ[{nbObj__CellObject}, $Failed]:= False;

CellBracketSelectedQ[nbObj_NotebookObject]:= CellBracketSelectedQ[nbObj, GetCellInformation[nbObj]];
CellBracketSelectedQ[nbObj_NotebookObject, cellInfo_List]:= CellBracketSelectedQ[cellInfo];
CellBracketSelectedQ[nbObj_NotebookObject, cellInfo_Association]:= CellBracketSelectedQ[cellInfo];
CellBracketSelectedQ[nbObj_NotebookObject, $Failed]:= False;

CellBracketSelectedQ[cellInfo_Association]:=
Module[{},
	(* Test if CellBracket is selected *)
	cellInfo["CursorPosition"] === "CellBracket"
];
CellBracketSelectedQ[___]:= False;

CellBracketSelectedQ[{___?OptionQ, (Rule|RuleDelayed)["CursorPosition", "CellBracket"], ___?OptionQ}]:= True;


CellContentsSelectedQ[]:= False;
CellContentsSelectedQ[{l__List}]:= False;

CellContentsSelectedQ[{___?OptionQ, (Rule|RuleDelayed)["CursorPosition", pos_], ___?OptionQ}]:=
Module[{},
	(* Test CusorPosition for selection *)
	!If[ListQ[pos], Equal@@pos, False]
];
CellContentsSelectedQ[cellInfo_Association]:=
Module[{pos},
	pos = cellInfo["CursorPosition"];
	(* Test CusorPosition for selection *)
	!If[ListQ[pos], Equal@@pos, False]
];
CellContentsSelectedQ[__]:= False;




(*******************************************************

 CursorInCellNoSelectionQ
	Test if Cusor is within cell and there is no selection
	in: CellInformation Association
	out: True | Flase

*******************************************************)

CursorInCellNoSelectionQ[]:= CursorInCellNoSelectionQ[GetCellInformation[InputNotebook[]]];

CursorInCellNoSelectionQ[{___?OptionQ, (Rule|RuleDelayed)["CursorPosition", cp_], ___?OptionQ}]:=
Module[{cursorPos = cp},

	(* False if CellBracket is selected *)
	If[cursorPos === "CellBracket",
		False,
	(* ELSE *)
		(* Test CusorPosition for selection *)
		Equal@@{cursorPos}
	]
];
CursorInCellNoSelectionQ[cellInfo_Association]:=
Module[{cursorPos = cellInfo["CursorPosition"]},

	(* False if CellBracket is selected *)
	If[cursorPos === "CellBracket",
		False,
	(* ELSE *)
		(* Test CusorPosition for selection *)
		If[ListQ[cursorPos], Equal@@cursorPos, False]
	]
];
CursorInCellNoSelectionQ[__]:= False;




(*******************************************************

 AllContentsSelectedQ
	Are all the contents of the cell selected?

	in: notebook
	out: True|False

*******************************************************)

AllContentsSelectedQ[nbObj_]:=
Module[{origSelection, allCharacters},

	(* *)
	origSelection = NotebookRead[nbObj];
	allCharacters = iAllContents @ NotebookRead[SelectedCells[nbObj]];

	If[MatchQ[origSelection, allCharacters], True, False]
];

iAllContents[{Cell[f_, ___]}] := iAllContents[f];
iAllContents[f:RowBox[{_, __}, ___]] := f;
iAllContents[BoxData[f_ /; Head[f] =!= List]] := f;
iAllContents[BoxData[f:{_, __}, ___]] := RowBox[f];
iAllContents[TextData[f:{_, __}, ___]] := f;
iAllContents[f_] := f;
iAllContents[f___] := Missed[f];




(**************************************

 InsertOrModifyStyledCell
	Depending on selection, either...
	a) insert new cell Cell[_, style]
	b) change existing cell to Cell[_, style]
	in: Style name: String
	out: None

***************************************)

InsertOrModifyStyledCell[style_String, opts___?OptionQ]:= InsertOrModifyStyledCell[InputNotebook[], style, opts];
InsertOrModifyStyledCell[nbObj_NotebookObject, style_String, opts___?OptionQ]:=
Module[{cellInfo, cursorPosition, contentDataForm, cells, cellStyle},
	(* Where is cursor? *)
	(* Check CellInformation *)
	cellInfo = GetCellInformation[nbObj];

	If[MatchQ[cellInfo, $Failed],
		(* Selection is between Cells *)
		InsertStyledCell[style, nbObj, opts];
		<|"action" -> "insert", "new" -> style|>,
	(* ELSE *)
		(* change style of selection (cells) *)
		Which[
			Head[cellInfo] === Association,
				(* single cell *)
				(* set some variables to use in query *)

				cursorPosition = cellInfo["CursorPosition"];
				contentDataForm = cellInfo["ContentDataForm"];
				cellStyle = cellInfo["Style"];

				Which[
					(* CellBracket selected, change cell style *)
					cursorPosition === "CellBracket",
						ChangeCellStyleOfSelection[nbObj, cellInfo, style, opts];
						<|"action" -> "modify", "old" -> cellStyle, "new" -> style|>,
					(* If cursor within cell and there is no selection, change cell style *)
					Length[Union@cursorPosition] > 1,
						(* Text selected within cell *)
						SelectCellBracket[nbObj, cellInfo];
						ChangeCellStyleOfSelection[nbObj, cellInfo, style, opts];
						<|"action" -> "modify", "old" -> cellStyle, "new" -> style|>,
					(* No selection, cursor is within cell *)
					CursorInCellNoSelectionQ[cellInfo],
						SelectCellBracket[nbObj, cellInfo];
						ChangeCellStyleOfSelection[nbObj, cellInfo, style, opts];
						<|"action" -> "modify", "old" -> cellStyle, "new" -> style|>,
					True,
						(* TODO: Clean up fail case *)
						SelectACellMessageDialog[];
						<|"action" -> "none"|>
(*						Message[InsertOrModifyStyledCell::info, "Cell not selected"]*)
				],
			(ListQ[cellInfo] && (Length[cellInfo] > 1)),
				(* multiple cells *)
				ChangeCellStyleOfSelection[nbObj, cellInfo, style, opts];
				<|"action" -> "modify", "old" -> cellStyle, "new" -> style|>,

			True, <|"action" -> "none"|>
		]
	]

];
InsertOrModifyStyledCell[a__] := Message[InsertOrModifyStyledCell::argx, {a}];
InsertOrModifyStyledCell::argx = "Argument should be a String: `1`";
InsertOrModifyStyledCell::info = "`1`";




(**************************************

 InsertStyledCell
	Insert new styled cell,
	Used within InsertOrModifyStyledCell[]

	in: Style name: String
	out: None

***************************************)

InsertStyledCell[style_String, nbObj_NotebookObject, opts___?OptionQ]:=
	InsertStyledCell[style, nbObj, "XXXX", opts];

InsertStyledCell[style_String, nbObj_NotebookObject, content_:"XXXX", opts___?OptionQ]:=
Module[{box, screenEnv},

	screenEnv = CurrentValue[nbObj, ScreenStyleEnvironment];

	(* BoxData, TextData, or ... *)
	Switch[CurrentValue[nbObj, {StyleDefinitions, style, FormatType}],
		System`TextForm, box = TextData,
		True, box = BoxData
	];

	(* If in slide show env and no selection, add cell after slide contents *)
	If[screenEnv === "Slideshow Presentation",
		SelectionMove[nbObj, All, Notebook];
		SelectionMove[nbObj, After, Cell];
	];

	MoveSelectionToBetweenCells[nbObj];
	(* Insert Cell *)
	FrontEndExecute[{
		FrontEnd`NotebookWrite[nbObj,
			Cell[box[content], style, Sequence@@{opts}],
		After]
	}];
	NotebookFind[nbObj, style, Previous, CellStyle];
	SelectionMove[nbObj, All, CellContents]
]




(************************************

 SelectCellBracket

************************************)

SelectCellBracket[nbObj_NotebookObject]:= SelectCellBracket[nbObj, GetCellInformation[nbObj] ];

(* case where GetCellInformation returns $Failed *)
SelectCellBracket[nbObj_NotebookObject, $Failed]:= SelectACellMessageDialog[];

SelectCellBracket[nbObj_NotebookObject, {cellInfo_List}]:= Message[SelectCellBracket::info, "List of Lists"];
SelectCellBracket[nbObj_NotebookObject, {cellInfo_Association}]:= Message[SelectCellBracket::info, "List of Associations"];
(*Module[{},
	SelectCellBracket[nbObj, cellInfo]
];*)
(* case where GetCellInformation is a List *)
SelectCellBracket[nbObj_NotebookObject, cellInfo:{___?OptionQ, (Rule|RuleDelayed)["CursorPosition", pos_], ___?OptionQ}]:=
If[!MatchQ[pos, "CellBracket"],
	ExpandSelectionAndSelectCellBracket[nbObj, True],
(* ELSE *)
	ExpandSelectionAndSelectCellBracket[nbObj, False]
]
(* case where GetCellInformation is an Association *)
SelectCellBracket[nbObj_NotebookObject, cellInfo_Association, ___?OptionQ]:=
If[!MatchQ[cellInfo["CursorPosition"], "CellBracket"],
	ExpandSelectionAndSelectCellBracket[nbObj, True],
(* ELSE *)
	ExpandSelectionAndSelectCellBracket[nbObj, False]
]

(*
SelectCellBracket[nbObj_NotebookObject, cellInfo:{___?OptionQ, (Rule|RuleDelayed)["CursorPosition", pos_], ___?OptionQ}]:=
Module[{info, CursorPosition},

	(* If CellBracket is not already selected *)
	If[!MatchQ[pos, "CellBracket"],

		(* If selection is within a Cell, Select that Cell *)
		(* can't use SelectionMove by itself, since cursor might be within an inline cell
		FrontEndExecute[FrontEnd`SelectionMove[nbObj, All, Cell, AutoScroll -> False]];
		*)

		While[
			(
				info = Developer`CellInformation[nbObj];
				(info =!= $Failed) &&
					Not[MemberQ["CursorPosition" /. info, "CellBracket"]]
			),
			FrontEndExecute[FrontEndToken[nbObj, "ExpandSelection"]]
		]
	]
];
*)
SelectCellBracket::argx = "`1`";
SelectCellBracket::info = "`1`";
SelectCellBracket[a__] := Message[SelectCellBracket::argx, {a}];


ExpandSelectionAndSelectCellBracket[nbObj_NotebookObject, True]:=
Module[{info},
	(* might be better / modern way of doing this *)

	(* If selection is within a Cell, Select that Cell *)
	(* can't use SelectionMove by itself, since cursor might be within an inline cell
	FrontEndExecute[FrontEnd`SelectionMove[nbObj, All, Cell, AutoScroll -> False]];
	*)

	While[
		(
			info = Developer`CellInformation[nbObj];
			(info =!= $Failed) &&
				Not[MemberQ["CursorPosition" /. info, "CellBracket"]]
		),
		FrontEndExecute[FrontEndToken[nbObj, "ExpandSelection"]]
	]

]
ExpandSelectionAndSelectCellBracket[nbObj_NotebookObject, ___]:= Null



(*******************************************************

 SelectParentCell

	in: notebook
	out: Selected cell

*******************************************************)

SelectParentCell[nbObj_NotebookObject]:=
Module[{selection},

	selection = SelectedCells[nbObj];
	If[selection =!= {},
		SelectionMove[
			ParentCell@First[selection], All, Cell],
	(* ELSE *)
		Message[SelectParentCell:info, "Unable to determine parent cell."]
	]
];
SelectParentCell::argx = "`1`";
SelectParentCell::info = "`1`";




(****************************

 GetSelection

****************************)

GetSelection[nbObj_NotebookObject] :=
Module[{cellInfo, cellStyle, cursorPosition},
	(* Check CellInformation *)
	cellInfo = GetCellInformation[nbObj];

	(* If selection is within a Cell, Select that Cell *)
(*	SelectCellBracket[nbObj, cellInfo]; *)
	If[cellInfo == $Failed,
		(* Selection is between Cells *)
		(* Select first Cell in Slide *)
		FrontEndExecute[FrontEndToken[nbObj, "SelectAll"]];
		SelectionMove[nbObj, Before, Cell];
		SelectionMove[nbObj, Next, Cell];
		SelectionMove[nbObj, Next, Cell];
	,
		(* "Unknown" *)
		(* TODO:  *)
		cellStyle = ("Style" /. cellInfo);
		cursorPosition = ("CursorPosition" /. cellInfo);
		If[
			MemberQ[cellStyle, "SlideShowNavigationBar"] && MemberQ[cursorPosition, "CellBracket"];
 			SelectionMove[nbObj, Next, Cell]
		,
			$Failed
		]
	]
];








(****************************************************************************

	DIALOGS

*****************************************************************************)


SelectACellMessageDialog[]:=
	MessageDialog["Please select a cell.", WindowSize -> {530, All}];

NoSideNotesMessageDialog[]:=
MessageDialog[Dynamic@FEPrivate`FrontEndResource["PresenterToolStrings", "NoSideNotesPresentDialog"] ];






















(****************************************************************************

	UTILITIES

*****************************************************************************)



AddCellsToEndOfNotebook[nbExpr_Notebook, cells_]:=
Module[{res = nbExpr, opts},
	(* Get options *)
	opts = Rest[List@@res];

	Notebook[Flatten[{ First@res, cells}, 1], Sequence@@opts]
];
AddCellsToBeginningOfNotebook[nbExpr_Notebook, cells_]:=
Module[{res = nbExpr, opts},
	(* Get options *)
	opts = Rest[List@@res];

	Notebook[Flatten[{cells, First@res}, 1], Sequence@@opts]
];




(*******************************************************

 MergeSets

	in: xxx
	out: yyy

*******************************************************)

MergeSets[setOne:{__?OptionQ}, setTwo:{__?OptionQ}]:=
	ResetOptions[setOne, setTwo];

MergeSets[setType_:"FontSet", theme_String, newSet:{__?OptionQ}]:=
Module[{themeSet},

	themeSet = GetThemeInformation[theme, setType];
	If[MatchQ[themeSet, $Failed] || Head[themeSet] === Missing,
		$Failed,
	(* ELSE *)
		ResetOptions[themeSet, newSet]
	]
];

MergeSets[setType_:"FontSet", theme_String, new_String]:=
Module[{themeSet, newSet},

	themeSet = GetThemeInformation[theme, setType];
	newSet = GetThemeInformation[new, setType];
	If[MatchQ[themeSet, $Failed] || Head[themeSet] === Missing, MatchQ[newSet, $Failed] || Head[newSet] === Missing,
		$Failed,
	(* ELSE *)
		ResetOptions[themeSet, newSet]
	]
];

MergeSets[___]:= $Failed;
MergeSets::argx = "`1`";
MergeSets::info = "`1`";




(*******************************************************

 GetSlides

	in: xxx
	out: yyy

*******************************************************)
Options[GetSlides] = {"DeleteNavigationBarWithFirstSlideStyle" -> True}

GetSlides[nbObj_NotebookObject, opts___]:= GetSlides[NotebookGet[nbObj], opts];

GetSlides[nbExpr_Notebook, opts___?OptionQ]:=
Module[{dnb = ("DeleteNavigationBarWithFirstSlideStyle"/.{opts}/.Options[GetSlides]), slideExprs, nbOpts},

	nbOpts = Rest[List@@nbExpr];

	slideExprs = First[nbExpr];

	slideExprs = Flatten[{ Notebook[{#}, nbOpts]& /@ slideExprs}];

	slideExprs = If[dnb === True, slideExprs /. Cell[_, "SlideShowNavigationBar", ___] :> Nothing, slideExprs /. Cell[_, "SlideShowNavigationBar", a___/;Not@MemberQ[{a}, "FirstSlide"]] :> Nothing];

	slideExprs

];
GetSlides[a___]:= Message[GetSlides::argx, {a}];
GetSlides::argx = "`1`";
GetSlides::info = "`1`";


(*
GetSlides[nbExpr_Notebook, opts___?OptionQ]:=
Module[{dnb = ("DeleteNavigationBarWithFirstSlideStyle"/.{opts}/.Options[GetSlides]), slideExprs, nbOpts, p1, p2, i},

	nbOpts = Rest[List@@nbExpr];

	slideExprs = First[nbExpr];

	If[FreeQ[slideExprs, Cell[_, "SlideShowNavigationBar", ___]],

		slideExprs = Cell[CellGroupData[Prepend[slideExprs, Cell["", "SlideShowNavigationBar", "FirstSlide", CellTags -> "SlideShowHeader"]], Open]],

		p1 = Position[slideExprs, Cell[CellGroupData[{Cell[_, "SlideShowNavigationBar", ___], ___}, _]]];
		p2 = Position[slideExprs, Cell[_, "SlideShowNavigationBar", ___]];
		i = (If[# === {}, {}, #[[1]]]&@Cases[Join[p1, p2], {a_Integer} :> a, 1, 1]);
		If[i =!= {} && i > 1,
			slideExprs = Prepend[Take[slideExprs, {2, -1}], Cell[CellGroupData[{Cell["", "SlideShowNavigationBar", CellTags -> "SlideShowHeader"],
												Sequence @@ Take[slideExprs, {1, i - 1}]}, Open]]]]];

	slideExprs = Flatten[{ Notebook[{#}, nbOpts]& /@ slideExprs}];

	slideExprs = If[dnb === True, slideExprs /. Cell[_, "SlideShowNavigationBar", ___] :> Nothing, slideExprs /. Cell[_, "SlideShowNavigationBar", a___/;Not@MemberQ[{a}, "FirstSlide"]] :> Nothing];

	slideExprs

];
*)




(** ResetOptions **)
(* actually reset options. it's faster to define this in terms of
   delete cases rather than have another call to DeleteOptions *)
ResetOptions[h_[stuff___,opts___?OptionQ], newopts__?OptionQ]:=
  h[stuff, Sequence@@Flatten@{newopts}, Sequence @@ DeleteCases[{opts}, _[Alternatives @@ (Flatten[{newopts}][[All, 1]]), _]]];

ResetOptions[h_[stuff___],newopts__?OptionQ]:=h[stuff,newopts];




UpdatePresentationTools[]:=
Module[{},

ChoiceDialog["Make a choice..."]
];




(**************************************

 VersionGreaterQ
	Compare two string version numbers,
	return True if the second is greater than and first,
	False if second if equal or less than the first.

	in: two string version numbers "0.9.1", "10.2", etc.
	out: True | False

***************************************)

VersionGreaterQ[str1_String, str2_String] :=
Module[{v1, v2, max},

	v1 = ToExpression@StringSplit[str1, "."];
	v2 = ToExpression@StringSplit[str2, "."];
	max = Max[Length[v1], Length[v2]];

	Not@OrderedQ[{
		PadRight[v1, max],
		PadRight[v2, max]
		}]
]
VersionGreaterQ[] := False;
VersionGreaterQ[__] := False;








(****************************************************************************

	LOGGER

*****************************************************************************)


(* logger *)
LogLevel = 0;

Logger[expr_, "DEBUG"]:= Logger[expr, 10];
Logger[expr_, "INFO"]:= Logger[expr, 20];

Logger[expr_]:= Logger[expr, "INFO"];

Logger[expr_, level_Integer]:=
Module[{logLevel},
	If[PresenterTools`Private`$DebugQ,
		If[LogLevel >= level,
(*			SetSelectedNotebook[MessagesNotebook[]]; *)
			Echo[expr]
(*			Print@ToString[expr]  *)
		]
	]
];
Logger[expr_, a_] := Message[Logger::argx, Head@a];
Logger::argx = "Argument should be a Integer: `1`";

logLine = "\n********************\n";

(* create timestamp *)
timestamp:=ToString[(StringForm["`1`/`2`/`3` `4`:"<>
  (#[[1]]<>":"<>#[[2]]&)[(If[#<10,"0"<>ToString@#,ToString[#]]&)/@{#5,Round[#6]}], ##]& @@ Date[])]






SlideshowNavigationBar[nbObj_]:=
Module[{},

Cell[
	BoxData[
		ToBoxes[
			Grid[
{{
	(* Slide n of m *)
Style["", FontSize -> 9],
(*
	PresenterTools`SlideNavigationPopupMenu[ButtonNotebook[]],
*)
	(* general navigation *)
    Grid[
     {{Pane[Grid[{{
     	(* first slide *)
		Button[
     		Dynamic[RawBoxes[FEPrivate`FrontEndResource["PresenterToolExpressions", "FirstSlideButton"]]],
			FrontEndExecute[FrontEnd`FrontEndToken[nbObj, "ScrollPageFirst"]],
			Appearance -> {"Default" -> None, "Pressed" -> None},
			Tooltip -> Dynamic[FEPrivate`FrontEndResource["SlideshowToolbarText", "TooltipFirstSlide"]],
			TooltipDelay -> 0.25
		],
		Button[
           Dynamic[RawBoxes[FEPrivate`FrontEndResource["PresenterToolExpressions", "PreviousSlideButton"]]],
			FrontEndExecute[FrontEnd`FrontEndToken[nbObj, "ScrollPagePrevious"]],
			Appearance -> {"Default" -> None, "Pressed" -> None},
			Tooltip -> Dynamic[FEPrivate`FrontEndResource["SlideshowToolbarText", "TooltipPreviousSlide"]],
			TooltipDelay -> 0.25
		],
		Button[
           Dynamic[RawBoxes[FEPrivate`FrontEndResource["PresenterToolExpressions", "NextSlideButton"]]],
			FrontEndExecute[FrontEnd`FrontEndToken[nbObj, "ScrollPageNext"]],
			Appearance -> {"Default" -> None, "Pressed" -> None},
			Tooltip -> Dynamic[FEPrivate`FrontEndResource["SlideshowToolbarText", "TooltipNextSlide"]],
			TooltipDelay -> 0.25
		],
		Button[
            Dynamic[RawBoxes[FEPrivate`FrontEndResource["PresenterToolExpressions", "LastSlideButton"]]],
			FrontEndExecute[FrontEnd`FrontEndToken[nbObj, "ScrollPageLast"]],
			Appearance -> {"Default" -> None, "Pressed" -> None},
			Tooltip -> Dynamic[FEPrivate`FrontEndResource["SlideshowToolbarText", "TooltipLastSlide"]],
			TooltipDelay -> 0.25
		]
	}}, Alignment -> Left,
         Background -> RGBColor[0.2, 0.2, 0.2], ItemSize -> Full,
         Spacings -> {{0, {0.5}, 0.}, {0., {0.2}, 0.}}],
        ContentPadding -> False, FrameMargins -> {{0, 0.}, {0, 0}},
        ImageMargins -> {{0, 0}, {0, 0}}]}}, Background ->
      RGBColor[0.2, 0.2, 0.2], Spacings -> {0, 0}, ItemSize -> Automatic],

	(* hamburglar *)
Style["", FontSize -> 9]
	(* end of row *)
	}},
  Frame -> {All, False}, FrameStyle -> Directive[RGBColor[0.2, 0.2, 0.2]],
  Background -> RGBColor[0.2, 0.2, 0.2],
  Alignment -> {{Left, Center, Right}},
  ItemSize -> {{Scaled[0.1035], Scaled[0.8], Scaled[0.1]}},
  Spacings -> {{0.35, {0.}, 0.}, {0.1, {0.2}, 0.2}}
			]
        ]
	],
	FontSize -> 9,
 Background -> RGBColor[0.2, 0.2, 0.2],
 Magnification -> 1,
 CellFrame->False,
 CellMargins->{{-1, -1}, {0, -1}},
 CellFrameMargins->{{5, 5}, {0, 0}},
 CellFrameColor->RGBColor[0.2, 0.2, 0.2]
]
];














(**************************************

 MakeNotebookThumbnailImage
	Create thumbnail image of theme
	in: StyleDefinitions of theme
	out: Raster / GraphicsBox

***************************************)

MakeNotebookThumbnailImage[styleDefinitions_]:=
Module[{},
(*	Framed[Pane[MakeNotebookImageFromStyleDefinitions[styleDefinitions], ImageSizeAction -> "ShrinkToFit", ImageSize -> {100, 70}, FrameMargins -> 0, ContentPadding -> False, ImageMargins -> 0], FrameStyle -> White, Background -> White]
*)
ToExpression@
GraphicsBox[
  TagBox[RasterBox[CompressedData["
1:eJztPAl4FtW18dm+8traZ4uvvqet1b5Xt6poW60VFRUQAmIVRGQVUIwaZFUQ
2cKqsimEPYFgEtYQAiHsgYSEJCSQEEKAJGTfN/In/zb73HfOufNPJn+istgY
l/vBZP6ZO3fuOffs59y5a/T4/m/9m4+Pz+ROcOg/auqzkyaNmjbgZvgxcNzk
d/zGjXnTd9z7Y/zGTHp89I1w8U34PwT+4zn7sf3Yfmw/th/b9TRFleCoK/jP
xXTGHMYNJ1OYoGk6/FGZxDRV1+iE6Tr00vCfrghwCc7gFLrBNdWtqtiRwXNw
hcmSLjNNkJgi4QVZ0R3YT2e6JuJ7Veik4Ft1xkeCG7Ku4S1ZhvFVuIJDtV+D
GWmaQ2KiCD8kmpXOBKYoiCUESVQIXRriBSHl0wYUiIwu6hwogXEkKYqMwwIQ
Gp7Ag7Ki6gqd4lGiPxqTYBxZUwBjNAmZqZqmifRSlVAJJ05VZ+2KDEUngGSV
2fGXqjAHgqvDsuuqDjPFaSKsOkKsMlHAvrhkAK5TQxzifNWmJqY6EQ4ZRyMs
Ap3ABSfHHxGPDI/qjMCHn0ADCtGGpGkuAV+EXYn2cBjm1omK2g8XAJGo2IEH
RBUYBQgE/6sqgSMT5dIkAWRAh6aoSDgSp2hgHkKExHmCcKIzt4JLq8uKm3MT
rDzyiC6yRuouu3Eo4EGFkwRgmsk6DqjrKqIPeVPl7wDy40O3V0OC1DSgbRQX
ugiEC1Plyw2XXQiMTISDogO6u/gzSPiKjgwFkwWaEmHRkcgRQQCF6EK+E5xw
CqShaTKur0OQATAkNgHpxY3PIvrdSBIopUR+G9gHcAR3JQ0ICJDmbjdsAA24
mYSTUtxJXe7c1+WBQ7/+NVNE4lxRV5ywWIrqZqKGSwRErYOMQAHoplWTVZSj
JGQ0V1lu5bFjgA1kdkXJfnU0DsolLiDRCeKUFS1aULQ9kbjPwVSQl4RkSUbk
Mlkk5nACGpBmiXAVnQub9mkoy3EtFNbQGPEfP2OK3d5gM+SiJOI6M6mmqADI
n5aei01kLCAPiSBVNFaVfgFGaTyZevrzYADTZhMAyH0+NzFcYcCjnS80U9Xs
Xk/lbIlSFbdE0tGF4HICA2pRJLgOaNbtML6iaB4uaT/BgejnYpBpx3x8Cs4k
Vx056c7L2N2p03afzgWpCYWz3gr1+UX6rqPFBzfG/qZzSsBCptTuvOe3ETfc
kLV+K8y20p6x2+en4X/8oyPrVG7ohsRf3bjJ598K885efO4huBt/1x0Hbr+j
qtEGWIK3lM+bsP9/bt7R6adilcDstsRbOyfe0QXkdlr/V3Z3/lVCr8ENhyP3
3/LnxEFvOplAS0KCq/2wQZoO/2lxPj4VU+dmzVlUH7M17YkBsOqHbr/NlZJS
+uF8B3NG+/w058Du/Td0KouPPHLDvwP23DRhkDaxf7gZpm1PO5Y/YnThXH93
VgP8TPV9ptjv9eShozIG9zr2wD+YBpqVFQ99IT86ntkyzgx642z3v5+bN+Pw
A3fkfxaW0/kXtU0oUmBFcmJ3h/7ER0y7KCHxgPAW2w0bXPW7dElyOA7e2Imp
LjAqLkdsyP7naJAYmff91ZYYn/7yUJ05tvj80l7ZaCtLL92zL/mhJ0TUKqhi
gOYP+/jAtC8fiKzwm3Th7ZdL96a4mSvnoX/kjfM/1W2wo6a4OD5JQrNFK+rZ
My+qoL4mJnf4nMw+vllvz7DV1TbZa9NuvVlBCeyO8PlNY3bx5dRDapUoC1yo
th8yYJIu5gJuBZQc+OVPwjv/ductnS9GhaXNXADrvrP3S5rWtMPnvxMHT9Bi
D6/72c8PP/8PZ2b8vu59SaSqqox0dXH29JU33pK/avOFtUuFy3WBPp1SN4Wf
HzQWue+WW9f/7qbqwG1k6Lqzxv8zf/+pxoqk+MFw173rlz5ht/5czCpK8vXl
kqnxQPhGn//c1uVuzWkjdS8JzNZ+6EBpjsYUijLdwW0MjRtdbrQtyMIGVYm6
xLDM8TKKV1HVDOtIa0Lu1sBEA1NeIl0pyqhJRbJHNBCOEjdWdeRLEsiaDQU4
w+cVtHsFQi5adHAdzkjNcNHRfg3NRMEJOh7NRrTBmejWaZ3wnGxKbl7pCIUT
7CqkX5XWDXwQXeNSXyPjG7QkaFdV0MiVQdMKzBVFcaCCQgsEDVTEK7o/tACq
5tAbEUuq1khzIUXuYNxcNzyB9rM3RIRbZny9YM4yyCyZVJ6ChjaQCclYBWwB
DThb4AaSxH0pXWkE25O5ZZ0sJ10xYASlqQsgJciodmtkrYOVgo9oZLMqQCcw
Pshhrq+BPBwcVaTEZaJJRSWbuF2JAxkf5TaaVWQ/uBAgXH34S56lSDYG2Euq
wPtL5F7I0A1voaEKBqPGXU1AoBOnD8gAyhE5WlHSMgMnMic2YhINXFxyfUGd
EtaRj9BxE8kkR2OMHm+3xg0clAASO3k6XbTb8YIm8fmLiAc3uRKcX4CQwVAD
MeCCu2ChEtLIiWNuJGhdbALFrIMpqxkuLWGcBAw6cGjzo5co43gkGRSdLArg
L4wPMGQrjRYIsKsj+oAF2xEfQKfILDsjt/r26dnQ0FBdXQsXU1NTUZTJyERn
z56FY21tLa2yfDzh6JQPJsL8U1NP7dwTWVBeELRqTWJcvOhEtZGTl/vuuPdg
oRucdgDD7nQ4XM6GRhv5ebrO2tdBv/oGWkBWXLFHD0yZOmnv3r0vvzxgzJgx
s2bN8vPzGz9+/MCBA+EiOJi9e/f+KCDgeHLC9BlTP14412/MW1OnTjuRmtS9
T/fjR4/Nnj7jvnvuPXfuHAzY54W+8z5e6NvvhZmzZ42fOGHZ55/d9rvbE5NO
cFRoHRgfPJAFtHok9sCwYUNWrlyZmJi0Zs2aBQsWvPnmm8uWLQsJCeHdRo4c
uWjZZyNHvx4UvOZtvzdiovfOnj1neeCK9SHrZs+cBd169OhRV1cH1BUdHR0W
FjZy9KgVKwOTT6bAC7o88vD5ixdAQXwHaAM9J+RQSRL4FYCdTABm/uQRGJX7
NShNKfanIHQyyUjd0xMVASMEw6DEaHBRAxlLI8AVI5jTIRtNT6X4ncSlnqYZ
sg9OOE74/PGnqpOnrVBn6kNCEU0sRdbNZw2W0M3R+AgdGQ9eTaNmrKwFJwZV
eK5Dk2RXM9zYVC4eeeM9AXGoJujcigqTSDpma71qJkkwZpA688CoGspOI3Ki
QBcxGvOwA+9mPsWIjDgGWmO1AzYPNvD8q6eqUZ6AS12OAc/qk2GqNQe3TU1q
sokVFR2aNjDsiYFrdMnQQrqWiD09Z+gLlKuq4tGkaGcLkpuQBZa9bNCV5umq
GxzKOgiWNLJ9ER6PIXAdk7JSlwGmxuUF5QLohNMYqSSDH03i+fabjjFR7j0r
5NtfgyVswuIle7HJpFq4UKVsHbOwUkdBgqeRxuRZHhCCMk8NXMM4XgTPl5un
HxEBFGEjQYxkohqChkgSRDN/RG3HNNKXNUzDUjBD5v4sI9Zuu331SGYH80Qm
9sBouY7AYkpP04kvNfDRKBt5RRj4svl8s41RQhklp8oqq+rzKmryyitLSivL
r7KVlJRUVVXBSVFRUVlZWWlpaRm1kory4vKy0uKSirLyspLSqorKorJSuFha
UlBVWVpVCQ9ifz7C1b70G2+EdwrV4RqKuliPgY5rIjFJkhjztlvQCuMZFyQP
jQmS4knuM0WkwgDNJI/2TcV/SRMw8ugEQq48u99vyMW1y68h8gZcUJhfYJgc
uqFBNC6SmVKemYWxHU05Hhkl6RhDY4pccP5cZUkhx4ZKVl2HwAaFsWBdc6PW
HXu9X4WtvCT+i4KDuxuT9sXNDmisrT2/eyPIloQ5/g01ZTJTzn0+m4nCmV1B
Qq3iyI9NWb8q//NpKZ8FlOaUOMoLYpdMgfFSFkyrzs12Mrls74bkgPfzT0We
nTszM/DzrGmza2pPlUUEg6LJPr1t1+O3CPWNQB8nQxeD6Dq1fLpQWXQyeLks
2AmrIuZ8Ma7YfikEKi8QwOmqT9x6cmgPLT897tVnU5fNSHjp8ZrY7XG+XTIn
Dj819qXkGdNcyFJVmx/9w4UvNsT95a70KUMSh/UD1blv4IPO7KPpy2acm/Bu
TXbcnocfvbh4ZnL/7kx35yx+r+LonksRM/f1udddVpC96vPDLz8S3e1JgNFx
/kj6R2NPTh18rM9f00b3POH/bF7Q6tQZI2K7P5y8ZA7FEJ0e57D9sksYfMPg
vVJ+ZPvWp7oU7gwrXza7NH5r5sqPTkybfGbqqOMz3sv+dJLWUIUiweWK6fWX
jBUfxQ8flL1vU9LYIVlrgs4Grz3+zsDCiG3n58w5OGVE6iy/utMppwJG6qyx
MfNAxKjXsoMCcj+Zenr8tNy1q48G+CePfAWGKoiPLoj+Inf7tPPTJx98f3LK
0uF2VzErTgp/9KGCmF3c+cf5YcmMq92wQfFymWGuXZS16iYdI8ZUxqHpqg11
oER1Soz7ZjITBBFlTZMR4a5shIc1RxOXjnbV2UTWNqhPmQtIWefJa+ZET09U
FUUWVXgXRj4VlFESczjdmF4E9DGqcgEzVSYBTMkavR2z0hTkFZibDFCKBsOk
MSMAADPFSV1o/li7oWFpgaLysiWXzLM/GC3WMW7OdBcmj7BrLRencMthmF0i
FXwICqJR5oF0QeJ2nijy2DsFhp0YbefPAANTLq8dsSETFkD628h9M4I4DEuW
DIdCI7MMzRJNNax3XEmJObmlBkhwMklXeUkS1baJhjfIJF4ERdEkRjF3AA8D
QzwShEV0biopo9C9iKagJnOTleiGautUod2wIRkQaCKuDeUdJaAIh04WtSYz
XsMhU9mGiz+gIaCiRsYDETYVjlGamjIOnFc0jgv0ad1G8orq3kg08pI5kRe8
wVkj4EDG2D2vHTIcGk4U7Vj4JXPbSyOGlbGwxUX5HSMMykDZaVgEhciyw9rp
5LELulskaUOFPeSMyOjpSJQ5xAyaCrKFORiFSSlpwpDviEvQSmdY24PGOXCW
kwqpNCdPczJefwichUksXdK+9fCQvbb64vZgEnEkLKkgEGUIzvYyd8Tqi7NF
jjCVHD1e1QaM39CA3E75ZRX9EUlyNoKCcJSVM/IKXU57TUUpYcQN/5vwIQcP
mfG3W/0da4jA40oYETbTXzYjA57OLaK7XmOageuvdbvMVpYQkrr2o/Nxqy+F
Lju9cWbO8bW5+0ISlo4DHjkfHVK6f59mzz754YBLkUvSd60Ts5PTNiwsPbEj
O2hW8rYlOQk7sncsr0nfd3jmSJCiZ3YHFkR+mrBx/tnIwJywOVlRgYCEPUPv
b8g6kRs8u/TQhtxNc89+NtZWLzBBtk7bKzhgQsEsIX1V9XIE+InCz3l83gq1
V3CPRym98Na6VSduyVww5sTad7LDl6atmhI70ff0zNdzt84HEkhdMiF5Yv/s
xSPOBgw4Hzj92IK3MwKGJc4alDhrYNnx4PPhM04ufv3YxyPSZg9MDXzPVl3q
TtqaNGdY3tYlOaGfHJ06IH7ROPBQEtcFnFz29oEJTyYHTW8qy4l8wqfyYqZk
AQrmqbcVPLTmNXhPlZoVh5gQaeVNe4Whroo2gMLBzFapTE+Qm7Cm0VVD6Wdm
kxpQjjaALLFhrRpRPFbhwBsabMwlyaLidDSRaBUEnosGNeuWhIYmEDu2pkaq
X0KpKoIFjh4SCFYHVpAywVz61rCbEFkTHF53PQEW1XTSrQTGhzXH5Le+ljAY
r9pQXapRSc4of4+KgUluZHSZ7A0JdQ9PPatmCa0nuSiA/MPHQV1RetqIKfGa
ECfZewIl8MHaICWvi6ZZYYXCuo5ebMI8wW3rLWIozUQdP3plytrE7Vc1iuK6
9SYsG5CoTk7BA+gRVbXDAI0Is5tsRje9VZBkp07qUtLJ2MdaAx0LdABXgmKE
kakiiGfrdSqKkEWNyjcEvKs2B9CsolK35CY4XLwbh8uMJFgEi0ZGmyaKbp41
owg/Jcc8Y15VzsvJqGwA1ksVS+02ihTS+JpBOkQQAiIIAJYNdQwyEwwreN9l
RxMVA7nItkaSqbxcUdNYTZaujNsZqBRUBS2kurA6lKwt3WFjHoI32dw8MSnE
zF2SxmlO55kNy6Uoe2hJIGrEPi3weaWEwQyzBwxwv1GjlqxZt+WLjZ8tnxGy
bevKwLXpuekrA+bu3r/zWPT+cQsXRezYFjBj5vbInVt2bJ80dkpVaTU81q9X
3+OJ8Z8uXbby0+UbgkMjoqJTkk6+Pnzk9sjdIZs2b92weVLA0i8WTv8iYucX
4UGLQ3akxsasWrDg4W49Dh06FBoaum7dupkzZyYlJWVnZ0+bNm3v3r3h4eGX
Ll0KCQmJiIiYNWtWcHBwXl7eli1boqL2ZGRkzJs3D65UVVVt3rwZ+m/ZEr53
75558+Y4HE3vvecP5598snDVqsDIyAgY/8CBA1488rW0AR14+c14/7GrV29c
sWrZlFnj+3brN3HCOEDRc493Gec/+rV+vT6a8+m5jIpP5vs/+Wyv9SFrn+r2
RODyYDCu/f3f2L4jfN26sCFDhvj6+qakpMBg0dHRa9eu7dat22uvvbZ48eKl
S5e+8847Q4cODQsL600tLS0t9siBrk88tity+5QPJsIxek9k4Iplw4YOGjFi
5MCBA1etWlVfXzt8+NAX+vUG0evbp+eChQGTJk147rnnli5ZPmTwiOeffz4g
YNbUDycNHz586tSpgDE4du3adfr06YAleN2DDz64cePGq5AYJkLwgERVX+dA
9SDbiJoFskVF3IdAVihRn8hz92huUQjY7rhMotdArFVXmmthnZJFylHtIfeH
MfWC/F5cVGCuHhcIVCSAdQWabmTSeR6EjwAXvV5qfZeXrLiSTKimcx9NUXjN
H85QlmSUc4Ii89I9CUueZJKGxuYjfIVuZOZpD0FzcYI1amqNUbOWliTPZXui
iJa4ogeNPNaKAkMTrdjAonRPJJZXYnPV46VPrdqKGypfiwrGCUMCeDCswd0q
kHi07LQ5gaK8fLMJEImxEUxTOYRmApL7nuZ8rOLLhM68aK0NsMTzm1fTHIdr
CvOfaWvpnhwxVxrmgFzXmBaaqaSunDy49w0WgKTLa5YFrt8QSrCjPyJxv0TF
hAgtHFVFoj/P8zG0vccosWVWoEwMWNfIOhOYtmlpe7iMmY9al5UehM5yyyve
Gqe14erVrtAiJR8Kg/9AGXOnzTAcbSAW8K8Y7cpxiejCa3pmWkbQ2pDwrbub
XE5JdXDfnpjLED3W18H02kzTf5mz4KUHrZNvlaowxve6a+KQP+tl3F5hkzVP
HlDXdoaFnyuvi8k8Fx+5ddCrL+9Jz8m1204eiYalWLl61faI1afiLiekVlAQ
AEODyNBc17cFNbNUd3jRjJc5wUWNtaeXQ0c9DYlhmuV03sJ2ZS1lqfXEylBf
0VRDdGC5796I7Um5xUt3R8eFhSz9bNGi6NikmprdoSGNDnYmNzMsInBPeO2S
tQluz7ZFMtsEnduozJtZWi8fa7GUqikNzCOPBloBbO2tWyz5FmO26ZV44epK
mMUc/4MPPmh+L69q8Eh7/Km18w7Nb6dxEgoKCjp48KBXsZbXzyt3jb+jrTVd
8eYlx65QJn8PmpW5iPs0M8L2w0GCVxMEAaydyZMnkjvcHFj7Cpn2vWwcxsLC
wptu+oVZbMw8Vk1BQUFxcbHD4fghoII3ANzlckiSYHc0gF9QW1cJNkBjY2NN
TU1MTExqairvdm0mzXeucbP/4sXzxSX5QBvHE47W1tfl5V/KyDwzf+GCzVu3
fCeq6K+/8bo13LXFnPNnTIvaHUOxTYWbG6IskQOh6p7SUOOplo4SY8bmC6sh
hOd8Q5dR386YJbzZQZtuOPVMcS1a8nF5QQVBYviblkCcd9yAtQzM8ktWg9DI
7dIv5jG/O7jwsfMPKlAJmMIrnzTaQ9y2Oa23ae1rtNPNyzamB/hRJU9IkxUq
dtI6sBVHO/GoEkOQZDttCzWKJ8nv5n4oduQBFmbxQ7krZNCPJaxhOla0rwPv
ULqBjnTosK2J8uyYREDhoAkqhbt0kbFmPxGLpo3lbvZMrQ67pn1JXRuvtdYR
TbrkIpdMkbSOK451XnWgUroIp61/PnNe2pmTsIqVleXoROvM5ZQyMjLJzVRi
ovdiSBAtNYyBFOYXOO2OrMyzFDjTeEi/qclWX1/LbdpLBfmVZYU5xWWUfsLd
ySrrGLXEbTZj17bExIogn/9iuvPvDz62Pz7t1UH9X331le7de3Z9ohtgo3//
V4YNG/LJp/NfHz6i21NPDxsytK9vn1dfGfjM092qKirf8x/r27c39O/b19fP
b0xWVub9998L/V/s13fYiOHnMtLu7fLo8089NmbcB77PPXnfQ3/psLTBw+DE
Le6qnbvh5/LlyyMPxlyurz4ef2xjcMiJhBS4GbQueP++vR8vnAuwbwnffCo1
bXNY+JFDh9esWg13TyQkhm8Ji4qKLCkpunQpFwy58PDQPXui4JE90TGuxtqs
S8XbQ1bviztZfDEraNOWvNyL3zbUbbc242x4ohgbFlTuxGGFMFZu0J4CDYPo
vDuvVyHhwYUKCRmKr+t8r5PmZcGS4sYTM7rOW5thQKuL7RWFbg3INWRP2mym
aoATmCSc48dHMI2A4Ii66tnHYaoXQ78YCteInMuYDvC4wLwClIez2JfYG16q
1itIxWfV5np5PWvGEq/T6bYGV62YEXMvAO+MnTZh35FDmWkZcfuPVFfV5xeX
1ddVFRVWguEavH5bRmZCTHScy92QkJhSVlJ+Lis9+USqogqJJ+Kyz+VsDFkn
SUpeXg40cP34yJ6EcouYiTWkbOY+vDDTZvyzNYq+7MrV4oSfwGz5UDE+t6mx
aWNmzF+1ceu8yZP7D+zx+4fu3hCyKznp+B233/3+B2O6Pt73w+lv9+41qO+L
3ce89e6kCRNjjxz43W2/Hzp0MIjQnJy8D6e9/9hjjw0Y8PK777779NNPe0Fn
jW/z615bUE2QW2dRvXBiIuoKQ8Ff0awJqRavwM8MSA2X60rzi/n3CCSsraVs
oCTX19WIoswNS749VqcKWMFt54MIIv8GocE14A6Lotia8luDaYLWOq5inreO
1FmLo9j10UYbFjUixUk5I6rTUNyodjT6NoUhB5qtUGYYUwp9BotKHEicYsJU
E81Noy2L3JjXG9skBtZSinphwKsirs06lutpLQNcJAllvhsL0eCWJZ6SVhXm
2askUIGnYugUOnrW1KzZ07zIz6QOPn8rg/DmtTO3ZZatOT/S+if7ck/qqhqM
sH79+tDQ0OLi4ri4uLq6usKyqvzSwtS0E05bvShpoZF70tKzck9m2Z1NJWWl
2RfOccxNeX/qpdy8jNNnUlPSGi5X19U0bNu25ciRQ4JbyTx7KjExCfrU1NQc
PXoUem/btu3YsWNlZWWCIKWmpkZFRbnd7tra2pSUFBCzHIpLly4lJycDioqK
iuBnaWlpQ0NDZWVldXW1zWaDicHjSUlJ6enp0N/pdFZUVMCcQ0JCYKgrr9P4
qqbjJwP69+k1a/Z8v+FDR498428PPdTt2WdcOuvyv/f0eqnfgGeefOttv8R9
MS/27XHv3fdNmjD5Nzd3XvF5IBBD1388+dnSFe/4je3R4/kHHrxn184DMfu3
P/jnx3v36fbh1Jk9e3YfMWLE+PHjQYr6+/sDFI888sidd94pCML9f/6Tv/87
3Z7u/uyz3SdPnvjXv3Xp2vWp4KAvIqPCn+/17IULOZ1+dlNK8ulevXy7PHz/
iy/1fGXIkD/dd/+748YPGvwamMF3//GutSsD1yxf3u2ZHg88+Ijf2+8WFBbz
eBRFY65DpxA2nLa6s1kXQEY6ne7q8jKK7MjVtTVN1TXAL0JDBecOWGi4AX1q
a+vhJCsr22ZrghM41tVXYT5bdZaX1UkyFztqfX19eXk5HGEdYcVhibltUFSc
B/q2pqYuMjIKTkBEBwVtKCut1vnGEcZycwpgBKAK+JmQeOTs+QvV9ZfBJ3Y4
HEAGtoZ6YFnJ5aworwHHAeYD9CaKfHuLxq4VG8TkKv9OEcYfdL5PHINdIlVP
8w+I6lT5TB+T1K3ykzez5tMzGRqjpV7gzcy90ldiuIRkZs7RkD8eUeORGxjD
536emYo1k5X8CmshXq4PGzxgh9FyBv6FxL8ny7cs8H2+5Msw44s+RqMghsKM
zw5wlUdfD6Oj4tleYUVFS/nWnLKxAG7CZQBrimWzEsDEs8kRstycp/YYpdfO
KTqpAIB67949Mfv3GchBGxs/+qhSDY6kwavlnNxLwCkulxkZa54z1ym6sTED
v59J1ThGJy9FaalbaLuAgUiF5zI4MehGXlg3rCyMG+GUmjH8DcXzeaAGgV+8
aKGgqDbbZbiSkpScfe5Mgw2NKIn0eX5hQebZ04cPxzKPaU0nqpltV3llDOGE
a14v0c5ZzKpw+bPw1/zAhUn5HlR7mu4dFdEtoWndU0TKLPR2TcggG0DCqoOg
1Ssa7K4//d+dtfV1/V70fXPwqNWbtqccObJm65aagpI5H3+cnnps6JDXOSfx
+ZtE0oxb3fyptVm7YtC58VHiZl+YHuaOnuqRG82vMGnAq67MyqrfQAiab5Ch
bUSbgtbgPjTBIcpSSVGp2y3S0humY21VYWNdla3RpXs+0mAUNdB9a4DdOjEv
QeoVQG6WpRbk8PsK397ieY6/SPFsYTEZh1lrC3UPCV0raZiTj4+PX7169TWO
8n1p5qqB3w0mAftG6O273Kx+8Q8k0/oVrS2vquOGcf/VTfdU6PGfP2RO8Yo1
dfDPUv2rm5fu81Lu3+P2/79kY2A=
	"], {{0, 100}, {90, 0}}, {0, 255},
	ColorFunction->RGBColor],
   BoxForm`ImageTag["Byte", ColorSpace -> "RGB", Interleaving -> True],
   Selectable->False],
  DefaultBaseStyle->"ImageGraphics",
  ImageSize->Automatic,
  ImageSizeRaw->{90, 100},
  PlotRange->{{0, 90}, {0, 100}}]
]






(****************************************************************************

	Icons / GraphicsBox

*****************************************************************************)


icon[_]:= Framed[Style["ICON", 12, FontFamily :> CurrentValue["PanelFontFamily"]]];



(**************************)


SlideBreakExpressionPanel[]:=
Module[{},
PanelBox[
       PaneBox[
        StyleBox["Slide Break" (*
         RowBox[{"Slide", " ",
           CounterBox["SlideShowNavigationBar"], " ", "Break"}] *), "DialogStyle", LineColor ->
         GrayLevel[0.25], System`FrontFaceColor -> GrayLevel[0.25], GraphicsColor ->
         GrayLevel[0.25], FontColor -> GrayLevel[0.25], BackFaceColor ->
         GrayLevel[0.25]], ImageSize -> 122, Alignment -> Center], Alignment ->
       Left, Appearance -> Image[CompressedData["
1:eJztXUmPXFcVtgSBBQobglg5UViZRoIFyQp1vAPTyya7xF5E2I6QYi8QTkuE
SImx3YmzACSEsrICCNYs+AX8BM/z1NXV5XlElqfHPY93yl999Z373uuqdjfp
KunTu8M53xnuqVtvavvVd3bPvvOVDRs2fD/hbwlfTSiKYoIJVg1Wgg4fO3Lk
SOE4evRoCe8fPny44Hnum4zS8THvI5frKZvKruL0trLNtpQNZUv5iDaVXWWT
Y2Q55ojid1kVN+tHOVP8nHfOL/K5TZThuDA3Sj5XQw6vx5MnTxYTTLDa8Ho8
e/Zs4Th37lxx/vz5fpvneMxkfdza3sc2zru+sqNkeY552UaOO4rTOZkLdZQd
tOFt5mPfkRPHWF/FquLhnPGYiiWXb/ZN5Rp9RX+V7Sh3bNvg9Xj58uUCcenS
pRI4trCwMDRvsHGfQx2UxzaOqXGfc65Ijv2L5t1H9AM5vc0yzIFjrKvy1Sbe
KC7ldxQXc7NP7J/yF9dRrSn7EOkrXyO7KOf1uLS4VDi6A+1u0ev2it6iYanE
UufZvKO3qMZ6Q2ORfLfTLW1p2e6AXq/iNj+7C93QvvLlf3H0+vJms9v3YdA/
96kHObExz1G3xqaKyfN5BXLzzE5vSNbtDcXTjXMb+dXn6jzL17M1rfK56PH1
BvzlHKP9IR8r/m60rp1qvDMcs9djr5fWqMLS0tIAfKzb7Q4cFXL63jZ9xcW8
3mf7OG9QNtl+nUwUl7LNftWN1+UnyifmKaffJKbcmtXpRD4jZ926N/Xd6/Hq
1auFw/SuXLky1GeocZd3XZ5DGeSIbDAH++VQnK4bcSsZ51Nxup2IC/Wi3HD8
kR6vAfuGfjAv+8n54TyqPCt9lX9ed9WP1kXF5vV48+bNwnHjxo0S169fHwDP
oxweFZe3cdw5fRz7qKds8ljkn/vEfA6MD2Wa8uX8dB5sKzuozznnOcwd54Hn
2FaUN+fFtooJ++iTWhMVZwTU8Xq8fft2wbhz505x69atoXEDjpuckon0mnCq
fmTHxn1OyeTiqPMh57OP5/hxTsnkYoryofhyucP8qHlvR77k1kvxNMl1tF5e
j/fu3Sscd+/eXXewvKyEfFveJvlfqTVq6+tK+OjPa+rk7t+/vyp10jafnodx
5Xaca7RSaLv246rnlciN74+j+DIOv9brvrwWsZprsdx6rIOfs3itYr/JnJ9b
+DyeB0Wyqh+1c6iTy8VTp4vnT01ii3SbIpJ323Zsy9kGq1mPKnd83svnv3bt
5UeDjXvb9Vg/ByXD59oOtKfmIz30OwLOYzuKHf1owhnFi/cOIlmMF/OOXOhb
3ZHjUNeaTX9Px1GP+D3zmOz6Hf31a/pr164NXOtj39t+7W99vCeB+nbEeeRR
Y+oeg3OoefZF8ebuebEM+os6GA/aYnnuo46Kw8dzdviIfBx3dK+Lc+NH9T1s
smeOWo9uw2vQa89zhPdo8R493svmZ0N435dlI10l5/p+v7XObht7eA+X+86l
+JTdpj7m7EVjURttup/qWRbHVLcGfN8bv0N4nrIS9Yi/zW4TfV5cXCw6nU4J
fGdBvcug3jtgOI8d1bN75kId67se6qNvzMN20H4URxRrG3kcQx9UP8eNY7l1
wBxFPkRrxvq+5nb0urW69PvsuFeOUo9qr7W+27Dvgdm2Z5rmy8WLF4sLFy7I
d4+8febMGfkuWPRelHrvDceRD3VsHHUVn8mgPuq4r8zDvik7Edi+8sX5kNfb
bFPlgo+Rf8wZ5Vv5FL3DZ2tv7+xYbfo7BlYjfo45aj2q+vTzYat9s2d1aN8T
G3/w4EEx+azvj9WA1YLVhNWG1YjvkzY+znr0WvRzRfsOmN1Hjx4Z55OE/yTc
SLCXNjoJl8eMhRXgHDdvG66FZci34R0n90IDWVvzqwkPHj58WP6WW43Y+Zxf
Z6g9shhxf/S90fZlG6s+5xP+lfCPhL8E+CIzV4dRdJ+3neVy/D/FGOHvCf9O
KGvDasT3SH/u36Qeo3tGfA/XOI3bzlvtXMG+B9Xnnwl/SPj9BOseVu/l76bV
iNWK/Z7iuyht6jH3DMT2XNt/7dzAzl+fPLGf6fLz1zWQhwnWBg4llLVh1zlW
K3a97b/XTeuR719y2+83ej3addXTp0+L6nNoDeRhgrWBQwkD9ei/2eo6u2iw
P6q9EevRzgvsWh/2x88TPq3wCcD6B2GcZeYDPdeZF3qfkgzOow6PK/ufBBw+
Nk+20PbBQG9e8KCteRrP+TNPcSodz/HBwJbKQy4XSk6taeTPnxPK2rAasWte
2x+j++NFph79+Z/aM/15ENYj7I9/SjhQYb9o74P2/kxbHXNyqo39/ZXtyLc6
Wwcyujx/IGMrFxf7y+P7gJvHVYyRzX2F9kn5wHyot6/QOfIxq4WyHu031OrR
9seoHv3fqMj9Xkfnk8Zp56dif/xjwscJe+mI2AvjHwVySk/NO9dHGRsfgy3W
U/7sDfSZR+nX+YpjKnaWZVtN7Kn8NclPztc26+U+23VtWRt2Dx3rUV3PRPXI
v9U87tfWxm02Tp8+jfX4WcIHCb9tgTbyHzSQX435DzPzdf1x5+l5x/9hMG+1
UNaG1cio9ch7Jf7Ng58/mo1Tp05hPf4uYS5hD2Cuwvtw3ENHxh6SnROyc9Te
E/DOBVysj3q/DvQ5Np5/H/RRRtlROVJ+5uxgvlAmyqvKr9KJ/OY8qbXwOdsr
ZT2qe+KqHv35S+5dTr+e8XvhVI/2vdgFeK867oYjAmV2kcyuDJTMexkbLMNy
kUydrRwPzo8rrhyPyvGujM3I1u4xyfwmoawN+/d5/J74KPtjtE9affvvNdXj
rxJ2VNhZYQeN4RxiewWX3U46v6jhQj3k3BnIKDvvwhjbZ2wvtM/Iw1zef5ds
eXw7STaCyivHy3GyjztIZ3ugE8W/I6NjsFro16PViv2m+nPsNvWontVgPRqn
/16bLbi+/mXC1oS3A2yl47aqvTUj87aQYdm3xPw2aL9Vw6M4eayJP9tqdFRc
6J/yoYk/kV4TnxUv+4p6vmY5nw1WC0P7Y3T/sen1jNorvR7tWbnZgv3RfP15
hdmEN6mPbZ9/U8yzHs6x/izZmxU6KM+cs0XeH+bLzbN9FQfzMU8uX7nYonhV
XlUuVI45v0o+sm31ObA/2j2ZNueP6lk11yU+u7Z6PHHiBNaj+TOTsCXhZ1V7
pmpvgXHsz9CYOvI8Y0bozYCNLUKW+VxH+cu8kb+Kn32YIR7l55ZiOC6e57jY
H2UX7f00yFcuP6gf2XN9q4WyNqxGvB79GXaTelT/poF6PoPnj1SPFuPmCm/A
0TFNfca00EeOzcARHV1mGvjeEHzTQoftoZ7yYXMR+8u2pwUX6kwT3zQdFSfb
j/zN9VEefeAYFbfKk+MnCQP747iuZ3AOr6/NxvHjx7EeLYbXKrxe4TWBaLzp
fFMgz48ayCzX3zqZ18fEE4FjU7lX8Te11SQ2xo8Tyto4duxYef6I70DyfZ22
98P53Ud/Xkj1+MOETQlTFb4Hxzpsoj7rcZ/lN4FcU5tTlR5zsYzimyK9qRpZ
9rnOropzKhiPcqRyE9mt08X+FOlNibkfJJS1YTXi9Yh/G7uc/VH9Zvv+aO9s
UD1+N+GVhJfFkdvc3yjmWfaVgHdjIK9s1vnBuk1lm6BONspVLr46Hs6r9zcu
g+tl0o30bPzVhH491r1v1ub3WtWj/62C2Xr8+HFRfb6T8FLCtyu8FCA31xSj
2BjVtzb+syz6vRI+fmsZ+VM+No2J2w6rhX49Wq3k/q6r7vo6t18ap/9dIe2P
30z4xgQTJLyY0K9H/jvD6PzRro8dpoew81DsuwzLwv749YQXJlj3+FqF/vUM
14+3fa76PPf/h2mCLx/GVSf/BZn4u04=
         "], "Byte", ColorSpace -> "RGB",
         Interleaving -> True], ImageSize -> {
         FrontEnd`AbsoluteCurrentValue[{WindowSize, 1}], Full}]
];





(*******************************************************

 CheckBoxMenuItem

	in: text of menu item, query whether item is checked
	out: Row

*******************************************************)
Options[checkBoxMenuItem] = {
	"Checkmark" -> "\[Checkmark]"
}
checkBoxMenuItem[text_, query_, OptionsPattern[]]:=
Module[{checkmark},

	checkmark = OptionValue["Checkmark"];
	With[{mark = checkmark, test = query},
		Row[Flatten@{ Style[mark, ShowContents -> test], text }, Spacer[2]]
	]
];
checkBoxMenuItem[a___]:= Message[checkBoxMenuItem::argx, {a}];
checkBoxMenuItem::argx = "`1`";
checkBoxMenuItem::info = "`1`";




colorPanel[text_:" ", background_]:=
With[{content = text, color = background},
	Panel[content, ImageSize -> {31, 15}, Background :> color]
]




(*******************************************************

 mouseover

	in: expr
	out: expr with mouseover, background

*******************************************************)
Options[mouseover] = {
	ImageMargins -> 0,
	FrameMargins -> {{0,0}, {2,2}},
	FrameStyle -> None,
	Alignment -> {Center, Center}
}
mouseover[image_String, OptionsPattern[]]:=
Module[{imageMargins, frameMargins, alignment, frameStyle},

	imageMargins = OptionValue[ImageMargins];
	frameMargins = OptionValue[FrameMargins];
	frameStyle = OptionValue[FrameStyle];
	alignment = OptionValue[Alignment];

	With[{img = RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, image]],
		im = imageMargins, fm = frameMargins, fs = frameStyle, align = alignment},
		MouseAppearance[
			Mouseover[
				Framed[
					Pane[Dynamic[img], FrameMargins -> 0, ImageMargins -> 0, Alignment -> {Center, Center} ],
					ImageMargins -> im,
					FrameMargins -> fm,
					Alignment -> align,
					FrameStyle -> fs,
					Background -> GrayLevel[0.95]
				],
				Framed[
					Pane[Dynamic[img], FrameMargins -> 0, ImageMargins -> 0, Alignment -> {Center, Center} ],
					ImageMargins -> im,
					FrameMargins -> fm,
					Alignment -> align,
					FrameStyle -> fs,
					Background -> GrayLevel[0.99]
				]
			], "LinkHand"
		]
	]
];


mouseover[{image_String, text_String}]:=
Module[{},
	With[{img = RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, image]],
		  str = FEPrivate`FrontEndResource["PresenterToolStrings", text]},
		MouseAppearance[
			Mouseover[
				Framed[
					Grid[{{
						Dynamic[img]
						}, {
						Style[Dynamic[str], 11, LineSpacing->{0, 15}, FontFamily :> CurrentValue["PanelFontFamily"],
							TextAlignment -> Center]
						}}, Alignment -> {Center, Center}, Spacings -> {0.5, 0.5}
					], ImageMargins -> 0, FrameMargins -> {{0,0}, {2,2}}, Alignment -> {Center, Center},
						FrameStyle -> None, Background -> GrayLevel[0.95]
				],
				Framed[
					Grid[{{
						Dynamic[img]
						}, {
						Style[Dynamic[str], 11, LineSpacing->{0, 15}, FontFamily :> CurrentValue["PanelFontFamily"],
							TextAlignment -> Center]
						}}, Alignment -> {Center, Center}, Spacings -> {0.5, 0.5}
					], ImageMargins -> 0, FrameMargins -> {{0,0}, {2,2}}, Alignment -> {Center, Center},
						FrameStyle -> None, Background -> GrayLevel[0.99]
				]
			], "LinkHand"
		]
	]
];

mouseoverPanel[image_String]:=
	With[{img = RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, image]]},
		MouseAppearance[
			Mouseover[
				Panel[
					Dynamic[img],
					ImageMargins -> 0, FrameMargins -> 10,
					Alignment -> {Center, Center}, Background -> GrayLevel[0.95]
				],
				Panel[
					Dynamic[img],
					ImageMargins -> 0, FrameMargins -> 10,
					Alignment -> {Center, Center}, Background -> GrayLevel[0.99]
				]
			], "LinkHand"
		]
	];

mouseoverText[txt_]:=
	MouseAppearance[
		Mouseover[
			Panel[
				txt,
				ImageMargins -> 0, FrameMargins -> 6,
				Alignment -> {Center, Center}, Background -> GrayLevel[0.95]
			],
			Panel[
				txt,
				ImageMargins -> 0, FrameMargins -> 6,
				Alignment -> {Center, Center}, Background -> GrayLevel[0.99]
			]
		], "LinkHand"
	];
(*
mouseover[a___]:= Message[mouseover::argx, {a}];
mouseover::argx = "`1`";
mouseover::info = "`1`";
*)


Options[mouseoverFramed] = {
	ImageMargins -> 0,
	FrameMargins -> {{0,0}, {2,2}},
	FrameStyle -> None,
	Alignment -> {Center, Center}
}
mouseoverFramed[name_, OptionsPattern[]]:=
Module[{imageMargins, frameMargins, alignment, frameStyle},

	imageMargins = OptionValue[ImageMargins];
	frameMargins = OptionValue[FrameMargins];
	frameStyle = OptionValue[FrameStyle];
	alignment = OptionValue[Alignment];

	With[{str = name,
		im = imageMargins, fm = frameMargins, fs = frameStyle, align = alignment},
		MouseAppearance[
			Mouseover[
				Framed[
					Pane[str, FrameMargins -> 0, ImageMargins -> 0, Alignment -> {Center, Center} ],
					ImageMargins -> im,
					FrameMargins -> fm,
					Alignment -> align,
					FrameStyle -> fs,
					Background -> GrayLevel[0.95]
				],
				Framed[
					Pane[str, FrameMargins -> 0, ImageMargins -> 0, Alignment -> {Center, Center} ],
					ImageMargins -> im,
					FrameMargins -> fm,
					Alignment -> align,
					FrameStyle -> fs,
					Background -> GrayLevel[0.99]
				]
			], "LinkHand"
		]
	]
];



(*******************************************************

	Toolbar dropdown button template

	in: img, color
	out: Framed[...]

*******************************************************)

Options[dropdown] = {
	FrameStyle -> GrayLevel[0.75],
	"DropDown" -> {Item[Dynamic@RawBoxes@FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, "TriangleDropDown.png"]], Alignment -> Right], SpanFromLeft}
}
dropdown[img_, color_, opts___?OptionQ]:=
Module[{frameStyle, dropDown},
	frameStyle = FrameStyle /. {opts} /. Options[dropdown];
	dropDown = "DropDown" /. {opts} /. Options[dropdown];

	With[{image = img, background = color, frame = frameStyle, dropval = dropDown},
		Framed[
			Grid[{
(*				{Style[" ", FontSize -> 2], SpanFromLeft}, *)
				{Dynamic@RawBoxes@image, SpanFromLeft},
				dropval
				}, Alignment -> {Center, {Bottom, Top}}, Spacings -> {0.5, {0, 0, 0}}, ItemSize -> {Automatic, {Automatic, 0}}],
			ImageMargins -> 0, FrameMargins -> 0, ImageSize -> {24, 24},
			RoundingRadius -> 3, Alignment -> {Center, Bottom}, FrameStyle -> frame,
			Background -> background
		]
	]
];

dropdown[img_String, opts___?OptionQ]:=
Module[{frameStyle, dropDown},
	frameStyle = FrameStyle /. {opts} /. Options[dropdown];
	dropDown = "DropDown" /. {opts} /. Options[dropdown];

	With[{image = FEPrivate`ImportImage[FrontEnd`FileName[{"PresenterTools"}, img]], (*background = color,*)
		  frame = frameStyle, dropval = dropDown},
		MouseAppearance[
			Mouseover[
				Framed[
					Grid[{
						{Dynamic@RawBoxes@image, SpanFromLeft},
						dropval
						}, Alignment -> {Center, {Bottom, Top}}, Spacings -> {0.5, {0, 0, 0}}, ItemSize -> {Automatic, {Automatic, 0}}],
					ImageMargins -> 0, FrameMargins -> 0, ImageSize -> {24, 24},
					RoundingRadius -> 3, Alignment -> {Center, Bottom}, FrameStyle -> frame,
					Background -> GrayLevel[0.95]
				],
				Framed[
					Grid[{
						{Dynamic@RawBoxes@image, SpanFromLeft},
						dropval
						}, Alignment -> {Center, {Bottom, Top}}, Spacings -> {0.5, {0, 0, 0}}, ItemSize -> {Automatic, {Automatic, 0}}],
					ImageMargins -> 0, FrameMargins -> 0, ImageSize -> {24, 24},
					RoundingRadius -> 3, Alignment -> {Center, Bottom}, FrameStyle -> frame,
					Background -> GrayLevel[0.99]
				]
			], "LinkHand"
		]
	]
];
(*
dropdown[a___]:= Message[dropdown::argx, {a}];
dropdown::argx = "`1`";
dropdown::info = "`1`";
*)


End[] (* End Private Context *)

EndPackage[]
