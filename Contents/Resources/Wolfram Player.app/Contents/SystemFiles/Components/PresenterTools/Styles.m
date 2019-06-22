(* :Title: PresenterTools *)

(* :Author: Andrew Hunt, andyh@wolfram.com *)

(* :Mathematica Version: 11.3.0 *)

(* Created by the Wolfram Workbench Feb 25, 2016 *)

(* :Discussion:

	Stylesheet manipulation for Presentation tools

	Some ideas and code snippets based on
	AuthorTools`StyleTools by Lou D'Andria

*)

(***********************************************************************************

	PresenterTools

	STYLES

***********************************************************************************)



(************ PUBLIC ************)

BeginPackage["PresenterTools`Styles`"]
(* Exported symbols added here with SymbolName::usage *)

(* TODO: GetStyleOptionValue needs better querying of option values *)
GetStyleOptionValue::usage = "GetStyleOptionValue[stylesheet, style, environment, option] returns the option value of style within the given stylesheet and environment.";
DefaultOptionValueQ::usage = "DefaultOptionValueQ";
NewCellStyleDialog::usage = "NewCellStyleDialog[notebook] allows user to create a new cell style within stylesheet.";


StylesheetQ::usage = "";
PrivateStylesheetQ::usage = "PrivateStylesheetQ[nb] returns True if nb has an embedded stylesheet.";

GetStylesheet::usage = "";
SetNotebookStylesheet::usage = "SetNotebookStylesheet[obj, expr] embeds stylesheet expr to Notebook object.";
GetStylesheetsParent::usage = "GetStylesheetsParent[stylesheet] returns parent stylesheet if they exist.";

InstallStylesheet::usage = "InstallStylesheet[filename, nb] saves a copy of nb to FileNameJoin[{$UserStylesheetDirectory, filename}]."


AddCellStyle::usage = "AddCellStyle[cell, style] adds style to cell expression.";
RemoveCellStyle::usage = "RemoveCellStyle[cell, style] removes style from cell expression.";


$UserStylesheetDirectory::usage = "$UserStylesheetDirectory gives the location for user defined stylesheets.";

UpdateStylesheet::usage = "";
CreateEmptyStylesheetNotebook::usage = "";
CreateStyleDataCell::usage = "";
SetStyleOptionsList::usage = "";
RemoveStyleOptionsList::usage = "";
ReplaceStyleOptionsList::usage = "";
GetUserCellStyleOptionValue::usage = "";

GetStyleOptions::usage = "";
GetStylesheetCells::usage = "";
GetStylesheetStyles::usage = "";
GetStylesheetEnvironments::usage = "";
GetStylesheetMenuStyles::usage = "";

CreatePrivateStylesheet::usage = "";

StylesheetsInLayout::usage = "";
EmbedStylesheet::usage = "";



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




Begin["`Private`"]
(* Implementation of the package *)

(* Print full messages *)
$MessagePrePrint = .;






(*******************************************************

 StylesheetsInLayout

	in: xxx
	out: yyy

*******************************************************)

StylesheetsInLayout[]:=
Module[{},

	Union @
	Flatten[
		Join[
			FileNames["*.nb", FileNameJoin[{#, "*", "FrontEnd", "StyleSheets"}]] & /@ $Path,
			FileNames["*.nb", #] & /@ (ToFileName[#] & /@ (CurrentValue[StyleSheetPath] /. System`ParentList :> {})),
			FileNames["*.nb", FileNameJoin[{$InstallationDirectory, "SystemFiles", "FrontEnd", "Stylesheets"}]]
		]
	]
];




(*******************************************************

 GetStylesheet
	Return a Notebook's StyleDefinitions Notebook expression

	in: NotebookObject | Notebook | File | String
	out: Notebook

*******************************************************)

GetStylesheet[nbExpr:Notebook[con__, opts___?OptionQ]] :=
Module[{styleSheet, parent},

	If[StylesheetQ[nbExpr],
		nbExpr,
	(* ELSE *)
		styleSheet = StyleDefinitions /. {opts};
		parent = Select[StylesheetsInLayout[], MatchQ[StringJoin[FileBaseName[#], ".nb"], Alternatives @@ styleSheet]& ];
		Get /@ parent
	]
]

GetStylesheet[nbObj_NotebookObject] :=
Module[{expr, styleDefinitions, fileName, stylesheetpath},
	expr = CurrentValue[nbObj, StyleDefinitions];
	If[Head[expr] === Notebook,
		expr,
	(* ELSE *)
		styleDefinitions = First[("StyleDefinitions" /. NotebookInformation[nbObj])];
		fileName = "FileName" /. NotebookInformation[styleDefinitions];
		If[fileName === "FileName", Return[None]];

		stylesheetpath = ToFileName[fileName];
		(* TODO: Error checcking needed *)
		Get[stylesheetpath]
	]
]
(* not finished *)
(*
GetStylesheet[file_String /; FileType[file] === File] :=
Module[{nbObj, styles, return},
	nbObj = NotebookOpen[file, Visible -> False];
	styles = CurrentValue[nbObj, StyleDefinitions];
	If[PrivateStylesheetQ[styles],
		return = Notebook[
	If[StylesheetQ[styles],
		return = styles,
	(* ELSE *)
		If[StylesheetQ[nbObj],
			return = NotebookGet[nbObj],
		(* ELSE *)
			Message[GetStylesheet::info, "Not stylesheet found"];
		],
	(* ELSE *)
		Message[GetStylesheet::info, "Not a stylesheet"];
	];
	NotebookClose[nbObj];

	return
];
*)

GetStylesheet[a___] := Message[GetStylesheet::argx, "A Notebook is expected."];
GetStylesheet::argx = "`1`";
GetStylesheet::info = "`1`";




(*******************************************************

 SetNotebookStylesheet

	in: xxx
	out: yyy

*******************************************************)

SetNotebookStylesheet[nbObj_NotebookObject]:=
Module[{stylesheet},

	stylesheet = CurrentValue[nbObj, StyleDefinitions];
	(* Still needed? *)
	SetOptions[nbObj, Visible -> True]
]

SetNotebookStylesheet[nbObj_NotebookObject, nbExpr_Notebook]:=
	CurrentValue[nbObj, StyleDefinitions] = nbExpr

SetNotebookStylesheet[a___]:= Message[SetNotebookStylesheet::argx, "A NotebookObject is expected."];
SetNotebookStylesheet::argx = "`1`";




(*******************************************************

 CreatePrivateStylesheet

	in: xxx
	out: yyy

*******************************************************)

CreatePrivateStylesheet[content_:Nothing]:=
Module[{},

	With[{cells = Flatten[{content}]},
		Notebook[cells,
			WindowSize -> {800, 800},
			WindowMargins -> {{Automatic, Automatic}, {Automatic, 0}},
			FrontEndVersion -> "11.1 for Mac OS X x86 (32-bit, 64-bit Kernel) (February 13, 2017)",
			StyleDefinitions -> "PrivateStylesheetFormatting.nb"
		]
	]
];
CreatePrivateStylesheet::argx = "`1`";
CreatePrivateStylesheet::info = "`1`";




(*******************************************************

 StylesheetQ

	in: xxx
	out: yyy

*******************************************************)

StylesheetQ[nbExpr_Notebook] :=
  Length[Cases[nbExpr, Cell[_StyleData,___], Infinity, 1]] > 0

StylesheetQ[nbObj_NotebookObject] :=
  Length[Cases[NotebookGet[nbObj], Cell[_StyleData,___], Infinity, 1]] > 0

StylesheetQ[a___]:= (
	Message[StylesheetQ::argx, "A Notebook is expected."]; False
	)

StylesheetQ::argx = "`1`";
StylesheetQ::info = "`1`";




(*******************************************************

 PrivateStylesheetQ

	in: xxx
	out: yyy

*******************************************************)

PrivateStylesheetQ[nbObj_NotebookObject]:=
Module[{styleSheet},
	styleSheet = CurrentValue[nbObj, StyleDefinitions];

	Which[
		Head[styleSheet] === String, False,
		Head[styleSheet] === Notebook, True,
		Head[styleSheet] === FrontEnd`FileName, True,
		True, False
	]
]

PrivateStylesheetQ[Notebook[con__, opts___?OptionQ]] :=
Module[{styleSheet},
	styleSheet = StyleDefinitions /. {opts};
	Head[styleSheet] === Notebook
]

PrivateStylesheetQ[a___]:= (
	Message[PrivateStylesheetQ::argx, "A Notebook or NotebookObject is expected."];
	False
	)
PrivateStylesheetQ::argx = "`1`";




(*******************************************************

 PathToStylesheet

	in: xxx
	out: yyy

*******************************************************)

PathToStylesheet[nbObj_NotebookObject] :=
Block[{styleDefs, fileName, stylesheetpath},
	styleDefs = First["StyleDefinitions" /. NotebookInformation[nbObj]];
	fileName = "FileName" /. NotebookInformation[styleDefs];
	If[fileName === "FileName", Return[None]];

	stylesheetpath = ToFileName[fileName];
	If[StringQ @ stylesheetpath, stylesheetpath, None]
]
PathToStylesheet[a___]:= Message[PathToStylesheet::argx, {a}];

PathToStylesheet::argx = "`1`";
PathToStylesheet::nofile = "No file `1` found.";




(*******************************************************

 GetStylesheetsParent

	in: Stylesheet
	out: List of stylesheets

*******************************************************)

GetStylesheetsParent[nb_NotebookObject /; StylesheetQ[nb]]:=
Module[{nbExpr},
	nbExpr = NotebookGet[nb];
	GetStylesheetsParent[nbExpr]
];

GetStylesheetsParent[nb_Notebook /; StylesheetQ[nb]]:=
Module[{stylesheet, parent, installedStylesheets},
	stylesheet = Cases[nb, Cell[StyleData[Rule[StyleDefinitions, ss_], ___], ___] :> ss, Infinity];
	If[stylesheet === {}, Return[$Failed]];

	stylesheet = stylesheet /. ss_FrontEnd`FileName :> ToFileName@ss;

	installedStylesheets =
		Join[
			FileNames["*.nb", FileNameJoin[{PresenterTools`Private`$PresenterToolsDirectory, "FrontEnd", "StyleSheets", "PresenterTools"}]],
			StylesheetsInLayout[]
			];
	parent = Select[installedStylesheets, MatchQ[StringJoin[FileBaseName[#], ".nb"], Alternatives @@ stylesheet]& ];
	Get /@ parent
];

GetStylesheetsParent[a___]:= (
	Message[GetStylesheetsParent::argx, "A Notebook or NotebookObject expected"]; $Failed
);

GetStylesheetsParent::argx = "`1`";
GetStylesheetsParent::info = "`1`";








(********************************************************************************************************)

(**************************************

 UpdateStylesheet

	in: List of Style / CellOptions updates to style sheet
		e.g.
		{"Text", "Slideshow Working", None, CellMargins -> margins}
		{"Text",
			"Slideshow Presentation",
			StyleDefinitions -> StyleData["Text", "Slideshow Working"],
			{
				CellMargins -> margins,
				...
			}
		}

	out: Notebook

***************************************)

UpdateStylesheet[{}, a___]:=
	CreateEmptyStylesheetNotebook[];

UpdateStylesheet[list_List ]:=
	UpdateStylesheet[list, CreateEmptyStylesheetNotebook[]];

UpdateStylesheet[list_List, styleSheet_String ]:=
	UpdateStylesheet[list, CreateEmptyStylesheetNotebook[styleSheet]];

UpdateStylesheet[list_List, styleSheet_FrontEnd`FileName ]:=
	UpdateStylesheet[list, CreateEmptyStylesheetNotebook[styleSheet]];


UpdateStylesheet[{}, styleNotebook_, ___]:= styleNotebook;

UpdateStylesheet[list_List, styleNotebook_Notebook]:=
Module[{styleSheet = styleNotebook, environment, inheritStyleData, pos, styleName, styleOptions, res, newStyleData},

	Logger["UpdateStylesheet:"];
	Logger[list];

	(
		styleName = Part[#, 1];
		environment = Part[#, 2];
		inheritStyleData = Part[#, 3];
		styleOptions = ReleaseHold[Part[#, 4]];
		(* FE adds MenuSortingValue -> 10000 by default *)
		AppendTo[styleOptions, System`MenuSortingValue->Inherited];

		Logger["-- Style name: " <> styleName];
		Logger["-- Style options: " <> ToString[styleOptions]];

		styleOptions = styleOptions /. Rule[o_, s_Hold] :> Rule[o, ReleaseHold[s]];
		styleOptions = styleOptions /. RuleDelayed[o_, s_Hold] :> RuleDelayed[o, ReleaseHold[s]];
		styleOptions = styleOptions /. Rule[o_, Dynamic[Hold[a_]]] :> Rule[o, Dynamic[ReleaseHold[a]]];
		styleOptions = styleOptions /. RuleDelayed[o_, Dynamic[Hold[a_]]] :> RuleDelayed[o, Dynamic[ReleaseHold[a]]];

		If[environment =!= "Working",
			If[inheritStyleData === None,
				pos = Position[styleSheet, Cell[StyleData[styleName, environment], ___]],
			(* ELSE *)
				pos = Position[styleSheet, Cell[StyleData[styleName, environment, inheritStyleData], ___]]
			],
		(* ELSE *)
			If[inheritStyleData === None,
				pos = Position[styleSheet, Cell[StyleData[styleName], ___]],
			(* ELSE *)
				pos = Position[styleSheet, Cell[StyleData[styleName, inheritStyleData], ___]]
			]
		];

		(* Add new options to existing style definitions *)
		If[TrueQ[pos =!= {}],
			(* Add new options to existing style definitions *)
			res = Extract[styleSheet, First@pos];
			newStyleData = PresenterTools`ResetOptions[res, styleOptions];

			styleSheet = ReplacePart[styleSheet, pos -> newStyleData],
		(* ELSE *)
			(* Create new style definition *)
			styleSheet = PresenterTools`AddCellsToEndOfNotebook[styleSheet,
							Flatten@{CreateStyleDataCell[styleName, environment, inheritStyleData, styleOptions]}]
		]

	) & /@ list;

	(* Close CellGroups within embedded stylesheet *)
	styleSheet = styleSheet /. CellGroupData[a, _] :> CellGroupData[a, Closed];
	(* Return stylesheet *)
	styleSheet
]
UpdateStylesheet[a__]:= Message[UpdateStylesheet::argx, {a}];
UpdateStylesheet::info = "`1`";



(**************************************

 CreateStyleDataCell

	in: Style name: String
	out: Cell

***************************************)

CreateStyleDataCell[style_String, env_String ]:=
	CreateStyleDataCell[style, env, None, {}]
(*  *)

CreateStyleDataCell[style_String, "Working", None, opts_List]:=
	Cell[StyleData[style], Sequence@@opts, MenuSortingValue->10000]

CreateStyleDataCell[style_String, "Working", inherit_, opts_List]:=
	Cell[StyleData[style, inherit], Sequence@@opts, MenuSortingValue->10000]


CreateStyleDataCell[style_String, env_String, inherit_, opts_List]:=
	Cell[StyleData[style, env, inherit], Sequence@@opts, MenuSortingValue->10000];

CreateStyleDataCell[style_String, env_String, None, opts_List]:=
	Cell[StyleData[style, env], Sequence@@opts, MenuSortingValue->10000];



(*
CreateStyleDataCell[style_String, env_String, inherit_, opts_List]:=

CreateStyleDataCell[style_String, env_String, inherit_, opts_List]:=
Module[{},

	If[env =!= "Working",
		Cell[StyleData[style, env], Sequence@@opts, MenuSortingValue->10000],
	(* ELSE *)
		Cell[StyleData[style], Sequence@@opts, MenuSortingValue->10000]
	]
]
*)


(**************************************

 CreateEmptyStylesheetNotebook

	in: None
	out: Notebook

***************************************)

CreateEmptyStylesheetNotebook[]:=
Module[{},
	Notebook[{
		Cell[StyleData[StyleDefinitions -> "Default.nb"]]
	}, StyleDefinitions -> "PrivateStylesheetFormatting.nb"]
]
CreateEmptyStylesheetNotebook[styleSheet_String]:=
Module[{},
	Notebook[{
		Cell[StyleData[StyleDefinitions -> styleSheet]]
	}, StyleDefinitions -> "PrivateStylesheetFormatting.nb"]
]
CreateEmptyStylesheetNotebook[styleSheet_FrontEnd`FileName]:=
Module[{},
	Notebook[{
		Cell[StyleData[StyleDefinitions -> styleSheet]]
	}, StyleDefinitions -> "PrivateStylesheetFormatting.nb"]
]




(*******************************************************

 EmbedStylesheet

	in: xxx
	out: yyy

*******************************************************)

EmbedStylesheet[nb_Notebook]:= EmbedStylesheet[nb, Automatic];

EmbedStylesheet[nbExpr:Notebook[con_, nbopts___?OptionQ], output_]:=
Module[{opts, styleSheet},
	(* Get stylesheet *)
	styleSheet = GetStylesheet[nbExpr];

	If[Head[styleSheet] =!= Notebook, Return[$Failed]];

	opts = DeleteCases[{nbopts}, StyleDefinitions -> _];

	Notebook[con, StyleDefinitions -> styleSheet, Sequence@@opts]
];
EmbedStylesheet[a___]:= Message[EmbedStylesheet::argx, {a}];
EmbedStylesheet::argx = "`1`";
EmbedStylesheet::info = "`1`";





(*******************************************************

 NewCellStyleDialog

	in: xxx
	out: yyy

*******************************************************)

NewCellStyleDialog[nbObj_, opts___?OptionQ]:=
Module[{returnValue, styleList, styleSheet, updatedStylesheet, newStyle, newStyleFontSize, newStyleMargins},

	styleList = FE`Evaluate[FEPrivate`GetPopupList["MenuListStyles"]];
	styleList = If[Head[styleList] === List, styleList /. Rule[sty_, _] :> sty, Return[$Failed]];

(*	styleSheet = CurrentValue[nbObj, StyleDefinitions];*)

	PresenterTools`Private`Logger[styleList];
	PresenterTools`Private`Logger[styleSheet];

	styleList =
		Join[{"None"->"None"},
			DeleteCases[styleList, Rule[All, _], Infinity]
			];
	PresenterTools`Private`Logger[styleList];

	With[{$CellContext`nb$ = nbObj},
		CreateDialog[
			DynamicModule[{styleName, inheritedStyle = "None"},
				Pane[
					Grid[{{
						Style[Dynamic[FEPrivate`FrontEndResource["PresenterToolStrings", "AddNewStyleLabel"]], 15, FontFamily :> CurrentValue["PanelFontFamily"]],
						SpanFromLeft
						}, {
						Style[Dynamic[FEPrivate`FrontEndResource["PresenterToolStrings", "StyleNameLabel"]], FontSize -> CurrentValue["PanelFontSize"], FontFamily :> CurrentValue["PanelFontFamily"]],
							Style[Dynamic[FEPrivate`FrontEndResource["PresenterToolStrings", "StyleInheritanceLabel"]], FontSize -> CurrentValue["PanelFontSize"], FontFamily :> CurrentValue["PanelFontFamily"]]
						}, {
						InputField[Dynamic[styleName], String, FieldHint -> Dynamic[FEPrivate`FrontEndResource["PresenterToolStrings", "NewStyleNameHintLabel"]], System`BoxID -> "styleInputID",
							BaseStyle -> {12}, FieldSize -> {20, 1}, FrameMargins :> FEPrivate`If[FEPrivate`$OperatingSystem === "MacOSX", Automatic, 4], Appearance -> Medium],
						PopupMenu[Dynamic[inheritedStyle], styleList, BaseStyle -> {12}, FieldSize -> {Automatic, 1}, Appearance -> Medium]
						}, {
						Item[
							ChoiceButtons[{
								Style["OK", FontColor :> FEPrivate`If[FEPrivate`$OperatingSystem === "MacOSX", GrayLevel[1], Automatic]]
								},
								{
									If[StringQ[styleName],
										DialogReturn[
											(*CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "AddNewStylePalette", "PaletteNumber"}] = 0;*)
											CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "Dialogs", "AddNewStyle"}] = 0;
											NewCellStyleDialogResults[nbObj, <|"StyleName" -> styleName, "InheritedStyle" -> inheritedStyle|>]
										],
									(* ELSE *)
										DialogReturn[
											(*CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "AddNewStylePalette", "PaletteNumber"}] = 0;*)
											CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "Dialogs", "AddNewStyle"}] = 0;
											$Canceled
										]
									],
									DialogReturn[
										(*CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "AddNewStylePalette", "PaletteNumber"}] = 0;*)
										CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "Dialogs", "AddNewStyle"}] = 0;
										$Canceled
									]
							}], Alignment -> Right], SpanFromLeft
						}},
					Alignment -> {Left, Center}, Spacings -> {1, {1, 2, 0.5, 2}}, Frame -> None],
					ImageMargins -> {{10, 10}, {15, 15}}
				],
				Initialization :> (
					FrontEnd`MoveCursorToInputField[EvaluationNotebook[], "styleInputID"]
				)
			],
			NotebookEventActions -> {
				"EscapeKeyDown" :> FE`Evaluate[FEPrivate`FindAndClickCancelButton[]],
				"WindowClose" :> (FE`Evaluate[FEPrivate`FindAndClickCancelButton[]];
									(*CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "AddNewStylePalette", "PaletteNumber"}] = 0*);
									CurrentValue[nbObj, {TaggingRules, "PresenterSettings", "Dialogs", "AddNewStyle"}] = 0;
									)
				}
(*
				,
			NotebookDynamicExpression :> (
				If[Not[MemberQ[Notebooks[], $CellContext`nb$]], NotebookClose[EvaluationNotebook[]]]
			)
*)
		]
	]
];
NewCellStyleDialog::argx = "`1`";
NewCellStyleDialog::info = "`1`";




NewCellStyleDialogResults[nbObj_, $Canceled]:= Return[];

NewCellStyleDialogResults[nbObj_, returnValue_]:=
Module[{newStyle, newStyleMargins, newStyleFontSize, styleSheet, updatedStylesheet},

	(* user cancels dialog *)
	If[returnValue === $Canceled, Return[]];

	styleSheet = CurrentValue[nbObj, StyleDefinitions];

	(* proceed with creating new style *)
	If[returnValue["InheritedStyle"] =!= "None",
		newStyle = {{returnValue["StyleName"], "Working", StyleDefinitions -> StyleData[returnValue["InheritedStyle"]], {MenuSortingValue -> 10000} }},
	(* ELSE *)
		newStyleMargins = PresenterTools`Styles`GetStyleOptionValue[styleSheet, "Text", CellMargins, "Slideshow Working"];
		newStyleFontSize = PresenterTools`Styles`GetStyleOptionValue[styleSheet, "Text", FontSize, "Slideshow Working"];

		newStyle =
			With[{marg = newStyleMargins, fontsz = newStyleFontSize},
				{{
					returnValue["StyleName"],
					"Working",
					None,
					{
						CellMargins -> marg,
						FontSize -> fontsz
					}
				}}
			]
	];

	updatedStylesheet = PresenterTools`Styles`UpdateStylesheet[newStyle, styleSheet];

	SetOptions[nbObj, StyleDefinitions -> updatedStylesheet];
]





(*******************************************************

 SetStyleOptionsList

	in: xxx
	out: yyy

*******************************************************)

(* multiple cell style support *)
SetStyleOptionsList[nbObj_NotebookObject, mod:{{{style_String, ___}, ___}, optval:(Rule|RuleDelayed)[opt_, val_] }(*, stylesheet_*)]:=
	SetStyleOptionsList[nbObj, {style, optval}(*, stylesheet*)];

SetStyleOptionsList[nbObj_NotebookObject, mod:{style_String, optval:(Rule|RuleDelayed)[opt_, val_] }(*, stylesheet_*)]:=
Module[{styleUpdateList, value = optval, defaultValue = {}, defaultValueQ = False, currentValue},

	(* get notebook level tagging rules of applied style options *)
	styleUpdateList = PresenterTools`GetUserModifications[nbObj];
	If[Head[styleUpdateList] =!= List, styleUpdateList = {{style, optval }}];

	(* Remove any existing option values, then add updated option value *)
	styleUpdateList = ReplaceStyleOptionsList[styleUpdateList, {style, value}];

	(* add change to TaggingRules *)
	PresenterTools`Private`Logger[styleUpdateList];
	PresenterTools`SetStyleUpdateList[nbObj, styleUpdateList]
];
SetStyleOptionsList::argx = "`1`";
SetStyleOptionsList::info = "`1`";

SetStyleOptionsList[nbObj_NotebookObject, {style_String, val_ }(*, stylesheet_*)] :=
	Message[SetStyleOptionsList::argx, "Expected Rule found: " <> ToString[val] ];
SetStyleOptionsList[a__] :=
	Message[SetStyleOptionsList::argx, ToString[{a}]];





(*******************************************************

 GetUserCellStyleOptionValue

	in: User options list
	out: List

*******************************************************)

GetUserCellStyleOptionValue[userOptions_List, sty_, optionName_]:=
Module[{styleOptions},

	styleOptions = Cases[userOptions, (Rule | RuleDelayed)[optionName, r_] :> r, Infinity]

];
GetUserCellStyleOptionValue[a___]:= Message[GetUserCellStyleOptionValue::argx, {a}];
GetUserCellStyleOptionValue::argx = "`1`";
GetUserCellStyleOptionValue::info = "`1`";




(*******************************************************

 RemoveStyleOptionsList

	in: xxx
	out: yyy

*******************************************************)

RemoveStyleOptionsList[nbObj_NotebookObject, {sty_, opt_}]:=
	RemoveStyleOptionsList[PresenterTools`GetUserModifications[nbObj], {sty, opt}];

RemoveStyleOptionsList[list_List, {sty_, opt_}]:=
Module[{},

	DeleteCases[list, {sty, (Rule|RuleDelayed)[opt, _]}, Infinity]
]
RemoveStyleOptionsList[a___]:= Message[RemoveStyleOptionsList::argx, {a}];
RemoveStyleOptionsList::argx = "`1`";




(*******************************************************

 ReplaceStyleOptionsList

	in: xxx
	out: yyy

*******************************************************)

ReplaceStyleOptionsList[list_, {sty_, newOpt:(Rule|RuleDelayed)[opt_, val_]}]:=
Module[{newList},

	newList = RemoveStyleOptionsList[list, {sty, opt}];
	PrependTo[newList, {sty, newOpt}];

	newList
]
ReplaceStyleOptionsList[a___]:= Message[ReplaceStyleOptionsList::argx, {a}];
ReplaceStyleOptionsList::argx = "`1`";




(***************************************************************************************************************)

(* Hackish to deal with FontVariations *)
GetStyleOptionValue[stylesheet_, style_String, Rule[FontVariations, {"Underline"}], environment_String] :=
	GetStyleOptionValue[stylesheet, style, "FontVariationsUnderline", environment];
(*
GetStyleOptionValue[stylesheet_NotebookObject, style_String, opt_, environment_String] :=
GetStyleOptionValue[NotebookGet[stylesheet], style, opt, environment];
*)
GetStyleOptionValue[stylesheet_, style_String, opt_, environment_String] :=
Module[{nb, result, option = opt},

	nb =
		(*NotebookPut@*)
			Notebook[{},
				StyleDefinitions -> stylesheet,
				ScreenStyleEnvironment -> environment,
				WindowFrame -> "Frameless", WindowElements -> {},
				WindowFrameElements -> {"CloseBox"},
				ScreenStyleEnvironment -> "Slideshow Presentation",
				Visible -> False,
				WindowSize -> 10
			];
	PresenterTools`Private`Logger["GetStyleOptionValue:"];
	PresenterTools`Private`Logger[{style, option, environment}];
	(* Use InputForm to resolve FontColor boxes to RGBColor, Dynamics, etc. *)
	result = AbsoluteCurrentValue[nb, {StyleDefinitions, style, option}];
	PresenterTools`Private`Logger[result];
	(*NotebookClose[nbObj];*)

	result
];
GetStyleOptionValue::info = "`1`";






GetStyleDataCellMargins[stylesheet_, style_String, environment_String] :=
Module[{nbObj, result},
	nbObj = NotebookPut@
		Notebook[{}, StyleDefinitions -> stylesheet, ScreenStyleEnvironment -> environment, Visible -> False];
	result = {AbsoluteCurrentValue[nbObj, {StyleDefinitions, style, CellMargins}]};
	NotebookClose[nbObj];

	result
];


GetStyleDataCellMargins[stylesheet_, styles_List, environment_String] :=
Module[{nbObj, result},

	nbObj = NotebookPut@
		Notebook[{}, StyleDefinitions -> stylesheet, ScreenStyleEnvironment -> environment, Visible -> False];

	result = GetStyleDataCellMargins[stylesheet, styles, environment, nbObj];
	NotebookClose[nbObj];

	result
];

GetStyleDataCellMargins[stylesheet_, styles_List, environment_String, nbObj_NotebookObject] :=
Module[{results},

	results = Rule[#, AbsoluteCurrentValue[nbObj, {StyleDefinitions, #, CellMargins}]] & /@ styles;

	results
];




(*******************************************************

 InstallStylesheet

	in: xxx
	out: yyy

*******************************************************)

Options[InstallStylesheet] = {
  "StylesheetDirectory" :> PresenterTools`Styles`$UserStylesheetDirectory,
  "OverwriteStylesheet" -> False
}

InstallStylesheet[filename_String, nb_Notebook, opts:OptionsPattern[]]:=
Module[{name, dir},
  name = If[StringMatchQ[filename, "*.nb"], filename, filename <> ".nb"];

  dir = OptionValue["StylesheetDirectory"];
  If[!StringQ[dir] || FileType[dir] =!= Directory,
    Message[InstallStylesheet::nodir, dir];
    dir = PresenterTools`Styles`$UserStylesheetDirectory
  ];

  If[FileType[FileNameJoin[{dir, name}]] === File && Not @ TrueQ @ OptionValue["OverwriteStylesheet"],
    Message[InstallStylesheet::exists, filename, dir];
    $Failed,
  (* ELSE *)
    Export[FileNameJoin[{dir, name}], GetStylesheet[nb]]
  ]
]
InstallStylesheet::exists = "A stylesheet called `1` already exists in `2`.";
InstallStylesheet::nodir = "The directory `1` does not exist. Using $UserStylesheetDirectory instead.";
InstallStylesheet::argx = "`1`";
InstallStylesheet::info = "`1`";





(***************************************************************************

	STYLES

***************************************************************************)



(*******************************************************

 AddCellStyle
 	Add cell style from Cell expression

	in: Notebook, Style
	out: Cell

*******************************************************)

AddCellStyle[nbObj_NotebookObject, style_String]:=
Module[{cells, newCell},

	cells = SelectedCells[nbObj];

	Which[
		(* single cell *)
		Length[cells] === 1,
			If[PresenterTools`CellBracketSelectedQ[nbObj],
				newCell = AddCellStyle[cells, style];
				NotebookWrite[nbObj, newCell, All],
			(* ELSE *)
				PresenterTools`SelectCellBracket[nbObj];
				newCell = AddCellStyle[SelectedCells[nbObj], style];
				NotebookWrite[nbObj, newCell, All]
			],
		(* multiple cells selected *)
		1 < Length[cells],
			(
				newCell = AddCellStyle[#, style];
				NotebookWrite[nbObj, newCell, All]
			) & /@ cells,
		True,
			"no cells selected"
 ]
(*
	If[Length[cells] > 0,
		(
			newCell = AddCellStyle[#, style];
			NotebookWrite[nbObj, newCell, All]
		) & /@ cells
	]
*)
];

(* no existing style *)
AddCellStyle[Cell[con_, opts___?OptionQ], style_String]:=
	Cell[con, style, opts]

(* multiple existing styles *)
AddCellStyle[Cell[con_, styles__, opts___?OptionQ], style_String]:=
	Cell[con, styles, style, opts]

(* CellObject *)
AddCellStyle[cell_CellObject, style_String]:=
	AddCellStyle[NotebookRead[cell], style]

AddCellStyle::argx = "`1`";
AddCellStyle::info = "`1`";




(*******************************************************

 RemoveCellStyle
 	Remove cell style from Cell expression

	in: Cell, Style
	out: Cell

*******************************************************)

RemoveCellStyle[nbObj_NotebookObject, style_String]:=
Module[{cells, newCell},

	cells = SelectedCells[nbObj];

	If[Length[cells] > 0,
		(
			newCell = RemoveCellStyle[#, style];
			NotebookWrite[nbObj, newCell, All]
		) & /@ cells
	]

];
RemoveCellStyle::argx = "`1`";
RemoveCellStyle::info = "`1`";


(* no existing style *)
RemoveCellStyle[Cell[con_, opts___?OptionQ], style_String]:=
	Cell[con, opts]

(* multiple existing styles *)
RemoveCellStyle[Cell[con_, misc__, opts___?OptionQ], style_String]:=
	Cell[con, Sequence@@DeleteCases[{misc}, style], opts]

(* CellObject *)
RemoveCellStyle[cell_CellObject, style_String]:=
	RemoveCellStyle[NotebookRead[cell], style]





(*******************************************************

 DefaultOptionValueQ

	Is the given option value within NotebookSelection the
	default value or does it differ from the current stylesheet

	in: NotebookObject, cellstyle, cell option
	out: True | False

*******************************************************)

DefaultOptionValueQ[nbObj_, cellStyle_, opt_] :=
Module[{},
	If[CurrentValue[NotebookSelection[nbObj], {"SelectionHasUpdatedStyles", "Absolute"}],
		MemberQ[
			With[{cellObj = NotebookSelection[nbObj], style = cellStyle},
				FE`Evaluate[FEPrivate`CellStyleComplement[cellObj, FEPrivate`ResolveCellStyle[cellObj, style]]]
				], opt
		],
		(* ELSE *)
		True
	]
]
DefaultOptionValueQ[a___]:= Message[DefaultOptionValueQ::argx, {a}];
DefaultOptionValueQ::argx = "`1`";
DefaultOptionValueQ::info = "`1`";







(*
AddStyles::exists = "The style `1` already exists.";

Options[AddStyles] = {AddStylesToSection -> "New Styles"}


(*
   Should AddStyles be able to add styles in a particular category
   too, or just section?
*)

AddStyles[nb_, c:Cell[StyleData[sty_, env___], ___], opts___] :=
Block[{s, sec, pos},
  s = GetStylesheet[nb];
  sec = AddStylesToSection /. Flatten[{opts, Options[AddStyles]}];

  (* if the style already exists, do nothing *)
  If[MemberQ[GetStylesheetStyles[s], StyleData[sty, env]],
    Message[AddStyles::exists, StyleData[sty, env]];
    Return[s]
  ];

  (* if the style exists in other environments, put this with those *)
  pos = Position[s, Cell[StyleData[sty,___],___]];
  If[pos =!= {},
    Return @ Insert[s, c, MapAt[#+1&, Last @ pos, -1]]
  ];

  (* if there's no match, add it in the right section *)
  If[MemberQ[StylesheetSections[s], sec],
    s /. Cell[CellGroupData[{Cell[sec,"Section",a___], b___},open_]] :>
         Cell[CellGroupData[{Cell[sec, "Section", a], b, Cell[StyleData[sty]],c}, open]],
    Insert[s, Cell[CellGroupData[{Cell[sec, "Section"], c}, Closed]], {1,-1}]
  ]
]


AddStyles[nb_, {c__Cell}, opts___] := Fold[AddStyles[##, opts]&, nb, {c}]



RemoveStyles[nb_, str_String] := RemoveStyles[nb, StyleData[str]];

RemoveStyles[nb_, sty_StyleData] := RemoveStyles[nb, Cell[sty, ___]];

RemoveStyles[nb_, c:Cell[_StyleData,___]] := DeleteCases[GetStylesheet[nb], c, Infinity]

RemoveStyles[nb_, lis_List] := Fold[RemoveStyles, nb, lis]



RemoveEnvironments[nb_, env_String] := RemoveEnvironments[nb, {env}]

RemoveEnvironments[nb_, {envs__String}] := DeleteCases[
  GetStylesheet[nb],
  Cell[StyleData[_, Alternatives @@ {envs}], ___],
  Infinity
]

*)

GetStyleOptions[nb_, str_String, res___] :=
  GetStyleOptions[nb, StyleData[str], res]

GetStyleOptions[nb_, sty_StyleData, {opts__}] :=
  Cases[GetStyleOptions[nb, sty], _[Alternatives[opts],_]]

GetStyleOptions[nb_, sty_StyleData, opt_] :=
  GetStyleOptions[nb, sty, {opt}]

GetStyleOptions[nb_, sty_StyleData] :=
Block[{cells},
  cells = GetStylesheetCells[nb, sty];
  If[cells === {}, {}, List @@ Rest[First @ cells] ]
]




SetStyleOptions::nosty = "The style `1` does not exist in the given stylesheet; use AddStyles to add the style.";

SetStyleOptions::mult = "The style `1` is defined multiple times in the given stylesheet. Aborting attempt.";

SetStyleOptions[nb_, str_String, res___] :=
  SetStyleOptions[nb, StyleData[str], res]

SetStyleOptions[nb_, sty_StyleData, opts___] :=
Block[{s, pos, cell},

  s = GetStylesheet[nb];
  If[!MemberQ[GetStylesheetStyles[s], sty],
    Message[SetStyleOptions::nosty, sty];
    Return[s]
  ];

  pos = Position[s, Cell[sty,___]];
  If[Length[pos] =!= 1,
    Message[SetStyleOptions::mult, sty];
    Return[s]
  ];

  cell = s[[ ## ]]& @@ First[pos];
  cell = Fold[resetopt, cell, Flatten[{opts}]];
  ReplacePart[s, cell, pos]
]



(*
   Design point: we refer to styles by the "name": StyleData[sty] or
   StyleData[sty, env].
*)


GetStylesheetCells[nb_] := GetStylesheetCells[nb, StyleData[___]];

GetStylesheetCells[nb_, sty_String] := GetStylesheetCells[nb, StyleData[___, sty, ___]]

GetStylesheetCells[nb_, pat_StyleData] := GetStylesheetCells[nb, Cell[pat, ___]]

GetStylesheetCells[nb_, pat_] := Cases[GetStylesheet[nb], pat, Infinity]


GetStylesheetStyles[args___] := First /@ GetStylesheetCells[args]

GetStylesheetStyles[nb_] := First /@ GetStylesheetCells[nb]

GetStylesheetStyles[nb_, sty_String] := GetStylesheetStyles[nb, StyleData[___, sty, ___]]

GetStylesheetStyles[nb_, pat_StyleData] := Cases[First /@ GetStylesheetCells[nb], pat]



GetStylesheetEnvironments[nb_, res___] :=
  Cases[GetStylesheetStyles[nb, res], StyleData[_,x_] :> x] // DeleteDuplicates



GetStylesheetMenuStyles[nb_] := GetStylesheetCells[nb,
  Cell[StyleData[s_], opts___ /; !MemberQ[{opts}, StyleMenuListing -> None]] :> StyleData[s]]



resetopt[Cell[a___, _[opt_, _], b___], r:(_[opt_, _])] := Cell[a, r, b]
resetopt[Cell[a___], r_] := Cell[a, r]



ClearStyleOptions[nb_, sty_StyleData, opts___] :=
If[{opts} === {All},
  GetStylesheet[nb] //. Cell[sty,___] :> Sequence[],
  GetStylesheet[nb] //.
    Cell[sty,x___, _[Alternatives[opts], _], y___] :> Cell[sty, x, y]
]









End[] (* End Private Context *)

EndPackage[]
