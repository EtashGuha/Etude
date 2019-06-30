(* ::Package:: *)

(* ::Title:: *)
(*Keyboard*)

(* ::Section:: *)
(*Annotations*)

(* :Title: Keyboard.m *)

(* :Authors:
        Brenton Bostick
        brenton@wolfram.com
        
        Arnoud Buzing
        arnoudb@wolfram.com
        
        Arnoud did most of the ToKeys definitions for boxes.
*)

(* :Package Version: 1.2 *)

(* :Mathematica Version: 7.0 *)

(* :Copyright: RobotTools source code (c) 2005-2008 Wolfram Research, Inc. All rights reserved. *)

(* :Discussion:
   Implementation of Keyboard-related functionality.
   KeyPress and KeyRelease can only take a single character. They are lower-level functions.
*)

(* ::Section:: *)
(*Information*)

`Information`CVS`$KeyboardId = "$Id: Keyboard.m,v 1.59 2008/07/08 14:03:52 brenton Exp $"

(* ::Section:: *)
(*Public*)

KeyboardForm::usage =
"KeyboardForm is deprecated. Use ToKeys."

KeyPress::usage =
"KeyPress[\"k\"] presses key k."

KeyRelease::usage =
"KeyRelease[\"k\"] releases key k."

KeyType::usage =
"KeyType[\"s\"] types the given string.
KeyType[nb, \"s\"] selects nb first."

Modifiers::usage =
"Modifiers is an option to keyboard, menu, and mouse functions."

ToKeys::usage =
"ToKeys[input] gives the keys necessary for typing the given box or string."

$InitialKeyTypeDelay::usage =
"$InitialKeyTypeDelay is the delay before typing is started."

$InterfaceLanguage::usage =
"$InterfaceLanguage is the language of the current interface, which is a notebook front end if connected or a stand-alone kernel."

$KeyPressDelay::usage =
"$KeyPressDelay is the delay before a key is pressed."

$KeyReleaseDelay::usage =
"$KeyReleaseDelay is the delay before a key is released."

$KeyTypeDelay::usage =
"$KeyTypeDelay is the delay before a key is typed."

$RotateModifiers::usage =
"$RotateModifiers is the list of modifiers for rotating 3D graphics."

$PanModifiers::usage =
"$PanModifiers is the list of modifiers for panning 3D graphics."

$ZoomModifiers::usage =
"$ZoomModifiers is the list of modifiers for zooming 3D graphics."

(* ::Section:: *)
(*Package*)

Begin["`Package`"]

characters

foldKeyModifiers

keyPress

keyRelease

keyType

sowKeyModifiers

sowKeyModifiersString

validateModifiers

$mkSymbolToStringRules

$mkStringToSymbolRules

End[] (*`Package`*)

(* ::Section:: *)
(*Private*)

(* ::Subsection:: *)
(*Begin*)

Begin["`Keyboard`Private`"]

(* ::Subsection:: *)
(*Messages*)

KeyPress::badchar =
"The given key cannot be pressed: `1`."

KeyRelease::badchar =
"The given key cannot be released: `1`."

KeyPress::multk = KeyRelease::multk =
"Multiple keys were specified, only one key should be specified: `1`."

KeyPress::mcp = KeyRelease::mcp =
"Meta characters cannot be typed: `1`."

KeyType::badkeys =
"The given keys cannot be typed: `1`."

(* ::Subsection:: *)
(*ToKeys*)

(* ToKeys needs to be idempotent, that is ToKeys[ToKeys[foo]] needs to be equal to ToKeys[foo] *)

(* TODO: support options for all boxes, not just strings *)

Options[ToKeys] = {Modifiers -> {}}

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], FractionBox[a_, b_, opts:OptionsPattern[]]] :=
	ToKeys[nb, FractionBox[a, b, opts]] =
	StringJoin[ToKeys /@ {
		"\[Placeholder]",
		"\[ShiftKey]\[LeftModified]\[TabKey]\[RightModified]",
		ItemKeys[nb, {"Insert", "Typesetting", "Fraction"}],
		ToKeys[nb, a], "\[TabKey]",
		ToKeys[nb, b],
		"\\[RightKey]"}]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], SqrtBox[a_, opts:OptionsPattern[]]] :=
	ToKeys[nb, SqrtBox[a, opts]] =
	StringJoin[ToKeys /@ {
		ItemKeys[nb, {"Insert", "Typesetting", "Radical"}],
		ToKeys[nb, a],
		"\\[RightKey]"}]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], RadicalBox[a_, b_, opts:OptionsPattern[]]] :=
	ToKeys[nb, RadicalBox[a, b, opts]] =
	StringJoin[ToKeys /@ {
		ItemKeys[nb, {"Insert", "Typesetting", "Radical"}],
		ToKeys[nb, a],
		ItemKeys[nb, {"Insert", "Typesetting", "Opposite Position"}],
		ToKeys[nb, b],
		"\\[RightKey]"}]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], SuperscriptBox[a_, b_, opts:OptionsPattern[]]] :=
	ToKeys[nb, SuperscriptBox[a, b, opts]] =
	StringJoin[ToKeys /@ {
		ToKeys[nb, a],
		ItemKeys[nb, {"Insert", "Typesetting", "Superscript"}],
		ToKeys[nb, b],
		"\\[RightKey]"}]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], SubscriptBox[a_, b_, opts:OptionsPattern[]]] :=
	ToKeys[nb, SubscriptBox[a, b, opts]] =
	StringJoin[ToKeys /@ {
		ToKeys[nb, a],
		ItemKeys[nb, {"Insert", "Typesetting", "Subscript"}],
		ToKeys[nb, b],
		"\\[RightKey]"}]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], UnderscriptBox[a_, b_, opts:OptionsPattern[]]] :=
	ToKeys[nb, UnderscriptBox[a, b, opts]] =
	StringJoin[ToKeys /@ {
		ToKeys[nb, a],
		ItemKeys[nb, {"Insert", "Typesetting", "Below"}],
		ToKeys[nb, b],
		"\\[RightKey]"}]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], OverscriptBox[a_, b_, opts:OptionsPattern[]]] :=
	ToKeys[nb, OverscriptBox[a, b, opts]] =
	StringJoin[ToKeys /@ {
		ToKeys[nb, a],
		ItemKeys[nb, {"Insert", "Typesetting", "Above"}],
		ToKeys[nb, b], "\\[RightKey]"}]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], SubsuperscriptBox[a_, b_, c_, opts:OptionsPattern[]]] :=
	ToKeys[nb, SubsuperscriptBox[a, b, c, opts]] =
	StringJoin[ToKeys /@ {
		ToKeys[nb, a],
		ItemKeys[nb, {"Insert", "Typesetting", "Subscript"}],
		ToKeys[nb, b],
		ItemKeys[nb, {"Insert", "Typesetting", "Opposite Position"}],
		ToKeys[nb, c],
		"\\[RightKey]"}]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], UnderoverscriptBox[a_, b_, c_, opts:OptionsPattern[]]] :=
	ToKeys[nb, UnderoverscriptBox[a, b, c, opts]] =
	StringJoin[ToKeys /@ {
		ToKeys[nb, a],
		ItemKeys[nb, {"Insert", "Typesetting", "Below"}],
		ToKeys[nb, b],
		ItemKeys[nb, {"Insert", "Typesetting", "Opposite Position"}],
		ToKeys[nb, c],
		"\\[RightKey]"}]

gridIndexFunction =
	Riffle[ToKeys /@ #1, If[First[#2] == 1, ItemKeys[{"Insert", "Table/Matrix", "Add Column"}], "\[TabKey]"]]&

gridKeys[grid_] :=
	Riffle[MapIndexed[gridIndexFunction, grid], ItemKeys[{"Insert", "Table/Matrix", "Add Row"}]]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], GridBox[grid:gridPat, opts:OptionsPattern[]]] :=
	ToKeys[nb, GridBox[grid, opts]] =
	StringJoin[ToKeys /@ {
		gridKeys[grid],
		"\\[RightKey]"}]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], GridBox[{{a_}}, opts:OptionsPattern[]]] :=
	ToKeys[nb, GridBox[{{a}}, opts]] =
	StringJoin[ToKeys /@ {ToKeys[nb, a]}]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], Cell[a_, opts:OptionsPattern[]]] :=
	ToKeys[nb, Cell[a, opts]] =
	StringJoin[ToKeys /@ {
		ItemKeys[nb, {"Insert", "Typesetting", "Start Inline Cell"}],
		ToKeys[nb, a],
		"\\[RightKey]"}]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], Cell[a_, "Input", opts:OptionsPattern[]]] :=
	ToKeys[nb, Cell[a, "Input", opts]] =
	(
		theCellStyle = "Input";
		StringJoin[ToKeys /@ {ToKeys[nb, a], "\\[RightKey]"}]
	)

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], Cell[a_, "Text", opts:OptionsPattern[]]] :=
	ToKeys[nb, Cell[a, "Text", opts]] =
	(
		theCellStyle = "Text";
		StringJoin[ToKeys /@ {ItemKeys[nb, {"Format", "Style", "Text"}], " \\[LeftKey]", "\[DeleteKey]",
			ToKeys[nb, a]}]
	)

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], Cell[a_, style_String, opts:OptionsPattern[]]] :=
	ToKeys[nb, Cell[a, style, opts]] =
	(
		theCellStyle = style;
		StringJoin[ToKeys /@ {ItemKeys[nb, {"Format", "Style", "Other..."}], ToKeys[nb, style],
			"\[EnterKey]", "\\[LeftKey]", "\[DeleteKey]", ToKeys[nb, a]}]
	)

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], BoxData[a_]] :=
	ToKeys[nb, BoxData[a]] =
	StringJoin[ToKeys /@ {ToKeys[nb, a]}]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], FormBox[a_, TraditionalForm]] :=
	ToKeys[nb, FormBox[a, TraditionalForm]] =
	StringJoin[ToKeys /@ {ToKeys[nb, a]}]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], FormBox[a_, "TraditionalForm"]] :=
	ToKeys[nb, FormBox[a, "TraditionalForm"]] =
	StringJoin[ToKeys /@ {ToKeys[nb, a]}]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], FormBox[a_, TextForm]] :=
	ToKeys[nb, FormBox[a, TextForm]] =
	StringJoin[ToKeys /@ {ToKeys[nb, a]}]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], TagBox[a_, b_, opts:OptionsPattern[]]] :=
	ToKeys[nb, TagBox[a, b, opts]] =
	StringJoin[ToKeys /@ {ToKeys[nb, a]}]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], TagBox[a_, opts:OptionsPattern[]]] :=
	ToKeys[nb, TagBox[a, opts]] =
	StringJoin[ToKeys /@ {ToKeys[nb, a]}]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], InterpretationBox[boxexpr_, expr_]] :=
	ToKeys[nb, InterpretationBox[boxexpr, expr]] =
	StringJoin[ToKeys /@ {ToKeys[nb, boxexpr]}]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], RowBox[a_List]] :=
	ToKeys[nb, RowBox[a]] =
	StringJoin[ToKeys[nb, #]& /@ a]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], TextData[a_List]] :=
	ToKeys[nb, TextData[a]] =
	StringJoin[ToKeys[nb, #]& /@ a]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], TextData[a_]] :=
	ToKeys[nb, TextData[a]] =
	StringJoin[ToKeys /@ {ToKeys[nb, a]}]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], CellGroupData[a_List, Open]] :=
	ToKeys[nb, CellGroupData[a, Open]] =
	StringJoin[ToKeys /@ {ToKeys[nb, #] <> "\\[DownKey]"& /@ a}]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], StyleBox[a_, style_String, opts:OptionsPattern[]]] :=
	ToKeys[nb, StyleBox[a, style, opts]] =
	StringJoin[ToKeys /@ {
		ItemKeys[nb, {"Format", "Style", "Other..."}],
		ToKeys[nb, style],
		"\[EnterKey]",
		ToKeys[nb, a],
		ItemKeys[nb, {"Format", "Style", "Other..."}],
		theCellStyle,
		"\[EnterKey]"}]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], StyleBox[a_String, FontSlant->"Italic"]] :=
	ToKeys[nb, StyleBox[a, FontSlant->"Italic"]] =
	StringJoin[ToKeys /@ {
		ItemKeys[nb, {"Format", "Face", "Italic"}],
		ToKeys[nb, a],
		ItemKeys[nb, {"Format", "Face", "Italic"}]}]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], StyleBox[a_String, FontWeight->"Bold"]] :=
	ToKeys[nb, StyleBox[a, FontWeight->"Bold"]] =
	StringJoin[ToKeys /@ {
		ItemKeys[nb, {"Format", "Face", "Bold"}],
		ToKeys[nb, a],
		ItemKeys[nb, {"Format", "Face", "Bold"}]}]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], StyleBox[a_, opts:OptionsPattern[]]] :=
	ToKeys[nb, StyleBox[a, opts]] =
	StringJoin[ToKeys /@ {ToKeys[nb, a]}]

(*
TODO: Fix reference to menu item "Expression..."
ToKeys[nb:focusedNotebookPat:MenuNotebook[], CounterBox[style_]] :=
	ToKeys[menuNotebook, CounterBox[style]] =
	StringJoin[ToKeys /@ {ItemKeys[menuNotebook, "Expression..."], "CounterBox[\"" <> ToKeys[menuNotebook, style] <> "\"]",
		"\[TabKey]", "\[EnterKey]"}]*)

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], b_ButtonBox] :=
	ToKeys[nb, b] =
	StringJoin[ToKeys /@ {"[button]"}]

$InterfaceLanguage =
	If[$Notebooks, Internal`CachedSystemInformation["FrontEnd", "Language"], $Language]

$RobotToolsKeyboardTextResourcesDirectory =
	FileNameJoin[{$RobotToolsTextResourcesDirectory, $InterfaceLanguage, "Keyboard"}]

$RobotToolsEnglishKeyboardTextResourcesDirectory =
	FileNameJoin[{$RobotToolsTextResourcesDirectory, "English", "Keyboard"}]

$virtualKeys =
	Module[{commonFileName, platformFileName},
		If[ValueQ[$InterfaceEnvironment],
			commonFileName = FileNameJoin[{$RobotToolsKeyboardTextResourcesDirectory, "Common", "VirtualKeys.m"}];
			platformFileName = FileNameJoin[{$RobotToolsKeyboardTextResourcesDirectory, $InterfaceEnvironment, "VirtualKeys.m"}];
			Join[Get[commonFileName], Get[platformFileName]]
			,
			{}
		]
	]

(* $englishVirtualKeys is defined as {} *)

$simpleKeys =
	Module[{commonFileName, platformFileName},
		If[ValueQ[$InterfaceEnvironment],
			commonFileName = FileNameJoin[{$RobotToolsKeyboardTextResourcesDirectory, "Common", "SimpleKeys.m"}];
			platformFileName = FileNameJoin[{$RobotToolsKeyboardTextResourcesDirectory, $InterfaceEnvironment, "SimpleKeys.m"}];
			Join[Get[commonFileName], Get[platformFileName]]
			,
			{}
		]
	]

$englishSimpleKeys =
	Module[{commonFileName, platformFileName},
		If[ValueQ[$InterfaceEnvironment],
			commonFileName = FileNameJoin[{$RobotToolsEnglishKeyboardTextResourcesDirectory, "Common", "SimpleKeys.m"}];
			platformFileName = FileNameJoin[{$RobotToolsEnglishKeyboardTextResourcesDirectory, $InterfaceEnvironment, "SimpleKeys.m"}];
			Join[Get[commonFileName], Get[platformFileName]]
			,
			{}
		]
	]

$keyTranslations =
	Module[{commonFileName, platformFileName},
		If[ValueQ[$InterfaceEnvironment],
			commonFileName = FileNameJoin[{$RobotToolsKeyboardTextResourcesDirectory, "Common", "KeyTranslations.m"}];
			platformFileName = FileNameJoin[{$RobotToolsKeyboardTextResourcesDirectory, $InterfaceEnvironment, "KeyTranslations.m"}];
			Join[Get[commonFileName], Get[platformFileName]]
			,
			{}
		]
	]

$englishKeyTranslations =
	Module[{commonFileName, platformFileName},
		If[ValueQ[$InterfaceEnvironment],
			commonFileName = FileNameJoin[{$RobotToolsEnglishKeyboardTextResourcesDirectory, "Common", "KeyTranslations.m"}];
			platformFileName = FileNameJoin[{$RobotToolsEnglishKeyboardTextResourcesDirectory, $InterfaceEnvironment, "KeyTranslations.m"}];
			Join[Get[commonFileName], Get[platformFileName]]
			,
			{}
		]
	]

$metaChars =
	{"\[LeftModified]", "\[RightModified]", "\[KeyBar]"}

(* specialChars is a list of characters that should not be typed using long names or short names *)
$specialChars =
	Join[$keyTranslations[[All, 1]], $simpleKeys, $metaChars]

characters[s:stringPat] :=
	DeleteCases[StringSplit[s, {p:$specialChars :> p, ""}], ""]

(*
convertChar converts any single Mathematica character into whatever keyboard characters are needed to enter it,
using escape names, long names, aliases, or hex names
*)
convertChar[c:stringPat] :=
	Module[{input, aliases, longName},
		Which[
			MemberQ[$specialChars, c],
			(* c is a special character that is ignored *)
			c
			,
			input = ToString[c, InputForm, CharacterEncoding -> "PrintableASCII"];
			StringMatchQ[input, "\"\\" ~~ _ ~~ "\""],
			(*
			c is an escaped character like \t or \!
			c may very well have long names and short names, but this special group of characters gets typed out with a leading \
			*)
			StringTake[input, {2, 3}]
			,
			aliases = characterData[c, "Aliases"];
			aliases != {},
			(* c has front end aliases, just use the first one *)
			"\[EscapeKey]" <> First[aliases] <> "\[EscapeKey]"
			,
			longName = characterData[c, "LongName"];
			longName != Missing["NotApplicable"],
			(* c has a long name *)
			longName
			,
			True,
			(* c can only be entered as a Unicode key combination *)
			input
		]
	]

(*
TODO: do optimize correctly

RegularExpression["(\[ShiftKey]\[LeftModified].\[RightModified])+"] only finds a single character between the modifieds, but
should also be able to find \\[] fake keys
*)
optimize[s_String] :=
	StringJoin @ Map[
		If[StringMatchQ[#, RegularExpression["(\[ShiftKey]\[LeftModified].\[RightModified])+"]], "\[ShiftKey]\[LeftModified]" <> StringTake[#, {3, -1, 4}] <> "\[RightModified]", #]&
		,
		StringCases[s, {RegularExpression["(\[ShiftKey]\[LeftModified].\[RightModified])+"], RegularExpression["(.)"] :> "$0"}]
	]

ToKeys[nb:focusedNotebookPat:FocusedNotebook[], s:stringPat, OptionsPattern[]] :=
	ToKeys[nb, s] =
	Module[{modifiers, vModifiers, buffer},
		Catch[
			(* step 1: set variables and validate user input *)
			throwIfMessage[
				modifiers = OptionValue[Modifiers];
				vModifiers = validateModifiers[ToKeys, modifiers]
			];
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							sowKeyModifiers[vModifiers,
								Sow[toKeys[s, {}]]
							]
						]
				]
			];
			(* step 3: execute the buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer]
			]
		]
	]

SetAttributes[ToKeys, ReadProtected]

ToKeys[args___] :=
	(ArgumentCountQ[ToKeys, System`FEDump`NonOptionArgCount[{args}], 1, 2]; $Failed)

SyntaxInformation[ToKeys] = {"ArgumentsPattern" -> {_., _}}

toKeys[s:stringPat, modifiers:keyModifierListPat] :=
	Module[{ss},
		ss = optimize[StringJoin[(characters /@ convertChar /@ characters[s]) /. $keyTranslations]];
		foldKeyModifiers[ss, modifiers]
	]

KeyboardForm :=
	(Message[KeyboardForm::usage];ToKeys)

(* ::Subsection:: *)
(*$PanModifiers*)

$PanModifiers =
	Switch[$InterfaceEnvironment,
		"Macintosh",
		{"\[ShiftKey]"}
		,
		"Windows",
		{"\[ShiftKey]"}
		,
		"X",
		{"\[ShiftKey]"}
		,
		(*HoldPattern[$InterfaceEnvironment]*)
		_,
		{}
	]

(* ::Subsection:: *)
(*$RotateModifiers*)

$RotateModifiers =
	Switch[$InterfaceEnvironment,
		"Macintosh",
		{}
		,
		"Windows",
		{}
		,
		"X",
		{}
		,
		(*HoldPattern[$InterfaceEnvironment]*)
		_,
		{}
	]

(* ::Subsection:: *)
(*$ZoomModifiers*)

$ZoomModifiers :=
	Switch[$InterfaceEnvironment,
		"Macintosh",
		{"\[CommandKey]", "\[OptionKey]"}
		,
		"Windows",
		{"\[ControlKey]", "\[AltKey]"}
		,
		"X",
		{"\[ControlKey]", "\[AltKey]"}
		,
		(*HoldPattern[$InterfaceEnvironment]*)
		_,
		{}
	]

(* ::Subsection:: *)
(*$KeyPressDelay*)

$KeyPressDelay = 0

(* ::Subsection:: *)
(*$KeyReleaseDelay*)

$KeyReleaseDelay = 0

(* ::Subsection:: *)
(*$InitialKeyTypeDelay*)

$InitialKeyTypeDelay = 1

(* ::Subsection:: *)
(*$KeyTypeDelay*)

$KeyTypeDelay = 0

(* ::Subsection:: *)
(*sowKeyModifiersString*)

Attributes[sowKeyModifiersString] = {HoldRest}

sowKeyModifiersString[modifiers:keyModifierListPat, expr_] :=
	(
		Sow /@ (# <> "\[LeftModified]"&) /@ modifiers;
		expr;
		Sow /@ ("\[RightModified]"&) /@ modifiers
	)

(* ::Subsection:: *)
(*wrapKeyModifiers*)

wrapKeyModifiers =
	#2 <> "\[LeftModified]" <> #1 <> "\[RightModified]"&

(* ::Subsection:: *)
(*sowKeyModifiers*)

Attributes[sowKeyModifiers] = {HoldRest}

sowKeyModifiers[modifiers:keyModifierListPat, expr_] :=
	(
		Sow /@ keyPress /@ modifiers;
		expr;
		Sow /@ keyRelease /@ Reverse[modifiers]
	)

(* ::Subsection:: *)
(*foldKeyModifiers*)

foldKeyModifiers[s:stringPat, modifiers:keyModifierListPat] :=
	Fold[wrapKeyModifiers, s, modifiers]

(* ::Subsection:: *)
(*KeyPress*)

Unprotect[KeyPress]

KeyPress[nb:focusedNotebookPat:FocusedNotebook[], c:stringPat] :=
	Module[{vChar, buffer},
		Catch[
			(* step 1: set variables and validate user input *)
			throwIfMessage[
				vChar = validateSimpleKey[KeyPress, c]
			];
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							Sow[keyPress[vChar]]
						]
				]
			];
			(* step 3: execute buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer]
			]
		]
	]

SetAttributes[KeyPress, ReadProtected]

KeyPress[arg_] :=
	(Message[KeyPress::string, 1, HoldForm[KeyPress[arg]]]; $Failed)

KeyPress[args___] :=
	(ArgumentCountQ[KeyPress, System`FEDump`NonOptionArgCount[{args}], 1, 2]; $Failed)

SyntaxInformation[KeyPress] = {"ArgumentsPattern" -> {_., _}}

Protect[KeyPress]

(* ::Subsection:: *)
(*KeyRelease*)

Unprotect[KeyRelease]

KeyRelease[nb:focusedNotebookPat:FocusedNotebook[], c:stringPat] :=
	Module[{vChar, buffer},
		Catch[
			(* step 1: set variables and validate user input *)
			throwIfMessage[
				vChar = validateSimpleKey[KeyRelease, c]
			];
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							Sow[keyRelease[vChar]]
						]
				]
			];
			(* step 3: execute buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer]
			]
		]
	]

SetAttributes[KeyRelease, ReadProtected]

KeyRelease[arg_] :=
	(Message[KeyRelease::string, 1, HoldForm[KeyRelease[arg]]]; $Failed)

KeyRelease[args___] :=
	(ArgumentCountQ[KeyRelease, System`FEDump`NonOptionArgCount[{args}], 1, 2]; $Failed)

SyntaxInformation[KeyRelease] = {"ArgumentsPattern" -> {_., _}}

Protect[KeyRelease]

(* ::Subsection:: *)
(*KeyType*)

$mkSymbolToStringRules =
	{
		"Command" -> If[$InterfaceEnvironment == "Macintosh", "\[CommandKey]", "\[AltKey]", ""],
		"Shift" -> "\[ShiftKey]",
		"Control" -> "\[ControlKey]",
		"Option" -> If[$InterfaceEnvironment == "Macintosh", "\[OptionKey]", "\[AltKey]", ""]
	}

$mkStringToSymbolRules =
	{
		(* "\[AltKey]" could technically go to either Command or Option, arbitrarily choose Command *)
		"\[AltKey]" -> "Command",
		"\[CommandKey]" -> "Command",
		"\[ControlKey]" -> "Control",
		"\[OptionKey]" -> "Option",
		"\[ShiftKey]" -> "Shift"
	}

keyBarSequence[keys___] :=
	foldKeyModifiers[Last[{keys}], Reverse[Take[{keys}, {1, -3, 2}]]]

(* keyType is referentially transparent *)
keyType[s:stringPat] :=
	Module[{keys},
		keys = characters[toKeys[s, {}]];
		(* convert key bar *)
		keys =
			ReplaceRepeated[keys,
				{a___, b:Longest[PatternSequence[PatternSequence[keyModifierPat, "\[KeyBar]"].., _]], c___} :>
					{a, Sequence @@ keyType[keyBarSequence[b]], c}
			];
		(* convert modifier keys *)
		keys =
			ReplaceRepeated[keys,
				{a___, modifier:keyModifierPat, "\[LeftModified]", b:Except["\[LeftModified]"]..., "\[RightModified]", c___} :>
					{a, delay[$KeyPressDelay], keyPress[modifier], b, delay[$KeyReleaseDelay], keyRelease[modifier], c}
			];
		(* convert everything else *)
		keys = Replace[keys, b:stringPat :> Sequence @@ {delay[$KeyTypeDelay], keyPress[b], keyRelease[b]}, {1}];
		keys
	]

Unprotect[KeyType]

Options[KeyType] = {Modifiers -> {}}

KeyType[nb:focusedNotebookPat:FocusedNotebook[], s:stringPat, i:wholePat:1, opts:OptionsPattern[]] :=
	KeyType[nb, {{s, i}}, opts]

KeyType[nb:focusedNotebookPat:FocusedNotebook[], is:iteratedStringPat, opts:OptionsPattern[]] :=
	KeyType[nb, {is}, opts]

KeyType[nb:focusedNotebookPat:FocusedNotebook[], cmds:keyCmdsPat, OptionsPattern[]] :=
	Module[{modifiers, vKeyboardCommands, vModifiers, buffer},
		Catch[
			(* step 1: set variables and validate user input *)
			throwIfMessage[
				modifiers = OptionValue[Modifiers];
				vKeyboardCommands = validateKeyCommands[KeyType, cmds];
				vModifiers = validateModifiers[KeyType, modifiers]
			];
			(* step 2: construct buffer *)
			throwIfMessage[
				RobotBlock[
					buffer =
						reapHeldList[
							Sow[delay[$InitialKeyTypeDelay]];
							sowKeyModifiers[vModifiers,
								Scan[
									Do[Sow /@ keyType[#[[1]]], {#[[2]]}]&
									,
									vKeyboardCommands
								]
							]
						]
				]
			];
			(* step 3: execute buffer *)
			throwIfMessage[
				RobotExecute[nb, buffer]
			]
		]
	]

SetAttributes[KeyType, ReadProtected]

KeyType[args___] :=
	(ArgumentCountQ[KeyType, System`FEDump`NonOptionArgCount[{args}], 1, 2]; $Failed)

SyntaxInformation[KeyType] = {"ArgumentsPattern" -> {_., _}}

Protect[KeyType]

(* ::Subsection:: *)
(*Low-Level Functions*)

keyPress[k:stringPat] :=
	(
		delay[$KeyPressDelay];
		$Robot@keyPress[k /. $virtualKeys /. $javaKeys]
	)

keyRelease[k:stringPat] :=
	(
		delay[$KeyReleaseDelay];
		$Robot@keyRelease[k /. $virtualKeys /. $javaKeys]
	)

(* ::Subsection:: *)
(*Validation*)

validateModifiers[head:symbolPat, modifiers_] :=
	Which[
		MatchQ[modifiers, modifierListPat],
		modifiers /. $mkSymbolToStringRules
		,
		True,
		Message[head::optvg, Modifiers, modifiers, "a list of modifier keys"];
		$Failed
	]

validateSimpleKey[head:symbolPat, c:stringPat] :=
	Which[
		(* the Verbatim is needed so that "*" doesn't make it return True all the time *)
		StringMatchQ[c, Verbatim[$simpleKeys]],
		c
		,
		StringMatchQ[c, Verbatim[$simpleKeys]..],
		Message[head::multk, c];
		$Failed
		,
		StringMatchQ[c, metaCharPat],
		Message[head::mcp, c];
		$Failed
		,
		True,
		Message[head::badchar, c];
		$Failed
	]

(*
checks that modified chars are balanced and there is a modifier to the left of \[LeftModified]
TODO: put in a check for only simple keys inside of \[LeftModified] and \[RightModified].
"\[ShiftKey]\[LeftModified]1+1\[RightModified]" should give an error because the + requires holding down Shift again, and so the last
1 is mistakenly typed as 1 instead of !. Stupid window managers and their lack of modifier key stacks...
*)
validateMetaChars[head:symbolPat, s:stringPat] :=
	Module[{keys},
		keys = characters[s];
		(* to check the balance of "\[LeftModified]" and "\[RightModified]" *)
		keys =
			ReplaceRepeated[keys,
				{a___, keyModifierPat, "\[LeftModified]", b:Except["\[LeftModified]"]..., "\[RightModified]", c___} :>
					{a, b, c}
			];
		(* to check \[KeyBar] *)
		keys =
			ReplaceRepeated[keys,
				{a___, keyModifierPat, "\[KeyBar]", b_, c___} :>
					{a, b, c}
			];
		Which[
			Cases[keys, metaCharPat] == {},
			s
			,
			True,
			Message[head::badkeys, s];
			$Failed
		]
	]

validateKeyCommands[head:symbolPat, cmds:keyCmdsPat] :=
	Module[{flag = False},
		Scan[
			Switch[#,
				stringPat,
				validateMetaChars[head, #]
				,
				iteratedStringPat,
				validateMetaChars[head, #[[1]]]
				,
				_,
				flag = True
			]&
			,
			cmds
		];
		If[flag,
			$Failed
			,
			cmds
		]
	]

(* ::Subsection:: *)
(*End*)

End[] (*`Keyboard`Private`*)
