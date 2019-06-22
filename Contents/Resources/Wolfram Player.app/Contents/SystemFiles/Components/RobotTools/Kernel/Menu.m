(* ::Package:: *)

(* ::Title:: *)
(*Menu*)


(* ::Section:: *)
(*Annotations*)


(* :Title: Menu.m *)

(* :Author:
        Brenton Bostick
        brenton@wolfram.com
*)

(* :Package Version: 1.2 *)

(* :Mathematica Version: 7.0 *)

(* :Copyright: RobotTools source code (c) 2005-2008 Wolfram Research, Inc. All rights reserved. *)

(* :Discussion:
   Implementation of menu-related functionality.
*)



(* ::Section:: *)
(*Information*)


`Information`CVS`$MenuId = "$Id: Menu.m,v 1.43 2015/01/26 21:55:48 carlosy Exp $"



(* ::Section:: *)
(*FrontEnd*)


(*
The FrontEnd` symbols below are from MenuSetup.tr and ContextMenus.tr and could be used in the FrontEnd`AddMenuCommands packet.
There are a lot of other symbols in MenuSetup.tr that I don't bring in since they are never used any where else.
These symbols can be considered public, and if MenuSetup.tr ever qualifies them with a context,
this whole section can go away (and old uses of these symbols will have to be updated with the new context).
*)

FrontEnd`AlternateItems
FrontEnd`HelpMenu
FrontEnd`KernelExecute
FrontEnd`LinkedItems
FrontEnd`MenuAnchor
FrontEnd`MenuEvaluator
FrontEnd`MenuKey
FrontEnd`Modifiers
FrontEnd`RawInputForm
FrontEnd`Scope
FrontEnd`ToggleMenuItem
FrontEnd`ToggleOptionListElement



(* ::Section:: *)
(*Public*)


ItemActivate::usage =
"ItemActivate[{path1, path2, ...}] activates the fully specified menu item in the front end.
ItemActivate[nb, item] makes sure that nb is selected, and then activates item."

ItemKeys::usage =
"ItemKeys[{path1, path2, ...}] returns a string of keys for activating the fully specified menu item in the front end.
ItemKeys[nb, item] returns keys assuming nb is selected."

(* $MenuLanguage is a run-time thing *)
$MenuLanguage::usage =
"$MenuLanguage is a variable that can be set to make RobotTools aware of what language is being used in menu functions."


(* ::Section:: *)
(*Package*)


Begin["`Package`"]

iaMouse

End[] (*`Package`*)



(* ::Section:: *)
(*Private*)


(* ::Subsection:: *)
(*Begin*)


Begin["`Menu`Private`"]



(* ::Subsection:: *)
(*Messages*)


GetPopupList::anchor =
"`1` is not a valid menu anchor."

ItemActivate::badnbobj = ItemKeys::badnbobj =
"An invalid NotebookObject was given: `1`."

ItemActivate::noitem = ItemKeys::noitem =
"The menu item you are trying to access does not exist: `1`."

Menu::syntax =
"The MenuSetup resource has a syntax error. Please check MenuSetup.tr for any extra commas, etc., and fix the error."



(* ::Subsection:: *)
(*Menu and MenuItem*)


Unprotect[Menu, MenuItem]

(* for the Macintosh menu *)
Menu["\.14", args___] :=
	Menu["Mathematica", args]

(* for the Windows and X menus, which use \t inside item names as a hack *)
MenuItem[tabbedName:_String /; StringMatchQ[tabbedName, __~~ "\t" ~~__], args__] :=
	Module[{name, keyString, keys},
		{name, keyString} = tabSplit[tabbedName];
		keys = plusSplit[keyString] /. $frontEndModifierKeys /. $mkSymbolToStringRules;
		(* because MenuItem is HoldRest, we use With to insert the arguments *)
		With[{name = name, menuKey = FrontEnd`MenuKey[Last[keys], FrontEnd`Modifiers -> Reverse[Most[keys]]]},
			MenuItem[name, args, menuKey]
		]
	]



(* ::Subsection:: *)
(*blockMenu*)


Attributes[blockMenu] = {HoldRest}

blockMenu[nb:focusedNotebookPat, expr_] :=
	Module[{$menu},
		$menu := menuFactory[nb];
		Block[{$Menu, MenuItem},
			Attributes[MenuItem] = {HoldRest};
			MenuItem[name_String, command_String, FrontEnd`MenuAnchor -> True] :=
				MenuItem[name, command, FrontEnd`MenuAnchor -> True] =
				Sequence @@ GetPopupList[FocusedNotebook[], command];
			$Menu :=
				$Menu =
				$menu;
			expr
		]
	]



(* ::Subsection:: *)
(*ItemKeys*)


Unprotect[ItemKeys]

Options[ItemKeys] = {Method -> Automatic, Modifiers -> {}}

ItemKeys[nb:focusedNotebookPat:FocusedNotebook[], path:menuPathPat, OptionsPattern[]] :=
	Module[{method, modifiers, resolvedNB, vPath, vMethod, vModifiers, buffer},
		blockInputNotebook[
			blockMenu[nb,
				Catch[
					(* step 1: set variables and validate user input *)
					throwIfMessage[
						{method, modifiers} = OptionValue[{Method, Modifiers}];
						resolvedNB = resolveFocusedNotebook[nb];
						vPath = validateMenuPath[ItemKeys, nb, path];
						vMethod = validateMenuMethod[ItemKeys, vPath, method];
						vModifiers = validateModifiers[ItemKeys, modifiers]
					];
					(* step 2: construct buffer *)
					throwIfMessage[
						RobotBlock[
							buffer =
								reapString[
									sowKeyModifiersString[vModifiers,
										Sow[
											Switch[vMethod,
												"ArrowKeys",
												ikArrowKeys[resolvedNB, vPath]
												,
												"MenuKey",
												ikMenuKey[resolvedNB, vPath]
												,
												"Keyboard",
												ikKeyboard[resolvedNB, vPath]
											]
										]
									]
								]
						]
					];
					(* step 3: execute buffer *)
					throwIfMessage[
						RobotExecute[FocusedNotebook[], buffer, CallInstallRobotTools -> False, CallSetFocusedNotebook -> False]
					]
				]
			]
		]
	]

SetAttributes[ItemKeys, ReadProtected]

ItemKeys[arg_] :=
	(Message[ItemKeys::menupath, 1, HoldForm[ItemKeys[arg]]]; $Failed)

ItemKeys[(*arg1:*)nbobjPat, args___] :=
	(Message[ItemKeys::menupath, 2, HoldForm[ItemKeys[args]]]; $Failed)

ItemKeys[arg1_, (*arg2:*)menuPathPat] :=
	(Message[ItemKeys::nbobj, 1, HoldForm[ItemKeys[arg1]]]; $Failed)

ItemKeys[args___] :=
	(ArgumentCountQ[ItemKeys, System`FEDump`NonOptionArgCount[{args}], 1, 3]; $Failed)

SyntaxInformation[ItemKeys] = {"ArgumentsPattern" -> {_., _, OptionsPattern[]}}

Protect[ItemKeys]

ikArrowKeys[(*nb:*)nbobjPat, path:menuPathPat] :=
	Module[{$tempMenu = $Menu, names, index},
		reapString[
			Do[
				If[i == 1,
					Switch[$InterfaceEnvironment,
						"Macintosh",
						Null
						,
						"Windows",
						Sow["\[AltKey]"]
						,
						"X",
						Sow["\[AltKey]\[LeftModified]"]
						,
						(*HoldPattern[$InterfaceEnvironment]*)
						_,
						Null
					]
				];
				names = itemName /@ DeleteCases[$tempMenu[[2]], Delimiter];
				index = Position[names, path[[i]]][[1]];
				Which[
					i == 1,
					(* "File" will already be selected, so use index - 1 *)
					Sow /@ Table["\\[RightKey]", Evaluate[index - 1]]
					,
					i == 2,
					Sow /@ Table["\\[DownKey]", Evaluate[index]]
					,
					True,
					(* the first menu item will already be selected, so use index - 1 *)
					Sow["\\[RightKey]"];
					Sow /@ Table["\\[DownKey]", Evaluate[index - 1]]
				];
				If[i == 1,
					Switch[$InterfaceEnvironment,
						"Macintosh",
						Null
						,
						"Windows",
						Null
						,
						"X",
						Sow["\[RightModified]"]
						,
						(*HoldPattern[$InterfaceEnvironment]*)
						_,
						Null
					]
				];
				names = itemName /@ $tempMenu[[2]];
				index = Position[names, path[[i]]][[1]];
				$tempMenu = Extract[$tempMenu[[2]], index]
				,
				{i, Length[path]}
			];
			Sow["\[EnterKey]"]
		]
	]

(*
Example:
In[1]:= convertMenuKey[FrontEnd`MenuKey["n", FrontEnd`Modifiers -> {FrontEnd`Control}]]

Out[1]= "\[ControlKey]\[LeftModified]n\[RightModified]"
*)
convertMenuKey[menuKey:FrontEnd`MenuKey[k_, OptionsPattern[]]] :=
	Module[{key, modifiers},
		key = k /. $frontEndKeys;
		modifiers = menuKey /. FrontEnd`MenuKey[_, OptionsPattern[]] :> (OptionValue[FrontEnd`Modifiers] /. $mkSymbolToStringRules);
		foldKeyModifiers[key, modifiers]
	]

ikMenuKey[(*nb:*)nbobjPat, path:menuPathPat] :=
	Module[{item, menuKey},
		item = menuItem[path];
		menuKey = First[Cases[item, FrontEnd`MenuKey[___]]];
		convertMenuKey[menuKey]
	]

ikKeyboard[(*nb:*)nbobjPat, path:menuPathPat] :=
	Module[{$tempMenu = $Menu, names, mnemonicPosTest, index, keys = {}, namesI, mnemonic, conflicts, enterFlag, iter, s},		
		(* TODO: make this not context sensitive, Last[keys], blah, like to use Reap/Sow *)
		Do[
			If[i == 1,
				Switch[$InterfaceEnvironment,
					"Macintosh",
					AppendTo[keys, "\[ControlKey]\[LeftModified]\\[F2Key]\[RightModified]\\[DownKey]\\[RightKey]"]
					,
					"Windows",
					AppendTo[keys, "\[AltKey]"]
					,
					"X",
					AppendTo[keys, "\[AltKey]\[LeftModified]"]
					,
					(*HoldPattern[$InterfaceEnvironment]*)
					_,
					Null
				]
			];
			(* set up variables *)
			names = DeleteCases[itemName /@ $tempMenu[[2]], ""];
			conflicts = {};
			mnemonic = "";
			index = strippedPosition[names, path[[i]]][[1]];
			enterFlag = True;
			(* namesI is the string of the menu item *)
			namesI = Extract[names, index];
			mnemonicPosTest = StringPosition[namesI, "&"];
			If[mnemonicPosTest != {},
				mnemonic = ToLowerCase[StringTake[namesI, {mnemonicPosTest[[1, 1]] + 1}]];
				conflicts = Flatten[StringCases[names, item:(___ ~~ "&" ~~ mnemonic ~~ ___) :> item, IgnoreCase->True]]
			];
			AppendTo[keys,
				If[Length[conflicts] == 1,
					(* if there is simply 1 mnemonic, then use it *)
					enterFlag = False;
					mnemonic
					,
					(*
					iter is the number of steps to repeat a certain action to traverse a menu,
					normally it's hitting the down key, but when there is a mnemonic conflict,
					the mnemonic has to be hit a certain number of times
					
					the || True is here so that mnemonics with conflicts are never used. Currently there is no easy way to detect if a 
					menu item is dimmed, and dimmed menu items are skipped over when cycling through mnemonics.
					This leads to interesting behavior. For instance, the Edit > Cut menu item and Edit > Paste As Plain Text menu
					item both have t for mnemonics. However, Cut is dimmed if nothing is in the clipboard. But even though
					ItemKeys["Paste As Plain Text"] returning "\[AltKey]\[LeftModified]ett\[EnterKey]\[RightModified]" is wrong,
					this was never detected because after typing the first t, Paste As Plain Text was activated because it was the
					only non-dimmed item with a t mnemonic. So even though it was doing the wrong thing, no one ever noticed it.
					It would have been detected if there was a something selected, so that Cut was no longer dimmed.
					 *)
					{iter, s} =
						If[conflicts == {} || True(* DO NOT REMOVE THIS! *),
							(* there are no mnemonics, so use arrow keys *)
							(* the -1 comes from always hiliting the first menu item of the next sub menu*)
							{index-1, If[i == 1, "\\[RightKey]", "\\[DownKey]"]}
							,
							(* there are mnemonics conflicts *)
							(* find out where namesI is in the list of conflicts *)
							{First[Position[conflicts, namesI]]-Boole[MemberQ[conflicts, names[[1]]]], mnemonic}
						];
					(* this is a tricky Which statement, but all cases for arrow keys and mnemonics should be taken care of *)
					Which[
						(* at the main menu bar, and going into a sub menu *)
						i == 1 && Length[path] > 1,
						(* "File" will already be selected, so use index - 1 *)
						{Table[s, Evaluate[iter]], "\\[DownKey]", ""}
						,
						(* only selecting something from the main menu *)
						i == 1,
						{Table[s, Evaluate[iter]], "\[RightModified]"}
						,
						(* used arrow keys at i == 1, and this menu item opens another sub menu *)
						i == 2 && MatchQ[Last[keys], {"\\[RightKey]"..., "\\[DownKey]"}] && Length[path] > 2,
						{Table[s, Evaluate[iter]], "\\[RightKey]"}
						,
						(* used arrow keys at i == 1, and staying at i == 2 *)
						i == 2 && MatchQ[Last[keys], {"\\[RightKey]"..., "\\[DownKey]"}],
						Table[s, Evaluate[iter]]
						,
						(* i >= 2 and opening a sub menu *)
						Length[path] > i,
						{Table[s, Evaluate[iter]], "\\[RightKey]"}
						,
						(* i >= 2 *)
						True,
						Table[s, Evaluate[iter]]
					]
				]
			];
			If[i == 1,
				Switch[$InterfaceEnvironment,
					"Macintosh",
					Null
					,
					"Windows",
					Null
					,
					"X",
					AppendTo[keys, "\[RightModified]"]
					,
					(*HoldPattern[$InterfaceEnvironment]*)
					_,
					Null
				]
			];
			(* reset variables *)
			names = itemName /@ $tempMenu[[2]];
			index = strippedPosition[names, path[[i]]][[1]];
			$tempMenu = Extract[$tempMenu[[2]], index]
			,
			{i, Length[path]}
		];
		(* if the last menu item was arrived at by arrow keys, then enter needs to be hit *)
		If[enterFlag, AppendTo[keys, "\[EnterKey]"]];
		StringJoin[keys]
	]

(*
TODO: if elipses, ampersands, spaces, or parentheses are specified, then they need to count towards the validation of the menu path.
Maybe detect which features a user-input item name has, and only strip the other ones from the front end menu, so that comparisons
would work
*)

strip[s:stringPat] :=
	Module[{jParensSPat = "\\(&.\\)", ampSPat = "&", elipSPat = "\\.{3}"},
		StringReplace[
			StringReplace[
				(* strip Japanese mnemonics, ampersands, and elipses *)
				StringReplace[s, RegularExpression[jParensSPat <> "|" <> ampSPat <> "|" <> elipSPat] -> ""
				], {
				(* leave string alone if it is "xx Point" or "x%" *)
				RegularExpression["^(\\d)+( Point|%)$"] -> "$0",
				(* strip index *)
				RegularExpression["^(\\d)+ (.*)$"] -> "$2"
				}
			]
			,
			(* strip surrounding spaces *)
			RegularExpression["^( )*([^ ].*[^ ])( )*$"] -> "$2"
		]
	]

strippedPosition[expr_, pattern_] :=
	Position[strip /@ expr, strip[pattern]]



(* ::Subsection:: *)
(*ItemActivate*)


Unprotect[ItemActivate]

Options[ItemActivate] = {Method -> Automatic, Modifiers -> {}}

ItemActivate[nb:focusedNotebookPat:FocusedNotebook[], path:menuPathPat, OptionsPattern[]] :=
	Module[{method, modifiers, resolvedNB, vMenuPath, vMethod, vModifiers, buffer},
		blockInputNotebook[
			blockMenu[nb,
				Catch[
					(* step 1: set variables and validate user input *)
					throwIfMessage[
						{method, modifiers} = OptionValue[{Method, Modifiers}];
						resolvedNB = resolveFocusedNotebook[nb];
						vMenuPath = validateMenuPath[ItemActivate, nb, path];
						vMethod = validateMenuMethod[ItemActivate, vMenuPath, method];
						vModifiers = validateModifiers[ItemActivate, modifiers];
						(*validateFocusedNotebook[ItemActivate, nb]*)
					i];
					(* step 2: construct buffer *)
					throwIfMessage[
						RobotBlock[
							buffer =
								reapHeldList[
									sowKeyModifiers[vModifiers,
										Switch[vMethod,
											"AppleScript",
											Sow[appleScriptExecute[clickMenuItemScript["Mathematica", vMenuPath]]]
											,
											"ArrowKeys",
											Sow[delay[$InitialKeyTypeDelay]];
											Sow /@ keyType[ikArrowKeys[resolvedNB, vMenuPath]]
											,
											"MenuKey",
											Sow[delay[$InitialKeyTypeDelay]];
											Sow /@ keyType[ikMenuKey[resolvedNB, vMenuPath]]
											,
											"Keyboard",
											Sow[delay[$InitialKeyTypeDelay]];
											Sow /@ keyType[ikKeyboard[resolvedNB, vMenuPath]]
											,
											"Mouse",
											Sow[iaMouse[resolvedNB, vMenuPath]]
										]
									]
								]
						]
					];
					(* step 3: execute buffer *)
					throwIfMessage[
						RobotExecute[nb, buffer, CalledByItemActivate -> True]
					]
				]
			]
		]
	]

SetAttributes[ItemActivate, ReadProtected]

ItemActivate[arg_] :=
	(Message[ItemActivate::menupath, 1, HoldForm[ItemActivate[arg]]]; $Failed)

ItemActivate[(*arg1:*)nbobjPat, args___] :=
	(Message[ItemActivate::menupath, 2, HoldForm[ItemActivate[args]]]; $Failed)

ItemActivate[arg1_, (*arg2:*)menuPathPat] :=
	(Message[ItemActivate::nbobj, 1, HoldForm[ItemActivate[arg1]]]; $Failed)

ItemActivate[args___] :=
	(ArgumentCountQ[ItemActivate, System`FEDump`NonOptionArgCount[{args}], 1, 3]; $Failed)

SyntaxInformation[ItemActivate] = {"ArgumentsPattern" -> {_., _, OptionsPattern[]}}

Protect[ItemActivate]

(*
iaMouse for Windows is a function because Windows does not correctly calculate menu rectangles of menus that have not been displayed
yet. Therefore, the rectangles have to be calculated at run-time, and iaMouse should be treated as low-level so that it doesn't
evaluate before run-time.
*)

iaMouse[(*nb:*)focusedNotebookPat, path:menuPathPat] /; $InterfaceEnvironment == "Windows" :=
	Module[{spec, indices, rects, mp, mp1},
		spec = menuItemSpec[path];
		(* subtract 1 because Win32 menu functions are 0-based *)
		indices = Take[spec, 2;;;;2] - 1;
		Do[
			rects = Win32`User32`GetMenuPathRectangles[indices];
			mp = Mean[rects[[i]]];
			mp1 = Mean[rects[[i-1]]];
			Switch[i,
				1,
				mouseMove[mp];
				mouseClick["Button1"]
				,
				2,
				Scan[mouseMove, {mp1, {mp1[[1]], mp[[2]]}, mp}];
				mouseClick["Button1"]
				,
				_,
				Scan[mouseMove, {mp1, {mp[[1]], mp1[[2]]}, mp}];
				mouseClick["Button1"]
			]
			,
			{i, Length[indices]}
		]
	]

(* TODO: use Win32 API PostMessage for activating menu items *)
iaPostMessage[(*nb:*)focusedNotebookPat, (*path:*)menuPathPat] :=
	Module[{},
		Null
	]



(* ::Subsection:: *)
(*MenuSetup*)


(* Windows is the only system that numbers recently opened notebooks *)

prependIndex[itemNames:listPat] :=
	MapIndexed[If[#2[[1]] <= 9, "&" <> ToString[#2[[1]], OutputForm] <> " " <> #1, ToString[#2[[1]], OutputForm] <> " &" <> #1]&, itemNames]

(* return the 2nd arg of Rule, if r is a rule, otherwise, return the whole expression *)
stripRule[r_] :=
	r /. Rule[_, b_] :> b

deleteAll[list:listPat] :=
	DeleteCases[list, All->All]

itemize[Delimiter] =
	Delimiter

itemize[s:menuItemNamePat] :=
	With[{ss = ToString[s, OutputForm]},
		MenuItem[ss, "xxx"]
	]

itemize[s:menuItemNamePat, menuKey:menuKeyPat] :=
	With[{ss = ToString[s, OutputForm]},
		MenuItem[ss, "xxx", menuKey]
	]

displayize[s:stringPat] :=
	s <> " Display"

(* expandAdditionalItems is a run-time front end function, and can introduce new symbols *)
(* FrontEndResource["palettesMenuAdditionalItems"] introduces KernelExecute and MenuEvaluator, so we have
to use Begin[] and End[] *)
expandAdditionalItems[resource:stringPat] :=
	Internal`WithLocalSettings[
		Begin["FrontEnd`"]
		,
		DeleteCases[
			Block[{FE`InstallDialog`InstallDialogFunc},
				ReplaceAll[
					menuSet=ToExpression[FrontEndResourceString[resource]]
					,
					(* list of all symbols from MenuSetup that might have a conflict. This has to be maintained manually. *)
					{RobotTools`Modifiers -> FrontEnd`Modifiers}
				]
			],
			Delimiter
		]
		,
		End[]
	]

(* the majority of GetPopupList  *)
gplSimple[vals_] :=
	itemize /@ stripRule /@ vals

gplMenuListNotebooksMenu[vals_] :=
	Switch[$InterfaceEnvironment,
		"Macintosh",
		itemize /@ stripRule /@ vals
		,
		"Windows",
		itemize /@ (prependIndex[stripRule /@ vals] /. {} -> {"(Empty)"})
		,
		"X",
		itemize /@ (stripRule /@ (vals /. {} -> {"(Empty)"}))
	]

gplComplexMenu[vals_, additionalItems_] :=
	Module[{dPos},
		(* get the position of the last Delimiter, everything below it will be nested menus. *)
		dPos = Position[vals, Delimiter][[-1, 1]];
		Flatten[{
			itemize /@ stripRule /@ vals[[;;dPos-1]]
			,
			Reap[
				Scan[
					Sow[
						#[[2]],
						(* TODO: when the front end is consistent with a style,
						remove support for the other one *)
						If[Head[#[[1]]] === FrontEnd`FileName,
							(* old style, returned by nested menu items, with FrontEnd`FileName *)
							#[[1, 1, 1]],
							(* new style, returned by top menu items, just a flat string *)
							Most[StringSplit[#[[1]], $PathnameSeparator]]
						]
					]&
					,
					vals[[dPos+1;;]]
				],
				_,
				Menu[#1, itemize /@ #2]&
			][[2]]
			,
			Delimiter
			,
			expandAdditionalItems[additionalItems]
		}]
	]

gplMenuListConvertFormatTypes[vals_] :=
	Switch[$InterfaceEnvironment,
		"Macintosh",
		itemize[#,
			Switch[#,
				"StandardForm",
				FrontEnd`MenuKey["n", FrontEnd`Modifiers->{"Command", "Shift"}]
				,
				"TraditionalForm",
				FrontEnd`MenuKey["t", FrontEnd`Modifiers->{"Command", "Shift"}]
				,
				_,
				Unevaluated[Sequence[]]
			]
		]
		,
		"Windows" | "X",
		itemize[#,
			Switch[#,
				"StandardForm",
				FrontEnd`MenuKey["n", FrontEnd`Modifiers->{"Control", "Shift"}]
				,
				"TraditionalForm",
				FrontEnd`MenuKey["t", FrontEnd`Modifiers->{"Control", "Shift"}]
				,
				_,
				Unevaluated[Sequence[]]
			]
		]
	]& /@ stripRule /@ vals

gplMenuListDisplayAsFormatTypes[vals_] :=
	itemize /@ displayize /@ stripRule /@ vals
gplMenuListCellTags[vals_] :=
	itemize /@ (stripRule /@ (vals /. {} -> {"(Empty)"}))

GetPopupList[nb:focusedNotebookPat:FocusedNotebook[], s:stringPat] /; $Notebooks :=
	Module[{vals, tempNB},
		vals = deleteAll[getPopupList[nb /. HoldPattern[FocusedNotebook[]] :> Sequence[], s]];

		Switch[s,
			"MenuListNotebooksMenu",
			gplMenuListNotebooksMenu[vals]
			,
			"MenuListStyles",
			gplSimple[vals]
			,
			"MenuListStyleDefinitions",
			gplComplexMenu[vals, "styleSheetMenuAdditionalItems"]
			,
			"MenuListScreenStyleEnvironments" | "MenuListPrintingStyleEnvironments",
			gplSimple[vals]
			,
			"MenuListConvertFormatTypes",
			gplMenuListConvertFormatTypes[vals]
			,
			"MenuListDisplayAsFormatTypes",
			gplMenuListDisplayAsFormatTypes[vals]
			,
			"MenuListCellTags",
			gplMenuListCellTags[vals]
			,
			"MenuListGlobalEvaluators" | "MenuListNotebookEvaluators" | "MenuListStartEvaluators" | "MenuListQuitEvaluators",
			gplSimple[vals]
			,
			"MenuListPalettesMenu",
			gplComplexMenu[vals, "palettesMenuAdditionalItems"]
			,
			"MenuListWindows",
			(* work around front end "optimization" where the Windows menu only gets refreshed when a new notebook is created *)
			tempNB = MathLink`CallFrontEnd[FrontEnd`NotebookCreateReturnObject[Visible -> False, WindowTitle -> ""]];
			MathLink`CallFrontEnd[FrontEnd`NotebookClose[tempNB]];
			gplSimple[vals]
			,
			"MenuListCDFPreview",
			(*Added new menu items CDF Preview> CDF Player and CDF Player Pro*)
            gplSimple[vals]
            ,
            "MenuListOpenItems",
			(*Added new menu items *)
            gplSimple[vals]
            ,
            "MenuListSaveItems",
			(*Added new menu items *)
            gplSimple[vals]
            ,
            "MenuListWolframCloudAccountMenu",
			(*Added new menu items *)
            gplSimple[vals]
			,
			(* catch very old or very new MenuAnchors *)
			_,
			Message[GetPopupList::anchor, s];
			$Failed
		]
	]

(* getPopupList is a run-time front end function *)
(* the False is defensive programming, just in case FEPrivate`GetPopupList ever gets hooked up to Dynamic *)
getPopupList[nb:nbobjPat, s:stringPat] :=
	MathLink`CallFrontEnd[FrontEnd`Value[FEPrivate`GetPopupList[nb, s], False]]
getPopupList[s:stringPat] :=
	MathLink`CallFrontEnd[FrontEnd`Value[FEPrivate`GetPopupList[s], False]]


(*
values for Scope
	GlobalPreferences
	NotebookDefault
	SelectionCell
	Selection
*)

(*Options[FrontEnd`Item] = {FrontEnd`MenuAnchor -> False, FrontEnd`MenuEvaluator -> Automatic, FrontEnd`Scope -> Selection, Visible -> True}*)

(*Attributes[FrontEnd`Item] = {HoldAllComplete}

FrontEnd`Item[_String, command_String, FrontEnd`MenuAnchor -> True] :=
	Sequence @@ GetPopupList[FocusedNotebook[], command]*)


(* treat ToggleItem the same as Item, until better tests are developed *)

Attributes[FrontEnd`ToggleMenuItem] = {HoldRest}

FrontEnd`ToggleMenuItem =
	MenuItem


(* Have LinkedItems and AlternateItems just "evaluate away" *)

FrontEnd`LinkedItems = Sequence@@#&


FrontEnd`AlternateItems = Sequence@@#&


itemName[menuHeadPat[name_String, __]] :=
	name

itemName[Delimiter] =
	""


Options[FrontEnd`MenuKey] = {FrontEnd`Modifiers -> {}}


(* this is the mini-API for $Menu. Some time, $Menu will become private and only the API functions will be used. *)

(*
Because the front end uses symbols that may be in System`, such as Item, define our own custom FrontEndMenu* version of such symbols.
This will help to not interfere with System` symbols, such as giving Item the attribute HoldAllComplete.
Options and things like Shift, Command, Option, and Control are skipped since there are no definitions for them.

KEEP Blocking FrontEnd`Item! This is to keep $menu in an unevaluated state so that when it is reevaluated, the menu anchors resolve
to the correct notebook.
*)

$frontEndKeys :=
	$frontEndKeys =
	Module[{fileName = FileNameJoin[{$RobotToolsTextResourcesDirectory, "FrontEndKeys.m"}]},
		If[$Notebooks, Get[fileName], {}]
	]

$frontEndModifierKeys :=
	$frontEndModifierKeys =
	FilterRules[$frontEndKeys, {"AltKey", "CommandKey", "ControlKey", "Ctrl", "OptionKey", "ShiftKey", "Shift"}]

tabSplit[s:stringPat] :=
	StringSplit[s, "\t"]

plusSplit[s:stringPat] :=
	StringSplit[s, "+"]

Module[{$unevaluatedMenu},
	(* dump all unresolved symbols into FrontEnd` *)
	Begin["FrontEnd`"];
	$unevaluatedMenu =
		If[$Notebooks,
			ReplaceAll[
				(* use ToExpression[FrontEndResourceString[]] instead of FrontEndResource[] for speed *)
				Check[
					ToExpression[FrontEndResourceString["MenuSetup"]]
					,
					Message[Menu::syntax];
					Menu["Mathematica", {}]
				],
				(* list of all symbols from MenuSetup that might have a conflict. This has to be maintained manually. *)
				{RobotTools`Modifiers -> FrontEnd`Modifiers}
			]
			,
			Menu["Mathematica", {}]
		];
	End[];
	With[{$unevaluatedMenu = $unevaluatedMenu},
		menuFactory[nb:focusedNotebookPat] :=
			Block[{FocusedNotebook},
				FocusedNotebook[] = nb;
				$unevaluatedMenu
			]
	]
]
(*
menuItemSpec gives the spec for Extract[menu, spec] of a menu item
menuItemSpec[FocusedNotebook[], {"&File", "&New", "&Notebook (.nb)"}] returns {2, 1, 2, 1, 2, 1}
*)
menuItemSpec[path:menuPathPat] :=
	Module[{$tempMenu = $Menu, i, posTest, spec = {}},
		Do[
			posTest = Position[$tempMenu, menuHeadPat[path[[i]], __], 2];
			spec = Join[spec, First[posTest]];
			$tempMenu = Extract[$tempMenu, First[posTest]]
			,
			{i, Length[path]}
		];
		spec
	]

(*
menuPathMnemonics returns a list of mnemonics in the menu path
an empty string is returned if there is no mnemonic
menuPathMnemonics[FocusedNotebook[], {"&File", "&New", "&Notebook"}] returns {"F", "N", "N"}
menuPathMnemonics[FocusedNotebook[], {"&Cell", "&Convert To", "TraditionalForm"}] returns {"C", "C", ""}
*)
menuPathMnemonics[(*nb:*)focusedNotebookPat, path:menuPathPat] :=
	Module[{(*names, *)pos},
		(*names = menuPathNames[nb, path];*)
		pos = Flatten /@ StringPosition[(*names*)path, "&"] + 1;
		(* the Unevaluated is to avoid the StringTake::seqs message when StringTake evaluates *)
		Thread[Unevaluated[StringTake[(*names*)path, pos]]]
	]

(*
hasMenuKey returns whether the specified menu item has a menu key
*)
hasMenuKey[path:menuPathPat] :=
	Module[{item},
		item = menuItem[path];
		Cases[item, FrontEnd`MenuKey[__]] != {}
	]

(*
menuPathItems returns a list of Menus and MenuItems for each menu item in the path.
Child menu items are removed from parent menus
*)
menuPathItems[path:menuPathPat] :=
	Module[{spec, namesSpec, items},
		spec = menuItemSpec[path];
		namesSpec = Reverse[NestWhileList[Drop[#, -2]&, spec, # =!= {}&, 1, Infinity, -1]];
		items = Extract[$Menu, namesSpec];
		(* shave off child items *)
		items = items /. (head:menuHeadPat)[name_, {___}] :> head[name, {}];
		items
	]

(*
menuItem returns the Menu or MenuItem for the specified menu item
*)
menuItem[path:menuPathPat] :=
	Module[{items, item},
		items = menuPathItems[path];
		item = Last[items];
		item
	]



(* ::Subsection:: *)
(*Validation*)


(* TODO: handle Macintosh menu paths like {"Mathematica", "Foo"}, where hasMenuKey is going to barf *)

validateMenuMethod[head:symbolPat, path:menuPathPat, method:blankPat] :=
	Which[
		method === Automatic,
		If[hasMenuKey[path],
			"MenuKey"
			,
			If[head === ItemKeys,
				"Keyboard"
				,
				(*ItemActivate*)
				Switch[$InterfaceEnvironment,
					"Macintosh", "Keyboard"
					,
					"Windows", "Keyboard"
					,
					"X", "Keyboard"
				]
			]
		]
		,
		head === ItemKeys && MatchQ[method, "ArrowKeys" | "MenuKey" | "Keyboard"],
		method
		,
		head === ItemActivate && MatchQ[method, "AppleScript" | "ArrowKeys" | "MenuKey" | "Keyboard" | "Mouse"],
		method
		,
		True,
		Message[head::optvg, Method, method, "a valid menu method"];
		$Failed
	]

(*
use === here just to ensure nothing funny happens in a stand-alone kernel
TODO: These may not have to be translated...
*)
isSpecialMacintoshMenuPath[path:menuPathPat] :=
	$InterfaceEnvironment === "Macintosh" && StringMatchQ[path[[1]], {"Apple", "Mathematica"}]

validateMenuPath[head:symbolPat, (*nb:*)focusedNotebookPat, path:menuPathPat] :=
	Module[{$tempMenu, tPath, itemNames, item, posTest, vPath = {}, $tempMenu2},
		If[isSpecialMacintoshMenuPath[path],
			(* is MacOSX and first menu is Apple or Mathematica, which we don't completely define,
			so punt on validating (for now) *)
			path
			,
 tPath = translateMenuPath[path];
            $tempMenu = $Menu;
			Do[
               item = tPath[[i]];
				$tempMenu2 = $tempMenu[[2]];
               itemNames = itemName /@ $tempMenu2;
               posTest = strippedPosition[itemNames, item];
				If[posTest != {},
                   $tempMenu = Extract[$tempMenu2, First[posTest]];
					AppendTo[vPath, itemName[$tempMenu]]
					,
					Break[]
				]
				,
				{i, Length[tPath]}
			];
			If[posTest == {},
				Message[head::noitem, item];
				$Failed
				,
				vPath
			]
		]
	]



(* ::Subsection:: *)
(*Translation*)


(* use := just to keep the connotation that $MenuLanguage is a run-time thing, it doesn't actually change behavior *)
$MenuLanguage :=
	$InterfaceLanguage

menuTranslation[lang:stringPat] :=
	menuTranslation[lang] =
	Module[{fileName = FileNameJoin[{$RobotToolsTextResourcesDirectory, $InterfaceLanguage, "Menu", "Translation.m"}]},
		If[lang == $InterfaceLanguage,
			{},
			Get[fileName]
		]
	]

translateMenuPath[path:menuPathPat] :=
	path /. menuTranslation[$MenuLanguage]



(* ::Subsection:: *)
(*End*)


End[] (*`Menu`Private`*)
