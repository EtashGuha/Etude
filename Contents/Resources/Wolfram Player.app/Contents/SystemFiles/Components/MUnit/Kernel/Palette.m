(* ::Package:: *)

PaletteClear::usage =
""

PaletteConvertCellGroup::usage =
""

$exclusionForms::usage =
""

$formCheck::usage =
""

PaletteConvertNotebook::usage =
""

PaletteNewTestNotebook::usage =
""

$windowtitle::usage =
""

PaletteRun::usage =
""

buttonWithIcon::usage =
""

buttonWithIconAndTooltip::usage =
""

buttonWithoutIcon::usage =
""

buttonWithTooltipButWithoutIcon::usage =
""

addMessages::usage =
""

addOptions::usage =
""

InsertOptionRule::usage =
""

PaletteAbort::usage =
""

PaletteInsertTest::usage =
""

PaletteAddTestIDs::usage =
""

PaletteInsertTestSection::usage =
""

testRerun::usage =
""

testReplaceOutput::usage =
""

testReplaceMessage::usage =
""

PaletteSaveAs::usage =
""

bottomCell::usage =
""

bottomCellWithRightButton::usage =
""

resultCell::usage =
""
(*
resultColorBar::usage =
""
*)
findFailedTest::usage =
""

actionMenuResultColorBarAndButton::usage =
""

testResultsDockedCell::usage =
"testResultsCell[tr] where tr is the TestReportObject for a set of tests, gives a cell intended to be docked under the existing docked cell in a test notebook. It displays the total number of tests, tests failed, message failures and skipped. It also has buttons for jumping to the first failed test and for deleting all test result cells as well as the docked test results cell."

goInsideNextTabbingCell::usage =
""

IncludeInSuperFunction::usage =
""

clearTestResults::usage =
""

buttonConvertCellGroup::usage = ""

convertNotebookToTest::usage = "Convert a standard notebook to a test notebook."

identifyIsolatedCells::usage = "identifyIsolatedCells[] finds all isolated cells in a testing notebook and sets $problemCellids to be equal to that list.";

$problemCellids::usage = "$problemCellids gives the list of cell ids of all isolated cells in a testing notebook. Set by identifyIsolatedCells[].";

defectiveCellsDockedCell::usage = "defectiveCellsDockedCell[$problemCellids] gives a cell corresponding to a given $problemCellids intended to be docked under the existing docked cell in a test notebook. It shows the number of elements given by $problemCellids. It has buttons \"Check notebook\", \"Find next problem cell\", \"Fix cell\" and \"Remove fix cells toolbar\".";

fixAllAndRunTests::usage = ""

findNextProblemCell::usage = "Given a cursor position in a test notebook, finds the next cell with CellID in the list $problemCellids. The find wraps if necessary.";

completeCellToTestCellGroup::usage = "";

addDockedCellConvertNotebook::usage = "";

attachInsideFrameLabel::usage = "";


Begin["`Package`"]

End[]

Begin["`Palette`Private`"]

Unprotect[attachInsideFrameLabel]

auxFindFailedTest[nb_, dir:("Next"|"Previous")]:=
	Module[{ci, failureIDs, allCellIDs, p, id},
		SelectionMove[nb, If[dir==="Next", Next, Previous], Cell, AutoScroll->False];
		ci = Developer`CellInformation[nb]; 
		If[ci === $Failed,
			SelectionMove[nb, If[dir === "Next", Before, After], Notebook, AutoScroll->False]; 
			SelectionMove[nb, If[dir === "Next", Next, Previous], Cell, AutoScroll->False]; 
			ci = Developer`CellInformation[nb]];
		If[Not[MemberQ[{{"TestFailure"}, {"TestMessageFailure"}, {"TestError"}}, "Style" /. ci]], 
			failureIDs = CurrentValue[#, CellID] & /@ Cells[nb, CellStyle -> ("TestFailure" | "TestMessageFailure" | "TestError")]; 
			allCellIDs = CurrentValue[#, CellID] & /@ Cells[nb]; 
			p = Position[allCellIDs, ("CellID" /. ci)[[1]]][[1, 1]]; 
			id = SelectFirst[If[dir === "Next", Take[allCellIDs, {p, -1}], Reverse@Take[allCellIDs, {1, p}]], MemberQ[failureIDs, #] &];
			If[id === Missing["NotFound"],
				SelectionMove[nb, If[dir === "Next", Before, After], Notebook, AutoScroll->False]; 
				SelectionMove[nb, If[dir === "Next", Next, Previous], Cell, AutoScroll->False];
				ci = Developer`CellInformation[nb]; 
				p = Position[allCellIDs, ("CellID" /. ci)[[1]]][[1, 1]]; 
				id = SelectFirst[If[dir === "Next", Take[allCellIDs, {p, -1}], Reverse@Take[allCellIDs, {1, p}]], MemberQ[failureIDs, #] &]];
			NotebookFind[nb, id, If[dir === "Next", Next, Previous], CellID, AutoScroll->False];
			SelectionMove[nb,All,CellGroup]];
		If[MemberQ[{{"TestFailure"}, {"TestMessageFailure"}, {"TestError"}}, "Style" /. ci], 
			SelectionMove[nb, All, CellGroup]]
	]

findFailedTest[nb_,dir:("Next"|"Previous")]:=
	Module[{testfailurecells,ci},
		If[(testfailurecells = Cells[nb,CellStyle->("TestFailure" | "TestMessageFailure" | "TestError")])==={},
			CurrentValue[nb, {TaggingRules, "$someTestsFailed"}] = Inherited
			,
			ci=Developer`CellInformation[nb];
			Which[(*The cursor is between cells or at the cell bracket of a "TestFailure" cell.*)
				ci===$Failed||(MatchQ[ci,{{___,"Style"->("TestFailure" | "TestMessageFailure" | "TestError"),___,"CursorPosition"->"CellBracket",___}}]&&Length@testfailurecells>1)
				,
				auxFindFailedTest[nb, dir]
				,
				(*Selecting a test failure cell group. *)
				MatchQ[ci, {{___, "Style" -> "VerificationTest", __}, __, {___, "Style" -> ("TestFailure" | "TestError" | "TestMessageFailure"), ___}, {"Style" -> "BottomCell", __}}]
				,
				SelectionMove[nb, If[dir === "Next", After, Before], CellGroup];
				auxFindFailedTest[nb, dir]
				,
				(*Inside a "TestFailure" cell.*)
				MatchQ[ci,{{___,"Style"->("TestFailure" | "TestError" | "TestMessageFailure"),___}}]
				,
				SelectionMove[nb,All,Cell,AutoScroll->False]
				,
				True,
				auxFindFailedTest[nb, dir]
			]
		]
	]
	
goInsideNextTabbingCell[] := 
	Module[{nb = InputNotebook[], ci, cellid, relevantIDs, p, id}, 
		ci = Developer`CellInformation[nb]; 
		If[MatchQ[ci, {{___, "CellID" -> _Integer, ___}}], 
			cellid = ("CellID" /. ci)[[1]];
			relevantIDs = CurrentValue[#, CellID] & /@ Cells[nb, CellStyle -> ("VerificationTest" | "ExpectedOutput" | "ActualOutput" | "ExpectedMessage" | "ActualMessage" | "TestOptions")];
			p = Position[relevantIDs, cellid][[1, 1]];
			id = SelectFirst[Take[relevantIDs, {p + 1, -1}], MemberQ[relevantIDs, #] &];
			If[id =!= Missing["NotFound"], NotebookFind[nb, id, Next, CellID, AutoScroll -> False]; SelectionMove[nb, Before, CellContents]]]]

Options[clearTestResults] = {IncludeInSuperFunction -> False};

clearTestResults[opts___] := 
	Module[{nb = ButtonNotebook[], i = (IncludeInSuperFunction/.{opts}/.Options[clearTestResults])}, 
		If[i === False, CurrentValue[nb, ShowSelection] = False];
		SelectionMove[nb, All, Notebook, AutoScroll -> False];
		CurrentValue[nb, DockedCells] = (If[Head@# === Cell, #, DeleteCases[#, Cell[_, "DockedCell", ___, CellTags -> "MUnitResultsCell", ___]][[1]]] &[CurrentValue[nb, DockedCells]]);
		NotebookWrite[nb, 
				DeleteCases[NotebookRead[nb], Cell[_, "ActualMessage" | "ActualOutput" | "TestFailure" | "TestMessageFailure" | "TestSuccess" | "TestError", ___], 
						Infinity] /. Cell[_, "BottomCell", ___] -> Cell[BoxData[ToBoxes@MUnit`bottomCell[]], "BottomCell"], 
				AutoScroll -> False];
		CurrentValue[nb, {TaggingRules, "$testsRun"}] = False;
		CurrentValue[nb, {TaggingRules, "$someTestsFailed"}] = Inherited;
		If[i === False, CurrentValue[nb, ShowSelection] = Inherited]]
               
reportGrid2by2[total_, successcount_, failurecount_, messagefailurecount_] := 
	Grid[{{Style[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "TotalTestsRun-Label"], 11, RGBColor[0.34902, 0.34902, 0.34902]], Framed[Style[ToString@total, 11, Bold, White],
			Background -> GrayLevel[.55], FrameStyle -> GrayLevel[.55], FrameMargins -> {{8, 8}, {0, 0}}], 
		Grid[{{Style[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "Failures-Label"], 11, RGBColor[0.34902, 0.34902, 0.34902]],
			Pane[Style[Grid[{{Style["(", 8], Style[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "WrongResults-Label"], 10], Style[")", 8]}}, Spacings -> 0],
					RGBColor[0.521569, 0.521569, 0.521569]], 
				ImageMargins -> {{Automatic, Automatic}, {-.3, Automatic}}]}},
					Spacings -> {{2 -> .3}, Automatic}, Alignment -> {Automatic, Center}], 
			Framed[Style[ToString@failurecount, 11, Bold, White], Background -> RGBColor[0.74902, 0.403922, 0.4], FrameStyle -> RGBColor[0.74902, 0.403922, 0.4],
																					FrameMargins -> {{8, 8}, {0, 0}}]},
		{Style[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "Successes-Label"], 11, RGBColor[0.34902, 0.34902, 0.34902]], Framed[Style[ToString@successcount, 11, Bold, White],
																			Background -> RGBColor[0.380392, 0.603922, 0.384314],
												FrameStyle -> RGBColor[0.380392, 0.603922, 0.384314], FrameMargins -> {{8, 8}, {0, 0}}], 
		Grid[{{Style["Failures", 11, RGBColor[0.34902, 0.34902, 0.34902]],
			Pane[Style[Grid[{{Style["(", 8], Style[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "Messages-Label"], 10], Style[")", 8]}}, Spacings -> 0],
				RGBColor[0.521569, 0.521569, 0.521569]], 
				ImageMargins -> {{Automatic, Automatic}, {-.3, Automatic}}]}}, Spacings -> {{2 -> .3}, Automatic}, 
			Alignment -> {Automatic, Center}],
			Framed[Style[ToString@messagefailurecount, 11, Bold, White], Background -> RGBColor[0.921569, 0.678431, 0.337255], FrameStyle -> RGBColor[0.921569, 0.678431, 0.337255],
																FrameMargins -> {{8, 8}, {0, 0}}]}}, Alignment -> Left, 
		BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 12}, Spacings -> {{2 -> 1, 3 -> 2}, .5}]
		
reportGrid2by3[total_, successcount_, errorcount_, failurecount_, messagefailurecount_] := 
	Grid[{{Style[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "TotalTestsRun-Label"], 11, RGBColor[0.34902, 0.34902, 0.34902]], 
		Framed[Style[ToString@total, 11, Bold, White], Background -> GrayLevel[.55], FrameStyle -> GrayLevel[.55], FrameMargins -> {{8, 8}, {0, 0}}], 
		Style[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "Successes-Label"], 11, RGBColor[0.34902, 0.34902, 0.34902]], 
		Framed[Style[ToString@successcount, 11, Bold, White], Background -> RGBColor[0.380392, 0.603922, 0.384314], FrameStyle -> RGBColor[0.380392, 0.603922, 0.384314], 
			FrameMargins -> {{8, 8}, {0, 0}}], 
		Grid[{{Style[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "Failures-Label"], 11, RGBColor[0.34902, 0.34902, 0.34902]],
			Pane[Style[Grid[{{Style["(", 8], Style[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "WrongResults-Label"], 10], Style[")", 8]}}, Spacings -> 0], 
												RGBColor[0.521569, 0.521569, 0.521569]], 
											ImageMargins -> {{Automatic, Automatic}, {-.3, Automatic}}]}},
			Spacings -> {{2 -> .3}, Automatic}, Alignment -> {Automatic, Center}], 
		Framed[Style[ToString@failurecount, 11, Bold, White], Background -> RGBColor[0.74902, 0.403922, 0.4], FrameStyle -> RGBColor[0.74902, 0.403922, 0.4], FrameMargins -> {{8, 8}, {0, 0}}]},
		{"", "", Style[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "Errors-Label"], 11, RGBColor[0.34902, 0.34902, 0.34902]], Framed[Style[ToString@errorcount, 11, Bold, White], Background -> RGBColor[0.945, 0.81, 0.314], 
															FrameStyle -> RGBColor[0.945, 0.81, 0.314], FrameMargins -> {{8, 8}, {0, 0}}], 
		Grid[{{Style[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "Failures-Label"], 11, RGBColor[0.34902, 0.34902, 0.34902]],
			Pane[Style[Grid[{{Style["(", 8], Style[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "Messages-Label"], 10], Style[")", 8]}}, Spacings -> 0],
												RGBColor[0.521569, 0.521569, 0.521569]], ImageMargins -> {{Automatic, Automatic}, {-.3, Automatic}}]}},
			Spacings -> {{2 -> .3}, Automatic}, Alignment -> {Automatic, Center}], 
		Framed[Style[ToString@messagefailurecount, 11, Bold, White], Background -> RGBColor[0.921569, 0.678431, 0.337255], FrameStyle -> RGBColor[0.921569, 0.678431, 0.337255], 
			FrameMargins -> {{8, 8}, {0, 0}}]}}, Alignment -> Left, BaseStyle -> {FontFamily -> "Helvetica", FontSize -> 12}, Spacings -> {{2 -> .7, 3 -> 1.5, 4 -> .7, 5 -> 1.5, 6 -> .7}, .5}]
(*      
actionMenuResultColorBarAndButton[successPositions_, errorPositions_, failurePositions_, messagePositions_, totalLengthMultiplier_, barheight_: 10] :=
	DynamicModule[{$barDisplayType="InSequence"}, 
		Grid[{{Dynamic[MUnit`resultColorBar[successPositions, errorPositions, failurePositions, messagePositions, totalLengthMultiplier, barheight]],
			ActionMenu[Mouseover[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitExpressions", "Arrow-Off"], 
						Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitExpressions", "Arrow-Hover"]],
				{Dynamic[If[$barDisplayType === "ByStatus",
					Grid[{{"\[Checkmark]", "by status"}}],
					Grid[{{Spacer[10], "by status"}}]]] :> ($barDisplayType = "ByStatus"), 
				Dynamic[If[$barDisplayType === "InSequence",
					Grid[{{"\[Checkmark]", "ordered in sequence"}}],
					Grid[{{Spacer[10], "ordered in sequence"}}]]] :> ($barDisplayType = "InSequence")},
				Method -> "Queued", 
				Appearance -> None, 
				ContentPadding -> False
			]},
			{Grid[{{Button[Tooltip[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitExpressions", "PreviousFailure"],
						Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "PreviousFailure-Tooltip"], TooltipDelay -> .35], 
					MUnit`findFailedTest[ButtonNotebook[], "Previous"], Appearance -> FEPrivate`FrontEndResource["MUnitExpressions", "ButtonAppearances"],
					ImageSize -> {Automatic, 28}, FrameMargins -> {{7, 7}, {0, 0}}], 
				Button["Failure", ImageSize -> {Automatic, 28}, Enabled -> False, Appearance -> FEPrivate`FrontEndResource["MUnitExpressions", "DefaultButtonAppearance"],
					FrameMargins -> {{7, 7}, {0, 0}}], 
				Button[Tooltip[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitExpressions", "NextFailure"],
						Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "NextFailure-Tooltip"], TooltipDelay -> .35], 
					MUnit`findFailedTest[ButtonNotebook[], "Next"], Appearance -> FEPrivate`FrontEndResource["MUnitExpressions", "ButtonAppearances"],
					ImageSize -> {Automatic, 28}, FrameMargins -> {{7, 7}, {0, 0}}]}},
				Spacings -> {{1 -> 0, 2 -> -.1, 3 -> -.1}, Automatic},
				BaseStyle -> {"DialogStyle", Bold, FontColor -> Darker[Gray]}],
			""}},
		Alignment -> {{Right, Left}, Automatic}, Spacings -> {{2->.5}, {2 -> .75}}
		],
		Initialization:>
		(MUnit`resultColorBar[successPositions1_, errorPositions1_, failurePositions1_, messagePositions1_, totalLengthMultiplier1_, barheight1_: 10] := 
			Module[{
				total = Length@Union[successPositions1, errorPositions1, failurePositions1, messagePositions1], 
				sp = If[$barDisplayType === "ByStatus",
					SplitBy[#, Last],
					SplitBy[SortBy[#, First], Last]] &[Join[{#, "s"} & /@ successPositions1, {#, "e"} & /@ errorPositions1,
										{#, "f"} & /@ failurePositions1, {#, "m"} & /@ messagePositions1]]
				}, 
				Graphics[
					If[total > 50, 
						#
						, 
						Join[#,{White, Sequence @@ Table[Rectangle[{N[i/total] totalLengthMultiplier1, 0},{(N[i/total] totalLengthMultiplier1 + .1), 1}],{i, total - 1}]}]
					]&
					[Join @@ 
						(Thread[List[
							Partition[Prepend[Accumulate[Length /@ sp], 0], 2, 1] /. i_Integer :> N[i/total], 
							Switch[#[[1, 2]], "s", "Green", "e", "Yellow", "m", "Orange", _, "Red"] & /@ sp]
						] 
						/. {{a_, b_}, c_String} :> 
							{Switch[c, "Green", RGBColor[0.380392, 0.603922, 0.384314], "Yellow", RGBColor[0.945, 0.81, 0.314], "Orange", RGBColor[0.921569, 0.678431, 0.337255],
								"Red", RGBColor[0.74902, 0.403922, 0.4]], 
								Rectangle[{a totalLengthMultiplier1, 0}, {b totalLengthMultiplier1, 1}]}
						)
					], 
					ImageSize -> {totalLengthMultiplier1 barheight1, barheight1}, 
					ImagePadding -> None, 
					PlotRangePadding -> None
				]
			]
		)
	]
*)

actionMenuResultColorBarAndButton[cellids_, successPositions_, errorPositions_, failurePositions_, messagePositions_, totalLengthMultiplier_, barheight_: 10] :=
	DynamicModule[{$barDisplayType="InSequence",resultColorBar}, 
		Grid[{{Dynamic[resultColorBar[cellids, successPositions, errorPositions, failurePositions, messagePositions, totalLengthMultiplier, barheight]],
			If[cellids === {},
				Unevaluated[Sequence[]],
				ActionMenu[Mouseover[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitExpressions", "Arrow-Off"], 
							Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitExpressions", "Arrow-Hover"]],
					{Dynamic[If[$barDisplayType === "ByStatus",
						Grid[{{"\[Checkmark]", #}}],
						Grid[{{Spacer[10], #}}]]&[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "ByStatus-Label"]]] :> ($barDisplayType = "ByStatus"), 
					Dynamic[If[$barDisplayType === "InSequence",
						Grid[{{"\[Checkmark]", #}}],
						Grid[{{Spacer[10], #}}]]&[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "BySequence-Label"]]] :> ($barDisplayType = "InSequence")},
					Method -> "Queued", 
					Appearance -> None, 
					ContentPadding -> False]]},
			{PaneSelector[{True -> Grid[{{Button[Tooltip[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitExpressions", "PreviousFailure"], 
									Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "PreviousFailure-Tooltip"], TooltipDelay -> .35], 
								Block[{$ContextPath}, Needs["MUnit`"]; MUnit`findFailedTest[ButtonNotebook[], "Previous"]], 
								Appearance -> FEPrivate`FrontEndResource["MUnitExpressions", "ButtonAppearances"], ImageSize -> {Automatic, 28}, 
								FrameMargins -> {{7, 7}, {0, 0}}, 
								Enabled -> FEPrivate`SameQ[FrontEnd`CurrentValue[FrontEnd`ButtonNotebook[], {TaggingRules, "$someTestsFailed"}, False], True],
								Method -> "Queued"], 
							Button[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "Failure-Label"], ImageSize -> {Automatic, 28}, Enabled -> False, 
								Appearance -> FEPrivate`FrontEndResource["MUnitExpressions", "DefaultButtonAppearance"], FrameMargins -> {{7, 7}, {0, 0}}], 
							Button[Tooltip[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitExpressions", "NextFailure"], 
								Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "NextFailure-Tooltip"], TooltipDelay -> .35], 
								Block[{$ContextPath}, Needs["MUnit`"]; MUnit`findFailedTest[ButtonNotebook[], "Next"]], 
								Appearance -> FEPrivate`FrontEndResource["MUnitExpressions", "ButtonAppearances"], ImageSize -> {Automatic, 28}, 
								FrameMargins -> {{7, 7}, {0, 0}}, 
								Enabled -> FEPrivate`SameQ[FrontEnd`CurrentValue[FrontEnd`ButtonNotebook[], {TaggingRules, "$someTestsFailed"}, False], True],
								Method -> "Queued"]}}, 
						Spacings -> {{1 -> 0, 2 -> -.1, 3 -> -.1}, Automatic}, BaseStyle -> {"DialogStyle", Bold, FontColor -> Darker[Gray]}],
					False -> Grid[{{Button[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitExpressions", "PreviousFailureDeactivated"], 
								Appearance -> FEPrivate`FrontEndResource["MUnitExpressions", "ButtonAppearances"], ImageSize -> {Automatic, 28}, 
								FrameMargins -> {{7, 7}, {0, 0}}, 
								Enabled -> FEPrivate`SameQ[FrontEnd`CurrentValue[FrontEnd`ButtonNotebook[], {TaggingRules, "$someTestsFailed"}, False], True]], 
							Button[Style[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "Failure-Label"],GrayLevel[.7]], ImageSize -> {Automatic, 28}, 
								Enabled -> False, 
								Appearance -> FEPrivate`FrontEndResource["MUnitExpressions", "DefaultButtonAppearance"], FrameMargins -> {{7, 7}, {0, 0}}], 
							Button[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitExpressions", "NextFailureDeactivated"], 
								Appearance -> FEPrivate`FrontEndResource["MUnitExpressions", "ButtonAppearances"], ImageSize -> {Automatic, 28}, 
								FrameMargins -> {{7, 7}, {0, 0}}, 
								Enabled -> FEPrivate`SameQ[FrontEnd`CurrentValue[FrontEnd`ButtonNotebook[], {TaggingRules, "$someTestsFailed"}, False], True]]}}, 
						Spacings -> {{1 -> 0, 2 -> -.1, 3 -> -.1}, Automatic}, BaseStyle -> {"DialogStyle", Bold, FontColor -> Darker[Gray]}]}, 
				FrontEnd`CurrentValue[FrontEnd`ButtonNotebook[], {TaggingRules, "$someTestsFailed"}, False], ImageSize -> All],
			""}},
		Alignment -> {{Right, Left}, Automatic}, Spacings -> {{2->.5}, {2 -> .75}}
		],
		Initialization:>
		(
		resultColorBar[cellids1_, successPositions1_, errorPositions1_, failurePositions1_,messagePositions1_, barlength1_, barheight1_] := 
			Module[{successPositions2 = {#, Extract[cellids1, {#}]} & /@ successPositions1,
			        errorPositions2 = {#, Extract[cellids1, {#}]} & /@ errorPositions1,
				failurePositions2 = {#, Extract[cellids1, {#}]} & /@ failurePositions1, 
				messagePositions2 = {#, Extract[cellids1, {#}]} & /@ messagePositions1, sp, testnumber, buttonlength}, 
			sp = Cases[(If[$barDisplayType === "ByStatus", SplitBy[#, Last], SplitBy[SortBy[#, #[[1, 1]] &], Last]] &[Join[{#, "s"} & /@ successPositions2, {#, "e"} & /@ errorPositions2,
							{#, "f"} & /@ failurePositions2, {#, "m"} & /@ messagePositions2]]) /. {{a_, n_}, s_String} :> {{a, s}, n}, {{_, _String}, _}, {1, Infinity}]; 
			testnumber = Length@successPositions1 + Length@errorPositions1 + Length@failurePositions1 + Length@messagePositions1; 
			If[testnumber > 0, buttonlength = barlength1/N@testnumber];
			If[testnumber > 50, 
				Graphics[Raster[{sp /. {{n_, a_String}, b_} :> Switch[a, "s", {0.380392, 0.603922, 0.384314}, "e", {0.945, 0.81, 0.314}, "m", {0.921569, 0.678431, 0.337255},
																	"f", {0.74902, 0.403922, 0.4}]}, {{0, 0}, {barlength1 + 50, barheight1}}],
					ImageSize -> {barlength1 + 50, barheight1}, PlotRange -> {{0, barlength1 + 50}, {0, barheight1}}], 
				Grid[If[testnumber === 0,
					{{}},
					{Riffle[sp /. {{n_, a_String}, b_} :> (Button[Tooltip[Graphics[{Switch[a, "s", RGBColor[0.380392, 0.603922, 0.384314], "e", RGBColor[0.945, 0.81, 0.314],
														"m", RGBColor[0.921569, 0.678431, 0.337255], "f", RGBColor[0.74902, 0.403922, 0.4]],
						Rectangle[{0, 0}, {buttonlength, barheight1}]}, ImagePadding -> 0, PlotRangePadding -> 0, ImageSize -> {buttonlength, barheight1}], 
												"Test " <> ToString@n, TooltipDelay -> .35], 
											NotebookFind[ButtonNotebook[], b, All, CellID]; SelectionMove[ButtonNotebook[], All, CellGroup], 
										Appearance -> None, Method -> "Queued"]), 
						Graphics[{White, Rectangle[{0, 0}, {1, barheight1}]}, ImagePadding -> 0, PlotRangePadding -> 0, ImageSize -> {1, barheight1}]]}], 
					Alignment -> {Automatic, Center}, Spacings -> {0, 0}]]]
		)
	]
      
testResultsDockedCell[cellids_, tReport_TestReportObject]:= 
	Cell[BoxData[PaneBox[TagBox[GridBox[{{ToBoxes@reportGrid2by3[tReport["TestsSucceededCount"]+tReport["TestsFailedCount"], 
		tReport["TestsSucceededCount"], tReport["TestsFailedWithErrorsCount"], tReport["TestsFailedWrongResultsCount"], tReport["TestsFailedWithMessagesCount"]],
						ItemBox["",ItemSize->Fit,StripOnInput->False],
						ToBoxes@actionMenuResultColorBarAndButton[cellids, tReport["TestsSucceededIndices"], tReport["TestsFailedWithErrorsIndices"], 
												tReport["TestsFailedWrongResultsIndices"], tReport["TestsFailedWithMessagesIndices"], 250, 12]}},
			GridBoxAlignment -> {"Columns" -> {{Left}}, "Rows" -> {{Right}}}, 
			GridBoxItemSize -> {"Columns" -> {{Automatic}}, "Rows" -> {{Automatic}}}], "Grid"], FrameMargins -> {{26, 12}, {Automatic, Automatic}}, ImageSize -> Full, BaseStyle -> {Background -> RGBColor[0.827, 0.827, 0.827]}]], Background -> RGBColor[0.827, 0.827, 0.827],"DockedCell",CellTags -> "MUnitResultsCell"]
			
testResultsDockedCell[cellids_, successindices_, errorindices_, failureindices_, messageindices_]:=
	Cell[BoxData[PaneBox[TagBox[GridBox[{{ToBoxes@reportGrid2by3[Length@successindices + Length@errorindices + Length@failureindices + Length@messageindices, 
		Length@successindices, Length@errorindices, Length@failureindices, Length@messageindices],
						ItemBox["",ItemSize->Fit,StripOnInput->False],
						ToBoxes@actionMenuResultColorBarAndButton[cellids, successindices, errorindices, failureindices, messageindices, 250, 12]}},
			GridBoxAlignment -> {"Columns" -> {{Left}}, "Rows" -> {{Right}}}, 
			GridBoxItemSize -> {"Columns" -> {{Automatic}}, "Rows" -> {{Automatic}}}], "Grid"], FrameMargins -> {{26, 12}, {Automatic, Automatic}}, ImageSize -> Full, BaseStyle -> {Background -> RGBColor[0.827, 0.827, 0.827]}]], Background -> RGBColor[0.827, 0.827, 0.827],"DockedCell",CellTags -> "MUnitResultsCell"]


buttonConvertCellGroup[nb_NotebookObject] :=
	Module[{cc, selection},
		(* get-and-put to add any CellIDs that need to be added *)
		selection = NotebookRead[nb];
		cc = convertCells[selection];
		If[{cc} === {}, 
			MessageDialog["Cells with symbols in the list given by Names[\"*Form\"] or Print with the exception of CapForm, EdgeForm, ExportForm, FaceForm, \
HoldForm, HornerForm, JoinForm, PrecedenceForm, RealBlockDiagonalForm and ResponseForm cannot be converted into tests.", WindowSize -> {400, 140}], 
			NotebookWrite[nb, cc, AutoScroll -> False]];
	]
	
convertNotebookToTest[nb_NotebookObject,extension_:"NB"] :=
    Module[ {},
    (*Select all cells, convert them, and apply stylesheet*)
        SelectionMove[nb,All,Notebook];
        MUnit`buttonConvertCellGroup[nb];
        SetOptions[nb,StyleDefinitions->FrontEnd`FileName[{"MUnit"}, "MUnit.nb"]];
        If[ extension==="WLT",
            MUnit`PaletteSaveAs[nb]
        ]
    ]
    
convertNotebookToTest[nbLocation_String,opts___] :=
    Module[ {},
        If[ FileExistsQ[nbLocation]===True,
            If[ FileExtension[nbLocation]==="nb",
                convertNotebookToTest[NotebookPut@Import[nbLocation,"NB"],opts],
                Message[Import::fmterr,nbLocation]
            ],
            Message[General::nffil,nbLocation]
        ]
    ]

$formCheck = True
	
$exclusionForms = Complement[Append[Names["*Form"], "Print"], {"CapForm", "EdgeForm", "ExportForm", "FaceForm", "HoldForm", "HornerForm", "JoinForm", "PrecedenceForm", "RealBlockDiagonalForm", "ResponseForm"}]

PaletteConvertNotebook[nb_NotebookObject] := 
	Module[{gt,convertedcells,nb1},
		gt = NotebookGet[nb];
		convertedcells = convertCells[gt];
		If[Not@MatchQ[convertedcells, Notebook[{}, ___]],
			nb1=CreateDocument[];
			SetOptions[nb1, StyleDefinitions -> FrontEnd`FileName[{"MUnit"}, "MUnit.nb"]];
			NotebookWrite[nb1,convertedcells];
			SelectionMove[nb1,After,CellGroup, AutoScroll -> False]];
		If[($formCheck=!=False)&&(Cases[gt, Cell[val_ /; Not@FreeQ[DeleteCases[val, RowBox[{"(*", __, "*)"}], Infinity], Alternatives @@ $exclusionForms], "Input", ___], Infinity] =!= {}),
			MessageDialog["Cells with symbols in the list given by Names[\"*Form\"] or Print with the exception of CapForm, EdgeForm, \
ExportForm, FaceForm, HoldForm, HornerForm, JoinForm, PrecedenceForm, RealBlockDiagonalForm and ResponseForm cannot be converted into tests.", WindowSize -> {400, 140}]]
	]
	
convertCells[cells_]:= 
	Module[{cells1},
		cells1 = If[($formCheck=!=False),
				DeleteCases[cells, Alternatives[CellGroupData[{Cell[val_ /; Not@FreeQ[DeleteCases[val, RowBox[{"(*", __, "*)"}], Infinity], Alternatives @@ $exclusionForms], "Input", ___], __}, ___],
							Cell[val_ /; Not@FreeQ[DeleteCases[val, RowBox[{"(*", __, "*)"}], Infinity], Alternatives @@ $exclusionForms], "Input", ___]], Infinity],
				cells];
		(cells1/.{
		CellGroupData[{
			Cell[val1_,"Input",opts1___], 
			Cell[val2_,"Output",opts2___]}, opts3___]:> 
		CellGroupData[{
			Cell[val1,"VerificationTest",opts1], 
			Cell[val2,"ExpectedOutput",opts2], 
			Cell[BoxData[ToBoxes@bottomCell[]],"BottomCell"]}, opts3]
		,
		msgGroup:CellGroupData[{
			Cell[val1_,"Input",opts1___], 
			Cell[_,"Message",___]..,
			Cell[val3_,"Output",opts3___]}, opts4___]:> 
		CellGroupData[{
			Cell[val1,"VerificationTest",opts1], 
			Cell[val3,"ExpectedOutput",opts3],
			Cell[BoxData[If[Length@# === 1, RowBox[{"{", #[[1]], "}"}], RowBox[{"{", RowBox[Riffle[#, ","]], "}"}]] &[Cases[msgGroup, TemplateBox[{__}, "MessageTemplate"],
			Infinity] /. TemplateBox[{a_, b_, __}, "MessageTemplate"] :> RowBox[{a, "::", b}]]], "ExpectedMessage"],
			Cell[BoxData[ToBoxes@bottomCell[]],"BottomCell"]}, opts4]
		,
		Cell[val1_,"Input",opts1___]:> 
		CellGroupData[{
			Cell[val1,"VerificationTest",opts1], 
			Cell[BoxData[""],"ExpectedOutput"], 
			Cell[BoxData[ToBoxes@bottomCell[]],"BottomCell"]}, Open]
		,
		Cell[val1_,"Output",opts1___]:> 
		CellGroupData[{
			Cell[BoxData[""],"VerificationTest"], 
			Cell[val1,"ExpectedOutput",opts1], 
			Cell[BoxData[ToBoxes@bottomCell[]],"BottomCell"]}, Open]
		})/.(Cell[_, "Print" | "Output", ___]|Cell[])->Sequence[]
	]
	
(*
PaletteAddTestIDs[nb_] := 
	Module[{bottomCellIDs = CurrentValue[#, CellID] & /@ Cells[nb, CellStyle -> "BottomCell"], cellids, pairs, to, li1, ba, li2, pairs2, to2, li3, li4, re,
		optionsCellPattern = Alternatives[Cell[BoxData[RowBox[{"{", "}"}]], "TestOptions", __],
							Cell[BoxData[RowBox[{"{", RowBox[{__, "\[Rule]", __}], "}"}]], "TestOptions", __],
							Cell[BoxData[RowBox[{"{", RowBox[{RowBox[{__, "\[Rule]", __}], ",", RowBox[{__, "\[Rule]", __}]}], "}"}]], "TestOptions", __],
					Cell[BoxData[RowBox[{"{", RowBox[{RowBox[{__, "\[Rule]", __}], ",", RowBox[{__, "\[Rule]", __}], ",", RowBox[{__, "\[Rule]", __}]}], "}"}]], "TestOptions", __]]},
		If[bottomCellIDs =!= {},
			CurrentValue[nb, Deployed] = True;
			CurrentValue[nb, ShowSelection] = False;
			cellids = CurrentValue[#, CellID] & /@ Cells[nb];
			pairs = Cases[Transpose[{# - 1, #}] &[Flatten[Position[cellids, #] & /@ bottomCellIDs]], {a_ /; a > 0, _}];
			to = Last /@ Select[pairs, If[# === {}, False, ("Style" /. Developer`CellInformation[#[[1]]]) === "TestOptions"] &[Cells[nb, CellID -> cellids[[#[[1]]]]]] &];
			li1 = {cellids[[#]], "TestOptions"} & /@ to;
			ba = Last /@ Select[pairs, If[# === {},
							False,
							Not@MemberQ[{"TestOptions", "TestSuccess", "TestFailure", "TestMessageFailure", "TestError"},
									("Style" /. Developer`CellInformation[#[[1]]])]] &[Cells[nb, CellID -> cellids[[#[[1]]]]]] &];
			li2 = {cellids[[#]], None} & /@ ba;
			pairs2 = Select[Select[pairs, Not@MemberQ[Union[to, ba], #[[2]]] &], #[[1]] > 1 &] - 1;
			to2 = Last /@ Select[pairs2, If[# === {}, False, ("Style" /. Developer`CellInformation[#[[1]]]) === "TestOptions"] &[Cells[nb, CellID -> cellids[[#[[1]]]]]] &];
 			li3 = {cellids[[#]], "TestOptions"} & /@ to2;
			li4 = {cellids[[#]], None} & /@ (Last /@ Select[pairs2, Not@MemberQ[to2, #[[2]]] &]);
 			SelectionMove[nb, Before, Notebook, AutoScroll -> False];
 			If[#[[2]] === "TestOptions",
 				NotebookFind[nb, #[[1]], Next, CellID, AutoScroll -> False];
 				SelectionMove[nb, Previous, Cell, AutoScroll -> False];
 				re = NotebookRead[nb];
 				If[MatchQ[re, optionsCellPattern],
					NotebookWrite[nb, re /. Cell[val_, "TestOptions", opts___] :> If[(MemberQ[ToExpression[val], TestID] || MemberQ[Part[ToExpression[val], All, 1], TestID]), 
														Cell[val, "TestOptions", opts], 
														If[Head[ToExpression[val]] === List, 
												Cell[BoxData[ToBoxes[Append[ToExpression[val], TestID -> ToString@CreateUUID[]]]], "TestOptions", opts],
								Cell[BoxData[ToBoxes[Append[List@ToExpression[val], TestID -> ToString@CreateUUID[]]]], "TestOptions", opts]]], AutoScroll -> False]],
				NotebookFind[nb, #[[1]], Next, CellID, AutoScroll -> False];
				SelectionMove[nb, Before, Cell, AutoScroll -> False];
				NotebookWrite[nb, Cell[BoxData[RowBox[{"{", RowBox[{"TestID", "\[Rule]", ToBoxes@ToString@CreateUUID[]}],"}"}]], "TestOptions"],
						AutoScroll -> False]] & /@ Sort[Join[li1, li2, li3, li4], OrderedQ[{Position[cellids, #[[1]]][[1, 1]], Position[cellids, #2[[1]]][[1, 1]]}] &];
			CurrentValue[nb, Deployed] = Inherited;
			CurrentValue[nb, ShowSelection] = Inherited]]
*)

PaletteAddTestIDs[nb_] := 
	Module[{bottomCellIDs = CurrentValue[#, CellID] & /@ Cells[nb, CellStyle -> "BottomCell"], re,
		optionsCellPattern = Alternatives[Cell[BoxData[RowBox[{"{", "}"}]], "TestOptions", __],
							Cell[BoxData[RowBox[{"{", RowBox[{__, "\[Rule]", __}], "}"}]], "TestOptions", __],
							Cell[BoxData[RowBox[{"{", RowBox[{RowBox[{__, "\[Rule]", __}], ",", RowBox[{__, "\[Rule]", __}]}], "}"}]], "TestOptions", __],
					Cell[BoxData[RowBox[{"{", RowBox[{RowBox[{__, "\[Rule]", __}], ",", RowBox[{__, "\[Rule]", __}], ",", RowBox[{__, "\[Rule]", __}]}], "}"}]], "TestOptions", __]]},
		If[bottomCellIDs =!= {},
			CurrentValue[nb, ShowSelection] = False;
			SelectionMove[nb, All, Notebook, AutoScroll -> False];
			CurrentValue[nb, Deployed] = True;
			re = NotebookRead[nb];
			NotebookWrite[nb, 
					re /. cg : Cell[CellGroupData[{Cell[__] .., Cell[_, "BottomCell", _]}, _]] :> If[FreeQ[cg, Cell[_, "TestOptions", __]],
				If[FreeQ[cg, Cell[_, "TestSuccess" | "TestFailure" | "TestMessageFailure" | "TestError", __]], 
					cg /. Cell[a_, "BottomCell", b__] :> Sequence[Cell[BoxData[RowBox[{"{", RowBox[{"TestID", "\[Rule]", ToBoxes@ToString@CreateUUID[]}], "}"}]], "TestOptions"],
											Cell[a, "BottomCell", b]], 
					cg /. Cell[a_, b : ("TestSuccess" | "TestFailure" | "TestMessageFailure" | "TestError"), c__] :> Sequence[Cell[BoxData[RowBox[{"{", RowBox[{"TestID", "\[Rule]", 
																		ToBoxes@ToString@CreateUUID[]}], "}"}]], "TestOptions"],
																			Cell[a, b, c]]],
				cg /. ce : Cell[val_, "TestOptions", opts___] :> If[MatchQ[ce, optionsCellPattern],
											If[(MemberQ[ToExpression[val], TestID] || MemberQ[Part[ToExpression[val], All, 1], TestID]), 
												Cell[val, "TestOptions", opts], 
												If[Head[ToExpression[val]] === List, 
													Cell[BoxData[ToBoxes[Append[ToExpression[val], TestID -> ToString@CreateUUID[]]]],
														"TestOptions", opts],
													Cell[BoxData[ToBoxes[Append[List@ToExpression[val], TestID -> ToString@CreateUUID[]]]],
														"TestOptions", opts]]],
											ce]],
					AutoScroll -> False];
			CurrentValue[nb, Deployed] = Inherited;
			CurrentValue[nb, ShowSelection] = Inherited]]			

PaletteNewTestNotebook[] := 
	NotebookPut[Notebook[{},StyleDefinitions->FrontEnd`FileName[{"MUnit"},"MUnit.nb",CharacterEncoding->"UTF-8"]]]

PaletteRun[nb_NotebookObject] :=
	Module[{gt, presetMessageOptionsValues = (MessageOptions /. Options[$FrontEnd, MessageOptions]), newMessageOptionsValues, cs, resultcells, cellids, styleList},
		If[Cases[$ProductInformation, _["ProductIDName", "WolframCDFPlayer"]] === {},
		$windowtitle = AbsoluteCurrentValue[nb, WindowTitle];
		(*$windowtitle = StringReplace[$windowtitle, ".nb"~~EndOfString -> ""];*)
		CurrentValue[nb, ShowSelection] = False;
		CurrentValue[nb, WindowTitle] = "Running... " <> $windowtitle;
		clearTestResults[IncludeInSuperFunction -> True];
		SetOptions[nb, Deployed -> True];
		gt = NotebookGet[nb];
		Catch[If[($formCheck=!=False)&&(Cases[DeleteCases[gt[[1]], RowBox[{"(*", __, "*)"}], Infinity], Alternatives @@ MUnit`$exclusionForms, Infinity] =!= {}), 
			Throw[CurrentValue[nb, WindowTitle] = If[Cases[NotebookInformation[nb], _["FileName", _]]==={},
								Inherited,
								$windowtitle];
				SetOptions[nb, Deployed -> Inherited];
				CurrentValue[nb, ShowSelection] = Inherited;
				MessageDialog["Expressions with symbols in the list given by Names[\"*Form\"] or Print with the exception of CapForm, EdgeForm, ExportForm, FaceForm, HoldForm, HornerForm, JoinForm, PrecedenceForm, RealBlockDiagonalForm and ResponseForm cannot be used in tests.", WindowFrame -> "ModalDialog"]]];
		identifyIsolatedCells[];
		If[($problemCellids =!= {}) && ($problemCellids =!= $Failed)
			,
			CurrentValue[nb, DockedCells] = List[CurrentValue[InputNotebook[], DockedCells], defectiveCellsDockedCell[$problemCellids]];
			MUnit`colorCellBrackets[nb];
			CurrentValue[nb, WindowTitle] = If[Cases[NotebookInformation[nb], _["FileName", _]]==={}, Inherited, $windowtitle]
			,
			newMessageOptionsValues = If[(cs = Cases[presetMessageOptionsValues, a : ("KernelMessageAction" -> _)]) === {}, 
						Append[presetMessageOptionsValues, "KernelMessageAction" -> {"Beep", "PrintToNotebook"}], 
						presetMessageOptionsValues /. ("KernelMessageAction" -> a_) :> ("KernelMessageAction" -> If[StringQ[a],
																	"PrintToConsole", 
																	Append[DeleteCases[a, "PrintToNotebook"], "PrintToConsole"]])];
			SetOptions[$FrontEnd, MessageOptions -> newMessageOptionsValues];
			TestRun[nb, TestRunTitle -> $windowtitle, Loggers -> {AssociationNBLogger[nb]}];
			SelectionMove[nb, After, Cell];
			If[(resultcells = Cells[nb, CellStyle -> ("TestSuccess" | "TestError" | "TestFailure" | "TestMessageFailure")]) =!= {}, 
				cellids = CurrentValue[#, CellID] & /@ resultcells; 
				styleList = ("Style" /. (Developer`CellInformation /@ resultcells));
				CurrentValue[nb, DockedCells] = List[CurrentValue[nb, DockedCells],
									testResultsDockedCell[cellids, Flatten@Position[styleList, "TestSuccess"], Flatten@Position[styleList, "TestError"], 
												Flatten@Position[styleList, "TestFailure"], Flatten@Position[styleList, "TestMessageFailure"]]]]
		];
		SetOptions[nb, Deployed -> Inherited];
		CurrentValue[nb, ShowSelection] = Inherited;
		CurrentValue[nb, WindowTitle] = If[Cases[NotebookInformation[nb], _["FileName", _]]==={},
							Inherited,
							$windowtitle];
		SetOptions[$FrontEnd, MessageOptions -> presetMessageOptionsValues]
	], MessageDialog["The Run button cannot be used in this product."]]]
	
addMessages[]:=
	Module[{bn = ButtonNotebook[], re},
		CurrentValue[bn, ShowSelection] = False;
		SelectionMove[bn, All, ButtonCell, AutoScroll -> False];
		SelectionMove[bn, All, Cell, AutoScroll -> False];
		SelectionMove[bn, All, CellGroup, AutoScroll -> False];
		re = NotebookRead[bn];
		Which[(*The test cell group does not begin with a test input cell followed by a test ouput cell. *)
			Not@MatchQ[re,
				Cell[CellGroupData[{Cell[_, "VerificationTest", ___],
				Cell[_, "ExpectedOutput", ___], ___,
				Cell[_, "BottomCell", ___]}, _]]]
			,
			SelectionMove[bn, Before, CellGroup, AutoScroll -> False];
			NotebookFind[bn, "BottomCell", Next, CellStyle, AutoScroll -> False];
			MessageDialog["Malformed Test"],
			(* The cell group does not have a message input cell. *)
			Not@MatchQ[re,
				Cell[CellGroupData[{Cell[_, "VerificationTest", ___],
				Cell[_, "ExpectedOutput", ___],
				___,
				Cell[_, "ExpectedMessage", ___],
				___}, _]]]
			,
			SelectionMove[bn, Before, CellGroup, AutoScroll -> False];
			If[Cases[re,Cell[_,"ActualOutput",___],Infinity]==={},
				(* No ActualOutput cell*)
				NotebookFind[bn, "ExpectedOutput", Next, CellStyle, AutoScroll -> False];
				,
				NotebookFind[bn, "ActualOutput", Next, CellStyle, AutoScroll -> False];
			];
			SelectionMove[bn, After, Cell, AutoScroll -> False];
			NotebookWrite[bn, Cell[BoxData[RowBox[{"{", "}"}]], "ExpectedMessage"], All, AutoScroll -> False];
			SelectionMove[bn,Before,CellContents,AutoScroll->False];
			SelectionMove[bn,Next,Character,AutoScroll->False];
			, 
			True
			,
			SelectionMove[bn, Before, CellGroup, AutoScroll -> False];
			NotebookFind[bn, "ExpectedMessage", Next, CellStyle, AutoScroll -> False];
			SelectionMove[bn, After, CellContents, AutoScroll -> False];
		];
		CurrentValue[bn, ShowSelection] = Inherited
	]
	
SetAttributes[buttonWithIcon,HoldAll]

buttonWithIcon[iconName_, label_, bfunc_]:=
	Button[Grid[{{Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitExpressions", iconName],
			Style[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", label], "ButtonText", AdjustmentBoxOptions -> {BoxBaselineShift -> -50}]}}, Alignment -> {Automatic, Center}],
		bfunc,
 		Appearance -> FEPrivate`FrontEndResource["MUnitExpressions", "ButtonAppearances"],
 		FrameMargins -> {{10, 10}, {0, 0}},
 		Method -> "Queued",
 		ImageSize -> Automatic]
 		
SetAttributes[buttonWithIconAndTooltip,HoldAll]

buttonWithIconAndTooltip[iconName_, label_, tooltip_, bfunc_, verticalAdjustment_:None] := 
	Button[Tooltip[Mouseover[Grid[{{If[verticalAdjustment === None, #, Pane[#, ImageMargins -> {{0, 0}, {verticalAdjustment, 0}}]],
					Style[#2, "ButtonText", AdjustmentBoxOptions -> {BoxBaselineShift -> -50}]}}, #3], 
				Grid[{{If[verticalAdjustment === None, #, Pane[#, ImageMargins -> {{0, 0}, {verticalAdjustment, 0}}]],
					Style[#2, "ButtonText", RGBColor[0.9059, 0.3451, 0.102]]}}, #3]] &[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitExpressions", iconName], 
														Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", label], 
														Alignment -> {Automatic, Center}], 
			Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", tooltip], TooltipDelay -> .5], bfunc, 
		Appearance -> FEPrivate`FrontEndResource["MUnitExpressions", "ButtonAppearances"], FrameMargins -> {{10, 10}, {0, 0}}, Method -> "Queued", ImageSize -> Automatic] 
 		
SetAttributes[buttonWithoutIcon,HoldAll]

buttonWithoutIcon[label_, bfunc_]:=
	Button[Mouseover[Style[#, "ButtonText", AdjustmentBoxOptions -> {BoxBaselineShift -> -50}],
			Style[#, "ButtonText", RGBColor[0.9059, 0.3451, 0.102], AdjustmentBoxOptions -> {BoxBaselineShift -> -50}]]&[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", label]],
		bfunc,
 		Appearance -> FEPrivate`FrontEndResource["MUnitExpressions", "ButtonAppearances"],
 		FrameMargins -> {{10, 10}, {0, 0}},
 		Method -> "Queued",
 		ImageSize -> Automatic]
 		
SetAttributes[buttonWithTooltipButWithoutIcon,HoldAll]

buttonWithTooltipButWithoutIcon[label_, tooltip_, bfunc_]:=
	Button[Tooltip[Mouseover[Style[#, "ButtonText", AdjustmentBoxOptions -> {BoxBaselineShift -> -50}],
			Style[#, "ButtonText", RGBColor[0.9059, 0.3451, 0.102], AdjustmentBoxOptions -> {BoxBaselineShift -> -50}]]&[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", label]],
			Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", tooltip], TooltipDelay->.5],
		bfunc,
 		Appearance -> FEPrivate`FrontEndResource["MUnitExpressions", "ButtonAppearances"],
 		FrameMargins -> {{10, 10}, {0, 0}},
 		Method -> "Queued",
 		ImageSize -> Automatic] 
 		
btnMessages[] := buttonWithIconAndTooltip["Add", "AddMessages-Label", "AddMessages-Tooltip", Block[{$ContextPath}, Needs["MUnit`"]; MUnit`addMessages[]]]

addOptions[]:=
	Module[{bn=ButtonNotebook[],re,case},
		CurrentValue[bn,ShowSelection]=False;
		SelectionMove[bn,All,ButtonCell,AutoScroll->False];
		SelectionMove[bn,All,Cell,AutoScroll->False];
		SelectionMove[bn,All,CellGroup,AutoScroll->False];
		re=NotebookRead[bn];
		Which[(*The test cell group does not begin with a test input cell followed by a test ouput cell.*)
			Not@MatchQ[re,
				Cell[CellGroupData[{Cell[_,"VerificationTest",___],
				Cell[_,"ExpectedOutput",___],
				___,
				Cell[_,"BottomCell",___]},_]]]
			,
			SelectionMove[bn,Before,CellGroup,AutoScroll->False];
			NotebookFind[bn,"BottomCell",Next,CellStyle,AutoScroll->False];
			MessageDialog["Malformed test"]
			,
			(*The cell group does not have a options input cell.*)
			Not@MatchQ[re,
				Cell[CellGroupData[{Cell[_,"VerificationTest",___],
					Cell[_,"ExpectedOutput",___],
					___,
					Cell[_,"TestOptions",___],
					___,
					Cell[_,"BottomCell",___]},_]]
				]
			,
			SelectionMove[bn,Before,CellGroup,AutoScroll->False];
			If[(case=Cases[re,Cell[_,q:"TestSuccess"|"TestFailure"|"TestMessageFailure"|"TestError",___]:>q,Infinity])==={},
				(* No ResultCell*)
				NotebookFind[bn, "BottomCell", Next, CellStyle, AutoScroll -> False];
				,
				NotebookFind[bn, First@case, Next, CellStyle, AutoScroll -> False];
			];
			SelectionMove[bn,Before,Cell,AutoScroll->False];
			NotebookWrite[bn,Cell[BoxData[RowBox[{"{", "}"}]],"TestOptions"],All,AutoScroll->False];
			SelectionMove[bn,Before,CellContents,AutoScroll->False];
			SelectionMove[bn,Next,Character,AutoScroll->False];
			,
			True
			,
			SelectionMove[bn,Before,CellGroup,AutoScroll->False];
			NotebookFind[bn, "TestOptions", Next, CellStyle, AutoScroll -> False];
			SelectionMove[bn, After, CellContents, AutoScroll -> False];
		];
	CurrentValue[bn,ShowSelection]=Inherited
	]
	
btnOptions[] := buttonWithIconAndTooltip["Add", "AddOptions-Label", "AddOptions-Tooltip", Block[{$ContextPath}, Needs["MUnit`"]; MUnit`addOptions[]]]

optionsInsertionGrid[] := Grid[{{btnOptions[], ActionMenu[Button["", 
					ContentPadding -> False, Appearance -> FEPrivate`FrontEndResource["MUnitExpressions", "ButtonAppearancesNoLeftBorder"]],
					{Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "MemoryConstraint-Label"] :> Block[{$ContextPath},
																	Needs["MUnit`"]; MUnit`InsertOptionRule["MemoryConstraint"]], 
					Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "SameTest-Label"] :> Block[{$ContextPath}, 
																	Needs["MUnit`"]; MUnit`InsertOptionRule["SameTest"]], 
					Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "TestID-Label"] :> Block[{$ContextPath}, 
																	Needs["MUnit`"]; MUnit`InsertOptionRule["TestID"]], 
					Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "TimeConstraint-Label"] :> Block[{$ContextPath}, 
																	Needs["MUnit`"]; MUnit`InsertOptionRule["TimeConstraint"]]},
				Method -> "Queued",
				Appearance -> None]}},
				GridBoxSpacings -> {"Columns" -> {{0}}, "Rows" -> {{Automatic}}}]
				
writeOptionRule[nb_, optName_] := (NotebookWrite[nb, #] & /@ {optName, "\[Rule]", TagBox[FrameBox["\"value\""], "Placeholder"]}; FrontEndExecute[FrontEndToken[nb, "SelectPrevious"]])

InsertOptionRule[optName_] := 
 Module[{nb = InputNotebook[], ci}, 
	SelectionMove[nb, All, ButtonCell]; 
	SelectionMove[nb, Previous, Cell];
	ci = Developer`CellInformation[nb];
	If[MatchQ[ci, {{___, "Style" -> "TestFailure" | "TestMessageFailure" | "TestSuccess" | "TestError", ___}}],
		SelectionMove[nb, Previous, Cell];
		ci = Developer`CellInformation[nb]];
	Which[Not@MatchQ[Developer`CellInformation[nb], {{___, "Style" -> "TestOptions", ___}}] && ci =!= $Failed,
		SelectionMove[nb, After, Cell]; 
		NotebookWrite[nb, Cell[BoxData[RowBox[{"{", "}"}]], "TestOptions"], All, AutoScroll -> False],
		ci === $Failed, 
		NotebookWrite[nb, Cell[BoxData[RowBox[{"{", "}"}]], "TestOptions"], All, AutoScroll -> False]];
	If[FreeQ[NotebookRead[nb], optName],
		SelectionMove[nb, After, CellContents]; 
		FrontEndExecute[FrontEndToken[nb, "SelectPrevious"]];
		If[NotebookRead[nb] === "}"
			,
			FrontEndExecute[FrontEndToken[nb, "MovePrevious"]];
			FrontEndExecute[FrontEndToken[nb, "SelectPrevious"]];
			If[NotebookRead[nb] === "{"
				, 
				FrontEndExecute[FrontEndToken[nb, "MoveNext"]];
				writeOptionRule[nb, optName]
				,
				re = NotebookRead[nb]; 
				If[re =!= {}, FrontEndExecute[FrontEndToken[nb, "MoveNext"]]]; 
				If[re =!= "{", NotebookWrite[nb, ","]];
				writeOptionRule[nb, optName]]
			, 
			SelectionMove[nb, All, Cell];
			MessageDialog["The content of the cell does not appear to be correct for writing the option."]]]]

bottomCell[]:= Pane[
	Grid[{{btnMessages[], optionsInsertionGrid[]}},
		ItemSize->{Automatic,2},
		Alignment->{Automatic,Center},
		Spacings->{{2->.5,3->2,4->.5},Automatic}
	],
	Full,
	Alignment->{Left,Center}
]

bottomCellWithRightButton[buttons_List] := Cell[BoxData[ToBoxes[Pane[Grid[{{bottomCell[], Item["", ItemSize -> Full], Sequence @@ buttons}}(*,
                                                                          Alignment -> {Left, Right}, ItemSize -> Full*)], Full]]], "BottomCell"]

auxPaletteInsertTest[nb_NotebookObject] := 
	NotebookWrite[nb, 
			{Cell[BoxData[""], "VerificationTest"], 
			Cell[BoxData[""], "ExpectedOutput"], 
			Cell[BoxData[ToBoxes@MUnit`bottomCell[]], "BottomCell"]}, AutoScroll -> False]

PaletteInsertTest[nb_NotebookObject] := 
	Module[{ci = Developer`CellInformation[nb],ci2,ci3,cellid},
		Catch[
			If[(* Cursor between cells. *)
				ci === $Failed
				,
				SelectionMove[nb, Next, Cell, AutoScroll -> False];
				ci2 = Developer`CellInformation[nb];
				If[ci2 === $Failed, 
					Throw[auxPaletteInsertTest[nb]]
				];
				If[(* The next cell is a "VerificationTest" cell. *)
					MatchQ[ci2, {{___, "Style" -> "VerificationTest", ___}}]
					, 
					Throw[SelectionMove[nb, Before, Cell, AutoScroll -> False]; auxPaletteInsertTest[nb]]
				];
				FrontEndExecute[FrontEndToken[nb, "ExpandSelection"]];
				ci3 = Developer`CellInformation[nb];
				If[(* The next cell is part of a test group. *)
					MatchQ[ci3, {{___, "Style" -> "VerificationTest", ___, "FirstCellInGroup" -> True, ___}, __}]
					, 
					Throw[SelectionMove[nb, After, Cell, AutoScroll -> False]; auxPaletteInsertTest[nb]]
					, 
					NotebookFind[nb, ("CellID" /. ci2)[[1]], All, CellID, AutoScroll->False]; 
					Throw[SelectionMove[nb, Before, Cell, AutoScroll -> False]; auxPaletteInsertTest[nb]]
				]
			];
			If[FreeQ[ci, "CursorPosition" -> "CellBracket"],
				expandToCell[nb]
				,
				If[(* Selecting the cell bracket of a cell group. *)
					MatchQ[ci, {{___, "FirstCellInGroup" -> True, ___}, __}]
					,
					Throw[SelectionMove[nb, After, Cell, AutoScroll -> False];auxPaletteInsertTest[nb]]
				]
			]; 
			ci2 = Developer`CellInformation[nb]; 
			cellid = ("CellID" /. ci2)[[1]]; 
			FrontEndExecute[FrontEndToken[nb, "ExpandSelection"]]; 
			ci3 = Developer`CellInformation[nb];
			If[Not@MatchQ[ci3, {{___, "FirstCellInGroup" -> True, ___, "CellID" -> _, ___}, __}],
				NotebookFind[nb, cellid, All, CellID, AutoScroll->False]
			]; 
			SelectionMove[nb, After, Cell, AutoScroll -> False]; 
			auxPaletteInsertTest[nb]
		]
	]

PaletteInsertTestSection[nb_NotebookObject]:= Module[{},
	NotebookWrite[nb, Cell["TestSection", "TestSection"], AutoScroll -> False]
]
(* 
deleteResultCell[]:= Button[
	Graphics[{EdgeForm[Thin],White,Disk[],Black,Line[{{-Sqrt[2],-Sqrt[2]}/2,{Sqrt[2],Sqrt[2]}/2}],Line[{{Sqrt[2],-Sqrt[2]}/2,{-Sqrt[2],Sqrt[2]}/2}]},ImageSize->15]
	,
	SelectionMove[ButtonNotebook[],All,ButtonCell, AutoScroll -> False];
	NotebookDelete[];
	FrontEndExecute[{FrontEnd`SelectionMove[ButtonNotebook[],After,CellGroup]}];
	,
	Appearance->None
]*)


testResultData[testpairs_]:=
	Quiet[(testpairs/.{"TestID"->Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "TestID-Detail"],
		"ActualOutput"->Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "ActualOutput-Detail"],
		"ExpectedOutput"->Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "ExpectedOutput-Detail"],
		"ActualMessages"->Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "ActualMessages-Detail"],
		"ExpectedMessages"->Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "ExpectedMessages-Detail"],
		"CPUTimeUsed"->Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "CPUTimeUsed-Detail"]})
		/.{a_Dynamic,b_}:>Grid[{{Style[a, RGBColor[0.458824, 0.458824, 0.458824]], Style[StringReplace[ToString[b], 
			"seconds" -> "s"], RGBColor[0.14902, 0.14902, 0.14902]]}}, BaseStyle -> {FontFamily -> "Helvetica", FontWeight -> "Bold"}, Spacings -> {{2 -> .5}, Automatic}],Message::name
	]
(*
resultCellDataDisplay[tr_TestResultObject]:=
	Module[{testList=
		If[!(tr["ExpectedMessages"]=={}&&tr["ActualMessages"]=={}),
			{"TestID","ActualOutput","ExpectedOutput","ActualMessages","ExpectedMessages","CPUTimeUsed"},
			{"TestID","ActualOutput","ExpectedOutput","CPUTimeUsed"}],
		testPairs,
		resultType=tr["Outcome"],
		resultCellSuccessIcon=Graphics[{RGBColor[0.09,0.76,0.23],Disk[],White,Inset[Style["\[Checkmark]",Bold,12,White],{.3,-.1},Background->RGBColor[0.09,0.76,0.23]]},ImageSize->16],
		resultCellFailureIcon:=Graphics[{RGBColor[0.86,0.46,0.43],Thickness[.5],Line[{{-Sqrt[2],-Sqrt[2]}/2,{Sqrt[2],Sqrt[2]}/2}],Line[{{Sqrt[2],-Sqrt[2]}/2,{-Sqrt[2],Sqrt[2]}/2}]},ImageSize->11],
		resultCellMessageFailureIcon:=Graphics[{GrayLevel[.5],Thickness[.5],Line[{{-Sqrt[2],-Sqrt[2]}/2,{Sqrt[2],Sqrt[2]}/2}],Line[{{Sqrt[2],-Sqrt[2]}/2,{-Sqrt[2],Sqrt[2]}/2}]},ImageSize->11]},
		testPairs={#,StandardForm@Short[tr[#],2/3]}&/@testList;
		ActionMenu[
			Grid[{{Switch[resultType,
				"Success",resultCellSuccessIcon,
				"Failure",resultCellFailureIcon,
				_,resultCellMessageFailureIcon],
				Style[Switch[resultType,
					"Success","Success",
					"Failure","Failure",
					_,"Messages Failure"
					],13
				]}},
				Alignment->{Automatic,Center}
			],
		testResultData[testPairs]
		]
	]
*)
testRerun[]:=
	Module[{nb},
		Catch[
			nb = EvaluationNotebook[];
			If[nb === $Failed,
				Throw[{"Value" -> Null, "Messages" -> {"VerificationTest", "Internal Error: EvaluationNotebook[] returned $Failed"}}, MUnitErrorTag]
			];
		SelectionMove[nb, All, ButtonCell, AutoScroll->False];
		SelectionMove[nb, All, Cell, AutoScroll->False];
		NotebookFind[nb,"VerificationTest",Previous,CellStyle, AutoScroll->False];
		SelectionEvaluate[nb]
		]
	]

testReplaceOutput[tr_TestResultObject]:=
	Module[{nb, tbr, sel},
		Catch[
			nb = EvaluationNotebook[];
			If[nb === $Failed,
				Throw[{"Value" -> Null, "Messages" -> {"VerificationTest", "Internal Error: EvaluationNotebook[] returned $Failed"}}, MUnitErrorTag]
			];
		Quiet[tbr = ToBoxes[ReleaseHold@tr["ActualOutput"]]];
		If[Not@FreeQ[tbr, StyleBox[TagBox[_, "SummaryHead"], "NonInterpretableSummary"]],
			Throw@If[(sel=Select[Notebooks[], CurrentValue[#, {TaggingRules, "NonInterpretableSummary"}]===True&])==={},
				MessageDialog["The actual output contains insufficient information to interpret the result and so cannot be used in the Expected Output cell.",
						TaggingRules -> {"NonInterpretableSummary" -> True}],
				SetSelectedNotebook[sel[[1]]]]];
		SelectionMove[nb, All, ButtonCell, AutoScroll->False];
		SelectionMove[nb, All, Cell, AutoScroll->False];
		NotebookFind[nb,"ExpectedOutput",Previous,CellStyle, AutoScroll->False];
		Quiet[NotebookWrite[nb, NotebookRead[nb]/.Cell[val_,"ExpectedOutput",opts___] :> Cell[BoxData[tbr],"ExpectedOutput",opts], AutoScroll -> False]];
		SelectionMove[nb, After, CellGroup, AutoScroll->False];
		]
	]

testReplaceMessage[tr_TestResultObject]:=
	Module[{nb},
		Catch[
			nb = EvaluationNotebook[];
			If[nb === $Failed,
				Throw[{"Value" -> Null, "Messages" -> {"VerificationTest", "Internal Error: EvaluationNotebook[] returned $Failed"}}, MUnitErrorTag]
			];
		SelectionMove[nb, All, ButtonCell, AutoScroll->False];
		SelectionMove[nb, All, Cell, AutoScroll->False];
		NotebookFind[nb,"ExpectedOutput",Previous,CellStyle, AutoScroll->False];
		SelectionMove[nb,Next,Cell, AutoScroll->False];
		If[MatchQ[Developer`CellInformation[nb],{{___,"Style"->"ExpectedMessage",___}}],
			NotebookWrite[nb, NotebookRead[nb]/.Cell[val_,"ExpectedMessage",opts___] :> Cell[BoxData[actualMessagesBoxStructure[tr["ActualMessages"]]], "ExpectedMessage", opts],
																						AutoScroll -> False]
			,
			SelectionMove[nb,Before,Cell, AutoScroll->False];
			NotebookWrite[nb, Cell[BoxData[actualMessagesBoxStructure[tr["ActualMessages"]]], "ExpectedMessage"], AutoScroll -> False]
		]
		SelectionMove[nb,After,CellGroup, AutoScroll->False];
	]
]

resultCellFirstPart[tr_TestResultObject] := 
	Module[{resultType = tr["Outcome"]}, 
		Grid[{{Style[Switch[resultType,
					"Success",
					Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "Success-TestSuccess"],
					"Failure",
					Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "Failure-TestFailure"],
					"Error",
					Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "Error-TestError"],
					_,
					Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "MessageFailure-TestMessageFailure"]], White, Bold, 14, FontFamily->"Arial"],
			If[resultType === "Success",
				Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitExpressions", "SuccessCheck"],
				Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitExpressions", "FailureX"]]}}, Alignment -> {Automatic, Center}]]
				
resultCellDataDisplay[tr_TestResultObject] := 
	Module[{testList = If[tr["Outcome"]==="Error", {}, If[!(tr["ExpectedMessages"] == {} && tr["ActualMessages"] == {}),
				{"TestID", "ActualOutput", "ExpectedOutput", "ActualMessages", "ExpectedMessages", "CPUTimeUsed"},
				{"TestID", "ActualOutput", "ExpectedOutput", "CPUTimeUsed"}]], testPairs},
		If[testList === {}, {},
		testPairs = {#, If[MemberQ[{"ActualMessages", "ExpectedMessages"}, #],
					tr[#] /. HoldForm[Message[MessageName[a_, b_], ___]] :> ToString@a <> "::" <> b,
					StandardForm@Short[tr[#], 2/3]]} & /@ testList; 
		ActionMenu[Grid[{{Style[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "Details-Label"], 10, White, FontFamily -> "Helvetica"], Style["\[RightGuillemet]", 12, White]}},
				Alignment -> {Automatic, Center}, Spacings -> {2 -> .3}],
				testResultData[testPairs], Appearance -> None]]]
 
(*
resultCell[tr_TestResultObject]:=Module[{},
	Which[tr["Outcome"]=="Success", 
		Cell[
			BoxData[ToBoxes[Pane[Grid[{{"Result:",resultCellDataDisplay[tr],Item["",ItemSize -> Fit](*,
				Framed[Button["Rerun",testRerun[],Appearance->"Frameless"],FrameMargins->1,Background->White],deleteResultCell[]*)}},
				Alignment->{Left,Right},
				ItemSize->Full
				],
			Full
			]]],
			"TestSuccess"
		]
		,
		tr["Outcome"] =="Failure",
		Cell[
			BoxData[ToBoxes[Pane[Grid[{{"Result:",resultCellDataDisplay[tr],Item["",ItemSize -> Fit](*,				
				Framed[Button["Replace expected output with actual output",testReplaceOutput[tr],Appearance->"Frameless"],FrameMargins->1,Background->White],
				Framed[Button["Rerun",testRerun[],Appearance->"Frameless"],FrameMargins->1,Background->White],deleteResultCell[]*)}},
				Alignment->{Left,Right},
				ItemSize->Full
				],
			Full
			]]],
			"TestFailure"
		]
		,
		tr["Outcome"] =="MessagesFailure",
		Cell[
			BoxData[ToBoxes[Pane[Grid[{{"Result:",resultCellDataDisplay[tr],Item["",ItemSize -> Fit](*,
				Framed[Button["Replace expected message names with actual message names",testReplaceMessage[tr],Appearance->"Frameless"],FrameMargins->1,Background->White],
				Framed[Button["Rerun",testRerun[],Appearance->"Frameless"],FrameMargins->1,Background->White],deleteResultCell[]*)}},
				Alignment->{Left,Right},
				ItemSize->Full
				],
			Full
			]]],
			"TestMessageFailure"
		]
	]
]
*)

resultCell[tr_TestResultObject] := 
	Cell[BoxData[ToBoxes[Pane[Grid[{{resultCellFirstPart[tr], Item["", ItemSize -> Fit], 
					If[#==={},
						Unevaluated[Sequence[]],
						#]&[resultCellDataDisplay[tr]]}}, 
					Alignment -> {Left, Right}, ItemSize -> Full], Full]]], Switch[tr["Outcome"], "Success", "TestSuccess", "Failure", "TestFailure", "Error", "TestError", _, "TestMessageFailure"]]

PaletteAbort[nb_NotebookObject] :=
	MathLink`CallFrontEnd[FrontEnd`EvaluatorAbort[Automatic]]

$oldFileName

PaletteSaveAs[nb_NotebookObject] :=
	Module[{stream, windowTitle, tests},
		Catch[
			windowTitle = WindowTitle /. AbsoluteOptions[nb, WindowTitle];
			$oldFileName =
				SystemDialogInput["FileSave", {
					If[ValueQ[$oldFileName] && $oldFileName =!= $Canceled, ToFileName[{DirectoryName[$oldFileName]}, windowTitle], windowTitle],
					{"Wolfram Language Test File (*.wlt)" -> {"*.wlt"}}
				}];
			If[$oldFileName =!= $Canceled,
				tests = NotebookToTests[InputNotebook[], PreserveDataInSections -> False];
				stream = OpenWrite[$oldFileName];
				If[stream === $Failed,
					Throw[{"Value" -> $oldFileName, "Messages" -> {"PaletteSaveAs", "OpenWrite returned $Failed"}}, MUnitErrorTag]
				];
				(* write out package header comment to let the front end recognize this as a package file *)




(* probably ok to not check that NotebookToTests returned $Failed *)


				WriteString[stream, tests];
				Close[stream]
			]
			,
			MUnitErrorTag
			,
			Function[{rules, tag},
				Module[{msgs = "Messages" /. rules, val = "Value" /. rules},
					MessageDialog[Column[{msgs[[1]], Style[msgs[[2]], Bold], val, If[Length[msgs] >= 3, msgs[[3]], Spacer[{0, 0}]]}], WindowTitle -> "MUnit Error"]
				]
			]
		];
	]

PaletteClear[nb_NotebookObject] :=
	(
		If[NotebookFind[nb, "TestSuccess", All, CellStyle] =!= $Failed,
			NotebookDelete[nb]];
		If[NotebookFind[nb, "TestFailure", All, CellStyle] =!= $Failed,
			NotebookDelete[nb]];
		If[NotebookFind[nb, "TestMessageFailure", All, CellStyle] =!= $Failed,
			NotebookDelete[nb]];
		If[NotebookFind[nb, "TestError", All, CellStyle] =!= $Failed,
			NotebookDelete[nb]];
		If[NotebookFind[nb, "Output", All, CellStyle] =!= $Failed,
			NotebookDelete[nb]]
	)
	
identifyIsolatedCells[] := 
	Module[{nb = ButtonNotebook[], mainstyles = {"VerificationTest", "ExpectedOutput", "ExpectedMessage", "TestOptions", "BottomCell"}, celldata, celldatawithissues}, 
		celldata = Developer`CellInformation[Cells[nb]]; 
		celldatawithissues = (celldata /. {___, "Style" -> (a_ /; Not@MemberQ[mainstyles, a]), ___} -> Sequence[]) //. {a___, PatternSequence[{___, "Style" -> "VerificationTest", ___}, {___, "Style" -> "ExpectedOutput", ___}, {___, "Style" -> "BottomCell", ___}] | 
			PatternSequence[{___, "Style" -> "VerificationTest", ___}, {___, "Style" -> "ExpectedOutput", ___}, {___, "Style" -> "TestOptions", ___}, {___, "Style" -> "BottomCell", ___}] |
			PatternSequence[{___, "Style" -> "VerificationTest", ___}, {___, "Style" -> "ExpectedOutput", ___}, {___, "Style" -> "ExpectedMessage", ___}, {___, "Style" -> "BottomCell", ___}] | 
			PatternSequence[{___, "Style" -> "VerificationTest", ___}, {___, "Style" -> "ExpectedOutput", ___}, {___, "Style" -> "ExpectedMessage", ___}, {___, "Style" -> "TestOptions", ___}, {___, "Style" -> "BottomCell", ___}], b___} :> {a, b}; 
		$problemCellids = If[celldatawithissues =!= {},
					celldatawithissues /. {___, "CellID" -> a_, ___} :> a,
					{}]]
    
$problemCellids = {}    
    
fixAllAndRunTests[]:= (While[$problemCellids =!= {}, MUnit`findNextProblemCell[]; MUnit`completeCellToTestCellGroup[]];PaletteRun[InputNotebook[]])

SetAttributes[defectiveCellsDockedCell, HoldAll]

defectiveCellsDockedCell[problemcellidlist_] := 
	Cell[BoxData[ToBoxes@Panel[Pane[Grid[{{Grid[{{Style[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "MalformedTestCellsFoundColon"], Bold, FontFamily -> "Helvetica",
								RGBColor[0.34902, 0.34902, 0.34902]], 
							Style[Dynamic@Refresh[Length@problemcellidlist, UpdateInterval -> 1], Bold, FontFamily -> "Helvetica", RGBColor[0.34902, 0.34902, 0.34902]]}}, Spacings -> {2 -> .5}], 
							Item["", ItemSize -> Fit], 
							buttonWithTooltipButWithoutIcon["FixNext-Label", "FixNext-Tooltip", MUnit`findNextProblemCell[]; MUnit`completeCellToTestCellGroup[]], 
							buttonWithTooltipButWithoutIcon["FixAll-Label", "FixAll-Tooltip", fixAllAndRunTests[]], 
							Button[Tooltip[Mouseover[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitExpressions", "Close-Off"], 
										Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitExpressions", "Close-Hover"]], 
									Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "RemoveToolbar-Tooltip"]], 
					Function[nb,
						(CurrentValue[nb, DockedCells] = If[Head@# === Cell,
										#,
										DeleteCases[#, Cell[_, "DockedCell", "ProblemDockedCell", ___]][[1]]] &[CurrentValue[nb, DockedCells]])][ButtonNotebook[]], 
									Appearance -> None, BaseStyle -> {FontColor -> GrayLevel[0], FontSize -> 12}, Evaluator -> Automatic, Method -> "Queued"]}}, 
							Alignment -> {Left, Right}], ImageMargins -> {{5, 5}, {15, 15}}, ImageSize -> Full], FrameMargins -> {{26, 22}, {Automatic, Automatic}},
				ImageMargins -> -1, Appearance -> FEPrivate`FrontEndResource["MUnitExpressions", "ErrorDockedCellNinePatch"]]],
		"DockedCell", "ProblemDockedCell", CellFrameMargins -> 0, CellTags -> "MUnitProblemDockedCell"]
  
expandToCell[x_NotebookObject] := 
 Module[{lnkre}, 
  While[(LinkWrite[$ParentLink, FrontEnd`CellInformation[x]]; lnkre = LinkRead[$ParentLink]);
        (lnkre =!= $Failed && Not[MemberQ["CursorPosition" /. lnkre, "CellBracket"]]), 
        FrontEndExecute[FrontEnd`SelectionMove[x, All, Cell, AutoScroll -> False]]]]
    
findNextProblemCell[] := 
 Module[{nb = ButtonNotebook[], currentProblemCellIDs, ci, candidateCellId}, 
  Catch@If[(currentProblemCellIDs = Cases[{#, Cells[nb, CellID -> #]} & /@ $problemCellids, x_ /; x[[2]] =!= {} :> x[[1]]]) === {}, 
           MessageDialog["No problem cells found"],
           expandToCell[nb]; 
           SelectionMove[nb, Next, Cell];
           ci = Developer`CellInformation[nb];
           If[ci === $Failed,
              SelectionMove[nb, Before, Notebook]; 
              Throw[NotebookFind[nb, currentProblemCellIDs[[1]], All, CellID]]]; 
           If[Not@MemberQ[currentProblemCellIDs, ("CellID" /. ci)[[1]]], 
              candidateCellId = SelectFirst[currentProblemCellIDs, Function[t, Position[t, #][[1, 1]] > Position[t, ("CellID" /. ci)[[1]]][[1, 1]]][CurrentValue[#, CellID] & /@ Cells[nb]] &]; 
              If[IntegerQ@candidateCellId, 
                 NotebookFind[nb, candidateCellId, Next, CellID], 
                 SelectionMove[nb, Before, Notebook]; 
                 NotebookFind[nb, currentProblemCellIDs[[1]], All, CellID]]]]]
      
revertCellBracketOptions[nb_]:= SetOptions[NotebookSelection[nb],
	CellBracketOptions -> {"Color" -> Inherited, "HoverColor" -> Inherited, "Margins" -> Inherited, "OverlapContent" -> Inherited, "Thickness" -> Inherited, "Widths" -> Inherited}]

completeCellToTestCellGroup[nb_:FrontEnd`ButtonNotebook[]] := 
	Module[{genericCells = {Cell[BoxData[""], "VerificationTest"], Cell[BoxData[""], "ExpectedOutput"], Cell[BoxData[RowBox[{"{", "}"}]], "ExpectedMessage"], 
				Cell[BoxData[RowBox[{"{", "}"}]], "TestOptions"], Cell[BoxData[ToBoxes@MUnit`bottomCell[]], "BottomCell"]},
		orderedStyles = {"VerificationTest", "ExpectedOutput", "ExpectedMessage", "TestOptions", "BottomCell"},
		allCellIDs = CurrentValue[#, CellID] & /@ Cells[nb], 
		styleOfProblemCell, j, posAmongAllCellIds, bag, orderedStylesbag, ids, cellsInNB, cellsToWrite, stylePositionAmongOrderedStyles}, 
	stylePositionAmongOrderedStyles[posAmongAllCellIds_] := Position[orderedStyles, "Style" /. Developer`CellInformation[Cells[nb, CellID -> allCellIDs[[posAmongAllCellIds]]][[1]]]][[1, 1]];
	(* Get the style positions in orderedStyles that already exist from which a test group will be constructed. *)
	styleOfProblemCell = ("Style" /. Developer`CellInformation[nb])[[1]]; 
	j = Position[orderedStyles, styleOfProblemCell][[1, 1]]; 
	posAmongAllCellIds = Position[allCellIDs, ("CellID" /. Developer`CellInformation[nb])[[1]]][[1, 1]];
	bag = {};
	orderedStylesbag = orderedStyles; 
	While[And[posAmongAllCellIds <= Length[allCellIDs],
			MemberQ[MUnit`$problemCellids, allCellIDs[[posAmongAllCellIds]]],
			MemberQ[orderedStylesbag, "Style" /. Developer`CellInformation[Cells[nb, CellID -> allCellIDs[[posAmongAllCellIds]]][[1]]]]], 
		AppendTo[bag, stylePositionAmongOrderedStyles[posAmongAllCellIds]];
		orderedStylesbag = Take[orderedStyles, {stylePositionAmongOrderedStyles[posAmongAllCellIds] + 1, 5}];
		posAmongAllCellIds++;
		j++];
	ids = Table[MUnit`$problemCellids[[j]], {j, #, # + Length[bag] - 1}] &[Position[MUnit`$problemCellids, ("CellID" /. Developer`CellInformation[SelectedCells[nb]])[[1]]][[1, 1]]];
	cellsInNB = NotebookRead /@ (Cells[nb, CellID -> #][[1]] & /@ ids);
	cellsToWrite = orderedStyles /. s_String :> If[MemberQ[Extract[orderedStyles, List /@ bag], s], 
      							(Cases[cellsInNB, Cell[_, s, ___]][[1]])/.(CellBracketOptions->_)->Sequence[], 
							If[MemberQ[{"VerificationTest", "ExpectedOutput", "BottomCell"}, s], Cases[genericCells, Cell[_, s, ___]][[1]], Unevaluated[Sequence[]]]];
	(NotebookFind[nb, #, All, CellID]; NotebookDelete[nb]) & /@ ids; 
	NotebookWrite[nb, cellsToWrite];
	MUnit`$problemCellids = DeleteCases[MUnit`$problemCellids, Alternatives @@ ids]]
    
addDockedCellConvertNotebook[] :=
	Module[{c, nb = InputNotebook[], presentDockedCellsValue, content, convertNotebookCell},
	c=FE`Evaluate@FEPrivate`CanAddToolbar[InputNotebook[]];
	presentDockedCellsValue = CurrentValue[nb, DockedCells];
	If[c===True&&((* The input notebook is not a testing notebook. *)Cases[presentDockedCellsValue, Cell[__, CellTags -> "MUnitStaticToolbar", ___], {0, Infinity}] === {}),
		If[(* There is no "Convert to Testing Notebook" docked cell. *)Cases[presentDockedCellsValue, Cell[__, CellTags->"MUnitConvertToTestingNotebook",___], {0, Infinity}] === {}
		,
		Needs["MUnit`"]; 
		content = Panel[Pane[Grid[{{Button[Tooltip[Grid[{{Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitExpressions", "ConvertNotebookToTestNotebook"],
						Style[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "ConverttoTestingNotebook-Label"], FontFamily -> "Sans Serif", FontSize -> 11, FontWeight -> Bold, FontColor -> RGBColor[0.458824, 0.458824, 0.458824], 
							AdjustmentBoxOptions -> {BoxBaselineShift -> -50}]}},
					Alignment -> {Automatic, Center}], Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "ConverttoTestingNotebook-Tooltip"], TooltipDelay -> .5], 
							Block[{$ContextPath}, CurrentValue[nb, DockedCells] = Inherited;
  										Needs["MUnit`"];
										MUnit`PaletteConvertNotebook[InputNotebook[]]], 
						Appearance -> FEPrivate`FrontEndResource["MUnitExpressions", "ButtonAppearances"], FrameMargins -> {{10, 10}, {0, 0}},
						Method -> "Queued", ImageSize -> Automatic],
					Item["", ItemSize -> Fit],
					Button[Tooltip[Mouseover[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitExpressions", "Close-Off"], 
										Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitExpressions", "Close-Hover"]], 
									Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "RemoveToolbar-Tooltip"]], 
					Function[nb, (CurrentValue[nb, DockedCells] = If[MatchQ[#,Cell[_, "DockedCell", ___, CellTags -> "MUnitConvertToTestingNotebook", ___]],
												Inherited, 
							DeleteCases[#, Cell[_, "DockedCell", ___, CellTags -> "MUnitConvertToTestingNotebook", ___]]] &[CurrentValue[nb, DockedCells]])][ButtonNotebook[]], 
									Appearance -> None, BaseStyle -> {FontColor -> GrayLevel[0], FontSize -> 12}, Evaluator -> Automatic, Method -> "Queued"]}},
					BaseStyle -> {"DialogStyle", Bold, FontColor -> Darker[Gray]},
					Alignment -> Left], ImageMargins -> {{5, 5}, {5, 5}}], Background -> GrayLevel[.9],FrameMargins -> {{26, 12}, {Automatic, Automatic}},
			ImageMargins -> -1];
		convertNotebookCell = Cell[BoxData[ToBoxes@content], "DockedCell", CellFrameMargins -> 0, Background -> GrayLevel[.8], ShowCellTags -> False, CellTags->"MUnitConvertToTestingNotebook"];
		CurrentValue[nb, DockedCells] = Which[presentDockedCellsValue==={}
							,
							convertNotebookCell
							,
							MatchQ[presentDockedCellsValue, Cell[___]]
							,
							{presentDockedCellsValue, convertNotebookCell}
							,
							True
							,
							Append[presentDockedCellsValue, convertNotebookCell]]
		,
		CurrentValue[nb, DockedCells] = Which[MatchQ[presentDockedCellsValue, Cell[___]]
							,
							Inherited
							,
							Length@presentDockedCellsValue === 2
							,
							DeleteCases[presentDockedCellsValue, Cell[___, CellTags->"MUnitConvertToTestingNotebook",___]][[1]]
							,
							True
							,
							DeleteCases[presentDockedCellsValue, Cell[___, CellTags->"MUnitConvertToTestingNotebook",___]]]]]]
	

attachInsideFrameLabel[cell_, pos_, opos_] := FrontEndExecute[FrontEnd`AttachCell[EvaluationCell[], cell, pos, opos]]

Protect[attachInsideFrameLabel]    

End[]
