(* ::Package:: *)

NotebookToTests::usage =
""

CellsToTests::usage =
""

validTestGroupPattern::usage =
""

$testoptions::usage =
""

TestCellEvaluationFunction::usage =
""

OtherCellEvaluationFunction::usage =
""

canonicalizeBoxData::usage =
""

fullFormCell::usage =
""

rerunButton::usage =
""

Begin["`Package`"]

cellEvaluationFunction

End[]

Begin["`Notebooks`Private`"]

handleTestOptionsContents[box_] :=
	Module[{stripped, stripped1},
		stripped =
			If[MatchQ[box, RowBox[{"{", ___, "}"}]],
				stripped1 = box[[1]];
				If[Length[stripped1] > 2,
					Take[stripped1, {2, -2}]
					,
					{}
				]
				,
				{box}
			];
		Switch[stripped,
			{""}, {}
			,
			{}, Sequence @@ {}
			,
			{RowBox[{_String ..}]}, Sequence @@ {"\[IndentingNewLine]", "\t",",", "\[IndentingNewLine]","\t", stripped}
			,
			{RowBox[__]}, Sequence @@ {"\[IndentingNewLine]", "\t",",", "\[IndentingNewLine]","\t", stripped}
			,
			_, Throw[{"Value" -> stripped, "Messages" -> {"VerificationTest Cell", "Invalid VerificationTest Options"}}, MUnitErrorTag]
		]
	]

getRequireBoxes[{opts___}] :=
	Module[{tags, interfaceEnvironments, languages, products, disabled},
		tags = CellTags /. Flatten[{opts}] /. CellTags -> {};
		(* normalize tags *)
		tags = Flatten[{tags}];
		interfaceEnvironments = Intersection[{"Macintosh", "Windows", "X"}, tags];
		languages = Intersection[{"English", "Japanese", "ChineseSimplified", "Spanish"}, tags];
		products = Intersection[{"Mathematica", "MathematicaPlayer", "MathematicaPlayerPro"}, tags];
		disabled = MemberQ[tags, "False" | "Obsolete" | "Disabled"];
		Which[
			disabled,
			"False"
			,
			Length[interfaceEnvironments] == 1,
			RowBox[{"$InterfaceEnvironment", "==", ToBoxes[interfaceEnvironments[[1]]]}]
			,
			interfaceEnvironments != {},
			RowBox[{"MemberQ", "[", RowBox[{ToBoxes[interfaceEnvironments], ",", "$InterfaceEnvironment"}], "]"}]
			,
			Length[languages] == 1,
			RowBox[{"$Language", "==", ToBoxes[languages[[1]]]}]
			,
			languages != {},
			RowBox[{"MemberQ", "[", RowBox[{ToBoxes[languages], ",", "$Language"}], "]"}]
			,
			Length[products] == 1,
			RowBox[{RowBox[{"(", RowBox[{"\"ProductIDName\"", "/.", "$ProductInformation"}], ")"}], "==", ToBoxes[products[[1]]]}]
			,
			products != {},
			RowBox[{"MemberQ", "[", RowBox[{ToBoxes[products], ",", RowBox[{"\"ProductIDName\"", "/.", "$ProductInformation"}]}], "]"}]
			,
			True,
			"True"
		]
	]

(* TODO: try to remember why I added "Subtitle" and "Subsubtitle" in the first place, since they are not grouping cell styles... *)
sectionStylePattern =
	"Subtitle" | "Subsubtitle" | "Chapter"| "Subchapter" | "Section" | "Subsection" | "Subsubsection" | "Subsubsubsection" | "TestSection" |
	(* from function paclet notebook *)
	"TemplatesSection"

platformStylePattern =
	"PlatformSection" | "MacintoshSection" | "WindowsSection" | "XSection"
	
contentHasErrorBox[content_] := 
	Quiet[With[{box = BoxData[If[ListQ@#, RowBox@#, #] &[DeleteCases[content, RowBox[{"(*", ___, "*)"}] | "\n" | "\[IndentingNewLine]", Infinity] /. RowBox[{RowBox[a_]}] :> RowBox[a]]]}, 
		Not@FreeQ[MakeExpression[StripBoxes@box, StandardForm], ErrorBox]], Syntax::sntxi]

cellsPreParse[cells_]:=
	Module[{tempCells},
		(*Remove the bottomCell and ResultCell before sending it to the parser*)
		tempCells=DeleteCases[cells, Cell[___, "BottomCell", ___]|Cell[___, "ActualOutput", ___]|Cell[___, "ActualMessage", ___]|Cell[___, "TestSuccess", ___]|Cell[___, "TestFailure", ___]|Cell[___, "TestMessageFailure", ___]|Cell[___, "TestError", ___], Infinity];
		tempCells = MapAt[fullFormCell, tempCells, Position[tempCells, Cell[BoxData[_], "VerificationTest"|"ExpectedOutput"|"ActualOutput", __]]];
		(*Replace Null*)
		ReplaceAll[tempCells,
				{Cell[BoxData[em_], "ExpectedMessage", other___]:>Cell[BoxData[If[ToExpression[em]===Null,ToBoxes[{}],em]], "ExpectedMessage",other],
				Cell[BoxData[to_], "TestOptions", other___]:>Cell[BoxData[If[contentHasErrorBox@to||(ToExpression[to]===Null),ToBoxes[{}],to]], "TestOptions",other]}
				
		]
	]
	

cellParser[cells_List, OptionsPattern[{testParser, sectionParser}]] :=
	Module[{testIDTitleString, testIDFormat, preserve},
		{testIDTitleString, testIDFormat, preserve} = OptionValue[{TestIDTitleString, TestIDFormat, PreserveDataInSections}];
		ReplaceAll[cells,
			{
				Cell[CellGroupData[{Cell[name_String, "Title", opts___], rest___Cell}, _]] :>
					sectionParser[name, {opts}, rest, TestIDTitleString -> StringReplace[name, "Test Results " ~~ file__ ~~ ".nb" -> file], TestIDFormat -> testIDFormat, PreserveDataInSections -> preserve]
				,
				Cell[CellGroupData[{Cell[name_, "Title", opts___], rest___Cell}, _]] :>
					sectionParser[name, {opts}, rest, TestIDTitleString -> DataToString[name], TestIDFormat -> testIDFormat, PreserveDataInSections -> preserve]
				,
				Cell[CellGroupData[{Cell[name_, (*style:*)sectionStylePattern, opts___], rest___Cell}, _]] :>
					sectionParser[name, {opts}, rest, TestIDTitleString -> testIDTitleString, TestIDFormat -> testIDFormat, PreserveDataInSections -> preserve]
				,
				Cell[CellGroupData[{Cell[name_, style:platformStylePattern, (*opts*)___], rest___Cell}, _]] :>
					platformSectionParser[name, style, rest, TestIDTitleString -> testIDTitleString, TestIDFormat -> testIDFormat]
				,
				(* handle non-grouped sections *)
				Cell[name_String, "Title", opts___] :>
					sectionParser[name, {opts}, TestIDTitleString -> StringReplace[name, ".nb" -> ""], TestIDFormat -> testIDFormat, PreserveDataInSections -> preserve]
				,
				Cell[name_, "Title", opts___] :>
					sectionParser[name, {opts}, TestIDTitleString -> DataToString[name], TestIDFormat -> testIDFormat, PreserveDataInSections -> preserve]
				,
				Cell[name_, (*style:*)sectionStylePattern, opts___] :>
					sectionParser[name, {opts}, TestIDTitleString -> testIDTitleString, TestIDFormat -> testIDFormat, PreserveDataInSections -> preserve]
				,
				Cell[name_, style:platformStylePattern, (*opts*)___] :>
					platformSectionParser[name, style, TestIDTitleString -> testIDTitleString, TestIDFormat -> testIDFormat]
				,
				Cell[text_String, "Text", (*opts*)___] :>
					textParser[text]
				,
				Cell[CellGroupData[{first___Cell, test:Cell[_, "VerificationTest", ___], rest:___Cell}, _]] :>
					testParser[{first, test, rest}, TestIDTitleString -> testIDTitleString, TestIDFormat -> testIDFormat]
				,
				test:Cell[_, "VerificationTest", ___] :>
					testParser[{test}, TestIDTitleString -> testIDTitleString, TestIDFormat -> testIDFormat]
				,
				Cell[CellGroupData[{Cell[BoxData[e_], "Environ", opts___], ___Cell}, _]] :>
					Catch[
						validateBoxes[e]
						,
						MUnitErrorTag
						,
						Function[{rules, tag},
							Throw[Append[rules, "CellID" -> OptionValue[Cell, {opts}, CellID]], MUnitErrorTag]
						]
					]
				,
				Cell[BoxData[e_], "Environ", opts___] :>
					Catch[
						validateBoxes[e]
						,
						MUnitErrorTag
						,
						Function[{rules, tag},
							Throw[Append[rules, "CellID" -> OptionValue[Cell, {opts}, CellID]], MUnitErrorTag]
						]
					]
				,
				Cell[r:RawData[_], ___] :>
					Throw[{"Value" -> r, "Messages" -> {"Cell Parsing", "Read RawData"}}, MUnitErrorTag]
				,
				Cell[BoxData[e_], "TestOptions", ___] :>
					Throw[{"Value" -> e, "Messages" -> {"Cell Parsing", "Orphaned TestOptions cell"}}, MUnitErrorTag]
				,
				Cell[BoxData[e_], "ExpectedOutput", ___] :>
					Throw[{"Value" -> e, "Messages" -> {"Cell Parsing", "Orphaned ExpectedOutput cell"}}, MUnitErrorTag]
				,
				Cell[BoxData[e_], "ExpectedMessage", ___] :>
					Throw[{"Value" -> e, "Messages" -> {"Cell Parsing", "Orphaned ExpectedMessage cell"}}, MUnitErrorTag]
				,
				Cell[BoxData[e_], "ExpectedMessages", ___] :>
					Throw[{"Value" -> e, "Messages" -> {"Cell Parsing", "Orphaned ExpectedMessages cell"}}, MUnitErrorTag]
				,
				Cell[args___] :>
					((*Message[cellParser::unrecognized, Cell[args]];*)Sequence @@ {})
			}
		]
	]

Options[sectionParser] = {PreserveDataInSections -> True}

sectionParser[name_, cellOpts_, rest___Cell, opts:OptionsPattern[{testParser, sectionParser}]] :=
	Module[{requireBoxes = getRequireBoxes[cellOpts], preserve = OptionValue[PreserveDataInSections]},
		Sequence @@ {
			RowBox[{
				"BeginTestSection", "[",
					If[requireBoxes === "True",
						ToBoxes[If[preserve, name, DataToString[name]]],
						RowBox[{ToBoxes[If[preserve, name, DataToString[name]]], Sequence @@ {",", requireBoxes}}]],
				"]"
			}]
			,
			Sequence @@ Flatten[{cellParser[{rest}, opts]}]
			,
			RowBox[{"EndTestSection", "[", "]"}]
		}
	]

platformSectionParser[(*name*)_String, style_String, rest___Cell, OptionsPattern[testParser]] :=
	Module[{testIDTitleString, testIDFormat},
		{testIDTitleString, testIDFormat} = OptionValue[{TestIDTitleString, TestIDFormat}];
		Sequence @@ {
			RowBox[{
				"BeginTestSection", "[",
					RowBox[{
						ToBoxes[style], ",", RowBox[{"$InterfaceEnvironment", "==", ToBoxes[StringReplace[style, p__ ~~ "Section" :> p]]}]
					}],
				"]"
			}]
			,
			Sequence @@ Flatten[{cellParser[{rest}, TestIDTitleString -> testIDTitleString, TestIDFormat -> testIDFormat]}]
			,
			RowBox[{"EndTestSection", "[", "]"}]
		}
	]


Options[testParser] = {TestIDTitleString -> "Untitled", TestIDFormat -> "CellID"}

testParser[list_List, (*opts:*)OptionsPattern[]] :=
	Module[{testIDTitleString, testIDFormat,list1},
		{testIDTitleString, testIDFormat} = OptionValue[{TestIDTitleString, TestIDFormat}];
		(* A hack to handle the null case for input/expected output*) 
		list1=ReplaceAll[list,
				{Cell[BoxData[""], "VerificationTest", x___]:>Cell[BoxData["Null"], "VerificationTest", x],
				Cell[BoxData[""], "ExpectedOutput", x___]:>Cell[BoxData["Null"], "ExpectedOutput", x]}];	
		ReplaceAll[list1, {
			(* single ExpectedMessages cell *)
			{
				t:Cell[BoxData[_], "VerificationTest", topts___?OptionQ],
				eo:Cell[BoxData[_], "ExpectedOutput", ___]:Cell[BoxData["True"], "ExpectedOutput"],
				em:Cell[BoxData[_], "ExpectedMessage", ___]:Cell[BoxData[RowBox[{"{", "}"}]], "ExpectedMessage"],
				to:Cell[BoxData[_], "TestOptions", ___]:Cell[BoxData[RowBox[{"{", "}"}]], "TestOptions"],
				(* if a test result cell is present, just ignore it *)
				tr:Cell[___, "TestSuccess" | "TestFailure" | "TestMessagesFailure" | "TestError", ___]:Cell[BoxData[""], "TestSuccess"]
			} :> testParser0[
					t[[1,1]], {topts},
					to[[1,1]], List@@to[[3;;]],
					em[[1,1]], List@@em[[3;;]],
					eo[[1,1]], List@@eo[[3;;]], TestIDTitleString -> testIDTitleString, TestIDFormat -> testIDFormat]
			,
			(* multiple ExpectedMessage cells *)
			{
				t:Cell[BoxData[_], "VerificationTest", topts___?OptionQ],
				to:Cell[BoxData[_], "TestOptions", ___]:Cell[BoxData[RowBox[{"{", "}"}]], "TestOptions"],
				ems:Cell[BoxData[_], "ExpectedMessage", ___]..,
				eo:Cell[BoxData[_], "ExpectedOutput", ___]:Cell[BoxData["True"], "ExpectedOutput"],
				(* if a test result cell is present, just ignore it *)
				tr:Cell[___, "TestSuccess" | "TestFailure" | "TestMessagesFailure" | "TestError", ___]:Cell[BoxData[""], "TestSuccess"]
			} :> testParser0[
					t[[1,1]], {topts},
					to[[1,1]], List@@to[[3;;]],
					(*TODO: dig through linear syntax of message and extract arguments and use newer argument checking functionality*)
					(*TODO: better error checking for mal-formed messages cells: empty cells but also edited *)
					RowBox[{"{", RowBox[Riffle[#[[1, 1, 1, 1, 1]] & /@ {ems}, ","]], "}"}], {},
					eo[[1,1]], List@@eo[[3;;]], TestIDTitleString -> testIDTitleString, TestIDFormat -> testIDFormat]
			,
			_ :> Throw[{"Value" -> list, "Messages" -> {"VerificationTest Parsing", "The order of the cells is not understood by MUnit"}}, MUnitErrorTag]
		}]
	]

testParser0[tb_, topts_, tob_, toopts_, emb_, emopts_, eob_, eoopts_, OptionsPattern[testParser]] :=
	Module[{testIDTitleString, testIDFormat, dateString, cellID, cellIDString, testID, tags, bugIDString, head},
		{testIDTitleString, testIDFormat} = OptionValue[{TestIDTitleString, TestIDFormat}];
		dateString = DateString[Flatten[CellChangeTimes /. Flatten[topts] /. CellChangeTimes :> {AbsoluteTime[]}][[1]], {"Year", "Month", "Day"}, TimeZone -> 0];
		tags = Flatten[{CellTags /. Flatten[topts] /. CellTags -> {}}];
		cellID = CellID /. topts;
		(* Commenting out unnecessary code. This is getting in the way of converting notebooks that does not have CellID
		If[cellID === CellID,
			Throw[{"Value" -> tb, "Messages" -> {"VerificationTest Parsing", "VerificationTest Cell must have a CellID"}}, MUnitErrorTag]
		];*)
		cellIDString = IntegerString[cellID, 36, 7];
		bugIDString = StringJoin[("-bug-" <> #)& /@ Flatten[StringCases[tags, "bug" ~~ id__ :> id]]];
		testID =
			Switch[testIDFormat,
				"CellID",
				cellID
				,
				"WRI",
				testIDTitleString <> "-" <> dateString <> "-" <> cellIDString <> bugIDString
				,
				_,
				cellID
			];
		head = Intersection[{"VerificationTest", "NTest", "OrTest"}, tags];
		If[head == {},
			head = "VerificationTest",
			head = head[[1]]
		];
		(*
		kind of a hack. run all of the boxes through the parser once, in order to get any errors propagated
		do each one separately so that we can add respective CellIDs
		*)
		Catch[
			validateBoxes[tb]
			,
			MUnitErrorTag
			,
			Function[{rules, tag},
				Throw[Append[rules, "CellID" -> OptionValue[Cell, topts, CellID]], MUnitErrorTag]
			]
		];
		Catch[
			validateBoxes[tob]
			,
			MUnitErrorTag
			,
			Function[{rules, tag},
				Throw[Append[rules, "CellID" -> OptionValue[Cell, toopts, CellID]], MUnitErrorTag]
			]
		];
		Catch[
			validateBoxes[emb]
			,
			MUnitErrorTag
			,
			Function[{rules, tag},
				Throw[Append[rules, "CellID" -> OptionValue[Cell, emopts, CellID]], MUnitErrorTag]
			]
		];
		Catch[
			validateBoxes[eob]
			,
			MUnitErrorTag
			,
			Function[{rules, tag},
				Throw[Append[rules, "CellID" -> OptionValue[Cell, eoopts, CellID]], MUnitErrorTag]
			]
		];
		RowBox[{
			head, "[", RowBox[{"(*", " ", ToBoxes[++$notebookTestIndex], " ", "*)"}], "\[IndentingNewLine]",
				RowBox[{
					(* provide a courtesy for those horribly mis-guided people who have multiple inputs in a Test cell *)
						If[Head[tb] === List,
						(* split into lines *)
						Module[{lines = Split[tb, #1 == #2 || #2 == "\[IndentingNewLine]" || #2 == "\n" &]},
							(Function[line,
								(* line is {RowBox[...], ""} *)
								(* may or may not have semi-colons at the end of the lines *)
								If[MatchQ[line, {RowBox[{__, ";"}], "\[IndentingNewLine]"|"\n"}],
									(* already has semi-colon *)
									{"\t",line}
									,
									(* wrap with semi-colon *)
									{"\t",RowBox[{line[[1]], ";"}], "\[IndentingNewLine]"}
								]
							] /@ Most[lines]) ~Join~ {"\t",Last[lines]}
						]
						,
						{"\t",tb}
					],
					"\[IndentingNewLine]",
					"\t",",", "\[IndentingNewLine]",
					"\t",eob, 
					Switch[emb, 
							{""}, Sequence @@ {"\t", {}}
							, 
							RowBox[{"{", "}"}], Sequence @@ {"\t", {}}
							, 
							_, Sequence @@ {"\[IndentingNewLine]","\t",",", "\[IndentingNewLine]","\t", emb}],
					handleTestOptionsContents[tob]
				}], "\[IndentingNewLine]","]"
		}]
	]

(* boxParser converts from boxes -> expressions *)
validateBoxes[boxes_] :=
	Quiet[
		Module[{oldContext, oldContextPath},
			Internal`WithLocalSettings[
				oldContext = $Context;
				oldContextPath = $ContextPath;
				$Context = Unique["MUnit`Notebooks`Private`Sandbox`"];
				$ContextPath = {$Context "System`"};
				,
				Check[
					Quiet[
						(* do ToExpression to generate any messages *)
						ToExpression[boxes, StandardForm, HoldComplete]
						,
						{Syntax::noinfo}
					]
					,
					Throw[{"Value" -> boxes, "Messages" -> {"Box Parsing", "The boxes could not be parsed"}}, MUnitErrorTag]
				]
				,
				$Context = oldContext;
				$ContextPath = oldContextPath;
				Quiet[Remove["MUnit`Notebooks`Private`Sandbox`*"], {Remove::rmnsm}]
			]
		];
		boxes
	]

textParser[text_String] := "(* ::Text:: *)\n(*"<>text<>"*)"

CellsToTests[nb_NotebookObject] :=
	Module[{selection},
		selection = NotebookRead[nb];
		If[MatchQ[selection, Cell[_CellGroupData]],
			(* make selection conform to {Cell[_CellGroupData]..} pattern *)
			selection = {selection};
		];
		cellsToTestsString[selection]
	]

CellsToTests[cells:{___Cell}] :=
	cellsToTestsString[{Cell[CellGroupData[cells, Open]]}]

NotebookToTests[nb_NotebookObject, Shortest[title0_:Automatic], opts:OptionsPattern[sectionParser]] :=
	Module[{nbg, cells, title},
		nbg = NotebookGet[nb];
		If[nbg === $Failed,
			(* nb is an invalid (closed) notebook *)
			Throw[{"Value" -> nb, "Messages" -> {"NotebookGet", "$Failed"}}, MUnitErrorTag]
		];
		title = title0;
		If[title === Automatic,
			title = WindowTitle /. AbsoluteOptions[nb, WindowTitle];
			title = StringReplace[title, ".nb"~~EndOfString -> ""];
		];
		cells = nbg[[1]];
		(* insert a Title cell group if it doesn't exist already *)
		If[!MatchQ[cells, {Cell[CellGroupData[{Cell[_, "Title", ___], __}, _]]}], 
			cells = {Cell[CellGroupData[{Cell[title, "Title"], Sequence @@ cells}, Open]]}
		];
		cellsToTestsString[cells, opts]
	]

(*
useful for running notebooks without a front end
*)
NotebookToTests[nb:Notebook[oldCells_, nbOpts:OptionsPattern[]], title_, opts:OptionsPattern[sectionParser]] :=
	Module[{cells},
		(* insert a Title cell group if it doesn't exist already *)
		If[!MatchQ[oldCells, {Cell[CellGroupData[{Cell[_, "Title", ___], __}, _]]}],
			(* cannot use OptionValue mechanism for getting value of WindowTitle, since it complains loudly about
			unrecognized options (since in a stand-alone kernel Options[Notebook] == {} ) *)
			cells = {Cell[CellGroupData[{Cell[title, "Title"], Sequence @@ oldCells}, Open]]}
			,
			cells = oldCells
		];
		cellsToTestsString[cells, opts]
	]

cellsToTestsString[cells_, opts:OptionsPattern[sectionParser]] :=
	Module[{parsed,parsedCells},
		parsedCells=cellsPreParse[cells];
		Block[{$notebookTestIndex = 0},
			parsed = cellParser[parsedCells, TestIDFormat -> "WRI", opts];
			If[parsed == {},
				Throw[{"Value" -> cells, "Messages" -> {"Cell Parsing", "Cell Parser returned {}"}}, MUnitErrorTag]
			];
			StringDrop[StringJoin[BoxesToReadableFormString[#] <> "\n\n" & /@ parsed], -1]
		]
	]

validTestGroupPattern[]=
	Cell[CellGroupData[{Cell[_, "VerificationTest", ___], Cell[_, "ExpectedOutput", ___], Cell[_, "ActualOutput", ___] | PatternSequence[], Cell[_, "ExpectedMessage", ___] | PatternSequence[], 
				Cell[_, "ActualMessage", ___] | PatternSequence[], Cell[_, "TestOptions", ___] | PatternSequence[],
				Cell[_, "TestSuccess" | "TestFailure" | "TestMessageFailure" | "TestError", ___] | PatternSequence[], Cell[_, "BottomCell", ___]}, _]]
				
$testoptions = {"MemoryConstraint", "SameTest", "TestID", "TimeConstraint"}

whiteSpaceOrNone = PatternSequence[_String?(StringMatchQ[#, Whitespace] &) ...]

testOptionPatterns = Alternatives[{},
	{RowBox[{"{", whiteSpaceOrNone, "}"}]},
	{RowBox[{"{", RowBox[{Alternatives @@ $testoptions, whiteSpaceOrNone, "\[Rule]", whiteSpaceOrNone, _}], "}"}]},
	{RowBox[{"{", RowBox[{RowBox[{a_ /; MemberQ[$testoptions, a], whiteSpaceOrNone, "\[Rule]", whiteSpaceOrNone, _}], ",",
		RowBox[{b_ /; MemberQ[$testoptions, b], whiteSpaceOrNone, "\[Rule]", whiteSpaceOrNone, _}]}], "}"}]},
	{RowBox[{"{", RowBox[{RowBox[{a_ /; MemberQ[$testoptions, a], whiteSpaceOrNone, "\[Rule]", whiteSpaceOrNone, _}], ",",
		RowBox[{b_ /; MemberQ[$testoptions, b], whiteSpaceOrNone, "\[Rule]", whiteSpaceOrNone, _}], ",",
		RowBox[{c_ /; MemberQ[$testoptions, c], whiteSpaceOrNone, "\[Rule]", whiteSpaceOrNone, _}]}], "}"}]},
	{RowBox[{"{", RowBox[{RowBox[{a_ /; MemberQ[$testoptions, a], whiteSpaceOrNone, "\[Rule]", whiteSpaceOrNone, _}], ",",
		RowBox[{b_ /; MemberQ[$testoptions, b], whiteSpaceOrNone, "\[Rule]", whiteSpaceOrNone, _}], ",",
		RowBox[{c_ /; MemberQ[$testoptions, c], whiteSpaceOrNone, "\[Rule]", whiteSpaceOrNone, _}], ",",
		RowBox[{d_ /; MemberQ[$testoptions, d], whiteSpaceOrNone, "\[Rule]", whiteSpaceOrNone, _}]}], "}"}]}]
				
malformedTestCellGroupMessageOpenQ[] := Select[Notebooks[], CurrentValue[#, {TaggingRules, "Message"}] === "MalformedTestCellGroup" &] =!= {}

disallowedSymbolPresentMessageOpenQ[] := Select[Notebooks[], CurrentValue[#, {TaggingRules, "Message"}] === "DisallowedSymbolPresent" &] =!= {}

defectiveTestOptionsMessageOpenQ[] := Select[Notebooks[], CurrentValue[#, {TaggingRules, "Message"}] === "DefectiveTestOptionsCell" &] =!= {}

TestCellEvaluationFunction[_, _] :=
	Module[{nb, grouped, re},
		Catch[
			nb = EvaluationNotebook[];
			If[nb === $Failed,
				Throw[{"Value" -> Null, "Messages" -> {"VerificationTest", "Internal Error: EvaluationNotebook[] returned $Failed"}}, MUnitErrorTag]
			];
			CurrentValue[nb, Deployed] = True;
			SelectionMove[nb, All, EvaluationCell];
			grouped = "FirstCellInGroup" /. Developer`CellInformation[nb][[1]];
			If[grouped,
				SelectionMove[nb, All, CellGroup]
			];
			CheckAbort[If[Or[Not@MatchQ[#,validTestGroupPattern[]],
					(MUnit`$formCheck =!= False)&&(Cases[DeleteCases[#, RowBox[{"(*", ___, "*)"}], Infinity],Alternatives@@MUnit`$exclusionForms,Infinity]=!={})]&[re = NotebookRead[nb]],
					
					CurrentValue[nb, Deployed] = Inherited;
					Abort[]];
					
					cellEvaluationFunction[nb, grouped];
					CurrentValue[nb, Deployed] = Inherited;,
					
					Which[Not@MatchQ[re,validTestGroupPattern[]] && malformedTestCellGroupMessageOpenQ[],
						SetSelectedNotebook[Select[Notebooks[], CurrentValue[#, {TaggingRules, "Message"}] === "MalformedTestCellGroup" &][[1]]];,
						disallowedSymbolPresentMessageOpenQ[],
						SetSelectedNotebook[Select[Notebooks[], CurrentValue[#, {TaggingRules, "Message"}] === "DisallowedSymbolPresent" &][[1]]];,
						True,
						If[Not@MatchQ[re,validTestGroupPattern[]],
							MessageDialog["The selection contained a malformed test cell group.", TaggingRules -> {"Message" -> "MalformedTestCellGroup"}];,
							If[(MUnit`$formCheck =!= False)&&(Cases[DeleteCases[re, RowBox[{"(*", ___, "*)"}], Infinity],Alternatives@@MUnit`$exclusionForms,Infinity]=!={}),
								MessageDialog["Expressions with symbols in the list given by Names[\"*Form\"] or Print with the exception of CapForm, EdgeForm, ExportForm, FaceForm, HornerForm, JoinForm, PrecedenceForm, RealBlockDiagonalForm and ResponseForm cannot be used in tests.", TaggingRules -> {"Message" -> "DisallowedSymbolPresent"}];,
								Null;];]];
					CurrentValue[nb, Deployed] = Inherited;]
			,
			MUnitErrorTag
			,
			Function[{rules, tag},
				Module[{msgs = "Messages" /. rules, val = "Value" /. rules},
					CurrentValue[nb, Deployed] = Inherited;
					MessageDialog[Column[{msgs[[1]], Style[msgs[[2]], Bold], val}], WindowTitle -> "MUnit Error"];
					(* return Null to prevent the MessageDialog's NotebookObject from being returned in the test notebook *)
					Null
				]
			]
		]
	]

OtherCellEvaluationFunction[_, _] :=
	Module[{nb, re},
		Catch[
			nb = EvaluationNotebook[];
			If[nb === $Failed,
				Throw[{"Value" -> Null, "Messages" -> {"VerificationTest", "Internal Error: EvaluationNotebook[] returned $Failed"}}, MUnitErrorTag]
			];
			CurrentValue[nb, Deployed] = True;
			SelectionMove[nb, All, EvaluationCell];
			SelectionMove[nb, All, CellGroup];
	(*
			If[Developer`CellInformation[nb] === $Failed,
				CurrentValue[nb, Deployed] = Inherited;
				(* cells are not grouped with a Test cell *)
				SelectionMove[nb, After, EvaluationCell];
				Throw[{"Value" -> Null, "Messages" -> {"VerificationTest", "Malformed VerificationTest CellGroup"}}, MUnitErrorTag]
			];
	*)
			CheckAbort[If[Or[Not@MatchQ[#,validTestGroupPattern[]],
					(MUnit`$formCheck =!= False)&&(Cases[DeleteCases[#, RowBox[{"(*", ___, "*)"}], Infinity],Alternatives@@MUnit`$exclusionForms,Infinity]=!={}),
					Not@MatchQ[Cases[#, Cell[BoxData[a_], "TestOptions", __] :> a, Infinity], testOptionPatterns]]&[re = NotebookRead[nb]],
					
					CurrentValue[nb, Deployed] = Inherited;
					Abort[]];
					
					cellEvaluationFunction[nb, True];
					CurrentValue[nb, Deployed] = Inherited;,
					
					If[malformedTestCellGroupMessageOpenQ[] || disallowedSymbolPresentMessageOpenQ[] || defectiveTestOptionsMessageOpenQ[],
						Null,
						Which[Not@MatchQ[re,validTestGroupPattern[]],
							MessageDialog["The selection contained a malformed test cell group.", TaggingRules -> {"Message" -> "MalformedTestCellGroup"}];,
							(MUnit`$formCheck =!= False)&&(Cases[DeleteCases[re, RowBox[{"(*", ___, "*)"}], Infinity],Alternatives@@MUnit`$exclusionForms,Infinity]=!={}),
							MessageDialog["Expressions with symbols in the list given by Names[\"*Form\"] or Print with the exception of CapForm, EdgeForm, ExportForm, FaceForm, HornerForm, JoinForm, PrecedenceForm, RealBlockDiagonalForm and ResponseForm cannot be used in tests.", TaggingRules -> {"Message" -> "DisallowedSymbolPresent"}];,
							True,
							MessageDialog["The selection contained a defective test options cell.", TaggingRules -> {"Message" -> "DefectiveTestOptionsCell"}];]];
					CurrentValue[nb, Deployed] = Inherited;]
			,
			MUnitErrorTag
			,
			Function[{rules, tag},
				Module[{msgs = "Messages" /. rules, val = "Value" /. rules},
					MessageDialog[Column[{msgs[[1]], Style[msgs[[2]], Bold], val}], WindowTitle -> "MUnit Error"];
					(* return Null to prevent the MessageDialog's NotebookObject from being returned in the test notebook *)
					Null
				]
			]
		]
	]


(* TODO: rewrite using CellPrint to be nice, and also some how simulate cell evaluating during test evaluation *)
(*
Precondition: if there is a cell group, then it is selected, if there is only a Test cell, then it is selected
*)

FixTimesComposition[r_] := 
	If[MatchQ[r, RowBox[{"Times", "[", RowBox[{PatternSequence[RowBox[{"CompoundExpression", ___}], ","] .., ___}], "]"}]],
		Replace[r /. RowBox[{"CompoundExpression", "[", RowBox[{a___, ",", "Null"}], "]"}] :> a, 
			RowBox[{"Times", "[", RowBox[{a___}], "]"}] :> RowBox[{"CompoundExpression", "[", RowBox[{a}], "]"}]], r]
			
removeNewLines2[content_] := 
	If[MatchQ[content, RowBox[_List]] && (Cases[content, "\[IndentingNewLine]" | "\n", {2}] =!= {}),
		DeleteCases[content[[1]], "\[IndentingNewLine]" | "\n"],
		content]

removeNewLines[content_] := 
	If[ListQ@content && (Cases[content, "\[IndentingNewLine]" | "\n", {1}] =!= {}), 
		DeleteCases[content, "\[IndentingNewLine]" | "\n"], content]

removeComments1[content_] := If[Head@content === RowBox && MatchQ[content, RowBox[{RowBox[{"(*", ___, "*)"}], _}]], content[[1, 2]], content]

removeComments2[content_] := 
	content //. {RowBox[{a__, RowBox[{"(*", ___, "*)"}], b___}] :> RowBox[{a, b}], RowBox[{a___, RowBox[{"(*", ___, "*)"}], b__}] :> RowBox[{a, b}]}

normalize[content_] := If[MatchQ[content, RowBox[{PatternSequence[RowBox[{__, ";"}], "\[IndentingNewLine]"] .., _}]], content[[1]], content]

normalize2[content_] := content //. RowBox[{a__, b_, "\[IndentingNewLine]"}] :> RowBox[{a, b}]

normalize3[content_] := content //. RowBox[{"\[IndentingNewLine]", a__, b_}] :> RowBox[{a, b}]

normalize4[content_] := content //. RowBox[{"\[IndentingNewLine]", a__, b_, "\[IndentingNewLine]"}] :> RowBox[{a, b}]

normalize5[content_] := content //. RowBox[{"\[IndentingNewLine]", RowBox[a_List]}] :> RowBox[a]

normalize6[content_] := content //. RowBox[{"\[IndentingNewLine]", RowBox[a_List], "\[IndentingNewLine]"}] :> RowBox[a]

normalize7[content_] := content //. RowBox[{RowBox[a_List], "\[IndentingNewLine]"}] :> RowBox[a]

canonicalizeBoxData[content_]:=removeNewLines2@removeNewLines@normalize7@normalize6@normalize5@normalize4@normalize3@normalize@normalize2@removeComments2@removeComments1@content

fullFormCell[Cell[BoxData[content_], opts___]] := 
	Cell[BoxData[FixTimesComposition[Function[x, MakeBoxes[FullForm[x], StandardForm][[1, 1]],
									{HoldAll}] @@ With[{box = BoxData[If[ListQ@#, RowBox@#, #] &[canonicalizeBoxData@content /. RowBox[{RowBox[a_]}] :> RowBox[a]]]}, 
															MakeExpression[StripBoxes@box, StandardForm]]]], opts]

fullFormCell[other_] := other

rerunButton[]:=buttonWithIconAndTooltip["Rerun", "Rerun-Label", "Rerun-Tooltip", MUnit`testRerun[]]

testResultsDockedCell[] := 
	Module[{significantCells = Cells[CellStyle -> ("TestSuccess" | "TestError" | "TestFailure" | "TestMessageFailure")],
		scells = Cells[CellStyle -> "TestSuccess"], ecells = Cells[CellStyle -> "TestError"],
		fcells = Cells[CellStyle -> "TestFailure"], mfcells = Cells[CellStyle -> "TestMessageFailure"], sindices, eindices, findices, mfindices, cellids}, 
		sindices = Flatten@Position[significantCells, Alternatives @@ scells];
		eindices = Flatten@Position[significantCells, Alternatives @@ ecells];
		findices = Flatten@Position[significantCells, Alternatives @@ fcells];
		mfindices = Flatten@Position[significantCells, Alternatives @@ mfcells];
		cellids = CurrentValue[#, CellID] & /@ significantCells; 
		Cell[BoxData[PaneBox[TagBox[GridBox[{{ToBoxes@MUnit`Palette`Private`reportGrid2by3[Length@sindices + Length@findices, Length@sindices, Length@eindices, Length@findices, Length@mfindices], 
							ItemBox["", ItemSize -> Fit, StripOnInput -> False], 
							ToBoxes@actionMenuResultColorBarAndButton[cellids, sindices, eindices, findices, mfindices, 250, 12]}}, 
							GridBoxAlignment -> {"Columns" -> {{Left}}, "Rows" -> {{Right}}}, 
							GridBoxItemSize -> {"Columns" -> {{Automatic}}, "Rows" -> {{Automatic}}}], "Grid"], 
					FrameMargins -> {{26, 12}, {Automatic, Automatic}}, 
					ImageSize -> Full, 
					BaseStyle -> {Background -> RGBColor[0.827, 0.827, 0.827]}]], 
			Background -> RGBColor[0.827, 0.827, 0.827], "DockedCell", 
			CellTags -> "MUnitResultsCell"]]

failureCell:= Cell[BoxData[ToBoxes[Pane[Grid[{{Grid[{{Style[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "Error-TestError"], White, Bold, 14, FontFamily -> "Arial"], 
          						Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitExpressions", "FailureX"]}},
          					Alignment -> {Automatic, Center}], Item["", ItemSize -> Fit]}}, Alignment -> {Left, Right}, ItemSize -> Full], Full]]], "TestError"]

cellEvaluationFunction[nb_NotebookObject, grouped:(True|False)] :=
	Module[{selection, cells, keyParts, containsErrorBoxQ, testoptionsErrorQ, testoptionspart, tempCells, tempCells2, test, tr, eop, eoAfterPosition, cellafterEO, style},
		selection = NotebookRead[nb];
		(* cells is a list of cells, {test} or {test, eo} etc. *)
		cells =
			If[grouped,
				DeleteCases[Cases[{selection}, Cell[CellGroupData[cells:{___Cell}, _]] :> cells][[1]],
					Cell[_, Except["VerificationTest" | "TestOptions" | "ExpectedMessages" | "ExpectedMessage" | "ExpectedOutput" | "BottomCell" ], ___],
					Infinity]
				,
				{selection}
			];
		(*NotebookDelete[nb];*)
		(* no more harm in writing CellGroupData with one cell, if selection was only a Test cell *)
		Quiet[
		keyParts = Cases[cells, Cell[BoxData[a_], "VerificationTest" | "ExpectedOutput" | "ActualOutput", ___] :> a, Infinity];
		containsErrorBoxQ = (Module[{a},Cases[Quiet[ReleaseHold[(Hold@MakeExpression[StripBoxes@a, StandardForm] /. a -> #) & /@ (BoxData[If[ListQ@#,
									RowBox@#, #] &[canonicalizeBoxData@# /. RowBox[{RowBox[a_]}] :> RowBox[a]]] & /@ keyParts)], Syntax::sntxi], ErrorBox[_]]] =!= {});
		testoptionsErrorQ = And[(testoptionspart = Cases[selection, Cell[BoxData[a_], "TestOptions", __] :> a, Infinity]) =!= {}, Not@MatchQ[testoptionspart, testOptionPatterns]];
		
		If[Not[containsErrorBoxQ || testoptionsErrorQ],
			tempCells = DeleteCases[cells, Cell[_ ,"BottomCell", ___],Infinity];
			tempCells2 = If[MatchQ[#,Cell[BoxData[_], "VerificationTest"|"ExpectedOutput"|"ActualOutput", __]], fullFormCell@#, #]&/@tempCells];
		Catch[
		If[Not[containsErrorBoxQ || testoptionsErrorQ],
			test = CellsToTests[tempCells2];
			(* from string -> expression *)
			tr = ToExpression[test];
			
			With[{atr = tr},
			  
				Which[tr["Outcome"] === "Failure",
			   
					eop = Position[cells, Cell[_, "ExpectedOutput", ___]][[1, 1]]; 
					cells = Insert[cells, Cell[BoxData[ToBoxes[ReleaseHold@tr["ActualOutput"]]], "ActualOutput"], eop + 1]; 
					cells = ReplacePart[cells,
						-1 -> bottomCellWithRightButton[{buttonWithIconAndTooltip["ReplaceOutput", "ReplaceOutput-Label", "ReplaceOutput-Tooltip", testReplaceOutput[atr], 3], 
										MUnit`rerunButton[]}]],
			   
					tr["Outcome"] === "MessagesFailure",
			   
					eoAfterPosition = Position[cells, Cell[_, "ExpectedOutput", ___]][[1, 1]] + 1; 
					cellafterEO = cells[[eoAfterPosition]];
					style = cellafterEO[[2]]; 
					cells = Insert[cells, 
							Cell[BoxData[actualMessagesBoxStructure[tr["ActualMessages"]]], "ActualMessage"], 
   							If[style === "ExpectedMessage", eoAfterPosition + 1, eoAfterPosition]]; 
					cells = ReplacePart[cells,
						-1 -> bottomCellWithRightButton[{buttonWithIconAndTooltip["ReplaceOutput", "ReplaceMessageList-Label", "ReplaceMessageList-Tooltip", testReplaceMessage[atr], 3], 
										MUnit`rerunButton[]}]],
			   
					True,
			   
					cells = ReplacePart[cells, -1 -> bottomCellWithRightButton[{rerunButton[]}]]]]];
			
			cells = If[containsErrorBoxQ || testoptionsErrorQ, Join[Drop[cells, -1], {failureCell, bottomCellWithRightButton[{rerunButton[]}]}], Insert[cells, resultCell[tr], -2]];
			
			NotebookWrite[nb, Cell[CellGroupData[cells, Open]]];
			
			CurrentValue[nb, {TaggingRules, "$testsRun"}] = True;
			CurrentValue[nb, {TaggingRules, "$someTestsFailed"}] = If[Cells[nb, CellStyle -> ("TestFailure" | "TestMessageFailure" | "TestError")] === {}, Inherited, True];
			
			CurrentValue[nb, DockedCells] = If[Head@# === Cell, 
								List[#, #2],
								# /. Cell[__, CellTags -> "MUnitResultsCell", ___] -> #2] &[CurrentValue[nb, DockedCells], testResultsDockedCell[]];
		], Syntax::sntxi]
	]


End[]
