System`VerificationTest
System`TestResultObject
System`TestReport
System`TestReportObject
System`MemoryConstraint
System`TestID
System`$TestFileName

actualMessagesBoxStructure::usage = "";

AssociationNBLogger::usage = "Special logger for use in Testing Notebooks.";

Begin["`Package`"]

Attributes[VerificationTest] = {HoldAllComplete, ReadProtected}
Attributes[TestResultObject] = {ReadProtected}
Attributes[TestReport] = {HoldFirst, ReadProtected}
Attributes[TestReportObject] = {ReadProtected}
Attributes[$TestFileName] = {ReadProtected}

Options[VerificationTest] = {
		SameTest -> SameQ,
		MemoryConstraint -> Infinity,
		TestID -> None,
		TimeConstraint -> Infinity,
		"TestClass" -> None
	}

Options[TestReport] = {
		SameTest -> SameQ,
		MemoryConstraint -> Infinity,
		TimeConstraint -> Infinity
	}

End[]

Begin["`VerificationTest`Private`"]

$TestFileName :=
    Module[ {source = MUnit`$CurrentTestSource},
        Switch[Head@source,
          NotebookObject, Replace[Quiet[NotebookFileName[source],{NotebookFileName::nosv}], $Failed->""], 
          _, Replace[source, None -> ""]
        ]
    ]

AppendTo[$ContextPath, "BoxForm`"]

iconSuccess = Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitExpressions", "SuccessIcon"]

iconFailure = Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitExpressions", "TestFailureIcon"]

iconMessagesFailure = Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitExpressions", "MessageFailureIcon"]

iconError = Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitExpressions", "ErrorIcon"]

(* SummaryBox code for TestResultObject *)
TestResultObject /: MakeBoxes[tso: TestResultObject[props_], fmt_] /; UseIcons && AssociationQ[Unevaluated @ props] := Module[{alwaysGrid, sometimesGrid},
	alwaysGrid = {MakeSummaryItem[{"Outcome: ", tso["Outcome"]}, fmt], MakeSummaryItem[{"Test ID: ", tso["TestID"]}, fmt]};
	sometimesGrid = {
		MakeSummaryItem[{"Input: ", ElisionsDump`expandablePane[Short[tso["Input"],2/3]]}, fmt],
		MakeSummaryItem[{"Expected output: ", ElisionsDump`expandablePane[Short[tso["ExpectedOutput"],2/3]]}, fmt], 
		MakeSummaryItem[{"Actual output: ", ElisionsDump`expandablePane[Short[tso["ActualOutput"],2/3]]}, fmt]
		};
		If[!(tso["ExpectedMessages"]=={}&&tso["ActualMessages"]=={}),
			AppendTo[sometimesGrid,MakeSummaryItem[{"Expected messages: ", ElisionsDump`expandablePane[Short[tso["ExpectedMessages"],2/3]]}, fmt]];
			AppendTo[sometimesGrid,MakeSummaryItem[{"Actual messages: ", ElisionsDump`expandablePane[Short[tso["ActualMessages"],2/3]]}, fmt]]
		];
		AppendTo[sometimesGrid,MakeSummaryItem[{"Time taken: ", ElisionsDump`expandablePane[Short[tso["AbsoluteTimeUsed"],2/3]]}, fmt]];
	ArrangeSummaryBox[TestResultObject, tso, iconWrapper@Symbol["MUnit`VerificationTest`Private`icon" <> tso["Outcome"]], alwaysGrid, sometimesGrid, fmt]
]


(* Parse arguments for VerificationTest and handle them appropriately *)
HoldPattern[e:VerificationTest[___]]:=
	Block[{parse,args,o,res,input},
		parse = System`Private`ArgumentsWithRules[e,{1,3},Hold];
		(
			{args,o} = parse;
			res = Switch[Length[args],
					1, iVerificationTest[Extract[args,{1},Hold],Hold[True],{},ReleaseHold@o],
					2, iVerificationTest[Extract[args,{1},Hold],Extract[args,{2},Hold],{},ReleaseHold@o],
					3, iVerificationTest[Extract[args,{1},Hold],Extract[args,{2},Hold],Extract[args,{3},Hold],ReleaseHold@o],
					_, $Failed
			];
			res/; res =!= $Failed
		) /; Length[parse] == 2
]


(* Hack to make sure messages are handled appropriately and don't leak *)
SetAttributes[iMUnitTest,HoldAllComplete];
iMUnitTest[in_,exp_, Hold[msgs_], opts_] := MUnit`Test[in,exp,msgs,opts];
iMUnitTest[in_,exp_, {Hold[msgs_]}, opts_] := MUnit`Test[in,exp,{msgs},opts];
iMUnitTest[args__] := MUnit`Test[args];

(* Main implementation of System`VerificationTest *)
iVerificationTest[Hold[input_], Hold[expected_], Shortest[expectedMsgs_:{}], opts:OptionsPattern[]] := 
	Module[{munitResult,newExpectedMsgs},
		With[{defaultOptions=Options[VerificationTest], newOptions=Quiet@SetOptions[VerificationTest,opts]},
			If[Quiet[Part[expectedMsgs,1,0]]===MessageName, newExpectedMsgs={expectedMsgs}, newExpectedMsgs=expectedMsgs]; (* Fix for bug 283126 *)
			munitResult = With[{xmsgs=newExpectedMsgs},iMUnitTest[input, expected, xmsgs, newOptions]];
			With[{result = TestResultObjectToAssociation[munitResult]},
				removeTestObject[munitResult]; (* Delete TestResultObject *)
				Remove[munitResult]; (* unclutter the symbol table *)
				SetOptions[VerificationTest,defaultOptions];
				result
			]
		]
	]

removeTestObject[a_]:=Remove[a];

TestResultObjectToAssociation[tr_?MUnit`TestResultQ] :=
	TestResultObject@Block[{Quantity},Association[
		"TestClass" -> TestClass[tr],
		"TestIndex" -> TestIndex[tr],
		"TestID" -> TestID[tr],
		"Outcome" -> FailureMode[tr],
		"Input" -> TestInput[tr],
		"ExpectedOutput" -> ExpectedOutput[tr],
		"ActualOutput" -> ActualOutput[tr],
		"ExpectedMessages" -> ExpectedMessages[tr],
		"ActualMessages" -> ActualMessages[tr],
		"AbsoluteTimeUsed" -> Quantity[TestAbsoluteTimeUsed[tr],"Seconds"],
		"CPUTimeUsed" -> Quantity[TestCPUTimeUsed[tr],"Seconds"],
		"MemoryUsed" -> Quantity[TestMemoryUsed[tr],"Bytes"]]
	]
	
TestResultObjectToAssociation[badtestresult_] := $Failed

(* Querying TestResultObject *)
TestResultObject[x_Association][property_String]:= If[property==="Properties", Keys[x], x[property]];
TestResultObject[x_Association][propertylist_List]:= KeyTake[x,propertylist];

(*****************************************************************************)

(* Parse arguments for TestReport and handle them appropriately *)
HoldPattern[e:TestReport[origargs___]]:=
	Block[{parse,args,o,res},
		parse = System`Private`Arguments[e,1,HoldForm];
		(
			{args,o} = parse;
			res = Switch[Length[args],
				1, iTestReport[origargs],
				_, $Failed];
			res/; res =!= $Failed
		) /; Length[parse] == 2
]

SetAttributes[iTestReport,HoldFirst];

(* Case of TestReport working on a list of VerificationTests - see bug 297800 *)
iTestReport[TROList_, opts:OptionsPattern[]]/;(MatchQ[Hold@TROList,Hold@{_TestResultObject..}] || (MatchQ[Hold@TROList,Hold@{_Symbol..}]&&MatchQ[TROList,{_TestResultObject..}]) ||
											  (Not[MemberQ[{List,Table,String,File},Extract[Hold@TROList,{1,0}]]]&&MatchQ[TROList,{_TestResultObject..}])):=
	Module[{testResults = <||>},
			AppendTo[testResults, "Title" -> "Automatic"];
			AppendTo[testResults, "TestResults" -> AssociationThread[Range[Length@TROList],TROList]];
			AppendTo[testResults, "TestsSucceededIndices" -> 
				Map[First, 
					Normal[Select[testResults["TestResults"], #["Outcome"] === "Success" &]]
				]
			];
			AppendTo[testResults, "TestsFailedIndices" ->
				Map[First,
					Normal[Select[testResults["TestResults"], (#["Outcome"] === "Failure" || #["Outcome"] === "MessagesFailure" || #["Outcome"] === "Error")&]]
				]
			];
			AppendTo[testResults, "TestsFailedWrongResultsIndices" ->
				Map[First,
					Normal[Select[testResults["TestResults"], #["Outcome"] === "Failure" &]]
				]
			];
			AppendTo[testResults, "TestsFailedWithMessagesIndices" ->
				Map[First,
					Normal[Select[testResults["TestResults"], #["Outcome"] === "MessagesFailure" &]]
				]
			];
			AppendTo[testResults, "TestsFailedWithErrorsIndices" ->
				Map[First,
					Normal[Select[testResults["TestResults"], #["Outcome"] === "Error" &]]
				]
			];
			AppendTo[testResults, "TimeElapsed" -> Total[#["AbsoluteTimeUsed"]&/@testResults["TestResults"]]];
			AppendTo[testResults, "TestsSucceededCount" -> Length@testResults["TestsSucceededIndices"]];
			AppendTo[testResults, "TestsFailedCount" -> Length@testResults["TestsFailedIndices"]];
			AppendTo[testResults, "TestsFailedWrongResultsCount" -> Length@testResults["TestsFailedWrongResultsIndices"]];
			AppendTo[testResults, "TestsFailedWithMessagesCount" -> Length@testResults["TestsFailedWithMessagesIndices"]];
			AppendTo[testResults, "TestsFailedWithErrorsCount" -> Length@testResults["TestsFailedWithErrorsIndices"]];
			AppendTo[testResults, "Aborted" -> False];
			TestReportObject[testResults]
]

(* Main TestReport case : Given a test file or test notebook, calls MUnit`TestRun *)
iTestReport[testSource_, opts:OptionsPattern[]] := 
	Module[{testResults},
			MUnit`TestRun[testSource, Loggers :> {AssociationLogger[testResults]}, opts];
			AppendTo[testResults, "TestsSucceededIndices" -> 
				Map[First, 
					Normal[Select[testResults["TestResults"], #["Outcome"] === "Success" &]]
				]
			];
			AppendTo[testResults, "TestsFailedIndices" ->
				Map[First,
					Normal[Select[testResults["TestResults"], (#["Outcome"] === "Failure" || #["Outcome"] === "MessagesFailure" || #["Outcome"] === "Error")&]]
				]
			];
			AppendTo[testResults, "TestsFailedWrongResultsIndices" ->
				Map[First,
					Normal[Select[testResults["TestResults"], #["Outcome"] === "Failure" &]]
				]
			];
			AppendTo[testResults, "TestsFailedWithMessagesIndices" ->
				Map[First,
					Normal[Select[testResults["TestResults"], #["Outcome"] === "MessagesFailure" &]]
				]
			];
			AppendTo[testResults, "TestsFailedWithErrorsIndices" ->
				Map[First,
					Normal[Select[testResults["TestResults"], #["Outcome"] === "Error" &]]
				]
			];
			TestReportObject[testResults]
]

(* Querying TestReportObject *)
TestReportObject[x_Association][property_String]:= Module[{},
		Which[property==="Properties", 
			Sort@Join[Keys[x],{"TestResultRules","TestsSucceeded","TestsFailedWrongResults","TestsFailedWithMessages","TestsFailedWithErrors","TestsFailed","AllTestsSucceeded"}]
			, 
			property==="TestsSucceeded",
				KeyTake[x["TestResults"],x["TestsSucceededIndices"]]
			,
			property==="TestsFailed",
				Association["TestsFailedWrongResults"->KeyTake[x["TestResults"],x["TestsFailedWrongResultsIndices"]], 
							"TestsFailedWithMessages"-> KeyTake[x["TestResults"],x["TestsFailedWithMessagesIndices"]], 
							"TestsFailedWithErrors"-> KeyTake[x["TestResults"],x["TestsFailedWithErrorsIndices"]]]
			,
			property==="TestsFailedWrongResults",
				KeyTake[x["TestResults"],x["TestsFailedWrongResultsIndices"]]
			,
			property==="TestsFailedWithMessages",
				KeyTake[x["TestResults"],x["TestsFailedWithMessagesIndices"]]
			,
			property==="TestsFailedWithErrors",
				KeyTake[x["TestResults"],x["TestsFailedWithErrorsIndices"]]
			,
			property==="TestResultRules",
				Association[
					"TestsSucceeded"->KeyTake[x["TestResults"],x["TestsSucceededIndices"]], 
					"TestsFailedWrongResults"->KeyTake[x["TestResults"],x["TestsFailedWrongResultsIndices"]], 
					"TestsFailedWithMessages"-> KeyTake[x["TestResults"],x["TestsFailedWithMessagesIndices"]],
					"TestsFailedWithErrors"-> KeyTake[x["TestResults"],x["TestsFailedWithErrorsIndices"]]]
			,
			property==="AllTestsSucceeded"
			,
				x["TestsFailedCount"]===0 && !x["Aborted"]
			,
			property==="TestResultsDataset"
			,
			    Dataset[Function[{tr}, 
                   Association[Function[{z}, z -> tr[z]] /@ Select[tr["Properties"],#=!="TestIndex"&]]] /@ Values[x["TestResults"]]]
			,
			True,
				x[property]
		]
];

(* Querying TestReportObject with a list of properties *)
TestReportObject[x_Association][propertylist_List]:= Module[{temp, res},
		If[!And@@Map[KeyExistsQ[x,#]&, propertylist],
			temp=Join[x, Association["TestsSucceeded"-> KeyTake[x["TestResults"],x["TestsSucceededIndices"]]
				, "TestsFailedWrongResults"-> KeyTake[x["TestResults"],x["TestsFailedWrongResultsIndices"]]
				, "TestsFailedWithMessages"-> KeyTake[x["TestResults"],x["TestsFailedWithMessagesIndices"]]
				, "TestsFailedWithErrors"-> KeyTake[x["TestResults"],x["TestsFailedWithErrorsIndices"]]
				, "TestsFailed"-> Association["TestsFailedWrongResults"->KeyTake[x["TestResults"],x["TestsFailedWrongResultsIndices"]], "TestsFailedWithMessages"-> KeyTake[x["TestResults"], x["TestsFailedWithMessagesIndices"]], "TestsFailedWithErrors"-> KeyTake[x["TestResults"],x["TestsFailedWithErrorsIndices"]]]
				, "TestResultRules"-> Association["TestsSucceeded"->KeyTake[x["TestResults"],x["TestsSucceededIndices"]], "TestsFailedWrongResults"->KeyTake[x["TestResults"],x["TestsFailedWrongResultsIndices"]], "TestsFailedWithMessages"-> KeyTake[x["TestResults"],x["TestsFailedWithMessagesIndices"]], "TestsFailedWithErrors"-> KeyTake[x["TestResults"],x["TestsFailedWithErrorsIndices"]]]]
			];
			res=KeyTake[temp,propertylist]
			,
			res=KeyTake[x,propertylist]
		]
]

iconWrapper[x_] := Graphics[{Inset[Pane[x, ElisionsDump`defaultIconSize, Alignment -> Center]]}, PlotRange -> {{0, 1}, {0, 1}}, 
				Background -> GrayLevel[0.93], Axes -> False, AspectRatio -> 1,
				ImageSize -> Dynamic@{Automatic, 3.5 CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]},
				Frame -> True, FrameTicks -> None, FrameStyle -> Directive[Thickness[Tiny], GrayLevel[0.55]]]

(* SummaryBox code for TestReportObject *)
TestReportObject /: MakeBoxes[tsao: TestReportObject[props_], fmt_] /; UseIcons && AssociationQ[Unevaluated @ props] := Module[
	{alwaysGrid, sometimesGrid, succeeded, totalfailed, resultfailed, messageFailed, error, total, icon},
	succeeded = tsao["TestsSucceededCount"];
	totalfailed = tsao["TestsFailedCount"];
	resultfailed = tsao["TestsFailedWrongResultsCount"];
	messageFailed = tsao["TestsFailedWithMessagesCount"];
	error = tsao["TestsFailedWithErrorsCount"];
	total = succeeded+totalfailed;
	Which[
		total === 0 || resultfailed =!= 0,
		icon = iconFailure,
		
		messageFailed =!= 0,
		icon = iconMessagesFailure,
		
		error =!= 0,
		icon = iconError,
		
		True,
		icon = iconSuccess
	];
	alwaysGrid = {
		{MakeSummaryItem[{"Title: ", tsao["Title"]}, fmt], SpanFromLeft}, 
		{
			MakeSummaryItem[{"Success rate: ", If[total===0, Indeterminate, Row[{Round[100. succeeded/total], "%"}]]}, fmt],
			MakeSummaryItem[{"Tests run: ", total}, fmt]
		}
	};
	sometimesGrid = {#, SpanFromLeft} & /@ {
		MakeSummaryItem[{"Succeeded: ", succeeded}, fmt], 
		MakeSummaryItem[{"Failed: ", totalfailed}, fmt], 
		MakeSummaryItem[{"Failed with wrong results: ", resultfailed}, fmt],
		MakeSummaryItem[{"Failed with messages: ", messageFailed}, fmt],
		MakeSummaryItem[{"Failed with errors: ", error}, fmt] 
	};
	ArrangeSummaryBox[TestReportObject, tsao, iconWrapper@icon, alwaysGrid, sometimesGrid, fmt]
]

(*****************************************************************************)
(* Special logger for use in TestReport *)
SetAttributes[AssociationLogger, HoldFirst]
AssociationLogger[dest_] :=
	With[{logger = Unique["MUnit`Loggers`Private`logger"], 
		tStart = Unique["MUnit`Loggers`AssociationLogger`Private`tStart"],
		results = Unique["MUnit`Loggers`AssociationLogger`Private`results"],
		sectionStack = Unique["MUnit`Loggers`AssociationLogger`Private`sectionStack"]},
		dest = <| |>;
		results = <| |>; (* accumulate individual test results *)
		sectionStack = <| |>; (* maintain this *)

		logger /: MUnit`LogStart[logger, title_] := 
			(
				tStart = AbsoluteTime[];
				AppendTo[dest, "Title" -> title]
			);

		logger /: LogBeginTestSection[logger, "",(*require*)_] := 
			(* ignore the empty test section generated by MUnit internals *)
			Null;

		logger /: LogBeginTestSection[logger, section_,(*require*)_] := 
			Null;

		logger /: LogEndTestSection[logger] := 
			Null;
   
		logger /: LogMessage[logger, msg_String] := 
			AppendTo[results, {"Event", "Message"} -> {DateList[], msg}];
   
		logger /: LogFatal[logger, msg_String] := 
			AppendTo[results, {"Event", "Fatal"} -> {DateList[], msg}];
   
		logger /: LogSuccess[logger, tr_?TestResultQ] :=
			MUnit`Loggers`AssociationLogger`Private`logTestResult[logger, tr];
   
		logger /: LogFailure[logger, tr_?TestResultQ] :=
			MUnit`Loggers`AssociationLogger`Private`logTestResult[logger, tr];
   
		logger /: LogMessagesFailure[logger, tr_?TestResultQ] :=
			MUnit`Loggers`AssociationLogger`Private`logTestResult[logger, tr];
   
		logger /: LogError[logger, tr_?TestResultQ] :=
			MUnit`Loggers`AssociationLogger`Private`logTestResult[logger, tr];
   
		logger /: MUnit`Loggers`AssociationLogger`Private`logTestResult[logger, 
			tr_?TestResultQ, extra_: {}] := 
			AppendTo[results, TestIndex[tr] -> 
				Fold[Append, TestResultObjectToAssociation[tr], extra]];

		logger /: LogEnd[logger, testCnt_, successCnt_, failCnt_, msgFailCnt_, 
			skippedTestCnt_, errorCnt_, abort_] :=
    			(Block[{Quantity},AppendTo[dest, "TimeElapsed" -> Quantity[Round[AbsoluteTime[] - tStart, 0.01],"Seconds"]]];
				(*AppendTo[dest, "TestsRun" -> testCnt];*)
				AppendTo[dest, "TestsSucceededCount" -> successCnt];
				AppendTo[dest, "TestsFailedCount" -> failCnt + msgFailCnt + errorCnt];
				AppendTo[dest, "TestsFailedWrongResultsCount" -> failCnt];
				AppendTo[dest, "TestsFailedWithMessagesCount" -> msgFailCnt];
				(* For future release
				AppendTo[dest, "TestSkipCount" -> skippedTestCnt];*)
				AppendTo[dest, "TestsFailedWithErrorsCount" -> errorCnt];
				AppendTo[dest, "Aborted" -> abort];
				AppendTo[dest, "TestResults" -> results];
				dest
			);
   
		logger
	];

(*****************************************************************************)

colorCellBrackets[nb_] := 
	Module[{len, i},
		len = Length@$problemCellids; i = 1; 
		SelectionMove[nb, Before, Notebook]; 
		While[i <= len, NotebookFind[nb, $problemCellids[[i]], All, CellID]; i++; SetOptions[NotebookSelection[nb], CellBracketOptions -> {"Color" -> RGBColor[0.831373, 0.266667, 0.0431373], "Thickness" -> 2}]]]
		
actualMessagesBoxStructure[actualmessages_] := 
	Module[{messagesNumber = Length@actualmessages}, 
		Switch[messagesNumber,
			0,
			RowBox[{"{", "}"}],
			1, 
			RowBox[{"{", (actualmessages /. HoldForm[Message[MessageName[a_, b_], ___]] :> RowBox[{ToString@a, "::", ToString@b}])[[1]], "}"}],
			_, 
			RowBox[{"{", RowBox[Riffle[actualmessages /. HoldForm[Message[MessageName[a_, b_], ___]] :> RowBox[{ToString@a, "::", ToString@b}], ","]], "}"}]]]
			
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

failureCell:= Cell[BoxData[ToBoxes[Pane[Grid[{{Grid[{{Style[Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitStrings", "Error-TestError"], White, Bold, 14, FontFamily -> "Arial"], 
          						Dynamic@RawBoxes@FEPrivate`FrontEndResource["MUnitExpressions", "FailureX"]}},
          					Alignment -> {Automatic, Center}], Item["", ItemSize -> Fit]}}, Alignment -> {Left, Right}, ItemSize -> Full], Full]]], "TestError"]

SetAttributes[AssociationNBLogger, HoldFirst]

(* Special logger for use in Testing Notebooks.
testNb is the test notebook object*)
AssociationNBLogger[testNB_] :=
	With[{logger = Unique["MUnit`Loggers`Private`AssociationNBLogger"],
		dynProg = Unique["MUnit`Loggers`Private`prog"],
		tStart = Unique["MUnit`Loggers`AssociationNBLogger`Private`tStart"],
		results = Unique["MUnit`Loggers`AssociationNBLogger`Private`results"],
		dest = Unique["MUnit`Loggers`AssociationNBLogger`Private`dest"]
		},
		dest = <| |>;
		results = <| |>; (* accumulate individual test results *)
		Module[{$depth = 0, tReport, cellids},
			logger /: MUnit`LogStart[logger, title_] := 
				(
				FrontEndExecute[{FrontEnd`SelectionMove[testNB,Before,Notebook, AutoScroll->False]}];
				tStart = AbsoluteTime[];
				AppendTo[dest, "Title" -> title]
			);
			logger /: LogTestRunProgress[logger, prog_] :=
				UsingFrontEnd[
					dynProg = prog;
				];
   
			logger /: LogMessage[logger, msg_String] :=(
				AppendTo[results, {"Event", "Message"} -> {DateList[], msg}];);

			logger /: LogSuccess[logger, tr_?TestResultQ] := 
				MUnit`Loggers`AssociationNBLogger`Private`logTestResult[logger, tr];

			logger /: LogFailure[logger, tr_?TestResultQ] :=
				MUnit`Loggers`AssociationNBLogger`Private`logTestResult[logger, tr];

			logger /: LogMessagesFailure[logger, tr_?TestResultQ] :=
				MUnit`Loggers`AssociationNBLogger`Private`logTestResult[logger, tr];

			logger /: LogError[logger, tr_?TestResultQ] :=
				MUnit`Loggers`AssociationNBLogger`Private`logTestResult[logger, tr];
			
			logger /: LogFatal[logger, msg_String] := 
			AppendTo[results, {"Event", "Fatal"} -> {DateList[], msg}];

			logger /: MUnit`Loggers`AssociationNBLogger`Private`logTestResult[logger, tr_?TestResultQ, extra_: {}] := (
				NotebookFind[testNB,"BottomCell",Next,CellStyle, AutoScroll->False];
				Module[{testCellGroupStyleLists = {{"VerificationTest", "ExpectedOutput", "BottomCell"},
				    					{"VerificationTest", "ExpectedOutput", "ExpectedMessage", "BottomCell"},
									{"VerificationTest", "ExpectedOutput", "TestOptions", "BottomCell"},
									{"VerificationTest", "ExpectedOutput", "ExpectedMessage", "TestOptions", "BottomCell"}},
					cells = Cells[testNB], cellids, verifCellIDs, presentCellID, presentPosition, verifCellIDAbove, positionAboveVerif, relevantCells, cellGroupStylesInPresentGroup, 
					preinitialList, keyParts, containsErrorBoxQ, testoptionsErrorQ, intialList, newList},
				With[{ntr = TestResultObjectToAssociation[tr]},
					cellids = CurrentValue[#, CellID] & /@ cells;
					verifCellIDs = CurrentValue[#, CellID] & /@ Cells[testNB, CellStyle -> "VerificationTest"];
					presentCellID = CurrentValue[SelectedCells[testNB][[1]], CellID];
					presentPosition = Position[cellids, presentCellID][[1, 1]];
					verifCellIDAbove = SelectFirst[Reverse@Take[cellids, {1, presentPosition}], MemberQ[verifCellIDs, #] &];
					If[verifCellIDAbove =!= Missing["NotFound"]
						,
						positionAboveVerif = Position[cellids, verifCellIDAbove][[1, 1]];
						relevantCells = Take[cells, {positionAboveVerif, presentPosition}];
						cellGroupStylesInPresentGroup = "Style" /. (Developer`CellInformation /@ relevantCells);
						If[MatchQ[cellGroupStylesInPresentGroup, Alternatives @@ testCellGroupStyleLists]
							,
							preinitialList = NotebookRead[Drop[relevantCells, -1]];
							keyParts = Cases[preinitialList, Cell[BoxData[a_], "VerificationTest" | "ExpectedOutput" | "ActualOutput", ___] :> a, Infinity];
							containsErrorBoxQ = (Module[{a},Cases[Quiet[ReleaseHold[(Hold@MakeExpression[StripBoxes@a, StandardForm] /. a -> #) & /@ (BoxData[If[ListQ@#,
								RowBox@#, #] &[MUnit`canonicalizeBoxData@# /. RowBox[{RowBox[a_]}] :> RowBox[a]]] & /@ keyParts)], Syntax::sntxi], ErrorBox[_]]] =!= {});
							testoptionsErrorQ = And[(testoptionspart = Cases[preinitialList, Cell[BoxData[a_], "TestOptions", __] :> a, Infinity]) =!= {}, 
										Not@MatchQ[testoptionspart, testOptionPatterns]];
							
							intialList = Append[preinitialList, "BottomCell"];
							If[Not[containsErrorBoxQ || testoptionsErrorQ],
							newList = Switch[ntr["Outcome"]
									, 
									"Failure"
									,
				If[containsErrorBoxQ || testoptionsErrorQ, DeleteCases[#, Cell[_, "ActualOutput", ___]], #]&[(intialList /. (a : Cell[_, "ExpectedOutput", __] :> Sequence[a,
										Cell[BoxData[ToBoxes[ReleaseHold@ntr["ActualOutput"]]], "ActualOutput"]])) /. 
									"BottomCell" -> Sequence[resultCell[ntr], 
				bottomCellWithRightButton[{buttonWithIconAndTooltip["ReplaceOutput", "ReplaceOutput-Label", "ReplaceOutput-Tooltip", testReplaceOutput[ntr], 3], MUnit`rerunButton[]}]]]
									,
									"MessagesFailure"
									, 
									If[Cases[intialList, Cell[_, "ExpectedMessage", __]] =!= {}, 
										intialList /. a : Cell[_, "ExpectedMessage", __] :> 
													Sequence[a, Cell[BoxData[actualMessagesBoxStructure[ntr["ActualMessages"]]], "ActualMessage"]], 
										intialList /. a : Cell[_, "ExpectedOutput", __] :> 
													Sequence[a, Cell[BoxData[actualMessagesBoxStructure[ntr["ActualMessages"]]], "ActualMessage"]]] /. 
									"BottomCell" -> Sequence[resultCell[ntr], bottomCellWithRightButton[{buttonWithIconAndTooltip["ReplaceOutput",
										"ReplaceMessageList-Label", "ReplaceMessageList-Tooltip", testReplaceMessage[ntr], 3],MUnit`rerunButton[]}]]
									, 
									_
									, 
									intialList /. "BottomCell" -> Sequence[resultCell[ntr], bottomCellWithRightButton[{MUnit`rerunButton[]}]]]];
									
							NotebookDelete@relevantCells; 
							NotebookWrite[testNB, Cell[CellGroupData[If[containsErrorBoxQ || testoptionsErrorQ,
													Join[(*Drop[newList, -2]*)preinitialList, {failureCell, bottomCellWithRightButton[{rerunButton[]}]}],
													newList], Open]]];
							AppendTo[results, TestIndex[tr] -> Fold[Append, ntr, extra]]]]]];
				
				CurrentValue[testNB, {TaggingRules, "$testsRun"}] = True;
				CurrentValue[testNB, {TaggingRules, "$someTestsFailed"}] = If[Cells[testNB, CellStyle -> ("TestFailure" | "TestMessageFailure" | "TestError")] === {}, Inherited, True]);
			
			logger /: LogBeginTestSection[logger, "",(*require*)_] := 
			(* ignore the empty test section generated by MUnit internals *)
				Null;

			logger /: LogBeginTestSection[logger, section_,(*require*)_] := 
				Null;
	
			logger /: LogEndTestSection[logger] := 
				Null;
   

			logger /: LogEnd[logger, testCnt_, successCnt_, failCnt_, msgFailCnt_, skippedTestCnt_, errorCnt_, abort_] :=
    			(
					Block[{Quantity},AppendTo[dest, "TimeElapsed" -> Quantity[Round[AbsoluteTime[] - tStart, 0.01],"Seconds"]]];
					(*AppendTo[dest, "TestsRun" -> testCnt];*)
					AppendTo[dest, "TestsSucceededCount" -> successCnt];
					AppendTo[dest, "TestsFailedCount" -> failCnt + msgFailCnt + errorCnt];
					AppendTo[dest, "TestsFailedWrongResultsCount" -> failCnt];
					AppendTo[dest, "TestsFailedWithMessagesCount" -> msgFailCnt];
					(* For future release
					AppendTo[dest, "TestSkipCount" -> skippedTestCnt];*)
					AppendTo[dest, "TestsFailedWithErrorsCount" -> errorCnt];
					AppendTo[dest, "Aborted" -> abort];
					AppendTo[dest, "TestResults" -> results];
                    (*Appending the indices*)
					AppendTo[dest, "TestsSucceededIndices" -> 
						Map[First, 
							Normal[Select[dest["TestResults"], #["Outcome"] === "Success" &]]
						]
					];
					AppendTo[dest, "TestsFailedIndices" ->
						Map[First,
							Normal[Select[dest["TestResults"], (#["Outcome"] === "Failure" || #["Outcome"] === "MessagesFailure" || #["Outcome"] === "Error")&]]
						]
					];
					AppendTo[dest, "TestsFailedWrongResultsIndices" ->
						Map[First,
							Normal[Select[dest["TestResults"], #["Outcome"] === "Failure" &]]
						]
					];
					AppendTo[dest,  "TestsFailedWithErrorsIndices" ->
						Map[First,
							Normal[Select[dest["TestResults"], #["Outcome"] === "Error" &]]
						]
					];
					AppendTo[dest, "TestsFailedWithMessagesIndices" ->
						Map[First,
							Normal[Select[dest["TestResults"], #["Outcome"] === "MessagesFailure" &]]
						]
					];
					tReport=TestReportObject[dest];
					(*If[(tReport["TestsSucceededCount"] + tReport["TestsFailedCount"])=!=0,
					   cellids = CurrentValue[#, CellID] & /@ Cells[CellStyle -> ("TestSuccess" | "TestError" | "TestFailure" | "TestMessageFailure")];
					   CurrentValue[testNB,DockedCells]=List[CurrentValue[testNB,DockedCells],testResultsDockedCell[cellids, tReport]]];*)
				);
   
			logger /: TestResultsNotebook[logger] := testNB;
			logger /: Format[logger, StandardForm] :=
				Interpretation[Row[{RawBoxes["\[SkeletonIndicator]"], RawBoxes["Notebook Logger"], RawBoxes["\[SkeletonIndicator]"]}], logger];
			logger /: Format[logger, OutputForm] :=
				"-Notebook Logger-";
			logger
		]
	]

End[]
