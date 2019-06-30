
PrintLogger::usage =
"PrintLogger[] is an MUnit logger that writes a simple form of test output to $Output."

ProgressLogger::usage =
"ProgressLogger[] is an MUnit logger that prints a progress indicator during the test run."

NotebookLogger::usage =
"NotebookLogger[] is an MUnit logger that writes test results to a notebook."

BatchPrintLogger::usage =
"BatchPrintLogger[] is an MUnit logger that writes batch test results to $Output."

VerbosePrintLogger::usage =
"VerbosePrintLogger[] is an MUnit logger that writes verbose test results to $Output."



LogBegin::usage =
"LogBegin[logger] logs the beginning of a test run."

LogTestRunProgress::usage =
"LogTestRunProgress[logger, progress] logs the progress of a test run."

LogMessage::usage =
"LogMessage[logger, string] logs a message."

LogSuccess::usage =
"LogSuccess[logger, tr] logs a test success."

LogFailure::usage =
"LogFailure[logger, tr] logs a test failure."

LogMessagesFailure::usage =
"LogMessagesFailure[logger, tr] logs a test messages failure."

LogError::usage =
"LogError[logger, tr] logs a test error."

LogTestInfo::usage =
"LogTestInfo[logger, testid, index, willRun] logs a test id and approximate test index, whether or not the test will run."

LogFatal::usage =
"LogFatal[logger, string] logs a fatal error."

LogBeginTestSection::usage =
"LogBeginTestSection[logger, section, require] logs the start of a section."

LogEndTestSection::usage =
"LogEndTestSection[logger] logs the end of a section."

LogCPUTimeUsed::usage =
"LogCPUTimeUsed[logger, time] logs the amount of CPU time used by the test run."

LogAbsoluteTimeUsed::usage =
"LogAbsoluteTimeUsed[logger, time] logs the amount of absolute time used by the test run."

LogMemoryUsed::usage =
"LogMemoryUsed[logger, mem] logs the amount of memory used by the test run."

LogBeginTestSource::usage =
"LogBeginTestSource[logger, source] logs the beginning of a test source in a test run."

LogEndTestSource::usage =
"LogEndTestSource[logger] logs the end of a test source in a test run."

LogEnd::usage =
"LogEnd[logger] logs the end of a test run."

LogTestCount::usage =
"LogTestCount[logger, cnt] logs the number of tests in a test run."

LogSuccessCount::usage =
"LogSuccessCount[logger, cnt] logs the number of successes in a test run."

LogFailureCount::usage =
"LogFailureCount[logger, cnt] logs the number of failures in a test run."

LogMessagesFailureCount::usage =
"LogMessagesFailureCount[logger, cnt] logs the number of messages failures in a test run."

LogSkippedTestCount::usage =
"LogSkippedTestCount[logger, cnt] logs the number of skipped tests in a test run."

LogErrorCount::usage =
"LogErrorCount[logger, cnt] logs the number of errors in a test run."

LogWasAborted::usage =
"LogWasAborted[logger, was] logs whether the test run was aborted."

LogWasFatal::usage =
"LogWasFatal[logger, was] logs whether the test run had a fatal error."

LogWasSyntax::usage =
"LogWasSyntax[logger, was] logs whether the test run had a syntax error."

LogWasSuccessful::usage =
"LogWasSuccessful[logger, succ] logs whether the test run was successful or not."

TestResultsNotebook::usage =
"TestResultsNotebook is an accessor for the notebook logger that returns the results notebook."



LogStart::usage =
"LogStart is deprecated."

Begin["`Package`"]

End[]

Begin["`Loggers`Private`"]

PrintLogger[] :=
	With[{logger = Unique["MUnit`Loggers`Private`logger"]},
		Module[{fatal = False},
			logger /: LogStart[logger, title_] :=
			If[title === None,
				WriteString[$Output, "Starting Test Run\n"],
				WriteString[$Output, "Starting test run \"" <> ToString@title <> "\"\n"]
			];
			logger /: LogBeginTestSection[logger, section_, (*require*)_] :=
				WriteString[$Output, "Starting test section ", section, "\n"];
			logger /: LogEndTestSection[logger] :=
				WriteString[$Output, "\nEnding test section\n"];
			logger /: LogMessage[logger, msg_String] :=
				WriteString[$Output, "\n" <> msg <> "\n"];
			logger /: LogFatal[logger, msg_String] :=
				WriteString[$Output, "\n" <> msg <> "\n"];
			logger /: LogSuccess[logger, (*tr*)_?TestResultQ] :=
				WriteString[$Output, "."];
			logger /: LogFailure[logger, tr_?TestResultQ] :=
				Module[{msg = TestFailureMessage[tr]},
					WriteString[$Output, "!"];
					If[msg != "", WriteString[$Output, "\n** " <> ToString[msg] <> " **\n"]]
				];
			logger /: LogMessagesFailure[logger, (*tr*)_?TestResultQ] :=
				WriteString[$Output, "*"];
			logger /: LogError[logger, tr_] :=
				Module[{msg = ErrorMessage[tr]},
					WriteString[$Output, "!"];
					If[msg != "", WriteString[$Output, "\n** " <> msg <> " **\n"]]
				];
			logger /: LogFatal[logger, msg_String] :=
				(fatal = True;
				WriteString[$Output, "\n" <> msg <> "\n"]);
			logger /: LogEnd[logger, testCnt_, (*successCnt*)_, failCnt_, msgFailCnt_, skippedTestCnt_, errorCnt_, (*abort*)_] :=
				(
					(*
					there is some bug with \n inside of StringForm
					*)
					WriteString[$Output, "\nTests run: " <> ToString[testCnt]];
					WriteString[$Output, "\nFailures: " <> ToString[failCnt]];
					WriteString[$Output, "\nMessages Failures: " <> ToString[msgFailCnt]];
					WriteString[$Output, "\nSkipped Tests: " <> ToString[skippedTestCnt]];
					WriteString[$Output, "\nErrors: " <> ToString[errorCnt]];
					WriteString[$Output, "\nFatal: " <> ToString[fatal]];
					WriteString[$Output, "\n\n"];
				);
			logger /: Format[logger, StandardForm] :=
				Interpretation[Row[{RawBoxes["\[SkeletonIndicator]"], RawBoxes["Print Logger"], RawBoxes["\[SkeletonIndicator]"]}], logger];
			logger /: Format[logger, OutputForm] :=
				"-Print Logger-";
			logger
		]
	]

ProgressLogger[] :=
	With[{logger = Unique["MUnit`Loggers`Private`logger"]},
		Module[{col = Column[{}], progStack = {}},
			logger /: LogStart[logger, title_] := PrintTemporary[Dynamic[col]];
			logger /: LogBeginTestSource[logger, source_] := (progStack = Append[progStack, 0];col = Column[ProgressIndicator /@ progStack]);
			logger /: LogEndTestSource[logger] := (progStack = Most[progStack];col = Column[ProgressIndicator /@ progStack]);
			logger /: LogTestRunProgress[logger, prog_] := (progStack[[-1]] = prog;col = Column[ProgressIndicator /@ progStack]);
			logger
		]
	]

testResultCell[testNb_NotebookObject, tr_?TestResultQ, "TestSuccess", opts___] :=
	Cell[BoxData[RowBox[{ToBoxes[tr], jumpToTestButtonBox[testNb, tr]}]], "TestSuccess", opts]

testResultCell[testNotebookFileName_?noteBookFileQ, tr_?TestResultQ, "TestSuccess", opts___] :=
	Cell[BoxData[RowBox[{ToBoxes[tr], jumpToTestButtonBox[testNotebookFileName, tr]}]], "TestSuccess", opts]

testResultCell[testFileName_String, tr_?TestResultQ, "TestSuccess", opts___] :=
	Cell[BoxData[RowBox[{ToBoxes[tr]}]], "TestSuccess", opts]


testResultCell[testNb_NotebookObject, tr_?TestResultQ, "TestFailure", opts___] :=
	Cell[BoxData[RowBox[{ToBoxes[tr], jumpToTestButtonBox[testNb, tr], replaceOutputButtonBox[testNb, tr]}]], "TestFailure", opts]

testResultCell[testNotebookFileName_?noteBookFileQ, tr_?TestResultQ, "TestFailure", opts___] :=
	Cell[BoxData[RowBox[{ToBoxes[tr], jumpToTestButtonBox[testNotebookFileName, tr]}]], "TestFailure", opts]

testResultCell[testFileName_String, tr_?TestResultQ, "TestFailure", opts___] :=
	Cell[BoxData[RowBox[{ToBoxes[tr]}]], "TestFailure", opts]


testResultCell[testNb_NotebookObject, tr_?TestResultQ, "TestMessagesFailure", opts___] :=
	Cell[BoxData[RowBox[{ToBoxes[tr], jumpToTestButtonBox[testNb, tr], replaceMessagesButtonBox[testNb, tr]}]], "TestMessagesFailure", opts]

testResultCell[testNotebookFileName_?noteBookFileQ, tr_?TestResultQ, "TestMessagesFailure", opts___] :=
	Cell[BoxData[RowBox[{ToBoxes[tr], jumpToTestButtonBox[testNotebookFileName, tr]}]], "TestMessagesFailure", opts]

testResultCell[testFileName_String, tr_?TestResultQ, "TestMessagesFailure", opts___] :=
	Cell[BoxData[RowBox[{ToBoxes[tr]}]], "TestMessagesFailure", opts]


testResultCell[testNb_NotebookObject, tr_?TestResultQ, "TestError", opts___] :=
	Cell[BoxData[RowBox[{ToBoxes[tr], jumpToTestButtonBox[testNb, tr]}]], "TestError", opts]

testResultCell[testNotebookFileName_?noteBookFileQ, tr_?TestResultQ, "TestError", opts___] :=
	Cell[BoxData[RowBox[{ToBoxes[tr], jumpToTestButtonBox[testNotebookFileName, tr]}]], "TestError", opts]

testResultCell[testFileName_String, tr_?TestResultQ, "TestError", opts___] :=
	Cell[BoxData[RowBox[{ToBoxes[tr]}]], "TestError", opts]


fatalCell[testNb_NotebookObject, rules_] :=
	Module[{msgs = "Messages" /. rules, val = "Value" /. rules},
		Cell[BoxData[RowBox[{
			ToBoxes[Column[{msgs[[1]], Style[msgs[[2]], Bold], val}]],
			If[("CellID" /. rules) == "CellID", Sequence @@ {}, Sequence @@ {jumpToFatalButtonBox[testNb, rules]}]
		}]], "TestFatal"]
	]

fatalCell[testNotebookFileName_String /; StringMatchQ[testNotebookFileName, "*.nb"], rules_] :=
	Module[{msgs = "Messages" /. rules, val = "Value" /. rules},
		Cell[BoxData[RowBox[{
			ToBoxes[Column[{msgs[[1]], Style[msgs[[2]], Bold], val}]],
			If[("CellID" /. rules) == "CellID", Sequence @@ {}, Sequence @@ {jumpToFatalButtonBox[testNotebookFileName, rules]}]
		}]], "TestFatal"]
	]

fatalCell[testFileName_String, rules_] :=
	Module[{msgs = "Messages" /. rules, val = "Value" /. rules},
		Cell[BoxData[RowBox[{
			ToBoxes[Column[{msgs[[1]], Style[msgs[[2]], Bold], val}]]
		}]], "TestFatal"]
	]


depthToSectionStyle[depth_Integer] := depth /. {1 -> "Title", 2 -> "Section", 3 -> "Subsection", 4 -> "Subsubsection", 5 -> "Subsubsubsection", _ -> "Unknownsection"}

sectionStyleToDepth[style_String] := style /. {"Title" -> 1, "Section" -> 2, "Subsection" -> 3, "Subsubsection" -> 4, "Subsubsubsection" -> 5, _ -> $UnknownDepth}


initializeNotebook[nbSym_, title_] :=
	If[Head[nbSym] =!= NotebookObject,
		nbSym =
			CreateDocument[
				{},
				WindowTitle -> If[title === None, "MUnit Test Run Results", title],
				StyleDefinitions -> FrontEnd`FileName[{"MUnit"}, "TestResults.nb"],
				Saveable->False,
				(* to accomodate the failure mode pane and the buttons *)
				WindowSize -> {Large, Automatic},
				(* ensure that the test results notebook does not take focus away from the InputNotebook[]
				during the test run, since there are hooks using InputNotebook[] *)
				(*WindowClickSelect -> False*)
				ShowCellTags -> True
			]
	]

(*
testNb is the test notebook object
*)
NotebookLogger[] :=
	With[{logger = Unique["MUnit`Loggers`Private`nblogger"], resNb = Unique["MUnit`Loggers`Private`nb"], dynProg = Unique["MUnit`Loggers`Private`prog"]},
		Module[{$depth = 0},
			logger /: LogStart[logger, title_] :=
				UsingFrontEnd[
					initializeNotebook[resNb, title];
					SetOptions[resNb, DockedCells -> {Cell[BoxData[ProgressIndicatorBox[Dynamic[dynProg]]], "Output"]}]
				];
			logger /: LogTestRunProgress[logger, prog_] :=
				UsingFrontEnd[
					dynProg = prog;
				];
			logger /: LogMessage[logger, msg_String] :=
				UsingFrontEnd[
					NotebookWrite[resNb, Cell[msg, "TestMessage"]];
				];
			logger /: LogSuccess[logger, tr_?TestResultQ] :=
				(KeepTestResult[tr];
				UsingFrontEnd[
					NotebookWrite[resNb, testResultCell[$CurrentTestSource, tr, "TestSuccess", CellTags -> TestTags[tr] ~Prepend~ ToString[TestID[tr]]]];
				]);
			logger /: LogFailure[logger, tr_?TestResultQ] :=
				(KeepTestResult[tr];
				UsingFrontEnd[
					NotebookWrite[resNb, testResultCell[$CurrentTestSource, tr, "TestFailure", CellTags -> TestTags[tr] ~Prepend~ ToString[TestID[tr]]]];
				]);
			logger /: LogMessagesFailure[logger, tr_?TestResultQ] :=
				(KeepTestResult[tr];
				UsingFrontEnd[
					NotebookWrite[resNb, testResultCell[$CurrentTestSource, tr, "TestMessagesFailure", CellTags -> TestTags[tr] ~Prepend~ ToString[TestID[tr]]]];
				]);
			logger /: LogError[logger, tr_?TestResultQ] :=
				(KeepTestResult[tr];
				UsingFrontEnd[
					NotebookWrite[resNb, testResultCell[$CurrentTestSource, tr, "TestError", CellTags -> TestTags[tr] ~Prepend~ ToString[TestID[tr]]]];
				]);
			logger /: LogFatal[logger, msg_String] :=
				UsingFrontEnd[
					NotebookWrite[resNb, Cell[msg, "TestFatal"]];
				];
			logger /: LogFatal[logger, rules:{_Rule...}] :=
				UsingFrontEnd[
					NotebookWrite[resNb, fatalCell[$CurrentTestSource, rules]];
				];
			logger /: LogBeginTestSection[logger, section_, require:True|False] :=
				UsingFrontEnd[
					$depth++;
					(*If[$depth == 1,
						SetOptions[nb, WindowTitle -> "MUnit Test Run Results for " <> section]
					];*)
					NotebookWrite[resNb, Cell[If[require, section, TextData[StyleBox[section, FontVariations->{"StrikeThrough"->True}]]], depthToSectionStyle[$depth]]];
				];
			(* LogEndTestSection goes back up the notebook and groups all of the cells *)
			logger /: LogEndTestSection[logger] :=
				Module[{(*info*)},
					UsingFrontEnd[
						(*While[(info = Developer`CellInformation[nb]) =!= $Failed && ("Style" /. info[[1]]) != depthToSectionStyle[$depth],
							FrontEndTokenExecute[nb, "SelectPreviousCell"]
						];
						$depth--;
						FrontEndTokenExecute[nb, "CellGroup"];
						SelectionMove[nb, After, Notebook]*)
						$depth--;
					]
				];
			logger /: LogCPUTimeUsed[logger, time_] :=
				UsingFrontEnd[
					NotebookWrite[resNb, Cell[TextData[{"Test run used ", Cell[BoxData[ToString[time, InputForm]], "Output"], " seconds of CPU time"}], "TestMessage"]];
				];
			logger /: LogAbsoluteTimeUsed[logger, time_] :=
				UsingFrontEnd[
					NotebookWrite[resNb, Cell[TextData[{"Test run used ", Cell[BoxData[ToString[time, InputForm]], "Output"], " seconds of absolute time"}], "TestMessage"]];
				];
			logger /: LogMemoryUsed[logger, mem_] :=
				UsingFrontEnd[
					NotebookWrite[resNb, Cell["Test run used " <> ToString[mem, OutputForm] <> " bytes of memory", "TestMessage"]];
				];
			logger /: LogEnd[logger, testCnt_, successCnt_, failCnt_, msgFailCnt_, skippedTestCnt_, errorCnt_, abortOrFatal_] :=
				UsingFrontEnd[
					NotebookWrite[resNb,
						Cell[ToString[StringForm["Tests run: `1`,  Failures: `2`,  Messages Failures: `3`,  Skipped Tests: `4`, Errors: `5`", testCnt, failCnt, msgFailCnt, skippedTestCnt, errorCnt], OutputForm],
							If[successCnt == testCnt && errorCnt == 0 && !abortOrFatal, "TestRunSuccess", "TestRunFailure"]]];
					SetOptions[resNb, DockedCells -> {}];
					(* restore clickability after test run *)
					(*SetOptions[resNb, WindowClickSelect -> True]*)
				];
			logger /: TestResultsNotebook[logger] := resNb;
			logger /: Format[logger, StandardForm] :=
				Interpretation[Row[{RawBoxes["\[SkeletonIndicator]"], RawBoxes["Notebook Logger"], RawBoxes["\[SkeletonIndicator]"]}], logger];
			logger /: Format[logger, OutputForm] :=
				"-Notebook Logger-";
			logger
		]
	]


BatchPrintLogger[] :=
	With[{logger = Unique["MUnit`Loggers`Private`logger"], printLogger = PrintLogger[]},
		Module[{$batchLength = 20, doBatchOutput, batchString = ""},
			doBatchOutput[force_:False] :=
				If[force || StringLength[batchString] >= $batchLength,
					WriteString[$Output, batchString];
					batchString = ""
				];
			logger /: LogStart[logger, title_] :=
				(
					LogStart[printLogger, title];
					batchString = ""
				);
			logger /: LogBeginTestSection[logger, section_, (*require*)_] :=
				(
					batchString = batchString <> "Starting test section " <> section <> "\n";
					doBatchOutput[True]
				);
			logger /: LogEndTestSection[logger] :=
				(
					batchString = batchString <> "\nEnding test section\n";
					doBatchOutput[True]
				);
			logger /: LogMessage[logger, msg_String] :=
				(
					batchString = batchString <> "\n" <> msg <> "\n";
					doBatchOutput[True]
				);
			logger /: LogFatal[logger, msg_String] :=
				(
					batchString = batchString <> "\n" <> msg <> "\n";
					doBatchOutput[True]
				);
			logger /: LogSuccess[logger, _?TestResultQ] :=
				(
					batchString = batchString <> ".";
					doBatchOutput[]
				);
			logger /: LogFailure[logger, tr_?TestResultQ] :=
				Module[{msg = TestFailureMessage[tr]},
					batchString = batchString <> "!" <> If[msg =!= "", "\n** " <> ToString[msg] <> " **\n", ""];
					doBatchOutput[]
				];
			logger /: LogMessagesFailure[logger, _?TestResultQ] :=
				(
					batchString = batchString <> "*";
					doBatchOutput[]
				);
			logger /: LogError[logger, tr_?TestResultQ] :=
				Module[{msg = ErrorMessage[tr]},
					batchString = batchString <> "!" <> If[msg =!= "", "\n** " <> ToString[msg] <> " **\n", ""];
					doBatchOutput[]
				];
			logger /: LogEnd[logger, testCnt_, successCnt_, failCnt_, msgFailCnt_, skippedTestCnt_, errorCnt_, abort_] :=
				(
					doBatchOutput[True];
					LogEnd[printLogger, testCnt, successCnt, failCnt, msgFailCnt, skippedTestCnt, errorCnt, abort]
				);
			logger /: Format[logger, StandardForm] :=
				Interpretation[Row[{RawBoxes["\[SkeletonIndicator]"], RawBoxes["Batch Print Logger"], RawBoxes["\[SkeletonIndicator]"]}], logger];
			logger /: Format[logger, OutputForm] :=
				"-Batch Print Logger-";
			logger
		]
	]




VerbosePrintLogger[] :=
	With[{logger = Unique["MUnit`Loggers`Private`logger"]},
		Module[{wasAbortedOrFatal = False},
			logger /: LogStart[logger, title_] :=
			If[title === None,
				WriteString[$Output, "Starting Test Run\n"],
				WriteString[$Output, "Starting test run \"" <> ToString@title <> "\"\n"]
			];
			logger /: LogBeginTestSection[logger, section_, (*require*)_] :=
				WriteString[$Output, "Starting test section ", section, "\n"];
			logger /: LogEndTestSection[logger] :=
				WriteString[$Output, "\nEnding test section\n"];
			logger /: LogMessage[logger, msg_String] :=
				WriteString[$Output, "\n" <> msg <> "\n"];
			logger /: LogFatal[logger, msg_String] :=
				WriteString[$Output, "\n" <> msg <> "\n"];
			logger /: LogSuccess[logger, (*tr*)_?TestResultQ] :=
				WriteString[$Output, "."];
			logger /: LogFailure[logger, tr_?TestResultQ] :=
				Module[{msg = TestFailureMessage[tr]},
					WriteString[$Output, "!\n"];
					WriteString[$Output, "Test number " <> ToString[TestIndex[tr], OutputForm] <> " with TestID " <> ToString[TestID[tr], OutputForm] <> " had a failure.\n"];
					WriteString[$Output, "\tInput: " <> ToString[TestInput[tr], InputForm] <> "\n"];
					WriteString[$Output, "\tExpected output: " <> ToString[ExpectedOutput[tr], InputForm] <> "\n"];
					WriteString[$Output, "\tActual output: " <> ToString[ActualOutput[tr], InputForm] <> "\n"];
					WriteString[$Output, "\tExpected messages: " <> ToString[ExpectedMessages[tr], InputForm] <> "\n"];
					WriteString[$Output, "\tActual messages: " <> ToString[ActualMessages[tr], InputForm] <> "\n"];
					If[msg =!= "", WriteString[$Output, "\n** " <> ToString[msg] <> " **\n"]]
				];
			logger /: LogMessagesFailure[logger, tr_?TestResultQ] :=
				Module[{msg = TestFailureMessage[tr]},
					WriteString[$Output, "*\n"];
					WriteString[$Output, "Test number " <> ToString[TestIndex[tr], OutputForm] <> " with TestID " <> ToString[TestID[tr], OutputForm] <> " had a messages failure.\n"];
					WriteString[$Output, "\tInput: " <> ToString[TestInput[tr], InputForm] <> "\n"];
					WriteString[$Output, "\tExpected messages: " <> ToString[ExpectedMessages[tr], InputForm] <> "\n"];
					WriteString[$Output, "\tActual messages: " <> ToString[ActualMessages[tr], InputForm] <> "\n"];
					If[msg =!= "", WriteString[$Output, "\n** " <> ToString[msg] <> " **\n"]]
				];
			logger /: LogError[logger, tr_?TestResultQ] :=
				Module[{msg = ErrorMessage[tr]},
					WriteString[$Output, "!\n"];
					WriteString[$Output, "Test number " <> ToString[TestIndex[tr], OutputForm] <> " with TestID " <> ToString[TestID[tr], OutputForm] <> " had an error.\n"];
					If[msg =!= "", WriteString[$Output, "\n** " <> msg <> " **\n"]]
				];
			logger /: LogWasAborted[logger, wasAborted_] :=
				(If[wasAborted, wasAbortedOrFatal = True]);
			logger /: LogWasFatal[logger, wasFatal_] :=
				(If[wasFatal, wasAbortedOrFatal = True]);
			logger /: LogEnd[logger, testCnt_, (*successCnt*)_, failCnt_, msgFailCnt_, skippedTestCnt_, errorCnt_, (*abort*)_] :=
				(
					If[wasAbortedOrFatal, WriteString[$Output, "\nTest run stopped before completion.\n"]];
					WriteString[$Output, "\nTests run: " <> ToString[testCnt]];
					WriteString[$Output, "\nFailures: " <> ToString[failCnt]];
					WriteString[$Output, "\nMessages Failures: " <> ToString[msgFailCnt]];
					WriteString[$Output, "\nSkipped Tests: " <> ToString[skippedTestCnt]];
					WriteString[$Output, "\nErrors: " <> ToString[errorCnt]];
					WriteString[$Output, "\nFatal: " <> ToString[wasAbortedOrFatal]];
					WriteString[$Output, "\n\n"];
				);
			logger /: Format[logger, StandardForm] :=
				Interpretation[Row[{RawBoxes["\[SkeletonIndicator]"], RawBoxes["Verbose Print Logger"], RawBoxes["\[SkeletonIndicator]"]}], logger];
			logger /: Format[logger, OutputForm] :=
				"-Verbose Print Logger-";
			logger
		]
	]

End[]
