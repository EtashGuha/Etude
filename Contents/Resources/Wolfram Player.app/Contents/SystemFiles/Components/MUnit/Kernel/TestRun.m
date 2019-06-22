(* ::Package:: *)

TestRun::usage =
"TestRun[tests] runs tests and returns True if all tests pass, and False otherwise."

TestRunTitle::usage =
"TestRunTitle is an option for TestRun."

BeginTestSection::usage =
"BeginTestSection[section] indicates the beginning of a related group of tests that can be skipped by giving TestFailureAction -> \"SkipSection\" in a Test.\n
BeginTestSection[section, require] gives a requirement for the section to run. If require is False, the section is skipped."

EndTestSection::usage =
"EndTestSection[] marks the end of a test section."

TestLog::usage =
"TestLog[msg] writes msg to the test loggers."

TestTerminate::usage =
"TestTerminate[msg] logs the given message and terminates the current test run."

Loggers::usage =
"Loggers is an option to TestRun that specifies the loggers to use."

CanaryCondition::usage =
"CanaryCondition is an option to TestRun that specfies if the first test of a section should skip the rest of the section on failure."

$CurrentTestSource::usage =
"$CurrentTestSource is the current source of running tests. \
It is either a String, a NotebookObject, None, or $Failed (indicating that the test source could not be read)."

KeepTestResult::usage =
"KeepTestResult[tr] prevents the test result tr from being released when the containing TestRun is finished. Used within loggers."

TestSuite::usage =
"TestSuite[files] allows nesting of multiple files within a test run."

MUnitErrorTag::usage =
"Throw tag used internally by MUnit"

$CurrentFile::usage =
"$CurrentFile gives the path to the currently executing test file. See also $CurrentTestSource."

Begin["`Package`"]

$testIndex
$dynamicTestIndex
$lexicalTestIndex

testSourceStack
progressStack

logTestResult
logInfo

MUnitGetFileHandler

noteBookFileQ

End[]

Begin["`TestRun`Private`"]

(*
set this here so that Tests can be run outside of TestRun
*)
$testIndex = 0
$dynamicTestIndex = 0
$lexicalTestIndex
testSourceStack = {None}
progressStack = {0}


$CurrentFile := $CurrentTestSource /. _NotebookObject -> None
$CurrentTestSource := Last[testSourceStack]

TestRun::badsec =
"Invalid section primitive: `1`."

TestRun::nostack =
"EndTestSection[] was called without a matching call to BeginTestSection."

TestRun::strnbobj = 
"String or NotebookObject expected in position `1` in `2`."

BeginTestSection::require =
"Section requirement did not evaluate to True or False: `1`."

KeepTestResult::tr =
"KeepTestResult was not given a test result: `1`."

TestRun::suite =
"Invalid TestSuite expression: `1`."

TestRun::empty =
"Test file `1` is empty"

Options[TestRun] = {
		Loggers :> {PrintLogger[]},
		CanaryCondition -> False,
		TestRunTitle -> Automatic,
		SameTest -> SameQ,
		MemoryConstraint -> Infinity,
		TimeConstraint -> Infinity
	}

SetAttributes[TestRun,HoldFirst];

(* Fix for bug 297171 : If there's a Table of Tests we construct the table without running the tests
before passing it to the main routine *)
TestRun[Table[head_?(#==Test||#==VerificationTest&)[expr__], t__], arg___]:= 
Module[{temp},
	With[{new1 = Table[temp[expr],t]},
		With[{new2 = Hold@new1/.temp->head},
			TestRun[new2, arg]
		]
	]
];

(* Also necessary for 297171, strip out a layer of Hold *)
TestRun[Hold[x_List], arg___]:= TestRun[x, arg];

(* Strip out File wrapper *)
TestRun[File[testSource_], arg___]:= TestRun[testSource, arg];

TestRun[testSource_, OptionsPattern[]] :=
	Module[{$loggers, ts, title, ccondition,
		cpuTimeUsed, absTimeUsed, memoryUsed, $actualTestIndex,
		successCnt, failureCnt, msgFailureCnt, skippedTestCnt,
		errorCnt, $abort, $fatal, $syntax,
		wasSuccessful,keptResObjNames = {}, 
		sameTest, mConstraint, tConstraint},
		
		{$loggers, title, ccondition, sameTest, mConstraint, tConstraint} = OptionValue[{Loggers, TestRunTitle, CanaryCondition, SameTest, MemoryConstraint, TimeConstraint}];
		
		(* Make sure the Title is set appropriately. See also bugs 281170 and 279263 *)
		If[title === Automatic && Switch[Part[Hold[testSource],1,0], 
                                         List, False,
                                         _, ts=testSource;(MatchQ[ts,_?StringQ]||MatchQ[ts,File[_?StringQ]])],
			title = "Test Report: " <> FileNameTake[testSource];
		];
		
		(* Start logging *)
		Scan[LogStart[#, title]&, $loggers];
		Scan[LogBegin[#, title]&, $loggers];
		Scan[LogStart[#, title, Unevaluated@testSource]&, $loggers];
		Scan[LogBegin[#, title, Unevaluated@testSource]&, $loggers];
		
		Catch[
			Block[{testSourceStack = {}, progressStack = {}, KeepTestResult},
				KeepTestResult[tr_?TestResultQ] := (AppendTo[keptResObjNames, Context[Evaluate[tr]] <> SymbolName[tr]]);
				KeepTestResult[tr_] := (Throw[StringForm[KeepTestResult::tr, tr], "internalMUnitTestTag"]);
				
				(*
				Manually passing the options works around a subtlety:
				Loggers uses a RuleDelayed, and we only want it to evaluate once
				if we simply passed in opts here, then Loggers will have evaluated once above,
				and then again inside testRun, resulting in 2 different loggers
				*)
				{cpuTimeUsed, absTimeUsed, memoryUsed, $actualTestIndex,
				  successCnt, failureCnt, msgFailureCnt, skippedTestCnt,
				  errorCnt, $abort, $fatal, $syntax} = midTestRun[testSource, Loggers -> $loggers, TestRunTitle -> title, CanaryCondition -> ccondition, SameTest -> sameTest, MemoryConstraint -> mConstraint, TimeConstraint -> tConstraint];
				wasSuccessful = (successCnt == $actualTestIndex && errorCnt == 0 && !($abort || $fatal || $syntax));
				
				(* Remove TestResultObjects to avoid memory leak *)
				With[{resObjNames = Complement[Names["MUnit`TestResultObjects`*"], keptResObjNames]}, Remove @@ resObjNames];
			];
			
			(* Test run is complete - log various information *)
			Scan[LogCPUTimeUsed[#, cpuTimeUsed]&, $loggers];
			Scan[LogAbsoluteTimeUsed[#, absTimeUsed]&, $loggers];
			Scan[LogMemoryUsed[#, memoryUsed]&, $loggers];
			Scan[LogTestCount[#, $actualTestIndex]&, $loggers];
			Scan[LogSuccessCount[#, successCnt]&, $loggers];
			Scan[LogFailureCount[#, failureCnt]&, $loggers];
			Scan[LogMessagesFailureCount[#, msgFailureCnt]&, $loggers];
			Scan[LogSkippedTestCount[#, skippedTestCnt]&, $loggers];
			Scan[LogErrorCount[#, errorCnt]&, $loggers];
			Scan[LogWasAborted[#, $abort]&, $loggers];
			Scan[LogWasFatal[#, $fatal]&, $loggers];
			Scan[LogWasSyntax[#, $syntax]&, $loggers];
			Scan[LogWasSuccessful[#, wasSuccessful]&, $loggers];
			Scan[LogEnd[#, $actualTestIndex, successCnt, failureCnt, msgFailureCnt, skippedTestCnt, errorCnt, $abort||$fatal||$syntax]&, $loggers];
			Scan[LogEnd[#]&, $loggers];
			wasSuccessful
			,
			MUnitErrorTag (* Catch MUnitErrorTags thrown during the test run *)
			,
			Function[{rules, tag},
				Module[{msgs = "Messages" /. rules, val = "Value" /. rules},
					(*
					TODO: fix the crazy 9's problem
					*)
					Scan[LogFatal[#, ToString[Column[{msgs[[1]], Style[msgs[[2]], Bold], val}]]]&, $loggers];
					Scan[LogEnd[#, 0, 0, 0, 0, 0, 0, True]&, $loggers];
					Scan[LogEnd[#]&, $loggers];
					False
				]
			]
		] (* End Catch *)
	] (* End Module *)



noteBookFileQ[n_String]:= If[StringQ[n],(StringMatchQ[n, "*.nb"]===True), False]


(* midTestRun is an intermediate step that takes the test source and prepares it for the 
low-level parsing in lowTestRun. *)
Options[midTestRun] = Options[TestRun]
SetAttributes[midTestRun, HoldFirst]

(* Avoid errors being thrown for these two incorrect cases *)
midTestRun[tests_/;MemberQ[{Test,VerificationTest},Part[Hold[tests],1,0]],OptionsPattern[]]:=
	Throw[{"Value" -> ToString[HoldForm[TestRun[tests]]], "Messages" -> {"TestReport", "TestReport could not handle the test source"}}, MUnitErrorTag];

(* given a list, TestRun interprets it as a list of tests *)
midTestRun[testList_/;Part[Hold[testList],1,0]===List, OptionsPattern[]] :=
	Module[{$loggers, title, ccondition, testStream, res, sameTest, mConstraint, tConstraint},
		{$loggers, title, ccondition, sameTest, mConstraint, tConstraint} = OptionValue[{Loggers, TestRunTitle, CanaryCondition, SameTest, MemoryConstraint, TimeConstraint}];
		(* check that all of the objects are VerificationTests *)
        Block[{VerificationTest,Test},
			If[Not@MatchQ[Union@Part[Hold[testList],1,All,0], {VerificationTest}|{Test}|{Test,VerificationTest}],
				Throw[{"Value" -> Unevaluated@testList, "Messages" -> {"TestReport", "TestReport only accepts a list of VerificationTest"}}, MUnitErrorTag]
			]];
		testSourceStack = Append[testSourceStack, "List of VerificationTest"];
		progressStack = Append[progressStack, 0];
		Scan[LogBeginTestSource[#, "Test Report"]&, $loggers];
		Scan[LogTestRunProgress[#, progressStack[[-1]]]&, $loggers];
		testStream = StringToStream[Block[{VerificationTest},ToString[Unevaluated[testList],InputForm]]];
		res = lowTestRun[testStream, Loggers -> $loggers, TestRunTitle -> title, CanaryCondition -> ccondition, SameTest -> sameTest, MemoryConstraint -> mConstraint, TimeConstraint -> tConstraint];
		Close[testStream];
		testSourceStack = Most[testSourceStack];
		progressStack = Most[progressStack];
		Scan[LogEndTestSource[#]&, $loggers];
		res
		
	]

(* As attribute is HoldFirst, evaluate the file name before proceeding further *)
midTestRun[testFileName:Except[_String|_List], OptionsPattern[]]/;Quiet@(StringQ[testFileName]||MatchQ[testFileName,File[_?StringQ]]) := 
	Module[{$loggers, title, ccondition, sameTest, mConstraint, tConstraint},
		{$loggers, title, ccondition, sameTest, mConstraint, tConstraint} = OptionValue[{Loggers, TestRunTitle, CanaryCondition, SameTest, MemoryConstraint, TimeConstraint}];
		With[{evaluatedName=(testFileName/.File[a_]:>a)}, 
			midTestRun[evaluatedName, Loggers -> $loggers, TestRunTitle -> title, CanaryCondition -> ccondition, SameTest -> sameTest, MemoryConstraint -> mConstraint, TimeConstraint -> tConstraint]
		]
	];

(* given a string, TestRun interprets it as a test file name *)
midTestRun[testFileName_String /; (StringQ[testFileName]&& !noteBookFileQ[testFileName]), OptionsPattern[]] :=
	Module[{$loggers, title, ccondition, testFileNameFound, testStream, res, sameTest, mConstraint, tConstraint},
		{$loggers, title, ccondition, sameTest, mConstraint, tConstraint} = OptionValue[{Loggers, TestRunTitle, CanaryCondition, SameTest, MemoryConstraint, TimeConstraint}];
		(* find and expand file name *)
		testFileNameFound = FindFile[testFileName];
		If[testFileNameFound === $Failed,
			Throw[{"Value" -> testFileName, "Messages" -> {"TestReport", "TestReport could not open the test file"}}, MUnitErrorTag]
		];
		testSourceStack = Append[testSourceStack, testFileNameFound];
		progressStack = Append[progressStack, 0];
		Scan[LogBeginTestSource[#, testFileNameFound]&, $loggers];
		Scan[LogTestRunProgress[#, progressStack[[-1]]]&, $loggers];
		testStream = OpenRead[testFileNameFound];
		(*
		manually passing the options works around a subtlety:
		Loggers uses a RuleDelayed, and we only want it to evaluate once
		if we simply passed in opts here, then Loggers will have evaluated once above,
		and then again inside testRun, resulting in 2 different loggers
		*)
		res = lowTestRun[testStream, Loggers -> $loggers, TestRunTitle -> title, CanaryCondition -> ccondition, SameTest -> sameTest, MemoryConstraint -> mConstraint, TimeConstraint -> tConstraint];
		Close[testStream];
		testSourceStack = Most[testSourceStack];
		progressStack = Most[progressStack];
		Scan[LogEndTestSource[#]&, $loggers];
		res
	]

(* given a NotebookObject, TestRun interprets as a test notebook *)
midTestRun[testNotebookObject_/;(Quiet[Head@testNotebookObject===NotebookObject]), OptionsPattern[]] :=
	Module[{$loggers, title, ccondition, notebookTests, testStream, res, sameTest, mConstraint, tConstraint},
		{$loggers, title, ccondition, sameTest, mConstraint, tConstraint} = OptionValue[{Loggers, TestRunTitle, CanaryCondition, SameTest, MemoryConstraint, TimeConstraint}];
		testSourceStack = Append[testSourceStack, testNotebookObject];
		progressStack = Append[progressStack, 0];
		Scan[LogBeginTestSource[#, testNotebookObject]&, $loggers];
		Scan[LogTestRunProgress[#, progressStack[[-1]]]&, $loggers];
		notebookTests = NotebookToTests[testNotebookObject, title];
		testStream = StringToStream[notebookTests];
		res = lowTestRun[testStream, Loggers -> $loggers, TestRunTitle -> title, CanaryCondition -> ccondition, SameTest -> sameTest, MemoryConstraint -> mConstraint, TimeConstraint -> tConstraint];
		Close[testStream];
		testSourceStack = Most[testSourceStack];
		progressStack = Most[progressStack];
		Scan[LogEndTestSource[#]&, $loggers];
		res
	]

midTestRun[testNotebookFileName_String?noteBookFileQ, OptionsPattern[]] :=
	Module[{$loggers, title, ccondition, testNotebookFileNameFound, testNotebook, notebookTests, testStream, res, sameTest, mConstraint, tConstraint},
		{$loggers, title, ccondition, sameTest, mConstraint, tConstraint} = OptionValue[{Loggers, TestRunTitle, CanaryCondition, SameTest, MemoryConstraint, TimeConstraint}];
		(* find and expand file name *)
		testNotebookFileNameFound = FindFile[testNotebookFileName];
		If[testNotebookFileNameFound === $Failed,
			Throw[{"Value" -> testNotebookFileName, "Messages" -> {"TestReport", "TestReport could not open the test file"}}, MUnitErrorTag]
		];
		testSourceStack = Append[testSourceStack, testNotebookFileNameFound];
		progressStack = Append[progressStack, 0];
		Scan[LogBeginTestSource[#, testNotebookFileNameFound]&, $loggers]; (* Unnecessary for VerificationTest *)
		Scan[LogTestRunProgress[#, progressStack[[-1]]]&, $loggers]; (* Unnecessary for VerificationTest *)
		testNotebook = Quiet[Get[testNotebookFileNameFound], {Get::noopen}];
		notebookTests = NotebookToTests[testNotebook, title];
		testStream = StringToStream[notebookTests];
		res = lowTestRun[testStream, Loggers -> $loggers, TestRunTitle -> title, CanaryCondition -> ccondition, SameTest -> sameTest, MemoryConstraint -> mConstraint, TimeConstraint -> tConstraint];
		Close[testStream];
		testSourceStack = Most[testSourceStack];
		progressStack = Most[progressStack];
		Scan[LogEndTestSource[#]&, $loggers];
		res
	]

(* Fallthrough case *)
midTestRun[args___] :=
	Throw[{"Value" -> ToString[HoldForm[TestRun[args]]], "Messages" -> {"TestReport", "TestReport could not handle the test source"}}, MUnitErrorTag]


(* lowTestRun works on an InputStream. It reads from the stream, one expression at a time. 
Note that lowTestRun heavily uses the concept of a "Section Stack": Every TestSection, TestRequirement, 
TestIgnore, etc is converted into a TestSection, possibly nested in various ways - this is the section stack.
If a test is inside a section with a requirement that is False, the test is skipped.
*)
Options[lowTestRun] = Options[TestRun]

lowTestRun[testStream_InputStream, OptionsPattern[]] :=
	Module[{$loggers, title, ccondition,
			$abort = False, $fatal = False, $syntax = False,
			$canary = 1,
			$sectionStack = {},
			curPos, oldPos, beginningPos, endingPos,
			expr,
			successCnt = 0,
			failureCnt = 0,
			msgFailureCnt = 0,
			skippedTestCnt = 0,
			errorCnt = 0, sameTest, mConstraint, tConstraint},
		{$loggers, title, ccondition, sameTest, mConstraint, tConstraint} = OptionValue[{Loggers, TestRunTitle, CanaryCondition, SameTest, MemoryConstraint, TimeConstraint}];
		Block[{TestLog, TestTerminate, MUnitGetFileHandler, TestSuite,
			$testIndex = 0, $dynamicTestIndex, $lexicalTestIndex, logTestResult, logInfo},
			
			(* ==== Begin by defining some utility functions ==== *)
			(* TestLog should accept anything *)
			TestLog[e_] :=
				Scan[LogMessage[#, ToString[e, OutputForm]]&, $loggers];			
			(* TestTerminate should accept anything *)
			TestTerminate[e_] :=
				(
					Scan[LogMessage[#, "Stopping test run on TestTerminate: " <> ToString[e, OutputForm]]&, $loggers];
					$abort = True
				);
				
			(* MUnitGetFileHandler is a GetFileEvent handler, added to the list of Internal`Handlers[] in MUnit.m *)
			MUnitGetFileHandler[HoldComplete[file_String, wrapper_, mode:First|Last]] :=
				Switch[mode,
					First,
					testSourceStack = Append[testSourceStack, file];
					progressStack = Append[progressStack, 0];
					Scan[LogBeginTestSource[#, file]&, $loggers];
					Scan[LogTestRunProgress[#, progressStack[[-1]]]&, $loggers];
					,
					Last,
					(* log this file at 100% before leaving, since there are no hooks to use for getting each expr out of the file *)
					progressStack[[-1]] = 1.0;
					Scan[LogTestRunProgress[#, progressStack[[-1]]]&, $loggers];
					
					testSourceStack = Most[testSourceStack];
					progressStack = Most[progressStack];
					Scan[LogEndTestSource[#]&, $loggers];
				];
				
			TestSuite[suite:{___String}] :=
				Module[{cpuTimeUsedSuite, absTimeUsedSuite, memoryUsedSuite, $testIndexSuite,
						successCntSuite, failureCntSuite, msgFailureCntSuite, skippedTestCntSuite,
						errorCntSuite, $abortSuite, $fatalSuite, $syntaxSuite,
						wasSuccessfulSuite},
					(If[StringQ[testSourceStack[[-1]]],
						SetDirectory[DirectoryName[testSourceStack[[-1]]]];
					];
					Catch[
						{cpuTimeUsedSuite, absTimeUsedSuite, memoryUsedSuite, $testIndexSuite,
						successCntSuite, failureCntSuite, msgFailureCntSuite, skippedTestCntSuite,
						errorCntSuite, $abortSuite, $fatalSuite, $syntaxSuite} = midTestRun[#, Loggers -> $loggers, TestRunTitle -> title, CanaryCondition -> ccondition, SameTest -> sameTest, MemoryConstraint -> mConstraint, TimeConstraint -> tConstraint];
						wasSuccessfulSuite = (successCntSuite == $testIndexSuite && errorCntSuite == 0 && !($abortSuite || $fatalSuite || $syntaxSuite));
						
						$dynamicTestIndex += $testIndexSuite;
						successCnt += successCntSuite;
						failureCnt += failureCntSuite;
						msgFailureCnt += msgFailureCntSuite;
						skippedTestCnt += skippedTestCntSuite;
						errorCnt += errorCntSuite;
						$abort = Or[$abort, $abortSuite];
						$fatal = Or[$fatal, $fatalSuite];
						$syntax = Or[$syntax, $syntaxSuite];
						
						,
						MUnitErrorTag
						,
						Function[{rules, tag},
							Module[{msgs = "Messages" /. rules, val = "Value" /. rules},
								(*
								TODO: fix the crazy 9's problem
								*)
								Scan[LogFatal[#, ToString[Column[{msgs[[1]], Style[msgs[[2]], Bold], val}]]]&, $loggers];
								$fatal = True;
							]
						]
					];
					If[StringQ[testSourceStack[[-1]]],
						ResetDirectory[];
					];)& /@ suite;
				];
			TestSuite[args___] := (
				Scan[LogFatal[#, ToString[StringForm[TestRun::suite, HoldForm[TestSuite[args]]]]]&, $loggers];
				$fatal = True;);
			
			logInfo[id_String] := (
				$lexicalTestIndex++;
				Scan[LogTestInfo[#, id, $lexicalTestIndex, And @@ $sectionStack]&, $loggers];
			);
			logInfo[tr_?TestResultQ] := logInfo[TestID[tr]];
			logInfo[id_] := logInfo[ToString[id]];
			
			logTestResult[tr_] :=
				Module[{fm = FailureMode[tr], action = ""},
					Switch[fm,
						"Success",
							successCnt++;
							Scan[LogSuccess[#, tr]&, $loggers]
						,
						"Failure",
							failureCnt++;
							Scan[LogFailure[#, tr]&, $loggers];
							action = TestFailureAction[tr]
						,
						"MessagesFailure",
							msgFailureCnt++;
							Scan[LogMessagesFailure[#, tr]&, $loggers];
							action = TestFailureAction[tr]
						,
						"Error",
							errorCnt++;
							Scan[LogError[#, tr]&, $loggers];
							action = TestErrorAction[tr]
					];
					If[action != "",
						Which[
							action == "Abort",
							$abort = True;
							Scan[LogMessage[#, "Aborting test run on TestFailureAction or TestErrorAction"]&, $loggers];
							Throw["Abort", "internalMUnitTestRunTag"]
							,
							action == "SkipSection" || ccondition && $testIndex == $canary,
							$sectionStack[[-1]] = False;
							Scan[LogMessage[#, "Skipping test section on TestFailureAction, TestErrorAction, or canary condition"]&, $loggers];
							Throw["SkipSection", "internalMUnitTestRunTag"]
						]
					]
				];
			(* ==== End of utility functions ==== *)
			
			beginningPos = StreamPosition[testStream];
			endingPos = SetStreamPosition[testStream, Infinity];
			SetStreamPosition[testStream, beginningPos];
			curPos = beginningPos;
			
			(* preprocess loop - go through the full stream and process it to check for errors *)
			Module[{newJunkNames = Names[]},
				Begin["MUnit`TestRun`Junk`"];
				While[True,
					Block[{$MessageNiceStringList = {}, MUnitMessageHandler = messageNiceStringHandler},
						Check[
							oldPos = curPos;
							expr = Read[testStream, HoldComplete[Expression]]; (* Read in expression *)
							curPos = StreamPosition[testStream];
							,
							Module[{len, badString},
								len = curPos - oldPos;
								SetStreamPosition[testStream, oldPos];
								badString = StringJoin[ReadList[testStream, Character, len]];
								
								Scan[LogFatal[#, "Syntax error reading " <> badString]&, $loggers];
								
								Function[{niceMsg}, Scan[LogFatal[#, niceMsg]&, $loggers]] /@ $MessageNiceStringList; 
								
								$syntax = True;
								Break[]
							]
						];
					];
					
					If[expr === EndOfFile, Break[]];
					
					(*
					preprocess
					1. handle errors
					*)
					expr = Replace[expr, HoldComplete[CompoundExpression[e:(_BeginTestSection|_TestRequirement|_EndTestSection|_EndRequirement|_TestIgnore|_EndIgnore), Null]] :> HoldComplete[e]];
					expr = Replace[expr, {
						(* valids *)
						HoldComplete[BeginTestSection[sec_, require_]] :>
							Null,
						HoldComplete[EndTestSection[]] :>
							Null,
						(* processes *)
						HoldComplete[BeginTestSection[sec_]] :>
							Null,
						HoldComplete[TestRequirement[require_, ___]] :>
							Null,
						HoldComplete[TestIgnore[ignore_, ___]] :>
							Null,
						HoldComplete[(EndRequirement|EndIgnore)[]] :>
							Null,
						(* errors *)
						HoldComplete[BeginTestSection[___]] :>
							(Scan[LogFatal[#, ToString[StringForm[TestRun::badsec, expr]]]&, $loggers];$fatal = True;Break[]),
						HoldComplete[TestRequirement[___]] :>
							(Scan[LogFatal[#, ToString[StringForm[TestRun::badsec, expr]]]&, $loggers];$fatal = True;Break[]),
						HoldComplete[TestIgnore[___]] :>
							(Scan[LogFatal[#, ToString[StringForm[TestRun::badsec, expr]]]&, $loggers];$fatal = True;Break[]),
						HoldComplete[EndTestSection[___]] :>
							(Scan[LogFatal[#, ToString[StringForm[TestRun::badsec, expr]]]&, $loggers];$fatal = True;Break[]),
						HoldComplete[EndRequirement[___]] :>
							(Scan[LogFatal[#, ToString[StringForm[TestRun::badsec, expr]]]&, $loggers];$fatal = True;Break[]),
						HoldComplete[EndIgnore[___]] :>
							(Scan[LogFatal[#, ToString[StringForm[TestRun::badsec, expr]]]&, $loggers];$fatal = True;Break[])}];
							
				]; (* end While *)
				End[];
				Quiet[Remove["MUnit`TestRun`Junk`*"], {Remove::rmnsm}];
				newJunkNames = Complement[Names[], newJunkNames];
				Quiet[Remove @@ newJunkNames, {Remove::rmnsm}];
			];
			
			(* return to the beginning of the stream *)
			SetStreamPosition[testStream, beginningPos];
			
			curPos = beginningPos; 
			
			Module[{cpuTimeUsed, absTimeUsed, memoryUsed},
				cpuTimeUsed = TimeUsed[];
				absTimeUsed = SessionTime[];
				memoryUsed = MemoryInUse[];
				
				If[endingPos===beginningPos, (* Handle case of empty test files *)
					Scan[LogFatal[#, ToString[StringForm[TestRun::empty,FileNameTake@First@testStream]]]&, $loggers];$fatal = True;
					,
					progressStack[[-1]] = (curPos - beginningPos)/(endingPos - beginningPos);
					Scan[LogTestRunProgress[#, progressStack[[-1]]]&, $loggers];
				];
				
				
				(* main loop - Evaluation happens here *)
				While[True,
					
					If[$abort || $fatal || $syntax, Break[]];
					
					oldPos = curPos;
					expr = Read[testStream, HoldComplete[Expression]]; (* Read in held expression *)
					curPos = StreamPosition[testStream];
					
					progressStack[[-1]] = (curPos - beginningPos)/(endingPos - beginningPos);
					Scan[LogTestRunProgress[#, progressStack[[-1]]]&, $loggers];
					
					If[expr === EndOfFile, Break[]];
					
					(*
					preprocess
					1. replace foo; with just foo
					2. add 2nd arg to BeginTestSection, normalize to BeginTestSection and EndTestSection
					*)
					expr = Replace[expr, HoldComplete[CompoundExpression[e:(_BeginTestSection|_TestRequirement|_EndTestSection|_EndRequirement|_TestIgnore|_EndIgnore), Null]] :> HoldComplete[e]];
					expr = Replace[expr, {
						(* valids *)
						HoldComplete[BeginTestSection[sec_, require_]] :>
							HoldComplete[BeginTestSection[sec, require]],
						HoldComplete[EndTestSection[]] :>
							HoldComplete[EndTestSection[]],
						(* processes *)
						HoldComplete[BeginTestSection[sec_]] :>
							HoldComplete[BeginTestSection[sec, True]],
						HoldComplete[TestRequirement[require_, ___]] :>
							HoldComplete[BeginTestSection["", require]],
						HoldComplete[TestIgnore[ignore_, ___]] :>
							HoldComplete[BeginTestSection["", !ignore]],
						HoldComplete[(EndRequirement|EndIgnore)[]] :>
							HoldComplete[EndTestSection[]]}];
					
					Switch[expr,
						HoldComplete[_BeginTestSection], (* Handle beginning of a TestSection *)
						Module[{section, require},
							{section, require} = Extract[expr[[1]], {{1}, {2}}];
							If[And @@ $sectionStack && require,
								$canary = $testIndex + 1
								,
								Null
								,
								Scan[LogFatal[#, ToString[StringForm[BeginTestSection::require, require]]]&, $loggers];
								$fatal = True;
							];
							(* unlike the LogMessages, which are not touched in a completely skipped section, we want
							to log EVERY section with LogBeginTestSection and LogEndTestSection *)
							$sectionStack = Append[$sectionStack, require];
							Scan[LogBeginTestSection[#, section, require]&, $loggers]
						]
						,
						
						HoldComplete[_EndTestSection], (* Handle end of a TestSection *)
						Scan[LogEndTestSection, $loggers];
						If[Length[$sectionStack] <= 0,
							Scan[LogFatal[#, ToString[StringForm[TestRun::nostack]]]&, $loggers];
							$fatal = True;
							,
							$sectionStack = Most[$sectionStack]
						]
						,
						
						_, (* General case - inside a TestSection *)
						$lexicalTestIndex = $testIndex;
						$dynamicTestIndex = $testIndex;
						Block[{logSkipped, DefectiveTest},
							
							(* Utility function for logging TestIDs of skipped tests *)
							logSkipped[e_] := 
							Block[{ids = Cases[e, (Rule|RuleDelayed)[TestID, id_] :> id, Infinity], $sectionStack = {False}},
								Function[testid,
									++skippedTestCnt;
									logInfo[testid]
								] /@ ids
							];
							
							(* special handling for DefectiveTest. Treat it like a miniature test section *)
							Attributes[DefectiveTest] = {HoldAllComplete};
							DefectiveTest[args__] := logSkipped[HoldComplete[{args}]];

							If[And @@ $sectionStack,
								(* The actual evaluation of the input and comparing of result to expected is done here. *)
								Internal`InheritedBlock[{VerificationTest},
									SetOptions[VerificationTest, SameTest -> sameTest, MemoryConstraint -> mConstraint, TimeConstraint -> tConstraint];
									Catch[
										ReleaseHold[expr];
										,
										"internalMUnitTestRunTag"
									]
								]
								,
								logSkipped[expr] (* Skipped section - log skipped tests*)
							]
						];
						$testIndex = Max[$lexicalTestIndex, $dynamicTestIndex];
					]; (* end Switch *)
					
				]; (* end While *)
				
				cpuTimeUsed = TimeUsed[] - cpuTimeUsed;
				absTimeUsed = SessionTime[] - absTimeUsed;
				memoryUsed = MemoryInUse[] - memoryUsed;
				
				{cpuTimeUsed, absTimeUsed, memoryUsed, $testIndex, successCnt, failureCnt, msgFailureCnt, skippedTestCnt, errorCnt, $abort, $fatal, $syntax}
			]
		] (* end Block *)
	] (* end Module *)
	
(* Fallthrough case *)
TestRun[args___, opts:OptionsPattern[]] :=
	Module[{$loggers, title},
		{$loggers, title} = OptionValue[{Loggers, TestRunTitle}];
		Scan[LogBegin[#, title]&, $loggers];
		Scan[LogFatal[#, ToString[StringForm[TestRun::strnbobj, 1, HoldForm[TestRun[args, opts]]]]]&, $loggers];
		Scan[LogEnd[#, 0, 0, 0, 0, 0, 0, True]&, $loggers];
		Scan[LogEnd[#]&, $loggers];
		False
	]
	

End[]
