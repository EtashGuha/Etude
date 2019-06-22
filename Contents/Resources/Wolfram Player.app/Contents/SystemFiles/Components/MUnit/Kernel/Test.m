(* ::Package:: *)

Test::usage =
"Test[input] tests whether input evaluates to True and no messages are generated.\n
Test[input, expected] tests whether input evaluates to expected and no messages are generated.\n
Test[input, expected, messages] tests whether input evaluates to expected and the given messages are generated.\n
Test is also a selector for TestResult."

TestMatch::usage =
"TestMatch[input, expected] and TestMatch[input, expected, messages] are like the corresponding versions of Test except they test MatchQ[input, expected] instead of SameQ."

TestStringMatch::usage =
"TestStringMatch[input, expected] and TestStringMatch[input, expected, messages] are like the corresponding versions of Test except they test StringMatchQ[input, expected] instead of SameQ."

TestFree::usage =
"TestFree[input, expected] and TestFree[input, expected, messages] are like the corresponding versions of Test except they test FreeQ[input, expected] instead of SameQ."

TestStringFree::usage =
"TestStringFree[input, expected] and TestStringFree[input, expected, messages] are like the corresponding versions of Test except they test StringFreeQ[input, expected] instead of SameQ."

TestResult::usage =
"TestResult is the head of the expression returned by Test and related functions. It is passed to loggers.
TestResult contains all the relevant information about a given Test.
Clients should always use selectors to extract the desired element from a TestResult.
For example, if tr is a TestResult, write Test[tr] instead of tr[[1]] to get the original Test expr out.
The internal structure of TestResult is not guaranteed to stay the same from one version of MUnit to the next."

TestResultQ::usage =
"TestResultQ[tr] returns True if tr is a valid TestResult, and False otherwise."

FailureMode::usage =
"FailureMode is a selector for TestResult."

TestInput::usage =
"TestInput is a selector for TestResult."

ExpectedOutput::usage =
"ExpectedOutput is a selector for TestResult."

ActualOutput::usage =
"ActualOutput is a selector for TestResult."

ExpectedMessages::usage =
"ExpectedMessages is a selector for TestResult."

ActualMessages::usage =
"ActualMessages is a selector for TestResult."

ErrorMessage::usage =
"ErrorMessage is a selector for TestResult."

TestIndex::usage =
"TestIndex is a selector for TestResult. It is used in test runs."

TestTags::usage =
"TestTags is an option to Test that specified meta information about that test. If a test comes from a notebook, its
cell tags are kept as TestTags."

EquivalenceFunction::usage =
"EquivalenceFunction is an option to Test and TestMatch that specifies the function that will be \
used to compare the actual and expected results.\n
The test input and expected output will have evaluated once before the equivalence function is evaluated.
The default for Test is SameQ, and for TestMatch it is MatchQ.
EquivalenceFunction is also a selector for TestResult."

MessagesEquivalenceFunction::usage =
"MessagesEquivalenceFunction is an option to Test and TestMatch that specifies the function that will be used to compare the actual and expected messages\n.
Custom MessagesEquivalenceFunction settings should also take into account that actual messages are passed in wrapped in HoldForm, \
and that a custom ExpectedMessagesWrapper will probably also be necessary to prevent any unwanted evaluations or messages."

$DefaultMessagesEquivalenceFunction::usage =
"$DefaultMessagesEquivalenceFunction is the default messages equivalence function. \
It is similar to MatchQ and understands Message and MessageName expressions."

TestWrapper::usage =
"TestWrapper is an option to Test."

TestInputWrapper::usage =
"TestInputWrapper is an option to Test"

ActualOutputSetFunction::usage =
"ActualOutputSetFunction is an option to Test that specifies the function that will be used to evaluate the input \
and set it to a variable.\n
It is a function that takes 2 arguments, the first argument is a symbol available for assigning the value of the second argument, the actual output.
ActualOutputSetFunction is also a selector for TestResult"

ExpectedOutputSetFunction::usage =
"ExpectedOutputSetFunction is an option to Test that specifies the function that will be used to evaluate the expected output \
and set it to a variable.\n
It is a function that takes 2 arguments, the first argument is a symbol available for assigning the value of the second argument, the expected output.
ExpectedOutputSetFunction is also a selector for TestResult"

ActualOutputWrapper::usage =
"ActualOutputWrapper is an option to Test that specifies how to wrap the actual output.
It is a function that takes one argument, and it is given a symbol that evaluates to the actual output.
You may choose to have the symbol not evaluate, to prevent side-effects."

$DefaultActualOutputWrapper::usage =
"$DefaultActualOutputWrapper is the default ActualOutputWrapper. It is similar to HoldForm."

ExpectedOutputWrapper::usage =
"ExpectedOutputWrapper is an option to Test that specifies how to wrap the expected output.
It is a function that takes one argument, and it is given a symbol that evaluates to the expected output.
You may choose to have the symbol not evaluate, to prevent side-effects."

$DefaultExpectedOutputWrapper::usage =
"$DefaultExpectedOutputWrapper is the default ExpectedOutputWrapper. It is similar to HoldForm."

ExpectedMessagesWrapper::usage =
"ExpectedMessagesWrapper is an option to Test that specifies the function that will be used to wrap expected messages in a TestResult expression. \
The default value is $DefaultExpectedMessagesWrapper, but it should be changed to an appropriate value for preserving custom expressions passed as expected messages."

$DefaultExpectedMessagesWrapper::usage =
"$DefaultExpectedMessagesWrapper is the default ExpectedMessagesWrapper. It understands MessageName vs. Message, lists, Alternatives, etc."

TestFailureMessage::usage =
"TestFailureMessage is an option to Test that specifies a message that will be logged if the test fails.\n
TestFailureMessage is also a selector for TestResult."

TestFailureAction::usage =
"TestFailureAction is an option to Test that specifies a string naming an action to take if the test fails.\n
The allowable values are Continue (the default) and Abort.\n
TestFailureAction is also a selector for TestResult."

TestErrorAction::usage =
"TestErrorAction is an option to Test that specifies a string naming an action to take if the test has an error.\n
The allowable values are Continue (the default) and Abort.\n
TestErrorAction is also a selector for TestResult."

$MemoryConstrained::usage =
"$MemoryConstrained is returned as actual output when a test met a memory constraint."

$TimeConstrained::usage =
"$TimeConstrained is returned as actual output when a test met a time constraint."

TestCPUTimeUsed::usage =
"TestCPUTimeUsed is a selector for test results."

TestAbsoluteTimeUsed::usage =
"TestAbsoluteTimeUsed is a selector for test results."

TestMemoryUsed::usage =
"TestMemoryUsed is a selector for test results."

$Error::usage =
"$Error is filled in for the arguments of a test result when an error occurs."

TestSource::usage = 
"TestSource is an option of TestResult that specifies the source of the test: a file, notebook, or stream."

TestClass::usage = "TestClass is a property of TestResult that specifies the class of the test.";

NTestFailureMessage::usage =
"NTestFailureMessage is a selector for NTest results"

OrNTestFailureMessages::usage =
"OrNTestFailureMessages is a selector for OrNTest results"

TestComment::usage =
"TestComment is deprecated."

TestComments::usage =
"TestComments is deprecated."

AllTestIndex::usage =
"AllTestIndex is deprecated."

Begin["`Package`"]

AddTestFunction

testError

End[]

Begin["`Test`Private`"]

Attributes[Test] = {HoldAllComplete}

Options[Test] = {
		ActualOutputSetFunction -> Set,
		ExpectedOutputSetFunction -> Set,
		SameTest -> SameQ,
		EquivalenceFunction -> SameQ,
		MemoryConstraint -> Infinity,
		TestWrapper -> HoldForm,
		TestInputWrapper -> HoldForm,
		ExpectedOutputWrapper -> $DefaultExpectedOutputWrapper,
		ActualOutputWrapper -> $DefaultActualOutputWrapper,
		MessagesEquivalenceFunction -> $DefaultMessagesEquivalenceFunction,
		ExpectedMessagesWrapper -> $DefaultExpectedMessagesWrapper,
		TestFailureMessage -> "",
		TestFailureAction -> "Continue",
		TestErrorAction -> "Continue",
		TestID -> None,
		TestTags -> {},
		TimeConstraint -> Infinity,
		TestComment -> "",
		TestComments -> "",
		NTestFailureMessage -> "",
		OrNTestFailureMessages -> {},
		"TestClass" -> None
	}

(*
Precondition: $Messages is set to the system default of something like {stdout}
*)

Test[input_, expected_:True, Shortest[expectedMsgs_:{}], opts:OptionsPattern[]] :=
	Module[{testWrapper, testInputWrapper, actualOutputSetFunction, expectedOutputSetFunction, sameTest, equivalenceFunction, msgsEquivFunc,
		expectedOutputWrapper, actualOutputWrapper, expectedMsgsWrapper, testIDHeld, actual,
		expectedEvaled, testEquivFuncEvaled, msgsEquivFuncEvaled, actualMsgs, testEquivFuncMsgStrings, msgsEquivFuncMsgStrings,
		failureMode, testFailureMessageHeld, testFailureAction, testErrorAction,
		timeConstraint, memoryConstraint, cpuTimeUsed, absTimeUsed, memoryUsed, testTags, actualRes, expectedRes,
		testEquivFuncRes, msgsEquivFuncRes, evalOccurred, testclass,
		(*WRI-specific*)
		ntestFailureMessage, orntestFailureMessages},
		Catch[
			Block[{
				(* these are to setup a nice environment for tests within tests! *)
				$actualTestIndex = 0,
				$allTestIndex = 0}
				,
				
				Block[{$Messages = {$bitbucket}, $MessageList = {}, MUnitMessageHandler = messageStringHandler, $MessageStringList = {}},
					(* treat any unrecognized options as an error, any invalid options will throw a message, even if they are not in the above list *)
					{testWrapper, testInputWrapper, actualOutputSetFunction, expectedOutputSetFunction, sameTest, equivalenceFunction, msgsEquivFunc,
						expectedOutputWrapper, actualOutputWrapper, expectedMsgsWrapper,
						testFailureAction, testErrorAction, timeConstraint, memoryConstraint, testTags, ntestFailureMessage, orntestFailureMessages, testclass} =
						OptionValue[Test, Automatic, {TestWrapper, TestInputWrapper, ActualOutputSetFunction, ExpectedOutputSetFunction, SameTest, EquivalenceFunction,
							MessagesEquivalenceFunction, ExpectedOutputWrapper, ActualOutputWrapper, ExpectedMessagesWrapper, TestFailureAction, TestErrorAction,
							TimeConstraint, MemoryConstraint, TestTags, NTestFailureMessage, OrNTestFailureMessages, "TestClass"}];					
					(*
					options that may evaluate. This section should grow as needed when examples of evaluating options are seen.
					This section is a favor to users. Tests should ideally be designed so that options can evaluate without having had
					to run the test first, e.g. Test[a=1;a+1, 2, {}, TestID->"test"<>IntegerString[a]] is an example of a bad test
					But if we can do a little bit to help users, then we should.
					This still requires users to use RuleDelayed instead of Rule when giving an option that will depend on the test itself.
					*)
					{testFailureMessageHeld, testIDHeld} =
						OptionValue[Test, Automatic, {TestFailureMessage, TestID}, Hold];
					If[$MessageStringList != {},
						Throw["Messages while evaluating options: " <> ToString[$MessageStringList], "internalMUnitTestTag"]
					];
				];
				
				(* log test info before it can hang/crash *)
				logInfo[ReleaseHold[testIDHeld]];
				
				memoryUsed = MemoryInUse[];
				cpuTimeUsed = TimeUsed[];
				absTimeUsed = SessionTime[];
				
				(* Evaluate "input" and assign to "actual" *)
				Block[{$Messages = {$bitbucket}, $MessageList = {}, MUnitMessageHandler = $MUnitMessageHandler, $MessageWithArgsList = {}},
					With[{actualOutputSetFunction = actualOutputSetFunction},
						actualRes = MUnitCheckAll[
							actualOutputSetFunction[
								actual
								,
								input
							]
							,
							opts
							,
							True
						];
						If[MatchQ[actualRes, Failure["TimeConstrained",_] | Failure["MemoryConstrained",_]],
							actual = actualRes
						];
						actualMsgs = $MessageWithArgsList
					]
				];
				
				cpuTimeUsed = TimeUsed[] - cpuTimeUsed;
				memoryUsed = MemoryInUse[] - memoryUsed;
				absTimeUsed = SessionTime[] - absTimeUsed;
				
				(* Evaluate "expected" and set to "expectedEvaled" *)
				Block[{$Messages = {$bitbucket}, $MessageList = {}, MUnitMessageHandler = $MUnitMessageHandler, $MessageWithArgsList = {}},
					With[{expectedOutputSetFunction = expectedOutputSetFunction},
						expectedRes = MUnitCheckAll[
							expectedOutputSetFunction[
								expectedEvaled
								,
								expected
							]
							,
							opts
							,
							True
						];
						If[MatchQ[expectedRes, Failure["TimeConstrained",_] | Failure["MemoryConstrained",_]],
							expectedEvaled = expectedRes
						]
					]
				];
				
				(* Compare actual and expectedEvaled using the specified SameTest *)
				Block[{$Messages = {$bitbucket}, $MessageList = {}, MUnitMessageHandler = messageStringHandler, $MessageStringList = {}},
					testEquivFuncRes = MUnitCheckAll[
						Reap[
							Set[
								testEquivFuncEvaled
								,
								If[sameTest===SameQ, sameTest = equivalenceFunction];
								ReleaseHold[Hold[sameTest[actual, expectedEvaled]] /. OwnValues[actual] ~Join~ OwnValues[expectedEvaled]]
							]
							,
							NTestError
							,
							Function[{tag, vals},
								orntestFailureMessages = Switch[#[[2]],
									"ULPS",
									"Error (ULPS):   " <> ToString[#[[1]], InputForm]
									,
									"CompareToPAT",
									StringJoin[Riffle[MUnit`WRI`Private`InformativeString /@ #[[1]], "\n"]]
								]& /@ vals;
								ntestFailureMessage = orntestFailureMessages[[1]];
							]
						]
						,
						opts
						,
						False
					];
					(* Guards against Sequence and other weirdness *)
					If[!(booleanQ[testEquivFuncEvaled]===True),testEquivFuncEvaled="Invalid"]; 
					testEquivFuncMsgStrings = $MessageStringList;
				];
				
				(* Compare actual and expected messages *)
				Block[{$Messages = {$bitbucket}, $MessageList = {}, MUnitMessageHandler = messageStringHandler, $MessageStringList = {}},
					(*
					TODO:
					if using $DefaultExpectedMessageHandler and given MessageNames as expected messages,
					then make sure to return MessagesNames as the actual messages
					*)
					msgsEquivFuncRes = MUnitCheckAll[
						Set[
							msgsEquivFuncEvaled
							,
							ReleaseHold[Hold[msgsEquivFunc[actualMsgs, expectedMsgs]] /. OwnValues[actualMsgs]]
						]
						,
						opts
						,
						False
					];
					msgsEquivFuncMsgStrings = $MessageStringList;
				];
				evalOccurred = ValueQ[actual]; (* Determine if evaluation of the test input actually happened. 
				                                  If it hasn't for some reason, bail out. *)
				If[!evalOccurred, 
					failureMode = "Error"; Throw["Unexpected error occurred", "internalMUnitTestTag"],
					failureMode = determineFailureMode[
									{testEquivFuncEvaled, msgsEquivFuncEvaled},
									{testEquivFuncRes, msgsEquivFuncRes},
									{testEquivFuncMsgStrings, msgsEquivFuncMsgStrings}];
				]
			]; (* end Block *)
			
			
			(* Create a test result object to return *)
			Module[{tr},
				With[{testWrapper = testWrapper, testInputWrapper = testInputWrapper},
					tr = Sow[
						newTestResult[
							testWrapper[Test[input, expected, expectedMsgs, opts]],
							failureMode,
							testInputWrapper[input],
							expectedOutputWrapper[expectedEvaled],
							actualOutputWrapper[actual],
							expectedMsgsWrapper[expectedMsgs],
							(* each msg is already wrapped in HoldForm *)
							actualMsgs,
							"",
							TestIndex -> ++$dynamicTestIndex,
							TestID -> If[ReleaseHold[testIDHeld]===Automatic, CreateUUID[], ReleaseHold[testIDHeld]],
							ActualOutputSetFunction -> actualOutputSetFunction,
							ExpectedOutputSetFunction -> expectedOutputSetFunction,
							ActualOutputWrapper -> actualOutputWrapper,
							ExpectedOutputWrapper -> expectedOutputWrapper,
							SameTest -> sameTest,
							MessagesEquivalenceFunction -> msgsEquivFunc,
							ExpectedMessagesWrapper -> expectedMsgsWrapper,
							TestFailureMessage -> ReleaseHold[testFailureMessageHeld],
							TestFailureAction -> testFailureAction,
							TestErrorAction -> testErrorAction,
							TestCPUTimeUsed -> cpuTimeUsed,
							TestAbsoluteTimeUsed -> absTimeUsed,
							TestMemoryUsed -> memoryUsed,
							TestTags -> testTags,
							TestSource -> $CurrentTestSource,
							NTestFailureMessage -> ntestFailureMessage,
							OrNTestFailureMessages -> orntestFailureMessages,
							"TestClass" -> testclass
						],
						{"MUnitTest"}
					];
				];
				logTestResult[tr];
				tr
			]
			,
			"internalMUnitTestTag"
			,
			Function[{value, tag},
				(*
				calling testError must wait until outside of the Block, so that $actualTestIndex can increment correctly
				*)
				If[TrueQ[evalOccurred],
					testError[value, 
							 {testWrapper[Test[input, expected, expectedMsgs, opts]],
							  testInputWrapper[input],
							  expectedOutputWrapper[expectedEvaled],
							  actualOutputWrapper[actual], 
							  expectedMsgsWrapper[expectedMsgs],
							  actualMsgs}, 
							opts]
					,
					testError[value, 
							 {testWrapper[Test[input, expected, expectedMsgs, opts]],
							  testInputWrapper[input],
							  "Error",
							  "Error", 
							  expectedMsgsWrapper[expectedMsgs],
							  "Error"}, 
							opts]
					]
			]
		]
	]

(* TODO : Replace this with System`Private`ArgumentsWithRules *)
Test[args___] :=
	With[{msg = "Incorrect arguments: " <> ToString[Unevaluated[{args}]]},
		testError[msg, {}, args]
	]


booleanQ[False|True] := True;
booleanQ[_] := False;


Attributes[testError] = {HoldAllComplete}

(* Utility function to create a TestResult object for an Error *)
testError[errMsg_String, list_List, ___, opts:OptionsPattern[]] :=
	Module[{testIDHeld, testErrorAction, tr, testWrapper},
		Quiet[
			{testErrorAction} = OptionValue[Test, {opts}, {TestErrorAction}];
			{testIDHeld} = OptionValue[Test, {opts}, {TestID}, Hold];
			{testWrapper} = OptionValue[Test,{opts},{TestWrapper},Hold];
		];
		If[list==={},
			tr = Sow[
				newTestResult[
					"Error",
					"Error",
					"Error",
					"Error",
					"Error",
					"Error",
					"Error",
					errMsg,
					TestIndex -> ++$dynamicTestIndex,
					TestID -> ReleaseHold[testIDHeld],
					TestSource -> $CurrentTestSource
				]
				,
				{"MUnitTest"}
			];
			,
			tr = Sow[
				newTestResult[
					Extract[list,{1}],
					"Error",
					Extract[list,{2}],
					Extract[list,{3}],
					Extract[list,{4}],
					Extract[list,{5}],
					Extract[list,{6}],
					errMsg,
					TestIndex -> ++$dynamicTestIndex,
					TestID -> ReleaseHold[testIDHeld],
					TestSource -> $CurrentTestSource
				]
				,
				{"MUnitTest"}
			]
		];
		logTestResult[tr];
		tr
	]

(*
MUnitCheckAll is a version of CheckAll customized for use in MUnit`Test.
All evaluation of input and expected output happens inside MUnitCheckAll, which properly
enforces time and memory constraints and handles unexpected aborts.
Note: MUnitCheckAll returns Null.
*)
SetAttributes[MUnitCheckAll, HoldAllComplete]
MUnitCheckAll[expr_, opts:OptionsPattern[],testexpr_:True] :=
	Module[{timeConstraint, memoryConstraint},
		{timeConstraint, memoryConstraint} = OptionValue[Test, {opts}, {TimeConstraint, MemoryConstraint}];
		CheckAll[
			(* use Reap here to not double-count any Tests within Tests in a test run *)
			Reap[
				(* the With here is to allow MemoryConstrained and TimeConstrained look better when they return unevaluated *)
				With[{timeConstraint = timeConstraint, memoryConstraint = memoryConstraint, isTestExpr=testexpr},
					Which[
						(isTestExpr === True) && And@@(MemberQ[{Infinity,None},#]&/@{timeConstraint,memoryConstraint}),
						StackBegin[expr]
						(*, This needs to be fine tuned later
						(isTestExpr === True) && timeConstraint <= 0,
						Message[TimeConstrained::timc,timeConstraint];
						StackBegin[expr]
						,
						(isTestExpr === True) && memoryConstraint <= 0,
						Message[MemoryConstrained::ipnfm,HoldForm@MemoryConstrained[expr,memoryConstraint],2];
						StackBegin[expr]*)
						,
						(isTestExpr === True) && (MemberQ[{Infinity,None},timeConstraint]),
						MemoryConstrained[StackBegin[expr], memoryConstraint, Failure["MemoryConstrained",  <|"MessageTemplate" -> StringTemplate["Current evaluation has exceeded the memory constraint: `memoryConstraint`"],"MessageParameters"-> <|"memoryConstraint" -> memoryConstraint|>|>]]
						,
						(isTestExpr === True) && (MemberQ[{Infinity,None},memoryConstraint]),
						TimeConstrained[StackBegin[expr], timeConstraint, Failure["TimeConstrained",  <|"MessageTemplate" -> StringTemplate["Current evaluation has exceeded the time constraint: `timeConstraint`"],"MessageParameters"-> <|"timeConstraint" -> timeConstraint|>|>]]
						,
						isTestExpr === True,
						TimeConstrained[MemoryConstrained[StackBegin[expr], memoryConstraint, Failure["MemoryConstrained",  <|"MessageTemplate" -> StringTemplate["Current evaluation has exceeded the memory constraint: `memoryConstraint`"],"MessageParameters"-> <|"memoryConstraint" -> memoryConstraint|>|>]], timeConstraint, Failure["TimeConstrained",  <|"MessageTemplate" -> StringTemplate["Current evaluation has exceeded the time constraint: `timeConstraint`"],"MessageParameters"-> <|"timeConstraint" -> timeConstraint|>|>]]
						,
						True,
						StackBegin[expr]
					]
				]
				,
				"MUnitTest"
			][[1]]
			,
			Replace[#2, {
				Hold[] :>
				(* everything is fine *)
				#1
				,
				Hold[_Abort] :>
				(* an Abort was encountered, in test or user action *)
				(TestTerminate["Aborted during evaluation"];
				Throw["Aborted during evaluation", "internalMUnitTestTag"])
				,
				Hold[SystemException[type_String, trace_List]] :>
				Throw["Unhandled " <> type <> ": " <> ToString[Last[trace]], "internalMUnitTestTag"]
				,
				Hold[SystemException[type_String, trace_]] :>
				Throw["Unhandled" <> type <> ": " <> ToString[HoldForm[trace]], "internalMUnitTestTag"]
				,
				Hold[t_] :>
				Throw["Unhandled error: " <> ToString[HoldForm[t]], "internalMUnitTestTag"]
			}]&
		]
	]

(*
There is a bit of machinery here to determine exactly what
kind of error message to throw and to throw it nicely

The Block[{$Context = "MUnit`Test`Private`"}, ...]'s are to make the string nice, without giving
symbols the full MUnit`Test`Private` name
*)

determineFailureMode[
		{testEquivFuncEvaled_, msgsEquivFuncEvaled_},
		{testEquivFuncRes_, msgsEquivFuncRes_},
		{testEquivFuncMsgStrings_, msgsEquivFuncMsgStrings_}] :=
	Module[{testEquivFuncResString, msgsEquivFuncResString},
		Switch[{testEquivFuncEvaled, msgsEquivFuncEvaled},
			{True, True},
			"Success"
			,
			{False, True | False},
			"Failure"
			,
			{True, False},
			"MessagesFailure"
			,
			{Except[True | False], _},
			Block[{$Context = "MUnit`Test`Private`"},
				testEquivFuncResString = ToString[testEquivFuncRes]
			];
			Throw["SameTest did not return True or False: \n\n" <> ((# <> "\n\n")& /@ testEquivFuncMsgStrings) <> testEquivFuncResString, "internalMUnitTestTag"]
			,
			{_, Except[True | False]},
			Block[{$Context = "MUnit`Test`Private`"},
				msgsEquivFuncResString = ToString[msgsEquivFuncRes]
			];
			Throw["MessagesEquivalenceFunction did not return True or False: \n\n" <> ((# <> "\n\n")& /@ msgsEquivFuncMsgStrings) <> msgsEquivFuncResString, "internalMUnitTestTag"]
		]
	]

(*
patternizeExpectedMessage is a utility function for $DefaultMessagesEquivalenceFunction that converts the
expectedMessages input to a standardized format which can be used for easier pattern matching.
Needs to be HoldFirst because we do things with the arguments, where they aren't held by Message any more
*)
SetAttributes[patternizeExpectedMessage, HoldFirst]
patternizeExpectedMessage[name_MessageName] := HoldForm[Message[name, ___]]
(*HoldPattern is necessary for the definition, when Message and MessageName are not blocked yet, they are only blocked at runtime*)
patternizeExpectedMessage[HoldPattern[Message][name_, args___]] :=
	With[{args2 = Sequence @@ HoldForm /@ Unevaluated[{args}]},
		With[{m = Message[name, args2]},
			HoldForm[m]
		]
	]
(* If given a fully specified Message wrapped in HoldForm, don't do any further patternizing on it *)
patternizeExpectedMessage[HoldForm[Message[name_,args___]]]:= HoldForm[Message[name,args]];
patternizeExpectedMessage[expr_/;!FreeQ[expr,_MessageName|_Message]]:= patternizeExpectedMessage/@expr
(* let other patterns fall-through *)
patternizeExpectedMessage[arg_] := arg


(* $DefaultMessagesEquivalenceFunction is the equivalence function that determines if a list of 
expected messages (possibly patterns) matches the actual list of thrown messages *)
SetAttributes[$DefaultMessagesEquivalenceFunction, HoldAllComplete]

$DefaultMessagesEquivalenceFunction[actualMsgs_, expectedMsgs_] :=
	Module[{expectedMsgs2},
		Block[{Message, MessageName},
			SetAttributes[Message, HoldAll];
			SetAttributes[MessageName, HoldFirst];
			expectedMsgs2 =
				Switch[expectedMsgs,
					_MessageName | _Message,
					{patternizeExpectedMessage[expectedMsgs]}
					,
					_List,
					Map[patternizeExpectedMessage, expectedMsgs]
					,
					_Alternatives,
					Replace[Map[patternizeExpectedMessage, expectedMsgs],arg:Except[_List]:>{arg},{1}]
					,
					_Repeated | _RepeatedNull,
					{patternizeExpectedMessage[expectedMsgs]}
					,
					_,
					expectedMsgs
				];
			MatchQ[actualMsgs, expectedMsgs2]
		]
	]

SetAttributes[$DefaultExpectedOutputWrapper, HoldFirst]
(*
This is a little trick to make sure that HoldForm contains what expectedEvaled evaluates to,
without actually evaluating.
We don't want to evaluate it, since there may be side-effects.
*)
$DefaultExpectedOutputWrapper[expectedEvaled_] := HoldForm[expectedEvaled] /. OwnValues[expectedEvaled]

SetAttributes[$DefaultActualOutputWrapper, HoldFirst]

$DefaultActualOutputWrapper[actual_] := HoldForm[actual] /. OwnValues[actual]


SetAttributes[$DefaultExpectedMessagesWrapper, HoldFirst]
Options[$DefaultExpectedMessagesWrapper] = {"ListWrapper" -> True}
(*
nothing depends on ExpectedMessages, so only wrap entire in HoldForm. The individual args do not
need to be wrapped in HoldForm, like with ActualMessages.
*)
$DefaultExpectedMessagesWrapper[expectedMsg:(_MessageName|_Message), OptionsPattern[]] := 
    If[OptionValue["ListWrapper"],
       {HoldForm[expectedMsg]},
       HoldForm[expectedMsg]];
$DefaultExpectedMessagesWrapper[expectedMsgs_List, OptionsPattern[]] := 
    Map[Function[u,$DefaultExpectedMessagesWrapper[u,"ListWrapper"->False],HoldFirst], Unevaluated[expectedMsgs]]
$DefaultExpectedMessagesWrapper[expectedMsgs_Alternatives, opts:OptionsPattern[]] := 
    Map[Function[u,$DefaultExpectedMessagesWrapper[u,opts],HoldFirst], Unevaluated[expectedMsgs]]
$DefaultExpectedMessagesWrapper[expectedMsgs_,OptionsPattern[]] := expectedMsgs


(* Create helpful variants of Test with different SameTest specifications : TestMatch, TestStringMatch etc. *)

AddTestFunction[testFunc_] :=
	Module[{},
		Attributes[testFunc] = Attributes[Test];
		Options[testFunc] = Options[Test];
		testFunc[input_, expected_:True, Shortest[messages_:{}], opts:OptionsPattern[]] :=
			With[{options = Join[Flatten[{opts}], Options[testFunc]]},
				Test[input, expected, messages, options]
			]
	]

AddTestFunction[TestMatch]

SetOptions[TestMatch, SameTest -> MatchQ]

AddTestFunction[TestStringMatch]

SetOptions[TestStringMatch, SameTest -> StringMatchQ]

AddTestFunction[TestFree]

SetOptions[TestFree, SameTest -> FreeQ]

AddTestFunction[TestStringFree]

SetOptions[TestStringFree, SameTest -> StringFreeQ]


(* Code for the TestResult and its constructor *)

TestResult::valid =
"The symbol `1` was not created by MUnit and is not a valid test result."

Options[TestResult] =
		{TestIndex -> 0,
		TestMemoryUsed -> 0,
		TestCPUTimeUsed -> 0,
		TestAbsoluteTimeUsed -> 0,
		TestSource -> Null} ~Join~ Options[Test]

TestResultQ[_] =
	False (* only objects created by newTestResult are valid, everything else is invalid*)

newTestResult[
	test_,
	failureMode_,
	input_,
	expectedOutput_,
	actualOutput_,
	expectedMsgs_,
	actualMsgs_,
	errorMsg_,
	opts:OptionsPattern[]] :=
	Module[{hash = Hash[{test, failureMode, input, expectedOutput, actualOutput, expectedMsgs, actualMsgs, errorMsg, opts}]},
		With[{obj = Symbol["MUnit`TestResultObjects`TestResult" <> IntegerString[hash, 16, 8] <> IntegerString[$SessionID, 16, 8] <> IntegerString[$ModuleNumber, 16, 8]]},
			Module[{},
				obj /: TestResultQ[obj] = True;
				obj /: Format[obj, OutputForm] :=
					Switch[FailureMode[obj],
						"Success",
						StringJoin["-", "Success", "-"]
						,					
						"MessagesFailure",
						StringJoin["-", "MessagesFailure: ", ToString[ActualMessages[obj], OutputForm], "-"]
						,
						"Failure",
						StringJoin["-", "Failure: ", ToString[ActualOutput[obj], OutputForm], "-"]
						,
						"Error",
						StringJoin["-", "Error: ", ErrorMessage[obj], "-"]
					];
				obj /: Format[obj, StandardForm] :=
					Switch[FailureMode[obj],
						"Success",
						Interpretation[Row[{RawBoxes["\[SkeletonIndicator]"], Row[{RawBoxes["TestResult"], RawBoxes["["], "Success", RawBoxes["]"]}], RawBoxes["\[SkeletonIndicator]"]}], obj]
						,
						"MessagesFailure",
						Interpretation[Row[{RawBoxes["\[SkeletonIndicator]"], Row[{RawBoxes["TestResult"], RawBoxes["["], Row[{"MessagesFailure:", " ", ActualMessages[obj]}], RawBoxes["  "], RawBoxes[compareMessagesButtonBox[Null, obj]], RawBoxes["]"]}], RawBoxes["\[SkeletonIndicator]"]}], obj]
						,
						"Failure",
						Interpretation[Row[{RawBoxes["\[SkeletonIndicator]"], Row[{RawBoxes["TestResult"], RawBoxes["["], Row[{"Failure:", " ", ActualOutput[obj]}], RawBoxes["  "], RawBoxes[compareOutputButtonBox[Null, obj]], RawBoxes["]"]}], RawBoxes["\[SkeletonIndicator]"]}], obj]
						,
						"Error",
						Interpretation[Row[{RawBoxes["\[SkeletonIndicator]"], Row[{RawBoxes["TestResult"], RawBoxes["["], Row[{"Error:", " ", ErrorMessage[obj]}], RawBoxes["]"]}], RawBoxes["\[SkeletonIndicator]"]}], obj]
					];
				obj /: FailureMode[obj] := failureMode;
				obj /: TestInput[obj] := input;
				obj /: ExpectedOutput[obj] := expectedOutput;
				obj /: ActualOutput[obj] := actualOutput;
				obj /: ExpectedMessages[obj] := expectedMsgs;
				obj /: ActualMessages[obj] := actualMsgs;
				obj /: ErrorMessage[obj] := errorMsg;
				obj /: TestID[obj] := OptionValue[TestResult, {opts}, TestID];
				obj /: TestIndex[obj] := OptionValue[TestResult, {opts}, TestIndex];
				obj /: TestInputSetFunction[obj] := OptionValue[TestResult, {opts}, TestInputSetFunction];
				obj /: ActualOutputSetFunction[obj] := OptionValue[TestResult, {opts}, ExpectedOutputSetFunction];
				obj /: ExpectedOutputSetFunction[obj] := OptionValue[TestResult, {opts}, ExpectedOutputSetFunction];
				obj /: ActualOutputWrapper[obj] := OptionValue[TestResult, {opts}, ActualOutputWrapper];
				obj /: ExpectedOutputWrapper[obj] := OptionValue[TestResult, {opts}, ExpectedOutputWrapper];
				obj /: SameTest[obj] := OptionValue[TestResult, {opts}, SameTest];
				obj /: MessagesEquivalenceFunction[obj] := OptionValue[TestResult, {opts}, MessagesEquivalenceFunction];
				obj /: ExpectedMessagesWrapper[obj] := OptionValue[TestResult, {opts}, ExpectedMessagesWrapper];
				obj /: TestFailureMessage[obj] := OptionValue[TestResult, {opts}, TestFailureMessage];
				obj /: TestFailureAction[obj] := OptionValue[TestResult, {opts}, TestFailureAction];
				obj /: TestErrorAction[obj] := OptionValue[TestResult, {opts}, TestErrorAction];
				obj /: TestCPUTimeUsed[obj] := OptionValue[TestResult, {opts}, TestCPUTimeUsed];
				obj /: TestAbsoluteTimeUsed[obj] := OptionValue[TestResult, {opts}, TestAbsoluteTimeUsed];
				obj /: TestMemoryUsed[obj] := OptionValue[TestResult, {opts}, TestMemoryUsed];
				obj /: TestTags[obj] := OptionValue[TestResult, {opts}, TestTags];
				obj /: TestSource[obj] := OptionValue[TestResult, {opts}, TestSource];
				obj /: TestClass[obj] := OptionValue[TestResult, {opts}, "TestClass"];
				(* WRI-specific *)
				obj /: NTestFailureMessage[obj] := OptionValue[TestResult, {opts}, NTestFailureMessage];
				obj /: OrNTestFailureMessages[obj] := OptionValue[TestResult, {opts}, OrNTestFailureMessages];
				obj
			]
		]
	]

(* deprecated *)
AllTestIndex[obj_] := TestIndex[obj]

End[]
