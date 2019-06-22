(* ::Package:: *)

(*
WRI.m provides a compatibility layer for the old-style Testing package
These symbols are reproduced from the Testing package.
*)

NTest::usage = 
"NTest[input, expected, prec] tests whether input evaluates to expected with prec numerical precision. \n
NTest[input, expected, prec, messages] tests whether input evaluates to expected with prec numerical precision and the given messages are issued."

ExactTest::usage =
"ExactTest[input, expected] tests whether input evaluates to expected while maintaining expected in unevaluated form.\n
ExactTest[input, expected, messages] tests whether input evaluates to expected while maintaining expected in unevaluated form and the given messages are generated."

OrTest::usage = 
"OrTest[input, {expected1, expected2,...}] tests whether input evaluates to any of the expected1, expected2, etc. \n
OrTest[input, {expected1, expected2,...}, messages] tests whether input evaluates to any of the expected1, expected2, etc. and the given messages are issued."

OrNTest::usage = 
"OrNTest[input, {expected1, expected2,...}, prec] tests whether input evaluates to any of the expected1, expected2, etc. with prec numerical precision. \n
OrNTest[input, {expected1, expected2,...}, prec, messages] tests whether input evaluates to any of the expected1, expected2, etc. with prec numerical precision and the given messages are issued."


ConditionalTest

DefectiveTest

EndIgnore

EndRequirement

ExactTest

ExactInputTest

ExactTestCaveat

NTest

NTestCaveat

OrTest

TestCaveat

TestExecute

TestIgnore

TestRequirement

$DefaultTestManager

$FailurePriority

$FailureReportProlog

$MinimumTestAbortTime

$MemoryMessages

$RunDefectiveTests

$SaveTiming

$ShowMemoryInUse

$ShowReport

$TestAbortTime

$TestAbortTimeInitial

$TestAbortTimeMultiplier

$TestMemoryLimitInitial

$TestMemoryLimit

$TestSearchPath

TestNameTemporary

OrNTest

Begin["`Package`"]

NTestError

End[]

Begin["`WRI`Private`"]

compatOptionValue[name_, opts_, sym_] :=
	name /. Flatten[opts] /. Options[sym]

Attributes[ConditionalTest] = {HoldAll}

Options[ConditionalTest] = Options[Test]

Attributes[$Hold] = {HoldAll}

Attributes[testtypeheld] = {HoldAll}

(* opts isn't using ___?OptionQ because OptionQ evaluates its arguments and we don't want that *)
(*TODO : Investigate using OptionsPattern[] here instead of checking for Rule/RuleDelayed *)
ConditionalTest[test_, input_, expectedList__List, opts:(_Rule | _RuleDelayed)...] :=
	Module[{expres, expresbool, boolposn, returnargs, t},
		expres = Map[Hold, Hold[{expectedList}], 3];
		expres = ReleaseHold[ReleaseHold[ReleaseHold[expres]]];
		expresbool = Map[ReleaseHold[#[[1]]] &, expres];
		boolposn = Position[expresbool, True];
		returnargs = Drop[expres[[boolposn[[1, 1]]]], 1];
		returnargs = Map[Apply[$Hold, #] &, returnargs];
		With[{returnargs = returnargs},
			testtypeheld[input, t, opts] /. t -> Apply[Sequence, returnargs] /. $Hold[x_] -> x /. testtypeheld :> test
		]
	]

ConditionalTest[args___] :=
	With[{msg = "Incorrect arguments: " <> ToString[Unevaluated[{args}]]},
		testError[msg, args]
	]

Attributes[DefectiveTest] = {HoldAllComplete}


Attributes[ExactTest] = {HoldAll}
Options[ExactTest] = Options[Test]
SetOptions[ExactTest,
	(* wrap input in Hold, even though it is already evaluated, to easily compare to Hold[expected] *)
	ActualOutputSetFunction -> Function[{actual, input}, actual = Hold[input], {HoldFirst}],
	ExpectedOutputSetFunction -> Function[{expectedEvaled, expected}, expectedEvaled = Hold[expected], {HoldAll}],
	(* actual already wrapped in Hold, so just replace *)
	ActualOutputWrapper -> Function[{actual}, HoldForm @@ actual, {}],
	(* expectedEvaled already wrapped in Hold, so just replace *)
	ExpectedOutputWrapper -> Function[{expectedEvaled}, HoldForm @@ expectedEvaled, {}]
]

ExactTest[input_, expected_:True, Shortest[messages_:{}], opts:OptionsPattern[]] :=
	With[{opts2 = Join[Flatten[{opts}], Options[ExactTest]]},
		Test[input, expected, messages, opts2]
	]

ExactTest[args___] :=
	With[{msg = "Incorrect arguments: " <> ToString[Unevaluated[{args}]]},
		testError[msg, args]
	]



Attributes[ExactInputTest] = {HoldAll}
Options[ExactInputTest] = Options[Test]
SetOptions[ExactInputTest,
	ActualOutputSetFunction -> Function[{actual, input}, actual = ToString[Unevaluated[input], InputForm], {HoldFirst}],
	ExpectedOutputSetFunction -> Function[{expectedEvaled, expected}, expectedEvaled = ToString[Unevaluated[expected], InputForm], {HoldAll}],
	ActualOutputWrapper -> Function[{actual}, HoldForm[actual], {}],
	ExpectedOutputWrapper -> Function[{expectedEvaled}, HoldForm[expectedEvaled], {}]
]

ExactInputTest[input_, expected_:True, Shortest[messages_:{}], opts:OptionsPattern[]] :=
	With[{opts2 = Join[Flatten[{opts}], Options[ExactInputTest]]},
		Test[input, expected, messages, opts2]
	]

ExactInputTest[args___] :=
	With[{msg = "Incorrect arguments: " <> ToString[Unevaluated[{args}]]},
		testError[msg, args]
	]
	
(* Utility functions for NTest when using ULPS *)
CompareNumerical[e2_, e1_, tol_] :=
	Module[{t1, t2, temp},
		(*If[Precision[e1] === Infinity,
			(*Message[CompareNumerical::exact, e1, e2]*)
			Null
		];*)
		t2 = RepNums[ReleaseHold[e2]];
		t1 = RepNums[ReleaseHold[e1]];
		If[First[t1] =!= First[t2],
			{False},
			temp = NumberCompare[Last[t1], Last[t2], tol];
			If[temp === True, {True}, {False, {"CompareNumerical", {False, temp}}}]
		]
	]

(* RepNums takes an expression and replaces the numbers in it with
   numbered slots.  For instance:
   2.3 + x^4.2  --> {$Number[1] + x^$Number[2], {2.3, 4.2}}
   We then check that the structure is the same and that the numbers
   are the same within tolerances. *)
(*
This version differs from the Testing` package because it uses Replace instead of Map.
This version works correctly for expressions whose heads have Hold* attributes.
In the old Map version, you could end up with results like

{SysBody[(If[NumberQ[#1] && #1 =!= Overflow[], 
      If[Precision[#1] < \[Infinity] || 
        Precision[#1] === MachinePrecision, 
       AppendTo[nums, #1]; $Number[numcount++], #1], #1] &)[1], 
  "<>"], {}}
  
for heads like SysBody that have HoldAll attribute, because the function was never being applied.

*)
RepNums[expr_] :=
	Module[{nums = {}, numcount = 1, t},
		t =
		Replace[
			expr
			,
			{n_?NumberQ :>
				With[{res = If[n =!= Overflow[],
								If[Or[Precision[n] < Infinity, Precision[n] === MachinePrecision],
									AppendTo[nums, n];
									$Number[numcount++]
									,
									n
								]
								,
								n
							]},
					res /; True]}
			,
			{-1}
		];
		{t, nums}
	]

(* NumberCompare0[ ] needs to be Listable *)
Attributes[NumberCompare0] = {Listable}

NumberCompare[e1_, e2_, tol_] :=
    AndNumerical[NumberCompare0[e1, e2, tol]]

NumberCompare0[n1_, n2_, tol_] :=
	Module[{t, n1a, n2a},
		If[Accuracy[n1] > Accuracy[n2] && !MachineNumberQ[n1],
			(*Message[CompareNumerical::fuzzy, n1, n2]*)
			Null
		];
		If[Abs[n1] < 10^-$MachinePrecision,
			n1a = n1 + 10.^-$MachinePrecision, n1a=n1
		];
		If[Abs[n2] < 10^-$MachinePrecision,
			n2a = n2 + 10.^-$MachinePrecision, n2a=n2
		];
		If[Or[Precision[n1a] === MachinePrecision, Precision[n1a] <= $MachinePrecision] && Abs[n1a] < .00000001 && Abs[n2a] < .00000001,
			(* had previously included NumberQ[Log[n1]] && NumberQ[Log[n2]] *)
			t = 10^(Abs[Log[10.,n1a] - Log[10.,n2a]]),
			(* note I wrap N[] around Min[] on 5-15-97 *)
			t = Abs[n1a-n2a] * 10^N[Min[Accuracy[n1a],Accuracy[n2a]]],
			(* note I wrap N[] around Min[] on 5-15-97 *)
			t = Abs[n1a-n2a] * 10^N[Min[Accuracy[n1a],Accuracy[n2a]]]
		];
		If[t < tol,
			True,
			t,
			t
		]
	]

AndNumerical[l_] :=
	Module[{numlist = Select[l, NumberQ]},
		If[numlist == {},
			Apply[And, l],
			Apply[Max, numlist]
		]
	]



Attributes[NTest] = {HoldAll}

Options[NTest] = Union[Options[Test], {AccuracyGoal -> None, PrecisionGoal -> None, Tolerance -> None}]

(* setting EquivalenceFunction can't work because options need to be passed to CompareToPAT *)

NTest[input_, expected_, Optional[preUlps:(_?NumberQ | None), 10^(3+$MachinePrecision-16)], Shortest[messages_:{}], opts___?OptionQ] :=
	Module[{accuracy, precision, tolerance, ulps},
		accuracy = compatOptionValue[AccuracyGoal, {opts}, NTest];
		precision = compatOptionValue[PrecisionGoal, {opts}, NTest];
		tolerance = compatOptionValue[Tolerance, {opts}, NTest];
		(* treat ulps of None same as default, for compatibility *)
		ulps = preUlps /. None -> 10^(3+$MachinePrecision-16);
		If[{accuracy, precision, tolerance} =!= {None, None, None},
			With[{
				(* use accuracy, precision, tolerance *)
				accuracy = accuracy, precision = precision, tolerance = tolerance,
				opts2 = FilterRules[Join[Flatten[{opts}], Options[NTest]], Except[FilterRules[Options[NTest], Except[Options[Test]]]]]},
				Test[
					input,
					expected,
					messages,
					SameTest -> $NTestPATEquivalenceFunction[{accuracy, precision, tolerance}],
					TestInputWrapper -> Function[Null, HoldForm[input], {HoldFirst}],
					opts2
				]
			]
			,
			(* use ulps *)
			With[{opts2 = FilterRules[Join[Flatten[{opts}], Options[NTest]], Except[FilterRules[Options[NTest], Except[Options[Test]]]]]},
				Test[
					input,
					expected,
					messages,
					SameTest -> $NTestULPSEquivalenceFunction[ulps],
					opts2
				]
			]
		]
	]

NTest[args___] :=
	With[{msg = "Incorrect arguments: " <> ToString[Unevaluated[{args}]]},
		testError[msg, args]
	]


InformativeString[_Internal`CompareToPAT] := "The arguments to the numerical comparison function were incorrect."

InformativeString[{string_String, _, {}}] = string;

InformativeString[{Tolerance, -Infinity, _}] := "Tolerance was satisfied"

InformativeString[{Precision | Accuracy, Infinity, _}] := StringJoin[
	ToString[pat],
	"Goal was satisfied"]

InformativeString[{Precision, perr_, {}}] := StringJoin[
	"Actual only matches expected to ",
	ToString[perr],
	" digits of Precision (relative error)"]

InformativeString[{Accuracy, aerr_, {}}] :=StringJoin[
	"Actual only matches expected to ",
	ToString[aerr],
	" digits of Accuracy (absolute error)"]

InformativeString[{Tolerance, terr_, {}}] :=StringJoin[
	"Actual deviates from expected with ",
	ToString[terr],
	" least significant digits (Tolerance)"]

InformativeString[{pat_, perr_, Condition[pos_List, Length[pos] > 0]}] := StringJoin[
	InformativeString[{pat, perr, {}}],
	" at position ",
	ToString[pos]]

InformativeString[{False, ulps_}] := StringJoin[
	"Error (ULPS):   ", 
	ToString[InputForm[ulps]]]

InformativeString[False] := "NOT AN ULPS FAILURE (check types and shape)"

$NTestPATEquivalenceFunction[{accuracy_, precision_, tolerance_}][actual_, expected_] :=
	Module[{res},
		res = Internal`CompareToPAT[
			expected,
			actual,
			AccuracyGoal -> accuracy,
			PrecisionGoal -> precision,
			Tolerance -> tolerance
		];
		If[TrueQ[res],
			True
			,
			Sow[{res, "CompareToPAT"}, NTestError];
			False
		]
	]

$NTestULPSEquivalenceFunction[ulps_][actual_, expected_] :=
	Module[{res, temp},
		res = CompareNumerical[Hold[actual], Hold[expected], ulps];
		Switch[res,
			{True},
			True
			,
			{False, {"CompareNumerical", {False, _}}},
			temp = res[[2,2,2]];
			Sow[{temp, "ULPS"}, NTestError];
			False
			,
			_,
			False
		]
	]


Attributes[OrTest] = Attributes[Test]
Options[OrTest] = Options[Test]
SetOptions[OrTest, SameTest -> Function[{actual, expected}, Or @@ (SameQ[actual, #]& /@ expected)]]

OrTest[input_, expected:(_[___]), Shortest[messages_:{}], opts___?OptionQ] :=
	With[{opts2 = Join[Flatten[{opts}], Options[OrTest]]},
		Test[input, expected, messages, opts2]
	]

OrTest[args___] :=
	With[{msg = "Incorrect arguments: " <> ToString[Unevaluated[{args}]]},
		testError[msg, args]
	]


Attributes[TestCaveat] = Attributes[Test]
Options[TestCaveat] = Options[Test]

TestCaveat[input_, expected_, caveat_, Shortest[messages_:{}], opts___?OptionQ] :=
	Test[input, expected, messages, TestFailureMessage :> caveat, opts]

TestCaveat[args___] :=
	With[{msg = "Incorrect arguments: " <> ToString[Unevaluated[{args}]]},
		testError[msg, args]
	]




Attributes[NTestCaveat] = Attributes[NTest]
Options[NTestCaveat] = Options[NTest]

NTestCaveat[input_, expected_, Optional[ulps:(_?NumberQ | None), 10^(3+$MachinePrecision-16)], caveat_, Shortest[messages_:{}], opts___?OptionQ] :=
	NTest[input, expected, ulps, messages, TestFailureMessage :> caveat, opts]

NTestCaveat[args___] :=
	With[{msg = "Incorrect arguments: " <> ToString[Unevaluated[{args}]]},
		testError[msg, args]
	]





Attributes[ExactTestCaveat] = Attributes[ExactTest]
Options[ExactTestCaveat] = Options[ExactTest]

ExactTestCaveat[input_, expected_, caveat_, Shortest[messages_:{}], opts___?OptionQ] :=
	Test[input, expected, messages, TestFailureMessage -> caveat, opts]

ExactTestCaveat[args___] :=
	With[{msg = "Incorrect arguments: " <> ToString[Unevaluated[{args}]]},
		testError[msg, args]
	]
	

(* TestExecute is currently a noop *)
TestExecute = Identity


(*TestRequirement[cond_] :=
	BeginTestSection["", cond]

EndRequirement = EndTestSection*)

TestNameTemporary[name_String:"TestTemp"] := ToFileName[{$TemporaryDirectory}, name]


Attributes[OrNTest] = Attributes[NTest]

Options[OrNTest] = Options[NTest]

OrNTest[input_, expected:(_[___]), Optional[preUlps:(_?NumberQ | None), 10^(3+$MachinePrecision-16)], Shortest[messages_:{}], opts___?OptionQ] :=
	Module[{accuracy, precision, tolerance, ulps},
		accuracy = compatOptionValue[AccuracyGoal, {opts}, OrNTest];
		precision = compatOptionValue[PrecisionGoal, {opts}, OrNTest];
		tolerance = compatOptionValue[Tolerance, {opts}, OrNTest];
		(* treat ulps of None same as default, for compatibility *)
		ulps = preUlps /. None -> 10^(3+$MachinePrecision-16);
		If[{accuracy, precision, tolerance} =!= {None, None, None},
			With[{
				(* use accuracy, precision, tolerance *)
				accuracy = accuracy, precision = precision, tolerance = tolerance,
				opts2 = FilterRules[Join[Flatten[{opts}], Options[OrNTest]], Except[FilterRules[Options[OrNTest], Except[Options[Test]]]]]},
				Test[
					input,
					expected,
					messages,
					SameTest -> $OrNTestPATEquivalenceFunction[{accuracy, precision, tolerance}],
					TestInputWrapper -> Function[Null, HoldForm[input], {HoldFirst}],
					opts2
				]
			]
			,
			(* use ulps *)
			With[{opts2 = FilterRules[Join[Flatten[{opts}], Options[OrNTest]], Except[FilterRules[Options[OrNTest], Except[Options[Test]]]]]},
				Test[
					input,
					expected,
					messages,
					SameTest -> $OrNTestULPSEquivalenceFunction[ulps],
					opts2
				]
			]
		]
	]

OrNTest[args___] :=
	With[{msg = "Incorrect arguments: " <> ToString[Unevaluated[{args}]]},
		testError[msg, args]
	]

$OrNTestPATEquivalenceFunction[{accuracy_, precision_, tolerance_}][actual_, expected_] :=
	Or @@ ($NTestPATEquivalenceFunction[{accuracy, precision, tolerance}][actual, #]& /@ expected)

$OrNTestULPSEquivalenceFunction[ulps_][actual_, expected_] :=
	Or @@ ($NTestULPSEquivalenceFunction[ulps][actual, #]& /@ expected)

End[]
