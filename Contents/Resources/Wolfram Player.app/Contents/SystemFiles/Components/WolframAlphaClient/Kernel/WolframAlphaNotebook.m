(* ::Package:: *)

(* ::Section::Closed:: *)
(*Package header*)


(*
	AlphaIntegration`CreateWolframAlphaNotebook,
	AlphaIntegration`NaturalLanguageInputAssistant,
	AlphaIntegration`NaturalLanguageInputBoxes,
	AlphaIntegration`NaturalLanguageInputParse,
	AlphaIntegration`NaturalLanguageInputEvaluate,
	AlphaIntegration`DuplicatePreviousCell
*)

Begin["WolframAlphaClient`Private`"];



(*$AlphaQueryExtrusionMonitor = True;*)



$WolframAlphaNotebookShowSteps = True;


(* ::Section::Closed:: *)
(*AlphaIntegration`NaturalLanguageInputParse*)


(*
When calling AlphaIntegration`ExtrusionEvaluate with $WolframAlphaNotebook set to True:
* We need to block replaceEvaluationCell, to avoid any side effects.
* A return value of Null means failure.
* A return value of XMLElement["examplepage", __] means that the example page should be shown.
* Any other return value will go through chooseAndReturnAValue.
* In this mode, that utility should always return a list of the form {Defer[parse], assumptions}
*)


ClearAll[AlphaIntegration`NaturalLanguageInputParse];

AlphaIntegration`NaturalLanguageInputParse["", opts___] = {Failure["NoQuery", <|"Query" -> ""|>], {}};

AlphaIntegration`NaturalLanguageInputParse[str_, opts___] :=
	(* 
		With the Block on $WolframAlphaNotebook below, that will put ExtrusionEvaluate into a mode
		where it returns {Defer[parse], assumptions} or Null.
	*)
	Block[{
			result,
			replaceEvaluationCell,
			$WolframAlphaNotebook = True
		},
		If[$Notebooks && !CurrentValue["InternetConnectionAvailable"],
			Return @ {Failure["Offline", <|"Query" -> str|>], {}}];
		If[$Notebooks && !CurrentValue["AllowDownloads"],
			Return @ {Failure["NetworkDisabled", <|"Query" -> str|>], {}}];
		If[$WolframAlphaNotebookShowSteps &&
			MemberQ[result = PreProcessSBS[str], "StepByStepQuery" -> True] &&
			sbsConfirmedQ[result],
			(* If this is a step-by-step query, show a step-by-step result *)
			With[{query = "QueryString" /. result},
				Return[{Defer[AlphaIntegration`WolframAlphaStepByStep[query]], {}}]
			]
			(* otherwise, keep going *)
		];
		
		replaceEvaluationCell[___] := Null;
		Replace[
			AlphaIntegration`ExtrusionEvaluate[str, InputForm, opts],
			{
				{Defer[expr_], assumptions_} :> {Defer[expr], assumptions},
				{None, assumptions_} :> {Failure["NoParse", <|"Query" -> str|>], assumptions},
				Null -> {Failure["NoParse", <|"Query" -> str|>], {}}
			}
		]
	]


sbsConfirmedQ[{___, "QueryString" -> query_String, ___}] := 
	Module[{sbsresult, tmpobj},
		tmpobj = PrintTemporary[Internal`LoadingPanel[Row[{
					stringResource["StepByStepCheck"],
					ProgressIndicator[Appearance -> "Ellipsis"]
				}]]];
		sbsresult = AlphaIntegration`WolframAlphaStepByStep[query];
		NotebookDelete[tmpobj];
		Switch[sbsresult,
			{___, "ReturnType" -> Except["Reinterpreted"], ___},
				(* Cache the result so we don't have to duplicate the query in AlphaIntegration`NaturalLanguageInputEvaluate *)
				$cachedSBSResults = {query, sbsresult};
				True,
			_,
				PrintTemporary[Internal`LoadingPanel[Row[{
					stringResource["NoStepByStepContinuing"],
					ProgressIndicator[Appearance -> "Ellipsis"]
				}]]];
				False
		]
	];
sbsConfirmedQ[other_] := False


(* ::Section::Closed:: *)
(*AlphaIntegration`NaturalLanguageInputAssistant*)


(* ::Subsection::Closed:: *)
(*NaturalLanguageInputAssistant*)


Options[AlphaIntegration`NaturalLanguageInputAssistant] = {InputAssumptions -> {}};

AlphaIntegration`NaturalLanguageInputAssistant[querystring_String, parse: (Automatic | _Defer), OptionsPattern[]] := 
DynamicModule[{Typeset`querydata, Typeset`update=0},
	Typeset`querydata = Association[
		"query" -> querystring,
		"assumptions" -> OptionValue[InputAssumptions],
		"parse" -> Replace[parse, Automatic -> None],
		"inputpredictions" -> {},
		"otherpredictions" -> {},
		"allassumptions" -> {},
		"summarizeassumptions" -> Automatic,
		"summarizepredictions" -> True
	];
	If[MatchQ[parse, _Defer],
		(* use the given parse without going to W|A *)
		Typeset`querydata["inputpredictions"] = getInputPredictions[parse];
		Typeset`querydata["otherpredictions"] = getOtherPredictions[Typeset`querydata];
		,
		(* otherwise, get a fresh parse from W|A *)
		updateNaturalLanguageAssistant[Typeset`querydata, Typeset`update, False]
	];
	Dynamic[AlphaIntegration`NaturalLanguageInputBoxes[1, Typeset`querydata, Typeset`update], TrackedSymbols -> {}],
	BaseStyle -> {Deployed -> True}
]


(* Other parses aren't handled yet *)
AlphaIntegration`NaturalLanguageInputAssistant[querystring_String, parse_, OptionsPattern[]] := 
	AlphaIntegration`NaturalLanguageInputAssistant[stringResource["ActionNotSupported"], Defer[parse;]]



SetAttributes[AlphaIntegration`NaturalLanguageInputBoxes, HoldAll];

AlphaIntegration`NaturalLanguageInputBoxes[version: 1, qd_, update_] := 
	Style[
		EventHandler[
			Dynamic[
				update;
				Column[
					Replace[
						Flatten[{
							naturalLanguageInputField[qd, update],
							naturalLanguageAssumptions[qd, update],
							naturalLanguagePredictions[qd, update]
						}],
						(*
							When the input field is the only item present, add some space
							below it to make the enclosing 9-patch look symmetric.
						*)
						{inputfield_Framed} :> {Pane[inputfield,
							ImageMargins -> {{0,0},{6,0}},
							BaselinePosition -> BaselinePosition]}
					],
					BaselinePosition -> {1,1}
				],
				TrackedSymbols :> {update}
			],
			{"MenuCommand", "HandleShiftReturn"} :>
				(updateNaturalLanguageAssistant[qd, update, True]),
			{"MenuCommand", "EvaluateCells"} :>
				(updateNaturalLanguageAssistant[qd, update, True]),
			Method -> "Queued"
		],
		"ControlStyle",
		ShowAutoStyles -> False
	]

AlphaIntegration`NaturalLanguageInputBoxes[___] := 
	Style[
		"Displaying this Natural Language interface requires a more recent version.",
		"ControlStyle",
		FontColor -> GrayLevel[0.6]
	]


(* ::Subsection::Closed:: *)
(*updateNaturalLanguageAssistant*)


SetAttributes[updateNaturalLanguageAssistant, HoldAll];

updateNaturalLanguageAssistant[qd_, update_, evaluateQ_] := 
	(
		(*
			When evaluating a cell, the insertion point should move after that
			cell, and after any output or output-like cells that follow it. This
			ensures that the output predictions can show up, for example.
		*)
		If[evaluateQ, moveAfterPreviousOutputs[EvaluationCell[], EvaluationNotebook[]]];
		
		{qd["parse"], qd["allassumptions"]} = AlphaIntegration`NaturalLanguageInputParse[qd["query"], InputAssumptions -> qd["assumptions"]];
		If[MatchQ[qd["parse"], _Defer],
			qd["inputpredictions"] = getInputPredictions[qd["parse"]];
			qd["otherpredictions"] = getOtherPredictions[qd];
			,
			qd["inputpredictions"] = {};
			qd["otherpredictions"] = {};
		];
		
		update++;
		If[evaluateQ,
			If[BoxForm`sufficientVersionQ[12.0],
				MathLink`CallFrontEnd[FrontEnd`CellEvaluate[EvaluationCell[]]],
				(* Before this packet, there was no way to do this without moving the selection. *)
				SelectionMove[EvaluationCell[], All, Cell];
				FrontEndTokenExecute[EvaluationNotebook[], "EvaluateCells"]
			]
		]
	)


(* ::Subsection::Closed:: *)
(*naturalLanguageInputField*)


SetAttributes[naturalLanguageInputField, HoldAll];

naturalLanguageInputField[qd_, update_] := 
	Framed[
		Grid[{{
			Pane[
				Style["\[FreeformPrompt]", FontSize -> 10, FontColor -> RGBColor[0.949219, 0.4375, 0.128906]],
				BaselinePosition -> Scaled[0.15]
			],
			InputField[Dynamic[qd["query"], (qd["query"] = #; qd["assumptions"] = {})&],
				String,
				BaseStyle -> {"NaturalLanguageInputField", AutoIndent -> False},
				BaselinePosition -> Baseline,
				ContinuousAction -> If[TrueQ[$CloudEvaluation], False, True],
				FrameMargins -> 5,
				Appearance -> None,
				ImageSize -> {Scaled[1], Automatic},
				System`ReturnEntersInput -> False,
				System`TrapSelection -> False
			]
		}}, Spacings -> {0,0}],
		Alignment -> {Left, Center},
		FrameMargins -> {{5,5},{0,0}},
		ImageMargins -> 0,
		BaselinePosition -> Baseline,
		FrameStyle -> Orange
	]


(* ::Subsection::Closed:: *)
(*naturalLanguageAssumptions*)


formulaAssumptionPat = XMLElement["assumption", {___, "type" -> Alternatives @@ $FormulaAssumptionTypes, ___}, _];
formulaAssumptionsQ[allassumptions_] := MatchQ[allassumptions, {XMLElement["assumptions", _, {___, formulaAssumptionPat, ___}]}]


SetAttributes[naturalLanguageAssumptions, HoldAll];

naturalLanguageAssumptions[qd_, update_] /; qd["allassumptions"] === {} := {}

naturalLanguageAssumptions[qd_, update_] :=
	Block[{formattedAssumptions, list, label, $WolframAlphaNotebook = True},
		formattedAssumptions = FormatAllAssumptions[
			"AssumptionSummary",
			qd["allassumptions"],
			Dynamic[qd["query"]], Dynamic[qd["query"]], Dynamic[{}]
		];
		If[formattedAssumptions === {{},{}}, Return @ {}];
		
		Replace[formattedAssumptions, {
			{"NonFormulaAssumptions" -> assumps_, {}} :> (list = assumps; label = stringResource["ShowAssumptions"]),
			{{}, "FormulaAssumptions" -> True} :> (list = {}; label = stringResource["ShowAssumptions"]),
			{"NonFormulaAssumptions" -> assumps_, "FormulaAssumptions" -> True} :> (list = assumps; label = stringResource["ShowAssumptionsFormulas"])
		}];
		
		label = Row[{label, RawBoxes[AdjustmentBox[StyleBox["\:203a", FontSize -> (1.5 * Inherited)], BoxBaselineShift -> 0.1]]}];
		
		Pane[#, ImageMargins -> {{5,0},{0,0}}]& @ 
		Row[
			Flatten[{
				list,
				Button[
					Mouseover[
						Style[label, "NaturalLanguageAssumptionLink"],
						Style[label, "NaturalLanguageAssumptionLinkActive"]
					],
					qd["summarizeassumptions"] = False; update++,
					Appearance -> None,
					BaseStyle -> {},
					DefaultBaseStyle -> {}
				]
			}],
			" | ",
			BaseStyle -> "NaturalLanguageAssumptions"
		]
	] /;
	Or[
		qd["summarizeassumptions"] === True,
		(* Interpret 'Automatic' as 'True' iff there are no formula assumptions *)
		qd["summarizeassumptions"] === Automatic && Not[formulaAssumptionsQ[qd["allassumptions"]]]
	]


naturalLanguageAssumptions[qd_, update_] :=
	Block[{formattedAssumptions, label, $WolframAlphaNotebook = True},
		formattedAssumptions = FormatAllAssumptions[
			"Assumptions",
			qd["allassumptions"],
			Dynamic[qd["query"]], Dynamic[qd["query"]], Dynamic[{}]
		];
		If[formattedAssumptions === {{},{}}, Return @ {}];
		
		formattedAssumptions = Replace[formattedAssumptions, Framed[Column[x_, ___], ___] :> x, {1}];
		
		(* FIXME: Style hacks that should be moved into WolframAlphaClient.m when the time comes *)
		formattedAssumptions = formattedAssumptions //. {
			"DialogStyle" -> "DialogStyles",
			Button[args___] :> Button[args, BaseStyle -> {}] /; FreeQ[Hold[{args}], BaseStyle],
			ActionMenu[args___] :> ActionMenu[args, DefaultBaseStyle -> {}] /; FreeQ[Hold[{args}], DefaultBaseStyle]
		};
		
		(* Change the button / action menu functions to the appropriate functions in this new interface *)
		formattedAssumptions = formattedAssumptions /. {
			HoldPattern[updateWithAssumptions][nb_, assumptioninputs_, Dynamic[q_], Dynamic[opts_]] :> 
			(
				qd["assumptions"] = removeDuplicateAssumptions[qd["assumptions"], assumptioninputs];
				updateNaturalLanguageAssistant[qd, update, True]
			),
			HoldPattern[setFormulaVariables][prefixes_, inputs_, Dynamic[q_], Dynamic[opts_]] :>
			(
				qd["assumptions"] = removeDuplicateAssumptions[
					qd["assumptions"],
					MapThread[assumptionFromInputAndPrefix, {inputs, prefixes}]
				];
				updateNaturalLanguageAssistant[qd, update, True]
			)
		};
		
		label = Row[{Style["\:2039", FontSize -> (1.5 * Inherited)], stringResource["HideAssumptions"]}];
		
		Pane[#, ImageMargins -> {{5,0},{0,0}}]& @ 
		Column[
			Flatten[{
				formattedAssumptions,
				Button[
					Mouseover[
						Style[label, "NaturalLanguageAssumptionLink"],
						Style[label, "NaturalLanguageAssumptionLinkActive"]
					],
					qd["summarizeassumptions"] = True; update++,
					Appearance -> None,
					BaseStyle -> {},
					DefaultBaseStyle -> {},
					ImageSize -> Automatic (* work around a Cloud rendering bug *)
				]
			}],
			BaseStyle -> "NaturalLanguageAssumptions",
			Spacings -> 0.8
		]
	]
		


(* ::Subsection::Closed:: *)
(*predictive interface utilities*)


$showInputPredictions = True;
$showOtherPredictions = True;
$stepByStepQTimeConstraint = 0.5;

$inputPredictionsMaxCount = 3;

loadPredictiveInterface[] := loadPredictiveInterface[] =
	(PredictiveInterface`PredictionControls[]; Null) (* forced autoloading *)


getInputPredictions[Defer[expr_]] := 
	Module[{semanticType, alternateTypes, predictions},
		loadPredictiveInterface[];
		If[$showInputPredictions =!= True, Return[{"", {}, {}}]];
		{semanticType, alternateTypes, predictions} =
			PredictiveInterfaceDump`GetSemanticTypesAndPredictions[expr,
					"PredictionType" -> "Input",
					"DocumentType" -> "AlphaNotebook"
				];
		With[{filtered = PredictiveInterfaceDump`removeIndirectPredictions[predictions]},
			If[ListQ[filtered], predictions = filtered] ];
		If[ListQ[predictions], predictions, {}]
	]



(* Utility to facilitate testing: *)
wanbInputPredictions[Defer[expr_]] := 
	Module[{predToRule, itemToRule},
		predToRule[{_, Predictions`Prediction[_, _, label_, HoldComplete[e_]&], _}] := label :> e;
		predToRule[{_, Predictions`Prediction[_, _, _, items: {__List}], _}] := itemToRule /@ items;
		itemToRule[{label_, HoldComplete[e_]&}] := label :> e;
		Flatten[predToRule /@ getInputPredictions[Defer[expr]]]
	]


(*
Most input predictions come from the PredictiveInterface code, but we choose to add
or not add two such buttons at display time: "show steps" and "Wolfram|Alpha results".

The "show steps" button should be shown whenever there is a parse for which the
StepByStepQ utility returns True.

The "Wolfram|Alpha results" button should be shown whenever there is a parse.
*)

SetAttributes[showInputPredictions, HoldAll]

showInputPredictions[qd_, update_] := 
Module[{allcontrols, controls, stepbutton, fullbutton, more},
	loadPredictiveInterface[];
	allcontrols = If[
		TrueQ[$showInputPredictions] && MatchQ[qd["inputpredictions"], {__}],
		(* FIXME: There's some work still to do here, but this will let most things work... *)
		Replace[
			Block[{
				PredictiveInterfaceDump`$DefaultButtonAppearance = (Appearance -> None),
				PredictiveInterfaceDump`$DefaultFrameMargins = 0 },
				PredictiveInterfaceDump`predictionsToControls[Dynamic[0], PredictiveInterfaceDump`KeyCellObjects[None, None], qd["inputpredictions"], True, "ButtonStyles"]
			],
			Style[Row[{controls__}, rowopts___], styleopts___] :> ({controls} /. PredictiveInterface`DoPredictionAction -> DoNLPredictionAction)
		],
		(* otherwise, there are no input predictions *)
		{}
	];
	stepbutton = If[MemberQ[qd["otherpredictions"], "StepByStep"], stepByStepButton[qd], {}];
	fullbutton = If[MemberQ[qd["otherpredictions"], "FullResults"], fullResultsButton[qd, allcontrols === {}], {}];
	allcontrols = Flatten[{stepbutton, allcontrols}];
	more[val_] := naturalLanguagePredictionButton[
		Style[If[val,
			Row[{bitmapResource["PredictionsUpPointer"], " ", stringResource["FewerPredictions"]}],
			stringResource["MorePredictions"] ], Italic],
		qd["summarizepredictions"] = val; ++update
	];
	
	If[Length[allcontrols] > $inputPredictionsMaxCount + 1,
		If[qd["summarizepredictions"],
			(* there is a more button, in the "closed" state *)
			controls = Flatten[{Take[allcontrols, $inputPredictionsMaxCount], more[False], fullbutton}]
			,
			(* there is a more button, in the "open" state *)
			controls = Flatten[{allcontrols, fullbutton, more[True]}]
		],
		(* there is no more button *)
		controls = Flatten[{allcontrols, fullbutton}];
	];
	
	If[controls === {}, {},
		Pane[
			Style[Row[controls, Style[RawBoxes["|"], GrayLevel[0.8]]],
				LineIndent -> 0,
				LinebreakAdjustments -> {1., 10, 1, 0, 1}
			],
			BaselinePosition -> Baseline,
			ImageMargins -> {{5,0},{0,0}}
		]
	]
]



getOtherPredictions[qd_] :=
	Flatten[{
		If[stepByStepPredictionQ[Replace[qd["parse"], Defer[expr_] :> Hold[expr]]], "StepByStep", {}],
		If[fullResultsPredictionQ[qd], "FullResults", {}]
	}]


stepByStepPredictionQ[heldparse_] :=
	TimeConstrained[
		And[
			TrueQ[$showOtherPredictions],
			Not[MatchQ[heldparse, Hold[_WolframAlpha | _AlphaIntegration`WolframAlphaStepByStep]]],
			TrueQ[$WolframAlphaNotebookShowSteps],
			TrueQ[StepByStepQ[heldparse, Quiet @ ReleaseHold @ heldparse]]
			(* FIXME: How to avoid duplicate evaluation of the parse -- once here, and once for the output? *)
		],
		$stepByStepQTimeConstraint,
		False
	]

fullResultsPredictionQ[qd_] := And[
		TrueQ[$showOtherPredictions]
	]



fullResultsButton[qd_, verbose_] :=
	naturalLanguagePredictionButton[
		If[TrueQ[verbose],
			Row[{stringResource["FullResultsButton"], stringResource["FullResultsTooltip"]}, " "],
			Tooltip[stringResource["FullResultsButton"], stringResource["FullResultsTooltip"]]
		],
		cellPrintSelectAndEvaluate[Cell[qd["query"], "WolframAlphaLong"]]
	]


stepByStepButton[qd_] :=
	naturalLanguagePredictionButton[
		stringResource["ShowStepsButton"],
		cellPrintSelectAndEvaluate[Cell["show steps " <> qd["query"], "NaturalLanguageInput"]]
	]



(*
For a given prediction, the DoNLPredictionAction should create a new
"DeployedNLInput" cell which already includes a parse and
corresponding input predictions, and then evaluate it.
*)
DoNLPredictionAction[Dynamic[mode_], keys_, {label_, action_}, inOutType_, inputQ_] :=
	Module[{in, out, theAction, expr, newquery, boxes},
		loadPredictiveInterface[];
		
		in = out = HoldComplete[None]; (* FIXME *)
		theAction = PredictiveInterfaceDump`addCellExpressions[action, keys];
		If[Head[theAction] =!= Function, Beep[]; Return[$Failed]];
		
		expr = PredictiveInterfaceDump`buildActionExpression[in, out, theAction, inOutType, inputQ];
		If[!MatchQ[expr, HoldComplete[_]], Beep[]; Return[$Failed]];
		
		expr = Defer @@ expr;
		(* Produce a new NL query from the WL parse. *)
		newquery = Symbol["NaturalLanguage`NaturalLanguage"][expr];
		If[!StringQ[newquery], newquery = label];
		boxes = BoxData[ToBoxes[AlphaIntegration`NaturalLanguageInputAssistant[newquery, expr]]];
		cellPrintSelectAndEvaluate[Cell[boxes, "DeployedNLInput"]]
	]


(* ::Subsection::Closed:: *)
(*naturalLanguagePredictions*)


SetAttributes[naturalLanguagePredictions, HoldAll];

naturalLanguagePredictions[qd_, update_] := {} /; qd["query"] === "" || !MatchQ[qd["parse"], _Defer]

naturalLanguagePredictions[qd_, update_] := showInputPredictions[qd, update]


(* naturalLanguagePredictionButton is used to format predictions which are added locally. *)
SetAttributes[naturalLanguagePredictionButton, HoldRest];
naturalLanguagePredictionButton[label_, func_, opts___] :=
	Button[
		Mouseover[label, Style[label /. "PredictionsUpPointer" -> "PredictionsUpPointerHot", "SuggestionsBarButtonLabelActive"]],
		func,
		Appearance -> None,
		BaselinePosition -> Baseline,
		BaseStyle -> "SuggestionsBarButtonLabel",
		DefaultBaseStyle -> {},
		Method -> "Queued",
		opts
	]


(* ::Section::Closed:: *)
(*AlphaIntegration`NaturalLanguageInputEvaluate*)


(* An empty query shouldn't do anything. *)
AlphaIntegration`NaturalLanguageInputEvaluate[query: "", fmt_] := Null


(* If we're evaluating a fresh cell, do the parse, and then insert and evaluate the interface cell. *)
AlphaIntegration`NaturalLanguageInputEvaluate[query_String, fmt_] := 
Module[{boxes},
	boxes = BoxData[ToBoxes[AlphaIntegration`NaturalLanguageInputAssistant[query, Automatic]]];
	(* replace the "NaturalLanguageInput" cell with a "DeployedNLInput" cell, and then evaluate it *)
	MathLink`CallFrontEnd[FrontEnd`RewriteExpressionPacket[Cell[boxes, "DeployedNLInput"]]];
	AlphaIntegration`NaturalLanguageInputEvaluate[boxes, fmt]
]


(*
Accept a RowBox of strings as a proxy for the StringJoin'd string, which is
what the Cloud will be sending. CLOUD-12925
*)
AlphaIntegration`NaturalLanguageInputEvaluate[BoxData[query: (_RowBox | _String)], fmt_] := 
Module[{querystring},
	querystring = query //. RowBox[{strs__String}] :> StringJoin[strs];
	AlphaIntegration`NaturalLanguageInputEvaluate[querystring, fmt] /; StringQ[querystring]
]


(* If we're evaluating an interface cell, remove any previous outputs, and then insert new ones. *)
AlphaIntegration`NaturalLanguageInputEvaluate[BoxData[HoldPattern[DynamicModuleBox][vars_, __]], fmt_] := 
Module[{qd, protected},
	qd = FirstCase[Hold[vars], HoldPattern[Set][Typeset`querydata$$, data_] :> data, $Failed, Infinity];
	InQueryData[$Line] = qd;
	(* delete any previous results *)
	deletePreviousOutputs[EvaluationCell[], EvaluationNotebook[]];
	
	Replace[qd["parse"], {
		(* If there was no query, revert to the simple cell. *)
		Failure["NoQuery", ___] :>
			MathLink`CallFrontEnd[FrontEnd`RewriteExpressionPacket[Cell["", "NaturalLanguageInput"]]],
		(* If there was a failure, show it as input, but do not produce an output cell. *)
		Failure[tag_String, ___] :> writeFailureInput[tag],
		(* If this query was flagged as a possible SBS query, try to go get the steps. *)
		Defer[AlphaIntegration`WolframAlphaStepByStep[query_String]] :> 
			Module[{tmpobj, sbsresult},
				If[MatchQ[$cachedSBSResults, {query, _}],
					(* if we cached these results when running sbsConfirmedQ for this query, use the cache *)
					sbsresult = $cachedSBSResults[[2]],
					(* otherwise, do the query *)
					tmpobj = PrintTemporary[Internal`LoadingPanel[Row[{
						stringResource["StepByStepCheck"],
						ProgressIndicator[Appearance -> "Ellipsis"]
					}]]];
					sbsresult = First[qd["parse"]];
					NotebookDelete[tmpobj]
				];
				(* either way, clear the cache *)
				$cachedSBSResults = {};
				Switch[FirstCase[sbsresult, ("ReturnType" -> type_) :> type, "NoParse"],
					"NoParse",
						writeFailureInput["NoParse"],
					"NoStepByStep" | "Reinterpreted",
						writeFailureInput["NoStepByStep"];
						FirstCase[sbsresult, ("Result" -> res_) :> res, $Failed],
					"HasStepByStep" | _,
						FirstCase[sbsresult, ("Result" -> res_) :> res, $Failed]		
				]
			],
		(* If the parse contains a Placeholder, write the new WLInput, but don't evaluate it *)
		Defer[expr_] /; !FreeQ[Defer[expr], _Placeholder] :> (
			writeNewWolframLanguageInput[EvaluationCell[], EvaluationNotebook[], qd["parse"], False];
			Null
		),
		(* If there was any other parse, show it in a new WLInput cell, and evaluate it *)
		Defer[expr_] :> (
			(* Set In[n] to be the parse -- FIXME: Is there a better way to do this? *)
			Internal`WithLocalSettings[
				protected = Unprotect[In], In[$Line] := expr, Protect @@ protected];
			writeNewWolframLanguageInput[EvaluationCell[], EvaluationNotebook[], qd["parse"], True];
			AlphaIntegration`NaturalLanguageInputEvaluate[Defer[expr]]
		),
		(* If there is a direct hit on an example page, construct and show the page. *)
		XMLElement["examplepage", {___, "url" -> url_String, ___}, _] :>
			showExamplesAsWolframAlphaNotebook[url],
		(* Otherwise, there was no parse. Don't produce any new inputs or outputs. *)
		(None | _) :>
			((*Beep[];*) Missing["NoParse", qd["query"]]; Null)
	}]
]



(*
The one-argument form is used when evaluating exprs in DeployedWLInput cells.
Currently, we only use this to toggle the setting of an Integrate option, but
there may be other options or other changes coming....
*)
AlphaIntegration`NaturalLanguageInputEvaluate[Defer[expr_]] :=
	Internal`InheritedBlock[
		{Integrate},
		SetOptions[Integrate, GeneratedParameters -> C];
		expr
	]


(* The procedure for determining previously generated output cells is:

	- If OutputAutoOverwrite is False for the notebook, stop.
	- Otherwise, examine the cell immediately after the input cell.
		- If it's not Deletable, stop.
		- If it's not CellAutoOverwrite, stop.
		- Otherwise, mark it for deletion, and examine the next cell.

This is not quite the same as the Desktop front end's algorithm. The FE checks
for Evaluatable -> False right after Deletable. But we can't do that, because we
have to be able to delete "DeployedWLInput" cells, which can be evaluated.

The FE also does something special if it encounters a cell group. But we're not
going to bother with that for now.
*)

previousOutputs[cellobj_, nbobj_] := 
	Module[{cells, objs = {}},
		If[Not @ TrueQ @ AbsoluteCurrentValue[nbobj, OutputAutoOverwrite], Return[{}]];
		cells = NextCell[cellobj, All];
		If[!MatchQ[cells, {__CellObject}], Return[{}]];
		Do[
			If[
				TrueQ @ AbsoluteCurrentValue[cell, Deletable] &&
				TrueQ @ AbsoluteCurrentValue[cell, CellAutoOverwrite], AppendTo[objs, cell], Break[]],
			{cell, cells}
		];
		objs
	]


deletePreviousOutputs[cellobj_, nbobj_] :=
	Replace[previousOutputs[cellobj, nbobj], {
		cells: {__CellObject} :> NotebookDelete[cells],
		_ :> None
	}]

moveAfterPreviousOutputs[cellobj_, nbobj_] := 
	Replace[previousOutputs[cellobj, nbobj], {
		{___, lastcell_CellObject} :> SelectionMove[lastcell, After, Cell],
		_ :> SelectionMove[cellobj, After, Cell]
	}]


writeFailureInput[tag: ("Offline" | "NetworkDisabled" | "NoParse" | "NoStepByStep")] := 
	CellPrint @ Cell[BoxData[ToBoxes[
		Grid[{{
			Switch[tag,
				"Offline", bitmapResource["NetworkFailureVector", BaselinePosition -> Scaled[0.1]],
				"NetworkDisabled", bitmapResource["NetworkDisabledVector", BaselinePosition -> Scaled[0.1]],
				"NoParse", bitmapResource["WAErrorIndicator"],
				"NoStepByStep", bitmapResource["WAErrorIndicator"]
			],
			Style[cachedMessageTemplate["WAStrings", "MessageTemplate:" <> tag], Deployed -> False]
			}},
			Alignment -> Left
		]]],
		"DeployedWLInput",
		"NaturalLanguageFailure"
	]


writeNewWolframLanguageInput[cellobj_, nbobj_, Defer[expr_], label_] := 
	Module[{boxes, obj, help},
		Which[
			MatchQ[Defer[expr], Defer[_WolframAlphaResult]],
				boxes = ToBoxes[Iconize[
					Unevaluated[expr],
					Tooltip[
						stringResource["WolframAlphaResultIconizeLabel"],
						ToString[Unevaluated[expr], InputForm]
					]
				]];
				help = None
			,
			(* otherwise, we have a real parse *)
			True,
				boxes = AlphaQueryMakeBoxes[expr, "MostlyInputForm"];
				help = constructHelpCell[Defer[expr]];
		];
		obj = cellPrintReturnObject @ Cell[BoxData[boxes],
			"DeployedWLInput",
			If[TrueQ[label],
				(* Fake the CellLabel, since this isn't actually the cell that was evaluated to get the output *)
				CellLabel -> ("In[" <> ToString[$Line] <>"]:="),
				Unevaluated[Sequence[]]
			]
		];
		If[Head[help] === Cell, attachHelpCell[obj, help]];
		obj
	]


(* ::Section::Closed:: *)
(*AlphaIntegration`CreateWolframAlphaNotebook*)


WolframAlphaNotebookPut[Notebook[cells_, opts___]] := 
	NotebookPut[Notebook[cells, opts, StyleDefinitions -> "WolframAlphaNotebook.nb"]]	


AlphaIntegration`CreateWolframAlphaNotebook[] := 
	Module[{nbobj},
		nbobj = WolframAlphaNotebookPut[Notebook[{Cell["", "NaturalLanguageInput"]}]];
		(* Move the insertion point into the first NLInput cell *)
		Replace[
			Cells[nbobj, CellStyle -> "NaturalLanguageInput"],
			{cellobj_, ___} :> SelectionMove[cellobj, After, CellContents]
		];
		nbobj
	]


(* ::Section::Closed:: *)
(*Example page support*)


showExamplesAsWolframAlphaNotebook[url_String] := 
	showExamplesAsWolframAlphaNotebook[Import[StringReplace[url, "-content.html" -> ".xml"], "XML"]]


showExamplesAsWolframAlphaNotebook[xml_] :=
	Module[{examples},
		examples = Flatten[Cases[xml, XMLElement["example", _, a_] :> a, Infinity]];
		examples = wanbExampleCell /@ examples;
		WolframAlphaNotebookPut[Notebook[examples]]
	]


wanbExampleCell[XMLElement["category", _, {a_String}]] := Cell[a, "Title"]
wanbExampleCell[XMLElement["section-title", _, {a_String}]] := Cell[a, "Section"]
wanbExampleCell[XMLElement["section-title", _, {XMLElement["link", _, {a_String}]}]] := Cell[a, "Section"]
wanbExampleCell[XMLElement["caption", _, {a_String}]] := Cell[a, "Text"]
wanbExampleCell[XMLElement["input", _, {a_String}]] := Cell[a, "NaturalLanguageInput"]


(* ::Section::Closed:: *)
(*Utilities*)


(* There's a packet for doing this in the 12.0 front end *)
cellPrintReturnObject[c_Cell] := 
	MathLink`CallFrontEnd[FrontEnd`CellPrintReturnObject[c]] /; BoxForm`sufficientVersionQ[12.0]

(* Otherwise, we have to fiddle with the selection *)
cellPrintReturnObject[c_Cell] :=
	Module[{before, after, nb, obj},
		nb = EvaluationNotebook[];
		before = Cells[nb];
		CellPrint[c];
		after = Cells[nb];
		Replace[Complement[after, before], {{one_CellObject} :> one, _ :> $Failed}]
	]


cellPrintSelectAndEvaluate[c_Cell] := 
	Replace[cellPrintReturnObject[c], {
		obj_CellObject :> (
			CurrentValue[obj, GeneratedCell] = Inherited;
			CurrentValue[obj, CellAutoOverwrite] = Inherited;
			SelectionMove[obj, All, Expression];
			FrontEndTokenExecute["HandleShiftReturn"];
			(*SelectionMove[obj, After, Cell];*)
		),
		_ :> $Failed
	}]


notebookWriteReturnObject[nbobj_NotebookObject, args__] := 
	Module[{before, after, obj},
		before = Cells[nbobj];
		NotebookWrite[nbobj, args];
		after = Cells[nbobj];
		(* FIXME: Obviously this fails if the NotebookWrite doesn't result in a new cell. *)
		Replace[Complement[after, before], {{one_CellObject} :> one, _ :> $Failed}]
	]


(* 
The AlphaIntegration`DuplicatePreviousCell utility maps to Copy Output
From Above in a W|A notebook. Copy Input from Above is handled
directly by the front end, starting in version 12.
*)
AlphaIntegration`DuplicatePreviousCell[nbobj_NotebookObject, style: "Output"] := 
	Module[{styles, cellobj, proxyobj},
		styles = {"Output"};
		
		cellobj = PreviousCell[EvaluationCell[], CellStyle -> styles];
		If[!MatchQ[cellobj, _CellObject],
			(* if the insertion point is between cells, add a proxy input cell from which to look for previous ones *)
			proxyobj = notebookWriteReturnObject[nbobj, Cell["", "Input"], All];
			cellobj = PreviousCell[proxyobj, CellStyle -> styles]
		];

		Replace[cellobj, {
			obj_CellObject :> NotebookWrite[
				nbobj,
				(* regardless of the style of the original cell, the copied down cell will be "Input" *)
				Replace[NotebookRead[obj], Cell[a_, Alternatives @@ styles, opts___] :>
					Cell[a, "Input", Sequence @@ DeleteCases[Flatten[{opts}], _[CellLabel, _]]]],
				After
			],
			_ :> (Quiet[NotebookDelete[proxyobj]]; Beep[])
		}]
	]



(* ::Section::Closed:: *)
(*Package footer*)


End[];
