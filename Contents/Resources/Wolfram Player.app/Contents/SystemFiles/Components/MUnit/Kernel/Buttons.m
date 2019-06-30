

Begin["`Package`"]

jumpToTestButtonBox

jumpToFatalButtonBox

replaceOutputButtonBox

replaceMessagesButtonBox

compareMessagesButtonBox

compareOutputButtonBox

rerunTestButtonBox

End[]

Begin["`Buttons`Private`"]

jumpToTestButtonBox[nb_, tr_?TestResultQ] :=
	ButtonBox["\<\"Jump to Test\"\>",
		ButtonFunction:>jumpToTestButtonFunction[nb, tr],
		Appearance->Automatic,
		ButtonFrame->"DialogBox",
		Evaluator->Automatic,
		Method->"Preemptive"
	]

jumpToFatalButtonBox[nb_, rules_] :=
	Module[{fatalID = "CellID" /. rules},
		ButtonBox["\<\"Jump to Cause of Fatal\"\>",
			ButtonFunction:>jumpToFatalButtonFunction[nb, rules],
			Appearance->Automatic,
			ButtonFrame->"DialogBox",
			Evaluator->Automatic,
			Method->"Preemptive",
			Enabled -> !(fatalID == "CellID")
		]
	]

(*
nbStale references the (possibly stale) main link, but jumpToTestButtonFunction is being run
on the preemptive link
*)
jumpToTestButtonFunction[nbStale_NotebookObject, tr_?TestResultQ] :=
	Module[{testID = ToExpression["36^^" <> StringTake[TestID[tr], -7]], found, nb},
		Catch[
			(*
			nb is the nb object, but referenced on the preemptive link
			here, $FrontEnd evaluates using the preemptive link
			*)
			nb = nbStale /. _FrontEndObject -> $FrontEnd;
			If[!MemberQ[Notebooks[], nb],
				Throw[{"Value" -> nb, "Messages" -> {"Test Cell", "Invalid NotebookObject"}}, MUnitErrorTag]
			];
			SetSelectedNotebook[nb];
			found = NotebookFind[nb, testID, All, CellID];
			If[found === $Failed,
				Throw[{"Value" -> HoldForm[NotebookFind][nb, testID, All, CellID], "Messages" -> {"Test Cell", "Cannot find CellID. NotebookFind returned $Failed"}}, MUnitErrorTag]
			]
			,
			MUnitErrorTag
			,
			Function[{rules, tag},
				Module[{msgs = "Messages" /. rules, val = "Value" /. rules},
					MessageDialog[Column[{msgs[[1]], Style[msgs[[2]], Bold], val}], WindowTitle -> "MUnit Error"]
				]
			]	
		]
	]

jumpToTestButtonFunction[nbFileName_?noteBookFileQ, tr_?TestResultQ] :=
	Module[{testID = ToExpression["36^^" <> StringTake[TestID[tr], -7]], found, nb, nbs},
		Catch[
			nbs = Notebooks[nbFileName];
			If[nbs == {},
				nb = NotebookOpen[nbFileName];
				,
				nb = nbs[[1]];
			];
			SetSelectedNotebook[nb];
			found = NotebookFind[nb, testID, All, CellID];
			If[found === $Failed,
				Throw[{"Value" -> HoldForm[NotebookFind][nb, testID, All, CellID], "Messages" -> {"Test Cell", "Cannot find CellID. NotebookFind returned $Failed"}}, MUnitErrorTag]
			]
			,
			MUnitErrorTag
			,
			Function[{rules, tag},
				Module[{msgs = "Messages" /. rules, val = "Value" /. rules},
					MessageDialog[Column[{msgs[[1]], Style[msgs[[2]], Bold], val}], WindowTitle -> "MUnit Error"]
				]
			]	
		]
	]

jumpToTestButtonFunction[_, tr_] :=
	Module[{},
		Catch[
			Throw[{"Value" -> tr, "Messages" -> {"Test Cell", "Invalid TestResultObject"}}, MUnitErrorTag]
			,
			MUnitErrorTag
			,
			Function[{rules, tag},
				Module[{msgs = "Messages" /. rules, val = "Value" /. rules},
					MessageDialog[Column[{msgs[[1]], Style[msgs[[2]], Bold], val}], WindowTitle -> "MUnit Error"]
				]
			]
		]
	]

(*
precondition: "CellID" prop in info has a value
*)
jumpToFatalButtonFunction[nbStale_NotebookObject, info_] :=
	Module[{fatalID = "CellID" /. info, found, nb},
		Catch[
			(*
			nb is the nb object, but referenced on the preemptive link
			here, $FrontEnd evaluates using the preemptive link
			*)
			nb = nbStale /. _FrontEndObject -> $FrontEnd;
			If[!MemberQ[Notebooks[], nb],
				Throw[{"Value" -> nb, "Messages" -> {"Fatal Error", "Invalid NotebookObject"}}, MUnitErrorTag]
			];
			SetSelectedNotebook[nb];
			found = NotebookFind[nb, fatalID, All, CellID];
			If[found === $Failed,
				Throw[{"Value" -> HoldForm[NotebookFind][nb, fatalID, All, CellID], "Messages" -> {"Fatal Error", "Cannot find CellID. NotebookFind returned $Failed"}}, MUnitErrorTag]
			]
			,
			MUnitErrorTag
			,
			Function[{rules, tag},
				Module[{msgs = "Messages" /. rules, val = "Value" /. rules},
					MessageDialog[Column[{msgs[[1]], Style[msgs[[2]], Bold], val}], WindowTitle -> "MUnit Error"]
				]
			]	
		]
	]

replaceOutputButtonBox[nb_NotebookObject, tr_?TestResultQ] :=
	ButtonBox["\<\"Replace ExpectedOutput with ActualOutput\"\>",
		ButtonFunction :> replaceOutputButtonFunction[nb, tr],							
		Appearance->Automatic,
		ButtonFrame->"DialogBox",
		Evaluator->Automatic,
		Method->"Queued"
	]

replaceOutputButtonFunction[nbStale_NotebookObject, tr_?TestResultQ] :=
	Module[{testID = ToExpression["36^^" <> StringTake[TestID[tr], -7]], info, grouped, read, cases, found, nb},
		Catch[
			nb = nbStale /. _FrontEndObject -> $FrontEnd;
			If[!MemberQ[Notebooks[], nb],
				Throw[{"Value" -> nb, "Messages" -> {"Test Cell", "Invalid NotebookObject"}}, 	MUnitErrorTag]
			];
			(*SetSelectedNotebook[nbMain];*)
			found = NotebookFind[nb, testID, All, CellID];
			If[found === $Failed,
				Throw[{"Value" -> HoldForm[NotebookFind][nb, testID, All, CellID], "Messages" -> {"Test Cell", "Cannot find CellID. NotebookFind returned $Failed"}}, MUnitErrorTag]
			];
			info = Developer`CellInformation[nb];
			grouped = "FirstCellInGroup" /. info[[1]];
			If[grouped,
				SelectionMove[nb, All, CellGroup];
				(* if  grouped, then have to dig around a little *)
				read = NotebookRead[nb];
				cases = Cases[read[[1, 1]], Cell[_, "ExpectedOutput", ___, CellID -> cellID_, ___] :> cellID];
				If[cases == {},
					(* no expected output cell, do something else *)
					cases = Cases[read[[1, 1]], Cell[_, "ExpectedMessages", ___, CellID -> cellID_, ___] :> cellID];
					If[cases == {},
						cases = Cases[read[[1, 1]], Cell[_, "ExpectedMessage", ___, CellID -> cellID_, ___] :> cellID];
						If[cases == {},
							(* no expected messages, do something else *)
							cases = Cases[read[[1, 1]], Cell[_, "TestOptions", ___, CellID -> cellID_, ___] :> cellID];
							If[cases == {},
								(* no test options, insert after test cell *)
								NotebookFind[nb, testID, All, CellID];
								SelectionMove[nb, After, Cell]
								,
								(* insert after test options cell *)
								NotebookFind[nb, cases[[1]], All, CellID];
								SelectionMove[nb, After, Cell]
							]
							,
							(* there are "ExpectedMessage" cells *)
							(* find the first expected message, and iterate until all expected message cells are selected *)
							NotebookFind[nb, cases[[1]], All, CellID];
							Do[FrontEndExecute[FrontEndToken[nb, "SelectNextCell"]], {Length[cases]-1}]
						]
						,
						(* insert after expected messages cell *)
						NotebookFind[nb, cases[[1]], All, CellID];
						SelectionMove[nb, After, Cell]
					]
					,
					(* expected output cell, select it to overwrite *)
					NotebookFind[nb, cases[[1]], All, CellID]
				]
				,
				(* only a test cell, so just insert expected output after *)
				SelectionMove[nb, After, Cell]
			];
			
			(*
			the reason for the InterpretationBox[boxes, x]:
			the front end takes certain liberties with boxes written to notebooks (even if using NotebookWrite)
			in an attempt to optimize expressions like graphics.
			So, it is not correct to think that doing a NotebookWrite then NotebookRead may necessarily give you the
			same box structure.
			Because of this, we need to do something like Interpretation[boxes, e]
			*)
			With[{expr = ActualOutput[tr]},
				With[{boxes = Function[{x}, ToBoxes[Unevaluated[x]], {HoldFirst}] @@ expr},
					NotebookWrite[nb, Cell[BoxData[Function[{x}, InterpretationBox[boxes, x], {HoldFirst}] @@ expr], "ExpectedOutput"]]
				]
			];
			
			(* place cursor after cellgroup *)
			SelectionMove[nb, All, CellGroup];
			SelectionMove[nb, After, CellGroup];
			,
			MUnitErrorTag
			,
			Function[{rules, tag},
				Module[{msgs = "Messages" /. rules, val = "Value" /. rules},
					MessageDialog[Column[{msgs[[1]], Style[msgs[[2]], Bold], val}], WindowTitle -> "MUnit Error"]
				]
			]
		]
	]

replaceOutputButtonFunction[nbStale_NotebookObject, tr_] :=
	Module[{},
		Catch[
			Throw[{"Value" -> tr, "Messages" -> {"Test Cell", "Invalid TestResultObject"}}, MUnitErrorTag]
			,
			MUnitErrorTag
			,
			Function[{rules, tag},
				Module[{msgs = "Messages" /. rules, val = "Value" /. rules},
					MessageDialog[Column[{msgs[[1]], Style[msgs[[2]], Bold], val}], WindowTitle -> "MUnit Error"]
				]
			]
		]
	]

replaceMessagesButtonBox[nb_NotebookObject, tr_?TestResultQ] :=
	ButtonBox["\<\"Replace ExpectedMessages with ActualMessages\"\>",
		ButtonFunction :> replaceMessagesButtonFunction[nb, tr],
		Appearance->Automatic,
		ButtonFrame->"DialogBox",
		Evaluator->Automatic,
		Method->"Queued"
	]

replaceMessagesButtonFunction[nbStale_NotebookObject, tr_?TestResultQ] :=
	Module[{testID = ToExpression["36^^" <> StringTake[TestID[tr], -7]], info, grouped, read, cases, found, nb},
		Catch[
			nb = nbStale /. _FrontEndObject -> $FrontEnd;
			If[!MemberQ[Notebooks[], nb],
				Throw[{"Value" -> nb, "Messages" -> {"Test Cell", "Invalid NotebookObject"}}, MUnitErrorTag]
			];
			(*SetSelectedNotebook[nbMain];*)
			found = NotebookFind[nb, testID, All, CellID];
			If[found === $Failed,
				Throw[{"Value" -> HoldForm[NotebookFind][nb, testID, All, CellID], "Messages" -> {"Test Cell", "Cannot find CellID. NotebookFind returned $Failed"}}, MUnitErrorTag]
			];
			info = Developer`CellInformation[nb];
			grouped = "FirstCellInGroup" /. info[[1]];
			If[grouped,
				SelectionMove[nb, All, CellGroup];
				(* if grouped, then have to dig around a little *)
				read = NotebookRead[nb];
				cases = Cases[read[[1, 1]], Cell[_, "ExpectedMessages", ___, CellID -> cellID_, ___] :> cellID];
				If[cases == {},
					(* possibly there are "ExpectedMessage" cells *)
					cases = Cases[read[[1, 1]], Cell[_, "ExpectedMessage", ___, CellID -> cellID_, ___] :> cellID];
					If[cases == {},
						(* no expected messages cell or expected message cells, do something else *)
						cases = Cases[read[[1, 1]], Cell[_, "TestOptions", ___, CellID -> cellID_, ___] :> cellID];
						If[cases == {},
							(* no test options, insert after test cell *)
							NotebookFind[nb, testID, All, CellID];
							SelectionMove[nb, After, Cell]
							,
							(* insert after test options cell *)
							NotebookFind[nb, cases[[1]], All, CellID];
							SelectionMove[nb, After, Cell]
						]
						,
						(* there are "ExpectedMessage" cells *)
						(* find the first expected message, and iterate until all expected message cells are selected *)
						NotebookFind[nb, cases[[1]], All, CellID];
						Do[FrontEndExecute[FrontEndToken[nb, "SelectNextCell"]], {Length[cases]-1}]
					]
					,
					(* expected messages cell, select it to overwrite *)
					NotebookFind[nb, cases[[1]], All, CellID]
				]
				,
				(* only a test cell, so just insert expected messages after *)
				SelectionMove[nb, After, Cell]
			];
			
			NotebookWrite[nb, Cell[BoxData[ToBoxes[ActualMessages[tr]] //. TagBox[msg_, HoldForm] :> msg], "ExpectedMessages"]];
			
			(* palce cursor after cellgroup *)
			SelectionMove[nb, All, CellGroup];
			SelectionMove[nb, After, CellGroup];
			,
			MUnitErrorTag
			,
			Function[{rules, tag},
				Module[{msgs = "Messages" /. rules, val = "Value" /. rules},
					MessageDialog[Column[{msgs[[1]], Style[msgs[[2]], Bold], val}], WindowTitle -> "MUnit Error"]
				]
			]
		]
	]

replaceMessagesButtonFunction[nbStale_NotebookObject, tr_] :=
	Module[{},
		Catch[
			Throw[{"Value" -> tr, "Messages" -> {"Test Cell", "Invalid TestResultObject"}}, MUnitErrorTag]
			,
			MUnitErrorTag
			,
			Function[{rules, tag},
				Module[{msgs = "Messages" /. rules, val = "Value" /. rules},
					MessageDialog[Column[{msgs[[1]], Style[msgs[[2]], Bold], val}], WindowTitle -> "MUnit Error"]
				]
			]
		]
	]

compareOutputButtonBox[nb:(_NotebookObject|Null), tr_?TestResultQ] :=
	ButtonBox["\<\"Compare ExpectedOutput with ActualOutput\"\>",
		ButtonFunction :> compareOutputButtonFunction[nb, tr],							
		Appearance->Automatic,
		ButtonFrame->"DialogBox",
		Evaluator->Automatic,
		Method->"Queued"
	]

compareOutputButtonFunction[nbStale:(_NotebookObject|Null), tr_?TestResultQ] :=
	Module[{nb, e, a},
		Catch[
			If[MatchQ[nbStale, _NotebookObejct],
				nb = nbStale /. _FrontEndObject -> $FrontEnd;
				If[!MemberQ[Notebooks[], nb],
					Throw[{"Value" -> nb, "Messages" -> {"Test Cell", "Invalid NotebookObject"}}, 	MUnitErrorTag]
				];
			];
			e = ToString[ExpectedOutput[tr], InputForm];
			a = ToString[ActualOutput[tr], InputForm];
			If[a == e,
				(*
				try using FullForm
				*)
				e = ToString[FullForm[ExpectedOutput[tr]]];
				a = ToString[FullForm[ActualOutput[tr]]];
			];
			e = StringTake[e, {10, -2}];
			a = StringTake[a, {10, -2}];
			CreateDialog[{Grid[{
				{"Expected:", Rasterize@e},
				{"Actual:", Rasterize@a}}, Alignment->Left], DefaultButton[]}, WindowTitle->"Compare ExpectedOutput with ActualOutput"]
			,
			MUnitErrorTag
			,
			Function[{rules, tag},
				Module[{msgs = "Messages" /. rules, val = "Value" /. rules},
					MessageDialog[Column[{msgs[[1]], Style[msgs[[2]], Bold], val}], WindowTitle -> "MUnit Error"]
				]
			]
		]
	]

compareOutputButtonFunction[nbStale_NotebookObject, tr_] :=
	Module[{},
		Catch[
			Throw[{"Value" -> tr, "Messages" -> {"Test Cell", "Invalid TestResultObject"}}, MUnitErrorTag]
			,
			MUnitErrorTag
			,
			Function[{rules, tag},
				Module[{msgs = "Messages" /. rules, val = "Value" /. rules},
					MessageDialog[Column[{msgs[[1]], Style[msgs[[2]], Bold], val}], WindowTitle -> "MUnit Error"]
				]
			]
		]
	]

compareMessagesButtonBox[nb:(_NotebookObject|Null), tr_?TestResultQ] :=
	ButtonBox["\<\"Compare ExpectedMessages with ActualMessages\"\>",
		ButtonFunction :> compareMessagesButtonFunction[nb, tr],							
		Appearance->Automatic,
		ButtonFrame->"DialogBox",
		Evaluator->Automatic,
		Method->"Queued"
	]

compareMessagesButtonFunction[nbStale:(_NotebookObject|Null), tr_?TestResultQ] :=
	Module[{nb, e, a},
		Catch[
			If[MatchQ[nbStale, _NotebookObejct],
				nb = nbStale /. _FrontEndObject -> $FrontEnd;
				If[!MemberQ[Notebooks[], nb],
					Throw[{"Value" -> nb, "Messages" -> {"Test Cell", "Invalid NotebookObject"}}, 	MUnitErrorTag]
				];
			];
			e = ToString[#, InputForm]& /@ ExpectedMessages[tr];
			a = ToString[#, InputForm]& /@ ActualMessages[tr];
			If[a == e,
				(*
				try using FullForm
				*)
				e = ToString[FullForm[#]]& /@ ExpectedMessages[tr];
				a = ToString[FullForm[#]]& /@ ActualMessages[tr];
			];
			e = StringTake[#, {10, -2}]& /@ e;
			a = StringTake[#, {10, -2}]& /@ a;
			CreateDialog[{Grid[{
				{"Expected:", Rasterize@e},
				{"Actual:", Rasterize@a}}, Alignment->Left], DefaultButton[]}, WindowTitle->"Compare ExpectedMessages with ActualMessages"]
			,
			MUnitErrorTag
			,
			Function[{rules, tag},
				Module[{msgs = "Messages" /. rules, val = "Value" /. rules},
					MessageDialog[Column[{msgs[[1]], Style[msgs[[2]], Bold], val}], WindowTitle -> "MUnit Error"]
				]
			]
		]
	]

compareMessagesButtonFunction[nbStale_NotebookObject, tr_] :=
	Module[{},
		Catch[
			Throw[{"Value" -> tr, "Messages" -> {"Test Cell", "Invalid TestResultObject"}}, MUnitErrorTag]
			,
			MUnitErrorTag
			,
			Function[{rules, tag},
				Module[{msgs = "Messages" /. rules, val = "Value" /. rules},
					MessageDialog[Column[{msgs[[1]], Style[msgs[[2]], Bold], val}], WindowTitle -> "MUnit Error"]
				]
			]
		]
	]

rerunTestButtonBox[nb_NotebookObject, tr_?TestResultQ] :=
	With[{testid = TestID[tr]},
		ButtonBox["\<\"Rerun Test\"\>",
			ButtonFunction :> rerunTestButtonFunction[nb, testid],
			Appearance->Automatic,
			ButtonFrame->"DialogBox",
			Evaluator->Automatic,
			Method->"Queued"
		]
	]

rerunTestButtonFunction[nbStale_NotebookObject, testid_String] :=
	rerunTestButtonFunction[nbStale, ToExpression["36^^" <> StringTake[testid, -7]]]
	
rerunTestButtonFunction[nbStale_NotebookObject, testid_Integer] :=
	Module[{found, nb},
		Catch[
			nb = nbStale /. _FrontEndObject -> $FrontEnd;
			If[!MemberQ[Notebooks[], nb],
				Throw[{"Value" -> nb, "Messages" -> {"Test Cell", "Invalid NotebookObject"}}, MUnitErrorTag]
			];
			If[testid == 0,
				Throw[{"Value" -> testid, "Messages" -> {"Test Cell", "Invalid TestID"}}, MUnitErrorTag]
			];
			found = NotebookFind[nb, testid, All, CellID];
			
			If[found === $Failed,
				Throw[{"Value" -> HoldForm[NotebookFind][nb, testid, All, CellID], "Messages" -> {"Test Cell", "Cannot find CellID. NotebookFind returned $Failed"}}, MUnitErrorTag]
			];
			(* setup precondition for cellEvaluationFunction *)
			SelectionMove[nb, All, CellGroup];
			If[Developer`CellInformation[nb] === $Failed,
				(* cells are not grouped with a Test cell *)
				SelectionMove[nb, After, EvaluationCell];
				Throw[{"Value" -> Null, "Messages" -> {"Test", "Malformed Test CellGroup"}}, MUnitErrorTag]
			];
			
			(* we know grouped is True because this is being run from a TestResult cell *)
			cellEvaluationFunction[nb, True]
			,
			MUnitErrorTag
			,
			Function[{rules, tag},
				Module[{msgs = "Messages" /. rules, val = "Value" /. rules},
					MessageDialog[Column[{msgs[[1]], Style[msgs[[2]], Bold], val}], WindowTitle -> "MUnit Error"]
				]
			]
		]
	]

End[]
