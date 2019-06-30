(*======WebSessionObject========*)
webSessionObjectAscQ[asc_?AssociationQ] := AllTrue[{"SessionID", "Browser", "Process", "URL", "Exec"}, KeyExistsQ[asc, #]&]
webSessionObjectAscQ[_] = False;

System`WebSessionObject /: MakeBoxes[obj : System`WebSessionObject[asc_? webSessionObjectAscQ], form : (StandardForm | TraditionalForm)] :=
		Module[{above, below},
		    above = { (* example grid *)
		      {
		        BoxForm`SummaryItem[{"Browser: ", asc["Browser"]}]
        },
		      {
		        BoxForm`SummaryItem[{"Session ID: ", asc["SessionID"]}], SpanFromLeft
        }
      };
      below = { (* example column *)
		      BoxForm`SummaryItem[{"Driver URL: ", asc["URL"]}],
		      BoxForm`SummaryItem[{"Driver Process: ", asc["Process"]}]
      (*BoxForm`SummaryItem[{"Exec: ", asc["Exec"]}],*)
        (*BoxForm`SummaryItem[{"Active: ", asc["Active"]}]*)
      };
        BoxForm`ArrangeSummaryBox[
          System`WebSessionObject, (* head *)
          obj, (* interpretation *)
          $WebDriverIcon, (* icon, use None if not needed *)
        (* above and below must be in a format suitable for Grid or Column *)
          above, (* always shown content *)
          below, (* expandable content *)
          form,
          "Interpretable" -> Automatic
        ]
    ];

WebSessionObject[assoc_Association][key_] := assoc[key];
WebSessionObject[assoc_Association]["Properties"] := {"Browser","SessionID","URL","Process"};

WebSessionObject /: HoldPattern[DeleteObject][wdSessionObject:WebSessionObject[sessionArg_]] := Block[
  {
    sessionLinkUUID = sessionArg["SessionID"]
  },
  If[!TrueQ[$Link[sessionLinkUUID]["Active"]],Message[WebExecute::inactiveSession,wdSessionObject];Return[$Failed]];
		If[MemberQ[Keys[$Link],sessionLinkUUID] && TrueQ[$Link[sessionLinkUUID]["Active"]],
			(
      StopWebSession[wdSessionObject[[1]]];
      KillProcess[$Link[sessionLinkUUID,"Process"]];
      (*$Links[sessionUUID,"StopTime"] = AbsoluteTime[];*)
      $Link[sessionLinkUUID,"Active"] = False;
			),
			(
				(*TODO: System`WebUnit::invalidSession to be associated? *)
				Message[WebExecute::invalidSession,wdSessionObject];
				Return[$Failed]
      )
    ]
]


(*======WebElementObject========*)

webElementObjectAscQ[asc_?AssociationQ] := AllTrue[{"SessionID", "Browser", "Process", "URL", "Exec", "ElementId"}, KeyExistsQ[asc, #]&]
webElementObjectAscQ[_] = False;



System`WebElementObject /: MakeBoxes[obj : System`WebElementObject[asc_? webElementObjectAscQ], form : (StandardForm | TraditionalForm)] :=
    Module[{above, below},
      above = { (* example grid *)
        {
          BoxForm`SummaryItem[{"Browser: ", asc["Browser"]}], SpanFromLeft
          (*BoxForm`SummaryItem[{"SessionId: ", asc["SessionID"]}], SpanFromLeft*)
        },
        {
          BoxForm`SummaryItem[{"ElementId: ", asc["ElementId"]}]
        }
      };
      below = { (* example column *)
        BoxForm`SummaryItem[{"Driver URL: ", asc["URL"]}],
        BoxForm`SummaryItem[{"Driver Process: ", asc["Process"]}]
        (*, BoxForm`SummaryItem[{"Exec: ", asc["Exec"]}]*)
      };
      BoxForm`ArrangeSummaryBox[
        System`WebElementObject, (* head *)
        obj, (* interpretation *)
        $WebDriverIcon, (* icon, use None if not needed *)
      (* above and below must be in a format suitable for Grid or Column *)
        above, (* always shown content *)
        below, (* expandable content *)
        form,
        "Interpretable" -> Automatic
      ]
    ];

WebElementObject[assoc_Association][key_] := assoc[key];
WebElementObject[assoc_Association]["Properties"] := {"Browser","SessionID","URL","Process","ElementId"};
(*======WindowObject========*)

webWindowObjectAscQ[asc_?AssociationQ] := AllTrue[{"SessionID", "Browser", "Process", "URL", "Exec", "WindowID"}, KeyExistsQ[asc, #]&]
webWindowObjectAscQ[_] = False;



System`WebWindowObject /: MakeBoxes[obj : System`WebWindowObject[asc_? webWindowObjectAscQ], form : (StandardForm | TraditionalForm)] :=
    Module[{above, below},
      above = { (* example grid *)
        {
          BoxForm`SummaryItem[{"Browser: ", asc["Browser"]}], SpanFromLeft
          (*BoxForm`SummaryItem[{"SessionId: ", asc["SessionID"]}], SpanFromLeft*)
        },
        {
          BoxForm`SummaryItem[{"WindowID: ", asc["WindowID"]}]
        }
      };
      below = { (* example column *)
        BoxForm`SummaryItem[{"Driver URL: ", asc["URL"]}],
        BoxForm`SummaryItem[{"Driver Process: ", asc["Process"]}]
        (*, BoxForm`SummaryItem[{"Exec: ", asc["Exec"]}]*)
      };
      BoxForm`ArrangeSummaryBox[
        System`WebWindowObject, (* head *)
        obj, (* interpretation *)
        $WebDriverIcon, (* icon, use None if not needed *)
      (* above and below must be in a format suitable for Grid or Column *)
        above, (* always shown content *)
        below, (* expandable content *)
        form,
        "Interpretable" -> Automatic
      ]
    ];

WebWindowObject[assoc_Association][key_] := assoc[key];
WebWindowObject[assoc_Association]["Properties"] := {"Browser","SessionID","URL","Process","WindowID"};

windowFromAssoc [wdSessionObject_, windowID_]:=
      WebWindowObject[
        <|
          "SessionID"-> wdSessionObject[[1]]["SessionID"],
          "Browser"->   wdSessionObject[[1]]["Browser"],
          "Process"->   wdSessionObject[[1]]["Process"],
          "URL"->       wdSessionObject[[1]]["URL"],
          "Exec"->      wdSessionObject[[1]]["Exec"],
          "WindowID"-> windowID
          |>
      ];

