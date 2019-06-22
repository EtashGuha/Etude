$Link = <||>;
$CurrentWebSession = None;
wdMakeLink[driver_?StringQ, methodOpts_] := Block[{linkUUID, session},

  (*connects with the driver and starts the session with suggested options*)
  session = wdStartWebSession[driver, methodOpts];
  If[!AssociationQ[session],
    Return[Failure["StartWebSession",  <|
      "MessageTemplate" -> "Unable to start `driver` driver process",
      "MessageParameters" -> <|"driver" -> driver|>
    |>]]
  ];
  linkUUID = session["SessionID"];
  $Link[linkUUID] = session;
  WebSessionObject[session]
];


System`WebSessions[] := Block[{activeAssocs, validLink},
  validLink = Select[$Link, !FailureQ[#] &];
  activeAssocs = $Link[#] & /@ Keys@Select[#Active &]@ validLink;
  WebSessionObject /@ activeAssocs
];

System`$CurrentWebSession := Block[{activeAssocs, validLink, lastActiveAssoc},
  validLink = Select[$Link, !FailureQ[#] &];
  activeAssocs = Select[#Active &]@validLink;
  If[activeAssocs === <||>, Return[None], lastActiveAssoc = Last@SortBy[#SessionTime &]@activeAssocs];
  WebSessionObject[lastActiveAssoc]
];

(*This should also work*)
(*$CurrentWebSession := Last[$WebSessions];*)

StartWebSession[opts:OptionsPattern[]] := StartWebSession[Automatic, opts];

StartWebSession[Automatic, opts:OptionsPattern[]] := Block[{browser},
  browser = chooseBrowser[];
  (* For now, on $Failed, try chrome anyway until we know chooseBrowser is robust. *)
  If[!StringQ[browser],
    browser = "Chrome"
  ];
  StartWebSession[browser, opts]
];


Options[StartWebSession]= {Visible -> True} ;
StartWebSession[driver_?StringQ, opts:OptionsPattern[]]:= Block[{selectedDriver, selectedMethod, optVisible,checkOpts},
  checkOpts = If[{opts}==={},  Options[StartWebSession],
    If[ListQ[opts], opts, {opts}]
  ];

  If[!MatchQ[checkOpts,{Visible -> ___ }],Message[StartWebSession::invalidmethod, opts]; Return[$Failed]];
  optVisible = OptionValue[Visible];
  If[ (driver === "Chrome" || driver === "Firefox"), selectedDriver = driver, (Message[StartWebSession::invalidbrowser, driver]; Return[$Failed])];
  If[ BooleanQ[optVisible]  ||  MatchQ[optVisible, "DisableImage"],
    wdMakeLink[selectedDriver, {Visible -> optVisible}],
    (Message[StartWebSession::invalidmethod, opts]; Return[$Failed])]
];

(*======================WebExecute=================*)
webExecuteArgQ[s_] := (MatchQ[s, None] || MatchQ[s, WebSessionObject[_]]);
System`WebExecute[wdSessionObject_?webExecuteArgQ, func_String -> arg_] := WebExecute[wdSessionObject, <|"Command" -> func, "Arguments" -> arg|>];
WebExecute[func_String -> arg_] := WebExecute[$CurrentWebSession, <|"Command" -> func, "Arguments" -> arg|>];

(*"cooked" form of the assoc - we just turn it into the assoc*)
WebExecute[wdSessionObject_?webExecuteArgQ, func_String -> args : {___}] := WebExecute[wdSessionObject, <|"Command" -> func, "Arguments" -> args|>];
WebExecute[ func_String -> args : {___}] := WebExecute[$CurrentWebSession, <|"Command" -> func, "Arguments" -> args|>];

WebExecute[wdSessionObject_?webExecuteArgQ, func_?StringQ] := WebExecute[wdSessionObject, <|"Command" -> func, "Arguments" -> None|>];
WebExecute[ func_?StringQ] := WebExecute[$CurrentWebSession, <|"Command" -> func, "Arguments" -> None|>];

WebExecute[wdSessionObject_WebSessionObject, input_?ListQ] := WebExecute[wdSessionObject, #]& /@ input;
(*WebExecute[input_?ListQ] := WebExecute[$CurrentWebSession, #]& /@ input;*)
WebExecute[input_?ListQ] := WebExecute[$CurrentWebSession, input];

WebExecute[input_?AssociationQ] := WebExecute[$CurrentWebSession, input];
WebExecute[wdSessionObject_WebSessionObject, input_?AssociationQ] := Block[
  {
    res,
    link,
    sessionUUID = $Link[[1]],
    returnInput,
    lenAssoc,
    tempArg
  },
  (
  If[!MemberQ[WebSessions[], wdSessionObject], Message[WebExecute::inactiveSession,wdSessionObject]; Return[$Failed]];
  (*TODO: Active link session check*)
    If[KeyExistsQ[wdSessionObject[[1]], "SessionID"] (*&& TrueQ[$Link[sessionUUID, "Active"]]*),
      (
        lenAssoc = Length[input];
        If[lenAssoc == 1, returnInput = <|"Command" -> First[Normal[input]][[1]], "Arguments" -> First[Normal[input]][[2]]|>, returnInput = input];

        browserEvaluationFunction[wdSessionObject, returnInput]
      ),
      (
        Message[WebExecute::invalidInput, input];
        Return[$Failed]
      )
    ]
  )
];


WebExecute[None, input_?AssociationQ] := Block[{tempSession, result},
  tempSession = StartWebSession[];
  result = WebExecute[tempSession, input];
  DeleteObject[tempSession];
  result
];

WebExecute[None, input_?ListQ] := Block[{res, tempSession},
  (
    tempSession = StartWebSession[];
    If[FailureQ[tempSession], $Failed ,
      (
      res = WebExecute[tempSession, #]& /@ input;
      DeleteObject[tempSession];
      res
      )
    ]
  )
];

WebExecute[___]:= Message[WebExecute::argr]; Return[$Failed];

(*Delete session*)
browserDeinitFunc[wdSessionObject_WebSessionObject] := With[
  {websession = KeyTake[{"SessionID", "Browser", "URL"}]@$WebSessionInfos[uuid]},
  KeyDropFrom[$WebSessionInfos, uuid];
  deletesession[websession]
];



