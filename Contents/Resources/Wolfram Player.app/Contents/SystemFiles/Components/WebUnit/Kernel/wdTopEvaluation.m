stringToFunctionAssoc = <|
  "RefreshWebPage" ->     RefreshWebPage,
  "RefreshPage" ->        RefreshWebPage,
  "PageBack" ->           PageBack,
  "PageForward" ->        PageForward,
  "PageTitle" ->          WebPageTitle,
  "PageURL" ->            GetURL,
  "PageSource" ->         GetPageHtml,
  "WebSessionStatus" ->   WebSessionStatus,
  "PageHyperlinks" ->     PageLinks,
  "WindowFullscreen" ->   WindowFullscreen,
  "WindowFullScreen" ->   WindowFullscreen,
  "OpenWebPage" ->        OpenWebPage,
  "OpenPage" ->           OpenWebPage,
  "JavascriptExecute" ->  JavascriptExecute,
  "ClickElement" ->       ClickElement,
  "HoverElement" ->       HoverElement,
  "HideElement" ->        HideElement,
  "ElementText" ->        ElementText,
  "ElementTag" ->         ElementName,
  "ElementSelected" ->    ElementSelected,
  "ElementEnabled" ->     ElementEnabled,
  "SubmitElement" ->      SubmitElement,
  "FocusFrame" ->         FocusFrame,
  "ShowElement" ->        ShowElement,
  "WindowSize" ->         GetWindowSize,
  "SetWindowSize" ->      SetWindowSize,
  "SetBrowserWindow" ->   SetBrowserWindow,
  "WindowMaximize" ->     WindowMaximize,
  "WindowMinimize" ->     WindowMinimize,
  "GetWindowPosition" ->  GetWindowPosition,
  "CapturePage"    ->     CaptureWebPage,
  "LocateElements"    ->  LocateElements,
  "LocateElement"    ->   LocateElement,
  "BrowserWindows"    ->  BrowserWindows ,
  "GetWindow"   ->        GetWindow,
  "TypeElement"   ->      TypeElement,
  "SetWindowPosition"  -> SetWindowPosition

|>;

listOfFunction[]:= TableForm[Sort@Keys[stringToFunctionAssoc], TableHeadings -> Automatic];

invalidNoneFail [command_] :=       Failure["NoArgument",  <|
  "MessageTemplate" -> "An argument is required for `command`",
  "MessageParameters" -> <|"command" -> command|>
|>];

validNoneFail [command_] := Failure["NoArgument",  <|
  "MessageTemplate" -> "`command` does not support arguments.",
  "MessageParameters" -> <|"command" -> command|>
|>];


(*This function is called from WebExecute*)
browserEvaluationFunction[wdSessionObject_WebSessionObject, input_?AssociationQ] := Block[
  {
    uuid = session["UUID"], (*session = ExternalSessionObject*)
    command (*= input /. {a_?AssociationQ :> a, f_String:> <|"function"->f,"args"->{}|>},*),
    args, res, wdValueId, foundElements, windowIDs, windowID,locatorOption,typedString
  },
  command = input["Command"];
  args = input["Arguments"];
  args = If[Head[args] === WebElementObject, args["ElementId"], args];
  Switch[command,
    "RefreshWebPage" | "RefreshPage" |"PageBack" | "PageForward" ,
    (
      If[!MatchQ[args, None], Return@validNoneFail [command]];
      res = stringToFunctionAssoc[command][wdSessionObject[[1]]];
      Which[res === Null || res === {}||res === {Null},
        Success[command, <|
          "MessageTemplate" :> "`command` was successful.",
          "MessageParameters" -> <|"command" -> command|>
        |>],
        True, Return[$Failed]
      ]),


  (*"PageHyperlinks", WindowFullscreen & "PageSource" returns list*)
    "PageTitle" | "PageURL" | "WebSessionStatus" | "PageHyperlinks" | "PageSource" |"WindowFullscreen"|"WindowFullScreen",
    (
      If[!MatchQ[args, None], Return@validNoneFail [command]];
      res = stringToFunctionAssoc[command][wdSessionObject[[1]]];
      If[StringQ[res]||ListQ[res],
        res,
        Failure["InvalidInput",  <|
          "MessageTemplate" -> "`command` failed.",
          "MessageParameters" -> <|"command" -> command|>
        |>]]
    ),


    "OpenWebPage"| "OpenPage",
    (
      If[MatchQ[args, None], Return@invalidNoneFail [command]];
      args = formatURL[args];
      res = stringToFunctionAssoc[command][wdSessionObject[[1]], args];
      Which[res === Null || res === {},
        Success[command, <|
          "MessageTemplate" :> "Page `page` opened successfully.",
          "MessageParameters" -> <|
            "page" -> args|>
        |>],

        MatchQ[res, KeyValuePattern["message" -> ___]],
        (Message[WebExecute::invalidURL,args];Return[$Failed]),
        True, Return[$Failed];
      ]
    ),
  (*Functions which return Null on Success on passing an ElementId*)
    "ClickElement" | "HoverElement" | "HideElement" | "SubmitElement" | "ShowElement" ,
    (
      If[MatchQ[args, None], Return@invalidNoneFail [command]];
      args = queryMethodGenerator[args];
      If[FailureQ[args], Return[args]];
      res = stringToFunctionAssoc[command][wdSessionObject[[1]], args];

      Which[res === Null || res === {}||res === {Null}||MatchQ[res, {Null ..}],
        Success[command, <|
          "MessageTemplate" :> "`command` was successful.",
          "MessageParameters" -> <|"command" -> command|>,
          "Element" -> If[MatchQ[input["Arguments"],_WebElementObject],input["Arguments"]["ElementId"],input["Arguments"]]
        |>],

        MatchQ[res, KeyValuePattern["message" -> ___]],
        Failure["InvalidElement",  <|
          "MessageTemplate" -> "`command` failed.",
          "MessageParameters" -> <|"command" -> command|>,
          "Element" -> If[MatchQ[input["Arguments"],_WebElementObject],input["Arguments"]["ElementId"],input["Arguments"]]
        |>],
        True, Return[$Failed]
      ]
    ),
  (*Functions which return String on Success*)
    "ElementText" | "ElementTag" | "ElementSelected" | "ElementEnabled"  ,
    (
      If[MatchQ[args, None], Return@invalidNoneFail [command]];
      args = queryMethodGenerator[args];
      If[FailureQ[args], Return[args]];
      res = stringToFunctionAssoc[command][wdSessionObject[[1]], args];

      Which[
        MatchQ[res, KeyValuePattern["message" -> ___]],
        Failure["InvalidElement",  <|
          "MessageTemplate" -> "`command` failed.",
          "MessageParameters" -> <|"command" -> command|>,
          "Element" -> If[MatchQ[input["Arguments"],_WebElementObject],input["Arguments"]["ElementId"],input["Arguments"]]
        |>],

        True, Return[res]
      ]),
    "JavascriptExecute" | "FocusFrame" ,
    (
    If[MatchQ[args, None], Return@invalidNoneFail [command]];
    res = stringToFunctionAssoc[command][wdSessionObject[[1]], args];
    Which[
      MatchQ[res, KeyValuePattern["message" -> ___]],
      Failure["InvalidInput",  <|
        "MessageTemplate" -> "`command` failed.",
        "MessageParameters" -> <|"command" -> command|>,
        "Element" -> If[MatchQ[input["Arguments"],_WebElementObject],input["Arguments"]["ElementId"],input["Arguments"]]
      |>],

      True, Return[res]
    ]
    ),

    "WindowSize",
    (
      If[!MatchQ[args, None], Return@validNoneFail [command]];
      {"width", "height"} /. GetWindowSize[wdSessionObject[[1]], BrowserWindows[wdSessionObject[[1]]][[1]]]
    ),
    "SetWindowSize",
    (
      If[MatchQ[args, None], Return@invalidNoneFail [command]];
      SetWindowSize[wdSessionObject[[1]], BrowserWindows[wdSessionObject[[1]]][[1]], args]
    ),
    "CapturePage",
    If[!MatchQ[args, None], Return@validNoneFail [command]];
    With[
    {
      res = CaptureWebPage[wdSessionObject[[1]]]
    },
  (*for chrome, capturing the web page will bring it to the front so we should bring the M front end back to front if we're able to for more expected usage*)
    If[wdSessionObject[[1]]["Browser"] === "Chrome" && $FrontEnd =!= Null, FrontEndTokenExecute["BringToFront"] ];
    (*If[ImageQ[res], Show[res, ImageSize -> Automatic], $Failed ]*)
    If[ImageQ[res], Image[res, ImageSize -> Automatic], $Failed ]
  ],

    "LocateElements",
    (
      If[MatchQ[args, None], Return@invalidNoneFail [command]];
      (*example: args = {XPath->//*[@id="_nav-search"]/a/i[2]}*)
      args = queryMethodGenerator[args];
      If[FailureQ[args],       Return@Failure["InvalidInput",  <|
        "MessageTemplate" -> "`command` failed.",
        "MessageParameters" -> <|"command" -> command|>
      |>]
      ];
      foundElements = LocateElements[wdSessionObject[[1]], args];
      If[FailureQ[foundElements], Return@Failure["InvalidInput",  <|
        "MessageTemplate" -> "`command` failed.",
        "MessageParameters" -> <|"command" -> command|>
      |>]
      ];
      Return[
        WebElementObject[
          <|
            "SessionID" -> wdSessionObject[[1]]["SessionID"],
            "Browser" -> wdSessionObject[[1]]["Browser"],
            "Process" -> wdSessionObject[[1]]["Process"],
            "URL" -> wdSessionObject[[1]]["URL"],
            "Exec" -> wdSessionObject[[1]]["Exec"],
            "ElementId" -> #
          |>
        ]& /@ foundElements
      ]
    ),
    "BrowserWindows",
    (
      If[!MatchQ[args, None], Return@validNoneFail [command]];
      windowIDs = BrowserWindows[wdSessionObject[[1]]];
      res = windowFromAssoc[wdSessionObject, #]& /@ windowIDs ;

      If[!MatchQ[res, {__WebWindowObject}],Return[$Failed]];
      res
),
    "GetWindow",
    (
      If[!MatchQ[args, None], Return@validNoneFail [command]];
      windowID = GetWindow[wdSessionObject[[1]]];
      Return[ windowFromAssoc[wdSessionObject, #] & /@ {windowID} ];
    ),
    "SetBrowserWindow" | "WindowMaximize" | "WindowMinimize" | "GetWindowPosition",
    (*WebWindowObject is a valid argument*)
    (
      If[MatchQ[args, None], Return@invalidNoneFail [command]];

      If[!MatchQ[args, _WebWindowObject], Return@Failure["InvalidArgument",  <|
        "MessageTemplate" -> "`command` failed.",
        "MessageParameters" -> <|"command" -> command|>
      |>]
];

      wdValueId = If[(Head[args] === WebWindowObject), args[[1]]["WindowID"], args];
      stringToFunctionAssoc[command][wdSessionObject[[1]], wdValueId]
    ),
    "TypeElement",
    (
      If[MatchQ[args, None], Return@invalidNoneFail [command]];
      If[Length[args]!= 2,        Return@Failure["InvalidLength",  <|
          "MessageTemplate" -> "The argument passed with `arguments` has invalid length.",
          "MessageParameters" -> <|"arguments" -> args|>
        |>]];
      locatorOption = args[[1]];
      typedString = args[[2]];
      If[(Head[locatorOption] === WebElementObject),
        wdValueId = locatorOption[[1]]["ElementId"],
        wdValueId = queryMethodGenerator[locatorOption]
      ];
      If[FailureQ[wdValueId], Return@Failure["InvalidInput",  <|
        "MessageTemplate" -> "`command` failed.",
        "MessageParameters" -> <|"command" -> command|>
      |>]
      ];
      res = TypeElement[wdSessionObject[[1]], wdValueId, typedString];

      Which[
        MatchQ[res, KeyValuePattern["message" -> ___]],
        Failure["InvalidInput",  <|
          "MessageTemplate" -> "`command` failed.",
          "MessageParameters" -> <|"command" -> command|>,
          "Element" -> If[MatchQ[input["Arguments"][[1]],_WebElementObject],input["Arguments"][[1]]["ElementId"],input["Arguments"][[1]]]
        |>],

        res === Null || res === {}||res === {Null},
        Success[command, <|
          "MessageTemplate" :> "`command` was successful.",
          "MessageParameters" -> <|"command" -> command|>,
          "Element" -> If[MatchQ[input["Arguments"][[1]],_WebElementObject],input["Arguments"][[1]]["ElementId"],input["Arguments"][[1]]]
        |>],

        True, Return[res]
      ]
    ),
    "SetWindowPosition",
    (*{WebWindowObject, {w, h}} is an valid argument here*)
    (
      If[MatchQ[args, None], Return@invalidNoneFail [command]];
      If[Length[args]!= 2,        Return@Failure["InvalidLength",  <|
        "MessageTemplate" -> "The argument passed with `arguments` has invalid length.",
        "MessageParameters" -> <|"arguments" -> args|>
      |>]];

      wdValueId = If[(Head[args[[1]]] === WebWindowObject), args[[1]][[1]]["WindowID"], args[[1]]];
      res = SetWindowPosition[wdSessionObject[[1]], wdValueId, args[[2]]];

      Which[
        !(ListQ[res]||MatchQ[res,Null]),
        Failure["InvalidArgument",  <|
          "MessageTemplate" -> "`command` failed.",
          "MessageParameters" -> <|"command" -> command|>
        |>],

        True, Return[res]
      ]
    ),
    _,Message[WebExecute::nocom, command];$Failed
  ]
]