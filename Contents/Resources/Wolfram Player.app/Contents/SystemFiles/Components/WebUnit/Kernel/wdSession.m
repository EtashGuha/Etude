
(* toplevel functions to api binding translations *)
WebSessionStatus[x___] := status[x];
WebSessions[sessionInfo_] /; sessionInfo[[2]] != "Firefox" := sessions[sessionInfo];
WebSessions[sessionInfo_] /; sessionInfo[[2]] == "Firefox" := Message[WebSessions::nnarg];
WebSessions::nnarg = "Not valid for Firefox web browser. Try WebSessionStatus[]";
StopWebSession[x___] := deletesession[x];
$SupportedWebDrivers = Switch[ $SystemID ,
  "Windows-x86-64", {"ChromeDriver", "InternetExplorerDriver", "MicrosoftWebDriver", "GeckoDriver"},
  "Linux-x86-64"  , {"ChromeDriver", "GeckoDriver"},
  "MacOSX-x86-64" , {"ChromeDriver", "GeckoDriver"},
  _, {}
];


GetURL[x___] := geturl[x];

(* ::Section:: *)
(* execute once to start the standalone driver *)
openNewPort[] := Block[{sock = SocketOpen[Automatic], port},
  port = sock["DestinationPort"];
  Close[sock];
  port
];

wdStartWebSession[opts:OptionsPattern[]] := wdStartWebSession["Chrome",opts];

driverFolderMap = <|"Firefox"->"GeckoDriver", "Chrome"->"ChromeDriver"|>

wdStartWebSession[driver_, methodOpts_] := Module[{dir, sessionID, webDriver, webDriverBaseURL, unusedPort, exec, wdProcess,maxRun = 10, loopIndex=0, fetchResult},

  unusedPort = openNewPort[];
  webDriverBaseURL = "http://localhost:" <> ToString[unusedPort];
  webDriver = Which[
    driver == "Chrome", "Chrome",
    driver == "InternetExplorer", "InternetExplorer" ,
    driver == "MicrosoftWeb", "Edge",
    driver == "Firefox", "Firefox",
    driver != "Chrome" || driver != "InternetExplorer" || driver != "MicrosoftWeb" || driver != "Firefox", Return[$Failed],
    _, Null ];


    dir = driverFolderMap[driver];
    If[!StringQ[dir], dir = driver <> "Driver"];
    dir = FileNameJoin[{ PacletManager`PacletResource["WebUnit", "Drivers"], dir, $SystemID }];
    Switch[ driver,
      "Chrome",
      Switch[ $OperatingSystem,
        "Windows", (exec = FileNameJoin[{dir, "chromedriver.exe"  }]; wdProcess = StartProcess[{exec, "--port=" <> ToString[unusedPort]}]),
        "MacOSX" , (exec = FileNameJoin[{dir, "chromedriver"      }]; wdProcess = StartProcess[{exec, "--port=" <> ToString[unusedPort]}]),
        "Unix", (exec = FileNameJoin[{dir, "chromedriver"      }]; wdProcess = StartProcess[{exec, "--port=" <> ToString[unusedPort]},
        ProcessEnvironment -> KeyDrop[{
          "ESPEAK_DATA", "GIO_LAUNCHED_DESKTOP_FILE",
          "GIO_LAUNCHED_DESKTOP_FILE_PID", "__KMP_REGISTERED_LIB_6466",
          "LD_LIBRARY_PATH", "MATHEMATICA_BASE", "MATHEMATICA_USERBASE",
          "OLDPWD", "PSRESOURCEPATH", "QT_PLUGIN_PATH",
          "XFILESEARCHPATH"}]  @  MapAt[StringJoin[
          Riffle[Select[! StringContainsQ[#, $InstallationDirectory] &]@StringSplit[#, ":"], ":"]] &, <|
          Rule @@@ (StringSplit[#, "=", 2] & /@
              StringSplit[RunProcess[$SystemShell, "StandardOutput", "env"], "\n"])|>, Key["PATH"]]]),
        _, Null
      ],

      "Firefox",
      Switch[ $OperatingSystem,
        "Windows"  , (exec = FileNameJoin[{ dir, "geckodriver.exe" }]; wdProcess = StartProcess[{exec , "--port=" <> ToString[unusedPort] }]),
        "MacOSX"   , (exec = FileNameJoin[{ dir, "geckodriver"     }]; wdProcess = StartProcess[{ exec, "--port=" <> ToString[unusedPort] }]),
        "Unix"    , (exec = FileNameJoin[{ dir, "geckodriver"     }];wdProcess = StartProcess[{ exec, "--port=" <> ToString[unusedPort] },
        ProcessEnvironment -> KeyDrop[{
          "ESPEAK_DATA", "GIO_LAUNCHED_DESKTOP_FILE",
          "GIO_LAUNCHED_DESKTOP_FILE_PID", "__KMP_REGISTERED_LIB_6466",
          "LD_LIBRARY_PATH", "MATHEMATICA_BASE", "MATHEMATICA_USERBASE",
          "OLDPWD", "PSRESOURCEPATH", "QT_PLUGIN_PATH",
          "XFILESEARCHPATH"}]  @  MapAt[StringJoin[
          Riffle[Select[! StringContainsQ[#, $InstallationDirectory] &]@
              StringSplit[#, ":"], ":"]] &, <|
          Rule @@@ (StringSplit[#, "=", 2] & /@
              StringSplit[RunProcess[$SystemShell, "StandardOutput", "env"], "\n"])|>, Key["PATH"]]]),
        _, Null
      ],

      "InternetExplorer",
      Switch[ $SystemID,
        "Windows-x86-64", Run["start " <> FileNameJoin[{ dir, "iedriverserver.exe" }]],
        _, Null
      ],
      "MicrosoftWeb",
      Switch[ $SystemID,
        "Windows-x86-64", StartProcess[{FileNameJoin[{ dir, "microsoftwebdriver.exe" }] , "--port=" <> ToString[unusedPort] }],
        _, Null
      ],
      "Safari",
      Switch[ $SystemID,
        "MacOSX-x86-64", Null,
        _, Null
      ],
      _, Null];


  While[!StringMatchQ[ProcessStatus[wdProcess],"Running"] && (loopIndex <= maxRun), Pause[0.05];loopIndex++ ];
  If[loopIndex > maxRun, Return[$Failed]];


  currentWebSession = Association[{"SessionID" -> 0, "Browser" -> webDriver, "URL" -> webDriverBaseURL}];
  sessionID = setsession[currentWebSession, webDriver, methodOpts];
  sessionID = testSetSessionResult[sessionID];
  If[sessionID === $Failed, Return[$Failed]];

  currentWebSession = Association[{"SessionID" -> sessionID, "Browser" -> webDriver, "URL" -> webDriverBaseURL,
    "Exec" -> exec,"Process"->wdProcess,"Active"->True, "SessionTime" -> AbsoluteTime[]}];
  Return[currentWebSession]
];
	
testSetSessionResult[result__] := Block[{sessionID = $Failed},
	Which[
		ListQ[result],
			(* if chromedriver can not find chrome binary *)
			If[Length[result] >= 2 && StringContainsQ[ToString[result[[2]]], "error"],
				Return[$Failed]
			];
			(* Test for an error message that means sessionId is invalid *)
			If[Length[result] >= 1 && result[[1]] =!= "sessionId",
				sessionID = result[[1]]
			];
		,
		result==="sessionId", (* if geckodriver can't find firefox binary *)
			sessionID = $Failed
		,
		StringQ[result],
			sessionID = result
		,
		True,
		sessionID = $Failed
	];
	sessionID
];

chooseBrowser[] := Block[{},
	If[checkChrome[],
		Return["Chrome"];
	];
	If[checkFirefox[],
		Return["Firefox"]
	];
	$Failed
];

checkChrome[] := Block[{paths = {}, homedir},
	homedir = $HomeDirectory;
	Switch[$OperatingSystem,
		"MacOSX"
		,paths = {
			"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
		  }
		,"Windows"
		,paths = {
			"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
			,"C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"
			,"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
			,homedir <> "\\AppData\\Local\\Google\\Chrome\\Application\\chrome.exe"
			,homedir <> "\\Local Settings\\Application Data\\Google\\Chrome\\Application\\chrome.exe"
		  }
		,"Unix",
		paths = {
			"/usr/bin/google-chrome"
			, "/usr/bin/chromium-browser"
		  }
	];
	AnyTrue[paths, FileExistsQ]
];

checkFirefox[] := Block[{paths = {}, homedir},
	homedir = $HomeDirectory;
	Switch[$OperatingSystem,
		"MacOSX"
		,paths = {
			"/Applications/Firefox.app/Contents/MacOS/firefox"
		  }
		,"Windows"
		,paths = {
			"C:\\Program Files (x86)\\Mozilla Firefox\\firefox.exe"
			, "C:\\Program Files\\Mozilla Firefox\\firefox.exe"
		  }
		,"Unix",
		paths = Join[
			{"/usr/bin/firefox", "/opt/firefox/"}
			, FileNames["/usr/lib/firefox-*/"]
			, FileNames["/usr/lib/mozilla-*/"]
			, FileNames["/usr/lib64/firefox-*/"]
		  ]
	];
	AnyTrue[paths, FileExistsQ]
];
