BeginPackage["ExternalEvaluateWebDriver`"]

Begin["`Private`"]

Needs["PacletManager`"]
Needs["WebUnit`"]

ExternalEvaluate::webdriverelem = "The element specification `1` is invalid - element specifications must be a string or an association with a single key."
ExternalEvaluate::webdriverelemassoc = "The element specification `1` has invalid keys."
ExternalEvaluate::webdriverelemval = "The element specification value `1` must be a string."
ExternalEvaluate::browsernoinstall = "Web browser `1` is not installed."
ExternalEvaluate::browserunknown = "Web browser `1` is not supported."
ExternalEvaluate::nocom = "Unable to communicate with web browser."

$WebSessionInfos = <||>;

$WebDriverIcon = Graphics[{Thickness[0.04], {FaceForm[{RGBColor[1., 1., 1.], Opacity[1.]}],
   FilledCurve[{{{1, 4, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}}}, {{{13., 22.}, {17.971, 22.}, {22.,
    17.971}, {22., 13.}, {22., 8.029}, {17.971, 4.}, {13., 4.}, {8.029, 4.}, {4., 8.029}, {4.,
    13.}, {4., 17.971}, {8.029, 22.}, {13., 22.}}}]}, {RGBColor[0.392, 0.392, 0.392], Opacity[1.],
   JoinForm[{"Miter", 10.}], JoinedCurve[{{{1, 4, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}}}, {{{13.,
    22.}, {17.971, 22.}, {22., 17.971}, {22., 13.}, {22., 8.029}, {17.971, 4.}, {13., 4.}, {8.029,
    4.}, {4., 8.029}, {4., 13.}, {4., 17.971}, {8.029, 22.}, {13., 22.}}}, CurveClosed -> {1}]},
  {RGBColor[0.392, 0.392, 0.392], Thickness[0.041639999999999996], Opacity[1.], CapForm["Round"],
   JoinForm[{"Miter", 10.}], JoinedCurve[{{{1, 4, 3}, {1, 3, 3}}}, {{{4.5835, 10.2964}, {6.5485,
    8.846400000000001}, {9.5855, 7.9164}, {12.9955, 7.9164}, {16.4815, 7.9164}, {19.5795, 8.8894},
    {21.5385, 10.3964}}}, CurveClosed -> {0}]}, {RGBColor[0.392, 0.392, 0.392],
   Thickness[0.041639999999999996], Opacity[1.], CapForm["Round"], JoinForm[{"Miter", 10.}],
   JoinedCurve[{{{1, 4, 3}, {1, 3, 3}}}, {{{5.5098, 17.6963}, {7.4428, 16.5953}, {10.0828,
    15.916300000000001}, {12.994800000000001, 15.916300000000001}, {15.9288, 15.916300000000001},
    {18.5888, 16.6053}, {20.5248, 17.7213}}}, CurveClosed -> {0}]},
  {RGBColor[0.392, 0.392, 0.392], Thickness[0.041639999999999996], Opacity[1.], CapForm["Round"],
   JoinForm[{"Miter", 10.}], JoinedCurve[{{{1, 4, 3}, {1, 3, 3}}}, {{{4.1802, 13.5786},
    {6.450200000000001, 12.5036}, {9.5562, 11.8666}, {12.984200000000001, 11.9156}, {16.4042,
    11.964599999999999}, {19.5062, 12.6896}, {21.7742, 13.8266}}}, CurveClosed -> {0}]},
  {RGBColor[0.392, 0.392, 0.392], Opacity[1.], CapForm["Round"], JoinForm[{"Miter", 10.}],
   JoinedCurve[{{{1, 4, 3}, {1, 3, 3}}}, {{{10.955100000000002, 21.586399999999998},
    {9.148100000000001, 19.720399999999998}, {7.962100000000001, 16.571399999999997},
    {7.962100000000001, 13.000399999999997}, {7.962100000000001, 9.3384}, {9.210100000000002,
    6.1203999999999965}, {11.095100000000002, 4.274399999999996}}}, CurveClosed -> {0}]},
  {RGBColor[0.392, 0.392, 0.392], Opacity[1.], CapForm["Round"], JoinForm[{"Miter", 10.}],
   JoinedCurve[{{{1, 4, 3}, {1, 3, 3}}}, {{{14.9048, 4.2739}, {16.7898, 6.1199}, {18.0378,
    9.338899999999999}, {18.0378, 13.000899999999998}, {18.0378, 16.5719}, {16.8508,
    19.721899999999998}, {15.0438, 21.587900000000005}}}, CurveClosed -> {0}]},
  {RGBColor[0.392, 0.392, 0.392], Opacity[1.], CapForm["Round"], JoinForm[{"Miter", 10.}],
   JoinedCurve[{{{0, 2, 0}}}, {{{13., 21.765599999999996}, {13., 4.011599999999994}}},
    CurveClosed -> {0}]}}, AspectRatio -> Automatic, ImageSize -> {25., 25.},
 PlotRange -> {{0., 25.}, {0., 25.}}];

browserDeinitFunc[uuid_,proc_] := Block[
	{webSession},
	webSession = $WebSessionInfos[uuid];
	$WebSessionInfos = KeyDrop[$WebSessionInfos, uuid];
	If[!MissingQ[webSession], DeleteObject[webSession]];
]

(* This just avoids a zmq socket startup *)
browserInitFunc[args___] := Block[{},
	(Null)
];

browserStartFunc[browser_,visible_:True] := 
Function[{uuid, exec, file, opts},
	Block[{session, proc},
		session = StartWebSession[browser, Visible->visible];
		If[!MatchQ[session, _WebSessionObject],Return[$Failed]];
		proc = session["Process"];
		If[!MatchQ[proc, _ProcessObject], Return[$Failed]];
		$WebSessionInfos[uuid] = session;
		proc
	]
];

browserEvaluationFunction[session_,input_,exprQ_] := Block[{func, args, formed, webSession, uuid},
	formed = input;
	If[AssociationQ[formed],
		func = formed["function"];
		args = formed["args"];
		If[!MissingQ[func],
			formed = func;
		];
		If[!MissingQ[args],
			If[ListQ[args] && Length[args] === 1,
				args = args[[1]]
			];
			formed = formed->args
		];
	];
	uuid = session["UUID"];
	webSession = $WebSessionInfos[uuid];
	WebExecute[webSession, formed]
]



(*register webdriver as a system*)
ExternalEvaluate`RegisterSystem["WebDriver-Chrome",
	<|
		"NonZMQDeinitializeFunction"-> browserDeinitFunc,
		"NonZMQInitializeFunction"-> browserInitFunc,
		"ScriptExecCommandFunction"-> browserStartFunc["Chrome"],
		"NonZMQEvaluationFunction"-> browserEvaluationFunction,
		"ExecutablePatternFunction"-> Function[{vers},
			StartOfString~~"chromedriver"~~Switch[$OperatingSystem,"Windows",".exe",_,""]~~EndOfString	
		],
		"ExecutablePathFunction"-> Function[{vers},FileNames[FileNameJoin[{ PacletManager`PacletResource["WebUnit","Drivers"], "ChromeDriver",$SystemID}]]],
		"ProcessEnvironmentFunction"-> Function[{},Switch[ $OperatingSystem,
				"Windows", (Inherited),
				"MacOSX", (Inherited),
				"Unix", KeyDrop[{
																						"ESPEAK_DATA", "GIO_LAUNCHED_DESKTOP_FILE",
																						"GIO_LAUNCHED_DESKTOP_FILE_PID", "__KMP_REGISTERED_LIB_6466",
																						"LD_LIBRARY_PATH", "MATHEMATICA_BASE", "MATHEMATICA_USERBASE",
																						"OLDPWD", "PSRESOURCEPATH", "QT_PLUGIN_PATH",
																						"XFILESEARCHPATH"}]  @  MapAt[StringJoin[
                                              Riffle[Select[! StringContainsQ[#, $InstallationDirectory] &]@
                                              StringSplit[#, ":"], ":"]] &, <|
                                              Rule @@@ (StringSplit[#, "=", 2] & /@
                                              StringSplit[RunProcess[$SystemShell, "StandardOutput", "env"], "\n"])|>, Key["PATH"]]]],
		(*the name for this paclet is ExternalEvalaute_WebDriver, not ExternalEvalaute_WebDriver-Chrome*)
		"PacletName"->"ExternalEvaluate_WebDriver",
		"Icon"->$WebDriverIcon,
		"VersionStringConformFunction"->Function[{versionString,userQ},
			StringTrim@First@StringSplit[StringDelete[versionString, StartOfString ~~ "ChromeDriver"],"("]
		]
	|>
]

(*headless is the same as chrome except for the init function*)
ExternalEvaluate`RegisterSystem["WebDriver-Chrome-Headless","WebDriver-Chrome",
	<|
		"ScriptExecCommandFunction"-> browserStartFunc["Chrome", False]
	|>
]


(*firefox inherits from chrome everything except the icon, the init function, and the executable paths/patterns*)
ExternalEvaluate`RegisterSystem["WebDriver-Firefox","WebDriver-Chrome",
	<|
		"ScriptExecCommandFunction"-> browserStartFunc["Firefox"],
		"ExecutablePatternFunction"->Function[{vers},
			StartOfString~~"geckodriver"~~Switch[$OperatingSystem,"Windows",".exe",_,""]~~EndOfString
		],
		"ExecutablePathFunction"->Function[{vers},FileNames[FileNameJoin[{ PacletManager`PacletResource["WebUnit","Drivers"], "GeckoDriver",$SystemID}]]],
		"VersionStringConformFunction"->Function[{versionString,userQ},
			StringTrim@StringDelete[First@StringSplit[versionString, "\n"],StartOfString ~~ "geckodriver"]
		]
	|>
]

(*firefox headless is same as normal except for the init function*)
ExternalEvaluate`RegisterSystem["WebDriver-Firefox-Headless","WebDriver-Firefox",
	<|
		"ScriptExecCommandFunction"-> browserStartFunc["Firefox", False]
	|>
]

End[]
EndPackage[]