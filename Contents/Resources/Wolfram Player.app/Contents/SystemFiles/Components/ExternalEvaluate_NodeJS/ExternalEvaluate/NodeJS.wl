BeginPackage["ExternalEvaluateNodeJS`"]

Begin["`Private`"]

Needs["PacletManager`"]


$NodeJSIcon = Graphics[\!\(\*
GraphicsBox[
{RGBColor[0.549, 0.7839999999999999, 0.29400000000000004`], EdgeForm[
      None], FilledCurveBox[{{{1, 4, 3}, {1, 3, 3}, {1, 3, 3}, {0, 1, 
       0}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 
       3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {
       1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 
       3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 
       3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {
       1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {0, 1, 0}, {1, 3, 
       3}, {1, 3, 3}}}, CompressedData["
1:eJxF0ztIXEEUxvGzj3vdRUVtTBFFLSxiMKyvJKAx2rhqI6QRBcWsFoqYiIJa
mEKsBBVNsxgLCQQRNAFfiBGC4oMUsRANuGksrIRtEtBCxc1/PAMWH7+dYR5n
5s7mRd6/eecTEQ/pI14S9It8CYqE8SdGrb9tf7IrchkQqU4SqcFBEuX3Cg5g
DMvwFP+56rFX+yfwG9ZjLxZgOWOHKcBl7CH7H7FPDn7Em0QiUYV36MN8+ucp
uBnrcArjrHGAMTzCPfyDGxjHXbzFU7zDC3TYL535GViC+diIL7ENX2M3VmA/
vsBdDGKGX9d/i1/NvZl7wVFcxmncxB1HpIPxfzlXGnvmct4tfIrbWIS/MBTQ
2ky71K9OOWqcucV4jQWYxF1lYype0Pcd1xnbSd8IczP5PgOYjZ3WFtv/xKfj
Lj06b5saU3BNdN1F0X3MmeJWU4ex1J7V1LkkWveq6Dl+iJ4rnXXHMYKtuI4h
PMdHmEUCpIE8JsPmLsgsabKOWc+p9RO+crQddbU9jwtmb1e/7zKeWV3mrOAz
n7YrrfX4GWtx2lrmqDESJh+82s7FPNzy6BknMYKNeEbaySCJkiyy79G3d409
ou85JPq+2eL+vRuPreb/YHzOfRfiEHbhDM6xRpjvc8LYq8DD//A/UbF7sA==

"]], FilledCurveBox[{{{1, 4, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 
       3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {
       1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 
       3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 
       3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3, 3}, {
       1, 3, 3}, {1, 3, 3}}}, CompressedData["
1:eJxF0kFIFUEYB/DpuW/1UAchDKFDER0MDYmKJOgkFBYYXhKUCDGkQwoKQsEr
q4MQCUEQGBj0KoQgi8IIgqyOChF6KOhWSFGIHuoSkf6GWejw57ez8803O7u7
s2+oa7AmhLBJuqUk28ohLOQhTHGJe+WDTMuElGtD6GEb19Qc5jgP8QKbOMzt
HOHvPPk+T/eni/HtYnyDV3idN3mZ79jPXzzAzbLiupWT5fRMHWxhPfdwKQth
N6usYYU/zA3zRezLibgfhzjDs5xnL8/LPmm09iBf8p+5P/yWpXO+YhdHYw82
so+PvMQdPMOf6ndxiiV2cs2L/a7nJ1YKt/ILn+mxwuOsVb8uHfJarsld+Sz3
pFVNlUf4lMeyVHeUy8V4S5bcn6X77cW4M55BTkqzdMtfa85xnrc4w6+8GHt6
vku8zwpXOcb2+L14lc/jfPw/+JjB2gds4h2e4Ch7eYoDxbeYZB3Hs7T/myyt
+Ri/mV4lc4txLz5hlXNs0KNOzek8nechZ/k2///fbgBnGFrz
"]]},
ImageSize->{40.7109375, Automatic}]\)[[1]], ImageSize -> {Automatic, Dynamic[3.5*CurrentValue["FontCapHeight"]]}];


ExternalEvaluate::nodejsfilename = "The code from `1` has been assigned to the variable `2`."

(*we load the regular expression and evaluate it once on a dummy string so it gets compiled *)
(*this string comes from https://mathiasbynens.be/demo/javascript-identifier-regex*)
$varNameRegex = RegularExpression[Import[PacletManager`PacletResource["ExternalEvaluate_NodeJS","VariableNameRegex"], "Text"]];
StringMatchQ["",$varNameRegex];


(*register nodejs as a system*)
ExternalEvaluate`RegisterSystem["NodeJS",
	<|
		"ExecutablePathFunction"->Function[{version},
				Switch[$OperatingSystem,
					"Windows",{
						FileNameJoin[{Environment["PROGRAMFILES"],"nodejs"}],
						FileNameJoin[{Environment["PROGRAMFILES(X86)"],"nodejs"}]
					},
					"MacOSX",{"/usr/bin","/usr/local/bin","/usr/local/sbin","/usr/local/Cellar/bin"},
					"Unix",{"/usr/bin","/usr/local/bin/","/usr/local/sbin"}
				]
			],
		
		"ExecutablePatternFunction"->Function[{version},
			With[{fileExtension = Switch[$OperatingSystem,"Windows",".exe",_,""]},
				"node"<>fileExtension
			]
		],
		
		"ProgramFileFunction"->Function[{version},PacletManager`PacletResource["ExternalEvaluate_NodeJS","NodeJSREPL"]],
		
		"DependencyTestFile"->PacletManager`PacletResource["ExternalEvaluate_NodeJS","NodeJSZMQTest"],
		
		"RunFileInSessionFunction"->Function[{file},
			Block[
				{
					filename = FixedPoint[FileBaseName,file]
				},
				(*check if the filename is an invalid variable name in NodeJS, if that's the case then generate a new version*)
				"var "<>If[StringMatchQ[filename,$varNameRegex],
					(*THEN*)
					(*filename is valid and we can use it*)
					filename,
					(*ELSE*)
					(*filename is invalid and we need to generate a new unique one and issue a message about it*)
					(
						newfilename = "_wl_" <> StringJoin[FromLetterNumber /@ RandomInteger[{1, 26}, 4]];
						Message[ExternalEvaluate::nodejsfilename,file,newfilename];
						newfilename
					)
				]<>" = require('"<>file<>"');"
			]
		],
		
		"Icon"->$NodeJSIcon,

		(*make an automatic paclet name out of this to get auto-updating behavior*)
		"PacletName"->Automatic,

        "DeserializationFunction" -> Function[
            With[
                {res = Developer`ReadRawJSONString[ByteArrayToString[#]]},
                Which[
                    FailureQ[res],
                    res,
                    TrueQ[res["is_expr"]] && StringQ[res["output"]],
                    ImportString[res["output"], "ExpressionJSON"],
                    KeyExistsQ[res, "error"],
                    Failure["NodeJSError", res["error"]],
                    True,
                    Lookup[res, "output", Null]
                ]
            ]
        ]
	|>
];

(*================================================*)
(*user defined function handle*)
(*================================================*)

ExternalEvaluate`Private`handleDefinedFunction["NodeJS"] = handleDefinedFunctionNodeJS;

handleDefinedFunctionNodeJS[session_ExternalSessionObject, functionDef_String] := Block[
  {x, defResult, func, system = session["System"]},

  Which[
    nodeFunctionITest[functionDef], defResult = ExternalEvaluate[session, functionDef],
    nodeFunctionIITest[functionDef], nodeFunctionII[session, functionDef],
    nodeFunctionIIITest[functionDef], nodeFunctionIII[session, functionDef],
    !(nodeFunctionITest[functionDef]||nodeFunctionIITest[functionDef]||nodeFunctionIIITest[functionDef]),
    (Message[ExternalFunction::nofunction,functionDef];Return[$Failed])

  ]

];

nodeFunctionITest[functionDef_] := Block[{funcName},
  funcName = StringCases[functionDef, z : WordCharacter .. ~~ WhitespaceCharacter .. ~~ "=" ~~ WhitespaceCharacter .. ~~ "function" -> z];
  MatchQ[funcName, {_String, ___}]
];

(*Type II:  function addJS (args) {function body}*)
nodeFunctionIITest[functionDef_] := Block[{funcName},
  funcName = StringCases[functionDef, "function" ~~ v : WhitespaceCharacter .. ~~ x : WordCharacter .. -> x];
  MatchQ[funcName, {_String, ___}]
];


nodeFunctionII[session_ExternalSessionObject, functionDef_] := Block[{funcName, func},
  If[funcName = StringCases[functionDef, "function" ~~ v : WhitespaceCharacter .. ~~ x : WordCharacter .. -> x];
  MatchQ[funcName, {_String, ___}],
    funcName = First[funcName],
    Message[ExternalFunction::invlDef];
    Return[$Failed];
  ];
  defResult = ExternalEvaluate[session, functionDef];
  If[defResult =!= Null,
    Message[ExternalFunction::invlDef];
    Return[defResult]
  ];
  func = ExternalEvaluate[session, funcName];
  If[! MatchQ[func, _ExternalFunction],
    Message[ExternalFunction::invlDef];
  ];
  Return[func]
];



nodeFunctionIIITest[functionDef_] := Block[{funcName},
  StringContainsQ[functionDef, "."] && (! StringContainsQ[functionDef, "function"])
];

nodeFunctionIII[session_ExternalSessionObject, functionDef_] := Block[{funcName, func, defResult},
  (
    Return[
      ExternalFunction[<|
        "Name"->functionDef,
        "Arguments"->None,
        "System"->system,
        "BuiltIn"->False
      |>]
    ]
  )
];



ExternalEvaluate`Private`defFunctionIdentifier["NodeJS"] = defFunctionIdentifierNodeJS;
defFunctionIdentifierNodeJS[funcName_] := If[StringContainsQ[funcName, {"function","."}],
  Return[True], Return[False]
];


End[]
EndPackage[]
