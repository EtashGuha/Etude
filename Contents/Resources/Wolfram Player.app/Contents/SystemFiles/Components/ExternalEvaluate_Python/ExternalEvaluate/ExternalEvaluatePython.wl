
BeginPackage["ExternalEvaluatePython`"]

Begin["`Private`"]

Needs["ExternalEvaluate`"];

Needs["PacletManager`"];

$PythonIcon = Graphics[\!\(\*
GraphicsBox[{
{Hue[0.5766283524904214, 0.6682027649769585, 0.651], EdgeForm[None],
       FilledCurveBox[{{{1, 4, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3,
         3}}, {{1, 4, 3}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1,
         0}, {1, 3, 3}, {1, 3, 3}, {0, 1, 0}, {0, 1, 0}, {1, 3, 3}, {
         0, 1, 0}, {1, 3, 3}, {0, 1, 0}, {1, 3, 3}}}, {{{58, 120}, {
         60, 120}, {62, 118}, {62, 115}, {62, 112}, {60, 110}, {58,
         110}, {55, 110}, {53, 112}, {53, 115}, {53, 118}, {55,
         120}, {58, 120}}, {{72, 128}, {44, 128}, {46, 116}, {46,
         116}, {46, 104}, {73, 104}, {73, 100}, {36, 100}, {36,
         100}, {18, 102}, {18, 74}, {18, 45}, {33, 46}, {33, 46}, {43,
          46}, {43, 59}, {43, 59}, {42, 75}, {58, 75}, {85, 75}, {85,
         75}, {99, 75}, {99, 89}, {99, 114}, {99, 114}, {102, 128}, {
         72, 128}}}]},
{Hue[0.1164, 0.745, 0.99], EdgeForm[None],
       FilledCurveBox[{{{1, 4, 3}, {1, 3, 3}, {1, 3, 3}, {1, 3,
         3}}, {{1, 4, 3}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1,
         0}, {1, 3, 3}, {1, 3, 3}, {0, 1, 0}, {0, 1, 0}, {1, 3, 3}, {
         0, 1, 0}, {1, 3, 3}, {0, 1, 0}, {1, 3, 3}}}, {{{88, 27}, {85,
          27}, {83, 29}, {83, 32}, {83, 34}, {85, 37}, {88, 37}, {91,
         37}, {93, 34}, {93, 32}, {93, 29}, {91, 27}, {88, 27}}, {{73,
          18}, {101, 18}, {99, 31}, {99, 31}, {99, 43}, {73, 43}, {73,
          47}, {110, 47}, {110, 47}, {128, 45}, {128, 73}, {128,
         102}, {112, 101}, {112, 101}, {103, 101}, {103, 87}, {103,
         87}, {104, 72}, {88, 72}, {61, 72}, {61, 72}, {46, 72}, {46,
         57}, {46, 33}, {46, 33}, {44, 18}, {73, 18}}}]}},
ImageSize->{38.171875, Automatic}]\)[[1]], ImageSize -> {Automatic, Dynamic[3.5*CurrentValue["FontCapHeight"]]}];


(*add the heuristics for python*)
ExternalEvaluate`RegisterSystem["Python",
	<|
		(*standard paths are possible install locations *)
		"ExecutablePathFunction"->Function[{vers},
				Switch[$OperatingSystem,
					"Windows",
					With[{maindrive = First[FileNameSplit[Environment["PROGRAMFILES"]]]},
						{
							FileNameJoin[{maindrive,"Python"}],
							FileNameJoin[{Environment["LOCALAPPDATA"],"Programs","Python","Python"}],
							FileNameJoin[{Environment["PROGRAMFILES"],"Python"}],
							FileNameJoin[{$UserBaseDirectory,"ApplicationData","SystemInstall","Python"}],

							(*versioned folder names*)
							FileNameJoin[{maindrive,"Python"<>StringJoin[Take[StringSplit[vers,"."],UpTo[2]]]}],
							FileNameJoin[{Environment["LOCALAPPDATA"],"Programs","Python","Python"<>StringJoin[Take[StringSplit[vers,"."],UpTo[2]]]}],
							FileNameJoin[{Environment["PROGRAMFILES"],"Python"<>StringJoin[Take[StringSplit[vers,"."],UpTo[2]]]}],

							(*32-bit executables have -32 at the end*)
							FileNameJoin[{Environment["PROGRAMFILES(X86)"],"Python-32"}],
							FileNameJoin[{Environment["LOCALAPPDATA"],"Programs","Python","Python"<>StringJoin[Take[StringSplit[vers,"."],UpTo[2]]]<>"-32"}],
							FileNameJoin[{Environment["PROGRAMFILES(X86)"],"Python"<>StringJoin[Take[StringSplit[vers,"."],UpTo[2]]]<>"-32"}]
						}
					],
					"MacOSX",Join[
						{
							"/usr/bin",
							"/usr/local/bin/",
							"/usr/local/sbin",
							"/usr/local/Cellar/bin"
						},
						(*this next path(s) is from the pkg installer on mac*)
						If[vers === "*",
							(*THEN*)
							(*we should search for any possible versions in /Library*)
							FileNames["/Library/Frameworks/Python.framework/Versions/*/bin"],
							(*ELSE*)
							(*try to use the specified version in the /Library path*)
							FileNameJoin[
								{
									"/Library/Frameworks/Python.framework/Versions",
									StringRiffle[Take[StringSplit[vers, "."], UpTo[2]], "."],
									"bin"
								}
							]
						]
					],
					"Unix",{"/usr/bin","/usr/local/bin/","/usr/local/sbin"}
				]
			],

		(*these are possible executable names, which like above are string templates with the configurable parameter of vers, which is the version of python*)
		"ExecutablePatternFunction"->
			(*the executable pattern function is a function that when given the pattern, returns a pattern that can be used with FileNames and the ExecutablePathFunction to attempt to locate a suitable executable*)
			With[{fileExtension = Switch[$OperatingSystem,"Windows",".exe",_,""]},
			Function[{vers},
					Alternatives@@If[vers==="*",
						(*THEN*)
						(*use the generic version that only matches numbers or sequences of numbers or a number with digits in between after the executable name*)
						{
							StartOfString~~"python"~~fileExtension~~EndOfString,
							StartOfString~~"python"~~(DigitCharacter..|(DigitCharacter~~("."~~DigitCharacter)...))~~fileExtension~~EndOfString
						},
						(*ELSE*)
						(*create the various different forms manually, as the pattern is like the above, but we only want to match it for a specific sequence of numbers, which greatly complicates the pattern*)
						{
							"python.exe",
							"python"<>StringJoin[Take[StringSplit[vers,"."],UpTo[1]]]<>fileExtension,
							"python"<>StringJoin[Take[StringSplit[vers,"."],UpTo[2]]]<>fileExtension,
							"python"<>StringJoin[Take[StringSplit[vers,"."],UpTo[3]]]<>fileExtension,
							"python"<> StringJoin[Riffle[Take[StringSplit[vers, "."], UpTo[2]], "."]]<>fileExtension,
							"python"<>StringJoin[Riffle[Take[StringSplit[vers, "."], UpTo[3]], "."]] <>fileExtension
						}
					]
				]
			],

		(*the program file is what is actually run as a repl - this is a function that takes as it's argument the version string*)
		(*specified by the user*)
		"ProgramFileFunction"->Function[{version},PacletManager`PacletResource["ExternalEvaluate_Python","PythonREPL"]],

		(*the dependency test file is used to test whether an installation of the language has the required libraries available - this will always include at least zmq and a form of json*)
		"DependencyTestFile"->PacletManager`PacletResource["ExternalEvaluate_Python","PythonZMQTest"],

		(*we use the -u flag to stop python from buffering output so that Mathematica can read from the process immediately*)
		(*in addition we provide the location of where to find the wolframclient package that can be used to export python types to wl*)
		"ScriptExecCommandFunction"->Function[
			{uuid,exec,file,opts},
			{exec, "-u", file, "start_externalevaluate", 
				"--path", Lookup[PacletInformation["WolframClientForPython"], "Location"], 
				"--installpath", $InstallationDirectory
			}
		],

		(*command that returns a code string that can be used to run a file inside the repl*)
		(*we are using an utility function that reads the source code and keeps the source for debugging in case a failure with a traceback is returned*)
		"RunFileInSessionFunction" -> Function[
			{file},
			"from wolframclient.utils.api import externalevaluate\nexternalevaluate.execute_from_file("<> ToString[file, InputForm] <>", locals())"
		],		

		(*icon is just an image to display with the summary box*)
		"Icon"->$PythonIcon,

		(*we can avoid any customization of the --version output screen by just evaluating the python code that tells us the version - this code works in all python versions*)
		(*for example anaconda will customize the --version so it's not easily parsable, but this sidesteps all of that*)
		"VersionExecCommandFunction"-> Function[{exec},{exec, "-c", "import sys; v=sys.version_info; print('%s.%s.%s' % (v.major, v.minor, v.micro))"}],

		(*make an automatic paclet name out of this*)
		"PacletName"->Automatic,

		"SessionProlog"->None,
		
		"DeserializationFunction" -> BinaryDeserialize,
		"SerializationFunction"   -> BinarySerialize
	|>
]

(*also add Python-NumPy*)
ExternalEvaluate`RegisterSystem[
	"Python-NumPy",
	"Python",
	(*only thing different is the default session prolog and the dependency checking file*)
	<|
		"SessionProlog" -> "import numpy",
		"DependencyTestFile"->PacletManager`PacletResource["ExternalEvaluate_Python","PythonNumpyZMQTest"]
	|>
]

(*also add Python-PIL*)
ExternalEvaluate`RegisterSystem[
	"Python-PIL",
	"Python",
	(*only thing different is the default session prolog and the dependency checking file*)
	<|
		"SessionProlog" -> "import PIL\nfrom PIL import Image",
		"DependencyTestFile"->PacletManager`PacletResource["ExternalEvaluate_Python","PythonPILZMQTest"]
	|>
]

(* This is a hidden mode for Python ExternalEvaluate in which all uncaught errors are writen
to disc, or to the file specified by ExternalEvaluatePython`$DebugLogFilePath 
Optional argument is --excepthook eventually followed by a filepath specifying the log file.
Otherwise root logger is used. *)

ExternalEvaluatePython`EnablePythonDebug[logfile_String] := ExternalEvaluate`RegisterSystem[
	"Python-Debug",
	"Python",
	<|
		"ScriptExecCommandFunction"->Function[
			{uuid,exec,file,opts},
			{exec, "-u", file, "start_externalevaluate", "--path", Lookup[PacletInformation["WolframClientForPython"], "Location"], "--excepthook", logfile, "--installpath", $InstallationDirectory}
		]
	|>
];

ExternalEvaluate`Plugins`Python`ImportExport`importPythonString::usage = "imports a python literal expression string as an expression.";

ExternalEvaluate`Plugins`Python`ImportExport`importPythonExpression::usage = "imports a python literal expression from the stream and returns the imported elements."

ExternalEvaluate`Plugins`Python`ImportExport`exportPythonExpression::usage = "exports a WL expression as a python literal string.";

(*EXPORT FUNCTIONS*)

(*for now, make None and Null just go to the same None object in Python*)
encodeExprToPython[None] := "None"
encodeExprToPython[Null] := "None"
encodeExprToPython[Missing[___]] := "None"
encodeExprToPython[True] := "True"
encodeExprToPython[False] := "False"
encodeExprToPython[list_List?ListQ] := "[" <> StringJoin[Riffle[encodeExprToPython /@ list, ", "]] <> "]"
encodeExprToPython[assoc_Association?AssociationQ] := StringJoin[
	"{",
	Riffle[
		KeyValueMap[encodeExprToPython[#1] <> ": " <> encodeExprToPython[#2] &, assoc],
		", "
	],
	"}"
];
encodeExprToPython[str_String?StringQ] := ToString[str, CForm];
encodeExprToPython[num_?NumberQ] := ToString[num, CForm];


encodeExprToPython[any___] := Throw[any, "invalid"]

ExternalEvaluate`Plugins`Python`ImportExport`exportPythonExpression[stream_OutputStream,expr_,opts___] :=
	Block[
		{encodeRes = Catch[encodeExprToPython[Last[expr]], _String]},
		If[
			! StringQ[encodeRes],
			(*THEN*)
			(*it's not a string and wasn't encoded properly, so issue message and fail*)
			Message[Export::invalidpythonexpr, encodeRes];
			$Failed,
			(*ELSE*)
			(*encoding was successful, so write it out to the stream*)
			WriteString[stream,encodeRes];
			encodeRes
		]
	]


(* IMPORT FUNCTIONS*)


(*the Import framework will return to us an InputStream, so just read through all of it and make a string out of it*)
(*we also need to return it as a list of rules, with the Element going to that data, but we only have the "Data" element, so we just return that*)
(*this function is what import framework calls - so it needs to use *)
Options[ExternalEvaluate`Plugins`Python`ImportExport`importPythonExpression] = {"CacheSession"->False}
ExternalEvaluate`Plugins`Python`ImportExport`importPythonExpression[stream_InputStream,opts:OptionsPattern[]] :=
	"Data"->ExternalEvaluate`Plugins`Python`ImportExport`importPythonString[ReadString[stream],opts,"CacheSession"->False]

(*only option to importPythonString is whether to use a cached session or not*)
(*not using a cached session results in no leftover ExternalSessionObject's or Stream's after it's done*)
(*but using a cached session is much faster, so we use that with ExternalEvaluate directly*)
Options[ExternalEvaluate`Plugins`Python`ImportExport`importPythonString] = {"CacheSession"->True}

(*sometimes ExternalEvaluate will give us a Null*)
(*so we should replace that with None, which is Python's equivalent*)
ExternalEvaluate`Plugins`Python`ImportExport`importPythonString[Null,opts:OptionsPattern[]]:=ExternalEvaluate`ImportExport`Symbols`importPythonString["None",opts]

ExternalEvaluate`Plugins`Python`ImportExport`importPythonString[inStr_String,opts:OptionsPattern[]]:=
	Block[
		{sessionspec},
		(*attempt to check on what specification we should use for the session to import the string*)
		sessionspec = If[
			TrueQ[OptionValue["CacheSession"]],
			(*we should use a cached session stored inside $ImporterPythonSession*)
			If[
				ValueQ[$ImporterPythonSession] && KeyExistsQ[First[$ImporterPythonSession]]@ExternalEvaluate`Private`$Links,
				(*THEN*)
				(*the cached importer session exists, use it*)
				$ImporterPythonSession,
				(*ELSE*)
				(*doesn't exist or isn't valid, so need to start one up*)
				(*first check if there are any evaluators available to use*)
				If[
					FailureQ[ExternalEvaluate`Private`resolveLangInstall["Python","*"]],
					(*we don't have any available evaluators, issue a message and fail*)
					Message[Import::nopythonevals,inStr];
					Return[$Failed],
					(*we have at least one evaluator we can use - so this will use the first found one as the cached session*)
					$ImporterPythonSession = StartExternalSession["Python"]
				]
			],
			(*ELSE*)
			(*don't cache anything, so use one-shot version*)
			(*first check if we have any available Python installations - as if we don't then we have to fail*)
			If[
				FailureQ[ExternalEvaluate`Private`resolveLangInstall["Python","*"]],
				(*we don't have any available evaluators, issue a message and fail*)
				Message[Import::nopythonevals,inStr];
				Return[$Failed],
				(*we have at least one evaluator we can use - so this will use the first found one*)
				"Python"
			]
		];
		ExternalEvaluate[
			(*use the session spec as determined above*)
			sessionspec,
			inStr
		]
	]

ExternalEvaluate`Private`handleDefinedFunction["Python"] = handleDefinedFunctionPython;
ExternalEvaluate`Private`handleDefinedFunction["Python-PIL"] = handleDefinedFunctionPython;
ExternalEvaluate`Private`handleDefinedFunction["Python-NumPy"] = handleDefinedFunctionPython;
handleDefinedFunctionPython[session_ExternalSessionObject, functionDef_String] := Block[
	{x, defResult, func, funcName, system},

	funcName = StringCases[functionDef, "def " ~~ Shortest[x__] ~~ "(" -> x];
	If[MatchQ[funcName, {_String,___}],
		funcName = First[funcName] ,
		system = session["System"];
		If[system === "Python" && StringContainsQ[functionDef, "lambda"],
			Return[
				ExternalFunction[<|
					"Name"->StringJoin["(", functionDef, ")"],
					"Arguments"->None,
					"System"->system,
					"BuiltIn"->False
				|>]
			]
		];
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
		Message[ExternalFunction::invlDef]
	];
	func
];

ExternalEvaluate`Private`defFunctionIdentifier["Python"] = defFunctionIdentifierPython;
ExternalEvaluate`Private`defFunctionIdentifier["Python-PIL"] = defFunctionIdentifierPython;
ExternalEvaluate`Private`defFunctionIdentifier["Python-NumPy"] = defFunctionIdentifierPython;
defFunctionIdentifierPython [funcName_]:= TrueQ[StringContainsQ[funcName, {"\n", ":"}]];

End[]

EndPackage[]
