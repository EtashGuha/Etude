
BeginPackage["MUnit`"]

`Information`$VersionNumber = 1.6

`Information`$ReleaseNumber = 0

`Information`$Version = "MUnit Version 1.6.0"

Begin["`Package`"]

$MUnitDirectory

End[]

AppendTo[$ContextPath, "MUnit`Package`"]

Begin["`MUnit`"]

$MUnitDirectory = DirectoryName[$InputFileName] 

$implementationFiles =
	{
		ToFileName[{$MUnitDirectory, "Kernel"}, "Buttons.m"],
		ToFileName[{$MUnitDirectory, "Kernel"}, "Formatting.m"],
		ToFileName[{$MUnitDirectory, "Kernel"}, "Loggers.m"],
		ToFileName[{$MUnitDirectory, "Kernel"}, "Messages.m"],
		ToFileName[{$MUnitDirectory, "Kernel"}, "Notebooks.m"],
		ToFileName[{$MUnitDirectory, "Kernel"}, "Palette.m"],
		ToFileName[{$MUnitDirectory, "Kernel"}, "Test.m"],
		ToFileName[{$MUnitDirectory, "Kernel"}, "TestRun.m"],
		ToFileName[{$MUnitDirectory, "Kernel"}, "WRI.m"]
	} ~Join~ {FileNameJoin[{$MUnitDirectory, "Kernel", "VerificationTest.m"}]}
	(*If[TrueQ[System`$CloudEvaluation] || TrueQ[MUnit`Package`$UseVerificationTest],
		{FileNameJoin[{$MUnitDirectory, "Kernel", "VerificationTest.m"}]},
		{}
	]*)

processDecls[file_] :=
	Module[{strm, e, moreLines = True},
		strm = OpenRead[file];
		If[Head[strm] =!= InputStream,
			Return[$Failed]
		];
		While[moreLines,
			e = Read[strm, Hold[Expression]];
			ReleaseHold[e];
			If[e === $Failed || MatchQ[e, Hold[_End]],
				moreLines = False
			]
		];
		Close[file]
	]

End[]

Scan[`MUnit`processDecls, `MUnit`$implementationFiles]

Scan[Get, `MUnit`$implementationFiles]

(*
Add appropriate handlers.
A problem with using MessageTextFilter is that it is "top-level",
it doesn't get called inside of Quiet 
*)
If[ListQ["MessageTextFilter" /. Internal`Handlers[]],
	Internal`AddHandler["MessageTextFilter", MUnitMessageHandler]
]

If[ListQ["GetFileEvent" /. Internal`Handlers[]],
	Internal`AddHandler["GetFileEvent", MUnitGetFileHandler]
]

EndPackage[]
