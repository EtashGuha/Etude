(* All loading of the paclet's Wolfram Language code should go through this file. *)

(* All loading of the paclet's Wolfram Language code should go through this file. *)

(* Print["In streaming loader"] *)

Streaming`StreamingMap;
Streaming`StreamingDataset;

Streaming`Private`autoloadSymbols = {
	"Streaming`StreamingMap", 
	"Streaming`StreamingDataset"
}

PacletManager`Package`loadWolframLanguageCode["Streaming", "Streaming`", DirectoryName[$InputFileName], "ModuleLoader.m",
           "AutoUpdate" -> True,
           "AutoloadSymbols" -> Streaming`Private`autoloadSymbols,
           "HiddenImports" -> {"TypeSystem`", "ResourceLocator`"}
];


If[!TrueQ @ System`Private`$buildingMX,
	Unprotect[ExampleData];
	ExampleData["Streaming"] = {{"Streaming", "StreamingMapCSVDataset"}};
	DownValues[ExampleData] =
		Module[{pos},
			pos =  Position[
				DownValues[ExampleData], 
				def_/;!FreeQ[def, "Dataset"] && FreeQ[def, DataPaclets`ExampleDataDump`$Keys],
				1
			];
			Insert[
				DeleteCases[DownValues[ExampleData], def_/;!FreeQ[def, "Streaming"]],
				HoldPattern[ExampleData[{"Streaming", key_}, prop___]] :> 
					Module[{allowOutput = True, output},
						output = Streaming`Examples`StreamingExample[key, prop];
						allowOutput = output =!= $Failed;
						output /; allowOutput
					],
				If[pos =!= {}, First @ First @ pos, -5]
			]
		];		
	Protect[ExampleData];
]

