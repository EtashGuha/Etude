
BeginPackage["LLVMLink`LLVMLoader`"]



Begin["`Private`"]


Needs["LLVMLink`"]


moduleDirectory = FileNameJoin[
	Drop[
		FileNameSplit[
			FindFile["LLVMLink`"]
		], -2
	]
];

Options[ InstallLLVM] =
	{
		"LLVMLinkLibraryLocation" -> Automatic,
		"LLVMLocation" -> Automatic
	}

initializedQ := False

InstallLLVM[ opts:OptionsPattern[]] := Module[
	{library, canary},
	If[!TrueQ[initializedQ],
		library = OptionValue["LLVMLocation"];
		If[ StringQ[library],
			SetLLVMLocation[library],
				
			SetLLVMLocation[FileNameJoin[{moduleDirectory, "LLVMResources",$SystemID}]]
		];
		library = OptionValue["LLVMLinkLibraryLocation"];
		If[ StringQ[library],
			SetLLVMLinkLibraryLocation[library],
			SetLLVMLinkLibraryLocation[FileNameJoin[{moduleDirectory, "LibraryResources", $SystemID}]]];
		loadLibraries[];

		(*
		try loading the first function that is typically called
		this will let us know if anything goes wrong with loading functions
		*)
		canary = LibraryFunctionLoad[LLVMLibraryName[], "LLVMLink_LLVMInitializeAllTargetInfos_Wrapper", {}, "Void"];
		If[FailureQ[canary],
			Throw[{"Could not load library LLVMLink", canary}]
		];

		initializedQ = True;
	];
	<|"LLVMLocation" -> $LLVMLocation,
		"LLVMToolResourcesPath" -> $LLVMToolResourcesPath,
		"LLVMLibraryResourcesPath" -> $LLVMLibraryResourcesPath,
		"LLVMLinkLibraryLocation" -> $LLVMLinkLLVMLibraryPath
	|> 
];


SetLLVMLinkLibraryLocation[dir_] :=
	$LLVMLinkLLVMLibraryPath = dir


SetLLVMLocation[dir_] :=
	(
	$LLVMLocation = dir;
	LLVMLink`$LLVMDirectory = dir;
	$LLVMToolResourcesPath = FileNameJoin[{dir, "bin"}];
	$LLVMLibraryResourcesPath = FileNameJoin[{dir, "lib"}];
	);


loadLibraries[] := Module[
	{res},
	PrependTo[$LibraryPath, $LLVMLibraryResourcesPath];
	PrependTo[$LibraryPath, $LLVMLinkLLVMLibraryPath];
	(*
	LLVM is statically linked on Windows, so no other loading is needed
	*)
	If[$OperatingSystem =!= "Windows",
		res = LibraryLoad["libLLVM"];
		If[FailureQ[res],
			Throw[{"Could not load library LLVM", LibraryLink`$LibraryError}]
		];
	];
	res = LibraryLoad["LLVMLink"];
	If[FailureQ[res],
		Throw[{"Could not load library LLVMLink", LibraryLink`$LibraryError}]
	];
];





End[]

EndPackage[]
