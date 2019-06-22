(*******************************************************************************

Common functions + Global Variables

*******************************************************************************)

Package["DAALLink`"]

PackageImport["GeneralUtilities`"]


(*----------------------------------------------------------------------------*)
(****** Define Library Functions ******)

PackageExport["DAALGetLastError"]
DAALGetLastError := DAALGetLastError = 
	LibraryFunctionLoad[$DAALLinkLib, "WL_GetLastError", {}, "UTF8String"];	

PackageExport["DAALGetThreadNumber"]
DAALGetThreadNumber := DAALGetThreadNumber = 
	LibraryFunctionLoad[$DAALLinkLib, "WL_GetThreadNum", {}, Integer];	

PackageExport["DAALSetThreadNumber"]
DAALSetThreadNumber := DAALSetThreadNumber = 
	LibraryFunctionLoad[$DAALLinkLib, "WL_SetThreadNum", {Integer}, Null];	

daalVersionInfo := daalVersionInfo = 
	LibraryFunctionLoad[$DAALLinkLib, "WL_DAAL_Version", LinkObject, LinkObject];

(*----------------------------------------------------------------------------*)
(****** Library Loader ******)

PackageExport["DAALSupportedPlatformQ"]
DAALSupportedPlatformQ[] := 
If[MemberQ[{"Linux-x86-64", "MacOSX-x86-64", "Windows-x86-64"}, $SystemID],
	True,
	False
]

$DAALLinkBaseDir = FileNameTake[$InputFileName, {1, -3}];

PackageScope["$LibraryResources"]
$LibraryResources = FileNameJoin[{
	ParentDirectory[DirectoryName @ $InputFileName], 
	"LibraryResources", 
	$SystemID
}];

PackageScope["$DAALLinkLib"]

$DAALLinkLib = Switch[$OperatingSystem,
	"MacOSX",
		FileNameJoin[{$LibraryResources, "DAALLink.dylib"}],
	"Windows",
		FileNameJoin[{$LibraryResources, "DAALLink.dll"}],
	"Unix",
		FileNameJoin[{$LibraryResources, "DAALLink.so"}]
]


$DAALLibrariesLoaded = False;
(* load libraries doesn't do anything on unsupported platforms *)
LoadDependencyLibs[] := If[!$DAALLibrariesLoaded && DAALSupportedPlatformQ[],
	LibraryLoad[First @ FileNames["*tbbmalloc*", $LibraryResources]];
	LibraryLoad[First @ FileNames["*tbb.*", $LibraryResources]];
	(* CreateManagedLibraryExpression requires the loading of a single 
	   librarylink function to work. safest to make a single library call during
	   paclet loading to guarantee CreateManagedLibraryExpression will work *)
	DAALGetThreadNumber[];
	$DAALLibrariesLoaded = True;
];

LoadDependencyLibs[];


(*----------------------------------------------------------------------------*)
General::daallibuneval = "Library function `` with args `` did not evaluate.";

PackageScope["safeLibraryInvoke"]

safeLibraryInvoke[func_, args___] :=
    Replace[
        func[args], 
        {
            _LibraryFunctionError :> DAALPanic[func],
            _LibraryFunction[___] :> ThrowFailure["daallibuneval", func[[2]], {args}]
        }
    ];

General::daalliberr = "C Function `` failed with DAAL-generated error: \n ``";

DAALPanic[f_] := Module[
	{lastError},
	lastError = DAALGetLastError[];
	If[TrueQ @ LibraryFunctionFailureQ[lastError], 
		lastError = "Unknown Error";
	];
	ThrowFailure["daalliberr", f[[2]], lastError];
]

(*----------------------------------------------------------------------------*)
PackageScope["getMLEID"]
getMLEID[x_ /; ManagedLibraryExpressionQ[x]] := ManagedLibraryExpressionID[x]
getMLEID[___] := Panic["Invalid argument to getMLEID"]

(*----------------------------------------------------------------------------*)
PackageExport["DAALVersionInfo"]

DAALVersionInfo[] := AssociationThread[
	{"MajorVersion", "MinorVersion", "UpdateVersion", "ProductStatus", "Build",
		"BuildRevision", "Name", "ProcessorOptimization"},
	daalVersionInfo[{}]
]