(*******************************************************************************

Common: global variables plus common utility functions

*******************************************************************************)

Package["ExternalStorage`"]

PackageImport["GeneralUtilities`"]

(*----------------------------------------------------------------------------*)
(* declare some common symbols, overloaded for different objects *)
PackageScope["getMLE"]
getMLE[___] := Panic["Invalid argument to getMLE"]

PackageScope["getMLEID"]
getMLEID[x_ /; ManagedLibraryExpressionQ[x]] := ManagedLibraryExpressionID[x]
getMLEID[___] := Panic["Invalid argument to getMLEID"]

(*----------------------------------------------------------------------------*)
(****** Global Variables ******)

$S3LinkBaseDir = FileNameTake[$InputFileName, {1, -4}];

PackageScope["$LibraryResources"]
$LibraryResources = FileNameJoin[{
	$S3LinkBaseDir, 
	"LibraryResources", 
	$SystemID
}];

PackageScope["$S3LinkLib"]

$S3LinkLib = Switch[$OperatingSystem,
	"MacOSX", 
		FileNameJoin[{$LibraryResources, "S3Link.dylib"}],
	"Windows",
		FileNameJoin[{$LibraryResources, "S3Link.dll"}],
	"Unix",
		FileNameJoin[{$LibraryResources, "S3Link.so"}]
]

(* For windows: we need to explicitly load dependencies, not rely on autoloading *)
If[$OperatingSystem === "Windows",
	LibraryLoad @ FileNameJoin[{$LibraryResources, "S3Link.dll"}];
];


(*----------------------------------------------------------------------------*)
PackageScope["LibraryFunctionFailureQ"]

LibraryFunctionFailureQ[call_] := (Head[call] === LibraryFunctionError)

(*----------------------------------------------------------------------------*)
PackageScope["s3Declare"]

SetAttributes[s3Declare, HoldFirst];
s3Declare[symbol_, args_] := s3Declare[symbol, args, "Void"];
s3Declare[symbol_Symbol, args_, ret_] := (
	Clear[symbol]; 
	With[{name = SymbolName[symbol]},
		Set[symbol, Replace[
			LibraryFunctionLoad[$S3LinkLib, name, args, ret],
			Except[_LibraryFunction] :> (
				Message[symbol::s3liberr1];
				Function[ThrowFailure[symbol::s3liberr1]]
			)
		]]
	]
);

General::s3liberr1 = "Error loading LibraryLink function.";

(*----------------------------------------------------------------------------*)
PackageScope["s3GetLastError"]
s3Declare[s3GetLastError, {}, "UTF8String"]

PackageScope["s3EraseLastError"]
s3Declare[s3EraseLastError, {}]

(*----------------------------------------------------------------------------*)
s3Declare[s3StartLogger, {"UTF8String"}]
s3Declare[s3StopLogger, {}]

PackageExport["S3StartLogger"]
S3StartLogger[name_String] := CatchFailureAndMessage @ s3Call[s3StartLogger, name]
S3StartLogger[] := S3StartLogger["s3_log_"]

PackageExport["S3StopLogger"]
S3StopLogger[] := CatchFailureAndMessage @ s3Call[s3StopLogger]

(*----------------------------------------------------------------------------*)
General::s3libuneval = "Library function `` with args `` did not evaluate.";

PackageScope["s3Call"]

s3Call[func_, args___] :=
    Replace[
        func[args], 
        {
            _LibraryFunctionError :> S3Panic[func],
            _LibraryFunction[___] :> ThrowFailure["s3libuneval", func[[2]], {args}]
        }
    ];

General::s3liberr2 = "Error from S3 C++ SDK: \"``\"";

S3Panic[f_] := Module[
	{lastError},
	lastError = s3GetLastError[];
	s3EraseLastError[];
	If[TrueQ @ LibraryFunctionFailureQ[lastError], 
		lastError = "Unknown Error";
	];
	ThrowFailure["s3liberr2", lastError];
]

(*----------------------------------------------------------------------------*)
PackageScope["S3Success"]

S3Success[f_] :=
Success[
	ToString[f],
	<|
		"MessageTemplate" :> "Operation completed.",
		"TimeStamp" -> DateString[]
	|>
]

(*----------------------------------------------------------------------------*)
PackageScope["fileConform"]

General::s3nffil = "The file `` does not exist.";
fileConform[file_String] := (
	If[!FileExistsQ[file],
		ThrowFailure["s3nffil", file];
	];
	ExpandFileName[file]
)

fileConform[File[file_]] := fileConform[file]

General::s3invfile = "Expression `` is not a String or File[...] object."
fileConform[file_] := ThrowFailure["s3invfile", file];
