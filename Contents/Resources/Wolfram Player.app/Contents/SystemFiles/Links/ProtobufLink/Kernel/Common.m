(*******************************************************************************

Common: global variables plus common utility functions

*******************************************************************************)

Package["ProtobufLink`"]

PackageImport["GeneralUtilities`"]

(*----------------------------------------------------------------------------*)
(****** Global Variables ******)
(* declare some common symbols, overloaded for different objects *)
PackageScope["getMLE"]
getMLE[___] := Panic["Invalid argument to getMLE"]

PackageScope["getMLEID"]
getMLEID[x_ /; ManagedLibraryExpressionQ[x]] := ManagedLibraryExpressionID[x]
getMLEID[___] := Panic["Invalid argument to getMLEID"]


PackageScope["$LibraryResources"]
$LibraryResources = FileNameJoin[{
	ParentDirectory[DirectoryName @ $InputFileName], 
	"LibraryResources", 
	$SystemID
}];

$ext = <|"MacOSX" -> "dylib", "Windows" -> "dll", "Unix" -> "so"|>

PackageScope["$ProtoLinkLib"]

$ProtoLinkLib = FileNameJoin[{
	$LibraryResources, 
	StringJoin["ProtobufLink.", Lookup[$ext, $OperatingSystem]]
}]


PackageExport["ProtobufGetVersion"]

ProtobufGetVersion = LibraryFunctionLoad[$ProtoLinkLib, "WL_GetProtobufVersion", 
	{},
	Integer		
]

(*----------------------------------------------------------------------------*)
PackageScope["clearReturnString"]
clearReturnString = LibraryFunctionLoad[$ProtoLinkLib, "WL_ClearReturnString", 
	{},
	"Void"		
]

PackageScope["clearErrorString"]
clearErrorString = LibraryFunctionLoad[$ProtoLinkLib, "WL_ClearErrorString", 
	{},
	"Void"
]

PackageScope["getErrorString"]
getErrorString = LibraryFunctionLoad[$ProtoLinkLib, "WL_GetErrorString", 
	{},
	"UTF8String"		
]

PackageScope["freeMessageMLE"]
iFreeMessageMLE = LibraryFunctionLoad[$ProtoLinkLib, "WL_Free_Message", 
	{Integer},
	"Void"		
]
freeMessageMLE[m_ProtobufMessage] := CatchFailure @ 
	safeLibraryInvoke[iFreeMessageMLE, getMLEID[m]]



(*----------------------------------------------------------------------------*)
PackageExport["ProtobufEnableLogger"]

SetUsage[ProtobufEnableLogger,
"ProtobufEnableLogger[] enables the Protobuf library to log to stdout via google::protobuf::SetLogHandler."
]

ProtobufEnableLogger = LibraryFunctionLoad[$ProtoLinkLib, "WL_EnableLogger", 
	{},
	"Void"		
]

(*----------------------------------------------------------------------------*)
(* Note: this is a callback function called by mathlink to load NumericArrays into frontend without
MathLink serialization
 *)

PackageScope["$popNumericArrayFromStore"]
$popNumericArrayFromStore = LibraryFunctionLoad[$ProtoLinkLib, "WL_PopNumericArray", 
	{Integer},
	"NumericArray"		
]

(* we are overloading here. Queit it to remove the warning about this *)
PackageScope["$popByteArrayFromStore"]
$popByteArrayFromStore = Quiet @ LibraryFunctionLoad[$ProtoLinkLib, "WL_PopNumericArray", 
	{Integer},
	"ByteArray"		
]

(*----------------------------------------------------------------------------*)
General::protobuflibuneval = "Library function `` with args `` did not evaluate.";

PackageScope["safeLibraryInvoke"]

safeLibraryInvoke[func_, args___] :=
    Replace[
        func[args], 
        {
            _LibraryFunctionError :> ProtobufPanic[func],
            _LibraryFunction[___] :> ThrowFailure["protobuflibuneval", func[[2]], {args}]
        }
    ];

General::protobufliberr = "C Function `` failed: \"``\"";

ProtobufPanic[f_] := Module[
	{lastError},
	lastError = getErrorString[];
	clearErrorString[];
	If[TrueQ @ LibraryFunctionFailureQ[lastError], 
		lastError = "Unknown Error";
	];
	ThrowFailure["protobufliberr", f[[2]], lastError];
]

(*----------------------------------------------------------------------------*)
PackageScope["fileConform"]

General::protobufnffil = "The file `` does not exist.";
fileConform[file_String] := (
	If[!FileExistsQ[file],
		ThrowFailure["protobufnffil", file];
	];
	ExpandFileName[file]
)

fileConform[File[file_]] := fileConform[file]

General::protobufinvfile = "Expression `` is not a String or File[...] object."
fileConform[file_] := ThrowFailure["protobufinvfile", file];

