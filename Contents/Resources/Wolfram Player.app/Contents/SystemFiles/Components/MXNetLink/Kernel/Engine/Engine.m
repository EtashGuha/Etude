Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["SetOpenMPThreadCount"]
PackageExport["$DefaultThreadCount"]

SetUsage @ "
SetOpenMPThreadCount[n$] sets the number of OpenMP threads used by MXNet to n$."

If[!ValueQ[$DefaultThreadCount], $DefaultThreadCount = Floor[0.75 * $ProcessorCount]];

mxlDeclare[mxlMXSetNumOMPThreads, "Integer"];

SetOpenMPThreadCount[None] := Null;
SetOpenMPThreadCount[n_Integer] := mxlMXSetNumOMPThreads[n]

DeclarePostloadCode[
	MXNetLink`SetOpenMPThreadCount[MXNetLink`$DefaultThreadCount];
];

(******************************************************************************)

PackageExport["GetMXNetVersion"]

SetUsage @ "
GetMXNetVersion[] returns the MXNet version integer."

mxlDeclare[mxlMXGetVersion, {}, "Integer"]

GetMXNetVersion[] := mxlCall[mxlMXGetVersion]

(******************************************************************************)

PackageExport["MXEngineSetBulkSize"]

SetUsage @ "
MXEngineSetBulkSize[size$] set bulk execution limit with integer size$."

mxlDeclare[mxlMXEngineSetBulkSize, "Integer", "Integer"]

MXEngineSetBulkSize[size_Integer] := mxlCall[mxlMXEngineSetBulkSize, size]

(******************************************************************************)

PackageExport["MXSeedRandom"]

mxlDeclare[mxlMXRandomSeed, "Integer"];

MXSeedRandom[i_Integer] := mxlMXRandomSeed[i];

(******************************************************************************)

PackageExport["SetErrorLogMode"]

mxlDeclare[mxlSetErrorLogMode, "Integer"];

$errorLogModes = <|"Console" -> 0, "Notebook" -> 1, None -> 2|>;
SetErrorLogMode[type_] := mxlSetErrorLogMode @ Lookup[$errorLogModes, type, Panic["InvalidErrorLogMode"]];

(******************************************************************************)

PackageExport["MXNotifyShutdown"]

mxlDeclare[mxlMXNotifyShutdown, {}];

MXNotifyShutdown[] := mxlMXNotifyShutdown[];

(******************************************************************************)

PackageExport["GetManagedLibraryKeys"]

SetUsage @ "
GetManagedLibraryKeys[name$] a list of Managed Library Expression Keys for name$ in \
{\"MXExecutor\", \"NDArray\", \"MXSymbol\", \"MXOptimizer\"}. 
GetManagedLibraryKeys[] returns an association of all Managed Library Expression Keys."

mxlDeclare[mxlGetManagedLibraryKeys, "Integer", "IntegerVector"]

GetManagedLibraryKeys[name_String] := CatchFailure@Scope[
	nameInt = Switch[name,
		"MXExecutor", 0,
		"NDArray", 1,
		"MXSymbol", 2,
		"MXKVStore", 3,
		"Optimizer", 4,
		_, ThrowFailure["InvalidName"]
	];
	mxlGetManagedLibraryKeys[nameInt]
]

GetManagedLibraryKeys[] := AssociationMap[
	GetManagedLibraryKeys, 
	{"MXExecutor", "NDArray", "MXSymbol", "MXKVStore", "Optimizer"}
]

(******************************************************************************)

(* SetEnvironment was tested to take less than 10^-6 on Windows, Linux + OSX. So calling this everything doesn't matter *)
PackageExport["SetCUDAMixedPrecision"]

SetCUDAMixedPrecision[mixedQ_] := SetEnvironment["MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION" -> If[mixedQ, "1", "0"]];


(******************************************************************************)

PackageExport["GetFloatLimit"]

mxlDeclare[mxlGetFloatLimit, {"Integer", "Boolean", "Boolean"}, "Real"]

(* rename this to GetNumericLimit of ints are supported in future *)
GetFloatLimit[type_String, limit_String] := Scope[
	type = StringReplace[type, "float" -> "Real"]; (* allow mx names *)
	typeCode = $NumericArrayTypeCode @ type;
	Switch[limit,
		"LargestPositive", mxlGetFloatLimit[typeCode, True, True],
		"LargestNegative", mxlGetFloatLimit[typeCode, True, True],
		"SmallestPositive", mxlGetFloatLimit[typeCode, False, False],
		"SmallestNegative", mxlGetFloatLimit[typeCode, False, False],
		_, Panic[]
	]
]
