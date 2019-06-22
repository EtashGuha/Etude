Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["MXGetLastError"]

SetUsage @ "
MXGetLastError[] returns a string that describes the last message by MXNet."

mxlDeclare[mxlMXGetLastError, {}, "String"];

MXGetLastError[] := mxlMXGetLastError[];

(******************************************************************************)

PackageExport["$MXLibraryErrorHandler"]

PackageExport["MXNetGenericPanic"]

$MXLibraryErrorHandler = MXNetGenericPanic;
MXNetGenericPanic[args___] := Panic["MXNetError", "``", args];


PackageScope["MXLibraryError"]

DeclarePostloadCode[
General::gpumemex = "Computation aborted: GPU memory exhausted.";
General::mxwpsupp = "Computation aborted: one or more layers is not compatible with WorkingPrecision->``.";
General::nogpusupp = "GPU support is not available for this platform.";
General::mxoldgpu = "Your GPU does not support the operations required to evaluate this network.";
]

MXLibraryError[] := Scope[
	errstr = MXGetLastError[];
	If[StringContainsQ[errstr, "libWolframEngine"],
		errstr = StringRiffle[
			Discard[StringSplit[errstr, "\n"], StringContainsQ["libWolframEngine"]],
			"\n"
		]
	];
	If[StringContainsQ[errstr, "float16" | "float64" | "Unsupported data type"], ThrowFailure["mxwpsupp", FromDataTypeCode[$DefaultDataTypeCode]]];
	If[StringContainsQ[errstr, "cudaMalloc failed"], ThrowFailure["gpumemex"]];
	If[StringContainsQ[errstr, "ompile with USE_CUDA"], ThrowFailure["nogpusupp"]];
	If[StringContainsQ[errstr, "no kernel image"], ThrowFailure["mxoldgpu"]];
	(* ^ see bug 362494... also TODO: remove/improve this when we can check compute capability of GPU *)
	$MXLibraryErrorHandler[errstr];
];

(******************************************************************************)

PackageExport["$MXNetLogger"]

$MXNetLogger = Hold;
