Package["NumericArrayUtilities`"]
PackageImport["GeneralUtilities`"]


DeclareLibraryFunction[byteArrayToNumericArray, 
	"general_utils_ByteArrayToNumericArray",
	{
		{"ByteArray", "Constant"},	(* byte array *)
		{"NumericArray", "Constant"}	(* out numeric array *)
	},
	"Void"
]

DeclareLibraryFunction[numericArrayToByteArray, 
	"general_utils_NumericArrayToByteArray",
	{
		{"NumericArray", "Constant"},
		Integer
	},
	"ByteArray"
]

(*----------------------------------------------------------------------------*)
PackageExport["NumericArrayDataByteCount"]

SetUsage @ "
NumericArrayDataByteCount[NumericArray[$$]] returns the byte count of the underlying \
data of NumericArray[$$]. Unlike ByteCount, the bytes of metadata such as dimensions etc. of NumericArray \
are not counted. Thus NumericArrayDataByteCount[NumericArray[$$]] < ByteCount[NumericArray[$$]].
"

$NumericArrayBytes = <|
	"Integer8" -> 1,
	"UnsignedInteger8" -> 1,
	"Integer16" -> 2,
	"UnsignedInteger16" -> 2,
	"Integer32" -> 4,
	"UnsignedInteger32" -> 4,
	"Integer64" -> 8,
	"UnsignedInteger64" -> 8,
	"Real16" -> 2, (* not here yet *)
	"Real32" -> 4,
	"Real64" -> 8,
	"Complex64" -> 8,
	"Complex128" -> 16
|>

(* one day, other types might be added, eg boole. Panic *)
General::nauinvnumtype = "The NumericArray type `` is unsupported. Please add support."


iNumericArrayDataByteCount[x_NumericArray] := CatchFailureAsMessage @ Scope[
	flatlen = Times @@ Dimensions[x];
	type = NumericArrayType[x];
	singleSize = Lookup[
		$NumericArrayBytes, type,
		ThrowFailure[General::nauinvnumtype, type]
	];
	singleSize * flatlen
]


NumericArrayDataByteCount::invin = "Invalid input."

NumericArrayDataByteCount[x_NumericArray] := 
	CatchFailureAsMessage @ iNumericArrayDataByteCount[x]

NumericArrayDataByteCount[___] := (Message[NumericArrayDataByteCount::invin]; $Failed)

(*----------------------------------------------------------------------------*)
PackageExport["ByteArrayToNumericArray"]

SetUsage @ "
ByteArrayToNumericArray[ByteArray[$$], type$, dims$] returns a NumericArray[$$] \
with type name type$ and dimensions dims$ using the ByteArray[$$] data reinterpreted \
as being of type type$. The number of bytes required to represent a NumericArray \
with type$ and dims$ must equal the length of the ByteArray[$$].
"

ByteArrayToNumericArray::invbc = 
	"The output NumericArray and input ByteArray have incompatible byte count: `` versus ``."
ByteArrayToNumericArray::invin = "Invalid input."

ByteArrayToNumericArray[byte_ByteArray, type_String, dims_List] := CatchFailureAsMessage @ Scope[
	out = Developer`AllocateNumericArray[type, dims];
	If[Not @ NumericArrayQ[out], Return[$Failed]];
	outBytes = iNumericArrayDataByteCount[out];
	If[outBytes =!= Length[byte], 
		ThrowFailure[ByteArrayToNumericArray::invbc, outBytes, Length[byte]]
	];
	NAUInvoke[byteArrayToNumericArray, byte, out];
	out
]

ByteArrayToNumericArray[___] := (Message[ByteArrayToNumericArray::invin]; $Failed)

(*----------------------------------------------------------------------------*)
PackageExport["NumericArrayToByteArray"]

SetUsage @ 
"NumericArrayToByteArray[NumericArray[$$]] returns a ByteArray[$$] with the \
underlying numeric data of NumericArray[$$] reinterpreted as an array of bytes. \
The type and dimensionality information of NumericArray[$$] is not preserved.
"

NumericArrayToByteArray::invin = "Invalid input."

NumericArrayToByteArray[x_NumericArray] := CatchFailureAsMessage @ Scope[
	bytes = iNumericArrayDataByteCount[x];
	NAUInvoke[numericArrayToByteArray, x, bytes]
]

NumericArrayToByteArray[___] := (Message[NumericArrayToByteArray::invin]; $Failed)
