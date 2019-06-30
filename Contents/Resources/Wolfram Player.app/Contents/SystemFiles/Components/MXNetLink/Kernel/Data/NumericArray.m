Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]

(******************************************************************************)

PackageScope["mxlCopyArrayToArray"]
PackageScope["mxlCopyArrayToArrayPermuted"]

mxlDeclare[mxlCopyArrayToArray, {"Array", "Array", "Integer", "Integer", "Boolean"}]
mxlDeclare[mxlCopyArrayToArrayPermuted, {"Array", "Array", "IntegerVector"}]

(******************************************************************************)

PackageExport["CreateConstantNumericArray"]

SetUsage @ "
CreateConstantNumericArray[dims$, value$] creates a default-typed numeric array of shape dims$ and initial value value$.
CreateConstantNumericArray[dims$, value$, 'type$'] users a specific NumericArray type."

CreateConstantNumericArray[dims_, value_, type_:Automatic] /; (Times @@ dims) < 4000 :=
	 NumericArray[ConstantArray[value, dims], FromDataTypeCode @ type];

CreateConstantNumericArray[dims_, value_, type_:Automatic] := Scope[
	arr = Developer`AllocateRawArray[FromDataTypeCode @ type, dims];
	mxlArraySetConstant[arr, N @ value];
	arr
];

(******************************************************************************)

PackageExport["ArraySetConstant"]

mxlDeclare[mxlArraySetConstant, {"Array", "Real"}]

ArraySetConstant[array_, value_] := mxlArraySetConstant[array, N @ value];

(******************************************************************************)

PackageExport["ArrayStatistics"]

mxlDeclare[mxlArrayStatistics, "Array", "RealVector"]

ArrayStatistics[arr_] := mxlArrayStatistics[arr];

(******************************************************************************)

PackageExport["CreateEmptyArray"]

SetUsage @ "
CreateEmptyArray[dims$] creates a real-valued packed array of shape dims$ and uninitialized contents.
CreateEmptyArray[dims$, intValued$] specifies whether to make it integer-valued or real-valued."

mxlDeclare[mxlCreateEmptyArray$int, {"IntegerVector", "Boolean"}, "IntegerTensor"]
mxlDeclare[mxlCreateEmptyArray$real, {"IntegerVector", "Boolean"}, "RealTensor"]

CreateEmptyArray[dims_, value_:0.0, intValued_:False] := 
	If[intValued, mxlCreateEmptyArray$int, mxlCreateEmptyArray$real][dims, intValued]

(******************************************************************************)

PackageExport["CheckNotAbnormal"]

SetUsage @ "
CheckNotAbnormal[array] checks a packed array for presence of NaNs or Infs, and calls \
$AbnormalValueCallback[array] if they exist."

DeclarePostloadCode[
General::netnan = "A floating-point overflow, underflow, or division by zero occurred while evaluating the net."
]

PackageExport["$AbnormalValueCallback"]

$AbnormalValueCallback = Function[ThrowFailure["netnan"]];

mxlDeclare[mxlAbnormalArrayQ, "Array", "Boolean"]
mxlDeclare[mxlAbnormalRealQ, "Real", "Boolean"]

CheckNotAbnormal[e_Real ? mxlAbnormalRealQ] := $AbnormalValueCallback[e];
CheckNotAbnormal[e_List ? mxlAbnormalArrayQ] := $AbnormalValueCallback[e]
CheckNotAbnormal[e_NumericArray ? mxlAbnormalArrayQ] := $AbnormalValueCallback[e]
CheckNotAbnormal[e_] := e

(******************************************************************************)

PackageScope["AbnormalValueQ"]

AbnormalValueQ[e_Real] := mxlAbnormalRealQ[e];
AbnormalValueQ[e_List | e_NumericArray] := mxlAbnormalArrayQ[e];
AbnormalValueQ[e_] := False

(******************************************************************************)

PackageScope["PackedOrNumericArrayQ"]

PackedOrNumericArrayQ[list_List] := Developer`PackedArrayQ[list]
PackedOrNumericArrayQ[na_NumericArray] := NumericArrayQ[na]
PackedOrNumericArrayQ[_] := False

PackageScope["arrayDimensions"]

arrayDimensions[e_, args___] := Dimensions[e, args, AllowedHeads -> {List, NumericArray}];
