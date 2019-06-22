Package["MXNetLink`"]

PackageImport["GeneralUtilities`"]


(******************************************************************************)

PackageExport["$DeviceCode"]

$DeviceCode = <|"CPU" -> 1, "GPU" -> 2|>;

(******************************************************************************)

PackageExport["$DeviceCodeReverse"]

$DeviceCodeReverse = AssociationMap[Reverse, $DeviceCode]

(******************************************************************************)

PackageExport["$DefaultContext"]

$DefaultContext = 1; (* equivalent to {"CPU", 0} *)

(******************************************************************************)

PackageExport["ToContextCode"]

ToContextCode[Automatic] := ToContextCode[$DefaultContext];
ToContextCode[name_String] := $DeviceCode[name];
ToContextCode[{name_String, i_Integer}] := BitOr[$DeviceCode[name], BitShiftLeft[i, 2]];
ToContextCode[i_Integer] := i;
ToContextCode[_] := Panic["InvalidDeviceContext"];

(******************************************************************************)

PackageExport["FromContextCode"]

FromContextCode[1] := {"CPU", 0};
FromContextCode[2] := {"GPU", 0};
FromContextCode[code_Integer] := {$DeviceCodeReverse @ BitAnd[code, 3], BitShiftRight[code, 2]}

(******************************************************************************)

PackageExport["$DataTypeCode"]

$DataTypeCode = <|
	"Real32" -> 0, 
	"Real64" -> 1, 
	"Real16" -> 2, 
	"UnsignedInteger8" -> 3,
	"Integer32" -> 4,
	"Integer8" -> 5,
	"Integer64" -> 6
|>;

(******************************************************************************)

PackageExport["$DataTypeBytes"]

$DataTypeBytes = <|
	"Real32" -> 4, 
	"Real64" -> 8, 
	"Real16" -> 2, 
	"UnsignedInteger8" -> 1,
	"Integer32" -> 4,
	"Integer8" -> 1,
	"Integer64" -> 8
|>;

(******************************************************************************)

PackageExport["$DataCodeMXNetName"]

$DataCodeMXNetName = <|
	0 -> "float32",
	1 -> "float64",
	2 -> "float16",
	3 -> "uint8",
	4 -> "int32",
	5 -> "int8",
	6 -> "int64"
|>;

(******************************************************************************)

PackageExport["$DataTypeCodeReverse"]

$DataTypeCodeReverse = AssociationMap[Reverse, $DataTypeCode]

(******************************************************************************)

PackageExport["$DefaultDataTypeCode"]

$DefaultDataTypeCode = 0; (* Real32 *)

(******************************************************************************)

PackageExport["ToDataTypeCode"]

ToDataTypeCode[Automatic] := $DefaultDataTypeCode;
ToDataTypeCode[name_String] := Lookup[$DataTypeCode, name, badType[name]];
ToDataTypeCode[i_Integer] := i;
ToDataTypeCode[e_] := badType[e];

badType[e_] := Panic["InvalidDataType", "`` is not valid type.", e];

(******************************************************************************)

PackageExport["FromDataTypeCode"]

$dataTypeTable = Keys @ Sort @ $DataTypeCode;
FromDataTypeCode[Automatic] := Part[$dataTypeTable, $DefaultDataTypeCode + 1]
FromDataTypeCode[i_Integer] := Check[Part[$dataTypeTable, i + 1], badType[i]]
FromDataTypeCode[str_String] := str;
FromDataTypeCode[e_] := badType[e]

(******************************************************************************)

PackageExport["ToMXNetDataTypeName"]

ToMXNetDataTypeName[Automatic] := Lookup[$DataCodeMXNetName, $DefaultDataTypeCode]
ToMXNetDataTypeName[name_String] := Lookup[$DataCodeMXNetName, 
	Lookup[$DataTypeCode, name, Panic["InvalidDataType"]]]
ToMXNetDataTypeName[i_Integer] := Lookup[$DataCodeMXNetName, i, Panic["InvalidDataType"]]
ToMXNetDataTypeName[_] := Panic["InvalidDataType"]

(******************************************************************************)

PackageExport["$GradientUpdateCode"]

$GradientUpdateCode = <|None -> 0, "Write" -> 1, "InPlace" -> 2, "Add" -> 3|>;

(******************************************************************************)

PackageExport["$GradientUpdateCodeReverse"]

$GradientUpdateCodeReverse = AssociationMap[Reverse, $GradientUpdateCode]

(******************************************************************************)

PackageScope["$NumericArrayTypeCode"]

(* needs to be kept in-sync with WolframRawArrayLibrary.h *)

$NumericArrayTypeCode = Data`UnorderedAssociation[
	"Integer8" -> 1,
	"Integer16" -> 3,
	"Integer32" -> 5,
	"Integer64" -> 7,
	"UnsignedInteger8" -> 2,
	"UnsignedInteger16" -> 4,
	"UnsignedInteger32" -> 6,
	"UnsignedInteger64" -> 8,
	"Real32" -> 9,
	"Real64" -> 10,
	"Real16" -> 13 (* this doesn't exist but we need to talk about it *)
]
