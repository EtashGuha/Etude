(* ::Package:: *)

ImportExport`RegisterExport[
 "BZIP2",
 System`Convert`BZIP2Dump`ExportBZIP2,
 "Sources" -> {"JLink`", "Convert`BZIP2`"},
 "FunctionChannels" -> {"FileNames"},
 "BinaryFormat" -> True,
 "Encoding" -> True
]
