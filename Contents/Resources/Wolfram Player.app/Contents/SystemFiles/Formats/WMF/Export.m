(* ::Package:: *)

(* Windows Metafile *)
ImportExport`RegisterExport[
	"WMF",
	System`ConvertersDump`ExportMetaFileElements,
	"AvailableElements" -> {"Graphics"},
	"DefaultElement" -> "Graphics",
	"InterfaceEnvironment" -> "*Windows*",
	"BinaryFormat" -> True
]

