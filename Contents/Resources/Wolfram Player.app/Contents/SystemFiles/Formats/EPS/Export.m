(* ::Package:: *)

ImportExport`RegisterExport["EPS",
	System`Convert`EPSDump`ExportEPSElements,
	"DefaultElement" -> "Graphics",
	"Options" -> {"PreviewFormat"},
	"BinaryFormat" -> True
]