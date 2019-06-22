(* ::Package:: *)

ImportExport`RegisterExport[
	"AVI",
	System`Convert`AVIDump`ExportAVI,
	"Sources" -> Join[ImportExport`DefaultSources["CommonGraphics"], ImportExport`DefaultSources["AVI"]],
	"FunctionChannels" -> {"Streams"},
	"Unevaluated" -> False,
	"BinaryFormat" -> True
	(* no "DefaultElement" explicitly.  Converter handles default element parsing *)
]
