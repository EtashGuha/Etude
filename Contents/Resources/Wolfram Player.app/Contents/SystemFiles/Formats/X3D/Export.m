(* ::Package:: *)

ImportExport`RegisterExport[
	"X3D",
	(System`Convert`X3DDump`export["X3D"][##]&),
	"Sources" -> ImportExport`DefaultSources[{"Common3D", "X3D"}],
	"FunctionChannels" -> {"Streams"},
	"Options" -> {"VerticalAxis"}
]
