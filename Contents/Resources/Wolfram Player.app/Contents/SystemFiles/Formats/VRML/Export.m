(* ::Package:: *)

ImportExport`RegisterExport[
	"VRML",
	(System`Convert`VRMLDump`export["VRML"][##]&),
	"Sources" -> ImportExport`DefaultSources[{"Common3D", "VRML"}],
	"Options" -> {"InvertNormals", "VerticalAxis"}
]
