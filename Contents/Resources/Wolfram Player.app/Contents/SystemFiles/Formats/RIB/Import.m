(* ::Package:: *)

ImportExport`RegisterImport[
 "RIB",
 System`Convert`RIBDump`ImportRIB,
 {
	"Graphics3D" :> System`Convert`RIBDump`CreateGraphics,
	"GraphicsComplex" :> System`Convert`RIBDump`CreateGC,
	"PolygonObjects" :> System`Convert`RIBDump`CreatePolygonObjects
 },
 "Sources" -> ImportExport`DefaultSources[{"Common3D", "RIB"}],
 "FunctionChannels" -> {"Streams"},
 "AvailableElements" -> {"Comments", "CreationDate", "Creator", "Graphics3D",
			"GraphicsComplex", "InvertNormals", "PolygonData", "PolygonObjects",
			"Scene", "VertexColors", "VertexData", "VertexNormals", "VerticalAxis"},
 "DefaultElement" -> "Graphics3D",
 "Options" -> {"Comments", "Scene", "Creator", "CreationDate", "VerticalAxis", "InvertNormals"}
]
