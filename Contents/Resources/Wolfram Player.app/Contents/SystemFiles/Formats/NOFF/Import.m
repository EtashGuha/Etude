(* ::Package:: *)

Begin["System`Convert`OFFDump`"]


ImportExport`RegisterImport[
 "NOFF",
 ImportNOFF,
 {	
    Automatic 		 :> LegacyCreateGraphics,
	"Graphics" 		:> LegacyCreateGraphics2D,
	"Graphics3D" 	  :> LegacyCreateGraphics3D,
	"GraphicsComplex"  :> CreateGraphicsComplex,
	"PolygonObjects"   :> CreatePolygonObjects
 },
 "Sources" -> ImportExport`DefaultSources[{"Common3D", "OFF"}],
 "FunctionChannels"	-> {"Streams"},
 "AvailableElements" -> {"BinaryFormat", "Graphics", "Graphics3D", "GraphicsComplex",
			"InvertNormals", "PolygonColors", "PolygonData", "PolygonObjects",
			"VertexColors", "VertexData", "VertexNormals", "VerticalAxis"},
 "DefaultElement" 	-> Automatic,
 "Options" 		-> {"BinaryFormat", "InvertNormals", "VerticalAxis"},
 "BinaryFormat" -> True
]


End[]
