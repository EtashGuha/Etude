(* ::Package:: *)

Begin["System`Convert`BYUDump`"]


ImportExport`RegisterImport[
    "BYU",
	ImportBYU,
	{
		"Graphics3D" 	 :> CreateGraphics,
		"GraphicsComplex" :> CreateGC,
		"PolygonObjects"  :> CreatePolygonObjects
	},
	"Sources" 		 -> {"Convert`Common3D`", "Convert`BYU`"},
	"FunctionChannels" -> {"Streams"},
	"DefaultElement"   -> "Graphics3D",
	"AvailableElements" -> {"Graphics3D", "GraphicsComplex", "PolygonData", "PolygonObjects", "VertexData", "VerticalAxis"},
	"Options" 		 -> {"VerticalAxis"}
]


End[]
