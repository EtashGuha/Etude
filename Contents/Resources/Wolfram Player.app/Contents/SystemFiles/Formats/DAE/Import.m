(* ::Package:: *)

Begin["System`Convert`DAEDump`"]


ImportExport`RegisterImport[
	"DAE",
	{
		"MeshRegion"			:> CreateMeshRegion,
		"BoundaryMeshRegion"	:> CreateBoundaryMeshRegion,
		"Graphics3D"			:> CreateGraphics,
		"GraphicsComplex" 	  :> CreateGraphicsComplex,
		"VertexData"			:> CreateVertexData,
        "VertexNormals"         :> CreateVertexNormals,
		"VertexColors"          :> CreateVertexColors,
        "LineData"              :> CreateLineData,
		"LineObjects"           :> CreateLineObjects,
		"PolygonData"		   :> CreatePolygonData,
		"PolygonObjects"		:> CreatePolygonObjects,
        "CoordinateTransform"   :> CreateCoordinateTransform,
        "Region"				:> CreateRegion,
		"Summary"        	   :> CreateSummary,
		CreateMeshRegion
	},
	"AvailableElements" -> {"Elements", "MeshRegion", 
		"BoundaryMeshRegion", "Graphics3D", "GraphicsComplex", 
		"VertexData", "VertexNormals", "VertexColors", "LineData", "LineObjects", 
		"PolygonData", "PolygonObjects", "CoordinateTransform", "Region", "Summary"
		},
	"DefaultElement" -> "MeshRegion",
	"FunctionChannels" -> {"Streams"},
	"Options" -> {},
	"Sources" -> ImportExport`DefaultSources[{"Common3D", "DAE"}]
]



End[]
