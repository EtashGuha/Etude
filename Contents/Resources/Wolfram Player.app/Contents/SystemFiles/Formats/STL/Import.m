(* ::Package:: *)

Begin["System`Convert`STLDump`"]


ImportExport`RegisterImport[
	"STL",
	{
		"MeshRegion"			:> CreateMeshRegion,
		"Graphics3D"			:> CreateGraphics,
		"GraphicsComplex" 	  :> CreateGraphicsComplex,
		"VertexData"			:> CreateVertexData,
		"PolygonData"		   :> CreatePolygonData,
		"PolygonObjects"		:> CreatePolygonObjects,
		"BinaryFormat"      	:> CreateBinaryFormat,
		"Comments"			  :> CreateName,
		"PolygonCount"		  :> CreateTriangleCount,
		"FacetNormals"		  :> CreateFacetNormals,
        "VerticalAxis"          :> CreateVerticalAxis,
		"CoordinateTransform"   :> CreateCoordinateTransform,
		"BoundaryMeshRegion"    :> CreateBoundaryMeshRegion,
		"Region"				:> CreateRegion,
		"Summary"        	   :> CreateSummary,
		If[ TrueQ[ NDSolve`FEM`FEMPackageLoaded],
			"ElementMesh"		:> CreateElementMesh
		, (* else *)
			Sequence[]
		],
		CreateMeshRegion
	},
	"AvailableElements" -> {"BinaryFormat", 
		"MeshRegion", "Graphics3D", "GraphicsComplex", 
		"PolygonData", "PolygonObjects", "VertexData", "VerticalAxis",
        "Comments", "PolygonCount", "FacetNormals", "BoundaryMeshRegion", 
		"CoordinateTransform", "Region", "Summary",
		If[ TrueQ[ NDSolve`FEM`FEMPackageLoaded], "ElementMesh", Sequence[]]
	  },
	"BinaryFormat" -> True,
	"DefaultElement" -> "MeshRegion",
	"FunctionChannels" -> {"Streams"},
	"Options" -> {"BinaryFormat", "VerticalAxis"},
	"Sources" -> ImportExport`DefaultSources[{"Common3D", "STL"}]
]



End[]
