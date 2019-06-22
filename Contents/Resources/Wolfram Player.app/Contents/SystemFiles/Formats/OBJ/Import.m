(* ::Package:: *)

Begin["System`Convert`OBJDump`"]


ImportExport`RegisterImport[
 "OBJ",
 {
    "Graphics3D"            :> CreateGraphics,
	"GraphicsComplex"       :> CreateGC,
	"InvertNormals"         :> CreateInvertNormals,
	"LineData"              :> CreateLineData,
	"LineObjects"           :> CreateLineObjects,
	"PointData"             :> CreatePointData, 
	"PointObjects"          :> CreatePointObjects,
	"PolygonColors"         :> CreatePolygonColors,
	"PolygonData"           :> CreatePolygonData,
	"PolygonObjects"        :> CreatePolygonObjects,
	"VertexData"            :> CreateVertexData,
	"VertexNormals"         :> CreateVertexNormals,
	"VerticalAxis"          :> CreateVerticalAxis,
    "CoordinateTransform"   :> CreateCoordinateTransform,
    "BoundaryMeshRegion"    :> CreateBoundaryMeshRegion,
    "MeshRegion"			:> CreateMeshRegion,
    "Region"				:> CreateRegion,
	"Summary"        	   :> CreateSummary,
	CreateMeshRegion
 },
 "Sources" -> ImportExport`DefaultSources[{"Common3D", "OBJ"}], 
 "FunctionChannels" -> {"Streams"},
 "AvailableElements" -> {"Graphics3D", "GraphicsComplex", "InvertNormals", "LineData",
			"LineObjects", "PointData", "PointObjects", "PolygonColors", "PolygonData",
			"PolygonObjects", "VertexData", "VertexNormals", "VerticalAxis",
            "MeshRegion", "BoundaryMeshRegion", "CoordinateTransform", "Region", "Summary"
  },
 "DefaultElement" -> "MeshRegion",
 "Options" -> {"InvertNormals", "VerticalAxis"},
 "BinaryFormat" -> True
]



End[]
