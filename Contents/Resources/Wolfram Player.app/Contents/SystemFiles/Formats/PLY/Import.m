(* ::Package:: *)

Begin["System`Convert`PLYDump`"]


ImportExport`RegisterImport[
  "PLY",
  ImportPLY,
  {
    "Elements"			  :> GetElements,
	"BinaryFormat"          :> CreateBinaryFormat,
    "Comments"              :> CreateComments,
    "DataFormat"            :> CreateDataFormat,
	"Graphics3D" 		   :> CreateGraphics,
	"GraphicsComplex"       :> CreateGraphicsComplex,
    "InvertNormals"         :> CreateInvertNormals,
    "LineData" 	         :> CreateLineData,
	"LineObjects" 	      :> CreateLineObjects,
    "PolygonData" 	      :> CreatePolygonData,
	"PolygonObjects" 	   :> CreatePolygonObjects,
    "UserExtensions"        :> CreateUserExtensions,
    "VertexColors"          :> CreateVertexColors,
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
  "Sources" 		  -> {"Convert`Common3D`", "Convert`PLY`"},
  "FunctionChannels"  -> {"Streams"},
  "AvailableElements" -> {"BinaryFormat", "Comments", "DataFormat", "Graphics3D",
			"GraphicsComplex", "InvertNormals", "LineData", "LineObjects", "PolygonData",
			"PolygonObjects", "UserExtensions", "VertexColors",	"VertexData", "VertexNormals", "VerticalAxis",
			"MeshRegion", "BoundaryMeshRegion", "CoordinateTransform", "Region", "Summary"
  },
  "DefaultElement"    -> "MeshRegion",
  "Options" 		  -> {"BinaryFormat", "DataFormat", "Comments", "VerticalAxis", "InvertNormals"},
  "BinaryFormat" 	  -> True
]


End[]
