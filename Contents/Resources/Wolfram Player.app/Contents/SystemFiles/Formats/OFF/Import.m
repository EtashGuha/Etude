(* ::Package:: *)

Begin["System`Convert`OFFDump`"]

ImportExport`RegisterImport[
  "OFF",
 ImportOFF,
  {
        "Elements"			   :> GetElements,
		"BinaryFormat"           :> CreateBinaryFormat,
		"Graphics3D" 		    :> CreateGraphics,
		"GraphicsComplex"	    :> CreateGraphicsComplex,
        "InvertNormals"          :> CreateInvertNormals,
        "PolygonColors"          :> CreatePolygonColors,
        "PolygonData" 	       :> CreatePolygonData,
		"PolygonObjects" 	    :> CreatePolygonObjects,
		"VertexColors"           :> CreateVertexColors,
        "VertexData"             :> CreateVertexData,
        "VertexNormals"          :> CreateVertexNormals,
        "VerticalAxis"           :> CreateVerticalAxis,
        "CoordinateTransform"    :> CreateCoordinateTransform,
		"BoundaryMeshRegion"     :> CreateBoundaryMeshRegion,
        "MeshRegion"			 :> CreateMeshRegion,
        "Region"				 :> CreateRegion,
		"Summary"        	    :> CreateSummary,
		CreateMeshRegion
  },
  "Sources" -> ImportExport`DefaultSources[{"Common3D", "OFF"}],
  "FunctionChannels" -> {"Streams"},
  "AvailableElements" -> {"BinaryFormat", "Graphics3D", "GraphicsComplex",
			"InvertNormals", "PolygonColors", "PolygonData", "PolygonObjects",
			"VertexColors", "VertexData", "VertexNormals", "VerticalAxis",
			"MeshRegion", "BoundaryMeshRegion", "CoordinateTransform", "Region", "Summary"
   },
  "DefaultElement"   -> "MeshRegion",
  "Options" 		 -> {"BinaryFormat", "InvertNormals", "VerticalAxis"},
  "BinaryFormat"     -> True
]

End[]
