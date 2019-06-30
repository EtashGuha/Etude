(* ::Package:: *)

Begin["System`Convert`LWODump`"]


ImportExport`RegisterImport[
  "LWO",
  ImportLightwave,
  {
 		"Graphics3D" :> CreateGraphics,
 		"GraphicsComplex" :> CreateGC,
 		"PolygonObjects" :> CreatePolygonObjects,
 		"LineObjects" :> CreateLineObjects,
 		"PointObjects" :> CreatePointObjects
  },
  "AvailableElements" -> {"Graphics3D", "GraphicsComplex", "ImportPatchData",
			"LineData", "LineObjects", "PointData", "PointObjects", "PolygonData",
			"PolygonObjects", "VertexData", "VerticalAxis"},
  "DefaultElement" -> "Graphics3D",
  "Sources" -> {"Convert`Common3D`", "Convert`LWO`"},
  "FunctionChannels" -> {"Streams"},
  "Options" -> {"VerticalAxis"},
  "BinaryFormat" -> True
]


End[]
