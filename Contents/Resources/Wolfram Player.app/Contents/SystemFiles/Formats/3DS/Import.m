(* ::Package:: *)

ImportExport`RegisterImport[
 "3DS",
 System`Convert`ThreeDSDump`Import3DS,
 {
  "Graphics3D" :> System`Convert`ThreeDSDump`CreateGraphics,
  "GraphicsComplex" :> System`Convert`ThreeDSDump`CreateGC,
  "PolygonObjects" :> System`Convert`ThreeDSDump`CreatePolygonObjects
  },
 "Sources" -> {"Convert`Common3D`", "Convert`ThreeDS`"},
 "FunctionChannels" -> {"Streams"},
 "AvailableElements" -> {"Graphics3D", "GraphicsComplex", "PolygonColors", "PolygonData",
						"PolygonObjects", "VertexData", "VerticalAxis"},
 "DefaultElement" -> "Graphics3D",
 "Options" -> {"VerticalAxis"},
 "BinaryFormat" -> True
]
