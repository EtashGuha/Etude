(* ::Package:: *)

Begin["System`Convert`DXFDump`"]


ImportExport`RegisterImport[
 "DXF",
 {
   "Region"              -> CreateRegion,
   "MeshRegion"          -> CreateMeshRegion,
   "BoundaryMeshRegion"  -> CreateBoundaryMeshRegion,
   
   "Graphics3D"          -> CreateGraphics,
   "GraphicsComplex"     -> CreateGraphicsComplex,
   
   "PolygonObjects"      -> CreatePolygonObjects,
   "LineObjects"         -> CreateLineObjects,
   "PointObjects"        -> CreatePointObjects,
   
   "PolygonData"         -> CreatePolygonData,
   "LineData"            -> CreateLineData,
   "PointData"           -> CreatePointData,
   
   "VertexData"          -> CreateVertexData,
   "VertexColors"        -> CreateVertexColors,
   
   "PlotRange"           -> CreatePlotRange,
   "ViewPoint"           -> CreateViewPoint,
   "CoordinateTransform" -> CreateCoordinateTransform,
   
 (*  "BinaryFormat"        -> CreateBinaryFormat,*)
   "Summary"             -> CreateSummary,
   CreateMeshRegion
 },
 "Sources" -> {"Convert`Common3D`", "Convert`DXF`"},
 "FunctionChannels" -> {"Streams"},
 "AvailableElements" -> {"MeshRegion", "Graphics3D", "GraphicsComplex", "LineData", "LineObjects",
		"PlotRange", "PointData", "PointObjects", "PolygonData",
		"PolygonObjects", "VertexColors", "VertexData", "ViewPoint", 
		"CoordinateTransform", "Region", "BoundaryMeshRegion",
		"BinaryFormat", "Summary"},
 "DefaultElement" -> "MeshRegion",
 "Options" -> {"PlotRange", "ViewPoint"}
]


End[]
