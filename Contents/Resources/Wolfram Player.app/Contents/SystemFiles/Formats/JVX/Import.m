(* ::Package:: *)

Begin["System`Convert`JVXDump`"]


ImportExport`RegisterImport[
 "JVX",
 { (* Raw Importers *)
   "Elements" :> getElements,
    ImportJVX
 },
 {
  Automatic -> RawDataToAutomatic,
  "Graphics" -> RawDataToGraphics,
  "Graphics3D" -> RawDataToGraphics3D,
  "VertexData" -> RawDataToData["VertexData"],
  "PolygonObjects" -> RawDataToPrimitives["PolygonData", Polygon],
  "PolygonData" -> RawDataToData["PolygonData"],
  "LineObjects" -> RawDataToPrimitives["LineData", Line],
  "LineData" -> RawDataToData["LineData"],
  "PointObjects" -> RawDataToPrimitives["PointData", Point],
  "PointData" -> RawDataToData["PointData"],
  "VertexNormals" -> RawDataToData["VertexNormals"],
  "VertexColors" -> RawDataToData["VertexColors"],
  "GraphicsComplex" -> RawDataToGraphicsComplex,
  "InvertNormals" -> ("InvertNormals"/.#)&,
  "VerticalAxis" -> ("VerticalAxis"/.#)&
 },
 "Options" -> {"ShortSummary", "Summary", "Creator", "CreationDate",
                "Keywords", "Title", "Version",  "InvertNormals", "VerticalAxis"
              },
 "AvailableElements" -> {"CreationDate", "Creator", "Graphics", "Graphics3D", "GraphicsComplex", "InvertNormals",
 				"Keywords", "LineData", "LineObjects", "PointData", "PointObjects", "PolygonData",
 				"PolygonObjects", "ShortSummary", "Summary", "Title", "Version", "VertexColors", "VertexData",
 				"VertexNormals", "VerticalAxis"},
 "DefaultElement" -> Automatic,
 "Sources" -> ImportExport`DefaultSources[{"Common3D", "JVX"}],
 "FunctionChannels" -> {"Streams"},
 "BinaryFormat" -> True
]


End[]
