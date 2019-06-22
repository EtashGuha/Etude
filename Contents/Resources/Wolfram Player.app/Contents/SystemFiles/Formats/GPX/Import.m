Begin["System`Convert`GPXDump`"]


ImportExport`RegisterImport[
 "GPX",
 {
  importGPX
 },
 {
  "Graphics" :> GPXGraphics,
  "GraphicsList" :> GPXGraphicsList,
  "LayerNames" :> getLayerName
 },
 "AvailableElements" -> {"Comments", "Data", "Graphics", "GraphicsList", "LayerNames", "Metadata", "Name", "SpatialRange"},
 "DefaultElement" -> "Graphics",
 "BinaryFormat" -> True
]


End[]