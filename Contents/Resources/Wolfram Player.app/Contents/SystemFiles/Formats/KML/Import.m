(* ::Package:: *)

Begin["System`Convert`KMLDump`"]


ImportExport`RegisterImport[
 "KML",
 {
	"Elements" :> getKMLElements,
	"Data"|"ImageOverlays"|"Graphics"|"GraphicsList"|"LayerTypes"|"LayerNames"|"SpatialRange":> ImportKMLData
 },
 {
  "Data":>KMLGetData,
  "LayerTypes" :>KMLGetLayerTypes,
  "LayerNames" :>KMLGetLayerNames,
  "SpatialRange" :> KMLGetSpacialRange, 
  "Graphics" :> KMLGraphics,
  "GraphicsList" :> KMLGraphicsList,
  "ImageOverlays" :> KMLGetOverlayImages
 },
 "AvailableElements" -> {"Data", "Graphics", "GraphicsList", "LayerNames", "LayerTypes", "ImageOverlays", "SpatialRange"},
 "DefaultElement" -> "Graphics",
 "FunctionChannels" -> {"FileNames", "Directories"},
 "BinaryFormat" -> True
]


End[]
