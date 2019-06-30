(* ::Package:: *)

Begin["System`Convert`GeoJSONDump`"]


ImportExport`RegisterImport[
  "GeoJSON",
 {  "Elements" :> getGeoJSONElements,
 	ImportGeoJSON
 },
 {
 "Graphics" :> GeoJSONGraphics,
 "GraphicsList" :> GeoJSONGraphicsList,
 "CoordinateSystem" :> getCoordinateSystem,
 "CoordinateSystemInformation" :> getCoordinateSystemInformation
 },
"DefaultElement" -> "Graphics", 
"BinaryFormat" -> True,
"FunctionChannels" -> {"Streams"}
 ]

End[]