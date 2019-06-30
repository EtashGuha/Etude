(* ::Package:: *)

Begin["System`Convert`GTOPO30Dump`"]


ImportExport`RegisterImport[
 "GTOPO30",
 {
    "Elements" :> gtopo30Elements,
    ("SpatialRange"|"SpatialResolution"|"DataFormat") :> evaluateElements,
    ImportLog
 },
 {
   "Graphics" :> getGraphics,
   "Image" :> getImage,
   "ReliefImage" :> getReliefImage
 },
 "AvailableElements" -> {"Data", "DataFormat", "Dimensions", "ElevationRange", "Graphics", "Image", "ReliefImage", "SpatialRange", "SpatialResolution"},
 "DefaultElement" -> "Graphics",
 "FunctionChannels"->{"FileNames","Directories"}
]


End[]
