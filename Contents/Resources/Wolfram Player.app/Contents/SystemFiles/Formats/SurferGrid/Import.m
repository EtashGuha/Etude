(* ::Package:: *)

Begin["System`Convert`SurferDump`"]


ImportExport`RegisterImport[
  "SurferGrid",
  ImportSurfer,
  {
  	"Graphics":> getGraphics,
  	"Image" :> getImage,
  	"ReliefImage" :> getReliefImage
  },
  "FunctionChannels" -> {"Streams"},
  "AvailableElements" -> {"Data", "ElevationRange", "Graphics", "Image", "RasterSize", "ReliefImage", "SpatialRange", "SpatialResolution"},
  "DefaultElement" -> "Graphics",
  "Options" -> {"DefaultElevation", "DownsamplingFactor", "SpatialRange"},
  "BinaryFormat" -> True
]


End[]
