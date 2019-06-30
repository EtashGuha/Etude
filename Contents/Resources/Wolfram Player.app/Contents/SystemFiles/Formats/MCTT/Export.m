(* ::Package:: *)

ImportExport`RegisterExport[
  "MCTT",
  WSMLink`Utilities`writeCombiTimeTableWithElements,
  "Options" -> {"StartTime", "StopTime", "InterpolationOrder", "SamplingPeriod"},
  "DefaultElement" -> Automatic,
  "Sources" -> {"WSM`"}
]
