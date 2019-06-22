(* ::Package:: *)

ImportExport`RegisterImport["MCTT",
  WSMLink`Utilities`readCombiTimeTable,
  {
    "Data" :> WSMLink`Utilities`readCombiTimeTableThinSlice,
    "TimeSeries" :> WSMLink`Utilities`readCombiTimeTableTimeSeries
  },
  "AvailableElements" -> {"Data", "TimeSeries"},
  "DefaultElement" -> "Data",
  "Sources" -> {"WSM`"}
]
