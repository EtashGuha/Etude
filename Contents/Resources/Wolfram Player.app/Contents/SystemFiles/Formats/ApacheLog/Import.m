(* ::Package:: *)

Begin["System`Convert`ApacheLogDump`"]


ImportExport`RegisterImport[
 "ApacheLog",
 {
   "Data" :> ImportLog[{"Data"}],
   {elems_} :> importFilter[elems]
 },
 "AvailableElements" -> {"Data", _String},
 "DefaultElement" -> "Data"
]


End[]
