(* ::Package:: *)

Begin["System`Convert`XPTDump`"]


ImportExport`RegisterImport[
  "XPORT",
  {
   (*Raw Imports*)
    "Elements" :> System`Convert`XPTDump`XPTgetElements,
    "Metadata" :> ImportXPTMetadata,
    ImportXPT
  },
  {(*Post Import*)
    "Data" :> makeData,
    "Labels" :> getVarNames
  },
  "Sources" -> ImportExport`DefaultSources["XPT"], 
  "FunctionChannels" -> {"Streams"},
  "AvailableElements" -> {"Data", "LabeledData", "Labels", "Metadata"},
  "DefaultElement" -> {"Data"},
  "BinaryFormat" -> True
]


End[]
