(* ::Package:: *)

Begin["System`Convert`EDFDump`"]


ImportExport`RegisterImport[
 "EDF",
 {
  "Data"           :>   ImportEDFData["Data"],
  "RecordsData"    :>   ImportEDFData["RecordsData"],
  "LabeledData"    :>   ImportEDFData["LabeledData"],
  {"Data",channel_String}        :> ImportEDFSignalData[channel,"Data"],
  {"LabeledData",channel_String} :> ImportEDFSignalData[channel,"LabeledData"],
  {"RecordsData",channel_String} :> ImportEDFSignalData[channel,"RecordsData"],
  "Annotations"    :>   ImportEDFData["Annotations"],
  "RecordTimes"  :>     ImportEDFData["RecordTimes"],
  (*"Elements"    :>      ImportEDFElements,*)
  ImportEDFHeader
 },
 {},
 "FunctionChannels" -> {"Streams"},
 "DefaultElement" -> "Data",
 "AvailableElements" -> {"Annotations", "ChannelCount", "Data", "DataRanges", "DataUnits",
                "Device", "FilterInformation", "LabeledData", "Labels", "PatientID",
                "RecordCount", "RecordLength", "RecordTimes", "RecordsData",
                "StartDate", "Title"},
 "BinaryFormat" -> True                
]


End[]
