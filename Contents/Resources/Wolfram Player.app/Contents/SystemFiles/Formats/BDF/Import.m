(* ::Package:: *)

Begin["System`Convert`BDFDump`"]


ImportExport`RegisterImport[
 "BDF",
 {
  "Data"           :>   ImportBDFData["Data"],
  "RecordsData"    :>   ImportBDFData["RecordsData"],
  "LabeledData"    :>   ImportBDFData["LabeledData"],
  {"Data",channel_String}        :> ImportBDFSignalData[channel,"Data"],
  {"LabeledData",channel_String} :> ImportBDFSignalData[channel,"LabeledData"],
  {"RecordsData",channel_String} :> ImportBDFSignalData[channel,"RecordsData"],
  "Annotations"    :>   ImportBDFData["Annotations"],
  "RecordTimes"  :>     ImportBDFData["RecordTimes"],
  ImportBDFHeader
 },
 {},
 "FunctionChannels" -> {"Streams"},
 "DefaultElement" -> "Data",
 "AvailableElements" -> {"Annotations", "ChannelCount", "Data", "DataRanges", 
 			"DataUnits", "Device", "FilterInformation", "LabeledData", "Labels",
			"PatientID", "RecordCount", "RecordLength", "RecordTimes", "RecordsData", 
			"StartDate", "Title"},
 "BinaryFormat" -> True
]


End[]
