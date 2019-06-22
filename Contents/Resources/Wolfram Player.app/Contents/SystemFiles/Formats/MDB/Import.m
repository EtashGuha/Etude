(* ::Package:: *)

Begin["System`Convert`MDBDump`"]


ImportExport`RegisterImport[
 "MDB",
 {
  "Elements"     :> ImportMDBElements,
  "Data"         :> ImportMDBData,
  "Datasets"     :> ImportMDBDatasets,
  {"Datasets", "Elements"} :>  ImportMDBDatasetsElements,
 {"Datasets", a_String}    :>  ImportMDBDatasetsName[a],
 {"Datasets", a_Integer}   :>  ImportMDBDatasetsName[a],
 {"Datasets", a_, "Elements"} :> ImportMDBDatasetsNameElements[a],
 {"Datasets", a_String, "Data" } :> ImportMDBDatasetsNameData[a],
 {"Datasets", a_Integer, "Data"} :> ImportMDBDatasetsNameData[a],
 {"Datasets", a_String, "Labels" } :> ImportMDBDatasetsNameLabels[a],
 {"Datasets", a_Integer, "Labels" } :> ImportMDBDatasetsNameLabels[a],
 {"Datasets", a_String, "LabeledData"} :> ImportMDBDatasetsNameLabeledData[a],
 {"Datasets", a_Integer, "LabeledData"} :>ImportMDBDatasetsNameLabeledData[a],
  ImportMDB
 },
 "Sources" -> Join[{"JLink`"},ImportExport`DefaultSources["MDB"]],
 "FunctionChannels" -> {"FileNames"},
 "SystemID" -> ("Windows*" | "Linux*" | "Mac*" | "Solari*"),
 "AvailableElements" -> {"Data", "Datasets"},
 "DefaultElement" -> "Data",
 "BinaryFormat" -> True
]


End[]
