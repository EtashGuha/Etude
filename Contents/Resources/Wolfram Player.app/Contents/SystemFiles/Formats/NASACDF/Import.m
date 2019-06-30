(* ::Package:: *)

Begin["System`Convert`NASACDFDump`"]


ImportExport`RegisterImport["NASACDF",
 {
   "Elements" :> CDFgetElements,
   {"DataEncoding"} :> CDFImportMetadataWrapper,
   {"Metadata"} :> GMetadataAll,
   {"Data"} :> CDFImportAllDataElementsWrapper,
   {"Datasets"} -> (fixDatasets[CDFImportMetadataWrapper[##]] &),
   {"Datasets", name_String} :> CDFImportDataElementsByNameWrapper[name],
   {"Datasets", name_Integer} :> CDFImportDataElementsByNumberWrapper[name],
    CDFImportMetadataWrapper
 },
 {
    {"Annotations"} :> CDFGetAnnotations,
    {"DataFormat"}	:> CDFGetDataFormat,
    {"Dimensions"}	:> CDFGetDimensions
 },
 "AvailableElements" -> {"Annotations", "Data", "DataEncoding", "DataFormat", "Datasets", "Dimensions", "Metadata"},
 "DefaultElement" -> "Datasets",
 "BinaryFormat" -> True,
 "Sources" -> ImportExport`DefaultSources[{"NASACDF", "CDF.exe", "DataCommon"}]
]


End[]
