(* ::Package:: *)

ImportExport`RegisterExport[
  "DICOM",
  System`Convert`DicomDump`ExportDICOM,
  "Sources" -> {"Convert`DicomDataDictionary`", "Convert`Dicom`"},
  "DefaultElement" -> Automatic,
  "BinaryFormat" -> True,
  "AlphaChannel" -> True
]
