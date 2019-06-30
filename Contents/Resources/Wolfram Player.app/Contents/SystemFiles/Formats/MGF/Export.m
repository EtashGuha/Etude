(* ::Package:: *)

ImportExport`RegisterExport[
  "MGF",
  System`Convert`BitmapDump`ExportMGF[##]&,
  "Sources" -> ImportExport`DefaultSources["Bitmap"],
  "BinaryFormat" -> True
]
