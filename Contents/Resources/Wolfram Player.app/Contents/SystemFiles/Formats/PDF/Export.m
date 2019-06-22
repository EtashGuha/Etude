(* ::Package:: *)

ImportExport`RegisterExport[
  "PDF",
  System`Convert`PDFDump`ExportPDF,
  "Sources" -> "Convert`PDF`",
  (* explicitly no default element here *)
  "Options" -> {"AllowRasterization"},
  "BinaryFormat" -> True
]

