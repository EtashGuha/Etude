(* ::Package:: *)

Begin["System`Convert`ZIPDump`"]


ImportExport`RegisterImport[
 "ZIP",
 {
  {"Elements"} -> ({"FileNames"->{}}&),
  {"FileNames", "Elements"} -> ImportFileNames[{"FileNames", "Elements"}],
  {"FileNames"} -> GetZIPFileName,
  {"FileNames", (files:(_String | {__String})), rest___} :> (ImportZip[#1, files, {rest}, ##2]&),
  {(filename:Except[({"FileNames"}|"FileNames"|All)]) , rest___}:> ImportZIPShortcut[filename, rest],
  GetZIPFileName
 },
 {
 },
 "Sources" -> {"JLink`", "Convert`ZIP`"},
 "AvailableElements" -> {"FileNames", _String},
 "DefaultElement" -> "FileNames",
 "BinaryFormat" -> True
]


End[]
