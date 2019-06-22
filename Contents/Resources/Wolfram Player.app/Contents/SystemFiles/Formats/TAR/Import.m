(* ::Package:: *)

Begin["System`Convert`TARDump`"]


ImportExport`RegisterImport[
 "TAR",
 {
  {"Elements"} -> ({"FileNames"->{}}&),
  {"FileNames", "Elements"} -> ImportFileNames[{"FileNames", "Elements"}],
  {"FileNames"} -> ImportFileNames[{"FileNames"}],
  {"FileNames", (files:(_String|{__String})), rest___} :> (ImportTar[#1, files, {rest}, ##2]&),
  {(filename:Except[({"FileNames"}|"FileNames"| All)]) , rest___}:> ImportTARShortcut[filename, rest],
  ImportFileNames[{"FileNames"}]
 },
 {
 },
 "Sources" -> {"JLink`", "Convert`TAR`"},
 "AvailableElements" -> {"FileNames", _String},
 "DefaultElement" -> "FileNames",
 "BinaryFormat" -> True
]


End[]
