(* ::Package:: *)

Begin["System`Convert`DumpDump`"]


ImportExport`RegisterImport[
 "MX",
 ImportMX,
 {"Expression" -> GetExpression},
 "AvailableElements" -> {"HeldExpression", "Expression"},
 "DefaultElement" -> "Expression",
 "Sources" -> ImportExport`DefaultSources["Dump"],
 "BinaryFormat" -> True
]


End[]
