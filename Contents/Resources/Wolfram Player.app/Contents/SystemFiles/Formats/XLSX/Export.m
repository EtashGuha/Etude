Begin["System`Convert`ExcelDump`"]

ImportExport`RegisterExport[
 "XLSX",
 (ExportExcel["XLSX"][##]&),
 "Options"->{"Formulas", "Images"},
 "Sources" -> Join[{"JLink`"}, ImportExport`DefaultSources["Excel"]],
 "DefaultElement" -> "Data",
 "BinaryFormat" -> True
]

End[]