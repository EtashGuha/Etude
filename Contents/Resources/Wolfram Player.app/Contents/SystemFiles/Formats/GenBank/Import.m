(* ::Package:: *)

Begin["System`Convert`GenBankDump`"]


ImportExport`RegisterImport[
 "GenBank",
 ImportGenBank,
 "AvailableElements" -> {"Accession","BaseCount","Comments","Contig","DatabaseSource",
		"Definition","Features","Keywords","Locus","Origin","Project","Reference",
		"Segment","SourceOrganism","Version", _String},
 "DefaultElement" -> "Sequence",
 "FunctionChannels" -> {"FileNames"}
]


End[]
