(* ::Package:: *)

Begin["System`Convert`FCSDump`"]


ImportExport`RegisterImport[
"FCS",
{
"Header":>ImportAll["Header"],
"ParameterName":>ImportAll["ParameterName"],
"ParameterReagent":>ImportAll["ParameterReagent"],
(*All-element import*)
ImportAll["All"]
},
{
"LabeledData":>LDImport,
"Data":>DataImport
},
(***************************************)
"DefaultElement"->"Events",
"AvailableElements"->{"Events","Header","ParameterName","ParameterReagent","Data","LabeledData"}
]

End[]