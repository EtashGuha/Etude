(* ::Package:: *)

Begin["System`Convert`SFFDump`"]


ImportExport`RegisterImport[
"SFF",
{
"Header":>ImportUserVisibleHeaderData,
"XMLManifest":>ImportXM,

"ReadName":>ImportReadNames,(*uses index if present*)
"Sequence":>ImportAllReads["Sequence"],
"Qualities":>ImportAllReads["Qualities"],
"ClipAdapter":>ImportAllReads["ClipAdapter"],
"ClipQualities":>ImportAllReads["ClipQualities"],
"FlowgramValues":>ImportAllReads["FlowgramValues"],
"FlowIndexPerBase":>ImportAllReads["FlowIndexPerBase"],

"LabeledData":>ImportAllReads["LabeledData"],
"Data":>ImportAllReads["Data"],

(******Named retrievers******)
{name_String,"ReadName"}:>(name->ImportNamedRead[name,"ReadName"][##]&),
{name_String,"Sequence"}:>(name->ImportNamedRead[name,"Sequence"][##]&),
{name_String,"Qualities"}:>(name->ImportNamedRead[name,"Qualities"][##]&),
{name_String,"ClipAdapter"}:>(name->ImportNamedRead[name,"ClipAdapter"][##]&),
{name_String,"ClipQualities"}:>(name->ImportNamedRead[name,"ClipQualities"][##]&),
{name_String,"FlowgramValues"}:>(name->ImportNamedRead[name,"FlowgramValues"][##]&),
{name_String,"FlowIndexPerBase"}:>(name->ImportNamedRead[name,"FlowIndexPerBase"][##]&),

{name_String,"LabeledData"}:> (name->ImportNamedRead[name,"LabeledData"][##]&),
{name_String,"Data"}:> (name->ImportNamedRead[name,"Data"][##]&),

(******Default -- DANGEROUS as it's potentially huge******)
ImportAllReads["Data"]
},
(*"AvailableElements"->{"Header","XMLManifest","Data","LabeledData","ReadName","Sequence","QualityScores","ClipAdapter","ClipQualities","FlowgramValues","FlowIndexPerBase"},*)
 "FunctionChannels"->{"Streams"},
"DefaultElement"->"Data",
"BinaryFormat"->True]


End[]
