

Begin["DeployedResourceShingle`"]

Begin["`Private`"]

$possibleShingleProperties={"UUID","Name","ShortName","ContributorInformation","SourceMetadata",
	"DisplayedSourceFields","ExampleNotebook","ResourceType","Originator",
	"ContentSize","CreationDate","SubmissionDate","ApprovalDate","ReleaseDate","LatestUpdate","Version",
	"Description","Details","MoreDetails","ExternalLinks","LicensingInformation","PricingInformation"}
	
processShingleData[rtype_,info_, target_]:=processShingleData[rtype,info["UUID"],info,target]

processShingleData[rtype_,id_,info_,target_]:=
With[{rules=KeyValueMap[processshingleData[rtype,id,info,target,##]&,getExampleNotebookContent[info]]},
	Join[defaultShingleData[rtype],Association[rules],additionalShingleData[rtype, id,info]]
]

processshingleData[__,"SourceMetadata",as_Association]:=("SourceMetadata"->sourceMetadataShingleString/@as)
processshingleData[__,"ContributorInformation",as_Association]:=KeyTake[as,"ContributedBy"]
processshingleData[rtype_,id_,info_, target_,"ExampleNotebook",nbo:(_NotebookObject|_Notebook)]:=("examples"->exampleSection[rtype,id,target,info,nbo])
processshingleData[rtype_,id_,info_, target_,"ExampleNotebook",_]:=("examples"->"")
processshingleData[__,"SourceMetadata",_]:=Nothing
processshingleData[__,"Details",details_]:=("Details"->StringJoin["<p>", StringReplace[ExportString[details, "HTMLFragment"], "\n" -> "</p>\n<p>"], "</p>"])
processshingleData[__,"ExternalLinks",links_List]:=("ExternalLinks"->toLink/@links)

processshingleData[args___]:=processshingledata[args]
processshingledata[__,key_,value_List]:=(key->processshingledatastr/@value)/;MemberQ[$possibleShingleProperties,key]
processshingledata[__,key_,value_]:=(key->processshingledatastr[value])/;MemberQ[$possibleShingleProperties,key]

toLink[h_Hyperlink]:=ExportString[h, "HTMLFragment"]
toLink[str_String]:=toLink[Hyperlink[str]]
toLink[expr_]:=TextString[expr]

processshingledatastr=TextString;

processshingledata[___]:=Nothing

getExampleNotebookContent[info:KeyValuePattern["ExampleNotebookData"->nb_Notebook]]:=
	Append[KeyDrop[info,{"ExampleNotebookData","ExampleNotebook"}],"ExampleNotebook"->nb]
	
getExampleNotebookContent[info:KeyValuePattern["ExampleNotebookData"->nbo_NotebookObject]]:=If[
	ResourceSystemClient`Private`notebookObjectQ[nbo],
	Append[KeyDrop[info,{"ExampleNotebookData","ExampleNotebook"}],"ExampleNotebook"->nbo],
	getExampleNotebookContent[KeyDrop[info,{"ExampleNotebook"}]]
]

getExampleNotebookContent[info:KeyValuePattern["ExampleNotebook"->nb_Notebook]]:=info
	
getExampleNotebookContent[info:KeyValuePattern["ExampleNotebook"->nbo_NotebookObject]]:=If[
	ResourceSystemClient`Private`notebookObjectQ[nbo],
	info,
	Append[KeyDrop[info,{"ExampleNotebookData","ExampleNotebook"}],
		"ExampleNotebook"->ResourceSystemClient`Private`openLocalExampleNotebook[info["UUID"],True ]]
]

getExampleNotebookContent[info:KeyValuePattern["ExampleNotebook"->file:HoldPattern[_CloudObject|_File|_LocalObject|_String?FileExistsQ]]]:=With[
	{nb=Import[file]},
	If[Head[nb]===Notebook,
		Append[KeyDrop[info,{"ExampleNotebookData","ExampleNotebook"}],"ExampleNotebook"->nb]
		,
		Append[KeyDrop[info,{"ExampleNotebookData","ExampleNotebook"}],
		"ExampleNotebook"->ResourceSystemClient`Private`openLocalExampleNotebook[info["UUID"],False ]]
	]
]


getExampleNotebookContent[info_,___]:=info

additionalShingleData[___]:=Association[]

defaultShingleData[_]:=Association["ContentStats"->""]

sourceMetadataShingleString[value_String]:= value
sourceMetadataShingleString[ent:HoldPattern[_Entity]]:=CommonName[ent]
sourceMetadataShingleString[date:(DateObject[{_}]|DateObject[{_,_}])]:=DateString[date]
sourceMetadataShingleString[date_DateObject]:=DateString[date, "Date"]
sourceMetadataShingleString[list_List]:=TextString[sourceMetadataShingleString/@list, ListFormat -> {"", ", ", ""}]
sourceMetadataShingleString[link_Hyperlink]:=ExportString[link,"HTMLFragment"]
sourceMetadataShingleString[url_URL]:=sourceMetadataShingleString[Hyperlink[First[url]]]
sourceMetadataShingleString[expr_]:=TextString[expr]
sourceMetadataShingleString[_,_]:=""

End[]

End[]