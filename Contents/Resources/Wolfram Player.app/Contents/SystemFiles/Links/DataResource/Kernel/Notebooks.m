(* Wolfram Language Package *)


BeginPackage["DataResource`"]

Begin["`Private`"] (* Begin Private Context *) 

$NotebookToolsDirectory=FileNameJoin[{$drDirectory,"Templates"}];

$CreateDataResourceTemplate=FileNameJoin[{$NotebookToolsDirectory,"CreateDataResourceTemplate.nb"}];
$CreateDataResourceBlank=FileNameJoin[{$NotebookToolsDirectory,"DataResourceDefinition.nb"}];

ResourceSystemClient`Private`repositoryCreateResourceFromNotebook["DataResource",nbo_]:=builddataResourceFromNotebook[nbo]

ResourceSystemClient`Private`repositorySubmitResourceFromNotebook["DataResource",args__]:=buildandsubmitDataResourceFromNotebook[args]

buildDataResourceFromNotebook[nbo_]:=Catch[
	With[{res=scrapeCreateResourceNotebook[nbo]},
		If[ListQ[res],
			printScrapedResource[res[[1]], nbo, {"CreateResourceResult","CreateResourceActionButton"}];
			res[[1]]
			,
			$Failed
		]
	]
	]
	
builddataResourceFromNotebook[nbo_]:=With[{res=scrapeCreateResourceNotebook[nbo]},
		If[ListQ[res],First[res],$Failed]
]

printScrapedResource[ro_, nbo_, tags:{resulttag_, _}]:=
	printscrapedResource[ro, nbo, tags, NotebookFind[nbo, resulttag, All, CellTags]]

printscrapedResource[ro_, nbo_, {resulttag_, _}, resultcell_NotebookSelection]:=
	NotebookWrite[nbo,Cell[BoxData[ToBoxes[ro]], "Output",ShowCellTags->False, CellTags->{resulttag}]]

printscrapedResource[ro_, nbo_, tags:{resulttag_, buttontag_}, $Failed]:=
	printscrapedresource[ro, nbo, tags,NotebookFind[nbo, buttontag, All, CellTags]]

printscrapedresource[ro_, nbo_, {resulttag_, _}, buttoncell_NotebookSelection]:=(
	SelectionMove[nbo, After, Cell];
	NotebookWrite[nbo,Cell[BoxData[ToBoxes[ro]], "Output",ShowCellTags->False, CellTags->{resulttag}]])

printscrapedResource[___]:=Null
printscrapedresource[___]:=Null


buildAndSaveDataResourceFromNotebook[nbo_]:=Block[{ro=builddataResourceFromNotebook[nbo], res},
	res=If[resourceObjectQ[ro],
		ResourceSystemClient`Private`saveResourceObject[ro]
	];
	If[Head[res]===ResourceObject,
		printScrapedResource[res, nbo, {"ResourceLocalSaveResult","SaveLocalResourceActionButton"}]
	]
]

buildAndDeployDataResourceFromNotebook[nbo_]:=Block[{ro=builddataResourceFromNotebook[nbo], res},
	res=If[resourceObjectQ[ro],
		CloudDeploy[ro]
	];
	If[Head[res]===CloudObject,
		printScrapedResource[res, nbo, {"ResourceCloudDeployResult","SaveCloudResourceActionButton"}]
	]
]

$ScrapePublisher=False;

buildAndSubmitDataResourceFromNotebook[nbo_]:=With[{res=buildandsubmitDataResourceFromNotebook[nbo]},
	If[MatchQ[res,(_System`ResourceSubmissionObject|_Success)],
		printScrapedResource[res, nbo, {"ResourceSubmitResult","SubmitResourceActionButton"}]
	]
]

buildandsubmitDataResourceFromNotebook[nbo_]:=buildandsubmitDataResourceFromNotebook[nbo, None]

buildandsubmitDataResourceFromNotebook[nbo_, update_]:=Catch[Block[{ro, 
	$CreateResourcePublisher=None, $ScrapePublisher=True, res},
	ro=builddataResourceFromNotebook[nbo];
	If[resourceObjectQ[ro],
		If[StringQ[$CreateResourcePublisher]&&$CreateResourcePublisher=!=$CreateResourcePublisherPlaceholder,
			System`ResourceSubmit[ro, update,System`PublisherID->$CreateResourcePublisher]
			,
			Message[ResourceSubmit::nopubid]
		]
		,
		Message[ResourceSubmit::invcon]
	]
]]

resultCells[ro_, tag:"ResourceResult"]:={
	Cell[resultSubsectionHeader[tag], "CreateResourceSubsection"],
	Cell[BoxData[ToBoxes[ro]], "Output", CellTags->{tag}]
}

resultSubsectionHeader["ResourceResult"]:="Resource"
resultSubsectionHeader["CloudDeployResult"]:="Cloud Resource Cache"
resultSubsectionHeader["LocalDeployResult"]:="Local Resource Cache"
resultSubsectionHeader["ResourceSubmitResult"]:="Resource System Submission"
resultSubsectionHeader[_]:=""

ResourceSystemClient`Private`createResourceNotebook[$DataResourceType,rest___]:=newDataResourceCreateNotebook[]


newdataResourceCreateNotebook[]:=With[{nb=Get[$CreateDataResourceBlank]},
	NotebookPut[prepareDataResourceCreateNotebook[nb]]	
]

prepareDataResourceCreateNotebook[nb_]:=With[{uuid=CreateUUID[]},
	Replace[nb,{DataResource`Private`$temporaryuuid:>uuid,"DataResource`Private`$temporaryuuid":>uuid},Infinity]
	]

newDataResourceCreateNotebook[]:=With[{nbo=newdataResourceCreateNotebook[]},
	If[Head[nbo]===NotebookObject,    
		SetOptions[nbo,{Visible->True}];  
		NotebookLocate["ResourceUUID"];
		SelectionMove[nbo, Previous, Cell];
		SetSelectedNotebook[nbo];
		nbo
		,
		$Failed
	]
]

scrapeCreateResourceNotebook[nbo_NotebookObject]:=(notebookNeeds[];
	scrapeCreateResourceNotebook0[NotebookGet[nbo]])

scrapeCreateResourceNotebook[nb_Notebook]:=(notebookNeeds[];
	scrapeCreateResourceNotebook0[nb])
	
scrapeCreateResourceNotebook0[nb_]:=Block[{as, ro, id},
	If[Head[nb]=!=Notebook,
		Return[$Failed]
	];
	
	as=scrapecreateResourceNotebook[nb];
	id=as["UUID"];
	If[AssociationQ[as]&&StringQ[id],
		ResourceSystemClient`Private`$localResources=DeleteCases[ResourceSystemClient`Private`$localResources,id];
		ro=ResourceSystemClient`Private`autoloadResource[as];
		If[FailureQ[updateExampleNotebook[nb,{id, as["Name"]},ro,False,as["ExampleNotebook"]]],
			as=KeyDrop[as,"ExampleNotebook"];
			ro=ResourceSystemClient`Private`autoloadResource[as]
		];
		
		{ro,nb}
		,
		Message[ResourceObject::invro];
		$Failed
	]
]

scrapecreateResourceNotebook[nb_]:=Block[{cat, ct, cont, info, id,as,examples},
	id=CreateUUID[];
	info=scrapecreateResourceNotebook[id, nb];
	examples=createExampleNotebookPlaceholder[];
	If[!AssociationQ[info],
		Return[$Failed]
	];
	{cat, ct}=scrapecreateDataResourceSortingProperties[nb];	
	cont=getCreateResourceContent[id, nb];
	If[TrueQ[$ScrapePublisher],
		$CreateResourcePublisher=getCreateResourcePublisher[nb]
	];
	as=Association[
		"ResourceType"->"DataResource",
		"UUID"->id,
		info,
		"Categories"->cat,
		"ContentTypes"->ct,
		cont,
		"SourceMetadata"->scrapecreateresourceNotebook["SourceMetadata", nb],
		examples
	];
	as
]

scrapecreateDataResourceSortingProperties[nb_]:=(
	ResourceSystemClient`Private`resourceSortingProperties["DataResource",False];
	scrapecreatedataResourceSortingProperties[nb,#]&/@{"Categories","ContentTypes"})

scrapecreatedataResourceSortingProperties[nb_,prop_]:=Block[{section},
	section = findCellTags[nb, "DataResource"<>prop];
	Flatten[Cases[section, CheckboxBox[check:Except[False], ___] :> check, 10]]
]

getcontentelements[nb_] := Quiet[With[{symb=Symbol[$Context <> "$$Object"]},
	Block[{DataResource`$$ContentConversion=Identity},
  ToExpression[
   Cases[nb, Cell[cont_, "DataResourceContentInput", ___] :> cont, 
    30]];
  {symb["FullContent"], symb["DefaultContentElement"], symb["ContentElementFunctions"]}
	]
  ]]



getCreateResourceContent[id_, nb_]:=Block[{elements, default,funcs},
	{elements, default, funcs}=getcontentelements[nb];
	If[AssociationQ[elements],
		With[{rules={
			getcreateResourceContent[elements],
			If[StringQ[default],
				"DefaultContentElement"->default
				,{}],
			If[AssociationQ[funcs],
				"ContentElementFunctions"->funcs,
				{}				
			]
		}},
		Association[rules]
		]
		,
		Association[]
	]
]

getcreateResourceContent[elems_]:=
With[{n=Length[elems]},
	If[TrueQ[n>0],
		GroupBy[elems,elementlocationQ]
		,
		Association[]
	]
]

$CreateResourceNotebookNamePlaceHolder=Placeholder["enter name"];
$CreateResourceNotebookContentPlaceHolder=Placeholder["enter content"];
getCreateResourceContentElement[cellgroup_]:=Block[{name,content},
	name=getCreateResourceContentElementName[cellgroup];
	If[!StringQ[name],
		{},
		content=getCreateResourceContentElementContent[cellgroup];
		If[Head[content]===Placeholder,
			{},
			If[TrueQ[getCreateResourceContentElementDefaultStatus[cellgroup]],$createResourceDefaultElem=name];
			{name->content}
		]
	]
]

getCreateResourceContentElementName[cellgroup_]:=With[{cells=Cases[cellgroup, Cell[_, "CreateResourceElementName", ___], 3]},
	scrapeonetextCell[cells]
]

getCreateResourceContentElementContent[cellgroup_]:=With[{cont=Cases[cellgroup, Cell[raw_, "CreateResourceElementInput", ___] :> raw, 3]},
	If[Length[cont]===1,
		ToExpression[First[cont]],
		Throw[$Failed]
	]
]

getCreateResourceContentElementDefaultStatus[cellgroup_]:=With[
	{radio=Cases[cellgroup, Cell[raw_, "CreateResourceDefaultElement", ___] :> raw, 3]},
	And@@Cases[radio,  RadioButtonBox[a_, b_] :> MemberQ[b, Setting[a]], {2}]
]



elementlocationQ[HoldPattern[_CloudObject|_LocalObject|_File|_URL]]:="ContentElementLocations"
elementlocationQ[str_String]:="ContentElementLocations"/;FileExistsQ[str]
elementlocationQ[_]:="ContentElements"


getNotebookResourcePublisher[nbo_]:=Block[{cell, publisherid},
	NotebookFind[nbo, "PublisherID", All, CellTags];
	cell=NotebookRead[nbo];
	publisherid=scrapeCellText[cell];
	If[StringQ[publisherid]&&StringLength[publisherid]>0,
		publisherid,
		System`$PublisherID
	]
]

getNotebookSubmissionNotes[nbo_]:=Block[{cell, notes},
	NotebookFind[nbo,"SubmissionNotes", All, CellTags];
	cell=NotebookRead[nbo];
	notes=scrapeCellText[cell];
	If[StringQ[notes]&&StringLength[notes]>0,
		notes,
		Missing[]
	]
]
getCreateResourcePublisher[nb_]:=scrapeoneTextCell["PublisherID",nb]

scrapecreateResourceNotebook[id_, nb_]:=DeleteCases[DeleteMissing@AssociationMap[
	scrapecreateresourceNotebook[#,nb]&,{"Name","Description","Details",
		"Keywords","ExternalLinks","SeeAlso","Originator","ContributorInformation"}
],{}]


findCellTags[nb_, tag_] := 
 Quiet[Cases[nb, c_Cell /; ! FreeQ[Options[c, CellTags], tag] :> c, Infinity],Options::optnf]

findCellTypes[nb_, type_] := 
 Cases[nb, c:Cell[_,type,___] :> c, Infinity]

$requiredKeys={"Name","Description","UUID"};
$listKeys={"Keywords","SeeAlso","ExternalLinks"};
scrapecreateresourceNotebook[key:("Name"|"Description"|"Details"|"Keywords"|"Originator"|"ContributedBy"), nb_]:=Block[{cells=findCellTags[nb, "Resource"<>key], str},
	If[Length[cells]===1,
		str=scrapeCellText[First[cells]];
		If[StringQ[str],
			formatScrapedString[key,str],
			If[MemberQ[$requiredKeys,key],
				Throw[$Failed]
				,
				Missing[]
			]
		]
		,
		If[MemberQ[$requiredKeys,key],
				Throw[$Failed]
				,
				Missing[]
			]
	]
]

scrapecreateresourceNotebook[key:("SeeAlso"|"ExternalLinks"), nb_]:=Block[{cells=findCellTags[nb, "Resource"<>key], cell},
	If[Length[cells]===1,
		cell=First[cells];
		If[Head[cell]==Cell,
			formatScrapedCell[key,cell],
			If[MemberQ[$requiredKeys,key],
				Throw[$Failed]
				,
				Missing[]
			]
		]
		,
		If[MemberQ[$requiredKeys,key],
				Throw[$Failed]
				,
				Missing[]
			]
	]
]

scrapecreateresourceNotebook["ContributorInformation", nb_]:=DeleteCases[DeleteMissing@AssociationMap[
	scrapecreateresourceNotebook[#,nb]&,{"ContributedBy"}
],{}]

formatScrapedString["Keywords",str_]:=StringTrim[StringSplit[str,"\n"]]
formatScrapedCell["ExternalLinks",expr_]:=scrapeExternalLinks[expr]
formatScrapedCell["SeeAlso",expr_]:=scrapeRelatedResourceUUIDs[expr]
formatScrapedString[_,str_]:=str

scrapeoneTextCell[ct_,nb_]:=scrapeonetextCell[findCellTags[nb, ct]]

scrapeonetextCell[cells_]:=Block[{str},
	If[Length[cells]===1,
		str=scrapeCellText[First[cells]];
		If[StringQ[str],
			str,
			Missing[]
		]
		,
		Missing[]
	]
]

celltype[key_]:="Resource"<>key

takeSingleValue[l_]:=If[MatchQ[l,{_}],First[l],l]

scrapecreateresourceNotebook["SourceMetadata", nb_]:=takeSingleValue/@DeleteCases[DeleteMissing[AssociationMap[scrapeSourceMetadata[#,nb]&,
	{"Creator","Contributor","GeographicCoverage","TemporalCoverage","Date","Citation",
	"Description","Language","Publisher","Rights","Source","Title"}]],{}]

smdcelltag[key_]:=key<>"SMD"

(*
scrapeSMDCell["Date",cell_]:=scrapeDate[First[cell]]
scrapeSMDCell[_,cell_]:=scrapeStringOrEntity[First[cell]]
*)
scrapeSMDCell["Rights"|"Source",cell_Cell]:=Block[{$allowNotLinkStrings=True},
	scrapeExternalLinks[cell]
]

scrapeSMDCell[key_,cell_Cell]:=scrapeSMDCell[key,First[cell]]
scrapeSMDCell[_,str_String]:=StringTrim[StringSplit[str,"\n"]]/.""->Nothing

$multipleValueSMDkeys=("Creator"|"Contributor"|"GeographicCoverage"|"TemporalCoverage"|"Date"|"Publisher"|"Language"|"Rights"|"Source"|"Citation");

scrapeSourceMetadata[key:$multipleValueSMDkeys,nb_]:=Block[{cells=findCellTags[nb, smdcelltag[key]], exprs},
	If[Length[cells]>0,
		exprs=scrapeSMDCell[key,First[cells]];
		DeleteMissing[exprs]
		,
		Missing[]
	]
]

scrapeSourceMetadata[key:("Description"|"Title"),nb_]:=scrapeonetextCell[findCellTags[nb, smdcelltag[key]]]

findcreateNotebookSections[nb_, section_, level_:{4,6}]:=Cases[nb, ( CellGroupData[_?(! FreeQ[#, section] &), ___]), level]  

scrapecreateresourceNotebookOneString[key_,nb_, style_:"CreateResourceText"]:=Block[{sec=findcreateNotebookSections[nb, key<>"Section"], res},
	If[Length[sec]===1,
		res=Cases[sec, Cell[str_ | _[str_String], style, ___] :> str, 6];
		If[Length[res]===1,
			getStringExpr[First[res]]
			,
			Missing[]
		]
		,
		Missing[]
	]
]

scrapeCellText[cell_]:=getStringExpr[First[cell]]

getStringExpr[box_BoxData]:=getStringExpr[First[box]]
getStringExpr[box:RowBox[l:{_String...}]]:=StringJoin[l]
getStringExpr[boxes_]:=With[{res=boxes//.RowBox[l:{_String...}]:>StringJoin[l]},
	If[FreeQ[res,RowBox],
		getStringExpr[res]
		,
		Missing[]
	]
]/;!FreeQ[boxes,RowBox[{_String...}]]
getStringExpr[boxes_]:=getStringExpr[First[Cases[boxes,_BoxData,Infinity]]]/;!FreeQ[boxes,BoxData]
getStringExpr[str_String]:=Missing[]/;StringMatchQ[str, Whitespace ...]
getStringExpr[str_String]:=StringTrim[ToExpression[str]]/;StringMatchQ[str,"\"*\""]
getStringExpr[str_String]:=StringTrim[str]
getStringExpr[expr_]:=Missing[]


scrapeRelatedResourceUUIDs[cell_]:=DeleteDuplicates[Flatten[scraperelatedResourceUUIDs[First[cell]]]]

scraperelatedResourceUUIDs[uuid_String]:=uuid/;uuidQ[uuid]
scraperelatedResourceUUIDs[str_String]:=StringTrim[StringSplit[str,"\n"]]
scraperelatedResourceUUIDs[cellcontents_]:=
StringTrim[
  Flatten[Join[StringSplit[Cases[cellcontents, _String, 2], "\n"],
    Cases[cellcontents, InterpretationBox[_, interp_, ___] :> interp["UUID"], 
     5]
    ]]] /. "" -> Nothing


scrapeExternalLinks[cell_]:=DeleteDuplicates[Flatten[scrapeexternalLinks[First[cell]]]]

scrapeexternalLinks[str_String]:=scrapeexternalLinkString/@StringTrim[fixCurlyQuotes[StringSplit[str,"\n"]]]
scrapeexternalLinks[boxes_]:=Join[scrapeexternalLinkBoxes[boxes],
 scrapeexternalLinkStrings[boxes]] 

scrapeexternalLinkBoxes[boxes_]:=Cases[
  Cases[boxes, box_BoxData :> ToExpression[box], 
   5], _String | _URL | _Hyperlink, {1}]

scrapeexternalLinkStrings[boxes_]:=With[{strs=
	Flatten[StringSplit[Cases[boxes, _String, 2], "\n"]] /. "" -> Nothing},
	Flatten[scrapeexternalLinkString/@fixCurlyQuotes[strs]]
]

$allowNotLinkStrings=False;
scrapeexternalLinkString[str_String]:=ToExpression[str]/;!StringFreeQ[str,"Hyperlink["]
scrapeexternalLinkString[str_String]:=ToExpression[str]/;!StringFreeQ[str,"URL["]
scrapeexternalLinkString[str_String]:=With[{res=Quiet[Interpreter["URL"][str]]},
	If[StringQ[res],res,
		If[$allowNotLinkStrings,str,{}]]
]

fixCurlyQuotes[str_]:=StringReplace[str, {"\[OpenCurlyDoubleQuote]" -> "\"", 
  "\[CloseCurlyDoubleQuote]" -> "\""}]

scrapeStringOrEntity[ent_Entity]:=ent
scrapeStringOrEntity[str_String]:=With[{ent=ToExpression[str]},
	If[FailureQ[ent],Missing[],ent]]/;!StringFreeQ[str,"Entity"]
scrapeStringOrEntity[str_String]:=getStringExpr[str]
scrapeStringOrEntity[_Missing]:=Missing[]
scrapeStringOrEntity[boxes_]:=With[{res=ToExpression[First[Cases[boxes,_TemplateBox,Infinity]]]},
	If[FreeQ[res,TemplateBox],
		scrapeStringOrEntity[res],
		Missing[]
	]
]/;!FreeQ[boxes,TemplateBox]
scrapeStringOrEntity[expr_]:=scrapeStringOrEntity[getStringExpr[expr]]


scrapeDate[do_DateObject]:=do
scrapeDate[str_String]:=With[{ent=ToExpression[str]},
	If[FailureQ[ent],Missing[],ent]]/;!StringFreeQ[str,"DateObject"]
scrapeDate[date:(_List|_Integer)]:=With[{res=DateObject[date]},
	If[DateObjectQ[res],res,Missing[]]]
scrapeDate[str_String]:=With[{res=DateObject[getStringExpr[str]]},
	If[DateObjectQ[res],res,Missing[]]]
scrapeDate[_Missing]:=Missing[]
scrapeDate[boxes_]:=With[{res=ToExpression[First[Cases[boxes,_TemplateBox,Infinity]]]},
	If[FreeQ[res,TemplateBox],
		scrapeDate[res],
		Missing[]
	]
]/;!FreeQ[boxes,TemplateBox]
scrapeDate[expr_]:=scrapeDate[getStringExpr[expr]]

getExpression[val_]:=With[{boxes=Cases[val, _BoxData, {0,6}]},
	If[Length[boxes]===1,
		Check[ToExpression[First@boxes],Missing[]]
		,
		Missing[]
	]
]


notebookNeeds[]:=(Needs["CloudObject`"];Needs["ResourceSystemClient`"];Needs["DataResource`"];)

importDataResourceNotebookContent[obj:HoldPattern[_CloudObject|_LocalObject]]:=Import[obj]
importDataResourceNotebookContent[obj:(_URL|_File)]:=Import[First[obj]]
importDataResourceNotebookContent[expr_]:=expr

createExampleNotebookPlaceholder[]:=("ExampleNotebook"->CreateDocument[{}, Visible -> False])

updateExampleNotebook[nb_,idname_,expr_,visible_,exnbo_NotebookObject]:=Block[{exnb},
	exnb=makeexampleNotebook[nb,idname,expr];
	If[Head[exnb]==Notebook,
		CreateDocument[exnb,exnbo,Visible->visible]
		,
		$Failed
	]
]

makeExampleNotebook[nb_,idname_,expr_,visible_]:=Block[{exnb},
	exnb=makeexampleNotebook[nb,idname,expr];
	If[Head[exnb]==Notebook,
		CreateDocument[exnb,Visible->visible]
		,
		$Failed
	]
]

makeexampleNotebook[nb_,{id_, name_},expr_]:=Block[{group},
	group=replaceCreateNotebookExampleSymbols[
		cleanCreateNotebookExamples[takeCreateNotebookExamples[nb]], id,expr];
	If[Head[group]===CellGroupData,
		Notebook[{exampleHeaderCell[If[StringQ[name],name, id]],group}, DockedCells->{},Options[nb, StyleDefinitions]]
		,
		$Failed
	]
]

exampleHeaderCell[str_String]:=exampleHeaderCell[Cell[str,"Title"]]
exampleHeaderCell[___]:=Nothing

takeCreateNotebookExamples[nb_]:=With[{groups=Cases[nb, 
 HoldPattern[
   CellGroupData][{_?(!FreeQ[#, "DataResourceExampleArea"] &), ___}, ___], 8]},
    If[Length[groups]>0,
    	First[groups]
    	,
    	$Failed
    ]
]
    
cleanCreateNotebookExamples[group_]:=Replace[group, 
	Cell[_, "DataResourceSubsection" | "DataResourceMoreInfo", ___] :> Nothing, 5]

replaceCreateNotebookExampleSymbols[group_,str_, expr_]:=
	Replace[group,
		{Cell[input_,"Input",rest___]:>Cell[replaceCreateNotebookExampleInput[input,str,expr],"Input",rest],
			Cell[input_,"Output",rest___]:>Cell[replaceCreateNotebookExampleOutput[input,str,expr],"Output",rest]
		}
		,40
	]


replaceCreateNotebookExampleInput[input_,str_,expr_]:=With[{boxes=RowBox[{"ResourceObject", "[", ToString[str, InputForm], "]"}]},
	replacePriority2[replacePriority1[input,str],boxes]
]

replaceCreateNotebookExampleOutput[input_,str_,expr_]:=With[{boxes=ToBoxes[expr]},
	replacePriority2[replacePriority1[input,str],boxes]
]

replacePriority1[group_,str_]:=Replace[
  group, {
  	RowBox[{"$$Object", "[", "\"\<FullContent\>\"", "]"}] -> 
  		RowBox[{"ResourceData", "[", RowBox[{ToString[str, InputForm], ",", "All"}], "]"}],
   "$$ObjectData" ->  RowBox[{"ResourceData", "[", RowBox[{ToString[str, InputForm], ",", "All"}], "]"}],  
   "$$Data" -> RowBox[{"ResourceData", "[", ToString[str, InputForm], "]"}]
   }, 50]

replacePriority2[group_,boxes_]:=Replace[group, "$$Object" -> boxes, 50]


DataResource`$$ContentConversion[as_Association] := 
 DataResource`Private`importDataResourceNotebookContent /@ as/; AssociationQ[as]
DataResource`$$ContentConversion[expr_] := (Message[ResourceObject::contfmt];expr)



(* deprecated *)


createSourceMetadataDetailCells[nbo_,id_]:=With[{cells=sourceMetadataDetailCells[id]},
	DataResource`Private`smdsectionOpen[id]=True;
	NotebookFind[nbo, "SourceMetadataDetails", All, CellTags];
	NotebookWrite[nbo,cells];
]

sourceMetadataDetailCells[id_]:=Cell[CellGroupData[{sourceMetadataDetailsHeaderCell[id], 
	Cell[
    CellGroupData[{Cell["", "DataResourceCreatorSMDSection"], 
    	Cell["", "DataResourceCreatorSMDMoreInfo", CellOpen -> False],
      Cell["", "DataResourceSMDTextInput", CellTags -> "CreatorSMD"]}, Open]],
      Cell[
    CellGroupData[{Cell["", "DataResourceContributorsSMDSection"], 
    	Cell["", "DataResourceContributorsSMDMoreInfo", CellOpen -> False],
      Cell["", "DataResourceSMDTextInput", CellTags -> "ContributorSMD"]}, Open]],
   Cell[
    CellGroupData[{Cell["", "DataResourceTitleSMDSection"], 
    	Cell["", "DataResourceTitleSMDMoreInfo", CellOpen -> False],
      Cell["", "DataResourceSMDTextInput", CellTags -> "TitleSMD"]}, Open]], 
   Cell["", "DataResourceDescriptionSMDSection"], 
       Cell[
    	CellGroupData[{Cell["", "DataResourceDescriptionSMDMoreInfo",  CellOpen -> False], 
      Cell["", "DataResourceSMDTextInput", CellTags -> "DescriptionSMD"]}, Open]], 
  	Cell["", "DataResourceDateSMDSection"], 
   	Cell[CellGroupData[{Cell["", "DataResourceDateSMDMoreInfo", CellOpen -> False], 
      Cell["", "DataResourceSMDTextInput", CellTags -> "DateSMD"]},  Open]], 
   Cell[CellGroupData[{Cell["", "DataResourcePublisherSMDSection"], 
   		Cell["", "DataResourcePublisherSMDMoreInfo", CellOpen -> False],
      Cell["", "DataResourceSMDTextInput",  CellTags -> "PublisherSMD"]}, Open]], 
 	Cell[CellGroupData[{Cell["", "DataResourceGeographicCoverageSMDSection"], 
 		Cell["", "DataResourceGeographicCoverageSMDMoreInfo", CellOpen -> False],
      Cell["", "DataResourceSMDTextInput", CellTags -> "GeographicCoverageSMD"]}, Open]], 
   Cell[CellGroupData[{Cell["", "DataResourceTemporalCoverageSMDSection"], 
   	Cell["", "DataResourceTemporalCoverageSMDMoreInfo", CellOpen -> False],
      Cell["", "DataResourceSMDTextInput",  CellTags -> "TemporalCoverageSMD"]}, Open]], 
       Cell[CellGroupData[{Cell["", "DataResourceLanguageSMDSection"], 
   	Cell["", "DataResourceLanguageSMDMoreInfo", CellOpen -> False],
      Cell["", "DataResourceSMDTextInput",  CellTags -> "LanguageSMD"]}, Open]], 
   Cell[CellGroupData[{Cell["", "DataResourceRightsSMDSection"], 
   	Cell["", "DataResourceRightsSMDMoreInfo", CellOpen -> False],
      Cell["", "DataResourceSMDTextInput", CellTags -> "RightsSMD"]}, Open]]}, 
     Dynamic[DataResource`Private`smdsectionOpen[id]]], CellContext -> "Global`"]


sourceMetadataDetailsHeaderCell[id_]:=Cell["DETAILED SOURCE INFORMATION", "DataResourceSubsection",
 Editable->False, Deletable->False, ShowCellBracket->True,
 CellMargins->{{50, Automatic}, {Automatic, Automatic}},
 Evaluatable->False, CellFrame->0,"WholeCellGroupOpener"->True,
	CellDingbat->DataResource`Private`sourceMetadataDetailsToggleButtonBox[id],
 CellContext->"Global`"]
 
sourceMetadataDetailsCreateButtonBox[id_]:=TagBox[
    GridBox[{{
    ButtonBox[GraphicsBox[{
    {RGBColor[
      NCache[
       Rational[61, 255], 0.23921568627450981`], 
      NCache[
       Rational[122, 255], 0.47843137254901963`], 
      NCache[
       Rational[158, 255], 0.6196078431372549]], 
     TagBox[PolygonBox[NCache[{{0, 0}, {0, 1}, {1, Rational[1, 2]}}, {{0, 0}, {0, 1}, {1, 0.5}}]],
      "Triangle"]}},
   ImageSize->{15, 15}],
  Appearance->None,
  ButtonFunction:>((
  Needs["ResourceSystemClient`"];
  Needs["DataResource`"];
  DataResource`Private`createSourceMetadataDetailCells[EvaluationNotebook[],id])&) ,
  Evaluator->Automatic,
  Method->"Preemptive"]
    }, {""}}, AutoDelete -> False, 
      GridBoxItemSize -> {"Columns" -> {{Automatic},2}, 
    "Rows" -> {All, 0.025}}]
 , "Grid"]
(*
sourceMetadataDetailsToggleButtonBox[id_]:=ButtonBox[DynamicBox[ToBoxes[
      If[TrueQ[DataResource`Private`smdsectionOpen[id]], "\[DownPointer]", "\[RightPointer]"], StandardForm]],
  Appearance->None,
  ButtonFunction:>((
  Needs["ResourceSystemClient`"];
  Needs["DataResource`"];
  DataResource`Private`smdsectionOpen[id]=If[TrueQ[DataResource`Private`smdsectionOpen[id]],False,True])&) ,
  Evaluator->Automatic,
  Method->"Queued"]
  *)
sourceMetadataDetailsToggleButtonBox[id_]:=ToBoxes[Button[
	Dynamic[
       Grid[{{
       	If[TrueQ[DataResource`Private`smdsectionOpen[id]], 
		Graphics[{RGBColor[61/255, 122/255, 158/255], 
  Triangle[{{0, 1}, {1, 1}, {1/2, 0}}]}, ImageSize -> {15, 15}], 
		Graphics[{RGBColor[61/255, 122/255, 158/255], 
  Triangle[{{0, 0}, {0, 1}, {1, 1/2}}]}, ImageSize -> {15, 15}]]
  }, {Null}},   ItemSize -> {{{Automatic},3}, {Full, 0.025}}]
  
  ],
	DataResource`Private`smdsectionOpen[id]=!TrueQ[DataResource`Private`smdsectionOpen[id]],
	Appearance->None,
  	Method->"Queued"
]]

DataResource`Private`sortingCheckBox[name_]:=Row[{Checkbox[False, {False, name}], 
	Spacer[10], Style[name,"CheckBoxBarOutput"]}]

End[] (* End Private Context *)

EndPackage[]