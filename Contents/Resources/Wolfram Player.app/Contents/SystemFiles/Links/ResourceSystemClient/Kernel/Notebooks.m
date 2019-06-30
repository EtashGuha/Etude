(* Wolfram Language Package *)

BeginPackage["ResourceSystemClient`"]

ResourceSystemClient`CreateResourceNotebook

Begin["`Private`"] (* Begin Private Context *) 


(Unprotect[#]; Clear[#])& /@ {ResourceSystemClient`CreateResourceNotebook}

ResourceSystemClient`CreateResourceNotebook[args___]:= Catch[createResourceObjectNotebook[args]]

$availableResourceCreateTemplates={"DataResource","NeuralNet"};

Options[ResourceSystemClient`CreateResourceNotebook]=Options[createResourceObjectNotebook]={"SuppressProgressBar"->False}

createResourceObjectNotebook[opts:OptionsPattern[]]:=Block[{rtype},
	CreateDialog[
		rtype = "Choose a type"; 
		Column[{
			PopupMenu[Dynamic[rtype], Prepend[$availableResourceCreateTemplates,"Choose a type"]],
			ChoiceButtons[{"Create Notebook", "Cancel"}, 
	   			{DialogReturn[createResourceObjectNotebook[rtype,opts]], DialogReturn[$Canceled]}, 
	     		{{Enabled -> Dynamic[rtype =!= "Choose a type"],Method -> "Queued"}, {}}]
	     	}]
	     ]
]

createResourceObjectNotebook[rtype_String,opts:OptionsPattern[]]:=(
	loadResourceType[rtype];
	createresourceObjectNotebook[rtype, OptionValue["SuppressProgressBar"]]
)

createresourceObjectNotebook[rtype_,True]:=Block[{PrintTemporary},
	createResourceNotebook[rtype]
]

createresourceObjectNotebook[rtype_,_]:=createResourceNotebook[rtype]

createResourceObjectNotebook[___]:=$Failed
createResourceNotebook[___]:=$Failed

createResourceFromNotebook[nb:(_NotebookObject|_Notebook)]:=With[{rtype=creationNotebookType[Options[nb,TaggingRules]]},
	createResourceFromNotebook[rtype,nb]
]

createResourceFromNotebook[None,nb_]:=(Message[ResourceObject::ronb1];$Failed)


createResourceFromNotebook[rtype_,nb_]:=(
	loadResourceType[rtype];
	repositoryCreateResourceFromNotebook[rtype,nb]
)


submitResourceFromNotebook[nbo_NotebookObject, update_]:=With[{rtype=creationNotebookType[Options[nbo,TaggingRules]]},
	submitResourceFromNotebook[rtype,nbo,update]
]
submitResourceFromNotebook[None,nb_,_]:=(Message[ResourceObject::ronb1];$Failed)
submitResourceFromNotebook[rtype_,nb_,update_]:=(
	loadResourceType[rtype];
	repositorySubmitResourceFromNotebook[rtype,nb,update]
)



repositoryCreateResourceFromNotebook[_,nb_]:=(Message[ResourceObject::ronb2];$Failed)


creationNotebookType[opts_]:=creationnotebookType[Lookup[opts,TaggingRules,{}]]
creationnotebookType[tr_List]:=Lookup[tr,"ResourceType",None]/;TrueQ[Lookup[tr,"ResourceCreateNotebook",None]]






(* Scraping Tools *)

$ScrapePublisher=False;

notebookScrapeNeeds[]:=(Needs["CloudObject`"];Needs["ResourceSystemClient`"];)

findCellTags[nb_, tag_] := 
 Quiet[Cases[nb, c_Cell /; ! FreeQ[Options[c, CellTags], tag] :> c, Infinity],Options::optnf]

findCellTypes[nb_, type_] := 
 Cases[nb, c:Cell[_,type,___] :> c, Infinity]

$requiredKeys={"Name","Description","UUID"};


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

formatScrapedString["RelatedSymbols",str_]:=validateSymbolNames[StringTrim[StringSplit[str,"\n"]]]
formatScrapedString["Keywords",str_]:=StringTrim[StringSplit[str,"\n"]]
formatScrapedString[_,str_]:=str

validateSymbolNames[l:{_String...}]:=l/;Complement[l,Names["System`*"]]==={}

formatScrapedCell["ExternalLinks",expr_]:=scrapeExternalLinks[expr]

scrapeExternalLinks[cell_]:=DeleteDuplicates[scrapeexternalLinks[First[cell]]]

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

formatScrapedCell["SeeAlso",expr_]:=scrapeRelatedResourceUUIDs[expr]
scrapeRelatedResourceUUIDs[cell_]:=DeleteDuplicates[scraperelatedResourceUUIDs[First[cell]]]

scraperelatedResourceUUIDs[uuid_String]:=uuid/;uuidQ[uuid]
scraperelatedResourceUUIDs[str_String]:=StringTrim[StringSplit[str,"\n"]]
scraperelatedResourceUUIDs[cellcontents_]:=
StringTrim[
  Flatten[Join[StringSplit[Cases[cellcontents, _String, 2], "\n"],
    Cases[cellcontents, InterpretationBox[_, interp_, ___] :> interp["UUID"], 
     5]
    ]]] /. "" -> Nothing


repositoryCreateResourceFromNotebook[rtype_,nbo_]:=With[{res=scrapeDefinitonNotebook[rtype,nbo]},
	If[FailureQ[res],
		$Failed,
		res
	]	
]

scrapeDefinitonNotebook[rtype_,nbo_NotebookObject]:=(notebookScrapeNeeds[];
	scrapedefinitonNotebook[rtype,NotebookGet[nbo]])

scrapeDefinitonNotebook[rtype_,nb_Notebook]:=(notebookScrapeNeeds[];
	scrapedefinitonNotebook[rtype,nb])
	
scrapedefinitonNotebook[rtype0_,nb_]:=Block[{as, ro, id, rtype},
	rtype=If[StringQ[rtype0],
		rtype0,
		creationNotebookType[Options[nb,TaggingRules]]
	];
	If[StringQ[rtype],
		loadResourceType[rtype]
	];
	If[Head[nb]=!=Notebook,
		Return[$Failed]
	];
	
	as=scrapedefinitonnotebook[rtype,nb];
	id=as["UUID"];
	If[AssociationQ[as]&&StringQ[id],
		$localResources=DeleteCases[$localResources,id];
		ro=autoloadResource[as];
		If[FailureQ[updateScrapedExampleNotebook[rtype,nb,{id, as["Name"]},ro,False,as["ExampleNotebook"]]],
			as=KeyDrop[as,"ExampleNotebook"];
			ro=autoloadResource[as]
		];
		ro
		,
		Message[ResourceObject::invro];
		$Failed
	]
]

scrapedefinitonnotebook[rtype_,nb_]:=Block[{sp, cont, info, id,as,examples},
	id=CreateUUID[];
	info=scrapeDefinitonNotebookProperties[rtype,id, nb];
	examples=exampleNotebookPlaceholder[];
	If[!AssociationQ[info],
		Return[$Failed]
	];
	sp=scrapecreateDefinitionNotebookSortingProperties[rtype,nb];	
	cont=scrapeDefinitonNotebookContent[rtype,id, nb];
	If[TrueQ[$ScrapePublisher],
		$DefinitionNotebookPublisherID=scrapeDefinitonNotebookPublisherID[nb]
	];
	as=Association[
		"ResourceType"->rtype,
		"UUID"->id,
		info,
		sp,
		cont,
		"SourceMetadata"->scrapeDefinitonNotebookProperty[rtype,"SourceMetadata", nb],
		examples,
		"DefinitionNotebook" -> includeDefinitionNotebook[ rtype, nb ]
	];
	DeleteMissing @ as
]



$includedDefinitionNotebookFormat = { "GZIP", "NB" };


includeDefinitionNotebook[ rtype_? includeDefinitionNotebookQ, nb_NotebookObject ] :=
  includeDefinitionNotebook[ rtype, NotebookGet @ nb ];

includeDefinitionNotebook[ rtype_? includeDefinitionNotebookQ, nb_Notebook ] :=
  <|
      "Data" -> ExportByteArray[ nb, $includedDefinitionNotebookFormat ],
      "Format" -> $includedDefinitionNotebookFormat
  |>;

includeDefinitionNotebook[ ___ ] :=
  Missing[ "NotAvailable" ];



(*
    To include the original full definition notebook, define this as True for a
    given resource type in the appropriate paclet. E.g.

    ResourceSystemClient`Private`includeDefinitionNotebookQ[ "Function" ] = True
*)
includeDefinitionNotebookQ[ rtype_ ] := False;




scrapeDefinitonNotebookContent[_,_,_]:=Association[]

scrapecreateDefinitionNotebookSortingProperties[rtype_,nb_]:=With[{props=
	resourceSortingProperties[rtype,False]},
	If[AssociationQ[props],
		AssociationMap[scrapecreatedefinitionNotebookSortingProperties[rtype,nb,#]&,Keys[props]]
		,
		Association[]
	]
]

scrapecreatedefinitionNotebookSortingProperties[rtype_,nb_,prop_]:=Block[{section},
	section = findCellTags[nb, "Resource"<>prop];
	Flatten[Cases[section, CheckboxBox[check:Except[False], ___] :> check, 10]]
]



scrapeDefinitonNotebookProperties[rtype_,id_, nb_]:=Join[
		scrapeCommonProperties[rtype,id,nb],
		scrapeResourceTypeProperties[rtype,id,nb]
]
	
	
scrapeCommonProperties[_,id_, nb_]:=
DeleteCases[DeleteMissing@AssociationMap[
	scrapeDefinitonNotebookProperty[rtype,#,nb]&,{"Name","Description","Details",
		"Keywords","ExternalLinks","SeeAlso","Originator","ContributorInformation","RelatedSymbols"}
],{}]


scrapeResourceTypeProperties[___]:=Association[]

scrapeDefinitonNotebookProperty[rtype_,key:("Name"|"Description"|"Details"|"Keywords"|"Originator"|"ContributedBy"|"RelatedSymbols"), nb_]:=Block[{cells=findCellTags[nb, "Resource"<>key], str},
	If[Length[cells]===1,
		str=scrapeCellText[First[cells]];
		If[StringQ[str],
			formatScrapedString[key,str],
			If[MemberQ[$requiredKeys,key],
				Message[ResourceObject::defkey,key];
				Throw[$Failed]
				,
				Missing[]
			]
		]
		,
		If[MemberQ[$requiredKeys,key],
				Message[ResourceObject::defkey,key];
				Throw[$Failed]
				,
				Missing[]
			]
	]
]

scrapeDefinitonNotebookProperty[rtype_,key:("SeeAlso"|"ExternalLinks"), nb_]:=Block[{cells=findCellTags[nb, "Resource"<>key], cell},
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

scrapeDefinitonNotebookProperty[rtype_,"ContributorInformation", nb_]:=DeleteCases[DeleteMissing@AssociationMap[
	scrapeDefinitonNotebookProperty[rtype,#,nb]&,{"ContributedBy"}
],{}]


scrapeDefinitonNotebookPublisherID[nb_]:=scrapeoneTextCell["PublisherID",nb]

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





(* scraped example notebook handling *)
exampleNotebookPlaceholder[]:=("ExampleNotebook"->CreateDocument[{}, Visible -> False])

updateScrapedExampleNotebook[rtype_,nb_,idname_,expr_,visible_,exnbo_NotebookObject]:=Block[{exnb},
	exnb=standaloneExampleNotebook[rtype,nb,idname,expr];
	If[Head[exnb]==Notebook,
		CreateDocument[exnb,exnbo,Visible->visible]
		,
		$Failed
	]
]

standaloneExampleNotebook[rtype_,nb_,{id_, name_},expr_]:=Block[{group},
	group=replaceScrapedNotebookExampleSymbols[rtype,
		cleanScrapedNotebookExamples[rtype,scrapeNotebookExamples[rtype,nb]], id,expr];
	If[Head[group]===CellGroupData,
		Notebook[{standaloneExampleHeaderCell[rtype,If[StringQ[name],name, id]],group}, DockedCells->{},
			Options[nb, StyleDefinitions], 
			Background->White]
		,
		$Failed
	]
]

standaloneExampleHeaderCell[_,str_String]:=standaloneExampleHeaderCell[_,Cell[str,"Title"]]
standaloneExampleHeaderCell[___]:=Nothing

scrapeNotebookExamples[_,nb_]:=With[{groups=Cases[nb, 
 HoldPattern[
   CellGroupData][{_?(!FreeQ[#, "ResourceExampleArea"] &), ___}, ___], 8]},
    If[Length[groups]>0,
    	First[groups]
    	,
    	$Failed
    ]
]
    
cleanScrapedNotebookExamples[_,group_]:=ReplaceRepeated[Replace[group, 
	Cell[_, "ResourceSubsection" | "ResourceMoreInfo"|"ResourceWhiteSection", ___] :> Nothing, 5], {before___, 
   HoldPattern[Cell][_, "Subsubsection", ___]} :> {before}]
   
   
replaceScrapedNotebookExampleSymbols[_,group_,str_, expr_]:=group

(* Source metadata scraping *)

takeSingleValue[l_]:=If[MatchQ[l,{_}],First[l],l]

scrapeDefinitonNotebookProperty[_,"SourceMetadata", nb_]:=takeSingleValue/@DeleteCases[DeleteMissing[AssociationMap[scrapeSourceMetadata[#,nb]&,
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

getResourcePublisher[nbo_]:=Block[{cell, publisherid},
	NotebookFind[nbo, "PublisherID", All, CellTags];
	cell=NotebookRead[nbo];
	publisherid=scrapeCellText[cell];
	If[StringQ[publisherid]&&StringLength[publisherid]>0,
		publisherid,
		System`$PublisherID
	]
]

getSubmissionNotes[nbo_]:=Block[{cell, notes},
	NotebookFind[nbo,"SubmissionNotes", All, CellTags];
	cell=NotebookRead[nbo];
	notes=scrapeCellText[cell];
	If[StringQ[notes]&&StringLength[notes]>0,
		notes,
		Missing[]
	]
]

End[] (* End Private Context *)

EndPackage[]


SetAttributes[{ResourceSystemClient`CreateResourceNotebook},
   {ReadProtected, Protected}
];

Get["ResourceSystemClient`CommonContentTools`"]
Get["ResourceSystemClient`CommonElementTools`"]
