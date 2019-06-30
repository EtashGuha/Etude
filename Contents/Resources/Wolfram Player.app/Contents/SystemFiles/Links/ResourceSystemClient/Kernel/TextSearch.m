(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["ResourceSystemClient`"]
(* Exported symbols added here with SymbolName::usage *)  


Begin["`Private`"] (* Begin Private Context *) 

$AllowResourceTextSearch=TrueQ[$VersionNumber>=11.0]

$textSearchIndexKeys:=Join[{"Name","Keywords","Description","Details","ResourceType","ContributedBy"},allSortingProperties[]]

$ResourceSearchIndexName="ResourceObjectSearchIndex";

buildLocalResourceSearchIndex[]:=buildLocalResourceSearchIndex[Infinity]

$indexedResources={};
buildLocalResourceSearchIndex[timeout_]:=Block[{files, t0=SessionTime[], toIndex=Complement[$localResources,$indexedResources], res},
	$searchProgressContent="Adding local resources to a search index \[Ellipsis]";
	If[checkDeleteSearchIndex[],
		Catch[(
			If[TrueQ[(SessionTime[]-t0)>timeout],
				$indexedResources=DeleteDuplicates[Flatten[{$indexedResources,Cases[toIndex, {before___, #, ___} :> before, {0}]}]];
				Throw[Null,"timeout"]];
			createTextSearchIndexFileCheck[#]
			)&/@toIndex;
			
			$indexedResources=$localResources;
		If[NumberQ[timeout],
			$searchIndexUpdateNeeded=True;
			,
			res=buildlocalResourceSearchIndex[];
			$searchIndexUpdateNeeded=False;
			res
		]
		,"timeout"]
		,
		$AllowResourceTextSearch=False;
		$Failed		
	]
]/;$AllowResourceTextSearch

buildlocalResourceSearchIndex[]:=(
	$searchProgressContent="Creating a local search index \[Ellipsis]";
	buildlocalresourceSearchIndex[]
)

buildlocalresourceSearchIndex[]:=With[{res=Quiet[CreateSearchIndex[textSearchDirectory[],$ResourceSearchIndexName]]},
	If[Head[res]===SearchIndexObject,
		$AllowResourceTextSearch=TrueQ[$VersionNumber>=11.0];
		res
		,
		$Failed
	]
]

checkCreateSearchIndex[args___]:=Quiet[With[{index=CreateSearchIndex[args]},
	If[Head[index]===SearchIndexObject,
		index,
		$AllowResourceTextSearch=False;
		$Failed
	]	
]]

updateLocalResourceSearchIndex[]:=updatelocalResourceSearchIndex[Quiet[SearchIndexObject[$ResourceSearchIndexName]]]
updatelocalResourceSearchIndex[HoldPattern[sio_SearchIndexObject]]:=(
	$searchProgressContent="Searching for local resources (updating index) \[Ellipsis]";
	With[{res=Quiet[UpdateSearchIndex[sio]]},
		If[res===$Failed,
			asynchRebuildSearchIndex[]
			,
			sio]
	];
	$searchIndexUpdateNeeded=False;)

updatelocalResourceSearchIndex[_]:=asynchRebuildSearchIndex[]

$asynchBuildResourceIndexTimeout=3.
asynchRebuildSearchIndex[]:=(
		$AllowResourceTextSearch=False;
		SessionSubmit[
			ScheduledTask[
			$AllowResourceTextSearch=TrueQ[$VersionNumber>=11.0];
			buildLocalResourceSearchIndex[$asynchBuildResourceIndexTimeout];
			,{1}]
		]
		)/;ResourceSystemClient`$AsyncronousResourceInformationUpdates
		
checkDeleteSearchIndex[]:=
	Quiet[
		DeleteSearchIndex[SearchIndexObject[$ResourceSearchIndexName]];
		SearchIndexObject[$ResourceSearchIndexName]===$Failed
	]


$updateResourceSearchIndex=True;

addToLocalSearchIndex[id_]:=addToLocalSearchIndex[id, getResourceInfo[id],resourceDirectory[id]]
addToLocalSearchIndex[id_, info_]:=addToLocalSearchIndex[id, info,resourceDirectory[id]]
addToLocalSearchIndex[id_, info_, dir_]:=With[{indexfile=createtextSearchIndexFile[id,info, dir]},
	If[TrueQ[Quiet[FileExistsQ[indexfile]]]&&$updateResourceSearchIndex,
		updateLocalResourceSearchIndex[]		
	]
]

createTextSearchIndexFileCheck[id_]:=With[{indexfile=textSearchIndexFile[id]},
	If[FileExistsQ[indexfile],
		indexfile,
		createTextSearchIndexFileCheck[id,indexfile,resourceDirectory[id]]
	]
]

indexIgnoreFlag[dir_]:=FileNameJoin[{dir,"indexingore"}]

createTextSearchIndexFileCheck[id_,indexfile_,dir_]:=Null/;FileExistsQ[indexIgnoreFlag[dir]]
createTextSearchIndexFileCheck[id_,indexfile_,dir_]:=createtextSearchIndexFile[id,indexfile,getResourceInfo[id],dir]

createtextSearchIndexFile[id_,info_Association, dir_]:=createtextSearchIndexFile[id,textSearchIndexFile[id],info, dir]

createtextSearchIndexFile[id_,indexfile_,info_Association, dir_]:=Block[{string},
	If[marketplacebasedResourceQ[info],
		Put[indexIgnoreFlag[dir]];
		Null
		,
		createDirectory[FileNameDrop[indexfile]];
		string=Quiet[createtextsearchIndexString[info]];
		If[!TrueQ[Quiet[StringLength[string]>0]],
			Return[Null]
		];
		Export[indexfile,string,"String"];
		indexfile
	]
]/;DirectoryQ[dir]
	
createtextSearchIndexFile[___]:=Null	

createtextsearchIndexString[info_]:=Block[{sourceinfostring,infostring},
	infostring=StringJoin[createTextSearchProperty[info,#]&/@$textSearchIndexKeys];
	sourceinfostring=createTextSearchSourceInfoString[info];
	StringJoin[{infostring,"\n",sourceinfostring}]
]

textSearchDirectory[]:=With[{dir=FileNameJoin[{resourceCacheDirectory[],"searchdata"}]},
	createDirectory[dir];
	textSearchDirectory[]=dir
]

textSearchIndexFile[id_]:=FileNameJoin[{textSearchDirectory[], StringTake[id, 3], id<>".txt"}]

createTextSearchProperty[info_,key_]:=""/;!KeyExistsQ[info,key]
createTextSearchProperty[info_,key_]:=createtextSearchProperty[key,info[key]]

createtextSearchProperty[key_String,l_List]:=StringJoin[createtextSearchProperty[key,#]&/@l]
createtextSearchProperty[key_String,str_String]:=
	StringJoin[{key," ",textSearchEscape[str]," \n"}]
createtextSearchProperty[key_String,ent:HoldPattern[_Entity]]:=createtextSearchProperty[key,CommonName[ent]]

createtextSearchProperty[__]:=""

createTextSearchSourceInfoString[info_Association]:=""/;!KeyExistsQ[info,"SourceMetadata"]
createTextSearchSourceInfoString[info_Association]:=createtextSearchSourceInfoString[DeleteMissing[info["SourceMetadata"]]]
createTextSearchSourceInfoString[_]=""

createtextSearchSourceInfoString[sourceinfo_Association]:=KeyValueMap[createtextSearchProperty,sourceinfo]
createtextSearchSourceInfoString[_]:=""

tsecapeRules[]:=(tsecapeRules[]=(#->"textsearch"<>#)&/@$textSearchIndexKeys)

textSearchEscape[str_]:=ToLowerCase[StringReplace[str,tsecapeRules[]]]

clearSearchIndex[id_]:=clearsearchIndex[textSearchIndexFile[id]]
clearsearchIndex[file_]:=(DeleteFile[file];
	UpdateSearchIndex[$ResourceSearchIndexName];
)/;FileExistsQ[file]

textSearchResources[q_]:=textsearchResources[q]/;$AllowResourceTextSearch

textSearchResources[_]:={}

textsearchResources[q_]:=Block[{index, contobjs},
    index=Quiet[SearchIndexObject[$ResourceSearchIndexName]];
    If[Head[index]=!=SearchIndexObject,
        index=buildLocalResourceSearchIndex[]
        ,
        If[$searchIndexUpdateNeeded,
        	updatelocalResourceSearchIndex[index]
        ]
    ];
    $searchProgressContent="Searching for local resources \[Ellipsis]";
    If[Head[index]===SearchIndexObject,
    	contobjs=normalSearchResults[TextSearch[index, preprocessQueryExpression[q]]];
    	getSearchResultIDs[contobjs]
    	,
    	$Failed
    ]    
]

getSearchResultIDs[l_List]:=Select[FileBaseName[#["FileName"]]&/@Cases[l,_ContentObject],StringQ]

normalSearchResults[HoldPattern[sro_SearchResultObject]]:=Normal[sro]
normalSearchResults[expr_]:=expr


preprocessQueryExpression[str_String]:=textSearchEscape[str]
preprocessQueryExpression[expr_]:=ReplaceRepeated[expr,HoldPattern[Rule][f_String,val_]:>preprocessQueryRule[f,val]]

preprocessQueryRule["Name",str_String]:=SearchAdjustment[StringJoin[{"Name"," ",textSearchEscape[str]}],MaxWordGap->8]
preprocessQueryRule["Keywords",str_String]:=SearchAdjustment[StringJoin[{"Keywords"," ",textSearchEscape[str]}],MaxWordGap->4]
preprocessQueryRule["Description",str_String]:=SearchQueryString[StringJoin[{"Description"," ",textSearchEscape[str]}]]
preprocessQueryRule["Details",str_String]:=SearchQueryString[StringJoin[{"Details"," ",textSearchEscape[str]}]]
preprocessQueryRule["ContributedBy",str_String]:=SearchAdjustment[StringJoin[{"ContributedBy"," ",textSearchEscape[str]}],MaxWordGap->8]

preprocessQueryRule[f_String,str_String]:=SearchAdjustment[StringJoin[{f," ",textSearchEscape[str]}],MaxWordGap->2]/;MemberQ[allSortingProperties[],f]
preprocessQueryRule[f_String,expr_]:=Replace[expr,str_String:>preprocessQueryRule[f,str],{0,Infinity}]
preprocessQueryRule[__]:={}

End[] (* End Private Context *)

EndPackage[]

SetAttributes[{},
   {ReadProtected, Protected}
];