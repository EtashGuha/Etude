(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {System`ResourceSearch}

BeginPackage["ResourceSystemClient`"]
(* Exported symbols added here with SymbolName::usage *)  

System`ResourceSearch
ResourceSystemClient`RandomResources
ResourceSystemClient`ResourceSearchData

Begin["`Private`"] (* Begin Private Context *) 

$ResourceSearchMaxItems=50;

Options[resourcesearch]=Options[System`ResourceSearch]={MaxItems->Automatic,
	Method->Automatic, System`ResourceSystemBase->Automatic};

System`ResourceSearch[args___]:=Catch[resourceSearch[args]]

ResourceSystemClient`ResourceSearchData=System`ResourceSearch;
ResourceSystemClient`ResourceSearchObjects[expr_,opts___?OptionQ]:=ResourceSearch[expr,"Objects",opts]

resourceSearch[q_?validResourceQueryQ, rest___]:=resourcesearch[parseResourceQuery[q],rest]
(*
resourceSearch[Rule[rtypes_,str_], rest___]:=resourceSearch[{rtypes,str}, rest]

resourceSearch[str_String, rest___]:=resourceSearch[{All,str}, rest]

resourceSearch[{rtypes_,str_}, rest___]:=resourcesearch[{rtypes,str}, rest]
*)
resourceSearch[]:=(Message[ResourceSearch::argt, ResourceSearch, 0, 1, 2];$Failed)

resourceSearch[expr_,___]:=(Message[ResourceSearch::invquery,expr];$Failed)

resourceSearch[___]:=$Failed

resourcesearch[q_, opts:OptionsPattern[]]:=resourcesearch[q, Automatic, opts]

$searchProgressContent="Searching for resources \[Ellipsis]";
resourcesearch[{rtypes_, q_}, resultformat:(_String|_Symbol|{_String..}), opts:OptionsPattern[]]:=Block[
	{list={},count,locations, temp, info,n,$progressID=Unique[]},
	If[!validSearchResultFormatQ[resultformat],
		Message[ResourceSearch::resformat,resultformat]
	];
	$searchProgressContent="Searching for resources \[Ellipsis]";
	n=OptionValue[MaxItems];
	count=Switch[n,
		Automatic|Infinity, Infinity,
		0,Return[{}],
		_Integer, If[n>0, n, Message[ResourceSearch::invcount,n];Throw[$Failed]],
		_,Message[ResourceSearch::invcount,n];Throw[$Failed]
	];	
    locations=DeleteDuplicates@Flatten[{getSearchLocations[OptionValue[Method]]}];
    If[!MatchQ[locations,{_String...}],
    	Message[ResourceSearch::invloc];
    	Throw[$Failed]
    ];
    temp=printTempOnce[$progressID,progressPanel[Dynamic[$searchProgressContent]]];
    list=If[MatchQ[locations,{}],
    	{},
   		resourceSearchLocation[First[locations],q,n,rtypes,count, Rest[locations],{},Association[opts]]
    ];
    clearTempPrint[temp];
	formatSearchResults[resultformat,minimalSearchInfo/@list]
]

resourcesearch[___]:=$Failed

resourceSearchLocation[location_,q_,n_,rtypes_,count_, nextlocations_,results_,opts_]:=Block[{list},
		list=resourcesearchLocation[location,q,n,count,rtypes,opts];
		If[Length[results]>0,
			list=DeleteDuplicates[Join[results,list],#1["UUID"]==#2["UUID"]&]
		];
		If[Length[list]>=count,	
			Take[list,UpTo[count]],
			If[Length[nextlocations]>0,
				resourceSearchLocation[First[nextlocations],q,n,rtypes,count, Rest[nextlocations],list,opts],
				list
			]
		]
]

resourceSearchLocation[__,results_,_]:=results

resourcesearchLocation["Local",q_,n_,_,rtypes_, opts_]:=(
		$searchProgressContent="Searching for local resources \[Ellipsis]";
		selectResourceTypes[resourcesearchLocal[q, n,Lookup[opts,Method,OptionValue[ResourceSearch,Method]]],rtypes]
		)
		
resourcesearchLocation["Cloud",q_,n_,count_,rtypes_,opts_]:=(
		$searchProgressContent="Searching for deployed resources \[Ellipsis]";
		selectResourceTypes[resourceSearchCloud[q, count, updateCloudIndexQ[Lookup[opts,Method,OptionValue[ResourceSearch,Method]]]],rtypes]
		)
		
resourcesearchLocation["ResourceSystemBase"|"Repository",q_,n_,count_,rtypes_,opts_]:=(
		$searchProgressContent="Searching for published resources \[Ellipsis]";   
		selectResourceTypes[resourcesearchResourceSystem[q, n,rtypes,
			Lookup[opts,Method,OptionValue[ResourceSearch,Method]],
			Lookup[opts,ResourceSystemBase,OptionValue[ResourceSearch,ResourceSystemBase]]],rtypes]
		)

resourcesearchLocation[___]:={}
		
$defaultSearchLocations={"ResourceSystemBase","Local"};
$allSearchLocations={"ResourceSystemBase","Local","Cloud"};

getSearchLocations[KeyValuePattern[("Locations"|"SearchLocations"|"Path"|Path)->Automatic]]:=$defaultSearchLocations
getSearchLocations[KeyValuePattern[("Locations"|"SearchLocations"|"Path"|Path)->All]]:=$allSearchLocations
getSearchLocations[KeyValuePattern[("Locations"|"SearchLocations"|"Path"|Path)->val_]]:=val
getSearchLocations[_]:=$defaultSearchLocations

formatSearchResults[HoldPattern["Dataset"|Dataset|Automatic],res_]:=createResourceDataset[res]
formatSearchResults["Object"|"Objects"|ResourceObject|"ResourceObject",res_]:=createResourceObjects[res]
formatSearchResults["List"|"Association"|"Associations"|List|Association,res_]:=Normal[createResourceDataset[res]]
formatSearchResults[key_String,res_]:=Lookup[res,key]/;KeyExistsQ[res[[1]],key]
formatSearchResults[keys_List,res_]:=KeyTake[res,keys]/;Complement[keys,Keys[res[[1]]]]=={}
formatSearchResults[key_,_]:=(Message[ResourceSearch::resformat,key];$Failed)

selectResourceTypes[info_,All|"All"|"all"]:=DeleteMissing[info]
selectResourceTypes[info_,rtypes_List]:=Select[DeleteMissing[info],MemberQ[getresourceType/@rtypes,Lookup[#,"ResourceType"]]&]
selectResourceTypes[info_,rtypes_]:=selectResourceTypes[info,{rtypes}]

minimalSearchInfo[info_]:=If[MemberQ[$loadedResources,info["UUID"]],usableResourceInfo[info],info]

updateCloudIndexQ[KeyValuePattern["UpdateCloudIndex"->val_]]:=val
updateCloudIndexQ[_]:=Automatic

resourceSearchCloud[q_, n_, _]:=resourcesearchLocal[q, n, Automatic]/;$CloudEvaluation

resourceSearchCloud[q_, count_, True]:=Block[{cloudindex=updateCloudResourceIndex[], info},
	If[Head[cloudindex]===Dataset,
		$cloudResourceIndex=cloudindex;
		resourceSearchCloud[q,count,Automatic]
		,
		{}
	]
]

resourceSearchCloud[q_, count_, update_]:=resourcesearchCloud[q,count, update]

resourcesearchCloud[q_String,count_, Automatic]:=resourceSearchDataset[q, count,$cloudResourceIndex]/;Head[$cloudResourceIndex]===Dataset
resourcesearchCloud[q_,count_, Automatic]:=resourceSearchContentObjects[q, count,$cloudResourceIndex,$cloudResourceSearchIndex]/;Head[$cloudResourceIndex]===Dataset
resourcesearchCloud[q_,count_, Automatic]:=resourceSearchCloud[q, count, True]

resourceSearchDataset[q_,count_, HoldPattern[ds_Dataset]]:=With[{info=Normal[createResourceQuery[q][ds]]},
	If[ListQ[info],
		DeleteCases[Quiet[Normal[Catch[autoloadResource[DeleteMissing[#]]]]&/@Take[info, UpTo[count]]],$Failed]
		,
		{}
	]
]


createResourceQuery[q_String]:=With[{sp = allSortingProperties[]},
	Query[
	Select[MemberQ[Lookup[#, "Keywords", {}], q] ||
		MemberQ[Flatten[Lookup[#, sp, {}]], q]||
		(MemberQ[TextWords[ToLowerCase[#Name]], ToLowerCase[q]])|| 
		(MemberQ[TextWords[ToLowerCase[Lookup[#, "Description",""]]], ToLowerCase[q]])  &]]
]

resourceSearchContentObjects[q_, count_,HoldPattern[ds_Dataset],index:HoldPattern[_SearchIndexObject]]:=Block[{contobjs, ids, info},
	contobjs=normalSearchResults[TextSearch[index, q]];
    ids=getSearchResultIDs[contobjs];
    If[Length[ids]>0,
    	info=SortBy[Normal[Select[ds,MemberQ[ids,#UUID]&]],(#UUID/.Thread[ids->Range[Length[ids]]])&];
    	If[ListQ[info],
			DeleteCases[Quiet[Normal[Catch[autoloadResource[DeleteMissing[#]]]]&/@Take[info, UpTo[count]]],
				$Failed]
				,
				{}
    	]
    	,
    	{}
    ]
]

resourceSearchContentObjects[q_, count_,HoldPattern[ds_Dataset],None]:=With[{index=CreateSearchIndex[prepareContentObjects[ds]]},
	If[Head[index]==SearchIndexObject,
		$cloudResourceSearchIndex=index;
		resourceSearchContentObjects[q, count,ds,$cloudResourceSearchIndex]
		,
		{}
	]
]

resourceSearchContentObjects[___]:={}

prepareContentObjects[ds_]:=With[{rows= Replace[
   Append[#, {"Title" -> #Name, "FileName" -> #UUID}] & /@ 
    Normal[KeyUnion[
      KeyTake[ds, 
       Prepend[ResourceSystemClient`Private`$textSearchIndexKeys, 
        "UUID"]], ""]], _Missing | None -> "", {0, Infinity}]},
  	ContentObject /@ rows     
]

$cloudResourceIndex=None;
$cloudResourceSearchIndex=None;

updateCloudResourceIndex[]:=Block[{info=cloudResourceSearchInfo[]},
	If[MatchQ[info,{_Association...}],
		Dataset[KeyUnion[info,defaultsearchvalue]]
		,
		None
	]
]

defaultsearchvalue["Keywords"]:={}
defaultsearchvalue["Name"|"Description"]:=""
defaultsearchvalue[prop_]:={}/;MemberQ[allSortingProperties[],prop]
defaultsearchvalue[_]:=Missing[]
	
addToCloudResourceIndex[info_]:=addToCloudResourceIndex[info,$cloudResourceIndex]

addToCloudResourceIndex[info_,None]:=With[{index=addToCloudResourceIndex[]},
	If[Head[index]===Dataset,
		addToCloudResourceIndex[info,index]
		,
		$Failed
	]
]

addToCloudResourceIndex[info_Association,HoldPattern[index_Dataset]]:=With[{id=info["UUID"]},
	If[StringQ[id],
		If[MemberQ[index[All,"UUID"],id],
			dropFromCloudResourceIndex[id]
		];
		addtoCloudResourceIndex[info, index]
		,
		$Failed
	]
]

addtoCloudResourceIndex[info_, index_]:=Block[{keys, first, newindex, indexinfo},
	If[Length[index]>0,
		first=Normal[First[index]];
		indexinfo=Last[KeyUnion[{first,KeyTake[info, Keys[first]]},defaultsearchvalue]];
		newindex=Check[Append[$cloudResourceIndex,indexinfo],$Failed];
		If[newindex=!=$Failed,
			$cloudResourceSearchIndex=None;
			$cloudResourceIndex=newindex
			,
			$Failed
		]
		,
		$cloudResourceIndex=Dataset[{info}]
	]
]

addToCloudResourceIndex[___]:=$Failed

dropFromCloudResourceIndex[id_]:=dropFromCloudResourceIndex[id,$cloudResourceIndex]

dropFromCloudResourceIndex[id_String,HoldPattern[index_Dataset]]:=Block[{newindex},
	newindex=Check[DeleteCases[index, _?(StringMatchQ[#UUID, id] &)],$Failed];
	If[newindex=!=$Failed,
		$cloudResourceSearchIndex=None;
		$cloudResourceIndex=newindex
		,
		$Failed
	]
]

dropFromCloudResourceIndex[___]:=Null

cloudResourceSearchInfo[]:=Quiet[importCloudResourceInfo[]]

resourcesearchLocal[q_, n_, method_Association]:=resourcesearchLocal[q, n, Lookup[method,"Local",Automatic]]

resourcesearchLocal[q_String, n_, "BruteForce"]:=Block[{ids, count=n/.Automatic->Infinity},
	ids=Select[$localResources,resourceKeywordMatchQ[q,#]&,UpTo[count]];
	getResourceInfo/@ids
]

resourcesearchLocal[_, _, "BruteForce"]:={}

resourcesearchLocal[q_, n_, "TextSearch"]:=Block[{ids},
	ids=textSearchResources[q];
	If[ListQ[ids],
		getResourceInfo/@ids
		,
		$Failed
	]
]

resourcesearchLocal[q_, n_, Automatic]:=Block[{res=None},
	If[$AllowResourceTextSearch,
		res=Quiet[resourcesearchLocal[q, n, "TextSearch"]]
	];
	If[ListQ[res],
		res
		,
		resourcesearchLocal[q, n, "BruteForce"]
	]
]
resourceKeywordMatchQ[str_,id_]:=Block[{info=Quiet[getResourceInfo[id,{"Name","Keywords"}]]},
	TrueQ[!FreeQ[info,str]]
]

resourcesearchLocalName[str_, n_, rtype_:All]:=Block[{ids, count=n/.Automatic->Infinity},
	ids=Select[$localResources,resourceNameMatchQ[rtype,str,#]&,UpTo[count]];
	ids
]

resourceNameMatchQ[rtype_,str_,id_]:=Block[{info=Quiet[
	getResourceInfo[id,{"Name","ResourceType", "RepositoryLocation"}]]},
	If[marketplacebasedResourceQ[info],
		AppendTo[localResourceNameMap,info["Name"]->id];
		persistResourceName[id,info["Name"], {"KernelSession"}];
		str===info["Name"]&&(rtype===All||rtype===info["ResourceType"])
		,
		False
	]
]

$resourceSystemSearchMethods={"Automatic",Automatic,"TextSearch","Table"};


resourcesearchResourceSystem[q_, n_, rtypes_,method_, rsbase_]:=Block[
	{res, opts={}, rtypestr=searchRTypeString[rtypes], resourcebase=resourceSystemBase[rsbase],str},
	If[AssociationQ[method],
		If[KeyExistsQ[method,"ResourceSystem"],
			If[MemberQ[$resourceSystemSearchMethods,method["ResourceSystem"]],
				opts={"SearchMethod"->method["ResourceSystem"]}
			]
		]
	];
	str=If[!StringQ[q],
		AppendTo[opts,"QueryFormatting"->"InputForm"];
		ToString[q,InputForm],
		q
	];
	res=apifun["SearchResources",Flatten[{"Query"->str,"Count"->n,"ResourceTypes"->rtypestr,opts}],System`ResourceSearch,resourcebase];
	If[KeyExistsQ[res,"Resources"],
		res=standardizeResourceInfo/@Select[Lookup[res,"Resources",{}],KeyExistsQ["UUID"]];
		res=fillResourceMetadata[#, Association["RepositoryLocation"->URL[resourcebase]]]&/@res;
		cacheresourceInfo[res, False];
		res
		,
		$Failed
	]
]

searchRTypeString[All|"All"|"all"]:="All"
searchRTypeString[l_List]:=StringRiffle[getresourceType/@l,","]
searchRTypeString[expr_]:=searchRTypeString[{expr}]

createResourceObjects[l_List]:=System`ResourceObject/@Select[l,AssociationQ]
createResourceObjects[expr_]:=expr

$ResourceSearchKeys={"Name","ResourceType","Description","ResourceObject","Description","RepositoryLocation","DocumentationLink"}

createResourceDataset[l_List]:=Dataset[KeyUnion[KeyTake[appendResourceSearch/@Select[l,Quiet[KeyExistsQ["UUID"][#]] &],$ResourceSearchKeys]]]
createResourceDataset[expr_]:=expr

appendResourceSearch[as_]:=Association[as,"ResourceObject"->ResourceObject[as["UUID"]],"DocumentationLink"->resourceURL[as]]

ResourceSystemClient`RandomResources[args___]:=Catch[randomResources[args]]

Options[ResourceSystemClient`RandomResources]={System`ResourceSystemBase:>System`$ResourceSystemBase};

randomResources[opts:OptionsPattern[ResourceSystemClient`RandomResources]]:=randomResources[1,opts]

randomResources[n_Integer,opts:OptionsPattern[ResourceSystemClient`RandomResources]]:=Block[{res, resourcebase},
	resourcebase=resourceSystemBase[OptionValue[ResourceSystemClient`RandomResources,{opts},System`ResourceSystemBase]];
    res=apifun["SearchResources",{"Query"->"","Count"->n,"Method"->"RandomSample"},System`ResourceSearch,resourcebase];
    If[KeyExistsQ[res,"Resources"],
        res=Lookup[res,"Resources",{}];
		res=fillResourceMetadata[#, Association["RepositoryLocation"->URL[$resourceSystemRequestBase]]]&/@res;
		cacheresourceInfo[res, False];
        System`ResourceObject[usableResourceInfo[#]]&/@res
        ,
        $Failed
    ]
]

randomResources[___]:=$Failed


validResourceQueryQ[l_List]:=AllTrue[l,validResourceQueryQ]
validResourceQueryQ[alt_Alternatives]:=AllTrue[alt,validResourceQueryQ]
validResourceQueryQ[_String]:=True
validResourceQueryQ[HoldPattern[SearchQueryString][_String]]:=True
validResourceQueryQ[HoldPattern[Rule][field_String,_?validResourceQueryQ]]:=MemberQ[$textSearchIndexKeys,field]
validResourceQueryQ[HoldPattern[FixedOrder|Except|SearchAdjustment][__]]:=True
validResourceQueryQ[HoldPattern[Less|NearestTo|Greater|Equal|EqualTo|
	LessThan|GreaterThan|Between|LessEqualThan|GreaterEqualThan|ContainsOnly][__]]:=True

parseResourceQuery[q_]:={All,q}/;FreeQ[q,"ResourceType"]
parseResourceQuery[q_]:=With[{types=Cases[q,HoldPattern[Rule]["ResourceType",t_]:>t,{0, Infinity}]},
	{Flatten[{types}],Replace[q,HoldPattern[Rule]["ResourceType",_]:>Nothing,{0,Infinity}]}
]

End[] (* End Private Context *)

EndPackage[]

SetAttributes[{ResourceSearch},
   {ReadProtected, Protected}
];