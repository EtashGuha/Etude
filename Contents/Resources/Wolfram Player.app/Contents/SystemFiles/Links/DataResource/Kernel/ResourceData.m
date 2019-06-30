(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {"System`ResourceData"}

BeginPackage["DataResource`"]
(* Exported symbols added here with SymbolName::usage *)  

System`ResourceData
DataResource`resourceTypeData
DataResource`resourceDataElement

Begin["`Private`"] (* Begin Private Context *) 
System`ResourceData[args___]:=Catch[resourceData[args]]

Options[ResourceData]=Options[resourceData]=Options[resourcedata]=Options[resourcedataName]={System`ResourceSystemBase:>$ResourceSystemBase};

resourceData[ro_System`ResourceObject,rest___]:=resourcedata[resourceObjectID[ro],rest]

resourceData[id_String,rest___]:=resourcedata[id,rest]/;uuidQ[id]

resourceData[name_String,rest___]:=Block[{$DataResourceNameLookup=name},
	resourcedata[Lookup[localResourceNameMap,name,$Failed],rest]/;KeyExistsQ[localResourceNameMap,name]
]

resourceData[name_String,rest___]:=resourcedataName[name,rest]

resourceData[___]:=$Failed

$DataResourceNameLookup=None;

resourcedataName[name_,elem___,opts:OptionsPattern[]]:=Block[{res,$DataResourceNameLookup=name},
	res=ResourceSystemClient`Private`findResourceObject[All, name, opts];
	If[resourceObjectQ[res],
		resourceData[res,elem,opts]
	]
]

resourcedata[id_String,rest___]:=resourcedatawithProgress[{id, ResourceSystemClient`Private`getResourceInfo[id]},rest]/;localResourceQ[id]

resourcedata[id_String,rest___]:=resourcedatawithProgress[{id, resourceInfo[id]},rest]/;MemberQ[$loadedResources,id]

resourcedata[id_String,rest___]:=With[{info=ResourceSystemClient`Private`loadResource[id]},
	If[AssociationQ[info],
		resourcedatawithProgress[{id, info},rest]
	]	
]

resourcedata[info_Association,rest___]:=resourcedatawithProgress[{info["UUID"],info},rest]

$resourceProgressIndicator=True;

resourcedatawithProgress[{id_String,info_Association}, rest___]:=Block[{$resourceProgressIndicator=False, temp,
	ResourceSystemClient`$progressID=Unique[]},
	ResourceSystemClient`Private`$ProgressIndicatorContent=ResourceSystemClient`Private`progressPanel["Importing the resource content\[Ellipsis]"];
	With[{res=resourcedata[{id, info},rest]},
		clearTempPrint[ResourceSystemClient`$progressID];
		res
	]
]/;$resourceProgressIndicator&&showResourceDataIndeterminateBar[id,info, rest]

showResourceDataIndeterminateBar[id_,_,elem_]:=False/;TrueQ[contentCached[id,elem]]
showResourceDataIndeterminateBar[___]:=True;

resourcedatawithProgress[{id_String,info_Association}, rest___]:=resourcedata[{id, info},rest]

resourcedata[{id_String,info_Association},opts:OptionsPattern[]]:=resourcedata[{id,info},Automatic, opts]

resourcedata[{id_,info_},elem_,opts:OptionsPattern[]]:=resourcedata[{id,info},elem,
	System`ResourceSystemBase->System`$ResourceSystemBase,opts]/;FreeQ[{opts},System`ResourceSystemBase]

resourcedata[{id_,info_},elem_,opts:OptionsPattern[]]:=
	resourceData[
		ResourceObject[If[StringQ[$DataResourceNameLookup],	$DataResourceNameLookup,info["UUID"]] ,opts],
		elem,opts
		]/;ResourceSystemClient`Private`optionMismatchQ[info,Association[opts]]

resourcedata[{id_,info_},elem_,opts:OptionsPattern[]]:=
	DataResource`resourceTypeData[Lookup[info,"ResourceType"],{id, info},elem,opts]/;!MatchQ[Lookup[info,"ResourceType"],$DataResourceType]

resourcedata[{id_,info_},elem_, rest___]:=resourcedataelement[{id, info},elem,rest]

Options[DataResource`resourceTypeData]=Options[DataResource`resourceDataElement]=Options[ResourceData];

$drLoadResourceType=True;

DataResource`resourceTypeData[rtype_String,{id_, info_},rest__]:=Block[{$drLoadResourceType=False},
	ResourceSystemClient`Private`loadResourceType[rtype];
	DataResource`resourceTypeData[rtype,{id,info},rest]
]/;!MatchQ[rtype,$DataResourceType]&&TrueQ[$drLoadResourceType]

DataResource`resourceTypeData[_,{_, _},__]:=(Message[ResourceData::rtype];$Failed)

DataResource`resourceDataElement[args___]:=resourcedataelement[args]

resourcedataelement[{id_String,info_Association},elem_String, ___]:=info["InformationElements",elem]/;KeyExistsQ[Lookup[info,"InformationElements",Association[]],elem]

resourcedataelement[{id_String,info_Association},Automatic,rest___]:=Block[{},
	If[KeyExistsQ[info,"DefaultContentElement"],
		resourcedata[{id,info},info["DefaultContentElement"],rest]
		,
		resourcedata[{id,info},All,rest]
	]
]/;multipartResourceQ[info]

resourcedataelement[{id_String,info_Association}, elems0_,rest___]:=Block[{allelems=Lookup[info,"ContentElements",{}], elems},
	If[AssociationQ[allelems],allelems=Keys[allelems]];
	If[!ListQ[allelems],allelems={}];
	elems=If[elems0===All,
		allelems
		,
		If[Complement[elems0, allelems]=!={},
			Message[ResourceData::invelem];Throw[$Failed]
		];
		elems0
	];
	AssociationMap[resourcedata[{id,info}, #,rest]&,elems]
]/;ListQ[elems0]||elems0===All

resourcedataelement[{id_String,info_Association},rest___]:=resourcedataUncached[{id, info}, rest]/;userdefinedResourceQ[info]&&!MemberQ[$localResources,id]

resourcedataelement[{id_String,info_Association},elem_,rest___]:=If[
	localStoredQ[info,elem]||!marketplacebasedResourceQ[info],
	If[localStoredQ[info,elem],
		resourcedataLocalCheck[{id,info},elem,rest],
		If[cloudStoredQ[info,elem],
			printTempOnce[ResourceSystemClient`$progressID];
			resourcedataCloud["DataResource",info,elem],
			$Failed
		]
	]
	,
	printTempOnce[ResourceSystemClient`$progressID];
	resourcedataResourceSystem[{id, info},elem,rest]	
]/;contentElementQ[info,elem]

$redownloadingElement=False;

resourcedataLocalCheck[{id_,info_},elem_,rest___]:=With[{res=Quiet[resourcedataLocal[info, elem],LocalObject::nso]},
	If[FailureQ[res],
		If[TrueQ[$redownloadingElement],Throw[$Failed]];
		ResourceSystemClient`Private`deleteElementContentFile[info,elem];
		ResourceSystemClient`Private`storeDownloadVersion[id,Association["Version"->None],
			Association["Element"->elem,"Location"->Missing["Deleted"],"Format"->Missing["Deleted"]]];
		Block[{$redownloadingElement=True},
			printTempOnce[ResourceSystemClient`$progressID];
			resourcedataelement[{id,resourceInfo[id]},elem,rest]
		]
		,
		res
	]
]

resourcedataelement[_,elem_,___]:=(Message[ResourceData::invelem,elem];Throw[$Failed])
resourcedataelement[___]:=Throw[$Failed]
resourcedata[___]:=Throw[$Failed]

End[] (* End Private Context *)

EndPackage[]

SetAttributes[{ResourceData},
   {ReadProtected, Protected}
];