(* Wolfram Language Package *)

BeginPackage["ResourceSystemClient`"]

Begin["`Private`"] (* Begin Private Context *) 

validateParameter[rtype_,HoldPattern[Rule|RuleDelayed][key_,value_]]:=Rule[key,validateParameter[rtype,key,value]]

validateParameter[_,key_,value_]:=With[{res=Interpreter[$ResourceMetadataSchema[key]][value]},
	If[res===$Failed||!FreeQ[res,Failure],
		(Message[System`ResourceSubmit::invparam, key];Throw[$Failed])
		,
		res		
	]
]/;KeyExistsQ[$ResourceMetadataSchema,key]


validateParameter[rtype_,key_,value_]:=With[{res=Interpreter[respositoryMetadataSchema[rtype][key]][value]},
	If[res===$Failed||Head[res]===Failure,
		(Message[System`ResourceSubmit::invparam, key];Throw[$Failed])
		,
		res		
	]
]/;KeyExistsQ[respositoryMetadataSchema[rtype],key]

resourceMaxNameLength[_]:=160;

validateParameter[rtype_,"Name",value_]:=With[{
	res=Interpreter[Restricted["String", lengthregex[$ResourceNameLengthMinimum,resourceMaxNameLength[rtype]]]][value]},
	If[res===$Failed||!FreeQ[res,Failure],
		(Message[System`ResourceSubmit::invparam, "Name"];Throw[$Failed])
		,
		res		
	]
]

validateParameter[_,"ExternalLinks",expr_]:=interpretAlternativesURL["ExternalLinks",Restricted["Expression", {Hyperlink, ButtonNote, Rule}] | "URL",expr]

validateParameter[_,"SourceMetadata",expr_]:=validateSourceMetadata[expr]

validateParameter[_,"ContributorInformation",expr_]:=validateContributor[expr]

validateParameter[_,"ExampleNotebook",expr_]:=validateexamplenotebook[expr]

validateParameter[_,"Asynchronous", expr_]:=expr

validateParameter[_,"PricingInformation",0]:=0
validateParameter[_,"PricingInformation",as_Association]:=validatePricingInfo[as]
validateParameter[_,"PricingInformation",rule_Rule]:=validatePricingInfo[Association[rule]]

validateParameter[_, "DefinitionNotebook", nb: KeyValuePattern[ "Data" -> _ByteArray? ByteArrayQ ] ] := nb;

validatePricingInfo[as_]:=as/;KeyExistsQ[as,"MarketplaceBilling"]
validatePricingInfo[as_]:=Association[
	"MarketplaceBilling"->"OneTimePurchase",
	as
]/;KeyExistsQ[as,"MarketplaceTier"]

validatePricingInfo[as_]:=Association[
	"MarketplaceBilling"->"PerUse",
	as
]/;KeyExistsQ[as,"BaseUsagePrice"]

validatePricingInfo[_]:=(Message[System`ResourceSubmit::invparam, "PricingInformation"];Throw[$Failed])

validateParameter[_,key_,___]:=(Message[System`ResourceSubmit::invparam, key];Throw[$Failed])
validateParameter[___]:=(Message[System`ResourceSubmit::invparam2];Throw[$Failed])


(********)

requiredparameters[expr_,___]:=Association["Name"->"String","UUID"->"String","ResourceType"->"String"]

resourceInfoOrder[_]:={
	"Name","UUID","ResourceType","Description",
	"RepositoryLocation","ResourceLocations",
	"ContentSize",
	"Version",
	"Keywords",
	"LatestUpdate"
};

basicInfoOrderFunc[rtype_]:=With[{order=resourceInfoOrder[rtype]},
	(# /. Thread[order -> Range[Length[order]]] &)];
	
sortBasicInfo[as_]:=KeySortBy[as, basicInfoOrderFunc[getResourceType[as]]]


$ResourceNameLengthMinimum=3;
$ResourceNameLengthLimit=160;
lengthregex[min_, max_] := 
  RegularExpression[StringJoin[".{", ToString[min], ",", ToString[max], "}"]]
$ShortMetadataLengthLimit=1000;
$MediumMetadataLengthLimit=10000;
$LongMetadataLengthLimit=100000;
$KeywordLengthLimit=40;

$ResourceMetadataSchema:=Association[
	"ShortName"->(With[{raw = #},Interpreter[Restricted["String", 
     	lengthregex[$ResourceNameLengthMinimum, $ResourceNameLengthLimit]], (URLEncode[URLDecode[#]] === raw) &]][#] &),
	"AlternativeNames"->RepeatingElement[Restricted["String", lengthregex[$ResourceNameLengthMinimum,$ResourceNameLengthLimit]]],
	"UUID"->Restricted["String", RegularExpression["\\w{8}-\\w{4}-\\w{4}-\\w{4}-\\w{12}$"]],
	"ResourceType"->$availableResourceTypes,
	"DisplayedSourceFields"->AnySubset[{"Contributor","Coverage","Creator","Author",
		"GeographicCoverage","TemporalCoverage","Citation",
		"Date","Description","Language","Publisher","Rights","Source","Title"}],
	"Description"->Restricted["String", Automatic, $ShortMetadataLengthLimit],
	"Details"->Restricted["String", "*",$LongMetadataLengthLimit],
	"Originator"->Restricted["String", "*",$ShortMetadataLengthLimit],
	"Keywords"->RepeatingElement[Restricted["String", lengthregex[0,$KeywordLengthLimit]]],
	"SeeAlso"->RepeatingElement[Restricted["Expression", {System`ResourceObject},Automatic,None] | Restricted["String", 
     	lengthregex[$ResourceNameLengthMinimum, $ResourceNameLengthLimit]]],
	"RelatedSymbols"->AnySubset[Names["System`*"]],
	"Discoverable"->"Boolean",
	"WolframLanguageVersionRequired"->Restricted["String", lengthregex[0,$KeywordLengthLimit]]
]

respositoryMetadataSchema[_]:=Association[]

validateSourceMetadata[as_Association]:=DeleteMissing[Association@KeyValueMap[validatesourceMetadata,as]]

validatesourceMetadata[key_,val_]:=With[{res=validatesourcemetadata[key, val]},
	If[res===$Failed||!FreeQ[res,Failure],
		(Message[System`ResourceSubmit::invparam, key];Throw[$Failed])
		,
		key->res		
	]
]

$sourcemetadataLengthLimit=1000;

validatesourcemetadata["Description",_List]:=$Failed
validatesourcemetadata[key_,l_List]:=Throw[$Failed]/;Length[l]>$sourcemetadataLengthLimit

$sourceinterpretertypes:=$sourceinterpretertypes={
	Restricted["String",Automatic,$ShortMetadataLengthLimit],"Person","Periodical","Company","University","School","Financial","Country","HistoricalCountry","City",
	"USState","USCounty","AdministrativeDivision","AstronomicalObservatory","Airport","Airline","WeatherStation",
	"Entity"
}

$sourceLinkTypes:=$sourceLinkTypes={Restricted["String","*",$ShortMetadataLengthLimit],"URL",Restricted["Expression", {Hyperlink, ButtonNote, Rule}]}

validatesourcemetadata["Contributor",expr_]:=interpretAlternatives["Contributor",$sourceinterpretertypes, expr]
validatesourcemetadata["GeographicCoverage",expr_]:=interpretAlternatives["GeographicCoverage",$sourceinterpretertypes, expr]
validatesourcemetadata["TemporalCoverage",expr_]:=interpretAlternatives["TemporalCoverage",$sourceinterpretertypes, expr]
validatesourcemetadata["Creator",expr_]:=interpretAlternatives["Creator",$sourceinterpretertypes, expr]
validatesourcemetadata["Author",expr_]:=interpretAlternatives["Author",$sourceinterpretertypes, expr]
validatesourcemetadata["Date",l_List]:=With[{dates=validatesourcemetadataDate/@l},
	If[FreeQ[dates,$Failed],dates,$Failed]
]
validatesourcemetadata["Date",expr_]:=validatesourcemetadataDate[expr]
validatesourcemetadataDate[date_DateObject]:=date
validatesourcemetadataDate[expr_]:=With[{do=DateObject[expr]},
	If[DateObjectQ[do],	do,	$Failed]]
	
validatesourcemetadata["Description",expr_]:=Interpreter[Restricted["String","*",$MediumMetadataLengthLimit]][expr]
validatesourcemetadata["Citation",expr_]:=Interpreter[Restricted["String","*",$MediumMetadataLengthLimit]][expr]
validatesourcemetadata["Language",ent:HoldPattern[Entity]["Language", "English"|"French"|"German"|"Spanish"]]:=ent
validatesourcemetadata["Language",expr_]:=interpretAlternatives["Language",
	{Restricted["String","*",$ShortMetadataLengthLimit],"Language"}, expr]
validatesourcemetadata["Publisher",expr_]:=interpretAlternatives["Publisher",$sourceinterpretertypes, expr]
validatesourcemetadata["Rights",expr_]:=interpretAlternativesURL["Rights",$sourceLinkTypes, expr]
validatesourcemetadata["Source",expr_]:=interpretAlternativesURL["Source",$sourceLinkTypes, expr]
validatesourcemetadata["Title",expr_]:=Interpreter[Restricted["String",Automatic,$ShortMetadataLengthLimit]][expr]

validatesourcemetadata[___]:=$Failed

validateSourceMetadata[___]:=$Failed

validateContributor[as_Association]:=DeleteMissing[Association@KeyValueMap[validatecontributor,as]]

validatecontributor[key_,val_]:=With[{res=validatecontributor0[key, val]},
	If[res===$Failed||!FreeQ[res,Failure],
		(Message[System`ResourceSubmit::invparam, key];Throw[$Failed])
		,
		key->res		
	]
]

validatecontributor0["ContributedBy",str_String]:=str/;StringLength[str]<=$ShortMetadataLengthLimit

validatecontributor0[___]:=$Failed

validateContributor[___]:=$Failed

interpretAlternativesURL[key_,types_,expr_]:=interpretAlternatives[key, types, expr]/;FreeQ[expr,URL]
interpretAlternativesURL[key_,types_,expr_]:=interpretAlternatives[key, types, Replace[expr,URL->Hyperlink,Infinity, Heads->True]]

interpretAlternatives[key_,alts_, expr_List]:=Interpreter[Alternatives@@alts][expr]
interpretAlternatives[key_,alts_, expr_]:=Catch[Last[interpretalternatives[#,expr]&/@alts]]

interpretalternatives[interp_,expr_]:=With[{res=Interpreter[interp][expr]},
	If[res===$Failed||!FreeQ[res,Failure],
		res
		,
		Throw[res]		
	]
]

$resourceShingleProperty="DocumentationLink"
resourceURL[info_Association]:=Lookup[info, $resourceShingleProperty, resourceurl[info]]

resourceurl[info_Association]:=With[{rtype=getResourceType[info]},
	loadResourceType[rtype];
	repositoryResourceURL[rtype, info]
]

repositoryResourceURL[rtype_,info_]:=If[marketplacebasedResourceQ[info],
	Block[{
		name=Lookup[info,"ShortName", Lookup[info,"UUID"]], 
		base=Lookup[info,"RepositoryLocation"]},
		base=repositoryDomain[rtype,base];
		If[StringQ[base]&&StringQ[name],
			repositoryresourceURL[rtype,base,name]
			,
			None]
		]
	,
	None
	]


repositoryDomain[rtype_]:=repositoryDomain[rtype,$CloudBase]
repositoryDomain[rtype_,url_URL]:=repositoryDomain[rtype,First[url]]

repositoryDomain[rtype_,url_String]:=With[{rtypepath=repositoryPath[rtype]},
	If[StringQ[rtypepath],
		Switch[url,
			"https://www.wolframcloud.com/objects/resourcesystem/api/1.0",
			"https://resources.wolframcloud.com/"<>rtypepath,
			"https://www.test.wolframcloud.com/objects/resourcesystem/api/1.0",
			"https://resources.test.wolframcloud.com/"<>rtypepath,
			"https://www.devel.wolframcloud.com/objects/resourcesystem/api/1.0",
			"https://resources.devel.wolframcloud.com/"<>rtypepath,
			_,
			toRepositoryResourcePath[url,rtypepath]
		]
		,
		None
	]]/;StringContainsQ[url,"wolframcloud"]


toRepositoryResourcePath[apibase_, rtypepath_] := Block[{p = URLParse[apibase]},
  p["Path"] = torepositoryResourcePath[p["Path"], rtypepath];
  URLBuild[p]
  ]

torepositoryResourcePath[{before___, "objects", user_, ___}, rtypepath_] := 
	{before, "objects", user, "published", rtypepath}
	
toresourceWebPath[___]:=$Failed

repositoryDomain[___]:=None

repositoryResourceURL[__]:=None

repositoryresourceURL[_,base_,name_]:=URL[URLBuild[{base,"resources",name}]]

repositoryPath[rtype_String]:=(repositoryPath[rtype]=StringReplace[rtype,"Resource"->""]<>"Repository");
repositoryPath[_]:=""


repositoryResourceURL[___]:=None

updateResourceInfo[id_String,update_Association]:=Block[{infofile,info},
	infofile=resourceInfoFile[id];
	info=DeleteCases[Association[Get[infofile],update],Missing["Deleted"]];
	If[AssociationQ[info],
		info=updateRepositoryResourceInfo[getResourceType[info],id, info, update];
		Put[info,infofile];
		resourceInfo[id]=info,
		getResourceInfo[id]
	]
]

updateRepositoryResourceInfo[_,_, info_, __]:=info


End[] (* End Private Context *)

EndPackage[]
