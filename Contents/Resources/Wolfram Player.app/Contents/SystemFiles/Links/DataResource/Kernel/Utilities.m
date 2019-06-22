
BeginPackage["DataResource`"]

Begin["`Private`"]

ResourceSystemClient`Private`getresourceType["Data"]="DataResource";

$DataResourceType=("DataResource"|"Data"|"data");

ResourceSystemClient`Private`usableresourceinfoKeys[$DataResourceType]={"ContentSize","ContentElements"};

ResourceSystemClient`Private`repositorystandardizeResourceInfo[$DataResourceType,info_Association]:=mapAt[
		Join[KeyDrop[info,"ElementInformation"],
			Lookup[info,"ElementInformation",Association[]], 
			DeleteMissing[KeyTake[info,"ContentSize"]]],	
			{
			"InformationElements"->uncompressInformationElements,
			"ContentSize"->bytecountQuantity,
			"ContentElements"->(DeleteCases[#,Automatic]&)			
			}]

ResourceSystemClient`Private`resourceInfoOrder[$DataResourceType]:={
	"Name","UUID",
	"Content","ContentElementLocations",
	"RepositoryLocation","ResourceLocations",
	"ResourceType","ContentSize","ContentElements",
	"Version","Description",
	"ContentTypes",
	"Categories","Keywords",
	"MyAccount","Attributes",
	"LatestUpdate","DownloadedVersion",
	"Formats","DefaultReturnFormat","Caching"
}

ResourceSystemClient`Private`repositoryCacheResourceInfo[$DataResourceType,id_, info_, dir_]:=
	cacheResourceInfoWithElementFunctions["DataResource",id, info,dir]

ResourceSystemClient`Private`addToResourceTypes/@{"DataResource","Data","data"}

getContentElementAccessType[as_]:=Lookup[as,"ContentElementAccessType",Lookup[as,"ContentType"]]

ResourceSystemClient`Private`sortingProperties["DataResource"]={"Categories","ContentTypes"}

datarepositorydomain[]:=datarepositorydomain[$CloudBase]
datarepositorydomain[url_URL]:=datarepositorydomain[First[url]]
datarepositorydomain[url_String]:="https://datarepository."<>Which[
	StringContainsQ[url, "test"],"test.",
	StringContainsQ[url, "devel"],"devel.",
	True,""]<>"wolframcloud.com"/;StringContainsQ[url,"wolframcloud"]

datarepositorydomain[___]:=None


ResourceSystemClient`Private`clearContentCache[$DataResourceType,id_,info_]:=clearInMemoryElementCache["DataResource",id, info]

End[] (* End Private Context *)

EndPackage[]
