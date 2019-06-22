(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["NeuralNetResource`"]

Begin["`Private`"] (* Begin Private Context *) 

$NeuralNetResourceType="NeuralNet";

ResourceSystemClient`Private`usableresourceinfoKeys[$NeuralNetResourceType]={"ContentSize","ContentElements"};

ResourceSystemClient`Private`repositorystandardizeResourceInfo[$NeuralNetResourceType,info_Association]:=mapAt[
		Join[KeyDrop[info,"ElementInformation"],Lookup[info,"ElementInformation",Association[]]],	
			{
			"ContentSize"->bytecountQuantity,
			"ContentElements"->(DeleteCases[#,Automatic]&)			
			}]

ResourceSystemClient`Private`resourceInfoOrder[$NeuralNetResourceType]:={
	"Name","UUID",
	"Content","ContentElementLocations",
	"RepositoryLocation","ResourceLocations",
	"ResourceType","ContentSize","ContentElements",
	"Version","Description", "ContentElementHashes",
	"ByteCount", "TrainingSetData",
	"TrainingSetInformation", "Performance",
	"InputDomains","TaskType",
	"Keywords",
	"MyAccount","Attributes",
	"LatestUpdate","DownloadedVersion",
	"Format", "Caching"
}

ResourceSystemClient`Private`repositoryCacheResourceInfo[$NeuralNetResourceType,id_, info_, dir_]:=
	cacheResourceInfoWithElementFunctions["NeuralNet",id, info,dir]

ResourceSystemClient`Private`addToResourceTypes["NeuralNet"];

ResourceSystemClient`Private`sortingProperties["NeuralNet"]={"InputDomains","TaskType"}

ResourceSystemClient`Private`clearContentCache["NeuralNet",id_,info_]:=DataResource`Private`clearInMemoryElementCache["NeuralNet",id, info]

ResourceSystemClient`Private`autoUpdateResourceQ["NeuralNet",_]:=True
End[] (* End Private Context *)

EndPackage[]



SetAttributes[{},
   {ReadProtected, Protected}
];