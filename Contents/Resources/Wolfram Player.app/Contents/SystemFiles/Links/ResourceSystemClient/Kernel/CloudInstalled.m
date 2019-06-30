(* Wolfram Language Package *)

BeginPackage["ResourceSystemClient`"]

Begin["`Private`"] (* Begin Private Context *) 

useCloudInstalledResourcesQ[]:=defaultRSBaseQ[$CloudBase,System`$ResourceSystemBase]/;$CloudEvaluation
useCloudInstalledResourcesQ[rsbase_]=defaultRSBaseQ[$CloudBase,rsbase]/;$CloudEvaluation

useCloudInstalledResourcesQ[___]=False

defaultRSBaseQ[cbase_String,rsbase_String]:=StringMatchQ[rsbase,URLBuild[{StringReplace[cbase,"/"~~EndOfString->""], 
	"objects","resourcesystem","api"}]<>"*"]
defaultRSBaseQ[___]:=False

loadResource[id_String]:=With[{info=loadCloudInstalledResource[id]},
	If[AssociationQ[info],
        loadresource[id, info];
		info
		,
		$cloudInstalledResources=DeleteCases[$cloudInstalledResources,id];
		loadResource[id]
	]]/;MemberQ[$cloudInstalledResources,id]

loadCloudInstalledResource[id_]:=loadCloudInstalledResource[id,cloudInstalledResourceDirectory[id]]

loadCloudInstalledResource[id_,dir_]:=Block[{lo=localObject[AbsoluteFileName@FileNameJoin[{dir,"metadata"}]], info},
	If[fileExistsQ[lo],
		info=Quiet[Get[lo]];
		If[AssociationQ[info],
			info=standardizeResourceInfo[info];
			resourceInfo[id]=info;
			info
			,
			$Failed
		]
		,
		$Failed
	]
]


$cloudInstalledResourcesDirectory="CloudInstalledResourceObjects/Resources";

cloudInstalledResourceIDs[]:=Block[{
	dirs=Select[Nest[
	Quiet[FileNames["*", Flatten[Select[#, DirectoryQ]],1], General::dirdep] &, 
		{$cloudInstalledResourcesDirectory}, 2],uuidQ[FileNameTake[#]]&]},
	addCloudInstalledNameToNameMap/@dirs;
	FileNameTake/@dirs
]

addCloudInstalledNameToNameMap[dir_String]:=Quiet[addCloudInstalledNameToNameMap[Get[LocalObject@FileNameJoin[{dir,"metadata"}]]]]

addCloudInstalledNameToNameMap[info_Association]:=addCloudInstalledNameToNameMap[info["UUID"],info["Name"]]

addCloudInstalledNameToNameMap[id_String,name_String]:=AppendTo[localResourceNameMap,name->id]

$cloudInstalledResources:=With[
	{ids=cloudInstalledResourceIDs[]},
	If[ListQ[ids],
		$cloudInstalledResources=ids,
		{}
	]
]/;useCloudInstalledResourcesQ[]

$cloudInstalledResources:={}

cloudInstalledResourceDirectory[id_]:=FileNameJoin[
	{$cloudInstalledResourcesDirectory,StringTake[id,3],id}]



End[] (* End Private Context *)

EndPackage[]