
(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["DataResource`"]

DataResource`$ResourceObjectEntityStores=If[AssociationQ[$resourceObjectEntityStores],
	$resourceObjectEntityStores,
	getPersistentResourceObjectEntityStores[]
]

$EntityStorePersistentObjectPrefix="EntityStore_";

Begin["`Private`"] (* Begin Private Context *) 

ResourceSystemClient`Private`registerResourceNameExtensions[$DataResourceType, rest___]:=registerDataResourceExtension[rest]

registerDataResourceExtension[id_, info_, locations_,as_]:=
	registerdataResourceExtension[info["ContentElementAccessType"],id, info, as]/;KeyExistsQ[info,"ContentElementAccessType"]&&!FreeQ[locations,"KernelSession"]
registerDataResourceExtension[___]:=Null

registerdataResourceExtension["EntityStore",id_, info_, as_]:=With[{es=ResourceData[id]},
	If[Head[es]===EntityStore,
		If[AssociationQ[$resourceObjectEntityStores],
			AppendTo[$resourceObjectEntityStores,id->es],
			getPersistentResourceObjectEntityStores[];
			If[AssociationQ[$resourceObjectEntityStores],
				AppendTo[$resourceObjectEntityStores,id->es],
				$resourceObjectEntityStores=Association[id->res]
			]
		]	
	]
]
registerdataResourceExtension[___]:=Null


ResourceSystemClient`Private`unregisterResourceNameExtensions[$DataResourceType, rest___]:=unregisterDataResourceExtension[rest]

unregisterDataResourceExtension[id_, info_, locations_]:=
	unregisterdataResourceExtension[info["ContentElementAccessType"],id, info]/;KeyExistsQ[info,"ContentElementAccessType"]&&!FreeQ[locations,"KernelSession"]

unregisterdataResourceExtension["EntityStore",id_, info_]:=If[AssociationQ[$resourceObjectEntityStores],
	KeyDropFrom[$resourceObjectEntityStores,id]
]

unregisterdataResourceExtension[___]:=Null


registerEntityStore[es_EntityStore,locations_,opts___]:=With[{ro=makeEntityStoreResource[es]},
	If[resourceObjectQ[ro],
		ResourceSystemClient`Private`resourceRegister[ro, locations,opts]
	]	
]

makeEntityStoreResource[es_EntityStore]:=ResourceObject[
	Association[
		"Name"->$EntityStorePersistentObjectPrefix<>CreateUUID[],
		"ResourceType"->"DataResource",
		"DefaultContentElement"->"EntityStore",
		"ContentElements"->Association["EntityStore"->es],
		"ContentElementAccessType"->"EntityStore"
	]
]

getPersistentResourceObjectEntityStores[]:=getPersistentResourceObjectEntityStores[$PersistencePath]
getPersistentResourceObjectEntityStores[locations_]:=Block[{
	ids=(#["Value"] & /@
   		PersistentObjects[
   			ResourceSystemClient`Private`$ResourceObjectPersistentValuePrefix<>
   			$EntityStorePersistentObjectPrefix<>"*",locations]), res},
   	res=Association[Cases[(#->Quiet[ResourceData[#]])&/@ids,HoldPattern[Rule][_String,_EntityStore],{1}]];
   	If[AssociationQ[res],
   		$resourceObjectEntityStores=res
   	]
]

End[] (* End Private Context *)

EndPackage[]