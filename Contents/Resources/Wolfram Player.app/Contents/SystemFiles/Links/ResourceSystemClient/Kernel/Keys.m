(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {
	ResourceSystemClient`ResourceKeys,
	ResourceSystemClient`CreateResourceKey,
	ResourceSystemClient`DeleteResourceKey,
	ResourceSystemClient`DeactivateResourceKey,
	ResourceSystemClient`ActivateResourceKey
	}

BeginPackage["ResourceSystemClient`"]

Begin["`Private`"] (* Begin Private Context *) 

ResourceSystemClient`ResourceKeys[args___]:=Catch[resourceKeys[args]]

resourceKeys[]:=resourceKeys["Dataset"]

resourceKeys["Keys"]:=With[{resp=apifun["GetResourceKeys",{}, ResourceObject]},
	If[AssociationQ[resp],
		Lookup[resp,"Keys",Lookup[resp,"ResourceKeys",Throw[$Failed]]],
		Throw[$Failed]
	]
]

resourceKeys["Dataset"]:=With[{resp=apifun["ResourceKeys",{"IncludeKeyInformation"->"True"}, ResourceObject]},
	If[AssociationQ[resp],
		Dataset[KeyTake[Association/@Lookup[resp,"Keys",Lookup[resp,"GetResourceKeys",Throw[$Failed]]],{"Key","Status"}]],
		Throw[$Failed]
	]
]

resourceKeys[__]:=$Failed


ResourceSystemClient`CreateResourceKey[args___]:=Catch[createResourceKey[args]]

createResourceKey[]:=With[{resp=apifun["ResourceKeyCreate",{}, ResourceObject]},
	If[AssociationQ[resp],
		Lookup[resp,"ResourceKey",Lookup[resp,"Key",Throw[$Failed]]],
		Throw[$Failed]
	]
]

createResourceKey[__]:=$Failed

ResourceSystemClient`DeleteResourceKey[args___]:=Catch[deleteResourceKey[args]]

deleteResourceKey[key_String]:=With[{resp=apifun["ResourceKeyDelete",{"ResourceKey"->key}, ResourceObject]},
	If[AssociationQ[resp],
		If[KeyExistsQ[resp,"DeletedResourceKey"],
			Success["ResourceKeyDeleted",<|"ResourceKey"->resp["DeletedResourceKey"]|>],
			Throw[$Failed]
		],
		Throw[$Failed]
	]
]

deleteResourceKey[HoldPattern[PermissionsKey][key_String]]:=deleteResourceKey[key]

deleteResourceKey[___]:=$Failed


ResourceSystemClient`DeactivateResourceKey[args___]:=Catch[deactivateResourceKey[args]]

deactivateResourceKey[key_String]:=With[{resp=apifun["ResourceKeyDeactivate",{"ResourceKey"->key}, ResourceObject]},
	If[AssociationQ[resp],
		If[KeyExistsQ[resp,"DeactivatedResourceKey"],
			Success["ResourceKeyDeactivated",<|"ResourceKey"->resp["DeactivatedResourceKey"]|>],
			Throw[$Failed]
		],
		Throw[$Failed]
	]
]

deactivateResourceKey[HoldPattern[PermissionsKey][key_String]]:=deactivateResourceKey[key]

deactivateResourceKey[___]:=$Failed


ResourceSystemClient`ActivateResourceKey[args___]:=Catch[activateResourceKey[args]]

activateResourceKey[key_String]:=With[{resp=apifun["ResourceKeyActivate",{"ResourceKey"->key}, ResourceObject]},
	If[AssociationQ[resp],
		If[KeyExistsQ[resp,"ActivatedResourceKey"],
			Success["ResourceKeyActivated",<|"ResourceKey"->resp["ActivatedResourceKey"]|>],
			Throw[$Failed]
		],
		Throw[$Failed]
	]
]

activateResourceKey[HoldPattern[PermissionsKey][key_String]]:=activateResourceKey[key]

activateResourceKey[___]:=$Failed

End[] (* End Private Context *)

EndPackage[]

SetAttributes[{
	ResourceSystemClient`ResourceKeys,
	ResourceSystemClient`CreateResourceKey,
	ResourceSystemClient`DeleteResourceKey,
	ResourceSystemClient`DeactivateResourceKey,
	ResourceSystemClient`ActivateResourceKey
	},
   {ReadProtected, Protected}
];