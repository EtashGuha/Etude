(* Wolfram Language Package *)

Unprotect[System`ResourceObject];

BeginPackage["ResourceSystemClient`"]
(* Exported symbols added here with SymbolName::usage *)  

Begin["`Private`"] (* Begin Private Context *) 

defineResourceUpValue[hfun_]:=System`ResourceObject/:hfun[ro_System`ResourceObject,after___]:=
		resourceAccess[hfun,ro,after]


defineResourceCurriedUpValue[fun_]:=
	System`ResourceObject/:HoldPattern[f_fun][ro_System`ResourceObject,after___]:=resourceAccess[f[#]&,ro,after];


$ResourceUpValueFunctions={HoldPattern[Get], HoldPattern[Values], HoldPattern[Dataset], 
 HoldPattern[Keys], HoldPattern[Select], HoldPattern[TimeSeries], 
 HoldPattern[EventSeries], HoldPattern[Options], 
 HoldPattern[SetOptions], HoldPattern[Normal], 
 HoldPattern[FormulaData]};


$ResourceUpValueCurriedFunctions={};
	
defineResourceUpValue/@$ResourceUpValueFunctions;
	
defineResourceCurriedUpValue/@$ResourceUpValueCurriedFunctions;

resourceAccess[held_, resource:rpat, rest___]:=resourceAccess[ReleaseHold[held],resource, rest]/;Head[held]===HoldPattern
resourceAccess[Normal,resource:rpat]:=usableResourceInfo[getResourceInfo[resourceObjectID[resource]]]
resourceAccess[Options,resource:rpat]:=sortBasicInfo[getResourceInfo[resourceObjectID[resource]]]
resourceAccess[SetOptions,resource:rpat, rest___]:=setResourceInfo[resourceObjectID[resource], rest]


resourceAccess[fun_,resource:rpat, rest___]:=Catch[With[{id=resourceObjectID[resource]},
	resourceAccess0[fun,id,resourceInfo[id], rest]
]]

resourceAccess0[fun_,id_,info_, rest___]:=With[{rtype=getResourceType[info]},
	If[!StringQ[rtype],Throw[$Failed]];
	loadResourceType[rtype];
	repositoryresourceaccess[rtype,fun,id, info,rest]
]

repositoryresourceaccess[___]:=$Failed

System`ResourceObject /: HoldPattern[CloudObject`CloudDeployActiveQ[_System`ResourceObject]] := True

System`ResourceObject /: HoldPattern[GenerateHTTPResponse[ro_System`ResourceObject, rest___]] := 
	GenerateHTTPResponse[HTTPResponse[createResourceShingle[ro],"ContentType" -> "text/html;charset=utf-8"], rest]

System`ResourceObject /: HoldPattern[CloudDeploy[ro_System`ResourceObject, rest___]] := Catch[cloudDeployResource[ro, rest]]

System`ResourceObject /: HoldPattern[LocalCache[ro_System`ResourceObject,args___]]:=Catch[resourceLocalCache[ro, args]]/;(Length[{args}]<=1)

System`ResourceObject /: HoldPattern[System`DeleteObject[ro_System`ResourceObject]]:=Catch[
	deleteResourceObject[ro]]


(* Hook up new v12 Information functionality for ResourceObject *)
System`ResourceObject /:
  Information`GetInformation[ ro_System`ResourceObject ] :=
    Append[ First @ ro, "ObjectType" -> "ResourceObject" ];


System`ResourceObject /:
  Information`GetInformationSubset[ ro_System`ResourceObject, props_List ] :=
    AssociationMap[ ro, props ];


System`ResourceObject /:
  Information`OpenerViewQ[ System`ResourceObject,
                           Alternatives[
                               "ContentElements",
                               "ExternalLinks",
                               "Keywords",
                               "ResourceLocations",
                               "SeeAlso"
                           ]
  ] :=
    True;

End[] (* End Private Context *)

EndPackage[]

SetAttributes[{ResourceObject},
   {ReadProtected, Protected}
];