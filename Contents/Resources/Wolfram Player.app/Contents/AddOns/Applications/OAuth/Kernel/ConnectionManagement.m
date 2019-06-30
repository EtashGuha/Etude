
(Unprotect[#]; Clear[#])& /@ {
  ServiceConnections`ServiceConnections,
  ServiceConnections`SavedConnections,
  ServiceConnections`SaveConnection,
  ServiceConnections`LoadConnection,
  ServiceConnections`DeleteConnection
}

Begin["ServiceConnections`"];

ServiceConnections`ServiceConnections;
ServiceConnections`SavedConnections;
ServiceConnections`SaveConnection;
ServiceConnections`LoadConnection;
ServiceConnections`DeleteConnection;

Begin["`Private`"];

ServiceConnections`ServiceConnections::usage="ServiceConnections returns a list of active ServiceObjects.";
ServiceConnections`SavedConnections::usage="SavedConnections returns a list of available saved ServiceConnections.";
ServiceConnections`SaveConnection::usage="SaveConnection saves an active ServiceObject in the user's account.";
ServiceConnections`LoadConnection::usage="LoadConnection loads a saved ServiceObject from the user's account.";
ServiceConnections`DeleteConnection::usage="DeleteConnection removes a saved connection from the user's account.";

(* Service Connections*)

ServiceConnections`ServiceConnections[args___]:=With[{res=Catch[serviceConnections[args]]},
	res/;!FailureQ[res]
]/;ArgumentCountQ[ServiceConnections`ServiceConnections,Length[{args}],{0,1}]

serviceConnections[all:All:All]:=Cases[serviceObject/@$authenticatedservices,_ServiceObject,{1}]

serviceConnections["OAuth"]:=Cases[serviceObject/@serviceconnections[$authenticatedservices,$oauthservices],_ServiceObject,{1}]
serviceConnections["APIKey"]:=Cases[serviceObject/@serviceconnections[$authenticatedservices,$keyservices],_ServiceObject,{1}]

serviceConnections[name_String]:=Cases[serviceObject/@serviceconnections[$authenticatedservices,{name}],_ServiceObject,{1}]/;MemberQ[$Services,name]

serviceConnections[expr_]:=(Message[ServiceConnections`ServiceConnections::wname,expr];Throw[$Failed])

serviceconnections[ids_,typenames_]:=Select[ids,MemberQ[typenames,serviceName[#]]&]

serviceObject[id_]:=With[{name=serviceName[id]},
	If[StringQ[name],
		ServiceObject[name,"ID"->id]
	]
]

(* Saved Connections*)

ServiceConnections`SavedConnections[args___]:=With[{res=Catch[savedConnections[args]]},
	res/;!FailureQ[res]
]/;ArgumentCountQ[ServiceConnections`SavedConnections,Length[{args}],{0,1}]

savedConnections[all:All:All]:= Block[{$all=True},Flatten[savedconnections/@$Services]];

savedConnections[name_String]:=savedconnections[name]/;MemberQ[$Services,name]

savedConnections[name_String]:=If[TrueQ[findandloadServicePaclet[name]],
	savedconnections[name],
	(Message[ServiceConnections`SavedConnections::wname,name];Throw[$Failed])
]

savedConnections[expr_]:=(Message[ServiceConnections`SavedConnections::wname,expr];Throw[$Failed])

savedconnections[name_]:=With[{ids=savedconnections0[name]},
	appendsavedservicelist[ids];
	ServiceObject[name,"ID"->#]&/@ids
]

savedconnections0[name_String]:=OAuthClient`findSavedOAuthConnections[name]/;MemberQ[$oauthservices,name]
savedconnections0[name_String]:=KeyClient`findSavedKeyConnections[name]/;MemberQ[$keyservices,name]

savedconnections0[name_String]:=If[TrueQ[findandloadServicePaclet[name]],
	If[MemberQ[Join[$oauthservices,$keyservices],name],
		savedconnections0[name],
    {}
	]
]/;TrueQ[$all]

savedconnections0[name_String]:=If[TrueQ[findandloadServicePaclet[name]],
	If[MemberQ[Join[$oauthservices,$keyservices],name],
		savedconnections0[name],
		(Message[ServiceConnections`SavedConnections::nosave,name];Throw[$Failed])
	]
]/;MemberQ[localpacletServices[],name]

savedconnections0[name_String]:=(Message[ServiceConnections`SavedConnections::nosave,name];$Failed)

(* Save Connections*)

ServiceConnections`SaveConnection[args___]:=With[{res=Catch[saveConnection[args]]},
	res/;!FailureQ[res]
]/;ArgumentCountQ[ServiceConnections`SaveConnection,Length[{args}],{1}]

saveConnection[so_ServiceObject]:=(Throw[$Failed])/;!ServiceObjectQ[so]

saveConnection[so_ServiceObject]:=saveconnection[so,getServiceID[so],getServiceName[so]]

saveConnection[expr_]:=(Message[ServiceConnections`SaveConnection::invso,expr];Throw[$Failed])

saveconnection[so_,id_,name_]:=With[{res=saveconnection0[so,name]},
	If[!FailureQ[res],
		appendsavedservicelist[id]
	];
	res
]/;authenticatedServiceQ[id]

saveconnection[so_,_,_]:=(Message[ServiceConnections`SaveConnection::invso,so];Throw[$Failed])

saveconnection0[so_,name_]:=OAuthClient`saveOAuthConnection[so]/;MemberQ[$oauthservices,name]
saveconnection0[so_,name_]:=KeyClient`saveKeyConnection[so]/;MemberQ[$keyservices,name]
saveconnection0[so_,name_]:=(Message[ServiceConnections`SaveConnection::nosave,name];Throw[$Failed])

(* Load Connections*)

Options[ServiceConnections`LoadConnection] = {"StorageLocation"->Automatic};

ServiceConnections`LoadConnection[args___]:=Module[
	{
		res,
		arguments = System`Private`Arguments[ServiceConnections`LoadConnection[args], {1, 2}]
	},
	(
	 res = Catch[loadConnection[args]];
	 res /; !FailureQ[res]
	) /; (arguments=!={})
]

loadConnection[name_String, id_ : Automatic, OptionsPattern[ServiceConnections`LoadConnection]]:= With[{loc = OptionValue["StorageLocation"]},
	If[!validuuidQ[id], Message[ServiceConnections`LoadConnection::invid,id]; Throw[$Failed]];
	If[!MatchQ[loc, Automatic|"Cloud"|"Local"|All], Message[ServiceConnections`LoadConnection::wstr, loc]; Throw[$Failed]];
	loadconnection[name, id, loc]
]/;MemberQ[$Services,name]

loadConnection[name_String, id_ : Automatic, OptionsPattern[ServiceConnections`LoadConnection]]:=With[{loc = OptionValue["StorageLocation"]},
	If[!validuuidQ[id], Message[ServiceConnections`LoadConnection::invid,id]; Throw[$Failed]];
	If[!MatchQ[loc, Automatic|"Cloud"|"Local"|All], Message[ServiceConnections`LoadConnection::wstr, loc]; Throw[$Failed]];
	If[TrueQ[findandloadServicePaclet[name]],
		loadconnection[name, id, loc],
		(Message[ServiceConnections`LoadConnection::wname,name];Throw[$Failed])
	]
]

loadConnection[expr_,___]:=(Message[ServiceConnections`LoadConnection::wname,expr];$Failed)

loadconnection[name_, id_, location_]:=With[{res=loadconnection0[name,id,location]},
	If[Quiet[ServiceObjectQ[res]],
		appendsavedservicelist[getServiceID[res]]
	];
	res
]

loadconnection0[name_, id_, location_]:=OAuthClient`loadOAuthConnection[name,id,location]/;MemberQ[$oauthservices,name]
loadconnection0[name_, id_, location_]:=KeyClient`loadKeyConnection[name,id,location]/;MemberQ[$keyservices,name]

loadconnection0[name_, id_, location_]:=If[TrueQ[findandloadServicePaclet[name]],
	If[MemberQ[Join[$oauthservices,$keyservices],name],
		loadconnection0[name,id,location],
		(Message[ServiceConnections`LoadConnection::nosave,name];Throw[$Failed])
	]
]/;MemberQ[localpacletServices[],name]

loadconnection0[name_, rest__]:=(Message[ServiceConnections`LoadConnection::nosave,name];$Failed)

ServiceConnections`LoadConnection[___]:=$Failed

(*Delete Connections*)

ServiceConnections`DeleteConnection[args___]:=With[{res=Catch[deleteConnection[args]]},
	res/;!FailureQ[res]
]/;ArgumentCountQ[ServiceConnections`DeleteConnection,Length[{args}],0,1]

deleteConnection[all:All:All]:=(deleteConnection/@Union[serviceConnections[All],savedConnections[All]])

deleteConnection[service_String]:=(deleteConnection/@Union[serviceConnections[service],savedConnections[service]])/;MemberQ[$Services,service]

deleteConnection[so_ServiceObject]:=(Throw[$Failed])/;!ServiceObjectQ[so]

deleteConnection[so_ServiceObject]:=deleteconnection[so,getServiceID[so],getServiceName[so]]

deleteConnection[expr_]:=(Message[ServiceConnections`DeleteConnection::invso,expr];Throw[$Failed])

deleteconnection[so_, id_, name_]:=With[{res=deleteconnection0[so,name]},
	If[!FailureQ[res],
		removefromsavedservicelist[id]
	];
  res
]

deleteconnection0[so_, name_]:=OAuthClient`deleteOAuthConnection[so]/;MemberQ[$oauthservices,name]
deleteconnection0[so_, name_]:=KeyClient`deleteKeyConnection[so]/;MemberQ[$keyservices,name]
deleteconnection0[so_, name_]:=(Message[ServiceConnections`SaveConnection::nosave,name];$Failed)

End[];
End[];

SetAttributes[{
	ServiceConnections`ServiceConnections,
	ServiceConnections`SavedConnections,
	ServiceConnections`SaveConnection,
	ServiceConnections`LoadConnection,
	ServiceConnections`DeleteConnection
	},
	{ReadProtected,Protected}
];
