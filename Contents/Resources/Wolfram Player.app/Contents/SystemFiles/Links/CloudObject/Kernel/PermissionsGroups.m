(* ::Package:: *)

BeginPackage["CloudObject`"]

System`PermissionsGroup;
System`CreatePermissionsGroup;
System`PermissionsGroups;
System`AddUsers;
System`RemoveUsers;
System`SetUsers;
System`PermissionsGroupMemberQ;
Hold[System`$PermissionsGroupBase];

Begin["`Private`"]

userUUIDToDisplay[uuid_String, base_] :=
    Module[{userDataStr, userData},
        userDataStr = responseToString @ execute[getCloud[base], "GET", {"users", uuid, "info"}];
        If[userDataStr === $Failed, Return[uuid]];
        userData = importFromJSON[userDataStr];
        Lookup[userData, "displayName", Lookup[userData, "email", uuid]]
    ]
userUUIDToDisplay[Null, base_] = Null
userUUIDToDisplay[$Failed, base_] = $Failed

groupMimeType = "application/vnd.wolfram.group"

execute[group_PermissionsGroup, rest___] := execute[CloudObject @@ group, rest]

(* $PermissionsGroupBase *)

$PermissionsGroupBase := CloudObject["/PermissionsGroup"]

(* PermissionsGroup *)

PermissionsGroup[name_String] :=
    Module[ {url, permissionsGroupObj},
        PermissionsGroup[url]
        /; (
        permissionsGroupObj = Quiet@CloudObject[name, $PermissionsGroupBase];
        url = First[permissionsGroupObj];
        url =!= name && !FailureQ[permissionsGroupObj]
        )
    ]

PermissionsGroup[user_String, name_String] :=
    With[ {permissionsGroupCloudObj = Quiet@CloudObject[name, "user:" <> user <> "/PermissionsGroup"]},
        If[ FailureQ[permissionsGroupCloudObj],
            permissionsGroupFailureObject[name],
            PermissionsGroup @@ permissionsGroupCloudObj
        ]
    ]

PermissionsGroup[PermissionsGroup[url_]] := PermissionsGroup[url]
PermissionsGroup /: CloudObject[PermissionsGroup[url_]] := CloudObject[url]

PermissionsGroup[url_]["Members"] :=
    Module[{members},
        members = responseToStringList[execute[CloudObject[url], {"group"}], PermissionsGroup];
        If[members === $Failed, Return[$Failed]];
        userUUIDToDisplay[#, url] & /@ members
    ]
PermissionsGroup[url_]["Creator"] :=
    CloudObjectInformation[CloudObject[url], "OwnerWolframID"]
PermissionsGroup[url_]["CreationDate"] :=
    CloudObjectInformation[CloudObject[url], "Created"]

permissionsGroupFailureObject[groupname_String] := (Message[PermissionsGroup::invgrp, groupname]; Return[Failure["InvalidPermissionsGroupName", <|"MessageTemplate" -> PermissionsGroup::invgrp, "MessageParameters" -> {groupname}|>]])

(* CreatePermissionsGroup *)

CreatePermissionsGroup[name_String] :=
    With[ {permissionsGroupCloudObj = Quiet@writeObject[CloudObject[name, $PermissionsGroupBase], "", groupMimeType,
        Automatic, None, Null, {}, {}, CreatePermissionsGroup]},
        If[ FailureQ[permissionsGroupCloudObj],
            permissionsGroupFailureObject[name],
            PermissionsGroup @@ permissionsGroupCloudObj
        ]
    ]

CreatePermissionsGroup[name_String, users_] :=
    Module[{group},
        group = CreatePermissionsGroup[name];
        If[group === $Failed, Return[$Failed]];
        If[iAddSetUsers["PUT", CreatePermissionsGroup, group, users] === $Failed, Return[$Failed]];
        group
    ]

CreatePermissionsGroup[name_] := (Message[CreatePermissionsGroup::string, 1, HoldForm[CreatePermissionsGroup[name]]]; $Failed)

CreatePermissionsGroup[args___] :=
    (ArgumentCountQ[CreatePermissionsGroup, Length[DeleteCases[{args}, _Rule, Infinity]], 1, 2]; Null /; False)

(* AddUsers, SetUsers *)

normalizeUserPermissions[head_Symbol, user_String, outerPerm_ : {}] := normalizeUserPermissions[head, {user}, outerPerm]
normalizeUserPermissions[head_Symbol, user_String -> perm_, outerPerm_ : {}] := normalizeUserPermissions[head, {user}, perm]
normalizeUserPermissions[head_Symbol, users_List, outerPerm_ : {}] := normalizeUserPermissions[head, users -> outerPerm]
normalizeUserPermissions[head_Symbol, users_List -> perm_, outerPerm_ : {}] := (normalizeUser[#] -> perm) & /@ users
normalizeUserPermissions[head_Symbol, ___] := (Message[head::notparam]; $Failed)

normalizeUser[user_String]:= If[UUIDQ[user], userUUIDPrefix <> user, user]
normalizeUser[user_] := user

(* TODO: Instead of this, we should implement and use SetPermissions. *)
updatePermissions[obj_CloudObject, perms_] :=
    Module[{existing, existingUsers, new},
        existing = Lookup[Options[obj, Permissions], Permissions];
        existingUsers = Keys[existing];
        new = Normal[Append[Association[existing], perms]];
        (* Do not set permissions for users that haven't had permissions before and who wouldn't get permissions.
        That avoids adding collaborators unnecessarily
        (which would lead to an error when the owner adds themselves to a group). *)
        new = DeleteCases[new, _?(FreeQ[existingUsers, #]&) -> {}];
        SetOptions[obj, Permissions->new];
    ]

iAddSetUsers[method_, head_, group_PermissionsGroup, users_] :=
    Module[{perms, usersList, result},
        perms = normalizeUserPermissions[head, users];
        If[perms === $Failed, Return[$Failed]];
        usersList = StringJoin[Riffle[First /@ perms, "\n"]];
        result = responseCheck[execute[group, method, {"group"}, Body->usersList], head];
        If[result === $Failed, Return[$Failed]];
        updatePermissions[Quiet[CloudObject @@ group], perms];
    ]

iAddSetUsers[method_, head_, group_, rest___] := iAddSetUsers[method, head, PermissionsGroup[group], rest]
iAddSetUsers[method_, head_, groups_List, rest___] := (iAddSetUsers[method, head, #, rest] & /@ groups;)

AddUsers[group_, users_] := iAddSetUsers["POST", AddUsers, group, users]

AddUsers[args___] :=
    (ArgumentCountQ[AddUsers, Length[DeleteCases[{args}, _Rule, Infinity]], {2}]; Null /; False)

SetUsers[group_, users_] := iAddSetUsers["PUT", SetUsers, group, users]

SetUsers[args___] :=
    (ArgumentCountQ[SetUsers, Length[DeleteCases[{args}, _Rule, Infinity]], {2}]; Null /; False)

(* RemoveUsers *)

RemoveUsers[group_PermissionsGroup, user_String] := 
    With[{userVal = If[UUIDQ[user], userUUIDPrefix <> user, user]},
        responseCheck @ execute[group, "DELETE", {"group", userVal}]
    ]

(* TODO: Batch-remove users in a single request. *)
RemoveUsers[group_PermissionsGroup, users_List] :=
    Scan[If[RemoveUsers[group, #] === $Failed, Return[$Failed]] &, users]

RemoveUsers[group_, rest___] := RemoveUsers[PermissionsGroup[group], rest]
RemoveUsers[groups_List, rest___] := RemoveUsers[#, rest] & /@ groups

(* PermissionsGroupMemberQ *)
PermissionsGroupMemberQ[group:PermissionsGroup[url_], user_String] := 
	Module[{members, json, data, userData},
        If[$CloudUserID === None, 
        	CloudConnect[];
        	If[!TrueQ[$CloudConnected], Message[head::notauth]; Return[$Failed]]
        ];
        members = responseToStringList[execute[CloudObject[url], {"group"}], PermissionsGroupMemberQ];
        If[UUIDQ[user],
        	MemberQ[StringReplace[#, userUUIDPrefix ~~ id__ :> id]& /@ members, user],
        (* else *)
        	json = 
        	Replace[execute[$CloudBase, "GET", {"users"}, Parameters -> {"id" ->  user}],
            {
                {_, bytes_List} :> FromCharacterCode[bytes],
                HTTPError[403, ___] :> Message[PermissionsGroupMemberQ::noaccess, user],                                       
                HTTPError[404, ___] :> Message[PermissionsGroupMemberQ::invusr, user],
                other_ :> checkError[other, PermissionsGroupMemberQ]
             }
        	];
        	If[StringQ[json],
        		data = importFromJSON[json];
        		If[MatchQ[data, {{__Rule}..}],
        			userData = Quiet[Select[data, KeyExistsQ[#, "uuid"] &][[1]]];
        			If[ListQ[userData],
        				MemberQ[StringReplace[#, userUUIDPrefix ~~ id__ :> id]& /@ members, Lookup[userData, "uuid"]],
        			    Message[PermissionsGroupMemberQ::srverr]; Return[$Failed]]
        		],
				$Failed
			]        	
        ]        
    ]   

(* PermissionsGroups *)

(* TODO: Take into account groups owned by other users that the authenticated user has access to. *)
PermissionsGroups[] := PermissionsGroup @@@ CloudObjects["/PermissionsGroup", groupMimeType]

(* isPermissionsGroup *)
isPermissionsGroup[group_PermissionsGroup] :=
	With[{co = Quiet[CloudObject@@group]},
		CloudObjectInformation[co,"MIMEType"]===groupMimeType
	]
    
invalidPermissionsGroups[permissions_]:=
    Module[{groups, nonExistantGroups, notAllValid},
    	groups = Cases[Replace[permissions, rl_Rule -> {rl}], HoldPattern[_PermissionsGroup->_]][[All,1]];
    	nonExistantGroups = Select[groups, !isPermissionsGroup[#]&];
    	notAllValid=(nonExistantGroups =!= {});
    	If[notAllValid, Message[PermissionsGroup::invusr, #]&/@nonExistantGroups];
    	notAllValid
    ]

End[]

EndPackage[]
