(* Wolfram Language Package *)

BeginPackage["CloudObject`UserManagement`"]

CloudUser
CloudUsers
CloudUserInformation
AddAdminStatus
RemoveAdminStatus

CreateCloudUser
DeleteCloudUser

Begin["CloudObject`Private`"]

$cloudUserInformation = {
    "cloudUserID" -> "CloudUserID",
    "cloudUserUUID" -> "CloudUserUUID",
    "userBaseURL" -> "UserURLBase",
    "lastLogin" -> "LastLogin",
    "active" -> "Active",
    "role" -> "Admin",
    "true" -> True,
    "false" -> False,
    "registrationDate" -> "RegistrationDate",
    "ROLE_ADMIN" -> "Admin"
};

getUserInformation[user_, fields_, msghd_] :=
    Module[ {json},
        If[ StringQ[user],
            json = Replace[execute[$CloudBase, "GET", {"users"}, Parameters -> {"id" -> user, "fields" -> fields}], {
                      {_String, content_List} :> FromCharacterCode[content],
                      HTTPError[404, ___] :> (Message[msghd::usernf];
                                              Return@$Failed),
                      other_ :> (checkError[other, msghd];
                                 Return[$Failed])
               }];
            Most@importFromJSON[json],
            Message[msghd::inv];
            $Failed
        ]
    ]

patternDetail = { role_, lastLogin_, cloudUserUUID_, active_, registrationDate_, cloudUserID_, userBaseURL_ };
patternBasic = { cloudUserID_, cloudUserUUID_, userBaseURL_ };

getCloudUser[user_] :=
    With[ {userInfo = getUserInformation[user, "basic", CloudUsers]},
        CloudUser[Association[#]] & /@ (ReplaceAll[#, patternBasic :> {"CloudUserUUID" -> Values@cloudUserUUID,
            "CloudUserID" -> Values@cloudUserID,
            "UserURLBase" -> Values@userBaseURL}] & /@ userInfo)
    ]

missingUser[user_] := Missing["NotFound", user]

CloudUsers[] := getCloudUser[""]

setAdminStatus[user_, msghd_] :=
    Replace[
        execute[$CloudBase, "PUT", {"users", user, "admin"}],
            {
                {_String, content_List} :> Null,
                other_ :> (checkError[other, msghd];
                           $Failed)
            }
        ]

removeAdminStatus[user_, msghd_] :=
    Replace[
        execute[$CloudBase, "DELETE", {"users", user, "admin"}],
            {
                {_String, _List} :> Null (* success *),
                other_ :> (checkError[other, msghd];
                           $Failed)
            }
        ]

Options[getCloudUserProperty] = {"Elements"-> Automatic}

getCloudUserProperty[user_Association, opts:OptionsPattern[]] :=
        ReplaceAll[user, info_ :> Lookup[info, OptionValue["Elements"]]]

CloudUserInformation[user_:""] :=
    With[ {allUserInfo = getUserInformation[user, "all", CloudUserInformation]},
        If[ user === "",
            Association[ReplaceAll[#, $cloudUserInformation]] & /@ allUserInfo,
            If[ allUserInfo === $Failed,
                missingUser[user],
                First[Association[ReplaceAll[#, $cloudUserInformation]] & /@ allUserInfo]
            ]
        ]
    ]

CloudUserInformation[user_, property_String] := With[{allCloudUserInfo = CloudUserInformation[user]},
	Map[getCloudUserProperty[#, "Elements" -> property] &, {allCloudUserInfo}]
]

CloudUserInformation[user_CloudUser] := With[{cloudUserID = getCloudUserProperty[user, "Elements" -> "CloudUserID"]},
	ReplaceAll[CloudUserInformation[cloudUserID], $cloudUserInformation]
]

CloudUserInformation[user_CloudUser, property_String] := With[{cloudUser = CloudUserInformation[getCloudUserProperty[user, "Elements" -> "CloudUserID"]]},
	Flatten[getCloudUserProperty[#, "Elements" -> property] & /@ {cloudUser}]
]

AddAdminStatus[user_CloudUser] := setAdminStatus[First[CloudUserInformation[user, "CloudUserUUID"]], AddAdminStatus]

AddAdminStatus[user_String] := With[{cloudUserUUID = First[CloudUserInformation[user, "CloudUserUUID"]]},
	setAdminStatus[cloudUserUUID, AddAdminStatus];
]

RemoveAdminStatus[user_CloudUser] := removeAdminStatus[First[CloudUserInformation[user, "CloudUserUUID"]], RemoveAdminStatus]

RemoveAdminStatus[user_String] := With[{cloudUserUUID = First[CloudUserInformation[user, "CloudUserUUID"]]},
	removeAdminStatus[cloudUserUUID, RemoveAdminStatus];
]

End[] (* End Private Context *)
EndPackage[]
