Begin["FacebookOAuth`"] 

Get["FacebookFunctions.m"];

Begin["`Private`"]

(******************************* Facebook *************************************)

Clear[facebookcookeddata]

(* Authentication information *)

facebookdata[]:=
    If[ TrueQ[OAuthClient`Private`$UseChannelFramework],
{
        "OAuthVersion"      -> "2.0",
        "ServiceName"       -> "Facebook", 
        "AuthorizeEndpoint" -> "https://www.facebook.com/v2.12/dialog/oauth", 
        "AccessEndpoint"    -> "https://graph.facebook.com/v2.12/oauth/access_token",
        "RedirectURI"       -> "WolframConnectorChannelListen",
        "Blocking"          -> False,
        "AuthorizationFunction"  -> "Facebook",
        "RedirectURLFunction"    -> (#1&),
        "VerifierLabel"     -> "code",
        "ClientInfo"        -> {"Wolfram","Token"},
        "AuthenticationDialog" -> "WolframConnectorChannel",
        "Gets"                :> Join[{"FriendIDs","Friends","UserData","PageData","PermissionList",
                "ActivityRecentHistory","ActivityTypes","ActivityWeeklyDistribution", "Cover",
                "Books","Feeds","Movies","Music","Photos","PhotoLinks","Picture","Posts",
                "WallMostCommentedPostData","WallMostLikedPostData",
                "WallPostLength","WallPostLengthTimeline","WallWordFrequencies",
                "PostEventSeries","PostTimeline"},
                facebookuserdataNames,DeleteCases[facebookpageNames,Alternatives@@commonuserpagenames]],
        "Posts"             -> {"PostLink","PostMessage"},
        "RawGets"           -> {"RawFriendsList", "RawUserData","RawPageData", "RawPermissions", "RawGetPosts","RawGetFeeds","RawUserBooks","RawUserMovies","RawUserMusic",
                "RawUserPhotos","RawUserPicture"},
        "RawPosts"          -> {"RawSendPost"},
        "Information"       -> "A service for sending and receiving data from a Facebook account",
        "AccessTokenExtractor"  -> "JSON/2.0"
    }
    ,
    {
        "OAuthVersion"      -> "2.0",
        "ServiceName"       -> "Facebook", 
        "AuthorizeEndpoint" -> "https://www.facebook.com/v2.12/dialog/oauth", 
        "AccessEndpoint"    -> "https://graph.facebook.com/v2.12/oauth/access_token",
        "RedirectURI"       -> "https://www.wolfram.com/oauthlanding?service=Facebook",
        "VerifierLabel"     -> "code",
        "ClientInfo"        -> {"Wolfram","Token"},
        "AuthenticationDialog" :> (OAuthClient`tokenOAuthDialog[#, "Facebook",fbicon]&),
        "Gets"              :> Join[{"FriendIDs","Friends","UserData","PageData","PermissionList",
                "ActivityRecentHistory","ActivityTypes","ActivityWeeklyDistribution", "Cover",
                "Books","Feeds","Movies","Music","Photos","PhotoLinks","Picture","Posts",
                "WallMostCommentedPostData","WallMostLikedPostData",
                "WallPostLength","WallPostLengthTimeline","WallWordFrequencies",
                "PostEventSeries","PostTimeline"},
                facebookuserdataNames,DeleteCases[facebookpageNames,Alternatives@@commonuserpagenames]],
        "Posts"             -> {"PostLink","PostMessage"},
        "RawGets"           -> {"RawFriendsList", "RawUserData","RawPageData", "RawPermissions", "RawGetPosts","RawGetFeeds","RawUserBooks","RawUserMovies","RawUserMusic",
                "RawUserPhotos","RawUserPicture"},
        "RawPosts"          -> {"RawSendPost"},
        "Information"       -> "A service for sending and receiving data from a Facebook account",
        "AccessTokenExtractor"  -> "JSON/2.0"
}
]

(* a function for importing the raw data - usually json or xml - from the service *)
facebookimport[$Failed]:=Throw[$Failed]
facebookimport[json_String]:= With[{res = Quiet@Developer`ReadRawJSONString[json]},
    If[ FailureQ[res], Throw[$Failed]];
    If[ !KeyExistsQ[res, "error"],
        res,
        Message[ServiceExecute::apierr, res["error"]["message"]];
        Throw[$Failed]
    ]
]

facebookimport[raw_]:=raw
 
(*** Raw ***) 
facebookdata["RawFriendsList"] = {
    "URL"                    -> (ToString@StringForm["https://graph.facebook.com/v2.12/`1`/friends", formatuser[##]]&), (* only supports "me" as user *)
    "PathParameters"        -> {"UserID"},
    "Parameters"            -> {"limit","fields"},
    "ResultsFunction"        -> facebookimport,
    "RequiredPermissions"    -> {"user_friends"}
}
 
$facebookuserdatapermissions = {};

(* 1 *)
facebookdata["RawUserData"] = {
    "URL"                    -> (ToString@StringForm["https://graph.facebook.com/v2.12/`1`", formatuser[##]]&),
    "PathParameters"         -> {"UserID"},
    "Parameters"             -> {"fields"},
    "HTTPSMethod"            -> "GET",
    "ResultsFunction"        -> facebookimport,
    "RequiredPermissions"    :> $facebookuserdatapermissions
}

(* 2 *)
facebookdata["RawUserBooks"] = {
    "URL"                    -> (ToString@StringForm["https://graph.facebook.com/v2.12/`1`/books", formatuser[##]]&),
    "PathParameters"         -> {"UserID"},
    "HTTPSMethod"            -> "GET",
    "ResultsFunction"        -> facebookimport,
    "RequiredPermissions"    :> $facebookuserdatapermissions
}

facebookdata["RawUserMovies"] = {
    "URL"                    -> (ToString@StringForm["https://graph.facebook.com/v2.12/`1`/movies", formatuser[##]]&),
    "PathParameters"         -> {"UserID"},
    "HTTPSMethod"            -> "GET",
    "ResultsFunction"        -> facebookimport,
    "RequiredPermissions"    :> $facebookuserdatapermissions
}

facebookdata["RawUserMusic"] = {
    "URL"                    -> (ToString@StringForm["https://graph.facebook.com/v2.12/`1`/music", formatuser[##]]&),
    "PathParameters"         -> {"UserID"},
    "Parameters"             -> {"limit"},
    "HTTPSMethod"            -> "GET",
    "ResultsFunction"        -> facebookimport,
    "RequiredPermissions"    :> $facebookuserdatapermissions
}

(* 3 *)
facebookdata["RawAccounts"] = {
    "URL"                    -> (ToString@StringForm["https://graph.facebook.com/v2.12/`1`/accounts", formatuser[##]]&),
    "PathParameters"         -> {"UserID"},
    "HTTPSMethod"            -> "GET",
    "ResultsFunction"        -> facebookimport,
    "RequiredPermissions"    -> {}
}
    
facebookdata["RawPermissions"] = {
    "URL"                    -> "https://graph.facebook.com/v2.12/me/permissions",
    "HTTPSMethod"            -> "GET",
    "ResultsFunction"        -> facebookimport,
    "RequiredPermissions"    -> {}
} 
    
facebookdata["RawGetPosts"] = {
    "URL"                    -> (ToString@StringForm["https://graph.facebook.com/v2.12/`1`/posts", formatuser[##]]&),
    "PathParameters"         -> {"UserID"},
    "Parameters"             -> {"limit","fields","summary"},
    "HTTPSMethod"            -> "GET",
    "ResultsFunction"        -> facebookimport,
    "RequiredPermissions"    -> {"user_posts"}
}
       
$facebookfeedpermissions = {};
facebookdata["RawGetFeeds"] = {
    "URL"                    -> (ToString@StringForm["https://graph.facebook.com/v2.12/`1`/feed", formatuser[##]]&),
    "PathParameters"         -> {"UserID"},
    "Parameters"             -> {"limit","fields","summary"},
    "HTTPSMethod"            -> "GET",
    "ResultsFunction"        -> facebookimport,
    "RequiredPermissions"    :> $facebookfeedpermissions
} 
     
(* Raw Post *)
facebookdata["RawSendPost"] = {
    "URL"                    -> "https://graph.facebook.com/v2.12/me/feed",
    "Parameters"             -> {"message","link"},
    "HTTPSMethod"            -> "POST",
    "ResultsFunction"        -> facebookimport,
    "RequiredPermissions"    -> {"publish_actions"}
}

$facebookphotopermissions={"user_photos"};
facebookdata["RawUserPhotos"] = {
    "URL"                    -> (ToString@StringForm["https://graph.facebook.com/v2.12/`1`/photos", formatuser[##]]&),
    "PathParameters"         -> {"UserID"},
    "Parameters"             -> {"limit","fields"},
    "HTTPSMethod"            -> "GET",
    "ResultsFunction"        -> facebookimport,
    "RequiredPermissions"    :> $facebookphotopermissions
}
    
facebookdata["RawUserPicture"] = {
    "URL"                    ->  (ToString@StringForm["https://graph.facebook.com/v2.12/`1`/picture", formatuser[##]]&),
    "PathParameters"         -> {"UserID"},
    "Parameters"             -> {"type","redirect","height","width"},
    "HTTPSMethod"            -> "GET",
    "ResultsFunction"        -> Identity,
    "RequiredPermissions"    -> {}
}

facebookdata["RawPageData"] = {
    "URL"                    -> (ToString@StringForm["https://graph.facebook.com/v2.12/`1`", formatpage[##]]&),
    "PathParameters"         -> {"PageID"},
    "Parameters"             -> {"fields"},
    "HTTPSMethod"            -> "GET",
    "ResultsFunction"        -> facebookimport,
    "RequiredPermissions"    -> {}
}

 
facebookdata["icon"]:=fbicon
   
facebookdata[___]:=$Failed
(****** Cooked Properties ******)

(******************** USERS **************************)
facebookuserdatafields={"id", "name", "first_name", "middle_name", "last_name", "gender", "locale", "languages", "link",
    "third_party_id", "timezone", "updated_time", "installed", "is_verified", "verified", "birthday", "picture", "cover",
    "currency", "devices", "email", "hometown", "location", "inspirational_people", "favorite_athletes", "favorite_teams",
    "sports", "quotes", "significant_other"};
facebookuserdataNames={"UserID", "FullName", "FirstName", "MiddleName", "LastName","Gender", "Locale", "Languages", "Link",
    "ThirdPartyID", "Timezone", "UpdatedTime", "WolframConnectedQ", "IsVerified", "Verified", "Birthday", "PictureLink", "CoverLink",
    "Currency", "Devices", "Email", "Hometown", "Location", "InspirationalPeople", "FavoriteAthletes", "FavoriteTeams",
    "Sports", "Quotes", "SignificantOther"};
facebookuserdataRules=Thread[facebookuserdataNames->facebookuserdatafields];
facebookuserdatapermissionrules={"quotes"->"user_likes","birthday"->"user_birthday","email"->"email",
    "hometown"->"user_hometown","inspirational_people"->"user_likes","favorite_athletes"->"user_likes","favorite_teams"->"user_likes",
    "languages"->"user_likes","location"->"user_location"};

commonuserpagenames={"Birthday", "CoverLink", "Hometown", "Link", "Location", "Username", "Website"}; 
              
facebookcookeddata["UserData",id_,args_]:= Block[
    {invalidParameters,$facebookuserdatapermissions={},params={},user,rawdata,data},

    invalidParameters = Select[Keys[args],!MemberQ[{"UserID"},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Facebook"]&/@invalidParameters;
        Throw[$Failed]
    ];

    user = ToString@Lookup[args, "UserID", "me"];
    If[ StringMatchQ[user, DigitCharacter..] || SameQ[user, "me"],
        AppendTo[params, "UserID"-> user]
        ,
        Message[ServiceExecute::nval,"UserID","Facebook"];
        Throw[$Failed]
    ];

    If[ user === "me",
        $facebookuserdatapermissions={"email","user_birthday","user_hometown","user_location","user_likes"}
    ];
    rawdata=OAuthClient`rawoauthdata[id,"RawUserData",Join[params,{"fields"->StringJoin[Riffle[facebookuserdatafields,","]]}]];      
    data=facebookimport[rawdata];
    Dataset[FFormatUserInformation[data]]
]

facebookcookeddata[prop:(Alternatives@@facebookuserdataNames),id_,args_]:=Block[
    {invalidParameters,field=Lookup[facebookuserdataRules, prop],$facebookuserdatapermissions,params={},user,rawdata,data},
    $facebookuserdatapermissions={Lookup[facebookuserdatapermissionrules,field,Sequence@@{}]};

    invalidParameters = Select[Keys[args],!MemberQ[{"UserID"},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Facebook"]&/@invalidParameters;
        Throw[$Failed]
    ];

    user = ToString@Lookup[args, "UserID", "me"];
    If[ StringMatchQ[user, DigitCharacter..] || SameQ[user, "me"],
        AppendTo[params, "UserID"-> ToString@user]
        ,
        Message[ServiceExecute::nval,"UserID","Facebook"];
        Throw[$Failed]
    ];
    If[ user =!= "me",
        $facebookuserdatapermissions={}
    ];
    rawdata=OAuthClient`rawoauthdata[id,"RawUserData",Join[params,{"fields"->field}]];          
    data=FFormatUserInformation[facebookimport[rawdata]];
    If[ KeyExistsQ[data,prop],
        data[prop],
        Missing["NotAvailable"]
    ]           
] /; !KeyExistsQ[args,"PageID"]

facebookcookeddata[prop:("Books"|"Movies"|"Music"),id_,args_]:=Block[
    {invalidParameters,data,user,params={"UserID"->"me"},rawdata,$facebookuserdatapermissions={"user_likes"}},

    invalidParameters = Select[Keys[args],!MemberQ[{},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Facebook"]&/@invalidParameters;
        Throw[$Failed]
    ];

    rawdata=OAuthClient`rawoauthdata[id,"RawUser"<>prop,params];          
    data=FFormatBMM[prop,#]&/@facebookimport[rawdata]["data"]
]

facebookcookeddata[prop:"Picture",id_,args_]:=Block[
    {invalidParameters,data,user,params={"redirect"->"false","width"->"320"},rawdata,rawprop,img,$facebookuserdatapermissions={}},

    invalidParameters = Select[Keys[args],!MemberQ[{"UserID"},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Facebook"]&/@invalidParameters;
        Throw[$Failed]
    ];

    user = ToString@Lookup[args, "UserID", "me"];
    If[ StringMatchQ[user, DigitCharacter..] || SameQ[user, "me"],
        AppendTo[params, "UserID"-> ToString@user]
        ,
        Message[ServiceExecute::nval,"UserID","Facebook"];
        Throw[$Failed]
    ];
    
    rawdata=OAuthClient`rawoauthdata[id,"RawUserPicture",params]; 
    If[ MatchQ[Lookup[params, "redirect"], "false"],
        (* we have the meta data *)
         data = facebookimport[rawdata];
         img = iImportImage[data["data"]["url"]]
        ,
        (* we have the image *)
        img = ImportString[rawdata]    
    ];
    img/;ImageQ[img]
]

facebookcookeddata[prop:"Cover",id_,args_]:=Block[
    {invalidParameters,data,user,params={"fields"->"cover"},rawdata,rawprop,img,$facebookuserdatapermissions={}},

    invalidParameters = Select[Keys[args],!MemberQ[{"UserID"},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Facebook"]&/@invalidParameters;
        Throw[$Failed]
    ];

    user = Lookup[args, "UserID", Lookup[args, "PageID", "me"]];
    If[ StringQ[user] || (IntegerQ[user] && Positive[user]),
        AppendTo[params, "UserID"-> ToString@user]
        ,
        Message[ServiceExecute::nval,If[KeyExistsQ[args,"PageID"],"PageID","UserID"],"Facebook"];
        Throw[$Failed]
    ];
    
    rawdata=OAuthClient`rawoauthdata[id,"RawUserData",params]; 
    data=Catch[facebookimport[rawdata]];
    If[ FailureQ[data], Return[Missing["NotAvailable"]]];
    If[ KeyExistsQ[data,"cover"],
        img=iImportImage[data["cover"]["source"]];        
        If[ ImageQ[img],
            img,
            Missing["NotAvailable"]
        ]
        ,
        Missing["NotAvailable"]
    ]
]

facebookcookeddata[prop:("FriendIDs"|"Friends"),id_,args_]:=Block[
    {invalidParameters,data,rawdata},

    invalidParameters = Select[Keys[args],!MemberQ[{},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Facebook"]&/@invalidParameters;
        Throw[$Failed]
    ];

    rawdata=OAuthClient`rawoauthdata[id,"RawFriendsList",{"limit"->"500"}];          
     data=facebookimport[rawdata];
     Switch[prop,
         "Friends", #["name"]& /@ data["data"],
         "FriendIDs", #["id"]& /@ data["data"]
     ]
]

$facebookphotolimit=20;

facebookcookeddata[prop:("Photos"|"PhotoLinks"),id_,args_]:=Block[
    {invalidParameters,res,limit,params={"fields"->"id,caption,created_time,images","UserID"->"me"},rawdata,links},

    invalidParameters = Select[Keys[args],!MemberQ[{MaxItems},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Facebook"]&/@invalidParameters;
        Throw[$Failed]
    ];

    limit = Lookup[args, MaxItems, $facebookphotolimit];
    If[ IntegerQ[limit] && limit>0,
        AppendTo[params, "limit"-> ToString@limit]
        ,
        Message[ServiceExecute::nval,MaxItems,"Facebook"];
        Throw[$Failed]
    ];

    rawdata=OAuthClient`rawoauthdata[id,"RawUserPhotos",params];          
    res=facebookimport[rawdata];
 
    links = Flatten[ Lookup[#,"source",Sequence@@{}]& /@ MaximalBy[#["height"]&] /@ Lookup[res["data"],"images",{}]];
    If[ prop === "Photos",
        iImportImage/@links
        ,
        URL/@links    
    ]
]

(************* PAGES ******************)

facebookpagefields={"id","about", "attire", "band_members", "best_page", "birthday", "booking_agent", "can_post", "category", "category_list", "checkins", 
    "company_overview", "cover", "current_location", "description", "directed_by", "founded", "general_info", "general_manager", 
    "hometown", "is_published", "is_unclaimed", "likes", "link", "location", "mission", "name", "parking", "phone", "press_contact", 
    "price_range", "products", "restaurant_services", "restaurant_specialties", "talking_about_count", "username", "website", "were_here_count"};
facebookpageNames={"PageID", "About", "Attire", "BandMembers", "BestPage", "Birthday", "BookingAgent", "CanPost", "Category", "CategoryList", "Checkins",
    "CompanyOverview", "CoverLink", "CurrentLocation", "Description", "DirectedBy", "Founded", "GeneralInfo", "GeneralManager", "Hometown", "IsPublished",
    "IsUnclaimed", "Likes", "Link", "Location", "Mission", "PageName", "Parking", "Phone", "PressContact", "PriceRange", "Products", "RestaurantServices",
    "RestaurantSpecialties", "TalkingAboutCount", "Username", "Website", "WereHereCount"};
facebookpageRules=Thread[facebookpageNames->facebookpagefields]

facebookcookeddata["PageData",id_,args_]:=Block[
    {invalidParameters,rawdata,page,params={},data},

    invalidParameters = Select[Keys[args],!MemberQ[{"PageID"},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Facebook"]&/@invalidParameters;
        Throw[$Failed]
    ];

    page = Lookup[args, "PageID", Message[ServiceExecute::nparam,"PageID"]; Throw[$Failed]];
    If[ StringQ[page] || (IntegerQ[page] && Positive[page]),
        AppendTo[params, "PageID"-> ToString@page]
        ,
        Message[ServiceExecute::nval,"PageID","Facebook"];
        Throw[$Failed]
    ];

    rawdata=OAuthClient`rawoauthdata[id,"RawPageData",Join[params,{"fields"->StringJoin[Riffle[facebookpagefields,","]]}]];      
    data=facebookimport[rawdata];
    Dataset[FFormatPageInformation[data]]
]

facebookcookeddata[prop:(Alternatives@@facebookpageNames),id_,args_]:=Block[
    {invalidParameters,field=Lookup[facebookpageRules, prop],params={},page,rawdata,data},

    invalidParameters = Select[Keys[args],!MemberQ[{"PageID"},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Facebook"]&/@invalidParameters;
        Throw[$Failed]
    ];

    page = Lookup[args, "PageID", Message[ServiceExecute::nparam,"PageID"]; Throw[$Failed]];
    If[ StringQ[page] || (IntegerQ[page] && Positive[page]),
        AppendTo[params, "PageID"-> ToString@page]
        ,
        Message[ServiceExecute::nval,"PageID","Facebook"];
        Throw[$Failed]
    ];

    rawdata=OAuthClient`rawoauthdata[id,"RawPageData",Join[params,{"fields"->field}]];          
    data=FFormatPageInformation[facebookimport[rawdata]];
    If[ KeyExistsQ[data,prop],
        data[prop],
        Missing["NotAvailable"]
    ]           
]

(********************* Posts **************************)

facebookpostfields={"id", "from", "to", "message", "message_tags", "picture", "link", "name", "caption", "description", "source", "icon", "actions",
    "privacy", "type", "likes", "place", "story", "with_tags", "comments", "object_id", "application", "created_time", "updated_time", "status_type"};

$facebookfeedlimit=20;

facebookcookeddata[prop:("Feeds"|"Posts"),id_,args_]:=Block[
    {invalidParameters,data,limit,params={"UserID"->"me"},rawdata,$facebookfeedpermissions={"user_posts"},rawprop},
        rawprop="RawGet"<>prop;

    invalidParameters = Select[Keys[args],!MemberQ[{MaxItems},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Facebook"]&/@invalidParameters;
        Throw[$Failed]
    ];

    limit = Lookup[args, MaxItems, $facebookfeedlimit];
    If[ IntegerQ[limit] && limit>0,
        AppendTo[params, "limit"-> ToString@limit]
        ,
        Message[ServiceExecute::nval,MaxItems,"Facebook"];
        Throw[$Failed]
    ];

    rawdata=OAuthClient`rawoauthdata[id,rawprop,Join[params,{"fields"->StringJoin[Riffle[facebookpostfields,","]],"limit"->"500"}]];          
     data=facebookimport[rawdata];
     Dataset[FFormatFeedInformation[StringDrop[prop,-1],#]&/@data["data"]]
]

facebookcookeddata[prop:("PostEventSeries"|"PostTimeline"),id_,args_]:=Block[
    {invalidParameters,data,user,limit,params={"UserID"->"me"},rawdata,$facebookfeedpermissions={"user_posts"},rawprop,dates,messages,ids},

    invalidParameters = Select[Keys[args],!MemberQ[{MaxItems},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Facebook"]&/@invalidParameters;
        Throw[$Failed]
    ];

    limit = Lookup[args, MaxItems, 500];
    If[ IntegerQ[limit] && limit>0,
        AppendTo[params, "limit"-> ToString@limit]
        ,
        Message[ServiceExecute::nval,MaxItems,"Facebook"];
        Throw[$Failed]
    ];

    rawdata=OAuthClient`rawoauthdata[id,"RawGetPosts",Join[params,{"fields"->"created_time,message,id"}]];          
    data=facebookimport[rawdata];
    data=Select[Lookup[data, "data", {}], !MissingQ[#["message"]]&];
    data[[All,1]]=readdate/@data[[All,1]];
    data=(Values/@data[[All,1;;2]]);
    Switch[prop,
        "PostEventSeries",
            If[ data === {},
                Missing["NotAvailable"],
                EventSeries[data]
            ],
        "PostTimeline",
            DateListPlot[MapThread[Tooltip[{#,1},#2]&, {data[[All,1]],data[[All,2]]}], Filling->Axis, FrameTicks->{None,{Automatic,Automatic}}, Joined->False]
    ]
]

facebookcookeddata["PostMessage",id_,args_]:=Module[
    {invalidParameters,rawdata,as,message,link,params={}},

    invalidParameters = Select[Keys[args],!MemberQ[{"Message","Link"},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Facebook"]&/@invalidParameters;
        Throw[$Failed]
    ];

    message = Lookup[args, "Message", Message[ServiceExecute::nparam, "Message"]; Throw[$Failed]];
    If[ StringQ[message],
        AppendTo[params, "message"-> message]
        ,
        Message[ServiceExecute::nval,"Message","Facebook"];
        Throw[$Failed]
    ];

    link = Lookup[args, "Link"];
    Which[
            MissingQ[link],
                Null,
            StringQ[link],
                AppendTo[params, "link"-> link],
            MatchQ[link, CloudObject[_] | URL[_]],
                AppendTo[params, "link"-> First@link],
            MatchQ[link, Hyperlink[__]],
                AppendTo[params, "link"-> Last@link],
            True,
                Message[ServiceExecute::nval,"Link","Facebook"];
                Throw[$Failed]
    ];

    rawdata=OAuthClient`rawoauthdata[id,"RawSendPost",params];
    as=facebookimport[rawdata];
    Lookup[params, "message"] /; AssociationQ[as]
]

facebookcookeddata["PostLink",id_,args_]:=Module[
    {invalidParameters,rawdata,as,link,params={}},

    invalidParameters = Select[Keys[args],!MemberQ[{"Link"},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Facebook"]&/@invalidParameters;
        Throw[$Failed]
    ];

    link = Lookup[args, "Link", Message[ServiceExecute::nparam, "Link"]; Throw[$Failed]];
    Which[
            StringQ[link],
                AppendTo[params, "link"-> link],
            MatchQ[link, CloudObject[_] | URL[_]],
                AppendTo[params, "link"-> First@link],
            MatchQ[link, Hyperlink[__]],
                AppendTo[params, "link"-> Last@link],
            True,
                Message[ServiceExecute::nval,"Link","Facebook"];
                Throw[$Failed]
    ];
    
    rawdata=OAuthClient`rawoauthdata[id,"RawSendPost",params];
    as=facebookimport[rawdata];
    Lookup[params, "link"] /; AssociationQ[as]
]

(******************* User Activity **********************)

facebookactivitydata[id_,args_]:=Block[
    {rawdata,as,limit,params={"UserID"->"me"},OAuthClient`$CacheResults=True},

    limit = Lookup[args, MaxItems, 50];
    If[ IntegerQ[limit] && limit>0,
        AppendTo[params, "limit"-> ToString@limit]
        ,
        Message[ServiceExecute::nval,MaxItems,"Facebook"];
        Throw[$Failed]
    ];

    rawdata=OAuthClient`rawoauthdata[id,"RawGetPosts",Join[params,
        {"fields"->"id,from,to,message,message_tags,picture,link,name,caption,description,source,properties,icon,actions,privacy,type,likes,place,story,story_tags,with_tags,comments,object_id,application,created_time,updated_time"}
    ]];
    as=facebookimport[rawdata];
    Replace[as["data"], asoc_Association :> Normal[asoc], {0, Infinity}]
]

facebookChartColor = "DarkRainbow";

facebookcookeddata["ActivityRecentHistory",id_,args_]:=Module[
    {invalidParameters,rawdata,times,data, date, gatherByAct,rules, days,bardata, items,
    daylength,dayposition,plength,ticks, res},

    invalidParameters = Select[Keys[args],!MemberQ[{MaxItems},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Facebook"]&/@invalidParameters;
        Throw[$Failed]
    ];

    rawdata=facebookactivitydata[id,args];
    If[ !MemberQ[rawdata,"updated_time", Infinity],
        Return[BarChart[{}]]
    ];
    times={"updated_time", "type"}/.rawdata;
    data ={readdate[#1], #2} & @@@ times;
    gatherByAct = GatherBy[data, Last];
    items = gatherByAct[[All, 1, 2]];
    date = Map[DateList,gatherByAct[[All, All, 1]],{2}];    
    rules = ToString[#1] -> #2 & @@@ Tally[#] & /@ date[[All, All, ;; 2]];
    days = Table[ToString[DateList[DateList[data[[-1, 1]]] + {0,i,0,0,0,0}][[ ;; 2]]],
              {i, 0, DateDifference[data[[-1, 1]], data[[1, 1]], "Month"][[1]]}];
    bardata = Transpose[days /. rules /. _String -> 0];
    days=ToExpression[days];
    (

         daylength = Length[days];
         dayposition = Select[FindDivisions[{1, daylength}, 7], IntegerQ[#] && (0 < # < daylength) &];
         plength = Length[dayposition];
         dayposition = Which[plength == 0, {1}, plength > 6, dayposition[[;; 6]], True, dayposition];
         ticks =
            Transpose[{dayposition, DateString[#, {"Month", "/", "Year"}] & /@ days[[dayposition]], ConstantArray[0, Length[dayposition]]}];
         res =
          BarChart[bardata,
             ChartLayout -> "Stacked",
             ChartStyle -> facebookChartColor,
             ChartLegends -> Placed[items, Below],
             FrameTicks -> {{Automatic, None}, {ticks, None}},
             Frame -> True, Axes -> None,
             AxesOrigin -> {0, 0},
             BarSpacing -> {0, 0},
             GridLines -> {None, Automatic},
             GridLinesStyle -> GrayLevel[.8],
             PlotRangeClipping -> True,
             PlotRangePadding -> {{Automatic, Automatic}, {None, Scaled[0.08]}}
          ];
          res /; (Head[res] =!= BarChart)
    ) /; (data =!= $Failed)
]

facebookcookeddata["ActivityTypes",id_,args_]:=Module[
    {invalidParameters,rawdata,types,tally,res},
    
    invalidParameters = Select[Keys[args],!MemberQ[{MaxItems},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Facebook"]&/@invalidParameters;
        Throw[$Failed]
    ];
    
    rawdata = facebookactivitydata[id,args];
    If[ rawdata =!= $Failed,
        If[ MatchQ[rawdata,_List] && Length[rawdata]==0, 
            Missing["NotAvailable"]
            ,
            types = "type" /. rawdata;
            tally = Tally[types /. "type" -> {}];
            tally = Transpose[tally];
            If[ ListQ[tally], 
                res = PieChart[tally[[2]], ChartLegends -> tally[[1]], ChartStyle->facebookChartColor];
                res,
                $Failed
            ]
        ]
        ,
        $Failed
    ]
        
]
  
facebookcookeddata["ActivityWeeklyDistribution",id_,args_]:=Module[
    {invalidParameters,rawdata,times,weeklydata,gatherByActWeekly,weeklyitems,dateweekly,rulesweekly,datelistweekly,
    dayrules,weeklyPart,bubbledata,chartelem,res},
    
    invalidParameters = Select[Keys[args],!MemberQ[{MaxItems},#]&];
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"Facebook"]&/@invalidParameters;
        Throw[$Failed]
    ];
    
    rawdata=facebookactivitydata[id,args];
    If[rawdata =!= $Failed,
    (
        If[MatchQ[rawdata,_List] && Length[rawdata]==0,
            Missing["NotAvailable"],
            (
                 times={"updated_time", "type"}/.rawdata;
                 weeklydata ={StringSplit@DateString[readdate[#1],{"Hour24", " ", "DayNameShort"}
                     (* Using the zero time zone gives backward compatibility with SocialMediaData *)
                     (*, TimeZone->0 *)], #2} & @@@ times;
                 gatherByActWeekly = GatherBy[weeklydata, Last];
                 weeklyitems = gatherByActWeekly[[All, 1, 2]];

                 dateweekly = gatherByActWeekly[[All, All, 1]];
                 rulesweekly = #1 -> #2 & @@@ Tally[#] & /@ dateweekly[[All, All, ;; 2]];
                 datelistweekly = Union[weeklydata[[All, 1]]];

                 dayrules = Thread[{"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"} -> Range[7]];
                 weeklyPart = datelistweekly /. rulesweekly /. {_String, _} -> 0;
                 bubbledata = Transpose[Join[Transpose[ToExpression[datelistweekly /. dayrules]], {Total[weeklyPart, {1}]}]];
                 (
                     chartelem = Table[PieChart[d,LabelingFunction->None, ChartStyle->facebookChartColor], {d, Transpose[weeklyPart]}];
                     res =BubbleChart[bubbledata,
                         ChartElements -> chartelem,
                         BubbleSizes -> {0.06, 0.14},
                         AspectRatio -> .4,
                        ChartLegends -> {None, weeklyitems},
                        FrameTicks -> {{List@@@Reverse[Rule @@@ dayrules, 2], None}, 
                            {List@@@Table[i -> DateString[{1, 1, 1, i}, {"Hour12Short", " ", "AMPMLowerCase"}], {i, 0, 21, 3}], None}}
                      ];

                     Legended[res, SwatchLegend[facebookChartColor, weeklyitems]] (*/; (Head[res] =!= BubbleChart)*)
                )
            )
        ]
    )
    ,
    $Failed]    
]


(******************* Wall Activity **********************)

facebookwalldata[id_,args_]:=Block[
    {rawdata,as,limit,params={"UserID"->"me"}},

    limit = Lookup[args, MaxItems, 50];
    If[ IntegerQ[limit] && 500>limit>0,
        AppendTo[params, "limit"-> ToString@limit]
        ,
        Message[ServiceExecute::nval,MaxItems,"Facebook"];
        Throw[$Failed]
    ];

    rawdata = OAuthClient`rawoauthdata[id,"RawGetFeeds",Join[params,{"limit"->"500","fields"->"id,message,type,likes.summary(1),story,picture,comments.summary(1),created_time"}]];
    as = facebookimport[rawdata];
    as["data"]
]


facebookcookeddata["WallMostCommentedPostData",id_,args_]:=Module[
    {wall,ncomments,max,pos},
    wall = facebookwalldata[id,args];
    If[ wall==={},Return[{}]];
    (
        ncomments=Replace[wall[[All, "comments", "summary", "total_count"]], Missing[_] -> 0];
        max=Max[ncomments];
        pos=Position[ncomments,max];
        Dataset[FFormatPost/@(Join[#,<|"CommentCount"->max|>]&/@Extract[wall,pos])]
    ) /; ( wall =!= $Failed )
]

facebookcookeddata["WallMostLikedPostData",id_,args_]:=Module[
    {wall,nlikes,max,pos},
    wall = facebookwalldata[id,args];
    If[ wall==={},Return[{}]];
    (
        nlikes=Replace[wall[[All, "likes", "summary", "total_count"]], Missing[_] -> 0];
        max=Max[nlikes];
        pos=Position[nlikes,max];
        Dataset[FFormatPost/@(Join[#,<|"LikeCount"->max|>]&/@Extract[wall,pos])]
    ) /; ( wall =!= $Failed )
]

facebookcookeddata["WallPostLength",id_,args_]:=Module[
    {wall,messagedata,char,gather,res},
    wall = facebookwalldata[id,args];
    If[ wall==={},Return[{}]];
    (
        messagedata = Lookup[wall, {"created_time", "message"}, ""];
        messagedata = {readdate[#1,;;2], #2} & @@@ messagedata;
        char = StringLength /@ messagedata[[All, 2]];
        gather = GatherBy[Transpose[{messagedata[[All, 1]], char}], First];
        res = Rule @@@ Transpose[{gather[[All, 1, 1]], Total[gather[[All, All, 2]], {2}]}];
        res /; ListQ[res]
    ) /; ( wall =!= $Failed )
]

facebookcookeddata["WallPostLengthTimeline",id_,args_]:=Module[
    {rawdata,messagedata,res},
    rawdata = facebookcookeddata["WallPostLength",id,args];
    If[ rawdata==={},Return[{}]];
    (
         messagedata = List@@@rawdata;
         res =
          DateListPlot[
            messagedata,
            Filling -> Bottom, FillingStyle -> Blue,
            GridLines -> {None, Automatic}, GridLinesStyle -> Opacity[.4],
            PlotRange -> {Automatic, {0, Automatic}},
            DateTicksFormat -> {"Month", "/", "Year"},
            FrameTicks -> {{Automatic, None}, {Automatic, None}}
          ];

         res /; (Head[res] =!= DateListPlot)

    ) /; (rawdata =!= $Failed)
]

facebookcookeddata["WallWordFrequencies",id_,args_]:=Module[
    {wall,message},
    wall = facebookwalldata[id,args];
    If[ wall==={},Return[{}]];
    (
     message = Lookup[wall, "message", ""];
     Rule@@@Reverse[SortBy[Tally[StringCases[StringJoin[Riffle[message, " "]], WordCharacter ..]], Last]]
    ) /; ( wall =!= $Failed )
]

facebookcookeddata[___]:=$Failed

(* Send Message *)
facebooksendmessage[id_,message_String]:=facebookcookeddata["PostMessage",id,"message"->message]

facebooksendmessage[___]:=$Failed
(******** Permission management **********)

facebookcookeddata["PermissionList",id_,args_]:= Block[
    {data},
    data=facebookimport@OAuthClient`rawoauthdata[id,"RawPermissions"];
    If[ KeyExistsQ[data,"data"],
        Flatten[checkFacebookPermissions[data["data"]]]
        ,
        {}
    ]
]
 
checkFacebookPermissions[l_List]:= Map[#["permission"] &, Select[l, (#["status"] === "granted") &]]

(** ---> THIS FUNCTION IS USED IN THE OAuth FRAMEWORK  **)
facebookcheckpermissions[id_]:=With[{res=facebookimport[OAuthClient`rawoauthdata[id,"RawPermissions"]]},
    If[ KeyExistsQ[res,"data"],
        checkFacebookPermissions[res["data"]],
        {}
    ]
]
(** THIS FUNCTION IS USED IN THE OAuth FRAMEWORK  <--- **)

$FacebookPermissionsURL="https://www.wolframcloud.com/objects/user-00e58bd3-2dfd-45b3-b80b-d281d360703a/facebookkey"

getfacebookkey[id_]:=ToExpression[URLFetch[$FacebookPermissionsURL,"Parameters"->If[TrueQ[ServiceConnections`Private`useChannel[id]],{"ChannelBrokerQ"->"True"},{}],"VerifyPeer"->False]]

requestedFacebookPermissions[_]:={};

(** ---> THIS FUNCTION IS USED IN THE OAuth FRAMEWORK  **)
facebookaddpermissions[id_,permissions_]:=facebookaddpermissions0[id,Complement[permissions,requestedFacebookPermissions[id]]]
(** THIS FUNCTION IS USED IN THE OAuth FRAMEWORK  <--- **)

facebookaddpermissions0[id_,{}]:=Null

facebookaddpermissions0[id_,permissions_]:=Module[{
    url, key=getfacebookkey[id], temp
    },
    requestedFacebookPermissions[id]=Join[requestedFacebookPermissions[id],permissions];
    If[!StringQ[key],Throw[$Failed]];
    url="https://www.facebook.com/dialog/oauth?client_id="<>
            (key)<>"&redirect_uri="<>
            OAuthClient`Private`createRedirect[("RedirectURI"/.Once[OAuthClient`Private`oauthservicedata["Facebook"]]),id,"Facebook"]<>"&auth_type=rerequest"<>"&scope="<>
            StringJoin[Riffle[permissions,","]];
            
    OAuthClient`oauthChannelVerify[{url, Identity, temp}, {"Facebook",id}];
    Message[ServiceExecute::addperm];
    $Failed
]

(****** utilities ************)
formatuser[]:="me"
formatuser[Automatic]:="me"
formatuser[str_String]:=str
formatuser[x_]:=ToString[x]
formatuser[__]:=Throw[$Failed]

formatpage[str_String]:=str
formatpage[x_]:=ToString[x]
formatpage[__]:=Throw[$Failed]

readdate[date_, part_:All]:= TimeZoneConvert[
    DateObject[DateList[{StringDrop[date, -5], 
        {"Year", "-", "Month", "-", "Day", "T", "Hour", ":", "Minute", ":", "Second"}}][[part]], TimeZone -> 0], $TimeZone]

iImportImage[_Missing] := Missing["NotAvailable"]
iImportImage[""] := Missing["NotAvailable"]

iImportImage[url_String] :=
    Block[{res},
        res = Quiet[Import[url]];
        If[ ImageQ[res],
            res,
            If[ StringMatchQ[url, __ ~~ "_" ~~ _ ~~ ".jpg"],
                res = StringReplacePart[url, "q", {-5, -5}];
                res = Quiet[Import[res]];
                If[ ImageQ[res], res, Missing["NotAvailable"]]
                ,
                Missing["NotAvailable"]
            ]
        ]
    ];
 
fbicon=Image[RawArray["Byte", {{{59, 87, 157, 1}, {59, 87, 157, 115}, {59, 87, 157, 243}, {59, 87, 157, 255}, 
  {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, 
  {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, 
  {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, 
  {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, 
  {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 243}, {59, 87, 157, 115}, {59, 87, 157, 0}}, 
  {{59, 87, 157, 115}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, 
  {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, 
  {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, 
  {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, 
  {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, 
  {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 115}}, {{59, 87, 157, 
  244}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 244}}, {{59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}}, {{59, 87, 157, 255}, {59, 87, 157, 255}, {59, 
  87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 
  87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 
  87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 
  87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 
  87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 
  87, 157, 255}, {59, 87, 157, 255}}, {{59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, 
  {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, 
  {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, 
  {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {70, 96, 163, 255}, 
  {151, 166, 203, 255}, {216, 222, 235, 255}, {248, 249, 251, 255}, {255, 255, 255, 255}, {246, 247, 
  250, 255}, {230, 234, 243, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 
  87, 157, 255}, {59, 87, 157, 255}}, {{59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, 
  {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, 
  {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, 
  {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {95, 119, 176, 255}, {230, 234, 243, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, 
  {59, 87, 157, 255}, {59, 87, 157, 255}}, {{59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {71, 97, 163, 255}, {236, 239, 246, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}}, {{59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {160, 173, 208, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}}, {{59, 87, 157, 255}, {59, 87, 157, 255}, {59, 
  87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 
  87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 
  87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {223, 228, 239, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {174, 186, 216, 255}, {71, 97, 163, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}}, {{59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {252, 252, 253, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {77, 103, 167, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, 
  {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, 
  {59, 87, 157, 255}, {59, 87, 157, 255}}, {{59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {60, 88, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 
  87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 
  87, 157, 255}, {59, 87, 157, 255}}, {{59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, 
  {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, 
  {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, 
  {59, 87, 157, 255}, {59, 87, 157, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}}, {{59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 
  87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 
  87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 
  87, 157, 255}, {59, 87, 157, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}}, {{59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {250, 251, 253, 255}, {62, 90, 159, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, 
  {59, 87, 157, 255}, {59, 87, 157, 255}}, {{59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, 
  {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 
  255, 255}, {255, 255, 255, 255}, {220, 226, 238, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 
  87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}}, {{59, 87, 157, 255}, {59, 87, 157, 255}, 
  {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, 
  {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {188, 197, 222, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}}, {{59, 87, 157, 255}, {59, 
  87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 
  87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 
  255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {156, 170, 206, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}}, {{59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}}, {{59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}}, {{59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}}, {{59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}}, {{59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}}, {{59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}}, {{59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}}, {{59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}}, {{59, 87, 
  157, 244}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 244}}, {{59, 87, 
  157, 123}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 123}}, {{59, 87, 
  157, 1}, {59, 87, 157, 115}, {59, 87, 157, 244}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 
  157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {255, 
  255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {255, 255, 255, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 255}, {59, 87, 157, 
  255}, {59, 87, 157, 255}, {59, 87, 157, 244}, {59, 87, 157, 115}, {59, 87, 157, 0}}}], "Byte", 
 ColorSpace -> "RGB", Interleaving -> True];
                   
End[] (* End Private Context *)
                   
End[]

SetAttributes[{},{ReadProtected, Protected}];

(* Return five functions to define oauthservicedata, oauthcookeddata, oauthsendmessage, checkpermission, add permissions  *)
{FacebookOAuth`Private`facebookdata,FacebookOAuth`Private`facebookcookeddata,
    FacebookOAuth`Private`facebooksendmessage,FacebookOAuth`Private`facebookcheckpermissions,
    FacebookOAuth`Private`facebookaddpermissions}
