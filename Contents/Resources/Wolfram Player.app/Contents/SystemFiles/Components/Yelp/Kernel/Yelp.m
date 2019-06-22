Begin["Yelp`"]

Get["YelpFunctions.m"]

Begin["`Private`"]

(******************************* Yelp *************************************)

(* Authentication information *)

yelpdata[]:={
        "ServiceName"       -> "Yelp",
         "URLFetchFun"        :> (With[{params=Lookup[{##2},"Parameters",{}]},
                URLRead[
                    HTTPRequest[#1, <|
                        "Query" -> Normal@KeyDrop[params,"apikey"],
                        "Headers" -> {"Authorization" -> "Bearer " <> Lookup[params,"apikey",""]}|>],
                    {"StatusCode", "Body"}
                    ]
                ]&)
            ,
        "ClientInfo"        :> OAuthDialogDump`Private`MultipleKeyDialog["Yelp",{"API Key" -> "apikey"},"https://www.yelp.com/developers/v3/manage_app","https://www.yelp.com/developers/api_terms"],
         "Gets"                -> {"BusinessList","BusinessDataset","BusinessInformation","Categories"},
         "Posts"                -> {},
         "RawGets"            -> {"RawSearch","RawBusiness","RawPhoneSearch"},
         "RawPosts"            -> {},        
         "Information"        -> "A service for exchanging data with a Yelp"
}

(**** Raw Requests ****)

yelpdata["RawSearch"] := {
        "URL"                -> "https://api.yelp.com/v3/businesses/search",
        "HTTPSMethod"        -> "GET",
        "Parameters"        -> {"term","location","latitude","longitude","radius","categories","locale","limit","offset","sort_by",
                                "price","open_now","open_at","attributes"},
        "RequiredParameters"-> {},
        "ResultsFunction"    -> yelpimport
    }

yelpdata["RawBusiness"] := {
        "URL"                -> (ToString@StringForm["https://api.yelp.com/v3/businesses/`1`", #]&),
        "HTTPSMethod"        -> "GET",
        "Parameters"        -> {"locale"},
        "PathParameters"    -> {"id"},
        "RequiredParameters"-> {"id"},
        "ResultsFunction"    -> yelpimport
    }
    
yelpdata["RawPhoneSearch"] := {
        "URL"                -> "https://api.yelp.com/v3/businesses/search/phone",
        "HTTPSMethod"        -> "GET",
        "Parameters"        -> {"phone"},
        "RequiredParameters"-> {"phone"},
        "ResultsFunction"    -> yelpimport
    }
  
yelpdata[___]:=$Failed   
   
(**** Cooked Requests ****)

yelpcookeddata[prop:("BusinessList"|"BusinessDataset"), id_, args_] := Module[{invalidParameters,location,params={},latitude,longitude,coordinates,
                                                                    point,radius,defaultRadius="40000",sort,sortVal,query,limit,maxPerPage=50,startIndex,
                                                                    calls,residual,progress=0,data,rawdata,totalResults,items={},result,cFilter,
                                                                    phone,argsCopy,interpreterQ=False,cFilterTmp},

    invalidParameters = Select[Keys[args],!MemberQ[{"Location","Radius","MaxItems",MaxItems,"StartIndex","SortBy",
                                                    "Query","Phone","Categories","InterpretEntities"},#]&]; 
    
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,prop]&/@invalidParameters;
        Throw[$Failed]
    ];
    
    argsCopy = ReplaceAll[args,Rule["MaxItems",m_]:>Rule[MaxItems,m]];
    
    If[ KeyExistsQ[args, "Phone"],
        phone = Lookup[args, "Phone"];
        AppendTo[params,"phone" -> phone];
        rawdata = KeyClient`rawkeydata[id,"RawPhoneSearch",params];
        data = yelpimport[rawdata];
        result = data["businesses"] (* Phone Search result *)
        ,
    (* geo search *)
    If[ KeyExistsQ[args,"Location"],
            location = Lookup[args, "Location"];
            
            (* this handles the case where the user gives a GeoPosition representation for more than one point e.g. polygons *)
            If[ MatchQ[Head[location], Polygon] && Quiet[MatchQ[Head[QuantityMagnitude[Latitude[location[[1]]], "AngularDegrees"]], List]],
                location=GeoBoundingBox[location]];
                
            Switch[ location,
                Entity["ZIPCode",_], (* US zip code *)
                    params = Append[params,"location"->EntityValue[location, "Name"]]
                ,
                _GeoPosition, (* radial search *)
                    latitude = QuantityMagnitude[Latitude[location], "AngularDegrees"] //ToString;
                    longitude = QuantityMagnitude[Longitude[location], "AngularDegrees"] //ToString;
                
                    AppendTo[params,"latitude"->latitude];
                    AppendTo[params,"longitude"->longitude];
                ,
                _Entity,
                    Switch[ EntityTypeName[location],
                        "Country",
                            latitude = QuantityMagnitude[Latitude[location], "AngularDegrees"] //ToString;
                            longitude = QuantityMagnitude[Longitude[location], "AngularDegrees"] //ToString;
                            
                            AppendTo[params,"latitude"->latitude];
                            AppendTo[params,"longitude"->longitude];
                        ,
                        "City",
                            latitude = QuantityMagnitude[Latitude[location], "AngularDegrees"] //ToString;
                            longitude = QuantityMagnitude[Longitude[location], "AngularDegrees"] //ToString;
                            
                            AppendTo[params,"latitude"->latitude];
                            AppendTo[params,"longitude"->longitude];
                        ,
                        _,
                            coordinates = LatitudeLongitude[location];
                            If[ MatchQ[Head[coordinates],List],
                                latitude = coordinates[[1,1]] // ToString;
                                longitude = coordinates[[2,1]] // ToString;
                                
                                AppendTo[params,"latitude"->latitude];
                                AppendTo[params,"longitude"->longitude];
                                ,
                                Message[ServiceExecute::nval,"Location","Yelp"];    
                                Throw[$Failed]
                            ]
                    ]
                ,
                _GeoDisk,
                    Switch[ location,
                        GeoDisk[],
                            point = $GeoLocation;
                            radius = defaultRadius;
                        ,    
                        GeoDisk[_],
                            point = location[[1]];
                            radius = defaultRadius;
                        ,
                        GeoDisk[_,_,___],
                            point = location[[1]];
                            radius = location[[2]];
                            radius = QuantityMagnitude[radius, "Meters"];
                            radius = ToString[Round[radius]]
                    ];
                
                    latitude = QuantityMagnitude[Latitude[point], "AngularDegrees"] //ToString;
                    longitude = QuantityMagnitude[Longitude[point], "AngularDegrees"] //ToString;

                    AppendTo[params,"latitude"->latitude];
                    AppendTo[params,"longitude"->longitude];
                    AppendTo[params,"radius" -> radius];
                ,
                _, (* unrecognized Location specification *)
                    Message[ServiceExecute::nval,"Location","Yelp"];    
                    Throw[$Failed]
            ]
        ,
        Message[ServiceExecute::nparam,"Location"];
        Throw[$Failed]       
    ];
    
    If[ KeyExistsQ[args,"Radius"],
        radius = Lookup[args,"Radius"];
        radius = QuantityMagnitude[radius, "Meters"];
        radius = ToString[Round[radius]];    
        AppendTo[params, "radius"->radius];        
    ];
    
    If[ KeyExistsQ[args,"SortBy"],
        Lookup[args,"SortBy"];
        If[ StringQ[sort],
            Switch[sort,
                "BestMatch",
                sortVal = "best_match",
                "Distance",
                sortVal = "distance",
                "Rating",
                sortVal = "rating",
                _,
                    Message[ServiceExecute::nval,"SortBy","Yelp"];    
                    Throw[$Failed]
            ];            
        ];        
        AppendTo[params, "sort_by"->sortVal];        
    ];
    
    If[ KeyExistsQ[args,"Query"],
        query = Lookup[args,"Query"];
        AppendTo[params,"term" -> query]            
    ];
    
    If[ KeyExistsQ[args,"InterpretEntities"],
        interpreterQ = Lookup[args,"InterpretEntities"];
        If[ !BooleanQ[interpreterQ],
            Message[ServiceExecute::nval,"InterpretEntities","Yelp"];    
            Throw[$Failed]
        ]
    ];
    
    If[ KeyExistsQ[args,"Categories"],
            cFilter = Lookup[args,"Categories"];
            Switch[ Head[cFilter],
                String,
                    cFilter = {cFilter},
                List,
                    None,
                _,
                    Message[ServiceExecute::nval,"Categories","Yelp"];    
                    Throw[$Failed]
            ];    
            cFilterTmp = {};
            If[ isAlias[#],
                cFilterTmp = Append[cFilterTmp,#],
                cFilterTmp = Join[cFilterTmp,findAlias[#]]
            ]&/@cFilter;
                
            cFilterTmp = StringJoin[StringRiffle[cFilterTmp, ","]];
            params = Append[params,"categories" -> cFilterTmp];        
    ];
    
    If[ KeyExistsQ[argsCopy,MaxItems],
        limit = Lookup[argsCopy,MaxItems];
        If[ !IntegerQ[limit],
            Message[ServiceExecute::nval,"MaxItems","Yelp"];
            Throw[$Failed]
        ];                        
        ,
        limit = maxPerPage;
    ];
    
    If[ KeyExistsQ[args,"StartIndex"],
        startIndex = Lookup[args,"StartIndex"];
        If[ Or[!IntegerQ[startIndex], limit + startIndex > 1000],
            Message[ServiceExecute::nval,"StartIndex","Yelp"];
            Throw[$Failed]
        ];
        ,
        startIndex = 0  
    ];
    
    calls = Quotient[limit, maxPerPage, 1];    
    residual = Mod[limit, maxPerPage, 1];
    
    params = Join[params,{"limit"->ToString[maxPerPage], "offset"->ToString[startIndex]}];
    
    (* this prints the progress indicator bar *)
    PrintTemporary[ProgressIndicator[Dynamic[progress], {0, calls + 1}]];

    result = Catch[    
        If[ calls > 0,
        (
            (
                params = ReplaceAll[params, Rule["offset",_] -> Rule["offset",ToString[startIndex+#*maxPerPage]]];
                rawdata = KeyClient`rawkeydata[id,"RawSearch",params];
                data = yelpimport[rawdata];
                totalResults = data["total"];
                data = data["businesses"];
                If[ totalResults > (startIndex + (#+1)*maxPerPage),
                    items = Join[items, data];
                    progress = progress + 1
                    ,
                    items = Join[items, data];
                    progress = calls + 1;
                    Throw[items]
                ];
            )& /@ Range[0,calls-1];     
            
        )];
        (* There's always a residual with Mod[limit, max, 1 ]*)
        (
            params = ReplaceAll[params,Rule["offset",_] -> Rule["offset",ToString[startIndex+calls*maxPerPage]]];
            params = ReplaceAll[params,Rule["limit",_] -> Rule["limit",ToString[residual]]];
            rawdata = KeyClient`rawkeydata[id,"RawSearch",params];
            data = yelpimport[rawdata];
            totalResults = data["total"];
            data = data["businesses"];
            If[ totalResults > 0,
                progress = calls + 1;
                Take[Join[items, data], UpTo[limit]],
                progress = calls + 1;
                items = {}
            ]
        )
    ];
    
    result = YFormatBusiness[#, interpreterQ]& /@ result
    ];
    
    If[ prop=="BusinessList",
        result,
        Dataset[result]
    ]    
]

yelpcookeddata["BusinessInformation", id_, args_] := Module[{rawdata,invalidParameters,bId,result={},interpreterQ=False,showQ=False},

    invalidParameters = Select[Keys[args],!MemberQ[{"ID","InterpretEntities","ShowThumbnails"},#]&];
    
    If[ Length[invalidParameters]>0,
        Message[ServiceObject::noget,#,"BusinessInformation"]&/@invalidParameters;
        Throw[$Failed]
    ];    
        
    If[ KeyExistsQ[args,"InterpretEntities"],
        interpreterQ = Lookup[args,"InterpretEntities"];
        If[ !BooleanQ[interpreterQ],
            Message[ServiceExecute::nval,"InterpretEntities","Yelp"];    
            Throw[$Failed]
        ]
    ];
    
    If[ KeyExistsQ[args,"ShowThumbnails"],
        showQ = Lookup[args,"ShowThumbnails"];
        If[ !BooleanQ[showQ],
            Message[ServiceExecute::nval,"ShowThumbnails","Yelp"];    
            Throw[$Failed]
        ]
    ];
    
    If[ KeyExistsQ[args,"ID"],
        bId = Lookup[args,"ID"];
        If[ StringQ[bId], bId = {bId}];
        If[ !MatchQ[bId, {__?StringQ}],
            Message[ServiceExecute::nval,"ID","Yelp"];    
            Throw[$Failed]
        ]
        ,
        Message[ServiceExecute::nparam,"ID"];            
        Throw[$Failed]
    ];

    (
        rawdata = KeyClient`rawkeydata[id,"RawBusiness",{"id"->ToString[#]}];
        result = Join[result,{yelpimport[rawdata]}];
    ) &/@ bId;

    result = YFormatBusiness[#, interpreterQ, showQ]& /@ result;

    If[ Length[result] == 1,
        Dataset[result[[1]]],
        Dataset[result]
    ]
]

yelpcookeddata["Categories", id_, args_] := Module[{invalidParameters,jsondata,filter,parent,result},
        invalidParameters = Select[Keys[args],!MemberQ[{"Query","Parent"},#]&]; 
    
        If[Length[invalidParameters]>0,
        (
            Message[ServiceObject::noget,#,"Categories"]&/@invalidParameters;
            Throw[$Failed]
        )];    
    
        jsondata = getCategories[];
        
        If[KeyExistsQ[args,"Query"],
            filter = "Query" /. args;
            If[!StringQ[filter],
            (    
                Message[ServiceExecute::nval,"Query","Yelp"];
                Throw[$Failed]
            )];    
            (* Filter categories using query term *)
            jsondata = Select[jsondata, StringContainsQ[ToLowerCase["Title"/.#],ToLowerCase[filter]]&];
        ];
        
        If[KeyExistsQ[args,"Parent"],
            parent = "Parent" /. args;
            If[!StringQ[parent],
            (    
                Message[ServiceExecute::nval,"Parent","Yelp"];
                Throw[$Failed]
            )];    
            (* Filter categories by parent *)
            jsondata = Select[jsondata, MemberQ["Parents"/.#,ToLowerCase[parent]]&];
        ];
        
        jsondata = ReplaceAll[jsondata,Rule["Alias",a_]:>Rule["CategoryAlias",a]];
        jsondata = ReplaceAll[jsondata,Rule["Title",a_]:>Rule["CategoryName",a]];
        
        result = Association/@jsondata;
        KeyTake[result, Union[Flatten[Keys[result]]]]
]

yelpcookeddata[___]:=$Failed

yelpsendmessage[___]:=$Failed

yelpimport[rawdata_] := Module[{status,data},
    status = rawdata[[1]];
    data = Developer`ReadRawJSONString[rawdata[[2]]];
    Which[
        status === 429,
            Message[ServiceExecute::serrormsg, "You have reached your daily rate limit for this client."];
            Throw[$Failed],
        status === 200,
            data,
        KeyExistsQ[data, "error"],
            Message[ServiceExecute::serrormsg, data["error"]["description"]];
            Throw[$Failed],
        True,
            Message[ServiceExecute::serror]
    ]
]

End[]

End[]

SetAttributes[{},{ReadProtected, Protected}];

(* Return three functions to define oauthservicedata, oauthcookeddata, oauthsendmessage  *)
{Yelp`Private`yelpdata,Yelp`Private`yelpcookeddata,Yelp`Private`yelpsendmessage}
