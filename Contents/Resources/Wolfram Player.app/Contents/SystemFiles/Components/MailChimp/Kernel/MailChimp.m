Begin["MailChimpAPI`"]

Begin["`Private`"]

(******************************* MailChimp *************************************)

(* Authentication information *)
mailchimpdata[]:={
        "ServiceName"		-> "MailChimp", 
        "URLFetchFun"		:> (Block[ {dc,url,params = Lookup[{##2},"Parameters",{}]},
                                     dc = Lookup[params,"apikey",Throw[$Failed]];
                                     url = StringCases[dc,"-"~~x__->x][[1]]<>"."<>#1;
                                     URLFetch[URLBuild[Association["Scheme"->"https","Domain"->url]],
                                         "ContentData",Sequence@@FilterRules[{##2},Except["Parameters"]],"Parameters" -> params]]
                                &),
        "ClientInfo"		:> (*OAuthDialogDump`Private`KeyDialog["MailChimp"]*)
        						OAuthDialogDump`Private`MultipleKeyDialog["MailChimp",{"API Key"->"apikey"},
                                        "https://admin.mailchimp.com/account/api/","http://mailchimp.com/legal/terms/"],
         "Gets"				-> {"ListAbuse","ListActivity","ListClients","ListMonthlyGrowth","Lists","ListLocations","Campaigns","CampaignSummary","CampaignSummaryTimeSeries","ListMembers","CampaignAbuse","CampaignClicks","CampaignLocations","CampaignRecipients","CampaignOpenedBy","CampaignNotOpenedBy","CampaignUnsubscriptions"},
         "Posts"			-> {},
         "RawGets"			-> {"RawLists","RawListMembers","RawListAbuseReports","RawListActivity","RawListClients","RawListGrowthHistory","RawListLocations","RawCampaigns","RawReportAbuse","RawReportClicks","RawReportGeoOpens","RawReportSummary","RawReportSentTo","RawReportOpened","RawReportNotOpened","RawReportUnsubscribes","RawTemplates"},
         "RawPosts"			-> {"RawAddSubscriberToList"},
         "Information"		-> "Access MailChimp data using Wolfram Language"
}

mailchimpimport[rawdata_] :=ImportString[FromCharacterCode[rawdata, "UTF-8"], "RawJSON"]

(* Raw *)
(*List-Related*)
mailchimpdata["RawLists"] :=
    {
    "URL"                -> URLBuild[{"api.mailchimp.com","2.0","lists","list.json"}],
    "HTTPSMethod"        -> "GET",
    "Parameters"        -> {"filters[list_id]","filters[list_name]","filters[from_name]","filters[from_email]","filters[from_subject]","filters[created_before]","filters[created_after]","filters[exact]","start","limit","sort_field","sort_dir"},
    "RequiredParameters"-> {},
    "ResultsFunction"    -> mailchimpimport
    } 

mailchimpdata["RawListMembers"] :=
    {
    "URL"               -> URLBuild[{"api.mailchimp.com","2.0","lists","members.json"}],
    "HTTPSMethod"       -> "GET",
    "Parameters"        -> {"id","status","opts[start]","opts[limit]","opts[sort_field]","opts[sort_dir]","opts[segment]"},
    "RequiredParameters"-> {"id"},
    "ResultsFunction"   -> mailchimpimport
    } 
 
 mailchimpdata["RawListAbuseReports"] :=
     {
     "URL"               -> URLBuild[{"api.mailchimp.com","2.0","lists","abuse-reports.json"}],
     "HTTPSMethod"       -> "GET",
     "Parameters"        -> {"id","start","limit","since"},
     "RequiredParameters"-> {"id"},
     "ResultsFunction"   -> mailchimpimport
     }
 
 
 mailchimpdata["RawListActivity"] :=
     {
     "URL"               -> URLBuild[{"api.mailchimp.com","2.0","lists","activity.json"}],
     "HTTPSMethod"       -> "GET",
     "Parameters"        -> {"id"},
     "RequiredParameters"-> {"id"},
     "ResultsFunction"   -> mailchimpimport
     }
 
 mailchimpdata["RawListClients"] :=
     {
     "URL"               -> URLBuild[{"api.mailchimp.com","2.0","lists","clients.json"}],
     "HTTPSMethod"       -> "GET",
     "Parameters"        -> {"id"},
     "RequiredParameters"-> {"id"},
     "ResultsFunction"   -> mailchimpimport
     }
 
 mailchimpdata["RawListGrowthHistory"] :=
     {
     "URL"               -> URLBuild[{"api.mailchimp.com","2.0","lists","growth-history.json"}],
     "HTTPSMethod"       -> "GET",
     "Parameters"        -> {"id"},
     "RequiredParameters"-> {"id"},
     "ResultsFunction"   -> mailchimpimport
     }
 
mailchimpdata["RawListLocations"] :=
    {
    "URL"               -> URLBuild[{"api.mailchimp.com","2.0","lists","locations.json"}],
    "HTTPSMethod"       -> "GET",
    "Parameters"        -> {"id"},
    "RequiredParameters"-> {"id"},
    "ResultsFunction"   -> mailchimpimport
    }


(* mailchimpdata["RawAddSubscriberToList"] = {
        "URL"               -> URLBuild[{"api.mailchimp.com","2.0","lists","subscribe.json"}],
        "Parameters"          -> {"id","email[email]"},
        "HTTPSMethod"       -> "POST",
        "ResultsFunction"   -> mailchimpimport
    }*)
 
(*Campaign-Related*)

mailchimpdata["RawCampaigns"] :=
    {
    "URL"               -> URLBuild[{"api.mailchimp.com","2.0","campaigns","list.json"}],
    "HTTPSMethod"       -> "GET",
    "Parameters"        -> {"filters[campaign_id]","filters[parent_id]","filters[list_id]","filters[folder_id]","filters[template_id]","filters[status]","filters[type]","filters[from_name]","filters[from_email]","filters[title]","filters[subject]","filters[sendtime_start]","filters[sendtime_end]","filters[uses_segment]","filters[exact]","start","limit","sort_field","sort_dir"},
    "RequiredParameters"-> {},
    "ResultsFunction"   -> mailchimpimport
    } 

(*Report-Related*)

mailchimpdata["RawReportAbuse"] :=
    {
    "URL"               -> URLBuild[{"api.mailchimp.com","2.0","reports","abuse.json"}],
    "HTTPSMethod"       -> "GET",
    "Parameters"        -> {"cid","opts[start]","opts[limit]","opts[since]"},
    "RequiredParameters"-> {"cid"},
    "ResultsFunction"   -> mailchimpimport
    }

mailchimpdata["RawReportClicks"] :=
    {
    "URL"               -> URLBuild[{"api.mailchimp.com","2.0","reports","clicks.json"}],
    "HTTPSMethod"       -> "GET",
    "Parameters"        -> {"cid"},
    "RequiredParameters"-> {"cid"},
    "ResultsFunction"   -> mailchimpimport
    }

mailchimpdata["RawReportGeoOpens"] :=
    {
    "URL"               -> URLBuild[{"api.mailchimp.com","2.0","reports","geo-opens.json"}],
    "HTTPSMethod"       -> "GET",
    "Parameters"        -> {"cid"},
    "RequiredParameters"-> {"cid"},
    "ResultsFunction"   -> mailchimpimport
    }

mailchimpdata["RawReportSummary"] :=
    {
    "URL"               -> URLBuild[{"api.mailchimp.com","2.0","reports","summary.json"}],
    "HTTPSMethod"       -> "GET",
    "Parameters"        -> {"cid"},
    "RequiredParameters"-> {"cid"},
    "ResultsFunction"   -> mailchimpimport
    }

mailchimpdata["RawReportSentTo"] :=
    {
    "URL"               -> URLBuild[{"api.mailchimp.com","2.0","reports","sent-to.json"}],
    "HTTPSMethod"       -> "GET",
    "Parameters"        -> {"cid","opts[status]","opts[start]","opts[limit]"},
    "RequiredParameters"-> {"cid"},
    "ResultsFunction"   -> mailchimpimport
    }

mailchimpdata["RawReportOpened"] :=
    {
    "URL"               -> URLBuild[{"api.mailchimp.com","2.0","reports","opened.json"}],
    "HTTPSMethod"       -> "GET",
    "Parameters"        -> {"cid","opts[start]","opts[limit]","opts[sort_field]","opts[sort_dir]"},
    "RequiredParameters"-> {"cid"},
    "ResultsFunction"   -> mailchimpimport
    }

mailchimpdata["RawReportNotOpened"] :=
    {
    "URL"               -> URLBuild[{"api.mailchimp.com","2.0","reports","not-opened.json"}],
    "HTTPSMethod"       -> "GET",
    "Parameters"        -> {"cid","opts[start]","opts[limit]"},
    "RequiredParameters"-> {"cid"},
    "ResultsFunction"   -> mailchimpimport
    }

mailchimpdata["RawReportUnsubscribes"] :=
    {
    "URL"               -> URLBuild[{"api.mailchimp.com","2.0","reports","unsubscribes.json"}],
    "HTTPSMethod"       -> "GET",
    "Parameters"        -> {"cid","opts[start]","opts[limit]"},
    "RequiredParameters"-> {"cid"},
    "ResultsFunction"   -> mailchimpimport
    }


(*mailchimpdata["RawTemplates"] :=
    {
    "URL"               -> URLBuild[{"api.mailchimp.com","2.0","templates","list.json"}],
    "HTTPSMethod"       -> "GET",
    "Parameters"        -> {"types[user]","types[gallery]","types[base]","filters[category]","filters[folder_id]","filters[include_inactive]","filters[inactive_only]","filters[include_drag_and_drop]"},
    "RequiredParameters"-> {},
    "ResultsFunction"   -> mailchimpimport
    } *)

(*Cooked*)
camelCase[text_] := Module[{split, partial}, (
    split = StringSplit[text, {" ","_","-"}];
    partial = Prepend[Rest[Characters[#]], ToUpperCase[Characters[#][[1]]]] & /@ split;
    StringJoin[partial]
    )]

mailchimpcookeddata["ListAbuse",id_,args_] :=Block[ {data,invalidParameters,withCamelTitles,listid,start = Null,limit = Null,since = Null,newparams},
	newparams=args/.{MaxItems:>"MaxItems"};
	invalidParameters = Select[Keys[newparams],!MemberQ[{"ListID","StartIndex","MaxItems","StartDate"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"MailChimp"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[ KeyExistsQ[newparams,"ListID"],
   	(
   		listid = ToString["ListID" /. newparams];                        
	),
	(
		Message[ServiceExecute::nparam,"ListID","MailChimp"];
		Throw[$Failed]
	)];
  	If[ KeyExistsQ[newparams,"StartIndex"],
 	(
    	If[!(IntegerQ["StartIndex"/.newparams]&&("StartIndex"/.newparams)>0),
		(	
			Message[ServiceExecute::nval,"StartIndex","MailChimp"];
			Throw[$Failed]
		)];
    	start = ToString[("StartIndex" /. newparams)-1];                        
  	)];
  	If[ KeyExistsQ[newparams,"MaxItems"],
   	(
   		If[!(IntegerQ["MaxItems"/.newparams]&&("MaxItems"/.newparams)>=0&&("MaxItems"/.newparams)<=1000),
		(	
			Message[ServiceExecute::nval,"MaxItems","MailChimp"];
			Throw[$Failed]
		)];
     	limit = ToString["MaxItems" /. newparams];                        
  	)];
	If[ KeyExistsQ[newparams,"StartDate"],
	(
		If[!(StringQ["StartDate"/.newparams]||MatchQ["StartDate"/.newparams,DateObject[__]]),
		(	
			Message[ServiceExecute::nval,"StartDate","MailChimp"];
			Throw[$Failed]
		)];
		since = DateObject[("StartDate" /. newparams)];
		If[MatchQ[since,DateObject[__String]],
		(	
			Message[ServiceExecute::nval,"StartDate","MailChimp"];
			Throw[$Failed]
		)];
        since = DateString[TimeZoneConvert[since,0], {"Year", "-", "Month", "-", "Day", " ", "Time"}]             
	)];
	data = FixedPoint[Normal,ServiceExecute["MailChimp", "RawListAbuseReports", DeleteCases[{"id" -> listid ,"start" -> start, "limit"-> limit,"since"-> since}, _ -> Null]]];
  	If[("status"/.data)=="error",
   	(
      	Message[ServiceExecute::serrormsg,("error"/.data)];
       	Throw[$Failed]
 	)];
  	withCamelTitles=Replace[("data" /. data), Rule[a_, b_] :> Rule[camelCase[a], b], Infinity]
   	/. {(y : "Date" -> x_) :> (y -> If[MatchQ[x, _String], DateObject[x, TimeZone -> 0], x]),
   		"CampaignId"->"CampaignID"
   	 	};
   	Dataset[Association @@@ withCamelTitles]
]

mailchimpcookeddata["ListActivity",id_,args_] :=Block[ {data,invalidParameters,withCamelTitles,listid,newparams},
    newparams=args;
    invalidParameters = Select[Keys[newparams],!MemberQ[{"ListID"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"MailChimp"]&/@invalidParameters;
		Throw[$Failed]
	)];
    If[ KeyExistsQ[newparams,"ListID"],
    (
     	listid = ToString["ListID" /. newparams];                        
     ),
	(
		Message[ServiceExecute::nparam,"ListID","MailChimp"];
		Throw[$Failed]
	)];
 	data = FixedPoint[Normal,ServiceExecute["MailChimp", "RawListActivity",{"id" -> listid}]];
  	If[("status"/.data)=="error",
   	(
      	Message[ServiceExecute::serrormsg,("error"/.data)];
       	Throw[$Failed]
 	)];
   	withCamelTitles=Replace[data, Rule[a_, b_] :> Rule[camelCase[a], b], Infinity]
 	/. {(y : "Day" -> x_) :> (y -> If[MatchQ[x, _String], DateObject[x, TimeZone -> 0], x]),
 		"UserId"->"UserID"
 		};
 	Dataset[Association@@@withCamelTitles]
]

mailchimpcookeddata["ListClients",id_,args_] :=Block[ {data,invalidParameters,withCamelTitles,listid,newparams},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"ListID"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"MailChimp"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[ KeyExistsQ[newparams,"ListID"],
  	(
      	listid = ToString["ListID" /. newparams];                        
   	),
	(
		Message[ServiceExecute::nparam,"ListID","MailChimp"];
		Throw[$Failed]
	)];
 	data =FixedPoint[Normal, ServiceExecute["MailChimp", "RawListClients",{"id" -> listid}]];
 	If[("status"/.data)=="error",
   	(
      	Message[ServiceExecute::serrormsg,("error"/.data)];
       	Throw[$Failed]
 	)];
   	withCamelTitles=Replace[data, Rule[a_, b_] :> Rule[camelCase[a], b], Infinity];
   	Dataset[Association @@ Replace[withCamelTitles, r : {__Rule} :> Association[r], -1]]
]

mailchimpcookeddata["ListMonthlyGrowth",id_,args_] :=Block[ {data,invalidParameters,withCamelTitles,listid,newparams},
  	newparams=args;
  	invalidParameters = Select[Keys[newparams],!MemberQ[{"ListID"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"MailChimp"]&/@invalidParameters;
		Throw[$Failed]
	)];
  	If[ KeyExistsQ[newparams,"ListID"],
   	(
    	listid = ToString["ListID" /. newparams];                        
   	),
	(
		Message[ServiceExecute::nparam,"ListID","MailChimp"];
		Throw[$Failed]
	)];
 	data = FixedPoint[Normal, ServiceExecute["MailChimp", "RawListGrowthHistory",{"id" -> listid}]];
	If[("status"/.data)=="error",
   	(
      	Message[ServiceExecute::serrormsg,("error"/.data)];
       	Throw[$Failed]
 	)];
	withCamelTitles=Replace[data, Rule[a_, b_] :> Rule[camelCase[a], b], Infinity]
  	/. {(y : "Month" -> x_) :> (y -> If[MatchQ[x, _String], DateObject[x], x])};
   	Dataset[Association@@@withCamelTitles]
]

mailchimpcookeddata["Lists",id_,args_] :=Block[ {data,invalidParameters,ignore,withCamelTitles,listid = Null,listname = Null,fromname = Null,fromemail = Null,fromsubject = Null,createdbefore = Null,createdafter = Null,exact = Null,start = Null, limit = Null, sortfield = Null,newparams},
	newparams=args/.{MaxItems:>"MaxItems"};
	invalidParameters = Select[Keys[newparams],!MemberQ[{"ListID","ListName","FromName","FromAddress","Subject","CreatedBefore","CreatedAfter","Exact","MaxItems","StartIndex","SortField",IgnoreCase},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"MailChimp"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[ KeyExistsQ[newparams,"ListID"],
	(
		If[!(MatchQ["ListID" /. newparams, {__String}]||StringQ["ListID" /. newparams]),
		(	
			Message[ServiceExecute::nval,"ListID","MailChimp"];
			Throw[$Failed]
		)];
		If[ListQ["ListID" /. newparams],
			listid=StringJoin @@ Riffle["ListID" /. newparams, ","]
		];
		If[StringQ["ListID" /. newparams],	
			listid = "ListID" /. newparams
		];                       
	)];
	If[ KeyExistsQ[newparams,"ListName"],
	(
		If[!StringQ["ListName" /. newparams],
		(	
			Message[ServiceExecute::nval,"ListName","MailChimp"];
			Throw[$Failed]
		)];
		listname = "ListName" /. newparams;                        
	)];
	If[ KeyExistsQ[newparams,"FromName"],
	(
		If[!StringQ["FromName" /. newparams],
		(	
			Message[ServiceExecute::nval,"FromName","MailChimp"];
			Throw[$Failed]
		)];
		fromname = "FromName" /. newparams;                        
	)];
	If[ KeyExistsQ[newparams,"FromAddress"],
	(
		If[!StringQ["FromAddress" /. newparams],
		(	
			Message[ServiceExecute::nval,"FromAddress","MailChimp"];
			Throw[$Failed]
		)];
		fromemail = "FromAddress" /. newparams;                        
	)];
	If[ KeyExistsQ[newparams,"Subject"],
	(
		If[!StringQ["Subject" /. newparams],
		(	
			Message[ServiceExecute::nval,"Subject","MailChimp"];
			Throw[$Failed]
		)];
		fromsubject = "Subject" /. newparams;                        
	)];
	If[ KeyExistsQ[newparams,"CreatedBefore"],
	(
		If[!(StringQ["CreatedBefore"/.newparams]||MatchQ["CreatedBefore"/.newparams,DateObject[__]]),
		(	
			Message[ServiceExecute::nval,"CreatedBefore","MailChimp"];
			Throw[$Failed]
		)];
		createdbefore = DateObject[("CreatedBefore" /. newparams)];
		If[MatchQ[createdbefore,DateObject[__String]],
		(	
			Message[ServiceExecute::nval,"CreatedBefore","MailChimp"];
			Throw[$Failed]
		)];
        createdbefore = DateString[TimeZoneConvert[createdbefore,0], {"Year", "-", "Month", "-", "Day", " ", "Time"}]             
	)];
	If[ KeyExistsQ[newparams,"CreatedAfter"],
	(
		If[!(StringQ["CreatedAfter"/.newparams]||MatchQ["CreatedAfter"/.newparams,DateObject[__]]),
		(	
			Message[ServiceExecute::nval,"CreatedAfter","MailChimp"];
			Throw[$Failed]
		)];
		createdafter = DateObject[("CreatedAfter" /. newparams)];
		If[MatchQ[createdafter,DateObject[__String]],
		(	
			Message[ServiceExecute::nval,"CreatedAfter","MailChimp"];
			Throw[$Failed]
		)];
        createdafter = DateString[TimeZoneConvert[createdafter,0], {"Year", "-", "Month", "-", "Day", " ", "Time"}]             
	)];                    
 	If[ KeyExistsQ[newparams,"Exact"],
 	(
     	If[!MemberQ[{True,False},"Exact" /. newparams],
		(	
			Message[ServiceExecute::nval,"Exact","MailChimp"];
			Throw[$Failed]
		)];
     	exact = ("Exact" /. newparams)/.{True->"true",False->"false"};                       
   	)];
 	If[ KeyExistsQ[newparams,"StartIndex"],
 	(
    	If[!(IntegerQ["StartIndex"/.newparams]&&("StartIndex"/.newparams)>0),
		(	
			Message[ServiceExecute::nval,"StartIndex","MailChimp"];
			Throw[$Failed]
		)];
    	start = ToString[("StartIndex" /. newparams)-1];                        
  	)];
  	If[ KeyExistsQ[newparams,"MaxItems"],
   	(
   		If[!(IntegerQ["MaxItems"/.newparams]&&("MaxItems"/.newparams)>=0&&("MaxItems"/.newparams)<=100),
		(	
			Message[ServiceExecute::nval,"MaxItems","MailChimp"];
			Throw[$Failed]
		)];
     	limit = ToString["MaxItems" /. newparams];                        
  	),
  	(
  		limit="10"
  	)];
  	If[KeyExistsQ[newparams,IgnoreCase],
	(
		If[!MemberQ[{True,False},IgnoreCase /. newparams],
		(	
			Message[ServiceExecute::nval,IgnoreCase,"MailChimp"];
			Throw[$Failed]
		)];
		ignore = IgnoreCase /. newparams                        
	),
	(
		ignore=True
	)];
  	If[ KeyExistsQ[newparams,"SortField"],
   	(
   		If[!StringMatchQ[ToString["SortField" /. newparams],  "Created"| "Web" , IgnoreCase -> ignore],
		(	
			Message[ServiceExecute::nval,"SortField","MailChimp"];
			Throw[$Failed]
		)];
   		sortfield = "SortField" /. newparams;                        
 	)];
  	data = FixedPoint[Normal, ServiceExecute["MailChimp", "RawLists",DeleteCases[{"filters[list_id]"->listid,"filters[list_name]" -> listname, "filters[from_name]" -> fromname, "filters[from_email]" -> fromemail, "filters[from_subject]" -> fromsubject, "filters[created_before]" -> createdbefore, "filters[created_after]" -> createdafter, "filters[exact]" -> exact, "start" -> start, "limit" -> limit, "sort_field" -> sortfield,"sort_dir" -> "ASC"},_->Null]]];
  	If[("status"/.data)=="error",
   	(
      	Message[ServiceExecute::serrormsg,("error"/.data)];
       	Throw[$Failed]
 	)];
  	withCamelTitles=Replace[Replace[("data" /. data), Rule[a_, b_] :> Rule[camelCase[a], b], Infinity]
  	, {(y : ("DateCreated" | "DateLastCampaign") -> x_) :> (y -> If[MatchQ[x, _String], DateObject[x, TimeZone -> 0], x]),
   		"SubscribeUrlShort"->"SubscribeURLShort","SubscribeUrlLong"->"SubscribeURLLong","Id"->"ID","WebId"->"WebID","DefaultFromEmail"->"DefaultFromAddress"  	
		},Infinity];
  	Dataset[Replace[Replace[withCamelTitles, r : {__Rule} :> Association[r], -1],Null -> Missing["NotAvailable"], Infinity]]
]
mailchimpcookeddata["Lists",id_] :=	Block[ {data,withCamelTitles},
	data = FixedPoint[Normal, ServiceExecute["MailChimp", "RawLists"]];
 	If[("status"/.data)=="error",
   	(
      	Message[ServiceExecute::serrormsg,("error"/.data)];
       	Throw[$Failed]
 	)];
 	withCamelTitles=Replace[("data" /. data), Rule[a_, b_] :> Rule[camelCase[a], b], Infinity]
  	/. {(y : "DateCreated"|"DateLastCampaign" -> x_) :> (y -> If[MatchQ[x, _String], DateObject[x, TimeZone -> 0], x]),
  		"SubscribeUrlShort"->"SubscribeURLShort","SubscribeUrlLong"->"SubscribeURLLong","Id"->"ID","WebId"->"WebID","DefaultFromEmail"->"DefaultFromAddress"  	
  		};
  	Dataset[Replace[Replace[withCamelTitles, r : {__Rule} :> Association[r], -1],Null -> Missing["NotAvailable"], Infinity]]
]


mailchimpcookeddata["ListLocations",id_,args_] :=Block[{data,invalidParameters,withCamelTitles,listid,newparams},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"ListID"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"MailChimp"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"ListID"],
	(
		listid = ToString["ListID" /. newparams];                        
	),
	(
		Message[ServiceExecute::nparam,"ListID","MailChimp"];
		Throw[$Failed]
	)];
	data = FixedPoint[Normal, ServiceExecute["MailChimp", "RawListLocations",{"id" -> listid}]];
	If[("status"/.data)=="error",
   	(
      	Message[ServiceExecute::serrormsg,("error"/.data)];
       	Throw[$Failed]
 	)];
	withCamelTitles=Replace[data, Rule[a_, b_] :> Rule[camelCase[a], b], Infinity]
	/. {("Country"->p_String):>("Country"->If[MatchQ[q=Interpreter["Country"][p],Failure[__]],p,q]),
		"Cc"->"CountryCode"
		};
	Dataset[Association@@@withCamelTitles]
]

mailchimpcookeddata["ListMembers",id_,args_] :=Block[{data,invalidParameters,withCamelTitles,ignore,listid,status = Null,start = Null, limit = Null, sortfield = Null,segment=Null,newparams},
	newparams=args/.{MaxItems:>"MaxItems"};
	invalidParameters = Select[Keys[newparams],!MemberQ[{"ListID","Status","MaxItems","StartIndex","SortField","Segment",IgnoreCase},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"MailChimp"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,"ListID"],
	(
		listid = ToString["ListID" /. newparams];                        
	),
	(
		Message[ServiceExecute::nparam,"ListID","MailChimp"];
		Throw[$Failed]
	)];
	If[KeyExistsQ[newparams,IgnoreCase],
	(
		If[!MemberQ[{True,False},IgnoreCase /. newparams],
		(	
			Message[ServiceExecute::nval,IgnoreCase,"MailChimp"];
			Throw[$Failed]
		)];
		ignore = IgnoreCase /. newparams                        
	),
	(
		ignore=True
	)];
	If[ KeyExistsQ[newparams,"Status"],
  	(
		If[!StringMatchQ[ToString["Status" /. newparams], "Unsubscribed" | "Subscribed" | "Cleaned", IgnoreCase -> ignore],
		(	
			Message[ServiceExecute::nval,"Status","MailChimp"];
			Throw[$Failed]
		)];  	
		status = "Status" /. newparams;                        
	)];
	If[ KeyExistsQ[newparams,"StartIndex"],
 	(
    	If[!(IntegerQ["StartIndex"/.newparams]&&("StartIndex"/.newparams)>0),
		(	
			Message[ServiceExecute::nval,"StartIndex","MailChimp"];
			Throw[$Failed]
		)];
    	start = ToString[("StartIndex" /. newparams)-1];                        
  	)];
  	If[ KeyExistsQ[newparams,"MaxItems"],
   	(
   		If[!(IntegerQ["MaxItems"/.newparams]&&("MaxItems"/.newparams)>=0&&("MaxItems"/.newparams)<=100),
		(	
			Message[ServiceExecute::nval,"MaxItems","MailChimp"];
			Throw[$Failed]
		)];
     	limit = ToString["MaxItems" /. newparams];                        
  	),
  	(
  		limit="10"
  	)];
	If[ KeyExistsQ[newparams,"SortField"],
   	(
   		If[!StringMatchQ[ToString["SortField" /. newparams],  "Email"| "Rating" | "LastUpdateTime" | "OptinTime", IgnoreCase -> ignore],
		(	
			Message[ServiceExecute::nval,"SortField","MailChimp"];
			Throw[$Failed]
		)];
   		sortfield = ToLowerCase["SortField" /. newparams]/.{"lastupdatetime"->"last_update_time","optintime"->"optin_time"};                        
 	)];
	If[ KeyExistsQ[newparams,"Segment"],
  	(
		segment = "Segment" /. newparams;                        
	)];
	data = FixedPoint[Normal, ServiceExecute["MailChimp", "RawListMembers",DeleteCases[{"id"->listid,"status" -> status, "opts[start]" -> start, "opts[limit]" -> limit, "opts[sort_field]" -> sortfield,"opts[sort_dir]" -> "ASC","opts[segment]" -> segment},_->Null]]];
	If[("status"/.data)=="error",
   	(
      	Message[ServiceExecute::serrormsg,("error"/.data)];
       	Throw[$Failed]
 	)];
	withCamelTitles=Replace["data" /. data, Rule[a_, b_] :> Rule[camelCase[a], b], Infinity]
	 /. {(y : "TimestampOpt"|"TimestampSignup"|"Timestamp"|"InfoChanged" -> x_) :> (y -> If[MatchQ[x, _String], DateObject[x, TimeZone -> 0], x]),
	 	"IpSignup"->"IPSignUp","Id"->"ID","IpOpt"->"IPOpt","WebId"->"WebID","Euid"->"EUID","Leid"->"LEID",
	 	"ListId"->"ListID","IconUrl"->"IconURL","Dstoff"->"DSTOff","Cc"->"CountryCode","Gmtoff"->"GMTOff","EMAIL"->"Email","FNAME"->"FirstName",
	 	"LNAME"->"LastName","Timezone"->"TimeZone"
	 	}
	  /. {"TimestampOpt"->"TimeStampOpt","TimestampSignup"->"TimeStampSignUp","Timestamp"->"TimeStamp"
	 	};
	Dataset[Replace[Replace[withCamelTitles, r : {__Rule} :> Association[r], -1],Null -> Missing["NotAvailable"], Infinity]]
]


mailchimpcookeddata["Campaigns",id_,args_] :=Block[ {data,withCamelTitles,invalidParameters,ignore,listid = Null,campaignid = Null,fromname = Null,fromemail = Null,subject = Null,parentid = Null,folderid = Null,exact = Null,start = Null, limit = Null, sortfield = Null, templateid = Null, status = Null, type = Null, title = Null, sendtimestart = Null, sendtimeend = Null, usessegment = Null,newparams},
	newparams=args/.{MaxItems:>"MaxItems"};
	invalidParameters = Select[Keys[newparams],!MemberQ[{"CampaignID","ListID","FromName","FromAddress","Subject","Title","FolderID","ParentID","TemplateID","Status","Type","SendTimeStart","SendTimeEnd","UsesSegment","Exact","MaxItems","StartIndex","SortField",IgnoreCase},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"MailChimp"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[ KeyExistsQ[newparams,"ListID"],
	(
		If[!(MatchQ["ListID" /. newparams, {__String}]||StringQ["ListID" /. newparams]),
		(	
			Message[ServiceExecute::nval,"ListID","MailChimp"];
			Throw[$Failed]
		)];
		If[ListQ["ListID" /. newparams],
			listid=StringJoin @@ Riffle["ListID" /. newparams, ","]
		];
		If[StringQ["ListID" /. newparams],	
			listid = "ListID" /. newparams
		];                       
	)];
	If[ KeyExistsQ[newparams,"CampaignID"],
	(
		If[!(MatchQ["CampaignID" /. newparams, {__String}]||StringQ["CampaignID" /. newparams]),
		(	
			Message[ServiceExecute::nval,"CampaignID","MailChimp"];
			Throw[$Failed]
		)];
		If[ListQ["CampaignID" /. newparams],
			campaignid=StringJoin @@ Riffle["CampaignID" /. newparams, ","]
		];
		If[StringQ["CampaignID" /. newparams],	
			campaignid = "CampaignID" /. newparams
		];           
	)];
	If[ KeyExistsQ[newparams,"FromName"],
	(
		If[!StringQ["FromName" /. newparams],
		(	
			Message[ServiceExecute::nval,"FromName","MailChimp"];
			Throw[$Failed]
		)];
		fromname = "FromName" /. newparams;                        
	)];
	If[ KeyExistsQ[newparams,"FromAddress"],
	(
		If[!StringQ["FromAddress" /. newparams],
		(	
			Message[ServiceExecute::nval,"FromAddress","MailChimp"];
			Throw[$Failed]
		)];
		fromemail = "FromAddress" /. newparams;                        
	)];
	If[ KeyExistsQ[newparams,"Subject"],
	(
		If[!StringQ["Subject" /. newparams],
		(	
			Message[ServiceExecute::nval,"Subject","MailChimp"];
			Throw[$Failed]
		)];
		subject = "Subject" /. newparams;                        
	)];
	If[ KeyExistsQ[newparams,"Title"],
	(
		If[!StringQ["Title" /. newparams],
		(	
			Message[ServiceExecute::nval,"Title","MailChimp"];
			Throw[$Failed]
		)];
		title = "Title" /. newparams;                        
	)];
	If[ KeyExistsQ[newparams,"ParentID"],
	(
		If[!(MatchQ["ParentID" /. newparams, {__String}]||StringQ["ParentID" /. newparams]),
		(	
			Message[ServiceExecute::nval,"ParentID","MailChimp"];
			Throw[$Failed]
		)];
		If[ListQ["ParentID" /. newparams],
			parentid=StringJoin @@ Riffle["ParentID" /. newparams, ","]
		];
		If[StringQ["ParentID" /. newparams],	
			parentid = "ParentID" /. newparams
		];			                  
	)];
	If[ KeyExistsQ[newparams,"FolderID"],
	(
		If[!(MatchQ["FolderID" /. newparams, {__Integer}]||IntegerQ["FolderID" /. newparams]),
		(	
			Message[ServiceExecute::nval,"FolderID","MailChimp"];
			Throw[$Failed]
		)];
		If[ListQ["FolderID" /. newparams],
			folderid=StringJoin @@ Riffle[ToString /@("FolderID" /. newparams), ","]
		];
		If[IntegerQ["FolderID" /. newparams],	
			folderid = ToString["FolderID" /. newparams]
		];			                  
	)];
	If[ KeyExistsQ[newparams,"TemplateID"],
	(
		If[!(MatchQ["TemplateID" /. newparams, {__Integer}]||IntegerQ["TemplateID" /. newparams]),
		(	
			Message[ServiceExecute::nval,"TemplateID","MailChimp"];
			Throw[$Failed]
		)];
		If[ListQ["TemplateID" /. newparams],
			templateid=StringJoin @@ Riffle[ToString /@("TemplateID" /. newparams), ","]
		];
		If[IntegerQ["TemplateID" /. newparams],	
			templateid = ToString["TemplateID" /. newparams]
		];                        
	)];
	If[KeyExistsQ[newparams,IgnoreCase],
	(
		If[!MemberQ[{True,False},IgnoreCase /. newparams],
		(	
			Message[ServiceExecute::nval,IgnoreCase,"MailChimp"];
			Throw[$Failed]
		)];
		ignore = IgnoreCase /. newparams                        
	),
	(
		ignore=True
	)];
	
	If[ KeyExistsQ[newparams,"Status"],
	(
		If[!(MatchQ["Status" /. newparams, {__String}]||StringQ["Status" /. newparams]),
		(	
			Message[ServiceExecute::nval,"Status","MailChimp"];
			Throw[$Failed]
		)];
		If[ListQ["Status" /. newparams],
			If[!(And@@StringMatchQ["Status" /. newparams, "Sent"| "Save"| "Paused"| "Schedule"| "Sending", IgnoreCase -> ignore]),
			(	
				Message[ServiceExecute::nval,"Status","MailChimp"];
				Throw[$Failed]
			)];
			status=StringJoin @@ Riffle["Status" /. newparams, ","]
		];
		If[StringQ["Status" /. newparams],	
			If[!StringMatchQ["Status" /. newparams, "Sent"| "Save"| "Paused"| "Schedule"| "Sending", IgnoreCase -> ignore],
			(	
				Message[ServiceExecute::nval,"Status","MailChimp"];
				Throw[$Failed]
			)];
			status = "Status" /. newparams
		];                       
	)];
	If[ KeyExistsQ[newparams,"Type"],
	(
		If[!(MatchQ["Type" /. newparams, {__String}]||StringQ["Type" /. newparams]),
		(	
			Message[ServiceExecute::nval,"Type","MailChimp"];
			Throw[$Failed]
		)];
		If[ListQ["Type" /. newparams],
			If[!(And@@StringMatchQ["Type" /. newparams, "Regular"| "Plaintext"| "Absplit"| "Rss"| "Auto", IgnoreCase -> ignore]),
			(	
				Message[ServiceExecute::nval,"Type","MailChimp"];
				Throw[$Failed]
			)];
			type=StringJoin @@ Riffle["Type" /. newparams, ","]
		];
		If[StringQ["Type" /. newparams],	
			If[!StringMatchQ["Type" /. newparams, "Regular"| "Plaintext"| "Absplit"| "Rss"| "Auto", IgnoreCase -> ignore],
			(	
				Message[ServiceExecute::nval,"Type","MailChimp"];
				Throw[$Failed]
			)];
			type = "Type" /. newparams
		];
 	)];
 	
	If[ KeyExistsQ[newparams,"SendTimeStart"],
	(
		If[!(StringQ["SendTimeStart"/.newparams]||MatchQ["SendTimeStart"/.newparams,DateObject[__]]),
		(	
			Message[ServiceExecute::nval,"SendTimeStart","MailChimp"];
			Throw[$Failed]
		)];
		sendtimestart = DateObject[("SendTimeStart" /. newparams)];
		If[MatchQ[sendtimestart,DateObject[__String]],
		(	
			Message[ServiceExecute::nval,"SendTimeStart","MailChimp"];
			Throw[$Failed]
		)];
        sendtimestart = DateString[TimeZoneConvert[sendtimestart,0], {"Year", "-", "Month", "-", "Day", " ", "Time"}]                   
	)];
	If[ KeyExistsQ[newparams,"SendTimeEnd"],
	(
		If[!(StringQ["SendTimeEnd"/.newparams]||MatchQ["SendTimeEnd"/.newparams,DateObject[__]]),
		(	
			Message[ServiceExecute::nval,"SendTimeEnd","MailChimp"];
			Throw[$Failed]
		)];
		sendtimeend = DateObject[("SendTimeEnd" /. newparams)];
		If[MatchQ[sendtimeend,DateObject[__String]],
		(	
			Message[ServiceExecute::nval,"SendTimeEnd","MailChimp"];
			Throw[$Failed]
		)];
        sendtimeend = DateString[TimeZoneConvert[sendtimeend,0], {"Year", "-", "Month", "-", "Day", " ", "Time"}]                    
	)];
	If[ KeyExistsQ[newparams,"UsesSegment"],
 	(
    	If[!MemberQ[{True,False},"UsesSegment" /. newparams],
		(	
			Message[ServiceExecute::nval,"UsesSegment","MailChimp"];
			Throw[$Failed]
		)];
    	usessegment = ("UsesSegment" /. newparams)/.{True->"true",False->"false"};                     
  	)];
	If[ KeyExistsQ[newparams,"Exact"],
 	(
     	If[!MemberQ[{True,False},"Exact" /. newparams],
		(	
			Message[ServiceExecute::nval,"Exact","MailChimp"];
			Throw[$Failed]
		)];
     	exact = ("Exact" /. newparams)/.{True->"true",False->"false"};                       
   	)];
	If[ KeyExistsQ[newparams,"StartIndex"],
 	(
    	If[!(IntegerQ["StartIndex"/.newparams]&&("StartIndex"/.newparams)>0),
		(	
			Message[ServiceExecute::nval,"StartIndex","MailChimp"];
			Throw[$Failed]
		)];
    	start = ToString[("StartIndex" /. newparams)-1];                        
  	)];
  	If[ KeyExistsQ[newparams,"MaxItems"],
   	(
   		If[!(IntegerQ["MaxItems"/.newparams]&&("MaxItems"/.newparams)>=0&&("MaxItems"/.newparams)<=1000),
		(	
			Message[ServiceExecute::nval,"MaxItems","MailChimp"];
			Throw[$Failed]
		)];
     	limit = ToString["MaxItems" /. newparams];                        
  	),
  	(
  		limit="10"
  	)];
  	If[ KeyExistsQ[newparams,"SortField"],
   	(
   		If[!StringMatchQ[ToString["SortField" /. newparams], "CreateTime" | "SendTime" | "Title" | "Subject" , IgnoreCase -> ignore],
		(	
			Message[ServiceExecute::nval,"SortField","MailChimp"];
			Throw[$Failed]
		)];
   		sortfield = ToLowerCase["SortField" /. newparams]/.{"createtime"->"create_time","sendtime"->"send_time"};                        
 	)];
	data = FixedPoint[Normal,ServiceExecute["MailChimp", "RawCampaigns",DeleteCases[{"filters[campaign_id]"->campaignid,"filters[parent_id]" -> parentid,"filters[list_id]"->listid,"filters[folder_id]" -> folderid,"filters[template_id]" -> templateid, "filters[status]" -> status,"filters[type]" -> type,"filters[from_name]" -> fromname, "filters[from_email]" -> fromemail, "filters[title]" -> title,"filters[subject]" -> subject, "filters[sendtime_start]" -> sendtimestart, "filters[sendtime_end]" -> sendtimeend,"filters[uses_segment]" -> usessegment, "filters[exact]" -> exact, "start" -> start, "limit" -> limit, "sort_field" -> sortfield,"sort_dir" -> "ASC"},_->Null]]];
 	If[("status"/.data)=="error",
   	(
      	Message[ServiceExecute::serrormsg,("error"/.data)];
       	Throw[$Failed]
 	)];
 	withCamelTitles=Replace[(DeleteCases[#, "summary" -> __] & /@ ("data" /. data)), Rule[a_, b_] :> Rule[camelCase[a], b], Infinity]
 	/. {(y : "SendTime"|"CreateTime"|"ContentUpdatedTime"|"TimewarpSchedule" -> x_) :> (y -> If[MatchQ[x, _String], DateObject[x, TimeZone -> 0], x]),
		"Id"->"ID","FolderId"->"FolderID","ListId"->"ListID","WebId"->"WebID","TemplateId"->"TemplateID","ArchiveUrlLong"->"ArchiveURLLong","ArchiveUrl"->"ArchiveURL","ParentId"->"ParentID",
		"HtmlClicks"->"HTMLClicks","Timewarp"->"TimeWarp","FromEmail"->"FromAddress"
 		};
 	Dataset[Replace[Replace[withCamelTitles, {r__Rule} :> Association[r], -1],Null->Missing["NotAvailable"],Infinity]]
]
mailchimpcookeddata["Campaigns",id_] :=Block[ {data,withCamelTitles},
	data = FixedPoint[Normal,ServiceExecute["MailChimp", "RawCampaigns"]];
 	If[("status"/.data)=="error",
   	(
      	Message[ServiceExecute::serrormsg,("error"/.data)];
       	Throw[$Failed]
 	)];
	withCamelTitles=Replace[(DeleteCases[#, "summary" -> __] & /@ ("data" /. data)), Rule[a_, b_] :> Rule[camelCase[a], b], Infinity]
 	/. {(y : "SendTime"|"CreateTime"|"ContentUpdatedTime"|"TimewarpSchedule" -> x_) :> (y -> If[MatchQ[x, _String], DateObject[x, TimeZone -> 0], x]),
		"Id"->"ID","FolderId"->"FolderID","ListId"->"ListID","WebId"->"WebID","TemplateId"->"TemplateID","ArchiveUrlLong"->"ArchiveURLLong","ArchiveUrl"->"ArchiveURL","ParentId"->"ParentID",
		"HtmlClicks"->"HTMLClicks","Timewarp"->"TimeWarp","FromEmail"->"FromAddress"	
 		};
  	Dataset[Replace[Replace[withCamelTitles, {r__Rule} :> Association[r], -1],Null->Missing["NotAvailable"],Infinity]]
]


mailchimpcookeddata["CampaignAbuse",id_,args_] :=Block[{data,invalidParameters,withCamelTitles,campaignid,start = Null,limit = Null,since = Null,newparams},
	newparams=args/.{MaxItems:>"MaxItems"};
	invalidParameters = Select[Keys[newparams],!MemberQ[{"CampaignID","StartIndex","MaxItems","StartDate"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"MailChimp"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[ KeyExistsQ[newparams,"CampaignID"],
  	(
   		campaignid = ToString["CampaignID" /. newparams];                        
   	),
	(
		Message[ServiceExecute::nparam,"CampaignID","MailChimp"];
		Throw[$Failed]
	)];
  	If[ KeyExistsQ[newparams,"StartIndex"],
 	(
    	If[!(IntegerQ["StartIndex"/.newparams]&&("StartIndex"/.newparams)>0),
		(	
			Message[ServiceExecute::nval,"StartIndex","MailChimp"];
			Throw[$Failed]
		)];
    	start = ToString[("StartIndex" /. newparams)-1];                        
  	)];
  	If[ KeyExistsQ[newparams,"MaxItems"],
   	(
   		If[!(IntegerQ["MaxItems"/.newparams]&&("MaxItems"/.newparams)>=0&&("MaxItems"/.newparams)<=100),
		(	
			Message[ServiceExecute::nval,"MaxItems","MailChimp"];
			Throw[$Failed]
		)];
     	limit = ToString["MaxItems" /. newparams];                        
  	)];
   	If[ KeyExistsQ[newparams,"StartDate"],
   	(
		If[!(StringQ["StartDate"/.newparams]||MatchQ["StartDate"/.newparams,DateObject[__]]),
		(	
			Message[ServiceExecute::nval,"StartDate","MailChimp"];
			Throw[$Failed]
		)];
		since = DateObject[("StartDate" /. newparams)];
		If[MatchQ[since,DateObject[__String]],
		(	
			Message[ServiceExecute::nval,"StartDate","MailChimp"];
			Throw[$Failed]
		)];
        since = DateString[TimeZoneConvert[since,0], {"Year", "-", "Month", "-", "Day", " ", "Time"}]             
	)];
  	data = FixedPoint[Normal,ServiceExecute["MailChimp", "RawReportAbuse", DeleteCases[{"cid" -> campaignid ,"opts[start]" -> start, "opts[limit]"-> limit,"opts[since]"-> since}, _ -> Null]]];
  	If[("status"/.data)=="error",
   	(
      	Message[ServiceExecute::serrormsg,("error"/.data)];
       	Throw[$Failed]
 	)];
  	withCamelTitles=Replace[("data" /. data), Rule[a_, b_] :> Rule[camelCase[a], b], Infinity]
 	/. {(y : "Date"|"TimestampOpt" | "TimestampSignup" | "Timestamp"|"InfoChanged" -> x_) :> (y -> If[MatchQ[x, _String], DateObject[x, TimeZone -> 0], x])};
 	Dataset[Replace[Replace[withCamelTitles, {r__Rule} :> Association[r], -1],Null->Missing["NotAvailable"],Infinity]]
 	(*Dataset[Association @@@ withCamelTitles]*)
]

mailchimpcookeddata["CampaignClicks",id_,args_] :=Block[{data,invalidParameters,withCamelTitles,campaignid,newparams},
 	newparams=args;
 	invalidParameters = Select[Keys[newparams],!MemberQ[{"CampaignID"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"MailChimp"]&/@invalidParameters;
		Throw[$Failed]
	)];
 	If[ KeyExistsQ[newparams,"CampaignID"],
   	(
      	campaignid = ToString["CampaignID" /. newparams];                        
  	),
	(
		Message[ServiceExecute::nparam,"CampaignID","MailChimp"];
		Throw[$Failed]
	)];
  	data = FixedPoint[Normal,ServiceExecute["MailChimp", "RawReportClicks",{"cid" -> campaignid}]];
  	If[("status"/.data)=="error",
   	(
      	Message[ServiceExecute::serrormsg,("error"/.data)];
       	Throw[$Failed]
 	)];
  	withCamelTitles=Replace[("total"/.data), Rule[a_, b_] :> Rule[camelCase[a], b], Infinity]
  	/.{"Tid"->"TrackingID","Url"->"URL"};
  	Dataset[Association@@@withCamelTitles]
]

mailchimpcookeddata["CampaignLocations",id_,args_] :=Block[{data,invalidParameters,withCamelTitles,campaignid,newparams},
 	newparams=args;
 	invalidParameters = Select[Keys[newparams],!MemberQ[{"CampaignID"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"MailChimp"]&/@invalidParameters;
		Throw[$Failed]
	)];
 	If[ KeyExistsQ[newparams,"CampaignID"],
  	(
       	campaignid = ToString["CampaignID" /. newparams];                        
   	),
	(
		Message[ServiceExecute::nparam,"CampaignID","MailChimp"];
		Throw[$Failed]
	)];
  	data = FixedPoint[Normal,ServiceExecute["MailChimp", "RawReportGeoOpens",{"cid" -> campaignid}]];
   	If[("status"/.data)=="error",
   	(
      	Message[ServiceExecute::serrormsg,("error"/.data)];
       	Throw[$Failed]
 	)];
   	withCamelTitles=Replace[Replace[data, Rule[a_, b_] :> Rule[camelCase[a], b], Infinity]
 	,{("Name"->p_String):>("Name"->If[MatchQ[q=Interpreter["Country"][p],Failure[__]],p,q])},{2}](*, {(y : "Name" -> x_) :> (y -> If[MatchQ[x, _String], Interpreter["City"][x], x])},{5}]*);
 	Dataset[Replace[withCamelTitles, {r__Rule} :> Association[r], -1]]
]

mailchimpcookeddata["CampaignSummary",id_,args_] :=Block[{data,invalidParameters,withCamelTitles,campaignid,newparams},
  	newparams=args;
  	invalidParameters = Select[Keys[newparams],!MemberQ[{"CampaignID"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"MailChimp"]&/@invalidParameters;
		Throw[$Failed]
	)];
  	If[ KeyExistsQ[newparams,"CampaignID"],
  	(
       	campaignid = ToString["CampaignID" /. newparams];                        
   	),
	(
		Message[ServiceExecute::nparam,"CampaignID","MailChimp"];
		Throw[$Failed]
	)];
   	data = FixedPoint[Normal,ServiceExecute["MailChimp", "RawReportSummary",{"cid" -> campaignid}]];
  	If[("status"/.data)=="error",
   	(
      	Message[ServiceExecute::serrormsg,("error"/.data)];
       	Throw[$Failed]
 	)];
  	withCamelTitles=Replace[Drop[data, -1], Rule[a_, b_] :> Rule[camelCase[a], b], Infinity]
  	/. {(y : "LastOpen"|"LastClick" -> x_) :> (y -> If[MatchQ[x, _String], DateObject[x, TimeZone -> 0], x]),
  		"Absplit"->"ABSplit","Timewarp"->"TimeWarp"
  		};
   	Dataset[Association @@ Replace[Replace[withCamelTitles, {r__Rule} :> Association[r], -1],Null->Missing["NotAvailable"],Infinity]]
]

mailchimpcookeddata["CampaignSummaryTimeSeries",id_,args_] :=Block[{data,invalidParameters,withCamelTitles,campaignid,newparams},
	newparams=args;
	invalidParameters = Select[Keys[newparams],!MemberQ[{"CampaignID"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"MailChimp"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[ KeyExistsQ[newparams,"CampaignID"],
	(
      	campaignid = ToString["CampaignID" /. newparams];                        
   	),
	(
		Message[ServiceExecute::nparam,"CampaignID","MailChimp"];
		Throw[$Failed]
	)];
  	data = FixedPoint[Normal,ServiceExecute["MailChimp", "RawReportSummary",{"cid" -> campaignid}]];
  	If[("status"/.data)=="error",
   	(
      	Message[ServiceExecute::serrormsg,("error"/.data)];
       	Throw[$Failed]
 	)];
  	withCamelTitles=Replace[("timeseries" /. data), Rule[a_, b_] :> Rule[camelCase[a], b], Infinity]
  	/. {(y : "Timestamp" -> x_) :> (y -> If[MatchQ[x, _String], DateObject[x, TimeZone -> 0], x])}
  	/. {"Timestamp"->"TimeStamp"};
  	Dataset[Association @@@ withCamelTitles]
]

mailchimpcookeddata["CampaignRecipients",id_,args_] :=Block[ {data,invalidParameters,ignore,withCamelTitles,campaignid,start = Null,limit = Null,status = Null,newparams},
	newparams=args/.{MaxItems:>"MaxItems"};
	invalidParameters = Select[Keys[newparams],!MemberQ[{"CampaignID","Status","MaxItems","StartIndex",IgnoreCase},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"MailChimp"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[ KeyExistsQ[newparams,"CampaignID"],
  	(
     	campaignid = ToString["CampaignID" /. newparams];                        
  	),
	(
		Message[ServiceExecute::nparam,"CampaignID","MailChimp"];
		Throw[$Failed]
	)];
	If[ KeyExistsQ[newparams,"StartIndex"],
 	(
    	If[!(IntegerQ["StartIndex"/.newparams]&&("StartIndex"/.newparams)>0),
		(	
			Message[ServiceExecute::nval,"StartIndex","MailChimp"];
			Throw[$Failed]
		)];
    	start = ToString[("StartIndex" /. newparams)-1];                        
  	)];
  	If[ KeyExistsQ[newparams,"MaxItems"],
   	(
   		If[!(IntegerQ["MaxItems"/.newparams]&&("MaxItems"/.newparams)>=0&&("MaxItems"/.newparams)<=100),
		(	
			Message[ServiceExecute::nval,"MaxItems","MailChimp"];
			Throw[$Failed]
		)];
     	limit = ToString["MaxItems" /. newparams];                        
  	),
  	(
  		limit="10"
  	)];
  	If[KeyExistsQ[newparams,IgnoreCase],
	(
		If[!MemberQ[{True,False},IgnoreCase /. newparams],
		(	
			Message[ServiceExecute::nval,IgnoreCase,"MailChimp"];
			Throw[$Failed]
		)];
		ignore = IgnoreCase /. newparams                        
	),
	(
		ignore=True
	)];
 	If[ KeyExistsQ[newparams,"Status"],
  	(
		If[!StringMatchQ[ToString["Status" /. newparams], "Sent" | "Hard" | "Soft", IgnoreCase -> ignore],
		(	
			Message[ServiceExecute::nval,"Status","MailChimp"];
			Throw[$Failed]
		)];  	
		status = ToLowerCase["Status" /. newparams];                        
	)];
 	data = Replace[FixedPoint[Normal,ServiceExecute["MailChimp", "RawReportSentTo", DeleteCases[{"cid" -> campaignid ,"opts[start]" -> start, "opts[limit]"-> limit,"opts[status]"-> status}, _ -> Null]]],Null->Missing["NotAvailable"],Infinity];
  	If[("status"/.data)=="error",
   	(
      	Message[ServiceExecute::serrormsg,("error"/.data)];
       	Throw[$Failed]
 	)];
  	withCamelTitles=Replace["data" /. data, Rule[a_, b_] :> Rule[camelCase[a], b], Infinity]
  	/. {(y : "TimestampOpt" | "TimestampSignup" | "Timestamp"|"InfoChanged" -> x_) :> (y -> If[MatchQ[x, _String], DateObject[x, TimeZone -> 0], x]),
  		"TzGroup"->"TimeZoneGroup","AbsplitGroup"->"ABSplitGroup",
	 	"IpSignup"->"IPSignUp","Id"->"ID","IpOpt"->"IPOpt","WebId"->"WebID","Euid"->"EUID","Leid"->"LEID",
	 	"ListId"->"ListID","IconUrl"->"IconURL","Dstoff"->"DSTOff","Cc"->"CountryCode","Gmtoff"->"GMTOff","EMAIL"->"Email","FNAME"->"FirstName",
	 	"LNAME"->"LastName","Timezone"->"TimeZone"
	 	}
	  /. {"TimestampOpt"->"TimeStampOpt","TimestampSignup"->"TimeStampSignUp","Timestamp"->"TimeStamp"
	 	};
  	Dataset[Replace[withCamelTitles, {r__Rule} :> Association[r], -1]]
]

mailchimpcookeddata["CampaignOpenedBy",id_,args_] :=Block[ {data,invalidParameters,ignore,withCamelTitles,campaignid,start = Null,limit = Null,sortfield = Null,newparams},
	newparams=args/.{MaxItems:>"MaxItems"};
	invalidParameters = Select[Keys[newparams],!MemberQ[{"CampaignID","SortField","MaxItems","StartIndex",IgnoreCase},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"MailChimp"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[ KeyExistsQ[newparams,"CampaignID"],
 	(
   		campaignid = ToString["CampaignID" /. newparams];                        
   	),
	(
		Message[ServiceExecute::nparam,"CampaignID","MailChimp"];
		Throw[$Failed]
	)];
	If[ KeyExistsQ[newparams,"StartIndex"],
 	(
    	If[!(IntegerQ["StartIndex"/.newparams]&&("StartIndex"/.newparams)>0),
		(	
			Message[ServiceExecute::nval,"StartIndex","MailChimp"];
			Throw[$Failed]
		)];
    	start = ToString[("StartIndex" /. newparams)-1];                        
  	)];
  	If[ KeyExistsQ[newparams,"MaxItems"],
   	(
   		If[!(IntegerQ["MaxItems"/.newparams]&&("MaxItems"/.newparams)>=0&&("MaxItems"/.newparams)<=100),
		(	
			Message[ServiceExecute::nval,"MaxItems","MailChimp"];
			Throw[$Failed]
		)];
     	limit = ToString["MaxItems" /. newparams];                        
  	),
  	(
  		limit="10"
  	)];
  	If[KeyExistsQ[newparams,IgnoreCase],
	(
		If[!MemberQ[{True,False},IgnoreCase /. newparams],
		(	
			Message[ServiceExecute::nval,IgnoreCase,"MailChimp"];
			Throw[$Failed]
		)];
		ignore = IgnoreCase /. newparams                        
	),
	(
		ignore=True
	)];
  	If[ KeyExistsQ[newparams,"SortField"],
   	(
   		If[!StringMatchQ[ToString["SortField" /. newparams],  "Opened"| "Opens" , IgnoreCase -> ignore],
		(	
			Message[ServiceExecute::nval,"SortField","MailChimp"];
			Throw[$Failed]
		)];
   		sortfield = "SortField" /. newparams;                        
 	)];
  	data = Replace[FixedPoint[Normal,ServiceExecute["MailChimp", "RawReportOpened", DeleteCases[{"cid" -> campaignid ,"opts[start]" -> start, "opts[limit]"-> limit,"opts[sort_field]"-> sortfield,"opts[sort_dir]"->"ASC"}, _ -> Null]]],Null->Missing["NotAvailable"],Infinity];
   	If[("status"/.data)=="error",
   	(
      	Message[ServiceExecute::serrormsg,("error"/.data)];
       	Throw[$Failed]
 	)];
   	withCamelTitles=Replace["data" /. data, Rule[a_, b_] :> Rule[camelCase[a], b], Infinity]
   	/. {(y : "TimestampOpt" | "TimestampSignup" | "Timestamp"|"InfoChanged" -> x_) :> (y -> If[MatchQ[x, _String], DateObject[x, TimeZone -> 0], x]),
	 	"IpSignup"->"IPSignUp","Id"->"ID","IpOpt"->"IPOpt","WebId"->"WebID","Euid"->"EUID","Leid"->"LEID",
	 	"ListId"->"ListID","IconUrl"->"IconURL","Dstoff"->"DSTOff","Cc"->"CountryCode","Gmtoff"->"GMTOff","EMAIL"->"Email","FNAME"->"FirstName",
	 	"LNAME"->"LastName","Timezone"->"TimeZone"
	 	}
	  /. {"TimestampOpt"->"TimeStampOpt","TimestampSignup"->"TimeStampSignUp","Timestamp"->"TimeStamp"
	 	};
   	Dataset[Replace[withCamelTitles, {r__Rule} :> Association[r], -1]]
]

mailchimpcookeddata["CampaignNotOpenedBy",id_,args_] :=Block[ {data,invalidParameters,withCamelTitles,campaignid,start = Null,limit = Null,sortfield = Null,newparams},
	newparams=args/.{MaxItems:>"MaxItems"};
	invalidParameters = Select[Keys[newparams],!MemberQ[{"CampaignID","MaxItems","StartIndex"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"MailChimp"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[ KeyExistsQ[newparams,"CampaignID"],
 	(
   		campaignid = ToString["CampaignID" /. newparams];                        
 	),
	(
		Message[ServiceExecute::nparam,"CampaignID","MailChimp"];
		Throw[$Failed]
	)];
  	If[ KeyExistsQ[newparams,"StartIndex"],
 	(
    	If[!(IntegerQ["StartIndex"/.newparams]&&("StartIndex"/.newparams)>0),
		(	
			Message[ServiceExecute::nval,"StartIndex","MailChimp"];
			Throw[$Failed]
		)];
    	start = ToString[("StartIndex" /. newparams)-1];                        
  	)];
  	If[ KeyExistsQ[newparams,"MaxItems"],
   	(
   		If[!(IntegerQ["MaxItems"/.newparams]&&("MaxItems"/.newparams)>=0&&("MaxItems"/.newparams)<=100),
		(	
			Message[ServiceExecute::nval,"MaxItems","MailChimp"];
			Throw[$Failed]
		)];
     	limit = ToString["MaxItems" /. newparams];                        
  	),
  	(
  		limit="10"
  	)];
 	data = Replace[FixedPoint[Normal,ServiceExecute["MailChimp", "RawReportNotOpened", DeleteCases[{"cid" -> campaignid ,"opts[start]" -> start, "opts[limit]"-> limit}, _ -> Null]]],Null->Missing["NotAvailable"],Infinity];
  	If[("status"/.data)=="error",
   	(
      	Message[ServiceExecute::serrormsg,("error"/.data)];
       	Throw[$Failed]
 	)];
  	withCamelTitles=Replace["data" /. data, Rule[a_, b_] :> Rule[camelCase[a], b], Infinity]
  	/. {(y : "TimestampOpt" | "TimestampSignup" | "Timestamp"|"InfoChanged" -> x_) :> (y -> If[MatchQ[x, _String], DateObject[x, TimeZone -> 0], x]),
	 	"IpSignup"->"IPSignUp","Id"->"ID","IpOpt"->"IPOpt","WebId"->"WebID","Euid"->"EUID","Leid"->"LEID",
	 	"ListId"->"ListID","IconUrl"->"IconURL","Dstoff"->"DSTOff","Cc"->"CountryCode","Gmtoff"->"GMTOff","EMAIL"->"Email","FNAME"->"FirstName",
	 	"LNAME"->"LastName","Timezone"->"TimeZone"
	 	}
	  /. {"TimestampOpt"->"TimeStampOpt","TimestampSignup"->"TimeStampSignUp","Timestamp"->"TimeStamp"
	 	};
   	Dataset[Replace[withCamelTitles, {r__Rule} :> Association[r], -1]]
]

mailchimpcookeddata["CampaignUnsubscriptions",id_,args_] :=Block[ {data,invalidParameters,withCamelTitles,campaignid,start = Null,limit = Null,sortfield = Null,newparams},
	newparams=args/.{MaxItems:>"MaxItems"};
	invalidParameters = Select[Keys[newparams],!MemberQ[{"CampaignID","MaxItems","StartIndex"},#]&]; 
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"MailChimp"]&/@invalidParameters;
		Throw[$Failed]
	)];
	If[ KeyExistsQ[newparams,"CampaignID"],
 	(
   		campaignid = ToString["CampaignID" /. newparams];                        
 	),
	(
		Message[ServiceExecute::nparam,"CampaignID","MailChimp"];
		Throw[$Failed]
	)];
  	If[ KeyExistsQ[newparams,"StartIndex"],
 	(
    	If[!(IntegerQ["StartIndex"/.newparams]&&("StartIndex"/.newparams)>0),
		(	
			Message[ServiceExecute::nval,"StartIndex","MailChimp"];
			Throw[$Failed]
		)];
    	start = ToString[("StartIndex" /. newparams)-1];                        
  	)];
  	If[ KeyExistsQ[newparams,"MaxItems"],
   	(
   		If[!(IntegerQ["MaxItems"/.newparams]&&("MaxItems"/.newparams)>=0&&("MaxItems"/.newparams)<=100),
		(	
			Message[ServiceExecute::nval,"MaxItems","MailChimp"];
			Throw[$Failed]
		)];
     	limit = ToString["MaxItems" /. newparams];                        
  	),
  	(
  		limit="10"
  	)];
	data = Replace[FixedPoint[Normal,ServiceExecute["MailChimp", "RawReportUnsubscribes", DeleteCases[{"cid" -> campaignid ,"opts[start]" -> start, "opts[limit]"-> limit}, _ -> Null]]],Null->Missing["NotAvailable"],Infinity];
 	If[("status"/.data)=="error",
   	(
      	Message[ServiceExecute::serrormsg,("error"/.data)];
       	Throw[$Failed]
 	)];
 	withCamelTitles=Replace["data" /. data, Rule[a_, b_] :> Rule[camelCase[a], b], Infinity]
   	/. {(y : "TimestampOpt" | "TimestampSignup" | "Timestamp"|"InfoChanged" -> x_) :> (y -> If[MatchQ[x, _String], DateObject[x, TimeZone -> 0], x]),
	 	"IpSignup"->"IPSignUp","Id"->"ID","IpOpt"->"IPOpt","WebId"->"WebID","Euid"->"EUID","Leid"->"LEID",
	 	"ListId"->"ListID","IconUrl"->"IconURL","Dstoff"->"DSTOff","Cc"->"CountryCode","Gmtoff"->"GMTOff","EMAIL"->"Email","FNAME"->"FirstName",
	 	"LNAME"->"LastName","Timezone"->"TimeZone"
	 	}
	  /. {"TimestampOpt"->"TimeStampOpt","TimestampSignup"->"TimeStampSignUp","Timestamp"->"TimeStamp"
	 	};
 	Dataset[Replace[withCamelTitles, {r__Rule} :> Association[r], -1]]
]


mailchimpcookeddata[___] :=$Failed

mailchimprawdata[___] :=$Failed

mailchimpsendmessage[args_] :=$Failed

End[]

End[]

SetAttributes[{},{ReadProtected, Protected}];

(* Return two functions to define oauthservicedata, oauthcookeddata  *)

{MailChimpAPI`Private`mailchimpdata,MailChimpAPI`Private`mailchimpcookeddata,MailChimpAPI`Private`mailchimpsendmessage,MailChimpAPI`Private`mailchimprawdata}
