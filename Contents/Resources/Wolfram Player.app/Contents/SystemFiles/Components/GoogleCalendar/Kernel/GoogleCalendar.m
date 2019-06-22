Begin["GoogleCalendar`"]

Begin["`Private`"]

(******************************* GoogleCalendar *************************************)

(* Authentication information *)

googlecalendardata[]:=
	If[TrueQ[OAuthClient`Private`$UseChannelFramework],{	
    	"OAuthVersion"			-> "2.0",
		"ServiceName"			-> "GoogleCalendar",
	    "AuthorizeEndpoint"		-> "https://accounts.google.com/o/oauth2/v2/auth",
	    "AccessEndpoint"		-> "https://www.googleapis.com/oauth2/v4/token",
	    "RedirectURI" 			-> "WolframConnectorChannelListen",
        "Blocking"				-> False,
        "VerifierLabel"			-> "code",
        "ClientInfo"			-> {"Wolfram","Token"},
        "AuthorizationFunction"	-> "GoogleCalendar",
        "RedirectURLFunction"	-> (#1&),
		"AccessTokenExtractor"	-> "Refresh/2.0",
		"RefreshAccessTokenFunction" -> Automatic,
		"VerifyPeer"			-> True,
        "AuthenticationDialog"	:> "WolframConnectorChannel",
	 	"Gets"				-> {"CalendarList","CalendarDataset","CalendarInformation","EventList","EventDataset","EventInformation"},
	 	"Posts"				-> {},
	 	"RawGets"			-> {"RawCalendarList","RawCalendarDetails","RawEventList","RawEventDetails","RawUserSettings"},
	 	"RawPosts"			-> {},
	 	"Scope"				-> {"https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcalendar+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcalendar.readonly"},
 		"Information"		-> "A service for receiving data from Google Calendars"
},
{	
    	"OAuthVersion"			-> "2.0",
		"ServiceName"			-> "GoogleCalendar",
	    "AuthorizeEndpoint"		-> "https://accounts.google.com/o/oauth2/v2/auth",
	    "AccessEndpoint"		-> "https://www.googleapis.com/oauth2/v4/token",
        "RedirectURI"			-> "https://www.wolfram.com/oauthlanding/?service=GoogleCalendar",
        "VerifierLabel"			-> "code",
        "ClientInfo"			-> {"Wolfram","Token"},
        "AuthorizationFunction"	-> "GoogleCalendar",
		"AccessTokenExtractor"	-> "Refresh/2.0",
		"RefreshAccessTokenFunction" -> Automatic,
	 	"AuthenticationDialog" :> (OAuthClient`tokenOAuthDialog[#, "GoogleCalendar"]&),
	 	"Gets"				-> {"CalendarList","CalendarDataset","CalendarInformation","EventList","EventDataset","EventInformation"},
	 	"Posts"				-> {},
	 	"RawGets"			-> {"RawCalendarList","RawCalendarDetails","RawEventList","RawEventDetails","RawUserSettings"},
	 	"RawPosts"			-> {},
	 	"Scope"				-> {"https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcalendar+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcalendar.readonly"},
 		"Information"		-> "A service for receiving data from Google Calendars"
}]

(* a function for importing the raw data - usually json or xml - from the service *)

googlecalendarimport[raw_]:= With[{res = Quiet[Developer`ReadRawJSONString[raw]]},
	If[ AssociationQ[res],
		If[ !KeyExistsQ[res, "error"],
			res,
			Message[ServiceExecute::apierr, res["error"]["message"]];
			Throw[$Failed]
		],
		Message[ServiceExecute::serror];
		Throw[$Failed]
	]
]
 
(*** Raw ***) 

(* details about parameters here https://developers.google.com/apis-explorer/#s/calendar/v3/calendar.settings.list *)
googlecalendardata["RawUserSettings"] = {
        "URL"					-> "https://www.googleapis.com/calendar/v3/users/me/settings",
        "Parameters"			-> {"maxResults","pageToken","syncToken","fields"},
        "RequiredParameters"	-> {},
        "HTTPSMethod"			-> "GET",
        "ResultsFunction"		-> googlecalendarimport
    }

(* details about parameters here https://developers.google.com/apis-explorer/#s/calendar/v3/calendar.calendarList.list *)    
googlecalendardata["RawCalendarList"] = {
        "URL"					-> "https://www.googleapis.com/calendar/v3/users/me/calendarList",
        "Parameters"			-> {"maxResults","pageToken","showDeleted","showHidden","minAccessRole","syncToken","fields"},
        "RequiredParameters"	-> {},
        "HTTPSMethod"			-> "GET",
        "ResultsFunction"		-> googlecalendarimport
    }

(* details about parameters here https://developers.google.com/apis-explorer/#s/calendar/v3/calendar.calendars.get *)
googlecalendardata["RawCalendarDetails"] = {
        "URL"					-> (ToString@StringForm["https://www.googleapis.com/calendar/v3/calendars/`1`",URLEncode[#]]&),
        "PathParameters"		-> {"calendarID"},
        "Parameters"			-> {"fields"},
        "RequiredParameters"	-> {"calendarID"},
        "HTTPSMethod"			-> "GET",
        "ResultsFunction"		-> googlecalendarimport
    }

(* details about parameters here https://developers.google.com/apis-explorer/#s/calendar/v3/calendar.events.list *)    
googlecalendardata["RawEventList"] = {
        "URL"					-> (ToString@StringForm["https://www.googleapis.com/calendar/v3/calendars/`1`/events",#]&),
        "PathParameters"		-> {"calendarID"},
        "Parameters"			-> {"orderBy","maxResults","pageToken","maxAttendees","alwaysIncludeEmail","iCalUID","privateExtendedProperty",
        							"q","sharedExtendedProperty","showDeleted","showHiddenInvitations","singleEvents","syncToken",
        							"timeMax","timeMin","timeZone","updatedMin","fields"},
        "RequiredParameters"	-> {"calendarID"},
        "HTTPSMethod"			-> "GET",
        "ResultsFunction"		-> googlecalendarimport
    }

(* details about parameters here https://developers.google.com/apis-explorer/#s/calendar/v3/calendar.events.get *)    
googlecalendardata["RawEventDetails"] = {
        "URL"					-> (ToString@StringForm["https://www.googleapis.com/calendar/v3/calendars/`1`/events/`2`",##]&),
        "PathParameters"		-> {"calendarID","eventID"},
        "Parameters"			-> {"alwaysIncludeEmail","maxAttendees","timeZone","fields"},
        "RequiredParameters"	-> {"calendarID","eventID"},
        "HTTPSMethod"			-> "GET",
        "ResultsFunction"		-> googlecalendarimport
    }

googlecalendardata[___]:=$Failed

(* Cooked *)
googlecalendarcookeddata[prop:("CalendarList"|"CalendarDataset"), id_, args_] := Block[{args2=args,params={},sd,sh,rawdata,invalidParameters,limit=100,maxPerPage=250,
											fieldnames,timezones,fixtimezones,result={},calls,nextPageToken,moreResultsAvailable,query,selection={}},
		invalidParameters = Select[Keys[args],!MemberQ[{"MaxItems",MaxItems,"ShowHidden","ShowDeleted","Query"},#]&]; 
	
		If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,prop]&/@invalidParameters;
			Throw[$Failed]
		)];	
		
		If[KeyExistsQ[args,"MaxItems"],AppendTo[args2,MaxItems->Lookup[args,"MaxItems"]]];

		If[KeyExistsQ[args2,MaxItems],
		(
			limit = Lookup[args2,MaxItems];
			If[!IntegerQ[limit] || NonPositive[limit],
			(	
				Message[ServiceExecute::nval,"MaxItems","GoogleCalendar"];
				Throw[$Failed]
			)];	
			AppendTo[params,"maxResults"->ToString[Min[limit,maxPerPage]]];					
		)];
		
		If[KeyExistsQ[args,"ShowDeleted"],
		(
			sd = Lookup[args,"ShowDeleted"];
			Switch[sd,
				True,
				AppendTo[params,"showDeleted"->"True"],
				False,
				AppendTo[params,"showDeleted"->"False"],
				_,
				(
					Message[ServiceExecute::nval,"ShowDeleted","GoogleCalendar"];	
					Throw[$Failed]
				)
			];		
		)];
		
		If[KeyExistsQ[args,"ShowHidden"],
		(
			sh = Lookup[args,"ShowHidden"];
			Switch[sh,
				True,
				AppendTo[params,"showHidden"->"True"],
				False,
				AppendTo[params,"showHidden"->"False"],
				_,
				(
					Message[ServiceExecute::nval,"ShowHidden","GoogleCalendar"];	
					Throw[$Failed]
				)
			];		
		)];
		
		calls = Ceiling[limit/maxPerPage];
		
		rawdata = googlecalendarimport@OAuthClient`rawoauthdata[id,"RawCalendarList",params];
		
		result = Join[result,Lookup[rawdata,"items"]];		
		
		If[calls > 1,
			moreResultsAvailable = KeyExistsQ[rawdata,"nextPageToken"];
			AppendTo[params,"pageToken"->""];
		(
			If[moreResultsAvailable,
			(
				nextPageToken = Lookup[rawdata,"nextPageToken"];
				params = ReplaceAll[params,Rule["pageToken",_]:>Rule["pageToken",nextPageToken]];
				rawdata = googlecalendarimport@OAuthClient`rawoauthdata[id,"RawCalendarList",params];
				moreResultsAvailable = KeyExistsQ[rawdata,"nextPageToken"];
				result = Join[result,Lookup[rawdata,"items"]];		
			)]
			
		)&/@Range[calls];
		];
		
		result = Take[result,UpTo[limit]];

		fieldnames = {"id","summary","description","location","etag","timeZone"};
		result = KeyTake[fieldnames] /@ result;
		result = KeyMap[Replace[{"id"->"ID","summary"->"Summary","description"->"Description","location"->"Location","timeZone"->"TimeZone","etag"->"ETag"}]] /@ result;
		timezones = result[[All, Key["TimeZone"]]];
		fixtimezones = Replace[(# -> Interpreter["TimeZone"][#]&) /@ DeleteDuplicates@DeleteMissing@timezones] /@ timezones;
		
		result[[All, Key["TimeZone"]]] = fixtimezones;
		
		If[KeyExistsQ[args,"Query"],
		(
			query = "Query" /. args;
			selection = Select[result, 
				Or[
					StringContainsQ[Lookup[#,"Summary",""], query, IgnoreCase->True],
					StringContainsQ[Lookup[#,"Description",""], query, IgnoreCase->True]
				]&]
		)];

		If[ SameQ[prop, "CalendarList"],
			result,
			Dataset[result]
		]	
]

googlecalendarcookeddata["CalendarInformation", id_, args_] := Block[{rawdata, invalidParameters,cId,fieldnames,result},
		invalidParameters = Select[Keys[args],!MemberQ[{"CalendarID"},#]&]; 
	
		If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"CalendarInformation"]&/@invalidParameters;
			Throw[$Failed]
		)];	
	
		If[KeyExistsQ[args,"CalendarID"],
			(
				cId = Lookup[args,"CalendarID"];
				If[!MatchQ[cId,_String],
				(	
					Message[ServiceExecute::nval,"CalendarID","GoogleCalendar"];
					Throw[$Failed]
				)]
			),
			(
				Message[ServiceExecute::nparam,"CalendarID"];			
				Throw[$Failed]
			)
		];
		
		rawdata = OAuthClient`rawoauthdata[id,"RawCalendarDetails",{"calendarID"->ToString[cId]}];
		result = googlecalendarimport[rawdata];

		fieldnames = {"id","summary","description","location","etag","timeZone"};
		result = KeyTake[fieldnames] @ result;
		result = KeyMap[Replace[{"id"->"ID","summary"->"Summary","description"->"Description","location"->"Location","timeZone"->"TimeZone","etag"->"ETag"}]] @ result;
		result[[Key["TimeZone"]]] = Replace[result[[Key["TimeZone"]]], timezone:Except[_Missing]:>Interpreter["TimeZone"][timezone]];

		Association[result]
]

googlecalendarcookeddata[prop:("EventList"|"EventDataset"), id_, args_] := Block[{args2=args,cId,maxAttendees,sort,query,params={},sd,rawdata,invalidParameters,limit=250,maxPerPage=2500,
											fieldnames,camelfields,result,updateMin,singleEvents=False,start,end,startIndex,
											startDate,endDate,dateRange,tmp,calls,nextPageToken,moreResultsAvailable},
		invalidParameters = Select[Keys[args],!MemberQ[{"CalendarID","MaxAttendees","MaxItems",MaxItems,"ShowHiddenInvitations","ShowDeleted","SortBy","Query",
														"ExpandRecurringEvents","Date","UpdatedMin","StartIndex"},#]&]; 
	
		If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,prop]&/@invalidParameters;
			Throw[$Failed]
		)];

		If[KeyExistsQ[args,"CalendarID"],
			cId = Lookup[args,"CalendarID"],
			cId = "primary"
		];
		AppendTo[params,"calendarID"->ToString[cId]];
		
		If[KeyExistsQ[args,"MaxAttendees"],
		(
			maxAttendees = Lookup[args,"MaxAttendees"];
			If[!IntegerQ[maxAttendees] || NonPositive[maxAttendees],
			(	
				Message[ServiceExecute::nval,"MaxAttendees","GoogleCalendar"];
				Throw[$Failed]
			)];	
			AppendTo[params,"maxAttendees"->ToString[maxAttendees]];			
		)];
		
		If[KeyExistsQ[args,"MaxItems"],AppendTo[args2,MaxItems->Lookup[args,"MaxItems"]]];

		If[KeyExistsQ[args2,MaxItems],
		(
			limit = Lookup[args2,MaxItems];
			If[!IntegerQ[limit] || NonPositive[limit],
			(	
				Message[ServiceExecute::nval,"MaxItems","GoogleCalendar"];
				Throw[$Failed]
			)];	
			AppendTo[params,"maxResults"->ToString[Min[limit,maxPerPage]]];					
		)];
		
		If[KeyExistsQ[args,"StartIndex"],
		(
			startIndex = Lookup[args,"StartIndex"];
			If[!IntegerQ[startIndex] || startIndex < 1,
			(	
				Message[ServiceExecute::nval,"StartIndex","GoogleCalendar"];
				Throw[$Failed]
			)];	
			limit = limit + startIndex - 1;
			params = ReplaceAll[params,Rule["maxResults",_]:>Rule["maxResults",ToString[Min[limit,maxPerPage]]]];				
		),
			startIndex = 1
		];
		
		If[KeyExistsQ[args,"Query"],
		(
			query = Lookup[args,"Query"];
			AppendTo[params,"q"->query];			
		)];
		
		If[KeyExistsQ[args,"ShowDeleted"],
		(
			sd = Lookup[args,"ShowDeleted"];
			Switch[sd,
				True,
				AppendTo[params,"showDeleted"->"True"],
				False,
				AppendTo[params,"showDeleted"->"False"],
				_,
				(
					Message[ServiceExecute::nval,"ShowDeleted","GoogleCalendar"];	
					Throw[$Failed]
				)
			];		
		)];
		
		If[KeyExistsQ[args,"ShowHiddenInvitations"],
		(
			sd = Lookup[args,"ShowHiddenInvitations"];
			Switch[sd,
				True,
				AppendTo[params,"showHiddenInvitations"->"True"],
				False,
				AppendTo[params,"showHiddenInvitations"->"False"],
				_,
				(
					Message[ServiceExecute::nval,"ShowHiddenInvitations","GoogleCalendar"];	
					Throw[$Failed]
				)
			];		
		)];
		
		If[KeyExistsQ[args,"ExpandRecurringEvents"],
		(
			singleEvents = Lookup[args,"ExpandRecurringEvents"];
			Switch[singleEvents,
				True,
				AppendTo[params,"singleEvents"->"True"],
				False,
				AppendTo[params,"singleEvents"->"False"],
				_,
				(
					Message[ServiceExecute::nval,"ExpandRecurringEvents","GoogleCalendar"];	
					Throw[$Failed]
				)
			];		
		)];
		
		If[KeyExistsQ[args,"SortBy"],
		(
			sort = Lookup[args,"SortBy"];
			Switch[sort,
				"StartTime" && singleEvents,
				AppendTo[params,"orderBy"->"startTime"],
				"Updated",
				AppendTo[params,"orderBy"->"updated"],
				_,
				(
					Message[ServiceExecute::nval,"SortBy","GoogleCalendar"];	
					Throw[$Failed]
				)
			];			
		)];
		
		If[KeyExistsQ[args,"Date"],
		(
			dateRange = Lookup[args,"Date"];
		
			Switch[dateRange,
				_String,
				(
					startDate = DateObject[dateRange];
					tmp = DateList[startDate];
					startDate = DateObject[Join[tmp[[1;;3]],{0,0,0.}]];
					endDate = DateObject[Join[tmp[[1;;3]],{23,59,59.}]];
				),
				_DateObject,
				(
					tmp = DateList[dateRange];
					startDate = DateObject[Join[tmp[[1;;3]],{0,0,0.}]];
					endDate = DateObject[Join[tmp[[1;;3]],{23,59,59.}]];
				),
				List[_,_],
				(
					startDate = dateRange[[1]];
					endDate = dateRange[[2]];
		
					Switch[startDate,
						_String|{Repeated[_?NumberQ, 6]},
						startDate = DateObject[startDate],
						DateObject,
						startDate
					];
					Switch[endDate,
						_String|{Repeated[_?NumberQ, 6]},
						endDate = DateObject[endDate],
						DateObject,
						endDate
					];
				),
				Interval[{_DateObject,_DateObject}],
				(
					startDate = dateRange /. Interval[{f_,t_}]:>f;
					endDate = dateRange /. Interval[{f_,t_}]:>t;				
				),
				_,
				(
					Message[ServiceExecute::nval,"Date","GoogleCalendar"];	
					Throw[$Failed]
				)
			];
		
			If[!DateObjectQ[startDate],
			(
				Message[ServiceExecute::nval,"Date","GoogleCalendar"];	
				Throw[$Failed]
			)];
		
			If[!DateObjectQ[endDate],
			(
				Message[ServiceExecute::nval,"Date","GoogleCalendar"];	
				Throw[$Failed]
			)];
		
			startDate = DateString[startDate, "ISODateTime"] <> "Z";
		
			endDate = DateString[endDate, "ISODateTime"] <> "Z";
				
			params = Join[params,{Rule["timeMax",endDate],Rule["timeMin",startDate]}];		
		
		)];
		
		If[KeyExistsQ[args,"UpdatedMin"],
		(
			updateMin = Lookup[args,"UpdatedMin"];
			If[!DateObjectQ[updateMin],
			(	
				Message[ServiceExecute::nval,"UpdatedMin","GoogleCalendar"];
				Throw[$Failed]
			)];
			updateMin = DateString[updateMin, "ISODateTime"] <> "Z";
			params = Append[params,"updatedMin"->updateMin];			
		)];

		calls = Ceiling[limit/maxPerPage];

		rawdata = googlecalendarimport@OAuthClient`rawoauthdata[id,"RawEventList",params];

		result = Lookup[rawdata,"items"];		
		moreResultsAvailable = KeyExistsQ[rawdata,"nextPageToken"];
		AppendTo[params,"pageToken"->""];

		If[calls > 1,
		(
			If[moreResultsAvailable,
			(
				nextPageToken = Lookup[rawdata,"nextPageToken"];
				params = ReplaceAll[params,Rule["pageToken",_]:>Rule["pageToken",nextPageToken]];
				rawdata = googlecalendarimport@OAuthClient`rawoauthdata[id,"RawEventList",params];
				moreResultsAvailable = KeyExistsQ[rawdata,"nextPageToken"];
				result = Join[result,Lookup[rawdata,"items"]];		
			)]
			
		)& /@ Range[calls]
		];

		result = Take[result,{Min[startIndex,Min[limit,Length[result]]+1],UpTo[limit]}];

		fieldnames = {"start","end","location","summary","created","organizer","attendees","updated","id","etag",
					"status","htmlLink","description","creator","endTimeUnspecified","recurrence","originalStartTime",
					"iCalUID","sequence","reminders","source","attachments"};

		camelfields = {"start" -> "Start", "end" -> "End", "location" -> "Location", "summary" -> "Summary", "created" -> "Created", "organizer" -> "Organizer",
		"attendees" -> "Attendees", "updated" -> "Updated", "id" -> "ID", "etag" -> "ETag", "status" -> "Status", "htmlLink" -> "HTMLLink", "description" -> "Description",
		"creator" -> "Creator", "endTimeUnspecified" -> "EndTimeUnspecified", "recurrence" -> "Recurrence", "originalStartTime" -> "OriginalStartTime", "iCalUID" -> "ICalUID",
		"sequence" -> "Sequence", "reminders" -> "Reminders", "source" -> "Source", "attachments" -> "Attachments", s_?StringQ :> camelCase[s]};
					
		result = KeyTake[fieldnames] /@ result;
		result = Replace[result, asoc_?AssociationQ :> KeyMap[Replace[camelfields]][asoc], {0, Infinity}];
		result = If[KeyExistsQ["Start"][#], MapAt[formatCalendarStartEnd, Key["Start"]][#], #] & /@ result;
		result = If[KeyExistsQ["End"][#], MapAt[formatCalendarStartEnd, Key["End"]][#], #] & /@ result;
		result = If[KeyExistsQ["Updated"][#], MapAt[formatCalendarDate, Key["Updated"]][#], #] & /@ result;
		result = If[KeyExistsQ["Created"][#], MapAt[formatCalendarDate, Key["Created"]][#], #] & /@ result;
		result = If[KeyExistsQ["OriginalStartTime"][#], MapAt[formatCalendarStartEnd, Key["OriginalStartTime"]][#], #] & /@ result;

		If[SameQ[prop,"EventList"],		
			result,
			Dataset[result]
		]	
]

googlecalendarcookeddata["EventInformation", id_, args_] := Block[{params={},rawdata, invalidParameters,cId,eId,maxAttendees,fieldnames,camelfields,result,start,end},
		invalidParameters = Select[Keys[args],!MemberQ[{"CalendarID","EventID","MaxAttendees"},#]&]; 
	
		If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"EventInformation"]&/@invalidParameters;
			Throw[$Failed]
		)];	
	
		If[KeyExistsQ[args,"CalendarID"],
			(
				cId = Lookup[args,"CalendarID"];
				If[!MatchQ[cId,_String],
				(	
					Message[ServiceExecute::nval,"CalendarID","GoogleCalendar"];
					Throw[$Failed]
				)]
			),
			cId = "primary"
		];
		AppendTo[params,"calendarID"->ToString[cId]];
		
		If[KeyExistsQ[args,"EventID"],
			(
				eId = Lookup[args,"EventID"];
				If[!MatchQ[eId,_String],
				(	
					Message[ServiceExecute::nval,"EventID","GoogleCalendar"];
					Throw[$Failed]
				)]
			),
			(
				Message[ServiceExecute::nparam,"EventID"];			
				Throw[$Failed]
			)
		];
		AppendTo[params,"eventID"->ToString[eId]];
		
		If[KeyExistsQ[args,"MaxAttendees"],
		(
			maxAttendees = Lookup[args,"MaxAttendees"];
			If[!IntegerQ[maxAttendees],
			(	
				Message[ServiceExecute::nval,"MaxAttendees","GoogleCalendar"];
				Throw[$Failed]
			)];	
			AppendTo[params,"maxAttendees"->ToString[maxAttendees]];			
		)];
		
		rawdata = OAuthClient`rawoauthdata[id,"RawEventDetails",params];
		result = googlecalendarimport[rawdata];

		fieldnames = {"start","end","location","summary","created","organizer","attendees","updated","id","etag",
					"status","htmlLink","description","creator","endTimeUnspecified","recurrence","originalStartTime",
					"iCalUID","sequence","reminders","source","attachments"};
		camelfields = {"start" -> "Start", "end" -> "End", "location" -> "Location", "summary" -> "Summary", "created" -> "Created", "organizer" -> "Organizer",
		"attendees" -> "Attendees", "updated" -> "Updated", "id" -> "ID", "etag" -> "ETag", "status" -> "Status", "htmlLink" -> "HTMLLink", "description" -> "Description",
		"creator" -> "Creator", "endTimeUnspecified" -> "EndTimeUnspecified", "recurrence" -> "Recurrence", "originalStartTime" -> "OriginalStartTime", "iCalUID" -> "ICalUID",
		"sequence" -> "Sequence", "reminders" -> "Reminders", "source" -> "Source", "attachments" -> "Attachments", s_?StringQ :> camelCase[s]};


		result = KeyTake[fieldnames] @ result;
		result = Replace[result, asoc_?AssociationQ :> KeyMap[Replace[camelfields]][asoc], {0, Infinity}];

		result = If[KeyExistsQ["Start"][#], MapAt[formatCalendarStartEnd, Key["Start"]][#], #]& @ result;
		result = If[KeyExistsQ["End"][#], MapAt[formatCalendarStartEnd, Key["End"]][#], #]& @ result;
		result = If[KeyExistsQ["Updated"][#], MapAt[formatCalendarDate, Key["Updated"]][#], #]& @ result;
		result = If[KeyExistsQ["Created"][#], MapAt[formatCalendarDate, Key["Created"]][#], #]& @ result;
		result = If[KeyExistsQ["OriginalStartTime"][#], MapAt[formatCalendarStartEnd, Key["OriginalStartTime"]][#], #]& @ result;
		
		result	
]

(* Send Message *)

googlecalendarsendmessage[___]:=$Failed

(*** Utilities ***)
camelCase[text_] := Block[{split, partial}, (
    split = StringSplit[text, {" ","_","-"}];
    partial = Prepend[Rest[Characters[#]], ToUpperCase[Characters[#][[1]]]] & /@ split;
    partial = StringJoin[partial];
    partial = StringReplace[partial,RegularExpression["[Uu][Rr][Ll]"]->"URL"];
    partial = StringReplace[partial,RegularExpression["^[Ii][Dd]$"]->"ID"];
    partial
    )]

formatCalendarStartEnd[date_]:= Block[{d, tz}, (* tz stands for timezone*)
	Which[
		KeyExistsQ[date,"Date"],
			DateObject[date["Date"]],
		KeyExistsQ[date,"DateTime"],
			d = First@StringCases[date["DateTime"], RegularExpression["\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\w?"]];
			tz = StringDelete[date["DateTime"], d];
			If[KeyExistsQ[date,"TimeZone"],
				tz = date["TimeZone"],
				If[StringLength[tz] > 0,
					tz = StringSplit[tz, ":"];
					tz = Internal`StringToDouble[tz[[1]]],
					tz = 0
				]
			];
			DateObject[d, TimeZone -> tz]
	]
]

formatCalendarDate[date_]:= DateObject[StringReplace[date, d:RegularExpression["\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\w?"] ~~ ___ :> d]]

End[] (* End Private Context *)
           		
End[]

SetAttributes[{},{ReadProtected, Protected}];

(* Return three functions to define oauthservicedata, oauthcookeddata, oauthsendmessage  *)
{GoogleCalendar`Private`googlecalendardata,GoogleCalendar`Private`googlecalendarcookeddata,GoogleCalendar`Private`googlecalendarsendmessage}
