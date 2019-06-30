Begin["GoogleContacts`"]

Begin["`Private`"]

(******************************* GoogleContacts *************************************)

(* Authentication information *)

googlecontactsdata[]:=
	If[TrueQ[OAuthClient`Private`$UseChannelFramework],{
    	"OAuthVersion"			-> "2.0",
		"ServiceName"			-> "GoogleContacts",
	    "AuthorizeEndpoint"		-> "https://accounts.google.com/o/oauth2/v2/auth",
	    "AccessEndpoint"		-> "https://www.googleapis.com/oauth2/v4/token",
	    "RedirectURI" 			-> "WolframConnectorChannelListen",
        "Blocking"				-> False,
        "VerifierLabel"			-> "code",
        "ClientInfo"			-> {"Wolfram","Token"},
        "AuthorizationFunction"	-> "GoogleContacts",
        "RedirectURLFunction"	-> (#1&),
		"AccessTokenExtractor"	-> "Refresh/2.0",
		"RefreshAccessTokenFunction" -> Automatic,
		"VerifyPeer"			-> True,
        "AuthenticationDialog"	:> "WolframConnectorChannel",
	 	"AuthenticationDialog" :> "WolframConnectorChannel",
	 	"Gets"				-> {"ContactsList","ContactsDataset","GroupList","GroupDataset","ContactInformation","ContactPhoto"},
	 	"Posts"				-> {},
	 	"RawGets"			-> {"RawContacts","RawContactDetails","RawGroups","RawContactPhoto"},
	 	"RawPosts"			-> {},
	 	"Scope"				-> {"https%3A%2F%2Fwww.google.com%2Fm8%2Ffeeds+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcontacts.readonly"},
 		"Information"		-> "A service for receiving data from Google Contacts"
},
{
    	"OAuthVersion"			-> "2.0",
		"ServiceName"			-> "GoogleContacts",
	    "AuthorizeEndpoint"		-> "https://accounts.google.com/o/oauth2/v2/auth",
	    "AccessEndpoint"		-> "https://www.googleapis.com/oauth2/v4/token",
        "RedirectURI"			-> "https://www.wolfram.com/oauthlanding/?service=GoogleContacts",
        "VerifierLabel"			-> "code",
        "ClientInfo"			-> {"Wolfram","Token"},
        "AuthorizationFunction"	-> "GoogleContacts",
		"AccessTokenExtractor"	-> "Refresh/2.0",
		"RefreshAccessTokenFunction" -> Automatic,
	 	"AuthenticationDialog" :> (OAuthClient`tokenOAuthDialog[#, "GoogleContacts"]&),
	 	"Gets"				-> {"ContactsList","ContactsDataset","GroupList","GroupDataset","ContactInformation","ContactPhoto"},
	 	"Posts"				-> {},
	 	"RawGets"			-> {"RawContacts","RawContactDetails","RawGroups","RawContactPhoto"},
	 	"RawPosts"			-> {},
	 	"Scope"				-> {"https%3A%2F%2Fwww.google.com%2Fm8%2Ffeeds+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcontacts.readonly"},
 		"Information"		-> "A service for receiving data from Google Contacts"
}]

(* a function for importing the raw data - usually json or xml - from the service *)

googlecontactsimport[$Failed]:=Throw[$Failed]
googlecontactsimport[raw_]:=raw
googlecontactsimportphoto[raw_]:=ImportString[FromCharacterCode[raw],"Image"]

formatresults[rawdata_] := ImportString[ToString[rawdata,CharacterEncoding->"UTF-8"],"JSON"]

(*** Raw ***) 
googlecontactsdata["RawContacts"] = {
        "URL"					-> (ToString@StringForm["https://www.google.com/m8/feeds/contacts/`1`/full",#]&),
        "PathParameters"		-> {"userEmail"},
        "Parameters"			-> {"max-results","start-index","updated-min","alt","q","orderby","showdeleted","requirealldeleted","sortorder","group"},
        "RequiredParameters"	-> {"userEmail"},
        "HTTPSMethod"			-> "GET",
        "Headers"				-> {"GData-Version"->"3.0"},
        "ResultsFunction"		-> googlecontactsimport
    }

googlecontactsdata["RawContactDetails"] = {
        "URL"					-> (ToString@StringForm["https://www.google.com/m8/feeds/contacts/`1`/full/`2`",##]&),
        "PathParameters"		-> {"userEmail","contactID"},
        "Parameters"			-> {"alt"},
        "RequiredParameters"	-> {"userEmail","contactID"},
        "HTTPSMethod"			-> "GET",
        "Headers"				-> {"GData-Version"->"3.0"},
        "ResultsFunction"		-> googlecontactsimport
    }
    
googlecontactsdata["RawContactPhoto"] = {
        "URL"					-> (ToString@StringForm["https://www.google.com/m8/feeds/photos/media/`1`/`2`",##]&),
        "PathParameters"		-> {"userEmail","contactID"},
        "Parameters"			-> {"alt"},
        "RequiredParameters"	-> {"userEmail","contactID"},
        "HTTPSMethod"			-> "GET",
        "ReturnContentData"		-> True,
        "Headers"				-> {"GData-Version"->"3.0"},
        "ResultsFunction"		-> googlecontactsimportphoto
    }

googlecontactsdata["RawGroups"] = {
        "URL"					-> (ToString@StringForm["https://www.google.com/m8/feeds/groups/`1`/full",#]&),
        "PathParameters"		-> {"userEmail"},
        "Parameters"			-> {"max-results","start-index","updated-min","alt","q","orderby","showdeleted","requirealldeleted","sortorder"},
        "RequiredParameters"	-> {"userEmail"},
        "HTTPSMethod"			-> "GET",
        "Headers"				-> {"GData-Version"->"3.0"},
        "ResultsFunction"		-> googlecontactsimport
    }

googlecontactsdata[___]:=$Failed

(* Cooked *)
googlecontactscookeddata[prop:("ContactsList"|"ContactsDataset"), id_, args_] := Module[{params={},date,query,sort,sortVal,sortDir,sd,group,rawdata,invalidParameters,limit,defaultPerPage=25,maxPerPage=250,startIndex,
											calls,residual,progress,data,fieldnames,result,totalResults,items={}},
		invalidParameters = Select[Keys[args],!MemberQ[{"MaxItems",MaxItems,"StartIndex","UpdatedDate","Query","SortBy","ShowDeleted","GroupID"},#]&]; 

		If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,prop]&/@invalidParameters;
			Throw[$Failed]
		)];

		If[KeyExistsQ[args,"MaxItems"],AppendTo[args,MaxItems->Lookup[args,"MaxItems"]]];
		params = Join[params,{"userEmail"->"default","alt"->"json"}];

		If[KeyExistsQ[args,"UpdatedDate"], (* minimum update date *)
		(
			date = Lookup[args,"UpdatedDate"];
			If[!DateObjectQ[date],
			(	
				Message[ServiceExecute::nval,"UpdatedDate","GoogleContacts"];
				Throw[$Failed]
			)];	
			date = DateString[date, "ISODateTime"];
			AppendTo[params,"updated-min"->date];
		)];

		If[KeyExistsQ[args,"Query"],
		(
			query = Lookup[args,"Query"];
			AppendTo[params,"q"->query];
		)];

		If[KeyExistsQ[args,"SortBy"],
		(
			sort = Lookup[args,"SortBy"];
			If[MatchQ[sort, {_String,_String}],
			(
				If[MatchQ[sort[[2]],"Ascending"|"Descending"],
					sortDir = ToLowerCase[sort[[2]]],
						(
							Message[ServiceExecute::nval,"SortBy","GoogleContacts"];	
							Throw[$Failed]
						)
				];	
				AppendTo[params,Rule["sortorder",sortDir]];
				sort = sort[[1]];
			)];
			Switch[sort,
				"LastModified",
				sortVal = "lastmodified",
				_,
				(
					Message[ServiceExecute::nval,"SortBy","GoogleContacts"];	
					Throw[$Failed]
				)
			];
			AppendTo[params,"orderby"->"lastmodified"];
		)];

		If[KeyExistsQ[args,"ShowDeleted"],
		(
			sd = Lookup[args,"ShowDeleted"];
			Switch[sd,
				True,
				AppendTo[params,"showdeleted"->"true"],
				False,
				AppendTo[params,"showdeleted"->"false"],
				_,
				(
					Message[ServiceExecute::nval,"ShowDeleted","GoogleContacts"];	
					Throw[$Failed]
				)
			];		
		)];

		If[KeyExistsQ[args,"GroupID"],
		(
			group = Lookup[args,"GroupID"];
			AppendTo[params,"group"->group];			
		)];

		If[KeyExistsQ[args,MaxItems],
		(
			limit = Lookup[args,MaxItems];
			If[!IntegerQ[limit],
			(
				Message[ServiceExecute::nval,"MaxItems","GoogleContacts"];
				Throw[$Failed]
			)];
		),
			limit = defaultPerPage;
		];
	
		If[KeyExistsQ[args,"StartIndex"],
		(
			startIndex = Lookup[args,"StartIndex"];
			If[!IntegerQ[startIndex],
			(	
				Message[ServiceExecute::nval,"StartIndex","GoogleContacts"];
				Throw[$Failed]
			)];
		),
			startIndex = 1		
		];
		
		calls = Quotient[limit, maxPerPage];	
		residual = limit - (calls*maxPerPage);
	
		params = Join[params,{"max-results"->ToString[maxPerPage], "start-index"->ToString[startIndex]}];
	
		(* this prints the progress indicator bar *)
		PrintTemporary[ProgressIndicator[Dynamic[progress], {0, calls}]];
	
		If[calls > 0,
		(
			(	
				params = ReplaceAll[params,Rule["start-index",_] -> Rule["start-index",ToString[startIndex+#*maxPerPage]]];
				
				rawdata = OAuthClient`rawoauthdata[id,"RawContacts",params];
				data = formatresults[rawdata];
				
				If[KeyExistsQ[data,"error"],
				(
					Message[ServiceExecute::serrormsg,Lookup[Lookup[data,"error"],"message"]];
					Throw[$Failed]
				)];
				
				data = Lookup[data,"feed"];
				totalResults = FromDigits[Lookup[Lookup[data,"openSearch$totalResults"],"$t"]];
				If[KeyExistsQ[data,"entry"],
					items = Join[items, If[totalResults>0,Lookup[data,"entry"],{}]]
				];
				progress = progress + 1;
			)& /@ Range[0,calls-1];
		
		)];
	
		If[residual > 0,
		(
			params = ReplaceAll[params,Rule["start-index",_] -> Rule["start-index",ToString[startIndex+calls*maxPerPage]]];
			params = ReplaceAll[params,Rule["max-results",_] -> Rule["max-results",ToString[residual]]];
			
			rawdata = OAuthClient`rawoauthdata[id,"RawContacts",params];
			data = formatresults[rawdata];
			
			If[KeyExistsQ[data,"error"],
			(
				Message[ServiceExecute::serrormsg,Lookup[Lookup[data,"error"],"message"]];
				Throw[$Failed]
			)];
			
			data = "feed" /. data;
			totalResults = FromDigits[Lookup[Lookup[data,"openSearch$totalResults"],"$t"]];
			If[KeyExistsQ[data,"entry"],
				items = Join[items, If[totalResults>0,Lookup[data,"entry"],{}]]
			];
		)];

		result = Take[items,UpTo[limit]];
	
		fieldnames = {"id","updated","title","gd$email","gd$organization","gd$phoneNumber","gd$postalAddress","content"};
		result = Normal[KeyTake[#, fieldnames]] & /@ result;
   
		result = ReplaceAll[result,Rule["id",y_]:>Rule["ID",Last[StringSplit["$t"/.y,"/"]]]];
		result = ReplaceAll[result,Rule["updated",y_]:>Rule["Updated",DateObject["$t"/.y]]];
		result = ReplaceAll[result,Rule["title",y_]:>Rule["Title","$t"/.y]];
		result = ReplaceAll[result,Rule["content",y_]:>Rule["Content","$t"/.y]];
		result = ReplaceAll[result,Rule["gd$email",y_]:>Rule["Email","address"/.y]];
		result = ReplaceAll[result,Rule["gd$organization",y_]:>Rule["Organization",FilterRules[#,{"gd$orgTitle","gd$orgName"}]&/@y]];
		result = ReplaceAll[result,Rule["Organization",y_]:>Rule["Organization",ReplaceAll[#,{Rule["gd$orgTitle",z_]:>Rule["Title","$t"/.z],Rule["gd$orgName",x_]:>Rule["OrganizationName","$t"/.x]}]&/@y]];
		result = ReplaceAll[result,Rule["Organization",y_]:>Rule["Organization",Association/@y]];
		result = ReplaceAll[result,Rule["gd$phoneNumber",y_]:>Rule["PhoneNumber","$t"/.y]];
		result = ReplaceAll[result,Rule["gd$postalAddress",y_]:>Rule["PostalAddress","$t"/.y]];
		
		If[Length[result]==0,
			result = Association[],
			result = Association /@ result
		];
		If[prop=="ContactsList",
			result,
			Dataset[result]
		]	
]

googlecontactscookeddata[prop:("GroupList"|"GroupDataset"), id_, args_] := Module[{params={},date,query,sort,sortVal,sortDir,sd,rawdata,invalidParameters,limit,defaultPerPage=25,maxPerPage=250,startIndex,
											calls,residual,progress,data,fieldnames,result,items={}},
		invalidParameters = Select[Keys[args],!MemberQ[{"MaxItems",MaxItems,"StartIndex","UpdatedDate","Query","SortBy","ShowDeleted"},#]&]; 
	
		If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,prop]&/@invalidParameters;
			Throw[$Failed]
		)];

		If[KeyExistsQ[args,"MaxItems"],AppendTo[args,MaxItems->Lookup[args,"MaxItems"]]];
		params = Join[params,{"userEmail"->"default","alt"->"json"}];

		If[KeyExistsQ[args,"UpdatedDate"], (* minimum update date *)
		(
			date = Lookup[args,"UpdatedDate"];
			If[!DateObjectQ[date],
			(	
				Message[ServiceExecute::nval,"UpdatedDate","GoogleContacts"];
				Throw[$Failed]
			)];	
			date = DateString[date, "ISODateTime"];
			AppendTo[params,"updated-min"->date];
		)];

		If[KeyExistsQ[args,"Query"],
		(
			query = Lookup[args,"Query"];
			AppendTo[params,"q"->query];
		)];

		If[KeyExistsQ[args,"SortBy"],
		(
			sort = Lookup[args,"SortBy"];
			If[MatchQ[sort, {_String,_String}],
			(
				If[MatchQ[sort[[2]],"Ascending"|"Descending"],
					sortDir = ToLowerCase[sort[[2]]],
						(
							Message[ServiceExecute::nval,"SortBy","GoogleContacts"];	
							Throw[$Failed]
						)
				];	
				AppendTo[params,Rule["sortorder",sortDir]];
				sort = sort[[1]];
			)];
			Switch[sort,
				"LastModified",
				sortVal = "lastmodified",
				_,
				(
					Message[ServiceExecute::nval,"SortBy","GoogleContacts"];	
					Throw[$Failed]
				)
			];
			AppendTo[params,"orderby"->"lastmodified"];
		)];

		If[KeyExistsQ[args,"ShowDeleted"],
		(
			sd = Lookup[args,"ShowDeleted"];
			Switch[sd,
				True,
				AppendTo[params,"showdeleted"->"true"],
				False,
				AppendTo[params,"showdeleted"->"false"],
				_,
				(
					Message[ServiceExecute::nval,"ShowDeleted","GoogleContacts"];	
					Throw[$Failed]
				)
			];		
		)];
		
		If[KeyExistsQ[args,MaxItems],
		(
			limit = Lookup[args,MaxItems];
			If[!IntegerQ[limit],
			(
				Message[ServiceExecute::nval,"MaxItems","GoogleContacts"];
				Throw[$Failed]
			)];
		),
			limit = defaultPerPage;
		];
	
		If[KeyExistsQ[args,"StartIndex"],
		(
			startIndex = Lookup[args,"StartIndex"];
			If[!IntegerQ[startIndex],
			(	
				Message[ServiceExecute::nval,"StartIndex","GoogleContacts"];
				Throw[$Failed]
			)];
		),
			startIndex = 1		
		];
		
		calls = Quotient[limit, maxPerPage];	
		residual = limit - (calls*maxPerPage);
	
		params = Join[params,{"max-results"->ToString[maxPerPage], "start-index"->ToString[startIndex]}];
	
		(* this prints the progress indicator bar *)
		PrintTemporary[ProgressIndicator[Dynamic[progress], {0, calls}]];
	
		If[calls > 0,
		(
			(	
				params = ReplaceAll[params,Rule["start-index",_] -> Rule["start-index",ToString[startIndex+#*maxPerPage]]];
				rawdata = OAuthClient`rawoauthdata[id,"RawGroups",params];
				data = formatresults[rawdata];
				
				If[KeyExistsQ[data,"error"],
				(
					Message[ServiceExecute::serrormsg,Lookup[Lookup[data,"error"],"message"]];
					Throw[$Failed]
				)];

				items = Join[items, If[KeyExistsQ[Lookup[data,"feed"],"entry"],Lookup[Lookup[data,"feed"],"entry"],{}]];	
				progress = progress + 1;	
			)& /@ Range[0,calls-1];		
		
		)];
	
		If[residual > 0,
		(
			params = ReplaceAll[params,Rule["start-index",_] -> Rule["start-index",ToString[startIndex+calls*maxPerPage]]];
			params = ReplaceAll[params,Rule["max-results",_] -> Rule["max-results",ToString[residual]]];
			rawdata = OAuthClient`rawoauthdata[id,"RawGroups",params];
			data = formatresults[rawdata];
			
			If[KeyExistsQ[data,"error"],
			(
				Message[ServiceExecute::serrormsg,Lookup[Lookup[data,"error"],"message"]];
				Throw[$Failed]
			)];
			items = Join[items, If[KeyExistsQ[Lookup[data,"feed"],"entry"],Lookup[Lookup[data,"feed"],"entry"],{}]];			
		)];
	
		result = Take[items,UpTo[limit]];
			
		fieldnames = {"id","updated","title"};
		result = Normal[KeyTake[#, fieldnames]] & /@ result;
   
		result = ReplaceAll[result,Rule["id",y_]:>Rule["GroupID","$t"/.y]];
		result = ReplaceAll[result,Rule["updated",y_]:>Rule["Updated",DateObject["$t"/.y]]];
		result = ReplaceAll[result,Rule["title",y_]:>Rule["Title","$t"/.y]];
		
		If[Length[result]==0,
			result = Association[],
			result = Association /@ result
		];
		If[prop=="GroupList",
			result,
			Dataset[result]
		]	
]

googlecontactscookeddata["ContactInformation", id_, args_] := Module[{rawdata, invalidParameters,cId,fieldnames,result},
		invalidParameters = Select[Keys[args],!MemberQ[{"ContactID"},#]&]; 
	
		If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"ContactInformation"]&/@invalidParameters;
			Throw[$Failed]
		)];	
	
		If[KeyExistsQ[args,"ContactID"],
			cId = Lookup[args,"ContactID"],
			(
				Message[ServiceExecute::nparam,"ContactID"];			
				Throw[$Failed]
			)
		];
		
		rawdata = OAuthClient`rawoauthdata[id,"RawContactDetails",{"userEmail"->"default","alt"->"json","contactID"->ToString[cId]}];
		rawdata = formatresults[rawdata];
				
		If[KeyExistsQ[rawdata,"error"],
		(
			Message[ServiceExecute::serrormsg,Lookup[Lookup[rawdata,"error"],"message"]];
			Throw[$Failed]
		)];
				
		rawdata = Lookup[rawdata,"entry"];
		
		fieldnames = {"id","updated","title","gd$email","gd$organization","gd$phoneNumber","gd$postalAddress","gd$deleted"};
		result = Normal[KeyTake[#, fieldnames]] & /@ rawdata;
   		
		result = ReplaceAll[result,Rule["id",y_]:>Rule["ID",Last[StringSplit["$t"/.y,"/"]]]];
		result = ReplaceAll[result,Rule["updated",y_]:>Rule["Updated",DateObject["$t"/.y]]];
		result = ReplaceAll[result,Rule["title",y_]:>Rule["Title","$t"/.y]];
		result = ReplaceAll[result,Rule["content",y_]:>Rule["Content","$t"/.y]];
		result = ReplaceAll[result,Rule["gd$email",y_]:>Rule["Email","address"/.y]];
		result = ReplaceAll[result,Rule["gd$organization",y_]:>Rule["Organization",FilterRules[#,{"gd$orgTitle","gd$orgName"}]&/@y]];
		result = ReplaceAll[result,Rule["Organization",y_]:>Rule["Organization",ReplaceAll[#,{Rule["gd$orgTitle",z_]:>Rule["Title","$t"/.z],Rule["gd$orgName",x_]:>Rule["OrganizationName","$t"/.x]}]&/@y]];
		result = ReplaceAll[result,Rule["Organization",y_]:>Rule["Organization",Association/@y]];
		result = ReplaceAll[result,Rule["gd$phoneNumber",y_]:>Rule["PhoneNumber","$t"/.y]];
		result = ReplaceAll[result,Rule["gd$postalAddress",y_]:>Rule["PostalAddress","$t"/.y]];
		result = ReplaceAll[result,Rule["gd$deleted",y_]:>Rule["Deleted",y]];
		
		Association[result]		
]

googlecontactscookeddata["ContactPhoto", id_, args_] := Module[{rawdata, invalidParameters,cId,tmp},
		invalidParameters = Select[Keys[args],!MemberQ[{"ContactID"},#]&]; 
	
		If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"ContactPhoto"]&/@invalidParameters;
			Throw[$Failed]
		)];	
	
		If[KeyExistsQ[args,"ContactID"],
			cId = Lookup[args,"ContactID"],
			(
				Message[ServiceExecute::nparam,"ContactID"];			
				Throw[$Failed]
			)
		];
		
		rawdata = OAuthClient`rawoauthdata[id,"RawContactPhoto",{"userEmail"->"default","contactID"->ToString[cId]}];
		tmp = ImportString[FromCharacterCode[rawdata]];
		If[StringQ[tmp] && tmp === "Photo not found",
			Missing["NotAvailable"],
			googlecontactsimportphoto[rawdata]
		]
]

(* Send Message *)
googlecontactssendmessage[___]:=$Failed


End[] (* End Private Context *)
           		
End[]


SetAttributes[{},{ReadProtected, Protected}];

(* Return three functions to define oauthservicedata, oauthcookeddata, oauthsendmessage  *)
{GoogleContacts`Private`googlecontactsdata,GoogleContacts`Private`googlecontactscookeddata,GoogleContacts`Private`googlecontactssendmessage}
