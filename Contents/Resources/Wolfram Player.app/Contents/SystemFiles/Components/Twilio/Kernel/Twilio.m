Begin["Twilio`"]
Begin["`Private`"]

(******************************* Twilio *************************************)

(* Authentication information *)

twiliodata[] :=
    {
    "ServiceName"   -> "Twilio",
    "URLFetchFun"   :> (With[ {params = Lookup[{##2},"Parameters",{}]},
                            URLFetch[#1,{"Content","StatusCode"},
                                Sequence@@FilterRules[{##2},Except["Parameters"]],
                                    "Parameters" -> FilterRules[params, Except[{"accountid", "authtoken"}]],
                                    "Username"->"accountid"/.params,
                                    "Password"->"authtoken"/.params
                                    ]
                        ]
                     &),
    "ClientInfo"    :>With[ {res = URLExecute[CloudObject["https://www.wolframcloud.com/objects/user-00e58bd3-2dfd-45b3-b80b-d281d360703a/twilioInternalApproval"]]},
                          If[ ListQ[res],
                              res,
                              OAuthDialogDump`Private`MultipleKeyDialog["Twilio",
                              {
                                  "AccountID"              ->    "accountid",
                                  "AuthenticationToken"    ->    {"authtoken",FieldMasked->True}
                              },
                              "https://www.twilio.com/login","https://www.twilio.com/legal/tos"]
                          ]
                      ],
    "Gets"            -> {"CallStatus","ListCalls","ListMessages","MessageStatus","MyNumbers","ViewIncomingPhoneNumbers"},
    "Posts"         -> {"Call","Send","AddIncomingPhoneNumber"},
    "RawGets"       -> {"RawCallStatus","RawListCalls","RawListMessages","RawMessageStatus","RawMyNumbers","RawViewIncomingPhoneNumbers"},
    "RawPosts"      -> {"RawAddIncomingPhoneNumber","RawCall","RawSend"}
    }

(***** Raw Requests *****)

twiliodata["RawAddIncomingPhoneNumber"] :=
    {
    "URL"               -> (ToString@StringForm["https://api.twilio.com/2010-04-01/Accounts/`1`/IncomingPhoneNumbers/`2`.json",#1, #2]&),
    "HTTPSMethod"       -> "POST",
    "PathParameters"    -> {"accountid","PhoneNumberSid"},
    "Parameters"        -> {"PhoneNumberSid","SmsUrl"},
    "RequiredParameters"-> {"PhoneNumberSid","SmsUrl"},
    "ResultsFunction"   -> TImport
    }

twiliodata["RawCall"] :=
    {
    "URL"               -> (ToString@StringForm["https://api.twilio.com/2010-04-01/Accounts/`1`/Calls.json", #]&),
    "HTTPSMethod"       -> "POST",
    (*"Headers"            -> {"Content-Type" -> "text/html; charset=UTF-8"},*)
    "PathParameters"    -> {"accountid"},
    "Parameters"        -> {"To","From","Url"},
    "RequiredParameters"-> {"To","From","Url"},
    "ResultsFunction"   -> TImport
    }

twiliodata["RawCallStatus"] :=
    {
    "URL"               -> (ToString@StringForm["https://api.twilio.com/2010-04-01/Accounts/`1`/Calls/`2`.json",#1, #2] &),
    "HTTPSMethod"       -> "GET",
    (*"Headers"            -> {"Content-Type" -> "text/html; charset=UTF-8"},*)
    "PathParameters"    -> {"accountid","CallSid"},
    "Parameters"        -> {"CallSid"},
    "RequiredParameters"-> {"CallSid"},
    "ResultsFunction"   -> TImport
    }

twiliodata["RawListCalls"] :=
    {
    "URL"               -> (ToString@StringForm["https://api.twilio.com/2010-04-01/Accounts/`1`/Calls.json",#] &),
    "HTTPSMethod"       -> "GET",
    (*"Headers"            -> {"Content-Type" -> "text/html; charset=UTF-8"},*)
    "PathParameters"    -> {"accountid"},
    "Parameters"        -> {"From","To","StartTime","Status"},
    "RequiredParameters"-> {},
    "ResultsFunction"   -> TImport
    }

twiliodata["RawListMessages"] :=
    {
    "URL"               -> (ToString@StringForm["https://api.twilio.com/2010-04-01/Accounts/`1`/Messages.json",#1] &),
    "HTTPSMethod"       -> "GET",
    (*"Headers"            -> {"Content-Type" -> "text/html; charset=UTF-8"},*)
    "PathParameters"    -> {"accountid"},
    "Parameters"        -> {"From","To","DateSent","Status"},
    "RequiredParameters"-> {},
    "ResultsFunction"   -> TImport
    }

twiliodata["RawMessageStatus"] :=
    {
    "URL"               -> (ToString@StringForm["https://api.twilio.com/2010-04-01/Accounts/`1`/Messages/`2`.json",#1, #2] &),
    "HTTPSMethod"       -> "GET",
    (*"Headers"            -> {"Content-Type" -> "text/html; charset=UTF-8"},*)
    "PathParameters"    -> {"accountid","MessageSid"},
    "Parameters"        -> {"MessageSid"},
    "RequiredParameters"-> {"MessageSid"},
    "ResultsFunction"   -> TImport
    }

twiliodata["RawMyNumbers"] :=
    {
    "URL"               -> (ToString@StringForm["https://api.twilio.com/2010-04-01/Accounts/`1`/IncomingPhoneNumbers.json", #]&),
    "HTTPSMethod"       -> "GET",
    "PathParameters"    -> {"accountid"},
    "Parameters"        -> {"FriendlyName","Page","PageSize","PhoneNumber"},
    "RequiredParameters"-> {},
    "ResultsFunction"   -> TImport
    }

twiliodata["RawSend"] :=
    {
    "URL"               -> (ToString@StringForm["https://api.twilio.com/2010-04-01/Accounts/`1`/Messages.json", #]&),
    "HTTPSMethod"       -> "POST",
    (*"Headers"            -> {"Content-Type" -> "text/html; charset=UTF-8"},*)
    "PathParameters"    -> {"accountid"},
    "Parameters"        -> {"To","From","Body","MediaUrl"},
    "RequiredParameters"-> {"To","From","Body"},
    "ResultsFunction"   -> TImport
    }

twiliodata["RawViewIncomingPhoneNumbers"] :=
    {
    "URL"               -> (ToString@StringForm["https://api.twilio.com/2010-04-01/Accounts/`1`/IncomingPhoneNumbers/`2`.json", #1, #2]&),
    "HTTPSMethod"       -> "POST",
    (*"Headers"            -> {"Content-Type" -> "text/html; charset=UTF-8"},*)
    "PathParameters"    -> {"accountid","PhoneNumberSid"},
    "Parameters"        -> {"PhoneNumberSid"},
    "RequiredParameters"-> {"PhoneNumberSid"},
    "ResultsFunction"   -> TImport
    }

(***** Cooked Requests *****)

twiliocookeddata[___] :=
    $Failed

twiliocookeddata["AddIncomingPhoneNumber", id_,args_] :=
    Block[ {params = Association[args],raw},
        raw = TImport[KeyClient`rawkeydata[id,"RawCall",Normal@params]];
        Dataset[raw]
    ]

twiliocookeddata["Call", id_,args_] :=
    Block[ {params = Association[args],raw},
        params["Url"] = TGetCloudXML[params["Url"],params["Body"]];
        raw = TImport[KeyClient`rawkeydata[id,"RawCall",Normal@KeyDrop[params,"Body"]]];
        Dataset[raw]
    ]

twiliocookeddata["CallStatus", id_,args_] :=
    Module[ {params = Association[args],raw},
        raw = TImport[KeyClient`rawkeydata[id,"RawCallStatus",Normal@params]];
        Dataset[raw]
    ]

twiliocookeddata["MessageStatus", id_,args_] :=
    Module[ {params = Association[args],raw,invalidParameters},
        invalidParameters = Select[Keys@params,!MemberQ[{"MessageSid"},#]&];
        If[ Length[invalidParameters]>0,
            (
                Message[ServiceObject::noget,#,"Twilio"]&/@invalidParameters;
                Throw[$Failed]
            )
        ];
        raw = TImport[KeyClient`rawkeydata[id,"RawMessageStatus",Normal@params]];
        Dataset[raw]
    ]

twiliocookeddata["MyNumbers", id_,args_] :=
    Block[ {params = Association[args],raw},
        raw = TImport[KeyClient`rawkeydata[id,"RawMyNumbers",Normal@params]];
        Dataset[raw]
    ]

twiliocookeddata["ListMessages", id_,args_] :=
    Module[ {params = Association[args],raw,invalidParameters,date},
        invalidParameters = Select[Keys@params,!MemberQ[{"From","To","Date","Status"},#]&];
        If[ Length[invalidParameters]>0,
            (
                Message[ServiceObject::noget,#,"Twilio"]&/@invalidParameters;
                Throw[$Failed]
            )
        ];
        If[ KeyExistsQ[params,"Date"],
            (
                If[ !(StringQ[params["Date"]]||MatchQ[params["Date"],DateObject[__]]),
                    (
                        Message[ServiceExecute::nval,"Date","Twilio"];
                        Throw[$Failed]
                    )
                ];
                date = DateObject[params["Date"]];
                If[ MatchQ[date,DateObject[__String]],
                    (
                        Message[ServiceExecute::nval,"Date","Twilio"];
                        Throw[$Failed]
                    )
                ];
                params["Date"] = DateString[TimeZoneConvert[date,0], {"Year", "-", "Month", "-", "Day"}];
                params = Association[ReplaceAll[Normal@params,"Date" -> "DateSent"]];
            )
        ];
        raw = TImport[KeyClient`rawkeydata[id,"RawListMessages",Normal@params]];
        Dataset[raw]
    ]

twiliocookeddata["ListCalls", id_,args_] :=
    Module[ {params = Association[args],raw},
        raw = TImport[KeyClient`rawkeydata[id,"RawListCalls",Normal@params]];
        Dataset[raw]
    ]

twiliocookeddata["Send", id_,args_] :=
    Module[ {params = Association[args],raw,invalidParameters},
        invalidParameters = Select[Keys@params,!MemberQ[{"To","From","Body","MediaURL"},#]&];
        If[ Length[invalidParameters]>0,
            (
                Message[ServiceObject::noget,#,"Twilio"]&/@invalidParameters;
                Throw[$Failed]
            )
        ];
        If[ KeyExistsQ[params,"MediaURL"],
            (
                If[ !(StringQ[params["MediaURL"]]),
                    (
                        Message[ServiceExecute::nval,"MediaURL","Twilio"];
                        Throw[$Failed]
                    )
                ];
                params = Association[ReplaceAll[Normal@params,"MediaURL" -> "MediaUrl"]];
            )
        ];
        raw = TImport[KeyClient`rawkeydata[id,"RawSend",Normal@params]];
        Dataset[raw]
    ]

twiliocookeddata["ViewIncomingPhoneNumbers", id_,args_] :=
    Module[ {params = Association[args],raw},
        raw = TImport[KeyClient`rawkeydata[id,"RawViewIncomingPhoneNumbers",Normal@params]];
        Dataset[raw]
    ]

(***** Auxiliary Functions *****)

TFormatRule[rule_Rule] :=
    StringJoin[
      Function[word,
      StringReplacePart[word, ToUpperCase[StringTake[word, 1]], 1]] /@
      StringSplit[
    StringReplace[
     rule[[1]], {
      "img" -> "Image",
      "url" -> "URL",
      "uri" -> "URI",
        "api" -> "API",
      "sms" -> "SMS",
      "mms" -> "MMS"
      }], "_"]] -> rule[[2]];

TImport[raw_] :=
    Module[ {response = ImportString[raw[[1]], "JSON"],status = raw[[2]]},
        If[ response===$Failed,
            Throw[$Failed]
        ];
        If[ !MatchQ[status,200|201],
            (Message[ServiceExecute::apierr,"message"/.response];
             Throw[$Failed]),
            Map[If[ Length[#] > 0 && Head[#[[1]]] === Rule,
                    Association[
                     TFormatRule /@ #],
                    #
                ] &,
            Association[TFormatRule /@ (response /. {
              ("date_updated" -> val_ /; val =!= Null) :> ("date_updated" -> DateObject[First[StringSplit[val, " +"]]]),
              ("date_created" -> val_ /; val =!= Null) :> ("date_created" -> DateObject[First[StringSplit[val, " +"]]]),
              ("date_sent" -> val_ /; val =!= Null) :> ("date_sent" -> DateObject[First[StringSplit[val, " +"]]])
              })],
              -2]
        ]
    ];

End[]

End[]

SetAttributes[{},{ReadProtected, Protected}];

{Twilio`Private`twiliodata,
 Twilio`Private`twiliocookeddata,
 Twilio`Private`twiliosendmessage}
