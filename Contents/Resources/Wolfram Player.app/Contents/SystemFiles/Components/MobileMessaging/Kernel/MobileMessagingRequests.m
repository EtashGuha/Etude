
Unprotect[MobileMessaging`MobileMessaging];
Clear[MobileMessaging`MobileMessaging];
Unprotect[System`$MobilePhone];
Clear[System`$MobilePhone];

BeginPackage["MobileMessaging`"]

Begin["`Private`"] (* Begin Private Context *)

SendMessage::nomethod = "The method `1` is not available for SendMessage.";
SendMessage::nval = "Invalid value for parameter `1` in service `2`.";
SendMessage::nval2 = "Invalid value for parameter `1`.";
SendMessage::invmet = "This search service does not support searches of this form.";
SendMessage::invopt = "`1` is not a valid value for option `2`.";
SendMessage::offline = General::offline;
(*SendMessage::notauth = "SendMessage requests are only valid when authenticated. Please try authenticating again.";*)
SendMessage::strlimit = "The query exceeds the 1000 character limit.";

SendMessage::nophone = "A verified mobile phone number is required. Rerun when you have added a phone number to your account."
SendMessage::notauth = "Unable to authenticate with Wolfram Cloud server. Please try authenticating again."
SendMessage::mmsnotvalid  = "A URL, LocalObject, CloudObject, File, or a list of such objects is expected instead of `1`"
SendMessage::smsnotvalid  = "A string is expected instead of `1`"
SendMessage::externalerror  = "The external service returns the next message: `Message`";

Options[MobileMessaging] = {
  Method -> "Twilio"
};

(*System`$MobilePhone:= MobileMessaging`MobilePhone[]*)
System`$MobilePhone :=
    (Clear[$set];
     Clear[MobileMessaging`MobilePhone];
     None) /; !$CloudConnected
mobilephoneverified[_] = Missing[];

System`$MobilePhone :=
    getMobilePhone[]

getMobilePhone[] :=
    With[ {id = $WolframUUID},
        getMobilePhone[id]
    ] /; $CloudConnected

getMobilePhone[id_] /; !TrueQ[$set[id]] :=
    With[ {res = URLRead[HTTPRequest[$MobilePhoneEndpoint],VerifySecurityCertificates -> False]},
        (
            $set[id] = True;
            MobileMessaging`MobilePhone[id] = formatPhoneNumber[ImportString[res["Body"], "RawJSON"]]
        ) /; res["StatusCode"]===200
    ]

getMobilePhone[id_] /; $set[id] :=
    MobileMessaging`MobilePhone[id]

getMobilePhone[_] :=
    None

$errortags = ("exception"| "unkwn" | "apierror" | "nopv" | "nosc" | "dialogerror" | "notos")

MobileMessaging`MobileMessaging[___] :=
    (Message[SendMessage::offline];
     $Failed)/;(!PacletManager`$AllowInternet)

MobileMessaging`MobileMessaging[args___] :=
    With[ {connected = CloudConnect[]},
        If[ $CloudConnected,
            MobileMessaging`MobileMessaging[args],
            Message[SendMessage::notauth];
            connected
        ]
    ]/; !$CloudConnected

MobileMessaging`MobileMessaging[args__] :=
    With[ {res = Catch[MobileMessaging1[args]]},
        res /; !MatchQ[res, "exception"]
    ]

MobileMessaging1[type_, args__,opt: OptionsPattern[MobileMessaging]] :=
    Block[ {r,args2 = args, body = "",imagePlaceholder,imageurl,imageurlList = {},params,paramsList = {},rawList = {},status,raw,response,p},
        If[ MatchQ[$MobilePhone,None],
            IntegratedServices`CreatePhoneVerificationDialogs["SendMessage"];
            Message[SendMessage::nophone];
            Throw["exception"]
        ];
        params = {"To"->$MobilePhone,"From"->getInternalPhoneNumber[]};
        Switch[type,
        "SMS",
              body = args2;
              params = Association[Join[params,{"Body"->body}]];
              If[ StringQ[body],
                  If[ phoneVerifiedQ[params],
                      response = IntegratedServices`RemoteServiceExecute[SendMessage, "Twilio", "RawSend", params];
                      If[ !FailureQ[response],
                          status = <|"Error"->False,"Code"->1|>
                      ],
                      returnMessage["PhoneVerified",False];
                      response = $Failed;
                  ],
                  Message[SendMessage::smsnotvalid, body];
                  Throw[$Failed]
              ]
              ,
        "MMS",
              If[ And[Not[MatchQ[args2,{}]],ListQ[args2]],
                  {body,imagePlaceholder} = args2,
                  imagePlaceholder = args2
              ];
              {response,status,p} =
                Switch[Head[imagePlaceholder],
                  String,proccessImageString[imagePlaceholder,body,params],
                  LocalObject|CloudObject|File|URL,proccessImageObjects[imagePlaceholder,body,params],
                  List,If[imageQInternal[imagePlaceholder],proccessImageList[imagePlaceholder,body,params],{$Failed,<|"Error"->False,"Code"->3|>,params}],
                  Graphics|Image|Graphics3D,proccessImage[imagePlaceholder,body,params],
                  __,status =  {$Failed,<|"Error"->False,"Code"->3|>,params}
                ],
        ___, Throw[$Failed]
        ];
        Switch[status["Code"],
            3, Message[SendMessage::mmsnotvalid, imagePlaceholder];
               Throw[$Failed],
            2, Message[SendMessage::smsnotvalid, body];
               Throw[$Failed],
            1, If[ MatchQ[Head[response],List],
                   createResponse[type, #[[1]], #[[2]]]&/@Transpose[{response, p}],
                   createResponse[type,response,p]
               ],
            0, createResponse[type,response,params,False],
            __, Throw[$Failed]
        ]
    ]

createResponse[tag_,resp_,params_,successQ_:True] :=
    If[ successQ,
        Success[tag, <|
            "Message" -> Short[prepareResponse[tag,resp,params]],
            (*"Recipient"->resp["To"],*)
            "SID" -> resp["Sid"]|>],
        Failure["ExternalError", <|
            "Message" :> SendMessage::externalerror,
            "MessageParameters" -> <|"Message" -> response["Message"]|>|>]
    ]

prepareResponse[tag_,resp_,params_] :=
    Module[ {smsQ = MatchQ[tag,"SMS"],body = resp["Body"],image},
        If[ smsQ,
            body,
            (
              image = Import[params["MediaUrl"]];
              Row[{body," ",image}]
            )
        ]
    ]

MobileMessaging1[___] :=
    Throw["exception"]

(*formatting mobile phone + country code*)
formatPhoneNumber[rawphone_Association]:= (Replace[rawphone["countryCode"], $CountryCodesRules] <> rawphone["phone"])
formatPhoneNumber[___]:=Missing["Unknown"]

(*list of supported countries*)
$CountryCodesRules = Dispatch[{"AT"->"+43 ", "AU"->"+61 ", "BE"->"+32 ", "BR"->"+55 ", "CH"->"+41 ", "CL"->"+56 ", "CZ"->"+420 ",
 "DK"->"+45 ", "EG"->"+20 ", "ES"->"+34 ", "FI"->"+358 ", "GB"->"+44 ", "GT"->"+502 ", "HK"->"+852 ", "HR"->"+385 ", "HU"->"+36 ",
 "IE"->"+353 ", "IL"->"+972 ", "IT"->"+39 ", "KR"->"+82 ", "LT"->"+370 ", "LV"->"+371 ", "MX"->"+52 ", "MY"->"+60 ", "NL"->"+31 ",
 "NO"->"+47 ", "PE"->"+51 ", "PH"->"+63 ", "PL"->"+48 ", "PT"->"+351 ", "SE"->"+46 ", "SG"->"+65 ", "TW"->"+886 ", "US"->"+1 "}]


SetAttributes[MobileMessaging`MobileMessaging, {Protected, ReadProtected}];
SetAttributes[System`$MobilePhone, {Protected, ReadProtected}];
End[];
EndPackage[];
