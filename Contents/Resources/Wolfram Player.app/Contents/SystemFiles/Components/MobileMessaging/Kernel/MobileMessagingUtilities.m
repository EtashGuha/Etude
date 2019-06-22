BeginPackage["MobileMessaging`"]
(* Exported symbols added here with SymbolName::usage *)

Begin["`Private`"]

$IntegratedServicesAPIBase :=
    CloudObject["user:services-admin@wolfram.com/deployedservices/deployedserviceAPIs"];

$MobilePhoneEndpoint :=
    CloudObject[URLBuild[{$IntegratedServicesAPIBase,"GetUserMobilePhone"}]];

getRandomImageName[] :=
    ToString@FromDigits[StringReplace[CreateUUID[], "-" -> ""], 2]

uploadImageCloud[image_] :=
    Module[ {filename = StringJoin[{"images/",getRandomImageName[],".png"}],cb=$CloudBase,co,
        $CloudBase = "https://www.wolframcloud.com"},
        CloudConnect[];
        co2=CloudObject[filename];
        co = CloudExport[image, "PNG", co2,CloudBase->"https://www.wolframcloud.com", Permissions -> "Public"];
        $CloudBase = cb;
        First[co]
    ]

phoneVerifiedQ[params_] :=
    With[ {wid = $WolframID},
        If[ MatchQ[wid,"services-admin@wolfram.com"],
            False,
            True
        ]
    ]

returnMessage["PhoneVerified",False] :=
    (Message[SendMessage::nophone];
     Throw[$Failed])

imageQInternal[imageList_] :=
    MatchQ[imageList,{(_URL|_CloudObject|_LocalObject|_Graphics | _Graphics3D | _?ImageQ) ..} | _URL | _CloudObject | _LocalObject | _Graphics | _Graphics3D | _?ImageQ]

getInternalPhoneNumber[___] :=
    Module[ {co = FileNameJoin[{$IntegratedServicesAPIBase, "PhoneNumber"}],resp},
        resp = URLExecute[co,{},"WL",VerifySecurityCertificates -> False];
        If[ FailureQ[resp],
            Message[SendMessage::notauth],
            resp
        ]
    ]

getUserMobilePhone[___] :=
    Module[ {pn = URLExecute[CloudObject["https://www.wolframcloud.com/objects/services-admin/PhoneDirectory"],{},"WL",VerifySecurityCertificates -> False]},
        If[ FailureQ[pn],
            Message[SendMessage::notauth],
            pn
        ]
    ]

TFormatRule[rule_] :=
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
         }], "_"]] -> rule[[2]]

TImport[raw__] :=
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
    ]

proccessImageString[url_,body_,params_,pro_:True] :=
    Module[ {image = Quiet[Import[url]],mediaURL,resp,np,status},
        mediaURL =
        If[ TrueQ[pro],
            If[ FailureQ[image],
                $Failed,
                uploadImageCloud[image]
            ],
            url
        ];
        If[ FailureQ[image],
            status = <|"Error"->True,"Code"->3|>,
            np = Association[Join[params,{"Body"->body}]];
            np = Join[np,<|"MediaUrl"->mediaURL|>];
            resp = IntegratedServices`RemoteServiceExecute[SendMessage, "Twilio", "RawSend", np];
            status = If[ FailureQ[resp],
                         <|"Error"->True,"Code"->0,"Message"->"Internal Failure, please try again later"|>,
                         <|"Error"->False,"Code"->1|>
                     ]
        ];
        {resp,status,np}
    ]

proccessImageList[imgList_,body_,params_] :=
    Module[ {empty = SameQ[imgList,{}]},
        If[ TrueQ[empty],
            {$Failed,<|"Error"->True,"Code"->3|>,params}
        ]
    ]

proccessImage[img_,body_,params_] :=
    Module[ {imgQ = imageQInternal[img],mediaURL},
        mediaURL = If[ TrueQ[imgQ],
                       uploadImageCloud[img],
                       $Failed
                   ];
        proccessImageString[mediaURL,body,params,False]
    ]

proccessImageObjects[object_,body_,params_] :=
    Module[ {img = If[ SameQ[Head[object],URL],
                       Quiet[URLExecute[object,{}]],
                       Quiet[Import[object]]
                   ]},
        proccessImage[img,body,params]
    ]

proccessImageList[imgList_,body_,params_] :=
    Module[ {resp, status},
      resp = Switch[Head[#],
              String, proccessImageString[#,body,params],
              LocalObject|CloudObject|File|URL, proccessImageObjects[#,body,params],
              Graphics|Image|Graphics3D, proccessImage[#,body,params],
              __, status={$Failed,<|"Error"->True,"Code"->3|>,params}
            ]&/@imgList;
      {resp[[All,1]],<|"Error"->False,"Code"->1|>,resp[[All,3]]}
    ]

End[]
EndPackage[]
