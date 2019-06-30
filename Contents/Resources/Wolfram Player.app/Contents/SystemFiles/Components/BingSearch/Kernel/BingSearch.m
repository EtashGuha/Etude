Get["BingSearchFunctions`"]

Begin["BingSearch`"] (* Begin Private Context *)

Begin["`Private`"](* Begin Private Context *)

(******************************* BingSearch *************************************)
(* Authentication information *)
bingsearchdata[] :=
    {
    "ServiceName"			-> "BingSearch",
    "URLFetchFun"			:> (With[ {params = Lookup[{##2}, "Parameters", {}]},
                                (
                                    URLRead[HTTPRequest[
                                        #1,
                                        <|
                                            Method -> "GET",
                                            "Headers" -> {"Ocp-Apim-Subscription-Key"-> Lookup[params, "key", ""]},
                                            "Query" -> KeyDrop[params, "key"]
                                        |>]
                                    ]
                                )
                             ]&)
    ,
    "ClientInfo"			:> OAuthDialogDump`Private`MultipleKeyDialog["BingSearch",{"Subscription Key"->"key"},
    								"https://portal.azure.com","https://azure.microsoft.com/en-us/support/legal"],
    "Gets"					-> {"Search"},
    "Posts"					-> {},
    "RawGets"				-> {"RawSearch"},
    "RawPosts"				-> {},
    "Information"			-> "Wolfram Language connection to BingSearch API"
    }

(****** Raw Properties ******)

bingsearchdata["RawSearch"] :=
    {
    "URL"                -> (ToString@StringForm["https://api.cognitive.microsoft.com/bing/v5.0/`1`", #]&),
    "HTTPSMethod"        -> "GET",
    "Parameters"        -> {"count", "offset", "q", "responseFilter", "safeSearch"},
    "PathParameters"    -> {"SearchTypePath"},
    "RequiredParameters"-> {"SearchTypePath"},
    "ResultsFunction"    -> BSFormatResults
    }

bingsearchdata[___] :=
    $Failed

BSFormatResults[response_] := Block[
    {stringResponse, error},

    stringResponse = Quiet[FromCharacterCode[response["BodyBytes"], "UTF-8"], {$CharacterEncoding::utf8}];

    If[response["StatusCode"] != 200,
        Quiet[
            error = Developer`ReadRawJSONString[stringResponse, "IssueMessagesAs" -> Symbol["BingSearch"]];
        ];

        If[AssociationQ[error],
            Message[ServiceExecute::serrormsg, Lookup[error, "message", "Missing error."]]
        ];

        Throw[$Failed]
    ,
        Developer`ReadRawJSONString[stringResponse, "IssueMessagesAs" -> Symbol["BingSearch"]]
    ]
]

(****** Cooked Properties ******)
bingsearchcookeddata[prop_,id_, rest__] :=
    bingsearchcookeddata[prop,id,{rest}]

bingsearchcookeddata["Search",id_,args_] :=
    Module[ {requestParameters = BSFormatRequestParameters[args]},
        BSCookedImport[BSPaginationCalls[id, "RawSearch", requestParameters], requestParameters,args]
    ]

bingsearchcookeddata[___] :=
    $Failed

bingsearchsendmessage[___] :=
    $Failed

End[]

End[]

(*SetAttributes[{},{ReadProtected, Protected}];*)

(* Return two functions to define oauthservicedata, oauthcookeddata  *)

{BingSearch`Private`bingsearchdata,
 BingSearch`Private`bingsearchcookeddata,
 BingSearch`Private`bingsearchsendmessage}
