Get["GoogleCustomSearchFunctions`"]

Begin["GoogleCustomSearch`"] (* Begin Private Context *)

Begin["`Private`"]

(******************************* GoogleCustomSearch *************************************)

(* Authentication information *)

googlecustomsearchdata[] :=
    {
    "ServiceName"         -> "GoogleCustomSearch",
    "URLFetchFun"        :> (With[ {params = Lookup[{##2}, "Parameters", {}]},
                                (
                                    URLRead[HTTPRequest[
                                        #1,
                                        <|
                                            Method -> "GET",
                                            "Query" -> params
                                        |>]
                                    ]
                                )
                             ] &),
    "ClientInfo"        :> OAuthDialogDump`Private`MultipleKeyDialog["GoogleCustomSearch",{"API key"->"key","Custom search engine ID"->"cx"},
                  "https://cse.google.com/cse/manage/all","https://developers.google.com/custom-search/terms"],
    "Gets"                -> {"Search"},
    "Posts"                -> {},
    "RawGets"            -> {"RawSearch"},
    "RawPosts"            -> {},
    "Information"        -> "Import Google Custom Search API data to the Wolfram Language"
    }

(**** Raw Requests ****)

googlecustomsearchdata["RawSearch"] :=
    {
    "URL"                -> "https://www.googleapis.com/customsearch/v1",
    "HTTPSMethod"        -> "GET",
    "Parameters"        -> {"cref","q","alt","callback","fields","prettyPrint","quotaUser","userIp","num","start","searchType","cr","lr","safe","dateRestrict"},
    "RequiredParameters"-> {"q"},
    "ResultsFunction"    -> GSFormatResults
    }

googlecustomsearchdata[___] :=
    $Failed

(**** Cooked Requests ****)

(*GCSFormat[rawdata__] := ImportString[FromCharacterCode[rawdata[[2]],"UTF8"],"RawJSON"]["items"]*)
GSFormatResults[response_] := If[response["StatusCode"] != 200,
    {"error" -> response["StatusCode"], "message" -> response["Body"]}
,
    Developer`ReadRawJSONString[Quiet[FromCharacterCode[response["BodyBytes"], "UTF-8"], {$CharacterEncoding::utf8}], "IssueMessagesAs" -> Symbol["GoogleCustomSearch"]]
]

googlecustomsearchcookeddata["Search", id_,args_List] := Module[
    {rparams = GCSFormatRequestParameters[args], requestParameters, type},

    {requestParameters, type} = rparams;
    GCSCookedImport[GCSPaginationCalls[id, "RawSearch", requestParameters], args, type]
]

googlecustomsearchcookeddata[___]:=$Failed

googlecustomsearchsendmessage[___]:=$Failed

End[]

End[]

(* Return three functions to define oauthservicedata, oauthcookeddata, oauthsendmessage  *)
{GoogleCustomSearch`Private`googlecustomsearchdata,GoogleCustomSearch`Private`googlecustomsearchcookeddata,GoogleCustomSearch`Private`googlecustomsearchsendmessage}
