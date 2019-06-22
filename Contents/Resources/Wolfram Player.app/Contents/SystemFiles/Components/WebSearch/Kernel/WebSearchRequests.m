System`WebSearch
System`AllowAdultContent

Unprotect[System`WebSearch];
Clear[System`WebSearch];

BeginPackage["WebSearch`"]

Begin["`Private`"] (* Begin Private Context *)

System`WebSearch::nomethod = "The method `1` is not available for WebSearch.";
System`WebSearch::nval = "Invalid value for parameter `1` in service `2`.";
System`WebSearch::nval2 = "Invalid value for parameter `1`.";
System`WebSearch::invmet = "This search service does not support searches of this form.";
System`WebSearch::invopt = "`1` is not a valid value for option `2`.";
System`WebSearch::offline = "The Wolfram Language is currently configured not to use the Internet. To allow Internet use, check the \"Allow the Wolfram Language to use the Internet\" box in the Help \[FilledRightTriangle] Internet Connectivity dialog.";
System`WebSearch::notauth = "WebSearch requests are only valid when authenticated. Please try authenticating again.";
System`WebSearch::strlimit = "The query exceeds the 1000 character limit.";

WSServiceName[name_] :=
    Switch[name,
            "Google" | "GoogleCustomSearch", "GoogleCustomSearch",
            "Bing" | "BingSearch", "BingSearch",
            _, Message[System`WebSearch::nomethod, name];
               Throw["exception"]
    ]


Options[WebSearch] = SortBy[{
  Method -> "Bing",
  "MaxItems" -> 10,
  "StartIndex"-> 0,
  "Site"-> Null,
  "Country"-> Null,
  MaxItems -> 10,
  Language -> $Language,
  AllowAdultContent -> False,
  "FileFormat" -> Null,
  SortedBy -> None},(ToString[#[[1]]]&)];

System`WebSearch[___] :=
    (Message[System`WebSearch::offline];
     $Failed)/;(!PacletManager`$AllowInternet)

System`WebSearch[args___] :=
    With[ {connected = CloudConnect[]},
        If[ $CloudConnected,
            System`WebSearch[args],
            Message[System`WebSearch::notauth];
            connected
        ]
    ]/; !$CloudConnected

System`WebSearch[args__] :=
    With[ {res = Catch[WebSearch1[args]]},
        res /; !MatchQ[res, "exception"]
    ]

WebSearch1[query_, elem_String:"Elements", maxItems_Integer:10, opt: OptionsPattern[WebSearch]] :=
    Block[ {engine,requestParameters,response,rParams,type,query2,opt2 = List@opt, result,sf},
        If[ !MemberQ[{"Title", "Titles", "PageTitle", "PageTitles", "Snippet", "Snippets", "Hyperlink", "Hyperlinks","PageHyperlinks", "Elements"}, elem],
            WSReturnMessage[Symbol["WebSearch"], "nval2", "Elements"];
            Throw["exception"]
        ];
        engine = WSServiceName[OptionValue[Method]];
        query2 = WebSearch`WSGetFormattedQuery[query, engine];
        If[ !KeyExistsQ[opt2, Language],
            opt2 = Join[opt2, {Language -> $Language}]
        ];
        If[ StringQ[query2],
            If[ StringLength[query2]>1000,
                Message[System`WebSearch::strlimit];
                Throw["exception"]
            ]
        ];
        If[ !KeyExistsQ[opt2, AllowAdultContent],
            opt2 = Join[opt2, {"ContentFiltering"->"High"}],
            opt2 = ReplaceAll[opt2, (AllowAdultContent->x_):>("ContentFiltering"->(x/.{False->"High",True->"Off",
                                         val_ :> (Message[System`WebSearch::invopt, val, AllowAdultContent];
                                                  Throw["exception"])}))];
        ];
        If[ !KeyExistsQ[opt2, MaxItems],
            opt2 = Join[opt2, {MaxItems->maxItems}]
        ];

        sf = Association[opt2][SortedBy];
        opt2 = Normal@KeyDrop[opt2,SortedBy];

        result = Switch[engine,
                "BingSearch",
                (
                Get["BingSearch`"];
                requestParameters = BingSearchFunctions`BSFormatRequestParameters[FilterRules[Append[{opt2},"Query"->query2],Except[Method]],"IntegratedServiceQ"->True];
                response = BSPaginationCalls[Symbol["WebSearch"], requestParameters];

                If[ !MatchQ[response, False | $Failed],
                    BingSearchFunctions`BSCookedImport[response,requestParameters,{opt}]
                    ,
                    Throw["exception"]
                ]
                ),
                "GoogleCustomSearch",
                (
                Get["GoogleCustomSearch`"];
                rParams = GoogleCustomSearchFunctions`GCSFormatRequestParameters[FilterRules[Append[{opt2},"Query"->query2],Except[Method]]];
                {requestParameters,type} = rParams;
                response = GCSPaginationCalls[Symbol["WebSearch"], requestParameters];

                If[ !MatchQ[response, False | $Failed | _Symbol],
                    GoogleCustomSearchFunctions`GCSCookedImport[response,requestParameters,type],
                    Throw["exception"]
                ]
          )
        ];

        result = result[All, <|"PageTitle" -> "Title", "Snippet" -> "Snippet", "Hyperlink" -> "Link"|>];
        result =
        Switch[elem,
            "Title" | "Titles" | "PageTitle" | "PageTitles",
                Normal[result[All, "PageTitle"]]
            ,

            "Snippet" | "Snippets",
                Normal[result[All, "Snippet"]]
            ,

            "Hyperlink" | "Hyperlinks" | "PageHyperlinks",
                Normal[result[All, "Hyperlink"]]
            ,

            _,
                result
        ];

        If[!MissingQ[sf],SortBy[result, sf],result]
    ]

WebSearch1[___] :=
    Throw["exception"]

SetAttributes[System`WebSearch, {Protected, ReadProtected}];
End[];
EndPackage[];
