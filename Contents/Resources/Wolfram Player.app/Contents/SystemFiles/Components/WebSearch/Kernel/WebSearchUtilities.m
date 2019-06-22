BeginPackage["WebSearch`"]
(* Exported symbols added here with SymbolName::usage *)

WSReturnMessage::usage = "";
WSGetFormattedQuery::usage = "";

Begin["`Private`"]

WSReturnMessage[tag_, errorcode_, params___] := With[
    {msg = MessageName[Evaluate@Symbol["WebSearch"], errorcode]},
    If[ MatchQ[tag, Symbol["WebSearch"]],
        Message[MessageName[System`WebSearch, errorcode], params];,
        MessageName[tag, errorcode] = msg;
        Message[MessageName[tag, errorcode], params];
    ]
]

WSGetFormattedQuery[q_, engine_, tag_ : Symbol["WebSearch"]] := Block[
    {query = q, newQuery},

    Which[
        StringQ[query],
            If[StringContainsQ[query, "\""],
                query
            ,
                "\"" <> query <> "\""
            ]
        ,

        MatchQ[query, SearchQueryString[_String?StringQ]],
            First[query]
        ,

        MatchQ[query, ContainsExactly[_String?StringQ]],
            "\"" <> First[query] <> "\""
        ,

        MatchQ[query, _Alternatives],
            newQuery = List@@query;
            newQuery = If[MatchQ[#, _Except], ContainsNone[First[#]], #] & /@ newQuery;
            If[MatchQ[newQuery, {(_String?StringQ | ContainsNone[_String?StringQ]) ..}],
                With[
                    {
                        searchText = "(" <> StringRiffle[
                            Select[newQuery, StringQ], " OR "
                        ] <> ")",
                        excludeText = StringRiffle[
                            StringInsert[
                                Cases[newQuery, ContainsNone[val_] :> val],
                                "-",
                                1
                            ],
                            " "
                        ]
                    },

                    If[!MatchQ[searchText, ""],
                        StringTrim[searchText <> " " <> excludeText]
                    ,
                        WSReturnMessage[tag, "nval", "Query", engine];
                        Throw["exception"]
                    ]
                ]
            ,
                WSReturnMessage[tag, "nval", "Query", engine];
                Throw["exception"]
            ]
        ,

        MatchQ[query, ContainsAny[{_String?StringQ..}]],
            StringRiffle[First[query], " OR "]
        ,

        MatchQ[query, FixedOrder[_String?StringQ..]],
            newQuery = List@@query;
            If[MatchQ[engine, "GoogleCustomSearch"],
                StringRiffle[newQuery, " * "]
            ,
                WSReturnMessage[tag, "invmet"];
                StringRiffle[newQuery, " "]
            ]
        ,

        MatchQ[query, ContainsAll[{_String?StringQ..}]],
            newQuery = First[query];
            newQuery = With[
                {
                    searchText = "(" <> StringRiffle[
                        Select[newQuery, StringQ], If[MatchQ[engine, "BingSearch"], " AND ", " "]
                    ] <> ")"
                },

                If[!MatchQ[searchText, ""],
                    StringTrim[searchText]
                ,
                    WSReturnMessage[tag, "nval", "Query", engine];
                    Throw["exception"]
                ]
            ];

            If[!MatchQ[engine, "BingSearch"],
                WSReturnMessage[tag, "invmet"]
            ];

            newQuery
        ,

        ListQ[query],
            query = If[MatchQ[#, _Except], ContainsNone[First[#]], #] & /@ query;
            If[MatchQ[query, {(_String?StringQ | ContainsNone[_String?StringQ]) ..}],
                newQuery = With[
                    {
                        searchText = "(" <> StringRiffle[
                            Select[query, StringQ], If[MatchQ[engine, "BingSearch"], " AND ", " "]
                        ] <> ")",
                        excludeText = StringRiffle[
                            StringInsert[
                                Cases[query, ContainsNone[val_] :> val],
                                "-",
                                1
                            ],
                            " "
                        ]
                    },

                    If[!MatchQ[searchText, ""],
                        StringTrim[searchText <> " " <> excludeText]
                    ,
                        WSReturnMessage[tag, "nval", "Query", engine];
                        Throw["exception"]
                    ]
                ];

                newQuery
            ,
                WSReturnMessage[tag, "nval", "Query", engine];
                Throw["exception"]
            ]
        ,

        True,
            WSReturnMessage[tag, "nval", "Query", engine];
            Throw["exception"]
    ]
]

BSPaginationCalls[tag_, p_, wisQ_:False]:= Module[
    {calls, residual, progress = 0, items, params = p, nresultsperpage, bar, elementsLeft, firstIteration = True, result = {}, totalEstimatedMatches = 0, requestedCount, offsetInit, tmpSize, response, searchType, cloudProblem = False},

    nresultsperpage = Switch[searchType = Lookup[params, "SearchType"],
        "News",
            100
        ,

        "Video" | "Videos",
            105
        ,

        "Image" | "Images" | "Pictures",
            150
        ,

        _,
            50
    ];
    params = KeyDrop[params, "SearchType"];
    params["count"] = FromDigits[params["count"]];
    offsetInit = params["offset"] = FromDigits[params["offset"]];
    calls = Quotient[params["count"], nresultsperpage];
    residual = params["count"] - (calls*nresultsperpage);
    bar = PrintTemporary[ProgressIndicator[Dynamic[progress], {0, calls}]];

    elementsLeft = requestedCount = Lookup[params, "count"];

    Catch[
        While[firstIteration || (elementsLeft > 0 && totalEstimatedMatches > offsetInit + Length[result]),
            firstIteration = False;

            tmpSize = Min[elementsLeft, nresultsperpage];
            params["offset"] = params["offset"] + progress * nresultsperpage;
            params["count"] = tmpSize;
            response = IntegratedServices`RemoteServiceExecute[tag, If[wisQ,"BingWebImageSearch","BingSearch"], "RawSearch", Association[Normal[ToString /@ params]]];

            Which[
                MatchQ[response, $Canceled],
                    cloudProblem = True;
                    Throw[$Failed, "cloudConnectProblem"]
                ,

                And[!MatchQ[response,False],!MatchQ[response,$Failed],!MatchQ[Head[response],Symbol]] ,
                    progress += 1;

                    totalEstimatedMatches = Lookup[Lookup[response, "webPages", response], "totalEstimatedMatches", 0]; (*this value gets more precise every time the offset increases*)
                    items = BingSearchFunctions`Private`BSGetItemsFromResponse[response, searchType];

                    tmpSize = Min[elementsLeft, Length[items]];
                    elementsLeft -= tmpSize;
                    items = Take[items, UpTo[tmpSize]];
                    result = Join[result, items]
                ,

                True,
                    Throw["exception"]
            ]
        ];
    ,
        "cloudConnectProblem" | "noMoreResults"
    ];

    progress = calls;
    NotebookDelete[bar];

    If[cloudProblem,
        $Failed
    ,
        result
    ]
]

GCSPaginationCalls[tag_, p_, wisQ_:False]:= Module[
    {calls, residual, progress = 0, rawdata, data, params = Association[p], items = {}, bar, nresultsperpage = 10, paramsList, currentCall, cloudProblem = False},

    If[MatchQ[params["searchType"], "web"], params = KeyDrop[params, "searchType"]];
    params = KeyDrop[params,"Elements"];
    params["num"] = FromDigits[params["num"]];
    params["start"] = FromDigits[params["start"]];
    calls = Quotient[params["num"], nresultsperpage];
    residual = params["num"] - (calls*nresultsperpage);
    paramsList = Table[
        Block[
            {tmpParam = params},
            tmpParam["start"] += 10 * currentCall;
            tmpParam["num"] = If[MatchQ[currentCall, calls], residual, nresultsperpage];
            Normal[ToString /@ tmpParam]
        ],
        {currentCall, 0, If[Positive[residual], calls, calls - 1]}
    ];
    bar = PrintTemporary[ProgressIndicator[Dynamic[progress], {0, Length[paramsList]}]];
    Catch[
        (
            rawdata = IntegratedServices`RemoteServiceExecute[tag, If[wisQ,"GoogleWebImageSearch","GoogleCustomSearch"], "RawSearch", Association[#]];

            Which[
                AssociationQ[rawdata],
                    data = Lookup[rawdata, "items", {}];
                    If[!Positive[Length[data]], progress = Length[paramsList]; Throw[$Failed, "noMoreResults"]];

                    items = Join[items,data]
                ,

                MatchQ[rawdata, $Canceled],
                    cloudProblem = True;
                    Throw[$Failed, "cloudConnectProblem"]
                ,
                True,
                    Throw["exception"]
            ];
            progress += 1
        ) & /@ paramsList
    ,
        "cloudConnectProblem" | "noMoreResults"
    ];

    NotebookDelete[bar];

    If[cloudProblem,
        $Failed
    ,
        items
    ]
]

End[]
EndPackage[]
