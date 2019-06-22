(* Created with the Wolfram Language : www.wolfram.com *)

BeginPackage["BingSearchFunctions`"];

BSadultMap::usage = "";
BSBooleanParse::usage = "";
BSCookedImport::usage = "";
BSFileFormatParse::usage = "";
BSFormatRequestParameters::usage = "";
BSGetDuration::usage = "";
BSGetPathFromType::usage = "";
BSPaginationCalls::usage = "";
BSParseHTMLErrors::usage = "";
BSSearchParameter::usage = "";
BSSiteParse::usage = "";

Begin["`Private`"];

Options[BSFormatRequestParameters] = {"IntegratedServiceQ" -> False}
BSFormatRequestParameters[args_, OptionsPattern[]] := Module[
    {args2 = args, params, invalidParameters, integratedServiceQ, tag = Null, country, language},
    
    invalidParameters = Select[Keys@args2,!MemberQ[{"Query", "FileFormat", "FileType", "Site", Language, "Language", "Country", "ContentFiltering", "SearchType", "MaxItems", MaxItems, "StartIndex", "Elements"}, #]&];
    integratedServiceQ = TrueQ[OptionValue["IntegratedServiceQ"]];
    If[ integratedServiceQ,
        tag = Symbol["WebSearch"]
    ];
    If[ Length[invalidParameters]>0,
        BSReturnMessage[tag, "noget", ServiceObject, #, "BingSearch"] & /@ invalidParameters;
        Throw[$Failed]
    ];
    If[ KeyExistsQ[args2,"SearchType"],
        If[ !MemberQ[{"News", "Web", "Image", "Video", "RelatedSearch", "SpellingSuggestions", "Images", "Videos", "Pictures"}, Lookup[args2, "SearchType"]],
            BSReturnMessage[tag, "nval", ServiceExecute, "SearchType", "BingSearch"];
            Throw[$Failed]
        ]
    ];
    params = <| "SearchType" -> Lookup[args2, "SearchType", "Web"] |>;

    params["SearchTypePath"] = BSGetPathFromType[Lookup[params, "SearchType"]];

    Switch[Lookup[params, "SearchType"],
        "Web",
            params["responseFilter"] = "Webpages"
        ,

        "RelatedSearch",
            params["responseFilter"] = "RelatedSearches"
        ,

        "SpellingSuggestions",
            params["responseFilter"] = "SpellSuggestions"
    ];

    params["count"] = Lookup[args2, MaxItems, Lookup[args2, "MaxItems", 10]];
    If[!IntegerQ[params["count"]] || !Positive[params["count"]],
        BSReturnMessage[tag, "nval", ServiceExecute, MaxItems, "BingSearch"];
        Throw[$Failed]
    ];

    params["offset"] = Lookup[args2, "StartIndex", 0];
    If[!IntegerQ[params["offset"]] || Negative[params["offset"]],
        BSReturnMessage[tag, "nval", ServiceExecute, "StartIndex", "BingSearch"];
        Throw[$Failed]
    ];

    params["q"] = ToString@BSBooleanParse@Lookup[args2, "Query", BSReturnMessage[tag, "nparam", ServiceExecute, "Query"]; Throw[$Failed]];

    If[KeyExistsQ[args2, "Elements"],
        If[!MemberQ[{"Data", "FileFormat", "FileFormats", "Image", "Images", "Thumbnail", "Thumbnails", "ImageLink", "ImageLinks", "ImageHyperlink", "ImageHyperlinks","ImageThumbnailsLink", "ImageThumbnailsLinks", "PageHyperlink", "PageHyperlinks", "PageTitle", "PageTitles", "Elements"}, Lookup[args2, "Elements"]],
            BSReturnMessage[tag, "nval", ServiceExecute, "Elements", "BingSearch"];
            Throw[$Failed]
        ]
    ];

    If[KeyExistsQ[args2, "Site"],
        params["q"] = params["q"] <> " " <> BSSiteParse[Lookup[args2, "Site"]]
    ];

    If[KeyExistsQ[args2, "FileFormat"] || KeyExistsQ[args2, "FileType"],
        params["q"] = params["q"] <> " " <> BSFileFormatParse[Lookup[args2, "FileFormat", Lookup[args2, "FileType"]]]
    ];

    If[KeyExistsQ[args2, "Country"],
        country = Which[
            MatchQ[Lookup[args2, "Country"], _Entity],
                Lookup[args2, "Country"]
            ,

            StringQ[Lookup[args2, "Country"]],
                Interpreter["Country"][Lookup[args2, "Country"]]
            ,

            True,
                Missing["NotAvailable"]
        ];

        country = If[MatchQ[country, _Entity],
            " location:" <> country["CountryCode"]
        ,
            BSReturnMessage[tag, "nval", ServiceExecute, "Country", "BingSearch"];
            ""
        ];

        If[MissingQ[country], country = ""];

        params["q"] = params["q"] <> country
    ];

    If[KeyExistsQ[args2, Language] || KeyExistsQ[args2, "Language"],
        language = Lookup[args2, Language, Lookup[args2, "Language"]];
        language = Which[
            MatchQ[language, _Entity],
                language
            ,

            StringQ[language],
                Interpreter["Language"][language]
            ,

            True,
                Missing["NotAvailable"]
        ];

        language = If[MatchQ[language, _Entity] && KeyExistsQ[BSLanguageRules, language],
            " language:" <> Lookup[BSLanguageRules, language]
        ,
            BSReturnMessage[tag, "nval", ServiceExecute, "Language", "BingSearch"];
            ""
        ];

        params["q"] = params["q"] <> language
    ];

    params["safeSearch"] = If[KeyExistsQ[args2,"ContentFiltering"],
        If[StringQ[Lookup[args2, "ContentFiltering"] && KeyExistsQ[BSadultMap, ToLowerCase[Lookup[args2, "ContentFiltering"]]]],
            Lookup[BSadultMap, ToLowerCase[Lookup[args2, "ContentFiltering"]]]
        ,
            BSReturnMessage[tag, "nval", ServiceExecute, "ContentFiltering", "BingSearch"];
        ]
    ,
        "Strict"
    ];

    params = ToString /@ params;

    Normal[params]
]

BSReturnMessage[newName_, errorcode_, origService_, params___] := With[
    {msg = Once[MessageName[origService, errorcode]]},
    
    If[ MatchQ[newName, Null],
        Message[MessageName[origService, errorcode], params],
        MessageName[newName, errorcode] = msg;
        Message[MessageName[newName, errorcode], params];
        Unset[MessageName[newName, errorcode]]
    ]
]

BSGetPathFromType[name_] :=
    Switch[name,
        "Image"|"Images"|"Picture"|"Pictures","images/search",
        "News","news/search",
        "Video"|"Videos"|"Clips","videos/search",
        "RelatedSearch"|"SpellingSuggestions"|"Web","search",
        __, Missing["NotAvailable"]
    ]

BSBooleanParse[e_] :=
    e //. {Verbatim[Alternatives][x_] :> x,
      Verbatim[Alternatives][x_, y__] :>
       "" ~~ x ~~ " OR " ~~ Alternatives[y] ~~ "",
      Verbatim[Except][x_] :> "-" ~~ x, List[x_] :> x,
      List[x_, y__] :> "" ~~ x ~~ " AND " ~~ List[y] ~~ ""}

BSSiteParse[e_] := Module[
    {tmp},

    tmp = e /. {Verbatim[Except][Verbatim[Alternatives][x___]] :>
    List[Sequence @@ Except /@ List[x]],
    Verbatim[Except][List[x___]] :>
    Alternatives[Sequence @@ Except /@ List[x]]};
    tmp = BSBooleanParse[tmp];
    tmp = StringJoin[
      Riffle[(If[ ! MemberQ[{"AND", "OR", "("}, #],
                  If[ StringMatchQ[#, "-" ~~ ___],
                      "-site:" ~~ StringDrop[#, 1],
                      "site:" ~~ #
                  ],
                  #
              ]) & /@
        Flatten[(If[ StringMatchQ[#, "(" ~~ ___],
                     {"(",
                     StringDrop[#, 1]},
                     #
                 ] & /@ StringSplit[tmp])], " "]];
    StringReplace[tmp, {"( " -> "", ")" -> ""}]
]

BSFileFormatParse[e_] := Module[
    {tmp},

    tmp = e /. {Verbatim[Except][Verbatim[Alternatives][x___]] :>
    List[Sequence @@ Except /@ List[x]],
    Verbatim[Except][List[x___]] :>
    Alternatives[Sequence @@ Except /@ List[x]]};
    tmp = tmp /. Entity["FileFormat", x_] :> StringReplace[EntityValue[Entity["FileFormat", x], "Extension"], "." -> ""];
    tmp = BSBooleanParse[tmp];
    tmp = StringReplace[tmp, "." -> ""];
    tmp = StringJoin[
      Riffle[(If[ ! MemberQ[{"AND", "OR", "("}, #],
                  If[ StringMatchQ[#, "-" ~~ ___],
                      "-filetype:" ~~ StringDrop[#, 1],
                      "filetype:" ~~ #
                  ],
                  #
              ]) & /@
        Flatten[(If[ StringMatchQ[#, "(" ~~ ___],
                     {"(",
                     StringDrop[#, 1]},
                     #
                 ] & /@ StringSplit[tmp])], " "]];
    StringReplace[tmp, {"( " -> "", ")" -> ""}]
]

BSCookedImport[items_,params_,args2_] := Module[
    {total = Length[items], current = 0, ds, pt, pos, interpreted, tmpData},

    If[ !MatchQ[total,0],
        Switch[Lookup[params, "SearchType"],
            "News",
                Dataset[
                    Association[{
                        "Title" -> #["name"],
                        "Source" -> (KeyMap[BSRenameNewsSource, #] & /@ #["provider"]),
                        "Link" -> BSFormatURL[#["url"]],
                        "Summary" -> #["description"]
                    }] & /@ KeyTake[items, {"name", "provider", "url", "description"}]
                ]
            ,

            "Web",
                Dataset[
                    Association[{
                        "Title" -> #["name"],
                        "Snippet" -> #["snippet"],
                        "Link" -> BSFormatURL[#["url"]]
                    }] & /@ KeyTake[items, {"name", "url", "snippet"}]
                ]
            ,

            "Image" | "Images" | "Pictures",
                Switch[Lookup[args2, "Elements", "Elements"],
                    "FileFormat" | "FileFormats",
                        tmpData = Lookup[items, "encodingFormat", {}];
                        interpreted = Interpreter["FileFormat"][tmpData];
                        pos = Position[interpreted, _Failure | _Missing, {1}, Heads -> False];
                        ReplacePart[interpreted, MapThread[Rule, {pos, Extract[tmpData, pos]}]]
                    ,

                    "Image" | "Images",
                        BSImportImages[Lookup[items, "contentUrl", {}]]
                    ,

                    "ImageHyperlink" | "ImageHyperlinks" | "ImageLink" | "ImageLinks",
                        BSFormatURL /@ Lookup[items, "contentUrl", {}]
                    ,

                    "ImageThumbnailsLink" | "ImageThumbnailsLinks",
                        BSFormatURL /@ Lookup[items, "thumbnailUrl", {}]
                    ,
                    
                    "PageHyperlink" | "PageHyperlinks",
                        BSFormatURL /@ Lookup[items, "hostPageUrl", {}]
                    ,

                    "PageTitle" | "PageTitles",
                        Lookup[items, "name", {}]
                    ,

                    "Thumbnail" | "Thumbnails",
                        BSImportImages[Lookup[items, "thumbnailUrl"]]
                    ,
                    
                    "Elements",
                        Which[
                            MatchQ[total, 1],
                                pt = PrintTemporary[Internal`LoadingPanel["Downloading 1 image...."]]
                            ,

                            total > 1,
                                pt = PrintTemporary@Dynamic[Internal`LoadingPanel[StringForm["Downloading `1` of `2` images....", current, total]]]
                        ];
                        
                        ds = Dataset[
                            (
                                current += 1;
                                Association[{
                                    "Thumbnail" -> Import[#["thumbnailUrl"]],
                                    "PageTitle" -> #["name"],
                                    "PageHyperlink" -> BSFormatURL[#["hostPageUrl"]],
                                    "ImageHyperlink" -> BSFormatURL[#["contentUrl"]]
                                }]
                            ) & /@ KeyTake[items, {"hostPageUrl", "name", "contentUrl", "thumbnailUrl"}]
                        ];

                        If[total >= 1,
                            NotebookDelete[pt]
                        ];
                        ds
                    ,

                    _,
                        Throw[$Failed]
                ]
            ,

            "Videos"|"Video",
                Which[
                    MatchQ[total, 1],
                        pt = PrintTemporary[Internal`LoadingPanel["Downloading 1 image...."]]
                    ,

                    total > 1,
                        pt = PrintTemporary@Dynamic[Internal`LoadingPanel[StringForm["Downloading `1` of `2` images....", current, total]]]
                ];

                ds = Dataset[
                    (
                        current += 1;
                        Association[{
                            "Thumbnail" -> EventHandler[Import[#["thumbnailUrl"]], {"MouseClicked" :> SystemOpen[#["name"]]}],
                            "Title" -> #["name"],
                            "Link" -> BSFormatURL[#["contentUrl"]],
                            "RunTime" -> BSGetDuration[#["duration"]]
                        }]
                    ) & /@ KeyTake[items, {"name", "contentUrl", "duration", "thumbnailUrl"}]
                ];
                
                If[total >= 1,
                    NotebookDelete[pt]
                ];
                ds
            ,

            "RelatedSearch",
                Dataset[
                    Association[{
                        "Tittle" -> #["displayText"],
                        "Link"-> BSFormatURL[#["webSearchUrl"]]
                    }] & /@ KeyTake[items, {"displayText", "webSearchUrl"}]
                ]
            ,

            "SpellingSuggestions",
                Dataset[
                    Association[{
                        "Value" -> #
                    }] & /@ Lookup[items, "displayText", {}]
                ]
        ]
    ,
        Dataset[Association[items]]
    ]
]

BSImportImages[list_List] := Module[
    {current = 0, images, length = Length[list], pt},

    If[length > 0,
        Which[
            MatchQ[length, 1],
                pt = PrintTemporary[Internal`LoadingPanel["Downloading 1 image...."]]
            ,

            length > 1,
                pt = PrintTemporary@Dynamic[Internal`LoadingPanel[StringForm["Downloading `1` of `2` images....", current, length]]]
        ];
        images = (current += 1; Import[#]) & /@ list;
        NotebookDelete[pt];
        images
    ,
        {}
    ]
]

BSPaginationCalls[id_,prop_,p_] := Module[
    {calls, residual, progress = 0, rawdata, items, params = p, nresultsperpage, bar, elementsLeft, firstIteration = True, result = {}, totalEstimatedMatches = 0, requestedCount, offsetInit, tmpSize, response, searchType},

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
            rawdata = KeyClient`rawkeydata[id, prop, Normal[ToString /@ params]];
            response = BingSearch`Private`BSFormatResults[rawdata];

            progress += 1;

            totalEstimatedMatches = Lookup[Lookup[response, "webPages", response], "totalEstimatedMatches", 0]; (*this value gets more precise every time the offset increases*)
            items = BSGetItemsFromResponse[response, searchType];
            
            tmpSize = Min[elementsLeft, Length[items]];
            elementsLeft -= tmpSize;
            items = Take[items, UpTo[tmpSize]];
            result = Join[result, items]
        ];
    ];

    progress = calls;
    NotebookDelete[bar];
    result
]

BSGetItemsFromResponse[response_, searchType_] := Switch[Lookup[response, "_type"],
    "SearchResponse",
        Switch[searchType,
            "RelatedSearch",
                Lookup[Lookup[response, "relatedSearches", {}], "value", {}]
            ,

            "SpellingSuggestions",
                Lookup[Lookup[response, "spellSuggestions", {}], "value", {}]
            ,

            _,
                Lookup[Lookup[response, "webPages", {}], "value", {}]
        ]
    ,

    "Images" | "News",
        Lookup[response, "value", {}]
    ,

    "Video" | "Videos",
        Lookup[response, "value",{}]
    ,

    _,
        {}
]

BSRenameNewsSource[key_] := Block[
    {},

    Which[
        MatchQ[key, "_type"],
            "Type"
        ,

        MatchQ[key, "name"],
            "Name"
        ,

        StringQ[key],
            Capitalize[key]
        ,

        True,
            key
    ]
]

SetAttributes[BSFormatURL, Listable];
BSFormatURL[url_String?StringQ] := Block[
    {tmp},

    tmp = StringSplit[url, "r="];
    tmp = If[Length[tmp] > 1,
        tmp = tmp[[2]];
        URL[URLDecode[First[StringSplit[tmp, "&"]]]]
    ,
        url
    ];

    tmp
]

BSFormatURL[other_] := other

BSGetDuration[duration_String] := Block[
    {time = Last[StringSplit[duration, "T"]],  hours, minutes, seconds},

    If[StringContainsQ[time, "H"],
        time = StringSplit[time, "H"];
        hours = ToExpression@First@time;
        time = Last@time
    ,
        hours = 0
    ];

    If[StringContainsQ[time, "M"],
        time = StringSplit[time, "M"];
        minutes = ToExpression@First@time;
        time = Last@time
    ,
        minutes = 0
    ];

    If[StringContainsQ[time, "S"],
        time = StringSplit[time, "S"];
        seconds = ToExpression@First@time;
        time = Last@time
    ,
        seconds = 0
    ];

    Quantity[MixedMagnitude[{hours, minutes, seconds}], MixedUnit[{"Hours", "Minutes", "Seconds"}]]
] /; StringStartsQ[duration, "P"] && StringContainsQ[duration, "T"]

BSGetDuration[___] := Missing["NotAvailable"]

BSadultMap = {
    "Off" -> "Off",
    "off" -> "Off",
    "Medium" -> "Moderate",
    "medium" -> "Moderate",
    "High" -> "Strict",
    "high" -> "Strict"
};

BSLanguageRules = {
    Entity["Language", "Afar"] -> "aa",
    Entity["Language", "Abkhaz"] -> "ab",
    Entity["Language", "Avestan"] -> "ae",
    Entity["Language", "Afrikaans"] -> "af",
    Entity["Language", "Akan"] -> "ak",
    Entity["Language", "Amharic"] -> "am",
    Entity["Language", "Aragonese"] -> "an",
    Entity["Language", "Arabic"] -> "ar",
    Entity["Language", "Assamese"] -> "as",
    Entity["Language", "Avar"] -> "av",
    Entity["Language", "AymaraCentral"] -> "ay",
    Entity["Language", "AzerbaijaniNorth"] -> "az",
    Entity["Language", "Bashkir"] -> "ba",
    Entity["Language", "Belarusan"] -> "be",
    Entity["Language", "Bulgarian"] -> "bg",
    Entity["Language", "Bislama"] -> "bi",
    Entity["Language", "TibetanCentral"] -> "bo",
    Entity["Language", "Breton"] -> "br",
    Entity["Language", "CatalanValencianBalear"] -> "ca",
    Entity["Language", "Chamorro"] -> "ch",
    Entity["Language", "Corsican"] -> "co",
    Entity["Language", "Czech"] -> "cs",
    Entity["Language", "Welsh"] -> "cy",
    Entity["Language", "Danish"] -> "da",
    Entity["Language", "German"] -> "de",
    Entity["Language", "Dzongkha"] -> "dz",
    Entity["Language", "Ewe"] -> "ee",
    Entity["Language", "Greek"] -> "el",
    Entity["Language", "English"] -> "en",
    Entity["Language", "Esperanto"] -> "eo",
    Entity["Language", "Spanish"] -> "es",
    Entity["Language", "Estonian"] -> "et",
    Entity["Language", "Basque"] -> "eu",
    Entity["Language", "Finnish"] -> "fi",
    Entity["Language", "Faroese"] -> "fo",
    Entity["Language", "French"] -> "fr",
    Entity["Language", "FrisianWestern"] -> "fy",
    Entity["Language", "IrishGaelic"] -> "ga",
    Entity["Language", "ScottishGaelic"] -> "gd",
    Entity["Language", "Manx"] -> "gv",
    Entity["Language", "Hausa"] -> "ha",
    Entity["Language", "Hebrew"] -> "he",
    Entity["Language", "Hindi"] -> "hi",
    Entity["Language", "MotuHiri"] -> "ho",
    Entity["Language", "Hungarian"] -> "hu",
    Entity["Language", "Armenian"] -> "hy",
    Entity["Language", "Herero"] -> "hz",
    Entity["Language", "Interlingua"] -> "ia",
    Entity["Language", "Indonesian"] -> "id",
    Entity["Language", "Igbo"] -> "ig",
    Entity["Language", "YiSichuan"] -> "ii",
    Entity["Language", "InupiatunNorthAlaskan"] -> "ik",
    Entity["Language", "Icelandic"] -> "is",
    Entity["Language", "Italian"] -> "it",
    Entity["Language", "Japanese"] -> "ja",
    Entity["Language", "Georgian"] -> "ka",
    Entity["Language", "Kwanyama"] -> "kj",
    Entity["Language", "Kazakh"] -> "kk",
    Entity["Language", "KhmerCentral"] -> "km",
    Entity["Language", "Korean"] -> "ko",
    Entity["Language", "Kashmiri"] -> "ks",
    Entity["Language", "Cornish"] -> "kw",
    Entity["Language", "Latin"] -> "la",
    Entity["Language", "Luxembourgeois"] -> "lb",
    Entity["Language", "Limburgisch"] -> "li",
    Entity["Language", "Lingala"] -> "ln",
    Entity["Language", "Lithuanian"] -> "lt",
    Entity["Language", "Latvian"] -> "lv",
    Entity["Language", "Marshallese"] -> "mh",
    Entity["Language", "Malayalam"] -> "ml",
    Entity["Language", "Marathi"] -> "mr",
    Entity["Language", "Maltese"] -> "mt",
    Entity["Language", "Burmese"] -> "my",
    Entity["Language", "NorwegianBokmal"] -> "nb",
    Entity["Language", "Nepali"] -> "ne",
    Entity["Language", "Ndonga"] -> "ng",
    Entity["Language", "Dutch"] -> "nl",
    Entity["Language", "NorwegianNynorsk"] -> "nn",
    Entity["Language", "Norwegian"] -> "no",
    Entity["Language", "Navajo"] -> "nv",
    Entity["Language", "Oriya"] -> "or",
    Entity["Language", "Polish"] -> "pl",
    Entity["Language", "Portuguese"] -> "pt",
    Entity["Language", "Romanian"] -> "ro",
    Entity["Language", "Russian"] -> "ru",
    Entity["Language", "Sanskrit"] -> "sa",
    Entity["Language", "Sango"] -> "sg",
    Entity["Language", "Sinhala"] -> "si",
    Entity["Language", "Slovak"] -> "sk",
    Entity["Language", "Samoan"] -> "sm",
    Entity["Language", "Shona"] -> "sn",
    Entity["Language", "Somali"] -> "so",
    Entity["Language", "Serbian"] -> "sr",
    Entity["Language", "Swati"] -> "ss",
    Entity["Language", "SothoSouthern"] -> "st",
    Entity["Language", "Swedish"] -> "sv",
    Entity["Language", "Swahili"] -> "sw",
    Entity["Language", "Tamil"] -> "ta",
    Entity["Language", "Thai"] -> "th",
    Entity["Language", "Tigrigna"] -> "ti",
    Entity["Language", "Tswana"] -> "tn",
    Entity["Language", "Turkish"] -> "tr",
    Entity["Language", "Tsonga"] -> "ts",
    Entity["Language", "Tatar"] -> "tt",
    Entity["Language", "Tahitian"] -> "ty",
    Entity["Language", "Ukrainian"] -> "uk",
    Entity["Language", "Urdu"] -> "ur",
    Entity["Language", "Venda"] -> "ve",
    Entity["Language", "Vietnamese"] -> "vi",
    Entity["Language", "Walloon"] -> "wa",
    Entity["Language", "Wolof"] -> "wo",
    Entity["Language", "Xhosa"] -> "xh",
    Entity["Language", "YiddishEastern"] -> "yi",
    Entity["Language", "Yoruba"] -> "yo",
    Entity["Language", "ChineseMandarin"] -> "zh",
    Entity["Language", "Zulu"] -> "zu"
}

End[];

EndPackage[];
