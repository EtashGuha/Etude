(* Created with the Wolfram Language : www.wolfram.com *)

BeginPackage["GoogleCustomSearchFunctions`"];

GCSAllowedCountryValues::usage = "";
GCSAllowedLanguageValues::usage = "";
GCSBooleanParser::usage = "";
GCSCookedImport::usage = "";
GCSCountryMap::usage = "";
GCSFiletypeParser::usage = "";
GCSFormatRequestParameters::usage = "";
GCSLangMap::usage = "";
GCSPaginationCalls::usage = "";
GCSSiteParser::usage = "";

Begin["`Private`"];

Options[GCSFormatRequestParameters] = {"IntegratedServiceQ" -> False}
GCSFormatRequestParameters[args_, OptionsPattern[]] := Module[
    {args2 = args, params, invalidParameters, integratedServiceQ, tag = Null, startIndex=1, limit=0, type="web", language, query, fileType, site, safe, country, dr, prefix, mag, unit},

    invalidParameters = Select[Keys[args2], !MemberQ[{"Query", "FileFormat", "FileType", "Site", Language, "Language", "Country", "ContentFiltering", "SearchType", "MaxItems", MaxItems, "StartIndex", "DateRestrict", "Elements", "IntegratedServiceQ"}, #] &];
    integratedServiceQ = TrueQ[OptionValue["IntegratedServiceQ"]];
    If[integratedServiceQ,
        tag = WebSearch
    ];
    If[Length[invalidParameters]>0,
        GCSReturnMessage[tag, "noget", ServiceObject, #, "GoogleCustomSearch"] & /@ invalidParameters;
        Throw[$Failed]
    ];

    If[KeyExistsQ[args2, "Query"],
        query = Lookup[args2, "Query"];
        query = GCSBooleanParser[query];
    ,
        GCSReturnMessage[tag, "nparam", ServiceExecute, "Query"];
        Throw[$Failed]
    ];

    If[KeyExistsQ[args2, "FileFormat"] || KeyExistsQ[args2, "FileType"],
        query = query <> " " <> GCSFiletypeParser[Lookup[args2, "FileFormat", Lookup[args2, "FileType"]]]
    ];

    If[KeyExistsQ[args2, "Site"],
        site = Lookup[args2, "Site"];
        query = query <> " " <> GCSSiteParser[site]
    ];

    params = <| "q" -> query |>;

    If[KeyExistsQ[args2, Language] || KeyExistsQ[args2, "Language"],
        language = Lookup[args2, Language, Lookup[args2, "Language"]];
        language = Which[
            StringQ[language],
                Interpreter["Language"][language]
            ,

            MatchQ[language, Entity["Language", _]],
                language
            ,

            True,
                Missing["NotAvailable "]
        ];

        If[MatchQ[language, Entity["Language", _]] && KeyExistsQ[GCSLangMap, language],
            params["lr"] = Lookup[GCSLangMap, language]
        ,
            GCSReturnMessage[tag, "nval", ServiceExecute, "Language", "GoogleCustomSearch"]
        ]
    ];
    If[KeyExistsQ[args2,"Country"],
        country = Lookup[args2, "Country"];
        country = Which[
            StringQ[country],
                Interpreter["Country"][country]
            ,

            MatchQ[country, Entity["Country", _]],
                country
            ,

            True,
                Missing["NotAvailable "]
        ];

        If[MatchQ[country, Entity["Country", _]] && KeyExistsQ[GCSCountryMap, country],
            params["cr"] = Lookup[GCSCountryMap, country]
        ,
            GCSReturnMessage[tag, "nval", ServiceExecute, "Country", "GoogleCustomSearch"]
        ]
    ];

    If[KeyExistsQ[args2, "ContentFiltering"],
        If[StringQ[safe = Lookup[args2, "ContentFiltering"]] && MemberQ[{"off", "medium", "high"}, safe = ToLowerCase[safe]],
            params["safe"] = safe;
        ,
            GCSReturnMessage[tag, "nval", ServiceExecute, "ContentFiltering", "GoogleCustomSearch"];
            Throw[$Failed]
        ]
    ];

    If[KeyExistsQ[args2, "SearchType"],
        If[StringQ[type = Lookup[args2, "SearchType"]] && MemberQ[{"web", "image", "images"}, type = ToLowerCase[type]],
            params["searchType"] = If[MatchQ[type, "web"], type, type = "image"]
        ,
            GCSReturnMessage[tag, "nval", ServiceExecute, "SearchType", "GoogleCustomSearch"];
            Throw[$Failed]
        ]
    ,
        params["searchType"] = "web"
    ];

    If[KeyExistsQ[args2, "DateRestrict"],
        dr = Lookup[args2, "DateRestrict"];
        prefix = "";
        mag = 0;
        Which[
            IntegerQ[dr],
                prefix = "d";
                mag = dr
            ,

            MatchQ[dr, _Quantity],
                mag = QuantityMagnitude[dr];
                unit = QuantityUnit[dr];
                prefix = Switch[unit,
                    "Days",
                        "d"
                    ,

                    "Weeks",
                        "w"
                    ,

                    "Months",
                        "m"
                    ,

                    "Years",
                        "y"
                    ,

                    _,
                        GCSReturnMessage[tag, "nval", ServiceExecute, "DateRestrict", "GoogleCustomSearch"];
                        Throw[$Failed]
                ]
            ,

            True,
                GCSReturnMessage[tag, "nval", ServiceExecute, "DateRestrict", "GoogleCustomSearch"];
                Throw[$Failed]
        ];

        params["dateRestrict"] = prefix <> ToString[mag]
    ];

    params["num"] = Lookup[args2, MaxItems, Lookup[args2, "MaxItems", 10]];
    If[!IntegerQ[params["num"]] || !Positive[params["num"]],
        GCSReturnMessage[tag, "nval", ServiceExecute, MaxItems, "GoogleCustomSearch"];
        Throw[$Failed]
    ];

    If[KeyExistsQ[args2,"Elements"],
        If[!MemberQ[{"Data", "FileFormat", "FileFormats", "Image", "Images", "Thumbnail", "Thumbnails", "ImageLink", "ImageLinks", "ImageHyperlink", "ImageHyperlinks","ImageThumbnailsLink", "ImageThumbnailsLinks", "PageHyperlink", "PageHyperlinks", "PageTitle", "PageTitles", "Elements"}, Lookup[args2, "Elements"]],
            GCSReturnMessage[tag, "nval", ServiceExecute, "Elements", "GoogleCustomSearch"];
            Throw[$Failed]
        ]
    ];
    
    params["start"] = Lookup[args2, "StartIndex", 1];
    If[!IntegerQ[params["start"]] || !Positive[params["start"]],
        GCSReturnMessage[tag, "nval", ServiceExecute, "StartIndex", "GoogleCustomSearch"];
        Throw[$Failed]
    ];

    params = ToString /@ params;

    {Normal[params], type}
]

GCSReturnMessage[newName_, errorcode_, origService_, params___] := With[
    {msg = Once[MessageName[origService, errorcode]]},

    If[ MatchQ[newName, Null],
        Message[MessageName[origService, errorcode], params],
        MessageName[newName, errorcode] = msg;
        Message[MessageName[newName, errorcode], params];
        Unset[MessageName[newName, errorcode]]
    ]
]

GCSBooleanParser[e_] :=
        e //. {Verbatim[Alternatives][x_] :> x,
        Verbatim[Alternatives][x_, y__] :>
        "(" ~~ x ~~ " OR " ~~ Alternatives[y] ~~ ")",
        Verbatim[Except][x_] :> "-" ~~ x, List[x_] :> x,
        List[x_, y__] :> "(" ~~ x ~~ " AND " ~~ List[y] ~~ ")"}

GCSSiteParser[e_] :=
        Module[ {tmp},
                (
                tmp = e /. {Verbatim[Except][Verbatim[Alternatives][x___]] :>
                List[Sequence @@ Except /@ List[x]],
                Verbatim[Except][List[x___]] :>
                Alternatives[Sequence @@ Except /@ List[x]]};
                tmp = GCSBooleanParser[tmp];
                tmp = StringJoin[Riffle[(If[ ! MemberQ[{"AND", "OR", "("}, #],
                                                                         If[ StringMatchQ[#, "-" ~~ ___],
                                                                                 "-site:" ~~ StringDrop[#, 1],
                                                                                 "site:" ~~ #
                                                                         ],
                                                                         #
                                                                 ]) & /@
                                        Flatten[(If[ StringMatchQ[#, "(" ~~ ___],
                                                                 {"(",StringDrop[#, 1]},
                                                                 #
                                                         ] & /@ StringSplit[tmp])], " "]];
                StringReplace[tmp, {"( " -> "", ")" -> ""}]
                )
        ]

GCSFiletypeParser[e_] :=
        Module[ {tmp},
                (
                tmp = e /. {Verbatim[Except][Verbatim[Alternatives][x___]] :>
                List[Sequence @@ Except /@ List[x]],
                Verbatim[Except][List[x___]] :>
                Alternatives[Sequence @@ Except /@ List[x]]};
                [tmp];
                tmp = tmp /. Entity["FileFormat", x_] :> StringReplace[EntityValue[Entity["FileFormat", x], "Extension"], "." -> ""];
                tmp = GCSBooleanParser[tmp];
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
                )
        ]

GCSPaginationCalls[id_,prop_,p_] := Module[
    {calls, residual, progress = 0, rawdata, data, params = Association[p], items = {}, bar, nresultsperpage = 10, paramsList},

    If[MatchQ[params["searchType"], "web"], params = KeyDrop[params, "searchType"]];
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
            rawdata = KeyClient`rawkeydata[id, prop, #];
            data = GoogleCustomSearch`Private`GSFormatResults[rawdata];
            data = Lookup[data, "items", {}];

            If[!Positive[Length[data]], progress = Length[paramsList]; Throw[$Failed, "noMoreResults"]];

            items = Join[items, data];
            progress += 1
        ) & /@ paramsList
    ,
        "noMoreResults"
    ];

    NotebookDelete[bar];

    items
]

GCSCookedImport[i_,args_,type_] := Module[
    {items = i, data, current = 0, total, pt, tmpData, interpreted, pos},

    total = Length[items];
    Switch[type,
        "web",
            data = Association[
                "Title" -> #["title"],
                "Snippet" -> #["snippet"],
                "Link" -> URL[#["link"]],
                "Metadata" -> Lookup[Lookup[#, "pagemap", {}], "metatags", Missing["NotAvailable"]]
            ] & /@ KeyTake[items, {"title", "snippet", "pagemap", "link"}];
            Dataset[data]
        ,

        "image",
            data = Association[
                "ThumbnailLink" -> URL[#["image"]["thumbnailLink"]],
                "PageTitle" -> #["title"],
                "PageHyperlink" -> URL[#["image"]["contextLink"]],
                "ImageHyperlink" -> URL[#["link"]],
                "FileFormat" -> Lookup[#, "mime", Missing["NotAvailable"]]
            ] & /@ KeyTake[items, {"thumbnailLink", "title", "link", "contextLink", "image", "fileFormat", "mime"}];

            Switch[Lookup[args, "Elements", Default],
                "FileFormat" | "FileFormats",
                    tmpData = Lookup[data, "FileFormat", {}];
                    interpreted = Interpreter["FileFormat"][tmpData];
                    pos = Position[interpreted, _Failure | _Missing, {1}, Heads -> False];
                    ReplacePart[interpreted, MapThread[Rule, {pos, Extract[tmpData, pos]}]]
                ,

                "Image" | "Images",
                    If[total > 0,
                        Which[
                            MatchQ[total, 1],
                                pt = PrintTemporary[Internal`LoadingPanel["Downloading 1 image...."]]
                            ,

                            total > 1,
                                pt = PrintTemporary@Dynamic[Internal`LoadingPanel[StringForm["Downloading `1` of `2` images....", current, total]]]
                        ];
                        data = (current += 1; Import[First[#]]) & /@ Lookup[data, "ImageHyperlink"];
                        NotebookDelete[pt]
                    ];

                    data
                ,

                "ImageHyperlink" | "ImageHyperlinks" | "ImageLink" | "ImageLinks",
                    Lookup[data, "ImageHyperlink", {}]
                ,

                "ImageThumbnailsLink" | "ImageThumbnailsLinks",
                    Lookup[data, "ThumbnailLink", {}]
                ,

                "PageHyperlink" | "PageHyperlinks",
                    Lookup[data, "PageHyperlink", {}]
                ,

                "PageTitle" | "PageTitles",
                    Lookup[data, "PageTitle", {}]
                ,

                "Thumbnail" | "Thumbnails",
                    If[total > 0,
                        Which[
                            MatchQ[total, 1],
                                pt = PrintTemporary[Internal`LoadingPanel["Downloading 1 image...."]]
                            ,

                            total > 1,
                                pt = PrintTemporary@Dynamic[Internal`LoadingPanel[StringForm["Downloading `1` of `2` images....", current, total]]]
                        ];
                        data = (current += 1; Import[First[#]]) & /@ Lookup[data, "ThumbnailLink"];
                        NotebookDelete[pt]
                    ];

                    data
                ,

                Default,
                    If[total > 0,
                        Which[
                            MatchQ[total, 1],
                                pt = PrintTemporary[Internal`LoadingPanel["Downloading 1 image...."]]
                            ,

                            total > 1,
                                pt = PrintTemporary@Dynamic[Internal`LoadingPanel[StringForm["Downloading `1` of `2` images....", current, total]]]
                        ];
                        data = Association[current += 1; KeyDrop[Prepend[#, Rule["Thumbnail", Import[First[Lookup[#, "ThumbnailLink"]]]]], {"FileFormat", "ThumbnailLink"}]] & /@ data;
                        NotebookDelete[pt]
                    ];

                    Dataset[data]
                ,

                _,
                    Throw[$Failed]
            ]
    ]
]

GCSAllowedLanguageValues = {
    "lang_ar",
    "lang_bg",
    "lang_ca",
    "lang_cs",
    "lang_da",
    "lang_de",
    "lang_el",
    "lang_en",
    "lang_es",
    "lang_et",
    "lang_fi",
    "lang_fr",
    "lang_hr",
    "lang_hu",
    "lang_id",
    "lang_is",
    "lang_it",
    "lang_iw",
    "lang_ja",
    "lang_ko",
    "lang_lt",
    "lang_lv",
    "lang_nl",
    "lang_no",
    "lang_pl",
    "lang_pt",
    "lang_ro",
    "lang_ru",
    "lang_sk",
    "lang_sl",
    "lang_sr",
    "lang_sv",
    "lang_tr",
    "lang_zh-CN",
    "lang_zh-TW"
 };

GCSAllowedCountryValues = {
    "countryAD",
    "countryAE",
    "countryAF",
    "countryAG",
    "countryAI",
    "countryAL",
    "countryAM",
    "countryAN",
    "countryAO",
    "countryAQ",
    "countryAR",
    "countryAS",
    "countryAT",
    "countryAU",
    "countryAW",
    "countryAZ",
    "countryBA",
    "countryBB",
    "countryBD",
    "countryBE",
    "countryBF",
    "countryBG",
    "countryBH",
    "countryBI",
    "countryBJ",
    "countryBM",
    "countryBN",
    "countryBO",
    "countryBR",
    "countryBS",
    "countryBT",
    "countryBV",
    "countryBW",
    "countryBY",
    "countryBZ",
    "countryCA",
    "countryCC",
    "countryCD",
    "countryCF",
    "countryCG",
    "countryCH",
    "countryCI",
    "countryCK",
    "countryCL",
    "countryCM",
    "countryCN",
    "countryCO",
    "countryCR",
    "countryCS",
    "countryCU",
    "countryCV",
    "countryCX",
    "countryCY",
    "countryCZ",
    "countryDE",
    "countryDJ",
    "countryDK",
    "countryDM",
    "countryDO",
    "countryDZ",
    "countryEC",
    "countryEE",
    "countryEG",
    "countryEH",
    "countryER",
    "countryES",
    "countryET",
    "countryEU",
    "countryFI",
    "countryFJ",
    "countryFK",
    "countryFM",
    "countryFO",
    "countryFR",
    "countryFX",
    "countryGA",
    "countryGD",
    "countryGE",
    "countryGF",
    "countryGH",
    "countryGI",
    "countryGL",
    "countryGM",
    "countryGN",
    "countryGP",
    "countryGQ",
    "countryGR",
    "countryGS",
    "countryGT",
    "countryGU",
    "countryGW",
    "countryGY",
    "countryHK",
    "countryHM",
    "countryHN",
    "countryHR",
    "countryHT",
    "countryHU",
    "countryID",
    "countryIE",
    "countryIL",
    "countryIN",
    "countryIO",
    "countryIQ",
    "countryIR",
    "countryIS",
    "countryIT",
    "countryJM",
    "countryJO",
    "countryJP",
    "countryKE",
    "countryKG",
    "countryKH",
    "countryKI",
    "countryKM",
    "countryKN",
    "countryKP",
    "countryKR",
    "countryKW",
    "countryKY",
    "countryKZ",
    "countryLA",
    "countryLB",
    "countryLC",
    "countryLI",
    "countryLK",
    "countryLR",
    "countryLS",
    "countryLT",
    "countryLU",
    "countryLV",
    "countryLY",
    "countryMA",
    "countryMC",
    "countryMD",
    "countryMG",
    "countryMH",
    "countryMK",
    "countryML",
    "countryMM",
    "countryMN",
    "countryMO",
    "countryMP",
    "countryMQ",
    "countryMR",
    "countryMS",
    "countryMT",
    "countryMU",
    "countryMV",
    "countryMW",
    "countryMX",
    "countryMY",
    "countryMZ",
    "countryNA",
    "countryNC",
    "countryNE",
    "countryNF",
    "countryNG",
    "countryNI",
    "countryNL",
    "countryNO",
    "countryNP",
    "countryNR",
    "countryNU",
    "countryNZ",
    "countryOM",
    "countryPA",
    "countryPE",
    "countryPF",
    "countryPG",
    "countryPH",
    "countryPK",
    "countryPL",
    "countryPM",
    "countryPN",
    "countryPR",
    "countryPS",
    "countryPT",
    "countryPW",
    "countryPY",
    "countryQA",
    "countryRE",
    "countryRO",
    "countryRU",
    "countryRW",
    "countrySA",
    "countrySB",
    "countrySC",
    "countrySD",
    "countrySE",
    "countrySG",
    "countrySH",
    "countrySI",
    "countrySJ",
    "countrySK",
    "countrySL",
    "countrySM",
    "countrySN",
    "countrySO",
    "countrySR",
    "countryST",
    "countrySV",
    "countrySY",
    "countrySZ",
    "countryTC",
    "countryTD",
    "countryTF",
    "countryTG",
    "countryTH",
    "countryTJ",
    "countryTK",
    "countryTM",
    "countryTN",
    "countryTO",
    "countryTP",
    "countryTR",
    "countryTT",
    "countryTV",
    "countryTW",
    "countryTZ",
    "countryUA",
    "countryUG",
    "countryUK",
    "countryUM",
    "countryUS",
    "countryUY",
    "countryUZ",
    "countryVA",
    "countryVC",
    "countryVE",
    "countryVG",
    "countryVI",
    "countryVN",
    "countryVU",
    "countryWF",
    "countryWS",
    "countryYE",
    "countryYT",
    "countryYU",
    "countryZA",
    "countryZM",
    "countryZW"
}

GCSLangMap = {
    Entity["Language", "Arabic"] -> "lang_ar",
    Entity["Language", "Bulgarian"] -> "lang_bg",
    Entity["Language", "CatalanValencianBalear"] -> "lang_ca",
    Entity["Language", "Croatian"] -> "lang_hr",
    Entity["Language", "Czech"] -> "lang_cs",
    Entity["Language", "Danish"] -> "lang_da",
    Entity["Language", "Dutch"] -> "lang_nl",
    Entity["Language", "English"] -> "lang_en",
    Entity["Language", "Estonian"] -> "lang_et",
    Entity["Language", "Finnish"] -> "lang_fi",
    Entity["Language", "French"] -> "lang_fr",
    Entity["Language", "German"] -> "lang_de",
    Entity["Language", "Greek"] -> "lang_el",
    Entity["Language", "Hebrew"] -> "lang_iw",
    Entity["Language", "Hungarian"] -> "lang_hu",
    Entity["Language", "Icelandic"] -> "lang_is",
    Entity["Language", "Indonesian"] -> "lang_id",
    Entity["Language", "Italian"] -> "lang_it",
    Entity["Language", "Japanese"] -> "lang_ja",
    Entity["Language", "Korean"] -> "lang_ko",
    Entity["Language", "Latvian"] -> "lang_lv",
    Entity["Language", "Lithuanian"] -> "lang_lt",
    Entity["Language", "Norwegian"] -> "lang_no",
    Entity["Language", "Polish"] -> "lang_pl",
    Entity["Language", "Portuguese"] -> "lang_pt",
    Entity["Language", "Romanian"] -> "lang_ro",
    Entity["Language", "Russian"] -> "lang_ru",
    Entity["Language", "Serbian"] -> "lang_sr",
    Entity["Language", "Slovak"] -> "lang_sk",
    Entity["Language", "Slovenian"] -> "lang_sl",
    Entity["Language", "Spanish"] -> "lang_es",
    Entity["Language", "Swedish"] -> "lang_sv",
    Entity["Language", "Turkish"] -> "lang_tr"
};

GCSCountryMap = {
    Entity["Country", "Afghanistan"] -> "countryAF",
    Entity["Country", "Albania"] -> "countryAL",
    Entity["Country", "Algeria"] -> "countryDZ",
    Entity["Country", "AmericanSamoa"] -> "countryAS",
    Entity["Country", "Andorra"] -> "countryAD",
    Entity["Country", "Angola"] -> "countryAO",
    Entity["Country", "Anguilla"] -> "countryAI",
    Entity["Country", "AntiguaBarbuda"] -> "countryAG",
    Entity["Country", "Argentina"] -> "countryAR",
    Entity["Country", "Armenia"] -> "countryAM",
    Entity["Country", "Aruba"] -> "countryAW",
    Entity["Country", "Australia"] -> "countryAU",
    Entity["Country", "Austria"] -> "countryAT",
    Entity["Country", "Azerbaijan"] -> "countryAZ",
    Entity["Country", "Bahamas"] -> "countryBS",
    Entity["Country", "Bahrain"] -> "countryBH",
    Entity["Country", "Bangladesh"] -> "countryBD",
    Entity["Country", "Barbados"] -> "countryBB",
    Entity["Country", "Belarus"] -> "countryBY",
    Entity["Country", "Belgium"] -> "countryBE",
    Entity["Country", "Belize"] -> "countryBZ",
    Entity["Country", "Benin"] -> "countryBJ",
    Entity["Country", "Bermuda"] -> "countryBM",
    Entity["Country", "Bhutan"] -> "countryBT",
    Entity["Country", "Bolivia"] -> "countryBO",
    Entity["Country", "BosniaHerzegovina"] -> "countryBA",
    Entity["Country", "Botswana"] -> "countryBW",
    Entity["Country", "Brazil"] -> "countryBR",
    Entity["Country", "Brunei"] -> "countryBN",
    Entity["Country", "Bulgaria"] -> "countryBG",
    Entity["Country", "BurkinaFaso"] -> "countryBF",
    Entity["Country", "Burundi"] -> "countryBI",
    Entity["Country", "Cambodia"] -> "countryKH",
    Entity["Country", "Cameroon"] -> "countryCM",
    Entity["Country", "Canada"] -> "countryCA",
    Entity["Country", "CapeVerde"] -> "countryCV",
    Entity["Country", "CaymanIslands"] -> "countryKY",
    Entity["Country", "CentralAfricanRepublic"] -> "countryCF",
    Entity["Country", "Chad"] -> "countryTD",
    Entity["Country", "Chile"] -> "countryCL",
    Entity["Country", "China"] -> "countryCN",
    Entity["Country", "ChristmasIsland"] -> "countryCX",
    Entity["Country", "CocosKeelingIslands"] -> "countryCC",
    Entity["Country", "Colombia"] -> "countryCO",
    Entity["Country", "Comoros"] -> "countryKM",
    Entity["Country", "DemocraticRepublicCongo"] -> "countryCG",
    Entity["Country", "DemocraticRepublicCongo"] -> "countryCD",
    Entity["Country", "CookIslands"] -> "countryCK",
    Entity["Country", "CostaRica"] -> "countryCR",
    Entity["Country", "IvoryCoast"] -> "countryCI",
    Entity["Country", "Cuba"] -> "countryCU",
    Entity["Country", "Cyprus"] -> "countryCY",
    Entity["Country", "CzechRepublic"] -> "countryCZ",
    Entity["Country", "Denmark"] -> "countryDK",
    Entity["Country", "Djibouti"] -> "countryDJ",
    Entity["Country", "Dominica"] -> "countryDM",
    Entity["Country", "DominicanRepublic"] -> "countryDO",
    Entity["Country", "EastTimor"] -> "countryTP",
    Entity["Country", "Ecuador"] -> "countryEC",
    Entity["Country", "Egypt"] -> "countryEG",
    Entity["Country", "ElSalvador"] -> "countrySV",
    Entity["Country", "EquatorialGuinea"] -> "countryGQ",
    Entity["Country", "Eritrea"] -> "countryER",
    Entity["Country", "Estonia"] -> "countryEE",
    Entity["Country", "Ethiopia"] -> "countryET",
    Entity["Country", "FalklandIslands"] -> "countryFK",
    Entity["Country", "FaroeIslands"] -> "countryFO",
    Entity["Country", "Fiji"] -> "countryFJ",
    Entity["Country", "Finland"] -> "countryFI",
    Entity["Country", "France"] -> "countryFR",
    Entity["Country", "FrenchGuiana"] -> "countryGF",
    Entity["Country", "FrenchPolynesia"] -> "countryPF",
    Entity["Country", "Gabon"] -> "countryGA",
    Entity["Country", "Gambia"] -> "countryGM",
    Entity["Country", "Georgia"] -> "countryGE",
    Entity["Country", "Germany"] -> "countryDE",
    Entity["Country", "Ghana"] -> "countryGH",
    Entity["Country", "Gibraltar"] -> "countryGI",
    Entity["Country", "Greece"] -> "countryGR",
    Entity["Country", "Greenland"] -> "countryGL",
    Entity["Country", "Grenada"] -> "countryGD",
    Entity["Country", "Guadeloupe"] -> "countryGP",
    Entity["Country", "Guam"] -> "countryGU",
    Entity["Country", "Guatemala"] -> "countryGT",
    Entity["Country", "Guinea"] -> "countryGN",
    Entity["Country", "GuineaBissau"] -> "countryGW",
    Entity["Country", "Guyana"] -> "countryGY",
    Entity["Country", "Haiti"] -> "countryHT",
    Entity["Country", "VaticanCity"] -> "countryVA",
    Entity["Country", "Honduras"] -> "countryHN",
    Entity["Country", "HongKong"] -> "countryHK",
    Entity["Country", "Hungary"] -> "countryHU",
    Entity["Country", "Iceland"] -> "countryIS",
    Entity["Country", "India"] -> "countryIN",
    Entity["Country", "Indonesia"] -> "countryID",
    Entity["Country", "Iran"] -> "countryIR",
    Entity["Country", "Iraq"] -> "countryIQ",
    Entity["Country", "Ireland"] -> "countryIE",
    Entity["Country", "Israel"] -> "countryIL",
    Entity["Country", "Italy"] -> "countryIT",
    Entity["Country", "Jamaica"] -> "countryJM",
    Entity["Country", "Japan"] -> "countryJP",
    Entity["Country", "Jordan"] -> "countryJO",
    Entity["Country", "Kazakhstan"] -> "countryKZ",
    Entity["Country", "Kenya"] -> "countryKE",
    Entity["Country", "Kiribati"] -> "countryKI",
    Entity["Country", "SouthKorea"] -> "countryKR",
    Entity["Country", "Kuwait"] -> "countryKW",
    Entity["Country", "Kyrgyzstan"] -> "countryKG",
    Entity["Country", "Laos"] -> "countryLA",
    Entity["Country", "Latvia"] -> "countryLV",
    Entity["Country", "Lebanon"] -> "countryLB",
    Entity["Country", "Lesotho"] -> "countryLS",
    Entity["Country", "Liberia"] -> "countryLR",
    Entity["Country", "Libya"] -> "countryLY",
    Entity["Country", "Liechtenstein"] -> "countryLI",
    Entity["Country", "Lithuania"] -> "countryLT",
    Entity["Country", "Luxembourg"] -> "countryLU",
    Entity["Country", "Macau"] -> "countryMO",
    Entity["Country", "Macedonia"] -> "countryMK",
    Entity["Country", "Madagascar"] -> "countryMG",
    Entity["Country", "Malawi"] -> "countryMW",
    Entity["Country", "Malaysia"] -> "countryMY",
    Entity["Country", "Maldives"] -> "countryMV",
    Entity["Country", "Mali"] -> "countryML",
    Entity["Country", "Malta"] -> "countryMT",
    Entity["Country", "MarshallIslands"] -> "countryMH",
    Entity["Country", "Martinique"] -> "countryMQ",
    Entity["Country", "Mauritania"] -> "countryMR",
    Entity["Country", "Mauritius"] -> "countryMU",
    Entity["Country", "Mayotte"] -> "countryYT",
    Entity["Country", "Mexico"] -> "countryMX",
    Entity["Country", "Micronesia"] -> "countryFM",
    Entity["Country", "Moldova"] -> "countryMD",
    Entity["Country", "Monaco"] -> "countryMC",
    Entity["Country", "Mongolia"] -> "countryMN",
    Entity["Country", "Montserrat"] -> "countryMS",
    Entity["Country", "Morocco"] -> "countryMA",
    Entity["Country", "Mozambique"] -> "countryMZ",
    Entity["Country", "Myanmar"] -> "countryMM",
    Entity["Country", "Namibia"] -> "countryNA",
    Entity["Country", "Nauru"] -> "countryNR",
    Entity["Country", "Nepal"] -> "countryNP",
    Entity["Country", "Netherlands"] -> "countryNL",
    Entity["Country", "NetherlandsAntilles"] -> "countryAN",
    Entity["Country", "NewCaledonia"] -> "countryNC",
    Entity["Country", "NewZealand"] -> "countryNZ",
    Entity["Country", "Nicaragua"] -> "countryNI",
    Entity["Country", "Niger"] -> "countryNE",
    Entity["Country", "Nigeria"] -> "countryNG",
    Entity["Country", "Niue"] -> "countryNU",
    Entity["Country", "NorfolkIsland"] -> "countryNF",
    Entity["Country", "NorthernMarianaIslands"] -> "countryMP",
    Entity["Country", "Norway"] -> "countryNO",
    Entity["Country", "Oman"] -> "countryOM",
    Entity["Country", "Pakistan"] -> "countryPK",
    Entity["Country", "Palau"] -> "countryPW",
    Entity["Country", "WestBank"] -> "countryPS",
    Entity["Country", "Panama"] -> "countryPA",
    Entity["Country", "PapuaNewGuinea"] -> "countryPG",
    Entity["Country", "Paraguay"] -> "countryPY",
    Entity["Country", "Peru"] -> "countryPE",
    Entity["Country", "Philippines"] -> "countryPH",
    Entity["Country", "PitcairnIslands"] -> "countryPN",
    Entity["Country", "Poland"] -> "countryPL",
    Entity["Country", "Portugal"] -> "countryPT",
    Entity["Country", "PuertoRico"] -> "countryPR",
    Entity["Country", "Qatar"] -> "countryQA",
    Entity["Country", "Reunion"] -> "countryRE",
    Entity["Country", "Romania"] -> "countryRO",
    Entity["Country", "Russia"] -> "countryRU",
    Entity["Country", "Rwanda"] -> "countryRW",
    Entity["Country", "SaintHelena"] -> "countrySH",
    Entity["Country", "SaintKittsNevis"] -> "countryKN",
    Entity["Country", "SaintLucia"] -> "countryLC",
    Entity["Country", "SaintPierreMiquelon"] -> "countryPM",
    Entity["Country", "SaintVincentGrenadines"] -> "countryVC",
    Entity["Country", "Samoa"] -> "countryWS",
    Entity["Country", "SanMarino"] -> "countrySM",
    Entity["Country", "SaoTomePrincipe"] -> "countryST",
    Entity["Country", "SaudiArabia"] -> "countrySA",
    Entity["Country", "Senegal"] -> "countrySN",
    Entity["Country", "Seychelles"] -> "countrySC",
    Entity["Country", "SierraLeone"] -> "countrySL",
    Entity["Country", "Singapore"] -> "countrySG",
    Entity["Country", "Slovakia"] -> "countrySK",
    Entity["Country", "Slovenia"] -> "countrySI",
    Entity["Country", "SolomonIslands"] -> "countrySB",
    Entity["Country", "Somalia"] -> "countrySO",
    Entity["Country", "SouthAfrica"] -> "countryZA",
    Entity["Country", "Spain"] -> "countryES",
    Entity["Country", "SriLanka"] -> "countryLK",
    Entity["Country", "Sudan"] -> "countrySD",
    Entity["Country", "Suriname"] -> "countrySR",
    Entity["Country", "Svalbard"] -> "countrySJ",
    Entity["Country", "Swaziland"] -> "countrySZ",
    Entity["Country", "Sweden"] -> "countrySE",
    Entity["Country", "Switzerland"] -> "countryCH",
    Entity["Country", "Syria"] -> "countrySY",
    Entity["Country", "Taiwan"] -> "countryTW",
    Entity["Country", "Tajikistan"] -> "countryTJ",
    Entity["Country", "Tanzania"] -> "countryTZ",
    Entity["Country", "Thailand"] -> "countryTH",
    Entity["Country", "Togo"] -> "countryTG",
    Entity["Country", "Tokelau"] -> "countryTK",
    Entity["Country", "Tonga"] -> "countryTO",
    Entity["Country", "TrinidadTobago"] -> "countryTT",
    Entity["Country", "Tunisia"] -> "countryTN",
    Entity["Country", "Turkey"] -> "countryTR",
    Entity["Country", "Turkmenistan"] -> "countryTM",
    Entity["Country", "TurksCaicosIslands"] -> "countryTC",
    Entity["Country", "Tuvalu"] -> "countryTV",
    Entity["Country", "Uganda"] -> "countryUG",
    Entity["Country", "Ukraine"] -> "countryUA",
    Entity["Country", "UnitedArabEmirates"] -> "countryAE",
    Entity["Country", "UnitedKingdom"] -> "countryUK",
    Entity["Country", "UnitedStates"] -> "countryUS",
    Entity["Country", "Uruguay"] -> "countryUY",
    Entity["Country", "Uzbekistan"] -> "countryUZ",
    Entity["Country", "Vanuatu"] -> "countryVU",
    Entity["Country", "Venezuela"] -> "countryVE",
    Entity["Country", "Vietnam"] -> "countryVN",
    Entity["Country", "BritishVirginIslands"] -> "countryVG",
    Entity["Country", "UnitedStatesVirginIslands"] -> "countryVI",
    Entity["Country", "WallisFutuna"] -> "countryWF",
    Entity["Country", "WesternSahara"] -> "countryEH",
    Entity["Country", "Yemen"] -> "countryYE",
    Entity["Country", "Zambia"] -> "countryZM",
    Entity["Country", "Zimbabwe"] -> "countryZW",
    Entity["Country", "NorthKorea"] -> "countryKP",
    Entity["Country", "Serbia"] -> "countryCS",
    Entity["Country", "Montenegro"] -> "countryCS",
    Entity["Country", "Croatia"] -> "countryHR"
};

End[];

EndPackage[];
