
Unprotect[System`TextTranslation];
Clear[System`TextTranslation];

BeginPackage["TextTranslation`"]
(* Exported symbols added here with SymbolName::usage *)

Begin["`Private`"] (* Begin Private Context *)

System`TextTranslation::nomethod = "The method `1` is not available for TextTranslation.";
System`TextTranslation::offline = "The Wolfram Language is currently configured not to use the Internet. To allow Internet use, check the \"Allow the Wolfram Language to use the Internet\" box in the Help \[FilledRightTriangle] Internet Connectivity dialog.";
System`TextTranslation::notauth = "TextTranslation requests are only valid when authenticated. Please try authenticating again.";
System`TextTranslation::invmet = "This translation service does not support this `1` language."; (*source/target*)
System`TextTranslation::invlan = "Invalid `1` language.";
System`TextTranslation::reqlan = "`1` language required.";
System`TextTranslation::strlimit = "The text exceeds the 1000 character limit.";

TTServiceName[name_] :=
    Switch[name,
        "Google","GoogleTranslate",
        "Microsoft","MicrosoftTranslator",
        Automatic, Automatic,
            _, Message[System`TextTranslation::nomethod, name];
               Throw["exception"]
    ]

Options[TextTranslation] = {Method -> Automatic, "LanguageRules" -> False};

System`TextTranslation[___] :=
    (Message[System`TextTranslation::offline];
     $Failed)/;(!PacletManager`$AllowInternet)

System`TextTranslation[args___] :=
    With[ {connected = CloudConnect[]},
        If[ $CloudConnected,
            System`TextTranslation[args],
            Message[System`TextTranslation::notauth];
            connected
        ]
    ]/; !$CloudConnected

System`TextTranslation[args___] :=
    With[ {res = Catch[TextTranslation1[args]]},
        res /; !MatchQ[res, "exception"]
    ]

TextTranslation1[text_String|text_List,(source_String|source_Entity)->target_, opt: OptionsPattern[TextTranslation]] :=
    Block[ {engine,response,prevParams,requestParameters},
        If[ StringQ[text],
            If[ StringLength[text]>1000,
                Message[System`TextTranslation::strlimit];
                Throw["exception"]
            ],
            If[ And @@ ((# > 1000 &) /@ (StringLength /@ text)),
                Message[System`TextTranslation::strlimit];
                Throw["exception"]
            ]
        ];
        engine = TTServiceName[OptionValue[Method]];
        prevParams = {"Text"-> text, "To"-> target, "LanguageRules" -> OptionValue["LanguageRules"], If[ !MatchQ[source,"Automatic"],
                                                                                                         "From"->source,
                                                                                                         Sequence@@{}
                                                                                                     ]};
        Get["GoogleTranslate`"];
        Get["MicrosoftTranslator`"];
        Switch[engine,
            Automatic,
                engine = "MicrosoftTranslator";
                requestParameters = Catch[
                    Quiet[
                        Check[
                            requestParameters = Catch[MicrosoftTranslatorFunctions`MTFormatRequestParameters[FilterRules[Join[{opt}, prevParams], Except[Method]]]]
                        ,
                            If[ FailureQ[requestParameters],
                                Throw[$Failed],
                                Throw[{$Failed -> requestParameters}]
                            ]
                        ]
                    ]
                ];
                requestParameters = Which[
                    FailureQ[requestParameters],
                        Throw["exception"]
                    ,

                    KeyExistsQ[requestParameters, $Failed],
                        engine = "GoogleTranslate";
                        Catch[
                            Quiet[
                                Check[
                                    requestParameters = Catch[GoogleTranslateFunctions`GTFormatRequestParameters[FilterRules[Join[{opt}, prevParams], Except[Method]]]]
                                ,
                                    If[ FailureQ[requestParameters],
                                        Throw[$Failed],
                                        Throw[{$Failed -> requestParameters}]
                                    ]
                                ]
                            ]
                        ]
                    ,

                    True,
                        requestParameters
                ];
                Which[
                    FailureQ[requestParameters],
                        Throw["exception"]
                    ,

                    KeyExistsQ[requestParameters, $Failed],
                        TTProcessError[Lookup[requestParameters, $Failed], engine, False]
                ];
                Switch[engine,
                    "MicrosoftTranslator",
                        response = TextTranslation`Private`MTProcessRequest[requestParameters];
                        If[ !MatchQ[response, $Failed],
                            MicrosoftTranslatorFunctions`MTCookedImport[response, requestParameters[[2]]],
                            Throw["exception"]
                        ]
                    ,

                    "GoogleTranslate",
                        response = TextTranslation`Private`GTProcessRequest[requestParameters];
                        If[ !MatchQ[response, $Failed],
                            GoogleTranslateFunctions`GTCookedImport[response, requestParameters],
                            Throw["exception"]
                        ]
                    ,

                    _,
                        Throw["exception"]
                ]
            ,

            "GoogleTranslate",
            (

                requestParameters = Catch[
                    Quiet[
                        Check[
                            requestParameters = Catch[GoogleTranslateFunctions`GTFormatRequestParameters[FilterRules[Join[{opt}, prevParams], Except[Method]]]]
                        ,
                            If[ FailureQ[requestParameters],
                                Throw[$Failed],
                                Throw[{$Failed -> requestParameters}]
                            ]
                        ]
                    ]
                ];
                Which[
                    FailureQ[requestParameters],
                        Throw["exception"]
                    ,

                    KeyExistsQ[requestParameters, $Failed],
                        TTProcessError[Lookup[requestParameters, $Failed], engine]
                ];
                response = TextTranslation`Private`GTProcessRequest[requestParameters];
                If[ !MatchQ[response, $Failed],
                    GoogleTranslateFunctions`GTCookedImport[response, requestParameters],
                    Throw["exception"]
                ]
            ),
            "MicrosoftTranslator",
            (
                requestParameters = Catch[
                    Quiet[
                        Check[
                            requestParameters = Catch[MicrosoftTranslatorFunctions`MTFormatRequestParameters[FilterRules[Join[{opt},prevParams],Except[Method]]]]
                        ,
                            If[ FailureQ[requestParameters],
                                Throw[$Failed],
                                Throw[{$Failed -> requestParameters}]
                            ]
                        ]
                    ]
                ];
                Which[
                    FailureQ[requestParameters],
                        Throw["exception"]
                    ,

                    KeyExistsQ[requestParameters, $Failed],
                        TTProcessError[Lookup[requestParameters, $Failed], engine]
                ];
                response = TextTranslation`Private`MTProcessRequest[requestParameters];
                If[ !MatchQ[response, $Failed],
                    MicrosoftTranslatorFunctions`MTCookedImport[response, requestParameters[[2]]],
                    Throw["exception"]
                ]
            )
        ]
    ]

TextTranslation1[text_String|text_List, opt : OptionsPattern[]] :=
    TextTranslation1[text,$Language,opt]

TextTranslation1[text_String|text_List, target_String|target_Entity, opt : OptionsPattern[]] :=
    TextTranslation1[text, "Automatic" -> target, opt]

TTProcessError[error_, engine_, shouldValidate_ : True] :=
    Block[ {},
        Switch[Lookup[error, "tag"],
            "nval",
                If[ shouldValidate && TTValidateLanguage[Lookup[error, "value"], engine],
                    Message[System`TextTranslation::invmet, TTProcessFieldName[Lookup[error, "param", ""]]],
                    Message[System`TextTranslation::invlan, TTProcessFieldName[Lookup[error, "param", ""]]]
                ]
            ,

            "nparam",
                Message[System`TextTranslation::reqlan, Capitalize[TTProcessFieldName[Lookup[error, "param", ""]]]]
            ,

            True,
                "missing"
        ];
        Throw["exception"]
    ]

TTProcessFieldName[name_String?StringQ] :=
    Switch[name,
    "From",
    "source"
    ,

    "To",
    "target"
    ,

    True,
    "missing"
    ]

TTValidateLanguage[value_, engine_String?StringQ] :=
    Block[ {langRules},
        Switch[engine, (*this engine has already been checked, so we validate the other engine*)
            "GoogleTranslate", (*we look into MT*)
                langRules = MicrosoftTranslatorFunctions`MTEntityToLanguageCodeAlignment;
                MemberQ[Join[Keys[langRules], Values[langRules]], value]
            ,

            "MicrosoftTranslator", (*we look into GT*)
                langRules = GoogleTranslateFunctions`GTEntityToLanguageCodeAlignment;
                MemberQ[Join[Keys[langRules], Values[langRules]], value]
            ,
            _,
                False
        ]
    ]

GTProcessRequest[p_] :=
    Module[ {text = p["Text"], from = p["From"], to = p["To"], msl = p["msl"], result = {}, textgroupby, statictext, preresult,response, cloudProblem = False},
        Catch[
            If[ MatchQ[Head[text], String],
                response = IntegratedServices`RemoteServiceExecute[Symbol["TextTranslation"], "GoogleTranslate", "RawTranslate", <|"q"->text,"source"->from,"target"->to, "format" -> "text"|>];
                If[ MatchQ[response, $Failed],
                    Throw[$Failed, "exception"]
                ];
                If[ MatchQ[response, $Canceled],
                    cloudProblem = True;
                    Throw[$Failed, "cloudConnectProblem"]
                ];
                response = "translatedText" /. response["data"]["translations"],
                If[ MatchQ[Head[text], List],
                    If[ msl,
                        textgroupby = GroupBy[MapThread[List, {from, text, Range[Length@text]}], First];
                        If[ MemberQ[from, to],
                            statictext = Cases[Flatten[List@@textgroupby, 1], {to, _, _Integer}];
                            textgroupby = Association[DeleteCases[Normal[textgroupby], to -> _]];,
                            statictext = {}
                        ];
                        preresult = Function[v,
                            Module[ {gtexts = textgroupby[v][[All, 2]], gindexes = textgroupby[v][[All, 3]],temp},
                                temp = IntegratedServices`RemoteServiceExecute[Symbol["TextTranslation"], "GoogleTranslate", "RawTranslate",<|"q"->gtexts,"source"->v,"target"->to, "format" -> "text"|>];
                                If[ MatchQ[response, $Failed],
                                    Throw[$Failed, "exception"]
                                ];
                                If[ MatchQ[temp, $Canceled],
                                    cloudProblem = True;
                                    Throw[$Failed, "cloudConnectProblem"]
                                ];
                                If[ KeyExistsQ[temp,"error"],
                                    (
                                            Message[ServiceExecute::serrormsg,("message" /. ("error" /. temp))];
                                            Throw["exception"]
                                    )
                                ];
                                MapThread[List,{gindexes, "translatedText"/.("translations"/.("data"/.temp))}]
                            ]]/@DeleteDuplicates[DeleteCases[from, to]];
                        If[ statictext==={}, (*If there's no text with source language = target language*)
                            response = SortBy[Flatten[preresult, 1], First][[All, 2]];,
                            response = SortBy[Union[Flatten[preresult, 1], (Reverse /@ statictext)[[All, 1 ;; 2]]], First][[All, 2]];
                        ];,
                        If[ MemberQ[from, to],
                            response = text;,
                            response = IntegratedServices`RemoteServiceExecute[Symbol["GoogleTranslate"], "GoogleTranslate","RawTranslate", <|"q" -> text, "source" -> If[ MatchQ[Head[from], List],
                                                                                                                                                                           from[[1]],
                                                                                                                                                                           from
                                                                                                                                                                       ], "target" -> to, "format" -> "text"|>];
                            If[ MatchQ[response, $Failed],
                                Throw[$Failed, "exception"]
                            ];
                            If[ MatchQ[response, $Canceled],
                                cloudProblem = True;
                                Throw[$Failed, "cloudConnectProblem"]
                            ];
                            If[ KeyExistsQ[result,"error"],
                                (
                                        Message[ServiceExecute::serrormsg,("message" /. ("error" /. response))];
                                        Throw["exception"]
                                )
                            ];
                            response = "translatedText" /. ("translations" /. ("data" /. response));
                        ];
                    ];
                ]
            ];
    ,
        "cloudConnectProblem" | "errorAPI" | "exception"
    ];
        If[ cloudProblem,
            $Failed,
            response
        ]
    ]

MTProcessRequest[p_] :=
    Module[ {textP = p[[2]]["Text"], fromP = p[[2]]["From"], toP = p[[2]]["To"], tuples = Lookup[p[[2]],"tuples",{}], rawdata, results, groups, f, current, cloudProblem = False},
        Catch[
            If[ MatchQ[textP, _String],
                (
                    rawdata = IntegratedServices`RemoteServiceExecute[Symbol["TextTranslation"],"MicrosoftTranslator", "RawTranslate",Association[{"text"->URLEncode[textP], "from"->fromP, "to"->toP}],"VerifyPeer" -> False];
                    If[ MatchQ[rawdata, $Failed],
                        results = rawdata;
                        Throw[$Failed, "exception"]
                    ];
                    If[ MatchQ[rawdata, $Canceled],
                        cloudProblem = True;
                        Throw[$Failed, "cloudConnectProblem"]
                    ];
                    results = MicrosoftTranslatorFunctions`MTEntityFromLanguageCode[fromP]->MicrosoftTranslator`Private`parseTranslateOutput[rawdata];
                ),
                (
                    groups = GroupBy[tuples, First] // Normal;
                    (
                        results = (
                            f = #[[1]];
                            current = #[[2]];
                            rawdata = IntegratedServices`RemoteServiceExecute[Symbol["TextTranslation"],"MicrosoftTranslator","RawTranslateArray",Association[{"Data"->MicrosoftTranslator`Private`translateArrayRequestXML[current[[All,2]],toP,f]}],"VerifyPeer" -> False];
                            If[ MatchQ[rawdata, $Failed],
                                results = rawdata;
                                Throw[$Failed, "exception"]
                            ];
                            If[ MatchQ[rawdata, $Canceled],
                                cloudProblem = True;
                                Throw[$Failed, "cloudConnectProblem"]
                            ];
                            rawdata = MicrosoftTranslator`Private`parseTranslateArrayOutput[rawdata];
                            current[[#]] -> rawdata[[#]] & /@ Range[Length[current]]
                        )&/@ groups;
                        results = Flatten[results];
                    )
                )
            ];
    ,
        "cloudConnectProblem" | "exception"
    ];
        If[ cloudProblem,
            $Failed,
            results
        ]
    ]

TextTranslation1[___] :=
    Throw["exception"]

SetAttributes[System`TextTranslation,{Protected,ReadProtected}];
End[];
EndPackage[];
