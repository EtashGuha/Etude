BeginPackage["MicrosoftTranslatorFunctions`"];

MTCookedImport::usage = "";
MTEntityToLanguageCodeAlignment::usage = "";
MTFormatRequestParameters::usage = "";
MTLanguageCodeToEntityAlignment::usage = "";
MTMapParameterToLanguageCode::usage = "";
MTProcessRequest::usage = "";

Begin["`Private`"];

MTAllowedLanguageCodes = {};
mti = MicrosoftTranslator`Private`microsofttranslatorimport;
MTCookedImport[result_, params_] := Module[
	{text=params["Text"], languageRules=params["LanguageRules"], tmpResult},

	If[MatchQ[text, _String],
		If[!languageRules,
			result[[2]],
			result
		],
		If[!languageRules,
			result[[All,2]],
			result
		]
    ]
]

MTReturnMessage[newName_, errorcode_, origService_, params___] := With[
	{msg = Once[MessageName[origService, errorcode]]},

    If[ MatchQ[newName, Null],
        Message[MessageName[origService, errorcode], params],
        MessageName[newName, errorcode] = msg;
        Message[MessageName[newName, errorcode], params];
        Unset[MessageName[newName, errorcode]]
    ]
]

Options[MTFormatRequestParameters] = {"IntegratedServiceQ" -> False}
MTFormatRequestParameters[args_, OptionsPattern[]] := Module[
	{params=<||>,params2=<||>,textP,toP,fromP,rawdata,dataxml,translations,results,languageRules=False,groups,tuples={},f,current,status,invalidParameters,msg, integratedServiceQ, tag = Null, tmpValue},
	invalidParameters = Select[Keys[args],!MemberQ[{"LanguageRules","Text","From","To"},#]&];
	integratedServiceQ = TrueQ[OptionValue["IntegratedServiceQ"]];
    If[integratedServiceQ,
        tag = WebSearch
    ];
	If[Length[invalidParameters]>0,
		(
			(*Message[ServiceObject::noget,#,"MicrosoftTranslator"]&/@invalidParameters;*)
			MTReturnMessage[tag, "noget", ServiceObject, #, "MicrosoftTranslator"] & /@ invalidParameters;
			Throw[$Failed]
		)];

	If[KeyExistsQ[args,"LanguageRules"],languageRules = "LanguageRules" /. args];

	If[KeyExistsQ[args,"Text"],
		textP = "Text" /. args,
		(
			(*Message[ServiceExecute::nparam,"Text"];*)
			MTReturnMessage[tag, "nparam", ServiceExecute, "Text"];
			Throw[$Failed]
		)
	];
	If[KeyExistsQ[args,"From"],
		(
			tmpValue = fromP = "From" /. args;
			fromP = MTMapParameterToLanguageCode[fromP];
			If[fromP == "Failed",
			(
				(*Message[ServiceExecute::nval,"From","MicrosoftTranslator"];*)
				MTReturnMessage[tag, "nval", ServiceExecute, "From","MicrosoftTranslator"];
				Throw[{"tag" -> "nval", "param" -> "From", "value" -> tmpValue}]
			)];
			If[MatchQ[Head[textP], List],tuples = {fromP,textP[[#]],#} &/@ Range[Length[textP]]]
		),
			If[MatchQ[Head[textP], String],(*Autodetects the language using WL*)
				fromP = MTMapParameterToLanguageCode[tmpValue = Classify["Language",textP]];
				If[fromP == "Failed",
					MTReturnMessage[tag, "nval", ServiceExecute, "From","MicrosoftTranslator"];
					Throw[{"tag" -> "nval", "param" -> "From", "value" -> tmpValue}]
				]
			,
				If[MatchQ[Head[textP], List],
					tuples = ({MTMapParameterToLanguageCode[Classify["Language",textP[[#]]]],textP[[#]],#})&/@Range[Length[textP]]
				,
					MTReturnMessage[tag, "nval", ServiceExecute, "Text","MicrosoftTranslator"];
					Throw[{"tag" -> "nval", "param" -> "From", "value" -> ""}]
				]
			];
	];
	If[KeyExistsQ[args,"To"],
		(
			tmpValue = toP = "To" /. args;
			toP = MTMapParameterToLanguageCode[toP];
			If[toP == "Failed",
			(
				(*Message[ServiceExecute::nval,"To","MicrosoftTranslator"];*)
				MTReturnMessage[tag, "nval", ServiceExecute, "To","MicrosoftTranslator"];
				Throw[{"tag" -> "nval", "param" -> "To", "value" -> tmpValue}]
			)]
		),
		(
			(*Message[ServiceExecute::nparam,"To"];*)
			MTReturnMessage[tag, "nparam", ServiceExecute, "To"];
			Throw[{"tag" -> "nparam", "param" -> "To"}]
		)
	];
	params2["Text"] = params["textP"] = textP;
	params2["From"] = params["fromP"] = fromP;
	params2["To"] = params["toP"] = toP;
	params2["LanguageRules"] = params["languageRules"] = languageRules;
	params2["tuples"] = params["tuples"] = tuples;

	{params, params2}
]

MTMapParameterToLanguageCode[param_] := Module[
	{code,entity},
	If[Head[param]===Entity, (* language represented as a WL Entity *)
	(
		code = MTEntityToLanguageCode[param];
		If[Head[code]===Missing, code = "Failed"]
	),
	(
		If[Head[param]===String,
		(
			If[MTAllowedLanguageCodes==={},MTAllowedLanguageCodes=allowedLanguages];
			If[!MemberQ[MTAllowedLanguageCodes,param],
			(
				entity = Interpreter["Language"][param];
				code = MTEntityToLanguageCode[entity];
				If[Head[code]===Missing, code = "Failed"]
			),
				code = param
			]
		),
			code = "Failed" (* Invalid language code *)
		]

	)];
	code
]
MTProcessRequest[id_, p_] := Module[
    {textP=p["Text"], fromP=p["From"], toP=p["To"], languageRules=p["LanguageRules"], tuples=Lookup[p,"tuples",{}],rawdata, dataxml, status, results, groups, f, current, msg,translation,sourceLanguages,translations},
		If[MatchQ[textP, _String],
			(
				jsonTextData=ExportString[List[Association["Text"->p["Text"]]],"JSON","Compact"->True];
				rawdata = KeyClient`rawkeydata[id,"RawTranslate",{"api-version" -> "3.0","Data"->jsonTextData, "from"->fromP, "to"->toP}];
				rawdata = mti[rawdata];
				translation = rawdata["Body"][[1]]["translations"][[1,1]];
				results = MTEntityFromLanguageCode[fromP]->translation
			),
				jsonTextData = ExportString[Association["Text" -> #] & /@p["Text"],"JSON","Compact"->True];
				rawdata = KeyClient`rawkeydata[id,"RawTranslate",{"api-version" -> "3.0","Data"->jsonTextData, "to"->toP}];
				rawdata=mti[rawdata];
				sourceLanguages = tuples[[All,1]];
				translations = rawdata["Body"][[All, 2]][[All, 1]][[All, 1]];
				results=(MTEntityFromLanguageCode[#[[1]]]->#[[2]])&/@Transpose[{sourceLanguages,translations}];
		];

		results

]
(*
MTProcessRequest[id_, p_] := Module[
    {textP=List[Association["Text"->p["Text"]]], fromP=p["From"], toP=p["To"], languageRules=p["LanguageRules"], tuples=Lookup[p,"tuples",{}], rawdata, dataxml, status, results, groups, f, current, msg},

    If[MatchQ[textP, _String],
		(
			rawdata = KeyClient`rawkeydata[id,"RawTranslate",{"api-version" -> "3.0","Data"->textP, "from"->fromP, "to"->toP}];
			rawdata = ToString[{rawdata[[1]], StringReplace[ToString[FromCharacterCode[rawdata[[2]], "UTF8"]], "[" ~~ data___ ~~ "]" :> data]}];
			dataxml = ImportString[StringReplace[rawdata, Shortest[___] ~~ "," ~~ xml___ ~~ "}" :> xml], "XML"];

			status = StringReplace[rawdata, "{" ~~ Shortest[s___] ~~ "," ~~ ___ :> s];
			If[status != "200",
			(
				msg = Cases[dataxml, XMLElement["p", _, content_List] :> content, Infinity];
				Message[ServiceExecute::serrormsg,msg];
				Throw[$Failed]
			)];

			results=MTEntityFromLanguageCode[fromP]->MicrosoftTranslator`Private`parseTranslateOutput[dataxml];
		),
		(
			groups = GroupBy[tuples, First] // Normal;
			(
				results = (
					f = #[[1]];
					current = #[[2]];
					rawdata = KeyClient`rawkeydata[id,"RawTranslateArray",{"Data"->MicrosoftTranslator`Private`translateArrayRequestXML[current[[All,2]],toP,f]}];
					rawdata = ToString[{rawdata[[1]],StringReplace[ToString[FromCharacterCode[rawdata[[2]], "UTF8"]], "[" ~~ data___ ~~ "]" :> data]}];
					dataxml = ImportString[StringReplace[rawdata, Shortest[___] ~~ "," ~~ xml___ ~~ "}" :> xml], "XML"];
					status = StringReplace[rawdata, "{" ~~ Shortest[s___] ~~ "," ~~ ___ :> s];
					If[status != "200",
					(
						msg = Cases[dataxml, XMLElement["p", _, content_List] :> content, Infinity];
						Message[ServiceExecute::serrormsg,msg];
						Throw[$Failed]
					)];

					rawdata = MicrosoftTranslator`Private`parseTranslateArrayOutput[dataxml];
					current[[#]] -> rawdata[[#]] & /@ Range[Length[current]]
				)&/@ groups;
				results = Flatten[results];
			)
		)];

    results
]
*)

MTEntityToLanguageCode[entity_] := If[KeyExistsQ[MTEntityToLanguageCodeAlignment,entity],
									entity /. MTEntityToLanguageCodeAlignment,
									Missing["NotAvailble"]]

MTEntityToLanguageCodeAlignment = {
	Entity["Language", "Arabic"] -> "ar",
	Entity["Language", "Bulgarian"] -> "bg",
	Entity["Language", "CatalanValencianBalear"] -> "ca",
	Entity["Language", "Czech"] -> "cs",
	Entity["Language", "Danish"] -> "da",
	Entity["Language", "Dutch"] -> "nl",
	Entity["Language", "English"] -> "en",
	Entity["Language", "Estonian"] -> "et",
	Entity["Language", "Finnish"] -> "fi",
	Entity["Language", "French"] -> "fr",
	Entity["Language", "German"] -> "de",
	Entity["Language", "Greek"] -> "el",
	Entity["Language", "HaitianCreoleFrench"] -> "ht",
	Entity["Language", "Hindi"] -> "hi",
	Entity["Language", "Hungarian"] -> "hu",
	Entity["Language", "Indonesian"] -> "id",
	Entity["Language", "Italian"] -> "it",
	Entity["Language", "Japanese"] -> "ja",
	Entity["Language", "Korean"] -> "ko",
	Entity["Language", "Latvian"] -> "lv",
	Entity["Language", "Lithuanian"] -> "lt",
	Entity["Language", "Malay"] -> "ms",
	Entity["Language", "Maltese"] -> "mt",
	Entity["Language", "Norwegian"] -> "no",
	Entity["Language", "FarsiEastern"] -> "fa",
	Entity["Language", "Polish"] -> "pl",
	Entity["Language", "Portuguese"] -> "pt",
	Entity["Language", "Romanian"] -> "ro",
	Entity["Language", "Russian"] -> "ru",
	Entity["Language", "Slovak"] -> "sk",
	Entity["Language", "Slovenian"] -> "sl",
	Entity["Language", "Spanish"] -> "es",
	Entity["Language", "Swedish"] -> "sv",
	Entity["Language", "Thai"] -> "th",
	Entity["Language", "Turkish"] -> "tr",
	Entity["Language", "Ukrainian"] -> "uk",
	Entity["Language", "Urdu"] -> "ur",
	Entity["Language", "Vietnamese"] -> "vi",
	Entity["Language", "Welsh"] -> "cy",
	Entity["Language", "FarsiWestern"] -> "fa",
	Entity["Language", "Chinese"] -> "zh-Hant",
	Entity["Language", "ChineseGan"] -> "zh-Hant",
	Entity["Language", "ChineseHakka"] -> "zh-Hant",
	Entity["Language", "ChineseHuizhou"] -> "zh-Hant",
	Entity["Language", "ChineseJinyu"] -> "zh-Hant",
	Entity["Language", "ChineseMandarin"] -> "zh-Hant",
	Entity["Language", "ChineseMinBei"] -> "zh-Hant",
	Entity["Language", "ChineseMinDong"] -> "zh-Hant",
	Entity["Language", "ChineseMinNan"] -> "zh-Hant",
	Entity["Language", "ChineseMinZhong"] -> "zh-Hant",
	Entity["Language", "ChinesePidginEnglish"] -> "zh-Hant",
	Entity["Language", "ChinesePuXian"] -> "zh-Hant",
	Entity["Language", "ChineseSignLanguage"] -> "zh-Hant",
	Entity["Language", "ChineseTibetanMongolian"] -> "zh-Hant",
	Entity["Language", "ChineseWu"] -> "zh-Hant",
	Entity["Language", "ChineseXiang"] -> "zh-Hant",
	Entity["Language", "ChineseYue"] -> "zh-Hant",
	Entity["WritingScript", "TraditionalChinese::bjw79"] -> "zh-Hant",
	Entity["WritingScript", "SimplifiedChinese::zzc7y"] -> "zh-Hans",
	Entity["Language", "Hebrew"] -> "he",
	Entity["Language", "HmongDaw"] -> "mww"
};

MTEntityFromLanguageCode[code_] := If[KeyExistsQ[MTLanguageCodeToEntityAlignment,code],
									code /. MTLanguageCodeToEntityAlignment,
									Missing["NotAvailble"]]

MTLanguageCodeToEntityAlignment = {
	"ar" -> Entity["Language", "Arabic"],
	"bg" -> Entity["Language", "Bulgarian"],
	"ca" -> Entity["Language", "CatalanValencianBalear"],
	"zh-Hans" -> Entity["WritingScript", "SimplifiedChinese::zzc7y"],
	"zh-Hant" -> Entity["WritingScript", "TraditionalChinese::bjw79"],
	"cs" -> Entity["Language", "Czech"],
	"da" -> Entity["Language", "Danish"],
	"nl" -> Entity["Language", "Dutch"],
	"en" -> Entity["Language", "English"],
	"et" -> Entity["Language", "Estonian"],
	"fi" -> Entity["Language", "Finnish"],
	"fr" -> Entity["Language", "French"],
	"de" -> Entity["Language", "German"],
	"el" -> Entity["Language", "Greek"],
	"ht" -> Entity["Language", "HaitianCreoleFrench"],
	"he" -> Entity["Language", "Hebrew"],
	"hi" -> Entity["Language", "Hindi"],
	"mww" -> Entity["Language", "HmongDaw"],
	"hu" -> Entity["Language", "Hungarian"],
	"id" -> Entity["Language", "Indonesian"],
	"it" -> Entity["Language", "Italian"],
	"ja" -> Entity["Language", "Japanese"],
	"ko" -> Entity["Language", "Korean"],
	"lv" -> Entity["Language", "Latvian"],
	"lt" -> Entity["Language", "Lithuanian"],
	"ms" -> Entity["Language", "Malay"],
	"mt" -> Entity["Language", "Maltese"],
	"no" -> Entity["Language", "Norwegian"],
	"fa" -> Entity["Language", "FarsiEastern"],
	"pl" -> Entity["Language", "Polish"],
	"pt" -> Entity["Language", "Portuguese"],
	"ro" -> Entity["Language", "Romanian"],
	"ru" -> Entity["Language", "Russian"],
	"sk" -> Entity["Language", "Slovak"],
	"sl" -> Entity["Language", "Slovenian"],
	"es" -> Entity["Language", "Spanish"],
	"sv" -> Entity["Language", "Swedish"],
	"th" -> Entity["Language", "Thai"],
	"tr" -> Entity["Language", "Turkish"],
	"uk" -> Entity["Language", "Ukrainian"],
	"ur" -> Entity["Language", "Urdu"],
	"vi" -> Entity["Language", "Vietnamese"],
	"cy" -> Entity["Language", "Welsh"]
};

allowedLanguages = {"af", "ar", "bs-Latn", "bg", "ca", "zh-Hans", "zh-Hant", "yue", "hr", "cs", "da", "nl", "en", "et", "fj", "fil", "fi", "fr", "de", "el", "ht", "he", "hi", "mww", "hu", "id", "it", "ja", "sw", "tlh", "tlh-Qaak", "ko", "lv", "lt", "mg", "ms", "mt", "yua", "no", "otq","fa", "pl", "pt", "ro", "ru", "sm", "sr-Cyrl", "sr-Latn", "sk", "sl", "es", "sv", "ty", "th", "to", "tr", "uk", "ur", "vi", "cy"};
End[];

EndPackage[];
