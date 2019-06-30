BeginPackage["GoogleTranslateFunctions`"];

GTMapParameterToLanguageCode::usage = "";
GTProcessRequest::usage = "";
GTCookedImport::usage = "";
GTFormatRequestParameters::usage = "";
GTEntityToLanguageCodeAlignment::usage = "";

Begin["`Private`"];

GTFormatRequestParameters[args_]:= Block[{invalidParameters,languagerules,text,from,to,msl = False, params=<||>, tmpValue},
	invalidParameters = Select[Keys[args],!MemberQ[{"LanguageRules","Text","From","To"},#]&];
	If[Length[invalidParameters]>0,
	(
		Message[ServiceObject::noget,#,"GoogleTranslate"]&/@invalidParameters;
		Throw[$Failed]
	)];
	
	languagerules = Lookup[args, "LanguageRules", False];
	If[ !BooleanQ[languagerules],
		Message[ServiceExecute::nval,"LanguageRules","GoogleTranslate"];
		Throw[$Failed]
	];

	text = Lookup[args, "Text", Message[ServiceExecute::nparam,"Text","GoogleTranslate"]; Throw[$Failed]];
	If[ !MatchQ[text, _?StringQ | {__?StringQ}],
		Message[ServiceExecute::nval,"Text","GoogleTranslate"];
		Throw[$Failed]
	];

	If[KeyExistsQ[args,"From"],
		(
			tmpValue = from = Lookup[args, "From"];
			If[ !MatchQ[from, _String | _Entity],
				Message[ServiceExecute::nval,"From","GoogleTranslate"];
				Throw[{"tag" -> "nval", "param" -> "From", "value" -> tmpValue}]
			];
			from = GTMapParameterToLanguageCode[from];
			If[FailureQ[from],
				Message[ServiceExecute::nval,"From","GoogleTranslate"];
				Throw[{"tag" -> "nval", "param" -> "From", "value" -> tmpValue}]
			];
			If[ListQ[text], from = ConstantArray[from, Length[text]]];

		),
			If[StringQ[text],
				from = Quiet[GTMapParameterToLanguageCode[tmpValue = Classify["Language", text]]];
				If[FailureQ[from],
					Message[ServiceExecute::nval,"From","GoogleTranslate"];
					Throw[{"tag" -> "nval", "param" -> "From", "value" -> tmpValue}]
				]
			,(*error on lists?*)
				from = Quiet[GTMapParameterToLanguageCode[Classify["Language", #]]& /@ text]
			];
	];

	If[ListQ[from] && (Length[DeleteDuplicates[from]] > 1), msl = True];

	If[KeyExistsQ[args,"To"],
		(
			tmpValue = to = Lookup[args, "To"];
			If[ !MatchQ[to, _String | _Entity],
				Message[ServiceExecute::nval,"To","GoogleTranslate"];
				Throw[{"tag" -> "nval", "param" -> "To", "value" -> tmpValue}]
			];
			to = Quiet[GTMapParameterToLanguageCode[to]];
			If[FailureQ[to],
				Message[ServiceExecute::nval,"To","GoogleTranslate"];
				Throw[{"tag" -> "nval", "param" -> "To", "value" -> tmpValue}]
			];
		),
			Message[ServiceExecute::nparam,"To","GoogleTranslate"];
			Throw[{"tag" -> "nparam", "param" -> "To"}]
	];
	params["Text"] = text;
	params["From"] = from;
	params["To"] = to;
	params["LanguageRules"] = languagerules;
	params["msl"] = msl;
	params
]

GTMapParameterToLanguageCode[param_] := Catch[Block[
	{code,entity},
	Switch[param,
			_Entity, (* language represented as a WL Entity *)
				code = entityToLanguageCode[param];
				If[MissingQ[code], Throw[$Failed], Throw[code]],
			_?StringQ,
				If[ !MemberQ[allowedLanguageCodes, param],
					entity = Quiet[Interpreter["Language"][param]];
					code = entityToLanguageCode[entity];
					If[ MissingQ[code], Throw[$Failed], Throw[code]]
					,
					Throw[param]
				],
			_,
				Throw[$Failed] (* Invalid language code *)
	];
]]

GTProcessRequest[id_,p_] := Block[
    {text=p["Text"], from=p["From"], to=p["To"], msl=p["msl"], result={}, textgroupby, statictext, preresult},
    If[StringQ[text], (*If just one string is provided*)
		If[SameQ[from,to], (*If source language is the same as target, the same string is returned.*)
		result = {text},
		result = GoogleTranslate`Private`GTFormatResults@KeyClient`rawkeydata[id,"RawTranslate",{"q" -> text, "source" -> from, "target" -> to, "format" -> "text"}];
		If[FailureQ[result], Throw[$Failed]];
		result = Lookup[result["data"]["translations"], "translatedText"];
		]
		,
		(*If a list with texts is provided*)
		If[msl,(*If multiple source languages are detected*)

			(*the texts are grouped by language to reduce the number of calls to the api. An index is assigned to texts to keep the original order at the end*)
			textgroupby = GroupBy[MapThread[List, {from, text, Range[Length@text]}], First];

			(*Cheks if any source language matches the target language and removes them from the list to be translated*)
			If[MemberQ[from, to], statictext = Cases[Flatten[List@@textgroupby, 1], {to, _, _Integer}]; textgroupby = KeyDrop[textgroupby, to], statictext={}];

			(*Calls the API*)
			preresult = Block[{gtexts = textgroupby[#][[All, 2]], gindexes = textgroupby[#][[All, 3]],temp},
				temp = GoogleTranslate`Private`GTFormatResults@KeyClient`rawkeydata[id,"RawTranslate",{"q"-> gtexts,"source"->#,"target"->to, "format" -> "text"}];
				If[FailureQ[temp],
   	    			temp = ConstantArray[$Failed, Length[gtexts]], (* guarantees that only what fails returns $Failed *)
   	    			temp = Lookup[temp["data"]["translations"], "translatedText"]
 				];
				MapThread[List, {gindexes, temp}]
				]&/@DeleteDuplicates[DeleteCases[from, to]];

			(*The original order is restored*)
			If[statictext === {},	(*If there's no text with source language = target language*)
				result = SortBy[Flatten[preresult, 1], First][[All, 2]];,
				result = SortBy[Union[Flatten[preresult, 1], (Reverse /@ statictext)[[All, 1 ;; 2]]], First][[All, 2]];
			]
			,
			(*Just one source language*)
			If[MemberQ[from, to],
				result = text,
				result = GoogleTranslate`Private`GTFormatResults@KeyClient`rawkeydata[id, {"q" -> text, "source" -> from[[1]], "target" -> to, "format" -> "text"}];
				If[FailureQ[result], Throw[$Failed]];
				result = Lookup[result["data"]["translations"], "translatedText"];
			]

		]
	];
    result
]

GTCookedImport[result_, params_]:= Block[
    {text=params["Text"], from=params["From"], languagerules=params["LanguageRules"]},
    Switch[text,
		_?StringQ,
			If[result === {},
				""
			,
				If[languagerules,
					Lookup[languageCodeToEntityAlignment,from] -> result[[1]],
					result[[1]]
				]
			]
		,

		_?ListQ,
			If[languagerules,
				MapThread[Rule, {Lookup[languageCodeToEntityAlignment,from], Flatten[result]}],
				Flatten[result]
			]
	]
]

entityToLanguageCode[entity_]:= Lookup[GTEntityToLanguageCodeAlignment, entity, Missing["NotAvailble"]]

entityFromLanguageCode[code_]:= Lookup[languageCodeToEntityAlignment, code, Missing["NotAvailble"]]

GTEntityToLanguageCodeAlignment = {Entity["Language", "Afrikaans"] -> "af",
 EntityClass["Language", "Albanian"] -> "sq",
 Entity["Language", "Arabic"] -> "ar",
 EntityClass["Language", "Azerbaijani"] -> "az",
 Entity["Language", "Basque"] -> "eu",
 Entity["Language", "Bengali"] -> "bn",
 Entity["Language", "Belarusan"] -> "be",
 Entity["Language", "Bulgarian"] -> "bg",
 Entity["Language", "CatalanValencianBalear"] -> "ca",
 Entity["Language", "Croatian"] -> "hr",
 Entity["Language", "Czech"] -> "cs",
 Entity["Language", "Danish"] -> "da",
 Entity["Language", "Dutch"] -> "nl",
 Entity["Language", "English"] -> "en",
 Entity["Language", "Esperanto"] -> "eo",
 Entity["Language", "Estonian"] -> "et",
 Entity["Language", "Filipino"] -> "tl",
 Entity["Language", "Finnish"] -> "fi",
 Entity["Language", "French"] -> "fr",
 Entity["Language", "Galician"] -> "gl",
 Entity["Language", "Georgian"] -> "ka",
 Entity["Language", "German"] -> "de",
 Entity["Language", "Greek"] -> "el",
 Entity["Language", "Gujarati"] -> "gu",
 Entity["Language", "HaitianCreoleFrench"] -> "ht",
 Entity["Language", "Hebrew"] -> "iw",
 Entity["Language", "Hindi"] -> "hi",
 Entity["Language", "Hungarian"] -> "hu",
 Entity["Language", "Icelandic"] -> "is",
 Entity["Language", "Indonesian"] -> "id",
 Entity["Language", "IrishGaelic"] -> "ga",
 Entity["Language", "Italian"] -> "it",
 Entity["Language", "Japanese"] -> "ja",
 Entity["Language", "Kannada"] -> "kn",
 Entity["Language", "Korean"] -> "ko",
 Entity["Language", "Latin"] -> "la",
 Entity["Language", "Latvian"] -> "lv",
 Entity["Language", "Lithuanian"] -> "lt",
 Entity["Language", "Macedonian"] -> "mk",
 Entity["Language", "Malay"] -> "ms",
 Entity["Language", "Maltese"] -> "mt",
 Entity["Language", "Norwegian"] -> "no",
 Entity["Language", "FarsiEastern"] -> "fa",
 Entity["Language", "Polish"] -> "pl",
 Entity["Language", "Portuguese"] -> "pt",
 Entity["Language", "Romanian"] -> "ro",
 Entity["Language", "Russian"] -> "ru",
 Entity["Language", "Serbian"] -> "sr",
 Entity["Language", "Slovak"] -> "sk",
 Entity["Language", "Slovenian"] -> "sl",
 Entity["Language", "Spanish"] -> "es",
 Entity["Language", "Swahili"] -> "sw",
 Entity["Language", "Swedish"] -> "sv",
 Entity["Language", "Tamil"] -> "ta",
 Entity["Language", "Telugu"] -> "te",
 Entity["Language", "Thai"] -> "th",
 Entity["Language", "Turkish"] -> "tr",
 Entity["Language", "Ukrainian"] -> "uk",
 Entity["Language", "Urdu"] -> "ur",
 Entity["Language", "Vietnamese"] -> "vi",
 Entity["Language", "Welsh"] -> "cy",
 Entity["Language", "YiddishEastern"] -> "yi",
 Entity["Language", "Chinese"] -> "zh-CN",
 Entity["Language", "ChineseGan"] -> "zh-CN",
 Entity["Language", "ChineseHakka"] -> "zh-CN",
 Entity["Language", "ChineseHuizhou"] -> "zh-CN",
 Entity["Language", "ChineseJinyu"] -> "zh-CN",
 Entity["Language", "ChineseMandarin"] -> "zh-CN",
 Entity["Language", "ChineseMinBei"] -> "zh-CN",
 Entity["Language", "ChineseMinDong"] -> "zh-CN",
 Entity["Language", "ChineseMinNan"] -> "zh-CN",
 Entity["Language", "ChineseMinZhong"] -> "zh-CN",
 Entity["Language", "ChinesePidginEnglish"] -> "zh-CN",
 Entity["Language", "ChinesePuXian"] -> "zh-CN",
 Entity["Language", "ChineseSignLanguage"] -> "zh-CN",
 Entity["Language", "ChineseTibetanMongolian"] -> "zh-CN",
 Entity["Language", "ChineseWu"] -> "zh-CN",
 Entity["Language", "ChineseXiang"] -> "zh-CN",
 Entity["Language", "ChineseYue"] -> "zh-CN",
 Entity["Language", "ChineseScript"] -> "zh-CN",
 Entity["Language", "FarsiWestern"] -> "fa"};

 languageCodeToEntityAlignment = {"af" -> Entity["Language", "Afrikaans"],
 "sq" -> EntityClass["Language", "Albanian"],
 "ar" -> Entity["Language", "Arabic"],
 "az" -> EntityClass["Language", "Azerbaijani"],
 "eu" -> Entity["Language", "Basque"],
 "bn" -> Entity["Language", "Bengali"],
 "be" -> Entity["Language", "Belarusan"],
 "bg" -> Entity["Language", "Bulgarian"],
 "ca" -> Entity["Language", "CatalanValencianBalear"],
 "hr" -> Entity["Language", "Croatian"],
 "cs" -> Entity["Language", "Czech"],
 "da" -> Entity["Language", "Danish"],
 "nl" -> Entity["Language", "Dutch"],
 "en" -> Entity["Language", "English"],
 "eo" -> Entity["Language", "Esperanto"],
 "et" -> Entity["Language", "Estonian"],
 "tl" -> Entity["Language", "Filipino"],
 "fi" -> Entity["Language", "Finnish"],
 "fr" -> Entity["Language", "French"],
 "gl" -> Entity["Language", "Galician"],
 "ka" -> Entity["Language", "Georgian"],
 "de" -> Entity["Language", "German"],
 "el" -> Entity["Language", "Greek"],
 "gu" -> Entity["Language", "Gujarati"],
 "ht" -> Entity["Language", "HaitianCreoleFrench"],
 "iw" -> Entity["Language", "Hebrew"],
 "hi" -> Entity["Language", "Hindi"],
 "hu" -> Entity["Language", "Hungarian"],
 "is" -> Entity["Language", "Icelandic"],
 "id" -> Entity["Language", "Indonesian"],
 "ga" -> Entity["Language", "IrishGaelic"],
 "it" -> Entity["Language", "Italian"],
 "ja" -> Entity["Language", "Japanese"],
 "kn" -> Entity["Language", "Kannada"],
 "ko" -> Entity["Language", "Korean"],
 "la" -> Entity["Language", "Latin"],
 "lv" -> Entity["Language", "Latvian"],
 "lt" -> Entity["Language", "Lithuanian"],
 "mk" -> Entity["Language", "Macedonian"],
 "ms" -> Entity["Language", "Malay"],
 "mt" -> Entity["Language", "Maltese"],
 "no" -> Entity["Language", "Norwegian"],
 "fa" -> Entity["Language", "FarsiEastern"],
 "pl" -> Entity["Language", "Polish"],
 "pt" -> Entity["Language", "Portuguese"],
 "ro" -> Entity["Language", "Romanian"],
 "ru" -> Entity["Language", "Russian"],
 "sr" -> Entity["Language", "Serbian"],
 "sk" -> Entity["Language", "Slovak"],
 "sl" -> Entity["Language", "Slovenian"],
 "es" -> Entity["Language", "Spanish"],
 "sw" -> Entity["Language", "Swahili"],
 "sv" -> Entity["Language", "Swedish"],
 "ta" -> Entity["Language", "Tamil"],
 "te" -> Entity["Language", "Telugu"],
 "th" -> Entity["Language", "Thai"],
 "tr" -> Entity["Language", "Turkish"],
 "uk" -> Entity["Language", "Ukrainian"],
 "ur" -> Entity["Language", "Urdu"],
 "vi" -> Entity["Language", "Vietnamese"],
 "cy" -> Entity["Language", "Welsh"],
 "yi" -> Entity["Language", "YiddishEastern"],
 "zh-CN" -> Entity["Language", "Chinese"],
 "zh-CN" -> Entity["Language", "ChineseGan"],
 "zh-CN" -> Entity["Language", "ChineseHakka"],
 "zh-CN" -> Entity["Language", "ChineseHuizhou"],
 "zh-CN" -> Entity["Language", "ChineseJinyu"],
 "zh-CN" -> Entity["Language", "ChineseMandarin"],
 "zh-CN" -> Entity["Language", "ChineseMinBei"],
 "zh-CN" -> Entity["Language", "ChineseMinDong"],
 "zh-CN" -> Entity["Language", "ChineseMinNan"],
 "zh-CN" -> Entity["Language", "ChineseMinZhong"],
 "zh-CN" -> Entity["Language", "ChinesePidginEnglish"],
 "zh-CN" -> Entity["Language", "ChinesePuXian"],
 "zh-CN" -> Entity["Language", "ChineseSignLanguage"],
 "zh-CN" -> Entity["Language", "ChineseTibetanMongolian"],
 "zh-CN" -> Entity["Language", "ChineseWu"],
 "zh-CN" -> Entity["Language", "ChineseXiang"],
 "zh-CN" -> Entity["Language", "ChineseYue"],
 "zh-CN" -> Entity["Language", "ChineseScript"],
 "fa" -> Entity["Language", "FarsiWestern"]};

allowedLanguageCodes = {"af", "am", "ar", "az", "be", "bg", "bn", "bs", "ca", "ceb", "co", "cs", "cy", "da", "de", "el", "en", "eo", "es", "et", "eu", "fa","fi", "fr", "fy", "ga", "gd", "gl", "gu", "ha", "haw", "hi", "hmn", "hr", "ht", "hu", "hy", "id", "ig", "is", "it", "iw", "ja", "jw", "ka", "kk", "km", "kn", "ko", "ku", "ky", "la", "lb", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "no", "ny", "pa", "pl", "ps", "pt", "ro", "ru", "sd", "si", "sk", "sl", "sm", "sn", "so", "sq", "sr", "st", "su", "sv", "sw", "ta", "te", "tg", "th", "tl", "tr", "uk", "ur", "uz", "vi", "xh", "yi", "yo", "zh", "zh-TW", "zu"};
(* GoogleTranslateAPI`Private`importLanguageCodes[KeyClient`rawkeydata[id,"RawLanguageCodeList",{}]] *)

End[];

EndPackage[];
