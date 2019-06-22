Get["MicrosoftTranslatorFunctions`"]

Begin["MicrosoftTranslator`"] (* Begin Private Context *)

Begin["`Private`"](* Begin Private Context *)

(******************************* AT&T Speech API*************************************)

(* Authentication information *)

microsofttranslatordata[]={
		"ServiceName" 		-> "MicrosoftTranslator",
        "URLFetchFun"		:> (With[{params=Lookup[{##2},"Parameters",{}]},
        	(
						URLRead[HTTPRequest[
								#1,
								<|
										Method -> "POST",
										"Headers" -> {"Ocp-Apim-Subscription-Key"-> Lookup[params, "key", ""]},
										"Query" -> KeyDrop[params, "key"]
								|>],{"StatusCode", "Body"}
						]
					)
					]&)
        	,
        "ClientInfo"		:> OAuthDialogDump`Private`MultipleKeyDialog["MicrosoftTranslator",{"Subscription Key"->"key"},
        				"https://portal.azure.com","https://azure.microsoft.com/en-us/support/legal"],
	 	"Gets"				-> {"LanguageList","LanguageEntityList"},
	 	"Posts"				-> {"Translate","RoundTripTranslate"},
	 	"RawGets"			-> {"RawGetLanguagesForTranslate","RawLanguageCodeList"},
	 	"RawPosts"			-> {"RawTranslateArray","RawTranslate"},
 		"Information"		-> "Use Microsoft Translator API with Wolfram Language"
}

(*microsofttranslatorimport[rawdata_]:=ImportString[StringReplace[StringReplace[ToString[rawdata], "[" ~~ data___ ~~ "]" :> data], ___ ~~ "," ~~ xml___ ~~ "}" :> xml], "XML"]*)
microsofttranslatorimport[rawdata_]:= Module[{},
(
	(*data = StringReplace[ToString[rawdata], "[" ~~ d___ ~~ "]" :> d]; This was before using ContentData*)
	Association["StatusCode"->rawdata[[1]],"Body"->ImportString[
 ToString[FromCharacterCode[rawdata[[2]], "UTF-8"],
  CharacterEncoding -> "UTF8"], "RawJSON"]]
	(*rawdata2 = ToString[{rawdata[[1]], StringReplace[ToString[FromCharacterCode[rawdata[[2]], "UTF8"]], "[" ~~ data___ ~~ "]" :> data]}];
	dataxml = ImportString[StringReplace[rawdata2, Shortest[___] ~~ "," ~~ xml___ ~~ "}" :> xml], "XML"];
	dataxml*)
	(*
	data = StringReplace[ToString[FromCharacterCode[rawdata[[2]], "UTF8"]], "[" ~~ d___ ~~ "]" :> d];
	StringReplace[data, Shortest[___] ~~ "," ~~ xml___ ~~ "}" :> xml]
	*)
)]

importLanguageCodes[rawdata_]:= Flatten[ImportString[microsofttranslatorimport[rawdata], "XML"][[2, 3]][[All, 3]]];
(*importLanguageCodes[rawdata_]:= Flatten[ImportString[StringReplace[ToString@rawdata,"[" ~~ xml___ ~~ "]" :> xml], "XML"][[2, 3]][[All, 3]]];*)

(* Raw *)
microsofttranslatordata["RawGetLanguagesForTranslate"] := {
        "URL"				-> "https://api.cognitive.microsofttranslator.com/languages",
		"HTTPSMethod"		-> "GET",
        "Parameters"		-> {"api-version"},
        "RequiredParameters"-> {"api-version"},
        "ResultsFunction"	-> microsofttranslatorimport
    }

microsofttranslatordata["RawLanguageCodeList"] := {
        "URL"				-> "https://api.cognitive.microsofttranslator.com/languages",
		"HTTPSMethod"		-> "GET",
        "Parameters"		-> {},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> importLanguageCodes
    }

microsofttranslatordata["RawTranslate"] := {
        "URL"				-> "https://api.cognitive.microsofttranslator.com/translate",
		"HTTPSMethod"		-> "POST",
		"Headers"			-> {"Content-Type" -> "application/json"},
		"Parameters"		-> {"api-version","from","to"},
		"BodyData"			-> {"ParameterlessBodyData"->"Data"},
    "RequiredParameters"-> {"to","Data"},
    "ResultsFunction"	-> microsofttranslatorimport
    }

microsofttranslatordata["RawTranslateArray"] := {
        "URL"				-> "http://api.microsofttranslator.com/V2/Http.svc/TranslateArray",
		"HTTPSMethod"		-> "POST",
		"BodyData"			-> {"ParameterlessBodyData"->"Data"},
		"Headers"			-> {"Content-Type" -> "text/xml"},
		"RequiredParameters"-> {"Data"},
        "ResultsFunction"	-> microsofttranslatorimport
    }



(* Cooked *)

microsofttranslatorcookeddata["LanguageList", id_, args_] := Block[{rawdata,dataxml,list,result,status,msg},
	If[Length[args]>0,
		(
			Message[ServiceObject::noget,#[[1]],"MicrosoftTranslator"]&/@args;
			Throw[$Failed]
		)];


	rawdata = KeyClient`rawkeydata[id,"RawGetLanguagesForTranslate",{"api-version"->"3.0"}];

	rawdata = microsofttranslatorimport[rawdata];
	Keys[rawdata["Body"]["translation"]]
]

microsofttranslatorcookeddata["LanguageEntityList", id_, args_] := Block[{rawdata,dataxml,list,result,status,newlist},
	If[Length[args]>0,
		(
			Message[ServiceObject::noget,#[[1]],"MicrosoftTranslator"]&/@args;
			Throw[$Failed]
		)];
	MTEntityToLanguageCodeAlignment[[All,1]]
]

microsofttranslatorcookeddata["Translate", id_, args_] := Block[
	{rParams = Check[Catch[MTFormatRequestParameters[args]], Throw[$Failed]], result},
	result = MTProcessRequest[id, rParams[[2]]];
	MTCookedImport[result,rParams[[2]]]
]

microsofttranslatorcookeddata["RoundTripTranslate", id_, args_] := Block[{params,textP,toP,fromP,rawdata,dataxml,translations,fwdTranslation,rwdTranslation,status,invalidParameters,msg},
	invalidParameters = Select[Keys[args],!MemberQ[{"Text","From","To"},#]&];

	If[Length[invalidParameters]>0,
		(
			Message[ServiceObject::noget,#,"MicrosoftTranslator"]&/@invalidParameters;
			Throw[$Failed]
		)];

	If[KeyExistsQ[args,"Text"],
		textP = URLEncode["Text" /. args],
		(
			Message[ServiceExecute::nparam,"Text"];
			Throw[$Failed]
		)
	];

	If[KeyExistsQ[args,"From"],
		(
			fromP = "From" /. args;
			fromP = MTMapParameterToLanguageCode[fromP];
			If[fromP == "Failed",
			(
				Message[ServiceExecute::nval,"From","MicrosoftTranslator"];
				Throw[$Failed]
			)]
		),
		(
			Message[ServiceExecute::nparam,"From"];
			Throw[$Failed]
		)
	];

	If[KeyExistsQ[args,"To"],
		(
			toP = "To" /. args;
			toP = MTMapParameterToLanguageCode[toP];
			If[toP == "Failed",
			(
				Message[ServiceExecute::nval,"To","MicrosoftTranslator"];
				Throw[$Failed]
			)]
		),
		(
			Message[ServiceExecute::nparam,"To"];
			Throw[$Failed]
		)
	];

	params = {"text"->textP, "from"->fromP, "to"->toP, "maxTranslations"->"1"};

	rawdata = KeyClient`rawkeydata[id,"RawGetTranslations",params];
	(*rawdata = ServiceExecute["MicrosoftTranslator", "RawGetTranslations", params];*)
	rawdata = StringReplace[ToString[rawdata], "[" ~~ data___ ~~ "]" :> data];
	dataxml = ImportString[StringReplace[rawdata, Shortest[___] ~~ "," ~~ xml___ ~~ "}" :> xml], "XML"];

	status = StringReplace[rawdata, "{" ~~ Shortest[s___] ~~ "," ~~ ___ :> s];
	If[status != "200",
	(
		msg = Cases[dataxml, XMLElement["p", _, content_List] :> content, Infinity];
		Message[ServiceExecute::serrormsg,msg];
		Throw[$Failed]
	)];

	fwdTranslation = Cases[dataxml,XMLElement["TranslatedText",_,text_]:>text,Infinity][[1,1]];

	params = {"text"->URLEncode[fwdTranslation], "from"->toP, "to"->fromP, "maxTranslations"->"1"};

	rawdata = KeyClient`rawkeydata[id,"RawGetTranslations",params];
	(*rawdata = ServiceExecute["MicrosoftTranslator", "RawGetTranslations", params];*)
	rawdata = StringReplace[ToString[rawdata], "[" ~~ data___ ~~ "]" :> data];
	dataxml = ImportString[StringReplace[rawdata, Shortest[___] ~~ "," ~~ xml___ ~~ "}" :> xml], "XML"];
	status = StringReplace[rawdata, "{" ~~ Shortest[s___] ~~ "," ~~ ___ :> s];
	If[status != "200",
	(
		msg = Cases[dataxml, XMLElement["p", _, content_List] :> content, Infinity];
		Message[ServiceExecute::serrormsg,msg];
		Throw[$Failed]
	)];

	rwdTranslation = Cases[dataxml,XMLElement["TranslatedText",_,text_]:>text,Infinity][[1,1]];

	Dataset@Association[Rule["Original",URLDecode[textP]],Rule["Translated to " <> toP,fwdTranslation],Rule["Translated to " <> fromP,rwdTranslation]]
]

microsofttranslatorcookeddata[prop_,id_,rules___Rule]:=microsofttranslatorcookeddata[prop,id,{rules}]

microsofttranslatorcookeddata[___]:=$Failed

microsofttranslatorsendmessage[___]:=$Failed


(* Utilities *)
getallparameters[str_]:=DeleteCases[Flatten[{"Parameters","PathParameters","BodyData","MultipartData"}/.microsofttranslatordata[str]],
	("Parameters"|"PathParameters"|"BodyData"|"MultipartData")]

getToken[subKey_] := URLFetch[
						"https://api.cognitive.microsoft.com/sts/v1.0/issueToken",
						"Method" -> "POST",
						"Headers" -> {"Ocp-Apim-Subscription-Key" -> subKey}
					]

parseTranslateOutput[output_] := Module[{xml},(
	(*xml = ImportString[StringReplace[ToString@output, "[" ~~ x___ ~~ "]" :> x], "XML"];*)
	Cases[output, XMLElement["string", _, text_] :> text, Infinity][[1, 1]]
)]

parseTranslateArrayOutput[output_] := Module[{xml},(
	(*xml = ImportString[StringReplace[ToString@output, "[" ~~ x___ ~~ "]" :> x],"XML"];*)
	Cases[output, XMLElement["TranslatedText", _, text_] :> text, Infinity] // Flatten
)]

formatTranslations[t_] :=
 Cases[t, {XMLElement["Count", {}, {count_}],
    XMLElement["MatchDegree", {}, {mDeg_}],
    XMLElement["MatchedOriginalText", {}, _],
    XMLElement["Rating", {}, {rating_}],
    XMLElement["TranslatedText", {}, {text_}]} :>
   Sequence[Rule["Count", count], Rule["MatchDegree", mDeg],
    Rule["Rating", rating], Rule["TranslatedText", text]]]

translateArrayRequestXML[textList_,to_,from_] := Module[{result=""},
	(
		result = result <> "<TranslateArrayRequest><AppId />";
		result = result <> "<From>" <> from <> "</From>";
		result = result <> "<Options> <Category xmlns='http://schemas.datacontract.org/2004/07/Microsoft.MT.Web.Service.V2' />";
		result = result <> "<ContentType xmlns='http://schemas.datacontract.org/2004/07/Microsoft.MT.Web.Service.V2'>text/plain</ContentType>";
		result = result <> "<ReservedFlags xmlns='http://schemas.datacontract.org/2004/07/Microsoft.MT.Web.Service.V2' />";
		result = result <> "<State xmlns='http://schemas.datacontract.org/2004/07/Microsoft.MT.Web.Service.V2' />";
		result = result <> "<Uri xmlns='http://schemas.datacontract.org/2004/07/Microsoft.MT.Web.Service.V2' />";
		result = result <> "<User xmlns='http://schemas.datacontract.org/2004/07/Microsoft.MT.Web.Service.V2' /> </Options>";
		result = result <> "<Texts>";

		(result = result <> "<string xmlns='http://schemas.microsoft.com/2003/10/Serialization/Arrays'>" <> # <> "</string>")&/@textList;

		result = result <> "</Texts> <To>" <> to <> "</To> </TranslateArrayRequest>";

		result
	)]



End[]

End[]

SetAttributes[{},{ReadProtected, Protected}];

(* Return three functions to define oauthservicedata, oauthcookeddata, oauthsendmessage  *)
{MicrosoftTranslator`Private`microsofttranslatordata,
 MicrosoftTranslator`Private`microsofttranslatorcookeddata,
 MicrosoftTranslator`Private`microsofttranslatorsendmessage}
