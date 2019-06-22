Get["GoogleTranslateFunctions`"]

Begin["GoogleTranslate`"] (* Begin Private Context *)

Begin["`Private`"]

(******************************* GoogleTranslate *************************************)

(* Authentication information *)
googletranslatedata[]:={
		"ServiceName" 		-> "Google Translate",
        "URLFetchFun"		:> (Block[{params=Lookup[{##2},"Parameters",{}], q}, q = Lookup[params,"q"];
        	If[ListQ[q], params = Join[Normal@KeyDrop[params, "q"], ("q"->#)&/@q]];
        	URLRead[HTTPRequest[#1, <|"Query" -> params|>], "BodyBytes"]]&),

        "ClientInfo"		:> OAuthDialogDump`Private`MultipleKeyDialog["GoogleTranslate",{"API Key"->"key"},"https://console.developers.google.com/flows/enableapi?apiid=translate","https://cloud.google.com/translate/v2/terms"],
	 	"Gets"				-> {"Translate"},
	 	"RawGets"			-> {"RawTranslate", "RawLanguageCodeList"},
	 	"Posts"				-> {},
	 	"RawPosts"			-> {},
 		"Information"		-> "Import Google Translate API data to the Wolfram Language"
 		}

GTFormatResults[response_]:= With[
	{data = Developer`ReadRawJSONString[Quiet[FromCharacterCode[response, "UTF8"], {$CharacterEncoding::utf8}], "IssueMessagesAs" -> Symbol["GoogleTranslate"]]},

	If[AssociationQ[data],
		If[KeyExistsQ[data, "error"],
			Message[ServiceExecute::serrormsg, data["error"]["message"]];
			$Failed
		,
			data
		]
	,
		$Failed
	]
]

importLanguageCodes[response_]:= With[
	{data = Developer`ReadRawJSONString[Quiet[FromCharacterCode[response, "UTF8"], {$CharacterEncoding::utf8}], "IssueMessagesAs" -> Symbol["GoogleTranslate"]]},
	
	If[AssociationQ[data],
		If[KeyExistsQ[data, "error"],
			Message[ServiceExecute::serrormsg, data["error"]["message"]];
			$Failed
		,
			Lookup[data["data"]["languages"], "language"]
		]
	,
		$Failed
	]
]

(* Raw *)
googletranslatedata["RawTranslate"] := {
        "URL"				-> "https://translation.googleapis.com/language/translate/v2",
        "HTTPSMethod"		-> "GET",
        "Parameters"		-> {"format","prettyprint","q","source","target"},
        "RequiredParameters"-> {"q","target"},
        "ResultsFunction"	-> GTFormatResults
        }

googletranslatedata["RawLanguageCodeList"] := {
        "URL"				-> "https://translation.googleapis.com/language/translate/v2/languages",
		"HTTPSMethod"		-> "GET",
        "Parameters"		-> {},
        "RequiredParameters"-> {},
        "ResultsFunction"	-> importLanguageCodes
    }

(* Cooked *)

googletranslatecookeddata["Translate", id_, args_] := Block[
	{rParams = Check[Catch[GTFormatRequestParameters[args]], Throw[$Failed]], result},
	result = GTProcessRequest[id, rParams];
	GTCookedImport[result, rParams]
]

(*Utilities*)

googletranslatecookeddata[___]:=$Failed

googletranslatesendmessage[___]:=$Failed

End[]

End[]

SetAttributes[{},{ReadProtected, Protected}];

(* Return three functions to define keyservicedata, keycookeddata, keysendmessage *)

{GoogleTranslate`Private`googletranslatedata,
 GoogleTranslate`Private`googletranslatecookeddata,
 GoogleTranslate`Private`googletranslatesendmessage}
