(* Wolfram Language package *)
BeginPackage["SurveyMonkeyFunctions`"] 

$langmap::usage="";
$fieldsRules::usage="";
$valfuns::usage="";
surveymonkeyformatdata::usage="";
convertLang::usage="";
camelCase::usage="";
capitalizekeys::usage="";
capitalizepages::usage="";
processResponsePerRespondent::usage="";
createRespResRow::usage="";
processAnswer::usage="";


Begin["`Private`"]

$langmap =<|Entity["Language", "English"] -> "en", 
 Entity["Language", "Chinese"] -> "zh", 
 Entity["Language", "Danish"] -> "da", 
 Entity["Language", "Dutch"] -> "nl", 
 Entity["Language", "Finnish"] -> "fi", 
 Entity["Language", "French"] -> "fr", 
 Entity["Language", "German"] -> "de", 
 Entity["Language", "Greek"] -> "el", 
 Entity["Language", "Italian"] -> "it", 
 Entity["Language", "Japanese"] -> "ja", 
 Entity["Language", "Korean"] -> "ko", 
 Entity["Language", "Malay"] -> "ms", 
 Entity["Language", "Norwegian"] -> "nb", 
 Entity["Language", "Polish"] -> "pl", 
 Entity["Language", "Russian"] -> "ru", 
 Entity["Language", "Spanish"] -> "es", 
 Entity["Language", "Swedish"] -> "sv", 
 Entity["Language", "Turkish"] -> "tr", 
 Entity["Language", "Ukrainian"] -> "uk", 
 Entity["Language", "Albanian"] -> "sq", 
 Entity["Language", "Arabic"] -> "ar", 
 Entity["Language", "Armenian"] -> "hy", 
 Entity["Language", "Basque"] -> "eu", 
 Entity["Language", "Bosnian"] -> "bs-BA", 
 Entity["Language", "Bulgarian"] -> "bg", 
 Entity["Language", "Croatian"] -> "hr", 
 Entity["Language", "Czech"] -> "cs", 
 Entity["Language", "Estonian"] -> "et", 
 Entity["Language", "Georgian"] -> "ka", 
 Entity["Language", "Hebrew"] -> "he", 
 Entity["Language", "Hindi"] -> "hi", 
 Entity["Language", "Hungarian"] -> "hu", 
 Entity["Language", "Icelandic"] -> "is", 
 Entity["Language", "Indonesian"] -> "id", 
 Entity["Language", "Latvian"] -> "lv", 
 Entity["Language", "Lithuanian"] -> "lt", 
 Entity["Language", "Macedonian"] -> "mk", 
 Entity["Language", "Romanian"] -> "ro", 
 Entity["Language", "Slovak"] -> "sk", 
 Entity["Language", "Slovenian"] -> "sl", 
 Entity["Language", "Swahili"] -> "sw", 
 Entity["Language", "Tamil"] -> "ta", 
 Entity["Language", "Telugu"] -> "te", 
 Entity["Language", "Thai"] -> "th", 
 Entity["Language", "Vietnamese"] -> "vi", 
 Entity["Language", "Welsh"] -> "cy"|>
	
(*Avoid changing ordering of keys. The idea is, more useful/important keys like ids,name,title should appear first in result*)
$fieldsRules = <|"SurveyID" -> "survey_id", 
   "TemplateID" -> "template_id", "RecipientID" -> "recipient_id", 
   "QuestionID" -> "question_id", "AnswerID" -> "answer_id", "ResponseID" -> "response_id",
   "CategoryID" -> "category_id", "CollectorID" -> "collector_id", 
   "RespondentID" -> "respondent_id", "PageID" -> "page_id", 
   "CustomID" -> "custom_id", "UserID" -> "user_id", 
   "Title" -> "title", "Name" -> "name", "Type" -> "type", 
   "Status" -> "status", "ResponseStatus" -> "response_status",
   "Category" -> "category", "CategoryName" -> "category_name",
   "CategoryDescription" -> "category_description","Description" -> "description",
   "DateCreated" -> "date_created", "DateModified" -> "date_modified",
   "DateLastLogin" -> "date_last_login", "DateStart" -> "date_start",
   "TotalTime" -> "total_time", "CollectionMode" -> "collection_mode", 
   "CustomVariables" -> "custom_variables", "Email" -> "email", "FirstName" -> "first_name", 
   "IPAddress" -> "ip_address", "IsAvailableToCurrentUser" -> "is_available_to_current_user", 
   "IsCertified" -> "is_certified", "IsFeatured" -> "is_featured", 
   "Language" -> "language", "LastName" -> "last_name", 
   "LongDescription" -> "long_description", "Open" -> "open", 
   "PageCount" -> "page_count", "Pages" -> "pages", "AnalysisURL" -> "analyze_url",
   "EditURL" -> "edit_url","PreviewURL" -> "preview", "QuestionCount" -> "question_count", 
   "RedirectURL" -> "redirect_url", "ResponseURL" -> "response_url", 
   "ShortDescription" -> "short_description", "URL" -> "url"|>
   
$valfuns = {"DateCreated" -> DateObject, "DateModified" -> DateObject, "Language" -> convertLang,
			"URL" -> URL,"PreviewURL" -> URL,"AnalysisURL" -> URL,"ResponseURL" -> URL,"RedirectURL" -> URL,
			"EditURL" -> URL}

convertLang[lang_String]:=With[{invlangrules = $langmap // Normal // GroupBy[Last -> First]},
	If[KeyExistsQ[invlangrules,lang],
		First@invlangrules[lang],
		(*else*)
		lang
	]
]

convertLang[lang_]:=lang

(*will take input as dataset containing list of associations*)
surveymonkeyformatdata[data_Dataset] := Module[{orgkeys, filteredkeys,newkeys,res,invfieldrules=$fieldsRules // Normal // GroupBy[Last -> First],
	valfuns},
   (*filter unwanted keys from result*)
   orgkeys =  Normal@data[1, Keys];
   filteredkeys =  Select[orgkeys, MemberQ[Values@$fieldsRules, #] &];
   (*convert to new keys (e.g date_created to DateCreated etc )*)
   newkeys = invfieldrules[#]& /@ filteredkeys // Flatten;
   (*Filter unwanted columns from result Dataset based on new keys*)
   res = data[All, AssociationThread[newkeys, filteredkeys]];
   (*finally convert dates and language values to respective Entities/DateObjects, if exists*)
   valfuns = FilterRules[$valfuns,newkeys];
   If[Length@valfuns>0,res[All,valfuns],res]
   ]
	
camelCase[text_] := Module[{split, partial},
    split = StringSplit[text, {" ","_","-"}];
    partial = Prepend[Rest[Characters[#]], ToUpperCase[Characters[#][[1]]]] & /@ split;
    StringJoin[partial]
    ]

(*Will capitalize(camel case to be precise) the keys in nested association*)
capitalizekeys[data_Association] := Replace[data, asoc_?AssociationQ :> KeyMap[camelCase][asoc], {0, Infinity}]

capitalizepages[data_Association] := Module[{newdata = data},
	If[!(MissingQ @ data["pages"]),
		newdata["pages"] = capitalizekeys[#]&/@data["pages"]
	];
	newdata
]

(*helper functions for SurveyResults request*)
processAnswer[lanswer_Association, ranswer_List] := 
 Module[{choicelist, choiceid, answerdata = {}, res},
  If[! KeyExistsQ[lanswer, "Choices"],
   	ranswer
   ];
  choicelist = lanswer["Choices"];
  (
     (*Select Text key from matching choice id*)
     If[KeyExistsQ[#, "ChoiceId"],
      (*surveyor selected one of the choices*)
      choiceid = #["ChoiceId"];
      res = SelectFirst[choicelist, MatchQ[#["Id"], choiceid] &];
      AppendTo[answerdata, res]
      ,
      (*else surveyor selected other option(not one of the choices provided)*)
      AppendTo[answerdata, #]
      ]
     ) & /@ ranswer;
  answerdata
  ]
  
(*unable to map response answer data(eg if lanswer is Missing), return actual answer as it is *)  
 processAnswer[lanswer_,ranswer_]:=  List@ranswer

createRespResRow[perpageresp_Association, pagedata_Association] := Module[{res, respques, pageques,lanswer,ranswer,answerdata},
   respques = perpageresp["Questions"];
   pageques = pagedata["Questions"];
   res = JoinAcross[pageques, respques, Key["Id"], "Inner", KeyCollisionFunction -> Function[x, {left[x], right[x]}]];
   If[KeyExistsQ[left["Answers"]]@# && KeyExistsQ[right["Answers"]]@#,
	lanswer = #[left["Answers"]];
	ranswer = #[right["Answers"]];
	answerdata = processAnswer[lanswer, ranswer];
	<|"QuestionID" -> #["Id"], "Headings" -> #["Headings"], "Answers" -> answerdata|>,
      (*else,no Key Collision*)
      <|"QuestionID" -> #["Id"], "Headings" -> #["Headings"], "Answers" -> #["Answers"]|>
      ] & /@ res
  ]
   
(*First param represents responsedata(pagewise detail) per respondent, Second param represents survey pages data fetched from SurveyDetail api *)
processResponsePerRespondent[responsedata_List, pagesdata_List] := Module[{quesdata, res = {},firstresp},
  (
     firstresp = #;
     (*Select page data corresponding to pageid in responsedata*)
     quesdata = SelectFirst[pagesdata, MatchQ[#["Id"], firstresp["Id"]] &];
     res = Join[res, createRespResRow[#, quesdata]]
     ) & /@ responsedata;
  res
  ]

End[]

EndPackage[]