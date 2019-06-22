System`WordFrequency;
System`WordFrequencyData;

Begin["System`WordFrequencyDump`Private`"];

(* | Messages | Will be moved to errmsg.m once this is in master *)
WordFrequency::norec = "Unable to locate resource files for WordFrequency.";
WordFrequencyData::norec = "Unable to locate resource files for WordFrequencyData.";

$failTagWF = "WordFrequencyCatchThrowFailTag";
$failTagWFD = "WordFrequencyDataCatchThrowFailTag";
$unevaluatedTagWF = "WordFrequencyUnevaluatedTag";
$unevaluatedTagWFD = "WordFrequencyDataUnevaluatedTag";
(*failure function if something goes wrong during loading; issues message and ensure the context is exited before throwing*)
loadingBailout[] := CompoundExpression[
	Message[WordFrequency::norec],
	End[],
	Throw[$Failed, "WordFrequencyLoadingFailed"]
]

loadingBailout[] := CompoundExpression[
	Message[WordFrequencyData::norec],
	End[],
	Throw[$Failed, "WordFrequencyDataLoadingFailed"]
]

(* Cache functions *)

initCache[] := (EntityFramework`Caching`Private`getLastUpdate["WordFrequencyData", _] := 0);

iAddToCache[{eargs___}, evalue_] := (
	initCache[];
	iAddToCache[{args___}, value_] := EntityFramework`Caching`Private`addToCache[Entity["WordFrequencyData", args], "API", value];
	iAddToCache[{eargs}, evalue]
)

iGetFromCache[{eargs___}] := (
	initCache[]; 
	iGetFromCache[{args___}] := EntityFramework`Caching`Private`getFromCache[Entity["WordFrequencyData", args], "API"];
	iGetFromCache[{eargs}]
)
  
(* interface with Wolfram Alpha *)
iWolframAlphaAPICompute[head_, type_, input___] := Module[{res, msg, return = None},
	If[type === "MWAWordFrequency",
		return = Replace[iGetFromCache[{input}], _Missing|_$Failed|$Failed :> None];
	];
	If[return === None,
		res = ReleaseHold[Internal`MWACompute[type, input, "MessageHead" -> WordFrequencyData]];
		If[! MatchQ[res, {___Rule}],
			Message[head::trfet];
			Return[res];
		];
		{res, msg} = Replace[{"Result", "Messages"}, res, {1}];
		EntityFramework`Private`issueMWAMessages[msg];
		return = Replace[res, "Result" -> {}];
		If[!MatchQ[return, _Missing|$Failed|_$Failed|None],
			iAddToCache[{input}, return];
		];
	];
	return
];

(* WordFrequency *)

$ValidNgramPOS = {"NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "NUM", "CONJ", "PRT", ".", "X", "noTag"};
$POStoGrammaticalUnit = Dispatch@{"NOUN" -> Entity["GrammaticalUnit", "Noun"], 
 "VERB" -> Entity["GrammaticalUnit", "Verb"], 
 "ADJ" -> Entity["GrammaticalUnit", "Adjective"], 
 "ADV" -> Entity["GrammaticalUnit", "Adverb"], 
 "PRON" -> Entity["GrammaticalUnit", "Pronoun"], 
 "DET" -> Entity["GrammaticalUnit", "Determiner"], 
 "ADP" -> Entity["GrammaticalUnit", "Adposition"], 
 "NUM" -> Entity["GrammaticalUnit", "Numeral"], 
 "CONJ" -> Entity["GrammaticalUnit", "Conjunction"], 
 "PRT" -> Entity["GrammaticalUnit", "Particle"], 
 "." -> Entity["GrammaticalUnit", "Punctuation"], 
 "X" -> None, 
 "noTag" -> Automatic};

iTagGrammaticalUnit[e:Entity["GrammaticalUnit", name_]] := With[{tag = e["Tag"]},
	If[MatchQ[tag, _String|Missing["NotAvailable"]|Missing["UnknownEntity", ___]],
		iTagGrammaticalUnit[e] = tag
		,
		tag
	]
];

iReverseTagGrammaticalUnit[s_String -> pos_String] := With[{r = Replace[pos, $POStoGrammaticalUnit]},
	Switch[r,
		_Entity, TextElement[s, Association["GrammaticalUnit"->r]],
		None, TextElement[s, Association["GrammaticalUnit"->Missing["NotAvailable"]]],
		_, s
	]
];

iNormalizeGrammaticalUnit[s_] := Missing["InvalidPOS"];
(iNormalizeGrammaticalUnit[#] = #)&/@$ValidNgramPOS;
iNormalizeGrammaticalUnit[s_String] := First[Intersection[{s}, $ValidNgramPOS], Missing["InvalidPOS"]];
iNormalizeGrammaticalUnit[e:Entity["GrammaticalUnit", name_]] := With[{simpletag = Replace[iTagGrammaticalUnit[e], s:Except[_String?StringQ] :> name]}, iNormalizeGrammaticalUnit[ToUpperCase@simpletag]];
iNormalizeGrammaticalUnit["PARTICLE"] = "PRT"; 
iNormalizeGrammaticalUnit["PUNCTUATION"] = ".";
iNormalizeGrammaticalUnit["PUNCT"] = ".";

iformatArg2[Automatic|Null] := {False, False, False};

iformatArg2[data_String | data_List] := Module[{tmp},
	tmp = Flatten[{data} /. {"PartOfSpeechVariants"|"PartsOfSpeechVariants" -> "POSVariants"}];
	(*Both Total and TimeSeries given, should it issue a warning? *)
	(*If[Length[Complement[{"TimeSeries", "Total"}, tmp]] === 0,
		(* TODO: Issue a message and continue. *)
		Null;
	];*)
	If[Length[Complement[tmp, {"TimeSeries", "CaseVariants", "POSVariants", "Total"}]] > 0,
		Message[WordFrequencyData::arg2v]; 
		Throw[$Failed, WordFrequencyTag]
	];
	tmp = {"TimeSeries", "CaseVariants", "POSVariants"} /. Thread[tmp -> True] /. {"TimeSeries" -> False, "CaseVariants" -> False, "POSVariants" -> False};
	Return@tmp;
];

totalCountsByLanguage := initWordFrequencyDataTotalCounts[];

initWordFrequencyDataTotalCounts[] := Module[{data},
	data = iWolframAlphaAPICompute[WordFrequencyData, "MWAWordFrequency", {{"TotalCounts", True}}];
	If[!MatchQ[data, {__}],
		Return[$Failed];
	];
	totalCountsByLanguage = Map[Association[Rule @@@ #] &, GroupBy[data, First -> (#[[2 ;;]] &)]];
	Return[totalCountsByLanguage];
]

$WordFrequencyValidLanguages := If[totalCountsByLanguage =!= $Failed, $WordFrequencyValidLanguages = Keys[totalCountsByLanguage], {}];

validLanguageWordFrequency[lang_] := $Failed;
validLanguageWordFrequency[Automatic] := "english";
validLanguageWordFrequency[lang_String?StringQ] := Replace[Intersection[{lang//ToLowerCase}, $WordFrequencyValidLanguages], {{s_String} :> s, _ -> $Failed}]
validLanguageWordFrequency[e:Entity["Language", lang_]] := validLanguageWordFrequency[Internal`LanguageCanonicalName[e]]; 

interpretResult[sourcecorpus_, date_, timeseries_][result_]:= Which[
	MatchQ[result, _Missing],
		result,
	MatchQ[result, $Failed],
		Throw[$Failed, WordFrequencyTag],
	True,
		If[MatchQ[date, _Integer],
			result / totalCountsByLanguage[sourcecorpus][date]
			,
			If[!MatchQ[result, _TemporalData],
				(*TODO: includ an error message, invalid returned data. *)
				Throw[$Failed, WordFrequencyTag];
			];
			If[timeseries ,
				 result / TimeSeries[ 
	 						result["DatePath"] /. {x_DateObject, y_} :> 
	 										{x,  totalCountsByLanguage[sourcecorpus][ x /. DateObject -> (List[#][[1, 1]] &)]}
	 						]
	 					,
			 	Divide@@Total/@Transpose[
		 					result["DatePath"] /. {x_DateObject, y_} :> 
	 										{y , totalCountsByLanguage[sourcecorpus][ x /. DateObject -> (List[#][[1, 1]] &)]}
									]		
				]
		]
	]

validWordFrequencyDataInputElementQ[_] := False;
validWordFrequencyDataInputElementQ[_String?StringQ] := True;
validWordFrequencyDataInputElementQ[HoldPattern[_TextElement]] := True;
validWordFrequencyDataInputElementQ[_String?StringQ -> _String?StringQ] := True;
validWordFrequencyDataInputElementQ[_String?StringQ -> Entity["GrammaticalUnit", __]] := True;
validWordFrequencyDataInputElementQ[Verbatim[Alternatives][alternatives__?validWordFrequencyDataInputElementQ]] := True;
validWordFrequencyDataInputQ[_] := False;
validWordFrequencyDataInputQ[_?validWordFrequencyDataInputElementQ] := True;
validWordFrequencyDataInputQ[{__?validWordFrequencyDataInputElementQ}] := True;
validWordFrequencyDataInputQ[{}] := True;

Options[WordFrequencyData] = {IgnoreCase -> False,  Language -> "English"};
WordFrequencyData[args___] := With[{finalresult = Catch[Module[{arg, opts, tmp, caseoption, date, sourcecorpus, timeseries, casevariants, posvariants, result, arg1pre, arg1, simpleoutput},
		{arg, opts} = Check[System`Private`Arguments[WordFrequencyData[args], {1,3}], Throw[$Failed, WordFrequencyTag]];
		
		arg[[1]] = ReplaceAll[arg[[1]], Entity["Word", word_String?StringQ]:> word];
		
		If[!validWordFrequencyDataInputQ[arg[[1]]],
			Message[WordFrequencyData::arg1];
			Throw[$Failed, WordFrequencyTag]
		];
		
		If[Length@arg >= 2 ,
			If[!MatchQ[arg[[2]], _String|{___String}|Automatic|Null],
				Message[WordFrequencyData::arg2];
				Throw[$Failed, WordFrequencyTag]
			,
				tmp = arg[[2]] 
			]
			,
			tmp = {}
		];
	(*	tmp = iformatArg2[tmp]; (*tmp = {timerseries, casevariants, posvariants}*)*)
		{timeseries, casevariants, posvariants} = iformatArg2[tmp];
		
		If[Length[arg] >= 3,
			date = arg[[3]] /. d_DateObject :> Quiet[Check[First[DateList[d]], d]] /. r_Real :> Round[r]
			,
			date = Automatic
		];

		With[{variable = #1, name = #2},
			variable = Check[OptionValue[WordFrequencyData, opts, name], Throw[$Failed, WordFrequencyTag]];
			If[!MatchQ[variable, True|False],
				Message[WordFrequencyData::opttf, name, variable];
				Throw[$Failed, WordFrequencyTag];
			];
		] & @@@ {{caseoption, IgnoreCase}};
		
		sourcecorpus = Check[validLanguageWordFrequency[OptionValue[WordFrequencyData, opts, Language]], Throw[$Failed, WordFrequencyTag]];
		If[!MatchQ[sourcecorpus, Automatic|_String],
			Message[WordFrequencyData::optlang];
			Throw[$Failed, WordFrequencyTag];
		];
		
		If[!MatchQ[date, All|Automatic|_Integer|{_Integer,_Integer}|{{__Integer}}|{{_Integer}..}|{_Integer, Infinity}|{-Infinity, _Integer}|{-Infinity, Infinity}],
			Message[WordFrequencyData::argdate];
			Throw[$Failed, WordFrequencyTag];
		];
		
		If[arg[[1]]==={},
			Return[Association[]];
		];
		
		arg1pre = arg[[1]] /. {(s_String->pos_) :> (s -> iNormalizeGrammaticalUnit[Replace[pos, ps_String :> Entity["GrammaticalUnit", ps]]]), TextElement[s_String] :> s, TextElement[s_String, a_?AssociationQ] :> (s -> iNormalizeGrammaticalUnit[a["GrammaticalUnit"]])} /. (s_String -> _Missing) :> s;
		arg1 = DeleteDuplicates@Flatten[{arg1pre /. Alternatives -> List}];
		If[MatchQ[arg1, _TextElement|{___,_TextElement|(_->Except[_String])|(Except[_String]->_),___}],
			Message[WordFrequencyData::invtext];
			Throw[$Failed, WordFrequencyTag];
		];
		If[!MatchQ[arg1, {(_String|Rule[_String, _String])..}],
			Message[WordFrequencyData::arg1];
			Throw[$Failed, WordFrequencyTag];
		];
		If[Head[arg[[1]]] =!= List && !casevariants && !posvariants,
			simpleoutput = True;
			arg1 = Replace[arg1, {one_} :> one];
		];

		result = iWolframAlphaAPICompute[WordFrequencyData,
			"MWAWordFrequency", 
			{arg1, "Year" -> date, IgnoreCase -> caseoption, "IncludeCaseVariants" -> casevariants, "IncludePOSVariants" -> posvariants, "SourceCorpus"-> sourcecorpus }
		];
		
		If[result === $Failed, Throw[$Failed, WordFrequencyTag]];
		
		If[simpleoutput && AssociationQ[result],
			result = joinAPIResults[Values[result]];
		];
		
		If[AssociationQ[result] && !casevariants && !posvariants,
			result = Association@Table[
					arg[[1]][[i]] -> joinAPIResults[Values[KeySelect[result, compareWordFrequencyDataKeys[caseoption][arg1pre[[i]], #]&]]]
				, {i, Length[arg[[1]]]}];
		,
			If[AssociationQ[result],
				result = KeyMap[ReplaceAll[#, (te:(_String -> _String)) :> iReverseTagGrammaticalUnit[te] ]&, result];
			];
		];

		If[AssociationQ[result],
			interpretResult[sourcecorpus, date, timeseries] /@ result
			,
			interpretResult[sourcecorpus, date, timeseries][result]
		]
	], WordFrequencyTag]},
	finalresult /; finalresult =!= $Failed
];

joinAPIResults[values:{}] := Missing["NotAvailable"];
joinAPIResults[values:{__?NumericQ}] := Total[values];
joinAPIResults[values:{TemporalData[___]..}] := TimeSeries@GroupBy[Join @@ (#["DatePath"] & /@ values), First -> Last, Total];

compareWordFrequencyDataKeys[True][w1_String, w2_String] := ToLowerCase[w1] === ToLowerCase[w2];
compareWordFrequencyDataKeys[False][w1_String, w2_String] := w1 === w2;
compareWordFrequencyDataKeys[True][w1_Rule, w2_Rule] := ToLowerCase[w1[[1]]] === ToLowerCase[w2[[1]]] && w1[[2]] === w2[[2]];
compareWordFrequencyDataKeys[False][w1_Rule, w2_Rule] := w1 === w2;
compareWordFrequencyDataKeys[ignorecase_][Verbatim[Alternatives][alt1_], any_] := compareWordFrequencyDataKeys[ignorecase][alt1, any];
compareWordFrequencyDataKeys[ignorecase_][Verbatim[Alternatives][alt1__], any_] := AnyTrue[{alt1}, compareWordFrequencyDataKeys[ignorecase][#, any]&];
compareWordFrequencyDataKeys[False][any1_, any2_] := SameQ[any1, any2];
compareWordFrequencyDataKeys[True][any1_, any2_] := SameQ[any1 /. s_String:>ToLowerCase[s], any2 /. s_String:>ToLowerCase[s]];

(* WordFrequency for local strings *)
validWordFrequencyInputElementQ[_] := False;
validWordFrequencyInputElementQ[{}] := True;
validWordFrequencyInputElementQ[_String?StringQ] := True;
validWordFrequencyInputElementQ[HoldPattern[_TextElement]] := True;
validWordFrequencyInputElementQ[HoldPattern[x_Entity]] := MatchQ[ x , Entity["Word", _String?StringQ]];
validWordFrequencyInputElementQ[{(_String?StringQ)..}] := True;
validWordFrequencyInputElementQ[Verbatim[Alternatives][alternatives__?validWordFrequencyInputElementQ]] := True;
validWordFrequencyInputQ[_] := False;
validWordFrequencyInputQ[_?validWordFrequencyInputElementQ] := True;
validWordFrequencyInputQ[{__?validWordFrequencyInputElementQ}] := True;
validWordFrequencyInputQ[{}] := True;

Options[WordFrequency] = {IgnoreCase -> False};
WordFrequency[args___] := With[{result = Catch[Module[{arg, opts, caseoption, casevariants, a1, arg2},
		{arg, opts} = Check[System`Private`Arguments[WordFrequency[args], {2,3}], Throw[$Failed, WordFrequencyTag]];
		
		If[!validWordFrequencyInputQ[arg[[1]]],
			Message[WordFrequency::arg1];
			Throw[$Failed, WordFrequencyTag]
		];
					
		If[ Length@arg >= 3 , 
			If[!MatchQ[arg[[3]], "CaseVariants" | {"CaseVariants"}],
				Message[WordFrequency::arg3];
				Throw[$Failed, WordFrequencyTag],
				casevariants = True;
			],
			casevariants = False;
		];
		
		With[{variable = #1, name = #2},
			variable = Check[OptionValue[WordFrequency, opts, name], Throw[$Failed, WordFrequencyTag]];
			If[!MatchQ[variable, True|False],
				Message[WordFrequency::opttf, name, variable];
				Throw[$Failed, WordFrequencyTag];
			];
		] & @@ {caseoption, IgnoreCase}; 
		
		arg[[2]] = ReplaceAll[arg[[2]], Entity["Word", word_String?StringQ]:> word];

		arg2 = normalizeInputSearch[caseoption || casevariants][arg[[2]]];
		
		Switch[arg[[1]],
			{},
				{}
				,
			(_String?StringQ|_TextElement),
				iLocalWordFrequency[arg[[1]], arg2, casevariants, opts]
				,
			{(_String?StringQ|_TextElement)..},
				Table[iLocalWordFrequency[a1, arg2, casevariants, opts], {a1, arg[[1]]}]
				,
			_,
				Message[WordFrequency::arg1];
				Throw[$Failed, WordFrequencyTag]
		]
	], WordFrequencyTag]},
	N@result /; result =!= $Failed
];

wordSequence::usage = "wordSequence[original, {sequence__}]local wrapper for words sequences";

compareWordSequence[w1_wordSequence, w2_wordSequence] := SameQ[w1[[2]], w2[[2]]];
compareWordSequence[Verbatim[Alternatives][alt1_], any_] := compareWordSequence[alt1, any];
compareWordSequence[Verbatim[Alternatives][alt1__], Verbatim[Alternatives][alt2__]] := SameQ[Sort[{alt1}], Sort[{alt2}]];
compareWordSequence[any_, Verbatim[Alternatives][alt2_]] := compareWordSequence[any, alt2];
compareWordSequence[any1_, any2_] := SameQ[any1, any2];

normalizeInputSearch[False][HoldPattern[s_TextElement]] := wordSequence[s,{s}];
normalizeInputSearch[True][HoldPattern[s_TextElement]] := s  /. x:TextElement[str_, gr_] :> wordSequence[ x,{TextElement[ToLowerCase@str, gr]} ];
normalizeInputSearch[True][s_String?StringQ] := wordSequence[s, ToLowerCase@System`TextWords[s]];
normalizeInputSearch[False][s_String?StringQ] := wordSequence[s, System`TextWords[s]];
normalizeInputSearch[ig_][list_List] := DeleteDuplicates[normalizeInputSearch[ig]/@Flatten[list], compareWordSequence];
normalizeInputSearch[ig_][Verbatim[Alternatives][alt__]] := normalizeInputSearch[ig]/@Alternatives@@Sort[Flatten[{alt}]];
normalizeInputSearch[_][_] := (Message[WordFrequency::arg2]; Throw[$Failed, WordFrequencyTag]);

getNGramDegree[_String] := 1;
getNGramDegree[Verbatim[Alternatives][alt__]] := getNGramDegree/@{alt};
getNGramDegree[multi_wordSequence] := Length[multi[[2]]];
getNGramDegree[group_List] := Union[Flatten[getNGramDegree/@group]];

compareWithWordSequence[True][s_String, ws_] := MatchQ[ToLowerCase@{s}, ws];
compareWithWordSequence[True][s:{__String}, ws_] := MatchQ[ToLowerCase@s, ws];
compareWithWordSequence[True][HoldPattern[s:{_TextElement}], ws_] := MatchQ[s /. TextElement[str_String, gr_] :> TextElement[ToLowerCase@str, gr] , ws];
compareWithWordSequence[False][s_String, ws_] := MatchQ[{s}, ws];
compareWithWordSequence[False][s:{__String}, ws_] := MatchQ[s, ws];
compareWithWordSequence[False][HoldPattern[s:{_TextElement}], ws_] := MatchQ[s, ws];

totalByGroups[list_] := Total[Reverse[List@@@Normal[GroupBy[list, Last -> First, Total]], 2]]
totalByGroupsNoMix[list_] := With[{grps = GroupBy[list, Last -> First, Total]}, If[Length[grps]>1, Message[WordFrequency::nongram]; Throw[$Failed, WordFrequencyTag], Total[Reverse[List@@@Normal[grps], 2]]]]

Options[iLocalWordFrequency] = Options[WordFrequency];

iLocalWordFrequency[inputString_String, {}, caseVariants_, opts:OptionsPattern[]] := Association[];

iLocalWordFrequency[inputString_String, inputSearch_, caseVariants_, opts:OptionsPattern[]] := Module[
	{ignorecase, allwords, outputTotalQ, neededAllW, iInputSearch, len, allalts, n, input, simpleInput, result, allwordsall, originalInput, sample, textwords},
	outputTotalQ = !ListQ[inputSearch]&&!caseVariants;
	ignorecase = Check[OptionValue[WordFrequency, {opts}, IgnoreCase], Throw[$Failed, WordFrequencyTag]] && !caseVariants;
	iInputSearch = Replace[inputSearch, i:Except[_List]:>{i}];
	neededAllW = getNGramDegree[iInputSearch];
	allalts = Map[Alternatives@@#&, GroupBy[DeleteDuplicates[Flatten[iInputSearch /. Alternatives->List]]/.wordSequence[_, seq_]:>seq, Length]];
	allwordsall = Association[];

	If[ignorecase,
		textwords = System`TextWords[inputString//ToLowerCase];
		,
		textwords = System`TextWords[inputString];
	];

	Table[
		allwords = Counts[Partition[textwords, n, 1]]; 
		len = Total[allwords];
		allwordsall = Union[allwordsall, Map[{#, len}&, KeySelect[allwords, compareWithWordSequence[caseVariants][#, allalts[n]]&]]];
	, {n, neededAllW}];
	Clear[allwords];
	
	result = DeleteDuplicates[Table[
		simpleInput = input /. wordSequence[_, seq_] :> seq;
		originalInput = (input /. (wordSequence[original_, _] :> original));
		sample = KeySelect[allwordsall, compareWithWordSequence[caseVariants][#, simpleInput]&];
		If[!caseVariants, 
			originalInput -> Replace[totalByGroupsNoMix[Values@sample], {{_, 0} -> 0, {c_, t_} :> N[c/t]}]
			,
			KeyValueMap[ With[{k = Replace[#1, {one_} :> one]}, k -> Replace[#2, {{_, 0} -> 0, {c_, t_} :> N[c/t]}]]&, sample]
		]
	, {input, iInputSearch}]];
	
	If[TrueQ[outputTotalQ],
		result = result[[1,2]];
		,
		result = Association@@result;
	];
	
	result
];

iLocalWordFrequency[HoldPattern[inputString_TextElement], inputSearch_, caseVariants_, opts:OptionsPattern[]] := Module[
	{ignorecase, allwords, outputTotalQ, neededAllW, iInputSearch, len, allalts, n, input, simpleInput, result, allwordsall, originalInput, sample, textwords},
	outputTotalQ = !ListQ[inputSearch]&&!caseVariants;
	ignorecase = Check[OptionValue[WordFrequency, {opts}, IgnoreCase], Throw[$Failed, WordFrequencyTag]] && !caseVariants;
	iInputSearch = Replace[inputSearch, i:Except[_List]:>{i}];
	neededAllW = getNGramDegree[iInputSearch];
	allalts = Map[Alternatives@@#&, GroupBy[DeleteDuplicates[Flatten[iInputSearch /. Alternatives->List]]/.wordSequence[_, seq_]:>seq, Length]];

	allwordsall = Association[];
	textwords = Cases[inputString, TextElement[_String, ___], Infinity]; 
	If[ignorecase,
		textwords = textwords /. TextElement[str_String, gr_] :> TextElement[ToLowerCase@str, gr] ;
		,
		textwords = textwords;
	];

	Table[
		allwords = Counts[Partition[textwords, n, 1]];
		len = Total[allwords];
		allwordsall = Union[allwordsall, Map[{#, len}&, KeySelect[allwords, compareWithWordSequence[caseVariants][#, allalts[n]]&]]];
	, {n, neededAllW}];
	Clear[allwords];
	
	result = DeleteDuplicates[Table[
		simpleInput = input /. wordSequence[_, seq_] :> seq;
		originalInput = (input /. (wordSequence[original_, _] :> original));
		sample = KeySelect[allwordsall, compareWithWordSequence[caseVariants][#, simpleInput]&];
		If[!caseVariants, 
			originalInput -> Replace[totalByGroupsNoMix[Values@sample], {{_, 0} -> 0, {c_, t_} :> N[c/t]}]
			,
			KeyValueMap[ With[{k = Replace[#1, {one_} :> one]}, k -> Replace[#2, {{_, 0} -> 0, {c_, t_} :> N[c/t]}]]&, sample]
		]
	, {input, iInputSearch}]];
	
	If[TrueQ[outputTotalQ],
		result = result[[1,2]];
		,
		result = Association@@result;
	];
	
	result
];
SetAttributes[{System`WordFrequency, System`WordFrequencyData},
   {ReadProtected, Protected}
];


End[];
