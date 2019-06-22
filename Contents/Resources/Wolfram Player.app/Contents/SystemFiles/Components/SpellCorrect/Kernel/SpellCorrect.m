Begin["System`SpellCorrectDump`Private`"];

$failTag = "SpellCorrectCatchThrowFailTag";
$unevaluatedTag = "SpellCorrectUnevaluatedTag";
(*failure function if something goes wrong during loading; issues message and ensure the context is exited before throwing*)
loadingBailout[] := CompoundExpression[
	Message[SpellingCorrectionList::norec],
	End[],
	Throw[$Failed, "SpellCorrectLoadingFailed"]
]

$spellCorrectLibrary = FindLibrary["libspellcor"];

If[!FileExistsQ[$spellCorrectLibrary],
	loadingBailout[]
]

(*Library functions to load during initialization; fail if they cannot be loaded...*)
addDicts = LibraryFunctionLoad[$spellCorrectLibrary, "setSpellCorrectDictionary", {"UTF8String", "UTF8String"}, "Void"];
removeDicts = LibraryFunctionLoad[$spellCorrectLibrary, "unsetSpellCorrectDictionary", {}, "Void"];
dictionaryEncoding = LibraryFunctionLoad[$spellCorrectLibrary, "dictionaryCharacterEncoding", {}, "UTF8String"];
correctSpellingQ = LibraryFunctionLoad[$spellCorrectLibrary, "correctSpellingQ", {"UTF8String"}, Integer];
checkSpelling = LibraryFunctionLoad[$spellCorrectLibrary, "spellCorrectString", {"UTF8String"}, {Integer, 2}];

If[!MatchQ[{addDicts, removeDicts, dictionaryEncoding, correctSpellingQ, checkSpelling}, {_LibraryFunction..}],
	loadingBailout[]
]

(*basic utility functions*)
fcc[{data__, 0 ...}] := Quiet[Check[
	FromCharacterCode[{data}, $characterEncoding], Throw[$unevaluatedTag, $failTag]
]]

Options[SpellingCorrectionList] = {Language :> $Language}

Needs["SpellingData`"];

languageFileName	=	SpellingData`SpellingDataDicts[];
$availableLanguages = Keys[languageFileName];

$initializedLanguage = None;
$characterEncoding = None;

makeDictionaryFileNames[head_, lang_] :=
    Module[{fn = languageFileName[lang], spellingDictionaryResources, resourcesSortedByVersionNumber, dictionaryPaths, result, aff, dic},
		If[!StringQ[fn], Throw[$unevaluatedTag, $failTag]];
		(* spellingDictionaryResources will look like: {{_Paclet, {"/path/to/SpellingDictionary/dir"..}}...} *)
	    spellingDictionaryResources = PacletManager`PacletResources["SpellingDictionary"];
	    (* Sort in order of decreasing version number across all paclets regardless of name. *)
	    resourcesSortedByVersionNumber = Sort[spellingDictionaryResources, !PacletManager`PacletNewerQ[First[#1], First[#2]] &];
	    (* Extract just the paths to the SpellingDictionaries dirs. *)
	    dictionaryPaths = Flatten[Last /@ resourcesSortedByVersionNumber];
	    (* Walk through the list of dictionary paths and find the first one that includes files with the desired names. *)
	    result = Scan[
	                Function[dictDir,
	                    aff = FileNameJoin[{dictDir, StringJoin[fn,".aff"]}];
	                    dic = FileNameJoin[{dictDir, StringJoin[fn, ".dic"]}];
	                    If[And[FileExistsQ[aff], FileExistsQ[dic]],
	                        Return[{aff, dic}]
	                    ]
	                ],
	                dictionaryPaths                    
	            ];
		If[MatchQ[result, {_String, _String}],
		    (* {aff, dic} pair *)
			result,
	    (* else *)
			Message[head::nodict, lang]; Throw[$unevaluatedTag, $failTag]
		]
    ]

SpellingCorrectionList[args___] := With[{res = Catch[iSpellingCorrectionList[languageAndArguments[SpellingCorrectionList, args]], $failTag]},
	res /; res =!= $unevaluatedTag
]

toLanguage[s_String] := s
toLanguage[HoldPattern[e:Entity["Language", s_String]]] := Internal`LanguageCanonicalName[e]
toLanguage[other_] := other

languageAndArguments[head_, args___, opts:OptionsPattern[]] := With[{
	lang = toLanguage[Check[OptionValue[head, {opts}, Language], $unevaluatedTag]],
	arguments = If[head === DictionaryWordQ, {OptionValue[head, {opts}, IgnoreCase], args}, {args}]
	},
	If[lang === $unevaluatedTag, Throw[$unevaluatedTag, $failTag]];
	If[Not[MemberQ[$availableLanguages, lang]],
		Message[head::nodict, lang];Throw[$unevaluatedTag, $failTag]
	];
	setUpDictionaries[head, lang];
	verifyArguments[head, arguments]
]

verifyArguments[DictionaryWordQ, {ic:(True|False), s_String}] := {ic, s}
verifyArguments[DictionaryWordQ, {other_, s_String}] := CompoundExpression[
	Message[DictionaryWordQ::opttf, IgnoreCase, other],
	Throw[$unevaluatedTag, $failTag]
]
verifyArguments[DictionaryWordQ, {_, arg_}] := CompoundExpression[
	Message[DictionaryWordQ::sstr, arg],
	Throw[$unevaluatedTag, $failTag]
]
verifyArguments[SpellingCorrectionList, {s_String}] := s
verifyArguments[head_, {arg_}] := CompoundExpression[
	Message[head::sstr, arg],
	Throw[$unevaluatedTag, $failTag]
]
verifyArguments[head_, {args___}] := CompoundExpression[
	System`Private`Arguments[head[args], {1, 1}],
	Throw[$unevaluatedTag, $failTag]
]

(*only bother trying to unload the dictionary if one was previously loaded*)
unloadDictionariesIfNeeded[lang_String] /; StringQ[$initializedLanguage] := removeDicts[]

(*only bother setting up dictionaries if the dictionary in use hasn't yet been initialized*)
setUpDictionaries[head_, lang_String] /; UnsameQ[$initializedLanguage, lang] := Module[{aff, dic},
	{aff, dic} = makeDictionaryFileNames[head, lang];
	unloadDictionariesIfNeeded[lang];
	addDicts[aff, dic];
	$characterEncoding = getDictionaryEncoding[head, lang];
	$initializedLanguage = lang;
]

getDictionaryEncoding[head_, lang_] := With[{res = dictionaryEncoding[]},
	If[MemberQ[$CharacterEncodings, res],
		res,
		Message[head::denc, lang, res];
		Throw[$unevaluatedTag, $failTag]
	]
]

iSpellingCorrectionList[word_String] := iSpellingCorrectionList[{word}]	
iSpellingCorrectionList[{word_String}] := If[
	correctSpellingQ[word] === 1, 
	{word},
	fcc /@ checkSpelling[word]
]
iSpellingCorrectionList[l_List] := Map[iSpellingCorrectionList, l]

  
iSpellingCorrectionList[___] := Throw[$unevaluatedTag, $failTag]

Options[DictionaryWordQ] = {IgnoreCase -> False, Language :> $Language}

DictionaryWordQ[args___] := With[{res = Catch[iDictionaryWordQ[languageAndArguments[DictionaryWordQ, args]], $failTag]},
	res /; res =!= $unevaluatedTag
]

iDictionaryWordQ[{False, word_String}] := spelledCorrectlyQ[word]
iDictionaryWordQ[{True, word_String}] := checkAllSpellingForms[word]
iDictionaryWordQ[___] := Throw[$unevaluatedTag, $failTag]

spelledCorrectlyQ[word_String] := And[SameQ[correctSpellingQ[word], 1], Not[StringMatchQ[word, ___~~"."~~___]]]
spelledCorrectlyQ[_] := False

checkAllSpellingForms[word_] := TrueQ[Or[
	spelledCorrectlyQ[word],
	spelledCorrectlyQ[ToLowerCase[word]],
	With[{w = StringJoin[ToUpperCase[StringTake[word,1]], ToLowerCase[StringDrop[word,1]]]}, 
		spelledCorrectlyQ[w]
	]
]]
	

SetAttributes[{DictionaryWordQ, SpellingCorrectionList}, {Protected, ReadProtected}]


End[];