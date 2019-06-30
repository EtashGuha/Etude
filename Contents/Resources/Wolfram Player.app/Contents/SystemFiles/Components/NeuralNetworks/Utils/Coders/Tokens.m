Package["NeuralNetworks`"]


PackageScope["$DummyWordLists"]

SetUsage @ "
$DummyWordLists can be set to a dictionary or function that provides word lists for testing purposes."


PackageScope["GetLanguageTokenEncoder"]

$DummyWordLists = None;

GetLanguageTokenEncoder[lang_, ic_] /; $DummyWordLists =!= None :=
	ToTokenEncodingData[$DummyWordLists[lang], ic];

GetLanguageTokenEncoder[lang_, ic_] := Scope[
	list = loadWordList[lang];
	If[StringVectorQ[list], 
		GetLanguageTokenEncoder[lang, ic] /; $DummyWordLists === None = 
			ToTokenEncodingData[list, ic],
		$Failed
	]
];

loadWordList["English"] := Scope[
	stop = Union[WordList["Stopwords", Language -> "English"],
		(* The following ones (single characters) were removed in 12.0.
			We add them again here for backward compatibility
		*)
		CharacterRange[48, 57], (* 0, ..., 9 *)
		CharacterRange[65, 90], (* A, ..., Z *)
		CharacterRange[97, 122] (* a, ..., z *)
	];
	common = WordList["CommonWords", Language -> "English"];
	If[StringVectorQ[stop] && StringVectorQ[common], 
		Union[stop, common],
		$Failed
	]
];

loadWordList[lang_] := WordList[Language -> lang];

$validLanguages = {"English", "Spanish", "French", "German"};


PackageScope["ToTokenEncodingData"]
PackageExport["TokenEncodingData"]

$TokenEncodingDataCache = <||>;

ToTokenEncodingData[ta_TokenEncodingData, _] := ta;

ToTokenEncodingData[Automatic, ic_] := ToTokenEncodingData["English", ic];

ToTokenEncodingData[lang_String, ic_] /; MemberQ[$validLanguages, lang] := 
	Replace[GetLanguageTokenEncoder[lang, ic], {
		ta_TokenEncodingData :> ta,
		_ :> FailCoder["Could not obtain a word list for ``. Please check your internet connection.", lang]
	}];	

(* we delete duplicates, and so we have to apply case folding before this
happens of course *)
ToTokenEncodingData[tokens_List ? StringVectorQ, True] := 
	ToTokenEncodingData[ToLowerCase @ tokens, False];

ToTokenEncodingData[tokens_List ? StringVectorQ, False] := Scope[
	uniqueTokens= DeleteDuplicates @ tokens;
	TokenEncodingData[
		1,
		If[ByteCount[uniqueTokens] < 256, uniqueTokens, CompressToByteArray[uniqueTokens]], 
		Length[uniqueTokens]
	]
];

ToTokenEncodingData[tokens_, _] := FailCoder[
	"Tokens should be a list of strings or one of ``.", 
	$validLanguages];

_ToTokenEncodingData := $Unreachable;


TokenEncodingData /: MakeBoxes[TokenEncodingData[1, data_, len_], StandardForm] :=
	ToBoxes @ Style[Skeleton[Row[{len, " strings"}]], ShowStringCharacters -> False];

TokenEncodingData /: Normal[TokenEncodingData[1, data_, _]] := 
	If[ByteArrayQ[data], UncompressFromByteArray[data], data];

TokenEncodingData /: Length[TokenEncodingData[1, _, len_]] := len;


PackageScope["TokenEncode"]
PackageScope["MakeTokenDictionary"]

MakeTokenDictionary[tokens_List] := 
	Association @ MapIndexed[
		Hash[#1] -> First[#2]&, 
		tokens
	];

ClearAll[TokenEncode]
TokenEncode[dict_, casefunc_, patt_][input_] := Scope[
	If[!StringQ[input], EncodeFail["input was not a string"]];
	inputTokens = TokenizeIntoWords[input, patt];
	mapTokens[dict, casefunc][inputTokens]
];

TokenEncode[dict_, casefunc_, patt_|None][HoldPattern @ inputTokens_TextElement] := Scope[
	input = Flatten @ ReplaceRepeated[inputTokens, HoldPattern[t_TextElement] :> First[t, $Failed]];
	If[!StringVectorQ[input],
		EncodeFail["input TextElement did not contain a list of strings or well-formed TextElement"];
	];
	mapTokens[dict, casefunc][input]
];

TokenEncode[dict_, casefunc_, None][inputTokens_] := Scope[
	If[!StringVectorQ[inputTokens], EncodeFail["input was not a list of strings"]];
	mapTokens[dict, casefunc][inputTokens]
];

mapTokens[dict_, casefunc_][input_] :=
	Lookup[dict, Map[Hash, casefunc @ input], Length[dict]+1];

(* SplitPattern option accepts the same argument type as StringSplit: StringMatchQ and/or rule(s) of conversion *)
PackageScope["StringPatternOrRulesQ"]
StringPatternOrRulesQ[e_] := StringPatternQ @ Cases[ToList[e], Except[(_?GeneralUtilities`StringPatternQ -> _String) | (_?GeneralUtilities`StringPatternQ :> _)]];

PackageScope["TokenizeIntoWords"]
PackageScope["$DefaultTokenSplitPattern"]

(* Note: using {WordBoundary, x:PunctuationCharacter :> x} as default gives a bad display in the Notebook *)
$DefaultTokenSplitPattern = WordBoundary; 
TokenizeIntoWords[input_, patt_] :=
	DeleteCases[StringTrim @ StringSplit[input, patt], ""];


PackageScope["ParseTokensSpec"]
PackageScope["ParseCharactersSpec"]

ParseTokensSpec[spec_] := If[spec === {},
	FailCoder["the list of tokens cannot be empty"],
	ToTokenEncodingData[spec, TrueQ @ PeekOption["IgnoreCase"]]
];
ParseCharactersSpec[spec_] := If[MatchQ[spec, {}|""],
	FailCoder["the list of characters cannot be empty"],
	MXNetLink`ToCharacterEncodingData[spec, spec, TrueQ @ PeekOption["IgnoreCase"]]
];