(* ::Package:: *)

BeginPackage["DocumentationSearch`", {"ResourceLocator`"}];

Needs["JLink`"]
Needs["DocumentationSearch`Skeletonizer`"]


FetchReferencedFunctions::usage="Fetches referenced functions from guide pages.";

CreateDocumentationIndex::usage = "CreateDocumentationIndex[dir] skeletonizes and creates an index for the given directory";
docQueryString::usage = "docQueryString[string] transforms the string into a SearchQueryString for SearchDocumentation";
(* Temporarily exposing this so that the button we show with $DebugDocumentationSearchQ works fine *)
$devTextSearchOutput::usage = "Output of devTextSearch for the query at hand";

DocumentationIndexMerger::usage="DocumentationIndexMerge[...] contains the index merger object.";
NewDocumentationIndexMerger::usage="NewDocumentationIndexMerger[dir] creates an index merger that merges index to output dir.";
MergeDocumentationIndex::usage="MergeDocumentationIndex[index] merges a single index directory into the output.";
MergeDocumentationIndexes::usage="MergeDocumentationIndexes[indexes] merges a list of index directories into the output.";
CloseDocumentationIndexMerger::usage="CloseDocumentationIndexMerger[merger] closes the index merger object.";

DocumentationNotebookIndexer::usage="DocumentationNotebookIndexer[...] contains the notebook indexer object.";
NewDocumentationNotebookIndexer::usage="NewDocumentationNotebookIndexer[directory] creates a new NotebookIndexer.";
CloseDocumentationNotebookIndexer::usage="CloseDocumentationNotebookIndexer[indexer] closes the notebook indexer object."

CreateSpellIndex::usage="CreateSpellIndex[indexDir, spellIndexDir] takes the words from indexDir and creates a spelling index at spellIndexDir.";

AddDocumentationNotebook::usage="AddDocumentationNotebook[indexer, notebook] adds a notebook to the index.";
AddDocumentationDirectory::usage="AddDocumentationDirectory[indexer, directory] adds a directory of notebooks to the index.";

SearchDocumentation::usage="SearchDocumentation[criteria] searches the documentation and returns results based on the search criteria.";
SearchDocumentationMetaData::usage="SearchDocumentationMetaData[] returns valid meta data that can be returned from a search."

DirectHitSearch::usage="DirectHitSearch[criteria] searches the documentation and returns a URI if there is a one specific hit for the given query.";

ExportSearchResults::usage="ExportSearchResults[results, format] exports the search results to the specified format.";

$NumberOfExtraPages::usage="$NumberOfExtraPages sets the number of pages on either side of the current page in the searh results navigation";
$NumberOfExtraPages = 3;

DocumentationIndexes::usage="DocumentationIndexes[] returns the index directories of documentation.";
DocumentationSpellIndexes::usage="DocumentationSpellIndexes[] returns the spell index directories of documentation.";
CloseDocumentationIndex::usage="CloseDocumentationIndex[indexDir] closes the indexDir";

InitializeDocumentationSearch::usage="Initialize the new DocumentationSearch functionality using TextSearch. Used by the FE start-up code."

ReindexLegacyPacletsAndSearch::usage = "Determines whether paclets exist that have only old-style documentation indexes, and triggers reindexing and re-search if necessary. Returns True/False to indicate whether reindexing was needed."


$SearchLanguage

`Information`$Version = "DocumentationSearch Version 2.0 (May 2017)";

`Information`$VersionNumber = 2.0;

`Information`$ReleaseNumber = 1;

`Information`$CVSVersion = "$Id$"

Begin["`Private`"];

decimalToString[n_Integer] := ToString[n];
decimalToString[n_Real] := StringTrim[ToString[n], "."];
decimalToString[___] = "";
dropIndexVersionFromPath[s_String] := If[StringMatchQ[FileBaseName[s], (DigitCharacter|".")..], FileNameDrop[s], s]

$UseTextSearchQ::usage = "Use the new TextSearch-based method in DocumentationSearch";
$DebugDocumentationSearchQ::usage = "Show debugging information for the new TextSearch-based version of DocumentationSearch";

ExportSearchResults::format="`1` is not a recognized ExportSearchResults format.";

If[!ValueQ[$SearchLanguage],
    (* Default to using the front end's lang setting, if available, or $Language. *)
    feLang = Quiet[CurrentValue[$FrontEnd, "Language"]];
    searchLang = If[StringQ[feLang], feLang, $Language];
    (* There will be cases where the language is set for something for which there are no main system docs in the layout,
       for example "German", or "English" in a Japanese version that only has a Japanese docs directory. In such cases, we
       want to pick a valid value ("English" in the case of "German", and "Japanese" in the Japanese example). We do this by
       taking the name of the first language subdirectory of the Documentation directory.
    *)
    langDirs = FileNameTake[#, -1]& /@ Select[FileNames["*", FileNameJoin[{$InstallationDirectory, "Documentation"}]], DirectoryQ];
    If[Length[langDirs] > 0 && !MemberQ[langDirs, searchLang],
        searchLang = First[langDirs]
    ];
    $SearchLanguage = searchLang
]
$UseTextSearchQ = ($SearchLanguage === "English");

$PackageDir = DirectoryName[$InputFileName];
$compatibleIndexVersions := If[
	TrueQ[$UseTextSearchQ],
	decimalToString /@ Range[TextSearch`PackageScope`$MinimumSupportedVersion, TextSearch`PackageScope`$CurrentVersion],
	{}
];
$indexVersion := If[
	TrueQ[$UseTextSearchQ],
	decimalToString[TextSearch`PackageScope`$CurrentVersion],
	""
];
$docDir = FileNameJoin[{$InstallationDirectory, "Documentation", $SearchLanguage}];
$textSearchIndexName = "TextSearchIndex"; (* M11.2 *)
$searchIndexName = "SearchIndex"; (* M11.3+ *)
(* horibble hack, hopefully temporary - see bug 343888 *)
$indexName := Which[
	!TrueQ[$UseTextSearchQ],
	"Index", (* M11.1- *)
	$indexVersion =!= "1",
	$searchIndexName,
	True,
	$textSearchIndexName
];

(* Temporary: for third party paclets that have old indices but not a TextSearchIndex, which we generate on the fly and store here. *)
ApplicationDirectoryAdd[FileNameJoin[{$UserBaseDirectory, "DocumentationIndices"}]]; 

DocumentationIndexes[] /; !TrueQ[$UseTextSearchQ] := First /@ ResourcesLocate[FileNameJoin[{"Documentation", ToString[$SearchLanguage], $indexName}]];

If[!TrueQ[$UseTextSearchQ] && !MemberQ[DocumentationIndexes[], 
     FileNameJoin[{$InstallationDirectory, "Documentation", ToString[$SearchLanguage], $indexName}]], 
  ResourceAdd[FileNameJoin[{$InstallationDirectory, "Documentation", ToString[$SearchLanguage], $indexName}], FileNameJoin[{"Documentation", ToString[$SearchLanguage],  $indexName}]]
];

DocumentationIndexes[] /; TrueQ[$UseTextSearchQ] := 
	DeleteDuplicatesBy[FileNameDrop[dropIndexVersionFromPath[#]]&] [
		First /@ Join[
			(* Reverse to get the most recent version of the index first *) 
			If[
				$indexVersion =!= "1", 
				Sequence @@ {
					Reverse@Sort@Flatten[ResourcesLocate[FileNameJoin[{"Documentation", ToString[$SearchLanguage], $indexName, #}]]& /@ $compatibleIndexVersions, 1],
					ResourceAdd[FileNameJoin[{$InstallationDirectory, "Documentation", ToString[$SearchLanguage], $indexName, $indexVersion}], FileNameJoin[{"Documentation", ToString[$SearchLanguage],  $indexName, $indexVersion}]]
				}, 
				{}
			],
			(* Hopefully temporary hack - we should kill the indices named "TextSearchIndex" ASAP - see bug 343888 *)
			If[
				MatchQ[$compatibleIndexVersions, {"1", ___}], 
				Sequence @@ {
					ResourcesLocate[FileNameJoin[{"Documentation", ToString[$SearchLanguage], $textSearchIndexName}]],
					ResourceAdd[FileNameJoin[{$InstallationDirectory, "Documentation", ToString[$SearchLanguage], $textSearchIndexName}], FileNameJoin[{"Documentation", ToString[$SearchLanguage],  $textSearchIndexName}]]
				}, 
				{}
			]
		]
	]

DocumentationSpellIndexes[] := 
  First /@ ResourcesLocate[ToFileName[{"Documentation", ToString[$SearchLanguage]}, "SpellIndex"]]



If[!MemberQ[DocumentationSpellIndexes[], 
     ToFileName[{$InstallationDirectory, "Documentation", "English"}, "SpellIndex"]], 
  ResourceAdd[ToFileName[{$InstallationDirectory, "Documentation", "English"}, "SpellIndex"], ToFileName[{"Documentation", "English"}, "SpellIndex"]]
];


(* This function is called from HelpLookupPacletURI and (as a one-time hit) re-indexes old paclets 
   that have only old-style indexes. It requires a Dynamic-ridden dialog to pop during a preemptive evaluation,
   which in turn requires a ScheduledTask that can do the heavy lifting in the background.
   This function takes all of the current args for HelpLookupPacletURI and will re-call HelpLookupPacletURI with
   those same args once the reindexing is done.
   It return True/False to indicate whether reindexing and re-search were triggered or not.
*)
ReindexLegacyPacletsAndSearch[searchArgs_] :=
    Module[{missingIndices, didReindex},
        didReindex = False;
        If[$firstSearch,
            $firstSearch = False;
            missingIndices = Replace[Quiet[missingTextSearchIndexes[]], Except[_List] :> {}];
            If[missingIndices =!= {},
                didReindex = True;
		        RunScheduledTask[
		            showReindexingDialog[missingIndices];
		            Documentation`HelpLookupPacletURI @@ searchArgs;
		            RemoveScheduledTask[$ScheduledTask],
		            {.01}
		        ]
            ]
        ];
        didReindex
    ]


missingTextSearchIndexes::usage="missingTextSearchIndexes[] returns a list of the Documentation folders that contain an Index but no TextSearchIndex."
missingTextSearchIndexes[] := With[
	{
		new = Block[{$UseTextSearchQ = True}, FileNameDrop[dropIndexVersionFromPath[#]]& /@ DocumentationIndexes[]],
		old = Block[{$UseTextSearchQ = False}, FileNameDrop /@ DocumentationIndexes[]]
	},
	(* Note: It is known that changing the setting of $UseTextSearchQ without re-getting DocumentationSearch` *)
	(* doesn't add the base index to DocumentationIndexes[], but that's not relevant here because we'll never release an M- version *)
	(* without the base index, and we don't want the user to re-index it if he accidentally deleted it because it takes a long time. *)
	Select[
		old, 
		StringFreeQ[#, $InstallationDirectory] && !MatchQ[FileNameTake[#, -3], Alternatives @@ (FileNameTake[#, -3]& /@ new)]&
	]
];

directoryWriteAccessibleQ[file_String] := 
	Module[
		{res = False, stream, type2}, 
		type2 = FileType[FileNameJoin[{file, "test"}]];
		Quiet[
			stream = OpenAppend[FileNameJoin[{file, "test"}], BinaryFormat -> True], 
			{OpenAppend::noopen}
		];
		If[
			stream === $Failed, 
			res = False, 
			Close[stream];
			If[type2 === None, DeleteFile[FileNameJoin[{file, "test"}]]];
			res = True
		];
		res
	]
	
(* we should have an option of CreateDirectory that avoids the error if the directory exists, and overwrites! *)
dirCreateIfNotExisting[dir_String] := If[DirectoryQ[dir], dir, CreateDirectory[dir]];

createTextSearchIndexesForPacletsWithOldIndexes::usage="createTextSearchIndexesForPacletsWithOldIndexes[] creates indices for all the folders returned by  missingTextSearchIndexes[]."
createTextSearchIndexesForPacletsWithOldIndexes[] := With[{paths = missingTextSearchIndexes[]},
	Scan[
		Function[
			path,
			CreateDocumentationIndex[
				path, 
				If[
					directoryWriteAccessibleQ[path], 
					dirCreateIfNotExisting[FileNameJoin[{path, $indexName}]], 
					With[
						{ubdPath = FileNameJoin[{$UserBaseDirectory, "DocumentationIndices", FileNameTake[path, -3]}]},
						If[
							!DirectoryQ[ubdPath] || !directoryWriteAccessibleQ[ubdPath],
							CreateDirectory[ubdPath]
						];
						(* Warning: this will overwrite an existing $UserBaseDirectory/SearchIndices/<paclet name>/Documentation/English/TextSearchIndex folder *)
						dirCreateIfNotExisting[FileNameJoin[{ubdPath, $indexName}]]
					]
				],
				$indexVersion,
				(* We're not indexing System` documentation *)
				"UseWolframLanguageData" -> False
			]
		],
		paths
	]
]

reindexingDialogNotebook[pkglist_, numpkgs_String] := 
  Notebook[{
    Cell[BoxData[RowBox[{StyleBox[numpkgs, "ReindexingDialogHeaderText"],
      DynamicBox[StyleBox[FEPrivate`FrontEndResource["MiscellaneousDialogs","ReindexingDialogExternalPackagesToReindex"],"ReindexingDialogHeaderText"]]}]],"ReindexingDialogHeader"],
    Cell[BoxData[DynamicBox[StyleBox[FEPrivate`FrontEndResource["MiscellaneousDialogs","ReindexingDialogPackageListPreamble"],"ReindexingDialogText"]]],"ReindexingDialogText"],
    Cell["","ReindexingDialogTopDelimiter"],
    Cell[BoxData[pkglist], "ReindexingDialogPackageList"],
    Cell["","ReindexingDialogBottomDelimiter"],
    Cell[BoxData[RowBox[{ToBoxes[ProgressIndicator[Appearance->"Necklace"]],
      ToBoxes[Spacer[10]],
      GridBox[{
        {DynamicBox[StyleBox[FEPrivate`FrontEndResource["MiscellaneousDialogs","ReindexingDialogCreatingNewIndices"],"ReindexingDialogText"]]},
        {DynamicBox[StyleBox[FEPrivate`FrontEndResource["MiscellaneousDialogs","ReindexingDialogMayTakeAWhile"],"ReindexingDialogText", FontSlant->"Italic"]]}}]}]],"ReindexingDialogBottomGrid"],
    Cell["","ReindexingDialogFooter"]},
    
    WindowTitle->Dynamic[FEPrivate`FrontEndResource["MiscellaneousDialogs", "ReindexingDialogWindowTitle"]],
    
    StyleDefinitions->Notebook[{
      Cell[StyleData["ReindexingDialogHeader"],CellMargins->{{30,30},{8,20}},FontFamily:>CurrentValue["PanelFontFamily"],FontSize->18,FontColor->GrayLevel[.2],ShowCellBracket->False,ShowStringCharacters->False],
      Cell[StyleData["ReindexingDialogHeaderText"],FontFamily:>CurrentValue["PanelFontFamily"],FontSize->18,FontColor->GrayLevel[.2],ShowCellBracket->False,ShowStringCharacters->False],
      Cell[StyleData["ReindexingDialogText"],CellMargins->{{30,30},{6,8}},FontFamily:>CurrentValue["PanelFontFamily"],FontColor->GrayLevel[.2],FontSize->13,ShowCellBracket->False,ShowStringCharacters->False],
      Cell[StyleData["ReindexingDialogTopDelimiter"],CellSize->{Automatic,1},ShowCellBracket->False,ShowStringCharacters->False,CellFrame->{{0,0},{.5,0}},CellMargins->{{0,0},{7,0}},CellFrameColor->GrayLevel[.75]],
      Cell[StyleData["ReindexingDialogBottomDelimiter"],CellSize->{Automatic,1},ShowCellBracket->False,ShowStringCharacters->False,CellFrame->{{0,0},{0,0.5}},CellMargins->{{0,0},{0,7}},CellFrameColor->GrayLevel[.75]],
      Cell[StyleData["ReindexingDialogPackageList"],CellMargins->{{30,30},{12,12}},ShowCellBracket->False,ShowStringCharacters->False],
      Cell[StyleData["ReindexingDialogBottomGrid"],CellMargins->{{30,30},{20,8}},GridBoxOptions->{GridBoxAlignment->{"Columns"->{{Left}}}},ShowCellBracket->False,ShowStringCharacters->False],
      Cell[StyleData["ReindexingDialogFooter"],CellMargins->{{0,0},{4,4}},ShowCellBracket->False,ShowStringCharacters->False,CellSize->{Automatic,1}]},
      StyleDefinitions->"Default.nb"],
      
    WindowSize->Fit,WindowElements->None,
    WindowFrame->"ModelessDialog",ShowSelection->False,Editable->False,Saveable->False,Selectable->False,
    Background->GrayLevel[1],WindowFrameElements->{"CloseBox"},
    ShowAutoSpellCheck->False];

dumpReindexingLog[paths_List]:= Module[{targetDir},
	targetDir = FileNameJoin[{$UserBaseDirectory, "Logs", "DocumentationSearch"}];
	If[Not@DirectoryQ[targetDir], CreateDirectory[targetDir]];
	PutAppend[paths, FileNameJoin[{targetDir, "MissingIndexCreation.m"}]]
];

showReindexingDialog[paths_List] := Module[{dialog, indexnames}, 
	$firstSearch = False;
	Internal`WithLocalSettings[
		indexnames = FileNameTake[#, {-3}] & /@ paths;
		(* Set threshold for dialog appearance based on number of paclets to index *)
		If[Length[indexnames] > 0,
			
			indexnames = Partition[Table[With[{i=i},
		      Style[indexnames[[i]],FontFamily:>CurrentValue["PanelFontFamily"],ShowStringCharacters->False,
		        FontColor->GrayLevel[0.537255], FontSlant->"Italic"]], {i, Length[indexnames]}], UpTo[4]];
		        
		dialog = CreateDialog[
		   reindexingDialogNotebook[ToBoxes[
		     Column[Row[#, Style["\[NonBreakingSpace]\[FilledSmallCircle] ",FontFamily:>CurrentValue["PanelFontFamily"], FontColor->GrayLevel[0.537255],
		       ShowStringCharacters->False]]& /@ indexnames]], ToString[Length[paths]]]
		    ]
		],
  
  	Quiet[dumpReindexingLog[paths];createTextSearchIndexesForPacletsWithOldIndexes[]],
	
	NotebookClose[dialog];
	];
	
	]


(* The default ContentFieldOptions used by CreateDocumentationIndex *)
$defaultContentFieldOptionsForDocIndex = <|
	"Title" -> <|"Stored" -> True, "Weight" -> 2|>,
	"ExactTitle" -> <|"Stored" -> True, "Tokenized" -> False, "LengthWeighted" -> False|>,
	"NormalizedTitle" -> <|"Tokenized" -> False, "LengthWeighted" -> False|>,
	"PossibleAlternateNames" -> <|"Weight" -> 1.5, "LengthWeighted" -> False|>,
	(*"OtherComputerLanguageEquivalents" -> <|"Weight" -> 1.5|>,*)
	"ShortNotations" -> <|"Tokenized" -> False, "LengthWeighted" -> False|>, (* for "/@", "@@" etc. *)
	"Synonyms" -> <|"Weight" -> 1.5, "Stored" -> True, "LengthWeighted" -> False|>,
	"Keywords" -> <|"Stored" -> True|>,
	"URI" -> <|"Stored" -> True, "Tokenized" -> False, "LengthWeighted" -> False|>,
	"SnippetPlaintext" -> <|"Stored" -> True, "Searchable" -> False, "Tokenized" -> False|>,
	"ReferredBy" -> <|"Weight" -> .5, "LengthWeighted" -> False|>,
	"Description" -> <|"Stored" -> True, "Searchable" -> False, "Tokenized" -> False|>,
	"PacletName" -> <|"Stored" -> True, "Searchable" -> False, "Tokenized" -> False|>,
	"Language" -> <|"Stored" -> True, "Searchable" -> False, "Tokenized" -> False|>,
	"Usage" -> <|"Weight" -> 1.5|>,
	"Abstract" -> <|"Weight" -> 1.5|>,
	"NotebookType" -> <|"Stored" -> True, "Tokenized" -> False, "BulkRetrievalOptimized" -> True, "LengthWeighted" -> False|>,
	"TokenizedNotebookType" -> <|"LengthWeighted" -> False|>,
	"DisplayedCategory" -> <|"Stored" -> True, "Searchable" -> False, "Tokenized" -> False|>,
	"URL" -> <|"Stored" -> True, "Searchable" -> False, "Tokenized" -> False|>,
	"Context" -> <|"Stored" -> True, "Searchable" -> False, "Tokenized" -> False|>,
	"SeeAlso" -> <| "LengthWeighted" -> False|>,
	"ExampleText" -> <|"Weight" -> .05|>,
	"Frequency" -> <|"Stored" -> True, "Searchable" -> False, "Tokenized" -> False, "BulkRetrievalOptimized" -> True|>,
	"NotebookStatus" -> <|"Stored" -> True, "Searchable" -> False, "Tokenized" -> False, "BulkRetrievalOptimized" -> True|>,
	"NotebookPackage" -> <|"Stored" -> True, "Searchable" -> False, "Tokenized" -> False, "BulkRetrievalOptimized" -> True|>,
	"Dictionary" -> <|"OmitTermFreqAndPositions" -> True|>
|>;

$notebookTypeRanking = {
	{"Symbol", "Guide", "Root Guide", "Upgrade Information" (* we only index linked symbol names for Compatibility pages, so that's enough as a penalty for those pages *), "WorkflowGuide"},
	{"CompiledType", "Format", "Entity", "ExternalEvaluationSystem","Predictor", "Classifier", "Method", "NetEncoder", "NetDecoder", "Workflow"},
	{"Interpreter", "Tutorial", "Overview", "Device Connection", "Service Connection"},
	{"Character Name", "MathLink C Function", "UnityLink C# Function"},
	{"System Program", "Program", "File", "Embedding Format", "Indicator"},
	{"Front End Object"},
	{"Example", "Menu Item", "Message", "Widget"}
};
notebookTypeRank[class_] := Replace[FirstPosition[$notebookTypeRanking, class][[1]], "NotFound" -> Length[$notebookTypeRanking]];
notebookTypeToWeight[class_String] := N[1-(notebookTypeRank[class]-1)/(2 (Length[$notebookTypeRanking]-1))]; (* max: 1, min: .5, linear scale *)

$notebookStatusToWeight = {
	"ObsoleteFlag" -> 0.000001, (* don't use 10^-6 or Rational[...] will cause issues in the web deployed search index *)
	"AwaitingFutureDesignReviewFlag" -> .25,
	"NewInOldVersion" -> 0.00001,
	"None" -> 1.
};

$notebookPackageToWeight = {
	(* we only index linked symbol names for Compatibility enough, so that's enough as a penalty for those pages *)
	"Compatibility" -> 1., 
	"Package" -> .2,
	"Mathematica" -> 1.
};

(* can't use AssociationMap per bug 340543 and TaliB - chiarab, Oct 2017 *)
associationMap[f_, list_] := Association @ Map[# -> f[#]&, list];

(* The default DocumentWeightingRules used by CreateDocumentationIndex *)
$defaultDocumentWeightingRulesForDocIndex = {
	"NotebookType" -> Normal[
		associationMap[
			notebookTypeToWeight, 
			Flatten[$notebookTypeRanking]
		]
	],
	"NotebookStatus" -> $notebookStatusToWeight,
	"NotebookPackage" -> $notebookPackageToWeight
};

(* The default DocumentWeightingFunction used by CreateDocumentationIndex *)
$defaultDocumentWeightingFunctionForDocIndex = {
	"min(max(1+(log10(Frequency)),0.1),1.5)"
}

ClearAll[CreateDocumentationIndex];
Options[CreateDocumentationIndex] = Join[
	Options[Skeletonize], 
	FilterRules[Options[CreateSearchIndex], Except[OverwriteTarget]], 
	{OverwriteTarget -> True, "DisableProgressBar" -> True}
];

CreateDocumentationIndex::nvldloc = "Invalid index directory."

(* Skeletonize and create a search index for the WL documentation *)
CreateDocumentationIndex[opts:OptionsPattern[]] := CreateDocumentationIndex[$docDir, Automatic, $textSearchIndexName, opts];
CreateDocumentationIndex[filesOrDir_, opts:OptionsPattern[]] := CreateDocumentationIndex[filesOrDir, Automatic, $textSearchIndexName, opts];
CreateDocumentationIndex[filesOrDir_, indexDir_, opts:OptionsPattern[]] := CreateDocumentationIndex[filesOrDir, indexDir, $textSearchIndexName, opts];
CreateDocumentationIndex[
		filesOrDirIn_, 
		indexDirIn_,
		name_, 
		opts:OptionsPattern[]
	] := Block[
		{
			skelOpts = FilterRules[{opts}, Options[Skeletonize]], 
			csiOpts = FilterRules[{opts}, FilterRules[Options[CreateSearchIndex], Except[ContentFieldOptions|ContentLocationFunction|OverwriteTarget|DocumentWeightingRules]]],
			contentFieldOptions = OptionValue[ContentFieldOptions],
			contentLocationFunction = OptionValue[ContentLocationFunction],
			overwriteTarget = OptionValue[OverwriteTarget],
			docWeightingRules = OptionValue[DocumentWeightingRules],
			docWeightingFunction = OptionValue[DocumentWeightingFunction],
			skeletonizerRes,
			index,
			filesOrDir = Replace[filesOrDirIn, Automatic :> $docDir],
			indexDir
		},
			indexDir = Replace[
				indexDirIn,
				Automatic :> If[
					TrueQ[Quiet[DirectoryQ[filesOrDir]]],
					filesOrDir,
					$Failed
				]
			];
			
			If[
				!TrueQ[Quiet[DirectoryQ[indexDir]]], 
				Message[CreateDocumentationIndex::nvldloc];
				Return[$Failed]
			];
			
		 	skeletonizerRes = Skeletonize[filesOrDir, Sequence@@skelOpts];
		 	
		 	If[skeletonizerRes === $Failed, Return[$Failed]];
		 	
		 	Block[{
		 		TextSearch`$DisableProgressBar = TrueQ[OptionValue["DisableProgressBar"]],
		 		TextSearch`$SearchIndicesDirectory = indexDir
		 		},
		 		CreateSearchIndex[
					skeletonizerRes,
					name,
					csiOpts,
					Language -> $SearchLanguage,
					ContentFieldOptions -> Join[
						$defaultContentFieldOptionsForDocIndex, 
						contentFieldOptions
					],
					OverwriteTarget -> overwriteTarget,
					ContentLocationFunction -> Replace[
						contentLocationFunction,
						Automatic :> <|
								"SnippetSource" -> ("field:SnippetPlaintext"&),
								"ReferenceLocation" -> Function[c, URL[StringReplace[c["URL"], "http://reference.wolfram.com/language/" -> "http://reference.wolframcloud.com/cloudplatform/"]]]
							|>
					],
					DocumentWeightingRules -> Join[
						$defaultDocumentWeightingRulesForDocIndex, 
						Replace[docWeightingRules, None -> {}]
					],
					DocumentWeightingFunction -> Join[
						$defaultDocumentWeightingFunctionForDocIndex, 
						Replace[docWeightingFunction, None -> {}]
					]
			 	]
		 	]	
];

(* A ranking function where we check if the query is a straightforward symbol name *)
(* Note: this is not used by the in-product doc search for the time being *)
(*directMatchRankingFunction[max_] := Function[{objects, args},
	Module[
		{directhit, tmp, qstring}, 
		
		qstring = Replace[args["Query"], SearchQueryString[s_] :> s];
		If[!StringQ[qstring], Return[objects]];
		directhit = Select[
			Take[objects, max],
			#["NotebookType"] === "Symbol" && StringMatchQ[FileBaseName[#["URL"]], qstring, IgnoreCase -> True]&
		];
		If[
			!MatchQ[directhit, {_ContentObject}]
			,
    		objects
    		,
			directhit = First[directhit];
			tmp = directhit;
			directhit = ContentObject[
				Association[
					Append[
						Normal[directhit][[1]], 
						"Status" -> "DirectMatch"
					]
				]
			];
    		Prepend[
    			DeleteCases[objects, tmp], 
    			directhit
    		]
		]
	]
];*)

SearchDocumentation::noinit = "Warning: Something went wrong with the initialization of DocumentationSearch."

InitializeDocumentationSearch[] := Quiet[
	Module[{indexDir, baseIndexDir = FileNameJoin[{$InstallationDirectory, "Documentation", ToString[$SearchLanguage], $indexName, $indexVersion}]}, 
		TextSearch;
		JLink`InstallJava[];
		RunScheduledTask[
			Quiet[
				Get["DocumentationSearch`"];
   				indexDir = DocumentationSearch`DocumentationIndexes[];
				DocumentationSearch`Private`$textSearchIndex = Quiet[TextSearch`PackageScope`addSubIndex[SearchIndexObject[File[baseIndexDir]], DeleteCases[indexDir, baseIndexDir]], SearchIndexObject::badind];
				DocumentationSearch`Private`$textSearchIndexDir = indexDir;
				TextSearch`PackageScope`warmUp[DocumentationSearch`Private`$textSearchIndex]
			], 
			{2},
			"AutoRemove" -> True
		];
	]
];


NewDocumentationIndexMerger[directory_String /; FileType[directory] === Directory] := 
JavaBlock[
	Module[{merger},
		InstallJava[];
		AddToClassPath[ToFileName[{DirectoryName[FindFile["DocumentationSearch`"],2], "Java", "Lucene21"}]];
		merger = JavaNew["com.wolfram.documentationsearch.IndexMerger", directory];
		KeepJavaObject[merger];
		DocumentationIndexMerger[merger]
	]
]

MergeDocumentationIndex[DocumentationIndexMerger[merger_?JavaObjectQ], directory_String /; FileType[directory] === Directory] := 
JavaBlock[
	Module[{},
		merger@addIndex[directory];
	]
]

MergeDocumentationIndexes[DocumentationIndexMerger[merger_?JavaObjectQ], directories_List] := 
JavaBlock[
	Module[{},
		merger@addIndexes[directories];
	]
]

CloseDocumentationIndexMerger[DocumentationIndexMerger[merger_?JavaObjectQ]] :=
JavaBlock[
	Module[{},
		merger@close[];
	]
]

Options[NewDocumentationNotebookIndexer] := {
  "Language"->$SearchLanguage,
  (* New option from tgayley; controls whether to use Lucene 2.1 libs for indexing (needed for in-product
     indexes, compatible with CLucene used for in-product searching), or the newer Lucene 3 libs for 
     web search and anywhere else searching is still being done with Java.
  *)
  "InProductIndexFormat"->True
};

NewDocumentationNotebookIndexer[directory_String, opts___Rule] :=
  JavaBlock[
    Module[{indexer, useOpts, lang, oldLucene, fieldBoostFile},
      fieldBoostFile = FileNameJoin[{$PackageDir, "fieldBoost.json"}];
	  spellingsFile = FileNameJoin[{$PackageDir, "misspellings.json"}];
      useOpts  = canonicalOptions[Flatten[{opts}]];
      {lang, oldLucene} = {"Language", "InProductIndexFormat"} /. useOpts /. Options[ NewDocumentationNotebookIndexer ];
      InstallJava[];
      (* This app supports both old- and new-style doc indexes. We add the appropriate subdir to the class path. 
         Old-style indices are used for in-product search, to be compatible with CLucene. This is the default.
      *)
      If[TrueQ[oldLucene],
          (* In the app layout, the Lucene 2.1 libs are in the Java/Lucene21 dir. Here we put them ahead of the 3.x
             libs, which are in the standard location of just the Java dir.
          *)
          AddToClassPath[ToFileName[{DirectoryName[FindFile["DocumentationSearch`"],2], "Java", "Lucene21"}]],
      (* else *)
          AddToClassPath[ToFileName[{DirectoryName[FindFile["DocumentationSearch`"],2], "Java", "Lucene30"}]]
      ];
      Switch[lang, 
			"Japanese", 
				indexer = JavaNew["com.wolfram.documentationsearch.JapaneseDocumentationIndexer", directory],
      		_, 
				indexer = JavaNew["com.wolfram.documentationsearch.DocumentationIndexer", directory, fieldBoostFile, spellingsFile]];
      KeepJavaObject[indexer];
      DocumentationNotebookIndexer[indexer]
    ]
  ]
  
CloseDocumentationNotebookIndexer[DocumentationNotebookIndexer[indexer_?JavaObjectQ]] := 
  indexer@close[];

AddDocumentationNotebook[indexers_List,  notebook_String /; FileType[notebook] === File] :=
JavaBlock[
	Module[{nb = quietGet[notebook], text, doc, taggingRules, metaData, index},
		doc = JavaNew["com.wolfram.documentationsearch.DocumentationNotebook"];
		(*plain text of notebook*)
		Developer`UseFrontEnd[text = Import[notebook, "Plaintext"]];

		(*gather meta data*)
		taggingRules = TaggingRules /. Options[nb] /. {TaggingRules -> {}};
		metaData = "Metadata" /. taggingRules /. {"Metadata" -> {}};
		index = "index" /. metaData;
		
		(*add notebook to index*)
		If[TrueQ[index],
			(
				If[MatchQ[#[[1]], _DocumentationNotebookIndexer] && MatchQ[#[[2]], _Function],
					AddDocumentationNotebook[#[[1]], text, metaData, #[[2]], notebook],
					Print[ "Skipping argument to indexers... not {DocumentationNotebookIndexer, Function} pair"]
				]
			) & /@ indexers,
						
			Print["Skipping " <> notebook];
		]
	]
]

FetchReferencedFunctions[nbgot_] :=
	DeleteDuplicates[
		Flatten[Cases[nbgot,
			Cell[CellGroupData[{Cell[___, "GuideReferenceSection", ___], rest___}, ___], ___] :>
				Cases[{rest},
					TemplateBox[{_, uri_String} /; StringMatchQ[uri, "paclet:ref/*"], "RefLink", ___]
						:> StringReplace[uri, "paclet:ref/" -> ""], Infinity], Infinity]]];
	 
	 
AddDocumentationNotebook[jo:DocumentationNotebookIndexer[indexer_?JavaObjectQ], notebook_String /; FileType[notebook] === File] := 
	AddDocumentationNotebook[DocumentationNotebookIndexer[indexer], notebook, Function[1]];
	
AddDocumentationNotebook[jo:DocumentationNotebookIndexer[indexer_?JavaObjectQ], notebook_String /; FileType[notebook] === File, boostFunc_Function] := 
  JavaBlock[
    Module[{nb = quietGet[notebook], text, doc, taggingRules, metaData, index},
      doc = JavaNew["com.wolfram.documentationsearch.DocumentationNotebook"];
      (* plain text of notebook *)
      Developer`UseFrontEnd[text = Import[notebook, "Plaintext"]];
	  
      (* gather meta data *)
      taggingRules = TaggingRules /. Options[nb] /. {TaggingRules -> {}};
      metaData = "Metadata" /. taggingRules /. {"Metadata" -> {}};   
	  index = "index" /. metaData;   
      (* add notebook to index *)
      If[TrueQ[index],
      	AddDocumentationNotebook[jo, text, metaData, boostFunc, notebook],
      	Print["Skipping " <> notebook];
      ]
]];

AddDocumentationNotebook[DocumentationNotebookIndexer[indexer_?JavaObjectQ], text_String, metaData_List] := 
	AddDocumentationNotebook[DocumentationNotebookIndexer[indexer], text, metaData, Function[1], ""];
	
AddDocumentationNotebook[DocumentationNotebookIndexer[indexer_?JavaObjectQ], text_String, metaData_List, notebook_String] := 
	AddDocumentationNotebook[DocumentationNotebookIndexer[indexer], text, metaData, Function[1], notebook];
	
AddDocumentationNotebook[DocumentationNotebookIndexer[indexer_?JavaObjectQ], text_String, metaData_List, boostFunc_Function, notebook_String] := 
  JavaBlock[
    Module[{ doc, type, context, keywords, name, summary, title, uri, synonyms, status, label, lang},
      
      doc = JavaNew["com.wolfram.documentationsearch.DocumentationNotebook"];

      {type, context, keywords, name, label, summary, title, uri, synonyms, status, lang} = 
        {"type", "context", "keywords", "paclet", "label", "summary", "title", "uri", "synonyms", "status", "language"} 
          /. metaData /. 
            {"type" -> "", "context" -> "", "keywords" -> {}, "paclet" -> "", "label"->"",
             "summary" -> "", "title" -> "", "uri"->"", "synonyms" -> {}, "status" -> "None", "language"->"en"};

	  keywords = Union[Flatten[StringSplit[#] & /@ keywords]];

	  boost = boostFunc[uri];

	  If[
	  StringMatchQ[uri, "guide/*"] && StringLength[notebook] > 0 && FileExistsQ[notebook],(
	    keywords = Union[keywords, FetchReferencedFunctions[quietGet[notebook]]];
      )]
	  
	  doc@setBoost[boost];
      doc@setType[type];
      doc@setContext[context];
      doc@setKeywords[keywords];
      doc@setSynonyms[synonyms];
      doc@setPacletName[name];
      doc@setTitle[title];
      doc@setLabel[label];
      doc@setSummary[summary];
      doc@setContent[ text ];
      doc@setURI[uri];
      doc@setStatus[status];
      doc@setLanguage[lang];
      
      indexer@addNotebook[doc];      
    ]
  ]

AddDocumentationDirectory[indexers_List, directory_String] := 
  Module[{files},
    If[FileType[directory] =!= Directory, 
      Return[];
    ];
    Block[{$rootDirectory = $rootDirectory},     
      If[$rootDirectory === Null, 
        $rootDirectory = directory;
      ];
      
      files = FileNames[ToFileName[directory, "*"]];
      (
        Switch[FileType[#], 
          Directory, 
            Which[
               StringMatchQ[#, "*ExampleData" | "*Examples" | "*RawGuides"],
                  Print["Skipping ", #];
                  Null, 
                True, 
                  AddDocumentationDirectory[indexers, #]
            ],          
          File, 
            AddDocumentationNotebook[indexers, #] 
        ]
      ) & /@ files;
    ];
  ]; 

AddDocumentationDirectory[DocumentationNotebookIndexer[indexer_?JavaObjectQ], directory_String] := 
	AddDocumentationDirectory[DocumentationNotebookIndexer[indexer], directory, Function[1]];
	
AddDocumentationDirectory[DocumentationNotebookIndexer[indexer_?JavaObjectQ], directory_String, boostFunc_Function] := 
  Module[{files},
    If[FileType[directory] =!= Directory, 
      Return[];
    ];
    Block[{$rootDirectory = $rootDirectory},     
      If[$rootDirectory === Null, 
        $rootDirectory = directory;
      ];
      
      files = FileNames[ToFileName[directory, "*"]];
      (
        Switch[FileType[#], 
          Directory, 
            Which[
               StringMatchQ[#, "*ExampleData" | "*Examples" | "*RawGuides"],
                  Print["Skipping ", #];
                  Null, 
                True, 
                  AddDocumentationDirectory[DocumentationNotebookIndexer[indexer], #, boostFunc]
            ],          
          File, 
            AddDocumentationNotebook[DocumentationNotebookIndexer[indexer], #, boostFunc] 
        ]
      ) & /@ files;
    ];
  ]; 
  
  
CreateSpellIndex[indexDir_String, spellIndexDir_String, fields_List] := 
  JavaBlock[
      InstallJava[];
      LoadJavaClass["com.wolfram.documentationsearch.spelling.DidYouMeanIndexer"];
      DidYouMeanIndexer`createSpellIndex[fields, indexDir, spellIndexDir];
  ]
  
CreateSpellIndex[indexDir_String, spellIndexDir_String] := 
  JavaBlock[
      InstallJava[];
      LoadJavaClass["com.wolfram.documentationsearch.spelling.DidYouMeanIndexer"];
      DidYouMeanIndexer`createSpellIndex["text", indexDir, spellIndexDir];
  ]

quietGet[name_String] :=
  Module[{expr},
    Off[Syntax::"newl"];
    expr = Get[name];
    On[Syntax::"newl"];
    expr
  ]; 

DirectHitSearch[criteria_String] := 
  Module[{resolvedLink, 
      pacletcrit = StringReplace[criteria, {"paclet:ref/" -> "", StartOfString ~~ "ref/" -> ""}]},
    resolvedLink=Documentation`ResolveLink["paclet:ref/" <> pacletcrit];
    If[resolvedLink===Null,
      DirectHitSearch[DocumentationIndexes[],criteria],
      {criteria,"ref/"<>pacletcrit}]];

DirectHitSearch[indexDir_String, criteria_String] := 
  DirectHitSearch[{indexDir}, criteria]

DirectHitSearch[indexDir:{__String}, criteria_String] := 
    Module[{startCriteria, limitCriteria, results, matches, systemSymbolMatch, systemFormatMatch, match, directHitCriteria},
  
        startCriteria = StringCases[criteria, RegularExpression[" start:(\\d+)"] -> "$1"];
        If[Length[startCriteria] > 0, Return[Null]];
    
        limitCriteria = StringCases[criteria, RegularExpression[" limit:(\\d+)"] -> "$1"];
        If[Length[limitCriteria] > 0, Return[Null]];    
    	
    	directHitCriteria = If[
    		!TrueQ[$UseTextSearchQ],
    		"+(exacttitle:\"" <> criteria <> "\") +type:(Symbol OR Format)",
    		"+(ExactTitle:\"" <> criteria <> "\") +NotebookType:(\"Symbol\" OR \"Format\" OR \"Entity\")"
    	];
    	
        results = SearchDocumentation[indexDir, {}, directHitCriteria, 
                                        "Limit"->3, "MetaData"->{"Title", "Type", "URI"}];
                                        
        If[!MatchQ[results, {___Rule}], (* Was an error in the LibraryFunction call; should not occur. *) Return[Null]];
        {matches} = {"Matches"} /. results  /. {"Matches" -> {}};
        If[Length[matches] > 0,
            (* We can get more than one result in two cases: (1) a system symbol and a format have the same name (like "C"), in which
               case we give preference to the symbol; and (2) a user paclet symbol has the same name as a system symbol or format, in
               which case we give preference to the system one.
               Note that we always require an exact case match.
            *)
            systemSymbolMatch = Cases[matches, {criteria, "Symbol", _?(StringMatchQ[#, "ref/*"]&), ___}];
            If[Length[systemSymbolMatch] > 0,
                Return[First[systemSymbolMatch][[{1,3}]]]
            ];
            systemFormatMatch = Cases[matches, {criteria, "Format", _?(StringMatchQ[#, "ref/format/*"]&), ___}];
            If[Length[systemFormatMatch] > 0,
                Return[First[systemFormatMatch][[{1,3}]]]
            ];
            (* Must be a user paclet symbol. If there is just one result, allow a direct hit, otherwise fail here and fall through to search results page. *)
            If[Length[matches] == 1,
                match = First[matches];
                If[MatchQ[match, {criteria, _, _, ___}],
                    Return[match[[{1,3}]]]
                ]
            ]
        ];
        (* Get here means no appropriate matches found. *)
        Null
    ]
  
  
CloseDocumentationIndex[indexDir_String]:=
    If[libFunctionsAvailable,
        closeSearcher[indexDir],
    (* else *)
        InstallJava[];
	    LoadJavaClass["com.wolfram.documentationsearch.DocumentationSearcher"];
	    DocumentationSearcher`closeSearcher[indexDir]
	]

Options[SearchDocumentation] := {
  "MetaData" -> {"Title", "Type", "ShortenedSummary", "URI", "Description", "Context"},
  "Start" -> 1, 
  "Limit" -> 10,
  "Output" -> Automatic
};

SearchDocumentation[criteria_String, opts___Rule] := 
  SearchDocumentation[DocumentationIndexes[], 
    DocumentationSpellIndexes[], criteria, opts]

SearchDocumentation[indexDir_String, criteria_String, opts___Rule] := 
  SearchDocumentation[{indexDir}, DocumentationSpellIndexes[], criteria, opts]

SearchDocumentation[indexDir:{__String}, criteria_String, opts___Rule] := 
  SearchDocumentation[indexDir, DocumentationSpellIndexes[], criteria, opts]

SearchDocumentation[indexDir_String, spellIndexDir_String, criteria_String, opts___Rule] := 
  SearchDocumentation[{indexDir}, {spellIndexDir}, criteria, opts]

(* Store the index and list of index dirs in use to avoid re-opening it every time even if the dirs are the same *)
$textSearchIndex = None;
$textSearchIndexDir = {};
$firstSearch = True;

(* New version, using TextSearch for searching *)
(* TODO: take care of searchLang *)
SearchDocumentation[indexDirIn:{__String}, spellIndexDirIn:{___String}, criteria_String, opts___Rule] /; TrueQ[$UseTextSearchQ] := 
  Module[{useOpts, metaData, start, limit, output, result, newCriteria = criteria, searchLang = $SearchLanguage, 
  		indexDir = indexDirIn, index, time, count, startCriteria, limitCriteria, res, postProcess, queryString, termSugg, suggestions},
    
    (* $firstSearch should be set to False already if it was needed, but in case it isn't, we set it here now. *)
    $firstSearch = False;
    
    useOpts  = canonicalOptions[Flatten[{opts}]];
    metaData = Flatten[{"MetaData" /. useOpts /. Options[ SearchDocumentation ]}, 1]; (* we need this to be a list for post-processing *)
    start    = "Start" /. useOpts /. Options[ SearchDocumentation ];
    limit    = "Limit" /. useOpts /. Options[ SearchDocumentation ];
    output   = "Output" /. useOpts /. Options[ SearchDocumentation ];
    
    If[!MemberQ[metaData, "Score"] && TrueQ[$DebugDocumentationSearchQ], metaData = Join[metaData, {"Score"}]];
    (* Process start in query *)
    startCriteria = StringCases[criteria, RegularExpression[" start:(\\d+)"] -> "$1"];
    If[Length[startCriteria] > 0, start = ToExpression[First[startCriteria]]];
    newCriteria = StringReplace[criteria, RegularExpression[" start:\\d+"] -> ""];
    
    (* Process limit in query *)
    limitCriteria = StringCases[newCriteria, RegularExpression[" limit:(\\d+)"] -> "$1"];
    If[Length[limitCriteria] > 0, limit = ToExpression[First[limitCriteria]]];
    newCriteria = StringReplace[newCriteria, RegularExpression[" limit:\\d+"] -> ""];
    
    queryString = docQueryString[newCriteria];
    
    postProcess = Replace[
    	Map[$postProcessingFunctions, metaData],
    	_Missing :> Identity,
    	{1}
    ];
    
    (* Replace doc search metadata with actual field names in the doc index *)
    metaData = Replace[
    	metaData, 
    	{
    		"Title" -> "ExactTitle", (* The "Title" field can contain also the subtitle *)
    		"Type" -> "NotebookType", (* "Type" can't the name of a Stored field in TextSearch *)
    		"Description" -> "DisplayedCategory", (* The "Description" is called "DisplayedCategory" in TextSearch... *)
    		"ShortenedSummary" -> "Description", (* + post-processing *)
    		"Summary" -> "Description", (* ...also because "Description" is something else (the "summary" metadata) *)
    		"Explanation" :> (Needs["JLink`"]; LoadJavaClass["com.wolfram.textsearch.LuceneDebug"]; LuceneDebug`enableScoreExplanation[]; "ScoreExplanation"),
    		"Boost" -> "Score" (* this is certainly wrong, but I can't see what means so I'm returning the score for now *)
    	}, 
    	{1}
    ];
    
   	If[
    	$textSearchIndex =!= None && indexDir === $textSearchIndexDir
    	,
    	index = $textSearchIndex
    	,
    	index = If[
    		Length[indexDir] > 1, 
    		With[
    			{baseIndexDir = FileNameJoin[{$InstallationDirectory, "Documentation", ToString[$SearchLanguage], $indexName, $indexVersion}]},
    			Quiet[TextSearch`PackageScope`addSubIndex[SearchIndexObject[File[baseIndexDir]], DeleteCases[indexDir, baseIndexDir]], SearchIndexObject::badind]
    		],
	    	SearchIndexObject[File[First[indexDir]]]
    	];
    	$textSearchIndexDir = indexDir;
		$textSearchIndex = index
    ];

    Block[{
    	TextSearch`PackageScope`$ShowJavaExceptions = TrueQ[$DebugDocumentationSearchQ]
    	},
		{time, result} = AbsoluteTiming[
			Quiet[
				TextSearch[
					index, 
					queryString
				],
				{TextSearch::interrcode}
			]	
		]
    ];
	
	If[
		output === SearchResultObject,
		Return[result]
	];
	
	If[
		MatchQ[result, _SearchResultObject]
		, 
		count = result["Count"];
		res = result[start;;Min[count, (start+limit-1)], metaData]
		, 
		count = 0;
		res = {}
	];
	
	(* post-processing, e.g. shortening the "Description" field for "ShortenedSummary" *)
	res = Map[
		Function[
			data, 
			MapThread[
				#1[#2]&, 
				{
					postProcess, 
					data
				}
			]
		], 
	 	res
	];
	
	termSugg = If[
		(* Don't output suggestions if there were results from the original search *)
		MatchQ[res, {__List}], 
		{}, 
		DeleteCases[result[{"TermSuggestions", "Dictionary"}], HoldPattern[Rule[w_String, _]] /; DictionaryWordQ[w]]
	];
	suggestions = If[MatchQ[termSugg,{__Rule}], StringReplace[newCriteria, termSugg, IgnoreCase -> True], Null];
	If[suggestions === newCriteria, suggestions = Null];
	
	{
		"Query" -> newCriteria,
		"ParsedQuery" -> newCriteria,
		"Start" -> start, 
		"Limit" -> limit,
		"SearchTime" -> time,
		"TotalMatches" -> count,
		"Suggestions"-> suggestions,
		"Matches" -> res,
		If[
			TrueQ[$DebugDocumentationSearchQ],
			"IndexLocation" -> index[[1,1]],
			Sequence@@{}
		]
     }
     
  ];

(* Old version, using either C++ or Java code for searching *)
SearchDocumentation[indexDirIn:{__String}, spellIndexDirIn:{___String}, criteria_String, opts___Rule] /; !TrueQ[$UseTextSearchQ]:= 
  Module[{useOpts, metaData, start, limit, result, newCriteria = criteria, searchLang = $SearchLanguage, 
  		indexDir = indexDirIn, spellIndexDir = spellIndexDirIn},
    useOpts  = canonicalOptions[Flatten[{opts}]];
    metaData = "MetaData" /. useOpts /. Options[ SearchDocumentation ];
    start    = "Start" /. useOpts /. Options[ SearchDocumentation ];
    limit    = "Limit" /. useOpts /. Options[ SearchDocumentation ];
    
    (* Process start in query *)
    startCriteria = StringCases[criteria, RegularExpression[" start:(\\d+)"] -> "$1"];
    If[Length[startCriteria] > 0, start = ToExpression[First[startCriteria]]];
    newCriteria = StringReplace[criteria, RegularExpression[" start:\\d+"] -> ""];
    
    (* Process limit in query *)
    limitCriteria = StringCases[newCriteria, RegularExpression[" limit:(\\d+)"] -> "$1"];
    If[Length[limitCriteria] > 0, limit = ToExpression[First[limitCriteria]]];
    newCriteria = StringReplace[newCriteria, RegularExpression[" limit:\\d+"] -> ""];
    	
    If[libFunctionsAvailable,
        search[indexDir, spellIndexDir, newCriteria, start, limit, metaData, searchLang] /. HoldPattern["Matches" -> m_] :> ("Matches" -> DeleteDuplicates[m]),
    (* else *)
        JavaBlock[        
            InstallJava[];
            (* If Java is used for searching (this is legacy only), we always use the new-style Java code. *)
            AddToClassPath[ToFileName[{DirectoryName[FindFile["DocumentationSearch`"],2], "Java", "Lucene30"}]];
            Switch[searchLang, 
                    "Japanese", 
                        LoadJavaClass["com.wolfram.documentationsearch.JapaneseDocumentationSearcher"];
                        result = JapaneseDocumentationSearcher`search[indexDir, newCriteria, start, limit],
                    _, 
                        LoadJavaClass["com.wolfram.documentationsearch.DocumentationSearcher"];
                        result = DocumentationSearcher`search[indexDir, spellIndexDir, newCriteria, start, limit]
            ];
            {"Query"->result@getQuery[],
             "ParsedQuery"->result@getParsedQuery[],
             "Start"->result@getStart[], 
             "Limit"->result@getLimit[],
             "SearchTime"->result@getSearchTime[], 
             "TotalMatches"->result@getTotalMatches[],  
             "Suggestions"->result@getSuggestion[],       
             "Matches"->result@getMatchValues[metaData]}
        ]
    ]
     
  ];

libFile = FindLibrary["DocSearch"];
libFunctionsAvailable =
    If[StringQ[libFile],
        Quiet[
            search = LibraryFunctionLoad[libFile, "search", LinkObject, LinkObject];
            closeSearcher = LibraryFunctionLoad[libFile, "closeSearcher", LinkObject, LinkObject]
        ];
        search =!= $Failed && closeSearcher =!= $Failed,
    (* else *)
        (* Library could not be found; will quietly fall back to using Java. *)
        False
    ]



SearchDocumentationMetaData[] := {
  "Title", "Summary", "URI", "URL", "Type", "Score", "Explanation", "Synonyms", "Boost",  
  "Keywords", "Context", "PacletName", "ShortenedSummary", "Language", "Description"
};

ExportSearchResults[results_List, "Notebook"] := 
  notebookSearchResult[results]

ExportSearchResults[results_List, format_] :=
  Message[ExportSearchResults::format, format] 


(* string to boxes *)
(* v12 -- no longer called, no more italics *)
ItalizeWordsInString[str_String]:=
Module[{res},
  res = StringReplace[str, RegularExpression["(J/Link|\\w+)"] :> italizeWord@"$1"];
  Which[
    Head@res === StringExpression, 
      Flatten@{ List@@res },
    Head@res === String,
      res,
    True,
      $Failed
  ]
];

italizeWord[w_String, o___?OptionQ] :=
Module[{},
  wordList = wordList /. {o} /. Options[italizeWord];
  If[! FreeQ[wordList, w],
    StyleBox[w, FontSlant -> "Italic"], w]
];
Options[italizeWord] = {
  wordList -> {"Mathematica", "J/Link", "MathLink", "DatabaseLink", "NETLink", "MathLM", "MonitorLM", "Combinatorica"}
  };


(* Return notebook expr containing search results *)
notebookSearchResult[s_List]:= 
Module[{header, cells, nbExpr, query, start, limit, suggestions, 
        searchTime, totalMatches, matches, indexLoc, resultInfo={}, resultSearchCountCell=" ",
        suggestionCell=" ", allResultsCount, allWolframSitesLine, debugButton = {}},
  query        = "Query"        /. s /. {"Query"->""};
  start        = "Start"        /. s /. {"Start"->1};
  limit        = "Limit"        /. s /. {"Limit"->"10"};
  searchTime   = "SearchTime"   /. s /. {"SearchTime"->0};
  totalMatches = "TotalMatches" /. s /. {"TotalMatches"->"Unknown"};
  suggestions  = "Suggestions"  /. s /. {"Suggestions"->Null};
  matches      = "Matches"      /. s /. {"Matches"->{}};
  indexLoc 	   = "IndexLocation" /. s /. {"IndexLocation"->None};
  
  allWolframSitesLine = Cell[BoxData[
      
      TagBox[PaneSelectorBox[{False->ButtonBox[ 
      	   RowBox[{Cell[StringJoin[localization["Try your search"], " ", localization["on all Wolfram sites"]], "SearchAllSites"], " ", newAllSitesIcon[]}], 
              BaseStyle->{"Hyperlink", FontColor->GrayLevel[0.5]}, 
              ButtonData->{URL["http://search.wolfram.com/?query="<>ExternalService`EncodeString[query]<>"&collection=tryonall"], None}
      ],
      True->ButtonBox[ 
      	   RowBox[{Cell[StringJoin[localization["Try your search"], " ", localization["on all Wolfram sites"]], "SearchAllSites"], " ", newAllSitesIcon["-over"]}], 
              BaseStyle->{"Hyperlink", FontColor->GrayLevel[0.5]}, 
              ButtonData->{URL["http://search.wolfram.com/?query="<>ExternalService`EncodeString[query]<>"&collection=tryonall"], None}
      ]},
      Dynamic[CurrentValue["MouseOver"]]], MouseAppearanceTag["LinkHand"]]
  ], "SearchAllSites"];
  Which[
    (* results found *)
    Length[matches] > 0, 
      resultSearchCountCell = 
        Cell[TextData[{
          ToString[start],
          " - ", 
          ToString[start + Length[matches] - 1],  
          " ", localization["of"], " ", 
          ToString[totalMatches], 
          " ", localization["for"], " ", 
          ButtonBox[ToString[query], 
             ButtonStyle -> "Link",
             ButtonData->query]
        }], "SearchCountCell"];
        suggestionCell = 
        If[suggestions =!= Null, 
          Cell[TextData[{
            localization["Did you mean"], ": ", 
            ButtonBox[suggestions, 
             ButtonStyle -> "Link",
             ButtonData->suggestions]
          }], "DidYouMean"], 
          " "
        ];
     ,
    (* no results found *)
    True, 
        resultInfo = {
          (* allWolframSitesLine, *)
          If[suggestions =!= Null, 
            Cell[TextData[{
              localization["Did you mean"], ": ", 
              ButtonBox[suggestions, 
                ButtonStyle -> "Link",
                ButtonData->suggestions]
            }], "DidYouMean"], 
            {}
          ],
          Cell[TextData[{
            warningSign[],
            localization[" Your search"], " - ", query, " - in ", 
            ButtonBox["Documentation Center", 
              BaseStyle->"Hyperlink",
              ButtonData->{URL["http://reference.wolfram.com/language"], None}
              ],
            localization[" did not match any documents"]
          }], "SearchCountCell", FontWeight->"Bold"],
          Cell[TextData[{
            localization["Suggestions"], "\n", 
            localization["\[FilledSmallSquare] Make sure all words are spelled correctly"],"\n", 
            localization["\[FilledSmallSquare] Try different keywords"], "\n", 
            localization["\[FilledSmallSquare] Try more general keywords"]        
          }], "SearchSuggestionsCell"]
        }
  ];
  cells = cellSearchResult/@matches;
  header = 
    Cell[BoxData[GridBox[{
      {
        Cell[TextData[{localization["Search Results"]}], "SearchPageHeading"],
        ItemBox[allWolframSitesLine,Alignment->{Right,Center}]
      },
      {
        resultSearchCountCell,
        "\[SpanFromAbove]"
      },
      If[suggestions=!=Null, 
      {
        suggestionCell,
        "\[SpanFromLeft]"
      },
      Nothing]
      }]], "SearchPageHeadingGrid"];
  (* pre-v12 *)
  (*header = 
    Cell[BoxData[GridBox[{
      {
        Cell[TextData[{localization["Search Results"]}], "SearchPageHeading"],
        resultSearchCountCell
      },
      {
        suggestionCell,
        allWolframSitesLine
      }
      }]], "SearchPageHeadingGrid"];*)
  If[
  	TrueQ[$UseTextSearchQ] && TrueQ[$DebugDocumentationSearchQ],
  	$devTextSearchOutput = TextSearch`PackageScope`devTextSearch[File[indexLoc], docQueryString[query]];
  	debugButton = Cell[
		  	BoxData[ButtonBox[
		  		"click here for more debugging information", 
		  		ButtonFunction :> CreateDocument[ExpressionCell[$devTextSearchOutput, "Output"]],
				Evaluator -> Automatic
			]],
			"SearchSuggestionsCell"
	  	]
  ];
  nbExpr = 
    Notebook[Flatten @ {
      header,
      resultInfo,
      debugButton,
      cells,
      searchPageLinkCell[query, totalMatches, start, limit, $NumberOfExtraPages]
    }, 
    StyleDefinitions->FrontEnd`FileName[{"Wolfram"}, "Reference.nb"],
    Saveable->False, 
    WindowTitle->"Search Results: " <> StringTake[query, Min[StringLength[query], 60]]];
  nbExpr
];

notebookSearchResult[f___]:= ($Failed; Message[notebookSearchResult::args, f];)
notebookSearchResult::args = "Incorrect arguments: `1`";   

(* Return a cell expr from a list : *)
(* {"Title", "Type", "Summary", "URI"} *)
Clear[cellSearchResult]
cellSearchResult[{title_String, type_String, summary_String, uri_String, description_String, context_String, score_:Null}]:= 
Module[{styledTitle, url=uri, typecolor},
styledTitle = title;
url = If[StringMatchQ[url, "note/*"], 
         dothtml @ StringJoin["http://reference.wolfram.com/mathematica/", url, ".html" ],
         "paclet:" <> url];
typecolor = Which[
      type === "Symbol", "SymbolColor",
      type === "Guide", "GuideColor", 
      MatchQ[type, "Workflow"|"Workflow Guide"], "WorkflowColor",
      MatchQ[type, "Tutorial"|"Monograph"|"Tech Note"|"Overview"], "TutorialColor",
      type === "Entity", "EntityColor",
      MatchQ[type, "Program"|"File"|"MathLink C Function"|"LibraryLink C Function"], "ProgramColor",
      
      True, "DefaultColor"];

(* TODO: fix missing score reporting *)

Cell[BoxData[TemplateBox[{
 StyleBox[If[type === "Character Name", 
 				StringReplace[styledTitle, {"\\"->"\\[Backslash]"}], 
 				styledTitle
 				],"SearchResultTitle"],
 StyleBox[ description , "SearchResultType"],
 If[StringLength[summary] > 0, 
   ItemBox[StyleBox[ summary, "SearchResultSummary"], ItemSize->Full],
   ""
 ],
 url}, "SearchResultCellTemplate", 
  BaseStyle->{
    GridBoxOptions->{FrameStyle->With[{color=typecolor},Quiet[Dynamic[CurrentValue[{StyleHints,color}]]]]}}]
    ], "SearchResultCell",
  Background->
      With[{colorbg=typecolor<>"BG"},Dynamic[If[CurrentValue["MouseOver"], CurrentValue[{StyleHints,colorbg}], Inherited]]],
  CellFrameMargins->Dynamic[If[CurrentValue["MouseOver"], {{0, 0}, {0, 0}}, Inherited]],
  GridBoxOptions->{GridBoxSpacings->Dynamic[If[CurrentValue["MouseOver"], {"Columns" -> {
     Offset[1.4], {
      Offset[0.5599999999999999]}, 
     Offset[0.]}, "Rows"->{Offset[1.],{Offset[1.]},Offset[1.4]}}, Inherited]]}]
 
(* pre-v12 *)
(*Cell[TextData[Flatten@{
 Cell[
 	If[StringMatchQ[url, ___ ~~ "tutorial/HandsOnFirstLook01" ~~ ___], 
 	TextData[
 		ButtonBox[styledTitle, 
 			BaseStyle->"Link", 
 			ButtonFunction:>(Documentation`HelpLookup["paclet:tutorial/HandsOnFirstLook01", FrontEnd`NotebookCreate[]]& ), 
 			Evaluator->Automatic] 
 			],
 	BoxData[
 		TemplateBox[{Cell[TextData[
 			If[type === "Character Name", 
 				StringReplace[styledTitle, {"\\"->"\\[Backslash]"}], 
 				styledTitle
 				]]],
 				url}, "SearchResultLink", BaseStyle->{"SearchResultTitle"}
 				]
 			]
 		], "SearchResultTitle"],
 " ", StyleBox["(", "SearchResultType"],
 StyleBox[ ItalizeWordsInString @ description , "SearchResultType"],
 StyleBox[")", "SearchResultType"],
 Sequence @@ If[
 	score =!= Null,
 	
 	{
 		"   ",
 		StyleBox["score: "<>ToString[score], FontColor -> Orange]
 	},
 	{}
 ],
 If[StringLength[summary] > 0, 
   "\n",
   ""
 ],
 StyleBox[ ItalizeWordsInString @ summary, "SearchResultSummary"]
}], "SearchResultCell", 
  CellDingbat-> StyleBox["\[FilledSquare]",
    Which[
      type === "Symbol", "SymbolColor",
      type === "Guide", "GuideColor", 
      True, "OtherColor"] ] ] *)];

cellSearchResult[f___]:= ($Failed; Message[cellSearchResult::args, f];)
cellSearchResult::args = "Incorrect arguments: `1`";   

searchPageLinkCell[query_String, totalResults_Integer, start_Integer, 
   limit_Integer, numPages_Integer] := 
  Module[{startPage, out = {}, currentPage = Ceiling[(start)/limit], 
    totalPages = Ceiling[totalResults/limit], prevnext={} },
   
   (*Check boundaries*)
   If[start > totalResults || totalResults < 0 || limit < 1 || numPages < 1 || totalResults <= limit, 
     Return[{}]
   ];
   
   (*determine starting page number*)
   If[Quotient[currentPage-1, numPages] === 0,    
     startPage = 1, 
     startPage = currentPage - numPages
   ];
   
   (*create range of pages*)
   For[i = startPage, i < currentPage + numPages + 1 && i <= totalPages, i++,
    If[i == currentPage,
      AppendTo[out, StyleBox[" "<>ToString[i]<>" ", FontColor -> GrayLevel[.133333333333`], FontWeight->"Bold"]];, 
      AppendTo[out, 
        TemplateBox[{" "<>ToString[i]<>" ", query <> " start:" <> ToString[(i - 1)*limit + 1] <> " limit:" <> ToString[limit]}, 
          "SearchResultPageLink", BaseStyle->"SearchResultPageLinks" 
         ]];
      ];
    ];
    
   (*add separator*)
   out = Riffle[out, "|"];
   
   (* Add first and last page *)
   If[startPage =!= 1, 
    out = Flatten@{TemplateBox[{"1", query <> " start:1 limit:" <> ToString[limit]}, 
        "SearchResultPageLink", BaseStyle->"SearchResultPageLinks"], " \[Ellipsis] ", out}];
   If[currentPage + numPages + 1 <= totalPages, 
    out = Flatten@{out, " \[Ellipsis] ", 
       TemplateBox[{ToString[totalPages], query <> " start:" <> ToString[(totalPages-1)*limit + 1] <> " limit:" <> ToString[limit]},
        "SearchResultPageLink", BaseStyle->"SearchResultPageLinks"]}];
   
   (*add prev and next *)
   If[currentPage =!= 1, 
    prevnext = Flatten@{
      TemplateBox[{localization["\[LeftGuillemet] PREVIOUS"], query <> " start:" <> ToString[(currentPage - 2) * limit + 1] <> " limit:" <> ToString[limit]}, 
        "SearchResultPageLink", BaseStyle->"SearchResultPageLinks"], prevnext}];
   If[currentPage < totalPages, 
    prevnext = Flatten@{prevnext,  
       TemplateBox[{localization["NEXT \[RightGuillemet]"], query <> " start:" <> ToString[currentPage * limit + 1] <> " limit:" <> ToString[limit]}, 
        "SearchResultPageLink", BaseStyle->"SearchResultPageLinks"]}];
   (*add separator*)
   prevnext = If[Length@prevnext > 1, Riffle[prevnext, " | "], prevnext];
   
   
   (*return cell expr*)
    Cell[BoxData[GridBox[{
      {
        Cell[BoxData[RowBox[Flatten@{prevnext}]], "SearchResultPageLinks", AutoSpacing->False],
        Cell[BoxData[RowBox[Flatten@{out}]], "SearchResultPageLinks", AutoSpacing->False]
      }
      }]], "SearchResultPageLinksGrid"]
 ];
 
(* html anchors need to be file.html# xxx, this handles old bug where they could be file # xxx.html *)
dothtml[s_String] :=
  If[StringMatchQ[s, "*#*"],
    (StringTake[s, #] <> ".html" <> StringDrop[s, #])&[StringPosition[s, "#"][[-1, -1]] - 1],
    s];


SetAttributes[ canonicalOptions, {Listable}];
canonicalOptions[name_Symbol -> val_] := SymbolName[name] -> val;
canonicalOptions[expr___] := expr;

(* Used to return a ShortenedSummary *)
shorten[s_String, maxLength_Integer] := Block[{str = s},
	If[
		StringLength[s] <= maxLength,
		Return[s]
	];
	str = StringTake[str, UpTo[maxLength]];
	str = With[
		{
			pos = StringPosition[str, " "]
		},
		If[
			!ListQ[pos] || pos === {},
			s,
			StringTake[str, Last[Last[pos]] - 1] <> "\[Ellipsis]"
		]
	];
	str
]
shorten[s_String, maxLength_] := s
shorten[___] = ""

(* The maximum length of a "ShortenedSummary" *)
$shortenedSummaryLength = 175;

(* Post-processing of the "Matches" returned by SearchDocumentation[] *)
$postProcessingFunctions = <|
	"ShortenedSummary" -> (Replace[shorten[#, $shortenedSummaryLength], " " -> ""]&)
|>;

quotedQ[s_String] := StringMatchQ[s, "\""~~__~~"\""]

$ignorePattern = "\\b(how\\s+((do\\s+(i|you))|to))|(what(\\s+(is|are|was)))\\b";
$optionalPattern = "\\b(((wolfram\\s+)?mathematica|(wolfram(\\s*language)?))|something)\\b";
preprocess[string_String] := Module[{s = string},
	s = StringDelete[s, RegularExpression[$ignorePattern], IgnoreCase -> True];
	s = StringReplace[s, RegularExpression[$optionalPattern] :> ("#" <> StringRiffle[StringSplit["$0"], " #"]), IgnoreCase -> True]
];

docQueryString[s_String] := SearchQueryString @ Which[
    	(* If it is a letter-free query, only match ShortNotations, as in "/@" *)
    	StringFreeQ[s, LetterCharacter|DigitCharacter],
    	"ShortNotations:\"" <> s <> "\"",
    	(* If it is a DirectHitSearch call, don't look at NormalizedTitle *)
    	!StringFreeQ[s, "ExactTitle:"],
    	s,
    	(* Make sure "almost exact" title matches are on top for e.g. "String Join" or "stringjoin" *)
    	True,
    	"(NormalizedTitle:" <> (If[quotedQ[#], #, "\""<>#<>"\""]&[StringDelete[ToLowerCase[s], WhitespaceCharacter..|"()"|"\""]]) <> ")^1000 OR (" <> preprocess[s] <> ")"
    ]


(* Localization of strings *)
localization[c___]:= c;

localization["Search Results"]:= localization["Search Results", ToString[$SearchLanguage] ]
localization["Search Results", language_String] :=
  Switch[language, 
    "Japanese", "\:691c\:7d22\:7d50\:679c",
    "ChineseSimplified", "\:641c\:7d22\:7ed3\:679c",
     _, "Search Results" ];
localization["Try your search"]:= localization["Try your search", ToString[$SearchLanguage] ]
localization["Try your search", language_String] :=
  Switch[language, 
    "Japanese", "\:691c\:7d22\:5bfe\:8c61\:ff1a",
    "ChineseSimplified", "\:5c1d\:8bd5\:60a8\:7684\:641c\:7d22\:ff1a",
     _, "Try your search" ];
localization["on all Wolfram sites"]:= localization["on all Wolfram sites", ToString[$SearchLanguage] ]
localization["on all Wolfram sites", language_String] :=
  Switch[language, 
    "Japanese", "\:3059\:3079\:3066\:306eWolfram\:30b5\:30a4\:30c8",
    "ChineseSimplified", "\:5728\:6240\:6709\:7684Wolfram \:7ad9\:70b9",
     _, "on all Wolfram sites" ];
localization["matches"]:= localization["matches", ToString[$SearchLanguage] ]
localization["matches", language_String] :=
  Switch[language, 
    "Japanese", "\:4ef6",
    "ChineseSimplified", "\:5339\:914d",
     _, "matches" ];
localization["Results"]:= localization["Results", ToString[$SearchLanguage] ]
localization["Results", language_String] :=
  Switch[language, 
    "Japanese", "\:7d50\:679c",
    "ChineseSimplified", "\:7ed3\:679c",
     _, "Results" ];
localization["of"]:= localization["of", ToString[$SearchLanguage] ]
localization["of", language_String] :=
  Switch[language, 
    "Japanese", "/",
    "ChineseSimplified", "/",
     _, "of" ];
localization["for"]:= localization["for", ToString[$SearchLanguage] ]
localization["for", language_String] :=
  Switch[language, 
    "Japanese", "\:691c\:7d22\:ff1a",
    "ChineseSimplified", "\:4f5c\:4e3a",
     _, "for" ];
localization["Did you mean"]:= localization["Did you mean", ToString[$SearchLanguage] ]
localization["Did you mean", language_String] :=
  Switch[language, 
    "Japanese", "\:3053\:306e\:5358\:8a9e\:3067\:3059\:304b\:ff1a",
    "ChineseSimplified", "\:60a8\:8ba4\:4e3a",
     _, "Did you mean" ];
localization["Your search"]:= localization["Your search", ToString[$SearchLanguage] ]
localization["Your search", language_String] :=
  Switch[language, 
    "Japanese", "\:691c\:7d22\:3055\:308c\:305f\:5358\:8a9e",
    "ChineseSimplified", "\:60a8\:7684\:641c\:7d22",
     _, "Your search" ];
localization["did not match any documents"]:= localization["did not match any documents", ToString[$SearchLanguage] ]
localization["did not match any documents", language_String] :=
  Switch[language, 
    "Japanese", "\:30de\:30c3\:30c1\:3057\:307e\:305b\:3093\:3067\:3057\:305f\:ff0e",
    "ChineseSimplified", "\:4e0d\:5339\:914d\:4efb\:4f55\:6587\:6863",
     _, "did not match any documents" ];
localization["Suggestions"]:= localization["Suggestions", ToString[$SearchLanguage] ]
localization["Suggestions", language_String] :=
  Switch[language, 
    "Japanese", "\:63d0\:6848",
    "ChineseSimplified", "\:5efa\:8bae",
     _, "SUGGESTIONS" ];
localization["Make sure all words are spelled correctly"]:= localization["Make sure all words are spelled correctly", ToString[$SearchLanguage] ]
localization["Make sure all words are spelled correctly", language_String] :=
  Switch[language, 
    "Japanese", "\:30b9\:30da\:30eb\:306e\:9593\:9055\:3044\:306f\:3042\:308a\:307e\:305b\:3093\:304b\:ff1f",
    "ChineseSimplified", "\:786e\:5b9a\:6240\:6709\:5b57\:7684\:62fc\:6cd5\:90fd\:6b63\:786e",
     _, "Make sure all words are spelled correctly" ];
localization["Try different keywords"]:= localization["Try different keywords", ToString[$SearchLanguage] ]
localization["Try different keywords", language_String] :=
  Switch[language, 
    "Japanese", "\:4ed6\:306e\:30ad\:30fc\:30ef\:30fc\:30c9\:3092\:304a\:8a66\:3057\:304f\:3060\:3055\:3044\:ff0e",
    "ChineseSimplified", "\:5c1d\:8bd5\:4e0d\:540c\:7684\:5173\:952e\:5b57",
     _, "Try different keywords" ];
localization["Try more general keywords"]:= localization["Try more general keywords", ToString[$SearchLanguage] ]
localization["Try more general keywords", language_String] :=
  Switch[language, 
    "Japanese", "\:4e00\:822c\:7684\:306a\:30ad\:30fc\:30ef\:30fc\:30c9\:3092\:304a\:8a66\:3057\:304f\:3060\:3055\:3044\:ff0e",
    "ChineseSimplified", "\:5c1d\:8bd5\:66f4\:591a\:901a\:7528\:7684\:5173\:952e\:5b57",
     _, "Try more general keywords" ];
localization["NEXT \[RightGuillemet]"]:= localization["NEXT \[RightGuillemet]", ToString[$SearchLanguage] ]
localization["NEXT \[RightGuillemet]", language_String] :=
  Switch[language, 
    "Japanese", "\:6b21\:3078",
    "ChineseSimplified", "\:4e0b\:4e00\:9875",
     _, "Next \[RightGuillemet]" ];
localization["\[LeftGuillemet] PREVIOUS"]:= localization["\[LeftGuillemet] PREVIOUS", ToString[$SearchLanguage] ]
localization["\[LeftGuillemet] PREVIOUS", language_String] :=
  Switch[language, 
    "Japanese", "\:623b\:308b",
    "ChineseSimplified", "\:4e0a\:4e00\:9875",
     _, "\[LeftGuillemet] Previous" ];


magnifyingGlass[]:=
Cell[GraphicsData["CompressedBitmap", "\<\
eJxVUk1vEkEY3rWlB436D8gSk8af0JignjQk8AdIOJDGpE1qlLaQoMA/4KZI
Qhq0bWI0XEwP6k2Kl2Js+IZtwZYFdoMW6IciSWUf3xm2tJ1kJrMz+7zPxzsO
9/Lco8fu5flZt/Rw0f10bn52SXrwZJGOJkRBEG7SvCEJbA/aGsskaAgTtAxh
DP4FYRK93ilarQ5kuYVarS2i3f4jCFeMW13/h/HQMQKZ0O2cQmkdoCQryOWq
CARCgohwOExsU8bfIypdP0OfjmpOodsdcEYGzmZrCISCMJvNIiySmfCmMY6x
kBpoWo/katoJgRXlAJWKgkymimCQA2Gx3BJ5bfLOiAyGI+zsqCiX6yiVftBU
UKCZz+8hl6mNwZJkZmLPCzBWBmbA7e0KXq++gd1ux/T0bdy9f4+5ZH4JLGFm
5g53SjGdFbhGkolZ1jg4GHwOl8uFVy8jeP/uLYKBZ8QocWaHwyFMjlGadsT1
JRIJ+LxLKBaL+PY9TdV1bG1tIRaLkVEL4vH4RZT6G/nSHnw+H76mkmg0GtTJ
XeSzORKlQ1VV7tRqtY5R1zkXs7ewsIDDbo/yOkSzqaC6s4tCoUD7JlZWVjjf
qBm68SJYQKwlstxANBqloFqcp9/v41f7J53LSKfTiEQisNls/M6ocB4QS1hV
j/Hx8ydsbn7BcDg0igxQr9dJWhlOpxNra2sjWvEi/ipPnHepeczjSiaT2N9X
MBj0kUql4PF4EAqFTGPC0eMFOp2/pP4EGxsfSOEL+P1+eL1erK+vX/Z5iVD8
Dwm2cTA=\
\>"], "Graphics",
 ImageSize->{15, 14},
 ImageMargins->3];

warningSign[]:=
Cell[GraphicsData["CompressedBitmap", "\<\
eJxVUstqIkEU7ba1e2ZgZn7Br5lZzReoCwkDCTiomex8g67cuApuRHcSCIK4
9IGIiogvFHwgMS4ERQJBRI2a8UzdshVT0HVvVZ9zz7lV9ct8d/37j/nu5sqs
/3lrtl3fXP3V/7Desi1JFAThG/u+6wXKwdLTJGG322GxWODl5YWtZrOZoGA6
nVJ2/ubzOV5fX7FerwUR7+/vjPsVy+WSIweDAVqtFprNJhqNxjnW63W+3+v1
MB6PqQJj7/d7xtah2+0iHA7D6/XC4/HA7XYLGrhcLkGG0+mkjPb4/0AggFwu
J2K73TKyhGg0ilQqhXK5zFbFYpHNpVKJMnXFZhE+n08kacbRIpPJIBKJ4Onp
Cc/PzxqMRiMG1Ol0MJlMfMX3ROTzeRF2u13EarViXBlvb2+wWq2oVCqcz2AU
tWqN4XCIYDCIWq3GOjwcDlyQRRQKBYRCIQIfBRXIsgyj0XiqgXQ6jfv7e04T
wQYn09hsNrDZbORHQ1Bm5JJMqn6/H+12W3PkHcMXVKvVkypHXrLo2GKxGMl9
YH2me4XFYkE2mz2zDAYD+v0+nSR3wLFqi5KqRndMN0R+FEXhSo+Pj7wWnQEf
B6i9HVkK/0M+yA/57HQ6cDgcmEwmJxXgn0rSqiQCUstk6eHhAYlE4lid+ZEu
ofK5m3g8zt8SvVQG06on/OnE+xDpkSeTSXr00rlT8eSGeOJ/ERVL1Q==\
\>"], "Graphics",
 ImageSize->{14, 15},
 ImageMargins->0];
 
newAllSitesIcon[arg_:""]:=
With[{file = "SearchAllSites"<>arg<>".png"}, Cell[BoxData[DynamicBox[
  FEPrivate`ImportImage[FrontEnd`FileName[{"Documentation", "Miscellaneous"}, file]]]
  ]
  ]];

(* English symbols *)
(* TODO: find out if/where these are used, and either kill or update! *)
englishSymbols = 
{"abort", "above", "abs", "accumulate", "accuracy", "active", \
"after", "alias", "alignment", "all", "alpha", "alternatives", \
"analytic", "and", "animate", "animator", "annotation", "apart", \
"appearance", "append", "application", "apply", "array", "arrow", \
"arrowheads", "assuming", "assumptions", "attributes", "automatic", \
"axes", "axis", "back", "background", "backslash", "backward", \
"band", "baseline", "because", "beep", "before", "begin", "below", \
"beta", "binary", "binomial", "bit", "black", "blank", "blend", \
"block", "blue", "bold", "bookmarks", "bottom", "bounds", "box", \
"boxed", "boxes", "break", "brown", "button", "byte", "cancel", \
"cap", "cases", "catch", "ceiling", "cell", "center", "character", \
"characters", "check", "chop", "circle", "clear", "clip", "clock", \
"close", "closed", "coefficient", "collect", "colon", "column", \
"commonest", "compile", "compiled", "complement", "complex", \
"complexes", "compose", "composition", "compress", "condition", \
"congruent", "conjugate", "connect", "constant", "constants", \
"context", "contexts", "continuation", "continue", "contours", \
"copyable", "correlation", "cosh", "cot", "count", "covariance", \
"cross", "cuboid", "cup", "cyan", "cylinder", "darker", "dashed", \
"dashing", "date", "debug", "decimal", "decompose", "decrement", \
"default", "defer", "definition", "degree", "deletable", "delete", \
"delimiter", "delimiters", "denominator", "deploy", "deployed", \
"depth", "derivative", "diagonal", "dialog", "diamond", \
"differences", "dimensions", "direction", "directive", "directory", \
"discriminant", "disk", "dispatch", "display", "distribute", \
"divide", "dividers", "divisible", "divisors", "do", "document", \
"dot", "dotted", "down", "drop", "dump", "dynamic", "edit", \
"editable", "eigenvalues", "element", "eliminate", "empty", \
"enabled", "encode", "end", "enter", "environment", "equal", \
"equilibrium", "evaluate", "evaluated", "evaluator", "except", \
"exclusions", "exists", "exit", "expand", "exponent", "export", \
"expression", "extension", "extract", "factor", "factorial", "fail", \
"false", "file", "filling", "find", "first", "fit", "fits", "flat", \
"flatten", "floor", "fold", "font", "for", "format", "forward", \
"frame", "framed", "front", "full", "function", "gamma", "general", \
"generic", "get", "give", "glow", "gradient", "graphics", "gray", \
"greater", "green", "grid", "hash", "head", "heads", "hessian", \
"hold", "holdall", "homepage", "horizontal", "hue", "hyperlink", \
"hyphenation", "identity", "if", "implies", "import", "in", \
"increment", "indent", "indeterminate", "inequality", "infinity", \
"infix", "information", "inherited", "initialization", "inner", \
"input", "insert", "inset", "install", "integer", "integers", \
"integral", "integrate", "interactive", "interlaced", \
"interpolation", "interpretation", "interrupt", "intersection", \
"interval", "inverse", "invisible", "italic", "item", "join", \
"joined", "label", "labeled", "language", "large", "larger", "last", \
"launch", "left", "length", "less", "level", "lexicographic", \
"lighter", "lighting", "limit", "line", "links", "list", "listen", \
"literal", "locked", "log", "longest", "magenta", "magnification", \
"magnify", "manipulate", "manipulator", "manual", "map", "mat", \
"maximize", "mean", "median", "medium", "mesh", "message", \
"messages", "metacharacters", "method", "minimize", "minors", \
"minus", "missing", "mod", "modal", "mode", "modular", "module", \
"modulus", "momentary", "monitor", "most", "multiplicity", "names", \
"nearest", "needs", "negative", "nest", "next", "none", "nor", \
"norm", "normal", "normalize", "not", "notebook", "notebooks", \
"null", "number", "numerator", "off", "offset", "on", "opacity", \
"open", "opener", "operate", "option", "optional", "options", "or", \
"orange", "order", "ordering", "orderless", "out", "outer", "over", \
"overflow", "overlaps", "pane", "panel", "paneled", "parameter", \
"parenthesize", "part", "partition", "path", "pattern", "pause", \
"permutations", "pi", "pick", "piecewise", "pink", "pivoting", \
"placeholder", "plain", "play", "plot", "plus", "ply", "point", \
"polygamma", "polygon", "polynomials", "position", "positive", \
"postscript", "power", "ppm", "precedence", "precedes", "precision", \
"prefix", "previous", "prime", "primes", "print", "product", \
"projection", "proportion", "proportional", "protect", "protected", \
"pseudoinverse", "purple", "put", "quartics", "quartiles", "quiet", \
"quit", "quotient", "random", "range", "raster", "rational", \
"rationalize", "rationals", "raw", "re", "read", "real", "reap", \
"record", "rectangle", "red", "reduce", "refine", "refresh", \
"reinstall", "release", "remove", "removed", "repeated", "replace", \
"rescale", "residue", "resolve", "rest", "resultant", "return", \
"reverse", "rib", "riffle", "right", "root", "roots", "rotate", \
"round", "row", "rule", "run", "save", "scale", "scaled", "scan", \
"select", "selectable", "selection", "sequence", "series", "set", \
"setbacks", "setter", "setting", "shading", "shallow", "share", \
"short", "shortest", "show", "sign", "signature", "simplify", "sin", \
"skeleton", "skewness", "skip", "slider", "slot", "small", "smaller", \
"socket", "solve", "sort", "sorted", "sound", "sow", "space", \
"spacer", "spacings", "span", "sphere", "splice", "split", "square", \
"stack", "standardize", "star", "streams", "string", "stub", "style", \
"subscript", "subset", "subsets", "subtract", "succeeds", "sum", \
"superscript", "superstar", "switch", "symbol", "syntax", "tab", \
"table", "take", "tally", "tan", "tar", "temporary", "text", \
"therefore", "thick", "thickness", "thin", "thread", "through", \
"throw", "ticks", "tiff", "tilde", "times", "timezone", "timing", \
"tiny", "together", "toggle", "toggler", "tolerance", "top", "total", \
"trace", "translate", "transpose", "trig", "trigger", "true", "tube", \
"underflow", "underlined", "unequal", "unevaluated", "union", \
"unique", "unitize", "unset", "up", "update", "upset", "using", \
"value", "variables", "variance", "verbatim", "verbose", "version", \
"vertical", "viewpoint", "visible", "wedge", "which", "while", \
"white", "window", "with", "word", "write", "xor", "yellow", "zeta", \
"zip"};

End[];
 
EndPackage[];
