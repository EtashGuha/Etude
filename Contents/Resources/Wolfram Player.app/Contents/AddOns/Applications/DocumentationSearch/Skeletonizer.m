BeginPackage["DocumentationSearch`Skeletonizer`"]

Skeletonize::usage = "Skeletonize[dir] scraps the nb files in the given directory and outputs an association";

Begin["`Private`"]

$DebugSkeletonizerQ::usage = "Show debugging information for the Skeletonizer used for Documentation search.";
debugPrint := If[TrueQ[$DebugSkeletonizerQ], Print, Nothing];

(* TODO:

- Detect when there are new nb types not seen before so that we can warn the
  person who monitors this code that they might need to add support for a new
  notebook type.
  - Write instructions for how to add support for a new notebook type.
*)

(*!
	\function Skeletonize
	
	\calltable
		Skeletonize[documentationDir] '' given the documentation directory, this function produces associations with key/value pairs for each WL notebook that should be included in the WL documentation search.
		Skeletonize[] '' assuming the default WL installation directory, this function produces associations with key/value pairs for each WL notebook that should be included in the WL documentation search.

	Examples:
	
	Skeletonize[]
	
	Skeletonize[$InstallationDirectory]
	
	\maintainer chiarab
*)
Options[Skeletonize] =
{
	"Percentage" -> All,							(*< when testing, a percentage of notebooks can be processed. *)
	"UseWolframLanguageData" -> True,				(*< Whether to use WolframLanguageData for data that only lives there, like "OtherComputerLanguageEquivalents" *) "WolframLanguageDataCacheFile" -> Automatic,	(*< The cache file to use for caching WolframLanguageData. *)
	"WolframLanguageDataCacheTimeSpan" -> None,		(*< If there's an existing download of WolframLanguageData, how old is too old for using that cached copy? *)
	"StoreWolframLanguageDataInCache" -> True		(*< After downloading the WolframLanguageData, should we store it in a cache file? *)
};

Skeletonize::invdir = "Invalid file or directory: `1`";
Skeletonize::fnfnd = "No .nb files were found among the provided files or in the provided directories.";
Skeletonize::nvldprc = "Invalid value for the \"Percentage\" option: `1`";
(* Skeletonize a single directory or file *)
Skeletonize[dir:(_String|_File), opts:OptionsPattern[]] := Which[
	!FileExistsQ[dir],
	Message[Skeletonize::invdir, dir],
	!DirectoryQ[dir],
	(* it's actually a single file *)
	Skeletonize[{dir}, opts],
	True,
	Skeletonize[FileNames["*.nb", dir, Infinity], opts]
];
(* Skeletonize a list of directories or files *)
Skeletonize[filesIn:List[(_String|_File)..], opts:OptionsPattern[]] := 
	Module[{files = filesIn, allFiles, numFiles, classes = skeletonizerClasses[], thisClass, thisClassAssoc, res, referredBy, referredByGraph, skel, frequencyAssoc, perc = OptionValue["Percentage"]},

		files = Select[Replace[filesIn, File[path_] :> path, {1}], FileExistsQ];
		
		If[files === {}, Message[Skeletonize::invdir, files];Return[$Failed]];
		
		allFiles = 
			Join[
				Select[
					files,
					MatchQ[FileExtension[#], "nb"|"NB"]&
				],
				FileNames[
					"*.nb"|"*.NB", 
					Select[files, DirectoryQ], 
					Infinity
				]
			];
		If[
			!ListQ[allFiles] || allFiles === {},
			Message[Skeletonize::fnfnd];
			Return[$Failed]
		];
		
		If[
			!NumericQ[perc] && !MatchQ[perc, All]
			,
			Message[Skeletonize::nvldprc, perc];
			Return[$Failed]
			,
			If[
				!MatchQ[OptionValue["Percentage"], All | 100 | 100.],
				numFiles = Ceiling[OptionValue["Percentage"] / 100 * Length[allFiles]];
				allFiles = Take[allFiles, UpTo[numFiles]]; 
			]
		];
		allFiles = KeyTake[
			GroupBy[
				allFiles, 
				filePathToNotebookClass
			],
			Keys[classes]
		];
		
		$ConvertToStringFailures = {};
		$frequencyAssociation = <||>;
		
		{res, referredByGraph} = Reap[
			Function[{thisClassItem},
				
				thisClassAssoc = None;
				
				Switch[
					thisClassItem,
					
					_Rule,
					thisClass = thisClassItem[[1]];
					thisClassAssoc = thisClassItem[[2]];
					,
					_String,
					thisClass = thisClassItem;
				];
				
				files = allFiles[thisClass];
				
				If [!MissingQ[files],
					{skel, frequencyAssoc} = Reap[
						skeletonizeFiles[
							thisClass,
							thisClassAssoc,
							files,
							Sequence @@ FilterRules[{opts}, Options[skeletonizeFiles]]
						],
						"FrequencyAssociation"
					];
					
					If[
						MatchQ[frequencyAssoc, {{_Association}}],
						$frequencyAssociation = Join[$frequencyAssociation, frequencyAssoc[[1,1]]];
					];
					
					DeleteCases[
						skel,
						<||>
					]
					,
					{}
				]
			] /@ classes
		,
		"ReferredBy"
		];
		
		referredByGraph = Graph[Flatten[referredByGraph]];
		Scan[
			(referredBy[#] = DeleteCases[VertexInComponent[referredByGraph, #, 1], #])&,
			VertexList[referredByGraph]
		];
		referredBy[_] = {};
		
		res = Join @@ res;
		
		res = Map[
			Append[#, "ReferredBy" -> Flatten[referredBy[#["Title"]]]]&,
			res
		];
		
		res
	];
	
Skeletonize[{}, ___] = {};

(* The "Styles" and "Metadata" that are common to all nb classes. *)
$commonStyles = <|
	"SeeAlso" -> <|
		"Field" -> "SeeAlso",
		"Transformation" -> splitSeeAlso
	|>
|>;
$commonMetadata = <|
	"title" -> <|
		"Field" -> "ExactTitle", 
		"Transformation" -> StringTrim
	|>,
	"keywords" -> <|"Field" -> "Keywords"|>,
	"synonyms" -> <|
		"Field" -> "Synonyms",
		"Transformation" -> deleteNumbers
	|>,
	"summary" -> <|"Field" -> "Description"|>,
	"label" -> <|"Field" -> "DisplayedCategory"|>,
	"uri" -> <|"Field" -> "URI"|>,
	"type" -> <|"Field" -> "NotebookType"|>,
	"context" -> <|"Field" -> "Context"|>,
	"status" -> <|
		"Field" -> "NotebookStatus",
		"Transformation" -> transformNotebookStatus
	|>,
	"paclet" -> <|"Field" -> "PacletName"|>,
	"language" -> <|"Field" -> "Language"|>
|>;

(*!
	\function skeletonizerClasses
	
	\calltable
		skeletonizerClasses[documentationDir] '' returns the classes of nb's that are indexed.
	
	For each index, we can define the cell styles we're interested
	in extracting. And for each style, we can indicate what field
	we want to associate those cell contents with.
	
	We can also supply a 'Transformation' function to modify string
	values.
	
	\related 'Skeletonize
	
	\maintainer chiarab
*)
skeletonizerClasses[] :=
	{
		(* "Symbols" must be on top so that they are processed before other classes *)
		(* in order to get a "Frequency" field for pages that have a "LinkedSymbols" field. *)
		"Symbols" -> 
		<|
			"Styles" ->
			<|
				"Usage" -> <||>, (* currently a duplicate of the "Description" field *)
				"Notes" -> <|"Field" -> "Text"|>,
				"ExampleText" -> <||>,
				"FunctionEssay" -> <||>,
				"3ColumnTableMod" -> <|"Field" -> "TableText"|>,
				"2ColumnTableMod" -> <|"Field" -> "TableText"|>
			|>
		|>,
		"C" ->
		<|
			"Styles" ->
			<|
				"Usage" -> <||>,
				"Notes" -> <|"Field" -> "Text"|>
			|>
		|>,
		"Callbacks" -> 
		<|
			"Styles" -> 
			<|
				"Usage" -> <||>,
				"Notes" -> <|"Field" -> "Text"|>,
				"ExampleText" -> <||>
			|>
		|>,
		"Characters" ->
		<|
			"Styles" ->
			<|
				"CharacterImage" -> <||>,
				"Notes" -> <|"Field" -> "Text"|>
			|>
		|>,
		"Classifiers" -> 
		<|
			"Usage" -> <||>,
			"Notes" -> <|"Field" -> "Text"|>,
			"ExampleText" -> <||>
		|>,
		"CompiledTypes" -> 
		<|
			"Usage" -> <||>,
			"Notes" -> <|"Field" -> "Text"|>,
			"ExampleText" -> <||>
		|>,
		"CSharp" -> 
		<|
			"Usage" -> <||>,
			"Notes" -> <|"Field" -> "Text"|>,
			"ExampleText" -> <||>
		|>,
		"Devices" ->
		<|
			"Styles" ->
			<|
				"DeviceSubtitle" -> <|"Field" -> "Title"|>,
				"DeviceAbstract" -> <|"Field" -> "Abstract"|>,
				(*"DeviceSubsection" -> <||>,*)
				"ExampleText" -> <||>,
				"DeviceNotes" -> <|"Field" -> "Text"|>,
				"Usage" -> <||>,
				"3ColumnTableMod" -> <|"Field" -> "TableText"|>
			|>
		|>,
		"EmbeddingFormats" -> 
		<|
			"Styles" ->
			<|
				"Usage" -> <||>,
				"Notes" -> <|"Field" -> "Text"|>,
				"ExampleText" -> <||>
			|> 
		|>,
		"Entities" ->
		<|
			"Styles" ->
			<|
				"EntityAbstract" -> <|"Field" -> "Abstract"|>,
				"EntityUsage" -> <|"Field" -> "Usage"|>,
				"Notes" -> <|"Field" -> "Text"|>,
				"ExampleText" -> <||>,
				"2ColumnTableMod" -> <|"Field" -> "TableText"|>
			|>
		|>,
		"ExamplePages" ->
		<|
			"Styles" ->
			<|
				"FeaturedExampleDetail" -> <|"Field" -> "Text"|>,
				"FeaturedExampleText" -> <|"Field" -> "Text"|>
			|>
		|>,
		"ExternalEvaluationSystems" -> 
		<|
			"Usage" -> <||>,
			"Notes" -> <|"Field" -> "Text"|>,
			"3ColumnTableMod" -> <|"Field" -> "TableText"|>
		|>,
		"Files" ->
		<|
			"Styles" ->
			<|
				"Usage" -> <||>,
				"Notes" -> <|"Field" -> "Text"|>
			|>
		|>,
		"Formats" ->
		<|
			"Styles" ->
			<|
				"ExampleText" -> <||>,
				"3ColumnTableMod" -> <|"Field" -> "TableText"|>,
				"2ColumnTableMod" -> <|"Field" -> "TableText"|>,
				"FormatUsage" -> <|"Field" -> "Usage"|>,
				"FormatNotes" -> <|"Field" -> "Text"|>
			|>
		|>,
		"FrontEndObjects" ->
		<|
			"Styles" ->
			<|
				"Usage" -> <||>,
				"Notes" -> <|"Field" -> "Text"|>
			|>
		|>,
		"Guides" ->
		<|
			"Styles" ->
			<|
				"GuideAbstract" -> <|"Field" -> "Abstract"|>,
				"GuideFunctionsSubsection" ->
				<|
					"Field" -> "FunctionsSubsection",
					"Transformation" -> removeRightGuillemet
				|>,
				"GuideText" -> 
				<|
					"Field" -> "Text"
				|>,
				"InlineFunctionSans" -> <| 
					(* TODO: maybe only the ones which have this guide page as the main guide page / only those that are alone on a line *)
					"Field" -> "LinkedSymbols",
					"Transformation" -> filterSymbols
				|>
			|>
		|>,
		"HowTos" ->
		<|
			"Styles" ->
			<|
				"HowToAbstract" -> <|"Field" -> "Abstract"|>,
				"HowToText" -> <|"Field" -> "Text"|>
			|>
		|>,
		"Indicators" ->
		<|
			"Styles" ->
			<|
				"IndicatorUsage" -> <|"Field" -> "Usage"|>,
				"ExampleText" -> <||>
			|>
		|>,
		"Interpreters" ->
		<|
			"Styles" ->
			<|
				"InterpreterUsage" -> <|"Field" -> "Usage"|>,
				"InterpreterNotes" -> <|"Field" -> "Text"|>,
				"ExampleText" -> <||>,
				"2ColumnTableMod" -> <|"Field" -> "TableText"|>(*,
				"RelatedInterpreters" -> <|
					"Field" -> "RelatedInterpreters",
					"Transformation" -> splitSeeAlso
				|>*)(* do we want these? They are a lot and they tend to dirty the results a lot, e.g. "city" *)
			|>
		|>,
		"MenuItems" ->
		<|
			"Styles" ->
			<|
				"Usage" -> <||>,
				"Notes" -> <|"Field" -> "Text"|>
			|>
		|>,
		"Messages" ->
		<|
			"Styles" ->
			<|
			|>
		|>,
		"Methods" ->
		<|
			"Styles" ->
			<|
				"Notes" -> <|"Field" -> "Text"|>,
				"ExampleText" -> <||>,
				"3ColumnTableMod" -> <|"Field" -> "TableText"|>
			|>
		|>,
		"NetDecoders" -> 
		<|
			"Styles" ->
			<|
				"Notes" -> <|"Field" -> "Text"|>,
				"ExampleText" -> <||>,
				"2ColumnTableMod" -> <|"Field" -> "TableText"|>
			|>
		|>,
		"NetEncoders" -> 
		<|
			"Styles" ->
			<|
				"Notes" -> <|"Field" -> "Text"|>,
				"ExampleText" -> <||>,
				"2ColumnTableMod" -> <|"Field" -> "TableText"|>,
				"3ColumnTableMod" -> <|"Field" -> "TableText"|>
			|>
		|>,
		"Predictors" -> 
		<|
			"Usage" -> <||>,
			"Notes" -> <|"Field" -> "Text"|>,
			"ExampleText" -> <||>
		|>,
		"Programs" ->
		<|
			"Styles" ->
			<|
				"Usage" -> <||>,
				"Notes" -> <|"Field" -> "Text"|>,
				"2ColumnTableMod" -> <|"Field" -> "TableText"|>
			|>
		|>,
		"Services" ->
		<|
			"Styles" ->
			<|
				"ServiceAbstract" -> <|"Field" -> "Abstract"|>,
				"ServiceNotes" -> <|"Field" -> "Text"|>,
				"3ColumnTableMod" -> <|"Field" -> "TableText"|>,
				"GuideText" -> <|"Field" -> "Text"|>,
				"ExampleText" -> <||>
			|>
		|>,
		"Tutorials" ->
		<|
			"Styles" ->
			<|
				"TableText" -> <||>,
				"DefinitionBox" -> <|
					"Field" -> "LinkedSymbols",
					"Transformation" -> extractSymbols
				|>,
				"MathCaption" -> <||>,
				"Text" -> <||>,
				"Caption" -> <||>
			|>
		|>,
		"Widgets" ->
		<|
			"Styles" ->
			<|
				"Usage" -> <||>,
				"Notes" -> <|"Field" -> "Text"|>,
				"ExampleText" -> <||>
			|>
		|>,
		"WorkflowGuides" ->
		<|
			"Styles" ->
			<|
				"WorkflowGuideEntry" -> <|"Field" -> "Text"|>,
				"WorkflowGuideSection" -> <|"Field" -> "Abstract"|>
			|>
		|>,
		"Workflows" ->
		<|
			"Styles" ->
			<|
				"WorkflowDescription" -> <|"Field" -> "Abstract"|>,
				"WorkflowStep" -> <|"Field" -> "Abstract"|>,
				"Text" -> <||>,
				"WorkflowNotesText" -> <|"Field" -> "Text"|>,
				"NotesText" -> <|"Field" -> "Text"|>,
				"WorkflowParenthetical" -> <|"Field" -> "ExampleText"|>
			|>
		|>
	}

(*!
	\function skeletonizeFiles
	
	\calltable
		skeletonizeFiles[class, classAssoc, files] '' runs the skeletonizer on the given files.
		
	class:		The type of notebook, such as "Symbols", "Guides", etc.
	classAssoc:	Parameters wrt how to process a nb of a given class can be supplied in this Association. Can be None.

	Examples:
	
	skeletonizeFiles[
		"Symbols",
		<||>,
		{
		FileNameJoin[{$InstallationDirectory, "Documentation", "English", "System", "ReferencePages", "Symbols", "Plot.nb"}],
		FileNameJoin[{$InstallationDirectory, "Documentation", "English", "System", "ReferencePages", "Symbols", "Reverse.nb"}]
		},
		"WolframLanguageDataCacheTimeSpan" -> Quantity[0, "Days"]
	];
	
	\related 'Skeletonize
	
	\maintainer chiarab
*)
Options[skeletonizeFiles] =
{
	
	"UseWolframLanguageData" -> True,				(*< Whether to use WolframLanguageData for data that only lives there, like "Frequencies" *)
	"WolframLanguageDataCacheFile" -> Automatic,	(*< The cache file to use for caching WolframLanguageData. *)
	"WolframLanguageDataCacheTimeSpan" -> None,		(*< If there's an existing download of WolframLanguageData, how old is too old for using that cached copy? *)
	"StoreWolframLanguageDataInCache" -> True		(*< After downloading the WolframLanguageData, should we store it in a cache file? *)
};

skeletonizeFiles::msgs = "Messages were issued while processing the notebook `1`";
skeletonizeFiles[class_, classAssoc_, files_List, opts:OptionsPattern[]] :=
	Block[{styleAssoc, styles, processingParams, metadataAssoc, metadata, nbRes, messageFlag, res, wld, symbols, data, missingIndices, locations = files, missingMD},
		
	styleAssoc = Join[Replace[classAssoc["Styles"], _Missing :> <||>], $commonStyles];
	styles = Alternatives @@ Keys[styleAssoc];
	processingParams = DeleteCases[styleAssoc, <||>];
	
	metadataAssoc = Join[$commonMetadata, Replace[classAssoc["Metadata"], _Missing :> <||>]];
	metadata = Alternatives @@ Keys[metadataAssoc];
	processingParams = Join[processingParams, DeleteCases[metadataAssoc, <||>]];
	
	{res, missingMD} = Reap[
		DeleteCases[
			Function[{file},
				messageFlag =
					Check[
						nbRes = Replace[notebookExtractViaStylesAndMetadata[file, styles, metadata, processingParams], Except[_Association] :> <||>],
						$MessageFlag
					];
				If [messageFlag === $MessageFlag,
					Message[
						skeletonizeFiles::msgs,
						file
					];
				];
				nbRes
			] /@ files,
			None
		],
		"missingMD"
	];
	
	If[
		(* we only have WLData for Symbols *)
		class === "Symbols" && TrueQ[OptionValue["UseWolframLanguageData"]]
		,
		symbols = If[!StringFreeQ[#, FileNameJoin[{"System","ReferencePages","Symbols"}]], FileBaseName[#], "notAnEntity"]& /@ files;
		
		wld =
			getWolframLanguageData[
				"Symbols" -> If [Length[symbols] < 100, DeleteDuplicates[symbols], All],
				Sequence @@ FilterRules[{opts}, Options[getWolframLanguageData]]
			];
		
		data = Lookup[wld, symbols];
		
		missingIndices =
			DeleteDuplicates[
				Join[
					Position[data, Missing["UnknownEntity", _]],
					Position[data, Missing["KeyAbsent", _], {1}]
				][[All, 1]]
			];
		
		If [missingIndices =!= {},
			data = ReplacePart[data, Partition[missingIndices, 1] -> <||>];
		];
		
		If[
			MemberQ[Keys[getWldProperties[]], "Frequencies"],
			Sow[Association[Rule@@@Transpose[{symbols, data[[All,"Frequency"]]}]], "FrequencyAssociation"]
		];
		
		res = MapThread[Join, {res, data}];
	];
	
	(* e.g. "#1" or "/:" are stored in the "synonyms" metadata, not in the "ShortNotations" WLDProperty *)
	res = Map[
		Function[
			assoc,
			Block[{
				tmp = assoc,
				specialInputForms = If[!MatchQ[assoc["Synonyms"], {__String}], {}, Select[assoc["Synonyms"], !StringFreeQ[#, Except[LetterCharacter|DigitCharacter|WhitespaceCharacter|"'"|"-"|"$"]]&]],
				shortNot = res["ShortNotations"]
			},
				tmp["ShortNotations"] = Join[If[!ListQ[shortNot], {}, shortNot], specialInputForms];
				tmp
			]
		],
		res
	];
	
	If[
		missingMD =!= {},
		debugPrint["Missing metadata for the following documents of class "<>class<>": ", Flatten[missingMD]]
	];
	
	res
	
	];

(* The WolframLanguageData properties we're interested in. *)
$wolframLanguageDataProperties =
{
	"AutocompletionPhrases",
	
	(*"DocumentationBasicExamplesCompressed",*)
	(*"",*)
	
	(* How often this symbol/function is used in the WolframLanguage.
	   Used to compute a static weight for the document to help with
	   ordering search results. *)
	"Frequencies" ->
	<|
		"Field" -> "Frequency",
		"TransformAllFuncton" -> transformFrequenciesToWeight
	|>,
	
	(*"Keywords",
	"Name" ->
	<|
		"Field" -> "Title"
	|>,*)
	"Options" ->
	<|
		(* Some properties, such as "Options", need a function applied to their
		   values to transform them into something to be stored in the index. The
		   "TransformAllFuncton" is passed a list of values, one per
		   WolframLanguageData entry, and returns a modified list. *)
		"TransformAllFuncton" -> transformOptions
	|>,
	(* not exposed in the WL since August 2016 *)
	(*"OtherComputerLanguageEquivalents" ->
	<|
		"TransformAllFuncton" -> transformOtherComputerLanguageEquivalents
	|>,*)
	(*"PlaintextUsage",*)(* replaced with the "summary" metadata *)
	"PossibleAlternateNames",
	"ShortNotations"(*,
	"Synonyms",
	"TextStrings" -> 
	<|
		"Field" -> "TextStrings",
		(* The first entry in "TextStrings" is the usage message, which we already have - with better formatting - in the "PlaintextUsage" property *)
		"TransformAllFuncton" -> Function[{all}, Replace[all, strings_List :> If[Length[strings]>=1, Rest[strings], strings], {1}]]
	|>,
	"URL" ->
	<|
		"TransformAllFuncton" -> Function[{all}, Replace[all, Hyperlink[inner_] :> inner, {1}]]
	|>*)
	
	(* Won't index this for now. *)
	(*"Translations"*)
};

(*!
	\function downloadWolframLanguageData
	
	\calltable
		downloadWolframLanguageData[] '' downloads WolframLanguageData. See also the WolframLanguageDataCacheFile option to Sekeletonize, etc.
	
	\related 'skeletonizeFiles 'WolframLanguageData
	
	\maintainer chiarab
*)
Options[downloadWolframLanguageData] =
{
	"Symbols" -> All				(*< The symbols to download. *)
};
downloadWolframLanguageData[OptionsPattern[]] :=
	Block[{properties, data},
		
		properties = getWldProperties[];
		
		WolframLanguageData[
			OptionValue["Symbols"],
			Keys[properties],
			"EntityPropertyAssociation"
		]
	];

(*!
	\function getWolframLanguageData
	
	\calltable
		getWolframLanguageData[] '' gets the WolframLanguageData, possibly using a local cache.
	
	\related 'downloadWolframLanguageData
	
	\maintainer chiarab
*)
Options[getWolframLanguageData] =
{
	"Symbols" -> All,								(*< The symbols to download. *)
	"WolframLanguageDataCacheFile" -> Automatic,	(*< The cache file to use for caching WolframLanguageData. *)
	"WolframLanguageDataCacheTimeSpan" -> None,		(*< If there's an existing download of WolframLanguageData, how old is too old for using that cached copy? *)
	"StoreWolframLanguageDataInCache" -> True		(*< After downloading the WolframLanguageData, should we store it in a cache file? *)
};
getWolframLanguageData[opts:OptionsPattern[]] :=
	Block[{data, file, properties},
		
		disableEntityFrameworkBoxes[];
		
		file = OptionValue["WolframLanguageDataCacheFile"];
		
		If [file === Automatic,
			file = $wolframLanguageDataCacheFile;
		];
		
		file = StringDrop[file, -2] <> "." <> ToString[Hash[OptionValue["Symbols"]]] <> ".m";
		If [OptionValue["WolframLanguageDataCacheFile"] === None ||
			fileTooOld[file, OptionValue["WolframLanguageDataCacheTimeSpan"]],
			
			debugPrint["Downloading WolframLanguageData..."];
			data = downloadWolframLanguageData[Sequence @@ FilterRules[{opts}, Options[downloadWolframLanguageData]]];
			
			If [TrueQ[OptionValue["StoreWolframLanguageDataInCache"]],
				Print["Saving WolframLanguageData..."];
				Put[data, file];
			];
			,
			debugPrint["Reading WolframLanguageData from disk..."];
			data = Get[file];
			(* Make sure that the we don't get unwanted properties from the cache *)
			properties = getWldProperties[];
			data = KeyTake[#, Keys[properties]]&/@data;
		];
		
		processWolframLanguageData[data]
	];

(* A file to cache WolframLanguageData in. *)
$wolframLanguageDataCacheFile := FileNameJoin[{$UserBaseDirectory, "WolframLanguageDataSkeletonizerCache.m"}]

(*!
	\function fileTooOld
	
	\calltable
		fileTooOld[file, maxAge] '' returns True if the cache file either doesn't exist or is too old.
	
	\maintainer chiarab
*)
fileTooOld[file_String, HoldPattern[maxAge_Quantity] | None] :=
	Which[
		!FileExistsQ[file],
		True
		,
		maxAge === None,
		False
		,
		True,
		DateDifference[FileDate[file], Date[], "Second"] > maxAge
	]

(*!
	\function processWolframLanguageData
	
	\calltable
		processWolframLanguageData[data] '' process raw WolframLanguageData, applying transformation functions where required, etc.

	Unit tests: processWolframLanguageData.mt

    Example:

    processWolframLanguageData[
        {Association["Name" -> "MySymbol", "Keywords" -> {"apple", "orange"}]}
    ]

    ===

    AssociationThread[
        {"Keywords"},
        Transpose[
            AssociationThread[
                Keys[{{"MySymbol", {"apple", "orange"}}}],
                Values[{{"MySymbol", {"apple", "orange"}}}]
            ]
        ]
    ]

    \maintainer chiarab
*)
processWolframLanguageData::msgs = "Messages were issued processing values for property '`1`'";
processWolframLanguageData[dataIn_] :=
	Block[{properties, transformAllFuncton, data, messageFlag, keys},
		
		data = Values[dataIn];
		
		properties = getWldProperties[];
		KeyValueMap[
			Function[{propertyName, propertyAssoc},
				Quiet[transformAllFuncton = propertyAssoc["TransformAllFuncton"], Part::keyw];
				If [!MissingQ[transformAllFuncton],
					messageFlag = Check[data[[All, propertyName]] = transformAllFuncton[Lookup[data, propertyName]], True];
					If [TrueQ[messageFlag],
						Message[processWolframLanguageData::msgs, propertyName];
					];
				]
			],
			properties
		];
		
		(* If any of the properties have "Field" values specified, then map those
		   keys to the given field. *)
		With[
			{
				keyReplacements =
					Normal[
						DeleteCases[Quiet[properties[[All, "Field"]], Part::keyw], _Missing]
					]
			},
			If [keyReplacements =!= {},
				data = replaceKeysOfAssociations[data, keyReplacements];
			];
		];
		AssociationThread[
			Keys[dataIn][[All, 2]],
			data
		]
	];

(*!
	\function getWldProperties
	
	\calltable
		getWldProperties[] '' returns an association of WolframLanguageData properties we're interested in.
	
	\maintainer chiarab
*)
getWldProperties[] := Association @@ Replace[$wolframLanguageDataProperties, name_String :> (name -> <||>), 1]

(*!
	\function disableEntityFrameworkBoxes
	
	\calltable
		disableEntityFrameworkBoxes[] '' disable the entity framework boxes. This is very useful because otherwise (as of June 2015), things like Entity[SoftwareProduct,MATLAB] are taking a couple seconds to render in my frontend.
	
	\maintainer chiarab
*)
disableEntityFrameworkBoxes[] :=
	(
		Unprotect[EntityFramework`MakeEntityFrameworkBoxes];
		Clear[EntityFramework`MakeEntityFrameworkBoxes];
		EntityFramework`MakeEntityFrameworkBoxes[___] := $Failed;
	);

(*!
	\function transformOptions
	
	\calltable
		transformOptions[listsOfOptions] '' transform lists of lists of options gotten from WolframAlphaData.

	Examples:
	
	transformOptions[
		{
			{
				Entity["WolframLanguageSymbol", "Permissions"] -> Entity["WolframLanguageSymbol", "Automatic"],
				Entity["WolframLanguageSymbol", "IconRules"] -> Entity["WolframLanguageSymbol", "Automatic"]
			},
			{"AlignmentPoint" -> "Center", "AspectRatio" -> "1/GoldenRatio", "Axes" -> "True"}
		}
	]

	===

	{{"Permissions", "IconRules"}, {"AlignmentPoint", "AspectRatio", "Axes"}}

	Unit tests: transformOptions.mt

	\maintainer chiarab
*)
transformOptions[listsOfOptions_] :=
	Replace[
		listsOfOptions,
		val:Except[_Missing] :>
		Replace[
			val[[All, 1]],
			(* Sometimes option LHSs are strings, but sometimes they are of the form
			   Entity[_, optionName_] *)
			e_Entity :> CanonicalName[e],
			{1}
		],
		{1}
	]

(*!
	\function transformOtherComputerLanguageEquivalents
	
	\calltable
		transformOtherComputerLanguageEquivalents[listOfLists] '' transform lists of lists of OtherComputerLanguageEquivalents values gotten from WolframAlphaData

	Examples:
	
	transformOtherComputerLanguageEquivalents[
		{
			CalculateGrid[
				{
					{Entity["SoftwareProduct", "Maple"], "DEplot", "odeplot", "plot", "replot"},
					{Entity["SoftwareProduct", "MATLAB"], "ezplot", "fplot", "plot", "plotyy"}
				},
				{None, None},
				{"ProcessedPartial" -> True, "Partial" -> 2, "EllipsisRow" -> None},
				"InvalidElements" -> {__Missing}
			],
			CalculateGrid[
				{
					{Entity["SoftwareProduct", "Mathcad"], "reverse"},
					{Entity["SoftwareProduct", "MATLAB"], "reflect"}
				},
				{None, None},
				{"ProcessedPartial" -> True, "Partial" -> 2, "EllipsisRow" -> None},
				"InvalidElements" -> {__Missing}
			]
		}
	]

	===

	{
		{"DEplot", "odeplot", "plot", "replot", "ezplot", "fplot", "plot", "plotyy"},
		{"reverse", "reflect"}
	}

	Unit tests: transformOtherComputerLanguageEquivalents.mt

	\maintainer chiarab
*)
(*transformOtherComputerLanguageEquivalents[listOfLists_] :=
	Replace[listOfLists, callGrid_Global`CalculateGrid :> Flatten[callGrid[[1, All, 2 ;;]]], {1}]*)

(* Extract symbol names from links on Guide pages *)
uriToSymbol[uri_String] := If[MatchQ[FileNameSplit[uri], {___, "paclet:ref" | "ref", _}], FileBaseName[uri], ""];

(* Very basic function to get rid of e.g. "[f, {x, xmin, xmax}]" in "Plot[f,{x, xmin, xmax}]" *)
Clear[deleteItalicizedArguments];
deleteItalicizedArguments[s_String] := If[
	StringFreeQ[s, "italicizedArgumentWrapper"],
	s,
	StringReplace[s, {"italicizedArgumentWrapper@" ~~ Shortest[Repeated[Except["@"]]]... ~~ "@" -> ""}]
];
deleteItalicizedArguments[s___] := s;

unwrapItalicizedArguments[s_String] := StringReplace[s, {"italicizedArgumentWrapper@" ~~ arg:(Shortest[Repeated[Except["@"]]]...) ~~ "@" :> ToString[arg]}];
unwrapItalicizedArguments[s___] := s;

unwrapSymbols[s_String] := StringReplace[s, {"symbolWrapper@" ~~ arg:(Shortest[Repeated[Except["@"]]]...) ~~ "@" :> arg}]
unwrapSymbols[s___] := s;

(*!
	\function convertToString
	
	\calltable
		convertToString[e] '' converts the given expression to a string.
	
Taken from Transmogrify's Utilities.m:
https://cvs-master.wolfram.com/viewvc/Pubs/OnlineProduction/Applications/Transmogrify/Kernel/Utilities.m?view=log

... in favor of introducing a direct dependency.

Many additions were made since, however.

    Example:

    convertToString[GridBox[{{"a", "b", "c"}, {"d", "e", "f"}}]] === "a\tb\tc\nd\te\tf"

    Unit tests: convertToString.mt

    \maintainer chiarab
*)

Clear[convertToString];

convertToString::unke = "Unhandled expression with head `1`";
convertToString[c_String] := If[
	StringFreeQ[c, "\!"], 
	c, 
	"italicizedArgumentWrapper@" <> c <> "@"
];
(*StringReplace[
	c, 
	"\"\!\(\*\nStyleBox[\"" ~~ 
		arg : Shortest[Repeated[Except["\""]]] ~~ 
		"\"," ~~ 
		WhitespaceCharacter... ~~ 
		("\"TI\"" | "\"TI2\"" | "\"TR\"") ~~ 
		WhitespaceCharacter ... ~~ 
		"]\)\>\"" 
		:> StringJoin["italicizedArgumentWrapper@", convertToString[arg], "@"]
];*)

convertToString[
	Alternatives[
		c_Cell,
		c_BoxData,
		c_FormBox,
		c_AdjustmentBox,
		c_TextData,
		c_TextCell,
		c_TagBox,
		c_FrameBox,
		c_ItemBox,
		c_InterpretationBox,
		c_TooltipBox,
		c_OpenerBox,
		c_PanelBox,
		c_RadioButtonBox,
		c_SliderBox,
		c_Slider2DBox,
		c_SetterBox,
		c_AnimatorBox,
		c:(FEPrivate`FrontEndResource["FEExpressions", "NecklaceAnimator"][___]),
		c:(FEPrivate`FrontEndResource["FEExpressions", "ArcUpAnimator"][___]),
		c:(FEPrivate`FrontEndResource["FEExpressions", "ArcDownAnimator"][___]),
		c:(FEPrivate`FrontEndResource["FEExpressions", "ArcUpFillAnimator"][___]),
		c:(FEPrivate`FrontEndResource["FEExpressions", "PercolateAnimator"][___]),
		c:(FEPrivate`FrontEndResource["FEExpressions", "EllipsisAnimator"][___]),
		c:ArcUpFillAnimator,
		c:PercolateAnimator,
		c:EllipsisAnimator
	]
] :=
	convertToString@c[[1]];

convertToString[StyleBox[e_, "TI"|"TI2"|"TR", ___]] := StringJoin[
	"italicizedArgumentWrapper@", 
	(* this avoids infinite recursion errors in deleteItalicizedArguments *)
	StringReplace[
		convertToString[e], 
		"italicizedArgumentWrapper@" ~~ inner:(Shortest[Except["@"] ..] ...) ~~ "@" :> inner
	], 
	"@"
];
convertToString[c_StyleBox] := convertToString[c[[1]]];
	
convertToString[c_List] := StringRiffle[convertToString /@ c, " "];
convertToString[c_RowBox] := StringJoin[convertToString /@ c[[1]]];
convertToString[c_SubsuperscriptBox | c_UnderoverscriptBox | c_CounterBox | c_RadicalBox] :=
	StringRiffle[convertToString /@ (List @@ c)]
convertToString[c_SuperscriptBox] := convertToString@c[[1]] <> "^" <> convertToString@c[[2]];
convertToString[c_UnderscriptBox] := convertToString@c[[1]] <> "_" <> convertToString@c[[2]];
convertToString[c_OverscriptBox] := convertToString@c[[1]] <> " " <> convertToString@c[[2]];
convertToString[c_SubscriptBox] := convertToString@c[[1]] <> "_" <> convertToString@c[[2]];
convertToString[f_FractionBox] := StringJoin[" ( ", convertToString@f[[1]], " ) / ( ", convertToString@f[[2]] , " ) "];
convertToString[c : ButtonBox[_, BaseStyle -> "Link", ButtonData -> uri_, ___]] := StringJoin["symbolWrapper@", uriToSymbol[uri], "@"];
convertToString[c_ButtonBox] := convertToString@c[[1]];
convertToString[TemplateBox[{}, ___]] := "";
convertToString[c : TemplateBox[_, "RefLink",BaseStyle -> "InlineFunctionSans"]] := uriToSymbol[c[[1,2]]];
convertToString[c : TemplateBox[_, "RefLink"|"OrangeLink"|"BlueLink"|"WebLink"|_, ___]] := convertToString[c[[1,1]]];
(*convertToString[c : TemplateBox[_List, ___]] := StringJoin[Riffle[convertToString /@ c[[1]], " "]];*)
convertToString[c_] :=
	(
	Message[convertToString::unke, Head[c]];
	AppendTo[$ConvertToStringFailures, c];
	ToString@c
	)
convertToString[c__] := (Message[convertToString::sequence]; StringJoin[convertToString /@ {c}]);
convertToString::sequence = "Warning: Sequence used as an argument, should be List";

convertToString[c_GridBox] :=
	StringRiffle[
		Function[{row},
			StringRiffle[convertToString /@ row, "\t"]
		] /@ First[c],
		"\n"
	]

convertToString[c_SqrtBox] := "\[Sqrt] " <> convertToString@c[[1]];
(* ex. FrontEndResource["FEBitmaps", "ManipulatePasteIcon"] *)
convertToString[
	Alternatives[
		_GraphicsBox,
		_Graphics3DBox,
		_FEPrivate`FrontEndResource,
		_FrontEndResource,
		_GraphicsData,
		_CheckboxBox,
		_PopupMenuBox,
		_NamespaceBox
	]
] := ""
convertToString[HoldPattern[Rule][a_, b_]] := convertToString[a] <> " -> " <> convertToString[b]
convertToString[c:(_Symbol|_Integer|_Real)] := ToString[c]
convertToString[c_If] := convertToString[c[[2]]] <> " " <> convertToString[c[[3]]]

(* A bit unclear what to do here, because of situations like
   this:
   
	DynamicBox[
		ToBoxes[If[$OperatingSystem === "MacOSX", "Return", "Enter"], StandardForm],
		ImageSizeCache -> {41., {1., 9.}}
	]
	
  By letting the first argument evaluate, we're effectively favoring
  the OS that the skeletonizer runs on.
*)
convertToString[c_DynamicBox] := convertToString@c[[1]]

If [!ValueQ[$ConvertToStringFailures],
	$ConvertToStringFailures = {};
];

(*!
	\function notebookExtractViaStylesAndMetadata
	
	\calltable
		notebookExtractViaStylesAndMetadata[nb, styles, processingParams] '' extract cells that match one of the given styles and convert their content to string. 'styles' should be Alternatives.
	
	processingParams:	Parameters wrt how to process each style can be supplied in this Association. Can be None. 

	Example:

	notebookExtractViaStylesAndMetadata[
		FileNameJoin[
			{
				$InstallationDirectory,
				"Documentation",
				"English",
				"System",
				"Guides",
				"ListManipulation.nb"
			}
		],
		"GuideTitle" | "GuideAbstract" | "GuideFunctionsSubsection",
		<|
			"GuideTitle" ->
			<|
				"Field" -> "Title"
			|>,
			"GuideFunctionsSubsection" ->
				<|
					"Transformation" ->
						Function[{str}, StringReplace[str, WhitespaceCharacter...~~"\[RightGuillemet]"~~EndOfLine :> ""]]
				|>
		|>
	]

	===

	<|
		"Title" -> {"List Manipulation"},
		"GuideAbstract" ->
		{
			"Lists are central constructs in the Wolfram Language, used to represent collections, arrays, sets, and sequences of all kinds. Lists can have any structure and size and can routinely involve even millions of elements. Well over a thousand built-in functions throughout the Wolfram Language operate directly on lists, making lists a powerful vehicle for interoperability. "
		},
		"GuideFunctionsSubsection" ->
		{
			"Constructing Lists",
			"Elements of Lists",
			"Finding Sublists",
			"Rearranging & Restructuring Lists",
			"Applying Functions to Lists",
			"Predicates on Lists",
			"Math & Counting Operations",
			"Displaying & Visualizing Lists",
			"Importing & Exporting Lists",
			"Creating Associations from Lists"
		}
	|>
	
	Unit tests: notebookExtractViaStyles.mt

	\maintainer chiarab
*)

processValue =
	Function[
		{val, styleName},
		Lookup[
			Replace[processingParams, None -> {}][styleName],
			"Transformation",
			Identity
		][val]
	]

$nonSnippetFields = {"Title", "LinkedSymbols", "SeeAlso"};

notebookExtractViaStylesAndMetadata::msgs = "Messages were issued while reading the notebook `1`";
notebookExtractViaStylesAndMetadata::nomd = "No metadata found for notebook `1`";
notebookExtractViaStylesAndMetadata[nb_, styles_, metadata_, processingParams_] :=
	Block[{processValue, processKey, res, notebookExpression, taggingRules, md, snippetPlaintext, description, nonSnippetStyles},
		
		xPrint[nb];
		xPrint[styles];
		
		(* Setup a function 'processValue' that will take
		   a string value from the notebook and transform it.
		   For example, we "GuideFunctionsSubsection" strings
		   and remove any instances of \[RightGuillemet]. *)
		processValue[val_, style_] := val;
		Function[{styleOrMetadata},
			With[{styleOrMetadataName = styleOrMetadata[[1]], styleOrMetadataAssoc = styleOrMetadata[[2]]},
				With[{func = styleOrMetadataAssoc["Transformation"]},
					If [!MissingQ[func],
						processValue[val_, styleOrMetadataName] := func[val]
					]
				]
			]
		] /@ Normal[Replace[processingParams, None -> {}]];
		
		If [Check[notebookExpression = Get[nb], $MessageFlag] === $MessageFlag,
			Message[
				notebookExtractViaStylesAndMetadata::msgs,
				nb
			];
		];
		
		md = FirstCase[
			notebookExpression,
			HoldPattern[Rule[TaggingRules, {___, Rule["Metadata", d_], ___}]] :> d,
			Missing["NotFound"],
			{1}
		];
		If[
			!FreeQ[md, "index" -> False],
			Return[<||>]
		];
		
		(* Extract all of the cells of interest from the notebook and apply
		   any desired transformations to them. Group them by the style
		   of cell that they came from. *)
		res =
			Cases[
				notebookExpression,
				c:Cell[__, st:styles, ___] :> {st, processValue[convertToStringAndReportIssues[c], st]},
				Infinity
			];
		
		(* SEARCH-711: we don't want the Title, SeeAlso, LinkedSymbols fields to be part of the snippet. *)
		nonSnippetStyles = Keys[
			Select[
				processingParams, 
				MatchQ[#["Field"], Alternatives @@ $nonSnippetFields] &
			]
		];
		
		snippetPlaintext = StringRiffle[DeleteCases[res, {Alternatives@@Join[nonSnippetStyles, $nonSnippetFields], ___}][[All,2]], "\[Paragraph]"];
		
		If[
			!MatchQ[md, OptionsPattern[]]
			,
			res = Join[res, {{"URL", "None"}}]; (* we need the "URL" field even in absence of metadata *)
			Sow[
				nb, 
				"missingMD"
			]
			,
			res = Join[
				res, 
				DeleteCases[
					MapThread[
						{#1, processValue[#2, #1]}&, 
						{
							List@@metadata, 
							Lookup[
								md, 
								List@@metadata
							]
						}
					], 
					{_, _Missing|{}}
				]
			];
		];
		
		(* Transform keys. ex. We want to replace the key "GuideAbstract" with just "Abstract". *)
		If [AssociationQ[processingParams],
			With[
				{
					keyReplacements =
						Normal[
							DeleteCases[Quiet[processingParams[[All, "Field"]], Part::keyw], _Missing]
						]
				},
				If [keyReplacements =!= {},
					res[[All, 1]] = Replace[res[[All, 1]], keyReplacements, {1}];
				]
			]
		];
		
		(* Create an association of form: field_ -> values_List. *)
		res = Map[
			(* Anything that is wrapped in italicizedArgumentWrapper is removed here *)
			Map[unwrapSymbols[deleteItalicizedArguments[#]]&, #]&,
			GroupBy[
				res
				,
				First -> Last
			]
		];
		
		(* For the SnippetPlaintext we want to keep the arguments, so we remove the italicizedArgumentWrapper *)
		res["SnippetPlaintext"] = unwrapSymbols[unwrapItalicizedArguments[snippetPlaintext]];
		
		(* For spelling correction *)
		res["Dictionary"] = unwrapSymbols[deleteItalicizedArguments[snippetPlaintext]];
		
		(* If a field only has one value, or if it is a list already, then don't wrap it in List. *)
		res = Replace[res, {{val_String} :> val, {None} :> None, {val:Repeated[_List]} :> Flatten[{val}, 1]}, {1}];
		
		res["URL"] = docURIToURL[res["URI"]];

		(* TODO: kill this? *)
		res["Location"] = nb;
		
		(* I set this field for the documents that miss it, or indices that cover only non-symbol documents will not work *)
		res["Frequency"] = Replace[res["Frequency"], _Missing :> 1.];
		
		(* "ExactTitle" is used for DirectHitSearch *)
		res["Title"] = Replace[DeleteMissing[Flatten[Join[{res["ExactTitle"]}, {Lookup[res, "Title", {}]}]]], {e_} :> e];
		
		(* Used for those queries that are "almost direct hits", modulo casing or whitespaces, e.g. "string join" or "stringjoin" *)
		(* Also matching the FileBaseName - bug 336457 *)
		res["NormalizedTitle"] = DeleteDuplicates[If[!StringQ[#], #, StringDelete[ToLowerCase[#], WhitespaceCharacter..|"()"]]& /@ {res["ExactTitle"], FileBaseName[res["URI"]]}];
		
		If[
			!MissingQ[res["PacletName"]],
			res["NotebookPackage"] = transformNotebookPackage[res["PacletName"]];
			
			If[
				res["NotebookPackage"] === "Compatibility",
				(* e.g. "Vector analysis" shouldn't have Compatibility/tutorial/VectorAnalysis on top. *)
				(* And actually, only symbols directly linked from the Compatibility page should be matched there (per SW). *)
				res = KeyDrop[res, {"Title", "NormalizedTitle"}]
			]
		];
		
		Sow[
			Map[
				Function[seealso, DirectedEdge[res["ExactTitle"], seealso]], 
				Replace[res["SeeAlso"], _Missing :> {}]
			], 
			"ReferredBy"
		];
		
		With[{
			linked = res["LinkedSymbols"]
		},
			If[
				ListQ[linked] && linked =!= {} && $frequencyAssociation =!= <||>,
				res["Frequency"] = Mean[Replace[Values[DeleteMissing[KeyTake[$frequencyAssociation, linked]]], {} :> {1.}]]
			]
		];
		
		(* Used for queries like "interpreter guide" or "AVI format" or "tiger export format" *)
		res["TokenizedNotebookType"] = DeleteDuplicates@Join[
			StringSplit[Replace[res["NotebookType"], Except[_String] :> ""]],
			StringSplit[Replace[res["DisplayedCategory"], Except[_String] :> ""], " "|"/" (*"Import/Export Format"*)]
		];

		If[
			res["NotebookStatus"] === "None",
			With[{
					versions = StringCases[FileNameTake[res["URI"], -1], ("NewIn" | "NewFeaturesIn") ~~ version:Repeated[DigitCharacter] :> version]
				},
				If[
					MatchQ[versions, {_String}] && FromDigits[StringDrop[First[versions], -1]] <= $VersionNumber - 2,
					res["NotebookStatus"] = "NewInOldVersion"
				]
			]
		];
		
		res
	];

(*!
	\function removeRightGuillemet
	
	\calltable
		removeRightGuillemet[str] '' remove a RightGuillemet from the end of a string.

	Examples:
    
    removeRightGuillemet["Testing \[RightGuillemet]"] === "Testing"

    Unit tests: removeRightGuillemet.mt

    \maintainer chiarab
*)
removeRightGuillemet[str_] := StringReplace[str, WhitespaceCharacter... ~~ "\[RightGuillemet]" ~~ EndOfLine :> ""]

(* Split SeeAlso at the \[FilledVerySmallSquare] separator *)
splitSeeAlso[str_] := StringSplit[str, WhitespaceCharacter ... ~~ "\[FilledVerySmallSquare]" ~~ WhitespaceCharacter ...]

(* Naif filter to allow potential symbol names in the "LinkedSymbols" field of guide pages, not ellipses and the like. *)
filterSymbols[str_] := If[StringMatchQ[str, (LetterCharacter | DigitCharacter | "$")..], str, ""];

extractSymbols[str_] := StringCases[str, {"symbolWrapper@" ~~ arg:(Shortest[Repeated[Except["@"]]]...) ~~ "@" :> filterSymbols[arg]}]

(*!
	\function convertToStringAndReportIssues
	
	\calltable
		convertToStringAndReportIssues[e] '' converts the given boxes expression to string, and if there are any messages issued, we issue another message indicating what expression was at fault.
	
	\related 'convertToString
	
	\maintainer chiarab
*)
convertToString::msgs = "Messages were issued converting an expression to a string: `1`";
convertToStringAndReportIssues[e_] :=
	Block[{messageFlag1, messageFlag2, res},
		
		messageFlag2 =
		Check[
			messageFlag1 =
			Check[
				res =
				convertToString[e],
				$Failure1,
				{convertToString::unke}
			],
			$Failure2
		];
		
		If [messageFlag2 === $Failure2 && messageFlag1 =!= $Failure1,
			Message[convertToString::msgs, e];
		];
		
		res
	];

(*!
	\function transformFrequenciesToWeight
	
	\calltable
		transformFrequenciesToWeight[listOfFrequencySets] '' transform a list of frequency data (for multiple documents) into a weight to be used when ranking documents.

	I'm using a Log here because we probably don't want a function's
	popularity the skew the ranking linearly. Otherwise, we'd be saying
	we care relatively less about how well the person's search matches
	the document and more about how popular the function is. That's OK
	to a point, but linear feels out of balance.

	Examples:
    
    transformFrequenciesToWeight[
        {
            {
                "All" -> 0.00044831285110258904,
                "StackExchange" -> 0.00279207226530683,
                "TypicalProductionCode" -> 0.0021898052514032628,
                "WolframAlphaCodebase" -> 9.47599996699051*^-6,
                "WolframDemonstrations" -> 0.0010785080908379521,
                "WolframDocumentation" -> 0.0033621900176985307
            },
            {
                "All" -> 0.000031393463967149486,
                "StackExchange" -> 0.00022089210576464182,
                "TypicalProductionCode" -> 0.00011897519806290583,
                "WolframAlphaCodebase" -> 1.005461457958953*^-6,
                "WolframDemonstrations" -> 0.00012433927864297193,
                "WolframDocumentation" -> 0.00013387425609850313
            }
        }
    ]

    ===

    {3.317051780489244, 0.702984853403019}

    Unit tests: transformFrequenciesToWeight.mt

    \maintainer chiarab
*)
transformFrequenciesToWeight[listOfFrequencySets_] :=
	(* This was previously a Block. I was seeing a bizarre problem
	   where the output of this function involved an unevaluated
	   TextSearch`Skeletonizer`PackagePrivate`data, which I don't
	   understand, given that 'data' is assigned to explicitly.
	   I tried changing the Block to Module and the problem
	   appears to have gone away. *)
	Module[{data = listOfFrequencySets},
		(* Create a dummy entry for symbols that are missing frequency data. *)
		data = Replace[data, _Missing :> {"All" -> 0.000005}, {1}];
		Lookup[data, "All"] / 0.00002
	]

(* Change the URI of a documentation page (as stored in the nb's metadata) into a full URL *)
docURIToURL[uri_] := If[
	!StringQ[uri],
	None,
	"http://reference.wolfram.com/language/" <> uri <> ".html"
];

(* Extract info from the "status" metadata that is then used for doc weighting *)
transformNotebookStatus[status_] := Replace[
	status, 
	Except["ObsoleteFlag"|"AwaitingFutureDesignReviewFlag"] :> "None"
]

(* Extract info from the "paclet" metadata that is then used for doc weighting *)
transformNotebookPackage[paclet_] := Replace[
	paclet, 
	{
		Except["Mathematica"|"Compatibility Package"|_Missing] :> "Package", 
		"Compatibility Package" :> "Compatibility"
	}
]

(* Delete synonyms in the form "2.11.1" or "A.11.1" from old Mathematica book. *)
deleteNumbers[l:List[__String]] := Select[l, !StringMatchQ[#, Repeated[DigitCharacter|LetterCharacter]~~"."~~Repeated[DigitCharacter|"."]]&]
deleteNumbers[e___] := e;

(*!
	\function replaceKeysOfAssociations
	
	\calltable
		replaceKeysOfAssociations[assocs, keyReplacements] '' replace the keys of the given associations according to the given replacement rules.

	Examples:

    replaceKeysOfAssociations[
        {Association["B" -> 1, "C" -> 2], Association["A" -> 0, "B" -> 1]},
        {"B" -> "!"}
    ]

    ===

    {<|"!" -> 1, "C" -> 2|>, <|"A" -> 0, "!" -> 1|>}

    Unit tests: replaceKeysOfAssociations.mt

    \maintainer chiarab
*)
replaceKeysOfAssociations[assocs_, keyReplacements_] :=
	Block[{innerKeys, innerValues},
		innerKeys = Replace[Keys[assocs], keyReplacements, {2}];
		innerValues = Values[assocs];
		AssociationThread @@@ Transpose[{innerKeys, innerValues}]
	];

(*!
	\function filePathToNotebookClass
	
	\calltable
		filePathToNotebookClass[filePath] '' given a documentation file, returns the name of the corresponding 'class' of notebooks, such as 'Guides', 'Messages', etc.

	Examples:
    
    filePathToNotebookClass[path] === TODO
	
	\related '
	
	\maintainer chiarab
*)
filePathToNotebookClass[filePath_] :=
	FileNameTake[
		filePath, 
 		Last[
 			Position[
 				FileNameSplit[filePath], 
    			"ReferencePages" | "English" | "System"
    		]
    	] + 1
    ]

End[];
 
EndPackage[];

