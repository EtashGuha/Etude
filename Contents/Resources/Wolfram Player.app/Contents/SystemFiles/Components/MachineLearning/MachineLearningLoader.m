(* All loading of the paclet's Wolfram Language code should go through this file. *)

(* Extra symbols declared with context so that may be used elsewhere without it *)

{
	System`RecognitionPrior
};

(* Developer maintains this list of symbols.
   autoloadSymbols must agree with the symbols listed in the Kernel extension in the PacletInfo.m file.
*)

Begin["MachineLearning`Private`"]

autoloadSymbols = {
	"System`Classify", 
	"System`ClassifierFunction", 
	"System`ClassifierMeasurements",
	"System`ClassifierMeasurementsObject",
	
	"System`Predict",
	"System`PredictorFunction",
	"System`PredictorMeasurements",
	"System`PredictorMeasurementsObject",
	
	"System`DimensionReduction",
	"System`DimensionReduce",
	"System`DimensionReducerFunction",

	"System`DistanceMatrix",
	"System`FeatureNearest",
	
	"System`ClusterClassify",
	
	"System`Dendrogram",
	"System`ClusteringTree",
	"System`ClusterDissimilarityFunction",
				
	"System`FeatureExtract",
	"System`FeatureExtraction",
	"System`FeatureExtractorFunction",
	"System`FeatureDistance",

	"System`FindClusters",
	"System`ClusteringComponents",
	
	"System`BayesianMinimization",
	"System`BayesianMaximization",
	"System`BayesianMinimizationObject",
	"System`BayesianMaximizationObject", 
	
	"System`ActiveClassification",
	"System`ActiveClassificationObject",
	"System`ActivePrediction",
	"System`ActivePredictionObject",
	
	"System`LearnDistribution",
	"System`LearnedDistribution",
	"System`SynthesizeMissingValues",
	"System`RarerProbability",

	"System`AnomalyDetection",
	"System`DeleteAnomalies",
	"System`AnomalyDetectorFunction",
	"System`FindAnomalies",

	"System`SequencePredict",
	"System`SequencePredictorFunction",	

	"MachineLearning`MLProcessor",
	"MachineLearning`SortedHashAssociation"
};

obsoleteSymbols = {"System`ClassifierInformation", "System`PredictorInformation"};


(* symbols that should not trigger the autoloading *)
extraSymbols = {
	(* If these are not present, local loading complains... keeping them for now *)
	"System`RecognitionPrior",
	"System`TimeGoal",
	"System`AcceptanceThreshold",
	"System`MissingValuePattern",
	"System`ComputeUncertainty"
};

symsToUnprotect = {
	"MachineLearning`PackageScope`Evaluation",
	"MachineLearning`PackageScope`PredictorEvaluation",
	"MachineLearning`PackageScope`ClassifierEvaluation",
	"MachineLearning`PackageScope`BuiltInFunction"
};

symsToProtect = Hold[Module[
	{names},
	names = Join[Names["MachineLearning`*"], Names["MachineLearning`PackageScope`*"]];
	names = Select[names,
		ToExpression[#, InputForm, 
			Function[{sym}, Length[DownValues[sym]] > 0 || Length[SubValues[sym]] > 0, HoldFirst]
		] &
	];
	names = Join[names, autoloadSymbols, obsoleteSymbols, extraSymbols];
	names = Complement[names, symsToUnprotect];
	names
]];


(***** General messages *****)

General::"mlbdopt" = "Value of option `1` is not valid.";
General::"mlobs" = "`1` is obsolete. It has been superseeded by `2` since version `3`."

(*Bad dataset*)
General::"mlbddata" = "The data is not formatted correctly."
General::"mldpsa" = "The dataset cannot be a SparseArray of depth larger than 2."
General::"mlbddataev" = "The data being evaluated is not formatted correctly."
General::"mlbftlgth" = "Examples should have the same number of features."
General::"mlbftlgth2" = "Example `1` should have `2` features instead of `3`."
General::"mlmpty" = "The dataset should contain at least one example."
General::"mlmptyv" = "The validation set should contain at least one example."
General::"mlmptymiss" = "The dataset should contain at least one non-missing example."
General::"mlinclgth" = "Incompatible lengths: all features should contain the same number of examples."
General::"mlunbal" = "The number of inputs (`1`), and the number of corresponding outputs (`2`) should be identical."

General::"mlemptyex" = "The dataset should contain at least one feature."


(*Data processing*)
General::"mlnoprocinv" = "The processor `1` cannot be inverted."
General::"mlinvprocft" = "Invalid feature type. Processor `1` needs feature types `2` instead of `3`"
General::"mlnoprocmiss" = "Processor `1` cannot process missing data."
General::"invlang" = "The processor cannot generate vector embedding for the language: \"`1`\"."

(*FeatureTypes*)
General::"mlbftyp" = "Value of option FeatureTypes \[Rule] `1` should be a string, a list of string, an association, or Automatic"
General::"mlbofttypel" = "Value of option FeatureTypes \[Rule] `1` should be a list of length `2`, or an association."
General::"mlukwnfttype" = "Unknown feature type `1` in value of option FeatureTypes \[Rule] `2`. Feature types can be Automatic`3`"
General::"mlunknwnft" = "Warning: feature `1` in value of option FeatureTypes \[Rule] `2` is not in the dataset."
General::"mlnotnomfeat" = "Feature `1` cannot be declared as non-nominal in value of option NominalVariables \[Rule] \[Ellipsis]."
General::"mlincfttp" = "Incompatible variable type (`2`) and variable value (`1`)."
General::"mlincftio" = "Data processors cannot be joined because their feature types or dimensions are incompatible.";

General::"mlbfttype" = "Invalid argument `1`. `3` is a `4` feature."
General::"mlbofttype" = "In value of option FeatureTypes \[Rule] \[Ellipsis], feature `1` cannot be of type `2`."


General::"mlcntinft" = "The type of feature `1` cannot be interpreted. In general, datasets cannot contain nested structures. " <>
"See \*ButtonBox[\"FeatureTypes\",\nBaseStyle->\"Link\",\n\
ButtonData->\"paclet:ref/FeatureTypes\"] to learn about possible feature types."
(*"Nominal features cannot be lists, associations, datasets, or any other data wrappers."*)

General::"mlincvecl" = "Vector `1` should be of length `2` instead of `3`."
General::"mlinctsdim" = "Tensor `1` should be of dimension `2` instead of `3`."
General::"mlincdim" = "`1` should have the dimension `2` instead of `3`."
General::"mlincdepth" = "`1` should be a `2` instead of a `3`."

General::"mlzerovec" = "Vector features cannot be of length zero."
General::"mlzerodim" = "Tensor features cannot have an emtpy dimension."

General::"mlnomvar" = "Inactive option NominalVariables. Use FeatureTypes instead."

(* FeatureNames *)
General::"mlbftnm" = "Value of option FeatureNames \[Rule] `1` is not compatible with feature names in the dataset."
General::"mlbftnmf" = "Value of option FeatureNames \[Rule] `1` should be a string, a list of string, or Automatic."
General::"mlbftnmlgth" = " Value of option FeatureNames \[Rule] `1` and the number of features (`2`) do not match."

General::"mlukwnfn" = "Unknown feature name `1`.";


(* FeatureExtractor *)
General::"mlinvfe" = "Invalid value of option FeatureExtractor."
General::"mlinvfeft" = "Invalid value of option FeatureExtractor. The feature types and dimensions cannot be inferred. Use a FeatureExtractorFunction[...] instead."


(* Weights *)
General::"mlbdwt" = "Value of option Weights \[Rule] `` should be Automatic or a list of real positive values."
General::"mlbdwtl" = "The number of weights (`1`) and the number of examples (`2`) should be equal."
General::"mlbdfwtl" = "The number of feature weights (`1`) and the number of features (`2`) should be equal."

(* Options *)
General::"mlbdutil" = "Value of option UtilityFunction \[Rule] `1` is not valid."
General::"mlbdth" = "Value of option IndeterminateThreshold \[Rule] `` should a positive real value."
General::"mlbdoptv" = "Value of option `1` in `2` is not valid."
General::"mlbdcp" = "Value of option ClassPriors \[Rule] `` should be an association of positive values."
General::"mlbdnv" = "Value of option NominalVariables \[Rule] `` should be Automatic, All, None, an integer, a list of integers, a string, or a list of string."
General::"mlbdmtspec" = "Invalid method specification `1`."
General::"mlbdopt" = "Value of option `1` is not valid."
General::"mlbdoptval" = "Value of option `1` \[Rule] `2` is not valid."
General::"mlbdopttyp" = "Option `1` \[Rule] `2` should have a `3` value."
General::"mlbdautopt" = "Option `1` cannot be set to Automatic."
General::"mlinvtg" = "`1` is not a valid TimeGoal specification, it should be a positive value (number of seconds) or a time Quantity."
General::"mlinvmiter" = "Value of option MaxIteration -> `` should be a positive integer."

(* Method *)
General::"mluknwnmtd" = "Unknown method `1` in value of option Method. Possible methods are: Automatic`2`."
General::"mlbdmtdind" = "Unknown feature index `1` in method specification `2`."
General::"mlbdmtdkey" = "Unknown feature name `1` in method specification `2`.";

(* Sound *)
General::"mlnosn" = "Sound features cannot contain SoundNote."

(*Interpretation*)
General::"mlintfail" = "Input `1` cannot be interpreted."

(* Property/Method check *)

General::"mlnaset" = "`2` is not an available `1`.";
General::"mlnaseth" = "`2` is not an available `1`. Did you mean `3` instead?";
General::"mlnasetl" = "`2` is not an available `1`. Possible `3` include `4`.";
General::"mlnasethl" = "`2` is not an available `1`. Did you mean `3` instead? Possible `4` also include `5`.";

(* Distances *)
General::"mlinvdf" = "Invalid value of option DistanceFunction."
General::"mlinvndist" = "The user-supplied distance function `1` does not give a real numeric distance when applied to the element pair `2` and `3`."


(* Miscellaneous *)
General::"mlukwnerr" = "An unknown error occured."
General::"mlwkey" = "``"
General::"mlwcol" = "Unknown variable index ``. The index of the output variable should a be string or an integer."
General::"mlcntconf" = "The data cannot be conformed to the appropriate type." (* temporary *)
General::"noth" = "The element Nothing cannot be used as input. "

General::"mlcntclnet" = "This neural network cannot be converted to a ClassifierFunction[\[Ellipsis]]."
General::"mlnetfl" = "An internal error occurred while constructing the neural network. "

(*Processors - NetModels*)
General::"mlinvmod" = "Unable to find the NetModel \"`1`\" to process the data."
General::"mlallow" = "Unable to download the NetModel `1` to process the data.\n The Wolfram Language is currently configured not to use the Internet.\n To allow Internet use, check the \"Allow the Wolfram Language to use the Internet\" box in the Help \[FilledRightTriangle] Internet Connectivity dialog or set $AllowInternet = True."
General::"mlinvhttp" = "Unable to connect to the server to download the NetModel `1` to process the data."
General::"mldlfailpr" = "Unable to download the NetModel `1` to process the data"

(***** End - General messages *****)





Options[findPackageFiles] = {
	"ExcludedDirectories" -> {}
};
findPackageFiles[package_, opts:OptionsPattern[]] := Module[
	{directories, rootdirectory, files, excludeddirectories, excludeddirectorychildren},
	rootdirectory = DirectoryName[$InputFileName];
	directories = Select[
		FileNames[All, rootdirectory, Infinity]
		,
		DirectoryQ
	];
	directories = Prepend[directories, rootdirectory];
	excludeddirectories = OptionValue["ExcludedDirectories"];
	excludeddirectories = FileNameJoin[{rootdirectory, #}] & /@ excludeddirectories;
	excludeddirectorychildren = Select[directories, 
		Function[{dir}, 
			Apply[
				Or, 
				ancesterDirectoryQ[#, dir] & /@ excludeddirectories
			]
		]
	];
	excludeddirectories = Join[excludeddirectories, excludeddirectorychildren];
	directories = Complement[directories, excludeddirectories];
	files = FileNames["*.m", #] & /@ directories;
	files = Flatten[files];
	files = Select[files, packageFileQ[#, package] &]; (* packageFileQ speeds-up the loading *)
	files
];
packageFileQ[file_, package_] := UnsameQ[
	FindList[file, "Package[\"" <> package <> "`\"]", 1]
	, 
	{}
];
ancesterDirectoryQ[ancesterdir_, file_] := Module[
	{a, f},
	a = FileNameSplit[ancesterdir];
	f = FileNameSplit[file];
	And[
		Length[f] > Length[a]
		,
		Take[f, Length[a]] === a
	]
];
Options[reorganizePackageFiles] = {
	"ExcludedDirectories" -> {}
};
reorganizePackageFiles[package_, opts:OptionsPattern[]] := Module[
	{initialfile, originaldir, loadmfiles, originalfiles, newfiles, newdir, dir, filelinks},
	originaldir = DirectoryName[$InputFileName];
	initialfile = FileNameJoin[{originaldir, "InitialFile.m"}];
	loadmfiles = FileExistsQ[initialfile];
	If[loadmfiles,
		originalfiles = findPackageFiles["MachineLearning", opts];
		originalfiles = DeleteDuplicates[Prepend[originalfiles, initialfile]];
		newdir = CreateDirectory[];
		newfiles = MapIndexed[newFileName[originaldir, newdir, #1, #2] &, originalfiles];
		MapThread[CopyFile[#1, #2, OverwriteTarget -> True] &, {originalfiles, newfiles}];
		initialfile = FileNameTake[First[newfiles]];
		filelinks = AssociationThread[
			FileNameDrop[#, FileNameDepth[originaldir]] & /@ originalfiles,
			FileNameTake /@ newfiles
		];
		dir = newdir;
		,
		dir = originaldir;
	];
	{dir, initialfile, loadmfiles, filelinks}
];
newFileName[originaldir_, newdir_, originalfile_, {counter_}] := FileNameJoin[{
	newdir,
	StringJoin[
		"file",
		ToString[counter],
		FileBaseName[originalfile],
		".m"
		(*FileNameSplit[FileNameDrop[originalfile, FileNameDepth[originaldir]]]*)
	]
}];


Options[LoadFlatPackage] = {
	"AutoloadSymbols" -> {},
	"SymbolsToProtect" -> Automatic,
	"HiddenImports" -> {},
	"ExcludedDirectories" -> {}
}; 
LoadFlatPackage::"usage" = "LoadFlatPackage[package] loads all the files of package, including in subdirectories. 
Files in main directory and in subdirectories are considered equivalent in terms of context."
LoadFlatPackage[package_, opts:OptionsPattern[]] := Module[
	{dir, initialfile, loadmfiles, filelinks},
	{dir, initialfile, loadmfiles, filelinks} = 
		reorganizePackageFiles["MachineLearning", "ExcludedDirectories" -> OptionValue["ExcludedDirectories"]];
	PacletManager`Package`loadWolframLanguageCode[
		package, 
		package <> "`", 
		dir, 
		initialfile,
		"AutoUpdate" -> True,
		"Lock" -> False,
		"AutoloadSymbols" -> OptionValue["AutoloadSymbols"],
		"HiddenImports" -> OptionValue["HiddenImports"],
		"SymbolsToProtect" -> OptionValue["SymbolsToProtect"]
	];
	If[loadmfiles && dir =!= DirectoryName[$InputFileName], (* to be safe *)
		DeleteDirectory[dir, DeleteContents -> True];
	];
	filelinks
];

$filelinks = LoadFlatPackage["MachineLearning", 
	"ExcludedDirectories" -> {},
	"AutoloadSymbols" -> autoloadSymbols,
	"HiddenImports" -> {
		"PacletManager`", "Developer`", "GeneralUtilities`", 
		"NumericArrayUtilities`", "LightGBMLink`", "DAALLink`"
	},
	"SymbolsToProtect" -> symsToProtect
];


(* Provide a way for tools to know that this package uses tabs for indentation. *)
$useTabsOrSpaces = "Tabs";


Unprotect[NearestFunction];
(* Temporary way to add definitions to NearestFunction.
	Needed for ClusterClassify. *)
Options[NearestFunction] = {
	PerformanceGoal -> Automatic,
	"BatchProcessing" -> Automatic,
	RandomSeeding -> Automatic
};

MachineLearning`MachineLearningSet[NearestFunction, MachineLearning`PackageScope`NearestEvaluation,
	"ArgumentNumber" -> {1, 2},
	"SubValue" -> True,
	"ParametersPattern" -> _Association
];

Options[NearestFunction] = {};

nearestInfo[NearestFunction[data_]] := Module[
	{},
	{
		{
			"Input type" -> MachineLearning`PackageScope`DisplayFeatureInformation[data["Preprocessor"]],
			"Output property" -> data["OutputProperty"]
		},
		{
			"Number of elements" -> data["ExampleNumber"],
			"Feature-space type" -> MachineLearning`PackageScope`DisplayFeatureInformation[data["PostProcessor"]],
			"Feature-space distance" -> data["DistanceFunction"]
		}
	}
]

MachineLearning`PackageScope`DefineFormatting[NearestFunction, nearestInfo];

Protect[NearestFunction];

End[];

