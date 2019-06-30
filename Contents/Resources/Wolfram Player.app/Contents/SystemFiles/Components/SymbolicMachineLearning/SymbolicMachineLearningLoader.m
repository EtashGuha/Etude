Begin["SymbolicMachineLearning`Private`"]

autoloadSymbols = {
	"System`FindDistribution",
	"System`FindFormula"
	};

symsToProtect = {};

symsToUnprotect = {};

(*
PacletManager`Package`loadWolframLanguageCode[
	"SymbolicMachineLearning", "SymbolicMachineLearning`",
	DirectoryName[$InputFileName], "FindDistribution.m",
	"AutoUpdate" -> True,
	"ForceMX" -> TrueQ[SymbolicMachineLearning`$ForceMX],
	"Lock" -> False,
	"AutoloadSymbols" -> SymbolicMachineLearning`Private`autoloadSymbols,
	"HiddenImports" -> {"PacletManager`", "Developer`", "GeneralUtilities`"},
	"SymbolsToProtect" -> SymbolicMachineLearning`Private`symsToProtect
]
*)
symsToProtect = Hold[Module[
	{names},
	names = Join[Names["SymbolicMachineLearning`*"], Names["SymbolicMachineLearning`PackageScope`*"]];
	names = Select[names,
		ToExpression[#, InputForm, 
			Function[{sym}, Length[DownValues[sym]] > 0 || Length[SubValues[sym]] > 0, HoldFirst]
		] &
	];
	names = Join[names, autoloadSymbols];
	names = Complement[names, symsToUnprotect];
	names
]];

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
	initialfile = FileNameJoin[{originaldir, "Optimization.m"}];
	loadmfiles = FileExistsQ[initialfile];
	If[loadmfiles,
		originalfiles = findPackageFiles["SymbolicMachineLearning", opts];
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
LoadFlatPackage::usage = "LoadFlatPackage[package] loads all the files of package, including in subdirectories. 
Files in main directory and in subdirectories are considered equivalent in terms of context."

LoadFlatPackage[package_, opts:OptionsPattern[]] := Module[
	{dir, initialfile, loadmfiles, filelinks},
	{dir, initialfile, loadmfiles, filelinks} = 
		reorganizePackageFiles["SymbolicMachineLearning", "ExcludedDirectories" -> OptionValue["ExcludedDirectories"]];
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

$filelinks = LoadFlatPackage["SymbolicMachineLearning", 
	"ExcludedDirectories" -> {},
	"AutoloadSymbols" -> autoloadSymbols,
	"HiddenImports" -> {"PacletManager`", "Developer`", "GeneralUtilities`", "MachineLearning`"},
	"SymbolsToProtect" -> symsToProtect
];

End[];
