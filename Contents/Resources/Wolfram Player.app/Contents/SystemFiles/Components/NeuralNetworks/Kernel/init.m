(* PreemptProtect is to avoid macro timeout issues that seem to happen due
to some scheduled tasks on 11.3 *)

BeginPackage["NeuralNetworks`"];

Begin["NeuralNetworks`Bootstrap`Private`"];

PreemptProtect @ Quiet[

(* load dependancies *)

TakeDrop; (* This causes loading of Language`PairFunctions` *)
Needs["GeneralUtilities`"];
Needs["MXNetLink`"];
Needs["NumericArrayUtilities`"];

If[NameQ["System`CastLayer"], Unprotect["System`CastLayer"]; Remove["System`CastLayer"]];

$nnSymbols = GeneralUtilities`ClearPacletExportedSymbols["NeuralNetworks"];
General::nnloaderr = "Could not load the neural network runtime. Neural network functions will be unavailable.";

initNN[] := (
	(* obtain files to load *)
	$basePath = DirectoryName[$InputFileName, 2];
	subPath[p__] := FileNameJoin[{$basePath, p}];
	$allFiles = Select[FileNames["*.m", $basePath, Infinity], GeneralUtilities`FileQ];
	$ignoreFiles = Flatten @ {
		FileNames["*.m", subPath /@ {"Layers","Encoders","Decoders"}, Infinity],
		{subPath["Kernel", "init.m"], subPath["PacletInfo.m"]}
	};
	$files = Complement[$allFiles, $ignoreFiles];
	utilsPath = $PathnameSeparator <> "Utils" <> $PathnameSeparator;
	sortOrder[str_] := {-StringCount[str, "Initial.m"], -StringCount[str, "Types.m"], -StringCount[str, "Utils"], -StringCount[str, "Types"], str};
	$files = SortBy[$files, sortOrder];

	(* compute the directroy where an MX cache would exist *)
	$version = Lookup[PacletManager`PacletInformation["NeuralNetworks"], "Version"];
	NeuralNetworks`Private`$NeuralNetworksVersionNumber = $version;
	NeuralNetworks`Private`$IsLoading = False;

	(* scan for scoped and exported symbols *)
	$lines = StringSplit[StringTrim @ FindList[$files, {"PackageScope", "PackageExport"}], {"[", "]", "\""}];
	$private = Cases[$lines, {"PackageScope", _, name_} :> name];
	$public =  Cases[$lines, {"PackageExport", _, name_} :> name];
	$public = Complement[$public, Names["System`*"]];

	(* create symbols in the right context *)
	createInContext[context_, names_] := Block[{$ContextPath = {}, $Context = context}, ToExpression[names, InputForm, Hold]];
	createInContext["NeuralNetworks`", $public];
	createInContext["NeuralNetworks`Private`", $private];

	(* introduce helper to make it easy to reload a single file *)
	NeuralNetworks`ReloadFile[subpath_] := Block[
		{$ContextPath = $contexts, NeuralNetworks`Private`$IsLoading = True},
		Unprotect @@ $nnSymbols;
		loadFile @ FileNameJoin[{$basePath, subpath}];
		Protect @@ $nnSymbols;
	];

	fileContext[file_] := Block[{dir, base},
		dir = FileNameTake[file, {-2}];
		base = FileBaseName @ If[StringEndsQ[dir, ".m"], dir, FileNameTake[file]];
		StringJoin["NeuralNetworks`Private`", base, "`"]
	];

	$fileTimings = Association[];
	loadFile[file_] := Block[
		{$Context = fileContext[file], contents, time},
		contents = GeneralUtilities`FileString[file];
		If[!StringStartsQ[contents, "Package[\"NeuralNetworks`\"]"], Return[]];
		contents = StringDrop[contents, 27];
		time = First @ AbsoluteTiming @ Check[
			GeneralUtilities`$CurrentFileName = file;
			ToExpression[contents, InputForm];
			GeneralUtilities`$CurrentFileName = None;
		,		
			errs = GeneralUtilities`FindSyntaxErrors[file];
			If[errs =!= {}, Print[errs]];
			Message[General::nnloaderr];
			Return[$Failed, PreemptProtect];
		];
		$fileTimings[FileNameTake[file]] = time;
	];

	NeuralNetworks`Private`$NNCacheDir = GeneralUtilities`PacletCacheDir["NeuralNetworks"];

	$contexts = {"System`", "Developer`", "Internal`", "GeneralUtilities`", "MXNetLink`", "NeuralNetworks`", "NeuralNetworks`Private`"};
	Block[{$ContextPath = $contexts, NeuralNetworks`Private`$IsLoading = True}, 
		(* load all the ordinary .m code files *)
		GeneralUtilities`CatchFailure[General, Scan[loadFile, $files]];
		(* load the layer + coder definition files *)
		NeuralNetworks`InitializeNeuralNetworks[];
	];
);

saveMX[file_] := (
	NeuralNetworks`ClearCache[];
	DumpSave[file, Evaluate @ Flatten @ {
		"NeuralNetworks`",
		General, (* capture all the new messages we introduced *)
		$nnSymbols
	}]
);

loadMX[file_] := (
	Get[file];
	NeuralNetworks`Private`ReinitializeNeuralNetworks[];
);

If[GeneralUtilities`PacletLoadCached["NeuralNetworks", initNN, saveMX, loadMX] === "Disabled", initNN[]];

(* protect all symbols *)
SetAttributes[Evaluate @ $nnSymbols, {Protected, ReadProtected}];	

(* end of PreemptProtect and Quiet *)
, {RuleDelayed::rhs, General::shdw}];

End[];

EndPackage[];