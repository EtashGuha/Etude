BeginPackage["NeuralFunctions`"]
Begin["NeuralFunctions`Private`Bootstrap`"];

(* load dependancies *)
Quiet @ Needs["GeneralUtilities`"];
Quiet @ Needs["NeuralNetworks`"];

(* evaluating a NeuralNetworks` symbol to work around hidden imports *)
Quiet[NetChain];

(* find all symbols exported by this paclet *)
syms = PacletExportedSymbols["NeuralFunctions"];
(* create all the exported symbols in context *)
ToExpression[syms, StandardForm, Hold];
(* clear protected symbols *)
Unprotect @@ syms;
ClearAll @@ syms;

(* obtain files to load *)
$basePath = DirectoryName[$InputFileName, 2];
subPath[p__] := FileNameJoin[{$basePath, p}];
$files = FileNames["*.m", $basePath, Infinity];
$ignoreFiles = Flatten[{
	subPath["Kernel", "init.m"],  
	subPath["PacletInfo.m"],
	FileNames["*", subPath["Resources"], Infinity]
}];
$loadFirstFiles = {
	subPath["Kernel", "Common.m"]
};

$files = Complement[$files, $ignoreFiles, $loadFirstFiles];
$files = Join[$loadFirstFiles, Sort[$files]];

(* scan for scoped and exported symbols *)
$lines = StringSplit[StringTrim @ FindList[$files, {"PackageScope", "PackageExport"}], {"[", "]", "\""}]
$private = Cases[$lines, {"PackageScope", _, name_} :> name];
$public =  Cases[$lines, {"PackageExport", _, name_} :> name];
$public = Complement[$public, Names["System`*"]];

(* create symbols in the right context *)
createInContext[context_, names_] := Block[{$ContextPath = {}, $Context = context}, ToExpression[names, InputForm, Hold]];

createInContext["NeuralFunctions`", $public];
createInContext["NeuralFunctions`Private`", $private];

(* load files *)
$contexts = {"System`", "Developer`", "Internal`", "GeneralUtilities`", "MXNetLink`", "NeuralNetworks`", "NeuralFunctions`", "NeuralFunctions`Private`"};

Block[{$ContextPath = $contexts}, 
	loadFile[file_, fileID_] := Block[
		{$Context = "NeuralFunctions`Private`file" <> ToString[First@fileID] <> FileBaseName[file] <> "`"},
		contents = FileString[file];
		Check[
			ToExpression[contents, InputForm],
			Print["Message occurred during file: ", file];
		];
	];
	ScanIndexed[loadFile, $files];
];

SetAttributes[#, {Protected, ReadProtected}]& /@ syms;

End[];
EndPackage[];

