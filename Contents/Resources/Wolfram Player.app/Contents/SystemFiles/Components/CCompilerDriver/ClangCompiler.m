(* Mathematica Package *)
BeginPackage["CCompilerDriver`ClangCompiler`", {"CCompilerDriver`"}]

ClangCompiler::usage = "ClangCompiler[ string, name] compiles the code in string into a DLL and returns the full path of the DLL."
ClangCompiler::nodir = "`1` was not found or is not a directory."

Begin["`Private`"] (* Begin Private Context *) 

`$ThisDriver = ClangCompiler;
Needs["CCompilerDriver`CCompilerDriverBase`"]
Needs["CCompilerDriver`CCompilerDriverRegistry`"]

$TRACE = True;
trace[expr_] := (
	If[SameQ[$TRACE,True],
		Print[expr];
	]
);

(* Build Options *)
(* Optimization Level 2*)
$OptFlag = "-O2";

(* Build innate compile options*)
$CompileOptions = (
	{
		"SystemCompileOptions" -> 
			(* all platforms *)
			Join[{"-fPIC", $OptFlag},
				If[$OperatingSystem === "MacOSX",
					{"-mmacosx-version-min=10.10","-framework", "Foundation"},
					{}
				]
			]
	}
);

(* Extend the CCompilerDriverBase Options*)
Options[ $ThisDriver] = DeriveOptions[$CompileOptions];

(*****************************************************************************)
(* Automatic installation detection *)

(*Evaluate $PATH  in a Mathematica List*)
$PathDirs = StringSplit[Environment["PATH"], $PlatformPathSeparator];

(* Detirmine the default executable file extentsion if one exists*)
$Clang = "clang"<>$PlatformExeExtension;

(* Test weather the executable is a file *)
validClangExeQ[path_] := File === Quiet[FileType[path]]

(* Build the default clang location given the $PathDirs *)
$DefaultCompilerInstallation =( 
	Select[FileNameJoin[{#, $Clang}]& /@ $PathDirs, validClangExeQ, 1] /. {
		{path_String} :> FileNameDrop[path, -1],
		_ :> $Failed
	}
);
(* The various install locations that Clang is located*)
ClangCompiler["Installations"][] := 
	FileNameDrop[#, -1]& /@
		Select[FileNameJoin[{#, $Clang}]& /@ $PathDirs, validClangExeQ];

(* Either a the default installation is picked or an given one.*)
ClangCompiler["ResolveInstallation"][Automatic] := $DefaultCompilerInstallation;
ClangCompiler["ResolveInstallation"][path_String] := path;

(* We can use the natural executable name or a given executable. *)
ClangCompiler["ResolveCompilerName"][Automatic] := $Clang;
ClangCompiler["ResolveCompilerName"][name_] := name;

(* 
    Using an installation and the compiler name/executable we produce an
    absolute path of the Clang Compiler Front-End.

*)
locations[installation_, name_] := (
	locations[installation, name] = {
			installation,
			FileNameJoin[{installation, name}], 
			FileNameJoin[{installation, "bin", name}]
		}
);
(*
	Generate the Clang command that will became executed. 
	Ensuring that the command is the appropiate one, through test.
*)
compilerCommand[installation_String] := compilerCommand[installation, Automatic, Automatic];

compilerCommand[installation_String, Automatic, Automatic | "C"] := compilerCommand[installation, $Clang];

compilerCommand[installation_String, Automatic, "C++"] := compilerCommand[installation, $Clang];

compilerCommand[installation_String, name_, _] := compilerCommand[installation, name];

compilerCommand[installation_String, name_String] := (
	Select[locations[installation, name], validClangExeQ, 1] /. {
		{path_} :> path,
		_ :> FileNameJoin[{installation, name}] (*possibly invalid, try anyway*)
	}
);

(* Test to ensure that our expected installation is valid.*)
ClangCompiler["ValidInstallationQ"][installation_] := TrueQ[validClangExeQ[compilerCommand[installation]]];
(* Test to ensure that our expected installation is valid, with the given name/exectuable*)
ClangCompiler["ValidInstallationQ"][installation_, name_, ___] := (
	TrueQ[validClangExeQ[compilerCommand[installation, name]]]
);
(*****************************************************************************)

(* Using our above installation detection, detirmine weather we have a  valid directory or file.*)
ClangCompiler["Available"] := With[{fileType = Quiet[FileType[$ThisDriver["ResolveInstallation"][Automatic]]]},
		TrueQ[MemberQ[{Directory, File}, fileType]]
];

ClangCompiler["Name"][] := "Clang";

ClangCompiler["Installation"][] := $ThisDriver["ResolveInstallation"][Automatic];

ClangCompiler["LibraryPathFlag"][] := "-L";

(* We must remove any optimizations, if a debug is to occur.*)
ClangCompiler["DebugSystemOptions"][] := (
	Append[DeleteCases[Options[$ThisDriver, "SystemCompileOptions"][[1,2]], $OptFlag], "-g"]
);

(* Allows for explicit Language choose of C or C++*)
ClangCompiler["LanguageOption"]["C"] := " -x c";
ClangCompiler["LanguageOption"]["C++"] := " -x c++";
ClangCompiler["LanguageOption"][_] := "";

exeLinkLibraries["MacOSX-x86" | "MacOSX-x86-64"] = "-lm -lpthread -lc++";
exeLinkLibraries["Linux" | "Linux-x86-64"] = "-lm -lpthread -lrt -lstdc++ -ldl -luuid";
exeLinkLibraries[_] := "-lm -lpthread -lrt -lstdc++ -ldl";

(* Evaluate what the flag is going to be for shared libraries *)
$CreateDLLFlag = (
	Switch[$SystemID,
	"MacOSX-x86" | "MacOSX-x86-64",
		"-dynamiclib",
	_,
		"-shared"]
);

dllLinkLibs = "-lc++";

(* Detirmine the target architecture for compilation *)
compilerBitnessOption["Linux-ARM"] := ""
compilerBitnessOption[target_] := compilerBitnessOption[Bits[target]]
compilerBitnessOption[b:(32|64)] := "-m"<>ToString[b]

(* Allows Clang to only produce object files*)
ClangCompiler["CreateObjectFileCommands"][
	errHd_, installation_, compilerName_, outFile_, workDir_, 
	compileOptions_, linkerOptions_, defines_, includePath_, srcFileRules_List, 
	tmSrcFiles_, cFiles_, syslibs_, libs_, libpath_, extraObjects_, 
	targetSystemID_, cleanIntermediate_, mprepOptions_, translib_, language_, opts_] :=
	Module[{},
		CommandJoin[
			compilerCommand[installation, compilerName, language],
			" -c", 
			" -o ", QuoteFile[WorkingOutputFile[workDir, outFile]],
			" ", compilerBitnessOption[targetSystemID], 
			$ThisDriver["LanguageOption"][language],
			" ", compileOptions,
			" ", defines,
			" ", includePath,
			" ", QuoteFiles[cFiles], 
			" 2>&1\n"
		]
	];

(* TODO: Unsure what this means*)
If[$OperatingSystem === "Unix",
	ClangCompiler["LinkWithMathLink"][CreateLibrary] := False;
]


(* Produces a command that builds libraries *)
ClangCompiler["CreateLibraryCommands"][
	errHd_, installation_, compilerName_, outFile_, workDir_, 
	compileOptionsIn_, linkerOptions_, defines_, includePath_, srcFileRules_List,
	tmSrcFiles_, cFiles_, syslibs_, libs_, libpath_, extraObjects_, 
	targetSystemID_, cleanIntermediate_, mprepOptions_, translib_, language_, opts_] :=
	Module[{libName, compileOptions},
		libName = FileNameTake[outFile];
		compileOptions = Join[{"-install_name @rpath/" <> libName, " "}, compileOptionsIn];
		CommandJoin[
			compilerCommand[installation, compilerName, language],
			" ", $CreateDLLFlag,
			" ", "-o ", QuoteFile[WorkingOutputFile[workDir, outFile]],
			" ", compilerBitnessOption[targetSystemID],
			$ThisDriver["LanguageOption"][language],
			" ", compileOptions,
			" ", defines,
			" ", includePath,
			" ", QuoteFiles[cFiles],
			" ", QuoteFiles[extraObjects],
			Map[" -Xlinker " <> # &, linkerOptions],
			" ", libpath,
			" ", formatLibraries[syslibs, libs, False, targetSystemID, translib],
			" ", dllLinkLibs,
			" 2>&1\n"
		]
	];

(* Generates a command that produces an exectable *)
ClangCompiler["CreateExecutableCommands"][
	errHd_, installation_, compilerName_, outFile_, workDir_, 
	compileOptions_, linkerOptions_, defines_, includePath_, srcFileRules_List, 
	tmSrcFiles_, cFiles_, syslibs_, libs_, libpath_, extraObjects_, 
	targetSystemID_, cleanIntermediate_, mprepOptions_, translib_, language_, opts_] :=
	Module[{},

		CommandJoin[
			MprepCalls[tmSrcFiles, workDir, translib, mprepOptions],

			compilerCommand[installation, compilerName, language],
			" -o ", QuoteFile[WorkingOutputFile[workDir, outFile]], 
			" ", compilerBitnessOption[targetSystemID], 
			$ThisDriver["LanguageOption"][language],
			" ", compileOptions,
			" ", defines,
			" ", includePath,
			" ", QuoteFiles[cFiles],
			" ", QuoteFiles[extraObjects],
			Map[" -Xlinker " <> # &, linkerOptions],
			" ", libpath,
			" ", formatLibraries[syslibs, libs, True, targetSystemID, translib],
			" ", exeLinkLibraries[targetSystemID],
			" 2>&1\n"
		]
	];

frameworkOptionQ[str_String] := StringMatchQ[str, StartOfString ~~ "-framework " ~~ ___];

protocollibframeworkOptionQ[str_String] := StringMatchQ[str, StartOfString ~~ "-framework" ~~ ___~~ ("mathlink" | "wstp")];

formatLibraries[libs_List, execQ_, systemID_, translib_] :=(
	(* Ensure that we never receive Null in the List *)
	DeleteCases[Riffle[formatLibrary[#,execQ, systemID, translib]& /@ libs, " "], Null]
);

formatLibraries[libs_List, libs2_List, execQ_, systemID_, translib_] := formatLibraries[Join[libs, libs2], execQ, systemID, translib]

formatLibrary[lib_?protocollibframeworkOptionQ, execQ_, systemID_, translib_] := (
	If[execQ === False,
		"-framework "<>QuoteFile[StringTrim[StringReplace[lib,"-framework"->""]]],
		If[!MatchQ[translib, "WSTP"],
			"-lMLi4 -L"<> QuoteFile[MathLinkCompilerAdditionsPath[systemID]], 
			"-lWSTPi4 -L"<> QuoteFile[WSTPCompilerAdditionsPath[systemID]]
		]
	]
);

formatLibrary[lib_?frameworkOptionQ, _, _, _] := 
	"-framework "<>QuoteFile[StringTrim[StringReplace[lib,"-framework"->""]]];

formatLibrary[lib_, _, _, _] := (
	If[LibraryPathQ[lib], 
		(* lib appears to be an explicit library file path, just quote it *)
		QuoteFile[lib], 
		(* lib appears to be a simple lib name, pass it to -l *)
		If[StringMatchQ[lib, ___~~"-l"],
			QuoteFile[lib],
			"-l"<>QuoteFile[lib]
		]
	]
);

LibraryPathQ[lib_] := (
	StringMatchQ[lib,
		(* Files ending in .a or .so followed by 0 or more .N extensions *)
		(___ ~~ (".a" | (".so" ~~ (("." ~~ NumberString) ...)))) | 
		(* Or files containing a directory separator *)
		(___ ~~ "/" ~~ ___)
	]
);

(*
	Fallback to the base implementation of CCompilerDriverBase, 
	assuming the method can't be matched.
*)
ClangCompiler[method_][args___] := CCompilerDriver`CCompilerDriverBase`BaseDriver[method][args]

CCompilerRegister[$ThisDriver]

End[] (* End Private Context *)

EndPackage[]
