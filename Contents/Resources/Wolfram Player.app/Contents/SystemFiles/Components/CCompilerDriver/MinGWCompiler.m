BeginPackage["CCompilerDriver`MinGWCompiler`", {"CCompilerDriver`"}]

MinGWCompiler::usage = "MinGWCompiler[src, name] compiles the code in src into a DLL and returns the full path of the DLL.";

Begin["`Private`"]

`$ThisDriver = MinGWCompiler

Needs["CCompilerDriver`CCompilerDriverBase`"]
Needs["CCompilerDriver`CCompilerDriverRegistry`"]

CCompilerRegister[ $ThisDriver, {"Windows"}]

Options[ $ThisDriver] = 
	DeriveOptions[
		{
			"SystemCompileOptions" -> {"-O2"} 
		}
	]

MinGWCompiler["Available"] :=
	TrueQ[Directory ===
		Quiet[FileType[$ThisDriver["ResolveInstallation"][Automatic]]]]

MinGWCompiler["Name"][] := "MinGW"

MinGWCompiler["Installation"][] := $ThisDriver["ResolveInstallation"][Automatic]

MinGWCompiler["LibraryPathFlag"][] := "-L"

MinGWCompiler["ResolveSystemLibraries"]["Windows", translib_] := 
        If[!MatchQ[translib, "WSTP"], {"ml32i4m"}, {"wstp32i4m"}]

MinGWCompiler["ResolveSystemLibraries"]["Windows-x86-64", translib_] := 
        If[!MatchQ[translib, "WSTP"], {"ml64i4m"}, {"wstp64i4m"}]

MinGWCompiler["CreateObjectFileCommands"][
	errHd_, installation_, compilerName_, outFile_, workDir_, 
	compileOptions_, linkerOptions_, defines_, includePath_, srcFileRules_List, 
	tmSrcFiles_, cFiles_, syslibs_, libs_, libpath_, extraObjects_, 
	targetSystemID_, cleanIntermediate_, mprepOptions_, translib_, language_, opts_] :=
	Module[{gccExe = compilerCommand[installation, compilerName]},

		{
			SetupEnvironment[gccExe],
			CommandJoin[
				gccExe, 
				" -c", 
				" -o ", QuoteFile[WorkingOutputFile[workDir, outFile]],
				" ", compilerBitnessOption[targetSystemID], 
				" ", compileOptions,
				" ", defines,
				" ", includePath,
				" ", QuoteFiles[cFiles], 
				" 2>&1\n"
			]
		}
	]

MinGWCompiler["CreateLibraryCommands"][
	errHd_, installation_, compilerName_, outFile_, workDir_, 
	compileOptions_, linkerOptions_, defines_, includePath_, srcFileRules_List, 
	tmSrcFiles_, cFiles_, syslibs_, libs_, libpath_, extraObjects_, 
	targetSystemID_, cleanIntermediate_, mprepOptions_, translib_, language_, opts_] :=
	Module[{gccExe = compilerCommand[installation, compilerName]},

		{
			SetupEnvironment[gccExe],
			CommandJoin[
				gccExe, 
				" ", $CreateDLLFlag,
				" ", "-o ", QuoteFile[WorkingOutputFile[workDir, outFile]],
				" ", compilerBitnessOption[targetSystemID], 
				" ", compileOptions,
				" ", defines,
				" ", includePath,
				" ", QuoteFiles[cFiles], 
				" ", QuoteFiles[extraObjects],
				Map[" -Xlinker " <> # &, linkerOptions],
				" ", libpath,
				" ", formatLibraries[syslibs, libs], 
				" 2>&1\n"
			]
		}
	]

MinGWCompiler["CreateExecutableCommands"][
	errHd_, installation_, compilerName_, outFile_, workDir_, 
	compileOptions_, linkerOptions_, defines_, includePath_, srcFileRules_List, 
	tmSrcFiles_, cFiles_, syslibs_, libs_, libpath_, extraObjects_, 
	targetSystemID_, cleanIntermediate_, mprepOptions_, translib_, language_, opts_] :=
	Module[{gccExe = compilerCommand[installation, compilerName]},

		{
			SetupEnvironment[gccExe],
			CommandJoin[
				MprepCalls[tmSrcFiles, workDir, translib, mprepOptions],
	
				gccExe, 
				" -o ", QuoteFile[WorkingOutputFile[workDir, outFile]], 
			        " ", compilerBitnessOption[targetSystemID],
				" ", compileOptions,
				" ", defines,
				" ", includePath,
				" ", QuoteFiles[cFiles], 
				" ", QuoteFiles[extraObjects], 
				Map[" -Xlinker " <> # &, linkerOptions],
				" ", libpath,
				" ", formatLibraries[syslibs, libs], 
				" ", exeLinkLibraries[targetSystemID], 
				" 2>&1\n"
			]
		}
	]

SetupEnvironment[gccPath_] := 
	"set PATH="<>DirectoryName[gccPath]<>";%PATH%"

exeLinkLibraries[_] := "-lgdi32"

$CreateDLLFlag = "-shared"

compilerBitnessOption[target_] := compilerBitnessOption[Bits[target]]

compilerBitnessOption[b:(32|64)] := "-m"<>ToString[b]

(*****************************************************************************)
(* Automatic installation detection *)

$PathDirs = StringSplit[Environment["PATH"], $PlatformPathSeparator]

$GCC = "gcc"<>$PlatformExeExtension

validGCCExeQ[path_] := File === Quiet[FileType[path]]

MinGWCompiler["ResolveInstallation"][Automatic] := findCompilerInstallation[]

findCompilerInstallation[] := 
	Select[Join[$PathDirs, gccLocations["C:/MinGW", $GCC],
		gccLocations[Environment["MINGW"], $GCC]], validGCCExeQ, 1] /. 
	{
		{path_String} :> FileNameDrop[path, -1],
		_ :> $Failed
	}

MinGWCompiler["Installations"][] := 
	Select[Join[$PathDirs, gccLocations["C:/MinGW", $GCC],
		gccLocations[Environment["MINGW"], $GCC]], validGCCExeQ]
	
MinGWCompiler["ResolveInstallation"][path_String] := path

MinGWCompiler["ResolveCompilerName"][Automatic] := $GCC

MinGWCompiler["ResolveCompilerName"][name_] := name

MinGWCompiler["ValidInstallationQ"][installation_] := 
	TrueQ[validGCCExeQ[compilerCommand[installation]]]

MinGWCompiler["ValidInstallationQ"][installation_, name_, ___] := 
	TrueQ[validGCCExeQ[compilerCommand[installation, name]]]

compilerCommand[installation_String] := 
	compilerCommand[installation, Automatic]

compilerCommand[installation_String, Automatic] := 
	compilerCommand[installation, $GCC]

compilerCommand[installation_String, name_String] :=
	Select[gccLocations[installation, name], validGCCExeQ, 1] /. {
		{path_} :> path,
		_ :> FileNameJoin[{installation, name}] (*possibly invalid, try anyway*)
	}

gccLocations[installation_, name_] := 
(
	gccLocations[installation, name] = 
		{
			installation,
			FileNameJoin[{installation, name}], 
			FileNameJoin[{installation, "bin", name}]
		}
)

(*****************************************************************************)
formatLibraries[libs_List] := 
	Riffle[formatLibrary /@ libs, " "]

formatLibraries[libs_List, libs2_List] := formatLibraries[Join[libs, libs2]]

formatLibrary[lib_] := 
	If[LibraryPathQ[lib], 
		(* lib appears to be an explicit library file path, just quote it *)
		QuoteFile[lib], 
		(* lib appears to be a simple lib name, pass it to -l *)
		If[StringMatchQ[lib, "-l" ~~ ___],
			QuoteFile[lib],
			"-l"<>QuoteFile[lib]
		]
	]

LibraryPathQ[lib_] := 
	StringMatchQ[lib,
		(* Files ending in .a or .so followed by 0 or more .N extensions *)
		(___ ~~ (".lib" | ".a" | (".so" ~~ (("." ~~ NumberString) ...)))) | 
		(* Or files containing a directorty separator *)
		(___ ~~ "/" ~~ ___)
	]

MinGWCompiler[method_][args___] := 
	CCompilerDriver`CCompilerDriverBase`BaseDriver[method][args]

CCompilerRegister[$ThisDriver]

End[]
EndPackage[]
