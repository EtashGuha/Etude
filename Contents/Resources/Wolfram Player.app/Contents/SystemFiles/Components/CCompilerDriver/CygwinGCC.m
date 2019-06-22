(* ::Package:: *)

BeginPackage["CCompilerDriver`CygwinGCC`", {"CCompilerDriver`"}]

CygwinGCC::usage = "CygwinGCC is a symbol that represents the Cygwin GCC compiler for compiling C code."

Begin["`Private`"]
`$ThisDriver = CygwinGCC

Needs["CCompilerDriver`CCompilerDriverBase`"]
Needs["CCompilerDriver`CCompilerDriverRegistry`"]

CCompilerRegister[ $ThisDriver, {"Windows"}]

Options[ $ThisDriver] = 
	DeriveOptions[
		{
		"SystemCompileOptions" -> {"-mno-cygwin", "-O2"} 
		}
	]

CygwinGCC["Available"] :=
	With[{fileType = 
		Quiet[FileType[$ThisDriver["ResolveInstallation"][Automatic]]]},
		TrueQ[MemberQ[{Directory, File}, fileType]]
	]

CygwinGCC["Name"][] := "Cygwin GCC"

CygwinGCC["Installation"][] := $ThisDriver["ResolveInstallation"][Automatic]

CygwinGCC["LibraryPathFlag"][] := "-L"

CygwinGCC["ResolveSystemLibraries"]["Windows", translib_] :=
        If[!MatchQ[translib, "WSTP"], {"ml32i4m"}, {"wstp32i4m"}]

CygwinGCC["ResolveSystemLibraries"]["Windows-x86-64", translib_] :=
        If[!MatchQ[translib, "WSTP"], {"ml64i4m"}, {"wstp64i4m"}]

CygwinGCC["CreateObjectFileCommands"][
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

CygwinGCC["CreateLibraryCommands"][
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
				" ", libpath,
				" ", formatLibraries[syslibs, libs], 
				" 2>&1\n"
			]
		}
	]

CygwinGCC["CreateExecutableCommands"][
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
				" ", libpath,
				" ", formatLibraries[syslibs, libs], 
				" ", dllLinkLibs[targetSystemID], 
				" 2>&1\n"
			]
		}
	]

$CreateDLLFlag = "-shared"

dllLinkLibs[_] := " -lm -lpthread -lstdc++ -mwindows"

compilerBitnessOption[target_] := compilerBitnessOption[Bits[target]]

compilerBitnessOption[b:(32|64)] := "-m"<>ToString[b]

(*****************************************************************************)
(* Automatic installation detection *)

SetupEnvironment[gccPath_] := 
	CommandJoin[{
		pathCommand[gccPath], "\n",
		"set CYGWIN=%CYGWIN% nodosfilewarning\n"
	}]

pathCommand[gccPath_String] := 
	"set PATH="<>StringDrop[DirectoryName[gccPath],-1]<>";%PATH%"

$PathDirs = StringSplit[Environment["PATH"], $PlatformPathSeparator]

$GCC = "gcc-3.exe" (* only gcc-3 supports -mno-cygwin and can be loaded into the Mathematica kernel *)

validGCCExeQ[path_] := File === Quiet[FileType[path]]

CygwinGCC["ResolveInstallation"][Automatic] := findCompilerInstallation[]

findCompilerInstallation[] := 
	Select[Join[$PathDirs, gccLocations["C:/Cygwin", $GCC]], validGCCExeQ, 1] /. 
	{
		{path_String} :> FileNameDrop[path, -1],
		_ :> $Failed
	}

CygwinGCC["Installations"][] := 
	Select[Join[$PathDirs, gccLocations["C:/Cygwin", $GCC]], validGCCExeQ]

CygwinGCC["ResolveInstallation"][path_String] := path

CygwinGCC["ResolveCompilerName"][Automatic] := $GCC

CygwinGCC["ResolveCompilerName"][name_] := name

CygwinGCC["ValidInstallationQ"][installation_] := 
	TrueQ[validGCCExeQ[compilerCommand[installation]]]

CygwinGCC["ValidInstallationQ"][installation_, name_, ___] := 
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
		(* Files ending in .lib, .a or .so followed by 0 or more .N extensions *)
		(___ ~~ (".lib" | ".a" | (".so" ~~ (("." ~~ NumberString) ...)))) | 
		(* Or files containing a directorty separator *)
		(___ ~~ "/" ~~ ___)
	]

CygwinGCC[method_][args___] := 
	CCompilerDriver`CCompilerDriverBase`BaseDriver[method][args]

CCompilerRegister[$ThisDriver]
End[];

EndPackage[];
