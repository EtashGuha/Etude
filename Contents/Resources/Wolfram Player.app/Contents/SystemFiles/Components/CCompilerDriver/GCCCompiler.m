BeginPackage["CCompilerDriver`GCCCompiler`", {"CCompilerDriver`"}]

GCCCompiler::usage = "GCCCompiler[ string, name] compiles the code in string into a DLL and returns the full path of the DLL."

GCCCompiler::nodir = "`1` was not found or is not a directory."

Begin["`Private`"]

`$ThisDriver = GCCCompiler

Needs["CCompilerDriver`CCompilerDriverBase`"]
Needs["CCompilerDriver`CCompilerDriverRegistry`"]

$OptFlag = "-O2"

Options[ $ThisDriver] = 
	DeriveOptions[
		{
			"SystemCompileOptions" -> 
				Join[{"-fPIC", $OptFlag}, (* all platforms *)
					If[$OperatingSystem === "MacOSX",
						{"-mmacosx-version-min=10.10",
						"-framework", "Foundation"},
						{}
					]
				]
		}
	];

GCCCompiler["Available"] :=
	With[{fileType = 
		Quiet[FileType[$ThisDriver["ResolveInstallation"][Automatic]]]},
		TrueQ[MemberQ[{Directory, File}, fileType]]
	]

GCCCompiler["Name"][] := "GCC"

GCCCompiler["Installation"][] := $ThisDriver["ResolveInstallation"][Automatic]

GCCCompiler["LibraryPathFlag"][] := "-L"

GCCCompiler["DebugSystemOptions"][] := 
	Append[
		DeleteCases[Options[$ThisDriver, "SystemCompileOptions"][[1,2]], 
			$OptFlag],
		"-g"]

GCCCompiler["LanguageOption"]["C"] := " -x c"
GCCCompiler["LanguageOption"]["C++"] := " -x c++"

GCCCompiler["LanguageOption"][_] := ""

GCCCompiler["CreateObjectFileCommands"][
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
	]

If[$OperatingSystem === "Unix",
	GCCCompiler["LinkWithMathLink"][CreateLibrary] := False
]

GCCCompiler["CreateLibraryCommands"][
	errHd_, installation_, compilerName_, outFile_, workDir_, 
	compileOptions_, linkerOptions_, defines_, includePath_, srcFileRules_List, 
	tmSrcFiles_, cFiles_, syslibs_, libs_, libpath_, extraObjects_, 
	targetSystemID_, cleanIntermediate_, mprepOptions_, translib_, language_, opts_] :=
	Module[{},

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
			" ", dllLinkLibs[targetSystemID],
			" 2>&1\n"
		]
	]

GCCCompiler["CreateExecutableCommands"][
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
	]

exeLinkLibraries["Linux" | "Linux-x86-64" | "Linux-ARM"] := 
	"-lm -lpthread -lrt -lstdc++ -ldl -luuid"

exeLinkLibraries["MacOSX-x86" | "MacOSX-x86-64"] := 
	"-lm -lpthread -lstdc++"

exeLinkLibraries[_] := "-lm -lpthread -lrt -lstdc++ -ldl"

$CreateDLLFlag = Switch[$SystemID,
	"MacOSX-x86" | "MacOSX-x86-64", "-dynamiclib",
	_, "-shared"]

dllLinkLibs["MacOSX-x86"] := "-lstdc++ -lgcc_eh"
(* -lgcc_eh: Link with the GCC exception handler on x86-based Mac *)

dllLinkLibs["MacOSX-x86-64"] := "-lstdc++"

dllLinkLibs[_] := ""

compilerBitnessOption["Linux-ARM"] := ""

compilerBitnessOption[target_] := compilerBitnessOption[Bits[target]]

compilerBitnessOption[b:(32|64)] := "-m"<>ToString[b]

(*****************************************************************************)
(* Automatic installation detection *)

$PathDirs = StringSplit[Environment["PATH"], $PlatformPathSeparator]

$GCC = "gcc"<>$PlatformExeExtension

$GPlusPlus = "g++"<>$PlatformExeExtension

validGCCExeQ[path_] := File === Quiet[FileType[path]]

$DefaultCompilerInstallation = 
	Select[FileNameJoin[{#, $GCC}]& /@ $PathDirs, validGCCExeQ, 1] /. {
		{path_String} :> FileNameDrop[path, -1],
		_ :> $Failed
	}

GCCCompiler["Installations"][] := 
	FileNameDrop[#, -1]& /@
		Select[FileNameJoin[{#, $GCC}]& /@ $PathDirs, validGCCExeQ]

GCCCompiler["ResolveInstallation"][Automatic] := $DefaultCompilerInstallation

GCCCompiler["ResolveInstallation"][path_String] := path

GCCCompiler["ResolveCompilerName"][Automatic] := $GCC

GCCCompiler["ResolveCompilerName"][name_] := name

GCCCompiler["ValidInstallationQ"][installation_] := 
	TrueQ[validGCCExeQ[compilerCommand[installation]]]

GCCCompiler["ValidInstallationQ"][installation_, name_, ___] := 
	TrueQ[validGCCExeQ[compilerCommand[installation, name]]]

compilerCommand[installation_String] := compilerCommand[installation, Automatic, Automatic]

compilerCommand[installation_String, Automatic, Automatic | "C"] := 
	compilerCommand[installation, $GCC]

compilerCommand[installation_String, Automatic, "C++"] :=
	compilerCommand[installation, $GPlusPlus]

compilerCommand[installation_String, name_, _] := 
	compilerCommand[installation, name]

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
frameworkOptionQ[str_String] := StringMatchQ[str, StartOfString ~~ "-framework " ~~ ___]

protocollibframeworkOptionQ[str_String] :=
        StringMatchQ[str, StartOfString ~~ "-framework" ~~ ___~~ "mathlink"] || StringMatchQ[str, StartOfString ~~ "-framework" ~~ ___~~ "wstp"]

formatLibraries[libs_List, execQ_, systemID_, translib_] :=(
	(* Ensure that we never receive Null in the List *)
	DeleteCases[Riffle[formatLibrary[#,execQ, systemID, translib]& /@ libs, " "], Null]
)

formatLibraries[libs_List, libs2_List, execQ_, systemID_, translib_] := formatLibraries[Join[libs, libs2], execQ,  systemID, translib]

formatLibrary[lib_?protocollibframeworkOptionQ, execQ_, systemID_, translib_] := (
	If[execQ === False,
		"-framework "<>QuoteFile[StringTrim[StringReplace[lib,"-framework"->""]]],
		If[translib === "WSTP",
			"-lWSTPi4 -L"<> QuoteFile[WSTPCompilerAdditionsPath[systemID]],
			"-lMLi4 -L"<> QuoteFile[MathLinkCompilerAdditionsPath[systemID]]
		]
	]
)

formatLibrary[lib_?frameworkOptionQ, _, _, _] := "-framework "<>QuoteFile[StringTrim[StringReplace[lib,"-framework"->""]]]

formatLibrary[lib_, _, _, _] := 
	If[LibraryPathQ[lib], 
		(* lib appears to be an explicit library file path, just quote it *)
		QuoteFile[lib], 
		(* lib appears to be a simple lib name, pass it to -l *)
		If[StringMatchQ[lib, ___~~"-l"],
			QuoteFile[lib],
			"-l"<>QuoteFile[lib]
		]
	]

LibraryPathQ[lib_] := 
	StringMatchQ[lib,
		(* Files ending in .a or .so followed by 0 or more .N extensions *)
		(___ ~~ (".a" | (".so" ~~ (("." ~~ NumberString) ...)))) | 
		(* Or files containing a directory separator *)
		(___ ~~ "/" ~~ ___)
	]

GCCCompiler[method_][args___] := 
	CCompilerDriver`CCompilerDriverBase`BaseDriver[method][args]

CCompilerRegister[$ThisDriver]

End[] (* `Private` *)

EndPackage[]
