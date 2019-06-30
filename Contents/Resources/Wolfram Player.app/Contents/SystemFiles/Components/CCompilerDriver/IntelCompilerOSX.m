BeginPackage["CCompilerDriver`IntelCompilerOSX`", 
	{"CCompilerDriver`", "CCompilerDriver`IntelCompiler`"}];

Begin["`Private`"];

Needs["CCompilerDriver`CCompilerDriverBase`"]
Needs["CCompilerDriver`CCompilerDriverRegistry`"]
Needs["CCompilerDriver`IntelCompiler`"]

`$ThisDriver = IntelCompiler

$OptFlag = "-O2"

(* Build innate compile options*)
$CompileOptions = (
        {
                "SystemCompileOptions" ->
                        (* all platforms *)
                        Join[{"-fPIC", $OptFlag, "-framework", "Foundation"}]
        }
);

(* Extend the CCompilerDriverBase Options*)
Options[ $ThisDriver] = DeriveOptions[$CompileOptions];

IntelCompiler["Available"] :=
	With[{fileType = 
		Quiet[FileType[$ThisDriver["ResolveInstallation"][Automatic]]]},
		TrueQ[MemberQ[{Directory, File}, fileType]]
	]

IntelCompiler["Name"][] := "Intel Compiler"

IntelCompiler["Installation"][] := 
	$ThisDriver["ResolveInstallation"][Automatic]

(* We must remove any optimizations, if a debug is to occur.*)
IntelCompiler["DebugSystemOptions"][] := (
        Append[DeleteCases[Options[$ThisDriver, "SystemCompileOptions"][[1,2]], $OptFlag], "-g"]
);

IntelCompiler["CreateObjectFileCommands"][
	errHd_, installation_, compilerName_, outFile_, workDir_, 
	compileOptions_, linkerOptions_, defines_, includePath_, srcFileRules_List, 
	tmSrcFiles_, cFiles_, syslibs_, libs_, libpath_, extraObjects_, 
	targetSystemID_, cleanIntermediate_, mprepOptions_, translib_, language_, opts_] :=
	Module[{cc = compilerCommand[installation, targetSystemID]},

		CommandJoin[
			SetupEnvironment[$ThisDriver, targetSystemID, installation], "\n",

			cc,
			" ", "-c", 
			" -o", QuoteFile[WorkingOutputFile[workDir, outFile]],
			" ", compileOptions,
			" ", defines,
			" ", includePath,
			" ", QuoteFiles[cFiles],
			" 2>&1\n"
		]
	]

IntelCompiler["CreateLibraryCommands"][
	errHd_, installation_, compilerName_, outFile_, workDir_, 
	compileOptions_, linkerOptions_, defines_, includePath_, srcFileRules_List, 
	tmSrcFiles_, cFiles_, syslibs_, libs_, libpath_, extraObjects_, 
	targetSystemID_, cleanIntermediate_, mprepOptions_, translib_, language_, opts_] :=
	Module[{cc = compilerCommand[installation, targetSystemID]},

		CommandJoin[
			SetupEnvironment[$ThisDriver, targetSystemID, installation], "\n",

			cc,
			" ", $CreateDLLFlag,
			" ", "-o ", QuoteFile[WorkingOutputFile[workDir, outFile]],
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

IntelCompiler["CreateExecutableCommands"][
	errHd_, installation_, compilerName_, outFile_, workDir_, 
	compileOptions_, linkerOptions_, defines_, includePath_, srcFileRules_List, 
	tmSrcFiles_, cFiles_, syslibs_, libs_, libpath_, extraObjects_, 
	targetSystemID_, cleanIntermediate_, mprepOptions_, translib_, language_, opts_] :=
	Module[{cc = compilerCommand[installation, targetSystemID]},

		CommandJoin[
			SetupEnvironment[$ThisDriver, targetSystemID, installation], " 2>&1\n",

			MprepCalls[tmSrcFiles, workDir, translib, mprepOptions],

			cc,
			" -o ", QuoteFile[WorkingOutputFile[workDir, outFile]], 
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

exeLinkLibraries["MacOSX-x86" | "MacOSX-x86-64"] := "-lm -lpthread -lstdc++"

exeLinkLibraries[_] := "-lm -lpthread -lrt -lstdc++"

$CreateDLLFlag = Switch[$SystemID,
	"MacOSX-x86" | "MacOSX-x86-64", "-dynamiclib",
	_, "-shared"]

dllLinkLibs["MacOSX-x86"] := "-lstdc++ -lgcc_eh"
(* -lgcc_eh: Link with the GCC exception handler on x86-based Mac *)

dllLinkLibs[_] := "-lstdc++"

SetupEnvironment[driver_, targetSystemID_, installation_] := 
	VCVarsCall[installation, targetSystemID]

(*****************************************************************************)
(* Automatic installation detection *)

$PathDirs = StringSplit[Environment["PATH"], $PlatformPathSeparator]

validCompilerCommandQ[file_] := File === Quiet[FileType[file]]

$CompilerExe = Select[FileNameJoin[{#, "icc"}]& /@ $PathDirs, 
	validCompilerCommandQ, 1] /. {
	{path_String} :> path,
	_ :> $Failed
}

IntelCompiler["Installations"][] := 
	Select[{IntelCompiler["ResolveInstallation"][Automatic]},
		validCompilerCommandQ]

IntelCompiler["ResolveInstallation"][Automatic] := 
	Module[{versions, version, builds, build},
		versions = Sort@Select[FileNames["*", "/opt/intel/Compiler"], 
			DirectoryQ];
		If[Head[versions] =!= List || versions === {},
			Return[$Failed]
		];
		version = Last[versions];
		builds = Sort@Select[FileNames["*", version], DirectoryQ];
		build = Last[builds]
	]

IntelCompiler["ResolveInstallation"][path_String] := path

IntelCompiler["ValidInstallationQ"][installation_] := 
	TrueQ[validCompilerCommandQ[compilerCommand[installation, $SystemID]]] (* TODO when ValidInstallationQ has a targetSystemID parameter change $SystemID to that *)

compilerCommand[installation_String, targetSystemID_] :=
	Select[iccLocations[installation, "icc", targetSystemID], 
		validCompilerCommandQ, 1] /. {
		{path_} :> path,
		_ :> installation (* possibly an invalid icc, but try it anyway *)
	}

iccLocations[installation_, name_, targetSystemID_] := 
	(iccLocations[installation, name, targetSystemID] = 
		{installation, FileNameJoin[{installation, name}], 
			FileNameJoin[{installation, "bin", name}],
			FileNameJoin[{installation, "bin", 
				VCVarsArch[targetSystemID], name}]})

(*****************************************************************************)

VCVarsCall[installation_, targetSystemID_] := 
	StringJoin["source ", QuoteFile[VCVarsPath[installation]], " ", 
		VCVarsArch[targetSystemID]]

VCVarsPath[installation_String] := 
	FileNameJoin[{installation, "bin", "iccvars.sh"}]

VCVarsPath[_] := $Failed

VCVarsArch["MaxOSX-x86"] := "ia32"

VCVarsArch["MacOSX-x86-64"] := "intel64"

frameworkOptionQ[str_String] := StringMatchQ[str, StartOfString ~~ "-framework " ~~ ___];

protocollibframeworkOptionQ[str_String] := 
        (StringMatchQ[str, StartOfString ~~ "-framework" ~~ ___~~ "mathlink"] || StringMatchQ[str, StartOfString ~~ "-framework" ~~ ___~~ "wstp"]);

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

formatLibrary[lib_?frameworkOptionQ, _, _, _] := (
        "-framework "<>QuoteFile[StringTrim[StringReplace[lib,"-framework"->""]]];
)

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

LibraryPathQ[lib_] := 
	StringMatchQ[lib,
		(* Files ending in .a or .so followed by 0 or more .N extensions *)
		(___ ~~ (".a" | (".so" ~~ (("." ~~ NumberString) ...)))) | 
		(* Or files containing a directory separator *)
		(___ ~~ "/" ~~ ___)
	]

IntelCompiler[method_][args___] := 
	CCompilerDriver`CCompilerDriverBase`BaseDriver[method][args]

CCompilerRegister[$ThisDriver]

End[];

EndPackage[];
