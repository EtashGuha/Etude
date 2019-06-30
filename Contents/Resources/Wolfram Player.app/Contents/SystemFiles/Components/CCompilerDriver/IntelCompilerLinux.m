BeginPackage["CCompilerDriver`IntelCompilerLinux`", 
	{"CCompilerDriver`", "CCompilerDriver`IntelCompiler`"}];

Begin["`Private`"];

Needs["CCompilerDriver`CCompilerDriverBase`"]
Needs["CCompilerDriver`CCompilerDriverRegistry`"]
Needs["CCompilerDriver`IntelCompiler`"]

`$ThisDriver = IntelCompiler

Options[ $ThisDriver] = 
	DeriveOptions[
		{
			"SystemCompileOptions" -> {"-fPIC", "-static-intel", 
				"-no-intel-extensions"}
		}
	];

IntelCompiler["Available"] :=
	With[{fileType = 
		Quiet[FileType[$ThisDriver["ResolveInstallation"][Automatic]]]},
		TrueQ[MemberQ[{Directory, File}, fileType]]
	]

IntelCompiler["Name"][] := "Intel Compiler"

IntelCompiler["Installation"][] := 
	$ThisDriver["ResolveInstallation"][Automatic]

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
			" ", formatLibraries[syslibs, libs], 
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
			" ", formatLibraries[syslibs, libs],
			" ", exeLinkLibraries[targetSystemID], 
			" 2>&1\n"
		]
	]

exeLinkLibraries[_] := "-lm -lpthread -lrt -ldl -luuid"

$CreateDLLFlag = "-shared"

dllLinkLibs[_] := ""

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
	FileNameDrop[#, -1]& /@
		Select[FileNameJoin[{#, "icc"}]& /@ $PathDirs, validCompilerCommandQ]

IntelCompiler["ResolveInstallation"][Automatic] := $CompilerExe

IntelCompiler["ResolveInstallation"][path_String] := path

IntelCompiler["ValidInstallationQ"][installation_] := 
	TrueQ[validCompilerCommandQ[compilerCommand[installation, $SystemID]]] (* TODO when ValidInstallationQ has a targetSystemID parameter change $SystemID to that *)

compilerCommand[installation_String, targetSystemID_] :=
	Select[iccLocations[installation, "icc", targetSystemID], 
		validCompilerCommandQ, 1] /. {
		{path_} :> path,
		_ :> installation (* possibly invalid, but try it anyway *)
	}
 
iccLocations[installation_, name_, targetSystemID_] := 
	(iccLocations[installation, name, targetSystemID] = 
		{installation, FileNameJoin[{installation, name}], 
			FileNameJoin[{installation, "bin", name}],
			FileNameJoin[{installation, "bin", 
				VCVarsArch[targetSystemID], name}]})

(*****************************************************************************)

cShellFamilyQ[] := 
	With[{pid = ToString[$ProcessID]}, 
		pid === 
		StringTrim[
			Import["!echo setenv WRISHELLTEST " <> pid <> 
				" > /tmp/.wrishelltest.csh
				source /tmp/.wrishelltest.csh
				rm -f /tmp/.wrishelltest.csh
				echo $WRISHELLTEST", "Text"
			]
		]
	]

bourneShellFamilyQ[] :=
	With[{pid = ToString[$ProcessID]},
		pid === 
		StringTrim[
			Import["!echo WRISHELLTEST=" <> pid <> " > /tmp/.wrishelltest.sh
				. /tmp/.wrishelltest.sh
				rm -f /tmp/.wrishelltest.sh
				echo $WRISHELLTEST", "Text"
			]
		]
	]

ProbeShellFamily[] := 
	If[Head[$ShellFamilyType] =!= String,
		$ShellFamilyType = 
			If[TrueQ@bourneShellFamilyQ[], "BourneShell", "CShell"];
		$ReadVCVars = If[$ShellFamilyType==="CShell", "source", "."];
		$VCVarsExt = If[$ShellFamilyType==="CShell", ".csh", ".sh"];
	]

VCVarsCall[installation_, targetSystemID_] :=
Module[{installationBin, iccvarsPath, vcvarsSource},
        ProbeShellFamily[];
        installationBin = FileNameJoin[{realInstallation[installation],"bin"}];
        iccvarsPath = FileNameJoin[{installationBin, VCVarsArch[targetSystemID],
                "iccvars_" <> VCVarsArch[targetSystemID] <> $VCVarsExt}];
        vcvarsSource = If[FileExistsQ[iccvarsPath],
                QuoteFile[iccvarsPath]
                ,
                {Identity[FileNameJoin[{installationBin,
                        "compilervars" <> $VCVarsExt}]], " ",
                        VCVarsArch[targetSystemID]}
        ];
        StringJoin[$ReadVCVars, " ", vcvarsSource]
]

realInstallation[path_String /; File === Quiet[FileType[path]]] := 
	(* path is like /opt/intel/bin/icc *)
	(* need to trim last parts *)
	FileNameDrop[path, -2]
	
realInstallation[path_String /; Directory === Quiet[FileType[path]]] := 
	(* path is either the default /opt/intel/bin or a user-specified path. *)
	With[{plusicc = FileNameJoin[{path, "icc"}]},
		If[validCompilerCommandQ[plusicc], FileNameDrop[path, -1], path]
	]

realInstallation[path_] := path

VCVarsArch["Linux"] := "ia32"

VCVarsArch["Linux-x86-64"] := "intel64"

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
		(___ ~~ (".a" | (".so" ~~ (("." ~~ NumberString) ...)))) | 
		(* Or files containing a directory separator *)
		(___ ~~ "/" ~~ ___)
	]

IntelCompiler[method_][args___] := 
	CCompilerDriver`CCompilerDriverBase`BaseDriver[method][args]

CCompilerRegister[$ThisDriver]

End[];

EndPackage[];
