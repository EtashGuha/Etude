BeginPackage["CCompilerDriver`VisualStudioCompiler`", {"CCompilerDriver`"}];

VisualStudioCompiler::usage = "VisualStudioCompiler[src, name] compiles the code in src into a DLL and returns the full path of the DLL.";

VisualStudioCompiler::sdk7vcvars = "The Windows SDK for Windows 7, version 7.0, should have installed either vcvarsamd64.bat or vcvarsx86_amd64.bat in the VC/bin directory of `1`.";

Begin["`Private`"];

Needs["CCompilerDriver`"] (* to pick up QuoteFile and QuoteFiles *)
Needs["CCompilerDriver`CCompilerDriverBase`"]
Needs["CCompilerDriver`CCompilerDriverRegistry`"]

`$ThisDriver = VisualStudioCompiler

CCompilerRegister[ VisualStudioCompiler, {"Windows", "Windows-x86-64"}]

Options[ $ThisDriver] = 
	DeriveOptions[
		{
			"SystemCompileOptions" -> {"/O2", "/MT"}
		}
	];

VisualStudioCompiler["Available"] :=
	With[{automaticInstallation = 
		$ThisDriver["ResolveInstallation"][Automatic]},
		TrueQ[$ThisDriver["ValidInstallationQ"][automaticInstallation]]
	]

VisualStudioCompiler["Name"][] := "Visual Studio"

VisualStudioCompiler["Installation"][] := 
	$ThisDriver["ResolveInstallation"][Automatic]
(******************************************************************************)
isLikeVS2017LayoutQ[installation_String] :=
        FileExistsQ[FileNameJoin[{installation, "VC", "Auxiliary", "Build", "vcvarsall.bat"}]]

isLikeVS2017LayoutQ[___] := False
(******************************************************************************)

VisualStudioCompiler["SupportedTargetSystemIDQ"]["Windows", ___] := True 

VisualStudioCompiler["SupportedTargetSystemIDQ"]["Windows-x86-64",
        installation_:Automatic] :=
		With[{installationPath = $ThisDriver["ResolveInstallation"][installation]},
			If[StringQ[installationPath],
				With[{vcbin = FileNameJoin[{installationPath, "VC", "bin"}], vcTools = FileNameJoin[{installationPath, "VC", "Tools", "MSVC"}]},
					(* Check layout for version 2015 and earlier *)
					DirectoryQ[FileNameJoin[{vcbin, "x86_amd64"}]] || DirectoryQ[FileNameJoin[{vcbin, "amd64"}]] ||
					(* Check layout for version 2017 *)
					If[DirectoryQ[vcTools],
						With[{wildCardDirs = FileNames["*", vcTools]},
						    AnyTrue[ wildCardDirs, 
						    	(StringQ[#] && DirectoryQ[FileNameJoin[{#, "bin", "HostX86", "x64"}]])&]
						],
						False
					]
				],
				False
			]
		]

VisualStudioCompiler["LibraryPathFlag"][] := "/LIBPATH:"

VisualStudioCompiler["DefineFlag"][] := "/D "

VisualStudioCompiler["IncludePathFlag"][] := "/I "

VisualStudioCompiler["DebugSystemOptions"][] := {"/MTd", "/Zi"}

VisualStudioCompiler["LanguageOption"]["C++"] := " /TP"

VisualStudioCompiler["LanguageOption"]["C"] := " /TC"

VisualStudioCompiler["LanguageOption"][_] := ""

VisualStudioCompiler["CreateObjectFileCommands"][
	errHd_, installation_, compilerName_, outFile_, workDir_, 
	compileOptions_, linkerOptions_, defines_, includePath_, srcFileRules_List, 
	tmSrcFiles_, cFiles_, syslibs_, libs_, libpath_, extraObjects_, 
	targetSystemID_, cleanIntermediate_, mprepOptions_, translib_, language_, opts_] :=
	{
		CommandJoin[SetupEnvironment[installation, targetSystemID], "\n"],
		CommandJoin[
			"cl.exe",
			" ", "/c", 
			$ThisDriver["LanguageOption"][language],
			" ", compileOptions,
			" ", defines,
			" ", includePath,
			" ", QuoteFiles[cFiles],
			" /Fo", QuoteFile[WorkingOutputFile[workDir, outFile]], "\n"
		]
	}

VisualStudioCompiler["CreateLibraryCommands"][ 
	errHd_, installation_, compilerName_, outFile_, workDir_, 
	compileOptions_, linkerOptions_, defines_, includePath_, srcFileRules_List, 
	tmSrcFiles_, cFiles_, syslibs_, libs_, libpath_, extraObjects_, 
	targetSystemID_, cleanIntermediate_, mprepOptions_, translib_, language_, opts_] :=
	Module[{libFile = FileNameJoin[{
		FileNameDrop[WorkingOutputFile[workDir, outFile], -1], 
		FileBaseName[outFile] <> ".lib"}]},

		{
			CommandJoin[SetupEnvironment[installation, targetSystemID], "\n"],
			CommandJoin[
				"cl.exe", 
				" /LD",
				$ThisDriver["LanguageOption"][language],
				" ", compileOptions,
				" ", defines,
				" ", includePath,
				" ", QuoteFiles[cFiles], 
				" /link ",
				" /implib:", QuoteFile[libFile],
				" ", Riffle[linkerOptions, " "],
				" ", libpath,
				" ", formatLibraries[syslibs, libs], 
				" ", QuoteFiles[extraObjects],
				" /out:", QuoteFile[WorkingOutputFile[workDir, outFile]], "\n"
			]
		}
	]

VisualStudioCompiler["CreateStaticLibraryCommands"][ 
	errHd_, installation_, compilerName_, outFile_, workDir_, 
	compileOptions_, linkerOptions_, defines_, includePath_, srcFileRules_List, 
	tmSrcFiles_, cFiles_, syslibs_, libs_, libpath_, extraObjects_, 
	targetSystemID_, cleanIntermediate_, mprepOptions_, translib_, language_, opts_] :=
	Module[{objFiles},
	
		objFiles = Map[FileBaseName[#]<>".obj"&, cFiles];

		{
			CommandJoin[SetupEnvironment[installation, targetSystemID], "\n"],
			CommandJoin[
				(* create object files *)
				"cl.exe", 
				" ", "/c", 
				$ThisDriver["LanguageOption"][language],
				" ", compileOptions,
				" ", defines,
				" ", includePath,
				" ", QuoteFiles[cFiles],
				"\n",

				"lib.exe ",
				" ", Riffle[linkerOptions, " "],
				" ", libpath,
				" ", formatLibraries[syslibs, libs], 
				" ", QuoteFiles[Join[objFiles, extraObjects]],
				" /out:", QuoteFile[WorkingOutputFile[workDir, outFile]], "\n"
			]
		}
	]

VisualStudioCompiler["CreateExecutableCommands"][
	errHd_, installation_, compilerName_, outFile_, workDir_, 
	compileOptions_, linkerOptions_, defines_, includePath_, srcFileRules_List, 
	tmSrcFiles_, cFiles_, syslibs_, libs_, libpath_, extraObjects_, 
	targetSystemID_, cleanIntermediate_, mprepOptions_, translib_, language_, opts_] :=
	{
		CommandJoin[SetupEnvironment[installation, targetSystemID], "\n"],
		CommandJoin[
			MprepCalls[tmSrcFiles, workDir, translib, mprepOptions],

			"cl.exe",
			$ThisDriver["LanguageOption"][language],
			" ", compileOptions,
			" ", defines,
			" ", includePath,
			" ", QuoteFiles[cFiles],
			" /link",
			" ", Riffle[linkerOptions, " "],
			" user32.lib kernel32.lib gdi32.lib ",
			" ", libpath,
			" ", formatLibraries[syslibs, libs],
			" ", QuoteFiles[extraObjects],
			" /out:", QuoteFile[WorkingOutputFile[workDir, outFile]], "\n"
		]
	}

SetupEnvironment[installation_, targetSystemID_] := 
	If[InstallationSupportsVCVarsAll[installation, targetSystemID],
		VCVarsCall[installation, targetSystemID]
		(* Else *),
		WindowsSDKSetEnvironment[installation, targetSystemID]
	]

(* 64-bit targets are supported by the commercial version, which has the 
	target-specific batch file located in the corresponding subdirectory of bin,
	e.g. VC/bin/amd64/vcvarsamd64.bat.
 *)
InstallationSupportsVCVarsAll[installation_, "Windows-x86-64"] :=
		InstallationSupportsVCVarsForCompilerQ[installation] ||
        	InstallationSupportsVCVarsForCompilerQ[installation, "amd64"] ||
             	InstallationSupportsVCVarsForCompilerQ[installation, "x86_amd64"]

(* 32-bit targets are always supported *)
InstallationSupportsVCVarsAll[___] := True

(******************************************************************************)
InstallationSupportsVCVarsForCompilerQ[installation_String, "amd64"] := 
	With[{amd64dir = FileNameJoin[{installation, "VC", "bin", "amd64"}]},
		Quiet[
			FileExistsQ[amd64dir] &&
			(FileExistsQ[FileNameJoin[{amd64dir, "vcvars64.bat"}]] ||
				FileExistsQ[FileNameJoin[{amd64dir, "vcvarsamd64.bat"}]])
		]
	]

InstallationSupportsVCVarsForCompilerQ[installation_String, 
	compiler:("x86" | "amd64" |  "x86_amd64")] := 
	Quiet[
		FileExistsQ[FileNameJoin[{installation, "VC", "bin", compiler, 
			"vcvars"<>compiler<>".bat"}]]
	]

InstallationSupportsVCVarsForCompilerQ[installation_String] :=
        Quiet[
                FileExistsQ[FileNameJoin[{installation, "VC", "Auxiliary", "Build", "vcvarsall.bat"}]]
        ]
(******************************************************************************)
SupportedWindowsSDKVersions = ("v7.0" | "v7.1")

(******************************************************************************)
WindowsSDKSetEnvironment[installation_, targetSystemID_] := 
	With[{sdkVer = WindowsSDKVersion[installation, targetSystemID]},
		If[MatchQ[sdkVer, SupportedWindowsSDKVersions],
			WindowsSDKSetEnvironment[installation, targetSystemID, sdkVer]
			,
			VCVarsCall[installation, targetSystemID]
		]
	]

WindowsSDKSetEnvironment[installation_, targetSystemID_, sdkVer:"v7.1"] := 
	Module[{winsdk = WindowsSDKLocation[sdkVer]},
		"call " <> QuoteFile[FileNameJoin[{winsdk, "Bin", "SetEnv.cmd"}]] <> 
			" /x64"
	]

WindowsSDKSetEnvironment[installation_, targetSystemID_, "v7.0"] :=
	StringJoin["call ", QuoteFile[VCVarsScriptForWindowsSDKv70[installation]]]

VCVarsScriptForWindowsSDKv70[installation_] := 
	Module[{candidates, results},
		candidates = Map[
			FileNameJoin[{installation, "VC", "bin", "vcvars"<>#<>".bat"}]&,
			{"64","x86_amd64"}];
		results = Select[candidates, FileExistsQ, 1];
		If[Length[results] === 0,
			Message[VisualStudioCompiler::vcvars, installation];
			Throw[$Failed],
			First[results]
		]
	]
	
(******************************************************************************)
WindowsSDKVersion[installation_, "Windows-x86-64"] := 
	Which[
		WindowsSDKFoundQ["v7.1"], "v7.1",
		WindowsSDKFoundQ["v7.0"], "v7.0",
		True, None
	]

WindowsSDKVersion[___] := None

(******************************************************************************)
WindowsSDKFoundQ[version : SupportedWindowsSDKVersions] := 
	With[{loc = WindowsSDKLocation[version]},
		Head[loc] === String && DirectoryQ[loc]
	]

(******************************************************************************)
WindowsSDKLocation[version : SupportedWindowsSDKVersions] := 
	Module[{
		key = "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Microsoft SDKs\\Windows",
		regkey
		},
		regkey = Quiet[Check[Developer`EnumerateRegistrySubkeys[key], $Failed]];
		If[regkey =!= $Failed && TrueQ@MemberQ[regkey, version],
			"InstallationFolder" /. 
				Developer`ReadRegistryKeyValues[key <> "\\"<>version]
			,
			$Failed
		]
	]

(******************************************************************************)
VisualStudioCompiler["ResolveInstallation"][Automatic] :=
                If[$vswhereWorks,
             		VisualStudioCompiler["ResolveInstallationUsingVswhere"][$vsPathTable],
                        VisualStudioCompiler["ResolveInstallationUsingEnvironment"][$VisualStudioVersions]
                ]

VisualStudioCompiler["ResolveInstallationUsingVswhere"][paths_List] :=
        Select[paths, $ThisDriver["ValidInstallationQ"][#]&, 1] /. {
                {path_, ___} :> path,
                _ :> $Failed
        }

VisualStudioCompiler["ResolveInstallationUsingEnvironment"][versions_List] :=
        Select[versions, $ThisDriver["ValidInstallationQ"][installPath[#]]&, 1] /. {
                {version_, ___} :> installPath[version],
                _ :> $Failed
        }

VisualStudioCompiler["ResolveInstallation"][path_String] := path

$VisualStudioVersions = {"2015", "2013", "2012", "2010", "2008", "2005"}

installPath["2005"] := installPathFromEnvironment["VS80COMNTOOLS"]

installPath["2008"] := installPathFromEnvironment["VS90COMNTOOLS"]

installPath["2010"] := installPathFromEnvironment["VS100COMNTOOLS"]

installPath["2012"] := installPathFromEnvironment["VS110COMNTOOLS"]

installPath["2013"] := installPathFromEnvironment["VS120COMNTOOLS"]

installPath["2015"] := installPathFromEnvironment["VS140COMNTOOLS"]

installPathFromEnvironment[env_] :=
        With[{loc = Environment[env]},
                If[StringQ[loc] && loc =!= "" && FileExistsQ[loc],
                        FileNameDrop[loc, -2],
                        $Failed
                ]
        ]

VisualStudioCompiler["Installations"][] :=
        If[$vswhereWorks, $vsPathTable, Map[installPath, $VisualStudioVersions]]

VisualStudioCompiler["ValidInstallationQ"][installation_] := (
        StringQ[installation] && installation =!= "" && FileExistsQ[installation] &&
        (Quiet[FileType[installation]] === Directory) &&
        ((Quiet[FileType[VCVarsPath[installation]]] === File) ||
          (Quiet[FileType[VCVarsPath2017[installation]]] === File)))

VCVarsCall[installation_, targetSystemID_] :=
        StringJoin["call \"", If[isLikeVS2017LayoutQ[installation], VCVarsPath2017[installation], VCVarsPath[installation]],
            "\" ", VCVarsArch[targetSystemID, installation]]

VCVarsPath[installation_] := 
	FileNameJoin[{installation, "VC", "vcvarsall.bat"}]

VCVarsPath2017[installation_] :=
            FileNameJoin[{installation, "VC", "Auxiliary", "Build", "vcvarsall.bat"}]

VCVarsArch["Windows"] := "x86"

VCVarsArch["Windows-x86-64"] := $VS64BitCompilerTag

VCVarsArch["Windows", _] := VCVarsArch["Windows"]

VCVarsArch["Windows-x86-64", installation_] := 
	Which[
		InstallationSupportsVCVarsForCompilerQ[installation, "amd64"], "amd64",
		InstallationSupportsVCVarsForCompilerQ[installation, "x86_amd64"],
			"x86_amd64",
		True, $VS64BitCompilerTag
	]

$VS64BitCompilerTag = If[$SystemID === "Windows", "x86_amd64", "amd64"]

(******************************************************************************)
$vswherePath = FileNameJoin[{DirectoryName[$InputFileName], "Resources", "Windows", "vswhere.exe"}]

$vsInfo := $vsInfo =
	Module[{result, vswhereCmd},
		vswhereCmd = "!" <> QuoteFile[$vswherePath] <> " -products * -legacy -utf8 -format json";
		(* We need to work around a bug where running a child process when Directory[] has a UNC path results in no output. *)
		SetDirectory[$vswhereDirectory];
		result = Quiet[Check[Import[vswhereCmd, "RawJSON"], $Failed]];
		ResetDirectory[];
		result
	]

$vswhereDirectory = $TemporaryDirectory

$vswhereWorks := $vswhereWorks = MatchQ[$vsInfo, {_?AssociationQ..}]

$vsPathTable := $vsPathTable = If[$vswhereWorks, Table[$vsInfo[[i]]["installationPath"], {i,Length[$vsInfo]}],
    False]

$vsVersionTable := $vsVersionTable = If[$vswhereWorks, StringTake[Table[$vsInfo[[i]]["installationVersion"],
    {i,Length[$vsInfo]}], 2], False]
(******************************************************************************)

formatLibraries[libs_List] := 
	Riffle[formatLibrary /@ libs, " "]

formatLibraries[libs_List, libs2_List] := formatLibraries[Join[libs, libs2]]

formatLibrary[lib_] := 
	If[LibraryPathQ[lib], 
		(* lib appears to be an explicit library file path, just quote it *)
		QuoteFile[lib], 
		(* lib appears to be a simple lib name, pass it to -l *)
		QuoteFile[lib<>".lib"]
	]

LibraryPathQ[lib_] := 
	StringMatchQ[lib,
		(* Files ending in .lib *)
		(___ ~~ ".lib") | 
		(* Or files containing a directorty separator *)
		(___ ~~ ("/" | "\\") ~~ ___)
	]

VisualStudioCompiler["ExtractErrors"][buildOutput_] := 
	Module[{lines, errors},
		lines = StringSplit[buildOutput, {"\n", "\r\n", "\r"}];
		errors = Select[lines, errorQ];
		errors
	]

errorQ[line_String] := 
	StringMatchQ[line, ___ ~~ ":" ~~ ___ ~~ "error " ~~ ___ ~~ ":" ~~ ___]

errorQ[_] := False

VisualStudioCompiler[method_][args___] := 
	CCompilerDriver`CCompilerDriverBase`BaseDriver[method][args]

CCompilerRegister[$ThisDriver]

End[];

EndPackage[];
