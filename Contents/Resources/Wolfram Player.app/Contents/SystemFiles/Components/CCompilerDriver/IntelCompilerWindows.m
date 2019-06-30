BeginPackage["CCompilerDriver`IntelCompilerWindows`", 
	{"CCompilerDriver`", "CCompilerDriver`IntelCompiler`"}];

Begin["`Private`"];

Needs["CCompilerDriver`CCompilerDriverBase`"]
Needs["CCompilerDriver`IntelCompiler`"]
Needs["CCompilerDriver`CCompilerDriverRegistry`"]

(****************************************************************
 * CCompilerDriver`IntelCompiler`IntelCompiler
 * Print["Context[IntelCompiler]: ", Context[IntelCompiler]];
*****************************************************************)

`$ThisDriver = IntelCompiler

Options[ $ThisDriver] = 
	DeriveOptions[
		{
			"SystemCompileOptions" -> {"/O2", "/MT"}
		}
	];

IntelCompiler["Available"] :=
	With[{automaticInstallation = 
		$ThisDriver["ResolveInstallation"][Automatic]},

		TrueQ[$ThisDriver["ValidInstallationQ"][automaticInstallation]]
	]

IntelCompiler["Name"][] := "Intel Compiler"

IntelCompiler["Installation"][] := 
	$ThisDriver["ResolveInstallation"][Automatic]

IntelCompiler["LibraryPathFlag"][] := "/LIBPATH:"

IntelCompiler["DefineFlag"][] := "/D "

IntelCompiler["IncludePathFlag"][] := "/I "

IntelCompiler["DebugSystemOptions"][] := {"/MT", "/Zi"}

$CompilerExe = "icl.exe"

IntelCompiler["CreateObjectFileCommands"][
	errHd_, installation_, compilerName_, outFile_, workDir_, 
	compileOptions_, linkerOptions_, defines_, includePath_, srcFileRules_List, 
	tmSrcFiles_, cFiles_, syslibs_, libs_, libpath_, extraObjects_, 
	targetSystemID_, cleanIntermediate_, mprepOptions_, translib_, language_, opts_] :=
	{
		CommandJoin[SetupEnvironment[installation, targetSystemID], "\n"],
		CommandJoin[
			$CompilerExe,
			" ", "/c", 
			" ", compileOptions,
			" ", defines,
			" ", includePath,
			" ", QuoteFiles[cFiles],
			" /Fo", QuoteFile[WorkingOutputFile[workDir, outFile]], "\n"
		]
	}

IntelCompiler["CreateLibraryCommands"][
	errHd_, installation_, compilerName_, outFile_, workDir_, 
	compileOptions_, linkerOptions_, defines_, includePath_, srcFileRules_List, 
	tmSrcFiles_, cFiles_, syslibs_, libs_, libpath_, extraObjects_, 
	targetSystemID_, cleanIntermediate_, mprepOptions_, translib_, language_, opts_] :=
	{
		CommandJoin[SetupEnvironment[installation, targetSystemID], "\n"],
		CommandJoin[
			$CompilerExe, 
			" ", compileOptions,
			" ", defines,
			" ", includePath,
			" ", QuoteFiles[cFiles], 
			" ", formatLibraries[syslibs, libs], 
			" ", QuoteFiles[extraObjects],
			" /link",
			" /link /dll",
			" ", Riffle[linkerOptions, " "],
			" ", libpath,
			" /out:", QuoteFile[WorkingOutputFile[workDir, outFile]], "\n"
		]
	}

IntelCompiler["CreateExecutableCommands"][
	errHd_, installation_, compilerName_, outFile_, workDir_, 
	compileOptions_, linkerOptions_, defines_, includePath_, srcFileRules_List, 
	tmSrcFiles_, cFiles_, syslibs_, libs_, libpath_, extraObjects_, 
	targetSystemID_, cleanIntermediate_, mprepOptions_, translib_, language_, opts_] :=
	{
		CommandJoin[SetupEnvironment[installation, targetSystemID], "\n"],
		CommandJoin[
			MprepCalls[tmSrcFiles, workDir, translib, mprepOptions],

			$CompilerExe,
			" ", compileOptions,
			" ", defines,
			" ", includePath,
			" ", QuoteFiles[cFiles],
			" ", formatLibraries[syslibs, libs],
			" ", QuoteFiles[extraObjects],
			" /link",
			" ", Riffle[linkerOptions, " "],
			" user32.lib kernel32.lib gdi32.lib ",
			" ", libpath,
			" /out:", QuoteFile[WorkingOutputFile[workDir, outFile]], "\n"
		]
	}

SetupEnvironment[installation_, targetSystemID_] := 
	VCVarsCall[installation, targetSystemID]

IntelCompiler["ResolveInstallation"][Automatic] := 
	SelectFirst[$ThisDriver["Installations"][],
		Function[path, $ThisDriver["ValidInstallationQ"][path]]
	]

IntelCompiler["ResolveInstallation"][path_String] := path

IntelCompiler["Installations"][] :=  
	{Environment["ICPP_COMPILER16"], Environment["ICPP_COMPILER15"], Environment["ICPP_COMPILER13"],
		Environment["ICPP_COMPILER12"]}

IntelCompiler["ValidInstallationQ"][installation_] := 
	StringQ[installation] && 
	(Quiet[FileType[installation]] === Directory) &&
	(Quiet[FileType[VCVarsPath[installation]]] === File)

VCVarsCall[installation_, targetSystemID_] := 
	StringJoin["call ", QuoteFile[VCVarsPath[installation]], " ", 
		VCVarsArch[targetSystemID]]

VCVarsPath[installation_] := 
	FileNameJoin[{installation, "bin", "iclvars.bat"}]

VCVarsArch["Windows"] := "ia32"

VCVarsArch["Windows-x86-64"] := "intel64"

formatLibraries[libs_List] := Riffle[formatLibrary /@ libs, " "]

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
		(* Files ending in .a or .so followed by 0 or more .N extensions *)
		(___ ~~ ".lib") | 
		(* Or files containing a directorty separator *)
		(___ ~~ ("/" | "\\") ~~ ___)
	]

IntelCompiler["ExtractErrors"][buildOutput_] := 
	Module[{lines, errors},
		lines = StringSplit[buildOutput, {"\n", "\r\n", "\r"}];
		errors = Select[lines, errorQ];
		errors
	]

errorQ[line_String] := 
	StringMatchQ[line, ___ ~~ ":" ~~ ___ ~~ "error " ~~ ___ ~~ ":" ~~ ___]

errorQ[_] := False

IntelCompiler[method_][args___] := 
	CCompilerDriver`CCompilerDriverBase`BaseDriver[method][args]

CCompilerRegister[$ThisDriver]

End[];

EndPackage[];
