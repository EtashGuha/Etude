(* Mathematica Package *)

BeginPackage["CCompilerDriver`GenericCCompiler`", { "CCompilerDriver`"}]

GenericCCompiler::usage = "GenericCCompiler is a symbol that represents a C compiler conforming to typical C compiler usage."

Begin["`Private`"] 

Needs["CCompilerDriver`CCompilerDriverBase`"]
Needs["CCompilerDriver`CCompilerDriverRegistry`"]

`$ThisDriver = GenericCCompiler

CCompilerRegister[ GenericCCompiler, {
	"Windows", "Windows-x86-64",
	"Linux", "Linux-x86-64", "Linux-ARM",
	"MacOSX-x86-64"}
]

Options[ $ThisDriver] = DeriveOptions[{
	"SystemLibraries" -> {},
	"CreateLibraryFlag" -> "-shared"
	}]

GenericCCompiler["Available"] := False

GenericCCompiler["Name"][] := "Generic C Compiler"

GenericCCompiler["Installation"][] := None

GenericCCompiler["Installations"][] := {}

GenericCCompiler["ResolveInstallation"][Automatic] := None

GenericCCompiler["ResolveInstallation"][path_] := path

GenericCCompiler["ResolveCompilerName"][Automatic] := None

GenericCCompiler["ResolveCompilerName"][name_] := name

GenericCCompiler["CreateObjectFileCommands"][
	errHd_, installation_, compilerName_, outFile_, workDir_, 
	compileOptions_, linkerOptions_, defines_, includePath_, srcFileRules_List, 
	tmSrcFiles_, cFiles_, syslibs_, libs_, libpath_, extraObjects_, 
	targetSystemID_, cleanIntermediate_, mprepOptions_, translib_, language_, opts_] :=
	Module[{},

		CommandJoin[
			compilerCommand[installation, compilerName],
			" -c", 
			" -o ", QuoteFile[WorkingOutputFile[workDir, outFile]],
			" ", compileOptions,
			" ", defines,
			" ", includePath,
			" ", QuoteFiles[cFiles], 
			" 2>&1\n"
		]
	]

GenericCCompiler["CreateLibraryCommands"][
	errHd_, installation_, compilerName_, outFile_, workDir_, 
	compileOptions_, linkerOptions_, defines_, includePath_, srcFileRules_List, 
	tmSrcFiles_, cFiles_, syslibs_, libs_, libpath_, extraObjects_, 
	targetSystemID_, cleanIntermediate_, mprepOptions_, translib_, language_, opts_] :=
	Module[{createLibraryFlag = 
		OptionValue[$ThisDriver, opts, "CreateLibraryFlag"]},

		CommandJoin[
			compilerCommand[installation, compilerName],
			" ", createLibraryFlag,
			" ", "-o ", QuoteFile[WorkingOutputFile[workDir, outFile]],
			" ", compileOptions,
			" ", defines,
			" ", includePath,
			" ", QuoteFiles[cFiles], 
			" ", QuoteFiles[extraObjects],
			Map[" " <> # &, linkerOptions],
			" ", libpath,
			" ", formatLibraries[syslibs, libs], 
			" 2>&1\n"
		]
	]

GenericCCompiler["CreateExecutableCommands"][
	errHd_, installation_, compilerName_, outFile_, workDir_, 
	compileOptions_, linkerOptions_, defines_, includePath_, srcFileRules_List, 
	tmSrcFiles_, cFiles_, syslibs_, libs_, libpath_, extraObjects_, 
	targetSystemID_, cleanIntermediate_, mprepOptions_, translib_, language_, opts_] :=
	Module[{},

		CommandJoin[
			MprepCalls[tmSrcFiles, workDir, translib, mprepOptions],

			compilerCommand[installation, compilerName],
			" -o ", QuoteFile[WorkingOutputFile[workDir, outFile]], 
			" ", compileOptions,
			" ", defines,
			" ", includePath,
			" ", QuoteFiles[cFiles], 
			" ", QuoteFiles[extraObjects], 
			Map[" " <> # &, linkerOptions],
			" ", libpath,
			" ", formatLibraries[syslibs, libs], 
			" 2>&1\n"
		]
	]

compilerCommand[installation_String, name_String] :=
	Select[locations[installation, name], validLocationQ, 1] /. {
		{path_} :> path,
		_ :> FileNameJoin[{installation, name}] (*possibly invalid, try anyway*)
	}

compilerCommand[installation_String, None] := installation

validLocationQ[path_] := 
	StringQ[path] && FileExistsQ[path] && File === FileType[path]

(* locations allows any of the following specifications:
   CompilerInstallation is the complete path to the compiler binary
   CompilerInstallation is the directory holding the compiler binary, and
     CompilerName is the compiler binary's filename
   CompilerInstallation/bin is the directory holding the compiler binary, and
     CompilerName is the compiler binary's filename
*) 
locations[installation_, name_] := 
	{
		installation,
		FileNameJoin[{installation, name}], 
		FileNameJoin[{installation, "bin", name}]
	}

formatLibraries[libs_List] := 
	Riffle[formatLibrary /@ libs, " "]

formatLibraries[libs_List, libs2_List] := formatLibraries[Join[libs, libs2]]

formatLibrary[lib_] := 
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
		(* Files ending in .lib *)
		(___ ~~ ".lib") | 
		(* Or files containing a directory separator *)
		(___ ~~ ("/" | "\\") ~~ ___)
	]

GenericCCompiler[method_][args___] := 
	CCompilerDriver`CCompilerDriverBase`BaseDriver[method][args]

CCompilerRegister[$ThisDriver]

End[]

EndPackage[]
