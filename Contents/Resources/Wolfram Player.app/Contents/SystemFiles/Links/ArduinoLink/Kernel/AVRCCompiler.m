(* ::Package:: *)

(* Wolfram Language Package *)
(*==========================================================================================================
			
					AVRCCompiler
			
Author: Ian Johnson
			
Copyright (c) 2015 Wolfram Research. All rights reserved.			


AVRCCompiler is a package to direct CCompilerDriver how to use the AVR GCC toolchain to compile C code for AVR
microchips, like the one in the Arduino Uno.

CURRENT SUPPORTED BOARDS:
~Arduino Uno

USER ACCESSIBLE FUNCTIONS:
AVRCCompiler

==========================================================================================================*)

BeginPackage["AVRCCompiler`"]
(* Exported symbols added here with SymbolName::usage *)  

AVRCCompiler::usage = "AVRCCompiler is a symbol that represents a C/C++ compiler that is implemented in the Arduino software with avr-g++ and avr-gcc."


Begin["`Private`"] (* Begin Private Context *) 


Needs[ "CCompilerDriver`"]
Needs["CCompilerDriver`CCompilerDriverBase`"];
Needs["CCompilerDriver`CCompilerDriverRegistry`"];


`$ThisDriver = AVRCCompiler;


CCompilerRegister[ AVRCCompiler, {
	"Windows", "Windows-x86-64",
	"Linux", "Linux-x86-64", "Linux-ARM",
	"MacOSX-x86-64"}
]


(*note, changed "CreateLibraryFlag" from "-shared" to blank, not certain of its purpose at this time*)
Options[ $ThisDriver] = DeriveOptions[{
	"SystemLibraries"->{},
	"CreateLibraryFlag"->""
}]


(*Not sure why this is false, but it is set as false in the GenericCCompiler*)
AVRCCompiler["Available"] := False;


AVRCCompiler["Name"][] := "AVR C/C++ Compiler"


(*Not sure why this is None, perhaps here we could check for an Arduino installation?*)
AVRCCompiler["Installation"][] := None


(*Again, not sure why this is blank*)
AVRCCompiler["Installations"][] := {}



(*I really don't understand how these resolve anything...*)
AVRCCompiler["ResolveInstallation"][Automatic] := None


AVRCCompiler["ResolveInstallation"][path_] := path


AVRCCompiler["ResolveCompilerName"][Automatic] := None


AVRCCompiler["ResolveCompilerName"][name_] := name


AVRCCompiler["CreateObjectFileCommands"][
	errHd_, installation_, compilerName_, outFile_, workDir_, 
	compileOptions_, linkerOptions_, defines_, includePath_, srcFileRules_List, 
	tmSrcFiles_, cFiles_, syslibs_, libs_, libpath_, extraObjects_, 
	targetSystemID_, cleanIntermediate_, mprepOptions_, translib_, language_, opts_] :=
	Module[{},

		CommandJoin[
			compilerCommand[QuoteFile@installation, compilerName],
			" -c", 
			(*this resets the output file extension to .o*)
			" -o ", QuoteFile[WorkingOutputFile[workDir, FileBaseName[outFile]<>".o"]],
			(*I think with compileOptions is where all the verbose output and such will be set...*)
			" ", compileOptions,
			(*there shouldn't have to be any defines with this*)
			" ", defines,
			(*these will probably either have to default to the arduino libraries, or whatever calls this
			will need to default to adding those here*)
			" ", includePath,
			(*might run into problems with quoting files*)
			" ", QuoteFiles[cFiles], 
			" 2>&1\n"
		]
	]


AVRCCompiler["CreateExecutableCommands"][
	errHd_, installation_, compilerName_, outFile_, workDir_, 
	compileOptions_, linkerOptions_, defines_, includePath_, srcFileRules_List, 
	tmSrcFiles_, cFiles_, syslibs_, libs_, libpath_, extraObjects_, 
	targetSystemID_, cleanIntermediate_, mprepOptions_, translib_, language_, opts_] :=
	Module[{},
		
		CommandJoin[
			

			compilerCommand[QuoteFile@installation, compilerName],
			(*this resets the extension of the output file to be .elf*)
			" -o ", QuoteFile[WorkingOutputFile[workDir, If[FileExtension[outFile]==="cpp",outFile<>".elf",FileBaseName[outFile]<>".elf"]]], 
			" ", compileOptions,
			" ", defines,
			" ", includePath,
			" ", QuoteFiles[cFiles], 
			" ", QuoteFiles[extraObjects], 
			" ", libpath,
			" ", formatLibraries[syslibs, libs], 
			" 2>&1\n"
		]
	]



	



(*this function renames the files to have a .cpp file extension instead of a .c extension*)
(*basically take the extension, if it is .cpp or .c leave it, else change it*)
cPlusPlusRename[files_]:=Module[{},
	(FileNameJoin[Flatten@{Most@FileNameSplit@#,FileBaseName[Last@FileNameSplit@#]<>
		If[FileExtension[Last@FileNameSplit@#]==="cpp"||FileExtension[Last@FileNameSplit@#]==="c",
			(*THEN*)
			"",
			(*ELSE*)
			".cpp"]}])&/@files
]


echo=(Print[#];#)&


(*These commands are copied verbatim from the GenericCCompiler*)

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
		"-l"<>QuoteFile[lib]
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


AVRCCompiler[method_][args___] := 
	CCompilerDriver`CCompilerDriverBase`BaseDriver[method][args]


CCompilerRegister[$ThisDriver]


End[] (* End Private Context *)

EndPackage[]