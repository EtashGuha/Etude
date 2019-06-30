BeginPackage["CCompilerDriver`", 
	{"ResourceLocator`"}];

$CCompiler::usage = "$CCompiler is the C compiler that will be used unless otherwise specified."

CCompilers::usage = "CCompilers[] returns a list of compilers supported for this version of Mathematica."

$CCompilerDirectory::usage = "$CCompilerDirectory is the location of the C Compiler Driver package."

CreateLibrary::usage = "CreateLibrary[source, name] compiles the C code source and creates a dynamic link library, name.ext.
CreateLibrary[{cfile1, cfile2, ...}, name] compiles the C source files cfile1, cfile2, ..."

CreateExecutable::usage = "CreateExecutable[source, name] compiles the C code source and creates an executable file.
CreateExecutable[{cfile1, cfile2, ...}, name] compiles the C source files cfile1, cfile2, ..."

CreateExecutable::tmclash = "The MathLink template output file `1` already exists.";

DefaultCCompiler::usage = "DefaultCCompiler[] returns the default C compiler for your version of Mathematica, or $Failed if one cannot be found."

$CCompilerDefaultDirectory::usage = "$CCompilerDefaultDirectory is the default directory used by CCompiler driver to store compiled results."

CCompilerDirectories::usage = "CCompilerDirectories[] returns the list of working directories used by CCompilerDriver."

$CCompilerInternalDirectory::usage = "$CCompilerInternalDirectory is a directory used by Mathematica to create DLLs."

CreateObjectFile::usage = "CreateObject[source, name] compiles the C code source into its object file.
CreateObject[{cfile}, name] compiles the C code source file into its object file."

QuoteFile::usage = "QuoteFile[filepath] returns filepath surrounded by quotes."

QuoteFiles::usage = "QuoteFiles[files] returns a list of file paths surrounded by quotes and separated by spaces."

(******************************************************************************)
(* Messages *)

CCompilers::system = "Target system specification \"TargetSystemID\" -> `1` should be "<>
	Switch[$OperatingSystem, 
		"Windows", "\"Windows\" or \"Windows-x86-64\"",
		"Unix", 
			If[$ProcessorType === "ARM",
				"Linux-ARM",
				"\"Linux\" or \"Linux-x86-64\""
			],
		"MacOSX", "\"MacOSX\", \"MacOSX-x86\", or \"MacOSX-x86-64\""
	];

CreateLibrary::nocomp = CreateExecutable::nocomp = CreateObjectFile::nocomp = "A C compiler cannot be found on your system. Please consult the documentation to learn how to set up suitable compilers." ;

CreateLibrary::badcomp = CreateExecutable::badcomp = CreateObjectFile::badcomp = "Compiler specification \"Compiler\" -> `1` does not specify a compiler driver listed by CCompilers[Full]." ;

CreateLibrary::cmperr = CreateExecutable::cmperr = CreateObjectFile::cmperr = "Compile error: `1`";

CreateLibrary::compnmtype = CreateExecutable::compnmtype = CreateObjectFile::compnmtype = "The name directive \"CompilerName\" -> `1` should be a string or Automatic.";

CreateLibrary::instl = CreateExecutable::instl = CreateObjectFile::instl = "The compiler installation directive \"CompilerInstallation\" -> `1` does not indicate a usable installation of `2`.";

CreateLibrary::instltype = CreateExecutable::instltype = CreateObjectFile::instltype = "The compiler installation directive \"CompilerInstallation\" -> `1` should be a string or Automatic.";

CreateLibrary::compopt = CreateExecutable::compopt = CreateObjectFile::compopt = "The compiler option directive \"`1`\" -> `2` should be a string or list of strings.";

CreateLibrary::lnkopt = CreateExecutable::lnkopt = CreateObjectFile::lnkopt = "The linker option directive \"`1`\" -> `2` should be a string or list of strings.";

CreateLibrary::targettype = CreateExecutable::targettype = CreateObjectFile::targettype = "Target specification \"TargetSystemID\" -> `1` should be a string.";

CreateLibrary::target = CreateExecutable::target = CreateObjectFile::target = "Target system specification \"TargetSystemID\" -> `1` is not available for `2` installation `3`."

CreateLibrary::tgtdir = CreateExecutable::tgtdir = CreateObjectFile::tgtdir = "Target directory specification \"TargetDirectory\" -> `1` should be a string."

CreateLibrary::wdtype = CreateExecutable::wdtype = CreateObjectFile::wdtype = "Working directory specification \"WorkingDirectory\" -> `1` should be a string or Automatic."

CreateLibrary::cleantype = CreateExecutable::cleantype = CreateObjectFile::cleantype = "Cleaning specification \"CleanIntermediate\" -> `1` should be one of True, False, Full, or Automatic."

CreateLibrary::def = CreateExecutable::def = CreateObjectFile::def = "Definition specification \"Defines\" -> `1` is not a string, nor is it a list of strings and rules with right-hand sides that are strings or integers."

CreateLibrary::sysdef = CreateExecutable::sysdef = CreateObjectFile::sysdef = "Definition specification \"SystemDefines\" -> `1` is not Automatic, a string, or a list of strings and rules with right-hand sides that are strings or integers."

CreateLibrary::nodir = CreateExecutable::nodir = CreateObjectFile::nodir = "Directory \"`1`\" was not found or is not a directory." 

CreateLibrary::dirlist = CreateExecutable::dirlist = CreateObjectFile::dirlist = "Directory list specification \"`1`\" -> `2` should be a string or list of strings."

CreateLibrary::sysdirlist = CreateExecutable::sysdirlist = CreateObjectFile::sysdirlist = "Directory list specification \"`1`\" -> `2` should be a string, list of strings, or Automatic."

CreateLibrary::filelist = CreateExecutable::filelist = CreateObjectFile::filelist = "File list specification \"`1`\" -> `2` should be a string or list of strings."

CreateLibrary::sysfilelist = CreateExecutable::sysfilelist = CreateObjectFile::sysfilelist = "File list specification \"`1`\" -> `2` should be a string, list of strings, or Automatic."

CreateLibrary::crbin = CreateExecutable::crbin = CreateObjectFile::crbin = "\"CreateBinary\" -> `1` should be True or False."

CreateLibrary::mpreptype = CreateExecutable::mpreptype = CreateObjectFile::mpreptype = "\"MprepOptions\" -> `1` should be a string or list of strings."

CreateLibrary::precompiletype = CreateExecutable::precompiletype = CreateObjectFile::precompiletype = "\"PreCompileCommands\" -> `1` should be a string."

CreateLibrary::postcompiletype = CreateExecutable::postcompiletype = CreateObjectFile::postcompiletype = "\"PostCompileCommands\" -> `1` should be a string."

CreateLibrary::debug = CreateExecutable::debug = CreateObjectFile::debug = "Debug specification \"Debug\" -> `1` should be True or False."

CreateLibrary::translib = CreateExecutable::translib = CreateObjectFile::translib = "TransferProtocolLibrary specification \"translib\" -> `1` is not True or False."

CreateLibrary::lang = CreateExecutable::lang = CreateObjectFile::lang = "Language specification \"Language\" -> `1` is not \"C\" or \"C++\"."

Begin["`Private`"];

(******************************************************************************)
QuoteFile[file_String] :=
	If[TrueQ@needsQuotesQ[file],
		StringJoin["\"", file, "\""],
		file
	]

needsQuotesQ[str_String] := !StringMatchQ[str, "\"" ~~ ___ ~~ "\""]

(******************************************************************************)
QuoteFiles[files_List] := Riffle[QuoteFile /@ files, " "]

(******************************************************************************)
Needs["ResourceLocator`"];
Needs["CCompilerDriver`CCompilerDriverRegistry`"];
Needs["CCompilerDriver`System`"];

$CCompilerDirectory = DirectoryName[ System`Private`$InputFileName]

(* Used by Compile for CompilationTarget settings *)
$CCompilerInternalDirectory = 
	FileNameJoin[{Quiet[ApplicationDataUserDirectory["CCompilerDriver"]], 
		"BuildFolder", $MachineName<>"-"<>ToString[$ProcessID]}]


$CCompilerDefaultDirectory = 
	FileNameJoin[{$UserBaseDirectory, "SystemFiles", "LibraryResources", $SystemID}]

CCompilerDirectories[] := 
	With[{target = $CCompilerDefaultDirectory},
		{target, FileNameJoin[{target, "Working"}]}
	]

Options[ CCompilers] = {"TargetSystemID" :> $SystemID}

CCompilers[ opts:OptionsPattern[]] :=
	Module[{compilers = CCompilers[Full, opts], 
		sysID = OptionValue[ "TargetSystemID"]},
		If[Head[compilers] =!= List || sysID === All,
			Return[compilers]
		];

		Select[
			compilers, 
			With[{driver = "Compiler" /. #, 
				installation = "CompilerInstallation" /. #},
				StringQ[installation] && 
					driver["SupportedTargetSystemIDQ"][sysID, installation]
			]&
		]
	]

CCompilers[Full, OptionsPattern[]] :=
	Module[{sysID = OptionValue[ "TargetSystemID"]},
		If[sysID === All,
			Return[$CCompilers]
		];

		If[!TrueQ[ValidTargetSystemIDQ[sysID]],
			Message[CCompilers::system, sysID];
			Return[$Failed]
		];

		Select[
			$CCompilers,
			With[{driver = "Compiler" /. #, 
				installation = "CompilerInstallation" /. #},
				driver["SupportedTargetSystemIDQ"][sysID, installation]
			]&
		]
	]

$CCompilerDriver = Null;

ValidTargetSystemIDQ[sys_] := MemberQ[$PlatformSystemIDs, sys]

(******************************************************************************)
(* DefaultCCompiler *)

DefaultCCompiler[] := 
(
	(* Make sure a default is selected *)
	If[$CCompilerDriver === Null || $CCompilerDriver === $Failed,
		$CCompilerDriver = SelectCCompiler[$CCompilers]
	];

	If[$CCompilerDriver === $Failed && $CCompiler === Automatic,
		Message[CreateLibrary::nocomp]
	];

	Driver[If[$CCompiler === Automatic, $CCompilerDriver, $CCompiler]]
)

Driver[{___, "Compiler" -> driver_, ___}] := driver

Driver[driver_] := driver

(******************************************************************************)
(* SelectCCompiler *)

SelectCCompiler[compilers_List] :=
	With[{list = Cases[compilers, 
			{___, "CompilerInstallation" -> _String, ___}]},

		If[Length[list] === 0, 
			$Failed,
			First[list]
		]
	]

(******************************************************************************)
$CCompiler = Automatic

CCompilerDriver`$ErrorMessageHead = Automatic

(******************************************************************************)
Options[ CreateLibrary] =  
	Join[
		{"Compiler" :> Automatic}, 
		FilterRules[
			CCompilerDriver`CCompilerDriverBase`Private`CommonCompilerOptions,
			Except[Alternatives@@CCompilerDriver`CCompilerDriverBase`BaseDriver["OptionsExceptions"]["CreateLibrary"]]
		]
	]

Options[ CreateExecutable] = Join[
		{"Compiler" :> Automatic},
		FilterRules[
			CCompilerDriver`CCompilerDriverBase`Private`CommonCompilerOptions,
			Except[Alternatives@@CCompilerDriver`CCompilerDriverBase`BaseDriver["OptionsExceptions"]["CreateExecutable"]]
		]
	]

Options[ CreateObjectFile] = Join[
		{"Compiler" :> Automatic},
		FilterRules[
				CCompilerDriver`CCompilerDriverBase`Private`CommonCompilerOptions,
				Except[Alternatives@@CCompilerDriver`CCompilerDriverBase`BaseDriver["OptionsExceptions"]["CreateObjectFile"]]
			]
	]

(******************************************************************************)
lhs:CreateLibrary[code:(_String|{(_String|_File) ..}|File[_String]), funName_String,
	optseq:OptionsPattern[]] :=
	CallCompiler[CreateLibrary, code, funName, 
		Function[{driver, opts}, driver["LibraryExtension"][Sequence@@opts]], 
		Unevaluated@lhs, {optseq}]

lhs:CreateExecutable[code:(_String|{(_String|_File) ..}|File[_String]), funName_String,
	optseq:OptionsPattern[]] :=
	CallCompiler[CreateExecutable, code, funName, 
		Function[{driver, opts}, driver["ExecutableExtension"][Sequence@@opts]], 
		Unevaluated@lhs, {optseq}]

lhs:CreateObjectFile[ code:(_String| {(_String|_File) ..}|File[_String]), objName_,
	optseq:OptionsPattern[]] :=
	CallCompiler[CreateObjectFile, code, objName, 
		Function[{driver, opts}, driver["ObjectFileExtension"][Sequence@@opts]], 
		Unevaluated@lhs, {optseq}]

CallCompiler[fn_, code_, funName_, extFn_, lhs_, optsArg_List] := 
	Module[{opts = optsArg, driver, opts1},		
		Quiet[driver = OptionValue[fn, opts, "Compiler"], {OptionValue::nodef}];

		If[ driver === Automatic, 
			driver = DefaultCCompiler[];
			If[$CCompiler =!= Automatic && ListQ[$CCompiler],
				opts = Join[optsArg, $CCompiler];
			];
		];

		If[ driver === $Failed, Return[$Failed]];

		If[ !RegisteredCCompilerQ[driver], 
			Message[fn::badcomp, driver];
			Return[$Failed]];
		If[ unknownOptionsCheck[fn, opts, driver, Unevaluated[lhs]] === $Failed,
			Return[$Failed]];
		opts1 = FilterRules[opts, Options[driver]];
		CCompilerDriver`CCompilerDriverBase`InvokeCompiler[driver, 
			fn, extFn[driver, opts1], code, funName, Sequence@@opts1
		]
	]

unknownOptionsCheck[fn_, optsRagged_List, driver_, lhs_] := 
	Module[{opts, fnOpts, driverOpts, leftover},
		(* Options coming from Compile always have a nested list from 
			Compile`$CCompilerOptions *)
		opts = Flatten[optsRagged, 1];
		fnOpts = FilterRules[opts, Options[fn]];
        driverOpts = FilterRules[
            opts,
            FilterRules[
                Options[driver],
                Except[Alternatives@@driver["OptionsExceptions"][ToString[fn]]]
            ]
        ];
		leftover = Complement[opts, Union[fnOpts, driverOpts, {"Name"->""}]];
		leftover = DeleteCases[leftover, "Name" -> _]; (* extra Name is OK *)
		If[MatchQ[leftover, {_Rule ..}],
			Message[fn::optx, leftover[[1,1]], HoldForm[lhs]];
			$Failed,
			Null
		]
	]

End[];

EndPackage[];
