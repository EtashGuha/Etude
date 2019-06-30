BeginPackage["CCompilerDriver`CCompilerDriverBase`"]

BaseDriver::usage = "BaseDriver[method][args] is called by compiler drivers to defer to base class functionality."

$Bits::usage = "$Bits is the number of bits used by the CPU architecture"

Bits::usage = "Bits[systemid] returns the number of bits used by the CPU architecture where $SystemID is systemid."

ChangeDirectoryBlock::usage = "ChangeDirectoryBlock"

CommandJoin::usage = "CommandJoin[commands] performs a StringJoin on a command list, with enhanced error reporting."

CompileOptions::usage = "CompileOptions[driver, opts] returns the compiler options passed in opts"

ConvertSourceToObject::usage = "ConvertSourceToObject[ file, ext] convert a source file to an object name, using the extension ext, dropping the file root."

DeriveOptions::usage = "DeriveOptions[opts, exceptions] returns an options list formed from a common list of default options and opts, and exluding exceptions."

$ILP64::usage = "$ILP64 returns True if the hardware architecture uses ILP64."

GetOption::usage = "GetOption[driver, errHd, options, name] returns the option value of name according to the options for driver, using errHd to generate messages for typical values of CreateObjectFile, CreateLibrary, and CreateExecutable."

IncludePath::usage = "IncludePath[driver, errHd, options] returns a list of include path directives based on the contents of the options list."

InvokeCompiler::usage = "InvokeCompiler[driver, topLevelFunction, commandFunction, ext,	code, funName, options] handles the construction and execution of the CreateObjectFile, CreateLibrary, and CreateExecutable functions."

LibraryPath::usage = "LibraryPath[driver, errHd, opts] returns the library path compiler command options obtained from the library directories named in opts."

MakeDefineDirectives::usage = "makeDefineDirectives[defines] takes a list of strings or rules returns the C compiler define options for them."

MakeIncludePathDirectives::usage = "MakeIncludePathDirectives[paths] generates the C compiler include path directives to search paths for include files."

MarkFileForCleanup::usage = "MarkFileForCleanup[file] will cause file to be deleted, depending on the \"CleanIntermediateFile\" option, at the end of the InvokeCompiler function."

MarkFileForCleanupIfFail::usage = "MarkFileForCleanupIfFail[file] will cause file to be deleted if the compilation failed, depending on the \"CleanIntermediateFile\" option, at the end of the InvokeCompiler function."

MprepCalls::usage = "MprepCalls[tmSrcFiles, workDir, driver, options] returns a list of shell commands to invoke mprep on tmSrcFiles."

$PlatformObjExtension::usage = "$PlatformObjExtension is the file extension used on this platform for compiler object files."

$PlatformDLLExtension::usage = "$PlatformDLLExtension is the file extension used on this platform for dynamic library files."

$PlatformStaticLibExtension::usage = "$PlatformStaticLibExtension is the file extension used on this platform for static library files."

$PlatformExeExtension::usage = "$PlatformExeExtension is the file extension used on this platform for executable program files."

$PlatformPathSeparator::usage = "$PlatformPathSeparator is the string that separates elements in the PATH environment variable."

ResolveCodeFile::usage = "ResolveCodeFile[driver, errHd, codeString, workDir, funName, opts] checks whether codeString is a filename or a C source string, and if the latter creates a file workDir/funName.c."

ResolveIncludeDirs::usage = "ResolveIncludeDirs[incDirs] returns a list of include directory paths, resolving the Automatic setting of the IncludeDirectories option."

ResolveLibraries::usage = "ResolveLibraries[libs] returns a list of library files to link with, resolving the Automatic setting of the Libraries option."

ShortenFilenameWithEnvVar::usage = "ShortenFilenameWithEnvVar[files] examines files looking for a common root directory in the files and returns a list {newfiles,cmd} where if possible the file names have been shortened with an environment variable set by cmd or if not possible then newfiles is the same as files and cmd is an empty string."

WorkingOutputFile::usage = "WorkingOutputFile[workDir, outFile] returns a file path located in workDir using the base filename of outFile."

MathLinkCompilerAdditionsPath::usage = "MathLinkCompilerAdditionsPath[sysid] returns the path to the MathLink library."

WSTPCompilerAdditionsPath::usage = "WSTPCompilerAdditionsPath[sysid] returns the path to the WSTP library."
(*** Messages ***)

General::csrcfile = "C source file `1` was not found during compilation."

General::wddirty = "The C compiler working directory is not empty, some files may be overwritten."

General::wddel = "The C compiler working directory is not empty, files will be removed before compilation."

CommandJoin::type = "Non-string expressions found: `1`"

Begin["`Private`"] (* Begin Private Context *) 

Needs["CCompilerDriver`"] (* to pick up QuoteFile and QuoteFiles *)

CommonCompilerOptions =
	{
	"CleanIntermediate" -> Automatic
	,
	"CreateBinary" -> True
	,
	"Debug" -> False
	,
	"Defines" -> {}
	,
	"CompileOptions" -> {}
	,
	"CompilerInstallation" -> Automatic
	,
	"CompilerName" -> Automatic
	,
	"ExtraObjectFiles" -> {} (* unused by CreateObjectFile *)
	,
	"IncludeDirectories" -> {}
	,
	"Language" -> Automatic
	,
	"Libraries" -> {} (* unused by CreateObjectFile *)
	,
	"LibraryDirectories" -> {} (* unused by CreateObjectFile *)
	,
	"LibraryType" -> "Dynamic" (* unused by CreateObjectFile and CreateExecutable *)
	,
	"LinkerOptions" -> {} (* unused by CreateObjectFile *)
	,
	"MprepOptions" -> "" (* unused by CreateObjectFile and CreateLibrary *)
	,
	"PreCompileCommands" -> ""
	,
	"PostCompileCommands" -> ""
	,
	"ShellCommandFunction" -> None
	,
	"ShellOutputFunction" -> None
	,
	"SystemCompileOptions" -> {}
	,
	"SystemDefines" -> Automatic
	,
	"SystemIncludeDirectories" -> Automatic
	,
	"SystemLibraries" -> Automatic (* unused by CreateObjectFile *)
	,
	"SystemLibraryDirectories" -> Automatic (* unused by CreateObjectFile *)
	,
	"SystemLinkerOptions" -> {}(* unused by CreateObjectFile *)
	,
	"TargetDirectory" -> CCompilerDriver`$CCompilerDefaultDirectory
	,
	"TargetSystemID" -> $SystemID
	,
	"TransferProtocolLibrary" -> Automatic
	,
	"WorkingDirectory" -> Automatic
	}

(* Option Exceptions. *)

BaseDriver["OptionsExceptions"]["CreateLibrary"] := {"MprepOptions"}

BaseDriver["OptionsExceptions"]["CreateExecutable"] := {"LibraryType"}

BaseDriver["OptionsExceptions"]["CreateObjectFile"] := 
	{"ExtraObjectFiles", "Libraries", "LibraryType", "LinkerOptions", 
		"MprepOptions", "SystemLibraries", "SystemLinkerOptions"}

(* System configuration of interest. *)
Bits[] := Bits[$SystemID]

Bits["Windows-x86-64" | "Linux-x86-64" | "MacOSX-x86-64"] := 64

Bits[_] := 32

SystemIDBits["Windows-x86-64" | "Linux-x86-64" | "MacOSX-x86-64"] := 64

SystemIDBits[_] := 32

$SystemIDBits = SystemIDBits[$SystemID]

$Bits = Bits[]

$ILP64 = TrueQ[Log[2., Developer`$MaxMachineInteger] > 32]

$RequiredDefines = If[$SystemIDBits === 64 && $ILP64, {}, {"\"MINT_32\""}]

$PlatformObjExtension = 
	Switch[$OperatingSystem, 
		"Windows", ".obj",
		_, ".o"]

$PlatformDLLExtension = 
	Switch[$OperatingSystem, 
		"MacOSX", ".dylib",
		"Windows", ".dll",
		_, ".so"]

$PlatformStaticLibExtension = If[$OperatingSystem === "Windows", ".lib", ".a"]

$PlatformExeExtension = 
	Switch[$OperatingSystem, 
		"Windows", ".exe",
		_, ""]

$PlatformPathSeparator = If[$OperatingSystem === "Windows", ";", ":"]

$MathLinkPath = FileNameJoin[{$InstallationDirectory, "SystemFiles", "Links", "MathLink"}]

$WSTPPath = FileNameJoin[{$InstallationDirectory, "SystemFiles", "Links", "WSTP"}]

(******************************************************************************)
BuildAndClean[driver_, errHd_, buildCommand_, outFiles:{outFile_, ___}, 
	funName_, workDir_, cleanIntermediate_, opts___] := 
	Module[{workingOutfile, workingOutfiles, buildOutput, handleShellOutput, 
		result, errors},
		workingOutfile = WorkingOutputFile[workDir, outFile];
		workingOutfiles = WorkingOutputFile[workDir, #]& /@ outFiles;
		Quiet[Scan[DeleteFile, outFiles]];
		ChangeDirectoryBlock[workDir,
			Quiet[Scan[DeleteFile, workingOutfiles]];

			buildOutput = Import["!"<>buildCommand<>" 2>&1", "Text"];

			result = 
				If[FileType[workingOutfile] === File, 
					Scan[Function[ofile,
						Quiet[DeleteFile[ofile]];
						CopyFile[WorkingOutputFile[workDir, ofile], ofile]],
						outFiles];
					outFile
				(* Else *), 
					$Failed
				];
			handleShellOutput = Quiet@OptionValue[driver, opts, 
				"ShellOutputFunction"];
			If[handleShellOutput =!= None, handleShellOutput[buildOutput]];

			errors = driver["ExtractErrors"][buildOutput];
			If[ListQ[errors],
				Do[Message[errHd::cmperr, err], {err, errors}]
			];

			CleanIntermediateFiles[driver, errHd, funName, workDir, outFiles, 
				opts];
		];
		result
	]

(******************************************************************************)
SetAttributes[ChangeDirectoryBlock, HoldRest];
ChangeDirectoryBlock[dir_, expr_] := 
	Module[{result},
		SetDirectory[dir];
		result = expr;
		ResetDirectory[];
		result
	]

(******************************************************************************)
CheckDirectory[driver_, errHd_Symbol, dir_] := 
	With[{result = EnsureDirectoryExists[dir]},
		If[StringQ[result],
			AbsoluteFileName[dir]
		,(* Else *)
			Message[errHd::nodir, dir];
			Throw[$Failed]
		]
	]

CheckDirectory[_] := Throw[$Failed]

(******************************************************************************)
CleanIntermediateFiles[driver_, errHd_Symbol, funName_String, workDir_String, 
	outFiles:{outFile_String, ___}, opts_] := 
	Module[{clean = GetOption[driver, errHd, {opts}, "CleanIntermediate"]},
		Switch[clean, 
			True | Automatic, 
				Quiet[DeleteFile[WorkingOutputFile[workDir, outFile]]];
				CleanIntermediateFiles[funName, workDir, outFiles],
			Full,
				DeleteFile[FileNames["*", workDir]];
		];

		If[MatchQ[clean, True | Automatic | Full],
			Quiet@Scan[DeleteFile,
				With[{n="ExtraIntermediateFiles"},
					OptionValue[{n -> {}}, opts, n]]
			]
		];
	]

CleanIntermediateFiles[funName_String, dir_String, outFiles_] :=
	Module[{files = FileNames[ funName <> ".*", dir]},
		files = Select[files, FileType[#] === File && !MemberQ[outFiles, #]&];
		DeleteFile[files];
	]

(******************************************************************************)
CommandJoin[commandseq___] := 
	With[{flatcommands = Flatten[{commandseq}]},
		With[{nonStrings = DeleteCases[flatcommands, _String]},
			If[nonStrings === {},
				With[{result = Quiet[StringJoin[flatcommands]]},
					If[StringQ[result],
						result
						(* Else *),
						Message[CommandJoin::type, InputForm[result]];
						Throw[$Failed]
					]
				],
				Scan[Message[CommandJoin::type, InputForm[#]]&, nonStrings];
				Throw[$Failed]
			]
		]
	]

(******************************************************************************)
CreateSourceFileBuildRules[errHd_Symbol, workDir_String, srcFiles_List] := 
	CreateSourceFileBuildRule[errHd, workDir, srcFiles, #]& /@ srcFiles

CreateSourceFileBuildRule[errHd_Symbol, workDir_, srcFiles_, 
	tmFile_String?MathLinkTemplateFileQ] := 
	With[{tmcFile = MathLinkTemplateOutputFile[workDir, tmFile]},
		If[FileExistsQ[tmcFile],
			Message[errHd::tmclash, tmcFile];
			Throw[$Failed]];
		tmFile -> tmcFile
	]

CreateSourceFileBuildRule[errHd_Symbol, workDir_, srcFiles_List, 
	srcFile_String] := srcFile

(******************************************************************************)
(*
Convert a source file to an object name, using the extension ext.
Note that the root of the file is dropped.
*)
ConvertSourceToObject[srcFiles_List, ext_] := 
	ConvertSourceToObject[#, ext]& /@ srcFiles

ConvertSourceToObject[ fname_String, ext_] :=
	Module[{base, dot},
		base = FileBaseName[ fname];
		dot = If[StringMatchQ[ext, "." ~~ ___], "", "."];
		FileNameJoin[ {base}] <> dot <> ext
	]

(******************************************************************************)
DeriveOptions[opts_List, exceptions_:{}] := 
	DeleteCases[
		stripDuplicates[Join[opts, CommonCompilerOptions]],
		Rule[Alternatives@@exceptions, _]
	]

stripDuplicates::usage = "stripDuplicates[rules] returns rules with only the first occurrence of each key."

stripDuplicates[{}] := {}

stripDuplicates[{r : Rule[x_, y_], xs___}] :=
	Prepend[Select[{xs}, First[#] =!= x &], r]
 
(******************************************************************************)
EnsureDirectoryExists[dir_] :=
(
	If[!QuietDirectoryQ[dir], 
		CreateDirectory[dir]
	];
	If[QuietDirectoryQ[dir],
		dir,
		$Failed
	]
)

QuietDirectoryQ[dir_String] := Quiet[DirectoryQ[dir]]

QuietDirectoryQ[___] := False

(******************************************************************************)

GetOption[driver_, errHd_Symbol, opts_List, n:"CleanIntermediate"] := 
	GetTypeCheckedOption[driver, opts, n, True | False | Automatic | Full,
			Message[errHd::cleantype, #]&]

GetOption[driver_, errHd_Symbol, opts_, n:"CompileOptions"] := 
	Module[{sysoptions, compileoptions},

		sysoptions = GetTypeCheckedOption[driver, opts, "SystemCompileOptions",
				_String | {_String ...}, 
				Message[errHd::compopt, "SystemCompileOptions", #]&
			] // normalizeToList;

		compileoptions = GetTypeCheckedOption[driver, opts, n,
			_String | {_String ...}, Message[errHd::compopt, n, #]&
		] // normalizeToList;

		Join[sysoptions, compileoptions]
	]

BaseDriver["DebugSystemOptions"][] := {"-g"}

GetOption[driver_, errHd_Symbol, opts_List, n:"CompilerInstallation"] :=
	With[{installation = GetTypeCheckedOption[driver, opts, n, 
		Automatic | _String, Message[errHd::instltype, #]&]},
		driver["ResolveInstallation"][installation]
	]

GetOption[driver_, errHd_Symbol, opts_List, n:"CompilerName"] :=
	With[{installation = GetTypeCheckedOption[driver, opts, n, 
		Automatic | _String, Message[errHd::compnmtype, #]&]},
		driver["ResolveCompilerName"][installation]
	]

GetOption[driver_, errHd_Symbol, opts_List, n:"CreateBinary"] := 
	GetTypeCheckedOption[driver, opts, n, True | False,
		Message[errHd::crbin, #]&]

GetOption[driver_, errHd_Symbol, opts_List, n:"Debug"] := 
	GetTypeCheckedOption[driver, opts, n, True | False,
		Message[errHd::debug, #]&]

GetOption[driver_, errHd_Symbol, opts_List, n:"TransferProtocolLibrary"] := 
	GetTypeCheckedOption[driver, opts, n, Automatic | "MathLink" | "WSTP",
		Message[errHd::translib, #]&]

$DefineRuleTypePattern = Rule[_String, _String | _Integer]

$DefinesTypePattern = _String | $DefineRuleTypePattern |
	{(_String | $DefineRuleTypePattern ) ...}

GetOption[driver_, errHd_Symbol, opts_List, n:"Defines"] := 
	With[{sysdefs = GetTypeCheckedOption[driver, opts, "SystemDefines",
			Automatic | $DefinesTypePattern, Message[errHd::sysdef, #]&],
		defs = GetTypeCheckedOption[driver, opts, n, $DefinesTypePattern,
			Message[errHd::def, #]&]},
		ResolveDefines[sysdefs, defs]
	]

GetOption[driver_, errHd_Symbol, opts_List, n:"ExtraObjectFiles"] := 
	GetTypeCheckedOption[driver, opts, n, _String | {_String ...},
		Message[errHd::filelist, n, #]&]  // normalizeToList

GetOption[driver_, errHd_Symbol, opts_List, "IncludePath", targetSystemID_] :=
	With[{sysincpath = GetTypeCheckedOption[driver, opts, 
			"SystemIncludeDirectories", Automatic | _String | {_String ...}, 
			Message[errHd::sysdirlist, "SystemIncludeDirectories", #]&],
		incpath = GetTypeCheckedOption[driver, opts, "IncludeDirectories", 
			_String | {_String ...}, 
			Message[errHd::dirlist, "IncludeDirectories", #]&],
		
	    	translib = GetOption[driver, errHd, opts, "TransferProtocolLibrary"]
	    }, ResolveIncludeDirs[sysincpath, incpath, targetSystemID, translib]
	]

GetOption[driver_, errHd_Symbol, opts_List, n:"Language"] :=
	If[MemberQ[Options[driver], Rule["Language", _]],
		GetTypeCheckedOption[driver, opts, n, Automatic | "C" | "C++",
			Message[errHd::lang, #]&],
		Null
	]

GetOption[driver_, errHd_Symbol, opts_List, n:"MprepOptions"] :=
	GetTypeCheckedOption[driver, opts, n, _String,
		Message[errHd::mpreptype, #]&]

GetOption[driver_, errHd_Symbol, opts_List, n:"PreCompileCommands"] :=
	GetTypeCheckedOption[driver, opts, n, _String,
		Message[errHd::precompiletype, #]&]

GetOption[driver_, errHd_Symbol, opts_List, n:"PostCompileCommands"] :=
	GetTypeCheckedOption[driver, opts, n, _String,
		Message[errHd::postcompiletype, #]&]

GetOption[driver_, errHd_, opts_List, n:"Libraries"] := 
	StringTrim[
		GetTypeCheckedOption[driver, opts, n, _String | {_String ...},
			Message[errHd::filelist, n, #]&]] // normalizeToList

GetOption[driver_, errHd_Symbol, opts_List, n:"LibraryDirectories"] :=
	GetTypeCheckedOption[driver, opts, n, _String | {_String ...},
		Message[errHd::dirlist, n, #]&] // normalizeToList

GetOption[driver_, errHd_Symbol, opts_List, n:"LinkerOptions"] := 
	Module[{sysoptions, linkeroptions},

		sysoptions = GetTypeCheckedOption[driver, opts, "SystemLinkerOptions",
				_String | {_String ...}, 
				Message[errHd::compopt, "SystemLinkerOptions", #]&
			] // normalizeToList;

		linkeroptions = GetTypeCheckedOption[driver, opts, n,
			_String | {_String ...}, Message[errHd::lnkopt, n, #]&
		] // normalizeToList;

		Join[sysoptions, linkeroptions]
	]

GetOption[driver_, errHd_Symbol, opts_List, n:"SystemLibraries", 
	targetSystemID_, method_] :=
	With[{translib = GetOption[driver, errHd, opts, "TransferProtocolLibrary"]},
		ResolveLibraries[driver, 
			GetTypeCheckedOption[driver, opts, n, 
				Automatic | All | True | _String | {_String ...},
				Message[errHd::sysfilelist, n, #]&], targetSystemID, method, translib]
	]

GetOption[driver_, errHd_Symbol, opts_List, n:"SystemLibraryDirectories", 
	targetSystemID_] :=
	ResolveLibraryDirectories[driver, errHd, GetTypeCheckedOption[driver, opts, n, 
		Automatic | _String | {_String ...}, Message[errHd::sysdirlist, n, #]&],
		targetSystemID, opts]

GetOption[driver_, errHd_Symbol, opts_List, n:"TargetDirectory"] := 
	Module[{dir = GetTypeCheckedOption[driver, opts, n, _String, 
		Message[errHd::tgtdir, #]&]},
		If[!QuietDirectoryQ[dir],
			MarkFileForCleanupIfFail[dir]
		];
		CheckDirectory[driver, errHd, dir]
	]

GetOption[driver_, errHd_Symbol, opts_List, n:"TargetSystemID"] :=
	GetTypeCheckedOption[driver, opts, n, _String,
		Message[errHd::targettype, #]&] /. {
		x_String?ValidSystemIDQ :> x,
		x_ :> (Message[errHd::system, x]; Throw[$Failed])
	}

GetOption[driver_, errHd_Symbol, opts_List, n:"WorkingDirectory", outDir_] := 
	With[{specifiedWorkingDir = GetTypeCheckedOption[driver, opts, n, 
			_String | Automatic, Message[errHd::wdtype, #]&]},
		ResolveWorkingDir[specifiedWorkingDir, outDir]
	]

GetOption[driver_, opts_List, name_String] := 
	getOptionValue[driver, opts, name]

normalizeToList[x_List] := x

normalizeToList[""] := {}

normalizeToList[x_String] := {x}

normalizeToList[r_Rule] := {r}

GetTypeCheckedOption[driver_, opts_List, optname_String, pattern_, onFail_] := 
	With[{specifiedValue = getOptionValue[driver, opts, optname]},
		If[MatchQ[specifiedValue, pattern],
			specifiedValue,
			onFail[specifiedValue];
			Throw[$Failed]
		]
	]

GetTypeCheckedOption::arg = "GetTypeCheckedOption was not passed a driver, options list, option name, pattern, and onFail function: `1`"

GetTypeCheckedOption[args___] :=
(
	Message[GetTypeCheckedOption::arg, {args}];
	Throw[$Failed]
)

getOptionValue::usage = "getOptionValue[driver, options, name] quietly retrieves the value for driver's option name given the list of supplied options."

getOptionValue[driver_, options_List, optname_String] := 
	Quiet[OptionValue[Options[driver], options, optname]]

GetOption::arg = "GetOption was not passed a driver, an option list, and an option name: `1`"

GetOption[args___] := 
(
	Message[GetOption::arg, {args}];
	$Failed
)

(******************************************************************************)
CompileOptions[driver_, errHd_, opts_List] := 
	Riffle[GetOption[driver, errHd, opts, "CompileOptions"], " "]

(******************************************************************************)
Defines[driver_, errHd_Symbol, opts_List] := 
	MakeDefineDirectives[
		GetOption[driver, errHd, opts, "Defines"], 
		"DefineFlag" -> driver["DefineFlag"][]
	]

BaseDriver["DefineFlag"][] := "-D"

(******************************************************************************)
BaseDriver["ExtractErrors"][buildOutput_] := 
	Module[{lines, errors},
		lines = StringSplit[buildOutput, {"\n", "\r\n", "\r"}];
		errors = Select[lines, errorQ];
		errors
	]

errorQ[line_String] := 
	StringMatchQ[line, ___ ~~ "error: " ~~ ___ ]

errorQ[_] := False
(******************************************************************************)
IncludePath[driver_, errHd_Symbol, opts_List, targetSystemID_] := 
	MakeIncludePathDirectives[
		GetOption[driver, errHd, opts, "IncludePath", targetSystemID], 
		"IncludePathFlag" -> driver["IncludePathFlag"][]
	]

BaseDriver["IncludePathFlag"][] := "-I"

(******************************************************************************)
BaseDriver["Installations"][] := {}
(******************************************************************************)
InvokeCompiler[driver_, compileFn_, ext_, code:(_String|{(_String|_File) ..}|File[_String]),
	funName_String, optseq___] := 
	Module[{opts = {optseq}, errHd = CCompilerDriver`$ErrorMessageHead /. 
				{Automatic :> compileFn, hd_ :> hd}, doClean, cmpRes},

		(* If Debug appears, replace it with debug settings *)
		If[TrueQ[GetOption[driver, errHd, opts, "Debug"]],
			opts = ReplaceDebugOptions[opts, {
				"ShellOutputFunction" -> Print,
				"CleanIntermediate" -> False,
				"SystemCompileOptions" -> driver["DebugSystemOptions"][]
			}];
			opts = Flatten[opts];
		];

		cmpRes = First@Reap[
			cmpRes = Reap[
				Catch[
					doClean = GetOption[driver, errHd, opts, "CleanIntermediate"];
					InvokeCompilerImpl[driver, compileFn, errHd, ext, code, funName, opts]
				],
				$IntermediateFileTag, cleanIntermediates[doClean, #2]&
			], $CleanupFileIfFailTag, cleanIntermediates[doClean && First[cmpRes] === $Failed, #2]&
		];
		If[ListQ[cmpRes],
			First[cmpRes],
			cmpRes
		]
	]

ReplaceDebugOptions[opts_, replacement_] := 
	ReplaceDebugOptions[opts, replacement, Position[opts, "Debug"->True]]
	(* Potential bug: this will find the rule "Debug"->True anywhere, even if
		it appears, for whatever reason, as an option value. *)

ReplaceDebugOptions[opts_, _, {}] := opts

ReplaceDebugOptions[opts_, replacement_, {pos_, ___}] := 
	ReplacePart[opts, pos -> replacement]

$IntermediateFileTag = "CCompilerDriver`CCompilerDriverBase`IntermediateFile"
$CleanupFileIfFailTag = "CCompilerDriver`CCompilerDriverBase`CleanupFileIfFail"

(* Clean a given set of files *)
cleanIntermediates[Automatic | True | Full, files_] :=
	Quiet[Scan[deleteFile, files]]

deleteFile[file_String] := 
(
	If[FileExistsQ[file],
		Switch[FileType[file],
			File, DeleteFile[file],
			Directory, DeleteDirectory[file, DeleteContents->True]
		]
	]
)

InvokeCompilerImpl[driver_, compileFn_, errMsgHd_, ext_,
	code:(_String|{(_String|_File) ..}|File[_String]), funName_String, opts_List] :=
	Module[{outputBaseName, outputFileName, compilerInstallation, compilerName, 
		cleanIntermediate, outDir, workDir, srcFiles, outFile, outFiles,
		targetSystemID, setupEnv, compileCommands, buildCommand, tmcFiles, 
		objFiles, compileOptions, linkerOptions, defines, includePath, 
		srcFileRules, tmSrcFiles, cFiles, syslibs, libs, libpath, extraObjects, 
		mprepOptions, createBinary, translib, language, buildScript, linkWithMathLinkQ,
		libType},

		(* Compute a simple name, e.g. "test", without the file extension *)
		outputBaseName = StringReplace[funName, 
			{ext ~~ (NumberString ...) ~~ EndOfString -> ""}, 
			IgnoreCase -> True];

		compilerInstallation = 
			GetOption[driver, errMsgHd, opts, "CompilerInstallation"];
		compilerName = GetOption[driver, errMsgHd, opts, "CompilerName"];
		targetSystemID = GetOption[driver, errMsgHd, opts, "TargetSystemID"];

		If[SameQ[False, driver["ValidInstallationQ"][compilerInstallation]] ||
			!TrueQ[driver["ValidInstallationQ"][compilerInstallation, 
				compilerName, targetSystemID]],
			Message[errMsgHd::instl, compilerInstallation, 
				driver["Name"][]];
			Throw[$Failed]
		];
		(* TODO do separate validation of compiler installation, name, and target system ID *)

		If[!TrueQ[driver["SupportedTargetSystemIDQ"][targetSystemID,
			compilerInstallation]],
			Message[errMsgHd::target, targetSystemID, 
				driver["Name"][], compilerInstallation];
			Throw[$Failed]
		];

		outDir = GetOption[driver, errMsgHd, opts, "TargetDirectory"];
		workDir = GetOption[driver, errMsgHd, opts, "WorkingDirectory", 
			outDir];
		If[!QuietDirectoryQ[workDir],
			CreateDirectory[workDir];
			MarkFileForCleanup[workDir];
		];

		cleanIntermediate = 
			GetOption[driver, errMsgHd, opts, "CleanIntermediate"];
		Preclean[errMsgHd, workDir, cleanIntermediate];
		srcFiles = ResolveCodeFile[driver, errMsgHd, code, workDir, 
			outputBaseName, opts];

		(* For the simple file name, base it on outputBaseName unless we 
		   stripped an extension from funName, in which case use funName.
		*)
		outputFileName = If[funName === outputBaseName, funName<>ext, funName];
		outFile = FileNameJoin[{outDir, outputFileName}];
		outFiles = {};
		(* The CleanIntermediate option controls whether the .exp and .lib files
			are treated as build output, so predicate fetching them on that. *)
		If[cleanIntermediate === Automatic,
			outFiles = driver["SecondaryOutputFiles"][compileFn, outFile];
			If[!ListQ[outFiles],
				outFiles = {}
			]
		];
		PrependTo[outFiles, outFile];

		(* Turn {a.c, b.tm, c.c} into {a.c, b.tm -> btm.c, c.c} *)
		srcFileRules = CreateSourceFileBuildRules[errMsgHd, workDir, srcFiles];

		compileOptions = CompileOptions[driver, errMsgHd, opts];
		linkerOptions = GetOption[driver, errMsgHd, opts, "LinkerOptions"];
		defines = Defines[driver, errMsgHd, opts];
		includePath = IncludePath[driver, errMsgHd, opts, targetSystemID];
		tmSrcFiles = Cases[srcFileRules, Rule[tmFile_, _] :> tmFile];
		cFiles = If[Head[#]===Rule, #[[2]], #]& /@ srcFileRules;
		libs = GetOption[driver, errMsgHd, opts, "Libraries"];
		libpath = LibraryPath[driver, errMsgHd, opts, targetSystemID];
		extraObjects = GetOption[driver, errMsgHd, opts, "ExtraObjectFiles"];
		mprepOptions = GetOption[driver, errMsgHd, opts, "MprepOptions"];
		translib = GetOption[driver, errMsgHd, opts, "TransferProtocolLibrary"];
		language = GetOption[driver, errMsgHd, opts, "Language"];

		(* Run a type check on options not retrieved yet, 
			before generating commands *)
		linkWithMathLinkQ = driver["LinkWithMathLink"][compileFn] ||
			Length[tmSrcFiles] > 0 ;
		syslibs = GetOption[driver, errMsgHd, opts, "SystemLibraries", 
			targetSystemID, linkWithMathLinkQ];
		createBinary = GetOption[driver, errMsgHd, opts, "CreateBinary"];

		libType = GetOption[driver, opts, "LibraryType"];

		compileCommands = createCompileCommands[driver, compileFn, libType,
			errMsgHd, compilerInstallation, compilerName, outFile, workDir,  
			compileOptions, linkerOptions, defines, includePath, srcFileRules, 
			tmSrcFiles, cFiles, syslibs, libs, libpath, extraObjects, 
			targetSystemID, cleanIntermediate, mprepOptions, translib, language, opts];
			
		If[MatchQ[compileCommands, {_String, _String}],
			{setupEnv, compileCommands} = compileCommands
			,
			setupEnv = ""
		];

		compileCommands = StringTrim@StringJoin@Riffle[
			{
				setupEnv,
				GetOption[driver, errMsgHd, opts, "PreCompileCommands"],
				compileCommands,
				GetOption[driver, errMsgHd, opts, "PostCompileCommands"]
			}, 
			"\n"];

		If[!StringQ[compileCommands],
			(* An internal error.  createCompileCommands should return a string *)
			Message[InvokeCompiler::cmdstr, InputForm[compileCommands]];
			Throw[$Failed]
		];

		{buildCommand, buildScript} = mkBuildCommand[workDir, outputBaseName, 
			compileCommands, createBinary];

		If[TrueQ[createBinary],
			MarkFileForCleanup[buildScript]
			,
			buildScript = FileNameJoin[{FileNameDrop[buildScript, -2], 
				FileNameTake[buildScript, -1]}];
			AppendTo[outFiles, buildScript]
		];

		With[{cmdFn = GetOption[driver, opts, "ShellCommandFunction"]},
			If[cmdFn =!= None, cmdFn[compileCommands]]];
		If[TrueQ@createBinary,
			tmcFiles = Cases[srcFileRules, Rule[_, out_] :> out];
			objFiles = ConvertSourceToObject[cFiles, $PlatformObjExtension];
			BuildAndClean[driver, errMsgHd, buildCommand, outFiles, 
				outputBaseName, workDir, cleanIntermediate,
				Append[opts,
					"ExtraIntermediateFiles" -> Join[tmcFiles, objFiles]]]
		(* Else *),
			Quiet@DeleteFile[buildScript];
			CopyFile[WorkingOutputFile[workDir, buildScript], buildScript];
			outFile
		]
	]

InvokeCompiler::cmdstr = "An internal error occurred while generating compilation commands. The InputForm of the generated command list is `1`"

If[$OperatingSystem === "Windows",
	BaseDriver["SecondaryOutputFiles"][CCompilerDriver`CreateLibrary, outFile_] := 
		With[{dir = DirectoryName[outFile], basename = FileBaseName[outFile]},
			{FileNameJoin[{dir, basename<>".lib"}], 
				FileNameJoin[{dir, basename<>".exp"}]}
		]
]

BaseDriver["SecondaryOutputFiles"][___] := {}

(* By default, CreateExecutable should not be using MathLink *)
BaseDriver["LinkWithMathLink"][CCompilerDriver`CreateExecutable] := False

BaseDriver["LinkWithMathLink"][___] := True

BaseDriver["ValidInstallationQ"][installation_, ___] := 
	MatchQ[Quiet[FileType[installation]], File | Directory]

Switch[$OperatingSystem,
	"Windows",
		BaseDriver["SupportedTargetSystemIDQ"][
			"Windows" | "Windows-x86-64", ___] := True,
	"Unix",
		BaseDriver["SupportedTargetSystemIDQ"][
			"Linux" | "Linux-x86-64" | "Linux-ARM", ___] := True,
	"MacOSX",
		BaseDriver["SupportedTargetSystemIDQ"][
			"MacOSX-x86-64", ___] := True
]

BaseDriver["SupportedTargetSystemIDQ"][targetSystemID_, ___] := False

createCompileCommands[driver_, createFn:CreateLibrary, "Static", args___] :=
	 driver["CreateStaticLibraryCommands"][args]

createCompileCommands[driver_, createFn_, _(*libType ignored*), args___] :=
	 driver[SymbolName[createFn]<>"Commands"][args]

If[$OperatingSystem === "Windows",
	mkBuildCommand[workDir_, funName_, compileCommands_, createBinary_] := 
		Module[{batBaseName = funName <> ".bat", batFile},
			batFile = FileNameJoin[{workDir, batBaseName}];
			Export[batFile, compileCommands, "Text"];
			{batBaseName, batFile}
		]
	,
	(* In Unix, use the commands directly *)
	mkBuildCommand[workDir_, funName_, compileCommands_, createBinary_] := 
		Module[{shellScriptBaseName = funName <> ".sh", shellScript},
			shellScript = FileNameJoin[{workDir, shellScriptBaseName}];
			If[!TrueQ[createBinary],
				Export[shellScript, "#!/bin/sh\n"<>compileCommands, "Text"];
			];
			{compileCommands, shellScript}
		]
]

(*****************************************************************************)
LibraryPath[driver_, errHd_Symbol, opts_List, targetSystemID_] := 
	MakeLibraryPathDirectives[driver,
		Join[
			GetOption[driver, errHd, opts, "SystemLibraryDirectories", 
				targetSystemID],
			GetOption[driver, errHd, opts, "LibraryDirectories"]
		]
	]

BaseDriver["LibraryPathFlag"][] := "-L"

(******************************************************************************)
Options[MakeDefineDirectives] = {"DefineFlag" -> "-D" }

MakeDefineDirectives[defines_String, opts:OptionsPattern[]] := 
	MakeDefineDirectives[StringSplit[defines], opts]

MakeDefineDirectives[defines:{(_String | _Rule) ...}, opts:OptionsPattern[]] := 
	StringJoin[Riffle[#, " "]& @
		Map[MakeDefineDirective[#, OptionValue["DefineFlag"]]&, defines]
	]
			
(* Simple define like -D WINDOWS *)
MakeDefineDirective[key_String, defineFlag_] := defineFlag<>key

(* Valued define like -D CPUARCH=x86 *)
MakeDefineDirective[key_String -> value_String, defineFlag_] := 
	MakeDefineDirective[key, defineFlag]<>"="<>value

MakeDefineDirective[key_String -> value:(_Integer | _Real), defineFlag_] := 
	MakeDefineDirective[key -> ToString[value], defineFlag]

(******************************************************************************)
Options[MakeIncludePathDirectives] = {"IncludePathFlag" -> "-I" };
MakeIncludePathDirectives[paths:{_String...}, OptionsPattern[]] := 
	StringJoin[Riffle[
		(OptionValue["IncludePathFlag"]<>
			QuoteFile[
				(* Strip trailing slashes, some compilers are picky *)
				StringReplace[#, {("/" | "\\") .. ~~ EndOfString -> ""}]]
		)& /@ paths, " "]]

(******************************************************************************)
MakeLibraryPathDirectives[driver_, dirs:{_String ...}] := 
	With[{libdirFlag = driver["LibraryPathFlag"][]},
		StringJoin[
			Map[Function[libdir,
				If[StringMatchQ[libdir, StartOfString ~~ "-F" ~~ ___], 
					{"-F",QuoteFile[StringTrim[StringReplace[libdir, "-F"->""]]], " "}, 
				{libdirFlag, QuoteFile[libdir], " "}]], dirs]
		]
	]

(******************************************************************************)
MarkFileForCleanup[file_String] := Sow[file, $IntermediateFileTag]
MarkFileForCleanupIfFail[file_String] := Sow[file, $CleanupFileIfFailTag]
 
(******************************************************************************)
MathLinkCompilerAdditionsPath[sysid_] = 
	FileNameJoin[{$MathLinkPath, "DeveloperKit", sysid, "CompilerAdditions"}]

WSTPCompilerAdditionsPath[sysid_] = 
	FileNameJoin[{$WSTPPath, "DeveloperKit", sysid, "CompilerAdditions"}]

(******************************************************************************)
MathLinkTemplateOutputFile[dir_, tmFile_] := 
	FileNameJoin[{dir, FileBaseName[tmFile] <> "tm.c"}]

(******************************************************************************)
MathLinkTemplateFileQ[path_] := StringMatchQ[path, ___ ~~ ".tm"]

(******************************************************************************)
MprepCalls[tmSrcFiles_List, workDir_String, translib_, mprepOptions_] := 
	Map[
		Function[tmSrc,
			MprepCall[MathLinkTemplateOutputFile[workDir, tmSrc], tmSrc, translib, mprepOptions]],
		tmSrcFiles
	]
(******************************************************************************)
MprepCall[outFile_, tmFile_, translib_, mprepOptions_] :=
(
	MarkFileForCleanup[outFile];
	StringJoin[QuoteFile[If[translib === "WSTP", $WSprepExePath, $MprepExePath]],
		" -o ", QuoteFile[outFile],
		If[StringQ[mprepOptions], " "<>mprepOptions, ""],
		" ", QuoteFile[tmFile], "\n"]
)

(******************************************************************************)
$MprepExePath = 
	FileNameJoin[{MathLinkCompilerAdditionsPath[$SystemID], "mprep"<>$PlatformExeExtension}]

$WSprepExePath = 
	FileNameJoin[{WSTPCompilerAdditionsPath[$SystemID], "wsprep"<>$PlatformExeExtension}]

(******************************************************************************)
BaseDriver["ExecutableExtension"][___] := $PlatformExeExtension

BaseDriver["LibraryExtension"][optseq___] := 
	With[{libType = OptionValue[CreateLibrary, {optseq}, "LibraryType"]},
		If[libType === "Static",
			$PlatformStaticLibExtension,
			$PlatformDLLExtension
		]
	]

BaseDriver["ObjectFileExtension"][___] := $PlatformObjExtension

(******************************************************************************)
Preclean[errHd_Symbol, workDir_String, doClean_] :=
	Module[{files},
		files = FileNames["*", workDir];
		If[Length[files] > 0,
			Switch[doClean,
				 False | True | Automatic,
				 	Message[errHd::wddirty],
				 Full,
				 	Message[errHd::wddel];
				 	DeleteFile[files]
			]
		]
	]

(******************************************************************************)
ResolveCodeFile[driver_, args___] := driver["ResolveCodeFile"][args]

BaseDriver["ResolveCodeFile"][errHd_, code_String, workDir_String, 
	funName_String, ___] := 
	With[{codeFile = FileNameJoin[{workDir, funName <> ".c"}]},
		Export[ codeFile, code, "Text"];
		MarkFileForCleanup[codeFile];
		{codeFile}
	]

BaseDriver["ResolveCodeFile"][errHd_, srcFiles:{(_String|_File) ..}, workDir_String,
	funName_String, opts_] := 
	Check[
		Map[
			Function[givenPath,
				With[{resolvedPath = FindFile[givenPath]},
					If[Quiet@FileType[resolvedPath] =!= File,
						Message[errHd::fnfnd, givenPath]];
					resolvedPath]],
			srcFiles],
		Throw[$Failed]
	]

BaseDriver["ResolveCodeFile"][errHd_, srcFile:File[path_String], workDir_String,
        funcName_String, opts_] :=
        BaseDriver["ResolveCodeFile"][errHd, {path}, workDir, funcName, opts]

(******************************************************************************)
ResolveDefines[Automatic, defs_] := 
	Join[$RequiredDefines, normalizeToList[defs]]

ResolveDefines[sysdefs_, defs_] :=
	Join[normalizeToList[sysdefs], normalizeToList[defs]]

(******************************************************************************)
ResolveIncludeDirs[Automatic, extDirs_, targetSystemID_, translib_, opts___?OptionQ] := 
	ResolveIncludeDirs[
		{
			FileNameJoin[{$InstallationDirectory, 
				"SystemFiles", "IncludeFiles", "C"}],

			FileNameJoin[{$InstallationDirectory, 
				"SystemFiles", "Links", If[translib === "WSTP", "WSTP", "MathLink"],
				"DeveloperKit", targetSystemID, "CompilerAdditions"}]
		},
	extDirs]

ResolveIncludeDirs[incDir_String, rest___] := 
	ResolveIncludeDirs[{incDir}, rest]

ResolveIncludeDirs[incDir_List, extIncDir_String, rest___] := 
	ResolveIncludeDirs[incDir, {extIncDir}, rest]

ResolveIncludeDirs[incDirs:{_String ...},extIncDirs:{_String ...}, ___] := 
	Join[ incDirs, extIncDirs]

(******************************************************************************)
(*
   The 3rd arg decides if we include the ML library or not.  If we are making 
   a DLL on Linux we do not, but for VS we do.
*)
ResolveLibraries[driver_, Automatic, 
	target: "MacOSX-x86-64", False, translib_] :=(
	ResolveLibraries[driver, Automatic, target, True, translib]
);

ResolveLibraries[driver_, Automatic, targetSystemID_, False, translib_] := {}

ResolveLibraries[driver_, Automatic, targetSystemID_, True, translib_] := 
	driver["ResolveSystemLibraries"][targetSystemID, translib] /. {
		lib_String :> {lib},
		libs_List :> libs
	}

ResolveLibraries[driver_, All | True, targetSystemID_, _, translib_] := 
	driver["ResolveSystemLibraries"][targetSystemID, translib] /. {
		lib_String :> {lib},
		libs_List :> libs
	}

BaseDriver["ResolveSystemLibraries"][targetSystemID_, translib_] :=
	MathLinkLib[targetSystemID, translib]

(* Not used *)
$MathLinkDeveloperKit = {$InstallationDirectory, "SystemFiles", "Links", 
	"MathLink", "DeveloperKit"}

ResolveLibraries[driver_, libDir_String, targetSystemID_, exeQ_, translib_] := 
	ResolveLibraries[driver, {libDir}, targetSystemID, exeQ, translib]

ResolveLibraries[driver_, libDirs:{_String ...}, _, _, _] := libDirs

MathLinkLib::system = "TargetSystemID `1` is not a valid $SystemID value"

MathLinkLib[sysid_, translib_] := 
	If[translib === "WSTP",
		Switch[sysid,
			"Windows", "wstp32i4m.lib",
			"Windows-x86-64", "wstp64i4m.lib",
			"Linux" | "Linux-ARM", "WSTP32i4",
			"Linux-x86-64", "WSTP64i4", 
			"MacOSX-x86-64", "-framework wstp",
			$Failed, {},
			_, Message[MathLinkLib::system, sysid]; {}
		],
		Switch[sysid,
			"Windows", "ml32i4m.lib",
			"Windows-x86-64", "ml64i4m.lib",
			"Linux" | "Linux-ARM", "ML32i4",
			"Linux-x86-64", "ML64i4", 
			"MacOSX-x86-64", "-framework mathlink",
			$Failed, {},
			_, Message[MathLinkLib::system, sysid]; {}
		]
	]

(******************************************************************************)
ResolveLibraryDirectories[driver_, errHd_Symbol, Automatic, sysid_, opts:OptionsPattern[]] :=
	Module[{translib, mlDir, wolframRTLDir},
		translib = GetOption[driver, errHd, opts, "TransferProtocolLibrary"];
		mlDir = If[translib === "WSTP", WSTPCompilerAdditionsPath[sysid], MathLinkCompilerAdditionsPath[sysid]];
		If[sysid === "MacOSX-x86-64", mlDir = "-F" <> mlDir];
		wolframRTLDir = FileNameJoin[{$InstallationDirectory, "SystemFiles", "Libraries", sysid}];
		{mlDir, wolframRTLDir}
	]

ResolveLibraryDirectories[driver_, errHd_Symbol, dir_String, t_, _] := ResolveLibraryDirectories[driver, errHd, {dir}, t]

ResolveLibraryDirectories[driver_, errHd_Symbol, dirs_List, ___] := dirs

(******************************************************************************)
ResolveWorkingDir[Automatic, workDir_] := 
	FileNameJoin[ {workDir, 
		StringJoin["Working-", $MachineName, "-", ToString[$ProcessID], "-",
			ToString[Developer`ThreadID[]], "-", ToString[++$WorkingDirCount]]}]

$WorkingDirCount = 0

ResolveWorkingDir[outDir_, _] := outDir

(******************************************************************************)
(*
ShortenFilenameWithEnvVar[srcFiles:{_String}]:= srcFiles

ShortenFilenameWithEnvVar[srcFiles_List] := 
	Module[{dirs},
		
	]
*)

(******************************************************************************)
ValidSystemIDQ[system_] := 
	system === $SystemID (* $SystemID is correct by definition *) ||
	MatchQ[system, "Windows" | "Windows-x86-64" | "Linux" | "Linux-x86-64" | 
		"Linux-ARM" | "MacOSX-x86-64"]

(******************************************************************************)
(* Return the path to the working copy of the output file,
   E.g. workDir/demo.dll instead of targetDir/demo.dll *)
WorkingOutputFile[workDir_, outFile_] := 
	FileNameJoin[{workDir, FileNameTake[outFile, -1]}]

End[] (* End Private Context *)

EndPackage[]
