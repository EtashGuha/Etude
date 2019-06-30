(*****************************************************************************)
(* :Name: NVCCCompiler.m *)
(* :Title: CUDALink NVCC Compiler Driver *)
(* :Context: CUDALink` *)
(* :Author: Abdul Dakkak *)
(* :Summary: *)
(* :Sources:*)
(* :Copyright: 2010, Wolfram Research, Inc. *)
(* :Mathematica Version: 8.0 *)
(* :Keywords: CUDA, GPU Computing, GPGPU, NVCC Compiler *)
(* :Warnings: None *)
(* :Limitations: None *)
(* :Discussion: *)
(* :Requirements: None *)
(* :Examples: None *)
(*****************************************************************************)

BeginPackage["CUDALink`NVCCCompiler`", {"CCompilerDriver`", "PacletManager`", "CUDALink`"}];

NVCCCompiler::usage = "NVCCCompiler[src, name] compiles the code in src into a Library and returns the full path of the Library";

CUDACCompilers::uasge = "CUDACCompilers[] returns a list of supported C compilers on system."

Begin["`Private`"]

`$ThisDriver = NVCCCompiler

Needs["CCompilerDriver`CCompilerDriverBase`"]
Needs["CCompilerDriver`CCompilerDriverRegistry`"]

CCompilerRegister[$ThisDriver, {
	"Windows", "Windows-x86-64",
	"Linux", "Linux-x86-64",
	"MacOSX-x86", "MacOSX-x86-64"
}]

NVCCCompiler["Available"] :=
	TrueQ[
		Quiet[FileType[$ThisDriver["ResolveInstallation"][Automatic]]] === Directory
	] &&
	TrueQ[
		MemberQ[{Directory, File}, Quiet[FileType[Catch@ResolveXCompilerInstallation[Automatic]]]]
	]

NVCCCompiler["Name"][] := "NVIDIA CUDA Compiler"

NVCCCompiler["Installations"][] :=
	Select[{NVCCCompiler["ResolveInstallation"][Automatic]},
		($ThisDriver["ValidInstallationQ"][#])&
	]

NVCCCompiler["Version"][] := NVCCCompiler["Version"][Automatic]
NVCCCompiler["Version"][x:(Automatic|_String), targetId_:$SystemID] :=
	NVCCCompiler["Version"][x, targetId] = ResolveNVCCVersion[Quiet[ResolveNVCCCompilerCommand[$ThisDriver["ResolveInstallation"][x], targetId]]]

NVCCCompiler["SupportedTargetSystemIDQ"]["Windows-x86-64", installation_:Automatic] :=
	Module[{xcmpinst},
		Catch[
			xcmpinst = Quiet[QuoteFile[ResolveXCompilerInstallation[Automatic, $ThisDriver, "Windows-x86-64"]]];
			Quiet[$ThisDriver["ValidInstallationQ"][installation]] &&
			TrueQ[CCompilerDriver`VisualStudioCompiler`VisualStudioCompiler["SupportedTargetSystemIDQ"]["Windows-x86-64", xcmpinst]]
		] =!= $Failed
	]

NVCCCompiler["RemoveFileCommand"][] :=
	If[$OperatingSystem === "Windows",
		CCompilerDriver`VisualStudioCompiler`VisualStudioCompiler["RemoveFileCommand"][],
		BaseDriver["RemoveFileCommand"][]
	]

NVCCCompiler["ExecutableExtension"][optseq___] :=
	Which[
		TrueQ["CreateCUBIN" /. {optseq}] || TrueQ["CreateCUBIN" /. Options[$ThisDriver]], ".cubin",
		TrueQ["CreatePTX" /. {optseq}] || TrueQ["CreatePTX" /. Options[$ThisDriver]], ".ptx",
		True, $PlatformExeExtension
	]

$ThisDriver["DebugSystemOptions"][] := {"-O0", "-g", "-DDEBUG=1", formatXCompileOptions["-DDEBUG=1"]}

Options[$ThisDriver] =
	Sort@DeriveOptions[
		{
		"SystemCompileOptions" -> {"-O3"}
		,
		"XCompileOptions" -> Automatic
		,
		"XCompilerInstallation" -> Automatic
		,
		"SystemLinkerOptions" -> Automatic
		,
		"CUDAArchitecture" -> Automatic
		,
		"CreateCUBIN" -> False
		,
		"CreatePTX" -> False
		,
		"UnmangleCode" -> False
		},
		(* exclusions *)
		{
		"Language"
		}
	]

$ThisDriver["OptionsExceptions"]["CreateLibrary"] := BaseDriver["OptionsExceptions"]["CreateLibrary"] ~Join~ {"CreateCUBIN", "CreatePTX"}
$ThisDriver["OptionsExceptions"]["CreateExecutable"] := {}
$ThisDriver["OptionsExceptions"]["CreateObjectFile"] := BaseDriver["OptionsExceptions"]["CreateObjectFile"] ~Join~ {"CreateCUBIN", "CreatePTX"}

NVCCCompiler["CreateObjectFileCommands"][errMsgHd_, installation_, compilerName_, outFile_, workDir_,
								  		 compileOptions_, linkerOptions_, defines_, includePath_, srcFileRules_List, tmSrcFiles_,
								  		 cFiles_, syslibs_, libs_, libpath_, extraObjects_, targetSystemID_,
								  		 cleanIntermediate_, mprepOptions_,translib_, language_, opts_] :=
	iCreateCompileCommands[
		errMsgHd, "CreateObjectFile", " -c", installation, compilerName, outFile, workDir, compileOptions, {},
		defines, includePath, {}, "", cFiles, {}, {}, "", {}, targetSystemID, cleanIntermediate, {}, opts
	]

NVCCCompiler["CreateLibraryCommands"][errMsgHd_, installation_, compilerName_, outFile_, workDir_,
								      compileOptions_, linkerOptions_, defines_, includePath_, srcFileRules_List, tmSrcFiles_,
								      cFiles_, syslibs_, libs_, libpath_, extraObjects_, targetSystemID_,
								      cleanIntermediate_, mprepOptions_,translib_, language_, opts_] :=
	iCreateCompileCommands[
		errMsgHd, "CreateLibrary", " -shared", installation, compilerName, outFile, workDir, compileOptions, linkerOptions,
		defines, includePath, srcFileRules, tmSrcFiles, cFiles, syslibs, libs, libpath, extraObjects,
		targetSystemID, cleanIntermediate, mprepOptions, opts
	]

NVCCCompiler["CreateExecutableCommands"][errMsgHd_, installation_, compilerName_, outFile_, workDir_,
								  	 	 compileOptions_, linkerOptions_, defines_, includePath_, srcFileRules_List, tmSrcFiles_,
								  		 cFiles_, syslibs_, libs_, libpath_, extraObjects_, targetSystemID_,
								  		 cleanIntermediate_, mprepOptions_, translib_,language_, opts_] :=
	iCreateCompileCommands[
		errMsgHd, "CreateExecutable", "", installation, compilerName, outFile, workDir, compileOptions, linkerOptions,
		defines, includePath, srcFileRules, tmSrcFiles, cFiles, syslibs, libs, libpath, extraObjects,
		targetSystemID, cleanIntermediate, mprepOptions, opts
	]

iCreateCompileCommands[errMsgHd_, compTarget_, compileFlag_, installation_, compilerName_, outFile_, workDir_,
					   compileOptions_, linkerOptions_, defines_, includePath_, srcFileRules_List, tmSrcFiles_,
					   cFiles_, syslibs_, libs_, libpath_, extraObjects_, targetSystemID_,
					   cleanIntermediate_, mprepOptions_, opts_] :=
	Module[{createCubinQ, createPTXQ, xcompilerInstallation},

		createCubinQ = getAndCheckOption[errMsgHd, opts, "CreateCUBIN"];
		createPTXQ = getAndCheckOption[errMsgHd, opts, "CreatePTX"];
		xcompilerInstallation = QuoteFile[getAndCheckOption[errMsgHd, opts, "XCompilerInstallation"]];

		If[!(Quiet@TrueQ[DirectoryQ[StringTrim[xcompilerInstallation, "\""]] || FileExistsQ[StringTrim[xcompilerInstallation, "\""]]]),
			Message[errMsgHd::invxcmpinstd, ToString@xcompilerInstallation];
			Throw[$Failed]
		];
		If[createCubinQ && createPTXQ,
			Message[errMsgHd::confarg];
			Throw[$Failed]
		];
		If[(createCubinQ || createPTXQ) && compTarget =!= "CreateExecutable",
			Message[errMsgHd::gpucmp, compTarget, If[createCubinQ, "CreateCUBIN", "CreatePTX"]];
			Throw[$Failed]
		];
		If[ResolveNVCCCompilerCommand[installation, targetSystemID] === $Failed,
			Message[errMsgHd::nonvcci, installation];
			Throw[$Failed]
		];
		{
			CommandJoin[
				If[MemberQ[{"Windows", "Windows-x86-64"}, targetSystemID],
	               	CCompilerDriver`VisualStudioCompiler`Private`VCVarsCall[
	               		StringTrim[xcompilerInstallation, "\""],
	               		targetSystemID
	               	] <> "\n"
	               	, (* else not windows *)
	                ""
				]
			],
			CommandJoin[
	            If[compTarget === "CreateExecutable" && !createCubinQ && !createPTXQ,
					MprepCalls[tmSrcFiles, workDir, mprepOptions],
					""
	            ],

	            ResolveNVCCCompilerCommand[errMsgHd, installation, targetSystemID],
	            If[createCubinQ || createPTXQ,
					If[createCubinQ,
						" -cubin",
						" -ptx"
					]
					,
					{
						compileFlag,
						With[{libpth = ResolveWGLLibraryPath[errMsgHd, installation, targetSystemID]},
							If[libpth =!= $Failed,
								{" -L", libpth},
								""
							]
						],
						" -L", ResolveCUDALibraryPath[errMsgHd, installation, targetSystemID],
						If[compTarget =!= "CreateObjectFile",
							linkerOptions,
							{}
						],
						If[compTarget === "CreateExecutable" && MemberQ[{"Windows", "Windows-x86-64"}, targetSystemID],
							formatLinkerOptions["user32.lib kernel32.lib gdi32.lib"],
							""
						],
						If[extraObjects =!= {}, " ", ""], QuoteFiles[extraObjects],
						If[libpath =!= "",
							{
								" ",
								formatLibPath[libpath, targetSystemID]
							},
							""
						] ,
						" ", formatLibraries[errMsgHd, syslibs, libs]
					}
	            ],
	            " ", compilerBitnessOption[targetSystemID],
				 If[$OperatingSystem==="Windows",Nothing,Sequence@@{" --compiler-bindir ", xcompilerInstallation}],
				getAndCheckOption[errMsgHd, opts, "XCompileOptions"],
				ResolveCUDAArchitecture[errMsgHd, targetSystemID, opts],
				If[compileOptions =!= "", " ", ""], compileOptions,
				If[includePath =!= "", " ", ""], includePath,
				If[defines =!= "", " ", ""], defines,
				" -o ", QuoteFile[WorkingOutputFile[workDir, outFile]],
				" ", QuoteFiles[cFiles],
				If[$OperatingSystem === "Windows" , "", " 2>&1"],
				"\n"
			]
		}
	]

formatLibPath[pth_String, sysId_] :=
	If[sysId === "MacOSX-x86-64" && StringMatchQ[pth, StartOfString ~~ "-F" ~~ ___],
		formatXCompileOptions[pth],
		pth
	]

If[$OperatingSystem === "Unix",
	NVCCCompiler["LinkWithMathLink"][CreateLibrary] := False
]

ResolveNVCCCompilerCommand[errMsgHd_, Automatic, targetSystemID_] :=
	ResolveNVCCCompilerCommand[errMsgHd, NVCCCompiler["ResolveInstallation"][Automatic], targetSystemID]
ResolveNVCCCompilerCommand[errMsgHd_, installation_, targetSystemID_] :=
	If[FileType[installation] === File,
		installation,
		Quiet@With[{nvcc = FileNameJoin[{installation, "nvcc" <> $PlatformExeExtension}]},
			If[FileType[nvcc] === File,
				QuoteFile[nvcc],
				With[{pths = FileNames["nvcc*", installation, 2]},
					If[pths =!= {},
						First[pths],
						Throw[Message[errMsgHd::instl, installation, $ThisDriver]; $Failed]
					]
				]
			]
		]
	]
ResolveCUDALibraryPath[errMsgHd_, Automatic, targetSystemID_] :=
	ResolveCUDALibraryPath[errMsgHd, NVCCCompiler["ResolveInstallation"][Automatic], targetSystemID]
ResolveCUDALibraryPath[errMsgHd_, installation_String, targetSystemID_] :=
	QuoteFile[
		FileNameJoin[{ParentDirectory[installation], Switch[targetSystemID,
					"Windows" | "Linux" | "Linux-x86" | "MacOSX-x86" | "MacOSX-x86-64", "lib",
					"Windows-x86-64" | "Linux-x86-64", FileNameJoin[{"lib","x64"}],
					_, Message[errMsgHd::invsys]; Throw[$Failed]
		]}]
	]

ResolveWGLLibraryPath[errMsgHd_, installation_String, targetSystemID_] :=
	If[installation === $CUDAResourcesPath,
		QuoteFile[FileNameJoin[{ParentDirectory[installation], "LibraryResources", targetSystemID}]],
		$Failed
	]

GetTypeCheckedOption[x___] := CCompilerDriver`CCompilerDriverBase`Private`GetTypeCheckedOption[x]

GetOption[$ThisDriver, errMsgHd_Symbol, opts_List, n:"LinkerOptions"] :=
	With[{xslo = GetTypeCheckedOption[$ThisDriver, opts, n, _String | {_String ...}, Message[errMsgHd::compopt, "SystemLinkerOptions", #]&],
		  xlo = GetTypeCheckedOption[$ThisDriver, opts, n, _String | {_String ...}, Message[errMsgHd::lnkopt, n, #]&],
		  createCubinQ = getAndCheckOption[errMsgHd, opts, "CreateCUBIN"],
		  createPTXQ = getAndCheckOption[errMsgHd, opts, "CreatePTX"],
		  nvccInstallation = ParentDirectory[NVCCCompiler["ResolveInstallation"][getOption[errMsgHd, opts, "CompilerInstallation"]]]},
		Flatten[
			Join[
				{
					Switch[xslo,
						Automatic,
							If[createCubinQ || createPTXQ,
								"",
								Switch[$OperatingSystem,
									"MacOSX",
										formatLinkerOptions[{"-rpath," <> FileNameJoin[{nvccInstallation, "lib"}], "-rpath,/usr/local/cuda/lib", "-framework,Foundation"}],
									_,
										""
								]
							],
						"" | None,
							"",
						_String,
							formatLinkerOptions[{xslo}],
						_List,
							formatLinkerOptions[xslo],
						_,
							Message[errMsgHd::compopt, "SystemLinkerOptions", xslo];
							Throw[$Failed]
					]
				},
				{
					formatLinkerOptions[Flatten[{xlo}]]
				}
			]
		]
	]

formatLinkerOptions[opts_List] :=
	formatLinkerOptions /@ RemoveBlanks[opts]
formatLinkerOptions[opt_String] /; StringTrim[opt] =!= "" :=
	" --linker-options " <> QuoteFile[opt]
formatLinkerOptions[___] := ""

getAndCheckOption[errMsgHd_, opts_, n:"XCompileOptions"] :=
	With[{xco = getOption[errMsgHd, opts, n],
		  createCubinQ = getAndCheckOption[errMsgHd, opts, "CreateCUBIN"],
		  createPTXQ = getAndCheckOption[errMsgHd, opts, "CreatePTX"],
		  xcompilerInstallation = getAndCheckOption[errMsgHd, opts, "XCompilerInstallation"],
		  targetID = getAndCheckOption[errMsgHd, opts, "TargetSystemID"]},
		Switch[xco,
			Automatic,
				If[createCubinQ || createPTXQ,
					"",
					If[$OperatingSystem === "Unix" || $OperatingSystem == "MacOSX",
						If[FileExistsQ[FileNameJoin[{xcompilerInstallation, "gcc"}]],
							StringJoin[
								formatXCompileOptions["-fPIC"],
								With[{cmd = "!" <> FileNameJoin[{xcompilerInstallation, "gcc"}] <> " --version"},
									If[StringQ[cmd] && StringTrim[cmd] =!= "" && StringMatchQ[First[ReadList[cmd, String]], __ ~~ "4.4." ~~ __],
										formatXCompileOptions[{"-fno-inline", "-fno-strict-aliasing"}],
										""
									]
								],
								If[targetID === "MacOSX-x86" || targetID === "Linux",
									formatXCompileOptions["-malign-double"],
									""
								]
							],
							""
						],
						If[getOption[errMsgHd, opts, "Debug"] === True,
							formatXCompileOptions[{"/MTd"}],
							formatXCompileOptions[{"/MT"}]
						]
					]
				],
			"" | None,
				"",
			_String,
				formatXCompileOptions[{xco}],
			_List,
				formatXCompileOptions[xco],
			_,
				Message[errMsgHd::invxcomp, xco];
				Throw[$Failed]
		]
	]

formatXCompileOptions[opts_List] :=
	formatXCompileOptions /@ RemoveBlanks[opts]
formatXCompileOptions[opt_String] /; StringTrim[opt] =!= "" :=
	" --compiler-options " <> StringTrim[opt]
formatXCompileOptions[___] := ""

computeCapabilities = 
<|
"Linux-x86-64"->{30, 35, 50, 52, 60, 61, 70, 75}
,
"MacOSX-x86-64"->{30, 35, 50, 52, 60, 61, 70, 75}
,
"Windows-x86-64"->{30, 35, 50, 52, 60, 61, 70, 75}
|>
nvccFlagCubinTemplate = StringTemplate[" -gencode=arch=compute_`1`,code=sm_`1`"]
createNvccFlagPTXTemplate = StringTemplate[" -gencode=arch=compute_`1`,code=compute_`1`"]
createNvccFlagsCubin = Function[{cmptCapabilities},Map[nvccFlagCubinTemplate,cmptCapabilities]]
createNvccFlags = 
Function[
	{cmptCapabilities}
	,
	StringJoin[
		createNvccFlagsCubin[cmptCapabilities],
		createNvccFlagPTXTemplate[Max[cmptCapabilities]]
	]
]

getSupportedArchFlags[systemID_:$SystemID] :=
Module[
	{supportedComputeCapabilities},
	supportedComputeCapabilities = Lookup[computeCapabilities, systemID];
	createNvccFlags[supportedComputeCapabilities]
]

ResolveCUDAArchitecture[errMsgHd_, targetID_, opts_] :=
	With[{arch = getAndCheckOption[errMsgHd, opts, "CUDAArchitecture"],
		  createCUBINQ = getAndCheckOption[errMsgHd, opts, "CreateCUBIN"],
		  createPTXQ = getAndCheckOption[errMsgHd, opts, "CreatePTX"]
		 },
		Switch[arch,
			Automatic,
				If[createCUBINQ || createPTXQ,
					" -arch=sm_50",
					
  					getSupportedArchFlags[$SystemID]
				],
			"" | {},
				"",
			_String,
				" -arch="<> arch,
			_List,
				" -arch=" <> StringJoin[Riffle[arch, ","]],
			_,
				Message[errMsgHd::invarch, arch];
				Throw[$Failed]
		]
	]

getAndCheckOption[errMsgHd_, opts_, n:"UnmangleCode"] :=
	getOption[errMsgHd, opts, n] /. {
		Automatic :> n /. Options[$ThisDriver],
		x:(True | False) :> x,
		other_ :> (Message[errMsgHd::invunmg, other]; Throw[$Failed])
	}

getAndCheckOption[errMsgHd_, opts_, n:"CreateCUBIN"] :=
	getOption[errMsgHd, opts, n] /. {
		Automatic :> n /. Options[$ThisDriver],
		x:(True | False) :> x,
		other_ :> (Message[errMsgHd::invcub, other]; Throw[$Failed])
	}

getAndCheckOption[errMsgHd_, opts_, n:"CreatePTX"] :=
	getOption[errMsgHd, opts, n] /. {
		Automatic :> n /. Options[$ThisDriver],
		x:(True | False) :> x,
		other_ :> (Message[errMsgHd::invptx, other]; Throw[$Failed])
	}

IncludeDirectoryExistsQ[errMsgHd_, dir_String] :=
	IncludeDirectoryExistsQ[errMsgHd, {dir}]
IncludeDirectoryExistsQ[errMsgHd_, dirs_List] :=
	If[Quiet[And@@(DirectoryQ /@ dirs)],
		dirs,
		Message[errMsgHd::nodir, dirs];
		Throw[$Failed]
	]

formatIncludeDirectory[inc_List, sysId_] :=
	formatIncludeDirectory[#, sysId]& /@ inc
formatIncludeDirectory[inc_String, sysId_] :=
	If[sysId === "MacOSX-x86-64" && StringMatchQ[inc, StartOfString ~~ "-F" ~~ ___],
		formatXCompileOptions[inc],
		inc
	]
GetOption[$ThisDriver, errMsgHd_Symbol, opts_List, "IncludePath", targetSystemID_] :=
	Module[{createCUBINQ = getAndCheckOption[errMsgHd, opts, "CreateCUBIN"],
		    createPTXQ = getAndCheckOption[errMsgHd, opts, "CreatePTX"],
		    systemIncludedirectories, includeDirectories},
		systemIncludedirectories = GetTypeCheckedOption[
			$ThisDriver,
			opts,
			"SystemIncludeDirectories",
			Automatic | _String | {_String ...},
			Message[errMsgHd::sysdirlist, "SystemIncludeDirectories", #]&
		];
		includeDirectories = GetTypeCheckedOption[
			$ThisDriver,
			opts,
			"IncludeDirectories",
			Automatic | _String | {_String ...},
			Message[errMsgHd::dirlist, "IncludeDirectories", #]&
		];
		If [createCUBINQ || createPTXQ,
			Flatten[Join[
				formatIncludeDirectory[IncludeDirectoryExistsQ[errMsgHd, #], targetSystemID]& /@
					ResolveIncludeDirs[If[systemIncludedirectories === Automatic, {}, systemIncludedirectories], includeDirectories, targetSystemID]
			]],
			Flatten[Join[
				formatIncludeDirectory[IncludeDirectoryExistsQ[errMsgHd, #], targetSystemID]& /@
					ResolveIncludeDirs[
						systemIncludedirectories,
						If[systemIncludedirectories === Automatic,
							Flatten[{
								$GPUToolsIncludes,
								With[{installation = ParentDirectory[NVCCCompiler["ResolveInstallation"][getOption[errMsgHd, opts, "CompilerInstallation"]]]},
									FileNameJoin[{installation, "include"}]
								],
								includeDirectories
							}],
							includeDirectories
						],
						targetSystemID,Automatic
					]
			]]
		]
	]

$GPUToolsIncludes = FileNameJoin[{Lookup[PacletInformation["GPUTools"],"Location"],"Includes"}]

GetOption[$ThisDriver, errMsgHd_Symbol, opts_List, n:"SystemLibraries",  targetSystemID_, includeMathlinkQ_] := (
	getOption[errMsgHd, opts, n] /. {
		Automatic :>
			Join[
				Flatten[{ResolveLibraries[CCompilerDriver`CCompilerDriverBase`BaseDriver, Automatic, targetSystemID, includeMathlinkQ,Automatic]}],
				If[$OperatingSystem === "Windows", {"cuda"}, {}]
			],
		libs_String :> {libs},
		libs_List :> libs,
		other_ :> (Message[errMsgHd::sysfilelist, "SystemLibraries", other]; Throw[$Failed])
	}
)

getAndCheckOption[errMsgHd_, opts_, n:"SystemLibraryDirectories"] := getAndCheckOption[errMsgHd, opts, n] =
	Module[{libPath = getOption[errMsgHd, opts, n],
		    cmpinst = getAndCheckOption[errMsgHd, opts, "CompilerInstallation"],
		    sysdir, possibleDirs},
		If[libPath === Automatic,
			With[{nvccInstallation = DirectoryName[NVCCCompiler["ResolveInstallation"][cmpinst]],
				 targetSystemID = getAndCheckOption[errMsgHd, opts, "TargetSystemID"]},
				sysdir = nvccInstallation;
				possibleDirs = FileNames[{"*cuda*so", "*cuda*dylib", "*cuda*lib"}, sysdir, Infinity, IgnoreCase -> True];
				possibleDirs = Which[
					Bits[targetSystemID] === 32 || targetSystemID === "MacOSX-x86-64",
						possibleDirs,
					targetSystemID === "Linux-x86-64",
						Select[possibleDirs, StringMatchQ[#, ___ ~~ "lib64" ~~ ___]&],
					targetSystemID === "Windows-x86-64",
						Select[possibleDirs, StringMatchQ[#, ___ ~~ "x64" ~~ ___]&]
				];
				If[possibleDirs == {},
					Throw[Message[errMsgHd::sysdirs, libPath]; $Failed],
					DirectoryName[First[possibleDirs]]
				]
			],
			libPath
		]
	]

getAndCheckOption[errMsgHd_, opts_, n_] :=
	getOption[errMsgHd, opts, n]

getOption[errMsgHd_, opts_, n_] := GetOption[$ThisDriver, opts, n]

compilerBitnessOption[target_] := compilerBitnessOption[Bits[target]]

compilerBitnessOption[b:(32|64)] := If[$SystemID==="MacOSX-x86-64","","-m"<>ToString[b]]

$CUDAResourcesPath := $CUDAResourcesPath =
	Module[{paclets = PacletFind["CUDAResources"]},
		If[paclets === {},
			{},
			PacletResource[First[paclets], "CUDAToolkit"]
		]
	]

getBinDir[id_] := "bin" <> If[id === "Windows-x86-64", "64", ""]

NVCCCompiler["ResolveInstallation"][Automatic] :=
	Module[{dirs, candidates},
        dirs = Flatten[{
	            If[$CUDAResourcesPath === {},
	            	{},
	            	$CUDAResourcesPath
	            ],
            	Select[Flatten[{
			    	If[$OperatingSystem === "Windows", {}, StringSplit[Environment["PATH"], $PlatformPathSeparator]],
	            	If[$OperatingSystem === "Windows",
	            		{
	            			cudaEnvironmentPaths[],
	            			If[#=== $Failed || Environment[#] === $Failed, {}, {Environment[#]}]& /@ Flatten[{"CUDA_PATH", "CUDA_BIN_PATH"}],
	            			{"C:\\CUDA\\bin", "C:\\CUDA\\bin64"}
	            		},
	                	"/usr/local/cuda/bin"
	            	]
            	}], #=!=$Failed&]
        }];
        dirs = Select[dirs, !StringMatchQ[#, ___ ~~ "sbin" ~~ __]&];
        candidates = Quiet@Select[FileNames["*nvcc*", #, 2, IgnoreCase -> True]& /@ dirs, #=!={}&, 1];
		If[Length[candidates] > 0,
			NVCCCompiler["ResolveInstallation"][Automatic] = NVCCCompiler["ResolveInstallation"][DirectoryName[First[First[candidates]]]],
			$Failed
		]
	]

cudaEnvironmentPaths[] /; $OperatingSystem === "Windows" := cudaEnvironmentPaths[] =
	Quiet[
		Select[
			Flatten[
				{FileNameJoin[{#, "bin"}], FileNameJoin[{#, "bin64"}]}& /@
				Select[
					Flatten[
						Table[Environment["CUDA_PATH_V" <> ToString[ii] <> "_" <> ToString[jj]], {ii, 3, 4}, {jj, 2, 4}]
					],
					# =!= $Failed&
				]
			],
			DirectoryQ
		]
	]

NVCCCompiler["ResolveInstallation"][path_String] := NVCCCompiler["ResolveInstallation"][path] =
	If[TrueQ[Quiet[DirectoryQ[path]]],
		With[{files = FileNames["nvcc*", path, 2]},
			If[files === {},
				$Failed,
				DirectoryName[First[files]]
			]
		],
		$Failed
	]
NVCCCompiler["ResolveInstallation"][___] := $Failed

getAndCheckOption[errMsgHd_, opts_, n:"XCompilerInstallation"] :=
	Module[{targetSystemID, xcompilerInstallation},
		targetSystemID = GetOption[$ThisDriver, errMsgHd, opts, "TargetSystemID"];
		xcompilerInstallation = ResolveXCompilerInstallation[getOption[errMsgHd, opts, n], errMsgHd, targetSystemID] /. {
			pth_String :> pth,
			other_ :> (Message[errMsgHd::invxpth, other]; Throw[$Failed])
		};
		If[MemberQ[{"Windows", "Windows-x86-64"}, targetSystemID],
			xcompilerInstallation,
			
			If[ targetSystemID==="MacOSX-x86-64","/usr/bin",xcompilerInstallation]
		]
	]

CUDACCompilers[] :=
	With[{pths = Select[Flatten[{
			CCompilerDriver`GCCCompiler`GCCCompiler["ResolveInstallation"][Automatic],
			CCompilerDriver`VisualStudioCompiler`VisualStudioCompiler["ResolveInstallation"][{"2015"}],
			CCompilerDriver`VisualStudioCompiler`VisualStudioCompiler["ResolveInstallation"][{"2013"}],
			CCompilerDriver`VisualStudioCompiler`VisualStudioCompiler["ResolveInstallation"][{"2012"}]
			}],
			Quiet[DirectoryQ[#]]&
		], compilers = CCompilers[]},
		Join[Map[
			Function[{pth},
				Flatten[Select[compilers, ("CompilerInstallation" /. #) === pth&]]
			],
			pths
		]]
	]

ResolveXCompilerInstallation[Automatic] :=
	ResolveXCompilerInstallation[Automatic, NVCCCompiler, $SystemID]

ResolveXCompilerInstallation[path_, errMsgHd_, targetSystemId_] := path

ResolveXCompilerInstallation[Automatic, errMsgHd_, targetSystemId_String] /; $OperatingSystem === "Unix" || $OperatingSystem === "MacOSX" :=
	Module[{candidate},
		candidate = CCompilerDriver`GCCCompiler`GCCCompiler["ResolveInstallation"][Automatic];
		If[candidate === $Failed,
			Message[errMsgHd::noxcomp];
		 	Throw[$Failed],
		 	candidate
		 ]
	]

ResolveXCompilerInstallation[Automatic, errMsgHd_, targetSystemId_String] /; $OperatingSystem === "Windows" :=
	Module[{installation},
		installation = CCompilerDriver`VisualStudioCompiler`VisualStudioCompiler["ResolveInstallation"][Automatic];
		If[targetSystemId =!= "Windows-x86-64" &&
			!TrueQ[Directory === Quiet[FileType[FileNameJoin[{installation, "VC", "bin", CCompilerDriver`VisualStudioCompiler`Private`VCVarsArch[targetSystemId]}]]]],
			Message[errMsgHd::novsarch, installation, CCompilerDriver`VisualStudioCompiler`Private`VCVarsArch[targetSystemId]];
			Throw[$Failed]
		];
		installation
	]

validNVCCExecQ[installation_] :=
	NVCCCompiler["ResolveInstallation"][installation] =!= $Failed

NVCCCompiler["ValidInstallationQ"][installation_] :=
	TrueQ[validNVCCExecQ[installation]]


NVCCCompiler["ResolveCodeFile"][errMsgHd_, code0_String, workDir_String, funName_String, opts_] :=
	With[{codeFile = FileNameJoin[{workDir, funName <> ".cu"}],
		  code = StringTrim[code0],
		  unmangleQ = getAndCheckOption[errMsgHd, opts, "UnmangleCode"]},
		Export[codeFile,
			StringJoin[
				If[TrueQ@unmangleQ && !StringMatchQ[code, "extern" ~~ ___],
					"extern \"C\" {\n", ""
				],
				code,
				If[TrueQ@unmangleQ && !StringMatchQ[code, "extern" ~~ ___],
					"\n}\n\n",
					"\n\n"
				]
			], "Text"
		];
		CCompilerDriver`CCompilerDriverBase`MarkFileForCleanup[codeFile];
		{codeFile}
	]

NVCCCompiler["ResolveCodeFile"][errMsgHd_, srcFiles:{srcFile_String}, workDir_String, funName_String, opts_] :=
	If[getAndCheckOption[NVCCCompiler, opts, "UnmangleCode"],
		Module[{src = Import[srcFile, "Text"]},
			NVCCCompiler["ResolveCodeFile"][errMsgHd, src, workDir, funName, opts]
		],
		CCompilerDriver`CCompilerDriverBase`BaseDriver["ResolveCodeFile"][errMsgHd, srcFiles, workDir, funName, opts]
	]

NVCCCompiler["ResolveCodeFile"][errMsgHd_, srcFiles:{_String ..}, workDir_String, funName_String, opts_] :=
	If[getAndCheckOption[NVCCCompiler, opts, "UnmangleCode"],
		Message[errMsgHd::unmfiles, srcFiles];
		Throw[$Failed],
		CCompilerDriver`CCompilerDriverBase`BaseDriver["ResolveCodeFile"][errMsgHd, srcFiles, workDir, funName, opts]
	]

RemoveBlanks[x_List] :=
	Select[x, # =!= Null && StringTrim[#] =!= ""&]

formatLibraries[errMsgHd_, libs_] :=
	If[ListQ[libs] && MemberQ[libs, Except[_String]],
		Message[errMsgHd::sysfilelist, "SystemLibraries", libs];
		Throw[$Failed],
		Riffle[formatLibrary /@ RemoveBlanks[libs], " "]
	]

formatLibraries[errMsgHd_, libs1_List, libs2_List] :=
	formatLibraries[errMsgHd, RemoveBlanks[Join[libs1, libs2]]]

formatLibrary[lib_] /; $OperatingSystem === "Windows" :=
	CCompilerDriver`VisualStudioCompiler`Private`formatLibrary[lib]

LibraryPathQ[lib_] /; $OperatingSystem === "Windows" :=
	CCompilerDriver`VisualStudioCompiler`Private`LibraryPathQ[lib]

formatLibrary[lib_] /; $OperatingSystem === "Unix" || $OperatingSystem === "MacOSX" :=
	CCompilerDriver`GCCCompiler`Private`formatLibrary[lib, True,$SystemID,Automatic]

LibraryPathQ[lib_] /; $OperatingSystem === "Unix" || $OperatingSystem === "MacOSX" :=
	CCompilerDriver`GCCCompiler`Private`LibraryPathQ[lib]


NVCCCompiler[method_][args___] :=
	CCompilerDriver`CCompilerDriverBase`BaseDriver[method][args]

ResolveNVCCVersion[pth_] :=
	Module[{strVer, out},
		If[pth === $Failed || !StringQ[pth] || !FileExistsQ[pth],
			Return[$Failed]
		];
		out = Import["!" <> pth <> " --version 2>&1", "Text"];
		strVer = StringReplace[out, ___ ~~ "release " ~~ x___ ~~ ", V" ~~ ___ -> x];
		ToExpression[strVer]
	]

CCompilerRegister[$ThisDriver]

End[] (* `Private` *)

EndPackage[]
