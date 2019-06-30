

BeginPackage["OpenCLLink`Experimental`", {"OpenCLLink`"}]

OpenCLSymbolicCGenerate::usage = "OpenCLSymbolicCGenerate[name, prog, {in,...}, {out,...}] generate symbolic C representation of OpenCL interface code."

OpenCLCodeStringGenerate::usage = "OpenCLSymbolicCGenerate[name, prog, {in,...}, {out,...}] generate C source string of OpenCL interface code."

OpenCLCodeGenerate::usage = "OpenCLSymbolicCGenerate[name, prog, {in,...}, {out,...}] generate C source file of OpenCL interface code."

OpenCLLibraryGenerate::usage = "OpenCLSymbolicCGenerate[name, prog, {in,...}, {out,...}] generate Wolfram LibraryLink library of OpenCL interface code."

OpenCLLibraryFunctionGenerate::usage = "OpenCLSymbolicCGenerate[name, prog, {in,...}, {out,...}] generate Wolfram LibraryLink function of OpenCL interface code."

OpenCLLink`Internal`$OpenCLSDKPath::usage = "OpenCLLink`Internal`$OpenCLSDKPath  path to the OpenCL SDK."

OpenCLLink`Internal`$OpenCLSDKBinPath::usage = "OpenCLLink`Internal`$OpenCLSDKBinPath  path to the OpenCL SDK bin directory."

OpenCLLink`Internal`$OpenCLLinkLibraryPath::usage = "OpenCLLink`Internal`$OpenCLSDKLibraryPath path to the OpenCL library include directory."

OpenCLLink`Internal`$OpenCLLinkIncludePath::usage = "OpenCLLink`Internal`$OpenCLSDKIncludePath path to the OpenCL header include directory."


Begin["`Private`"]

Needs["GPUTools`"]
Get["GPUTools`CodeGenerator`"]
Get["GPUTools`CompileCodeGenerator`"]
Get["CUDALink`NVCCCompiler`"]

ClearAll[OpenCLSymbolicCGenerate, OpenCLCodeStringGenerate, OpenCLCodeGenerate, OpenCLLibraryGenerate, OpenCLLibraryFunctionGenerate]


Unprotect[Compile]
Compile[args___, CompilationTarget->"OpenCL", rest___] :=
  OpenCLCompile[Compile[args, rest], "cf"<>ToString[Hash[{args} ~Join~ DateList[]]]]
Protect[Compile]

OpenCLCompile[cf_, name_, opts___] :=
  Module[{dll, types = Catch[toGPUFunctionLoadInputs[Compile, cf]]},
  	If[types === $Failed,
  		$Failed,
	  	dll = OpenCLLibraryGenerate[cf, name, opts];
	  	OpenCLFunctionLoad[{dll}, name, types, 256]
  	]
  ]



nonOptionPattern = PatternSequence[___, Except[_?OptionQ]]

(* ::Section:: *)
(* Code Generation *)


Options[OpenCLLibraryFunctionGenerate] := Options[OpenCLLibraryGenerate]

OpenCLLibraryGenerate[cf_CompiledFunction, args___, opts:OptionsPattern[]] :=
	Module[{res},
		res = If[OpenCLQ[],
			iGPUCompileLibraryGenerate[cf, args,
				"Compiler" -> Automatic,
				"SystemDefines" -> Flatten[{
					"CONFIG_USE_OPENCL"->1,
					If[TrueQ[OpenCLLink`Internal`SupportsDoublePrecisionQ[$OpenCLPlatform, $OpenCLDevice]],
						"CONFIG_USE_DOUBLE_PRECISION"->1,
						{}
					]
				}],
				"SystemIncludeDirectories" -> {
					CUDALink`NVCCCompiler`Private`$GPUToolsIncludes,
					FileNameJoin[{$InstallationDirectory, "SystemFiles", "IncludeFiles", "C"}],
					OpenCLLink`Internal`$OpenCLLinkIncludePath
				}, 
				"SystemLibraryDirectories" -> {
					OpenCLLink`Internal`$OpenCLLinkLibraryPath
				},
				"SystemLibraries" -> {
					StringReplace[
						If[TrueQ[OpenCLLink`Internal`SupportsDoublePrecisionQ[$OpenCLPlatform, $OpenCLDevice]],
							OpenCLLink`Private`$Library["Double"],
							OpenCLLink`Private`$Library["Single"]
						], ".dll" -> ".lib"
					]
				},
				"APITarget" -> "OpenCL",
				opts
			],
			Message[OpenCLLibraryGenerate::noocl];
			$Failed
		];
		res /; (res =!= $Failed && FileExistsQ[res])
	]
	
OpenCLLibraryFunctionGenerate[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, OpenCLLibraryFunctionGenerate], HoldFirst]) :=
	Module[{res},
		res = If[OpenCLLink`Internal`$OpenCLSDKPath === $Failed || MemberQ[OpenCLLink`Internal`$OpenCLSDKPath, $Failed],
			Message[OpenCLLibraryFunctionGenerate::nooclsdk];
			$Failed,
			True
		];
		res = Catch[iGPULibraryFunctionGenerate[
			OpenCLLibraryFunctionGenerate,
			args,
			"SystemIncludeDirectories" -> OptionValue[OpenCLLibraryFunctionGenerate, {opts}, "SystemIncludeDirectories"],
			"SystemLibraryDirectories" -> OptionValue[OpenCLLibraryFunctionGenerate, {opts}, "SystemLibraryDirectories"],
			"SystemLibraries" -> OptionValue[OpenCLLibraryFunctionGenerate, {opts}, "SystemLibraries"],
			"SystemCompileOptions" -> OptionValue[OpenCLLibraryFunctionGenerate, {opts}, "SystemCompileOptions"],
			"APITarget"->"WGLOpenCL",
			opts
		]] /; res =!= $Failed;
		res /; res =!= $Failed
	]
	
ResolveCCompilerSystemLibraries[] :=
	If[Quiet[DefaultCCompiler[]] === $Failed,
		{},
		CCompilerDriver`CCompilerDriverBase`ResolveLibraries[DefaultCCompiler[], {"OpenCL"}, $SystemID, DefaultCCompiler[]["LinkWithMathLink"][CreateLibrary]]
	]
	
Options[OpenCLLibraryGenerate] := Quiet[Sort@Flatten[List[
		Catch@FilterRules[
			Options[iGPULibraryGenerate],
			Except["APITarget" | "CUDAArchitecture" | "SystemIncludeDirectories" | "SystemLibraries" | "SystemCompileOptions" | "SystemLibraryDirectories" | "Device" | "Platform"]
		],
		"Platform" :> $OpenCLPlatform,
		"Device" :> $OpenCLDevice,
		"SystemIncludeDirectories" -> Catch@If[OpenCLLink`Internal`$OpenCLSDKIncludePath === $Failed,
			"SystemIncludeDirectories" /. Options[iGPULibraryGenerate],
			CCompilerDriver`CCompilerDriverBase`ResolveIncludeDirs[Automatic, OpenCLLink`Internal`$OpenCLSDKIncludePath, $SystemID]
		],
		"SystemLibraryDirectories" -> Catch@If[OpenCLLink`Internal`$OpenCLSDKLibraryPath === $Failed,
			"SystemLibraryDirectories" /. Options[iGPULibraryGenerate],
			Flatten[Join[
				List[CCompilerDriver`CCompilerDriverBase`Private`ResolveLibraryDirectories[Automatic, $SystemID]],
				List[OpenCLLink`Internal`$OpenCLSDKLibraryPath]
			]]
		],
		"SystemLibraries" -> Catch@
			ResolveCCompilerSystemLibraries[],
		"SystemCompileOptions" -> Catch@
			Switch[$SystemID,
				"MacOSX-x86",
					Flatten[{"SystemCompileOptions" /. Options[iGPULibraryGenerate], "-malign-double", "-framework OpenCL"}],
				"MacOSX-x86-64",
					Flatten[{"SystemCompileOptions" /. Options[iGPULibraryGenerate], "-framework OpenCL"}],
				"Linux",
					Flatten[{"SystemCompileOptions" /. Options[iGPULibraryGenerate], "-malign-double"}],
				_,
					"SystemCompileOptions" /. Options[iGPULibraryGenerate]
			]
]]]

OpenCLLibraryGenerate[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, OpenCLLibraryGenerate], HoldFirst]) :=
	Module[{res},
		res = If[OpenCLLink`Internal`$OpenCLSDKPath === $Failed || MemberQ[OpenCLLink`Internal`$OpenCLSDKPath, $Failed],
			Message[OpenCLLibraryGenerate::nooclsdk];
			$Failed,
			True
		];
		(
			res = Catch[iGPULibraryGenerate[
				OpenCLLibraryGenerate,
				args,
				"APITarget"->"WGLOpenCL",
				"SystemIncludeDirectories" -> OptionValue[OpenCLLibraryGenerate, {opts}, "SystemIncludeDirectories"],
				"SystemLibraryDirectories" -> OptionValue[OpenCLLibraryGenerate, {opts}, "SystemLibraryDirectories"],
				"SystemLibraries" -> OptionValue[OpenCLLibraryGenerate, {opts}, "SystemLibraries"],
				"SystemCompileOptions" -> OptionValue[OpenCLLibraryGenerate, {opts}, "SystemCompileOptions"],
				opts
			]]
		) /; res =!= $Failed;
		res /; res =!= $Failed
	]

Options[OpenCLCodeGenerate] := Options[OpenCLCodeStringGenerate]

OpenCLCodeGenerate[cf_CompiledFunction, args___, opts:OptionsPattern[]] :=
	With[{res = iGPUCompileCCodeGenerate[cf, args, "APITarget" -> "OpenCL", opts]},
		Print[res];
		res /; (res =!= $Failed && FileExistsQ[res])
	]
OpenCLCodeGenerate[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, OpenCLCodeGenerate], HoldFirst]) :=
	Module[{res = Catch[iGPUCodeGenerate[OpenCLCodeGenerate, args, "APITarget"->"WGLOpenCL", opts]]},
		res /; res =!= $Failed
	]
	
Options[OpenCLCodeStringGenerate] = Sort@Flatten[List[
	FilterRules[
		Options[iGPUCodeStringGenerate],
		Except[
			"APITarget" | "CUDAArchitecture" | "Device" | "Platform"
		]
	],
	"Platform" :> $OpenCLPlatform,
	"Device" :> $OpenCLDevice
]]


OpenCLCodeStringGenerate[cf_CompiledFunction, args___, opts:OptionsPattern[]] :=
	With[{res = iGPUCompileCCodeStringGenerate[cf, args, "APITarget" -> "OpenCL", opts]},
		res /; res =!= $Failed
	]

OpenCLCodeStringGenerate[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, OpenCLCodeStringGenerate], HoldFirst]) :=
	Module[{res = Catch[iGPUCodeStringGenerate[OpenCLCodeStringGenerate, args, "APITarget"->"WGLOpenCL", opts]]},
		res /; res =!= $Failed
	]	

Options[OpenCLSymbolicCGenerate] = Sort@Flatten[List[
	FilterRules[
		$GPUCodeGenerateOptions,
		Except[
			"APITarget" | "CUDAArchitecture" | "Device" | "Platform"
		]
	],
	"Platform" :> $OpenCLPlatform,
	"Device" :> $OpenCLDevice
]]

OpenCLSymbolicCGenerate[cf_CompiledFunction, args___, opts:OptionsPattern[]] :=
	With[{res = iGPUCompileSymbolicCGenerate[cf, args, "APITarget" -> "OpenCL", opts]},
		res /; (res =!= $Failed && FileExistsQ[res])
	]

OpenCLSymbolicCGenerate[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, OpenCLSymbolicCGenerate], HoldFirst]) :=
	Module[{res = Catch[iGPUSymbolicCGenerate[OpenCLSymbolicCGenerate, args, "APITarget"->"WGLOpenCL", opts]]},
		res /; res =!= $Failed
	] 

OpenCLLink`Internal`$OpenCLSDKPath := OpenCLLink`Internal`$OpenCLSDKPath =
	Quiet[Check[
		Which[
			GPUTools`Internal`UsingNVIDIAQ,
				Module[{paclet, dir},
					paclet = PacletFind["CUDAResources"];
					dir = If[paclet === {},
						Automatic,
						PacletResource[First@paclet, "CUDAToolkit"]
					];
					ParentDirectory[NVCCCompiler["ResolveInstallation"][dir]]
				],
			$OperatingSystem == "MacOSX",
				ParentDirectory[GPUTools`Internal`$OpenCLLibraryDirectory],
			True,
				GPUTools`Internal`$ATISTREAMSDKROOT
		], $Failed
	]]

OpenCLLink`Internal`$OpenCLSDKBinPath :=
	If[OpenCLLink`Internal`$OpenCLSDKPath === $Failed,
		$Failed,
		Switch[$OperatingSystem,
			"Windows" | "Unix",
				If[GPUTools`Internal`UsingNVIDIAQ,
					FileNameJoin[{
						OpenCLLink`Internal`$OpenCLSDKPath,
						If[$SystemID === "Windows",
							"bin",
							"bin64"
						]
					}],
					FileNameJoin[{
						OpenCLLink`Internal`$OpenCLSDKPath,
						"bin",
						If[$ProcessorType === "x86",
							"x86",
							"x86_64"
						]
					}]
				],
			"MacOSX",
				OpenCLLink`Internal`$OpenCLSDKPath
		]
	]

OpenCLLink`Internal`$OpenCLLinkLibraryPath :=
	Quiet[Check[
		If[OpenCLLink`Internal`$OpenCLSDKPath === $Failed,
			$Failed,
			Switch[$OperatingSystem,
				"Windows" | "Unix",
					If[GPUTools`Internal`UsingNVIDIAQ,
						FileNameJoin[{
							OpenCLLink`Internal`$OpenCLSDKPath,
							If[$ProcessorType === "x86",
								"lib",
								"lib64"
							]
						}],
						FileNameJoin[{
							OpenCLLink`Internal`$OpenCLSDKPath,
							"lib",
							If[$ProcessorType === "x86",
								"x86",
								"x86_64"
							]
						}]
					],
				"MacOSX",
					FileNameJoin[{OpenCLLink`Internal`$OpenCLSDKPath, "Libraries"}]
			]
		], $Failed
	]]

OpenCLLink`Internal`$OpenCLLinkIncludePath :=
	If[OpenCLLink`Internal`$OpenCLSDKPath === $Failed,
		$Failed,
		Switch[$OperatingSystem,
			"Windows" | "Unix",
				FileNameJoin[{OpenCLLink`Internal`$OpenCLSDKPath, "include"}],
			"MacOSX",
				FileNameJoin[{OpenCLLink`Internal`$OpenCLSDKPath, "Headers"}]
		]
	]

(******************************************************************************)
(******************************************************************************)
(******************************************************************************)


GPUTools`Utilities`DefineMessage[
	{OpenCLSymbolicCGenerate, OpenCLCodeStringGenerate, OpenCLCodeGenerate, OpenCLLibraryGenerate, OpenCLLibraryFunctionGenerate},
	{
		{"invtype", iGPUSymbolicCGenerate::invtype},
		{"invfunn", iGPUSymbolicCGenerate::invfunn},
		{"invapi", iGPUSymbolicCGenerate::invapi},
		{"invcdtrgt", iGPUSymbolicCGenerate::invcdtrgt},
		{"nonvcc", iGPUSymbolicCGenerate::nonvcc},
		{"invkcopt", iGPUSymbolicCGenerate::invkcopt},
		{"invprog", iGPUSymbolicCGenerate::invprog},
		{"invkrnnam", iGPUSymbolicCGenerate::invkrnnam},
		{"nooclsdk", "An OpenCL SDK was not located and is required for compilation."},
		{"argtu", "`1` was called with invalid arguments."}
	}
]

End[] (* End Private Context *)
EndPackage[] (* Experimental *)

