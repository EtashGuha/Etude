

BeginPackage["CUDALink`Experimental`", {"CUDALink`"}]


CUDASymbolicCGenerate::usage = "CUDASymbolicCGenerate[name, prog, {in,...}, {out,...}] generate symbolic C representation of CUDA interface code."

CUDACodeStringGenerate::usage = "CUDACodeStringGenerate[name, prog, {in,...}, {out,...}] generate C source string of CUDA interface code."

CUDACodeGenerate::usage = "CUDACodeGenerate[name, prog, {in,...}, {out,...}] generate C source file of CUDA interface code."

CUDALibraryGenerate::usage = "CUDALibraryGenerate[name, prog, {in,...}, {out,...}] generate Wolfram LibraryLink library of CUDA interface code."

CUDALibraryFunctionGenerate::usage = "CUDASymbolicCGenerate[name, prog, {in,...}, {out,...}] generate Wolfram LibraryLink function of CUDA interface code."


Begin["`Private`"]

Needs["GPUTools`"]
Needs["CUDALink`NVCCCompiler`"]
Get["GPUTools`CodeGenerator`"]
Get["GPUTools`CompileCodeGenerator`"]

ClearAll[CUDASymbolicCGenerate, CUDACodeStringGenerate, CUDACodeGenerate, CUDALibraryGenerate, CUDALibraryFunctionGenerate];

Unprotect[Compile]
Compile[args___, CompilationTarget->"CUDA", rest___] :=
  CUDACompile[Compile[args, rest], "cf"<>ToString[Hash[{args} ~Join~ DateList[]]]]
Protect[Compile]

CUDACompile[cf_, name_, opts___] :=
  Module[{dll, types = Catch[toGPUFunctionLoadInputs[Compile, cf]]},
  	If[types === $Failed,
  		$Failed,
	  	dll = CUDALibraryGenerate[cf, name, opts];
	  	CUDAFunctionLoad[{dll}, name, types, 256]
  	]
  ]


nonOptionPattern = PatternSequence[___, Except[_?OptionQ]]

(* ::Section:: *)
(* Code Generation *)
Options[CUDALibraryFunctionGenerate] := Options[CUDALibraryGenerate]
CUDALibraryFunctionGenerate[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDALibraryFunctionGenerate], HoldFirst]) :=
	With[{res = iGPULibraryFunctionGenerate[CUDALibraryFunctionGenerate, args, "APITarget" -> "WGLCUDA", opts]},
		res /; Quiet[Head[res]] === LibraryFunction
	]

Options[CUDALibraryGenerate] := Sort@Flatten[List[
	FilterRules[
		Options[iGPULibraryGenerate],
		Except[
			"Platform" | "Device" | "TargetDirectory" | "APITarget"
		]
	],
	"TargetDirectory" -> $CUDALinkBinaryDirectory,
	"Device" -> Automatic
]]

CUDALibraryGenerate[cf_CompiledFunction, args___, opts:OptionsPattern[]] :=
	Module[{res},
		res = If[CUDAQ[],
			iGPUCompileLibraryGenerate[cf, args,
				"Compiler" -> Automatic,
				"SystemDefines" -> Flatten[{
					"CONFIG_USE_CUDA"->1,
					If[TrueQ[CUDALink`Internal`SupportsDoublePrecisionQ[$CUDADevice]],
						"CONFIG_USE_DOUBLE_PRECISION"->1,
						{}
					]
				}],
				"SystemIncludeDirectories" -> {
					CUDALink`NVCCCompiler`Private`$GPUToolsIncludes,
					FileNameJoin[{$InstallationDirectory, "SystemFiles", "IncludeFiles", "C"}],
					DirectoryName[First[FileNames["cuda.h", DirectoryName[First@NVCCCompiler["Installations"][]], Infinity]]]
				}, 
				"SystemLibraryDirectories" -> {
					CUDALink`Internal`$CUDALinkLibraryPath
				},
				"SystemLibraries" -> {
					StringReplace[
						If[TrueQ[CUDALink`Internal`SupportsDoublePrecisionQ[$CUDADevice]],
							CUDALink`Private`$Library["Double"],
							CUDALink`Private`$Library["Single"]
						], ".dll" -> ".lib"
					]
				},
				"APITarget" -> "CUDA",
				opts
			],
			Message[CUDALibraryGenerate::nocuda];
			$Failed
		];
		res /; (res =!= $Failed && FileExistsQ[res])
	]
	
CUDALibraryGenerate[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDALibraryGenerate], HoldFirst]) :=
	Module[{res = iGPULibraryGenerate[CUDALibraryGenerate, args, "APITarget" -> "WGLCUDA", opts]},
		res /; (res =!= $Failed && FileExistsQ[res])
	]

Options[CUDACodeGenerate] := Options[CUDACodeStringGenerate]

CUDACodeGenerate[cf_CompiledFunction, args___, opts:OptionsPattern[]] :=
	With[{res = iGPUCompileCCodeGenerate[cf, args, "APITarget" -> "CUDA", opts]},
		res /; (res =!= $Failed && FileExistsQ[res])
	]
CUDACodeGenerate[args__, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDACodeGenerate], HoldFirst]) :=
	With[{res = Catch[iGPUCodeGenerate[CUDACodeGenerate, args, "APITarget" -> "WGLCUDA", opts]]},
		res /; FileExistsQ[res]
	]
	
Options[CUDACodeStringGenerate] := Sort@Flatten[List[
	FilterRules[
		Options[iGPUCodeStringGenerate],
		Except[
			"Device" | "Platform" | "Device" | "APITarget"
		]
	],
	"Device" -> Automatic
]]

CUDACodeStringGenerate[cf_CompiledFunction, args___, opts:OptionsPattern[]] :=
	With[{res = iGPUCompileCCodeStringGenerate[cf, args, "APITarget" -> "CUDA", opts]},
		res /; res =!= $Failed
	]
CUDACodeStringGenerate[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDACodeStringGenerate], HoldFirst]) :=
	With[{res = Catch[iGPUCodeStringGenerate[CUDACodeStringGenerate, args, "APITarget" -> "WGLCUDA", opts]]},
		res /; res =!= $Failed
	]	

Options[CUDASymbolicCGenerate] := Sort@Flatten[List[
	FilterRules[
		$GPUCodeGenerateOptions,
		Except[
			"Platform" | "Device" | "APITarget" | "Debug"
		]
	],
	"Device" -> Automatic
]]

CUDASymbolicCGenerate[cf_CompiledFunction, args___, opts:OptionsPattern[]] :=
	With[{res = iGPUCompileSymbolicCGenerate[cf, args, "APITarget" -> "CUDA", opts]},
		res /; (res =!= $Failed && FileExistsQ[res])
	]
CUDASymbolicCGenerate[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDASymbolicCGenerate], HoldFirst]) :=
	With[{res = Catch[iGPUSymbolicCGenerate[CUDASymbolicCGenerate, args, "APITarget" -> "WGLCUDA", opts]]},
		res /; res =!= $Failed
	]

(******************************************************************************)
(******************************************************************************)
(******************************************************************************)

End[] (* End Private Context *)
EndPackage[] (* Experimental *)

