(*****************************************************************************)
(* :Name: OpenCLLink.m *)
(* :Title: OpenCLLink *)
(* :Context: OpenCLLink` *)
(* :Author: Abdul Dakkak *)
(* :Summary: *)
(* :Sources:*)
(* :Copyright: 2010, Wolfram Research, Inc. *)
(* :Mathematica Version: 8.0 *)
(* :Keywords: OpenCLLink, GPU Computing, GPGPU *)
(* :Warnings: None *)
(* :Limitations: None *)
(* :Discussion: *)
(* :Requirements: None *)
(* :Examples: None *)
(*****************************************************************************)

BeginPackage["OpenCLLink`", {"CCompilerDriver`", "SymbolicC`"}]


(******************************************************************************)

(* ::Section:: *)
(* Symbols *)
OpenCLLink`Internal`$Symbols = {
	OpenCLLink,
	OpenCLQ,
	OpenCLInformation,
	OpenCLFunctionLoad,
	OpenCLMemoryLoad,
	OpenCLMemoryUnload,
	OpenCLMemoryGet,
	OpenCLFunctionInformation,
	OpenCLMemoryInformation,
	OpenCLMemoryCopyToHost,
	OpenCLMemoryCopyToDevice,
	OpenCLMemoryAllocate,
	OpenCLFunction,
	OpenCLProgram,
	OpenCLMemory,
	SymbolicOpenCLFunction,
	SymbolicOpenCLThreadIndex,
	SymbolicOpenCLBlockIndex,
	SymbolicOpenCLBlockDimension,
	SymbolicOpenCLCalculateKernelIndex,
	SymbolicOpenCLDeclareIndexBlock,
	SymbolicOpenCLKernelIndex,
	OpenCLFractalRender3D,
	OpenCLMersenneTwister,
	OpenCLImplicitRender3D
};

(******************************************************************************)

(* ::Section:: *)
(* Unprotect *)
Unprotect /@ OpenCLLink`Internal`$Symbols

(******************************************************************************)
(* ::Section:: *)
(* ClearAll *)
ClearAll /@ OpenCLLink`Internal`$Symbols

(******************************************************************************)
(* ::Section:: *)
(* Usages *)
$OpenCLPlatform::usage = "$OpenCLPlatform is OpenCL platform used for OpenCL computation."

$OpenCLDevice::usage = "$OpenCLDevice is OpenCL device used for OpenCL computation."

$OpenCLLinkPath::usage = "$OpenCLLinkPath is path to the OpenCLLink application."

$OpenCLSupportFilesPath::usage = "$OpenCLSupportFilesPath gives path to the OpenCLLink support files."

$OpenCLLinkLibraryPath::usage = "$OpenCLLinkLibraryPath is path to the OpenCLLink library files."

OpenCLLink::usage = "OpenCLLink allows users to program the GPU from within Mathematica."

OpenCLQ::usage = "OpenCLQ[] checks if OpenCLLink is supported on system."

OpenCLInformation::usage = "OpenCLInformation[] gives information on OpenCL platforms and devices detected."

OpenCLProgram::usage = "OpenCLProgram is an internal symbol used by OpenCLFunction"

OpenCLFunction::usage = "OpenCLFunction is handle to a function loaded using OpenCLFunctionLoad."

OpenCLMemoryLoad::usage = "OpenCLMemoryLoad[lst] loads a Mathematica expression into OpenCL memory."

OpenCLMemory::usage = "OpenCLMemory is handle to OpenCL memory."

OpenCLMemoryAllocate::usage = "OpenCLMemoryAllocate[type, dims] allocates memory and registers it with OpenCL memory manager."

OpenCLMemoryUnload::usage = "OpenCLMemoryUnload[mem] unloads memory from the OpenCLLink memory manager."

OpenCLMemoryGet::usage = "OpenCLMemoryGet[mem] gets memory from the OpenCLLink memory manager into Mathematica."

OpenCLMemoryInformation::usage = "OpenCLMemoryInformation[mem] gives information on the OpenCLMemory."

OpenCLMemoryCopyToHost::usage = "OpenCLMemoryCopyToHost[mem] force copy memory from GPU to CPU."

OpenCLMemoryCopyToDevice::usage = "OpenCLMemoryCopyToDevice[mem] force copy memory from CPU to GPU."

OpenCLFunctionLoad::usage = "OpenCLFunctionLoad[code, funName, {args, ...}, blockDim] loads OpenCL function from code."

OpenCLFunctionInformation::usage = "OpenCLFunctionInformation[OpenCLfun] gives information on a loaded OpenCL function."

SymbolicOpenCLFunction::usage = "SymbolicOpenCLFunction[name, args, body]  is a symbolic representation of a OpenCL function."

SymbolicOpenCLThreadIndex::usage = "SymbolicOpenCLThreadIndex[dim] is a symbolic representation of a OpenCL kernel thread index call."

SymbolicOpenCLBlockIndex::usage = "SymbolicOpenCLBlockIndex[dim] is a symbolic representation of a OpenCL kernel block index call."

SymbolicOpenCLBlockDimension::usage = "SymbolicOpenCLBlockDimension[dim] is a symbolic representation of a OpenCL kernel block dimension call."

SymbolicOpenCLCalculateKernelIndex::usage = "SymbolicOpenCLCalculateKernelIndex[dim] is a symbolic representation of a OpenCL kernel index calculation."

SymbolicOpenCLDeclareIndexBlock::usage = "SymbolicOpenCLDeclareIndexBlock[dim] is a symbolic representation of a OpenCL kernel index declaration."

SymbolicOpenCLKernelIndex::usage = "SymbolicOpenCLKernelIndex[dim] is a symbolic representation of a OpenCL kernel index call."

OpenCLFractalRender3D::usage = "OpenCLFractalRender3D[width, height] renders a mandelbulb with controls at resolution width x height."

OpenCLMersenneTwister::usage = "OpenCLMersenneTwister[n] generates n random numbers using parallel Mersenne Twisters."

OpenCLImplicitRender3D::usage = "OpenCLImplicitRender3D[width, height, eq, boundingrad] renders the isosurface eq(x,y,z) = 0 inside a sphere at the origin of radius boundingrad with resolution width x height."

OpenCLProfile::usage = "OpenCLProfile[args] runs the low level OpenCL profiler on the arguments."


Begin["`Private`"] 

Needs["ResourceLocator`"]
Needs["LibraryLink`"]
Needs["GPUTools`"]
Needs["CUDALink`NVCCCompiler`"]
Needs["GPUTools`SymbolicGPU`"]
Needs["GPUTools`CodeGenerator`"]


(******************************************************************************)

$ThisAPI = "OpenCL"

$ThisHead = OpenCLLink

ClearAll[$Symbols]
$Symbols = OpenCLLink`Internal`$Symbols

OpenCLLink`Internal`$Version = $APIVersion

(******************************************************************************)

$WGLInvalidTypes = {
	"Void",
	"Byte[3]",
	"UnsignedByte[3]",
	"Short[3]",
	"UnsignedShort[3]",
	"Integer[3]",
	"UnsignedInteger[3]",
	"Long[3]",
	"UnsignedLong[3]",
	"Float[3]",
	"Double[3]",
	"ComplexFloat",
	"ComplexDouble",
	Complex,
	"MatrixFloat",
	"MatrixTransposedFloat",
	"MatrixDouble",
	"MatrixTransposedDouble",
	"MatrixComplexFloat",
	"MatrixTransposedComplexFloat",
	"MatrixComplexDouble",
	"MatrixTransposedComplexDouble"
}

(******************************************************************************)

$ExtraLibraryFunctions[libPath_, singleOrDouble_:"Single"] := {}

(******************************************************************************)

$ThisFile = System`Private`$InputFileName
$ThisDirectory = DirectoryName[$ThisFile]

(******************************************************************************)

	
(******************************************************************************)


If[$OpenCLPlatform === Unevaluated[$OpenCLPlatform],
	$OpenCLPlatform = Automatic,
	$WGLPlatform = $OpenCLPlatform
]

If[$OpenCLDevice === Unevaluated[$OpenCLDevice],
	$OpenCLDevice = Automatic,
	$WGLDevice = $OpenCLDevice
]

$OpenCLLinkPath = DirectoryName[System`Private`$InputFileName];

$OpenCLLinkLibraryPath = FileNameJoin[{$OpenCLLinkPath, "LibraryResources", $SystemID}];
If[FreeQ[$LibraryPath, $OpenCLLinkLibraryPath],
	PrependTo[$LibraryPath, $OpenCLLinkLibraryPath]
];


$OpenCLSupportFilesPath = FileNameJoin[{$ThisDirectory, "SupportFiles"}]

(******************************************************************************)

Get["GPUTools`WGLPrivate`"]
	
(******************************************************************************)

Unprotect[Image]
OpenCLMemory /: Image[mem:OpenCLMemory[id_Integer, ___], opts___] :=
	With[{imgType = GetImageType[WGLMemoryGetType[mem]], data = OpenCLMemory[mem]},
		If[ImageQ[data],
			Image[data, opts],
			Image[data, imgType, opts]
		]
	] 
Protect[Image]

(******************************************************************************)

(* ::Section:: *)
(* Helper Methods *)
GetAndCheckOption[head_, n_, opts_] :=
	GetAndCheckOption[head, head, n, opts]

MersenneTwiserReadDataFile[fileName_String] :=
	Module[{data},
		Assert[FileExistsQ[fileName]];
  		data = BinaryReadList[fileName, "Integer32"];
  		Assert[Divisible[Length[data], 4]];
  		Transpose[Partition[data, 4]]
	]
	
(******************************************************************************)
(******************************************************************************)


SyntaxInformation[OpenCLQ] = {"ArgumentsPattern" -> {}}
OpenCLQ[___] ? (Function[{arg}, OptionsCheck[arg, OpenCLQ, 0, 0], HoldFirst]) :=
	If[Developer`$ProtectedMode,
		False,
		TrueQ[Quiet[Catch[
			With[{qry = PlatformCount[OpenCLQ]},
				If[Head[qry] === Rule,
					("Number Of Platforms" /. qry) >= 1,
					False
				]
			]
		]]]
	]

SyntaxInformation[OpenCLInformation] = {"ArgumentsPattern" -> {_., _., _.}}
OpenCLInformation[] :=
	With[{res = Catch[Check[WGLQuery[OpenCLInformation, OpenCLQ], $Failed]]},
		res /; (res =!= $Failed)
	]
OpenCLInformation[args:nonOptionPattern] ? (Function[{arg}, OptionsCheck[arg, OpenCLInformation, 1, 3], HoldFirst]) :=
	With[{res = Catch[Check[iOpenCLInformation[args], $Failed]]},
		res /; (res =!= $Failed)
	]
iOpenCLInformation[query:("CurrentMemoryUsage" | "MaximumUsableMemory")] /; ListQ[OpenCLInformation[]] :=
	Module[{res},
		res = OpenCLQ[];
		If[res,
			WGLQuery[OpenCLInformation, query],
			$Failed
		]
	]
iOpenCLInformation[platform_Integer] /; ListQ[OpenCLInformation[]] :=
	WGLQuery[OpenCLInformation, platform]
iOpenCLInformation[platform_Integer, prop_String] /; ListQ[OpenCLInformation[]] :=
	Module[{res = WGLQuery[OpenCLInformation, platform, prop]},
		res = If[res === $Failed || prop === "Devices",
			res,
			(prop /. res)
		];
		res
	]
iOpenCLInformation[platform_Integer, device_Integer] /; ListQ[OpenCLInformation[]] :=
	WGLQuery[OpenCLInformation, platform, device]
iOpenCLInformation[platform_Integer, device_Integer, prop_String] /; ListQ[OpenCLInformation[]] :=
	Module[{res = WGLQuery[OpenCLInformation, platform, device, prop]},
		If[res === $Failed,
			$Failed,
			(prop /. res)
		]
	]

iOpenCLInformation[args___] /; (Message[OpenCLInformation::args, {args}]; False) := $Failed

$WGLPlatformQueryList = {
	"Version" -> "Platform Version",
	"Name" -> "Platform Name",
	"Vendor" -> "Platform Vendor",
	"Extensions" -> {"Platform Extensions", SpaceSplit}
}

$WGLDeviceQueryList = {
	"Type" -> "Device Type",
	"Name" -> "Device Name",
	"Version" -> "Device Version",
	"Extensions" -> {"Device Extensions", SpaceSplit},
	"Driver Version",
	"Vendor" -> "Device Vendor",
	"Profile" -> "Device Profile",
	"Vendor ID" -> "Device Vendor ID",
	"Compute Units" -> "Maximum Compute Units",
	"Core Count",
	"Maximum Work Item Dimensions",
	"Maximum Work Item Sizes",
	"Maximum Work Group Size",
	"Preferred Vector Width Character",
	"Preferred Vector Width Short",
	"Preferred Vector Width Integer",
	"Preferred Vector Width Long",
	"Preferred Vector Width Float",
	"Preferred Vector Width Double",
	"Maximum Clock Frequency",
	"Address Bits",
	"Maximum Memory Allocation Size",
	"Image Support",
	"Maximum Read Image Arguments",
	"Maximum Write Image Arguments",
	"Maximum Image2D Width",
	"Maximum Image2D Height",
	"Maximum Image3D Width",
	"Maximum Image3D Height",
	"Maximum Image3D Depth",
	"Maximum Samplers",
	"Maximum Parameter Size",
	"Memory Base Address Align", 
	"Memory Data Type Align Size",
	"Floating Point Precision Configuration" -> {"Single Floating Point Precision Configuration", CommaSplit},
	"Global Memory Cache Type",
	"Global Memory Cache Line Size",
	"Global Memory Cache Size",
	"Global Memory Size",
	"Maximum Constant Buffer Size",
	"Maximum Constant Arguments",
	"Local Memory Type",
	"Local Memory Size",
	"Error Correction Support",
	"Profiling Timer Resolution",
	"Endian Little",
	"Available",
	"Compiler Available",
	"Execution Capabilities" -> {"Execution Capabilities", CommaSplit},
	"Command Queue Properties" -> {"Command Queue Properties", CommaSplit}
}

(******************************************************************************)
(******************************************************************************)

OpenCLInformation`SystemInformation[] :=
	If[OpenCLQ[],
		Quiet[Catch[
			Check[
				{
					"Driver Version" -> 
						If[$OperatingSystem === "MacOSX" || StringQ[GPUTools`Internal`$OpenCLDriverVersion],
							GPUTools`Internal`$OpenCLDriverVersion,
							LibraryVersionString[GPUTools`Internal`$OpenCLDriverVersion]
						],
					"Library Name" -> GPUTools`Internal`$OpenCLLibraryName,
					"Fastest Platform" -> OpenCLLink`Internal`FastestPlatform[],
					"Fastest Device" -> OpenCLLink`Internal`FastestDevice[],
					"Detailed Information" -> {
						"Driver Path" -> 
							Which[
								$OperatingSystem === "MacOSX",
									"",
								GPUTools`Internal`UsingATIQ,
									GPUTools`Internal`$ATIDriverPath,
								True,
									"Path" /. GPUTools`Internal`$NVIDIADriverLibraryVersion
							],
						"Library Path" -> GPUTools`Internal`$OpenCLLibraryPath,
						"OpenCL Provider" ->
							Which[
								$OperatingSystem === "MacOSX",
									"Apple",
								GPUTools`Internal`UsingATIQ,
									"ATI",
								True,
									"NVIDIA"
							] 
					}
				} ~Join~ (# -> OpenCLInformation`SystemInformation[#]& /@ Range[Catch[Length[OpenCLInformation[]]]]),
				$Failed
			]
		]]
		,
		$Failed
	]

OpenCLLink`Internal`FastestPlatformDevice[] := OpenCLLink`Internal`FastestPlatformDevice[] =
	Module[{devinfo =
            Flatten[
                Table[
                    Table[{Join[
                    		If[OpenCLInformation[platid, devid, "Type"] === "GPU",
                    			1000,
                    			1
                    		] * OpenCLInformation[platid, devid, "Maximum Clock Frequency"] * Total[OpenCLInformation[platid, devid, "Maximum Work Item Sizes"]] * OpenCLInformation[platid, devid, "Core Count"],
                            {platid, devid}
                        ]}, {devid, Length[OpenCLInformation[platid, "Devices"]]}
                    ], {platid, Length[OpenCLInformation[]]}
                ], 2
            ]
            },
		Last[Sort[devinfo]] // Last
	]

OpenCLLink`Internal`FastestPlatform[] := OpenCLLink`Internal`FastestPlatform[] =
    First[OpenCLLink`Internal`FastestPlatformDevice[]]	

OpenCLLink`Internal`FastestDevice[] := OpenCLLink`Internal`FastestDevice[] =
    Last[OpenCLLink`Internal`FastestPlatformDevice[]]	
    
OpenCLInformation`SystemInformation[platform_Integer?Positive] :=
	{
		"Name" -> OpenCLInformation[platform, "Name"],
		"Version" -> OpenCLInformation[platform, "Version"],
		"Vendor" -> OpenCLInformation[platform, "Vendor"],
		"Detailed Information" ->
			Select[OpenCLInformation[platform], (FreeQ[{"Name", "Version", "Vendor", "Devices"}, First[#]] && StringQ[First[#]])&]
	} ~Join~ (# -> OpenCLInformation`SystemInformation[platform, #]& /@ Range[Length[OpenCLInformation[platform, "Devices"]]])

OpenCLInformation`SystemInformation[platform_Integer?Positive, device_Integer?Positive] :=
	{
		"Device Type" -> OpenCLInformation[platform, device, "Type"],
		"Device Name" -> OpenCLInformation[platform, device, "Name"],
		"Device Vendor" -> OpenCLInformation[platform, device, "Vendor"],
		"Maximum Core Units" -> OpenCLInformation[platform, device, "Core Count"],
		"Global Memory Size" -> OpenCLInformation[platform, device, "Global Memory Size"],
		"Supports Double Precision" -> SupportsDoublePrecisionQ[OpenCLInformation, platform, device],
		"Detailed Information" ->
			Select[OpenCLInformation[platform, device], FreeQ[{"Type", "Name", "Vendor", "Core Count", "Global Memory Size"}, First[#]]&]
	}

OpenCLData`FormatSystemInformation[_, info_] :=
	info
OpenCLData`FormatSystemInformation["Global Memory Size", info_] :=
	GPUTools`Utilities`PrettyPrintMemory[info]
	
(******************************************************************************)
(******************************************************************************)

$OpenCLFunctionHead = OpenCLFunction
$OpenCLProgramObjectHead = OpenCLProgram
$OpenCLMemoryHead = OpenCLMemory

Options[OpenCLFunctionLoad] = FilterRules[Options[GPUFunctionLoad], Except["BinaryQ" | "BuildOptions" | "Source"]]
SyntaxInformation[OpenCLFunctionLoad] = {"ArgumentsPattern" -> {_, _, _, _, OptionsPattern[]}}

OpenCLFunctionLoad[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, OpenCLFunctionLoad, 3, 4], HoldFirst]) :=
	With[{res = Catch[GPUFunctionLoad[$OpenCLFunctionHead, $OpenCLProgramObjectHead, OpenCLFunctionLoad, args, opts]]},
		res /; (res =!= $Failed)
	]
	
SyntaxInformation[OpenCLFunctionLoad] = {"ArgumentsPattern" -> {_}}
OpenCLFunctionInformation[args:nonOptionPattern] ? (Function[{arg}, OptionsCheck[arg, OpenCLFunctionInformation, 1, 1], HoldFirst]) :=
	With[{res = GPUFunctionInformation[$OpenCLFunctionHead, $OpenCLProgramObjectHead, OpenCLFunctionInformation, args]},
		res /; ListQ[res]
	]

$OpenCLFunctionHead[functionArgs__][callArgs__] :=
	With[{res = GPUFunctionLaunch[$OpenCLFunctionHead, $OpenCLProgramObjectHead, $OpenCLMemoryHead, $OpenCLFunctionHead, {functionArgs}, {callArgs}]},
		res /; ListQ[res]
	]

Format[$OpenCLFunctionHead[$OpenCLProgramObjectHead[id_Integer, ___], kernelName_String, args_List, opts:OptionsPattern[]], StandardForm] :=
	$OpenCLFunctionHead["<>", kernelName, args]

Options[OpenCLMemoryLoad] = Options[OpenCLMemoryAllocate] = Options[GPUAddMemory]

SyntaxInformation[OpenCLFunctionLoad] = {"ArgumentsPattern" -> {_, _., OptionsPattern[]}}
OpenCLMemoryLoad[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, OpenCLMemoryLoad, 1, 2], HoldFirst]) :=
	With[{res = Catch[GPUAddMemory[$OpenCLMemoryHead, OpenCLMemoryLoad, args, opts]]},
		res /; Head[res] === $OpenCLMemoryHead
	]
OpenCLMemoryLoad[args___] /; (Message[OpenCLMemoryLoad::args, {args}]; False) := $Failed 
	
SyntaxInformation[OpenCLMemoryAllocate] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}}
OpenCLMemoryAllocate[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, OpenCLMemoryAllocate, 2, 2], HoldFirst]) :=
	With[{res = Catch[GPUAllocateMemory[$OpenCLMemoryHead, OpenCLMemoryAllocate, args, opts]]},
		res /; Head[res] === $OpenCLMemoryHead
	]
	
SyntaxInformation[OpenCLMemoryUnload] = {"ArgumentsPattern" -> {__}}
OpenCLMemoryUnload[args:nonOptionPattern] ? (Function[{arg}, OptionsCheck[arg, OpenCLMemoryUnload, 1, Infinity], HoldFirst]) :=
	Module[{res = GPUDeleteMemory[$OpenCLMemoryHead, OpenCLMemoryUnload, args]},
		res = Apply[And, TrueQ /@ res];
		Null /; TrueQ[res]
	]
	
SyntaxInformation[OpenCLMemoryGet] = {"ArgumentsPattern" -> {_}}
OpenCLMemoryGet[args:nonOptionPattern, opt_:"Original"] ? (Function[{arg}, OptionsCheck[arg, OpenCLMemoryGet, 1, 2], HoldFirst]) :=
	With[{res = GPUGetMemory[$OpenCLMemoryHead, OpenCLMemoryGet, args, opt]},
		res /; res =!= $Failed
	]
	
Format[mem:$OpenCLMemoryHead[id_Integer, ___], StandardForm] := $OpenCLMemoryHead["<" <> ToString[WGLMemoryGetID[mem]] <> ">", WGLMemoryGetType[mem]]

SyntaxInformation[OpenCLMemoryCopyToDevice] = {"ArgumentsPattern" -> {_}}
OpenCLMemoryCopyToDevice[args:nonOptionPattern] ? (Function[{arg}, OptionsCheck[arg, OpenCLMemoryCopyToDevice, 1, 1], HoldFirst]) :=
	With[{res = GPUCopyMemoryToDevice[$OpenCLMemoryHead, OpenCLMemoryCopyToDevice, args]},
		res /; res =!= $Failed
	]

SyntaxInformation[OpenCLMemoryCopyToHost] = {"ArgumentsPattern" -> {_}}
OpenCLMemoryCopyToHost[args:nonOptionPattern] ? (Function[{arg}, OptionsCheck[arg, OpenCLMemoryCopyToHost, 1, 1], HoldFirst]) :=
	With[{res = GPUCopyMemoryToHost[$OpenCLMemoryHead, OpenCLMemoryCopyToHost, args]},
		res /; res =!= $Failed
	]
	
SyntaxInformation[OpenCLMemoryInformation] = {"ArgumentsPattern" -> {_}}
OpenCLMemoryInformation[args:nonOptionPattern] ? (Function[{arg}, OptionsCheck[arg, OpenCLMemoryInformation, 1, 1], HoldFirst]) :=
	With[{res = GPUMemoryInformation[$OpenCLMemoryHead, OpenCLMemoryInformation, args]},
		res /; ListQ[res]
	]


(******************************************************************************)
(******************************************************************************)

Options[OpenCLProfile] = Options[OpenCLProfile]
SetAttributes[OpenCLProfile, Attributes[GPUProfile]]
OpenCLProfile[args__, opts:OptionsPattern[]] :=
	Module[{res},
		res = If[OpenCLQ[],
			True,
			If[$InitializeWGLError === {},
				Message[OpenCLProfile::nodev],
				InitializationErrorMessage[OpenCLProfile, $InitializeWGLError]
			];
			False
		];
		(
			res = Catch[GPUProfile[OpenCLProfile, Unevaluated[args], opts]];
			res /; res =!= $Failed
		) /; TrueQ[res]
	]

(******************************************************************************)
(******************************************************************************)

	
ParseLightingOption[lighting_] := Module[
	{numLights, lights, i},
	(* Lights are formatted as {{type}, {color}, {position}, {direction}, {halfangle, spotexponent}, {attenuation}} and then flattened. *)
	If[lighting === Automatic,
			numLights = 4;
			lights = Flatten[{
				(*{{2.0}, {1.0, 1.0, 1.0}, {15.0, 10.0, 5.0}, {0.0, 0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0, 0.0}}, 
				{{2.0}, {0.4, 0.4, 0.4}, {-15.0, -10.0, -5.0}, {0.0, 0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0, 0.0}}*)
				{{0.0}, {0.312, 0.188, 0.4}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0, 0.0}},
				{{1.0}, {0.8, 0.0, 0.0}, {0.0, 0.0, 0.0}, {-10.0, 0.0, -10.0}, {0.0, 0.0}, {0.0, 0.0, 0.0}},
				{{1.0}, {0.0, 0.8, 0.0}, {0.0, 0.0, 0.0}, {-10.0, -10.0, -10.0}, {0.0, 0.0}, {0.0, 0.0, 0.0}},
				{{1.0}, {0.0, 0.0, 0.8}, {0.0, 0.0, 0.0}, {0.0, -10.0, -10.0}, {0.0, 0.0}, {0.0, 0.0, 0.0}}
			}],
			numLights = Length[lighting];
			lights = ConstantArray[{}, numLights];
			For[i=1, i<=numLights, i++,
				lights[[i]] = Switch[lighting[[i]][[1]],
					"Ambient", 
						Print[{{0.0}, N[{lighting[[i]][[2]][[1]], lighting[[i]][[2]][[2]], lighting[[i]][[2]][[3]]}], {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0, 0.0}}];
						{{0.0}, N[{lighting[[i]][[2]][[1]], lighting[[i]][[2]][[2]], lighting[[i]][[2]][[3]]}], {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0, 0.0}},
					"Directional",
						If[Length[lighting[[i]][[3]]] == 3,
							{{1.0}, N[{lighting[[i]][[2]][[1]], lighting[[i]][[2]][[2]], lighting[[i]][[2]][[3]]}], {0.0, 0.0, 0.0}, -lighting[[i]][[3]], {0.0, 0.0}, {0.0, 0.0, 0.0}},
							{{1.0}, N[{lighting[[i]][[2]][[1]], lighting[[i]][[2]][[2]], lighting[[i]][[2]][[3]]}], {0.0, 0.0, 0.0}, lighting[[i]][[3]][[2]] - lighting[[i]][[3]][[1]], {0.0, 0.0}, {0.0, 0.0, 0.0}}
						],
					"Point",
						If[Length[lighting[[i]]] == 3,
							{{2.0}, N[{lighting[[i]][[2]][[1]], lighting[[i]][[2]][[2]], lighting[[i]][[2]][[3]]}], lighting[[i]][[3]], {0.0, 0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0, 0.0}},
							{{2.0}, N[{lighting[[i]][[2]][[1]], lighting[[i]][[2]][[2]], lighting[[i]][[2]][[3]]}], lighting[[i]][[3]], {0.0, 0.0, 0.0}, {0.0, 0.0}, lighting[[i]][[4]]}
						],
					"Spot",
						Switch[Length[lighting[[i]]],
							4,
								If[Length[lighting[[i]][[3]]] == 3,
									{{3.0}, N[{lighting[[i]][[2]][[1]], lighting[[i]][[2]][[2]], lighting[[i]][[2]][[3]]}], lighting[[i]][[3]], -lighting[[i]][[3]], {lighting[[i]][[4]], 0.0}, {0.0, 0.0, 0.0}},
									{{3.0}, N[{lighting[[i]][[2]][[1]], lighting[[i]][[2]][[2]], lighting[[i]][[2]][[3]]}], lighting[[i]][[3]][[1]], lighting[[i]][[3]][[2]], {lighting[[i]][[4]], 0.0}, {0.0, 0.0, 0.0}}
								],
							5,
								{{3.0}, N[{lighting[[i]][[2]][[1]], lighting[[i]][[2]][[2]], lighting[[i]][[2]][[3]]}], lighting[[i]][[3]][[1]], lighting[[i]][[3]][[2]], {lighting[[i]][[4]][[1]], lighting[[i]][[4]][[2]]}, lighting[[i]][[5]]},
							_,
								Message[OpenCLImplicitRender3D::invlight, lighting[[i]]];
								Return[$Failed]
						],
					_, (* Implement spot lights later. Much later. *)
						Message[OpenCLImplicitRender3D::invlight, lighting[[i]]];
						Return[$Failed]
				]
			];
			lights = Flatten[lights]
		];
		Return[{numLights, lights}];
	]


RandomMachineInteger[n___] :=
	RandomInteger[{-1*Developer`$MaxMachineInteger, Developer`$MaxMachineInteger}, n]

NVIDIADeviceQ[Automatic, Automatic] := NVIDIADeviceQ[1, 1]
NVIDIADeviceQ[platform_, device_] := NVIDIADeviceQ[platform, device] =
	StringMatchQ[OpenCLInformation[platform, device, "Vendor"], ___ ~~ "NVIDIA" ~~ ___, IgnoreCase -> True]
ATIDeviceQ[platform_, device_] := ATIDeviceQ[platform, device] =
	!NVIDIADeviceQ[platform, device]

(* ::Section:: *)
(* Mandelbulb *)


(* From the David Bucciarelli's MadelbulbGPU ${basedir}/ExtraComponents/Other/mandelbulbGPU-v1.0/rendering_kernel.cl
 * http://davibu.interfree.it/opencl/mandelbulbgpu/mandelbulbGPU.html 
 *)

RealQ[x_] := NumericQ[x] && Im[x] == 0;

(* ::Subsection:: *)
(* Option Checking *)
(* Specify what IterateIntertsect function to use *)
GetAndCheckOption[head:OpenCLFractalRender3D, errMsgHd:OpenCLFractalRender3D, n:Method, opts_] :=
	Module[{fractalType = OptionValue[head, {opts}, n]},
		If[fractalType == "Triplex" || fractalType == "Quaternion" || fractalType == "Custom",
			fractalType,
			Message[errMsgHd::invfmed, fractalType];
			Throw[$Failed]
		]
	]
	
(* Specify what type of fractal to use *)
GetAndCheckOption[head:OpenCLFractalRender3D, errMsgHd:OpenCLFractalRender3D, n:"Type", opts_] :=
	Module[{fractalType = OptionValue[head, {opts}, n]},
		If[fractalType == "Julia" || fractalType == "Mandelbrot",
			fractalType,
			Message[errMsgHd::invftype, fractalType];
			Throw[$Failed]
		]
	]
	
(* If Method->"Custom", supply an IterateIntersect function *)
GetAndCheckOption[head:OpenCLFractalRender3D, errMsgHd:OpenCLFractalRender3D, n:"IterateIntersectFunction", opts_] :=
	Module[{iif = OptionValue[head, {opts}, n]},
		If[StringQ[iif] || iif === None,
			iif,
			Message[errMsgHd::inviif, iif];
			Throw[$Failed]
		]
	]
	
(* Specify a surface color *)
GetAndCheckOption[head:OpenCLFractalRender3D, errMsgHd:OpenCLFractalRender3D, n:"Color", opts_] :=
	Module[{color = OptionValue[head, {opts}, n]},
		If[Length[color] =!= 3,
			Message[errMsgHd::invclr, color];
			Throw[$Failed],
			If[VectorQ[color, RealQ],
				N[color],
				Message[errMsgHd::invclr, color];
				Throw[$Failed]
			]
		]
	]

(* Specify a specular exponent *)
GetAndCheckOption[head:OpenCLFractalRender3D, errMsgHd:OpenCLFractalRender3D, n:"SpecularExponent", opts_] :=
	Module[{specularExponent = OptionValue[head, {opts}, n]},
		If[RealQ[specularExponent],
			N[specularExponent],
			Message[errMsgHd::invspece, specularExponent];
			Throw[$Failed]
		]
	]
	
(* Specify the coefficient of specularity *)
GetAndCheckOption[head:OpenCLFractalRender3D, errMsgHd:OpenCLFractalRender3D, n:"Specularity", opts_] :=
	Module[{specularity = OptionValue[head, {opts}, n]},
		If[RealQ[specularity],
			N[specularity],
			Message[errMgsHd::invspec, specularity];
			Throw[$Failed]
		]
	]
	
(* Specify whether or not shadows are to be used. *)
GetAndCheckOption[head:OpenCLFractalRender3D, errMsgHd:OpenCLFractalRender3D, n:"Shadows", opts_] :=
	Module[{useShadows = OptionValue[head, {opts}, n]},
		If[!MemberQ[{True, False}, useShadows],
			Message[errMsgHd::invshdw, useShadows];
			Throw[$Failed]
		];
		If[TrueQ[useShadows],
			1,
			0
		]
	]
	
(* useFloor specifies both whether or not to use a floor, and where the floor is located. It's presumed a floor is orthogonal to the Y-axis. *)
GetAndCheckOption[head:OpenCLFractalRender3D, errMsgHd:OpenCLFractalRender3D, n:"Floor", opts_] :=
	Module[{useFloor = OptionValue[head, {opts}, n]},
		If[MemberQ[{True, False}, useFloor],
			If[TrueQ[useFloor], {1, -2.0}, {0, 0.0}],
			If[RealQ[useFloor],
				{1, N[useFloor]},
				Message[errMsgHd::invflr, useFloor];
				Throw[$Failed]
			]
		]
	]
	
(* Specify maximum number of iterations to be used in IterateIntersect *)
GetAndCheckOption[head:OpenCLFractalRender3D, errMsgHd:OpenCLFractalRender3D, n:"MaxIterations", opts_] :=
	Module[{maxIterations = OptionValue[head, {opts}, n]},
		If[IntegerQ[maxIterations] && maxIterations > 0,
			maxIterations,
			Message[errMsgHd::invmaxit, maxIterations];
			Throw[$Failed]
		]
	]
	
(* Specify how close to the surface a ray must be before it is considered a hit *)
GetAndCheckOption[head:OpenCLFractalRender3D, errMsgHd:OpenCLFractalRender3D, n:"Precision", opts_] :=
	Module[{precision = OptionValue[head, {opts}, n]},
		If[RealQ[precision],
			precision,
			Message[errMsgHd::invprec, precision];
			Throw[$Failed]
		]
	]
	
(* Set to True to render a single frame, rather than creating an interactive display *)
GetAndCheckOption[head:OpenCLFractalRender3D, errMsgHd:OpenCLFractalRender3D, n:"SingleFrame", opts_] :=
	Module[{singleFrame = OptionValue[head, {opts}, n]},
		If[MemberQ[{True, False}, singleFrame],
			singleFrame,
			Message[errMsgHd::invsfrm, singleFrame];
			Throw[$Failed]
		]
	]
	
(* Supply parameters if "SingleFrame"->True *)
GetAndCheckOption[head:OpenCLFractalRender3D, errMsgHd:OpenCLFractalRender3D, n:"SingleFrameParameters", opts_] :=
	Module[{singleFrameParams = OptionValue[head, {opts}, n]},
		If[Length[singleFrameParams]==3 || singleFrameParams === None,
			singleFrameParams,
			Message[errMsgHd::invfrprms, singleFrameParams];
			Throw[$Failed]
		]
	]
	
(* Specify whether or not to use antialiasing and (optionally) how many samples per pixel are to be used *)
GetAndCheckOption[head:OpenCLFractalRender3D, errMsgHd:OpenCLFractalRender3D, n:"Multisampling", opts_] :=
	Module[{multisampling = OptionValue[head, {opts}, n]},
		If[MemberQ[{True, False}, multisampling] || (IntegerQ[multisampling] && multisampling > 0),
			multisampling,
			Message[errMsgHd::invmsmp, multisampling];
			Throw[$Failed]
		]
	]

(* Specify the radius of the bounding sphere to be used. *)
GetAndCheckOption[head:OpenCLFractalRender3D, errMsgHd:OpenCLFractalRender3D, n:"BoundingRadius", opts_] :=
	Module[{bRad = OptionValue[head, {opts}, n]},
		If[RealQ[bRad] && bRad > 0,
			N[bRad],
			Message[errMsgHd::invbrad, bRad];
			Throw[$Failed]
		]
	]
	
GetAndCheckOption[head:OpenCLFractalRender3D, errMsgHd:OpenCLFractalRender3D, n:ImageSize, opts_] :=
	Module[{imgSize = OptionValue[head, {opts}, n]},
		If[RealQ[imgSize] && imgSize > 0,
			IntegerPart[imgSize],
			Message[errMsgHd::invimgsz, imgSize];
			Throw[$Failed]
		]
	]
	
GetAndCheckOption[head:OpenCLFractalRender3D, errMsgHd:OpenCLFractalRender3D, n:AspectRatio, opts_] :=
	Module[{aspectRatio = OptionValue[head, {opts}, n]},
		If[RealQ[aspectRatio],
			N[aspectRatio],
			Message[errMsgHd::invar, aspectRatio];
			Throw[$Failed]
		]
	]
	
GetAndCheckOption[head:OpenCLFractalRender3D, errMsgHd:OpenCLFractalRender3D, n:PerformanceGoal, opts_] :=
	Module[{perfGoal = OptionValue[head, {opts}, n]},
		If[MemberQ[{"Quality", "Speed", Automatic}, perfGoal],
			perfGoal,
			Message[errMsgHd::invprfgl, perfGoal];
			Throw[$Failed]
		]
	]
	
GetAndCheckOption[head:OpenCLFractalRender3D, errMsgHd:OpenCLFractalRender3D, n:Lighting, opts_] :=
	Module[{lighting = OptionValue[head, {opts}, n]},
		If[VectorQ[lighting, MemberQ[{"Ambient", "Directional", "Point", "Spot"}, #[[1]]]&] || lighting===Automatic,
			lighting,
			Message[errMsgHd::invlight, lighting];
			Throw[$Failed]
		]
	]
	
(* ::Subsection:: *)
(* Implementation *)

Options[OpenCLFractalRender3D] = Sort[{
	Lighting -> {{"Point", {1.0, 1.0, 1.0}, {15.0, 10.0, 5.0}}}(*Automatic*), (* The default Lighting behavior is awful for this *)
	ImageSize -> 256,
	AspectRatio -> 1.0,
	PerformanceGoal -> Automatic,
	"Color" -> {0.9, 0.35, 0.15},
	"SpecularExponent" -> 30.0,
	"Specularity" -> 0.65,
	"Shadows" -> True,
	"Floor" -> True,
	"MaxIterations" -> 5,
	"Precision" -> 0.001,
	"BoundingRadius" -> 2.0,
	Method -> "Triplex",
	"Type" -> "Mandelbrot",
	"IterateIntersectFunction" -> None,
	"SingleFrame" -> False,
	"SingleFrameParameters" -> None,
	"Multisampling" -> False,
	"Platform" -> Automatic,
	"Device" -> Automatic,
	"TargetPrecision" -> "Single"
}]

SyntaxInformation[OpenCLFractalRender3D] = {"ArgumentsPattern" -> {OptionsPattern[]}}
OpenCLFractalRender3D[opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, OpenCLFractalRender3D, 0, 0], HoldFirst]) :=
	With[{res = Catch[iOpenCLFractalRender3D[opts]]},
		res /; res =!= $Failed
	]
iOpenCLFractalRender3D[opts:OptionsPattern[]] :=
	Module[{
			iifcode, color, specExp, spec, useShadows, useFloor, maxIterations, fractalType, platform, device, targetPrecision, file, 
			code, code1, code2, code3, hImgData, dImgData, hRenderingConfig, dRenderingConfig, hCamera, RenderMandelbulb, r, y, phi, mu1, mu2, mu3, mu4, 
			lx, ly, lz, isJulia, isQuaternion, mu4Min, mu4Max, mu4Start, mu4Name, r4Transform, precision, fractalMethod, isSingleFrame, singleFrameParams, 
			singleFrameOutput, doMultisampling, numSamples, bRad, commonh, imgWidth, imgHeight, aspectRatio, 
			imgRoundUp, perfGoal, imgWidthSmall, imgHeightSmall, dImgDataSmall, lighting, numLights, lights, fractalFunction
		   },
		   
		If[!OpenCLQ[],
			Return[$Failed]
		];
		
		lighting = GetAndCheckOption[OpenCLFractalRender3D, Lighting, {opts}];
		perfGoal = GetAndCheckOption[OpenCLFractalRender3D, PerformanceGoal, {opts}];
		imgWidth = GetAndCheckOption[OpenCLFractalRender3D, ImageSize, {opts}];
		aspectRatio = GetAndCheckOption[OpenCLFractalRender3D, AspectRatio, {opts}];
		color = GetAndCheckOption[OpenCLFractalRender3D, "Color", {opts}];
		specExp = GetAndCheckOption[OpenCLFractalRender3D, "SpecularExponent", {opts}];
		spec = GetAndCheckOption[OpenCLFractalRender3D, "Specularity", {opts}];
		useShadows = GetAndCheckOption[OpenCLFractalRender3D, "Shadows", {opts}];
		useFloor = GetAndCheckOption[OpenCLFractalRender3D, "Floor", {opts}];
		maxIterations = GetAndCheckOption[OpenCLFractalRender3D, "MaxIterations", {opts}];
		precision = GetAndCheckOption[OpenCLFractalRender3D, "Precision", {opts}];
		fractalMethod = GetAndCheckOption[OpenCLFractalRender3D, Method, {opts}];
		fractalType = GetAndCheckOption[OpenCLFractalRender3D, "Type", {opts}];
		iifcode = GetAndCheckOption[OpenCLFractalRender3D, "IterateIntersectFunction", {opts}];
		isSingleFrame = GetAndCheckOption[OpenCLFractalRender3D, "SingleFrame", {opts}];
		singleFrameParams = GetAndCheckOption[OpenCLFractalRender3D, "SingleFrameParameters", {opts}];
		doMultisampling = GetAndCheckOption[OpenCLFractalRender3D, "Multisampling", {opts}];
		bRad = GetAndCheckOption[OpenCLFractalRender3D, "BoundingRadius", {opts}];
		platform = GetAndCheckOption[OpenCLFractalRender3D, "Platform", {opts}];
		device = GetAndCheckOption[OpenCLFractalRender3D, "Device", {opts}];
    	targetPrecision = GetAndCheckOption[OpenCLFractalRender3D, "TargetPrecision", {opts}];

        bRad = bRad^2;
        
        (* Need camera location, fractal parameters, and light location to render a single frame. *)
        If[isSingleFrame && singleFrameParams===None,
        	Message[OpenCLFractalRender3D::invfrprms, singleFrameParams];
        	Return[$Failed]
        ];
        
        (* Default to 4 samples for multisampling *)
        If[doMultisampling == True,
        	numSamples = 4,
        	numSamples = 1
        ];
        If[IntegerQ[doMultisampling],
        	numSamples = doMultisampling;
        	doMultisampling = True
        ];
        
        isJulia = If[fractalType == "Julia", 1, 0];
		isQuaternion = If[fractalMethod == "Quaternion", 1, 0];
		
		(* Import header, with defines and helper functions *)
		file = FileNameJoin[{$OpenCLSupportFilesPath, "Mandelbulb_defines.h"}];
		code1 = Import[file, "Text"];
		
		(* code2 contains definition of IterateIntersect *)
		code2 = If[fractalMethod == "Custom",
			iifcode,
			If[isQuaternion == 1,
				"static float IterateIntersect(const float4 z0, const float4 c0, const unsigned int maxIterations) { return IterateIntersectQuaternion(z0, c0, maxIterations); }",
				"static float IterateIntersect(const float4 z0, const float4 c0, const unsigned int maxIterations) { return IterateIntersectTriplex(z0, c0, maxIterations); }"
			]
		];
		
		(* Import everything else (__kernel function, shading stuff, bounding sphere and floor intersection tests, etc...) *)
		file = FileNameJoin[{$OpenCLSupportFilesPath, "Mandelbulb_kernel.cl"}];
		code3 = Import[file, "Text"];
		
		(* Import common.h *)
		file = FileNameJoin[{$OpenCLSupportFilesPath, "common.h"}];
		commonh = Import[file, "Text"];
		(* Combine into one string of code *)
		code = StringJoin[Riffle[{commonh, code1, code2, code3}, "\n\n"]];
		
		
		imgRoundUp = 256 / GCD[imgWidth, 256]; (* 256 = workgroup size *)
		imgHeight = imgRoundUp * Ceiling[imgWidth / (aspectRatio * imgRoundUp)];
		
		If[perfGoal === Automatic || perfGoal == "Speed",
			imgWidthSmall = Floor[imgWidth / $PerformanceSpeedImageDivisor];
			imgRoundUp = 256 / GCD[imgWidthSmall, 256];
			imgHeightSmall = imgRoundUp * Ceiling[imgWidthSmall / (aspectRatio * imgRoundUp)];
			If[imgHeightSmall*aspectRatio - imgWidthSmall > 0.5*imgRoundUp,
				imgHeightSmall -= imgRoundUp
			];
			dImgDataSmall = OpenCLMemoryAllocate["Float", {imgHeightSmall, imgWidthSmall, 3}, "Platform"->platform, "Device"->device, "TargetPrecision"->targetPrecision]
		];
		
		dImgData = OpenCLMemoryAllocate["Float", {imgHeight, imgWidth, 3}, "Platform"->platform, "Device"->device, "TargetPrecision"->targetPrecision];
		
		specExp = ToString[specExp] <> "f";
		spec = ToString[spec] <> "f";
		color = (ToString[#] <> "f")& /@ color;
		
		{numLights, lights} = ParseLightingOption[lighting];
		
		mu4Min = If[isQuaternion==1, -2.0, 1.0];
		mu4Max = If[isQuaternion==1, 2.0, 16.0];
		mu4Start = If[isQuaternion==1, -0.5, 8.0];
		mu4Name = If[isQuaternion==1, "\[Mu]4", "Exponent"];
		
		(* Not sure if the transform to R4 will actually produce different-looking fractals or not... *)
		r4Transform = {
			{1.0, 0.0, 0.0},
			{0.0, 1.0, 0.0},
			{0.0, 0.0, 1.0},
			{0.0, 0.0, 0.0}
		}; (* currently just corresponds to subspace w = 0 *)

		fractalFunction = OpenCLFunctionLoad[
			code,
			"MandelbulbGPU",
			{{"Float", _, "Output"}, {"Float", _, "Input"}, _Integer, _Integer, "Float", "Float", _Integer, _Integer, _Integer},
			256,
			"Platform"->platform, 
			"Device"->device,
			"IncludeDirectories" -> {},
			"TargetPrecision" -> targetPrecision,
			"CompileOptions" ->
				If[NVIDIADeviceQ[platform, device] && $OperatingSystem =!= "MacOSX",
					"-cl-nv-maxrregcount=32",
					""
				],
			"Defines"-> {
				"JULIA"-> isJulia,
				"COLOR_R"->color[[1]],
				"COLOR_G"->color[[2]],
				"COLOR_B"->color[[3]],
				"SPECULAR_EXP"->specExp,
				"SPECULARITY"->spec,
				"ENABLE_SHADOW"->useShadows,
				"ENABLE_FLOOR"->useFloor[[1]],
				"FLOOR_POSITION"->useFloor[[2]],
				"QUATERNION"->isQuaternion,
				"BOUNDING_RADIUS_2"->bRad
			}
		];
		RenderMandelbulb[cameraPosition_, coefficients_, lightPosition_, cameraTarget_:{0.0, 0.0, 0.0}, optSeq__] :=
			Module[{res, camera3, camera4, cameraP, R, Y, Phi, i, enableAccum, multisamplingOffset, controlActive = TrueQ["ControlActiveQ" /. {optSeq}]},
				For[i=0, i<numSamples, i++,
					enableAccum = If[i==0, 0, 1];
					(* We take sampleX,sampleY = 0.0,0.0 if i == 0 so that non-multisampled images will not jitter *)
					multisamplingOffset = If[i==0, {0.0, 0.0}, RandomReal[{0.0, 1.0}, 2]]; 
					
					{R, Y, Phi} = cameraPosition;
					cameraP = Re[{R*Cos[Phi], Y, R*Sin[Phi]}];
					camera3 = Normalize[cameraTarget - cameraP];
					camera4 = (imgWidth*0.5135/imgHeight)*Normalize[Cross[camera3, {0, 1, 0}]];
					
					{numLights, lights} = ParseLightingOption[{{"Point", {1.0, 1.0, 1.0}, lightPosition}}];
						
					(* Transform camera into R4. When using actual 3D (ie, triplex) fractals, make sure r4Transform is I4x3 *)
					hCamera = 1.0 * {
						cameraP,
						cameraTarget,
						camera3,
						camera4,
						0.5135 * Normalize[Cross[camera4, camera3]]
					};
					hCamera = Dot[r4Transform, #]&/@hCamera;
					
					hRenderingConfig = If[(perfGoal === "Speed") || ((perfGoal === Automatic) && controlActive),
						Flatten[{2.0*precision, coefficients, hCamera, lights}],
						Flatten[{precision, coefficients, hCamera, lights}]
					];
					dRenderingConfig = OpenCLMemoryLoad[hRenderingConfig, "Platform"->platform, "Device"->device, "TargetPrecision"->targetPrecision];

					res = Check[
						If[(perfGoal === "Quality") || (perfGoal === Automatic && controlActive === False),
							fractalFunction[dImgData, dRenderingConfig, maxIterations, enableAccum, multisamplingOffset[[1]], multisamplingOffset[[2]], imgWidth, imgHeight, numLights, imgWidth*imgHeight],
							fractalFunction[dImgDataSmall, dRenderingConfig, maxIterations-2, enableAccum, multisamplingOffset[[1]], multisamplingOffset[[2]], imgWidthSmall, imgHeightSmall, numLights, imgWidthSmall*imgHeightSmall]
						],
						$Failed
					];
					
					OpenCLMemoryUnload[dRenderingConfig];
				];
				
				If[res === $Failed,
					$Failed,
					If[(perfGoal === "Speed") || ((perfGoal === Automatic) && controlActive),
						hImgData = OpenCLMemoryGet[dImgDataSmall];
						Return[ImageResize[ImageMultiply[Image[hImgData], 1.0 / numSamples], imgWidth]],  (* Why does this sometimes produce an overflow unreliably? *)
						hImgData = OpenCLMemoryGet[dImgData];
						Return[ImageMultiply[Image[hImgData], 1.0 / numSamples]]
					]
				]
			];
			
		If[isSingleFrame,
			singleFrameOutput = RenderMandelbulb[singleFrameParams[[1]], singleFrameParams[[2]], singleFrameParams[[3]], "ControlActiveQ"->False];
			OpenCLMemoryUnload[dImgData];
			If[MemberQ[{Automatic, "Speed"}, perfGoal], OpenCLMemoryUnload[dImgDataSmall]];
			Return[singleFrameOutput];
		];
			
		RenderMandelbulb[{5.0, 0.0, 0.0}, {0.5, -0.5, 0.5, 0.02}, {15.0, 10.0, 5.0}, "ControlActiveQ"->False];
		
		DynamicModule[{},
			Manipulate[
				RenderMandelbulb[{r, y, phi}, {mu1, mu2, mu3, mu4}, {lx, ly, lz}, "ControlActiveQ" -> ControlActive[]],
				Style["Camera Position (Cylindrical)", 12, Bold],
				{{r, 5.0, "r"}, 0.0, 10.0, 0.1},
				{{y, 0.0, "y"}, -2.0, 8.0, 0.1},
				{{phi, 0.0, "\[Phi]"}, -2.0*Pi, 2.0*Pi, 0.1},
				Delimiter,
				Style["Light Position (Cartesian)", 12, Bold],
				{{lx, 15.0, "x"}, -20.0, 20.0, 0.5},
				{{ly, 10.0, "y"}, 0.0, 20.0, 0.5},
				{{lz, 5.0, "z"}, -20.0, 20.0, 0.5},
				Delimiter,
				Style["Fractal Parameters", 12, Bold],
				{{mu1, 0.5, "\[Mu]1"}, -2.0, 2.0, 0.02},
				{{mu2, -0.5, "\[Mu]2"}, -2.0, 2.0, 0.02},
				{{mu3, 0.5, "\[Mu]3"}, -2.0, 2.0, 0.02},
				{{mu4, mu4Start, mu4Name}, mu4Min, mu4Max, 0.02},
				ControlPlacement -> Left,
				Alignment -> Center,
				Deployed -> True,
				Deinitialization :> {
					OpenCLMemoryUnload[dImgData];
					If[MemberQ[{Automatic, "Speed"}, perfGoal], OpenCLMemoryUnload[dImgDataSmall]];
				},
				SynchronousUpdating->False
			]
		]
	]

(* ::Section:: *)
(* Mersenne Twister *)

MersenneTwiserReadDataFile[fileName_String] :=
	Module[{data},
		Assert[FileExistsQ[fileName]];
  		data = BinaryReadList[fileName, "Integer32"];
  		Assert[Divisible[Length[data], 4]];
  		Transpose[Partition[data, 4]][[;;3]]
	]

(* ::Subsection:: *)
(* Option checking *)

(* Specify a file with initial state *)
GetAndCheckOption[head:OpenCLMersenneTwister, errMsgHd_, n:"SeedFile", opts_] :=
	Module[{file = OptionValue[head, {opts}, n]},
		If[(StringQ[file] && FileExistsQ[file]) || file === None,
			file,
			Message[errMsgHd::invfile, file];
			Throw[$Failed]
		]
	]
	
(* Specify a seed *)
GetAndCheckOption[head:OpenCLMersenneTwister, errMsgHd_, n:"SeedValue", opts_] :=
	Module[{seed = OptionValue[head, {opts}, n]},
		If[(IntegerQ[seed] && Developer`MachineIntegerQ[seed]) || seed === Automatic,
			If[seed === Automatic,
				RandomMachineInteger[],
				seed
			],
			Message[errMsgHd::invseed, seed];
			Throw[$Failed]
		]
	]

(* ::Subsection:: *)
(* Implementation *)

Options[OpenCLMersenneTwister] = Sort[{
	"SeedFile" -> None,
	"SeedValue" -> Automatic,
	"Platform" -> Automatic,
	"Device" -> Automatic,
	"TargetPrecision" -> "Single"
}]
SyntaxInformation[OpenCLMersenneTwister] = {"ArgumentsPattern" -> {_, OptionsPattern[]}}
OpenCLMersenneTwister[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, OpenCLMersenneTwister, 1, 1], HoldFirst]) :=
	With[{res = Catch[iOpenCLMersenneTwister[args, opts]]},
		res /; res =!= $Failed
	]
iOpenCLMersenneTwister[numberOfRandomNumbers_, opts:OptionsPattern[]] :=
	Module[
		{
		codeFile, mersenneTwister, platform, device, targetPrecision, MTRNGCount, PATHN, NPerRng, RANDN, hsMatrixA, hsMaskB, hsMaskC, hsSeed, seedFile, seedValue,
		dsMatrixA, dsMaskB, dsMaskC, dsSeed, outputVector, res
		},
	
		If[!OpenCLQ[],
			Throw[Message[OpenCLLink::nodev]; $Failed]
		];
		If[!IntegerQ[numberOfRandomNumbers] || !Positive[numberOfRandomNumbers],
			Throw[Message[OpenCLMersenneTwister::numrand, numberOfRandomNumbers]; $Failed]
		];
		
		seedFile = GetAndCheckOption[OpenCLMersenneTwister, "SeedFile", {opts}];
		seedValue = GetAndCheckOption[OpenCLMersenneTwister, "SeedValue", {opts}];
		platform = GetAndCheckOption[OpenCLMersenneTwister, "Platform", {opts}];
		device = GetAndCheckOption[OpenCLMersenneTwister, "Device", {opts}];
    	targetPrecision = GetAndCheckOption[OpenCLMersenneTwister, "TargetPrecision", {opts}];
    
		codeFile = FileNameJoin[{$OpenCLSupportFilesPath, "MersenneTwister_kernel.cl"}];
		
		(* Determine number of random numbers to generate per thread (NPerRng) and total number of numbers to be generated (RANDN) *)
		MTRNGCount = 4096;
		PATHN = numberOfRandomNumbers;
		NPerRng = Ceiling[PATHN / MTRNGCount];
		RANDN = MTRNGCount * NPerRng;
		
		(* Read/generate initial state *)
		{hsMatrixA, hsMaskB, hsMaskC} = If[seedFile === None,
        	RandomMachineInteger[{3, MTRNGCount}],
        	MersenneTwiserReadDataFile[seedFile]
        ];

        hsSeed = ConstantArray[seedValue, MTRNGCount];
        
		dsMatrixA = OpenCLMemoryLoad[hsMatrixA, "Device"->device, "Platform"->platform, "TargetPrecision"->targetPrecision];
		dsMaskB = OpenCLMemoryLoad[hsMaskB, "Device"->device, "Platform"->platform, "TargetPrecision"->targetPrecision];
		dsMaskC = OpenCLMemoryLoad[hsMaskC, "Device"->device, "Platform"->platform, "TargetPrecision"->targetPrecision];
		dsSeed = OpenCLMemoryLoad[hsSeed, "Device"->device, "Platform"->platform, "TargetPrecision"->targetPrecision];
		outputVector = OpenCLMemoryAllocate[Real, RANDN, "Device"->device, "Platform"->platform, "TargetPrecision"->targetPrecision];
		
		
		res = Check[
			mersenneTwister = OpenCLFunctionLoad[
				{codeFile}, "MersenneTwister", {{_Real, _, "Output"}, {_Real, _, "Input"}, {_Real, _, "Input"}, {_Real, _, "Input"}, {_Real, _, "Input"}, _Integer}, 128,
				"Device"->device, "Platform"->platform, "TargetPrecision" -> targetPrecision
			];
			mersenneTwister[outputVector, dsMatrixA, dsMaskB, dsMaskC, dsSeed, NPerRng, 4096],
			$Failed
		];
				
		If[res =!= $Failed,
			res = OpenCLMemoryGet[First@res]
		];
		
		OpenCLMemoryUnload /@ {outputVector, dsMatrixA, dsMaskB, dsMaskC, dsSeed};
		
		If[res === $Failed,
			$Failed,
			Take[res, numberOfRandomNumbers]
		]
	]

(* ::Section:: *)
(* Implicit ray-tracer *)

$PerformanceSpeedImageDivisor = 8

(* ::Subsection:: *)
(* Option checking *)
GetAndCheckOption[head:OpenCLImplicitRender3D, errMsgHd:OpenCLImplicitRender3D, n:"Shadows", opts_] :=
	Module[{shadowsQ = OptionValue[head, {opts}, n]},
		If[MemberQ[{True, False}, shadowsQ],
			If[shadowsQ, 1, 0],
			Message[errMsgHd::invshdw, shadowsQ];
			Throw[$Failed]
		]
	]
	
GetAndCheckOption[head:OpenCLImplicitRender3D, errMsgHd:OpenCLImplicitRender3D, n:"Floor", opts_] :=
	Module[{floorQ = OptionValue[head, {opts}, n]},
		If[MemberQ[{True, False}, floorQ],
			If[floorQ, 1, 0],
			Message[errMsgHd::invflr, floorQ];
			Throw[$Failed]
		]
	]

GetAndCheckOption[head:OpenCLImplicitRender3D, errMsgHd:OpenCLImplicitRender3D, n:"FloorPosition", opts_] :=
	Module[{floorPos = OptionValue[head, {opts}, n]},
		If[RealQ[floorPos],
			ToString[N[floorPos]] <> "f",
			Message[errMsgHd::invflr, floorPos];
			Throw[$Failed]
		]
	]
	
GetAndCheckOption[head:OpenCLImplicitRender3D, errMsgHd:OpenCLImplicitRender3D, n:"Precision", opts_] :=
	Module[{precision = OptionValue[head, {opts}, n]},
		Which[
			RealQ[precision] && precision > 0.0,
				N[precision],
			precision === Automatic,
				precision,
			True,
				Message[errMsgHd::invprec, precision];
				Throw[$Failed]
		]
	]
	
GetAndCheckOption[head:OpenCLImplicitRender3D, errMsgHd:OpenCLImplicitRender3D, n:"SingleFrame", opts_] :=
	Module[{singleFrameQ = OptionValue[head, {opts}, n]},
		If[MemberQ[{True, False}, singleFrameQ],
			singleFrameQ,
			Message[errMsgHd::invsngf, singleFrameQ];
			Throw[$Failed]
		]
	]
	
GetAndCheckOption[head:OpenCLImplicitRender3D, errMsgHd:OpenCLImplicitRender3D, n:"SingleFrameParameters", opts_] :=
	Module[{singleFrameParams = OptionValue[head, {opts}, n]},
		Which[
			singleFrameParams === None,
				None,
			VectorQ[singleFrameParams, VectorQ[#, RealQ]&] && Length[singleFrameParams] == 3,
				N[singleFrameParams],
			True,
				Message[errMsgHd::invsngf, singleFrameParams];
				Throw[$Failed]
		]
	]
	
GetAndCheckOption[head:OpenCLImplicitRender3D, errMsgHd:OpenCLImplicitRender3D, n:"SliderParameters", opts_] :=
	Module[{sliderParams = OptionValue[head, {opts}, n]},
		If[(VectorQ[sliderParams, Head[#]===Real&] && Length[sliderParams]==3) || Head[sliderParams]===Real,
			If[RealQ[sliderParams],
				{-20.0, 20.0, N[sliderParams]}
			];
			sliderParams,
			Message[errMsgHd::invsld, sliderParams];
			Throw[$Failed]
		]
	]
	
GetAndCheckOption[head:OpenCLImplicitRender3D, errMsgHd:OpenCLImplicitRender3D, n:"BlockSize", opts_] :=
	Module[{wgSize = OptionValue[head, {opts}, n]},
		If[IntegerQ[wgSize] && wgSize>0 && Divisible[wgSize, 32],
			wgSize,
			Message[errMsgHd::invwgsz, wgSize];
			Throw[$Failed]
		]
	]
	
GetAndCheckOption[head:OpenCLImplicitRender3D, errMsgHd:OpenCLImplicitRender3D, n:AspectRatio, opts_] :=
	Module[{aspectRatio = OptionValue[head, {opts}, n]},
		If[RealQ[aspectRatio],
			N[aspectRatio],
			Message[errMsgHd::invar, aspectRatio];
			Throw[$Failed]
		]
	]
	
GetAndCheckOption[head:OpenCLImplicitRender3D, errMsgHd:OpenCLImplicitRender3D, n:ImageSize, opts_] :=
	Module[{imgSize = OptionValue[head, {opts}, n]},
		If[RealQ[imgSize] && imgSize > 0,
			IntegerPart[imgSize],
			Message[errMsgHd::invimgsz, imgSize];
			Throw[$Failed]
		]
	]
	
GetAndCheckOption[head:OpenCLImplicitRender3D, errMsgHd:OpenCLImplicitRender3D, n:PerformanceGoal, opts_] :=
	Module[{perfGoal = OptionValue[head, {opts}, n]},
		If[MemberQ[{"Quality", "Speed", Automatic}, perfGoal],
			perfGoal,
			Message[errMsgHd::invprfgl, perfGoal];
			Throw[$Failed]
		]
	]
	
GetAndCheckOption[head:OpenCLImplicitRender3D, errMsgHd:OpenCLImplicitRender3D, n:Lighting, opts_] :=
	Module[{lighting = OptionValue[head, {opts}, n]},
		If[VectorQ[lighting, MemberQ[{"Ambient", "Directional", "Point", "Spot"}, First[#]]&] || lighting===Automatic,
			lighting,
			Message[errMsgHd::invlight, lighting];
			Throw[$Failed]
		]
	]

GetAndCheckOption[head:(OpenCLImplicitRender3D | OpenCLFractalRender3D), errMsgHd_, n:"TargetPrecision", opts_] :=
	Module[{opt = OptionValue[head, {opts}, n]},
		Switch[opt,
			"Single" | "Double",
				opt,
			Automatic,
				With[{
					platform = GetAndCheckOption[head, errMsgHd, "Platform", opts];
					device = GetAndCheckOption[head, errMsgHd, "Device", opts];
					},
					If[SupportsDoublePrecisionQ[errMsgHd, platform, device] ? "Single", "Double"]
				],
			_,
				Message[errMsgHd::invtrgtprs, opt];
				Throw[$Failed]
		]
	]
	
(* ::Subsection:: *)
(* Implementation *)

PolynomialToString[eq_, vars_] := Module[
	{coeffs, monomials, tempcoeff, temp, res, i},
	coeffs = CoefficientRules[eq, vars];
	monomials = ConstantArray["", Length[coeffs]];
	For[i=1, i<=Length[coeffs], i++,
		tempcoeff = ToString[N[coeffs[[i]][[2]]]];
		tempcoeff = tempcoeff <> "f";
		
		temp = Join[{tempcoeff}, ConstantArray["x", coeffs[[i]][[1]][[1]]], ConstantArray["y", coeffs[[i]][[1]][[2]]], ConstantArray["z", coeffs[[i]][[1]][[3]]]];
		If[Length[vars] == 4,
			temp = Join[temp, ConstantArray["w", coeffs[[i]][[1]][[4]]]]
		];
		
		monomials[[i]] = ToCCodeString[COperator[Times, temp]];
	];
	
	If[Length[monomials]>=1,
		If[Length[monomials]>=2,
			res = ToCCodeString[COperator[Plus, monomials]],
			res = monomials[[1]]
		],
		res = "0.0f"
	];
	
	Return[res];
]

(* Returns four strings; the first is code for affine arithmetic evaluation of the polynomial eq, the second, third, and fourth comprise the gradient of eq (ignoring w) *)
PolynomialOpenCLify[eq_, vars_] := Module[
	{coeffs, monomials, tempcoeff, temp, i, gradXPoly, gradYPoly, gradZPoly, gradX, gradY, gradZ},
	 
	coeffs = CoefficientRules[eq, vars];
	monomials = ConstantArray["", Length[coeffs]];
	For[i=1, i<=Length[coeffs], i++,
		tempcoeff = ToString[N[coeffs[[i]][[2]]]];
		tempcoeff = tempcoeff <> "f";
		
		temp = {};
		temp = Join[temp, {CAssign["a", CCast["raf", CCall["", {tempcoeff, "0.0f", "0.0f", "0.0f"}]]]}];
		temp = Join[temp, ConstantArray[CAssign["a", CCall["raf_mul", {"a", "rx"}]], coeffs[[i]][[1]][[1]]]];
		temp = Join[temp, ConstantArray[CAssign["a", CCall["raf_mul", {"a", "ry"}]], coeffs[[i]][[1]][[2]]]];
		temp = Join[temp, ConstantArray[CAssign["a", CCall["raf_mul", {"a", "rz"}]], coeffs[[i]][[1]][[3]]]];
		If[Length[vars]==4,
			temp = Join[temp, ConstantArray[CAssign["a", CCall["raf_mul_scalar", {"a", "w"}]], coeffs[[i]][[1]][[4]]]]
		];
		monomials[[i]] = temp;
	];
	
	gradX = D[eq, vars[[1]]];
	gradY = D[eq, vars[[2]]];
	gradZ = D[eq, vars[[3]]];
	
	gradXPoly = PolynomialToString[gradX, vars];
	gradYPoly = PolynomialToString[gradY, vars];
	gradZPoly = PolynomialToString[gradZ, vars];
	
	(* Rather than returning rafPoly, return monomials and construct the sum using intermediate steps *)
	Return[{monomials, gradXPoly, gradYPoly, gradZPoly}];
]

Options[OpenCLImplicitRender3D] = Sort[{
	ImageSize->256,
	AspectRatio->1.0,
	PerformanceGoal->Automatic,
	Lighting->Automatic,
	"BlockSize"->128,
	"GeneratedCodeDisplayFunction"->None,
	"SliderParameters"->{-20.0, 20.0, 0.0}, (* min max default *)
	"Precision"->Automatic,
	"SingleFrame"->False,
	"SingleFrameParameters"->None,
	"Shadows" -> True,
	"Floor" -> False,
	"FloorPosition" -> -5.0,
	"Platform" -> Automatic,
	"Device" -> Automatic,
	"TimingOutputFunction" -> None,
	"TargetPrecision" -> "Single"
}]
SyntaxInformation[OpenCLImplicitRender3D] = {"ArgumentsPattern" -> {_, _, _, OptionsPattern[]}}
OpenCLImplicitRender3D[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, OpenCLImplicitRender3D, 3, 3], HoldFirst]) :=
	With[{res = Catch[iOpenCLImplicitRender3D[args, opts]]},
		res /; res =!= $Failed
	]
iOpenCLImplicitRender3D[eq_, vars_, boundingRad0_, opts:OptionsPattern[]] :=
	Module[
		{
			codeFile, code, affinearithmetich, commonh, boundingRad, platform, device, timingFun, targetPrecision, bRad, hImgData, dImgData, 
			RenderIsosurface, r, camy, phi, lx, ly, lz, paramw, openCLPolys, useShadows, useFloor, floorPos, precision, singleFrameQ, singleFrameParams,
			singleFrameOutput, generatedCode, sliderParams, testRun, workgroupSize, codeDisplayFun, imgWidth, imgHeight, imgRoundUp, aspectRatio, perfGoal, hImgDataSmall,
			dImgDataSmall, imgWidthSmall, imgHeightSmall, lighting, lights, numLights, implicitFunction
		},
	
		If[!OpenCLQ[],
			Throw[Message[OpenCLLink::nodev]; $Failed]
		];
		
		If[!NumericQ[boundingRad0]|| !Positive[boundingRad0],
			Throw[Message[OpenCLImplicitRender3D::bound, boundingRad0]; $Failed]
		];
		
		boundingRad = N[boundingRad0];
		
		lighting = GetAndCheckOption[OpenCLImplicitRender3D, Lighting, {opts}];
		perfGoal = GetAndCheckOption[OpenCLImplicitRender3D, PerformanceGoal, {opts}];
		imgWidth = GetAndCheckOption[OpenCLImplicitRender3D, ImageSize, {opts}];
		aspectRatio = GetAndCheckOption[OpenCLImplicitRender3D, AspectRatio, {opts}];
		workgroupSize = GetAndCheckOption[OpenCLImplicitRender3D, "BlockSize", {opts}];
		codeDisplayFun = GetAndCheckOption[OpenCLImplicitRender3D, "GeneratedCodeDisplayFunction", {opts}];
		sliderParams = GetAndCheckOption[OpenCLImplicitRender3D, "SliderParameters", {opts}];
		singleFrameQ = GetAndCheckOption[OpenCLImplicitRender3D, "SingleFrame", {opts}];
		singleFrameParams = GetAndCheckOption[OpenCLImplicitRender3D, "SingleFrameParameters", {opts}];
		precision = GetAndCheckOption[OpenCLImplicitRender3D, "Precision", {opts}];
		useShadows = GetAndCheckOption[OpenCLImplicitRender3D, "Shadows", {opts}];
		useFloor = GetAndCheckOption[OpenCLImplicitRender3D, "Floor", {opts}];
		floorPos = GetAndCheckOption[OpenCLImplicitRender3D, "FloorPosition", {opts}];
		platform = GetAndCheckOption[OpenCLImplicitRender3D, "Platform", {opts}];
		device = GetAndCheckOption[OpenCLImplicitRender3D, "Device", {opts}];
    	timingFun = GetAndCheckOption[OpenCLImplicitRender3D, "TimingOutputFunction", {opts}];
    	targetPrecision = GetAndCheckOption[OpenCLImplicitRender3D, "TargetPrecision", {opts}];
		
		(* Check eq to see if it is a valid polynomial (in x, y, z, and w (user parameter) *)
		If[Not[PolynomialQ[eq, vars[[1]]] && PolynomialQ[eq, vars[[2]]] && PolynomialQ[eq, vars[[3]]]],
			Message[OpenCLImplicitRender3D::invpoly, eq];
			Return[$Failed]
		];
		If[Length[vars]==4 && Not[PolynomialQ[eq, vars[[4]]]],
			Message[OpenCLImplicitRender3D::invpoly, eq];
			Return[$Failed]
		];
		
		If[Not[MemberQ[{3,4},Length[vars]]],
			Message[OpenCLImplicitRender3D::invvars, vars];
			Return[$Failed]
		];
		
		(* Get the polynomial in affine arithmetic form and its gradient in C form *)
		openCLPolys = PolynomialOpenCLify[eq, vars];
		
		generatedCode = ConstantArray["", 2];
			
		Module[{loopCode, codeBody},
			loopCode = {#, CAssign["res", CCall["raf_add", {"res", "a"}]]}&/@openCLPolys[[1]];
			codeBody = Flatten[{CDeclare["raf", "res"], CDeclare["raf", "a"], CAssign["res", CCast["raf", CCall["", {"0.0f", "0.0f", "0.0f", "0.0f"}]]], loopCode, CReturn["res"]}];
			generatedCode[[1]] = ToCCodeString[CFunction["raf", "evaluate_raf",
				{{"raf", "rx"}, {"raf", "ry"}, {"raf", "rz"}, {"float", "w"}, {"float", "nan"}, {"float", "inf"}},
				codeBody
			]];
		];
		
		generatedCode[[2]] = ToCCodeString[CFunction["float4", "EvaluateGradient", {{"float4", "pos"}, {"float", "w"}},
			{
				CDeclare["float", "x"], CDeclare["float", "y"], CDeclare["float", "z"],
				CAssign["x", "pos.x"], CAssign["y", "pos.y"], CAssign["z", "pos.z"],
				CDeclare["float", "gradX"], CDeclare["float", "gradY"], CDeclare["float", "gradZ"],
				CAssign["gradX", openCLPolys[[2]]], CAssign["gradY", openCLPolys[[3]]], CAssign["gradZ", openCLPolys[[4]]],
				CReturn[CCast["float4", CCall["", {"gradX", "gradY", "gradZ", "0.0f"}]]]
			}
		]];
		
		codeDisplayFun[generatedCode[[1]]];
		codeDisplayFun[generatedCode[[2]]];

		codeFile = FileNameJoin[{$OpenCLSupportFilesPath, "implicit_kernel.cl"}];
		Assert[FileExistsQ[codeFile]];
		code = Import[codeFile, "Text"];
        
        codeFile = FileNameJoin[{$OpenCLSupportFilesPath, "affine_arithmetic.h"}];
        Assert[FileExistsQ[codeFile]];
        affinearithmetich = Import[codeFile, "Text"];
        
        codeFile = FileNameJoin[{$OpenCLSupportFilesPath, "common.h"}];
        Assert[FileExistsQ[codeFile]];
        commonh = Import[codeFile, "Text"];
        
        code = StringJoin[Riffle[{commonh, affinearithmetich, generatedCode[[1]], generatedCode[[2]], code}, "\n\n"]];
		
		If[precision === Automatic,
			precision = Min[boundingRad / 100.0, 0.01];
		];
		
        bRad = boundingRad^2;
		
		imgRoundUp = workgroupSize / GCD[imgWidth, workgroupSize];
		imgHeight = imgRoundUp * Ceiling[imgWidth / (aspectRatio * imgRoundUp)];
		If[imgHeight*aspectRatio - imgWidth > 0.5*imgRoundUp,
			imgHeight -= imgRoundUp
		];
        

		dImgData = OpenCLMemoryAllocate["Float", {imgHeight, imgWidth, 3}, "Platform"->platform, "Device"->device, "TargetPrecision"->targetPrecision];

		If[perfGoal === Automatic || perfGoal == "Speed",
			imgWidthSmall = Floor[imgWidth / $PerformanceSpeedImageDivisor];
			imgRoundUp = workgroupSize / GCD[imgWidthSmall, workgroupSize];
			imgHeightSmall = imgRoundUp * Ceiling[imgWidthSmall / (aspectRatio * imgRoundUp)];
			If[imgHeightSmall*aspectRatio - imgWidthSmall > 0.5*imgRoundUp,
				imgHeightSmall -= imgRoundUp
			];
			dImgDataSmall = OpenCLMemoryAllocate["Float", {imgHeightSmall, imgWidthSmall, 3}, "Platform"->platform, "Device"->device, "TargetPrecision"->targetPrecision]
		];
		
		(* Parse lighting *)
		{numLights, lights} = ParseLightingOption[lighting];
		
		implicitFunction = OpenCLFunctionLoad[
			code,
			"IsosurfaceGPU",
			{{"Float", _, "Output"}, {"Float", _, "Input"}, "Integer32", "Integer32", "Integer32"},
			workgroupSize,
			"TargetPrecision" -> targetPrecision,
			"Platform"->platform, 
			"Device"->device,
			"Defines"-> {
				"ENABLE_SHADOWS"->useShadows,
				"ENABLE_FLOOR"->useFloor,
				"FLOOR_POSITION"-> floorPos,
				"BOUNDING_RADIUS_2"-> ToString[CForm[bRad]]<>"f"
			}
		];

		RenderIsosurface[cameraPosition_, parameters_, lightPosition_, cameraTarget_:{0.0, 0.0, 0.0}, optSeq__] := 
			Module[{res, camera3, camera4, cameraP, R, Y, Phi, hRenderingConfig, dRenderingConfig, hCamera, controlActive = TrueQ["ControlActiveQ" /. {optSeq}]},
				{R, Y, Phi} = cameraPosition;
				cameraP = Re[{R*Cos[Phi], Y, R*Sin[Phi]}];
				camera3 = Normalize[cameraTarget - cameraP];
				camera4 = (imgWidth*0.5135/imgHeight)*Normalize[Cross[camera3, {0, 1, 0}]];
					
				hCamera = 1.0 * {
					cameraP,
					cameraTarget,
					camera3,
					camera4,
					0.5135 * Normalize[Cross[camera4, camera3]]
				};
				If[(perfGoal === "Speed") || (perfGoal === Automatic && controlActive),
					hRenderingConfig = Flatten[{10.0*precision, hCamera, parameters, lights}],
					hRenderingConfig = Flatten[{precision, hCamera, parameters, lights}]
				];

				dRenderingConfig = OpenCLMemoryLoad[hRenderingConfig, "Float", "Platform"->platform, "Device"->device, "TargetPrecision"->targetPrecision];
				res = Check[
					If[(perfGoal === "Quality") || (perfGoal === Automatic && controlActive === False),
						implicitFunction[dImgData, dRenderingConfig, numLights, imgWidth, imgHeight, imgWidth*imgHeight],
						implicitFunction[dImgDataSmall, dRenderingConfig, numLights, imgWidthSmall, imgHeightSmall, imgWidthSmall*imgHeightSmall]
					],
					$Failed
				];
				
				Assert[res =!= $Failed];
					
				OpenCLMemoryUnload[dRenderingConfig];
				
				(*hImgData = OpenCLRunGetMemory[dImgData];*)
				If[res === $Failed,
					$Failed,
					If[(perfGoal === "Quality") || (perfGoal === Automatic && controlActive === False),
						hImgData = OpenCLMemoryGet[dImgData];
						Image[hImgData],
						hImgDataSmall = OpenCLMemoryGet[dImgDataSmall];
						ImageResize[Image[hImgDataSmall], imgWidth]
					]
				]
			];
		
		If[singleFrameQ,
			singleFrameOutput = RenderIsosurface[singleFrameParams[[1]], singleFrameParams[[2]], singleFrameParams[[3]], "ControlActiveQ" -> False];
			If[MemberQ[{Automatic, "Speed"}, perfGoal], OpenCLMemoryUnload[dImgDataSmall]];
			OpenCLMemoryUnload[dImgData];
			Return[singleFrameOutput];
		];
		
		testRun = RenderIsosurface[{5.0, 0.0, 0.0}, {0.5, -0.5, 0.5, 0.02}, {15.0, 10.0, 5.0}, "ControlActiveQ" -> False];
		If[testRun === $Failed,
			Return[$Failed]
		];
		DynamicModule[{},
			Manipulate[
				RenderIsosurface[{r, camy, phi}, {paramw}, {lx, ly, lz}, "ControlActiveQ" -> ControlActive[]],
				Style["Camera Position (Cylindrical)", 12, Bold],
				{{r, 3.0*boundingRad, "r"}, 0.0, 6.0*boundingRad, boundingRad/20.0},
				{{camy, 0.0, "y"}, -2.0*boundingRad, 2.0*boundingRad, boundingRad/20.0},
				{{phi, 0.0, "\[Phi]"}, -2.0*Pi, 2.0*Pi, 0.1},
				Delimiter,
				Style["Parameter", 12, Bold],
				{{paramw, sliderParams[[3]], "w"}, sliderParams[[1]], sliderParams[[2]], (sliderParams[[2]]-sliderParams[[1]])/100.0},
				ControlPlacement -> Left,
				Alignment -> Center,
				Deployed -> True,
				Deinitialization :> {
					OpenCLMemoryUnload[dImgData];
					If[MemberQ[{Automatic, "Speed"}, perfGoal], OpenCLMemoryUnload[dImgDataSmall]];
				},
				SynchronousUpdating->False
			]
		]
	]

(******************************************************************************)
(******************************************************************************)
(******************************************************************************)

(* ::Section:: *)
(* SymbolicGPU *)

SymbolicOpenCLFunction[args___] := GPUKernelFunction["OpenCL", args]
SymbolicOpenCLThreadIndex[args___] := GPUKernelThreadIndex["OpenCL", args]
SymbolicOpenCLBlockIndex[args___] := GPUKernelBlockIndex["OpenCL", args]
SymbolicOpenCLBlockDimension[args___] := GPUKernelBlockDimension["OpenCL", args]
SymbolicOpenCLCalculateKernelIndex[args___] := GPUCalculateKernelIndex["OpenCL", args]
SymbolicOpenCLDeclareIndexBlock[args___] := GPUDeclareIndexBlock["OpenCL", args]
SymbolicOpenCLKernelIndex[args___] := GPUKernelIndex["OpenCL", args]
	

(******************************************************************************)
(******************************************************************************)

(* ::Section:: *)
(* Error Messages *)

OpenCLLink::invpdid = "Specified \"Platform\"->`1` and \"Device\"->`2` are not valid platforms or devices."

GPUTools`Utilities`DefineMessage[
	{OpenCLFunctionLoad, OpenCLMemoryAllocate, OpenCLMemoryLoad, OpenCLMemoryGet, OpenCLMemoryUnload},
	{
		{"invtype", "Encountered an invalid type. Only Lists of Reals/Integers, Reals, and Integers are supported."},
		{"mixdinpttyp", "Input list cannot contain mixed types and must be packable."},
		{"invtrgtprs", "\"TrargetPrecision\"->`1` is not a valid option value."},
		{"unsupdbl", "Device does not support double precision computation."},
		{"invunq", "\"Unique\"->`1` is not a valid option value."},
		{"cnffrs", "\"Unique\"->`2` conflicts with specified property `1`"},
		{"unbladd", "Unable to add memory."},
		{"reinit", "Attempting to reinitialize to a different platform and device."},
		{"reinitplt", "Attempting to reinitialize platform `1` with platform `2`, by setting \"Platform\"->`2`."},
		{"reinitdev", "Attempting to reinitialize device `1` with device `2`, by setting \"Device\"->`2`."}
	}
]

GPUTools`Utilities`DefineMessage[
	{OpenCLFunctionLoad},
	{
		{"invsrc", "Encountered invalid source input. The source input must be either a string containing the program, or a list of one element indicating the file containing the program."},
		{"emptin", "Input parameters cannot be empty."},
		{"emtout", "Output parameters cannot be empty."},
		{"invcmpopts", "\"CompileOptions\"->`1` are not valid compile options."},
		{"invsyncin", "\"SynchronizeInputs\"->`1` is not a valid option value."},
		{"invsyncout", "\"SynchronizeOutputs\"->`1` is not a valid option value."},
		{"def", "Definition specification \"Defines\" -> `1` is not a string, nor is it a list of strings and rules with right-hand sides that are strings or integers."},
		{"dirlist", "Directory list specification \"`1`\" -> `2` is not a string or list of strings."},
		{"emptsrc", "The source string specified is empty."},
		{"invsrcf", "The source code file `1` specified is not valid."},
		{"invws", "The local work size `1` specified is not valid."},
		{"invgws", "The global work group size `1` specified is not valid."},
		{"invpres", "\"Persistent\"->`1` is not a valid option value."},
		{"invinpt", "Invalid input parameters."},
		{"invout", "Invalid output parameters."},
		{"outmem", "Computation ran out of memory, it required `1` bytes of memory, but only `2` are available on system."},
		{"cmpf", "The kernel compilation failed. Consider setting the option \"ShellOutputFunction\"->Print to display the compiler error message."},
		{"nonmchint", "Input `1` is not a machine integer."},
		{"nonmchre", "Input `1` is not a machine real."},
		{"invlocaltyp", "Local type `1` specified is not valid. Types can be either Integer or Real"},
		{"invlocal", "Local memory size is not a machine sized integer."}
	}
]


GPUTools`Utilities`DefineMessage[
	{OpenCLInformation},
	{
		{"stdoin", "Specified platform query `1` is not valid."},
		{"strin", "Specified device query `1` is not valid."},
		{"invopt", "Invalid option."}
	}
]

GPUTools`Utilities`DefineMessage[
	{OpenCLLibraryGenerate, OpenCLLibraryFunctionGenerate},
	{
		(* CCompilerDriver messages *)
		{"nocomp", CreateLibrary::nocomp},
		{"badcomp", CreateLibrary::badcomp},
		{"compnmtype", CreateLibrary::compnmtype},
		{"instl", CreateLibrary::instl},
		{"instltype", CreateLibrary::instltype},
		{"compopt", CreateLibrary::compopt},
		{"targettype", CreateLibrary::targettype},
		{"target", CreateLibrary::target},
		{"tgtdir", CreateLibrary::tgtdir},
		{"wdtype", CreateLibrary::wdtype},
		{"cleantype", CreateLibrary::cleantype},
		{"def", CreateLibrary::def},
		{"dirlist", CreateLibrary::dirlist},
		{"sysdirlist", CreateLibrary::cleantype},
		{"filelist", CreateLibrary::filelist},
		{"sysfilelist", CreateLibrary::sysfilelist},
		{"crbin", CreateLibrary::crbin},
		{"mpreptype", CreateLibrary::mpreptype},
		{"precompiletype", CreateLibrary::precompiletype},
		{"postcompiletype", CreateLibrary::postcompiletype},
		{"debug", CreateLibrary::debug}
	}
]

GPUTools`Utilities`DefineMessage[
	{OpenCLFractalRender3D},
	{
		{"invftype", "Fractal type `1` is not a supported type of fractal. Valid types are \"Mandelbrot\" and \"Julia\"."},
		{"infiif", "Supplied Iterate Intersect function `1` is invalid."},
		{"invhw", "Invalid input dimensions. Width set to `1` and Height set to `1` must be a multiple of 64."},
		{"invfmed", "The Method `1` specified is not valid. Valid methods are \"Quaternion\", \"Triplex\", and \"Custom\"."},
		{"invclr", "The \"Color\" option set to `1` is not valid. \"Color\" must be a vector of Reals."},
		{"invspece", "The \"SpecularExponent\" option set to `1` is not valid. \"SpecularExponent\" must be a Real."},
		{"invspec", "The \"Specularity\" option set to `1` is not valid. \"Specularity\" must be a Real between 0 and 1."},
		{"invshdw", "The \"Shadows\" option set to `1` is not valid. \"Shadows\" must be True or False."},
		{"invflr", "The \"Floor\" option set to `1` is not valid. \"Floor\" must be True, False, or a Real."},
		{"invmaxit", "The \"MaxIterations\" option set to `1` is not valid. \"MaxIterations\" must be positive integer."},
		{"invprec", "The \"Precision\" option set to `1` is not valid. \"Precision\" must be a Real."},
		{"invmsmp", "The \"Multisampling\" option set to `1` is not valid. \"Multisampling\" must be True, False or a positive integer."},
		{"invbrad", "The \"BoundingRadius\" option set to `1` is invalid. \"BoundingRadius\" must be a positive real number."},
		{"invsfrm", "The \"SingleFrame\" option set to `1` is invalid. \"SingleFrame\" must be True or False."},
		{"invfrprms", "The \"SingleFrameParams\" option set to `1` is invalid. \"SingleFrameParams\" should be a triplet specifying camera location, fractal parameters, and light location."},
		{"invsldr", "The \"SliderParameters\" option set to `1` is invalid. \"SliderParameters\" should be a quartet in which each element specifies a lower bound, upper bound, initial value, and step size for the sliders."}
	}
]
GPUTools`Utilities`DefineMessage[
	{OpenCLMersenneTwister},
	{
		{"invfile", "The \"SeedFile\" option set to `1` is invalid. \"SeedFile\" must be a string containing the location of a file."},
		{"invseed", "The \"SeedValue\" option set to `1` is invalid. \"SeedValue\" must be a machine integer."}
	}
]
GPUTools`Utilities`DefineMessage[
	{OpenCLImplicitRender3D},
	{
		{"invflr", "The \"Floor\" or \"FloorPosition\" option set to `1` is invalid. \"Floor\" must be True or False, and \"FloorPosition\" must be a real number."},
		{"invshdw", "The \"Shadows\" option set to `1` is invalid. \"Shadows\" must be True or False."}
	}
]	


GPUTools`Utilities`DefineMessage[
	$Symbols,
	{
		{"unspecplt", "The \"Device\" option was set without specifying the \"Platform\" option."}
	}
]

(******************************************************************************)
(******************************************************************************)

SetAttributes[OpenCLProfile, HoldAll]
SetAttributes[Evaluate@$Symbols, {ReadProtected, Protected}]


(******************************************************************************)
(******************************************************************************)


End[] (* End Private Context *)

EndPackage[]




