(******************************************************************************)

Needs["GPUTools`Detection`"]

(******************************************************************************)

If[Unevaluated[$WGLPlatform] === $WGLPlatform,
	$WGLPlatform = Automatic
];

If[Unevaluated[$WGLDevice] === $WGLDevice,
	$WGLDevice = Automatic
];

(******************************************************************************)

$APIVersion = StringJoin[Riffle[ToString /@ {IntegerPart@$VersionNumber, IntegerPart[Ceiling[10 * FractionalPart@$VersionNumber]], $ReleaseNumber, $MinorReleaseNumber}, "."]]
$SingleOrDoubleLibrary="Double"
$LibraryExtensions =
	Switch[$OperatingSystem,
		"Windows",
			"dll",
		"Unix",
			"so",
		"MacOSX",
			"dylib"
	]

(******************************************************************************)

$WGLLibraryPath[api:("OpenCL" | "CUDA"), "Single"] :=
	"Unsupported" 
$WGLLibraryPath[api:("OpenCL" | "CUDA"), singleOrDouble_] := 
    If[$OperatingSystem === "Windows", "", "lib"] <> api <> "Link_" <> singleOrDouble <> "." <> $LibraryExtensions

WGLLibraryLoadedQ[] := !WGLLibraryNotLoadedQ[]
WGLLibraryNotLoadedQ[] := $WGLLibraryLoadStatus === Unevaluated[$WGLLibraryLoadStatus]

ExportedFunction[funName_, singleOrDouble_] := $ThisAPI <> "_" <> funName <> "_" <> singleOrDouble

$WGLBaseFunctions[libraryPath_String, singleOrDouble_:"Single"] := $WGLBaseFunctions[libraryPath, singleOrDouble] = {
	{cInitializeWolframGPULibrary[singleOrDouble], libraryPath, ExportedFunction["WolframGPULibrary_initialize", singleOrDouble], {"UTF8String"}, "Boolean"},
	{cGetGPUInformation[singleOrDouble], libraryPath, ExportedFunction["oGetGPUInformation", singleOrDouble], {"UTF8String", _Integer, _Integer}, {_, 1, "Shared"}},
	{cGetGPUInformationType[singleOrDouble], libraryPath, ExportedFunction["oGetGPUInformationType", singleOrDouble], {"UTF8String"}, "UTF8String"},
	{cGetCurrentMemoryUsage[singleOrDouble], libraryPath, ExportedFunction["oGetCurrentMemoryUsage", singleOrDouble], {}, {_, 1, Automatic}},
	{cGetMaximumUsableMemory[singleOrDouble], libraryPath, ExportedFunction["oGetMaximumUsableMemory", singleOrDouble], {}, {_, 1, Automatic}},
	{cSuccessQ[singleOrDouble], libraryPath, ExportedFunction["oSuccessQ", singleOrDouble], {}, "Boolean"},
	{cGetErrorMessage[singleOrDouble], libraryPath, ExportedFunction["oGetErrorMessage", singleOrDouble], {}, "UTF8String"},
	{cSetPlatform[singleOrDouble], libraryPath, ExportedFunction["oSetPlatform", singleOrDouble], {_Integer}, _Integer},
	{cSetDevice[singleOrDouble], libraryPath, ExportedFunction["oSetDevice", singleOrDouble], {_Integer}, _Integer},
	{cAddProgramFromSource[singleOrDouble], libraryPath, ExportedFunction["oAddProgramFromSource", singleOrDouble], {"UTF8String", "UTF8String"}, _Integer},
	{cAddProgramFromSourceFile[singleOrDouble], libraryPath, ExportedFunction["oAddProgramFromSourceFile", singleOrDouble], {"UTF8String", "UTF8String"}, _Integer},
	{cAddProgramFromBinary[singleOrDouble], libraryPath, ExportedFunction["oAddProgramFromBinary", singleOrDouble], {"UTF8String"}, _Integer},
	{cAddProgramFromBinaryFile[singleOrDouble], libraryPath, ExportedFunction["oAddProgramFromBinaryFile", singleOrDouble], {"UTF8String"}, _Integer},
	{cSetDeleteProgramFileOnExit[singleOrDouble], libraryPath, ExportedFunction["oSetDeleteProgramFileOnExit", singleOrDouble], {"UTF8String"}, _Integer},
	{cSetKernelFunction[singleOrDouble], libraryPath, ExportedFunction["oSetKernelFunction", singleOrDouble], {_Integer}, "Void"},
	{cSetKernel[singleOrDouble], libraryPath, ExportedFunction["oSetKernel", singleOrDouble], {"UTF8String"}, "Void"},
	{cGetProgramInformation[singleOrDouble], libraryPath, ExportedFunction["oGetProgramInformation", singleOrDouble], {_Integer, "UTF8String", "UTF8String", "UTF8String"}, "UTF8String"},
	{cAddMemory[singleOrDouble], libraryPath, ExportedFunction["oAddMemory", singleOrDouble], {{_, _, "Shared"}, "UTF8String", "UTF8String", "UTF8String", "Boolean"}, _Integer},
	{cAllocateMemory[singleOrDouble], libraryPath, ExportedFunction["oAllocateMemory", singleOrDouble], {"UTF8String", {_, _, "Shared"}, "UTF8String"}, _Integer},
	{cGetMemory[singleOrDouble], libraryPath, ExportedFunction["oGetMemory", singleOrDouble], {_Integer, "Boolean"}, {_, _, "Manual"}},
	{cCopyMemory[singleOrDouble], libraryPath, ExportedFunction["oCopyMemory", singleOrDouble], {_Integer, "Boolean", "UTF8String"}, "Void"},
	{oCopyConstantMemory[singleOrDouble], libraryPath, ExportedFunction["oCopyConstantMemory", singleOrDouble], {"UTF8String", _Integer, "UTF8String"}, "Void"},
	{oBindToTexture[singleOrDouble], libraryPath, ExportedFunction["oBindTexture", singleOrDouble], {"UTF8String", _Integer, {_Integer, 1, "Shared"}, "UTF8String", "UTF8String"}, "Void"},
	{cDeleteMemory[singleOrDouble], libraryPath, ExportedFunction["oDeleteMemory", singleOrDouble], {_Integer}, {"Boolean"}},
	{cGetMemoryInformation[singleOrDouble], libraryPath, ExportedFunction["oGetMemoryInformation", singleOrDouble], {_Integer, "UTF8String"}, {"UTF8String"}},
	{cSetKernelCharArgument[singleOrDouble], libraryPath, ExportedFunction["oSetKernelCharArgument", singleOrDouble], {_Integer}, _Integer},
	{cSetKernelUnsignedCharArgument[singleOrDouble], libraryPath, ExportedFunction["oSetKernelUnsignedCharArgument", singleOrDouble], {_Integer}, _Integer},
	{cSetKernelShortArgument[singleOrDouble], libraryPath, ExportedFunction["oSetKernelShortArgument", singleOrDouble], {_Integer}, _Integer},
	{cSetKernelUnsignedShortArgument[singleOrDouble], libraryPath, ExportedFunction["oSetKernelUnsignedShortArgument", singleOrDouble], {_Integer}, _Integer},
	{cSetKernelIntegerArgument[singleOrDouble], libraryPath, ExportedFunction["oSetKernelIntegerArgument", singleOrDouble], {_Integer}, _Integer},
	{cSetKernelUnsignedIntegerArgument[singleOrDouble], libraryPath, ExportedFunction["oSetKernelUnsignedIntegerArgument", singleOrDouble], {_Integer}, _Integer},
	{cSetKernelLongArgument[singleOrDouble], libraryPath, ExportedFunction["oSetKernelLongArgument", singleOrDouble], {_Integer}, _Integer},
	{cSetKernelFloatArgument[singleOrDouble], libraryPath, ExportedFunction["oSetKernelFloatArgument", singleOrDouble], {_Real}, _Integer},
	{cSetKernelDoubleArgument[singleOrDouble], libraryPath, ExportedFunction["oSetKernelDoubleArgument", singleOrDouble], {_Real}, _Integer},
	{cSetKernelMemoryArgument[singleOrDouble], libraryPath, ExportedFunction["oSetKernelMemoryArgument", singleOrDouble], {_Integer, "UTF8String"}, _Integer},
	{cSetLocalMemoryArgument[singleOrDouble], libraryPath, ExportedFunction["oSetLocalMemoryArgument", singleOrDouble], {_Integer}, "Void"},
	{cLaunchKernel[singleOrDouble], libraryPath, ExportedFunction["oLaunchKernel", singleOrDouble], {}, "Boolean"},
	{cSetBlockDimensions[singleOrDouble], libraryPath, ExportedFunction["oSetBlockDimensions", singleOrDouble], {_Integer, _Integer, _Integer, _Integer}, "Boolean"},
	{cSetGridDimensions[singleOrDouble], libraryPath, ExportedFunction["oSetGridDimensions", singleOrDouble], {_Integer, _Integer, _Integer, _Integer}, "Boolean"},
	{cShareWolframGPULibraryData[singleOrDouble], libraryPath, ExportedFunction["oShareWolframGPULibraryData", singleOrDouble], {"UTF8String"}, "Boolean"},
	{cEnableProfiler[singleOrDouble], libraryPath, ExportedFunction["oEnableProfiler", singleOrDouble], {}, "Void"},
	{cDisableProfiler[singleOrDouble], libraryPath, ExportedFunction["oDisableProfiler", singleOrDouble], {}, "Void"},
	{cClearProfilerTable[singleOrDouble], libraryPath, ExportedFunction["oClearProfilerTable", singleOrDouble], {}, "Void"},
	{cGetProfileTableElementCount[singleOrDouble], libraryPath, ExportedFunction["oGetProfileTableElementCount", singleOrDouble], {}, _Integer},
	{cGetProfileElementInformation[singleOrDouble], libraryPath, ExportedFunction["oGetProfileElementInformation", singleOrDouble], {_Integer, "UTF8String"}, "UTF8String"},
	{cGetProfileElementTime[singleOrDouble], libraryPath, ExportedFunction["oGetProfileElementTime", singleOrDouble], {_Integer, "UTF8String"}, {_Integer, _, Automatic}},
	{cGetProfileElementTimeByID[singleOrDouble], libraryPath, ExportedFunction["oGetProfileElementTimeByID", singleOrDouble], {_Integer, "UTF8String"}, {_Integer, _, Automatic}}
}

InitializationErrorMessage[errMsgHd_, {tag_}] :=
	Message[MessageName[errMsgHd, tag]]
InitializationErrorMessage[errMsgHd_, {tag_, args__}] :=
	Message[MessageName[errMsgHd, tag], args]

FailReason[errMsgHd_] :=
	If[Developer`$ProtectedMode,
		Message[errMsgHd::sndbx],
		If[ListQ[$InitializeWGLError] && $WGLLibraryLoadStatus === $Failed,
			Throw[InitializationErrorMessage[errMsgHd, $InitializeWGLError]; $Failed],
			If[$ThisAPI === "CUDA",
				GPUTools`Internal`CUDAQueryFailReason[errMsgHd],
				GPUTools`Internal`OpenCLQueryFailReason[errMsgHd]
			]
		]
	]

InitializeWGL[errMsgHd_] /; WGLLibraryLoadedQ[] :=
	$WGLLibraryLoadStatus
InitializeWGL[errMsgHd_] /; (!WGLLibraryLoadedQ[]):=
	InitializeWGL[errMsgHd, $WGLPlatform, $WGLDevice]
InitializeWGL[errMsgHd_, platform:(_Integer | Automatic), device:(_Integer | Automatic), singleOrDouble_:"Double"] /; MemberQ[{"Single", "Double"}, singleOrDouble] :=
	Module[{loadLibs, sandboxQ, libraryPath, initWGL, libraryFuncs, defaultSystemOptions},
		
		$SystemResourcesPath;
		FailReason[errMsgHd, False];
		
		If[Quiet[TrueQ[GPUTools`Utilities`IsSandboxedQ[errHd] ||
		   !TrueQ[GPUTools`Internal`ValidSystemQ[]] ||
		   ($ThisAPI === "CUDA" && GPUTools`Internal`$IsLegacyGPU === True)]],
			Throw[$Failed]	
		];
		
		libraryPath = FindLibrary[$WGLLibraryPath[$ThisAPI, singleOrDouble]];
		
		`$Library[singleOrDouble] = libraryPath;
		
		If[ListQ[$InitializeWGLError],
			$WGLLibraryLoadStatus = $Failed
		];
		If[WGLLibraryLoadedQ[] && $WGLLibraryLoadStatus === $Failed,
			If[FailReason[errMsgHd],
				$InitializeWGLError = {"initwgl", libraryPath};
				Throw[InitializationErrorMessage[$InitializeWGLError]; $Failed],
				Throw[$Failed]
			]
		];
		
		If[WGLLibraryLoadedQ[],
			Return[$WGLLibraryLoadStatus, Module]
		];
		
		If[$WGLLibraryLoadStatus === $Failed,
			$InitializeWGLError = {"initwgl", libraryPath};
			Throw[InitializationErrorMessage[errMsgHd, $InitializeWGLError]; $Failed]
		];
		
		sandboxQ = Check[Catch[GPUTools`Utilities`IsSandboxedQ[errMsgHd]], True];
		If[sandboxQ,
			Throw[$WGLLibraryLoadStatus = $Failed]
		];

		If[FileNames[libraryPath <> "*"],
			$WGLLibraryLoadStatus = $Failed;
			$InitializeWGLError = {"nolib", libraryPath};
			Throw[InitializationErrorMessage[errMsgHd, $InitializeWGLError]; $Failed]
		];
		
        GPUTools`Utilities`Logger[" ==== Loading Library Files ==== "];
		loadLibs = If[$ThisAPI === "CUDA",
			GPUTools`Internal`LoadCUDALibraries[errMsgHd, $SystemResourcesPath],
			GPUTools`Internal`LoadOpenCLLibraries[errMsgHd]
		];
		If[loadLibs === $Failed,
			$WGLLibraryLoadStatus = $Failed;
			If[FailReason[errMsgHd],
				$InitializeWGLError = {"syslibfld"};
				Throw[InitializationErrorMessage[errMsgHd, $InitializeWGLError]; $Failed],
				Throw[$Failed]
			]
		];
		
		defaultSystemOptions = SystemOptions["DynamicLibraryOptions"]; 	 
        SetSystemOptions["DynamicLibraryOptions" -> 	 
        	{"DynamicLibraryGlobal" -> False, "DynamicLibraryLazy" -> True} 	 
        ];
        GPUTools`Utilities`Logger[" ==== Loading Library Functions ==== "];
		libraryFuncs = GPUTools`Internal`LibraryFunctionSafeLoad[
			Join[
				$WGLBaseFunctions[libraryPath, singleOrDouble],
				$ExtraLibraryFunctions[libraryPath, singleOrDouble]
			]
		];
		SetSystemOptions[defaultSystemOptions];
		
		If[libraryFuncs === $Failed,
			$WGLLibraryLoadStatus = $Failed;
			If[FailReason[errMsgHd],
				$InitializeWGLError = {"libldfld"};
				Throw[InitializationErrorMessage[errMsgHd, $InitializeWGLError]; $Failed],
				Throw[$Failed]
			]
		];
		
        GPUTools`Utilities`Logger[" ==== Initializing ", $ThisAPI, " ==== "];
		initWGL = Quiet[cInitializeWolframGPULibrary[singleOrDouble][$ThisAPI]];
		If[GPUTools`Utilities`ValidLibraryFunctionReturnQ[initWGL],
			$WGLLibraryLoadStatus = True;
			$SingleOrDoubleLibrary = singleOrDouble;
			If[SupportsDoublePrecisionQ[errMsgHd, platform, device] && singleOrDouble === "Single",
        		GPUTools`Utilities`Logger[" ==== Double Precision Support Detected ==== "];
				Clear[$WGLLibraryLoadStatus];
        		GPUTools`Utilities`Logger[" ==== Unloading Library Functions ==== "];
				Quiet[
					LibraryFunctionUnload[First[#]]& /@ Join[
						$WGLBaseFunctions[libraryPath, singleOrDouble],
						$ExtraLibraryFunctions[libraryPath, singleOrDouble]
					]
				];
        		GPUTools`Utilities`Logger[" ==== Done Unloading Library Functions ==== "];
        		GPUTools`Utilities`Logger[" ==== Loading Double Precision Libraries ==== "];
				InitializeWGL[errMsgHd, platform, device, "Double"]
			],
			$WGLLibraryLoadStatus = $Failed;
			If[FailReason[errMsgHd],
				$InitializeWGLError = {"init"};
				Throw[InitializationErrorMessage[errMsgHd, $InitializeWGLError]; $Failed],
				Throw[$Failed]
			]
		]
	]

If[$PlatformInitialized === Unevaluated[$PlatformInitialized],
	$PlatformInitialized = False
]

InitializeWGL[errMsgHd_, Automatic, Automatic] /; !$PlatformInitialized :=
	InitializeWGL[errMsgHd, errMsgHd, "Platform" -> $WGLPlatform, "Device" -> $WGLDevice]
	
InitializeWGL[errMsgHd_, opts:OptionsPattern[]] /; !$PlatformInitialized :=
	InitializeWGL[errMsgHd, errMsgHd, opts]

InitializeWGL[head_, errMsgHd_, opts:OptionsPattern[]] /; !$PlatformInitialized :=
	Module[{plt = GetAndCheckOption[head, errMsgHd, "Platform", {opts}],
			dev = GetAndCheckOption[head, errMsgHd, "Device", {opts}],
			res
		   },
		res = Check[
			InitializeWGL[errMsgHd, plt, dev];
			$WGLPlatform = SetPlatform[plt];
			$WGLDevice = SetDevice[dev];

			If[$ThisAPI === "CUDA",
				CUDALink`$CUDADevice = $WGLDevice,

				OpenCLLink`$OpenCLPlatform = $WGLPlatform;
				OpenCLLink`$OpenCLDevice = $WGLDevice
			],
			$Failed
		];
		$PlatformInitialized = True;
		$WGLLibraryLoadStatus = res;
		If[res === $Failed,
			Throw[res],
			True
		]
	]

(******************************************************************************)

ClearAll[$MessageHead]
$MessageHead = Swich[$ThisAPI,
	"CUDA",
		CUDALink`CUDALink,
	"OpenCL",
		OpenCLLink`OpenCLLink,
	_,
		GPUTools`GPUTools
]
SuccessQ[] := ReportMessage[cSuccessQ[$SingleOrDoubleLibrary][]] 
SuccessQ[x_] := SuccessQ[] && GPUTools`Utilities`ValidLibraryFunctionReturnQ[x]
FailQ[] := !SuccessQ[]
FailQ[x_] := !SuccessQ[x]
SetErrorMessageHead[errMsgHd_] := ($MessageHead = errMsgHd)
ReportMessage[True] = True
ReportMessage[False] :=
	Module[{tag = cGetErrorMessage[$SingleOrDoubleLibrary][]},
		If[tag === "Success",
			True,
			ErrorMessage[$MessageHead, {tag}];
			False
		]
	]

(******************************************************************************)

ShareWolframGPULibraryData[pth_String] :=
	If[MemberQ[GPUTools`Internal`LoadedLibraries, FindLibrary[pth]],
		True,
		With[{res = cShareWolframGPULibraryData[$SingleOrDoubleLibrary][pth]},
			If[SuccessQ[res],
				If[FreeQ[GPUTools`Internal`LoadedLibraries, $WGLLibraryPath[$ThisAPI, $SingleOrDoubleLibrary]],
					GPUTools`Internal`LibraryFunctionSafeLoad[$WGLLibraryPath[$ThisAPI, $SingleOrDoubleLibrary]];
				];
				PrependTo[GPUTools`Internal`LoadedLibraries, FindLibrary[pth]];
				True,
				Throw[$Failed]
			]
		]
	]

SetPlatform[x:(Automatic | _Integer?Positive)] :=
	With[{platform = cSetPlatform[$SingleOrDoubleLibrary][If[x === Automatic, -1, x]]},
		If[SuccessQ[platform],
			platform,
			Throw[$Failed]
		]
	]
	
SetDevice[x:(Automatic | _Integer?Positive)] := 
	With[{device = cSetDevice[$SingleOrDoubleLibrary][If[x === Automatic, -1, x]]},
		If[SuccessQ[device],
			device,
			Throw[$Failed]
		]
	]
	
AddProgramFromSource[errMsgHd_, prog_String, buildOptions_String] :=
	Module[{progid},
		progid = cAddProgramFromSource[$SingleOrDoubleLibrary][prog, buildOptions];
		If[SuccessQ[progid],
			{progid, "Source"},
			Throw[$Failed]
		]
	]
	
AddProgramFromSourceFile[errMsgHd_, {prog_String}, buildOptions_String] :=
	Module[{progid},
		progid = cAddProgramFromSourceFile[$SingleOrDoubleLibrary][prog, buildOptions];
		If[SuccessQ[progid],
			{progid, "SourceFile"},
			Throw[$Failed]
		]
	]
	
AddProgramFromBinary[errMsgHd_, prog_String] :=
	Module[{progid},
		progid = cAddProgramFromBinary[$SingleOrDoubleLibrary][prog];
		If[SuccessQ[progid],
			{progid, "Binary"},
			Throw[$Failed]
		]
	]
	
AddProgramFromBinaryFile[errMsgHd_, {prog_String}] :=
	AddProgramFromBinaryFile[errMsgHd, prog]
AddProgramFromBinaryFile[errMsgHd_, prog_String] :=
	Module[{progid},
		progid = cAddProgramFromBinaryFile[$SingleOrDoubleLibrary][prog];
		If[SuccessQ[progid],
			{progid, "BinaryFile"},
			Throw[$Failed]
		]
	]

AddProgram["ptx" | "cubin" | "PTX" | "CUBIN" | "clb" | "bin"][errMsgHd_, prog_, ___] := 
	If[ListQ[prog],
		AddProgramFromBinaryFile[errMsgHd, prog],
		AddProgramFromBinary[errMsgHd, prog]
	]
AddProgram[ext:("so" | "dll" | "DLL" | "dylib")][errMsgHd_, prog_, functionName_, args_, ___] :=
	Module[{file = First[prog], libArgs = ConvertToLibraryFunctionArguments[args] ~Join~ {{_Integer, 1, Automatic}, {_Integer, 1, Automatic}}, libfun},
		ShareWolframGPULibraryData[file];
		libfun = Catch[GPUTools`Internal`LibraryFunctionSafeLoad[file, functionName, libArgs, _Integer]];
		If[libfun === $Failed,
			Throw[Message[errMsgHd::libload, file, functionName]; $Failed],
			{libfun, "LibraryFunction"}
		]
	]

AddProgram[_][errMsgHd_, prog_, functionName_, args_, buildOptions_] := 
	If[ListQ[prog],
		AddProgramFromSourceFile[errMsgHd, prog, buildOptions],
		AddProgramFromSource[errMsgHd, prog, buildOptions]
	]

SetDeleteProgramFileOnExit[fileName_String] :=
	Module[{res},
		res = Quiet[
			cSetDeleteProgramFileOnExit[$SingleOrDoubleLibrary][fileName]
		];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]
		
SetKernelFunction[progId_Integer] :=
	Module[{res},
		res = cSetKernelFunction[$SingleOrDoubleLibrary][progId];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]

SetKernel[kernelName_String] :=
	Module[{res},
		res = cSetKernel[$SingleOrDoubleLibrary][kernelName];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]
	
GetProgramBuildLog[id_Integer, source_String, buildOptions_String] :=
	Module[{res},
		res = cGetProgramBuildLog[$SingleOrDoubleLibrary][id, source, buildOptions];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]

AddConstantMemory[memoryHd_, errMsgHd_, symbol_, type_String, info_List] :=
	memoryHd[
		symbol,
		"Type" -> type,
		"Dimensions" -> Indeterminate,
		"ByteCount" -> Indeterminate,
		"Residence" -> "DeviceOnly",
		"Sharing" -> Constant,
		"Unique" -> True,
		"Platform" -> $WGLPlatform,
		"Device" -> $WGLDevice,
		"MathematicaType" -> First[info],
		"TypeInfromation" -> Flatten[info[[2;;]]]
	]
		
AddMemory[memoryHd_, errMsgHd_, data_, type_String, info_List, residence_String, sharing_String, unique:(True | False)] :=
	Module[{res, tensor},

		tensor = Developer`ToPackedArray[data, GetTensorType[type]];
		If[!Developer`PackedArrayQ[tensor] || !ListQ[tensor],
			Throw[Message[errMsgHd::unpkd, data]; $Failed];
		];
		
		res = cAddMemory[$SingleOrDoubleLibrary][tensor, type, residence, sharing, unique];
		If[SuccessQ[res],
			memoryHd[
				res,
				"Type" -> type,
				"Dimensions" -> GPUTools`Utilities`ArrayDim[tensor],
				"ByteCount" -> ResolveByteCount[type] * Apply[Times, GPUTools`Utilities`ArrayDim[tensor]],
				"Residence" -> residence,
				"Sharing" -> sharing,
				"Unique" -> unique,
				"Platform" -> $WGLPlatform,
				"Device" -> $WGLDevice,
				"MathematicaType" -> First[info],
				"TypeInfromation" -> Flatten[info[[2;;]]]
			],
			Throw[$Failed]
		]
	]
	
AllocateMemory[memoryHd_, errMsgHd_, type_, dims0_List, residence_String, usingSinglePrecisionQ_] :=
	Module[{res, dims},

		dims = Developer`ToPackedArray[dims0];
		If[!Developer`PackedArrayQ[dims],
			Throw[Message[errMsgHd::unpkd, dims0]; $Failed];
		];
		If[!VectorQ[dims, Positive],
			Throw[Message[errMsgHd::dims, dims]; $Failed]
		];
		res = cAllocateMemory[$SingleOrDoubleLibrary][type, dims, residence];
		If[SuccessQ[res],
			memoryHd[
				res,
				"Type" -> type,
				"Dimensions" -> dims,
				"ByteCount" -> ResolveByteCount[type] * Apply[Times, dims],
				"Residence" -> residence,
				"Sharing" -> "Shared",
				"Unique" -> True,
				"Platform" -> $WGLPlatform,
				"Device" -> $WGLDevice,
				"MathematicaType" -> List,
				"TypeInfromation" -> {}
			],
			Throw[$Failed]
		]
	]

GetMemory[id_Integer, forceQ_] :=
	Module[{res},
		res = cGetMemory[$SingleOrDoubleLibrary][id, forceQ];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]
	
CopyMemory[id_Integer, force:(True | False), direction:("ToHost" | "ToDevice")] :=
	Module[{res},
		res = cCopyMemory[$SingleOrDoubleLibrary][id, force, direction];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]
	
CopyConstantMemory[symbol_, id_, direction:("To" | "From")] :=
	Module[{res},
		res = oCopyConstantMemory[$SingleOrDoubleLibrary][symbol, id, direction];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]

BindToTexture[symbol_, id_, addressMode_, filterMode_, flags_] :=
	Module[{res},
		res = oBindToTexture[$SingleOrDoubleLibrary][symbol, id, Developer`ToPackedArray[addressMode], filterMode, flags];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]
	
DeleteMemory[id_Integer] :=
	Module[{res},
		res = cDeleteMemory[$SingleOrDoubleLibrary][id];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]
DeleteMemory[x___] := Throw[Message[DeleteMemory::mdel, {x}]; $Failed]
	
SetKernelIntegerArgument[x_Integer] :=
	Module[{res},
		res = cSetKernelIntegerArgument[$SingleOrDoubleLibrary][x];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]
	
SetKernelUnsignedIntegerArgument[x_Integer] :=
	Module[{res},
		res = cSetKernelUnsignedIntegerArgument[$SingleOrDoubleLibrary][x];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]
	
SetKernelLongArgument[x_Integer] :=
	Module[{res},
		res = cSetKernelLongArgument[$SingleOrDoubleLibrary][x];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]

SetKernelFloatArgument[x_Real] :=
	Module[{res},
		res = cSetKernelFloatArgument[$SingleOrDoubleLibrary][x];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]

SetKernelDoubleArgument[x_Real] :=
	Module[{res},
		res = cSetKernelDoubleArgument[$SingleOrDoubleLibrary][x];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]

SetKernelMemoryArgument[id_Integer, inputOutpuParam_String] :=
	Module[{res},
		res = cSetKernelMemoryArgument[$SingleOrDoubleLibrary][id, inputOutpuParam];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]

SetKernelLocalMemoryArgument[size_Integer] :=
	Module[{res},
		res = cSetLocalMemoryArgument[$SingleOrDoubleLibrary][size];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]

SetGridDimensions[gridDimensions_List] :=
	Module[{res},
		res = cSetGridDimensions[$SingleOrDoubleLibrary][
			Length[gridDimensions],
			First[gridDimensions],
			If[Length[gridDimensions] > 1,
				gridDimensions[[2]],
				1
			],
			If[Length[gridDimensions] > 2,
				gridDimensions[[3]],
				1
			]
		];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]

SetBlockDimensions[blockDimensions_List] :=
	Module[{res},
		res = cSetBlockDimensions[$SingleOrDoubleLibrary][
			Length[blockDimensions],
			First[blockDimensions],
			If[Length[blockDimensions] > 1,
				blockDimensions[[2]],
				1
			],
			If[Length[blockDimensions] > 2,
				blockDimensions[[3]],
				1
			]
		];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]

LaunchKernel[] :=
	Module[{res},
		res = cLaunchKernel[$SingleOrDoubleLibrary][];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]

GetProgramBuildLog[singleOrDouble_][args__] :=
	Module[{res},
		res = cGetProgramInformation[singleOrDouble][args, "BuildLog"];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]

GetProgramBuildOptions[singleOrDouble_][args__] :=
	Module[{res},
		res = cGetProgramInformation[singleOrDouble][args, "BuildOptions"];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]
	
GetProgramInputFile[singleOrDouble_][args__] :=
	Module[{res},
		res = cGetProgramInformation[singleOrDouble][args, "InputFile"];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]
	
GetProgramSource[singleOrDouble_][args__] :=
	Module[{res},
		res = cGetProgramInformation[singleOrDouble][args, "ProgramSource"];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]

GetProgramInformation[type:("BinaryFile" | "Binary")][program:(_[_Integer, ___])] :=
	Join[
		{
			"Source" -> WGLProgramGetSource[program]
		},
		GetProgramInformation[type][WGLProgramGetID[program]]
	]

GetProgramInformation[type_][program:(_[_Integer, ___])] :=
	GetProgramInformation[type][WGLProgramGetID[program]]
	
GetProgramInformation[type_][progId_Integer] :=
	Module[{res},
		res = Check[Catch[{
			"ID" -> progId,
			"BuildLog" -> StringTrim@GetProgramBuildLog[$SingleOrDoubleLibrary][progId, "", ""],
			"BuildOptions" -> StringTrim@GetProgramBuildOptions[$SingleOrDoubleLibrary][progId, "", ""],
			If[MemberQ[{"BinaryFile", "Binary"}, type], "BinaryFile", "File"] -> StringTrim@GetProgramInputFile[$SingleOrDoubleLibrary][progId, "", ""],
			"Source" -> StringTrim@GetProgramSource[$SingleOrDoubleLibrary][progId, "", ""]
		}], $Failed];

		If[SuccessQ[res],
			Select[res, Last[#] =!= ""&],
			Throw[$Failed]
		]
	]

cGetMemoryHostStatus[sd_][args__] := 
	With[{res = cGetMemoryInformation[sd][args, "HostStatus"]},
		If[SuccessQ[res],
			StringTrim[res],
			Throw[$Failed]
		]
	]
cGetMemoryDeviceStatus[sd_][args__] :=
	With[{res = cGetMemoryInformation[sd][args, "DeviceStatus"]},
		If[SuccessQ[res],
			StringTrim[res],
			Throw[$Failed]
		]
	]
cGetMemoryResidence[sd_][args__] :=
	With[{res = cGetMemoryInformation[sd][args, "Residence"]},
		If[SuccessQ[res],
			StringTrim[res],
			Throw[$Failed]
		]
	]
cGetMemorySharing[sd_][args__] :=
	With[{res = cGetMemoryInformation[sd][args, "Sharing"]},
		If[SuccessQ[res],
			StringTrim[res],
			Throw[$Failed]
		]
	]

MemoryInformation[memId_Integer] :=
	Module[{res},
		res = Check[{
			"ID" -> memId,
			"HostStatus" -> cGetMemoryHostStatus[$SingleOrDoubleLibrary][memId],
			"DeviceStatus" -> cGetMemoryDeviceStatus[$SingleOrDoubleLibrary][memId],
			"Residence" -> cGetMemoryResidence[$SingleOrDoubleLibrary][memId],
			"Sharing" -> cGetMemorySharing[$SingleOrDoubleLibrary][memId]
		}, $Failed];
		If[SuccessQ[res],
			Select[res, Last[#] =!= ""&],
			Throw[$Failed]
		]
	]
	
	
EnableProfiler[] :=
	Module[{res},
		res = cEnableProfiler[$SingleOrDoubleLibrary][];
		If[SuccessQ[res],
			True,
			Throw[$Failed]
		]
	]
	
DisableProfiler[] :=
	Module[{res},
		res = cDisableProfiler[$SingleOrDoubleLibrary][];
		If[SuccessQ[res],
			True,
			Throw[$Failed]
		]
	]
	
ClearProfiler[] :=
	Module[{res},
		res = cClearProfilerTable[$SingleOrDoubleLibrary][];
		If[SuccessQ[res],
			True,
			Throw[$Failed]
		]
	]
	
GetProfilerInformation[] :=
	Module[{res},
		res = cGetProfileTableElementCount[$SingleOrDoubleLibrary][];
		If[SuccessQ[res],
			Table[GetProfileElementInformation[ii], {ii, res}],
			{}
		]
	]

GetProfileElementInformation[index_Integer] :=
	Module[{res, granularity, start, stop},
		res = Check[{
			granularity = ToExpression[cGetProfileElementInformation[$SingleOrDoubleLibrary][index, "Granularity"]];
			"ID" -> ToExpression[cGetProfileElementInformation[$SingleOrDoubleLibrary][index, "ID"]],
			"Category" -> cGetProfileElementInformation[$SingleOrDoubleLibrary][index, "Category"],
			"Message" -> cGetProfileElementInformation[$SingleOrDoubleLibrary][index, "Message"],
			"Function" -> cGetProfileElementInformation[$SingleOrDoubleLibrary][index, "Function"],
			"File" -> cGetProfileElementInformation[$SingleOrDoubleLibrary][index, "File"],
			"ParentID" -> ToExpression[cGetProfileElementInformation[$SingleOrDoubleLibrary][index, "ParentID"]],
			"LineNumber" -> ToExpression[cGetProfileElementInformation[$SingleOrDoubleLibrary][index, "LineNumber"]],
			"Granularity" -> granularity,
			"Start" -> (start = formatTime[granularity, cGetProfileElementTime[$SingleOrDoubleLibrary][index, "Start"]]),
			"End" -> (stop = formatTime[granularity, cGetProfileElementTime[$SingleOrDoubleLibrary][index, "End"]]),
			"Elapsed" -> If[stop < start, 0.0, stop - start]
		}, $Failed];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]

addonTime[] := addonTime[] = AbsoluteTime[{1970}]
formatTime[granularity_, {sec_, fracSec_}] := N[(sec + addonTime[]) + (fracSec / granularity)]
formatTime[___] := $Failed

(******************************************************************************)

(* WGLMemory[id, type, dimensions, byteCount, residence, sharing, uniqueQ, platform, device] *)

ErrorMessage[errMsgHd_, {tag_}] :=
	Message[MessageName[errMsgHd, tag]]
ErrorMessage[errMsgHd_, {tag_, args__}] :=
	Message[MessageName[errMsgHd, tag], args]

APIMessage[tag_, args___] /; $ThisAPI === "CUDA" :=
	ErrorMessage[CUDALink`CUDALink, {tag, args}]
APIMessage[tag_, args___] /; $ThisAPI === "OpenCL" :=
	ErrorMessage[OpenCLLink`OpenCLLink, {tag, args}]

WGLMemoryGetID[_[id_Integer, ___]] := id
WGLMemoryGetID[args___] := Throw[APIMessage["getid", args]; $Failed]

WGLMemoryGetConstantSymbol[_[symbol_String, ___]] := symbol
WGLMemoryGetConstantSymbol[args___] := Throw[APIMessage["getsym", args]; $Failed]

WGLMemoryGetType[_[id_Integer, args___]] := ("Type" /. {args})
WGLMemoryGetType[args___] := Throw[APIMessage["gettyp", args]; $Failed]

WGLMemoryGetDimensions[_[id_Integer, args___]] := ("Dimensions" /. {args})
WGLMemoryGetDimensions[args___] := Throw[APIMessage["getdims", args]; $Failed]

WGLMemoryGetRank[mem:(_[id_Integer, args___])] := Length[WGLMemoryGetDimensions[mem]]
WGLMemoryGetRank[args___] := Throw[APIMessage["getrank", args]; $Failed]

WGLMemoryGetByteCount[_[id_Integer, args___]] := ("ByteCount" /. {args})
WGLMemoryGetByteCount[args___] := Throw[APIMessage["getbytc", args]; $Failed]

WGLMemoryGetResidence[_[id_Integer, args___]] := ("Residence" /. {args})
WGLMemoryGetResidence[args___] := Throw[APIMessage["getres", args]; $Failed]

WGLMemoryGetSharing[_[id_Integer, args___]] := ("Sharing" /. {args})
WGLMemoryGetSharing[args___] := Throw[APIMessage["getshr", args]; $Failed]

WGLMemoryGetUniqueQ[_[id_Integer, args___]] := ("Unique" /. {args})
WGLMemoryGetUniqueQ[args___] := Throw[APIMessage["getunq", args]; $Failed]

WGLMemoryGetPlatform[_[id_Integer, args___]] := ("Platform" /. {args})
WGLMemoryGetPlatform[args___] := Throw[APIMessage["getplt", args]; $Failed]

WGLMemoryGetDevice[_[id_Integer, args___]] := ("Device" /. {args})
WGLMemoryGetDevice[args___] := Throw[APIMessage["getdev", args]; $Failed]

WGLMemoryGetMathematicaType[_[id_Integer, args___]] := ("MathematicaType" /. {args})
WGLMemoryGetMathematicaType[args___] := Throw[APIMessage["getmtyp", args]; $Failed]

WGLMemoryGetTypeInfromation[_[id_Integer, args___]] := ("TypeInfromation" /. {args})
WGLMemoryGetTypeInfromation[args___] := Throw[APIMessage["gettypinfo", args]; $Failed]

WGLMemoryImageQ[mem_] := WGLMemoryGetMathematicaType[mem] === Image
WGLMemoryListQ[mem_] := WGLMemoryGetMathematicaType[mem] === List

WGLMemoryGetColorSpace[mem_] := "ColorSpace" /. WGLMemoryGetTypeInfromation[mem]

(******************************************************************************) 

WGLAddConstantMemory[memoryHd_, errMsgHd_, symbol_, type0_, info_List, usingSinglePrecisionQ_] :=
	Module[{type =  ResolveType[type0, usingSinglePrecisionQ]},
		If[FreeQ[WGLTypes, type],
			Throw[Message[errMsgHd::type, type]; $Failed]
		];
		SetErrorMessageHead[errMsgHd];
		AddConstantMemory[memoryHd, errMsgHd, symbol, type, info]
	]

WGLAddMemory[memoryHd_, errMsgHd_, data_, type0_, info_List, residence_, sharing_, usingSinglePrecisionQ_] :=
	Module[{type = ResolveType[type0, usingSinglePrecisionQ], typewidth},
		If[FreeQ[WGLTypes, type],
			Throw[Message[errMsgHd::type, type]; $Failed]
		];
		typewidth = ResolveElementWidth[type];
		If[typewidth =!= 1 && !Divisible[Last[Dimensions[data]], typewidth],
			Throw[Message[errMsgHd::typwid, typewidth, Last[Dimensions[data]]]; $Failed]
		];
		SetErrorMessageHead[errMsgHd];
		AddMemory[memoryHd, errMsgHd, data, type, info, residence, sharing, True]
	]

WGLAllocateMemory[memoryHd_, errMsgHd_, type0_, dims0_, residence_, usingSinglePrecisionQ_] :=
	Module[{type = ResolveType[type0, usingSinglePrecisionQ], typewidth, dims},
		If[FreeQ[WGLTypes, type],
			Throw[Message[errMsgHd::type, type]; $Failed]
		];
		typewidth = ResolveElementWidth[type];
		dims = If[typewidth =!= 1,
			Flatten[{dims0, typewidth}],
			dims0
		];
		SetErrorMessageHead[errMsgHd];
		AllocateMemory[memoryHd, errMsgHd, type, dims, residence, usingSinglePrecisionQ]
	]
	
WGLGetMemory[errMsgHd_, mem_, forceQ_:False] :=
	Module[{res},
		If[!MemberQ[{True, False}, forceQ],
			Throw[Message[errMsgHd::invfrce, forceQ]; $Failed]
		];
		SetErrorMessageHead[errMsgHd];
		res = GetMemory[WGLMemoryGetID[mem], forceQ];
		res
	]
	
WGLDeleteMemory[errMsgHd_, {mem_, _String, _}] :=
	WGLDeleteMemory[errMsgHd, mem]
WGLDeleteMemory[errMsgHd_, mem:hd_[id_Integer, ___]] :=
	(
		SetErrorMessageHead[errMsgHd];
		DeleteMemory[WGLMemoryGetID[mem]]
	)
WGLDeleteMemory[errMsgHd_, x___] := Throw[Message[errMsgHd::memdel, {x}]; $Failed]

WGLMemoryInformation[errMsgHd_, mem:_[id_Integer, ___]] :=
	Module[{},
		SetErrorMessageHead[errMsgHd];

		Join[
			MemoryInformation[WGLMemoryGetID[mem]],
			{
				"Type" -> WGLMemoryGetType[mem],
				"ByteCount" -> WGLMemoryGetByteCount[mem],
				"Dimensions" -> WGLMemoryGetDimensions[mem]
			},
			If[$ThisAPI === "CUDA",
				{},
				{"Platform" -> WGLMemoryGetPlatform[mem]}
			],
			{
				"Device" -> WGLMemoryGetDevice[mem],
				"MathematicaType" -> WGLMemoryGetMathematicaType[mem],
				"TypeInformation" -> WGLMemoryGetTypeInfromation[mem]
			}
		]
	]

WGLCopyMemory[errMsgHd_, mem_, opt:("ToHost" | "ToDevice")] :=
	(
		SetErrorMessageHead[errMsgHd];
		CopyMemory[WGLMemoryGetID[mem], False, opt];
		mem
	)
	
WGLCopyMemoryToDevice[errMsgHd_, mem_] := WGLCopyMemory[errMsgHd, mem, "ToDevice"]

WGLCopyMemoryToHost[errMsgHd_, mem_] := WGLCopyMemory[errMsgHd, mem, "ToHost"]

WGLCopyConstantMemory[direction_][errMsgHd_, constmem_, regmem_] :=
	(
		SetErrorMessageHead[errMsgHd];
		CopyConstantMemory[WGLMemoryGetConstantSymbol[constmem], WGLMemoryGetID[regmem], direction];
		constmem
	)
	
	
WGLBindTexture[errMsgHd_, textmem_, mem_, addressMode_, filterMode_, flags_] :=
	(
		SetErrorMessageHead[errMsgHd];
		BindToTexture[textmem, WGLMemoryGetID[mem], toAddressMode /@ addressMode, filterMode, flags];
		textmem
	)
	
toAddressMode[mode_] :=
	Switch[mode,
		"Periodic" | "Wrap", 0,
		"Fixed" | "Clamp", 1,
		"Reversed" | "Mirror", 2,
		"Border", 3,
		_, $Failed
	]
AddressModeQ[mode_] := IntegerQ[toAddressMode[mode]]

WGLAddProgram[programHd_, errMsgHd_, prog:({_String?FileExistsQ} | _String), functionName_String, args_List, source_, binaryQ:(True | False), buildOptions:(None | _String)] :=
	Module[{res},
		SetErrorMessageHead[errMsgHd];
		
		res = Which[
			ListQ[prog],
				AddProgram[FileExtension[First[prog]]][errMsgHd, prog, functionName, args, buildOptions],
			TrueQ[binaryQ],
				AddProgramFromBinaryFile[errMsgHd, prog],
			True,
				AddProgramFromSource[errMsgHd, prog, buildOptions]
		];

		Which[
			res =!= $Failed && Last[res] === "LibraryFunction",
				programHd[RandomInteger[Developer`$MaxMachineInteger], "Type" -> Last[res], "Source" -> prog, "LibraryFunction" -> First[res]],
			res =!= $Failed && binaryQ,
				programHd[First[res], "Type" -> Last[res], "Source" -> source, "Binary" -> prog],
			res =!= $Failed,
				programHd[First[res], "Type" -> Last[res], "Source" -> prog],
			True,
				Throw[$Failed]
		]
	]

WGLProgramGetID[_[id_Integer, ___]] := id
WGLProgramGetType[_[id_Integer, args___]] := "Type" /. {args}
WGLProgramGetSource[_[id_Integer, args___]] := "Source" /. {args}
WGLProgramGetLibraryFunction[_[id_Integer, args___]] := "LibraryFunction" /. {args}

WGLProgramGetBuildLog[errMsgHd_, $Failed, {prog_?FileExistsQ}, buildOptions_] :=
	WGLProgramGetBuildLog[errMsgHd, $Failed, prog, buildOptions]
WGLProgramGetBuildLog[errMsgHd_, $Failed, prog_String, buildOptions_] :=
	(
		SetErrorMessageHead[errMsgHd];
		GetProgramBuildLog[$SingleOrDoubleLibrary][0, prog, buildOptions]
	)
WGLProgramGetBuildLog[errMsgHd_, _[progId_, ___], {progf_?FileExistsQ}, buildOptions_] :=
	(
		SetErrorMessageHead[errMsgHd];
		GetProgramBuildLog[$SingleOrDoubleLibrary][progId, progf, buildOptions]
	)
WGLProgramGetBuildLog[errMsgHd_, _[progId_, ___], prog_String, buildOptions_] :=
	(
		SetErrorMessageHead[errMsgHd];
		GetProgramBuildLog[$SingleOrDoubleLibrary][progId, prog, buildOptions]
	)
WGLProgramGetBuildLog[errMsgHd_, args___] :=
	Throw[Message[errMsgHd::args, {args}]; $Failed]
WGLSetDeleteProgramFileOnExit[errMsgHd_, fileName_] /; FileExistsQ[fileName] :=
	(
		SetErrorMessageHead[errMsgHd];
		Catch[
			SetDeleteProgramFileOnExit[fileName]
		]
	)

WGLProgramInformation[errMsgHd_, program_] :=
	If[WGLProgramGetType[program] === "LibraryFunction",
		With[{libraryFunction = WGLProgramGetLibraryFunction[program]},
			{
				"ID" -> WGLProgramGetID[program],
			 	"Type" -> WGLProgramGetType[program],
			 	"Source" -> WGLProgramGetSource[program],
			 	"LibraryFunction" -> libraryFunction,
			 	"LibraryFunctionInformation" -> LibraryFunctionInformation[libraryFunction]
			}
		],
		(* Else *)
		SetErrorMessageHead[errMsgHd];
		GetProgramInformation[WGLProgramGetType[program]][program]
	]
	
WGLSetProgram["Source" | "SourceFile" | "Binary" | "BinaryFile"][programHd_, errMsgHd_, programHd_[id_Integer, ___]] :=
	(
		SetErrorMessageHead[errMsgHd];
		SetKernelFunction[id]
	)

WGLSetFunction["Source" | "SourceFile" | "Binary" | "BinaryFile"][functionHd_, errMsgHd_, kernelName_, ___] :=
	(
		SetErrorMessageHead[errMsgHd];
		SetKernel[kernelName]
	)
	
WGLSetFunctionIntegerArgument["Source" | "SourceFile" | "Binary" | "BinaryFile"][errMsgHd_, x_Integer] :=
	(
		SetErrorMessageHead[errMsgHd];
		SetKernelIntegerArgument[x]
	)
	
WGLSetFunctionUnsignedIntegerArgument["Source" | "SourceFile" | "Binary" | "BinaryFile"][errMsgHd_, x_Integer] :=
	(
		SetErrorMessageHead[errMsgHd];
		SetKernelUnsignedIntegerArgument[x]
	)
	
WGLSetFunctionLongArgument["Source" | "SourceFile" | "Binary" | "BinaryFile"][errMsgHd_, x_Integer] :=
	(
		SetErrorMessageHead[errMsgHd];
		SetKernelLongArgument[x]
	)
	
WGLSetFunctionFloatArgument["Source" | "SourceFile" | "Binary" | "BinaryFile"][errMsgHd_, x_Real] :=
	(
		SetErrorMessageHead[errMsgHd];
		SetKernelFloatArgument[x]
	)
	
WGLSetFunctionDoubleArgument["Source" | "SourceFile" | "Binary" | "BinaryFile"][errMsgHd_, x_Real] :=
	(
		SetErrorMessageHead[errMsgHd];
		SetKernelDoubleArgument[x]
	)
	
WGLSetFunctionRealArgument[prog:("Source" | "SourceFile" | "Binary" | "BinaryFile")][errMsgHd_, x_Real, useSinglePrecisionQ_] :=
	If[useSinglePrecisionQ,
		WGLSetFunctionFloatArgument[prog][errMsgHd, x],
		WGLSetFunctionDoubleArgument[prog][errMsgHd, x]
	]
	
WGLSetFunctionMemoryArgument["Source" | "SourceFile" | "Binary" | "BinaryFile"][memoryHd_, errMsgHd_, id_, inputOutpuParam_] :=
	(
		SetErrorMessageHead[errMsgHd];
		SetKernelMemoryArgument[id, inputOutpuParam]
	)
	
WGLSetFunctionLocalMemoryArgument["Source" | "SourceFile" | "Binary" | "BinaryFile"][memoryHd_, errMsgHd_, size_] :=
	(
		SetErrorMessageHead[errMsgHd];
		SetKernelLocalMemoryArgument[size]
	)
	
WGLSetGridDimensions[funType:("Source" | "SourceFile" | "Binary" | "BinaryFile")][errMsgHd_, gridDimensions_Integer] :=
	WGLSetGridDimensions[funType][errMsgHd, {gridDimensions}]
WGLSetGridDimensions["Source" | "SourceFile" | "Binary" | "BinaryFile"][errMsgHd_, gridDimensions_List] :=
	(
		SetErrorMessageHead[errMsgHd];
		SetGridDimensions[Flatten[gridDimensions]]
	)

WGLSetBlockDimensions[funType:("Source" | "SourceFile" | "Binary" | "BinaryFile")][errMsgHd_, blockDimensions_Integer] :=
	WGLSetBlockDimensions[funType][errMsgHd, {blockDimensions}] 
WGLSetBlockDimensions["Source" | "SourceFile" | "Binary" | "BinaryFile"][errMsgHd_, blockDimensions_List] :=
	(
		SetErrorMessageHead[errMsgHd];
		SetBlockDimensions[Flatten[blockDimensions]]
	)
		
WGLLaunchFunction["Source" | "SourceFile" | "Binary" | "BinaryFile"][errMsgHd_] :=
	(
		SetErrorMessageHead[errMsgHd];
		LaunchKernel[]
	)
	
WGLLaunchFunction["LibraryFunction"][errMsgHd_, progObj_, kernelArgs_, gridDimensions0_, blockDimensions0_] :=
	Module[{args = getLibraryFunctionLaunchArgs[kernelArgs], res, gridDimensions, blockDimensions},
		gridDimensions = Developer`ToPackedArray[Flatten[{gridDimensions0}]];
		blockDimensions = Developer`ToPackedArray[Flatten[{blockDimensions0}]];
		res = Quiet[WGLProgramGetLibraryFunction[progObj][Sequence@@args, gridDimensions, blockDimensions]];
		
		Which[
			Head[res] === LibraryFunctionError,
				Throw[Message[errMsgHd::liblnch, First[res]]; $Failed],
			res == 0,
				SuccessQ[res],
			Head[res] === Integer,
				res,
			SuccessQ[res],
				True,
			True,
				Throw[$Failed]
		]
	]

WGLEnableProfiler[errMsgHd_] :=
	(
		SetErrorMessageHead[errMsgHd];
		EnableProfiler[]
	)
	
WGLDisableProfiler[errMsgHd_] :=
	(
		SetErrorMessageHead[errMsgHd];
		DisableProfiler[]
	)
	
WGLClearProfiler[errMsgHd_] :=
	(
		SetErrorMessageHead[errMsgHd];
		ClearProfiler[]
	)
	
WGLGetProfilerInformation[errMsgHd_] :=
	(
		SetErrorMessageHead[errMsgHd];
		GetProfilerInformation[]
	)

WGLGetProfileElementInformation[errMsgHd_, index_Integer] :=
	(
		SetErrorMessageHead[errMsgHd];
		GetProfileElementInformation[index]
	)
	
(******************************************************************************)

getLibraryFunctionLaunchArgs[args_] :=
	getLibraryFunctionLaunchArg /@ args
	
getLibraryFunctionLaunchArg[{arg_Integer, Integer}] :=
	arg
getLibraryFunctionLaunchArg[{arg_Real, Real}] :=
	arg
getLibraryFunctionLaunchArg[{arg_Complex, Complex}] :=
	arg
getLibraryFunctionLaunchArg[{arg_String, "UTF8String"}] :=
	arg
getLibraryFunctionLaunchArg[{arg:(True | False), "Boolean"}] :=
	arg
getLibraryFunctionLaunchArg[{arg_List, List}] :=
	arg
getLibraryFunctionLaunchArg[{arg_Integer, "Local" | "Shared"}] :=
	arg
getLibraryFunctionLaunchArg[{mem_, _, _}] :=
	WGLMemoryGetID[mem]
(******************************************************************************)

WGLTypes = Complement[Flatten[{
	"Shared",
	"Local",
	"Void",
	"Byte",
	"Byte[1]",
	"Byte[2]",
	"Byte[3]",
	"Byte[4]",
	"Byte[8]",
	"Byte[16]",
	"UnsignedByte",
	"UnsignedByte[1]",
	"UnsignedByte[2]",
	"UnsignedByte[3]",
	"UnsignedByte[4]",
	"UnsignedByte[8]",
	"UnsignedByte[16]",
	"Bit16",
	"Bit16[1]",
	"Bit16[2]",
	"Bit16[3]",
	"Bit16[4]",
	"Bit16[8]",
	"Bit16[16]",
	"Short",
	"Short[1]",
	"Short[2]",
	"Short[3]",
	"Short[4]",
	"Short[8]",
	"Short[16]",
	"UnsignedBit16",
	"UnsignedBit16[1]",
	"UnsignedBit16[2]",
	"UnsignedBit16[3]",
	"UnsignedBit16[4]",
	"UnsignedBit16[8]",
	"UnsignedBit16[16]",
	"UnsignedShort",
	"UnsignedShort[1]",
	"UnsignedShort[2]",
	"UnsignedShort[3]",
	"UnsignedShort[4]",
	"UnsignedShort[8]",
	"UnsignedShort[16]",
	Verbatim[_Integer],
	Integer,
	Integer[1],
	Integer[2],
	Integer[3],
	Integer[4],
	Integer[8],
	Integer[16],
	"Integer32",
	"Integer32[1]",
	"Integer32[2]",
	"Integer32[3]",
	"Integer32[4]",
	"Integer32[8]",
	"Integer32[16]",
	Sequence@@If[$64BitMint,
		{
			"UnsignedInteger32",
			"UnsignedInteger32[1]",
			"UnsignedInteger32[2]",
			"UnsignedInteger32[3]",
			"UnsignedInteger32[4]",
			"UnsignedInteger32[8]",
			"UnsignedInteger32[16]",
			"Integer64",
			"Integer64[1]",
			"Integer64[2]",
			"Integer64[3]",
			"Integer64[4]",
			"Integer64[8]",
			"Integer64[16]"
		},
		{}
	],
	Verbatim[_Real],
	Real,
	Real[1],
	Real[2],
	Real[3],
	Real[4],
	Real[8],
	Real[16],
	"Float",
	"Float[1]",
	"Float[2]",
	"Float[3]",
	"Float[4]",
	"Float[8]",
	"Float[16]",
	"Double",
	"Double[1]",
	"Double[2]",
	"Double[3]",
	"Double[4]",
	"Double[8]",
	"Double[16]",
	"ComplexFloat",
	"ComplexDouble",
	Verbatim[_Complex],
	Complex,
	"MatrixFloat",
	"MatrixTransposedFloat",
	"MatrixDouble",
	"MatrixTransposedDouble",
	"MatrixComplexFloat",
	"MatrixTransposedComplexFloat",
	"MatrixComplexDouble",
	"MatrixTransposedComplexDouble"
}], $WGLInvalidTypes]


Do[
	WGLTypeQ[type] = True
	,
	{type, WGLTypes}
]
WGLTypeQ[_] := False

WGLMaxType[t1_] := t1
WGLMaxType[t1_, t2_] :=
	Which[
		 GetTensorType[t1] === Integer,
		 	GetTensorType[t2],
		 GetTensorType[t1] === Real,
		 	If[GetTensorType[t2] === Integer,
		 		Real,
		 		GetTensorType[t2]
		 	],
		 True,
		 	Complex
	]
WGLMaxType[t1_, t2_, rest__] :=
	WGLMaxType[WGLMaxType[t1, t2], rest]

ListTypeQ[type_String] := ListTypeQ[type] = FreeQ[{"Local", "Shared", "Void"}, type] && MemberQ[WGLTypes, type]

GetImageType[type_] := GetImageType[type] =
	Switch[type,
		Automatic | "Byte" | "Byte[1]" | "Byte[2]" | "Byte[3]" | "Byte[4]" | "Byte[8]" | "Byte[16]" |
		"UnsignedByte" | "UnsignedByte[2]" | "UnsignedByte[3]" | "UnsignedByte[4]" | "UnsignedByte[8]" | "UnsignedByte[16]",
			"Byte",
		"Bit16" | "Bit16[1]" | "Bit16[2]" | "Bit16[3]" | "Bit16[4]" | "Bit16[8]" | "Bit16[16]" |
		"Short" | "Short[1]" | "Short[2]" | "Short[3]" | "Short[4]" | "Short[8]" | "Short[16]" |
		"UnsignedBit16" | "UnsignedBit16[1]" | "UnsignedBit16[2]" | "UnsignedBit16[3]" | "UnsignedBit16[4]" | "UnsignedBit16[8]" | "UnsignedBit16[16]" |
		"UnsignedShort" | "UnsignedShort[1]" | "UnsignedShort[2]" | "UnsignedShort[3]" | "UnsignedShort[4]" | "UnsignedShort[8]" | "UnsignedShort[16]",
			"Bit16",
		Integer | Integer[1] | Integer[2] | Integer[3] | Integer[4] | Integer[8] | Integer[16] |
		"Integer32" | "Integer32[1]" | "Integer32[2]" | "Integer32[3]" | "Integer32[4]" | "Integer32[8]" | "Integer32[16]" |
		"UnsignedInteger32" | "UnsignedInteger[1]" | "UnsignedInteger32[2]" | "UnsignedInteger32[3]" | "UnsignedInteger32[4]" | "UnsignedInteger32[8]" | "UnsignedInteger32[16]" |
		"Integer64" | "Integer64[2]" | "Integer64[3]" | "Integer64[4]" | "Integer64[8]" | "Integer64[16]",
			"Byte",
		"Float" | "Float[2]" | "Float[3]" | "Float[4]" | "Float[8]" | "Float[16]",
			"Real32", 
		Real | "Double" | "Double[2]" | "Double[3]" | "Double[4]" | "Double[8]" | "Double[16]",
			"Real",
		_,
			Throw[Message[General::gimgtyp, type]; $Failed]
	]

GetTensorType[type_] := GetTensorType[type] =
	Switch[type,
		"Byte" | "Byte[1]" | "Byte[2]" | "Byte[3]" | "Byte[4]" | "Byte[8]" | "Byte[16]" |
		"UnsignedByte" | "UnsignedByte[1]" | "UnsignedByte[2]" | "UnsignedByte[3]" |
		"UnsignedByte[4]" | "UnsignedByte[8]" | "UnsignedByte[16]" |
		"Bit16" | "Bit16[1]" | "Bit16[2]" | "Bit16[3]" | "Bit16[4]" | "Bit16[8]" | "Bit16[16]" |
		"Short" | "Short[1]" | "Short[2]" | "Short[3]" | "Short[4]" | "Short[8]" | "Short[16]" |
		"UnsignedBit16" | "UnsignedBit16[1]" | "UnsignedBit16[2]" | "UnsignedBit16[3]" | "UnsignedBit16[4]" | "UnsignedBit16[8]" | "UnsignedBit16[16]" |
		"UnsignedShort" | "UnsignedShort[1]" | "UnsignedShort[2]" | "UnsignedShort[3]" | "UnsignedShort[4]" | "UnsignedShort[8]" | "UnsignedShort[16]" |
		Integer | Integer[1] | Integer[2] | Integer[3] | Integer[4] | Integer[8] | Integer[16] |
		"Integer32" | "Integer32[1]" | "Integer32[2]" | "Integer32[3]" | "Integer32[4]" | "Integer32[8]" | "Integer32[16]" |
		"UnsignedInteger32" | "UnsignedInteger32[1]" | "UnsignedInteger32[2]" | "UnsignedInteger32[3]" | "UnsignedInteger32[4]" | "UnsignedInteger32[8]" | "UnsignedInteger32[16]" |
		"Integer64" | "Integer64[1]" | "Integer64[2]" | "Integer64[3]" | "Integer64[4]" | "Integer64[8]" | "Integer64[16]",
			Integer,
		Real | Real[1] | Real[2] | Real[3] | Real[4] | Real[8] | Real[16] |
		"Float" | "Float[1]" | "Float[2]" | "Float[3]" | "Float[4]" | "Float[8]" | "Float[16]" |
		"Double" | "Double[1]" | "Double[2]" | "Double[3]" | "Double[4]" | "Double[8]" | "Double[16]",
			Real,
		Complex | "ComplexFloat" | "MatrixComplexFloat" | "MatrixTransposedComplexFloat" |
		"ComplexDouble" | "MatrixComplexDouble" | "MatrixTransposedComplexDouble",
			Complex,
		_,
			Throw[Message[General::gtenstyp, type]; $Failed]
	]
TypeRealQ[type_] := TypeRealQ[type] =
	Switch[type,
		Real | Real[1] | Real[2] | Real[3] | Real[4] | Real[8] | Real[16] |
		"Float" | "Float[1]" | "Float[2]" | "Float[3]" | "Float[4]" | "Float[8]" | "Float[16]" |
		"ComplexFloat" | "MatrixComplexFloat" | "MatrixTransposedComplexFloat" | Complex |
		"Double" | "Double[1]" | "Double[2]" | "Double[3]" | "Double[4]" | "Double[8]" | "Double[16]" |
		"ComplexDouble" | "MatrixComplexDouble" | "MatrixTransposedComplexDouble",
			True,
		_,
			False
	]


TypeDoublePrecisionQ[type_] := TypeDoublePrecisionQ[type] =
	Switch[type,
		Real | Real[1] | Real[2] | Real[3] | Real[4] | Real[8] | Real[16] | Complex,
			SupportsDoublePrecisionQ[TypeDoublePrecisionQ],
		"Double" | "Double[1]" | "Double[2]" | "Double[3]" | "Double[4]" | "Double[8]" | "Double[16]" |
		"ComplexDouble" | "MatrixComplexDouble" | "MatrixTransposedComplexDouble",
			True,
		_,
			False
	]

ResolveByteCount[type_] :=
	ResolveByteCount[type, SupportsDoublePrecisionQ[ResolveByteCount] === False]
ResolveByteCount[type_, useSinglePrecisionQ_] := ResolveByteCount[type, useSinglePrecisionQ] =
	Ceiling[
		Switch[type,
			"Byte" | "Byte[1]" | "Byte[2]" | "Byte[3]" | "Byte[4]" | "Byte[8]" | "Byte[16]" |
			"UnsignedByte" | "UnsignedByte[1]" | "UnsignedByte[2]" | "UnsignedByte[3]" |
			"UnsignedByte[4]" | "UnsignedByte[8]" | "UnsignedByte[16]",
				ResolveElementWidth[type],
			"Bit16" | "Bit16[1]" | "Bit16[2]" | "Bit16[3]" | "Bit16[4]" | "Bit16[8]" | "Bit16[16]" |
			"Short" | "Short[1]" | "Short[2]" | "Short[3]" | "Short[4]" | "Short[8]" | "Short[16]" |
			"UnsignedBit16" | "UnsignedBit16[1]" | "UnsignedBit16[2]" | "UnsignedBit16[3]" | "UnsignedBit16[4]" | "UnsignedBit16[8]" | "UnsignedBit16[16]" |
			"UnsignedShort" | "UnsignedShort[1]" | "UnsignedShort[2]" | "UnsignedShort[3]" | "UnsignedShort[4]" | "UnsignedShort[8]" | "UnsignedShort[16]",
				2 * ResolveElementWidth[type],
			Verbatim[_Integer] | Integer | Integer[1] | Integer[2] | Integer[3] | Integer[4] | Integer[8] | Integer[16],
				If[$64BitMint,
					ResolveByteCount["Integer64", useSinglePrecisionQ],
					ResolveByteCount["Integer32", useSinglePrecisionQ]
				],
			"Integer32" | "Integer32[1]" | "Integer32[2]" | "Integer32[3]" | "Integer32[4]" | "Integer32[8]" | "Integer32[16]" |
			"UnsignedInteger32" | "UnsignedInteger32[1]" | "UnsignedInteger32[2]" | "UnsignedInteger32[3]" | "UnsignedInteger32[4]" | "UnsignedInteger32[8]" | "UnsignedInteger32[16]",
				$SizeOfInteger32 * ResolveElementWidth[type],
			"Integer64" | "Integer64[1]" | "Integer64[2]" | "Integer64[3]" | "Integer64[4]" | "Integer64[8]" | "Integer64[16]",
				$SizeOfInteger64 * ResolveElementWidth[type],
			Verbatim[_Real] | Real | Real[1] | Real[2] | Real[3] | Real[4] | Real[8] | Real[16],
				If[useSinglePrecisionQ,
					$SizeOfInteger32,
					8
				],
			"Float" | "Float[1]" | "Float[2]" | "Float[3]" | "Float[4]" | "Float[8]" | "Float[16]" | "MatrixFloat" | "MatrixTransposedFloat",
				$SizeOfInteger32 * ResolveElementWidth[type],
			"Double" | "Double[1]" | "Double[2]" | "Double[3]" | "Double[4]" | "Double[8]" | "Double[16]" | "MatrixDouble" | "MatrixTransposedDouble",
				8 * ResolveElementWidth[type],
			"ComplexFloat" | "MatrixComplexFloat" | "MatrixTransposedComplexFloat",
				8,
			Verbatim[_Complex] |Complex | "Complex",
				If[useSinglePrecisionQ,
					8,
					16
				],
			"ComplexDouble" |  "MatrixComplexDouble" | "MatrixTransposedComplexDouble",
				16,
			_,
				Throw[Message[General::resbc, type]; $Failed]
		]
	]

ResolveBaseType[type_] :=
	ResolveBaseType[type, SupportsDoublePrecisionQ[ResolveBaseType] === False]
ResolveBaseType[type_, useSinglePrecisionQ_] := ResolveBaseType[type, useSinglePrecisionQ] =
	Switch[type,
		"Byte" | "Byte[1]" | "Byte[2]" | "Byte[3]" | "Byte[4]" | "Byte[8]" | "Byte[16]",
			"Byte",
		"UnsignedByte" | "UnsignedByte[1]" | "UnsignedByte[2]" | "UnsignedByte[3]" |
		"UnsignedByte[4]" | "UnsignedByte[8]" | "UnsignedByte[16]",
			"UnsignedByte",
		"Bit16" | "Bit16[1]" | "Bit16[2]" | "Bit16[3]" | "Bit16[4]" | "Bit16[8]" | "Bit16[16]" |
		"Short" | "Short[1]" | "Short[2]" | "Short[3]" | "Short[4]" | "Short[8]" | "Short[16]",
			"Short",
		"UnsignedBit16" | "UnsignedBit16[1]" | "UnsignedBit16[2]" | "UnsignedBit16[3]" | "UnsignedBit16[4]" | "UnsignedBit16[8]" | "UnsignedBit16[16]" |
		"UnsignedShort" | "UnsignedShort[1]" | "UnsignedShort[2]" | "UnsignedShort[3]" | "UnsignedShort[4]" | "UnsignedShort[8]" | "UnsignedShort[16]",
			"UnsignedShort",
		Verbatim[_Integer] | Integer | Integer[1] | Integer[2] | Integer[3] | Integer[4] | Integer[8] | Integer[16],
			If[$64BitMint,
				"Integer64",
				"Integer32"
			],
		"Integer32" | "Integer32[1]" | "Integer32[2]" | "Integer32[3]" | "Integer32[4]" | "Integer32[8]" | "Integer32[16]",
			"Integer32",
		"UnsignedInteger32" | "UnsignedInteger32[1]" | "UnsignedInteger32[2]" | "UnsignedInteger32[3]" | "UnsignedInteger32[4]" | "UnsignedInteger32[8]" | "UnsignedInteger32[16]",
			"UnsignedInteger32",
		"Integer64" | "Integer64[1]" | "Integer64[2]" | "Integer64[3]" | "Integer64[4]" | "Integer64[8]" | "Integer64[16]",
			"Integer64",
		Verbatim[_Real] | Real | Real[1] | Real[2] | Real[3] | Real[4] | Real[8] | Real[16],
			If[useSinglePrecisionQ,
				"Float",
				"Double"
			],
		"Float" | "Float[1]" | "Float[2]" | "Float[3]" | "Float[4]" | "Float[8]" | "Float[16]" | "MatrixFloat" | "MatrixTransposedFloat",
			"Float",
		"Double" | "Double[1]" | "Double[2]" | "Double[3]" | "Double[4]" | "Double[8]" | "Double[16]" | "MatrixDouble" | "MatrixTransposedDouble",
			"Double",
		"ComplexFloat" | "MatrixComplexFloat" | "MatrixTransposedComplexFloat",
			"ComplexFloat",
		Verbatim[_Complex] |Complex | "Complex",
			If[useSinglePrecisionQ,
				"ComplexFloat",
				"ComplexDouble"
			],
		"ComplexDouble" |  "MatrixComplexDouble" | "MatrixTransposedComplexDouble",
			"CompexDouble",
		_,
			Throw[Message[General::bstyp, type]; $Failed]
	]

ResolveByteCount[type_, byteCount_Integer] :=
	Ceiling[
		Switch[type,
			Verbatim[_Integer] | Integer | Integer[1] | Integer[2] | Integer[3] | Integer[4] | Integer[8] | Integer[16],
				If[$64BitMint,
					ResolveByteCount["Integer64", byteCount],
					ResolveByteCount["Integer32", byteCount]
				],
			"Integer32" | "Integer32[1]" | "UnsignedInteger32" | "UnsignedInteger32[1]",
				If[$64BitMint,
					Ceiling[byteCount/2],
					byteCount
				],
			"Integer64" | "Integer64[1]",
				If[$64BitMint,
					byteCount,
					2 * byteCount
				],
			"Double" | "Double[1]" | "ComplexDouble" | "ComplexDouble[1]" |
			"MatrixDouble" | "MatrixDouble[1]" | "MatrixTransposedDouble" | "MatrixTransposedDouble[1]" |
			"MatrixComplexDouble" | "MatrixComplexDouble[1]" | "MatrixTransposedComplexDouble" | "MatrixTransposedComplexDouble[1]",
				byteCount,
			"Byte" | "Byte[1]" | "Byte[2]" | "Byte[3]" | "Byte[4]" | "Byte[8]" | "Byte[16]" |
			"UnsignedByte" | "UnsignedByte[1]" | "UnsignedByte[2]" | "UnsignedByte[3]" | "UnsignedByte[4]" |
			"UnsignedByte[8]" | "UnsignedByte[16]",
				(byteCount / 4) * ResolveElementWidth[type],
			"Bit16" | "Bit16[1]" | "Short" | "Short[1]" | "UnsignedBit16" | "UnsignedBit16[1]" | "UnsignedShort" | "UnsignedShort[1]",
				byteCount / 2,
			"Bit16[2]" | "Bit16[3]" | "Bit16[4]" | "Bit16[8]" | "Bit16[16]" |
			"Short[2]" | "Short[3]" | "Short[4]" | "Short[8]" | "Short[16]" |
			"UnsignedBit16[2]" | "UnsignedBit16[3]" | "UnsignedBit16[4]" | "UnsignedBit16[8]" | "UnsignedBit16[16]" |
			"UnsignedShort[2]" | "UnsignedShort[3]" | "UnsignedShort[4]" | "UnsignedShort[8]" | "UnsignedShort[16]",
				ResolveByteCount["Short", byteCount] * ResolveElementWidth[type],
			"Integer32[2]" | "Integer32[3]" | "Integer32[4]" | "Integer32[8]" | "Integer32[16]" |
			"UnsignedInteger32[2]" | "UnsignedInteger32[3]" | "UnsignedInteger32[4]" | "UnsignedInteger32[8]" | "UnsignedInteger32[16]",
				ResolveByteCount["Integer32", byteCount] * ResolveElementWidth[type],
			"Integer64[2]" | "Integer64[3]" | "Integer64[4]" | "Integer64[8]" | "Integer64[16]",
				ResolveByteCount["Integer64", byteCount] * ResolveElementWidth[type],
			"Float" | "Float[1]" | "Float[2]" | "Float[3]" | "Float[4]" | "Float[8]" | "Float[16]" | "MatrixFloat" | "MatrixTransposedFloat",
				(byteCount / 2) * ResolveElementWidth[type],
			"Double[2]" | "Double[3]" | "Double[4]" | "Double8" | "Double16",
				ResolveByteCount["Double", byteCount] * ResolveElementWidth[type],
			"ComplexFloat" | "MatrixComplexFloat" | "MatrixTransposedComplexFloat",
				byteCount / 2
		]
	]

ResolveElementWidth[type:(Verbatim[_Integer] | Verbatim[_Real] | Verbatim[_Complex])] := 1
ResolveElementWidth[type_String] := ResolveElementWidth[type] =
	Switch[type,
		"Byte[2]" | "UnsignedByte[2]" | "Bit16[2]" | "UnsignedBit16[2]" | "Short[2]" | "UnsignedShort[2]" |
        "Integer32[2]" | "UnsignedInteger32[2]" | "Integer64[2]" | "Float[2]" | "Double[2]",
			2,
		"Byte[3]" | "UnsignedByte[3]" | "Bit16[3]" | "UnsignedBit16[3]" | "Short[3]" | "UnsignedShort[3]" |
        "Integer32[3]" | "UnsignedInteger32[3]" | "Integer64[3]" | "Float[3]" | "Double[3]",
			3,
		"Byte[4]" | "UnsignedByte[4]" | "Bit16[4]" | "UnsignedBit16[4]" | "Short[4]" | "UnsignedShort[4]" |
        "Integer32[4]" | "UnsignedInteger32[4]" | "Integer64[4]" | "Float[4]" | "Double[4]",
			4,
		"Byte[8]" | "UnsignedByte[8]" | "Bit16[8]" | "UnsignedBit16[8]" | "Short[8]" | "UnsignedShort[8]" |
        "Integer32[8]" | "UnsignedInteger32[8]" | "Integer64[8]" | "Float[8]" | "Double[8]",
			8,
		"Byte[16]" | "UnsignedByte[16]" | "Bit16[16]" | "UnsignedBit16[16]" | "Short[16]" | "UnsignedShort[16]" |
        "Integer32[16]" | "UnsignedInteger32[16]" | "Integer64[16]" | "Float[16]" | "Double[16]",
			16,
		_,
			1
	]

ResolveType[var___] := ResolveType[var] = iResolveType[var]
iResolveType[x_] := iResolveType[x,  SupportsDoublePrecisionQ[ResolveType] === False]
iResolveType[x_String, usingSinglePrecisionQ_] := x 
iResolveType[Verbatim[_Integer] | Integer, usingSinglePrecisionQ_] := If[$64BitMint, "Integer64", "Integer32"]
iResolveType[Integer[i_Integer], usingSinglePrecisionQ_] := ResolveType[Integer] <> "[" <> ToString[i] <> "]"
iResolveType[Real | Verbatim[_Real], usingSinglePrecisionQ_] := If[usingSinglePrecisionQ, "Float", "Double"]
iResolveType[Real[i_Integer], usingSinglePrecisionQ_] := If[usingSinglePrecisionQ, "Float[" <> ToString[i] <> "]" , "Double[" <> ToString[i] <> "]"]
iResolveType[Complex | Verbatim[_Complex], usingSinglePrecisionQ_] := If[usingSinglePrecisionQ, "ComplexFloat", "ComplexDouble"]
iResolveType[a:{type_, rest___}, usingSinglePrecisionQ_] := 
	If[$progType === "LibraryFunction",
		a,
		{iResolveType[type, usingSinglePrecisionQ], rest}
	]
iResolveType[___] := "UnknownType"

(******************************************************************************)

Options[GPUAddMemory] = Options[iGPUAddMemory] = Options[GPUAllocateMemory] = {"TargetPrecision" -> Automatic, "Platform" -> Automatic, "Device" -> Automatic}

GPUAddMemory[memoryHd_, errMsgHd_, {}, opts:OptionsPattern[]] :=
	Return[Message[errMsgHd::empty]; $Failed]
GPUAddMemory[memoryHd_, errMsgHd_, img0_?Image`PossibleImageQ, opts:OptionsPattern[]] :=
	If[$ThisAPI === "CUDA",
		GPUAddMemory[memoryHd, errMsgHd, img0, "UnsignedByte", {Image}, opts],
		GPUAddMemory[memoryHd, errMsgHd, img0, If[$64BitMint, "Integer64", "Integer32"], {Image}, opts]
	]
GPUAddMemory[memoryHd_, errMsgHd_, tensor_List, opts:OptionsPattern[]] :=
	With[{type = GPUTools`Utilities`ArrayType[tensor]},
		iGPUAddMemory[memoryHd, errMsgHd, tensor, type, {List}, {}, opts]
	]
GPUAddMemory[memoryHd_, errMsgHd_, tensor_ /; !ListQ[tensor] && !Image`PossibleImageQ[tensor], ___] :=
	Throw[Message[errMsgHd::inputlst, tensor]; $Failed]
GPUAddMemory[memoryHd_, errMsgHd_, img_?Image`PossibleImageQ, type_, opts:OptionsPattern[]] :=
	iGPUAddMemory[memoryHd, errMsgHd, img, If[$ThisAPI === "CUDA" && type === Automatic, "UnsignedByte", type], {Image}, {}, opts]
GPUAddMemory[memoryHd_, errMsgHd_, tensor_List, type_, opts:OptionsPattern[]] :=
	iGPUAddMemory[memoryHd, errMsgHd, tensor, type, {List}, {}, opts]
GPUAddMemory[memoryHd_, errMsgHd_, img_?Image`PossibleImageQ, type_, {Image}, opts:OptionsPattern[]] :=
	iGPUAddMemory[memoryHd, errMsgHd, img, type, {Image}, {}, opts]
GPUAddMemory[memoryHd_, errMsgHd_, tensor_List, type_, {List}, opts:OptionsPattern[]] :=
	iGPUAddMemory[memoryHd, errMsgHd, tensor, type, {List}, {}, opts]
GPUAddMemory[memoryHd_, errMsgHd_, args___, opts:OptionsPattern[]] /; (ArgumentCountQ[errMsgHd, Length[{args}], 1, 2]; False) := $Failed

(******************************************************************************)

iGPUAddMemory[memoryHd_, errMsgHd_, img0_?Image`PossibleImageQ, type0_, {Image}, props_, opts:OptionsPattern[]] :=
	Module[{type, imgData, img, typeWidth},
		Catch[
			img = ToImage[errMsgHd, img0];
			typeWidth = ResolveElementWidth[type0];
			If[typeWidth > 1 && ImageChannels[img] =!= typeWidth,
				Throw[Message[errMsgHd::chnls, ImageChannels[img], typeWidth]; $Failed]
			];
			type = GetImageType[type0];
			imgData = ImageData[img, type];
			iGPUAddMemory[memoryHd, errMsgHd, imgData, If[type0 === Automatic, type, type0], {Image, Image`ImageInformation[img]}, props, opts]
		]
	]
iGPUAddMemory[memoryHd_, errMsgHd_, tensor:{}, type0_, info_List, props0_, opts:OptionsPattern[]] :=
	Throw[Message[errMsgHd::empty, tensor]; $Failed]
iGPUAddMemory[memoryHd_, errMsgHd_, tensor_List, type0_, info_List, props0_, opts:OptionsPattern[]] :=
	Block[{type, props, usingSinglePrecisionQ},
		Catch[
			type = If[type0 === Automatic,
				GPUTools`Utilities`ArrayType[tensor],
				type0
			];
			InitializeWGL[iGPUAddMemory, errMsgHd, opts];
			usingSinglePrecisionQ = GetAndCheckOption[iGPUAddMemory, errMsgHd, "TargetPrecision", {opts}] === "Single";
			props = ResolveMemoryLoadProperties[errMsgHd, props0];
			CheckCompatibleType[errMsgHd, tensor, type];
			WGLAddMemory[memoryHd, errMsgHd, tensor, type, info, Sequence@@props, usingSinglePrecisionQ]
		]
	]

(******************************************************************************)

GPUAllocateMemory[memoryHd_, errMsgHd_, type_, {}, opts:OptionsPattern[]] :=
	Throw[Message[errMsgHd::emptydim]; $Failed]
GPUAllocateMemory[memoryHd_, errMsgHd_, type_, dim_Integer, opts:OptionsPattern[]] :=
	If[Positive[dim],
		GPUAllocateMemory[memoryHd, errMsgHd, type, {dim}, opts],
		Throw[Message[errMsgHd::dims, dim]; $Failed]
	]
GPUAllocateMemory[memoryHd_, errMsgHd_, type_, dims_List, opts:OptionsPattern[]] :=
	Catch[
		InitializeWGL[GPUAllocateMemory, errMsgHd, opts];
		With[{useSinglePrecisionQ = GetAndCheckOption[iGPUAddMemory, errMsgHd, "TargetPrecision", {opts}] === "Single"},
			WGLAllocateMemory[memoryHd, errMsgHd, type, dims, "DeviceHost", useSinglePrecisionQ] (* TODO: REMOVE HARDCODED VALUES *)
		]
	]
GPUAllocateMemory[memoryHd_, errMsgHd_, type_, dims_ /; !(ListQ[dims] || IntegerQ[dims]), opts:OptionsPattern[]] := Throw[Message[errMsgHd::dims, dims]; $Failed]
GPUAllocateMemory[memoryHd_, errMsgHd_, args___, opts:OptionsPattern[]] /; (ArgumentCountQ[errMsgHd, Length[{args}], 2, 2]; False) := $Failed

(******************************************************************************)


Options[GPUAddConstantMemory] = Options[iGPUAddConstantMemory] = {"TargetPrecision" -> Automatic, "Platform" -> Automatic, "Device" -> Automatic}
GPUAddConstantMemory[memoryHd_, errMsgHd_, symbol_ /; !(StringQ[symbol] || Head[symbol] === Symbol), ___] :=
	Return[Message[errMsgHd::symbol, symbol]; $Failed]
GPUAddConstantMemory[memoryHd_, errMsgHd_, symbol:(_String | _Symbol), type_, opts:OptionsPattern[]] :=
	iGPUAddConstantMemory[memoryHd, errMsgHd, symbol, type, {List}, opts]

(******************************************************************************)

iGPUAddConstantMemory[memoryHd_, errMsgHd_, symbol_, type_, info_List, opts:OptionsPattern[]] :=
	Block[{props, usingSinglePrecisionQ},
		Catch[
			InitializeWGL[iGPUAddConstantMemory, errMsgHd, opts];
			usingSinglePrecisionQ = GetAndCheckOption[iGPUAddConstantMemory, errMsgHd, "TargetPrecision", {opts}] === "Single";
			WGLAddConstantMemory[memoryHd, errMsgHd, symbol, type, info, usingSinglePrecisionQ]
		]
	]
	
(******************************************************************************)

ResolveMemoryLoadProperties[errMsgHd_, props_String] := ResolveMemoryLoadProperties[errMsgHd, {props}]
ResolveMemoryLoadProperties[errMsgHd_, props0_List] :=
	Module[{props = {}},
		With[{int = Intersection[props0, {"DeviceOnly", "DeviceHost"}]},
			Switch[Length[int],
				0,
					AppendTo[props, "DeviceHost"],
				1,
					AppendTo[props, First[int]],
				_,
					Throw[Message[errMsgHd::resd, int]; $Failed]
			]
		];
		With[{int = Intersection[props0, {"Manual", "Shared"}]},
			Switch[Length[int],
				0,
					AppendTo[props, "Manual"],
				1,
					AppendTo[props, First[int]],
				_,
					Throw[Message[errMsgHd::shr, int]; $Failed]
			]
		];
		props
	]

(******************************************************************************)

CheckCompatibleType[errMsgHd_, tens_, type:(Verbatim[_Integer] | Verbatim[_Complex] | Verbatim[_Real])] :=
	CheckCompatibleType[errMsgHd, tens, First[type]]
CheckCompatibleType[errMsgHd_, tens_, type:(Integer | Complex | Real)] :=
	With[{tensType = GPUTools`Utilities`ArrayType[tens]},
		If[tensType =!= type && !(tensType === Integer && type === Real),
			Throw[Message[errMsgHd::tenstyp, tensType, type]; $Failed]
		]
	]
CheckCompatibleType[errMsgHd_, tens_, x_String] := True
CheckCompatibleType[errMsgHd_, tens_, x_ /; MemberQ[WGLTypes, x]] := True
CheckCompatibleType[errMsgHd_, tens_, x_] := Throw[Message[errMsgHd::notype, tens, x]; $Failed]

(******************************************************************************)

GPUGetMemory[memoryHd_, errMsgHd_, mem_, opt_:"Original"] /; MemberQ[{"Original", List, Image}, opt] :=
	Module[{tens},
		InitializeWGL[errMsgHd];
		If[Head[mem] =!= memoryHd,
			Return[Message[errMsgHd::gethdfld, memoryHd, Head[mem]]; $Failed]
		];
		Catch[
			tens = WGLGetMemory[errMsgHd, mem];
			Which[
				opt === List || WGLMemoryListQ[mem],
					tens,
				True,
					Image[tens, GetImageType[WGLMemoryGetType[mem]], "ColorSpace" -> WGLMemoryGetColorSpace[mem]]
			]
		]
	]
GPUGetMemory[memoryHd_, errMsgHd_, args___] := Throw[Message[errMsgHd::getmem, {args}]; $Failed]

(******************************************************************************)


GPUDeleteMemory[memoryHd_, errMsgHd_, mem__] :=
	iGPUDeleteMemory[memoryHd, errMsgHd, #]& /@ {mem}
iGPUDeleteMemory[memoryHd_, errMsgHd_, mem_] /; Head[mem] === memoryHd :=
	Catch[
		InitializeWGL[errMsgHd];
		WGLDeleteMemory[errMsgHd, mem]
	]
iGPUDeleteMemory[memoryHd_, errMsgHd_, args___] :=
	(
		Message[errMsgHd::unlmem, {args}];
		$Failed
	)
	
(******************************************************************************)

GPUCopyMemoryToHost[memoryHd_, errMsgHd_, mem_] /; Head[mem] === memoryHd :=
	Catch[
		InitializeWGL[errMsgHd];
		WGLCopyMemoryToHost[errMsgHd, mem]
	]
	
(******************************************************************************)

GPUCopyMemoryToDevice[memoryHd_, errMsgHd_, mem_] /; Head[mem] === memoryHd :=
	Catch[
		InitializeWGL[errMsgHd];
		WGLCopyMemoryToDevice[errMsgHd, mem]
	]	
(******************************************************************************)

GPUMemoryInformation[memoryHd_, errMsgHd_, mem_] /; Head[mem] === memoryHd :=
	Catch[
		InitializeWGL[errMsgHd];
		WGLMemoryInformation[errMsgHd, mem]
	]	
	
(******************************************************************************)

GPUCopyConstantMemory[direction_][memoryHd_, errMsgHd_, constmem_, regmem_] :=
	Catch[
		InitializeWGL[errMsgHd];
		WGLCopyConstantMemory[direction][errMsgHd, constmem, regmem]
	]
	
(******************************************************************************)

GPUBindTexture[memoryHd_, errMsgHd_, textmem_, mem_, args_List] :=
	Module[{addressMode, filterMode, flags},
		addressMode = getTextureOption["Address", args, mem];
		filterMode 	= getTextureOption["Filter", args, mem];
		flags 		= getTextureOption["Flags", args, mem];
		GPUBindTexture[memoryHd, errMsgHd, textmem, mem, {addressMode, filterMode, flags}]
	]
GPUBindTexture[memoryHd_, errMsgHd_, textmem_, mem_, {addressMode_, filterMode_, flags_}] :=
	Catch[
		InitializeWGL[errMsgHd];
		WGLBindTexture[errMsgHd, textmem, mem, addressMode, filterMode, flags]
	]

getTextureOption[opt_, {}, mem_] :=
	defaultTextureOption[opt, mem]
getTextureOption[opt_, rules_, mem_] :=
	Quiet@If[(opt /. rules) =!= opt,
		opt /. rules,
		defaultTextureOption[opt, mem]
	]

defaultTextureOption["Address", mem_] := 
	Module[{rank = WGLMemoryGetRank[mem]},
		ConstantArray["Fixed", rank]
	]
defaultTextureOption["Filter", mem_] = "Linear"  
defaultTextureOption["Flags", mem_] = "ReadAsInteger"

(******************************************************************************)

$TemporaryMemoryFlag = "TemporaryMemoryFlag";
$OutputMemoryFlag = "OutputMemoryFlag";
$OutputTensorFlag = "OutputTensorFlag";

(******************************************************************************)

Options[GPUFunctionLoad] = {
	"Platform" -> Automatic,
	"Device" -> Automatic,
	"Source" -> None,
	"BinaryQ" -> False,
	"BuildOptions" -> None,
	"Defines" -> {},
	"SystemDefines" -> Automatic,
	"SystemIncludeDirectories" -> {},
	"IncludeDirectories" -> {},
	"CompileOptions" -> {},
	"ShellCommandFunction" -> None,
	"ShellOutputFunction" -> None,
	"TargetPrecision" -> Automatic
}
(******************************************************************************)
(*Support File[..] as an argument*)
isFile[exp_,head_:CUDAFunctionLoad]:=StringQ[Quiet[getFile[exp,head]]];

getFile[exp_,head_:CUDAFunctionLoad]:=Block[
	{pos,file,hasProperHead},
	hasProperHead=MatchQ[Head[exp],String|System`File];
	pos=Position[exp, _String];
	file = Extract[exp,pos];
	If[Length[file]>=1 && hasProperHead,
		file[[1]]
		,
		Message[head::invfile, exp];
		$Failed
		]
	
]
(*********************************)
GPUFunctionLoad[functionHd_, progObjHd_, errMsgHd_, {}, ___] :=
	Throw[Message[errMsgHd::emptyprog]; $Failed]
GPUFunctionLoad[functionHd_, progObjHd_, errMsgHd_, "", ___] :=
	Throw[Message[errMsgHd::invsrc, ""]; $Failed]
GPUFunctionLoad[functionHd_, progObjHd_, errMsgHd_, prog:{_} /; !isFile[First@prog,functionHd], ___] :=
	Throw[Message[errMsgHd::invprog, First[prog]]; $Failed]
GPUFunctionLoad[functionHd_, progObjHd_, errMsgHd_, prog_ /; (!StringQ[prog] && !ListQ[prog] && Head[prog]=!=File), ___] :=
	Throw[Message[errMsgHd::invprog, prog]; $Failed]
GPUFunctionLoad[functionHd_, progObjHd_, errMsgHd_, prog:{_String} /; !FileExistsQ[First@prog], ___] :=
	Throw[Message[errMsgHd::nofile, First[prog]]; $Failed]
GPUFunctionLoad[functionHd_, progObjHd_, errMsgHd_, prog:({_String} | _String), name_ /; !StringQ[name], ___]:=
	Throw[Message[errMsgHd::kernnam, name]; $Failed]
GPUFunctionLoad[functionHd_, progObjHd_, errMsgHd_, prog:({_String} | _String), name_String, {}, ___]:=
	Throw[Message[errMsgHd::emptyargs]; $Failed]
GPUFunctionLoad[functionHd_,
				progObjHd_,
				errMsgHd_,
				progexp_File,
				kernelName_String,
				args0_List /; args0 =!= {},
				blockDimensions_,
				opts:OptionsPattern[]] :=GPUFunctionLoad[functionHd,progObjHd,errMsgHd,{progexp},kernelName,args0,blockDimensions,opts]
GPUFunctionLoad[functionHd_,
				progObjHd_,
				errMsgHd_,
				progexp_,
				kernelName_String,
				args0_List /; args0 =!= {},
				blockDimensions_,
				opts:OptionsPattern[]] :=
	Module[{progObj, platform, device, defines, systemDefines, args,
		    systemIncludeDirectories, includeDirectories, progType,
		    compileOptions, shellCommandFunction, shellOutputFunction,
		    targetPrecision, buildOptions, binaryQ, source, optRes, res,prog},
		prog=If[ListQ[progexp],{getFile[First[progexp],functionHd]},progexp];
		optRes = Catch[
			
			If[ListQ[prog] && !FileExistsQ[First[prog]],
				Throw[Message[errMsgHd::nofile, First[prog]]; $Failed]
			];
			If[!((IntegerQ[blockDimensions] && Positive[blockDimensions]) || (ListQ[blockDimensions] && blockDimensions =!= {})),
				Throw[Message[errMsgHd::invblk, blockDimensions]; $Failed]
			];
			
			InitializeWGL[GPUFunctionLoad, errMsgHd, opts];
			
			platform = GetAndCheckOption[GPUFunctionLoad, errMsgHd, "Platform", {opts}];
			device = GetAndCheckOption[GPUFunctionLoad, errMsgHd, "Device", {opts}];
			source = GetAndCheckOption[GPUFunctionLoad, errMsgHd, "Source", {opts}];
			binaryQ = GetAndCheckOption[GPUFunctionLoad, errMsgHd, "BinaryQ", {opts}];
			buildOptions = GetAndCheckOption[GPUFunctionLoad, errMsgHd, "BuildOptions", {opts}];
			
			
			defines = GetAndCheckOption[GPUFunctionLoad, errMsgHd, "Defines", {opts}];
			systemDefines = GetAndCheckOption[GPUFunctionLoad, errMsgHd, "SystemDefines", {opts}];
			systemIncludeDirectories = GetAndCheckOption[GPUFunctionLoad, errMsgHd, "SystemIncludeDirectories", {opts}];
			includeDirectories = GetAndCheckOption[GPUFunctionLoad, errMsgHd, "IncludeDirectories", {opts}];
			compileOptions = GetAndCheckOption[GPUFunctionLoad, errMsgHd, "CompileOptions", {opts}];
			shellCommandFunction = GetAndCheckOption[GPUFunctionLoad, errMsgHd, "ShellCommandFunction", {opts}];
			shellOutputFunction = GetAndCheckOption[GPUFunctionLoad, errMsgHd, "ShellOutputFunction", {opts}];
			targetPrecision = GetAndCheckOption[GPUFunctionLoad, errMsgHd, "TargetPrecision", {opts}];
		];
		If[optRes =!= $Failed,
			res = Catch[
				buildOptions = If[binaryQ,
					buildOptions,
					StringJoin[
						Riffle[
							Select[StringTrim /@ {
								compileOptions,
								systemIncludeDirectories,
								includeDirectories,
								systemDefines,
								defines
							}, # =!= ""&],
							" "
						]
					]
				];
				
				progType = Switch[prog,
					_List,
						Which[
							MemberQ[{"cubin", "ptx"}, FileExtension[First[prog]]],
								"BinaryFile",
							MemberQ[{"dll", "so", "dylib"}, FileExtension[First[prog]]],
								"LibraryFunction",
							True,
								"SourceFile"
						],
					_String,
						"Source"
				];
	
				args = Block[{$progType = "LibraryFunction"},
					ResolveType[#, targetPrecision === "Single"]& /@ args0
				];
				
				ValidFunctionArguments[progType][errMsgHd, args];
				ValidBlockDimensions[errMsgHd, blockDimensions];
	
				progObj = WGLAddProgram[progObjHd, errMsgHd, prog, kernelName, args, source, binaryQ, buildOptions];
				WGLSetFunction[WGLProgramGetType[progObj]][functionHd, errMsgHd, kernelName, prog];
				functionHd[
					progObj,
					kernelName,
					args,
					"Platform" -> platform,
					"Device" -> device,
					"Defines" -> defines,
					"SystemDefines" -> systemDefines,
					"SystemIncludeDirectories" -> systemIncludeDirectories,
					"IncludeDirectories" -> includeDirectories,
					"CompileOptions" -> compileOptions,
					"ShellCommandFunction" -> shellCommandFunction,
					"ShellOutputFunction" -> shellOutputFunction,
					"TargetPrecision" -> targetPrecision,
					"BlockDimensions" -> blockDimensions
				]
			];
			If[$ThisAPI === "OpenCL",
				If[shellCommandFunction =!= None,
					shellCommandFunction[buildOptions]
				];
				If[shellOutputFunction =!= None && (!ListQ[prog] || FreeQ[{"clb", "dll", "so", "dylib"}, ToLowerCase[FileExtension[First[prog]]]]),
					shellOutputFunction[WGLProgramGetBuildLog[errMsgHd, If[res === $Failed, $Failed, progObj], prog, buildOptions]]
				]
			]
		];
		If[optRes === $Failed,
			optRes,
			res
		]
	]

GPUFunctionLoad[functionHd_, progObjHd_, errMsgHd_, args___, opts:OptionsPattern[]] := 
	If[ArgumentCountQ[errMsgHd, Length[{args}], 4, 4],
		Throw[Message[errMsgHd::args, {args}]; $Failed],
		Throw[$Failed]
	]

(******************************************************************************)

GPUFunctionLaunch[functionHd_,
				  progObjHd_,
				  memoryHd_,
				  errMsgHd_,
				  _,
				  {}] :=
	Throw[Message[errMsgHd::emptycall]; $Failed]
GPUFunctionLaunch[functionHd_,
				  progObjHd_,
				  memoryHd_,
				  errMsgHd_,
				  {progObj:progObjHd_[progId_, opts___], kernelName_String, args_List, funOpts:OptionsPattern[]},
				  funargs:{callArgs__}] := 
	Module[{runRes, res, usingSinglePrecisionQ, kernelArgs, blockDimensions, gridDimensions, funType = WGLProgramGetType[progObj]},
		runRes = Check[Reap[Catch[
			
			InitializeWGL[GPUFunctionLaunch, errMsgHd, opts];
			
			If[!ValidLaunchArguments[errMsgHd, funType, memoryHd, args, funargs],
				Throw[$Failed]
			];

			blockDimensions = Flatten[List[("BlockDimensions" /. {funOpts})]];
			gridDimensions = Flatten[List[
				If[Length[funargs] === Length[args],
					GuessGridDimensions[errMsgHd, args, funargs],
					Last[funargs]	
				]
			]];
			
			If[Length[gridDimensions] === 1 && Length[blockDimensions] === 2,
				gridDimensions = {First[gridDimensions], Last[blockDimensions]}
			];
			
			If[MemberQ[gridDimensions, 0],
				Throw[Message[invgrd, gridDimensions]; $Failed]
			];

			Which[
				$ThisAPI === "CUDA" && Length[blockDimensions] < Length[gridDimensions],
					Throw[Message[errMsgHd::invgrdblk, gridDimensions, blockDimensions]; $Failed],
				$ThisAPI === "OpenCL" && Length[blockDimensions] =!= Length[gridDimensions],
					Throw[Message[errMsgHd::invgrdblk, gridDimensions, blockDimensions]; $Failed]
			];
			
			gridDimensions = MakeGridDimensionsMultipleOfBlockSize[errMsgHd, gridDimensions, blockDimensions];
			
			If[$ThisAPI === "CUDA",
				gridDimensions = MapThread[Divide, {gridDimensions, blockDimensions[[;;Length[gridDimensions]]]}] 
			];

			usingSinglePrecisionQ = ("TargetPrecision" /. {funOpts}) === "Single";

			WGLSetProgram[funType][progObjHd, errMsgHd, progObj];
			WGLSetFunction[funType][functionHd, errMsgHd, kernelName];
				
			kernelArgs = SetFunctionArguments[funType][memoryHd, errMsgHd, args, If[Length[funargs] === Length[args], funargs, Most[funargs]], usingSinglePrecisionQ];
			
			If[funType === "LibraryFunction",
				
				WGLLaunchFunction[funType][errMsgHd, progObj, kernelArgs, gridDimensions, blockDimensions]
				
				, (* Else *)
				WGLSetGridDimensions[funType][errMsgHd, gridDimensions];
				WGLSetBlockDimensions[funType][errMsgHd, blockDimensions];
				WGLLaunchFunction[funType][errMsgHd];
			]
			
		]], $Failed];

		res = Switch[runRes,
			$Failed,
				$Failed,
			{_Integer, ___},
				With[{tmp = GetMemory[First[runRes], False]},
					DeleteMemory[First[runRes]];
					tmp
				],
			_,
				Catch[
					GetKernelOutputs[memoryHd, errMsgHd, kernelArgs]
				]
		];

		If[ListQ[runRes],
			WGLDeleteMemory[errMsgHd, #]& /@ Flatten[Last[runRes]]
		];
		
		res
	]
	
(******************************************************************************)

GPUFunctionInformation[functionHd_, progObjHd_, errMsgHd_, fun:_[prog_, kernelName_String, args_List, funOpts:OptionsPattern[]]] /; Head[fun] === functionHd && Head[prog] === progObjHd :=
	Catch[
		InitializeWGL[errMsgHd];
		WGLProgramInformation[errMsgHd, prog] ~Join~ 
			{
				"Function" -> kernelName,
				"ArgumentTypes" -> args,
				"Options" -> If[WGLProgramGetType[prog] === "LibraryFunction", {}, {funOpts}]
			}
	]
GPUFunctionInformation[functionHd_, progObjHd_, errMsgHd_, arg_] /; Head[arg] =!= functionHd := Throw[Message[errMsgHd::invfun, arg]; $Failed]
GPUFunctionInformation[functionHd_, progObjHd_, errMsgHd_, args___] /; (ArgumentCountQ[errMsgHd, Length[{args}], 1, 1]; False) := Throw[$Failed]

(******************************************************************************)
GuessGridDimensions[errMsgHd_, templateArguments_List, funargs_List] :=
	MaximumGridDimensions[errMsgHd, Select[MapThread[GuessGridDimension[errMsgHd, #1, #2]&, {templateArguments, funargs}], ListQ]] 

GuessGridDimension[errMsgHd_, _List, arg_?Image`PossibleImageQ] :=
	Block[{img = ToImage[errMsgHd, arg], dims},
		ImageDimensions[img]
	]
GuessGridDimension[errMsgHd_, _List, arg_List] :=
	With[{dims = Dimensions[arg]},
		Flatten[
			Switch[ArrayDepth[arg],
				1,
					{dims},
				2,
					dims,
				_,
					Take[dims, 2]
			]
		]
	]
GuessGridDimension[errMsgHd_, _List, arg:_[_Integer, ___]] :=
	With[{dims = WGLMemoryGetDimensions[arg]},
		Flatten[
			Switch[Length[dims],
				1,
					{dims},
				2,
					dims,
				_,
					Take[dims, 2]
			]
		]
	]
GuessGridDimension[___] := None

MaximumGridDimensions[errMsgHd_, candidateDimensions_List] := 
	With[{res = Reverse[SortBy[candidateDimensions, {Apply[Times, #]&, ArrayDepth, First}]]},
		If[res === {},
			Throw[Message[errMsgHd::nogrdguess]; $Failed],
			First[res]
		]
	]

(******************************************************************************)

MakeGridDimensionsMultipleOfBlockSize[errMsgHd_, gridDimensions_, blockDimensions_] :=
	Which[
		Length[gridDimensions] > Length[blockDimensions],
			Throw[Message[errMsgHd::invgrblk, gridDimensions, blockDimensions]; $Failed],
		And@@MapThread[Divisible, {gridDimensions, blockDimensions[[;;Length[gridDimensions]]]}],
			gridDimensions,
		True,
			(* TODO: Maybe issue warning here. *)
			MapThread[Ceiling[#1/#2]*#2&, {gridDimensions, blockDimensions[[;;Length[gridDimensions]]]}]
	]
	
(******************************************************************************)

ClearAll[GetKernelOutput]
GetKernelOutput[memoryHd_, errMsgHd_, output_, "InputOutput" | "Output", memoryHd_] := output
GetKernelOutput[memoryHd_, errMsgHd_, output_, "InputOutput" | "Output", _] /; Head[output] === memoryHd := GPUGetMemory[memoryHd, errMsgHd, output]
GetKernelOutput[___] := None

ClearAll[GetKernelOutputs]
GetKernelOutputs[memoryHd_, errMsgHd_, outputs_List] := Select[GetKernelOutput[memoryHd, errMsgHd, Sequence@@#]& /@ outputs, Head[#] === memoryHd || ListQ[#] || ImageQ[#]&]

(******************************************************************************)

SetFunctionArguments[prog_][memoryHd_, errMsgHd_, functionArgs_List, callArguments_List, usingSinglePrecisionQ_] :=
	(
		WGLSetFunctionArgumentCount[prog][errMsgHd, Length[functionArgs] + 2];
		Select[MapThread[SetFunctionArgument[prog][memoryHd, errMsgHd, #1, #2, usingSinglePrecisionQ]&, {functionArgs, callArguments}], # =!= {}&]
	)
SetFunctionArgument[prog_][memoryHd_, errMsgHd_, functionArg:Verbatim[_Integer], callArg_Integer, usingSinglePrecisionQ_] :=
	(
		If[$64BitMint, 
			WGLSetFunctionLongArgument[prog][errMsgHd, callArg],
			WGLSetFunctionIntegerArgument[prog][errMsgHd, callArg]
		];
		{callArg, Integer}
	)
SetFunctionArgument[prog_][memoryHd_, errMsgHd_, "Integer32", callArg_Integer, usingSinglePrecisionQ_] :=
	(
		WGLSetFunctionIntegerArgument[prog][errMsgHd, callArg];
		{callArg, Integer}
	)
SetFunctionArgument[prog_][memoryHd_, errMsgHd_, "UnsignedInteger32", callArg_Integer, usingSinglePrecisionQ_] :=
	(
		WGLSetFunctionUnsignedIntegerArgument[prog][errMsgHd, callArg];
		{callArg, Integer}
	)
SetFunctionArgument[prog_][memoryHd_, errMsgHd_, "Integer64", callArg_Integer, usingSinglePrecisionQ_] :=
	(
		WGLSetFunctionLongArgument[prog][errMsgHd, callArg];
		{callArg, Integer}
	)
SetFunctionArgument[prog_][memoryHd_, errMsgHd_, functionArg:(Verbatim[_Real] | "Float" | "Double"), callArg:(_Real | _Integer), usingSinglePrecisionQ_] :=
	(
		WGLSetFunctionRealArgument[errMsgHd, callArg, 
			Switch[functionArg,
				"Float",
					WGLSetFunctionRealArgument[prog][errMsgHd, N@callArg, True],
				"Double",
					WGLSetFunctionRealArgument[prog][errMsgHd, N@callArg, False],
				_,
					WGLSetFunctionRealArgument[prog][errMsgHd, N@callArg, usingSinglePrecisionQ]
			]
		];
		{1.0*callArg, Real}
	)
SetFunctionArgument["LibraryFunction"][memoryHd_, errMsgHd_, {Verbatim[_Integer] | Verbatim[_Real] | Verbatim[_Complex], Verbatim[Blank[]] | _Integer, "Shared" | "Manual" | Automatic}, input_List, usingSinglePrecisionQ_] :=
	{input, List}
SetFunctionArgument["LibraryFunction"][memoryHd_, errMsgHd_, "Boolean", input:(True | False), usingSinglePrecisionQ_] :=
	{input, "Boolean"}
SetFunctionArgument["LibraryFunction"][memoryHd_, errMsgHd_, ("UTF8String" | "Boolean"), input_String, usingSinglePrecisionQ_] :=
	{input, "UTF8String"}
SetFunctionArgument[prog_][memoryHd_, errMsgHd_, functionArg:("Local" | "Shared"), callArg_Integer, usingSinglePrecisionQ_] :=
	(
		WGLSetFunctionLocalMemoryArgument[prog][memoryHd, errMsgHd, callArg];
		{callArg, functionArg}
	)
SetFunctionArgument[prog_][memoryHd_, errMsgHd_, functionArg:{("Local" | "Shared"), Verbatim[_Integer] | Verbatim[_Real] | Verbatim[_Complex] | "Integer32" | "Integer64" | "Float" | "Double"}, callArg_Integer, usingSinglePrecisionQ_] :=
	SetFunctionArgument[prog][memoryHd, errMsgHd, "Local", callArg * ResolveByteCount[Last[functionArg], usingSinglePrecisionQ], usingSinglePrecisionQ]
SetFunctionArgument[prog_][memoryHd_, errMsgHd_, functionArg:{(Texture | "Texture"), name_String, opts___}, callArg_, usingSinglePrecisionQ_] /; Head[callArg] === memoryHd :=
	Module[{type},
		type = If[MemberQ[{Automatic, None}, textureType[functionArg]],
			WGLMemoryGetType[callArg],
			textureType[functionArg]
		];
		If[!sameTypeQ[type, WGLMemoryGetType[callArg]],
			Throw[Message[errMsgHd::invtext, type, WGLMemoryGetType[callArg]]; $Failed]
		];
		If[textureType[functionArg] === None,
			GPUBindTexture[memoryHd, errMsgHd, name, callArg, {opts}],
			GPUBindTexture[memoryHd, errMsgHd, name, callArg, Rest[{opts}]]
		];
		{callArg, functionArg}
	]

textureType[{Texture | "Texture", name_}] := None
textureType[{Texture | "Texture", name_, type_, ___}] := type	

SetFunctionArgument[prog_][memoryHd_, errMsgHd_, functionArg_List, input:(_List | _?Image`PossibleImageQ), usingSinglePrecisionQ_] :=
	Block[{mem},
		mem = GPUAddMemory[
			memoryHd,
			errMsgHd,
			input,
			FunctionArgumentGetType[errMsgHd, functionArg],
			If[Head[input] === List,
				{List},
				{Image}
			],
			{},
			"TargetPrecision" -> If[usingSinglePrecisionQ, "Single", "Double"]
		];
		If[Head[mem] =!= memoryHd,
			Throw[$Failed]
		];

		Sow[mem, $TemporaryMemoryFlag];
		
		WGLSetFunctionMemoryArgument[prog][memoryHd, errMsgHd, WGLMemoryGetID[mem], FunctionArgumentGetInputOutputParameter[functionArg]];
		{mem, FunctionArgumentGetInputOutputParameter[functionArg], Head[input]}
	]
SetFunctionArgument[prog_][memoryHd_, errMsgHd_, functionArg_List, mem_, usingSinglePrecisionQ_] /; Head[mem] === memoryHd :=
	(
		WGLSetFunctionMemoryArgument[prog][memoryHd, errMsgHd, WGLMemoryGetID[mem], FunctionArgumentGetInputOutputParameter[functionArg]];
		{mem, FunctionArgumentGetInputOutputParameter[functionArg], memoryHd}
	)
SetFunctionArgument[prog_][memoryHd_, errMsgHd_, args___] := Throw[Message[errMsgHd::invfunarg, prog, {args}]; $Failed]

(******************************************************************************)

FunctionArgumentGetType[errMsgHd_, {x:(Verbatim[_Integer] | Verbatim[_Real] | Verbatim[_Complex]), ___}] := First[x]
FunctionArgumentGetType[errMsgHd_, {x_String, ___}] := x
FunctionArgumentGetType[errMsgHd_, {Texture | "Texture", ___}] := Texture
FunctionArgumentGetType[errMsgHd_, opt_] := Throw[Message[errMsgHd::argtype, opt]; $Failed]

(******************************************************************************)

FunctionArgumentGetInputOutputParameter[_[_]] := "InputOutput"
FunctionArgumentGetInputOutputParameter[_[_, inputOutput_String]] := inputOutput
FunctionArgumentGetInputOutputParameter[_[_, Verbatim[_] | _Integer]] := "InputOutput"
FunctionArgumentGetInputOutputParameter[_[_, _, inputOutput_]] := inputOutput 

(******************************************************************************)

ValidLaunchArguments[errMsgHd_, funType_, memoryHd_, functionArgs_List, callArguments_List] :=
	Which[
		Length[functionArgs] === Length[callArguments],
			MapThread[ValidLaunchArgument[errMsgHd, funType, memoryHd, #1, #2]&, {functionArgs, callArguments}],
		Length[functionArgs] === Length[callArguments] - 1,
			MapThread[ValidLaunchArgument[errMsgHd, funType, memoryHd, #1, #2]&, {functionArgs, Most[callArguments]}] && ValidGridDimensions[errMsgHd, Last[callArguments]],
		True,
			Throw[Message[errMsgHd::invarglen, Length[functionArgs], Length[callArguments]]; $Failed]
	]

ValidLaunchArgument[errMsgHd_, funType_, memoryHd_, functionArg:(Verbatim[_Integer] | "Integer32" | "UnsignedInteger32" | "Integer64"), callArg_Integer] := 
	If[Developer`MachineIntegerQ[callArg],
		True,
		Throw[Message[errMsgHd::machint, callArg]; $Failed]
	]
ValidLaunchArgument[errMsgHd_, funType_, memoryHd_, functionArg:(Verbatim[_Real] | "Float" | "Double"), callArg:(_Real | _Integer)] :=
	If[Developer`MachineRealQ[callArg] || Developer`MachineIntegerQ[callArg],
		True,
		Throw[Message[errMsgHd::machreal, callArg]; $Failed]
	]
ValidLaunchArgument[errMsgHd_, funType_, memoryHd_, functionArg:("Local" | "Shared"), callArg:(_Real | _Integer)] := 
	If[Developer`MachineIntegerQ[callArg] || Developer`MachineRealQ[callArg],
		True,
		Throw[Message[errMsgHd::machnum, callArg]; $Failed]
	]
ValidLaunchArgument[errMsgHd_, funType_, memoryHd_, functionArg:{("Local" | "Shared"), Verbatim[_Integer] | Verbatim[_Real] | "Integer32" | "Integer64" | "Float" | "Double"}, callArg:(_Real | _Integer)] :=
	If[Developer`MachineIntegerQ[callArg] || Developer`MachineRealQ[callArg],
		True,
		Throw[Message[errMsgHd::machnum, callArg]; $Failed]
	]
ValidLaunchArgument[errMsgHd_, "LibraryFunction", memoryHd_, {Verbatim[_Integer] | Verbatim[_Real] | Verbatim[_Complex], Verbatim[Blank[_]] | _Integer, "Shared" | "Manual" | Automatic}, callArg_List] :=
	If[Developer`MachineIntegerQ[Last[callArg]] || Developer`MachineRealQ[Last[callArg]],
		True,
		Throw[Message[errMsgHd::machnum, Last[callArg]]; $Failed]
	]
ValidLaunchArgument[errMsgHd_, "LibraryFunction", memoryHd_, "UTF8String", callArg_] := 
	If[StringQ[callArg],
		True,
		Throw[Message[errMsgHd::nostr, callArg]; $Failed]
	]
ValidLaunchArgument[errMsgHd_, "LibraryFunction", memoryHd_, "Boolean", arg_] := 
	If[MemberQ[{True, False}, arg],
		True,
		Throw[Message[errMsgHd::nobool, arg]; $Failed]
	]
ValidLaunchArgument[errMsgHd_, funType_, memoryHd_, functionArg_List, callArg_List] :=
	Module[{type, tensorType},
		If[Length[functionArg] > 1 && IntegerQ[functionArg[[2]]],
			If[ArrayDepth[callArg] =!= functionArg[[2]],
				Throw[Message[errMsgHd::invrnk, functionArg, ArrayDepth[callArg]]; $Failed]
			]
		];
		type = ResolveType[First[functionArg]];
		tensorType = GPUTools`Utilities`ArrayType[callArg];
		Which[
			type === tensorType || GetTensorType[type] === tensorType,
				True,
			GetTensorType[type] === Real && GetTensorType[tensorType] === Integer,
				True,
			True,
				Throw[Message[errMsgHd::tenstype, type, tensorType]; $Failed]
		]
	]
ValidLaunchArgument[errMsgHd_, funType_, memoryHd_, functionArg_List, callArg_?Image`PossibleImageQ] := True (* We check for this later in the code *)
ValidLaunchArgument[errMsgHd_, funType_, memoryHd_, functionArg_List, callArg_] /; Head[callArg] === memoryHd :=
	Module[{type, memType},
		If[Length[functionArg] > 1 && IntegerQ[functionArg[[2]]],
			If[WGLMemoryGetRank[callArg] =!= functionArg[[2]],
				Throw[Message[errMsgHd::invrnk, functionArg, WGLMemoryGetRank[callArg]]; $Failed]
			]
		];
		type = FunctionArgumentGetType[errMsgHd, functionArg];
		memType = WGLMemoryGetType[callArg];
		If[funType === "LibraryFunction" || sameTypeQ[type, memType] || ToString[type] == memType || MemberQ[{Real, Complex}, type],
			True,
			Throw[Message[errMsgHd::memtype, type, memType]; $Failed]
		]
	]
ValidLaunchArgument[errMsgHd_, funType_, memoryHd_, functionArg_, callArg_] := Throw[Message[errMsgHd::invcallarg, functionArg, callArg]; $Failed]


sameTypeQ[a_, b_] := 
	WGLSameTypeQ[ResolveType[a], ResolveType[b]]
	
(******************************************************************************)

ValidGridDimensions[errMsgHd_, gridDimensions_List] := 
	If[gridDimensions =!= {} && Length[gridDimensions] <= 2 && VectorQ[gridDimensions, Developer`MachineIntegerQ[#] && Positive[#]&],
		True,
		Throw[Message[errMsgHd::invgrd, gridDimensions]; $Failed]
	]
ValidGridDimensions[errMsgHd_, gridDimensions_Integer?Positive] := If[Developer`MachineIntegerQ[gridDimensions], True, Throw[Message[errMsgHd::invgrd, gridDimensions]; $Failed]]
ValidGridDimensions[errMsgHd_, gridDimensions___] := Throw[Message[errMsgHd::invgrd, {gridDimensions}]; $Failed]

(******************************************************************************)

ValidBlockDimensions[errMsgHd_, blockDimensions_List] := 
	If[blockDimensions =!= {} && Length[blockDimensions] <= 3 && VectorQ[blockDimensions, Developer`MachineIntegerQ[#] && Positive[#]&],
		True,
		Throw[Message[errMsgHd::invblk, blockDimensions]; $Failed]
	]
ValidBlockDimensions[errMsgHd_, blockDimensions_Integer?Positive] := If[Developer`MachineIntegerQ[blockDimensions], True, Throw[Message[errMsgHd::invblk, blockDimensions]; $Failed]]
ValidBlockDimensions[errMsgHd_, blockDimensions___] := Throw[Message[errMsgHd::invblk, {blockDimensions}]; $Failed]

(******************************************************************************)

ValidFunctionArguments[progType_][errMsgHd_, args_List] :=
	Scan[ValidFunctionArgument[progType][errMsgHd, #]&, args]
	
ValidFunctionArgument[progType_][errMsgHd_, arg:(Integer | Verbatim[_Integer] | Real | Verbatim[_Real] | "Byte" | "UnsignedByte" | "Bit16" | "UnsignedBit16" |"Integer32" | "UnsignedInteger32" | "Integer64" | "Float" | "Double")] := True
ValidFunctionArgument["LibraryFunction"][errMsgHd_, Integer | Real | Complex] := True
ValidFunctionArgument["LibraryFunction"][errMsgHd_, "Boolean" | "UTF8String"] := True
ValidFunctionArgument[progType_][errMsgHd_, arg_List] :=
	If[Catch[ValidFunctionTensorArgument[progType][arg]] === $Failed,
		Throw[Message[errMsgHd::invtypspec, arg]; $Failed],
		True
	]
ValidFunctionArgument[progType_][errMsgHd_, arg:("Shared" | "Local")] := True
ValidFunctionArgument[progType_][errMsgHd_, arg:{"Shared" | "Local", Verbatim[_Integer] | Verbatim[_Real] | "Integer32" | "Integer64" | "Float" | "Double"}] := True
ValidFunctionArgument[progType_][errMsgHd_, arg:{Texture | "Texture", name_String}] /; $ThisAPI === "CUDA" := True
ValidFunctionArgument[progType_][errMsgHd_, arg:{Texture | "Texture", name_String, "Byte" | "UnsignedByte" | "Bit16" | "UnsignedBit16" | "Integer32" | "Float"}] /; $ThisAPI === "CUDA" := True
ValidFunctionArgument[progType_][errMsgHd_, arg:{Texture | "Texture", name_String, "Byte" | "UnsignedByte" | "Bit16" | "UnsignedBit16" | "Integer32" | "Float", args___}] /; $ThisAPI === "CUDA" := TrueQ[And@@(TextureArgumentQ /@ args)]
ValidFunctionArgument[progType_][errMsgHd_, arg:{Texture | "Texture", name_String, type_?WGLTypeQ, args___}] /; $ThisAPI === "CUDA" :=
	ValidFunctionArgument[progType][errMsgHd, {Texture, name, ResolveBaseType[type], args}] 
ValidFunctionArgument[progType_][errMsgHd_, arg_] := Throw[Message[errMsgHd::invtypspec, arg]; $Failed]

(******************************************************************************)

TextureArgumentQ["Flags" -> ("ReadAsInteger" | "NormalizeCoordinates")] = True
TextureArgumentQ["Filter" -> ("Point" | "Linear")] = True
TextureArgumentQ["AdressMode" -> mode_String] := AddressModeQ[mode]
TextureArgumentQ["AdressMode" -> modes_List] := TrueQ[And@@(AddressModeQ /@ modes)]
TextureArgumentQ[___] := False

(******************************************************************************)

TensorArgumentQ[Verbatim[_Integer] | Verbatim[_Real] | Verbatim[_Complex]] := True
TensorArgumentQ[type_ /; MemberQ[WGLTypes, type]] := TensorArgumentQ[type] = True
TensorArgumentQ[___] := Throw[$Failed]

(******************************************************************************)

ValidFunctionTensorArgument[progType_][{type_?TensorArgumentQ}] := True
ValidFunctionTensorArgument[progType_][{type_?TensorArgumentQ, rank:(Verbatim[Blank[]] | _Integer?Positive)}] := True
ValidFunctionTensorArgument[progType_][{type_?TensorArgumentQ, inputOutputParameter:("Input" | "Output" | "InputOutput")}] := True
ValidFunctionTensorArgument[progType_][{type_?TensorArgumentQ, rank:(Verbatim[Blank[]] | _Integer?Positive), inputOutputParameter:("Input" | "Output" | "InputOutput")}] := True
ValidFunctionTensorArgument["LibraryFunction"][{type_?TensorArgumentQ, rank:(Verbatim[Blank[]] | _Integer?Positive), libraryFunctionParameters:("Shared" | "Manual" | Automatic)}] := True
ValidFunctionTensorArgument[progType_][___] := Throw[$Failed]

(******************************************************************************)

ConvertToLibraryFunctionArguments[args_] := ConvertToLibraryFunctionArgument /@ args

ConvertToLibraryFunctionArgument[{Verbatim[_Integer] | Verbatim[_Real] | Verbatim[_Complex]}] := _Integer
ConvertToLibraryFunctionArgument[{type_ /; MemberQ[WGLTypes, type]}] := _Integer
ConvertToLibraryFunctionArgument[{Verbatim[_Integer] | Verbatim[_Real] | Verbatim[_Complex], _Integer | Verbatim[_]}] := _Integer
ConvertToLibraryFunctionArgument[{type_ /; MemberQ[WGLTypes, type], _Integer | Verbatim[_]}] := _Integer
ConvertToLibraryFunctionArgument[{Verbatim[_Integer] | Verbatim[_Real] | Verbatim[_Complex], _Integer | Verbatim[_], "Input" | "Output" | "InputOutput"}] := _Integer
ConvertToLibraryFunctionArgument[{Verbatim[_Integer] | Verbatim[_Real] | Verbatim[_Complex], "Input" | "Output" | "InputOutput"}] := _Integer
ConvertToLibraryFunctionArgument[{type_ /; MemberQ[WGLTypes, type], _Integer | Verbatim[_], "Input" | "Output" | "InputOutput"}] := _Integer
ConvertToLibraryFunctionArgument[{type_ /; MemberQ[WGLTypes, type], "Input" | "Output" | "InputOutput"}] := _Integer
ConvertToLibraryFunctionArgument[val:{Verbatim[_Integer] | Verbatim[_Real] | Verbatim[_Complex], _Integer | Verbatim[_], "Shared" | "Manual" | Automatic}] := val
ConvertToLibraryFunctionArgument[val:(Verbatim[_Integer] | Verbatim[_Real] | Verbatim[_Complex])] := val
ConvertToLibraryFunctionArgument["Integer" | "Integer32" | "Integer64"] := _Integer
ConvertToLibraryFunctionArgument["Float" | "Double" | "Real"] := _Real
ConvertToLibraryFunctionArgument["ComplexFloat" | "ComplexDouble" | "Complex"] := _Complex
ConvertToLibraryFunctionArgument["Local" | "Shared"] := _Integer
ConvertToLibraryFunctionArgument[{"Local" | "Shared", _Integer}] := _Integer
ConvertToLibraryFunctionArgument["UTF8String"] := "UTF8String"
ConvertToLibraryFunctionArgument["Boolean"] := "Boolean"
ConvertToLibraryFunctionArgument[args___] := Throw[Message[General::optx, {args}]; $Failed]

(******************************************************************************)

stripDuplicates[{}] := {}
stripDuplicates[{r:Rule[x_, y_], xs___}] := Prepend[stripDuplicates[Select[{xs}, First[#] =!= x &]], r]
	
(******************************************************************************)

FormatIncludeDirectories[dir_] := "-I" <> ToString[dir]
FormatIncludeDirectories[dirs_List] := StringJoin[Riffle[FormatIncludeDirectories /@ dirs, " "]]

(******************************************************************************)

FormatDefines[def_String] := "-D" <> ToString[def]
FormatDefines[Rule[def_, val_]] := "-D" <> ToString[def] <> "=" <> ToString[val]
FormatDefines[defs_List] := StringJoin[Riffle[FormatDefines /@ defs, " "]]

(******************************************************************************)

GetAndCheckOption[head_, errMsgHd_, n:"Platform", opts_] /; $ThisAPI === "CUDA" := 1	
GetAndCheckOption[head_, errMsgHd_, n:"Platform", opts_] /; $ThisAPI === "OpenCL" :=
	GetAndCheckOption[head, errMsgHd, n, opts] =
	With[{opt = OptionValue[head, {opts}, n]},
		Which[
			ValidPlatformQ[errMsgHd, opt],
				opt,
			opt === Automatic,
				Which[
					$ThisAPI === "CUDA",
						1,
					IntegerQ[$WGLPlatform] && $ThisAPI === "CUDA",
						$WGLPlatform,
					IntegerQ[OpenCLLink`$OpenCLPlatform] && $ThisAPI === "OpenCL",
						OpenCLLink`$OpenCLPlatform,
					True,
						Automatic
				],
			True,
				Throw[Message[errMsgHd::invplt, opt]; $Failed]
		]
	]
GetAndCheckOption[head_, errMsgHd_, n:"Device", opts_] :=
	GetAndCheckOption[head, errMsgHd, n, opts] =
	With[{opt = OptionValue[head, {opts}, n],
		  plt = GetAndCheckOption[head, errMsgHd, "Platform", opts]
		 },
		Which[
			ValidDeviceQ[errMsgHd, plt, opt],
				opt,
			opt === Automatic,
				Which[
					IntegerQ[$WGLDevice],
						$WGLDevice,
					IntegerQ[CUDALink`$CUDADevice] && $ThisAPI === "CUDA",
						CUDALink`$CUDADevice,
					IntegerQ[OpenCLLink`$OpenCLDevice] && $ThisAPI === "OpenCL",
						OpenCLLink`$OpenCLDevice,
					True,
						Automatic
				],
			True,
				Throw[Message[errMsgHd::invdev, opt]; $Failed]
		]
	]
GetAndCheckOption[head:GPUFunctionLoad, errMsgHd_, n:"Defines", opts_] :=
	GetAndCheckOption[head, errMsgHd, n, opts] =
	With[{opt = OptionValue[head, {opts}, n]},
		If[StringQ[opt] || opt === {} || (ListQ[opt] && VectorQ[opt, Head[#] == Rule || IntegerQ[#] || StringQ[#] &]),
			FormatDefines[opt],
			Throw[Message[errMsgHd::def, opt]; $Failed]
		]
	]
GetAndCheckOption[head:GPUFunctionLoad, errMsgHd_, n:"SystemDefines", opts_] :=
	GetAndCheckOption[head, errMsgHd, n, opts] =
	With[{opt = OptionValue[head, {opts}, n]},
		Which[
			StringQ[opt] || opt === {} || (ListQ[opt] && VectorQ[opt, Head[#] == Rule || IntegerQ[#] || StringQ[#] &]),
				FormatDefines[opt],
			opt === Automatic,
				With[{trgtPres = GetAndCheckOption[head, errMsgHd, "TargetPrecision", opts]},
					FormatDefines[
						Join[
							{
								"USING_" <> ToUpperCase[$ThisAPI] <> "_FUNCTION" -> 1
							}, 
							{
								If[$ThisAPI === "OpenCL",
									"OPENCLLINK_USING_" <> ToUpperCase[ResolveVendorName[$ThisAPI]],
									{}
								]
							},
							{
								"mint" -> If[$64BitMint,
									Switch[$SystemID,
										"Windows-x86-64" | "Linux-x86-64" | "MacOSX-x86-64",
											"long",
										_,
											"int"
									],
									"int"
								]
							},
							If[trgtPres === "Single",
								{"Real_t" -> "float"},
								{"Real_t" -> "double", "USING_DOUBLE_PRECISIONQ" -> 1}
							]
						]
					]
				],
			True,
				Throw[Message[errMsgHd::sysdef, opt]; $Failed]
		]
	]
GetAndCheckOption[head:GPUFunctionLoad, errMsgHd_, n:"SystemIncludeDirectories", opts_]  :=
	GetAndCheckOption[head, errMsgHd, n, opts] =
	With[{opt = OptionValue[head, {opts}, n]},
		If[StringQ[opt] || opt === {} || VectorQ[opt, StringQ],
			FormatIncludeDirectories[opt],
			Throw[Message[errMsgHd::sysinc, opt]; $Failed]
		]
	]
GetAndCheckOption[head:GPUFunctionLoad, errMsgHd_, n:"IncludeDirectories", opts_] :=
	GetAndCheckOption[head, errMsgHd, n, opts] =
	With[{opt = OptionValue[head, {opts}, n]},
		If[StringQ[opt] || opt === {} || VectorQ[opt, StringQ],
			FormatIncludeDirectories[opt],
			Throw[Message[errMsgHd::inc, opt]; $Failed]
		]
	]
GetAndCheckOption[head:GPUFunctionLoad, errMsgHd_, n:"CompileOptions", opts_] :=
	GetAndCheckOption[head, errMsgHd, n, opts] =
	With[{opt = OptionValue[head, {opts}, n]},
		Which[
			StringQ[opt] || ListQ[opt],
				StringJoin[Riffle[Flatten[{opt}], " "]],
			opt === {} || opt == None,
				"",
			True,
				Throw[Message[errMsgHd::cmpopt, opt]; $Failed]
		]
	]
GetAndCheckOption[head:GPUFunctionLoad, errMsgHd_, n:"ShellCommandFunction", opts_] :=
	With[{opt = OptionValue[head, {opts}, n]},
		If[MemberQ[{Symbol, Function}, Head[opt]],
			opt,
			Throw[Message[errMsgHd::shcmdfun, opt]; $Failed]
		]
	]
GetAndCheckOption[head:GPUFunctionLoad, errMsgHd_, n:"ShellOutputFunction", opts_] :=
	With[{opt = OptionValue[head, {opts}, n]},
		If[MemberQ[{Symbol, Function}, Head[opt]],
			opt,
			Throw[Message[errMsgHd::shoutfun, opt]; $Failed]
		]
	]
GetAndCheckOption[head_, errMsgHd_, n:"TargetPrecision", opts_] :=
	GetAndCheckOption[head, errMsgHd, n, opts] =
	Module[{platform, device, opt = OptionValue[head, {opts}, n]},
		platform = GetAndCheckOption[head, errMsgHd, "Platform", opts];
		device = GetAndCheckOption[head, errMsgHd, "Device", opts];
		Which[
			opt === Automatic,
				With[{supdbl = SupportsDoublePrecisionQ[$ThisHead, platform, device]},
					 If[supdbl,
					 	"Double",
					 	"Single"
					 ]
				],
			MemberQ[{"Single", "Double"}, opt],
				opt,
			True,
				Throw[Message[errMsgHd::invtrgtprs, opt]; $Failed]
		]
	]

GetAndCheckOption[head_, errMsgHd_, n:"Source", opts_] :=
	With[{opt = OptionValue[head, {opts}, n]},
		Which[
			StringQ[opt] || ListQ[opt]||Head[opt]===File,
				opt,
			opt === None,
				"",
			True,
				Throw[Message[errMsgHd::invsrc, opt]; $Failed]
		]
	]
	
GetAndCheckOption[head_, errMsgHd_, n:"BinaryQ", opts_] :=
	GetAndCheckOption[head, errMsgHd, n, opts] =
	With[{opt = OptionValue[head, {opts}, n]},
		If[MemberQ[{True, False}, opt],
			opt,
			Throw[Message[errMsgHd::invbinq, opt]; $Failed]
		]
	]
	
GetAndCheckOption[head_, errMsgHd_, n:"BuildOptions", opts_] :=
	GetAndCheckOption[head, errMsgHd, n, opts] =
	With[{opt = OptionValue[head, {opts}, n]},
		Which[
			StringQ[opt],
				opt,
			opt === None,
				"",
			True,
				Throw[Message[errMsgHd::invbuildopts, opt]; $Failed]
		]
	]

(******************************************************************************)

SetAttributes[GPUProfile, {HoldAll}]
Options[GPUProfile] := {"VisualizationFunction" -> "TimeLine"}
GPUProfile[errMsgHd_, Unevaluated[args___], opts:OptionsPattern[]] :=
	Module[{res, visualizationFunction},
		visualizationFunction = GetAndCheckOption[GPUProfile, errMsgHd, "VisualizationFunction", {opts}];
		WGLClearProfiler[errMsgHd];
		WGLEnableProfiler[errMsgHd];
		args;
		WGLDisableProfiler[errMsgHd];
		res = WGLGetProfilerInformation[errMsgHd];
		visualizeProfile[res, visualizationFunction]
	]

visualizeProfile[prof_, viz_:"TimeLine"] :=
	Module[{profObj, res},
		profObj = initializeProfile[];
				
		profObj["ProfileElements"] = prof;
		
		setProfileCategories[profObj];
		setProfileColorScheme[profObj];
		
		makeProfileTreeGraph[profObj];
		
		makeProfileElementAccessors[profObj];
		
		res = Switch[viz,
			"TimeLine",
				makeRectangleData[profObj];
				drawTimeLine[profObj],
			"Pie" | "PieChart" | PieChart,
				drawPieChart[profObj],
			"DetailedPie" | "DetailedPieChart",
				drawDetailedPieChart[profObj],
			"Raw" | None,
				makeRectangleData[profObj];
				res = profObj["ProfileElements"]
		];
		Quiet[Remove[profObj]];
		res
	]

initializeProfile[] :=
	Module[{profObj},
		profObj["NumberOfLevels"] = 0;
		profObj["ColorScheme"] = "DarkBands";
		profObj["TimeEpsilon"] = $MachineEpsilon;
		profObj["Opacity"] = 1.0;
		profObj
	]
	
makeProfileTreeGraph[profObj_] :=
	(
		profObj["EdgeList"] = Table[
			DirectedEdge["ParentID" /. pe, "ID" /. pe],
			{pe, profObj["ProfileElements"]}
		];
		profObj["TreeGraph"] = TreeGraph[profObj["EdgeList"]]
	)
	
makeProfileElementAccessors[profObj_] :=
	Do[
		profObj["ProfileElements", ii] = profObj["ProfileElements"][[ii]],
		{ii, Length[profObj["ProfileElements"]]}
	]

getProfileElementDepth[profObj_, 0] := 0
getProfileElementDepth[profObj_, ii_] :=
	Module[{id = ii, depth = 1},
		While[getProfileElementParent[profObj, id] != 0,
			depth++;
			id = getProfileElementParent[profObj, id]
		];
		depth
	]

getProfileElementParent[profObj_, 0] := 0
getProfileElementParent[profObj_, ii_] :=
	"ParentID" /. profObj["ProfileElements", ii]
	
getProfileElementElapsedTime[profObj_, ii_] :=
	("Elapsed" /. profObj["ProfileElements", ii]) + profObj["TimeEpsilon"]
	
getProfileElementStartTime[profObj_, ii_] :=
	"Start" /. profObj["ProfileElements", ii]

getProfileElementEndTime[profObj_, ii_] :=
	"End" /. profObj["ProfileElements", ii]

getProfileElementCategory[profObj_, ii_] :=
	"Category" /. profObj["ProfileElements", ii]
	
getProfileElementChildren[profObj_, ii_] :=
	Module[{graph = profObj["TreeGraph"]},
		Select[EdgeList[NeighborhoodGraph[graph, ii]], First[#] == ii&]
	]

leafNodeQ[profObj_, ii_] :=
	getProfileElementChildren[profObj, ii] === {}

setProfileCategories[profObj_] :=
	(
		profObj["Categories"] =
			Union["Category" /. profObj["ProfileElements"]]
	)

setProfileColorScheme[profObj_] :=
	Module[{presetCategories, categories, numColors},
		presetCategories = {"Memory", "Compute" "Launch", "Initialize", "Other"};
		categories = Complement[profObj["Categories"], presetCategories];
		numColors = Length[categories] + 1;
		profObj["Colors"] = DeleteDuplicates[
			Flatten[{
				presetColors[presetCategories],
				MapIndexed[
					(#1 -> ColorData[profObj["ColorScheme"]][1.25 * First[#2] / numColors]) &,
					categories
				]
			}],
			First[#1] == First[#2]&
		]
	]

presetColors[presetCategories_] :=
	{
		"Other" -> LightGray,
		MapIndexed[(#1 -> ColorData[6][First[#2]])&, presetCategories]
	}
	
getProfileElementsInLevel[projObj_, 0] := {}
getProfileElementsInLevel[profObj_, level_] :=
	profObj["ProfileElements", #]& /@ getIDsInLevel[profObj, level]

getIDsInLevel[profObj_, 0] := {0}
getIDsInLevel[profObj_, level_] :=
	Module[{res},
		res = Last[
			Reap[
				DepthFirstScan[
					profObj["TreeGraph"],
					0,
					"DiscoverVertex" -> (If[#3 == level && #1 =!= 0, Sow[#1]]&)
				];
			]
		];
		If[res === {},
			res,
			First[res]
		]
	]

makeRectangleData[profObj_] :=
	Module[{totalTime, depth, firstElement, firstStart, start},
		totalTime = getProfileElementEndTime[profObj, Length[profObj["ProfileElements"]]] - getProfileElementStartTime[profObj, 1];
		firstElement = getProfileElementDepth[profObj, 1];
		firstStart = getProfileElementStartTime[profObj, 1];
		Table[
			depth = getProfileElementDepth[profObj, ii];
			profObj["RectangleElements", ii] = Rectangle[
				{
					start = (getProfileElementStartTime[profObj, ii] - firstStart) / totalTime,
					0
				},
				{
					start + (getProfileElementElapsedTime[profObj, ii] / totalTime),
					0.5 * If[depth == 1,
						1,
						1.0 - (depth / (depth + 1))/1.2
					] / GoldenRatio
				}
			],
			{ii, Length[profObj["ProfileElements"]]}
		]
	]

getProfileElementColor[profObj_, ii_] :=
	getProfileElementCategory[profObj, ii] /. profObj["Colors"]
		
makeTooltipLabel[profObj_, ii_] := Column[{
		Row[{"Category: " getProfileElementCategory[profObj, ii]}],
		Row[{"Elapsed Time: ", getProfileElementElapsedTime[profObj, ii]}]
	}]

drawTimeLine[profObj_] :=
	Graphics[
		{
			EdgeForm[
				Directive[
					Which[
						Length[profObj["ProfileElements"]] < 100,
							Thick,
						Length[profObj["ProfileElements"]] < 1000,
							Thin,
						True,
							None
					],
					LightGray
				]
			],
			Opacity[profObj["Opacity"]],
			{
				"Other" /. profObj["Colors"],
				Rectangle[
					{0, 0},
					{1, 0.5/GoldenRatio}
				]
			},
			Table[
				{
					getProfileElementColor[profObj, ii],
					Tooltip[
						profObj["RectangleElements", ii],
						makeTooltipLabel[profObj, ii]
					]
				},
				{ii, Length[profObj["ProfileElements"]]}
			]
		},
		ContentSelectable -> False,
		ImageSize -> {Full, Scaled[0.1]},
		Axes -> {True, False}
	]

drawPieChart[profObj_] :=
	Module[{level1Ids, pieData, times, categories},
		level1Ids = getIDsInLevel[profObj, 1];
		pieData = Table[
			{
				getProfileElementElapsedTime[profObj, ii],
				getProfileElementCategory[profObj, ii]
			},
			{ii, level1Ids}
		];
		pieData = GatherBy[pieData, Last];
		times = Total /@ Map[First, pieData, {2}];
		categories = First /@ Map[Last, pieData, {2}];
		PieChart[
			MapThread[
				Tooltip[#1, #2]&,
				{times, categories}
			],
			ChartStyle -> Map[# /. profObj["Colors"]&, categories],
			SectorOrigin -> {Automatic, 1},
			ChartBaseStyle -> EdgeForm[Thin]
		]
	]

drawDetailedPieChart[profObj_] :=
	Module[{level1Ids, pieData, colors},
		level1Ids = getIDsInLevel[profObj, 1];
		pieData = Table[
			{
				getProfileElementElapsedTime[profObj, ii],
				getProfileElementCategory[profObj, ii]
			},
			{ii, level1Ids}
		];
		colors = Table[
			getProfileElementColor[profObj, ii],
			{ii, level1Ids}
		];
		PieChart[
			Tooltip@@@pieData,
			ChartStyle -> colors,
			SectorOrigin -> {Automatic, 1},
			ChartBaseStyle -> EdgeForm[Thin]
		]
	]

GetAndCheckOption[head:GPUProfile, errMsgHd_, n:"VisualizationFunction", opts_] :=
	With[{opt = OptionValue[head, {opts}, n]},
		If[MemberQ[{None, "TimeLine", PieChart, "DetailedPieChart"}, opt],
			opt,
			Throw[Message[errMsgHd::vizf, opt]; $Failed]
		]
	]
(******************************************************************************)

WGLQuery[errMsgHd_, WGLQ_] :=
	If[!WGLQ[],
		If[FailReason[errMsgHd],
			Throw[Message[errMsgHd::nodev]; $Failed],
			Throw[$Failed]
		],
		(# -> WGLQuery[errMsgHd, #])& /@ Range["Number Of Platforms" /. PlatformCount[errMsgHd]]
	]
WGLQuery[errMsgHd_, query:("CurrentMemoryUsage" | "MaximumUsableMemory")] :=
	With[{res = If[query == "CurrentMemoryUsage", cGetCurrentMemoryUsage[$SingleOrDoubleLibrary][], cGetMaximumUsableMemory[$SingleOrDoubleLibrary][]]},
		If[SuccessQ[res],
			UncastInformation[res, "SizeT"],
			$Failed
		]
	]
WGLQuery[errMsgHd_, platform_Integer] := WGLQuery[errMsgHd, platform] =
	If[TrueQ[platform > 0 && platform <= ("Number Of Platforms" /. PlatformCount[errMsgHd])],
		Join[
			WGLQuery[errMsgHd, #, platform, 1]& /@ $WGLPlatformQueryList,
			(# -> WGLQuery[errMsgHd, platform, #])& /@ Range["Number Of Devices" /. DeviceCount[errMsgHd, platform]]
		],
		Throw[Message[errMsgHd::invplt, platform]; $Failed]
	]
WGLQuery[errMsgHd_, platform_Integer, prop_] := WGLQuery[errMsgHd, platform, prop] =
	If[TrueQ[platform > 0 && platform <= ("Number Of Platforms" /. PlatformCount[errMsgHd])],
		If[prop === "Devices",
			(# -> WGLQuery[errMsgHd, platform, #])& /@ Range["Number Of Devices" /. DeviceCount[errMsgHd, platform]],
			WGLQuery[errMsgHd, WGLFindPlatformQuery[errMsgHd, prop], platform, 1]
		],
		Throw[Message[errMsgHd::invplt, platform]; $Failed]
	]
WGLQuery[errMsgHd_, platform_Integer, device_Integer] := WGLQuery[errMsgHd, platform, device] =
	If[TrueQ[device > 0 && device <= ("Number Of Devices" /. DeviceCount[errMsgHd, platform])],
		WGLQuery[errMsgHd, #, platform, device]& /@ $WGLDeviceQueryList,
		Throw[Message[errMsgHd::invdev, device]; $Failed]
	]
WGLQuery[errMsgHd_, platform_Integer, device_Integer, prop_] := WGLQuery[errMsgHd, platform, device, prop] =
	If[TrueQ[device > 0 && device <= ("Number Of Devices" /. DeviceCount[errMsgHd, platform])],
		WGLQuery[errMsgHd, WGLFindDeviceQuery[errMsgHd, prop], platform, device],
		Throw[Message[errMsgHd::invdev, device]; $Failed]
	]

WGLFindDeviceQuery[errMsgHd_, prop_String] :=
	With[{res = Select[$WGLDeviceQueryList, If[StringQ[#], # === prop, First[#] === noSpaces[prop]]&, 1]},
		If[res === {},
			Throw[Message[errMsgHd::invqry, prop]; $Failed],
			First[res]
		]
	]
WGLFindPlatformQuery[errMsgHd_, prop_String] :=
	With[{res = Select[$WGLPlatformQueryList, If[StringQ[#], # === noSpaces[prop], First[#] === noSpaces[prop]]&, 1]},
		If[res === {},
			Throw[Message[errMsgHd::invqry, prop]; $Failed],
			First[res]
		]
	]

noSpaces[x_String] := StringReplace[x, " "-> ""]

WGLQuery[errMsgHd_, name_String, platform_Integer, device_Integer] :=
	WGLQuery[errMsgHd, name -> name, platform, device]
WGLQuery[errMsgHd_, Rule[name_String, queryName_String], platform_Integer, device_Integer] :=
	WGLQuery[errMsgHd, name -> {queryName, Identity}, platform, device] 
WGLQuery[errMsgHd_, Rule[name_String, {queryName0_String, fun_}], platform_Integer, device_Integer] :=
	xWGLQuery[errMsgHd, name -> {queryName0, fun}, platform, device] =
		Module[{res, type, queryName},
			InitializeWGL[errMsgHd];
			
			SetErrorMessageHead[errMsgHd];

			queryName = noSpaces[queryName0];

			res = Quiet[cGetGPUInformation[$SingleOrDoubleLibrary][queryName, platform, device]];

	
			If[FailQ[res],
				Throw[Message[errMsgHd::qry, name]; $Failed]
			];
			type = Quiet[cGetGPUInformationType[$SingleOrDoubleLibrary][queryName]];

			If[FailQ[res],
				Throw[Message[errMsgHd::qrytyp, name]; $Failed]
			];

			name -> fun[UncastInformation[res, type]]
		]

WGLQuery[errMsgHd_, name:"Core Count", platform_Integer, device_Integer] := WGLQuery[errMsgHd, name, platform, device] = 
	name -> ResolveCoreCount[$ThisAPI][errMsgHd, platform, device]

(******************************************************************************)

ResolveCoreCount["OpenCL"][errMsgHd_, platform_, device_] :=
	With[{computeCount = ("Maximum Compute Units" /. WGLQuery[errMsgHd, "Maximum Compute Units", platform, device]),
		  type = ("Device Type" /. WGLQuery[errMsgHd, "Device Type", platform, device])},
	  If[type === "CPU",
	  	computeCount,
	  	Switch[ResolveVendorName[$ThisAPI, platform, device],
	  		"NVIDIA",
	  			If[("Global Memory Cache Size" /. WGLQuery[errMsgHd, "Global Memory Cache Size", platform, device]) === 0,
	  				computeCount * 8, (* on non-fermi cards the cache size is 0 *)
	  				computeCount * 32
	  			],
	  		"ATI" | "AMD",
	  			Switch[("Device Name" /. WGLQuery[errMsgHd, "Device Name", platform, device]),
	  				"Cypress" | "ATI RV770",
	  					computeCount * 80,
	  				"ATI RV670" | "ATI R600",
	  					computeCount * 32,
	  				"ATI R580",
	  					computeCount * 24,
	  				_,
	  					computeCount
	  			],
	  		_,
	  			computeCount
	  	]
	  ]
	]
	
ResolveCoreCount["CUDA"][errMsgHd_, platform_, device_] :=
	With[{multiprocCount = ("Multiprocessor Count" /. WGLQuery[errMsgHd, "Multiprocessor Count", platform, device]),
		  computeCapability = ("Compute Capabilities" /. WGLQuery[errMsgHd, "Compute Capabilities", platform, device])},
		If[computeCapability >= 2.0,
			multiprocCount * 32,
			multiprocCount * 8
		]
	]
(******************************************************************************)

ResolveVendorName[api_] := ResolveVendorName[api, $WGLPlatform, $WGLDevice]
ResolveVendorName["CUDA", ___] = "NVIDIA"
ResolveVendorName["OpenCL", platform_, device_] := 
	With[{vendor = ("Device Vendor" /. WGLQuery[ResolveVendorName, "Device Vendor", platform, device])},
		Which[
			StringMatchQ[vendor, ___ ~~ "NVIDIA" ~~ ___, IgnoreCase -> True],
				"NVIDIA",
			StringMatchQ[vendor, ___ ~~ ("AMD" | "ATI") ~~ ___, IgnoreCase -> True],
				"AMD",
			StringMatchQ[vendor, ___ ~~ "Intel" ~~ ___, IgnoreCase -> True],
				"Intel",
			True,
				"Generic"
		]
	]

(******************************************************************************)

SupportsDoublePrecisionQ[errMsgHd_] := SupportsDoublePrecisionQ[errMsgHd] =
	If[MatchQ[$SingleOrDoubleLibrary, "Single" | "Double"],
		$SingleOrDoubleLibrary === "Double",
		With[{res =
			Block[{tag = If[$ThisAPI === "CUDA", CUDALink`CUDALink, OpenCLLink`OpenCLLink], $SingleOrDoubleLibrary = "Single"},
				TrueQ@Apply[
					Or,
					Flatten[
						Table[
							Table[
								SupportsDoublePrecisionQ[tag, plt, dev],
								{dev, "Number Of Devices" /. DeviceCount[SupportsDoublePrecisionQ, plt]}
							],
							{plt, "Number Of Platforms" /. PlatformCount[SupportsDoublePrecisionQ]}
						]
					]
				]
			]},
			Clear[$SingleOrDoubleLibrary];
			$SingleOrDoubleLibrary = If[res, "Double", "Single"];
			res
		]
	]

SupportsDoublePrecisionQ[errMsgHd_, Automatic, Automatic] :=
	SupportsDoublePrecisionQ[errMsgHd, FastestPlatform[], FastestDevice[]]
SupportsDoublePrecisionQ[errMsgHd_, device_Integer] := SupportsDoublePrecisionQ[errMsgHd, 1, device]
SupportsDoublePrecisionQ[errMsgHd_, platform_Integer, Automatic] :=
	SupportsDoublePrecisionQ[errMsgHd, platform, FastestDevice[]]
SupportsDoublePrecisionQ[errMsgHd_, platform_Integer, device_Integer] := SupportsDoublePrecisionQ[errMsgHd, platform, device] = 
	TrueQ[Catch[iSupportsDoublePrecisionQ[$ThisAPI][errMsgHd, platform, device]]]

iSupportsDoublePrecisionQ["CUDA"][errMsgHd_, platform_Integer, device_Integer] :=
	With[{computeCapability = ("Compute Capabilities" /. WGLQuery[errMsgHd, "Compute Capabilities", platform, device])},
		If[computeCapability >= 1.3,
			True,
			False
		]
	]

iSupportsDoublePrecisionQ["OpenCL"][errMsgHd_, platform_Integer, device_Integer] :=
	With[{extensions = ("Device Extensions" /. WGLQuery[errMsgHd, "Device Extensions", platform, device])},
		StringMatchQ[extensions, ___ ~~ ("cl_khr_fp64" | "cl_amd_fp64") ~~ ___]
	]
	
	
FastestPlatform[] := FastestPlatform[] =
	If[$ThisAPI === "CUDA",
		1,
		OpenCLLink`Internal`FastestPlatform[]
	]
FastestDevice[] := FastestDevice[] =
	If[$ThisAPI === "CUDA",
		CUDALink`Internal`FastestDevice[],
		OpenCLLink`Internal`FastestDevice[]
	]
(******************************************************************************)

$64BitMint = TrueQ[Log[2.0, Developer`$MaxMachineInteger] > 32]

$BitsInChar = $SystemWordLength/8;
$SizeOfInteger32 = 4;
$SizeOfInteger64 = 2*$SizeOfInteger32;

$HalfMaxSignedInteger = BitShiftLeft[1, $SizeOfInteger32*$BitsInChar - 2]
$MaxSignedInteger = 2*$HalfMaxSignedInteger - 1

$HalfMaxSignedInteger64 = BitShiftLeft[1, $SizeOfInteger64*$BitsInChar - 1]
$MaxSignedLong = 2*$HalfMaxSignedInteger64 - 1

UncastInformation[res_, "String"] := StringTrim[FromCharacterCode[res]]
UncastInformation[res_, "Integer" | "Integer32"] := First[res]
UncastInformation[res_, "Double"] := First[res]
UncastInformation[res_, "IntegerArray"] := res
UncastInformation[res_, "UnsignedInteger"] := UncastInformation[res, "SizeT"]
UncastInformation[res_, "SizeT"] := FromUnsignedInteger64@@res
UncastInformation[res_, "SizeTArray"] := UncastInformation[#, "SizeT"]& /@ Partition[res, 2]
UncastInformation[res_, "Integer64"] /; $64BitMint := FromInteger64@@res
UncastInformation[res_, "Integer64"] := First[res]
UncastInformation[res_, "Boolean"] := If[res === {1}, True, False]

FromUnsignedInteger[x_Integer] := If[x < 0, Developer`$MaxMachineInteger + x, x]
FromInteger64[low_Integer, high_Integer] := FromUnsignedInteger[high]*Developer`$MaxMachineInteger + FromUnsignedInteger[low]
FromUnsignedInteger64[low_Integer, high_Integer] :=
	 With[{res = FromInteger64[low, high]},
	 	If[res < 0,
	 		$MaxSignedLong + res,
	 		res
	 	]
	 ]

(******************************************************************************)

SpaceSplit[s_String] := StringTrim /@ StringSplit[s, Whitespace]
CommaSplit[s_String] := StringTrim /@ StringSplit[s, ","]

(******************************************************************************)

ToImage[errMsgHd_, img0_?Image`PossibleImageQ] :=
	With[{img = Quiet[Image[img0]]},
		If[!ImageQ[img],
			Throw[Message[errMsgHd::invimg, img0]]
		];
		img
	]
	
(******************************************************************************)

WGLSameTypeQ[t1_, t2_] :=
	Which[
		t1 == t2,
			True,
		Head[t1] == Blank,
			First[t1] == t2 || ToString[First[t1]] == t2,
		Head[t2] == Blank,
			First[t2] == t1 || ToString[First[t2]] == t1,
		Head[t1] == Symbol,
			ToString[t1] == t2,
		Head[t2] == Symbol,
			ToString[t2] == t1,
		True,
			False
	]

(******************************************************************************)

PlatformCount[errMsgHd_] := PlatformCount[errMsgHd] =
	With[{res = Catch[iPlatformCount[$ThisAPI][errMsgHd]]},
		If[res === $Failed,
			"Number Of Platforms" -> 0,
			res
		]
	]
	
iPlatformCount["OpenCL"][errMsgHd_] := WGLQuery[errMsgHd, "Number Of Platforms", 1, 1]
iPlatformCount["CUDA"][errMsgHd_] := "Number Of Platforms" -> 1

(******************************************************************************)

DeviceCount[errMsgHd_, Automatic] :=
	Throw[Message[errMsgHd::unspecplt]; $Failed] 
DeviceCount[errMsgHd_, plt_Integer:1] := DeviceCount[errMsgHd, plt] =
	With[{res = Catch[iDeviceCount[$ThisAPI][errMsgHd, plt]]},
		If[res === $Failed,
			"Number Of Devices" -> 0,
			res
		]
	]
DeviceCount[errMsgHd_, plt_] := Throw[Message[errMsgHd::invplt, plt]; $Failed] 
iDeviceCount["CUDA"][errMsgHd_, platform_] := WGLQuery[errMsgHd, "Number Of Devices", 1, 1]
iDeviceCount["OpenCL"][errMsgHd_, platform_] := WGLQuery[errMsgHd, "Number Of Devices", platform, 1]

(******************************************************************************)

ValidPlatformQ[errMsgHd_, Automatic] := 
	With[{numplts = "Number Of Platforms" /. PlatformCount[errMsgHd]},
		TrueQ[numplts > 0]
	]
ValidPlatformQ[errMsgHd_, plt_Integer] :=
	If[$ThisAPI === "CUDA" && plt != 1,
		False,
		If[TrueQ[plt > 0 && plt <= ("Number Of Platforms" /. PlatformCount[errMsgHd])],
			True,
			False
		]
	]
ValidPlatformQ[errMsgHd_, plt_] := False

(******************************************************************************)

ValidDeviceQ[errMsgHd_, Automatic, Automatic] := 
	With[{numdevs = "Number Of Devices" /. DeviceCount[errMsgHd, FastestPlatform[]]},
		TrueQ[numdevs > 0]
	]
ValidDeviceQ[errMsgHd_, plt_, dev_Integer] :=
	If[TrueQ[dev > 0 && dev <= ("Number Of Devices" /. DeviceCount[errMsgHd, plt])],
		True,
		False
	]
ValidDeviceQ[errMsgHd_, plt_, dev_] := False

(******************************************************************************)

ClearAll[OptionsCheck];
Attributes[OptionsCheck] = HoldFirst;
OptionsCheck[b:(f_[args___, opts:OptionsPattern[]]), ifn_] :=
	Module[{bad, good},
		good = Join[Options[f], Options[ifn]];
		bad = FilterRules[{opts}, Except[good]];
		If[Length[bad] > 0,
			Message[f::"optx", First[bad], HoldForm[b]];
			False,
			True
		]
	]
OptionsCheck[b:(f_[args___, opts:OptionsPattern[]]), ifn_, low_, high_] :=
	OptionsCheck[b, ifn] && ArgumentCountQ[f, Length[{args}], low, high]

nonOptionPattern = PatternSequence[___, Except[_?OptionQ]]

(******************************************************************************)


$ThisLink = $ThisAPI <> "Link"

GPUTools`Utilities`DefineMessage[
	$Symbols,
	{
		{"init", GPUTools`Message`init[$ThisAPI]},
		{"noblib", GPUTools`Message`nolib[$ThisAPI]},
		{"nodev", GPUTools`Message`nodev[$ThisAPI]},
		{"invsys", GPUTools`Message`invsys[$ThisAPI]},
		{"legcy", GPUTools`Message`legcy[$ThisAPI]},
		{"invdevnm", GPUTools`Message`invdevnm[$ThisAPI]},	
		{"invdriv", GPUTools`Message`invdrv[$ThisAPI]},
		{"nodriv", GPUTools`Message`invdrv[$ThisAPI]},		
		{"invdrv", GPUTools`Message`invdrv[$ThisAPI]}, 	
		{"invdrvp", GPUTools`Message`invdrvp[$ThisAPI]},
		{"invdrivver", GPUTools`Message`invdrivver[$ThisAPI]},
		{"invdrivverv", GPUTools`Message`invdrivverv[$ThisAPI]},
		{"invdrivverd", GPUTools`Message`invdrivverd[$ThisAPI]},
		{"invclib", GPUTools`Message`invclib[$ThisAPI]},
		{"invclibp", GPUTools`Message`invclibp[$ThisAPI]},
		{"invcudaver", GPUTools`Message`invcudaver[$ThisAPI]},
		{"invcudaverv", GPUTools`Message`invcudaverv[$ThisAPI]},
		{"invcudaverd", GPUTools`Message`invcudaverd[$ThisAPI]},
		{"invtkver", GPUTools`Message`invtkver[$ThisAPI]},
		{"invtkverv", GPUTools`Message`invtkverv[$ThisAPI]},
		{"syslibfld", GPUTools`Message`syslibfld[$ThisAPI]},
		{"libldfld", GPUTools`Message`libldfld[$ThisAPI]},
		{"gpures",  GPUTools`Message`gpures[$ThisAPI]},
		{"initlib", "Failed to initialize library `1`."},
		{"unpkd", "Input `1` is not a valid input array."},
		{"invtrgtprs", "\"TargetPrecision\" specified is not valid. \"TargetPrecision\" can be set to either Automatic, \"Single\", or \"Double\"."},
		{"initwgl", "Failed to initialize " <> $ThisLink <> " from `1`."},
		{"nopklt", "Failed to find " <> $ThisAPI <> " paclet."},
		{"nocudautils", "Failed to find " <> $ThisAPI <> " Utilities."},
		{"nobindir", $ThisAPI <> " Toolkit bin directory does not exist."},
		{"memtype", "Input memory type `1` conflicts with the type `2` defined while loading the function."},
		{"typwid", "Input dimension `1` is not a multiple of the vector type width `2`."},
		{"args", "Invalid arguments `1`."},
		{"libload", "Failed to load library function `2` from `1`."},
		{"notype", "Failed to load `1` because of the type `2` is not supported."},
		{"getid", "Failed to get the memory id for `1`. Make sure that the input is valid memory registered with the memory manger."},
		{"getsym", "Failed to get the memory symbol name for `1`. Make sure that the input is valid memory registered with the memory manger."},
		{"gettype", "Failed to get the memory type for `1`. Make sure that the input is valid memory registered with the memory manger."},
		{"getdims", "Failed to get the memory dimensions for `1`. Make sure that the input is valid memory registered with the memory manger."},
		{"getrank", "Failed to get the memory rank for `1`. Make sure that the input is valid memory registered with the memory manger."},
		{"getbytc", "Failed to get the memory byte count for `1`. Make sure that the input is valid memory registered with the memory manger."},
		{"getres", "Failed to get the memory residence for `1`. Make sure that the input is valid memory registered with the memory manger."},
		{"getshr", "Failed to get the memory sharing for `1`. Make sure that the input is valid memory registered with the memory manger."},
		{"getunq", "Failed to get the memory uniqueness for `1`. Make sure that the input is valid memory registered with the memory manger."},
		{"getplt", "Failed to get the memory platform for `1`. Make sure that the input is valid memory registered with the memory manger."},
		{"getdev", "Failed to get the memory device for `1`. Make sure that the input is valid memory registered with the memory manger."},
		{"getmtyp", "Failed to get the memory mathematica type for `1`. Make sure that the input is valid memory registered with the memory manger."},
		{"gettypinfo", "Failed to get the memory type information for `1`. Make sure that the input is valid memory registered with the memory manger."},
		{"invtypspec", "`1` is not a valid argument type."},
		{"type", "`1` is not a valid type."},
		{"invcallarg", "Argument `2` is not a valid value for argument type `1`."},
		{"invarglen", "Input argument count `2` does not match function argument count `1` specified."},
		{"invgrd", $ThisLink <> " was called with invalid grid dimensions `1`. Grid dimensions must be positive and either an Integer or a list of Integers."},
		{"invgrdblk", $ThisLink <> " was called with invalid grid dimensions. The kernel cannot be launched with `1` set as the grid dimensions, while `2` set as the block dimensions."},
		{"gethdfld", $ThisLink <> " was not able to get memory handle with head `2`, required head is `1`."},
		{"inputlst", "Input `1` is not valid a List or Image."},
		{"getid", "Input memory `1` was not found in memory manager."},
		{"qry", $ThisLink <> " was not able to  query property `1`."},
		{"unlmem", "Unable to unload memory `1`. Make sure it is a valid " <> $ThisLink <> " memory."},
		{"liblnch", "During the evaluation, an error `1` was raised when launching the library function."},
		{"argtype", "Unable to set function argument `1` because type is not supported."}
	}
]

GPUTools`Utilities`DefineMessage[
	$Symbols,
	{
		(* C messages *)
		
		{"invstt", $ThisLink <> " encountered an invalid state."},
		{"invval", $ThisLink <> " encountered an invalid value."},
		{"invinpt", $ThisLink <> " encountered an invalid input."},
		{"outmem", $ThisLink <> " ran out of available memory, possibly due to not freeing memory using the memory manager."},
		{"notinit", $ThisLink <> " is not initialized."},
		{"notdeinit", $ThisLink <> " failed to deinitialize."},
		{"nodev", $ThisLink <> " failed to detect a usable device."},
		{"nodeva", $ThisLink <> " failed to detect an available device."},
		{"invwld", $ThisLink <> " encountered invalid Wolfram Library data."},
		{"invplt", $ThisLink <> " platform is invalid."},
		{"invdev", $ThisLink <> " device is invalid."},
		{"invdevt", $ThisLink <> " device is of an invalid type."},
		{"invctx", $ThisLink <> " context is invalid."},
		{"invqup", $ThisLink <> " encountered invalid queue properties."},
		{"invcmdcq", $ThisLink <> " command queue is invalid."},
		{"invbin", $ThisLink <> " encountered an invalid binary."},
		{"invsrc", $ThisLink <> " encountered invalid source input. The source input must be either a string containing the program, or a list of one element indicating the file containing the program."},
		{"invsrcf", $ThisLink <> " encountered an invalid source file."},
		{"invout", $ThisLink <> " encountered an invalid output file."},
		{"invmem", $ThisLink <> " encountered invalid memory."},
		{"invresp", $ThisLink <> " encountered an invalid residence property."},
		{"invshrp", $ThisLink <> " encountered an invalid sharing property."},
		{"nocmp", $ThisLink <> " failed to detect a usable compiler."},
		{"noxcmp", $ThisLink <> " failed to detect a usable compiler."},
		{"cmpf", $ThisLink <> " encountered a compilation failure."},
		{"invbldopt", $ThisLink <> " encountered invalid build options."},
		{"invprop", $ThisLink <> " encountered an invalid property."},
		{"invprog", $ThisLink <> " encountered an invalid program."},
		{"invker", $ThisLink <> " encountered an invalid CUDA kernel."},
		{"invkernam", $ThisLink <> " encountered an invalid or missing kernel name."},
		{"invtyp", $ThisLink <> " encountered an invalid type."},
		{"notfnd", $ThisLink <> " resource not found."},
		{"lnchfld", $ThisLink <> " experienced a kernel launch failure."},
		{"lchoutres", $ThisLink <> " experienced an out of resources error."},
		{"lnchtout", "A " <> $ThisLink <> " kernel timed out."},
		{"invdim", $ThisLink <> " encountered an object with invalid dimensions."},
		{"invimg", $ThisLink <> " encountered an invalid image."},
		{"invblksz", $ThisLink <> " block size is invalid."},
		{"invgrdsz", $ThisLink <> " grid size is invalid."},
		{"invblkdim", $ThisLink <> " block dimension is invalid."},
		{"invgrddim", $ThisLink <> " grid dimension is invalid."},
		{"allocf", "A " <> $ThisLink <> " memory allocation failed."},
		{"memcpy", "A " <> $ThisLink <> " memory copy failed."},
		{"internal", $ThisLink <> " experienced an internal error."},
		{"nodblp", $ThisLink <> " does not support double precision on this system."},
		{"unknown", $ThisLink <> " experienced an unknown error."},
		{"msgunk", $ThisLink <> " experienced an unknown error."},
		{"noimpl", "Current operation is not implemented."}
	}
]

(******************************************************************************)

