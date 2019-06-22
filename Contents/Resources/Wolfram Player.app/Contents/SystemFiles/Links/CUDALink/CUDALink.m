(******************************************************************************)
(* :Name: CUDALink.m *)
(* :Title: CUDALink *)
(* :Context: CUDALink` *)
(* :Author: Abdul Dakkak *)
(* :Summary: *)
(* :Sources:*)
(* :Copyright: 2010, Wolfram Research, Inc. *)
(* :Mathematica Version: 8.0 *)
(* :Keywords: CUDA, GPU Computing, GPGPU *)
(* :Warnings: None *)
(* :Limitations: None *)
(* :Discussion: *)
(* :Requirements: None *)
(* :Examples: None *)
(******************************************************************************)

BeginPackage["CUDALink`", {"CCompilerDriver`", "SymbolicC`", "CUDALink`NVCCCompiler`"}]


(******************************************************************************)

CUDALink`Internal`$Symbols = {
	CUDALink,
	CUDATranspose,
	CUDAQ,
	CUDAProgram,
	CUDAInformation,
	CUDAFunction,
	CUDAMemory,
	CUDAMemoryLoad,
	CUDAMemoryAllocate,
	CUDAMemoryUnload,
	CUDAMemoryGet,
	CUDAMemoryInformation,
	CUDAMemoryCopyToHost,
	CUDAMemoryCopyToDevice,
	CUDAFunctionLoad,
	CUDAFunctionInformation,
	CUDAProfile,
	CUDAImageConvolve,
	CUDABoxFilter,
	CUDAImageAdd,
	CUDAImageSubtract,
	CUDAImageMultiply,
	CUDAImageDivide,
	CUDAErosion,
	CUDADilation,
	CUDAOpening,
	CUDAClosing,
	CUDAFourier,
	CUDAInverseFourier,
	CUDADot,
	CUDATotal,
	CUDAArgMinList,
	CUDAArgMaxList,
	CUDAFluidDynamics,
	CUDAFinancialDerivative,
	CUDAVolumetricDataRead,
	CUDAVolumetricRender,
	SymbolicCUDAFunction,
	SymbolicCUDAThreadIndex,
	SymbolicCUDABlockIndex,
	SymbolicCUDABlockDimension,
	SymbolicCUDACalculateKernelIndex,
	SymbolicCUDADeclareIndexBlock,
	SymbolicCUDAKernelIndex,
	CUDASort,
	CUDAFoldList,
	CUDAFold,
	CUDAMap,
	CUDAClamp,
	CUDAColorNegate,
	CUDAResourcesInstall,
	CUDAResourcesUninstall,
	CUDAResourcesInformation,
	CUDALink`Internal`CUDAConstantMemoryCopy
}

(******************************************************************************)

Unprotect /@ CUDALink`Internal`$Symbols

(******************************************************************************)

ClearAll  /@ CUDALink`Internal`$Symbols

(******************************************************************************)

$CUDADevice::usage = "$CUDADevice is device used for CUDA computation."

$CUDALinkPath::usage = "$CUDALinkPath gives path to the CUDALink application."

$CUDALinkLibraryPath::usage = "$CUDALinkLibraryPath gives path to the CUDALink library resources folder."

$CUDALinkExampleDataPath::usage = "$CUDALinkExampleData path to the CUDALink example data folder."

$CUDAResourcesVersion::usage = "$CUDAResourcesVersion version number of CUDAResources paclet."

$CUDADeviceCount::usage = "$CUDADeviceCount is number of devices on system." 

CUDADriverVersion::usage = "CUDADriverVersion[] get the system video driver version."

CUDAResourcesInstall::usage = "CUDAResourcesInstall[] install the CUDAResources paclet from Wolfram data server."

CUDAResourcesUninstall::usage = "CUDAResourcesUninstall[] uninstall all CUDAResources paclets on system."

CUDAResourcesInformation::usage = "CUDAResourcesInformation gives information on installed CUDAResources paclets."

CUDALink::usage = "CUDALink allows users to program the GPU from within Mathematica."

CUDAQ::usage = "CUDAQ[] checks if CUDALink is supported on system."

CUDAInformation::usage = "CUDAInformation[] gives information on CUDA devices detected."

CUDAProgram::usage = "CUDAProgram is handle to CUDA program."

CUDAFunction::usage = "CUDAFunction is handle to a function loaded using CUDAFunctionLoad."

CUDAMemory::usage = "CUDAMemory gives handle to CUDA memory loaded using CUDAMemoryLoad or CUDAMemoryAllocate."

CUDAMemoryLoad::usage = "CUDAMemoryLoad[lst] loads a Mathematica expression into CUDA memory."

CUDAMemoryAllocate::usage = "CUDAMemoryAllocate[type, dims] allocates memory and registers it with CUDA memory manager."

CUDAMemoryUnload::usage = "CUDAMemoryUnload[mem] unloads memory from the CUDALink memory manager."

CUDAMemoryGet::usage = "CUDAMemoryGet[mem] gets memory from the CUDALink memory manager into Mathematica."

CUDAMemoryInformation::usage = "CUDAMemoryInformation[mem] gives information on the CUDAMemory."

CUDAMemoryCopyToHost::usage = "CUDAMemoryCopyToHost[mem] force copy memory from GPU to CPU."

CUDAMemoryCopyToDevice::usage = "CUDAMemoryCopyToDevice[mem] force copy memory from CPU to GPU."

CUDAFunctionLoad::usage = "CUDAFunctionLoad[code, funName, {args, ...}, blockDim] loads CUDA function from code."

CUDAFunctionInformation::usage = "CUDAFunctionInformation[cudafun] gives information on a loaded CUDA function."

CUDAImageConvolve::usage = "CUDAImageConvolve[img, kern] gives img convolved with kern."

CUDABoxFilter::usage = "CUDABoxFilter[img, rad] gives img convolved with BoxMatrix[rad]."

CUDAImageAdd::usage = "CUDAImageAdd[img, val] gives img added to val."

CUDAImageSubtract::usage = "CUDAImageSubtract[img, val] gives img subtracted from val."

CUDAImageMultiply::usage = "CUDAImageMultiply[img, val] gives img multiplied by val."

CUDAImageDivide::usage = "CUDAImageDivide[img, val] gives img divided by val."

CUDAErosion::usage = "CUDAErosion[img, rad] gives morphological erosion of img with radius rad."

CUDADilation::usage = "CUDADilation[img, rad] gives morphological dilation of img with radius rad."

CUDAOpening::usage = "CUDAOpening[img, rad] gives morphological opening of img with radius rad."

CUDAClosing::usage = "CUDAClosing[img, rad] gives morphological closing of img with radius rad."

CUDAFourier::usage = "CUDAFourier[lst] performs Fourier transform on lst."

CUDAInverseFourier::usage = "CUDAInverseFourier[lst] performs inverse Fourier transform on lst."

CUDATranspose::usage = "CUDATranspose[mat] gives transpose of input matrix mat."

CUDADot::usage = "CUDADot[a, b] gives products of vectors and matricies."

CUDATotal::usage = "CUDATotal[vec] gives the total of the absolute value of a vector vec."

CUDAArgMinList::usage = "CUDAArgMinList[lst] gives the index of the minimum element in lst."

CUDAArgMaxList::usage = "CUDAArgMaxList[lst] gives the index of the maximum element in lst"

CUDAFluidDynamics::usage = "CUDAFluidDynamics[] demonstrates computational fluid dynamics using CUDALink."

CUDAFinancialDerivative::usage = "CUDAFinancialDerivative[instrument, params, ambientparams] gives the value of the specified financial instrument."

CUDAVolumetricDataRead::usage = "CUDAVolumetricDataRead[file, height, depth] reads volumetric data stored in file with specified height and depth."

CUDAVolumetricRender::usage = "CUDAVolumetricRender[vol] performs volumetric rendering using the input data."

SymbolicCUDAFunction::usage = "SymbolicCUDAFunction[name, args, body]  is a symbolic representation of a CUDA function."

SymbolicCUDAThreadIndex::usage = "SymbolicCUDAThreadIndex[dim] is a symbolic representation of a CUDA kernel thread index call."

SymbolicCUDABlockIndex::usage = "SymbolicCUDABlockIndex[dim] is a symbolic representation of a CUDA kernel block index call."

SymbolicCUDABlockDimension::usage = "SymbolicCUDABlockDimension[dim] is a symbolic representation of a CUDA kernel block dimension call."

SymbolicCUDACalculateKernelIndex::usage = "SymbolicCUDACalculateKernelIndex[dim] is a symbolic representation of a CUDA kernel index calculation."

SymbolicCUDADeclareIndexBlock::usage = "SymbolicCUDADeclareIndexBlock[dim] is a symbolic representation of a CUDA kernel index declaration."

SymbolicCUDAKernelIndex::usage = "SymbolicCUDAKernelIndex[dim] is a symbolic representation of a CUDA kernel index call."

CUDASort::usage = "CUDASort[lst] sorts the values of input."

CUDAFoldList::usage = "CUDAFoldList[f, x, lst] folds the values of lst with function f and initial value x returning intermediate steps."

CUDAFold::usage = "CUDAFold[f, x, lst] folds the values of lst with function f and initial value x."

CUDAMap::usage = "CUDAMap[f, lst] maps function f on lst."

CUDAClamp::usage = "CUDAClamp[img, low, high] clamps the values of img between high and low"

CUDAColorNegate::usage = "CUDAColorNegate[img] negates the colors of input img."

CUDAProfile::usage = "CUDAProfile[args] runs the low level CUDA profiler on the arguments."

CUDALink`Internal`CUDAConstantMemoryCopy::usage = "CUDALink`Internal`CUDAConstantMemoryCopy[mem1, mem2] copies constant memory to and from host memory"

(******************************************************************************)

Begin["`Private`"]

(******************************************************************************)

$Symbols = CUDALink`Internal`$Symbols

(******************************************************************************)

Needs["GPUTools`"]
Needs["LibraryLink`"]
Needs["GPUTools`Detection`"]
Needs["GPUTools`SymbolicGPU`"]
Needs["ResourceLocator`"]

(******************************************************************************)

$CUDASupportFiles = FileNameJoin[{$ThisDirectory, "SupportFiles", "CUDA"}]

CUDALink`Internal`$MinPacletVersion = "10.0.0.0"
CUDALink`Internal`$Version = StringJoin[Riffle[ToString /@ {If[IntegerPart[$VersionNumber] == $VersionNumber, ToString[$VersionNumber] <> "0", $VersionNumber], $ReleaseNumber, $MinorReleaseNumber}, "."]]
CUDALink`Internal`$InputFileName = System`Private`$InputFileName
CUDALink`Internal`$CUDALinkPath := DirectoryName[CUDALink`Internal`$InputFileName]
CUDALink`Internal`$CUDALinkSystemResourcesPath := Catch[$SystemResourcesPath]
$CUDALinkPath := CUDALink`Internal`$CUDALinkPath
$CUDALinkLibraryPath := (CUDALink`Internal`$CUDALinkSystemResourcesPath; CUDALink`Internal`$CUDALinkLibraryPath)
$CUDALinkExampleDataPath := (CUDALink`Internal`$CUDALinkSystemResourcesPath; CUDALink`Internal`$CUDALinkExampleDataPath)
$CUDAResourcesVersion := (CUDALink`Internal`$CUDALinkSystemResourcesPath; $APIVersion)

(******************************************************************************)

CUDALink`Internal`$ToolkitVersion /; StringQ[$CUDALinkLibraryPath] := 
	If[NumericQ[getToolkitVersionFromPaclet[]],
		getToolkitVersionFromPaclet[],
		4.0
	]
	
CUDALink`Internal`$MinDriverVersion /; StringQ[$CUDALinkLibraryPath] := 
	If[NumericQ[getMinDriverVersionFromPaclet[]],
		getMinDriverVersionFromPaclet[],
		270.
	]

getToolkitVersionFromPaclet[] := getToolkitVersionFromPaclet[] =
	Quiet[getInformationFromPaclet[Global`ToolkitVersion]]

getMinDriverVersionFromPaclet[] := getMinDriverVersionFromPaclet[] =
	Quiet[getInformationFromPaclet[Global`MinimumDriver]]

getInformationFromPaclet[info_] :=
	Module[{paclet = PacletFind["CUDAResources"], description, exp},
		If[paclet === {},
			$Failed,
			description = "Description" /. PacletInformation[First[paclet]];
			If[!StringQ[description] || description === "",
				$Failed,
				exp = ToExpression[description];
				info /. exp
			]
		]
	]

(******************************************************************************)

CUDALink`Internal`SupportsDoublePrecisionQ[dev_] := 
	SupportsDoublePrecisionQ[CUDALink, 1, dev]

(******************************************************************************)
(******************************************************************************)
(******************************************************************************)

$ThisAPI = "CUDA"

$ThisHead = CUDALink

If[$CUDADevice === Unevaluated[$CUDADevice],
	$CUDADevice = Automatic,
	$WGLDevice = $CUDADevice
]

$WGLInvalidTypes = {
	"Void",
	"Byte[8]",
	"Byte[16]",
	"UnsignedByte[8]",
	"UnsignedByte[16]",
	"Short[8]",
	"Short[16]",
	"UnsignedShort[8]",
	"UnsignedShort[16]",
	"Integer[8]",
	"Integer[16]",
	"UnsignedInteger[8]",
	"UnsignedInteger[16]",
	"Long[8]",
	"Long[16]",
	"UnsignedLong[8]",
	"UnsignedLong[16]",
	"Float[8]",
	"Float[16]",
	"Double[8]",
	"Double[16]",
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


$ThisFile = System`Private`$InputFileName
$ThisDirectory = DirectoryName[$ThisFile]

(******************************************************************************)


$ExtraLibraryFunctions[libraryPath_, singleOrDouble_:"Single"] = {
	{cTranspose[singleOrDouble], libraryPath, "oTranspose", {_Integer, _Integer}, _Integer},
	
	{cAMax[singleOrDouble], libraryPath, "oAMax", {_Integer}, _Integer},
	{cAMin[singleOrDouble], libraryPath, "oAMin", {_Integer}, _Integer},
	{cASum[singleOrDouble], libraryPath, "oASum", {"UTF8String", _Integer}, _Complex},
	{cDot[singleOrDouble],  libraryPath, "oDot",  {_Integer, _Integer, "Boolean"}, _Complex},
	{cGemv[singleOrDouble], libraryPath, "oGemv", {_Complex, "UTF8String", _Integer, _Integer, _Complex, _Integer}, "Void"},
	{cGemm[singleOrDouble], libraryPath, "oGemm", {"UTF8String", "UTF8String", "UTF8String", _Complex, _Integer, _Integer, "UTF8String", _Complex, _Integer}, "Void"},
	
	{cCUFFTExec[singleOrDouble], libraryPath, "oCUFFT_Exec", {_Integer, _Integer, "UTF8String"}, "Void"},
	
	{cImageProcessingConvolution[singleOrDouble], libraryPath, "oCUDA_ImageProcessing_Convolution", {_Integer, _Integer, _Integer, "UTF8String", _Real, _Integer}, _Integer},
	{cImageProcessingBoxFilter[singleOrDouble], libraryPath, "oCUDA_ImageProcessing_BoxFilter", {_Integer, _Integer, _Integer, _Integer, "UTF8String", _Real, _Integer}, _Integer},
	{cImageProcessingDilation[singleOrDouble], libraryPath, "oCUDA_ImageProcessing_Dilation", {_Integer, _Integer, _Integer, "UTF8String", _Real, _Integer}, _Integer},
	{cImageProcessingErosion[singleOrDouble], libraryPath, "oCUDA_ImageProcessing_Erosion", {_Integer, _Integer, _Integer, "UTF8String", _Real, _Integer}, _Integer},
	{cImageProcessingOpening[singleOrDouble], libraryPath, "oCUDA_ImageProcessing_Opening", {_Integer, _Integer, _Integer, "UTF8String", _Real, _Integer}, _Integer},
	{cImageProcessingClosing[singleOrDouble], libraryPath, "oCUDA_ImageProcessing_Closing", {_Integer, _Integer, _Integer, "UTF8String", _Real, _Integer}, _Integer},
	{cImageProcessingAdd[singleOrDouble], libraryPath, "oCUDA_ImageProcessing_Add", {_Integer, _Integer, _Integer, _Integer}, _Integer},
	{cImageProcessingAddConstant[singleOrDouble], libraryPath, "oCUDA_ImageProcessing_AddConstant", {_Integer, _Integer, "UTF8String", _Real, _Integer}, _Integer},
	{cImageProcessingSubtract[singleOrDouble], libraryPath, "oCUDA_ImageProcessing_Subtract", {_Integer, _Integer, _Integer, _Integer}, _Integer},
	{cImageProcessingSubtractConstant[singleOrDouble], libraryPath, "oCUDA_ImageProcessing_SubtractConstant", {_Integer, _Integer, "UTF8String", _Real, _Integer}, _Integer},
	{cImageProcessingMultiply[singleOrDouble], libraryPath, "oCUDA_ImageProcessing_Multiply", {_Integer, _Integer, _Integer, _Integer}, _Integer},
	{cImageProcessingMultiplyConstant[singleOrDouble], libraryPath, "oCUDA_ImageProcessing_MultiplyConstant", {_Integer, _Integer, "UTF8String", _Real, _Integer}, _Integer},
	{cImageProcessingDivide[singleOrDouble], libraryPath, "oCUDA_ImageProcessing_Divide", {_Integer, _Integer, _Integer, _Integer}, _Integer},
	{cImageProcessingDivideConstant[singleOrDouble], libraryPath, "oCUDA_ImageProcessing_DivideConstant", {_Integer, _Integer, "UTF8String", _Real, _Integer}, _Integer},
	{cImageProcessingClamp[singleOrDouble], libraryPath, "oCUDA_ImageProcessing_Clamp", {_Integer, _Integer, "UTF8String", _Real, _Real, _Integer}, _Integer},
	
	{cCUDARand[singleOrDouble], libraryPath, "oCURAND_generate", {_Integer, "UTF8String", "UTF8String", "UTF8String", _Integer, "UTF8String", _Real, _Real}, "Void"}
}

(******************************************************************************)

$CUDAResourcesPacletName = "CUDAResources"
$CUDALinkPacletName = "CUDALink"
$GPUToolsPacletName = "GPUTools"
$OpenCLLinkPacletName = "OpenCLLink"
$PacletDependencies = {
	$CUDALinkPacletName,
	$GPUToolsPacletName,
	$OpenCLLinkPacletName,
	$CUDAResourcesPacletName
}

PacletInstalledQ[] :=
	PacletFind[$CUDAResourcesPacletName] =!= {}

$SystemResourcesPath := $SystemResourcesPath = 
	Module[{paclets = PacletFind[$CUDAResourcesPacletName], paclet, sysdir},
		If[paclets === {},
			paclet = Check[CUDAResourcesInstall["Web"], Return[$Failed]];
			If[paclet === {} || paclet === $Failed,
				$InitializeWGLError = {"nopaclet", $CUDAResourcesPacletName};
				Return[InitializationErrorMessage[CUDALink, $InitializeWGLError]; $Failed]
			];
			paclets = PacletFind[$CUDAResourcesPacletName]
		];
		If[paclets === {} || paclets === $Failed,
			$InitializeWGLError = {"nocudares"};
			Return[InitializationErrorMessage[CUDALink, $InitializeWGLError]; $Failed],
			paclet = First[paclets];
			sysdir = getCUDAToolkitLibraryPath[PacletResource[paclet, "CUDAToolkit"], $SystemID];
			CUDALink`Internal`$CUDALinkExampleDataPath = PacletResource[paclet, "ExampleData"];
			If[Quiet[DirectoryQ[sysdir]],
				$APIVersion = "Version" /. PacletInformation[paclet];
				$APIVersion = If[PacletNewerQ[CUDALink`Internal`$Version, $APIVersion],
					CUDALink`Internal`$Version,
					$APIVersion
				];
				If[!(PacletNewerQ[$APIVersion, CUDALink`Internal`$MinPacletVersion] || $APIVersion === CUDALink`Internal`$MinPacletVersion),
					$InitializeWGLError = {"pacletold", $APIVersion, CUDALink`Internal`$Version};
					InitializationErrorMessage[CUDALink, $InitializeWGLError]
				];
				CUDALink`Internal`$CUDALinkLibraryPath = If[Quiet[StringQ[PacletResource[paclet, "LibraryResources"]]],
					FileNameJoin[{PacletResource[paclet, "LibraryResources"], $SystemID}],
					InitializationErrorMessage[CUDALink, $InitializeWGLError]
				];
				If[!StringQ[CUDALink`Internal`$CUDALinkLibraryPath],
					CUDALink`Internal`$CUDALinkLibraryPath = $Failed
				];
				If[StringQ[CUDALink`Internal`$CUDALinkLibraryPath] && FreeQ[$LibraryPath, CUDALink`Internal`$CUDALinkLibraryPath],
					PrependTo[$LibraryPath, CUDALink`Internal`$CUDALinkLibraryPath]
				];

				Needs["GPUResources`", FileNameJoin[{"Location" /. PacletInformation[paclet], "PostInstall.m"}]];
				sysdir,
				
				$InitializeWGLError = {"nobindir", sysdir};
				InitializationErrorMessage[CUDALink, $InitializeWGLError];
				$Failed
			]
		]
	]

Options[CUDAResourcesInstall] = {Update -> False, "Force" -> True}
CUDAResourcesInstall[opts:OptionsPattern[]] := CUDAResourcesInstall["Web", opts]
CUDAResourcesInstall[pth_, opts:OptionsPattern[]] :=
	Module[{updateQ, forceQ, paclet},
		If[FileExistsQ[pth],
			
			updateQ = GetAndCheckOption[CUDAResourcesInstall, Update, {opts}];
			If[updateQ === $Failed,
				Return[$Failed]
			];
			
			forceQ = GetAndCheckOption[CUDAResourcesInstall, "Force", {opts}];
			
			If[updateQ,
				CUDAResourcesUninstall[]
			];
			
			paclet = PacletFind[$CUDAResourcesPacletName];
			
			If[paclet =!= {} && !forceQ,
				Return[paclet]
			];
			
			PacletManager`PacletInstall[pth];
			paclet = PacletFind[$CUDAResourcesPacletName];
			If[paclet === {},
				$InitializeWGLError = {"pacletpth", pth};
				InitializationErrorMessage[CUDAResourcesInstall, $InitializeWGLError];
				$Failed,
				RunPostInstall[First@paclet];
				PacletFind["CUDAResources"]
			]
		]
	]

CUDAResourcesInstall[Automatic | "Web", opts:OptionsPattern[]] :=
	Module[{paclet, updateQ},
		
		updateQ = GetAndCheckOption[CUDAResourcesInstall, Update, {opts}];
		If[updateQ === $Failed,
			Return[$Failed]
		];
		
		If[updateQ,
			CUDAResourcesUninstall[]
		];
		
        paclet = PacletFind[$CUDAResourcesPacletName];
        
        If[paclet =!= {},
            Return[paclet]
        ];
        
        Check[
            Do[
		PacletTools`PacletGet[pac, CUDALink],
		{pac, $PacletDependencies}
	    ],
            Return[$Failed]
        ];
        paclet = PacletFind[$CUDAResourcesPacletName];
        If[paclet === {},
            $Failed,
            RunPostInstall[First@paclet];
            PacletFind["CUDAResources"]
        ]
	]

RunPostInstall[paclet_] :=
	With[{dir = "Location" /. PacletInformation[paclet]},
		Needs["GPUResources`", FileNameJoin[{dir, "PostInstall.m"}]];
		CUDALink`Internal`$ToolkitVersion = getToolkitVersionFromPaclet[];
		GPUResources`PostInstall[CUDALink, paclet]
	]
	
CUDAResourcesUninstall[] :=
	(
		Quiet[PacletUninstall /@ Flatten[PacletFind /@ $PacletDependencies],{PacletUninstall::nodelete}];
		PacletManager`RebuildPacletData[]
	)

CUDAResourcesInformation[] :=
	CUDAResourcesInformation[All]
	
CUDAResourcesInformation["Short"] :=
	Module[{paclet},
		paclet = paclet = PacletFind[$CUDAResourcesPacletName];
		If[paclet === {},
			$InitializeWGLError = {"nopaclet"};
			Return[InitializationErrorMessage[CUDAResourcesInformation, $InitializeWGLError]; {}],
			PacletInformation[paclet] 
		]
	]	

CUDAResourcesInformation[All] :=
	Module[{paclet},
		paclet = paclet = PacletFind[$CUDAResourcesPacletName];
		If[paclet === {},
			$InitializeWGLError = {"nopaclet"};
			Return[InitializationErrorMessage[CUDAResourcesInformation, $InitializeWGLError]; {}],
			
			Map[
				Flatten[{
					PacletInformation[#],
					"Hash" -> getCUDAResourcesPacletHash[#]
				}]&,
				paclet
			] 
		]
	]
	
getCUDAResourcesPacletHash[paclet_] := getCUDAResourcesPacletHash[paclet] =
	IntegerString[
		Hash[
			FileHash[#, "MD5"] & /@ 
				Cases[FileInformation /@ FileNames["*", "Location" /. PacletInformation[paclet], Infinity], rules : {___, FileType -> File, ___} :> (File /. rules)], 
     		"MD5"
     	],
     	16,
     	32
	]
	

(******************************************************************************)

pacletVersionTooOldQ[version_] :=
	Apply[ToExpression[#1] < ToExpression[#2] &, {Last@StringSplit[version, "."], Last@StringSplit[CUDALink`Internal`$Version, "."]}]


(******************************************************************************)

GetAndCheckOption[head:CUDAResourcesInstall, errMsgHd_, n:Update, opts_] :=
	With[{opt = OptionValue[head, opts, n]},
		If[MemberQ[{True, False}, opt],
			opt,
			$InitializeWGLError = {"update", opt};
			Message[errMsgHd::update, opt]; $Failed
		]
	]

GetAndCheckOption[head:CUDAResourcesInstall, errMsgHd_, n:"Force", opts_] :=
	With[{opt = OptionValue[head, opts, n]},
		If[MemberQ[{True, False}, opt],
			opt,
			$InitializeWGLError = {"force", opt};
			Message[errMsgHd::update, opt]; $Failed
		]
	]

(******************************************************************************)

quoteFileName[file_String] :=
	If[TrueQ@needsQuotesQ[file],
		StringJoin["\"", file, "\""],
		file
	]

needsQuotesQ[str_String] := !StringMatchQ[str, "\"" ~~ ___ ~~ "\""]

(******************************************************************************)

CUDALink`Internal`$ExamplesLibraryPath := CUDALink`Internal`$ExamplesLibraryPath = If[CUDAQ[],
	LibraryLoad[$WGLLibraryPath["CUDA", $SingleOrDoubleLibrary]];
	FindLibrary["Examples_" <> $SingleOrDoubleLibrary <> "." <> $LibraryExtensions],
	$Failed
]	

getCUDAToolkitLibraryPath[pth_, "Windows" | "Windows-x86-64"] := 
	With[{pths = FileNames["*nvcc*", pth, Infinity, IgnoreCase->True]},
		If[pths === $Failed || !ListQ[pths] || pths === {},
			$Failed,
			DirectoryName[First[pths]]
		]
	]
getCUDAToolkitLibraryPath[pth_, "Linux" | "MacOSX-x86" | "MacOSX-x86-64"] := FileNameJoin[{pth, "lib"}]
getCUDAToolkitLibraryPath[pth_, "Linux-x86-64"] :=
	With[{sel = Select[FileNameJoin[{pth, #}]& /@ {"lib", "lib64"}, DirectoryQ, 1]},
		If[sel === {},
			$Failed,
			First[sel]
		]
	]



(******************************************************************************)

Get["GPUTools`WGLPrivate`"]

(******************************************************************************)

(* ::Section:: *)
(* CUDA Information *)

SyntaxInformation[CUDAQ] = {"ArgumentsPattern" -> {}}
CUDAQ[___] ? (Function[{arg}, OptionsCheck[arg, CUDAQ, 0, 0], HoldFirst]) :=
	If[Developer`$ProtectedMode,
		False,
		TrueQ[Quiet[Catch[
			With[{qry = DeviceCount[CUDAQ]},
				If[Head[qry] === Rule,
					("Number Of Devices" /. qry) >= 1,
					False
				]
			]
		]]]
	]

SyntaxInformation[CUDAInformation] = {"ArgumentsPattern" -> {_., _.}}	
CUDAInformation[] :=
	With[{res = Catch[Check[WGLQuery[CUDAInformation, CUDAQ], $Failed]]},
		(1 /. res) /; (res =!= $Failed)
	]
	
CUDAInformation[args:nonOptionPattern] ? (Function[{arg}, OptionsCheck[arg, CUDAInformation, 1, 2], HoldFirst]) :=
	With[{res = Catch[Check[iCUDAInformation[args], $Failed]]},
		res /; res =!= $Failed
	]

iCUDAInformation[query:("CurrentMemoryUsage" | "MaximumUsableMemory")] /; ListQ[CUDAInformation[]]  :=
	Module[{res},
		res = CUDAQ[];
		If[res,
			WGLQuery[CUDAInformation, query],
			$Failed
		]
	]
iCUDAInformation[device_Integer] /; ListQ[CUDAInformation[]]  :=
	WGLQuery[CUDAInformation, 1, device]
iCUDAInformation[device_Integer, prop_String] /; ListQ[CUDAInformation[]]  :=
	With[{res = WGLQuery[CUDAInformation, 1, device, prop]},
		(prop /. res) /; (res =!= $Failed)
	]
iCUDAInformation[args___] /; (Message[CUDAInformation::args, {args}]; False) := $Failed

CUDAInformation`SystemInformation[] /; !PacletInstalledQ[] :=
	$Failed
CUDAInformation`SystemInformation[] /; PacletInstalledQ[] :=
	If[CUDAQ[] === False,
		$Failed,
		Quiet[Catch[
			{
				"Driver Version" -> CUDADriverVersion[],
				"Library Version" -> LibraryVersionString[GPUTools`Internal`$CUDALibraryVersion],
				"Fastest Device" -> CUDALink`Internal`FastestDevice[],
				"Toolkit Version" -> CUDALink`Internal`$ToolkitVersion,
				"Detailed Information" ->
					{
						"Driver Path" -> "Path" /. GPUTools`Internal`$NVIDIADriverLibraryVersion,
						"Library Path" -> "Path" /. GPUTools`Internal`$CUDALibraryVersion,
						"System Libraries" -> GPUTools`Internal`$CUDALinkSystemLibaries
					}
			} ~Join~ (CUDAInformation`SystemInformation /@ Range[$CUDADeviceCount])
		]]
	]

CUDAInformation`SystemInformation[devId_Integer /; Positive[devId]] /; PacletInstalledQ[] := 
	If[CUDALink`Internal`SystemSupportedQ[] === False,
		$Failed,
		
		If[devId > $CUDADeviceCount,
			Message[CUDAInformation::invdevid, devId];
			Return[$Failed]
		];
		devId ->
			{"Name" -> CUDAInformation[devId, "Name"],
			 "Total Memory" -> CUDAInformation[devId, "Total Memory"],
			 "Clock Rate" -> CUDAInformation[devId, "Clock Rate"],
			 "Core Count" -> CUDAInformation[devId, "Core Count"],
			 "Supports Double Precision" -> CUDALink`Internal`SupportsDoublePrecisionQ[devId],
			 "Detailed Information" ->
			 	Select[CUDAInformation[devId], FreeQ[{"Name", "Total Memory", "Clock Rate", "Core Count"}, First@#]&]
			}
	]

CUDAInformation`FormatSystemInformation[_, info_] :=
	info
CUDAInformation`FormatSystemInformation["Total Memory", info_] :=
	GPUTools`Utilities`PrettyPrintMemory[info]


CUDALink`Internal`FastestDevice[] := CUDALink`Internal`FastestDevice[] =
	Module[{devinfo = Table[{If[CUDAInformation[i, "Compute Capabilities"] >= 2.0, 10000, 1]*CUDAInformation[i, "Clock Rate"]*CUDAInformation[i, "Core Count"], i}, {i, $CUDADeviceCount}]},
		If[Flatten[devinfo] === {},
			$Failed,
			Last[Last[Sort[devinfo]]]
		]
	]

CUDADriverVersion[] :=
	With[{res = CUDADriverVersion[GPUTools`Internal`$NVIDIADriverLibraryVersion]},
		res /; StringQ[res]
	]

CUDADriverVersion[ver_List | ver_Association] :=
	Which[
		ver === {},
			"",
		$OperatingSystem === "Unix" || $OperatingSystem === "MacOSX",
			StringJoin[Riffle[ToString /@ {
				GPUTools`Utilities`LibraryGetMajorVersion[ver],
				GPUTools`Utilities`LibraryGetMinorVersion[ver],
				GPUTools`Utilities`LibraryGetRevisionVersion[ver]
			}, "."]],
		True,
			With[{v = GPUTools`Utilities`LibraryGetRevisionVersion[ver]},
				ToString[Mod[v * 100, 1000]]
			]
	]

CUDADriverVersion[_] /; (Message[CUDADriverVersion::nodriv]; False) := $Failed

$CUDADeviceCount := Length[CUDAInformation[]]

$WGLPlatformQueryList = {}

$WGLDeviceQueryList := Flatten[{
	"Name" -> "Device Name",
	"Clock Rate",
	"Compute Capabilities",
	"GPU Overlap",
	"Maximum Block Dimensions",
	"Maximum Grid Dimensions",
	"Maximum Threads Per Block",
	"Maximum Shared Memory Per Block",
	"Total Constant Memory",
	"Warp Size",
	"Maximum Pitch",
	"Maximum Registers Per Block",
	"Texture Alignment",
	"Multiprocessor Count",
	"Core Count",
	"Execution Timeout",
	"Integrated",
	"Can Map Host Memory",
	"Compute Mode",
	"Texture1D Width",
	"Texture2D Width",
	"Texture2D Height",
	"Texture3D Width",
	"Texture3D Height",
	"Texture3D Depth",
	"Texture2D Array Width",
	"Texture2D Array Height",
	"Texture2D Array Slices",
	"Surface Alignment",
	"Concurrent Kernels",
	"ECC Enabled",
	"TCC Enabled",
	"Total Memory"
}]

(******************************************************************************)

(* ::Subsection:: *)
(* CUDA Memory *)

Options[CUDAMemoryLoad] = Options[CUDAMemoryAllocate] = 
	FilterRules[Options[GPUAddMemory], Except["Platform"]]
	

SyntaxInformation[CUDAMemoryLoad] = {"ArgumentsPattern" -> {_, _., OptionsPattern[]}}
CUDAMemoryLoad[arg__, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDAMemoryLoad, 1, 2], HoldFirst]):=
	With[{res = iCUDAMemoryLoad[CUDAMemoryLoad, arg, opts]},
		res /; Head[res] === CUDAMemory
	] 
CUDAMemoryLoad[args___] /; (ArgumentCountQ[CUDAMemoryLoad, Length[{args}], 1, 2]; False) := $Failed 

iCUDAMemoryLoad[errMsgHd_, symbol_, {"Constant" | Constant, type_}, opts:OptionsPattern[]] :=
	With[{res = Catch[GPUAddConstantMemory[CUDAMemory, errMsgHd, symbol, type, opts]]},
		res
	]
iCUDAMemoryLoad[errMsgHd_, args__, opts:OptionsPattern[]] :=
	With[{res = Catch[GPUAddMemory[CUDAMemory, errMsgHd, args, opts]]},
		res
	]
iCUDAMemoryLoad[errMsgHd_, args___] /; (Message[errMsgHd::args, {args}]; False) := $Failed 


SyntaxInformation[CUDAMemoryAllocate] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}}
CUDAMemoryAllocate[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDAMemoryAllocate, 2, 2], HoldFirst]) :=
	With[{res = iCUDAMemoryAllocate[CUDAMemoryAllocate, args, opts]},
		res /; Head[res] === CUDAMemory
	]
CUDAMemoryAllocate[args___] /; (ArgumentCountQ[CUDAMemoryAllocate, Length[{args}], 2, 2]; False) := $Failed

iCUDAMemoryAllocate[errMsgHd_, args__, opts:OptionsPattern[]] :=
	With[{res = Catch[GPUAllocateMemory[CUDAMemory, errMsgHd, args, opts]]},
		res
	]
 

SyntaxInformation[CUDAMemoryUnload] = {"ArgumentsPattern" -> {__}}
CUDAMemoryUnload[args:nonOptionPattern] ? (Function[{arg}, OptionsCheck[arg, CUDAMemoryUnload, 1, Infinity], HoldFirst]) :=
	Module[{res = GPUDeleteMemory[CUDAMemory, CUDAMemoryUnload, args]},
		res = Apply[And, TrueQ /@ res];
		Null /; TrueQ[res]
	]	
CUDAMemoryUnload[args___] /; (ArgumentCountQ[CUDAMemoryUnload, Length[{args}], 1, Infinity]; False) := $Failed 
	
SyntaxInformation[CUDAMemoryGet] = {"ArgumentsPattern" -> {_}}
CUDAMemoryGet[args_, opt_:"Original"] ? (Function[{arg}, OptionsCheck[arg, CUDAMemoryGet, 1, 2], HoldFirst]) :=
	Module[{res = Catch[GPUGetMemory[CUDAMemory, CUDAMemoryGet, args, opt]]},
		res /; res =!= $Failed
	]
CUDAMemoryGet[args___] /; (ArgumentCountQ[CUDAMemoryGet, Length[{args}], 1, 2]; False) := $Failed 
	
Format[mem:CUDAMemory[_Integer, ___], StandardForm] := CUDAMemory["<" <> ToString[WGLMemoryGetID[mem]] <> ">", WGLMemoryGetType[mem]]
Format[mem:CUDAMemory[symbol_String, ___, "Type" -> type_, ___, "Sharing" -> Constant, ___], StandardForm] := CUDAMemory["<" <> symbol <> ">", {Constant, type}]

SyntaxInformation[CUDAMemoryCopyToDevice] = {"ArgumentsPattern" -> {_}}
CUDAMemoryCopyToDevice[args:nonOptionPattern] ? (Function[{arg}, OptionsCheck[arg, CUDAMemoryCopyToDevice, 1, 1], HoldFirst]) :=
	With[{res = GPUCopyMemoryToDevice[CUDAMemory, CUDAMemoryCopyToDevice, args]},
		res /; Head[res] === CUDAMemory
	]

SyntaxInformation[CUDAMemoryCopyToHost] = {"ArgumentsPattern" -> {_}}
CUDAMemoryCopyToHost[args:nonOptionPattern] ? (Function[{arg}, OptionsCheck[arg, CUDAMemoryCopyToHost, 1, 1], HoldFirst]) :=
	With[{res = GPUCopyMemoryToHost[CUDAMemory, CUDAMemoryCopyToHost, args]},
		res /; Head[res] === CUDAMemory
	]
	
SyntaxInformation[CUDAMemoryInformation] = {"ArgumentsPattern" -> {_}}
CUDAMemoryInformation[args:nonOptionPattern] ? (Function[{arg}, OptionsCheck[arg, CUDAMemoryInformation, 1, 1], HoldFirst]) :=
	With[{res = GPUMemoryInformation[CUDAMemory, CUDAMemoryInformation, args]},
		res /; ListQ[res]
	]

(******************************************************************************)

CUDALink`Internal`CUDAConstantMemoryCopy[constmem_?constantMemoryQ, regmem0_] := 
	Module[{regmem, res},
		regmem = If[Head[regmem0] === CUDAMemory,
			regmem0,
			CUDAMemoryLoad[regmem0]
		];
		res = Catch[GPUCopyConstantMemory["To"][CUDAMemory, CUDALink`Internal`CUDAConstantMemoryCopy, constmem, regmem]];
		res /; res =!= $Failed
	]
CUDALink`Internal`CUDAConstantMemoryCopy[regmem0_, constmem_?constantMemoryQ] := 
	Module[{regmem, res},
		regmem = If[Head[regmem0] === CUDAMemory,
			regmem0,
			CUDAMemoryLoad[regmem0]
		];
		res = Catch[GPUCopyConstantMemory["From"][CUDAMemory, CUDALink`Internal`CUDAConstantMemoryCopy, constmem, regmem]];
		res /; res =!= $Failed
	]	
CUDALink`Internal`CUDAConstantMemoryCopy[regmem1_, regmem2_] /; (Message[CUDALink`Internal`CUDAConstantMemoryCopy::noconst]; False) := $Failed 

constantMemoryQ[CUDAMemory[_String, ___, "Sharing" -> Constant, ___]] := True
constantMemoryQ[___] := False
	
(******************************************************************************)

Unprotect[Image]
CUDAMemory /: Image[mem:CUDAMemory[_Integer, ___], opts___] :=
	With[{imgType = GetImageType[WGLMemoryGetType[mem]], data = CUDAMemoryGet[mem]},
		If[ImageQ[data],
			Image[data, opts],
			Image[data, imgType, opts]
		]
	] 
Protect[Image]

(******************************************************************************)
	
(******************************************************************************)

DeriveOptions[opts_List] := 
	stripDuplicates[Join[opts, Options[NVCCCompiler], Options[GPUFunctionLoad]]]

$CUDALinkBinaryDirectory = Quiet[
	Check[FileNameJoin[{
			ApplicationDataUserDirectory["CUDALink"], 
    		"BuildFolder", $MachineName <> "-" <> ToString[$ProcessID]
    	}],
  		$Failed
	]
  ]

$Epilog :=
	Quiet[Catch[
		If[DirectoryQ[$CUDALinkBinaryDirectory],
			DeleteDirectory[$CUDALinkBinaryDirectory]
		]
	]]
(******************************************************************************)
(*Support File[..] as an argument*)
isFile[exp_,head_:CUDAFunctionLoad]:=StringQ[getFile[exp,head]];

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
(* ::Subsection:: *)
(* CUDAFunction *)

Options[CUDAFunctionLoad] = Options[CompileCUDAKernel] = Options[iCompileCUDAKernel] = 
	Sort@FilterRules[
		DeriveOptions[{
			"Device" -> Automatic
			,
			"CUDAArchitecture" -> Automatic
			,
			"CreateCUBIN" -> True
			,
			"CreatePTX" -> False
			,
			"UnmangleCode" -> True
			,
			"TargetPrecision" -> Automatic
			,
			"TargetDirectory" -> $CUDALinkBinaryDirectory
			,
			"Defines" -> {}
			,
			"SystemDefines" -> Automatic
		}], Except[
			"ExtraObjectFiles" | "SystemLibraries" | "Libraries" | "CompilerName" | "LibraryDirectories" | 
			"XCompileOptions" | "TargetSystemID" | "CreateBinary" | "SystemCompileOptions" | "SystemLinkerOptions" | "LinkerOptions" |
			"PostCompileCommands" | "PreCompileCommands" | "SystemLibraryDirectories" | "BinaryQ" | "Platform" | "MprepOptions" | "BuildOptions" | "Source"
		]
	]

SyntaxInformation[CUDAFunctionLoad] = {"ArgumentsPattern" -> {_, _, _, _., OptionsPattern[]}}
CUDAFunctionLoad[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDAFunctionLoad, 3, 4], HoldFirst]) :=
	Module[{res = Catch[iCUDAFunctionLoad[CUDAFunctionLoad, args, opts]]},
		res /; res =!= $Failed
	]

iCUDAFunctionLoad[errMsgHd_, "", _, _, _, OptionsPattern[]] :=
	Throw[Message[errMsgHd::invsrc, ""]; $Failed]
iCUDAFunctionLoad[errMsgHd_, {}, _, _, _, OptionsPattern[]] :=
	Throw[Message[errMsgHd::emptyprog]; $Failed]
iCUDAFunctionLoad[errMsgHd_, prog:{_} /; !(Quiet@isFile[First@prog]), ___] :=
	Throw[Message[errMsgHd::invprog, First[prog]]; $Failed]
iCUDAFunctionLoad[errMsgHd_, prog_ /; (!StringQ[prog] && !ListQ[prog] && Head[prog]=!=File), ___] :=
	Throw[Message[errMsgHd::invprog, prog]; $Failed]
iCUDAFunctionLoad[errMsgHd_, prog:({_?isFile}|_File) /; !(FileExistsQ[First@prog]||FileExistsQ[prog]), ___] :=
	Throw[Message[errMsgHd::nofile, First[prog]]; $Failed]
iCUDAFunctionLoad[errMsgHd_, prog_, kernelName_, args_, blockDimensions_:256, opts:OptionsPattern[]] :=
	Module[{res = $Failed, cmpres = Reap[CompileCUDAKernel[CUDAFunctionLoad, prog, opts], $TemporaryFile], fileName, shellOut, shellCommand},
		
		If[!FreeQ[cmpres, $Failed],
			Throw[$Failed]
		];
		{fileName, shellOut, shellCommand} = First[cmpres];
		res = Catch[
			GPUFunctionLoad[
				CUDAFunction, CUDAProgram, CUDAFunctionLoad, fileName, kernelName, args, blockDimensions,
				"Source" -> prog, "BinaryQ" -> True, "BuildOptions" -> shellCommand,
				Sequence@@FilterRules[{opts},  Options[GPUFunctionLoad]]
			]
		];

		If[res === $Failed,
			Quiet[DeleteFile /@ Flatten[Last[cmpres]]],
			Quiet[WGLSetDeleteProgramFileOnExit[errMsgHd, #]& /@ Flatten[First[Last[cmpres]]]]
		];
		
		res
	]

CUDAFunctionInformation[args:nonOptionPattern] ? (Function[{arg}, OptionsCheck[arg, CUDAFunctionInformation, 1, 1], HoldFirst]) :=
	With[{res = Catch[GPUFunctionInformation[CUDAFunction, CUDAProgram, CUDAFunctionInformation, args]]},
		res /; ListQ[res]
	]

CUDAFunction[functionArgs__][callArgs___] ? (Function[{arg}, OptionsCheck[arg, CUDAFunctionInformation], HoldFirst]) :=
	With[{res = Catch[GPUFunctionLaunch[CUDAFunction, CUDAProgram, CUDAMemory, CUDAFunction, {functionArgs}, {callArgs}]]},
		res /; ListQ[res]
	]

Format[CUDAFunction[CUDAProgram[_Integer, ___], kernelName_String, args_List, OptionsPattern[]], StandardForm] :=
	CUDAFunction["<>", kernelName, args]
	

CompileCUDAKernel[errMsgHd_, prog_, opts:OptionsPattern[]] :=
	Catch[
		iCompileCUDAKernel[errMsgHd, prog, opts]
	]
iCompileCUDAKernel[errMsgHd_, prog_, OptionsPattern[]] :=
	Throw[Message[errMsgHd::invprog, prog]; $Failed]
iCompileCUDAKernel[errMsgHd_, progexp:_File, opts:OptionsPattern[]]:=(iCompileCUDAKernel[errMsgHd, {progexp}, opts])
iCompileCUDAKernel[errMsgHd_, progexp:(_String | {_?isFile}), opts:OptionsPattern[]] :=
	Module[{
			fileName, createLibraryOpts, device, useSinglePrecisionQ, arch, progText, timeStamp,prog,
			sysDefines, defines, shellOut, shellCommand, shellCommandFunction, compileOpts, shellOutputFunction
		   },
		
		InitializeWGL[iCompileCUDAKernel, errMsgHd, opts];
		
		If[ListQ[progexp],
			prog=Flatten[getFile[#]&/@progexp];
			Which[
				MemberQ[{"ptx", "cubin", "so", "dll", "dylib"}, FileExtension[First[prog]]],
					Return[{prog, "", ""}],
				!FileExistsQ[First[prog]],
					Throw[Message[errMsgHd::nofile, First[prog]]; $Failed]
			],
			prog=progexp;
		];
		
		device = GetAndCheckOption[errMsgHd, "Device", {opts}];
		useSinglePrecisionQ = GetAndCheckOption[errMsgHd, "TargetPrecision", {opts}] === "Single";
		
		arch = OptionValue[iCompileCUDAKernel, {opts}, "CUDAArchitecture"];
		sysDefines = OptionValue[iCompileCUDAKernel, {opts}, "SystemDefines"];
		defines = OptionValue[iCompileCUDAKernel, {opts}, "Defines"];
		compileOpts = OptionValue[iCompileCUDAKernel, {opts}, "CompileOptions"];
		shellCommandFunction = OptionValue[iCompileCUDAKernel, {opts}, "ShellCommandFunction"];
		shellOutputFunction = OptionValue[iCompileCUDAKernel, {opts}, "ShellOutputFunction"];
		
		timeStamp = If[ListQ[prog],
			FileDate[First[prog], "Modification"],
			DateList[]
		];
		
		With[{cached = CachedCompile[prog, If[ListQ[prog], timeStamp, prog], arch, compileOpts, sysDefines, defines, useSinglePrecisionQ]},
			If[ListQ[cached] && FileExistsQ["FileName" /. cached],
				InvokeShellCommandFunction[shellCommandFunction,  "ShellCommand" /. cached];
				InvokeShellOutputFunction[shellOutputFunction,  "ShellOutput" /. cached];
				Return[{"FileName" /. cached, "ShellOutput" /. cached, "ShellCommand" /. cached}]
			]
		];
		
		progText = If[ListQ[prog],
			Import[First[prog], "Text"],
			prog
		];
		
        fileName = "CUDAFunction-" <> ToString[RandomInteger[{0, 10000}]]; 
        createLibraryOpts = Quiet@stripDuplicates[Flatten[Join[{
        	"Compiler" -> NVCCCompiler,
    		"CreateBinary" -> True,
    		"CUDAArchitecture" -> {},
    		"Defines" -> Flatten[
    			If[sysDefines === Automatic,
    				Join[
        				{"USING_CUDA_FUNCTION"->1},
        				{
							"mint" -> If[TrueQ[$64BitMint],
								Switch[$SystemID,
									"Windows-x86-64",
										"\"long long\"",
									"Linux-x86-64" | "MacOSX-x86-64",
										"\"long long int\"",
									_,
										"int"
								],
								"int"
							]
						},
						If[useSinglePrecisionQ,
							{"Real_t"->"float"},
							{"Real_t"->"double", "USING_DOUBLE_PRECISIONQ"->1}
						]
    				],
    				sysDefines
    			] ~Join~ {defines}
    		],
    		"ShellOutputFunction" -> ((shellOut = #)&),
    		"ShellCommandFunction" -> ((shellCommand = #)&),
    		"SystemCompileOptions" -> 
    			With[{sm = If[arch === Automatic, getCUDAArchitecture[device], arch]},
    				"-arch=" <> sm
    			],
    		FilterRules[{opts},  Options[NVCCCompiler]],
    		FilterRules[FilterRules[Options[iCompileCUDAKernel], Except[Alternatives @@ (First /@ Select[{opts}, ListQ[#] && # =!= {}&])]],  Options[NVCCCompiler]]
    	}]]];
    	
    	Block[{CCompilerDriver`$ErrorMessageHead = errMsgHd},
        	fileName = Check[
        		Quiet[
					CreateExecutable[
						prog, fileName, Sequence@@createLibraryOpts
					], errMsgHd::wddirty
        		],
				$Failed
        	]
    	];
    	If[fileName =!= $Failed,
			InvokeShellCommandFunction[shellCommandFunction,  shellCommand];
			InvokeShellOutputFunction[shellOutputFunction,  shellOut];
    		CachedCompile[prog, If[ListQ[prog], timeStamp, prog], arch, compileOpts, sysDefines, defines, useSinglePrecisionQ] = {
    			"FileName" -> fileName,
    			"ShellOutput" -> shellOut,
    			"ShellCommand" -> shellCommand
    		};
    		Sow[{fileName, shellOut, shellCommand}, $TemporaryFile],
    		
    		(* else *)
    		If[FreeQ[$MessageList, HoldPattern[MessageName[errMsgHd, _]]], 
    			Message[errMsgHd::cmpf]
    		];
    		If[shellCommandFunction =!= None,
    			shellCommandFunction[shellCommand]
    		];
    		If[shellOutputFunction =!= None,
    			shellOutputFunction[shellOut]
    		];
    		Throw[$Failed]
    	]
	]

iCompileCUDAKernel[errMsgHd_, ___] := Throw[Message[errMsgHd::cmpf]; $Failed]

(******************************************************************************)

textImport[file_] :=
	Module[{strm, str},
		strm = OpenRead[file];
		str = StringJoin[Riffle[ReadList[strm, String], "\n"]];
		Close[strm];
		str
	]

(******************************************************************************)

getCUDAArchitecture[device_] := getCUDAArchitecture[device] =
	Which[
		CUDAInformation[device, "Compute Capabilities"] == 5.3,
			"sm_53",
		CUDAInformation[device, "Compute Capabilities"] == 5.2,
			"sm_52",
		CUDAInformation[device, "Compute Capabilities"] == 5.1,
			"sm_51",
		CUDAInformation[device, "Compute Capabilities"] == 5.0,
			"sm_50",
		CUDAInformation[device, "Compute Capabilities"] == 3.5,
			"sm_35",
		CUDAInformation[device, "Compute Capabilities"] == 3.0,
			"sm_30",
		CUDAInformation[device, "Compute Capabilities"] == 2.1,
			"sm_21",
		CUDAInformation[device, "Compute Capabilities"] == 2.0,
			"sm_20",
		CUDAInformation[device, "Compute Capabilities"] == 1.3,
			"sm_13",
		CUDAInformation[device, "Compute Capabilities"] == 1.2,
			"sm_12",
		CUDAInformation[device, "Compute Capabilities"] == 1.1,
			"sm_11",
		CUDAInformation[device, "Compute Capabilities"] == 1.0,
			"sm_10",
		True,
			"sm_" <> 
				Module[{cc = StringReplace[ToString[CUDAInformation[device, "Compute Capabilities"]], "." -> ""]},
					If[StringLength[cc] == 1,
						cc <> "0",
						cc
					]
				]
	]
(******************************************************************************)

InvokeShellCommandFunction[fun_,  cmd_String] :=
	If[fun =!= None,
		fun[cmd]
	]
InvokeShellOutputFunction[fun_,  out_String] :=
	If[fun =!= None,
		fun[out]
	]
(******************************************************************************)
(******************************************************************************)

(* ::Subsection:: *)
(* CUDAProfile *)
Options[CUDAProfile] = Options[CUDAProfile]
SetAttributes[CUDAProfile, Attributes[GPUProfile]]
CUDAProfile[args__, opts:OptionsPattern[]] :=
	Module[{res},
		res = If[CUDAQ[],
			True,
			If[$InitializeWGLError === {},
				Message[CUDAProfile::nodev],
				InitializationErrorMessage[CUDAProfile, $InitializeWGLError]
			];
			False
		];
		(
			res = Catch[GPUProfile[CUDAProfile, Unevaluated[args], opts]];
			res /; res =!= $Failed
		) /; TrueQ[res]
	]
(******************************************************************************)
(******************************************************************************)

(* ::Subsection:: *)
(* CUDA Image Processing *)


Options[CUDAImageConvolve] = Options[iCUDAImageConvolve] = {"Device" -> Automatic, Padding -> "Fixed", "OutputMemory" -> None} 

SyntaxInformation[CUDAImageConvolve] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}}
CUDAImageConvolve[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDAImageConvolve, 2, 2], HoldFirst]) :=
	With[{res = iCUDAImageConvolve[CUDAImageConvolve, args, opts]},
		res /; MemberQ[{Image, List, CUDAMemory}, Head[res]]
	]
iCUDAImageConvolve[errMsgHd_, input_, kernel_, opts:OptionsPattern[]] :=
	Module[{res, runRes, inputMem, outputMem, kernelMem, border, device},
		res = Check[
			runRes = Reap[Catch[
				InitializeWGL[iCUDAImageConvolve, errMsgHd, opts];
				device = GetAndCheckOption[errMsgHd, "Device", {opts}];
				border = GetAndCheckOption[errMsgHd, Padding, {opts}];
				outputMem = GetAndCheckOption[errMsgHd, "OutputMemory", {opts}];
				
				inputMem = AddTemporaryMemory[errMsgHd, input, If[Image`PossibleImageQ[input], Real, Automatic]];
				If[inputMem === $Failed,
					Throw[$Failed]
				];
				kernelMem = AddTemporaryMemory[errMsgHd, kernel, If[MemberQ[{"Byte", "UnsignedByte", "Short"}, WGLMemoryGetType[inputMem]], WGLMemoryGetType[inputMem], Automatic]];
				If[outputMem === None,
					outputMem = If[Head[input] === CUDAMemory,
						AllocateClonedMemory[errMsgHd, inputMem],
						AllocateTemporaryClonedMemory[errMsgHd, inputMem]
					]
				];
				oCUDAImageConvolve[errMsgHd, inputMem, outputMem, kernelMem, border]
			]],
			$Failed
		];
		
		res = If[FreeQ[res, $Failed],
			If[Head[input] === CUDAMemory,
				outputMem,
				Catch[GetOutputMemory[errMsgHd, outputMem, opts]]
			],
			$Failed
		];

		If[ListQ[runRes],
			GPUDeleteMemory[CUDAMemory, errMsgHd, #]& /@ Flatten[Last[runRes]]
		];
		
		res
	]

oCUDAImageConvolve[errMsgHd_, inputMem_CUDAMemory, outputMem_CUDAMemory, kernel_CUDAMemory, border:("Fixed" | _Integer | _Real), blockDim_:16] :=
	Module[{res, borderStrategy, borderValue},
		
		SetErrorMessageHead[errMsgHd];
		
		{borderStrategy, borderValue} = If[border === "Fixed",
			{"Fixed", 0.0},
			{"Constant", 1.0 * border}
		];
		
		res = cImageProcessingConvolution[$SingleOrDoubleLibrary][
			WGLMemoryGetID[inputMem],
			WGLMemoryGetID[outputMem], 
			WGLMemoryGetID[kernel],
			borderStrategy,
			borderValue,
			blockDim
		];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]
(******************************************************************************)

Options[CUDABoxFilter] = Options[iCUDABoxFilter] = Options[CUDAImageConvolve]
SyntaxInformation[CUDABoxFilter] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}}
CUDABoxFilter[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDABoxFilter, 2, 3], HoldFirst]) := 
	With[{res = iCUDABoxFilter[CUDABoxFilter, args, opts]},
		res /; MemberQ[{Image, List, CUDAMemory}, Head[res]]
	]

(******************************************************************************)

iCUDABoxFilter[errMsgHd_, input_, radius:(_Integer | _Real), scale_:1, opts:OptionsPattern[]] /; NumericQ[scale] :=
	Module[{res, runRes, inputMem, outputMem, border, device},		
		Which[
			scale == 0,
				Return[Message[errMsgHd::invscl, scale]; $Failed],
			IntegerPart[scale] != scale,
				Return[Message[errMsgHd::intscale, scale]; $Failed],
			Negative[radius],
				Return[Message[errMsgHd::radneg, radius]; $Failed],
			radius == 0,
				Which[
					ListQ[input] || ImageQ[input],
						Return[input],
					Image`PossibleImageQ[input],
						Return[Image[input]],
					Head[input] === CUDAMemory,
						Return[CUDAMemoryLoad[CUDAMemoryGet[input]]]
				]
		];
		res = Check[
			runRes = Reap[Catch[
				InitializeWGL[iCUDABoxFilter, errMsgHd, opts];
				device = GetAndCheckOption[errMsgHd, "Device", {opts}];
				border = GetAndCheckOption[errMsgHd, Padding, {opts}];
				outputMem = GetAndCheckOption[errMsgHd, "OutputMemory", {opts}];
				inputMem = AddTemporaryMemory[errMsgHd, input, If[Image`PossibleImageQ[input], Real, Automatic]];
				If[outputMem === None,
					outputMem = If[Head[input] === CUDAMemory,
						AllocateClonedImageMemory[errMsgHd, inputMem, input],
						AllocateTemporaryClonedImageMemory[errMsgHd, inputMem, input]
					]
				];
				oCUDABoxFilter[errMsgHd, inputMem, outputMem, IntegerPart[radius], IntegerPart[scale], border]
			]],
			$Failed
		];
		
		res = If[FreeQ[res, $Failed],
			If[Head[input] === CUDAMemory,
				outputMem,
				Catch[GetOutputMemory[errMsgHd, outputMem, opts]]
			],
			$Failed
		];

		If[ListQ[runRes],
			GPUDeleteMemory[CUDAMemory, errMsgHd, #]& /@ Flatten[Last[runRes]]
		];
		
		res
	]
iCUDABoxFilter[errMsgHd_, args___, OptionsPattern[]] := Return[Message[errMsgHd::invargs, {args}]; $Failed]

oCUDABoxFilter[errMsgHd_, inputMem_CUDAMemory, outputMem_CUDAMemory, radius_Integer, scale_Integer, border:("Fixed" | _Integer | _Real), blockDim_:16] :=
	Module[{res, borderStrategy, borderValue},
		
		SetErrorMessageHead[errMsgHd];

		{borderStrategy, borderValue} = If[border === "Fixed",
			{"Fixed", 0.0},
			{"Constant", 1.0 * border}
		];
		
		res = cImageProcessingBoxFilter[$SingleOrDoubleLibrary][
			WGLMemoryGetID[inputMem],
			WGLMemoryGetID[outputMem],
			radius,
			scale,
			borderStrategy,
			borderValue,
			blockDim
		];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]
	
(******************************************************************************)

Options[CUDAImageAdd] = Options[CUDAImageSubtract] = Options[CUDAImageMultiply] = Options[CUDAImageDivide] = Options[iCUDAImageBinaryOperator] = 
	{"Device" -> Automatic, "OutputMemory" -> None}
SyntaxInformation[CUDAImageAdd] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}}
CUDAImageAdd[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDAImageAdd, 2, 2], HoldFirst]) :=
	With[{res = iCUDAImageAdd[args, opts]},
		res /; MemberQ[{Image, List, CUDAMemory}, Head[res]]
	]
iCUDAImageAdd[inputMem_, constant:(_Real | _Integer | _Rational), opts:OptionsPattern[]] :=
	iCUDAImageBinaryOperator["AddConstant"][CUDAImageAdd, inputMem, constant, opts]
iCUDAImageAdd[inputMem1_, inputMem2_ /; Head[inputMem2] === CUDAMemory || ListQ[inputMem2] || Image`PossibleImageQ[inputMem2], opts:OptionsPattern[]] :=
	iCUDAImageBinaryOperator["Add"][CUDAImageAdd, inputMem1, inputMem2, opts]
iCUDAImageAdd[args___, OptionsPattern[]] :=
	Return[Message[CUDAImageAdd::invargs, {args}]; $Failed]
		
(******************************************************************************)
	
SyntaxInformation[CUDAImageSubtract] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}}
CUDAImageSubtract[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDAImageSubtract, 2, 2], HoldFirst]) :=
	With[{res = iCUDAImageSubtract[args, opts]},
		res /; MemberQ[{Image, List, CUDAMemory}, Head[res]]
	]
iCUDAImageSubtract[inputMem_, constant:(_Real | _Integer | _Rational), opts:OptionsPattern[]] :=
	iCUDAImageBinaryOperator["SubtractConstant"][CUDAImageSubtract, inputMem, constant, opts]
iCUDAImageSubtract[inputMem1_, inputMem2_ /; Head[inputMem2] === CUDAMemory || ListQ[inputMem2] || Image`PossibleImageQ[inputMem2], opts:OptionsPattern[]] :=
	iCUDAImageBinaryOperator["Subtract"][CUDAImageSubtract, inputMem1, inputMem2, opts]
iCUDAImageSubtract[args___, OptionsPattern[]] :=
	Return[Message[CUDAImageSubtract::invargs, {args}]; $Failed]
		
(******************************************************************************)

SyntaxInformation[CUDAImageMultiply] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}}
CUDAImageMultiply[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDAImageMultiply, 2, 2], HoldFirst]) :=
	With[{res = iCUDAImageMultiply[args, opts]},
		res /; MemberQ[{Image, List, CUDAMemory}, Head[res]]
	]
iCUDAImageMultiply[inputMem_, constant:(_Real | _Integer | _Rational), opts:OptionsPattern[]] :=
	iCUDAImageBinaryOperator["MultiplyConstant"][CUDAImageMultiply, inputMem, constant, opts]
iCUDAImageMultiply[inputMem1_, inputMem2_ /; Head[inputMem2] === CUDAMemory || ListQ[inputMem2] || Image`PossibleImageQ[inputMem2], opts:OptionsPattern[]] :=
	iCUDAImageBinaryOperator["Multiply"][CUDAImageMultiply, inputMem1, inputMem2, opts]
iCUDAImageMultiply[args___, OptionsPattern[]] :=
	Return[Message[CUDAImageMultiply::invargs, {args}]; $Failed]
		
(******************************************************************************)

SyntaxInformation[CUDAImageDivide] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}}
CUDAImageDivide[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDAImageDivide, 2, 2], HoldFirst]) :=
	With[{res = iCUDAImageDivide[args, opts]},
		res /; MemberQ[{Image, List, CUDAMemory}, Head[res]]
	]
iCUDAImageDivide[inputMem_, constant_ /; N[constant] === 0.0, opts:OptionsPattern[]] :=
	Return[Message[CUDAImageDivide::divzero]; $Failed]
iCUDAImageDivide[inputMem_, constant:(_Real | _Integer | _Rational), opts:OptionsPattern[]] :=
	iCUDAImageBinaryOperator["DivideConstant"][CUDAImageDivide, inputMem, constant, opts]
iCUDAImageDivide[inputMem1_, inputMem2_ /; Head[inputMem2] === CUDAMemory || ListQ[inputMem2] || Image`PossibleImageQ[inputMem2], opts:OptionsPattern[]] :=
	iCUDAImageBinaryOperator["Divide"][CUDAImageDivide, inputMem1, inputMem2, opts]
iCUDAImageDivide[args___, OptionsPattern[]] :=
	Return[Message[CUDAImageDivide::invargs, {args}]; $Failed]
	
(******************************************************************************)

iCUDAImageBinaryOperator[op:("AddConstant" | "SubtractConstant" | "DivideConstant" | "MultiplyConstant")][errMsgHd_, input_, constant_, opts:OptionsPattern[]] :=
	Module[{res, runRes, inputMem, outputMem, device},
		If[!NumericQ[constant],
			Throw[Message[errMsgHd::const, constant]; $Failed]
		];
		res = Check[
			runRes = Reap[Catch[
				InitializeWGL[iCUDAImageBinaryOperator, errMsgHd, opts];
				device = GetAndCheckOption[errMsgHd, "Device", {opts}];
				outputMem = GetAndCheckOption[errMsgHd, "OutputMemory", {opts}];
				inputMem = AddTemporaryMemory[errMsgHd, input, If[Image`PossibleImageQ[input], Real, Automatic]];
				If[outputMem === None,
					outputMem = If[Head[input] === CUDAMemory,
						AllocateClonedImageMemory[errMsgHd, inputMem, input],
						AllocateTemporaryClonedImageMemory[errMsgHd, inputMem, input]
					]
				];
				oCUDAImageBinaryOperator[op][errMsgHd, inputMem, outputMem, N[constant], SupportsDoublePrecisionQ[errMsgHd, device]]
			]],
			$Failed
		];
		
		res = If[FreeQ[res, $Failed],
			If[Head[input] === CUDAMemory,
				outputMem,
				Catch[GetOutputMemory[errMsgHd, outputMem, opts]]
			],
			$Failed
		];

		If[ListQ[runRes],
			GPUDeleteMemory[CUDAMemory, errMsgHd, #]& /@ Flatten[Last[runRes]]
		];
		
		res
	]

oCUDAImageBinaryOperator[op:("AddConstant" | "SubtractConstant" | "DivideConstant" | "MultiplyConstant")][errMsgHd_, inputMem_CUDAMemory, outputMem_CUDAMemory, constant_, usingSinglePrecisionQ_, blockDim_:16] :=
	Module[{res},
		
		SetErrorMessageHead[errMsgHd];
		
		res = Switch[op,
			"AddConstant",
				cImageProcessingAddConstant,
			"SubtractConstant",
				cImageProcessingSubtractConstant,
			"MultiplyConstant",
				cImageProcessingMultiplyConstant,
			"DivideConstant",
				cImageProcessingDivideConstant
		][$SingleOrDoubleLibrary][
			WGLMemoryGetID[inputMem],
			WGLMemoryGetID[outputMem],
			Which[
				IntegerQ[constant],
					"Integer",
				Head[constant] === Real,
					If[usingSinglePrecisionQ,
						"Float",
						"Double"
					],
				True,
					Throw[Message[errMsgHd::invcnst, constant]; $Failed]
			],
			1.0 * constant,
			blockDim
		];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]

iCUDAImageBinaryOperator[op:("Add" | "Subtract" | "Divide" | "Multiply")][errMsgHd_, input1_, input2_, opts:OptionsPattern[]] :=
	Module[{res, runRes, inputMem1, inputMem2, outputMem, device},
		res = Check[
			runRes = Reap[Catch[
				InitializeWGL[iCUDAImageBinaryOperator, errMsgHd, opts];
				
				Which[
					Image`PossibleImageQ[input1] && Image`PossibleImageQ[input2],
						If[ImageDimensions[input1] =!= ImageDimensions[input2],
							Throw[Message[errMsgHd::unmchdims, ImageDimensions[input1], ImageDimensions[input2]]; $Failed]
						],
					ListQ[input1] && ListQ[input2],
						If[Dimensions[input1] =!= Dimensions[input2],
							Throw[Message[errMsgHd::unmchdims, Dimensions[input1], Dimensions[input2]]; $Failed]
						]
				];
				
				device = GetAndCheckOption[errMsgHd, "Device", {opts}];
				outputMem = GetAndCheckOption[errMsgHd, "OutputMemory", {opts}];
				inputMem1 = AddTemporaryMemory[errMsgHd, input1, If[Image`PossibleImageQ[input1], Real, Automatic]];
				inputMem2 = AddTemporaryMemory[errMsgHd, input2, If[Image`PossibleImageQ[input2], Real, Automatic]];
				If[outputMem === None,
					outputMem = If[Head[input1] === CUDAMemory,
						AllocateClonedImageMemory[errMsgHd, inputMem1, input1],
						AllocateTemporaryClonedImageMemory[errMsgHd, inputMem1, input1]
					]
				];
				oCUDAImageBinaryOperator[op][errMsgHd, inputMem1, inputMem2, outputMem, SupportsDoublePrecisionQ[errMsgHd, device]]
			]],
			$Failed
		];

		res = If[FreeQ[res, $Failed],
			If[Head[input1] === CUDAMemory || Head[input2] === CUDAMemory, 
				outputMem,
				Catch[GetOutputMemory[errMsgHd, outputMem, opts]]
			],
			$Failed
		];

		If[ListQ[runRes],
			GPUDeleteMemory[CUDAMemory, errMsgHd, #]& /@ Flatten[Last[runRes]]
		];
		
		res
	]

oCUDAImageBinaryOperator[op:("Add" | "Subtract" | "Divide" | "Multiply")][errMsgHd_, inputMem1_CUDAMemory, inputMem2_CUDAMemory, outputMem_CUDAMemory, _, blockDim_:16] :=
	Module[{res},
		
		SetErrorMessageHead[errMsgHd];
		
		res = Switch[op,
			"Add",
				cImageProcessingAdd,
			"Subtract",
				cImageProcessingSubtract,
			"Multiply",
				cImageProcessingMultiply,
			"Divide",
				cImageProcessingDivide
		][$SingleOrDoubleLibrary][
			WGLMemoryGetID[inputMem1],
			WGLMemoryGetID[inputMem2],
			WGLMemoryGetID[outputMem],
			blockDim
		];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]


(******************************************************************************)

Options[CUDAErosion] = Options[CUDADilation] = Options[CUDAOpening] = Options[CUDAClosing] = Options[iCUDAImageMorphology] =
	{"Device" -> Automatic, Padding -> "Fixed", "OutputMemory" -> None}

SyntaxInformation[CUDAErosion] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}}
CUDAErosion[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDAErosion, 2, 2], HoldFirst]) :=
	With[{res = iCUDAErosion[args, opts]},
		res /; MemberQ[{Image, List, CUDAMemory}, Head[res]]
	]
iCUDAErosion[input_, radius_, opts:OptionsPattern[]] :=
	iCUDAImageMorphology["Erosion"][CUDAErosion, input, radius, opts]
	
(******************************************************************************)
	
SyntaxInformation[CUDADilation] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}}
CUDADilation[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDADilation, 2, 2], HoldFirst]) :=
	With[{res = iCUDADilation[args, opts]},
		res /; MemberQ[{Image, List, CUDAMemory}, Head[res]]
	]
iCUDADilation[input_, radius_, opts:OptionsPattern[]] :=
	iCUDAImageMorphology["Dilation"][CUDADilation, input, radius, opts]
		
(******************************************************************************)
		
SyntaxInformation[CUDAOpening] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}}
CUDAOpening[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDAOpening, 2, 2], HoldFirst]) :=
	With[{res = iCUDAOpening[args, opts]},
		res /; MemberQ[{Image, List, CUDAMemory}, Head[res]]
	]
iCUDAOpening[input_, radius_, opts:OptionsPattern[]] :=
	iCUDAImageMorphology["Opening"][CUDAOpening, input, radius, opts]
	
(******************************************************************************)
		
SyntaxInformation[CUDAClosing] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}}	
CUDAClosing[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDAClosing, 2, 2], HoldFirst]) :=
	With[{res = iCUDAClosing[args, opts]},
		res /; MemberQ[{Image, List, CUDAMemory}, Head[res]]
	]
iCUDAClosing[input_, radius_, opts:OptionsPattern[]] :=
	iCUDAImageMorphology["Closing"][CUDAClosing, input, radius, opts]
	
(******************************************************************************)

iCUDAImageMorphology["Erosion" | "Dilation" | "Opening" | "Closing"][errMsgHd_, _, radius:(_Integer | _Real)?Negative, OptionsPattern[]] :=
	Return[Message[errMsgHd::radneg, radius]; $Failed]
iCUDAImageMorphology["Erosion" | "Dilation" | "Opening" | "Closing"][errMsgHd_, _, radius_ /; !IntegerQ[radius] && Head[radius] =!= Real, OptionsPattern[]] :=
	Return[Message[errMsgHd::rad, radius]; $Failed]
iCUDAImageMorphology[op:("Erosion" | "Dilation" | "Opening" | "Closing")][errMsgHd_, input_, radius:(_Integer | _Real)?NonNegative, opts:OptionsPattern[]] :=
	Module[{res, runRes, inputMem, outputMem, border, device},
		res = Check[
			runRes = Reap[Catch[
				InitializeWGL[iCUDAImageMorphology, errMsgHd, opts];
				device = GetAndCheckOption[errMsgHd, "Device", {opts}];
				border = GetAndCheckOption[errMsgHd, Padding, {opts}];
				outputMem = GetAndCheckOption[errMsgHd, "OutputMemory", {opts}];
				inputMem = AddTemporaryMemory[errMsgHd, input, Automatic];
				If[outputMem === None,
					outputMem = If[Head[input] === CUDAMemory,
						AllocateClonedMemory[errMsgHd, inputMem, Automatic],
						AllocateTemporaryClonedMemory[errMsgHd, inputMem, Automatic]
					]
				];
				oCUDAImageMorphology[op][errMsgHd, inputMem, outputMem, IntegerPart[radius], border]
			]],
			$Failed
		];
		
		res = If[FreeQ[res, $Failed],
			If[Head[input] === CUDAMemory,
				outputMem,
				Catch[GetOutputMemory[errMsgHd, outputMem, opts]]
			],
			$Failed
		];
		If[ListQ[runRes],
			GPUDeleteMemory[CUDAMemory, errMsgHd, #]& /@ Flatten[Last[runRes]]
		];
		
		res
	]

oCUDAImageMorphology[op_][errMsgHd_, inputMem_CUDAMemory, outputMem_CUDAMemory, radius_Integer, border:("Fixed" | _Integer | _Real), blockDim_:16] :=
	Module[{res, borderStrategy, borderValue},
		
		SetErrorMessageHead[errMsgHd];

		{borderStrategy, borderValue} = If[border === "Fixed",
			{"Fixed", 0.0},
			{"Constant", 1.0 * border}
		];
		
		res = Switch[op,
			"Dilation",
				cImageProcessingDilation,
			"Erosion",
				cImageProcessingErosion,
			"Opening",
				cImageProcessingOpening,
			"Closing",
				cImageProcessingClosing
		][$SingleOrDoubleLibrary][
			WGLMemoryGetID[inputMem],
			WGLMemoryGetID[outputMem],
			radius,
			borderStrategy,
			borderValue,
			blockDim
		];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]
	
(******************************************************************************)


(******************************************************************************)
(* ::Section:: *)
(* Fourier *)
(******************************************************************************)


Options[CUDAInverseFourier] = Options[CUDAFourier] = Options[iCUDAFourier] = Options[iCUDAInverseFourier] =
		{"Device" -> Automatic, "OutputMemory" -> None}
		

(* ::Subsection:: *)
(* Fourier *)

SyntaxInformation[CUDAFourier] = {"ArgumentsPattern" -> {_, OptionsPattern[]}}
CUDAFourier[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDAFourier, 1, 1], HoldFirst]) :=
	Module[{res = iCUDAFourier[CUDAFourier, args, opts]},
		res /; res =!= $Failed
	]
CUDAFourier[arg_List /; MemberQ[Dimensions[arg], 0], ___] /; (Message[CUDAFourier::zerodims, Dimensions[arg]]; False) := $Failed
CUDAFourier[args___] /; (ArgumentCountQ[CUDAFourier, Length[{args}], 1, 1]; False) := $Failed

iCUDAFourier[errMsgHd_, elem_ /; (!ListQ[elem] && Head[elem] =!= CUDAMemory), opts:OptionsPattern[]] :=
	Return[Message[errMsgHd::zerodims, elem, {0}]; $Failed]
iCUDAFourier[errMsgHd_, {elem:(_Integer | _Real | _Complex)}, opts:OptionsPattern[]] :=
	{elem + 0.0I}
iCUDAFourier[errMsgHd_, mem:CUDAMemory[___, "Dimensions" -> {1}, ___], opts:OptionsPattern[]] :=
	CUDAMemoryLoad[CUDAMemoryGet[mem] + 0.0 I]
iCUDAFourier[errMsgHd_, input_List /; MemberQ[Dimensions[input], 1], opts:OptionsPattern[]] :=
	Return[Message[errMsgHd::onedim, input, Dimensions[input]]; $Failed]
iCUDAFourier[errMsgHd_, mem:CUDAMemory[___, "Dimensions" -> dims:{___, 1, ___}, ___], opts:OptionsPattern[]] :=
	Return[Message[errMsgHd::onedim, mem, dims]; $Failed]
iCUDAFourier[errMsgHd_, input_, opts:OptionsPattern[]] :=
	Module[{res, runRes, inputMem, outputMem},
		res = Check[
			runRes = Reap[Catch[
				InitializeWGL[iCUDAInverseFourier, errMsgHd, opts];
				If[Total[Dimensions[input]] === 0,
					Return[Message[errMsgHd::zerodims]; $Failed]
				];
				outputMem = GetAndCheckOption[errMsgHd, "OutputMemory", {opts}];
				inputMem = AddTemporaryMemory[errMsgHd, input, If[ListQ[input] && GPUTools`Utilities`ArrayType[input] === Integer, Real, Automatic]];
				If[outputMem === None,
					outputMem = If[Head[input] === CUDAMemory,
						AllocateClonedMemory[errMsgHd, inputMem, Complex],
						AllocateTemporaryClonedMemory[errMsgHd, inputMem, Complex]
					],
					If[!TypeRealQ[WGLMemoryGetType[outputMem]],
						Throw[Message[errMsgHd::omemc, outputMem]; $Failed]
					]
				];
				CUFFTExec[errMsgHd, inputMem, outputMem, "Inverse"]
			]],
			$Failed
		];
		
		res = If[FreeQ[res, $Failed],
			If[Head[input] === CUDAMemory,
				outputMem,
				GPUGetMemory[CUDAMemory, errMsgHd, outputMem]
			],
			$Failed
		];

		If[ListQ[runRes],
			GPUDeleteMemory[CUDAMemory, errMsgHd, #]& /@ Flatten[Last[runRes]]
		];
		res
	]
(* ::Subsection:: *)
(* InverseFourier *)

SyntaxInformation[CUDAInverseFourier] = {"ArgumentsPattern" -> {_, OptionsPattern[]}}	
CUDAInverseFourier[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDAInverseFourier, 1, 1], HoldFirst]) :=
	With[{res = iCUDAInverseFourier[CUDAInverseFourier, args, opts]},
		res /; res =!= $Failed
	]
CUDAInverseFourier[arg_List /; MemberQ[Dimensions[arg], 0], ___] /; (Message[CUDAInverseFourier::zerodims, Dimensions[arg]]; False) := $Failed
CUDAInverseFourier[args___] /; (ArgumentCountQ[CUDAInverseFourier, Length[{args}], 1, 1]; False) := $Failed 

iCUDAInverseFourier[errMsgHd_, elem_ /; (!ListQ[elem] && Head[elem] =!= CUDAMemory), opts:OptionsPattern[]] :=
	Return[Message[errMsgHd::zerodims, elem, {0}]; $Failed]
iCUDAInverseFourier[errMsgHd_, {elem:(_Integer | _Real | _Complex)}, opts:OptionsPattern[]] :=
	{elem + 0.0I}
iCUDAInverseFourier[errMsgHd_, mem:CUDAMemory[___, "Dimensions" -> {1}, ___], opts:OptionsPattern[]] :=
	CUDAMemoryLoad[CUDAMemoryGet[mem] + 0.0 I]
iCUDAInverseFourier[errMsgHd_, input_List /; MemberQ[Dimensions[input], 1], opts:OptionsPattern[]] :=
	Return[Message[errMsgHd::onedim, input, Dimensions[input]]; $Failed]
iCUDAInverseFourier[errMsgHd_, mem:CUDAMemory[___, "Dimensions" -> dims:{___, 1, ___}, ___], opts:OptionsPattern[]] :=
	Return[Message[errMsgHd::onedim, mem, dims]; $Failed]
iCUDAInverseFourier[errMsgHd_, input_, opts:OptionsPattern[]] :=
	Module[{res, runRes, inputMem, outputMem},
		res = Check[
			runRes = Reap[Catch[
				InitializeWGL[iCUDAInverseFourier, errMsgHd, opts];
				If[Total[Dimensions[input]] === 0,
					Return[Message[errMsgHd::zerodims]; $Failed]
				];
				outputMem = GetAndCheckOption[errMsgHd, "OutputMemory", {opts}];
				inputMem = Which[
					Head[input] === CUDAMemory,
						input,
					GPUTools`Utilities`ArrayType[input] === Symbol,
						Throw[Message[errMsgHd::notype, input, Symbol]; $Failed],
					True,
						AddTemporaryMemory[errMsgHd, input + 0.0I]
				];

				If[outputMem === None,
					outputMem = If[Head[input] === CUDAMemory,
						AllocateClonedMemory[errMsgHd, inputMem, Complex],
						AllocateTemporaryClonedMemory[errMsgHd, inputMem, Complex]
					],
					If[!TypeRealQ[WGLMemoryGetType[outputMem]],
						Throw[Message[errMsgHd::omemc, outputMem]; $Failed]
					]
				];
				CUFFTExec[errMsgHd, inputMem, outputMem, "Forward"]
			]],
			$Failed
		];
		
		res = If[FreeQ[res, $Failed],
			If[Head[input] === CUDAMemory,
				outputMem,
				GPUGetMemory[CUDAMemory, errMsgHd, outputMem]
			],
			$Failed
		];

		If[ListQ[runRes],
			GPUDeleteMemory[CUDAMemory, errMsgHd, #]& /@ Flatten[Last[runRes]]
		];
		res
	]
	
(******************************************************************************)

CUFFTExec[errMsgHd_, inputMem_, outputMem_, direction_] :=
	Module[{res},
		res = Catch[
			SetErrorMessageHead[errMsgHd];
			iCUFFTExec[WGLMemoryGetID[inputMem], WGLMemoryGetID[outputMem], direction];
		];
		If[res =!= $Failed,
			outputMem,
			$Failed
		]
	]
	
(******************************************************************************)

iCUFFTExec[input_Integer, output_Integer, direction_String] :=
	Module[{res},
		res = cCUFFTExec[$SingleOrDoubleLibrary][input, output, direction];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]

(******************************************************************************)


(******************************************************************************)
(* ::Section:: *)
(* BLAS *)
(******************************************************************************)

(* ::Subsection:: *)
(* Transpose *)


Options[CUDATranspose] = Options[iCUDATranspose] = Options[CUDADot] = Options[iCUDADot] =
	{"Device" -> Automatic, "OutputMemory" -> None}

SyntaxInformation[CUDATranspose] = {"ArgumentsPattern" -> {_, OptionsPattern[]}}	
CUDATranspose[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDATranspose, 1, 1], HoldFirst]) := 
	Module[{res = iCUDATranspose[CUDATranspose, args, opts]},
		res /; MemberQ[{Image, List, CUDAMemory}, Head[res]]
	] 
	
(******************************************************************************)

iCUDATranspose[errMsgHd_, input_, opts:OptionsPattern[]] :=
	Module[{res, runRes, inputMem, outputMem, device},
		res = Check[
			runRes = Reap[Catch[
				InitializeWGL[iCUDATranspose, errMsgHd, opts];
				device = GetAndCheckOption[errMsgHd, "Device", {opts}];
				outputMem = GetAndCheckOption[errMsgHd, "OutputMemory", {opts}];
				
				inputMem = AddTemporaryMemory[errMsgHd, input];
				If[outputMem === None,
					outputMem = If[Head[input] === CUDAMemory,
						AllocateClonedTransposedMemory[errMsgHd, inputMem],
						AllocateTemporaryClonedTransposedMemory[errMsgHd, inputMem]
					]
				];
				oTranspose[errMsgHd, inputMem, outputMem]
			]],
			$Failed
		];
		
		res = If[FreeQ[res, $Failed],
			If[Head[input] === CUDAMemory,
				outputMem,
				Catch[GetOutputMemory[errMsgHd, outputMem, opts]]
			],
			$Failed
		];

		If[ListQ[runRes],
			GPUDeleteMemory[CUDAMemory, errMsgHd, #]& /@ Flatten[Last[runRes]]
		];
		
		res
	]
	
(******************************************************************************)

oTranspose[errMsgHd_, inputMem_, outputMem_] :=
	Module[{res},
		SetErrorMessageHead[errMsgHd];
		res = cTranspose[$SingleOrDoubleLibrary][WGLMemoryGetID[inputMem], WGLMemoryGetID[outputMem]];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]

(******************************************************************************)

CUDAMemoryQ[CUDAMemory[_Integer, ___]] := True
CUDAMemoryQ[___] := False

(******************************************************************************)
ClearAll[isVectorQ]
isVectorQ[x_SparseArray] := ArrayDepth[x] === 1
isVectorQ[x_List] := VectorQ[x, NumberQ]
isVectorQ[x_?CUDAMemoryQ] := Length[WGLMemoryGetDimensions[x]] === 1
isVectorQ[_] := False

ClearAll[isMatrixQ]
isMatrixQ[x_SparseArray] := ArrayDepth[x] === 2
isMatrixQ[x_List] := MatrixQ[x, NumberQ]
isMatrixQ[x_?CUDAMemoryQ] := Length[WGLMemoryGetDimensions[x]] === 2
isMatrixQ[_] := False

(******************************************************************************)

SyntaxInformation[CUDADot] = {"ArgumentsPattern" -> {___, OptionsPattern[]}}
CUDADot[{}, {}, opts:OptionsPattern[]] := 0
CUDADot[{}, opts:OptionsPattern[]] := {}
CUDADot[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDADot, 2, 2], HoldFirst]) :=
	Module[{res = Catch[iCUDADot[CUDADot, args, opts]]},
		res /; MemberQ[{List, CUDAMemory, Integer, Real, Complex}, Head[res]]
	] 

iCUDADot[errMsgHd_, a_, x_, opts:OptionsPattern[]] :=
	Module[{res, runRes, aMem, bMem, outputMem, device, outType},
		Which[
			isVectorQ[a] && isVectorQ[x],
				Return[iCUDADotProduct[errMsgHd, a, x, opts]],
			!(isMatrixQ[a] && (isVectorQ[x] || isMatrixQ[x])),
				Return[Message[errMsgHd::inpt, a, x]; $Failed]
		];
		
		res = Check[
			runRes = Reap[Catch[
				InitializeWGL[iCUDADot, errMsgHd, opts];
				device = GetAndCheckOption[errMsgHd, "Device", {opts}];
				outputMem = GetAndCheckOption[errMsgHd, "OutputMemory", {opts}];
				
				aMem = AddTemporaryMemory[errMsgHd, a];
				bMem = AddTemporaryMemory[errMsgHd, x];
				
				If[isVectorQ[bMem] || isMatrixQ[bMem],
					If[!isMatrixQ[aMem],
						Throw[Message[errMsgHd::nobls, a, x]; $Failed]
					];
					If[outputMem === None,
						outType = WGLMaxType[WGLMemoryGetType[aMem], WGLMemoryGetType[bMem]];
						outputMem = If[Head[a] === CUDAMemory || Head[x] === CUDAMemory,
							If[isMatrixQ[x],
								iCUDAMemoryAllocate[errMsgHd, outType, {WGLMemoryGetDimensions[aMem][[1]], WGLMemoryGetDimensions[bMem][[2]]}],
								iCUDAMemoryAllocate[errMsgHd, outType, First[WGLMemoryGetDimensions[aMem]]]
							],
							If[isMatrixQ[x],
								Sow[iCUDAMemoryAllocate[errMsgHd, outType, {WGLMemoryGetDimensions[aMem][[1]], WGLMemoryGetDimensions[bMem][[2]]}], $TemporaryMemoryFlag],
								Sow[iCUDAMemoryAllocate[errMsgHd, outType, First[WGLMemoryGetDimensions[aMem]]], $TemporaryMemoryFlag]
							]
						]
					];
					If[isVectorQ[bMem],
						oGemv[errMsgHd, aMem, bMem, outputMem],
						oGemm[errMsgHd, aMem, bMem, outputMem]
					],
					(* else *)
					Throw[Message[errMsgHd::nobls, a, x]; $Failed]
				]
			]],
			$Failed
		];

		res = If[FreeQ[res, $Failed],
			If[Head[a] === CUDAMemory || Head[x] === CUDAMemory,
				outputMem,
				Catch[GetOutputMemory[errMsgHd, outputMem, opts]]
			],
			$Failed
		];

		If[ListQ[runRes],
			GPUDeleteMemory[CUDAMemory, errMsgHd, #]& /@ Flatten[Last[runRes]]
		];
		
		res
	]
		
(******************************************************************************)

Options[iCUDADotProduct] = {"Device" -> Automatic}

iCUDADotProduct[errMsgHd_, a_, b_, opts:OptionsPattern[]] :=
	Module[{aMem, bMem, device, res, runRes},
		If[!isVectorQ[a] || !isVectorQ[b],
			Return[Message[errMsgHd::inpt, a, b]; $Failed]
		];
		
		InitializeWGL[iCUDADot, errMsgHd, opts];
		device = GetAndCheckOption[errMsgHd, "Device", {opts}];
		runRes = Catch[Reap[
			aMem = AddTemporaryMemory[errMsgHd, a];
			bMem = AddTemporaryMemory[errMsgHd, b];
			res = xDot[errMsgHd, aMem, bMem]
		]];

		If[ListQ[runRes],
			GPUDeleteMemory[CUDAMemory, errMsgHd, #]& /@ Flatten[Last[runRes]]
		];
		res
	]
		
(******************************************************************************)

Options[CUDAArgMinList] = Options[CUDAArgMaxList] = Options[CUDATotal] = {"Device" -> Automatic}

SyntaxInformation[CUDAArgMaxList] = {"ArgumentsPattern" -> {_, OptionsPattern[]}}	

CUDAArgMaxList[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDAArgMaxList, 1, 1], HoldFirst]) :=
	Module[{res = Catch[iCUDABlas1[AMax][CUDAArgMaxList, args, opts]]},
		res /; NumericQ[res]
	]

SyntaxInformation[CUDAArgMinList] = {"ArgumentsPattern" -> {_, OptionsPattern[]}}	
CUDAArgMinList[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDAArgMinList, 1, 1], HoldFirst]) :=
	Module[{res = Catch[iCUDABlas1[AMin][CUDAArgMinList, args, opts]]},
		res /; NumericQ[res]
	]

SyntaxInformation[CUDATotal] = {"ArgumentsPattern" -> {_, OptionsPattern[]}}
CUDATotal[{}, opts:OptionsPattern[]] := 0
CUDATotal[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDATotal, 1, 1], HoldFirst]) :=
	Module[{res = Catch[iCUDABlas1[ASum][CUDATotal, args, opts]]},
		res /; NumericQ[res]
	]
	
iCUDABlas1[op_][errMsgHd_, a_, opts:OptionsPattern[]] :=
	Module[{mem, device},
		If[!isVectorQ[a],
			Return[Message[errMsgHd::inptvec, a]; $Failed]
		];
		
		Catch[Check[
			InitializeWGL[iCUDADot, errMsgHd, opts];
			device = GetAndCheckOption[errMsgHd, "Device", {opts}];
			
			mem = AddTemporaryMemory[errMsgHd, a];
			op[errMsgHd, mem],
			$Failed
		]]
	]
		

(******************************************************************************)


(******************************************************************************)
(* ::Section:: *)
(* Level 1 *)
(******************************************************************************)

(* ::Subsection:: *)
(* AMax *)

AMax[errMsgHd_, mem_] :=
	Module[{},
		
		SetErrorMessageHead[errMsgHd];
		iAMax[WGLMemoryGetID[mem]]
	]
	
(******************************************************************************)

iAMax[id_Integer] :=
	Module[{res},
		res = cAMax[$SingleOrDoubleLibrary][id];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]

(******************************************************************************)

(* ::Subsection:: *)
(* AMin *)

AMin[errMsgHd_, mem_] :=
	Module[{},
		
		SetErrorMessageHead[errMsgHd];
		iAMin[WGLMemoryGetID[mem]]
	]
	
(******************************************************************************)

iAMin[id_Integer] :=
	Module[{res},
		res = cAMin[$SingleOrDoubleLibrary][id];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]

(******************************************************************************)

(* ::Subsection:: *)
(* ASum *)

ASum[errMsgHd_, mem_] :=
	Module[{},
		
		SetErrorMessageHead[errMsgHd];
		castTo[WGLMemoryGetType[mem], iASum[WGLMemoryGetType[mem], WGLMemoryGetID[mem]]]
	]
	
(******************************************************************************)

iASum[scalarT_String, id_Integer] :=
	Module[{res},
		res = cASum[$SingleOrDoubleLibrary][scalarT, id];
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]

(******************************************************************************)

(* ::Subsection:: *)
(* xDot *)

(* renamed so it does not conflict with Mathematica's Dot *)

xDot[errMsgHd_, x_, y_] :=
	Module[{outType},		
		SetErrorMessageHead[errMsgHd];
		outType = WGLMaxType[WGLMemoryGetType[x], WGLMemoryGetType[y]];
		castTo[outType, iDot[WGLMemoryGetID[x], WGLMemoryGetID[y]]]
	]
	
(******************************************************************************)

iDot[x_Integer, y_Integer] :=
	Module[{res},
		res = cDot[$SingleOrDoubleLibrary][x, y, False]; (* TODO: pass in the conjugate option *)
		If[SuccessQ[res],
			res,
			Throw[$Failed]
		]
	]

(******************************************************************************)

(******************************************************************************)
(* ::Section:: *)
(* Level 2 *)
(******************************************************************************)


oGemv[errMsgHd_, a_, x_, y_] := 
	Module[{res},
		res = Catch[
			oGemv[errMsgHd, "n", 1.0, a, x, 0.0, y]
		];
		y /; res =!= $Failed
	] 
	
(******************************************************************************)

oGemv[errMsgHd_, trans_String, alpha_, a_, x_, beta_, y_] :=
	(
		SetErrorMessageHead[errMsgHd];
		iGemv[toComplex[alpha], trans, WGLMemoryGetID[a], WGLMemoryGetID[x], toComplex[beta], WGLMemoryGetID[y]]
	)
	
(******************************************************************************)

iGemv[alpha_Complex, trans_String, a_Integer, x_Integer, beta_Complex, y_Integer] :=
	Module[{res},
		res = cGemv[$SingleOrDoubleLibrary][alpha, trans, a, x, beta, y];
		If[SuccessQ[res],
			y,
			Throw[$Failed]
		]
	]

(******************************************************************************)
(* ::Section:: *)
(* Level 3 *)
(******************************************************************************)

(* ::Subsection:: *)
(* Gemm *)

oGemm[errMsgHd_, a_, b_, c_] := 
	Module[{res},
		res = Catch[
			oGemm[errMsgHd, "n", "n", 1.0, a, b, 0.0, c]
		];
		c /; res =!= $Failed
	] 
	
(******************************************************************************)

oGemm[errMsgHd_, transa_String, transb_String, alpha_, a_, b_, beta_, c_] :=
	Module[{alphaT, betaT},
		
		{alphaT, betaT} = toTypeName /@ {alpha, beta};

		SetErrorMessageHead[errMsgHd];
		iGemm[transa, transb, alphaT, toComplex[alpha], WGLMemoryGetID[a], WGLMemoryGetID[b], betaT, toComplex[beta], WGLMemoryGetID[c]]
	]
	
(******************************************************************************)

iGemm[transa_String, transb_String, alphaT_String, alpha_Complex, a_Integer, b_Integer, betaT_String, beta_Complex, c_Integer] :=
	Module[{res},
		res = cGemm[$SingleOrDoubleLibrary][transa, transb, alphaT, alpha, a, b, betaT, beta, c];
		If[SuccessQ[res],
			c,
			Throw[$Failed]
		]
	]

(******************************************************************************)
(* TODO need to figure out the precision, and then determine the scalar type in here *)
toTypeName[_Integer] := "Integer"
toTypeName[_Real] := toTypeName[1.0, $SingleOrDoubleLibrary]
toTypeName[_Real, "Single"] := "Float"
toTypeName[_Real, "Double"] := "Double"
toTypeName[_Complex] := toTypeName[1.0 + 1.0I, $SingleOrDoubleLibrary]
toTypeName[_Complex, "Single"] := "ComplexFloat"
toTypeName[_Complex, "Double"] := "ComplexDouble"

(******************************************************************************)

toComplex[x_Complex] := x
toComplex[x:(_Integer | _Real)] := x + 0.0I
toComplex[args__] := Throw[Message[toComplex::cmplx, {args}]; $Failed]

(******************************************************************************)

castTo["Byte" | "Short" | "UnsignedByte" | "Bit16" | "UnsignedBit16" | "Integer" | "Integer32" | "UnsignedInteger32" | "Integer64" | Integer, x_Integer] := x
castTo["Byte" | "Short" | "UnsignedByte" | "Bit16" | "UnsignedBit16" | "Integer" | "Integer32" | "UnsignedInteger32" | "Integer64" | Integer, x_Real] := IntegerPart[x]
castTo["Byte" | "Short" | "UnsignedByte" | "Bit16" | "UnsignedBit16" | "Integer" | "Integer32" | "UnsignedInteger32" | "Integer64" | Integer, x_Complex] := IntegerPart[Re[x]]

castTo["Float" | "Double" | Real, x_Integer] := 1.0*x
castTo["Float" | "Double" | Real, x_Real] := x
castTo["Float" | "Double" | Real, x_Complex] := Re[x]

castTo["ComplexFloat" | "ComplexDouble" | Complex, x_Integer] := x + 0.0 I
castTo["ComplexFloat" | "ComplexDouble" | Complex, x_Real] := x + 0.0 I
castTo["ComplexFloat" | "ComplexDouble" | Complex, x_Complex] := x

(******************************************************************************)


(******************************************************************************)


GetAndCheckOption[head_, opt_, opts_] := GetAndCheckOption[head, head, opt, opts] 

GetAndCheckOption[head_, errMsgHd_, n:Padding, opts_] :=
	With[{opt = OptionValue[head, opts, n]},
		If[opt === "Fixed" || NumericQ[opt],
			opt,
			Throw[Message[errMsgHd::imgpadm, opt]; $Failed]
		]
	]
	
GetAndCheckOption[head_, errMsgHd_, n:"OutputMemory", opts_] :=
	With[{opt = OptionValue[head, opts, n]},
		If[opt === None || Head[opt] === CUDAMemory,
			opt,
			Throw[Message[errMsgHd::omem, opt]; $Failed]
		]
	]
	
(******************************************************************************)

AddTemporaryMemory[_, mem_CUDAMemory, ___] := mem
AddTemporaryMemory[errMsgHd_, input_SparseArray, type_:Automatic] :=
	iAddTemporaryMemory[errMsgHd, Normal[input], type]
AddTemporaryMemory[errMsgHd_, {}, ___] :=
	(Message[errMsgHd::empty]; $Failed)  
AddTemporaryMemory[errMsgHd_, input_List, type_:Automatic] :=
	iAddTemporaryMemory[errMsgHd, input, type] 
AddTemporaryMemory[errMsgHd_, input_?Image`PossibleImageQ, type_:Automatic] :=
	iAddTemporaryMemory[errMsgHd, input, type] 
iAddTemporaryMemory[errMsgHd_, input_, type_:Automatic] := 
	With[{mem = Check[iCUDAMemoryLoad[errMsgHd, input, type, "Device" -> $CUDADevice], $Failed]},
		Which[
			mem === $Failed,
				Throw[$Failed],
			Head[mem] =!= CUDAMemory,
				Throw[Message[errMsgHd::unbladd, input]; $Failed],
			True,
				Sow[mem , $TemporaryMemoryFlag]
		]
	]
AddTemporaryMemory[errMsgHd_, x_, ___] := Throw[Message[errMsgHd::noinpt, x]; $Failed]
(******************************************************************************)

AllocateClonedMemory[errMsgHd_, mem_CUDAMemory, type_:Automatic] :=
	iAllocateClonedMemory[errMsgHd, mem, type, False]

AllocateClonedTransposedMemory[errMsgHd_, mem_CUDAMemory, type_:Automatic] :=
	iAllocateClonedMemory[errMsgHd, mem, type, True]
	
iAllocateClonedMemory[errMsgHd_, mem0_, type0_, transposed_] :=
	Module[{mem, type, dims, targetPrecision = Automatic},
		type = If[type0 === Automatic,
			WGLMemoryGetType[mem0],
			targetPrecision = If[TypeDoublePrecisionQ[type0],
				"Double",
				"Single"
			];
			type0
		];
		dims = If[transposed,
			Reverse[WGLMemoryGetDimensions[mem0]],
			WGLMemoryGetDimensions[mem0]
		];
		mem = Check[iCUDAMemoryAllocate[errMsgHd, type, dims, "TargetPrecision" -> targetPrecision, "Device" -> $CUDADevice], $Failed];
		Which[
			mem === $Failed,
				Throw[$Failed],
			Head[mem] =!= CUDAMemory,
				Throw[Message[errMsgHd::unbladd, mem0]; $Failed]
		];
		CUDAMemory[
			WGLMemoryGetID[mem],
			"Type" -> WGLMemoryGetType[mem],
			"Dimensions" -> WGLMemoryGetDimensions[mem],
			"ByteCount" -> WGLMemoryGetByteCount[mem],
			"Residence" -> WGLMemoryGetResidence[mem],
			"Sharing" -> WGLMemoryGetSharing[mem],
			"Unique" -> WGLMemoryGetUniqueQ[mem],
			"Platform" -> WGLMemoryGetPlatform[mem],
			"Device" -> WGLMemoryGetDevice[mem],
			"MathematicaType" -> WGLMemoryGetMathematicaType[mem0],
			"TypeInfromation" -> WGLMemoryGetTypeInfromation[mem0]
		]
	]

AllocateTemporaryClonedMemory[errMsgHd_, mem_CUDAMemory, type_:Automatic] :=
	Sow[AllocateClonedMemory[errMsgHd, mem, type], $TemporaryMemoryFlag]

AllocateTemporaryClonedTransposedMemory[errMsgHd_, mem_CUDAMemory, type_:Automatic] :=
	Sow[AllocateClonedTransposedMemory[errMsgHd, mem, type], $TemporaryMemoryFlag]
		


AllocateClonedImageMemory[errMsgHd_, inputMem_, input_] :=
	AllocateClonedMemory[errMsgHd, inputMem, If[Image`PossibleImageQ[input], Real, Automatic]]
AllocateTemporaryClonedImageMemory[errMsgHd_, inputMem_, input_] :=
	AllocateTemporaryClonedMemory[errMsgHd, inputMem, If[Image`PossibleImageQ[input], Real, Automatic]]
		
(******************************************************************************)

GetOutputMemory[errMsgHd_, mem_, opts:OptionsPattern[]] :=
	With[{outputMem = GetAndCheckOption[errMsgHd, "OutputMemory", {opts}]},
		If[outputMem === None,
			CUDAMemoryGet[mem],
			mem
		]
	]
(******************************************************************************)
(******************************************************************************)
(******************************************************************************)

(* ::Section:: *)
(* SymbolicGPU *)

SymbolicCUDAFunction[args___] := GPUKernelFunction["CUDA", args]
SymbolicCUDAThreadIndex[args___] := GPUKernelThreadIndex["CUDA", args]
SymbolicCUDABlockIndex[args___] := GPUKernelBlockIndex["CUDA", args]
SymbolicCUDABlockDimension[args___] := GPUKernelBlockDimension["CUDA", args]
SymbolicCUDACalculateKernelIndex[args___] := GPUCalculateKernelIndex["CUDA", args]
SymbolicCUDADeclareIndexBlock[args___] := GPUDeclareIndexBlock["CUDA", args]
SymbolicCUDAKernelIndex[args___] := GPUKernelIndex["CUDA", args]
	
(******************************************************************************)
(******************************************************************************)
(******************************************************************************)


(* ::Section:: *)
(* Examples *)

(******************************************************************************)
(******************************************************************************)
(******************************************************************************)


(* ::Section:: *)
(* Fluid Dynamics *)

If[CUDALink`Internal`ExamplesIntializedQ === Unevaluated[CUDALink`Internal`ExamplesIntializedQ],
	CUDALink`Internal`ExamplesIntializedQ = False
]
CUDAFluidDynamics`$DomainSize = 128
CUDAFluidDynamics`$DefaultColorScheme = "DarkBands"

CUDALink`Internal`ExamplesInitialize[] := 
	Module[{initFun, libraryFuncs},
		If[CUDALink`Internal`ExamplesIntializedQ =!= False,
			If[CUDALink`Internal`ExamplesIntializedQ =!= True,
				Message[CUDALink::initlib, CUDALink`Internal`$ExamplesLibraryPath]
			];
			Return[CUDALink`Internal`ExamplesIntializedQ]
		];
		If[CUDALink`Internal`$ExamplesLibraryPath === $Failed || CUDAQ[] === False,
			CUDALink`Internal`ExamplesIntializedQ = $Failed;
			Return[Message[CUDALink::initlib, CUDALink`Internal`$ExamplesLibraryPath]; $Failed]
		];
		
		initFun = Catch[ShareWolframGPULibraryData[CUDALink`Internal`$ExamplesLibraryPath]];
		If[initFun === $Failed,
			Return[Message[CUDALink::initwgl, CUDALink`Internal`$ExamplesLibraryPath]; $Failed]
		];
		
		libraryFuncs = GPUTools`Internal`LibraryFunctionSafeLoad[{
			{cInitFluidDynamics, CUDALink`Internal`$ExamplesLibraryPath, "oFluidDynamics_Initialize",
				{_Integer},
                "Void"
			},
			{cNewParticlesFrame, CUDALink`Internal`$ExamplesLibraryPath, "oFluidDynamics_StepAsParticles",
				{{_Real, 2, "Shared"}, Integer},
			    {_Real, 2, "Shared"}
			},
			{cNewPixelFrame, CUDALink`Internal`$ExamplesLibraryPath, "oFluidDynamics_StepAsPixels",
				{{_Real, 2, "Shared"}, Integer},
			    {_Real, 2, "Shared"}
			},
			{cMouseMovementFluidDynamics, CUDALink`Internal`$ExamplesLibraryPath, "oFluidDynamics_MouseMovement",
				{
                    _Integer, (* button *)
                    _Integer, (* state *)
                    _Integer, (* x *)
                    _Integer  (* y *)
                },
                "Void"
			},
			{cMotionFluidDynamics, CUDALink`Internal`$ExamplesLibraryPath, "oFluidDynamics_Motion",
				{
                    _Integer, (* x *)
                    _Integer  (* y *)
                },
                "Void"
			},
			{cResetParticles, CUDALink`Internal`$ExamplesLibraryPath, "oFluidDynamics_ResetParticles",
				{},
                "Void"
			},
			{cSetParticles, CUDALink`Internal`$ExamplesLibraryPath, "oFluidDynamics_SetParticles",
				{{_Real, 2, "Shared"}},
                "Void"
			},
			
			{cSetTimeDelta, CUDALink`Internal`$ExamplesLibraryPath, "oFluidDynamics_SetTimeDelta",
				{_Real},
                "Void"
			},
			{cSetViscosity, CUDALink`Internal`$ExamplesLibraryPath, "oFluidDynamics_SetViscosity",
				{_Real},
                "Void"
			},
			{cSetForce, CUDALink`Internal`$ExamplesLibraryPath, "oFluidDynamics_SetForce",
				{_Real},
                "Void"
			},
			{cSetForceRadius, CUDALink`Internal`$ExamplesLibraryPath, "oFluidDynamics_SetForceRadius",
				{_Integer},
                "Void"
			},
			{cInitVolumetricRender, CUDALink`Internal`$ExamplesLibraryPath, "oVolumetricRendering_Initialize",
				{
                    {_Integer, _, "Shared"}, (* volume Data *)
                    {_Integer, _, "Shared"}, (* output tensor *)
                    {_Real,    2, "Shared"}  (* output tensor *)
                },
                "Void"
			},
			{cNewVolumeFrame, CUDALink`Internal`$ExamplesLibraryPath, "oVolumetricRendering_NewFrame",
				{},
			    "Void"
			},
			{cMouseMovementVolumetricRender, CUDALink`Internal`$ExamplesLibraryPath, "oVolumetricRendering_MouseMovement",
				{
                    _Integer, (* button *)
                    _Integer, (* state *)
                    _Integer, (* x *)
                    _Integer  (* y *)
                },
                "Void"
			},
			{cMotionVolumetricRender, CUDALink`Internal`$ExamplesLibraryPath, "oVolumetricRendering_Motion",
				{
                    _Integer, (* x *)
                    _Integer  (* y *)
                },
                "Void"
			},
			{cSetVolumeBrightness, CUDALink`Internal`$ExamplesLibraryPath, "oVolumetricRendering_SetBrightness",
				{
                    _Real (* brightness *)
                },
                "Void"
			},
			{cSetVolumeTransferScale, CUDALink`Internal`$ExamplesLibraryPath, "oVolumetricRendering_SetTransferScale",
				{
                    _Real (* transfer scale *)
                },
                "Void"
			},
			{cSetVolumeTransferOffset, CUDALink`Internal`$ExamplesLibraryPath, "oVolumetricRendering_SetTransferOffset",
				{
                    _Real (* transfer offset *)
                },
                "Void"
			},
			{cSetVolumeDensity, CUDALink`Internal`$ExamplesLibraryPath, "oVolumetricRendering_SetDensity",
				{
                    _Real (* density *)
                },
                "Void"
			},
			{cSetVolumeTransferFunction, CUDALink`Internal`$ExamplesLibraryPath, "oVolumetricRendering_SetTransferFunction",
				{
                    {_Real, 2, "Shared"} (* Transfer function *)
                },
                "Void"
			}
		}];
		If[libraryFuncs === $Failed || MemberQ[libraryFuncs, $Failed],
			Throw[Message[CUDALink::initlib, CUDALink`Internal`$ExamplesLibraryPath]; CUDALink`Internal`ExamplesIntializedQ = $Failed],
			CUDALink`Internal`ExamplesIntializedQ = True
		]
	]
	
CUDALink`Internal`ExamplesInitialize[device_] :=
	Module[{res},
		res = CUDALink`Internal`ExamplesInitialize[];
		res = GPUAddMemory[CUDAMemory, CUDALink, {1}, "Device" -> device];
		True
	]

InitializeFluid[] :=  InitializeFluid[] =
    If[GPUTools`Utilities`ValidLibraryFunctionReturnQ[Check[cInitFluidDynamics[128], $Failed]],
    	True,
    	False
    ]
	
InitializeExample[device_] :=
	Module[{},
		CUDALink`Internal`ExamplesInitialize[device];
		If[CUDALink`Internal`ExamplesIntializedQ,
			InitializeFluid[];
			CUDALink`Internal`CUDAFluidDynamicsIntializeValue,
			True
		]
	]
InitializeExample[___] := Throw[Message[CUDALink::init]; $Failed]

ParticlesQ[x_] := Head[x] == CUDAFluidDynamics`ParticlesObject

GetParticlesData[CUDAFluidDynamics`ParticlesObject[data_, ___]] := data
GetParticlesDevice[CUDAFluidDynamics`ParticlesObject[_, device_, ___]] := device

UpdateParticles[x_CUDAFluidDynamics`ParticlesObject, dim_Integer] /; CUDALink`Internal`ExamplesIntializedQ :=
    Module[{res = Check[cNewParticlesFrame[GetParticlesData[x], dim], $Failed]},
    	If[GPUTools`Utilities`ValidLibraryFunctionReturnQ[res],
    		res,
    		$Failed
    	]
    ]
UpdateParticles[___] := Message[UpdateParticles::invpobj]

IntegerMousePosition[] :=
	IntegerPart[1024*#]& /@ MousePosition["Graphics"]

FluidDynamicsMouseButtonEvent[{}, state_] := 
	FluidDynamicsMouseButtonEvent[1, state]
FluidDynamicsMouseButtonEvent[but_List, state_] :=
	FluidDynamicsMouseButtonEvent[First@but, state]
FluidDynamicsMouseButtonEvent[but_Integer, state_] :=
	With[{pos=IntegerMousePosition[]},
		cMouseMovementFluidDynamics[but-1, state, IntegerPart@pos[[1]], IntegerPart@pos[[2]]]
	];
FluidDynamicsMouseButtonEvent[{}, state_, x_, y_] := 
	FluidDynamicsMouseButtonEvent[1, state, x, y]
FluidDynamicsMouseButtonEvent[but_List, state_, x_, y_] :=
	FluidDynamicsMouseButtonEvent[First@but, state, x, y]
FluidDynamicsMouseButtonEvent[but_, state_, x_, y_] :=
	FluidDynamicsMouseButtonEvent[IntegerPart[but], IntegerPart[state], IntegerPart[x], IntegerPart[y]]
FluidDynamicsMouseButtonEvent[but_Integer, state_Integer, x_Integer, y_Integer] /; CUDALink`Internal`ExamplesIntializedQ === True :=
	cMouseMovementFluidDynamics[but, state, x, y]

 
FluidDynamicsMouseMotionEvent[] /; CUDALink`Internal`ExamplesIntializedQ === True :=
	With[{pos=IntegerMousePosition[]},
		FluidDynamicsMouseMotionEvent[pos[[1]],pos[[2]]]
	]
FluidDynamicsMouseMotionEvent[x_, y_] /; CUDALink`Internal`ExamplesIntializedQ === True :=
	FluidDynamicsMouseMotionEvent[IntegerPart[x], IntegerPart[y]]
FluidDynamicsMouseMotionEvent[x_Integer, y_Integer] /; CUDALink`Internal`ExamplesIntializedQ === True :=
	cMotionFluidDynamics[x, y]


GetParticleColors[colorScheme_String] := GetParticleColors[colorScheme] = Map[ColorData[colorScheme], Range[0, 1, 1/(CUDAFluidDynamics`$DomainSize^2 - 1)]] //. RGBColor -> List;

SyntaxInformation[CUDAFluidDynamics] = {"ArgumentsPattern" -> {OptionsPattern[]}}
Options[CUDAFluidDynamics] = Options[iCUDAFluidDynamics] = Sort@{ImageSize -> 360, "NumberOfParticles" -> Automatic, "Device" -> Automatic}
CUDAFluidDynamics[args___, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDAFluidDynamics, 0, 0], HoldFirst]) :=
	With[{res = Catch[iCUDAFluidDynamics[args, opts]]},
		res /; res =!= $Failed
	]
iCUDAFluidDynamics[opts:OptionsPattern[]] :=
	Module[{numParts, device, particlesObject, force = 5.8, viscosity = 0.0025, timeScale = 0.35,  forceRadius = 4, imageSize},
		
		InitializeWGL[iCUDAFluidDynamics, CUDAFluidDynamics, opts];
		device = GetAndCheckOption[CUDAFluidDynamics, "Device", {opts}];
		
		numParts = OptionValue[CUDAFluidDynamics, {opts}, "NumberOfParticles"];
		If[numParts === Automatic,
			numParts = CUDAFluidDynamics`$DomainSize,
			If[!IntegerQ[numParts] && Positive[numParts],
				Message[CUDAFluidDynamics::invnumpts, numParts];
				Return[$Failed],
				numParts = IntegerPart[Sqrt[numParts]];
			]
		];
    	
		imageSize = OptionValue[CUDAFluidDynamics, {opts}, ImageSize];
		If[!(NumberQ[imageSize] && Im[imageSize] === 0 && imageSize > 0),
			Message[CUDAFluidDynamics::invimgsiz, imageSize];
			Return[$Failed]
		];
		
		InitializeExample[device];
		
		CUDAFluidDynamics`$DomainSize = numParts;
    	particlesObject = CUDAFluidDynamics`ParticlesObject[ConstantArray[0.0, {CUDAFluidDynamics`$DomainSize^2, 2}], device, CUDAFluidDynamics`$DomainSize];
        
    	DynamicModule[{particleColors = GetParticleColors[CUDAFluidDynamics`$DefaultColorScheme], colorScheme = CUDAFluidDynamics`$DefaultColorScheme},
	        Deploy@Panel[
				Column[{
					EventHandler[
						Graphics[{AbsolutePointSize[1], Point[Dynamic[Refresh[UpdateParticles[particlesObject, numParts], UpdateInterval -> 0]], VertexColors->Dynamic[particleColors]]}, ImageSize -> {Full, imageSize}, Background->Black],
							{
								"MouseDown" :>
									With[{mousePos = IntegerMousePosition[], btn = CurrentValue["MouseButtons"]},
										If[ListQ[mousePos] && ListQ[btn],
											FluidDynamicsMouseButtonEvent[btn, 1, mousePos[[1]], mousePos[[2]]]
										]
									],
								"MouseUp" :>
									With[{mousePos = IntegerMousePosition[], btn = CurrentValue["MouseButtons"]},
										If[ListQ[mousePos] && ListQ[btn],
											FluidDynamicsMouseButtonEvent[btn, 0, mousePos[[1]], mousePos[[2]]]
										]
									],
								"MouseDragged" :>
									With[{mousePos = IntegerMousePosition[]},
										If[ListQ[mousePos],
											FluidDynamicsMouseMotionEvent[mousePos[[1]], mousePos[[2]]];
										]
									]
							}
					],
					OpenerView[{"Options",
						Column[{
							Row[{
								"Color Scheme:",
									Spacer[10],
								PopupMenu[Dynamic[colorScheme, (colorScheme = #1; particleColors=GetParticleColors[#1])&], ColorData["Gradients"]],
									Spacer[20],
								Button["Reset Particles", cResetParticles[]]
							}],
							Row[{"Time Scale:   	 ", Slider[Dynamic[timeScale, (cSetTimeDelta[#1]; timeScale = #1)&], {0.01, 0.4, 0.01}]}],
							Row[{"Viscosity:    	 ", Slider[Dynamic[viscosity, (cSetViscosity[#1]; viscosity = #1)&], {0.001, 0.2, 0.001}]}],
							Row[{"Force:        	 ", Slider[Dynamic[force, (cSetForce[#1]; force = #1)&], {0.1, 10, 0.01}]}],
							Row[{"Force Radius:      ", Slider[Dynamic[forceRadius, (cSetForceRadius[#1]; forceRadius = #1)&], {1, 10, 1}]}]
						}]
					}]
				}]
	        ]
		]
	]

(******************************************************************************)
(******************************************************************************)
(******************************************************************************)


CUDAVolumetricRender`$DefaultWidth = 512
CUDAVolumetricRender`$DefaultHeight = 512
CUDAVolumetricRender`$VolumetricExampleDataPath = FileNameJoin[{$CUDALinkPath, "ExampleData"}];
CUDAVolumetricRender`$DefaultTransferFunction =
		Developer`ToPackedArray[
			{{0.76321, 0.364813, 0.209552, 0.},
			 {0.557758, 0.577019, 0.537758, 0.065},
			 {0.530678, 0.711261, 0.720946, 0.015},
			 {0.599635, 0.804093, 0.825361, 0.785},
			 {0.704279, 0.87076, 0.878447, 0.71},
			 {0.801399, 0.914065, 0.885946, 0.5},
			 {0.862031, 0.931614, 0.845355, 0.6},
			 {0.870519, 0.921438, 0.75574, 0.7},
			 {0.825522, 0.886011, 0.623904, 0.495},
			 {0.742983, 0.834634, 0.466914, 0.9},
			 {0.661036, 0.784207, 0.310994, 1.}
		    }
		];

CUDAVolumetricRender`$DefaultDensity = 0.765                                
CUDAVolumetricRender`$DefaultBrightness = 1.35
CUDAVolumetricRender`$DefaultTransferScale = 1.0
CUDAVolumetricRender`$DefaultTransferOffset = 0.43

InitializeVolume[device:(Automatic | _Integer), data_List, tf_List, width_Integer, height_Integer] := 
    Module[{VolumeObjectData},
        VolumeObjectData = ConstantArray[1, {height, width, 4}];
        If[GPUTools`Utilities`ValidLibraryFunctionReturnQ[Check[cInitVolumetricRender[data, VolumeObjectData, tf], $Failed]],
        	{CUDAVolumetricRender`VolumeObject[1, VolumeObjectData, device, width, height], True},
        	Throw[$Failed]
        ] 
   ]
GetAndCheckOption[head_, errMsgHd_, n:"Width", opts_] :=
	Module[{width = OptionValue[head, {opts}, n]},
		If[IntegerQ[width] && Positive[width],
			width,
			Throw[Message[errMsgHd::invwidth, width]; $Failed]
		]
	]
GetAndCheckOption[head_, errMsgHd_, n:"Height", opts_] :=
	Module[{height = OptionValue[head, {opts}, n]},
		If[IntegerQ[height] && Positive[height],
			height,
			Throw[Message[errMsgHd::invheight, height]; $Failed]
		]
	]
	
InitializeExample[device_, volumeData_, tf_, width_, height_] := 
	Module[{},
		CUDALink`Internal`ExamplesInitialize[device];
		If[CUDALink`Internal`ExamplesIntializedQ,
			With[{res = InitializeVolume[device, volumeData, tf, width, height]},
				If[Last[res],
					First[res],
					Throw[$Failed]
				]
			],
			Throw[Message[CUDALink::init]; $Failed]
		]
	]
InitializeExample[___] := Throw[Message[CUDALink::init]; $Failed]

VolumeQ[x_] := Head[x] == CUDAVolumetricRender`VolumeObject

GetVolumeId[CUDAVolumetricRender`VolumeObject[id_, ___]] := id
GetVolumeData[CUDAVolumetricRender`VolumeObject[_, data_, ___]] := data
GetVolumeDevice[CUDAVolumetricRender`VolumeObject[_, _, device_, ___]] := device
GetVolumeWidth[CUDAVolumetricRender`VolumeObject[_, _, _, width_, ___]] := width
GetVolumeHeight[CUDAVolumetricRender`VolumeObject[_, _, _, _, height_, ___]] := height

toImage[CUDAVolumetricRender`VolumeObject[_, data_, _, width_, height_]] := Image[data, "Byte", ImageSize->{width, height}]
toImageData[x_] := GetVolumeData[x]

UpdateVolume[x_CUDAVolumetricRender`VolumeObject] :=
    (
    	cNewVolumeFrame[];
    	toImage[x]
    )

VolumetricRenderMouseButtonEvent[{}, _] := None
VolumetricRenderMouseButtonEvent[but_List, state_] :=
	VolumetricRenderMouseButtonEvent[First@but, state]
VolumetricRenderMouseButtonEvent[but_Integer, state_] /; CUDALink`Internal`ExamplesIntializedQ === True :=
	With[{pos=MousePosition[]},
		cMouseMovementVolumetricRender[but, state, IntegerPart@pos[[1]], IntegerPart@pos[[2]]]
	];
VolumetricRenderMouseButtonEvent[{}, _, _, _] /; CUDALink`Internal`ExamplesIntializedQ === True := None
VolumetricRenderMouseButtonEvent[but_List, state_, x_, y_] /; CUDALink`Internal`ExamplesIntializedQ === True :=
	VolumetricRenderMouseButtonEvent[First@but, state, x, y]
VolumetricRenderMouseButtonEvent[but_, state_, x_, y_] /; CUDALink`Internal`ExamplesIntializedQ === True:=
	VolumetricRenderMouseButtonEvent[IntegerPart[but], IntegerPart[state], IntegerPart[x], IntegerPart[y]]
VolumetricRenderMouseButtonEvent[but_Integer, state_Integer, x_Integer, y_Integer] /; CUDALink`Internal`ExamplesIntializedQ === True:=
	cMouseMovementVolumetricRender[but, state, x, y]

 
VolumetricRenderMouseMotionEvent[]  /; CUDALink`Internal`ExamplesIntializedQ === True :=
	With[{pos=MousePosition[]},
		VolumetricRenderMouseMotionEvent[pos[[1]],pos[[2]]]
	]
VolumetricRenderMouseMotionEvent[x_, y_] /; CUDALink`Internal`ExamplesIntializedQ === True :=
	VolumetricRenderMouseMotionEvent[IntegerPart[x], IntegerPart[y]]
VolumetricRenderMouseMotionEvent[x_Integer, y_Integer] /; CUDALink`Internal`ExamplesIntializedQ === True :=
	cMotionVolumetricRender[x, y]


SetBrightness[b_] := 
	cSetVolumeBrightness[b]
SetTransferScale[ts_Real] := 
	cSetVolumeTransferScale[ts]
SetTransferOffset[to_Real] := 
	cSetVolumeTransferOffset[to]
SetDensity[dens_Real] :=
	cSetVolumeDensity[dens]
SetTransferFunction[tf0_List] := 
	Module[{tf = Developer`ToPackedArray[tf0]},
		If[Developer`PackedArrayQ[tf],
			cSetVolumeTransferFunction[tf]
		]
	]
		
(* for stented height = 512, depth = 174 *)
CUDAVolumetricDataRead[pathexp_?FileExistsQ, height_Integer?Positive, depth_Integer?Positive] :=
	Module[{volumeData, strm,path},
		path = getFile[pathexp];
		strm = OpenRead[path, BinaryFormat -> True];
		volumeData = Partition[Partition[BinaryReadList[strm, "UnsignedInteger8"], depth], height];
		Close[strm];
		volumeData
	]

Options[CUDAVolumetricRender] := Sort@{"Device" -> Automatic, "Width" -> CUDAVolumetricRender`$DefaultWidth, "Height" -> CUDAVolumetricRender`$DefaultHeight}
SyntaxInformation[CUDAVolumetricRender] = {"ArgumentsPattern" -> {_, OptionsPattern[]}}	

CUDAVolumetricRender[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDAVolumetricRender, 1, 1], HoldFirst]) :=
	With[{res = Catch[iCUDAVolumetricRender[args, opts]]},
		res /; res =!= $Failed
	]
iCUDAVolumetricRender[volumeData_, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDAVolumetricRender, 1, 1], HoldFirst]) :=
	Module[{device, width, height, volumeRender},
		
		If[!ListQ[volumeData],
			Throw[Message[CUDAVolumetricRender::invvol, volumeData]; $Failed]
		];
		device = GetAndCheckOption[CUDAVolumetricRender, "Device", {opts}];
		width = GetAndCheckOption[CUDAVolumetricRender, "Width", {opts}];
		height = GetAndCheckOption[CUDAVolumetricRender, "Height", {opts}];
		
		volumeRender = InitializeExample[device, volumeData, CUDAVolumetricRender`$DefaultTransferFunction, width, height];
        If[CUDALink`Internal`ExamplesIntializedQ =!= True,
        	Throw[Message[CUDAVolumetricRender::init]; $Failed]
        ];
        
    	DynamicModule[{transferOffset, transferScale, brightness, density, colorScheme = "IslandColors"},
    		density = CUDAVolumetricRender`$DefaultDensity;
	        brightness = CUDAVolumetricRender`$DefaultBrightness;
			transferScale = CUDAVolumetricRender`$DefaultTransferScale;
			transferOffset = CUDAVolumetricRender`$DefaultTransferOffset;
			
			SetBrightness[brightness];
	        SetTransferScale[transferScale];
	        SetTransferOffset[transferOffset];
	        SetDensity[density];

	        Deploy[Panel[
				Column[{
	        		EventHandler[
						Dynamic[Refresh[UpdateVolume[volumeRender], UpdateInterval -> 0.05]],
							{
								"MouseDown" :>
									With[{mousePos = MousePosition["Graphics"], btn = CurrentValue["MouseButtons"], ctrl = CurrentValue["ControlKey"]},
										If[ctrl,
											VolumetricRenderMouseButtonEvent[3, 1, mousePos[[1]], mousePos[[2]]],
											VolumetricRenderMouseButtonEvent[btn, 1, mousePos[[1]], mousePos[[2]]]
										];
									],
								"MouseUp" :>
									With[{mousePos = MousePosition["Graphics"], btn = CurrentValue["MouseButtons"], ctrl = CurrentValue["ControlKey"]},
										If[mousePos != None,
											If[ctrl,
												VolumetricRenderMouseButtonEvent[3, 0, mousePos[[1]], mousePos[[2]]],
												VolumetricRenderMouseButtonEvent[btn, 0, mousePos[[1]], mousePos[[2]]]
											];
										];
									],
								"MouseDragged" :>
									With[{mousePos = MousePosition["Graphics"]},
										If[mousePos =!= None,
											VolumetricRenderMouseMotionEvent[mousePos[[1]], mousePos[[2]]];
										];
									]
							}
					],
					OpenerView[{"Options",
						Column[{
							Row[{"Transfer Offset:   ", Slider[Dynamic[transferOffset, (SetTransferOffset[#1]; transferOffset = #1)&], {0, 1, 0.01}]}],
							Row[{"Transfer Scale:    ", Slider[Dynamic[transferScale, (SetTransferScale[#1]; transferScale = #1)&], {0, 5, 0.1}]}],
							Row[{"Brightness:        ", Slider[Dynamic[brightness, (SetBrightness[#1]; brightness = #1)&], {0, 7, 0.01}]}],
							Row[{"Density:           ", Slider[Dynamic[density, (SetDensity[#1]; density = #1)&], {0, 1, 0.001}]}],
							
							Spacer[20],
							
							PopupMenu[Dynamic[colorScheme], ColorData["Gradients"]],
							DynamicModule[{tt = Table[{k, k}, {k, 0, 1, 0.1}], tf},
		 						Manipulate[
		  							tt[[All, 1]] = Range[0, 1, 0.1]; 
		  							tf = N[Apply[Append[List @@ ColorData[colorScheme][#1], #2] &, tt, {1}]];
		  							Refresh[SetTransferFunction[tf], TrackedSymbols -> {tt}];
		  							Graphics[{Raster[{Table[List @@ ColorData[colorScheme][k], {k, 0, 1, 0.01}]}, {{0, 0}, {1, 1}}], {Black, Line[tt]}}, AspectRatio -> 1/2],
		  							{tt, {0, 0}, {1, 1}, Locator, LocatorAutoCreate -> {All, {2, 16}}},  Deployed -> True, LocalizeVariables -> False
		  						]
		 					]
						}]
					}]
				}]
	        ]
		]]
	]

(******************************************************************************)
(******************************************************************************)
(******************************************************************************)

(* ::Section:: *)
(* CUDASort *)
Options[CUDASort] = Options[iCUDASort] = {"Device" -> Automatic}
SyntaxInformation[CUDASort] = {"ArgumentsPattern" -> {_, _., OptionsPattern[]}}

CUDASort[{}, opts:OptionsPattern[]] := {}
CUDASort[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDASort, 1, 2], HoldFirst]) :=
	With[{res = iCUDASort[args, opts]},
		res /; res =!= $Failed
	]
iCUDASort[elem_, opts:OptionsPattern[]] :=
	Catch[iCUDASort[CUDASort, elem, Less, opts]]
iCUDASort[elem_, op:(Less | Greater), opts:OptionsPattern[]] :=
	Catch[iCUDASort[CUDASort, elem, op, opts]]
iCUDASort[elem_, op_, ___] :=
	Return[Message[CUDASort::op, op]; $Failed]
iCUDASort[errMsgHd_, elem_, op_, opts:OptionsPattern[]] :=
	Module[{device, dims, mem, runRes, res, cudaSortFunction},
		
		If[!ListQ[elem] && Head[elem] =!= CUDAMemory,
			Throw[Message[errMsgHd::input, elem]; $Failed]
		];
		
		InitializeWGL[iCUDASort, errMsgHd, opts];
		device = GetAndCheckOption[errMsgHd, "Device", {opts}];
		
		dims = If[Head[elem] === CUDAMemory,
			WGLMemoryGetDimensions[elem],
			Dimensions[elem]
		];
		
		If[Length[dims] =!= 1 && !(Head[elem] === CUDAMemory && Length[dims] == 2 && Last[dims] == ResolveElementWidth[WGLMemoryGetType[elem]]),
			Throw[Message[errMsgHd::novec, dims]; $Failed]
		];  
		
		cudaSortFunction = CUDAFunctionLoad[{CUDALink`Internal`$ExamplesLibraryPath}, "oCUDASort", {{_Integer, "InputOutput"}, "UTF8String"}, 256, "Device" -> device];
		If[Head[cudaSortFunction] =!= CUDAFunction,
			Throw[$Failed]
		];
		
		runRes = Catch[Reap[
			mem = AddTemporaryMemory[errMsgHd, elem];
			res = cudaSortFunction[mem, ToString[op], 256]
		]];

		res = If[FreeQ[res, $Failed] && FreeQ[runRes, $Failed],
			If[Head[elem] === CUDAMemory,
				mem,
				CUDAMemoryGet[mem]
			],
			$Failed
		];
		
		If[ListQ[runRes],
			GPUDeleteMemory[CUDAMemory, errMsgHd, #]& /@ Flatten[Last[runRes]]
		];
		
		res
	]
	
(******************************************************************************)
(******************************************************************************)
(******************************************************************************)

(* ::Section:: *)
(* CUDAFoldList *)
Options[CUDAFoldList] = Options[iCUDAFoldList] = {"Device" -> Automatic, "OutputMemory" -> None}
SyntaxInformation[CUDAFoldList] = {"ArgumentsPattern" -> {_, _, _, OptionsPattern[]}}

CUDAFoldList[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDAFoldList, 3, 3], HoldFirst]) :=
	With[{res = iCUDAFoldList[args, opts]},
		res /; res =!= $Failed
	]
iCUDAFoldList[op_, initialValue_, elem_, opts:OptionsPattern[]] :=
	Catch[iCUDAFoldList[CUDAFoldList, op, initialValue, elem, opts]]
iCUDAFoldList[errMsgHd_, op_, initialValue_, elem_, opts:OptionsPattern[]] :=
	Module[{device, dims, input, output, runRes, res, cudaScanFunction},

		InitializeWGL[iCUDAFoldList, errMsgHd, opts];
		device = GetAndCheckOption[errMsgHd, "Device", {opts}];
		
		If[!ListQ[elem] && Head[elem] =!= CUDAMemory,
			Throw[Message[errMsgHd::input, elem]; $Failed]
		];
		
		If[!MemberQ[{Min, Max, Plus, Minus, Times, Divide}, op],
			Throw[Message[errMsgHd::op, op]; $Failed]
		];
		
		dims = If[Head[elem] === CUDAMemory,
			WGLMemoryGetDimensions[elem],
			Dimensions[elem]
		];
		
		If[Head[elem] === CUDAMemory && ResolveElementWidth[WGLMemoryGetType[elem]] > 1,	 
			Throw[Message[errMsgHd::memvec, WGLMemoryGetType[elem]]; $Failed]	 
		];
		
		If[Length[dims] =!= 1,
			Throw[Message[errMsgHd::novec, dims]; $Failed]
		];  
		
		cudaScanFunction = CUDAFunctionLoad[{CUDALink`Internal`$ExamplesLibraryPath}, "oCUDAScan", {_Real, {_Integer, "Input"}, {_Integer, "Output"}, "UTF8String"}, 256, "Device" -> device];
		If[Head[cudaScanFunction] =!= CUDAFunction,
			Throw[$Failed]
		];
		
		runRes = Catch[Reap[
			input = AddTemporaryMemory[errMsgHd, elem];
			output = GetAndCheckOption[errMsgHd, "OutputMemory", {opts}];
			If[output === None,
				output = If[Head[elem] === CUDAMemory,
					AllocateClonedMemory[errMsgHd, input, Automatic],
					AllocateTemporaryClonedMemory[errMsgHd, input, Automatic]
				]
			];
			res = cudaScanFunction[1.0 * initialValue, input, output, ToString[op], 256]
		]];

		res = If[FreeQ[res, $Failed] && FreeQ[runRes, $Failed],
			If[Head[elem] === CUDAMemory,
				output,
				CUDAMemoryGet[output]
			],
			$Failed
		];
		
		If[ListQ[runRes],
			GPUDeleteMemory[CUDAMemory, errMsgHd, #]& /@ Flatten[Last[runRes]]
		];
		
		res
	]
	
(******************************************************************************)
(******************************************************************************)
(******************************************************************************)

(* ::Section:: *)
(* CUDAFold *)
Options[CUDAFold] = Options[iCUDAFold] = {"Device" -> Automatic}
SyntaxInformation[CUDAFold] = {"ArgumentsPattern" -> {_, _, _, OptionsPattern[]}}

CUDAFold[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDAFold, 3, 3], HoldFirst]) :=
	With[{res = iCUDAFold[args, opts]},
		res /; res =!= $Failed
	]
iCUDAFold[op_, initialValue_, elem_, opts:OptionsPattern[]] :=
	Catch[iCUDAFold[CUDAFold, op, initialValue, elem, opts]]
iCUDAFold[errMsgHd_, op_, initialValue_, elem_, opts:OptionsPattern[]] :=
	Module[{device, dims, input, runRes, res, cudaFoldFunction, out = Developer`ToPackedArray[{0.0}]},

		InitializeWGL[iCUDAFold, errMsgHd, opts];
		device = GetAndCheckOption[errMsgHd, "Device", {opts}];
		
		If[!ListQ[elem] && Head[elem] =!= CUDAMemory,
			Throw[Message[errMsgHd::input, elem]; $Failed]
		];
		
		If[!MemberQ[{Min, Max, Plus, Minus, Times, Divide}, op],
			Throw[Message[errMsgHd::op, op]; $Failed]
		];
		
		dims = If[Head[elem] === CUDAMemory,
			WGLMemoryGetDimensions[elem],
			Dimensions[elem]
		];
		
		If[Head[elem] === CUDAMemory && ResolveElementWidth[WGLMemoryGetType[elem]] > 1,	 
			Throw[Message[errMsgHd::memvec, WGLMemoryGetType[elem]]; $Failed]	 
		];
		
		If[Length[dims] =!= 1,
			Throw[Message[errMsgHd::novec, dims]; $Failed]
		];  
		
		cudaFoldFunction = CUDAFunctionLoad[{CUDALink`Internal`$ExamplesLibraryPath}, "oCUDAReduce", {{_Real, _, "Shared"}, _Real, {_Integer, "Input"}, "UTF8String"}, 256, "Device" -> device];
		If[Head[cudaFoldFunction] =!= CUDAFunction,
			Throw[$Failed]
		];
		
		runRes = Catch[Reap[
			input = AddTemporaryMemory[errMsgHd, elem];
		]];
		
		res = If[FreeQ[runRes, $Failed],
			cudaFoldFunction[out, 1.0 * initialValue, input, ToString[op], 256],
			$Failed
		];

		res = If[FreeQ[res, $Failed],
			castTo[WGLMemoryGetType[input], First[out]],
			$Failed
		];

		If[ListQ[runRes],
			GPUDeleteMemory[CUDAMemory, errMsgHd, #]& /@ Flatten[Last[runRes]]
		];
		
		res
	]

(******************************************************************************)
(******************************************************************************)
(******************************************************************************)

(* ::Section:: *)
(* CUDAMap *)
Options[CUDAMap] = Options[iCUDAMap] = {"Device" -> Automatic, "OutputMemory" -> None}
SyntaxInformation[CUDAMap] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}}	

CUDAMap[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDAMap, 2, 2], HoldFirst]) :=
	With[{res = iCUDAMap[args, opts]},
		res /; res =!= $Failed
	]
iCUDAMap[op_, elem_, opts:OptionsPattern[]] :=
	Catch[iCUDAMap[CUDAMap, op, elem, opts]]
iCUDAMap[errMsgHd_, op_, elem_, opts:OptionsPattern[]] :=
	Module[{device, dims, input, output, runRes, res, cudaMapFunction},

		InitializeWGL[iCUDAMap, errMsgHd, opts];
		device = GetAndCheckOption[errMsgHd, "Device", {opts}];
		
		If[!ListQ[elem] && Head[elem] =!= CUDAMemory,
			Throw[Message[errMsgHd::input, elem]; $Failed]
		];
		
		If[!MemberQ[{Cos, Sin, Tan, ArcCos, ArcSin, ArcTan, Cosh, Sinh, Exp, Log, Log10, Sqrt, Ceiling, Floor, Abs}, op],
			Throw[Message[errMsgHd::op, op]; $Failed]
		];
		
		dims = If[Head[elem] === CUDAMemory,
			WGLMemoryGetDimensions[elem],
			Dimensions[elem]
		];
		
		If[Head[elem] === CUDAMemory && ResolveElementWidth[WGLMemoryGetType[elem]] > 1,	 
			Throw[Message[errMsgHd::memvec, WGLMemoryGetType[elem]]; $Failed]	 
		];
		
		If[Length[dims] =!= 1,
			Throw[Message[errMsgHd::novec, dims]; $Failed]
		];  
		
		cudaMapFunction = CUDAFunctionLoad[{CUDALink`Internal`$ExamplesLibraryPath}, "oCUDAMap", {{_Integer, "Input"}, {_Integer, "Output"}, "UTF8String"}, 256, "Device" -> device];
		If[Head[cudaMapFunction] =!= CUDAFunction,
			Throw[$Failed]
		];
		
		runRes = Catch[Reap[
			input = AddTemporaryMemory[errMsgHd, elem];
			output = GetAndCheckOption[errMsgHd, "OutputMemory", {opts}];
			If[output === None,
				output = If[Head[elem] === CUDAMemory,
					AllocateClonedMemory[errMsgHd, input, Automatic],
					AllocateTemporaryClonedMemory[errMsgHd, input, Automatic]
				]
			];
			res = cudaMapFunction[input, output, ToString[op], 256]
		]];

		res = If[FreeQ[res, $Failed] && FreeQ[runRes, $Failed],
			If[Head[elem] === CUDAMemory,
				output,
				CUDAMemoryGet[output]
			],
			$Failed
		];
		
		If[ListQ[runRes],
			GPUDeleteMemory[CUDAMemory, errMsgHd, #]& /@ Flatten[Last[runRes]]
		];

		res
	]

(******************************************************************************)
(******************************************************************************)
(******************************************************************************)

(* ::Section:: *)
(* CUDAClamp *)
Options[CUDAClamp] = Options[iCUDAClamp] = {"Device" -> Automatic, "OutputMemory" -> None}
SyntaxInformation[CUDAClamp] = {"ArgumentsPattern" -> {_, _., _., OptionsPattern[]}}	

CUDAClamp[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDAClamp, 1, 3], HoldFirst]) :=
	With[{res = iCUDAClamp[args, opts]},
		res /; res =!= $Failed
	]
CUDAClamp[opts:OptionsPattern[]] /; (Message[CUDAClamp::args, opts]; False) := $Failed
iCUDAClamp[elem_, low_:Automatic, high_:Automatic, opts:OptionsPattern[]] :=
	Catch[iCUDAClamp[CUDAClamp, elem, low, high, opts]]
iCUDAClamp[errMsgHd_, elem_, low0_, high0_, opts:OptionsPattern[]] :=
	Module[{device, dims, input, output, runRes, res, low, high, clampFunction},

		InitializeWGL[iCUDAClamp, errMsgHd, opts];
		device = GetAndCheckOption[errMsgHd, "Device", {opts}];
		
		If[!ListQ[elem] && Head[elem] =!= CUDAMemory && !Image`PossibleImageQ[elem],
			Throw[Message[errMsgHd::input, elem]; $Failed]
		];
		
		If[low0 =!= Automatic && !NumericQ[low0],
			Throw[Message[errMsgHd::low, low0]; $Failed]
		];
		
		If[high0 =!= Automatic && !NumericQ[high0],
			Throw[Message[errMsgHd::low, high0]; $Failed]
		];
		
		dims = If[Head[elem] === CUDAMemory,
			WGLMemoryGetDimensions[elem],
			Dimensions[elem]
		];
		
		If[Head[elem] === CUDAMemory && ResolveElementWidth[WGLMemoryGetType[elem]] > 1,	 
			Throw[Message[errMsgHd::memvec, WGLMemoryGetType[elem]]; $Failed]	 
		];
		
		clampFunction = CUDAFunctionLoad[{CUDALink`Internal`$ExamplesLibraryPath}, "oCUDAClamp", {{_Integer, "Input"}, {_Integer, "Output"}, _Real, _Real}, 256, "Device" -> device];
		If[Head[clampFunction] =!= CUDAFunction,
			Throw[$Failed]
		];
		
		runRes = Catch[Reap[
			input = AddTemporaryMemory[errMsgHd, elem];
			output = GetAndCheckOption[errMsgHd, "OutputMemory", {opts}];
			If[output === None,
				output = If[Head[elem] === CUDAMemory,
					AllocateClonedMemory[errMsgHd, input, Automatic],
					AllocateTemporaryClonedMemory[errMsgHd, input, Automatic]
				]
			];
			low = If[low0 === Automatic,
				0.0,
				N[low0]
			];
			high = If[high0 === Automatic,
				If[TypeRealQ[WGLMemoryGetType[input]],
					1.0,
					255.0
				],
				N[high0]
			];
			res = If[high < low,
				Message[errMsgHd::highlow, If[high0 === Automatic, high, high0], If[low0 === Automatic, low, low0]];
				$Failed,
				clampFunction[input, output, 1.0*low, 1.0*high, 256]
			];
		]];

		res = If[FreeQ[res, $Failed] && FreeQ[runRes, $Failed],
			If[Head[elem] === CUDAMemory,
				output,
				CUDAMemoryGet[output]
			],
			$Failed
		];
		
		If[ListQ[runRes],
			GPUDeleteMemory[CUDAMemory, errMsgHd, #]& /@ Flatten[Last[runRes]]
		];

		res
	]

(******************************************************************************)
(******************************************************************************)
(******************************************************************************)

(* ::Section:: *)
(* CUDAColorNegate *)
Options[CUDAColorNegate] = Options[iCUDAColorNegate] = {"Device" -> Automatic, "OutputMemory" -> None}
SyntaxInformation[CUDAColorNegate] = {"ArgumentsPattern" -> {_, OptionsPattern[]}}	

CUDAColorNegate[args:nonOptionPattern, opts:OptionsPattern[]] ? (Function[{arg}, OptionsCheck[arg, CUDAColorNegate, 1, 1], HoldFirst]) :=
	With[{res = Catch[iCUDAColorNegate[CUDAColorNegate, args, opts]]},
		res /; res =!= $Failed
	]
iCUDAColorNegate[errMsgHd_, elem_, opts:OptionsPattern[]] :=
	Module[{device, dims, input, output, runRes, res, colorNegateFunction, blockDims},

		InitializeWGL[iCUDAColorNegate, errMsgHd, opts];
		device = GetAndCheckOption[errMsgHd, "Device", {opts}];
		
		If[!ListQ[elem] && Head[elem] =!= CUDAMemory && !Image`PossibleImageQ[elem],
			Throw[Message[errMsgHd::input, elem]; $Failed]
		];
		
		dims = If[Head[elem] === CUDAMemory,
			WGLMemoryGetDimensions[elem],
			Dimensions[elem]
		];
		
		If[Head[elem] === CUDAMemory && ResolveElementWidth[WGLMemoryGetType[elem]] > 1,	 
			Throw[Message[errMsgHd::memvec, WGLMemoryGetType[elem]]; $Failed]	 
		];
		
		blockDims = If[(ListQ[elem] && ArrayDepth[elem] >= 2) || Image`PossibleImageQ[elem] || (Head[elem] === CUDAMemory && Length[WGLMemoryGetDimensions[elem]] >= 2),
			{16, 16},
			256
		];

		colorNegateFunction = CUDAFunctionLoad[{CUDALink`Internal`$ExamplesLibraryPath}, "oCUDAColorNegate", {{_Integer, "Input"}, {_Integer, "Output"}}, blockDims, "Device" -> device];
		If[Head[colorNegateFunction] =!= CUDAFunction,
			Throw[$Failed]
		];
		
		runRes = Catch[Reap[
			input = AddTemporaryMemory[errMsgHd, elem];
			output = GetAndCheckOption[errMsgHd, "OutputMemory", {opts}];
			If[output === None,
				output = If[Head[elem] === CUDAMemory,
					AllocateClonedMemory[errMsgHd, input, Automatic],
					AllocateTemporaryClonedMemory[errMsgHd, input, Automatic]
				]
			];
			res = colorNegateFunction[input, output];
		]];

		res = If[FreeQ[res, $Failed] && FreeQ[runRes, $Failed],
			If[Head[elem] === CUDAMemory,
				output,
				CUDAMemoryGet[output]
			],
			$Failed
		];
		
		If[ListQ[runRes],
			GPUDeleteMemory[CUDAMemory, errMsgHd, #]& /@ Flatten[Last[runRes]]
		];

		res
	]

(******************************************************************************)
(******************************************************************************)
(******************************************************************************)

(* ::Section:: *)
(* Options Pricing *)

(* From the CUDA SDK ${basedir}/ExtraComponents/CUDA_SDK/3.0/Linux-x86-64/C/src *)
(* Uses code from BlackScholes, BinomialOptions, and MonteCarlo *)

(* ::Subsection:: *)
(* Option Checking *)

GetAndCheckOption[head:CUDAFinancialDerivative, errMsgHd:CUDAFinancialDerivative, n:Method, opts_] :=
	Module[{method = OptionValue[head, {opts}, n]},
		If[method == "Binomial" || method == "MonteCarlo" || method == "BlackScholes" || method == Automatic,
			method,
			Throw[Message[errMsgHd::invmthd, method]; $Failed]
		]
	]
	
GetAndCheckOption[head_, errMsgHd_, n:"DerivativeMethod", opts_] :=
	Module[{derivMethod = OptionValue[head, {opts}, n]},
		If[derivMethod === Automatic || derivMethod == "Numeric",
			derivMethod,
			Throw[Message[errMsgHd::invdmthd, derivMethod]; $Failed]
		]
	]
	
(* ::Subsection:: *)
(* Implementation *)

(* Parse inputs (FinancialDerivative style) *)
CUDAOptionsPricingParseInputs[errMsgHd_, instrument_, params_, ambientParams_, prop_:{}] :=
	Module[
		{
			currentPrices, strikePrices, expiration, volatility, interestRate, dividend, barrier, type, callOrPut, tempLHS, tempRHS, goal, exchangeRate, 
			exchangeVolatility, foreignRiskFree, correlation, rebate, validParameterTypes, validInstrumentTypes, validAmbientParameterTypes, validCalcTypes
		},
		
		validInstrumentTypes = {
			"American", "European", "AsianArithmetic", "AsianGeometric", "BarrierDownIn",
			"BarrierDownOut", "BarrierUpIn", "BarrierUpOut", "AmericanQuantoFixedExchange",
			"LookbackFixed", "LookbackFloating"
		};		
		validParameterTypes = Sort[{
			"StrikePrice", "Expiration", "Barriers"
		}];
		validAmbientParameterTypes = Sort[{
			"CurrentPrice", "Dividend", "Volatility", "InterestRate",
			"ExchangeRate", "ExchangeVolatility", "ForeignInterestRate",
			"Correlation", "Rebate"
		}];
		validCalcTypes = Sort[{
			"Value", "Delta", "Vega", "Theta", "Rho", "Gamma",
			"Vanna", "Charm", "Vomma", "DVegaDTime", "Speed",
			"Zomma", "Color"
		}];
				
		If[Length[instrument] < 2 || Length[instrument] > 3,
			Throw[Message[errMsgHd::invoptt, instrument]; $Failed]
		];
		
		type = instrument[[1]];
		If[FreeQ[validInstrumentTypes, type],
			Throw[Message[errMsgHd::invoptt, instrument[[1]]]; $Failed]
		];
		
		callOrPut = instrument[[2]];
		If[FreeQ[{"Call", "Put"}, callOrPut],
			Throw[Message[errMsgHd::invoptt, instrument[[2]]]; $Failed]
		];
		
		goal = Which[
			Length[instrument] == 3,
				goal = instrument[[3]];
				If[FreeQ[validCalcTypes, goal],
					Throw[Message[errMsgHd::invoptt, goal]; $Failed],
					goal
				],
			prop =!= {},
				goal = First[prop];
				Which[
					FreeQ[validCalcTypes, goal], 
						Throw[Message[errMsgHd::invoptt, goal]; $Failed],
					Length[prop] =!= 1,
						Throw[Message[errMsgHd::invprop, prop]; $Failed],
					True,
						goal
				],
			True,
				"Value"
		];
		
		If[!ListQ[params],
			Throw[Message[errMsgHd::invparamsl, params]; $Failed]
		];
		
		If[!ListQ[ambientParams],
			Throw[Message[errMsgHd::invambparamsl, ambientParams]; $Failed]
		];
		
		If[Select[params, Head[#] =!= Rule&, 1] =!= {},
			Throw[Message[errMsgHd::invparamsr, First[Select[params, Head[#] =!= Rule&, 1]]]; $Failed]
		];
		
		If[Select[ambientParams, Head[#] =!= Rule&, 1] =!= {},
			Throw[Message[errMsgHd::invambparamsr, First[Select[ambientParams, Head[#] =!= Rule&, 1]]]; $Failed]
		];

		If[Union[validParameterTypes, First /@ params] =!= validParameterTypes,
			Throw[Message[errMsgHd::invparams, Complement[First /@ params, validParameterTypes]]; $Failed]
		];
		
		If[Union[validAmbientParameterTypes, First /@ ambientParams] =!= validAmbientParameterTypes,
			Throw[Message[errMsgHd::invambparams, Complement[First /@ ambientParams, validAmbientParameterTypes]]; $Failed]
		];
		
		(* Default values. *)
		(* Should not cause a crash for any option type. *)
		currentPrices = 1.0;
		strikePrices = 1.0;
		expiration = 1.0;
		dividend = 0.0;
		barrier = 0.0;
		volatility = 0.1;
		interestRate = 0.0;
		exchangeRate = 1.0;
		exchangeVolatility = 0.0;
		foreignRiskFree = 0.0;
		correlation = 0.0;
		rebate = 0.0;
		
		(
			tempLHS = First[#];
			tempRHS = Last[#];

			Which[
				IntegerQ[tempRHS],
					tempRHS = 1.0 * tempRHS, 
				ListQ[tempRHS] && GPUTools`Utilities`ArrayType[tempRHS] === Integer,
					tempRHS = 1.0 * tempRHS,
				ListQ[tempRHS] && GPUTools`Utilities`ArrayType[tempRHS] =!= Real,
					Throw[Message[errMsgHd::invinpt, tempRHS]; $Failed],
				FreeQ[{List, Real}, Head[tempRHS]],
					Throw[Message[errMsgHd::invinpt, tempRHS]; $Failed]
			];

			Switch[tempLHS,
				"CurrentPrice",
					currentPrices = tempRHS,
				"StrikePrice",
					strikePrices = tempRHS,
				"Expiration",
					expiration = tempRHS,
				"Dividend",
					dividend = tempRHS,
				"Volatility",
					volatility = tempRHS,
				"InterestRate",
					interestRate = tempRHS,
				"Barriers",
					barrier = tempRHS,
				"ExchangeRate",
					exchangeRate = tempRHS,
				"ExchangeVolatility",
					exchangeVolatility = tempRHS,
				"ForeignInterestRate",
					foreignRiskFree = tempRHS,
				"Correlation",
					correlation = tempRHS,
				"Rebate",
					rebate = tempRHS,
				_,
					Throw[$Failed] (* This should be caught above by the invparams and invambparams checks. *)
			]
		)& /@ Join[params, ambientParams];
	
		Return[{type, callOrPut, goal, currentPrices, strikePrices, expiration, interestRate, dividend, volatility, barrier, exchangeRate, exchangeVolatility, foreignRiskFree, correlation, rebate}];
	]

Options[CUDAFinancialDerivative] = Options[iCUDAFinancialDerivative] = Options[oCUDAFinancialDerivative] = Sort[{
	Method -> Automatic,
	"DerivativeMethod" -> Automatic,
	"Device" -> Automatic,
	"TargetPrecision" -> Automatic
}]

SyntaxInformation[CUDAFinancialDerivative] = {"ArgumentsPattern" -> {_, _, _, _., OptionsPattern[]}}

CUDAFinancialDerivative[instrument_, parameters_, ambientParams_, prop_:{}, opts:OptionsPattern[]] :=
	With[{res = Catch[iCUDAFinancialDerivative[instrument, parameters, ambientParams, prop, opts]]},
		res /; res =!= $Failed
	]
CUDAFinancialDerivative[args___] /; (ArgumentCountQ[CUDAFinancialDerivative, Length[{args}], 3, 4]; False) := $Failed


iCUDAFinancialDerivative[instrument_, parameters_, ambientParams_, prop_:{}, opts:OptionsPattern[]] :=
	oCUDAFinancialDerivative[CUDAFinancialDerivative, instrument, parameters, ambientParams, prop, opts]



iCUDAFinancialDerivative[errMsgHd_, instrument_, parameters_, ambientParams_, prop:{_String, _String..}, opts:OptionsPattern[]] :=
	iCUDAFinancialDerivative[errMsgHd, instrument, parameters, ambientParams, #, opts]& /@ prop
oCUDAFinancialDerivative[errMsgHd_, instrument_, parameters_, ambientParams_, prop_:{}, opts:OptionsPattern[]] :=
	Module[
		{
			dllFile, device, targetPrecision, res, numOptions, currentPrices, strikePrices, result, hResult1, hResult2,
			method, doComputation, delta, derivMethod, type, callOrPut, hSpotPrices, hStrikePrices, hExpiration, hVolatilities,
			hInterest, hDividend, hBarrier, goal, hExchangeRate, hExchangeVolatility, hForeignRiskFree, 
			hCorrelation, hRebate, callPutFactor,
			
			callPrices, putPrices, volatility, expiration, interest, dividends, barrier, rebate,
			
			binomialOptionsFun, blackScholesFun, monteCarloFun
		},
		
		InitializeWGL[oCUDAFinancialDerivative, errMsgHd, opts];
		
		If[CUDALink`Internal`$ExamplesLibraryPath,
			Throw[Message[errMsgHd::nodev]; $Failed]
		];
		
		dllFile = CUDALink`Internal`$ExamplesLibraryPath;

		method = GetAndCheckOption[errMsgHd, Method, {opts}];
		derivMethod = GetAndCheckOption[errMsgHd, "DerivativeMethod", {opts}];
		device = GetAndCheckOption[errMsgHd, "Device", {opts}];
		targetPrecision = GetAndCheckOption[errMsgHd, "TargetPrecision", {opts}];
		
		delta = 0.0001; (* For numeric differentiation *)
				
		res = CUDAOptionsPricingParseInputs[errMsgHd, instrument, parameters, ambientParams, If[ListQ[prop], prop, {prop}]];
		
		{
			type,
			callOrPut,
			goal,
			hSpotPrices,
			hStrikePrices,
			hExpiration,
			hInterest,
			hDividend,
			hVolatilities,
			hBarrier,
			hExchangeRate,
			hExchangeVolatility,
			hForeignRiskFree,
			hCorrelation,
			hRebate
		} = res;		

		(* Use optimal method *)
		If[method === Automatic,
			method = Switch[type,
				"European", "BlackScholes",
				"American", "Binomial",
				"AsianArithmetic", "MonteCarlo",
				"AsianGeometric", "BlackScholes",
				"BarrierDownIn", "MonteCarlo",
				"BarrierDownOut", "MonteCarlo",
				"BarrierUpIn", "MonteCarlo",
				"BarrierUpOut", "MonteCarlo",
				"AmericanQuantoFixedExchange", "Binomial",
				"LookbackFixed", "MonteCarlo",
				"LookbackFloating", "MonteCarlo", (* Do analytic implementation of this! *)
				_, Throw[Message[errMsgHd::invoptt, type]; $Failed]
			];
		];

	    Switch[method,
	      "BlackScholes",
	        If[FreeQ[{"European", "AsianGeometric", "BarrierDownIn", "BarrierDownOut", "BarrierUpIn", "BarrierUpOut", "LookbackFixed", "LookbackFloating"}, type],
	          Throw[Message[errMsgHd::invmthd, method]; $Failed]
	        ],
	      "Binomial",
	        If[FreeQ[{"European", "American", "AmericanQuantoFixedExchange"}, type],
	          Throw[Message[errMsgHd::invmthd, method]; $Failed]
	        ],
	      "MonteCarlo",
	        If[FreeQ[{"European", "AsianArithmetic", "BarrierDownIn", "BarrierDownOut", "BarrierUpIn", "BarrierUpOut", "LookbackFixed", "LookbackFloating"}, type],
	          Throw[Message[errMsgHd::invmthd, method]; $Failed]
	        ]
	    ];
	    
		
		(* Determine if any inputs are arrays, ensure all array inputs are equal in length, and allocate device memory appropriately *)
		numOptions = Max[Length/@{hSpotPrices, hStrikePrices, hExpiration, hVolatilities, hInterest, hDividend, hBarrier, hExchangeRate, hExchangeVolatility, hForeignRiskFree, hCorrelation, hRebate}];
		If[FreeQ[{0,numOptions}, Length[#]],
			Throw[Message[errMsgHd::invinptsz, #]; $Failed]
		]& /@ {
			hSpotPrices, hStrikePrices, hExpiration, hVolatilities,
			hInterest, hDividend, hBarrier, hExchangeRate,
			hExchangeVolatility, hForeignRiskFree, hCorrelation, hRebate
		};
		
		If[numOptions == 0,
			numOptions = 1
		];


		
		blackScholesFun := CUDAFunctionLoad[
			{dllFile},
			"oBlackScholes",
			{
				{_Real, _, "Output"},	(* Call *)
				{_Real, _, "Output"},	(* Put *)
				{_Real, _, "Input"},	(* Current Prices *)
				{_Real, _, "Input"},	(* Strike Prices *)
				{_Real, _, "Input"},	(* Expiration *)
				{_Real, _, "Input"},	(* Interest *)
				{_Real, _, "Input"},	(* Voltality *)
				{_Real, _, "Input"},	(* Dividend *)
				{_Real, _, "Input"},	(* Barrier *)
				{_Real, _, "Input"},	(* Rebate *)
				_Integer,				(* Calculation Type *)
				_Integer,				(* Option Type *)
				_Integer,				(* Call or Put *)
				_Integer				(* Number of Options *)
			},
			128,
			"TargetPrecision" -> targetPrecision,
			"Device" -> device
		];
		
		binomialOptionsFun := CUDAFunctionLoad[
			{dllFile},
			"oBinomialMethod",
			{
				{_Real, _, "Output"},	(* Result *)
				{_Real, _, "Input"},	(* Spot Prices *)
				{_Real, _, "Input"},	(* Strike Prices *)
				{_Real, _, "Input"},	(* Expiration *)
				{_Real, _, "Input"},	(* Interest *)
				{_Real, _, "Input"},	(* Volatility *)
				{_Real, _, "Input"},	(* Dividend *)
				_Integer,				(* Number of Options *)
				_Integer,				(* Option Type *)
				_Integer				(* Call or Put *)
			},
			256,
			"TargetPrecision" -> targetPrecision,
			"Device" -> device
		];
		
		monteCarloFun := CUDAFunctionLoad[
			{dllFile},
			"oMonteCarloMethod",
			{
				{_Real, _, "Output"},	(* Result *)
				{_Real, _, "Input"},	(* Current Prices *)
				{_Real, _, "Input"},	(* Strike Prices *)
				{_Real, _, "Input"},	(* Expiration *)
				{_Real, _, "Input"},	(* Interest *)
				{_Real, _, "Input"},	(* Voltality *)
				{_Real, _, "Input"},	(* Dividend *)
				{_Real, _, "Input"},	(* Barrier *)
				_Integer,				(* Number of Options *)
				_Integer,
				_Integer
			},
			128,
			"TargetPrecision" -> targetPrecision,
			"Device" -> device
		];

		callPrices = CUDAMemoryAllocate[Real, numOptions, "Device"->device, "TargetPrecision"->targetPrecision];
		putPrices = CUDAMemoryAllocate[Real, numOptions, "Device"->device, "TargetPrecision"->targetPrecision];
						
		currentPrices = loadHostMemory[errMsgHd, hSpotPrices, numOptions, device, targetPrecision];
		strikePrices = loadHostMemory[errMsgHd, hStrikePrices, numOptions, device, targetPrecision];

		volatility = loadHostMemory[errMsgHd, hVolatilities, numOptions, device, targetPrecision];
		expiration = loadHostMemory[errMsgHd, hExpiration, numOptions, device, targetPrecision];
		interest = loadHostMemory[errMsgHd, hInterest, numOptions, device, targetPrecision];
		dividends = loadHostMemory[errMsgHd, hDividend, numOptions, device, targetPrecision];
		barrier = loadHostMemory[errMsgHd, hBarrier, numOptions, device, targetPrecision];
		rebate = loadHostMemory[errMsgHd, hRebate, numOptions, device, targetPrecision];

		If[Head[monteCarloFun] =!= CUDAFunction || Head[binomialOptionsFun] =!= CUDAFunction || Head[blackScholesFun] =!= CUDAFunction,
			Throw[$Failed]
		];
		
		callPutFactor = If[callOrPut == "Call",
			1,
			-1
		];
		
		doComputation[computationMethod_] :=
			Module[{},
				res = Switch[computationMethod,
					"BlackScholes",
						blackScholesFun[
							callPrices, putPrices, currentPrices, strikePrices,
							expiration, interest, volatility, dividends, barrier, rebate, numOptions,
							If[derivMethod === "Numeric", -1, toCalculate[goal]], getOptionType[type], callPutFactor, 512 
						],
					"Binomial",
						binomialOptionsFun[
							callPrices, currentPrices, strikePrices, expiration,
							interest, volatility, dividends, numOptions,
							getOptionType[type], callPutFactor
						],
					"MonteCarlo",
						monteCarloFun[
							callPrices, currentPrices, strikePrices, expiration,
							interest, volatility, dividends, barrier, numOptions,
							getOptionType[type], callPutFactor
						]
				];
				If[res === $Failed,
					$Failed,
					Switch[computationMethod,
						"BlackScholes" | "MonteCarlo",
							If[callOrPut === "Call",
								CUDAMemoryGet[callPrices],
								CUDAMemoryGet[putPrices]
							],
						"Binomial",
							With[{call = CUDAMemoryGet[callPrices], exchangeRate = If[Length[hExchangeRate] === 1, First[hExchangeRate], hExchangeRate]},
								If["AmericanQuantoFixedExchange" === type,
									call * exchangeRate,
									call
								]
							]
					]
				]
			];
		
		result = doComputation[method];
		
		If[goal =!= "Value" && (method=!="BlackScholes" || derivMethod == "Numeric"),
			hResult1 = result;
			result[[1]] = hResult1[[1]];
			result = Switch[goal,
				"Delta",
					hSpotPrices += delta;
					
					CUDAMemoryUnload[currentPrices];
					currentPrices = loadHostMemory[errMsgHd, hSpotPrices, numOptions, device, targetPrecision];

					hResult1 = doComputation[method];
					result = (hResult1 - result)/delta,
				"Vega",
					hVolatilities += delta;
					
					CUDAMemoryUnload[volatility];
					volatility = loadHostMemory[errMsgHd, hVolatilities, numOptions, device, targetPrecision];
					
					hResult1 = doComputation[method];
					result = (hResult1 - result)/delta,
				"Gamma",
				
					CUDAMemoryUnload[currentPrices];
					currentPrices = loadHostMemory[errMsgHd, hSpotPrices - delta, numOptions, device, targetPrecision];
					hResult1 = doComputation[method];
					
					CUDAMemoryUnload[currentPrices];
					currentPrices = loadHostMemory[errMsgHd, hSpotPrices + delta, numOptions, device, targetPrecision];
					hResult2 = doComputation[method];
					
					result = (hResult2 + hResult1 - 2*result)/(delta^2),
				"Theta", (* Negative of FinancialDerivative's output??? *)
					hExpiration -= delta;
					
					CUDAMemoryUnload[expiration];
					expiration = loadHostMemory[errMsgHd, hExpiration, numOptions, device, targetPrecision];
					
					hResult1 = doComputation[method];
					result = (hResult1 - result)/delta,
				"Rho",
					hInterest += delta;
					
					CUDAMemoryUnload[interest];
					interest = loadHostMemory[errMsgHd, hInterest, numOptions, device, targetPrecision];
					
					hResult1 = doComputation[method];
					result = (hResult1 - result)/delta,
				_,
					Message[errMsgHd::invnderiv, goal];
					$Failed
			]
		];
		
		CUDAMemoryUnload[currentPrices, strikePrices, callPrices, putPrices, expiration, volatility, interest, dividends, barrier, rebate];
		
		If[numOptions == 1 && ListQ[result],	
			First[result],
			result
		]
	]

getOptionType[type_] :=
	Switch[type,
		"European",
			100,
		"American" | "AmericanQuantoFixedExchange",
			101,
		"AsianArithmetic",
			102,
		"AsianGeometric",
			109,
		"BarrierUpIn",
			103,
		"BarrierDownIn",
			104,
		"BarrierUpOut",
			105,
		"BarrierDownOut",
			106,
		"LookbackFixed",
			107,
		"LookbackFloating",
			108,
		_,
			-1
	]
toCalculate[goal_] :=
	Switch[goal,
		"Value", 0,
		"Delta", 1,
		"Vega", 2,
		"Theta", 3,
		"Rho", 4,
		"Gamma", 5,
		"Vanna", 6,
		"Charm", 7,
		"Vomma", 8,
		"DVegaDTime", 9,
		"Speed", 10,
		"Zomma", 11,
		"Color", 12,
		_, 0
	]


loadHostMemory[errMsgHd_, hostMem_, len_, device_, targetPrecision_] :=
	If[Length[hostMem] === len,
		iCUDAMemoryLoad[errMsgHd, hostMem, "Device"->device, "TargetPrecision"->targetPrecision],
		iCUDAMemoryLoad[errMsgHd, ConstantArray[hostMem, len], "Device"->device, "TargetPrecision"->targetPrecision]
	]
(******************************************************************************)
(******************************************************************************)
(******************************************************************************)

(* ::Section:: *)
(* CUDA Developer *)

CUDALink`Developer`$Library := `$Library


(******************************************************************************)

CUDALink::initlib 		= "CUDALink failed to initialize `1`. " <> GPUTools`Message`ReferToSystemRequirements["CUDALink"]

GPUTools`Utilities`DefineMessage[
	{
		CUDAFunctionLoad
	},
	{
		{"invinpt", "The input `1` specified is not valid. Input can only be a machine Integer, Real, or a list of Real or Integer."},
		{"nonmchint", "The input `1` specified is not a machine integer."},
		{"nonmchre", "The input `1` specified is not a machine real."},
		{"nodir", "The directory `1` specified does not exist."},
		{"invblk", "The block dimensions specified are not valid. Block dimensions can be an integer or a list and must not exceed the dimentions retured by CUDAData."},
		{"invgrd", "The grid dimensions specified are not valid. Grid dimensions can be an integer or a list and must not exceed the dimentions retured by CUDAData."},
		{"cmpf", "The kernel compilation failed. Consider setting the option \"ShellOutputFunction\"->Print to display the compiler error message."},
		{"invprogd", "The kernel program specified is not valid. Valid kernel programs are a string or a one element list."},
		{"invsyncin", "The \"SynchronizeInputs\" set to `1` is not valid. \"SynchronizeInputs\" can only be set to True or False."},
		{"invsyncout", "The \"SynchronizeOutputs\" set to `1` is not valid. \"SynchronizeOutputs\" can only be set to True or False."},
		{"autknbin", "Passing in the GPU binary `1` requires setting the \"KernelName\" option to name of the kernel to be executed."},
		{"invkernam", "The \"KernelName\" set to `1` is not valid. If set to Automatic, consider setting the \"KernelName\" option manually."},
		{"nofile", "The file `1` does not exist."},
		{"invfile", "The input file `1` is not valid."}
	}
]

GPUTooles`Utilities`DefineMessage[
	{
		CUDAMemoryLoad, CUDAMemoryAllocate
	},
	{
		{"tenstyp", ""}
	}
]

GPUTools`Utilities`DefineMessage[
	{
		CUDAFunctionLoad
		(*, CUDASymbolicCGenerate, CUDACodeStringGenerate, CUDACodeGenerate, CUDALibraryGenerate, CUDALibraryFunctionGenerate *)
	},
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
		{"debug", CreateLibrary::debug},
		{"lnkopt", CreateLibrary::lnkopt},
		{"compopt", CreateLibrary::compopt}
		
	}
]

GPUTools`Utilities`DefineMessage[
	{
		NVCCCompiler, CreateLibrary, CreateExecutable, CreateObjectFile, CUDAFunctionLoad,
		CUDASymbolicCGenerate, CUDACodeStringGenerate, CUDACodeGenerate,
		CUDALibraryGenerate, CUDALibraryFunctionGenerate
	},
	{
		(* NVCC messages *)
		{"invxcmpinstd", "A C compiler was not found in specified directory `1`. Set \"XCompilerInstallation\" to the directory containing a supported C compiler. Supported C compilers are detailed in the " <> GPUTools`Message`MakeDocumentationLink["NVCCCompiler documentation page", "CUDALink/ref/NVCCCompiler"] <> "."},
		{"confarg", "Conflicting arguments were passed in the NVCCCompiler. \"CreateCUBIN\" and \"CreatePTX\" cannot be both be set to True."},
		{"gpucmp", "\"CreateCUBIN\" or \"CreatePTX\" are only valid options to CreateExecutable."},
		{"nonvcci", "An NVIDIA NVCC compiler was not found in specified directory `1`. Set \"CompilerInstallation\" to the directory containing the CUDA Toolkit. Consult " <> GPUTools`Message`MakeDocumentationLink["NVCC Requirements", "CUDALink/tutorial/Setup"] <> " for more information."},
		{"invsys", "The NVIDIA NVCC compiler is not supported on the target system. Supported systems are MacOSX-x86, MacOSX-x86-64, Windows, Windows-x86-64, Linux, and Linux-x86-64."},
		{"invxcomp", "The \"XCompileOptions\" option set to `1` is not valid. \"XCompileOptions\" must be a string or a list of strings."},
		{"invarch", "The \"CUDAArchitecture\" option set to `1` is not valid. \"CUDAArchitecture\" must be a string or a list of strings."},
		{"invcub", "The \"CreateCUBIN\" option set to `1` is not valid. \"CreateCUBIN\" must be either True or False."},
		{"invptx", "The \"CreatePTX\" option set to `1` is not valid. \"CreatePTX\" must be either True or False."},
		{"invunmg", "The \"UnmangleCode\" option set to `1` is not valid. \"UnmangleCode\" must be either True or False."},
		{"invxpth", "The \"XCompilerInstallation\" option set to `1` is not valid. \"XCompilerInstallation\" must be a string pointing to the C compiler directory."},
		{"noxcomp", "A C compiler was not found in specified directory `1`. Set \"XCompilerInstallation\" to the directory containing a supported C compiler. Supported C compilers are detailed in the " <> GPUTools`Message`MakeDocumentationLink["NVCCCompiler documentation page", "CUDALink/ref/NVCCCompiler"] <> "."},
		{"novsarch", "The C compiler used cannot be used to compile on target system. Consult " <> GPUTools`Message`MakeDocumentationLink["NVCC Requirements", "CUDALink/tutorial/Setup"] <> " for more information."},
		{"unmfiles", "The \"UnmangleCode\" option set to True cannot be used to compile `1` source files specified. set \"UnmangleCode\" to False, or pass in the source files one at a time."},
		{"nonvcc", "The `1` Target API requires an unavailable NVIDIA NVCC compiler. Refer to " <> GPUTools`Message`MakeDocumentationLink["NVCC Requirements", "CUDALink/tutorial/Setup"] <> " for more information."}
	}
]

GPUTools`Utilities`DefineMessage[
	{
		CUDASymbolicCGenerate, CUDACodeStringGenerate, CUDACodeGenerate,
		CUDALibraryGenerate, CUDALibraryFunctionGenerate
	},
	{
		{"invtype", "Input `1` is of invalid type."},
		{"invfunn", "Function name `1` is not valid."},
		{"invapi", "Target API `1` specified is not valid."},
		{"invcdtrgt", "Code target `1` specified is not valid."},
		{"invkcopt", "The `1` kernel compile options are not valid."},
		{"invprog", "The input program `1` is not valid."},
		{"invkrnnam", "The kernel name `1` is not valid."}
	}
]

GPUTools`Utilities`DefineMessage[
	{
	 CUDAImageConvolve, CUDABoxFilter, CUDAClamp, CUDAErosion, CUDADilation, CUDAOpening,
	 CUDAClosing, CUDAColorNegate, CUDAImageAdd, CUDAImageSubtract, CUDAImageMultiply, CUDAImageDivide,
	 CUDAFunctionLoad, CUDAMemoryLoad, CUDAFourier, CUDAInverseFourier 
	 
	},
	{
		{"invargs", "Supplied arguments, `1`, are invalid."}
	}
]

GPUTools`Utilities`DefineMessage[
	{
	 CUDAImageConvolve, CUDABoxFilter, CUDAClamp, CUDAErosion, CUDADilation, CUDAOpening,
	 CUDAClosing, CUDAColorNegate, CUDAImageAdd, CUDAImageSubtract, CUDAImageMultiply, CUDAImageDivide
	},
	{
		{"invoutid", "The specified output memory `1` is not valid."},
		{"invbrdr", "The specified border value `1` is not valid."},
		{"invopt", "The `1` option value `2` is not valid."},
		{"invtypv", "Input type is `1`. Only Integer and Real are valid."},
		{"invinpt", "Input is not valid."},
		{"mixdinpttyp", "Input cannot contain mixed types and must be packable."},
		{"invouttyp", "The output type is not valid. The input type `1` and output type `2` must agree."},
		{"invout", "Output is not valid."},
		{"unmchdims", "Unmatching input and output dimensions. The input dimension `1` and output dimension `2` must agree."},
		{"const", "The input constant is invalid. Valid constants must be numeric."},
		{"invcnst", "The input constant is invalid. A valid constant must be either an Integer or a Real."}
	}
]

GPUTools`Utilities`DefineMessage[
	{CUDABoxFilter, CUDAErosion, CUDADilation, CUDAOpening, CUDAClosing},
	{
		{"radneg", "The input radius `1` is not valid. Valid radiuses must be positive."},
		{"zerorad", "The input radius `1` is not valid. Valid radiuses must be positive."},
		{"radlarge", "The specified radius `1` is too large."},
		{"rad", "The input radius `1` is not valid. A valid radius must be either an Integer or a Real."}
	}
]

GPUTools`Utilities`DefineMessage[
	{
		CUDAFourier, CUDAInverseFourier
	},
	{
		{"zerodims", "Input list dimensions `1` has 0 dimensional element."},
		{"onedim", "Input `1` has dimensions `2`. CUDAFourier and CUDAInverseFourier cannot function on an input with one of its dimensions being 1."}
	}
]

GPUTools`Utilities`DefineMessage[
	{
		CUDAClamp, CUDAColorNegate
	},
	{
		{"input", "The input specified is invalid. A valid input must be a List, an Image, or CUDAMemory."},
		{"low", "The bound specified (`1`) is invalid. A valid bound must be Automatic or a Numeric."},
		{"highlow", "The range (`1`, `2`) specified is not valid."},
		{"memvec", "The input specified is invalid. A valid CUDAMemory input may not use a built-in vector type."}
	}
]

CUDAImageConvolve::kernnotmat = "The kernel `1` is not a valid matrix."
CUDAImageConvolve::kernnotodd = "The kernel `1` is not odd, CUDAImageConvolve requires an odd matrix."

CUDAImageConvolve::invkern = "The input kernel is not valid."
CUDAImageConvolve::kernlarge = "The specified kernel dimensions `1` to CUDAImageConvolve is too large."

CUDAFluidDynamics::invnumpts = "The number of particles specified is invalid. It must be a positive integer.";
CUDAFluidDynamics::invimgsiz = "The image size specified is invalid. It must be an integer.";

CUDABoxFilter::invscl = "The scaling factor `1` specified is not valid."
CUDABoxFilter::intscale = "The scaling factor `1` must be an integer."

CUDAImageDivide::divzero = "CUDAImageDivide constant divisor cannot be zero."

CUDAInformation::invopt = "`1` is not a valid query option."

CUDAVolumetricRender::reinitvol = "Volume has already been initialized, cannot initialize twice.";

CUDAMap::op = "Specified operation, `1`, is invalid. Valid operations are: Cos, Sin, Tan, ArcCos, ArcSin, ArcTan, Cosh, Sinh, Exp, Log, Log10, Sqrt, Ceiling, Floor, Abs";

CUDAFold::op = "Specified operation, `1`, is invalid. Valid operations are: Min, Max, Plus, Minus, Times, Divide.";
CUDAFoldList::op = "Specified operation, `1`, is invalid. Valid operations are: Min, Max, Plus, Minus, Times, Divide.";

CUDASort::op = "Specified operation, `1`, is invalid. Valid operations are: Less and Greater.";

toComplex::cmplx = "Input argument is invalid. Valid input types are Integers, Reals, and Complexes.";

GPUTools`Utilities`DefineMessage[
	{
		CUDASort, CUDAFold, CUDAFoldList, CUDAMap
	},
	{
		{"input", "The specified input is invalid. Input must be a List or CUDAMemory."},
		{"dims", "Specified input has invalid dimension. Input must be a Vector or a Matrix for built-in vector types."},
		{"memvec", "Specified input has invalid type. CUDAMemory input may not use a built-in vector type."},
		{"novec", "Specified input has invalid dimension. Input must be 1 dimensional."}
	}
]

GPUTools`Utilities`DefineMessage[
	{
		CUDAFinancialDerivative
	},
	{
		{"invmthd", "The Method option set to `1` is not valid. Accepted methods are \"Binomial\", \"MonteCarlo\", and Automatic. \"Binomial\" supports only American and European options; \"MonteCarlo\" supports only Asian and European options."},
		{"invdmthd", "The \"DerivativeMethod\" set to `1` is not valid. Accepted values are \"Numeric\" and Automatic."},
		{"invsmpl", "The Monte Carlo sampling parameters specified are invalid. \"PathCount\" and \"PathLength\" must both be positive integers, and their product must be even."},
		{"invtree", "The \"TreeDepth\" option set to `1` is not valid. \"TreeDepth\" must be a positive integer."},
		{"invinptsz", "The inputs specified are invalid. Every input must be either a real number of a list of real numbers, and all lists must be the same length."},
		{"invoptt", "The option type `1` specified is invalid. The option type should be a doublet specifying the type of option, put or call, and optionally the derivative to be calculated."},
		{"invnderiv", "The numerical derivative type `1` specified is only supported with Black Scholes for European vanilla options. Binomial and Monte Carlo method support only the first order numerical derivatives (\"Delta\", \"Vega\", \"Theta\", \"Rho\")."},
		{"invinptrl", "The input `1` is not in the form of a rule."},
		{"invinptstr", "The string `1` is not recognized as a valid parameter."},
		{"invinpt", "The input `1` is invalid. It must be either a real number or a list of real numbers."},
		{"msnginpt", "A necessary parameter has not been supplied. All calls to CUDAOptionsPricing must include StrikePrice, CurrentPrice, and Expiration. Additionally, in the case of a Barrier option, the Barrier must be supplied."},
		{"invprop", "The specified property to calculate, `1`, is invalid. Valid properties are \"Value\" and any sensitivity."},
		{"invparamsl", "The input parameters are invalid. Valid parameters must be a list."},
		{"invambparamsl", "The input ambient parameters are invalid. Valid ambient parameters must be a list."},
		{"invparamsr", "The input parameters are invalid. Each element of the parameter list must be a Rule."},
		{"invambparamsr", "The input ambient parameters are invalid. Each element of the ambient parameter list must be a Rule."},
		{"invparams", "One or more of the input parameters is invalid. Valid parameters are: StrikePrice, Expiration, Barriers."},
		{"invambparams", "One or more of the input ambient parameters is invalid. Valid ambient parameters are: CurrentPrice, Dividend, Volatility, InterestRate, ExchangeRate, ExchangeVolatility, ForeignInterestRate, Correlation, Rebate."}
	}
]

GPUTools`Utilities`DefineMessage[
	{
		CUDADot, CUDAArgMinList, CUDAArgMaxList, CUDATotal
	},
	{
		{"inpt", "The input supplied `1` and `2` are invalid. CUDADot supports matrix-matrix, matrix-vector, and vector-vector products."},
		{"nobls", "Failed to allocate CUDA memory `1` and `2`, or is not a valid operation."},
		{"inptvec", "The input supplied `1` is invalid. Valid input must be a vector."}
	}
]

GPUTools`Utilities`DefineMessage[
	{
		CUDAVolumetricRender
	},
	{
		{"invwidth", "The specified width `1` is invalid. Width must be a positive integer."},
		{"invheight", "The specified height `1` is invalid. Height must be a positive integer."},
		{"invvol", "Specified volume data `1` is invalid. Volume data must be a List."},
		{"init", "Failed to initialize."}
	}
]

GPUTools`Utilities`DefineMessage[
	$Symbols,
	{
		{"invdevid", "`1` is not a valid device id."},
		{"imgpadm", "`1` is not a valid image padding specification."},
		{"omem", "Invalid \"OutputMemory\" `1`. \"OutputMemory\" must be None or a CUDAMemory object."},
		{"omemc", "Invalid \"OutputMemory\" `1`. \"OutputMemory\" must be None or a CUDAMemory object of complex type."},
		{"unbladd", "Failed to add to CUDA memory `1`."},
		{"noinpt", "Invalid input `1`. Input must be non-empty."},
		{"update", "Update -> `1` is not valid. Accepted values for Update are True or False."},
		{"pacletold", "CUDAResources paclet installed version `1` is too old. A minimum version of `2` is required."},
		{"pacletpth", "CUDAResources paclet path `1` is not valid."},
		{"nopaclet", "CUDAResources was not found. Make sure that you are connected to the internet and Mathematica is allowed access to the internet."},
		{"pacletdown", "Failed to download the `1` paclet. Make sure that you are connected to the internet and Mathematica is allowed access to the internet."},
		{"nocudares", "CUDAResources was not installed. CUDAResources is required for CUDALink to work. " <> GPUTools`Message`ReferToSystemRequirements["CUDALink"]},
		{"empty", "Input list is empty."}
	}
]

CUDADriverVersion::nodriv = "CUDALink was not able to locate the NVIDIA driver binary. " <> GPUTools`Message`ReferToSystemRequirements["CUDALink"]


(******************************************************************************)
(******************************************************************************)
(******************************************************************************)

SetAttributes[CUDAProfile, HoldAll]
SetAttributes[Evaluate@$Symbols, {ReadProtected, Protected}]


	
(******************************************************************************)

Quiet[
	With[{
		paclets = PacletFind[$CUDAResourcesPacletName];
	},
		If[paclets =!= {},
			With[{
				paclet = First[paclets]
			},
				PrependTo[
					$LibraryPath,
					FileNameJoin[{
						"Location" /. PacletInformation[paclet],
						"LibraryResources",
						$SystemID
					}]
				]
			]
		]
	]
]


(******************************************************************************)
(******************************************************************************)
(******************************************************************************)
End[] (* End Private Context *)

EndPackage[] (* CUDALink *)

