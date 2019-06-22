(* Mathematica Package *)

(* Created by the Wolfram Workbench Feb 10, 2010 *)

BeginPackage["GPUTools`CodeGenerator`", {"SymbolicC`", "CCompilerDriver`", "GPUTools`Utilities`"}]

$GPUCodeGenerateOptions::usage = "$GPUCodeGenerateOptions  "

iGPUSymbolicCGenerate::usage = "iGPUGenerateSymbolicC  "

iGPUCodeGenerate::usage = "iGPUGenerateCode  "

iGPUCodeStringGenerate::usage = "iGPUGenerateCodeString  "

iGPULibraryFunctionGenerate::usage = "iGPUGenerateLibraryFunction  "

iGPULibraryGenerate::usage = "iGPUGenerateLibrary  "

GPUTools`Utilities`DefineMessage[
	{
		iGPUSymbolicCGenerate,
		iGPUCodeGenerate,
		iGPUCodeStringGenerate,
		iGPULibraryFunctionGenerate,
		iGPULibraryGenerate
	},
	{
		{"invtype", "Input `1` is of invalid type."},
		{"invfunn", "Function name `1` is not valid."},
		{"invapi", "Target API `1` specified is not valid."},
		{"invcdtrgt", "Code target `1` specified is not valid."},
		{"nonvcc", "The `1` Target API requires an unavailable NVIDIA NVCC compiler. Refer to " <> GPUTools`Message`MakeDocumentationLink["NVCC System Requirements", "CUDALink/ref/NVCCCompiler"] <> " for more information."},
		{"invkcopt", "The `1` kernel compile options specified are not valid."},
		{"invprog", "The input program `1` specified is not valid."},
		{"invkrnnam", "The kernel name `1` specified is not valid."}
	}
]

Begin["`Private`"]

Needs["CCompilerDriver`"]
Needs["CUDALink`NVCCCompiler`"]

$TabCharacter = "\t"

	
(* ::Section:: *)
(* API Names *)
$APIs = {
	$WolframLibrary = "WolframLibrary",
	$WolframRTL = "WolframRTL",
	$WolframRTLHeader = "WolframRTLHeader",
	$WGLCUDA = "WGLCUDA",
	$WGLOpenCL = "WGLOpenCL",
	$C = "C"
}

(* ::Section:: *)
(* Default Values *)

defaults["True" | True] := "True"
defaults["False" | False] := "False"
defaults["Runtime"] := "wriRuntime"
defaults["RuntimeInitializeQ"] := "wriRuntimeInitializeQ"
defaults["WolframCompile"] := "wriCompile"
defaults["libData"] := "libData"
defaults["wglData"] := "wglData"
defaults["GPUKernel"] := "gpuKernel"
defaults["GPUKernelProgramHandle"] := "gpuProgHandle"
defaults["GPUDeviceHandle"] := "gpuDevice"
defaults["GPUPlatformHandle"] := "gpuPlatform"
defaults["ErrorName"] := "libErr"
defaults[$C | $WolframRTL | $WolframRTLHeader | $WolframLibrary, "ErrorName"] := defaults["ErrorName"]
defaults[$WGLCUDA | $WGLOpenCL, "ErrorName"] := wglGetError[]
defaults["ChooseFastestGPUDeviceFunction"] := "chooseFastestGPUDevice"
defaults["CommandQueue"] := "queue"
defaults["GPUKernelHandle"] := "gpuKernelHandle"
defaults["DeviceNumberName"] := "deviceId"
defaults["ElementSize"] := "elemSize"
defaults["BlockName"] := "gpuBlock"
defaults["GridName"] := "gpuGrid"
defaults["KernelFunction"] := "gpuKernelFunction"
defaults["IncludeHeaders"] := {}
defaults["IncludeDebugHeaders"] := "stdio.h"
defaults["Defines"] := {}

(* ::Section:: *)
(* Header *)


includeHeaders[debugQ_] :=
	Flatten[If[TrueQ[debugQ],
		{CComment["Debugging Include Files"], includeHeaders[defaults["IncludeDebugHeaders"]]},
		includeHeaders[defaults["IncludeHeaders"]]
	]]
includeHeaders[h_String] :=
	CInclude[h]
includeHeaders[{}] := {}
includeHeaders[hs_List] :=
	CInclude[Select[hs, # =!= {} && # =!= "" &]]
	
includeHeaders[funName_, apiTarget_, codeTarget_, debugQ_] :=
	If[codeTarget === $WolframRTL,
		CInclude[funName <> ".h"],
		Flatten[{
			includeHeaders[debugQ],
			If[codeTarget === $WolframRTLHeader,
				apiIncludeHeaders[$WolframRTLHeader],
				apiIncludeHeaders[$WolframLibrary]
			],
			apiIncludeHeaders[apiTarget]
		}]
	]

apiIncludeHeaders[api:($WolframLibrary|$WolframRTLHeader|$WGLCUDA|$WGLOpenCL)] :=
	{CComment[ToString[api] <> " Include Files"], 
		Switch[api,
			$WolframRTLHeader,
				CInclude[{"WolframRTL.h"}],
			$WolframLibrary,
				CInclude[{"WolframLibrary.h", "WolframCompileLibrary.h"}],
			$WGLCUDA | $WGLOpenCL,
				CInclude["wgl.h"]
		]
	}

(* ::Section:: *)
(* Defines *)

defines[] :=
	Default["Defines"]
defines[x_] :=
	{CDefine[x]}
defines[x_List] :=
	CDefine /@ x

(* ::SubSection:: *)
(* Macros *)

defineMacros[debugQ_] :=
	Flatten[If[TrueQ[debugQ],
		{CComment["Debugging Macros"],
		 CDefine["WGL_SAFE_CALL(stmt)     stmt; if (wglData->getError(wglData)->code != WGL_Success) { goto cleanup; }"] 
		}
	]]

(* ::Section:: *)
(* Types Functions *)
typeData[x:(_Integer|_Real)] := typeName[x]
typeId[x:(_Integer|_Real)] := typeName[x]
typeData[x_mtensor] := mtensorName[x] <> "_data"
typeData[x_memory] := memoryName[x] <> "_data"
typeId[x_memory] := memoryName[x] <> "_id"
typeGPUData[x:(_Integer|_Real)] := typeName[x]
typeGPUData[x_mtensor] := mtensorGPUData[x]
typeGPUData[x_memory] := memoryGPUData[x]

typeIntermediateData[x_Integer] := typeName[x]
typeIntermediateData[x_Real] :=
	"f_" <> typeName[x]
typeIntermediateData[x_mtensor] :=
	If[mtensorType[x] === Real,
		"f_",
		""
	] <> typeData[x]

typeName[x:(_mtensor|_Integer|_Real|_memory)] := First[x]

mtensorQ[_mtensor] := True
mtensorQ[___] := False
mtensorName[x_mtensor] := typeName[x]
mtensorLength[x_mtensor] := mtensorGetFlattenedLength[mtensorName[x]]
mtensorLength[x_String] := mtensorGetFlattenedLength[x]
mtensorRank[x_mtensor] := x[[3]]
mtensorType[mtensor[_, type_, ___]] := type
mtensorData[x_mtensor] := mtensorData[mtensorName[x], mtensorType[x]]
mtensorData[x_, type_] :=
	Switch[type,
		"mint" | Integer, mtensorGetIntegerData[x],
		"mreal" | "double" | Real, mtensorGetIntegerData[x],
		_, "MTensorDataType not supported"
	]
mtensorGPUData[x_mtensor] :=
	"gpu_" <> ToString[mtensorName[x]] <> "_data"
mtensorCType[x_mtensor] :=
	Switch[mtensorType[x],
		Integer, "mint",
		Real, "mreal",
		_, mtensorType[x]
	] 

memoryQ[_memory] := True
memoryQ[___] := False
memoryName[x_memory] := typeName[x]
memoryRank[memory[_, _, rank_, ___]] := rank
memoryType[memory[_, type_, ___]] := type
memoryIdType[_memory] := "mint"

memoryGetIntegerData[debugQ_, x_memory] :=
	wglGetIntegerData[debugQ, memoryName[x]]
memoryGetRealData[debugQ_, x_memory] :=
	wglGetRealData[debugQ, memoryName[x]]
memoryGetComplexData[debugQ_, x_memory] :=
	wglGetComplexData[debugQ, memoryName[x]]

memoryGetFlattenedLength[x_memory] :=
	CPointerMember[memoryName[x], "flattenedLength"]
memoryGetRank[x_memory] :=
	CPointerMember[memoryName[x], "rank"]
memoryGetType[x_memory] :=
	CPointerMember[memoryName[x], "type"]
	
outputParamQ[var:(_Integer|_Real|_mtensor)] := Last[var]
outputParamQ[var_memory] := MemberQ[{"Output", "InputOutput"}, Last[var]]

outputParam[var:(_Integer|_Real|_mtensor|_memory)] := Last[var]
	
(* ::Section:: *)
(* Mathematica Helper Functions *)


stripDuplicates[{}] := {}
stripDuplicates[{r : Rule[x_, y_], xs___}] :=
	Prepend[stripDuplicates[Select[{xs}, First[#] =!= x &]], r]

removeNulls[{ii_List}] := removeNulls[ii]
removeNulls[x_List] := Select[x, # =!= Null &]
removeNulls[x___] := x

(* ::Section:: *)
(* SymbolicC Helper Functions *)

externFunction[typeArg_, id_, args_List,body_] := CFunction[{"EXTERN_C", typeArg}, id, args, body]
CExtern[p___] := CDeclare["EXTERN_C", cBlock[{p}]];
CPrintf[msg_] := CCall["printf", CString[msg]]
CPrintf[sig_, args__] := CCall["printf", Join[{CString[sig]}, List[args]]] 
CAssert[cond_] := CCall["WGL_assert", cond]
cBlock[x_List] := CBlock[removeNulls[x]]
cBlock[x___] := CBlock[removeNulls[List[x]]]

cIf[cond_, x_] :=
	CIf[cond,
		If[Head[x] === cBlock,
			x,
			cBlock[If[ListQ[x], x, {x}]]
		]
	]
cIf[cond_, x_, y_] :=
	Apply[CIf,
		Flatten[Join[{
			{
				cond,
				If[Head[x] === cBlock,
					x,
					cBlock[If[ListQ[x], x, {x}]]
				]
			},
			{
				If[y === Null,
					{},
					If[Head[y] === cBlock,
						y,
						cBlock[If[ListQ[y], y, {y}]]
					]
				]
			}
		}]]
	]

(* ::Section:: *)
(* Wolfram Runtime Helper Functions *)

rtlInitialize[version_] := CCall["WolframRTL_initialize", {version}]
rtlNewWolframLibraryFunction[version_] := CCall["WolframLibraryData_new", {version}]
rtlFreeWolframLibraryFunction[data_] := CCall["WolframLibraryData_free", {data}]

(* ::Section:: *)
(* Wolfram Compile Library Helper Functions *)

wolframCompileLibraryDataCleanup[libData_, x_] :=	
	CCall[CPointerMember[defaults["WolframCompile"], "WolframLibraryData_cleanUp"], {libData, x}]

(* ::Section:: *)
(* LibraryLink Helper Functions *)

libraryFunStructCall[fun_, body_:{}] := 
	CCall[CPointerMember["libData", fun], body]
libraryGetCompileLibraryFunctions[] :=
	libraryFunStructCall["compileLibraryFunctions"]
libraryVersionNumber[] :=
	libraryFunStructCall["VersionNumber"]
librariestringDisown[str_] :=
	libraryFunStructCall["UTF8String_disown", str]	
libraryMessage[msg_] := 
	libraryFunStructCall["Message", CString[msg]]
libraryAbortQ[] := 
	libraryFunStructCall["AbortQ"]
libraryGetMathLink[] := 
	libraryFunStructCall["getMathLink"]
libraryProcessMathLink[link_] := 
	libraryFunStructCall["processMathLink", link]
	
libraryVersionInformationFunction[] :=
	{
		externFunction[{"DLLEXPORT", "mint"}, "WolframLibrary_getVersion", {},
			cBlock[{
				CReturn["WolframLibraryVersion"]
			}]
		]
	}
	
declareWGLData[] :=
	CDeclare[
		{"static", "WolframGPULibraryData"}, CAssign[defaults["wglData"], CCast["WolframGPULibraryData", 0]]
	]
declareFunStruct[] :=
	CDeclare[
		{"static", "WolframLibraryData"}, CAssign[defaults["libData"], CCast["WolframLibraryData", 0]]
	]
declareRuntimeData[] :=
	CDeclare[
		{"static", "WolframRTLData"}, CAssign[defaults["Runtime"], CCast["WolframRTLData", 0]]
	]
declareWolframCompileData[] :=
	CDeclare[
		{"static", "WolframCompileLibrary_Functions"}, CAssign[defaults["WolframCompile"], CCast["WolframCompileLibrary_Functions", 0]]
	]
	
libraryFunction[funName_, body___] :=
	CFunction[{"EXTERN_C", "DLLEXPORT", "int"}, funName, {{"WolframLibraryData ", "libData"}, {"mint", "ArgC"}, {CPointerType["MArgument"], "Args"}, {"MArgument", "Res"}},
		cBlock[{
			body
		}]
	]

libraryDeclareInputArguments[apiTarget_, codeTarget_][params_List] :=
	libraryDeclareInputArgument[apiTarget, codeTarget] /@ params

libraryDeclareInputArgument[$WGLCUDA | $WGLOpenCL, codeTarget_][param_memory] :=
	{
		CDeclare[memoryIdType[param], typeId[param]],
		CDeclare[cType[param], typeName[param]]
	}
	
libraryDeclareInputArgument[apiTarget_, codeTarget_][param_] :=
	CDeclare[cType[param], typeName[param]]

libraryAssignInputArguments[apiTarget_, codeTarget_][params_List] :=
	MapIndexed[libraryAssignInputArgument[apiTarget, codeTarget], params]

libraryAssignInputArgument[$WGLCUDA | $WGLOpenCL, codeTarget_][arg_, {idx_}] :=
	CAssign[typeId[arg], mArgumentGet[arg, idx-1]]

libraryAssignInputArgument[apiTarget_, codeTarget_][arg_, {idx_}] :=
	CAssign[typeName[arg], mArgumentGet[arg, idx-1]]

wglFindMemory[apiTarget_, codeTarget_][params_List] :=
	removeNulls[wglFindMemory[apiTarget, codeTarget] /@ params]

wglFindMemory[apiTarget_, codeTarget_][mem_memory] :=
	CAssign[typeName[mem], wglFindMemory[typeId[mem]]]
wglFindMemory[apiTarget_, codeTarget_][___] := Null

libraryAssignOutputArguments[params_List] :=
	mArgumentSet /@ Select[params, outputParamQ]
	
libraryDisownTensors[params_List] :=
	mtensorDisownAll /@ Select[params, mtensorQ]

parameterFunctionSignature[codeTarget_][blockName_, gridName_, params_List] :=
	Join[
		If[codeTarget === $WolframRTL || codeTarget === $WolframRTLHeader, {}, {{"WolframLibraryData ", "libData"}}],
		{cType[#], typeName[#]}& /@ params,
		{
			{"MTensor", blockName},
			{"MTensor", gridName}
		}
	]
parameterFunctionCall[name_, blockName_, gridName_, params_List] :=
	CCall[name,
		Join[
			{"libData"},
			typeName /@ params,
			{blockName, gridName}
		]
	]

(* ::Section:: *)
(* WGL Interface *)

wolframGPULibraryNew[ver_:"WGL_VERSION"] := CCall["WolframGPULibrary_New", {ver}]
wolframGPULibraryFree[wglData_:"wglData"] := CCall["WolframGPULibrary_Free", {wglData}]

wglQ[apiTarget_] := MemberQ[{$WGLCUDA, $WGLOpenCL}, apiTarget]

wglAPI[$WGLCUDA] := "WGL_API_CUDA_DRIVER"
wglAPI[$WGLOpenCL] := "WGL_API_OpenCL"
wglAutomatic[] := "WGL_Automatic"
										
											
wglCall[debugQ_, fun_, args_:{}] :=
	If[debugQ,
		CCall["WGL_SAFE_CALL", CCall[CPointerMember["wglData", fun], Flatten[{"wglData", args}]]]
		,
		CCall[CPointerMember["wglData", fun], Flatten[{"wglData", args}]]
	]
	
wglSuccessQ[] :=
	COperator[Equal, {CPointerMember[wglGetError[], "code"], "WGL_Success"}]

wglPrintError[msg_, args_List] :=
	CCall["fprintf", {"stderr"} ~Join~ {msg} ~Join~ args]
wglPrintErrorMsg[] :=
	wglPrintError["\"Error Message: %s\\n\"", {"wglData->getError(wglData)->msg"}]
wglPrintErrorFunctionName[] := 
	wglPrintError["\"Function Name: %s\\n\"", {"wglData->getError(wglData)->funName"}]
wglPrintErrorBuildLog[] :=
	wglPrintError["\"Build log: %s\\n\"", {"wglData->getProgramBuildLog(wglData, prog)"}]

wglPrintErrorFileInfo[] :=
	wglPrintError["\"File::ln %s::%ld\\n\"", {"wglData->getError(wglData)->fileName", "wglData->getError(wglData)->lineNumber"}]
wglPrintErrorDimensions[tensor_] :=
	CSwitch[mtensorLength[tensor],
		1,
			{
				wglPrintError["\"%s Length: 1\\n Tensor Data: %ld\\n\"", {"\"" <> ToString[tensor] <> "\"", CArray[mtensorData[tensor, Integer], 0]}],
				CBreak[]
			},
		2,
			{
				wglPrintError["\"%s Length: 2\\n Tensor Data: {%ld, %ld}\\n\"", {"\"" <> ToString[tensor] <> "\"", CArray[mtensorData[tensor, Integer], 0],
					CArray[mtensorData[tensor, Integer], 1]}],
				CBreak[]
			},
		3,
			{
				wglPrintError["\"%s Length: 3\\n Tensor Data: {%ld, %ld, %ld}\\n\"", {"\"" <> ToString[tensor] <> "\"", CArray[mtensorData[tensor, Integer], 0],
					CArray[mtensorData[tensor, Integer], 1], CArray[mtensorData[tensor, Integer], 2]}],
				CBreak[]
			}
		]
		
wglPrintErrorInputArguments[apiTarget_, codeTarget_][params_List] :=
	MapIndexed[wglPrintErrorInputArgument[apiTarget, codeTarget], params]

wglPrintErrorInputArgument[$WGLCUDA | $WGLOpenCL, codeTarget_][arg_, {idx_}] :=
	wglPrintError["\"%s = %ld\\n\"", {"\"" <> ToString[typeId[arg]] <> "\"", mArgumentGet[arg, idx-1]}]

wglGetVersion[debugQ_] :=
		wglCall[debugQ, "getAPI"]
wglInitialize[debugQ_, api:($WGLCUDA | $WGLOpenCL)] :=
	wglCall[debugQ, "initialize", wglAPI[api]]
wglUnitialize[debugQ_] :=
		wglCall[debugQ, "uninitialize"]
wglSetWolframLibraryData[debugQ_, libData_:"libData"] :=
	wglCall[debugQ, "setWolframLibraryData", libData]
(* TODO: Turn on Debugging *)
Options[wglGetError] := {"Debug"->False} 
wglGetError[opts:OptionsPattern[]] :=
	Module[{debugQ = TrueQ@OptionValue["Debug"]}, 
		wglCall[debugQ, "getError"]
	] 
wglClearError[debugQ_] :=
	wglCall[debugQ, "clearError"]
wglSetPlatform[debugQ_, plat_:Automatic] :=
	wglCall[debugQ, "setPlatform", If[plat === Automatic, wglAutomatic[], plat]]
wglSetDevice[debugQ_, dev_:Automatic] :=
	wglCall[debugQ, "setDevice", If[dev === Automatic, wglAutomatic[], dev]]
wglSetProgram[debugQ_, prog_] :=
		wglCall[debugQ, "setProgram", prog]
wglNewPropgramFromSource[debugQ_, src_, buildOptions_] :=
	CAssign["WGL_Program_t prog", wglCall[debugQ, "newProgramFromSource", {src, If[buildOptions === Null, "NULL", buildOptions]}]]
wglNewPropgramFromSourceFile[debugQ_, src_, buildOptions_] :=
	wglCall[debugQ, "newProgramFromSourceFile", {src, If[buildOptions === Null, "NULL", buildOptions]}]
wglNewPropgramFromBinary[debugQ_, prog_, size_] :=
	wglCall[debugQ, "newProgramFromBinary", {prog, size}]
wglNewPropgramFromBinaryFile[debugQ_, src_] :=
	wglCall[debugQ, "newProgramFromBinaryFile", src]
wglGetProgramBuildLog[debugQ_, prog_] :=
	wglCall[debugQ, "getProgramBuildLog", prog]
wglSetKernel[debugQ_, kernelName_] :=
	wglCall[debugQ, "setKernel", kernelName]
wglSetKernelIntegerArgument[debugQ_, arg_, opts:OptionsPattern[]] :=
	wglCall[debugQ, "setKernelIntegerArgument", arg]
wglSetKernelLongArgument[debugQ_, arg_] :=
	wglCall[debugQ, "setKernelLongArgument", arg]
wglSetKernelFloatArgument[debugQ_, arg_] :=
	wglCall[debugQ, "setKernelFloatArgument", arg]
wglSetKernelDoubleArgument[debugQ_, arg_] :=
	wglCall[debugQ, "setKernelDoubleArgument", arg]
wglSetKernelMemoryArgument[debugQ_, arg_, cpy_] :=
	Module[{io},
		io = Switch[cpy,
			"Input",
				"WGL_Memory_Argument_Input",
			"Output",
				"WGL_Memory_Argument_Output",
			"InputOutput",
				"WGL_Memory_Argument_InputOutput"
		];
		wglCall[debugQ, "setKernelMemoryArgument", {arg, io}]
	]
wglSetKernelRawElementArgument[debugQ_, arg_, size_] :=
	wglCall[debugQ, "setKernelRawElementArgument", arg, size]
wglSetKernelLocalMemoryArgument[debugQ_, size_] :=
	wglCall[debugQ, "setKernelLocalMemoryArgument", size]
wglSetBlockDimensions[debugQ_, dim_Integer] :=
	wglSetBlockDimensions[debugQ, {dim}]
wglSetBlockDimensions[debugQ_, dims_List] :=
	wglCall[debugQ, "setBlockDimensions", Length[dims], Sequence@@dims]
wglSetBlockDimensions[debugQ_, dim_, x_, y_:1, z_:1] :=
	wglCall[debugQ, "setBlockDimensions", {dim, x, y, z}]
wglSetGridDimensions[debugQ_, dim_Integer] :=
	wglSetGridDimensions[debugQ, {dim}]
wglSetGridDimensions[debugQ_, dims_List] :=
	wglCall[debugQ, "setGridDimensions", Length[dims], Sequence@@dims]
wglSetGridDimensions[debugQ_, dim_, x_, y_:1, z_:1] :=
	wglCall[debugQ, "setGridDimensions", {dim, x, y, z}]
wGLLaunchKernel[debugQ_] :=
	wglCall[debugQ, "launchKernel"]
wglSynchronize[debugQ_] :=
	wglCall[debugQ, "synchronize"]
wglNewTensorMemory[debugQ_, tens_, type_, res_, shr_, uniq_] :=
	wglCall[debugQ, "newTensorMemory", {tens, type, res, shr, uniq}]
wglNewRawMemory[debugQ_, hm_, res_, uniq_] :=
	wglCall[debugQ, "newRawMemory", {hm, res, uniq}]
wglGetIntegerData[debugQ_, mem_] :=
	wglCall[debugQ, "MTensorMemory_getIntegerData", {mem}]
wglGetRealData[debugQ_, mem_] :=
	wglCall[debugQ, "MTensorMemory_getRealData", {mem}]
wglGetComplexData[debugQ_, mem_] :=
	wglCall[debugQ, "MTensorMemory_getComplexData", {mem}]
	
	
Options[wglGetHostData] := {"Debug"->False}
wglGetHostData[mem_, opts:OptionsPattern[]] :=
	Module[{debugQ = TrueQ@OptionValue["Debug"]},
		wglCall[debugQ, "MTensorMemory_getHostData", {mem}]
	]
	
(* FIXME: enable debugging *)
wglCopyMemoryToDevice[mem_, force_:True] :=
	wglCall[False, "copyMemoryToDevice", {mem, ToString[force]}]
wglCopyMemoryToHost[mem_, force_:True] :=
	wglCall[False, "copyMemoryToHost", {mem, ToString[force]}]
	

Options[wglFindMemory] := {"Debug"->False}
wglFindMemory[id_] :=
	Module[{debugQ = TrueQ@OptionValue["Debug"]},
		wglCall[debugQ, "findMemory", {id}]
	]
	
Options[WGLFreeMemory] := {"Debug"->False}
WGLFreeMemory[mem_] :=
	Module[{debugQ = TrueQ@OptionValue["Debug"]},
		wglCall[debugQ, "freeMemory", {mem}]
	]

wglLoadProgram[$WGLCUDA, codeTarget_][prog_, buidOpts_, debugQ_] := 
	If[StringQ[prog] || ListQ[prog],
		wglNewPropgramFromBinary[debugQ, defaults["GPUKernel"], defaults["GPUKernel"] <> "Size"],
		Message[General::invprog, prog]
	] 

wglLoadProgram[$WGLOpenCL, codeTarget_][prog_, buidOpts_, debugQ_] := 
	If[StringQ[prog] || ListQ[prog],
		wglNewPropgramFromSource[debugQ, defaults["GPUKernel"], If[StringQ[buidOpts], buidOpts, Null]],
		Message[General::invprog, prog]
	]
	
Options[wglSetKernelArguments] := {"Debug"->False}
wglSetKernelArguments[params_, singlePrecisionQ_, opts:OptionsPattern[]] :=
	Module[{debugQ = TrueQ@OptionValue["Debug"]},
		wglSetKernelArgument[debugQ, #, singlePrecisionQ]& /@ params
	]
	
wglSetKernelArgument[debugQ_, arg_Integer, _] :=
	wglSetKernelLongArgument[debugQ, typeName[arg]]
wglSetKernelArgument[debugQ_, arg_Real, singlePrecisionQ_] :=
	If[singlePrecisionQ,
		wglSetKernelFloatArgument[debugQ, typeName[arg]],
		wglSetKernelDoubleArgument[debugQ, typeName[arg]]
	]
	
wglSetKernelArgument[debugQ_, arg_memory, _] :=
		wglSetKernelMemoryArgument[debugQ, typeName[arg], outputParam[arg]]

setGridAndBlockDimensions[debugQ_, gridTensor_, blockTensor_] :=
	{
		setGridDimensions[debugQ, gridTensor],
		setBlockDimensions[debugQ, blockTensor]
	}

setGridDimensions[debugQ_, gridTensor_] :=
	setLaunchDimensions[debugQ, wglSetGridDimensions, gridTensor]
setBlockDimensions[debugQ_, gridTensor_] :=
	setLaunchDimensions[debugQ, wglSetBlockDimensions, gridTensor]	
setLaunchDimensions[debugQ_, setFun_, tensor_] :=
	CSwitch[mtensorLength[tensor],
		1,
			{
				setFun[debugQ, 1, CArray[mtensorData[tensor, Integer], 0]],
				CBreak[]
			},
		2,
			{
				setFun[debugQ, 2, CArray[mtensorData[tensor, Integer], 0], CArray[mtensorData[tensor, Integer], 1]],
				CBreak[]
			},
		3,
			{
				setFun[debugQ, 3, CArray[mtensorData[tensor, Integer], 0], CArray[mtensorData[tensor, Integer], 1], CArray[mtensorData[tensor, Integer], 2]],
				CBreak[]
			}
	]
(* ::Section:: *)
(* Interface Utils *)

declareWGLMemories[params_List] :=
	Flatten[{
		CDeclare["mint", typeId[#]]& /@ Select[params, mtensorQ],
		CDeclare["WGL_Memory_t", typeName[#]]& /@ Select[params, mtensorQ]
	}]

declareTensorData[params_List] :=
	CDeclare[CPointerType[cType[mtensorType[#]]], CAssign[typeData[#], "NULL"]]& /@ Select[params, mtensorQ]

declareIntermediateData[params_List] :=
	Function[{arg},
		CDeclare[CPointerType["float"], CAssign[typeIntermediateData[arg], "NULL"]] 
	]/@ Select[params, (mtensorQ[#] && mtensorType[#] === Real)&]
assignTensorData[params_List] :=
	CAssign[typeData[#], mtensorGetData[#]]& /@ Select[params, mtensorQ]
allocGPUData[api_, functionPrefix_, useSinglePrecisionQ_, params_List, debugQ_:False] :=
	Module[{type},
		Function[{arg},
			type = If[useSinglePrecisionQ === True && mtensorType[arg] === Real,
				"float",
				cType[mtensorType[arg]]
			];
			memoryAlloc[api,
				defaults[api, "ErrorName"], functionPrefix<>typeGPUData[arg],
				COperator[Times, {CSizeOf[type], mtensorLength[arg]}], "Debug"->debugQ
			] 
		]/@ Select[params, mtensorQ]
	]
copyGPUData[api_, direction_, functionPrefix_, useSinglePrecisionQ_, params_List, debugQ_:False] :=
	Module[{host, src, dst, type},
		Function[{arg},
			{type, host} = If[useSinglePrecisionQ === True && mtensorType[arg] === Real,
				{"float", typeIntermediateData[arg]},
				{cType[mtensorType[arg]], typeData[arg]}
			];
			{src, dst} = If[direction === "HostToDevice",
				{host, functionPrefix<>typeGPUData[arg]},
				{functionPrefix<>typeGPUData[arg], host}
			];
			If[direction === "HostToDevice" || (direction === "DeviceToHost" && outputParamQ[arg]),
				MemoryCopy[api, src, dst, COperator[Times, {CSizeOf[type], mtensorLength[arg]}], direction, "Debug"->debugQ],
				{}
			] 
		]/@ Select[params, mtensorQ]
	]

allocIntermediateData[params_List] :=
	Function[{arg},
		memoryAlloc[$C,
			defaults[$C, "ErrorName"], typeIntermediateData[arg], "float",
			COperator[Times, {CSizeOf["float"], mtensorLength[arg]}]
		] 
	]/@ Select[params, (mtensorQ[#] && mtensorType[#] === Real)&]
freeIntermediateData[params_List] :=
	Function[{arg},
		memoryFree[$C, typeIntermediateData[arg]] 
	]/@ Select[params, (mtensorQ[#] && mtensorType[#] === Real)&]
copyToIntermediateData[params_List] :=
	copyIntermedateData[
		cBlock[{
			CAssign[
				CArray[typeIntermediateData[#1], #2],
				CCast["float",
					CArray[typeData[#1], #2]
				]
			]
		}]&, params
	]
copyFromIntermediateData[params_List] :=
	copyIntermedateData[
		cBlock[{
			CAssign[
				CArray[typeData[#1], #2],
				CCast["mreal",
					CArray[typeIntermediateData[#1], #2]
				]
			]
		}]&, params
	]
copyIntermedateData[body_, params_List] :=
	Module[{name},
		Function[{arg},
			name = "ii";
			CFor[CDeclare["mint", CAssign[name, 0]], COperator[Less, {name, mtensorLength[arg]}], COperator[Increment, {name}],
				body[arg, name]
			] 
		]/@ Select[params, (mtensorQ[#] && mtensorType[#] === Real)&]
	]
(* ::Subsection:: *)
(* Tensor Utils *)
mtensorGetRank[x_mtensor] :=
	mtensorGetRank[mtensorName[x]]
mtensorGetRank[arg_] :=
	libraryFunStructCall["MTensor_getRank", {arg}]
mtensorGetDimensions[x_mtensor] :=
	mtensorGetDimensions'[mtensorName[x]]
mtensorGetDimensions[arg_] :=
	libraryFunStructCall["MTensor_getDimensions", {arg}]
mtensorGetType[x_mtensor] :=
	mtensorGetType[mtensorName[x]]
mtensorGetType[arg_] :=
	libraryFunStructCall["MTensor_getType", {arg}]
mtensorGetFlattenedLength[x_mtensor] :=
	mtensorGetFlattenedLength[mtensorName[x]]
mtensorGetFlattenedLength[arg_] :=
	libraryFunStructCall["MTensor_getFlattenedLength", {arg}]

mtensorGetType[x_mtensor] :=
	mtensorGetType[mtensorName[x]]
mtensorGetType[arg_] :=
	libraryFunStructCall["MTensor_getType", {arg}]

mtensorShareCount[x_mtensor] :=
	mtensorShareCount[mtensorName[x]]
mtensorShareCount[arg_] :=
	libraryFunStructCall["MTensor_shareCount", {arg}]
mtensorFree[x_mtensor] :=
	mtensorFree[mtensorName[x]]
mtensorFree[arg_] :=
	libraryFunStructCall["MTensor_free", {arg}]
mtensorClone[x_mtensor] :=
	mtensorClone[mtensorName[x]]
mtensorClone[arg_] :=
	libraryFunStructCall["MTensor_clone", {arg}]
mtensorDisown[x_mtensor] :=
	mtensorDisown[mtensorName[x]]	
mtensorDisown[arg_] :=
	libraryFunStructCall["MTensor_disown", {arg}]
mtensorDisownAll[x_mtensor] :=
	mtensorDisownAll[mtensorName[x]]	
mtensorDisownAll[arg_] :=
	libraryFunStructCall["MTensor_disownAll", {arg}]

mtensorVectorSet[t_mtensor, idx_] :=
	mtensorVectorSet[mtensorType[t], mtensorName[t], idx]
mtensorVectorSet[type_, arg_, idx_] :=
	Switch[type,
		Integer | "Integer", mtensorVectorSetInteger[arg, idx],
		Real | "Real", mtensorVectorSetReal[arg, idx],
		Complex | "Complex", mtensorVectorSetComplex[arg, idx],
		_, "UnknownMTensorVectorGetData"
	]
mtensorVectorSetInteger[t_mtensor, idx_] :=
	mtensorVectorSetInteger[mtensorName[t], idx]
mtensorVectorSetInteger[arg_, idx_] :=
	libraryFunStructCall["MTensorVector_setInteger", {arg, idx}]
mtensorVectorSetReal[t_mtensor, idx_] :=
	mtensorVectorSetReal[mtensorName[t], idx]
mtensorVectorSetReal[arg_, idx_] :=
	libraryFunStructCall["MTensorVector_setReal", {arg, idx}]
mtensorVectorSetComplex[t_mtensor, idx_] :=
	mtensorVectorSetComplex[mtensorName[t], idx]
mtensorVectorSetComplex[arg_, idx_] :=
	libraryFunStructCall["MTensorVector_setComplex", {arg, idx}]

mtensorVectorGet[t_mtensor, idx_] :=
	mtensorVectorGet[mtensorType[t], mtensorName[t], idx]
mtensorVectorGet[type_, arg_, idx_] :=
	Switch[type,
		Integer | "Integer", mtensorVectorGetInteger[arg, idx],
		Real | "Real", mtensorVectorGetReal[arg, idx],
		Complex | "Complex", mtensorVectorGetComplex[arg, idx],
		_, "UnknownMTensorVectorGetData"
	]
mtensorVectorGetInteger[t_mtensor, idx_] :=
	mtensorVectorGetInteger[mtensorName[t], idx]
mtensorVectorGetInteger[arg_, idx_] :=
	libraryFunStructCall["MTensorVector_getInteger", {arg, idx}]
mtensorVectorGetReal[t_mtensor, idx_] :=
	mtensorVectorGetReal[mtensorName[t], idx]
mtensorVectorGetReal[arg_, idx_] :=
	libraryFunStructCall["MTensorVector_getReal", {arg, idx}]
mtensorVectorGetComplex[t_mtensor, idx_] :=
	mtensorVectorGetComplex[mtensorName[t], idx]
mtensorVectorGetComplex[arg_, idx_] :=
	libraryFunStructCall["MTensorVector_getComplex", {arg, idx}]

mtensorSet[t_mtensor, idx_] :=
	mtensorSet[mtensorType[t], mtensorName[t], idx]
mtensorSet[type_, arg_, idx_] :=
	Switch[type,
		Integer | "Integer", mtensorSetInteger[arg, idx],
		Real | "Real", mtensorSetReal[arg, idx],
		Complex | "Complex", mtensorSetComplex[arg, idx],
		_, "UnknownMTensorSetData"
	]
mtensorSetInteger[t_mtensor, idx_] :=
	mtensorSetInteger[mtensorName[t], idx]
mtensorSetInteger[arg_, idx_] :=
	libraryFunStructCall["MTensor_setInteger", {arg, idx}]
mtensorSetReal[t_mtensor, idx_] :=
	mtensorSetReal[mtensorName[t], idx]
mtensorSetReal[arg_, idx_] :=
	libraryFunStructCall["MTensor_setReal", {arg, idx}]
mtensorSetComplex[t_mtensor, idx_] :=
	mtensorSetComplex[mtensorName[t], idx]
mtensorSetComplex[arg_, idx_] :=
	libraryFunStructCall["MTensor_setComplex", {arg, idx}]

mtensorGet[t_mtensor, idx_] :=
	mtensorGet[mtensorType[t], mtensorName[t], idx]
mtensorGet[type_, arg_, idx_] :=
	Switch[type,
		Integer | "Integer", mtensorGetInteger[arg, idx],
		Real | "Real", mtensorGetReal[arg, idx],
		Complex | "Complex", mtensorGetComplex[arg, idx],
		_, "UnknownMTensorGetData"
	]
mtensorGetInteger[t_mtensor, idx_] :=
	mtensorGetInteger[mtensorName[t], idx]
mtensorGetInteger[arg_, idx_] :=
	libraryFunStructCall["MTensor_getInteger", {arg, idx}]
mtensorGetReal[t_mtensor, idx_] :=
	mtensorGetReal[mtensorName[t], idx]
mtensorGetReal[arg_, idx_] :=
	libraryFunStructCall["MTensor_getReal", {arg, idx}]
mtensorGetComplex[t_mtensor, idx_] :=
	mtensorGetComplex[mtensorName[t], idx]
mtensorGetComplex[arg_, idx_] :=
	libraryFunStructCall["MTensor_getComplex", {arg, idx}]

mtensorGetData[t_mtensor] :=
	mtensorGetData[mtensorType[t], mtensorName[t]]
mtensorGetData[type_, arg_] :=
	Switch[type,
		Integer | "Integer", mtensorGetIntegerData[arg],
		Real | "Real", mtensorGetRealData[arg],
		Complex | "Complex", mtensorGetComplexData[arg],
		_, "UnknownGetData"
	]
mtensorGetIntegerData[t_mtensor] :=
	mtensorGetIntegerData[mtensorName[t]]
mtensorGetIntegerData[arg_] :=
	libraryFunStructCall["MTensor_getIntegerData", {arg}]
mtensorGetRealData[t_mtensor] :=
	mtensorGetRealData[mtensorName[t]]
mtensorGetRealData[arg_] :=
	libraryFunStructCall["MTensor_getRealData", {arg}]
mtensorGetComplexData[t_mtensor] :=
	mtensorGetComplexData[mtensorName[t]]
mtensorGetComplexData[arg_] :=
	libraryFunStructCall["MTensor_getComplexData", {arg}]

(* ::Section:: *)
(* Types *)

libraryCallType[type_String] := type
libraryCallType[Boolean] = "Boolean"
libraryCallType[Integer | Verbatim[_Integer] | _Integer] = "Integer"
libraryCallType[Real | Verbatim[_Real] | _Real] = "Real"
libraryCallType[Complex] = "Complex"
libraryCallType[List | _mtensor] = "MTensor"
libraryCallType[_memory] = "Integer"
libraryCallType[String] = "UTF8String"

cType[Boolean] = "mbool"
cType[Integer | _Integer | "mint"] = "mint"
cType[Real | _Real | "double" | "mreal"] = "mreal"
cType[Real | _Real | "double" | "mreal", useSinglePrecisionQ:True] = "float"
cType[Complex | "mcomplex"] = "mcomplex"
cType[List | _mtensor | "MTensor"] = "MTensor"
cType[mtensor | _mtensor] = "MTensor"
cType[mtensor[_,_]] = "MTensor"
cType[_memory | "Memory"] = "WGL_Memory_t"
cType[VoidType | "Void" | "void"] = "void"
cType[MArgument | "MArgument"] = "MArgument"
cType[x_, _] := cType[x]

LibraryType[Integer] = "MType_Integer"
LibraryType[Real] = "MType_Real"
LibraryType[Complex] = "MType_Complex"

(* ::Section:: *)
(* MArgument Tools *)

mArgumentGet[to_, idx_Integer] :=
	CCall["MArgument_get" <> libraryCallType[to], CArray["Args", idx]]
mArgumentGet[type_, to_, from:(_mtensor|_Integer|_Real)] :=
	CCall["MArgument_get" <> libraryCallType[to], First@from]
mArgumentGet[type_, to_, from_] :=
	CCall["MArgument_get" <> libraryCallType[to], from]
	
mArgumentGetAddress[to_, idx_Integer] :=
	CCall["MArgument_get" <> libraryCallType[to] <> "Address", CArray["Args", idx]]
mArgumentGetAddress[to_, from:(_mtensor|_Integer|_Real)] :=
	CCall["MArgument_get" <> libraryCallType[to] <> "Address", First@from]
mArgumentGetAddress[to_, from_] :=
	CCall["MArgument_get" <> libraryCallType[to] <> "Address", from]

mArgumentSet[from_] :=
	mArgumentSet["Res", from]
mArgumentSet[to_, from:(_mtensor|_Integer|_Real)] :=
	CCall["MArgument_set" <> libraryCallType[from], {to, First@from}]
mArgumentSet[to_, from_] :=
	CCall["MArgument_set" <> libraryCallType[from], {to, from}]
	
mArgumentSetAddress[from_] :=
	mArgumentSetAddress["Res", from]
mArgumentSetAddress[to_, from_] :=
	CCall["MArgument_set" <> libraryCallType[from] <> "Address", {to, from}]

(* ::Subsection:: *)
(* Error Values *)

declareErrorVariable[api:($C | $WolframRTL | $WolframRTLHeader | $WolframLibrary), prefix_:""] :=
	CDeclare[ErrorType[api], ErrorVariableName[api, prefix]]

declareErrorVariable[api:($WGLCUDA | $WGLOpenCL), prefix_:""] := {}

ErrorVariableName[api:($C | $WolframRTL | $WolframRTLHeader | $WolframLibrary), prefix_:""] :=
	prefix <> defaults[api, "ErrorName"]

ErrorVariableName[api:($WGLCUDA | $WGLOpenCL), prefix_:""] :=
	wglGetError[]
		
ErrorType[$C | $WolframLibrary | $WolframRTL] := "int"

error[api_, "Success"] := success[api]
success[$C | $WolframLibrary | $WolframRTL] := "LIBRARY_NO_ERROR"
success[$WGLCUDA | $WGLOpenCL] := "WGL_Success" 

error[api_, "Type"] := typeError[api]
typeError[$C | $WolframLibrary | $WolframRTL] := "LIBRARY_TYPE_ERROR"
typeError[$WGLCUDA | $WGLOpenCL] := "WGL_Error_Invalid_Type"

error[api_, "Rank"] := rankError[api]
rankError[$C | $WolframLibrary | $WolframRTL] := "LIBRARY_RANK_ERROR"
rankError[$WGLCUDA | $WGLOpenCL] := "WGL_Error_Internal"

error[api_, "DimensionError"] := dimensionError[api]
dimensionError[$C | $WolframLibrary | $WolframRTL] := "LIBRARY_DIMENSION_ERROR"
dimensionError[$WGLCUDA | $WGLOpenCL] := "WGL_Error_Invalid_Dimensions"
	
error[api_, "NumericalError"] := numericalError[api]
numericalError[$C | $WolframLibrary | $WolframRTL] := "LIBRARY_NUMERICAL_ERROR"
numericalError[$WGLCUDA | $WGLOpenCL] := "WGL_Error_Internal"

error[api_, "MemoryError"] := memoryError[api]
memoryError[$C | $WolframLibrary | $WolframRTL] := "LIBRARY_MEMORY_ERROR"
memoryError[$WGLCUDA | $WGLOpenCL] := "WGL_Error_Invalid_Memory"
	
error[api_, "FunctionError"] := functionError[api]
functionError[$C | $WolframLibrary | $WolframRTL] := "LIBRARY_FUNCTION_ERROR"
functionError[$WGLCUDA | $WGLOpenCL] := "WGL_Error_Internal"

error[api_, _] := functionError[api]

(* ::Section:: *)
(* Grid Dimensions *)

declareGridDimensions[api:($WGLCUDA|$WGLOpenCL)] :=
	declareGridDimensions[api, defaults["GridName"]]
declareGridDimensions[$WGLCUDA|$WGLOpenCL, name_] :=
	CDeclare["mint", CArray[name, 2]]

(* ::Section:: *)
(* Block Dimensions *)

declareBlockDimensions[api:($WGLCUDA | $WGLOpenCL)] :=
	declareBlockDimensions[api, defaults["BlockName"]]
declareBlockDimensions[$WGLCUDA | $WGLOpenCL, name_] :=
	CDeclare["mint", CArray[name, 3]]

(* ::Section:: *)
(* Kernel Functions *)

gpuKernelEmbed[api:$WGLCUDA, deviceId0_, errMsgHd_, varName_, progText_, compileGPUProgramQ_, kernelCompileOptions:(_List|None)] :=
	Module[{arch, deviceId, outfile, compiledCode, ccode, compileOptions},
		compiledCode = If[compileGPUProgramQ,
			outfile = "CUDACodeGenerate_" <> api <> ToString[RandomInteger[10]];
			deviceId = If[deviceId0 === Automatic,
				CUDALink`Internal`FastestDevice[],
				deviceId0
			];
			arch = If[TrueQ[CUDALink`CUDAInformation[deviceId, "Compute Capabilities"] >= 2.0],
				"sm_20",
				"sm_10"
			];
			compileOptions = stripDuplicates[Flatten[Join[{
	    		"CreatePTX"->False, "CreateCUBIN"->True, "UnmangleCode"->True, "CUDAArchitecture"->arch,
	    		If[kernelCompileOptions === None,
	    			{},
	    			FilterRules[{kernelCompileOptions},  Options[CUDALink`NVCCCompiler`NVCCCompiler]]
	    		]
	    	}]]];
			Block[{CCompilerDriver`$ErrorMessageHead = errMsgHd},
				outfile = Quiet[
					CreateExecutable[
						progText, outfile, "Compiler" -> CUDALink`NVCCCompiler`NVCCCompiler, Sequence@@compileOptions
					], errMsgHd::wddirty
				]
			];
			If[outfile === $Failed,
				Return[$Failed]
			];
			Import[outfile, "UnsignedInteger8"],
			(* file extension passed in was either .ptx or .cubin *)
			progText
		];
		ccode = embedBinary[api, varName, compiledCode];
		Quiet[DeleteFile[outfile]];
		ccode
	]	
gpuKernelEmbed[api:$WGLOpenCL, deviceId_, errMsgHd_, varName_, progText_, compileGPUProgramQ_, compileOpts:(_List|None)] :=
	embedBinary[api, varName, progText];


(* ::Section:: *)
(* Embed Binary *)

embedBinary[api:$WGLCUDA, varName_String, prog_] :=
	Module[{dump, csv},
		dump = Riffle[
			Partition[
 				csv = Riffle[
  					"0x" <> IntegerString[#, 16, 2] & /@ prog
  					, ", "
  				], 64, 64, 1, {}
 			], "\n" <> $TabCharacter
    	];
    	{
	    	CDeclare[{"const", "size_t"}, CAssign[varName <> "Size", Length[dump]]],
	    	CDeclare[{"const", "unsigned", "char"}, 
	    		CAssign[
	    			CArray[varName, {}],
	    			StringJoin[
	    				"\n" <> $TabCharacter <> "{",
	    					dump,
	    				"}"
	    			]
				]
	   		]
    	}
  	]
embedBinary[api:$WGLOpenCL, varName_String, prog_String] :=
	Module[{pprog = StringSplit[prog, "\n"]},
		CDeclare[CPointerType[{"const", "char"}],
			CAssign[
				varName,
				StringJoin[
					If[Length[pprog] === 1,
						"",
						{$TabCharacter, "\\\n"}
					],
					Riffle[removeNulls[Flatten[formatEmbeddedLine /@ Most[pprog]]], $TabCharacter <> "\\\n"],
					$TabCharacter, "\\\n",
					formatEmbeddedLine[Last[pprog]]
				]
			]
		]
	]

formatEmbeddedLine[line0_String] :=
	Module[{line = StringTrim[line0]},
		If[StringMatchQ[line, ""],
			Null,
			Riffle[
				List[$TabCharacter <> ToCCodeString[CString[line]]],
				$TabCharacter <> "\\\n"
			]
		]
	]

(* ::Subsection:: *)
(* Memory Free *)

Options[memoryFree] = {
	"Debug"				->		False
}

memoryFree[$C, ptr_, opts:OptionsPattern[]] :=
	memoryFree[$C, "", ptr, opts]
memoryFree[$C, err_, ptr_, opts:OptionsPattern[]] :=
	Module[{
			debugQ = TrueQ@OptionValue["Debug"]
		   },
		{
		cIf[COperator[Unequal, {ptr, "NULL"}],
			CCall["free", ptr]
			,
			If[debugQ,
				CPrintf["Encountered Null on line %d\\n", "__LINE__"]
			]
		]
		}
	]


(* ::Subsection:: *)
(* Memory Copy *)

Options[memoryAlloc] = {
	"Debug"				->		False
}

memoryAlloc[api:$C, ptr_, size_, opts:OptionsPattern[]] :=
	memoryAlloc[api, defaults[api, "ErrorName"], ptr, size, opts]
memoryAlloc[api:$C, err_, ptr_, size_, opts:OptionsPattern[]] :=
	memoryAlloc[api, err, ptr, "void", size, opts]
memoryAlloc[api:$C, err_, ptr_, type_, size_, opts:OptionsPattern[]] :=
	CAssign[ptr, CCast[CPointerType[type], CCall["malloc", size]]]
	
(* ::Subsection:: *)
(* Initialization Functions *)


Options[initializationFunction] := {"Debug"->False}

initializationFunction[$WGLCUDA|$WGLOpenCL, codeTarget_][platformId_, deviceId_, opts:OptionsPattern[]] :=
	Module[{debugQ = TrueQ@OptionValue[initializationFunction, {opts}, "Debug"]},
		If[codeTarget === $WolframRTL,
			{
				CComment["===== Set Platform and Device =====", {"\n", "\n"}],
				If[platformId === Automatic,
					wglSetPlatform[debugQ, wglAutomatic[]],
					wglSetPlatform[debugQ, platformId]
				],
				If[deviceId === Automatic,
					wglSetDevice[debugQ, wglAutomatic[]],
					wglSetDevice[debugQ, deviceId]
				]
			},
			{}
		]
	]

initializationFunction[funName_String, codeTarget:($WolframRTL | $WolframLibrary), apiTarget:($WGLCUDA | $WGLOpenCL), platformId_, deviceId_, prefix_, opts:OptionsPattern[]] :=
	Module[{debugQ = TrueQ@OptionValue[initializationFunction, {opts}, "Debug"]},
		{
			externFunction[{"DLLEXPORT", "int"},
							 If[codeTarget === $WolframRTL,
							 	"Initialize_" <> funName,
							 	"WolframLibrary_initialize"
							 ],
							 If[codeTarget === $WolframRTL,
							 	{},
							 	{{"WolframLibraryData", "arg"}}
							 ],
				removeNulls[{
					If[codeTarget === $WolframRTL,
						cIf[COperator[Equal, {defaults["RuntimeInitializeQ"], defaults[True]}],
							CReturn[success[codeTarget]]
						]
					],
					If[codeTarget === $WolframRTL,
						{
							CComment["===== Initialize Wolfram Runtime =====", {"\n", "\n"}],
							rtlInitialize["WolframLibraryVersion"],
							CAssign[defaults["libData"], rtlNewWolframLibraryFunction["WolframLibraryVersion"]],
							CAssign[defaults["WolframCompile"], libraryGetCompileLibraryFunctions[]],
							CAssign[defaults["RuntimeInitializeQ"], defaults[True]],
							wolframGPULibraryNew[],
							wglInitialize[debugQ, apiTarget]
						},
						{}
					],
					CReturn[error[codeTarget, "Success"]]
				}]
			],
			If[MemberQ[{$WGLCUDA, $WGLOpenCL}, apiTarget] && codeTarget =!= $WolframRTL,
				externFunction[{"DLLEXPORT", "int"}, "WolframGPULibrary_initialize", {{"WolframGPULibraryData", "arg"}},
					cBlock[{
						CAssign["wglData", "arg"],
						cIf[COperator[And, {COperator[Unequal, {"wglData", "NULL"}], COperator[Unequal, {CPointerMember[defaults[apiTarget, "ErrorName"], "code"] , error[apiTarget, "Success"]}]}],
                        	CReturn[error[codeTarget, "FunctionError"]]
                    	],
						CReturn[error[codeTarget, "Success"]]
					}]
				],
				{}
			]
		}
	]

(* ::Subsection:: *)
(* Uninitialization Functions *)

unitializationFunction[funName_String, apiTarget_, $WolframRTL] :=
	{
		externFunction[{"DLLEXPORT", "void"}, "Uninitialize_" <> funName, {},
			cIf[COperator[Equal, {defaults["RuntimeInitializeQ"], defaults[False]}], 
				{
					rtlFreeWolframLibraryFunction[defaults["libData"]],
					If[MemberQ[{$WGLCUDA, $WGLOpenCL}, apiTarget],
						wolframGPULibraryFree[],
						{}
					],
					CAssign[defaults["libData"], CCast["WolframLibraryData", 0]],
					unitializationFunction[apiTarget],
					CAssign[defaults["RuntimeInitializeQ"], defaults[False]],
					CReturn[]
				}
			]
		]
	}
unitializationFunction[funName_String, apiTarget_, $WolframLibrary] :=
	{
		externFunction[{"DLLEXPORT", "void"}, "WolframLibrary_uninitialize", {{"WolframLibraryData", "libData"}},
			cBlock[{
				unitializationFunction[apiTarget],
				CReturn[]
			}]
		]
	}

unitializationFunction[$WGLOpenCL | $WGLCUDA] := {}

(* ::Section:: *)
(* Input Parsing *)

parseArguments[errMsgHd_, apiTarget_, codeTarget_][args_List] :=
	MapIndexed[resolveType[errMsgHd, apiTarget, codeTarget], args]

resolveType[errMsgHd_, apiTarget_, codeTarget_][Verbatim[_Integer] | "Integer32" | "Integer64", (idx_Integer | {idx_Integer})] := Integer["I" <> ToString[idx], idx, False]
resolveType[errMsgHd_, apiTarget_, codeTarget_][Verbatim[_Real] | "Float" | "Double", (idx_Integer | {idx_Integer})] := Real["R" <> ToString[idx], idx, False]
resolveType[errMsgHd_, apiTarget_, codeTarget_][{type_}, (idx_Integer | {idx_Integer})] := resolveType[errMsgHd, apiTarget, codeTarget][{type, 1, "InputOutput"}, idx]
resolveType[errMsgHd_, apiTarget_, codeTarget_][{type_, cpy:("Input"|"Output"|"InputOutput")}, (idx_Integer | {idx_Integer})] := resolveType[errMsgHd, apiTarget, codeTarget][{type, _, cpy}, idx]
resolveType[errMsgHd_, apiTarget_, codeTarget_][{type0_, rank_, inputOutputQ_}, (idx_Integer | {idx_Integer})] :=
	Module[{type = listType[errMsgHd, type0]}, 
		If[wglQ[apiTarget], memory, mtensor][
			"WGLMemory_" <> ToString[idx],															(* name *)
			type,																						(* type *)
			rank,																						(* rank *)
			idx,																						(* index *)
			inputOutputQ												                    			(* outputQ *)
		]
	]

listType[errMsgHd_, Verbatim[_Integer] | Integer | "Integer64"] := Integer
listType[errMsgHd_, "Integer32"] := "Integer32"
listType[errMsgHd_, "UnsignedByte"] := "UnsignedByte"
listType[errMsgHd_, Verbatim[_Real] | Real | "Double" | "Float"] := Real
listType[errMsgHd_, Verbatim[_Complex] | Complex | "Complex"] := Complex
listType[errMsgHd_, x___] := Throw[Message[errMsgHd::ilsttype, {x}]; $Failed]


(* ::Section:: *)
(* Choose and Set Device *)

defineChooseFastestGPUDevice[api:($WGLCUDA|$WGLOpenCL), codeTarget_] := {}

callChooseFastestGPUDevice[api:($WGLCUDA | $WGLOpenCL)] := wglAutomatic[]

(* ::Section:: *)
(* Generation Functions *)
	

$GPUCodeGenerateOptions = Sort@{
	"FunctionPrefix"		->		Automatic,
	"TargetPrecision"		->		Automatic,
	"Debug"					->		False,
	"APITarget"				->		"wglCUDA",		    (* also, "wglCUDA" and "WGLOpenCL" *)
	"CodeTarget"			->		"WolframLibrary", 	(* also, "WolframRTLHeader" and "WolframRTL" *)
	"Platform"				->		Automatic,
	"Device"				->		Automatic,
	"GeneratedCode"			->		All, 				(* also, None and "LibraryFunction" *)
	"KernelCompileOptions"	->		Automatic,
	"CUDAArchitecture"		->		"sm_20"
}

declareGlobalVariables[apiTarget_, codeTarget_] :=
	Switch[codeTarget,
		$WolframRTL,
			{
				declareRuntimeData[],
				declareFunStruct[],
				declareWolframCompileData[],
				CDeclare[{"static", "mbool"}, CAssign[defaults["RuntimeInitializeQ"], defaults[False]]]
			},
		_,
			{}
	] ~Join~ {declareWGLData[]}

declareWolframInitializationFunction[funName_, apiTarget_, $WolframRTLHeader] := CFunction[{"EXTERN_C", "DLLEXPORT", "int"}, "Initialize_" <> funName, {}]
declareWolframUninitializationFunction[funName_, apiTarget_, $WolframRTLHeader] := CFunction[{"EXTERN_C", "DLLEXPORT", "void"}, "Uninitialize_" <> funName, {}]
declareFunctionPrototype[funName_, apiTarget_, $WolframRTLHeader, params_List] := 
	Module[{blockName 			= "MTensor_"<>defaults["BlockName"],
			gridName 			= "MTensor_"<>defaults["GridName"]
		   },
		CFunction[
			{"EXTERN_C", "DLLEXPORT", "int"},
			funName,
			parameterFunctionSignature[$WolframRTLHeader][blockName, gridName, params]
		]
	]

Options[declareHeader] := {"Debug" -> False}	
declareHeader[funName_, apiTarget_, codeTarget:$WolframRTLHeader, opts:OptionsPattern[]] :=
	Module[{
			debugQ = OptionValue[declareHeader, {opts}, "Debug"]
		    },
		{
			includeHeaders[funName, apiTarget, codeTarget, debugQ],
			declareWolframInitializationFunction[funName, apiTarget, codeTarget],
			declareWolframUninitializationFunction[funName, apiTarget, codeTarget]
		}
	]
declareHeader[funName_, apiTarget_, codeTarget:($WolframLibrary | $WolframRTL), opts:OptionsPattern[]] :=
	Module[{
			debugQ = OptionValue[declareHeader, {opts}, "Debug"]
		    },
		removeNulls[{
			includeHeaders[funName, apiTarget, codeTarget, debugQ],	
			declareGlobalVariables[apiTarget, codeTarget]
		}]
	]
	
Options[declareMacros] := {"Debug" -> False}
declareMacros[opts:OptionsPattern[]] :=
	Module[{
			debugQ = OptionValue[declareMacros, {opts}, "Debug"]
			},
			defineMacros[debugQ]
	]
			
		
Options[generateLibraryFunction] := {"APITarget" -> $WGLCUDA, "CodeTarget" -> ("CodeTarget" /. $GPUCodeGenerateOptions), "FunctionPrefix" -> Automatic,
	 "Debug" -> False}
generateLibraryFunction[errMsgHd_, funName_String, params_List, opts:OptionsPattern[]] :=
	Module[{
			codeTarget = OptionValue[generateLibraryFunction, {opts}, "CodeTarget"],
			apiTarget = OptionValue[generateLibraryFunction, {opts}, "APITarget"],
			functionPrefix = OptionValue[generateLibraryFunction, {opts}, "FunctionPrefix"],
			blockName = "MTensor_"<>defaults["BlockName"],
			gridName = "MTensor_"<>defaults["GridName"],
			debugQ = TrueQ@OptionValue[generateLibraryFunction, {opts}, "Debug"]
		   },
		If[functionPrefix === Automatic,
			functionPrefix = ToString[apiTarget] <> "_"
		]; 
		libraryFunction[funName,
			declareErrorVariable[codeTarget],
			CComment["===== Declare Input Arguments =====", {"\n"}],
			libraryDeclareInputArguments[apiTarget, codeTarget][params],
			CComment["===== Declare Block/Grid Arguments =====", {"\n"}],
			CDeclare["MTensor", {blockName, gridName}],
			
			CComment["===== Assign Input Arguments =====", {"\n"}],
			libraryAssignInputArguments[apiTarget, codeTarget][params],
			If[debugQ, {
				CComment["===== Debugging Information Input Arguments =====", {"\n", "\n"}],
				wglPrintErrorInputArguments[apiTarget, codeTarget][params]
			}],
			CComment["===== Assign Block/Grid Arguments =====", {"\n"}],
			CAssign[gridName, mArgumentGet["MTensor", Length[params]]],
			CAssign[blockName, mArgumentGet["MTensor", Length[params]+1]],
			CComment["===== Register WolframLibData with WolframGPULibraryData =====", {"\n", "\n"}],
			wglSetWolframLibraryData[debugQ, defaults["libData"]],
			CComment["===== Find " <> apiTarget <> " Memory =====", {"\n", "\n"}],
			wglFindMemory[apiTarget, codeTarget][params],
			
			removeNulls[{Which[
					TrueQ[memoryRank[#] >= 0],{},
					TrueQ[memoryRank[#] === _],	{},
					True, Throw[Message[errMsgHd::invtypspec, params]; $Failed]
				] & /@ Select[params, memoryQ]
			}],
			cIf[COperator[Or, COperator[Equal, {memoryName[#], "NULL"}] & /@ Select[params, memoryQ]], 
				CReturn[error[apiTarget, "MemoryError"]]],
			CComment["===== Call " <> apiTarget <> " Implementation =====", {"\n"}],
			CAssign[ErrorVariableName[codeTarget], 
				parameterFunctionCall[functionPrefix<>funName, blockName, gridName, params]
			],
			CComment["===== Disown Grid and Block Dimensions =====", {"\n"}],
			mtensorDisownAll[blockName],
			mtensorDisownAll[gridName],
			CComment["===== Assign Return Error Code =====", {"\n"}],
			If[debugQ, { 
				CComment["===== Debugging Information =====", {"\n", "\n"}], 
				CLabel["cleanup"],
				wglPrintErrorMsg[]	
				}],
			CReturn[ErrorVariableName[codeTarget]]
		]
	]

Options[gpuInterfaceFunction] := $GPUCodeGenerateOptions

gpuInterfaceFunction[apiTarget:($WGLOpenCL | $WGLCUDA), codeTarget_][prog_, kernelFunctionName_, funName_String, params_List, opts:OptionsPattern[]] :=
	Module[{
			deviceId 			= OptionValue[gpuInterfaceFunction, {opts}, "Device"],
			platformId 			= OptionValue[gpuInterfaceFunction, {opts}, "Platform"],
			functionPrefix 		= OptionValue[gpuInterfaceFunction, {opts}, "FunctionPrefix"],
			kernelCompileOptions = OptionValue[gpuInterfaceFunction, {opts}, "KernelCompileOptions"],
			blockName 			= "MTensor_"<>defaults["BlockName"],
			gridName 			= "MTensor_"<>defaults["GridName"],
			debugQ 				= TrueQ@OptionValue[gpuInterfaceFunction, {opts}, "Debug"],
			useSinglePrecisionQ
		   },
	  	useSinglePrecisionQ = With[{optval = OptionValue[gpuInterfaceFunction, {opts}, "TargetPrecision"]},
	  		If[optval === Automatic,
	  			TrueQ@If[codeTarget === $WGLOpenCL,
	  				OpenCLLink`Internal`SupportsDoublePrecisionQ[platformId, deviceId],
	  				CUDALink`Internal`SupportsDoublePrecisionQ[deviceId]
	  			],
	  			optval === "Single"
	  		]
	  	];
		If[functionPrefix === Automatic,
			functionPrefix = If[codeTarget === $WolframRTL,
				"",
				ToString[apiTarget] <> "_"
			]
		];
		CFunction[If[codeTarget === $WolframRTL, {"EXTERN_C", "DLLEXPORT", "int"}, {"static", "int"}], functionPrefix<>funName, parameterFunctionSignature[codeTarget][blockName, gridName, params], cBlock[{
			CComment["===== Register WolframLibData with WolframGPULibraryData =====", {"\n", "\n"}],
			wglSetWolframLibraryData[debugQ, defaults["libData"]],
			CComment["===== Load Program =====", {"\n"}],
			wglLoadProgram[apiTarget, codeTarget][prog, kernelCompileOptions, debugQ],
			CComment["===== Set Kernel Name =====", {"\n"}],
			wglSetKernel[debugQ, CString[kernelFunctionName]],
			CComment["===== Set Kernel Arguments =====", {"\n"}],
			wglSetKernelArguments[params, useSinglePrecisionQ, "Debug"->debugQ],
			If[debugQ, {
				CComment["===== Print Debugging Information ====", {"\n", "\n"}],
				wglPrintErrorDimensions[gridName],
				wglPrintErrorDimensions[blockName]
			}],
			CComment["===== Set Grid and Block Dimensions =====", {"\n"}],
			setGridAndBlockDimensions[debugQ, gridName, blockName],
			CComment["===== Launch Kernel =====", {"\n"}],
			wGLLaunchKernel[debugQ],
			CComment["===== Return Error Code =====", {"\n"}],
			If[debugQ, { 
				CComment["===== Debugging Information =====", {"\n", "\n"}],
				CLabel["cleanup"],
				wglPrintErrorMsg[],
				wglPrintErrorFunctionName[],
				wglPrintErrorFileInfo[],
				wglPrintErrorBuildLog[]
			}],
			cIf[wglSuccessQ[],
				CReturn["LIBRARY_NO_ERROR"],
				CReturn["LIBRARY_FUNCTION_ERROR"]
			]
		}]]
	]

gpuInterfaceFunction[apiTarget_, codeTarget_][prog_, kernelFunctionName_, funName_String, params_List, opts:OptionsPattern[]] :=
	Module[{
			deviceId 			= OptionValue[gpuInterfaceFunction, {opts}, "Device"],
			platformId 			= OptionValue[gpuInterfaceFunction, {opts}, "Platform"],
			functionPrefix 		= OptionValue[gpuInterfaceFunction, {opts}, "FunctionPrefix"],
			DebugQ				= OptionValue[gpuInterfaceFunction, {opts}, "Debug"],
			kernelCompileOptions = OptionValue[gpuInterfaceFunction, {opts}, "KernelCompileOptions"],
			fermiSupportQ		= OptionValue[gpuInterfaceFunction, {opts}, "CUDAArchitecture"] === "sm_20",
			blockName 			= "MTensor_"<>defaults["BlockName"],
			gridName 			= "MTensor_"<>defaults["GridName"],
			useSinglePrecisionQ, usesDoublesQ, elemSizeName = defaults["ElementSize"], offsetName = defaults["Offset"],
			deviceNumberName = defaults["DeviceNumberName"]
		   },
	  	useSinglePrecisionQ = With[{optval = OptionValue[gpuInterfaceFunction, {opts}, "TargetPrecision"]},
	  		If[optval === Automatic,
	  			If[codeTarget === $OpenCL,
	  				OpenCLLink`Internal`SupportsDoublePrecisionQ[platformId, deviceId],
	  				CUDALink`Internal`SupportsDoublePrecisionQ[deviceId]
	  			],
	  			optval === "Single"
	  		]
	  	];
		usesDoublesQ = Length[Select[params, (mtensorQ[#] && mtensorType[#] === Real)&, 1]] > 0; 
		If[functionPrefix === Automatic,
			functionPrefix = If[codeTarget === $WolframRTL,
				"",
				ToString[apiTarget] <> "_"
			]
		];
		CFunction[If[codeTarget === $WolframRTL, {"EXTERN_C", "DLLEXPORT", "int"}, {"static", "int"}], functionPrefix<>funName, parameterFunctionSignature[codeTarget][blockName, gridName, params], cBlock[{
			declareErrorVariable[codeTarget],
			declareErrorVariable[apiTarget],
			Switch[apiTarget,
				$WGLCUDA | $WGLOpenCL,
					{}
			],
			CComment["===== Declare the Block dimensions =====", {"\n"}],
			declareBlockDimensions[apiTarget],
			CComment["===== Declare the Grid dimensions =====", {"\n"}],
			declareGridDimensions[apiTarget],
			CComment["===== Declare Tensor Data Variables =====", {"\n"}],
			declareTensorData[params],
			CComment["===== Declare Tensor " <> apiTarget <> " Variables =====", {"\n"}],
			declareGPUData[apiTarget, "", useSinglePrecisionQ, params],
			
			If[useSinglePrecisionQ && usesDoublesQ,
				{
					CComment["===== Declare Intermediate Data types (only if doubles are used and UseDoublePrecision is set to False) =====", {"\n", "\n"}],
					declareIntermediateData[params]
				}
			],
			CComment["===== Assign return value =====", {"\n"}],
			CAssign[ErrorVariableName[codeTarget], error[codeTarget, "FunctionError"]],
			CComment["===== Assign Tensor Data Variables =====", {"\n"}],
			assignTensorData[params],
			CComment["===== Assign Device ID (TODO) =====", {"\n", "\n"}],
			CComment["===== Allocate " <> apiTarget <> " Memory =====", {"\n"}],
			allocGPUData[apiTarget, "", useSinglePrecisionQ, params, DebugQ],
			If[useSinglePrecisionQ && usesDoublesQ,
				{
					CComment["===== Allocate Intermediate Memory (only if doubles are used and UseDoublePrecision is set to False) =====", {"\n", "\n"}],
					allocIntermediateData[params],
					CComment["===== Copy Intermediate Memory (only if doubles are used and UseDoublePrecision is set to False) =====", {"\n", "\n"}],
					copyToIntermediateData[params]
				}
			],
			CComment["===== Copy Memory from Host to Device =====", {"\n"}],
			copyGPUData[apiTarget, "HostToDevice", "", useSinglePrecisionQ, params, DebugQ],
			CComment["===== Assign the Block dimensions =====", {"\n"}],
			assignBlockDimensions[apiTarget, blockName],
			CComment["===== Assign the Grid dimensions =====", {"\n"}],
			assignGridDimensions[apiTarget, gridName],
            CComment["===== Set Kernel Arguments =====", {"\n", "\n"}],
            setKernelArguments[apiTarget, fermiSupportQ, useSinglePrecisionQ, elemSizeName, offsetName, params],
			CComment["===== Launch Kernel =====", {"\n"}],
			setBlockDimensions[DebugQ, apiTarget, defaults["BlockName"]],
			kernelLaunch[apiTarget, defaults["BlockName"], defaults["GridName"]],
			kernelSynchronize[apiTarget],
			CComment["===== Copy Memory from Device to Host =====", {"\n", "\n"}],
			copyGPUData[apiTarget, "DeviceToHost", "", useSinglePrecisionQ, params, DebugQ],
			CComment["===== Assign error to success =====", {"\n"}],
			CAssign[ErrorVariableName[codeTarget], error[codeTarget, "Success"]],
			CLabel["cleanupMemory"],
			CComment["===== Free Resources =====", {"\n"}],
			CComment[">>>>> Free " <> apiTarget <> " Memory <<<<<", {"\n"}],
			memoryFreeGPUData[apiTarget, "", params, DebugQ],
            CLabel["cleanupResources"],
			CLabel["exit"],
			If[codeTarget === $WolframRTL,
				{
					CComment["===== Cleanup Wolfram Runtime =====", {"\n", "\n"}],
					wolframCompileLibraryDataCleanup[defaults["libData"], defaults[True]]
				}
			],
			CComment["===== Return Error Result =====", {"\n"}],
			CReturn[ErrorVariableName[codeTarget]]
		}]]
	]



	
(* ::Section:: *)
(* Implementation Functions *)


Options[iGPULibraryFunctionGenerate] := Options[iGPULibraryGenerate]
iGPULibraryFunctionGenerate[errMsgHd_, prog_, kernelFunctionName_String, parameters_List, opts:OptionsPattern[]] :=
	Module[{libraryFile, funName = toFunctionName[kernelFunctionName], 
			apiTarget = OptionValue[iGPUSymbolicCGenerate, {opts}, "APITarget"],
			codeTarget = OptionValue[iGPUSymbolicCGenerate, {opts}, "CodeTarget"],
			params},
		libraryFile = iGPULibraryGenerate[errMsgHd, prog, parameters, opts];
		If[libraryFile === $Failed,
			Return[$Failed, Module]
		];
		params = parseArguments[errMsgHd, apiTarget, codeTarget][parameters];
		LibraryFunctionLoad[
			libraryFile,
			funName,
			Join[
				libraryFunctionLoadParameter /@ params,
				{{Integer, 1, "Shared"}, {Integer, 1, "Shared"}}
			],
			First[libraryFunctionLoadParameter /@ Select[params, outputParamQ, 1]]
		]
	]

libraryFunctionLoadParameter[param_Integer] :=
	Integer
libraryFunctionLoadParameter[param_Real] :=
	Real
libraryFunctionLoadParameter[param_mtensor] :=
	{mtensorType[param], mtensorRank[param], "Shared"}

usingNVCCQ[] := ListQ[CCompilers[]] && MemberQ["Name" /. CCompilers[], "NVIDIA CUDA Compiler"]

Options[iGPULibraryGenerate] := 
	FilterRules[
		Join[
			$GPUCodeGenerateOptions,
			If[usingNVCCQ[],
				FilterRules[
					Options[CUDALink`NVCCCompiler`NVCCCompiler],
					Except[Apply[Alternatives, {"Debug"} ~Join~ CUDALink`NVCCCompiler`NVCCCompiler["OptionsExceptions"]["CreateLibrary"]]]
				],
				Options[CreateLibrary]
			]
		], Except["CreateBinary" | "ExtraObjectFiles" | "CodeTarget"]
	]
iGPULibraryGenerate[errMsgHd_, prog_, funName_, args___, opts:OptionsPattern[]] :=
	Module[{srcFile, workingDir, compileOptions, res, apiTarget, compiler},
		workingDir = OptionValue[iGPULibraryGenerate, {opts}, "WorkingDirectory"];
		apiTarget = OptionValue[iGPULibraryGenerate, {opts}, "APITarget"];
		srcFile = iGPUCodeGenerate[
			errMsgHd, workingDir, prog, funName, args,
			Sequence[
				FilterRules[{opts},  Options[iGPUCodeGenerate]]
			],
			"CodeTarget" -> "WolframLibrary"
		];
		If[srcFile === $Failed,
			Return[$Failed]
		];
		compiler = If[apiTarget === $WGLCUDA,
			CUDALink`NVCCCompiler`NVCCCompiler,
			Automatic
		];
		compileOptions = FilterRules[{opts}, Options[
			If[apiTarget === $WGLCUDA,
				CUDALink`NVCCCompiler`NVCCCompiler,
				CreateLibrary
			]
		]];
    	res = Block[{CCompilerDriver`$ErrorMessageHead = errMsgHd},
        	Quiet[
				CreateLibrary[
				   {srcFile}, ToString[Unique[funName]], "Compiler"->compiler, (*"CUDAArchitecture"-> "sm_11",*) Sequence@@compileOptions
				], errMsgHd::wddirty
        	]
		];
    	If[OptionValue[iGPULibraryGenerate, {opts}, "CleanIntermediate"],
    		Quiet@DeleteFile[srcFile]
    	];
    	res
	]

Options[iGPUCodeGenerate] = $GPUCodeGenerateOptions ~Join~ {"Indent" -> None}
iGPUCodeGenerate[errMsgHd_, dir0_, args___, opts:OptionsPattern[]] :=
	Module[{dir,
			src,
			kernelName,
			apiTarget = OptionValue[iGPUCodeGenerate, {opts}, "APITarget"],
			codeTarget = OptionValue[iGPUCodeGenerate, {opts}, "CodeTarget"]},
		dir = If[dir0 === Automatic,
			FileNameJoin[{$UserBaseDirectory, "SystemFiles", "LibraryResources", $SystemID}],
			dir0
		];
		src = iGPUCodeStringGenerate[errMsgHd, args, opts];
		If[src === $Failed,
			$Failed,
			If[!DirectoryQ[dir],
				CreateDirectory[dir]
			];
			kernelName = With[{arg = List[args]},
				If[Length[arg] === 1,
					wglFunctionGetKernelName[arg],
					arg[[2]]
				]
			];
			Export[
				FileNameJoin[{dir, kernelName <> If[codeTarget === $WolframRTLHeader, ".h", ".c"]}],
				src,
				"Text"
			]
		]
	]


Options[iGPUCodeStringGenerate] = $GPUCodeGenerateOptions ~Join~ {"Indent" -> None}
iGPUCodeStringGenerate[args___, opts:OptionsPattern[]] :=
	Module[{symb = iGPUSymbolicCGenerate[args, Sequence@@FilterRules[{opts}, Except["Indent"]]],
			indentQ = OptionValue[iGPUCodeStringGenerate, {opts}, "Indent"]},
		If[symb === $Failed,
			$Failed, 
			ToCCodeString[symb, "Indent" -> indentQ]
		]
	]	

wglFunctionGetKernelName[x_List] := wglFunctionGetKernelName[First[x]]
wglFunctionGetKernelName[_[id_, kernelName_, __]] := kernelName
wglProgramGetSource[_[id_, args__]] := "Source" /. {args}

Options[iGPUSymbolicCGenerate] = $GPUCodeGenerateOptions
iGPUSymbolicCGenerate[errMsgHd_, fun:_[prog_, kernelName_String, args_List, rest___], opts:OptionsPattern[]] :=
	iGPUSymbolicCGenerate[errMsgHd, wglProgramGetSource[prog], kernelName, args, opts]
iGPUSymbolicCGenerate[errMsgHd_, prog0_, kernelFunctionName_String, parameters_List, opts:OptionsPattern[]] :=
	Module[{params, functionPrefix, generateCode, apiTarget, codeTarget, platformId, deviceId,
			debugQ, progText, gpuProg, kernelCompileOptions, compileGPUProgramQ = False, funName
		   },
		   
		functionPrefix = getAndCheckOption[iGPUSymbolicCGenerate, errMsgHd, {opts}, "FunctionPrefix"];
		generateCode = getAndCheckOption[iGPUSymbolicCGenerate, errMsgHd, {opts}, "GeneratedCode"];
		apiTarget = getAndCheckOption[iGPUSymbolicCGenerate, errMsgHd, {opts}, "APITarget"];
		codeTarget = getAndCheckOption[iGPUSymbolicCGenerate, errMsgHd, {opts}, "CodeTarget"];
		platformId = getAndCheckOption[iGPUSymbolicCGenerate, errMsgHd, {opts}, "Platform"];
		deviceId = getAndCheckOption[iGPUSymbolicCGenerate, errMsgHd, {opts}, "Device"];
		debugQ = getAndCheckOption[iGPUSymbolicCGenerate, errMsgHd, {opts}, "Debug"];
		kernelCompileOptions= getAndCheckOption[iGPUSymbolicCGenerate, errMsgHd, {opts}, "KernelCompileOptions"];
		
		funName = toFunctionName[kernelFunctionName];
		
		If[!StringQ[funName],
			Message[errMsgHd::invfunn, funName];
			Return[$Failed]
		];

		params =  parseArguments[errMsgHd, apiTarget, codeTarget][parameters];
		If[codeTarget === $WolframRTLHeader,
			Return[CProgram[{
				declareHeader[funName, apiTarget, codeTarget, "Debug" -> debugQ],
				declareMacros["Debug"-> debugQ],
				declareFunctionPrototype[funName, apiTarget, codeTarget, params]
			}]]
		];
		
		If[apiTarget === $WGLCUDA && !usingNVCCQ[],
			Throw[Message[errMsgHd::nonvcc, apiTarget]; $Failed]
		];
		
		Which[
			StringQ[prog0],
				compileGPUProgramQ = True;
				progText = prog0,
			ListQ[prog0] && Length[prog0] == 1 && StringQ[First[prog0]] && FileExistsQ[First[prog0]],
				progText = Import[First@prog0, "Text"];
				If[ListQ[prog0] && FreeQ[{"cubin", "ptx"}, FileExtension[First[prog0]]] && (apiTarget == $WGLCUDA),
					compileGPUProgramQ = True
				],
			True,
				Throw[Message[errMsgHd::invprog, prog0]; $Failed]
		];
		gpuProg = Switch[apiTarget,
			 $WGLCUDA | $WGLOpenCL,
				gpuKernelEmbed[apiTarget, deviceId, errMsgHd, defaults["GPUKernel"], progText, compileGPUProgramQ, kernelCompileOptions],
			_,
				{progText}
		];
		If[gpuProg === $Failed,
			Return[$Failed]
		];
		If[generateCode === None,
			Return[CProgram[gpuProg]]
		];
		
		If[functionPrefix === Automatic,
			functionPrefix = ToString[apiTarget] <> "_"
		];

		Apply[CProgram,
			Flatten[Join[removeNulls[{
				declareHeader[funName, apiTarget, codeTarget, "Debug" -> debugQ],
				If[debugQ, declareMacros["Debug" -> debugQ]],
				If[generateCode === All,
					removeNulls[{
						gpuProg,
						If[deviceId === Automatic,
							defineChooseFastestGPUDevice[apiTarget, codeTarget]
						],
						initializationFunction[funName, codeTarget, apiTarget, platformId, deviceId, functionPrefix, Sequence[FilterRules[{opts}, Options[initializationFunction]]]],
						If[codeTarget === $WolframLibrary,
							libraryVersionInformationFunction[]
						],
						unitializationFunction[funName, apiTarget, codeTarget],
						gpuInterfaceFunction[apiTarget, codeTarget][prog0, kernelFunctionName, funName, params, Sequence[FilterRules[{opts}, Options[gpuInterfaceFunction]]]]
					}]
				],
				If[codeTarget === $WolframLibrary,
					generateLibraryFunction[errMsgHd, funName, params, Sequence[FilterRules[{opts}, Options[generateLibraryFunction]]]]
				]
			}]]]
		]
	]

toFunctionName[kernelFunctionName_] := "o" <> ToUpperCase[StringTake[kernelFunctionName, 1]] <>  StringTake[kernelFunctionName, {2, -1}]


(******************************************************************************)
			
getAndCheckOption[head_, opt_, opts_] := getAndCheckOption[head, head, opt, opts] 

getAndCheckOption[head_, errMsgHd_, n:"FunctionPrefix"] :=
	With[{opt = OptionValue[head, opts, n]},
		If[StringQ[opt],
			opt,
			Throw[Message[errMsgHd::invfp, opt]; $Failed]
		]
	]
getAndCheckOption[head_, errMsgHd_, opts_, n:"GeneratedCode"] :=
	With[{opt = OptionValue[head, opts, n]},
		If[opt === None || opt === True || opt === All,
			opt,
			Throw[Message[errMsgHd::invgc, opt]; $Failed]
		]
	]
getAndCheckOption[head_, errMsgHd_, opts_, n:"APITarget"] :=
	With[{opt = OptionValue[head, opts, n]},
		If[MemberQ[{$WGLCUDA, $WGLOpenCL}, opt],
			opt,
			Throw[Message[errMsgHd::invtrgtapi, opt]; $Failed]
		]
	]
getAndCheckOption[head_, errMsgHd_, opts_, n:"CodeTarget"] :=
	With[{opt = OptionValue[head, opts, n]},
		If[MemberQ[{$WolframLibrary, $WolframRTL, $WolframRTLHeader}, opt],
			opt,
			Throw[Message[errMsgHd::invcdtrgt, opt]; $Failed]
		]
	]
getAndCheckOption[head_, errMsgHd_, opts_, n:"Platform"] :=
	With[{opt = OptionValue[head, opts, n]},
		If[IntegerQ[opt] || opt === Automatic,
			opt,
			Throw[Message[errMsgHd::invplt, opt]; $Failed]
		]
	]
getAndCheckOption[head_, errMsgHd_, opts_, n:"Device"] :=
	With[{opt = OptionValue[head, opts, n]},
		If[IntegerQ[opt] || opt === Automatic,
			opt,
			Throw[Message[errMsgHd::invdev, opt]; $Failed]
		]
	]
getAndCheckOption[head_, errMsgHd_, opts_, n:"Debug"] :=
	With[{opt = OptionValue[head, opts, n]},
		If[opt === True || opt === False,
			opt,
			Throw[Message[errMsgHd::invdbg, opt]; $Failed]
		]
	]

getAndCheckOption[head_, errMsgHd_, opts_, n:"KernelCompileOptions"] :=
	With[{opt = OptionValue[head, opts, n], api = getAndCheckOption[head, errMsgHd, opts, "APITarget"]},
		Which[
			opt === None || StringQ[opt] || ListQ[opt],
				opt,
			opt === Automatic && api === $WGLOpenCL,
				{
					"-Dmint=" <>
						If[$ILP64,
							Switch[$SystemID,
								"Windows-x86-64" | "Linux-x86-64" | "MacOSX-x86-64",
									"long",
								_,
									"int"
							],
							"int"
						]
				},
			opt === Automatic && api === $WGLCUDA,
				{"Defines" -> {mintDefine}},
			True,
				Throw[Message[errMsgHd::invkcopt, opt]; $Failed]
		]
	]
	
mintDefine = "mint" -> If[$ILP64,
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

$ILP64 = TrueQ[Log[2.0, Developer`$MaxMachineInteger] > 32]

End[]
EndPackage[]

