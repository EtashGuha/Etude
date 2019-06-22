(* Mathematica Package *)

(* Created by the Wolfram Workbench 12-May-2008 *)
(*
  Get the maximum dimension of any tensor and use this 
  to init the array.

*)


BeginPackage["GPUTools`CompileCCodeGenerator`", {"SymbolicC`"}]
(* Exported symbols added here with SymbolName::usage *) 

iGPUCompileLibraryGenerate::usage = "iGPUCompileLibraryGenerate[ cfun, name, opts] generates a shared library from the compiled function cfun using name as the exported function name."

iGPUCompileCCodeGenerate::usage = "iGPUCompileCCodeGenerate[ cfun, name, opts] generates C code from the compiled function cfun using name as the exported function name and saves in the file name.c.  GPUCompileCCodeGenerate[ cfun, name, filename] generates output in filename. GPUCompileCCodeGenerate[ cfun, name, {filename, headerfile}] generates any C header output in headerfile."

iGPUCompileSymbolicCGenerate::usage = "iGPUCompileSymbolicCGenerate[ cfun, name, opts] generates symbolic C from the compiled function cfun using name as the exported function name."

iGPUCompileCCodeStringGenerate::usage = "iGPUCompileCCodeStringGenerate[ cfun, name, opts] generates C code from the compiled function cfun using name as the exported function name."

$iGPUCompileCCodeGenerateDirectory::usage ="$iGPUCompileCCodeGenerateDirectory is the location of the C Code Generate Package."

iGPUCompileCCodeExport::usage = "iGPUCompileCCodeExport[ cfun, filename] exports the compiled function cfun as C code using the file filename. GPUCompileCCodeExport[ cfun, {filename, headerfilename}] also exports a header file into the file headerfilename."

iGPUCompileCCodeGenerate::wmreq = "The expression `1` requires Mathematica to be evaluated.   The function will be generated but can be expected to fail with a nonzero error code when executed."

iGPUCompileCCodeGenerate::nosupp = "The expression `1` is not supported for code generation."

iGPUCompileCCodeGenerate::nodp = "The tensor register `1` has not been correctly set up for code generation operations."

toGPUFunctionLoadInputs;

Begin["`Private`"]

Needs /@ {"CompiledFunctionTools`", "CompiledFunctionTools`Opcodes`", "CCompilerDriver`", "CUDALink`NVCCCompiler`"};



compileArgs[HoldPattern[CompiledFunction[_, args_, ___]]] := args

toGPUFunctionLoadInputs[err_, cf_CompiledFunction] := toGPUFunctionLoadInputs[err, compileArgs[cf]]
toGPUFunctionLoadInputs[err_, args_] := toGPUFunctionLoadInput[err, #]& /@ args
toGPUFunctionLoadInput[err_, type:(Verbatim[_Integer] | Verbatim[_Real] | Verbatim[_Complex])] := type
toGPUFunctionLoadInput[err_, {type_}] := toGPUFunctionLoadInput[err, type]
toGPUFunctionLoadInput[err_, {type_, rank_}] := {type, rank, "Input"}
toGPUFunctionLoadInput[err_, a_] := Throw[Message[err::invtyp, a]; $Failed]

iGPUCompileCCodeExport[ cfun_CompiledFunction, fileNameIn_] :=
	Module[ {cproc, files, fileName, name, headerName},
		cproc = ToCompiledProcedure[cfun];
		files = Flatten[ {fileNameIn}];
		If[ Length[ files] < 1, Return[ $Failed]];
		fileName = Part[ files, 1];
		name = FileBaseName[ fileName];
		If[ StringLength[ name] === 0, Return[ $Failed]];
		headerName = If[ Length[ files] > 1, FileBaseName[ Part[files, 2]], Automatic];
		iGPUCompileCCodeGenerate[ cproc, name, fileName, "CodeTarget" -> "WolframRTL", "APITarget" -> "CUDA", "HeaderName" -> headerName];
		If[ Length[ files] > 1
				,
				iGPUCompileCCodeGenerate[ cproc, name, Part[files,2], "CodeTarget" -> "WolframRTLHeader"];
		];
		fileNameIn
	]


$iGPUCompileCCodeGenerateDirectory = DirectoryName[ System`Private`$InputFileName]

Options[iGPUCompileCCodeStringGenerate] = {"CodeTarget" -> "WolframRTL", "APITarget" -> "CUDA", "WrapperFunction" -> None, "HeaderName" -> Automatic, "LifeCycleFunctionNames" -> Automatic}


iGPUCompileCCodeStringGenerate[cfun: Except[_List], name:Except[_List], opts:OptionsPattern[]] :=
	iGPUCompileCCodeStringGenerate[{cfun}, {name}, opts]
	
iGPUCompileCCodeStringGenerate[{cfun_CompiledFunction, funs___}, {name_String, names___}, opts:OptionsPattern[]] :=
	iGPUCompileCCodeStringGenerate[{ToCompiledProcedure[cfun], funs}, {name, names}, opts]
	
iGPUCompileCCodeStringGenerate[{cproc_CompiledProcedure, cprocs___}, {name_String, names___}, opts:OptionsPattern[]] := 
Module[{symbC},
	symbC = SymbolicCGenerateMain[ cproc, name, {cprocs}, {names}, FilterRules[{opts}, Options[SymbolicCGenerateMain]]];
	If[symbC === $Failed, Return[$Failed, Module]];
	ToCCodeString[symbC, "Indent" -> Automatic]
	]


Options[iGPUCompileSymbolicCGenerate] = {"CodeTarget" -> "WolframRTL", "APITarget" -> "CUDA", "WrapperFunction" -> None, "HeaderName" -> Automatic, "LifeCycleFunctionNames" -> Automatic}

iGPUCompileSymbolicCGenerate[cfun: Except[_List], name:Except[_List], opts:OptionsPattern[]] :=
	iGPUCompileSymbolicCGenerate[{cfun}, {name}, opts]

iGPUCompileSymbolicCGenerate[{cfun_CompiledFunction, funs___}, {name_String, names___}, opts:OptionsPattern[]] :=
	iGPUCompileSymbolicCGenerate[{ToCompiledProcedure[cfun],funs}, {name, names}, opts]

iGPUCompileSymbolicCGenerate[{cproc_CompiledProcedure,funs___}, {name_String, names___}, opts:OptionsPattern[]] := 
	SymbolicCGenerateMain[cproc, name, {funs}, {names}, FilterRules[{opts}, Options[SymbolicCGenerateMain]]]


Options[iGPUCompileCCodeGenerate] = {"CodeTarget" -> "WolframRTL", "APITarget" -> "CUDA", "WrapperFunction" -> None, "HeaderName" -> Automatic, "LifeCycleFunctionNames" -> Automatic}


iGPUCompileCCodeGenerate[cfun: Except[_List], name:Except[_List], opts:OptionsPattern[]] :=
	iGPUCompileCCodeGenerate[{cfun}, {name}, opts]

iGPUCompileCCodeGenerate[cfun: Except[_List], name:Except[_List], filename_String, opts:OptionsPattern[]] :=
	iGPUCompileCCodeGenerate[{cfun}, {name}, filename, opts]

iGPUCompileCCodeGenerate[{cfun_CompiledFunction, funs___}, {name_String, names___}, opts:OptionsPattern[]] :=
	iGPUCompileCCodeGenerate[{ToCompiledProcedure[cfun], funs}, {name, names}, Automatic, opts]
	
iGPUCompileCCodeGenerate[ {cfun_CompiledFunction, funs___}, {name_String, names___}, filename_String, opts:OptionsPattern[]] :=
	iGPUCompileCCodeGenerate[{ToCompiledProcedure[cfun], funs}, {name, names}, filename, opts]
	
iGPUCompileCCodeGenerate[ {cfun_CompiledProcedure, funs___}, {name_String, names___}, opts:OptionsPattern[]] :=
	iGPUCompileCCodeGenerate[{cfun, funs}, {name, names}, Automatic, opts]
	
iGPUCompileCCodeGenerate[{cproc_CompiledProcedure, funs___}, {name_String, names___}, filenameIn:(_String|Automatic), opts:OptionsPattern[]] := 
Module[{symbC, codestring, codeTarget, filename},
	codeTarget = OptionValue[ "CodeTarget"];
	filename = filenameIn;
	If[ filename === Automatic, filename = name <> If[ codeTarget === "WolframRTLHeader", ".h", ".c"]];
	symbC = SymbolicCGenerateMain[cproc, name, {funs}, {names}, FilterRules[{opts}, Options[SymbolicCGenerateMain]]];
	If[symbC === $Failed, Return[$Failed, Module]];
	codestring = ToCCodeString[symbC, "Indent" -> Automatic];
	If[!StringQ[codestring], Return[$Failed, Module]];
	Export[filename, codestring, "Text"];
	filename
	]
	
	
iGPUCompileLibraryGenerate[cfun: Except[_List], name:Except[_List], opts:OptionsPattern[]] :=
	iGPUCompileLibraryGenerate[{cfun}, {name}, opts]
		
iGPUCompileLibraryGenerate[{cfun_CompiledFunction, funs___}, {name_String, names___}, opts:OptionsPattern[]] :=
	iGPUCompileLibraryGenerate[{ToCompiledProcedure[cfun], funs}, {name, names}, opts]
		
Options[iGPUCompileLibraryGenerate] = {"CodeTarget" -> "WolframLibrary", "APITarget" -> "CUDA", "WrapperFunction" -> None, "TargetDirectory" -> Automatic, "LifeCycleFunctionNames" -> Automatic}
		
iGPUCompileLibraryGenerate[{cproc_CompiledProcedure,funs___}, {name_String, names___}, opts:OptionsPattern[]] :=
Module[{symbC, cstring, codeTarget},
	codeTarget = OptionValue[ iGPUCompileLibraryGenerate, FilterRules[{opts}, Options[iGPUCompileLibraryGenerate]], "CodeTarget"];
	If[ codeTarget =!= "WolframLibrary" && codeTarget =!= "WolframRTL", Return[$Failed, Module]];
	symbC = SymbolicCGenerateMain[cproc, name, {funs}, {names}, Join[{"CodeTarget"->codeTarget}, FilterRules[{opts}, Options[SymbolicCGenerateMain]]]];
	If[symbC === $Failed, Return[$Failed, Module]];
	cstring = ToCCodeString[symbC];
	If[!StringQ[cstring], Return[$Failed, Module]];
	If[ codeTarget === "WolframLibrary",
		CreateLibrary[ cstring, name, FilterRules[{opts}, Options[CreateLibrary]]]
		,
		GenerateRTL[ cproc, cstring, name, opts]
	]
]
	

GenerateRTL[ cproc_, cstring_, name_, opts:OptionsPattern[]] :=
	Module[ {dir, libDir, hname},
		libDir = FileNameJoin[ {$InstallationDirectory, "SystemFiles", "Libraries", $SystemID}];
		dir = OptionValue[ iGPUCompileLibraryGenerate, FilterRules[{opts}, Options[iGPUCompileLibraryGenerate]], "TargetDirectory"];
		If [!StringQ[ dir], Return[ $Failed, Module]];
		hname = FileNameJoin[ {dir, name <> ".h"}];
		iGPUCompileCCodeGenerate[ cproc, name, hname, "CodeTarget" -> "WolframRTLHeader", FilterRules[{opts}, Options[iGPUCompileCCodeGenerate]]];
		CreateLibrary[ cstring, name, "IncludeDirectories" -> {dir}, "Libraries" -> {"WolframRTL.lib"},"LibraryDirectories" -> {libDir}, FilterRules[{opts}, Options[CreateLibrary]]]
	]




Options[SymbolicCGenerateMain] = {"CodeTarget"->"WolframRTL", "APITarget" -> "CUDA", "WrapperFunction"->None, "CallerData"->Null, "HeaderName" -> Automatic, "LifeCycleFunctionNames" -> Automatic};

SymbolicCGenerateMain[ cfun_CompiledFunction, name_String, cfuns_, names_, opts:OptionsPattern[]] :=
	SymbolicCGenerateMain[ToCompiledProcedure[cfun], name, cfuns, names, opts]
	
SymbolicCGenerateMain[ cproc_CompiledProcedure, name_String, cfuns_, names_, opts:OptionsPattern[]] := 
Module[{objData},
	If[ !ListQ[ cfuns] || !ListQ[ names] || Length[ cfuns] =!= Length[ names], Return[ $Failed, Module]];
	SymbolicCGenerateDriver[objData, name, cproc, Transpose[ {cfuns, names}], opts]
]

(*
	Some SymbolicC utilities
*)
TimesC[args___] := COperator[Times, {args}]
PlusC[args___] := COperator[Plus, {args}]
SubtractC[args___] := COperator[Subtract, {args}]
MinusC[arg_] := COperator[Minus, {arg}]
EqualC[ arg1_, arg2_] := COperator[ Equal, {arg1, arg2}]
LessC[ arg1_, arg2_] := COperator[ Less, {arg1, arg2}]
GreaterC[ arg1_, arg2_] := COperator[ Greater, {arg1, arg2}]
OrC[ args___] := COperator[ Or, {args}]
AndC[ args___] := COperator[ And, {args}]

BitAndC[ arg1_, arg2_] := COperator[ BitAnd, {arg1, arg2}]
BitShiftRightC[ arg1_, arg2_] := COperator[ BitShiftRight, {arg1, arg2}]

initFunctionData[ obj_, _, _, _] := 
	obj[FunctionData]


(*
Init the Function data,  called here the first entry to the 
code generator,  ie not nested calls.  

If setupName is True, then we map the initProc to the initName, 
this is done to make sure that recursive calls to WolframDLL -> False, 
ref the original function rather than generating a duplicate. 
*)
initFunctionData[ Null, setupName_, initName_, initProc_] :=
	Module[ {funcObj},
		If[ setupName,
			funcObj[CalledFunctionName][initProc] = initName];
		funcObj[CalledFunctionCount] = 0;
		funcObj[CalledFunctions] = {};
		funcObj[FunctionPointerCount] = 0;
		funcObj[DLLFunctions] = {};
		funcObj[DLLFunctionArgNum] = 0;
		funcObj[TensorConstantCount] = 0;
		funcObj[ConstantDeclarationCode] = {};
		funcObj[ConstantInitializationCode] = {};
		funcObj[ConstantDeallocationCode] = {};
		funcObj[ErrorLabel] = False;
		funcObj[StringBufferLength] = 0;
		funcObj[NextIndex] = 1;
		funcObj
	]


(*
 Initialize the data object used by the conversion
*)

initObjData[ objData_, lines_, codeTarget_, apiTarget_, callerData_, wrapper_, cons_, name_, proc_] :=
Module[{cinfo = First[proc]},
	objData[Tensor] = False;
	objData[ArrayArgs] = SameQ[codeTarget, "WolframLibrary"];
	objData[WolframRTL] = SameQ[codeTarget, "WolframRTL"] || SameQ[codeTarget, "WolframRTLHeader"];
	objData[ExtraWolframRTL] = SameQ[codeTarget, "ExtraWolframRTL"];
	objData[ResultWGLArgument] = (callerData === Null) && !objData[WolframRTL];
	objData[returnAtEnd] = True;
	objData[ReturnLabel] = False;
	objData[lineLength] = Length[ lines];
	objData[ struct[x_]] := CPointerMember[objData[LibraryData], x];
	objData[ gpuStruct[x_]] := CPointerMember[objData[WGLData], x];
	objData[ structCompile[x_]] := (If[objData[WolframRTL], objData[LibraryData]]; CPointerMember["funStructCompile", x]);
	objData[DataPointerCount] = 0;
	objData[DataPointer[_]] = None;
	objData[DataPointerDeclaration[__]] = {};
	objData[DataDimensions[_]] = None;
	objData[DataDimensionsDeclaration[__]] = {};
	objData[DataNElems[_]] = None;
	objData[DataNElemsDeclaration[__]] = {};
	objData[DimensionArrayLength] = 0;
	objData[ErrorLabel] = False;
	objData[ExpressionPointerCount] = 0;
	objData[DLLFunctionArgNum] = 0;
	objData[CallerData] = callerData;
	If[callerData === Null,
		objData[Index] = 0;
		objData[FunctionData] = initFunctionData[ callerData, codeTarget === None || codeTarget === "NestedFunction", name, proc]
	(* else *),
		objData[FunctionData] = callerData[FunctionData];
		objData[Index] = objData[FunctionData][NextIndex];
		objData[FunctionData][NextIndex] = objData[Index] + 1;
	];
	objData[Wrapper] = If[objData[ArrayArgs], wrapper, None];
	objData[Constants][_] := $Failed; (* Default value for non constant registers *)
	Scan[makeConstantRule[objData], cons];
	objData[RuntimeFlags] = rtFlags;
	objData[ArithmeticFlags] = cinfo["ArithmeticFlags"];
	objData[RuntimeFlags] = cinfo["RuntimeFlags"];
	objData[LibraryData] = "libData";
	objData[gpuData] = "wglData";
];

(*
 Type utilities
*)

convertType[ Boolean] = Boolean
convertType[ Integer] = Integer
convertType[ Real]    = Real
convertType[ Complex] = Complex
convertType[VoidType] = VoidType

convertFullType[ Tensor[ type_, rank_]] := {convertType[type], rank}
convertFullType[ type_] := {convertType[ type], 0}
convertFullType[ Instruction[Set, type_, _]] := convertFullType[ type]
convertFullType[ Register[ type_, _]] := convertFullType[ type]

dllFunctionArgumentType[Tensor[ type_, rank_]] := {convertType[type], rank, "Constant"}
dllFunctionArgumentType[type_] := {convertType[type], 0}
dllFunctionArgumentType[Instruction[Set, type_, _]] := dllFunctionArgumentType[type]
dllFunctionArgumentType[Register[ type_, _]] := dllFunctionArgumentType[type]

typeToCType[ Boolean] = "mbool"
typeToCType[ Integer] = "mint"
typeToCType[ Real] = "mreal"
typeToCType[Complex] = "mcomplex"
typeToCType[ Tensor] = "WGL_Memory_t"
typeToCType[ Tensor[ _,_]] = "WGL_Memory_t"
typeToCType[ VoidType] = "void"


typeToCType[MArgument] = "MArgument"
typeToCType[WGLArgument] = "WGLArgument"

typeToVariable[ Boolean] = "B"
typeToVariable[ Integer] = "I"
typeToVariable[ Real] = "R"
typeToVariable[Complex] = "C"
typeToVariable[ Tensor] = "M"

(* This is used to determine which 
   types can be used with the C operators
   like +, *, etc. *)
CBasicTypeQ[{Boolean, 0}] = True;
CBasicTypeQ[{Integer, 0}] = True;
CBasicTypeQ[{Real, 0}] = True;
CBasicTypeQ[_] = False;

getTensorEnumType[ Register[ Tensor[ type_, rank_], num_]] := ToTypeEnum[type]

getRank[ Register[ Tensor[ type_, rank_], num_]] = rank
getRank[Register[type_, num_]] = 0;

getType[Register[Tensor[type_, rank_], num_]] = type
getType[Register[type_, num_]] = type;

isTensor[ Register[ Tensor[ type_, rank_], num_]] := True

isTensor[ Register[ type_, num_]] := False 

RankZeroTensorQ[Register[Tensor[type_, 0], _]] := True;
RankZeroTensorQ[_] := False;

BooleanEnumType = ToTypeEnum[Boolean]
IntegerEnumType = ToTypeEnum[Integer]
RealEnumType = ToTypeEnum[Real]
ComplexEnumType = ToTypeEnum[Complex]



getCType[ Register[ type_, reg_]] :=
	typeToCType[type]

getCType[ Register[ Tensor[type_,rank_], reg_]] :=
	typeToCType[Tensor]
	
WGLArgumentName[Register[_Tensor, _]] := "Memory"
WGLArgumentName[Register[type_Symbol, _]] := SymbolName[type]

WGLArgumentGetInteger[reg_] := CCall["MArgument_getInteger", {reg}]
WGLArgumentSetInteger[reg_, val_] := CCall["MArgument_setInteger", {reg, val}]
WGLArgumentGetFunction[reg_] := StringJoin["WGLArgument_get", WGLArgumentName[reg]]
WGLAgumentGetAddressFunction[reg_] := StringJoin["WGLArgument_get", WGLArgumentName[reg], "Address"]
WGLArgumentSetFunction[reg_] := StringJoin["WGLArgument_set", WGLArgumentName[reg]]

(*
 Format atoms,  might be registers, arguments, or numbers
*)

makeAtom[ base_, num_, objData_] := 
Module[{index = objData[Index]},
	If[Not[IntegerQ[index]],
		base <> ToString[num],
	(* else *)
		base <> ToString[index] <> "_" <> ToString[num]
	]
];

makeAtom[ Register[Tensor[type_, _], num_], objData_] :=
	makeAtom[ typeToVariable[Tensor], num, objData]
	
makeAtom[ Register[ type_, num_], objData_] :=
	makeAtom[ typeToVariable[ type], num, objData]
	
makeAtom[ Argument[num_], objData_] :=
	makeAtom[ "A", num, None]
	
makeAtom[LibraryData, objData_] := objData[LibraryData]
makeAtom[ArgumentCount, objData_] := "Argc"
makeAtom[ ArgumentArray[], objData_] := "Args"
makeAtom[ Result[], objData_] := "Res"

makeAtom[i_ ?NumberQ, objData_] := ToString[i, CForm]

makeAtom[True, objData_] = 1
makeAtom[False, objData_] = 0


(* For termporary arguments used for inline code *)
makeAtom[Temporary[num_], objData_] := makeAtom["S", num, None]

(*
	Like makeAtom, but dereferences Memory variables
*)
makeTDAtom[Register[Tensor[type_, _], num_], objData_] :=
	CDereference[makeAtom[ typeToVariable[Tensor], num, objData]]
	
makeTDAtom[r_, objData_] := makeAtom[r, objData]

(*
	Gets the address of a register - for tensor locals
	the local is the address 
*)
makeAtomAddress[reg:Register[Tensor[type_, _], num_], objData_] := makeAtom[reg, objData]

makeAtomAddress[reg_, objData_] := CAddress[makeAtom[reg, objData]]

(* This handles the 0 length array case to avoid compiler errors *)
DeclareArray[type_, var_, 0] := CDeclare[type, CAssign[CDereference[var], 0]]
DeclareArray[type_, var_, n_] := CDeclare[type, CArray[var, n]]

(* Data pointers used to inline vector setting/getting.

   Whenever a Tensor register, reg,  is modified in a way that
   could reallocate the data, setDataPointer[objData, reg]
   should be called immediately after the possible 
   reallocation.
   
   Then, for commands that can use the data pointer directly,
   use getDataPointer[objData, reg].  The first time this is 
   called, it sets up a declaration of the data pointer
   and an assignment that will be put in where
   setDataPointer[objData, reg].
   
   Note that setDataPointer[objData, reg] should be treated as
   generating a line of C code, so a typical use would be
   {CCall[possible reallocation function, {reg}],
    setDataPointer[objData, reg]}
   
*)

SetAttributes[UndefinedQ, HoldAll];
UndefinedQ[fun_[args___]] := SameQ[Head[fun[args]], fun]

getMemoryDataFunction[objData_, type_] := objData[structCompile["Memory_get" <> ToString[type] <> "Data"]]
	
DPDeclareType[type_, locality_] := 
Module[{dtype = typeToCType[type]},
	If[locality === "static", dtype = {"static", dtype}];
	dtype
];

makeDataPointer[objData_, reg_, var_, count_, type_, locality_] := (
	objData[DataPointer[reg]] := (
		If[UndefinedQ[objData[DataPointerCode[var]]],
			objData[DataPointerDeclaration[count, locality]] = CDeclare[DPDeclareType[type, locality], CDereference[var]];
     		objData[DataPointerCode[var]] = CAssign[var, CCall[getMemoryDataFunction[objData, type], {makeTDAtom[reg, objData]}]]
     	];
   		var
    )
)

getMemoryDimensionsFunction[objData_] := objData[structCompile["Memory_getDimensions"]]
	
makeDataDimensions[objData_, reg_, var_, count_, locality_] := (
 	objData[DataDimensions[reg]] := (
		If[UndefinedQ[objData[DataDimensionsCode[var]]],
			objData[DataDimensionsDeclaration[count, locality]] = CDeclare[DPDeclareType[Integer, locality], CDereference[var]];
     		objData[DataDimensionsCode[var]] = CAssign[var, CCall[getMemoryDimensionsFunction[objData], {makeTDAtom[reg, objData]}]]
     	];
  	 	var
 	)
)

getMemoryNElemsFunction[objData_] := objData[structCompile["Memory_getFlattenedLength"]]

makeDataNElems[objData_, reg_, var_, count_, locality_] := (
 	objData[DataNElems[reg]] := (
		If[UndefinedQ[objData[DataNElemsCode[var]]],
			objData[DataNElemsDeclaration[count, locality]] = CDeclare[DPDeclareType[Integer, locality], var];
     		objData[DataNElemsCode[var]] = CAssign[var, CCall[getMemoryNElemsFunction[objData], {makeTDAtom[reg, objData]}]]
     	];
  	 	var
    );
)

DataPointerNum[objData_, reg_] := 
Set[
	DataPointerNum[objData, reg],
	Module[{n = objData[DataPointerCount]},
		objData[DataPointerCount] = n + 1;
		n
	]
];

(* Sets data pointer for reg
   Note: This returns the instruction for the setting *)
setDataPointer[objData_, reg:Register[Tensor[type_, rank_], num_]] := 
Module[{var, dims, nelems, n = DataPointerNum[objData, reg], locality},
	locality = If[UnsameQ[objData[Constants][reg], $Failed], "static", "local"];
	var = "P" <> ToString[n];
	dims = "D" <> ToString[n];
	nelems = "L" <> ToString[n];
	makeDataPointer[objData, reg, var, n, type, locality];
	makeDataDimensions[objData, reg, dims, n, locality];
	makeDataNElems[objData, reg, nelems, n, locality];
	{objData[DataPointerCode[var]], objData[DataDimensionsCode[dims]], objData[DataNElemsCode[nelems]]}
]

(* No code for scalar registers *)
setDataPointer[objData_, reg_] := {};

DataPointerCheck[None, reg_, objData_] := Throw[Message[iGPUCompileCCodeGenerate::nodp, reg]; $Failed, "UnsetDataPointer"]
DataPointerCheck[dp_, reg_, objData_] := dp

getDataPointer[objData_, reg_] := DataPointerCheck[objData[DataPointer[reg]], reg, objData]

getDataDimensions[objData_, reg_] := DataPointerCheck[objData[DataDimensions[reg]], reg, objData]

getDataNElems[objData_, reg_] := DataPointerCheck[objData[DataNElems[reg]], reg, objData]


unsetDataPointer[reg:Register[Tensor[rank_, type_], num_]] := If[StringQ[DataPointer[reg]], Unset[DataPointer[reg]]]
(* This definition allows us to call it on non tensor registers without any risk of damage *)
unsetDataPointer[_] := Null

(* This returns a variable that can
   is a mint array that be used for dimensions
   The argument here is the length required for this use *)
getDimensionArray[objData_, len_] := 
(
	objData[DimensionArrayLength] = Max[objData[DimensionArrayLength], len];
	"dims"
)

(* 
	Error checking.
	
	CheckError[objData, code] generates
	
	err = code;
	if (err) return err;
*)

makeAtom[ErrorVariable, objData_] := "err"
ErrorVariableCType = "int";

CheckError[objData_, code_] := 
(
	If[objData[ErrorLabel] === False, objData[ErrorLabel] = "error_label"];
	{
		CAssign[makeAtom[ErrorVariable, objData], code],
		CIf[makeAtom[ErrorVariable, objData], CGoto[objData[ErrorLabel]]]
	}
)

CheckError[objData_, code_, comment_] := 
(
	If[objData[ErrorLabel] === False, objData[ErrorLabel] = "error_label"];
	{
		CAssign[makeAtom[ErrorVariable, objData], code],
		comment,
		CIf[makeAtom[ErrorVariable, objData], CGoto[objData[ErrorLabel]]]
	}
)

(*
	Check register reg for a floating point exception and
	return LIBRARY_NUMERICAL_ERROR if there is one
*)
CheckException[objData_, reg_, checkflags_] := 
Module[{type},
	type = If[getRank[reg] > 0, 0, ToTypeEnum[getType[reg]]];
	CheckError[objData, 
		CConditional[
			CCall[
				objData[structCompile["checkFloatingPointException"]], 
				{makeAtomAddress[reg, objData], type, checkflags}
			], 
			"LIBRARY_NUMERICAL_ERROR", 
			0
		]
	]
];


(* Force an error return:
   errorcode should be a nonzero int *)
ForceError[objData_, errcode_] := 
(
	If[objData[ErrorLabel] === False, objData[ErrorLabel] = "error_label"];
	{
		CAssign[makeAtom[ErrorVariable, objData], errcode],
		CGoto[objData[ErrorLabel]]
	}
)

(* 
 Set up functions to hook into macros in WolframLibrary.h for Re, Im of mcomplex 
*)

ReC[ x_] := CCall["mcreal", {x}]
ImC[ x_] := CCall["mcimag", {x}]


getDerefPointer[ Register[ type_, num_]] :=
	ToString[typeToCType[ type]] <> "*"

getDerefPointer[ Register[ Tensor[ type_, _], num_]] :=
	ToString[typeToCType[ Tensor]] <> "*"



(*
  build an individual argument
*)
buildIndividualArg[ type_, num_, objData_] := {typeToCType[ type], makeAtom[ Argument[num], objData]}

(*
 This constructs the arguments to the function, 
 either we put the args in or we just put a single 
 argument.
*)
buildFunctionArgs[ args_List, objData_] :=
	Module[ {resultType},
		If[objData[ArrayArgs],
			{{"WolframLibraryData", makeAtom[LibraryData, objData]},
			 {typeToCType[Integer], makeAtom[ArgumentCount, objData]},
			 {typeToCType[MArgument], CDereference[makeAtom[ ArgumentArray[], objData]]},
			 {typeToCType[MArgument], makeAtom[ Result[], objData]}}
		(* else *),		
			resultType = getCType[ objData[ResultRegister]];
			Join[
				{{"WolframLibraryData", objData[LibraryData]}},
				Table[ buildIndividualArg[ args[[i]], i, objData], {i,Length[ args]}],
				{{resultType, CDereference[makeAtom[ Result[], objData]]}}
			]
		]
	];

buildArgumentAssign[ initInst_List, objData_] :=
Module[{tensorargs},
	objData[ ArgumentTypes] = Map[ dllFunctionArgumentType, initInst];
	tensorargs = Flatten[Position[objData[ArgumentTypes], {_, x_ /; x > 0, _}, {1}]];
	objData[TensorArgs] = Table[makeAtom[initInst[[i, 2]], objData], {i, tensorargs}];
	If[objData[ArrayArgs],
		buildArgumentArrayAssign[initInst, objData],
		buildPassedArgumentAssign[initInst, objData]
	]
]

buildPassedArgumentAssign[ initInst_List, objData_] := 
	Table[makePassedArgumentAssign[objData, inst], {inst, initInst}]
		
buildArgumentArrayAssign[ initInst_List, objData_] :=
	Table[ makeArgumentArrayAssign[ objData, initInst[[i]], i-1], {i, Length[initInst]}]

getArrayReference[ objData_, reg:Register[Tensor[__], _], arrRef_, var_] :=
	CAssign[var, CCall[CPointerMember["wglData", "getMemory"], {objData[gpuData], WGLArgumentGetInteger[arrRef]}]]

getArrayReference[ objData_, reg_, arrRef_, var_] :=
	CAssign[var, CCall[WGLArgumentGetFunction[reg], {arrRef}]]

setWolframLibraryData[objData_] :=
	CCall[objData[structCompile["SetWolframLibraryData"]], objData[LibraryData]]

makePassedArgumentAssign[objData_, Instruction[Set, reg_, arg_]] :=
	If[getRank[reg] > 0, 
		{
			CAssign[makeAtom[reg, objData], CAddress[makeAtom[arg, objData]]],
			setDataPointer[objData, reg]
		}
	(* else *),
		CAssign[makeAtom[reg, objData], makeAtom[arg, objData]]
	];
	

makeArgumentArrayAssign[ objData_, Instruction[Set, reg_, arg_], argNum_] :=
	Module[ {arrRef},
		arrRef = makeAtom[ ArgumentArray[], objData];
		arrRef =  CArray[ arrRef, argNum];
		{
			getArrayReference[ objData, reg, arrRef, makeAtom[ reg, objData]],
			If[getRank[reg] > 0, setDataPointer[objData, reg],{}]
		}
	]



(*
 Compute the return code,  this is called from the main driver 
 or from processing a return instruction that does not come at 
 the end and when we are not deviating to the end.
*)

getReturnCode[ objData_] :=
	Module[ {resRef, resReg, resVar, resT},
		resReg = objData[ ResultRegister];
		resVar = objData[ resVariable];
		{
			If[objData[ReturnLabel] =!= False,
				CLabel[objData[ReturnLabel]]
				,
				{}
			],
			If[resVar === "void",
				{}
			,
				resRef = makeAtom[Result[], objData];
				If[isTensor[resReg],
					If[objData[ArrayArgs] && objData[ ResultWGLArgument],
						CBlock[{
							CDeclare["WGL_Memory_t", CAssign["resMem", "NULL"]],
							CCall[objData[structCompile["Memory_copy"]], {CAddress["resMem"], CDereference[resVar]}],
							WGLArgumentSetInteger[resRef, CPointerMember["resMem", "id"]]
						}]
					(* else *), 
						resT = resRef;
						CCall[objData[structCompile["Memory_copy"]], {resT, CDereference[resVar]}]
					]
			(* else *),
					If[ objData[ ResultWGLArgument],
						CCall[WGLArgumentSetFunction[resReg], {resRef, makeAtom[resReg, objData]}]
						,
						resT = CDereference[resRef];
						CAssign[ resT, makeAtom[ resReg, objData]]
					]
				]
			],
			If[objData[ErrorLabel] =!= False,
				CLabel[objData[ErrorLabel]]
				,
				{}
			],
			If[StringQ[objData[TensorLocals]],
				CCall[
					objData[structCompile["ReleaseInitializedMemories"]], 
					{objData[TensorLocals]}
				]
			(* else *),
				{}
			],
			If[objData[WolframRTL] || objData[ExtraWolframRTL],
				CCall[
					objData[structCompile["WolframLibraryData_cleanUp"]], 
					{objData[LibraryData], If[TrueQ[BitAnd[objData[RuntimeFlags], CleanRegistersFlag] > 0], 1, 0]}
				]
			(* else *),
				{}
			],
			CReturn[If[objData[ErrorLabel] =!= False, makeAtom[ErrorVariable, objData], 0]]
		}
	]
					



(*
  build local variables to the function
*)

declareLocals[objData_, cType_, {}, tensorQ_, count_Symbol] := {};

declareLocals[objData_, cType_, locals_, tensorQ_, count_Symbol] := 
Module[{decls, cdecls, initF = Function[#], cLType = cType, base},
	If[tensorQ, 
		cLType = CPointerType[cType];
		initF = Function[CAssign[#, 0]]
	];
	count = 0;
	{decls, cdecls} = Part[Reap[
		Table[
			If[objData[Constants][local] === $Failed,
				Sow[CDeclare[cLType, local], 0],
			(* else *)
				count++;
				If[tensorQ, 
					base = local <> "B";
					Sow[CDeclare[{"static", cType}, CAssign[base, 0]], 1];
					Sow[CDeclare[{"static", cLType}, CAssign[local, CAddress[base]]], 1]
				(* else *),
					Sow[CDeclare[{"static", cType}, local], 1]
				]
			],
			{local, locals}
		],
		{0, 1}
	], 2];
	If[Length[cdecls] > 0, 
		objData[FunctionData][ConstantDeclarationCode] = {objData[FunctionData][ConstantDeclarationCode], cdecls};
	];
	decls
];

buildLocal[ objData_, {type_, num_}] :=
	Module[ {cType, baseName, locals, decls, count},
		cType = typeToCType[type];
		baseName = typeToVariable[ type];
		locals = Table[ makeAtom[ baseName, i, objData], {i,0,num-1}];
		decls = declareLocals[objData, cType, locals, type === Tensor, count];
		If[(type === Tensor) && (Length[locals] > 0),
			objData[Tensor] = True;
			locals = Complement[locals, objData[TensorArgs]];
			If[Length[locals] > count, 
				locals = Select[locals, Function[SameQ[objData[Constants][#], $Failed]]];
				objData[TensorLocals] = "Minit";
				decls = Append[decls, CDeclare["MemoryInitializationData", objData[TensorLocals]]];
				objData[TensorLocalsInit] = {
					CAssign[objData[TensorLocals], 
						CCall[objData[structCompile["GetInitializedMemories"]], {objData[LibraryData], Length[locals]}]],
					Table[
						CAssign[locals[[i]], 
							CCall["MemoryInitializationData_getMemory", {objData[TensorLocals], i - 1}]],
						{i, Length[locals]}
					]
				};
			];
		];
		decls
	];	

buildLocals[ objData_, locals_List] :=
	Map[ buildLocal[objData, #]&, locals]
	
	
	
	
(*
 Tools for working with Constants
*)

(*
	For now this is identical to setting  
	in the body of the function, so avoid repeated code.
*)
makeAtom[TensorConstant[i_], objData_] := makeAtom["TC", i, objData];

setupConstant[objData_, instr:Instruction[Set, _, _]] :=
(
	buildLine[0, objData, instr]
)

cleanupConstant[objData_, instr:Instruction[Set, r:Register[_Tensor, _], _]] :=
Module[{tc = makeTDAtom[r, objData]},
	CIf[tc, 
		CBlock[
			CCall[objData[structCompile["Memory_free"]], {tc}],
			CAssign[tc, 0]
		]
	]
];

cleanupConstant[objData_, instr:Instruction[Set, _, _]] := {};


setupConstants[ objData_, consData_] :=
Module[{funcData = objData[FunctionData], errlab = objData[ErrorLabel]},
	(* Localize this so we get errors in init function *)
	objData[ErrorLabel] = False;
	funcData[ConstantInitializationCode] = Flatten[{
		funcData[ConstantInitializationCode], 
		Map[setupConstant[ objData, #]&, consData]
	}];
	funcData[ConstantDeallocationCode] = Flatten[{
		funcData[ConstantDeallocationCode], 
		Map[cleanupConstant[ objData, #]&, consData]
	}];
	If[SameQ[funcData[ErrorLabel], False], funcData[ErrorLabel] = objData[ErrorLabel]];
	objData[ErrorLabel] = errlab;
];

makeConstantRule[objData_][Instruction[Set, reg_Register, val_]] :=
With[{r = makeAtom[reg, objData], head = objData[Constants]},
	Set[head[reg], val];
	Set[head[r], val]
];


Options[ SymbolicCGenerateDriver] = {"CodeTarget" -> "WolframRTL", "APITarget" -> "CUDA", "CallerData" -> Null, "WrapperFunction"->None, "HeaderName" -> Automatic, "LifeCycleFunctionNames" -> Automatic}

SymbolicCGenerateDriver[
	objData_,
	name_,
	proc:CompiledProcedure[
		CompiledInfo[ args_List, locals_List, "RuntimeFlags"->rtFlags_],
		CompiledSetup[ initInst_],
		CompiledConstants[ cons_],
		CompiledResult[ result_],
		lines_List,
		codes_
		], 
		extraFuns_List,
		OptionsPattern[]] :=
Catch[
	Block[{CBlock}, (* Flattening is delayed to improve code readability *)
	Module[ {resType, funArgs, localVars, initCode, assignArgs, body, code, setLibraryData,
			cleanCode, retCode, function,  codeTarget, apiTarget, callerData, wrapper, consCode, extraCode,
			headerName, lifeCycleFuncNames},
		codeTarget = OptionValue["CodeTarget"];
		apiTarget = OptionValue["APITarget"];
		callerData = OptionValue["CallerData"];
		headerName = OptionValue[ "HeaderName"];
		lifeCycleFuncNames = OptionValue[ "LifeCycleFunctionNames"];
		If[ headerName === Automatic, headerName = name];
		wrapper = OptionValue["WrapperFunction"];
		initObjData[ objData, lines, codeTarget, apiTarget, callerData, wrapper, cons, name, proc];
		objData[ LifeCycleFunctionNames] = lifeCycleFuncNames;
		objData[ ResultRegister] = result;
		objData[ ReturnType] = convertFullType[result];
		resType = getCType[ result];
		If[ resType === "void",
			objData[resVariable] = "void",
			objData[resVariable] = makeAtom[ result, objData]];
		setLibraryData = setWolframLibraryData[objData];
		assignArgs = buildArgumentAssign[ initInst, objData];
		consCode = setupConstants[ objData, cons];
		localVars = Flatten[buildLocals[ objData, locals]];
		initCode = {setLibraryData};
		cleanCode = {};
		retCode = {};
		code = buildLines[ lines, objData];
		code = fixLabels[ objData, code];
		If[objData[DimensionArrayLength] > 0,
			localVars = {localVars, CDeclare["mint", CArray["dims", {objData[DimensionArrayLength]}]]}
		];
		If[objData[DataPointerCount] > 0,
			localVars = {localVars, 
				Table[{objData[DataPointerDeclaration[i, "local"]], objData[DataDimensionsDeclaration[i, "local"]], objData[DataNElemsDeclaration[i, "local"]]}, 
					{i, 0, objData[DataPointerCount] - 1}]};
			objData[DataPointerCode[_]] := {};
			objData[DataDimensionsCode[_]] := {};
			objData[DataNElemsCode[_]] := {};
		];
		If[objData[DLLFunctionArgNum] > 0,
			localVars = {localVars, CDeclare["WGLArgument", 
				CArray[makeAtom[FunctionPointerArguments, objData], 
					objData[DLLFunctionArgNum]]]}
		];
		If[objData[ErrorLabel] =!= False,
			localVars = {localVars,
				CDeclare[ErrorVariableCType, CAssign[makeAtom[ErrorVariable, objData], 0]]};
		];
		localVars = Flatten[localVars];
		If[StringQ[objData[TensorLocals]], 
			initCode = {initCode, objData[TensorLocalsInit]};
		];
		retCode = getReturnCode[ objData];
		body = CBlock[ {localVars,  initCode, assignArgs, code, cleanCode, retCode}];
		funArgs = buildFunctionArgs[args, objData];
		wrapper = buildWrapperFunction[objData];
		resType = "int";
		extraCode = Flatten[ Map[ getExtraCode[ objData, codeTarget, #]&, extraFuns]];
		Switch[codeTarget, 
			"WolframLibrary", 
				function = CFunction[ {"DLLEXPORT", resType}, name, funArgs, body];
				CProgram[
					getIncludes[objData],
					getDLLHeader[objData],
					getCompileDLLHeader[objData, False],
					getVersionFunction[],
					getInitializeFunction[objData, name], 
					getUninitializeFunction[objData, name],
					getDLLFunction[objData, name],
					getWGLFunction[objData, name],
					getDLLUninitializeFunction[objData, name],
					getNameFunction[objData, name],
					wrapper,
					getArgumentsFunction[objData],
					CProgram @@ objData[FunctionData][ CalledFunctions],
					"",
					function],
			"WolframRTL", 
				function = CFunction[ {"DLLEXPORT", resType}, name, funArgs, body];
				CProgram[
					getIncludes[objData],
					getRTLHeader[objData],
					getCompileDLLHeader[objData, True],
					includeThisHeader[objData, headerName],
					getInitializeFunction[objData, name], 
					getUninitializeFunction[objData, name],
					objData[FunctionData][ CalledFunctions],
					function,
					extraCode],
			"WolframRTLHeader",
					{
					buildThisHeaderFile[objData, name,  CFunction[ {"DLLEXPORT", resType}, name, funArgs]],
					extraCode
					},
			"NestedFunction",
				CFunction[ {"static", resType}, name, funArgs, body],
			"ExtraWolframRTL",
				CFunction[ {"DLLEXPORT", resType}, name, funArgs, body],
			"ExtraWolframRTLHeader",
				CFunction[ {"DLLEXPORT", resType}, name, funArgs],
			_,
				CProgram[
					CFunction[ resType, name, funArgs, body]
				]
		]
	(*Module*)](*Block*)],
	_ (* pattern for Catch *)
]




getExtraCode[ objData_, codeTarget_, {cfun_CompiledFunction, name_String}] :=
	getExtraCode[ objData, codeTarget, {ToCompiledProcedure[cfun], name}]
	
	


getExtraCode[ objData_, codeTarget_, {cfun_CompiledProcedure, name_String}] :=
	Module[ {newTarget},
		newTarget =
			Switch[
				codeTarget,
					"WolframRTL", "ExtraWolframRTL",
					"WolframRTLHeader", "ExtraWolframRTLHeader",
					_, None
			];
		If[ newTarget === None, Return[ {}, Module]];
		iGPUCompileSymbolicCGenerate[cfun, name, "CodeTarget" -> newTarget, "CallerData" -> objData]
	]
	
getExtraCode[ _,_,_] :=
	{}	


getIncludes[objData_] :=
	{
		CInclude["math.h"](* This is essential if any math library function 
		                       are used.  TODO: perhaps only do this if we
		                       use the math library *)
	}




getDLLHeader[objData_] :=
	{
		CInclude[ "wgl.h"]
	}

getCompileDLLHeader[objData_, RTL_] :=
	Module[{funcData},
		funcData = objData[FunctionData];
		Flatten[{
			If[ RTL, 
				CDeclare[ {"static", "WolframGPUCompileLibrary_Functions"}, "funStructCompile"]
			(* else *),
				{
				CInclude[ "wgl_compile.h"],
				CInclude[ "wgl_types.h"],
				CDeclare[ {"static", "WolframGPULibraryData"}, CAssign["wglData", "NULL"]],
				CDeclare[ {"static", "WolframGPUCompileLibrary_Functions"}, CAssign["funStructCompile", "NULL"]]
				}
			]
			,
			If[objData[ExpressionPointerCount] > 0,
				CProgram[Flatten[{
					Table[
						CDeclare[ {"static", "void *"}, CAssign[makeAtom[ExpressionPointer[i], objData], 0]],
						{i, 0, objData[ExpressionPointerCount] - 1}
					]
				}]]
			,
				{}
			],
			If[StringQ[funcData[MathUnitIncrements]],
				{
					CDeclare[{"static", "const", typeToCType[Integer]}, 
						CAssign[ CArray[funcData[MathUnitIncrements], {3}],"{1, 1, 1}"
				]]}
			,
				{}
			],
			If[funcData[FunctionPointerCount] > 0,
				CProgram[Flatten[{
					Table[
						CDeclare[{"static", funcData[FunctionPointerType][i]}, 
							makeAtom[FunctionPointer[i], objData]], 
						{i, 0, funcData[FunctionPointerCount] - 1}
					],
					If[funcData[DLLFunctionArgNum] > 0, 
						CDeclare[{"static", "WGLArgument"}, 
							CArray[makeAtom[FunctionPointerArguments, objData], 
								funcData[DLLFunctionArgNum]]]
					(* else *), 
						{}
					]
				}]]
				,
				{}
			],
			funcData[ConstantDeclarationCode],
			If[objData[DataPointerCount] > 0,
				Table[{objData[DataPointerDeclaration[i, "static"]], objData[DataDimensionsDeclaration[i, "static"]], objData[DataNElemsDeclaration[i, "static"]]}, 
					{i, 0, objData[DataPointerCount] - 1}],
				{}
			],
			If[funcData[StringBufferLength] > 0,
				{CDeclare["static char", CArray[makeAtom[StringBuffer, objData], {funcData[StringBufferLength]}]]}
			(* else *),
				{}
			],
			CDeclare[{"static",  "mbool"}, CAssign[initializeVar, 1]]
		}]
	]


getRTLHeader[objData_] :=
	{
		CInclude[ "wgl_compile.h"]
	}
	
includeThisHeader[objData_, name_] := 
	CInclude[name <> ".h"]
	
buildThisHeaderFile[objData_, name_, fun_] := 
Module[{args},
	args = {{"WolframGPULibraryData", "wglData"}};
	CProgram[
		CInclude["wgl.h"],
		CFunction[{"DLLEXPORT", "int"}, InitializeFunctionName[objData, name], args],
		CFunction[{"DLLEXPORT", "void"}, UninitializeFunctionName[objData, name], args],
		fun
	]	
]
		

InitExportString[objData_] := 
	If[TrueQ[objData[ArrayArgs]], 
		{"DLLEXPORT"},
		"static"
	];
	

getDLLFunction[ objData_, name_] :=
	CFunction[ {InitExportString[objData], "int"}, "WolframLibrary_initialize", 
		{{"WolframLibraryData", "libData"}},
		CBlock[
			{
			CReturn["LIBRARY_NO_ERROR"]
			}
		]]	

getWGLFunction[ objData_, name_] :=
	CFunction[ {InitExportString[objData], "int"}, "WolframGPULibrary_initialize", 
		{{"WolframGPULibraryData ", "wglData0"}},
		CBlock[
			{
			CReturn[CCall[InitializeFunctionName[objData, name], {"wglData0"}]]
			}
		]]
	
(* 
	Initializations for using the RuntimeLibrary
*)

getVersionFunction[] := 
	CFunction[{"DLLEXPORT", typeToCType[Integer]}, "WolframLibrary_getVersion", {},
		CBlock[{
			CReturn["WolframLibraryVersion"]
		}]
	]

getNameFunction[objData_, name_] :=
	CFunction[{"DLLEXPORT", "char *"}, "WolframGPUCompileLibrary_exportName", {},
		CBlock[{
			CReturn[CString[name]]
		}]
	]
	
	
(* Fits within Microsoft limitation of 2048 bytes *)
$StringConstantLengthLimit = 2000;

makeAtom[StringBuffer, objData_] := "strbuf"

SetAttributes[GetStringConstant, HoldFirst];
GetStringConstant[bufvar_Symbol, string_String, objData_] := 
Module[{len = StringLength[string], funcData},
	If[len <= $StringConstantLengthLimit,
		bufvar = CString[string];
		{} (* no code *)
	(* else *),
		funcData = objData[FunctionData];
		If[funcData[StringBufferLength] < len + 1,
			funcData[StringBufferLength] = len + 1;
		];
		bufvar = makeAtom[StringBuffer, objData];
		SplitUpStringConstant[bufvar, string, objData]
	]
]
		
SplitUpStringConstant[buf_, string_String, objData_] := 
Module[{len, i, j, k, n, temp, substrings, csub},
	len = StringLength[string];
	n = $StringConstantLengthLimit;
	i = makeAtom[Temporary[0], objData];
	j = makeAtom[Temporary[1], objData];
	k = makeAtom[Temporary[2], objData];
	temp = makeAtom[Temporary[3], objData];
	csub = makeAtom[Temporary[4], objData];
	substrings = Apply[StringJoin, Partition[Characters[string], n, n, {1, 1}, {}], {1}];
	{
		CDeclare[typeToCType[Integer], i],
		CDeclare[typeToCType[Integer], CAssign[j, 0]],
		CDeclare[typeToCType[Integer], CAssign[k, 0]],
		CDeclare["char *", temp],
		CDeclare["char *", 
			CAssign[
				CArray[csub, {}],
				Map[CString, substrings]
			]
		],
		CFor[CAssign[i, n], LessC[i, len], CAssign[AddTo, i, n], 
			CBlock[{
				CAssign[temp, CArray[csub, {COperator[Increment, k]}]],
				CFor["", LessC[j, i], COperator[Increment, j],
					CAssign[CArray[buf, {j}], CDereference[COperator[Increment, temp]]]
				]
			}]
		],
		CAssign[temp, CArray[csub, {k}]],
		CFor["", LessC[j, len], COperator[Increment, j],
			CAssign[CArray[buf, {j}], CDereference[COperator[Increment, temp]]]
		],
		CAssign[CArray[buf, {len}], 0]
	}
]

buildWrapperFunction[objData_] := 
If[objData[Wrapper] === None, 
	""
(* else *),
	Module[{cfstring = ToString[FullForm[objData[Wrapper]]], code, res},
		code = GetStringConstant[res, cfstring, objData];
		code = Flatten[{code, CReturn[res]}];
		CFunction[{"DLLEXPORT", "char *"}, "WolframGPUCompileLibrary_wrapper", {}, CBlock[code]]
	]
]

getArgumentsFunction[objData_] := 
Module[{nargs = Length[objData[ArgumentTypes]], types, ranks, rest},
	ranks = objData[ArgumentTypes][[All,2]];
	ranks = Append[ranks, getRank[objData[ResultRegister]]];
	If[objData[resVariable] === "void", 
		rest = VoidType,
		rest = getType[objData[ResultRegister]]
	];
	types = Append[objData[ArgumentTypes][[All,1]], rest];
	types = Map[ToTypeEnum, types];
	CFunction[{"DLLEXPORT", "mint"}, "WolframGPUCompileLibrary_getArgumentTypes", {
			{"int **", "pt"},
			{"mint **", "pr"}
		},
		CBlock[{
			CDeclare["static int", CAssign[CArray["types", {}], ToString[types, InputForm]]],
			CDeclare["static mint", CAssign[CArray["ranks", {}], ToString[ranks, InputForm]]],
			CAssign[CDereference["pt"], CAddress[CArray["types", {0}]]],
			CAssign[CDereference["pr"], CAddress[CArray["ranks", {0}]]],
			CReturn[nargs]
		}]
	]
]


InitializeFunctionName[objData_, name_] := 
	Module[ {val},
		val = objData[ LifeCycleFunctionNames];
		Which[
			StringQ[ val], "Initialize_" <> val,
			MatchQ[ val, {_String, _String}], First[val],
			True, "Initialize_" <> name]
	]

initializeVar = "initialize";

getInitializeFunction[objData_, name_] :=
Module[{funcData = objData[FunctionData], errlab = objData[ErrorLabel], body},
	(* Localize action of CheckError to this function.
	   funcData has error label needs for constant code *)
	objData[ErrorLabel] = funcData[ErrorLabel];
	body = CIf[initializeVar,
		CBlock[{
			{
				CAssign["wglData", "wglData0"], 
				CAssign["funStructCompile", CPointerMember["wglData", "compileLibraryFunctions"]]
			},
			funcData[ConstantInitializationCode],
			If[funcData[FunctionPointerCount] > 0,
				Table[
					{
						funcData[FunctionPointerCode][i], 
						If[funcData[MathFunctionOriginalName][i] === Null,
							{},
							CComment[ToString[ funcData[MathFunctionOriginalName][i]]]
						],
						CIf[Evaluate[COperator[Equal, {makeAtom[FunctionPointer[i], objData], 0}]],
						CReturn["LIBRARY_FUNCTION_ERROR"]]
					}, 
					{i, 0, funcData[FunctionPointerCount] - 1}]
			(* else *),
				{}
			],
			If[objData[ExpressionPointerCount] > 0,
				Table[{
						objData[ExpressionPointerCode[i]], 
						CIf[Evaluate[COperator[Equal, {makeAtom[ExpressionPointer[i], objData], 0}]],
							CReturn["LIBRARY_FUNCTION_ERROR"]]
					}, 
					{i, 0, objData[ExpressionPointerCount] - 1}
				]
			(* else *),
				{}
			],
			CAssign[initializeVar, 0]
		}]
	];
	If[UnsameQ[objData[ErrorLabel], False],
		body = {
			CDeclare[ErrorVariableCType, CAssign[makeAtom[ErrorVariable, objData], 0]],
			body, 
			CLabel[objData[ErrorLabel]],
			CReturn[makeAtom[ErrorVariable, objData]]
		}
	(* else *),
		body = {body, CReturn[0]}
	];
	objData[ErrorLabel] = errlab;
	body = CBlock[body];
	CFunction[
		If[objData[WolframRTL], {"DLLEXPORT", "int"}, {"static", "int"}],
		InitializeFunctionName[objData, name], 
		{{"WolframGPULibraryData", "wglData0"}},
		body
	]
];


UninitializeFunctionName[objData_, name_] := 
	Module[ {val},
		val = objData[ LifeCycleFunctionNames];
		Which[
			StringQ[ val], "Uninitialize_" <> val,
			MatchQ[ val, {_String, _String}], Last[val],
			True, "Uninitialize_" <> name]
	]
	
		
getUninitializeFunction[objData_, name_] :=
Module[{funcData = objData[FunctionData], body, decl},
	body = CBlock[
			CIf[COperator[Not, {initializeVar}], 
				CBlock[
					funcData[ConstantDeallocationCode],
					CAssign[initializeVar, 1]
				]
			]
		];
	decl = {If[objData[WolframRTL], {"DLLEXPORT"}, "static"], "void"};
	CFunction[decl, UninitializeFunctionName[objData, name], {{"WolframLibraryData", "libData"}}, body]
]
		

getDLLUninitializeFunction[objData_, name_] :=
	CFunction[ {"DLLEXPORT", "void"}, "WolframLibrary_uninitialize", 
		{{"WolframLibraryData", "libData"}},
		CBlock[
			If[objData[ArrayArgs], 
				{CCall[UninitializeFunctionName[objData, name], {"libData"}]},
				{}
			]
		]	
	];

(* nAry math operators that are expanded by symbolic C *)
CnAryMathOperatorQ[Plus] = True;
CnAryMathOperatorQ[Times] = True;
CnAryMathOperatorQ[Subtract] = True;
CnAryMathOperatorQ[Divide] = True;
CnAryMathOperatorQ[_] := False;

VoidPointer[arg_] := CCast[ CPointerType["void"], arg];

	
Apply[
	Function[Math1ArgCode[#1] = #2],  
	Cases[DownValues[Arg1Name], 
 		Rule[
 			RuleDelayed[Literal[HoldPattern][Arg1Name[n_Integer]], s_], 
 			{s, n}
 		]
	],
	{1}
];

(* This should really be in the CStandardQ things from 
   SymbolicC, but we did not want to change that package
   for now, so I added these definitions.  The code to change
   in SymbolicC is 
   
CStandardList={{ArcTan,"atan"},{Sin,"sin"},{Cos,"cos"},{Tan,"tan"},{Exp,
        "exp"},{Log,"log"}};
        
*)
CStandardMath[ArcCos] = "acos";
CStandardMath[ArcSin] = "asin";
CStandardMath[ArcTan] = "atan";
CStandardMath[Cos] = "cos";
CStandardMath[Sin] = "sin";
CStandardMath[Tan] = "tan";
CStandardMath[Cosh] = "cosh";
CStandardMath[Sinh] = "sinh";
CStandardMath[Tanh] = "tanh";
CStandardMath[Exp] = "exp";
CStandardMath[Log] = "log";
CStandardMath[Log10] = "log10";
CStandardMath[Sqrt] = "sqrt";

CStandardMathQ[fun_] := StringQ[CStandardMath[fun]]

isUnaryMathFunction[fun_] := And[MatchQ[Math1ArgCode[fun], _Integer], SameQ[fun, Arg1Name[Math1ArgCode[fun]]]];

callUnaryMathFunction[objData_, Minus, out:Register[Integer, _], arg:Register[Integer, _], aflag_] := 
	CAssign[makeAtom[out, objData], COperator[Minus, {makeAtom[arg, objData]}]]

callUnaryMathFunction[objData_, Minus, out:Register[Real, _], arg:Register[Real, _], aflag_] := 
	CAssign[makeAtom[out, objData], COperator[Minus, {makeAtom[arg, objData]}]]
	
callUnaryMathFunction[objData_, Abs, out:Register[Integer, _], arg:Register[Integer, _], aflag_] := 
Module[{arga = makeAtom[arg, objData]},
	CAssign[makeAtom[out, objData], CConditional[COperator[Less, {arga, 0}], COperator[Minus, {arga}], arga]]
]

callUnaryMathFunction[objData_, Abs, out:Register[Real, _], arg:Register[Real, _], aflag_] := 
Module[{arga = makeAtom[arg, objData]},
	CAssign[makeAtom[out, objData], CConditional[COperator[Less, {arga, 0}], COperator[Minus, {arga}], arga]]
]

callUnaryMathFunction[objData_, "Reciprocal", out:Register[Real, _], arg:Register[Real, _], aflag_ /; Not[CheckAnyflowQ[aflag]]] := 
	CAssign[ makeAtom[out, objData], COperator[ Divide, {1, makeAtom[arg, objData]}]]
	
callUnaryMathFunction[objData_, Square, out:Register[type:(Integer | Real), _], arg:Register[type_, _], aflag_ /; Not[CheckAnyflowQ[aflag]]] := 
	CAssign[ makeAtom[out, objData], COperator[Times, {makeAtom[arg, objData], makeAtom[arg, objData]}]]
	
callUnaryMathFunction[objData_, Square, out:Register[Complex, _], arg:Register[Complex, _], aflag_ /; Not[CheckAnyflowQ[aflag]]] := 
Module[{in = makeAtom[arg, objData], res = makeAtom[out, objData]},
	{
	CAssign[ReC[res], COperator[Subtract, {COperator[Times, {ReC[in], ReC[in]}], COperator[Times, {ImC[in], ImC[in]}]}]],
	CAssign[ImC[res], COperator[Times, {2., ReC[in], ImC[in]}]]
	}
]

callUnaryMathFunction[objData_, fun_?CStandardMathQ, out:Register[Real, _], arg:Register[Real, _], aflag_ /; Not[CheckAnyflowQ[aflag]]] := 
	CAssign[makeAtom[out, objData], CCall[CStandardMath[fun], {makeAtom[arg, objData]}]]
			
callUnaryMathFunction[objData_, fun_, out:Register[Real, _], arg:Register[Real, _], aflag_] := 
	BuildMathFunctionCall[objData, fun, out, {arg}, aflag]
		
(* Inline code for simple complex ops *)
callUnaryMathFunction[objData_, Re, out:Register[Real, _], arg:Register[Complex, _], aflag_] :=
	CAssign[makeAtom[out, objData], ReC[makeAtom[arg, objData]]]

callUnaryMathFunction[objData_, Im, out:Register[Real, _], arg:Register[Complex, _], aflag_] :=
	CAssign[makeAtom[out, objData], ImC[makeAtom[arg, objData]]]

callUnaryMathFunction[objData_, Conjugate, out:Register[Complex, _], arg:Register[Complex, _], aflag_] := 
	{CAssign[ReC[makeAtom[out, objData]], ReC[makeAtom[arg, objData]]],
	 CAssign[ImC[makeAtom[out, objData]], COperator[Minus, {ImC[makeAtom[arg, objData]]}]]};
	
callUnaryMathFunction[objData_, Minus, out:Register[Complex, _], arg:Register[Complex, _], aflag_] := 
	{CAssign[ReC[makeAtom[out, objData]], COperator[Minus, {ReC[makeAtom[arg, objData]]}]],
	 CAssign[ImC[makeAtom[out, objData]], COperator[Minus, {ImC[makeAtom[arg, objData]]}]]};
	
callUnaryMathFunction[objData_, fun_, out_, arg_, aflag_] := 
	BuildMathFunctionCall[objData, fun, out, {arg}, aflag]
	
Apply[
	Function[Math2ArgCode[#1] = #2],  
	Cases[DownValues[Arg2Name], 
 		Rule[
 			RuleDelayed[Literal[HoldPattern][Arg2Name[n_Integer]], (s_Symbol | s_String)], 
 			{s, n}
 		]
	],
	{1}
];

isBinaryMathFunction[fun_] := And[MatchQ[Math2ArgCode[fun], _Integer], SameQ[fun, Arg2Name[Math2ArgCode[fun]]]];

(* Use pow for Power with real args *)
callBinaryMathFunction[objData_, Power, out:Register[Real, _], {arg1:Register[Real, _], arg2:Register[Real, _]}, aflag_ /; Not[CheckAnyflowQ[aflag]]] := 
	CAssign[makeAtom[out, objData], CCall["pow", {makeAtom[arg1, objData], makeAtom[arg2, objData]}]]
		
(* Set up code with binary decomposition for power with integer exponent 
   Integer case is different from real case because of different exceptions *) 
callBinaryMathFunction[objData_, Power, out:Register[type:Integer, _], {arg1:Register[type_, _], arg2:Register[Integer, _]}, aflag_ /; Not[CheckIntegerOverflowQ[aflag]]] := 
Module[{res = makeAtom[out, objData], x = makeAtom[arg1, objData], in = makeAtom[arg2, objData], n = makeAtom[Temporary[0], objData], z = makeAtom[Temporary[1], objData]},
	CIf[COperator[Equal, {x, 1}],
		CBlock[CAssign[res, 1]],
		CIf[COperator[Equal, {x, -1}],
			CBlock[CAssign[res, CConditional[BitAndC[in, 1], -1, 1]]],
			CIf[COperator[Less, {in, 0}],
				CBlock[{ForceError[objData, 1]}],
				CIf[COperator[Equal, {in, 0}],
					CBlock[{
						CIf[COperator[Equal, {x, 0}], CBlock[{ForceError[objData, 1]}]], 
						CAssign[res, 1]
					}],
					CBlock[
						CDeclare[typeToCType[Integer], CAssign[n, makeAtom[arg2, objData]]],
						CDeclare[typeToCType[type], CAssign[z, makeAtom[arg1, objData]]],
						CAssign[res, 1],
						CWhile[n,
							CBlock[{
								CIf[BitAndC[n, 1],CAssign[res, COperator[Times, {z, res}]]],
								CAssign[z, COperator[Times, {z, z}]],
								CAssign[n, BitShiftRightC[n, 1]]
							}]
						]
					]
				]
			]
		]
	]
]
	
callBinaryMathFunction[objData_, Power, out:Register[type:Real, _], {arg1:Register[type_, _], arg2:Register[Integer, _]}, aflag_ /; Not[CheckAnyflowQ[aflag]]] := 
Module[{res = makeAtom[out, objData], n = makeAtom[Temporary[0], objData], z = makeAtom[Temporary[1], objData], b = makeAtom[Temporary[2], objData]},
	CIf[COperator[Equal, {makeAtom[arg2, objData], 0}],
		CBlock[{CIf[COperator[Equal, {makeAtom[arg1, objData], 0}], 
			CBlock[{ForceError[objData, 1]}], 
			CAssign[res, 1]]}],
		CBlock[
			CDeclare[typeToCType[Integer], CAssign[n, makeAtom[arg2, objData]]],
			CDeclare[typeToCType[type], CAssign[z, makeAtom[arg1, objData]]],
			CDeclare[typeToCType[Boolean], CAssign[b, 0]],
			CIf[LessC[ n, 0],
				CBlock[{CAssign[b, 1], CAssign[n, MinusC[n]]}]
			],
			CAssign[res, 1],
			CWhile[n,
				CBlock[{
					CIf[BitAndC[n, 1],CAssign[res, COperator[Times, {z, res}]]],
					CAssign[z, COperator[Times, {z, z}]],
					CAssign[n, BitShiftRightC[n, 1]]
				}]
			],
			CIf[b, CAssign[res, COperator[Divide, {1, res}]]]
		]
	]
]
	
(* Insert code for basic complex operations *)
callBinaryMathFunction[objData_, op:(Plus | Subtract), out:Register[Complex, _], {arg1:Register[Complex, _], arg2:Register[Complex, _]}, aflag_ /; Not[CheckAnyflowQ[aflag]]] := 
	{CAssign[ReC[makeAtom[out, objData]], COperator[op, {ReC[makeAtom[arg1, objData]], ReC[makeAtom[arg2, objData]]}]],
	 CAssign[ImC[makeAtom[out, objData]], COperator[op, {ImC[makeAtom[arg1, objData]], ImC[makeAtom[arg2, objData]]}]]}



callBinaryMathFunction[objData_, Times, out:Register[Complex, _], 
	{arg1:Register[Complex, _], arg2:Register[Complex, _]}, aflag_ /; Not[CheckAnyflowQ[aflag]]] := 
Module[{t, creal = typeToCType[Real], r = "mcreal", i = "mcimag", k = 0, 
	     outa = makeAtom[out, objData], arg1a = makeAtom[arg1, objData], arg2a = makeAtom[arg2, objData]},
	CBlock[
		Outer[
			CDeclare[creal, CAssign[ t[#1,#2] = makeAtom[Temporary[k++], objData], CCall[#2, {#1}]]]&,
			{arg1a, arg2a}, {r, i}],
		CAssign[CCall[r,{outa}], COperator[Subtract, {COperator[Times, {t[arg1a,r], t[arg2a,r]}], COperator[Times, {t[arg1a,i], t[arg2a,i]}]}]],
		CAssign[CCall[i,{outa}], COperator[Plus, {COperator[Times, {t[arg1a,r], t[arg2a,i]}], COperator[Times, {t[arg1a,i], t[arg2a,r]}]}]]
	]
]

callBinaryMathFunction[objData_, fun_, out_, args_List, aflag_] := 
	BuildMathFunctionCall[objData, fun, out, args, aflag]
	
(*
	n-ary Memory functions
*)
Apply[
	Function[MemoryNAryCode[#1] = #2],  
	Cases[DownValues[ArgNName], 
 		Rule[
 			RuleDelayed[Literal[HoldPattern][ArgNName[n_Integer]], s_Symbol], 
 			{s, n}
 		]
	],
	{1}
];

MemoryNAryQ[fun_] := And[MatchQ[MemoryNAryCode[fun], _Integer], SameQ[fun, ArgNName[MemoryNAryCode[fun]]]];

callMemoryNAryFunction[objData_, fun_, out_, args_] := 
Module[{n = Length[args], tar = makeAtom[Temporary[0], objData]},
	CBlock[
		CDeclare[typeToCType[Tensor], CArray[tar, n]],
		Table[CAssign[CArray[tar, i - 1], args[[i]]], {i, n}],
		CCall[objData[structCompile["Memory_nArg"]], {MemoryNAryCode[fun], n, tar, out}]
	]
]


(*
 Label utilities
*)

makeLabel[ i_] := "lab" <> ToString[i]

createLabel[ objData_, number_] :=
		objData[ labelData, number] = makeLabel[number]

fixLabels[ objData_, lines_List] :=
	Table[ fixLabel[ i, objData, lines[[i]]], {i, Length[lines]}]
	
fixLabel[ num_, objData_, line_] :=
	Module[ {lab},
		lab = objData[labelData, i];
		If[ StringQ[ lab], {CLabel[ lab], line}, line]
	]


(*
 Process the instructions
*)

buildLines[ lines_List, objData_] :=
	Table[ buildLine[i, objData, lines[[i]]], {i, Length[ lines]}]


buildLine[ num_, objData_, Instruction["Version", _]] := {}

(*
	Basic Memory properties 
*)
buildLine[num_, objData_, Instruction[Length, out_, arg:Register[Tensor[_, rank_],_]]] :=
	CAssign[makeAtom[out, objData], CArray[getDataDimensions[objData, arg],  0]]

buildLine[num_, objData_, Instruction[Dimensions, out_, {arg:Register[Tensor[type_, rank_],_], nreg_}]] :=
Module[{len = makeAtom[Temporary[0], objData], dims = getDataDimensions[objData, arg]},
CBlock[
	CDeclare[typeToCType[Integer], CAssign[len, makeAtom[nreg, objData]]],
	CIf[GreaterC[ len, rank], CAssign[len, rank]],
	CheckError[objData, 
		CCall[objData[structCompile["Memory_allocate"]], 
			{makeAtom[out, objData], ToTypeEnum[Integer], 1, CAddress[len]}]
	],
	setDataPointer[objData, out],
	CWhile[COperator[ Decrement, len], CAssign[CArray[getDataPointer[objData, out], len], CArray[dims, len]]]
]]

(* Build a Memory from a constant list *)
buildLine[ num_, objData_, Instruction[Set, oreg:Register[Tensor[type_, rank_], no_], ConstantTensor[ct_List]]] :=
Module[{dims = makeAtom[Temporary[0], objData], out = makeAtom[oreg, objData], fun},
	CBlock[
		CDeclare[typeToCType[Integer], CArray[dims, rank]],
		MapIndexed[CAssign[CArray[dims, {#2[[1]] - 1}], makeAtom[#1, objData]]&, Dimensions[ct]],
		CheckError[objData, 
			CCall[objData[structCompile["Memory_allocate"]], 
				{out, ToTypeEnum[type], rank, dims}]
		],
		setDataPointer[objData, oreg],
		If[type === Complex, 
			fun = Function[{
				CAssign[ReC[CArray[getDataPointer[objData, oreg], {#2[[1]] - 1}]], makeAtom[Re[#1], objData]],
				CAssign[ImC[CArray[getDataPointer[objData, oreg], {#2[[1]] - 1}]], makeAtom[Im[#1], objData]]
			}],
			fun = Function[CAssign[CArray[getDataPointer[objData, oreg], {#2[[1]] - 1}], makeAtom[#1, objData]]]
		];
		MapIndexed[fun, Flatten[ct]]
	]
]

buildLine[ num_, objData_, Instruction[Set, out_, inp_]] :=
	Which[
		isTensor[out],
			{
				CheckError[objData, 
					CCall[objData[structCompile["Memory_copyUnique"]], 
						{makeAtom[out, objData], makeTDAtom[inp, objData]}]
				],
	 			setDataPointer[objData, out]
	 		},
	 	MatchQ[ inp, x_Complex /; NumberQ[x]] && MatchQ[ out, Register[Complex, _]],
	 		{CAssign[ CCall[ "mcreal", {makeAtom[out, objData]}], makeAtom[Re[inp], objData]],
	 		 CAssign[ CCall[ "mcimag", {makeAtom[out, objData]}], makeAtom[Im[inp], objData]]},
	 	getType[out] === getType[inp],
			CAssign[ makeAtom[out, objData], makeAtom[inp, objData]],
	 	True,
			CAssign[ makeAtom[out, objData], CCast[typeToCType[getType[out]], makeAtom[inp, objData]]]
	]
	
(* 
	Make a rank 0 tensor from a scalar
*)
buildLine[num_, objData_, Instruction["To0Rank", out:Register[Tensor[type_, 0], _], scalar_]] :=
{
	CheckError[objData, 
		CCall[objData[structCompile["Memory_allocate"]], 
			{makeAtom[out, objData], ToTypeEnum[type], 0, 0}]
	],
	setDataPointer[objData, out],
	CAssign[CDereference[getDataPointer[objData, out]], makeAtom[scalar, objData]]
}

(* Build a Memory from scalar data in registers *)
buildLine[ num_, objData_, Instruction[List, regout:Register[Tensor[type_, 1], _], regElems_List]] :=
Module[{dims = makeAtom[Temporary[0], objData], out = makeAtom[regout, objData]},
CBlock[
	CDeclare[typeToCType[Integer], CAssign[dims, makeAtom[Length[regElems], objData]]],
	CheckError[objData, 
		CCall[objData[structCompile["Memory_allocate"]], 
			{out, ToTypeEnum[type], 1, CAddress[dims]}]
	],
	setDataPointer[objData, regout],
	Table[
		CAssign[CArray[getDataPointer[objData, regout], k - 1], makeAtom[regElems[[k]], objData]],
		{k, Length[regElems]}
	]
]]
		
(* Build a Memory from tensor data in registers *)		
buildLine[ num_, objData_, Instruction[List, regout:Register[Tensor[type_, rank_], _], regElems_List]] :=
Module[{i = makeAtom[Temporary[0], objData], dims = makeAtom[Temporary[1], objData], olddims = getDataDimensions[objData, regElems[[1]]],  
	     out = makeAtom[regout, objData], setfun = objData[structCompile["Memory_setMemory"]]},
CBlock[
	CDeclare[typeToCType[Integer], i],
	CDeclare[typeToCType[Integer], CArray[dims, rank]],
	CAssign[CArray[dims, 0], makeAtom[Length[regElems], objData]],
	CFor[CAssign[i, 1], LessC[ i, rank], COperator[ Increment, i], 
		CAssign[CArray[dims, i], CArray[olddims, COperator[ Subtract, {i, 1}]]]
	],
	CheckError[objData, 
		CCall[objData[structCompile["Memory_allocate"]], 
			{out, ToTypeEnum[type], rank, dims}]
	],
	setDataPointer[objData, regout],
	Table[{
			CAssign[i, makeAtom[k, objData]],
			CheckError[objData,
				CCall[setfun, 
					{CDereference[out], makeTDAtom[regElems[[k]], objData], CAddress[i], 1}]
			]
		},
		{k, Length[regElems]}
	]
]]

(*
	Coercion
*)

buildLine[num_, objData_, Instruction["SetComplex", out_, {re_, im_}]] :=
{
	CAssign[ReC[makeAtom[out, objData]], makeAtom[re, objData]],
	CAssign[ImC[makeAtom[out, objData]], makeAtom[im, objData]]
}

LoopOverElements[objData_, n_, arrays_, rawbody_] :=
Module[{i = makeAtom[Temporary[0], objData], body, x, subs},
	subs = Array[x, Length[arrays]];
	body = rawbody /. Thread[arrays->subs];
	body = body /. Thread[subs->Map[CArray[#, i]&, arrays]];
	CBlock[{
		CDeclare[typeToCType[Integer], i],
		CFor[
			CAssign[i, 0],
			COperator[Less, {i, n}],
			COperator[ Increment, i],
			body
		]
	}]
]

(*
  Generate code for a return instruction.
  
  If we are the last instruction, then we use the returnCode 
  that is already computed.
 
  Otherwise
  
  This might be at the very end, in which case we would 
  just insert a GOTO, or we might actually insert the 
  return itself.  If this is the last instruction we 
  won't bother with the GOTO -- it would not be necessary.
  
  The return will be at the end if we have special code 
  to insert.  This might be because of cleanup,  ie we 
  have tensors to free.  It might also be because we 
  are passing arguments in a char* array.
  
  The returnAtEnd controls whether we jump to the end 
  to return.  This could be an option to the generator.
*)


buildLine[ num_, objData_, Instruction[Return]] :=
	If[ num === objData[ lineLength],
		{},
		If[ objData[ returnAtEnd],
			If[objData[ReturnLabel] === False, objData[ReturnLabel] = "return_label"];
			CGoto[objData[ReturnLabel]]
		(* else *),
			getReturnCode[ objData]
		]
	]

(*
 Various branching and loop instructions
*)


buildLine[ num_, objData_, Instruction["Jump", Line[number_]]] :=
	CGoto[ createLabel[ objData, number]]


buildLine[ num_, objData_, Instruction["Branch", reg_, Line[number_]]] :=
	CIf @@ { COperator[ Not, makeAtom[ reg, objData]], CGoto[createLabel[ objData, number]]}


buildLine[ num_, objData_, Instruction["LoopIncr", {reg1_, reg2_}, goto_]] :=
	CIf @@ { COperator[ LessEqual, {COperator[ PreIncrement, makeAtom[ reg1, objData]], makeAtom[ reg2, objData]}], 
		CGoto[createLabel[ objData, goto]]}



(*
 Array instructions
*)

buildLine[ num_, objData_, Instruction[Table, regout_, regDims_List]] :=
Module[ {rank, type, dims},
	rank = Length[ regDims];
	type = getTensorEnumType[ regout];
	dims = getDimensionArray[objData, rank];
	{
		Table[CAssign[CArray[dims,{i-1}], makeAtom[regDims[[i]], objData]], {i,rank}],
		CheckError[objData,
			CCall[objData[structCompile["Memory_allocate"]], 
		 		{makeAtom[regout, objData], type, rank, dims}]
		 ],
		 setDataPointer[objData, regout]
	}
]	
	
(*
getElementGetFunction[ type_, rank_] :=
	Which[
		rank > 1,   "Memory_GetElementMemory",
		type == IntegerEnumType, "Memory_GetElementInt",
		type == RealEnumType, "Memory_GetElementReal",
		True, "Memory_GetElementComplex"]


buildLine[ num_, objData_, Instruction[Part, regRes_, {regTen_, {0, Register[Integer, regPart_]}}]] :=
	Module[ {rank, type, pos, function},
		rank = getRank[ regTen];
		type = getTensorEnumType[ regTen];
		pos = makeAtom[ Register[Integer, regPart]];
		function = getElementGetFunction[ type, rank];
		If[rank == 1, 
			CAssign[makeAtom[regRes], CCall[function, {makeAtom[regTen], pos}]],
			CCall[function, {makeAtom[regRes], makeAtom[regTen], pos}]
		]
	]	
*)

(*
	This is used in Part, Take, Drop
	Makes two arrays, one 
	int * for the types
	void ** for the data
*)
PartTypeRegister[reg:Register[Integer, _]] := {0, reg}
PartTypeRegister[{reg:Register[Tensor[_, 1], _], List}] := {1, reg}
PartTypeRegister[All] := {2, 0};
PartTypeRegister[{reg:Register[Tensor[_, 1], _], Span}] := {3, reg}

callWithCollectedSpecifications[objData_, func_, res_, arg_, specs_] :=
Module[{tar = makeAtom[Temporary[0], objData], sar = makeAtom[Temporary[1], objData], n = Length[specs], stype, sreg},
	CBlock[
		CDeclare["int", CArray[tar, n]],
		CDeclare[ CPointerType["void"], CArray[sar, n]],
		Table[
			{stype, sreg} = PartTypeRegister[specs[[i]]];
			{
			CAssign[CArray[tar, i - 1], stype], 
			CAssign[CArray[sar, i - 1], If[NumberQ[sreg], sreg, VoidPointer[makeAtomAddress[sreg, objData]]]]
			},
			{i, n}
		],
		CheckError[objData, 
			CCall[func, {makeAtom[res, objData], makeTDAtom[arg, objData], n, tar, sar}]
		]
	]
]
	
	
(*
	Part
*)

getElementFunction[objData_, type_, rank_, depth_] := 
	If[rank == 0,
		objData[structCompile["Memory_get" <> ToString[type]]]
	(* else *),
		objData[structCompile["Memory_getMemoryInitialized"]]
	]

setElementFunction[objData_, type_, rank_, depth_] := 
	If[rank == 0,
		objData[structCompile["Memory_set" <> ToString[type]]]
	(* else *),
		objData[structCompile["Memory_setMemory"]]
	]

GetOrSetPart[objData_, setQ_, res_, {arg_, pspecs_}] :=
Module[{type, resrank, argrank, depth, function, pos, into, from, intor, fromr, prt},
Flatten[{
	{type, resrank} = convertFullType[res];
	argrank = getRank[arg];
	depth = Length[pspecs];
	If[(argrank - resrank) == depth,
		(* All are of  type ONEPART:
		   In these functions, for Part the first arg is the result,
		   but for SetPart, the first arg is the thing you are setting
		   ininto, so we effect a switch of these args *)
		{intor, fromr} = If[setQ, {arg, res}, {res, arg}];
		into = makeAtom[intor, objData];
		from = makeTDAtom[fromr, objData];
		function = If[TrueQ[setQ], 
			setElementFunction[objData, type, resrank, depth],
			getElementFunction[objData, type, resrank, depth]
		];
		If[depth == 1,
			If[resrank == 0,
				CBlock[
					pos = makeAtom[Temporary[0], objData];
					prt = makeAtom[pspecs[[1]], objData];
					CDeclare[typeToCType[Integer], CAssign[pos, CArray[getDataDimensions[objData, arg], 0]]],
					CIf[GreaterC[ prt, 0],
						CBlock[{
							CIf[GreaterC[ prt, pos], CReturn["LIBRARY_DIMENSION_ERROR"]],
							CAssign[pos, COperator[Subtract, {prt, 1}]]
						}],(* else *) CBlock[{
							CIf[OrC[EqualC[prt, 0], LessC[ prt, COperator[ Minus, pos]]], CReturn["LIBRARY_DIMENSION_ERROR"]],
							CAssign[pos, COperator[Plus, {pos, prt}]]
						}]
					],
					If[setQ,
						CAssign[CArray[getDataPointer[objData, intor], pos], from],
						CAssign[into, CArray[getDataPointer[objData, fromr], pos]]
					]
				],
			(* else resrank > 0 *)
				If[setQ, into = CDereference[into]];
				CheckError[objData,
					CCall[function, {into, from, CAddress[makeAtom[pspecs[[1]], objData]], 1}]
				]
			],
		(* else Create code to set up position array *)
			pos = makeAtom[Temporary[0], objData];
			CBlock[
				CDeclare[typeToCType[Integer], CArray[pos, depth]],
				Table[CAssign[CArray[pos, i - 1], makeAtom[pspecs[[i]], objData]], {i, depth}],
				If[resrank == 0, 
					args = If[TrueQ[setQ],
						{CDereference[into], pos, from},
						{from, pos, CAddress[into]}
					];
					CheckError[objData, CCall[function, args]]
				(* else resrank > 0 *),
					CheckError[objData, 
						CCall[function,
							If[setQ,
								{CDereference[into], from, pos, makeAtom[depth, objData]},
								{into, from, pos, makeAtom[depth, objData]}
							]
						]
					]
				]
			]
		],
	(* else *)
		function = objData[structCompile[If[setQ, "Memory_setPart", "Memory_getPart"]]];
		callWithCollectedSpecifications[objData, function, res, arg, pspecs]
	]
}]
]

buildLine[num_,objData_, Instruction["GetElement", res_, {arg:Register[Tensor[type_, argrank_], _], inds__}]] := 
Module[{rank, depth, indices = {inds}, dims, pos, indexC, index0C},
	depth = Length[indices];
	rank = argrank - depth;
	indexC[i_] := makeAtom[indices[[i]], objData];
	index0C[i_] := COperator[Subtract, {indexC[i], 1}];
	If[Length[indices] == 1, 
		If[rank == 0,
			CAssign[makeAtom[res, objData], CArray[getDataPointer[objData, arg], index0C[1]]],
		(* else *)
			{
				CCall[getElementFunction[objData, type, rank, 1], {makeAtomAddress[res, objData], makeTDAtom[arg, objData], CAddress[indexC[1]], 1}],
				setDataPointer[objData, res]
			}
		],
	(* else *)
		dims = getDataDimensions[objData, arg];
		pos = makeAtom[Temporary[0], objData];
		CBlock[
			If[rank == 0,
			{
				CDeclare[typeToCType[Integer], CAssign[pos, index0C[1]]],
				Table[CAssign[pos, PlusC[TimesC[pos, CArray[dims, i]], index0C[i + 1]]],
					{i, 1, depth - 1}],
				CAssign[makeAtom[res, objData], CArray[getDataPointer[objData, arg], pos]]
			}, (* else *) {
				CDeclare[typeToCType[Integer], CArray[pos, depth]],
				Table[CAssign[CArray[pos, i - 1], indexC[i]],
					{i, depth}],
				CCall[objData[structCompile["Memory_getMemoryInitialized"]], {makeAtomAddress[res, objData], makeTDAtom[arg, objData], pos, depth}],
				setDataPointer[objData, res]
			}]
		]
	]
]

buildLine[ num_, objData_, Instruction[Part, res_, {arg_, pts__}]] :=
{
	GetOrSetPart[objData, False, res, {arg, {pts}}],
	If[getRank[res] > 0, setDataPointer[objData, res], {}]
}

buildLine[ num_, objData_, Instruction["SetPart", res_, {arg_, pts__}]] :=
	(* Notice switch of order -- the "result" for the compiler is
	   what you are setting to to follow return 
	   of rhs in Part[...] = rhs *)
	GetOrSetPart[objData, True, res, {arg, {pts}}]

buildLine[ num_, objData_, Instruction["SetElement", regTen_, regPos_, regValue_]] :=
	Module[ {rank, type, pos},
		rank = getRank[ regTen];
		type = getTensorEnumType[ regTen];
		pos = makeAtom[ regPos, objData];
		If[isTensor[regValue],
			CheckError[objData, 
				CCall[objData[structCompile["Memory_insertMemory"]], 
					{ makeTDAtom[regTen, objData], makeTDAtom[ regValue, objData], CAddress[pos]}]
			]
		(* else *),
			CAssign[CArray[getDataPointer[objData, regTen], COperator[ Increment, pos]], makeAtom[regValue, objData]]
		]
	]	

(*
 Operators and functions
*)

(* Comparisons *)


ComparisonEnum[SameQ] = 1
ComparisonEnum[UnsameQ] = 2
ComparisonEnum[Less] = 3
ComparisonEnum[LessEqual] = 4
ComparisonEnum[Equal] = 5
ComparisonEnum[GreaterEqual] = 6
ComparisonEnum[Greater] = 7
ComparisonEnum[Unequal] = 8

ComparisonQ[x_] := MatchQ[ComparisonEnum[x], _Integer]

(* Needed for complex stuff *)
InequalityQ[Less] = True;
InequalityQ[LessEqual] = True;
InequalityQ[Greater] = True;
InequalityQ[GreaterEqual] = True;
InequalityQ[_] = False;

ComparisonOperator[SameQ, arg_] = COperator[Equal, arg]
ComparisonOperator[UnsameQ, arg_] = COperator[Unequal, arg]
ComparisonOperator[comp_, arg_] = COperator[comp, arg]

buildLine[num_, objData_, Instruction[fun_?ComparisonQ, out_, args:{Register[Boolean, _]..}]] :=
Module[{pargs},
	pargs = Partition[Map[makeAtom[#, objData]&, args], 2, 1];
	CAssign[makeAtom[out, objData], 
		Apply[AndC,
			Table[ComparisonOperator[fun, arg], {arg, pargs}]
		]
	]
]
	
buildLine[num_, objData_, Instruction[fun_?ComparisonQ, out_, args:{Register[Integer, _]..}]] :=
Module[{pargs},
	pargs = Partition[Map[makeAtom[#, objData]&, args], 2, 1];
	CAssign[makeAtom[out, objData], 
		Apply[AndC,
			Table[ComparisonOperator[fun, arg], {arg, pargs}]
		]
	]
]
	
buildLine[num_, objData_, Instruction[fun_?ComparisonQ, out_, {tol_, argseq:(Register[Real, _]..)}]] :=
Module[{pargs, n, args = {argseq}, ctol = objData[Constants][tol]},
	If[SameQ[ctol, 0.],
		pargs = Partition[Map[makeAtom[#, objData]&, args], 2, 1];
		CAssign[makeAtom[out, objData],
			Apply[AndC,
				Table[ComparisonOperator[fun, arg], {arg, pargs}]
			]
		],
	(* else: call function *)
		pargs = makeAtom[Temporary[0], objData];
		n = Length[args];
		CBlock[
			CDeclare[typeToCType[Real], CArray[pargs, n]],
			Table[CAssign[CArray[pargs, i - 1], makeAtom[args[[i]], objData]], {i, n}],
			CAssign[makeAtom[out, objData], CCall[objData[structCompile["Compare_R"]], Flatten[{ComparisonEnum[fun], makeAtom[tol, objData], n, pargs}]]]
		]
	]
]

buildLine[num_, objData_, Instruction[fun_?ComparisonQ, out_, {tol_, argseq:(Register[Complex, _]..)}]] :=
Module[{pargs, n, args = {argseq}, ctol = objData[Constants][tol]},
	If[SameQ[ctol, 0.] && !InequalityQ[fun],
		pargs = Partition[Map[makeAtom[#, objData]&, args], 2, 1];
		CAssign[makeAtom[out, objData],
			Apply[AndC,
				Flatten[
					Table[Map[ComparisonOperator[fun, #[arg]]&, {ReC, ImC}], {arg, pargs}]
				]
			]
		],
	(* else call function *)
		pargs = makeAtom[Temporary[0], objData];
		n = Length[args];
		CBlock[
			CDeclare[typeToCType[Complex], CArray[pargs, n]],
			Table[CAssign[CArray[pargs, i - 1], makeAtom[args[[i]], objData]], {i, n}],
			CheckError[
				objData, 
				CCall[objData[structCompile["Compare_C"]], Flatten[{ComparisonEnum[fun], makeAtom[tol, objData], n, pargs, makeAtomAddress[out, objData]}]]
			]
		]
	]
]

buildLine[num_, objData_, Instruction[fun_?ComparisonQ, out_, {tol_, argseq:(Register[Tensor[_, _], _]..)}]] :=
Module[{pargs, n, cenum = ComparisonEnum[fun], args = {argseq}},
	pargs = makeAtom[Temporary[0], objData];
	n = Length[args];
	CBlock[
		CDeclare[typeToCType[Tensor], CArray[pargs, n]],
		Table[CAssign[CArray[pargs, i - 1], makeTDAtom[args[[i]], objData]], {i, n}],
		CAssign[makeAtom[out, objData], CCall[objData[structCompile["Compare_T"]], Flatten[{cenum, makeAtom[tol, objData], n, pargs}]]]
	]
]

(* 
	Logical operators:  
*)

buildLine[num_, objData_, Instruction[And, out_, args_List]] :=
	CAssign[makeAtom[out, objData], Apply[AndC, Map[makeAtom[#, objData]&, args]]]

buildLine[num_, objData_, Instruction[Or, out_, args_List]] :=
	CAssign[makeAtom[out, objData], Apply[OrC, Map[makeAtom[#, objData]&, args]]]

buildLine[num_, objData_, Instruction[Not, out_, arg_]] :=
	CAssign[makeAtom[out, objData], COperator[Not, makeAtom[arg, objData]]]

(*
	Inline Xor
*)
buildLine[num_, objData_, Instruction[Xor, out_, args_List]] := 
Module[{count = makeAtom[Temporary[0], objData]},
	CBlock[{
		CDeclare[typeToCType[Integer], CAssign[count, 0]],
		Table[
			CIf[Evaluate[makeAtom[argi, objData]], COperator[ Increment, count]],
			{argi, args}
		],
		CAssign[makeAtom[out, objData], BitAndC[count, 1]]
	}]
]
	
(* Operators that can be handled for C basic types with n arguments *)
buildLine[num_, objData_, Instruction[fun_?CnAryMathOperatorQ, out_, args_List]] := 
Module[{typeout = convertFullType[out], typeargs = Map[convertFullType,args], aflag = objData[ArithmeticFlags]},
	If[CBasicTypeQ[typeout] && 
			(Not[CheckIntegerOverflowQ[aflag]] && (Union[typeargs[[All,1]]] === {Integer})) ||
	   		(UnsameQ[First[typeout], Integer] && Not[CheckAnyflowQ[aflag]] && Apply[And, Map[CBasicTypeQ, typeargs]]),
		CAssign[makeAtom[out, objData], COperator[ fun,Map[ makeAtom[#, objData]&, args]]],
	(* else *) 
		If[Length[args] == 2,
			callBinaryMathFunction[objData, fun, out, args, aflag]
			,
			lines =
				Flatten[
					{
					Instruction[fun, out, Take[ args, 2]]
					,
					Map[ Instruction[fun, out, {out, #}]&, Drop[ args, 2]]	
					}];
			Map[ buildLine[ num, objData, #]&, lines]		
		]
	]
]

buildLine[num_, objData_, Instruction[fun_?isUnaryMathFunction, out_, args_List]] :=
Module[{aflag = objData[ArithmeticFlags]},
	{
		If[Length[args] == 1, 
			callUnaryMathFunction[objData, fun, out, First[args], aflag],
			callBinaryMathFunction[objData, fun, out, args, aflag]
		],
		setDataPointer[objData, out]
	}
]

buildLine[ num_, objData_, Instruction[fun_?isBinaryMathFunction, out_, {arg1_, arg2_}]] :=
Module[{aflag = objData[ArithmeticFlags]},
	{
		callBinaryMathFunction[objData, fun, out, {arg1, arg2}, aflag],
		setDataPointer[objData, out]
	}
]

buildLine[ num_, objData_, Instruction["RuntimeError"]] :=
	CReturn["LIBRARY_FUNCTION_ERROR"]

(*
	Compiler internal function calls:
	This is essentilly just like calling a
	LibraryFunction function, but the callback 
	to get the pointer is different.
*)

buildLine[num_, objData_, Instruction["FunctionCall", fun_, out_, args_List]] :=
	BuildFunctionPointerCall[objData, fun, out, args]


(*
	Arithmetic flags decoding
*)
CheckIntegerOverflowQ[flags_] := UnsameQ[BitAnd[flags, IntegerOverflowFlag], 0]
CheckOverflowQ[flags_] := UnsameQ[BitAnd[flags, OverflowFlag], 0]
CheckUnderflowQ[flags_] := UnsameQ[BitAnd[flags, UnderflowFlag], 0]
CheckAnyflowQ[flags_] := UnsameQ[BitAnd[flags, BitOr[UnderflowFlag, UnderflowFlag]], 0]

(*
	MathFunctionPointer, ArgumentSpace: used for calling other LibraryFunctions
*)

getMathFunctionPointerCall[objData_, fcode_Integer, {type1_Integer}] := 
CCall[
	objData[structCompile["getUnaryMathFunction"]], 
	{fcode, type1}
]

getMathFunctionPointerCall[objData_, fcode_Integer, {type1_Integer, type2_Integer}] := 
CCall[
	objData[structCompile["getBinaryMathFunction"]], 
	{fcode, type1, type2}
]

BuildMathFunctionPointerCall[objData_, fun_, funcode_, out_, regs_, types_, aflag_] := 
Module[{name, cnt, funcData, nargs, status = makeAtom[Temporary[0], objData], call, comment},
	nargs = Length[regs];
	funcData = objData[FunctionData];
	(* TODO this collection of MathFunctionName is always failing *)
	name = funcData[MathFunctionName][funcode, types];
	If[!StringQ[name]
		,
		cnt = ((funcData[FunctionPointerCount])++);
		name = makeAtom[FunctionPointer[cnt], objData];
		funcData[MathFunctionOriginalName][cnt] = fun;
		funcData[FunctionPointerCode][cnt] = CAssign[name, getMathFunctionPointerCall[objData, funcode, types]];
		funcData[FunctionPointerType][cnt] = If[nargs == 1, "UnaryMathFunctionPointer", "BinaryMathFunctionPointer"];
		funcData[MathFunctionName][funcode, types] = name;
	];
	If[!StringQ[funcData[MathUnitIncrements]],
		funcData[MathUnitIncrements] = "UnitIncrements"
	];
	comment = CComment[ToString[fun]];
	call = CCall[name, 
		Flatten[{
			Map[VoidPointer[makeAtomAddress[#, objData]]&, Flatten[{out, regs}]],
			1, 
			funcData[MathUnitIncrements],
			aflag
		}]
	];
	CBlock[
		CDeclare[typeToCType[Integer], CAssign[status, call]],
		comment,
		CheckError[objData, 
			CConditional[COperator[Equal, {status, 0}], 0, "LIBRARY_NUMERICAL_ERROR"]
		]
	]
]

BuildMathFunctionCall[objData_, fun_, out_, regs_List, aflag_] := 
Module[{funcode, name, nargs = Length[regs], types, outtype, ranks},
	funcode = If[nargs == 1, Math1ArgCode[fun], Math2ArgCode[fun]];
	outtype = ToTypeEnum[getType[out]];
	types = Map[ToTypeEnum[getType[#]]&, regs];
	ranks = Map[getRank, regs];
	name = Compile`GetCompilerMathFunction[funcode, types, ranks, outtype];
	{
	Switch[name,
		"Call", 
			BuildMathFunctionPointerCall[objData, fun, funcode, out, regs, types, aflag],
		"Math_V_V" | "Math_VV_V",
			CheckError[objData,
				CCall[objData[structCompile[name]],
					Flatten[{funcode, 
						aflag, 
						MapThread[{If[#3 > 0, 0, #2], VoidPointer[makeAtomAddress[#1, objData]]}&, {regs, types, ranks}],
						outtype,
						VoidPointer[makeAtomAddress[out, objData]]
					}]
				]
			],
		"Math_T_T" | "Math_TT_T",
			CheckError[objData,
				CCall[objData[structCompile[name]],
					Flatten[{funcode,
						aflag,
						Map[makeTDAtom[#, objData]&, regs],
						outtype, makeAtomAddress[out, objData]
					}]
				]
			]
	],
	setDataPointer[objData, out]
	}
]

(*
	FunctionPointer, ArgumentSpace: used for calling other LibraryFunctions
*)
makeAtom[FunctionPointer[i_], objData_] := "FP"<>ToString[i];
makeAtom[FunctionPointerArguments, objData_] := "FPA";


getFunctionPointerCall[objData_, dllfun:LibraryFunction[file_String, name_String, __]] := 
CCall[
	objData[structCompile["getLibraryFunctionPointer"]], 
	{CString[name], ToString[InputForm[file]]}
]

getFunctionPointerCall[objData_, fname_String] := 
CCall[
	objData[structCompile["getFunctionCallPointer"]], 
	{CString[fname]}
]

BuildFunctionPointerCall[objData_, fun_, out_, regs_List] := 
	Module[ {name, cnt, funcData, numArgs, args, assignArgs, resArg, call, comment, convert, c = 0, r, f, afun, flowcheck = False},
		funcData = objData[FunctionData];
		name = funcData[DLLFunctionName][fun];
		If[ !StringQ[name]
			,
			cnt = ((funcData[FunctionPointerCount])++);
			name = makeAtom[FunctionPointer[cnt], objData];
			funcData[DLLFunctionName][fun] = name;
			If[objData[WolframRTL] && (Head[fun] === LibraryFunction), 
				Message[iGPUCompileCCodeGenerate::wmreq, fun]
			];
			funcData[DLLFunctions] = Prepend[ funcData[DLLFunctions], {name, fun}];
			funcData[FunctionPointerCode][cnt] = CAssign[name, getFunctionPointerCall[objData, fun]];
			funcData[FunctionPointerType][cnt] = "WGL_LibraryFunctionPointer";
			funcData[MathFunctionOriginalName][cnt] = Null;
		];
		If[Head[fun] === LibraryFunction, 
			(* Check that the LibraryFunction has consistent type/rank *)
			afun = Append[LibraryFunctionArgumentTypes[fun], LibraryFunctionResultType[fun]];
			args = Append[regs, out];
			If[Apply[Or, MapThread[UnsameQ[convertFullType[#1], convertFullType[#2]]&, {afun, args}]],
				Throw[$Failed, "InconsistentArguments"];
			];
			(* From an unknown LibraryFunction we may need to check overunderflow *)
			flowcheck = (CheckAnyflowQ[objData[ArithmeticFlags]] && MatchQ[getType[out], Real | Complex]);
		];
		numArgs = Length[ regs];
		funcData[DLLFunctionArgNum] = Max[numArgs + 1, funcData[DLLFunctionArgNum]];
		objData[DLLFunctionArgNum] = Max[numArgs + 1, objData[DLLFunctionArgNum]];
		args = makeAtom[FunctionPointerArguments, objData];
		convert = Flatten[Last[Reap[
			assignArgs =
					Table[
						r = regs[[i]];
						afun = WGLAgumentGetAddressFunction[r];
						If[RankZeroTensorQ[r],
							f = r /. Tensor[type_, 0]->type;
							r = Temporary[c];
							Sow[
								CDeclare[
									typeToCType[Tensor], 
									CAssign[
										makeAtom[r, objData], 
										CCall[
											objData[structCompile["getRankZeroMemory"]], 
											{VoidPointer[makeAtomAddress[f, objData]], ToTypeEnum[getType[f]], c}
										]
									]
								]
							];
							c++;
						];
						CAssign[
							CCall[afun, {CArray[args, i - 1]}],
							makeAtomAddress[r, objData]
						],
						{i, numArgs}]
		]]];
		resArg = CArray[args, numArgs];
		afun = WGLAgumentGetAddressFunction[out];
		If[StringQ[fun], comment = CComment[fun], comment = Sequence[]];
		call =
		  If[(Head[fun] === LibraryFunction) && ReturnsTensor[fun],
			Module[{oat = makeTDAtom[out, objData], temp = "Temp"}, CBlock[{
				CDeclare[typeToCType[Tensor], CAssign[ temp, oat]],
				CAssign[CCall[afun, {resArg}], CAddress[temp]],
				CheckError[objData, 
					CCall[name, {objData[LibraryData], numArgs, args, resArg}], comment],
				CIf[COperator[Unequal, {temp, oat}],
					CBlock[{
						CCall[objData[structCompile["Memory_copy"]], {makeAtom[out, objData], temp}],
						CCall[objData[structCompile["Memory_free"]], {temp}]
					}]
				]
			}]]
		, (*else *)
			If[UnsameQ[First[convertFullType[out]], VoidType],
				AppendTo[
					assignArgs, 
					CAssign[CCall[afun, {resArg}], makeAtomAddress[out, objData]]
				]
			];
			CheckError[objData, CCall[name, {objData[LibraryData], numArgs, args, resArg}], comment]
		];
		Flatten[{
			If[c > 0,
				CBlock[{convert, assignArgs, call}],
				{assignArgs, call,
					If[flowcheck,
						CheckException[objData, out, objData[ArithmeticFlags]],
						{}
					]
				}
			], 
			setDataPointer[objData, out]
		}]
	]



ReturnsTensor[LibraryFunction[_, _, _, {type_, rank_?Positive, ___}]] = True;
ReturnsTensor[_LibraryFunction] = False;

MakeRegisterType[type_] := type;
MakeRegisterType[{type_, 0, ___}] := type;
MakeRegisterType[{type_, rank_?Positive, ___}] := Tensor[type, rank];
	
LibraryFunctionResultType[LibraryFunction[_, _, _, res_]] := MakeRegisterType[res];
LibraryFunctionArgumentTypes[LibraryFunction[_, _, args_, _]] := Map[MakeRegisterType, args];
(*
	ExpressionPointer: one is used for each external evaluation function
*)
makeAtom[ExpressionPointer[i_], objData_] := "E"<>ToString[i]

TypeRankReference[reg:Register[Tensor[type_, rank_], _], objData_] := {type, rank, VoidPointer[makeAtomAddress[reg, objData]]}
TypeRankReference[Register[VoidType, _], objData_] := {VoidType, 0, 0}
TypeRankReference[reg:Register[type_, _], objData_] := {type, 0, VoidPointer[makeAtomAddress[reg, objData]]}
	
getExpressionPointer[objData_, Hold[fun_]] := 
(
	getExpressionPointer[objData, Hold[fun]] = 
	Module[{c, ep, epstring, epcode, buf},
		c = ((objData[ExpressionPointerCount])++);
		If[objData[WolframRTL] && (c == 0),
			Message[iGPUCompileCCodeGenerate::wmreq, fun]
		];
		ep = makeAtom[ExpressionPointer[c], objData];
		epstring = ToString[FullForm[Hold[fun]]];
		epcode = GetStringConstant[buf, epstring, objData];
		objData[ExpressionPointerCode[c]] = CBlock[{
			epcode,
			CAssign[ep, CCall[objData[structCompile["getExpressionFunctionPointer"]], {objData[LibraryData], buf}]]
		}];
		ep
	]
)

(*
 	TODO: Someone should figure out why we produce two different forms
 	      of the "MainEvaluate" instruction in CompiledFunctionTools
*)

AssignTypeReference[objData_, reg:Register[Tensor[__], _], i_, tvar_, ref_] := {CAssign[CArray[tvar, i], 0], CAssign[CArray[ref, i], VoidPointer[makeAtomAddress[reg, objData]]]}
AssignTypeReference[objData_, Register[VoidType, _], i_, tvar_, ref_] := {CAssign[CArray[tvar, i], ToTypeEnum[VoidType]], CAssign[CArray[ref, i], VoidPointer[0]]}
AssignTypeReference[objData_, reg:Register[type_, _], i_, tvar_, ref_] := {CAssign[CArray[tvar, i], ToTypeEnum[type]], CAssign[CArray[ref, i], VoidPointer[makeAtomAddress[reg, objData]]]}

BuildExternalEvaluation[objData_, hfun_, out_, regs_List] := 
	BuildExternalEvaluation[objData, hfun, out, regs, {{},{}}]

BuildExternalEvaluation[objData_, hfun_, out_Register, regs_List, {dvars_, dregs_}] := 
	BuildExternalEvaluation[objData, hfun, {out}, regs, {dvars, dregs}]

BuildExternalEvaluation[objData_, Hold[fun_], out_List, fregs_List, {dvars_, dregs_}] := 
Module[{n, m, s, ep, type, data, otype, rank, ref, regs},  
	m = Length[dregs];
	s = Length[out] - 1;
	If[m > 0, 
		ep = getExpressionPointer[objData, Hold[{dvars, fun}]],
		ep = getExpressionPointer[objData, Hold[fun]]
	];	
	regs = Join[dregs, fregs];
	n = Length[regs];
	type = makeAtom[Temporary[0], objData];
	data = makeAtom[Temporary[1], objData];
	CBlock[
		DeclareArray["int", type, n],
		DeclareArray["void *", data, n],
		Table[AssignTypeReference[objData, regs[[i]], i - 1, type, data], {i, n}],
		{otype, rank, ref} = TypeRankReference[out[[1]], objData];
		CheckError[objData, CCall[objData[structCompile["evaluateFunctionExpression"]], {objData[LibraryData], ep, m, s, n, type, data, ToTypeEnum[otype], rank, ref}]],
		Map[setDataPointer[objData, #]&, out]
	]
]

BuildGeneratedFunctionCall[objData_, Hold[funRef_], cproc_CompiledProcedure, out_, regs_List] := 
Module[{name, code, funcData, args, flowcheck = 0, aflags = objData[ArithmeticFlags]},
	If[(CheckAnyflowQ[objData[ArithmeticFlags]] && MatchQ[getType[out], Real | Complex]), 
		aflags = First[cproc]["ArithmeticFlags"];
		flowcheck = 0;
		If[CheckOverflowQ[objData[ArithmeticFlags]] && !CheckOverflowQ[aflags],
			flowcheck = OverflowFlag];
		If[CheckUnderflowQ[objData[ArithmeticFlags]] && !CheckUnderflowQ[aflags],
			flowcheck = BitOr[flowcheck, UnderflowFlag]];
	];
	funcData = objData[FunctionData];
	name = funcData[CalledFunctionName][cproc];
	If[!StringQ[name]
		,
		name = "F"<>ToString[(funcData[CalledFunctionCount])++];
		funcData[CalledFunctionName][cproc] = name;
		code = iGPUCompileSymbolicCGenerate[cproc, name, "CodeTarget"->"NestedFunction", "CallerData"->objData];
		(* TODO: should check consistency of arg types *)
		funcData[CalledFunctions] = Prepend[ funcData[CalledFunctions], code]
	];
	args = {Map[makeTDAtom[#, objData]&, regs], makeAtomAddress[out, objData]};
	args = {objData[LibraryData], args};
	args = Flatten[args];
    Flatten[{
		CheckError[objData, CCall[name, args]],
		If[flowcheck > 0, 
			CheckException[objData, out, flowcheck],
			{}
		],
		setDataPointer[objData, out]
    }]
];

TryGeneratedFunctionCall[objData_, hf_, cproc_CompiledProcedure, out_, regs_List] := 
Module[{cinfo = First[cproc]},
	If[Length[cinfo["RuntimeAttributes"]] > 0,
		(* TODO: This is Listable -- we could still check to see
		   if input dimensions match the cf.  *)
		BuildExternalEvaluation[objData, hf, out, regs]
	(* else *),
		BuildGeneratedFunctionCall[objData, hf, cproc, out, regs]
	]
];

(* Comes from CC_COMPILEDFUNCTION *)
buildLine[ num_, objData_, Instruction["CompiledFunctionCall", Hold[fun_CompiledFunction], out_, regs_List]] :=
	TryGeneratedFunctionCall[objData, Hold[fun], ToCompiledProcedure[fun], out, regs]

(* Comes from CC_EVALREG *)
buildLine[ num_, objData_, Instruction["MainEvaluate", Hold[fun_CompiledFunction], out_, regs_List]] :=
	TryGeneratedFunctionCall[objData, Hold[fun], ToCompiledProcedure[fun], out, regs]

buildLine[ num_, objData_, Instruction["MainEvaluate", Hold[fun_LibraryFunction], out_, regs_List]] :=
	BuildFunctionPointerCall[objData, fun, out, regs]

buildLine[ num_, objData_, Instruction["MainEvaluate", Hold[fun_], out_, regs_List]] :=
Module[{svh},
	If[TrueQ[BitAnd[objData[RuntimeFlags], CleanRegistersFlag] > 0],
		svh = Compile`SymbolValueHead[fun]
	];
	Switch[svh,
		CompiledFunction,
			TryGeneratedFunctionCall[objData, Hold[fun], ToCompiledProcedure[fun], out, regs],
		LibraryFunction,
			BuildFunctionPointerCall[objData, fun, out, regs],
		_,
			BuildExternalEvaluation[objData, Hold[fun], out, regs]
	]
]

(* This comes from CC_EVALARG *)
buildLine[ num_, objData_, Instruction["MainEvaluate", fun_Function, out_, regs_List]] :=
	BuildExternalEvaluation[objData, Hold[fun], out, regs]

(* This comes from CC_EVALDYNAMIC *)
buildLine[ num_, objData_, Instruction["MainEvaluate", fun_Function, out_, regs_List, dynamics_List]] :=
	BuildExternalEvaluation[objData, Hold[fun], out, regs, dynamics]

(* 
	Unimplemented (in the sense that there is no C callback) functions
	If $IssueGeneric is true, then make an instruction with the Mathematica name
*)

$IssueGeneric = False;

buildLine[ num_, objData_, inst:Instruction[fun_, out_, args_, ___]] :=
(
	SystemAssert[False];
	If[TrueQ[$IssueGeneric],
		CAssign[makeAtom[out, objData], CCall[fun, If[ListQ[args], Map[makeAtom[#, objData]&, args], makeAtom[args, objData]]]],
		Throw[Message[iGPUCompileCCodeGenerate::nosupp, inst]; $Failed, "Not Implemented"]
	]
)

buildLine[ num_, objData_, InstructionIf[ regTest_, argsTrue_, argsFalse_:{}]] :=
	Module[ {ifTest, trueInsts, falseInsts},
		ifTest = makeAtom[ regTest, objData];
		trueInsts = Map[ buildLine[ num, objData, #]&, argsTrue];
		falseInsts = Map[ buildLine[ num, objData, #]&, argsFalse];	
		If[ argsFalse === {},
			CIf[ ifTest, trueInsts],
			CIf[ ifTest, trueInsts, falseInsts]]
	]

buildLine[ num_, objData_, InstructionFor[ regCount_, regInit_, regLimit_, body_]] :=
	Module[ {atomCount, atomInit, atomLimit, bodyInsts},
		atomCount = makeAtom[ regCount, objData];
		atomInit = makeAtom[ regInit, objData];
		atomLimit = makeAtom[ regLimit, objData];
		bodyInsts = Map[ buildLine[ num, objData, #]&, body];
		CFor[ CAssign[atomCount, atomInit], COperator[ Less, {atomCount, atomLimit}], COperator[ Increment, atomCount], bodyInsts]
	]

buildLine[ num_, objData_, InstructionWhile[ cond__, regTest_, body_]] :=
	Module[ {atomTest, condInsts, bodyInsts},
		atomTest = makeAtom[ regTest, objData];
		condInsts = Map[ buildLine[ num, objData, #]&, cond];
		bodyInsts = Map[ buildLine[ num, objData, #]&, body];
		{
		condInsts,
		CWhile[ atomTest, {bodyInsts,condInsts}]
		}
	]


End[]

EndPackage[]
