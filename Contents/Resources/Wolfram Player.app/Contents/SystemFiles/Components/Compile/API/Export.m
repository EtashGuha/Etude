

BeginPackage["Compile`API`Export`"]


Begin["`Private`"]

Needs["Compile`"]
Needs["LLVMTools`"]
Needs[ "Compile`API`Utilities`"]

prots = Unprotect[FunctionCompileExport, FunctionCompileExportString, FunctionCompileExportByteArray]

Clear[FunctionCompileExport]
Clear[FunctionCompileExportString]
Clear[FunctionCompileExportByteArray]

FunctionCompileExport::path = "Path `1` is not a string."
FunctionCompileExport::pathw = "Cannot open `1`."
FunctionCompileExport::file = "Input `1` cannot be exported to `2`."
FunctionCompileExport::form = "Format `1` is invalid. Valid formats include \"LLVM\", \"LLVMBinary\", \"Assembler\" and \"Binary\"."


FunctionCompileExport::fun = FunctionCompileExportByteArray::fun = FunctionCompileExportString::fun = 
	"Argument `1` is expected to be a Function or a CompiledCodeFunction."
	
FunctionCompileExport::ccfinp = FunctionCompileExportByteArray::ccfinp = FunctionCompileExportString::ccfinp = 
	"`1` is not recognised as a CompiledCodeFunction that was created by the Wolfram Compiler and cannot be used for export."
	
FunctionCompileExportString::form = "Format `1` is invalid. Valid formats include \"LLVM\" and \"Assembler\"."
FunctionCompileExport::trg = FunctionCompileExportByteArray::trg = FunctionCompileExportString::trg = 
	"TargetSystem `1` is invalid. Valid targets include \"MacOSX-x86-64\",  \"Windows\", and \"Linux-ARM\"."
	
FunctionCompileExport::llvm = FunctionCompileExportByteArray::llvm = FunctionCompileExportString::llvm = "Input `1` cannot be converted to LLVM."
FunctionCompileExportString::asm = "Input `1` cannot be converted to assembler."
FunctionCompileExport::comp = FunctionCompileExportByteArray::comp = FunctionCompileExportString::comp = "Input `1` cannot be compiled."

FunctionCompileExportByteArray::form = "Format `1` is invalid. Valid formats include \"LLVMBinary\" and \"Binary\"."



(*
 FunctionCompileExport
*)

Options[FunctionCompileExport] = 
	{
	CompilerOptions -> Automatic,
	TargetSystem -> Automatic	
	}


FunctionCompileExport[ path_, func_, opts:OptionsPattern[]] :=
	Module[ {res = Catch[
					iFunctionCompileExportDriver[exportData, path, func, determineFormat[path], opts],
					_compileExportException,
					Part[#2,1]&]
	},
		res /; res =!= Null
	]

(*
  Need the condition on format because an option setting in opts might be rejected.
*)
FunctionCompileExport[ path_, func_, format_ /; !OptionQ[format], opts:OptionsPattern[]] :=
	Module[ {res = Catch[
					iFunctionCompileExportDriver[exportData, path, func, format, opts],
					_compileExportException,
					Part[#2,1]&]
	},
		res /; res =!= Null
	]
	
FunctionCompileExport[args___ /; (compileArgumentError[{args}, FunctionCompileExport, 2, 3]; False)] :=
	Null

	
autoFormats = 
<|
  "asm" -> "Assembler",
  "s" -> "Assembler",
  "o" -> "Binary",
  "obj" -> "Binary",
  "ll" -> "LLVM",
  "bc" -> "LLVMBinary"
|>

determineFormat[path_String] :=
	Module[{ext = FileExtension[path]},
		Lookup[autoFormats, ext, "Binary"]
	]

(*
 Error will be reported later
*)
determineFormat[path_] :=
	"LLVM"



exportData = <| 
	"checkPath" -> checkFilePath,
	"checkFormat" -> checkFileFormat,
	"head" -> FunctionCompileExport, 
	"generationFunction" -> processToFile
|>



(*
 FunctionCompileExportString
*)

Options[FunctionCompileExportString] = 
	{
	CompilerOptions -> Automatic,
	TargetSystem -> Automatic	
	}


FunctionCompileExportString[ func_, opts:OptionsPattern[]] :=
	Module[ {res = Catch[
					iFunctionCompileExportDriver[exportStringData, Null, func, "LLVM", opts],
					_compileExportException,
					Part[#2,1]&]
	},
		res /; res =!= Null
	]
	
FunctionCompileExportString[ func_, format_ /; !OptionQ[format], opts:OptionsPattern[]] :=
	Module[ {res = Catch[
					iFunctionCompileExportDriver[exportStringData, Null, func, format, opts],
					_compileExportException,
					Part[#2,1]&]
	},
		res /; res =!= Null
	]

FunctionCompileExportString[args___ /; (compileArgumentError[{args}, FunctionCompileExportString, 1, 2]; False)] :=
	Null

exportStringData = <|
	"checkPath" -> (Null&),
	"checkFormat" -> checkStringFormat,
	"head" -> FunctionCompileExportString, 
	"generationFunction" -> processToString
|>

(*
 FunctionCompileExportByteArray
*)

Options[FunctionCompileExportByteArray] = 
	{
	CompilerOptions -> Automatic,
	TargetSystem -> Automatic	
	}


FunctionCompileExportByteArray[ func_, opts:OptionsPattern[]] :=
	Module[ {res = Catch[
					iFunctionCompileExportDriver[exportByteArrayData, Null, func, "LLVMBinary", opts],
					_compileExportException,
					Part[#2,1]&]
	},
		res /; res =!= Null
	]
	
FunctionCompileExportByteArray[ func_, format_ /; !OptionQ[format], opts:OptionsPattern[]] :=
	Module[ {res = Catch[
					iFunctionCompileExportDriver[exportByteArrayData, Null, func, format, opts],
					_compileExportException,
					Part[#2,1]&]
	},
		res /; res =!= Null
	]

FunctionCompileExportByteArray[args___ /; (compileArgumentError[{args}, FunctionCompileExportByteArray, 1, 2]; False)] :=
	Null

exportByteArrayData = <|
	"checkPath" -> (Null&),
	"checkFormat" -> checkByteArrayFormat,
	"head" -> FunctionCompileExportByteArray, 
	"generationFunction" -> processToByteArray
|>


(*
  On error throw compileExportException[ Null] to return unevaluated and 
  compileExportException[ arg] to return arg. 
*)

iFunctionCompileExportDriver[ data_, pathIn_, funcIn_, format_, opts:OptionsPattern[]] :=
	Module[ {func, path, mod, trg, ccFOpts = getCompilerOptions[data["head"], compileExportException, {opts}]},
		func = checkFunctionForExport[ data, funcIn];
		path = data["checkPath"][ data, pathIn];
		data["checkFormat"][ data, format];
		trg = OptionValue[data["head"], {opts}, TargetSystem];
		checkTarget[ data, trg];
		mod = Quiet[CompileToLLVMModule[func, ccFOpts, "TargetSystemID" -> trg]];
		If[ !MatchQ[mod, LLVMModule[_Integer]],
			With[ {h = data["head"]},
				Message[h::comp, func]; 
				Throw[Null, compileExportException[makeFailure[mod]]]]];
		data["generationFunction"][ data, path, func, mod, format, opts]
	]

isLLVM[ form_] :=
	MemberQ[ {"LLVM", "LLVMBinary"}, form]

isMachineCode[ form_] :=
	MemberQ[ {"Assembler", "Binary"}, form]

isBinary[ form_] :=
	StringMatchQ[form, "*Binary"]


processToFile[ data_, path_, func_, LLVMModule[id_], format_?isLLVM, opts:OptionsPattern[]] :=
	Module[ {res, binary = isBinary[format]},
		res = If[binary,
				Quiet[LLVMToBitcodeFile[path, LLVMModule[id]]],
				Quiet[LLVMToLLFile[path, LLVMModule[id]]]];
		If[ res,
			With[ {h = data["head"]},
				Message[h::file, func, path]; 
				Throw[Null, compileExportException[$Failed]]]];
		path
	]

processToFile[ data_, path_, func_, LLVMModule[id_], format_?isMachineCode, opts:OptionsPattern[]] :=
	Module[ {trg, res, binary = isBinary[format]},
		trg = OptionValue[data["head"], {opts}, TargetSystem];
		res = If[binary,
				Quiet[LLVMToMachineCodeFile[LLVMModule[id], path, "TargetSystemID" -> trg]],
				Quiet[LLVMToMachineCodeFile[LLVMModule[id], path, "OutputType" -> "Assembly", "TargetSystemID" -> trg]]];
		If[ res === $Failed,
			With[ {h = data["head"]},
				Message[h::file, func, path]; 
				Throw[Null, compileExportException[$Failed]]]];
		path
	]



processToString[ data_, path_, func_, LLVMModule[id_], "LLVM", opts:OptionsPattern[]] :=
	Module[ {res},
		res = Quiet[LLVMToString[LLVMModule[id]]];
		If[ !StringQ[res],
			With[ {h = data["head"]},
				Message[h::llvm, func]; 
				Throw[Null, compileExportException[makeFailure[res]]]]];
		res
	]

processToString[ data_, path_, func_, LLVMModule[id_], "Assembler", opts:OptionsPattern[]] :=
	Module[ {res, trg},
		trg = OptionValue[data["head"], {opts}, TargetSystem];
		res = Quiet[LLVMToMachineCodeString[LLVMModule[id], "OutputType" -> "Assembly", "TargetSystemID" -> trg]];
		If[ !StringQ[res],
			With[ {h = data["head"]},
				Message[h::asm, func]; 
				Throw[Null, compileExportException[makeFailure[res]]]]];
		res
	]


processToByteArray[ data_, pathDumy_, func_, LLVMModule[id_], format_, opts:OptionsPattern[]] :=
	Module[ {path, bytes},
		path = FileNameJoin[ {CreateDirectory[], "tmp.out"}];
		path = processToFile[data, path, func, LLVMModule[id], format, opts];
		bytes = ReadList[path, "Byte", "RecordSeparators" -> {}];
		ByteArray[bytes]
	]





checkStringFormat[ data_, format_] :=
	If[! MemberQ[{"LLVM", "Assembler"}, format],
		With[ {h = data["head"]},
			Message[h::form, format];
			Throw[Null, compileExportException[Null]]]
	]

checkByteArrayFormat[ data_, format_] :=
	If[! MemberQ[{"LLVMBinary", "Binary"}, format],
		With[ {h = data["head"]},
			Message[h::form, format];
			Throw[Null, compileExportException[Null]]]
	]

checkFileFormat[ data_, format_] :=
	If[! MemberQ[{"LLVM", "Assembler", "LLVMBinary", "Binary"}, format],
		With[ {h = data["head"]},
			Message[h::form, format];
			Throw[Null, compileExportException[Null]]]
	]


checkTarget[ data_, Automatic] :=
	Null
	
checkTarget[ data_, "WebAssembly"] :=
	Null

checkTarget[ data_, trg_] :=
	If[ !StringQ[Quiet[LLVMTripleFromSystemID[trg]]],
		With[ {h = data["head"]},
			Message[h::trg, trg];
			Throw[Null, compileExportException[Null]]]
		]




End[]

EndPackage[]
