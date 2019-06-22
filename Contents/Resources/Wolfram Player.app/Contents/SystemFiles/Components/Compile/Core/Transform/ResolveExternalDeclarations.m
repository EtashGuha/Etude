BeginPackage["Compile`Core`Transform`ResolveExternalDeclarations`"]

ResolveExternalDeclarationsPass

Begin["`Private`"] 

Needs["Compile`"]
Needs["Compile`Core`PassManager`ProgramModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Callback`"]
Needs["LLVMCompileTools`"]
Needs["LLVMTools`"]
Needs["LLVMCompileTools`FunctionData`"]



addFunctionData[pm_, tyEnv_, mod_, dataIn_, "Function"] :=
	Module[ {name, funTy, data = dataIn},
		name = Lookup[ data, "Name", Null];
		funTy = Lookup[ data, "Type", Null];
		If[ name === Null || funTy === Null,
			ThrowException[{"Incorrect function data specification.", data}]];
		name = Native`PrimitiveFunction[name];
		tyEnv["declareFunction", name, funTy];
		data = Prepend[data, "LLVMModule" -> mod];
		data = Prepend[data, "Linkage" -> "LLVMModule"];
		pm["externalDeclarations"]["addFunction", name, data];
	]


processDeclaration[ pm_, LLVMString[ str_String]] :=
	Module[ {cont = GetLLVMContext[], mod, functionData, tyEnv},
		mod = LLVMToModule[str, "LLVMContext" -> cont];
		If[ !MatchQ[ mod, LLVMModule[_Integer]],
			ThrowException[{"Could not convert LLVMString to an LLVM module", {str}}]];
		(*
		  Now we need to get all the function definitions, add 
		  them as functions with name/type and linkage.
		*)
		functionData = GetFunctionData[ mod];
		tyEnv = pm["typeEnvironment"];
		tyEnv["reopen"];
		Scan[ addFunctionData[ pm, tyEnv, mod, #, #["class"]]&, functionData];
		tyEnv["finalize"];
	]

processDeclaration[args___] :=
    ThrowException[{"Invalid call to processDeclaration.", {args}}]

run[pm_, opts_] :=
	Module[{rawData = pm["externalDeclarations"]["rawData"]["get"]},
		Scan[ processDeclaration[pm,#]&, rawData]
	]


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"ResolveExternalDeclarations",
	"The pass resolves all external declarations."
];

ResolveExternalDeclarationsPass = CreateProgramModulePass[<|
	"information" -> info,
	"runPass" -> run
|>];

RegisterPass[ResolveExternalDeclarationsPass]
]]

End[]
	
EndPackage[]
