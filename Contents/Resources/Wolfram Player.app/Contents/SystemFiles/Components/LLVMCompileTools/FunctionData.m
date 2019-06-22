BeginPackage["LLVMCompileTools`FunctionData`"]

GetFunctionData

Begin["`Private`"]

Needs["LLVMTools`"]
Needs["CompileUtilities`Error`Exceptions`"]
Needs["LLVMLink`"]
Needs["CompileUtilities`Reference`"]

(*
  Implementation of GetFunctionData
  
  GetFunctionData[ LLVMModule[mod]]
  
  returns a list of elements
  
  <|"mode" -> "Function", "name" -> name, "type" -> type|>
  
  for each function in the module.
*)


GetFunctionData[ LLVMModule[mod_]] :=
	Module[ {list = CreateReference[{}], funIterator},
		funIterator = LLVMLibraryFunction["LLVMGetFirstFunction"][mod];
		While[funIterator =!= 0,
			list["appendTo", processFunction[funIterator]];
			funIterator = LLVMLibraryFunction["LLVMGetNextFunction"][funIterator]];
		list["get"]
	]


processFunction[ fun_] :=
	Module[{name = LLVMLibraryFunction["LLVMGetValueName"][fun], 
			type = GetType[LLVMLibraryFunction["LLVMTypeOf"][fun]]},
		If[ MatchQ[ type, "Handle"[_]],
			type = First[type]];
		<|"class" -> "Function", "Name" -> name, "Type" -> TypeSpecifier[type]|>
	]

GetType[ty_] :=
	Module[{kind = LLVMLibraryFunction["LLVMGetTypeKind"][ty]},
		Which[
			kind === LLVMEnumeration["LLVMTypeKind", "LLVMVoidTypeKind"],
				"Void"
			,
			kind === LLVMEnumeration["LLVMTypeKind", "LLVMPointerTypeKind"],
				getPointerType[ty]
			,
			kind === LLVMEnumeration["LLVMTypeKind", "LLVMFunctionTypeKind"],
				getFunctionType[ty]
			,
			kind === LLVMEnumeration["LLVMTypeKind", "LLVMIntegerTypeKind"],
				getIntegerType[ty]
			,
			
			True,
				ThrowException[{"Type kind is not handled", kind}]]
	]




getPointerType[ ty_] :=
	"Handle"[ GetType[ LLVMLibraryFunction["LLVMGetElementType"][ty]]]

getFunctionType[ ty_] :=
	Module[ {num = LLVMLibraryFunction["LLVMCountParamTypes"][ty], args, argsArray, res},

		ScopedAllocation["LLVMOpaqueTypeObjectPointer"][ Function[{argsArray},
			LLVMLibraryFunction["LLVMGetParamTypes"][ty, argsArray];
			args = Table[ LLVMLibraryFunction["LLVMLink_getLLVMOpaqueTypeObjectPointer"][argsArray, i], {i, 0, num-1}];
		], num ];

		args = Map[ GetType, args];
		res = GetType[ LLVMLibraryFunction["LLVMGetReturnType"][ty]];
		args -> res
	]
	
$intTypes =
<|
	8 -> "Integer8",
	16 -> "Integer16",
	32 -> "Integer32",
	64 -> "Integer64"
|>

getIntegerType[ ty_] :=
	Module[{width, res},
		width = LLVMLibraryFunction["LLVMGetIntTypeWidth"][ty];
		res = Lookup[$intTypes, width, Null];
		If[ res === Null,
			ThrowException[{"Integer type is not handled", width}]
		];
		res
	]


End[]


EndPackage[]

