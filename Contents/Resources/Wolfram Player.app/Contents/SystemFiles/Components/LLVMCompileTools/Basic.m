

BeginPackage["LLVMCompileTools`Basic`"]


AddConstantMInt::usage = "AddConstantMInt  "

AddConstantInteger::usage = "AddConstantInteger  "

AddConstantMReal

AddConstantReal

AddConstantBoolean::usage = "AddConstantBoolean  "

AddNotBoolean::usage = "AddNotBoolean  "

AddFunctionCall::usage = "AddFunctionCall  "

AddBuildCall

AddBuildInvoke

AddRuntimeFunctionCall

AddRuntimeFunctionInvoke

AddExternalLibraryCall

(*
GetFunctionType::usage = "GetFunctionType  "
*)

AddConstantBoolean::usage = "AddConstantBoolean  "

AddIfTrueBasicBlock::usage = "AddIfTrueBasicBlock  "

AddIfFalseBasicBlock

AddIfElseBasicBlocks::usage = "AddIfElseBasicBlocks  "

AddTypeCast

AddTypeDownCast

AddCastIntegerToPointer

AddCastPointerToInteger

AddSameQ::usage = "AddSameQ  "

AddIntegerToIntegerTypeCast::usage = "AddIntegerToIntegerTypeCast  "

AddIntegerToRealTypeCast::usage = "AddIntegerToRealTypeCast  "

AddLLVMCodeCall::usage = "AddLLVMCodeCall  "

AddConstantString

AddConstantArray

AddNull

AddGetElement

AddSetElement

AddExtractElement

AddInsertElement

AddGetStructureElement

AddSetStructureElement


SetupGlobalFunctionTypes

LLVMAddFunctionAttribute

LLVMAddFunctionParameterAttribute

AddFunctionParameterAttributes

LLVMCreateFunction

AddLLVMGlobalLinkageAttribute

AddLLVMGlobalVisibilityAttribute

AddLLVMGlobalUnnamedAddressAttribute

AddLLVMDLLStorageClass

AddFunctionSystemMetadata

AddExtractValue

Begin["`Private`"]

Needs["LLVMLink`"]
Needs["LLVMCompileTools`Types`"]
Needs["LLVMCompileTools`Globals`"]
Needs["LLVMCompileTools`Complex`"]
Needs["LLVMCompileTools`"]
Needs["CompileUtilities`Error`Exceptions`"]
Needs["LLVMTools`"]
Needs["LLVMLink`LLVMInformation`"]


AddConstantMInt[data_?AssociationQ, val_] :=
	If[val === 0,
		LLVMLibraryFunction["LLVMConstNull"][GetMIntType[data]],
		LLVMLibraryFunction["LLVMConstInt"][GetMIntType[data], val, 0]
	]

AddConstantInteger[data_?AssociationQ, size_, val_] :=
	If[val === 0,
		LLVMLibraryFunction["LLVMConstNull"][GetIntegerType[data, size]],
		LLVMLibraryFunction["LLVMConstInt"][GetIntegerType[data, size], val, 0]
	]

AddConstantMReal[data_?AssociationQ, val_] :=
	If[val === 0.0,
		LLVMLibraryFunction["LLVMConstNull"][GetMRealType[data]],
		LLVMLibraryFunction["LLVMConstReal"][GetMRealType[data], val]
	]


AddConstantReal[data_?AssociationQ, size_, val_] :=
	Module[ {ty = GetRealType[data, size]},
		If[val === 0.0,
			LLVMLibraryFunction["LLVMConstNull"][ty],
			LLVMLibraryFunction["LLVMConstReal"][ty, val]
		]
	]

AddConstantBoolean[data_?AssociationQ, val_] :=
	With[ {bval = If[ TrueQ[val],1,0]},
		LLVMLibraryFunction["LLVMConstInt"][GetBooleanType[data], bval, 0]
	]

(*
  Create a null terminated sequence of bytes from a string argument.  
  This is only used to create constants such as MString or Expr.  
  Perhaps the allocation should be done on the stack, so that memory mangement is taken care of.

  TODO, think about memory management, and fix length of UTF-8 bytes
*)
AddConstantString[ data_?AssociationQ, val_String] :=
	Module[ {strTy, globId, len, strId, t1, arrId, id1, id2, id3},
		len = StringLength[val];
		strTy = LLVMLibraryFunction["LLVMArrayType"][GetIntegerType[data, 8], len+1];
		globId = AddLocalGlobal[data, strTy, "string"];
		strId = LLVMLibraryFunction["LLVMConstStringInContext"][data["contextId"], val, len, 0];
		LLVMLibraryFunction["LLVMSetInitializer"][globId, strId];
		t1 = LLVMLibraryFunction["LLVMConstNull"][GetIntegerType[data, 32]];
		id1 = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildInBoundsGEP"][data["builderId"], globId, #, 2, ""]&, {t1, t1}];
		arrId = LLVMLibraryFunction["LLVMConstInt"][GetIntegerType[data, 32], 1, 0];
		id2 = LLVMLibraryFunction["LLVMBuildArrayAlloca"][data["builderId"], GetCharStarType[data], arrId, ""];
		LLVMLibraryFunction["LLVMBuildStore"][data["builderId"], id1, id2];
		id3 = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], id2, ""];
		id3
	]

AddConstantArray[ data_?AssociationQ, elemTyFun_, consFun_, value_List] :=
	Module[ {elemTy, len, arrayTy, globId, valIds, arrayId, t1, arrId, id1, id2, id3},
		elemTy = elemTyFun[data];
		len = Length[value];
		arrayTy = LLVMLibraryFunction["LLVMArrayType"][elemTy, len];
		globId = AddLocalGlobal[data, arrayTy, "array"];
		valIds = Map[ consFun[data,#]&, value];
		arrayId = WrapIntegerArray[ LLVMLibraryFunction["LLVMConstArray"][elemTy, #, len]&, valIds];
		LLVMLibraryFunction["LLVMSetInitializer"][globId, arrayId];
		t1 = LLVMLibraryFunction["LLVMConstNull"][GetIntegerType[data, 32]];
		id1 = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildInBoundsGEP"][data["builderId"], globId, #, 2, ""]&, {t1, t1}];
		arrId = LLVMLibraryFunction["LLVMConstInt"][GetIntegerType[data, 32], 1, 0];
		id2 = LLVMLibraryFunction["LLVMBuildArrayAlloca"][data["builderId"], GetPointerType[data, elemTy], arrId, ""];
		LLVMLibraryFunction["LLVMBuildStore"][data["builderId"], id1, id2];
		id3 = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], id2, ""];
		id3
	]


(*
 Function Calls
*)


SetupGlobalFunctionTypes[] :=
	Module[ {funData},
		funData =
		<|
			"GetEFAIL" -> ({GetBaseExprType[#], {}}&)
			,
			"EFAIL" -> ({GetBaseExprType[#], None}&)
			,
			"ENULL" -> ({GetBaseExprType[#], None}&)
			,
			"eTrue" -> ({GetBaseExprType[#], None}&)
			,
			"eFalse" -> ({GetBaseExprType[#], None}&)			
			,
			"eNull" -> ({GetBaseExprType[#], None}&)			
			,
			"EternalMTensor" -> ({GetMTensorType[#], None}&)			
			,
			"CreateStringExpr" -> ({ GetBaseExprType[#], {GetCharStarType[#]}}&)
			,
			"CreateMIntegerExpr" -> ({ GetBaseExprType[#], {GetIntegerType[#, 64], GetIntegerType[#, 32], GetIntegerType[#, 32]}}&)
			,
			"CreateMRealExpr" -> ({GetBaseExprType[#], {GetMRealType[#]}}&)
			,
			"Plus_E_E_E" -> ({GetBaseExprType[#], {GetBaseExprType[#], GetBaseExprType[#]}}&)
			,
			"Times_E_E_E" -> ({GetBaseExprType[#], {GetBaseExprType[#], GetBaseExprType[#]}}&)
			,
			"Part_E_E_E" -> ({GetBaseExprType[#], {GetBaseExprType[#], GetBaseExprType[#]}}&)
			,
			"Part_E_I_E" -> ({GetBaseExprType[#], {GetBaseExprType[#], GetMIntType[#]}}&)
			,
			"Equal_E_E_E" -> ({GetBaseExprType[#], {GetBaseExprType[#], GetBaseExprType[#]}}&)
			,
			"Unequal_E_E_E" -> ({GetBaseExprType[#], {GetBaseExprType[#], GetBaseExprType[#]}}&)
			,
			"Greater_E_E_E" -> ({GetBaseExprType[#], {GetBaseExprType[#], GetBaseExprType[#]}}&)
			,
			"GreaterEqual_E_E_E" -> ({GetBaseExprType[#], {GetBaseExprType[#], GetBaseExprType[#]}}&)
			,
			"Less_E_E_E" -> ({GetBaseExprType[#], {GetBaseExprType[#], GetBaseExprType[#]}}&)
			,
			"LessEqual_E_E_E" -> ({GetBaseExprType[#], {GetBaseExprType[#], GetBaseExprType[#]}}&)
			,
			"SameQ_E_E_E" -> ({GetBaseExprType[#], {GetBaseExprType[#], GetBaseExprType[#]}}&)
			,
			"UnsameQ_E_E_E" -> ({GetBaseExprType[#], {GetBaseExprType[#], GetBaseExprType[#]}}&)
			,
			"Cast_E_Boolean" -> ({GetBooleanType[#], {GetBaseExprType[#]}}&)
			,
			"Runtime_DecrementRefCount" -> ({GetMIntType[#], {GetPointerType[#, GetBaseExprType[#]]}}&)
			,
			"Runtime_FreeExpr" -> ({GetMIntType[#], {GetBaseExprType[#]}}&)
			,
			"Runtime_DummyCall" -> ({GetMIntType[#], {GetBaseExprType[#]}}&)
			,
			"MString_toExpr" -> ({GetBaseExprType[#], {GetMStringType[#]}}&)
			,
			"MString_fromExpr" -> ({GetMStringType[#], {GetBaseExprType[#]}}&)
			,
			"UTF8String_toMString" -> ({GetMStringType[#], {GetUTF8StringType[#]}}&)
			,
			"MString_fromUTF8String" -> ({GetUTF8StringType[#], {GetMStringType[#]}}&)
			,
			"StringJoin_MString_MString_MString" -> ({GetMStringType[#], {GetMStringType[#], GetMStringType[#]}}&)
			,
			"StringTake_MString_I_I_MString" -> ({GetMStringType[#], {GetMStringType[#], GetMIntType[#], GetMIntType[#]}}&)
			,
			"StringLength_MString_I" -> ({GetMIntType[#], {GetMStringType[#]}}&)
			,
			"ToCharacterCode_MString_VectorI" -> ({GetMTensorType[#], {GetMStringType[#]}}&)
			,
			"Runtime_RunGarbageCollect" -> ({GetVoidType[#], {}}&)
			,
			"Runtime_CheckGarbageCollect" -> ({GetVoidType[#], {}}&)
			,
			"Runtime_PushStackFrame" -> ({GetVoidType[#], {GetPointerType[#, GetMObjectType[#]], GetMIntType[#]}}&)
			,
			"Runtime_PopStackFrame" -> ({GetVoidType[#], {}}&)
			,
			"SameQ_MObject_MObject_B" -> ({GetBooleanType[#], {GetMObjectType[#], GetMObjectType[#]}}&)
			,
			"Print_MString_Void" -> ({GetVoidType[#], {GetMStringType[#]}}&)
			,
			"NewMString_UI8_MString" -> ({GetMStringType[#], {GetPointerType[#, GetIntegerType[#, 8]]}}&)
			,
			"MObject_setGlobal" -> ({GetBooleanType[#], {GetMObjectType[#]}}&)
			,
			"LookupSymbol_E_E" -> ({GetBaseExprType[#], {GetBaseExprType[#]}}&)
			,
			"CreateHeaded_IE_E" -> ({GetBaseExprType[#], {GetMIntType[#], GetBaseExprType[#]}}&)
			,
			"SetElement_EIE_Void" -> ({GetVoidType[#], {GetBaseExprType[#], GetMIntType[#], GetBaseExprType[#]}}&)
			,
			"Evaluate_E_E" -> ({GetBaseExprType[#], {GetBaseExprType[#]}}&)
			|>;
		funData	
	]

(*
GetFunctionType[data_?AssociationQ, name_] :=
	Module[ {tyFun, res, args, ty},
		tyFun = Lookup[ data["globalFunctionTypes"], name, Null];
		If[ tyFun === Null,
			ThrowException[{"Cannot find runtime function type", name}]
		];
		{res, args} = tyFun[data];
		If[ args === None,
			Return[ res]];
		ty = WrapIntegerArray[ LLVMLibraryFunction["LLVMFunctionType"][res, #, Length[args], 0]&, args];
		ty = LLVMLibraryFunction["LLVMPointerType"][ty, 0];
		ty
	]
*)

AddFunctionCall[ data_?AssociationQ, name_, inputs_] :=
	AddFunctionCall[data, name, inputs, ""]

AddFunctionCall[ data_?AssociationQ, name_, inputs_, varName_] :=
	Module[{id},
		id = GetExternalConstant[data, name];
		id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildCall"][data["builderId"], id, #, Length[inputs], varName]&, inputs];
		id
	]

AddBuildCall[ data_?AssociationQ, funId_, inputs_] :=
	Module[{id},
		id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildCall"][data["builderId"], funId, #, Length[inputs], ""]&, inputs];
		id
	]

AddBuildInvoke[ data_?AssociationQ, funId_, inputs_, toBB_, unwindBB_] :=
	Module[{id},
		id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildInvoke"][data["builderId"], funId, #, Length[inputs], toBB, unwindBB, ""]&, inputs];
		id
	]


AddExternalLibraryCall[ data_?AssociationQ, name_, inputs_] :=
	AddExternalLibraryCall[data, name, inputs, ""]

AddExternalLibraryCall[ data_?AssociationQ, name_, inputs_, varName_] :=
	Module[{funId, id},
		funId = GetExternalLibraryFunction[ data, name];
		id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildCall"][data["builderId"], funId, #, Length[inputs], varName]&, inputs];
		id
	]

AddRuntimeFunctionCall[ data_?AssociationQ, name_, inputs_] :=
	AddRuntimeFunctionCall[data, name, inputs, ""]

AddRuntimeFunctionCall[ data_?AssociationQ, name_, inputs_, varName_] :=
	Module[{funId, id},
		funId = GetRuntimeFunction[ data, name];
		id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildCall"][data["builderId"], funId, #, Length[inputs], varName]&, inputs];
		id
	]

AddRuntimeFunctionInvoke[ data_?AssociationQ, name_, inputs_, toBB_, unwindBB_] :=
	AddRuntimeFunctionInvoke[data, name, inputs, toBB, unwindBB, ""]

AddRuntimeFunctionInvoke[ data_?AssociationQ, name_, inputs_, toBB_, unwindBB_, varName_] :=
	Module[{funId, id},
		funId = GetRuntimeFunction[ data, name];
		id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildInvoke"][data["builderId"], funId, #, Length[inputs], toBB, unwindBB, varName]&, inputs];
		id
	]


AddLLVMCodeCall[ data_?AssociationQ, name_, inputs_] :=
	Module[{codeFun, id},
		codeFun = Lookup[ data["codeFunctions"], name, Null];
		If[codeFun === Null,
			ThrowException[{"Cannot find LLVMCode function", name}]
		];
		id = codeFun[data, name, inputs];
		If[ !IntegerQ[id],
			ThrowException[{"Cannot execute LLVMCode function", name}]];
		id
	]

GetGlobalFromName[ data_?AssociationQ, name_, tyId_] :=
	Module[ {id},
		If[ data["globalData"]["keyExistsQ", name],
			id = data["globalData"]["lookup", name]["id"],
			id = AddLocalGlobal[data, tyId, name];
			data["globalData"]["associateTo", name -> <|"id" -> id, "typeId" -> tyId, "name" -> name|>];
		];
		id = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], id, name];
		id
	]


(*
 Create 2 basic blocks bbTrue and bbFalse
 
 If cond is true go to bbTrue and if false go to bbFalse.
 
 Execute fun for bbTrue, if it wants to join with bbFalse it must add code for this.
 
 At the end the bb is on bbFalse.   So we need to update the BasicBlock Map.
 
 Return a List with result of the custom function and the TrueBB.
 
 Note it does not transfer control from the False to the True BB.
*)

AddIfTrueBasicBlock[data_?AssociationQ, cond_, fun_] :=
	Module[ {bbTrue, bbFalse, id, ef},
		bbTrue = LLVMLibraryFunction["LLVMAppendBasicBlockInContext"][data["contextId"], data["functionId"]["get"], data["getBBName"]["true"]];
		bbFalse = LLVMLibraryFunction["LLVMAppendBasicBlockInContext"][data["contextId"], data["functionId"]["get"], data["getBBName"]["false"]];
        id = LLVMLibraryFunction["LLVMBuildCondBr"][data["builderId"], cond, bbTrue, bbFalse];
        LLVMLibraryFunction["LLVMPositionBuilderAtEnd"][data["builderId"], bbTrue];
        ef = fun[data, bbFalse];
        LLVMLibraryFunction["LLVMPositionBuilderAtEnd"][data["builderId"], bbFalse];
        data["updateBasicBlockMap"][data, bbFalse];
        {ef, bbTrue}
	]

AddIfFalseBasicBlock[data_?AssociationQ, cond_, fun_] :=
	Module[ {bbTrue, bbFalse, id, ef},
		bbFalse = LLVMLibraryFunction["LLVMAppendBasicBlockInContext"][data["contextId"], data["functionId"]["get"], data["getBBName"]["false"]];
		bbTrue = LLVMLibraryFunction["LLVMAppendBasicBlockInContext"][data["contextId"], data["functionId"]["get"], data["getBBName"]["true"]];
        id = LLVMLibraryFunction["LLVMBuildCondBr"][data["builderId"], cond, bbTrue, bbFalse];
        LLVMLibraryFunction["LLVMPositionBuilderAtEnd"][data["builderId"], bbFalse];
        ef = fun[data, bbTrue];
        LLVMLibraryFunction["LLVMPositionBuilderAtEnd"][data["builderId"], bbTrue];
        data["updateBasicBlockMap"][data, bbTrue];
        {ef, bbFalse}
	]


(*
 Create 3 basic blocks bbTrue, bbFalse and bbEnd
 
 If cond is true go to bbTrue and if false go to bbFalse.
 
 Execute funT for bbTrue.
 
 Add branch to bbEnd.
 
 Execute funF for bbFalse.
 
 Add branch to bbEnd.
 
 Make bbEnd the default BB.
  
 Return {funTRes, bbTrue, funFRes, bbFalse}
*)
AddIfElseBasicBlocks[data_?AssociationQ, cond_, funT_, funF_] :=
	Module[ {bbTrue, bbFalse, bbEnd, id, efT, efF},
		bbTrue = LLVMLibraryFunction["LLVMAppendBasicBlockInContext"][data["contextId"], data["functionId"]["get"], data["getBBName"]["true"]];
		bbFalse = LLVMLibraryFunction["LLVMAppendBasicBlockInContext"][data["contextId"], data["functionId"]["get"], data["getBBName"]["false"]];
		bbEnd = LLVMLibraryFunction["LLVMAppendBasicBlockInContext"][data["contextId"], data["functionId"]["get"], data["getBBName"]["end"]];
        id = LLVMLibraryFunction["LLVMBuildCondBr"][data["builderId"], cond, bbTrue, bbFalse];
        LLVMLibraryFunction["LLVMPositionBuilderAtEnd"][data["builderId"], bbTrue];
        efT = funT[data, bbEnd];
        LLVMLibraryFunction["LLVMBuildBr"][data["builderId"], bbEnd];
        LLVMLibraryFunction["LLVMPositionBuilderAtEnd"][data["builderId"], bbFalse];
       	efF = funF[data, bbEnd];
       	LLVMLibraryFunction["LLVMBuildBr"][data["builderId"], bbEnd];
       	LLVMLibraryFunction["LLVMPositionBuilderAtEnd"][data["builderId"], bbEnd];
        data["updateBasicBlockMap"][data, bbEnd];
        {efT, bbTrue, efF, bbFalse}		
	]


AddCastIntegerToPointer[data_?AssociationQ, src_, trgtTy_] :=
	Module[ {id},
		id = LLVMLibraryFunction["LLVMBuildIntToPtr"][data["builderId"], src, trgtTy, "Cast"];
		id
	]

AddCastPointerToInteger[data_?AssociationQ, src_, trgtTy_] :=
	Module[ {id},
		id = LLVMLibraryFunction["LLVMBuildPtrToInt"][data["builderId"], src, trgtTy, "Cast"];
		id
	]

AddTypeDownCast[data_?AssociationQ, src_, trgtTy_] :=
	Module[ {id},
		id = LLVMLibraryFunction["LLVMBuildTrunc"][data["builderId"], src, trgtTy, "DownCast"];
		id
	]


AddTypeCast[data_?AssociationQ, src_, trgtTy_] :=
	Module[ {id},
		id = LLVMLibraryFunction["LLVMBuildBitCast"][data["builderId"], src, trgtTy, "Cast"];
		id
	]

AddIntegerToIntegerTypeCast[data_?AssociationQ, signedQ_, src_, trgtTy_] :=
	Module[ {id},
		id = If[ TrueQ[signedQ], 
				LLVMLibraryFunction["LLVMBuildSExt"][data["builderId"], src, trgtTy, "Cast"],
				LLVMLibraryFunction["LLVMBuildZExt"][data["builderId"], src, trgtTy, "Cast"]];
		id
	]

AddIntegerToRealTypeCast[data_?AssociationQ, signedQ_, src_, trgtTy_] :=
	Module[ {id},
		id = If[ TrueQ[signedQ], 
				LLVMLibraryFunction["LLVMBuildSIToFP"][data["builderId"], src, trgtTy, "Cast"],
				LLVMLibraryFunction["LLVMBuildUIToFP"][data["builderId"], src, trgtTy, "Cast"]];
		id
	]


AddSameQ[data_?AssociationQ, arg1_, arg2_] :=
	Module[ {eqOp, id},
		eqOp = data["LLVMIntPredicate"][SameQ];
		id = LLVMLibraryFunction["LLVMBuildICmp"][data["builderId"], eqOp, arg1, arg2, ""];
		id
	]


AddNotBoolean[ data_?AssociationQ, _, {src_}] :=
	Module[ {id},
		id = LLVMLibraryFunction["LLVMBuildNot"][data["builderId"], src, ""]; 
		id
	]

AddNull[ data_?AssociationQ, ty_] :=
	LLVMLibraryFunction["LLVMConstNull"][ty]


AddGetElement[data_?AssociationQ, src_, offsets_] :=
	Module[ {id},
		id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildInBoundsGEP"][data["builderId"], src, #, Length[offsets], ""]&, offsets];
		LLVMLibraryFunction["LLVMBuildLoad"][ data["builderId"], id, ""]
	]

AddSetElement[data_?AssociationQ, trgt_, offsets_, src_] :=
	Module[ {id},
		id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildInBoundsGEP"][data["builderId"], trgt, #, Length[offsets], ""]&, offsets];
		LLVMLibraryFunction["LLVMBuildStore"][ data["builderId"], src, id]
	]


AddCodeFunction["ExtractElement", AddExtractElement]


AddExtractElement[data_?AssociationQ, _, {src_, ind_}] :=
	Module[ {id},
		id = LLVMLibraryFunction["LLVMBuildExtractElement"][ data["builderId"], src, ind, ""];
		id
	]

AddInsertElement[data_?AssociationQ, trgt_, arg_, ind_] :=
	Module[ {id},
		id = LLVMLibraryFunction["LLVMBuildInsertElement"][ data["builderId"], trgt, arg, ind, ""];
		id
	]

AddGetStructureElement[data_?AssociationQ, src_, index_] :=
	Module[ {ind1, ind2, id},
		ind1 = AddConstantInteger[data, 32, 0];
		ind2 = AddConstantInteger[data, 32, index];
		id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildInBoundsGEP"][data["builderId"], src, #, 2, ""]&, {ind1, ind2}];
		LLVMLibraryFunction["LLVMBuildLoad"][ data["builderId"], id, ""]
	]

AddSetStructureElement[data_?AssociationQ, trgt_, index_, src_] :=
	Module[ {ind1, ind2, id},
		ind1 = AddConstantInteger[data, 32, 0];
		ind2 = AddConstantInteger[data, 32, index];
		id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildInBoundsGEP"][data["builderId"], trgt, #, 2, ""]&, {ind1, ind2}];
		LLVMLibraryFunction["LLVMBuildStore"][ data["builderId"], src, id]
	]

AddExtractValue[data_?AssociationQ, src_, ind_] :=
	Module[ {id},
		id = LLVMLibraryFunction["LLVMBuildExtractValue"][ data["builderId"], src, ind, ""];
		id
	]



(*
 This should probably used the ReinterpretCast mechanism
*)
AddCodeFunction["cast_Real32_Real64", addRealToRealCast[#1, #2, #3, GetRealType[#1, 64]]&]

AddCodeFunction["cast_Integer64_Real64", addIntToRealCast[#1, #2, #3, GetRealType[#1, 64]]&]
AddCodeFunction["cast_UnsignedInteger64_Real64", addUnsignedIntToRealCast[#1, #2, #3, GetRealType[#1, 64]]&]
AddCodeFunction["cast_Integer32_Real64", addIntToRealCast[#1, #2, #3, GetRealType[#1, 64]]&]
AddCodeFunction["cast_UnsignedInteger32_Real64", addUnsignedIntToRealCast[#1, #2, #3, GetRealType[#1, 64]]&]
AddCodeFunction["cast_Integer16_Real64", addIntToRealCast[#1, #2, #3, GetRealType[#1, 64]]&]
AddCodeFunction["cast_UnsignedInteger16_Real64", addUnsignedIntToRealCast[#1, #2, #3, GetRealType[#1, 64]]&]
AddCodeFunction["cast_Integer8_Real64", addIntToRealCast[#1, #2, #3, GetRealType[#1, 64]]&]
AddCodeFunction["cast_UnsignedInteger8_Real64", addUnsignedIntToRealCast[#1, #2, #3, GetRealType[#1, 64]]&]

AddCodeFunction["cast_Integer64_Real32", addIntToRealCast[#1, #2, #3, GetRealType[#1, 32]]&]
AddCodeFunction["cast_UnsignedInteger64_Real32", addUnsignedIntToRealCast[#1, #2, #3, GetRealType[#1, 32]]&]
AddCodeFunction["cast_Integer32_Real32", addIntToRealCast[#1, #2, #3, GetRealType[#1, 32]]&]
AddCodeFunction["cast_UnsignedInteger32_Real32", addUnsignedIntToRealCast[#1, #2, #3, GetRealType[#1, 32]]&]
AddCodeFunction["cast_Integer16_Real32", addIntToRealCast[#1, #2, #3, GetRealType[#1, 32]]&]
AddCodeFunction["cast_UnsignedInteger16_Real32", addUnsignedIntToRealCast[#1, #2, #3, GetRealType[#1, 32]]&]
AddCodeFunction["cast_Integer8_Real32", addIntToRealCast[#1, #2, #3, GetRealType[#1, 32]]&]
AddCodeFunction["cast_UnsignedInteger8_Real32", addUnsignedIntToRealCast[#1, #2, #3, GetRealType[#1, 32]]&]

AddCodeFunction["cast_Integer8_UnsignedInteger8", addIdentityCast[#1, #2, #3, GetUnsignedIntegerType[#1, 8]]&]

AddCodeFunction["cast_Integer16_UnsignedInteger16", addIdentityCast[#1, #2, #3, GetUnsignedIntegerType[#1, 16]]&]
AddCodeFunction["cast_Integer8_UnsignedInteger16", addIntToIntCast[#1, #2, #3, GetUnsignedIntegerType[#1, 16]]&]
AddCodeFunction["cast_UnsignedInteger8_UnsignedInteger16", addIntToUnsignedIntCast[#1, #2, #3, GetUnsignedIntegerType[#1, 16]]&]

AddCodeFunction["cast_Integer8_Integer16", addIntToIntCast[#1, #2, #3, GetIntegerType[#1, 16]]&]
AddCodeFunction["cast_UnsignedInteger8_Integer16", addIntToUnsignedIntCast[#1, #2, #3, GetIntegerType[#1, 16]]&]

AddCodeFunction["cast_Integer32_UnsignedInteger32", addIdentityCast[#1, #2, #3, GetUnsignedIntegerType[#1, 32]]&]
AddCodeFunction["cast_Integer16_UnsignedInteger32", addIntToIntCast[#1, #2, #3, GetUnsignedIntegerType[#1, 32]]&]
AddCodeFunction["cast_Integer8_UnsignedInteger32", addIntToIntCast[#1, #2, #3, GetUnsignedIntegerType[#1, 32]]&]
AddCodeFunction["cast_UnsignedInteger16_UnsignedInteger32", addIntToUnsignedIntCast[#1, #2, #3, GetUnsignedIntegerType[#1, 32]]&]
AddCodeFunction["cast_UnsignedInteger8_UnsignedInteger32", addIntToUnsignedIntCast[#1, #2, #3, GetUnsignedIntegerType[#1, 32]]&]

AddCodeFunction["cast_Integer16_Integer32", addIntToIntCast[#1, #2, #3, GetIntegerType[#1, 32]]&]
AddCodeFunction["cast_Integer8_Integer32", addIntToIntCast[#1, #2, #3, GetIntegerType[#1, 32]]&]
AddCodeFunction["cast_UnsignedInteger16_Integer32", addIntToUnsignedIntCast[#1, #2, #3, GetIntegerType[#1, 32]]&]
AddCodeFunction["cast_UnsignedInteger8_Integer32", addIntToUnsignedIntCast[#1, #2, #3, GetIntegerType[#1, 32]]&]

AddCodeFunction["cast_Integer64_UnsignedInteger64", addIdentityCast[#1, #2, #3, GetUnsignedIntegerType[#1, 64]]&]
AddCodeFunction["cast_Integer32_UnsignedInteger64", addIntToIntCast[#1, #2, #3, GetUnsignedIntegerType[#1, 64]]&]
AddCodeFunction["cast_Integer16_UnsignedInteger64", addIntToIntCast[#1, #2, #3, GetUnsignedIntegerType[#1, 64]]&]
AddCodeFunction["cast_Integer8_UnsignedInteger64", addIntToIntCast[#1, #2, #3, GetUnsignedIntegerType[#1, 64]]&]
AddCodeFunction["cast_UnsignedInteger32_UnsignedInteger64", addIntToUnsignedIntCast[#1, #2, #3, GetUnsignedIntegerType[#1, 64]]&]
AddCodeFunction["cast_UnsignedInteger16_UnsignedInteger64", addIntToUnsignedIntCast[#1, #2, #3, GetUnsignedIntegerType[#1, 64]]&]
AddCodeFunction["cast_UnsignedInteger8_UnsignedInteger64", addIntToUnsignedIntCast[#1, #2, #3, GetUnsignedIntegerType[#1, 64]]&]

AddCodeFunction["cast_Integer32_Integer64", addIntToIntCast[#1, #2, #3, GetIntegerType[#1, 64]]&]
AddCodeFunction["cast_Integer16_Integer64", addIntToIntCast[#1, #2, #3, GetIntegerType[#1, 64]]&]
AddCodeFunction["cast_Integer8_Integer64", addIntToIntCast[#1, #2, #3, GetIntegerType[#1, 64]]&]
AddCodeFunction["cast_UnsignedInteger32_Integer64", addIntToUnsignedIntCast[#1, #2, #3, GetIntegerType[#1, 64]]&]
AddCodeFunction["cast_UnsignedInteger16_Integer64", addIntToUnsignedIntCast[#1, #2, #3, GetIntegerType[#1, 64]]&]
AddCodeFunction["cast_UnsignedInteger8_Integer64", addIntToUnsignedIntCast[#1, #2, #3, GetIntegerType[#1, 64]]&]


addIdentityCast[data_?AssociationQ, fun_, {src_}, trgtTy_] :=
	src

addIntToUnsignedIntCast[data_?AssociationQ, fun_, {src_}, trgtTy_] :=
	Module[ {id},
		id = LLVMLibraryFunction["LLVMBuildZExt"][data["builderId"], src, trgtTy, "Cast"];
		id
	]

addIntToIntCast[data_?AssociationQ, fun_, {src_}, trgtTy_] :=
	Module[ {id},
		id = LLVMLibraryFunction["LLVMBuildSExt"][data["builderId"], src, trgtTy, "Cast"];
		id
	]

addRealToRealCast[data_?AssociationQ, fun_, {src_}, trgtTy_] :=
	Module[ {id},
		id = LLVMLibraryFunction["LLVMBuildFPExt"][data["builderId"], src, trgtTy, "Cast"];
		id
	]

addIntToRealCast[data_?AssociationQ, fun_, {src_}, trgtTy_] :=
	Module[ {id},
		id = LLVMLibraryFunction["LLVMBuildSIToFP"][data["builderId"], src, trgtTy, "Cast"];
		id
	]
	
addUnsignedIntToRealCast[data_?AssociationQ, fun_, {src_}, trgtTy_] :=
	Module[ {id},
		id = LLVMLibraryFunction["LLVMBuildUIToFP"][data["builderId"], src, trgtTy, "Cast"];
		id
	]
		
AddCodeFunction["cast_Real64_ComplexReal64", addCastReal64Complex[#1, #2, #3]&]

addCastReal64Complex[ data_?AssociationQ, _, {src_}] :=
	Module[ {},
		AddCreateVectorComplex[data, Null, {src, AddConstantReal[data, 64, 0]}]
	]

getAttributeEnum[name_] :=
	LLVMLibraryFunction["LLVMGetEnumAttributeKindForName"][name, StringLength[name]]

LLVMAddFunctionAttribute[ data_, funId_, name_] :=
	Module[ {kind, attr},
		kind = getAttributeEnum[name];
		If[ kind === 0,
			ThrowException[{"LLVM attribute not known", name}]
		];
		attr = LLVMLibraryFunction["LLVMCreateEnumAttribute"][data["contextId"], kind, 0];
        LLVMLibraryFunction["LLVMAddAttributeAtIndex"][
        		funId, 
        		LLVMEnumeration["anonymousEnum0", "LLVMAttributeFunctionIndex"],
        		attr
        ];
	]

LLVMAddFunctionAttribute[ data_, funId_, name_ -> value_] :=
	Module[ {attr},
		attr = LLVMLibraryFunction["LLVMCreateStringAttribute"][data["contextId"], name, StringLength[name], value, StringLength[value]];
        LLVMLibraryFunction["LLVMAddAttributeAtIndex"][funId, 
        		LLVMEnumeration["anonymousEnum0", "LLVMAttributeFunctionIndex"], attr];
	]


AddFunctionParameterAttributes[data_, funId_] :=
	AddFunctionParameterAttributes[data, funId, data["callFunctionType"]["get"]]


AddFunctionParameterAttributes[data_, funId_, funTy_] :=
	Module[ {argTys, retTy, i},
        If[!MatchQ[funTy, TypeSpecifier[_List -> _]], 
            Return[]];
        argTys = Part[funTy, 1, 1];
        retTy = Part[funTy, 1, 2];
        Do[checkSigned[data, funId, Part[argTys,i], i], {i,Length[argTys]}];
        checkSigned[data, funId, retTy, -1];
	]


$signExtend = 
<|
	"Integer8" -> "signext",
	"Integer16" -> "signext",
	"UnsignedInteger8" -> "zeroext",
	"UnsignedInteger16" -> "zeroext"
|>


checkSigned[data_, funId_, ty_, index_] :=
	Module[ {ext = Lookup[$signExtend,ty, Null]},
		If[ ext =!= Null,
			LLVMAddFunctionParameterAttribute[data, funId, ext, index]];
	]

getParameterEnum[index_] :=
	index
	
getParameterEnum[ index_ /; index < 1] :=
	LLVMEnumeration["anonymousEnum0", "LLVMAttributeReturnIndex"]

LLVMAddFunctionParameterAttribute[ data_, funId_, name_, index_] :=
	Module[ {kind, enum, attr},
		kind = getAttributeEnum[name];
		enum = getParameterEnum[index];
		If[ kind === 0,
			ThrowException[{"LLVM attribute not known", name}]
		];
		attr = LLVMLibraryFunction["LLVMCreateEnumAttribute"][data["contextId"], kind, 0];
        LLVMLibraryFunction["LLVMAddAttributeAtIndex"][
        		funId, 
        		enum,
        		attr
        ];
	]


Options[LLVMCreateFunction] = {
    "LLVMLinkage" -> None,
    "LLVMVisibility" -> None,
    "LLVMUnnamedAddress" -> None,
    "LLVMDLLStorageClass" -> None
}

LLVMCreateFunction[ data_, name_, tyId_, unresTy_, opts:OptionsPattern[]] :=
    LLVMCreateFunction[data, name, tyId, unresTy, <| opts |>]
LLVMCreateFunction[ data_, name_, tyId_, unresTy_, opts_?AssociationQ] :=
	With[{
		funId = LLVMLibraryFunction["LLVMAddFunction"][data["moduleId"], name, tyId]
	},
		If[!IntegerQ[funId],
			ThrowException[CompilerException[{"function Id is not an integer", funId}]]
		];
		AddFunctionParameterAttributes[data, funId, unresTy];
        AddLLVMGlobalLinkageAttribute[data, funId, Lookup[opts, "LLVMLinkage", None]];
        AddLLVMGlobalVisibilityAttribute[data, funId, Lookup[opts, "LLVMVisibility", None]];
        AddLLVMGlobalUnnamedAddressAttribute[data, funId, Lookup[opts, "LLVMUnnamedAddress", None]];
        AddLLVMDLLStorageClass[data, funId, Lookup[opts, "LLVMDLLStorageClass", None]];
        AddFunctionSystemMetadata[data, funId];
        Switch[Lookup[opts, "LLVMInline", None],
            None,
                LLVMAddFunctionAttribute[data, funId, "noinline"],
            "Hint", 
                LLVMAddFunctionAttribute[data, funId, "inlinehint"],
            "Always",
                LLVMAddFunctionAttribute[data, funId, "alwaysinline"]
        ];
		funId
	]

AddFunctionSystemMetadata[data_, funId_] :=
    If[TrueQ[data["MachineArchitecture"]] && (data["targetSystemID"] === Automatic || data["targetSystemID"] === $SystemID),
        LLVMAddFunctionAttribute[data, funId, "target-cpu" -> LLVMLibraryFunction["LLVMGetHostCPUName"][]];
        LLVMAddFunctionAttribute[data, funId, "target-features" -> LLVMLibraryFunction["LLVMGetHostCPUFeatures"][]]
    ];

AddLLVMGlobalLinkageAttribute[data_?AssociationQ, id_, None] := Nothing; (* nothing to do *)
AddLLVMGlobalLinkageAttribute[data_?AssociationQ, id_, linkage_] :=
    LLVMLibraryFunction["LLVMSetLinkage"][id, LLVMEnumeration["LLVMLinkage", linkage]]
    
AddLLVMGlobalVisibilityAttribute[data_?AssociationQ, id_, None] := Nothing; (* nothing to do *)
AddLLVMGlobalVisibilityAttribute[data_?AssociationQ, id_, vis_] :=
    LLVMLibraryFunction["LLVMSetVisibility"][id, LLVMEnumeration["LLVMVisibility", vis]]
    
AddLLVMGlobalUnnamedAddressAttribute[data_?AssociationQ, id_, None] := Nothing; (* nothing to do *)
AddLLVMGlobalUnnamedAddressAttribute[data_?AssociationQ, id_, unnamed_] :=
    If[$LLVMInformation["LLVM_VERSION"] >= 7.0,
        LLVMLibraryFunction["LLVMSetUnnamedAddress"][id, LLVMEnumeration["LLVMUnnamedAddr", unnamed]],
        Nothing (* not supported on older versions of llvm *)
    ]
    
AddLLVMDLLStorageClass[data_?AssociationQ, id_, None] := Nothing; (* nothing to do *)
AddLLVMDLLStorageClass[data_?AssociationQ, id_, storage_] :=
    LLVMLibraryFunction["LLVMSetDLLStorageClass"][id, LLVMEnumeration["LLVMDLLStorageClass", storage]]

End[]


EndPackage[]

