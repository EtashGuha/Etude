

BeginPackage["LLVMCompileTools`Globals`"]


GetGlobalFromName

GetLinkedFunctionFromName

CreateGlobal

GetExistingGlobal

SetExistingGlobal

GetExprConstant

GetStringConstant

GetPackedArrayConstant

GetExternalConstant

ProcessConstants

GetCStringConstant

AddLocalGlobal

Begin["`Private`"]

Needs["LLVMLink`"]
Needs["LLVMCompileTools`Types`"]
Needs["LLVMCompileTools`Basic`"]
Needs["CompileUtilities`Error`Exceptions`"]
Needs["LLVMTools`"]


CreateGlobal[ data_?AssociationQ, name_, tyId_, valId_:None] :=
	Module[ {id},
		id = AddGlobal[data, name, tyId, False];
		If[ valId =!= None,
			LLVMLibraryFunction["LLVMSetInitializer"][id, valId]];
		id
	]

GetExistingGlobal[data_?AssociationQ, name_] :=
	Module[ {id, spec},
		spec = data["globalData"]["lookup", name, Null];
		If[ spec === Null,
			ThrowException[{"Existing global cannot be found ", name}]];
		id = spec["id"];
		id = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], id, name];
		id
	]

SetExistingGlobal[data_?AssociationQ, name_, valId_] :=
	Module[ {spec, id},
		spec = data["globalData"]["lookup", name, Null];
		If[ spec === Null,
			ThrowException[{"Existing global cannot be found ", name}]];
		id = spec["id"];
		LLVMLibraryFunction["LLVMBuildStore"][data["builderId"], valId, id];
	]

GetGlobalFromName[ data_?AssociationQ, name_, tyId_] :=
	Module[ {id},
		If[ data["globalData"]["keyExistsQ", name],
			id = data["globalData"]["lookup", name]["id"],
			id = AddGlobal[data, name, tyId, True];
			];
		id = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], id, name];
		id
	]

AddGlobal[ data_?AssociationQ, name_, tyId_, needsInit_] :=
	Module[ {id},
		id = AddLocalGlobal[data, tyId, name];
		data["globalData"]["associateTo", name -> 
					<|"id" -> id, "typeId" -> tyId, "name" -> name, "needsInit" -> needsInit |>];
		id
	]

GetLinkedFunctionFromName[ data_?AssociationQ, name_, tyId_] :=
	Module[ {id},
		If[ data["globalData"]["keyExistsQ", name],
			id = data["globalData"]["lookup", name]["id"],
			id = AddLinkedFunction[data, name, tyId];
			];
		id
	]



AddLinkedFunction[data_?AssociationQ, name_, tyId_] :=
	Module[ {id, tyIdElem},
		tyIdElem = LLVMLibraryFunction["LLVMGetElementType"][tyId];
		id = LLVMLibraryFunction["LLVMAddFunction"][data["moduleId"], name, tyIdElem];
		AddFunctionParameterAttributes[data, id];
		data["globalData"]["associateTo", name -> 
					<|"id" -> id, "typeId" -> tyIdElem, "name" -> name, "needsInit" -> False |>];
		id
	]




(*
  Exprs are initialized to Null.  This is fine since this is all outside 
  of refcounting.  The exprs are actually initialized in the initialization function,.
*)
GetExprInit[data_?AssociationQ] :=
	Module[ {id},
		id = AddNull[data, GetBaseExprType[data]];
		id
	]


(*
   Functions for working with Constants that need global initialization
*)

(*
  CreateMInteger takes the value (as BIT64),  the size and signedQ.
*)
createConstantMIntegerExpr[data_?AssociationQ, value_] :=
	Module[ {valId, sizeId, signedId = AddConstantInteger[data, 32, 1]},
		valId = AddConstantMInt[data, value];
		If[data["machineIntegerSize"] === 32,
			valId = LLVMLibraryFunction["LLVMBuildSExt"][data["builderId"], valId, GetIntegerType[data, 64], "Cast"];
			sizeId = AddConstantInteger[data, 32, 32]
			,
			sizeId = AddConstantInteger[data, 32, 64]];	
		valId = AddFunctionCall[ data, "CreateMIntegerExpr", {valId, sizeId, signedId}];
		valId
	]

GetMIntegerExprConstant[data_?AssociationQ, value_] :=
	Module[ {locationId, name},
		GetExternalConstant[ data, "CreateMIntegerExpr"];
		name = "MIntConstant" <> ToString[value];
		locationId = AddLocalGlobal[data, GetBaseExprType[data], name];
		<| "locationId" -> locationId, "value" -> value, "initializer" -> GetExprInit, "creator" -> createConstantMIntegerExpr, "name" -> name|>
	]


createConstantStringExpr[data_?AssociationQ, value_] :=
	Module[ {valId},
		valId = AddConstantString[data, value];
		valId = AddFunctionCall[ data, "CreateStringExpr", {valId}];
		valId
	]


GetStringExprConstant[data_?AssociationQ, value_] :=
	Module[ {locationId, name},
		GetExternalConstant[ data, "CreateStringExpr"];
		name = "ExprString";
		locationId = AddLocalGlobal[data, GetBaseExprType[data], name];
		<| "locationId" -> locationId, "value" -> value, "initializer" -> GetExprInit, "creator" -> createConstantStringExpr, "name" -> name|>
	]


createConstantSymbolExpr[data_?AssociationQ, value_] :=
	Module[ {valId},
		valId = AddConstantString[data, value];
		valId = AddFunctionCall[ data, "CreateStringExpr", {valId}];
		valId = AddFunctionCall[ data, "LookupSymbol_E_E", {valId}];
		valId
	]


GetSymbolExprConstant[data_?AssociationQ, Primitive`GlobalSymbol[value_]] :=
	Module[ {locationId, name},
		name = "ExprSymbol";
		GetExternalConstant[ data, "CreateStringExpr"];
		GetExternalConstant[ data, "LookupSymbol_E_E"];
		locationId = AddLocalGlobal[data, GetBaseExprType[data], name];
		<| "locationId" -> locationId, "value" -> value, "initializer" -> GetExprInit, "creator" -> createConstantSymbolExpr, "name" -> name|>
	]


createConstantExpr[data_?AssociationQ, value_, funName_] :=
	Module[ {valId},
		valId = AddFunctionCall[ data, funName, {}];
		valId
	]
	
(*
  This is only used for EFAIL,  but could probably be done differently.
*)
GetConstantExpr[data_?AssociationQ, consName_, funName_] :=
	Module[ {locationId},
		GetExternalConstant[ data, funName];
		locationId = AddLocalGlobal[data, GetBaseExprType[data], consName];
		<| "locationId" -> locationId, "value" -> Null, "initializer" -> GetExprInit, "creator" -> (createConstantExpr[#1, #2, funName]&),  "name" -> consName|>
	]

addConstantData[ data_?AssociationQ, key_, value_] :=
	(
	data["constantData"]["associateTo", key -> value];
	data["constantDataList"]["appendTo", value]
	)



createConstantGeneralExpr[data_?AssociationQ, value_] :=
	Module[ {valStr, valId},
		valStr = ToString[value, InputForm];
		valId = AddConstantString[data, valStr];
		valId = AddFunctionCall[ data, "CreateGeneralExpr", {valId}];
		valId
	]


GetGeneralExprConstant[data_?AssociationQ, value_] :=
	Module[ {locationId, name},
		name = "ExprGeneral";
		GetExternalConstant[ data, "CreateGeneralExpr"];
		locationId = AddLocalGlobal[data, GetBaseExprType[data], name];
		<| "locationId" -> locationId, "value" -> value, "initializer" -> GetExprInit, "creator" -> createConstantGeneralExpr, "name" -> name|>
	]



(*
  Wrap the value in the type,  so make sure the same value with different types 
  can be added.
*)
GetExprConstant[data_?AssociationQ, value_] :=
	Module[ {info, id},
		info = data["constantData"]["lookup", "Expr"[value], Null];
		info =
			Which[
				info =!= Null,
					info,
				Developer`MachineIntegerQ[value],
					GetMIntegerExprConstant[data, value],
				StringQ[value],
					GetStringExprConstant[data, value],
				MatchQ[value, Primitive`GlobalSymbol[_String]],
				    GetSymbolExprConstant[data, value],
				value === Expr`EFAIL,
					GetConstantExpr[data, "EFAIL", "GetEFAIL"],
				MatchQ[value, HoldComplete[_]],
					GetGeneralExprConstant[data, value],
					
				True,
					ThrowException[{"Unhandled Expression constant ", value}]
			];
		addConstantData[ data, "Expr"[value], info];
		id = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], info["locationId"], info["name"]];
		id
	]




GetPackedArrayConstant[data_?AssociationQ, value_] :=
	Module[ {locationId, name, info, id, initializer},
		info = data["constantData"]["lookup", "PackedArray"[value], Null];
		If[ info === Null,
			name = "PackedArrayConstant";
			locationId = AddLocalGlobal[data, GetMTensorType[data], name];
			initializer = 
				With[ {ty = GetMTensorType[data]}, 
					AddNull[#, ty]&];
			info = <| "locationId" -> locationId, "value" -> value, "initializer" -> initializer, 
		   			"creator" -> constantPackedArrayCreate, "name" -> name|>;
		   	addConstantData[ data, "PackedArray"[value], info]];
		id = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], info["locationId"], info["name"]];
		id
	]


$packedArrayInformation =
	<|
	Integer -> <|"elementType" -> 2,
				"elementTypeFunction" -> (GetMIntType[#]&), 
				"constantFunction" -> (AddConstantMInt[#1,#2]&),
				"sizeFunction" -> getMIntSize|>,
	Real -> <|"elementType" -> 3, 
				"elementTypeFunction" -> (GetMRealType[#]&),
				"constantFunction" -> (AddConstantMReal[#1,#2]&),
				"sizeFunction" -> (8&)|>,
	Complex -> <|"elementType" -> 4, 
				"elementTypeFunction" -> (GetMRealType[#]&),
				"constantFunction" -> (AddConstantMReal[#1,#2]&),
				"sizeFunction" -> (8&)|>
	|>


getMIntSize[data_] :=
	If[ data["machineIntegerSize"] === 32,
			4,
			8]

constantPackedArrayCreate[ data_?AssociationQ, value_] :=
	Module[ {flatValue = Flatten[value], head, info, rankId, dimsId, typeId, valsId, pa, paData, lenBytesId, sizeMul},	
		head = Head[First[flatValue]];
		info = Lookup[$packedArrayInformation, head, Null];	
		If[info === Null,
			ThrowException[{"Unhandled constant PackedArray", value}]];
		If[ head === Complex,
			flatValue = Flatten[ Map[{Re[#], Im[#]}&, flatValue]]];
		valsId = AddConstantArray[data, info["elementTypeFunction"], info["constantFunction"], flatValue];
		typeId = AddConstantInteger[ data, 32, info["elementType"]];
		rankId = AddConstantMInt[ data, TensorRank[value]];
		dimsId = AddConstantArray[data, GetMIntType, AddConstantMInt, Dimensions[value]];
		pa = AddRuntimeFunctionCall[data, "CreatePackedArray", {typeId, rankId, dimsId}];
		sizeMul = info["sizeFunction"][data];
		lenBytesId = AddConstantMInt[ data, Length[flatValue]*sizeMul];
		paData = AddLLVMCodeCall[data, "GetMTensorData", {pa}];
		valsId = AddTypeCast[data, valsId, GetCharStarType[data]];
		If[ data["machineIntegerSize"] === 32,
			AddLLVMCodeCall[data, "memcpyIntrinsicAligned32", {paData, valsId, lenBytesId}],
			AddLLVMCodeCall[data, "memcpyIntrinsicAligned64", {paData, valsId, lenBytesId}]];
		pa
	]


(*
  Wrap the value in the type,  so make sure the same value with different types 
  can be added.
*)
GetStringConstant[data_?AssociationQ, value_] :=
	Module[ {locationId, name, info, id, initializer},
		info = data["constantData"]["lookup", "String"[value], Null];
		If[ info === Null,
			GetExternalConstant[ data, "NewMString_UI8_MString"];
			GetExternalConstant[ data, "MObject_setGlobal"];
			name = "StringConstant";
			locationId = AddLocalGlobal[data, GetMStringType[data], name];
			initializer = 
				With[ {ty = GetMStringType[data]}, 
					AddNull[#, ty]&];
			info = <| "locationId" -> locationId, "value" -> value, "initializer" -> initializer, 
		   			"creator" -> constantMStringCreate, "name" -> name|>;
		   	addConstantData[ data, "String"[value], info]];
		id = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], info["locationId"], info["name"]];
		id
	]


constantMStringCreate[ data_?AssociationQ, value_] :=
	Module[ {valId, obj, obj1},
		valId = AddConstantString[data, value];
		obj = AddFunctionCall[ data, "NewMString_UI8_MString", {valId}];
		obj1 = AddTypeCast[data, obj, GetMObjectType[data]];
		AddFunctionCall[data, "MObject_setGlobal", {obj1}];
		obj
	]


GetCStringConstant[data_?AssociationQ, value_] :=
	Module[ {locationId, name, info, id, initializer},
		info = data["constantData"]["lookup", "CString"[value], Null];
		If[ info === Null,
			name = "CStringConstant";
			locationId = AddLocalGlobal[data, GetCharStarType[data], name];
			initializer = 
				With[ {ty = GetCharStarType[data]}, 
					AddNull[#, ty]&];
			info = <| "locationId" -> locationId, "value" -> value, "initializer" -> initializer, 
		   			"creator" -> AddConstantString, "name" -> name|>;
		   	addConstantData[ data, "CString"[value], info]];
		id = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], info["locationId"], info["name"]];
		id
	]



getExternalType[data_?AssociationQ, name_] :=
	Module[ {tyFun, res, args, ty},
		tyFun = Lookup[ data["globalFunctionTypes"], name, Null];
		If[ tyFun =!= Null,
			{res, args} = tyFun[data];
			If[ args === None,
				Return[ res]];
			ty = WrapIntegerArray[ LLVMLibraryFunction["LLVMFunctionType"][res, #, Length[args], 0]&, args];
			ty = LLVMLibraryFunction["LLVMPointerType"][ty, 0]
			,
			ty = data["getExternalFunctionType"][data, name];
			If[ ty === Null,
				ThrowException[{"Cannot find external type", name}]];
		];
		ty
	]
	



GetExternalConstant[data_?AssociationQ, name_] :=
	Module[ {locationId, info, id, initializer, index, ty, addr},
		info = data["constantData"]["lookup", "External"[name], Null];
		If[ info === Null,
			ty = getExternalType[data, name];
			locationId = AddLocalGlobal[data, ty, name];
			initializer = 
				With[ {ty1 = ty}, 
					AddNull[#, ty1]&];
					
			addr = Lookup[ data["externalData"], name, Null];
			index = Lookup[ data["externalNameIndices"], name, Null];
			If[ index === Null,
				ThrowException[{"Cannot find external name index", name}]];
			info = <| "locationId" -> locationId, "value" -> {name, index, addr, ty}, "initializer" -> initializer, 
		   			"creator" -> externalConstantCreate, "name" -> name|>;
			addConstantData[ data, "External"[name], info]];
		id = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], info["locationId"], info["name"]];
		id
	]

(*
  The address comes from the argument to the initialization.
  
  
*)	
externalConstantCreate[ data_?AssociationQ, {name_, index_, addr_, ty_}] :=
	Module[ {argId, memId, indexId, addrId, valId},
		argId = data["initializationArgumentId"]["get"];
		indexId = AddConstantInteger[data, 32, index];
		memId = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildInBoundsGEP"][data["builderId"], argId, #, 1, ""]&, {indexId}];
		addrId = LLVMLibraryFunction["LLVMBuildLoad"][ data["builderId"], memId, ""];
		valId = LLVMLibraryFunction["LLVMBuildBitCast"][data["builderId"], addrId, ty, "FunCast"];
		valId
	]

ProcessConstants[ data_?AssociationQ] :=
	Module[ {consData = data["constantDataList"]["get"]},
		Scan[ ProcessConstant[data, #]&, consData]
	]


	
(*ProcessConstants[ data_?AssociationQ] :=
	Module[ {consData},
		While[ data["constantData"]["length"] > 0,
			consData = data["constantData"]["values"];
			data["constantData"]["set", <||>];
			Scan[ ProcessConstant[data, #]&, consData];
		]
	]*)

(*
  Process all constants that have been saved up.  
  Note that the global setting needs to be initialized.
  
  Fields of the info
    locationId  
        the Id of the Global which was created,  this is loaded whenever a global is used.
    creator 
        a function which will be applied to the value to create the Id of the object
    value
        the value to use to create the object,  this might be unused by the creator function   
    initializer
        a function to get an initializer to set to the global value,  the constant processing 
        code is only called in a function,  so the globals exist before
  
  I don't think the SetInitializer is a problem here since this is 
  being set in a function. It is a problem for global initializers
  which cause problems with JIT ORC trying to load an unexplained VTable.
*)
ProcessConstant[data_?AssociationQ, info_] :=
	Module[ {name, locationId, creator, initializer, consId, value, initId},
		name = info["name"];
		locationId = info["locationId"];
		creator = info["creator"];
		initializer = info["initializer"];
		value = info["value"];
		If[ AnyTrue[{name, locationId, value, initializer}, MissingQ],
			ThrowException[{"Badly structured constant data: ", info}]];
		initId = initializer[data];
		LLVMLibraryFunction["LLVMSetInitializer"][locationId, initId];
		consId = creator[ data, value];
		LLVMLibraryFunction["LLVMBuildStore"][data["builderId"], consId, locationId];
	]


AddLocalGlobal[ data_, ty_, name_] :=
	Module[{
			id = LLVMLibraryFunction["LLVMAddGlobal"][data["moduleId"], ty, name]
		},
		AddLLVMGlobalLinkageAttribute[data, id, "LLVMInternalLinkage"];
		id
	]
	
End[]


EndPackage[]
