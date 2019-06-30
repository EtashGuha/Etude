

BeginPackage["LLVMCompileTools`ExprFunctions`"]

AddExprPart

AddExprLength

AddMIntegerToExpr

AddGetEFAIL

AddTestGetBooleanExpr

AddTestGetIntegerExpr

AddTestGetRealExpr

AddTestGetComplexExpr

AddTestGetStringExpr

AddMStringToExpr

AddReturnEFAILIfFalse

AddRawTypeQ

AddGetRawContents

AddDecrementRefCount

AddIncrementRefCount

AddRealToExpr

AddMBooleanToExpr

AddSizedMIntegerToExpr

AddMComplexToExpr

AddMComplexFromExpr

AddTestGetFunctionExpr

AddFunctionToExpr

AddTestGetVoidExpr

AddVoidToExpr

Begin["`Private`"]

Needs[ "LLVMLink`"]
Needs["LLVMCompileTools`"]
Needs["LLVMCompileTools`Types`"]
Needs["LLVMCompileTools`Basic`"]
Needs["LLVMCompileTools`Globals`"]
Needs["LLVMCompileTools`MTensor`"]
Needs["CompileUtilities`Error`Exceptions`"]
Needs["LLVMTools`"]

 
getExprContentsPosition[data_?AssociationQ] :=
	If[data["isDebug"], 7, 4]
	
(*
  Get the location of the contents of the expr cast to the appropriate expr type
  Actually this could be done by getting the contents and then casting to that, 
  not sure if there is any difference.
  
  TODO convert Normal and Integer exprs to use this.
*)	
AddExprContents[data_?AssociationQ, src_, typeFun_] :=
	Module[ {id, t1, t2, contsPos},
     	(*  cast to the appropriate Expr type *)
        id = LLVMLibraryFunction["LLVMBuildBitCast"][data["builderId"], src, typeFun[data], ""];      
 		(* Get the contents position *)
		contsPos = getExprContentsPosition[data];
		t1 = LLVMLibraryFunction["LLVMConstNull"][GetIntegerType[data, 32]];
		t2 = LLVMLibraryFunction["LLVMConstInt"][GetIntegerType[data, 32], contsPos, 0];
        id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildGEP"][data["builderId"], id, #, 2, ""]&, {t1, t2}];
  		id
	]	


	
AddExprPart[data_?AssociationQ, src_, index_Integer, checkLen_] :=
    Module[ {id, t1, t2, indexId},
		If[ data["expressionInterface"] === "Inlined",
			(*  get the NormalExpr contents *)
	     	id = AddExprContents[data, src, GetNormalExprType];
	        (*  Get the 3rd element, 
	        this should be a pointer to an expr which is really the first element of an array of exprs *)
	        t1 = LLVMLibraryFunction["LLVMConstNull"][GetIntegerType[data, 32]];
	        t2 = LLVMLibraryFunction["LLVMConstInt"][GetIntegerType[data, 32], 3, 0];
	        id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildGEP"][data["builderId"], id, #, 2, ""]&, {t1, t2}];
	        (* Get the index-th element *)
	        t2 = LLVMLibraryFunction["LLVMConstInt"][GetIntegerType[data, 64], index, 0];
	        id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildGEP"][data["builderId"], id, #, 1, ""]&, {t2}];
	        (* Load this into result *)
	        id = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], id, ""];
	        (* Cast back to an Expr *)
	        id = LLVMLibraryFunction["LLVMBuildBitCast"][data["builderId"], id, GetBaseExprType[data], ""];
	        ,
	        indexId = AddConstantMInt[data, index];
	        id = AddFunctionCall[ data, "Part_E_I_E", {src, indexId}]];
        id
    ]

AddExprLength[data_?AssociationQ, _, {src_}] :=
	Module[ {id, t1, t2},
		If[ data["expressionInterface"] === "Inlined",
			(*  get the NormalExpr contents *)
	     	id = AddExprContents[data, src, GetNormalExprType];
			(* Get the 2nd element, this should be a pointer to the length field *)
			t1 = LLVMLibraryFunction["LLVMConstNull"][GetIntegerType[data, 32]];
			t2 = LLVMLibraryFunction["LLVMConstInt"][GetIntegerType[data, 32], 2, 0];
	        id = WrapIntegerArray[LLVMLibraryFunction["LLVMBuildGEP"][data["builderId"], id, #, 2, ""]&, {t1, t2}];
	        (* load this into memory *)
			id = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], id, ""];
			,
			id = AddFunctionCall[ data, "Length_E_I", {src}]];
		id
	]
 

 
AddReturnEFAILIfFalse[data_?AssociationQ, src_, resId_] :=
	Module[ {retBB, mainBB, id},
		retBB = LLVMLibraryFunction["LLVMAppendBasicBlockInContext"][data["contextId"], data["functionId"]["get"], data["getBBName"]["returnError"]];
		mainBB = LLVMLibraryFunction["LLVMAppendBasicBlockInContext"][data["contextId"], data["functionId"]["get"], data["getBBName"]["main"]];
        id = LLVMLibraryFunction["LLVMBuildCondBr"][data["builderId"], src, mainBB, retBB];
        LLVMLibraryFunction["LLVMPositionBuilderAtEnd"][data["builderId"], retBB];
        id = AddGetEFAIL[data];
        LLVMLibraryFunction["LLVMBuildStore"][data["builderId"], id, resId];
        id = AddConstantInteger[data, 32,0];
        LLVMLibraryFunction["LLVMBuildRet"][data["builderId"], id];
        LLVMLibraryFunction["LLVMPositionBuilderAtEnd"][data["builderId"], mainBB];
        data["updateBasicBlockMap"][data, mainBB];		
	]


AddGetEFAIL[data_?AssociationQ] :=
	Module[ {id},
		id = AddFunctionCall[data, "GetEFAIL", {}];
		id
	]

(*
 Should be in data, but set to match names/values in exprtypes.h
 Could be queried from Kernel.
*)
$exprTypes = <|
   "TMINTEGER" -> 0,
   "TINTEGER" -> 1,
   "TMREAL" -> 2,
   "TREAL" -> 3,
   "TRATIONAL" -> 4,
   "TCOMPLEX" -> 5,
   "TNORMAL" -> 6,
   "TSYMBOL" -> 7,
   "TSTRING" -> 8,
   "TRAW" -> 9
|>

AddExprTypeQ[data_?AssociationQ, src_, type_String] :=
	Module[ {id, t1, eqOp, typeVal},
		typeVal = $exprTypes[type];
		id = AddExprType[data, Null, {src}];
		t1 = LLVMLibraryFunction["LLVMConstInt"][GetIntegerType[data, 16], typeVal, 0];
		eqOp = data["LLVMIntPredicate"][SameQ];
		id = LLVMLibraryFunction["LLVMBuildICmp"][data["builderId"], eqOp, id, t1, ""];
		id
		
	]

AddCodeFunction["Expr`Type", AddExprType]

AddExprType[data_?AssociationQ, _, {src_}] :=
	Module[ {id, t1},
		id = AddGetFlags[data,src];
        t1 = LLVMLibraryFunction["LLVMConstInt"][GetIntegerType[data, 16], 15, 0];
		id = LLVMLibraryFunction["LLVMBuildAnd"][data["builderId"], id, t1, ""];
		id
	]


(*
  Return the flags
*)
AddGetFlags[data_?AssociationQ, src_] :=
	Module[ {t1, t2, id},

		If[ data["expressionInterface"] === "Inlined",

	     	(*  get the address of the flags *)
	        t1 = LLVMLibraryFunction["LLVMConstNull"][GetIntegerType[data, 32]];
	        t2 = LLVMLibraryFunction["LLVMConstInt"][GetIntegerType[data, 32], 1, 0];
	        id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildGEP"][data["builderId"], src, #, 2, ""]&, {t1, t2}];
	        (* Load the flags into memory *)
	        id = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], id, ""];
	        ,
	        id = AddFunctionCall[ data, "Flags_E_UI16", {src}]
	        ];
		id
	]

(*
  Return the RefCount Address
*)
AddGetRefCountAddress[data_?AssociationQ, src_] :=
	Module[ {t1, id},
     	(*  get the address of the refcount *)
        t1 = LLVMLibraryFunction["LLVMConstNull"][GetIntegerType[data, 32]];
        id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildGEP"][data["builderId"], src, #, 2, "refcntaddr"]&, {t1, t1}];
		id
	]

(*
 Boolean Expr Functions
*)	

AddMBooleanToExpr[data_?AssociationQ, ty_, src_] :=
	Module[ {trueId, trueBB, falseId, falseBB, phiId},
		{trueId, trueBB, falseId, falseBB} = 
			AddIfElseBasicBlocks[data, src,
				GetExternalConstant[#, "eTrue"]&, 
				GetExternalConstant[#, "eFalse"]&];
		phiId = LLVMLibraryFunction["LLVMBuildPhi"][ data["builderId"], GetBaseExprType[data],  ""];
		WrapIntegerArray[LLVMLibraryFunction["LLVMAddIncoming"][ phiId, #1, #2, 2]&, {trueId, falseId}, {trueBB, falseBB}];
		phiId
	]

AddCodeFunction["ExprToBoolean", AddTestGetBooleanExpr[ #1, "Boolean", First[#3], Last[#3]]&]

AddTestGetBooleanExpr[data_, ty_, ref_, src_] :=
	Module[ {trueExpr, falseExpr, trueId, falseId, test, off},
		trueExpr = GetExternalConstant[data, "eTrue"];
		falseExpr = GetExternalConstant[data, "eFalse"];
		trueId = AddSameQ[ data, src, trueExpr];
		falseId = AddSameQ[ data, src, falseExpr];
		off = AddConstantInteger[data, 64, 0];
		AddSetArray[ data, ref, off, trueId];
		test = LLVMLibraryFunction["LLVMBuildOr"][data["builderId"], trueId, falseId, ""];
		test
	]


(*
 Integer Expr Functions
*)	


$integerTypeData =
<|
	"UnsignedInteger8" -> {8, False},
	"Integer8" -> {8, True},
	"UnsignedInteger16" -> {16, False},
	"Integer16" -> {16, True},
	"UnsignedInteger32" -> {32, False},
	"Integer32" -> {32, True},
	"UnsignedInteger64" -> {64, False},
	"Integer64" -> {64, True}
|>


(*
  Called from CreateWrapper and Casting from an Expression
*)
AddTestGetIntegerExpr[data_, ty_, ref_, src_] :=
	Module[ {size, sizeVal, signed, test, id, len, eref, id1, signedArg, comp, eqOp},
		If[ !KeyExistsQ[$integerTypeData, ty],
			ThrowException[{"Type not handled ", ty}]];
		{size, signed} = $integerTypeData[ty];
		sizeVal = AddConstantInteger[data, 32, size];
		len = LLVMLibraryFunction["LLVMConstInt"][GetIntegerType[data, 32], 1, 0];
		eref = LLVMLibraryFunction["LLVMBuildArrayAlloca"][data["builderId"], GetIntegerType[data, 64], len, ""];
		signedArg = AddConstantInteger[data, 32, If[signed,1,0]];
		id = AddFunctionCall[ data, "TestGet_Integer", {src, sizeVal, signedArg, eref}];
		id1 = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], eref, ""];
		(*
		 Now truncate the result and put into ref
		*)
		id1 = ConvertInteger[data, id1, size, signed];
		LLVMLibraryFunction["LLVMBuildStore"][data["builderId"], id1, ref];
		comp = AddConstantInteger[data, 16, 1];
		eqOp = data["LLVMIntPredicate"][SameQ];
		test = LLVMLibraryFunction["LLVMBuildICmp"][data["builderId"], eqOp, id, comp, ""];
		test
	]



(*
 Sized Integer Expr Functions
*)	

AddCodeFunction["Integer32ToExpr", AddSizedMIntegerToExpr[32, True]]
AddCodeFunction["Integer64ToExpr", AddSizedMIntegerToExpr[64, True]]

AddMIntegerToExpr[data_?AssociationQ, ty_, src_] :=
	Module[ {size, signed, id},
		If[ !KeyExistsQ[$integerTypeData, ty],
			ThrowException[{"Type not handled ", ty}]];
		{size, signed} = $integerTypeData[ty];
		id = AddSizedMIntegerToExpr[ size, signed][data, Null, {src}];
		id
	]


AddSizedMIntegerToExpr[size_, signedQ_][data_?AssociationQ, _, {src_}] :=
	Module[ {id, fun, argTy, argSize, sizeId = AddConstantInteger[data, 32, size], signedId = AddConstantInteger[data, 32, If[signedQ,1,0]]},
		argTy = GetIntegerType[data, 64];
		argSize = LLVMLibraryFunction["LLVMGetIntTypeWidth"][argTy];
		Which[
			argSize == size,
			(* do nothing *)
			id = src
			,
			argSize < size,
			id = LLVMLibraryFunction["LLVMBuildTrunc"][data["builderId"], src, GetIntegerType[data, 64], ""];
			,
			argSize > size,
			fun = If[ signedQ, LLVMLibraryFunction["LLVMBuildSExt"], LLVMLibraryFunction["LLVMBuildZExt"]];
			id = fun[data["builderId"], src,  GetIntegerType[data, argSize], ""];
		];
		id = AddFunctionCall[data, "CreateMIntegerExpr", {id, sizeId, signedId}];
		id
	]
	
	
AddCodeFunction["ExprToInteger32", AddTestGetIntegerExpr[ #1, "Integer32", First[#3], Last[#3]]&]
AddCodeFunction["ExprToInteger64", AddTestGetIntegerExpr[ #1, "Integer64", First[#3], Last[#3]]&]
	

(*
  Convert the integer id to size/signed
*)
ConvertInteger[ data_, id_, size_, signed_] :=
	Module[ {ty, width, id1},
		ty = LLVMLibraryFunction["LLVMTypeOf"][id];
        width = LLVMLibraryFunction["LLVMGetIntTypeWidth"][ty];
        Which[
        	size === width,
			(* Do nothing *)
        	id1 = id;
        	,
        	size < width,
        	id1 = LLVMLibraryFunction["LLVMBuildTrunc"][data["builderId"], id, GetIntegerType[data, size], ""];
        	,
        	size > width && signed,
        	id1 = LLVMLibraryFunction["LLVMBuildSExt"][data["builderId"], id, GetIntegerType[data, size], ""];
			,
        	size > width && !signed,
        	id1 = LLVMLibraryFunction["LLVMBuildZExt"][data["builderId"], id, GetIntegerType[data, size], ""];
        ];
		id1
	]


(*
 Real Expr Functions
*)

AddCodeFunction["ExprToReal16", AddTestGetRealExpr[#1, "Real16", Part[#3,1], Part[#3,2]]&]
AddCodeFunction["ExprToReal32", AddTestGetRealExpr[#1, "Real32", Part[#3,1], Part[#3,2]]&]
AddCodeFunction["ExprToReal64", AddTestGetRealExpr[#1, "Real64", Part[#3,1], Part[#3,2]]&]
AddCodeFunction["ExprToReal128", AddTestGetRealExpr[#1, "Real128", Part[#3,1], Part[#3,2]]&]


$realTypeData =
<|
    "Real16" -> 16,
    "Real32" -> 32,
    "Real64" -> 64,
    "Real128" -> 128
|>

(*
  Deals with Real64 and Real32 could easily deal with Real16
  Used from CreateWrapper and Casting from Expression
*)

AddTestGetRealExpr[data_, ty_, ref_, src_] :=
	Module[ {size, sizeVal, test, id, len, eref, id1, comp, eqOp},
		If[ !KeyExistsQ[$realTypeData, ty],
			ThrowException[{"Type not handled ", ty}]];
		size = $realTypeData[ty];
		sizeVal = AddConstantInteger[data, 32, size];
		len = LLVMLibraryFunction["LLVMConstInt"][GetIntegerType[data, 32], 1, 0];
		eref = LLVMLibraryFunction["LLVMBuildArrayAlloca"][data["builderId"], GetRealType[data, 64], len, ""];
		id = AddFunctionCall[ data, "TestGet_Float", {src, sizeVal, eref}];
		id1 = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], eref, ""];
		(*
		 Now truncate the result and put into ref
		*)
		id1 = ConvertReal[data, id1, 64, size];
		LLVMLibraryFunction["LLVMBuildStore"][data["builderId"], id1, ref];
		comp = AddConstantInteger[data, 16, 1];
		eqOp = data["LLVMIntPredicate"][SameQ];
		test = LLVMLibraryFunction["LLVMBuildICmp"][data["builderId"], eqOp, id, comp, ""];
		test
	]


AddRealQ[data_?AssociationQ, _, {src_}] :=
	Module[ {},
		AddExprTypeQ[data,src,"TMREAL"]
	]

kindToName[data_, ty_, kind_] :=
    Which[
        kind === LLVMEnumeration["LLVMTypeKind", "LLVMPointerTypeKind"],
            Module[{
                elementType,
                elementKind
            },
                elementType = LLVMLibraryFunction["LLVMGetElementType"][ty];
                elementKind = LLVMLibraryFunction["LLVMGetTypeKind"][elementType];
                kindToName[data, elementType, elementKind]
            ],
        kind === LLVMEnumeration["LLVMTypeKind", "LLVMHalfTypeKind"],
            "Real16",
        kind === LLVMEnumeration["LLVMTypeKind", "LLVMFloatTypeKind"],
            "Real32",
        kind === LLVMEnumeration["LLVMTypeKind", "LLVMDoubleTypeKind"],
            "Real64",
        kind === LLVMEnumeration["LLVMTypeKind", "LLVMFP128TypeKind"],
            "Real128",
        True,
            ThrowException[{"Real type not handled [cannot get type name] ", ty, kind}]
    ]

GetRealTypeName[data_, id_] :=
    With[{
        ty = LLVMLibraryFunction["LLVMTypeOf"][id]
    },
    With[{
        kind = LLVMLibraryFunction["LLVMGetTypeKind"][ty]
    },
        kindToName[data, ty, kind]
    ]];
    
GetRealTypeWidth[data_, id_] :=
    $realTypeData[GetRealTypeName[data, id]];

RealPointerToDoublePointer[data_, _, src_] :=
    Module[{
        alloca,
        srcVal,
        trgtVal,
        doubleTy,
        srcTy
    },
        If[GetRealTypeName[data, src] === "Real64",
            Return[src]
        ];
        doubleTy = GetDoubleType[data];
        srcTy = LLVMLibraryFunction["LLVMTypeOf"][src];
        srcVal = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], src, ""];
        alloca = LLVMLibraryFunction["LLVMBuildAlloca"][
            data["builderId"],
            doubleTy,
            ""
        ];
        trgtVal = RealToDouble[data, Null, srcVal];
        LLVMLibraryFunction["LLVMBuildStore"][
            data["builderId"],
            trgtVal,
            alloca,
            ""
        ];
        alloca
    ]
    
    
(*
  Convert the real id (sizeIn) to sizeOut
  TODO,  clean functions and switch to this
*)
ConvertReal[ data_, id_, sizeIn_, sizeOut_] :=
	Module[ {id1},
        Which[
        	sizeIn === sizeOut,
			(* Do nothing *)
        	id1 = id;
        	,
        	sizeIn < sizeOut,
        	id1 = LLVMLibraryFunction["LLVMBuildFPExt"][data["builderId"], id, GetRealType[data, sizeOut], ""];
        	,
        	True,
        	id1 = LLVMLibraryFunction["LLVMBuildFPTrunc"][data["builderId"], id, GetRealType[data, sizeOut], ""];
        ];
		id1
	]
    
    
RealToDouble[data_, _, src_] :=
    With[{size = GetRealTypeWidth[data, src]},
        toMReal[size][data, Null, src]
    ];
    
toMReal[size_][data_?AssociationQ, ty_, {src_}] :=
    toMReal[size][data, ty, src]
toMReal[size_][data_?AssociationQ, _, src_] :=
    Which[
	    size === 64,
	       (* Nothing *)
	       src,
	    size < 64,
	       LLVMLibraryFunction["LLVMBuildFPExt"][data["builderId"], src, GetMRealType[data], "fpext"],
	    size > 64,
	       LLVMLibraryFunction["LLVMBuildFPTrunc"][data["builderId"], src, GetMRealType[data], "fptrunc"], 
        True,
            ThrowException[{"Real type not handled [cannot convert to mreal] ", ty}]
	];
    
AddCodeFunction["Real16ToExpr", AddRealToExpr["Real16"]]
AddCodeFunction["Real32ToExpr", AddRealToExpr["Real32"]]
AddCodeFunction["Real64ToExpr", AddRealToExpr["Real64"]]
AddCodeFunction["Real128ToExpr", AddRealToExpr["Real64"]]

AddRealToExpr[ty_][data_, _, src_] :=
    AddRealToExpr[ty][data, ty, src]
    
AddRealToExpr[ty_][data_, _, src_] :=
    Module[ {size},
        If[ !KeyExistsQ[$realTypeData, ty],
            ThrowException[{"Real type not handled ", ty}]
        ];
        size = $realTypeData[ty];
        iAddRealToExpr[ size][data, ty, src]
    ]
    
iAddRealToExpr[size_][data_?AssociationQ, ty_, src0_] :=
	Module[ {src},
	    src = toMReal[size][data, ty, src0];
		AddFunctionCall[data, "CreateMRealExpr", {src}]
	]


(*
 Complex conversions
*)


AddCodeFunction["ExprToReal64ReIm", TestGetExprToReIm]

TestGetExprToReIm[data_, _, {refRe_, refIm_, src_}] :=
	Module[ {test, sizeVal, id, comp, eqOp},
		sizeVal = AddConstantInteger[data, 32, 64];
		id = AddFunctionCall[ data, "TestGet_ComplexFloat", {src, sizeVal, refRe, refIm}];
		comp = AddConstantInteger[data, 16, 1];
		eqOp = data["LLVMIntPredicate"][SameQ];
		test = LLVMLibraryFunction["LLVMBuildICmp"][data["builderId"], eqOp, id, comp, ""];
		test
	]


(*
  Used from Complex
*)



AddMRealFromExpr[data_?AssociationQ, _, {src_}] :=
	Module[ {id, t1, t2},
		(*  get the MInteger Expr contents *)
     	id = AddExprContents[data, src, GetMRealExprType];
		(* Get the 1st element, this should be a pointer to the integer data *)
       	t1 = LLVMLibraryFunction["LLVMConstNull"][GetIntegerType[data, 32]];
       	t2 = LLVMLibraryFunction["LLVMConstNull"][GetIntegerType[data, 32]];
		id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildGEP"][data["builderId"], id, #, 2, ""]&, {t1, t2}];
        (* load this into memory *)
		id = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], id, ""];
		id
	]

(*
 Should revisit this to use the TestGet_ComplexFloat functionality used by Casts
*)
AddTestGetComplexExpr[data_, ty_, succRef_, src_] :=
	Module[ {test, eTrue, bbTrue},
		test = AddMComplexQ[data, Null, {src}];
		{eTrue, bbTrue} = 
			AddIfTrueBasicBlock[ data, test,
				Function[{data1, bbFalse},
					Module[ {id, off},
						id = AddMComplexFromExpr[data1, Null, {src}];
						off = AddConstantInteger[data1, 64, 0];
						AddSetArray[ data1, succRef, off, id];
						LLVMLibraryFunction["LLVMBuildBr"][data1["builderId"], bbFalse];
					]]];
		test
	]



AddGetComplexElement[ data_?AssociationQ, src_, elem_] :=
	Module[ {id, t1, t2, idElems, idElem},
		id = AddExprContents[data, src, GetComplexExprType];
        t1 = LLVMLibraryFunction["LLVMConstNull"][GetIntegerType[data, 32]];
        t2 = LLVMLibraryFunction["LLVMConstInt"][GetIntegerType[data, 32], If[elem === Re,0,1], 0];
        idElems = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildGEP"][data["builderId"], id, #, 2, ""]&, {t1, t2}];
		idElem = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], idElems, ""];
		idElem
	]



AddMComplexQ[data_?AssociationQ, _, {src_}] :=
	Module[ {test, efT, efF, bbTrue, idRe, idIm, idRes, initBB,  phiId},
		test = AddExprTypeQ[data,src,"TCOMPLEX"];
		initBB = data["getCurrentBasicBlock"][data];
		{efT, bbTrue} = 
			AddIfTrueBasicBlock[ data, test,
				Function[{data1, bbFalse},
					idRe = AddGetComplexElement[data1, src, Re];
 					idIm = AddGetComplexElement[data1, src, Im];
 					idRe = AddRealQ[data1, Null, {idRe}];
 					idIm = AddRealQ[data1, Null, {idIm}];
					idRes = LLVMLibraryFunction["LLVMBuildAnd"][data1["builderId"], idRe, idIm, "reimReal"];
					LLVMLibraryFunction["LLVMBuildBr"][data1["builderId"], bbFalse];
					idRes
					]];
		phiId = LLVMLibraryFunction["LLVMBuildPhi"][ data["builderId"], GetBooleanType[data], ""];
		efF = AddConstantBoolean[data, False];
		WrapIntegerArray[LLVMLibraryFunction["LLVMAddIncoming"][ phiId, #1, #2, 2]&, {efT, efF}, {bbTrue, initBB}];
		phiId
	]


 
AddMComplexToExpr[data_?AssociationQ, ty_, src_] :=
	Module[ {idRe, idIm, id},
		idRe = AddExtractElement[data, Null, {src, AddConstantInteger[data, 32,0]}];
		idIm = AddExtractElement[data, Null, {src, AddConstantInteger[data, 32,1]}];
		id = AddFunctionCall[data, "CreateComplex_RR_E", {idRe, idIm}];
		id
	]


AddMComplexFromExpr[data_?AssociationQ, _, {src_}] :=
	Module[ {idRe, idIm, idComp},
		idRe = AddGetComplexElement[data, src, Re];
		idIm = AddGetComplexElement[data, src, Im];
		idRe = AddMRealFromExpr[data, Null, {idRe}];
		idIm = AddMRealFromExpr[data, Null, {idIm}];
		idComp = AddNull[data, GetMComplexType[data]];
		idComp = AddInsertElement[data, idComp, idRe, AddConstantInteger[data, 32,0]];
		idComp = AddInsertElement[data, idComp, idIm, AddConstantInteger[data, 32,1]];
		idComp
	]
	

(*
 String Expr Functions
*)	
AddStringQ[data_?AssociationQ, _, {src_}] :=
	Module[ {},
		AddExprTypeQ[data,src,"TSTRING"]
	]


AddCodeFunction["StringToExpr", AddStringToExpr]

AddStringToExpr[data_, _, {src_}] :=
	AddMStringToExpr[data, "String", src]

AddMStringToExpr[data_?AssociationQ, ty_, src_] :=
	Module[ {id},
		id = AddFunctionCall[data, "MString_toExpr", {src}];
		id
	]


AddCodeFunction["ExprToString", AddTestGetStringExpr[ #1, "String", Part[#3,1], Part[#3,2]]&]

AddTestGetStringExpr[data_, ty_, ref_, src_] :=
	Module[ {test, eTrue, bbTrue, id, off},
		test = AddStringQ[data, Null, {src}];
		{eTrue, bbTrue} = 
			AddIfTrueBasicBlock[ data, test,
				Function[{data1, bbFalse},
					id = AddFunctionCall[data1, "MString_fromExpr", {src}];
					off = AddConstantInteger[data1, 64, 0];
					AddSetArray[ data1, ref, off, id];
					LLVMLibraryFunction["LLVMBuildBr"][data1["builderId"], bbFalse];
					]];
		test
	]


(*
 Should be converted to a test function, which throws an exception.
*)
AddCodeFunction["StringFromExpr", AddMStringFromExpr]

AddMStringFromExpr[data_?AssociationQ, _, {src_}] :=
	Module[ {id},
		id = AddFunctionCall[data, "MString_fromExpr", {src}];
		id
	]

(*
  This should be reworked to call the Kernel directly
*)	
AddCodeFunction["ExprToCString", AddExprToCString]

AddExprToCString[data_?AssociationQ, _, {src_}] :=
	Module[ {id, t1, t2},
		(*  get the String Expr contents *)
     	id = AddExprContents[data, src, GetExprStringType];	
		(* Get the element at index 1, this should be an array of chars *)
        t1 = LLVMLibraryFunction["LLVMConstNull"][GetIntegerType[data, 32]];
        t2 = LLVMLibraryFunction["LLVMConstInt"][GetIntegerType[data, 32], 1, 0];
		id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildGEP"][data["builderId"], id, #, 2, ""]&, {t1, t2}];
        (* Cast from the address of the start of the array -used in the Expr type- to a char* *)
		id = AddTypeCast[data, id, GetCharStarType[data]];
		id
	]

AddCodeFunction["StringExprToLength", AddStringExprToLength]

AddStringExprToLength[data_?AssociationQ, _, {src_}] :=
	Module[ {id, t1, t2},
		(*  get the String Expr contents *)
     	id = AddExprContents[data, src, GetExprStringType];
		(* Get the element at index 4, this should be a pointer to the char data *)
        t1 = LLVMLibraryFunction["LLVMConstNull"][GetIntegerType[data, 32]];
        t2 = LLVMLibraryFunction["LLVMConstInt"][GetIntegerType[data, 32], 1, 0];
		id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildGEP"][data["builderId"], id, #, 2, ""]&, {t1, t2}];
        (* load this into memory *)
		id = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], id, ""];
		id
	]


(*
 Raw Exprs
*)

AddCodeFunction["Expr`RawType", AddRawType]

AddRawType[data_?AssociationQ, _, {src_}] :=
	Module[{id, t1, t2},
     	id = AddExprContents[data, src, GetRawExprType];
        (* Get the raw type *)
        t1 = LLVMLibraryFunction["LLVMConstNull"][GetIntegerType[data, 32]];
        t2 = LLVMLibraryFunction["LLVMConstInt"][GetIntegerType[data, 32], 3, 0];
        id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildGEP"][data["builderId"], id, #, 2, ""]&, {t1, t2}];
        (* load this into memory *)
		id = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], id, ""];
		id
	]

$rawTypes =
<|
  "RPACKED" -> 1,
  "RARRAY" -> 4,
  "ROBJECTINSTANCE" -> 26
|>

(*
 Perhaps put the And into another function.
 TODO use a BB to get lazy And.
*)
AddRawTypeQ[data_?AssociationQ, src_, type_String] :=
	Module[ {rawQId, id, t1, t2, eqOp, typeNum},
		rawQId = AddExprTypeQ[data, src, "TRAW"];
     	(*  get the RawExpr contents *)
     	id = AddExprContents[data, src, GetRawExprType];

        (* Get the raw type *)
        t1 = LLVMLibraryFunction["LLVMConstNull"][GetIntegerType[data, 32]];
        t2 = LLVMLibraryFunction["LLVMConstInt"][GetIntegerType[data, 32], 3, 0];
        id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildGEP"][data["builderId"], id, #, 2, ""]&, {t1, t2}];
        (* load this into memory *)
		id = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], id, ""];
		(*
		  Compare argument type with raw type
		*)
		typeNum = $rawTypes[type];
		t1 = LLVMLibraryFunction["LLVMConstInt"][GetIntegerType[data, 16], typeNum, 0];
		eqOp = data["LLVMIntPredicate"][SameQ];
		id = LLVMLibraryFunction["LLVMBuildICmp"][data["builderId"], eqOp, id, t1, ""];
		id = LLVMLibraryFunction["LLVMBuildAnd"][data["builderId"], rawQId, id, ""];
		id
	]


AddCodeFunction["Expr`RawContents", AddGetRawContents[#1, First[#3]]&]

AddGetRawContents[data_?AssociationQ, src_] :=
	Which[
		TrueQ[data["expressionVersion"] >= 2],
			AddGetRawContents2[data, src],
		True,
			AddGetRawContentsBase[data, src]
	]


AddGetRawContents2[data_?AssociationQ, src_] :=
	Module[ {id, t1, t2},
     	(*  get the RawExpr contents *)
     	id = AddExprContents[data, src, GetRawExprType];
        (* Get the raw contents (data) *)
        t1 = LLVMLibraryFunction["LLVMConstNull"][GetIntegerType[data, 32]];
        t2 = LLVMLibraryFunction["LLVMConstInt"][GetIntegerType[data, 32], 1, 0];
        id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildGEP"][data["builderId"], id, #, 2, ""]&, {t1, t2}];
        (* load this into memory *)
		id = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], id, ""];
		id
	]	


AddGetRawContentsBase[data_?AssociationQ, src_] :=
	Module[ {id, t1, t2},
     	(*  get the RawExpr contents *)
     	id = AddExprContents[data, src, GetRawExprType];
        (* Get the raw contents (data) *)
        t1 = LLVMLibraryFunction["LLVMConstNull"][GetIntegerType[data, 32]];
        t2 = LLVMLibraryFunction["LLVMConstInt"][GetIntegerType[data, 32], 5, 0];
        id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildGEP"][data["builderId"], id, #, 2, ""]&, {t1, t2}];
        (* load this into memory *)
		id = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], id, ""];
		id
	]	

(*
 Need to get the address of the expr argument and pass that 
 to the DecrementRefCount function.
 This is a rather inefficient way since it creates a stack allocation to get the address.
*)
AddDecrementRefCountOld[data_?AssociationQ, src_] :=
	Module[ {len, eref, ind, id},
		len = LLVMLibraryFunction["LLVMConstInt"][GetIntegerType[data, 32], 1, 0];
		eref = LLVMLibraryFunction["LLVMBuildArrayAlloca"][data["builderId"], GetBaseExprType[data], len, ""];
		ind = LLVMLibraryFunction["LLVMConstNull"][GetIntegerType[data, 32]];
		id = WrapIntegerArray[ LLVMLibraryFunction["LLVMBuildGEP"][data["builderId"], eref, #, 1, ""]&, {ind}];
		id = LLVMLibraryFunction["LLVMBuildStore"][data["builderId"], src, eref];
		id = AddFunctionCall[data, "Runtime_DecrementRefCount", {eref}];
        id
	]

(*
 Decrement the expr refcount.  If this goes to 0 or less than 0 call FreeExpr.
*)
AddCodeFunction["DecrementReferenceCount", AddDecrementRefCount]

AddDecrementRefCount[data_?AssociationQ, _, {exprId_}] :=
	Module[ {refAddrId, refId, zeroId, condId, lessEqOp},
		refAddrId = AddGetRefCountAddress[data,exprId];
		refId = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], refAddrId, "refcnt"];
		refId = LLVMLibraryFunction["LLVMBuildAdd"][ data["builderId"], refId, AddConstantInteger[data, 32, -1], "refcnt"];
		LLVMLibraryFunction["LLVMBuildStore"][data["builderId"], refId, refAddrId];
		lessEqOp = data["LLVMIntPredicate"][LessEqual];
		zeroId = LLVMLibraryFunction["LLVMConstNull"][GetIntegerType[data, 32]];
		condId = LLVMLibraryFunction["LLVMBuildICmp"][data["builderId"], lessEqOp, refId, zeroId, "freecond"];
		AddIfTrueBasicBlock[ data, condId, AddCallFreeExpr[#1, #2, exprId]&];
        refId
	]


(*
 Increment the expr refcount.  TODO,  what about wraparound check?
*)

AddCodeFunction["IncrementReferenceCount", AddIncrementRefCount]

AddIncrementRefCount[data_?AssociationQ, _, {exprId_}] :=
	Module[ {refAddrId, refId},
		refAddrId = AddGetRefCountAddress[data,exprId];
		refId = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], refAddrId, "refcnt"];
		refId = LLVMLibraryFunction["LLVMBuildAdd"][ data["builderId"], refId, AddConstantInteger[data, 32, 1], "refcnt"];
		LLVMLibraryFunction["LLVMBuildStore"][data["builderId"], refId, refAddrId];
        refId
	]


(*
  Come here if the refcount is less than or equal to zero.  
  In this case we call FreeExpr on the expr and join the falseBB.
*)
AddCallFreeExpr[data_?AssociationQ, falseBB_, exprId_] :=
	Module[{id},
		id = AddFunctionCall[data, "Runtime_FreeExpr", {exprId}, "freeExprRes"];
		LLVMLibraryFunction["LLVMBuildBr"][data["builderId"], falseBB];
	]




(*
  Test and get a FunctionExpr
*)
AddTestGetFunctionExpr[data_, funTy_, ref_, src_] :=
	Module[ {tyString, stringId, len, eref, erefCast, id, comp, eqOp, test, tyId, funId},
		tyString = GetTypeString[data, funTy];
		stringId = GetCStringConstant[data, tyString];
		tyId = GetLLVMType[data, funTy];
		len = LLVMLibraryFunction["LLVMConstInt"][GetIntegerType[data, 32], 1, 0];
		
		eref = LLVMLibraryFunction["LLVMBuildArrayAlloca"][data["builderId"], GetVoidHandleType[data], len, ""];
		
		id = AddFunctionCall[ data, "TestGet_Function", {src, stringId, eref}];
	
		erefCast = LLVMLibraryFunction["LLVMBuildBitCast"][data["builderId"], eref, GetHandleType[data, tyId], ""];
		funId = LLVMLibraryFunction["LLVMBuildLoad"][data["builderId"], erefCast, ""];
		LLVMLibraryFunction["LLVMBuildStore"][data["builderId"], funId, ref];
		comp = AddConstantInteger[data, 16, 1];
		eqOp = data["LLVMIntPredicate"][SameQ];
		test = LLVMLibraryFunction["LLVMBuildICmp"][data["builderId"], eqOp, id, comp, ""];
		test
	]

GetTypeString[data_, ty_] :=
	ToString[ty, InputForm]


AddFunctionToExpr[data_, ty_, id_] :=
	Module[{},
		ThrowException[{"Returning a functions from a CompiledCodeFunction is not currently supported."}]
	]

(*
  Test and get a Void argument.  These are not supported so just throw an exception.
*)
AddTestGetVoidExpr[data_, ty_, ref_, src_] :=
	Module[{},
		ThrowException[{"Void is not handled as an input argument to a CompiledCodeFunction."}]
	]


(*
  Should return Null
*)
AddVoidToExpr[data_, ty_, _] :=
	Module[{},
		GetExternalConstant[data, "eNull"]
	]


End[]


EndPackage[]
