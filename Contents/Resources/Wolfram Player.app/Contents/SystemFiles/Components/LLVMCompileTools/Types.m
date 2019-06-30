

BeginPackage["LLVMCompileTools`Types`"]

GetIntegerType::usage = "GetIntegerType  "

GetUnsignedIntegerType

GetWrappedExprType::usage = "GetWrappedExprType  "

GetBaseExprType::usage = "GetBaseExprType  "

GetNormalExprType::usage = "GetNormalExprType  "

GetMIntegerExprType::usage = "GetMIntegerExprType  "

GetComplexExprType


GetStringType

GetRawExprType::usage = "GetRawExprType  "

GetPackedArrayBaseType::usage = "GetPackedArrayBaseType  "

GetPackedArrayType::usage = "GetPackedArrayType  "

GetMTensorType

GetMNumericArrayType

GetCompilerErrorType

GetMStringType::usage = "GetMStringType  "

GetCUDABaseArrayType::usage = "GetCUDABaseArrayType  "

GetMTensorPointerType::usage = "GetMTensorPointerType  "

GetMIntPointerType::usage = "GetMIntPointerType  "

GetMRealPointerType

GetMIntType::usage = "GetMIntType  "

GetRealType

GetMRealType::usage = "GetMRealType  "

GetMBoolType::usage = "GetMBoolType "

GetMComplexType

GetVectorComplexType

GetComplexType

GetVectorType

GetStructType

GetCharStarType

GetMTensorPropertiesType::usage = "GetMTensorPropertiesType  "

GetBooleanType::usage = "GetBooleanType  "

GetDoubleType

GetPointerType

GetHandleType

(*
GetAddressType
*)

GetMStringType

GetMObjectType

GetVoidType

GetVoidPointerType

GetVoidHandleType

GetMRealExprType::usage = "GetMRealExprType  "

GetStructureType

GetLLVMType

GetUTF8StringType

GetBaseFunctionType

GetOpaqueStructureType

GetExprStringType

Getutf8strType

Begin["`Private`"]

Needs["LLVMLink`"]
Needs["CompileUtilities`Error`Exceptions`"]
Needs["LLVMTools`"]


GetBooleanType[data_?AssociationQ] :=
	GetIntegerType[data, 1]

GetIntegerType[data_?AssociationQ, size_] :=
    Module[ {fun},
        fun =
         Switch[size,
              1, LLVMLibraryFunction["LLVMInt1TypeInContext"],
              8, LLVMLibraryFunction["LLVMInt8TypeInContext"],
              16, LLVMLibraryFunction["LLVMInt16TypeInContext"],
              32, LLVMLibraryFunction["LLVMInt32TypeInContext"],
              64, LLVMLibraryFunction["LLVMInt64TypeInContext"],
              128, LLVMLibraryFunction["LLVMInt128TypeInContext"],
              _,   ThrowException[{"Unsupported integer type", size}]
         ];
        fun[data["contextId"]]
    ]
GetIntegerPointerType[data_?AssociationQ, size_] :=
	LLVMLibraryFunction["LLVMPointerType"][GetIntegerType[data, size], 0]
	
GetRealType[data_, size_] :=
   Module[ {fun},
        fun =
         Switch[size,
              16, LLVMLibraryFunction["LLVMHalfTypeInContext"],
              32, LLVMLibraryFunction["LLVMFloatTypeInContext"],
              64, LLVMLibraryFunction["LLVMDoubleTypeInContext"],
              128, LLVMLibraryFunction["LLVMFP128TypeInContext"],
              _,   ThrowException[{"Unsupported real type", size}]
         ];
        fun[data["contextId"]]
    ]

GetRealType[args___] :=
    ThrowException[{"Invalid usage GetRealType", {args}}]



(*
 LLVM doesn't distinguish
*)
GetUnsignedIntegerType[data_?AssociationQ, size_] :=
    GetIntegerType[data, size]

GetTypeTType[data_?AssociationQ] :=
	GetIntegerType[data, 32]


GetOpaqueExprType[data_?AssociationQ] :=
    Module[ {ty},
        ty = LLVMLibraryFunction["LLVMGetTypeByName"][data["moduleId"], 
          "struct.expr_opaque_struct"];
        If[ ty === 0,
            ty = LLVMLibraryFunction["LLVMStructCreateNamed"][
              data["contextId"], "struct.expr_opaque_struct"]
        ];
        LLVMLibraryFunction["LLVMPointerType"][ty, 0]
    ]

(*
 Outer wrapping for Expr types, the contents might be an opaque expr, 
 or a normal, integer etc...   This expr supports refcount, flags etc...
*)
GetWrappedExprType[data_?AssociationQ, contents_, name_] :=
    Module[ {ty, args},
        ty = LLVMLibraryFunction["LLVMStructCreateNamed"][data["contextId"],
           name];
        args = If[ data["isDebug"],
                   {GetIntegerType[data, 32], GetIntegerType[data, 16], GetIntegerType[data, 8], 
                    GetIntegerType[data, 8], GetOpaqueExprType[data], 
                    GetOpaqueExprType[data], GetIntegerType[data, 32], contents},
                   {GetIntegerType[data, 32], GetIntegerType[data, 16], GetIntegerType[data, 8], 
                    GetIntegerType[data, 8], contents}
               ];
        WrapIntegerArray[ LLVMLibraryFunction["LLVMStructSetBody"][ty, #, Length[args], 0]&, args];
        ty
    ]

(*
 This is a base undifferentiated expr,  ie one that doesn't know if it is 
 Normal, Integer etc...
*)
GetBaseExprType[data_?AssociationQ] :=
    Module[ {ty, conts},
        ty = LLVMLibraryFunction["LLVMGetTypeByName"][data["moduleId"], "struct.expr_struct"];
        If[ ty === 0,
            conts = LLVMLibraryFunction["LLVMStructCreateNamed"][data["contextId"], "struct_expr_contents"];
            WrapIntegerArray[ LLVMLibraryFunction["LLVMStructSetBody"][ conts, #, 1, 0]&, { GetMIntType[data]}];
            ty = GetWrappedExprType[data, conts, "struct.expr_struct"];
        ];
        LLVMLibraryFunction["LLVMPointerType"][ty, 0]
    ]

GetNormalExprType[data_?AssociationQ] :=
    Module[ {ty, conts},
        ty = LLVMLibraryFunction["LLVMGetTypeByName"][data["moduleId"], 
          "struct.expr_normal_struct"];
        If[ ty === 0,
            conts = 
             LLVMLibraryFunction["LLVMStructCreateNamed"][data["contextId"], 
              "struct_expr_normal_contents"];
             WrapIntegerArray[ LLVMLibraryFunction["LLVMStructSetBody"][conts, #, 4, 0]&, {
            		GetIntegerType[data, 64], GetIntegerType[data, 32], GetMIntType[data], GetBaseExprType[data]}];
            ty = GetWrappedExprType[data, conts, "struct.expr_normal_struct"];
        ];
        LLVMLibraryFunction["LLVMPointerType"][ty, 0]
    ]

GetMIntegerExprType[data_?AssociationQ] :=
    Module[ {ty, conts},
        ty = LLVMLibraryFunction["LLVMGetTypeByName"][data["moduleId"], 
          "struct.expr_minteger_struct"];
        If[ ty === 0,
            conts = 
             LLVMLibraryFunction["LLVMStructCreateNamed"][data["contextId"], 
              "struct_expr_minteger_contents"];
             WrapIntegerArray[ LLVMLibraryFunction["LLVMStructSetBody"][conts,
             #, 1, 0]&, {GetMIntType[data]}];             
            ty = GetWrappedExprType[data, conts, "struct.expr_minteger_struct"];
        ];
        LLVMLibraryFunction["LLVMPointerType"][ty, 0]
    ]
	
GetMRealExprType[data_?AssociationQ] :=
    Module[ {ty, conts},
        ty = LLVMLibraryFunction["LLVMGetTypeByName"][data["moduleId"], 
          "struct.expr_mreal_struct"];
        If[ ty === 0,
            conts = 
             LLVMLibraryFunction["LLVMStructCreateNamed"][data["contextId"], 
              "struct_expr_mreal_contents"];
             WrapIntegerArray[LLVMLibraryFunction["LLVMStructSetBody"][conts,
             #, 1, 0]&, {GetMRealType[data]}];             
            ty = GetWrappedExprType[data, conts, "struct.expr_mreal_struct"];
        ];
        LLVMLibraryFunction["LLVMPointerType"][ty, 0]
    ]

GetRawExprType[data_?AssociationQ] :=
	Which[
		TrueQ[data["expressionVersion"] >= 2],
			GetRawExprType2[data],
		True,
			GetRawExprTypeBase[data]
	]
		

GetRawExprType2[data_?AssociationQ] :=
	Module[ {ty, conts, rawData},
        ty = LLVMLibraryFunction["LLVMGetTypeByName"][data["moduleId"], 
          "struct.expr_raw_struct"];
        If[ ty === 0,
        	rawData = GetMIntType[data];
        	rawData = LLVMLibraryFunction["LLVMPointerType"][rawData, 0];
            conts = 
             LLVMLibraryFunction["LLVMStructCreateNamed"][data["contextId"], 
              "struct_expr_raw_contents"];
             WrapIntegerArray[LLVMLibraryFunction["LLVMStructSetBody"][conts,#, 5, 0]&, {
             	GetOpaqueExprType[data],  (* expr rawhead *)
             	rawData,                  (* void* rawData *)
             	GetIntegerType[data, 32],       (* cctype rawcc *)
             	GetIntegerType[data, 16],       (* UBIT16 rawtype *)
             	GetIntegerType[data, 16]       (* UBIT16 rawflags *)
             	
             }];
            ty = GetWrappedExprType[data, conts, "struct.expr_raw_struct"];
        ];
        LLVMLibraryFunction["LLVMPointerType"][ty, 0]
	]

GetRawExprTypeBase[data_?AssociationQ] :=
	Module[ {ty, conts, rawData},
        ty = LLVMLibraryFunction["LLVMGetTypeByName"][data["moduleId"], 
          "struct.expr_raw_struct"];
        If[ ty === 0,
        	rawData = GetMIntType[data];
        	rawData = LLVMLibraryFunction["LLVMPointerType"][rawData, 0];
            conts = 
             LLVMLibraryFunction["LLVMStructCreateNamed"][data["contextId"], 
              "struct_expr_raw_contents"];
             WrapIntegerArray[LLVMLibraryFunction["LLVMStructSetBody"][conts,#, 6, 0]&, {
             	GetOpaqueExprType[data],  (* expr rawhead *)
             	GetMIntType[data],       (* mint raw length *)
             	GetIntegerType[data, 32],       (* cctype rawcc *)
             	GetIntegerType[data, 16],       (* UBIT16 rawtype *)
             	GetIntegerType[data, 16],       (* UBIT16 rawflags *)
             	rawData                   (* void* rawData *)
             }];
            ty = GetWrappedExprType[data, conts, "struct.expr_raw_struct"];
        ];
        LLVMLibraryFunction["LLVMPointerType"][ty, 0]
	]

GetComplexExprType[data_?AssociationQ] :=
    Module[ {ty, conts},
        ty = LLVMLibraryFunction["LLVMGetTypeByName"][data["moduleId"], 
          "struct.expr_complex_struct"];
        If[ ty === 0,
            conts = 
             LLVMLibraryFunction["LLVMStructCreateNamed"][data["contextId"], 
              "struct_expr_complex_contents"];
             WrapIntegerArray[ LLVMLibraryFunction["LLVMStructSetBody"][conts, #, 2, 0]&, {
            		GetBaseExprType[data], GetBaseExprType[data]}];
            ty = GetWrappedExprType[data, conts, "struct.expr_complex_struct"];
        ];
        LLVMLibraryFunction["LLVMPointerType"][ty, 0]
    ]



GetStringType[data_?AssociationQ] :=
    Module[ {ty},
        ty = LLVMLibraryFunction["LLVMGetTypeByName"][data["moduleId"], 
          "struct.mobject_string_struct"];
        If[ ty === 0,
            ty = LLVMLibraryFunction["LLVMStructCreateNamed"][
              data["contextId"], "struct.mobject_string_struct"]
        ];
        LLVMLibraryFunction["LLVMPointerType"][ty, 0]
    ]


GetPackedArrayBaseType[data_?AssociationQ] :=
	GetPackedArrayType[data]
	
	(*Module[ {ty},
        ty = LLVMLibraryFunction["LLVMGetTypeByName"][data["moduleId"], 
          "struct.MTensorBase"];
        If[ ty === 0,
            ty = LLVMLibraryFunction["LLVMStructCreateNamed"][
              data["contextId"], "struct.MTensorBase"]
        ];
		LLVMLibraryFunction["LLVMPointerType"][ty, 0]
	]*)


$TypeCache = <||>

GetPackedArrayType[data_?AssociationQ] :=
	GetMTensorType[data]

(*
  Lookup a type from a type resolution Module,  this is for types that are defined 
  in the runtime.  But the names change every time the runtime is cloned into this module.
*)

LookupTypeFromCreator[ data_?AssociationQ, name_] :=
	Module[ {ty, funId, funTy},
		funId = LLVMLibraryFunction["LLVMGetNamedFunction"][data["typeResolutionModule"], name];
		If[funId === 0,
			ThrowException[{"Cannot find the creator ", name}]];
		funTy = LLVMLibraryFunction["LLVMTypeOf"][funId];
		ty = LLVMLibraryFunction["LLVMGetElementType"][funTy];
		ty = LLVMLibraryFunction["LLVMGetReturnType"][ty];
		ty
	]

	
GetMTensorType[data_?AssociationQ] :=
	Module[ {ty},
		ty = LookupTypeFromCreator[ data, "mtensor_null"];
		ty	
	]

GetMNumericArrayType[data_?AssociationQ] :=
	Module[ {ty},
		ty = LookupTypeFromCreator[ data, "mnumericarray_null"];
		ty	
	]

GetCompilerErrorType[data_?AssociationQ] :=
	Module[ {ty},
		ty = LookupTypeFromCreator[ data, "CompilerError_null"];
		ty	
	]


(*
  This doesn't match the definition, which now comes from the runtime.
  TODO fix this.
*)

GetMTensorPropertiesType[data_?AssociationQ] :=
	Module[ {name = "struct.TensorProperty", ty, precTy, dimsTy, rankTy, typeTy, flagsTy},
		ty = Lookup[$TypeCache, name, Null];
		If[ ty =!= Null,
			Return[ty]];
		ty = LLVMLibraryFunction["LLVMStructCreateNamed"][data["contextId"],name];
		precTy = GetMRealType[data];
		dimsTy = GetMIntPointerType[data];
		rankTy = GetMIntType[data];
		typeTy = GetTypeTType[data];
		flagsTy = GetUnsignedIntegerType[data, 32];
		WrapIntegerArray[ LLVMLibraryFunction["LLVMStructSetBody"][ty, #, 5, 1]&, { precTy, dimsTy, rankTy, typeTy, flagsTy}];
		ty = LLVMLibraryFunction["LLVMPointerType"][ty, 0];
		AssociateTo[$TypeCache, name -> ty];
		ty
	]


GetMTensorPointerType[data_?AssociationQ] :=
	Module[ {ty},
		ty = GetMTensorType[data];
		LLVMLibraryFunction["LLVMPointerType"][ty, 0]
	]	

(*
 M Type functions
*)

GetMIntPointerType[ data_?AssociationQ] :=
	Module[ {tyId},
		tyId = GetMIntType[data];
		LLVMLibraryFunction["LLVMPointerType"][tyId, 0]
	]

GetMIntType[data_?AssociationQ] :=
	Module[ {machineIntegerSize},
    machineIntegerSize = data["machineIntegerSize"];
		GetIntegerType[data, machineIntegerSize]
	]

GetHalfType[data_] :=
	GetRealType[data, 16]
	
GetFloatType[data_] :=
	GetRealType[data, 32]

GetDoubleType[data_] :=
	GetRealType[data, 64]

GetReal128Type[data_] :=
    GetRealType[data, 128]

GetMRealType[data_?AssociationQ] :=
	GetDoubleType[data]

GetMRealPointerType[ data_?AssociationQ] :=
	Module[ {tyId},
		tyId = GetMRealType[data];
		LLVMLibraryFunction["LLVMPointerType"][tyId, 0]
	]


GetMBoolType[data_?AssociationQ] :=
	Module[{},
		GetIntegerType[data, 32]
	]
	
GetVectorType[data_?AssociationQ, ty_, len_] :=
	Module[ {},
		LLVMLibraryFunction["LLVMVectorType"][ty, len]
	]	

GetVectorComplexType[data_?AssociationQ, {ty_}] :=
	GetVectorComplexType[data, ty]


GetVectorComplexType[data_?AssociationQ, ty_] :=
	Module[ {},
		GetVectorType[data,  ty, 2]
	]	

GetMComplexType[data_?AssociationQ] :=
	Module[ {tyId},
		tyId = GetMRealType[data];
		GetVectorComplexType[data, tyId]
	]	


GetCharStarType[data_?AssociationQ] :=
	GetPointerType[data, GetIntegerType[data, 8]]

GetPointerType[data_?AssociationQ, ty_] :=
	GetHandleType[data, ty]

GetCArrayType[data_?AssociationQ, {ty_}] :=
	GetHandleType[data, ty]

GetConstantArrayType[data_?AssociationQ, {ty_, n_?IntegerQ}] :=
	LLVMLibraryFunction["LLVMArrayType"][GetLLVMType[ty], n] 
GetConstantArrayType[args___] :=
	ThrowException[{"Unsupported constant type", {ty, n}}]

GetHandleType[data_?AssociationQ, {ty_}] :=
	GetHandleType[data, ty]

GetHandleType[data_?AssociationQ, ty_] :=
	LLVMLibraryFunction["LLVMPointerType"][ty, 0]

(*
GetAddressType[data_?AssociationQ] :=
	GetIntegerType[data, 64]

*)

GetMTensorPropertiesType[data_?AssociationQ] :=
	Module[ {name = "struct.TensorProperty", ty, precTy, dimsTy, rankTy, typeTy, flagsTy},
		ty = Lookup[$TypeCache, name, Null];
		If[ ty =!= Null,
			Return[ty]];
		ty = LLVMLibraryFunction["LLVMStructCreateNamed"][data["contextId"],name];
		precTy = GetMRealType[data];
		dimsTy = GetMIntPointerType[data];
		rankTy = GetMIntType[data];
		typeTy = GetTypeTType[data];
		flagsTy = GetUnsignedIntegerType[data, 32];
		WrapIntegerArray[LLVMLibraryFunction["LLVMStructSetBody"][ty, #, 5, 1]&, { precTy, dimsTy, rankTy, typeTy, flagsTy}];
		ty = LLVMLibraryFunction["LLVMPointerType"][ty, 0];
		AssociateTo[$TypeCache, name -> ty];
		ty
	]


GetNamedStructReference[data_?AssociationQ, name_] :=
	Module[ {ty},
		ty = Lookup[$TypeCache, name, Null];
		If[ ty =!= Null,
			Return[ty]];
		ty = GetNamedStruct[data, name <> "_Reference"];
		ty = GetPointerType[ data, ty];
		AssociateTo[$TypeCache, name -> ty];
		ty
	]
	(*LLVMLibraryFunction["LLVMPointerType"][ty, 0]*)


GetNamedStruct[data_?AssociationQ, name_] :=
	Module[ {ty},
		ty = Lookup[$TypeCache, name, Null];
		If[ ty =!= Null,
			Return[ty]];
		ty = LLVMLibraryFunction["LLVMStructCreateNamed"][data["contextId"], name];
		AssociateTo[$TypeCache, name -> ty];
		ty
	]

	
GetMClassType[data_?AssociationQ] :=
	GetNamedStructReference[data, "MClass"]

(*
typedef struct st_utf8str
{
	mint nbytes;     /* number of bytes (not characters) in the string */
	mint length;     /* actual string length (number of characters) */
	expr expr_String;   /* was memory explicitly malloced when creating this string? */
	unsigned char *string;
} *utf8str;
*)
GetUTF8StringType[data_?AssociationQ] :=
	Module[ {name = "struct.UTF8String", ty, bytesTy, lengthTy, exprTy, stringTy},
        ty = LLVMLibraryFunction["LLVMGetTypeByName"][data["moduleId"], name];
        If[ ty === 0,
			ty = LLVMLibraryFunction["LLVMStructCreateNamed"][data["contextId"], name];
			bytesTy = GetMIntType[data];
			lengthTy = GetMIntType[data];
			exprTy = GetExprStringType[ data];
			stringTy = GetCharStarType[data];
			WrapIntegerArray[LLVMLibraryFunction["LLVMStructSetBody"][ty, #, 4, 1]&, { bytesTy, lengthTy, exprTy, stringTy}];
        ];
        LLVMLibraryFunction["LLVMPointerType"][ty, 0]
	]
	
(*
typedef struct st_MObject {
	void* data;
	UBIT32 flags;
	struct st_MObject *next;
	MClass objectClass;
} *MObject;
*)
GetMObjectType[data_?AssociationQ] :=
	Module[ {name = "struct.MObject", ty, dataTy, flagsTy, nextTy, objectClassTy},
        ty = LLVMLibraryFunction["LLVMGetTypeByName"][data["moduleId"], name];
        If[ ty === 0,
			ty = LLVMLibraryFunction["LLVMStructCreateNamed"][data["contextId"], name];
			dataTy = GetVoidPointerType[data];
			flagsTy = GetIntegerType[data, 32];
			nextTy = GetPointerType[ data, ty];
			objectClassTy = GetMClassType[data];
			WrapIntegerArray[LLVMLibraryFunction["LLVMStructSetBody"][ty, #, 4, 1]&, { dataTy, flagsTy, nextTy, objectClassTy}];
        ];
        LLVMLibraryFunction["LLVMPointerType"][ty, 0]
	]

GetMStringType[data_?AssociationQ] :=
	GetMObjectType[data];

(*
 struct string_struct {
	umint byte_length : (MINTBITS - 1); /* Length of character string in bytes */
	umint StringIsASCII : 1;  /* contains only ASCII chars 1-127? */
	umint unicode_length : (MINTBITS - 1);   /* number of Unicode chars in String */
	umint StringIsUTF8 : 1;   /* doesn't contain a \:0000 character? */
	char[len] string_data; /* Array for character string with space (at least) for a terminating character */
 };
 where 
   - MINTBITS = $SystemWordLength
 
 
*)
GetExprStringType[data_?AssociationQ] :=
	Module[ {name = "struct.expr_string_struct", ty, lengthInfoTy, dataTy},
        ty = LLVMLibraryFunction["LLVMGetTypeByName"][data["moduleId"], name];
        If[ ty === 0,
			ty = LLVMLibraryFunction["LLVMStructCreateNamed"][data["contextId"], "struct.expr_string_contents"];
			(*
			 Different layout for different word sizes.
			*)
			If[	
				data["machineIntegerSize"] === 64,
				lengthInfoTy = GetIntegerType[data, 128];
				dataTy = LLVMLibraryFunction["LLVMArrayType"][GetIntegerType[data, 8], 16]
				,
				lengthInfoTy = GetIntegerType[data, 64];
				dataTy = LLVMLibraryFunction["LLVMArrayType"][GetIntegerType[data, 8], 8]];
			WrapIntegerArray[LLVMLibraryFunction["LLVMStructSetBody"][ty, #, 2, 0]&, { lengthInfoTy, dataTy}];
	        ty = GetWrappedExprType[data, ty, name];
        ];
        LLVMLibraryFunction["LLVMPointerType"][ty, 0]
	]


GetVoidType[data_?AssociationQ] :=
	LLVMLibraryFunction["LLVMVoidTypeInContext"][data["contextId"]]


GetVoidHandleType[ data_?AssociationQ] :=
	GetPointerType[data, GetUnsignedIntegerType[data, 8]]

GetVoidPointerType[ data_?AssociationQ] :=
	GetVoidHandleType[data]



GetStructureType[data_?AssociationQ, args_List] :=
	Module[{id},
		id = WrapIntegerArray[LLVMLibraryFunction["LLVMStructTypeInContext"][data["contextId"], #, Length[args], 0]&, args];
		id
	]

GetOpaqueStructureType[data_?AssociationQ, name_String] :=
	Module[{ty = LLVMLibraryFunction["LLVMGetTypeByName"][data["moduleId"], name]},
		If[ty === 0,
			ty = LLVMLibraryFunction["LLVMStructCreateNamed"][data["contextId"], name]];
		ty
	]


(*
  Function to get LLVM Type IDs from an unresolved type specification.
*)

validLLVMType[ x_] :=
	IntegerQ[x]


GetLLVMType[data_?AssociationQ, Type[ty_]] :=
	GetLLVMType[data, ty]

GetLLVMType[data_?AssociationQ, TypeSpecifier[ty_]] :=
	GetLLVMType[data, ty]

GetLLVMType[data_?AssociationQ, {args___} -> res_] :=
	Module[{argIds, resId, tyId},
		argIds = Map[ GetLLVMType[data, #]&, {args}];
		resId = GetLLVMType[data, res];
		tyId = WrapIntegerArray[ LLVMLibraryFunction["LLVMFunctionType"][resId, #, Length[argIds], 0]&, argIds];
		tyId = LLVMLibraryFunction["LLVMPointerType"][tyId, 0];
		tyId
	]


$typeApplicationFuns =
	<|
	"CArray" -> GetCArrayType,
	"C`ConstantArray" -> GetConstantArrayType,
	"Complex" -> GetVectorComplexType,
	"Handle" -> GetHandleType,
	"PackedArray" -> (GetMTensorType[ #1]&)
	|>

GetLLVMType[data_, Type[apTy_][ args__]] :=
	GetLLVMType[data, apTy[args]]

GetLLVMType[data_, TypeSpecifier[apTy_][ args__]] :=
	GetLLVMType[data, apTy[args]]


(*
 Note that this won't work when we have broader sets of eg Complex, since the 
 GetVectorComplexType only works for the base being something that can be turned 
 into a vector.
*)
GetLLVMType[data_?AssociationQ, apTy_[ args__]] :=
	Module[{argIds, fun, tyId},
		argIds = Map[ GetLLVMType[data, #]&, {args}];
		fun = Lookup[ $typeApplicationFuns, apTy, Null];
		If[fun === Null,
			ThrowException[{"Unsupported type application", apTy}]
		];
		tyId = fun[data, argIds];
		If[ !validLLVMType[tyId],
			ThrowException[{"Did not convert type application", apTy[args]}]
		];
		tyId
	]

$typeFuns =
	<|
	"C`bool" -> GetBooleanType,
	"Boolean" -> GetBooleanType,
	"C`int" -> (GetIntegerType[#, 32]&),
	"C`int8" -> (GetIntegerType[#, 8]&),
	"C`int16" -> (GetIntegerType[#, 16]&),
	"C`int32" -> (GetIntegerType[#, 32]&),
	"C`int64" -> (GetIntegerType[#, 64]&),
	"Integer8" -> (GetIntegerType[#, 8]&),
	"Integer16" -> (GetIntegerType[#, 16]&),
	"Integer32" -> (GetIntegerType[#, 32]&),
	"Integer64" -> (GetIntegerType[#, 64]&),
	"MachineInteger" -> GetMIntType,
	"C`uint" -> (GetIntegerType[#, 32]&),
	"C`uint8" -> (GetIntegerType[#, 8]&),
	"C`uint16" -> (GetIntegerType[#, 16]&),
	"C`uint32" -> (GetIntegerType[#, 32]&),
	"C`uint64" -> (GetIntegerType[#, 64]&),
	"C`size_t" -> (GetIntegerType[#, 64]&),
	"UnsignedInteger8" -> (GetIntegerType[#, 8]&),
	"UnsignedInteger16" -> (GetIntegerType[#, 16]&),
	"UnsignedInteger32" -> (GetIntegerType[#, 32]&),
	"UnsignedInteger64" -> (GetIntegerType[#, 64]&),
	"Real16" -> GetHalfType,
	"C`half" -> GetHalfType,
	"Real32" -> GetFloatType,
	"C`float" -> GetFloatType,
	"Real64" -> GetDoubleType,
    "Real128" -> GetReal128Type,
	"C`double" -> GetHalfType,
	"Expression" -> GetBaseExprType,
	"MObject" -> GetMObjectType,
	"MTensor" -> GetMTensorType,
	"MNumericArray" -> GetMNumericArrayType,
	"Void" -> GetVoidType,
	"String" -> GetStringType,
	"VoidHandle" -> GetVoidHandleType,
	"CString" -> GetCharStarType
|>


GetLLVMType[data_?AssociationQ, cons_String] :=
	Module[{fun},
		fun = Lookup[ $typeFuns, cons, Null];
		If[fun === Null,
			ThrowException[{"Unsupported type constructor", cons}]];
		fun[data]
	]

(*
 There is no conversion for TypeLiteral so we just return the expr.
 This will cause an error if a constructor tries to use it. 
 Typically, it won't.
*)
GetLLVMType[ data_?AssociationQ, TypeFramework`TypeLiteral[val_, ty_]] :=
	TypeFramework`TypeLiteral[val, ty]

GetLLVMType[ data_?AssociationQ, unk_] :=
	ThrowException[{"Unsupported type argument", unk}]



(*
 funTy is the input form of a function type, 
 return the ID of a 
*)
GetBaseFunctionType[data_, funTy_] :=
	Module[{},
		GetLLVMType[data, funTy]
	]

(*

typedef struct st_utf8str
{
	mint nbytes;     /* number of bytes (not characters) in the string */
	mint length;     /* actual string length (number of characters) */
	expr expr_String;   /* was memory explicitly malloced when creating this string? */
	unsigned char *string;
} *utf8str;

*)

Getutf8strType[data_?AssociationQ] :=
	Module[ {name = "struct.utf8str", ty, nbytesTy, lengthTy, exprTy, dataTy},
        ty = LLVMLibraryFunction["LLVMGetTypeByName"][data["moduleId"], name];
        If[ ty === 0,
			ty = LLVMLibraryFunction["LLVMStructCreateNamed"][data["contextId"], name];
			nbytesTy = GetMIntType[data];
			lengthTy = GetMIntType[data];
			exprTy = GetVoidHandleType[ data];
			dataTy = GetCharStarType[ data];
			WrapIntegerArray[LLVMLibraryFunction["LLVMStructSetBody"][ty, #, 4, 1]&, { nbytesTy, lengthTy, exprTy, dataTy}];
        ];
        LLVMLibraryFunction["LLVMPointerType"][ty, 0]
	]



End[]


EndPackage[]