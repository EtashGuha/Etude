BeginPackage["LLVMCompileTools`ReinterpretCast`"]


Begin["`Private`"]

Needs["LLVMLink`"]
Needs["LLVMCompileTools`"]
Needs["LLVMCompileTools`Types`"]
Needs["LLVMCompileTools`Basic`"]
Needs["LLVMCompileTools`Complex`"]
Needs["CompileUtilities`Error`Exceptions`"]


(* Format is {SrcType, TargetType} *)
reinterpretCastTable := reinterpretCastTable = <|
    {"Integer8", "Integer8"} -> Identity,
    {"Integer8", "Integer16"} -> LLVMLibraryFunction["LLVMBuildSExt"],
    {"Integer8", "Integer32"} -> LLVMLibraryFunction["LLVMBuildSExt"],
    {"Integer8", "Integer64"} -> LLVMLibraryFunction["LLVMBuildSExt"],
    {"Integer8", "UnsignedInteger8"} -> Identity,
    {"Integer8", "UnsignedInteger16"} -> LLVMLibraryFunction["LLVMBuildSExt"],
    {"Integer8", "UnsignedInteger32"} -> LLVMLibraryFunction["LLVMBuildSExt"],
    {"Integer8", "UnsignedInteger64"} -> LLVMLibraryFunction["LLVMBuildSExt"],
    
    {"Integer16", "Integer8"} -> LLVMLibraryFunction["LLVMBuildTrunc"],
    {"Integer16", "Integer16"} -> Identity, 
    {"Integer16", "Integer32"} -> LLVMLibraryFunction["LLVMBuildSExt"],
    {"Integer16", "Integer64"} -> LLVMLibraryFunction["LLVMBuildSExt"],
    {"Integer16", "UnsignedInteger8"} -> LLVMLibraryFunction["LLVMBuildTrunc"],
    {"Integer16", "UnsignedInteger16"} -> Identity,
    {"Integer16", "UnsignedInteger32"} -> LLVMLibraryFunction["LLVMBuildSExt"],
    {"Integer16", "UnsignedInteger64"} -> LLVMLibraryFunction["LLVMBuildSExt"],
    
    {"Integer32", "Integer8"} -> LLVMLibraryFunction["LLVMBuildTrunc"],
    {"Integer32", "Integer16"} -> LLVMLibraryFunction["LLVMBuildTrunc"],
    {"Integer32", "Integer32"} -> Identity,
    {"Integer32", "Integer64"} -> LLVMLibraryFunction["LLVMBuildSExt"],
    {"Integer32", "UnsignedInteger8"} -> LLVMLibraryFunction["LLVMBuildTrunc"],
    {"Integer32", "UnsignedInteger16"} -> LLVMLibraryFunction["LLVMBuildTrunc"],
    {"Integer32", "UnsignedInteger32"} -> Identity,
    {"Integer32", "UnsignedInteger64"} -> LLVMLibraryFunction["LLVMBuildSExt"],

    {"Integer64", "Integer8"} -> LLVMLibraryFunction["LLVMBuildTrunc"],
    {"Integer64", "Integer16"} -> LLVMLibraryFunction["LLVMBuildTrunc"],
    {"Integer64", "Integer32"} -> LLVMLibraryFunction["LLVMBuildTrunc"],
    {"Integer64", "Integer64"} -> Identity,
    {"Integer64", "UnsignedInteger8"} -> LLVMLibraryFunction["LLVMBuildTrunc"],
    {"Integer64", "UnsignedInteger16"} -> LLVMLibraryFunction["LLVMBuildTrunc"],
    {"Integer64", "UnsignedInteger32"} -> LLVMLibraryFunction["LLVMBuildTrunc"],
    {"Integer64", "UnsignedInteger64"} -> Identity,
    


    {"UnsignedInteger8", "Integer8"} -> Identity,
    {"UnsignedInteger8", "Integer16"} -> LLVMLibraryFunction["LLVMBuildZExt"],
    {"UnsignedInteger8", "Integer32"} -> LLVMLibraryFunction["LLVMBuildZExt"],
    {"UnsignedInteger8", "Integer64"} -> LLVMLibraryFunction["LLVMBuildZExt"],
    {"UnsignedInteger8", "UnsignedInteger8"} -> Identity,
    {"UnsignedInteger8", "UnsignedInteger16"} -> LLVMLibraryFunction["LLVMBuildZExt"],
    {"UnsignedInteger8", "UnsignedInteger32"} -> LLVMLibraryFunction["LLVMBuildZExt"],
    {"UnsignedInteger8", "UnsignedInteger64"} -> LLVMLibraryFunction["LLVMBuildZExt"],
    
    {"UnsignedInteger16", "Integer8"} -> LLVMLibraryFunction["LLVMBuildTrunc"],
    {"UnsignedInteger16", "Integer16"} -> Identity, 
    {"UnsignedInteger16", "Integer32"} -> LLVMLibraryFunction["LLVMBuildZExt"],
    {"UnsignedInteger16", "Integer64"} -> LLVMLibraryFunction["LLVMBuildZExt"],
    {"UnsignedInteger16", "UnsignedInteger8"} -> LLVMLibraryFunction["LLVMBuildTrunc"],
    {"UnsignedInteger16", "UnsignedInteger16"} -> Identity,
    {"UnsignedInteger16", "UnsignedInteger32"} -> LLVMLibraryFunction["LLVMBuildZExt"],
    {"UnsignedInteger16", "UnsignedInteger64"} -> LLVMLibraryFunction["LLVMBuildZExt"],
    
    {"UnsignedInteger32", "Integer8"} -> LLVMLibraryFunction["LLVMBuildTrunc"],
    {"UnsignedInteger32", "Integer16"} -> LLVMLibraryFunction["LLVMBuildTrunc"],
    {"UnsignedInteger32", "Integer32"} -> Identity,
    {"UnsignedInteger32", "Integer64"} -> LLVMLibraryFunction["LLVMBuildZExt"],
    {"UnsignedInteger32", "UnsignedInteger8"} -> LLVMLibraryFunction["LLVMBuildTrunc"],
    {"UnsignedInteger32", "UnsignedInteger16"} -> LLVMLibraryFunction["LLVMBuildTrunc"],
    {"UnsignedInteger32", "UnsignedInteger32"} -> Identity,
    {"UnsignedInteger32", "UnsignedInteger64"} -> LLVMLibraryFunction["LLVMBuildZExt"],

    {"UnsignedInteger64", "Integer8"} -> LLVMLibraryFunction["LLVMBuildTrunc"],
    {"UnsignedInteger64", "Integer16"} -> LLVMLibraryFunction["LLVMBuildTrunc"],
    {"UnsignedInteger64", "Integer32"} -> LLVMLibraryFunction["LLVMBuildTrunc"],
    {"UnsignedInteger64", "Integer64"} -> Identity,
    {"UnsignedInteger64", "UnsignedInteger8"} -> LLVMLibraryFunction["LLVMBuildTrunc"],
    {"UnsignedInteger64", "UnsignedInteger16"} -> LLVMLibraryFunction["LLVMBuildTrunc"],
    {"UnsignedInteger64", "UnsignedInteger32"} -> LLVMLibraryFunction["LLVMBuildTrunc"],
    {"UnsignedInteger64", "UnsignedInteger64"} -> Identity,
    
    

    {"Real16", "Integer8"} -> LLVMLibraryFunction["LLVMBuildFPToSI"],
    {"Real16", "Integer16"} -> LLVMLibraryFunction["LLVMBuildFPToSI"],
    {"Real16", "Integer32"} -> LLVMLibraryFunction["LLVMBuildFPToSI"],
    {"Real16", "Integer64"} -> LLVMLibraryFunction["LLVMBuildFPToSI"],
    {"Real16", "UnsignedInteger8"} -> LLVMLibraryFunction["LLVMBuildFPToUI"],
    {"Real16", "UnsignedInteger16"} -> LLVMLibraryFunction["LLVMBuildFPToUI"],
    {"Real16", "UnsignedInteger32"} -> LLVMLibraryFunction["LLVMBuildFPToUI"],
    {"Real16", "UnsignedInteger64"} -> LLVMLibraryFunction["LLVMBuildFPToUI"],
    
    {"Integer8", "Real16"} -> LLVMLibraryFunction["LLVMBuildSIToFP"],
    {"Integer16", "Real16"} -> LLVMLibraryFunction["LLVMBuildSIToFP"],
    {"Integer32", "Real16"} -> LLVMLibraryFunction["LLVMBuildSIToFP"],
    {"Integer64", "Real16"} -> LLVMLibraryFunction["LLVMBuildSIToFP"],
    {"UnsignedInteger8", "Real16"} -> LLVMLibraryFunction["LLVMBuildUIToFP"],
    {"UnsignedInteger16", "Real16"} -> LLVMLibraryFunction["LLVMBuildUIToFP"],
    {"UnsignedInteger32", "Real16"} -> LLVMLibraryFunction["LLVMBuildUIToFP"],
    {"UnsignedInteger64", "Real16"} -> LLVMLibraryFunction["LLVMBuildUIToFP"],
    
    {"Real16", "Real16"} -> Identity,
    {"Real16", "Real32"} -> LLVMLibraryFunction["LLVMBuildFPExt"],
    {"Real16", "Real64"} -> LLVMLibraryFunction["LLVMBuildFPExt"],
    {"Real16", "Real128"} -> LLVMLibraryFunction["LLVMBuildFPExt"],
    

    {"Real32", "Integer8"} -> LLVMLibraryFunction["LLVMBuildFPToSI"],
    {"Real32", "Integer16"} -> LLVMLibraryFunction["LLVMBuildFPToSI"],
    {"Real32", "Integer32"} -> LLVMLibraryFunction["LLVMBuildFPToSI"],
    {"Real32", "Integer64"} -> LLVMLibraryFunction["LLVMBuildFPToSI"],
    {"Real32", "UnsignedInteger8"} -> LLVMLibraryFunction["LLVMBuildFPToUI"],
    {"Real32", "UnsignedInteger16"} -> LLVMLibraryFunction["LLVMBuildFPToUI"],
    {"Real32", "UnsignedInteger32"} -> LLVMLibraryFunction["LLVMBuildFPToUI"],
    {"Real32", "UnsignedInteger64"} -> LLVMLibraryFunction["LLVMBuildFPToUI"],
    
    {"Integer8", "Real32"} -> LLVMLibraryFunction["LLVMBuildSIToFP"],
    {"Integer16", "Real32"} -> LLVMLibraryFunction["LLVMBuildSIToFP"],
    {"Integer32", "Real32"} -> LLVMLibraryFunction["LLVMBuildSIToFP"],
    {"Integer64", "Real32"} -> LLVMLibraryFunction["LLVMBuildSIToFP"],
    {"UnsignedInteger8", "Real32"} -> LLVMLibraryFunction["LLVMBuildUIToFP"],
    {"UnsignedInteger16", "Real32"} -> LLVMLibraryFunction["LLVMBuildUIToFP"],
    {"UnsignedInteger32", "Real32"} -> LLVMLibraryFunction["LLVMBuildUIToFP"],
    {"UnsignedInteger64", "Real32"} -> LLVMLibraryFunction["LLVMBuildUIToFP"],
    
    {"Real32", "Real16"} -> LLVMLibraryFunction["LLVMBuildFPTrunc"],
    {"Real32", "Real32"} -> Identity,
    {"Real32", "Real64"} -> LLVMLibraryFunction["LLVMBuildFPExt"],
    {"Real32", "Real128"} -> LLVMLibraryFunction["LLVMBuildFPExt"],


    {"Real64", "Integer8"} -> LLVMLibraryFunction["LLVMBuildFPToSI"],
    {"Real64", "Integer16"} -> LLVMLibraryFunction["LLVMBuildFPToSI"],
    {"Real64", "Integer32"} -> LLVMLibraryFunction["LLVMBuildFPToSI"],
    {"Real64", "Integer64"} -> LLVMLibraryFunction["LLVMBuildFPToSI"],
    {"Real64", "UnsignedInteger8"} -> LLVMLibraryFunction["LLVMBuildFPToUI"],
    {"Real64", "UnsignedInteger16"} -> LLVMLibraryFunction["LLVMBuildFPToUI"],
    {"Real64", "UnsignedInteger32"} -> LLVMLibraryFunction["LLVMBuildFPToUI"],
    {"Real64", "UnsignedInteger64"} -> LLVMLibraryFunction["LLVMBuildFPToUI"],
    
    {"Integer8", "Real64"} -> LLVMLibraryFunction["LLVMBuildSIToFP"],
    {"Integer16", "Real64"} -> LLVMLibraryFunction["LLVMBuildSIToFP"],
    {"Integer32", "Real64"} -> LLVMLibraryFunction["LLVMBuildSIToFP"],
    {"Integer64", "Real64"} -> LLVMLibraryFunction["LLVMBuildSIToFP"],
    {"UnsignedInteger8", "Real64"} -> LLVMLibraryFunction["LLVMBuildUIToFP"],
    {"UnsignedInteger16", "Real64"} -> LLVMLibraryFunction["LLVMBuildUIToFP"],
    {"UnsignedInteger32", "Real64"} -> LLVMLibraryFunction["LLVMBuildUIToFP"],
    {"UnsignedInteger64", "Real64"} -> LLVMLibraryFunction["LLVMBuildUIToFP"],
    
    {"Real64", "Real16"} -> LLVMLibraryFunction["LLVMBuildFPTrunc"],
    {"Real64", "Real32"} -> LLVMLibraryFunction["LLVMBuildFPTrunc"],
    {"Real64", "Real64"} -> Identity,
    {"Real64", "Real128"} -> LLVMLibraryFunction["LLVMBuildFPExt"],

    {"Real128", "Integer8"} -> LLVMLibraryFunction["LLVMBuildFPToSI"],
    {"Real128", "Integer16"} -> LLVMLibraryFunction["LLVMBuildFPToSI"],
    {"Real128", "Integer32"} -> LLVMLibraryFunction["LLVMBuildFPToSI"],
    {"Real128", "Integer64"} -> LLVMLibraryFunction["LLVMBuildFPToSI"],
    {"Real128", "UnsignedInteger8"} -> LLVMLibraryFunction["LLVMBuildFPToUI"],
    {"Real128", "UnsignedInteger16"} -> LLVMLibraryFunction["LLVMBuildFPToUI"],
    {"Real128", "UnsignedInteger32"} -> LLVMLibraryFunction["LLVMBuildFPToUI"],
    {"Real128", "UnsignedInteger64"} -> LLVMLibraryFunction["LLVMBuildFPToUI"],
    
    {"Integer8", "Real128"} -> LLVMLibraryFunction["LLVMBuildSIToFP"],
    {"Integer16", "Real128"} -> LLVMLibraryFunction["LLVMBuildSIToFP"],
    {"Integer32", "Real128"} -> LLVMLibraryFunction["LLVMBuildSIToFP"],
    {"Integer64", "Real128"} -> LLVMLibraryFunction["LLVMBuildSIToFP"],
    {"UnsignedInteger8", "Real128"} -> LLVMLibraryFunction["LLVMBuildUIToFP"],
    {"UnsignedInteger16", "Real128"} -> LLVMLibraryFunction["LLVMBuildUIToFP"],
    {"UnsignedInteger32", "Real128"} -> LLVMLibraryFunction["LLVMBuildUIToFP"],
    {"UnsignedInteger64", "Real128"} -> LLVMLibraryFunction["LLVMBuildUIToFP"],
    
    {"Real128", "Real16"} -> LLVMLibraryFunction["LLVMBuildFPTrunc"],
    {"Real128", "Real32"} -> LLVMLibraryFunction["LLVMBuildFPTrunc"],
    {"Real128", "Real64"} -> LLVMLibraryFunction["LLVMBuildFPTrunc"],
    {"Real128", "Real128"} -> Identity
    
|>

AddCodeFunction["ReinterpretCast", addReinterpretCast]


firstArgumentType[s_String] := s
firstArgumentType[TypeSpecifier[args_ -> _]] := First[args]
resultType[s_String] := s
resultType[TypeSpecifier[_ -> res_]] := res

complexBase[ "Complex"[ b_]] :=
	b

complexBase[ _] :=
	Null

addReinterpretCast[ data_, _, {s1_, _}] :=
    Module[{funTy, argTy, resultTy, argumentComplexBase, resultComplexBase},
        funTy = data["callFunctionType"]["get"];
        If[Head[funTy] =!= TypeSpecifier, 
            ThrowException[{"Invalid function type when creating reinterpret cast:", funTy}]
        ];
        argTy = firstArgumentType[funTy];
        resultTy = resultType[funTy];
        argumentComplexBase = complexBase[argTy];
        resultComplexBase = complexBase[resultTy];
        Which[
        	argumentComplexBase =!= Null && resultComplexBase =!= Null,
        		Module[{s1Re, s1Im, reId, imId},
        			If[argumentComplexBase === resultComplexBase,
        				Return[s1]];
        			s1Re = AddExtractElement[data, Null, {s1, AddConstantMInt[data, 0]}];
       				s1Im = AddExtractElement[data, Null, {s1, AddConstantMInt[data, 1]}];
       				reId = addReinterpretCastWork[data, s1Re, argumentComplexBase, resultComplexBase];
      				imId = addReinterpretCastWork[data, s1Im, argumentComplexBase, resultComplexBase];
      				AddCreateVectorComplex[data, Null, {reId, imId}]
        		]
        	,
        	resultComplexBase =!= Null,
   				Module[{reId = addReinterpretCastWork[data, s1, argTy, resultComplexBase], restyId, imId},
   					restyId = GetLLVMType[data, resultComplexBase];
   					imId = AddNull[data, restyId];
   					AddCreateVectorComplex[data, Null, {reId, imId}]
   				]
        	,
        	argumentComplexBase =!= Null,
        		ThrowException[{"ReinterpretCast is not implemented from Complex to Real.", argTy, resultTy}]
   			,
   			True,
   				addReinterpretCastWork[data, s1, argTy, resultTy]
        ]
    ]

addReinterpretCastWork[data_, s1_, argTy_, resultTy_] :=
	Module[ {tyTuple, llvmFunc, targetLLVMTypeID, id},
   		tyTuple = {argTy, resultTy};
    	llvmFunc = reinterpretCastTable[tyTuple];
    	If[MissingQ[llvmFunc],
    		ThrowException[{"Unable to generate cast instruction for ", tyTuple}]
    	];
        If[llvmFunc === Identity,
            Return[s1]
        ];
        targetLLVMTypeID = GetLLVMType[data, resultTy];
        id = llvmFunc[ data["builderId"], s1, targetLLVMTypeID, "ReinterpretCast"];
        id
	]


End[]


EndPackage[]

