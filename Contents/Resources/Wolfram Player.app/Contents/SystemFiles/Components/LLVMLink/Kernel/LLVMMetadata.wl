If[llvmVersion <= 6,

    LLVMLibraryFunction["LLVMTemporaryMDNode"] :=
    LLVMLibraryFunction["LLVMTemporaryMDNode"] = (InstallLLVM[];LibraryFunctionLoad[LLVMLibraryName[],
        "LLVMLink_LLVMTemporaryMDNode_Wrapper",
        {
            (* Type[LLVMContextRef -> struct LLVMOpaqueContext *] *)
            Integer,
            (* Type[LLVMMetadataRef * -> struct LLVMOpaqueMetadata * *] *)
            Integer,
            (* Type[size_t] *)
            Integer
        },
        (* Type[LLVMMetadataRef -> struct LLVMOpaqueMetadata *] *)
        Integer
    ]);

    LLVMLibraryFunction["LLVMDisposeTemporaryMDNode"] :=
    LLVMLibraryFunction["LLVMDisposeTemporaryMDNode"] = (InstallLLVM[];LibraryFunctionLoad[LLVMLibraryName[],
        "LLVMLink_LLVMDisposeTemporaryMDNode_Wrapper",
        {
            (* Type[LLVMMetadataRef -> struct LLVMOpaqueMetadata *] *)
            Integer
        },
        (* Type[void] *)
        "Void"
    ]);

    LLVMLibraryFunction["LLVMMetadataReplaceAllUsesWith"] :=
    LLVMLibraryFunction["LLVMMetadataReplaceAllUsesWith"] = (InstallLLVM[];LibraryFunctionLoad[LLVMLibraryName[],
        "LLVMLink_LLVMMetadataReplaceAllUsesWith_Wrapper",
        {
            (* Type[LLVMMetadataRef -> struct LLVMOpaqueMetadata *] *)
            Integer,
            (* Type[LLVMMetadataRef -> struct LLVMOpaqueMetadata *] *)
            Integer
        },
        (* Type[void] *)
        "Void"
    ])

]

LLVMLibraryFunction["LLVMSetMDNodeOperand"] :=
LLVMLibraryFunction["LLVMSetMDNodeOperand"] = (
    InstallLLVM[];
    LibraryFunctionLoad[LLVMLibraryName[],
        "LLVMLink_LLVMSetMDNodeOperand_Wrapper",
        {
            Integer, (* LLVMValueRef *)
            Integer, (* Integer offset *)
            Integer (* LLVMValueRef *)
        },
        "Void"
    ]
)