
If[llvmVersion <= 6,

    LLVMLibraryFunction["LLVMGetHostCPUName"] :=
    LLVMLibraryFunction["LLVMGetHostCPUName"] = (InstallLLVM[];LibraryFunctionLoad[LLVMLibraryName[],
        "LLVMLink_LLVMGetHostCPUName_Wrapper",
        {

        },
        (* Type[char *] *)
        "UTF8String"
    ]);
    
    LLVMLibraryFunction["LLVMGetHostCPUFeatures"] :=
    LLVMLibraryFunction["LLVMGetHostCPUFeatures"] = (InstallLLVM[];LibraryFunctionLoad[LLVMLibraryName[],
        "LLVMLink_LLVMGetHostCPUFeatures_Wrapper",
        {

        },
        (* Type[char *] *)
        "UTF8String"
    ]);
]