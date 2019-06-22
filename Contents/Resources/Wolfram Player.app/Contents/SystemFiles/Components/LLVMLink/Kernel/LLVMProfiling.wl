LLVMLibraryFunction["LLVMWLAddInstrProfPass"] :=
LLVMLibraryFunction["LLVMWLAddInstrProfPass"] = LibraryFunctionLoad[LLVMLibraryName[],
	"LLVMLink_LLVMWLAddInstrProfPass_Wrapper",
		{
			Integer (* LLVMPassManagerRef *)
		},
		"Void"
	]
