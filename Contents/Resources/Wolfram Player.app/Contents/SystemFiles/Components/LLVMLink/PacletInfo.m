(* Paclet Info File *)

Paclet[
	Name -> "LLVMLink",
	Version -> "1.0.0",
	WolframVersion -> "12+",
	Updating -> Automatic,

	Extensions -> {
		{
			"Kernel",
			Root -> "Kernel",
			Context -> {"LLVMLink`"}
		}
		,
		{ "LibraryLink" }
	}

]

