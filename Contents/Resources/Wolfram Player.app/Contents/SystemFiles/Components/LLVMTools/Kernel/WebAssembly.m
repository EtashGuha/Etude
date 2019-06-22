BeginPackage["LLVMTools`WebAssembly`"]


(* TODO: It would be nice to at least create a summary box representation of LLVMModule, but I don't
         see anything in LLVMLink, or the LLVM C API, to get anything useful out of a ModuleRef. *)


Begin["`Private`"]

Needs["LLVMTools`"]
Needs["LLVMLink`"]



LLDBinary[] :=
	FileNameJoin[{$LLVMDirectory, "bin", "lld"}]
	

runProcessError[ head_, args_] :=
	Module[{res},
		res = RunProcess[args];
		If[ Lookup[res, "ExitCode"],
			Print[res];
			$Failed,
			res
			]
	]

LLVMToWebAssembly[ mod_LLVMModule, outputFile_String] :=
	Module[ {objFile, lld, wasmFile, out},
		wasmFile = outputFile;
		objFile = CreateFile[];
		objFile = LLVMToObjectFile[mod, objFile, "TargetTriple" -> "wasm32-unknown-unknown-wasm"];
		If[ objFile === $Failed,
			Return[$Failed]];
		lld = LLDBinary[];
		out = runProcessError[ LLVMToWebAssembly, {lld, "-flavor", "wasm", "--no-entry",  objFile, "-o", wasmFile, "--allow-undefined", "--import-memory"}];
		If[ out === $Failed,
			$Failed,
			outputFile]
	]



End[] (* Private *)

EndPackage[]

