(* Wolfram Language Source File *)

BeginPackage["LLVMLink`"]

InstallLLVM::usage = "InstallLLVM[] installs the LLVM libraries for use by the system."

LLVMLibraryFunction::usage = ""
LLVMEnumeration::usage = ""

LLVMLibraryName


LLVMLink`Internal`$BuildInfo



Begin["`Private`"]

Needs["LLVMLink`LLVMLoader`"]
Needs["LLVMLink`llvmc`"]
Needs["LLVMLink`BuildInfo`"]
Needs["LLVMLink`Error`"]


End[]

EndPackage[]
