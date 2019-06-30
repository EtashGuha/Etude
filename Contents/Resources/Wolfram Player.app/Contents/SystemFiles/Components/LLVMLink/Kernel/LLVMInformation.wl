BeginPackage["LLVMLink`LLVMInformation`"]
$LLVMInformation
Begin["`Private`"]
$LLVMInformation = <|
	"LLVM_DEFAULT_TARGET_TRIPLE" -> "x86_64-apple-darwin17.6.0",
	"LLVM_ENABLE_ABI_BREAKING_CHECKS" -> False,
	"LLVM_ENABLE_THREADS" -> True,
	"LLVM_HAS_ATOMICS" -> True,
	"LLVM_HOST_TRIPLE" -> "x86_64-apple-darwin17.6.0",
	"LLVM_NATIVE_ARCH" -> "X86",
	"LLVM_NATIVE_ASMPARSER" -> "LLVMInitializeX86AsmParser",
	"LLVM_NATIVE_ASMPRINTER" -> "LLVMInitializeX86AsmPrinter",
	"LLVM_NATIVE_DISASSEMBLER" -> "LLVMInitializeX86Disassembler",
	"LLVM_NATIVE_TARGET" -> "LLVMInitializeX86Target",
	"LLVM_NATIVE_TARGETINFO" -> "LLVMInitializeX86TargetInfo",
	"LLVM_NATIVE_TARGETMC" -> "LLVMInitializeX86TargetMC",
	"LLVM_ON_UNIX" -> True,
	"LLVM_ON_WIN32" -> False,
	"LLVM_USE_INTEL_JITEVENTS" -> False,
	"LLVM_USE_OPROFILE" -> False,
	"LLVM_VERSION" -> 6.0,
	"LLVM_VERSION_MAJOR" -> 6,
	"LLVM_VERSION_MINOR" -> 0,
	"LLVM_VERSION_PATCH" -> 1,
	"LLVM_VERSION_STRING" -> "6.0.1"
|>;
End[]
EndPackage[]
