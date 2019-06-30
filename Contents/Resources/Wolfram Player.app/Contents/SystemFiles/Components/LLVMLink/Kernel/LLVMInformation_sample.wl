
BeginPackage["LLVMLink`LLVMInformation`"]

$LLVMInformation

Begin["`Private`"]

(* generated using  with conf being the content of llvm/Config/llvm-config.h
 * Flatten[StringCases[StringSplit[conf, "\n"], 
 *       StartOfLine ~~ "#define " ~~ param__ ~~ " " ~~ val__ ~~ EndOfLine :> ("\"" <> param <> "\"" -> val)]]
 *)

$LLVMInformation = <|
    "LLVM_DEFAULT_TARGET_TRIPLE"->"x86_64-apple-darwin15.6.0",
    "LLVM_ENABLE_ABI_BREAKING_CHECKS"->0,
    "LLVM_ENABLE_THREADS"->1,
    "LLVM_HAS_ATOMICS"->1,
    "LLVM_HOST_TRIPLE"->"x86_64-apple-darwin15.6.0",
    "LLVM_NATIVE_ARCH"->"X86",
    "LLVM_NATIVE_ASMPARSER"->"LLVMInitializeX86AsmParser",
    "LLVM_NATIVE_ASMPRINTER"->"LLVMInitializeX86AsmPrinter",
    "LLVM_NATIVE_DISASSEMBLER"->"LLVMInitializeX86Disassembler",
    "LLVM_NATIVE_TARGET"->"LLVMInitializeX86Target",
    "LLVM_NATIVE_TARGETINFO"->"LLVMInitializeX86TargetInfo",
    "LLVM_NATIVE_TARGETMC"->"LLVMInitializeX86TargetMC",
    "LLVM_ON_UNIX"->1,
    "LLVM_PREFIX"->"/usr/local/llvm_head",
    "LLVM_USE_INTEL_JITEVENTS"->0,
    "LLVM_USE_OPROFILE"->0,
    "LLVM_VERSION_MAJOR"->4,
    "LLVM_VERSION_MINOR"->0,
    "LLVM_VERSION_PATCH"->0,
    "LLVM_VERSION_STRING"->"4.0.0svn"   
|>

End[]

EndPackage[]
