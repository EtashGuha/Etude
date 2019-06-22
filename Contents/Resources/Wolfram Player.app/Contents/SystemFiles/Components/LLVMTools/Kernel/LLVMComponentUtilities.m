BeginPackage["LLVMTools`LLVMComponentUtilities`"]



Begin["`Private`"]

Needs["LLVMTools`"]
Needs["LLVMLink`"]
Needs["CompileUtilities`Error`Exceptions`"]

LLVMToBitcodeFile[ path_, LLVMModule[id_]] :=
	Module[{res},
		res = LLVMLibraryFunction["LLVMWriteBitcodeToFile"][id, path];
		res =!= 0
	]

LLVMToLLFile[ path_, LLVMModule[ id_]] :=
	Module[{errFlag},
		ScopedAllocation["CharObjectPointer"][Function[{errorStrRef},
			Module[{err},
				err = LLVMLibraryFunction["LLVMPrintModuleToFile"][id, path, errorStrRef];
				errFlag = err =!= 0;
				If[ errFlag,
					errString = LLVMLibraryFunction["LLVMLink_getCharObjectPointer"][errorStrRef, 0];
					Message[LLVMToLLFile::llvmerr, "LLVMPrintModuleToFile",
                        errString]
					];
			]], 1];
		errFlag
	]


LLVMToString[ LLVMModule[ id_]] :=
	Module[{strRef, str},
		strRef = LLVMLibraryFunction["LLVMPrintModuleToString_toPointer"][id];
		str = LLVMLibraryFunction["setUTF8String"][strRef];
		LLVMLibraryFunction["LLVMDisposeMessage_fromPointer"][strRef];
		str
	]

LLVMToString[ LLVMFunction[ id_]] :=
	Module[{strRef, str},
		strRef = LLVMLibraryFunction["LLVMPrintValueToString_toPointer"][id];
		str = LLVMLibraryFunction["setUTF8String"][strRef];
		LLVMLibraryFunction["LLVMDisposeMessage_fromPointer"][strRef];
		str
	]

LLVMToString[ LLVMValue[ id_]] :=
	Module[{strRef, str},
		strRef = LLVMLibraryFunction["LLVMPrintValueToString_toPointer"][id];
		str = LLVMLibraryFunction["setUTF8String"][strRef];
		LLVMLibraryFunction["LLVMDisposeMessage_fromPointer"][strRef];
		str
	]

LLVMToString[ LLVMType[ id_]] :=
	Module[{strRef, str},
		strRef = LLVMLibraryFunction["LLVMPrintTypeToString_toPointer"][id];
		str = LLVMLibraryFunction["setUTF8String"][strRef];
		LLVMLibraryFunction["LLVMDisposeMessage_fromPointer"][strRef];
		str
	]

LLVMToString[args___] :=
	ThrowException[{"Unrecognized call to LLVMToString", {args}}]


LLVMVerifyModule[ LLVMModule[ modId_]] :=
	Module[ {strRef, strPtr, str = "", res, action},
		action = LLVMEnumeration["LLVMVerifierFailureAction", "LLVMReturnStatusAction"];
		ScopedAllocation["CharObjectPointer"][Function[{strPtr},
			res = LLVMLibraryFunction["LLVMVerifyModule"][modId, action, strPtr];
			strRef = LLVMLibraryFunction["dereferenceCharPointerPointer"][strPtr];
			str = LLVMLibraryFunction["setUTF8String"][strRef];
			LLVMLibraryFunction["LLVMDisposeMessage_fromPointer"][strRef];
		], 1];
		If[ res =!= 0,
  			<| "valid" -> False, "errorString" -> str|>,
  			<| "valid" -> True,  "errorString" -> ""|>]
	]

LLVMVerifyModule[args___] :=
	ThrowException[{"Unrecognized call to LLVMVerifyModule", {args}}]


LLVMTripleFromSystemID[systemID_String] := Switch[systemID,
	"Windows",
		"i686-pc-win32",
	"Windows-x86-64",
		"x86_64-pc-win32",
	"Linux",
		"i386-pc-linux",
	"Linux-x86-64",
		"x86_64-pc-linux",
	"MacOSX",
		"i386-apple-darwin",
	"MacOSX-x86-64",
		"x86_64-apple-darwin",
	"Linux-ARM",
		"armv6-unknown-linux-gnueabihf",
	"WebAssembly",
		"wasm32-unknown-unknown-wasm",
	_,
		Message[LLVMTripleFromSystemID::sysid, systemID];
		Return[$Failed]
];

ResolveLLVMTargetTriple[tripleIn_, systemIDIn_] :=
	Module[{triple = tripleIn, systemID = systemIDIn},
		If[!StringQ[triple],
            (* If TargetTriple is not specified, create a triple based on TargetSystemID. *)
            If[StringQ[systemID],
				triple = LLVMTripleFromSystemID[systemID],
            (* else *)
			(* Neither TargetTriple nor TargetSystemID options were specified.
               Default to using current machine. *)
				triple = LLVMTripleFromSystemID[$SystemID]
            ];
        ];
        triple
	]

LLVMDataLayoutFromSystemID[systemID_String] := Switch[systemID,
    "Windows",
        "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32",
    "Windows-x86-64",
        "e-m:w-i64:64-f80:128-n8:16:32:64-S128",
    "Linux",
        "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128",
    "Linux-x86-64",
        "e-m:e-i64:64-f80:128-n8:16:32:64-S128",
    "MacOSX",
        "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128",
    "MacOSX-x86-64",
        "e-m:o-i64:64-f80:128-n8:16:32:64-S128",
    "Linux-ARM",
        "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64",
    "WebAssembly",
        "e-m:e-p:32:32-i64:64-n32:64-S128",
    _,
        Message[LLVMTripleFromSystemID::sysid, systemID];
        Return[$Failed]
];


ResolveLLVMDataLayout[datalayoutIn_, systemIDIn_] :=
    Module[{datalayout = datalayoutIn, systemID = systemIDIn},
        If[!StringQ[datalayout],
            (* If DataLayout is not specified, create a datalayout based on TargetSystemID. *)
            If[StringQ[systemID],
                datalayout = LLVMDataLayoutFromSystemID[systemID],
            (* else *)
            (* Neither DataLayout nor TargetSystemID options were specified.
               Default to using current machine. *)
                datalayout = LLVMDataLayoutFromSystemID[$SystemID]
            ];
        ];
        datalayout
    ]



End[]

EndPackage[]
