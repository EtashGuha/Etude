
BeginPackage["LLVMLink`Error`"]


LLVMFatalErrorHandler::usage = "LLVMFatalErrorHandler handles fatal errors from LLVM."

LLVMDiagnosticHandler::usage = "LLVMDiagnosticHandler handles diagnostic messages from LLVM."

LLVMLinkMessageHandler::usage = "LLVMLinkMessageHandler handles messages from LLVMLink."

Begin["`Private`"]


Needs["LLVMLink`"]


LLVMLibraryFunction["getDiagnosticHandlerAddress"] :=
LLVMLibraryFunction["getDiagnosticHandlerAddress"] = LibraryFunctionLoad[LLVMLibraryName[],
    "getDiagnosticHandlerAddress",
    {},
    Integer
]


LLVMFatalErrorHandler::eftl = "A fatal error occurred. The kernel will now exit. LLVM gave the following reason: `1`"

LLVMDiagnosticHandler::diag = "LLVM Diagnostic: `1`. Severity: `2`"

LLVMLinkMessageHandler::msg = "`1`"

(*
   Do this to send message:
    Block[{Internal`$MessageMenu = False},
      Message[...]]
 
 Normally, the front end sends more traffic on the preemptive link for formatting the message.
 The kernel will not be able to respond, because it is crashing.
 Because it expecting a response, The front end then displays an error box
 with "An unknown box name (ToBoxes) was sent as the BoxForm for the expression. Check the format rules for the expression." 

 Blocking on Internal`$MessageMenu uses the old style message formatting -- not as pretty, but free of
 dynamic content.


Amendment: running inside MUnit will prevent messages from being printed out. So just Print here.
*)

LLVMFatalErrorHandler[reason_] :=
	Print[StringTemplate[LLVMFatalErrorHandler::eftl][reason]]



$lastSeverity

LLVMDiagnosticHandler[description_, severity_] :=
	Module[{},

		(*
		notes are added to other messages, so use the original severity to determine whether to print
		*)
		If[severity =!= LLVMEnumeration["LLVMDiagnosticSeverity", "LLVMDSNote"],
			$lastSeverity = severity
		];

		If[willPrint[$lastSeverity],
			Print[StringTemplate[LLVMDiagnosticHandler::diag][description, severityToString[severity]]]
		]
	]

willPrint[LLVMEnumeration["LLVMDiagnosticSeverity", "LLVMDSError"]] := True
willPrint[LLVMEnumeration["LLVMDiagnosticSeverity", "LLVMDSWarning"]] := True
(* reduce noise by not printing Remarks *)
willPrint[LLVMEnumeration["LLVMDiagnosticSeverity", "LLVMDSRemark"]] := False
willPrint[LLVMEnumeration["LLVMDiagnosticSeverity", "LLVMDSNote"]] := True

severityToString[LLVMEnumeration["LLVMDiagnosticSeverity", "LLVMDSError"]] := "ERROR"
severityToString[LLVMEnumeration["LLVMDiagnosticSeverity", "LLVMDSWarning"]] := "WARNING"
severityToString[LLVMEnumeration["LLVMDiagnosticSeverity", "LLVMDSRemark"]] := "REMARK"
severityToString[LLVMEnumeration["LLVMDiagnosticSeverity", "LLVMDSNote"]] := "NOTE"




LLVMLinkMessageHandler[reason_] :=
	Print[StringTemplate[LLVMLinkMessageHandler::msg][reason]]


End[]

EndPackage[]

