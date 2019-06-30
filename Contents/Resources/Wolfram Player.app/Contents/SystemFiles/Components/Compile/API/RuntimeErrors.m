(* This file contains code to handle and report errors that occur during compiled code execution. The exported function,
   RuntimeErrorHandler, is called from the kernel C code (externalcode.mc) when compiled code returns a CompilerErrror struct.
   These runtime errors include wrong argument count, bad arg types, aborts, and various errors like divide-by-zero.
*)

BeginPackage["Compile`API`RuntimeErrors`"]


ErrorTextFromErrorCode
ErrorCodeFromErrorText

ProcessUncompiledFunction

Begin["`Private`"]

(* Called from kernel C code (externalcode.mc) *)
Unprotect[Compile`Internal`RuntimeErrorHandler]
Clear[Compile`Internal`RuntimeErrorHandler]



(* TODO: Give these the standard kernel message treatment *)
CompiledCodeFunction::init = "Initialization of the compiled code runtime failed."
CompiledCodeFunction::argtype = "Argument type at position `1` does not match function signature."
CompiledCodeFunction::argr = "Compiled code function called with 1 argument; `2` arguments are expected."
CompiledCodeFunction::argrx = "Compiled code function called with `` arguments; `` arguments are expected."
CompiledCodeFunction::argx = "Compiled code function called with `` arguments; 1 argument is expected."
CompiledCodeFunction::runtime = "A compiled code runtime error occurred: ``."
CompiledCodeFunction::abort = "Compiled code execution was aborted."
CompiledCodeFunction::unkn = "An unknown error occurred during compiled code execution: ``."
CompiledCodeFunction::uncomp = "A compiled code runtime error occurred; reverting to uncompiled evaluation: ``."



(* 
    Called only from kernel C code when a call to a compiledCodeFunction returned 
    an error.
    
    If the userErrorFunction is not Null or Automatic then this is called with the arguments.
    
    If userErrorFunction is Automatic,  then a runtime error recomputes with the original uncompiled
    function, other errors lead to messages and a Failure result is returned.
    
    If userErrorFunction is Null, then messages are always issued and a Failure result returned.
    
    The idea is that FunctionCompile tries to use the original uncompiled
    function, while a direct call to CompileToCodeFunction does not.
*)
Compile`Internal`RuntimeErrorHandler[compiledCodeFunction_, args_, {errType_Integer, errDetail_Integer}] :=
    Module[{funData, userErrorFunction, uncompiledExpr, failureType, msg, msgParams, expectedArgCount, actualArgCount},
        funData = First[compiledCodeFunction];
        userErrorFunction = funData["ErrorFunction"];
        uncompiledExpr = funData["Input"];
        (*
          userErrorFunction of Null comes f
        *)
        If[userErrorFunction =!= Null && userErrorFunction =!= Automatic,
            (* If user supplied an ErrorFunction value, it replaces this default implementation. *)
            Return[userErrorFunction[compiledCodeFunction, args, {errType, errDetail}]]
        ];
        Switch[errType,
            1,
                failureType = "RuntimeInitialization";
                msg = Hold[CompiledCodeFunction::init];
                msgParams = {},
            2,
                failureType = "ArgumentCount";
                expectedArgCount = errDetail;
                actualArgCount = Length[args];
                msg = 
                    Which[
                        actualArgCount == 1,
                            Hold[CompiledCodeFunction::argr],
                        expectedArgCount == 1,
                            Hold[CompiledCodeFunction::argx],
                        True,
                            Hold[CompiledCodeFunction::argrx]
                    ];
                msgParams = {actualArgCount, expectedArgCount},
            3,
                failureType = "ArgumentType";
                msg = Hold[CompiledCodeFunction::argtype];
                (* errDetail holds the index of the bad argument *)
                msgParams = {errDetail},
            4,
                failureType = "RuntimeError";
                msg = Hold[CompiledCodeFunction::runtime];
                msgParams = {ErrorTextFromErrorCode[errDetail]},
            5,
                failureType = "Abort";
                msg = Hold[CompiledCodeFunction::abort];
                msgParams = {},
            _,
                (* This is only a fallthrough. There is currently no known way for this to occur. *)
                failureType = "Unknown";
                msg = Hold[CompiledCodeFunction::unkn];
                msgParams = {errType}
        ];
        If[userErrorFunction === Automatic && (failureType == "RuntimeInitialization" || failureType == "RuntimeError") && !MissingQ[uncompiledExpr],
            (* This ::uncomp message is Off by default. Note that we reuse the ::init message text as the detail message for ::uncomp. *)
            Message[CompiledCodeFunction::uncomp, If[failureType == "RuntimeError", Sequence @@ msgParams, CompiledCodeFunction::init]];
            (* Evaluate the uncompiled expression, returning whatever it returns, and whatever messages it fires. *)
            processUncompiled[uncompiledExpr] @@ args,
        (* else *) 
            (* We issue a message and return a Failure object *)
            Function[{msgName}, Message[msgName, Sequence @@ msgParams], HoldAll] @@ msg;
            Failure[failureType, <|"MessageTemplate" -> ReleaseHold[msg], "MessageParameters" -> msgParams|>]
        ]
    ]

(*
 Remove instances of KernelFunction and Typed,  perhaps this should be 
 done in the evaluator,  but it would interfere with the operation
 of KernelFunction/Typed.
*)

ProcessUncompiledFunction[fun_] :=
	processUncompiled[fun]

processUncompiled[ fun_] :=
	stripTyped[stripKernelFunction[fun]]
	
stripKernelFunction[fun_] :=	
	ReplaceAll[fun, Typed[KernelFunction[f_], _] :> f]
 
stripTyped[fun_] :=	
	ReplaceAll[fun, Typed[e_, _] :> e]
 
Protect[Compile`Internal`RuntimeErrorHandler]


ErrorTextFromErrorCode[code_Integer] :=
    If[StringQ[#],
        #,
    (* else *)
        (* fallthrough for errors with no special text *)
        "Error type " <> ToString[code]
    ]& [Lookup[$codeToStringTable, code]]

ErrorCodeFromErrorText[text_String] :=
    If[IntegerQ[#],
        #,
    (* else *)
        (* fallthrough for errors with no special text *)
        19  (* "Unknown* *)
    ]& [Lookup[$stringToCodeTable, text]]


(* This association maps runtime error subtypes (the second int field of the CompilerError struct) into textual descriptions.
   Must remain in sync with error.hpp.inc in WolframRTL.
*)
$codeToStringTable = <|
    0 -> "Success",
    1 -> "IntegerOverflow",
    2 -> "Indeterminate",
    3 -> "IntegerInexactDivision",
    4 -> "FloatingPointInexact",
    5 -> "FloatingPointInvalid",
    6 -> "FloatingPointInfinite",
    7 -> "FloatingPointOverflow",
    8 -> "FloatingPointUnknown",
    9 -> "FloatingPointNotANumber",
   10 -> "FloatingPointSubnormal",
   11 -> "Argument",
   12 -> "Logic",
   13 -> "Range",
   14 -> "Domain",
   15 -> "System",
   16 -> "DivideByZero",
   17 -> "InvalidType",
   18 -> "InvalidRange",
   19 -> "Unknown",
   20 -> "Unimplemented",
   21 -> "Memory",
   22 -> "RankError",
   23 -> "DimensionError",
   24 -> "DotTensorLength",
   25 -> "StreamNoData",
   26 -> "StreamNoSpace",
   27 -> "ArrayPartError",
   28 -> "ExpressionConversion",
   29 -> "MathTensorError",
   30 -> "DotTensorError",
   31 -> "StreamDataNotAvailable",
   32 -> "StreamSpaceNotAvailable",
   33 -> "Abort",
   34 -> "RandomNumberError",
   35 -> "TakeError",
   36 -> "PartitionError",
   37 -> "ReverseError",
   38 -> "FlattenError",
   39 -> "RotateError",
   40 -> "TransposeError",
   41 -> "SortError",
   42 -> "IntegerCast",
   43 -> "StringExtract"
|>

$stringToCodeTable = AssociationMap[Reverse, $codeToStringTable]


End[]

EndPackage[]

