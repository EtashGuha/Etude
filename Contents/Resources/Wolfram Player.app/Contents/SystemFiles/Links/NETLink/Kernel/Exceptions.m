(* :Title: Exceptions.m *)

(* :Author:
        Todd Gayley
        tgayley@wolfram.com
*)

(* :Package Version: 1.7 *)

(* :Mathematica Version: 5.0 *)
             
(* :Copyright: .NET/Link source code (c) 2003-2019, Wolfram Research, Inc. All rights reserved.

   Use is governed by the terms of the .NET/Link license agreement, which can be found at
   www.wolfram.com/solutions/mathlink/netlink.
*)

(* :Discussion:
    
   This file is a component of the .NET/Link Mathematica source code.
   It is not a public API, and should never be loaded directly by users or programmers.

   .NET/Link uses a special system wherein one package context (NETLink`) has its implementation
   split among a number of .m files. Each component file has its own private context, and also
   potentially introduces public symbols (in the NETLink` context) and so-called "package" symbols,
   where the term "package" comes from Java terminology, referring to symbols that are visible
   everywhere within the implementation of .NET/Link, but not to clients. The NETLink.m file itself
   is produced by an automated tool from the component files and contains only declarations.
   
   Do not modify the special comment markers that delimit Public- and Package-level exports.
*)


(*<!--Public From Exceptions.m

GetNETException::usage =
"GetNETException[] returns the .NET exception object that was thrown in the most recent call from the Wolfram Language to .NET. \
It returns Null if no exception was thrown in the most recent call. You can use GetNETException in conjunction with \
$NETExceptionHandler to implement a custom exception-handling scheme in the Wolfram Language."

$NETExceptionHandler::usage =
"$NETExceptionHandler allows you to control how exceptions thrown in .NET are handled in the Wolfram Language. The default behavior \
is for exceptions to appear as messages in the Wolfram Language. If you want to override this behavior (e.g., to temporarily \
silence messages from exceptions), assign a value to $NETExceptionHandler. The value of $NETExceptionHandler is treated as \
a function that will be passed 3 arguments: the symbol associated with the message (usually the symbol NET), \
the message tag (the string \"netexcptn\" for a typical exception or \"netpexcptn\" for an exception generated \
by a \"manual return\" method where the exception occurs after the method has manually sent its result back to the Wolfram Language), \
and the descriptive string of text associated with the message. You will typically set $NETExceptionHandler within a Block \
so that its effect will be limited to a precisely defined segment of code, as in the following example that silences messages: \
Block[{$NETExceptionHandler = Null&}, obj@Method[]]. You can use GetNETException[] within your handler function to obtain \
the actual .NET exception object that was thrown."

NET::usage =
"NET is only used as a generic symbol for some messages."

-->*)


(*<!--Package From Exceptions.m

$internalNETExceptionHandler

(* Used for changing the generic message NET::netexcptn to SomeSymbol::netexcptn. Use this when a function in .NET/Link
   uses the CallPacket mechanism but is not a user-level ctor/method call. For example, CreateCOMObject is implemented with
   its own CallPacket and can generate exceptions with messages that come from inside COM. It's convenient to let these
   exceptions percolate up to top level as NET::netexcptn, and then associate them with CreateCOMObject as CreateCOMObject::netexcptn.
*)
associateMessageWithSymbol

// All these are called only from .NET
prepareForManualReturn
handleException
manualException
specialException

-->*)


(* Current context will be NETLink`. *)

Begin["`Exceptions`Private`"]


(*********************************  GetNETException  ************************************)

GetNETException[] := nGetException[]


(*********************************  General messages  ************************************)

General::netobj = "Invalid .NET object reference."
General::netobj1 = "Invalid .NET object reference: `1`."
General::netexcptn = "A .NET exception occurred: `1`."
General::netpexcptn = "A .NET exception occurred after the result was returned to the Wolfram Language: `1`."


(*****************************  Special exception messages  *****************************)

(* These are the "special exceptions", meaning they have special message text. The args to these
   messages will have the member name as `1` and the type name as `2`.
   These message tags are referred to directly in Install.cs, so they must be changed in both places.
*)

NET::noctor = "No public constructor exists for the .NET type `2`."
NET::nomethod = "No public instance method named `1` exists for the .NET type `2`."
NET::nofield = "No public field or property named `1` exists for the .NET type `2`."
NET::noindexer = "Type `2` has no indexer (that is, no default parameterized property)."
NET::noprop = "No public property named `1` exists for the .NET type `2`."

NET::nocommeth = "No method named `1` exists for the given COM object."
NET::nocomprop = "No property named `1` exists for the given COM object."

NET::ctorargc = "Wrong number of arguments supplied for constructor of the .NET type `2`."
NET::methargc = "Wrong number of arguments supplied for method `1`."
NET::indxrargc = "Wrong number of arguments supplied for indexer (default parameterized property in VB terminology) of the .NET type `2`."
NET::parampropargc = "Wrong number of arguments supplied for parameterized property named `1`."

NET::ctorargs = "Improper arguments supplied for constructor of the .NET type `2`."
NET::methodargs = "Improper arguments supplied for method named `1`."
NET::fieldtype = "Wrong argument type for assignment to field named `1`."
NET::proptype = "Wrong argument type for assignment to property named `1`."
NET::indxrargs = "Improper arguments supplied for indexer (default parameterized property in VB terminology) of the .NET type `2`."
NET::parampropargs = "Improper arguments supplied for parameterized property named `1`."

NET::fieldnoset = "Field named `1` can only be read, not assigned to."
NET::propnoget = "Property named `1` has no 'get' accessor."
NET::propnoset = "Property named `1` can only be read, not assigned to."
NET::indxrnoget = "Indexer (default parameterized property in VB terminology) of the .NET type `2` has no 'get' accessor."
NET::indxrnoset = "Indexer (default parameterized property in VB terminology) of the .NET type `2` can only be read, not assigned to."

NET::event = "You can only refer to the `1` event within AddEventHandler or RemoveEventHandler."

NET::cast = "Object of type `1` cannot be cast to type `2`."

(* Users should not see these errors. I believe they could only reflect bugs in .NET/Link. *)
NET::badobj = "An unexpected error occurred. A call is being made on an invalid .NET object."
NET::badtype = "An unexpected error occurred. A .NET type is being used that has not been loaded into .NET/Link."


(********************************  Exception reporting  **********************************)

(* This code keeps the link as an arg that is passed around, so it is forward-looking to the possibility
   of more than one VM running at a time.
*)

(* Exception handling has to accommodate manual/non-manual functions, and for manuals, whether the exception
   is thrown before, during, or after the user has sent a complete result on the link. The basic idea is that
   if a function has a manual return, in most cases Mathematica must read twice from the link: the first time
   to get the answer, and again to get exception info. The only exception (no pun intended) is where
   the exception occurs before anything has been put on the link--in this case there is only one read, which
   gets the exception info.

   Several of the functions here are called directly from .NET. they are:
        prepareForManualReturn
        handleException
        manualException      wrapper around exception thrown after a function declares itself manual.
        specialException     wrapper around special exception info.
        
   Exceptions are reported from .NET by sending back handleException[exc] (non-manual exceptions) or manualException[exc].
   The exc itself will either be a string to be displayed as NET::netexcptn (e.g., exceptions generated during invocation
   of the target method), or they will be specialException["tag", memberName:(_String | Null), typeName:(_String | Null)]
   which is a special wrapper to indicate that this message has a special tag and args. Such a message must be defined
   above (e.g., NET::methodnotfound).
*)

(* Called from .NET when ml.BeginManual() is executed. Basically, this function bails us out of the default
   ExternalAnswer loop and into our own equivalent, readManualResult. Once we're in readManualResult, we know
   we're in "manual mode" and can interpret things properly.
*)
prepareForManualReturn[link_LinkObject] := readManualResult[link]


readManualResult[link_LinkObject] :=
    Module[{res, afterAborted},
        (* Shutting off $Messages here is so that the LinkRead::linkep message ("unexpected end of packet",
           new in 4.0) will not show up if an exception occurs in the middle of sending the result. This message
           is technically correct, but unnecessarily disturbing since this is part of the plan. But I do want the
           LinkRead::linkd (link died) message to appear, so this is handled below in the fall-through Switch branch.
        *)
        Block[{$Messages}, res = LinkReadHeld[link]];
        Switch[res,
            Hold[$Aborted],
                (* Either an exc was thrown during the interval where the result was being put on the link,
                   or the func deliberately returned $Aborted as the result. In the latter case, readTrailingExceptionInfo
                   will just get Null.
                *)
                afterAborted = readTrailingExceptionInfo[link];
                If[afterAborted === Null,
                    $Aborted,
                (* else *)
                    handleException[afterAborted]
                ],
            Hold[_EvaluatePacket],
                If[LinkWrite[link, ReturnPacket[CheckAbort[res[[1,1]], $Aborted]]] === $Failed,
                    $Failed,
                (* else *)
                    readManualResult[link]
                ],
            Hold[_prepareForManualReturn],
                (* Multiple calls to BeginManual(). Just ignore it and call readManualResult again. *)
                readManualResult[link],
            Hold[_manualException],
                (* Was a manual return, but nothing was put on link by user. Just call handleException.
                   This is the one case where a function declares itself manual but this does not result in
                   two reads from the link.
                *)
                handleException @@ res,   (* Apply just gets rid of the Hold. *)
            Hold[_ReturnPacket],
                (* Normal manual return, but user wrapped it in ReturnPacket. *)
                handlePostException[readTrailingExceptionInfo[link]];
                res[[1,1]],
            _Hold,
                (* Normal manual return. *)
                handlePostException[readTrailingExceptionInfo[link]];
                res[[1]],
            _,
                (* I think it will always be $Failed, and that link is dead. *)
                If[!MemberQ[Links[], link],
                    Message[LinkObject::linkd, ToString[link]],
                (* else *)
                    (* Don't think this will ever be taken, but if link is still alive there is presumably
                       exception info remaining to be read.
                    *)
                    readTrailingExceptionInfo[link]
                ];
                res
        ]
    ]

(* This performs the "second read" for exception info. This will either be Null for no exception, or manualException
   or autoException if there was an exception. The result of readException is fed into a "handle" function that knows
   whether it's a clean, dirty, or post exception.
*)
readTrailingExceptionInfo[link_LinkObject] := LinkRead[link]

handleException[e_String] := ( reportException[e, False]; $Failed )
handleException[e_specialException] := ( reportException[e]; $Failed )
handleException[e_manualException] := ( reportException[First[e], False]; $Failed )

(* Post exceptions are those that are thrown after the (manual) result has been completely sent. *)
handlePostException[Null] = Null
handlePostException[e_manualException] := reportException[First[e], True]

(* Strategy for the exception-reporting functions is as follows: If user has defined a $NETExceptionHandler,
   use it, otherwise use $internalNETExceptionHandler, which has a default value (issue as a Message) but can
   also be temporarily reset by the internals of .NET/Link (e.g., to associateMessageWithSymbol).
   
   Note that we always report specialExceptions (these are CallNETException). An example is the exception
   thrown when the wrong number of arguments is sent. Users can only override reporting of exceptions thrown
   by the _internals_ of methods that they call, not those thrown by the .NET/Link infrastructure that prepares
   the call. Such exceptions are bona fide errors and should be seen.
*)

reportException[specialException[tag_, memberName_, typeName_]] :=
    Message[MessageName[NET, tag], memberName, typeName]

reportException[e_String, isPost:(True | False)] :=
    Module[{excHandler},
        excHandler = If[ValueQ[$NETExceptionHandler], $NETExceptionHandler, $internalNETExceptionHandler];
        If[isPost, 
            excHandler[NET, "netpexcptn", e],
        (* else *)
            excHandler[NET, "netexcptn", e]
        ]
    ]


(*****************************  $internalNETExceptionHandler  ***************************)

$internalNETExceptionHandler := forceLongMessage

(* Used to replace the sym from sym::tag with the specified symbol before issuing the message. *)
associateMessageWithSymbol[sym_Symbol] := forceLongMessage[sym, #2, ##3]&

(* Use this for .NET messages, which often have long stack traces and get destroyed
   if Short is used on them. This technique preserves the use of the original $MessagePrePrint
   if the message is longer than my 2000 limit.
*)
forceLongMessage =
    Function[{sym, msgName, msgStr},
        If[(ValueQ[$MessagePrePrint] && $MessagePrePrint =!= Automatic && $MessagePrePrint =!= Identity) ||
                (StringQ[msgStr] && StringLength[msgStr] > 2000),
            Message[MessageName[sym, msgName], msgStr],
        (* else *)
            (* HoldForm instead of the more obvious Identity fixes exception-formatting breakage in 7.0.  *)
            Block[{$MessagePrePrint = HoldForm},
                Message[MessageName[sym, msgName], msgStr]
            ]
        ]
    ]


End[]
