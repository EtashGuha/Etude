(* :Title: Delegates.m *)

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


(*<!--Public From Delegates.m

NETNewDelegate::usage =
"NETNewDelegate[type, func] creates a new instance of the specified .NET delegate type whose action is to call the named \
Wolfram Language function when triggered. The type argument can be a string name, a NETType expression, or a Type object. The \
func argument can be the name of a function as a symbol or string, or a pure function. The function you supply will be \
called with whatever arguments the delegate type takes. NETNewDelegate is a low-level function that is not often used; \
see AddEventHandler if you want to create a Wolfram Language callback triggered by some user interface action."

SendDelegateArguments::usage =
"SendDelegateArguments is an option to AddEventHandler and NETNewDelegate that specifies which of the delegate arguments you want \
to be passed to your Wolfram Language callback function. By default, all the arguments in the delegate's signature are sent to the \
Wolfram Language function assigned to the delegate. If you are not interested in some or all of the arguments, you can make the \
callback more efficient by using SendDelegateArguments to eliminate some of the arguments. Efficiency is generally only a concern \
for arguments that are objects, not primitive types like integers or strings. The default value is All, but you can use a list of \
numbers that represent argument indices to specify which arguments to send. For example, SendDelegateArguments -> {1,3} means to \
only send the first and third arguments. Use None or {} to specify that no arguments should be sent."

CallsUnshare::usage =
"CallsUnshare is an option to AddEventHandler and NETNewDelegate that specifies whether or not the Wolfram Language callback function \
assigned to the delegate calls UnshareKernel or UnshareFrontEnd. UnshareKernel and UnshareFrontEnd are advanced functions \
that most programmers will not call directly, preferring to use DoNETModeless instead, which encapsulates the use of these \
functions. However, if you are calling an Unshare function directly in a Wolfram Language callback from a delegate, you must use the \
CallsUnshare -> True option. The default is False."

WrapInNETBlock::usage =
"WrapInNETBlock is an option to AddEventHandler and NETNewDelegate that specifies whether or not the Wolfram Language callback function \
assigned to the delegate should be implicitly wrapped in NETBlock. The default is True, so that objects sent to callback \
functions or created within them are treated as temporary and released when the callback completes. If you need an object created \
in a callback function to persist in the Wolfram Language after the callback completes, use WrapInNETBlock -> False."

DefineNETDelegate::usage =
"DefineNETDelegate[name, returnType, parameterTypes] creates a new .NET delegate type with the given name, return type, \
and parameter types. This is a rarely-used function whose main use is for creating delegates for DLL function pointers. \
In such cases there is probably not an existing .NET delegate type that is suitable, so you need to create one. DefineNETDelegate \
simply lets you do this entirely in Wolfram Language code, without resorting to writing in C# or Visual Basic. You typically go \
on to call NETNewDelegate to create a new instance of the new delegate type."

AddEventHandler::usage =
"AddEventHandler[obj@eventName, func] assigns the specified Wolfram Language function to be called when the given event fires. \
You use AddEventHandler to wire up Wolfram Language callbacks for events in .NET user interfaces, like a button click. \
The func argument can be the name of a Wolfram Language function as a string or symbol, or a pure function. The function will be \
called with whatever arguments the event sends. You can also manually create a delegate using NETNewDelegate and pass that \
instead of a function for the second argument. AddEventHandler returns a delegate object. You can pass this delegate object \
to RemoveEventHandler to remove the callback function."

RemoveEventHandler::usage =
"RemoveEventHandler[obj@eventName, delegate] removes the specified delegate from the named event. The delegate object you pass \
must have been returned from a call to AddEventHandler for that same event."

-->*)

(*<!--Package From Delegates.m

// Called directly from .NET.
delegateCallbackWrapper

-->*)


(* Current context will be NETLink`. *)

Begin["`Delegates`Private`"]


(*****************************************  NETNewDelegate  ********************************************)

NETNewDelegate::sendargs = "Invalid specification for the SendDelegateArguments option: `1`. It will be ignored."
NETNewDelegate::typeobj = "A .NET object passed as the first argument must be an instance of the System.Type class."
NETNewDelegate::args = "Improper type or number of arguments." 

Options[NETNewDelegate] = {SendDelegateArguments -> All, CallsUnshare -> False, WrapInNETBlock -> True}

(* Note that the correct full context name for the function name is captured, so there will be no problems with functions
   in packages.
*)
NETNewDelegate[type:(_String | _NETType | _?NETObjectQ), mFunc_Symbol, opts___?OptionQ] :=
    NETNewDelegate[type, Context[mFunc] <> SymbolName[mFunc], opts]

NETNewDelegate[type:(_String | _NETType | _?NETObjectQ), mFunc_String, opts___?OptionQ] :=
    Module[{typeName, send, callsUnshare, wrapInBlock},
        typeName =
            Switch[type,
                _String,
                    type,
                _NETType,
                    getAQTypeName[type],
                _?NETObjectQ,
                    If[InstanceOf[type, "System.Type"],
                        type@AssemblyQualifiedName,
                    (* else *)
                        Message[NETNewDelegate::typeobj];
                        Return[$Failed]
                    ]
            ];
        {send, callsUnshare, wrapInBlock} =
                contextIndependentOptions[{SendDelegateArguments, CallsUnshare, WrapInNETBlock}, Flatten[{opts}], Options[NETNewDelegate]];
        (* 'send' is a bit field such that bit n set means include arg n. Limited to 16 args. *)
        Switch[send,
            All,
                send = 65535,
            None,
                send = 0,
            {},
                send = 0,
            {__Integer},
                send = BitOr @@ (2^#& /@ (send - 1)),
            _,
                Message[NETNewDelegate::sendargs, send];
                send = 65535
        ];
        If[!IntegerQ[send],
            Message[NETNewDelegate::sendargs, send];
            send = 65535
        ];
        nCreateDelegate[typeName, mFunc, send, TrueQ[callsUnshare], TrueQ[wrapInBlock]]
    ]

(* This version mainly for pure functions. Must come after the def above. *)
NETNewDelegate[type:(_String | _NETType | _?NETObjectQ), mFunc_, opts___?OptionQ] :=
    NETNewDelegate[type, ToString[Unevaluated[mFunc], InputForm], opts]

NETNewDelegate[___] := (Message[NETNewDelegate::args]; $Failed)


(***********************************  delegateCallbackWrapper  *************************************)

(* delegateCallbackWrapper is the function called from .NET for callbacks for delgates created
   by NETNewDelegate (including event callbacks from AddEventHandler). Callbacks that might
   turn off kernel sharing need to be handled specially--thus the callsUnshare argument. It is 
   not required that a callback with callsUnshare actually turn off sharing, but if you are going
   to turn off sharing, you must have it set to True. Behavior is better if you don't set it, though,
   so it should only be used for callbacks that execute UnshareKernel.
*)

(* HoldRest so that objects are not entered into M before the NETBlock takes effect. *)
Attributes[delegateCallbackWrapper] = {HoldRest}

(* This is the function called from .NET for callbacks for delegates created by NETNewDelegate
   (including event callbacks from AddEventHandler). 
*)
delegateCallbackWrapper[func_String, args_List, callsUnshare:(True | False), wrapInNETBlock:(True | False)] :=
    If[wrapInNETBlock,
        NETBlock[delegateCallbackWrapper0[func, args, callsUnshare]],
    (* else *)
        Module[{result},
            (* No NETBlock on this call, but it might be the case that a callback that _did_ use NETBlock is
               on the stack. Therefore we promote all the NETObjects that are part of the result from this call
               so that they escape from that outer NETBlock.
               A common example is when a DoNETModal dialog is closed from a button@Click handler.
               The callback that assigns the result from DoNETModal happens in the context of the callback from
               the button@Click event, which is likely set up to use NETBlock. Note that this is not a very robust
               solution as it only works if at most one NETBlock-wrapped callback is in progress. However, this is
               by far the most common case.
            *)
            result = delegateCallbackWrapper0[func, args, callsUnshare];
            KeepNETObject[Cases[result, _?NETObjectQ, Infinity]];
            result
        ]
    ]

delegateCallbackWrapper0[func_, args_, callsUnshare_] :=
    Module[{wasShared, result},
        If[!callsUnshare,
            result = ToExpression[func] @@ args;
            (* This is the value returned to .NET from the delegate callback: *)
            {argTypeToInteger[result], result},
        (* else *)
            wasShared = KernelSharedQ[NETLink[]];
            result = ToExpression[func] @@ args;
            (* If sharing was turned off in this call we need to do some fixup on $Line. *)
            If[wasShared && KernelSharedQ[NETLink[]], $Line--];
            Null
        ]
    ]


(***************************************  DefineNETDelegate  *******************************************)

(* If no user-supplied name, create a default one. *)
DefineNETDelegate[retType_String, paramTypes:{___String}] :=
    DefineNETDelegate["DynamicDelegate$" <> ToString[$ModuleNumber], retType, paramTypes]

DefineNETDelegate[name_String, retType_String, paramTypes:{___String}] :=
    Module[{newTypeName},
        If[Head[InstallNET[]] =!= LinkObject,
            Message[DefineNETDelegate::netlink, DefineNETDelegate];
            Return[$Failed]
        ];
        newTypeName = nDefineDelegate[name, fixType[retType], fixType /@ paramTypes];
        If[StringQ[newTypeName],
            LoadNETType[newTypeName],
        (* else *)
            (* Message emitted by nDefineDelegate. *)
            $Failed
        ]
    ]


(***************************************  Add/RemoveEventHandler  *****************************************)

(* Note that we never actually allow the eventRef (obj@EventName, or obj@prop@fld@EventName, or Class`StaticEventName) passed
   in to Add/RemoveEventHandler to evaluate. This is not a meaningful operation in .NET--you cannot access events like
   they were fields. You will get a compiler error if you try. We mirror this in Mathematica by issuing a message
   whenever someone tries to get or set the value of an event field. For instance events the error is an exception
   thrown from C# code, and for static events it is issued by Mathematica defs attached to the symbols when the class is loaded.
   
   The callsUnshare argument is used to indicate that the callback _might_ execute UnshareKernel (or UnshareFrontEnd).
   If a call from .NET causes kernel sharing to be turned off, it must be executed within an EnterTextPacket, not EvaluatePacket.
   This argument tells .NET to wrap the callback in EnterTextPacket. It's not a big problem if the callback doesn't actually cause
   kernel sharing to be turned off. EnterTextPacket increments $Line and therefore screws up %, so it's not the default.
*)


NET::event = "You cannot get or set the value of the event `1`. You can only refer to an event within AddEventHandler or RemoveEventHandler."

AddEventHandler::badstaticevent = RemoveEventHandler::badstaticevent = "The name `1` does not appear to refer to a valid static event. Did you misspell the name or forget to load the class?"
AddEventHandler::badobj = RemoveEventHandler::badobj = "Invalid object reference for the event `1`."
AddEventHandler::dlg = "Could not create the necessary delegate of type `1`."


Options[AddEventHandler] = {SendDelegateArguments -> All, CallsUnshare -> False, WrapInNETBlock -> True}

Attributes[AddEventHandler] = {HoldFirst}


AddEventHandler[eventRef_, delegateOrMFunc_Symbol, opts___?OptionQ] := 
    If[NETObjectQ[delegateOrMFunc],
        (* Is a delegate object. Perhaps a warning here if user supplied opts, as they will be ignored--they
           must be applied when the delegate was created in NETNewDelegate.
        *)
        processEventHandler[eventRef, True, "", delegateOrMFunc, opts],
    (* else *)
        (* Is the name of a Mathematica function. Note that the correct full context name for the function
           name is captured, so there will be no problems with functions in packages.
        *)
        processEventHandler[eventRef, True, Context[delegateOrMFunc] <> SymbolName[delegateOrMFunc], Null, opts]
    ]
        
AddEventHandler[eventRef_, mFunc_String, opts___?OptionQ] := 
    processEventHandler[eventRef, True, mFunc, Null, opts]

(* For pure functions, mainly. *)
AddEventHandler[eventRef_, mFunc_, opts___?OptionQ] := 
    processEventHandler[eventRef, True, ToString[Unevaluated[mFunc], InputForm], Null, opts]
        


RemoveEventHandler::delegate = "The argument `1` is not a valid delegate returned by AddEventHandler."

Attributes[RemoveEventHandler] = {HoldFirst}

RemoveEventHandler[eventRef_, delegate_?NETObjectQ] := 
    processEventHandler[eventRef, False, "", delegate] /; InstanceOf[delegate, "System.Delegate"]

RemoveEventHandler[eventRef_, bad_] := (Message[RemoveEventHandler::delegate, bad]; $Failed)

 
(*********  The worker function used by both AddEventHandler and RemoveEventHandler  **********)

Attributes[processEventHandler] = {HoldFirst}

processEventHandler[eventRef_, isAdd_, mFunc_String, dlg_?NETObjectQ, opts___?OptionQ] := 
    NETBlock[
        Module[{aqTypeName, eventsObject, heldEvt, evtName, delegateTypeName, delegate, callsUnshare, sendDlgArgs, wrapInBlock},
            If[MatchQ[Hold[eventRef], Hold[_Symbol]],
                (* Static event. *)
                eventsObject = Null;
                aqTypeName = aqTypeNameFromStaticSymbol[eventRef];
                If[!StringQ[aqTypeName],
                    If[isAdd,
                        Message[AddEventHandler::badstaticevent, ToString[HoldForm[eventRef], OutputForm]],
                    (* else *)
                        Message[RemoveEventHandler::badstaticevent, ToString[HoldForm[eventRef], OutputForm]]
                    ];
                    Return[$Failed]
                ];
                evtName = ToString[HoldForm[eventRef], OutputForm];
                ctxtPos = Flatten[StringPosition[evtName, "`"]];
                If[ctxtPos =!= {},
                    evtName = StringDrop[evtName, Last[ctxtPos]]
                ],
            (* else *)
                (* Instance event. *)
                aqTypeName = "";
                heldEvt = Level[Hold[eventRef], {-1}, HoldForm, Heads->False];  (* e.g., Hold[Click] *)
                evtName = ToString[heldEvt, OutputForm];                        (* e.g., "Click"     *)
                ctxtPos = Flatten[StringPosition[evtName, "`"]];
                If[ctxtPos =!= {},
                    evtName = StringDrop[evtName, Last[ctxtPos]]
                ];
                eventsObject = evalToContainingObject[Hold[eventRef], heldEvt];
                (* Here is how to get the object with only string manipulation. Is this better or not??
                    eventsObject = ToExpression[StringReplace[ToString[Unevaluated[eventRef], InputForm], ("[" <> evtName <> "]") -> ""]];
                *)
                If[!NETObjectQ[eventsObject],
                    If[isAdd,
                        Message[AddEventHandler::badobj, evtName],
                    (* else *)
                        Message[RemoveEventHandler::badobj, evtName]
                    ];
                    Return[$Failed]
                ]
            ];
            If[isAdd && dlg === Null,
                (* The typical case where the user supplied a function for the callback. *)
                {callsUnshare, sendDlgArgs, wrapInBlock} =
                        contextIndependentOptions[{CallsUnshare, SendDelegateArguments, WrapInNETBlock}, Flatten[{opts}], Options[AddEventHandler]];
                delegateTypeName = nDlgTypeName[eventsObject, aqTypeName, evtName];
                delegate = NETNewDelegate[delegateTypeName, mFunc, CallsUnshare -> callsUnshare,
                                            SendDelegateArguments -> sendDlgArgs, WrapInNETBlock -> wrapInBlock];
                If[!NETObjectQ[delegate],
                    Message[AddEventHandler::dlg, delegateTypeName];
                    Return[$Failed]
                ],
            (* else *)
                (* User supplied their own delegate for the callback. *)
                delegate = dlg
            ];
            If[isAdd,
                (* This function returns the delegate. *)
                nAddHandler[eventsObject, aqTypeName, evtName, delegate],
            (* else *)
                nRemoveHandler[eventsObject, aqTypeName, evtName, delegate]
            ]
        ]
    ]



(* This takes Hold[f@g@h@event] ---> f@g@h without evaluating anything except f@g@h. It evaluates to the object reference
   that is holding the desired event.
*)
evalToContainingObject[expr_Hold, HoldForm[heldEvent_]] := ReleaseHold[expr /. HoldPattern[x_[heldEvent]] :> x]


End[]
