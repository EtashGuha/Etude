Package["Databases`Common`"]

PackageImport["Databases`"]

PackageExport["DBSymbolMessageHandler"]
PackageExport["DBSelectiveQuiet"]
PackageExport["DBCreateMessageHandler"]


(*
**  The message handler for Internal`HandlerBlock. Allows one to intercept
**  a message and do something.
**
**  If the registry function DBSymbolMessageHandler does not have a specific rule
**  for a given symbol and message (returns Null), then this will do nothing.
**
**  There are two actions we currently support:
**  1. Specific handler for the message: if this key is present, then the control
**     is completely passed to that handler. It can, in particular, call DBRaise to
**     stop the current execution, or do anything else (the current execution will
**     continue unless the handler calls DBRaise at some point)
**  2. "Abort" - in this case, DBRaise is called here, with the information about the
**     message / context, passed along.
*)
ClearAll[defaultMessageHandler];
defaultMessageHandler[Hold[m: Message[MessageName[s_, _], ___], _]] :=
	With[{messageActionInfo = DBSymbolMessageHandler[s, Hold[m]]},
		If[messageActionInfo =!= Null,
			Which[
				KeyExistsQ[messageActionInfo, "Handler"],
					messageActionInfo["Handler"][Hold[m]],
				TrueQ[messageActionInfo["Abort"]],
					DBRaise[defaultMessageHandler, "message_generated", {messageActionInfo}]
			]
		]
	]


(*
**  A hook that any particular symbol can redefine, preferably via UpValues to that
**  symbol.
*)

ClearAll[DBSymbolMessageHandler];
SetAttributes[DBSymbolMessageHandler, HoldFirst];
DBSymbolMessageHandler[s_Symbol, _] := Null (* No-op by default *)


(*
** Messages matching the pattern {symPattern::msgPattern}, are wrapped in Quiet
** dynamically; Other messages are genrated as normal / passed unchanged
*)
ClearAll[DBSelectiveQuiet];
DBSelectiveQuiet[symPattern_, msgPattern_] :=
    Function[
        code
        ,
        Internal`InheritedBlock[{Message},
            Module[{inMsg, quiet},
                Block[{quiet},
                    SetAttributes[quiet, HoldAll];
                    quiet[Verbatim[MessageName][symPattern, msgPattern], ___] :=
                        Function[c, Quiet @ c, HoldAll];
                    quiet[x_, ___] := # &;

                    Unprotect[Message];
                    (call : Message[args___]) /; ! TrueQ[inMsg] :=
                        Block[{inMsg = True},
                            quiet[args][call]
                        ];
                    Protect[Message];

                    code
                ]
            ]
        ]
        ,
        HoldAll
    ]


$defaultMessageSilencer := $defaultMessageSilencer = DBSelectiveQuiet[
    s_Symbol /; StringStartsQ[Context[s], "Databases`"], _
];


(*
** Main function to handle internal messages. It works as follows.
**
** 1. If quietForeignMessages is set to False (default), then all internal 
** messages pass through. If it is set to True, then all internal messages are 
** always silenced for the outside code.
**
** 2. Messages, which should prompt special behavior, are:
**      a. First, silenced by the message silencer function (default behavior for
**      them is Quite and ignore them)
**      b. Are processed by messageHandler function (custom handlers have to be
**      register using DBSymbolMessageHandler, if ignoring them is not the right
**      behavior)
**
** 3. All other messages would:
**     a. Cause abnormal code execution (an exception thrown with DBRaise), if 
**     the abortOnForeignMessages is set to True (default)
**     b. Otherwise, pass through, if quietForeignMessages is set to False, or 
**     be silently ignored if quietForeignMessages is set to True.
*)
ClearAll[DBCreateMessageHandler];
DBCreateMessageHandler[
	messageSilencer: Except[True | False]: Automatic,
	abortOnForeignMessages: True | False : True,
    quietForeignMessages : True | False : False,
    messageHandler_: Automatic
] :=
    With[{
        handler = If[messageHandler === Automatic, defaultMessageHandler, messageHandler],
        silencer = If[messageSilencer === Automatic, $defaultMessageSilencer, messageSilencer]
        },
    	Function[
    		code,
    		Module[{result},
    			If[TrueQ[quietForeignMessages], Quiet, Identity] @ Check[
    				result = Internal`HandlerBlock[
    					{ "Message", handler},
    					silencer[code]
    				]
    				,
    				If[abortOnForeignMessages,
    					DBRaise[Databases`Databases, "message_generated_in_inner_function_call", {result}],
    					(* else *)
    					result
    				]
    			]
    		]
    		,
    		HoldAll
    	]
    ]
