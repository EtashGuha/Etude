(* :Title: UI.m *)

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


(*<!--Public From UI.m

DoNETModal::usage =
"DoNETModal[form] displays the specified .NET form in the foreground and does not return until the form window is closed. \
DoNETModal[form, expr] evaluates expr just before the form is closed and returns the result. \
Typically, DoNETModal is used to implement a modal dialog box that needs to interact with the Wolfram Language while the dialog box \
is active, or one that returns a result to the Wolfram Language when it is dismissed."

DoNETModeless::usage =
"DoNETModeless[form] displays the specified .NET form in the foreground and then returns. The form can interact with the Wolfram Language \
while it is active, but it will not interfere with normal use of the Wolfram Language via the notebook front end. That is what is meant \
by the \"modeless\" state--the form does not monopolize the Wolfram Language kernel while it is active."

EndNETModal::usage =
"EndNETModal[] causes DoNETModal to return. It is rarely called directly by programmers. When a form is activated with DoNETModal, \
.NET/Link arranges for EndNETModal[] to be called automatically when the form is closed. In advanced scenarios, programmers might \
want to call EndNETModal directly."

ShowNETWindow::usage =
"ShowNETWindow[form] displays the specified .NET form in the foreground. It is used internally by DoNETModal and DoNETModeless, \
so programmers using either of those functions will not need to call it. You can call ShowNETWindow to activate a form that does \
not need to interact with the kernel (and therefore does not need DoNETModal or DoNETModeless), or to ensure that a form that has \
previously been displayed is brought in front of any notebook windows and un-minimized if necessary."

FormStartPosition::usage =
"FormStartPosition is an option to DoNETModal, DoNETModeless, ShowNETWindow, and ShowNETConsole that controls the location on \
screen of the form when it first appears. The possible values are Center (the form will be centered on the screen), Automatic \
(the form will have the Windows default location), and Manual (the form will appear at a location specified elsewhere, for example, \
by setting the form's Location property). The default value is Center. This option only controls the location of the form \
when it is first made visible."

ActivateWindow::usage =
"ActivateWindow is an option to DoNETModeless that specifies whether to make the window visible. The default is True. Set it to \
False if you want to enter the modeless state but not display the window until a later time."

ShowNETConsole::usage =
"ShowNETConsole[] displays the .NET console window and begins capturing output sent to the Console.Out and Console.Error \
streams. Anything written to these streams before ShowNETConsole is first called will not appear, and closing the console window \
will stop capturing the streams (until ShowNETConsole is called again). ShowNETConsole[\"stdout\"] captures only Console.Out, \
and ShowNETConsole[\"stderr\"] captures only Console.Error."

-->*)

(*<!--Package From UI.m

-->*)


(* Current context will be NETLink`. *)

Begin["`UI`Private`"]

(**********************************************  DoNETModal  ************************************************)

(* DoNETModal enters a loop that waits until the Java side sends back
        EvaluatePacket[EndModal[___]]
   Relies on J/Link's DoModal.
*)

DoNETModal::form = 
"You must specify a valid object of type System.Windows.Forms.Form (or any subclass) as the first argument to DoNETModal. \
If you are certain that you do not want the modal state governed by the lifespan of a Form window, you can pass Null as \
the first argument. If you do this you must make sure that you have previously arranged for a way to end the modal state."

DoNETModal::arg =
"You must specify a valid object of type System.Windows.Forms.Form (or any subclass), or Null, as the first argument to DoNETModal."


Attributes[DoNETModal] = {HoldRest}   (* So the 2nd arg (the return value code) won't be evaluated. *)

Options[DoNETModal] = {FormStartPosition -> Center}


DoNETModal[form_?NETObjectQ] := DoNETModal[form, Null]

DoNETModal[form_?NETObjectQ, opt:(_Rule | _RuleDelayed)] := DoNETModal[form, Null, opt]

DoNETModal[form_?NETObjectQ, returnValue_, opts:(_Rule | _RuleDelayed)...] :=
    Module[{isUsingReturnValue, retCode, modalResult, userResult, endModalDelegate, returnValueDelegate = Null},
        startPos = contextIndependentOptions[FormStartPosition, {opts}, Options[DoNETModal]];
        setFormStartPosition[form, startPos];
        isUsingReturnValue = Unevaluated[returnValue] =!= Null;
        retCode = 
            Which[
                !isUsingReturnValue,
                    Null,   (* Value will not be used anyway. *)
                StringQ[Unevaluated[returnValue]],
                    "(" <> ToString[userResult] <> " = (" <> returnValue <> "))&",
                MatchQ[Unevaluated[returnValue], _Function],
                    (userResult = returnValue[])&,
                True,
                    "(" <> ToString[userResult] <> " = (" <> ToString[Unevaluated[returnValue], InputForm] <> "))&"
            ];
        AbortProtect[
            If[form =!= Null,
                If[isUsingReturnValue,
                    returnValueDelegate = AddEventHandler[form@Closing, retCode, SendDelegateArguments -> None, WrapInNETBlock -> False]
                ];
                endModalDelegate = AddEventHandler[form@Closed, "EndNETModal[]&", SendDelegateArguments -> None]
            ];
            If[nModal[True, form] === $Failed,
                (* Don't go into DoModal if nModal fails (e.g., trying to resurrect a previously-closed window). *)
                Return[$Failed]
            ];
            modalResult = DoModal[NETLink[]];
            nModal[False, Null];
            If[form =!= Null,
                If[isUsingReturnValue,
                    RemoveEventHandler[form@Closing, returnValueDelegate]
                ];
                RemoveEventHandler[form@Closed, endModalDelegate]
            ]
        ];
        Which[
            isUsingReturnValue,
                userResult,
            MatchQ[modalResult, {_Integer, _}],
                (* We get here if user is not using the returnValue argument (it was Null). In that case, we
                   fall back to using the actual result of DoModal, which is the result from a delegateCallbackWrapper
                   and therefore it is an {argType, expr} pair. The expr will be Null, as that is what EndNETModal[]
                   returns. However, in advanced scenarios the programmer might be calling EndNETModal manually
                   and relying on the behavior that DoModal returns what the call that executed EndModal returns.
                *)
                Last[modalResult],
            True,
                (* It is possible in very rare cases that the result of the call that executed EndNETModal[] will not
                   come back as an {argType, expr} pair. This would only happen if the programmer made the callback
                   in a direct, low-level way, without using the AddEventHandler mechanism.
                *)
                modalResult
        ]
    ] /; form === Null || InstanceOf[form, "System.Windows.Forms.Form"]

DoNETModal[] := (Message[DoNETModal::form]; $Failed)

DoNETModal[form_] := (Message[DoNETModal::arg]; $Failed)

(* EndNETModal is just an alias for J/Link's EndModal. *)
EndNETModal = EndModal


(**********************************************  DoNETModeless  ************************************************)

DoNETModeless::form = 
"You must specify a valid .NET object of type System.Windows.Forms.Form (or any subclass) as the first argument."


Options[DoNETModeless] = {FormStartPosition -> Center, ActivateWindow -> True, ShareFrontEnd -> False, SharingPrompt -> ""}


DoNETModeless[form_?NETObjectQ, opts___?OptionQ] :=
    Module[{startPos, activate, shareFE, prompt},
        {startPos, activate, shareFE, prompt} =
                contextIndependentOptions[{FormStartPosition, ActivateWindow, ShareFrontEnd, SharingPrompt}, Flatten[{opts}], Options[DoNETModeless]];
        setFormStartPosition[form, startPos];
        If[TrueQ[shareFE],
            (* Using filterOptions[ShareKernel, ...] here is not a typo. ShareFrontEnd does not declare its own options,
               but forwards any passed-in options to its internal ShareKernel call.
            *)
            With[{token = ShareFrontEnd[NETLink[], SharingPrompt -> prompt, filterOptions[ShareKernel, opts]]},
                AddEventHandler[form@Closed, UnshareFrontEnd[token]&, SendDelegateArguments -> None, CallsUnshare -> True]
            ],
        (* else *)
            With[{token = ShareKernel[NETLink[], SharingPrompt -> prompt, filterOptions[ShareKernel, opts]]},
                AddEventHandler[form@Closed, UnshareKernel[token]&, SendDelegateArguments -> None, CallsUnshare -> True]
            ]
        ];
        If[TrueQ[activate],
            (* nShow just shows the form and forces it to the foreground, a task that needs Windows API calls. *)
            nShow[form]
        ];
    ] /; InstanceOf[form, "System.Windows.Forms.Form"]

DoNETModeless[___] := (Message[DoNETModeless::form]; $Failed)


(*******************************************  ShowNETConsole  **********************************************)

Options[ShowNETConsole] = {FormStartPosition -> Center}


ShowNETConsole[opts___?OptionQ] := ShowNETConsole["stdout", "stderr", opts]

ShowNETConsole[None, opts___?OptionQ] := ShowNETConsole["none", opts]

ShowNETConsole[strms__String, opts___?OptionQ] :=
    NETBlock[
        Module[{console, startPos},
            InstallNET[];
            LoadNETType["Wolfram.NETLink.UI.ConsoleWindow"];
            LoadNETType["Wolfram.NETLink.UI.ConsoleWindow+StreamType"];
            console = ConsoleWindow`Instance;
            startPos = contextIndependentOptions[FormStartPosition, {opts}, Options[ShowNETConsole]];
            setFormStartPosition[console, startPos];
            (* By using Symbol[] here we defer creation of these contexts until ShowNETConsole is first called. *)
            console@StreamsToCapture = BitOr @@ (Union[{strms}] /.
                {"stdout" -> NETObjectToExpression[Symbol["Wolfram`NETLink`UI`ConsoleWindow`StreamType`Out"]],
                 "stderr" -> NETObjectToExpression[Symbol["Wolfram`NETLink`UI`ConsoleWindow`StreamType`Error"]],
                 "none" -> NETObjectToExpression[Symbol["Wolfram`NETLink`UI`ConsoleWindow`StreamType`None"]]});
            ShowNETWindow[console];
            console
        ]
    ]


(**********************************************  ShowNETWindow  ************************************************)

ShowNETWindow::form = 
"The argument `1` is not a valid .NET object of type System.Windows.Forms.Form (or a subclass)."

Options[ShowNETWindow] = {FormStartPosition -> Center}


ShowNETWindow[form_?NETObjectQ, opts___?OptionQ] := 
    Module[{startPos},
       startPos = contextIndependentOptions[FormStartPosition, {opts}, Options[ShowNETWindow]];
       setFormStartPosition[form, startPos];
       nShow[form]
    ] /; InstanceOf[form, "System.Windows.Forms.Form"]

ShowNETWindow[obj_] := (Message[ShowNETWindow::form, obj]; $Failed)



setFormStartPosition[form_, pos_] :=
    Which[
        pos === Center || pos === "Center",
            form@StartPosition = 1,  (* FormStartPosition.CenterScreen as an integer, to avoid loading the class. *)
        pos === Manual || pos === "Manual",
            form@StartPosition = 0   (* FormStartPosition.Manual as an integer, to avoid loading the class. *)
        (* Do nothing for Automatic, so that in addition to using the default Windows location,
            that setting allows users to specify their own setting for the StartPosition property
            (there are other values besides the 3 that are covered by option values).
        *)
    ]


End[]
