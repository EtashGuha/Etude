(* :Title: FrontEndServer.m *)

(* :Author:
        Todd Gayley
        tgayley@wolfram.com
*)

(* :Package Version: 4.9 *)

(* :Mathematica Version: 4.0 *)

(* :Copyright: J/Link source code (c) 1999-2019, Wolfram Research, Inc. All rights reserved.

   Use is governed by the terms of the J/Link license agreement, which can be found at
   www.wolfram.com/solutions/mathlink/jlink.
*)

(* :Discussion:
   This package implements some utility functions for Mathematica code that wants to use
   the front end as a typesetting/graphics server. Such Mathematica programs will typically
   execute in a situation where the front end is not being used as the interface. Typically,
   a MathLink program is driving the kernel, and the MathLink program needs services
   of the front end (for graphics rendering or typeset output).

   This file is a component of the J/Link Mathematica source code.
   It is not a public API, and should never be loaded directly by users or programmers.

   J/Link uses a special system wherein one package context (JLink`) has its implementation
   split among a number of .m files. Each component file has its own private context, and also
   potentially introduces public symbols (in the JLink` context) and so-called "package" symbols,
   where the term "package" comes from Java terminology, referring to symbols that are visible
   everywhere within the implementation of J/Link, but not to clients.
*)



(* Usage messages for these are a bit superfluous, as they will typically be called from a MathLink or J/Link program,
   not an interactive Mathematica session.
*)

UseFrontEnd::usage =
"UseFrontEnd[expr] evaluates expr in an environment where the kernel can make use of the services of the notebook front end. The most important such service is producing images involving typeset expressions. The front end will be launched if required. This function will typically be called only from an external program that is driving the kernel, not from code executing in a front end notebook."

ForceLaunch::usage =
"ForceLaunch is an option to ConnectToFrontEnd that forces a new instance of the front end to be launched as opposed to sharing a currently-running instance. The default is False. It is currently only supported on Windows (on other platforms a new instance is always launched)."

CloseFrontEnd::usage =
"CloseFrontEnd[] closes the link to the front end that was opened by UseFrontEnd[] or ConnectToFrontEnd[]. After using either of those functions, you should ensure that CloseFrontEnd[] is called before you quit the kernel."

ConnectToFrontEnd::usage =
"ConnectToFrontEnd[] establishes a link to the notebook front end for use by the UseFrontEnd[] function. It returns True to indicate that the link was established correctly, and False otherwise. The front end will be launched if required. Although UseFrontEnd will call ConnectToFrontEnd if necessary, ConnectToFrontEnd is provided to allow programmers to conveniently control when the front end is launched and to receive a True/False indication if it was successful. This function will typically be called only from an external program that is driving the kernel, not from code executing in a front end notebook."

FrontEndLink::usage =
"FrontEndLink[] returns the link to the front end that will be used by UseFrontEnd[]. It will be Null if no link to the front end has been established."

$FrontEndLaunchCommand::usage =
"$FrontEndLaunchCommand specifies the command line that will be used by ConnectToFrontEnd[] to launch the front end. You can modify its value if you have some application-specific needs."

$FrontEndInitializationFunction::usage =
"$FrontEndInitializationFunction is a function that you can assign to execute when the front end link is first established by ConnectToFrontEnd[]. Your function will be passed the link to the front end, and it should return False to indicate that your initialization was not successful and the front end connection should be abandoned. Any other return value will allow the front end connection to proceed normally."


Begin["`Package`"]
(* No Package-level exports, but Begin/End are needed by tools. *)
End[]


(*
	In a 5.1 and later kernel, these JLink API functions are implemented via Developer`UseFrontEnd
	and related functions (the JLink API is maintained and still functions similarly despite the
	change in internal implementation). The legacy implementation is kept for older kernels that
	do not have the Developer`UseFrontEnd functionality.
*)


(* Current context will be JLink`. *)

Begin["`FrontEndServer`Private`"]



If[!ValueQ[$FrontEndInitializationFunction],
	$FrontEndInitializationFunction = True&
]


Options[ConnectToFrontEnd] = {ForceLaunch -> False}

ConnectToFrontEnd[cmdLine_String] :=
	Block[{$FrontEndLaunchCommand = cmdLine},
		ConnectToFrontEnd[]
	]

ConnectToFrontEnd[opts___?OptionQ] :=
	Block[{cmd, forceLaunch, feInitFunc, fePath, launchFlags, server, parts, flags},   (* Block only for speed. *)
		If[hasFrontEnd[],
			(* A no-op if front end is caller or front end is being shared. *)
			True,
        (* else *)
            forceLaunch = TrueQ[ForceLaunch /. Flatten[{opts}] /. Options[ConnectToFrontEnd]];
			(* Here we translate the J/Link style of specifying properties for the launch into
			   the Developer`InstallFrontEnd style.
			*)
			{feInitFunc, fePath, launchFlags, server} =
				{Developer`InitializationFunction, Developer`LaunchCommand, Developer`LaunchFlags, "Server"} /. Options[Developer`InstallFrontEnd];
			If[$FrontEndInitializationFunction =!= (True&),
				feInitFunc = $FrontEndInitializationFunction
			];
			If[!forceLaunch && osIsWindows[] && !StringQ[$FrontEndLaunchCommand],
				(* Developer`InstallFrontEnd defaults to forcing the launch of a new FE whereas
				   J/Link does not, so we have to feed it our own path if user has not specified
				   ForceLaunch.
				*)
				fePath = $TopDirectory <> "\\Mathematica.exe";
				PrependTo[launchFlags, "-nogui"]
			];
			If[StringQ[$FrontEndLaunchCommand],
				(* Split a command line like:  mathematica -mathlink -display :1 -nogui -geometry 1000x500+10+10 *)
				parts = StringSplit[$FrontEndLaunchCommand, " -"];
				{fePath, flags} = {First[parts], Rest[parts]};
				If[StringMatchQ[fePath, "'*"],
					(* Strip off enclosing '' around path if present. Will be reinserted by InstallFrontEnd. *)
					fePath = First[StringSplit[fePath, Characters["'"]]]
				];
				(* Restore the leading - in flags, stripped off by the StringSplit. *)
				flags = ("-" <> #)& /@ flags;
				launchFlags = flags ~Join~ Flatten[{launchFlags}];
				(* Set server to False so that Developer`InstallfrontEnd doesn't default to its True value,
				   which would have the effect of always using -server when the user supplied a path
				   even if they wanted to launch an interactive FE. If user supplies a command line
				   and they want -server behavior, they need to include -server explicitly on the line.
				*)
				server = False
			];
			Developer`InstallFrontEnd[
						Developer`InitializationFunction -> feInitFunc,
						Developer`LaunchCommand -> fePath,
						Developer`LaunchFlags -> launchFlags,
						"Server" -> TrueQ[server]
			] // (Head[#] === LinkObject&)
		]
	]


(* CloseFrontEnd is slightly inconsistent with the other functions. It always operates on the external FE link,
   whereas the other functions are satisfied if $FrontEnd is set, meaning they will deal with the "standard"
   FE if it is available.
*)
CloseFrontEnd[] := Developer`UninstallFrontEnd[]



SetAttributes[UseFrontEnd, HoldFirst]

UseFrontEnd[expr_] :=
	If[ConnectToFrontEnd[],
		(* We deliberately go through J/Link's ConnectToFrontEnd[], which will rely on
		   the Developer` equivalent, but we need to call it to get processing of
		   legacy features like $FrontEndLaunchCommand.
		*)
		Developer`UseFrontEnd[expr],
	(* else *)
		expr
	]


(* On the chance that a user might want access to the link, this public function returns it. 
    This is not the same thing as the legacy J/Link implementation, because this forces
    the fe to launch if not present. When/if there is a Developer` function that
    returns the fe link, use it here instead of InstallFrontEnd.
*)
FrontEndLink[] := Developer`InstallFrontEnd[]



End[]

