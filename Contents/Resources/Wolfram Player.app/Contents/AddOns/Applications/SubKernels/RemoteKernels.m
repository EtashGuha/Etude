(* :Title: RemoteKernels.m -- launch kernels through MathLink and remote login *)

(* :Context: SubKernels`RemoteKernels` *)

(* :Author: Roman E. Maeder *)

(* :Copyright: © 2008-2016 by Wolfram Research, Inc. *)

(* :Package Version: 1.0  *)

(* :Mathematica Version: 6 *)

(* :History:
   1.0 first released version.
*)

(* this implementation of the remote kernels interface uses the methods previously in PCT/RemoteKernels for launching kernels
  through MathLink calls out of Mathematica. It tries to maintain b/w compatibility so it uses the same names as were used in
  PCT/Configuration.m. Launching local kernels ("localhost") is no longer part of this implementation. See LocalKernels.m
 *)

BeginPackage["SubKernels`RemoteKernels`", { "SubKernels`" }]
  
Needs["SubKernels`Protected`"]

(* the configuration language. A kernel is described as a RemoteMachine[...] data element *)
 
RemoteMachine::usage = "RemoteMachine[host, (template), (n), opts...] is a description of an available remote machine.
	The host, template, and n are the arguments to LaunchRemote; the default template is $RemoteCommand."

$RemoteCommand::usage = "$RemoteCommand is the template for launching a kernel on a remote machine.
	It should be a string template suitable for use in LinkLaunch or Run.
	The first slot `1` is replaced by the name of the remote machine.
	For connection type LinkCreate, the second slot `2` is replaced by the address to which to connect, the third slot `3` is replaced by the user name, and the fourth slot `4` is the link protocol specification."

$RemoteUserName::usage = "$RemoteUserName has been superceded by $RemoteUsername."

(* short forms of kernel descriptions recognized by this implementation:
 	- "hostname" (including a FQDN)
 *)

 (* options *)
 
If[!ValueQ[System`ConnectionType::usage], System`ConnectionType::usage = "ConnectionType is an option of LaunchRemote.
	ConnectionType->LinkLaunch uses LinkLaunch to start a remote kernel.
	ConnectionType->LinkCreate creates a local link and uses Run to start a remote kernel, which is expected to open a connection to the link created.
	ConnectionType->Automatic uses LinkLaunch for local kernels and LinkCreate for nonlocal kernels."]
If[!ValueQ[System`LinkName::usage], System`LinkName::usage = "LinkName->name is an option of LaunchRemote that gives the name of the link to be created with ConnectionType->LinkCreate."]

(* class methods and variables *)

remoteKernelObject::usage = "remoteKernelObject[method] is the remote kernels class object."

(* additional constructors, methods *)

LaunchRemote::usage = "LaunchRemote[host, template, ConnectionType->LinkLaunch, opts] launches a kernel on the given host using LinkLaunch; opts is passed as options to LinkLaunch.
	LaunchRemote[host, template, ConnectionType->LinkCreate, opts..] creates a local link and uses Run to start a remote kernel, which is expected to open a connection to the link created. opts is passed as options to LinkCreate."

Options[LaunchRemote] = {
	ConnectionType -> Automatic,
	LinkName -> Automatic,
	LinkProtocol -> Automatic,
	LinkHost -> "",
	KernelSpeed -> 1
}

(* destructor *)

(* the data type is public, for easier subclassing and argument type check *)

remoteKernel::usage = "remoteKernel[..] is a remote subkernel."

LaunchRemote::rsh = "Command `1` may have failed (exit code `2`)."
LaunchRemote::ct = "ConnectionType `1` is not one of `2`."
LaunchRemote::lnk = "Creating link with `1` failed."

 (* remember context now *)

remoteKernelObject[subContext] = Context[]


Begin["`Private`"]
 
`$PackageVersion = 0.9;
`$thisFile = $InputFileName
 
Needs["ResourceLocator`"]

$packageRoot = DirectoryName[System`Private`$InputFileName]
textFunction = TextResourceLoad[ "SubKernels", $packageRoot]


(* data type:
 	remoteKernel[ lk[link, descr, arglist, speed] ]
 		link	associated LinkObject
 		descr	host name, if known, other identifier otherwise
 		arglist	list of arguments used in constructor, so that it may be relaunched if possible
 		speed	speed setting, mutable
 *)
 
SetAttributes[remoteKernel, HoldAll] (* data type *)
SetAttributes[`lk, HoldAllComplete] (* the head for the base class data *)
 
(* private selectors; pattern is remoteKernel[ lk[link_, descr_, arglist_, id_, ___], ___ ]  *)
 
remoteKernel/: linkObject[ remoteKernel[lk[link_, ___], ___]] := link
remoteKernel/: descr[remoteKernel[lk[link_, descr_, ___], ___]] := descr
remoteKernel/: arglist[remoteKernel[lk[link_, descr_, arglist_, ___], ___]] := arglist
remoteKernel/: kernelSpeed[remoteKernel[lk[link_, descr_, arglist_, speed_, ___], ___]] := speed
remoteKernel/: setSpeed[remoteKernel[lk[link_, descr_, arglist_, speed_, ___], ___], r_] := (speed = r)

(* description language methods *)

RemoteMachine/: KernelCount[RemoteMachine[host_, cmd_String:"", n_Integer:1, opts:OptionsPattern[]]] := n

(* format of description items *)

Format[RemoteMachine[host_, cmd_String:"", n_Integer:1, OptionsPattern[]]/;n==1] :=
	StringForm["\[LeftSkeleton]a kernel on `1`\[RightSkeleton]", host]
Format[RemoteMachine[host_, cmd_String:"", n_Integer:1, OptionsPattern[]]/;n>1] :=
	StringForm["\[LeftSkeleton]`1` kernels on `2`\[RightSkeleton]", n, host]

(* factory method *)

RemoteMachine/: NewKernels[RemoteMachine[args___], opts:OptionsPattern[]] := LaunchRemote[args, opts]


(* interface methods *)

remoteKernel/:  subQ[ remoteKernel[ lk[link_, descr_, arglist_, ___] ] ] := Head[link]===LinkObject

remoteKernel/:  LinkObject[ kernel_remoteKernel ]  := linkObject[kernel]
remoteKernel/:  MachineName[ kernel_remoteKernel ] := descr[kernel]
remoteKernel/:  Description[ kernel_remoteKernel ] := RemoteMachine@@arglist[kernel]
remoteKernel/:  Abort[ kernel_remoteKernel ] := kernelAbort[kernel]
remoteKernel/:  SubKernelType[ kernel_remoteKernel ] := remoteKernelObject
(* KernelSpeed: use generic implementation *)

(* kernels should be cloneable; speed setting may have changed after initial launch *)

remoteKernel/:  Clone[kernel_remoteKernel] := NewKernels[Description[kernel], KernelSpeed->kernelSpeed[kernel]]


(* list of open kernels *)

`$openkernels = {}

remoteKernelObject[subKernels] := $openkernels


(* constructors *)

(* slots in the cmd StringForm template:
	`1`	hostname
	`2` linkname (to connect to) (set to "" for LinkLaunch)
	`3` the remote user name $RemoteUsername
	`4` a string specifying the linkprotocol, if not Automatic (set to "" for LinkLaunch)
*)

LaunchRemote[host_String, opts:OptionsPattern[]] := LaunchRemote[host, $RemoteCommand, opts]
LaunchRemote[host_String, n_Integer, opts:OptionsPattern[]] := LaunchRemote[host, $RemoteCommand, n, opts]
(* default n is 1 *)
LaunchRemote[host_String, cmd_String, opts:OptionsPattern[]] := firstOrFailed[ LaunchRemote[host, cmd, 1, opts] ]

(* parallel launching *)

LaunchRemote[host_String, cmd_String, n_Integer?NonNegative, opts:OptionsPattern[]] :=
Module[{ct, nm, lp, lh, lnk},
    {ct,nm,lp,lh} = OptionValue[{ConnectionType, LinkName, LinkProtocol, LinkHost}];

    feedbackObject["name", RemoteMachine[host, n], n];

    Switch[ct,

     LinkLaunch,
     	If[ lp===Automatic, lp = "Pipes" ]; (* best guess for network connections *)
        With[{args=Sequence@@Flatten[{ToString[StringForm[cmd, host, "", $RemoteUsername, ""]], LinkProtocol->lp,
        	System`Utilities`FilterOptions[LinkLaunch, opts, Options[LaunchRemote]]}]},
          Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "Launching `3` kernels on `1` with `2`", host, HoldForm[LinkLaunch[args]], n];
          lnk = Table[feedbackObject["tick"]; LinkLaunch[args], {n}];
          Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "Links launched as `1`", lnk];
          lnk = deleteFailed[lnk, LaunchRemote];
        ]
	,
     LinkCreate | Automatic,
     	If[ lp===Automatic, lp = "TCPIP" ]; (* best guess for network connections *)
        With[{args=Sequence@@Flatten[{nm /. Automatic->{}, LinkProtocol->lp, LinkHost->lh,
        	System`Utilities`FilterOptions[LinkCreate, opts, Options[LaunchRemote]]}]},
          Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "Creating `2` listening links with `1`", HoldForm[LinkCreate[args]], n];
          lnk = Table[LinkCreate[args], {n}];
          Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "Links created as `1`", lnk];
          lnk = deleteFailed[lnk, LaunchRemote];
        ];
        (* slot values and run *)
        Function[{ln}, Module[{arg, code},
          With[{slot1 = host, slot2 = ln[[1]], slot3 = $RemoteUsername, slot4 = If[ lp===Automatic, "", "-linkprotocol "<>lp ]},
               arg = ToString[StringForm[cmd, slot1, slot2, slot3, slot4 ]] ];
          Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "Running command \"`1`\"", arg];
          TimeConstrained[ (* this does not work... *)
            code = Run[ arg ]; feedbackObject["tick"];
           , (* timeout *)
            Parallel`Settings`$MathLinkTimeout,
            Message[LaunchRemote::time, Run, Parallel`Settings`$MathLinkTimeout];
            LinkClose[ln];
            code = $Failed];
          If[code != 0, Message[LaunchRemote::rsh, arg, code] ];
        ]] /@ lnk
	,
     _,
     	Message[LaunchRemote::ct, ct, {LinkCreate, LinkLaunch, Automatic}];
        Return[{}]; (* unknown ConnectionType *)
    ];
    initLink[ lnk, host, {host,cmd,opts}, OptionValue[LaunchRemote, {opts}, KernelSpeed] ]
]


(* destructor; use generic implementation *)

remoteKernel/: Close[kernel_remoteKernel?subQ] := (
	$openkernels = DeleteCases[$openkernels, kernel];
	kernelClose[kernel, True]
)


(* handling short forms of kernel descriptions *)
(* exclude the names matched by LocalKernels *)

remoteKernelObject[try][Except["localhost"|"local",s_String], args___]/; StringMatchQ[s,RegularExpression["\\w+(\\.\\w+)*"]] :=
	LaunchRemote[s, args] (* hostname, but not the special one for local kernels *)


(* class name *)

remoteKernelObject[subName] = textFunction["RemoteKernelsName"]


(* raw constructor; several at once *)

initLink[links_List, host_, args_, sp_] :=
 Module[{kernels},
 	(* each kernel gets its own set of variables for the mutable fields *)
 	kernels = Module[{speed=sp}, remoteKernel[ lk[#, host, args, speed] ]]& /@ links;
 	(* local init *)
 	$openkernels = Join[$openkernels, kernels];
 	(* base class init *)
 	kernelInit[kernels]
 ]

(* single one *)

initLink[link_, args__] := firstOrFailed[ initLink[{link}, args] ]


(* persistent config uses description language with additional args Enabled->True/False,
   and UseDefault->True/False for default remote command, and list of options *)

canopts[list_List] := Flatten[list]
canopts[___] := {}

RemoteMachine/: chost[RemoteMachine[host_, ___]] := host
RemoteMachine/: ccmd[RemoteMachine[host_, cmd_, ___]] := cmd
RemoteMachine/: cnum[RemoteMachine[host_, cmd_, n_, ___]] := n
RemoteMachine/: cena[RemoteMachine[host_, cmd_, n_, ena_, ___]] := ena
RemoteMachine/: ccust[RemoteMachine[host_, cmd_, n_, ena_, cust_, ___]] := cust
RemoteMachine/: copts[RemoteMachine[host_, cmd_, n_, ena_, cust_, opts_, ___]] := canopts[opts]

`hosts = {}; (* in-core copy *)

`config

config[configQ] = True
config[nameConfig] = remoteKernelObject[subName]

config[setConfig] := config[setConfig, {}]

(* update from pre-9.1 short forms (no options) *)

config[setConfig, descr:{__RemoteMachine}] := (hosts = descr /. RemoteMachine[h_, cmd_, n_, ena_, cust_] :> RemoteMachine[h,cmd,n,ena,cust,{}];)

(* safety net *)

config[setConfig, __] := config[setConfig, {}]


config[getConfig] := hosts

config[useConfig] := Module[{use},
	use = Select[hosts, cena[#]&&cnum[#]>0&]; (* only enabled ones, with nonzero kernels *)
	use = use /. {
		r_RemoteMachine/;  ccust[r] :> RemoteMachine[chost[r], ccmd[r], cnum[r], copts[r]],
		r_RemoteMachine/; !ccust[r] :> RemoteMachine[chost[r], cnum[r], copts[r]]
	};
	use
]


`$selected

`$RKtabwidth = If[NumberQ[Parallel`Palette`Private`tabwidth], Parallel`Palette`Private`tabwidth, 500]
`$RKMainMenuWidth = Scaled[0.97]
`$RKConfigMenuWidth = Scaled[0.50]
`$RKSettingsMenuWidth = Scaled[0.45]
`$RKHostNameSize = Scaled[1]
`$RKHostNameSize1 = Scaled[0.95]

configuredHostMenu[] := Dynamic[Block[{a},
	Grid[{{textFunction["RemoteKernelsHostname"], textFunction["RemoteKernelsKernels"], textFunction["RemoteKernelsEnable"]},
		Sequence@@Table[With[{r=hosts[[ind]],ind=ind},
		  { Setter[Dynamic[$selected], ind, Dynamic[Style[chost[r],If[cena[r],Bold,Gray]]],
				Appearance -> "Palette", ImageSize->$RKHostNameSize],
			Spinner[Dynamic[hosts[[ind,3]]]],
			Checkbox[Dynamic[hosts[[ind,4]]]]
		  }],{ind,Length[hosts]}
		],
		{Invisible[Setter[1,1,"hostname",ImageSize->$RKHostNameSize]],
		 Invisible[Spinner[Dynamic[a]]],
		 Invisible[Checkbox[True]]}
	  },
	  Alignment -> {{Left, Center, Center}, Center}, Spacings -> {1, 1/2}
	]
]]

settingsHostMenu[] := Dynamic[
If[1<=$selected<=Length[hosts],
  Column[{
	textFunction["RemoteKernelsHostnameSettings"],
	InputField[Dynamic[hosts[[$selected,1]]],String, ImageSize->$RKHostNameSize1],
	"",
	Row[{Checkbox[Dynamic[hosts[[$selected,5]]]], textFunction["RemoteKernelsCustomLaunch"]}],
	InputField[Dynamic[hosts[[$selected,2]]], String, ImageSize->{$RKHostNameSize1,All},
		FieldSize->Automatic, Enabled->Dynamic[hosts[[$selected,5]]]],
	textFunction["RemoteKernelsCustomOptions"],
	InputField[Dynamic[hosts[[$selected,6]]], Expression, ImageSize->{$RKHostNameSize1,All},
		FieldSize->Automatic, Enabled->True]
  }]
 , textFunction["RemoteKernelsNoHost"]
] ]


functionsMenu[] := Switch[$OperatingSystem,
    "MacOSX", Panel[Row[{addHostButton, removeHostButton, duplicateHostButton},
          Spacer[5]], ImageSize -> $RKMainMenuWidth],
    _, Panel[Row[{addHostButton, removeHostButton, duplicateHostButton},
                  Spacer[5]], FrameMargins -> 0, Alignment -> Left, ImageSize -> $RKMainMenuWidth]
]

(* button definer *)

(#1 = Switch[$OperatingSystem,
    "MacOSX", Button[#2, #3[], Appearance -> "Palette"],
    _, Mouseover[Button[#2, Null,
        Appearance -> None,  FrameMargins -> 5],
        Button[#2, #3[]]]
])& @@@ {
	{addHostButton, Style[textFunction["RemoteKernelsAdd"], Bold], addHostAction},
	{removeHostButton, Style[textFunction["RemoteKernelsRemote"], Bold], removeHostAction},
	{duplicateHostButton, Style[textFunction["RemoteKernelsDuplicate"], Bold], duplicateHostAction}
}

addHostAction[] := (
	AppendTo[hosts, RemoteMachine["new host", $RemoteCommand, 1, False, False, {}]];
	$selected=Length[hosts]
)

removeHostAction[] := (
	If[1<=$selected<=Length[hosts], hosts=Delete[hosts,$selected]];
	If[$selected>Length[hosts], $selected=Length[hosts]]
)

duplicateHostAction[] := (
	If[1<=$selected<=Length[hosts], AppendTo[hosts, hosts[[$selected]]]];
	$selected=Length[hosts]
)


configuredMenu[] := Panel[
    configuredHostMenu[],
    BaselinePosition -> Top, ImageSize -> $RKConfigMenuWidth]

settingsMenu[] := Panel[
    settingsHostMenu[],
    BaselinePosition -> Top,
    ImageSize -> $RKSettingsMenuWidth]

config[tabConfig] := (
  $RKtabwidth = If[NumberQ[Parallel`Palette`Private`tabwidth], Parallel`Palette`Private`tabwidth, 600];
  Panel[
    Grid[{{functionsMenu[], SpanFromLeft}, {configuredMenu[], settingsMenu[]}},
         Frame -> None, Alignment -> Left],
    Appearance-> "Frameless", ImageSize -> {$RKtabwidth, All}]
)

remoteKernelObject[subConfigure] = config; (* configure class method *)


(* class variable defaults *)
 
sf[s_String] := StringReplace[s, " " -> "\\ "] (* escape spaces *)
qs[s_String] := "\"" <> s <> "\"" (* quote string *)

 (* We cannot really know what the remote kernel's OS is nor where they stashed
   the 'math' script. This assumes a remote Unix/csh from a local Windows
   master, and a remote Unix from a local Unix with ssh
*)

(* slots available in $RemoteCommand:

`1`	hostname	first argument of LaunchSlave

for ConnectionType->LinkCreate only:

`2`	linkname	as returned by LinkCreate[]

`3`	remote user	$RemoteUsername, default is $Username

`4`	protocol	if LinkProtocol is Automatic: ""
			otherwise: "-linkprotocol PROTO"

*)

`defaultRemoteCommand = Switch[$OperatingSystem,
    "Windows", "rsh `1` -n -l `3` \"wolfram -wstp -linkmode Connect `4` -linkname `2`" <> stdargs <> " >& /dev/null &\"",
    _,		   "ssh -x -f -l `3` `1` wolfram -wstp -linkmode Connect `4` -linkname '`2`'" <> stdargs
]
If[ !ValueQ[$RemoteCommand], $RemoteCommand = defaultRemoteCommand]

(* compatibility with old naming $RemoteUserName *)

If[ ValueQ[$RemoteUserName], $RemoteUsername = $RemoteUserName ]

(* format *)

setFormat[remoteKernel, "remote"]

End[]

Protect[ LaunchRemote, RemoteMachine, remoteKernelObject, remoteKernel ]

(* registry *)
addImplementation[remoteKernelObject]
 
EndPackage[]
