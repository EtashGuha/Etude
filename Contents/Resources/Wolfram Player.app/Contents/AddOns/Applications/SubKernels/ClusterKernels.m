(* :Title: ClusterKernels.m -- launch kernels via ssh on a lot of identically configured nodes *)

(* :Context: SubKernels`ClusterKernels` *)

(* :Author: Roman E. Maeder *)

(* :Copyright: © 2016 by Wolfram Research, Inc. *)

(* :Package Version: 1.0  *)

(* :Mathematica Version: 11 *)

(* :History:
   1.0 first released version.
*)

(* use ssh in a cluster setup with potentially many remote nodes with idential configuration *)

(* static variables affecting this package:
	Parallel`Static`ClusterLinkHost	initial value of LinkHost option
	Parallel`Static`ClusterEPC		switch to a version with linbrarylink SSH for private cloud 
*)

BeginPackage["SubKernels`ClusterKernels`", { "SubKernels`" }]

Needs["SubKernels`Protected`"]

(* the configuration language. A kernel is described as a ClusterKernels[...] data element *)
 
ClusterKernels::usage = "ClusterKernels[{nodes..}, (n), opts...] is a description of an available cluster for n kernels on each node."

$RemoteScriptTemplate::usage = "The string template for the remote bash script to launch the kernels."
$BulkSshTemplate::usage = "The string template for the ssh command to connect to a remote node."
$ExtraSshOptions::usage = "Additional options to give to the ssh command."
$RemoteKernelCommand::usage = "Absolute or relative name of kernel on remote machines."
$LocalKernelCommand::usage = "Absolute name of local kernel command."
$RemoteForwardPort::usage = "$RemoteForwardPort is the default port number for remote port forwarding."

(* short forms of kernel descriptions recognized by this implementation:
 	<none>
 *)

 (* options *)
 
(* class methods and variables *)

clusterKernelObject::usage = "clusterKernelObject[method] is the Cluster kernels class object."

(* additional constructors, methods *)

LaunchCluster::usage = "LaunchCluster[{nodes..}, (n), opts...] launches n kernels on each node."

(* the options are usually given inside ClusterKernels[] objects, and are then passed to the launcher *)

Options[LaunchCluster] = {
	BulkSshTemplate :> $BulkSshTemplate,
	ClusterName -> Automatic,
	LinkHost -> If[ValueQ[Parallel`Static`ClusterLinkHost], Parallel`Static`ClusterLinkHost, Automatic],
	ExtraSshOptions :> $ExtraSshOptions,
	RemoteForward -> True,
	RemoteKernelCommand :> $RemoteKernelCommand,
	RemoteScriptTemplate :> $RemoteScriptTemplate,
	RemoteUsername :> $RemoteUsername,
	Timeout -> 16,
Nothing}

RemoteScriptTemplate::usage = "RemoteScriptTemplate:>$RemoteScriptTemplate is an option of 
	LaunchCluster and ClusterKernels that gives the remote bash script template to use."
BulkSshTemplate::usage = "BulkSshTemplateplate:>$BulkSshTemplate is an option of 
	LaunchCluster and ClusterKernels that gives the string template for the ssh command to connect to a remote node."
ExtraSshOptions::usage = "Extra$ExtraSshOptions:>$ExtraSshOptions is an option of 
	LaunchCluster and ClusterKernels that gives the additional options for the ssh command."
RemoteKernelCommand::usage = "RemoteKernelCommand:>$RemoteKernelCommand is an option of 
	LaunchCluster and ClusterKernels that gives the absolute or relative name of kernel on remote machines."
RemoteUsername::usage = "RemoteUsername:>$RemoteUsername is an option of 
	LaunchCluster and ClusterKernels that gives the remote user name (login name)."
ClusterName::usage = "Clustername->\"name\" is an option of LaunchCluster and ClusterKernels that gives
	the name to use for the given set of kernels."
RemoteForward::usage = "RemoteForward->True is an option of  LaunchCluster and ClusterKernels that speficies whether to forward all WSTP connections from remote kernels through ssh;
	RemoteForward->\"port\" uses the given remote port.
	RemoteForward->True|Automatic uses the port specified by $RemoteForwardPort for remote forwarding.
	RemoteForward->False disables remote port forwarding."

(* destructor *)

(* the data type is public, for easier subclassing and argument type check *)

clusterKernel::usage ="clusterKernel[..] is a cluster subkernel."

timings

LaunchCluster::cmd = "Command `1` may have failed: `2`."
LaunchCluster::bad = "WSTP links that have failed: `1`"
LaunchCluster::host = "Notice: Master host has more than one address `1`;
	an option setting LinkHost->\"ip\" may be needed."
LaunchCluster::portnum = "`1` is not an IP port number."

(* remember context now *)

clusterKernelObject[subContext] = Context[]

(* special settings for private cloud, library link *)

If[ !TrueQ[Parallel`Static`ClusterEPC], Parallel`Static`ClusterEPC=False ];


Begin["`Private`"]
 
`$PackageVersion = 1.0;
`$thisFile = $InputFileName
 

(* data type:
 	clusterKernel[ lk[link, descr, arglist, speed] ]
 		link	associated LinkObject
 		descr	generic description field. We do *not* know the node name!
 		arglist	list of options used in constructor, so that it may be cloned exactly
 *)
 
SetAttributes[clusterKernel, HoldAll] (* data type *)
SetAttributes[`lk, HoldAllComplete] (* the head for the base class data *)
 
(* private selectors; pattern is clusterKernel[ lk[link_, descr_, arglist_, ls_, ___], ___ ]  *)

clusterKernel/: linkObject[ clusterKernel[lk[link_, ___], ___]] := link
clusterKernel/: descr[clusterKernel[lk[link_, descr_, ___], ___]] := descr
clusterKernel/: arglist[clusterKernel[lk[link_, descr_, arglist_, ___], ___]] := arglist
clusterKernel/: linkServer[clusterKernel[lk[link_, descr_, arglist_, ls_, ___], ___]] := ls

(* description language methods *)

(* fill in the default *)
ClusterKernels[node_String, args___] := ClusterKernels[{node}, args]
ClusterKernels[nodes_, opts:OptionsPattern[]] := ClusterKernels[nodes, 1, opts]

ClusterKernels/: KernelCount[ClusterKernels[hosts_List, n_Integer, opts:OptionsPattern[]]] := n*Length[hosts]

(* format of description items *)

Format[ClusterKernels[nodes_List, n_Integer, OptionsPattern[]]] :=
	StringForm["\[LeftSkeleton]`1` kernels each on `2` nodes\[RightSkeleton]", n, Length[nodes]]

(* factory method *)

ClusterKernels/: NewKernels[HoldPattern[ClusterKernels[args___]], opts:OptionsPattern[]] := LaunchCluster[args, opts]


(* interface methods *)

clusterKernel/:  subQ[ clusterKernel[ lk[link_, descr_, arglist_, ___] ] ] := Head[link]===LinkObject

clusterKernel/:  LinkObject[ kernel_clusterKernel ]  := linkObject[kernel]
clusterKernel/:  MachineName[ kernel_clusterKernel ] := descr[kernel]
clusterKernel/:  Description[ kernel_clusterKernel ] := ClusterKernels[descr[kernel], 1, Sequence@@arglist[kernel]]
clusterKernel/:  Abort[ kernel_clusterKernel ] := kernelAbort[kernel]
clusterKernel/:  SubKernelType[ kernel_clusterKernel ] := clusterKernelObject
clusterKernel/:  kernelSpeed[ kernel_clusterKernel ] := 1 (* fake it *)
clusterKernel/:  setSpeed[ kernel_clusterKernel, _ ] := 1 (* not supported *)

(* kernels should be cloneable *)

clusterKernel/: Clone[kernel_clusterKernel] := NewKernels[Description[kernel]]


(* list of open kernels *)

`$openkernels = {}

clusterKernelObject[subKernels] := $openkernels


(* constructors *)

(* slots in the BulkSshTemplate StringForm shell template:
	`1`	nodename/ip
	`2` remote user name
	`3` the link server address to connect to
	`4` the filename with the remote script
	`5` extra options for ssh
	`6` the number of kernels to launch
   note: this script is for the $SystemShell on the master
   it must return quickly, something like "ssh ..... &" will do nicely
*)

If[ !ValueQ[$BulkSshTemplate],
	$BulkSshTemplate = "ssh -x `5` `2`@`1` "<>$SystemShell<>" -s `6` `3` < `4` &\n"
]


(* slots in the RemoteScriptTemplate StringForm bash template:
	`1`	the cmd to launch a kernel including all non-MathLink options space-separated
	`2` the link name (link server address) to connect to
	`3` the number of kernels to launch
  note: this script is for bash, unless the above shell template specifies a different interpreter
  
  the alternate link options, -linkprotocol TCPIP -linkoptions 260, break ssh forwarding
*)

If[ !ValueQ[$RemoteScriptTemplate],
  If[Parallel`Static`ClusterEPC,
	$RemoteScriptTemplate = "t=1; for i in $(seq 1 $1) ; do `1` -mathlink -linkmode Connect -linkprotocol UUIDTCPIP -linkname `2` &>/dev/null & sleep $t ; t=0.05 ; \
done; wait",
	$RemoteScriptTemplate = "t=1; for i in $(seq 1 $1) ; do `1` -mathlink -linkmode Connect -linkprotocol UUIDTCPIP -linkname `2` &>/dev/null & sleep $t ; t=0.05 ; \
done; wait"
  ]]

(* not used for EPC *)
If[ !ValueQ[$ExtraSshOptions],
	$ExtraSshOptions = "-o ConnectTimeout=5"
]

If[ !ValueQ[$RemoteForwardPort],
	$RemoteForwardPort = "11235" (* same as default local LinkServer port *)
]

(* the arguments for the LinkCreate, except LinkHost *)
$linkServerCreateOptions = {}

(* the options for the remote wolfram command, except -linkname, starting with a space *)

$mathOptions = stdargs <> " -noprompt"

LaunchCluster[node_String, args___] := LaunchCluster[{node}, args]
LaunchCluster[nodes_, opts:OptionsPattern[]] := LaunchCluster[nodes, 1, opts]

If[ Parallel`Static`ClusterEPC, (* special setup for EPC; need the cloud ssh launcher library *)
	Get["CloudRemoteKernelLauncher`"];
	sshCommand = LibraryFunctionLoad["CloudRemoteKernelLauncher", "runSshCommand",
		{"UTF8String", "UTF8String", "UTF8String"}, "Void"];
	sshCommandString = LibraryFunctionLoad["CloudRemoteKernelLauncher", "runSshCommandRetString",
		{"UTF8String", "UTF8String", "UTF8String"}, "UTF8String"];
]


$busywait = 0.05;
$processDiag

SetAttributes[record, HoldFirst]

record[expr_, tag_Symbol] := With[{res=AbsoluteTiming[expr]},
	tag/: time[tag] = time[tag]+First[res];
	tag/: count[tag] = count[tag]+1;
	res[[2]]
]
time[_] = 0.0
count[_] = 0

tags = {runTime, connQTime, connTime, readyQTime, initTime, busyTime}

timings[] := Append[Through[{Identity,time,count}[#]]& /@ tags, {"Total", Total[time/@tags]}]

(* linkhost warning; this is only relevant for direct (non-forwarded) connections, and not in the cloud *)

$linkHostWarning = !TrueQ[Parallel`Static`ClusterEPC]
linkHostWarning[] :=
	If[$linkHostWarning && Length[Select[$MachineAddresses,StringMatchQ[Repeated[DigitCharacter|"."]]]]>1, (* only IPv4 *)
		Message[ LaunchCluster::host, $MachineAddresses ];
		$linkHostWarning=False;
	]

(* create listening socket, ssh into each node and start kernels;
   This will return as soon as all links are connected and LinkReadyQ (with the prompt,
   presumably); This code does not actually talk to the kernels *)

$localhost="localhost";

LaunchCluster[node_String, args___] := LaunchCluster[{node}, args]
LaunchCluster[nodes_, opts:OptionsPattern[]] := LaunchCluster[nodes, 1, opts]

LaunchCluster[nodes0_List, n_Integer?NonNegative, opts:OptionsPattern[]]/; !TrueQ[Parallel`Static`ClusterEPC] :=
Module[{lh, rst, bst, eso, math, user, cluster, cmdfile, serverlink, servername, incoming, ready, kernels,
		time, wt, to, tol, cmd, res, maxn, fwport, ropt, lport, args, nodes },
    {lh,rst,bst,eso,math,user,cluster,tol,fwport} =
    	OptionValue[{LinkHost, RemoteScriptTemplate, BulkSshTemplate, ExtraSshOptions, RemoteKernelCommand, RemoteUsername, ClusterName, Timeout , RemoteForward}];
	If[ListQ[tol], to=First[tol]; tol=Rest[tol], to=tol; tol={}];
	(* allow for individual kernel numbers in node descriptions *)
	nodes = Replace[nodes0, s_String :> {s, n}, {1}];
	maxn = Total[nodes[[All,2]]];
    feedbackObject["name", ClusterKernels[nodes0, n], maxn];
    Clear/@tags;

    (* remote port forwarding canonicalization *)
    If[fwport===True || fwport===Automatic, fwport=$RemoteForwardPort];
    If[!StringQ[fwport], fwport=False ]; (* anything else maps to False *)
    If[StringQ[fwport] && !StringMatchQ[fwport,Repeated[DigitCharacter]],
    	Message[LaunchCluster::portnum, fwport];
    	Return[$Failed]
    ];

	(* create link server socket *)
	If[StringQ[fwport],
		lh = $localhost; (* create server socket on localhost, irrespective of LinkHost *)
		, (* else produce warning if more than one interface *)
	    If[!StringQ[lh],
	    	linkHostWarning[] ];
	];
	args = $linkServerCreateOptions;
	If[StringQ[lh] && StringLength[lh]>0, PrependTo[args,lh]]; (* the host or port@host *)
	Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "Creating server socket with `1`", Inactive[WSTP`LinkServer`LinkServerCreate]@@args];
	serverlink = WSTP`LinkServer`LinkServerCreate@@args;
	Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "Socket created as `1`", serverlink];
	If[FailureQ[serverlink], Return[$Failed]]; (* should already have printed a message *)
	servername = First[serverlink];

	If[StringQ[fwport], (* port forwarding adjustments *)
		lport = StringReplace[servername, p__~~"@"~~__ :> p];
		servername = fwport<>"@"<>$localhost;
		ropt = " -o ExitOnForwardFailure=yes -R "<>fwport<>":"<>$localhost<>":"<>lport;
		eso = eso<>ropt;
	    Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "Port forwarding: `1`", ropt];
	];

    (* write remote script file *)
    cmdfile = OpenWrite[FormatType -> OutputForm, PageWidth -> Infinity]; (* new tmp file *)
    Write[cmdfile, ToString[StringForm[rst, math <> $mathOptions, servername, n]]];
	cmdfile = Close[cmdfile];
	Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "RemoteScript:\n`1`", ToString[StringForm[rst, math <> $mathOptions, servername, n]]];

	(* issue ssh commands; must put themselves into background! *)
	CheckAbort[
		$processDiag = Association[]; (* debugging RunProcess results *)
		  Function[node, 
			cmd=ToString[StringForm[bst, node[[1]], user, servername, cmdfile, eso, node[[2]]]];
			Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "Running command \"`1`\"", cmd];
			record[$processDiag[node] = res = Run[cmd], runTime];
			If[res!=0, Message[LaunchCluster::cmd, cmd, res]];
			res
		  ] /@ nodes
		, (* clean up after abort *)
		Abort[]; $Aborted
	];
	DeleteFile[cmdfile];

	(* now collect ready kernel connections until we got them all or are tired of waiting *)
	time = AbsoluteTime[]; wt = 0;
	ready = {};
	Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "Waiting for `1` connections", maxn];
	CheckAbort[
		While[Length[ready]<maxn && wt<to,
			record[ incoming = WSTP`LinkServer`GetLinks[serverlink], connQTime ];
			If[ Length[incoming] == 0, (* no progress: wait a bit and increment timer *)
				record[Pause[$busywait], busyTime]; wt = AbsoluteTime[] - time;
			  , (* else: reset timer *)
			    ready = Join[ready, incoming];
			    feedbackObject["tick",Length[incoming]];
			    time = AbsoluteTime[]; wt = 0;
			  	If[Length[tol]>1, to=First[tol]; tol=Rest[tol]];
			];
		];
		If[ Length[ready]<maxn, (* too bad *)
			Message[LaunchCluster::timekernels, Length[ready], maxn];
		];
	, (* clean up after abort *)
		Abort[]; $Aborted
	];
	Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "Received `1` connections", Length[ready]];
	(* close server socket *)
	Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "Closing socket `1`", serverlink];
	WSTP`LinkServer`LinkServerClose[serverlink];

	(* turn into kernel objects *)
	If[cluster===Automatic, (* we don't really know the name, make a guess *)
		Which[
			Length[nodes]==1 && StringQ[First[nodes]], cluster=First[nodes],
			Length[nodes]==1 && StringQ[nodes[[1,1]]], cluster=nodes[[1,1]],
			True,             cluster="Generic"
		]
	];
	record[kernels = initLink[ ready, cluster, Flatten[{opts}] ], initTime];
	kernels
]

(* EPC version with librarylink; no remote port forwarding supported *)

LaunchCluster[nodes_List, n_Integer?NonNegative, opts:OptionsPattern[]]/; TrueQ[Parallel`Static`ClusterEPC] :=
Module[{lh, rst, bst, eso, math, user, cluster, cmdfile, serverlink, servername, incoming, ready, kernels,
		time, wt, to, tol, res, maxn=0, args, nname, nn },
    {lh,rst,bst,eso,math,user,cluster,tol} =
    	OptionValue[{LinkHost, RemoteScriptTemplate, BulkSshTemplate, ExtraSshOptions, RemoteKernelCommand, RemoteUsername, ClusterName, Timeout}];
	If[ListQ[tol], to=First[tol]; tol=Rest[tol], to=tol; tol={}];
    feedbackObject["name", ClusterKernels[nodes, n], maxn];
    Clear/@tags;

	(* create link server socket *)
	args = $linkServerCreateOptions;
	If[StringQ[lh] && StringLength[lh]>0, PrependTo[args,lh]]; (* the host or port@host *)
	Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "Creating server socket with `1`", Inactive[WSTP`LinkServer`LinkServerCreate]@@args];
	serverlink = WSTP`LinkServer`LinkServerCreate@@args;
	Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "Socket created as `1`", serverlink];
	If[FailureQ[serverlink], Return[$Failed]]; (* should already have printed a message *)
	servername = First[serverlink];

    (* write remote script file *)
    cmdfile = OpenWrite[FormatType -> OutputForm, PageWidth -> Infinity]; (* new tmp file *)
    Write[cmdfile, ToString[StringForm[rst, math <> $mathOptions, servername, n]]];
	cmdfile = Close[cmdfile];
	Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "RemoteScript:\n`1`", ToString[StringForm[rst, math <> $mathOptions, servername, n]]];

	(* issue ssh commands; must put themselves into background! *)
	CheckAbort[
		  Function[node, 
		  	{nname,nn} = Replace[node, s_String :> {s, n}]; (* allow {node,nn} *)
		  	maxn += nn;
			Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "Running LibraryLink command \"`1`\"",
				Inactive[sshCommand][nname, ToString[nn], cmdfile]];
		  	res = sshCommand[nname, ToString[nn], cmdfile];
			res
		  ] /@ nodes
		, (* clean up after abort *)
		Abort[]; $Aborted
	];
	DeleteFile[cmdfile];

	(* now collect ready kernel connections until we got them all or are tired of waiting *)
	time = AbsoluteTime[]; wt = 0;
	ready = {};
	Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "Waiting for `1` connections", maxn];
	CheckAbort[
		While[Length[ready]<maxn && wt<to,
			record[ incoming = WSTP`LinkServer`GetLinks[serverlink], connQTime ];
			If[ Length[incoming] == 0, (* no progress: wait a bit and increment timer *)
				record[Pause[$busywait], busyTime]; wt = AbsoluteTime[] - time;
			  , (* else: reset timer *)
			    ready = Join[ready, incoming];
			    Do[feedbackObject["tick"],{Length[incoming]}];
			  	time = AbsoluteTime[]; wt = 0;
			  	If[Length[tol]>1, to=First[tol]; tol=Rest[tol]];
			];
		];
		If[ Length[ready]<maxn, (* too bad *)
			Message[LaunchCluster::timekernels, Length[ready], maxn];
		];
	, (* clean up after abort *)
		Abort[]; $Aborted
	];
	Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "Received `1` connections", Length[ready]];
	(* close server socket *)
	Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "Closing socket `1`", serverlink];
	WSTP`LinkServer`LinkServerClose[serverlink];

	(* turn into kernel objects *)
	If[cluster===Automatic, (* we don't really know the name, make a guess *)
		Which[
			Length[nodes]==1 && StringQ[First[nodes]], cluster=First[nodes],
			Length[nodes]==1 && StringQ[nodes[[1,1]]], cluster=nodes[[1,1]],
			True,             cluster="Generic"
		]
	];
	record[kernels = initLink[ ready, cluster, Flatten[{opts}] ], initTime];
	kernels
]


(* destructor; use generic implementation *)

clusterKernel/: Close[kernel_clusterKernel?subQ] := (
	$openkernels = DeleteCases[$openkernels, kernel];
	kernelClose[kernel, True]
)


(* handling short forms of kernel descriptions *)

(** remoteKernelObject[try][Except["localhost"|"local",s_String]]/; StringMatchQ[s,RegularExpression["\\w+(\\.\\w+)*"]] :=
	LaunchRemote[s] **)


(* class name *)

clusterKernelObject[subName] = "Cluster Kernels"


(* raw constructor; several at once *)

(* if a single host name is given, duplicate it *)

initLink[links_List, cluster_String, args_] :=
 Module[{kernels},
 	(* no mutable fields here; set cluster name for all of them *)
 	kernels = Map[ clusterKernel[lk[#1, cluster, args]]&, links ];
 	(* local init *)
 	$openkernels = Join[$openkernels, kernels];
 	(* base class init *)
 	kernelInit[kernels]
 ]

(* single one *)

initLink[link_, args__] := firstOrFailed[ initLink[{link}, args] ]


(* class variable defaults *)
 
sf[s_String] := StringReplace[s, " " -> "\\ "] (* escape spaces *)
qs[s_String] := "\"" <> s <> "\"" (* quote string *)

 (* We cannot really know what the remote kernel's OS is nor where they stashed
   the 'wolfram' script. Assume it is in the same place as on master.
*)

(* the actual kernel launched here; may be useful for $RemoteKernelCommand *)

$LocalKernelCommand = StringReplace[First[$CommandLine], "SystemFiles/Kernel/Binaries/" ~~ __ -> "Executables/MathKernel"];
If[ FileExistsQ[$LocalKernelCommand], $LocalKernelCommand = sf[$LocalKernelCommand],
	$LocalKernelCommand = sf[ToFileName[{$InstallationDirectory, "Executables"}, "wolfram"]]
]


(* best guess for kernel run script, but this is really for the remote nodes! *)
If[ !ValueQ[$RemoteKernelCommand],
		$RemoteKernelCommand = "wolfram"
]

(* format *)

setFormat[clusterKernel, "cluster"]


End[]

Protect[ LaunchCluster, ClusterKernels, clusterKernelObject, clusterKernel ]
Protect[ BulkSshTemplate,ClusterName,ExtraSshOptions,RemoteForward,RemoteKernelCommand,RemoteScriptTemplate,RemoteUsername ]

(* registry *)
addImplementation[clusterKernelObject]
 
EndPackage[]
