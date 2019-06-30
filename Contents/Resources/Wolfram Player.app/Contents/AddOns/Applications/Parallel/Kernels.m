(* :Title: Kernels.m -- Subkernel Support for PCT *)

(* :Context: Parallel`Kernels` *)

(* :Author: Roman E. Maeder *)

(* :Summary:
   Basic process control for parallel evaluation of Mathematica expressions.
 *)

(* :Package Version: 3.0  *)

(* :Mathematica Version: 6+ *)

(* :History:
	Version 2.0 for PCT 2.1, named RemoteKernels.n
	Version 3.0 for PCT 3.0. renamed Kernels.m
*)

(* the package context is normally not on $ContextPath; put the most important symbols into Parallel` *)

BeginPackage["Parallel`Kernels`"]

System`LaunchKernels
System`AbortKernels
System`CloseKernels

(* Close and Abort are documented in SubKernels` *)

(* semi-hidden ones *)

(* class variables *)

System`Kernels
System`$KernelCount

(* the data type is semi-public, for easier subclassing *)
`kernel::usage = "kernel[...] is the data type for kernels."

(* globals *)

System`$KernelID
System`$ConfiguredKernels
(** System`KernelObject (* moved to Parallel`Protected` because of UKCS *) **)

 (* low-level communication *)
 
`send::usage = "send[kernel,cmd] sends cmd for evaluation to the given remote kernel."
`receive::usage = "receive[kernel, True] waits for a result and returns it.
	receive[kernel, False] returns $notReady if no result is available immediately.
	The result received is wrapped in HoldComplete.
	If no evaluation is pending, receive issues a message and returns $Failed."
`receive0::usage = "receive0[kernel] returns a result from kernel and $notReady if no result is available immediately or no evaluation is pending.
	The result received is wrapped in HoldComplete."
`putback::usage = "putback[kernel, result] puts a received evaluation result back so it can be received again. The result should be wrapped in HoldComplete. There can be only one putback."
`poll::usage = "poll[kernel] checks for a callback from a kernel. It returns True if one was serviced. It puts back any evaluation result received."

`$notReady::usage = "$notReady is returned by receive if no result is available."

receive::nothing = "No pending evaluation on `1`, cannot wait to receive anything."
putback::oops = "There is already a putback for `1` (`2`)."

`launchFeedback::usage = "launchFeedback[launchcmd] provides launch feedback."


BeginPackage["Parallel`Developer`"] (* developer context for lesser functionality *)

`LaunchKernel::usage = "LaunchKernel[kernel description, opts...] launches a kernel using SubKernels`New[] and initializes it for Parallel Computing Toolkit.
	LaunchKernel[kernel] clones the given kernel.
	LaunchKernel[] launches a local kernel.
	LaunchKernel[descr, n, opts...] launches n similar kernels."
SyntaxInformation[LaunchKernel] = { "ArgumentsPattern" -> {___} }

`ConnectKernel::usage = "ConnectKernel[subkernel] connects to an open subkernel.
	ConnectKernel[subkernel, KernelID->id] sets the new kernel's ID to id if possible."
SyntaxInformation[ConnectKernel] = { "ArgumentsPattern" -> {_, OptionsPattern[]} }
Options[ConnectKernel] = { KernelID->Automatic }

`CloseError::usage = "CloseError[kernel] closes a failed subkernel."
SyntaxInformation[CloseError] = { "ArgumentsPattern" -> {_} }

`LaunchDefaultKernels::usage = "LaunchDefaultKernels[] launches kernels if necessary and
possible given the default configuration admits kernels. Returns True, if at least two kernels
are running, False otherwise."
SyntaxInformation[LaunchDefaultKernels] = { "ArgumentsPattern" -> {} }

(* selectors etc. *)

`KernelID::usage = "KernelID[kernel] gives the ID number of kernel, its value of $KernelID."
SyntaxInformation[KernelID] = { "ArgumentsPattern" -> {_} }

`SubKernel::usage = "SubKernel[kernel] gives the subkernel object underlying kernel."
SyntaxInformation[SubKernel] = { "ArgumentsPattern" -> {_} }

(* KernelSpeed is defined in SubKernels`, in the System` context *)

`KernelName::usage = "KernelName[kernel] gives the name (usually machine name) of the given subkernel.
	KernelName[kernel]=name sets the name."
SyntaxInformation[KernelName] = { "ArgumentsPattern" -> {_} }

`EvaluationCount::usage = "EvaluationCount[kernel] is the number of pending evaluations on kernel.
	EvaluationCount[] is the total number of evaluations on all kernels."
SyntaxInformation[EvaluationCount] = { "ArgumentsPattern" -> {_.} }

`ClearKernels::usage = "ClearKernels[] resets all subkernels and clears all exported definitions and shared variables."
SyntaxInformation[ClearKernels] = { "ArgumentsPattern" -> {} }

`KernelFromID::usage = "KernelFromID[id] gives the kernel with the given ID."
SyntaxInformation[KernelFromID] = { "ArgumentsPattern" -> {_} } (* Listable *)

(* Description::usage = "Description[kernel] gives the subkernel description needed to launch it." *)

(* LinkObject and MachineName are documented in SubKernels` *)

`$InitCode::usage = "ReleaseHold[$InitCode] is run on every new subkernel for user initialization."

EndPackage[]


BeginPackage["Parallel`Protected`"] (* semi-hidden methods, aka protected *)

`KernelObject (* moved here because of UKCS *)

`PacketHandler::usage = "PacketHandler[packet, kernel] is called by receive to handle packets other than evaluation results received from a remote kernel."
SyntaxInformation[PacketHandler] = { "ArgumentsPattern" -> {_, _} }

`slaveQ::usage = "slaveQ[kernel] is True, if kernel is available."
`kernelQ::usage = "kernelQ[kernel] is the kernel type predicate."

`$kernels::usage = "$kernels is the raw list of open subkernels."
`$sortedkernels::usage = "$sortedkernels is $kernels sorted by speed."
`$kernelsIdle::usage = "$kernelsIdle is True, if kernels are idle."
`$seQ::usage = "$seQ is True, if no kernels are available."
`$FEQ::usage = "$FEQ is True, if we are running under the frontend."
`$maxProcessQueueSize::usage = "$maxProcessQueueSize is the constant max size of a process queue."
`dead::usage = "dead is a tag to signal a kernel failure."
`tryRelaunch::usage = "tryRelaunch[] tries to revive dead kernels."
`AppendInitCode::usage = "AppendInitCode[code] is for load-time setup of init code."
`AddInitCode::usage = "AddInitCode[code] sends code to running subkernels and uses AppendInitCode to remember it."
`kernelInitialize;
`registerCloseCode;
`registerAbortCode;
`registerResetCode;
`registerReplyHead;

(* some private selectors are really protected ones; we should make them available to all dependent packages *)

{readyQ} (* stage 1 *)
{neval} (* stage 2 *)
{enqueue, dequeue, lqueue, rqueue, processes} (* stage 3 *)

(* setting up autolaunching of kernels *)

`declareAutolaunch::usage = "declareAutolaunch[s1,...] declares the si as symbols whose first use causes kernel autolaunching."
`doAutolaunch::usage = "doAutolaunch[True/False] launches default kernels with optional feedback."
`clearAutolaunch::usage = "clearAutolaunch[] clears autolaunching for all symbols declared with declareAutolaunch[]"

`kernelEvaluate::usage = "kernelEvaluate[cmd, ({kernels..}] evaluates cmd on all kernels."

(* same as in SubKernels` *)

General::failinit = "`2` of `1` kernels failed to initialize."

`deleteFailed
`firstOrFailed

EndPackage[]

(* messages *)

Kernels::rdead = "Subkernel connected through `1` appears dead."
Kernels::noid = "No parallel kernel with ID `1` found."
LaunchKernels::clone = "Kernel `1` resurrected as `2`."
LaunchKernels::final = "Cloning kernel `1` failed."
LaunchKernels::nodef = "Some subkernels are already running. Not launching default kernels again."
LaunchKernels::noconf = "No kernels configured for automatic launching."
LaunchKernels::launch = "Launching `1` kernels..."
LaunchKernels::unicore = "The default parallel kernel configuration does not launch any kernels on a single-core machine.\
	Use LaunchKernels[n] to launch n kernels anyway."
PacketHandler::default = "Unhandled packet `1` received and discarded from `2`."
LaunchKernels::nosub = "Kernel `1` has $SubKernel set to False and cannot be used as a subkernel."
LaunchKernels::forw = "Kernel version `1` is newer than master kernel."
LaunchKernels::obso = "Subkernel version `1` is not supported."
LaunchKernels::prune = "Dropping `1` failed subkernels, out of `2`."

(* configurable launch feedback *)
LaunchKernels::feedbackFE = "Launching `1` `3`/`4`";
LaunchKernels::feedbackSA = "Launching kernels...";


(* master side of symbols in Parallel`Client` *)
Parallel`Client`HoldCompound
Parallel`Client`$ClientLanguageVersion


Begin["`Private`"]

`$PackageVersion = 3.0;
`$thisFile = $InputFileName

Needs["SubKernels`"]; Needs["SubKernels`Protected`"]

(* Parallel`Debug`MathLink is set up in SubKernels`, which is required for us, so we do not need it again *)
Parallel`Debug`Private`RegisterTrace[ Parallel`Debug`SendReceive, "SendReceive is a tracer that triggers when expressions are sent to remote kernels and results are received from remote kernels." ]


protected = Unprotect[TextPacket, ExpressionPacket, MessagePacket, SuspendPacket]

$FEQ = ValueQ[$FrontEnd] && ($FrontEnd =!= Null) (* under FE? *)

$kernels = {}; (* list of active kernels *)
dead (* tag for throwing kernel errors *)
`$kernelID = 0; (* last used ordinal *)

$seQ; (* True if Length[$kernels]==0, for faster sequential tests *)
$sortedkernels; (* sorted version, by speed *)

(* call this everytime you change $kernels *)
sortkernels[] := ($sortedkernels = Reverse[SortBy[$kernels, {speed}]]; $seQ = Length[$kernels]==0;)

sortkernels[]

(* timeout for external commands and initial LinkWrite to new kernels;
   recovery modes are set through SetSystemOptions["ParallelOptions" -> "MathLinkTimeout" -> ]
   allowed values nonegative numbers; reflected in Parallel`Settings`$MathLinkTimeout *)

(* pause after sending Abort[] to remote kernel, before checking for a reply
   recovery modes are set through SetSystemOptions["ParallelOptions" -> "AbortPause" -> ]
   allowed values nonegative numbers; reflected in Parallel`Settings`$AbortPause  *)

SetAttributes[kernel, HoldAll] (* data type *)

(* the data type stage 1:
	kernel[bk[sub, id, name], ___] (pattern kernel[bk[sub_, id_, name_, ___], ___] )
	
	sub:		subkernel object
	id:		numbering, 1..., <=0 if not valid, mutable
	name:	kernel's human-readable name, default MachineName[sub], mutable

 *)

SetAttributes[`bk, HoldAllComplete] (* the head for the base class data *)

kernel/: subKernel[kernel[bk[sub_, ___], ___]] := sub
kernel/: kid[kernel[bk[sub_, id_, ___], ___]] := id
kernel/: setId[kernel[bk[sub_, id_, ___], ___], r_Integer] := (id = r)
kernel/: markFailed[kernel[bk[sub_, id_, ___], ___]] := If[id>0, id=-id] (* remember as negative val *)
kernel/: name[kernel[bk[sub_, id_, name_, ___], ___]] := name
kernel/: setName[kernel[bk[sub_, id_, name_, ___], ___], r_] := (name = r)

If[TrueQ[Parallel`Debug`$Debug],
	(#[args___] := Throw[{#, args}])& /@ {subKernel, kid, name}
]

(* delegate *)
kernel/: linkObject[ kernel_kernel ]  := LinkObject[subKernel[kernel]]
kernel/: machineName[ kernel_kernel ] := MachineName[subKernel[kernel]]
kernel/: readyQ[ kernel_kernel ]  := kernelReadyQ[subKernel[kernel]]

(* constructor *)
stage1[sub_] := Module[{id=0, name=MachineName[sub]}, kernel[bk[sub, id, name]] ]


(* extension of kernel[...] stage 2:
	kernel[super__, ek[neval, pb, rd], ___] (pattern kernel[__, ek[nev_, pb_, rd_, ___], ___] )

	neval:		number of pending evals (send operations), mutable
	pb:		putback data, mutable; Null means no data
	rd:		True if remote is in LinkRead state, mutable

  the neval count must include any putback data present
*)

SetAttributes[`ek, HoldAllComplete] (* the head for the subtype data *)

kernel/: neval[kernel[__, ek[nev_, pb_, rd_, ___], ___]] := nev
neval/: HoldPattern[++neval[kernel[__, ek[nev_, pb_, rd_, ___], ___]]] := ++nev
neval/: HoldPattern[--neval[kernel[__, ek[nev_, pb_, rd_, ___], ___]]] := --nev
kernel/: cleareval[kernel[__, ek[nev_, pb_, rd_, ___], ___]] := (nev=0)
kernel/: pb[kernel[__, ek[nev_, pb_, rd_, ___], ___]] := pb
kernel/: setpb[kernel[__, ek[nev_, pb_, rd_, ___], ___], newpb_:Null] := (pb = newpb)
kernel/: pbQ[kernel[__, ek[nev_, pb_, rd_, ___], ___]] := (pb =!= Null)
kernel/: rd[kernel[__, ek[nev_, pb_, rd_, ___], ___]] := rd
kernel/: rdT[kernel[__, ek[nev_, pb_, rd_, ___], ___]] := (rd=True)
kernel/: rdF[kernel[__, ek[nev_, pb_, rd_, ___], ___]] := (rd=False)

If[TrueQ[Parallel`Debug`$Debug],
	(#[args___] := Throw[{#, args}])& /@ {neval, pb, rd}
]

(* extension constructor *)
stage2[ kernel[super___] ] :=
Module[{res, nev=0, pb=Null, rd=False},
	res = kernel[super, ek[nev, pb, rd]];
	cleareval[res];
	setpb[res];
	rdF[res];
	res
]


(* extension of kernel[...] stage 3:
	kernel[super__, sk[q, n0, n1], ___] (pattern kernel[__, sk[q_, n0_, n1_, ___], ___] )

	q:	FIFO array of queued processes, mutable
	n0:	first item, mutable
	n1:	next free item, mutable

	FIFO queue methods:
		rqueue:		initialize fresh queue
		enqueue:	enter job into queue
		dequeue:	remove job and return it
		lqueue:		length of queue
		processes:		list of processes in queue
*)

(* the internal process queue is a fixed-length circular buffer *)
(* allow a backdoor to set the max queue size before loading the toolkit *)

If[!(ValueQ[$maxProcessQueueSize] && IntegerQ[$maxProcessQueueSize] && $maxProcessQueueSize>1),
    $maxProcessQueueSize = 8 ]

SetAttributes[pqh, HoldAllComplete] (* process queue head *)
SetAttributes[`sk, HoldAllComplete] (* the head for the subtype data *)

(* no error checking for queue overflow/underflow *)
With[{n = $maxProcessQueueSize, newqueue = pqh@@Table[Null, {$maxProcessQueueSize}]},
  kernel/: rqueue[kernel[__, sk[q_, n0_, n1_, ___], ___]] := (q = newqueue; n0=n1=1;);
  kernel/: enqueue[kernel[__, sk[q_, n0_, n1_, ___], ___], job_] := (q[[n1]]=job; n1=Mod[n1,n]+1;);
  kernel/: dequeue[kernel[__, sk[q_, n0_, n1_, ___], ___]] := With[{job=q[[n0]]}, q[[n0]]=Null; n0=Mod[n0,n]+1; job];
  kernel/: lqueue[kernel[__, sk[q_, n0_, n1_, ___], ___]] := Mod[n1-n0, n];
  kernel/: processes[k:kernel[__, sk[q_, n0_, n1_, ___], ___]] := List@@Take[RotateLeft[q, n0-1], lqueue[k]];
]

If[TrueQ[Parallel`Debug`$Debug],
	(#[args___] := Throw[{#, args}])& /@ {lqueue}
]

(* extension constructor *)
stage3[ kernel[super___] ] :=
Module[{res, q, n0, n1},
	res = kernel[super, sk[q, n0, n1]];
	rqueue[res];
	res
]

(* validity predicates *)

slaveQ[_] := False
kernel/: slaveQ[kernel[bk[sub_, id_, ___], ___]] := id>0
kernelQ[_] := False
kernel/: kernelQ[kernel[bk[sub_, id_, ___], ___]] := IntegerQ[id]

(* raw constructor  *)

initKernel[sub_]:=
    Module[{res = sub},
      res = stage1[res];
      res = stage2[res];
      res = stage3[res];
      res
    ]

(* kernel formatting for tracing *)

kernel/: traceform[k_kernel] := StringForm["kernel `1`", KernelID[k]]


(* public constructors *)

(* to make sending unevaluated expressions easier, Parallel`Client`HoldCompound has attribute
   HoldAllComplete on the master, but turns into CompoundExpression on the subkernel side.
   Note that you must always use its full context name, or you can use the shortcut, provided it's
   evaluated! The subkernel definitions are in Client.m *)

SetAttributes[Parallel`Client`HoldCompound, {HoldAllComplete, Protected}]
holdCompound = Parallel`Client`HoldCompound

(* init code. There are two lists, $clientCode and $clientOriginalCode.
   Both are ordinary lists, whose elements should be expressions wrapped
   in Parallel`Client`HoldCompound (or holdCompound).
   $clientOriginalCode init code survives ClearKernels[].
   The elements of $clientCode are sent in separate batches, to avoid
   premature creation of symbols (think of Needs[]).
   Any return values of the code batches are suppressed so as not to take up bandwidth
*)

$clientOriginalCode = $clientCode = { }

(* AppendInitCode is for load-time setup of init code. should obviously be permanent *)

AppendInitCode[code_, permanent_:True] := (
	$clientCode = Append[$clientCode, code];
	If[ permanent, $clientOriginalCode = Append[$clientOriginalCode, code] ]
)

(* AddInitCode[code] sends code to running subkernels and uses AppendInitCode to remember it.
   It is intended for dynamic init code that is not set up at load time, such as in DistributeDefinitions *)

AddInitCode[code_, permanent_:False] := (
	kernelEvaluate[code;Null]; (* for current kernels *)
	AppendInitCode[ code, permanent ]; (* for new ones *)
	Null
)


(* reverse mapping of processorID to kernel *)
processor[_] := $Failed


(* connect to open subkernel and init it *)


firstOrFailed[l_List] := If[ Length[l]<1, $Failed, First[l] ]

(* prune any failed kernel objects. Search for $Failed in list of results *)

deleteFailed[kernels_, results_, msghead_:Null] :=
With[{pos = Position[results, $Failed]},
	If[Length[pos]>0,
		If[ msghead=!=Null, Message[msghead::failinit, Length[kernels], Length[pos]]];
		Close /@ subKernel /@ Extract[kernels,pos];
		Delete[kernels, pos]
	, (* else all is well *)
		kernels
	]
]

(* parallel version *)

Parallel`Kernels`marker (* unique token *)

ConnectKernel[subs:{___?subQ}, OptionsPattern[]] :=
Module[{new, res, fails},
	Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "Connecting to subkernels `1`", subs];

	new = initKernel /@ subs; (* data object *)
	(* TODO handle KernelID option? *)
	(* give out IDs *)
	setId[#, ++$kernelID]& /@ new;

	(* check subkernel language version and make sure it *is* a subkernel *)
	Block[{Parallel`Client`$ClientLanguageVersion = Null, System`Parallel`$SubKernel = Null},
		kernelInitialize[marker[{System`Parallel`$SubKernel, Parallel`Client`$ClientLanguageVersion, $VersionNumber}], new];
		res = kernelGetPattern[new, ReturnPacket[marker[{__}]]];
		With[{pos = Position[res, $Failed]}, (* removed failed ones, but preserve structure of res *)
		If[ Length[pos]>0,
			Message[ConnectKernel::failinit, Length[res], Length[pos]];
			res = Delete[res, pos]; {new,fails} = {Delete[new,pos],Extract[new,pos]};
			Close /@ subKernel /@ fails; ]];
		res = res[[All, 1, 1, 1]]; (* strip marker *)
	];
	res = MapThread[ checkVersion, {new,res}];
	new = deleteFailed[new, res, ConnectKernel];

	(* processor id, a poor man's ParallelDispatch, but do not wait for results *)
	res = Function[{k}, With[{id=KernelID[k]}, processor[id] = k; kernelInitialize[$KernelID=id;Protect[$KernelID];, k]]] /@ new;

	(* init code registries *)
	Function[e, kernelSwallow[new]; res = kernelInitialize[e;, new]] /@ $clientCode;

	(* final sanity check, and swallow all pending results *)
	kernelInitialize[marker[], new];
	res = kernelGetPattern[new, ReturnPacket[marker[]]];
	new = deleteFailed[new, res, ConnectKernel];

	(* user init code; use normal evaluate here, to allow for callback handling *)
	If[ValueQ[$InitCode], With[{clientCode=$InitCode}, res = kernelEvaluate[ ReleaseHold[clientCode];, new ]]; ];

	$kernels = Join[$kernels, new]; sortkernels[]; (* add to list of kernels *)

	Parallel`Debug`Private`RemoteKernelInit /@ new; (* WWB debugging hook *)  
	Parallel`Parallel`Private`initDistributedDefinitions[new]; (* define initial distributed symbols *)

	checkCloudCredentials[]; (* check whether we need to set up credential forwarding; needs to go somwhere *)

	new
]

(* just in case a subkernel constructor returns junk; filter out everything that is not a subkernel *)

ConnectKernel[misc_List, opts___] :=
	With[{good = Select[misc, subQ]},
		Message[LaunchKernels::prune, Length[misc]-Length[good],Length[misc]];
		ConnectKernel[good, opts]
	]

ConnectKernel[sub_?subQ, opts___] := firstOrFailed[ ConnectKernel[{sub}, opts] ]

(* anything else is an error *)

ConnectKernel[___] := $Failed

(* version checker; return $Failed for bad ones, True of good ones *)
checkVersion[sub_, {subKernel_, subLanguageVersion_, subMathVersion_}] :=
	Which[
		subKernel === Null, (* unknown on subkernel: V6 or earlier is no longer supported *)
			Message[LaunchKernels::obso, subMathVersion];
			$Failed,
		!TrueQ[subKernel], (* False or junk *)
			Message[LaunchKernels::nosub, sub];
			$Failed,
		TrueQ[subLanguageVersion === Null], (* 7.0 *)
			Message[LaunchKernels::obso, subMathVersion];
			$Failed,
		TrueQ[subLanguageVersion < Parallel`Private`$ParallelLanguageVersion],
			Needs["Parallel`OldClient`"];
			Parallel`OldClient`initOldKernel[sub, subLanguageVersion];
			True,
		TrueQ[subLanguageVersion > Parallel`Private`$ParallelLanguageVersion], (* in the future, not supported *)
			Message[LaunchKernels::forw, subMathVersion];
			$Failed,
		True, (* OK: same parallel language version *)
			True
	]

checkVersion[__] := $Failed

(* the master kernel subkernel object, handle with care *)

(*** This does not work ...
Needs["SubKernels`LinkKernels`"]
`masterproto = "TCPIP"

ConnectMaster[] := Module[{sub, tap, new},
	tap = LinkCreate[LinkProtocol -> masterproto];
	MathLink`AddSharingLink[tap, AllowPreemptive -> True];
	sub = ConnectLink[tap[[1]], LinkProtocol -> masterproto];
	If[!subQ[sub], Return[$Failed]];
	(* shortened form of ConnectKernel above *)
	new = initKernel[sub]; (* data object *) (* TODO find out why this hangs *)
	setID[new, 0]; (* patch it up *)
	If[new===$Failed, Return[new]];
	Parallel`Client`Private`$link=tap;
	MathLink`LinkSetPrintFullSymbols[tap, True];
    With[{id=KernelID[new]},
      	  processor[id] = new; (* reverse id mapping *)
    ];
    new
]
***)

(* general init codes, anything not that urgent that it must go into ConnectKernel[] above *)
(* note that these codes are resent on ClearKernels[] *)

(* fix random seeding *)
(* SeedRandom[$KernelID*Round[AbsoluteTime[]/$TimeUnit]] *)

(* set $Context to avoid new system symbols appearing in Global`; this is bug 214819 *)
(* $Context = "System`"; change reverted (bug 257029) so bug 214819 is open again *)

(* for performance, we collect all of these in a single batch *)

AppendInitCode[holdCompound[
	SeedRandom[$KernelID*Round[AbsoluteTime[]/$TimeUnit]]
]]

(* open subkernel(s) given a description and possibly other arguments, as well as our options *)
(* NewKernels[] may also return a list of subkernels *)


LaunchKernel[descr__, opts:OptionsPattern[]] :=
Module[{sub},
	clearAutolaunch[];
    Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "Opening kernel with `1`", HoldForm[NewKernels[descr, opts]]];
	sub = NewKernels[descr, opts];
	ConnectKernel[sub, opts]
]

(* default local kernel *)
LaunchKernel[] := LaunchKernel["localhost"]

(* clone; try to reuse kernelID *)
(* LaunchKernel[sub_?subQ] := ConnectKernel[Clone[sub]] *)
LaunchKernel[ker_kernel] := (
	Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "Cloning kernel `1`", ker];
	ConnectKernel[Clone[subKernel[ker]], KernelID->-kid[ker]]
)

(* multilauncher, public smart constructor *)

LaunchKernels[l_List, opts:OptionsPattern[]] := Flatten[ launchFeedback[LaunchKernel[#, opts]& /@ l] ]

(* handling of LaunchKernels[n] is delegated to the heuristic constructor of LocalKernels` *)

(* other descriptions are passed to LaunchKernel[] *)

LaunchKernels[descr__, opts:OptionsPattern[]] := launchFeedback[LaunchKernel[descr, opts]]

(* the empty forms launch only if no kernels have already been launched *)

LaunchKernels[opts:OptionsPattern[]]/;$KernelCount>0 := ( Message[LaunchKernels::nodef]; $Failed )

(* be friendly to single-core systems *)

LaunchKernels[opts:OptionsPattern[]]/;ListQ[$ConfiguredKernels] && Total[KernelCount/@$ConfiguredKernels]==0 && $ProcessorCount==1 :=
  (Message[LaunchKernels::unicore];
   LaunchKernels[$ConfiguredKernels, opts]
  )

LaunchKernels[opts:OptionsPattern[]]/;ListQ[$ConfiguredKernels] :=
	LaunchKernels[$ConfiguredKernels, opts]


(* parallel code helper *)

LaunchDefaultKernels[] := (
	If[$KernelCount==0, (* try to launch defaults *)
			LaunchKernels[]
	];
	(* see whether it worked *)
	$KernelCount>1
)

(* forward cloud credentials *)

`$cloudState = Automatic; (* whether we have already set up forwarding: True|False|Automatic *)
If[!ValueQ[Parallel`Settings`$ForwardCloudCredentials],
	Parallel`Settings`$ForwardCloudCredentials=False ] (* built in default; the normal default is in Kernel/Preferences.m *)

checkCloudCredentials[] := Which[
	(* been there, done that *)
	$cloudState===True || $cloudState===False, $cloudState,

	(* preference is off, don't do anything, but test again later *)
	!TrueQ[Parallel`Settings`$ForwardCloudCredentials], False,

	(* automatic, but not logged in: test again later *)
	!TrueQ[$CloudConnected], False,
	
	(* we are logged in; try to set up forwarding *)
	True, $cloudState = setupCloudForwarding[]
]

setupCloudForwarding[] := With[{cred = CloudObject`Internal`GetAuthentication[]},
	If[ListQ[cred], 
		AddInitCode[ holdCompound[CloudObject`Internal`SetAuthentication@@cred] ];
		True
	  , False (* didn't work; don't try again *)
	]
]


(* destructor, no inheritance *)

(* destructor code registry. The code is called as a function with kernel as arg *)

$closeCode = {}
registerCloseCode[code_] := AppendTo[$closeCode, code]

kernel/: Close[kernel_kernel?kernelQ] := (
	If[ kid[kernel]>0,
	    AbortProtect[With[{i=kid[kernel]},
	      $kernels = DeleteCases[$kernels, k_/;kid[k]==i, {1}, 1]; (* preserve order *)
	      sortkernels[];
	      markFailed[kernel];
	      Unset[processor[i]];
	      checkIdle;
	    ]];
 	    Through[$closeCode[kernel]]; (* registered close code *)
	];
    Close[subKernel[kernel]]; (* clean up the subkernel instance *)
    (* do not bother to clear instance variables *)
    kernel
  )

(* public, give kernel, id, or list *)

CloseKernels[kernel_kernel] := Close[kernel]
CloseKernels[id_Integer?Positive] := CloseKernels[KernelFromID[id]]
CloseKernels[$Failed] := $Failed
CloseKernels[l_List] := CloseKernels /@ l
CloseKernels[] := (CloseKernels[$kernels])


(* relaunch failed kernels *)

(* is enabled through SetSystemOptions["ParallelOptions" -> "RelaunchFailedKernels" -> ]
   allowed values are True/False; reflected in Parallel`Settings`$RelaunchFailedKernels *)

`$recentlyFailed = {}; `relaunching = False;

(* clean up after a linkRead/Write failure *)

CloseError[kernel_kernel?kernelQ] := (
	Close[kernel];
	If[Parallel`Settings`$RelaunchFailedKernels && !MemberQ[$recentlyFailed,kernel], AppendTo[$recentlyFailed, kernel]];
    kernel
)

(* try to clone *)

tryRelaunch[]/; (!Parallel`Settings`$RelaunchFailedKernels || $recentlyFailed==={}) := Null

tryRelaunch[] := If[!relaunching, Block[{relaunching=True}, (* avoid recursion *)
  With[{corpses=$recentlyFailed},
	$recentlyFailed={}; (* clear first, in case of master kernel aborts *)
	Scan[Function[corpse, Module[{new},
		new = LaunchKernel[corpse];
		If[slaveQ[new], Message[LaunchKernels::clone, corpse, new], Message[LaunchKernels::final, corpse]];
	]], corpses]; (* try to clone *)
]]]


(* reset and check *)

(* basic data type *)
reset[kernel_kernel] :=
Module[{res},
	If[ pbQ[kernel], setpb[kernel]; --neval[kernel] ]; (* clean up putback first *)
	If[ neval[kernel]>0,
            res = Abort[kernel];
            , (* else simply check that it is ok *)
            res = kernelFlush[subKernel[kernel]];
 	];
	If[ res===$Failed, Message[Kernels::rdead, kernel]; CloseError[kernel] ];
	rdF[kernel];
	rqueue[kernel];
	kernel
]

(* higher level; should do it in parallel *)

(* registry for package specific reset code. Note that there is also an abort code registry *)

$resetCode = holdCompound[];
SetAttributes[registerResetCode, HoldFirst]
registerResetCode[code_] := ($resetCode = Append[$resetCode, Unevaluated[code]])

AbortKernels[] :=
Module[{},
    CompoundExpression @@ $resetCode; (* registered reset code *)
    reset /@ $kernels;
    tryRelaunch[];
    $kernels
]


(* ClearSlaves: put things back into pristine state *)

(* code to run on the master kernel, other PCT packages can register such code through registerClearCode[] *)

`$clearCode = holdCompound[];

SetAttributes[registerClearCode, HoldFirst]

registerClearCode[code_] :=
	($clearCode = Append[$clearCode, Unevaluated[code]])

ClearKernels[] := 
  Module[{res},
    AbortKernels[]; (* flush and check, close if it failed *)

    (* registered package cleanup handlers for master kernel *)
    CompoundExpression @@ $clearCode; 

    (* reset $Path to saved value to forget loaded packages *)
    kernelInitialize[ $ContextPath = Parallel`Client`Private`$cp; ];

	(* processor id, a poor man's ParallelDispatch *)
	res = Function[{k}, With[{id=KernelID[k]}, kernelInitialize[Unprotect[$KernelID];$KernelID=id;Protect[$KernelID];, k]]] /@ $kernels;

    (* reinit, see constructor *)
    $clientCode = $clientOriginalCode; (* forget all collected junk *)
	Function[e, kernelSwallow[$kernels]; res = kernelInitialize[e;, $kernels]] /@ $clientCode;

	(* final sanity check, and swallow all pending results *)
	kernelInitialize[marker[], $kernels];
	res = kernelGetPattern[$kernels, ReturnPacket[marker[]]];

	With[{pos = Position[res, $Failed]}, (* prune failures by hand, for proper call of destructor *)
	  If[ Length[pos]>0,
		Message[ClearKernels::failinit, Length[$kernels], Length[pos]];
		CloseError /@ Extract[$kernels,pos]; ]];

	(* user init code, see constructor *)
	If[ValueQ[$InitCode],  With[{clientCode=$InitCode}, res = kernelEvaluate[ ReleaseHold[clientCode];, $kernels ]]; ];

    sortkernels[]; $kernels
  ]


(* public *)

prot = Unprotect[MachineName, KernelSpeed]

SetAttributes[{KernelName, KernelID, KernelSpeed, SubKernel, KernelFromID, EvaluationCount}, Listable]

kernel/: MachineName[k_kernel] := machineName[k]
kernel/: KernelName[k_kernel] := name[k]
KernelName[___] := Null
KernelName/: (KernelName[k_] = new_)/;Head[k]===kernel := setName[k, new]
kernel/: LinkObject[k_kernel] := linkObject[k]
kernel/: KernelID[k_kernel] := Max[kid[k], 0] (* map negative values to 0 *)
kernel/: SubKernel[k_kernel] := subKernel[k]
kernel/: EvaluationCount[k_kernel] := neval[k]

(* speed: delegate *)
kernel/: KernelSpeed[kernel_kernel] := KernelSpeed[subKernel[kernel]]
KernelSpeed/: (KernelSpeed[kernel_] = new_)/; kernelQ[kernel] && new>0 := KernelSpeed[subKernel[kernel]]=new
(* kernel/: Description[kernel_kernel?kernelQ] := SubKernels`Description[Kernel[kernel]] (* delegate *) *)

(* map IDs to kernels, unknown ones turn into $Failed *)

KernelFromID[id_Integer] := (If[processor[id]===$Failed, Message[Kernels::noid, id]]; processor[id])


Kernels[] := $kernels (* read only public value *)
$KernelCount := Length[$kernels]

(* make it available to subkernels, too. shared variable defs are in Client.m *)

registerPostInit[ Parallel`Protected`declareSystemVariable[$KernelCount] ]


EvaluationCount[] := Total[EvaluationCount /@ $kernels]


Protect[Evaluate[prot]]


(* linkRead/linkWrite read/write using the subKernels interface kernelRead/kernelWrite;
   these functions throw an error on failure *)

(* actually, there can be only one addtl arg in linkRead, but how to express that? *)

SetAttributes[{linkWrite,linkWriteRaw}, {HoldRest,SequenceHold}]

(* make them more robust, to ignore failed kernels *)
linkReadRaw[$Failed, ___]  := $Failed
linkWriteRaw[$Failed, ___] := $Failed

linkReadRaw[kernel_, h___] := kernelRead[subKernel[kernel], h]
linkWriteRaw[kernel_, expr_] := kernelWrite[subKernel[kernel], expr]

(* these versions throw an error on failure *)

linkRead[kernel_, h___] := With[{res = linkReadRaw[kernel, h]},
	If[res===$Failed, Throw[$Failed,dead], res]
]
linkWrite[kernel_, expr_] := With[{res = linkWriteRaw[kernel, expr]},
	If[res===$Failed, Throw[$Failed,dead], res]
]

(* send and receive: keep track of pending evaluations *)

(* detect when all kernels are idle; we need to check at all send/receive operations [but not poll/putback] *)
(* for Dynamic's benefit, we should only assign to the variable if we actually change it *)

$kernelsIdle = True; (* there aren't any at this point *)
notIdle := If[$kernelsIdle, $kernelsIdle=False] (* if known to not be idle *)
(* is this faster: notIdle/;$kernelsIdle = (...) ? *)
checkIdle := With[{idle = EvaluationCount[] == 0}, If[$kernelsIdle=!=idle, $kernelsIdle=idle]]

(* debug: suppress the expression sent in some trace outputs *)

With[{sanitize = {Parallel`Combine`Private`haha[]:>Sequence[]}}, (* output cleanup *)
Which[
	TrueQ[Parallel`Debug`$SystemDebug], (* show the full ugly truth *)
		hideValue[expr_] = expr,
	TrueQ[Parallel`Debug`$Debug], (* hide ugly internal messes *)
		hideValue[expr_] := If[ TrueQ[Parallel`Debug`Private`$hideVals], "-internal value-", expr /. sanitize],
	True, (* no debugging, no tracing *)
		hideValue = Identity
]]

(* for use in PacketHandlers, sane version of System`Write alias LinkWrite *)

kernel/: Write[k_kernel, expr_] := Catch[linkWrite[k, expr], dead, (Message[Kernels::rdead, k]; CloseError[k]; $Failed)&]


SetAttributes[{send, `sendCatch}, {HoldRest,SequenceHold}]

(* throws `dead *)

send[kernel_kernel, expr_] := (
    Parallel`Debug`Private`trace[Parallel`Debug`SendReceive, "Sending to `1`: `2` (q=`3`)", traceform[kernel], hideValue[HoldForm[expr]], neval[kernel]];
    ++neval[kernel]; notIdle; (* counter of pending evaluations *)
    If[ValueQ[Parallel`Debug`Private`RemoteEvaluateWrapper],
    	With[{wrapper=Parallel`Debug`Private`RemoteEvaluateWrapper},
    		linkWrite[kernel, EvaluatePacket[wrapper[expr]]]
    	]
    , (* else no wrapper *)
    	linkWrite[kernel, EvaluatePacket[expr]]
    ];
    kernel
)

(* handle the error, but do not try to close kernel *)
sendCatch[kernel_kernel, expr_] := Catch[send[kernel,expr], dead, (Message[Kernels::rdead, kernel]; $Failed)&]

(* receive returns its result wrapped in HoldComplete *)

(* if send fails, the argument of receive may not be a valid kernel, but $Failed *)

receive[$Failed, ___] := $Failed

(* nothing to receive? Complain *)

receive[kernel_, wait_:True]/; neval[kernel]==0 := (
	Message[receive::nothing, kernel];
	$Failed
)

(* if putback is available, dish it out *)

receive[kernel_, wait_:True]/; pbQ[kernel] :=
    With[{res = pb[kernel]},
	AbortProtect[--neval[kernel]; checkIdle; setpb[kernel]];
	Parallel`Debug`Private`trace[Parallel`Debug`SendReceive, "Delivering putback from `1`: `2` (q=`3`)", traceform[kernel], hideValue[HoldForm@@res], neval[kernel]];
	res
    ]

(* we assume that if CheckAbort[linkRead[]] fires, we did not read anything *)

receive[kernel_, wait_:True] :=
 AbortProtect[
    While[ wait || readyQ[kernel], (* we want to catch failure, too *)
        Replace[CheckAbort[test=linkRead[kernel, Hold], Abort[];Return[$Aborted]], {
            Hold[ReturnPacket[e___]] :> (--neval[kernel]; checkIdle; Parallel`Debug`Private`trace[Parallel`Debug`SendReceive, "Receiving from `1`: `2` (q=`3`)", traceform[kernel], hideValue[HoldForm[e]], neval[kernel]]; Return[HoldComplete[e]]),
            Hold[packet_] :> (PacketHandler[packet, kernel]; If[!wait, Return[$notReady]]),
            junk_ :> Throw[junk] (* should not happen *)
        }]; (* dispatch *)
    ];
    (* no waiting: nothing there *)
    $notReady
 ]

(* handle the error, but do not try to close kernel *)

receiveCatch[k_, args___] := Catch[receive[k, args], dead, (Message[Kernels::rdead, k]; $Failed)&]

(* special version for Parallel`queueRun[] *)

receive0[kernel_]/; neval[kernel]==0 := $Failed
receive0[kernel_] := receive[kernel, False]

(* poll *)

poll[kernel_]/; pbQ[kernel] := False (* must not read anything again before collecting putback *)

poll[kernel_] :=
 AbortProtect[ Catch[ 
    If[ readyQ[kernel], (* we want to catch failure, too *)
        Replace[CheckAbort[linkRead[kernel, Hold], Abort[];Return[$Aborted]], {
            Hold[ReturnPacket[e___]] :> (--neval[kernel]; Parallel`Debug`Private`trace[Parallel`Debug`SendReceive, "Received result during poll of `1`: `2` (q=`3`)", traceform[kernel], hideValue[HoldForm[e]], neval[kernel]]; putback[kernel, HoldComplete[e]]),
            Hold[packet_] :> (PacketHandler[packet, kernel]; Return[True]),
            $Failed :> Throw[$Failed, dead],
            junk_ :> Throw[junk] (* should not happen *)
        }];
    ], dead, (Message[Kernels::rdead, kernel]; CloseError[kernel]; $Failed)& ];
    (* otherwise and for all fall-throughs: there was no progress *)
    False
 ]

(* putback; could check that it is wrapped in HoldComplete *)

putback[kernel_, oldres_] := (
	If[ pbQ[kernel], Message[putback::oops, kernel, pb[kernel]] ];
	setpb[kernel, oldres];
	++neval[kernel];
        Parallel`Debug`Private`trace[Parallel`Debug`SendReceive, "Putback to `1`: `2` (q=`3`)", traceform[kernel], hideValue[HoldForm@@oldres], neval[kernel]];
	Null)


(* send evaluations, but do not wait for result, eventually needs a call of kernelSwallow or kernelGetPattern *)

SetAttributes[{kernelInitialize,kernelEvaluate}, {HoldFirst,SequenceHold}]

(* several kernels, catch errors; this is used in the parallel constructor and at run time; does not go through send/receive *)

If[ TrueQ[Parallel`Debug`$Debug], (* debug version *)
    kernelInitialize[cmd_, links_List] :=
        Block[{Parallel`Debug`Private`$hideVals=True}, Function[e, linkWriteRaw[e, EvaluatePacket[cmd]]] /@ links ];
, (* else non-debug version *)
    kernelInitialize[cmd_, links_List] := Function[e, linkWriteRaw[e, EvaluatePacket[cmd]]] /@ links;
]

kernelInitialize[cmd_, link_] := firstOrFailed[kernelInitialize[cmd, {link}]] (* send to one *)

kernelInitialize[cmd_] := kernelInitialize[cmd, $kernels] (* send to all *)

(* corresponding drain of pending results; make sure busy/idle remains correct *)

kernelSwallow[kernels_List] := kernelSwallow /@ kernels

kernelSwallow[kernel_] :=
Module[{res = Null},
	While[ res=!=$Failed && readyQ[kernel], res = linkReadRaw[kernel, Hold] ];
	If[ res===$Failed, res, Null ]
]

(* read from all kernels until each delivers a result matching patt, wrapped in Hold *)

(* nothing is gained from more clever parallelization, as we eventually have to read from each one *)

kernelGetPattern[kernels_List, patt_] := kernelGetPattern[#, patt]& /@ kernels

kernelGetPattern[kernel_, patt_] :=
Module[{res}, While[True,
	Switch[res = linkReadRaw[kernel, Hold],
		Hold[patt], Return[res],
		$Failed, Return[$Failed],
		_, True
	]
]]


(* low-level version of ParallelEvaluate, without the help of Parallel.m *)

If[ TrueQ[Parallel`Debug`$Debug], (* debug version *)
    kernelEvaluate[cmd_, links_List] :=
        Block[{Parallel`Debug`Private`$hideVals=True}, receiveCatch /@ Function[e, sendCatch[e, cmd]] /@ links ];
, (* else non-debug version *)
    kernelEvaluate[cmd_, links_List] := receiveCatch /@ Function[e, sendCatch[e, cmd]] /@ links;
]

kernelEvaluate[cmd_] := kernelEvaluate[cmd, $kernels] (* send to all *)


(* default PacketHandler *)

SetAttributes[PacketHandler, HoldFirst]

PacketHandler[packet_, kernel_] := Message[PacketHandler::default, HoldForm[packet], kernel]

(* aux function to forward packets to FE *)

sendToFrontend[packet_, kernel_] := LinkWrite[ $ParentLink, packet ]

(* or print it if no FE *)

printLabelled[text_, kernel_] := (
Print[StringForm["From `1`:", kernel]]; Print[text];)

MessagePacket/: printPacket[MessagePacket[sym_, tag_], kernel_] := Null; (* forget it *)
printPacket[packet_, kernel_] := printLabelled[packet, kernel]; (* default *)

(* the correct one to use *)

forwardPacket = If[$FEQ, sendToFrontend, printPacket ]

(* use upvalues to define specific handlers *)

(* forward/print MessagePacket, 2 argument form *)

MessagePacket/: PacketHandler[t:MessagePacket[sym_, tag_], kernel_] :=
	CheckAbort[forwardPacket[t, kernel], Abort[]]

(* SuspendPacket is the kernel's way of telling us it's going to die *)

SuspendPacket/: PacketHandler[SuspendPacket[Null], kernel_] := (
	Message[Kernels::rdead, subKernel[kernel]];
	CloseError[kernel]
)

(* better treatment of remote error messages *)

(* make them appear in one piece and carry forward to master's formatting *)
(* only extract needed options, to guard against additional ones not known before (bug 222472) *)

(* bug 363361 text formatting needs a charcter encoding of Unicode, so override the setting *)
msgOptionOverrides = {CharacterEncoding->"Unicode"}

With[{opts = Select[Options[$Messages], 
                    MemberQ[{BinaryFormat, FormatType, PageWidth, PageHeight, TotalWidth, TotalHeight, NumberMarks}, First[#]] &],
      msgOptionOverrides=msgOptionOverrides},
	AppendInitCode[holdCompound[ Unprotect[MessagePacket], Clear[MessagePacket],
		SetOptions[$Messages, Sequence@@msgOptionOverrides, Sequence@@opts] ]];
]

kernelString[kernel_] := ToString[StringForm["(kernel `1`)", KernelID[kernel]], OutputForm]

If[$FEQ,
  MessagePacket/: PacketHandler[t:MessagePacket[sym_, tag_, text_], kernel_] :=
      CellPrint[Cell[text, "Message", "MSG", CellLabel->kernelString[kernel], ShowCellLabel->True]]
  , (* else no frontend *)
  MessagePacket/: PacketHandler[t:MessagePacket[sym_, tag_, text_], kernel_] :=
      printLabelled[text, kernel]; (* 3 arg form *)
]

(* tag random Print output from remote kernels. The FE strips the terminal \n in a textpacket,
   but not in CellPrint, so we strip it. This overrides the general print handler in
   Kernels` for this case. Also allow for formatted print output (ExpressionPacket) *)

If[$FEQ,
  TextPacket/: PacketHandler[TextPacket[text_String], kernel_] :=
      CellPrint[Cell[StringReplace[text, RegularExpression["\\n$"] -> ""], "Print",
                     CellLabel->kernelString[kernel], ShowCellLabel->True]];
  ExpressionPacket/: PacketHandler[ExpressionPacket[cont_], kernel_] :=
      CellPrint[Cell[cont, "Print",
                     CellLabel->kernelString[kernel], ShowCellLabel->True]]
  , (* else no frontend; should not receive any expression packets *)
  TextPacket/: PacketHandler[TextPacket[text_String], kernel_] :=
	  printLabelled[text, kernel]
]


(* heads of stuff from a remote after which it waits for a reply *)

$replyHeads = {} 

registerReplyHead[head_Symbol] := AppendTo[$replyHeads, head]


(* abort code registry. The code is called as a function with kernel as arg *)

$abortCode = {}
registerAbortCode[code_] := AppendTo[$abortCode, code]

kernel/: Abort[kernel_kernel] :=
  Module[{res = Null},
     If[ pbQ[kernel], setpb[kernel]; --neval[kernel] ]; (* clean up putback *)
     CheckAbort[ (* if we get aborted ourselves, mark kernel as failed *)
      Catch[ While[ neval[kernel]>0, (* abort pending evaluations *)
		Which[ (* tread very carefully here; find out what is best *)
		 readyQ[kernel], (* try it gently first *)
	    	   res = linkRead[kernel, Hold];
	    	   Replace[res, { (* dispatch on type of packet *)
	    	      Hold[ReturnPacket[___]] :> --neval[kernel], (* lucky *)
	    	      Hold[h_[___]/;MemberQ[$replyHeads, h]] :> ( (* send poisoned reply *)
	    	      		linkWrite[kernel, Abort[]]; Pause[Parallel`Settings`$AbortPause]; ),
	    	      $Failed :> Throw[res, dead], (* bad luck; probably redundant, as linkRead throws exc. *)
	    	      _ :> Null (* just swallow it, they probably do not expect a reply *)
	    	   }],
		 rd[kernel], (* remote is in LinkRead; send poisoned reply *)
	    	   linkWrite[kernel, Abort[]]; rdF[kernel]; Pause[Parallel`Settings`$AbortPause],
		 True, (* else try the hard way *)
	    	   res = Abort[subKernel[kernel]]; (* abort *)
	           Break[]; (* does that abort *all* evaluations? *)
		]; (* Which *)
      ], dead, (Message[Kernels::rdead, kernel]; CloseError[kernel]; res=$Failed)& ]; (* While and Catch *)
      If[ res === $Failed, Return[res] ]; (* bad luck *)
      If[ res =!= $Aborted, (* clean up if not done already *)
      	res = kernelFlush[subKernel[kernel]];
        If[res === $Failed, Message[Kernels::rdead, kernel]; CloseError[kernel]; Return[res] ];
      ]; (* clean up if not done already *)
      cleareval[kernel]; checkIdle; (* reset eval counter *)
      Through[$abortCode[kernel]]; (* registered cleanup code *)
      Return[$Aborted]; (* OK *)
     , $Failed ] (* CheckAbort: if we get aborted, signal failure *)
  ]


Format[kernel_kernel?slaveQ] :=
	"KernelObject"[KernelID[kernel], KernelName[kernel]]

Format[kernel_kernel?kernelQ] :=
	"KernelObject"[If[kid[kernel]<0, -kid[kernel], 0], KernelName[kernel], "<defunct>"]


(* specials *)

$KernelID = 0; Protect[$KernelID] (* master kernel value, for sequential fallback *)


(* launch feedback *)
(* some of these variables are also access in Parallel`Status` for the feedback in the kernel status display *)

(* defaults for undefined cases *)

resetFeedback := (
	grandTotalCount = 0;
	grandCount = 0;
	currentCount = 0;
	currentName = "kernels";
)

feedbackPause = 0; (* to slow things down, for debugging *)

feedbackHandler["name", name_, n_Integer:1] := (currentName = name; currentCount=0; grandTotalCount+=n)
feedbackHandler["tick", n_Integer:1] := (currentCount+=n; grandCount+=n; If[grandCount>grandTotalCount, grandTotalCount=grandCount]; Pause[feedbackPause])

(* make it a class method of SubKernels, but it can be used by anyone *)

SubKernels`Protected`feedbackObject = feedbackHandler


Parallel`Static`$launchFeedback = Automatic; (* False: no feedback, Automatic: only with FE, True: always *)

textstyles = 
  Sequence @@ {FontFamily -> "Verdana", FontSize -> 11, FontColor -> RGBColor[0.2, 0.4, 0.6]};
framestyles = 
  Sequence @@ {FrameMargins -> {{24, 24}, {8, 8}}, FrameStyle -> RGBColor[0.2, 0.4, 0.6], Background -> RGBColor[0.96, 0.98, 1.]};

SetAttributes[launchFeedback, HoldAll]

launchFeedback[cmd_] := Module[{res, nb, fb, textformFE, textformSA},
  If[TrueQ[Parallel`Static`$enableLaunchFeedback], fb=Parallel`Static`$launchFeedback, fb=False];
  textformFE = If[StringQ[LaunchKernels::feedbackFE], LaunchKernels::feedbackFE, "Launching `1` `2`/`3`/`4`"];
  textformSA = If[StringQ[LaunchKernels::feedbackSA], LaunchKernels::feedbackSA, "Launching kernels..."];
  resetFeedback;
  CheckAbort[Which[
	Parallel`Protected`$FEQ && (fb === True || fb === Automatic),
		nb = DisplayTemporary[ Framed[Style[ Dynamic[StringForm[textformFE, currentName, currentCount, grandCount, grandTotalCount]], textstyles], framestyles] ],
	!Parallel`Protected`$FEQ && fb === True,
		Print[textformSA]
    ];
    res = cmd;
  , (* abort handler *)
    If[ValueQ[nb], NotebookDelete[nb]; Clear[nb]]; Abort[]
  ];
  If[ValueQ[nb], NotebookDelete[nb]; Clear[nb]];
  res
]

resetFeedback (* init values *)

(* kernel auto launching support *)

`autolaunchActive = False; (* trigger is armed; this is also used in Status.m *)

declareAutolaunch[s___Symbol] := (defineAutolaunch /@ {s}; autolaunchActive = True;)

(* symbols that cause autolaunching; must not have a value, are assumed to be protected *)
(* their Hold* attributes are not modified, so arguments are preserved as they should *)

$autoSymbols = Hold[]
$autoDownValues[_] := {}

defineAutolaunch[s_Symbol] := (
	AppendTo[$autoSymbols, s]; Unprotect[s];
	$autoDownValues[s] = DownValues[s]; DownValues[s] = {};
	s[args___] := (doAutolaunch[TrueQ[Parallel`Static`$enableLaunchFeedback]]; s[args]);
	Protect[s];
)

clearAutolaunch[] := (
	List @@ ((Unprotect[#]; DownValues[#] = $autoDownValues[#]; Protect[#])& /@ $autoSymbols);
	$autoSymbols = Hold[]; Clear[$autoDownValues]; $autoDownValues[_] := {};
	autolaunchActive = False;
)

(* launch only if none are running *)

doAutolaunch[___]/; $KernelCount>0 := (clearAutolaunch[];Null)

doAutolaunch[feedback_:False] := Module[{},
  (* avoid any race conditions *)
  clearAutolaunch[];
  (* check default config *)
  If[!ListQ[$ConfiguredKernels], Return[False] ]; (* no config, nothing to launch *)
  (* unicore *)
  If[ Total[KernelCount/@$ConfiguredKernels]==0 && $ProcessorCount==1,
  	If[feedback, Message[LaunchKernels::unicore]];
  	Return[False]
  ];
  (* go for it *)
  If[ n==0, If[feedback, Message[LaunchKernels::noconf]]; Return[]];
  Block[{Parallel`Static`$launchFeedback=feedback},
  	LaunchKernels[]
  ];
]

(* the list of autolaunch symbols is defined in Kernel/autoload.m *)

Protect[Evaluate[protected]]

End[]

Protect[ LaunchKernel, LaunchKernels, ConnectKernel, CloseKernels, AbortKernels, ClearKernels, CloseError ]
Protect[ KernelID, SubKernel, KernelName, EvaluationCount ]
Protect[ KernelFromID ]
Protect[ send, receive, receive0, putback, poll, $notReady ]
Protect[ PacketHandler ]
Protect[ slaveQ, kernelQ ]
Protect[ kernel ]
Protect[ Kernels, $KernelCount ]
Protect[ LaunchDefaultKernels ]

EndPackage[]
