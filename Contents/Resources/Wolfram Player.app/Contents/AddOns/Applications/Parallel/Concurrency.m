(* :Title: Concurrency.m -- support for processes and general concurrency *)

(* :Context: Parallel`Concurrency` *)

(* :Author: Roman E. Maeder *)

(* :Summary:
   Basic process control for parallel evaluation of Mathematica expressions.
 *)

(* :Package Version: 1.0  *)

(* :Mathematica Version: 7 *)

(* :History:
   1.0 derived from old Parallel.m
*)

BeginPackage["Parallel`Concurrency`"]

System`ParallelSubmit

System`WaitAll
System`WaitNext

(* applications *)

System`ParallelTry
System`CriticalSection

(* globals *)

System`EvaluationObject

(* locking shared functions *)
`acquire
`release


BeginPackage["Parallel`Developer`"] (* developer context for lesser functionality *)

`SetQueueType::usage = "SetQueueType[constructor, args...] sets the type of process queue to use, by giving its constructor and the arguments with which to call it.
	A list of available queues is in Parallel`Queue`Interface`$QueueTypes."
SyntaxInformation[SetQueueType] = { "ArgumentsPattern" -> {_, ___} }

`$QueueType::usage = "$QueueType gives the current type of process queue used."


(* $QueueTypes is from Parallel`Queue`Interface and normally not on $ContextPath *)

`Scheduling::usage = "Scheduling->data is an option of ParallelSubmit[] to specify scheduling info that may be used by the queue or scheduler."

`QueueRun::usage = "QueueRun[] checks all subkernels for finished evaluations and submits waiting evaluations to available kernels."
SyntaxInformation[QueueRun] = { "ArgumentsPattern" -> {} }

`DoneQ::usage = "DoneQ[eid] returns True if the given concurrent evaluation has finished."
SyntaxInformation[DoneQ] = { "ArgumentsPattern" -> {_} }

`ConcurrentEvaluate::usage = "ConcurrentEvaluate[HoldComplete[exprs..]] evaluates all exprs concurrently."
SyntaxInformation[ConcurrentEvaluate] = { "ArgumentsPattern" -> {_} }

`ResetQueues::usage = "ResetQueues[] waits for all running concurrent evaluations and abandons any queued evaluations.
	It can be used to clean up after aborting WaitAll[]."
SyntaxInformation[ResetQueues] = { "ArgumentsPattern" -> {} }

`Load::usage = "Load[kernel] gives the load (length of input queue) on a remote kernel. Load[] gives the sum of all loads."
SyntaxInformation[Load] = { "ArgumentsPattern" -> {_.} }

`$Queue::usage = "$Queue is the input queue of processes waiting to be evaluated."
`$QueueLength::usage = "$QueueLength is the length of the input queue $Queue."
`$LoadFactor::usage = "$LoadFactor determines the maximum number of pending processes on a kernel. Values larger than 1 implement latency hiding."

`Process::usage = "Process[pid] gives the expression that pid is to evaluate, wrapped in HoldForm[]."
SyntaxInformation[Process] = { "ArgumentsPattern" -> {_} }

`ProcessID::usage = "ProcessID[pid] gives the integer process id of pid."
SyntaxInformation[ProcessID] = { "ArgumentsPattern" -> {_} }

`ProcessState::usage = "ProcessState[pid] gives the queueing state of pid."
SyntaxInformation[ProcessState] = { "ArgumentsPattern" -> {_} }

Scheduling::usage = Scheduling::usage <> " Scheduling[pid] gives the scheduling info of pid."
SyntaxInformation[Scheduling] = { "ArgumentsPattern" -> {_} }

`ProcessResult::usage = "ProcessResult[pid] gives the result of an evaluation, wrapped in HoldComplete,
	or $Failed if not available."
SyntaxInformation[ProcessResult] = { "ArgumentsPattern" -> {_} }

(* semi-public queue states *)
`created::usage = "created is a process state indicating that the process has been created."
`queued::usage = "queued is a process state indicating that the process is in the input queue, waiting to be evaluated on a remote processor."
`running::usage = "running[kernel] is a process state indicating that the process is running on kernel."
`finished::usage = "finished[result] is a process state indicating that the process is finished with the given result."
`dequeued::usage = "dequeued[result] is a process state indicating that the process has been removed from the output queue."

EndPackage[]


BeginPackage["Parallel`Protected`"] (* semi-hidden methods, aka protected *)

`newEvaluation::usage = "newEvaluation[cmd, (sched)] creates a new evaluation object."

`breakLocks::usage = "breakLocks[] clears all currently locked variables."

EndPackage[]


Options[ParallelSubmit] = {
	Scheduling -> Null
}

Options[ParallelTry] = {
	DistributedContexts :> $DistributedContexts
}

(* messages *)

QueueRun::req = "Requeueing evaluations `2` assigned to `1`."
QueueRun::aba = "Abandoning evaluations `2` assigned to `1`."
QueueRun::hmm = "Received unexpected result `2` from `1`."
QueueRun::oops = "Received unexpected evaluation from `1`: `3` instead of `2`."
SetQueueType::nope = "Evaluation queue is not empty; cannot change its type."
SetQueueType::badq = "Constructor `1` does not create a valid queue."
WaitNext::none = "No valid evaluations found in `1`."
ParallelTry::toofew = "Only `1` results received, rather than `2`."


Begin["`Private`"] (*****************************************************)

`$PackageVersion = 1.0;
`$thisFile = $InputFileName

Needs["Parallel`Kernels`"] (* a protected base class, really *)
Needs["Parallel`Parallel`"]

(* some of its internal features *)
holdCompound = Parallel`Client`HoldCompound

protected = Unprotect[kernel]

(* debug tracers used in this package *)

Parallel`Debug`Private`RegisterTrace[ Parallel`Debug`Queueing, "Queueing is a tracer that triggers when processes are queued/dequeued or sent to, or received from, remote kernels." ]


(* load[link] is number of queued jobs on link *)

SetAttributes[load, Listable]
kernel/: load[k_kernel] := lqueue[k]
totalLoad=0;

(* wrapper around kernel enqueue/dequeue that keep totalLoad in sync *)

putjob[k_, job_] := (totalLoad++; enqueue[k, job])
getjob[k_] := (totalLoad--; dequeue[k])

(* controlled access to $LoadFactor; bound by $maxProcessQueueSize *)

`$loadFactor = 1

$LoadFactor/: HoldPattern[$LoadFactor = new_Integer/;new>1&&Parallel`VirtualShared`Private`$sharingActive] :=
	(Message[$LoadFactor::shared]; $loadFactor=1)
$LoadFactor/: HoldPattern[$LoadFactor = new_Integer?Positive] :=
	($loadFactor = Min[new, $maxProcessQueueSize-1])
$LoadFactor/: HoldPattern[$LoadFactor = new_] :=
	(Message[$LoadFactor::intpm, HoldForm[$LoadFactor = new], 2]; $loadFactor)

(* do not use if shared variables exist. always use $LoadFactor internally, not $loadFActor *)
$LoadFactor := If[TrueQ[Parallel`VirtualShared`Private`$sharingActive], 1, $loadFactor]
Protect[$LoadFactor]

(* process data type	pid[id, cmd, sched, state]
	id	integer process id
	cmd	the cmd to be evaluated, wrapped in HoldComplete[]
	sched	scheduler data
	state	queue state, mutable: 
		created				job is new
		queued				job is queued in $queue
		running[kernel]		running on kernel
		finished[result]	job has finished with this result
		dequeued[result]	job has been dequeued
		invalid[$Failed]	error state
 *)

SetAttributes[EvaluationObject, HoldAllComplete]; SetAttributes[makePid, {HoldFirst,SequenceHold}]

{`created, `queued, `running, `finished, `dequeued, `invalid} (* private versions *)

SetAttributes[{finished,dequeued}, HoldAllComplete]
$invalid = invalid[$Failed] (* shared value *)

makePid[cmd_, id_, sched_, state0_:created] :=
	Module[{state=state0}, EvaluationObject[id, cmd, sched, state] ]

EvaluationObject/: id[EvaluationObject[id_, cmd_, sched_, state_]] := id
EvaluationObject/: cmd[EvaluationObject[id_, cmd_, sched_, state_]] := HoldComplete[cmd]
EvaluationObject/: sched[EvaluationObject[id_, cmd_, sched_, state_]] := sched
EvaluationObject/: state[EvaluationObject[id_, cmd_, sched_, state_]] := state
EvaluationObject/: setState[EvaluationObject[id_, cmd_, sched_, state_], new_] := (state=new)

(* for nonexisting objects *)
state[_] := invalid["Undefined"]

(* destructor *)

EvaluationObject/: delete[EvaluationObject[id_, cmd_, sched_, state_]] := Clear[state]

(* public *)

SetAttributes[{Parallel`Developer`finished, Parallel`Developer`dequeued}, HoldAllComplete]

(* the "Undefined" state may happen when a dynamicly displayed evaluation object is redisplayed after
   the frontend is relaunched, as the underlying object is no longer there *)

pubtrans = {
	created -> Parallel`Developer`created,
	queued -> Parallel`Developer`queued,
	running[k_] :> Parallel`Developer`running[k],
	finished[] :> Parallel`Developer`finished[Sequence[]],
	dequeued[] :> Parallel`Developer`dequeued[Sequence[]],
	finished[r__] :> Parallel`Developer`finished[r],
	dequeued[r__] :> Parallel`Developer`dequeued[r],
	invalid[r_] :> r,
	_ :> "Undefined"
}

EvaluationObject/: Process[job_EvaluationObject] := HoldForm@@cmd[job]
EvaluationObject/: ProcessID[job_EvaluationObject] := id[job]
EvaluationObject/: Scheduling[job_EvaluationObject] := sched[job]
EvaluationObject/: ProcessState[job_EvaluationObject] := state[job] /. pubtrans

EvaluationObject/: ProcessResult[job_EvaluationObject] :=
	state[job] /. {
		finished[]|dequeued[] :> HoldComplete[Sequence[]], (* tricky case *)
		finished[r__]|dequeued[r__] :> HoldComplete[r],
		_ :> $Failed}

(* internal constructor *)

lastpid=0; (* last used pid *)
lastconpid=0; (* for internal queue *)

SetAttributes[newEvaluation, {HoldFirst,SequenceHold}]

newEvaluation[cmd_, sched_:Null] := makePid[cmd, ++lastpid, sched]

(* queues *)

Needs[ "Parallel`Queue`Interface`" ]

EvaluationObject/: Priority[job_EvaluationObject] := sched[job] (* for the queue interface *)

If[ Length[$QueueTypes] == 0, (* load default queue types *)
	Needs[ "Parallel`Queue`FIFO`" ];
	Needs[ "Parallel`Queue`Priority`" ];
]

`$queue (* the raw queue *)
`$queueType (* its type, internal version *)

(* emptyQueue returns a fresh empty queue, each time it is evaluated *)

SetQueueType[make_, args___]/; MemberQ[$QueueTypes, make] :=
  Module[{newq},
	If[ ValueQ[$queue] && !EmptyQ[$queue], Message[SetQueueType::nope]; Return[$Failed] ];
	newq = make[args]; (* try out constructor *)
	If[ !qQ[newq], (* oops *)
	    Message[SetQueueType::badq, HoldForm[make[args]]]; Return[$Failed];
	];
	delete[$queue]; (* clean up old queue *)
	$queueType = make;
	emptyQueue := make[args]; (* the constructor *)
	$queue = newq;
	$queueType
  ]

SetQueueType[$QueueTypes[[1]]]; (* initialize with default constructor *)

(* readonly values *)
$QueueType := $queueType;
$Queue := $queue
$QueueLength := Size[$queue]

(* the private queue for ConcurrentEvaluate *)

Needs[ "Parallel`Queue`FIFO`" ];

emptyConQueue := FIFOQueue[]
$conqueue = emptyConQueue


(* error recovery:
- Close calls tryRecover which collects orphaned pids, if enabled.
- Send does no recovery, but returns $Failed if LinkWrite failed.
  Code that calls Send may not need to check for this, but if it does,
  should emit Kernels::rdead.
- receive (and Receive, ReceiveIfReady) does check for errors and calls
  Close in case of error and emits Kernels::rdead.
*)



(* job control *)

(* trace output formatting for evids *)

EvaluationObject/: traceform[job_EvaluationObject] := Subscripted["eid"[id[job]]][Short[Process[job],0.3]]


EvaluationObject/: queue[job_EvaluationObject] := (
        AbortProtect[ PreemptProtect[EnQueue[$queue, job]]; setState[job, queued] ];
        Parallel`Debug`Private`trace[Parallel`Debug`Queueing, "`1` queued (`2`)", traceform[job], $QueueLength-1];
		job
    )

EvaluationObject/: reQueue[job_EvaluationObject] := (
        AbortProtect[ PreemptProtect[EnQueue[$queue, job]]; setState[job, queued] ];
        Parallel`Debug`Private`trace[Parallel`Debug`Queueing, "`1` requeued (`2`)", traceform[job], $QueueLength-1];
		Null (* don't return job!! *)
    )


SetAttributes[{ParallelSubmit}, HoldAllComplete]

(* Queue0 is for controlled evaluation of some arguments of ParallelSubmit *)

(** should convince myself that there is not too much ambiguity in these patterns. Critical case is ParallelSubmit[{a}, Scheduling->5] vs ParallelSubmit[{a}, a].
when in doubt, use ParallelSubmit[{}, ...]

The rules turn any valid form of ParallelSubmit[....] into one of
	Queue0[ HoldComplete[cmd], prio ]
	Queue0[ HoldComplete[cmd], Hold[vars], prio ]
 **)

(* optionQ that does not eval its argument *)
SetAttributes[optionQ, HoldAllComplete]
optionQ[opt_] := OptionQ[Unevaluated[opt]]

(* ParallelSubmit without closures, no options: use default *)

ParallelSubmit[cmd_] := Queue0[ HoldComplete[cmd], Scheduling /. Options[ParallelSubmit] ]

(* here we cannot use OptionQ because of evaluation *)

ParallelSubmit[cmd_, opts__?optionQ] :=
	Queue0[ HoldComplete[cmd], Scheduling /. Flatten[{opts}] /. Options[ParallelSubmit] ]

(* ParallelSubmit with closures; here we can use opts___ (triple blank) *)

ParallelSubmit[{vars___Symbol}, cmd_, opts___?optionQ] := 
	Queue0[ HoldComplete[cmd], Hold[vars], Scheduling /. Flatten[{opts}] /. Options[ParallelSubmit] ]


(* Queue0 with 3 args performs closure *)

Queue0[hcmd_, Hold[vars___], sched_] := Function[{vars}, Queue0[hcmd, sched]][vars]

(* finally, Queue0 without closure, create new pid and enter into queue *)

Queue0[HoldComplete[cmd_], sched_ ] := queue[newEvaluation[cmd, sched]]


(* QueueRun returns True, if at least one job was (de)queued *)

QueueRun[] := queueRun[] (* public version *)

(* sequential fallback: direct evaluation. *)

(* protect local eval from all nonlocal flows of control, as much as possible *)

SetAttributes[localEval,HoldAllComplete]

localEval[expr_] := Module[{res = $Failed},
  Block[{Quit := Throw[Null]}, 
  		Reap[res = CheckAll[expr, Hold]]
  ];
  If[res === $Failed, Return[$Failed]]; (* shouldn't happen *)
  If[MemberQ[res[[2]], Unevaluated[Abort[]]], Abort[]]; (* re-throw the abort *)
  First[res]
]

queueRun[extra_List:{}]/; $seQ :=
  Module[ {next = 1, extras = Length[extra], progress = False},
        While[ next<=extras && !MatchQ[state[extra[[next]]], created|queued], next++ ];
        PreemptProtect[ While[ !EmptyQ[$queue] && state[Top[$queue]] =!= queued, DeQueue[$queue] ]];
        Which[
          next<=extras,
              With[ {job = extra[[next++]]},
                With[ {r = localEval[cmd[job][[1]]]}, (* eval here and now *)
                  setState[job, finished[r]]; (* stash result *)
                ];
                Parallel`Debug`Private`trace[Parallel`Debug`Queueing, "`1` evaluated locally", traceform[job]];
              ];
              progress = True,
          !EmptyQ[$queue],
              With[ {job = DeQueue[$queue]},
                With[ {r = localEval[cmd[job][[1]]]}, (* eval here and now *)
                  setState[job, finished[r]]; (* stash result *)
                ];
                Parallel`Debug`Private`trace[Parallel`Debug`Queueing, "`1` evaluated locally", traceform[job]];
              ];
              progress = True,
          True, Null(* nothing to evaluate *)
        ];
	    progress
  ]

(* performance counters, debug only, if possible *)
$queueruns = 0; $idleruns = 0;

(* to dispatch jobs:
	sort kernels  on speed
	collect pending results
	dispatch one job to each kernel with room in input queue
	code is optimized for steady state: queues are full and kernels are busy
	job is tagged as tag[expr, id]

   we use raw send/receive/$notReady from Kernels.m
   receive0 is a version that does not complain if no evaluation is pending
*)

If[ TrueQ[Parallel`Debug`$Debug], (* debug version tags jobs with their pid *)

With[{tag = $`j}, (* used as tag; should be short *)
  queueRun[extra_List:{}] := Module[{next=1, extras=Length[extra], progress=False},
  	Scan[ Function[k, Catch[
  		(* received something? *)
  		Replace[ receive0[k], { (*a Switch[] in disguise *)
            HoldComplete[tag[r___, id0_]] :> (
                AbortProtect[ With[{rjob = getjob[k]},
                  setState[rjob, finished[r]]; (* stash result *)
                  If[ id[rjob]=!=id0, Message[QueueRun::oops, k, id[rjob], id0] ]; (* same one? *)
                  Parallel`Debug`Private`trace[Parallel`Debug`Queueing, "`1` received from `2`", traceform[rjob], traceform[k]];
                ]];
                progress = True; (* got at least one *)
              ),
            HoldComplete[r:$Aborted] :> ( (* the abort drops the tag[..] *)
                AbortProtect[ With[{rjob = getjob[k]},
                  setState[rjob, finished[r]]; (* stash result *)
                  (* hope it is the same one *)
                  Parallel`Debug`Private`trace[Parallel`Debug`Queueing, "`1` aborted on `2`", traceform[rjob], traceform[k]];
                ]];
                progress = True; (* got at least one *)
              ),
            HoldComplete[junk_] :> Message[QueueRun::hmm, k, HoldForm[junk]],
            $Failed|$notReady :> Null (* ignore *)
		}];
		(* can we send something? *)
		If[ load[k]<$LoadFactor,
			(* find something to send *)
			While[ next<=extras && !MatchQ[state[extra[[next]]], created|queued], next++ ];
			PreemptProtect[While[ !EmptyQ[$queue] && state[Top[$queue]] =!= queued, DeQueue[$queue] ]];
			Which[
			  next<=extras,
				  With[{job=extra[[next]]},
				  	Replace[{cmd[job], id[job]}, {HoldComplete[cmd_], id_} :> send[k, tag[cmd, id]]];
				    AbortProtect[ setState[job, running[k]]; putjob[k, job]; next++ ];
				    Parallel`Debug`Private`trace[Parallel`Debug`Queueing, "`1` sent to `2`", traceform[job], traceform[k]];
				  ];
				  progress=True,
			  !EmptyQ[$queue],
				  With[{job=Top[$queue]},
				  	Replace[{cmd[job], id[job]}, {HoldComplete[cmd_], id_} :> send[k, tag[cmd, id]]];
				    AbortProtect[ setState[job, running[k]]; putjob[k, job]; DeQueue[$queue] ];
				    Parallel`Debug`Private`trace[Parallel`Debug`Queueing, "`1` sent to `2`", traceform[job], traceform[k]];
				  ];
				  progress=True,
			  True, Null(* nothing to send *)
			];
		];
	  , dead, (Message[Kernels::rdead, k]; CloseError[k];)&]
	],
	$sortedkernels];
	progress
 ]
]
, (* nondebug, no tag[] *)

  queueRun[extra_List:{}] := Module[{next=1, extras=Length[extra], progress=False},
 	Scan[ Function[k, Catch[
  		(* received something? *)
  		Replace[ receive0[k], { (*a Switch[] in disguise *)
            HoldComplete[r___] :> (
                AbortProtect[ With[{rjob = getjob[k]},
                  setState[rjob, finished[r]]; (* stash result *)
                  (* omit test for correct id *)
                ]];
                progress = True; (* got at least one *)
              ),
            HoldComplete[r:$Aborted] :> ( (* this is actually redundant *)
                AbortProtect[ With[{rjob = getjob[k]},
                  setState[rjob, finished[r]]; (* stash result *)
                  (* hope it is the same one *)
                ]];
                progress = True; (* got at least one *)
              ),
            HoldComplete[junk_] :> Message[QueueRun::hmm, k, HoldForm[junk]],
            $Failed|$notReady :> Null (* ignore *)
		}];
		(* can we send something? *)
		If[ load[k]<$LoadFactor,
			(* find something to send *)
			While[ next<=extras && !MatchQ[state[extra[[next]]], created|queued], next++ ];
			While[ !EmptyQ[$queue] && state[Top[$queue]] =!= queued, DeQueue[$queue] ];
			Which[
				next<=extras,
				  With[{job=extra[[next]]},
				  	Replace[cmd[job], HoldComplete[cmd_] :> send[k, cmd]];
				    AbortProtect[ setState[job, running[k]]; putjob[k, job]; next++ ] ];
				  progress=True,
				!EmptyQ[$queue],
				  With[{job=Top[$queue]},
				  	Replace[cmd[job], HoldComplete[cmd_] :> send[k, cmd]];
				    AbortProtect[ setState[job, running[k]]; putjob[k, job]; DeQueue[$queue] ] ];
				  progress=True,
				True, Null(* nothing to send *)
			];
		];
	  , dead, (Message[Kernels::rdead, k]; CloseError[k];)&]
	],
	$sortedkernels];
	progress
   ]

] (* debug switch *)


(* list run, schedule a list of pids (which are not necessarily in a queue *)

If[ TrueQ[Parallel`Debug`$Debug], (* debug version tags jobs with their pid *)

With[{tag = $`k}, (* used as tag; should be short *)
listRun[eids_] := Module[{next=1, nids=Length[eids], nrec=0, progress}, CheckAbort[
 Block[{$queue=$conqueue, Parallel`Settings`$RecoveryMode="Retry"},
  While[nrec<nids, (* more to come *)
  	progress=False;
  	If[ $seQ, (* do it ourselves *)
          tryRelaunch[];
          Which[
           !EmptyQ[$queue],
              With[ {job = DeQueue[$queue]},
                With[ {r = localEval[cmd[job][[1]]]}, (* eval here and now *)
                  setState[job, finished[r]]; (* stash result *)
                  nrec++; ] ];
              progress = True,
         	next<=nids,
              With[ {job = eids[[next++]]},
                  With[ {r = localEval[cmd[job][[1]]]}, (* eval here and now *)
                      setState[job, finished[r]]; (* stash result *)
                      nrec++; ] ];
              progress = True
		  ]
    ];
  	Scan[ Function[k, Catch[
  		(* received something? *)
  		Replace[ receive0[k], { (*a Switch[] in disguise *)
            HoldComplete[tag[r___, id0_]] :> (
                AbortProtect[ With[{rjob = getjob[k]},
                  setState[rjob, finished[r]]; (* stash result *)
                  nrec++;
                  If[ id[rjob]=!=id0, Message[QueueRun::oops, k, id[rjob], id0] ]; (* same one? *)
                  Parallel`Debug`Private`trace[Parallel`Debug`Queueing, "`1` done", traceform[rjob]];
                ]];
                progress = True; (* got at least one *)
              ),
            HoldComplete[r:$Aborted] :> (
                AbortProtect[ With[{rjob = getjob[k]},
                  setState[rjob, finished[r]]; (* stash result *)
                  nrec++;
                  (* hopefully same one? *)
                  Parallel`Debug`Private`trace[Parallel`Debug`Queueing, "`1` aborted", traceform[rjob]];
                ]];
                progress = True; (* got at least one *)
              ),
            HoldComplete[junk_] :> Message[QueueRun::hmm, k, HoldForm[junk]],
            $Failed|$notReady :> Null (* ignore *)
		}];
		(* can we send something? *)
		If[ load[k]<$LoadFactor,
			(* find something to send *)
			Which[
			  !EmptyQ[$queue],(* a recovered item? *)
				  With[{job=Top[$queue]},
				  	Replace[{cmd[job], id[job]}, {HoldComplete[cmd_], id_} :> send[k, tag[cmd, id]]];
				    AbortProtect[ setState[job, running[k]]; putjob[k, job]; DeQueue[$queue] ] ];
				  progress=True,
			  next<=nids,
				  With[{job=eids[[next]]},
				  	Replace[{cmd[job], id[job]}, {HoldComplete[cmd_], id_} :> send[k, tag[cmd, id]]];
				    AbortProtect[ setState[job, running[k]]; putjob[k, job]; next++ ] ];
				  progress=True,
			  True, Null(* nothing to send *)
			];
		];
	  , dead, (Message[Kernels::rdead, k]; CloseError[k];)&] (* Catch *)
	], $sortedkernels]; (* Scan *)
	If[ !progress, tryRelaunch[]; Pause[Parallel`Settings`$BusyWait] ];
  ]], AbortKernels[]; Abort[] (* CheckAbort *)
]]
]
, (* nondebug *)

listRun[eids_] := Module[{next=1, nids=Length[eids], nrec=0, progress}, CheckAbort[
 Block[{$queue=$conqueue, Parallel`Settings`$RecoveryMode="Retry"},
  While[nrec<nids, (* more to come *)
  	progress=False;
  	If[ $seQ, (* do it ourselves *)
          tryRelaunch[];
          Which[
           !EmptyQ[$queue],
              With[ {job = DeQueue[$queue]},
                With[ {r = localEval[cmd[job][[1]]]}, (* eval here and now *)
                  setState[job, finished[r]]; (* stash result *)
                  nrec++; ] ];
              progress = True,
         	next<=nids,
              With[ {job = eids[[next++]]},
                  With[ {r = localEval[cmd[job][[1]]]}, (* eval here and now *)
                      setState[job, finished[r]]; (* stash result *)
                      nrec++; ] ];
              progress = True
		  ]
    ];
  	Scan[ Function[k, Catch[
  		(* received something? *)
  		Replace[ receive0[k], { (*a Switch[] in disguise *)
            HoldComplete[r___] :> (
                AbortProtect[ With[{rjob = getjob[k]},
                  setState[rjob, finished[r]]; (* stash result *)
                  nrec++;
                  (* omit test for correct id *)
                ]];
                progress = True; (* got at least one *)
              ),
            HoldComplete[r:$Aborted] :> ( (* TODO do we need this? *)
                AbortProtect[ With[{rjob = getjob[k]},
                  setState[rjob, finished[r]]; (* stash result *)
                  nrec++;
                  (* hopefully same one? *)
                ]];
                progress = True; (* got at least one *)
              ),
            HoldComplete[junk_] :> Message[QueueRun::hmm, k, HoldForm[junk]],
            $Failed|$notReady :> Null (* ignore *)
		}];
		(* can we send something? *)
		If[ load[k]<$LoadFactor,
			(* find something to send *)
			Which[
			  !EmptyQ[$queue],(* a recovered item? *)
				  With[{job=Top[$queue]},
				  	Replace[{cmd[job], id[job]}, {HoldComplete[cmd_], id_} :> send[k, cmd]];
				    AbortProtect[ setState[job, running[k]]; putjob[k, job]; DeQueue[$queue] ] ];
				  progress=True,
			  next<=nids,
				  With[{job=eids[[next]]},
				  	Replace[{cmd[job], id[job]}, {HoldComplete[cmd_], id_} :> send[k, cmd]];
				    AbortProtect[ setState[job, running[k]]; putjob[k, job]; next++ ] ];
				  progress=True
			];
		];
	  , dead, (Message[Kernels::rdead, k]; CloseError[k];)&] (* Catch *)
	], $sortedkernels]; (* Scan *)
	If[ !progress, tryRelaunch[]; Pause[Parallel`Settings`$BusyWait] ];
  ]], AbortKernels[]; Abort[] (* CheckAbort *)
]]

]


(* need to treat invalid/dequeued pids as done *)

doneQ[job_EvaluationObject] := MatchQ[state[job], finished[___]|invalid[_]|dequeued[___]]

DoneQ[job_EvaluationObject] := doneQ[job]

(* precondition: doneQ[job]; careful about Sequence
   the replacement uncovers the result (if finished[res]) or the failure
   reason (if invalid[res]). also handle attempts to dequeue twice
*)

deQueue[job_EvaluationObject] :=
  With[{res = state[job] /. dequeued[___]:>$invalid},
    setState[job, state[job] /. {finished[r___]:>dequeued[r], _:>$invalid}];
    Parallel`Debug`Private`trace[Parallel`Debug`Queueing, "`1` dequeued", traceform[job]];
    Replace[ res, h_[e___]:>e ]
  ]

(* version for resetQueues. do not eval result, throw it away *)
deQueue0[job_EvaluationObject] :=
  With[{},
    setState[job, $invalid];
    Parallel`Debug`Private`trace[Parallel`Debug`Queueing, "`1` dequeued and abandoned", traceform[job]];
    Null
  ]


(* client-side queueing is not supported, but at least define the
  symbols for future client-side scheduling: ParallelSubmit/WaitAll/WaitNext/QueueRun/DoneQ *)
(* These definitions are now in Client.m *)


(* WaitAll for processes *)

pidOk[job_EvaluationObject] := MatchQ[state[job], created|queued|running[_]|finished[___]] (* is in system *)

(* make EvaluationObject[...] evaluate to result of job, original idea due to Dan Grayson *)

`hot (* causes pids to melt; works also for invalid pids *)

EvaluationObject/: hot[job_EvaluationObject] := (
    While[ !doneQ[job],
      If[!queueRun[{job}], tryRelaunch[]; Pause[Parallel`Settings`$BusyWait] ]; (* avoid spinning *)
    ];
    deQueue[job]
)

(* WaitAll is now (almost) trivial; the //. allows for new pids to appear *)
(* if possible, call performance monitor hooks *)

WaitAll[anything_] := Module[{res},
	$queueruns = 0; $idleruns = 0;
	CheckAbort[
		parStart;
		res = anything //. job_EvaluationObject :> hot[job];
		parStop;
	, (* aborted *)
		AbortKernels[]; Abort[]
	];
	res
]

(* abort action: we let everything run, no special treatment necessary *)

WaitNext[pids:{___EvaluationObject}] :=
    Module[{is, res},
      $queueruns = 0; $idleruns = 0;
      While[True,
        res = queueRun[pids];
        is = Position[pids, _?doneQ, 1];
        If[Length[is]>0, With[{i=is[[1,1]]},
           Return[{deQueue[pids[[i]]], pids[[i]], Drop[pids,{i}]}]
        ]];
        If[ !res,  (* sanity check: is there anything left to wait for? *)
            If[Count[pids, _?pidOk]==0,
                Message[WaitNext::none, pids];
                Return[{$Failed, Null, pids}];
            ];
            tryRelaunch[];
            Pause[Parallel`Settings`$BusyWait];  (* avoid spinning *)
        ];
      ]
    ]


kernel/: Load[link_kernel] := load[link]
(* TODO: can we use totalLoad? *)
Load[] := Total[load /@ Kernels[]]


(* recovery mode handlers *)

tryRecover[link_] := tryRecover[link, Parallel`Settings`$RecoveryMode]

tryRecover[link_, "ReQueue"|"Retry"] :=
With[{jobs=processes[link]},
    If[ Length[jobs]>0, Message[QueueRun::req, link, id/@jobs] ];
    totalLoad -= load[link]; (* keep in sync *)
    rqueue[link]; (* reinit local processor queue *)
    reQueue /@ jobs; (* requeue those poor saps *)
]

tryRecover[link_, "Abandon"] :=
With[{jobs=processes[link]},
    If[ Length[jobs]>0,
        Message[QueueRun::aba, link, id/@jobs];
        Parallel`Debug`Private`trace[Parallel`Debug`Queueing, "Processes `1` on `2` abandoned", traceform/@jobs, traceform[link]];
    ];
    totalLoad -= load[link]; (* keep in sync *)
    rqueue[link]; (* reinit local processor queue *)
    setState[#, $invalid]& /@ jobs; (* pretend they'r finished *)
]

(* other modes are ignored; load counts may become inconsistent *)

(* recovery modes are set through SetSystemOptions["ParallelOptions" -> "RecoveryMode" -> ]
   allowed values are "Abandon" and "Retry"; reflected in Parallel`Settings`$RecoveryMode
   for easier migration, the old value "ReQueue" is also accepted and mapped to "Retry" *)

(* register process cleanup for Close *)

registerCloseCode[tryRecover]

(* process cleanup after an abort; dare we use $RecoveryMode?
   This is effective only for a stand-alone Abort[kernel]. If called via AbortKernels, the queues are already cleared. *)

registerAbortCode[tryRecover]


(* to clean up after aborting WaitAll[]; try not to eval any results returned *)

SetAttributes[{queue0}, HoldAll] (* version of ParallelSubmit[] during reset *)
queue0 = Null& (* don't eval argument, throw it away *)

resetQueues[drain_:True] :=
  Module[{},
    With[{alljobs = Normal[$queue]}, If[ Length[alljobs]>0,
      Parallel`Debug`Private`trace[Parallel`Debug`Queueing, "Processes `1` removed from input queue", traceform/@alljobs];
      setState[#, $invalid]& /@ alljobs; (* forget queued ones *)
    ]];
    delete[$queue]; $queue = emptyQueue; (* reinit queues *)
    delete[$conqueue]; $conqueue = emptyConQueue;
    With[{alljobs = Flatten[processes/@$kernels]}, If[ Length[alljobs]>0,
      If[ drain,
        Block[{Parallel`Settings`$RecoveryMode="Abandon", deQueue=deQueue0, ParallelSubmit=queue0},
          WaitAll[alljobs]; (* drain all queues *)
        ]
      , (* else invalidate the pending pids; do not abort kernels here *)
        Parallel`Debug`Private`trace[Parallel`Debug`Queueing, "Processes `1` removed from processor queues", traceform/@alljobs];
        setState[#, $invalid]& /@ alljobs;
        rqueue /@ $kernels; (* reinit local processor queues *)
        totalLoad = 0;
      ]
    ]];
    (* lastpid = 0; *)
    Kernels[]
  ]

ResetQueues[] := resetQueues[True] (* graceful way: wait for pending jobs *)

registerResetCode[resetQueues[False]] (* for our ParallelSubmit/WaitAll: fix up queues *)


(* ConcurrentEvaluate *)

(* evaluate the elements of an expression concurrently, a generalization of ParallelDispatch *)

SetAttributes[ConcurrentEvaluate, {HoldFirst,SequenceHold}]

ConcurrentEvaluate[h_[elems___]] :=
Module[{eids, res},
	eids = newEvaluation /@ Unevaluated[{elems}];
	CheckAbort[
		listRun[eids];
		res = h @@ Join @@ ProcessResult /@ eids
	  , (* aborted *)
		AbortKernels[]; Clear[eids];
		Abort[]
	];
	res
]


(* ParallelTry *)

(* whether or not the args are evaluated locally depends on their container; we preserve that choice *)

(* TODO: should we use the internal queue? *)

ParallelTry[f_, _[args___], k_Integer?NonNegative, o:OptionsPattern[]] :=
    Module[{res = {}, r, id, ids},
    	Parallel`Protected`AutoDistribute[{f, args}, ParallelTry, OptionValue[ParallelTry,{o},DistributedContexts]]; (* send definitions *)
    	parStart;
        ids = List@@ParallelSubmit/@f/@Hold[args];
        CheckAbort[
			While[Length[res] < k && Length[ids] > 0, (* try one more *)
				{r, id, ids} = WaitNext[ids];
				If[r =!= $Failed, AppendTo[res, r]];
			]
		  , (* in case of an Abort *)
			AbortKernels[]; Clear[ids];
			Abort[]
		];
        AbortKernels[]; (* stop the slower ones *)
        parStop;
        If[Length[res]<k, Message[ParallelTry::toofew, Length[res], k]];
        res
    ]

(* only one result, but beware of $Failed and $Aborted *)

ParallelTry[f_, args_, o:OptionsPattern[]] := Replace[ParallelTry[Unevaluated[f], Unevaluated[args], 1, o], {{r_} :> r, {} :> $Failed}]


(* fancy evaluation object display *)

diskColors = {RGBColor[0.4, 0.68, 1.], RGBColor[0.49, 0.83, 0.27], 
   RGBColor[0.737, .737, 0.737], RGBColor[0.837, .837, 0.837], RGBColor[0.988, 0.384, 0.0345]};
bgColors = {RGBColor[0.89, 0.947, 1.], RGBColor[0.925, 1., 0.848], 
   RGBColor[0.957, 0.957, 0.957], RGBColor[1., 1., 1.], RGBColor[1., 0.9, 0.9]};
exclam = {{4.789`, 5.`}, {4.102`, 5.`}, {4.102`, 5.688`}, {4.789`, 
     5.688`}, {4.789`, 5.`}, {4.789`, 5.`}, {4.711`, 6.391`}, {4.18`, 
     6.391`}, {4.102`, 9.156`}, {4.102`, 10.203`}, {4.789`, 
     10.203`}, {4.789`, 9.156`}, {4.711`, 6.391`}, {4.711`, 
     6.391`}, {4.711`, 6.391`}} /. {x_, y_} :> .23 {x - 4.3, y - 7.5};

evalIcon[5] := 
 Graphics[{{EdgeForm[{RGBColor[1., 0.325, 0.319], Thickness[.06]}], 
   RGBColor[1., 0.9, 0.9], Disk[]}, {RGBColor[0.7, 0, 0], 
   Polygon[exclam]}}, ImageSize -> {25., Automatic}]

evalIcon[state_] := 
 Graphics[{{EdgeForm[GrayLevel[.5]], GrayLevel[.9], Disk[]}, 
   diskColors[[state]], 
   Disk[{0, 0}, 1, Pi*({0, .5} + {0, -.5 , -1, -1.5 }[[state]])]}, 
  ImageSize -> {25., Automatic}]

stateTrans = {created|queued -> 1, running[_] -> 2, finished[___] -> 3, dequeued[___] -> 4, _ -> 5}

(* the formatting rules should match only actual objects with at least 4 elements *)

inputFormatter[held_] := Pane[
	Style[Short[held, 0.6], Small],
	ImageSize -> {Full, 18}, ImageSizeAction -> "ResizeToFit"
]

If[ $FEQ && TrueQ[Parallel`Debug`$Debug], (* fancy display *)
    Format[p:EvaluationObject[id_Integer, proc_, _, __], StandardForm] :=
        Interpretation[Framed[Row[{Dynamic[evalIcon[ProcessState[p] /. stateTrans]], 
           Spacer[6], 
           Column[{inputFormatter[Process[p]],
             Dynamic[Style[ProcessState[p] /. {
             	 created -> "new",
             	 queued -> "ready for processing",
                 running[k_] :> StringForm["running on kernel `1`", KernelID[k]],
                 finished[r___] :> "received",
                 dequeued[r___] :> "finished",
                 e_ :> e}, "Label", Gray]]}, ItemSize -> {10, Automatic}]}], 
         RoundingRadius -> 6, FrameStyle -> GrayLevel[.5], FrameMargins -> 5,
          Background -> Dynamic[bgColors[[ProcessState[p] /. stateTrans]]]],
        p]
]

(* fallback *)

Format[p:EvaluationObject[id_Integer, proc_, _, __]] :=
        EvaluationObject[id, Short[proc,0.5], "<>"]


(* format job tag for the benefit of trace *)

$`j/: traceform[$`j[cmd___, id_]] := Subscript["eid",id][HoldForm[cmd]]
$`k/: traceform[$`k[cmd___, id_]] := Subscript["iid",id][HoldForm[cmd]]
Format[$`j[cmd___, id_]] := Subscript["eid",id][HoldForm[cmd]]
Format[$`k[cmd___, id_]] := Subscript["iid",id][HoldForm[cmd]]

(* cleanup *)

Protect[Evaluate[protected]]

Protect[$`j]

SetAttributes[{hot}, {Protected,Locked}]


(* critical sections *)

(* a lock is usually the kernel's $KernelID, treat any nonintegers as unlocked *)

SetAttributes[locked, HoldFirst];
locked[vars_List] := Or @@ locked /@ Unevaluated[vars]
locked[var_] := TrueQ[ValueQ[var] && IntegerQ[var] && var >= 0]

(* list of active lock variables, to be used in interrupt handlers to break deadlocks *)

$activeLocks = Hold[];
SetAttributes[{addLocks, remLocks}, HoldFirst]
addLocks[{vars___}] := ($activeLocks = Union[$activeLocks, Hold[vars]];)
remLocks[{vars___}] := ($activeLocks = Complement[$activeLocks, Hold[vars]];)

(* used inside AbortKernels[] to clean up *)
breakLocks[] := (ReleaseHold[Unset/@$activeLocks]; $activeLocks = Hold[];)

registerResetCode[breakLocks[] ]

(* acquire and release; these are shared functions, but sharing is set up at load time in Client.m *)

SetAttributes[acquire, HoldFirst];
acquire[locks_List, val_] := 
	If[locked[locks], False, Function[var, var = val, HoldAll] /@ Unevaluated[locks]; addLocks[locks]; True]
Protect[acquire]
registerPostInit[ Parallel`Protected`declareSystemDownValue[acquire] ]

(* release always clears values *)

SetAttributes[release, HoldFirst];
release[locks_List] := (Unset /@ Unevaluated[locks]; remLocks[locks]; True)
Protect[release]
registerPostInit[ Parallel`Protected`declareSystemDownValue[release] ]

(* TODO protect from Throw, ... *)

SetAttributes[CriticalSection, HoldAll]
CriticalSection[locks_List, code_] :=
CheckAbort[
	Module[{res},
		While[ ! acquire[locks, $KernelID], QueueRun[] ];
		res = (code);
		release[locks];
	res ]
	, (* when aborted, release locks and propagate abort *)
	release[locks]; Abort[]
]

(* client side definition (similar) is in Client.m *)


End[]

Protect[created, queued, running, finished, dequeued]
Protect[ EvaluationObject ]
Protect[ $QueueLength, $Queue, SetQueueType, $QueueType ]
Protect[ ParallelSubmit,QueueRun,DoneQ,WaitAll,WaitNext,ResetQueues ]
Protect[ Load ]
Protect[ ProcessID, Process, ProcessResult, ProcessState ]
Protect[ ParallelTry, CriticalSection, ConcurrentEvaluate ]
Protect[ breakLocks ]

EndPackage[]
