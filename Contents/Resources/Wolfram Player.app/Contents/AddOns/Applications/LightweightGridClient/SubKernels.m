(* :Title: SubKernels.m -- Implement the SubKernels interface for Remote Services kernels *)

(* :Context: LightweightGridClient`SubKernels` *)

(* :Author: Joel F. Klein *)

(* :Copyright: 2010 by Wolfram Research, Inc. *)

BeginPackage["LightweightGridClient`SubKernels`", 
	{"SubKernels`", "LightweightGridClient`"}];

Needs["SubKernels`Protected`"];

RemoteServicesParallelKernel::usage=
"RemoteServicesParallelKernel is the head of an expression representing a
kernel started with Remote Services and used for parallel computing."

RemoteServicesParallelKernel[subContext] = "LightweightGridClient`";

Begin["`Private`"]

Needs["LightweightGridClient`ParallelConfiguration`"];

`AgentShortName = LightweightGridClient`Private`AgentShortName

(* data type:
 	remoteServicesKernel[ lk[link, arglist, speed] ]
 		link	associated LinkObject
 		arglist	list of arguments used in constructor, so that it may be relaunched if possible
 		speed	speed setting, mutable
 *)

SetAttributes[remoteServicesKernel, HoldAll] (* data type *)
SetAttributes[`lk, HoldAllComplete] (* the head for the base class data *)

(* private selectors; pattern is localKernel[ lk[link_, arglist_, id_, ___], ___ ]  *)

remoteServicesKernel/: linkObject[ remoteServicesKernel[lk[link_, ___], ___]] := link
remoteServicesKernel/: descr[remoteServicesKernel[lk[link_, ___], ___]] := 
	With[{info = RemoteKernelInformation[link]},
		AgentShortName[info@"Agent"]
		(*
		If[MatchQ[info, _LightweightGridClient`RemoteServicesKernel],
			AgentShortName[info@"Agent"],
			(* If for some reason we can't get the RemoteKernelInformation, 
				look inside the LinkObject *)
			link[[1]]]*)]
remoteServicesKernel/: arglist[remoteServicesKernel[lk[link_, arglist_, ___], ___]] := arglist
remoteServicesKernel/: kernelSpeed[remoteServicesKernel[lk[link_, arglist_, speed_, ___], ___]] := speed
remoteServicesKernel/: setSpeed[remoteServicesKernel[lk[link_, arglist_, speed_, ___], ___], r_] := (speed = r)

(* factory method *)
LightweightGrid/: SubKernels`NewKernels[args_LightweightGrid, opts:OptionsPattern[]] := 
	Module[{agent, service, localLinkMode, timeout, kernelCount, speed=1, 
		timeLeft, timeoutTime, kernelsLeft, kernelsAtATime, retval={}},

		{agent, service, localLinkMode, timeout, kernelCount} = 
			NormalizeLaunchSettings[args];

    	(* launch feedback through the Subkernels` hook; could add host/agent to name *)
    	feedbackObject["name", StringForm["\[LeftSkeleton]`1` grid kernels\[RightSkeleton]",kernelCount], kernelCount];

		timeoutTime = AbsoluteTime[] + timeout;
		timeLeft[] := timeoutTime - AbsoluteTime[];

		kernelsAtATime = $KernelsAtATime;
		If[!IntegerQ[kernelsAtATime] || !NonNegative[kernelsAtATime],
			kernelsAtATime = 2];

		kernelsLeft = kernelCount;

		While[kernelsLeft > 0 && timeLeft[] > 0.,
			If[kernelsAtATime > kernelsLeft,
				kernelsAtATime = kernelsLeft];
			retval = Join[retval,
				(feedbackObject["tick"];initLink[args, speed][#])& /@ 
					RemoteKernelOpen[Table[agent,{kernelsAtATime}], 
						"Service" -> service, "LocalLinkMode" -> localLinkMode, 
						"Timeout" -> timeLeft[], opts]];
			kernelsLeft -= kernelsAtATime;
		];

		If[kernelsLeft > 0,
			Message[RemoteKernelOpen::timeout, agent];
			(* do not add $Failed to retval, leave it short *)
		];

		retval
	]

(* Convert a RemoteServices description into a LightweightGrid one *)
RemoteServices/: SubKernels`NewKernels[args_RemoteServices, opts:OptionsPattern[]] :=
	SubKernels`NewKernels[LightweightGrid @@ args, opts];

LightweightGrid/: SubKernels`KernelCount[LightweightGrid[{___, "KernelCount" -> n_, ___}]] := n
LightweightGrid/: SubKernels`KernelCount[_LightweightGrid] := 1

NormalizeLaunchSettings[hd_[agent_String]] := 
	NormalizeLaunchSettings[hd[{"Agent" -> agent}]];
NormalizeLaunchSettings[hd_[agent_String, n_Integer]] := 
	NormalizeLaunchSettings[hd[{"Agent" -> agent, "KernelCount" -> n}]];
NormalizeLaunchSettings[_[rules:{_Rule..}]] := 
	NormalizeLaunchSettings[rules];
NormalizeLaunchSettings[rules:{_Rule..}] := 
	{"Agent", "Service", "LocalLinkMode", "Timeout", "KernelCount"} /. 
	Join[rules, {"Agent" -> "", "KernelCount" -> 1}, Options[RemoteKernelOpen]];

getAgent[obj_LightweightGrid] := getProperty[obj, "Agent"]
getService[obj_LightweightGrid] := getProperty[obj, "Service"]
getLocalLinkMode[obj_LightweightGrid] := getProperty[obj, "LocalLinkMode"]
getTimeout[obj_LightweightGrid] := getProperty[obj, "Timeout"]
getProperty[LightweightGrid[{___, key_ -> retval_, ___}], key_] := retval
getProperty[LightweightGrid[{___, key_ -> retval_, ___}], _] := $Failed

(* The setX functions are functional; they don't update in place, they return a 
	new object *)
setService[obj_LightweightGrid, value_] := setProperty[obj, "Service", value]
setLocalLinkMode[obj_LightweightGrid, value_] := 
	setProperty[obj, "LocalLinkMode", value]
setTimeout[obj_LightweightGrid, value_] := setProperty[obj, "Timeout", value]
setProperty[LightweightGrid[{before___, key_ -> _, after___}], key_, value_] :=
	(* key is already found *)
	LightweightGrid[{before, key -> value, after}]
setProperty[LightweightGrid[rules_], key_, value_] := 
	(* key is not found, append it *)
	LightweightGrid[Append[rules, key -> value]]

(* interface methods *)

remoteServicesKernel/:  subQ[ remoteServicesKernel[ lk[link_, arglist_, ___] ] ] := Head[link]===LinkObject

remoteServicesKernel/:  LinkObject[ kernel_remoteServicesKernel ]  := linkObject[kernel]
remoteServicesKernel/:  MachineName[ kernel_remoteServicesKernel ] := descr[kernel]
remoteServicesKernel/:  SubKernels`Description[ kernel_remoteServicesKernel ] := 
	LightweightGrid@@arglist[kernel]
remoteServicesKernel/:  Abort[ kernel_remoteServicesKernel ] := kernelAbort[kernel]
remoteServicesKernel/:  SubKernels`SubKernelType[ kernel_remoteServicesKernel ] := RemoteServicesParallelKernel

(* KernelSpeed: use generic implementation *)

(* kernels should be cloneable; speed setting may have changed after initial launch *)

remoteServicesKernel/:  Clone[kernel_remoteServicesKernel] := 
	First@NewKernels[Description[kernel], KernelSpeed->kernelSpeed[kernel]]

(* list of open kernels *)

`$openkernels = {}
(* This list is represents a subset of the kernels tracked in RemoteServicesLinks[] *)

RemoteServicesParallelKernel[subKernels] := $openkernels

(* destructor; use generic implementation and let it close the link *)

remoteServicesKernel/: Close[kernel_remoteServicesKernel?subQ] := 
	Module[{link = linkObject[kernel], timeout, closeResult},
		timeout = getProperty[arglist[kernel], "Timeout"];
		If[!NumberQ[timeout],
			timeout = "Timeout" /. Options[RemoteKernelClose]];
		closeResult = RemoteKernelClose[link, "Timeout" -> timeout];
		$openkernels = DeleteCases[$openkernels, kernel];
		kernelClose[kernel, True];
		kernel
	];

(* handling short forms of kernel descriptions *)
RemoteServicesParallelKernel[try][Except["localhost" | "local", agent_String], args___] /; 
	StringMatchQ[agent, StartOfString ~~ ("http://" | "https://") ~~ ___] := 
(*
RemoteServicesParallelKernel[try][agent:Except[("local"|"localhost"), _String], args___] := 
	(* TODO client to support probing for recognized hostnames, use that when available *)
*)
	NewKernels[LightweightGrid[agent], args] (* hostname *)

(* class name *)

RemoteServicesParallelKernel[subName] = "remote services kernel"

(* raw constructor *)

initLink[args_, sp_] := Function[link, initLink[link, args, sp]]
initLink[link_LinkObject, args_, sp_] :=
 Module[{kernel, speed = sp},

	(* Tell the remote kernel to initialize as a subkernel *)
	LinkWriteHeld[link, 
		Hold[Parallel`Private`masterLink[LightweightGridClient`Kernel`$MasterLink]]];
	LinkRead[link];

 	kernel = With[{args2=removeKernelCount[args]},
 		remoteServicesKernel[ lk[link, args2, speed] ]];
 	(* local init *)
 	AppendTo[$openkernels, kernel];
 	(* base class init *)
 	kernelInit[kernel]
 ]

initLink[(*link*)_, ___] := $Failed

removeKernelCount[LightweightGrid[args_List]] := 
  LightweightGrid[DeleteCases[args, "KernelCount" -> _]];
removeKernelCount[other_] := other

setFormat[remoteServicesKernel, "remoteservices"]

(* Configuration *)

`Config

RemoteServicesParallelKernel[subConfigure] = Config

Config[configQ] = True
Config[nameConfig] = RemoteServicesParallelKernel[subName]

(* Set live in-memory settings from saved configuration expression *)
Config[setConfig, cfg_] := 
	LightweightGridClient`ParallelConfiguration`loadConfiguration[cfg]

Needs["LightweightGridClient`"];
Needs["LightweightGridClient`UserInterface`"];

(* Return configuration expression from in-memory settings *)
Config[getConfig] := 
	LightweightGridClient`ParallelConfiguration`getConfiguration[]

(* Return launch descriptions from in-memory configuration *)
Config[useConfig] := LightweightGridClient`ParallelConfiguration`getKernels[]

Config[tabConfig] := (
	LightweightGridClient`Private`obtainClient[]; (* start DNS-SD browsing *)
	LightweightGridClient`UserInterface`configEditor[]
);

End[] (*`Private`*)

(* registry *)
If[`$HasRegistered =!= True,
	addImplementation[RemoteServicesParallelKernel];
	`$HasRegistered = True];

EndPackage[]; (* LightweightGridClient`SubKernels` *)
