(* :Title: SubKernels/Kernel/init.m -- subkernel interface *)

(* :Context: SubKernels` *)

(* :Author: Roman E. Maeder *)

(* :Copyright: © 2008 by Wolfram Research, Inc. *)

(* :Package Version: 2.0  *)

(* :History:
   1.0 first released version.
   2.0 new launch feedback, parallelize low-level communication
*)

(* :Summary:
   Needs["SubKernels`"]
*)

(* this is the abstract base class for remote kernels *)

(* Discussion:
    Description[] and Clone[] can also be used on dead kernels.
    constructors that fail return $Failed
 *)

BeginPackage[ "SubKernels`" ]

(* class name as symbol, used for instance for InstanceOf *)

SubKernels::usage = "SubKernels is the subkernels interface."

(* class variables and methods *)

$SubKernelTypes::usage = "$SubKernelTypes is the list of available implementations of the subkernels interface."
$SubKernels::usage = "$SubKernels gives the list of open subkernels."

(* factory methods *)

NewKernels::usage = "NewKernels[subkernel description, opts...] makes a new subkernel or a number of subkernels available."

Clone::usage = "Clone[kernel] make a new kernel available that has the same specification as the given one."

(* destructor *)

System`Close (* do not add message *)

(* methods *)

subQ::usage = "subQ[kernel] is True, if kernel is a remote kernel object implementing the RemoteKernels interface."

System`LinkObject (* do not add message *)

Description::usage = "Description[kernel] gives the subkernel description that was used to launch this kernel."

(* we need this in System` because it is also used (by delegation) in Parallel`Kernels` *)
If[!ValueQ[System`KernelSpeed::usage], System`KernelSpeed::usage = "KernelSpeed[kernel] is a relative performance measure.
	KernelSpeed[kernel] = r sets the speed.
	KernelSpeed->r can be used to specify the speed in the constructor."]

SubKernelType::usage = "SubKernelType[kernel] is the type (class object) of a subkernel."

(* methods for kernel descriptions *)

KernelCount::usage = "KernelCount[descr] gives the number of kernels that are expected to be launched from descr."

(* stuff that appears in more than one implementations, so it needs to be pulled up here *)

$RemoteUsername::usage = "$RemoteUsername is the default user (or login) name on a remote machine. Its default is $Username."


(* semi-hidden methods *)

BeginPackage["SubKernels`Protected`"]

(* instance variables [most likely] *)

kernelSpeed::usage = "kernelSpeed[kernel] is the kernel's speed setting."
setSpeed::usage = "setSpeed[kernel, r] sets the kernel's speed to r."

(* aux *)

kernelRead::usage = "kernelRead[kernel] reads an expression from the kernel's link object.
	kernelRead[kernel, h] wraps the result in h before returning it."
kernelWrite::usage = "kernelWrite[kernel, expr] writes expr unevaluated to the kernel's link object."
kernelFlush::usage = "kernelFlush[kernel] clears any pending output and checks that kernel is alive."
kernelReadyQ::usage = "kernelReadyQ[kernel] is True, if input is available from kernel."

kernelInit::usage = "kernelInit[kernel]	initializes a new kernel or list of kernels."
kernelAbort::usage = "kernelAbort[kernel] aborts kernel by sending mathlink abort and cleaning up."
kernelClose::usage = "kernelClose[kernel] base class destructor. optionally closes the mathlink contained within."

(* methods of the implementation's class object CLASS *)

subName::usage = "CLASS[subName] is the name of an implementation."
subContext::usage = "CLASS[subContext] is the context of an implementation."
subKernels::usage = "CLASS[subKernels] is the list of all kernels of an implementation."
(* optional *)
try::usage = "CLASS[try][...] is the heuristic constructor of an implementation."
subConfigure::usage = "CLASS[subConfigure] is the configuration class object."

(* methods of the configuration class object CONFIG *)

configQ::usage = "CONFIG[configQ] is True if configuration is provided."
setConfig::usage = "CONFIG[setConfig, (val)] initializes the local configuration from external sources."
getConfig::usage = "CONFIG[getConfig] gives an external representation suitable for persistent storage."
useConfig::usage = "CONFIG[useConfig] gives the list of kernel descriptions of the current configuration."
tabConfig::usage = "CONFIG[tabConfig] is a user interface for editing configurations."
nameConfig::usage = "CONFIG[nameConfig] is the name of this configuration (the subkernel implementation name)."

(* aux stuff *)


`deleteFailed
`firstOrFailed

Spinner::usage = "Spinner[Dynamic[var,(updater)],{min,max,step}] is a spinner.
	The default min is 0, max is Infinity, and step is 1."

`setFormat
`addImplementation::usage = "addImplementation[classobject] adds an implementation to the registry."
`stdargs::usage = "standard cmdline argument for subkernels (excluding mathlink arguments)"

(* for use by launch feedback *)
`$initCounter::usage = "$initCounter is incremented for each call of the constructor."

(* launch feedback hooks; called by subclass constructors, assigned by client code, such as PT kernel launcher *)

`feedbackObject

EndPackage[]

(* private methods; use with full name SubKernels`Private`WAWA *)
(*
noclone[kernel]		a version of Clone[] that fails (for kernels that are not cloneable)
*)


(* messages *)

SubKernels::registry = "Registration `1` failed."
NewKernels::argchk = "Desired numbers of kernels are not integers satisfying `1`<=`2`<=`3`."
General::time = "Operation `1` timed out after `2` seconds."
General::lnk = "MathLink call `1` failed."
SubKernels::noclone = "Kernel `1` is not cloneable."
General::timekernels = "Timeout for subkernels. Received only `1` of `2` connections."
General::somefail = "`2` of `1` kernels failed to launch."

Begin["`Private`"]

`$PackageVersion = 1.0;
`$thisFile = $InputFileName

(* we use the Parallel` debug interface *)

If[ !ValueQ[Parallel`Debug`$Debug], Get["Parallel`Debug`Null`"] ]

(* this is also used in Parallel`Kernels`, but we need it standalone *)
Parallel`Debug`Private`RegisterTrace[ Parallel`Debug`MathLink, "MathLink is a tracer that triggers when MathLink commands are issued to launch or close remote kernels." ]

(* private class variables *)

stdargs = " -subkernel -noinit -nopaclet"

`$implementations = {};

(* timeout for external commands: use Parallel`Settings`$MathLinkTimeout *)


(* generics *)

(* by default, kernels are not cloneable *)

Clone[kernel_?subQ] := (Message[SubKernels::noclone, kernel]; $Failed)

(* Close[] and Abort[] also have generic implementations, but we cannot use upvalues! *)

(* speed setting, using the protected speed/setSpeed methods *)

KernelSpeed[kernel_?subQ] := kernelSpeed[kernel]
KernelSpeed/: (KernelSpeed[kernel_] = new_)/; subQ[kernel] && new>0 := setSpeed[kernel, new]

subQ[junk_] = False (* catchall *)

(* default count is 1 *)

KernelCount[_] := 1

(* protected/private methods *)

deleteFailed[l_List, msghead_:Null] :=
With[{nl = DeleteCases[l, $Failed]},
	If[Length[nl]<Length[l] && msghead=!=Null, Message[msghead::somefail, Length[l], Length[l]-Length[nl]]];
	nl
]

firstOrFailed[l_List] := First[l, $Failed]

addImplementation[obj_] := AppendTo[`$implementations, obj]
t:addImplementation[___]/; Message[RemoteKernels::registry, HoldForm[t]] := Null (* syntax error *)

(* smarter LinkRead; tries to return $Failed in more cases *)
(* instrumented versions; avoid creating symbols if not used *)

kernelRead[kernel_?subQ, h___] :=
    With[ {res = LinkRead[LinkObject[kernel], h]},
		Parallel`Debug`Private`trace[Symbol["Parallel`Debug`WriteRead"], "Reading from `1`: `2`", kernel, HoldForm[res]];
        If[ Head[res] === LinkRead, $Failed, res ]
    ]

(* should return True in case of failure *)

kernelReadyQ[kernel_?subQ] := LinkReadyQ[LinkObject[kernel]]=!=False

(* the setting in kernelInit[] above ensures that only system symbols are sent unqualified.
  as a result, any new symbols in contexts other than the system context are created in
  the correct context. New symbols in the system context will be created in the global
  context, as the remote context path is still set to its default
*)

SetAttributes[{kernelWrite, kernelWriteTimed}, {HoldRest,SequenceHold}]

kernelWrite[kernel_?subQ, expr_] := (
    Parallel`Debug`Private`trace[Symbol["Parallel`Debug`WriteRead"], "Writing to `1`: `2`", kernel, HoldForm[expr]];
	With[{res = LinkWrite[LinkObject[kernel], Unevaluated[expr]]},
    	If[res===$Failed || Head[res]===LinkWrite, Return[$Failed] ];
    	kernel
	]
)

kernelWriteTimed[kernel_?subQ, expr_] :=
    Module[{res},
        TimeConstrained[
            res = kernelWrite[kernel, expr],
            Parallel`Settings`$MathLinkTimeout,
            Message[kernelFlush::time, LinkWrite, Parallel`Settings`$MathLinkTimeout];
                Return[$Failed]
        ];
        res
    ]

(* flush, parallel version *)

(* todo: timeout read *)

kernelFlush[kernels0:{___?subQ}] :=
	With[{lux = `ConnectionTest},
    Module[{kernels = kernels0, good = {}, res, posbad, posgood},
        kernels = kernelWriteTimed[#, EvaluatePacket[lux]]& /@ kernels;
        kernels = deleteFailed[kernels, LinkWrite];

        (* concurrent read until we receive the correct reply from each *)
        While[Length[kernels]>0,
        	res = kernelRead[#, Hold]& /@ kernels; (* read one item from each *)
        	posbad  = Position[res, $Failed];
         	posgood = Position[res, Hold[ReturnPacket[lux]]];
        	good = Join[good, Extract[kernels, posgood]];
       		kernels = Delete[kernels, Join[posbad,posgood]];
        ];
        good
    ]]

kernelFlush[kernel_?subQ] := firstOrFailed[ kernelFlush[{kernel}] ]


(* abort helper function, which implementations may use *)
(* note: do not use if the kernel is not busy; it may crash otherwise *)

kernelAbort[kernel_?subQ] :=
	Module[{res},
	    Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "Aborting `1`", kernel];
		LinkInterrupt[LinkObject[kernel], 3];
		kernelWrite[kernel, EvaluatePacket[`throwaway]]; (* must send something here in case it was idle *)
		res = kernelFlush[kernel];
		If[res===$Failed, res, $Aborted]
	]

(* initializer, must be called by implementations (just like a base class constructor) *)

(* more careful waiting for links to become ready; do not wait for actual LinkReadyQ.
   Although this should only be called with subQ objects, we silently ignore any junk.
 *)

$timeouts={10.0}

kernelInit[kernels0_List] := Module[{kernels,incoming, connected,time, wt, selected, work, to, tol=$timeouts},
	kernels = Select[kernels0, subQ];
	incoming = AssociationThread[Range[Length[kernels]]->(LinkObject/@kernels)]; (* index them *)
	connected = Association[];
	If[ListQ[tol], to=First[tol]; tol=Rest[tol], to=tol; tol={}];
	time = AbsoluteTime[]; wt = 0;
	CheckAbort[
		While[Length[incoming]>0 && wt<to,
			selected = Select[ incoming, LinkConnectedQ ]; work=Length[selected];
			connected = Join[connected, selected]; KeyDropFrom[incoming, Keys[selected]];
			(* call LinkConnect, but only for those non-connected ones that are ready *)
			Scan[ If[LinkReadyQ[#], LinkConnect[#]]&, incoming]; (* does not alter links *)
			If[ work == 0, (* no progress: wait a bit and increment timer *)
				Pause[Parallel`Settings`$BusyWait]; wt = AbsoluteTime[] - time;
			  , (* else: reset timer *)
			  	time = AbsoluteTime[]; wt = 0;
			  	If[Length[tol]>1, to=First[tol]; tol=Rest[tol]];
			];
		];
		If[ Length[incoming]>0, (* too bad *)
			Message[SubKernels::timekernels, Length[connected], Length[kernels]];
			Quiet[LinkClose /@ incoming];
		];
		, (* clean up after abort *)
		Quiet[LinkClose /@ incoming];
	];
	MathLink`LinkSetPrintFullSymbols[#, True]& /@ connected;
	(* pick the good ones *)
	Part[kernels, Sort[Keys[connected]]]
]

kernelInit[kernel_?subQ] := firstOrFailed[kernelInit[{kernel}]]

(* destructor; optional link close with timeout. *)

kernelClose[kernel_?subQ, doClose_:False] :=
Module[{},
	If[ doClose,
 	   Parallel`Debug`Private`trace[Parallel`Debug`MathLink, "Closing link `1`.", LinkObject[kernel]];
 	   TimeConstrained[
        Quiet[LinkClose[LinkObject[kernel]]],
        Parallel`Settings`$MathLinkTimeout,
        Message[LinkClose::time, LinkClose, Parallel`Settings`$MathLinkTimeout]
	   ];
	];
    (* all done *)
    kernel
]

(* heuristic for recognizing short forms of kernel descriptions; is used if no upvalue for New[] fires
   may return a kernel or list of kernels (in multi-argument form) *)

subOrFailedQ[kernel_] := subQ[kernel] || kernel===$Failed

NewKernels[wawa__] :=
Module[{res},
	Catch[Scan[Function[class,
				res=class[try][wawa];
				If[MatchQ[res, _?subOrFailedQ | {___?subOrFailedQ}], Throw[res]]
		],
		Reverse[$implementations]
	];$Failed]
]

(* read-only public fields *)

$SubKernelTypes := $implementations

$SubKernels := Join @@ Composition[Through, $implementations][subKernels]

(* formatting helper for subkernel objects *)

setFormat[type_, head_] := (Format[q_type?subQ] := head[MachineName[q]])

(* constants for implementations *)

If[ !ValueQ[$RemoteUsername], $RemoteUsername = $Username ]

(* our very own spinner *)

Options[Spinner] = {
	Enabled->True
}

Spinner[dvar_, opts:OptionsPattern[]] := Spinner[dvar, {0, Infinity}, opts]
Spinner[dvar_, {x0_, x1_}, opts:OptionsPattern[]] := Spinner[dvar, {x0, x1, 1}, opts]

Spinner[Dynamic[dvar_], args___] := Spinner[Dynamic[dvar, (dvar = #) &], args] (* provide default updater *)

Spinner[Dynamic[dvar_, updater_], {x0_, x1_, dx_}, opts:OptionsPattern[]] :=
Module[{enabled, force},
  enabled = Enabled -> OptionValue[Enabled];
  If[(IntegerQ[x0] || x0 === -Infinity) && (IntegerQ[x1] || x1 === Infinity) && IntegerQ[dx],
  	force[x_] := Max[x0, Min[Round[x], x1]], force[x_] := Max[x0, Min[N[x], x1]]];
  Row[{InputField[Dynamic[dvar, updater[force[#], dvar] &], 
     Number, System`Utilities`FilterOptions[InputField,opts], FieldSize -> {{2,5},1}, enabled], 
    ButtonBar[{"+" :> updater[force[dvar + dx], dvar], 
 			   "-" :> updater[force[dvar - dx], dvar]},
     ImageSize -> Small, enabled]}]
]


(* debug warnings *)

If[TrueQ[Parallel`Debug`$Debug],
	(#[args___] := Throw[{#, args}])& /@ {kernelClose, kernelInit, kernelAbort, kernelFlush, kernelRead, kernelWrite, kernelReadyQ}
]

Protect[ kernelRead, kernelWrite, kernelFlush, kernelReadyQ, kernelInit, kernelAbort, kernelClose ]

End[]

Protect[ SubKernels, $SubKernelTypes, $SubKernels, NewKernels, Clone, subQ, Description, KernelSpeed, SubKernelType, KernelCount ]

EndPackage[]
