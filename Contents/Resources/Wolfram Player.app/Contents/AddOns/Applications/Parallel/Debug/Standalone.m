(* :Title: Debug/Standalone -- debugging support for Parallel Tools *)

(* :Context: Parallel`Debug` *)

(* :Author: Roman E. Maeder *)

(* :Summary:
   user debug functions for the Parallel Computing Toolkit.
   Allow monitoring communication with remote kernels and queueing of processes.
 *)

(* :Package Version: 3.0  *)

(* :Mathematica Version: 7 *)

(* :Note: this package must be read *before* the Parallel Computing Toolkit itself *)

(* check for loading after PCT *)

If[ NameQ["Parallel`Private`$PackageVersion"],
	Parallel`Debug`$Debug::toolate = "The debugging package cannot be read after the Parallel Computing Toolkit itself.";
	Message[Parallel`Debug`$Debug::toolate];
	Abort[]
]


BeginPackage["Parallel`Debug`"]

`$Debug::usage = "$Debug is True, if parallel debugging is enabled."
`$Parallel::usage = "$Parallel is the parallel debug object."

OptionValues::usage = "OptionValues[debugoption] gives the possible values of a debug option of $Parallel."

Tracers::usage = "Tracers->{tracers...} is a debug option of $Parallel that specifies events to trace. OptionValues[Tracers] gives the names of possible tracers."
(* the tracers are defined in the packages where they are used *)

TraceHandler::usage = "TraceHandler->\"Print\"|\"Save\" is a debug option of $Parallel that specifies how trace events are handled."

newTraceList::usage = "newTraceList[] initializes the trace list."
TraceList::usage = "TraceList[] gives the current list of trace events."

(* globally defined trace triggers, packages may add additional ones *)

OptionValues[Tracers] = {}
OptionValues[TraceHandler] = {"Print", "Save", "Display"}

Options[$Parallel] = {
	Tracers -> OptionValues[Tracers],
	TraceHandler->"Print"
}

BeginPackage["Parallel`Protected`"]

(* performance measurement hooks *)
`parStart
`parStop

(* tracing helpers *)

`traceform

EndPackage[]


Begin["`Private`"]

`$PackageVersion = 3.0;
`$thisFile = $InputFileName

$Debug = True (* to enable debugging conditional code in the PCT *)

{`$trace, `$traceHandler} (* local option values, clean copies *)

(* trace functions and converter *)

(* template for trace calls:
	Parallel`Debug`Private`trace[Parallel`Debug`TRIGGER, "stringform template", arguments...]
*)

`trace (* the function variable whose value is one of the trace functions print/save *)

traceHandlerRules = {
	"Print" -> `tracePrint,
	"Save"  -> `traceSave,
	"Display" -> `traceDisplay
}

(* handle all option settings *)

$Parallel/: SetOptions[$Parallel, opts0:OptionsPattern[]] :=
    Module[{names, tracers, th, opts = Flatten[{opts0}]},
        names = First /@ opts; (* all options modified in this call *)

		(* Trace *)
        If[ MemberQ[names, Tracers], 
            tracers = Tracers /. opts;
            If[tracers===None, tracers={}]; (* lazyness *)
            If[Head[tracers]===Symbol, tracers = {tracers} ]; (* lazyness *)
            If[ListQ[tracers],
            	$trace = Intersection[tracers, OptionValues[Tracers]],
            	Return[$Failed] (* syntax *)
            ]
        ];

		(* TraceHandler *)
        If[ MemberQ[names, TraceHandler], 
            th = TraceHandler /. opts;
            If[ MemberQ[OptionValues[TraceHandler], th],
                $traceHandler = th;
                trace = $traceHandler /. traceHandlerRules;
            ]
        ];

        (* keep track of changed (and unchanged) settings *)
        Unprotect[$Parallel];
        Options[$Parallel] = {
            Tracers -> $trace,
            TraceHandler -> $traceHandler
        };
        Protect[$Parallel];

        Options[$Parallel, names]	(* return the new settings *)
    ]

SetOptions[$Parallel, Tracers->{}, TraceHandler->"Print"] (* init *)

(* stuff to be used in the PCT code.
   Use with full context: Parallel`Debug`Private`*
*)


(* tracing *)

(* need reasonably fast AppendTo alternative *)

SetAttributes[tlh, HoldFirst] (* our cons() *)
`$traceList (* the global trace event list *)

newTraceList[] := ($traceList = tlh[];)

newTraceList[] (* init *)

TraceList[] := Reverse[List@@Flatten[$traceList, Infinity, tlh]]

(* thefunctions for Print/Save tracing, these can be assigned to `trace *)
(* CheckAbort ensures that we do not waste time printing something if we trace
   inside AbortProtect[] *)

tracePrint[trap_, sf_, args___]/; MemberQ[$trace, trap] :=
	CheckAbort[Print[trap, ": ", StringForm[sf, args]], Abort[]]
traceSave[trap_, sf_, args___]/; MemberQ[$trace, trap] :=
	($traceList = tlh[{trap, StringForm[sf, args]}, $traceList];)
traceDisplay[trap_, sf_, args___]/; MemberQ[$trace, trap] :=
	addLine[trap, sf, args]
(* to ignore tracing , e.g. during status updates *)
traceIgnore[___] := Null

(* register trace triggers with optional usage message *)

`RegisterTrace[symbol_Symbol, msg_:None] := (
	OptionValues[Tracers] = Union[OptionValues[Tracers], {symbol}];
	If[ Head[msg] === String, symbol::usage = msg ];
)

(* formatting for tracing catchall *)

traceform[junk_] := Short[junk, 0.5]

(* hiding certain values in trace output *)

`$hideVals = False;


(* wrapper for WWB into DistributeDefinitions; should be Identity if not used *)

If[ !ValueQ[`ExportEnvironmentWrapper], `ExportEnvironmentWrapper=Identity ]

(* wrapper for WWB into EvaluatePacket[], should not have a value if not used *)

`RemoteEvaluateWrapper

(* hook for WWB kernel initialization in initKernel[]; need not have a value if not used *)

`RemoteKernelInit

End[]

Protect[ $Parallel, Tracers, TraceHandler, newTraceList, TraceList ]

EndPackage[]
