(* :Package Version: 1.0  *)

(* :Summary:
   a simple client for exercisising subkernels independ of PCT
*)

BeginPackage["SubKernels`Client`", { "SubKernels`" }]

flush::usage = "flush[kernel|{kernel..}] throws away any pending output and checks that the kernel is ready."
send::usage = "send[kernel, expr] sends expr (unevaluated) to kernel for evaluation."
receive::usage = "receive[kernel, h:Identity] returns an evaluation result wrapped in h."
eval::usage = "eval[kernel, expr] evaluates expr on kernel."
readyQ::usage = "readyQ[kernel] is True if input is available from kernel."

Begin["`Private`"]
Needs["SubKernels`Protected`"]

flush[kernel_?subQ] := kernelFlush[kernel]
flush[kernels:{__?subQ}] := kernelFlush[kernels]

SetAttributes[send, {HoldRest,SequenceHold}]

send[kernel_?subQ, expr_] := kernelWrite[kernel, EvaluatePacket[expr]]

receive[kernel_?subQ, h_:Identity] :=
	Module[{},
		While[True,
			Replace[ kernelRead[kernel, Hold], {
				Hold[ReturnPacket[e___]] :> Return[h[e]],
				$Failed :> Return[$Failed],
				Hold[junk_] :> Print[StringForm["Unexpected expression from `1`: `2`.", kernel, HoldForm[junk]]],
				junk_ :> Print[StringForm["Very unexpected expression from `1`: `2`.", kernel, HoldForm[junk]]]
			}];
		];
		$TimedOut
	]

SetAttributes[eval, {HoldRest,SequenceHold}]

eval[kernel_?subQ, expr_, rest___] := receive[ send[kernel, expr], rest ]

readyQ[kernel_?subQ] := kernelReadyQ[kernel]

End[]

EndPackage[]
