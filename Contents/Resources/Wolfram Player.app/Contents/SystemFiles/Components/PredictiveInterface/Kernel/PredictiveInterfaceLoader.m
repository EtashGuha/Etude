(* ::Package:: *)

(* ::Section:: *)
(*Load the paclet*)


Get["PredictiveInterface`"]


(* MathLink`CallFrontEnd[FrontEnd`FlushTextResourceCaches[]] *)


(* ::Section:: *)
(*Download updates, if any, in the background*)


Begin["PredictiveInterfaceLoader`Private`"]


backgroundPacletUpdate[pacletName_String] :=
	Module[{p, task, pacletDownloadFunc},
		p = PacletCheckUpdate[pacletName];
		If[Length[p] > 0 && Head[First[p]] === Paclet && $AllowInternet,
			pacletDownloadFunc = If[
				Length[Names["PacletManager`Package`downloadPaclet"]] > 0,
				Symbol["PacletManager`Package`downloadPaclet"],
				Symbol["PacletManager`Manager`Private`downloadPaclet"]
			];
			task = pacletDownloadFunc[First[p]];
			If[Head[task] === AsynchronousTaskObject,
				With[{task = task},
					$startTime = AbsoluteTime[];
					(*
						After starting the download, run the tryFinishUpdate function
						every second. That function is instantaneous if the download
						hasn't finished, but if the download is done, the install process,
						which is synchronous, can take time. To prevent that delay from
						kicking in right away, set this parameter to a higher number.
					*)
					If[!ValueQ[$initialDelay], $initialDelay = 2];
					
					(*
						Wait for $initialDelay seconds, then run the tryFinishUpdate
						function once every second until it removes itself.
					*)
					RunScheduledTask[tryFinishUpdate[task], 1, AbsoluteTime[] + $initialDelay]
				]
			]
		]
	]


(*
Install the paclet once the background download is finished. The install itself is
synchronous, and can take some time, as it involves unzipping the paclet file.
*)

tryFinishUpdate[task_AsynchronousTaskObject] :=
	Quiet[
		Which[
			!MemberQ[AsynchronousTasks[], task],
				(* Task finished with either success or http error. *)
				RemoveScheduledTask[$ScheduledTask];
				PacletInstall[task],
			AbsoluteTime[] - $startTime > 30,
				(*
					The download has taken longer than half a minute, so give up. Stop and
					remove the download task, and remove this scheduled task. This is
					probably unnecessary, since the task should terminate on its own.
				*)
				RemoveScheduledTask[$ScheduledTask];
				StopAsynchronousTask[task];
				RemoveAsynchronousTask[task]
		]
	]


End[]


PredictiveInterfaceLoader`Private`backgroundPacletUpdate["PredictiveInterface"]
