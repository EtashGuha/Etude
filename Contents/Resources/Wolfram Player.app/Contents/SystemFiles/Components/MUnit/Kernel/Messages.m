
Begin["`Package`"]

$MessageWithArgsList

$MUnitMessageHandler

MUnitMessageHandler

messageStringHandler

$MessageStringList

messageNiceStringHandler

$MessageNiceStringList

$bitbucket

End[]

Begin["`Messages`Private`"]

(*
$bitbucket is an OutputStream used by MUnit internally.
A side-effect of this implementation is that MUnit leaks out this stream (via Streams[])
*)
$bitbucket = Module[{f = Switch[$OperatingSystem, "Windows", "NUL", "MacOSX" | "Unix" | "iOS", "/dev/null"], strs},
	            strs = Streams[f];
	            If[strs === {},
	            	(* bitbucket stream is not open yet, so open it *)
	            	Quiet@OpenWrite[f]
	            	,
	            	strs[[1]]
	            ]
             ]

(*
The args pattern used to be args___HoldForm, but then we realized that
message arguments are not always wrapped in HoldForm.
Some times they are, some times they are not.
Note that this is different than expected messages, which are never expected to have HoldForm around the args.

MUnitMessageHandler is added to the list of Internal`Handlers[] in MUnit.m
*)
$MUnitMessageHandler[ctlString_String, Hold[msg_MessageName], Hold[Message[msg_MessageName, args___]]] :=
	(
	If[(* stdout would have been in $Messages *)
		$Messages === {$bitbucket} ||
			(* $Messages has been changed, most likely to $Messages or {} *)
			(ListQ[$Messages] && MemberQ[$Messages, OutputStream["stdout", _]])
		,
		AppendTo[$MessageWithArgsList, HoldForm[Message[msg, args]]];
	])

(*
TODO: remove mention of MUnit`Test`Private
Test.m is the only user of messageStringHandler right now, but it would be better to make more modular
*)
messageStringHandler[ctlString_String, Hold[msg_MessageName], Hold[Message[msg_MessageName, args___]]] :=
	(
	If[(* stdout would have been in $Messages *)
		$Messages === {$bitbucket} ||
			(* $Messages has been changed, most likely to $Messages or {} *)
			(ListQ[$Messages] && MemberQ[$Messages, OutputStream["stdout", _]])
		,
		Block[{$Context = "MUnit`Test`Private`"},
			AppendTo[$MessageStringList, ToString[StringForm[ctlString, args]]];
		]
	])

(* messageNiceStringHandler provides more detailed error messages and explains how to fix problems.
It is used in the pre-processing loop in TestRun. *)
messageNiceStringHandler[ctlString_String, Hold[msg_MessageName], Hold[Message[msg_MessageName, args___]]] :=
	(
	If[(* stdout would have been in $Messages *)
		$Messages === {$bitbucket} ||
			(* $Messages has been changed, most likely to $Messages or {} *)
			(ListQ[$Messages] && MemberQ[$Messages, OutputStream["stdout", _]])
		,
		Switch[Unevaluated[msg],
			HoldPattern[Syntax::sntunc],
			If[{args} === {HoldForm[""]},
				AppendTo[$MessageNiceStringList,
					"You have \"\\[\" in your code. You probably meant to have \"\\\\[\"." <>
					"\nOriginal Message:\n" <> ToString[StringForm[ctlString, args]]];
				,
				AppendTo[$MessageNiceStringList, ToString[StringForm[ctlString, args]]];
			];
			,
			HoldPattern[Syntax::com],
			AppendTo[$MessageNiceStringList,
				"You have stray commas in your code." <>
				"\nOriginal Message:\n" <> ToString[StringForm[ctlString, args]]]
			,
			_,
			AppendTo[$MessageNiceStringList, ToString[StringForm[ctlString, args]]];
		];
	])

End[]
