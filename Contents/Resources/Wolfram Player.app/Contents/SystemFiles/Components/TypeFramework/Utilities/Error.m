(* Wolfram Language package *)

BeginPackage["TypeFramework`Utilities`Error`"]

TypeFailure
$LastTypeError
CatchTypeFailure

ThrowTypeFailure

Begin["`Private`"] (* Begin Private Context *) 

Needs["TypeFramework`"]
Needs["CompileUtilities`Error`Exceptions`"]

$DebugMode = False;
$LastTypeError = None;

ClearAll[TypeFailure]

TypeFailure[] := 
	TypeFailure["GenericError"];


ThrowTypeFailure[f_] := 
(

	If[$DebugMode, Print[f]];
    Internal`$LastUncaughtFailure = f;
    ThrowException[TypeError[stripTypeError[f]]]
)


TypeFailure[tag_String] :=
    StackInhibit @ ThrowTypeFailure @ Failure[tag, <|"MessageTemplate" -> None, "MessageParameters" -> None|>];

TypeFailure[tag_String, msg_String, args___] := 
    StackInhibit @ ThrowTypeFailure @ Failure[tag, <|
    												"MessageTemplate" -> msg,
    												"MessageParameters" -> {args},
    												"Body" -> {args},
    												"Stack" -> {},
    												"Exception" -> tag |>];

TypeFailure[ _, TypeError[f_Failure]] :=
	StackInhibit @ ThrowTypeFailure @ f
	
	
p_TypeFailure := (Set[$LastTypeError, Hold[p]]; TypeFailure["MalformedTypeError", "See $LastTypeError for more information."]);


ClearAll[CatchTypeFailure]

SetAttributes[ CatchTypeFailure, HoldFirst]
	
CatchTypeFailure[ expr_, tag0_, fun_] :=
	With[{tag = tag0},
		Module[ {errTag},
			CatchException[
				expr
				,
				{{
					_,
					Function[
						With[{failure = makeFailure[tag, ##]},
							Assert[FailureQ[failure] || Head[failure] === TypeError];
							errTag = failure[[1]];
							If[Head[errTag] === Failure,
								errTag = errTag[[1]]
							];
							If[ tag === All || MatchQ[errTag, tag] || Head[errTag] === errTag,
								fun[failure],
								ThrowTypeFailure[failure]
							]
						]
					]
				}}
			]
		]
	]
	
stripTypeError[TypeError[f_]] := f
stripTypeError[args___] := args

getTag[Failure[t_, ___]] := t
getBody[Failure[_, args___]] := args
	
ClearAll[makeFailure]
makeFailure[tag_, args_?AssociationQ] /; ListQ[Lookup[args, "Body", None]] &&
								   Length[Lookup[args, "Body"]] >= 1 &&
								   Head[First[Lookup[args, "Body"]]] === TypeError :=
	With[{
		failure = stripTypeError[Part[args["Body"], 1, 1]],
		exception = args["Exception"]
	},
	Module[{
		failureBody = getBody[failure],
		failureException
	},
		failureException = failureBody["Exception"];
		If[MissingQ[exception] && !MissingQ[failureException],
			Return[args]
		];
		failureBody["Exception"] = Which[
			MissingQ[exception],
				failure["Exception"],
			MissingQ[failureException],
				ToString[exception],
			True,
				exception
		];
		failureBody["FailureTagStack"] = Append[
			Lookup[failureBody, "FailureTagStack", {}],
			tag
		];
		TypeError[Failure[getTag[failure], failureBody]] 
	]]
makeFailure[_, args___] := CreateQuietFailure[args]

End[];

EndPackage[];
