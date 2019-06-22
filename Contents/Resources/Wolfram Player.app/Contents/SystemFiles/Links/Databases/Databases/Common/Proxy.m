Package["Databases`Common`"]

PackageImport["Databases`"]


PackageExport["DBCreateProxyDefinitions"]


$defaultMessageHandler := $defaultMessageHandler = DBCreateMessageHandler[]

$identityMessageHandler = Identity

(* TODO: replace $identityMessageHandler with $defaultMessageHandler, when we fix the
message-handling *)

SetAttributes[messageHandler, HoldAll]
messageHandler[code_] := $identityMessageHandler[code]

(*
**  Main error handler for external functions. It takes the following arguments:
**
**   - failureMapping:  an assoc of the form
**
**          <| "failureType" -> Function[{sym, failure}, ...], ... |>
**
**       where the value can be any function (not necesarily pure function) with
**       arguments:
**
**          - sym:  top-level symbol whose execution led to the failure
**          - failure: resulting internal FailureObject
**
**      which is free to perform any necessary actions (issue messages etc), and
**      return the final result
**  - sym: top-level symbol whose execution led to the failure
**  - failure: resulting internal FailureObject
**
**  In addition to dedicated specific error handlers, it automatically handles
**  thrown Failures, either corresponding to errors for which no dedicated handlers
**  were supplied, or caused by error messages issued by inner function calls (the
**  error type "message_generated_in_inner_function_call")
*)

ClearAll[errorHandler];
errorHandler[failureMapping_Association?AssociationQ][sym_, failure_] :=
	With[{errorType = DBGetErrorType[failure]},
		Which[
			KeyExistsQ[failureMapping, errorType],
				failureMapping[errorType][sym, failure]
            ,
			errorType === "message_generated_in_inner_function_call",
				Failure["DatabasesFailure", <|
					"FailureType" -> "InternalMessage",
					"Messages" -> $MessageList
				|>]
            ,
			True,
				failure
		]
	]


(* TODO: add generic automated option validation functionality *)


(*
**  Function to construct proxied definitions. Takes the following arguments:
**
**      - sym: top-level symbol that would serve as a proxy
**      - defs: _List | _RuleDelayed - proxied definitions as local rules
**      - failureMapping: an assoc mapping failure types (strings) to handlers
**      - options
**
**  Creates proxied definitions for a symbol, which add error-handling step, as well
**  as handle internal messages issued during function's execution
**
**  Note that DBCreateProxyDefinitions is not a "complete" solution, intentionally.
**  It doesn't care about top-level symbol's other rules (e.g. what happens when
**  arguments don't match the pattern, top-level symbol atttributes, etc.). It only
**  creates proxied definitions proper. This allows it to be minimally opinionated
**  about symbol's behavior.
*)
ClearAll[DBCreateProxyDefinitions]
SetAttributes[DBCreateProxyDefinitions, HoldFirst]
Options[DBCreateProxyDefinitions] = {
	"ErrorHandler" :> errorHandler,
	"MessageHandler" :> messageHandler
};
DBCreateProxyDefinitions[
	sym_Symbol,
	defs_List,
	failureMapping: _Association?AssociationQ: <||>,
	opts: OptionsPattern[]
] :=
	Scan[DBCreateProxyDefinitions[sym, #, failureMapping, opts]&, defs ]

DBCreateProxyDefinitions[
	sym_Symbol,
	lhs_ :> rhs_,
	failureMapping: _Association?AssociationQ: <||>,
	opts: OptionsPattern[]
] :=
	With[{
		errorHandler = OptionValue["ErrorHandler"],
		messageHandler = OptionValue["MessageHandler"]
		},
		lhs := DBHandleError[sym, errorHandler[failureMapping]][
            messageHandler[rhs]
        ]
	]
