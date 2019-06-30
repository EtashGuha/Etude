(* Handling of compile-time exceptions. *)

BeginPackage["CompileUtilities`Error`Exceptions`"]

ThrowException
CatchException
CreateFailure
CreateQuietFailure
Finally::usage = "Finally is an option for CatchException that is a function that is called before exiting CatchException."

CompilerException
InternalException

ThrowingTodo


Begin["`Private`"]


CompilerException::usage = "An exception that represents an error in the code being compiled by the Wolfram Language compiler."


(* ThrowException is almost universally called as one of

     ThrowException["msg"]
     ThrowException[{"msg"}]
     ThrowException[{"msg", arg1, arg2,...}]    (often seems like caller expects msg and args to be spliced together into a string. other times the code explicitly does this. )
     ThrowException[{"msg", {args}}]
     
     ThrowException[{expr not string, ...}]
     
     ThrowException[CompilerException[{"msg", arg1, ...}]]
     ThrowException[CompilerException["msg", arg1, ...]]     ( rare )

   Some of the wrappers:
        CompilerException
        InterpreterException
        LanguageException
        InternalException (rare)

   The ThrowException definitions below all work to build to a Throw[assoc, "ThrownExceptionTag"],
   where assoc is <|"Stack" -> {}, "Body" -> whateverWasWrappedInTheThrowException, "Tag" -> (CompilerException | LanguageExeption | NativeException | ...)|>
*)

(* The ordering of these defs is important *)

ThrowException[msg_String] :=
    ThrowException[CompilerException[{msg}]]

ThrowException[body_List] :=
    ThrowException[CompilerException[body]]

ThrowException[tag_[]] :=
    ThrowException[tag[{}]]

ThrowException[tag_[msg_String]] :=
    ThrowException[tag[{msg}]]

ThrowException[tag_[a_Association]] :=
    ThrowException[<|"Exception" -> tag, "Stack" -> {} |> ~Join~ KeyDrop[a, "Exception"]]

(* This is always a re-throw *)
ThrowException[a_Association] :=
    Throw[a, "ThrownExceptionTag"]

(* listify the args if the exception was created with a sequence of args instead of a list.*)
ThrowException[tag_[arg1_, args__]] :=
    ThrowException[tag[{arg1, args}]]

ThrowException[tag_[body_List]] :=
    Throw[<|"Exception" -> tag, "Body" -> body, "Stack" -> {}|>, "ThrownExceptionTag"]

ThrowException[args___] :=
    ThrowException[<|"Exception" -> InternalException, "Body" -> {args}, "Stack" -> {}|>]



(*

Typical usage is:

CatchException[
	expr
	,
	{{_, CreateFailure}}
]

and ThrowException[...] is called somewhere inside of expr


A more elaborate usage is:

CatchException[
	expr
	,
	{{SomeException, doOneThing},
	{SomeOtherException, doAnotherThing},
	{_, CreateFailure}}
]

*)

Options[CatchException] = {Finally -> (Null&)}

SetAttributes[CatchException, HoldAllComplete]

CatchException[expr_, forms:{{_, _}...}, OptionsPattern[]] :=
Module[{expr0, finally = OptionValue[Finally]},
	Catch[
		expr0 = expr;
		finally[];
		expr0
		,
		"ThrownExceptionTag"
		,
		processCatchException[forms, finally]
	]
]

CatchException[expr_, opts:OptionsPattern[]] :=
	CatchException[expr, {}, opts]
	
CatchException[arg___] :=
	ThrowException[InternalException[{"Invalid call to CatchException:", HoldComplete[arg]}]]

(* We want a System` symbol to attach compiler error messages to. We choose Compile, even though that function is part of the old compiler functionality. *)
Compile::err = "`1`"

(*
forms is a list of {pat, func}:
{{SomeException, doOneThing}, {SomeOtherException, doAnotherThing}, {_, CreateFailure}}

Returns a Function f to the Catch, where it is called with f[value, tag] where value and tag are the original Throw arguments. Value will be the Association
from ThrowException and tag will be the literal string "ThrownExceptionTag".
*)
processCatchException[forms_, finally_] :=
	Function[{ass},
		Module[{tag, form, handler},
			tag = ass["Exception"];
			(*
			form is {SomeException, doOneThing}
			*)
			form = SelectFirst[forms, MatchQ[tag, #[[1]]]&];
			If[form === Missing["NotFound"],
				(* exception is not handled, so continue as if it were never caught; rethrow it *)
				finally[];
				ThrowException[ass]
			];
			handler = form[[2]];
			Module[{expr0},
				Catch[
	    			expr0 = handler[ass];
	    			finally[];
	    			expr0
	    			,
	    			"ThrownExceptionTag"
	    			,
	    			Function[{assn},
	    			    finally[];
	    			    ThrowException[assn]
	    			]
	    		]
			]
		]
    ]


(*
Save the $ContextPath at package load time, and use it when calling ToString
This will allow ToString to stringify symbols as, e.g., "AbstractType" instead of the full-qualified "TypeFramework`AbstractType"
*)
$contextPathAtLoadTime = $ContextPath

CreateQuietFailure[a_Association] :=
	Module[{tag, body, isNestedFailure, failure, exc, msg, details},
		exc = Lookup[a, "Exception", InternalException];
        body = Lookup[a, "Body", {}];
		(* For a nested Failure (e.g., the Body of the arriving assoc is something like {TypeFramework`TypeError[Failure[...]]}), we want to use the wrapped
		   Failure (perhaps with some tweaks below), not create a new one that does little more than wrap the old one.
		*)
		isNestedFailure = MatchQ[body, {_Symbol[_Failure]}];
		Block[{$ContextPath = $contextPathAtLoadTime},
			Which[
			    isNestedFailure,
			        (* The Body of the arriving assoc is something like {TypeFramework`TypeError[Failure[...]]}.
                       Use the head (TypeFramework`TypeError in this case) as the Tag element. We could also decide
			           to keep the original tag (e.g., "LookupNotFound"), or accumulate them: {"TypeError", "LookupNotFound"}.
			        *)
			        tag = SymbolName[Head[First[body]]],
			    Developer`SymbolQ[exc],
				    tag = SymbolName[exc],
				True,
				    tag = ToString[exc]
			]
		];
        If[isNestedFailure,
            (* Here we re-use the nested Failure, but replace its tag with the head that wrapped the original Failure (e.g., TypeFramework`TypeError) *)
            failure = ReplacePart[body[[1,1]], 1 -> tag],
        (* else *)
            (* Create a new Failure object *)
            {msg, details} = userMessageFromException[a];
            failure = Failure[tag, <|"MessageTemplate" -> msg, "MessageParameters" -> {}, "Details" -> details|> ~Join~ a];
        ];
		failure
	]
	
CreateQuietFailure[f_Failure] :=
    f


CreateFailure[a_Association] :=
	Module[{failure},
		failure = CreateQuietFailure[a];
		CreateFailure[failure]
	]




issueStringFormMessage[StringForm[ template_, args___]] :=
	Module[{},
		Compile::err1 = template;
		Message[Compile::err1, args]
	]

(*
 Issue a message based on the Failure and return the Failure.
 
 Wrap the ToString to be quiet,  it can happen that the msg has context marks 
 as well as StringForm place holders.
*)
CreateFailure[f_Failure] :=
    Module[{exceptionHead, msg, details, msgTemplate, msgParams},
        exceptionHead = ToString[First[f]];
        msgTemplate = Lookup[f[[2]], "MessageTemplate"];
        msgParams = Lookup[f[[2]], "MessageParameters", Nothing];
        details = Lookup[f[[2]], "Details", None];
  
  		If[Head[msgTemplate] === StringForm && msgParams === {},
  			issueStringFormMessage[msgTemplate];
  			,
	        (*If[!StringEndsQ[msgTemplate, "."], msgTemplate = msgTemplate <> "."];*)
	        
	        (*
	          Turning the details into a string like this often goes very wrong.  I think
	          the message generated should stick with the MessageTemplate and MessageParameters.
	        *)
	        
	        (*If[details =!= None, msgTemplate = msgTemplate <> " Details: " <> ToString[details]];*)
	        (*
	          If the msgParams consider this as template,  else consider as a StringForm.
	        *)
	        If[ AssociationQ[msgParams],
	        	msg = TemplateApply[StringTemplate[exceptionHead <> ". " <> msgTemplate], msgParams]
	        	,
		        msg = StringForm[exceptionHead <> ". " <> msgTemplate, ##]& @@ msgParams];
	        (* This creates a message like "CompilerException: blah blah blah. Details: {blah, blah}" *)
		    Block[{$MessagePrePrint = Identity},
		       Message[Compile::err, Quiet[ToString[msg, StandardForm]]]
		    ]
	    ];
	    f
    ]

CreateFailure[args___] :=
	CreateFailure[<|"Body" -> {args}|>]


(* userMessageFromException tries to make a useful string out of the list of items in the Body element of the thrown exception (these are whatever items the caller wrapped in
   the original ThrowException: ThrowException[{item1, item2, ...}]). It returns a list of ("msg string for user", None | extraDetails_List}. The first item is a string intended
   to become the main text of an error message displayed to the user, and also in the Failure object that will be returned. The second element is extra information that could not be 
   sensibly stringified into the main message. This will become the "Details" element in the Failure association, and appended as "Details: ..." in the message string.
   
   This is all based on heuristics, doing the best we can to do something useful with the wide set of potential arguments supplied in the original ThrowException. Ideally, developers
   will create their own strings in the call to ThrowException, rather than just supplying a list of items with no guidance on how to stringify them.

   Wrap in Quiet to be very defensive. Don't want bugs here to trigger any confusing messages.
*)
userMessageFromException[a_Association] :=
    Quiet @ 
    Module[{items, numItems, firstItem, str},
        (* It should not be possible for "Body" to be absent from the association, but be defensive anyway. *)
        items = Lookup[a, "Body", {""}];
        numItems = Length[items];
        (* First test: If body is just a single string, use it directly. If it just a single expression, ToString it and use it. *)
        If[numItems == 1,
            If[StringQ[First[items]],
                Return[{First[items], None}],
            (* else *)
                Return[{"Error: " <> ToString[First[items]], None}]
            ]
        ];
        (* Getting here means more than one item in body. *)
        firstItem = First[items];
        
        (* Next test: If it is a StringForm-style string, then use StringForm to splice it together. Look for a `1` to decide this. *)
        If[StringQ[firstItem] && StringContainsQ[firstItem, "`1`"],
            Return[{StringForm @@ items, None}]
        ];
        
        (* Next test: Some arg sequences imply that the caller wanted the items stringified and concatenated. The logic here is to
           start with the first string and see if it ends with a space or colon. If so, stringify next item and concat. If next item
           after that is a string that starts with space, concat. If it also ends with a space, stringify and concat next item, etc.
        *)
        If[StringQ[firstItem] && StringEndsQ[firstItem, " " | ":"],
            str = firstItem <> ToString[items[[2]]];
            If[numItems > 2 && StringQ[items[[3]]] && StringStartsQ[items[[3]], " "],
                str = str <> items[[3]];
                If[numItems > 3 && StringEndsQ[items[[3]], " " | ":"],
                    str = str <> ToString[items[[4]]];
                    If[numItems > 4 && StringQ[items[[5]]] && StringStartsQ[items[[5]], " "],
                        str = str <> items[[5]];
                        If[numItems > 5 && StringEndsQ[items[[5]], " " | ":"],
                            str = str <> ToString[items[[6]]]
                        ]
                    ]
                ]
            ];
            Return[{str, None}]
        ];
        
        (* All the previous Return points in this function have consumed all the items in the exception body, leaving nothing for the Detail part.
           Here, we give up on the attempt to create a single string out of the exception, and instead create a Detail component that
           includes all the items in the body after the first one.
        *)
        {ToString[firstItem], Rest[items]} 
    ]



ThrowingTodo[methd_Symbol, args___] :=
	ThrowingTodo[SymbolName[methd], args]
	
ThrowingTodo[methd_String, args___] :=
	ThrowException[{StringJoin["TODO ::: ", methd, " not implemented"], {args}}]



End[]

EndPackage[]
