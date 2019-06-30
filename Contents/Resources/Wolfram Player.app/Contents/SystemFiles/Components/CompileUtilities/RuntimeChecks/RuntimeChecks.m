
BeginPackage["CompileUtilities`RuntimeChecks`"]

EnableRuntimeChecks


InsertDownValueCatchalls

EnableBadCallChecks


Unhandled



Begin["`Private`"]

catchAllQ
insertCatchAll


Needs["CompileUtilities`RuntimeChecks`Ignored`"]


Unhandled[expr_] :=
Module[{contexts, context},
	(*
		get the contexts used by all of the symbols on the stack

		try to use this to detect which context the current function was defined in

		Just use the first non-System context.
		
		This is very hacky.
	*)
	contexts = Cases[ Stack[_], s_?( Function[{s}, Developer`SymbolQ[Unevaluated[s]], {HoldAll}] ) :>
		Context[s] , {0, Infinity}, Heads->True ];
	contexts = DeleteDuplicates[contexts];
	contexts = DeleteCases[contexts, Alternatives @@ {"System`", "CompileUtilities`RuntimeChecks`",
		"CompileUtilities`RuntimeChecks`Ignored`", "CompileUtilities`RuntimeChecks`Private`", "Developer`"}];
	context = If[contexts === {}, None, contexts[[1]]];
	If[context === None || !MemberQ[$IgnoredContexts, context],
		Block[{$ContextPath = {"System`"}},
			Print[StringTemplate["Unhandled call: `expr`. Context: `context`"][<|"expr"->ToString[FullForm[expr]], "context"->context|>]];
		]
	];
]


EnableRuntimeChecks[] :=
Module[{},
	InsertDownValueCatchalls[];
	EnableBadCallChecks[];
	Print["Runtime checks are enabled"];
]



InsertDownValueCatchalls[] :=
Module[{},

	(* add catch-alls *)

	(* make sure non-System symbols have their full context *)
	oldContextPath = $ContextPath;
	BeginPackage["tmp`"];
	$names = Names[];
	EndPackage[];
	$ContextPath = oldContextPath;

	$names = DeleteCases[$names, s_ /;
		(* delete System` symbols *)
		( Length[StringSplit[s, "`"]] == 1 ||
			(* or ignored symbols *)
			MemberQ[$IgnoredContexts, StringRiffle[Most[StringSplit[s, "`"]], "`"] <> "`"] ) ];

	(* delete symbols that do not have DownValues *)
	$namesForCatchAlls = DeleteCases[$names, s_ /; ToExpression[s, InputForm, DownValues] == {}];

	(*
		this is a bit of a hack right now.

		Do not add a catch-all for a function that also has SubValues,
		since the SubValues are relying on the unevaluated-DownValues call
	*)
	$namesForCatchAlls = DeleteCases[$namesForCatchAlls, s_ /; ToExpression[s, InputForm, SubValues] != {}];

	(*
		delete functions that already have catch-alls
	*)
	$namesForCatchAlls = Cases[$namesForCatchAlls, s_ /; !(Or @@ (catchAllQ /@ ToExpression[s, InputForm, DownValues]))];
	
	If[$Debug, Print["Unprotecting..."]];

	Scan[unprotect, $names];

	If[$Debug, Print["Adding catch-alls to " <> ToString[Length[$namesForCatchAlls]] <> " functions. "]];

	If[$Debug, Print["Top-level contexts: ", Union[(StringSplit[#, "`"][[1]]<>"`")& /@ $namesForCatchAlls]]];

	If[$Debug, Print["All contexts: ", Union[StringJoin[{StringRiffle[Most[StringSplit[#, "`"]], "`"], "`"}]& /@ $namesForCatchAlls]]];

	Scan[insertCatchAll, $namesForCatchAlls];



	(*
	TODO: only protect what was previously protected
	If[$Debug, Print["...Protecting"]];

	Scan[protect, $names];
	*)
]

EnableBadCallChecks[] :=
Module[{},
	(*
		add checks for calls to $Failed[1] or Null[1]
	*)







	(*
	Calls like $Failed[1] are always unhandled
	*)

	Unprotect[$Failed];

	$Failed[args___] /; (!TrueQ[insideFailed]) :=
	Block[{insideFailed = True},
	  Unhandled[$Failed[args]];
	  $Failed[args]
	];

	(*

	$Aborted is locked

	Unprotect[$Aborted];

	$Aborted[args___] /; (!TrueQ[insideAborted]) :=
	Block[{insideAborted = True},
	  Unhandled[$Aborted[args]];
	  $Aborted[args]
	];
	*)


	(*Autoload*)
	Failure;

	Unprotect[Failure];

	Failure[args1___][args2___] /; (!TrueQ[insideFailure]) :=
	Block[{insideFailure = True},
	  Unhandled[Failure[args1][args2]];
	  Failure[args1][args2]
	];


	(*Autoload*)
	Missing;

	Unprotect[Missing];

	Missing[args1___][args2___] /; (!TrueQ[insideMissing]) :=
	Block[{insideMissing = True},
	  Unhandled[Missing[args1][args2]];
	  Missing[args1][args2]
	];

	Unprotect[Null];

	Null[args___] /; (!TrueQ[insideNull]) :=
	Block[{insideNull = True},
	  Unhandled[Null[args]];
	  Null[args]
	];

	Unprotect[None];

	None[args___] /; (!TrueQ[insideNone]) :=
	Block[{insideNone = True},
	  Unhandled[None[args]];
	  None[args]
	];

	Unprotect[Undefined];

	Undefined[args___] /; (!TrueQ[insideUndefined]) :=
	Block[{insideUndefined = True},
	  Unhandled[Undefined[args]];
	  Undefined[args]
	];

	Unprotect[Indeterminate];

	Indeterminate[args___] /; (!TrueQ[insideIndeterminate]) :=
	Block[{insideIndeterminate = True},
	  Unhandled[Indeterminate[args]];
	  Indeterminate[args]
	];

	Unprotect[Expr`EFAIL];

	Expr`EFAIL[args___] /; (!TrueQ[insideEFAIL]) :=
	Block[{insideEFAIL = True},
	  Unhandled[Expr`EFAIL[args]];
	  Expr`EFAIL[args]
	];

	Unprotect[Compile`Uninitialized];

	Compile`Uninitialized[args___] /; (!TrueQ[insideUninitialized]) :=
	Block[{insideUninitialized = True},
	  Unhandled[Compile`Uninitialized[args]];
	  Compile`Uninitialized[args]
	];

	Unprotect[Automatic];

	Automatic[args___] /; (!TrueQ[insideAutomatic]) :=
	Block[{insideAutomatic = True},
	  Unhandled[Automatic[args]];
	  Automatic[args]
	];

	(*

	TODO: Investigate why some kind of hang happens with Integer and Real.

	Unprotect[Integer];

	i_Integer[args___] /; (!TrueQ[insideInteger]) :=
	Block[{insideInteger = True},
	  Unhandled[i[args]];
	  i[args]
	];

	Unprotect[Real];

	r_Real[args___] /; (!TrueQ[insideReal]) :=
	Block[{insideReal = True},
	  Unhandled[r[args]];
	  r[args]
	];
	*)

	(*
		currently too noisy

	s_String[args___] /; (!TrueQ[insideString]) :=
	Block[{insideString = True},
	  Unhandled[s[args]];
	  s[args]
	];
	*)

	(*

	True and False are Locked, so cannot add definitions to them

	*)














	(*
	Calls like Equal[symbol1, symbol2] are unhandled only if they return unevaluated
	*)

	Unprotect[Equal];

	Equal[args___] /; (!TrueQ[insideEqual]) :=
	Block[{insideEqual = True},
	  Module[{res},
	    res = Equal[args];
	    If[Head[res] === Equal,
	      Unhandled[Equal[args]]
	    ];
	    res
	  ]
	];

	Unprotect[LibraryFunction];

	LibraryFunction[args1___][args2___] /; (! TrueQ[insideLibraryFunction]) := 
	Block[{insideLibraryFunction = True}, 
		Module[{res},
			res = LibraryFunction[args1][args2]; 
			If[MatchQ[Head[res], _LibraryFunction],
				Unhandled[LibraryFunction[args1][args2]]
			];
			res
		]
	];












	(*
	Calls like Times[a, b] may or may not be unhandled.
	Times[] and Power[] may be used symbolically.

	So play it safe and just check for bad symbols
	*)

	Unprotect[Times];

	Times[args1___] /; (! TrueQ[insideTimes]) := 
  	Block[{insideTimes = True}, 
   		Module[{res},
   		  res = Times[args1]; 
    	  
   		  (*
   		  	Times is more subtle because it may evaluate to Power: Null*Null evaluates to Power[Null, 2]

   		  	and also it may be used symbolically: not a problem to have a*b for 2 symbols
   		  	and lots of other combinations

   		  	So only test whether a handful of bad expressions is present

   		  *)

   		  If[Head[res] === Times,
		  	If[!FreeQ[res, $Failed | $Aborted | _Failure | _Missing | Null | None | Undefined |
		  			Indeterminate | Expr`EFAIL | Compile`Uninitialized | Automatic],
		  		Unhandled[Times[args1]]
		  	]
   		  ];

    	  res
    	]
    ];


    Unprotect[Power];

	Power[args1___] /; (! TrueQ[insidePower]) := 
  	Block[{insidePower = True}, 
   		Module[{res},
   		  res = Power[args1];
   		  If[Head[res] === Power,
		  	If[!FreeQ[res, $Failed | $Aborted | _Failure | _Missing | Null | None | Undefined |
		  			Indeterminate | Expr`EFAIL | Compile`Uninitialized | Automatic],
		  		Unhandled[Power[args1]]
		  	]
   		  ];

    	  res
    	]
    ];
    
    (*
    enableNonStandardEvalBadCalls[];
	*)
]

EnableRuntimeChecks[args___] :=
	Throw[{"Unrecognized call to EnableRuntimeChecks", args}]


enableNonStandardEvalBadCalls[] := (
	(*
	 	non-standard evaluation is tricky.
	 	
	 	this can happen: Which[a==b, Unevaluated[Sequence[]], True, foo]

	 	but the evaluation here would strip out the Unevaluated
	 *)

	Unprotect[Which];

	Attributes[Which] = {HoldAllComplete};

	Which[args___] /; (!TrueQ[insideWhich]) :=
	Block[{insideWhich = True},
	  Module[{res},
	    res = Which[args];
	    If[Head[res] === Which,
	      Unhandled[Which[args]]
	    ];
	    res
	  ]
	];

	Unprotect[If];

	Attributes[If] = {HoldAllComplete};

	If[args___] /; (!TrueQ[insideIf]) :=
	Block[{insideIf = True},
	  Module[{res},
	    res = If[args];
	    If[Head[res] === If,
	      Unhandled[If[args]]
	    ];
	    res
	  ]
	];

	Unprotect[Switch];

	Attributes[Switch] = {HoldAllComplete};

	Switch[args___] /; (!TrueQ[insideSwitch]) :=
	Block[{insideSwitch = True},
	  Module[{res},
	    res = Switch[args];
	    If[Head[res] === Switch,
	      Unhandled[Switch[args]]
	    ];
	    res
	  ]
	];
)



catchAllQ[Verbatim[RuleDelayed][Verbatim[HoldPattern][_[Verbatim[BlankNullSequence][]]], _]] := True
catchAllQ[Verbatim[RuleDelayed][Verbatim[HoldPattern][_[Verbatim[Pattern][_, Verbatim[BlankNullSequence][]]]], _]] := True
catchAllQ[___] := False


unprotect[symbolName_String] :=
(
	ToExpression[StringTemplate[
		"Unprotect[`symbolName`]; "][
		<|
		"symbolName" -> symbolName,
		"tick" -> "`"
		|>]];
)

insertCatchAll[symbol_String] :=
(
	ToExpression[StringTemplate[
		"`symbol`[args___] /; (!TrueQ[inside`symbolName`]) := \n " <>
		"Block[{inside`symbolName` = True}, \n" <>
		"	CompileUtilities`tick`RuntimeChecks`tick`Unhandled[`symbol`[args]];\n " <>
		"	`symbol`[args]\n " <>
		"]"][
		<|
		"symbol" -> symbol,
		"symbolName" -> StringSplit[symbol, "`"][[-1]],
		"tick" -> "`"
		|>]];
)


protect[symbolName_String] :=
(
	ToExpression[StringTemplate[
		"Protect[`symbolName`]; "][
		<|
		"symbolName" -> symbolName,
		"tick" -> "`"
		|>]];
)


End[]

EndPackage[]
