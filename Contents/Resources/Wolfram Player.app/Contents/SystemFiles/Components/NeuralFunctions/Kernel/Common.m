PackageScope["DBPrint"]
SetUsage[DBPrint,
"
DBPrint[expr$] prints the expression expr$ if $NeuralFunctionsDebug is set to True.
"
]
DBPrint[expr___] /; GeneralUtilities`$DebugMode := Print[expr]


PackageScope["GetOption"]
SetUsage[
GetOption,
"
GetOption[name$] is a utility which can be used to access option values of a caller function from the callee.

GetOption has to be locally redefined from the caller (within a Block or Scope) in the following way:
	GetOption = OptionValue[{CallerSymbol$, HiddenOptions$}, PassedOptions$, #]&;
where HiddenOptions$ (if present) is a list of additional options not exposed \
in Options[CallerSymbol$] and PassedOptions$ is the list of options passed from top level.

Otherwise, it is automatically available when using DefineFunction.

GetOption is an unevaluated symbol outside of the caller scope.
"
]

PackageScope["DefineFunction"]
PackageScope["$FunctionName"]
PackageScope["$AllowFailure"]

SetUsage[
DefineFunction,
"
DefineFunction[symbol$, interface$, argnum$] is a utility that adds a definition for symbol$.

It creates a downvalue for symbol$ which evaluates interface$[arguments$, options$] \
where arguments$ and options$ are parsed using

	System`Private`Arguments[symbol$[input], argnum$]

It supports the following options:
| \"ArgumentsPattern\" | Automatic | pattern for arguments of the top-level function |
| \"ExtraOptions\" | {} | a list of extra options not declared in Options[symbol$] |
| 'Parser' | Arguments | function to validate argument number and options |
| \"Wrapper\" | List |  a wrapper for the arguments$ and options$ sequence |
Option \"ArgumentsPattern\" defaults to ___. Can be set to a more restrictive pattern to \
enable unevaluated operator forms like Map[f].
Option 'Parser' can take the following values:
* System`Private`Arguments
* System`Private`ArgumentsWithRules

DefineFunction defines the utility GetOption as

	GetOption[name_] := OptionValue[{symbol$, extraOpts$}, options$, name]

and the variable

	$FunctionName = symbol$

that can be used anywhere in the dynamic scope of symbol$.

ThrowFailure[tag$, args$] can be used inside the function scope \
and it is caught by CatchFailureAsMessage using symbol$ as message head. When a failure \
occurs, the return value can either be the unevaluated call of $Failed, depending on the \
value of $AllowFailure (False by default).
"
]

SetUsage[
$FunctionName,
"
$FunctionName is a variable that contains the name of a function defined via DefineFunction.
"
]

SetUsage[
$AllowFailure,
"
$AllowFailure can be set to True in the scope of any top-level function \
(defined with DefineFunction) to return $Failed instead of the unevaluated call when a \
ThrowFailure is encountered. The unevaluated call is usually returned when input \
argument validation fails, while $Failed is returned when a runtime error occurs.
"
]

Options[DefineFunction] = {
	"Wrapper" -> List,
	"ExtraOptions" -> {},
	"ArgumentsPattern" -> Automatic,
	"ArgumentsParser" -> System`Private`Arguments
} // SortBy[ToString @* First];

DefineFunction[f_, interface_, argnum_, OptionsPattern[DefineFunction]] := 
With[
	{
		wrapper   = OptionValue["Wrapper"],
		extraOpts = OptionValue["ExtraOptions"],
		pattern = Replace[OptionValue["ArgumentsPattern"], Automatic -> ___],
		parser = Replace[OptionValue["ArgumentsParser"], 
			Except[System`Private`ArgumentsWithRules] -> System`Private`Arguments]
	},

	f[input : pattern] := Block[
		{
			a, args, opts, result,
			$FunctionName, $AllowFailure,
			GetOption
		},
		
		a = parser[f[input], argnum, wrapper, extraOpts];

		(
			$AllowFailure = False;
			$FunctionName = f;
			{args, opts} = a;
			With[
				{passedOpts = opts},
				GetOption[name_] := OptionValue[{f, extraOpts}, passedOpts, name]
			];

			result = CatchFailureAsMessage[f, interface[args, opts]];

			result /; !FailureQ[result] || $AllowFailure
		) /; Length[a] == 2
	];
]

(* Caching *)

PackageScope["Cached"]
PackageScope["$Caching"]

SetAttributes[Cached, HoldFirst]

Cached[expr_] /; $Caching :=
Scope[

	key = Hash@Replace[
		Hold[expr],
		Verbatim[Evaluate][x_] :> RuleCondition[x],
		Infinity
	];

	res = Internal`CheckImageCache[key];

	If[
		FailureQ[res],
		res = Check[expr, $Failed];
		If[
			!FailureQ[res],
			Internal`SetImageCache[key, res]
		]
	];

	res

]

Cached[expr_] := expr

(* NetModel utilities *)

PackageScope["GetNetModel"]
PackageScope["SafeNetEvaluate"]

SetUsage[SafeNetEvaluate,
"
SafeNetEvaluate[netExpr$] evaluates netExpr$, checks for the result \
and automatically handles errors.
SafeNetEvaluate[netExpr$, test$] uses test$ to validate the net result.
"
]


SetAttributes[SafeNetEvaluate, HoldFirst];

SafeNetEvaluate[netExpr_, test_: Not@*FailureQ] := Block[
	{netOutput, messages},
	Quiet@Check[
		netOutput = netExpr,
		messages = $MessageList
	];

	Which[
		!TrueQ[test[netOutput]],
			DBPrint[StringTemplate["`1`: net evaluation `2` failed"][$FunctionName, HoldForm[netExpr]]];
			$AllowFailure = True;
			ThrowFailure["nneverr"],
		Length[messages] > 0,
			DBPrint[
				StringTemplate[
					"`1`: messages have been generated by the Neural Network framework: "
				][$FunctionName],
				messages
			];
			netOutput,
		True,
			netOutput
	]
]

SetUsage[GetNetModel,
"GetNetModel[args$] evaluates NetModelInternal[$FunctionName, args$] \
and:
* uses ThrowFailure[] if a failure occurred
* uses ThrowFailure[\"nnlderr\"] if some garbage is returned
"
]

GetNetModel[args__] := With[
	{model = NetModelInternal[$FunctionName, args]},
	Which[
	(* valid network *)
		TrueQ[ValidNetQ[model]],
		model,

	(* NetModelInternal returned $Failed *)
	(* and a message should have been issued already *)
	(* therefore we fail silently *)
		FailureQ[model],
		$AllowFailure = True;
		ThrowFailure[],

	(* something very wrong happened *)
	(* fail with the generic broken net message *)
		True,
		DBPrint[
			StringTemplate["`1`: error downloading `2`"][$FunctionName, {args}]
		];
		$AllowFailure = True;
		ThrowFailure["nnlderr"]
	]
]

(* Package loader *)

PackageScope["LoadPaclet"]

SetUsage[LoadPaclet,
"
LoadPaclet['pacletname$'] loads the corresponding paclet whose loading file is Loader.m. 
LoadPaclet['pacletname$', 'filename$'] will load the file 'filename$' instead. \
If 'filename$' does not exists, Loader.m will be tried.
LoadPaclet caches its result when the loading is successful.
"
]

LoadPaclet[pacletname_] := LoadPaclet[pacletname, "Loader.m"];
LoadPaclet[pacletname_, file_] := 
Module[
	{p, result, name, loaderfile},

	name = Last[StringSplit[pacletname, "_"]];
	p = Quiet[PacletManager`Package`getPacletWithProgress[
		pacletname,
		name,
		"UpdateSites" -> False
	]]; 

	Which[
		Head[p] === PacletManager`Paclet,
			(* We have a paclet to use. *)
			loaderfile = FileNameJoin[{p["Location"], file}];
			If[!FileExistsQ[loaderfile], 
				DBPrint[StringTemplate["Error loading the paclet `1`"][pacletname]];
				$AllowFailure = True;
				ThrowFailure["interr", $FunctionName];
			];
			Check[(*Cache only when succesful*)
				result = Get[loaderfile];
				LoadPaclet[pacletname, file] = result,

				DBPrint[StringTemplate["Error loading the paclet `1`"][pacletname]];
				$AllowFailure = True;
				ThrowFailure["interr", $FunctionName];
			];
			result,

		Length[PacletManager`PacletFindRemote[pacletname]] == 0,
			(* There is no paclet of that name available. *)
			DBPrint[StringTemplate["No paclet with the name `1` is available"][pacletname]];
			$AllowFailure = True;
			ThrowFailure["interr", $FunctionName],

		PacletManager`$AllowInternet === False,
			(* There is a paclet of that name available, but we cannot download it because of user prefs. *)
			$AllowFailure = True;
			ThrowFailure["offline"],

		True,
			(* An error of some type occurred downloading and installing the paclet. *)
			DBPrint[StringTemplate["Error installing the paclet `1`"][pacletname]];
			$AllowFailure = True;
			ThrowFailure["interr", $FunctionName]
	]
];

(* Package loader - END *)

