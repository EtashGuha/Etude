(* Wolfram Language Package *)

(* Exported symbols added here with SymbolName::usage *)

System`ExternalFunction;

ExternalFunction::nosup = "ExternalFunction not yet supported for system `1`";
ExternalFunction::nofunction = "ExternalFunction does not support function `1`";
ExternalFunction::invlDef = "Error defining function.";
ExternalFunction::cnvt = "Conversion of type `1` not supported.";

ExternalValue::nosup = "ExternalValue not yet supported for system `1`";

Begin["`Private`"]

Unprotect[ExternalFunction, ExternalValue];

System`ExternalFunction/:MakeBoxes[System`ExternalFunction[assoc_Association /; AtomQ[Unevaluated @ assoc] && AssociationQ[assoc]],StandardForm|TraditionalForm]:=
	BoxForm`ArrangeSummaryBox[
		(*first argument is the head to use*)
		System`ExternalFunction,
		(*second argument is the expression*)
		System`ExternalFunction[assoc],
		(*third argument is the icon to use*)
		None,
		(*the next argument is the always visisble properties*)
		MapAt[
			(*this will make sure that we span on the top row*)
			Append[#,SpanFromLeft]&,
			(*this bit of code will make sure that we never include the Session or the Source keys*)
			(*and that we take up to of the first entries in the association*)
			(*and make them into a grid matrix*)
			Partition[
				KeyValueMap[
					Function[{key,val},BoxForm`SummaryItem[{key<>": ",val}]],
					KeyTake[assoc, {"Name", "Arguments", "System", "BuiltIn"}]
				],
				UpTo[2]
			],
			-1
		],
		(*the next argument is the optional items that come down when the plus button is pressed*)
		KeyValueMap[
			Function[{key,val},BoxForm`SummaryItem[{key<>": ",val}]],
			orderFunctionObject@If[KeyExistsQ["Source"]@#,
				MapAt[
					makeScrollableCodePanel,
					#,
					Key["Source"]
				],
				#
			]&@assoc
		],
		(*lastly,the display form we want to display this with*)
		StandardForm,
		(*we use complete replacement to completely ignore the first set of displayed values*)
		(*with the second one when the button is clicked*)
		"CompleteReplacement"->True
	];


(*this enforces a semantically pleasing ordering*)
(*of the keys in the summary box for ExternalObject*)
orderFunctionObject[assoc_] := Join[
	KeyTake[
		assoc, 
		{"Name", "Arguments", "System", "BuiltIn"}
	], 
	KeyDrop[
		assoc,
		{"Name","Arguments", "System", "BuiltIn", "Session"}
	], 
	KeyTake[assoc, "Session"]
]

(* Save function session, so we do not have to restart it each function call. *)
defaultExternalSessions = <||>;

(* Helper function to check for a valid saved session. *)
getExternalSession[system_String] := Block[{session, uuid},
	session = Lookup[defaultExternalSessions, system];
	If[MatchQ[session, _ExternalSessionObject],
		uuid = First[session];
		If[KeyExistsQ[uuid]@$Links && TrueQ[$Links[uuid,"Active"]],
			Return[session]
		]
	];
	session = ExternalEvaluate`FE`$CellSessions[system];
	If[MatchQ[session, _ExternalSessionObject],
		uuid = First[session];
		If[KeyExistsQ[uuid]@$Links && TrueQ[$Links[uuid,"Active"]],
			defaultExternalSessions[system] = session;
			Return[session]
		]
	];
	session = StartExternalSession[system];
	If[MatchQ[session, _ExternalSessionObject],
		defaultExternalSessions[system] = session;
		If[Not[ValueQ[ExternalEvaluate`FE`$CellSessions]],
			ExternalEvaluate`FE`$CellSessions = <||>
		];
		ExternalEvaluate`FE`$CellSessions[system] = session
	];
	session
]
getExternalSession[assoc_Association] := 
	Replace[
		assoc["Session"], {
			_Missing|None|Inherited|Automatic :> getExternalSession[assoc["System"]],
			uuid_String :> ExternalSessionObject[uuid]
		}
	]
getExternalSession[_] := $Failed

(* Perform evaluations using ExternalFunction. *)
$DefaultSession = Inherited;

ExternalFunction[assoc:Except[KeyValuePattern["Session" -> _], _Association?AssociationQ]] := 
	ExternalFunction[Append[assoc, "Session" -> $DefaultSession]]
ExternalFunction[assoc_Association?AssociationQ][args___] :=
	Module[{
		session  = getExternalSession[assoc], 
		funcName = Lookup[assoc, "Name"], 
		module   = Lookup[assoc, "Module"]
		},
		If[
			! MatchQ[session, _ExternalSessionObject], 
			Return[$Failed]
		];
		If[
			And[
				StringQ[module],
				! MemberQ[{"__builtin__", "builtins"}, module]
			],
			funcName = StringJoin[module, ".", funcName];
		];
		ExternalEvaluate[session, <|"Command" -> funcName, "Arguments" -> {args}|>]
	];

ExternalFunction[system_String, funcName_String] :=
	Replace[
		getExternalSession[system],
		session_ExternalSessionObject :> ExternalFunction[session, funcName]
	]

ExternalFunction[session_ExternalSessionObject, funcName_String] :=
	With[{system = session["System"]},
		Replace[
			If[
				defFunctionIdentifier[system][funcName],
				handleDefinedFunction[system][session, funcName],
				ExternalEvaluate[session, funcName]
			], {
				ExternalFunction[assoc_Association?AssociationQ] :>
					ExternalFunction @ Append[assoc, "Session" -> session],
				_ :> 
					Failure["ExternalFunction", <|"Message" -> "Invalid function name."|>]
			}
		]
	]


(* Get an external value. *)
ExternalValue[system_String, symbol_String] := 
	Replace[
		getExternalSession[system],
		session_ExternalSessionObject :> ExternalEvaluate[session, symbol]
	]

ExternalValue[session_ExternalSessionObject, symbol_String] := 
	ExternalEvaluate[session, symbol]

Protect[ExternalFunction, ExternalValue];

End[] (*End of `Private` context*)
