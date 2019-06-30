Package["ReinforcementLearning`"]

PackageImport["GeneralUtilities`"]
PackageImport["PacletManager`"]

(*----------------------------------------------------------------------------*)
If[!AssociationQ[$GymEnvironments], $GymEnvironments = <||>];

$OpenAILogo := $OpenAILogo = Import[PacletResource["ReinforcementLearning", "OpenAIGymLogo"]];
getOpenAILogo[___] := $OpenAILogo

(*----------------------------------------------------------------------------*)
DeviceFramework`Devices`OpenAIGym::pythonerr := "Error from Python environment: ``"

safeEE[session_, command_] := Scope[
	(* Block: from docs of ExternalEvaluate, 
		"Individual write operations to standard output are immediately printed 
		to the notebook or terminal." 
		Some operations keep printing warning messages
	*)
	out = Block[{Print}, ExternalEvaluate[session, command]];
	If[FailureQ @ out, 
		ThrowFailure[DeviceFramework`Devices`OpenAIGym::pythonerr, out["Message"]]
	];
	out
]

(*----------------------------------------------------------------------------*)
DeviceFramework`Devices`OpenAIGym::invenv := "No environment `` exists in the installed version of OpenAIGym."
DeviceFramework`Devices`OpenAIGym::devopenexe := "The \"Executable\" `` does not exist."
DeviceFramework`Devices`OpenAIGym::devopeninv := "The argument to DeviceOpen `` is invalid."
DeviceFramework`Devices`OpenAIGym::devopenenvfail := "Cannot create environment ``. Ensure that it is correctly installed."
DeviceFramework`Devices`OpenAIGym::devopenextses := 
"Cannot execute StartExternalSession[\"Python\"]. The documentation \
workflow/ConfigurePythonForExternalEvaluate might be useful."


gymEnvironmentCreate[uuid_, name_] :=  gymEnvironmentCreate[uuid, name, None]

gymEnvironmentCreate[uuid_, name_, arg_] := CatchFailureAsMessage @ Module[
	{sessInfo, session, comm, observedState, err},

	sessInfo = <|"System" -> "Python"|>;
	Which[
		MatchQ[arg, "Executable" -> _String|_File],
			sessInfo["Executable"] = Last[arg];
			If[!FileExistsQ[sessInfo["Executable"]], 
				ThrowFailure[DeviceFramework`Devices`OpenAIGym::devopenexe, sessInfo["Executable"]]
			];
		,
		arg =!= None,
			ThrowFailure[DeviceFramework`Devices`OpenAIGym::devopeninv, arg]
	];

	session = StartExternalSession[sessInfo];
	If[FailureQ[session], 
		ThrowFailure[DeviceFramework`Devices`OpenAIGym::devopenextses]
	];

	safeEE[session, "import gym"];
	safeEE[session, "import gym.spaces"];

	comm = "env = gym.make(\"" <> name <> "\")";
	err = CatchFailure @ safeEE[session, comm];

	If[FailureQ[err],
		(* get all environments *)
		envs = safeEE[session, "[env_spec.id for env_spec in gym.envs.registry.all()]"];
		If[!MemberQ[envs, name],
			ThrowFailure[DeviceFramework`Devices`OpenAIGym::invenv, name],
			ThrowFailure[DeviceFramework`Devices`OpenAIGym::devopenenvfail, name]
		];
		
	];

	(* we reset all gym environments to have a state to read *)
	observedState = safeEE[session, "env.reset()"];

	$GymEnvironments[uuid] = 
		<|"State" -> session, "ObservedState" -> observedState, "Ended" -> False|>;
]

DeviceFramework`Devices`OpenAIGym::invopen := "DeviceOpen requires two arguments for OpenAIGym."
gymEnvironmentCreate[x_] := (Message[DeviceFramework`Devices`OpenAIGym::invopen]; $Failed)

(*----------------------------------------------------------------------------*)
gymEnvironmentClose[{id_, _}] := CatchFailure @ Scope[
	env = Lookup[$GymEnvironments, id];
	(* delete session *)
	safeEE[env["State"], "env.close()"];
	DeleteObject[env["State"]];
	(* remove key *)
	KeyDropFrom[$GymEnvironments, id];
]

(*----------------------------------------------------------------------------*)
gymEnvironmentExecute[{id_, _}, "Reset"] := CatchFailure @ Scope[
	env = Lookup[$GymEnvironments, id];
	<|"ObservedState" -> safeEE[env["State"], "env.reset()"]|>
]

gymEnvironmentExecute[{id_, _}, "Step", arg_] := CatchFailure @ Scope[
	env = Lookup[$GymEnvironments, id];
(* 	If[render, safeEE[state, "env.render()"]]; *)

	stepString = "env.step(" <> ToString[arg] <> ")";
	out = safeEE[env["State"], stepString];
	<|"ObservedState" -> First[out], "Reward" -> out[[2]], "Ended" -> out[[3]]|>
]

gymEnvironmentExecute[{id_, _}, "RandomAction"] := CatchFailure @ Scope[
	env = Lookup[$GymEnvironments, id];
	safeEE[env["State"], "env.action_space.sample()"]
]

gymEnvironmentExecute[{id_, _}, "Render"] := CatchFailure @ Scope[
	env = Lookup[$GymEnvironments, id];
	safeEE[env["State"], "env.render()"]
]

(*----------------------------------------------------------------------------*)
gymEnvironmentGetProperty[dev_, "ActionSpace"] := CatchFailure @ Scope[
	(* out = DeviceFramework`DeviceManagerHandle[dev];
	state = Lookup[$GymEnvironments, id];
	safeEE[state, "env.action_space.sample()"] *)
	Automatic
]

gymEnvironmentGetProperty[dev_, "ObservationSpace"] := CatchFailure @ Scope[
	(* out = DeviceFramework`DeviceManagerHandle[dev];
	state = Lookup[$GymEnvironments, id];
	safeEE[state, "env.action_space.sample()"] *)
	Automatic
]

(*----------------------------------------------------------------------------*)
gymEnvironmentRead[{id_, _}] := CatchFailure @ Scope[
	env = Lookup[$GymEnvironments, id];
	KeyTake[env, {"ObservedState", "Ended"}]
]

(*----------------------------------------------------------------------------*)
(* Space Conversions *)


(*     def _get_space_properties(self, space):
        info = {}
        info['name'] = space.__class__.__name__
        if info['name'] == 'Discrete':
            info['n'] = space.n
        elif info['name'] == 'Box':
            info['shape'] = space.shape
            # It's not JSON compliant to have Infinity, -Infinity, NaN.
            # Many newer JSON parsers allow it, but many don't. Notably python json
            # module can read and write such floats. So we only here fix "export version",
            # also make it flat.
            info['low']  = [(x if x != -np.inf else -1e100) for x in np.array(space.low ).flatten()]
            info['high'] = [(x if x != +np.inf else +1e100) for x in np.array(space.high).flatten()]
        elif info['name'] == 'HighLow':
            info['num_rows'] = space.num_rows
            info['matrix'] = [((float(x) if x != -np.inf else -1e100) if x != +np.inf else +1e100) for x in np.array(space.matrix).flatten()]
        return info *)

(*----------------------------------------------------------------------------*)
(* RLEnvironmentRender[RLEnvironment["Gym", id_, _]] :=  
	CatchFailure @ safeEE[getState[id], "env.render()"] *)