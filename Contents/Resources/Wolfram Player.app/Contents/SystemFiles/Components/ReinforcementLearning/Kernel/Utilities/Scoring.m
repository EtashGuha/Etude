Package["ReinforcementLearning`"]

PackageImport["GeneralUtilities`"]


PackageExport["MeanEpisodicReturn"]

Clear[MeanEpisodicReturn];

Options[MeanEpisodicReturn] = {
	"MaxSteps" -> 100
};

MeanEpisodicReturn[env_, policy_, n_:1, OptionsPattern[]] := Block[
	{maxSteps, observedState, action, actionResult, episodeReward, reward},
	maxSteps = OptionValue["MaxSteps"];
	Mean @ Table[
		observedState = DeviceExecute[env, "Reset"]["ObservedState"];
		episodeReward = 0.0;
		Do[
			action = policy[observedState];
			actionResult = DeviceExecute[env, "Step", action];
			{observedState, reward, ended} = Lookup[
				actionResult,
				{"ObservedState", "Reward", "Ended"},
				Panic["BadEnvironment"]
			];
			episodeReward += reward;
			If[ended, Break[]];
		,
			maxSteps
		];
		episodeReward
	,
		n
	]
]
