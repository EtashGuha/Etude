Package["ReinforcementLearning`"]


discountedSum[rewards_List, gamma_] := Dot[rewards, gamma ^ Range[0, Length[rewards] - 1]];

applyDiscount[rewards_List, gamma_] := 
	Table[discountedSum[Drop[rewards, i-1], gamma], {i, Length@rewards}]
	
applyDiscount[rewards_, None] := rewards;


PackageImport["GeneralUtilities`"]
PackageImport["PacletManager`"]

PackageExport["RLCollectEpisode"]

Options[RLCollectEpisode] = {
	"MaxSteps" -> 100,
	"DiscountFactor" -> 0.99,
	RandomSeeding -> None
};

RLCollectEpisode[env_, policy_, OptionsPattern[]] := Module[
	{
		observedStates, rewards, actions, values, 
		actionResult, observedState, reward, action, ended,
		discountedRewards
	},

	observedStates = Internal`Bag[];
	rewards = Internal`Bag[];
	actions = Internal`Bag[];
	values = Internal`Bag[];
	
withRandomSeeding[OptionValue[RandomSeeding],

	observedState = Lookup[
		DeviceExecute[env, "Reset"],
		"ObservedState", Panic["BadEnvironment"]
	];

	(* action loop *)
	Do[
		Internal`StuffBag[observedStates, observedState];
		action = policy[observedState];
		actionResult = DeviceExecute[env, "Step", action];
		{observedState, reward, ended} = Lookup[
			actionResult,
			{"ObservedState", "Reward", "Ended"},
			Panic["BadEnvironment"]
		];
		Internal`StuffBag[rewards, reward];
		Internal`StuffBag[actions, action];
		If[ended, Break[]];
	,
		OptionValue["MaxSteps"]
	];
];

	rewards = Internal`BagPart[rewards, All];
	actions = Internal`BagPart[actions, All];
	observedStates = Internal`BagPart[observedStates, All];

	discountedRewards = applyDiscount[rewards, OptionValue["DiscountFactor"]];

	Association[
		"ObservedState" -> observedStates,
		"Action" -> actions,
		"Reward" -> rewards,
		"DiscountedReward" -> discountedRewards
	]
];

PackageExport["RLCollectMultipleEpisodes"]

Options[RLCollectMultipleEpisodes] = {
	"MaxStepsPerEpisode" -> 100,
	"DiscountFactor" -> 0.99,
	RandomSeeding -> Automatic
}

RLCollectMultipleEpisodes[env_, policy_, numberSteps_Integer, OptionsPattern[]] := Module[
	{actions, observedStates, rewards, discountedRewards, opts, currentSteps, result},

	actions = Internal`Bag[];
	observedStates = Internal`Bag[];
	rewards = Internal`Bag[];
	discountedRewards = Internal`Bag[];

	maxSteps = OptionValue["MaxStepsPerEpisode"];
	discountFactor = OptionValue["DiscountFactor"];
	randomSeeding = OptionValue["RandomSeeding"];

	opts = Sequence["DiscountFactor" -> discountFactor];
	
	currentSteps = 0; episode = 0;

withRandomSeeding[OptionValue[RandomSeeding],

	While[
		currentSteps < numberSteps
	,
		result = RLCollectEpisode[
			env, policy, "MaxSteps" -> Min[maxSteps, numberSteps - currentSteps], 
			RandomSeeding -> RandomInteger[2^31], opts
		];

		Internal`StuffBag[actions, result["Action"], 1];
		Internal`StuffBag[observedStates, result["ObservedState"], 1];
		Internal`StuffBag[rewards, result["Reward"], 1];
		Internal`StuffBag[discountedRewards, result["DiscountedReward"], 1];

		currentSteps += Length @ First @ result;
	];
];

	Association[
		"Action" -> Internal`BagPart[actions, All],
		"ObservedState" -> Internal`BagPart[observedStates, All],
		"Reward" -> Internal`BagPart[rewards, All],
		"DiscountedReward" -> Internal`BagPart[discountedRewards, All]
	]
]