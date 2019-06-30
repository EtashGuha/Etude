Package["ReinforcementLearning`"]

PackageImport["GeneralUtilities`"]

(*----------------------------------------------------------------------------*)
(* copy of https://github.com/openai/baselines/blob/master/baselines/ppo2/policies.py *)
(* Note: ensure inputs are scaled to range (0, 1) *)

(* TODO: add recurrent options *)

PackageExport["OpenAIBaselineConvPolicy"]
Options[OpenAIBaselineConvPolicy] = {
	"Input" -> Automatic
}

OpenAIBaselineConvPolicy[actions_List, OptionsPattern[]] := Module[
	{net},
	in = OptionValue["Input"];
	net = NetChain[{
		ConvolutionLayer[32, 8, "Stride" -> 4], Ramp,
		ConvolutionLayer[64, 8, "Stride" -> 2], Ramp,
		ConvolutionLayer[32, 8, "Stride" -> 1], Ramp,
		LinearLayer[512], Ramp
	}, "Input" -> in];
	NetGraph[<|
		"policy" -> net, 
		"critic" -> LinearLayer[1, "Output" -> "Scalar"], 
		"actor" -> LinearLayer[]
		|>,
		{"policy" -> "critic" -> NetPort["Value"],
		 "policy" -> "actor" -> NetPort["Action"]
		 }, 
		 "Action" -> NetDecoder[{"Class", actions}]
	]
]

(* TODO: add critic *)
PackageExport["OpenAIBaselineMLPPolicy"]
Options[OpenAIBaselineMLPPolicy] = {
	"Input" -> Automatic
}

OpenAIBaselineMLPPolicy[actions_List, OptionsPattern[]] := Module[
	{actor, critic},
	in = OptionValue["Input"];
	actor = NetChain[{
		LinearLayer[64], Tanh,
		LinearLayer[64], Tanh,
		LinearLayer[]
	}, "Input" -> in];
	critic = NetChain[{
		LinearLayer[64], Tanh,
		LinearLayer[64], Tanh,
		LinearLayer[1, "Output" -> "Scalar"]
	}];
	NetGraph[<|"actor" -> actor, "critic" -> critic|>, 
	{"actor" -> NetPort["Action"], "critic" -> NetPort["Value"]},
	"Action" -> NetDecoder[{"Class", actions}]
	]
]