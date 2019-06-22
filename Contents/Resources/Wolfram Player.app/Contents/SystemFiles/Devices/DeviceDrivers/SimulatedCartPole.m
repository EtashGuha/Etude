(* ::Package:: *)

(* Mathematica Package *)

BeginPackage["DeviceFramework`Drivers`SimulatedCartPole`"]  
Begin["`Private`"] (* Begin Private Context *) 

(*need to load NeuralNetworks before registering the class*)
Needs["ReinforcementLearning`"];

If[!Devices`DeviceAPI`DeviceDump`knownClassQ["SimulatedCartPole"],
	DeviceFramework`DeviceClassRegister["SimulatedCartPole",
		"DeregisterOnClose" -> True,
		"ExecuteFunction" -> ReinforcementLearning`SimulatedCartPole`PackagePrivate`deviceEnvExecute,
		"Properties" -> {"ActionSpace" -> {Left, Right}},
		"OpenFunction" -> ReinforcementLearning`SimulatedCartPole`PackagePrivate`deviceEnvCreate,
		"CloseFunction" -> ReinforcementLearning`Environments`SimulatedCartPole`PackagePrivate`deviceEnvClose,
		(* "DeviceIconFunction" -> ReinforcementLearning`Environments`SimulatedCartPole`PackagePrivate`getOpenAILogo, *)
		"MakeManagerHandleFunction" -> ReinforcementLearning`PackageScope`environmentCreateHandle,
		(* Allow more than one device to be opened at once *)
		"ReadFunction" -> ReinforcementLearning`SimulatedCartPole`PackagePrivate`deviceEnvRead,
		"Singleton" -> False,
		"DriverVersion" -> 1.0
	]
]

End[] (* End Private Context *)

EndPackage[]