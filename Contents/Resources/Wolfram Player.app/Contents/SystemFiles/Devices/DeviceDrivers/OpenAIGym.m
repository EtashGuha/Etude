(* ::Package:: *)

(* Mathematica Package *)

BeginPackage["DeviceFramework`Drivers`OpenAIGym`"]  
Begin["`Private`"] (* Begin Private Context *) 

(*need to load NeuralNetworks before registering the class*)
Needs["ReinforcementLearning`"];

If[!Devices`DeviceAPI`DeviceDump`knownClassQ["OpenAIGym"],
	DeviceFramework`DeviceClassRegister["OpenAIGym",
		"DeregisterOnClose" -> True,
		"ExecuteFunction" -> ReinforcementLearning`OpenAIGymLink`PackagePrivate`gymEnvironmentExecute,
		"Properties" -> {"ActionSpace" -> Automatic, "ObservationSpace" -> Automatic},
		"OpenFunction" -> ReinforcementLearning`OpenAIGymLink`PackagePrivate`gymEnvironmentCreate,
		"CloseFunction" -> ReinforcementLearning`OpenAIGymLink`PackagePrivate`gymEnvironmentClose,
		"DeviceIconFunction" -> ReinforcementLearning`OpenAIGymLink`PackagePrivate`getOpenAILogo,
		"MakeManagerHandleFunction" -> ReinforcementLearning`PackageScope`environmentCreateHandle,
		(* Allow more than one device to be opened at once *)
		"GetPropertyFunction" -> ReinforcementLearning`OpenAIGymLink`PackagePrivate`gymEnvironmentGetProperty,
		"ReadFunction" -> ReinforcementLearning`OpenAIGymLink`PackagePrivate`gymEnvironmentRead,
		"Singleton" -> False,
		"DriverVersion" -> 1.0
	]
]

End[] (* End Private Context *)

EndPackage[]