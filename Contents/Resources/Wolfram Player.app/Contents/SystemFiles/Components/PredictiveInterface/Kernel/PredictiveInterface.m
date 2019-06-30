
(* Safeguard against package reloading *)

Block[{PredictiveInterfaceLoader`Private`list},
	PredictiveInterfaceLoader`Private`list = {
		"Predictions`MakePredictions",
	
		"PredictiveInterface`NotebookPredictions",
		"PredictiveInterface`ShowPredictions",
		"PredictiveInterface`PredictionControls",
		"PredictiveInterface`PlaceholderMenu",
		"PredictiveInterface`DoPredictionAction",
		"PredictiveInterface`DoPredictionsChangeSemanticType",
		"PredictiveInterface`DoPredictionsRollup",
		"PredictiveInterface`DoPredictionsPrune"
	};
	Unprotect @@ PredictiveInterfaceLoader`Private`list;
	ClearAll @@ PredictiveInterfaceLoader`Private`list;
]


(* Predictions.m takes care of loading PredictiveRulesCompiled.m and PredictiveAlphaUtilities.mx as needed *)

Get["PredictiveInterface`Predictions`"];

Get["PredictiveInterface`PredictiveInterfaceCode`"];



(* Message init *)

Off[PredictiveInterfaceDump`GetSemanticTypesAndPredictions::badpred]

Off[PredictiveInterface`DoPredictionAction::badact]



(* Paclet introspection *)

PredictiveInterface`$PredictiveInterfacePaclet = First[ PacletManager`PacletFind["PredictiveInterface"] ];

PredictiveInterface`$PredictiveInterfaceInformation := PacletManager`PacletInformation[ PredictiveInterface`$PredictiveInterfacePaclet ];

PredictiveInterface`$PredictiveInterfaceVersion := "Version" /. PredictiveInterface`$PredictiveInterfaceInformation;

PredictiveInterface`$PredictiveInterfaceLocation := "Location" /. PredictiveInterface`$PredictiveInterfaceInformation;

