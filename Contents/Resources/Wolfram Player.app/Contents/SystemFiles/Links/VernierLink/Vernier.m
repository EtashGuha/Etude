(* Wolfram Language package *)
BeginPackage["DeviceFramework`Drivers`Vernier`"]
(* Exported symbols added here with SymbolName::usage *)  

Begin["`Private`"] (* Begin Private Context *) 

(*need to load VernierLink before registering the class*)
Needs["VernierLink`"];

If[Not[Devices`DeviceAPI`DeviceDump`knownClassQ["Vernier"]],
	(*THEN*)
	(*there aren't any vernier drivers registered, so we should register it one*)
	DeviceFramework`DeviceClassRegister["Vernier",
		"FindFunction" -> (List /@ VernierLink`findDevice[] &),
		"OpenFunction" -> VernierLink`deviceOpen,
		"CloseFunction" -> VernierLink`deviceClose,
		"ReadFunction" -> VernierLink`deviceRead,
		"ReadBufferFunction" -> VernierLink`deviceReadBuffer,
		"Properties" -> {
			"SensorID" -> Null,
			"SensorName" -> Null,
			"BufferLength"->1200, (*a constant - never changes*)
			"SensorDescription" -> Null,
			"MeasurementInterval" -> Null,
			"MinMeasurementInterval" -> Null,
			"MaxMeasurementInterval" -> Null,
			"ProbeType" -> Null,
			"EquationType" -> Null
		},
		"GetPropertyFunction" -> VernierLink`deviceProperties,
		"ConfigureFunction" -> VernierLink`deviceConfigure,
		"PreconfigureFunction"-> VernierLink`preConfigure,
		"DeviceIconFunction" -> VernierLink`iconf,
		"DeregisterOnClose" -> True
	]
	(*ELSE*)
	(*there already is a driver registered, so don't need to register anything*)
]


End[] (* End Private Context *)

EndPackage[]
