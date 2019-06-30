(* ::Package:: *)

(* Mathematica Package *)

(*this just calls DeviceClassRegister with the ArduinoLink functions, so that ArduinoLink doesn't have to be loaded manually with Needs*)

(*driver version: 1.1*)

BeginPackage["DeviceFramework`Drivers`Arduino`"]
(* Exported symbols added here with SymbolName::usage *)  

Begin["`Private`"] (* Begin Private Context *) 

(*need to load ArduinoLink before registering the class*)
Needs["ArduinoLink`"];

If[Not[Devices`DeviceAPI`DeviceDump`knownClassQ["Arduino"]],
	(*THEN*)
	(*there aren't any arduino drivers registered, so we should register it one*)
	DeviceFramework`DeviceClassRegister["Arduino",
		"Firmata",
		"ReadFunction"->ArduinoLink`Private`ArduinoReadDriver,
		"WriteFunction"->ArduinoLink`Private`ArduinoWriteDriver,
		"ConfigureFunction"->ArduinoLink`Private`ArduinoConfigureDriverWrapper,
		"PreconfigureFunction"->ArduinoLink`Private`ArduinoPreConfigureDriver,
		"DeregisterOnClose"->False,
		"ExecuteFunction"->ArduinoLink`Private`ArduinoExecuteDriver,
		"Properties"->{
			"PinConfigurations"->ArduinoLink`Private`$DeviceStates,
			"ArduinoInstallLocation"->None,
			"SerialPort"->"",
			"BoardType"->Default
			(*TODO: Implement the scheduled task handler property*)
			(*"ScheduledTaskValues"->{}*)
		},
		"GetPropertyFunction"->ArduinoLink`Private`ArduinoPropertyGetDriver,
		"SetPropertyFunction"->ArduinoLink`Private`ArduinoPropertySetDriver,
		"OpenFunction"->ArduinoLink`Private`ArduinoOpenDriver,
		"DeviceIconFunction"->ArduinoLink`Private`ArduinoCommunityLogo,
		"MakeManagerHandleFunction"->ArduinoLink`Private`ManagerHandleDriver,
		"DriverVersion"->1.1,
		"ReadTimeSeriesFunction"->DeviceRead
	]
	(*ELSE*)
	(*there already is a driver registered, so don't register anything*)
]


End[] (* End Private Context *)

EndPackage[]
