(* $Id$ *)

(* A discoverable singleton class that executes a shell command. *)

BeginPackage["DeviceAPI`Drivers`Demos`ShellCommandDemo`Dump`"];

DeviceFramework`DeviceClassRegister["ShellCommandDemo",
	"ReadFunction" -> (ReadList["!"<>#2, Record]&),
	"ExecuteAsynchronousFunction" -> (Missing["NotAvailable"]&),
	"Singleton" -> True,
	"DriverVersion" -> 0.001
];

EndPackage[];
