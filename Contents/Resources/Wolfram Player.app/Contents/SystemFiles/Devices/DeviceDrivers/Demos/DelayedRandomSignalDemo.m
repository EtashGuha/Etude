(* $Id$ *)

(* Simulates long measurements. Uses inheritance. *)

BeginPackage["DeviceAPI`Drivers`Demos`DelayedRandomSignalDemo`Dump`"];
Begin["`Private`"];

$delay = .1;

open[_] := open[Null,0]
open[_,delay_] := ($delay = delay; CreateUUID[])

DeviceFramework`DeviceClassRegister["DelayedRandomSignalDemo", "RandomSignalDemo",
	"ReadFunction" :> (
		(
			Pause[$delay];
			(* the parent's function *)
			DeviceFramework`DeviceDriverOption["RandomSignalDemo", "ReadFunction"][##]
		)&
	),
	"OpenFunction" -> open,
	"Singleton" -> False,
	"FindFunction" -> ({}&),
	"DriverVersion" -> 0.001
];

End[];
EndPackage[];
