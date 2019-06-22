(* $Id$ *)

(* Implements top-level and native properties. *)


BeginPackage["DeviceAPI`Drivers`Demos`PropertiesDemo`"];

Begin["`Private`"];

nproperties[_]["N1"] = "n1";
nproperties[_]["N2"] = "n2";
nproperties[_]["X"] = "nativeX";

getProp[devHandle_,prop_] := nproperties[devHandle][prop]
setProp[devHandle_,prop_,rhs_] := nproperties[devHandle][prop] = rhs
		   
(*-----------------------------------------------------------------*)  

DeviceFramework`DeviceClassRegister["PropertiesDemo",
	"Properties" -> {"P1" -> "p1", "P2" -> "p2", "X" -> "x"},
	"NativeProperties" -> {"N1", "N2", "X"},
	"GetNativePropertyFunction" -> getProp,
	"SetNativePropertyFunction" -> setProp,
	"CloseFunction" -> (Quiet[
		nproperties[ #[[2]] ]["N1"] =.;
		nproperties[ #[[2]] ]["N2"] =.;
		nproperties[ #[[2]] ]["X"] =.;
	]&),
	"DriverVersion" -> 0.001
];

End[];

EndPackage[];
