(* $Id$ *)

(*
	DeviceOpen returns a new device for every distinct first argument 
	(which also has a default value).
*)

BeginPackage["DeviceAPI`Drivers`Demos`SingletonDemo`"];

PrependTo[$ContextPath, "DeviceFramework`"];

$OpenCounter = 0;

Begin["`Private`"];

$default = "foo";

ClearAll[open1, single1, get1];

open1[_] := open1[Null, $default ]
open1[_, f_, ___] := ($OpenCounter++; f)

single1[dev_, args_] := SameQ @@ get1 /@ {DeviceOpenArguments[dev], args}

get1[{}] := $default
get1[{f_, ___}] := f

DeviceClassRegister["SingletonDemo", 
	"Singleton" -> single1,
	"OpenFunction" -> open1,
	"DriverVersion" -> 0.001
]

End[];
EndPackage[];
