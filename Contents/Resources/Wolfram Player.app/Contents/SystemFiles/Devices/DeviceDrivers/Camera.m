(* Mathematica Package *)

(* $Id$ *)

BeginPackage["DeviceFramework`Drivers`Camera`"]
(* Exported symbols added here with SymbolName::usage *)  

Begin["`Private`"] (* Begin Private Context *) 

System`CurrentImage;

DeviceFramework`DeviceClassRegister[ "Camera",
	"FindFunction" -> IMAQ`Driver`Discover,
	"OpenFunction" -> IMAQ`Driver`Open,
	"CloseFunction" -> IMAQ`Driver`Close,
	"ConfigureFunction" -> IMAQ`Driver`Configure,
	"Properties" -> IMAQ`Driver`$Properties,
	"GetPropertyFunction" -> IMAQ`Driver`GetProperty,
	"SetPropertyFunction" -> IMAQ`Driver`SetProperty,
	"ReadFunction" -> IMAQ`Driver`ReadFunction,
	"StatusLabelFunction" -> IMAQ`Driver`StatusLabelFunction,
	(*"DeregisterOnClose" -> True,*)
	"Singleton" -> IMAQ`Driver`SingletonCriterion,
	"DeviceIconFunction" -> IMAQ`Driver`IconFunction,
	"PreconfigureFunction" -> IMAQ`Driver`Preconfigure
]

End[] (* End Private Context *)

EndPackage[]