(* $Id$ *)

(* error handling; default arguments *)

BeginPackage["DeviceAPI`Drivers`Demos`DefaultShellCommandDemo`Dump`"];

Begin["`Private`"];

properties[_][_] = {};

open[_,str:(_String):"date"] := Module[
{
	h = RandomInteger[10^10]
},
	properties[h]["cmnd"] = str;
	h
]

open[_,e_] := (
	Message[DeviceOpen::blnulst,e,2];
	$Failed
)

open[_,args__] := $Failed


read[{_,h_}] := read[Null, properties[h]["cmnd"] ]
read[_,c_String] := ReadList["!"<>c, Record]
read[_,c_,p_:{}] := read[Null, StringJoin[ Riffle[ToString/@Flatten[{c,p}]," "] ] ]

DeviceFramework`DeviceClassRegister["DefaultShellCommandDemo",
	"ReadFunction" -> read,
	"WriteFunction" -> read,
	"OpenFunction" -> open,
	"ExecuteFunction" -> read,
	"DriverVersion" -> 0.001
];

End[];

EndPackage[];
