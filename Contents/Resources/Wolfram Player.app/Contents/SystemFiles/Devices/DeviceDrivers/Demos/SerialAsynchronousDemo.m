(* Mathematica Package *)

BeginPackage["DeviceAPI`Drivers`Demos`SerialAsynchronousDemo`Dump`"]
(* Exported symbols added here with SymbolName::usage *)  

Needs["SerialLink`"];

Begin["`Private`"] (* Begin Private Context *) 


(*** re-write!!! *)

async[{_, port_SerialPort}, "Read", fun_]:= RunSerialPortReadAsynchronousTask[port,"Byte","ReadHandler"->fun];

async[{_, port_SerialPort}, "Read","Format"->"String", fun_]:= RunSerialPortReadAsynchronousTask[port,"String","ReadHandler"->fun];

async[___] := $Failed 


(*-----------------------------------------------------------------*) 

DeviceFramework`DeviceClassRegister[ "Serial", "SerialAsynchronousDemo",
	"ExecuteAsynchronousFunction" -> async
]

End[] (* End Private Context *)

EndPackage[]