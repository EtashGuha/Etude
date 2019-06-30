(* $Id$ *)

BeginPackage["DeviceAPI`Drivers`Demos`RandomSignalDemo`"];

PrependTo[$ContextPath, "DeviceFramework`"];

Begin["`Private`"];

open[_] := "dummy"

open[_,seed_] := (
	SeedRandom[seed];
	CreateUUID[]
)

read[_,"Integer"] := RandomInteger[]
read[_,"Complex"] := RandomComplex[]
read[_,type_:"Real"] := RandomReal[]
read[_,types__] := read[Null,#]&/@{types}

readBuffer[{_,h_},_] := RandomReal[{-1,1},
	 DeviceObjectFromHandle[h]["BufferLength"]
]
readBuffer[_,n_Integer,_] := RandomReal[{-1,1},n]
readBuffer[___] := $Failed

configure[{_,h_},rules___?OptionQ] := Module[
{
	dev = DeviceObjectFromHandle[h],
	props, vals
},
	{props, vals} = Transpose[List @@@ Flatten[{rules}]];
	dev[props] = vals
]

configure[_,bad__] := (
	Message[DeviceConfigure::nonopt, {bad}, 1, 
		HoldForm[DeviceConfigure]["RandomSignalDemo",{bad}]
	];
	$Failed
)

set[dev_, p:"BufferLength", v_Integer?Positive] := DeviceSetProperty[dev,p,v]

set[dev_, p:"BufferLength", v_] :=
	Message[DeviceObject::pstvnt,"\"BufferLength\"", HoldForm[DeviceObject[]]["\"property\""]]

set[args__] := DeviceSetProperty[args]


DeviceClassRegister["RandomSignalDemo",
	"OpenFunction" -> open,
	"Properties" -> {"BufferLength" -> 5},
	"SetPropertyFunction" -> set,
	"ReadFunction" -> read,
	"ReadBufferFunction" -> readBuffer,
	"ConfigureFunction" -> configure,
	"FindFunction" -> ({{}}&),
	"DeviceIconFunction" -> (ListLinePlot[RandomReal[{0, 1}, 40], 
		Frame -> True, FrameTicks -> None, Axes -> False, 
		AspectRatio -> 1
	]&),
	"DeregisterOnClose" -> True,
	"DriverVersion" -> 0.001
];

End[];
EndPackage[];
