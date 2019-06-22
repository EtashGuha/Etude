
(* ::Package:: *)

(* Mathematica Init File *)

Needs["PacletManager`"]

Module[{paclet, dir},
	Quiet[
		paclet = PacletFind["CUDAResources"];
		If[paclet =!= {},
			paclet = First[paclet];
			dir = "Location" /. PacletInformation[paclet];
			If[TrueQ[StringQ[dir]],
				PrependTo[$Path, dir]
			]
		]
	];
	
	Get["OpenCLLink`OpenCLLink`"]
]
