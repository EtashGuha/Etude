(* Wolfram Language Package *)

(Unprotect[#]; Clear[#])& /@ {}

BeginPackage["NeuralNetResource`"]

Begin["`Private`"] (* Begin Private Context *) 

checkPaclet[name_]:=Block[{paclets},
	paclets=PacletFind[name];
	If[Length[paclets]>0,
		True
		,
		paclets=PacletFindRemote[name];
		If[Length[paclets]>0,
			True
			,
			False
		]
	]
]

If[TrueQ[checkPaclet["ResourceSystemClient"]],
	Needs["ResourceSystemClient`"]
	,
	Message[ResourceData::norsys, "NeuralNet"]
]

If[TrueQ[checkPaclet["DataResource"]],
	Needs["DataResource`"]
	,
	Message[ResourceData::nopacl, "NeuralNet","DataResource"]
]

End[];

EndPackage[]

SetAttributes[{},
   {ReadProtected, Protected}
];