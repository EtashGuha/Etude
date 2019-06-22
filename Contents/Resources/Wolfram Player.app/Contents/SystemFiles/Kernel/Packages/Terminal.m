(* Graphics output for dumb terminals *)

(* These variables can be reset during a session to alter subsequently
   created graphics windows: *)

Begin["System`Private`"]

Unprotect[ $DisplayFunction]

Clear[ $DisplayFunction]

$Display = {}

Unprotect[ $SoundDisplayFunction]

Clear[ $SoundDisplayFunction]

$SoundDisplay = {}

$SoundDisplayFunction = Identity

If[Developer`InstallFrontEnd[]=!=$Failed,
	If[!($BatchOutput || $Linked || $ParentLink =!= Null),
		Print[" -- Terminal graphics initialized -- "]
	];

	$DisplayFunction =  Module[{str},
							str = Quiet@ExportString[#, "TTY"];
							If[StringQ[str],
								WriteString[$Output,str]
							];
							#
						]&,
	Message[Graphics::nogr, "Terminal"];
	$DisplayFunction = Identity
]

End[]

Null

