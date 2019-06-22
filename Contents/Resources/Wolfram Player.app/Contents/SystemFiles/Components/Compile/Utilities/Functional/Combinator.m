
BeginPackage["Compile`Utilities`Functional`Combinator`"]

Begin["`Private`"]

Needs["Compile`Utilities`Functional`Base`"]


Functional[{"Combinator", "Apply"}] =
	Function[{f},
		Function[{x},
			f[x]
		]
	]


Functional[{"Combinator", "Compose"}] =
	Function[{f},
		Function[{g},
			Function[{x},
				f[g[x]]
			]
		]
	]

Functional[{"Combinator", "Constant"}] =
	Function[{a},
		Function[{b},
			a
		]
	]
	

Functional[{"Combinator", "Fix"}] =
	Function[{f},
		With[{
			g = Function[{h},
				Function[{x},
					f[h[h]][x]
				]
			]
		},
			g[g]
		]
	]
	
Functional[{"Combinator", "Flip"}] =
	Function[{f},
		Function[{a},
			Function[{b},
				f[b][a]
			]
		]
	]
	
Functional[{"Combinator", "Identity"}] =
	Function[{f},
		f
	]
	
Functional[{"Combinator", "Psi"}] =
	Function[{f},
		Function[{g},
			Function[{x},
				Function[{y},
					f[g[x]][g[y]]
				]
			]
			
		]
	]
	
Functional[{"Combinator", "Substitution"}] =
	Function[{f},
		Function[{g},
			Function[{x},
				f[x][g[x]]
			]
		]
	]
	
Functional[{"Combinator", "Thrush"}] =
	Function[{x},
		Function[{f},
			f[x]
		]
	]
	
	
End[]

EndPackage[] 

