BeginPackage["Compile`Core`Optimization`CoerceCast`"]
CoerceCastPass;

Begin["`Private`"] 

Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Instruction`CopyInstruction`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]




unresolve[TypeSpecifier[t_]] := TypeSpecifier[t]
unresolve[Type[t_]] := TypeSpecifier[t]
unresolve[t_?TypeObjectQ] := t["unresolve"]

booleanTypeQ[t_] := MatchQ[unresolve[t], TypeSpecifier["Boolean"]]
integerTypeQ[t_] := MatchQ[unresolve[t], 
	TypeSpecifier["Integer"] | 
	TypeSpecifier["Integer64"] | 
	TypeSpecifier["Integer32"] | 
	TypeSpecifier["Integer16"] | 
	TypeSpecifier["Integer8"] |
	TypeSpecifier["UnsignedInteger"] | 
	TypeSpecifier["UnsignedInteger64"] | 
	TypeSpecifier["UnsignedInteger32"] | 
	TypeSpecifier["UnsignedInteger16"] | 
	TypeSpecifier["UnsignedInteger8"]
]
realTypeQ[t_] := MatchQ[unresolve[t], TypeSpecifier["Real"] | TypeSpecifier["Real64"] | TypeSpecifier["Real32"] | TypeSpecifier["Real16"]]
complexTypeQ[t_] := MatchQ[unresolve[t],
	TypeSpecifier["Complex"["Real"]] |
	TypeSpecifier["Complex"["Real64"]] |
	TypeSpecifier["Complex"["Real32"]] |
	TypeSpecifier["Complex"["Real16"]]
];


cast[typ_, const_] :=
	Which[
		booleanTypeQ[typ],
			TrueQ[const],
		integerTypeQ[typ],
			Floor[const],
		realTypeQ[typ],
			N[const],
		complexTypeQ[typ],
			If[MatchQ[const, _Complex],
				const,
				Complex[const, 0]
			]
	]

run[fm_, opts_] :=
	CreateInstructionVisitor[
		<|
			"visitTypeCastInstruction" -> Function[{st, inst},
			    Which[
				    ConstantValueQ[inst["source"]],
				        	With[{
				        		typ = inst["target"]["type"]
				        	},
				        		If[typ["sameQ", inst["source"]["type"]],
				        			With[{load = CreateCopyInstruction[
														inst["target"],
														inst["source"],
														inst["mexpr"]
										]},
										load["moveAfter", inst];
										load["setId", inst["id"]];
										inst["unlink"];
					                ];
				        			Return[]
				        		];
					        With[{
					        	const = cast[typ, inst["source"]["value"]]
					        },
				            With[{
				            	val = CreateConstantValue[const]
				            },
				                val["setType", typ];
				                (** we now need to replace the instruction with a load instruction.
								  * we use the same id so we do not invalidate other passes that reference
								  * the instruction id
								  *)
				                With[{load = CreateCopyInstruction[
													inst["target"],
													val,
													inst["mexpr"]
									]},
									load["moveAfter", inst];
									load["setId", inst["id"]];
									inst["unlink"];
				                ]
				            ]]],
				    VariableQ[inst["source"]],
				        	With[{
				        		trgtTyp = inst["target"]["type"],
				        		srcTyp = inst["source"]["type"]
				        	},
				        		If[deadResolveType[trgtTyp]["sameQ", deadResolveType[srcTyp]],
				        			With[{load = CreateCopyInstruction[
													inst["target"],
													inst["source"],
													inst["mexpr"]
									]},
										load["moveAfter", inst];
										load["setId", inst["id"]];
										inst["unlink"];
				                	]
				        		] 
				        	]
				    		

			    ]
			],
			"traverse" -> "reversePostOrder"
		|>,
		fm,
		"IgnoreRequiredInstructions" -> True
	]


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"CoerceCast",
	"Casts for constants and variables are replaced with load instructions with coerced types."
];

CoerceCastPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run
|>];

RegisterPass[CoerceCastPass]
]]

End[] 

EndPackage[]
