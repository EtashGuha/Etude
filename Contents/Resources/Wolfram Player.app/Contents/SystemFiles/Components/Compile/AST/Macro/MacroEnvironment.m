BeginPackage["Compile`AST`Macro`MacroEnvironment`"]

CreateMacroEnvironment
MacroEnvironmentQ
RegisterMacro
ExtendMacro
MacroEnvironment

Begin["`Private`"]


Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileAST`Create`Construct`"]
Needs["CompileAST`PatternMatching`Replace`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"] (* for MetaData *)


RegisterCallback["DeclareCompileClass", Function[{st},
MacroEnvironmentClass = DeclareClass[
	MacroEnvironment,
	<|
		"hasRules" -> Function[{key}, hasRules[Self, key]],
		"getRules" -> Function[{key}, getRules[Self, key]],		
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
	|>,
	{
		"rules"
	},
	Predicate -> MacroEnvironmentQ
]
]]


CreateMacroEnvironment[] :=
	CreateObject[MacroEnvironment, <|
		"rules" -> CreateReference[<||>]
	|>]

ruleMExpr := ruleMExpr = CreateMExprSymbol[Rule]
	
SetAttributes[toMExprReplacer, HoldAllComplete]

toMExprReplacer[env_, name_, rules__] :=
	With[{
		hd = CreateMExprSymbol[List],
		args = Apply[
			List,
			Map[
				Function[elem,
					With[{
						expr = CreateMExpr[elem]
					},
						(* If the rule is of the form
						 * 		MetaData[..]@rhs -> lhs
						 * then we set the metadata as on
						 * the expr with the metadata specified
						 *)
						If[TrueQ[
							name =!= MetaData &&
							expr["hasHead", Rule] &&
						   	expr["part", 1]["normalQ"] &&
						   	expr["part", 1]["head"]["normalQ"] &&
						   	expr["part", 1]["head"]["hasHead", TypeFramework`MetaData]
						   ],
						   Module[{
						   		metadata = First[ReleaseHold[expr["part", 1]["head"]["toExpression"]]]
						    },
						    	Which[
						    		Head[metadata] === Rule,
						    			metadata = Association[{metadata}],
						    		ListQ[metadata] && MatchQ[metadata, {__Rule}],
						    			metadata = Association[metadata],
						    		ListQ[metadata],
						    			metadata = Association["properties" -> metadata],
						    		AssociationQ[metadata],
						    			None (* do nothing *)
						    	];
						    		
							With[{
								lhs = expr["part", 1]["part", 1],
								rhs = expr["part", 2]
							},
							With[{
								newExpr = CreateMExprNormal[ruleMExpr, {lhs, rhs}]
							},
								newExpr["setProperties", metadata];
								newExpr
							]]],
							expr
						]
					],
					{HoldAllComplete}
				],
				Unevaluated[{rules}]
			]
		]},
		CreateMExprReplacer[CreateMExprNormal[hd, args]]
	]
	
SetAttributes[RegisterMacro, HoldAllComplete]
SetAttributes[iExtendMacro, HoldAllComplete]
SetAttributes[iRegisterMacro, HoldAllComplete]

iExtendMacro[env_?MacroEnvironmentQ, name_, rules__] :=
	With[{
		newReplacer = toMExprReplacer[env, name, rules],
		currentReplacer = env["rules"]["lookup", name] 
	},
		env["rules"]["associateTo",
			name -> If[MissingQ[currentReplacer],
				newReplacer,
				newReplacer["join", currentReplacer]
			]
		]
	]
	
iRegisterMacro[env_?MacroEnvironmentQ, name_, rules__] :=
	env["rules"]["associateTo",
		name -> toMExprReplacer[env, name, rules]
	]
		
RegisterMacro[env_?MacroEnvironmentQ, name_, rules__, "Extend" -> True] :=
	iExtendMacro[env, name, rules]
	
RegisterMacro[env_?MacroEnvironmentQ, name_, rules__, "Extend" -> False] :=
	iRegisterMacro[env, name, rules]
	
RegisterMacro[env_?MacroEnvironmentQ, name_, rules__] :=
	RegisterMacro[env, name, rules, "Extend" -> False]
	
RegisterMacro[args___] := ThrowException[{"Invalid usage for RegisterMacro using value ", args}]

hasRules[ self_, key_] :=
	self["rules"]["keyExistsQ", key]

getRules[ self_, key_] :=
	self["rules"]["lookup", key, {}]


(**************************************************)

icon := Graphics[Text[
	Style["MAC\nENV",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[obj_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"MacroEnvironment",
		obj,
  		icon,
		{
			BoxForm`SummaryItem[{Pane["rule length: ", {90, Automatic}], obj["rules"]["length"]}]
  		},
  		{
   		},
  		fmt
  	]


toString[env_] := "MacroEnvironment[<>]"





End[]

EndPackage[]
