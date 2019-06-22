BeginPackage["Compile`AST`Macro`Builtin`MetaData`"]
Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`AST`Macro`MacroEnvironment`"]
Needs["TypeFramework`"] (* for MetaData *)
Needs["CompileAST`Utilities`MExprVisitor`"]

addPropertyVisitor[st_, expr_] :=
	(
		expr["properties"]["join", st["props"]];
		expr["setProperty", "MacroExpandAgain" -> True]
	)

RegisterCallback["DeclareCompileClass", Function[{st},
DeclareClass[
	RecursiveAddPropertyVisitor,
	<|
		"visitSymbol" -> (addPropertyVisitor[Self, ##]&),
		"visitInteger" -> (addPropertyVisitor[Self, ##]&),
		"visitReal" -> (addPropertyVisitor[Self, ##]&),
		"visitBoolean" -> (addPropertyVisitor[Self, ##]&),
		"visitString" -> (addPropertyVisitor[Self, ##]&),
		"visitNormal" -> (addPropertyVisitor[Self, ##]&)
	|>,
	{
		"props"
	},
	Extends -> {MExprVisitorClass}]
]]


recursiveAddProperty[expr_, props_] :=
	With[{
		visitor = CreateObject[RecursiveAddPropertyVisitor, <|
			"props" -> props
		|>]
	},
		expr["accept", visitor];
		expr
	]

addProperties[expr_, props0_] :=
	Module[{
		props = ReleaseHold[props0["toExpression"]]
	},
		Which[
			AssociationQ[props],
				None, (* do nothing *)
			Head[props] === Rule,
				props = Association[{props}],
			ListQ[props],
				props = <| "properties" -> props |>,
			!ListQ[props],
				props = <| "properties" -> {props} |>
		];
		If[Lookup[props, "MetaDataRecursive", False] === True,
			KeyDropFrom[props, "MetaDataRecursive"];
			recursiveAddProperty[expr, props];
		];
		expr["setProperty",
			expr["setProperty", "MacroExpandAgain" -> True]
		];
		expr["setProperties",
			expr["properties"]["join", props]
		];
		expr
	];

setupMacros[st_] :=
	Module[ {env = st["macroEnvironment"]},
	
	RegisterMacro[env, MetaData,
	    MetaData[props_][expr_] ->
	    	Compile`Internal`MacroEvaluateHeld[
				addProperties[expr, props]
			]
	]
]

RegisterCallback["SetupMacros", setupMacros]


End[]
EndPackage[]
