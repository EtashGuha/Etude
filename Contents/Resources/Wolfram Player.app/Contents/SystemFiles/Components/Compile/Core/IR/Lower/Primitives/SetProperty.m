BeginPackage["Compile`Core`IR`Lower`Primitives`SetProperty`"]

Begin["`Private`"]


Needs["Compile`Core`IR`Lower`Utilities`LanguagePrimitiveLoweringRegistry`"]
Needs["Compile`Core`IR`Lower`Primitives`LanguagePrimitive`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Callback`"]



lower[state_, mexpr_, opts_] :=
	Module[{builder, sym, prop},
		If[mexpr["length"] =!= 2,
			ThrowException[{"Invalid number for the Native`SetProperty statement in " <> mexpr["toString"]}]
		];
		builder = state["builder"];
		sym = state["lower", mexpr["part", 1], opts];
		prop = ReleaseHold[mexpr["part", 2]["toExpression"]];
		If[!AssociationQ[prop],
			prop = Association[prop]
		];
		If[!AssociationQ[prop], (* unable to create an association *)
			ThrowException[{"Invalid usage of Native`SetProperty statement in " <> mexpr["toString"] <> " the form must be either " <>
							"Native`SetProperty[var, \"key\" -> val] or Native`SetProperty[var, <| \"key\"... -> vals, ... |>]"
			}]
		];
		If[sym =!= Undefined,
			sym["properties"]["join", prop];
		];
	    sym
	];


RegisterCallback["RegisterPrimitive", Function[{st},
RegisterLanguagePrimitiveLowering[CreateSystemPrimitive[Native`SetProperty], lower]
]]

End[]

EndPackage[]
