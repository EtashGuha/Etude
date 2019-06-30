
BeginPackage["TypeFramework`Environments`FunctionDefinitionLookup`"]

CreateFunctionDefinitionLookup
FunctionDefinitionLookupQ

Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]



(*
  specific holds non-overloaded functions,  ie each function name only has one definition
  monomorphic holds monomorphic functions ...  might be overloaded though
  polymorphic holds polymorphic functions ... might be overloaded and written with generic type variables
*)

RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
FunctionDefinitionLookupClass = DeclareClass[
	FunctionDefinitionLookup,
	<|
		"finalizeDefinition" -> Function[{tyEnv, name, def, ty}, finalizeDefinition[Self, tyEnv, name, def, ty]],
		"finalizeAtomDefinition" -> Function[{tyEnv, name, def, ty}, finalizeAtomDefinition[Self, tyEnv, name, def, ty]],
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
	|>,
	{
	},
	Predicate -> FunctionDefinitionLookupQ
]
]]

CreateFunctionDefinitionLookup[ ] :=
	CreateObject[FunctionDefinitionLookup, <| |>]

finalizeDefinition[ self_, tyEnv_, name_, def_, ty_] :=
	Module[ {},
		def
	]

finalizeAtomDefinition[ self_, tyEnv_, name_, def_, ty_] :=
	Module[ {},
		def
	]

(**************************************************)

icon := Graphics[Text[
	Style["FunDef",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[env_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"FunctionDefinitionLookup",
		env,
  		icon,
		{
			BoxForm`SummaryItem[{Pane["specific: ", {90, Automatic}], env["specific"]}],
			BoxForm`SummaryItem[{Pane["generic: ", {90, Automatic}], env["generic"]}]
  		},
  		{

  		}, 
  		fmt
  	]


toString[env_] := "FunctionDefinitionLookup[<>]"







End[]

EndPackage[]
