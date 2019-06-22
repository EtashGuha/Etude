
BeginPackage["TypeFramework`Environments`TypeConstructorEnvironment`"]

TypeConstructorEnvironmentQ
CreateTypeConstructorEnvironment

Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]



RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
TypeConstructorEnvironmentClass = DeclareClass[
	TypeConstructorEnvironment,
	<|
		"add" -> Function[{ty}, addConstructor[Self, ty]],
		"scanAll" -> Function[ {data, fun}, scanAll[Self, data, fun]],
		"lookup" -> Function[{name}, lookup[Self, name]],
		"dispose" -> Function[{}, dispose[Self]],
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
	|>,
	{
		"types"
	},
	Predicate -> TypeConstructorEnvironmentQ
]
]]


addConstructor[ self_, ty_] :=
	self["types"]["associateTo", ty["name"] -> ty]

lookup[ self_, name_] :=
	self["types"]["lookup", name, Undefined]



dispose[self_] :=
	(
	self["setTypes", Null];
	)

CreateTypeConstructorEnvironment[] :=
	CreateObject[TypeConstructorEnvironment, <|
			"types" -> CreateReference[<||>]
		|>]


scanAll[self_, data_, fun_] :=
	Scan[ fun[data, #]&, self["types"]["values"]]



(**************************************************)

icon := Graphics[Text[
	Style["TConsEnv",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[env_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"TypeConstructorEnvironment",
		env,
  		icon,
		{
			BoxForm`SummaryItem[{Pane["length: ", {90, Automatic}], Length[env["types"]["get"]]}]
  		},
  		{
  			BoxForm`SummaryItem[{Pane["types: ", {90, Automatic}], env["types"]}]
  		}, 
  		fmt
  	]


toString[env_] := "TypeConstructorEnvironment[<>]"


End[]

EndPackage[]

