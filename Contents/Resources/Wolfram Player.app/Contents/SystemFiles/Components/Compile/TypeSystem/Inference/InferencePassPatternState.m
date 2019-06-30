
BeginPackage["Compile`TypeSystem`Inference`InferencePassPatternState`"]

CreateInferencePassPatternState
InferencePassPatternStateQ

Begin["`Private`"]

Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`Inference`PatternInferenceState`"]



RegisterCallback["DeclareCompileClass", Function[{st},
InferencePassPatternStateClass = DeclareClass[
	InferencePassPatternState,
	<|
		
		"addBinding" -> Function[{var, type}, addBinding[Self, var, type]],
		"variable" -> Function[{varId}, variable[Self, varId]],
		"patternVariables" -> Function[{}, patternVariables[Self]],
		"patternTypes" -> Function[{}, patternTypes[Self]],
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
	|>,
	{
		"variableMap",
		"patternState",
		"passState"
	},
	Predicate -> InferencePassPatternStateQ
]
]]

CreateInferencePassPatternState[passState_] :=
	Module[{},
		CreateObject[
			InferencePassPatternState,
			<|
			    "variableMap" -> CreateReference[<||>],
				"patternState" -> CreatePatternInferenceState[Null],
				"passState" -> passState
			|>
		]
	]
	

addBinding[self_, var_, type_] :=
	Module[ {},
		self["variableMap"]["associateTo", var["id"] -> var];
		self["patternState"]["addBinding",var["id"], type];
	]
	
variable[self_, varId_] :=
	Module[ {var},
		var = self["variableMap"]["lookup", varId, Null];
		If[ var === Null,
			ThrowException[{"Cannot find variable", varId}]
		];	
		var
	]

patternVariables[self_] :=
	Map[self["variable",#]&, self["patternState"]["patternVariables"]]

patternTypes[self_] :=
	self["patternState"]["patternTypes"]


(**************************************************)

icon := icon = Graphics[Text[
	Style["INF\nPST",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
        
toBoxes[t_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"InferencePassPatternState",
		t,
  		icon,
		{
            BoxForm`MakeSummaryItem[{Pane["variableMap: ",          {90, Automatic}], t["variableMap"]}, fmt]
        },
        {
            BoxForm`MakeSummaryItem[{Pane["patternState: ", 	    {90, Automatic}], t["patternState"]}, fmt]
        },
  		fmt
  	]


toString[typ_] := "InferencePassPatternState[<>]"

End[]

EndPackage[]


