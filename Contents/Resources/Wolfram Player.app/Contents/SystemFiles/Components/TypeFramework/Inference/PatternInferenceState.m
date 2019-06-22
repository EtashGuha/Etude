
BeginPackage["TypeFramework`Inference`PatternInferenceState`"]

PatternInferenceStateQ
CreatePatternInferenceState

Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["TypeFramework`"]



RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
PatternInferenceStateClass = DeclareClass[
	PatternInferenceState,
	<|
		"addBinding" -> Function[ {var, ty}, addBinding[ Self, var, ty]],
		"generate" -> Function[ {args}, generate[ Self, args]],
		"patternVariables" -> Function[ {}, patternVariables[ Self]],
		"patternTypes" -> Function[ {}, patternTypes[ Self]],
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
	|>,
	{
		"generateFunction",
		"constraints",
		"binding"
	},
	Predicate -> PatternInferenceStateQ
]
]]



CreatePatternInferenceState[ generateFunction_] :=
	CreateObject[PatternInferenceState, <|
			"generateFunction" -> generateFunction,
			"constraints" -> CreateReference[{}],
			"binding" -> CreateReference[<||>]
		|>]



patternVariables[ self_] :=
	self["binding"]["keys"]

patternTypes[ self_] :=
	self["binding"]["values"]


addBinding[ self_, var_, ty_] :=
	(
	self["binding"]["associateTo", var -> ty];
	)


 

(*
  Not totally sure if this list code shouldn't be part of the 
  generateFunction.
*)
 
generate[ self_, args_List] :=
	Map[ self["generateFunction"][self,#]&,   args]
 


generate[ args___] :=
	ThrowException[TypeInferenceException[{"Unknown argument to generate", {args}}]] 




(**************************************************)

icon := Graphics[Text[
	Style["PState",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]
      
toBoxes[env_, fmt_]  :=
	BoxForm`ArrangeSummaryBox[
		"PatternInferenceState",
		env,
  		icon,
		{
			BoxForm`SummaryItem[{Pane["name: ", {90, Automatic}], env["name"]}]
  		},
  		{
  			BoxForm`SummaryItem[{Pane["constructors: ", {90, Automatic}], env["typeconstructors"]}]
  		}, 
  		fmt
  	]


toString[env_] := "PatternInferenceState[<>]"



End[]

EndPackage[]

