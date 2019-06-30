
BeginPackage["Compile`Core`IR`TypeDeclaration`"]

CreateTypeDeclaration
TypeDeclarationQ

Begin["`Private`"] 

Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Markup`"]
Needs["TypeFramework`"]



icon := Graphics[Text[
  Style["TD", GrayLevel[0.7], Bold, 
   1.2*CurrentValue["FontCapHeight"]/
     AbsoluteCurrentValue[Magnification]]], $FormatingGraphicsOptions];  

toBoxes[self_, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		"TypeDeclaration",
		self,
  		icon,
  		{
  		    BoxForm`SummaryItem[{"type: ", ToString[self["type"]]}]
  		},
		{
		},
  		fmt,
		"Interpretable" -> False
  	]
	
toString[self_] :=
	StringJoin[
		BoldBlackText["DeclareType"],
		"[",
		BoldBlackText["Type"],
		"[",
		BoldRedText[ ToString[self["type"]]],
		"]]",
		If[self["hasProperty", "private"] && self["getProperty", "private", False],
			GrayText["\t\t  (* Private *) "],
			""
		]
	]


RegisterCallback["DeclareCompileClass", Function[{st},
TypeDeclarationClass = DeclareClass[
	TypeDeclaration,
	<|
		"resolve" -> Function[{tyEnv}, resolve[Self, tyEnv]],
		"resolveWithVariables" -> Function[{tyEnv, tyVarMap}, resolveWithVariables[Self, tyEnv, tyVarMap]],
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
	|>,
	{
		"properties",
		"type" -> Undefined
	},
	Predicate -> TypeDeclarationQ,
	Extends -> {
		ClassPropertiesTrait
	}
]
]]


CreateTypeDeclaration[type_] :=
	CreateObject[
		TypeDeclaration,
		<|
			"type" -> type,
			"properties" -> CreateReference[<||>]
		|>
	]

CreateTypeDeclaration[args___] :=
	ThrowException[{"Invalid arguments when CreateTypeDeclaration ", {args}}];
	
	
resolve[self_, tyEnv_] :=
	If[!TypeObjectQ[self["type"]],
		self["setType", tyEnv["resolve", self["type"]]]
	]
resolveWithVariables[self_, tyEnv_, tyVarMap_] :=
	If[!TypeObjectQ[self["type"]],
		self["setType", tyEnv["resolveWithVariables", self["type"], tyVarMap]]
	]
	
End[]
EndPackage[]
