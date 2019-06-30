
BeginPackage["Compile`Core`IR`FunctionDeclaration`"]

CreateFunctionDeclaration
FunctionDeclarationQ

Begin["`Private`"] 

Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Markup`"]



icon := Graphics[Text[
  Style["FD", GrayLevel[0.7], Bold, 
   1.2*CurrentValue["FontCapHeight"]/
     AbsoluteCurrentValue[Magnification]]], $FormatingGraphicsOptions];  

toBoxes[self_, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		"FunctionDeclaration",
		self,
  		icon,
  		{
  		    BoxForm`SummaryItem[{"name: ", self["name"]["toString"]}],
  		    BoxForm`SummaryItem[{"decl: ", self["data"]["toString"]}]
  		},
		{
		},
  		fmt
  	]
	
toString[self_] :=
	StringJoin[
		BoldBlackText["DeclareFunction"],
		"[",
		BoldBlackText[ToString[self["name"]]],
		"[",
		BoldRedText[ ToString[self["data"]]],
		"]]"
	]


RegisterCallback["DeclareCompileClass", Function[{st},
FunctionDeclarationClass = DeclareClass[
	FunctionDeclaration,
	<|
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
	|>,
	{
		"properties",
		"name",
		"data"
	},
	Predicate -> FunctionDeclarationQ,
	Extends -> {
		ClassPropertiesTrait
	}
]
]]


CreateFunctionDeclaration[name_, data_] :=
	CreateObject[
		FunctionDeclaration,
		<|
			"name" -> name,
			"data" -> data,
			"properties" -> CreateReference[<||>]
		|>
	]

CreateFunctionDeclaration[args___] :=
	ThrowException[{"Invalid arguments when CreateFunctionDeclaration ", {args}}];
	
End[]
EndPackage[]
