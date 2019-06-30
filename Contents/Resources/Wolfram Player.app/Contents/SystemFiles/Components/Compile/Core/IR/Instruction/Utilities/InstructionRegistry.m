
BeginPackage["Compile`Core`IR`Instruction`Utilities`InstructionRegistry`"]

RegisterInstruction

$RegisteredInstructions

InstructionInformation
InstructionInformationQ
InstructionInformationClass
CreateInstructionInformation

Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Asserter`Assert`"]
Needs["CompileUtilities`Callback`"]



If[!AssociationQ[$RegisteredInstructions],
	$RegisteredInstructions = <||>
]

icon := Graphics[Text[
	Style["Instr\nInfo",
		  GrayLevel[0.7],
		  Bold,
		  1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]

toString[self_] :=
	self["name"]
toBoxes[self_, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		"InstructionInformation",
		self["toString"],
  		icon,
		{
			BoxForm`SummaryItem[{"name: ", self["name"]}],
  		    BoxForm`SummaryItem[{"properties: ", self["properties"]}]
  		},
  		{}, 
  		fmt
  	]

RegisterCallback["DeclareCompileClass", Function[{st},
InstructionInformationClass = DeclareClass[
	InstructionInformation,
	<|
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
	|>,
	{
	    "class",
		"name",
		"predicate",
		"constructor",
		"deserialize" -> Null,
		"instance",
		"properties"
	},
	Predicate -> InstructionInformationQ
]
]]

CreateInstructionInformation[info_] :=
	CreateObject[
		InstructionInformation,
		info
	]

RegisterInstruction[opts_?AssociationQ] := (
	AssertThat["Instruction has not been previously registered", opts
		]["named", opts["name"]
		]["satisfies", !KeyExistsQ[$RegisteredInstructions, #["name"]]&
	];
	$RegisteredInstructions[opts["name"]] = CreateInstructionInformation[opts]
)

RegisterInstruction[args___] :=
	ThrowException[{"Invalid RegisterInstruction arguments ", args}]

End[]
EndPackage[]
