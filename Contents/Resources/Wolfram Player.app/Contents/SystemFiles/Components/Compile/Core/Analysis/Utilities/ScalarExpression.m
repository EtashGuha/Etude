

BeginPackage["Compile`Core`Analysis`Utilities`ScalarExpression`"]


CreateScalarExpression
ScalarExpressionQ


Begin["`Private`"] 

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`Instruction`InstructionQ`"] (* for InstructionQ *)
Needs["Compile`Core`IR`Instruction`CallInstruction`"]
Needs["Compile`Core`IR`Instruction`BinaryInstruction`"]
Needs["Compile`Core`IR`Instruction`UnaryInstruction`"]
Needs["Compile`Core`IR`Instruction`CompareInstruction`"]
Needs["CompileUtilities`Markup`"] (* For $UseANSI *)

$HashAlgorithm = "SHA256"
$Seed = "--WLCompiler--"
$shortenLength = 10

RegisterCallback["DeclareCompileClass", Function[{st},
DeclareClass[
	ScalarExpressionClass,
	<|
		"hash" -> (hash[Self,##]&), 
		"toString" -> Function[{}, toString[Self]], 
		"toBoxes" -> Function[{fmt},
			toBoxes[Self, fmt]
		]
	|>,
	{
		"operator",
		"operands",
		"properties"
	},
	Predicate -> ScalarExpressionQ,
	Extends -> {
		ClassPropertiesTrait
	}
]
]]


CreateScalarExpression[inst_?InstructionQ] :=
	Block[{$FrontEnd = Null, $UseANSI = False},
		With[{
			res = CreateScalarExpression[
				Which[
					CallInstructionQ[inst],
						"Call$$" <> inst["function"]["toString"],
					BinaryInstructionQ[inst],
						"BinaryOperator$$" <> ToString[inst["operator"]],
					UnaryInstructionQ[inst],
						"UnaryOperator$$" <> ToString[inst["operator"]],
					CompareInstructionQ[inst],
						"CompareOperator$$" <> ToString[inst["operator"]],
					True,
						inst["_instructionName"]
				],
				Which[
					inst["hasOperands"],
						inst["operands"],
					inst["hasSource"],
						inst["source"],
					True,
						{}
				]
			]
		},
			res["setProperty", "instruction" -> inst];
			res
		]
	];

CreateScalarExpression[op_] :=
	CreateScalarExpression[op, {}]


CreateScalarExpression[operator_, operands_] :=
	Module[{obj},
		obj = CreateObject[
			ScalarExpressionClass,
			<|
				"operator" -> operator,
				"operands" -> operands,
				"properties" -> CreateReference[<||>]
			|>
		];
		obj
	]

CreateScalarExpression[args___] :=
    ThrowException[{"Invalid call to CreateScalarExpression.", {args}}]

hash[self_, seed_:$Seed] :=
	Block[{$FrontEnd = Null, $UseANSI = False},
	With[{
		operator = self["operator"],
		operands = self["operands"]
	},
	Module[{
		repr
	},
		repr = StringJoin[
			seed,
			"[",
			If[StringQ[operator],
				operator,
				operator["toString"]
			],
			"][",
			Riffle[
				Table[
					If[StringQ[operand],
						operand,
						operand["toString"]
					],
					{operand, operands}
				],
				", "
			],
			"]"
		];
		hashString[repr]
	]]];

hashString[s_] := Hash[s, $HashAlgorithm, "HexString"]

icon := icon = Graphics[Text[
  Style["Sca\nExp", GrayLevel[0.7], Bold, 
   CurrentValue["FontCapHeight"]/
     AbsoluteCurrentValue[Magnification]]], $FormatingGraphicsOptions];  
      
toBoxes[obj_, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		"ScalarExpression",
		"",
  		icon,
  		Flatten[{
  			If[obj["hasProperty", "instruction"],
  				BoxForm`SummaryItem[{Pane["instruction: ", {90, Automatic}], obj["getProperty", "instruction"]["toString"]}],
  				{
  					BoxForm`SummaryItem[{Pane["operator: ", {90, Automatic}], obj["operator"]}],
  					BoxForm`SummaryItem[{Pane["operands: ", {90, Automatic}], obj["operands"]}]
  				}
  			],
  			BoxForm`SummaryItem[{Pane["hash: ",     {90, Automatic}], shorten[obj["hash"]]}]
  		}],
  		{
  		}, 
  		fmt
  	]
 

toString[obj_] :=
	StringJoin[
		"ScalarExpression[<",
		shorten[obj["hash"]],
		">]"
	]

shorten[s_?StringQ] :=
	With[{
		len = StringLength[s]
	},
		If[len <= $shortenLength,
			s,
			StringJoin[
				StringTake[s, $shortenLength / 2],
				ToString[StringSkeleton[
					len - $shortenLength
				]],
				StringTake[s, -$shortenLength / 2]
			]
		]
	]

End[]
EndPackage[]
