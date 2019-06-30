
BeginPackage["Compile`Core`IR`Internal`VariableTrait`"]

VariableTrait

Begin["`Private`"]

Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Asserter`Assert`"]
Needs["Compile`Core`IR`Instruction`InstructionQ`"]


VariableTrait := VariableTrait = 
	ClassTrait[<|
		"def" -> Function[{},
			If[Self["defs"] === {},
				None,
				First[Self["defs"]]
			]
		],
		"defs" -> Function[{},
			Self["_defs"]
		],
		"setDefs" -> Function[{insts},
			Self["set_defs", insts];
			Self
		],
		"setDef" -> Function[{inst},
			Self["set_defs", {inst}];
			Self
		],
		"addDef" -> Function[{inst},
			defs = Self["defs"];
            If[!MemberQ[#["id"]& /@ defs, inst["id"]],
                Self["setUses", Append[defs, inst]]
            ];
			Self
		],
		"addUse" -> Function[{inst},
			uses = Self["uses"];
			If[!MemberQ[#["id"]& /@ uses, inst["id"]],
				Self["setUses", Append[uses, inst]]
			];
			Self
		],
		"clearDefs" -> Function[{},
			Self["setDefs", {}];
			Self
		],
		"clearDef" -> Function[{},
			Self["clearDefs"]
		],
		"clearUses" -> Function[{},
			Self["setUses", {}];
			Self
		]
	|>]
	
End[]
EndPackage[]