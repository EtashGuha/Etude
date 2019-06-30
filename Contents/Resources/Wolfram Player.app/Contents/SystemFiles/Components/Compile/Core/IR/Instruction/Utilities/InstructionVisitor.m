(* TODO:
 * The current implementation of create instruction visitor will invoke the visitor
 * when passed a module (either a program module, function module, or basic block ).
 * Based on the usage this might not be a good idea
 *)
 
BeginPackage["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]

InstructionVisitor
InstructionVisitorQ
CreateInstructionVisitor

Begin["`Private`"]

Needs["Compile`Core`IR`ProgramModule`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`IR`BasicBlock`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionRegistry`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Asserter`Assert`"]
Needs["Compile`Core`IR`Instruction`InstructionQ`"]
Needs["CompileUtilities`Callback`"]


Options[CreateInstructionVisitor] = {
	"TraversalOrder" -> Automatic,
    "IgnoreRequiredInstructions" -> False
}	

RegisterCallback["DeclareCompileClassPostRegisterInstruction", Function[{st},
Module[{methods, fields},
	fields = 
		Join[
			Table[ "_visit" <> inst, {inst, Keys[$RegisteredInstructions]}]
			,
			{
				"state" -> None,
				"traversalOrder" -> Automatic,
				"_visitProgramModule",
				"_visitFunctionModule" ,
				"_visitBasicBlock",
				"_visitInstruction"
			}
		];
	methods =
		Join[
			Table[
				With[{visitInst = "visit" <> inst, underVisitInst = "_visit" <> inst},
					visitInst -> (Self[underVisitInst][Self["state"], ##]&)
				],
				{inst, Keys[$RegisteredInstructions]}]
			,
			{
			"traverse" -> (traverse[Self, ##]&),
			"visit" -> (visit[Self, ##]&),
			"visitProgramModule" -> (Self["_visitProgramModule"][Self["state"], ##]&),
			"visitFunctionModule" -> (Self["_visitFunctionModule"][Self["state"], ##]&),
			"visitBasicBlock" -> (Self["_visitBasicBlock"][Self["state"], ##]&),
			"visitInstruction" -> (Self["_visitInstruction"][Self["state"], ##]&),
			"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]]
			}
		];
	methods = Association[ methods];
	DeclareClass[InstructionVisitor, methods, fields]
]
]]

visit[st_, v_] :=
	If[InstructionQ[v],
		st["_visit" <> v["_instructionName"]][st["state"], v],
		traverse[st, v]
	]

CreateInstructionVisitor[visitors_?AssociationQ, opts:OptionsPattern[]] :=
	CreateInstructionVisitor[Lookup[visitors, "state", None], visitors, opts]
CreateInstructionVisitor[state_, visitors_?AssociationQ, opts:OptionsPattern[]] :=
	Module[{
		fields,
		visitInstruction,
		unvisited,
		visitsAllInstructionQ = False,
		inst,
		identity = #2&,
		traversalOrder = Automatic
	},
		
		(* Check if all the instructions required to visited are visited, or the user knows what 
		 * he is doing and has passed IgnoreRequiredInstructions->True to avoid the exception.
		 *)
		Which[
		    OptionValue[CreateInstructionVisitor, {opts}, "IgnoreRequiredInstructions"] === True,
		    		visitsAllInstructionQ = True,
		    KeyExistsQ[visitors, "visitInstruction"],
		    		visitsAllInstructionQ = True,
		    (unvisited = Complement[
		        Keys[
		            Select[$RegisteredInstructions, MemberQ[#["properties"], "visitingRequired"] &]
		        ],
		        StringTrim[#, RegularExpression["^visit*"]]& /@ Keys[visitors]
		    ]) === {},
		    		visitsAllInstructionQ = True
		];
		
		With[{
			order = OptionValue[CreateInstructionVisitor, {opts}, "TraversalOrder"]
		},
			If[!MemberQ[validTraversalOrders, order],
				ThrowException[{"When calling CreateInstructionVisitor:: Invalid traversal order ", order}]
			];
			traversalOrder = order
		];
		
		If[visitsAllInstructionQ === False,
		    ThrowException[{"Not all required instructions are visited :: ", unvisited}]
		];
		visitInstruction = Lookup[visitors, "visitInstruction", identity];
		fields = Association[
			"state" -> state,
			"traversalOrder" -> traversalOrder,
			Join[
				Table[
					"_visit" <> inst -> Lookup[visitors, "visit" <> inst, visitInstruction],
					{inst, Keys[$RegisteredInstructions]}
				],
				{
					"_visitProgramModule" -> Lookup[visitors, "visitProgramModule", identity],
					"_visitFunctionModule" -> Lookup[visitors, "visitFunctionModule", identity],
					"_visitBasicBlock" -> Lookup[visitors, "visitBasicBlock", identity],
					"_visitInstruction" -> visitInstruction
				}
			]
		];
		CreateObject[
			InstructionVisitor,
			fields
		]
	]

irModuleQ[mod_] := FunctionModuleQ[mod] || ProgramModuleQ[mod] || BasicBlockQ[mod]

CreateInstructionVisitor[visitors_?AssociationQ, mod_?irModuleQ, opts:OptionsPattern[]] :=
	CreateInstructionVisitor[None, visitors, mod, opts]
CreateInstructionVisitor[state_, visitors_?AssociationQ, mod_?irModuleQ, opts:OptionsPattern[]] :=
	Module[{inst},
		Assert[ProgramModuleQ[mod] || FunctionModuleQ[mod] || BasicBlockQ[mod]];
		inst = CreateInstructionVisitor[state, visitors, opts];
		inst["traverse", mod]
	]
	
CreateInstructionVisitor[args___] :=
	ThrowException[{"Invalid call to CreateInstructionVisitor", args}]
	
InstructionVisitorQ[ObjectInstance[InstructionVisitor, accessor_, st_?ReferenceQ]] := True
InstructionVisitorQ[___] := False

traverse[visitor_, mod_] :=
	traverse[visitor, visitor["traversalOrder"], mod]
traverse[visitor_, order_, pm_?ProgramModuleQ] := (
	pm["scanFunctionModules",
		Function[{fm},
			visitor["visitFunctionModule", fm];
			visitor["traverse", order, fm]
		]
	];
	pm
)
traverse[visitor_, order_, fm_?FunctionModuleQ] := Module[{f},
	f = Function[{bb},
		visitor["visitBasicBlock", bb];
		visitor["traverse", order, bb] 
	];
	Switch[order,
		Automatic,
			fm["topologicalOrderScan", f],
		"reversePostOrder",
			fm["reversePostOrderScan", f],
		"postOrder",
			fm["postOrderScan", f],
		"reversePreOrder",
			fm["reversePreOrderScan", f],
		"preOrder",
			fm["preOrderScan", f],
		"topologicalOrder",
			fm["topologicalOrderScan", f],
		"postTopologicalOrder",
			fm["postTopologicalOrderScan", f],
		"anyOrder",
			fm["scanBasicBlocks", f],
		_,
			checkTraversalOrder[order];
	];
	fm
];
checkTraversalOrder[order_] :=
	AssertThat["The traversal order must be valid",
		order]["named",
		"traversalOrder"]["isMemberOf", validTraversalOrders]
validTraversalOrders = {
	Automatic,
	"reversePostOrder",
	"postOrder",
	"reversePreOrder",
	"preOrder",
	"topologicalOrder",
	"postTopologicalOrder",
	"anyOrder"
}

traverse[visitor_, order_, bb_?BasicBlockQ] :=
	Module[ {f = Function[{inst},
				visitor[inst["_visitInstructionName"], inst]
			]},
		If[ order === "reverseInstructions",
			bb["reverseScanInstructions", f],
			bb["scanInstructions", f]];
		bb
	]


traverse[visitor_, args___] :=
	ThrowException[{"Invalid argument to traversal ", args}]
	
(*********************************************************)

icon := Graphics[Text[
	Style["Instr\nVisitor",
		  GrayLevel[0.7],
		  Bold,
		  CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
	]], $FormatingGraphicsOptions
]

toBoxes[var_?InstructionVisitorQ, fmt_] :=
	Module[{visitors},
		visitors = var["_fields"];
		BoxForm`ArrangeSummaryBox[
			"InstructionVisitor",
			var,
  			icon,
  			{
  		    	BoxForm`SummaryItem[{"visitors: ", visitors}]
  			},
  			{}, 
  			fmt,
			"Interpretable" -> False
  		]
	]
  	
End[]
EndPackage[]
