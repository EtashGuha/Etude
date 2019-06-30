BeginPackage["Compile`Core`Lint`IR`"]

LintIR
LintIRPass

Begin["`Private`"] 

Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["CompileUtilities`Error`Exceptions`"]
Needs["Compile`Core`IR`ProgramModule`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`Lint`Utilities`"] (* For LintFailure *)

(* ::Subsection:: *)
(* Utilities *)

(*
 Set up for basic blocks,  keep list of bb name to bb
 also a map for each bb to reach children and parents
*)
getBasicBlocks[st_, fm_] :=
	Module[{},
		fm["topologicalOrderScan",
			Function[{bb},
				Module[{children, parents},
					st["basicBlocks"]["associateTo", bb["fullName"] -> bb];
					children = Map[ #["fullName"]&, bb["children"]["get"]];
					parents = Map[ #["fullName"]&, bb["parents"]["get"]];
					st["basicBlockChildren"]["associateTo", bb["fullName"] -> children];
					st["basicBlockParents"]["associateTo", bb["fullName"] -> parents];
				]
			]
		];
	]



(* Get some internal helper functions from Compile`Core`Lint`Utilities` *)
toString := toString = Compile`Core`Lint`Utilities`Private`toString
toFailure := toFailure = Compile`Core`Lint`Utilities`Private`toFailure
printIfNotQuiet := printIfNotQuiet = Compile`Core`Lint`Utilities`Private`printIfNotQuiet
error := error = Compile`Core`Lint`Utilities`Private`error
warn := warn = Compile`Core`Lint`Utilities`Private`warn
(***********************************************************************)

(* ::Subsection:: *)
(* Error Message *)


errorVariableKnown[st_, var_, inst_] :=
	error[
		"KnownVariable",
		inst,
		TemplateApply[
			StringTemplate[
				"The variable `var` has been already been visited in `inst`.",
				InsertionFunction -> toString
			],
			<|
				"var" -> var,
				"inst" -> st["variables"]["lookup", var["id"]]
			|>
		]
	]
errorVariableUnknown[st_, var_, inst_] :=
	error[
		"KnownVariable",
		inst,
		TemplateApply[
			StringTemplate[
				"The variable `var` is unknown.",
				InsertionFunction -> toString
			],
			<|
				"var" -> var
			|>
		]
	]
errorVariableDefined[st_, var_, inst_] :=
	error[
		"DuplicateVariable",
		inst,
		TemplateApply[
			StringTemplate[
				"The variable `var` defined by `inst` has already been defined.",
				InsertionFunction -> toString
			],
			<|
				"var" -> var,
				"inst" -> st["definedVariables"]["lookup", var["id"]]
			|>
		]
	]
errorVariableUndefined[st_, var_, inst_] := (
	Assert[StringQ[var["name"]]];
	error[
		"UndefinedVariable",
		inst,
		TemplateApply[
			StringTemplate[
				"The variable `var` with name '`name`' has not been defined before use.",
				InsertionFunction -> toString
			],
			<|
				"var" -> var,
				"name" -> var["name"]
			|>
		]
	]
)

errorVariableIsReference[st_, var_, inst_] :=
	error[
		"ReferenceVariable",
		inst,
		TemplateApply[
			StringTemplate[
				"The variable `var` is expected to be a register, but it is defined as a reference.",
				InsertionFunction -> toString
			],
			<|
				"var" -> var
			|>
		]
	]
errorVariableIsNotReference[st_, var_, inst_] :=
	error[
		"NotReferenceVariable",
		inst,
		TemplateApply[
			StringTemplate[
				"The variable `var` defined in `inst` is expected to be a reference, but it is not defined as a reference.",
				InsertionFunction -> toString
			],
			<|
				"inst" -> st["definedVariables"]["lookup", var["id"]],
				"var" -> var
			|>
		]
	]

errorBasicBlockUndefined[st_, bb_, inst_] :=
	error[
		"UndefinedBasicBlock",
		inst,
		TemplateApply[
			StringTemplate[
				"The basic block `bb` is not defined within the function module.",
				InsertionFunction -> toString
			],
			<|
				"bb" -> bb
			|>
		]
	]

errorBasicBlockNotParent[st_, bb_, inst_] :=
	error[
		"BasicBlockNotParent",
		inst,
		TemplateApply[
			StringTemplate[
				"The basic block `bb` is not a parent of basic block for `inst`.",
				InsertionFunction -> toString
			],
			<|
				"bb" -> bb, "inst" -> inst
			|>
		]
	]

errorBasicBlockNotChild[st_, bb_, inst_] :=
	error[
		"BasicBlockNotChild",
		inst,
		TemplateApply[
			StringTemplate[
				"The basic block `bb` is not a child of basic block for `inst`.",
				InsertionFunction -> toString
			],
			<|
				"bb" -> bb, "inst" -> inst
			|>
		]
	]
	
(* ::Subsection:: *)
(* Lint Rules *)

basicBlockDefined[st_, bb_] :=
	With[{ 
		name = bb["fullName"]
	}, 
		st["basicBlocks"]["keyExistsQ", name]
	]
		
		
checkBasicBlockDefined[st_, bb_, inst_] :=
	With[{ 
		name = bb["fullName"]
	}, 
		If[!st["basicBlocks"]["keyExistsQ", name],
			Print[{st["basicBlocks"]["keys"], name}];
			errorBasicBlockUndefined[st, name, inst];
			False,
			True
		]
	];

(*
 Check that bb is a parent of the basic block of inst
*)
checkBasicBlockParent[st_, bb_, inst_] :=
	With[{ 
		bbs = st["basicBlockParents"]["lookup", inst["basicBlock"]["fullName"]],
		name = bb["fullName"]
	}, 
		If[!MemberQ[bbs, name],
			Print[{st["basicBlocks"]["keys"], name}];
			errorBasicBlockNotParent[st, name, inst["toString"]];
			False,
			True
		]
	];

(*
 Check that bb is a child of the basic block of inst
*)		
checkBasicBlockChild[st_, bb_, inst_] :=
	With[{ 
		bbs = st["basicBlockChildren"]["lookup", inst["basicBlock"]["fullName"]],
		name = bb["fullName"]
	}, 
		If[!MemberQ[bbs, name],
			Print[{st["basicBlocks"]["keys"], name}];
			errorBasicBlockNotChild[st, name, inst["toString"]];
			False,
			True
		]
	];
		
		
checkVariableKnown[st_, var_?VariableQ, inst_] :=
	With[{ 
		name = var["id"]
	}, 
		If[!st["variables"]["keyExistsQ", name],
			errorVariableKnown[st, var, inst];
			False,
			True
		]
	];
		
checkVariableUnknown[st_, var_?VariableQ, inst_] :=
	With[{ 
		name = var["id"]
	},
		If[st["variables"]["keyExistsQ", name],
			errorVariableUnknown[st, var, inst];
			False,
			True
		]
	];
	
checkVariableDefined[st_, var_?VariableQ, inst_] :=
	With[{ 
		name = var["id"]
	}, 
		If[!st["definedVariables"]["keyExistsQ", name],
			errorVariableUndefined[st, var, inst];
			False,
			True
		]
	];

checkVariableDefinedReference[st_, var_?VariableQ, inst_] :=
	checkVariableDefined[st, var, inst] &&
	checkVariableIsReference[st, var, inst];
	
checkVariableDefinedNonReference[st_, var_?VariableQ, inst_] :=
	checkVariableDefined[st, var, inst] &&
	checkVariableIsNotReference[st, var, inst];

checkVariableDefinedNonReference[st_, var_?ConstantValueQ, inst_] :=
	True


checkVariableDefinedNonReference[args___] :=
	ThrowException[{"Unrecognized call to checkVariableDefinedNonReference", {args}}]

checkVariableUndefined[st_, var_?VariableQ, inst_] :=
	With[{ 
		name = var["id"]
	}, 
		If[st["definedVariables"]["keyExistsQ", name],
			errorVariableDefined[st, var, inst];
			False,
			True
		]
	];

checkVariableIsNotReference[st_, var_?VariableQ, inst_] :=
	With[{ 
		name = var["id"]
	}, 
		If[st["referenceVariables"]["keyExistsQ", name],
			errorVariableIsReference[st, var, inst];
			False,
			True
		]
	];
	
checkVariableIsReference[st_, var_?VariableQ, inst_] :=
	With[{ 
		name = var["id"]
	}, 
		If[st["referenceVariables"]["keyExistsQ", name],
			errorVariableIsNotReference[st, var, inst];
			False,
			True
		]
	];

(* ::Subsection:: *)
(* Register *)	

addPhiSource[ st_, instBB_, bbs_] :=
	Module[ {instBBId = instBB["id"], bbList},
		If[ 
			!st["phiSource"]["keyExistsQ", instBBId],
			 	st["phiSource"]["associateTo", instBBId -> CreateReference[{}]]];
		bbList = st["phiSource"]["lookup", instBBId];
		Map[ bbList["appendTo", #]&, bbs];
	]
	
addVariableToReferenceVariables[st_, var_, inst_] :=
	If[VariableQ[var],
		st["referenceVariables"]["associateTo", var["id"] -> inst],
		warn[
			"AddingNonVariable",
			inst,
			TemplateApply[
				StringTemplate[
					"Attempting to add a non-variable `var` to the reference variable list while analyzing the instruction `inst`",
					InsertionFunction -> toString
				],
				<|
					"var" -> var,
					"inst" -> inst
				|>
			]
		]
	]
addVariableToDefinedVariables[st_, var_, inst_] :=
	If[VariableQ[var],
		st["definedVariables"]["associateTo", var["id"] -> inst],
		warn[
			"AddingNonVariable",
			inst,
			TemplateApply[
				StringTemplate[
					"Attempting to add a non-variable `var` to the defined variable list while analyzing the instruction `inst`",
					InsertionFunction -> toString
				],
				<|
					"var" -> var,
					"inst" -> inst
				|>
			]
		]
	]

addVariableToKnownVariables[st_, var_, _] :=
	If[VariableQ[var],
		st["variables"]["associateTo", var["id"] -> var],
		warn[
			"AddingNonVariable",
			inst,
			TemplateApply[
				StringTemplate[
					"Attempting to add a non-variable `var` to the known variable list",
					InsertionFunction -> toString
				],
				<|
					"var" -> var
				|>
			]
		]
	]

(*
 Verify that the live[in] and live[out] properties of the basic blocks are consistent 
 with those of the instructions.
*)

verifyBasicBlockLive[ st_, fm_, bb_] :=
    Module[ {liveIn, liveOut, insIn, insOut, inst},
        inst = bb["firstNonLabelInstruction"];
        liveIn = Map[#["id"]&, bb["getProperty", "live[in]"]];
        insIn = Map[#["id"]&, inst["getProperty", "live[in]"]];
        If[ Sort[liveIn] =!= Sort[insIn],
            warn[
	            "live[in] Mismatch",
	            inst,
	            TemplateApply[
	                StringTemplate[
	                    "live[in] of BasicBlock `bb` = `liveIn` does not match that of its first instruction `inst` = `insIn`",
	                    InsertionFunction -> toString
	                ],
	                <|
	                	"bb" -> bb,
	                	"inst" -> inst,
	                	"liveIn" -> liveIn,
	                	"insIn" -> insIn
                	|>
	            ]
			]
        ];
        liveOut = Map[#["id"]&, bb["getProperty", "live[out]"]];
        insOut = Map[#["id"]&, bb["getLastInstruction"]["getProperty", "live[out]"]];
        If[ Sort[liveOut] =!= Sort[insOut],
            warn[
	            "live[out] Mismatch",
	            bb["getLastInstruction"],
	            TemplateApply[
	                StringTemplate[
	                    "live[out] of BasicBlock `bb` = `liveIn` does not match that of its first instruction `inst` = `insIn`",
	                    InsertionFunction -> toString
	                ],
	                <|
	                	"bb" -> bb,
	                	"inst" -> bb["getLastInstruction"],
	                	"liveIn" -> liveOut,
	                	"insIn" -> insOut
                	|>
	            ]
	          ]
        ]
    ]


equalIds[ list1_, list2_] :=
	Sort[ Map[ #["id"]&, list1]] === Sort[ Map[ #["id"]&, list2]]

containsId[ list1_, item_] :=
	MemberQ[ Map[ #["id"]&, list1], item["id"]]


(*
  Verify that 
     children matches branch
     source of each Phi Instruction points to a parent
     the parents of the children contain this basic block
     each child of the parents contains this basic block
*)
verifyBasicBlockLinks[ st_, fm_, bb_] :=
	Module[ {children, last, parents, phiBBs},
		children = bb["getChildren"];
		last = bb["lastInstruction"];
		If[ last["_instructionName"] === "BranchInstruction" && !equalIds[ children, last["operands"]],
			warn[
            "children/branch Mismatch",
            last,
            TemplateApply[
                StringTemplate[
                    "The children of BasicBlock `bb` do not match that of its last instruction `inst`",
                    InsertionFunction -> toString
                ], <|"bb" -> bb["name"], "inst" -> last |>]]];		
		parents = bb["getParents"];
		phiBBs = st["phiSource"]["lookup", bb["id"], {}];
		If[ !ListQ[ phiBBs], phiBBs = phiBBs["toList"]];
		Scan[ 
			If[ !containsId[ parents, #],
				warn[
            	"parents/phi Mismatch",
            	#,
            	TemplateApply[
                	StringTemplate[
                    	"BasicBlock `PhiSource` is referenced in a phi instruction but does not in the list of parents of `bb`",
                    	InsertionFunction -> toString
                	], <|"PhiSource" -> #, "bb" -> bb["name"]|>]]]&, phiBBs];
		Scan[
			(If[ !basicBlockDefined[ st, #],
				warn[
            	"child NotFound",
            	#,
            	TemplateApply[
                	StringTemplate[
                    	"BasicBlock `child` is declared a child of `bb` but is not found in the Function Module",
                    	InsertionFunction -> toString
                	], <|"child" -> #["name"], "bb" -> bb["name"]|>]]];
			
			If[ !containsId[ #["getParents"], bb],
				warn[
            	"child Mismatch",
            	#,
            	TemplateApply[
                	StringTemplate[
                    	"BasicBlock `child` is a child of `bb` but does not contain it in its list of parents",
                    	InsertionFunction -> toString
                	], <|"child" -> #["name"], "bb" -> bb["name"]|>]]])&, children];
		Scan[ 
			(If[ !basicBlockDefined[ st, #],
				warn[
            	"parent NotFound",
            	#,
            	TemplateApply[
                	StringTemplate[
                    	"BasicBlock `child` is declared a parent of `bb` but is not found in the Function Module",
                    	InsertionFunction -> toString
                	], <|"child" -> #["name"], "bb" -> bb["name"]|>]]];
			If[ !containsId[ #["getChildren"], bb],
				warn[
            	"parent Mismatch",
            	#,
            	TemplateApply[
                	StringTemplate[
                    	"BasicBlock `parent` is a parent of `bb` but does not contain it in its list of children",
                    	InsertionFunction -> toString
                	], <|"parent" -> #["name"], "bb" -> bb["name"]|>]]])&, parents];
		
	]


verifyBasicBlocks[ st_, fm_] :=
	Module[ {},
		fm["topologicalOrderScan",
			(
			(*verifyBasicBlockLive[st, fm,#];*)
			verifyBasicBlockLinks[st, fm,#];
			)&];
	]


(* ::Subsection:: *)
(* Visitors *)	
visitStackAllocateInstruction[st_, inst_] :=
	With[{
		trgt = inst["target"],
		size = inst["size"]
	},
		checkVariableDefinedNonReference[st, size, inst];
		checkVariableUndefined[st, trgt, inst];
		
		(*addVariableToReferenceVariables[st, trgt, inst];*)
		addVariableToDefinedVariables[st, trgt, inst];
		addVariableToKnownVariables[st, trgt, inst];
	]


visitBinaryInstruction[st_, inst_] :=
	With[{
		trgt = inst["definedVariable"],
		ops = inst["usedVariables"]
	},
		checkVariableDefinedNonReference[st, #, inst]& /@ ops;
		checkVariableUndefined[st, trgt, inst];

		addVariableToDefinedVariables[st, trgt, inst];
		addVariableToKnownVariables[st, trgt, inst];
	]
visitBranchInstruction[st_, inst_] :=
	If[inst["isUnconditional"],
		checkBasicBlockDefined[st, inst["getOperand", 1], inst];
		checkBasicBlockChild[st, inst["getOperand", 1], inst];
		, (* Else *)
		With[{
			condVar = inst["condition"],
			thenBBName = inst["getOperand", 1],
			elseBBName = inst["getOperand", 2]
		},
			checkVariableDefinedNonReference[st, condVar, inst];
			checkBasicBlockDefined[st, thenBBName, inst];
			checkBasicBlockChild[st, thenBBName, inst];
			checkBasicBlockDefined[st, elseBBName, inst];
			checkBasicBlockChild[st, elseBBName, inst];
		]
	]
visitCallInstruction[st_, inst_] :=
	With[{
		trgt = inst["definedVariable"],
		ops = inst["usedVariables"]
	},
		checkVariableDefinedNonReference[st, #, inst]& /@ ops;
		checkVariableUndefined[st, trgt, inst];

		addVariableToDefinedVariables[st, trgt, inst];
		addVariableToKnownVariables[st, trgt, inst];
	]
visitTypeCastInstruction[st_, inst_] :=
	With[{
		trgt = inst["definedVariable"],
		src = inst["source"]
	},
		checkVariableDefined[st, src, inst];
		checkVariableUndefined[st, trgt, inst];
		(* We want to propagate the reference property if the RHS 
		 * is a reference
		 *)
		If[st["referenceVariables"]["keyExistsQ", src["id"]],
			addVariableToReferenceVariables[st, trgt, inst];
		];
		addVariableToDefinedVariables[st, trgt, inst];
		addVariableToKnownVariables[st, trgt, inst];
	]
visitCompareInstruction[st_, inst_] :=
	With[{
		trgt = inst["definedVariable"],
		ops = inst["usedVariables"]
	},
		checkVariableDefinedNonReference[st, #, inst]& /@ ops;
		checkVariableUndefined[st, trgt, inst];

		addVariableToDefinedVariables[st, trgt, inst];
		addVariableToKnownVariables[st, trgt, inst];
	]
visitReturnInstruction[st_, inst_] :=
	If[inst["hasValue"],
		checkVariableDefinedNonReference[st, inst["value"], inst]
	]
visitUnaryInstruction[st_, inst_] :=
	With[{
		trgt = inst["definedVariable"],
		op = inst["operand"]
	},
		checkVariableDefinedNonReference[st, op, inst];
		checkVariableUndefined[st, trgt, inst];

		addVariableToDefinedVariables[st, trgt, inst];
		addVariableToKnownVariables[st, trgt, inst];
	]
visitUnreachableInstruction[st_, inst_] :=
	warn[
		"UnreachableInstruction",
		inst,
		TemplateApply[
			StringTemplate[
				"Located an unreachable instruction `inst`."
			],
			<|
				"inst" -> inst["toString"]
			|>
		]
	]
visitSelectInstruction[st_, inst_] :=
	0;
visitLabelInstruction[st_, inst_] :=
	0;
	
visitLambdaInstruction[st_, inst_] :=
	processSourceTarget[st, inst]
	
	
visitLoadInstruction[st_, inst_] :=
	With[{
		trgt = inst["target"],
		src = inst["source"]
	},
		checkVariableDefinedNonReference[st, src, inst];
		checkVariableUndefined[st, trgt, inst];
		
		addVariableToDefinedVariables[st, trgt, inst];
		addVariableToKnownVariables[st, trgt, inst];
	]

(*
  TODO,  switch other instructions eg Copy and Load to this.
*)
processSourceTarget[ st_, inst_] :=
	With[{
		trgt = inst["target"],
		src = inst["source"]
	},
		checkVariableDefinedNonReference[st, src, inst];
		checkVariableUndefined[st, trgt, inst];

		addVariableToDefinedVariables[st, trgt, inst];
		addVariableToKnownVariables[st, trgt, inst];
	]



visitLoadArgumentInstruction[st_, inst_] :=
	With[{
		trgt = inst["target"]
	},
		checkVariableUndefined[st, trgt, inst];
		
		addVariableToDefinedVariables[st, trgt, inst];
		addVariableToKnownVariables[st, trgt, inst];
	];
visitLoadGlobalInstruction[st_, inst_] :=
	With[{
		trgt = inst["target"]
	},
		checkVariableUndefined[st, trgt, inst];
		
		addVariableToDefinedVariables[st, trgt, inst];
		addVariableToKnownVariables[st, trgt, inst];
	];

	
	
visitStoreInstruction[st_, inst_] :=
	With[{
		value = inst["value"],
		trgt = inst["target"]
	},
		checkVariableDefinedNonReference[st, value, inst];
		checkVariableDefinedReference[st, trgt, inst];
	]
	
visitInertInstruction[st_, inst_] :=
	With[{
		trgt = inst["definedVariable"],
		ops = inst["usedVariables"]
	},
		checkVariableDefinedNonReference[st, #, inst]& /@ ops;
		checkVariableUndefined[st, trgt, inst];

		addVariableToDefinedVariables[st, trgt, inst];
		addVariableToKnownVariables[st, trgt, inst];
	]

	
	
visitPhiInstructionTarget[st_, inst_] :=
	With[{
		trgt = inst["target"],
		bbs = inst["getSourceBasicBlocks"],
		ops = inst["getSourceVariables"]
	},
		(* We check for three things
		 * 1. the target variable is not previously defined 
		 * 2. the basic blocks are known
		 * 3. the target variable is not equal to any of the 
		 *    source variables
		 * A second visit of the FM will invoke visitPhiInstructionSourceVariables
		 * which will check the source variables.
		**)
		checkVariableUndefined[st, trgt, inst];
		checkBasicBlockDefined[st, #, inst]& /@ bbs;
		checkBasicBlockParent[st, #, inst]& /@ bbs;
		
		With[{
			trgtName = trgt["id"],
			opNames = (#["id"]& /@ Select[ ops, VariableQ])
		},
			If[MemberQ[opNames, trgtName],
				error[
					"TargetOccursAsPhiVariable",
					inst,
					TemplateApply[
						StringTemplate[
							"The target variable `trgt` of the phi instruction `inst` appears as one of it's source variables `vars`.",
							InsertionFunction -> toString
						],
						<|
							"inst" -> inst,
							"trgt" -> trgt,
							"vars" -> StringRiffle[opNames, {"{", ", ", "}"}]
						|>
					]
				]
			]
		];
		addPhiSource[ st, inst["basicBlock"], bbs];
		addVariableToDefinedVariables[st, trgt, inst];
		addVariableToKnownVariables[st, trgt, inst];
	]
visitPhiInstructionSourceVariables[st_, inst_] :=
	With[{
		ops = inst["getSourceVariables"]
	},
		(* we perform the phi visit after performing a 
		 * pass on all the variables. This is because 
		 * the source variables may not have been added 
		 * to the definedVariables lists during the 
		 * forward pass
		**) 
		
		checkVariableDefinedNonReference[st, #, inst]& /@ ops;
	]
visitRecordExtendInstruction[st_, inst_] :=
	0;
visitRecordRestrictInstruction[st_, inst_] :=
	0;
visitRecordSelectInstruction[st_, inst_] :=
	0;
visitSetElementInstruction[st_, inst_] :=
	0;
	
visitGetElementInstruction[st_, inst_] :=
	Module[ {trgt, src, off},
		trgt = inst["target"];
		src = inst["source"];
		off = inst["offset"];
		If[ Length[off] =!= 1,
			ThrowException[CompilerException[{"GetElement offset incorrect length", Length[off]}]]
		];
		off = First[off];
		checkVariableDefinedNonReference[st, src, inst];
		checkVariableDefinedNonReference[st, off, inst];
		checkVariableUndefined[st, trgt, inst];

		addVariableToDefinedVariables[st, trgt, inst];
		addVariableToKnownVariables[st, trgt, inst];
	]
	
visitSetFieldInstruction[st_, inst_] :=
	0;
visitGetFieldInstruction[st_, inst_] :=
	0;
visitNewRecordInstruction[st_, inst_] :=
	0;
visitCopyInstruction[st_, inst_] :=
	With[{
		trgt = inst["target"],
		src = inst["source"]
	},
		checkVariableDefinedNonReference[st, src, inst];
		checkVariableUndefined[st, trgt, inst];

		addVariableToDefinedVariables[st, trgt, inst];
		addVariableToKnownVariables[st, trgt, inst];
	]

visitInvokeInstruction[st_, inst_] :=
	With[{
		trgt = inst["definedVariable"],
		ops = inst["usedVariables"]
	},
		checkVariableDefinedNonReference[st, #, inst]& /@ ops;
		checkVariableUndefined[st, trgt, inst];

		addVariableToDefinedVariables[st, trgt, inst];
		addVariableToKnownVariables[st, trgt, inst];
	]


visitLandingPadInstruction[st_, inst_] :=
	0;

visitResumeInstruction[st_, inst_] :=
	0;

(* ::Subsection:: *)
(* Defined *)
	
run[fm_, opts_] :=
	Module[{st},
		(*Print["Linting..."];*)
		st = <|
			"phiSource" -> CreateReference[<||>],
			"referenceVariables" -> CreateReference[<||>],
			"definedVariables" -> CreateReference[<||>],
			"variables" -> CreateReference[<||>],
			"basicBlocks" -> CreateReference[<||>],
			"basicBlockParents" -> CreateReference[<||>],
			"basicBlockChildren" -> CreateReference[<||>]
		|>;
		getBasicBlocks[st, fm];
		CreateInstructionVisitor[
			st,
			<|
				"visitStackAllocateInstruction" -> visitStackAllocateInstruction,
				"visitBinaryInstruction" -> visitBinaryInstruction,
				"visitBranchInstruction" -> visitBranchInstruction,
				"visitCallInstruction" -> visitCallInstruction,
				"visitTypeCastInstruction" -> visitTypeCastInstruction,
				"visitCompareInstruction" -> visitCompareInstruction,
				"visitReturnInstruction" -> visitReturnInstruction,
				"visitUnaryInstruction" -> visitUnaryInstruction,
				"visitUnreachableInstruction" -> visitUnreachableInstruction,
				"visitSelectInstruction" -> visitSelectInstruction,
				"visitLabelInstruction" -> visitLabelInstruction,
				"visitLambdaInstruction" -> visitLambdaInstruction,
				"visitLoadInstruction" -> visitLoadInstruction,
				"visitLoadArgumentInstruction" -> visitLoadArgumentInstruction,
				"visitLoadGlobalInstruction" -> visitLoadGlobalInstruction,
				"visitStoreInstruction" -> visitStoreInstruction,
				"visitInertInstruction" -> visitInertInstruction,
				(*
				 * For the phi instruction, we just are interested in linting the 
				 * target variable. We will perform a second iteration to just check
				 * the phis source varaiables
				 *)
				"visitPhiInstruction" -> visitPhiInstructionTarget, 
				"visitRecordExtendInstruction" -> visitRecordExtendInstruction,
				"visitRecordRestrictInstruction" -> visitRecordRestrictInstruction,
				"visitRecordSelectInstruction" -> visitRecordSelectInstruction,
				"visitSetElementInstruction" -> visitSetElementInstruction,
				"visitGetElementInstruction" -> visitGetElementInstruction,
				"visitSetFieldInstruction" -> visitSetFieldInstruction,
				"visitGetFieldInstruction" -> visitGetFieldInstruction,
				"visitNewRecordInstruction" -> visitNewRecordInstruction,
				"visitCopyInstruction" -> visitCopyInstruction,
				"visitInvokeInstruction" -> visitInvokeInstruction,
				"visitLandingPadInstruction" -> visitLandingPadInstruction,
				"visitResumeInstruction" -> visitResumeInstruction
			|>,
			fm,
			"TraversalOrder" -> "reversePostOrder"
		];
		CreateInstructionVisitor[
			st,
			<|
				(* now that we know about all the variables, we can lint the
				 * phi instructions
				 *)
				"visitPhiInstruction" -> visitPhiInstructionSourceVariables
			|>,
			fm,
			"TraversalOrder" -> "reversePostOrder",
			"IgnoreRequiredInstructions" -> True
		];
		verifyBasicBlocks[st, fm];
	]
	
LintIR[pm_?ProgramModuleQ] :=
	pm["scanFunctionModules", run]
LintIR[fm_?FunctionModuleQ] :=
	run[fm, <||>]

(* ::Subsection:: *)
(* Register *)


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"LintIR",
	"Checks for common errors in the SSA IR. " <>
	"If errors or warning do occur, then they are printed. " <>
	"Printout can be suppressed using Quiet."
	,
	"This pass statically checks for common and easily-identified constructs" <>
	"which produce undefined or likely unintended behavior in LLVM IR. Based on Lint.cpp in LLVM"
];

LintIRPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"passClass" -> "Analysis"
|>];

RegisterPass[LintIRPass]
]]



End[]

EndPackage[]
