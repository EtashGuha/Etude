BeginPackage["Compile`Core`IR`BasicBlock`"]

(**
a sequence of statements with one program point of entry (at the start of the block) and one
point of exit (at the end of the block) ... i.e. there is no side exists. More formally,
a sequence of statements Subscript[s, 0],Subscript[s, 1],\[Ellipsis],Subscript[s, n] forms a
basic block iff Subscript[s, i] dominates Subscript[s, j] if i>j and Subscript[s, i] is not a
jump when i<n.
*)
BasicBlock
BasicBlockClass
CreateBasicBlock;
BasicBlockQ
DeserializeBasicBlock
Compile`Debug`$ShowProperties

Begin["`Private`"] 

Needs["Compile`Core`IR`GetElementTrait`"]
Needs["Compile`Utilities`Serialization`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Asserter`Assert`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`Instruction`InstructionQ`"]
Needs["Compile`Core`IR`Instruction`BranchInstruction`"]
Needs["Compile`Core`IR`Instruction`CopyInstruction`"]
Needs["Compile`Core`IR`Instruction`LabelInstruction`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`IR`Instruction`PhiInstruction`"]
Needs["CompileUtilities`Markup`"]


If[!ValueQ[$NextBasicBlockId],
	$NextBasicBlockId = 1
]

getBasicBlockId[] :=
	$NextBasicBlockId++ 


(*
  TODO
    Think about unlink,  is this really necessary.  It is probably wrong since it removes it 
    from the FM BB list but doesn't put it back.
*)

RegisterCallback["DeclareCompileClass", Function[{st},
BasicBlockClass = DeclareClass[
	BasicBlock,
	<|
		"clone" -> (clone[Self,##]&),
		"sameQ" -> Function[{other},
			BasicBlockQ[other] && Self["id"] === other["id"]
		],
		"initId" -> Function[{},
			Self["setId", getBasicBlockId[]]
		],
		"setId" -> Function[{val},
			If[Self["functionModule"] =!= None,
				Self["functionModule"]["basicBlockMap"]["keyDropFrom", Self["id"]];
				Self["functionModule"]["basicBlockMap"]["associateTo", val -> Self];
			];
			SetData[Self["id"], val];
			Self
		],
		"getTerminator" -> Function[{},
			With[{inst = Self["lastInstruction"]},
				AssertThat["The instruction is a branch or return instruction", inst
					]["satisfies", (TrueQ[#["isA", "BranchInstruction"]] || TrueQ[#["isA", "ReturnInstruction"]])&
				];
				inst
			]
		],
		"hasInstruction" -> Function[{id}, MemberQ[Self["instructionIds"], id]],
		"firstNonLabelInstruction" -> Function[{},
			Module[{inst = Self["firstInstruction"]},
				If[LabelInstructionQ[Self["firstInstruction"]],
				    inst["next"],
				    Self["firstInstruction"]
				]
			]
		],
		"lastPhiInstruction" -> Function[{},
			Module[{inst = Self["firstInstruction"]},
				While[PhiInstructionQ[inst["next"]],
					inst = inst["next"]];
					inst
			]
		],
		"instructionIds" -> Function[{}, #["id"]& /@ Self["getInstructions"]],
		"isEmpty" -> Function[{},
			With[{nInsts = Length[Self["getInstructions"]]},
				nInsts === 0 || (nInsts == 1 && LabelInstructionQ[Self["firstInstruction"]])
			]
		],
		
		"appendParentsList" -> Function[{parent},
			If[Self["parents"]["freeQ", parent],
				Self["parents"]["pushBack", parent]
			]
		],
		"appendChildrenList" -> Function[{child},
			If[Self["children"]["freeQ", child],
				Self["children"]["pushBack", child]
			]
		],
		"addParent" -> Function[{parent},
			parent["appendChildrenList", Self];
			Self["appendParentsList", parent]
		],
		"addChild" -> Function[{child},
			child["appendParentsList", Self];
			Self["appendChildrenList", child]
		],
		
		"removeParent" -> Function[{parent}, 
			Self["parents"]["deleteCases", parent];
			parent["children"]["deleteCases", Self];
		], 
		"removeChild" -> Function[{child}, 
			Self["children"]["deleteCases", child];
			child["parents"]["deleteCases", Self];
		],
		"remove" -> (remove[Self]&),
		"joinParentList" -> Function[{parents},
			Self["parents"]["join", parents]
		],
		"joinChildList" -> Function[{parents},
			Self["children"]["join", parents]
		],
		"getParents" -> Function[{},
			Self["parents"]["toList"]
		],
		"getChildren" -> Function[{},
			Self["children"]["toList"]
		],
		"getParentsIds" -> Function[{},
			#["id"]& /@ Self["getParents"]
		],
		"getChildrenIds" -> Function[{},
			#["id"]& /@ Self["getChildren"]
		],
		"hasParents" -> Function[{},
			!Self["parents"]["isEmpty"]
		],
		"hasChildren" -> Function[{},
			!Self["children"]["isEmpty"]
		],
		
		"moveBefore" -> Function[{other},
			Self["unlink"];
			Scan[
				(
					#["addChild", Self];
					#["removeChild", other]
				)&,
				other["getParents"]
			];
			Self["addChild", other];
			Self
		],
		"moveAfter" -> Function[{other},
			Self["unlink"];
			Scan[
				(
					#["addParent", Self];
					#["removeParent", other];
				)&,
				other["getChildren"]
			];
			Self["addParent", other];
			Self
		],
		"unlink" -> Function[{},
			Module[{children, parents},
				If[FunctionModuleQ[Self["functionModule"]],
					Self["functionModule"]["unlinkBasicBlock", Self]];
				children = Self["getChildren"];
				parents = Self["getParents"];
				Scan[
					Self["removeParent", #]&,
					parents
				];
				Scan[
					Self["removeChild", #]&,
					children
				];
				Self["parents"]["clear"];
				Self["children"]["clear"]
			]
		],
		
		"prependInstruction" -> Function[{inst},
			With[{fst = Self["firstInstruction"]},
				Self["addInstructionAfter", fst, inst];
			]
		],
        "addInstructions" -> Function[{insts},
            Self["addInstruction", #]& /@ insts;
            Last[insts]
        ],
		"addInstruction" -> Function[{inst},
			inst["setBasicBlock", Self];
			If[Self["firstInstruction"] === None,
				Self["setFirstInstruction", inst]
				, (* Else *)
				Self["lastInstruction"]["setNext", inst];
				inst["setPrevious", Self["lastInstruction"]]
			];
			Self["setLastInstruction", inst];
			Self["instructionMap"]["associateTo", inst["id"] -> inst];
			inst
		],
		"addInstructionBefore" -> (addInstructionBefore[Self, ##]&),
		"addInstructionAfter" -> Function[{inst, instToInsert},
			If[inst["sameQ", instToInsert],
				Return[]
			];
			If[BasicBlockQ[instToInsert["basicBlock"]],
				With[{bb = instToInsert["basicBlock"]},
					bb["removeInstruction", instToInsert]
				]
			];
			instToInsert["moveAfter", inst];
			instToInsert
		],
		"removeInstruction" -> Function[{inst},
			If[inst === Self["firstInstruction"],
				Self["setFirstInstruction", inst["next"]]
			];
			If[inst === Self["lastInstruction"],
				Self["setLastInstruction", inst["previous"]]
			];
			Self["instructionMap"]["keyDropFrom", inst["id"]];
			inst["unlink"]
		],
		(** We get the instruction list so that we can modify the 
		  * instructions within the scan loop
		  *)  
		"scanInstructions" -> Function[{fun}, 
			Scan[fun, Self["getInstructions"]];
			Self
		],
		"reverseScanInstructions" -> Function[{fun},
			Scan[fun, Self["getReversedInstructions"]];
			Self
		],
		"getInstructions" -> Function[{},
			Module[{insts = Internal`Bag[], inst},
				inst = Self["firstInstruction"];
				While[inst =!= None,
					Internal`StuffBag[insts, inst];
					inst = inst["next"];
				];
				Internal`BagPart[insts, All]
			]
		],
		"getReversedInstructions" -> Function[{},
			Reverse[Self["getInstructions"]]
		],
		"getInstruction" -> Function[{id},
			Self["instructionMap"]["lookup", id, $Failed]
		],
		"renumberInstructions" -> Function[{id0},
			Module[{id = id0},
			    Self["scanInstructions", #["renumber", id++]&];
				Self["instructionMap"]["set",
					Association[#["id"] -> #& /@ Self["instructionMap"]["values"]]
				];
			    Self
			]
		],
		
		"getElements" -> Function[ {},  Self["getInstructions"]],

		"visitedQ" -> Function[{}, Self["visited"]],
		"clearVisited" -> Function[{},
			Self["scanInstructions", #["clearVisited"]&];
			Self["setVisited", False]
		],
		"makePrettyPrintBoxes" -> Function[{},
			makePrettyPrintBoxes[Self]
		],
		"fullName" -> Function[{},
			fullName[Self]
		],
		"addUse" -> Function[{inst},
			AssertThat["The basic block use must be an instruction.",
				inst]["named", "Instruction"]["satisfies", InstructionQ];
			
			If[!MemberQ[#["id"]& /@ Self["uses"], inst["id"]],
                Self["setUses", Append[Self["uses"], inst]]
            ];
			Self
		],
		"clearUses" -> Function[{},
			Self["setUses", {}];
			Self
		],
		"exprTypeQ" ->  Function[{},
			Self["functionModule"]["exprTypeQ"]
		],
		"splitAfter" -> Function[{inst}, splitAfter[Self, inst]],
		(*
		 TODO,  rename to be called switchPhiInstruction
		*)
		"fixPhiInstructions" -> Function[{oldBB, newBB}, fixPhiInstructions[Self, oldBB, newBB]],
		"rewritePhiInstructions" -> (rewritePhiInstructions[Self]&),
		"serialize" -> (serialize[ Self, #]&),
		"fixLinks" -> (fixLinks[ Self, #]&),
		"fixNameId" -> (fixNameId[ Self, #]&),
		"unlinkInstruction" -> Function[{inst}, unlinkInstruction[Self, inst]],	
		"dispose" -> Function[{}, dispose[Self]],
		"toString" -> (toString[Self]&), 
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"initialize" -> Function[{},
			Self["setParents", CreateReference[{}]];
			Self["setProperties", CreateReference[<||>]];
			Self["setChildren", CreateReference[{}]];
			Self["setInstructionMap", CreateReference[<||>]];
		]
	|>,
	{
		"id" -> 0,
		"mexpr",
		"name" -> "unknown",
		"functionModule" -> None,
		"firstInstruction" -> None,
		"lastInstruction" -> None,
		"instructionMap",
		"unlinkedInstructions",
		"parents",
		"children",
		"dominator" -> None,
		"immediateDominator" -> None,
		"visited" -> False,
		"uses" -> {} , (**< Instruction ids that use this basic block *)
		"properties"
	},
	Predicate -> BasicBlockQ,
	Extends -> {
		ClassPropertiesTrait,
		GetElementTrait
	}
];
]]


addInstructionBefore[self_, inst_, instToInsert_] := Module[{},
	If[inst["sameQ", instToInsert],
		Return[]
	];
	If[BasicBlockQ[instToInsert["basicBlock"]],
		With[{bb = instToInsert["basicBlock"]},
			bb["removeInstruction", instToInsert]
		]
	];
	instToInsert["moveBefore", inst];
	instToInsert
];
	
CreateBasicBlock[args___] :=
	createBasicBlock[args];
	
createBasicBlock[] :=
	CreateObject[BasicBlock, <|"unlinkedInstructions" -> CreateReference[{}]|>]

createBasicBlock[name_] :=
	CreateObject[BasicBlock, <| "name" -> name, "unlinkedInstructions" -> CreateReference[{}] |>]

fullName[self_] :=
	self["name"] <> "(" <> ToString[self["id"]] <> ")"



fixLinks[self_, env_] :=
	Module[ {bb,list},
		list =
			Map[
				(
				bb = env["getBasicBlock", #];
				If[ !BasicBlockQ[bb],
					ThrowException[CompilerException[{"Cannot find BasicBlock ", #}]]
				];
				bb
				)&, self["children"]["get"]];
		self["children"]["set", list];
		list =
			Map[
				(
				bb = env["getBasicBlock", #];
				If[ !BasicBlockQ[bb],
					ThrowException[CompilerException[{"Cannot find BasicBlock ", #}]]
				];
				bb
				)&, self["parents"]["get"]];
		self["parents"]["set", list];
	]
	
DeserializeBasicBlock[ env_, "BasicBlock"[ data_]] :=
	deserialize[ env, data, data["instructions"]]

deserialize[ env_, data_, insts_] :=
	Module[ {bb, id, name, newId, children, parents},
		id = data["id"];
		newId = id;
		name = data["name"];
		If[ env["uniqueID"],
			newId = getBasicBlockId[];
			name = name <> ToString[newId]];
		bb = CreateBasicBlock[name];
		bb["setId", newId];
		Map[ bb["addInstruction",env["deserialize", #]]&, insts];
		If[ env["isClone"],
			children = Map[ "BasicBlockID"[#1["id"]]&, data["children"]["get"]];
			parents =  Map[ "BasicBlockID"[#1["id"]]&, data["parents"]["get"]];
			,
			children = env["getElementNoDeserialize", data, "children"];
			parents = env["getElementNoDeserialize", data, "parents"]];
		bb["children"]["set", children];
		bb["parents"]["set", parents];
		env["setBasicBlock", id, bb];
		bb
	]
	


serialize[ self_, env_] :=
	Module[ {data},
		data = <|"name" -> self["name"], 
				  "id" -> self["id"],
				  "instructions" -> Map[ #["serialize", env]&, self["getInstructions"]],
				  "parents" -> Map[ "BasicBlockID"[#["id"]]&, self["parents"]["get"]],
				  "children" -> Map[ "BasicBlockID"[#["id"]]&, self["children"]["get"]]
				  |>;
		"BasicBlock"[ data]
	]
	

clone[self_, env_] :=
	deserialize[ env, self, self["getInstructions"]]
	
clone[self_] :=
	clone[ self, CreateCloneEnvironment[]]


unlinkInstruction[ self_, inst_] :=
	self["unlinkedInstructions"]["appendTo", inst]

dispose[self_] :=
	Module[{},
		If[ TrueQ[self["hasProperty", "disposed", False]],
			Return[]];
		self["properties"]["set", <|"disposed" -> True|>];
		self["scanInstructions", #["dispose"]&];
		Scan[ #["dispose"]&, self["unlinkedInstructions"]["get"]];
		self["setDominator",Null];
		self["setImmediateDominator",Null];
		self["setUses", {}];
		self["parents"]["clear"];
		self["children"]["clear"];
		self["setFunctionModule", Null];
	]


(*
  Splits the basicBlock after the current instruction introducing a new basic 
  basic block and joining the two with an unconditional branch.
  
  For now can only do this for CallInstruction,  but it could be any instruction that does
  not start or end a basic block.
  
  TODO,  need to create the new Id properly.
  
  Create the new Basic Block 
  Create a label Instruction
  Go through all later instructions adding them to the new BB.
  Add a branch instruction to the current BB to branch to the new BB
  Move the children of the old BB to be children of the new BB
  Fix any Phi instructions of the children to be Phi of the new BB
  Add the new BB to the Function Module.
  Return the new BB.
*)

splitAfter[bb_, inst_?InstructionQ] :=
	Module[ {name, newBB, newId, nextInst, workInst, brInst, children},
(*		If[ !(CallInstructionQ[inst] || CompareInstructionQ[inst] || BinaryInstructionQ[inst] || UnaryInstructionQ[inst] ),
			ThrowException[CompilerException[{"Can only split basic blocks after a CallInstruction ", inst}]]
		];	
*)		(* create BB *)
		newId = getBasicBlockId[];
		name = "SplitBasicBlock" <> ToString[newId];
		newBB = CreateBasicBlock["SplitBasicBlock" <> ToString[newId]];
		newBB["setId", newId];
		(* create Label *)
		nextInst = CreateLabelInstruction[ "Split" <> ToString[newId]];
		newBB[ "addInstruction", nextInst];
		(* move instructions *)
		nextInst = inst["next"];
		Assert[InstructionQ[nextInst] || inst === None];
		While[ nextInst =!= None,
			workInst = nextInst;
			nextInst = nextInst["next"];
			workInst["unlink"];
			newBB[ "addInstruction", workInst]
		];
		(* Add branch inst *)
		brInst = CreateBranchInstruction[{newBB}];
		bb["addInstruction", brInst];
		(* fix child BBs *)
		children = bb["getChildren"];
		Map[ newBB["addChild", #]&, children];
		(* fix Phi Instructions in children *)
		Map[ #["fixPhiInstructions", bb, newBB]&, children];
		Scan[bb["removeChild", #]&, children];
		bb["addChild", newBB];
		bb["functionModule"]["linkBasicBlock", newBB];
		bb["functionModule"]["checkLast", newBB, bb];
		newBB
	]
splitAfter[bb_, args___] :=
	ThrowException["Bad args to  splitAfter: ", {args}]

(*
  Look at Phi Instructions in self.  Any that have oldBB as a source should 
  be switched to new.
*)
fixPhiInstructions[self_, oldBB_, newBB_] :=
	self["scanInstructions",
		Function[{inst}, If[ inst["_instructionName"] === "PhiInstruction", fixPhi[ inst, oldBB, newBB]]]]

	
fixPhi[ inst_, oldBB_, newBB_] :=
	Module[{srcBBs},
		srcBBs = inst["getSourceBasicBlocks"];
		If[ AnyTrue[ srcBBs, #["id"] === oldBB["id"]&],
			srcBBs = Map[ If[ #["id"] === oldBB["id"], newBB, #]&, srcBBs];
			inst["setSourceBasicBlocks", srcBBs]];
	]



(*
 fix any PhiInstructions that reference this BB
*)
rewritePhiInstructions[ bb_] :=
	Scan[ rewritePhi[bb, #]&, bb["uses"]]

(*
  Get the BBs for this instruction and drop this BB.  If we end up 
  with only one BB then turn the Phi into a Copy.
*)
rewritePhi[bb_, inst_?PhiInstructionQ] :=
	Module[ {src, newList, newInst},
		src = inst["source"];
		newList = src["select",  (Part[#,1]["id"] =!= bb["id"])&];
		If[ Length[newList] =!= 1,
			src["set", newList]
			,
			newInst = CreateCopyInstruction[ inst["target"], newList[[1,2]], inst["mexpr"]];
			newInst["moveAfter", inst];
			newInst["setId", inst["id"]];
			inst["unlink"];
		]
	]

(*
Catch other instructions such as BranchInstruction
*)
rewritePhi[bb_, _] :=
	Null

(*
 Remove this BasicBlock
 
 Remove parent from this BB
 Look at each child,  if they only have one parent which is this BB then remove them, else just remove.
 Fix up any PhiInstructions that reference this BB
 remove this BB from the FMs list of BasicBlocks
 
 TODO, this is very similar to unlink.
*)
remove[self_] :=
	Module[ {parents, children},
		parents = self["getParents"];
		Scan[ self["removeParent", #]&, parents];
		children = self["getChildren"];
		Scan[ self["removeChild", #]&, children];
		rewritePhiInstructions[self];
		self["functionModule"]["unlinkBasicBlock", self];
	]


fixNameId[self_, baseName_] :=
	Module[ {newId, name},
		newId = getBasicBlockId[];
		name = baseName <> ToString[newId];
		self["setName", name];
		self["setId", newId];
	]
	
(**************************************************)
(**************************************************)
(**************************************************)

makePrettyPrintBoxesInstruction[inst_] :=
	If[LabelInstructionQ[inst],
		Nothing,
		{
		    If[inst["hasProperty", "live[in]"],
		    		TooltipBox[
		    			StyleBox[inst["id"], Gray],
		    			GridBox[{
						{"live[in] = ", StringRiffle[#["toString"]& /@ inst["getProperty", "live[in]"], ", "]},
						{"live[out] = ", StringRiffle[#["toString"]& /@ inst["getProperty", "live[out]"], ", "]}
					}]
		    		],
		    		StyleBox[inst["id"], Gray]
		    ],
		    inst["makePrettyPrintBoxes"]
		}
	]
makePrettyPrintBoxes[self_] :=
	Module[{},
		With[{insts = self["getInstructions"]},
			FrameBox[
				GridBox[{
					{
						StyleBox[self["fullName"] <> ":", Bold, "SubSection"]
					},
					{
					    GridBox[
					    		makePrettyPrintBoxesInstruction /@ insts,
					    		GridBoxDividers->{"Columns" -> {False, True}},
					    		GridBoxAlignment->{"Columns" -> {{Left}}},
					    		GridBoxItemSize->{"Columns" -> {{Automatic}}, "Rows" -> {{Automatic}}}
						]
					}
				}]
			]
		]
	]


(**************************************************)
(**************************************************)
(**************************************************)

instLabel[ inst_, maxLen_] :=
	InstructionNumberText[StringPadLeft[ToString[inst["id"]], maxLen]]

instLabel[ inst_, 0] :=
	""

(*
 Formatting tools for the property printing functionality.
*)
formatPropertyValue[val_Compile`Utilities`Class`Impl`ObjectInstance] :=
	formatPropertyValue[val["toString"]]

formatPropertyValue[val_] :=
	formatPropertyValue[ToString[val]]
	
formatPropertyValue[val_String] :=
	BoldGrayText[val]

formatPropertyValue[val_List] :=
	{GrayText["{"], Riffle[ Map[formatPropertyValue, val], GrayText[", "]], GrayText["}"]}

getPropertyValue[self_, prop_] :=
	Module[{val = self["getProperty", prop]},
		If[MissingQ[val],
			Return[{}]
		];
		{PropertyText[prop], GrayText[" = "], formatPropertyValue[val]}
	]

getProperties[self_] :=
	Module[{props = Compile`Debug`$ShowProperties, vals},
		If[StringQ[props],
			props = {props}
		];
		If[ !MatchQ[props, {__String}],
			Return[{}]
		];
		vals = Riffle[ Map[getPropertyValue[self,#]&, props], GrayText[", "]]
	]

toString[self_] :=
	Module[{insts, maxLen, props=getProperties[self]},
		insts = self["getInstructions"];
		maxLen = Max[IntegerLength[#["id"]]&/@insts];
		StringJoin[
			{ (** We will print the bb["name"] <> (bb["id"]) in place of the label instruction *)
				LabelText[self["fullName"]],
				":  ",
				If[props === {},
                    "",
                    {GrayText["(* "], props, GrayText[" *)"]}
                ],
				"\n"
			},
			Riffle[
				Map[
					Function[{inst},
					    With[{instProps = getProperties[inst]},
							If[LabelInstructionQ[inst],
								Nothing,
								{						
									instLabel[inst, maxLen], VeryLightGrayText[" |"],  "\t", inst["toString"]
									,
									" "
									,
									If[instProps === {},
									    "",
									    {GrayText["(* "], instProps, GrayText[" *)"]}
									]
								}
							]
					    ]
					],
					insts
				],
				"\n"
			],
			"\n"
		]
	]
(*
icon := Graphics[{FaceForm[], 
  Rectangle[], {FaceForm[GrayLevel[.7]], EdgeForm[], 
   Table[Rectangle[{0.05, y + 0.05}, {0.9, y + 0.15}], {y, 0, 0.9, 
     0.2}]}}, commonGraphicsOptions]
*)
icon := Graphics[Text[
  Style["BB", GrayLevel[0.7], Bold, 
   1.2*CurrentValue["FontCapHeight"]/
     AbsoluteCurrentValue[Magnification]]], $FormatingGraphicsOptions];   



toBoxes[bb_?BasicBlockQ, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		"BasicBlock",
		bb,
  		icon,
  		{
  		    BoxForm`SummaryItem[{"id: ", bb["id"]}],
  		    BoxForm`SummaryItem[{"name: ", bb["name"]}]
  		},
  		Map[BoxForm`SummaryItem[{LazyFormat[InstructionQ, #]}] &, bb["getInstructions"]], 
  		fmt
  	]

End[]
EndPackage[]
