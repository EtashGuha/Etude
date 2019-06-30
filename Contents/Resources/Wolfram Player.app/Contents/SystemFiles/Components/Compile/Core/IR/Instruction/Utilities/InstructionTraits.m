

BeginPackage["Compile`Core`IR`Instruction`Utilities`InstructionTraits`"]

InstructionTraits

DeserializeInstruction


Begin["`Private`"]


Needs["Compile`Core`IR`BasicBlock`"]
Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionRegistry`"]
Needs["Compile`Utilities`Serialization`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Markup`"] (* For $UseANSI and $UseHTML *)

Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)

sameQ[self_, other_] :=
	ObjectInstanceQ[other] &&
	other["hasField", "_instructionName"] &&
	self["_instructionName"] === other["_instructionName"] &&
	self["id"] === other["id"]

InstructionTraits = ClassTrait[<|
	"initialize" -> Function[{}, 
		Self["setProperties", CreateReference[<||>]];
		Self["set_next", None];
		Self["set_previous", None];
	],
	"serialize" -> Function[{env}, ThrowException[CompilerException[{"Cannot serialize instruction", Self}]]],
	"serializeBase" -> (serializeBase[Self, #]&),
	"serializeBaseNoOperands" -> (serializeBaseNoOperands[Self, #]&),
	"serializeType" -> (serializeType[Self, ##]&),
	"sameQ" -> Function[{other},
		sameQ[Self, other]
	],
	"hasPrevious" -> Function[{}, Self["previous"] =!= None],
	"hasNext" -> Function[{}, Self["next"] =!= None],
	"next" -> Function[{},
		Self["_next"]
	],
	"previous" -> Function[{},
		Self["_previous"]
	],
	"getNext" -> Function[{},
		Self["_next"]
	],
	"getPrevious" -> Function[{},
		Self["_previous"]
	],
	"setNext" -> Function[{val},
		Self["set_next", val];
		val
	],
	"setPrevious" -> Function[{val},
		Self["set_previous", val];
		val
	],
	"setId" -> Function[{val},
		If[Self["basicBlock"] =!= None,
			Self["basicBlock"]["instructionMap"]["keyDropFrom", Self["id"]];
			Self["basicBlock"]["instructionMap"]["associateTo", val -> Self];
		];
		SetData[Self["id"], val];
		Self
	],
	"hasFunction" -> Function[{}, Self["hasField", "function"]],
	"hasOperator" -> Function[{}, Self["hasField", "operator"]],
	"hasTarget" -> Function[{}, Self["hasField", "target"]],
	"hasSource" -> Function[{}, Self["hasField", "source"]],
	"hasOperands" -> Function[{},
		Self["hasMethod", "operands"] ||
		Self["hasField", "operand"] ||
		Self["hasField", "operands"]
	],
	"moveBefore" -> (moveBefore[Self, #]&),
	"moveAfter" -> (moveAfter[Self, #]&),
	"unlink" -> (unlink[Self]&),
	"getMexprId" -> Function[{},
		If[Self["mexpr"] =!= None,
			Self["mexpr"]["id"],
			-1
		]
	],
	"renumber" -> Function[{id},
		Self["setId" -> id];
		Self
	],
	"usedVariables" -> Function[{},
	    ThrowException[{"usedVariables function is not defined for " <> Self["_instructionName"]}]
	],
	"definedVariable" -> Function[{},
	    ThrowException[{"definedVariable function is not defined for " <> Self["_instructionName"]}]
	],
	"hasUse" -> Function[{var},
		VariableQ[var] && AnyTrue[Self["usedVariables"], #["sameQ", var]&]
	],
	"hasDef" -> Function[{var},
		VariableQ[var] && Self["definesVariableQ"] && Self["definedVariable"]["sameQ", var]
	],
	"definesVariableQ" -> Function[{},
		Self["definedVariable"] =!= None
	],
	"replaceOperand" -> (replaceOperand[Self, ##]&),
	"clone" -> (clone[Self, ##]&),
	"visitedQ" -> Function[{}, Self["visited"]],
	"clearVisited" -> Function[{}, Self["setVisited", False]],
	"dispose" -> Function[{}, dispose[Self]],
	"disposeExtra" -> Function[{}, Null],
	"prettyPrint" -> (prettyPrint[Self]&),
    "toHTML" -> (toHTML[Self]&),
	"makePrettyPrintBoxes" -> Function[{}, RowBox[{"Unimplemented makePrettyPrintBoxes ", Self["_class"]}]]
|>]

	
prettyPrint[self_] :=
	CellPrint[
	    Cell[
	        BoxData[
	            self["makePrettyPrintBoxes"]
			],
   			"Text"
	    ]
	]
	
toHTML[self_] :=
    Block[{$FrontEnd = Null, $UseANSI = False, $UseHTML = True},
        self["toString"]
    ]
	
unlink[self_] :=
	Module[{},
		If[self["hasPrevious"],
			self["previous"]["setNext", self["next"]]
		];
		If[self["hasNext"],
			self["next"]["setPrevious", self["previous"]]
		];
		If[BasicBlockQ[self["basicBlock"]],
			self["basicBlock"]["unlinkInstruction", self];
			self["basicBlock"]["instructionMap"]["keyDropFrom", self["id"]];
			
			If[self["basicBlock"]["firstInstruction"] === self,
				Assert[self["hasPrevious"] === False];
				self["basicBlock"]["setFirstInstruction",
					self["next"]
				]
			];
			If[self["basicBlock"]["lastInstruction"] === self,
				Assert[self["hasNext"] === False];
				self["basicBlock"]["setLastInstruction",
					self["previous"]
				]
			]
		];
		self["setNext", None];
		self["setPrevious", None];
		self["setBasicBlock", None];
		self
	]
moveBefore[self_, other_] :=
	With[{prev = other["previous"]},
		self["unlink"];
		If[BasicBlockQ[other["basicBlock"]],
			self["setBasicBlock", other["basicBlock"]];
			other["basicBlock"]["instructionMap"]["associateTo", self["id"] -> self];
			If[other["basicBlock"]["firstInstruction"] === other,
				other["basicBlock"]["setFirstInstruction", self]
			]
		];
		If[prev =!= None,
			prev["setNext", self]
		];
		self["setPrevious", prev];
		self["setNext", other];
		other["setPrevious", self];
		self
	]
moveAfter[self_, other_] :=
	With[{next = other["next"]},
		self["unlink"];
		If[BasicBlockQ[other["basicBlock"]],
			self["setBasicBlock", other["basicBlock"]];
			other["basicBlock"]["instructionMap"]["associateTo", self["id"] -> self];
			If[other["basicBlock"]["lastInstruction"] === other,
				other["basicBlock"]["setLastInstruction", self]
			]
		];
		If[next =!= None,
			next["setPrevious", self]
		];
		self["setNext", next];
		self["setPrevious", other];
		other["setNext", self];
		self
	]


serializeBaseNoOperands[self_, env_] :=
	Module[ {ef = <||>},
		ef["instructionName"] =  self["_instructionName"];
		ef["id"] = self["id"];
		If[self["hasTarget"],
			ef["target"] = self["target"]["serialize", env]];
		ef
	]

serializeBase[self_, env_] :=
	Module[ {ef = serializeBaseNoOperands[self, env]},
		If[self["hasOperands"],
			(*
				TODO: revisit this check for None when LoadArgumentInstruction has been fixed up.
				i.e., evaluate CompileToWIRSerialization[Function[Typed[arg, "Integer64"], arg + 1]] and
				verify that None is no longer passed in. Then remove this check for None.
			*)
			ef["operands"] = Map[ If[#===None, #, #["serialize", env]]&, self["operands"]]];
		ef
	]



serializeType[ self_, env_, type_] :=
	env["serializeType", type]

	

(*
 No need to worry about ID,  they will get fixed up by the TopologicalOrderRenumberPass.
*)
DeserializeInstruction[ env_, "Instruction"[ data_]] :=
	deserialize[env, data, data["instructionName"]]

deserialize[ env_, data_, name_] :=
	Module[ {deserializer, deserializerFun, ins},
		deserializer = Lookup[ $RegisteredInstructions, name, Null];
		If[ deserializer === Null,
			ThrowException[{"Cannot find Deserializer", "Instruction"[ data]}]
		];
		deserializerFun = deserializer["deserialize"];
		If[ deserializerFun === Null,
			ThrowException[{"Cannot find Deserializer", "Instruction"[ data]}]
		];
		ins = deserializerFun[env, data];
		ins["setId", data["id"]];
		ins
	]

clone[ self_, env_] :=
	deserialize[env, self, self["_instructionName"]]

clone[self_] :=
	clone[ self, CreateCloneEnvironment[]]

replaceOperand[self_, oldVar_, newVar_] :=
	Module[{len, test, i},
		If[ !self["hasOperands"],
			(*  Maybe an error *)
			Return[]];
		len = Length[ self["operands"]];
		Do[
			test = self["getOperand", i];
			If[ oldVar["sameQ", test],
				self["setOperand", i, newVar]];
			,
			{i, len}];
	]

dispose[self_] :=
	Module[{},
		If[ TrueQ[self["hasProperty", "disposed", False]],
			Return[]];
		self["properties"]["set", <|"disposed" -> True|>];
		If[self["hasTarget"],
			self["target"]["dispose"]];
		(*
		 Don't visit BasicBlocks here,  it can lead to recursion problems.
		 BasicBlocks are disposed elsewhere.
		*)
		If[self["hasOperands"],
			Map[If[!BasicBlockQ[#], #["dispose"]]&, self["operands"]]];
		If[self["hasField", "condition"] && self["condition"] =!= None,
			self["condition"]["dispose"]];
		Map[#["dispose"]&, self["usedVariables"]];
		self["disposeExtra"];
		self["properties"]["set", <||>];
		self["setBasicBlock", Null];	
		self["set_next", None];
		self["set_previous", None];
	]



End[]
EndPackage[]
