(**
 * This file builds functions which are composed of basic blocks each of 
 * which is composed of a sequence of instructions.
 **) 

BeginPackage["Compile`Core`IR`Lower`Builder`FunctionModuleBuilder`"]


FunctionModuleBuilder;
FunctionModuleBuilderQ;
FunctionModuleBuilderClass;
CreateFunctionModuleBuilder;

Begin["`Private`"] 

Needs["CompileAST`Class`Symbol`"]; (* For MExprSymbolQ *)
Needs["Compile`Core`IR`Instruction`Utilities`InstructionRegistry`"]
Needs["Compile`Core`IR`BasicBlock`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`IR`Lower`Builder`SSABuilder`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Asserter`Assert`"]
Needs["Compile`Core`IR`Lower`Builder`ProgramModuleBuilder`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`Instruction`InstructionQ`"]


(** Starting index of the first basic block *)
$TopBasicBlockId = 1

(** Starting index of the first instruction *)
$TopInstructionId = 1


(**
 * Dispatch trait for all registered instructions producing methods such as
 * "createLoadInstruction", "create...Instruction". An important aspect
 * is that instructions that define a variable must make that association
 * in to the SSABuilder. 
 *
 * If we are in return Mode then create the instruction but don't actually add it.
 *)
makeInstructionDispatch[name_String, constructor0_] :=
	With[{methodName = StringJoin["create", name],
	      constructor = constructor0},
			methodName -> (
				Module[{inst = constructor[##]},
					If[ Self["returnMode"],
						Null,
						addInstruction[Self, inst];
						If[inst["definesVariableQ"] &&
						(* the SetElementInstruction and  SetFieldInstruction behave differently from
						 * other instructions, since they modify the source variable. Maybe we should
						 * change the fact that definesVariableQ returns true for them, since they 
						 * infact do not define anything, but they do write to the source.
						 *)
							!inst["isA", "SetElementInstruction"] &&
							!inst["isA", "SetFieldInstruction"],
					    	writeVariable[Self, inst["basicBlock"], inst["definedVariable"], inst]
						];
						inst
					]
				]&
			)
	]
makeInstructionDispatch[instInfo_] :=
	makeInstructionDispatch[instInfo["name"], instInfo["constructor"]]

RegisterCallback["InstructionDispatchTrait", Function[{st},
InstructionDispatchTrait = ClassTrait[<| makeInstructionDispatch /@ Values[$RegisteredInstructions] |>]
]]

RegisterCallback["DeclareCompileClassPostRegisterInstruction", Function[{st},
FunctionModuleBuilderClass = DeclareClass[
	FunctionModuleBuilder,
	<|
		"initialize" -> Function[{},
			Self["setNextInstructionId", CreateReference[$TopInstructionId]];
			Self["setBasicBlocksMap", CreateReference[<||>]];
			Self["setCurrentBasicBlock", CreateReference[$TopInstructionId]];
			Self["setProperties", CreateReference[<||>]];
		],
		"firstBasicBlock" -> Function[{},
			Self["basicBlocksMap"]["lookup", $TopBasicBlock, None]
		],
		"lastInstruction" -> Function[{},
			If[Self["currentBasicBlock"] === Undefined,
				None,
				Self["currentBasicBlock"]["lastInstruction"]
			]
		],
		"addBasicBlock" -> (addBasicBlock[Self, ##]&),
		"freshBasicBlock" -> (freshBasicBlock[Self, ##]&),
		"createBasicBlock" -> (createBasicBlock[Self, ##]&),

		"getBasicBlocks" -> Function[{},
			Self["basicBlocksMap"]["get"]
		],
		"getFunctionModule" -> Function[{},
			Module[{fm},
				fm = CreateFunctionModule[
					Self["programModule"],
					Self["name"],
					Self["firstBasicBlock"],
					Self["lastBasicBlock"],
					Self["mexpr"],
					Self["result"],
					Self["arguments"]
				];
				If[ Self["type"] =!= Undefined,
					fm["setType", Self["type"]]
				];
				fm["setMetaData",Self["getMetaData"]];
				fm["information"]["addMetaData", fm["metaData"]];
				fm["setProperty", "localFunction" -> Self["getProperty", "localFunction", False]];
				fm["setProperty", "entryQ" -> Self["getProperty", "entryQ", False]];
                fm["setProperty", "exported" -> Self["getProperty", "exported", False]];
				If[Self["hasProperty", "closureVariablesConsumed"],
				    fm["setProperty", "closureVariablesConsumed" -> Self["getProperty", "closureVariablesConsumed"]]
				];
				fm["setBodyType", Self["bodyType"]];
				fm["setTypeEnvironment", Self["typeEnvironment"]];
				fm
			]
		],
		"sealBasicBlock" -> (sealBasicBlock[Self, ##]&),
		"writeVariable" -> (writeVariable[Self, ##]&),
		"readVariable" -> (readVariable[Self, ##]&),
		"addReturn" -> Function[ {val, bb}, addReturn[Self, val, bb]],
		"finish" -> Function[ {builder, res}, finish[Self, builder, res]],
		"addArgument" -> Function[{arg},
		    Self["setArguments", Append[Self["arguments"], arg]];
		    Self["arguments"]
		],
		"dispose" -> Function[{}, dispose[Self]],
		"toString" -> Function[{},
			Self["getFunctionModule"]["toString"]
		],
		"toBoxes" -> Function[{fmt},
			toBoxes[Self, fmt]
		]
	|>,
	{
		"id" -> 0,
		"mexpr",
		"name",
		"programModule",
		"nextInstructionId",
		"basicBlocksMap",
		"currentBasicBlock" -> None,
		"lastBasicBlock",
		"arguments" -> {},
		"result",
		"properties",
		"variableBuilder",
		"typeEnvironment",
		"llvmDebug",
		"bodyType" -> Null,
		"returnData",
		"returnMode" -> False,
		"type" -> Undefined,
		"metaData" -> Null
	},
	Predicate -> FunctionModuleBuilderQ,
	Extends -> {InstructionDispatchTrait, ClassPropertiesTrait}
]
]]

Options[CreateFunctionModuleBuilder] = {
	"LLVMDebug" -> Automatic
}

CreateFunctionModuleBuilder[pmb_?ProgramModuleBuilderQ, id_, name_, mexpr_, opts:OptionsPattern[]] :=
	Module[{fmb, firstBB, lastBB, varBuilder},
	    FunctionModuleBuilderClass; (* We need to load the class *)
		fmb = CreateObject[FunctionModuleBuilder, <|
			"mexpr" -> mexpr,
			"id" -> id,
			"name" -> name,
			"programModule" -> pmb,
			"returnData" -> CreateReference[{}],
			"llvmDebug" -> TrueQ[Lookup[<|opts|>, "LLVMDebug", Automatic]]
		|>];
		varBuilder = CreateSSABuilder[];
		fmb["setVariableBuilder", varBuilder];
		(* we create the first basic block and add it to the function module.
		 * this garantees that the function module is never empty.
		 *)
		firstBB = fmb["createBasicBlock", $TopBasicBlock, mexpr];
		fmb["addBasicBlock", firstBB];
		
		lastBB = fmb["createBasicBlock", $LastBasicBlock, mexpr];
		fmb["addBasicBlock", lastBB];
		fmb["setLastBasicBlock", lastBB];
		
		fmb["setCurrentBasicBlock", firstBB];
		fmb["sealBasicBlock", firstBB];
		fmb
	]
CreateFunctionModuleBuilder[args___] :=
	ThrowException[{"Invalid arguments when CreateFunctionModuleBuilder ", {args}}]
	
	

	

(** Creates and adds a basic block. It is up to the user
  * to connect the basic block
  *)
addBasicBlock[builder_, name_, mexpr_:None] :=
	With[{bb = createBasicBlock[builder, name, mexpr]},
		addBasicBlock[builder, bb]
	]
(** Adds a the basic block to the function module. It is up to the user
  * to connect the basic block
  *)
addBasicBlock[builder_, bb_?BasicBlockQ] :=
	Module[{name = bb["name"], varBuilder},
		If[
			builder["basicBlocksMap"]["keyExistsQ", name],
			name = name <> ToString[bb["id"]];
			If[builder["basicBlocksMap"]["keyExistsQ", name],
				ThrowException[{"Duplicate basic block name " <> name <> " in function module"}]
			];
		];
		builder["setCurrentBasicBlock", bb];
		builder["basicBlocksMap"]["associateTo", name -> bb];
		
		(** We now add the basic block to the SSA Builder *)
		varBuilder = builder["variableBuilder"];
		varBuilder["addBasicBlock", builder, bb];
		
		bb
	]
	

(** Creates a new empty basic block
  * the differnce between it and createBasicBlock is that the
  * latter adds a label statement
  * @{Return} BasicBlock
  *)
freshBasicBlock[builder_, name_, mexpr_:None] :=
	Module[{bb},
		bb = CreateBasicBlock[name];
		bb["initId"];
		bb["setMexpr", mexpr];
		bb
	]

(** Create a new basic block with specified name.
 *  A label instruction is placed in the basic block.
 *  This does not add basic block to the function module.
 *  It is up to the user to add it by using builder["addBasicBloc"]
 *  @{Return} BasicBlock
 *) 
createBasicBlock[builder_, name_, mexpr_:None] :=
	Module[{ref},
		ref = freshBasicBlock[builder, name, mexpr];
		builder["setCurrentBasicBlock", ref];
		(* this is a label instruction which 
		   would be a no-op in most cases *)
		builder["createLabelInstruction", name, mexpr];
		ref
	]
createBasicBlock[args___] :=
	ThrowException[{"Invalid arguments for createBasicBlock ", args}]

(** Add an instruction in the current basic basic block *)
addInstruction[builder_, inst_] :=
	Module[{bb},
		AssertThat["The argument addInstruction should be an instruction",
			inst]["named", "instruction"]["satisfies", InstructionQ];
		bb = builder["currentBasicBlock"];
		inst["setId", builder["nextInstructionId"]["increment"]];
		bb["addInstruction", inst];
		inst
	]

(*************************************)


sealBasicBlock[self_] :=
	sealBasicBlock[self, self["currentBasicBlock"]]
sealBasicBlock[self_, bb_] :=
	With[{varBuilder = self["variableBuilder"]},
		varBuilder["sealBasicBlock", self, bb]
	]
sealBasicBlock[self_, args___] :=
	ThrowException[{"Undefined usage of sealBasicBlock with ", args}]
	
writeVariable[self_, var_, val_] := (
	(* Print["Wrote variable ", var["toString"], " for ", val["toString"]]; *)
	writeVariable[self, self["currentBasicBlock"], var, val]
)

writeVariable[self_, bb_, var_, val_] :=
	With[{varBuilder = self["variableBuilder"]},
		varBuilder["writeVariable", self, bb, var, val]
	]
	
writeVariable[self_, args___] :=
	ThrowException[{"Undefined usage of writeVariable with ", args}]
	
ClearAll[readVariable]

readVariable[self_, name_?StringQ, mexpr_?MExprSymbolQ] :=
	Module[{varBuilder = self["variableBuilder"]},
		varBuilder["readVariable", self, self["currentBasicBlock"], name, mexpr]
	];
readVariable[self_, args___] :=
	ThrowException[{"Undefined usage of readVariable with ", args}]

(*
  Add value/basic block to the return Data,  this is used when 
  a Return is found.
*)
addReturn[ self_, val_, bb_] :=
	self["returnData"]["appendTo", {val, bb}]
	
	


finish[ self_, state_, res_] :=
	Module[{builder = state["builder"], lastBB, retData, retVar, mexpr},

		If[res === Null,
			mexpr = Null,
			mexpr = res["mexpr"]
		];

		If[ self["returnMode"],
			self["setReturnMode", False];
			,
			self["addReturn", builder["currentBasicBlock"], res]
		];
		lastBB = self["lastBasicBlock"];
		builder["createBranchInstruction", lastBB];
		builder["currentBasicBlock"]["addChild", lastBB];
		builder["setCurrentBasicBlock", lastBB];
		(*
		  Not really necessary to seal,  but good for completeness.
		*)
		builder["sealBasicBlock", lastBB];
		retData = self["returnData"]["get"];
		If[ Length[retData] === 0,
			ThrowException[{"Cannot find return data"}]
		];
		If[ Length[retData] === 1,
			retVar = Part[retData,1,2];,
			retVar = state["createFreshVariable", mexpr];
			builder["createPhiInstruction", retVar, retData, mexpr]
		];
		builder["createReturnInstruction", retVar, mexpr]
	]



dispose[self_] :=
	Module[{},
		self["variableBuilder"]["dispose"];
		self["setProgramModule", Null];
		self["setProperties", Null];
	]

(*************************************)

icon := Graphics[Text[
  Style["FMB", GrayLevel[0.7], Bold, 
   CurrentValue["FontCapHeight"]/
     AbsoluteCurrentValue[Magnification]]], $FormatingGraphicsOptions];   
     
toBoxes[builder_, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		"FunctionModuleBuilder",
		builder,
  		icon,
  		{
  			BoxForm`SummaryItem[{"ssa: ", builder["variableBuilder"]}],
  			BoxForm`SummaryItem[{"mexpr: ", builder["mexpr"]}]
  		},
		{}, 
  		fmt,
		"Interpretable" -> False
  	]
	
End[]
EndPackage[]
