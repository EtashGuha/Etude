BeginPackage["Compile`TypeSystem`Inference`InferencePass`"]

InferencePass
InferTypeInstruction


Begin["`Private`"]


Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`ProgramModulePass`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`ProgramModule`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`TypeSystem`Inference`InferencePassState`"]
Needs["Compile`TypeSystem`Inference`InferencePassPatternState`"]
Needs["TypeFramework`"]
Needs["TypeFramework`TypeObjects`TypeVariable`"]
Needs["TypeFramework`TypeObjects`TypeArrow`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`Transform`ResolveTypes`"]
Needs["Compile`Core`Transform`ResolveSizeOf`"]
Needs["Compile`Core`IR`Instruction`CallInstruction`"]

(*
  Infer the type for inst,  to get the function marked with the type object 
  (needed for implementation metadata).
*)
InferTypeInstruction[ pm_, inst_?CallInstructionQ] :=
	Module[ {inferSt, state},
		inferSt = CreateInferencePassState[pm, Null];
		state = <|"inferenceState" -> inferSt|>;
		baseUpFunctionCall[ state, inst["arguments"], inst["target"], inst["function"], inst];
		CatchException[
			inferSt["solve"];
			True
			,
			{{_, False&}}
		]
	]


upCallInstruction[st_, inst_] :=
	Module[{func = inst["function"], name, fm, ty, inferSt},
		If[!ConstantValueQ[func],
			Return[
				upFunctionCall[st, inst, func]
			]
		];
		name = func["value"];
		fm = st["programModule"]["getFunctionModule", name];
		If[MissingQ[fm], 
			Return[
				upFunctionCall[st, inst, func]
			],
			ty = fm["type"]; 
			inferSt = st["inferenceState"];
			If[ ty === Undefined,
				ty = st["functionMap"]["lookup", name, Null];
				If[ ty === Null,
					ThrowException[{"Cannot find type in function map", name, inst}]
				];
				inferSt["addData", func, ty];
			];
			func["setType", ty];
			baseUpFunctionCall[st, inst["arguments"], inst["target"], func, inst]
		]
	]


upLambdaInstruction[ st_, inst_] :=
	Module[ {src = inst["source"], trgt = inst["target"], name, fm, ty, inferSt = st["inferenceState"]},
		If[ !ConstantValueQ[src],
			ThrowException[{"Lambda instruction must have a constant value for a source", src, inst}]
		];
		If[ !checkTypeEqual[ st, src["type"], trgt["type"]],
			name = src["value"];
			fm = st["programModule"]["getFunctionModule", name];
			If[ !MissingQ[fm], 
				ty = fm["type"]; 
				If[ ty === Undefined,
					ty = st["functionMap"]["lookup", name, Null];
					If[ ty === Null,
						ThrowException[{"Cannot find type in function map", name, inst}]
					];
					inferSt["addData", src, ty];
					,
					src["setType", ty]
				];
				inferSt["processTarget", trgt, ty, inst]
			]
		];
	]
		

upStackAllocateInstruction[ st_, inst_] :=
	baseUpFunctionCall[ st, {inst["size"]}, inst["target"], inst["operator"], inst]
	

upStoreInstruction[ st_, inst_] :=
	baseUpFunctionCall[ st, {inst["target"], inst["value"]}, Null, inst["operator"], inst]
	
upSetElementInstruction[ st_, inst_] :=
	baseUpFunctionCall[ st, Join[ {inst["target"]}, inst["offset"], {inst["source"]}], Null, inst["operator"], inst]
	
	
upLoadInstruction[ st_, inst_] :=
	baseUpFunctionCall[ st, {inst["source"]}, inst["target"], inst["operator"], inst]
	
upGetElementInstruction[ st_, inst_] :=
	baseUpFunctionCall[ st, Join[{inst["source"]}, inst["offset"]], inst["target"], inst["operator"], inst]



upInertInstruction[ st_, inst_] :=
	baseUpFunctionCall[ st, inst["arguments"], inst["target"], inst["head"], inst]
	
upBinaryInstruction[st_, inst_] :=
	upFunctionCall[st, inst, inst["operator"]]

upCompareInstruction[st_, inst_] :=
	upFunctionCall[st, inst, inst["operator"]]
	
upUnaryInstruction[st_, inst_] :=
	upFunctionCall[st, inst, inst["operator"]]

upInvokeInstruction[st_, inst_] :=
	upCallInstruction[st, inst]


(*
  up FunctionCall when the instruction has the typical shape of operands, target and operator
*)
upFunctionCall[st_, inst_, fun_] :=
	baseUpFunctionCall[st, inst["operands"], inst["target"], fun, inst]

getName[ arg_?ConstantValueQ] :=
	getName[arg["value"]]
	
getName[ arg_?VariableQ] :=
	getName[arg["name"]]
	
getName[ arg_] :=
	ToString[arg]
	

checkTypeEqual[st_, ty1_?TypeObjectQ, ty2_?TypeObjectQ] :=
	ty1["sameQ", ty2]
	
checkTypeEqual[ st_, ty1_, ty2_] :=
	False


(*
  Optimization code for call instructions.  If the type of the operator 
  matches the args/result type then this is all fully resolved and we 
  don't need to do any inferencing.
*)
checkCall[ st_, operTy_?TypeArrowQ, argTys_, trgTy_?TypeObjectQ] :=
	Module[ {operArgs, operTrgt},
		operArgs = operTy["arguments"];
		operTrgt = operTy["result"];
		Length[operArgs] =!= argTys && trgTy["sameQ", operTrgt] &&
			AllTrue[ Transpose[ {operArgs, argTys}], First[#]["sameQ", Last[#]]&]
	]


checkCall[ st_, operTy_, argTys_, trgTy_] :=
	False

(*
  Called by upFunctionCall,  and directly by instructions that don't have the typical shape.
  If the instruction does not have a return target,  then the trgt is Null.  This is treated as 
  a result of Void in the signature.
*)
baseUpFunctionCall[st_, args_List, trgt_, oper_, inst_] :=
	Module[{argTys, operTy, funTy, resTy, inferSt = st["inferenceState"], trgtTy},
		operTy = inferSt["processSource", oper, inst];
		argTys = inferSt["processSource", args, inst];
		trgtTy = If[ trgt === Null,
			inferSt["resolve", TypeSpecifier[ "Void"]],
			trgt["type"]
		];
		If[ checkCall[st, operTy, argTys, trgtTy],
			Return[]];
		If[ trgt === Null,
			resTy = "Void";
			,
			resTy = CreateTypeVariable[getName[oper] <> "res"];
			inferSt["processTarget", trgt, resTy, inst]
		];
        funTy = inferSt["resolve", TypeSpecifier[ argTys -> resTy]];
        operTy = inferSt["resolve", TypeSpecifier[ operTy]];
        inferSt["appendEqualConstraint", operTy, funTy, inst];
	]

upLoadGlobalInstruction[ st_, inst_] :=
	upCopyInstruction[st, inst]


upBranchInstruction[st_, inst_] :=
	Module[ {inferSt = st["inferenceState"]},
		If[inst["isConditional"],
			With[ {
				condTy = inferSt["processSource", inst["condition"]],
				boolTy = inferSt["resolve", TypeSpecifier["Boolean"]]
			},
				inferSt["appendEqualConstraint", condTy, boolTy, inst];
			]
		]
	]
	
downBranchInstruction[st_, inst_] :=
	Module[ {},
		downPushInstruction[st, inst];
		Scan[st["pushBasicBlock", #]&, inst["operands"]]
	]

downInvokeInstruction[st_, inst_] :=
	Module[ {},
		downPushInstruction[st, inst];
		st["pushBasicBlock", inst["to"]];
		st["pushBasicBlock", inst["unwind"]];
	]


upPhiInstruction[st_, inst_] :=
	Module[ {trgt, args, argTys, inferSt = st["inferenceState"]},
		trgt = inst["target"];
		args = inst["getSourceVariables"];
		inferSt["propagateAssumptions", trgt, args, inst];
		argTys = inferSt["processSource", args, inst];
		inferSt["processTarget", trgt, argTys, inst];
	]


upCopyInstruction[st_, inst_] :=
	Module[ {src, trgt, srcTy, inferSt = st["inferenceState"]},
		src = inst["source"];
		trgt = inst["target"];
		If[!checkTypeEqual[ st, src["type"], trgt["type"]],
			srcTy = inferSt["processSource", inst["source"], inst];
			inferSt["propagateAssumptions", trgt, src, inst];
			inferSt["processTarget", trgt, srcTy, inst]
		];
	]


upReturnInstruction[st_, inst_] :=
	Module[{var, ty,inferSt = st["inferenceState"]},
		If[inst["hasValue"],
			var = inst["value"];
			ty = inferSt["processSource", var, inst],
			ty = inferSt["resolve", TypeSpecifier["Void"]]
		];
		inferSt["type"]["set", ty]
	]

downPushInstruction[ st_, inst_] :=
	st["pushInstruction", inst]

TypeInferInstructions[ pm_?ProgramModuleQ, opts:OptionsPattern[]] :=
	TypeInferInstructions[pm, <| opts |>]
	
TypeInferInstructions[ pm_?ProgramModuleQ, opts_?AssociationQ] :=
	Module[ {state},
		state = CreateInferencePassState[pm, typeInferPM];
		typeInferPM[state, pm];
		If[ Lookup[ opts, "SolveConstraints", True],
			state["solve"];
			,
			pm["setProperty", "inferenceState" -> state];
		];
		pm
	]
	

initializeFM[ navigator_, fm_] :=
	Module[ {ty = fm["type"], name = fm["name"]},
		If[ ty === Undefined,
			ty = CreateTypeVariable[name]
		];
		navigator["functionMap"]["associateTo", name -> ty];
		If[ ty["variableCount"] > 0,
			navigator["inferenceState"]["addData", fm, ty]
		];
	]

typeInferPM[ state_, pm_?ProgramModuleQ] :=
	Module[ {funRes, navigator},
		navigator = createNavigator[state, pm];
		Scan[
			initializeFM[navigator, #]&, pm["getFunctionModules"]];
		funRes = Map[ generateFM[navigator,#]&, pm["getFunctionModules"]];
		navigator["dispose"];
	]
	
generateArgument[ state_, var_?VariableQ] :=
	Module[{type = var["type"]},
		If[type === Undefined,
			type = CreateTypeVariable["FunVar$" <> ToString[var["id"]]]];
		state["addBinding", var, type];
		type
  	]
	

generateFM[ navigator_, fm_?FunctionModuleQ] :=
	Module[ {fmType, b2, retType, funType, inferState, typeState, argTypes, args, pattState, dom, ran, ranVar},
		navigator["setup"];
		inferState = navigator["inferenceState"];
		typeState = inferState["typeInferenceState"];
		args = fm["arguments"];
		pattState = CreateInferencePassPatternState[inferState];
		argTypes = Map[generateArgument[pattState, #]&, args];
    	dom = pattState["patternVariables"];
    	ran = pattState["patternTypes"];
    	ranVar = Select[ran, TypeVariableQ];
		typeState["setMonomorphicSet", CreateReference[ranVar]];
		navigator["pushBasicBlock", fm["firstBasicBlock"]];
		navigator["iterate"];
		b2 = navigator["functionMap"]["lookup", fm["name"]];
		retType = inferState["type"]["get"];
		fmType = If[fm["type"] === Undefined,
			Undefined,
			inferState["resolve", fm["type"]]
		];
		If[fmType =!= Undefined && TypeArrowQ[fmType] && !checkTypeEqual[ navigator, retType, fmType["result"]],
			inferState["appendEqualConstraint", retType, fmType["result"], (* inst = *) None]
		];
		funType = inferState["resolve", TypeSpecifier[ argTypes -> retType]];
    	inferState["appendEqualConstraint", b2, funType, (* inst = *) None];
    	Apply[ 
    		Function[ {var, type},
    			If[inferState["hasAssumption", var],
    				Map[ addEqualConstraintIfDifferent[ navigator, inferState, type, #]&, inferState["lookupAssumptions", var]];
    				inferState["dropAssumption", var]
    			]
    		],
    		Transpose[ {dom, ran}],
    		{1}
    	]
	]


addEqualConstraintIfDifferent[ st_, inferState_, ty1_, ty2_] :=
	If[!checkTypeEqual[ st, ty1, ty2],
		inferState["appendEqualConstraint", ty1, ty2, (* inst = *) None]
	]



iterate[ self_] :=
	Module[ {bb, ins},
		While[ (bb = self["popBasicBlock"]) =!= Null,
			self["downVisitor"]["traverse", bb];
		];
		While[(ins = self["popInstruction"]) =!= Null,
			self["upVisitor"]["visit", ins]];
	]
	



createDownVisitor[data_] :=
	CreateInstructionVisitor[
		data,
		<|
			"visitBinaryInstruction" -> downPushInstruction,
			"visitBranchInstruction" -> downBranchInstruction,
			"visitCompareInstruction" -> downPushInstruction,
			"visitCallInstruction" -> downPushInstruction,
			"visitCopyInstruction" -> downPushInstruction,
			"visitGetElementInstruction" -> downPushInstruction,
			"visitInertInstruction" -> downPushInstruction,
			"visitInvokeInstruction" -> downInvokeInstruction,
			"visitLoadInstruction" -> downPushInstruction,
			"visitLoadGlobalInstruction" -> downPushInstruction,
			"visitLambdaInstruction" -> downPushInstruction,
			"visitPhiInstruction" -> downPushInstruction,
			"visitSetElementInstruction" -> downPushInstruction,
			"visitStackAllocateInstruction" -> downPushInstruction,
			"visitStoreInstruction" -> downPushInstruction,
			"visitReturnInstruction" -> downPushInstruction,
			"visitUnaryInstruction" -> downPushInstruction
		|>,
		"IgnoreRequiredInstructions" -> True
	]

createUpVisitor[data_] :=
	CreateInstructionVisitor[
		data,
		<|
			"visitBinaryInstruction" -> upBinaryInstruction,
			"visitBranchInstruction" -> upBranchInstruction,
			"visitCompareInstruction" -> upCompareInstruction,
			"visitCallInstruction" -> upCallInstruction,
			"visitCopyInstruction" -> upCopyInstruction,
			"visitGetElementInstruction" -> upGetElementInstruction,
			"visitInertInstruction" -> upInertInstruction,
			"visitInvokeInstruction" -> upInvokeInstruction,
			"visitLambdaInstruction" -> upLambdaInstruction,
			"visitLoadInstruction" -> upLoadInstruction,
			"visitLoadGlobalInstruction" -> upLoadGlobalInstruction,
			"visitPhiInstruction" -> upPhiInstruction,
			"visitSetElementInstruction" -> upSetElementInstruction,
			"visitStackAllocateInstruction" -> upStackAllocateInstruction,
			"visitStoreInstruction" -> upStoreInstruction,
			"visitReturnInstruction" -> upReturnInstruction,
			"visitUnaryInstruction" -> upUnaryInstruction
		|>,
		"IgnoreRequiredInstructions" -> True
	]


RegisterCallback["DeclareCompileClass", Function[{st},
NavigatorClass = DeclareClass[
	Navigator,
	<|
		"initialize" -> Function[ {}, initializeNavigator[Self]],
		"popInstruction" -> Function[{}, popInstruction[Self]],
		"pushInstruction" -> Function[{bb}, pushInstruction[Self, bb]],
		"popBasicBlock" -> Function[{}, popBasicBlock[Self]],
		"pushBasicBlock" -> Function[{bb}, pushBasicBlock[Self, bb]],
		"iterate" -> Function[ {}, iterate[Self]],
		"setup" -> Function[ {}, setup[Self]],
		"dispose" -> Function[ {}, navigatorDispose[Self]],
		"toString" -> Function[{}, toNavigatorString[Self]],
		"toBoxes" -> Function[{fmt}, toNavigatorBoxes[Self, fmt]]
	|>,
	{
		"inferenceState",
		"downVisitor",
		"upVisitor",
		"instructionStack",
		"basicBlocksStack",
		"basicBlocksSeen",
		"functionMap",
		"programModule"
	},
	Predicate -> NavigatorQ
]
]]



navigatorDispose[self_] :=
	Module[{},
		self["setUpVisitor", Null];
		self["setDownVisitor", Null];
	]


toNavigatorBoxes[typ_, fmt_]  :=
	StringJoin[
	"Navigator[",
	"<>",
	"]"
]


toNavigatorString[typ_] := StringJoin[
	"Navigator[",
	"<>",
	"]"
]


createNavigator[state_, pm_] :=
	CreateObject[
			Navigator,
			<|
				"programModule" -> pm,
				"inferenceState" -> state
			|>
		]
		
setup[self_] :=
	(
	self["setInstructionStack", CreateReference[{}]];
	self["setBasicBlocksStack", CreateReference[{}]];
	self["setBasicBlocksSeen", CreateReference[<||>]];
	)

initializeNavigator[self_] :=
	Module[ {},
		self["setFunctionMap", CreateReference[<||>]];
		self["setDownVisitor", createDownVisitor[self]];
		self["setUpVisitor", createUpVisitor[self]];
	]

pushBasicBlock[ self_, bb_] :=
	Module[ {},
		If[ self["basicBlocksSeen"]["keyExistsQ", bb["id"]],
			Return[]];
		self["basicBlocksSeen"]["associateTo", bb["id"] -> bb];
		self["basicBlocksStack"]["pushFront", bb];
	]


popBasicBlock[self_] :=
	If[self["basicBlocksStack"]["length"] === 0, Null, self["basicBlocksStack"]["popFront"]]


pushInstruction[ self_, ins_] :=
	self["instructionStack"]["pushFront", ins];

popInstruction[self_] :=
	If[self["instructionStack"]["length"] === 0, Null, self["instructionStack"]["popFront"]]


run[pm_, opts_] :=
	(
	TypeInferInstructions[pm, opts];
	pm)



RegisterCallback["RegisterPass", Function[{st},
passInfo = CreatePassInformation[
		"TypeInference",
		"The pass infers the types of each variable in an instruction using unification."
];

InferencePass = CreateProgramModulePass[<|
	"information" -> passInfo,
	"runPass" -> run,
	"traversalOrder" -> "reversePostOrder",
	"requires" -> {
		ResolveTypesPass
	},
	"postPasses" -> {
		ResolveSizeOfPass
	}
|>];

RegisterPass[InferencePass]
]]

End[]

EndPackage[]
