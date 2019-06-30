BeginPackage["Compile`Core`IR`Lower`Builder`SSABuilder`"]

(*
[Single Static Assignment (SSA)]  A property of the IR form such that a virtual register
is only assigned once. This implies that there is only one def for each virtual register.
It simplifies a lot of analysis. In live range analysis, for example, one needs to look at
the preceeding def to find the def-use chain. 
See http://www.cs.ucr.edu/~gupta/teaching/553-07/Papers/ssa.pdf for more information.

The algorithm used in based on 'Simple and Efficient Construction of Static Single Assignment Form - CC 2013' 
[http://www.cdl.uni-saarland.de/papers/bbhlmz13cc.pdf, http://www.cdl.uni-saarland.de/projects/ssaconstr/]
*)


SSABuilder;
SSABuilderQ;
SSABuilderClass;
CreateSSABuilder;

Begin["`Private`"]

Needs["CompileAST`Class`Symbol`"]; (* For MExprSymbolQ *)
Needs["Compile`Core`IR`Instruction`CopyInstruction`"]
Needs["Compile`Core`IR`Instruction`UnreachableInstruction`"]
Needs["CompileAST`Class`Literal`"]
Needs["CompileAST`Class`Normal`"]
Needs["Compile`Core`IR`BasicBlock`"]
Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Asserter`Assert`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`Instruction`InstructionQ`"]



(**
 * Each basic block contains a list of phi functions that have not
 * been fully inhabited (not all the inputs of the phi functions
 * are known) along with all the variables that are defined within
 * the basic block (these are a list of local definitions).
 * A basic block is sealed if all the parents have already been
 * defined, otherwise the basic block is unsealed (default).
 *)
addBasicBlock[self_, builder_, bb_, sealedQ_:False] :=
	Module[{bbState, bbid = bb["id"]},
		bbState = CreateReference[<|
			"incompletePhis" -> CreateReference[<||>],
			"currentDefs" -> CreateReference[<||>]
		|>];
		self["basicBlocks"]["associateTo", bbid -> bbState];
		If[TrueQ[sealedQ],
			sealBasicBlock[self, builder, bb]
		]
	]
(**
 * Gets all the phi functions that are incomplete for a basic block.
 * These are phi functions where some of the sources are not known.
 *)
incompletePhis[self_, bb_] :=
	Module[{bbState},
		AssertThat["incompletePhis:: The basic block " <> ToString[bb["name"]] <> "must already be added.",
			self["basicBlocks"]["keyExistsQ", bb["id"]]]["isTrue"];
		bbState = self["basicBlocks"]["lookup", bb["id"]];
		bbState["lookup", "incompletePhis"]
	]
(**
 * Gets a mapping from variables to instructions for all variables
 * that are defined within a basic block. The definitions are local.
 *)
currentDefs[self_, bb_] :=
	Module[{bbState},
		If[bb === Undefined,
			ThrowException[{bb}]
		];
		AssertThat["currentDefs:: The basic block " <> ToString[bb["name"]] <> " must already be added",
			self["basicBlocks"]["keyExistsQ", bb["id"]]]["isTrue"];
		bbState = self["basicBlocks"]["lookup", bb["id"]];
		(* Print[
			bbState["lookup", "currentDefs"]["get"]
		];
		*)
		bbState["lookup", "currentDefs"]
	]
(**
 * Checks if the basic block is sealed. A basic block is 
 * sealed if all the parents have been added (the parents
 * may not be themselves sealed).
 *)
sealedQ[self_, bb_] :=
	self["sealedBlocks"]["lookup", bb["id"], False]
	
sealBasicBlock[self_, builder_, bb_] :=
	Module[{bbid, iphis, var, phi},
		
		AssertThat["Basicblock is already sealed", sealedQ[self, bb]]["isFalse"];
		bbid = bb["id"];
		iphis = incompletePhis[self, bb];
		(* Print["iphis = " , bb, "   ", iphis["values"]]; *)
		Do[
			(*bb["addInstructionAfter",
				bb["firstInstruction"],
				iphis["lookup", var]
			];*)
			phi = iphis["lookup", var];
			addPhiOperands[
				self,
				builder,
				phi,
				var,
				phi["mexpr"]
			],
			{var, iphis["keys"]}
		];
		self["sealedBlocks"]["associateTo", bbid -> True]
	]

(* We need to differentiate between mexpr ids and variable ids
 * since these may clash when stored in the hash table
 *)
getId[mexpr_?MExprLiteralQ] :=
	StringRiffle[{"L", mexpr["data"]}, ""]
		
getId[mexpr_?MExprSymbolQ] :=
	mexpr["lexicalName"]	
	
getId[mexpr_?MExprNormalQ] :=
	StringRiffle[{"N", mexpr["toString"]}, ""]

(* An MExpr Id is preffered when hashing variables, since 
 * two variables may point to the same mexpr, but they never
 * point to the same variable (since it's unique)
 *)
getId[var_?VariableQ] :=
	If[var["mexpr"] =!= None && var["mexpr"] =!= Null,
	    getId[var["mexpr"]],
	    StringRiffle[{"V", var["id"]}, ""]
	]
getId[var_?ConstantValueQ] :=
	StringRiffle[{"C", var["id"]}, ""]
getId[var_?StringQ] :=
	var

getId[args___] :=
	ThrowException[{"Unrecognized call to getId", {args}//FullForm}]


writeVariable[self_, builder_, bb_, var_, value_] :=
	With[{defs = currentDefs[self, bb],
		  id = getId[var]},
		defs["associateTo", id -> value];
		var
	]

ClearAll[readVariableRecursive]
readVariableRecursive[self_, builder_, bb_, var_, mexpr:(_?MExprSymbolQ | None) : None] :=
	Module[{iphis, val = $Failed, inst},
		Which[
			!sealedQ[self, bb],
				inst = val = addPhiInstruction[self, builder, bb, mexpr];
				iphis = incompletePhis[self, bb];
				iphis["associateTo", var -> inst],
			bb["parents"]["length"] === 1,
				val = readVariable[self, builder, bb["parents"]["first"], var, mexpr],
			True,
				inst = addPhiInstruction[self, builder, bb, mexpr];
				writeVariable[self, builder, bb, var, inst];
				val = addPhiOperands[self, builder, inst, var, mexpr];
		];
		writeVariable[self, builder, bb, var, val];
		If[val =!= Undefined,
			optimizeRead[self, val]
			,
			val
		]
	]
readVariableRecursive[self_, args___] :=
        ThrowException[{"Bad args to readVariableRecursive: ", {args}}]

readLocalVariable[self_, builder_, bb_, var_String] :=
	With[{defs = currentDefs[self, bb],
		  id = getId[var]},
		If[defs["keyExistsQ", id],
			With[{val = defs["lookup", id]},
				(* Print["local var = ", var -> val]; *)
				val
			],
			$Failed
		]
	]

readVariable[self_, builder_, bb_, var_String, mexpr:(_?MExprSymbolQ | None) : None] :=
	With[{val = readLocalVariable[self, builder, bb, var]},
		(*Print["SSABuilder/readVariable", "[", var -> val["toString"], "]"];*)
		If[!FailureQ[val],
			optimizeRead[self, val],
			readVariableRecursive[self, builder, bb, var, mexpr]
		]
	]
readVariable[self_, args___] :=
	ThrowException[{"Unrecognized call to SSABuilder/readVariable"}]

appendPhiOperands[self_, phi_, bb_, var_] :=
	phi["addSource", bb, var]

appendPhiOperands[self_, phi_, bb_, Undefined] :=
	Null
	
addPhiOperands[self_, builder_, phi_, var_, mexpr:(_?MExprSymbolQ | None) : None] :=
	Module[{bb},
		bb = phi["basicBlock"];
		AssertThat["the phi instruction is associated with a basic block", BasicBlockQ[bb]]["isTrue"];
		Do[
			appendPhiOperands[
				self,
				phi,
				pred, (** Source basic block *)
				readVariable[self, builder, pred, var, mexpr] (** Value from pred *)
			], 
			{pred, bb["getParents"]}
		];
		tryRemoveTrivialPhi[self, builder, phi]
	]

addPhiInstruction[self_, builder_, bb_, mexpr:(_?MExprSymbolQ | None) : None] :=
	Module[{currBB, inst, first, var},
		currBB = builder["currentBasicBlock"];
		builder["setCurrentBasicBlock", bb];
		first = bb["lastPhiInstruction"];
		Assert[first =!= None];
		var = CreateVariable[];
		If[MExprSymbolQ[mexpr],
			var["setProperty", "variableValue" -> mexpr["lexicalName"]]];
		inst = builder["createPhiInstruction", var, {}, mexpr];
		bb["addInstructionAfter", first, inst];
		builder["setCurrentBasicBlock", currBB];
		inst
	]
 
tryRemoveTrivialPhi[self_, builder_, phi_] :=
	Module[{inst},
		Which[
			phi["source"]["isEmpty"],
				phi["unlink"];
				CreateUnreachableInstruction[],
			
			(* any of the source variables are the same as the target variable *)
			AnyTrue[phi["getSourceVariables"], phi["target"]["sameQ", #]&],
				(* We will remove the source variable in this case *)
				Module[{trgt, source},
					trgt = phi["target"];
					source = phi["source"]["get"];
					source = MapThread[
						Function[{blk, src},
							If[trgt["sameQ", src],
								Nothing,
								{blk, src}
							]
						],
						Transpose[source]
					];
					phi["source"]["set", source];
					tryRemoveTrivialPhi[self, builder, phi]
				],
				
			phi["source"]["length"] === 1 || (* has only one argument *)
			AllTrue[phi["getSourceVariables"], phi["getOperand", 1]["sameQ", #]&] (* Arguments are all the same *),
				(* Change into a load instruction *)
				inst = CreateCopyInstruction[phi["target"], phi["getOperand", 1], phi["mexpr"]];
				inst["moveAfter", phi];
				phi["unlink"];
				inst,
			True,
				phi (* TODO: Maybe there are other optimizations *)
		]
	]



(**
 * The SSABuilderClass is the way to construct program modules in the
 * compiler. One can create and add function modules, external declarations, 
 * global values, and meta information to the program module using this builder
 *)
RegisterCallback["DeclareCompileClass", Function[{st},
SSABuilderClass = DeclareClass[
	SSABuilder,
	<|
		"initialize" -> Function[{},
			Self["setBasicBlocks", CreateReference[<||>]];
			Self["setSealedBlocks", CreateReference[<||>]];
		],
		"addBasicBlock" -> (addBasicBlock[Self, ##]&),
		"sealBasicBlock" -> (sealBasicBlock[Self, ##]&),
		"writeVariable" -> (writeVariable[Self, ##]&),
		"readVariable" -> (readVariable[Self, ##]&),
		"dispose" -> Function[{}, dispose[Self]],
		"toString" -> Function[{}, toString[Self]],
		"toBoxes" -> Function[{fmt},
			toBoxes[Self, fmt]
		]

	|>,
	{
		"basicBlocks",
		"sealedBlocks"
	},
	Predicate -> SSABuilderQ
]
]]


CreateSSABuilder[] :=
	CreateObject[
		SSABuilder,
		<|
		|>
	]


(*********************************************************************)
(*********************************************************************)

ClearAll[optimizeRead]
optimizeRead[self_, const_?ConstantValueQ] :=
	 const
	 
optimizeRead[self_, var_?VariableQ] := 
	var

optimizeRead[self_, inst_?InstructionQ] := 
	inst["target"]
	
optimizeRead[self_, args___] :=
	ThrowException[{"Invalid arguments for optimizeRead ", {args}}]



dispose[self_] :=
	Module[{vals},
		self["setBasicBlocks", Null];
		(*vals = self["basicBlocks"]["values"];
		Print[ vals];
		Print[ self["sealedBlocks"]];*)
	]



(*********************************************************************)
(*********************************************************************)
	
(**
  * # Formating code
  *)
icon := Graphics[Text[
  Style["SSA", GrayLevel[0.7], Bold, 
   CurrentValue["FontCapHeight"]/
     AbsoluteCurrentValue[Magnification]]], $FormatingGraphicsOptions];   
     
toBoxes[builder_, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		"SSABuilder",
		builder,
  		icon,
  		{
  			BoxForm`SummaryItem[{"basicBlocks: ", builder["basicBlocks"]}],
  			BoxForm`SummaryItem[{"sealedBlocks: ", builder["sealedBlocks"]}]
  		},
		{}, 
  		fmt,
		"Interpretable" -> False
  	]
	
toString[self_] := (
	StringJoin[
		"SSABuilder[",
		"\n",
		self["basicBlocks"]["toString"],
		"\n",
		self["sealedBlocks"]["toString"],
		"\n]"
	]
)


	
End[]
EndPackage[]
