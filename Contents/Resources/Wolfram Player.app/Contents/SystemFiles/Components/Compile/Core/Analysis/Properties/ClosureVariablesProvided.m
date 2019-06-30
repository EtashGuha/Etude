
BeginPackage["Compile`Core`Analysis`Properties`ClosureVariablesProvided`"]

ClosureVariablesProvidedPass

Begin["`Private`"]

Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`ProgramModulePass`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`Transform`Closure`Utilities`"]
Needs["Compile`Core`IR`Instruction`InstructionQ`"]
Needs["Compile`Core`IR`Instruction`CallInstruction`"]
Needs["Compile`Core`IR`Instruction`LoadArgumentInstruction`"]
Needs["Compile`Core`Analysis`DataFlow`Def`"]

processingPropertyName = "processingClosureVariablesProvidedPass"

getLocalFunction[fm_, func_] :=
	Module[{pm, name, fms},
		pm = fm["programModule"];
		name = func["value"];
		fms = pm["functionModules"]["get"];
		SelectFirst[fms, #["name"] === name&]
	];

(*
  See code in PackClosureEnvironment.
  This is a weak attempt to fix up closure calls by recording 
  the remapping of aliased variables with the target of a LoadClosure
  variable.
  
  This is used for recursive calls in PackClosureEnvironment, but I think it
  probably should be more general.
*)
collectLoadClosure[ state_, inst_] :=
	Module[ {trgt = inst["target"], aliased},
		aliased = trgt["getProperty", "aliasesVariable"];
		state["closureVariablesBinding"]["associateTo", aliased["id"] -> trgt]
		]


visitGeneric[data_, inst_, func_] :=
	Module[{
		localFM,
		opts = data["opts"],
		fm = data["fm"],
		localFMCaptures,
		newVars
	},
		If[!ConstantValueQ[func],
			Return[];
		];
		If[func["value"] === Native`LoadClosureVariable,
			collectLoadClosure[ data, inst]];
		localFM = getLocalFunction[fm, func];
		If[MissingQ[localFM],
			Return[]
		];
		runRecur[localFM, data["callChain"], opts];
		(*
		   inner FMs are nested inside of outer FMs
		   
		   closureVariablesProvided are variables in an FM which are captured by 
		   inner FMs.
		   
		   closureVariablesConsumed are variables in an outer FM which are captured by this FM 
		*)
		localFMCaptures = If[localFM["hasProperty", "closureVariablesConsumed"],
			localFM["getProperty", "closureVariablesConsumed"]["get"],
			{}
		];
		newVars = Join[
				fm["getProperty", "closureVariablesProvided", {}],
				localFMCaptures
			];
		newVars = DeleteDuplicates[newVars, #1["id"] === #2["id"]&];
		fm["setProperty", "closureVariablesProvided" -> newVars];
	];
	
visitCall[data_, inst_] :=
	visitGeneric[data, inst, inst["function"]]
visitLambda[data_, inst_] :=
	visitGeneric[data, inst, inst["source"]]
visitInert[data_, inst_] :=
	visitGeneric[data, inst, inst["head"]]
	
runRecur[fm_, callChain_, opts_] :=
	Module[{
		state, visitor, varsProvided, varBinding
	},
		If[fm["getProperty", processingPropertyName, False],
			Return[]
		];
		fm["setProperty", processingPropertyName -> True];
		callChain["associateTo", fm["id"] -> fm];
		state = <|
			"opts" -> opts,
			"fm" -> fm,
			"callChain" -> callChain,
			"closureVariablesBinding" -> CreateReference[<||>]
		|>;
		visitor = CreateInstructionVisitor[
			state, 
			<|
				"visitCallInstruction" -> visitCall,
				"visitInertInstruction" -> visitInert,
				"visitLambdaInstruction" -> visitLambda
			|>,			
			fm,
			"IgnoreRequiredInstructions" -> True
		];
		callChain["keyDropFrom", fm["id"]];
		varBinding = state["closureVariablesBinding"];
		varsProvided = fm[ "getProperty", "closureVariablesProvided", Null];
		If[ varsProvided =!= Null,
			Scan[ If[!varBinding["keyExistsQ", #["id"]], 
					fixVariableBinding[ fm, callChain, varBinding, #]]&, varsProvided]];		
		fm["setProperty", "closureVariablesBinding" -> varBinding];
		fm
	]

(*
  varProvided is not found in varBinding. There are three cases:
  1) varProvided is defined in fm,  in this case we just add to varBinding  #["id"] -> #
  2) varProvided is defined in an fm up the callChain,  but there was not a Native`LoadClosureVariable
     call.  In this case we should add the call,  add varProvided to closureVariablesConsumed 
     and fix varBinding
  3) varProvided is not defined in an fm up the callChain,  this is an error
*)
fixVariableBinding[ fm_, callChain_, varBinding_, varProvided_] :=
	Module[ {def = varProvided["def"], varFM, newVar},
		If[ !InstructionQ[def],
			ThrowException[{"variable does not have a valid definition", varProvided}]];
		varFM = def["basicBlock"]["functionModule"];
		Which[
			fm["id"] === varFM["id"],
				newVar = varProvided
			,
			callChain["keyExistsQ", varFM["id"]],
				newVar = addLoadClosureVariable[fm, varProvided];
			,
			True,
				ThrowException[{"closure variable not found in call hierarchy", varProvided}]];
		varBinding["associateTo", varProvided["id"] -> newVar]
	]
	
(*
 
*)
addLoadClosureVariable[ fm_, var_] :=
	Module[ {fun = CreateConstantValue[ Native`LoadClosureVariable], newInst, trgt, bb, inst},
		newInst = CreateCallInstruction["closureConsumer", fun, {}];
		trgt = newInst["target"];
		bb = fm["firstBasicBlock"];
		inst = bb["firstNonLabelInstruction"];
		inst = findInstruction[ inst];
		newInst["moveBefore", inst];
		AddClosureCapturerProperties[ fm, trgt, var];
		trgt
	]

findInstruction[ inst_] :=
	If[ LoadArgumentInstructionQ[inst], 
		findInstruction[inst["next"]],
		inst]

	
(*
 Var should be a local var of fm,  serious error if not, but needs Use/Def set up.
*)
checkVarLocal[ fm_, var_] :=
	Module[ {varFM = var["def"]["basicBlock"]["functionModule"]},
		If[varFM["id"] =!= fm["id"],
			ThrowException[{"variable expected to be a local of function module", var, fm, varFM}]];
	]

	
run[ pm_, opts_] :=
	Module[{callChain = CreateReference[<||>]},
		pm["scanFunctionModules",
			#["removeProperty", "closureVariablesProvided"]&];
		pm["scanFunctionModules",
			runRecur[#, callChain, opts]&];
		pm["scanFunctionModules",
			#["removeProperty", processingPropertyName]&];
		pm
	]
	

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"ClosureVariablesProvided",
		"Computes the set of variables that are defined by the function and are used by any closure function that's called by " <>
		"the current function. This property is transitive."
];

ClosureVariablesProvidedPass = CreateProgramModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"requires" -> {
		DefPass
	},
	"passClass" -> "Analysis"
|>];

RegisterPass[ClosureVariablesProvidedPass]
]]


End[]

EndPackage[]
