BeginPackage["Compile`Core`Transform`Closure`Utilities`"]

HasClosureQ
ClosureEnvironmentTypeName
CapturesVariablesQ
CapturedVariables
CalleeCapturedVariables
LoadClosureVariableCallQ
ClosureEnvironmentType
ClosureEnvironmentVariableName
ResolveEnvironmentVariableType

SetClosureEnvironmentFunctionName
GetClosureEnvironmentFunctionName
SetClosureEnvironmentFunction
GetClosureEnvironmentFunction

CreateTypedConstant
CreateBitCastTo

RemoveClosureCapteeProperties
RemoveClosureCapturerProperties

AddClosureCapteeProperties
AddClosureCapturerProperties

GetClosureFunctionType

Begin["`Private`"] 

Needs["Compile`Core`IR`ProgramModule`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["TypeFramework`"]
Needs["TypeFramework`TypeObjects`TypeArrow`"]
Needs["CompileUtilities`Asserter`Assert`"]
Needs["Compile`Core`IR`Variable`"]
Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`IR`Instruction`CallInstruction`"]

HasClosureQ[pm_?ProgramModuleQ] :=
	pm["getProperty", "hasClosure", False]

SetClosureEnvironmentFunctionName[fm_, ii_] :=
	With[{
		envVarName = ClosureEnvironmentTypeName[fm]
	},
		"Set$$" <> envVarName <> "$$elem$$" <> ToString[ii]
	]
GetClosureEnvironmentFunctionName[fm_, ii_] :=
	With[{
		envVarName = ClosureEnvironmentTypeName[fm]
	},
		"Get$$" <> envVarName <> "$$elem$$" <> ToString[ii]
	]


makeClosureSetEnvironmentAccessorFunction[fm_, ii_, envTy_] :=
	With[{
		pm = fm["programModule"],
		name = SetClosureEnvironmentFunctionName[fm, ii]
	},
	With[{
		tyEnv = pm["typeEnvironment"],
		res = CreateConstantValue[name]
	},
		res["setType", TypeSpecifier[{"Handle"[envTy], "MachineInteger", TypeProjection[envTy, ii]} -> "Void"]];
		res 
	]];

makeClosureGetEnvironmentAccessorFunction[fm_, ii_, envTy_] :=
	With[{
		pm = fm["programModule"],
		name = GetClosureEnvironmentFunctionName[fm, ii]
	},
	With[{
		tyEnv = pm["typeEnvironment"],
		res = CreateConstantValue[name]
	},
		res["setType", TypeSpecifier[{"Handle"[envTy], "MachineInteger"} -> TypeProjection[envTy, ii]]];
		res 
	]];

makeClosureEnvironmentAccessorFunction[fm_, ii_, envTy_] :=
	<|
		"set" -> makeClosureSetEnvironmentAccessorFunction[fm, ii, envTy],
		"get" -> makeClosureGetEnvironmentAccessorFunction[fm, ii, envTy]
	|>
ClosureEnvironmentAccessorFunction[fm_, ii_, envTy_] :=
	With[{
		pm = fm["programModule"]
	},
		If[!pm["hasProperty", "closureAccessors"],
			pm["setProperty", "closureAccessors" -> <||>]
		];
		
		AssertThat["The closure accessor property on the program module is an association",
			pm["getProperty", "closureAccessors"]
		]["named", "closureAccessors"
		]["satisfies", AssociationQ
		];
			
	With[{
		acc = Lookup[pm["getProperty", "closureAccessors"], closureAccessor[fm["id"], ii]]
	},
		If[MissingQ[acc],
			With[{
				new = makeClosureEnvironmentAccessorFunction[fm, ii, envTy]
			},
				pm["setProperty", "closureAccessors",
					Append[
						pm["getProperty", "closureAccessors", <||>],
						new
					]
				];
				new
			],
			acc
		]
	]];
SetClosureEnvironmentFunction[fm_, ii_, envTy_] :=
	With[{
		acc = ClosureEnvironmentAccessorFunction[fm, ii, envTy]
	},
		acc["set"]
	]
GetClosureEnvironmentFunction[fm_, ii_, envTy_] :=
	With[{
		acc = ClosureEnvironmentAccessorFunction[fm, ii, envTy]
	},
		acc["get"]
	];

ClosureEnvironmentTypeName[fm_?FunctionModuleQ] :=
	With[ {tyName = fm["getProperty", "ClosureEnvironmentTypeName", Null]},
		If[ tyName === Null,
			With[ {name = "closureEnv" <> fm["name"] <> SymbolName[Unique["base"]]},
				fm["setProperty", "ClosureEnvironmentTypeName" -> name];
				name
			]
			,
			tyName]
	]

ClosureEnvironmentTypeName[args___] :=
	ThrowException[{"Unrecognized call to ClosureEnvironmentTypeName", {args}}]
	
ClosureEnvironmentVariableName[fm_?FunctionModuleQ] :=
	With[{
		name = fm["name"]
	},
		"var$$" <> name <> "$$closure$env"
	];
ClosureEnvironmentVariableName[args___] :=
	ThrowException[{"Unrecognized call to ClosureEnvironmentVariableName", {args}}]
	
	
ClosureEnvironmentType[fm_?FunctionModuleQ] :=
	With[{
		name = ClosureEnvironmentTypeName[fm]
	},
		TypeSpecifier[name]
	];
ClosureEnvironmentType[args___] :=
	ThrowException[{"Unrecognized call to ClosureEnvironmentType", {args}}]
	


CapturesVariablesQ[fm_] := 
	fm["hasProperty", "closureVariablesConsumed"] && 
	fm["getProperty", "closureVariablesConsumed"]["length"] > 0
	
CapturesVariablesQ[args___] :=
	ThrowException[{"Unrecognized call to CapturesVariablesQ", {args}}]

CapturedVariables[fm_] := 
	fm["getProperty", "closureVariablesConsumed"]["get"]
	
CalleeCapturedVariables[fm_] := 
	fm["getProperty", "closureVariablesProvided", {}]
	
CapturedVariables[args___] :=
	ThrowException[{"Unrecognized call to CapturedVariables", {args}}]

LoadClosureVariableCallQ[v_?ConstantValueQ] :=
	v["value"] === Native`LoadClosureVariable
LoadClosureVariableCallQ[___] :=
	False
	
mkTypes[tyEnv_, var_] := 
	Module[{ty = var["type"]},
		If[TypeObjectQ[ty] || Head[ty] === Type || Head[ty] === TypeSpecifier,
			Return[ty]
		];
		If[tyEnv["resolvableQ", ty],
			Return[ty]
		];
		If[!VariableQ[var] && !ConstantValueQ[var],
			ThrowException[{"Unrecognized call to mkTypes. Expecting either a constant or a variable", {var}}]
		];
		ty = TypeSpecifier[TypeVariable[var["name"]]];
		var["setType", ty];
		ty
	];
	
	
(* Create the environment variable type *)
ResolveEnvironmentVariableType[tyEnv_, fm_, captured_] :=
	Module[{capturedVarTypes, envTyName, envTy},
		envTyName = ClosureEnvironmentTypeName[fm];
		capturedVarTypes = mkTypes[tyEnv, #]& /@ captured;
		envTy = TypeSpecifier[Apply[envTyName, capturedVarTypes]];
		envTy["setProperty", "closureEnvironmentQ" -> True]; 
		envTy
	];


CreateTypedConstant[ tyEnv_, val_, ty_] :=
	Module[{cons = CreateConstantValue[val]},
		cons["setType", ty];
		cons
	]

(*
  var is captured by a closure,  remove closure properties from it and from fm
*)
RemoveClosureCapteeProperties[ fm_?FunctionModuleQ, var_?VariableQ] :=
	Module[{callee},
		var["removeProperty", "capturedByVariables"];
		var["removeProperty", "isCapturedVariable"];
		callee = fm["getProperty", "closureVariablesProvided", {}];
		callee = Select[ callee, #["id"] =!= var["id"]&];
		If[ Length[callee] === 0,
			fm["removeProperty", "closureVariablesProvided"],
			fm["setProperty", "closureVariablesProvided" -> callee]];
	]

RemoveClosureCapteeProperties[args___] :=
	ThrowException[{"Unrecognized call to RemoveClosureCapteeProperties", {args}}]

AddClosureCapteeProperties[ fm_?FunctionModuleQ, var_?VariableQ, newVars_] :=
	Module[{callee, captured = var["getProperty", "capturedByVariables", {}]},
		captured = Join[ captured, newVars];
		var["setProperty", "isCapturedVariable" -> True];
		var["setProperty", "capturedByVariables" -> captured];
		callee = fm["getProperty", "closureVariablesProvided", {}];
		callee = Append[ callee, var];
		fm["setProperty", "closureVariablesProvided" -> callee];
	]

AddClosureCapteeProperties[args___] :=
	ThrowException[{"Unrecognized call to RemoveClosureCapteeProperties", {args}}]

(*
  var captures a variable from a higher level,  remove closure properties from it and from fm
*)
RemoveClosureCapturerProperties[ fm_?FunctionModuleQ, var_?VariableQ, varMain_?VariableQ] :=
	Module[{captured},
		var["removeProperty", "aliasesVariable"];
		var["removeProperty", "isClosureVariable"];
		captured = fm["getProperty", "closureVariablesConsumed", Null];
		If[ captured === Null,
			ThrowException[{"Unexpected form for closureVariablesConsumed", {captured}}]];
		captured["deleteCases", x_ /; (x["id"] === varMain["id"])];
		If[ captured["length"] === 0,
			fm["removeProperty", "closureVariablesConsumed"]];
	]

RemoveClosureCapturerProperties[args___] :=
	ThrowException[{"Unrecognized call to RemoveClosureCapturerProperties", {args}}]

AddClosureCapturerProperties[ fm_?FunctionModuleQ, var_?VariableQ, varMain_?VariableQ] :=
	Module[{captured},
		var["setProperty", "aliasesVariable" -> varMain];
		var["setProperty", "isClosureVariable" -> True];
		captured = fm["getProperty", "closureVariablesConsumed", Null];
		If[ captured === Null,
			captured = CreateReference[{}];
			fm["setProperty", "closureVariablesConsumed" -> captured]];
		captured["appendTo", varMain];
	]

AddClosureCapturerProperties[args___] :=
	ThrowException[{"Unrecognized call to AddClosureCapturerProperties", {args}}]


CreateBitCastTo[pm_, var_, ty_] :=
	Module[{name = Native`PrimitiveFunction["BitCast"]},
		pm["externalDeclarations"]["lookupUpdateFunction", pm["typeEnvironment"], name];
		createCallInstruction[pm, name, {var}, ty]
	]

createCallInstruction[pm_, fun_, vars_List, ty_] :=
	Module[ {conFun, funTy, newInst, tyEnv = pm["typeEnvironment"]},
		conFun = CreateConstantValue[ fun];
		funTy = tyEnv["resolve", TypeSpecifier[ Map[#["type"]&, vars] -> ty]];
		conFun["setType", funTy];
		newInst = CreateCallInstruction[ "callres", conFun, vars];
		newInst["target"]["setType", tyEnv["resolve", TypeSpecifier[ ty]]];
		newInst
	]


(*
	Fix the function type.
*)

GetClosureFunctionType[tyEnv_, fm_, envRefTy_] :=
	Module[{funTy, args},
		funTy = fm["type"];
		If[ !TypeArrowQ[funTy],
			ThrowException[{"Unexpected function type", fm, funTy}]];
		args = Prepend[ funTy["arguments"], envRefTy];
		funTy = tyEnv["resolve", TypeSpecifier[ args -> funTy["result"]]];
		funTy
	];


End[]
	
EndPackage[]
