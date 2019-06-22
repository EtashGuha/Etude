BeginPackage["Compile`Core`Transform`Closure`DesugarClosureEnvironmentStructure`"]


DesugarClosureEnvironmentStructurePass

Begin["`Private`"] 

Needs["Compile`Core`PassManager`ProgramModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Variable`"]
Needs["Compile`Core`IR`ProgramModule`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["TypeFramework`TypeObjects`TypeConstructor`"]
Needs["TypeFramework`TypeObjects`TypeApplication`"]
Needs["Compile`Core`Transform`ResolveTypes`"]

ClearAll[isHandleType]
isHandleType[var_?VariableQ] :=
	isHandleType[var["type"]]
isHandleType[var_?ConstantValueQ] :=
	isHandleType[var["type"]]
isHandleType[ty_?TypeApplicationQ] :=
	TypeConstructorQ[ty["type"]] &&
	ty["type"]["typename"] === "Handle";
isHandleType[___] := False

ClearAll[underlyingType]
underlyingType[var_?VariableQ] :=
	underlyingType[var["type"]]
underlyingType[var_?ConstantValueQ] :=
	underlyingType[var["type"]]
underlyingType[ty_?TypeApplicationQ] :=
	If[isHandleType[ty],
		underlyingType[First[ty["arguments"]]],
		ty["type"]
	];


ClearAll[structTypeQ]
structTypeQ[var_?VariableQ] :=
	structTypeQ[underlyingType[var["type"]]]
structTypeQ[var_?ConstantValueQ] :=
	structTypeQ[underlyingType[var["type"]]]
structTypeQ[ty_?TypeConstructorQ] :=
	With[{
		metadata = ty["getProperty", "metadata", Null]
	},
		metadata =!= Null &&
		KeyExistsQ[metadata, "Fields"]
	];
structTypeQ[___] := False;
	
ClearAll[dropFieldsProperty]
dropFieldsProperty[var_?VariableQ] :=
	dropFieldsProperty[underlyingType[var["type"]]]
dropFieldsProperty[var_?ConstantValueQ] :=
	dropFieldsProperty[underlyingType[var["type"]]]
dropFieldsProperty[ty_?TypeConstructorQ] :=
	With[{
		metadata = ty["getProperty", "metadata", <||>]
	},
		ty["setProperty", "structQ" -> True];
		ty["setProperty", "metadata" -> KeyDrop[metadata, "Fields"]]
	];

desugar[var_, offsets_] :=
	Module[{structTy, fields, changed = False, newOffsets},	
		structTy = underlyingType[var];
		fields = Lookup[structTy["getProperty", "metadata"], "Fields"];
		newOffsets = Table[
			Which[
				!ConstantValueQ[offset],
					offset,
				!StringQ[offset["value"]],
					offset,
				MissingQ[Lookup[fields, offset["value"]]],
					offset,
				True,
					changed = True;
					With[{
						newOffset = CreateConstantValue[Lookup[fields, offset["value"]]]
					},
						newOffset["setType", TypeSpecifier["MachineInteger"]];
						newOffset["setProperty", "fieldName" -> offset["value"]];
						newOffset
					]
			],
			{offset, offsets}
		];
		{changed, newOffsets}
	];	

visitSetElement[_, inst_] :=
	Module[{changed, res},
		If[!structTypeQ[inst["target"]],
			Return[]
		];
		res = desugar[inst["target"], inst["offset"]];
		changed = res[[1]];
		If[changed === True,
			inst["setOffset", res[[2]]]
		];
	];
	
visitGetElement[_, inst_] :=
	Module[{changed, res},
		If[!structTypeQ[inst["source"]],
			Return[]
		];
		res = desugar[inst["source"], inst["offset"]];
		changed = res[[1]];
		If[changed === True,
			inst["setOffset", res[[2]]]
		];
	];
	
finalizeSetElement[_, inst_] :=
	If[structTypeQ[inst["target"]],
		dropFieldsProperty[inst["target"]]
	];
finalizeGetElement[_, inst_] :=
	If[structTypeQ[inst["source"]],
		dropFieldsProperty[inst["source"]]
	];

run[fm_?FunctionModuleQ, opts_] :=
	Module[{},
		CreateInstructionVisitor[
			<|
				"visitSetElementInstruction" -> visitSetElement,
				"visitGetElementInstruction" -> visitGetElement
			|>,			
			fm,
			"IgnoreRequiredInstructions" -> True
		];
		fm
	];
run[pm_?ProgramModuleQ, opts_] :=
	pm["scanFunctionModules", Function[{fm}, run[fm, opts]]];
	
finalize[fm_?FunctionModuleQ, opts_] :=
	Module[{},
		CreateInstructionVisitor[
			<|
				"visitSetElementInstruction" -> finalizeSetElement,
				"visitGetElementInstruction" -> finalizeGetElement
			|>,			
			fm,
			"IgnoreRequiredInstructions" -> True
		];
		fm
	];
finalize[pm_?ProgramModuleQ, opts_] := 0
	xpm["scanFunctionModules", Function[{fm}, finalize[fm, opts]]];
	

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
		"DesugarClosureEnvironmentStructure",
		"The pass replaces SetElement/GetElement where the index is a string name, with an index for structure types",
		"For instructions where the type is a structure, this pass replaces SetElement/GetElement where the index is a string name " <>
		"to the corresponding offset index." <>
		"e.g. \n" <>
		"if $CEnv[I64, I64] is a struct with fields \"X\" -> 1 and \"Y\" -> 2 then the instruction \n" <>
		"\tSetElement[%138:H[$CEnv[I64,I64]], \"X\"] = 3 \n" <>
		"is translated to \n" <>
		"\tSetElement[%138:H[$CEnv[I64,I64]], 1] = 3 \n" <>
		"this keeps both the lowering and the type system happy. \n" <> 
		"This pass then deletes the fields property from the closure environment type"
];

DesugarClosureEnvironmentStructurePass = CreateProgramModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"finalizePass" -> finalize,
	"requires" -> {
		ResolveTypesPass
	}
|>];

RegisterPass[DesugarClosureEnvironmentStructurePass]
]]

End[] 

EndPackage[]