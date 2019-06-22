BeginPackage["Compile`Core`IR`Lower`Builder`ProgramModuleBuilder`"]

ProgramModuleBuilder;
ProgramModuleBuilderQ;
ProgramModuleBuilderClass;
CreateProgramModuleBuilder;



Begin["`Private`"] 


Needs["Compile`Core`IR`TypeDeclaration`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionRegistry`"]
Needs["Compile`Core`IR`Lower`Builder`SymbolBuilder`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["Compile`Core`IR`Lower`Builder`FunctionModuleBuilder`"]
Needs["Compile`Core`IR`ProgramModule`"]
Needs["CompileUtilities`Callback`"]


(**
 * For each instruction registered, we create a "create..Instruction" and dispatch the method
 * to the currentFunctionModuleBuilder. If the currentFunctionModuleBuilder is undefined, then
 * we throw an exception
 *) 
makeInstructionDispatch[instructionName_String] :=
	With[{method = StringJoin["create", instructionName]},
		method -> (
			If[Self["currentFunctionModuleBuilder"] === Undefined,
				ThrowException[{"No FunctionModuleBuilder set in the ProgramModuleBuilder instance. " <> "Cannot call " <> method <> " without FunctionModuleBuilder being set"}]
			];
			Self["currentFunctionModuleBuilder"][method, ##]&
		)
	]

RegisterCallback["InstructionDispatchTrait", Function[{st},
InstructionDispatchTrait = ClassTrait[<| makeInstructionDispatch /@ Keys[$RegisteredInstructions] |>]
]]

(**
 * The ProgramModuleBuilderClass is the way to construct program modules in the
 * compiler. One can create and add function moduels, external declarations, 
 * global values, and meta information to the program module using this builder
 *)

RegisterCallback["DeclareCompileClassPostRegisterInstruction", Function[{st},
ProgramModuleBuilderClass = DeclareClass[
	ProgramModuleBuilder,
	<|
		"initialize" -> Function[{},
			Self["setNextFunctionModuleId", CreateReference[1]];
			Self["setGlobalValues", CreateReference[{}]];
			Self["setMetaInformation", CreateReference[<||>]];
			Self["setExternalDeclarations", CreateReference[<||>]];
			Self["setTypeDeclarations", CreateReference[{}]];
			Self["setFunctionModuleBuilders", CreateReference[{}]];
			Self["setSymbolBuilder", CreateSymbolBuilder[Self]];
			Self["setProperties", CreateReference[<||>]];
		],
		"currentBasicBlock" -> (
			If[Self["currentFunctionModuleBuilder"] === Undefined,
				ThrowException[{"No FunctionModuleBuilder set in the ProgramModuleBuilder instance"}],
				Self["currentFunctionModuleBuilder"]["currentBasicBlock"]
			]&
		),
		"setCurrentBasicBlock" -> (
			If[Self["currentFunctionModuleBuilder"] === Undefined,
				ThrowException[{"No FunctionModuleBuilder set in the ProgramModuleBuilder instance"}],
				Self["currentFunctionModuleBuilder"]["setCurrentBasicBlock", ##]
			]&
		),
		"setLastBasicBlock" -> (
			If[Self["currentFunctionModuleBuilder"] === Undefined,
				ThrowException[{"No FunctionModuleBuilder set in the ProgramModuleBuilder instance"}],
				Self["currentFunctionModuleBuilder"]["setLastBasicBlock", ##]
			]&
		),
		"firstBasicBlock" -> (
			If[Self["currentFunctionModuleBuilder"] === Undefined,
				ThrowException[{"No FunctionModuleBuilder set in the ProgramModuleBuilder instance"}],
				Self["currentFunctionModuleBuilder"]["firstBasicBlock", ##]
			]&
		),
		"lastBasicBlock" -> (
			If[Self["currentFunctionModuleBuilder"] === Undefined,
				ThrowException[{"No FunctionModuleBuilder set in the ProgramModuleBuilder instance"}],
				Self["currentFunctionModuleBuilder"]["lastBasicBlock", ##]
			]&
		),
		"addArgument" -> (
			If[Self["currentFunctionModuleBuilder"] === Undefined,
				ThrowException[{"No FunctionModuleBuilder set in the ProgramModuleBuilder instance"}],
				Self["currentFunctionModuleBuilder"]["addArgument", ##]
			]&
		),
		(**< add an existing external declaration to the program module *)
		"addExternalDeclaration" -> (addExternalDeclaration[Self, ##]&),
		(**< creates an external declaration linked to the program module (this will add it to the pmb) *)
		"createExternalDeclaration" -> (createExternalDeclaration[Self, ##]&),
		
		(**< add an existing type declaration to the program module *)
		"addTypeDeclaration" -> (addTypeDeclaration[Self, ##]&),
		(**< creates a type declaration linked to the program module (this will add it to the pmb) *)
		"createTypeDeclaration" -> (createTypeDeclaration[Self, ##]&),
		
		(**< add an existing global value to the program module *)
		"addGlobalValue" -> (addGlobalValue[Self, ##]&),
		(**< creates a global value linked to the program module (this will add it to the pmb) *)
		"createGlobalValue" -> (createGlobalValue[Self, ##]&),
		
		(**< creates a function module linked to the program module (this will add it to the pmb) *)
		"createFunctionModule" -> (createFunctionModule[Self, ##]&),

		(**< add an existing meta information object to the program module *)
		"addMetaInformation" -> (addMetaInformation[Self, ##]&),
		(**< creates a meta information object linked to the program module (this will add it to the pmb) *)
		"createMetaInformation" -> (createMetaInformation[Self, ##]&),
		
		(**< Adds a basic block in the current function module builder *)
		"addBasicBlock" -> (
			If[Self["currentFunctionModuleBuilder"] === Undefined,
				ThrowException[{"No FunctionModuleBuilder set in the ProgramModuleBuilder instance"}],
				Self["currentFunctionModuleBuilder"]["addBasicBlock", ##]
			]&
		),
		(**< Creates a fresh basic block inside the current function module builder *)
		"freshBasicBlock" -> (
			If[Self["currentFunctionModuleBuilder"] === Undefined,
				ThrowException[{"No FunctionModuleBuilder set in the ProgramModuleBuilder instance"}],
				Self["currentFunctionModuleBuilder"]["freshBasicBlock", ##]
			]&
		),
		
		(**< Adds the instruction to the basic block *)
		"addInstruction" -> (
			If[Self["currentFunctionModuleBuilder"] === Undefined,
				ThrowException[{"No FunctionModuleBuilder set in the ProgramModuleBuilder instance"}],
				Self["currentFunctionModuleBuilder"]["addInstruction", ##]
			]&
		),
		(**< Creates an basic block inside the current function module builder *)
		"createBasicBlock" -> (
			If[Self["currentFunctionModuleBuilder"] === Undefined,
				ThrowException[{"No FunctionModuleBuilder set in the ProgramModuleBuilder instance"}],
				Self["currentFunctionModuleBuilder"]["createBasicBlock", ##]
			]&
		),
		
		
		"sealBasicBlock" -> (
			If[Self["currentFunctionModuleBuilder"] === Undefined,
				ThrowException[{"No FunctionModuleBuilder set in the ProgramModuleBuilder instance"}],
				Self["currentFunctionModuleBuilder"]["sealBasicBlock", ##]
			]&
		),

		(**< get all external declrations currently added to the program module as a list *)
		"getExternalDeclarations" -> Function[{},
			Self["externalDeclarations"]["get"]
		],
		(**< get all type declarations currently added to the program module as a list *)
		"getTypeDeclarations" -> Function[{},
			Self["typeDeclarations"]["get"]
		],
		(**< get all global values currently added to the program module as a list *)
		"getGlobalValues" -> Function[{},
			Self["globalValues"]["get"]
		],
		(**< get all function modules currently added to the program module as an association *)
		"getFunctionModules" -> Function[{},
			Map[#["getFunctionModule"]&, Self["functionModuleBuilders"]["get"]]
		],
		(**< get all function modules currently added to the program module as an association *)
		"functionModules" -> Function[{},
			CreateReference[Self["getFunctionModules"]]
		],
		(**< get all meta information currently added to the program module as a list *)
		"getMetaInformation" -> Function[{},
			Self["metaInformation"]["get"]
		],
		(**< generates a program module from the class instance *)
		"getProgramModule" -> Function[{},
			Module[{pm},
				pm = CreateProgramModule[
						Self["mexpr"],
						Self["globalValues"],
						Self["typeDeclarations"],
						Self["externalDeclarations"],
						Self["functionModules"],
						Self["metaInformation"]
				];
				pm["scanFunctionModules",
					Function[{fm}, fm["setProgramModule", pm]]
				];
				pm["setTypeEnvironment",
					Self["typeEnvironment"]
				];
				pm
			]
		],
		"dispose" -> Function[{}, dispose[Self]],
		"toString" -> Function[{},
			"ProgramModuleBuilder[<>]"
		],
		"toBoxes" -> Function[{fmt},
			toBoxes[Self, fmt]
		]

	|>,
	{
		"mexpr",
		"nextFunctionModuleId",
		"globalValues",
		"metaInformation",
		"externalDeclarations",
		"typeDeclarations",
		"functionModuleBuilders",
		"currentFunctionModuleBuilder" -> Undefined,
		"typeEnvironment",
		"symbolBuilder",
		"properties",
		"llvmDebug"
	},
	Predicate -> ProgramModuleBuilderQ,
	Extends -> {
		InstructionDispatchTrait,
		ClassPropertiesTrait
	}
]
]]


Options[CreateProgramModuleBuilder] = {
	"LLVMDebug" -> Automatic
}
CreateProgramModuleBuilder[opts:OptionsPattern[]] :=
	CreateProgramModuleBuilder[<| opts |> ]
CreateProgramModuleBuilder[opts_?AssociationQ] :=
	CreateProgramModuleBuilder[None, opts]
CreateProgramModuleBuilder[mexpr_, opts:OptionsPattern[]] :=
	CreateProgramModuleBuilder[mexpr, <| opts |>]
CreateProgramModuleBuilder[mexpr_, opts_?AssociationQ] :=
	CreateProgramModuleBuilder[mexpr, <||>, opts]
CreateProgramModuleBuilder[mexpr_, override_, opts:OptionsPattern[]] :=
	CreateProgramModuleBuilder[mexpr, override, <| opts |>]
CreateProgramModuleBuilder[mexpr_, override_, opts_?AssociationQ] :=
	Module[{st},
	    ProgramModuleBuilderClass; (* Load the class *)
		st = Join[
			<|
				"mexpr" -> mexpr,
				"llvmDebug" -> TrueQ[
					Lookup[opts, "LLVMDebug", False]
				]
			|>,
			override
		];
		CreateObject[ProgramModuleBuilder, st]
	]


addExternalDeclaration[builder_, args___] :=
	ThrowingTodo[addExternalDeclaration, builder, args]
	
createExternalDeclaration[builder_, args___] :=
	ThrowingTodo[createExternalDeclaration, builder, args]
	
addTypeDeclaration[builder_, decl_?TypeDeclarationQ] := (
	builder["typeDeclarations"]["appendTo", decl];
	decl
)
addTypeDeclaration[builder_, args___] :=
	ThrowException[{"Invalid arguments when addTypeDeclaration ", {args}}]
	
createTypeDeclaration[builder_, arg_] :=
	Module[{decl = CreateTypeDeclaration[arg]},
		addTypeDeclaration[builder, decl];
		decl
	]
	
addGlobalValue[builder_, args___] :=
	ThrowingTodo[addGlobalValue, builder, args]
	
createGlobalValue[builder_, args___] :=
	ThrowingTodo[createGlobalValue, builder, args]
	
	
createFunctionModule[builder_, name_String, mexpr_:None] :=
	Module[{fmb, id},
		id = builder["nextFunctionModuleId"]["increment"];
		fmb = CreateFunctionModuleBuilder[
			builder,
			id,
			name,
			mexpr,
			"LLVMDebug" -> builder["llvmDebug"]
		];
		builder["functionModuleBuilders"]["appendTo", fmb];
		builder["setCurrentFunctionModuleBuilder", fmb];
		fmb["setTypeEnvironment", builder["typeEnvironment"]];
		fmb
	]
createFunctionModule[args___] :=
	ThrowException[{"Invalid arguments when creatingFunctionModule ", {args}}]

addMetaInformation[builder_, key_ -> value_] :=
	builder["metaInformation"]["associateTo", key -> value]
	
createMetaInformation[builder_, args___] :=
	ThrowingTodo[createMetaInformation, builder, args]


dispose[self_] :=
	Module[{},
		Scan[#["dispose"]&, self["functionModuleBuilders"]["get"]];
		self["symbolBuilder"]["dispose"];
		self["setProperties", Null];
	]


(*********************************************************************)
(*********************************************************************)

(**
  * # Formating code
  *)
icon := Graphics[Text[
  Style["PMB", GrayLevel[0.7], Bold, 
   CurrentValue["FontCapHeight"]/
     AbsoluteCurrentValue[Magnification]]], $FormatingGraphicsOptions];   
     
toBoxes[builder_, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		"ProgramModuleBuilder",
		builder,
  		icon,
  		{
  			BoxForm`SummaryItem[{"mexpr: ", builder["mexpr"]}]
  		},
		{}, 
  		fmt,
		"Interpretable" -> False
  	]
	
	
End[]
EndPackage[]
