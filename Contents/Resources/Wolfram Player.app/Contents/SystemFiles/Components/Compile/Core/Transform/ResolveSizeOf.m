BeginPackage["Compile`Core`Transform`ResolveSizeOf`"]

ResolveSizeOfPass

Begin["`Private`"] 

Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Instruction`CopyInstruction`"]
Needs["TypeFramework`"]
Needs["TypeFramework`TypeObjects`TypeConstructor`"]
Needs["TypeFramework`TypeObjects`TypeApplication`"]
Needs["CompileUtilities`Error`Exceptions`"]

isSizeOfCall[inst_] :=
	Length[inst["arguments"]] === 1 &&
	ConstantValueQ[inst["function"]] &&
	inst["function"]["value"] === Native`SizeOf

getSize[ ty_?TypeConstructorQ] :=
	Module[{size = ty["getProperty", "ByteCount", Null]},
		If[size === Null,
			Return[Null]];
		size
	]

(*
 Could also look at the ByteCount for the type,  this might
 be good eg, for CArray.
*)
getSize[ ty_?TypeApplicationQ] :=
	Module[{sizeFun = ty["type"]["getProperty", "ByteCountFunction", Null], sizes},
		If[sizeFun === Null,
			Return[Null]];
		sizes = Map[ getSize, ty["arguments"]];
		sizeFun[sizes]
	]

getSize[ ty_] :=
	Module[ {},
		Null
	]

visit[st_, inst_] :=
	With[{},
		If[!isSizeOfCall[inst],
			Return[]
		];
		With[{
			val = getSize[inst["getArgument", 1]["type"]]
		},
			If[ !IntegerQ[val],
				ThrowException[{"Cannot resolve SizeOf", inst["mexpr"]["toString"]}]];
			With[{
				load = CreateCopyInstruction[
					inst["target"],
					CreateConstantValue[val],
					inst["mexpr"]
				]
			},
				load["moveAfter", inst];
				load["setId", inst["id"]];
				inst["unlink"];
	        ]
		]
	]

run[fm_, opts_] :=
	Module[{ visitor, tyEnv, pm = fm["programModule"]},
		tyEnv = pm["typeEnvironment"];
		visitor = CreateInstructionVisitor[
			<| "tyEnv" -> tyEnv |>,
			<|
				"visitCallInstruction" -> visit
			|>,
			"IgnoreRequiredInstructions" -> True
		];
		visitor["traverse", fm];
	    fm
	]


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"ResolveSizeOf",
	"The pass resolves all size of types within a program module --- transforming SizeOf[Type[<<string>>]] to the byte count of the object " <>
	"if the size is known at compile time, otherwise it keeps the sizeof computation unchanged."
];

ResolveSizeOfPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run
|>];

RegisterPass[ResolveSizeOfPass]
]]

End[]
	
EndPackage[]
