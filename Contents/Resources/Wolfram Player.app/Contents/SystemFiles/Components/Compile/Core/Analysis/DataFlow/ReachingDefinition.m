(**
  * A defintion $d reaches point $p if these is some path from $d to $p
  * where $d is not redefined. This is forward may dataflow problem.
  *)

BeginPackage["Compile`Core`Analysis`DataFlow`ReachingDefinition`"]

ReachingDefinitionPass;

Begin["`Private`"] 

Needs["CompileUtilities`Debug`Logger`"]
Needs["Compile`Utilities`DataStructure`PointerGraph`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`IR`BasicBlock`"]
Needs["Compile`Core`Analysis`DataFlow`Def`"]
Needs["Compile`Core`Analysis`DataFlow`Use`"]
Needs["CompileUtilities`Callback`"]




logger := logger = CreateLogger["ReachingDefinition", "TRACE"]


show[fm_?FunctionModuleQ] :=
	Module[{pg},
		pg = CreatePointerGraph[];
		fm["topologicalOrderScan",
			Function[{bb},
				AddVertex[pg, bb["id"]];
				AddVertexProperty[
					pg,
					bb["id"],
					"Tooltip" ->
						Column[{
							Row[{"Instruction:\n", 
								bb["toString"]
							}],
							Row[{"ReachingDefinitions(In):\n", 
								StringJoin[
									Riffle[
										#["toString"]& /@ bb["getProperty", "reachingDefinitions[in]"],
										"\n"
									]
								]
							}]
						}]
				];
				Scan[AddEdge[pg, DirectedEdge[bb["id"], #]]&, bb["getChildrenIds"]]
			]
		];
		ToGraph[pg, fm["firstBasicBlock"]["id"]]
	]
	
initialize[fm_?FunctionModuleQ, opts_] :=
	fm["scanBasicBlocks",
		Function[{bb},
			bb["setProperty", "reachingDefinitions[in]" -> {}];
			bb["setProperty", "reachingDefinitions[out]" -> {}]
		]
	]
finalize[fm_?FunctionModuleQ, opts_] := (
	fm["scanBasicBlocks",
		Function[{bb},
			bb["setProperty", "reachingDefinitions[in]" ->
				DeleteDuplicates[
					Table[
						def["target"],
						{def, bb["getProperty", "reachingDefinitions[in]"]}
					],
					#1["id"] === #2["id"]&
				]
			];
			bb["setProperty", "reachingDefinitions[out]" ->
				DeleteDuplicates[
					Table[
						def["target"],
						{def, bb["getProperty", "reachingDefinitions[out]"]}
					],
					#1["id"] === #2["id"]&
				]
			];
		]
	];
	fm["setProperty", "reachingDefinitionsGraph" -> show[fm]]
)


run[fm_?FunctionModuleQ, opts_] :=
	Module[{changed, old},
		changed = True;
		While[changed === True,
			changed = False;
			fm["postTopologicalOrderScan",
				Function[{bb},
					old = bb["getProperty", "reachingDefinitions[in]"];
					bb["setProperty", "reachingDefinitions[in]" ->
						Apply[
							Union,
							Table[
								pred["getProperty", "reachingDefinitions[out]"],
								{pred, bb["getParents"]}
							]
						]
					];
					bb["setProperty", "reachingDefinitions[out]" ->
						Union[
							gen[bb],
							Complement[
								bb["getProperty", "reachingDefinitions[in]"],
								kill[bb]
							]
						]
					];
					If[old =!= bb["getProperty", "reachingDefinitions[in]"],
						changed = True
					]
				]
			]
		];
		fm
	]

gen[bb_?BasicBlockQ] :=
	Module[{addGen, genSet = {}},
		addGen[st_, inst_] := AppendTo[genSet, inst["id"]];
		CreateInstructionVisitor[
			<|
				"visitBinaryInstruction" -> addGen,
				"visitUnaryInstruction" -> addGen,
				"visitCompareInstruction" -> addGen,
				"visitCallInstruction" -> addGen,
				"visitLoadInstruction" -> addGen,
				"visitLoadArgumentInstruction" -> addGen,
				"visitLoadGlobalInstruction" -> addGen,
				"visitCopyInstruction" -> addGen,
				"visitSelectInstruction" -> addGen
			|>,
			bb
		];
		genSet
	]

kill[bb_?BasicBlockQ] :=
	Module[{addKill, killSet = {}},
		addKill[st_, inst_] := Join[
			killSet,
			Complement[ (**< The definition kills only all preceeding definitions, 
	                                 but we reuse the defs pass to geenrate a bigger kill
	                                 set. This does not matter in the analysis because of
					 properties in the flow equation *) 
				{inst["target"]["def"]},
				{inst["id"]}
			]
		];
		CreateInstructionVisitor[
			<|
				"visitBinaryInstruction" -> addKill,
				"visitUnaryInstruction" -> addKill,
				"visitCompareInstruction" -> addKill,
				"visitCallInstruction" -> addKill,
				"visitLoadInstruction" -> addKill,
				"visitCopyInstruction" -> addKill,
				"visitLoadArgumentInstruction" -> addKill,
				"visitLoadGlobalInstruction" -> addKill,
				"visitSelectInstruction" -> addKill
			|>,
			bb
		];
		killSet
	]
	
(**********************************************************)
(**********************************************************)
(**********************************************************)




RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"ReachingDefinition",
	"Computes the reaching defintion of each basic block in the function module",
	"A definition `a` reaches a use `b` if it is used at `b` (or subsequent instructions) and there is no intervening definitions."
];

ReachingDefinitionPass = CreateFunctionModulePass[<|
	"information" -> info,
	"runPass" -> run,
	"initializePass" -> initialize,
	"finalizePass" -> finalize,
	"requires" -> {
		UsePass,
		DefPass
	}
|>];

RegisterPass[ReachingDefinitionPass]
]]


End[] 

EndPackage[]
