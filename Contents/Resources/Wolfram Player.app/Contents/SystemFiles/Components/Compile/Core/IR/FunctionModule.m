BeginPackage["Compile`Core`IR`FunctionModule`"]

(**
a set of arguments and basic blocks
*)
FunctionModule;
FunctionModuleClass;
CreateFunctionModule;
FunctionModuleQ;

$TopBasicBlock
$LastBasicBlock

DeserializeFunctionModule

Begin["`Private`"]

Needs["Compile`Utilities`DataStructure`PointerGraph`"]
Needs["Compile`Core`IR`Internal`FunctionModuleTraversal`"]
Needs["Compile`Core`IR`BasicBlock`"]
Needs["Compile`Core`IR`GetElementTrait`"]
Needs["Compile`Core`Utilities`Visualization`DotGraphics`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Utilities`Serialization`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Debug`Logger`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Asserter`Assert`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`PassManager`PassRunner`"]
Needs["CompileUtilities`Markup`"]
Needs["Compile`Core`IR`FunctionInformation`"]


logger := logger = CreateLogger["FunctionModule", "ERROR"]


(** Name of the starting basic block -- a constant *)
$TopBasicBlock = "start"
$LastBasicBlock = "end"

$FunctionModuleId = 1;

getFunctionModuleId[] :=
	$FunctionModuleId++

RegisterCallback["DeclareCompileClass", Function[{st},
FunctionModuleClass = DeclareClass[
	FunctionModule,
	<|
		"clone" -> (clone[Self, ##]&),
		"cloneFull" -> (cloneFull[Self, ##]&),
		"getBasicBlocks" -> Function[{},
			Module[{bbs = Internal`Bag[]},
				Self["topologicalOrderScan", Internal`StuffBag[bbs, #]&];
				Internal`BagPart[bbs, All]
			]
		],
		"getBasicBlockIds" -> Function[{},
			Module[{bbs = Internal`Bag[]},
				Self["topologicalOrderScan", Internal`StuffBag[bbs, #["id"]]&];
				Internal`BagPart[bbs, All]
			]
		],
		"linkBasicBlock" -> (linkBasicBlock[Self, #]&),
		"unlinkBasicBlock" -> (unlinkBasicBlock[Self, #]&),
		"getInstructions" -> Function[{},
			Module[{insts = Internal`Bag[]},
				Self["topologicalOrderScan", Internal`StuffBag[insts, #["getInstructions"]]&];
				Flatten[Internal`BagPart[insts, All]]
			]
		],
        "getBasicBlock" -> Function[{id},
        	   With[{
        	   	   bb = Self["basicBlockMap"]["lookup", id]
        	   },
        	       AssertThat["basic block with the id requested exists", bb
                  ]["named", "basic block"
                  ]["satisfies", !MissingQ[#]&
               ];
        	       If[MissingQ[bb],
        	       	   ThrowException[CompilerException[{"Cannot find basic block with id = ", id}]],
        	       	   bb
        	       ]
        	   ] 
        ],
		"getInstruction" -> Function[{id},
			Module[{bbs, inst = $Failed},
				(**< This is a bit too much check
					AssertThat["The instruction id is valid.",
						id]["named", "instruction id"]["isMemberOf", #["id"]& /@ Self["getInstructions"]];
				*)
				bbs = Self["basicBlockMap"]["values"];
				Scan[
					Function[{bb},
						AssertThat["instruction is not found", inst
						  ]["named", "instruction"][
						    "isEqualTo", $Failed
						];
						inst = bb["getInstruction", id];
						If[!FailureQ[inst],
							Return[inst]
						]
					],
					bbs
				];
				inst
			]
		],
		"visitedQ" -> Function[{}, Self["visited"]],
		"clearVisited" -> Function[{},
			Self["topologicalOrderScan", #["clearVisited"]&];
			Self["setVisited", False]
		],
		"controlFlowGraph" -> Function[{}, controlFlowGraph[Self]],
		"controlFlowPointerGraph" -> Function[{}, controlFlowPointerGraph[Self]],
		
        "runPass" -> (runPass[Self, ##]&),
		"getElements" -> Function[ {},  Self["getBasicBlocks"]],
		"makePrettyPrintBoxes" -> Function[{},
			With[{bbs = Self["getBasicBlocks"]},
				RowBox[
				    Riffle[
				    		#["makePrettyPrintBoxes"]& /@ bbs,
				    		"\[NewLine]"
					]
				]
			]
		],
		"exprTypeQ" ->  Function[{},
			Self["programModule"]["exprTypeQ"]
		],
		"disposeNotElements" -> Function[{}, disposeNotElements[Self]],
		"dispose" -> Function[{}, dispose[Self]],
		"checkLast" -> (checkLast[Self,#1,#2]&),
		"serialize" -> (serialize[ Self, #]&),
        "toString" -> (toString[Self]&),
        "toHTML" -> (toHTML[Self]&),
		"toDot" -> Function[{}, ToDotGraphics[Self]],
		"toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
		"initialize" -> Function[{},
			Self["setType", Undefined];
			Self["setProperties", CreateReference[<||>]];
			Self["setBasicBlockMap", CreateReference[<||>]];
			Self["setUnlinkedBasicBlocks", CreateReference[{}]];
		]
	|>,
	{
		"id" -> 0,
		"name" -> "unknownFun",
		"mexpr",
		"programModule" -> None,
		"firstBasicBlock" -> None,
		"lastBasicBlock" -> None,
		"basicBlockMap",
		"unlinkedBasicBlocks",
		"visited" -> False,
		"properties",
		"information" -> None,
		"result" -> None,
		"arguments" -> {},
		"typeEnvironment",
		"type",
		"bodyType" -> Null,
		"metaData" -> Null
	},
	Predicate -> FunctionModuleQ,
	Extends -> {
		FunctionModuleTraversalTrait,
		GetElementTrait,
		ClassPropertiesTrait
	}
]
]]


CreateFunctionModule[] :=
	createFunctionModule[]

CreateFunctionModule[name_] :=
	createFunctionModule[name]

CreateFunctionModule[pm_, name_, firstBB_, lastBB_, mexpr_, result_, arguments_] :=
	createFunctionModule[pm, name, firstBB, lastBB, mexpr, result, arguments]

CreateFunctionModule[args___] :=
	ThrowException[{"Unrecognized call to CreateFunctionModule", {args}}]

createFunctionModule[] :=
	CreateObject[FunctionModule]

createFunctionModule[name_] :=
	Module[{fm = CreateObject[FunctionModule, <|"name" -> name|>]},
		fm["setInformation", CreateFunctionInformation[fm]];
		fm
	]
	
createFunctionModule[pm_, name_, firstBB_, lastBB_, mexpr_, result_, arguments_] :=
	Module[{fm},
		fm = CreateObject[FunctionModule, <|
			"id" -> getFunctionModuleId[],
			"programModule" -> pm,
			"name" -> name,
			"firstBasicBlock" -> firstBB,
			"lastBasicBlock" -> lastBB,
			"mexpr" -> mexpr,
			"result" -> result,
			"arguments" -> arguments
		|>];
		fm["setInformation", CreateFunctionInformation[fm]];
		If[mexpr =!= None,
			If[mexpr["hasProperty", "sourceFilePath"],
				fm["setProperty", "sourceFilePath" -> mexpr["getProperty", "sourceFilePath"]];
			];
		];
		linkBasicBlocks[fm];
		fm
	]

runPass[pm_, passes_?ListQ, opts:OptionsPattern[]] :=
    RunPasses[passes, pm, opts]
runPass[pm_, pass_, opts:OptionsPattern[]] :=
    RunPass[pass, pm, opts]
runPass[pm_, args___] :=
    ThrowException[{"Invalid call to runPass. Expecting a pass or a list of passes, but not ", {args}}]
    
DeserializeFunctionModule[ env_, "FunctionModule"[ data_]] :=
	deserialize[env, data, data["basicBlocks"]]

deserialize[ env_, data_, bbsList_] :=
	Module[ {fm, bbs, result, arguments, type, newId, info},
		result = env["getElement", data, "result"];
		arguments = env["getElementList", data, "arguments"];
		bbs = Map[ env["deserialize", #]&, bbsList];
		Scan[#["fixLinks", env]&, bbs];
		fm = CreateFunctionModule[ None, data["name"], First[bbs], Last[bbs], None, result, arguments];
		newId = If[ env["uniqueID"], getFunctionModuleId[], data["id"]];
		fm["setId", newId];

		type = env["getType", data["type"]];
		fm["setType", type];
		CreateInstructionVisitor[
			env,
			<|
				"visitPhiInstruction" -> (#2["fixBasicBlocks", #1]&),
				"visitBranchInstruction" -> (#2["fixBasicBlocks", #1]&)
			|>,
			fm,
			"IgnoreRequiredInstructions" -> True
		];
		If[KeyExistsQ[data, "information"],
		    info = env["deserialize", data["information"]];
		    fm["setInformation", info]
		];
		fm
	]


serialize[ self_, env_] :=
	Module[ {data},
		data = <|"name" -> self["name"], 
				  "id" -> self["id"],
				  "basicBlocks" -> Map[ #["serialize", env]&, self["getBasicBlocks"]],
				  "result" -> self["result"]["serialize", env],
				  "arguments" -> Map[ #["serialize", env]&, self["arguments"]],
				  "type" -> env["serializeType", self["type"]]
		|>;
		If[self["information"] =!= None,
		    data["information"] = self["information"]["serialize", env]
		];
		"FunctionModule"[ data]
	]


clone[self_, env_] := 
	deserialize[env, self, self["getBasicBlocks"]]

clone[self_] :=
	clone[self, CreateCloneEnvironment[]] 


cloneFull[self_, name_] :=
	Module[ {fm1, pm, props},
		pm = self["programModule"];
		fm1 = self["clone"];
		fm1["setName", name];
		fm1["setProgramModule", pm];
		pm["addFunctionModule", fm1];
		fm1["setTypeEnvironment", self["typeEnvironment"]];
		props = self["getProperties"];
		KeyValueMap[ fm1["setProperty", #]&, props];
		fm1
	]

disposeNotElements[ self_] :=
	Module[{},
		self["setFirstBasicBlock", None];
		self["dispose"]
	]

dispose[self_] :=
	Module[{},
		If[ self["firstBasicBlock"] =!= None,
			Scan[ #["dispose"]&, self["getBasicBlocks"]]];
		Scan[ #["dispose"]&, self["unlinkedBasicBlocks"]["get"]];
		self["information"]["dispose"];
		self["setTypeEnvironment", Null];
		self["setProgramModule", Null];
		self["properties"]["set", <||>];
	]


(*
Populate fm["basicBlockMap"] with bb["id"] -> bb, starting with fm["firstBasicBlock"]
Insertion order into fm["basicBlockMap"] is random

It's nice to not use any built-in methods for FunctionModule, such as the old
implementation using fm["scanBasicBlocks"], because the FunctionModule is still being
constructed at this point and a chicken-and-egg problem may occur if e.g., fm["scanBasicBlocks"]
changes to rely on fm["basicBlockMap"] already being initialized.
*)
linkBasicBlocks[fm_] :=
Module[{first, basicBlockMap, nodesToVisit, currentId, current, childId},

	first = fm["firstBasicBlock"];
	basicBlockMap = fm["basicBlockMap"];

	nodesToVisit = CreateReference[<||>];
	nodesToVisit["associateTo", first["id"] -> first];

	While[!nodesToVisit["isEmpty"],
		(* no guaranteed order from firstKey *)
		currentId = nodesToVisit["firstKey"];
		current = nodesToVisit["lookup", currentId];
		nodesToVisit["keyDropFrom", currentId];

		Do[
			childId = child["id"];
			If[!basicBlockMap["keyExistsQ", childId],
				nodesToVisit["associateTo", childId -> child]
			]
			,
			{child, current["getChildren"]}
		];

		current["setFunctionModule", fm];
		basicBlockMap["associateTo", currentId -> current];
	];
]



unlinkBasicBlock[ fm_, bb_] :=
	(
	If[ AnyTrue[{fm["firstBasicBlock"], fm["lastBasicBlock"]}, #["sameQ", bb]&], 
		ThrowException[{"Cannot unlink first or last block"}]
	];
	fm["basicBlockMap"]["keyDropFrom", bb["id"]];
	fm["unlinkedBasicBlocks"]["appendTo", bb];
	)

linkBasicBlock[ fm_, bb_] :=
	(
	bb["setFunctionModule", fm];
	fm["basicBlockMap"]["associateTo", bb["id"] -> bb]
	)


(*
  If bb2 is the lastBasicBlock then switch to bb1.
*)
checkLast[fm_,bb1_,bb2_] :=
	If[fm["lastBasicBlock"]["sameQ", bb2], 
		fm["setLastBasicBlock", bb1]]


controlFlowGraph[fm_] :=
    Module[{verts = {}, edges = {}},
        fm["depthFirstScan",
               Function[{bb},
                  AppendTo[verts, bb["id"]];
                  edges = Join[edges, DirectedEdge[bb["id"], #]& /@ bb["getChildrenIds"]]  
               ]
        ];
        Graph[
        	   verts,
        	   edges
        	]   
    ];


controlFlowPointerGraph[fm_] :=
	iControlFlowPointerGraph[fm["getBasicBlocks"]]

iControlFlowPointerGraph[bbs_] :=
	Module[{pg, children},
		pg = CreatePointerGraph[];
		Do[
			children = bb["getChildrenIds"];
			AddVertexProperty[pg, bb["id"], "Tooltip" -> bb["toString"]];
			Scan[AddEdge[pg, DirectedEdge[bb["id"], #]]&, children]
			,
			{bb, bbs}
		];
		pg
	]
	
toStringHeader[fm_] :=
	StringJoin[
        If[fm["information"] =!= None,
            StringJoin[{
               BoldGreenText[fm["name"]],
               GrayText["::Information="] ,
               fm["information"]["toString"],
               "\n"
            }],
            {}   
        ],
		BoldGreenText[fm["name"]],
	    If[fm["type"] === Undefined,
	        "",
	        " : " <> BoldRedText[fm["type"]["name"]]
	    ],
	    If[fm["hasProperty", "closureVariablesConsumed"] && fm["getProperty", "closureVariablesConsumed"]["length"] > 0,
	       StringJoin[{
	           GrayText["\t\t  (* Consumed "] ,
	           StringRiffle[#["toString"]& /@ fm["getProperty", "closureVariablesConsumed"]["get"], ", "],
	           GrayText[" *)"] 
	       }],
	       {}
	    ],
	    If[fm["hasProperty", "closureVariablesProvided"] && Length[fm["getProperty", "closureVariablesProvided"]] > 0,
	    	StringJoin[{
	           GrayText["\t\t  (* Provided "] ,
	           StringRiffle[#["toString"]& /@ fm["getProperty", "closureVariablesProvided"], ", "],
	           GrayText[" *)"] 
	        }],
	    	{}
	    ]
	]

toString[fm_] := StringJoin[
    toStringHeader[fm],
	"\n",
	#["toString"]& /@ fm["getBasicBlocks"]
]

toHTML[self_] :=
    Block[{$FrontEnd = Null, $UseANSI = False, $UseHTML = True},
        StringReplace[
            self["toString"],
            "\n" -> "<br/>"
        ]
    ]
    
(**************************************************)
(**************************************************)

icon := Graphics[Text[
  Style["Fun\nMod", GrayLevel[0.7], Bold, 
   CurrentValue["FontCapHeight"]/
     AbsoluteCurrentValue[Magnification]]], $FormatingGraphicsOptions];


toBoxes[fm_?FunctionModuleQ, fmt_] :=
	BoxForm`ArrangeSummaryBox[
		"FunctionModule",
		fm,
  		icon,
  		{
  		    BoxForm`SummaryItem[{"name: ", fm["name"]}],
  		    If[fm["type"] === Undefined,
  		    	   Nothing,
  		    	   BoxForm`SummaryItem[{"type: ", fm["type"]["toString"]}]
  		    ]
  		},
  		Join[
  			MapIndexed[
  				BoxForm`SummaryItem[{"Argument "<> ToString[ First[#2]] <> ": ", #1}]&,
  				fm["getArguments"]
  			],
  			{
  				BoxForm`SummaryItem[{"Result: ", fm["getResult"]}],
				If[fm["type"] === Undefined,
  			   		BoxForm`SummaryItem[{"Type: ", Undefined}],
  			   		BoxForm`SummaryItem[{"Type: ", fm["type"]["name"]}]
  			   	]
  			},
            If[fm["information"] =!= None,
               {
                   BoxForm`SummaryItem[{"Information: ", fm["information"]}]
               },
               {}
            ],
  			If[fm["hasProperty", "closureVariablesConsumed"],
  			   {
  			       BoxForm`SummaryItem[{"Consumed Variables: ", fm["getProperty", "closureVariablesConsumed"]}]
  			   },
  			   {}
  			],
  			Map[BoxForm`SummaryItem[{LazyFormat[BasicBlockQ, #]}] &, fm["getBasicBlocks"]]
  		]
  		, 
  		fmt,
		"Interpretable" -> False
  	]



	
End[]
EndPackage[]
