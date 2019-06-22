BeginPackage["Compile`Core`PassManager`PassRunner`"]

$NumPassesRun
RunPasses;
RunPass;

Begin["`Private`"] 

Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileAST`Class`Base`"]
Needs["Compile`Core`PassManager`MExprPass`"]
Needs["Compile`Core`PassManager`CompiledProgramPass`"]
Needs["Compile`Core`PassManager`ProgramModulePass`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`BasicBlockPass`"]
Needs["Compile`Core`PassManager`LoopPass`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`PassManager`Pass`"]
Needs["Compile`Core`PassManager`PassLogger`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Asserter`Assert`"]
Needs["Compile`Core`IR`ProgramModule`"]
Needs["Compile`Core`IR`CompiledProgram`"]
Needs["CompileUtilities`Error`Suggestions`"]

ClearAll[RunPass]
ClearAll[RunPasses]

$NumPassesRun = 0

defaultlogger := defaultlogger = CreateDefaultPassLogger[ "WARN"]

getLogger[ opts_] :=
	Module[ {val},
		val = Lookup[opts, "PassLogger", Automatic];
		Which[ 
			val === None,
				Null,
			val === Automatic,
				defaultlogger,
			True,
				val
		]
	]


passDependencies[pass_] :=
    Join[pass["requires"], pass["preserves"]]
  
getPass[pass_?PassQ] :=
    pass;
getPass[passName_?StringQ] :=
    With[{
    	pass = Lookup[$Passes, passName]
    },
    	If[MissingQ[pass],
    		Lookup[$Passes,	StringTrim[passName, "Pass"], ThrowException[{"Invalid Pass name ", {passName}}]],
    		pass
    	]
    ]

getPass[args___] :=
	ThrowException[{"Invalid Pass name ", {args}}]

       
       
runPassDependencies[pass_, obj_, opts_:<||>] :=
	Module[{passes, res = obj, arg},
		passes = passDependencies[pass];
		If[passes =!= {},
			arg = getLogger[opts][ "prePass", pass, passes];
			Scan[
			   (res = RunPass[#, res, opts])&,
			   passes
			];
			getLogger[opts][ "prePassEnd", arg];
		];
		res
	]

runPostPasses[pass_, obj_, opts_:<||>] :=
	Module[{passes, res = obj, arg},
		passes = pass["postPasses"];
		If[passes =!= {},
			arg = getLogger[opts][ "postPass", pass, passes]; 
			Scan[
			   (res = RunPass[#, res, opts])&,
			   passes
			];
			getLogger[opts][ "postPassEnd", arg];
		];
		res
	]



(* This runs the passes in topological order.
 * It does not account for the fact that certain passes invalidate
 * previously run ones, and thus is not correct.
 * Extra information needs to be placed in the passes to signify whether
 * they destroy previosly run analysis (the point of the preserves field)
 *) 	
xRunPasses[passes_List, obj_, opts_:<||>] := 
	Module[{sortedPasses, grph},
		grph = passDependenceGraph[passes];
		Print[grph];
		sortedPasses = Map[If[ListQ[#], First[#], #]&, TopologicalSort[grph]];
		Scan[
			oRunPass[#, obj, opts]&,
			sortedPasses
		]
	]
passDependenceGraph[passes_List] := 
	Module[{grph, queue, verts, edges, deps, pass, pass0, makeTooltip, inputPass},
	 	(* We want to force each input pass to be unique --- this is a hacky way
	 	 * of doing just that
	 	 *) 
		verts = MapThread[inputPass, {passes, Range[Length[passes]]}];
		edges = {};
		queue = verts;
		While[Length[queue] > 0,
			pass0 = First[queue];
			queue = Drop[queue, 1];
			pass = If[Head[pass0] === inputPass,
				First[pass0],
				pass0
			];
			deps = passDependencies[pass];
			verts = {verts, pass0};
			edges = {edges, Map[DirectedEdge[#, pass0]&, deps]};
			queue = Join[queue, deps];
		];
		verts = DeleteDuplicates[Flatten[verts]];
		edges = DeleteDuplicates[Flatten[edges]];
		makeTooltip[inputPass[vert_, ii_]] :=
			With[{name = vert["name"]},
				Tooltip[Labeled[inputPass[vert, ii], name], name]
			];
		makeTooltip[vert_] /; !ListQ[vert] :=
			With[{name = vert["name"]},
				Tooltip[Labeled[vert, name], name]
			];
		grph = Graph[makeTooltip /@ verts, edges];
		grph
	]

(* Todo: Replace with xRunPasses once it's fixed *)

(*
  If obj is not a ProgramModule or if the name of the pass is not 
  in the passStore then run the pass.
  
  If the pass was run then if this is not an Analysis pass flush the 
  pass store.

  Add the name of this pass to the store.
  
  The pass is always run if the "Force" option is given
*)
RunPasses[{}, obj_, opts_?AssociationQ] := 
    obj
RunPasses[passes0_?ListQ, obj_, opts_?AssociationQ] :=
	Module[{passes = Flatten[passes0], force, res = obj, wasRun},
		Scan[
			Function[{pass0},
				Module[{pass = getPass[pass0]},
					If[!TrueQ[pass["constraint"][obj, opts]],
						Return[obj]
					];
					res = runPassDependencies[pass, res, opts];
					wasRun = False;
	                force = Lookup[opts, "Force", False];
					If[force || !ProgramModuleQ[obj] || !obj["passStore"]["keyExistsQ", pass["name"]],
					    wasRun = True;
						res = oRunPass[pass, res, opts]
				    ];
					If[ wasRun && ProgramModuleQ[obj],
					    If[pass["passClass"] =!= "Analysis",
						   obj["setPassStore", CreateReference[<||>]]
						];
						obj["passStore"]["associateTo", pass["name"] -> True]
				    ];
	(*				If[ProgramModuleQ[obj] && pass["name"] =!= "LintIR",
						res = oRunPass[Compile`Core`Lint`IR`LintIRPass, res, opts]];
	*)				
					res = runPostPasses[pass, res, opts];
				]
			],
			Select[passes, # =!= Null&]
		];
		res
	]
	
RunPasses[passes_, obj_, opts:OptionsPattern[]] :=
    RunPasses[passes, obj, <| opts |>]
RunPasses[args___] :=
    ThrowException[{"Invalid RunPasses call ", {args}}]
    
lookupPassString[p_?StringQ] :=  Which[
	KeyExistsQ[$Passes, p],
		$Passes[p],
	KeyExistsQ[$Passes, StringReplace[p, RegularExpression["Pass$"] -> ""]],
		$Passes[StringReplace[p, RegularExpression["Pass$"] -> ""]],
	True,
		Missing[{"pass not found", p, " Suggestions: ", Suggestions[p, Keys[$Passes]]}, p]
];
		 
lookupPassString[___] := False
  
passQ[p_] := PassQ[p] || PassQ[lookupPassString[p]]

mexprPassQ[p_] := MExprPassQ[p] || MExprPassQ[lookupPassString[p]]

compiledProgramPassQ[p_] := CompiledProgramPassQ[p] || CompiledProgramPassQ[lookupPassString[p]]

programModulePassQ[p_] := ProgramModulePassQ[p] || ProgramModulePassQ[lookupPassString[p]]

functionModulePassQ[p_] := FunctionModulePassQ[p] || FunctionModulePassQ[lookupPassString[p]]

basicBlockPassQ[p_] := BasicBlockPassQ[p] || BasicBlockPassQ[lookupPassString[p]]

loopPassQ[p_] := LoopPassQ[p] || LoopPassQ[lookupPassString[p]]

RunPass[pass_?passQ, obj_, opts:OptionsPattern[]] :=
    RunPass[pass, obj, <| opts |>]
RunPass[pass_?passQ, obj_, opts_?AssociationQ] :=
	RunPasses[{pass}, obj, opts]

passName[s_?StringQ] := s
passName[s_?PassQ] := s["name"]
passName[s_] := ToString[s]

RunPass[pass_, args___] /; !passQ[pass] :=
	ThrowException[{"Invalid pass ", pass, " suggestions: ", Suggestions[passName[pass], Keys[$Passes]], " called with args ", {args}}]
	
RunPass[args___] :=
	ThrowException[{"Invalid RunPass call ", {args}}]

oRunPass[pass_?passQ, obj_, opts_] :=
	Module[{arg, res},
		arg = getLogger[opts][ "startRunPass", pass, obj]; 
		res = iRunPass[pass, obj, opts];
		getLogger[opts]["endRunPass", pass, arg, res];
		$NumPassesRun++;
		res
	]

(*
  Calls the map so that the CP gets mutated since MExpr passes don't all work 
  by modifying but by returning new.  This could change.
*)
iRunPass[pass_?mexprPassQ, cp_?CompiledProgramQ, opts_] := 
	(
	cp["mapFunctionData", iRunPass[pass,#, opts]&];
	cp
	)

iRunPass[pass_?mexprPassQ, expr_?MExprQ, opts_] := 
	iRunPassWorker[pass, expr, " MExpr pass", opts]

iRunPass[pass_?compiledProgramPassQ, cp_?CompiledProgramQ, opts_] := 
	iRunPassWorker[pass, cp, " CompiledProgram pass", opts]

iRunPass[pass_?programModulePassQ, pm_ /; pm["isA", "ProgramModule"], opts_] :=
	iRunPassWorker[pass, pm, " program module pass", opts]
	
iRunPass[pass_?functionModulePassQ, pm_ /; pm["isA", "ProgramModule"], opts_] := (
	pm["scanFunctionModules", iRunPass[pass, #, opts]&];
	pm
)

iRunPass[pass_?loopPassQ, pm_ /; pm["isA", "ProgramModule"], opts_] := (
	pm["scanFunctionModules", iRunPass[pass, #, opts]&];
	pm
)
	
iRunPass[pass_?functionModulePassQ, fm_ /; fm["isA", "FunctionModule"], opts_] := 
	iRunPassWorker[pass, fm, " function module pass", opts]
	
iRunPass[pass_?loopPassQ, fm_ /; fm["isA", "FunctionModule"], opts_] := (
	If[MissingQ[fm["getProperty", "loopinformation"]],
		RunPass["LoopNestingForest", fm]
	];
	Module[{
		loopInfo = fm["getProperty", "loopinformation"]
	},
		iRunPassWorker[pass, loopInfo, " loop pass", opts]
	];
	fm
)
iRunPass[pass_?loopPassQ, info_ /; info["isA", "LoopInformation"], opts_] := (
	iRunPassWorker[pass, info, " loop pass", opts];
	If[Lookup[opts, "recursive", True],
		Do[
			If[!child["isA", "BasicBlock"],
				iRunPass[pass, child, opts]
			],
			{child, info["children"]}
		]
	];
	info	
)

iRunPass[pass_?basicBlockPassQ, pm_ /; pm["isA", "ProgramModule"], opts_] := (
	pm["scanFunctionModules", iRunPass[pass, #, opts]&];
	pm
)
iRunPass[pass_?basicBlockPassQ, fm_ /; fm["isA", "FunctionModule"], opts_] := (
	Switch[pass["traversalOrder"],
		"reversePostOrder",
			fm["reversePostOrderScan", iRunPass[pass, #, opts]&],
		"postOrder",
			fm["postOrderScan", iRunPass[pass, #, opts]&],
		"reversePreOrder",
			fm["reversePreOrderScan", iRunPass[pass, #, opts]&],
		"preOrder",
			fm["preOrderScan", iRunPass[pass, #, opts]&],
		"topologicalOrder",
			fm["topologicalOrderScan", iRunPass[pass, #, opts]&],
		"postTopologicalOrder",
			fm["postTopologicalOrderScan", iRunPass[pass, #, opts]&],
		"anyOrder",
			fm["topologicalOrderScan", iRunPass[pass, #, opts]&],
		_,
			AssertThat["The traversal order must be valid",
				pass["traversalOrder"]]["named",
				"traversalOrder"]["isMemberOf", {
					"reversePostOrder",
					"postOrder",
					"reversePreOrder",
					"preOrder",
					"topologicalOrder",
					"postTopologicalOrder",
					"anyOrder"
				}];
	];
	fm
)

iRunPass[pass_?basicBlockPassQ, bb_ /; bb["isA", "BasicBlock"], opts_] := 
	iRunPassWorker[pass, bb, " basic block pass", opts]


iRunPass[args___] := (
    ThrowException[{"Invalid iRunPass call ", {args}}]
)

iRunPassWorker[pass0_?passQ, arg_, text_, opts_] :=
	Module[{res, pass = pass0},
		If[StringQ[pass],
			pass = Lookup[$Passes, pass];
			If[MissingQ[pass],
				ThrowException[{"unable to find ", pass0 , " in the pass registry while running on ", arg, " Suggestions: ", Suggestions[pass, Keys[$Passes]]}]
			]
		];
		If[!TrueQ[pass["constraint"][arg, opts]],
			Return[arg]
		];
		If[pass["initializePass"] =!= Undefined,
			pass["initializePass"][arg, opts];
		];
		res = pass["runPass"][arg, opts];
		If[pass["finalizePass"] =!= Undefined,
			pass["finalizePass"][arg, opts];
		];
		If[!TrueQ[pass["verifyPass"][arg, opts]],
			ThrowException[{"Verification failed for ", pass , " running on ", arg}]
		];
		If[res === Null,
		   res = arg
		];
		AssertThat["The return type of the pass is a class", res
			]["named", pass["name"] <> " pass result"
			]["satisfies", ObjectInstanceQ
		];
		AssertThat["The return type of the pass is expected", res
			]["named",  pass["name"] <> " pass result"
			]["satisfiesAnyOf", {
			   MExprQ,
			   #["isA", "BasicBlock"]&,
			   #["isA", "FunctionModule"]&,
			   #["isA", "LoopInformation"]&,
			   #["isA", "ProgramModule"]&
			}
		];
		res
	]



End[] 

EndPackage[]
