
BeginPackage["Compile`Core`Analysis`Function`FunctionThrows`"]

FunctionThrowsPass

Begin["`Private`"]

Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`ProgramModulePass`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Reference`"]


(*
Algorithm for determining whether a function can throw an exception, or calls a function that can throw an exception.

We have a set of functions which call each other.  Each function can also call an external function that can throw.


1) Each function that calls an external function that throws should be marked as throwing.

2) Make a map from each function to functions that call it

3) Make a work list of all functions


Loop until worklist is empty.
{
Take fun from worklist.

If fun marked as throwing an exception
  Loop over parents
  {
    If parent is not marked as throwing
      mark parent as throwing and put into worklist
  }

}

*)


visitConstantFunction[ state_, inst_, funValue_] :=
    Module[ {current = state["current"]["get"],  ext, calledFun, ent},
        ext = state["programModule"]["externalDeclarations"]["lookupFunction", funValue];
        calledFun = state["programModule"]["getFunctionModule", funValue];
        Which[ 
            !MissingQ[ext] && TrueQ[Lookup[ext, "Throws", False]],
                setThrows[current, True];
                inst["setProperty", "Throws" -> True];
            ,
            FunctionModuleQ[calledFun],
                ent = state["parentMap"]["lookup", calledFun["id"], Null];
                If[ ent === Null,
                    ent = CreateReference[<||>];
                    state["parentMap"]["associateTo", calledFun["id"] -> ent]
                ];
                ent["associateTo", current["id"] -> current];
            ,
            True,
                Null];
            
    ]

visitCall[state_, inst_] :=
    Module[ {fun, current = state["current"]["get"]},
        fun = inst["function"];
        If[ ConstantValueQ[fun],
            visitConstantFunction[ state, inst, fun["value"]]
            ,
            setThrows[current, True];
            inst["setProperty", "Throws" -> True]];
    ]
    
throws[fm_?FunctionModuleQ] :=
    TrueQ[fm["information"]["Throws"]]
    
throws[___] := False

setThrows[fm_?FunctionModuleQ, val_] :=
    fm["information"]["setThrows", val]

functionThrows[ state_, fun_?ConstantValueQ] :=
    Module[ {ext, calledFun},
        ext = state["programModule"]["externalDeclarations"]["lookupFunction", fun["value"]];
        calledFun = state["programModule"]["getFunctionModule", fun["value"]];
        Which[ 
            !MissingQ[ext],
                TrueQ[Lookup[ext, "Throws", False]]
            ,
            FunctionModuleQ[calledFun],
                throws[calledFun]
            ,
            True,
                False
        ]
    ]

functionThrows[state_, fun_] :=
    False

finishCall[ state_, inst_] :=
    Module[ {},
        If[functionThrows[state, inst["function"]],
            inst["setProperty", "Throws" -> True]
        ];
    ]


createVisitor[state_] :=
    CreateInstructionVisitor[
        state,
        <|
            "visitCallInstruction" -> visitCall
        |>,
        "IgnoreRequiredInstructions" -> True
    ]

setupFunction[ state_, visitor_, fm_] :=
    Module[ {ent},
        ent = state["parentMap"]["lookup", fm["id"], Null];
        If[ ent === Null,
            state["parentMap"]["associateTo", fm["id"] -> CreateReference[<||>]]
        ];
        If[MissingQ[fm["information"]["Throws"]],
            setThrows[fm, False]
        ];
        state["current"]["set", fm];
        addToWorkList[state, fm];
        visitor["traverse", fm];
    ]

addToWorkList[state_, elem_] :=
    If[!state["workMap"]["keyExistsQ", elem["id"]],
        state["workMap"]["associateTo", elem["id"] -> elem];
        state["workList"]["appendTo", elem]
    ];

popFromWorkList[ state_] :=
    Module[ {elem = state["workList"]["popFront"]},
        state["workMap"]["keyDropFrom", elem["id"]];
        elem
    ]

run[pm_, opts_] := 
    Module[{state, visitor, work, parents},
        state = <| 
            "programModule" -> pm, 
            "current" -> CreateReference[],
            "parentMap" -> CreateReference[<||>], 
            "workMap" -> CreateReference[<||>],
            "workList" -> CreateReference[{}]
        |>;
        visitor = createVisitor[state];
        pm["scanFunctionModules",
            setupFunction[state, visitor, #]&
        ];
        While[ state["workList"]["length"] > 0,
            work = popFromWorkList[state];
            If[ throws[work],
                parents = state["parentMap"]["lookup", work["id"]]["values"];
                Scan[
                    If[ !throws[#],
                        setThrows[#, True];
                        addToWorkList[state, #]
                    ]&
                    ,
                    parents
                ]
             ];
        ];
        visitor = CreateInstructionVisitor[
            state,
            <|
                "visitCallInstruction" -> finishCall
            |>,
            "IgnoreRequiredInstructions" -> True
        ];
        visitor["traverse", pm];
        pm
    ]


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
    "FunctionThrows",
    "This pass fills out Throws Function property based on leaf functions."
];

FunctionThrowsPass = CreateProgramModulePass[<|
    "information" -> info,
    "runPass" -> run,
    "passClass" -> "Analysis"
|>];

RegisterPass[FunctionThrowsPass];

]]

End[]

EndPackage[]
