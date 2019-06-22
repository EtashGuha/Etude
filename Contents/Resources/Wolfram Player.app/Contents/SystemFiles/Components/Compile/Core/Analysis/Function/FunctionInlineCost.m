
BeginPackage["Compile`Core`Analysis`Function`FunctionInlineCost`"]

FunctionInlineCostPass

Begin["`Private`"]

Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`Analysis`Properties`FunctionCallLinkage`"]
Needs["CompileUtilities`Asserter`Assert`"]

cost = <|
  "Free" -> 0,
  "Arithmetic" -> 1,
  "Load" -> 3, 
  "LocalCall" -> 10, 
  "IntrinsicCall" -> 1,
  "LLVMCall" -> 10,
  "RuntimeCall" -> 20, 
  "ExternalCall" -> 50 
|>


callCost[st_, inst_] :=
    Module[{},
        If[!inst["hasProperty", "linkage"],
            Return["ExternalCall"]
        ];
        
        Switch[inst["hasProperty", "linkage"],
            "LLVMCompareFunction" | "LLVMCompileTools" | "LLVMInternal",
                "LLVMCall",
            "Runtime",
                "RuntimeCall",
             "Local",
                "LocalCall",
            _,
                "ExternalCall"
        ]
    ]

iRun[fm_?FunctionModuleQ, opts_:<||>] :=
    With[{
        costs = CreateReference[{}]
    },
        If[TrueQ[fm["getProperty", "entryQ"]],
            Return[Infinity]
        ];
        CreateInstructionVisitor[
            costs,
            <|
                "visitLabelInstruction" -> Function[{st, inst},
                    st["appendTo", "Free"];
                ],
                "visitStackAllocateInstruction" -> Function[{st, inst},
                    st["appendTo", "Free"];
                ],
                "visitLoadArgumentInstruction" -> Function[{st, inst},
                    st["appendTo", "Free"];
                ],
                "visitLoadInstruction" -> Function[{st, inst},
                    st["appendTo", "Load"];
                ],
                "visitStoreInstruction" -> Function[{st, inst},
                    st["appendTo", "Free"];
                ],
                "visitBinaryInstruction" -> Function[{st, inst},
                    st["appendTo", "Arithmetic"];
                ],
                "visitReturnInstruction" -> Function[{st, inst},
                    st["appendTo", "Free"];
                ],
                "visitBranchInstruction" -> Function[{st, inst},
                    st["appendTo", "Free"];
                ],
                "visitCompareInstruction" -> Function[{st, inst},
                    st["appendTo", "Arithmetic"];
                ],
                "visitCopyInstruction" -> Function[{st, inst},
                    st["appendTo", "Free"];
                ],
                "visitUnaryInstruction" -> Function[{st, inst},
                    st["appendTo", "Arithmetic"];
                ],
                "visitCallInstruction" -> Function[{st, inst},
                    st["appendTo", callCost[st, inst]];
                ],
                "visitGetElementInstruction" -> Function[{st, inst},
                    st["appendTo", "Load"];
                ],
                "visitSetElementInstruction" -> Function[{st, inst},
                    st["appendTo", "Free"];
                ],
                "visitInertInstruction" -> Function[{st, inst},
                    st["appendTo", callCost[st, inst]];
                ],
                "visitLambdaInstruction" -> Function[{st, inst},
                    st["appendTo", "Free"];
                ],
                "visitTypeCastInstruction" -> Function[{st, inst},
                    st["appendTo", "Free"];
                ],
                "visitLoadGlobalInstruction" -> Function[{st, inst},
                    st["appendTo", "Load"];
                ],
                "visitResumeInstruction" -> Function[{st, inst},
                    st["appendTo", "Free"];
                ],
                "visitSelectInstruction" -> Function[{st, inst},
                    st["appendTo", "Free"];
                ]
            |>,
            fm,
            "IgnoreRequiredInstructions" -> True
        ];
        Total[cost /@ costs["get"]]
    ]
    
run[fm_?FunctionModuleQ, opts_] :=
    With[{
        cost = iRun[fm, opts]
    },
        fm["information"]["inlineInformation"]["setInlineCost", cost];
        fm
    ];

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
        "FunctionInlineCost",
        "This pass computes the inline cost for each function."
];

FunctionInlineCostPass = CreateFunctionModulePass[<|
    "information" -> info,
    "runPass" -> run,
    "requires" -> {
        FunctionCallLinkagePass
    },
    "passClass" -> "Analysis"
|>];

RegisterPass[FunctionInlineCostPass]
]]

End[]

EndPackage[]
