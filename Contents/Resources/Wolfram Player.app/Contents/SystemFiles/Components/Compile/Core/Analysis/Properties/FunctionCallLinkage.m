
BeginPackage["Compile`Core`Analysis`Properties`FunctionCallLinkage`"]

FunctionCallLinkagePass

Begin["`Private`"]

Needs["CompileUtilities`Reference`"]
Needs["Compile`Core`IR`FunctionModule`"]
Needs["Compile`Core`PassManager`FunctionModulePass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`IR`ConstantValue`"]
Needs["Compile`Core`IR`Instruction`Utilities`InstructionVisitor`"]
Needs["Compile`Core`IR`Instruction`InstructionQ`"] (* for InstructionQ *)


run[st_, inst_?InstructionQ] :=
    Module[{
        pm = st["programModule"],
        funNames = st["funNames"],
        fun = inst["function"],
        funData, funName
    },
        If[!ConstantValueQ[fun],
            Return[]
        ];
        funName = fun["value"];

        funData = pm["externalDeclarations"]["lookupFunction", funName];
        Which[
            AssociationQ[funData] && KeyExistsQ[funData, "Linkage"],
                inst["setProperty", "linkage" -> funData["Linkage"]]
            ,
            MemberQ[funNames, funName],
                inst["setProperty", "linkage" -> "Local"],
            True,
                None
        ]
    ]


run[fm_?FunctionModuleQ, opts_:<||>] :=
    Module[{
        pm = fm["programModule"],
        funNames
    },
        funNames = #["name"]& /@ pm["getFunctionModules"];
        CreateInstructionVisitor[
            <|
                "programModule" -> fm["programModule"],
                "functionModule" -> fm,
                "funNames" -> funNames
            |>,
            <|
                "visitCallInstruction" -> run
            |>,
            fm,
            "IgnoreRequiredInstructions" -> True
        ];
        Total[cost /@ costs["get"]]
    ]
    

RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
        "FunctionCallLinkage",
        "This pass computes the linkage information for each call instruction. This information is used by other passes such as the FunctionInlineCostPass."
];

FunctionCallLinkagePass = CreateFunctionModulePass[<|
    "information" -> info,
    "runPass" -> run,
    "passClass" -> "Analysis"
|>];

RegisterPass[FunctionCallLinkagePass]
]]

End[]

EndPackage[]
