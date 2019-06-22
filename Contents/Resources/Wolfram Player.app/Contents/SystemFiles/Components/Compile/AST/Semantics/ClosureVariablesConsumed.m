
(**
 * This pass annotates each Function mexpr with "closureVariablesConsumed" property.
 * The "closureVariablesConsumed" property contains a list of variables which are 
 * used the current function, but are defined within an outer function. For example,
 * 
 * mexpr = Function[{x}, Function[{y}, x + y]]
 *
 * then mexpr["part", 2]["getProperty", "closureVariablesConsumed"] === {x}
 *
 * This pass is similar to the escapedScopeVariables pass (which annotates the outer 
 * functions where the variable is defined).
 *)

BeginPackage["Compile`AST`Semantics`ClosureVariablesConsumed`"]

MExprClosureVariablesConsumed
MExprClosureVariablesConsumedPass


Begin["`Private`"] 


Needs["CompileAST`Utilities`MExprVisitor`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`PassManager`MExprPass`"]
Needs["CompileAST`Class`Base`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
    "MExprClosureVariablesConsumed",
    "Performs analysis to figure out which variables are captured by a function. " <>
    "These are the variables which are used within the function, but are definined " <> 
    "in an outer scoped function. This analysis requires the MExprClosureVariablesPass to run, " <>
    "and an error is emitted otherwise."
];


MExprClosureVariablesConsumedPass = CreateMExprPass[<|
    "information" -> info,
    "runPass" -> MExprClosureVariablesConsumed
|>];
RegisterPass[MExprClosureVariablesConsumedPass]
]]

visitSymbol[self_, mexpr_] := (
    If[mexpr["getProperty", "isClosureVariable", False],
        self["capturedVars"]["appendTo", mexpr]
    ];
    mexpr
);

visitFunction[self_, mexpr_] :=
    Module[{
        oldEscapedVars = self["capturedVars"],
        boundNames = #["name"]& /@ mexpr["getProperty", "boundVariables", {}],
        captured
    },
        self["setCapturedVars", CreateReference[{}]];
        mexpr["part", 2]["accept", self];
        captured = Select[
            self["getCapturedVars"]["get"],
            FreeQ[boundNames, #["name"]]&
        ];
        mexpr["setProperty", "closureVariablesConsumed" -> captured];
        self["setCapturedVars", oldEscapedVars];
        mexpr
    ];
    
visitNormal[self_, mexpr_] :=
    If[mexpr["hasHead", Function] || mexpr["hasHead", Compile],
        visitFunction[self, mexpr],
        mexpr
    ];


RegisterCallback["DeclareCompileClass", Function[{st},
DeclareClass[MExprClearClosureVariableConsumed,
    <| 
        "visitNormal"   -> (#["removeProperty", "closureVariablesConsumed"]&)
    |>,
    {},
    Extends -> {MExprVisitorClass}
];
]]

RegisterCallback["DeclareCompileClass", Function[{st},
DeclareClass[MExprClosureVariableConsumedCollector,
    <| 
        "visitSymbol"   -> (visitSymbol[Self, #] &),
        "visitNormal"   -> (visitNormal[Self, #] &)
    |>,
    {
        "capturedVars"
    },
    Extends -> {MExprVisitorClass}
];
]]

MExprClosureVariablesConsumed[mexpr_?MExprQ, opts_:<||>] :=
    With[{
       clearer = CreateObject[MExprClearClosureVariableConsumed],
       visitor = CreateObject[MExprClosureVariableConsumedCollector,
            <|
                "capturedVars" -> CreateReference[{}]
            |>
       ]
    },
       mexpr["accept", clearer];
       mexpr["accept", visitor];
       mexpr
    ]
    

End[]

EndPackage[]
