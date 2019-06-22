(**
 * This pass annotates each Symbol mexpr with "isClosureVariable" property.
 * The "isClosureVariable" property is True if the function is defined by 
 * a function but used within another. Say the defining scope is "smexpr"
 * then the symbol is a closure variable if smexpr["scopeId"] < mexpr["functionScopeId"]
 *)

BeginPackage["Compile`AST`Semantics`ClosureVariables`"]

MExprClosureVariables
MExprClosureVariablesPass



Begin["`Private`"] 

Needs["CompileAST`Utilities`MExprVisitor`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`PassManager`MExprPass`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileAST`Class`Base`"]
Needs["Compile`Core`PassManager`PassInformation`"]



RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
    "MExprClosureVariables",
    "Performs analysis to figure out which variables are considered closure variables. This occurs " <>
    "when the function binding scope of the variable is greater than the scope id of the binding function." <>
    "I.e. the variable is being used in a function that's inner to the one where it's defined. " <>
    "This analysis requires the MExprBindingRewritePass to run, and an error is emitted otherwise."
];


MExprClosureVariablesPass = CreateMExprPass[<|
    "information" -> info,
    "runPass" -> MExprClosureVariables
|>];
RegisterPass[MExprClosureVariablesPass]
]]


visitSymbol[self_, mexpr_] := ( 
    With[{
        scopeBinder = mexpr["getProperty", "scopeBinder"],
        functionScopeId = mexpr["getProperty", "functionScopeId"]
    },
        If[MissingQ[scopeBinder],
            Return[mexpr]
        ];
        If[scopeBinder === None,
            Return[mexpr]
        ];
        With[{
            scopeBinderId = scopeBinder["getProperty", "scopeId"]
        },
            If[functionScopeId > scopeBinderId,
                mexpr["setProperty", "isClosureVariable" -> True];
            ];
            mexpr 
        ]
    ]
);


RegisterCallback["DeclareCompileClass", Function[{st},
DeclareClass[MExprClearClosureVariable,
    <| 
        "visitSymbol"   -> (#["removeProperty", "isClosureVariable"]&)
    |>,
    {},
    Extends -> {MExprVisitorClass}
];
]]

RegisterCallback["DeclareCompileClass", Function[{st},
DeclareClass[MExprClosureVariableCollector,
    <| 
        "visitSymbol"   -> (visitSymbol[Self, #] &)
    |>,
    {
        "closureVars"
    },
    Extends -> {MExprVisitorClass}
]
]]

MExprClosureVariables[mexpr_?MExprQ, opts_:<||>] :=
    With[{
       clearer = CreateObject[MExprClearClosureVariable],
       visitor = CreateObject[MExprClosureVariableCollector,
            <|
                "closureVars" -> CreateReference[{}]
            |>
       ]
    },
       mexpr["accept", clearer];
       mexpr["accept", visitor];
       mexpr
    ]
    

End[]

EndPackage[]
