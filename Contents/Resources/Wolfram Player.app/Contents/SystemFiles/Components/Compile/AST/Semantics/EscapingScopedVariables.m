
(**
 * This pass annotates each Function mexpr with "escapedScopeVariables" property.
 * The "escapedScopeVariables" property contains a list of variables which are 
 * used by a nested function, but is defined within the current function. For example,
 * 
 * mexpr = Function[{x}, Function[{y}, x + y]]
 *
 * then mexpr["getProperty", "escapedScopeVariables"] === {x}
 *
 * This pass is similar to the captureScopedVariables pass (which annotates the inner 
 * functions where the variable is used).
 *)

BeginPackage["Compile`AST`Semantics`EscapingScopedVariables`"]

MExprEscapingScopedVariables
MExprEscapingScopedVariablesPass



Begin["`Private`"] 

Needs["CompileAST`Utilities`MExprVisitor`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileUtilities`Callback`"]
Needs["Compile`Core`PassManager`MExprPass`"]
Needs["CompileAST`Class`Base`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["CompileUtilities`Asserter`Assert`"]


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
    "MExprEscapingScopedVariables",
    "Performs analysis to figure out which local variables are captured by another function. " <>
    "These are the variables which are used within an inner-scoped function, but are definined " <> 
    "within the current function. This analysis requires the MExprClosureVariablesPass to run, " <>
    "and an error is emitted otherwise."
];


MExprEscapingScopedVariablesPass = CreateMExprPass[<|
    "information" -> info,
    "runPass" -> MExprEscapingScopedVariables
|>];
RegisterPass[MExprEscapingScopedVariablesPass]
]]

visitSymbol[self_, mexpr_] := (
	If[mexpr["getProperty", "isClosureVariable", False],
	   AssertThat[
	          "Expecting the scopeBinder property to be set. " <>
	          "The closure variables pass requires the MExprBindingRewrite to be called on the MExpr",
	          mexpr
	        ]["named", "expr"
	        ]["satisfies", #["hasProperty", "scopeBinder"]&
	   ];
	   (* We only care about variables which are defined by a function *)
	   With[{
	       functionBinder = mexpr["getProperty", "functionBinder"]
	   },
	       If[functionBinder === None, (* Does not make sense *)
	            Return[mexpr]
	       ];
	       
           If[!functionBinder["hasProperty", "escapedScopeVariables"],
                functionBinder["setProperty", "escapedScopeVariables" -> {}]
           ];
           functionBinder["setProperty", "escapedScopeVariables" ->
                 Append[
                    functionBinder["getProperty", "escapedScopeVariables"],
                    mexpr
                 ]
           ]
	   ];
	];
	mexpr
);


RegisterCallback["DeclareCompileClass", Function[{st},
DeclareClass[MExprClearEscapingScopedVariableProperties,
    <| 
        "visitNormal"   -> (#["removeProperty", "escapedScopeVariables"]&)
    |>,
    {},
    Extends -> {MExprVisitorClass}
];
]]

RegisterCallback["DeclareCompileClass", Function[{st},
DeclareClass[MExprEscapingScopedVariableCollector,
    <| 
        "visitSymbol"   -> (visitSymbol[Self, #] &)
    |>,
    {
        "escapingVars"
    },
    Extends -> {MExprVisitorClass}
];
]]

MExprEscapingScopedVariables[mexpr_?MExprQ, opts_:<||>] :=
    With[{
       clearer = CreateObject[MExprClearEscapingScopedVariableProperties],
       visitor = CreateObject[MExprEscapingScopedVariableCollector,
            <|
                "escapingVars" -> CreateReference[{}]
            |>
       ]
    },
       mexpr["accept", clearer];
       mexpr["accept", visitor];
       mexpr
    ]
    

End[]

EndPackage[]
