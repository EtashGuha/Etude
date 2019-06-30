(**
 * Scoping reassociates all the "binder" fields in the bound MExprs to
 * reference their binding environment. Two symbols with the same binding
 * environment will have the same name but will be different mexpr instances 
 *
 * An unbound Symbol will not have the "binder" property (this means
 * that the variable is global).
 *
 * For example:
 *     Module[{x}, Module[{x}, x]]
 * is actually
 *     Module[{x}, Module[{x1}, x1]]
 * where x1 are different instances with the same name. Renaming one will not rename the other.
 *
 * Care is taken to evaluate the rhs in the propper environment (the original
 * implementation did not do that) so Module[{x}, Module[{x=x}, x]] gets evaluated
 * to Module[{x}, Module[{x1 = x}, x1]] as expected.
 *
 * The name field in the MExpr is changed to be unique, but an extra field
 * exists in the MExpr (sourceName) that is unchanged and can be used to 
 * pretty print the original program.  (TODO,  perhaps use a property for this?)
 *
 * The visitor is different enough in here that we do not use the recursive
 * MExpr visitor since that would complicate the code for no benefit.
 *
 * Computed Extra Properties:
 * --------------------------
 *
 * This pass adds extra properties on each MExpr symbol:
 *
 * - scopeBinder : the mexpr that binds the mexpr (declares it as formal
 * variable). Multiple MExprs can have the same scopeBinder.
 * If two MExprs have the same name and the same scopeBinder, then they are the same 
 * variable (not the same mexpr instance though). If scopeBinder is undefined or None
 * then the variable is global.  Note that after the MExprBindingRewritePass, two MExprs 
 * cannot have the same name and different scopeBinders.
 * - functionBinder : the mexpr of the function that defined the variable.
 * If the functionBinder is undefined or None then the variable was not defined via a function.
 * For example Function[{a}, Module[{x}, x]] the x's have no functionBinder set.
 *
 * - bindingScopeId : the nearest scopeId which encapsulates the variable, for example
 * mexpr = Function[{a}, Module[{x}, x]] then the scopeBinderId for both x's is the Module mexpr scopeId;
 * i.e. mexpr["part", 2]["getProperty", "scopeId"] .  
 * If the bindingScopeId is undefined or None, then the variable is global.
 * - functionScopeId : the scopeId if the function which encapsulates the variable, for example
 * mexpr = Function[{a}, Module[{x}, x] then the functionScopeId for both x's is mexpr["getProperty", "scopeId"]. 
 *
 * Note that the bindingScopeId and functionScopeId do not look at where
 * the variable was defined, but what is the encosing scoping construct. For example
 * mexpr = Function[{x}, Module[{x}, Module[{z}, x]]
 * then for x = mexpr["part", 2]["part", 2]["part", 2] the functionScopeId is the scopeId of the mexpr
 * and the bindingScopeId is the scopeId of Module[{z}, x] === mexpr["part", 2]["part", 2]
 *
 * This pass also adds extra properties on each scoping MExpr (Function, Module, ...):
 *
 * - scopeId : an integer signifying the scope id for the mexpr binder. This Id is the one
 * referenced by bindingScopeId and functionScopeId. This scopeId is unique to each binding
 * mexpr and is similar in principle to a de brujin index. 
 * - boundVariables : the list of variables that are bound by the scoping mexpr. For example
 * mexpr = Module[{x, y}, ...] the boundVariables for mexpr would be {x, y}. Note that the
 * boundVariables only include the formal variables bound by the mexpr, and do not include 
 * the occurences of the variable within the scoping body. 
 *
 *******************************************************************************
 *******************************************************************************
 *
 * Complete Example. Suppose we have an mexpr=
 * 
 * 1. Function[{x},
 * 2.     Module[{ii = 5},
 * 3.          Function[{y},
 * 4.             x + ii + y
 * 5. ]]]
 *
 * The properties for the above program are:
 *
 * - Line 1: Function has a ScopeId=1
 * - Line 1: The variable x has bindingScopeId=1, functionScopeId=1, scopeBinder=functionBinder=mexpr
 * - Line 2: The module mexpr (mexpr["part", 2]) has a scopeId=2
 * - Line 2: The variable ii has bindingScopeId=2, functionScopeId=1, scopeBinder=mexpr["part", 2], functionBinder=None
 * - Line 3: The function has a scopeId=3
 * - Line 3: The variable y has bindingScopeId=3, functionScopeId=3, scopeBinder=functionBinder=mexpr["part", 2]["part", 2]
 * - Line 4: The variable x has bindingScopeId=3, functionScopeId=3, scopeBinder=functionBinder=mexpr
 * - Line 4: The variable ii has bindingScopeId=3, functionScopeId=3, scopeBinder=mexpr["part", 2], functionBinder=None
 * - Line 4: The variable y has bindingScopeId=3, functionScopeId=3, scopeBinder=functionBinder=mexpr["part", 2]["part", 2]
 *
 *******************************************************************************
 *******************************************************************************
 *)


BeginPackage["Compile`AST`Semantics`Binding`"]

MExprBindingRewrite;
MExprBindingRewritePass;

Begin["`Private`"] 

Needs["Compile`Core`PassManager`PassRegistry`"]
Needs["Compile`Core`PassManager`MExprPass`"]
Needs["Compile`Core`PassManager`PassInformation`"]
Needs["CompileAST`Class`Literal`"]
Needs["CompileAST`Class`Normal`"]
Needs["CompileAST`Class`Base`"]
Needs["CompileAST`Class`Symbol`"]
Needs["CompileUtilities`Reference`"]
Needs["CompileAST`Utilities`Node`"]
Needs["CompileAST`Create`Construct`"]
Needs["CompileUtilities`Asserter`Assert`"]
Needs["CompileUtilities`Error`Exceptions`"]
Needs["CompileUtilities`Callback`"]
Needs["CompileAST`Utilities`MExprVisitor`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)

$UniqueExprsQ = True


RegisterCallback["RegisterPass", Function[{st},
info = CreatePassInformation[
	"MExprBindingRewrite",
	(* Information *)
	"Scoping reassociates all the \"binder\" fields in the bound MExprs to " <>
	"reference their binding environment. Two symbols with the same binding " <>
 	"environment will have the same name (they will not be pointers to the " <> 
	"same exprs -- i.e. they do not alias). " <> 
	"This pass also adds extra properties on mexprs detailing the binding information " <>
	"(see description below for more details).",
	(* Description *)
	"This pass also adorns scoped with a scopeId property and variables with " <>
	"scopeBinder (the scope mexpr where the variable is defined), "<> 
	"functionBinder (the function mexpr where the variable is defined), " <> 
	"bindingScopeId (the scope id of the scope where the variable is being used), and " <>
    "functionScopeId (the scope id of the function where the variable is being used)."
];

MExprBindingRewritePass = CreateMExprPass[<|
	"information" -> info,
	"runPass" -> MExprBindingRewrite
|>];

RegisterPass[MExprBindingRewritePass]
]]


RegisterCallback["DeclareCompileClass", Function[{st},
DeclareClass[MExprClearBindingRewriteProperties,
    <| 
        "visitSymbol"   -> (
            (
                #["removeProperty", "bindingScopeId"];
                #["removeProperty", "functionScopeId"];
                #["removeProperty", "scopeBinder"];
                #["removeProperty", "functionBinder"];
            )&
        ),
        "visitNormal"   -> (
            (
                #["removeProperty", "scopeId"];
                #["removeProperty", "boundVariables"];
            )&
         )
    |>,
    {},
    Extends -> {MExprVisitorClass}
];
]]


MExprBindingRewrite[mexpr_?MExprQ, opts_:<||>] :=
	With[{
        clearer = CreateObject[MExprClearBindingRewriteProperties],
	    st = <|
			"currentScope" -> None, (**< the current scoping MExpr --- None is Global *)
            "currentFunction" -> None, (**< the current function MExpr --- None is Global *)
			"nextScopeIndex" -> CreateReference[1], 
			"nextIndex" -> CreateReference[<||>], (**< maps from symbol names to the next id *)
			"globalBindingEnv" -> CreateReference[<||>], (**< maps from symbol names to unbound MExpr references *)
			"bindingEnv" -> CreateReference[<||>], (**< maps from symbol names to MExpr references *)
			"topMExpr" -> mexpr
		|>
	},
	    mexpr["accept", clearer];
		traverseExpr[st, mexpr]
	]

traverseExpr[st_, expr_] :=
	Module[{head, nodeType, args},
		head = expr["getHead"];
		nodeType = Lookup[ASTNodeType, head, None];
		Switch[nodeType,
			"Atom",
				If[MExprSymbolQ[expr],
					traverseSymbol[st, expr],
					traverseLiteral[st, expr]
				],
			"Function",
				traverseExpr[st, expr["head"]];
				traverseFunction[st, expr],
			"Scope",
				traverseExpr[st, expr["head"]];
				traverseScope[st, expr],
			"DynamicScope",
				traverseExpr[st, expr["head"]];
				traverseScope[st, expr],
			(*"DeclareVariable",
				traverseDeclareVariable[st, expr],*)
			_,
				Which[
					MExprLiteralQ[expr],
						traverseLiteral[st, expr],
					MExprSymbolQ[expr],
						traverseSymbol[st, expr],
					True,
						AssertThat["The binding expression is assumed to be a normal",
							expr]["named", expr]["satisfies", MExprNormalQ];
						args = traverseExpr[st, #]& /@ expr["arguments"];
						AssertThat["None of the arguments should be failed",
							args]["named", "expr"]["elementsSatisfy", (!FailureQ[#])&];
						head = traverseExpr[st, expr["head"]];
						expr["setArguments", args];
						expr["setHead", head];
						expr
				]
		]
	]

traverseDeclareVariable[st_, expr_] :=
	Module[{boundVar, localBoundVariables, rhs = Missing[]},
		AssertThat["The declared expression is assumed contain 1 or two arguments",
			expr]["named", expr]["satisfies", expr["length"] === 1 || expr["length"] === 2];
		If[expr["length"] =!= 1 && expr["length"] =!= 2,
			Return[expr]
		];
		boundVar = expr["part", 1];
		If[expr["length"] == 2,
			rhs = expr["part", 2]
		];
		localBoundVariables = CreateReference[{}];
		expr["setPart", 1, bindVariableList[st, bindDeclareVariable, boundVar, localBoundVariables]];
        expr["setProperty", "boundVariables" -> localBoundVariables["get"]];
       	st["topMExpr"]["addPropertyHolder", expr];
        st["currentScope"]["setProperty", "boundVariables" -> Join[
        	st["currentScope"]["getProperty", "boundVariables", {}],
        	localBoundVariables["get"]
        ]];
		st["topMExpr"]["addPropertyHolder", st["currentScope"]];
		expr
	]

traverseScope[st_, expr_] :=
	Module[{boundVars, localBoundVariables, body, newSt},
		boundVars = expr["part", 1];
		body = expr["part", 2];
		(** we need to bind the RHS of the arguments, since those
		  * bind to the outside scope
		  *)
		Do[
			If[var["hasHead", Set] && var["length"] === 2,
				var["setPart", 2, traverseExpr[st, var["part", 2]]]
			],
			{var, boundVars["arguments"]}
		];
		(** TODO: maybe this can call traverseFunctionLike, since
		  * we have already mutated the rhs of the exprs
		  *)
		  
		newSt = st;
		newSt["currentScope"] = expr;
		newSt["bindingEnv"] = st["bindingEnv"]["clone"];
        expr["setProperty", "scopeId" -> st["nextScopeIndex"]["increment"]];
		localBoundVariables = CreateReference[{}];
		expr["setPart", 1, bindVariableList[newSt, bindScopeVariable, boundVars, localBoundVariables]];
		expr["setPart", 2, traverseExpr[newSt, body]];
        expr["setProperty", "boundVariables" -> localBoundVariables["get"]];
        st["topMExpr"]["addPropertyHolder", expr];
		expr
	]


traverseFunctionLike[st_, expr_, binder_] :=
	Module[{boundVars, localBoundVariables, body, newSt},
		boundVars = expr["part", 1];
		body = expr["part", 2];
		boundVars = If[MExprNormalQ[boundVars] && boundVars["isList"],
			boundVars,
			CreateMExprNormal[List, {boundVars}]
		];
		
		newSt = st;
		newSt["currentScope"] = expr;
		newSt["currentFunction"] = expr;
		newSt["bindingEnv"] = st["bindingEnv"]["clone"];
        expr["setProperty", "scopeId" -> st["nextScopeIndex"]["increment"]];
		localBoundVariables = CreateReference[{}];
		expr["setPart", 1, bindVariableList[newSt, binder, boundVars, localBoundVariables]];
		expr["setPart", 2, traverseExpr[newSt, body]];
        expr["setProperty", "boundVariables" -> localBoundVariables["get"]];
        st["topMExpr"]["addPropertyHolder", expr];
		expr
	]
		
traverseFunction[st_, expr_] :=
	traverseFunctionLike[st, expr, bindFunctionVariable]

getBindingName[ expr_] :=
	{expr["context"], expr["name"]}

getSourceBindingName[ expr_] :=
	{expr["context"], expr["sourceName"]}

getNameElement[{cont_, name_}] :=
	name

getNameElement[ x___] :=
	ThrowException[{"Unexpected arguments to binding getNameElement ", x}]
		

traverseSymbol[st_, expr_] :=
	Module[{name},
		AssertThat["Expecting the expression to be a Symbol in traverseSymbol",
			expr]["named", "expr"]["satisfies", MExprSymbolQ];
		(** we are intereseted in the original name, rather than the potentially mutated name *)
		name = getSourceBindingName[expr];
		Which[
			st["bindingEnv"]["keyExistsQ", name],
				With[ {
					e = st["bindingEnv"]["lookup", name]
				},
				If[getBindingName[e] =!= getBindingName[expr],
					expr["setName", e["name"]]
				];
			        expr["setProperty", "bindingScopeId" -> If[st["currentScope"] =!= None,
			            st["currentScope"]["getProperty", "scopeId"],
			            None
			        ]];
			        expr["setProperty", "functionScopeId" -> If[st["currentFunction"] =!= None,
			            st["currentFunction"]["getProperty", "scopeId"],
			            None
			        ]];
                    expr["setProperty", "scopeBinder" -> e["getProperty", "scopeBinder", None]];
                    expr["setProperty", "functionBinder" -> e["getProperty", "functionBinder", None]];
                    st["topMExpr"]["addPropertyHolder", expr];
					expr
				],
			st["globalBindingEnv"]["keyExistsQ", name],
				With[ {
					e = st["globalBindingEnv"]["lookup", name]
				},
				If[getBindingName[e] =!= getBindingName[expr],
					expr["setName", e["name"]]
				];
                expr["setProperty", "functionBinder" -> None];
                expr["setProperty", "scopeBinder" -> None];
				expr["setProperty", "scopeId" -> 0];
				expr
				],
			True,
			    If[st["nextIndex"]["keyExistsQ", name],
                    expr["setName", getNameElement[name] <> "$" <> ToString[0]],
                    st["nextIndex"]["associateTo", name -> CreateReference[1]]
                ];
                expr["setProperty", "functionBinder" -> None];
                expr["setProperty", "scopeBinder" -> None];
                st["globalBindingEnv"]["associateTo", name -> expr];
				expr
		]
	]

traverseLiteral[st_, expr_] :=
	expr

bindVariable[st_, localVars_, var_, isFunctionScopeQ_:False] :=
	Module[{name, idx},
		AssertThat["The expression should be a Symbol in bindVariable",
			var]["named", "var"]["satisfies", MExprSymbolQ];
		(** we are intereseted in the original name, rather than the potentially mutated name *)
		name = getSourceBindingName[ var];
		If[
			st["bindingEnv"]["keyExistsQ", name] ||
			st["nextIndex"]["keyExistsQ", name], (**< if a global value was bound first *)
				idx = st["nextIndex"]["lookup", name]["increment"];
				var["setName", StringRiffle[{getNameElement[name], idx}, "$$"]],
			(* Else *)
			st["nextIndex"]["associateTo", name -> CreateReference[1]]
		];
        localVars["appendTo", var];
        var["setProperty", "bindingScopeId" -> If[st["currentScope"] =!= None,
            st["currentScope"]["getProperty", "scopeId"],
            None
        ]];
        var["setProperty", "functionScopeId" -> If[st["currentFunction"] =!= None,
            st["currentFunction"]["getProperty", "scopeId"],
            None
        ]];
		var["setProperty", "scopeBinder" -> st["currentScope"]];
		If[TrueQ[isFunctionScopeQ],
            var["setProperty", "functionBinder" -> st["currentFunction"]];
		];
		st["topMExpr"]["addPropertyHolder", var];
		st["bindingEnv"]["associateTo", name -> var];
		var
	]


bindVariableList[st_, binder_, var_, localVars_] :=
	Module[{args},
		AssertThat["Expecting the variables for bindVariableList to be a normal expression",
			var]["named", "var"]["satisfies", MExprNormalQ];
		args = var["arguments"];
		traverseExpr[st, var["head"]];
		args = binder[st, localVars, #]& /@ args;
		var["setArguments", args];
		var
	]
	
	

bindScopeVariable[st_, localVars_, mexpr_] :=
	Module[{var},
		var = If[mexpr["hasHead", Set] && mexpr["length"] === 2,
			traverseExpr[st, mexpr["head"]];
			mexpr["part", 1], 
			mexpr
		];
		var = Which[
			MExprSymbolQ[var], 
				bindVariable[st, localVars, var, False],	
			MExprNormalQ[var] && isTypedMarkup[var] && MExprSymbolQ[var["part", 1]],
				traverseExpr[st, var["head"]];
				var["setPart", 1, bindVariable[st, localVars, var["part", 1], False]];
				var,
			True,
				ThrowingTodo["bindScopeSymbol needs to handle errors if the Scoping arg is not a symbol :: " <> var["toString"]];
				$Failed (* or other error *)
		];
		Which[
			!FailureQ[var] && mexpr["hasHead", Set] && mexpr["length"] === 2,
				mexpr["setPart", 1, var],
			True,
				var
		]	
	]

bindDeclareVariable[st_, localVars_, mexpr_] :=
	0

		
isTypedMarkup[mexpr_] :=
	mexpr["normalQ"] &&
    With[{hd = mexpr["head"]},
        hd["symbolQ"] &&
        hd["fullName"] === "System`Typed" &&
        mexpr["length"] === 2
    ]

bindFunctionVariable[st_, localVars_, var_] :=
	Module[{e1},
		e1 = Which[
			var["symbolQ"], 
				bindVariable[st, localVars, var, True],
			isTypedMarkup[var] && var["part", 1]["symbolQ"],
				traverseExpr[st, var["head"]];
				bindVariable[st, localVars, var["part", 1], True],
			True, 
				ThrowingTodo["bindFunctionVariable needs to handle errors if the Function arg is not a symbol. var = ", var["toString"]];
				$Failed (* or other error *)
		];
		If[!FailureQ[e1] && var["normalQ"],
			var["setPart", 1, e1],
			e1
		]
	]




End[]

EndPackage[]
