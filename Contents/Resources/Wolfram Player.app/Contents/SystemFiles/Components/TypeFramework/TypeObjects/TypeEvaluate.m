(* Wolfram Language package *)
BeginPackage["TypeFramework`TypeObjects`TypeEvaluate`"]

TypeEvaluateQ
CreateTypeEvaluate
TypeEvaluateObject

Begin["`Private`"]

Needs["TypeFramework`"]
Needs["CompileUtilities`ClassSystem`"]
Needs["TypeFramework`TypeObjects`TypeBase`"]
Needs["TypeFramework`TypeObjects`TypeLiteral`"]
Needs["TypeFramework`TypeObjects`TypeVariable`"]
Needs["TypeFramework`Inference`Substitution`"]
Needs["CompileUtilities`Error`Exceptions`"] (* for ThrowException *)
Needs["CompileUtilities`Format`"] (* for $FormatingGraphicsOptions *)
Needs["CompileUtilities`Callback`"]
Needs["CompileUtilities`Reference`"]
Needs["TypeFramework`TypeObjects`TypeForAll`"]



sameQ[self_, other_?TypeEvaluateQ] :=
    self["id"] === other["id"] ||
    (
        self["function"] === other["function"] &&
        Length[self["arguments"]] === Length[other["arguments"]] &&
		AllTrue[Transpose[{self["arguments"], other["arguments"]}], #[[1]]["sameQ", #[[2]]]&] 
    )
        
sameQ[___] := 
    Module[{},
        False
    ]


format[ self_, shortQ_:True] :=
    "TypeEvaluate[" <> ToString[self["function"]] <> ", {" <> Riffle[Map[ #["format", shortQ]&, self["arguments"]], ","] <> "}" <> "]"

accept[ self_, vst_] :=
    vst["visitLiteral", self]

RegisterCallback["DeclareTypeFrameworkClass", Function[{st},
TypeEvaluateClass = DeclareClass[
    TypeEvaluateObject,
    <|
        "computeFree" -> (computeFree[Self]&),
        "variableCount" -> (variableCount[Self]&),
        "toScheme" -> (toScheme[Self]&),
        "sameQ" -> (sameQ[Self, ##]&),
        "solve" -> (solve[Self, ##]&),
        "execute" -> (execute[Self, ##]&),
        "clone" -> (clone[Self, ##]&),
        "unresolve" -> Function[ {}, unresolve[Self]],
        "accept" -> Function[{vst}, accept[Self, vst]],
        "toString" -> Function[{}, toString[Self]],
        "toBoxes" -> Function[{fmt}, toBoxes[Self, fmt]],
        "format" -> (format[ Self, ##]&)
    |>,
    {
        "function",
        "arguments"
    },
    Predicate -> TypeEvaluateQ,
    Extends -> TypeBaseClass
];
RegisterTypeObject[TypeEvaluateObject];
]]

    
CreateTypeEvaluate[function_, args:{__?TypeObjectQ}, opts_:<||>] :=
    With[{
        tycon = CreateObject[TypeEvaluateObject, <|
            "id" -> GetNextTypeId[],
            "function" -> function,
            "arguments" -> args
        |>]
    },
        tycon
    ]

stripType[ Type[arg_]] :=
	arg

stripType[ TypeSpecifier[arg_]] :=
	arg

unresolve[ self_] :=
    TypeSpecifier[
    	TypeEvaluate[
	    	self["function"],
	    	Map[ stripType[ #["unresolve"]]&, self["arguments"]]
	    ]]



clone[self_] :=
    clone[self, CreateReference[<||>]];

clone[self_, varmap_] :=
	If[varmap["keyExistsQ", self["id"]],
        varmap["lookup", self["id"]],
		With[ {
			ty =
				CreateTypeEvaluate[
					self["function"],
					Map[#["clone", varmap]&, self["arguments"]]]
		},
			ty["cloneProperties", self]
		]
	]
	

(**************************************************)


computeFree[self_] := 
	Join @@ Map[
	   #["free"]&,
	   self["arguments"]
	]
	
variableCount[self_] :=
	Total[Map[ #["variableCount"]&, self["arguments"]]]

toScheme[self_] :=
    CreateTypeForAll[{}, self];
    
(**************************************************)
(*
   Execute TypeEvaluate
*)

$executeHandlers =
<|
   Unequal -> executeUnequal,
   Plus -> executePlus,
   Greater -> executeGreater,
   Max -> executeMax
|>

noHandler[ self_, env_] :=
	ThrowException[{"Unknown TypeEvaluate function", self["function"], self}]

throwEvaluateSameQ[self_] :=
	ThrowException[{"TypeEvaluate arguments not all the same type", Map[#["type"]["unresolve"]&, self["arguments"]], self}]
	
throwEvaluateLength[self_] :=
	ThrowException[{"TypeEvaluate expecting two arguments of the same type", Map[#["type"]["unresolve"]&, self["arguments"]], self}]
	
throwEvaluateTypeLiteral[self_, val_] :=
	ThrowException[{"TypeEvaluate expecting two arguments of the same type", Map[#["type"]["unresolve"]&, self["arguments"]], self}]

throwEvaluateArgcount[self_] :=
	ThrowException[{"TypeEvaluate expecting two arguments", Map[#["type"]["unresolve"]&, self["arguments"]], self}]
	
throwEvaluateLiteralArguments[self_] :=
	ThrowException[{"TypeEvaluate arguments expected to be variables or TypeLiteral ", Map[#["type"]["unresolve"]&, self["arguments"]], self}]
	


execute[self_, tyEnv_] :=
	If[
		AllTrue[self["arguments"], Length[#["free"]] === 0&],
		With[ {
			handler = Lookup[$executeHandlers, self["function"], noHandler]	
		},
			handler[self, tyEnv]
		],
		self
	]




(*
  The arguments should all be TypeLiteral,  but they might not all be 
  Integer type literals.  It would be an error if they were not all the
  same.
*)

(*
  
*)
executeGreater[self_, tyEnv_] :=
	Module[ {val},
		val = executeFunBinary[ self, tyEnv, Greater, self["arguments"]];
		
		val
	]

executeMax[self_, tyEnv_] :=
	Module[ {val},
		val = executeFunBinary[ self, tyEnv, Max, self["arguments"]];
		val
	]


executeFunBinary[self_, tyEnv_, fun_, _] :=
	throwEvaluateLength[self]

executeFunBinary[self_, tyEnv_, fun_, {arg1_, arg2_}] :=
	Module[ {val, ty},
		If[ !TypeLiteralQ[arg1] || !TypeLiteralQ[arg2],
			throwEvaluateLiteralArguments[self]];
		If[ !arg1["type"]["sameQ", arg2["type"]],
			throwEvaluateLength[self]];
		val = fun[arg1["value"], arg2["value"]];
		ty = tyEnv[ "getLiteralType", val];
		If[ ty === Null,
			throwEvaluateTypeLiteral[self, val]];
		CreateTypeLiteral[ val, ty]
	]

	
executePlus[self_, tyEnv_] :=
	executeFun[ self, tyEnv, Plus, self["arguments"]]	
	
executeFun[self_, tyEnv_, fun_, args_] :=
	Module[ {},
		If[!AllTrue[args, TypeLiteralQ],
			throwEvaluateLiteralArguments[self]];
		Fold[
			If[#1["type"]["sameQ", #2["type"]], 
					CreateTypeLiteral[fun[#1["value"], #2["value"]], #1["type"]], 
					throwEvaluateSameQ[self]]&, args]
	]

executeUnequal[self_, tyEnv_] :=
	Module[{args = self["arguments"], val, ty},
		If[Length[args] =!= 2,
			throwEvaluateArgcount[self]];
		val = !First[args]["sameQ", Last[args]];
		ty = tyEnv[ "getLiteralType", val];
		If[ ty === Null,
			throwEvaluateTypeLiteral[self, val]];
		CreateTypeLiteral[ val, ty]
	]




$solveHandlers =
<|
   Plus -> solvePlus
|>





solve[self_, tyEnv_, ty_?TypeLiteralQ] :=
	If[
		AllTrue[self["arguments"], TypeLiteralQ],
			CreateTypeSubstitution["TypeEnvironment" -> tyEnv],
			With[ {
				handler = Lookup[$solveHandlers, self["function"], noHandler]	
			},
				handler[self, tyEnv, ty]
			]
	]

(*
  The arguments of the TypeEvaluate should contain one variable.
*)
solvePlus[self_, tyEnv_, ty_?TypeLiteralQ] :=
	Module[ {groups, var, literals, tySum},
		groups = GroupBy[self["arguments"], TypeVariableQ];
		var = Lookup[groups, True, {}];
		literals = Lookup[groups, False, {}];
		If[ Length[var] =!= 1,
			ThrowException[{"TypeEvaluate arguments do not contain one variable for solving", Map[#["type"]["unresolve"]&, self["arguments"]], self}]
		];
		var = First[ var];
		tySum = executeFun[self, tyEnv, Plus, literals];
		tyRes = executeFun[self, tyEnv, Subtract, {ty, tySum}];
		CreateTypeSubstitution[var -> tyRes, "TypeEnvironment" -> tyEnv]
	]



(**************************************************)

icon := Graphics[Text[
    Style["TEval",
          GrayLevel[0.7],
          Bold,
          1.2*CurrentValue["FontCapHeight"]/AbsoluteCurrentValue[Magnification]
    ]], $FormatingGraphicsOptions
]


toBoxes[typ_, fmt_]  :=
    BoxForm`ArrangeSummaryBox[
        "TypeEvaluate",
        typ,
        icon,
        Join[
        {
            BoxForm`SummaryItem[{Pane["function: ", {90, Automatic}], typ["function"]}]
        }
        ,
        Map[ BoxForm`SummaryItem[{Pane["arg: ", {90, Automatic}], #["toString"]}]&, typ["arguments"]]
        ],
        {},
        fmt
    ]


toString[typ_] := "TypeEvaluate[" <> ToString[typ["function"]] <> ", {" <> Riffle[Map[ #["toString"]&, typ["arguments"]], ","] <> "}" <> "]"

End[]

EndPackage[]

