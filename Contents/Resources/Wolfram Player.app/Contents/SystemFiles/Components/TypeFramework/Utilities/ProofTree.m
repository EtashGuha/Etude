BeginPackage["TypeFramework`Utilities`ProofTree`"]

(*
	type ProofTree alpha = Axiom (List alpha)
	                     | Inference (List (ProofTree alpha)) (List alpha),
		where the type alpha implements the (judgmentForm :: Box) method

	Axiom is not really necessary: a ProofTree should really be a RoseTree.
*)

Axiom
Inference
LambdaConstraintsToTree
MExprConstraintsToTree

Begin["`Private`"]

Needs["TypeFramework`Basic`Lambda`"]
Needs["TypeFramework`ConstraintObjects`ConstraintBase`"]
Needs["TypeFramework`TypeObjects`TypeVariable`"]
Needs["CompileAST`Class`Base`"]



desugar = TypeFramework`Basic`Lambda`Private`desugar

Axiom /: MakeBoxes[ax_Axiom, fmt_] :=
	With[{boxes = toBoxes[ax, fmt]}, InterpretationBox[boxes, ax] /; Head[boxes] =!= toBoxes];

Inference /: MakeBoxes[inf_Inference, fmt_] :=
	With[{boxes = toBoxes[inf, fmt]}, InterpretationBox[boxes, inf] /; Head[boxes] =!= toBoxes];
	
toBoxes[Axiom[cs_], fmt_] :=
	StyleBox[
		RowBox[judgmentForm[#]& /@ cs],
        FontFamily -> "Verdana"
	]

toBoxes[Inference[ts_, cs_], fmt_] :=
	PanelBox[
	   StyleBox[
		   GridBox[{
					{RowBox[toBoxes[#, fmt]& /@ ts]},
					{RowBox[judgmentForm[#]& /@ cs]}
				},
				RowLines -> True,
				ColumnsEqual->False,
				RowsEqual->False,
				BaselinePosition -> Center,
				FrameStyle -> GrayLevel[0.65]
			],
		    FontFamily -> "Verdana"
		]
	];
	
LambdaConstraintsToTree[cs_, e_] :=
	With[{
		ds = Union[
			cs,
			(* This needs to be changed: currently it only follows assumptions one level *)
			Cases[getSource[#]& /@ cs, HoldPattern[_String -> List[_?TypeVariableQ]]]
		]
	},
		lambdaConstraintsToTree[ds, desugar[e]]
	];

lambdaConstraintsToTree[cs_, e:Constant[_]] :=
	Axiom[selectRelevantConstraints[cs, e]];

lambdaConstraintsToTree[cs_, e:Variable[_]] :=
	Axiom[selectRelevantConstraints[cs, e]];

lambdaConstraintsToTree[cs_, e:Lambda[args_?ListQ, body_]] :=
	Inference[
	    Append[
			lambdaConstraintsToTree[cs, #]& /@ args,
			lambdaConstraintsToTree[cs, body]
		],
		selectRelevantConstraints[cs, e]
	];

lambdaConstraintsToTree[cs_, e:App[f_, vars_?ListQ]] :=
	Inference[
	   Join[
			{lambdaConstraintsToTree[cs, f]},
			lambdaConstraintsToTree[cs, #]& /@ vars
	   ],
       selectRelevantConstraints[cs, e]
    ];

lambdaConstraintsToTree[cs_, e:IfCond[b_, e1_, e2_]] :=
	Inference[
	    {
            lambdaConstraintsToTree[cs, b],
			lambdaConstraintsToTree[cs, e1],
			lambdaConstraintsToTree[cs, e2]
		},
        selectRelevantConstraints[cs, e]
	];

lambdaConstraintsToTree[cs_, e:Assign[lhs_, rhs_]] :=
    Inference[
    	{
            lambdaConstraintsToTree[cs, lhs],
            lambdaConstraintsToTree[cs, rhs]
        },
        selectRelevantConstraints[cs, e]
    ];

lambdaConstraintsToTree[cs_, e:Let[a_, body_]] :=
    Inference[
    	{
            lambdaConstraintsToTree[cs, a],
            lambdaConstraintsToTree[cs, body]
        },
        selectRelevantConstraints[cs, e]
    ];

MExprConstraintsToTree[cs_, mexpr_?MExprQ] :=
	With[{
		ds = Union[
			cs,
			Cases[getSource[#]& /@ cs, HoldPattern[_String -> List[_?TypeVariableQ]]]
		]
	},
		If[
			mexpr["atomQ"],
			atomConstraintsToTree[ds, mexpr],
			normalConstraintsToTree[ds, mexpr]
		]
	];
	
atomConstraintsToTree[cs_, mexpr_] :=
	With[{
		syntaxConstraints = selectRelevantConstraints[cs, mexpr]
	},
		Inference[
			Flatten[constraintConstraintsToTree[cs,#]& /@ syntaxConstraints],
			syntaxConstraints
		]
	];
	
normalConstraintsToTree[cs_, mexpr_] :=
	With[{
		syntaxConstraints = selectRelevantConstraints[cs, mexpr]
	},
		Inference[
			Join[
				{ MExprConstraintsToTree[cs, mexpr["head"]] },
				MExprConstraintsToTree[cs, #]& /@ mexpr["arguments"],
				Flatten[constraintConstraintsToTree[cs, #]& /@ syntaxConstraints]
			],
			syntaxConstraints
		]
	];

(* Slightly different from the other cases.
   ConstraintConstraintsToTree : Constraint -> List (ProofTree Constraint)
*)
constraintConstraintsToTree[cs_, constraint_] :=
	With[{
		children = selectRelevantConstraints[cs, constraint]
	},
		If[
			Length[children] === 0,
			{},
			Inference[constraintConstraintsToTree[cs, #], {#}]& /@ children
		]
	]

selectRelevantConstraints[cs_, e_] := Select[cs, (getSource[#] === e &)];

getSource[c_?BaseConstraintQ] := c["properties"]["lookup", "source"];

getSource[_String -> List[tyVar_?TypeVariableQ]] := tyVar["properties"]["lookup", "source"];

judgmentForm[c_?BaseConstraintQ] := c["judgmentForm"];

judgmentForm[ident_String -> List[tyVar_?TypeVariableQ]] :=
	StyleBox[
		GridBox[{
			{RowBox[{ident, "\[RightArrow]", "v" <> ToString[tyVar["id"]]}]},
			{StyleBox[
				RowBox[{"(" <>
					ToString[tyVar["properties"]["lookup", "source"]] <>
					")"}],
				FontSize -> Small]}
		}],
		FontFamily -> "Verdana"
	];

End[]

EndPackage[]
