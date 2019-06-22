(* ::Package:: *)

(* ::Chapter:: *)
(*prolog*)


Needs["NumericalCalculus`"]


BeginPackage["AlphaScannerFunctions`", {"AlphaScannerFunctions`CommonFunctions`", "NumericalCalculus`"}]


StationaryPoints::usage= "Computes the stationary points of a function f(x)."
LocalExtremaFinder::usage=""


Begin["StationaryPoints`Private`"]


(* ::Chapter:: *)
(*main code*)


$singlesteptimelimit=1


(* ::Section::Closed:: *)
(*StationaryPoints*)


ClearAll[StationaryPoints]
StationaryPoints[expr_, sym_Symbol, opts: OptionsPattern[]]:= LocalExtremaFinder[expr, Automatic, {sym}, opts]
StationaryPoints[expr_, vars_List, opts: OptionsPattern[]]:= LocalExtremaFinder[expr, Automatic, vars, opts]
StationaryPoints[{expr_, cond_}, sym_Symbol, opts: OptionsPattern[]]:= LocalExtremaFinder[expr, cond, {sym}, opts] 
StationaryPoints[{expr_, cond_}, vars_List, opts: OptionsPattern[]]:= LocalExtremaFinder[expr, cond, vars, opts] 


(* ::Input:: *)
(*StationaryPoints[-3+2 x+x^2,x]*)


(* ::Input:: *)
(*StationaryPoints[{Sin[x], -4 <= x <= 4}, {x}]*)


(* ::Input:: *)
(*StationaryPoints[{Sin[x] Cos[x] - Tan[x], -10 <= x <= 10}, {x}]*)


(* ::Input:: *)
(*StationaryPoints[{Sin[1/x], -4 <= x <= 4}, {x}]*)


(* ::Input:: *)
(*StationaryPoints[-x^4 + 15 x^2 + 1 + -y^4 + 15 y^2 + 1, {x, y}]*)


(* ::Input:: *)
(*StationaryPoints[{Sin[x + Sqrt[2] Sin[x^2]], -\[Pi] < x < \[Pi]}, {x}] //N*)


(* ::Input:: *)
(*StationaryPoints[Sin[x^3 y^3],{x, y}]*)


(* ::Input:: *)
(*StationaryPoints[{Exp[x] Sin[y], x^2 + y^2 == 1}, {x, y}]*)


(* ::Section::Closed:: *)
(*LocalExtremaFinder*)


Clear[LocalExtremaFinder];
Options[LocalExtremaFinder] = {"Type" -> Automatic};

LocalExtremaFinder[___] := $Failed

LocalExtremaFinder[_List, _, _List, OptionsPattern] := $Failed

LocalExtremaFinder[expr_, x_Symbol] := LocalExtremaFinder[expr, Automatic, {x}]

LocalExtremaFinder[expr_, Automatic, x_Symbol, opts___] := LocalExtremaFinder[expr, Automatic, {x}, opts]

LocalExtremaFinder[a_. Sinc[f_]^b_. + c_., Automatic, vars_List, OptionsPattern[]] := With[{type = OptionValue["Type"]},
	With[{inst = FindInstance[f== 0, vars]},

		{a+c, #}& /@ inst /; MatchQ[inst, {{__Rule}}]

	] /; MatchQ[type, Automatic|"Maximum"]
] /; VectorQ[{a, b, c}, NumericQ] && a > 0 && b > 0

LocalExtremaFinder[f_, Automatic, vars_, OptionsPattern[]] /; 
	!PolynomialQ[f, vars] && 
	TimeConstrained[
		TrueQ @ Refine[Element[f, Integers], Element[vars, Integers]], 
		0.15, 
		False
	]  := $Failed

(*
In[45]:= LocalExtremaFinder[Fibonacci[n], Automatic, {n}] // Timing
Out[45]= {0.000394, $Failed}
*)

LocalExtremaFinder[in_, cons_, vars_List, opts: OptionsPattern[]] := Catch @ Module[
	{expr, type, solns, xpts, nsol, sensible, eqns, tryNsol = False},
  	(* univariate case *)
  	type = OptionValue["Type"];
  	expr = gentleComplexExpand[in];

	Which[
		cons === Automatic && MatchQ[expr,HeavisideTheta[__]],
			Return[{}],
		cons === Automatic,
	  		eqns = Thread[D[expr, {vars}] == 0] /. Automatic -> True;
	  		If[FreeQ[eqns, Derivative],
  				Quiet[
   				solns = TimeConstrained[
   							Reduce[eqns, vars, Reals, Backsubstitution -> True],
   							2 $singlesteptimelimit
   						];
   				If[MatchQ[Head[solns],Reduce] && ContainsQ[eqns,HoldPattern[((*TriangleWave|*)SquareWave)[__]]], solns = False]
   				],
   				solns = $Failed
   			];
   			
   			(*solns = False;*)
   			print["Reduce solution is :: ", solns, rReduce[eqns, vars, Reals, Backsubstitution -> True]],
   		MatchQ[cons, Abs[_Symbol] < _?NumericQ | (Less | LessEqual)[_?NumericQ, _Symbol, _?NumericQ]], (* and possibly some more... *)
	  		eqns = Thread[D[expr, {vars}] == 0] /. Automatic -> True;
  			Quiet[
  				If[FreeQ[eqns, Derivative],
   					solns = TimeConstrained[
   							Reduce[Flatten[{eqns, cons}], vars, Reals, Backsubstitution -> True],
   							2 $singlesteptimelimit
   						];
   					If[MatchQ[Head[solns],Reduce] && ContainsQ[eqns,HoldPattern[((*TriangleWave|*)SquareWave)[__]]], solns = False]
   					,
   					solns = $Failed
  				]
   			];
   			print["Reduce solution is :: ", solns],   			
   		True,
   			solns = False
	]; 

	If[solns === False && cons === Automatic, Return[$Failed]];
	
  	If[MatchQ[solns, {} | {{}} | $Failed | $Aborted | _Reduce | False],
  		If[cons === Automatic && elementaryQ[expr, vars], 
  			Return[LocalExtremaFinder[expr, Last[AlphaScannerFunctions`SuggestPlotRange`Private`Get1DRange[expr, vars[[1]]]]/.{{x_, l_, r_} :> l<=x<=r}, vars, "Type" -> type]]
  		];
  		tryNsol = True; 
  		nsol =  Quiet @ NLocalExtremaFinder[expr, cons, vars, "Type" -> type];
  		print["Extrema from NLocalExtremaFinder are :: ", nsol,nNLocalExtremaFinder[expr, cons, vars, "Type" -> type]];
  		If[nsol === $Failed,
  			Throw[$Failed],
  			solns = nsol[[All, -1]];
  			solns = {N[expr /. #], #}& /@ Select[solns, (expr /. #) \[Element] Reals &];
  			solns = DeleteDuplicates @ Cases[solns, r_ /; Count[N[r], n_ /; Abs[n] > 10^17, Infinity] == 0];
  			If[type =!= Automatic && cons === Automatic,
  				solns = Select[solns, ExtremaTest[expr, #[[-1]]] === type&];
  			];
  			Throw[ solns ]
  		],

  		sensible = SensibleResults[solns, vars];

  		If[sensible === {}, Throw[$Failed]];
  		solns = Cases[#, _Equal, Infinity] & /@ sensible[[All, 1]];
  		solns = solns /. Equal -> Rule
	];

  	solns = Select[solns, Element[#[[All, -1]] /. C[_] -> 1, Reals] &];
  	solns = Select[solns, Length[#] == Length[vars]&];
  	solns = SortByPosition[#, First, vars]& /@ solns;
  	
  	If[solns === {} && Length[vars] != Length[VarList[eqns]],
  		Throw[$Failed]
  	];
  	
  	If[solns === {},
  		solns = FindInstance[eqns, vars, Reals];
  		If[!MatchQ[solns, HoldPattern @ {{Rule[__]..}..}], Throw[$Failed]]
  	];

  	xpts = Which[
  		type =!= Automatic && cons === Automatic,
  			If[Length[vars] === 1, 
  				Select[solns, NthDerivativeTest[expr, #] === type &],
  				Select[solns, SecondDerivativeTest[expr, #] === type &]
  			],
    	type =!= Automatic && cons =!= Automatic,
    		If[Length[vars] === 1, 
  				Select[solns, NthDerivativeTest[expr, #] === type &],
  				Select[solns, SecondDerivativeTest[expr, #] === type &]
  			],
    	True,
    		solns
    ];
    
  	If[xpts === {} && tryNsol === False && !PolynomialQ[expr, vars], 
  		nsol =  Quiet @ NLocalExtremaFinder[expr, cons, vars, "Type" -> type];
 		print["Extrema from NLocalExtremaFinder are :: ", nsol];
  		If[nsol === $Failed,
  			Throw[$Failed],
  			solns = nsol[[All, -1]];
  			solns = Quiet @ Check[ {N[expr /. #], #}& /@ Select[solns, (expr /. #) \[Element] Reals &], $Failed, General::unfl ];
  			If[ContainsQ[solns, $Failed], Throw[$Failed]];
  			solns = DeleteDuplicates @ Cases[solns, r_ /; Count[N[r], n_ /; Abs[n] > 10^17, Infinity] == 0];
  			solns = If[cons === Automatic && type =!= Automatic,
  						If[Length[vars] === 1, 
  							Select[solns, NthDerivativeTest[expr, #] === type &],
  							Select[solns, SecondDerivativeTest[expr, Last[#]] === type &]
  							],
  						solns
    				];
  			Throw[ solns ]  
  		]
  	];
  	xpts = Select[xpts, Simplify[(expr /. #) /. C[_] -> 1] \[Element] Reals &];
  	(* no RootReduce if not necessary, ex: stationary points x+x^2/2+x^4/4 *)
  	xpts = With[{simple = Simplify[expr /. #, Element[C[_], Integers]]}, Chop[{If[ContainsQ[simple, Root], RootReduce@simple, simple], #}]]& /@ xpts;
  	(* Adding N in Count below for example: LocalExtremaFinder[(x^5 + x^9 - x - 1)^3, Automatic, {x}] *)
  	DeleteDuplicates @ Cases[xpts, r_ /; Count[N[r], n_ /; Abs[n] > 10^17, Infinity] == 0]
]
  

(*
In[380]:= LocalExtremaFinder[-x^4 + 15 x^2 + 1 + -y^4 + 15 y^2 + 1, Automatic, {x, y}, "Type" -> "Minimum"]

Out[380]= {{2, {x -> 0, y -> 0}}}

In[381]:= LocalExtremaFinder[-x^4 + 15 x^2 + 1 + -y^4 + 15 y^2 + 1, Automatic, {x, y}, "Type" -> "Maximum"]

Out[381]= {{229/2, {x -> -Sqrt[(15/2)], y -> -Sqrt[(15/2)]}}, {2292, {x -> -Sqrt[(15/2)], y -> Sqrt[15/2]}}, 
{2292, {x -> Sqrt[15/2], y -> -Sqrt[(15/2)]}}, {2292, {x -> Sqrt[15/2], y -> Sqrt[15/2]}}}

In[382]:= LocalExtremaFinder[Sin[x + Sqrt[2] Sin[x^2]], -\[Pi] < x < \[Pi], {x},"Type" -> "Minimum"] // N

Out[382]= {{-1., {x -> -2.87307}}, {-1., {x -> -2.68057}}, {-1., {x-> -1.73858}}, 
{-0.176321, {x -> -0.356426}}, {0.40943, {x -> 1.35457}}, {0.672405, {x -> 2.13209}}, {-0.884908, {x -> 2.8248}}}

In[280]:= LocalExtremaFinder[Sin[x^3 y^3], Automatic, {x, y}]
Out[280]= {{0, {x -> 0, y -> 0}}}

In[376]:= LocalExtremaFinder[Exp[x] Sin[y], x^2 + y^2 == 1, {x, y}, "Type" -> "Maximum"]
Out[376]= {{1.32116, {x -> 0.673612, y -> 0.739085}}}

In[183]:= LocalExtremaFinder[(9 (4 x^2 Sin[x^2] - 2 Cos[x^2]))/5000, 0 < x < 6, {x}, "Type" -> "Maximum"]
Out[183]= {{0.0149327, {x -> 1.47466}}, {0.147422, {x -> 4.52697}}, {0.23775, {x -> 5.74736}}}

In[140]:= LocalExtremaFinder[Re[(1 + I x^2)/(1 - 3 I x)], Automatic, {x}, "Type" -> "Maximum"]
Out[140]= {{1., {x -> -7.36548*10^-9}}}

In[173]:= LocalExtremaFinder[Im[(1 + I x^2)/(1 - 3 I x)] + Re[(1 + I x^2)/(1 - 3 I x)], Automatic, {x}]
Out[173]= {{0.0383277, {x -> -0.509137}}, {1.21676, {x -> 0.141665}}}

In[54]:= LocalExtremaFinder[x y z, x + y + z == 100, {x, y, z}, "Type" -> "Maximum"]
Out[54]= {{37037., {x -> 33.3333, y -> 33.3333, z -> 33.3333}}}

In[160]:= LocalExtremaFinder[(4 E^-t^2 Erf[t])/Sqrt[\[Pi]], Automatic, {t}, "Type" -> "Maximum"]
Out[160]= {{0.951747, {t -> 0.620063}}}

In[161]:= LocalExtremaFinder[(4 E^-t^2 Erf[t])/Sqrt[\[Pi]], Automatic, {t}, "Type" -> "Minimum"]
Out[161]= {{-0.951747, {t -> -0.620063}}}
*)

elementaryQ[expr_, vars_] := Simplify`FunctionSingularities[expr, vars, "ELEM"] =!= $Failed["NONELEM"]

NLocalExtremaFinder[f_, const_, vars_List, "Type" -> "Minimum"] := Module[
	{findMinimumResult, otherFindMinimumResultPos, otherFindMinimumResultNeg, time, nMinimizeResult, 
		otherMinimizeResult, minima, others = {}},
  	findMinimumResult = Quiet @ If[const === Automatic,
  					Check[FindMinimum[f, ##], $Failed]& @@ Partition[vars, 1],
  					Check[FindMinimum[{f, const}, ##], $Failed]& @@ Partition[vars, 1]
  			];
  	
  	otherFindMinimumResultPos = If[const === Automatic,
  		Quiet @ Check[FindMinimum[f, Thread[{vars, 1 + RandomReal[]/100}]], $Failed],
  		Quiet @ Check[FindMinimum[{f, const}, Thread[{vars, 1 + RandomReal[]/100}]], $Failed]
  	];
  	
  	otherFindMinimumResultNeg = If[const === Automatic,
  		Quiet @ Check[FindMinimum[f, Thread[{vars, -1 - RandomReal[]/100}]], $Failed],
  		Quiet @ Check[FindMinimum[{f, const}, Thread[{vars, -1 - RandomReal[]/100}]], $Failed]
  	];
  	
  	print["FindMinimum results ", {findMinimumResult, otherFindMinimumResultPos, otherFindMinimumResultNeg}];

	{time, nMinimizeResult} = Timing @ Quiet @ Check[NMinimize[{f, const} /. Automatic -> True, vars], $Failed];
	(* Try NMinimize with another setting and hope it will find a different minimum. SamB 0410 *)
	otherMinimizeResult = If[time < 0.15,
						TimeConstrained[ 
							Quiet @ Check[
										NMinimize[{f, const} /. Automatic -> True, vars, 
											Method -> {"DifferentialEvolution", "CrossProbability" -> 0.125, "ScalingFactor" -> 2}],
										$Failed
									],
						$singlesteptimelimit/3,
						{}],
						{}
					];

	print["NMinimize results ", {nMinimizeResult, otherMinimizeResult}];

	If[Length[vars] == 1,
		others = extremaFromNumericalRootFinding[f, const, vars];
		others = Select[others, NthDerivativeTest[f, #] === "Minimum" &]
	];
	print["others", others];
	minima = Union[
			Cases[
				{findMinimumResult, otherFindMinimumResultPos, otherFindMinimumResultNeg, nMinimizeResult, otherMinimizeResult, others} /. {} -> Sequence[],
				{_, {__Rule}},
				Infinity
			],
			SameTest -> (SetPrecision[#1, 3] === SetPrecision[#2, 3]&)
		];

	minima = DeleteCases[minima, FindMinimum[__] | $Failed];
  	minima = DeleteCases[minima, {_?(!NumericQ[#] || !Abs[#] < $MaxMachineNumber &), {__Rule}}&];
  	If[MatchQ[minima, {{_, {__Rule}}..}] && FreeQ[minima, Indeterminate],
  		If[const === Automatic && Length[vars] === 1 && Length[minima] === 1 && EvenFnQ[f, vars],
   			Sort[ AppendTo[minima, {First[minima], {First[vars] -> -minima[[-1, 1, -1]]}}] ],
   			minima
  		],
   		$Failed
   	]
]


gentleComplexExpand[expr_] /; ContainsQ[expr, Abs] := ComplexExpand[expr, TargetFunctions -> {Re, Im}]
gentleComplexExpand[expr_] := expr 


(* ::Section::Closed:: *)
(*NLocalExtremaFinder*)


NLocalExtremaFinder[f_, const_, vars_List, "Type" -> "Minimum"] := Module[
	{findMinimumResult, otherFindMinimumResultPos, otherFindMinimumResultNeg, time, nMinimizeResult, 
		otherMinimizeResult, minima, others = {}},
  	findMinimumResult = Quiet @ If[const === Automatic,
  					Check[FindMinimum[f, ##], $Failed]& @@ Partition[vars, 1],
  					Check[FindMinimum[{f, const}, ##], $Failed]& @@ Partition[vars, 1]
  			];
  	
  	otherFindMinimumResultPos = If[const === Automatic,
  		Quiet @ Check[FindMinimum[f, Thread[{vars, 1 + RandomReal[]/100}]], $Failed],
  		Quiet @ Check[FindMinimum[{f, const}, Thread[{vars, 1 + RandomReal[]/100}]], $Failed]
  	];
  	
  	otherFindMinimumResultNeg = If[const === Automatic,
  		Quiet @ Check[FindMinimum[f, Thread[{vars, -1 - RandomReal[]/100}]], $Failed],
  		Quiet @ Check[FindMinimum[{f, const}, Thread[{vars, -1 - RandomReal[]/100}]], $Failed]
  	];
  	
  	print["FindMinimum results ", {findMinimumResult, otherFindMinimumResultPos, otherFindMinimumResultNeg}];

	{time, nMinimizeResult} = Timing @ Quiet @ Check[NMinimize[{f, const} /. Automatic -> True, vars], $Failed];
	(* Try NMinimize with another setting and hope it will find a different minimum. SamB 0410 *)
	otherMinimizeResult = If[time < 0.15,
						TimeConstrained[ 
							Quiet @ Check[
										NMinimize[{f, const} /. Automatic -> True, vars, 
											Method -> {"DifferentialEvolution", "CrossProbability" -> 0.125, "ScalingFactor" -> 2}],
										$Failed
									],
						$singlesteptimelimit/3,
						{}],
						{}
					];

	print["NMinimize results ", {nMinimizeResult, otherMinimizeResult}];

	If[Length[vars] == 1,
		others = extremaFromNumericalRootFinding[f, const, vars];
		others = Select[others, NthDerivativeTest[f, #] === "Minimum" &]
	];
	print["others", others];
	minima = Union[
			Cases[
				{findMinimumResult, otherFindMinimumResultPos, otherFindMinimumResultNeg, nMinimizeResult, otherMinimizeResult, others} /. {} -> Sequence[],
				{_, {__Rule}},
				Infinity
			],
			SameTest -> (SetPrecision[#1, 3] === SetPrecision[#2, 3]&)
		];

	minima = DeleteCases[minima, FindMinimum[__] | $Failed];
  	minima = DeleteCases[minima, {_?(!NumericQ[#] || !Abs[#] < $MaxMachineNumber &), {__Rule}}&];
  	If[MatchQ[minima, {{_, {__Rule}}..}] && FreeQ[minima, Indeterminate],
  		If[const === Automatic && Length[vars] === 1 && Length[minima] === 1 && EvenFnQ[f, vars],
   			Sort[ AppendTo[minima, {First[minima], {First[vars] -> -minima[[-1, 1, -1]]}}] ],
   			minima
  		],
   		$Failed
   	]
]

NLocalExtremaFinder[f_, const_, vars_List, "Type" -> "Maximum"] := Module[
	{findMaximumResult, otherFindMaximumResultPos, otherFindMaximumResultNeg, time, nMaximizeResult, 
		otherMaximizeResult, maxima, others = {}},
		
  	findMaximumResult = Quiet @ If[const === Automatic,
  					Check[FindMaximum[f, ##], $Failed]& @@ Partition[vars, 1],
  					Check[FindMaximum[{f, const}, ##], $Failed]& @@ Partition[vars, 1]
  			];

  	otherFindMaximumResultPos = If[const === Automatic,
  		Quiet @ Check[FindMaximum[f, Thread[{vars, 1 + RandomReal[]/100}]], $Failed],
  		Quiet @ Check[FindMaximum[{f, const}, Thread[{vars, 1 + RandomReal[]/100}]], $Failed]
  	];
  	
  	otherFindMaximumResultNeg = If[const === Automatic,
  		Quiet @ Check[FindMaximum[f, Thread[{vars, -1 - RandomReal[]/100}]], $Failed],
  		Quiet @ Check[FindMaximum[{f, const}, Thread[{vars, -1 - RandomReal[]/100}]], $Failed]
  	];
  	
  	print["FindMaximum results ", {findMaximumResult, otherFindMaximumResultPos, otherFindMaximumResultNeg}];

	{time, nMaximizeResult} = Timing @ Quiet @ Check[NMaximize[{f, const} /. Automatic -> True, vars], $Failed];
	(* Try NMinimize with another setting and hope it will find a different minimum. SamB 0410 *)
	otherMaximizeResult = If[time < 0.15,
						TimeConstrained[ 
							Quiet @ Check[
									NMaximize[{f, const} /. Automatic -> True, vars, 
										Method -> {"DifferentialEvolution", "CrossProbability" -> 0.125, "ScalingFactor" -> 2}],
									$Failed],
						$singlesteptimelimit/3,
						{}],
						{}
					];
	
	print["NMaximize results ", {nMaximizeResult, otherMaximizeResult}];
					
	If[Length[vars] == 1,
		others = extremaFromNumericalRootFinding[f, const, vars];
		others = Select[others, NthDerivativeTest[f, #] === "Maximum" &]
	];
	
	maxima = Union[
				Cases[
					{findMaximumResult, otherFindMaximumResultPos, otherFindMaximumResultNeg, nMaximizeResult, otherMaximizeResult, others} /. {} -> Sequence[],
					{_, {__Rule}},
					Infinity
				], 
				SameTest -> (SetPrecision[Chop[#1, 10^-6], 3] === SetPrecision[Chop[#2, 10^-6], 3]&)
			];

	maxima = DeleteCases[maxima, FindMaximum[__] | $Failed];
	maxima = DeleteCases[maxima, {_?(!NumericQ[#] || !Abs[#] < $MaxMachineNumber &), {__Rule}}&];
  	If[MatchQ[maxima, {{_, {__Rule}}..}] && FreeQ[maxima, Indeterminate],
  		If[const === Automatic && Length[vars] === 1 && Length[maxima] === 1 && EvenFnQ[f, vars],
   			Sort[ AppendTo[maxima, {First[maxima], {First[vars] -> -maxima[[-1, 1, -1]]}}] ],
   			maxima
  		],
   		$Failed
   	]
]

NLocalExtremaFinder[f_, const_, vars_List, "Type" -> Automatic] := Block[
	{mins, maxs, x = First[vars], others, lst},
	mins = NLocalExtremaFinder[f, const, vars, "Type" -> "Minimum"];
	maxs = NLocalExtremaFinder[f, const, vars, "Type" -> "Maximum"];
	DeleteDuplicates[DeleteCases[Flatten[{mins, maxs}, 1], $Failed], Norm[N[Last[#1][[All, 2]] - Last[#2][[All, 2]]]] < 10^-5 &]
]


(* ::Section::Closed:: *)
(*ExtremaTest and friends*)


ExtremaTest[expr_, r:{__Rule}] /; Length[r] == 1 := NthDerivativeTest[expr, r]
ExtremaTest[expr_, r:{__Rule}] := SecondDerivativeTest[expr, r]

Clear[NthDerivativeTest]

NthDerivativeTest[expr_, {_, {x_ -> x0_}}, n_: 2] := NthDerivativeTest[expr, {x -> x0}, n]

NthDerivativeTest[expr_, {x_ -> x0_}, n_: 2] := Module[{diff, Deval, eval},
	(* We transform Abs[x] -> Sqrt[x^2] because D[] does not do so well with Abs. Fixes ExtremaTest[x^2 - 2 Abs[x], {x -> 1.`}] SamB 0110 *)	
	If[n > 4, Return @ approximateExtremaTest[expr, x -> x0]];
	
	diff = D[expr /. Abs[arg_] :> Sqrt[arg^2], x];
	If[ContainsQ[diff, _Derivative], Return @ numericalNthDerivativeTest[expr, {x -> x0}, n]];
	Deval = Quiet@(N[If[FreeQ[#,Indeterminate],#,Limit[diff/.C[_]->1,x->x0]]]&@((diff /. x -> x0) /. C[_] -> 1)) ; (*added the limit for the sake of functions like f[x_]:=x^2(1-Abs[x]), where the it has a derivative at a*)
	
	Quiet @ If[!PossibleZeroQ[Chop[Deval, 10^-5]], Return["NotAnExtrema"]];
  	eval = (If[FreeQ[#,Indeterminate],#,Limit[diff/.C[_]->1,x->x0]])&@(D[expr /. Abs[arg_] :> Sqrt[arg^2], {x, n}] /. x -> x0) /. C[_] -> 1;
  	eval = If[MachineNumberQ[eval], Chop[eval, 10^-4], N[eval]];
  	If[PossibleZeroQ[eval], Return @ NthDerivativeTest[expr, {x -> x0}, n + 1]];

  	If[EvenQ[n],
  		If[Sign[eval] < 0, 
  			"Maximum", 
  			"Minimum"
  		],
  		"Inflection"
  	]
]

(*
In[68]:= NthDerivativeTest[x^3, {0, {x -> 0}}]
Out[68]= "Inflection"
*)


numericalNthDerivativeTest[expr_, {x_ -> x0_}, n_: 2] := Module[{Deval, eval},
	
	If[n > 4, Return @ approximateExtremaTest[expr, x -> x0]];
	
	Deval = NumericalCalculus`ND[expr /. C[_] -> 1, x, x0];
	Quiet @ If[!PossibleZeroQ[Chop[Deval, 10^-3]], Return["NotAnExtrema"]];
  	eval = ND[expr /. C[_] -> 1, {x, n}, x0];

  	If[PossibleZeroQ[Chop[eval, 10^-4]], Return @ numericalNthDerivativeTest[expr, {x -> x0}, n + 1]];

  	If[EvenQ[n],
  		If[Sign[eval] < 0, 
  			If[approximateExtremaTest[expr, x -> x0] === "Maximum", "Maximum", $Failed], 
  			If[approximateExtremaTest[expr, x -> x0] === "Minimum", "Minimum", $Failed]
  		],
  		"Inflection"
  	]
]

(*
In[66]:= numericalNthDerivativeTest[Re[(1 + I x^2)/(1 - 3 I x)], {x -> -7.365475492130124`*^-9}]
Out[66]= "Maximum"
*)

Options[approximateExtremaTest] = {WorkingPrecision -> 
    MachinePrecision};

approximateExtremaTest[expr_, x_ -> x0_, OptionsPattern[]] := 
 Module[{prec, p, epsilon, ndl, ndr, der},
  prec = OptionValue[WorkingPrecision];
  epsilon = SetPrecision[0.0001, prec];
  p = x0 /. C[_] -> 1;
  der = D[expr, x];
  If[! FreeQ[der, Derivative],
   ndl = der /. x -> p - \[Epsilon];
   ndr = der /. x -> p + \[Epsilon];
   If[! NumberQ[ndl] || ! NumberQ[ndr],
    	ndl = ND[expr, x, p - epsilon, Terms -> 32, WorkingPrecision -> prec];
    	ndr = ND[expr, x, p + epsilon, Terms -> 32, WorkingPrecision -> prec]	
    ],
   ndl = ND[expr, x, p - epsilon, Terms -> 32, WorkingPrecision -> prec];
   ndr = ND[expr, x, p + epsilon, Terms -> 32, WorkingPrecision -> prec]
   ];
   
  (* Not sufficiently steep to warrant calling an extrema, most likely due
  to numerical approximation errors. SamB *)
  If[Abs[ndl] < 10^-6 || Abs[ndr] < 10^-6, Return @ "NotAnExtrema"]; 
  
  If[ndl ndr < 0,
   	If[ndl > 0,
    		"Maximum",
    		"Minimum"
    	],
   	"Inflection"
   ]
  ]

(*
In[41]:= approximateExtremaTest[x^16/10^7, x -> 0]
Out[41]= "Minimum"

In[42]:= approximateExtremaTest[TriangleWave[x], x -> 3/4]
Out[42]= "Minimum"
*)

SecondDerivativeTest[expr_, r : {__Rule}] := Module[{eigenSigns},
	print["SecondDerivativeTest testing : ", r ," for the function ", expr];
	Quiet @ If[!PossibleZeroQ[(D[expr /. Abs[arg_] :> Sqrt[arg^2], {r[[All,1]]}] /. r) /. C[_] -> 1//Chop], Return["NotAnExtrema"]];
  	eigenSigns = Chop@DeleteDuplicates[Sign /@ Eigenvalues[HessianH[expr, r[[All, 1]]] /. r] /. C[_] -> 1, #1==#2&];
  	Which[
  		ContainsQ[eigenSigns, Eigenvalues],
  			$Failed,
   		eigenSigns == {1},
   			"Minimum",
   		eigenSigns == {-1},
   			"Maximum",
   		eigenSigns == {1, -1} || eigenSigns == {-1, 1},
   			"Saddle",
   		True,
   			CalculateScan`InflectionPointScanner`SecondDerivativeBackup[expr, r]
   	]
]

tcSimp[expr_] := TimeConstrained[ToRadicals @ FullSimplify[expr], $singlesteptimelimit, expr]

saneSolutionQ[soln_] := (FreeQ[soln, Indeterminate | _DirectedInfinity] || ContainsQ[soln, Piecewise])

ExactNumbersOnly[expr_] := MatchQ[Cases[expr, x_ /; NumericQ[x] :> Precision[x], {0, Infinity}], {Infinity ...}]

EvenFnQ[expr_, {x_}] := TimeConstrained[TrueQ @ FullSimplify[expr == (expr /. x -> -x), Element[x, Reals]], $singlesteptimelimit/4, False]



resultsToCoords[res_] /; MatchQ[res, {{_, {_Rule ..}} ..}] := Flatten /@ Map[Reverse, res /. r_Rule :> Last[r]]

resultsToCoords[res_] /; MatchQ[res, {_, {_Rule ..}}] := Flatten[{res[[2, All, -1]], First[res]}]

resultsToCoords[res_] /; MatchQ[res, {{_, {_Rule ..}, __} ..}] := Flatten /@ Map[Reverse, res[[All, {1, 2}]] /. r_Rule :> Last[r]]

computeParameters[solns_List, plotrange_List, {x_}] := Block[{cs, min, max, range, eval},
  	cs = Cases[solns, (_ -> r_) :> r, {0, Infinity}];
  	min = Min @@ Floor @@@ Map[C[1] /. Solve[# == plotrange[[2]], C[1]] &, cs];
  	max = Max @@ Floor @@@ Map[C[1] /. Solve[# == plotrange[[3]], C[1]] &, cs];
  	range = Range @@ Sort[{min, max}];
  	eval = Map[solns /. C[1] -> # &, range];
  	eval = Flatten[resultsToCoords /@ eval, 1];
  	Select[eval, plotrange[[2]] <= #[[1]] <= plotrange[[3]] &]
]

(*
In[526]:= computeParameters[{{Sin[
    2 \[Pi] C[1]], {x -> 2 \[Pi] C[1]}}, {-Sin[
     2 \[Pi] C[1]], {x -> \[Pi] + 2 \[Pi] C[1]}}}, {x, -25, 25}, {x}]

Out[526]= {{-7 \[Pi], 0}, {-6 \[Pi], 0}, {-5 \[Pi], 0}, {-4 \[Pi], 
  0}, {-3 \[Pi], 0}, {-2 \[Pi], 0}, {-\[Pi], 0}, {0, 0}, {\[Pi], 
  0}, {2 \[Pi], 0}, {3 \[Pi], 0}, {4 \[Pi], 0}, {5 \[Pi], 
  0}, {6 \[Pi], 0}, {7 \[Pi], 0}}
*)

ConstraintToInterval[expr_] := Block[{ex = FullSimplify[LogicalExpand @ expr]}, 
  	DeleteDuplicates @ Flatten[
  		{Cases[ex, HoldPattern[Inequality[a_, Less | LessEqual, x_, Less | LessEqual, b_]] :> {x, a, b}, {0, Infinity}],
    	Cases[expr, (Less | LessEqual)[a_, x_, b_] :> {x, a, b}, {0, Infinity}]}
    , 1]
]

(*
In[533]:= ConstraintToInterval[-\[Pi] < x < \[Pi]]
Out[533]= {{x, -\[Pi], \[Pi]}}

In[534]:= ConstraintToInterval[-1 < x < 1 || x >= 5 && x < 9]
Out[534]= {{x, 5, 9}, {x, -1, 1}}
*)

IntervalToConstraint[lst_] := #2 <= #1 <= #3 & @@ lst

(*
In[537]:= IntervalToConstraint[{x, 5, 9}]
Out[537]= 5 <= x <= 9
*)


(* ::Chapter:: *)
(*epilog*)


End[]


EndPackage[]
