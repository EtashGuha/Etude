(* ::Package:: *)

(* ::Chapter::Closed:: *)
(*prolog*)


BeginPackage["AlphaScannerFunctions`", {"AlphaScannerFunctions`CommonFunctions`"}]


InflectionPoints::usage = "Computes the inflection points of a function f(x)"


Begin["`InflectionPoints`Private`"]


(* ::Chapter:: *)
(*main code*)


(* ::Section::Closed:: *)
(*Preliminary Code*)


(* Original code authors:  Erin Craig and Sam Blake, 2009 *)
(* Adapted as ResourceFunction by Paco Jain, 2018 *)


InflectionPoints::usage = "Scanner for finding inflection points.";
PlotExtrema::usage = "Plotting code for inflection points scanner and max/min scanner.";
SecondDerivativeTest::usage = "SecondDerivativeTest[hessian, r : {Rule}, expr] determines if a point is a max, min, or saddle point";
SecondDerivativeBackup::usage = "SecondDerivativeBackup[expr, {{x-> xpt, y-> ypt}} | {x -> xpt, y -> ypt}] determines whether or not a point is a saddle point";
SaddlePtQ::usage = "temporary";
plotFunctionalResult::usage = "";

$InflectionPointScannerDebug = False;
Attributes[dPrint]={HoldAll};
dPrint[args__] := If[$InflectionPointScannerDebug, Print[args]]


(* ::Section:: *)
(*IPS downvalues*)


(* ::Subsection::Closed:: *)
(*main case*)


ClearAll[InflectionPoints]
Options[InflectionPoints] = {"Domain" -> Automatic};
$singlesteptimelimit = 1;
InflectionPoints[expr_, x_Symbol, opts: OptionsPattern[]]:= InflectionPoints[expr, x, "Classify", opts][[All, 1]]
InflectionPoints[expr_, x_Symbol, "Classify", OptionsPattern[]] := Module[
  	{input, xys, inflpoints, domain, statement, res, period, moutput, first, last, method, rounded, doSteps, dom},

  	input = preprocessUserInput[expr];
  	dom = OptionValue["Domain"];
  	(*xys = chooseVariables[expr, input];*)
      If[MatchQ[input, $Failed], Return[]];
  	If[Length[xys] === 2, Return[InflectionPoints[str, Hold[CalculateSaddlePoints[expr]], CalculateSaddlePoints[expr]]]];
  	If[Length[xys] > 1, Return[]];

	(*x = If[ListQ[xys],First[xys],xys];*)
    res = If[dom === Automatic, InflectionPointFinder[input, x], InflectionPointFinder[input, x, "Domain" -> dom]] // ReleaseHold; 
    	inflpoints = First[res];
   		{domain, period, moutput, method} = {"Domain", "Period", "MOutput", Method} /. Rest[res];
   		If[period =!= "Period",
   			If[method =!= "Numeric",
	   			moutput = moutput /. (x -> z_) :> (x -> z + period C[1]);
	   			inflpoints = inflpoints /. {_, {x -> z_}, o___} :> {Simplify[input /. x -> z + period C[1], Element[C[1], Integers]], {x -> z + period C[1]}, o},
	   			inflpoints = Join[
	   				inflpoints /. {_, {x -> z_}, o___} :> {input /. x -> z - period, {x -> z - period}, o},
	   				inflpoints,
	   				inflpoints /. {_, {x -> z_}, o___} :> {input /. x -> z + period, {x -> z + period}, o}
	   			];
	   			moutput = Join[moutput, moutput /. (x -> z_) :> (x -> z + period), moutput /. (x -> z_) :> (x -> z + 2period)];
	   			If[ListQ[domain],
	   				domain = First[domain] + {0, 3period}
	   			]
   			]
   		];
   Flatten /@ inflpoints[[All, 2;;]]
]


(* ::Subsection::Closed:: *)
(*other cases*)


InflectionPoints[str_, Hold[CalculateInflectionPoints[expr_, "Point" -> pt_]], _, OptionsPattern[]] := Module[
  	{input, xys, inflpoints, moutput},  
myPrint["case 2"];  		
	If[ContainsQ[expr, CalculateData|Quantity|Unit], Return[]];
  	input = preprocessUserInput[expr];
  	xys = chooseVariables[expr, input];
  	If[!NumericQ[pt], Return[]];
  	If[Length[xys] === 2, Return[Return[InflectionPoints[str, Hold[CalculateSaddlePoints[expr, "Point" -> pt]], CalculateSaddlePoints[expr, "Point" -> pt]]]]];
	If[Length[xys] > 1, Return[]];
	
  	inflpoints = InflectionPointNear[input, {First @ xys, pt}];
  	
  	If[ContainsQ[inflpoints, $Failed], 
    	Return[
   			createplot["Plot", input, {}, ImageSize->AlphaScaled[1.25]] 
   			]
   	];
   	{inflpoints, moutput} = inflpoints;
	moutput = "MOutput"/.moutput;

  	PlotExtrema[input, Automatic, {N[inflpoints[[All,{1,2}]]], {pt,"PointFlag" -> False}}, xys]
]





InflectionPoints[str_, Hold[CalculateInflectionPoints[expr_, "Domain" -> {_Symbol, lo_?NumericQ, hi_?NumericQ}|{lo_?NumericQ, hi_?NumericQ}]], _, OptionsPattern[]] := Module[
  	{input, res, inflpoints, domain, moutput, method, x},
myPrint["case 3"];
	If[ContainsQ[expr, CalculateData|Quantity|Unit], Return[]];
	
  	input = preprocessUserInput[expr];
  	x = chooseVariables[expr, input];
  	If[Length[x] > 1, Return[]];
  	x = First@x;

  	res = InflectionPointFinder[input, x, "Domain" -> lo <= x <= hi];
  	
  	If[res === $Failed || MatchQ[res, {{}, ___}],
   		Return[
    		createplot["Plot", input, {}, ImageSize->AlphaScaled[1.25]]
    	],
   		inflpoints = First[res];
   		{domain, moutput, method} = {"Domain", "MOutput", Method} /. Rest[res];
   	];

  	PlotExtrema[input, lo <= x <= hi, N[ inflpoints[[All, {1, 2}]] ], {x}] 
]


InflectionPoints[str_, Hold[CalculateInflectionPoints[expr_, "Domain" -> dom_]], _, OptionsPattern[]] := Module[
  	{input, x, res, inflpoints, domain, moutput, method},
myPrint["case 4"];  	
  	If[ContainsQ[expr, CalculateData|Quantity|Unit], Return[]];
  	
  	input = preprocessUserInput[expr];
  	x = chooseVariables[expr, input];
  	If[MatchQ[input, $Failed], Return[]];
  	If[Length[x] === 2, Return[InflectionPoints[str, Hold[CalculateSaddlePoints[expr, "Domain" -> dom]], CalculateSaddlePoints[expr, "Domain" -> dom]]]];
	If[Length[x] > 1, Return[]];
  
  	res = InflectionPointFinder[input, x, "Domain" -> dom];
  	
  	If[res === $Failed || MatchQ[res, {{}, ___}],
   		Return[
    		createplot["Plot", input, {}, ImageSize->AlphaScaled[1.25]]
    	],
   		{inflpoints, domain, moutput, method} = res
   	];
  
  	PlotExtrema[input, dom, N[inflpoints[[All, {1, 2}]]], x]
]


InflectionPoints[str_, Hold[CalculateSaddlePoints[expr_]], _, OptionsPattern[]] := Module[
  	{input, xys, res, moutput},
myPrint["case 5"];
	If[ContainsQ[expr, CalculateData|Quantity|Unit], Return[]];
  	
  	input = preprocessUserInput[expr];
  	xys = chooseVariables[expr, input];
  	
  	If[Length[xys] < 2, Return[]];
  	
  	res = SaddlePointFinder[input, xys];
  		
  	If[res === $Failed || res[[1]] === {}, 
  		Return[ 
     	createplot["Plot", input, {}, ImageSize->AlphaScaled[1.25]]
  		]
   	];
   	{res, moutput} = res;
    moutput = "MOutput"/.moutput;
   	
  	PlotExtrema[input, Automatic, N[res], xys]
]


InflectionPoints[str_, Hold[CalculateSaddlePoints[expr_, "Domain" -> dom_]], _, OptionsPattern[]]:=Module[	
	{input, xys, res, moutput},
myPrint["case 6"];
	If[ContainsQ[expr, CalculateData|Quantity|Unit], Return[]];

  	input = preprocessUserInput[expr];
  	xys = chooseVariables[expr, input];

  	If[Length[xys] < 2, Return[]];
   	
  	res = SaddlePointFinder[input, xys, "Domain"->dom];
  	
  	If[res === $Failed || res[[1]] === {}, 
   		Return[
    	createplot["Plot", input, {}, ImageSize->AlphaScaled[1.25]] ]
   	];
    {res, moutput} = res;
    moutput = "MOutput"/.moutput;

  	PlotExtrema[input, dom, N[res], xys]
]



InflectionPoints[str_, Hold[CalculateSaddlePoints[expr_, "Point" -> {xpt_, ypt_, ___}]], _, OptionsPattern[]]:=Module[	
	{input, xys, res, moutput},
myPrint["case 7"];
	If[ContainsQ[expr, CalculateData|Quantity|Unit], Return[]];

  	input = preprocessUserInput[expr];
  	xys = chooseVariables[expr, input];

  	If[Length[xys] =!= 2, Return[]];
  	
  	res = SaddlePointNear[input, {xys, {xpt, ypt}}];

  	If[res === $Failed || res[[1]] === {}, 
   		Return[
    		createplot["Plot", input, {}, ImageSize->AlphaScaled[1.25]] 
    	]
   	];

   	{res, moutput} = res;
    moutput = "MOutput"/.moutput;

  	PlotExtrema[input, Automatic, N[res], xys]
]


(* ::Subsection::Closed:: *)
(*InflectionPoint Scanner helpers*)


furtherApproximationsPossibleQ[data_] := !MatchQ[Cases[data, n_?NumericQ :> Precision[n], {0, Infinity}], {Infinity..}]

formatNumericalResults[input_, x_, data_, point_, prec_, heldexpr_] :=Module[{y},
	y = If[prec == MachinePrecision, ichop[N[input /. First[point]]], ichop[N[input /. First[point], prec]]];
	{
		RowTemplate[Localize["`1`<GrayText>  at  </GrayText>`2`",10214],{tildeTildeOrEqual[HoldForm@@process[heldexpr],y],tildeTildeOrEqual[x,point[[1,-1]]]}],
		"MInput" -> If[prec == MachinePrecision,
						(Hold[ FindRoot[D[input, x, x], {x, #1, #2}] ]& @@ N[{data[[2, 1, -1]] - 1/1000, data[[2, 1, -1]] + 1/1000}] ),
						(Hold[ FindRoot[D[input, x, x], {x, #1, #2}, WorkingPrecision -> prec] ]& @@ N[{data[[2, 1, -1]] - 1/1000, data[[2, 1, -1]] + 1/1000}] )
					],
		"MOutput" -> point
	}]

formatNumericalResultsDetails[input_, x_, data_, point_, prec_, D1expr_, heldexpr_] := Module[{y, dy},
	y = If[prec == MachinePrecision, ichop[N[input /. First[point]]], ichop[N[input /. First[point], prec]]];
	dy = If[prec == MachinePrecision, ichop[N[eval[D1expr, x ->point[[1, -1]] ]]], ichop[N[eval[D1expr, x ->point[[1, -1]] ], prec]]];
	 {
		 CalculateGrid[
		  {
		   {RowTemplate[Localize["`1`<GrayText>  at  </GrayText>`2`",10215],{tildeTildeOrEqual[HoldForm@@process[heldexpr],y],tildeTildeOrEqual[x,point[[1,-1]]]}] },
		   {dy},
		   {formatinflpoint["Properties" /. data[[3]], dy]}
		   },
		   Style[#, GrayLevel[.3]]&/@{Localize["point",10216], Localize["derivative",10217], Localize["type",10218], {}}
		  ],
		"MInput" -> If[prec == MachinePrecision,
						(Hold[ FindRoot[D[input, x, x], {x, #1, #2}] ]& @@ N[{data[[2, 1, -1]] - 1/1000, data[[2, 1, -1]] + 1/1000}] ),
						(Hold[ FindRoot[D[input, x, x], {x, #1, #2}, WorkingPrecision -> prec] ]& @@ N[{data[[2, 1, -1]] - 1/1000, data[[2, 1, -1]] + 1/1000}] )
					],
		"MOutput" -> point
 }]

formatres[input_, xys_?(Length[#]==1&), res_, moutput_, heldexpr_] := 
	{
		RowTemplate[Localize["`1`<GrayText>  at  </GrayText>`2``3`",10219],{tildeTildeOrEqual[HoldForm@@process[heldexpr],ichop[First[res]]],processInfLoc[tildeTildeOrEqual[First[xys],ichop[res[[2,1,-1]]]]],Sequence@@res[[4]]}],
        (*Aaron should take a look at StepByStepMultidisplay to remove this double Hold*)
        "MInput" -> Hold@Hold[Reduce[D[input, {xys, 2}] == 0, xys]],
        "MOutput" -> moutput
    } /. r_Root?NumericQ :> N[r]

formatresdetails[input_, xys_?(Length[#]==1&), res_, moutput_, D1expr_, heldexpr_] :=
With[{dy = Quiet@ichop@FullSimplify[eval[D1expr, First@xys ->res[[2, 1, -1]] ], Element[C[_], Integers] && Element[Symbol["n"], Integers]]},
 {
 CalculateGrid[
  {
   {RowTemplate[Localize["`1`<GrayText>  at  </GrayText>`2``3`",10220],{tildeTildeOrEqual[HoldForm@@process[heldexpr],ichop[First[res]]],tildeTildeOrEqual[First[xys],ichop[res[[2,1,-1]]]],Sequence@@res[[4]]}]},
   {dy},
   {formatinflpoint["Properties" /. res[[3]], dy]}
   
   },
   {Localize["point",10221], Localize["derivative",10222], Localize["type",10223], {}}
  ],
   "MInput" -> Hold[Reduce[D[input, {xys, 2}] == 0, xys]],
   "MOutput" -> moutput
 } /. r_Root?NumericQ :> N[r]
]

formatres[input_, {x_,y_}, res_, moutput_, heldexpr_] := 
	{
		RowTemplate[Localize["`1`<GrayText>  at  </GrayText>`2` `3`",10224],{tildeTildeOrEqual[HoldForm@@process[heldexpr],ichop[res[[1]]]],tildeTildeOrEqual[PointFormat[{x,y}],ParenthesisForm[ichop[res[[2,1,-1]]],ichop[res[[2,2,-1]]]]],Sequence@@res[[3]]}],
		"MInput" -> Hold[Reduce[D[input, {{x,y}, 2}] == 0, {x,y}]],
		"MOutput" -> moutput
    }

formatres[input_, xys_, res_, moutput_, heldexpr_] :=
 {
 	
 	RowTemplate[Localize["`1`<GrayText>  at  </GrayText>`2``3`",10225],{tildeTildeOrEqual[HoldForm@@process[heldexpr],res[[1]]],tildeTildeOrEqual[PointFormat[xys],ParenthesisForm@@Flatten[ichop/@(({Thread[xys/. #1[[2]]]}&)[res]/. {{a_},b_}->{a,b})]],Sequence@@res[[3]]}], 
 	"MInput" -> Hold[Reduce[D[input, {xys, 2}] == 0, xys]], 
	"MOutput" -> moutput
}

nformatres[input_, xys_?(Length[#]==1&), res_, prec_, moutput_, heldexpr_] := 
	{
		RowTemplate[Localize["`1`<GrayText>  at  </GrayText>`2``3`",10226],{tildeTildeOrEqual[HoldForm@@process[heldexpr],Chop[CalculateN[First[res],WorkingPrecision->prec]]],processInfLoc[tildeTildeOrEqual[First[xys],Chop[CalculateN[res[[2,1,-1]],WorkingPrecision->prec]]]],Sequence@@res[[4]]}],
        "MInput" -> Hold[N @ Reduce[D[input, {xys, 2}] == 0, xys]],
        "MOutput" -> N@moutput
    }


nformatresdetails[input_, xys_?(Length[#]==1&), res_, prec_, moutput_, D1expr_, heldexpr_] :=
With[{dy = Quiet[Chop @ CalculateN[nSimplify@eval[D1expr, First@xys ->res[[2, 1, -1]] ], WorkingPrecision -> prec] /. _DirectedInfinity|ComplexInfinity -> Indeterminate]},
 {
 CalculateGrid[
  {
   {RowTemplate[Localize["`1`<GrayText>  at  </GrayText>`2``3`",10227],{tildeTildeOrEqual[HoldForm@@process[heldexpr],Chop[CalculateN[First[res],WorkingPrecision->prec]]],tildeTildeOrEqual[First[xys],Chop[CalculateN[res[[2,1,-1]],WorkingPrecision->prec]]],Sequence@@res[[4]]}]},
   {dy},
   {formatinflpoint["Properties" /. res[[3]], dy]}
   
   },
   {Localize["point",10228], Localize["derivative",10229], Localize["type",10230], {}}
  ],
   "MInput" -> Hold[Reduce[D[input, {xys, 2}] == 0, xys]],
   "MOutput" -> moutput
 }
]

nSimplify[expr_] := With[{vars = VarList[expr]},
	If[vars === {}, expr, Simplify[expr, Element[vars, Integers]]]
];

nformatres[input_, {x_,y_}, res_, prec_, moutput_, heldexpr_] := 
	{
		RowTemplate[Localize["`1`<GrayText>  at  </GrayText>`2` `3`",10231],{tildeTildeOrEqual[HoldForm@@process[heldexpr],Chop[CalculateN[res[[1]],WorkingPrecision->prec]]],tildeTildeOrEqual[PointFormat[{x,y}],ParenthesisForm[Chop[CalculateN[res[[2,1,-1]],WorkingPrecision->prec]],Chop[CalculateN[res[[2,2,-1]],WorkingPrecision->prec]]]],Sequence@@res[[3]]}],

		"MInput" -> Hold[N @ Reduce[D[input, {{x,y}, 2}] == 0, {x,y}]],
		"MOutput" -> N@moutput
    }

nformatres[input_, xys_, res_, prec_, moutput_, heldexpr_] :=
 {  
 	RowTemplate[Localize["`1`<GrayText>  at  </GrayText>`2``3`",10232],{tildeTildeOrEqual[HoldForm@@process[heldexpr],Chop[CalculateN[res[[1]],WorkingPrecision->prec]]],tildeTildeOrEqual[PointFormat[xys],ParenthesisForm@@Flatten[(Chop[CalculateN[#1,WorkingPrecision->prec]]&)/@(({Thread[xys/. #1[[2]]]}&)[res]/. {{a_},b_}->{a,b})]],Sequence@@res[[3]]}], 
  "MInput" -> Hold[Reduce[D[input, {xys, 2}] == 0, xys]], 
  "MOutput" -> moutput}


formatinflpoint[props_, dy_] := 
Which[
	FreeQ[dy, ComplexInfinity|Indeterminate|Undefined|Interval|Overflow[]|Underflow[]] && Length[props] == 2,
		GrayText[RowTemplate[Localize["`1` and `2`",10233],{props[[1]],props[[2]]}], BaseStyle->"Serif"],
	Length[props] >= 1,
		GrayText[props[[1]], BaseStyle -> "Serif"]
	]

processInfLoc[h_[x_, res_]] /; !FreeQ[res, Symbol["n"]] := 
	With[{n = Symbol["n"]},
		Block[{ns},
			ns = Union[Cases[res, Subscript[n, _], {0, Infinity}]];
			If[ns === {}, ns = {n}];
			RowTemplate["`1`<GrayText>  where  </GrayText>`2`", {h[x, res], CalculateAnd[Element[#, Integers]& /@ ns]}]
		]
	]

processInfLoc[expr_] := expr

(* Special formatting for results when there are infinitely many inflection points. *)

incaseofconstants[point_] := Module[{constants, statement, newpoint, n = Symbol["n"]},
  	constants = Sort[DeleteDuplicates[Cases[point, C[_], {0, Infinity}]] /. {_} -> {Symbol["n"]}]/.{C[a_]:>Subscript[n, a]};
  	
  	statement = If[ContainsQ[point, C[_]],
    	{Spacer[7], Style[RowTemplate[Localize["<GrayText BaseStyle=\"Serif\">for integer </GrayText>`1`",10234],{GrayText[commaList[constants],BaseStyle->"Serif"]}], GrayLevel[.6]], Spacer[10]},
    	{Spacer[10]}
    ];
  	
  	newpoint = If[constants === {n},
    				point[[{1, 2}]] /. {C[_] -> n},
    				point[[{1, 2}]]/.{C[a_]:>Subscript[n, a]}
    ];
    
  	{newpoint[[1]], newpoint[[2]], point[[3]], statement}
  ]
  
incaseofconstants[{z_,{x_,y_}}] := Module[{constants, statement, newpoint},
	constants = Sort[DeleteDuplicates[Cases[{x,y,z}, C[_], {0, Infinity}]] /. {_} -> {Symbol["n"]}]/.{C[a_]:>Subscript[Symbol["n"], a]};
  	
  	statement = If[ContainsQ[{x,y,z}, C[_]],
    	{Spacer[7], Style[RowTemplate[Localize["<GrayText BaseStyle=\"Serif\">for integer </GrayText>`1`",10235],{commaList[constants]}], GrayLevel[.6]], Spacer[10]},
    	{Spacer[10]}
    ];
  	
  	newpoint = If[constants === {Symbol["n"]},
    				{z,{x,y}} /. {C[_] -> Symbol["n"]},
    				{z,{x,y}}/.{C[a_]:>Subscript[Symbol["n"], a]}
    ];
    
    {newpoint[[1]], newpoint[[2]], statement}
	
	]

incaseofconstants[{z_,xys_}] := Module[{constants, statement, newpoint},
	constants = Sort[DeleteDuplicates[Cases[{xys,z}, C[_], {0, Infinity}]] /. {_} -> {Symbol["n"]}]/.{C[a_]:>Subscript[Symbol["n"], a]};
  	
  	statement = If[ContainsQ[{xys,z}, C[_]],
    	{Spacer[7], Style[RowTemplate[Localize["<GrayText BaseStyle=\"Serif\">for integer </GrayText>`1`",10236],{commaList[constants]}], GrayLevel[.6]], Spacer[10]},
    	{Spacer[10]}
    ];
  	
  	newpoint = If[constants === {Symbol["n"]},
    				{z,xys} /. {C[_] -> Symbol["n"]},
    				{z,xys}/.{C[a_]:>Subscript[Symbol["n"], a]}
    ];
    
    {newpoint[[1]], newpoint[[2]], statement}
	
	]


process[expr_]:= expr/.{
 Hold[_Symbol == rhs_] :> Hold[rhs], 
 Hold[lhs_ == _Symbol] :> Hold[lhs],
 Hold[_?usersymbolQ[__Symbol] == rhs_] :> Hold[rhs],
 Hold[lhs_ == _?usersymbolQ[__Symbol]] :> Hold[lhs]
 }

(*****************************************
*
*	COMPUTE THE INFLECTION POINTS
*
******************************************)


(* ::Section::Closed:: *)
(*InflectionPointFinder*)


ClearAll[InflectionPointFinder]
Options[InflectionPointFinder] = {Method -> Automatic, "Domain" -> True};

InflectionPointFinder[expr_, x_, opts:OptionsPattern[]] :=
	Block[{xpts},
		xpts = inflectionLocations[expr, x, OptionValue["Domain"], OptionValue[Method]];
		classifyInflectionPoint[expr, x, xpts, OptionValue[Method]] /; xpts =!= $Failed
	]

InflectionPointFinder[___] = $Failed;


ClearAll[inflectionLocations]
inflectionLocations[expr_, {x_}, args___] := inflectionLocations[expr, x, args]

inflectionLocations[expr_, x_] := inflectionLocations[expr, x, True, Automatic]

inflectionLocations[expr_, x_, dom_] := inflectionLocations[expr, x, dom, Automatic]

inflectionLocations[expr_, x_, domain_, Automatic | "Symbolic" | "SBS"] /; !MixedPolyAndTrancendentalQ[D[expr, {x, 2}], x] :=
	Block[{res},
		res = StepByStepInflectionPoints[Hold[expr], x, domain, "StepsType" -> None, "TimeStepLimit" -> $singlesteptimelimit];
		(
			AppendInflectionPointData[res, {"Domain" -> Automatic, "MOutput" -> IPMOutput[res], Method -> "SBS"}]
			
		) /; Head[res] =!= StepByStepInflectionPoints
	]

inflectionLocations[expr_, x_, domain_, Automatic | "Numeric" | "Numerical"] :=
	Block[{res},
		res = numericalInflectionPoints[expr, x, domain];
		(
			AppendInflectionPointData[res, {"MOutput" -> IPMOutput[res], Method -> "Numeric"}]
			
		) /; res =!= $Failed
	]

inflectionLocations[expr_, x_, _, "Legacy"] := 
	LegacyInflectionPointFinder[expr, {x}, "ShowSolutionMethod" -> True]

inflectionLocations[___] = $Failed;


(* ------------ Numerical Inflection Point Finder ------------ *)
ClearAll[numericalInflectionPoints]
numericalInflectionPoints[expr_, x_] := numericalInflectionPoints[expr, x, True]

numericalInflectionPoints[expr_, x_, True] :=
	Block[{period, res},
		period = FunctionPeriod[expr, x];
		(
			res = iNumericalInflectionPoints[expr, {x, 0, period}, True];
			
			{res, "Period" -> period}
			
		) /; period > 0
	]

numericalInflectionPoints[expr_, x_, dom_] := 
	Block[{rng, res, qrng, res2},
		rng = numericalInflectionPointDomain[expr, x, dom];
		(
			res = iNumericalInflectionPoints[expr, First[rng], dom];
			(
				If[Length[rng] == 2 && Length[res] < 4,
					res2 = iNumericalInflectionPoints[expr, Last[rng], dom];
					If[ListQ[res2] && Length[res] < Length[res2] && Length[res2] > 10,
						qrng = Median[res2[[All, 1, 2]]] + {-1, 1}N[InterquartileRange[res2[[All, 1, 2]]]];
						res2 = Select[res2, Between[#[[1, 2]], qrng]&]
					];
					If[Length[res] < Length[res2] && (Length[res] == 0 || Length[res2] < 30),
						res = res2
					];
				];
				
				{res, "Domain" -> N[Rest[If[res =!= res2, First[rng], Last[rng]]]]}
				
			) /; res =!= $Failed
			
		) /; MatchQ[rng, {{x, _, _}..}]
	]

numericalInflectionPoints[___] = $Failed;


numericalInflectionPointDomain[_, x_, (Less|LessEqual)[a_?NumericQ, x_, b_?NumericQ]] := {{x, a, b}}

numericalInflectionPointDomain[_, x_, (Greater|GreaterEqual)[a_?NumericQ, x_, b_?NumericQ]] := {{x, b, a}}

numericalInflectionPointDomain[_, x_, HoldPattern[Inequality][a_?NumericQ, Less|LessEqual, x_, Less|LessEqual, b_?NumericQ]] := {{x, a, b}}

numericalInflectionPointDomain[_, x_, HoldPattern[Inequality][a_?NumericQ, Greater|GreaterEqual, x_, Greater|GreaterEqual, b_?NumericQ]] := {{x, b, a}}

numericalInflectionPointDomain[expr_, x_, _] := AlphaScannerFunctions`SuggestPlotRange`Private`Get1DRange[expr, x]


iNumericalInflectionPoints[expr_, {x_, a_, b_}, dom_] :=
	Block[{\[Delta], d2, cands, cvals, ipos, res},
		\[Delta] = .0100183(b-a);
		d2 = D[D[expr, x], x];
		cands = numericalInflectionCandidates[expr, d2, {x, a - \[Delta], b + \[Delta]}, dom];
		(
			cvals = Quiet@Sign[d2 /. x -> Mean[#]]& /@ Partition[Join[{a-\[Delta]}, cands[[All, 1, 2]], {b+\[Delta]}], 2, 1];
			
			ipos = Flatten[Position[Partition[cvals, 2, 1], {-1,1}|{1,-1}, {1}]];
			
			res = If[Simplify`FunctionSingularities[expr, x, {"ALL", "IGNORE", "REALS"}] === {{}, {}, {}, {}, {}},
				cands[[ipos]],
				Select[cands[[ipos]], numericalContinuousQ[expr, #]&]
			];
			
			Quiet @ Pick[res, TrueQ /@ (dom /. res), True]
			
		) /; cands =!= $Failed
	]

iNumericalInflectionPoints[___] = $Failed;


numericalContinuousQ[expr_, {x_ -> x0_}] := 
	Quiet @ Block[{val, llim, rlim},
		val = expr /. x -> x0;
		If[!NumericQ[val],
			False,
			llim = numericalApprox[expr, x -> x0, Direction -> -1];
			rlim = numericalApprox[expr, x -> x0, Direction -> 1];
			NumericQ[llim] && NumericQ[rlim] && Max[Norm[val - llim], Norm[val - rlim], Norm[rlim - llim]] <= 1.*10^-8
		]
	]


numericalInflectionCandidates[expr_, d2_, {x_, a_, b_}, dom_] :=
	Block[{NIntegrate, zeros, discont, discont2, res},
		zeros = d2 == 0;
		discont = AlphaScannerFunctions`FunctionDiscontinuities`Private`DiscontinuityConditions[d2];
		(
			res = NReduce[zeros || discont, {x, a, b}];
			(
				List /@ Thread[x -> (Union @@ Append[("IntervalRoots" /. res), ("IsolatedRoots" /. res)[[All, 1]]])]
					
			) /; res =!= $Failed
			
		) /; FreeQ[discont, $Failed]
	]

numericalInflectionCandidates[___] = $Failed;


(* ------------ Inflection Point Classification ------------ *)
classifyInflectionPoint[expr_, {x_}, args___] := classifyInflectionPoint[expr, x, args]

classifyInflectionPoint[_, _, res_, "Legacy"] := res

classifyInflectionPoint[expr_, x_, res_, _] := classifyInflectionPoint[expr, x, res]

classifyInflectionPoint[expr_, x_, {{}, o___}] := {{}, o}

classifyInflectionPoint[expr_, x_, {pts_, o___}] :=
	Quiet @ Block[{d, vals, dvals, sgn, eps, stat, risefall},
		d = D[expr, x];
		vals = carefulChop[expr /. pts];
		dvals = d /. pts;
		
		sgn = Sign[dvals];
		eps = 1.*10^-6 Min[1, Differences[Sort[x /. pts]]];
		
		stat = If[TrueQ[# == 0], "stationary", "non-stationary"]& /@ sgn;
		risefall = MapThread[
			Which[TrueQ[# == 1], "rising", TrueQ[# == -1], "falling", approxRisingQ[d, #2, #3, eps], "rising", True, "falling"]&,
			{sgn, pts, dvals}
		];
		
		{MapThread[{#1, #2, "Properties" -> {#3, #4}}&, {vals, pts, stat, risefall}], o}
	]

(* ------------ Utilities ------------ *)

MixedPolyAndTrancendentalQ = commonMixedPolyAndTrancendentalQ;

AppendInflectionPointData[res_, data_Rule] := AppendInflectionPointData[res, {data}]

AppendInflectionPointData[res:{{_Rule}...}, data_List] := Prepend[data, res]

AppendInflectionPointData[res_, data_List] := Join[res, data]

IPMOutput[res:{{_Rule}...}] := res

IPMOutput[{res_, ___}] := res

SetAttributes[carefulChop, Listable];

carefulChop[x_?InexactNumberQ] /; Abs[x] <= 1.*10^-12 := N[0, Precision[x]]

carefulChop[x_] := x

approxRisingQ[d1_, {x_ -> x0_}, dval_, eps_] := TrueQ[dval < Min[d1 /. x -> x0-eps, d1 /. x -> x0+eps]]


commonMixedPolyAndTrancendentalQ[expr_, x_] := Block[{toAlg},
	
	toAlg = expr /. {_Sin|_Cos|Power[_, _?(!FreeQ[#, x]&)] :> RandomReal[]};
	
	And[
		!AlgebraicExpressionQ[expr, x],
		!FreeQ[toAlg, x] && AlgebraicExpressionQ[toAlg, x]
	]
]

AlgebraicExpressionQ[_Symbol, _] := True
AlgebraicExpressionQ[c_, x_] /; FreeQ[c, x] := True
AlgebraicExpressionQ[Power[f_, _Integer|_Rational], x_] := AlgebraicExpressionQ[f, x]
AlgebraicExpressionQ[HoldPattern[Plus|Times][args__], x_] := VectorQ[{args}, AlgebraicExpressionQ[#, x]&]
AlgebraicExpressionQ[__] := False


(* ------------ Legacy Inflection Point Finder ------------ *)
ichop[s_, tol_:10^-12] := Round[s] + Chop[s - Round[s], tol]
iformat[num_, numdigits_] := ichop@N[IntegerPart[num] + IntegerPart[FractionalPart[num] 10^numdigits]/10^numdigits]

Options[LegacyInflectionPointFinder] = {"Domain" -> Automatic, "ShowSolutionMethod" -> False};

LegacyInflectionPointFinder[expr_, {}|{{}}, OptionsPattern[]] := $Failed;

LegacyInflectionPointFinder[exp_, {x_}, OptionsPattern[]] := Module[
  	{D2, cand, epsilon, inflPts, domain , D1, savemoutput, method = "Reduce", domainListForm, expr},
  	If[VarList[exp] === {}, Return[$Failed]];
  	expr = exp /. Abs[e_] :> Sqrt[e^2];
  	domain = OptionValue["Domain"];
  	domain = If[MatchQ[domain, _List], {#[[2]], #[[1]], #[[3]]} &[domain], domain];
  	D2 = D[expr, {x, 2}];
  	Which[
   		domain === Automatic,
   			cand = Quiet[
     			TimeConstrained[
      					Reduce`ReduceToRules[Reduce[D2 == 0, x, Reals]],
      				$singlesteptimelimit
     			]
   			];
   			If[MatchQ[cand, $Failed | False | $Aborted | _Reduce],
   				(* try numerical root finding... Code is in the solve scanner. SamB 0110 *)
   				cand = Check[FindNumericalRoots[D2, x, "ReturnDomain" -> True], $Failed];
   				If[MatchQ[cand, $Failed | _FindNumericalRoots], Return[$Failed]];
   				{cand, domain} = cand;
   				domain = If[MatchQ[domain, {_Symbol, _?NumericQ, _?NumericQ}],
   							{#[[2]], #[[1]], #[[3]]} &[domain],
   						  Automatic];
   				cand = cand /. n_?NumericQ :> ichop[n];
   				method = "FindRoot";
   				cand = Select[cand, NumericQ[expr /. #]&]
   			];
   			savemoutput = cand,
   		domain =!= Automatic,
   			domainListForm = {#[[2]], #[[1]], #[[3]]} &[domain/.{x < a_-> -Infinity<x<a, x > a_ -> a < x < Infinity, a_ < x -> a<x<Infinity, a_ > x -> -Infinity < x < a}];
   			cand = Quiet[
     			TimeConstrained[
      				Reduce`ReduceToRules[ Reduce[{D2 == 0, domain}, x, Reals] ],
      				$singlesteptimelimit
     			]
   			];
   			If[MatchQ[cand, $Failed | False | $Aborted | _Reduce],
   				(* try numerical root finding... Code is in the solve scanner. SamB 0110 *)
   				cand = Check[FindNumericalRoots[D2, x, "Domain" -> domainListForm], $Failed];
   				If[MatchQ[cand, $Failed|_FindNumericalRoots], Return[$Failed]];
   				cand = cand /. n_?NumericQ :> ichop[n];
   				method = "FindRoot";
   				cand = Select[cand, NumericQ[expr /. #]&]
   			];
   			savemoutput = cand
   	];
  	If[cand === {}||ContainsQ[cand, $Failed], Return[$Failed]];
	cand = Select[cand, FreeQ[exp /. #, Complex | I]&];
	
  	epsilon = Min[Min[Abs[Differences[N[( x /. (cand/.{C[_]->1}) )]]]/4], 1/1000];
  	(* Using Simplify below, motivated by the input 
  		CalculateSimpleExpression[Hold[CalculateInflectionPoints[(x^2 - 15) (x - 1) (x - 5)]]] SamB 1109 *)

    inflPts = Map[
			    	{TimeConstrained[
			    		FullSimplify[expr /. #, Element[C[_], Integers]],
			    		1/17 $singlesteptimelimit,
			    		Simplify[expr /. #]]
			    		, #}&, 
			    	Select[cand, InflPtQ[N[D2], N[#], N[epsilon]] &]
		    	];

  	If[inflPts === {}, Return[$Failed]];
  	
  	(* we now determine the type of each inflection point - rising or falling and stationary or non-stationary *)
  	D1 = D[expr, x];
  	
  	domain = If[domain === Automatic, Automatic, {domain[[1]], domain[[3]]}];

  	{determineType[D1, #, epsilon] & /@ inflPts, "Domain" -> domain, "MOutput"->savemoutput, 
  		If[OptionValue["ShowSolutionMethod"], "Method" -> method, Sequence @@ {}]}
]



Options[InflectionPointNear] = {"Precision" -> MachinePrecision};

InflectionPointNear[exp_, {x_, pt_}, OptionsPattern[]] := Module[
  	{epsilon, D1, D2, rapprox, ineq, lapprox, all, epsilonfortypefinder, point, sortedlist, oneortwo, savemoutput, expr},
  	epsilon = 1/100;
  	expr = exp /. Abs[e_] :> Sqrt[e^2];
  	D1 = D[N[expr, OptionValue["Precision"]], x];
  	D2 = D[N[expr, OptionValue["Precision"]], {x, 2}];
  
  	rapprox = {
  				TimeConstrained[Quiet@FindMaximum[{D1, x > #}, {x, pt}] &[pt + epsilon],.25],
    			TimeConstrained[Quiet@FindMinimum[{D1, x > #}, {x, pt}] &[pt + epsilon],.25]
    		};

  	rapprox = DeleteCases[rapprox, $Failed | $Aborted | _FindMinimum | _FindMaximum | ({_, {x -> xpt_}} /; Abs[xpt - (pt + epsilon)] < 10^-4)];
  	rapprox = First @ SortBy[If[rapprox === {}, rapprox = {{expr /. {x -> pt}, {x -> pt}}}, rapprox], Abs[#[[2, 1, -1]] - pt]&];
  
  	lapprox = {
  				TimeConstrained[Quiet@FindMaximum[{D1, x < #}, {x, pt}] &[pt - epsilon],.25],
    			TimeConstrained[Quiet@FindMinimum[{D1, x < #}, {x, pt}] &[pt - epsilon],.25]
    		};
    			
  	lapprox = DeleteCases[lapprox, $Failed | $Aborted | _FindMinimum | _FindMaximum | {_, {x -> N[pt - epsilon]}}];
  	lapprox = Last @ SortBy[If[lapprox === {}, lapprox = {{expr /. {x -> pt}, {x -> pt}}}, lapprox], Abs[#[[2, 1, -1]] - pt]&];
  
  	ineq = Rationalize[(x /. lapprox[[2]]) - epsilon, 0] <= x <= Rationalize[(x /. rapprox[[2]]) + epsilon, 0];

	all = Quiet[
     			TimeConstrained[
      				Reduce`ReduceToRules[ Reduce[{D2 == 0, ineq}, x, Reals] ],
      				1,
      				TimeConstrained[
      					FindNumericalRoots[D2, x, "Domain" -> {#[[2]], #[[1]], #[[3]]} &[ineq]],
      					$singlesteptimelimit,
      					$Failed
      				]
      			]
   			]; 
	all = If[ContainsQ[all, Complex], Chop[all], all];
  	If[MatchQ[all, _Reduce | $Failed | {} | {{}} | False], all = {}];
  	savemoutput = all;
  	all = Select[all, InflPtQ[N[D2], N[#], epsilon] &];

  	all = Select[all, Element[N[#[[All, -1]]], Reals] && Element[expr /. {x -> N @ #[[All, -1]]}, Reals] &];

  	If[all === {}, Return[$Failed]];
  
  	epsilonfortypefinder = Min[Min[Differences[(x /. #) & /@ all]]/2, 1];

	sortedlist = SortBy[all, Abs[N[(x /. #) - pt]] &];
	
	oneortwo = 
		If[Length[sortedlist]>1,
			If[Abs[(x/.sortedlist[[1]]) - pt] === Abs[(x/.sortedlist[[2]])-pt], 
				sortedlist[[;;2]], 
				First@sortedlist],
		First@sortedlist];

  	point = {RootReduce[expr /. #], #} &/@oneortwo;
    
    {determineType[D1, #, epsilonfortypefinder]&/@point, "MOutput" -> savemoutput}
]




(* ::Section::Closed:: *)
(*InflectionPoint classification*)


(*****************************************
*
*	INFLECTION POINT CLASSIFICATION
*
******************************************)

InflPtQ[D2expr_, {var_ -> pt_}, eps_] := Module[{point},
   	point = pt /. {C[_] -> 1};
   	Sign[eval[D2expr, var -> Rationalize[point - eps,0]]] Sign[eval[D2expr, var -> Rationalize[point + eps,0]]] === -1
]
StationaryPtQ[D1expr_, {var_ -> pt_}] := Module[{point},
   	point = pt /. {C[_] -> 1};
   	Chop[eval[D1expr, var -> point]] == 0
]

RisingPtQ[D1expr_, {var_ -> pt_}, eps_] := Module[{sign},
  	sign = Sign[Chop[eval[D1expr, var -> pt]/.C[_] -> 1]];
  	Which[
   		sign === 1, 
   			True, 
   		sign === -1, 
   			False, 
   		sign === 0, 
   			RisingPtQ2[D1expr, {var -> pt}, eps],
   		True,
   			$Failed
   	]
]

RisingPtQ2[D1expr_, {var_ -> pt_}, eps_] := Module[
   	{neps, D1zeros, const, signleft, signright },
   	const = Rationalize[LessEqual[pt - eps, var, pt + eps ]/.C[_]->1, 0];
   	D1zeros = Reduce`ReduceToRules[Reduce[{D1expr == 0, const}, var, Reals]];
   	If[D1zeros === {}, D1zeros = {var -> {}}];
   	
   	neps = Min[Min[Differences[(var /. D1zeros)]]/2, eps];
   	
   	signleft = Sign[(D1expr /. {var -> pt - neps})/.C[_]->1];
   	signright = Sign[(D1expr /. {var -> pt + neps})/.C[_]->1];
   	Which[
    	signleft === signright === 1,
    		True,
    	signleft === signright === -1,
    		False,
    	True,
    		"else"
    	]
]

determineType[D1expr__, {y_, x_ -> pt_}, eps_] := determineType[D1expr, {y,{x->pt}}, eps]
determineType[D1expr__, {y_, {x_ -> pt_}}, eps_] := Module[{stationary, rising},
  
  	stationary = TimeConstrained[StationaryPtQ[D1expr, {x -> pt}] /. {True -> Localize["stationary",26996], False -> Localize["non-stationary",26997]}, .2 $singlesteptimelimit, Null] ;
  	rising =  TimeConstrained[RisingPtQ[D1expr, {x -> pt}, eps] /. {True -> Localize["rising",26998], False -> Localize["falling",26999], "else" -> $Failed}, .4 $singlesteptimelimit, Null];
  	
  	{y, {x -> pt}, "Properties" -> DeleteCases[{stationary, rising}, $Failed | Null | _ == 0]}
]
eval[expr_, var_ -> point_] := Quiet[Check[expr /. var -> point, Limit[expr, var -> point]]]


(*****************************************
*
*	COMPUTE THE SADDLE POINTS
*
******************************************)

Options[SaddlePointFinder] = {"Domain" -> Automatic};

SaddlePointFinder[exp_, xys_, OptionsPattern[]] := Module[
   {cand, domain, partials, saddlePts, hessian, goodpoints, savemoutput, howmany = Length[xys], expr},
   domain = OptionValue["Domain"];
   domain = Which[
   				domain =!= Automatic && MatchQ[domain, {{_, _, _} ..}],
					(#[[2]] <= #[[1]] <= #[[3]]) & /@ domain,
				domain =!= Automatic && MatchQ[domain, {_, _, _}],
					(#[[2]] <= #[[1]] <= #[[3]]) &@domain,
				domain =!= Automatic && MatchQ[domain, {Repeated[{_,_}, Length[xys]]}],
					MapThread[{##}[[1, 1]] <= {##}[[-1]] <= {##}[[1, -1]] &,{domain, xys} ],
   				True,
	   				domain
   ];
   expr = exp /. Abs[e_] :> Sqrt[e^2];
   partials = D[expr, {xys}];
   Which[
   	domain === Automatic,
	 	cand = Quiet[
	 				TimeConstrained[
	 						Reduce[partials == 0, xys, Reals, Backsubstitution -> True],
	    					2 $singlesteptimelimit
	 				]
	 	];
	 	If[MatchQ[ cand, $Failed|$Aborted|_Reduce],
	    					cand = 
	    						TimeConstrained[
	     							FindInstance[Thread[partials == 0], xys, Reals],
	     							2 $singlesteptimelimit,
	     							$Failed
	     						]
	    	], 

	domain =!= Automatic,
	 	cand = Quiet[
   				TimeConstrained[
    					Reduce[Flatten[{partials == 0, domain}], xys, Reals, Backsubstitution -> True], 
    					2 $singlesteptimelimit
   				]
	 	];
	    If[MatchQ[cand, $Failed|$Aborted|_Reduce] || FreeQ[cand/.And->List, Repeated[_Equal, {howmany}]],
	    				cand = 	
	    					TimeConstrained[
     							FindInstance[Flatten[{Thread[partials == 0], domain}], xys, Reals],
     							3 $singlesteptimelimit,
     							$Failed
     						]
    	]
   	];
 	If[!MatchQ[cand, {{__Rule}..}] && (MatchQ[cand, _FindInstance] || functionalResultQ[cand, xys] || ( FreeQ[cand, _?(ContainsQ[xys, #] &) == _])),
 		cand = FindInstance[Flatten[{Thread[partials == 0], domain}] /. Automatic -> True, xys, Reals]
 	];
	If[MatchQ[cand, $Failed | False | _Reduce | _FindInstance | {} | {{}}], Return[$Failed]];
	savemoutput = cand;
   	cand = Select[cand, FreeQ[N@#, Complex] &];
   	If[cand === {}, Return[$Failed]];
   	cand = pickpoints[cand];
   	If[ContainsQ[cand, $Failed], Return[$Failed]];
   	cand = If[MatchQ[cand, {{_->_}, {_->_}}], cand = cand, Select[cand, Count[#, _?(ContainsQ[xys, #] &) -> _] == Length[xys] &]];
   	If[MatchQ[cand, {{_->_},{_->_}}], cand = cand/.{{{c_->a_},{d_->b_}}/;c=!=d->{{c->a, d->b}}}, cand = cand];
   	cand = If[domain =!= Automatic,
   		 		appropriatesolns[cand, domain, xys],
   		 		cand];
   	hessian = CalculateScan`LinearAlgebraScanner`HessianH[expr, xys];
   (*SPF was returning points in the order {y->_, x->_} and thus flipping the coordinates:
   CalculateSaddlePoints[x^2 - y^2 - 2 x + 4 y + 6]*)
   goodpoints = If[Length[xys]===2,
   						{First@xys -> (First@xys /. #), Last@xys -> (Last@xys /. #)} & /@Select[cand, SaddlePtQ[hessian, # /. {C[_] -> 1}, expr] &]/.{(a_->{b_}) -> (a->b)},
   						SortBy[#, #[[1]] &] & /@Select[cand, SaddlePtQ[hessian, # /. {C[_] -> 1}, expr] &]
   				];
   				
   saddlePts =  
   				  Map[
   					{TimeConstrained[
			    		FullSimplify[expr /. #, Element[C[_], Integers]],
			    		1/30 $singlesteptimelimit,
			    		Simplify[expr /. #]
			    	]
			    	, #} 
			    	&, goodpoints];
	{saddlePts, "MOutput"->savemoutput}
   ];


functionalResultQ[expr_, {x_, y_}] := Cases[expr, a_Symbol == rhs_ /; MemberQ[{x, y}, a] && ! FreeQ[rhs, Alternatives @@ {x, y}], {1, Infinity}] =!= {}
   
pickpoints[a_Or] := Module[{temp = a},
  temp = temp /. {Or -> List};
  temp = Reduce`ReduceToRules /@ temp;
  Sequence @@ # & /@ DeleteCases[temp, $Failed]
  ];
pickpoints[a_And] := Reduce`ReduceToRules[a];
pickpoints[a_Equal] := Reduce`ReduceToRules[a];

pickpoints[expr_] := expr
 
appropriatesolns[soln_, dom_,xys_] := Module[
  {finalsoln, constants},
  finalsoln = Select[soln, FreeQ[dom /. #, False] &] ;
  constants =
   		{#, TimeConstrained[Reduce`ReduceToRules@Reduce[#, C[1], Integers], $singlesteptimelimit, $Failed] & /@ (dom /. #)} & /@ finalsoln;
  constants = DeleteCases[constants, True | False | {} | {{}}, {0, Infinity}];
  If[MatchQ[constants, {{{Repeated[_->_, {Length[xys]}]}}..}], Return[constants]];
  constants = Partition[Flatten[{#[[1]] /. #[[2]]} & /@ constants],2]
  ];

SaddlePointNear[exp_, {{x_, y_}, {xt_, yt_}}] :=  Module[{savemoutput, cand, partials, eps, hess, firstzero, distance, xpt, ypt, temp, expr, deltax, deltay},
  xpt = Rationalize[xt, 0];
  ypt = Rationalize[yt, 0];
  
  expr = exp /. Abs[e_] :> Sqrt[e^2];
  partials = D[expr, {{x, y}}];
  
  {deltax, deltay} =  BlockRandom[
    SeedRandom[DateString@$TimeStamp];
    RandomReal[{-1/100, 1/100}, 2]
	];

  
  firstzero =  Quiet[Check[
  				TimeConstrained[
    				FindRoot[
     						partials == 0, {x, xpt}, {y, ypt}
     				]
     			, $singlesteptimelimit, $Failed]
     			, 
     			TimeConstrained[
    				FindRoot[
     						partials == 0, {x, xpt + deltax}, {y, ypt + deltay}
     				]
     			, $singlesteptimelimit, $Failed]
     			]];

  If[firstzero === $Failed, Return[]];
  hess = CalculateScan`LinearAlgebraScanner`HessianH[expr, {x, y}];

  If[FreeQ[N@firstzero, Complex] && SaddlePtQ[hess, firstzero , expr], 
  Return[ {{{FullSimplify[expr /. #], #}} & @ firstzero, "MOutput" -> firstzero}]
  ]; 
  	
  eps = Rationalize[1.1 Norm[{xpt, ypt} - {x, y} /. firstzero] + .05, 0];

  (* the /.{C[_]->0} is to account for the fact that Reduce returns complicated results 
  with the new definition of the neighborhood around the point of interest (square instead of circle) *)
  temp = Quiet[TimeConstrained[
  				Reduce[{partials == 0, (x - xpt) + (y - ypt) <= eps}, {x, y}, Reals, Backsubstitution -> True] /. {C[_] -> 0},
  				$singlesteptimelimit, $Failed
  		]];
  cand = Reduce`ReduceToRules[ temp ];

  (* ReduceToRules fails a lot. The next three lines are motivated by results like:
  (x==0 && -201/200 <= y <= -201/200) || (-201/200 <= x && y==0)..  for the function x^3y^3 that has a clear saddle point at x=0.
  For now, we guess a point and use SaddlePtQ later to see if we're right. 
  This works nicely on: 
  Reduce[{{-Sin[x] Sin[y], Cos[x] Cos[y]} == 0, -Sqrt[2] + x + y <= 114693816/23290387}, {x, y}, Reals, Backsubstitution -> True] /. {C[_] -> 0} 
  (from cspe[Hold[CalculateSaddlePoints[Cos[x] Sin[y], "Point" -> {Sqrt[2]/2, Sqrt[2]/2}]]])
  *)
  If[ContainsQ[cand, $Failed] && temp =!= $Failed, 
  		savemoutput = cand;
  		cand = choosepoint[temp, {{x,y},{xpt,ypt}}],
  		savemoutput = cand;
  		cand
  ];
  If[MatchQ[temp, $Failed | False | _Reduce | {} | {{}}], 
   eps = 2 eps;
   temp = Quiet[TimeConstrained[
  				Reduce[{partials == 0, (x - xpt) + (y - ypt) <= eps}, {x, y}, Reals, Backsubstitution -> True] /. {C[_] -> 0},
  				1.5 $singlesteptimelimit, $Failed
  		]];
  cand = Reduce`ReduceToRules[ temp ];
  If[ContainsQ[cand, $Failed] && temp =!= $Failed, 
  		cand = {{x->First@DeleteDuplicates[Cases[temp, x == a_ :> a, {0, Infinity}]],y->First@DeleteDuplicates[Cases[temp, y == a_ :> a, {0, Infinity}]]}},
  		cand
  ]
  ];
  
   If[MatchQ[cand, $Failed | False | _Reduce | {} | {{}}] || ContainsQ[cand, _->Sequence[]], Return[$Failed]];
   
  cand = Select[cand, FreeQ[N@#, Complex] &];
  cand = Select[cand, SaddlePtQ[hess, #, expr] &];
  cand = {#, Norm[{#[[1, -1]], #[[2, -1]]} - {xpt, ypt}]} & /@ cand;
  distance = Min[cand[[All, -1]]];
  cand = Select[cand, #[[2]] === distance &];

  {{expr /. #, #} & /@ cand[[All, 1]], "MOutput" -> savemoutput}
  
  ]
  
(*****************************************
*
*	CHECK THE SADDLE POINTS
*
******************************************)
choosepoint[options_, {{x_,y_},{xval_,yval_}}]:= Module[{xlist, ylist},
	xlist = DeleteDuplicates[Cases[options, x == a_ :> a, {0, Infinity}]];
	If[xlist == {}, Return[$Failed]];
	ylist = DeleteDuplicates[Cases[options, y == a_ :> a, {0, Infinity}]];
	If[ylist == {}, Return[$Failed]];
	{{
	x-> First@SortBy[xlist, #-xval&],
	y-> First@SortBy[ylist, #-yval&]	
	}}

]

HessianH[f_, x_List?VectorQ] := D[f, {x, 2}];

SaddlePtQ[hessian_, {{x_->a_, y_->b_}}, expr_]:= SaddlePtQ[hessian, {x->a, y->b}, expr]
SaddlePtQ[hessian_, {a_?(MatchQ[#, {Repeated[_->_]}]&)}, expr_]:= SaddlePtQ[hessian, a, expr]
SaddlePtQ[hessian_, r : {__Rule}, expr_] := FastSecondDerivativeTest[hessian, r, expr] === "Saddle"

SaddlePtQBackup[expr_, points : {{_ -> _} ..}, eps_] := 
	Module[{sadeval = {}, i = 1},
  		While[
   			FreeQ[sadeval, True] && Length[sadeval] < Length[points]&&i<Length[points], i++; 
   			AppendTo[sadeval, SaddlePtQHelper[expr, points[[i]], eps]]
   		];
  	If[ContainsQ[sadeval, True], "Saddle", $Failed]
  	]
  	
SaddlePtQHelper[expr_, {var_ -> pt_}, eps_] := Quiet[Sign[Chop[eval[expr, var -> pt - eps]]] Sign[Chop[eval[expr, var -> pt + eps]]] === -1]

SecondDerivativeTest[expr_, r_] := FastSecondDerivativeTest[HessianH[expr, r[[All,1]]], r, expr ]
(* for use in MaxMin *)
FastSecondDerivativeTest[hessian_, r : {__Rule}, expr_] := Module[{eigenSigns},
  eigenSigns = DeleteDuplicates[(Sign /@ Eigenvalues[hessian /. r]) /. C[_] -> 1];
  Which[
   eigenSigns === {1},
   	"Minimum",
   eigenSigns === {-1},
   	"Maximum",
   MatchQ[eigenSigns, {1, -1} | {-1, 1}],
   	"Saddle",
   Length[r[[All,1]] ]>2, 
   	"Undecided",
   True,
   	SecondDerivativeBackup[expr, r]
   ]
  ]

SecondDerivativeBackup[expr_,  {{x_ -> xpt_, y_ -> ypt_}} | {x_ -> xpt_, y_ -> ypt_}] := 
	Module[ {zvalue, parametrized, r, t = Symbol["t"], roots, epsilon},
		If[PossibleZeroQ[expr], Return[False]];
  		zvalue = expr /. {x -> xpt, y -> ypt};
  		r = 1/100;
  		parametrized = expr /. {x -> r Cos[t]-xpt, y -> r Sin[t]-ypt};
  		roots = Quiet[ 
    				TimeConstrained[
     					Reduce[{parametrized - zvalue == 0, 0 <= t <= 2 Pi}, t, Reals],
     				1.1 $singlesteptimelimit, {}]
    			];
    	If[MatchQ[roots, _Reduce | $Failed | {} | {{}} | False], Return[$Failed]];
    	roots =	If[roots =!= {}, pickpoints[roots], {}];
  		roots = Select[roots, FreeQ[N[#], Complex] &];
 		
 		epsilon = Min[Abs[Differences[t /. roots]]/2, 1];
  
  		SaddlePtQBackup[parametrized - zvalue, roots, epsilon]
	]


(* ::Section::Closed:: *)
(*SBS-IP*)


(* ::Subsection::Closed:: *)
(*StepByStepInflectionPoints*)


ClearAll[StepByStepInflectionPoints]
SetAttributes[StepByStepInflectionPoints, HoldFirst];
Options[StepByStepInflectionPoints] = {"FunctionName" -> Automatic, "TimeStepLimit" -> \[Infinity], "StepsType" -> None};

StepByStepInflectionPoints[expr_?realFunctionQ, x_, cons_:True, opts:OptionsPattern[]] /; FreeQ[cons, Rule] :=
	Block[{res, $TimeLimit},
		
		$TimeLimit = OptionValue["TimeStepLimit"];
		If[!TrueQ[$TimeLimit > 0], $TimeLimit = \[Infinity]];
		
		res =
			iStepByStepInflectionPoints[
				expr, x, cons,
				ObtainF[expr, x, OptionValue["FunctionName"]]
			];

		res /; res =!= $Failed
	]


(* ::Subsection::Closed:: *)
(*iStepByStepInflectionPoints*)


SetAttributes[iStepByStepInflectionPoints, HoldFirst];


iStepByStepInflectionPoints[expr_, x_, cons_, f_] /; Internal`LinearQ[expr, x] :=
	(
		sowIPIntro[expr, f[x], cons];
		sowLinearIP[expr, f[x]];
		
		{}
	)


iStepByStepInflectionPoints[Hold[expr_], x_, ocons_, f_] :=
	Block[{cons = ocons, period, restrictedPQ, domain, \[ScriptCapitalD]2, roots, discont, cands, ints, candrules, testpts, res},
		sowIPIntro[expr, f[x], cons];
		sowIPDefinition[f[x]];
		sowConcavityDefinition[f[x]];

		(* restrict to domain *)
		{cons, period, restrictedPQ} = reduceAndSowPeriodIP[expr, f[x], cons];
		domain = obtainAndSowDomainIP[expr, f[x], cons];
	
		(* find f''[x] *)
		\[ScriptCapitalD]2 = obtainAndSowSecondDerivativeIP[expr, f[x]];
	
		(* restrict domain when f''[x] is periodic but f[x] is not *)
		If[!TrueQ[restrictedPQ],
			{cons, period, restrictedPQ} = reduceAndSowPeriodIP[\[ScriptCapitalD]2, f''[x], cons];
			If[restrictedPQ, domain = domain && cons]
		];

		(* find inflection point candidates *)
		roots = obtainAndSowDerivativeRootsIP[\[ScriptCapitalD]2, f[x], domain];
		discont = obtainAndSowNotDifferentiableIP[\[ScriptCapitalD]2, f[x], domain];
		cands = combineRootsAndDiscontAndSowIP[roots, discont, f[x], expr, domain];

		(* short circuit if there are no IP candidates *)
		If[cands === False,
			Return[sowNoInflectionPointCandidates[expr, f[x], domain, restrictedPQ, \[ScriptCapitalD]2]]
		];

		(* short circuit if everything is an IP candidate *)
		If[TrueQ[Simplify`SimplifyInequalities[cands || !domain]],
			Return[sowAllInflectionPointCandidates[expr, f[x], domain, restrictedPQ, \[ScriptCapitalD]2]]
		];

		(* if possible, shift domain to have inflection point candidate on boundary of domain *)
		If[restrictedPQ,
			{cands, domain} = shiftAndSowDomainIP[cands, domain, f[x], period];
			If[TrueQ[domain], restrictedPQ = False; cons = ocons]
		];
		
		(* find intervals and test points *)
		{ints, candrules, testpts} = obtainAndSowIntervalsAndTestPtsIP[cands, domain, f[x]];
		
		(* classify intervals and summarize *)
		res = obtainAndSowConcavityClassification[ints, candrules, testpts, f[x], \[ScriptCapitalD]2];

		(* determine inflection points *)
		res = sowInflectionPoints[res, expr, f[x], ocons, period];

		(* if possible, generalize results to all periods *)
		If[restrictedPQ && res =!= {},
			res = unrestrictIPPeriodicity[res, expr, \[ScriptCapitalD]2, f[x]]
		];
		
		res /. (x -> a_) :> {x -> a}
	]


(* ::Subsubsection::Closed:: *)
(*ObtainF*)


SetAttributes[ObtainF, HoldFirst];
ObtainF[expr_, x_] := ObtainF[expr, x, Automatic]

ObtainF[expr_, x_, Automatic] := With[{contextexpr = MyContext[{Hold[expr], x}]}, 
	SBSSymbol@ToString@First[Replace[Select[MyContext /@ fs, FreeQ[contextexpr, #]&], {} :> {Last[fs]}, {0}]]
]

ObtainF[_, _, f_] := f

fs = {Symbol["f"], Symbol["y"], Symbol["g"], Symbol["h"], Symbol["\[ScriptF]"]};

MyContext[expr_] := expr /. {v_Symbol /; v =!= Hold :> iMyContext[v]}
iMyContext[sym_] := Symbol[SymbolName[sym]]


(* ::Subsection::Closed:: *)
(*obtainAndSowDomainIP*)


(* ::Subsubsection::Closed:: *)
(*obtainAndSowDomainIP*)


SetAttributes[obtainAndSowDomainIP, HoldFirst];
obtainAndSowDomainIP[expr_, f_[x_], cons_] := Module[{domain, format, closedEndPts, interiorDomain, formatInterior},
	domain = functionDomain[{Hold[expr], cons}, {x}, Reals];
	If[domain === $Failed, SBSFail[]];
	
	domain = Reduce@Simplify[domain];
	
	If[domain === False || !FreeQ[domain, C|Element|NotElement], SBSFail[]];
	
	If[domain === Reduce[Simplify[cons]], 
		Return[cons]
	];

	format = CalculateScan`FunctionPropertiesScanner`formatDomainOrRange[domain, {x}] /. CalculateScan`CommonSymbols`Commented[g_, ___] :> g;
	domain = format[[1, 4]] //. {ParenthesisForm[e_] :> e, CalculateScan`CommonSymbols`CalculateAnd[l_] :> And @@ l, CalculateScan`CommonSymbols`CalculateOr[l_] :> Or @@ l};
	domain = domain /. Symbol["n"] -> C[1];
	
	closedEndPts = closedEndpoints[domain, x];
	interiorDomain = If[closedEndPts =!= {}, 
		Reduce[domain && And @@ Unequal @@@ Flatten[closedEndPts], x],
		domain
	];
	formatInterior = CalculateScan`FunctionPropertiesScanner`formatDomainOrRange[interiorDomain, {x}] /. CalculateScan`CommonSymbols`Commented[g_, ___] :> g; 

	If[!TrueQ[domain],
		hint["Only look for changes in concavity in the interior of the domain of `1`", {f[x]}];
		If[TrueQ[cons],
			step["The domain of `1` is `2`", {f[x] == expr, format}],
			step["The domain of `1` on `2` is `3`", {f[x] == expr, cons, format}]
		];
		If[closedEndPts =!= {},
			step[Localize["The interior of the domain is `1`", 59635], {formatInterior}];
		];
		mathPrint[Row[{f[x] == expr, " when ", interiorDomain}]]
	];

	interiorDomain
]


functionDomain[args__] := 
	With[{domain = Quiet[FunctionDomain[args]]},
		(
			domain /. {Element[f_, Integers] :> Sin[\[Pi] f] == 0, NotElement[f_, Integers] :> Sin[\[Pi] f] != 0}
			
		) /; Head[domain] =!= FunctionDomain
	]


functionDomain[___] = $Failed;


(* ::Subsubsection::Closed:: *)
(*obtainAndSowSecondDerivativeIP*)


obtainAndSowSecondDerivativeIP[expr_, f_[x_]] := Block[{\[ScriptCapitalD]1, \[ScriptCapitalD]2, d1},
	\[ScriptCapitalD]1 = dIP[expr, x];
	\[ScriptCapitalD]2 = postProcessPiecewise @ dIP[\[ScriptCapitalD]1, x];
	\[ScriptCapitalD]1 = postProcessPiecewise @ \[ScriptCapitalD]1;

	hint["To find `1`, first compute `2`", {f''[x], f'[x]}];
	step[{mathWrapper[Dt["", x]][expr] == \[ScriptCapitalD]1}];
	If[FreeQ[\[ScriptCapitalD]1, Piecewise],
		mathPrint[showDetailsWrapper[f'[x] == \[ScriptCapitalD]1, StepByStepDerivative, expr, x, "CompareTo" -> \[ScriptCapitalD]1]],
		mathPrint[f'[x] == \[ScriptCapitalD]1]
	];

	d1 = If[!FreeQ[\[ScriptCapitalD]1, Piecewise], f'[x], \[ScriptCapitalD]1];

	hint["To find `1`, differentiate `2`", {f''[x], f'[x]}];
	step[{mathWrapper[Dt["", x]][d1] == \[ScriptCapitalD]2}];
	If[FreeQ[\[ScriptCapitalD]2, Piecewise],
		mathPrint[showDetailsWrapper[f''[x] == \[ScriptCapitalD]2, StepByStepDerivative, \[ScriptCapitalD]1, x, "CompareTo" -> \[ScriptCapitalD]2]],
		mathPrint[f''[x] == \[ScriptCapitalD]2]
	];
	
	\[ScriptCapitalD]2
]


dIP[expr_, x_] := lightPiecewiseExpand[iDIP[expr, x]]


iDIP[expr_, x_] /; PolynomialQ[expr, x] := D[expr, x]


iDIP[expr_, x_] /; !FreeQ[expr, Piecewise|Sign|Abs] := D[expr, x]


iDIP[expr_, x_] := Simplify[externalDForm[Simplify[D[internalDForm[expr], x], x \[Element] Reals]]]


lightPiecewiseExpand[expr_] /; !FreeQ[expr, Piecewise|DiracDelta] := 
	Module[{$eq},
		PiecewiseExpand[expr /. {Equal -> $eq, DiracDelta[f_]|Derivative[_][DiracDelta][f_] :> Piecewise[{{Indeterminate, f == 0}}]}, Method -> {
			"ConditionSimplifier" -> Simplify`SimplifyInequalities, "ValueSimplifier" -> Identity, "RefineConditions" -> False,  
			"EliminateConditions" -> False, "FactorInequalities" -> False, "ExpandSpecialPiecewise" -> False}] /. {$eq -> Equal}
	]

lightPiecewiseExpand[expr_] := expr

postProcessPiecewise[expr_] := expr /. HoldPattern[Piecewise][{{Indeterminate, cond_}}, else_] :> Piecewise[{{else, Simplify[!cond]}}, Indeterminate]


(* ::Text:: *)
(*Hack to fix TriangleWave inputs*)


internalDForm[expr_] := expr /. {TriangleWave -> triangleWave, SquareWave -> squareWave}


externalDForm[expr_] := expr /. {triangleWave -> TriangleWave, squareWave -> SquareWave, Abs'[g_] :> g/Abs[g]}


triangleWave /: Derivative[1][triangleWave] = Piecewise[{{4SquareWave[# + 1/4], Cos[2Pi #] != 0}}, Indeterminate]&;


squareWave /: Derivative[_][squareWave] = Piecewise[{{0, Sin[2Pi #] != 0}}, Indeterminate]&;


(* ::Subsubsection::Closed:: *)
(*obtainAndSowDerivativeRootsIP*)


obtainAndSowDerivativeRootsIP[\[ScriptCapitalD]2_, f_[x_], cons_] := Block[{d2dom, roots},
	
	d2dom = functionDomain[\[ScriptCapitalD]2, x, Method -> {"Reduced" -> False}];
	
	roots = IPTimeConstrained[Quiet@iReduce[\[ScriptCapitalD]2 == 0 && d2dom && cons, x, Reals]];
	If[!FreeQ[roots, Reduce|C|Element|NotElement|$Failed], SBSFail[]];
	
	roots = EliminateExtraneousRoots[\[ScriptCapitalD]2 == 0 && cons, x, roots];
	roots = refineInflectionPoints[roots, x];
	roots = roots /. HoldPattern[x == s_] /; !FreeQ[s, r:(_Root|_RootSum) /; NumericQ[r]] :> TildeTilde[x, s];
	roots
]


d2RootsShowDetailsWrapper[expr_, _, bad_] /; !FreeQ[bad, Root|RootSum|Piecewise] := expr


d2RootsShowDetailsWrapper[expr_, details_, _] := showDetailsWrapper[expr, details]


iReduce[expr_, args___] := Reduce[trigZerosToPiecewise @ expr, args] /. Equal[x_, f_] /; !FreeQ[f, _Root] :> x == Simplify[f]


trigZerosToPiecewise[expr_] := expr /. {Sin[f_] != 0 :> TriangleWave[f/(2Pi)] != 0, Cos[f_] != 0 :> TriangleWave[f/(2Pi) + 1/4] != 0}


refineInflectionPoints[True, x_] := True
refineInflectionPoints[r_Reduce, x_] := r
(*refineInflectionPoints[roots_, x_] := Or @@ And @@@ Join @@@ CombineModuli[SensibleResults[roots, {x}, GeneratedParameters -> SBSSymbol["m"]], GeneratedParameters -> SBSSymbol["m"]]*)
refineInflectionPoints[roots_, x_]:= roots


(* ::Subsubsection::Closed:: *)
(*obtainAndSowNotDifferentiableIP*)


obtainAndSowNotDifferentiableIP[\[ScriptCapitalD]2_, f_[x_], cons_] := Block[{discont, showdetails, fpeq, d2disp},
	
	discont = IPTimeConstrained[Or@@AlphaScannerFunctions`FunctionDiscontinuities[\[ScriptCapitalD]2, x]];
	discont = discont /. HoldPattern[x == s_] /; !FreeQ[s, r:(_Root|_RootSum) /; NumericQ[r]] :> TildeTilde[x, s];
	showdetails = mathWrapper[StepByStepDiscontinuities[\[ScriptCapitalD]2, x]];
	
	If[!FreeQ[discont, $Failed|C|Element|NotElement], SBSFail[]];

	fpeq = If[!FreeQ[\[ScriptCapitalD]2, Piecewise], f''[x], f''[x] == \[ScriptCapitalD]2];

	If[TrueQ[cons],
		hint["Find where `1` does not exist", {fpeq}],
		hint["Find where `1` does not exist when `2`", {fpeq, cons}]
	];
	
	d2disp = If[FreeQ[\[ScriptCapitalD]2, x], f''[x] == \[ScriptCapitalD]2, \[ScriptCapitalD]2];
	Which[
		discont === False && TrueQ[cons],
			step["`1` exists everywhere", {f''[x]}];
			mathPrint[d2RootsShowDetailsWrapper[RowTemplate["`1` exists everywhere", {d2disp}], showdetails, {discont, \[ScriptCapitalD]2}]],
		discont === False,
			step["`1` exists for all `2` when `3`", {f''[x], x, cons}];
			mathPrint[d2RootsShowDetailsWrapper[RowTemplate["`1` exists for all `2` such that `3`", {d2disp, x, cons}], showdetails, {discont, \[ScriptCapitalD]2}]],
		True,
			discont = DeleteDuplicates[discont];
			step["`1` does not exist when `2`", {f''[x], discont}];
			discont = discont /. TildeTilde -> Equal;
			mathPrint[d2RootsShowDetailsWrapper[discont, showdetails, {discont, \[ScriptCapitalD]2}]]
	];
	
	discont
]



(* ::Subsection::Closed:: *)
(*combineRootsAndDiscontAndSowIP*)


(* ::Subsubsection::Closed:: *)
(*combineRootsAndDiscontAndSowIP*)


combineRootsAndDiscontAndSowIP[roots_, discont_, f_[x_], expr_, cons_] := Block[{res},

	res = SortAndOr[Or[roots, discont /. {}->False]];
	res = refineCriticalPoints[res, x];

	(*hint["Collect results by gathering all points where `1` is `2` or does not exist", {f''[x], 0}];*)
	Which[
		res === False && TrueQ[cons],
			step["`1` has no inflection point candidates", {f[x]}];
			mathPrint[RowTemplate["`1` has no inflection point candidates", {expr}]],
		res === False,
			step["`1` has no inflection point candidates in `2`", {f[x], cons}];
			mathPrint[RowTemplate["`1` has no inflection point candidates when `2`", {expr, cons}]],
		MatchQ[res, x == _?NumericQ],
			step["The only inflection point candidate of `1` occurs at `2`", {f[x], res}];
			mathPrint[listToComma[OrList[res]]],
		True,
			step["The inflection point candidates of `1` occur at `2`", {f[x], listToCommaWithAnd[OrList[res]]}];
			mathPrint[listToComma[OrList[res]]]
	];
	
	res
]


(* ::Subsubsection::Closed:: *)
(*reduceAndSowPeriodIP*)


reduceAndSowPeriodIP[expr_, f_[x_], ocons_] :=
	Block[{cons, period},
		period = functionPeriod[expr, x];
		If[periodicFunctionQ[expr, x] && TrueQ[ocons] && !FreeQ[expr, x],
			cons = chooseinterval[expr, x, period];
			{cons, period, True},
			{ocons, 0, False}
		]
	];


(* ::Subsection::Closed:: *)
(*Utilities*)


$TimeLimit = Infinity;
SetAttributes[IPTimeConstrained, HoldFirst];


IPTimeConstrained[expr_] := IPTimeConstrained[expr, 1, $Failed]


IPTimeConstrained[expr_, fac_] := IPTimeConstrained[expr, fac, $Failed]


IPTimeConstrained[expr_, fac_, ___] /; !TrueQ[fac $TimeLimit > 0] || fac $TimeLimit === \[Infinity] := expr


IPTimeConstrained[expr_, fac_, failexpr_] := TimeConstrained[expr, fac $TimeLimit, failexpr]


(* ::Subsection::Closed:: *)
(*obtainAndSowConcavityClassification*)


(* ::Subsubsection::Closed:: *)
(*obtainAndSowConcavityClassification*)


obtainAndSowConcavityClassification[ints_, candrules_, testpts_, f_[x_], \[ScriptCapitalD]2_] :=
	Block[{fpeq, table, signs, grid, lintpts, evals, tab},
		signs = Block[{evals = niceN[Quiet[\[ScriptCapitalD]2 /. {x->#}& /@ testpts] /. $badNumbersPattern -> Undefined]},
			Sign[evals]
		];
		{ints, signs}
	]



(* ::Subsection::Closed:: *)
(*sowAllInflectionPointCandidates*)


(* ::Subsubsection::Closed:: *)
(*sowAllInflectionPointCandidates*)


sowAllInflectionPointCandidates[expr_, f_[x_], domain_, restrictedPQ_, \[ScriptCapitalD]2_] :=
	(
		If[TrueQ[domain],
			hint[Localize["Determine the inflection points of `1`", 67540], {f[x]}];
			step[Localize["For every value `1` on the real line, either `2` or `3` does not exist. This means `4` is never concave up or concave down and we conclude that", 67541], 
				{x, f''[x] == 0, f''[x], f[x] == expr}];
			mathPrint[RowTemplate[Localize["`1` has no inflection points", 67542], {expr}]],
			hint[Localize["Determine the inflection points of `1` over `2`", 67543], {f[x], domain}];
			step[Localize["For every value `1`, either `2` or `3` does not exist. This means `4` is never concave up or concave down and we conclude that", 67544], 
				{domain, f''[x] == 0, f''[x], f[x] == expr}];
			mathPrint[RowTemplate[Localize["`1` has no inflection points over `2`", 67545], {expr, domain}]]
		];
		
		If[restrictedPQ,
			unrestrictIPPeriodicity[{}, expr, \[ScriptCapitalD]2, f[x]]
		];
		
		{}
	)


(* ::Subsection::Closed:: *)
(*sowInflectionPoints*)


(* ::Subsubsection::Closed:: *)
(*sowInflectionPoints*)


sowInflectionPoints[{ints_, concavity_}, expr_, f_[x_], cons_, period_] /; periodicInflectionEndPointQ[ints, period] :=
	Block[{left, nints, nconcavity},
		left = ints[[1, 1, 1]];
		nints = Prepend[ints, Interval[{ints[[-1, 1, 1]] - period, left}]];
		nconcavity = Prepend[concavity, Last[concavity]];
		
		hint["List an extra interval of concavity in order to test if an inflection point occurs at `1`", {x == left}];
		step["To determine if an inflection point occurs at `1`, use the period of `2` to list an extra interval of concavity", 
			{x == left, If[periodicFunctionQ[expr, x], f[x], f''[x]]}];
		mathPrint[Column[
			concavityResultText[f[x], ##]& @@@ Transpose[{nints, nconcavity}]
		]];
		
		sowInflectionPoints[{nints, nconcavity}, expr, f[x], cons]
	]


sowInflectionPoints[{ints_, concavity_}, expr_, f_[x_], cons_, _] :=
	sowInflectionPoints[{ints, concavity}, expr, f[x], cons]


sowInflectionPoints[{ints_, concavity_}, expr_, f_[x_], cons_] := 
	Block[{dcands, dconts, dpos, ips, gpts},
		dcands = DeleteDuplicates[(Join @@ ints[[All, 1]])[[2 ;; -2]]];
		dconts = Select[dcands, !possiblyContinuousQ[expr, {x, #}]&];
		
		dpos = Flatten[Position[ints, Interval[{_, Alternatives @@ dconts}], {1}, Heads -> False]];
		
		If[Length[dpos] > 0,
			ips = ints[[dpos, 1, 2]];
			gpts = Replace[ints, (# -> Style[#, GrayLevel[0.75]]& /@ dconts), {3}];
			hint["Determine the interval endpoints where `1` is discontinuous", {f[x]}];
			If[Length[dpos] == 1,
				step["Since `1` is discontinuous at `2`, mark the corresponding endpoint gray. This indicates an inflection point cannot occur there", {f[x], listToComma[Thread[x == ips]]}],
				step["Since `1` is discontinuous at `2`, mark the corresponding endpoints gray. This indicates inflection points cannot occur there", {f[x], listToComma[Thread[x == ips]]}]
			];
			mathPrint[Column[
				concavityResultText[f[x], ##]& @@@ Transpose[{gpts, concavity}]
			]]
		];
			
		sowInflectionPoints[{ints, concavity, dpos}, expr, f[x], cons]
	]


sowInflectionPoints[{ints_, concavity_, dpos_}, expr_, f_[x_], cons_] :=
	Block[{groups, cc, ips},
		groups = Partition[concavity, 2, 1];
		cc = Complement[Flatten[Position[groups, {-1,1}|{1,-1}, {1}]], dpos];
		ips = Select[Transpose[{ints[[cc, 1, 2]], ints[[cc + 1, 1, 1]]}], Apply[Equal]][[All, 1]];
		groups = Delete[groups, List /@ dpos];
		
		hint["Find where the points of inflection occur by examining the intervals of concavity"];
		Which[
			Length[ips] > 0,
				concavityChangeStep[f[x], Count[groups, {-1, 1}], Count[groups, {1, -1}], cons, Length[dpos] == 0];
				If[Length[ips] > 1,
					step["These points of change correspond to the inflection points which occur at"],
					step["This point of change corresponds to the only inflection point and occurs at"]
				];
				mathPrint[listToComma[Thread[x == ips]]],
			SameQ @@ concavity,
				If[Length[dpos] == 0,
					step["The concavity of `1` never changes", {f[x]}],
					step["At values where `1` is continuous, the concavity of `1` never changes", {f[x]}]
				];
				mathPrint[RowTemplate["`1` has no inflection points", {expr}]],
			True,
				If[Length[dpos] == 0,
					step["The concavity of `1` never changes from concave up to concave down or vice versa", {f[x]}],
					step["At values where `1` is continuous, the concavity of `1` never changes from concave up to concave down or vice versa", {f[x]}]
				];
				mathPrint[RowTemplate["`1` has no inflection points", {expr}]]
		];
		
		Thread[x -> ips]
	]


periodicInflectionEndPointQ[ints_, period_] := 
	period > 0 && ints[[-1, 1, 2]] - ints[[1, 1, 1]] == period


concavityChangeStep[f_[x_], np_, 0, True, False] :=
	step["At values where `1` is continuous, the concavity of `1` changes from concave down to concave up `2`.", {f[x], numtimes[np]}]


concavityChangeStep[f_[x_], 0, pn_, True, False] :=
	step["At values where `1` is continuous, the concavity of `1` changes from concave up to concave down `2`.", {f[x], numtimes[pn]}]


concavityChangeStep[f_[x_], np_, pn_, True, False] := 
	step["At values where `1` is continuous, the concavity of `1` changes from concave down to concave up `2` and from concave up to concave down `3`.", {f[x], numtimes[np], numtimes[pn]}]


concavityChangeStep[f_[x_], np_, 0, cons_, False] :=
	step["At values where `1` is continuous, the concavity of `1` when `2` changes from concave down to concave up `3`.", {f[x], cons, numtimes[np]}]


concavityChangeStep[f_[x_], 0, pn_, cons_, False] :=
	step["At values where `1` is continuous, the concavity of `1` when `2` changes from concave up to concave down `3`.", {f[x], cons, numtimes[pn]}]


concavityChangeStep[f_[x_], np_, pn_, cons_, False] := 
	step["At values where `1` is continuous, the concavity of `1` when `2` changes from concave down to concave up `3` and from concave up to concave down `4`.", {f[x], cons, numtimes[np], numtimes[pn]}]


concavityChangeStep[f_[x_], np_, 0, True, True] :=
	step["The concavity of `1` changes from concave down to concave up `2`.", {f[x], numtimes[np]}]


concavityChangeStep[f_[x_], 0, pn_, True, True] :=
	step["The concavity of `1` changes from concave up to concave down `2`.", {f[x], numtimes[pn]}]


concavityChangeStep[f_[x_], np_, pn_, True, True] := 
	step["The concavity of `1` changes from concave down to concave up `2` and from concave up to concave down `3`.", {f[x], numtimes[np], numtimes[pn]}]


concavityChangeStep[f_[x_], np_, 0, cons_, True] :=
	step["The concavity of `1` when `2` changes from concave down to concave up `3`.", {f[x], cons, numtimes[np]}]


concavityChangeStep[f_[x_], 0, pn_, cons_, True] :=
	step["The concavity of `1` when `2` changes from concave up to concave down `3`.", {f[x], cons, numtimes[pn]}]


concavityChangeStep[f_[x_], np_, pn_, cons_, True] := 
	step["The concavity of `1` when `2` changes from concave down to concave up `3` and from concave up to concave down `4`.", {f[x], cons, numtimes[np], numtimes[pn]}]


numtimes[1] = "once";
numtimes[2] = "twice";
numtimes[n_] := RowTemplate["`1` times", {n}]


(* ::Subsubsection::Closed:: *)
(*unrestrictIPPeriodicity*)


(* ::Text:: *)
(*TODO incorporate original constraint to restrict intervals.*)


unrestrictIPPeriodicity[{}, expr_, \[ScriptCapitalD]2_, f_[x_]] /; periodicFunctionQ[\[ScriptCapitalD]2, x] := 
	(
		hint["Finally, generalize the result to all periods of `1`", {f[x]}];
		step["Since `1` has no inflection points over a single period, we conclude that", {f[x]}];
		mathPrint[RowTemplate["`1` has no inflection points", {expr}]];
		
		{}
	)


unrestrictIPPeriodicity[res_, expr_, \[ScriptCapitalD]2_, f_[x_]] /; periodicFunctionQ[\[ScriptCapitalD]2, x] := 
	Block[{k, period, term, newres},
		k = If[x =!= Symbol["k"], SBSSymbol["k"], SBSSymbol["n"]];
		period = functionPeriod[\[ScriptCapitalD]2, x];
		term = period*k;
		newres = res /. (x -> a_) :> (x -> a + term);
		
		hint["Finally, generalize the results to all periods of `1`", {f[x]}];
		step["Add `1` for `2` to each value", {period*k, k \[Element] Integers}];
		mathPrint[Column[RowTemplate["`1` <GrayText>for</GrayText> `2`", {#, k \[Element] Integers}]& /@ Equal @@@ newres]];
		
		{res, "Period" -> period}
	]


unrestrictPeriodicityIP[res_, __] := res


(* ::Subsection::Closed:: *)
(*Utilities*)


(* ::Subsubsection::Closed:: *)
(*BlockIPSteps*)


SetAttributes[BlockIPSteps, HoldFirst];


BlockIPSteps[code_] :=
	Block[{sowIPIntro, sowIPDefinition, sowLinearIP, sowConcavityIntro, sowConcavityDefinition, 
			sowLinearConcavity, restrictedIntervalStep, sowConcavitySummary},
		
		SetAttributes[{sowIPIntro, sowIPDefinition, sowLinearIP, sowConcavityIntro, sowConcavityDefinition, 
			sowLinearConcavity, restrictedIntervalStep, sowConcavitySummary}, HoldAllComplete];
		
		code
	]


reduceAndSowPeriodIP[expr_, f_[x_], ocons_] :=
	Block[{cons, period},
		period = functionPeriod[expr, x];
		If[periodicFunctionQ[expr, x] && TrueQ[ocons] && !FreeQ[expr, x],
			(* then *)
			cons = chooseinterval[expr, x, period];
			{cons, period, True},
			(* else *)
			{ocons, 0, False}
		]
	];


Clear[functionPeriod]
functionPeriod[expr_, x_] := functionPeriod[expr, x] = Block[{Internal`PrecAccur, res},
	Internal`PrecAccur[_] = \[Infinity];
	res = Assuming[x \[Element] Reals, Periodic`PeriodicFunctionPeriod[iTrigReduce@expr, x, True]];
	If[TrueQ[Negative[res]], -res, res];
	res = refinePeriod[expr, x, res];
	
	If[TrueQ[Im[res] != 0],
		$Failed,
		res
	]
]

iTrigReduce[expr_] /; FreeQ[expr, (Sin|Cos)[v_ /; !FreeQ[v, Sin|Cos]]] := TrigReduce[expr]
iTrigReduce[expr_] := expr

(* FunctionPeriod can be off by an integer multiple. Test a factor of 2. See StepByStepInflectionPoints[Cos[Cos[x]], x] *)
refinePeriod[expr_, x_, res_?NumericQ] /; Quiet[PossibleZeroQ[(expr /. x -> x + res/2) - expr]] := res/2
refinePeriod[__, res_] := res



(* ::Subsubsection::Closed:: *)
(*periodicFunctionQ*)


periodicFunctionQ[expr_, x_] := functionPeriod[expr, x]=!=$Failed


(* ::Subsection::Closed:: *)
(*obtainAndSowIntervalsAndTestPtsIP*)


(* ::Subsubsection::Closed::*)
(*obtainAndSowIntervalsAndTestPtsIP*)
obtainAndSowIntervalsAndTestPtsIP[False,True,f_[x_]]:=Null
obtainAndSowIntervalsAndTestPtsIP[cands_,domain_,f_[x_]]:= Block[
	{
		candsbd, candrules={}, canddom=True, ints, intervalsDisplay,
		testpts, bd, nonlinints, nonlintestpts
	},
	candsbd=Or@@DeleteDuplicates[OrList[Or@@(inequalityBoundary[#,x]&/@OrList[cands/.TildeTilde->Equal])]];
	If[candsbd=!=False,candrules={ToRules[candsbd]};
	If[!MatchQ[candrules,{{_Rule}..}],SBSFail[]];
	canddom=And@@Through[(Or[#<x,#>x]&/.candrules)[x]]];
	ints=inequalitiesToIntervals[domain&&canddom,x];
	testpts=chooseTestPoint/@ints;
	{ints,testpts}=Transpose[Select[Transpose[{ints,testpts}],(inequalityClosure[domain]/.x->Last[#])=!=False&]];
	{ints,candrules,testpts}
]


(* ::Subsection::Closed:: *)
(*shiftAndSowDomainIP*)


(* ::Subsubsection::Closed::*)(*shiftAndSowDomainIP*)shiftAndSowDomainIP[cands_,domain:(a_<=x_<b_),f_[x_],period_]/;period==b-a:=Block[{min,newdomain,newcands},min=domainShift[cands,{x,a,b}];
(newdomain=min<=x<min+period;
newcands=shiftInequality[cands,x,period,min];
hint["Shift the domain so an inflection point candidate is on its boundary"];
step["Since the domain is a single period of `1`, it can shift by any amount. Shifting to place the smallest inflection point candidate on the boundary of the domain ensures all intervals of concavity found will not be broken up",{f[x]}];
mathPrint[RowTemplate["Determine concavity over `1`",{newdomain}]];
{newcands,newdomain})/;min=!=$Failed]


shiftAndSowDomainIP[cands_,domain_,__]:={cands,domain}


domainShift[cands_,{x_,a_,b_}]/;MatchQ[OrList[cands],{(x==_)..}]:=With[{min=Min[OrList[cands][[All,2]]]},min/;min>a]


domainShift[cands_,{x_,a_,b_}]/;!FreeQ[OrList[cands],a<=x<_,{1}]:=With[{min=First[Cases[OrList[cands],a<=x<c_:>c]]},min/;min>a]


domainShift[___]=$Failed;


shiftInequality[args__]:=iShiftInequality[args]/.{HoldPattern[Or][a___,d1_<=x_<d2_,b___,Inequality[d3_,Less,x_,Less,d1_]|Less[d3_,x_,d1_],c___]:>Or[a,b,d3<x<d2,c]}


iShiftInequality[(h:And|Or)[args__],x_,period_,min_]:=h@@(iShiftInequality[#,x,period,min]&/@{args})


iShiftInequality[HoldPattern[Inequality][c1_,h1_,x_,h2_,c2_],x_,period_,min_]/;c1<min:=Inequality[c1+period,h1,x,h2,c2+period]


iShiftInequality[HoldPattern[Inequality][x_,h_,c_],x_,period_,min_]/;c<min:=Inequality[x,h,c+period]


iShiftInequality[HoldPattern[Inequality][c_,h_,x_],x_,period_,min_]/;c<min:=Inequality[c+period,h,x]


iShiftInequality[(h:Less|LessEqual|Greater|GreaterEqual)[c1_,x_,c2_],x_,period_,min_]/;c1<min:=h[c1+period,x,c2+period]


iShiftInequality[(h:Less|LessEqual|Greater|GreaterEqual)[x_,c_],x_,period_,min_]/;c<min:=h[x,c+period]


iShiftInequality[(h:Less|LessEqual|Greater|GreaterEqual)[c_,x_],x_,period_,min_]/;c<min:=h[c+period,x]


iShiftInequality[x_==c_,x_,period_,min_]:=x==Mod[c,period,min]


iShiftInequality[expr_,__]:=expr


(* ::Chapter::Closed:: *)
(*epilog*)


End[]


EndPackage[]
