(* ::Package:: *)

(* ::Text:: *)
(*This is a stand-alone Mathematica package that suggest plot ranges for mathematical functions.*)
(*For the use outside of Wolfram|Alpha, the code of this package has been split of from *)
(*CalculateScan/PlotterScanner.m and other packages.*)
(**)
(*Note that some private functions from this package are used inside CalculateScan/PlotterScanner.m*)


(* ::Chapter::Closed:: *)
(*prolog*)


BeginPackage["AlphaScannerFunctions`", {"AlphaScannerFunctions`CommonFunctions`"}]


Needs["DifferentialEquations`InterpolatingFunctionAnatomy`"]; 


(* ::Input:: *)
(*(* No other Wolfram|Alpha packages can be included here! *)
(*   If used within WA,  we need to know special functions. *)*)


CalculateUtilities`SuggestPlotRanges`StandAloneModeQ === True


MakeScaledPlots::usage = "Generates code for held plots and potential annotations." 


SuggestPlotRange::usage = 
"SuggestPlotRange[fun, var, options] suggests 'good' plot ranges to plot the univariate numerical function fun
as a function of the independent variable var. func can also be a list of functions."


Begin["`SuggestPlotRange`Private`"]


(* ::Chapter:: *)
(*code for plot ranges in 1D*)


(* ::Section:: *)
(*Notes*)


(* ::Text:: *)
(*The main functions defined below are:*)
(**)
(*1) MakeScaledPlots. The main plot-range sugegsting function used in PlotterScanner.*)
(**)
(*2) Get1DRange. Called with a single function, determines a plotting variable and suggests ranges.*)
(**)
(*3) SuggestPlotRanges1D. For use outside of WA in the Wolfram calculator.*)
(**)
(*This Section of code is maintianed by mtrott.*)


(* ::Section::Closed:: *)
(*Debug*)


(* ::Input:: *)
(*debugComputerName = "mtrott2mac";   *)
(*debugMode = $MachineName === debugComputerName;   *)
(*debugModePlotRangeDetails = $MachineName === debugComputerName;   *)


(* ::Input:: *)
(*If[debugMode,  Get[ "/home/usr1/mtrott/WolframWorkspaces/Base/Alpha/Source/CalculateScan/CommonFunctions.m"]];   *)


(* ::Text:: *)
(*   Load package   *)


(* ::Input:: *)
(*Needs["DifferentialEquations`InterpolatingFunctionAnatomy`"]   *)


(* ::Input:: *)
(*CalculateTimeConstrained = TimeConstrained;*)


(* ::Input:: *)
(*$reColor1=Hue[0.67,.6,0.6];*)
(*$reColor2=Hue[0.906068,0.6,0.6];*)
(*$imColor=RGBColor[1,0.3,0];$plotstyle = {{Directive[$reColor1, AbsoluteThickness[1]]},{Directive[$reColor2,AbsoluteThickness[1]]}};*)
(*$plotstyleim = Directive[$imColor, AbsoluteThickness[1]];*)


(* ::Input:: *)
(*$genericscanneroptions = {};*)


(* ::Section::Closed:: *)
(*Standalone mode*)


If[CalculateUtilities`SuggestPlotRanges`StandAloneModeQ === False, Null,
   ToExpression["CalculateTimeConstrained = TimeConstrained"]]


(* ::Section:: *)
(*Code for Get1DRange and GetReal1DRange*)


(* ::Subsection:: *)
(*Code for 'good' independent variable ranges  (Get1DRange)*)


(* ::Text:: *)
(*This is some simple-minded attempt to get plot ranges that might be used for ParametricPlot, PolarPlot. *)
(*Generically, one cannot reduce this to the 1D case because of dependencies and correlations.*)


(* ::Subsubsection::Closed:: *)
(*General (function Get1DRange)*)


(* ::Text:: *)
(*For parametric plotting of periodic functions, use one period only to avoid 'overplotting'.*)


reduceRangeToPeriod[PRData[{{xl_, xu_}, ___}, ___], x_, "Trig" -> True] := 
With[{\[CapitalDelta]x = Abs[xu - xl], xM = Mean[{xl, xu}]},
     {x, xM - \[CapitalDelta]x/4, xM + \[CapitalDelta]x/4}] 


reduceRangeToPeriod[PRData[{{xl_, xu_}, ___}, ___], x_, "Trig" -> True, "Periods" -> n_] := 
With[{\[CapitalDelta]x = Abs[xu - xl], xM = Mean[{xl, xu}]},
     {x, xM - n \[CapitalDelta]x/4, xM + n \[CapitalDelta]x/4}] 


reduceRangeToPeriod[PRData[{{xl_, xu_}, ___}, ___], x_, "Trig" -> False, ___] := {x, xl, xu} 


uniteGPlotRangesGDR[prsList_, boundedQ_, intersections_] := 
 Module[{}, 
         If[TrueQ[Equal @@ prsList], prsList[[1]],
             Which[MatchQ[Union[Length /@ prsList], {1} | {1, 2}],
                    {Append[unitePlotRangeDataGDR[First /@ prsList, boundedQ, intersections],
                            "HorizontalPlotRangeType" -> "ShowSomething"]},
                    Union[Length /@ prsList] === {2},
                    {Append[unitePlotRangeDataGDR[First /@ prsList, boundedQ, intersections], 
                            "HorizontalPlotRangeType" -> "ShowCenterPart"], 
                     Append[unitePlotRangeDataGDR[Last /@ prsList, boundedQ, intersections], 
                            "HorizontalPlotRangeType" -> "ShowEnlargedMore"]}
                    ]
            ]
        ]


unitePlotRangeDataGDR[prsDataList_, boundedQ_, intersections_] := 
 Module[{theRanges, theUnitedRange, theOptions, workingPrecs, theUnitedOptions}, 
         theRanges = Flatten[#[[1, 1]]] & /@ Flatten[prsDataList];
         theUnitedRange = rangeCompromiseGDR[theRanges, boundedQ, intersections];
         theOptions = If[Length[#[[1]]] > 1, #[[1, 2]], {}] & /@ Flatten[prsDataList];
         workingPrecs = (WorkingPrecision /. #) & /@ theOptions;
         theUnitedOptions = If[And @@ (NumberQ /@ workingPrecs), WorkingPrecision -> Max[workingPrecs], {}];
         PRData[{theUnitedRange, theUnitedOptions}]
        ]


rangeCompromiseGDR[ranges_, boundedQ_, intersections_] := 
 Module[{xys, try1, preRange, intervalLength, minDistRight, minDistLeft},
      preRange = 
         Which[boundedQ === True && IntervalMemberQ[Interval[ranges[[1]]], Interval[ranges[[2]]]], ranges[[1]],
               boundedQ === True && IntervalMemberQ[Interval[ranges[[2]]], Interval[ranges[[1]]]], ranges[[2]],
               boundedQ === True, {Min[#], Max[#]} &[ Transpose[ranges] ],
               (IntervalIntersection @@ (Interval /@ ranges)) =!= Interval[],
               xys = Transpose[ranges];
               try1 = Sort[{Mean[#1], Mean[#2]} & @@ xys]; 
               If[Unequal @@ try1, try1, {Min[xys], Max[xys]}],
               True, {Min[#], Max[#]} &[ Transpose[ranges] ]
              ];
       (* try to include nearby intersections *)
       Which[intersections === {}, preRange,
             Or @@ (IntervalMemberQ[Interval[preRange], #]& /@ intersections), preRange,
             True,
             intervalLength = Abs[Subtract @@ preRange];
             minDistRight = Min[Abs[Max[preRange] - #]& /@ intersections];
             minDistLeft = Min[Abs[Min[preRange] - #]& /@ intersections];
             Which[minDistRight < 1.5 intervalLength,
                   {Min[preRange], Max[preRange] + 1.5 minDistRight},
                   minDistLeft < 1.5 intervalLength,
                   {Min[preRange] -  1.5 minDistRight, Max[preRange]},
                   True, preRange
                  ] 
             ]
 
          ] 


boundedQ[f_, x_] := 
With[{F = Re[f]^2 + Im[f]^2},
   Module[{max = TimeConstrained[Maximize[F, x], 0.5] // Quiet},
          If[Head[max] === List, TrueQ[max[[1]] < Infinity], False]]
  ]


boundedQ[fL_List, x_] := And @@ (boundedQ[#, x]& /@ fL)


getUserVariables1D[Hold[expr_], standAlone_, linearQ___] := getUserVariables[Hold[expr], standAlone, linearQ]
getUserVariables1D[Hold[l_List],  standAlone_, linearQ___] := getUserVariables[Hold @@ {DeleteCases[l, _Symbol, {1}]},  standAlone, linearQ]
getUserVariables1D[Hold[s_Symbol],  standAlone_, linearQ___] := s


(* ::Input:: *)
(*getUserVariables1D[Hold[Evaluate[{E^(x-1) ,E^(1-x)}]], True]*)


simpleIntersections[{lhs_, rhs_}, x_] := 
       CalculateTimeConstrained[Quiet[Cases[Chop[ x/. Solve[lhs == rhs, x]], _Real]], 0.12, {}]

simpleIntersections[_, x_] := {}


(* ::Input:: *)
(*simpleIntersections[ {(1/(1 + x))^(1/x), 0.6}, x]*)


Options[Get1DRange] = {"Periods" -> 1};


(* linear case *)
Get1DRange[x_Symbol + c_?NumericQ, opts:OptionsPattern[]] := 
{{x, -Abs[c] -0.2 Abs[c], -Abs[c] + 0.2 Abs[c]}, {x, -1.5 Abs[c], 0.5 Abs[c]}} 


Get1DRange[expr_, opts:OptionsPattern[]] := 
Module[{vars = If[MatchQ[#,_Symbol],{#},#]&@getUserVariables1D[Hold[expr], True](*, 
        aux, xm1, plotRanges, expr1, scaleInfo, prs, prsList1G, bQ, ov, sIs*)},  
       If[Length[vars] =!= 1, $Failed,
          If[MatchQ[#, {_List..}], {#1, Sequence @@ Sort[{#2, #3}, Less]}& @@@ #, #]& @ Get1DRangeWithVariable[expr, vars[[1]], opts]
         ]
       ]


(* for autocompletion purposes with specified variable *)
Get1DRange[expr_, var_, opts:OptionsPattern[]] := 
Module[{},  
       If[(* quick check if it is a plottable, univariate function *) NumericQ[expr /. var -> (1/Pi + E^2)] || NumericQ[expr /. var -> (Sqrt[5]+135/867)], 
          If[MatchQ[#, {_List..}], {#1, Sequence @@ Sort[{#2, #3}, Less]}& @@@ #, #]& @ Get1DRangeWithVariable[expr, var, opts],
          $Failed]
       ]


Options[Get1DRangeWithVariable]= {"Periods" -> Automatic, "Range"-> "Narrow"}
Get1DRangeWithVariable[expr_, x_, opts:OptionsPattern[]] := 
Module[{aux, xm1, plotRanges, expr1, scaleInfo, prs = {}, prsList1G, bQ, sIs, period, trigFlag, periodicFlag, ovPeriods, ovRange},
   ovRange = OptionValue["Range"];
If[aux = Get1DRangeSpecialCase[expr, x]; Head[aux] === List,  
  Which[ ovRange === "Wide", Take[aux, -1],
         True, Take[aux, +1]
       ],    
(* try reducing to 1D *)   
   xm1 = x;  
   plotRanges = 
   If[Head[expr] =!= List,
      If[plottableFunctionQ[expr, xm1, True] || MatchQ[expr , xm1 + _],  
         {expr1, scaleInfo} = rescale[expr, xm1];    
          prs = Flatten[proposedPlotRangeAndFunctionFeaturesRRLPRSC[expr1, xm1]] /. fuzzFactorRationalTrig -> 1, 
         (* not plottable *) {}], 
      (* lists of functions *) 
      If[(And @@ ((plottableFunctionQ[#, xm1, True] || MatchQ[# , xm1 + _])& /@ expr)) ||
         (* {others ... , x , others ...} *)
         With[{nonxParts = DeleteCases[expr, _. xm1]}, nonxParts =!= {1} && (And @@ (plottableFunctionQ[#, xm1, True]& /@ nonxParts))],   
         {expr1, scaleInfo} = Transpose[rescale[#, xm1]& /@ expr];  
         prsList1G = (Flatten[refineRanges @ largerPlotRangeSanityCheck @ 
                              proposedPlotRangeAndFunctionFeaturesPre[#, xm1]] /. fuzzFactorRationalTrig -> 1) & /@  expr1;
         bQ = boundedQ[expr1, xm1]; 
         sIs = simpleIntersections[expr1, xm1];
         prs = ExpandAll @ uniteGPlotRangesGDR[prsList1G, bQ, sIs],
         (* not plottable *) {}
       ]
     ];   

     (* are trigs involved? *)
     trigFlag = MemberQ[ExpToTrig[expr], (Cos| Sin | Tan | Cot | Csc | Sec)[a_. xm1 + b_.]/; Im[a] == 0, {0, \[Infinity]}];

     periodicFlag = Block[{Periodic`Private`PDWellDefinedFunctionQ},
                           Periodic`Private`PDWellDefinedFunctionQ[___] := True;
                           period = Periodic`PeriodicFunctionPeriod[expr, xm1];
                           If[NumericQ[period], True, False]];

     If[prs === {}, $Failed,
         ovPeriods = OptionValue["Periods"];
         Which[ MatchQ[ovPeriods, _?NumericQ ],  
                   If[periodicFlag === True, 
                      (* potentially use period... the if statement currently evaluates to same result in both cases *)
                      MapAt[reduceRangeToPeriod[#, xm1, "Trig" -> trigFlag, "Periods" -> ovPeriods]&, prs, 1],
                      MapAt[reduceRangeToPeriod[#, xm1, "Trig" -> trigFlag, "Periods" -> ovPeriods]&, prs, 1]
                     ],
               ovRange === "Narrow", 
                        If[Length[prs] === 2, reduceRangeToPeriod[#, xm1, "Trig" -> trigFlag]& /@ Take[prs, +1], Take[prs, +1]],
               ovRange === "Wide", Take[prs, -1],
               True, $Failed]
               ] /. (* take what we have *) PRData[{{xl_, xu_}, ___}, ___] :> {xm1, xl, xu}
    ]
     ]


(* ::Subsubsection::Closed:: *)
(*Tests for Get1DRange*)


(* ::Input:: *)
(*Get1DRangeWithVariable[Sin[x], x]*)


(* ::Input:: *)
(*Get1DRange[Sin[x]]*)


(* ::Input:: *)
(*Get1DRange[Log[3^x]]*)


(* ::Input:: *)
(*Get1DRange[Tan[x]]*)


(* ::Input:: *)
(*Get1DRange[((x^3+5*x^2)^.5)/x]*)


(* ::Input:: *)
(*Get1DRange[((x^3+5*x^2)^(1/2))/x]*)


(* ::Input:: *)
(*Get1DRange[Sin[x]]*)


(* ::Input:: *)
(*Get1DRange[Sin[x],"Periods"->3]*)


(* ::Input:: *)
(*Get1DRange[1-Exp[-InverseErf[z^2]]]*)


(* ::Input:: *)
(*Get1DRange[Cos[x]] *)


(* ::Input:: *)
(*Get1DRange[Cos[2x]+x-1] *)


(* ::Input:: *)
(*Get1DRange[{E^(x-1) ,E^(1-x)}] *)


(* ::Input:: *)
(*Get1DRange[{x,Log[x]}] *)


(* ::Input:: *)
(*Get1DRange[{x^2, x^3}] *)


(* ::Input:: *)
(*Get1DRange[{1+x,3}]*)


(* ::Input:: *)
(*Get1DRange[-2x-3] *)


(* ::Input:: *)
(*Get1DRange[2x-5] *)


(* ::Input:: *)
(*Get1DRange[-1+E^x-x] *)


(* ::Input:: *)
(*Get1DRange[x+3] *)


(* ::Input:: *)
(*Get1DRange[-x+3] *)


(* ::Input:: *)
(*Get1DRange[2x+3] *)


(* ::Input:: *)
(*Get1DRange[Sin[Sqrt[2] x] + Sin[Sqrt[3] x]] *)


(* ::Input:: *)
(*Get1DRange[{Cos[x] Sqrt[1 + Cos[3/2 x]], Sin[x] Sqrt[1 + Cos[3/2 x]]}] *)


(* ::Input:: *)
(*Get1DRange[Tan[x^2]+Sin[x^2]/Cos[1/2]]*)


(* ::Input:: *)
(*Get1DRange[{Cos[x] , Sin[x] }] *)


(* ::Input:: *)
(*Get1DRange[{-2x-3, 2x+5}] *)


(* ::Input:: *)
(*Get1DRange[{(4+y)/(1+y),(7.7+y)/(22.090000000000003+y)}]*)


(* ::Input:: *)
(*Get1DRange[ {(1/(1 + x))^(1/x), 0.6}]*)


(* ::Input:: *)
(*Get1DRange[ {Sin[x - x^2],x^2 }]*)


(* ::Input:: *)
(*Get1DRange[ Sin[x - x^2] -x^2 ]*)


(* ::Input:: *)
(*Get1DRange[{-(1/Sqrt[-2+x^2]),ArcTan[Sqrt[-2+x^2]/Sqrt[2]]/Sqrt[2]}] *)


(* ::Input:: *)
(*(* should fail *)*)
(*Get1DRange[y == Sin[x]]*)


(* ::Subsubsection::Closed:: *)
(*Special cases*)


Get1DRangeSpecialCase[{a1_. (Cos | Sin | Tan | Cot | Sec | Csc)[k1_. x_Symbol],
                                                    a2_.  (Cos | Sin | Tan | Cot | Sec | Csc)[k2_. x_Symbol]}, x_] := 
With[{p =  2Pi/GCD[k1, k2]}, {{x, -p/2, p/2}, {x, -2p, 2p}}] /; 
NumericQ[a1] &&  NumericQ[a2]&&IntegerQ[k1]&&IntegerQ[k2]


Get1DRangeSpecialCase[{a1_. (Cos | Sin | Tan | Cot | Sec | Csc)[k1_. x_Symbol],
                       a2_.  (Cos | Sin | Tan | Cot | Sec | Csc)[k2_. x_Symbol]}, x_] := 
With[{p = LCM[Denominator[k1], Denominator[k2]] 2Pi}, {{x, -p/2, p/2}, {x, -2p, 2p}}] /;
 NumericQ[a1] &&  NumericQ[a2]&&
   ( IntegerQ[k1]|| Head[k1] === Rational) &&  ( IntegerQ[k2]|| Head[k2] === Rational)


Get1DRangeSpecialCase[{a1_. (Cos | Sin | Tan | Cot | Sec | Csc)[k1_. x_Symbol],
                       a2_.  (Cos | Sin | Tan | Cot | Sec | Csc)[k2_. x_Symbol]}, x_] := 
With[{p =  2Pi/GCD[k1, k2]}, {{x, -p/2, p/2}, {x, -2p, 2p}}] /; 
NumericQ[a1] && NumericQ[a2] && IntegerQ[k1] && IntegerQ[k2]


(* for polar plots of periodic functions *)
Get1DRangeSpecialCase[{Cos[x_Symbol?(Not[NumericQ[#]]&)] r_, Sin[x_Symbol?(Not[NumericQ[#]]&)] r_}, x_] := 
Module[{},
      userRadVars = Union @ Cases[r, _Symbol?(Context[#] === "Global`"&), \[Infinity]];
      If[userRadVars =!= {x}, $Failed,
         radPeriod = Block[{Periodic`Private`PDWellDefinedFunctionQ}, 
         				Periodic`Private`PDWellDefinedFunctionQ[___] := True;
                        Periodic`PeriodicFunctionPeriod[Cos[x] r, x]] ;
         				If[NumericQ[radPeriod],
            				{{x, -radPeriod/2, radPeriod/2}, {x, -4 radPeriod/2, 4 radPeriod/2}},
            				$Failed
           				]
      ]
]


Get1DRangeSpecialCase[{a1_.  x_Symbol^\[Alpha]1_.,a2_.  x_Symbol^\[Alpha]2_.}, x_] := 
                      {{x, -1, 1}, {x, -2, 2}} /; NumericQ[a1] &&  NumericQ[a2] && Not[NumericQ[x]]


(* degenerate case *)
Get1DRangeSpecialCase[x_Symbol, x_] := {{x, -1, 1}, {x, -10, 10}}


(* ::Subsection::Closed:: *)
(*Code for 'good' parameter variable ranges  (function GetParamatrized1DRange)*)


(* ::Text:: *)
(*The result from GetParamatrized1DRange is for interactive use. This means*)
(*- the range can be larger than what Get2Range would return because at the end we plot only a 1D function which is much faster*)
(*- if possible, the range should be 'nice', meaning we do not want reals with many digits*)
(*- we want the range large enough to potentially see the function 'blow up' at the interval boundaries, but we want to start with a nice value*)


(* find n nice point in the given interval that do not make the expression degenerate *)
insideFindDivisions[{yMin_, yMax_}, n_, {expr_, y_}] := 
Module[{n0, sel, sel2, aux},
       n0 = n - 1;
       While[sel = Select[FindDivisions[{yMin, yMax}, n0], IntervalMemberQ[Interval[{yMin, yMax}], #]&];
             Length[sel] < n || (* no bad value *)
            (sel2 = First /@ Cases[{#,  Quiet[Check[aux = expr /. y -> #;  
                                              Not[MatchQ[aux, 0 | 0. | 0. _]] &&
                                              FreeQ[aux, ComplexInfinity | Indeterminate | _DirectedInfinity |_Inderflow | _Overflow , {0, \[Infinity]}], False]]}& /@ sel, 
                                   {_, True}]; Length[sel2] < n), n0++];
       Take[sel2, n]  ]


(* ::Input:: *)
(*insideFindDivisions[{-10, 10}, 4, {Sin[x y], y}]*)


(* ::Input:: *)
(*insideFindDivisions[{-Pi/2, Pi/2}, 4, {Tan[x]/Tan[y], y}]*)


(* ::Input:: *)
(*insideFindDivisions[{-30,30}, 4, {a x^2 - x/a, a}]*)


(* ::Input:: *)
(*GetParametrized1DRange[a x^2 - x/a, {x, 0, 5}]*)


Options[GetParametrized1DRange] = {"SnapshotValues" -> 4};


GetParametrized1DRange[expr_, {x_, xMin_, xMax_}, OptionsPattern[]] := 
Module[{addVars, res, exprs1, prBag, grs, yVar, sel2, sel2MinMaxes, yMin, yMax, yMin\[CapitalDelta], yMax\[CapitalDelta],
        yMin2, yMax2, \[CapitalDelta], roundingScale, yMin3, yMax3, rs, globalRange, selInnerMinMaxes,
        yMinI, yMaxI, innerMeanPoint, pP, allGrs, m, snvs},
   addVars = Complement[Union[Cases[expr, _Symbol?(Context[#]==="Global`"&), {0, \[Infinity]}]], {x}];
   If[Length[addVars] =!= 1, (* multivariate after instanciation of x *) $Failed,  
       m = OptionValue["SnapshotValues"];
       yVar = addVars[[1]]; 
       (*'representative' values *)
       exprs1 = DeleteDuplicates[(expr /. x -> xMin  + # (xMax - xMin))& /@ {0, 1, 1/2, 1/Pi, 2/E, 0.923641}];
       prBag = {};
       CalculateTimeConstrained[Do[AppendTo[prBag, Get1DRange[exprs1[[j]]]], {j, Length[exprs1]}], 1]; 
       grs =  Cases[prBag, _List] // N;
       res = 
        If[grs === {}, {{0, 1}, "SuggestedStartingValue" -> 1/2},
     
          sel2 = Select[grs, Length[#] === 2&];
          If[Length[sel2] >= 2, 
             (* hopefully we have enough ranges *)
             sel2MinMaxes = Transpose[Rest /@ (Last /@ sel2)];
             yMin = Mean[sel2MinMaxes[[1]]];
             yMax = Mean[sel2MinMaxes[[2]]];
             yMin\[CapitalDelta] = StandardDeviation[sel2MinMaxes[[1]]];
             yMax\[CapitalDelta] = StandardDeviation[sel2MinMaxes[[2]]];
             (* extend *)
             yMin2 = yMin - 2/3 yMin\[CapitalDelta];
             yMax2 = yMax + 2/3 yMax\[CapitalDelta];
             (* potentially symmetrize *)
             If[((expr /. yVar -> yVar) - expr) === 0 && Abs[yMin2] =!= Abs[yMax2], 
                {yMin2, yMax2} = {-1, 1} Mean[Abs[{yMin2, yMax2}]]];
             \[CapitalDelta] = yMax2 - yMin2;
             roundingScale = 10^Round[Log10[\[CapitalDelta]/10]];
             {yMin3, yMax3} =  Round[{yMin2, yMax2}, rs = roundingScale]; 
             (* potentially round to smaller scale *)
             If[Abs[yMax3 - yMin3]/\[CapitalDelta] < 0.8,  Round[{yMin2, yMax2}, rs = roundingScale/2]];
             If[Abs[yMax3 - yMin3]/\[CapitalDelta] < 0.8, {yMin3, yMax3} =  Round[{yMin2, yMax2}, rs = roundingScale/5]];
             If[Abs[yMax3 - yMin3]/\[CapitalDelta] < 0.8, {yMin3, yMax3} =  Round[{yMin2, yMax2}, rs = roundingScale/10]];
             globalRange = {yMin3, yMax3};
             (* find starting point from inner ranges *)
             selInnerMinMaxes = Transpose[Rest /@ (First /@ sel2)]; 
             yMinI = Mean[selInnerMinMaxes[[1]]];
             yMaxI = Mean[selInnerMinMaxes[[2]]];
             innerMeanPoint = Mean[Flatten[selInnerMinMaxes]];  
             (* between inner and outer range *)
             pP = Round[(yMaxI + yMax)/2, rs];
             If[Quiet[PossibleZeroQ[D[expr /. yVar -> pP, x]]], pP = Round[(innerMeanPoint + yMaxI)/2, rs/10]];  
             {globalRange, "SuggestedStartingValue" -> pP} 
             , 
             (* use extrema *)
             allGrs = Rest /@ Flatten[grs, 1];
             {{Min[allGrs], Max[allGrs]}, "SuggestedStartingValue" -> Mean[allGrs]}  
             ] 
            ];
          snvs = insideFindDivisions[res[[1]], m, {expr, yVar}] ;
          Append[res, "SuggestedSnapshotValues" -> snvs] 
      ]
           
         ]


(* ::Input:: *)
(*GetParametrized1DRange[Sin[x y], {x, 0, 5}]*)


(* ::Input:: *)
(*GetParametrized1DRange[x^2 - x y^3 - 2 y + 2 -x, {x, 0, 5}]*)


(* ::Input:: *)
(*GetParametrized1DRange[x Sin[x y], {x, 0, 5}]*)


(* ::Input:: *)
(*GetParametrized1DRange[x Sin[ y], {x, 0, 5}]*)


(* ::Input:: *)
(*GetParametrized1DRange[a x^2 - x/a, {x, 0, 5}]*)


(* ::Subsection::Closed:: *)
(*Code for 'good' independent variable ranges for complex-valued functions that somewhere are purely real  (function GetReal1DRange) EXPERIMENTAL*)


(* ::Text:: *)
(*First call Get1DRange. Then call GetReal1DRange (in the form GetReal1DRange[expr, Get1DRangeReults]) *)
(*and get a new horizontal plot ranges for the parts where is explicit real-valued.*)
(*In case the function is explcitly real-valued over the ranges given as arguments, return 'Inherited'.*)
(**)
(*(We cannot just return the intervals given because the segments where the function is real-valued might not be contiguous. (think Log[Sin[x]]).*)
(*In the case of noncontiguous we still want to plot just where the function is purely real and nothing on the segments where it is complex-valued.*)
(*So, we must differentaite between two identical plot ranges somehow to avoid generating two times the same graphics.)*)
(**)
(*To plot a function where it is real,  RealImaginaryPlot with option setting "PlotOnlyRealPart" -> False.*)
(*(RealImaginaryPlot analyzes a plotted curve at recalculate time and does automatically a plot of real and imaginary part in case a function is complex-valued.)*)


(* return values: 
   True -- real over the whole interval 
   False -- no real values on a 1D set
   $Failed -- unable to compute because of timeout or missing methods
   intervals -- intervals in which expr is real-valued 
 *)
symbolicRealRanges[expr_, {x_, xL_, xU_}, maxTime_] :=
Module[{isObviouslyRealQ, red, sels, coverageLength, envelopeCoverageLength},
    (* obviously noncomplex *)
    isObviouslyRealQ = TrueQ[Assuming[xL < x < xU, Assumptions`ARealIfDefinedQ[expr]]];
    If[isObviouslyRealQ, True, 
       (* try harder *)
       red = CalculateTimeConstrained[Quiet @ Reduce[Element[expr, Reals] && xL < x < xU, 
                                                x, Reals], 
                                      maxTime];
       Which[(* full interval *)
             MatchQ[red, HoldPattern[Inequality[_?NumericQ, LessEqual | Less, x, 
                                                LessEqual | Less, _?NumericQ]]] && 
             TrueQ[{red[[1]], red[[-1]]} == {xL, xU}], True, 
             (* not real *)
             red === False, False,
             (* too complicated to analyze *)
             Head[red] === Reduce || (* parametrized regions, say from Log[Sin[1/x]] *)
             MemberQ[red, HoldPattern[Inequality[_?(Not[NumericQ[#]]&), LessEqual | Less, x, 
                                                                        LessEqual | Less, _?(Not[NumericQ[#]]&)]], 
                     {0, \[Infinity]}], $Failed,
             (* time-out *)
             red === $Aborted, $Failed,
             (* mostly unevaluated *)
             MemberQ[red, Element[_, Reals], {0, \[Infinity]}], $Failed,
             (* explicit intervals *)
             True,
             sels = {#1, #5}& @@@ Cases[red, HoldPattern[Inequality[_?NumericQ, LessEqual | Less, x, 
                                                                                LessEqual | Less, _?NumericQ]], 
                                        {0, \[Infinity]}];
             If[sels === {}, False, 
                coverageLength = Total[Abs[Subtract @@@ sels]];
                envelopeCoverageLength = Max[sels] - Min[sels];
                Which[(* just a few infinities somewhere *)
                      coverageLength > 0.999 Abs[xU - xL], True, 
                      (* not everywhere real, but use original interval *)
                      envelopeCoverageLength > 0.9 Abs[xU - xL], {xL, xU},  
                      (* some part is real and plotting will sample it *)
                      coverageLength > 0.02 Abs[xU - xL], {Min[sels], Max[sels]}, 
                     
                      (* mostly complex *)
                      True, $Failed]
               ]
            ] 
          ]
        ] 


(* ::Input:: *)
(*symbolicRealRanges[Sin[x], {x, -11/7243, 3.33}, 0.5]*)


(* ::Input:: *)
(*symbolicRealRanges[Sin[x]+x^2 Sin[1/x], {x, -Pi, 2.}, 0.5]*)


(* ::Input:: *)
(*symbolicRealRanges[Sin[x+ Cos[x + Tan[x]]], {x, -Pi, 2}, 0.5]*)


(* ::Input:: *)
(*symbolicRealRanges[Log[x], {x, -3, 3}, 0.5]*)


(* ::Input:: *)
(*symbolicRealRanges[x^x,{x, -3, 3}, 0.5]*)


(* ::Input:: *)
(*symbolicRealRanges[Log[Tan[x]],{x, -3, 3}, 0.5]*)


(* ::Input:: *)
(*symbolicRealRanges[Log[Sin[x]],{x, -12,12}, 0.5]*)


(* ::Input:: *)
(*symbolicRealRanges[PolyLog[Log[x],x],{x, -3, 3}, 0.5]*)


(* ::Input:: *)
(*symbolicRealRanges[Sqrt[-x^2],{x, -3, 3}, 0.5]*)


(* ::Input:: *)
(*symbolicRealRanges[Log[Sin[1/x]],{x, -3, 3}, 0.5]*)


findAnyRealRange[expr_, x_, maxTime_] :=
Module[{red, fs, xVals},
       red = CalculateTimeConstrained[Reduce[Element[expr, Reals], x, Reals], maxTime/2];
       If[red === False || Head[red] === Reduce || red === $Aborted, False,
          fs = CalculateTimeConstrained[FindInstance[If[red === True, expr, red], x, Reals, 3], maxTime/2];
          If[Length[fs] < 4, False, xVals = x /. fs; {Max[xVals], Min[xVals]}
               ]
            ] 
        ] 


(* in case Reduce does not succeed *)
numericRealRanges[expr_, {x_, xL_, xU_}] :=
Module[{pp = 601, L = xU - xL,
        constituents, cf, exprNValues, xTFList, realPairGroups, sels, coverageLength, envelopeCoverageLength},
       constituents = 1.Prepend[Select[Union[Level[expr, {0, \[Infinity]}]], MemberQ[#, x, {0, \[Infinity]}]&], x]; 
       cf = Compile @@ ((Hold @@ {{}, Table\[DoubleStruckCapitalH][constituents /. x -> x + I 0.,  {x, xL, xU, (xU - xL)/pp}]}) /. Table\[DoubleStruckCapitalH] -> Table);
       If[And @@ (NumericQ /@ Flatten[cf[[4]]]),
          exprNValues =  cf[],
         CheckAbort[TimeConstrained[SetSystemOptions["CompileOptions" -> {"TableCompileLength" -> Infinity}];
                                    exprNValues = Table[Evaluate[constituents], {x, xL, xU, (xU - xL)/pp}],
                                    0.5], 
                    exprNValues = $Aborted;
                    SetSystemOptions["CompileOptions" -> {"TableCompileLength" -> 250}];
                   ]
         ];
       Which[FreeQ[Chop[exprNValues, 10^4 $MachineEpsilon], _Complex, {0, \[Infinity]}], True,
             Head[exprNValues] === List,  
             xTFList = {Re[#[[1]]], Sign @ Length @ Cases[Chop[Rest[#], 10^4 $MachineEpsilon], Except[_Real]]}& /@ exprNValues;
             realPairGroups = Cases[Split[xTFList, #1[[2]] === #2[[2]]&], {{_, 0} ...}];
             sels = {Min[#], Max[#]}&[First /@ #]& /@ realPairGroups;
             coverageLength = Total[Abs[Subtract @@@ sels]];
             envelopeCoverageLength = Max[sels] - Min[sels];
             Which[sels === {}, False, 
                   (* not everywhere real, but use original interval *)
                   envelopeCoverageLength > 0.9 Abs[xU - xL], {xL, xU},  
                    (* shorten the original interval *)
                    coverageLength > 0.1 Abs[xU - xL],  {Max[xL, Min[sels] - L/2/pp], Min[xU, Max[sels] + L/2/pp]}, 
                    True, $Failed],
            True, $Failed
         ] 
        ]


(* ::Input:: *)
(*numericRealRanges[Sin[x], {x, -10, 10}]*)


(* ::Input:: *)
(*numericRealRanges[Log[x], {x, -3, 3}]*)


(* ::Input:: *)
(*numericRealRanges[Log[Sin[1/x]], {x, -3, 3}]*)


(* ::Input:: *)
(*numericRealRanges[PolyLog[Log[x],x], {x, -3, 3}]*)


(* ::Input:: *)
(*debugG1DRQ = False; *)


getRealRange[expr_, {x_, xL2_, xU2_}] :=
Module[{maxTime =  0.4, srr, nrr},
       srr = CalculateTimeConstrained[symbolicRealRanges[expr, {x, xL2, xU2}, maxTime], maxTime];
       Which[Head[srr] === List || srr === True, If[debugG1DRQ, Print[Style["In getRealRange: symbolicRealRanges succeeded", Darker[Green, 0.8]]]]; srr,
             srr === False, If[debugG1DRQ, Print[Style["In getRealRange: symbolicRealRanges returned False", Darker[Green, 0.8]]]]; $Failed,
             True,
             If[debugG1DRQ, Print[Style["In getRealRange: symbolicRealRanges returned no decision", Darker[Green, 0.8]]]];
             nrr = CalculateTimeConstrained[numericRealRanges[expr, {x, xL2, xU2}], maxTime];
             If[Head[nrr] === List || nrr === True, 
                If[debugG1DRQ, Print[Style["In getRealRange: numericRealRanges succeeded", Darker[Green, 0.8]]]]; nrr, 
                   If[debugG1DRQ, Print[Style["In getRealRange: numericRealRanges failed", Darker[Green, 0.8]]]]; $Failed]
         ]
       ]


(* ::Input:: *)
(*getRealRange[Sin[x], {x, -3, 3}]*)


(* ::Input:: *)
(*getRealRange[Log[x], {x, -3, 3}]*)


(* ::Input:: *)
(*getRealRange[Sqrt[-x^2], {x, -3, 3}]*)


(* ::Input:: *)
(*getRealRange[PolyLog[Log[x],x], {x, -3, 3}]*)


GetReal1DRange::usage = 
"GetReal1DRange[expression, resultOfGet1DRange] tries to determine good plot ranges for the real-valed regions of expression" <>
"given the plotranges proposed by Get1DRange. In case that the function is real-valued over resultOfGet1DRange, " <> 
"the result 'Inherited' is returned.";


GetReal1DRange[expr_, {innerRange:{x_, xL1_, xU1_}, outerRange:{x_, xL2_, xU2_}}] := 
Module[{timeS = 0.3, desperationFlag, grr, center, \[Lambda],
        originalOuterInterval, intLengthOriginal, envelopeInterval2, intLength2,
        envelopeInterval1, intLength1},
       desperationFlag = False;
       grr = getRealRange[expr, outerRange];
       (* one more try with a larger interval *)
       If[grr === $Failed,
          center = Mean[{xL2, xU2}];
          \[Lambda] = Abs[xU2 - xL2];
          grr = getRealRange[expr, {x, (center - 2 \[Lambda]), (center + 2 \[Lambda])}];
          ];  
       (* desperation mode *)  
       If[grr === $Failed || grr === False,
          desperationFlag = True;
          grr = findAnyRealRange[expr, x, 0.25]
          ];
       (* compare with original intervals  *)
       Which[grr === $Failed, (* give up *) $Failed,
             grr === True, (* all real on the given intervals *) Inherited,
             desperationFlag === True && grr =!= $Failed, Flatten[{x, grr}], 
             True,
             (* form smaller and larger plot range *)
             originalOuterInterval = Interval[{xL2, xU2}];
             intLengthOriginal = originalOuterInterval[[1, 2]] - originalOuterInterval[[1, 1]];
             envelopeInterval2 = Interval[grr];
             intLength2 = envelopeInterval2[[1, 2]] - envelopeInterval2[[1, 1]];
             (* not enough change to warrant a different range *)
             If[envelopeInterval2 > 0.8 intLengthOriginal, 
                envelopeInterval2 = originalOuterInterval]; 
             envelopeInterval1 = IntervalIntersection[Interval[{xL1, xU1}], envelopeInterval2];
             intLength1 = If[envelopeInterval1 === Interval[], 0, envelopeInterval1[[1, 2]] - envelopeInterval1[[1, 1]]]; 
             (* compare inner and outer intervals *)
             Which[(* inner one becomes too small *) 
                   intLength1 < 10^4 $MachineEpsilon, 
                      {{x, envelopeInterval2[[1, 1]], envelopeInterval2[[1, 2]]}},
                   (* both are useful *) 
                   intLength2 >= 2 intLength1, 
                     {{x, envelopeInterval1[[1, 1]], envelopeInterval1[[1, 2]]},
                      {x, envelopeInterval2[[1, 1]], envelopeInterval2[[1, 2]]}},
                   (* use only outer interval *) 
                   True,
                     {{x, envelopeInterval2[[1, 1]], envelopeInterval2[[1, 2]]}}
          
                   ]
             ]
        ]


GetReal1DRange[expr_, {oneRange:{x_, xL_, xU_}}] :=  GetReal1DRange[expr, {{x, 0, 0}, {x, xL, xU}}]


GetReal1DRange[expr_, $Failed] := $Failed


(* ::Input:: *)
(*GetReal1DRange[Sin[x], {{x, -3, 3}, {x, -12, 12}}]*)


(* ::Input:: *)
(*GetReal1DRange[Sin[x+ Cos[x + Tan[x]]], {{x, -3, 3}, {x, -12, 12}}]*)


(* ::Input:: *)
(*GetReal1DRange[Log[x], {{x, -3, 3}, {x, -12, 12}}]*)


(* ::Input:: *)
(*GetReal1DRange[Log[Sin[x]], {{x, -3, 3}, {x, -12, 12}}]*)


(* ::Input:: *)
(*GetReal1DRange[x^x, {{x, -3, 3}, {x, -12, 12}}]*)


(* ::Input:: *)
(*GetReal1DRange[Sin[x]^Cos[x], {{x, -3, 3}, {x, -12, 12}}]*)


(* ::Input:: *)
(*GetReal1DRange[Sin[1/x]^Cos[1/x], {{x, -3, 3}, {x, -12, 12}}]*)


(* ::Input:: *)
(*GetReal1DRange[Sin[x], {{x, -5, 5}}]*)


(* ::Input:: *)
(*GetReal1DRange[Log[x], {{x, -5, 5}}]*)


(* ::Subsection::Closed:: *)
(*Tests*)


(* ::Input:: *)
(*Get1DRangeSpecialCase[{E^(x-1) ,E^(1-x)}, x]  *)


(* ::Input:: *)
(*Get1DRangeSpecialCase[{Cos[x] Sqrt[1 + Cos[3/2 x]], Sin[x] Sqrt[1 + Cos[3/2 x]]}, x]  *)


(* ::Input:: *)
(*Get1DRangeSpecialCase[{Cos[x] Sqrt[1 + Cos[3/2 x] + Sqrt[Cos[x]]], Sin[x] Sqrt[1 + Cos[3/2 x] + Sqrt[Cos[x]]]}, x]  *)


(* ::Input:: *)
(*Get1DRange[x]*)


(* ::Input:: *)
(*Get1DRange[Log[x]]*)


(* ::Input:: *)
(*Get1DRange[Sinh[Cosh[x]]]*)


(* ::Input:: *)
(*Get1DRange[Cos[2x]]*)


(* ::Input:: *)
(*Get1DRange[Sin[x]]*)


(* ::Input:: *)
(*Get1DRange[{Cos[x], Sin[x]}]*)


(* ::Input:: *)
(*Get1DRange[{Cos[2x], Sin[x]}]*)


(* ::Input:: *)
(*Get1DRange[{Cos[3x], Sin[5x]}]*)


(* ::Input:: *)
(*Get1DRange[{Cos[3x/7], Sin[5x]}]*)


(* ::Input:: *)
(*Get1DRange[{Cos[3x], Sin[5x]}]*)


(* ::Input:: *)
(*Get1DRange[{Cos[Sqrt[2]x], Sin[Sqrt[3]x]}]*)


(* ::Input:: *)
(*Get1DRange[3 x]*)


(* ::Input:: *)
(*Get1DRange[{x, x^2}]*)


(* ::Input:: *)
(*Get1DRange[{x^3, x^2}]*)


(* ::Input:: *)
(*Get1DRange[{x Sin[x], x^2 - x^3}]*)


(* ::Input:: *)
(*Get1DRange[ x^2 - x^3]*)


(* ::Input:: *)
(*(* should fail *)*)
(*Get1DRange[ x y]*)


(* ::Input:: *)
(*(* should fail *)*)
(*Get1DRange[{x, y-x}]*)


(* ::Section::Closed:: *)
(*Code for SuggestPlotRange (exported)*)


(* ::Subsection:: *)
(*Code*)


Options[SuggestPlotRange] = {"Periods" -> Automatic, "Range" -> "Narrow"};


SuggestPlotRange[expr_, x_, opts:OptionsPattern[]] := First[Get1DRangeWithVariable[expr, x, opts], $Failed]


(* ::Section::Closed:: *)
(*Code for SuggestPlotRange1D (new)*)


(* ::Subsection:: *)
(*Code*)


Options[SuggestPlotRanges1D] = {"Periods" -> 1};


SuggestPlotRanges1D[expr_, x_, opts:OptionsPattern[]] := Rest /@ Get1DRangeWithVariable[expr, x, opts]


(* ::Subsection::Closed:: *)
(*Tests*)


(* ::Input:: *)
(*SuggestPlotRange1D[Sin[x], x]*)


(* ::Input:: *)
(*Get1DRange[Sin[x]]*)


(* ::Section::Closed:: *)
(*Code for MakeScaledPlots and its support functions*)


(* ::Subsection::Closed:: *)
(*Top-level package function MakeScaledPlots (scalar and list arguments)*)


Options[MakeScaledPlots] = {"Linear" -> False, "PlotRange" -> Automatic};


MakeScaledPlots[expr_, {x_, parameters_}, Plot, opts___] := 
Module[{exprRealizations =  expr /.(Rule @@@ Transpose[parameters, #]) & /@ 
        parameterRealizations[expr, {x, parameters}]},
       plotRanges = proposedPlotRangeAndFunctionFeatures[#, x, Plot]&/@ exprRealizations;
       {unitePlotRanges[plotRanges], Last /@ plotRanges}
       ]


MakeScaledPlots[expr_List, x_, Plot] :=
Module[{},  
       prs = proposedPlotRangeAndFunctionFeatures[#, x, Plot]& /@ expr;
       unitePlotRanges[prs]
       ]


(*
MakeScaledPlots[{expr_, x_, Plot}, axesLabelOption___] :=
Module[{},
Which[plottableFunctionQ[expr, x],  
      {expr1, scaleInfo} = rescale[expr, x];  
      prs = Flatten[refineRanges @ largerPlotRangeSanityCheck @ 
                     proposedPlotRangeAndFunctionFeatures[expr1, x]]; 
      DeleteCases[Function[{pr, tag},
        If[Equal @@ pr[[1]], $Failed, 
           wpOption = WorkingPrecision -> workingPrecisionNeeded[expr1, x, pr[[1]]]; 
       expr2 = If[Precision[expr1] < Infinity && 
                  wpOption =!= (WorkingPrecision -> MachinePrecision),
                   SetPrecision[expr, wpOption[[2]]], expr1]; 
      plotOptions = Cases[List @@ Rest[pr], HoldPattern[PlotRange -> _] | 
                                      HoldPattern[Ticks -> _], Infinity];
      hintOptions = Complement[List @@ Rest[pr], plotOptions];
      thePoles = Flatten["Poles" /. hintOptions /. "Poles" -> {}];
     {Hold[Plot][expr2, Flatten[{x, pr[[1]]}], 
            wpOption, If[thePoles =!= {}, 
                         Exclusions -> ((x == #)& /@ thePoles), Sequence @@ {}],
             axesLabelOption,
             Sequence @@ plotOptions,
             PlotRange -> {Automatic, Automatic}], 
             Sequence @@ hintOptions, Sequence @@ scaleInfo, tag}]] @@@ prs, $Failed],
    (* for number theoretic functions *) 
     listPlottableFunctionQ[expr, x],  
    lpData = Quiet[ListPlotData["ListPlotCase", expr, x]];  
    (* If[Head[lpData] === List && Length[lpData] > 20,
          {{Hold[ListPlot]["DataList" -> lpData, Filling -> Axis, PlotRange -> {All, Automatic}],
            "HorizontalPlotRangeType" -> "ShowPositiveArguments"}}, $Failed
          ];
     *) 
    makeInnerOuterListPlotsLists[lpData],
    True, $Failed]
]
*)


rewritePlottigrand[expr_, x_] := expr


(* avoid overflow in expressions such as (1-10^(-10))^((625*x)/11) *)
rewritePlottigrand[ HoldPattern[expr:Times[(_?NumericQ)^(_?NumericQ x_), (_?NumericQ)^(_?NumericQ x_)  ..]] , x_] := 
Module[{exp}, 
       exp = Total[N[PowerExpand[Log[List @@ expr]]]];
       Exp[exp] /; (And @@ (TrueQ[#[[1]] > 0]& /@ (List @@ expr)))
        ]


(* ::Input:: *)
(*rewritePlottigrand[(1-10^(-10))^((625*x)/11), x]*)


(* ::Input:: *)
(*rewritePlottigrand[Sin[x], x]*)


makeIndicesInteger[expr_] := 
 expr /. (bf:(BesselJ | BesselY | BesselI | BesselK))[index_Real, x_] :> bf[Round[index], x] /; index == Round[index]


MakeScaledPlots[{expr_, x_, Plot}, opts:OptionsPattern[]] :=
Module[{},    
 
(* should continuous plot and list plot be made? *)  
plFQ = plottableFunctionQ[expr, x, True] || 
       (* experimentally allow linear plotting *)
       If[OptionValue["Linear"] === False, 
          MatchQ[expr, x + (_Integer | _Rational | _Real)], False] ||
       If[OptionValue["Linear"] === True, 
          MatchQ[expr, (x + c_. /; NumericQ[c]) | {x + c_. /; NumericQ[c]} ], False];    
lplFQ = listPlottableFunctionQ[expr, x];   

(* continuous plot case; potentially rewrite function to be plotted *)
exprPL = expr /. (Log[a_. Exp[b_. x^n_.]] :> PowerExpand[Log[a Exp[b x^n]]]/; (a > 0 && Im[b] == 0)) /. 
                 {ThreeJSymbol\[DoubleStruckCapitalH] -> ThreeJSymbol, SixJSymbol\[DoubleStruckCapitalH] -> SixJSymbol, ClebschGordan\[DoubleStruckCapitalH] -> ClebschGordan,
                  x^0.5 -> Sqrt[x], x^-0.5 -> 1/Sqrt[x]} /.
                  HoldPattern[A:Times[(_?NumericQ)^(_?NumericQ x), (_?NumericQ)^(_?NumericQ x ) ..]] :> rewritePlottigrand[A, x];

containsUnevaluatedDerivativesQ = MemberQ[N[expr], Derivative[__][_][__], {0, \[Infinity]}];   

plRes = If[TrueQ[plFQ], 
           {expr1, scaleInfo} = rescale[exprPL, x];  
           verticalPlotRangeOptionValue = Cases[{opts}, HoldPattern["PlotRange" -> _]];
           vPO = Which[MatchQ[verticalPlotRangeOptionValue, {"PlotRange" -> {_?NumericQ, _?NumericQ}}], 
                              verticalPlotRangeOptionValue[[1]],
                       True, 
                            Sequence @@ {}]; 
           prs = Flatten[proposedPlotRangeAndFunctionFeaturesRRLPRSC[expr1 /. {CubeRoot[y_] :> y^(1/3), Surd[y_, n_] :> y^(1/n)}, x, vPO]];  
           
           DeleteCases[Function[{pr, tag},
             If[Equal @@ pr[[1]], $Failed, 
                wpOption = WorkingPrecision -> workingPrecisionNeeded[expr1, x,  pr[[1]]];  
                 expr2 = If[Precision[expr1] < Infinity && 
                            wpOption =!= (WorkingPrecision -> MachinePrecision),
                            makeIndicesInteger @ SetPrecision[exprPL, wpOption[[2]]], 
                            expr1]; 
                  svp = specialVerticalPlotRanges[expr, x, tag];
                  plotOptions = Cases[List @@ Rest[pr], 
                                      HoldPattern[PlotRange -> _] | 
                                      HoldPattern[Ticks -> _] | 
                                      HoldPattern[PlotPoints -> _] |  
                                      HoldPattern[MaxRecursion -> _], 
                                      Infinity];     
           (* add special vertical plotrange if present *)
           If[(MemberQ[plotOptions, PlotRange -> {Automatic, Automatic}] || FreeQ[plotOptions, PlotRange, \[Infinity]]) && 
              MatchQ[svp, {_, _}],
              plotOptions = Append[plotOptions, PlotRange -> {Automatic, svp}]];
           hintOptions = Complement[List @@ Rest[pr], plotOptions];
           thePoles = Flatten["Poles" /. hintOptions /. "Poles" -> {}];
           {Hold[Plot][expr2, Flatten[{x, pr[[1]]}], 
            wpOption, If[thePoles =!= {}, 
                         Exclusions -> ((x == #)& /@ thePoles), Sequence @@ {}],
             Sequence @@ Cases[{opts}, HoldPattern[AxesLabel -> _]], 
             Sequence @@ plotOptions,
             (* For unevaluated derivatives [such as Zeta'[s], Derivative[2, 0][BesselJ[1, x]],
                we can in principle make a plot.
                But the numerical differentiation needed is slow, so we use a small value
                of the MaxRecursion option setting 
             *)
             If[containsUnevaluatedDerivativesQ, MaxRecursion -> 1, Sequence @@ {}], 
             PlotRange -> {Automatic, 
                           If[Cases[{opts}, HoldPattern["PlotRange" -> _]] === {}, Automatic, 
                              Cases[{opts}, HoldPattern["PlotRange" -> _]][[1, 2]]
                             ]
                           }], 
             Sequence @@ hintOptions, Sequence @@ scaleInfo, tag}]] @@@ prs, $Failed],
          $Failed]; 
(* list plot case *)
lplRes = If[TrueQ[lplFQ], 
            lpData = Quiet[ListPlotData["ListPlotCase", expr, x]];  
            makeInnerOuterListPlotsLists[lpData, x], 
            $Failed];  

ffRes = 
Which[{plFQ, lplFQ} === {False, True }, lplRes,
      {plFQ, lplFQ} === {True,  False}, plRes,
      {plFQ, lplFQ} === {False, False}, $Failed,
      (* for inputs such is Fibonacci[n], show both plots *)
      {plFQ, lplFQ} === {True, True},
      If[plRes === lplRes === $Failed, $Failed,
         Join[If[Head[lplRes] === List, lplRes, {}],
              If[Head[plRes] === List, plRes, {}]] 
         ]
      ];

ffRes
   ] 


(* ::Input:: *)
(*MakeScaledPlots[{CubeRoot[x-30], x, Plot}] *)


(* ::Input:: *)
(*MakeScaledPlots[{CubeRoot[x] - x^(1/3), x, Plot}] *)


(* ::Input:: *)
(*MakeScaledPlots[{x-5, x, Plot}]*)


(* ::Input:: *)
(*MakeScaledPlots[{36-36*x^70/36^70, x, Plot}]*)


(* ::Input:: *)
(*MakeScaledPlots[{36-36*x^60/36^60, x, Plot}]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[x], x, Plot}]*)


(* ::Input:: *)
(*MakeScaledPlots[{x^2, x, Plot}, "PlotRange" -> {0, 9}]*)


(* ::Input:: *)
(*MakeScaledPlots[{EulerPhi[n], n, Plot}]*)


MakeScaledPlots[{expr_List, x_, Plot}, OptionsPattern[]] :=
 Module[{prLin, prsList, symmQ, uprs, uprsAI, iis},
     (* special case for set of linear equations *)   
     prLin = linearListPlotRange[expr, x];  
     If[prLin =!= $Failed,   uprs = prLin,
         prsList = Flatten[refineRanges @ largerPlotRangeSanityCheck @ 
                                          proposedPlotRangeAndFunctionFeatures[#, x]] & /@  expr;  
         (* if most curves have two plots, make them for all *)
         If[Count[Length /@ prsList, 2] > 3 Count[Length /@ prsList, 1],
            prsList = If[Length[#]===1, {#[[1]], #[[1]]}, #]& /@prsList 
            ]; 
         (* for mirror pair functions, use symmetric range *)
         symmQ = If[Length[expr] === 2 && Sort[expr] === Sort[expr /. x -> -x], True, False]; 
         uprs = uniteGPlotRanges[prsList, symmQ]; uprs0 = uprs;  
       ];    
     
      (* for rational functions, add all intersections *)
      If[Length[uprs] === 1, 
         uprsAI = allIntersectionPointRange[expr, x]; 
         If[uprsAI =!= $Failed,
            Which[(* intersections are in the proposed interval ==> use original interval *)
                  IntervalMemberQ[Interval[uprs[[1, 1, 1]]], Interval[uprsAI[[1, 1]]]], Null,
                  (* no overlap or intersection interval is larger ==> use both intervals *)
                  iis = IntervalIntersection[Interval[uprsAI[[1, 1]]], Interval[uprs[[1, 1, 1]]]];
                  iis === Interval[] || Abs[Subtract @@ uprsAI[[1, 1]]]/Abs[Subtract @@ uprs[[1, 1, 1]]] > 2,
                  AppendTo[uprs, uprsAI],
                  (* some overlap ==> extend interval *)
                  True, 
                  uprs = {PRData[{{Min[uprs[[1, 1, 1]], uprsAI[[1, 1]]], Max[uprs[[1, 1, 1]], uprsAI[[1, 1]]]},{}},
                          "HorizontalPlotRangeType" -> "AllIntersections"]}
                 ];
             ];
          ];
   
      {Hold[Plot][expr, Flatten[{x, #[[1, 1]]}], 
                  Sequence @@ Flatten[DeleteCases[Rest[#[[1]]], 
                                           ("Zeros" | "Extrema" |  "InflectionPoints" | "HorizontalPlotRangeType") -> _]],
                     If[MatchQ[OptionValue["PlotRange"], {_?NumericQ, _?NumericQ}], 
                     		PlotRange -> {Automatic, OptionValue["PlotRange"]},
                     		Sequence @@ {}
                     ]
                 ],
             #[[2]]} & /@ uprs
       
         ]


(* ::Input:: *)
(*MakeScaledPlots[{(1+E^(1+x)^(-1))^(-1), x, Plot}]*)


(* ::Input:: *)
(*MakeScaledPlots[{{x, 2x}, x, Plot}]*)


(* ::Input:: *)
(*MakeScaledPlots[{{Sin[x], Cos[x]}, x, Plot}]*)


(* ::Input:: *)
(*MakeScaledPlots[{{x, x^(1/2), x^(1/3), x^(1/4), x^(1/5)}, x, Plot}]*)


(* ::Input:: *)
(*MakeScaledPlots[{{x,11 x + 1100}, x, Plot}]*)


(* ::Input:: *)
(*MakeScaledPlots[{{(1/(1+x))^(1/x),0.6}, x, Plot}]*)


(* ::Subsubsection::Closed:: *)
(*Set of linear functions (special)*)


linearListPlotRange[l_, x_] := 
Module[{zeros, uZeros, \[CapitalDelta], \[CapitalDelta]i, \[CapitalDelta]o, mean}, 
If[And @@ ((PolynomialQ[#, x] && Exponent[#, x] <= 1)& /@ l), 
   zeros = Union[Last /@ Cases[Flatten[Solve[# == 0, x]& /@ l], HoldPattern[x -> _]]];
   Which[zeros == {}, 
        {PRData[{{-1., 1.}, {}}, "HorizontalPlotRangeType"->"ShowGlobalShape"]},
         Length[zeros] == 1, 
         zeros = If[Im[#] == 0., #, Abs[#]]& /@ zeros;
         \[CapitalDelta] = If[zeros == {0.}, 1, Abs[zeros[[1]]]/2];
         {PRData[{zeros[[1]] + {-1., 1.} \[CapitalDelta], {}}, "HorizontalPlotRangeType"->"ShowGlobalShape"]},
          True, 
         zeros = If[Im[#] == 0., #, Abs[#]]& /@ zeros;
         If[Max[zeros] - Min[zeros] == 0., uZeros = Union[N[zeros]]; zeros = If[uZeros == {0.}, {-1, 1}/2, uZeros[[1]] {1/2, 2}]];
         \[CapitalDelta]i = Max[zeros] - Min[zeros];
         \[CapitalDelta]o = Max[Abs[zeros]];
         mean = Mean[zeros];
         If[\[CapitalDelta]i < 1/3 \[CapitalDelta]o, 
            {PRData[{mean + {-1., 1.} \[CapitalDelta]i, {}}, "HorizontalPlotRangeType"->"ShowZeros"],
             PRData[{{-1.5, 1.5} \[CapitalDelta]o, {}}, "HorizontalPlotRangeType"->"ShowZeros"]},
            {PRData[{mean + {-1., 1.} Max[{\[CapitalDelta]i, \[CapitalDelta]o}]}, "HorizontalPlotRangeType"->"ShowZeros"]}
           ]
         ], 
   $Failed]
        ]


(* ::Input:: *)
(*linearListPlotRange[{x/.9,x/.8,x}, x] *)


(* ::Input:: *)
(*linearListPlotRange[{x+I}, x] *)


(* ::Input:: *)
(*linearListPlotRange[{x+1}, x] *)


(* ::Input:: *)
(*linearListPlotRange[{x, 2x}, x] *)


(* ::Input:: *)
(*linearListPlotRange[{x + 100, x^2}, x] *)


(* ::Input:: *)
(*linearListPlotRange[{x + 100, x+101}, x] *)


(* ::Input:: *)
(*linearListPlotRange[{x, x+4}, x] *)


(* ::Subsubsection::Closed:: *)
(*All intersection points for lists of rational functions*)


quickIntersectionTry[{f1_, f2_}, x_] := 
Module[{solPre, sol1},
       solPre = CalculateTimeConstrained[Quiet[Solve[f1 == f2, x]], 0.15, $Failed];
       If[Head[solPre] === List,
          sol1 = Cases[N[Chop[N[x/. solPre]]], _Real];
          If[Length[sol1] =!= {}, sol1, 
             $Failed], 
          $Failed
          ] 
      ];

quickIntersectionTry[_, x_] := $Failed


(* ::Input:: *)
(*quickIntersectionTry[{(1/(1+x))^(1/x),0.6}, x]*)


(* ::Input:: *)
(*quickIntersectionTry[{(1/(1+x))^(1/x),0.6^x-x}, x]*)


allIntersectionPointRange[l_List, x_] := 
Module[{n = Length[l], listOfRationalFunctionQ, ais, min, max, \[CapitalDelta]},
    listOfRationalFunctionQ = And @@ (Internal`RationalFunctionQ[#, x]& /@ l);
    (* for pairs of functions, try a quick call to Solve *)
    qIST = quickIntersectionTry[l, x];
    If[listOfRationalFunctionQ || Not[qIST === $Failed], 
       If[listOfRationalFunctionQ,
          ais = N @ Union[allIntersections = Last /@ Cases[Table[Reduce[N[l[[i]] == l[[j]]], x, Reals], {i, n}, {j, i - 1}], x == _, \[Infinity]]],
          (* use the quick Solve result *)
          ais = qIST
          ];
       Which[ais === {}, 
             $Failed, 
             Length[ais] === 1,
             PRData[{If[ais[[1]] == 0., {-1, 1}, ais[[1]] {1/2, 3/2}], {}},"HorizontalPlotRangeType" -> "AllIntersections"],
             True,
             {min, max} = {Min[ais], Max[ais]};
             \[CapitalDelta] = max - min;
             PRData[{{min - \[CapitalDelta]/12, max + \[CapitalDelta]/12},{}},"HorizontalPlotRangeType"->"ShowMorePeriods"] 
            ] ,
       $Failed
        ]
      ]


(* ::Input:: *)
(*allIntersectionPointRange[{1-x^2, -1+x^2}, x]*)


(* ::Input:: *)
(*allIntersectionPointRange[{1-x^2, -1+x^2, x^4/20}, x]*)


(* ::Input:: *)
(*allIntersectionPointRange[{(1/(1+x))^(1/x),0.6}, x]*)


(* ::Subsubsection::Closed:: *)
(*Derive second (inner) ListPlot*)


makeInnerOuterListPlotsLists[lpData_, var_] := 
Module[{  minValue, minArgs, xCenter, lpDataSorted, \[Lambda]Count, lpDataInnermost, meanAvs },
       Which[Head[lpData] =!= List, $Failed,
             Length[lpData] < 3, $Failed,
             Length[lpData] <= 24, 
             {{Hold[ListPlot]["DataList" -> lpData, Filling -> Axis, PlotRange -> {All, Automatic},
                              If[MemberQ[lpData, {0, _} | {0., _}| {1, _} | {1., _}], AxesOrigin -> {0, Automatic}, Sequence @@ {}]],
                              "HorizontalPlotRangeType" -> "ShowPositiveArguments",
                              "HorizontalAxisVariable" -> var}},
             True,
             If[(* if data start at 0 or 1 and are contiguous and are not to different aver averaging,
                   then take the first ones for the first plot 
                 *)
                Length[lpData] > 50 &&
               lpData[[1, 1]]== 0 || lpData[[1, 1]]== 1  &&  
               (lpData[[-1, 1]] - lpData[[1, 1]])/(Length[lpData] - 1) == 1 &&
               (meanAvs = Mean /@ Partition[Abs[Last /@ lpData], 15,1] ;
                Min[meanAvs]  == 0. || Max[meanAvs]/ Min[meanAvs] < 25),
                \[Lambda]Count = Min[12, Round[Length[lpData]/3]];
                lpDataInnermost =  Take[lpData, \[Lambda]Count],
                (* otherwise show around the smallest values *)
                minValue = Min[Abs[Last /@ lpData]];
                minArgs = Cases[{#1, Abs[#2]}& @@@ lpData, {_, minValue}]; 
                xCenter = Sort[minArgs, Abs[#1[[1]]] < Abs[#2[[1]]]&][[1, 1]];
                lpDataSorted = Sort[lpData, Abs[#1[[1]] - xCenter] < Abs[#2[[1]] - xCenter]&];
                \[Lambda]Count = Min[12, Round[Length[lpData]/3]];
                lpDataInnermost =  Take[lpDataSorted, \[Lambda]Count]
               ];
             {{Hold[ListPlot]["DataList" -> Sort @ lpDataInnermost, Filling -> Axis, PlotRange -> {All, Automatic},
                              If[MemberQ[lpData, {0, _} | {0., _}| {1, _} | {1., _}], AxesOrigin -> {0, Automatic}, Sequence @@ {}]],
                              "HorizontalPlotRangeType" -> "ShowSmallestValuesPositiveArguments",
                              "HorizontalAxisVariable" -> var},
              {Hold[ListPlot]["DataList" -> lpData, Filling -> Axis, PlotRange -> {All, Automatic},
                              If[MemberQ[lpData, {0, _} | {0., _}| {1, _} | {1., _}], AxesOrigin -> {0, Automatic}, Sequence @@ {}]],
                              "HorizontalPlotRangeType" -> "ShowPositiveArguments",
                              "HorizontalAxisVariable" -> var}
             }
            ]
        ]


(* ::Input:: *)
(*makeInnerOuterListPlotsLists[N @ Table[{n, BernoulliB[n]}, {n, 40}], n] *)


(* ::Subsubsection::Closed:: *)
(*Special vertical plot ranges*)


specialVerticalPlotRanges[f_. Sin[(a_. x_ + b_.)]/(a_. x_ + b_.)^2 /; NumericQ[f], x_, 
                          "HorizontalPlotRangeType" -> "ShowEnlargedMore"]  :=  f 0.098 {-1, 1}


specialVerticalPlotRanges[f_. Sin[(a_. x_ + b_.)]^m_. * (a_. x_ + b_.)^n_ /; NumericQ[f] && n < 0 && -n > m + 1, x_, 
                          "HorizontalPlotRangeType" -> "ShowEnlargedMore"]  :=  
                          With[{root = With[{n1 = Rationalize[n, 0]}, Reduce[ (m x Cos[x]+ n1 Sin[x])  == 0 && Pi < x < 2 Pi, x]]}, 
                               If[MatchQ[root, x == _] && NumericQ[root[[2]]], f 2 (Sin[x]^m  x^n /. x -> N[root[[2]]]) {-1, 1}]]


specialVerticalPlotRanges[f_. Exp[g_ x_^n_.] /; NumericQ[f] && NumericQ[g] && Re[g] == 0 && NumericQ[n], x_, 
                          _]  :=  f 1.25 {-1, 1}


specialVerticalPlotRanges[f_. Exp[g_ x_^n_.] /; NumericQ[f] && NumericQ[g] && Im[g] == 0 && g < 0 && NumericQ[n] && EvenQ[n] , x_, 
                          _]  :=  f {-0.1, 1.1}


(* ::Subsection::Closed:: *)
(*Debug*)


(* ::Input:: *)
(*CalculateTimeConstrained = TimeConstrained;*)


(* ::Subsection::Closed:: *)
(*Extract independent and dependent variable from the held input (function getUserVariables)*)


(* ::Text:: *)
(*   The plotter scanner accepts inputs of the form   *)
(*      *)
(*   f(x)   *)
(*      *)
(*   y = f(x)   *)
(*      *)
(*   y == f(x)   *)
(*      *)
(*   y(x) = f(x)   *)
(*      *)
(*   y(x) == f(x)   *)
(*      *)
(*   where x and y are either one-variable symbols, or of the form x(integer) or xInteger or x_Integer   *)


goodUserVariableQ[x_Symbol] :=
With[{s = ToString[x]},
     Context[x] =!= "System`" &&  (LetterQ[StringTake[s, 1]] &&  Length[StringCases[s, LetterCharacter]] <= 2) || StringMatchQ[s, "QuestionMark"~~__]
     ]


getIndependentVariable[f_] := 
Module[{theVar, vars, varsC},
             theVar = 
             If[NumericQ[f], $Failed,
                 (* variable is a user one-letter symbol *)
                 vars  = Select[Union[Cases[f, _Symbol, {0, Infinity}]], goodUserVariableQ];
            Which[Length[vars] === 1,
                     If[FreeQ[f, vars[[1]][___], {0, \[Infinity]}], vars, $Failed], 
                     (* more than one variable found *)
                       Length[vars] > 1, $Failed,
                       Length[vars] === 0, 
                       (* variable is of the form symbol[_] *)
                        varsC  = Select[Union[Cases[f, _Symbol[_?AtomQ], {0, Infinity}]],
                                        goodUserVariableQ[#[[0]]]&];
                        If[Length[varsC] === 1, varsC, $Failed]]
                         ]
               ]


(* ::Input:: *)
(*getIndependentVariable[x ]*)


(* ::Text:: *)
(*   getUserVariables returns a list with one or two elements.   *)
(*   If the list has one element , this element is the independent variable.   *)
(*   If the list has two elements, the first element is the independent variable and the second the dependent variable.   *)


(* ::Input:: *)
(*uvDebug = False;*)


(* special cases for direct plotting of linear functions *)
getUserVariables[Hold[x_Symbol + c_.], True, True] := {x}  /; NumericQ[c]

getUserVariables[Hold[y_Symbol == x_Symbol + c_.], True, True] := {x, y}  /; NumericQ[c]

getUserVariables[Hold[y_Symbol == one_ x_Symbol + c_.], True, True] := {x, y}  /; NumericQ[c] && Quiet[TrueQ[one == 1]]

getUserVariables[Hold[f_Symbol[x_Symbol] == x_Symbol + c_.], True, True] := {x, f[x]}  /; NumericQ[c]

getUserVariables[Hold[f_Symbol[x_Symbol] == one_ x_Symbol + c_.], True, True] := {x, f[x]}  /; NumericQ[c] && Quiet[TrueQ[one == 1]]

(*added the following three downvalues to deal with the splat on inputs such as "plot x^2/x" -aaronw*)

getUserVariables[Hold[a_. x_Symbol^n_./x_Symbol^m_.+b_.],True,True]:={x} /;(VectorQ[{a,b,m,n},NumericQ]&&n-m==1)

getUserVariables[Hold[y_Symbol == a_. x_Symbol^n_./x_Symbol^m_.+b_.],True,True]:={x,y} /;(VectorQ[{a,b,m,n},NumericQ]&&n-m==1)

getUserVariables[Hold[f_Symbol[x_Symbol] == a_. x_Symbol^n_./x_Symbol^m_.+b_.],True,True]:={x,f[x]} /;(VectorQ[{a,b,m,n},NumericQ]&&n-m==1)


(* experimental *)
getUserVariables[Hold[x_Symbol + (_Integer | _Rational | _Real)], True, False] := {x} 

getUserVariables[Hold[y_Symbol == x_Symbol + (_Integer | _Rational | _Real)], True, False] := {x, y} 


getUserVariables[Hold[f_], standAloneQ_, ___]:= 
Module[{(*f1, independentVar, xy, var, hasSubscriptsQ, theVars*)},
              If[uvDebug === True, Print[Style[Row[{"in: getUserVariables (nonlist): ", {Hold[f], standAloneQ}}], Darker[Blue]]]];
              (* form function *)  
              f1 = ReleaseHold[Hold[f] /. Set -> Equal];
              (* replace subscripts by indexed variables *)
              hasSubscriptsQ = MemberQ[f1, Subscript[_Symbol?goodUserVariableQ, _Integer], {0, \[Infinity]}];
              If[hasSubscriptsQ, f1 = f1 //. Subscript[s_Symbol?goodUserVariableQ, k_Integer] :> s[k]];
              theVars = 
                Which[(* disguised lists, such as x^Range[3] *)
                      Head[f1] === List, 
                      If[uvDebug === True, Print[Style["in: getUserVariables -- disguised lists", Darker[Blue]]]];
                      getUserVariables[Hold @@ {f1}, True], 
                       (* of the form y == f (x) *)
                      Head[f1] === Equal && Length[f1] === 2, 
                      If[uvDebug === True, Print[Style["in: getUserVariables -- head Equal", Darker[Blue]]]];
                         independentVar = getIndependentVariable[f1[[2]]];
                         xy = Which[independentVar === $Failed, $Failed,
                                 (* dependent variable is a symbol *)
                                     Head[f1[[1]]] === Symbol &&      
                                  Context[Evaluate[f1[[1]]]] =!= "System`",
                                  (* dependent variable is of the form y (x) *) 
                                    {independentVar[[1]], f1[[1]]},
                                     (MatchQ[f1[[1]], _Symbol[independentVar[[1]]] |
                                                                   _Symbol[_Integer]] &&     
                                  Context[Evaluate[f1[[1, 0]]]] =!= "System`")       ||
                                   (* dependent variable is of the form y (i)(x) *) 
                                    (MatchQ[f1[[1]], _Symbol[_Integer][independentVar[[1]]]] &&     
                                  Context[Evaluate[f1[[1, 0, 0]]]] =!= "System`"), 
                                     {independentVar[[1]], f1[[1]]},
                                      True, $Failed
                                   ];
                        If[xy === $Failed, $Failed,
                            If[plottableFunctionQ[f1[[2]], xy[[1]], standAloneQ] || 
                               listPlottableFunctionQ[f1[[2]], xy[[1]]], xy, $Failed]
                          ],
                        (* of the form y == f (x) == g (x) -- not plottable *)
                         Head[f1] === Equal && Length[f1]1 > 2, 
                        If[uvDebug === True, Print[Style["in: getUserVariables -- multi-Equal", Darker[Blue]]]];
                         $Failed,
                        (* of the form f (x) *)
                         True,  
                         If[uvDebug === True, Print[Style["in: getUserVariables -- f(x)", Darker[Blue]]]];
                          var = getIndependentVariable[f1];  
                          If[var === $Failed, $Failed, 
                             If[plottableFunctionQ[f1, var[[1]], standAloneQ, Hold[f]] || listPlottableFunctionQ[f1, var[[1]]], 
                                 var, $Failed]
                              ]
                    ];
          (* if needed, replace indexed variables back by subscripts *)
           If[hasSubscriptsQ, theVars /. x_[k_Integer] :> Subscript[x, k], theVars]
               ]


(* ::Input:: *)
(*getUserVariables[Hold[x Sin[x]],  True]*)


(* ::Input:: *)
(*getUserVariables[Hold[2x-1],  True]*)


(* ::Input:: *)
(*getUserVariables[Hold[y=2x-1],  True]*)


(* ::Input:: *)
(*getUserVariables[Hold[2x-1],  False]*)


(* ::Input:: *)
(*getUserVariables[Hold[x+1],  True]*)


(* ::Input:: *)
(*getUserVariables[Hold[If[x > 0, D[x^3 + 3 x^2 + 4 x Cos[x^2] + 2, x], D[x^3 + 3 x^2 + 4 x Sin[x^2] + 2, x]]],  True]*)


getLinearFunctionUserVariable[Hold[x_Symbol?(Context[#] === "Global`"&) + c_.]] := {x}


getUserVariables[Hold[l_List], standAlone_, ___]:= 
Module[{l1, l2, varL, varL1, vars, singleLinearFunctions},  
       If[uvDebug === True, Print[Style["in: getUserVariables (list)", Darker[Blue]]]];
       l1 = Map[Hold, Hold[l], {2}][[1]]; 
       If[(* {1, x} and friends *)
          MatchQ[l1, {Hold[_?NumericQ], Hold[_Symbol?(Context[#] === "Global`"&)]} |
                     {Hold[_Symbol?(Context[#] === "Global`"&)], Hold[_?NumericQ]}],
          If[uvDebug === True, Print[Style["in: getUserVariables  (list)-- {1, x}", Darker[Blue]]]];
          {Cases[l1, _Symbol?(Context[#] === "Global`"&), \[Infinity]][[1]]}, 
          (* list of linear functions *)
          (* general case*)
          singleLinearFunctions = Union @ Cases[l1, Hold[_Symbol?(Context[#] === "Global`"&)] | 
                                                    Hold[_Symbol?(Context[#] === "Global`"&) + _?NumericQ]]; 
          l2 = DeleteCases[l1, Hold[_?NumericQ]];
          Which[singleLinearFunctions === {},
                If[uvDebug === True, Print[Style["in: getUserVariables (list) -- no linear functions", Darker[Blue]]]];
                varL = getUserVariables[#, False]& /@ l2;   
                If[Length[Union[varL]] === 1, varL[[1]], $Failed],
                (* allow linear function *)
                Length[singleLinearFunctions] === 1,
                If[uvDebug === True, Print[Style["in: getUserVariables (list) -- one linear function", Darker[Blue]]]];
                l2 = DeleteCases[l1, Hold[_?NumericQ]];
                varL = getUserVariables[#, False]& /@ DeleteCases[l2, Alternatives @@ singleLinearFunctions];   
                varL1 = getLinearFunctionUserVariable /@ singleLinearFunctions; 
                If[Union[Flatten[varL]] === Flatten[varL1] || Flatten[varL] === {}, Flatten[varL1[[1]]], $Failed],
                (* only linear function *)
                Length[singleLinearFunctions] === Length[Union[singleLinearFunctions]],   
                If[uvDebug === True, Print[Style["in: getUserVariables (list) -- only linear functions", Darker[Blue]]]];
                varL = getUserVariables[#, False]& /@ DeleteCases[l2, Alternatives @@ singleLinearFunctions];  
                varL1 = getLinearFunctionUserVariable /@ singleLinearFunctions;
                vars = Union[Flatten[{varL, varL1}]];
                If[Length[vars] === 1, {vars[[1]]}, $Failed], 
                (* other *)
                True, $Failed 
           ]
         ]
       ]


(* ::Input:: *)
(*getUserVariables[Hold[GoldbachFunction[n]], True]   *)


(* ::Input:: *)
(*getUserVariables[Hold[{ y==x,y==x+3}], True]   *)


(* ::Input:: *)
(*getUserVariables[Hold[{ x,x+3}], True]   *)


(* ::Input:: *)
(*getUserVariables[Hold[{ y==x,y==x^2}], True]   *)


(* ::Input:: *)
(*getUserVariables[Hold[{y == x, y == -x}], True]   *)


(* ::Input:: *)
(*getUserVariables[Hold[{y == x^3, y == -x^2}], True]   *)


(* ::Input:: *)
(*getUserVariables[Hold[x^(Range[5]^(-1))], True]   *)


(* ::Input:: *)
(*getUserVariables[Hold[{1, x}], True]   *)


(* ::Input:: *)
(*getUserVariables[Hold[{1, x^2}], True]   *)


(* ::Input:: *)
(*getUserVariables[Hold[{E^x,1+x,3+2*x}], True]   *)


(* ::Input:: *)
(*getUserVariables[Hold[{x, x^2}], True]   *)


(* ::Input:: *)
(*(* should fail *)*)
(*getUserVariables[Hold[{x, x}], True]   *)


(* ::Input:: *)
(*(* should fail *)*)
(*getUserVariables[Hold[{x, x^2, x}], True]   *)


(* ::Input:: *)
(*   getUserVariables[Hold[{Sin[x], Tan[x]}], True]   *)


(* ::Input:: *)
(*   getUserVariables[Hold[Sin[x]], True]   *)


(* ::Input:: *)
(*   getUserVariables[Hold[EulerPhi[x]], True]   *)


(* ::Input:: *)
(*   getUserVariables[Hold[Sin[Subscript[x, 1]]], True]   *)


(* ::Input:: *)
(*   getUserVariables[Hold[Sin[1 + 1 + x]], True]   *)


(* ::Input:: *)
(*   getUserVariables[ Hold[y ==Sin[1 + 1 +x+Pi]], True]   *)


(* ::Input:: *)
(*   getUserVariables[Hold[y == Sin[1 + 1 + x]], True]   *)


(* ::Input:: *)
(*   getUserVariables[Hold[Subscript[y, 1] == Sin[1 + 1 + x]], True]   *)


(* ::Input:: *)
(*   getUserVariables[Hold[Subscript[y, 1] == Sin[1 + 1 + Subscript[x, 1]]], True]   *)


(* ::Input:: *)
(*   getUserVariables[Hold[Subscript[y, 1] [Subscript[x, 1]]== Sin[1 + 1 + Subscript[x, 1]]], True]   *)


(* ::Input:: *)
(*   getUserVariables[Hold[y = Sin[1 + 1 + x]], True]   *)


(* ::Input:: *)
(*   getUserVariables[Hold[y[x] = Sin[1 + 1 + x]], True]   *)


(* ::Input:: *)
(*   getUserVariables[Hold[z= Sin[1 + 1 + x[3]]], True]   *)


(* ::Input:: *)
(*   getUserVariables[Hold[z[3]= Sin[1 + 1 + x]], True]   *)


(* ::Input:: *)
(*   getUserVariables[Hold[z[3]= Sin[1 + 1 + x[3]]], True]   *)


(* ::Input:: *)
(*   getUserVariables[Hold[z[x[3]]= Sin[1 + 1 + x[3]]], True]   *)


(* ::Input:: *)
(*   getUserVariables[Hold[x^x^x], True]   *)


(* ::Input:: *)
(*   getUserVariables[ Hold[y[1] ==Sin[1 + 1 +x]], True]   *)


(* ::Input:: *)
(*   getUserVariables[ Hold[ya ==Sin[1 + 1 +xa]]]   *)


(* ::Input:: *)
(*   getUserVariables[ Hold[Subscript[ya, 2] ==Sin[1 + 1 +xa]], True]   *)


(* ::Input:: *)
(*   (* three-letter user-symbol *)   *)
(*   getUserVariables[Hold[Sin[1 + 1 + xxa]], True]   *)


(* ::Input:: *)
(*   (* two independent variables *)   *)
(*   getUserVariables[Hold[Sin[1 + 1 + x+u]], True]   *)


(* ::Input:: *)
(*   (* only single subscripts are allowed *)   *)
(*   getUserVariables[ Hold[Sin[1 + 1 +Subscript[x, 1,1]]], True]   *)


(* ::Input:: *)
(*   (* subscript should be an integer *)   *)
(*   getUserVariables[Hold[Sin[Subscript[x, a]]], True]   *)


(* ::Input:: *)
(*   (* index should be an integer *)   *)
(*   getUserVariables[Hold[Sin[x[a]]], True]   *)


(* ::Input:: *)
(*   (* implicit equation *)   *)
(*   getUserVariables[ Hold[x[y] ==Sin[1 + 1 +x]], True]   *)


(* ::Input:: *)
(*   (* lhs is a sum *)   *)
(*   getUserVariables[Hold[a+b= Sin[1 + 1 + x[3]]], True]   *)


(* ::Input:: *)
(*   (* rhs has two variables *)   *)
(*   getUserVariables[Hold[Sin[1 + 1 +x+y]], True]   *)


(* ::Input:: *)
(*   (* rhs is a sum *)   *)
(*   getUserVariables[ Hold[y ==Sin[1 + 1 +x+y]], True]   *)


(* ::Input:: *)
(*   (* lhs is too complex for now *)   *)
(*   getUserVariables[ Hold[y[1, x] ==Sin[1 + 1 +x]], True]   *)


(* ::Input:: *)
(*   (* lhs evaluates to True *)   *)
(*   getUserVariables[Hold[Equal[Sin[x]]], True]   *)


(* ::Input:: *)
(*   (* both variables *)   *)
(*   getUserVariables[Hold[Subscript[y, 1][x] = Sin[1 + 1 + x]], True]   *)


(* ::Input:: *)
(*   (* independent variables don't match *)   *)
(*   getUserVariables[Hold[Subscript[y, 1] [Subscript[x, 1]]== Sin[1 + 1 + Subscript[x, 3]]], True]   *)


(* ::Subsection::Closed:: *)
(*Function class patterns*)


integerDomainFunctionPatterns[n_] := 
Alternatives @@ 
With[{nPat = _?(MemberQ[#, n, {0, \[Infinity]}]&)},
    {BernoulliB[nPat], EulerE[nPat], BellB[nPat], 
     BernoulliB[nPat, _?NumericQ], EulerE[nPat, _?NumericQ], 
     BellB[nPat, _?NumericQ], NorlundB[nPat, _?NumericQ],
     StirlingS1[nPat, _Integer], StirlingS1[_Integer, nPat],
     StirlingS2[nPat, _Integer], StirlingS2[_Integer, nPat],  
     PartitionsP[nPat], PartitionsQ[nPat],
     RamanujanTau[nPat], 
     SquaresR[_Integer, nPat],  SquaresR[nPat, _Integer], 
     GCD[nPat, _Integer], GCD[_Integer, nPat],
     LCM[nPat, _Integer], LCM[_Integer, nPat],
     IntegerExponent[nPat, _Integer], IntegerExponent[_Integer, nPat],
     JacobiSymbol[nPat, _Integer], JacobiSymbol[_Integer, nPat],
     Prime[nPat], PrimePi[nPat], 
     StieltjesGamma[nPat], 
     PrimitiveRoot[nPat],
     EulerPhi[nPat], MoebiusMu[nPat], CarmichaelLambda[nPat], 
     MultiplicativeOrder[_Integer, nPat], MultiplicativeOrder[nPat, _Integer],
     DivisorSigma[_?NumericQ, nPat], DivisorSigma[nPat, _Integer],
     GoldbachFunction[nPat],  
     AiryAiZero[nPat], 
     AiryBiZero[nPat], 
     MertensFunction[nPat], 
     Composite[nPat], 
     f_ThreeJSymbol\[DoubleStruckCapitalH] /; MemberQ[f, nPat, {0, \[Infinity]}],
     f_SixJSymbol\[DoubleStruckCapitalH] /; MemberQ[f, nPat, {0, \[Infinity]}],
     f_ClebschGordan\[DoubleStruckCapitalH] /; MemberQ[f, nPat, {0, \[Infinity]}],
     KroneckerDelta[nPat],
     KroneckerDelta[nPat, _Integer], KroneckerDelta[_Integer, nPat],
     ChampernowneNumber[nPat]
     }];


(* extension to functions that are mostly occurring for integer argument *)
integerNaturalDomainFunctionPatterns[n_] := 
Alternatives @@ 
With[{nPat = _?(MemberQ[#, n, {0, \[Infinity]}]&)},
    {(* extension to functions that are mostly occurring for integer argument *)
     Fibonacci[nPat], LucasL[nPat], Factorial[nPat], Factorial2[nPat], CatalanNumber[nPat] 
     }];


(* ::Subsection::Closed:: *)
(*Check for a plottable function or listplottable function (plottableFunctionQ and listPlottableFunctionQ )*)


numberTheoryFunctionHeads = DeleteCases[Head /@ integerDomainFunctionPatterns[C], PrimePi];


plottableFunctionQ[exprIn_, x_, standAloneQ_, heldexpr_:Null] := 
With[{expr = exprIn /. {ThreeJSymbol\[DoubleStruckCapitalH] -> ThreeJSymbol, SixJSymbol\[DoubleStruckCapitalH] -> SixJSymbol, ClebschGordan\[DoubleStruckCapitalH] -> ClebschGordan,
                        if_If :> (Evaluate //@ if)}},  
 Which[MatchQ[Head[expr], List | Equal], False,
       FreeQ[expr, numberTheoryFunctionHeads, {0, \[Infinity]}, Heads -> True] && 
       (* a quick check *)
       (* bug 73129 *) If[standAloneQ, Not[MatchQ[expr, x | x + _?NumericQ]] || And[heldexpr =!= Null, !MatchQ[heldexpr, Hold[x + a_.] /; NumericQ[a]]], True] &&
       Quiet[NumericQ[expr /. x -> Pi^(Sqrt[3]/Log[2])] || 
             (* derivatives of special functions *)
             NumericQ[expr /. x -> Pi^(Sqrt[3]/Log[2`20])]], True,
       True, False]
    ]


(* ::Input:: *)
(*plottableFunctionQ[KroneckerDelta[x], x, True]*)


(* ::Input:: *)
(*plottableFunctionQ[x, x, True]*)


(* ::Input:: *)
(*plottableFunctionQ[x, x, False]*)


(* ::Input:: *)
(*plottableFunctionQ[Sin[x], x, True]*)


(* ::Input:: *)
(*plottableFunctionQ[PrimePi[x], x, True]*)


(* ::Input:: *)
(*plottableFunctionQ[AiryAiZero[x], x, True]*)


(* ::Text:: *)
(*Allow plotting of some number-theoretical functions for polynomial arguments*)


integerPolyQ[expr_, n_] := PolynomialQ[expr, n] && Element[CoefficientList[expr, n], Integers]


(* allow for summatory functions *)
resolveSumsAndProducts[expr_] := 
Flatten[{DeleteCases[expr, _Sum | _Product, {0, \[Infinity]}],
         Cases[expr, _Sum | _Product, {0, \[Infinity]}] //. 
               {HoldPattern[(Sum | Product)[f_, {j_, n2_}]] :> {f /. j -> n2},
                HoldPattern[(Sum | Product)[f_, {j_, n1_, n2_}]] :> {f /. j -> n1, f /. j -> n2}}
               }]


(* just a quick check *)
listPlottableFunctionQ[expr_, n_] := 
Module[{}, 
       intFuns = Cases[resolveSumsAndProducts[expr] , integerDomainFunctionPatterns[n] | 
                                                      integerNaturalDomainFunctionPatterns[n], {0, \[Infinity]}];
       MatchQ[n, Symbol["c"] | Symbol["d"] | Symbol["b"] |
                 Symbol["i"] | Symbol["j"] | Symbol["k"] | Symbol["m"] | 
                 Symbol["n"] | Symbol["\[Mu]"] | Symbol["\[Nu]"] | 
                (* especially requested *) Symbol["x"] | Symbol["y"] |  Symbol["F"] ] && 
       intFuns =!= {} &&
       (args = Flatten[Select[List @@@ intFuns, MemberQ[#, n, {0, \[Infinity]}]&]];
       (* (TrueQ @ Resolve @ Exists[n, Element[n, Integers], Element[args, Integers]]) *)
       And @@ ((integerPolyQ[#, n] || listPlottableFunctionQ[#, n])& /@ args) 
       )
      ] 


(* ::Input:: *)
(*listPlottableFunctionQ[KroneckerDelta[j] , j]*)


(* ::Input:: *)
(*listPlottableFunctionQ[ThreeJSymbol\[DoubleStruckCapitalH][{1,0},{10,0},{j,0}] , j]*)


(* ::Input:: *)
(*listPlottableFunctionQ[x!, x]*)


(* ::Input:: *)
(*listPlottableFunctionQ[(x!)^(x!), x]*)


(* ::Input:: *)
(*listPlottableFunctionQ[AiryAiZero[d], d]*)


(* ::Input:: *)
(*listPlottableFunctionQ[EulerPhi[d], d]*)


(* ::Input:: *)
(*listPlottableFunctionQ[EulerPhi[x], x]*)


(* ::Input:: *)
(*listPlottableFunctionQ[EulerPhi[n],n]*)


(* ::Input:: *)
(*listPlottableFunctionQ[EulerPhi[k],k]*)


(* ::Input:: *)
(*listPlottableFunctionQ[k Prime[k] Prime[k + 1], k]*)


(* ::Input:: *)
(*listPlottableFunctionQ[Sum[k Prime[k] Prime[k + 1], {k, n}], n]*)


(* ::Input:: *)
(*listPlottableFunctionQ[2 k +2EulerPhi[k], k]*)


(* ::Input:: *)
(*listPlottableFunctionQ[Sum[EulerPhi[k], {k, n}]  ,n]*)


(* ::Input:: *)
(*listPlottableFunctionQ[EulerPhi[k], k]*)


(* ::Input:: *)
(*listPlottableFunctionQ[EulerPhi[k^2], k]*)


(* ::Input:: *)
(*listPlottableFunctionQ[EulerPhi[EulerPhi[k]], k]*)


(* ::Input:: *)
(*listPlottableFunctionQ[EulerPhi[3k + EulerPhi[k]], k]*)


(* ::Input:: *)
(*listPlottableFunctionQ[EulerPhi[Pi k], k]*)


(* ::Input:: *)
(*listPlottableFunctionQ[Sin[k], k]*)


(* ::Input:: *)
(*listPlottableFunctionQ[k^2, k]*)


(* ::Input:: *)
(*listPlottableFunctionQ[k Fibonacci[k + 1], k]*)


(* ::Input:: *)
(*listPlottableFunctionQ[ThreeJSymbol[{1,0},{1,0},{j,0}], j]*)


(* ::Subsection::Closed:: *)
(*Multiple plots at once   *)


symmetrize[{a_, b_}, symmQ_] := If[symmQ, Mean[{Abs[a], Abs[b]}] {-1, 1}, {a, b}]


uniteGPlotRanges[prsList_, symmQ_] := 
 Module[{},
         If[TrueQ[Equal @@ prsList], prsList[[1]],
             Which[MatchQ[Union[Length /@ prsList], {1} | {1, 2}],
                    {Append[unitePlotRangeData[First /@ prsList, False],
                            "HorizontalPlotRangeType" -> "ShowSomething"]},
                    Union[Length /@ prsList] === {2},
                    {Append[unitePlotRangeData[First /@ prsList, symmQ], 
                            "HorizontalPlotRangeType" -> "ShowCenterPart"],
                     Append[unitePlotRangeData[Last /@ prsList, symmQ], 
                             "HorizontalPlotRangeType" -> "ShowEnlargedMore"]}
                    ]
            ]
        ]


unitePlotRangeData[prsDataList_, symmQ_] := 
 Module[{
         theRanges, theUnitedRange, theOptions, workingPrecs, theUnitedOptions
         },
         theRanges = Flatten[#[[1, 1]]] & /@ Flatten[prsDataList]; 
         theUnitedRange = symmetrize[rangeCompromise[theRanges], symmQ];
         theOptions = #[[1, 2]] & /@ Flatten[prsDataList];
         workingPrecs = (WorkingPrecision /. #) & /@ theOptions;
         theUnitedOptions = If[And @@ (NumberQ /@ workingPrecs), WorkingPrecision -> Max[workingPrecs], {}];
         PRData[{theUnitedRange, theUnitedOptions}]
        ]


rangeCompromise[ranges_] := 
 Module[{ xys, try1 },
         If[(IntervalIntersection @@ (Interval /@ ranges)) =!= Interval[],
            xys = Transpose[ranges];
            try1 = Sort[{Mean[#1], Mean[#2]} & @@ xys]; 
            If[Unequal @@ try1, try1, {Min[xys], Max[xys]}],
             {Min[#], Max[#]} &[ Transpose[ranges] ]
             ]
         ]    


(* ::Subsection::Closed:: *)
(*PlotRange after-checks   *)


largerPlotRangeSanityCheck[ip:{ip1:PRData[{range1_, rest1___}, type1_], 
                               ip2:PRData[{range2_, rest2___}, type2_]}] := 
Which[TrueQ[Min[range2] <= Min[range1]] || TrueQ[Max[range2] >= Max[range1]], ip,
      True, {ip1, PRData[enlargePlotRange[{range1, rest2}, 2], type2]}
      ];

largerPlotRangeSanityCheck[ip_] := ip    


(* ::Subsection::Closed:: *)
(*Plot pre-evaluation (quick evaluation over the proposed plot range)    *)


equalSlopeSignSegments[l_] := 
Module[{l1, l2, s1, posis, segmentPosis},
       l1 = Last /@ l;
       l2 = Table[s1 = Partition[l1[[k]], 3, 1];
                  posis = Position[s1, _?(((#[[1, 2]] - #[[2, 2]])*(#[[3, 2]] - #[[2, 2]]) > 0) ||
                                            #1[[1, 2]] #1[[3, 2]] < 0&), 
                                   {1}, Heads -> False];
                  segmentPosis =  Partition[Flatten[{1, ({+1, -1} + # &) /@ 
                                            Flatten[posis], Length[s1]}], 2];
                  Map[First, Take[s1, #]& /@  segmentPosis, {2}],
                  {k, Length[l1]}];
       Flatten[l2, 1]
      ]


Clear[refineRanges];

refineRanges[expr_] := expr

refineRanges[hps:{hp1:{Hold[Plot][body_, {x_, xMin1_, xMax1_}, options1___],
                       structures1_, hints1_},
                  hp2:{Hold[Plot][body_, {x_, xMin2_, xMax2_}, options2___],
                       structures2_, hints2_}}
            ] := 
Module[{},
       (* extract oscillations *)
       quickPlot1 = Plot[body, {x, xMin1, xMax1}, MaxRecursion -> 2, PlotPoints -> 30, options1];
       dataLines1  = Cases[quickPlot1, _Line, Infinity];
       ess = equalSlopeSignSegments[dataLines1];
       If[Length[ess] <= 2, hps,
          lengthsAndCentersAndLogMeans = 
           {Abs[#[[1, 1]] - #[[-1, 1]]], Abs[#[[1, 1]] + #[[-1, 1]]]/2,
            Log[10, Mean[Abs[Last /@ #]]], {#[[1, 1]],  #[[-1, 1]]}}& /@ ess;
          fullLengths = Total[(#[[1, -1, 1]] - #[[1, 1, 1]])& /@ dataLines1];
          mainIntervals = Select[lengthsAndCentersAndLogMeans, (#[[1]] > fullLengths/24)&];
          sizeExponents = #[[3]]& /@ mainIntervals;
         (* avoid oscillations and many scales *)
          If[Max[sizeExponents] - Min[sizeExponents] < 4, hps,
             nearScaleZeroExponent = Sort[sizeExponents, Abs[#1] < Abs[#2]&][[1]];
             nearScaleZeroSegment = Select[mainIntervals, #[[3]] == nearScaleZeroExponent&][[1]];
             segmentBag = {nearScaleZeroSegment};
             segmentPool = DeleteCases[mainIntervals, nearScaleZeroSegment];
             While[newSegments = Select[segmentPool,
                       (Abs[#[[3]] - nearScaleZeroExponent] < 1 &&
                        DeleteCases[Function[y, IntervalIntersection[Interval[#[[-1]]] ,
                                                              Interval[y[[-1]]]]] /@ segmentBag,
                                     Interval[]] =!= {})&
                                           ];
                   newSegments =!= {},
                   segmentBag = Join[segmentBag, newSegments];
                   segmentPool = DeleteCases[segmentPool, Alternatives @@ newSegments]
                   ];
             {newxMin, newxMax} = {Min[#], Max[#]}& @ (Last /@ segmentBag);
            {{Hold[Plot][body, {x, newxMin, newxMax}, options1], structures1, hints1},
                  hp2}
            ]
          ]
        (*
         quickPlot2 = Plot[body2, {x, xMin2, xMax2}, MaxRecursion -> 2, options2]; 
          *)
       ]


(* ::Subsection::Closed:: *)
(*General dispatcher for individual plots   *)


classTry[expr_, x_, pr_, class_, maxTime_] := 
Module[{$FailedTC, res},  
       If[debugMemoryQ, Print[Style[
          Row[{"beginning class: ",  class,  
              "  (current memory use: ", Ceiling[MemoryInUse[]/10^6], "MB, ", Ceiling[MaxMemoryUsed[]/10^6],"MB)"}],
                                    Darker[Green, 0.8], 10]]];  
         If[Head[pr] === List, pr, 
            print[Row[{"preDispatch", "  ", class}]];  
            If[TrueQ[TimeConstrained[makeQFunction[class][expr, x], 1, False]], 
               print[Row[{"dispatch", " ", class}]];    
               If[debugMemoryQ, Print[Style[Row[{"dispatch to: ", class}], Darker[Green, 0.8]]]];
               startTimePRF = AbsoluteTime[];
               startMemory = MemoryInUse[];
               res = CalculateTimeConstrained[PRF[class, expr, x], maxTime, $FailedTC];
               If[debugMemoryQ, Print[Style[Row[{res,  "   (", AbsoluteTime[] - startTimePRF, "\[ThinSpace]sec)", 
                                                       "  (", MemoryInUse[] - startMemory,"MB )"}], Darker[Green, 0.8]]]];
               res /. $FailedTC -> $Failed,
               $Failed]]
       ]


(* ::Input:: *)
(*classTry[2x-3, x, $Failed, "Constant", 2] *)


proposedPlotRangeAndFunctionFeatures[expr_, x_] :=
        proposedPlotRangeAndFunctionFeaturesPre[expr, x] /. fuzzFactorRationalTrig -> 1.05


proposedPlotRangeAndFunctionFeaturesRRLPRSC[expr_, x_] := refineRanges @ largerPlotRangeSanityCheck @ proposedPlotRangeAndFunctionFeatures[expr, x]


(* ::Input:: *)
(*proposedPlotRangeAndFunctionFeatures[(1+E^(1+x)^(-1))^(-1), x]*)


(* ::Input:: *)
(*refineRanges @ largerPlotRangeSanityCheck @ proposedPlotRangeAndFunctionFeatures[(1+E^(1+x)^(-1))^(-1), x]*)


lastPPRFFPBag = {};


proposedPlotRangeAndFunctionFeaturesPre[expr_, x_] :=
Module[{tMax = 1},       
    (* use chached results if possible *)  
    seenExprAndRanges = Cases[lastPPRFFPBag , {{expr, x}, _}];
    If[seenExprAndRanges =!= {}, seenExprAndRanges[[1, 2]],   
       cPR = $Failed;  
       cPR = classTry[expr, x, cPR, "SpecialCases", tMax];
       cPR = classTry[expr, x, cPR, "Constant", tMax]; 
       cPR = classTry[expr, x, cPR, "Polynomial", tMax]; 
       cPR = classTry[expr, x, cPR, "RationalFunction", tMax];
       cPR = classTry[expr, x, cPR, "RationalLinearTrigonometric", tMax];
       cPR = classTry[expr, x, cPR, "RationalLinearFresnelQ", tMax];
       cPR = classTry[expr, x, cPR, "RestrictedFunction", tMax];
       cPR = classTry[expr, x, cPR, "RationalLinearExponential", tMax];
       cPR = classTry[expr, x, cPR, "RationalLinearExponentialIntegrated", tMax];
       cPR = classTry[expr, x, cPR, "RationalTrigonometric", tMax];
       cPR = classTry[expr, x, cPR, "RationalLinearFresnel", tMax];
       cPR = classTry[expr, x, cPR, "LogOfAlgebraic", tMax];
       cPR = classTry[expr, x, cPR, "ComplexComponents", tMax];
       cPR = classTry[expr, x, cPR, "Periodic", tMax];
       cPR = classTry[expr, x, cPR, "Algebraic", tMax];
       cPR = classTry[expr, x, cPR, "TrigonometricsOfRational", tMax];
       cPR = classTry[expr, x, cPR, "TableLookUp", tMax];
       cPR = classTry[expr, x, cPR, "TrigonometricsOfAlgebraic", tMax];
       cPR = classTry[expr, x, cPR, "Elementary", tMax];       
       cPR = classTry[expr, x, cPR, "ContainsTrigonometrics", tMax];
       cPR = classTry[expr, x, cPR, "AnalyticOscillatory", tMax];
       cPR = classTry[expr, x, cPR, "AnalyticOscillatoryAtOrigin", tMax];
       cPR = classTry[expr, x, cPR, "Piecewise", tMax]; 
       cPR = classTry[expr, x, cPR, "NDSolvable", tMax];
       cPR = classTry[expr, x, cPR, "SingularPoints", tMax];  
       cPR = classTry[expr, x, cPR, "TanRescaledPlot", tMax]; 
       cPR = classTry[expr, x, cPR, "ReIm", (* uses Reduce *) 2 tMax]; 
       cPR = classTry[expr, x, cPR, "Monotonic",  2 tMax]; 
    (* to be tested *)
    (* cPR = classTry[expr, x, cPR, "TanRescaledDerivativesPlotCache", 2 tMax]; *)
       cPR = classTry[expr, x, cPR, "General", tMax];
       cPR = classTry[expr, x, cPR, "TableDesperate", tMax]; 
       cPR = classTry[expr, x, cPR, "FallThrough", tMax];
       cPR = classTry[expr, x, cPR, "SuggestOtherPlotType", tMax];

      If[cPR =!= $Failed, 
         PrependTo[lastPPRFFPBag, {{expr, x}, cPR}];  
         lastPPRFFPBag = Take[lastPPRFFPBag, Min[10, Length[lastPPRFFPBag]]]
         ]; 
       cPR
       ]
       ] // Quiet


(* ::Input:: *)
(*lastPPRFFPBag={};*)
(*proposedPlotRangeAndFunctionFeaturesPre[(1+E^(1+x)^(-1))^(-1), x]*)


makeQFunction["Constant"] := ConstantQ
makeQFunction["SpecialCases"] := SpecialCasesQ
makeQFunction["Polynomial"] := PolynomialQ
makeQFunction["RationalFunction"] := RationalFunctionQ
makeQFunction["RationalLinearTrigonometric"] := RationalLinearTrigonometricQ
makeQFunction["RationalLinearFresnel"] := RationalLinearFresnelQ
makeQFunction["RestrictedFunction"] := RestrictedFunctionQ 
makeQFunction["RationalLinearExponential"] := RationalLinearExponentialQ
makeQFunction["RationalLinearExponentialIntegrated"] := RationalLinearExponentialIntegratedQ
makeQFunction["RationalTrigonometric"] := RationalTrigonometricQ
makeQFunction["LogOfAlgebraic"] := LogOfAlgebraicQ
makeQFunction["ComplexComponents"] := ComplexComponentsQ
makeQFunction["Periodic"] := PeriodicQ
makeQFunction["Algebraic"] := AlgebraicQ
makeQFunction["TrigonometricsOfRational"] := TrigonometricsOfRationalQ
makeQFunction["TableLookUp"] := TableLookUpQ
makeQFunction["TrigonometricsOfAlgebraic"] := TrigonometricsOfAlgebraicQ
makeQFunction["Elementary"] := ElementaryQ
makeQFunction["ContainsTrigonometrics"] := ContainsTrigonometricsQ
makeQFunction["AnalyticOscillatory"] := AnalyticOscillatoryQ
makeQFunction["AnalyticOscillatoryAtOrigin"] := AnalyticOscillatoryAtOriginQ
makeQFunction["Piecewise"] := PiecewiseQ
makeQFunction["SuggestOtherPlotType"] := SuggestOtherPlotTypeQ
makeQFunction["NDSolvable"] := NDSolvableQ
makeQFunction["SingularPoints"] := SingularPointsQ
makeQFunction["General"] := GeneralQ
makeQFunction["TableDesperate"] := TableDesperateQ
makeQFunction["TanRescaledPlot"] := TanRescaledPlotQ
makeQFunction["FallThrough"] := FallThroughQ 
makeQFunction["ReIm"] := ReImQ 
makeQFunction["Monotonic"] := MonotonicQ 


(* ::Subsection::Closed:: *)
(*Preprocessing functions   *)


realify[expr_, maxTime_] := 
Module[{try},
       If[MemberQ[expr, _Complex, {0, Infinity}],
          try = TimeConstrained[Expand[ExpToTrig[expr]], maxTime, expr];
          If[try =!= expr && FreeQ[try, _Complex, {0, Infinity}],
             try, expr],
          expr]
         ]


rescale[expr_, x_] := 
Module[{},
       expr0 = realify[expr, 0.25];
       xTerms = Cases[expr0, _?(NumericQ[#] && Positive[#]&) x^n_., Infinity];
       xFactors = xTerms /. x -> 1;
       res1 = If[Length[xFactors] === 1 && Abs[xFactors[[1]]] > 10^20 &&
                  FreeQ[expr0 /. xTerms[[1]] -> C, x, {0, Infinity}],
                  {expr0/. xFactors[[1]] x^n_. :> x^n, 
                   {"IndependentVariableScaleFactor" -> 
                       (xTerms /. a_ x^n_. :> a^(1/n))}},
                  {expr0, {}}];
       expr2 = TimeConstrained[Factor[res1[[1]]], 0.1, res1[[1]]]; 
       res2 = If[MatchQ[expr2, _?(NumericQ[#] && Im[#] == 0 && Abs[Log[10, #]] > 100&) _], 
                 expr2 /.    (f_?(NumericQ[#] && Im[#] == 0 && Abs[Log[10, #]] > 100&) rest_) :> 
                  With[{sf = Round[Log[10, Abs[f]]]},  
                      {Sign[f] f/10^sf rest, Join[ res1[[2]], 
                               {"DependentVariableScaleFactor" -> Abs[10^sf]}]}],
                  res1]
      ]


rescale[a_ + b_ x_^n_, x_] := {a + b x^n, {}}


(* ::Input:: *)
(*rescale[36-36*x^70/36^70, x]*)


(* ::Input:: *)
(*rescale[36-36*x^60/36^60, x]*)


(* ::Input:: *)
(*rescale[Sin[4565667*x], x]*)


(* ::Input:: *)
(*rescale[4*1.51*^-22*((0.263/x)^12-(0.263/x)^6), x]*)


(* ::Input:: *)
(*rescale[Sin[x], x]*)


(* ::Input:: *)
(*rescale[10^100 Sin[x], x]*)


EvenFunctionQ[f_, x_] := 
TimeConstrained[Simplify[f - (f /. x -> -x), Element[x, Reals]] === 0, 0.1, False]

OddFunctionQ[f_, x_] := 
TimeConstrained[Simplify[f + (f /. x -> -x), Element[x, Reals]] === 0, 0.1, False]

EvenOddFunctionQ[f_, x_] := EvenFunctionQ[f, x] || OddFunctionQ[f, x]


unitePlotRanges[prs_] := {Min[#1], Max[#2]}& @@@ Transpose[First /@ prs]


(* ::Subsection::Closed:: *)
(*Auxiliary functions    *)


workingPrecisionNeeded[expr_, x_, pr_] := 
Module[{ingredients, vals0, vals, logMax, prePrec, prePrecSum},
       ingredients =  Union @ Level[expr, {0, Infinity}];
       vals0 = Table[Evaluate[N[ingredients, 10]], 
                    {x, Min[pr], Max[pr], (Max[pr] - Min[pr])/12}] // Quiet;
       vals = DeleteCases[vals0, _?(# == 0.&) | _DirectedInfinity | Indeterminate | Underflow[] | Overflow[], 
                          {0, Infinity}] // Quiet; 
       logMax = Min[Max[Log[10, Max[Abs[vals]]]], 1000]; 
       prePrec = If[TrueQ[logMax > 15], Min[Round[2 logMax], 200], MachinePrecision];
       (* sums with cancelling terms *)
       If[Head[expr] === Plus && Length[expr] >= 3,
          prePrecSum = workingPrecisionNeededSum[expr, x, pr]; 
          Max[prePrec, prePrecSum],
          prePrec
         ]
          
      ]


(* ::Input:: *)
(*workingPrecisionNeeded[36-36*x^70/36^70, x, {-37, 37}]*)


workingPrecisionNeededSum[expr_Plus, x_, pr_] := 
Module[{sumTerms, vals0, sT1, sT2, vRandom},
       sumTerms = List @@ expr;
       vals0 = Table[N[Evaluate[sumTerms], 20], 
                     Evaluate[SetPrecision[
                              {x, Min[pr], Max[pr], (Max[pr] - Min[pr])/12}, 20]]] // Quiet;
       sT1 = Abs[Max[If[Max[Abs[#]] == 0, 1, Total[Abs[#]]/Max[Abs[#]]]& /@ vals0]]; 
       Which[TrueQ[sT1 != 0],
             sT2 = Log[10, sT1];
             If[TrueQ[sT2 < -12], Ceiling[18 + Abs[sT2]], MachinePrecision],
             True,
             (* all terms did cancel at any x-point *)
             vRandom = N[expr /. x -> SetPrecision[Min[pr] + 1/Pi (Max[pr] - Min[pr]), Infinity], 20];
             Ceiling[2 Abs[Log[10, Abs[vRandom]]]]
             ]
      ]


padRange[{min_, max_}, \[Alpha]_] := 
With[{\[CapitalDelta] = Abs[max-min]}, {Min[{min, max}] - \[Alpha] \[CapitalDelta], Max[{min, max}] + \[Alpha] \[CapitalDelta]}]


enlargePlotRange[{{min_, max_}, _}, \[Alpha]_] := 
With[{\[CapitalDelta] = Abs[max-min]}, {{Min[{min, max}] - \[Alpha] \[CapitalDelta], Max[{min, max}] + \[Alpha] \[CapitalDelta]}, {}}]


(* ::Subsection::Closed:: *)
(*Numerics-related functions   *)


(* deal with trivial but annoying multiplicity *)
preProcess[poly_] := 
Which[Head[poly] === Times, Times @@ (If[MatchQ[#, Power[_, _]],#[[1]], #]&/@ 
                                                     (List @@ poly)),
      MatchQ[poly, Power[_, _]], poly[[1]],
      True, poly]


(*
realRoots[poly_, x_, prec___] :=  
Block[{},
nr = NRoots[preProcess[poly]== 0, x, prec];
nr1 = Chop[x /. {ToRules[If[nr === False || nr === True, Or @@ {}, nr]]}];
If[Head[nr1] === List, Select[Union @ nr1, Im[#] == 0&], {}] // N
  ]
*)


realRoots[poly_, x_, prec___] := 
Module[{(*poly1, reduceRoots, numRoots,  nr1, nr2*)},
       poly1 = preProcess[poly];
       reduceRoots = CalculateTimeConstrained[Reduce[(* make exact! *) Rationalize[poly1, 0] == 0, x, Reals], 0.15]; 
       If[reduceRoots =!= $Aborted,
          nr2 = Chop[x /. {ToRules[If[reduceRoots === False || reduceRoots === True, Or @@ {}, N[#, prec]& @ reduceRoots]]}];
          If[Head[nr2] === List, Select[Union @ nr2, Im[#] == 0&], {}] // N,
          (* try numerically *)
          numRoots = NRoots[poly1 == 0, x, prec];
          nr1 = Chop[x /. {ToRules[If[numRoots === False || numRoots === True, Or @@ {}, numRoots]]}];
          If[Head[nr1] === List, Select[Union @ nr1, Im[#] == 0&], {}] // N 
          ]
       ]


(* ::Input:: *)
(*realRoots[42+x^2 + 2.Pi,x]*)


(* ::Input:: *)
(*realRoots[-167+4*x^9927,x]*)


(* ::Input:: *)
(*realRoots[x^32,x]*)


maxRealRoot[poly_, x_] := Max[realRoots[poly, x]]
minRealRoot[poly_, x_] := Min[realRoots[poly, x]]


quickZeroTry[f_, {x_, x0_}] := 
x /. Check[CalculateTimeConstrained[FindRoot[f == 0, {x, x0}, PrecisionGoal -> 3], 0.1] /. $Aborted -> {}, {}]


nearOriginTranscendentalZerosN[f_, x_, n_:1] := 
Module[{\[CapitalLambda] = 4, pp\[CapitalLambda] = 2 (* , \[CurlyEpsilon] = 10^-6*) },
       expT = -\[CapitalLambda];
       zeroList = {};
       While[preZeros = (Flatten[{quickZeroTry[f, {x, -10^expT}], 
                                  quickZeroTry[f, {x, +10^expT}]}] /.
                                 Complex[r_, i_] :> r /; Abs[i/r] < 10^-3) // Union;
             preZeros1 = DeleteCases[preZeros, x | _Complex];
             zeroList = Union[Chop @ Flatten[{zeroList, preZeros1}], 
                              SameTest -> (If[#1 == #2 == 0, True, 
                                             Abs[#1 - #2]/Max[Abs[{#1, #2}]]]&)];
             Length[zeroList] < n && expT < \[CapitalLambda], expT = expT + 1/pp\[CapitalLambda]];
        zeroList] // Quiet    


nearOriginZerosN[f_, x_] := 
Module[{num},
       If[PolynomialQ[f, x], realRoots[f, x], 
          If[PolynomialQ[num = Numerator[Together[f]], x], realRoots[num, x], 
             nearOriginTranscendentalZerosN[f, x] 
             ]
         ]
       ]


guessedCenterN[f_, x_] := 
Module[{}, 
       zeros = nearOriginZerosN[f, x];
       If[zeros =!= {}, zeros,
          extrema = nearOriginZerosN[D[f, x], x];
          If[extrema =!= {}, extrema, $Failed]
          ] 
      ]


guessRangeN[arguments_, x_, L_: 2 Pi] := 
Module[{xL, xR},
       lefts = DeleteCases[Flatten[guessedCenterN[# - L, x]& /@ arguments], $Failed];
       rights = DeleteCases[Flatten[guessedCenterN[# + L, x]& /@ arguments], $Failed];
       xL = Which[Length[lefts] >= 3, lefts[[-3]], 
                  1 <= Length[lefts] <= 2, lefts[[1]], 
                  True, -2];
       xR = Which[Length[rights] >= 3, rights[[3]], 
                  1 <= Length[rights] <= 2, rights[[1]], 
                  True, 2];    
       If[xL === xR, {xL - 1/2, xL +1/2},   
          \[CapitalDelta]= xR - xL;
          Sort[{xL - \[CapitalDelta]/16, xR + \[CapitalDelta]/16}]
          ]
       ]


(* ::Subsection::Closed:: *)
(*Symbolics-related functions   *)


nearOriginTranscendentalZerosS[f_, x_, n_:1] := 
Module[{sol},
       sol = CalculateTimeConstrained[x /. Solve[f == 0, x] // N, 0.25, {}] /. x -> {};
       sol] // Quiet    


(* ::Subsection::Closed:: *)
(*proposedPlotRangeAndFunctionFeatures with explicitly given vertical plot range*)


proposedPlotRangeAndFunctionFeaturesRRLPRSC[f_?(Head[#] =!= List&), x_, "PlotRange" -> {y1_, y2_}] := 
Module[{},
       (* try finding inverse image *)
       sol1 = x /. Solve[f == y1, x];
       sol2 = x /. Solve[f == y2, x];
       xValues = Select[Chop @ N @ Flatten[{sol1, sol2}], (NumericQ[#] && Im[#] == 0)&];  
       prx = If[FreeQ[{sol1, sol2}, _Solve, {0, \[Infinity]}],
                {xMiny, xMaxy} = {Min[xValues], 
                 (* must be plottable *) Min[Max[xValues], $MaxMachineNumber/10^10]};
                If[xMiny != xMaxy,  {xMiny, xMaxy}, $Failed],
                $Failed];
       If[prx =!= $Failed && FreeQ[prx, DirectedInfinity, {0, \[Infinity]}], 
          {PRData[{prx, {}}, "HorizontalPlotRangeType" -> "DerivedFromVerticalPlotRange"]},
          (* fall back *)
          proposedPlotRangeAndFunctionFeatures[f, x] 
         ]
      ]


(* ::Input:: *)
(*proposedPlotRangeAndFunctionFeatures[x^2, x, PlotRange -> {0, 9}]*)


(* ::Input:: *)
(*proposedPlotRangeAndFunctionFeatures[x^2+x^4, x]*)


(* ::Input:: *)
(*proposedPlotRangeAndFunctionFeatures[Sin[Tan[x]], x, "PlotRange"->{1.56`,1.58`}]*)


(* ::Input:: *)
(*proposedPlotRangeAndFunctionFeatures[Sin[Tan[x]], x, "PlotRange"->{0.56`,0.58`}]*)


(* ::Subsection::Closed:: *)
(*SPECIALICED CODE FOR THE COMMON SPECIAL CLASSES OF FUNCTIONS   *)


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Very special cases   *)


SpecialCasesQ[expr_, x_] := Head[PRF["SpecialCases", expr, x]] =!= PRF


PRF["SpecialCases", x^x^x, x_] :=
{PRData[{{-1, 2.2}, {}}, "HorizontalPlotRangeType" -> "ShowSomething"]}


PRF["SpecialCases", x^x^x^x, x_] :=
{PRData[{{-1, 1.6}, {}}, "HorizontalPlotRangeType" -> "ShowSomething"]}


PRF["SpecialCases", x^(1/x), x_] :=
{PRData[{{-1, 6}, {}}, "HorizontalPlotRangeType" -> "ShowSomething"]}


PRF["SpecialCases", MinkowskiQuestionMarkFunction[a_. x_ + b_.], x_] :=
{PRData[{{-b/a, (1 - b)/a}, {WorkingPrecision -> 12}}, "HorizontalPlotRangeType" -> "ShowNaturalDomain"]}


PRF["SpecialCases", CantorFunction[a_. x_ + b_.], x_] :=
{PRData[{{-b/a, (1 - b)/a}, {}}, "HorizontalPlotRangeType" -> "ShowNaturalDomain"]}


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Constant   *)


Clear[ConstantQ];
ConstantQ[expr_, x_] := 
     (
      TimeConstrained[Simplify[D[expr, x] /. {Ceiling' -> (0&), Floor' -> (0&), Round' -> (0&)}] === 0 &&  
                               FreeQ[Simplify[expr, Element[x, Reals]], x, {0, \[Infinity]}],
                               0.1, False] ||
     Rationalize[expr] == 0
       )


(* ::Input:: *)
(*ConstantQ[-2x-3, x]*)


(* ::Input:: *)
(*ConstantQ[7^(t/10)/3^(t/5)-7^(0.1*t)/9^(0.1*t), t]*)


(* ::Input:: *)
(*ConstantQ[ 1 - Exp[-InverseErf[z^2]], z]*)


(* ::Input:: *)
(*ConstantQ[t+4 (-1+Ceiling[t/4]), t]*)


(* ::Input:: *)
(*ConstantQ[Log[x+I]+Log[x-I]-Log[x^2+1], x]*)


(* ::Input:: *)
(*ConstantQ[Sin[Pi*Ceiling[x]], x]*)


PRF["Constant", expr_, x_] := 
Module[{val},   
       val = Quiet[N[expr /. x -> 1/Pi, 20]]; 
      {PRData[{{-1, 1},
              WorkingPrecision -> 30,
              Which[TrueQ[Im[val] == 0. && val != 0.], PlotRange -> {Full, {-val/4, 5/4 val}}, 
                    TrueQ[Im[val] == 0. && Re[val] == 0.], PlotRange -> {Full, {-0.1, 1.1}}, 
                   True, Sequence @@ {}]}, 
               "HorizontalPlotRangeType" -> "ShowUnitInterval"
              ]}
           ]


(* ::Input:: *)
(*PRF["Constant", Log[x+I]+Log[x-I]-Log[x^2+1], x]*)


(* ::Input:: *)
(*PRF["Constant", 1 - Exp[-InverseErf[z^2]], z]*)


(* ::Input:: *)
(*PRF["Constant", 2x-3, x]*)


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Polynomials   *)


PRF["Polynomial", expr0_, x_] :=
Module[{verticalRangeFactor = 2/3, horizontalRangeFactor = 0.05, symmFlagQ = False, 
        expr, re, im, res, int, 
        flagP, polyF, deg, structurePlotRange, globalPlotRange, x0, cl,
        polyScale, polyValue0, lc, lrValues, zeros, min, max, aV, bV,
        zeroPoints, extremaPoints, extremas, inflections, inflectionPoints,
        minXValue, maxXValue, xStructureSize,
        meanPolyCoeffsSize, \[CapitalDelta]X, minPolyValue, maxPolyValue, \[CapitalDelta]Y, goalMin, goalMax,
        minPre, maxPre, serP, serN , auxZero, monomials, coeffScale, sols, sel, y,
        xSymm, \[CapitalDelta]xSymm,  minSel, maxSel 
         },
       expr = TimeConstrained[Factor[expr0], 0.1, expr0]; 
       flagP = True;
       polyF[y_] = (expr /. x -> y);
       deg = Exponent[expr, x]; 
       Which[(* constant *)
             deg === 0, {{-1, 1}}, 
             deg === 1, structurePlotRange = With[{m=Coefficient[expr0,x,1], b=Coefficient[expr0,x,0]},  
             									If[PossibleZeroQ@b, {{-1,1}}, {{-1.5*Abs[b/m],1.5*Abs[b/m]}}]
             								],
             (* x^n with very large n *)
             MatchQ[expr0, x^_Integer?(# > 1000&)],         
             structurePlotRange = {{-0.5, 0.5}, {}};
             globalPlotRange = {{-1., 1.}, {}};
             flagP = False, 
             (* x^n + a *)
             MatchQ[expr0, x^_Integer?(EvenQ[#] && # > 20&) + a_], 
             flagP = False;
             structurePlotRange = {Abs[2 expr0 /. x -> 0]^(1/deg) {-1., 1.}, {}};
             globalPlotRange = {Abs[100 expr0 /. x -> 0]^(1/deg) {-1., 1.}, {}},  
             (* (x + c)^n + a *)
             MatchQ[expr, (a_ + b_. x)^n_?(# > 2&) + c_. /; FreeQ[{a, b}, x, \[Infinity]]], 
              flagP = False;
              {aV, bV} = Re @ CoefficientList[Cases[expr, _ + _. x, \[Infinity]][[1]], x]; 
              If[TrueQ[bV != 0],
	              structurePlotRange = {{-aV/bV - 1/bV, -aV/bV + 1/bV}, {}}; 
	              If[Length[Union[Sign[structurePlotRange[[1]]]]] === 2,
	                 globalPlotRange = {},
	                 globalPlotRange = {Sort[N[{-2 aV/bV, 2 aV/bV}]], {}}
	              ],
	              re = PRF["Polynomial", ComplexExpand[Re[expr]], x];
	              im = PRF["Polynomial", ComplexExpand[Im[expr]], x];
	              res = Join @@ Join[re[[1, 1, 3;;4]], im[[1, 1, 3;;4]]][[All, 2]];
			 	  int = Mean[res[[All, 2]]] + {-1, 1} StandardDeviation[res[[All, 2]]];
				  int = Select[res[[All, 2]], # > First[int] && # < Last[int] &];
				  int = Mean[int] + {-1, 1} StandardDeviation[int];
				  res = Select[res, #[[2]] > First[int] && #[[2]] < Last[int] &];
				  structurePlotRange = {Min[res[[All, 1]]], Max[res[[All, 1]]]};
				  structurePlotRange = {structurePlotRange + {-1, 1}Abs[structurePlotRange]/20, {}};
	              globalPlotRange = {Last[SortBy[{re[[1, 1, 1]], im[[1, 1, 1]]}, Last]], {}};
              ],
            (* quadratic *)
             deg === 2, 
             (* show extrema, zeros, or scale of coefficients *)
             x0 = N[x /. Solve[D[expr, x] == 0, x][[1]]];
             polyScale = Mean[DeleteCases[Abs[CoefficientList[expr, x]], 0]];
             polyValue0 = expr /. x -> x0;
             lc = Sign[Coefficient[expr, x, 2]];
             lrValues = N[x /. Solve[expr == polyValue0 + lc polyScale, x]];
             zeros = realRoots[expr, x];
             {min, max} = {Min[#], Max[#]}&[{lrValues, zeros}];
             xSymm = symmetryPoint[expr, x];
             If[xSymm =!= $Failed, 
                \[CapitalDelta]xSymm = Mean[{Abs[xSymm - min], Abs[xSymm - max]}];
                symmFlagQ = True;
                {min, max} = xSymm + \[CapitalDelta]xSymm {-1, 1}
               ];
             zeroPoints = ({#, 0}& /@ zeros);
             extremaPoints = ({#, polyF[#]}& /@ {x0});
             inflectionPoints = {}; 
              structurePlotRange =  {padRange[{min, max}, 0.1], 
               "Zeros" -> zeroPoints,
               "Extrema" -> extremaPoints,
               "InflectionPoints" -> inflectionPoints},
            (* low degree *)
             deg < 30,   
             (* find all structures *)
             If[MemberQ[expr, _Complex, {0, \[Infinity]}],
               cl = CoefficientList[expr, x];
               expr = Total[MapIndexed[#1 x^(#2[[1]])&, Abs[cl] ]]];
             zeros = realRoots[expr, x];  
             extremas = realRoots[D[expr, x], x]; 
             inflections = realRoots[D[expr, x, x], x]; 
             {min, max} = {Min[#], Max[#]}&[{zeros, extremas, inflectionPoints}];
             zeroPoints = ({#, 0}& /@ zeros);
             extremaPoints = ({#, polyF[#]}& /@ extremas);
             inflectionPoints = ({#, polyF[#]}& /@ inflections);    
             (* analyze structures *)
             {minXValue, maxXValue} = {Min[#], Max[#]}&[ {zeros, extremas, inflections}];
             If[{minXValue, maxXValue} === {\[Infinity], -\[Infinity]}, {minXValue, maxXValue} = {-1, 1}];
             xStructureSize = maxXValue - minXValue;   
             meanPolyCoeffsSize = Mean[DeleteCases[Abs[CoefficientList[expr, x]],0]];
             \[CapitalDelta]X = If[xStructureSize/meanPolyCoeffsSize < 10^-6, 1, 3/2 xStructureSize];
             {minPolyValue, maxPolyValue} = {Min[#], Max[#]}&[
                                            Join[Last /@ extremaPoints, Last /@ inflectionPoints]];
             \[CapitalDelta]Y = 2 (maxPolyValue - minPolyValue);
             If[\[CapitalDelta]Y == 0., \[CapitalDelta]Y = 3 meanPolyCoeffsSize];  
             (* find plot range *)   
             {goalMin, goalMax} = {minPolyValue - verticalRangeFactor \[CapitalDelta]Y,
                                   maxPolyValue + verticalRangeFactor \[CapitalDelta]Y};
             {minPre, maxPre} = {minRealRoot[expr - goalMin, x], maxRealRoot[expr - goalMax, x]};   
             {min, max} = {Min[minPre, minXValue - horizontalRangeFactor \[CapitalDelta]X/2],
                           Max[maxPre, maxXValue + horizontalRangeFactor \[CapitalDelta]X/2]};
             (* adjust for symmetric polynomials *)   
             xSymm = symmetryPoint[expr, x];
             If[xSymm =!= $Failed, 
                \[CapitalDelta]xSymm = Max[{Abs[xSymm - min], Abs[xSymm - max]}];
                symmFlagQ = True; 
                {min, max} = xSymm + \[CapitalDelta]xSymm {-1, 1}
               ];
             (*
             If[Chop[Expand[expr - (expr /. x -> -x)]] == 0, 
                {min, max} = {Min[{min, max, -min, -max}], Max[min, max, -min, -max]}];
              *) 
             structurePlotRange =  {padRange[{min, max}, 0.01], 
               "Zeros" -> zeroPoints,
               "Extrema" -> extremaPoints,
               "InflectionPoints" -> inflectionPoints}, 
            (* high degree *)
             deg >= 30,
             flagP = False;
             serP = Normal @ Series[expr, {x,  Infinity, 3}];
             serN = Normal @ Series[expr, {x, -Infinity, 3}];
             {min, max} = {Min[#], Max[#]}& @ 
              Flatten[{Table[realRoots[D[serP, {x, j}], x], {j, 0, 2}],
                       Table[realRoots[D[serN, {x, j}], x], {j, 0, 2}]}];
             If[min == max, {min, max} = {min - 1, max + 1}]; 
             Which[(Not[NumericQ[min]] &&  NumericQ[max]) || (Not[NumericQ[max]] &&  NumericQ[min]),
                   {min, max} = Cases[{min, max}, _?NumericQ][[1]] + {-1., 1.},
                   Not[NumericQ[min]] && Not[NumericQ[max]],
                   {min, max} = {-1., 1.}
                   ];
             structurePlotRange = {padRange[{min, max}, 0.01], {}}; 
             globalPlotRange = enlargePlotRange[structurePlotRange, 1],
              True, print["Polynomial exception"]
        ];  
         (* global view *)
         If[flagP, 
            globalPlotRange = 
            If[Exponent[polyF[y], y] <= 1, 
                If[Exponent[polyF[y], y] === 1 && 
                   Length[Union[Sign[structurePlotRange[[1]]]]] < 2,
                   auxZero = ("Zeros" /. Rest[structurePlotRange])[[1, 1]];
                   If[NumericQ[auxZero], 
                     {Sort[{-2 auxZero, 2 auxZero}], {}},
                    {}], 
                 {}], 
             CalculateTimeConstrained[
             monomials = MapIndexed[#1 y^(#2[[1]] - 1)&, CoefficientList[polyF[y], y]];
             coeffScale = Max[Abs[CoefficientList[polyF[y], y]]]; 
            (* sols = Flatten[NSolve[Last[monomials]/# == 6 || Last[monomials]/# == -6, y]& /@ 
                                                                      DeleteCases[Most[monomials], 0]] // Union;  
             sel = Select[Last /@ sols, Abs[Im[#]] < 10^-6 coeffScale&] // Re; 
            *)  
             sols = Flatten[{apSolve[Last[monomials]/# == 6, y], apSolve[Last[monomials]/# == -6, y]}& /@ 
                                                                         DeleteCases[Most[monomials], 0]] // Union;
            
             sel = Cases[N @ sols, _Real];
         
             If[Length[sel] <= 1, {}, 
                minSel = Min[sel]; 
                maxSel = Max[sel]; 
                If[symmFlagQ, 
                   \[CapitalDelta]xSymm = Mean[{Abs[xSymm - minSel], Abs[xSymm - maxSel]}]; 
                  {minSel, maxSel} = xSymm + \[CapitalDelta]xSymm {-1, 1}
                  ];
                {{minSel, maxSel}, {}}], 
                                       0.4 (* seconds *), {}]]];   
         (* return plot ranges *) 
         {PRData[structurePlotRange, "HorizontalPlotRangeType" -> "ShowZerosExtremasInflections"],
          If[globalPlotRange =!= {},
             PRData[globalPlotRange, "HorizontalPlotRangeType" -> "ShowGlobalShape"],
             Sequence @@ {}]}
             ] 


symmetryPoint[poly_, x_] := 
Module[{clS, solS, clA, solA},
      (* try symmetric *)
       clS = CoefficientList[(poly /. x -> C + x) - (poly /. x -> C - x), x];
       solS = Solve[clS == 0, C];
       If[solS =!= {} && solS =!= {{}} &&  Im[C /. solS[[1]]] === 0,  
          C /. solS[[1]] ,
         (* try antisymmetric *)
          clA = CoefficientList[(poly /. x -> C + x) + (poly /. x -> C - x), x];
          solA = Solve[clA == 0, C];
          If[solA =!= {} && solA =!= {{}} &&  Im[C /. solA[[1]]] === 0,  
             C /. solA[[1]], 
            (* not symmetric and not antisymmetric *) $Failed]] 
       ]


(* ::Input:: *)
(*{symmetryPoint[x^2 (x + 1)^2, x], symmetryPoint[ (x + 1)^2 + (x+1)^4, x],*)
(*symmetryPoint[(x-1)^3 + x^5, x], symmetryPoint[x^2+2x+1 , x]}*)


(* for real roots of simple real-valued polynomials *)
apSolve[eq_, x_] := 
Module[{red},
       red = Reduce[N[eq], x];
       x /. {ToRules[red]} 
       ]


(* special case for direct calls *)
PRF["Polynomial", x_ + c_., x_] := 
{PRData[{If[c == 0., {-1, 1}, {-2, 2} Abs[c]], 
         "Zeros"-> If[Im[c] == 0, {{-c, 0}}, {}], 
         "Extrema"->{}, "InflectionPoints" -> {}}, "HorizontalPlotRangeType" -> "ShowZerosExtremasInflections"]
   } /; NumericQ[c]


(* ::Input:: *)
(*PRF["Polynomial",x+I, x] *)


PRF["Polynomial", 0. x_ + c_., x_] := 
{PRData[{{-1, 1}, 
         "Zeros"-> {}, "Extrema"->{}, "InflectionPoints" -> {}}, "HorizontalPlotRangeType" -> "ShowZerosExtremasInflections"]
 } /; NumericQ[c]


PRF["Polynomial",  f_ x_^n_, x_] := 
{PRData[{{-1, 1} (10.^-100/Abs[f])^(1/n), 
        "Zeros"-> {}, "Extrema" -> If[EvenQ[n], {{0, 0}}, {}], "InflectionPoints" -> If[EvenQ[n], {}, {{0, 0}}]}, 
         "HorizontalPlotRangeType" -> "ShowZerosExtremasInflections"],
 PRData[{{-1, 1} (10.^100/Abs[f])^(1/n), {}}, 
         "HorizontalPlotRangeType" -> "ShowGlobalShape"]
 } /; n > 1000 && IntegerQ[n]


(* ::Input:: *)
(* PRF["Polynomial",x^99999 , x]*)


(* ::Input:: *)
(* PRF["Polynomial",x^2+2x+1 , x]*)


(* ::Input:: *)
(* PRF["Polynomial",456973959814329828038934528*z^9002, z]*)


(* ::Input:: *)
(* PRF["Polynomial",36-36*x^60/36^60, x]*)


(* ::Input:: *)
(* PRF["Polynomial",36-36*x^70/36^70, x]*)


(* ::Input:: *)
(* PRF["Polynomial", 2 z^5 - 2 I z^3 + 4z - 12 I, z]*)


(* ::Input:: *)
(* PRF["Polynomial", (z + I)^3, z]*)


(* ::Input:: *)
(*(* can give messages *) *)
(*PRF["Polynomial", x - 1, x]*)


(* ::Input:: *)
(*(* should give two plot ranges *) *)
(*PRF["Polynomial", LegendreP[20,x-1], x]*)


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Rational Functions   *)


Clear[RationalFunctionQ1];
RationalFunctionQ1[expr_Plus, x_] := And @@ (RationalFunctionQ1[#, x]& /@ (List @@ expr))

RationalFunctionQ1[expr_?PolynomialQ, x_] := True /; PolynomialQ[expr, x]

RationalFunctionQ1[a_/b_, x_] := RationalFunctionQ1[a, x] && RationalFunctionQ1[b, x]

RationalFunctionQ1[a_ b_, x_] := RationalFunctionQ1[a, x] && RationalFunctionQ1[b, x]

RationalFunctionQ1[Power[a_, b_Integer], x_] := RationalFunctionQ1[a, x]

RationalFunctionQ1[Power[a_, b_Rational], x_] := FreeQ[a, x, {0,Infinity}]

RationalFunctionQ1[_?NumericQ, x_] := True


Clear[RationalFunctionQ2];
RationalFunctionQ2[expr_, x_] :=
With[{tog = Together[expr]},
     PolynomialQ[Numerator[tog], x] &&  PolynomialQ[Denominator[tog], x]]


Clear[RationalFunctionQ];
RationalFunctionQ[expr_, x_] := 
TimeConstrained[RationalFunctionQ2[expr, x], 0.5, RationalFunctionQ1[expr, x]]


(* ::Input:: *)
(*RationalFunctionQ[(1+1/(200*x))^200*x, x]*)


PRF["RationalFunction", exprIn_, x_] :=
Module[{verticalRangeFactor = 1/3, horizontalRangeFactor = 0.02, tMax = 0.1,
        expr,
        y, cc, ratF, tog, num, den, numDeg, denDeg, zeros, poles, d1Num, d2Num,
        extremas, inflections, zeroPoints, extremaPoints, inflectionPoints,
        minXValue, maxXValue, minPolyValue, maxPolyValue, \[CapitalDelta], mima, goalMin, goalMax,
        minPre, maxPre, goalMinR, goalMaxR, minPreR, maxPreR, min, max,
        minyValueTaken, maxyValueTaken, yMin, yMax, verticalPlotRange,
        leftAsymptotic, rightAsymptotic, spr, cspr, globalPlotRange,
        removableSingularities, removableSingularityValues, \[CapitalDelta]X
        },

       expr = exprIn;
       (* rational functions with conjugate poles; after aparting *)
       TimeConstrained[If[MatchQ[ComplexExpand[Im[expr]], 0. _], expr = ComplexExpand[Re[expr]]], 0.1];
          
       ratF[y_] = (expr /. x -> y);
       tog = TimeConstrained[Factor @ Together[expr], tMax];
       structurePlotRange = 
       Which[tog === $Aborted, {padRange[{-1, 1}, 0.01], {}},
             (* common high-degree cases *)
             MatchQ[tog, f_. x^e1_. (a_ + b_. x^e2_.)^e3_. /; FreeQ[{f, a, b, e1, e2, e3}, x]],   
                    With[{s\[Sigma] = (tog /. f_. x^e1_. (a_ + b_. x^e2_.)^e3_. :> Abs[(-(a/b))^(1/e2)])}, 
                         {padRange[s\[Sigma] {-1, 1}, s\[Sigma] 0.001], 
                          With[{tPoles = x /. {ToRules[Reduce[Denominator[tog] == 0, x, Reals]]} /. x -> {}}, 
                               If[tPoles =!= {}, {"Poles" -> N[tPoles]}, {}]]}],
             True,
             (* should be able to differentiate and together *)
             {num, den} = {Numerator[#], Denominator[#]}&[tog];
             {numDeg, denDeg} = Exponent[{num, den}, x];
             zeros = realRoots[num, x];
             poles = realRoots[den, x];
             d1Num = Numerator[Together[D[expr, x]]];
             d2Num = Numerator[Together[D[expr, x, x]]];
             extremas = realRoots[d1Num, x];
             inflections = realRoots[d2Num, x];
             zeroPoints = ({#, 0}& /@ zeros);
             extremaPoints = ({#, ratF[#]}& /@ extremas);
             inflectionPoints = ({#, ratF[#]}& /@ inflections); 
             removableSingularities = CalculateTimeConstrained[findRemovableSingularities[expr, x], 0.15];
             (* analyze structures *)
             {minXValue, maxXValue} = {Min[#], Max[#]}&[{zeros, poles, extremas, inflections, removableSingularities}];
              If[{minXValue, maxXValue} === {\[Infinity], -\[Infinity]}, 
                 {minXValue, maxXValue} = {-1, 1}];
              \[CapitalDelta]X = If[# == 0., 50, #]&[maxXValue - minXValue];
              removableSingularityValues = Limit[expr, x -> #]& /@ removableSingularities;
              {minPolyValue, maxPolyValue} = {Min[#], Max[#]}&[Re @ Join[Last /@ extremaPoints, Last /@ inflectionPoints, removableSingularityValues] ];
              If[minPolyValue === +\[Infinity], minPolyValue = -10];
              If[maxPolyValue === -\[Infinity], maxPolyValue = 10];
              \[CapitalDelta] = maxPolyValue - minPolyValue;
              If[\[CapitalDelta] == 0., \[CapitalDelta] = Abs[maxPolyValue]]; 
           (* find good vertical plot range *) 
              mima = CalculateTimeConstrained[
              {goalMin, goalMax} = Re @ {minPolyValue - verticalRangeFactor \[CapitalDelta],
                                         maxPolyValue + verticalRangeFactor \[CapitalDelta]}; 
              {minPre, maxPre} = Re @ {minRealRoot[Numerator[Together[expr - goalMin]], x], 
                                       maxRealRoot[Numerator[Together[expr - goalMax]], x]};   
              {goalMinR, goalMaxR} = Re @ {minPolyValue + verticalRangeFactor \[CapitalDelta],
                                           maxPolyValue - verticalRangeFactor \[CapitalDelta]};
              {minPreR, maxPreR} = Re @ {minRealRoot[Numerator[Together[expr - goalMinR]], x], 
                                         maxRealRoot[Numerator[Together[expr - goalMaxR]], x]};   
              {min, max} = Re @ {Min[minPre, minPreR, minXValue - horizontalRangeFactor \[CapitalDelta]X],
                                 Max[maxPre, maxPreR, maxXValue + horizontalRangeFactor \[CapitalDelta]X]},   
                0.25];   
               {min, max} = If[mima =!= $Aborted, mima,
                               (* high-degree polynomials *)
                               If[TrueQ[minXValue != maxXValue],
                                 {minXValue, maxXValue},
                                 {-1, 1} 
                              ] 
                            ];  
               minyValueTaken = TimeConstrained[Re @ Minimize[{expr, min <= x <=  max}, x][[1]], 0.15, $Failed]; 
               maxyValueTaken = TimeConstrained[Re @ Maximize[{expr, min <= x <=  max}, x][[1]], 0.15, $Failed]; 

               yMin = Min[Select[Re @ {goalMin, goalMinR}, NumericQ]];
               If[yMin === -Infinity, -1];
               yMax = Max[Select[Re @ {goalMax, goalMaxR}, NumericQ]]; 
               If[yMax === Infinity, 1];
   
            verticalPlotRange  =
               Which[(* numerator/denominator cancel and we have only one point -- False condition added for 269821 *) 
                     (minPolyValue == maxPolyValue) =!= False, {Automatic, Automatic}, 
                     TrueQ[NumberQ[yMin] && NumberQ[yMax] && Abs[yMax - yMin]/Abs[maxyValueTaken - minyValueTaken] < 10^-2],
                      {Automatic, Automatic},
                     TrueQ[NumberQ[yMin] && NumberQ[yMax] && Abs[yMax - yMin]/Abs[maxyValueTaken - minyValueTaken] < 10^-2],
                      {yMin, yMax},
                     True,  
                      {yMin, yMax}
                    ];    

            (* in case a symmetry center exists, potentially symmetrize *)   
            symmetryCenters = 
              TimeConstrained[
                  If[(* special case *) Simplify[ratF[y]^2 - ratF[-y]^2] == 0, {0}, 
                      Select[Flatten[{cc /. Solve[ ratF[y + cc]  == -ratF[-y + cc], cc],
                                      cc /. Solve[ ratF[y + cc]  ==  ratF[-y + cc], cc]}],
                                     FreeQ[#, y, {0, \[Infinity]}] && Im[#] == 0&]
                      ],
                         0.1, {}]; 

            If[Length[symmetryCenters] === 1,
               {min, max} = symmetryCenters[[1]] + {-Mean[Abs[{min, max}]], +Mean[Abs[{min, max}]]}
               ]; 

            {padRange[{min, max}, 0.01],
               "Zeros" -> zeroPoints,
               "Extrema" -> extremaPoints,
               "Poles" -> poles,
               "InflectionPoints" -> inflectionPoints,
               "RemovableSingularities" -> Transpose[{removableSingularities, removableSingularityValues}],
               PlotRange -> N[verticalPlotRange]}
         ]; (* end claculating structurePlotRange *)   
         theFinalPoles = Cases[structurePlotRange,  HoldPattern["Poles"-> _], {0, \[Infinity]}];
         (* global view *) 
       TimeConstrained[
             leftAsymptotic = Normal[Series[ratF[y], {y, -Infinity, 0}]];
             rightAsymptotic = Normal[Series[ratF[y], {y, +Infinity, 0}]]; ,
                                0.2];
         globalPlotRange = 
         If[MemberQ[leftAsymptotic, y, {0, Infinity}] || MemberQ[rightAsymptotic, y, {0, Infinity}],
            spr = structurePlotRange[[1]];
            cspr = Mean[spr]; \[CapitalDelta]spr = Abs[Subtract @@ spr];
            {{cspr - 3 \[CapitalDelta]spr, cspr + 3 \[CapitalDelta]spr}, 
             (* keep poles for plotting exclusion *) If[theFinalPoles === {}, {}, Sequence @@ theFinalPoles]}, 
           {}];
          {PRData[structurePlotRange, "HorizontalPlotRangeType" -> "ShowZerosExtremasInflections"],
           If[globalPlotRange =!= {},
              PRData[globalPlotRange, "HorizontalPlotRangeType" -> "ShowGlobalShape"],  
              PRData[{enlargePlotRange[{structurePlotRange[[1]], {}}, 2][[1]], 
                      (* don't annotate, but avoid vertical lines at tangents *)
                      If[# =!= {}, Sequence @@ #, #]& @ Cases[structurePlotRange, HoldPattern["Poles"->_], \[Infinity]]}, 
                                      "HorizontalPlotRangeType" -> "ShowGlobalShape"]]}
       ]


findRemovableSingularities[rf_, x_] :=
Module[{(*numDenPairs, gcdList, rsList*)},
       numDenPairs = Cases[rf, HoldPattern[_?(MemberQ[#, x, {0, \[Infinity]}]&& PolynomialQ[#, x]&)/_?(MemberQ[#, x, {0, \[Infinity]}]&& PolynomialQ[#, x]&)], {0, \[Infinity]}];
       (* just call Reduce here; time-wise assume we don't get too bad input *)
       gcdList  = DeleteCases[PolynomialGCD[Numerator[#], Denominator[#], Extension -> Automatic]& /@ numDenPairs, 1];
       rsList = DeleteCases[Flatten[x /. {ToRules[Reduce[# == 0, x, Reals]]}& /@ gcdList], (* from approximate gcds *) x];
       rsList 
      ]


(* ::Input:: *)
(*findRemovableSingularities[PRF["RationalFunction", (x^2-4)/(x+2), x] , x]*)


(* ::Input:: *)
(*findRemovableSingularities[(1+1/(200*x))^200*x, x]*)


(* ::Input:: *)
(*findRemovableSingularities[(x^2+2x-3)/(x+3), x]*)


(* ::Input:: *)
(*findRemovableSingularities[((-4+x^2)/(-x+x^4) + 1)/(x-4), x]*)


(* ::Input:: *)
(*findRemovableSingularities[((-4.+x^2)/(-x+x^4) + 1)/(x-4.), x]*)


(* ::Input:: *)
(*findRemovableSingularities[(x^2-1)/(x+1), x]*)


(* ::Input:: *)
(*findRemovableSingularities[(x^2-1.)/(x+1.), x]*)


(* ::Input:: *)
(*PRF["RationalFunction", (x^2-x)/(x^2-3x+2), x] *)


(* ::Input:: *)
(*PRF["RationalFunction", (4*x^3)/(4+x^4), x] *)


(* ::Input:: *)
(*PRF["RationalFunction", x^2/(x-1), x] *)


(* ::Input:: *)
(*PRF["RationalFunction", (4*x^3)/(x^4-4), x] *)


(* ::Input:: *)
(*PRF["RationalFunction", (x^2-4)/(x+2), x] *)


(* ::Input:: *)
(*PRF["RationalFunction", (7.7+x)/(22.090000000000003+x), x] *)


(* ::Input:: *)
(*PRF["RationalFunction", ((-4+x^2)/(-x+x^4) + 1)/(x-4), x] *)


(* ::Input:: *)
(*PRF["RationalFunction", (-4+x^2)/(-x+x^4), x] *)


(* ::Input:: *)
(*PRF["RationalFunction", 6/x^4, x] *)


(* ::Input:: *)
(*PRF["RationalFunction", 1/(x-1), x] *)


(* ::Input:: *)
(*PRF["RationalFunction", 0.25/(x-I*2^0.5-2)+0.25/(x+I*2^0.5-2)+ -1/(2*x), x] *)


(* ::Input:: *)
(*PRF["RationalFunction",1 + 1/x^2, x] *)


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Algebraic Functions   *)


Clear[AlgebraicQ];
AlgebraicQ[expr_Plus, x_] := And @@ (AlgebraicQ[#, x]& /@ (List @@ expr))

(*AlgebraicQ[expr_?PolynomialQ, x_] := True /; PolynomialQ[expr, x]*)

AlgebraicQ[a_/b_, x_] := AlgebraicQ[a, x] && AlgebraicQ[b, x]

AlgebraicQ[a_ b_, x_] := AlgebraicQ[a, x] && AlgebraicQ[b, x]

AlgebraicQ[Power[a_, b_Integer | b_Rational], x_] := AlgebraicQ[a, x]

AlgebraicQ[_?NumericQ, x_] := True

AlgebraicQ[x_, x_] := True


(* ::Input:: *)
(*AlgebraicQ[x^20 Sin[x], x]*)


Clear[replaceStep];
replaceStep[{bag_, expr_}, x_] := 
Module[{},
       roots = Cases[expr, Power[_?(RationalFunctionQ[#, x]&), e_Rational], {0, \[Infinity]}];
       \[Lambda] = Length[bag];
       rules = MapIndexed[(#1 -> C[\[Lambda] + #2[[1]]])&, roots];
       newExpr = expr //. rules;
       {Flatten[{bag, rules}], newExpr}
      ]


PRF["Algebraic", expr_, x_] := 
Module[{horizontalRangeFactor = 0.02,
        y, yI, fpl, auxEqs, body, auxVars, extremaData, thePoly, theDPoly, xZerosPre,
        xZeros, xPolesPre, xPoles, xExtremasP, xExtremasI, xExtremas, exprD, inflections, theDDPoly, \[CapitalDelta]Extrema,
        xInflectionsPre, exprDD, xPoints, allNumbers, aScale, onsetTerms, xOnsets,
        baseOnsetTerms, curvatureCenter, xMin, xMax, sols, sel,
        structurePlotRange, globalPlotRange,
        extendedExtremaData, maxExtremaIntervalValues, maxExtendedExtremaIntervalValues, red, xValueList, maxE, minE,
        rightxValues, leftxValues, fastIncreasingFunctionFlag,
        leftAsymptotic, leftAsymptotic2, rightAsymptotic, rightAsymptotic2 
        },
       fastIncreasingFunctionFlag = False;
       fpl = FixedPoint[replaceStep[#, x]&, {{}, expr}];
       auxEqs = Numerator[Together[(#1[[1]] - #2^Denominator[#1[[2]]])& @@@ First[fpl]]];
       body = fpl[[2]];
       auxVars = Cases[auxEqs, _C, \[Infinity]] // Union;
       extremaData =
        CalculateTimeConstrained[
           gb = GroebnerBasis[Flatten[{auxEqs, Numerator[Together[y[x] - body]]}], 
                                          {}, auxVars,
                              MonomialOrder -> EliminationOrder,
                              CoefficientDomain -> InexactNumbers[300]];
        thePoly = Rationalize[gb, 10^-5][[1]];
        theDPoly = D[thePoly, x] /. y'[x] -> 0;
        xZerosPre = realRoots[thePoly /. y[x] -> 0, x]; 
        xZeros = Select[xZerosPre, Chop[N[expr /. x -> #], 10^-6] == 0&];
        xPolesPre = realRoots[Numerator[Together[thePoly /. y[x] -> 1/yI[x]]] /. yI[x] -> 0, x];
        xPoles = Select[xPolesPre, Chop[1/N[expr /. x -> #], 10^-6] == 0&];
        xExtremasP = Select[DeleteCases[Flatten[{Chop[x/. 
                                 NSolve[{thePoly == 0, theDPoly == 0}, {x}, {y[x]}]]}], x],
                           Im[#] == 0&];
        exprD =  D[expr, x];
        xExtremasI = Select[xExtremasP, Chop[N[exprD /. x -> #]] == 0&]; 
        xExtremas = Union[N[Flatten[{xExtremasP, xExtremasI}]]];
        (* must try harder *)
        inflections = 
         If[Flatten[{xZeros, xPoles, xExtremas}] =!= {}, {},  
            theDDPoly = D[thePoly, x, x] /. y''[x] -> 0;
            xInflectionsPre = Select[DeleteCases[Flatten[{Chop[x/. 
                                 NSolve[{thePoly, D[thePoly, x] == 0, theDDPoly == 0}, 
                                        {x}, {y[x], y'[x]}]]}], x],
                           Im[#] == 0&];
            exprDD =  D[expr, x, x];
            Select[xInflectionsPre, Chop[N[Im[exprDD /. x -> #]]] == 0&];
            {}
            ];
        {xZeros, xPoles, xExtremas, inflections}, 
             0.5] // Quiet;  
       (* end extrema data calculation *)
        xOnsets =. ;
        allNumbers = Cases[expr, _?NumberQ, {0, \[Infinity]}];
        aScale = If[allNumbers === {}, 1, Mean[Abs @ allNumbers]];
        \[CapitalDelta]Extrema = Max[extremaData] - Min[extremaData];
        xPoints = 
         Which[extremaData === $Aborted, (* too complicated function *) aScale {-1, 1}, 
               Flatten[extremaData] =!= {} && 
              (Mean[Flatten[extremaData]] != 0 && \[CapitalDelta]Extrema/Abs[Mean[Flatten[extremaData]]] > 10^-2) || \[CapitalDelta]Extrema > scale/10,   
               If[\[CapitalDelta]Extrema > 10^-1 aScale,  
                  extendedExtremaData = {Min[extremaData] - \[CapitalDelta]Extrema/2, Max[extremaData] + \[CapitalDelta]Extrema/2};
                  (* for fast increasing functions, extend the range less *) 
                  maxExtremaIntervalValues = Quiet[Max[Select[Abs[ expr /. ( {x -> #}& /@  Flatten[ extremaData])], 
                                                       (* exclude poles, removable singularities *) NumberQ]]];
                  maxExtendedExtremaIntervalValues  = Max[Abs[Abs[expr /. ( {x -> #}& /@  extendedExtremaData)]]];
                  If[TrueQ[maxExtendedExtremaIntervalValues/maxExtremaIntervalValues > 8 || maxExtremaIntervalValues == 0.], 
                     (* try to find tighter extension *)
                    red = CalculateTimeConstrained[Quiet[Reduce[expr == 3 maxExtremaIntervalValues || expr == -3 maxExtremaIntervalValues , x, Reals]] , 0.25];
                    If[Head[red] === $Aborted || Not[Head[red] === Or], extendedExtremaData,  
                       xValueList = x /. {ToRules[red]};
                       maxE = Max[extremaData];
                       minE = Min[extremaData];
                       rightxValues =  Sort[Select[xValueList, # > maxE&]];
                       leftxValues =  Reverse @ Sort[Select[xValueList, # < minE&]];
                       If[rightxValues =!= {} && leftxValues =!= {},  fastIncreasingFunctionFlag = True; {leftxValues[[1]], rightxValues[[1]]},
                         extendedExtremaData]  
                         ], 
                      extendedExtremaData
                        ],
                  aScale {-1, 1}],
               True,   
               (* try onsets of reals instead *)
               onsetTerms = First /@ fpl[[1]];
               baseOnsetTerms = onsetTerms /. _C -> 0;
               baseOnsetTermsPolys = Numerator[Together[First /@ Select[baseOnsetTerms, MemberQ[#, x, {0, \[Infinity]}]&]]];
               xOnsets = Union[Flatten[realRoots[#, x]& /@ baseOnsetTermsPolys]];
               {extremaData /. $Aborted -> {}, xOnsets} // Flatten] // Flatten // Union;
           
       {xMin, xMax} = If[xPoints === {}, {0, 0}, {Min[#], Max[#]}&[xPoints]];
       Which[-Infinity < xMin < xMax < Infinity, \[CapitalDelta]x = (xMax - xMin) * If[fastIncreasingFunctionFlag, 1/100, 1/4],
             (* use scale *) xMax == xMin, (*\[CapitalDelta]x = Abs[aScale];*)
             (* The following Replace come from bug 281655. Vertical scale factors shouldn't effect the horizontal plot range in theory. *)
             \[CapitalDelta]x = If[# === {}, 1, Mean[Abs @ #]]&[Cases[Replace[expr, a_?NumberQ b_. :> b, {1}], _?NumberQ, {0, \[Infinity]}]];, 
            (* function has no structure and is scale-free *)
            {xMin, xMax} == {Infinity, -Infinity},
            curvatureCenter = DeleteCases[x/. Solve[D[ expr, {x, 3}] == 0, x], x]; 
            If[curvatureCenter === {},
               {xMin, xMax} = {-1, 1}; \[CapitalDelta]x = 1,
               {xMin, xMax} = curvatureCenter[[1]] {1, 1}; \[CapitalDelta]x = 1;
               ]
            ];

       structurePlotRange = {padRange[{xMin - \[CapitalDelta]x, xMax + \[CapitalDelta]x}, horizontalRangeFactor], {}};
       
       globalPlotRange = 
       TimeConstrained[
       leftAsymptotic = Series[expr, {x, -Infinity, 0}];
       leftAsymptotic2 = Series[expr - Normal[leftAsymptotic], {x, -Infinity, 0}];
       rightAsymptotic = Series[expr, {x, Infinity, 0}];
       rightAsymptotic2 = Series[expr - Normal[rightAsymptotic], {x, Infinity, 0}];    
       If[MemberQ[Normal[leftAsymptotic], x] && MemberQ[Normal[leftAsymptotic2], x] &&
          MemberQ[Normal[rightAsymptotic], x] && MemberQ[Normal[rightAsymptotic2], x],
          sols = Flatten[{Solve[Normal[leftAsymptotic] == 5 Normal[leftAsymptotic2] || 
                                Normal[leftAsymptotic] == -5 Normal[leftAsymptotic2], x],
                          Solve[Normal[leftAsymptotic] == 5 Normal[leftAsymptotic2] || 
                                Normal[leftAsymptotic] == -5 Normal[leftAsymptotic2], x]}];
          sel = Select[Last /@ N[sols], Abs[Im[#]] == 0.&];
          If[Length[sel] <= 1 && Min[sel] < xMin - \[CapitalDelta]x && Max[sel] > xMax + \[CapitalDelta]x, 
             enlargePlotRange[structurePlotRange, 3], 
            {{Min[sel], Max[sel]}, {}}],
          enlargePlotRange[structurePlotRange, 3]   
          ], 1/2, {}];
         {PRData[structurePlotRange, "HorizontalPlotRangeType" -> "ShowZerosExtremasInflections"],
          If[globalPlotRange =!= {},
             PRData[globalPlotRange, "HorizontalPlotRangeType" -> "ShowGlobalShape"],
             Sequence @@ {}]}

          ]


(* ::Input:: *)
(*PRF["Algebraic",(x+I*Sqrt[4-x^2])^4-4*(x+I*Sqrt[4-x^2])^2, x] *)


(* ::Input:: *)
(*PRF["Algebraic",(Sqrt[1/(x^2+x)]+Sqrt[x/(1+x)])^2, x] *)


(* ::Input:: *)
(*PRF["Algebraic",x Sqrt[1 - x^4], x] *)


(* ::Input:: *)
(*PRF["Algebraic",Sqrt[x^3+5*x^2]/x, x] *)


(* ::Input:: *)
(*PRF["Algebraic",((x-x^9+1)^2)^(1/3), x] *)


(* ::Input:: *)
(*PRF["Algebraic",(0.7 - 0.8 I) Sqrt[x], x] *)


(* ::Input:: *)
(*PRF["Algebraic",(x - 9)/Sqrt[4*x^2 + 3*x + 2], x] *)


(* ::Input:: *)
(*PRF["Algebraic",(Sqrt[1+x^6]-Sqrt[1-2 x^4])/(x+x^2), x] *)


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Rational linear trigonometric   *)


Clear[RationalLinearTrigonometricQ, RationalLinearTrigonometricQ1];

RationalLinearTrigonometricQ[expr_, x_] := 
        RationalLinearTrigonometricQ1[Rationalize[expr], x] || 
       TrueQ[RationalLinearTrigonometricQ2[Rationalize[expr], x] ]


RationalLinearTrigonometricQ1[expr_Plus, x_] := 
         And @@ (RationalLinearTrigonometricQ1[#, x]& /@ (List @@ expr))

RationalLinearTrigonometricQ1[a_/b_, x_] := RationalLinearTrigonometricQ1[a, x] && 
                                            RationalLinearTrigonometricQ1[b, x]

RationalLinearTrigonometricQ1[a_ b_, x_] := RationalLinearTrigonometricQ1[a, x] && 
                                            RationalLinearTrigonometricQ1[b, x]

RationalLinearTrigonometricQ1[Power[a_, b_Integer], x_] := RationalLinearTrigonometricQ1[a, x]

RationalLinearTrigonometricQ1[_?NumericQ, x_] := True

RationalLinearTrigonometricQ1[(head:(Sin | Cos | Tan | Cot | Sec | Csc))[arg_], x_] :=
                       PolynomialQ[arg, x] && Exponent[arg, x] <= 1 &&
                       DeleteCases[CoefficientList[arg, x], _Integer | _Rational] === {}


(* sums that contain say (Pi + 1/2) n x in trig arguments; 
   argument is linear in x with a potential nonrational prefactor; 
   but the prefactor is the same for all terms *) 
RationalLinearTrigonometricQ2[HoldPattern[Plus[p:((_. (Sin | Cos | Tan | Cot | Sec | Csc)[_. x_]) ..)]], x_] := 
NumericQ[ Plus[p] /. {(_Sin | _Cos | _Tan | _Cot | _Sec | _Csc) :> RandomReal[]} ] &&
Module[{P1, P2, P2a, factors},
       P1 = {p};
       factors =  P1 /. (_Sin | _Cos | _Tan | _Cot | _Sec | _Csc) :> 1;
       (And @@ (NumericQ /@ factors)) &&
        (P2 = {p} /. (_. Sin[f_]  | _. Cos[f_]  | _. Tan[f_]  | _. Cot[f_]  | _. Sec[f_]  | _. Csc[f_] ) :> f;
         P2a = Cancel[P2/P2[[1]]];
         MatchQ[P2a, {(_Integer | _Rational) ..}] 
        ) 
       ]


(* ::Input:: *)
(*RationalLinearTrigonometricQ2[Sin[2 E x] + Tan[5 E x], x]*)


(* ::Input:: *)
(*RationalLinearTrigonometricQ2[Sum[1/(2 k+1) Sin[(2 k+1) Pi x/(1/2)],{k,0,10}] + Tan[24 Pi x], x]*)


(* ::Input:: *)
(*RationalLinearTrigonometricQ2[Sin[Pi x] + 4 Sin[2 Pi x^2], x]*)


(* ::Input:: *)
(*RationalLinearTrigonometricQ[Sin[x] + 4 Sin[4 x], x]*)


(* ::Input:: *)
(*RationalLinearTrigonometricQ[Sin[x] + 4 Sin[4 Pi x], x]*)


(* ::Input:: *)
(*RationalLinearTrigonometricQ[Sum[1/(2 k+1) Sin[(2 k+1) Pi x/(1/2)],{k,0,10}], x]*)


explicitVerticalTrigPlotRange[a_. (Sin | Cos)[b_. x_ + c_.]^n_Integer?Positive /; NumericQ[a] && NumericQ[b] && NumericQ[c], x_] := 
                              a If[EvenQ[n], {-0.01, 1.01}, {-1.01, 1.01}]

explicitVerticalTrigPlotRange[_, _] := $Failed


(* ::Input:: *)
(*explicitVerticalTrigPlotRange[Sin[x]^23, x]*)


PRF["RationalLinearTrigonometric", expr_, x_] :=
Module[{(* verticalRangeFactor = 1/3, horizontalRangeFactor = 0.02, tMax = 0.1*)},
       theOccurringTrigs = Cases[Rationalize[expr], 
                          _Cos | _Sin | _Tan | _Cot | _Csc | _Sec, {0, Infinity}];
       theTrigLinArguments = DeleteCases[Coefficient[Union[First /@ theOccurringTrigs], x, 1], 0];
       shortestSingleWaveLength = 1/Max[theTrigLinArguments];
       numberOfShortestWavesInSinglePeriod = 
             If[Length[theTrigLinArguments] === 1, 1, 
                1/GCD @@ (theTrigLinArguments shortestSingleWaveLength)] ;
       theUsedPeriodFraction = If[numberOfShortestWavesInSinglePeriod < 12, 
                                  numberOfShortestWavesInSinglePeriod, 12]/numberOfShortestWavesInSinglePeriod;
       periodLength = If[MemberQ[theOccurringTrigs, (Sin|Cos|Sec|Csc)[_]], 1, 1/2] numberOfShortestWavesInSinglePeriod shortestSingleWaveLength;
       theIntervalLength = theUsedPeriodFraction periodLength 2Pi; 
       {PRData[{theIntervalLength {-1, 1} fuzzFactorRationalTrig,
                Ticks -> {Table[k theIntervalLength {1, 1}, {k, -1, 1, 1/2}], Automatic}, 
                If[explicitVerticalTrigPlotRange[expr, x] === $Failed, Sequence @@ {}, 
                   PlotRange -> explicitVerticalTrigPlotRange[expr, x]]},
                "HorizontalPlotRangeType" -> "ShowFewPeriods"],
        PRData[{theIntervalLength 4  {-1, 1} fuzzFactorRationalTrig, 
                (* help plot to detect all structures *)
                If[numberOfShortestWavesInSinglePeriod < 20, Sequence @@ {},
                   PlotPoints -> Min[1000, Ceiling[4 numberOfShortestWavesInSinglePeriod]]], 
                If[explicitVerticalTrigPlotRange[expr, x] === $Failed, Sequence @@ {}, 
                   PlotRange -> explicitVerticalTrigPlotRange[expr, x]]},
               "HorizontalPlotRangeType" -> "ShowMorePeriods"] 
        }
       ]


(* ::Input:: *)
(*PRF["RationalLinearTrigonometric", Sin[x], x]*)


(* ::Input:: *)
(*PRF["RationalLinearTrigonometric", Tan[x], x]*)


(* ::Input:: *)
(*PRF["RationalLinearTrigonometric", Sin[2*Pi*x*440]+Sin[2*Pi*x*450], x]*)


(* ::Input:: *)
(*PRF["RationalLinearTrigonometric", Sum[1/(2 k+1) Sin[(2 k+1) Pi x/(1/2)],{k,0,10}], x]*)


(* ::Input:: *)
(*PRF["RationalLinearTrigonometric",Sin[x], x] *)


(* ::Input:: *)
(*PRF["RationalLinearTrigonometric",Sin[x/2], x] *)


(* ::Input:: *)
(*PRF["RationalLinearTrigonometric",Sin[x/3], x] *)


(* ::Input:: *)
(*PRF["RationalLinearTrigonometric",Sin[x]^23, x] *)


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Rational linear Fresnel   *)


RationalLinearFresnelQ[expr_, x_] := 
         Cases[expr, (FresnelC | FresnelS)[_. (_. x + _.)], {0, \[Infinity]}] =!= {}


PRF["RationalLinearFresnel", expr_, x_] :=
Module[{(* verticalRangeFactor = 1/3, horizontalRangeFactor = 0.02, tMax = 0.1*)
        theOccurringFresnels, prePeriod, periodLength, theIntervalLength, center},
        theOccurringFresnels = Cases[expr, (FresnelC | FresnelS)[_. (_. x + _.)], {0, \[Infinity]}];
        prePeriod = Periodic`PeriodicFunctionPeriod[
			Table[C[k], {k, Length[theOccurringFresnels]}].(theOccurringFresnels /. {FresnelC -> Cos, FresnelS -> Sin }), x];
        periodLength = If[NumericQ[prePeriod], prePeriod,
                          (* incommensurable *) 3Pi/Max[Exponent[First[#], x]& /@ theOccurringFresnels]];  
        center = Mean[Flatten[(x /. Solve[#[[1]] == 0, x])& /@ theOccurringFresnels]];
        theIntervalLength = Sqrt[periodLength]; 
       {PRData[{center + theIntervalLength 1. {-1, 1}, {}},
                "HorizontalPlotRangeType" -> "ShowFewQuasiPeriods"],
        PRData[{center + theIntervalLength  2. {-1, 1}, {}},
               "HorizontalPlotRangeType" -> "ShowMoreQuasiPeriods"] 
        }
       ]


(* ::Input:: *)
(*PRF["RationalLinearFresnel", *)
(*Sqrt[Pi/6]*(Cos[11/12]*FresnelC[(1+6*x)/Sqrt[6*Pi]]-FresnelS[(1+6*x)/Sqrt[6*Pi]]*Sin[11/12]), x] // N*)


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Restricted functions*)


RestrictedFunctionQ[expr_, x_] := MemberQ[expr, _InverseErf, {0, \[Infinity]}]


PRF["RestrictedFunction", expr_, x_] :=
Module[{tMax = 0.1, 
        invProbFunctions, args, zeros, plusOnes, minusOnes, min0, max0, min1, max1, \[CapitalDelta]i, mi,
        \[CapitalDelta]o, mo, pri, pro},
       invProbFunctions = Cases[expr, InverseErf[_?(MemberQ[#, x, {0, \[Infinity]}]&)], {0, \[Infinity]}];
       args = Union[First /@ invProbFunctions];
       zeros = Flatten[TimeConstrained[ x /. Solve[# == 0, x], 0.1, {}]& /@ args];
       plusOnes = Cases[Chop[N @ #], _Real]& @ Flatten[CalculateTimeConstrained[ x /. Solve[# == 1, x], tMax, {}]& /@ args];
       minusOnes = Cases[Chop[N @ #], _Real]& @ Flatten[CalculateTimeConstrained[ x /. Solve[# == -1, x], tMax, {}]& /@ args];
       {min0, max0} = {Min[zeros], Max[zeros]};
       {min1, max1} = {Min[{minusOnes, plusOnes}], Max[{minusOnes, plusOnes}]};
       \[CapitalDelta]i = If[zeros === {}, 0, max0 - min0];
       mi = If[zeros === {}, 0, Mean[{min0, max0}]];
       \[CapitalDelta]o = If[Flatten[{minusOnes, plusOnes}] === {}, 0, max1 - min1];
       mo = If[Flatten[{minusOnes, plusOnes}] === {}, 0, Mean[{min1, max1}]];
       If[\[CapitalDelta]i == \[CapitalDelta]o == 0, $Failed,
          pri = Which[\[CapitalDelta]i =!= 0, {mi - \[CapitalDelta]i, mi + \[CapitalDelta]i},
                      \[CapitalDelta]i === 0 && \[CapitalDelta]o =!= 0, {mi - \[CapitalDelta]o/3, mi + \[CapitalDelta]o/3}];
          pro = Which[\[CapitalDelta]i =!= 0, {mi - \[CapitalDelta]i, mi + \[CapitalDelta]i},
                      \[CapitalDelta]i === 0 && \[CapitalDelta]o =!= 0, {mo - 0.52 \[CapitalDelta]o, mo + 0.52 \[CapitalDelta]o}];

      {PRData[{pri, {}}, "HorizontalPlotRangeType" -> "ShowInnerShape"],
       PRData[{pro, {}}, "HorizontalPlotRangeType"->"ShowGlobalShape"]} 
      ] 
       ]


(* ::Input:: *)
(*PRF["RestrictedFunction",1 - Exp[-InverseErf[z^2]], z] *)


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Rational linear exponential   *)


Clear[RationalLinearExponentialQ, RationalLinearExponentialQ1];

(* let simple cases be handled by table looukup *)

RationalLinearExponentialQ[expr_, x_] := 
        Not[MatchQ[expr, a_. Power[b_, c_. x] + d_. /; And @@ (NumericQ /@ {a, c, d}) ]] && 
        Not[MatchQ[expr, a_. Power[b_, c_. x^2] + d_. /; (b == E) && (And @@ (NumericQ /@ {a, c, d})) ]] &&
        Not[MatchQ[expr, f_./(a_ + b_. base_^(c_. x)) /; Im[a] == 0 && Im[b] == 0 && Im[c] == 0 && Im[base] == 0]] &&
        RationalLinearExponentialQ1[Rationalize[expr, 0], x]


RationalLinearExponentialQ1[expr_, x_] := 
Module[{allExps, y},
       allExps = Cases[TrigToExp @ expr, Power[_?NumericQ, _?(MemberQ[#, x, {0, Infinity}]&)], Infinity] // Union; 
       Length[Union[First /@ allExps]] === 1 && allExps[[1, 1]] > 0 &&
       NumericQ[expr /. (Alternatives @@ allExps) :> E/Sqrt[RandomInteger[{1, 100}]]] &&
       (And @@ (MatchQ[Head[#], Integer | Rational]& /@ ((Last /@ allExps)/allExps[[1, 2]]))) &&
       generalizedAlgebraicQ[Together[PowerExpand[expr /. x -> Log[allExps[[1, 1]], y]]], y]
       ]


generalizedAlgebraicQ[expr_, y_] := 
       Complement[Select[Union[Cases[expr, _Symbol, {-1}, Heads -> True]], Not[NumericQ[#]]&],
                  {Plus, Power, Times, y}] === {}


(* ::Input:: *)
(*RationalLinearExponentialQ[1/(1-2^x), x]*)


(* ::Input:: *)
(*RationalLinearExponentialQ[3^x/(3^x+2^x), x]*)


(* ::Input:: *)
(*generalizedAlgebraicQ[(y^(3/2) - y)/(y^Sqrt[2]-1), y]*)


(* ::Input:: *)
(*RationalLinearExponentialQ[1/(1+E^(-(x/0.5`))), x]*)


(* ::Input:: *)
(*RationalLinearExponentialQ[1/(1+E^(-2.8284271247461903` x)), x]*)


(* ::Input:: *)
(*RationalLinearExponentialQ1[1/(1+E^(-2.8284271247461903` x)), x]*)


(* ::Input:: *)
(*RationalLinearExponentialQ1[(Exp[3x] - Exp[x])/(Exp[2x] - Exp[x]), x]*)


(* ::Input:: *)
(*RationalLinearExponentialQ1[SinIntegral[Exp[x]], x]*)


(* ::Input:: *)
(*RationalLinearExponentialQ1[(1+Exp[3x] +E^(-((x-100)/2)))^(-1), x] *)


PRF["RationalLinearExponential", (a_ + E^(b_. (c_. (d_. x_ + e_.))))^n_, x_] :=
With[{center = -1. e/d, width = Abs[b c d]}, 
     {PRData[{{center - 3/width, center + 3/width}}, "HorizontalPlotRangeType" -> "ShowZerosExtremasInflections"],
      PRData[{{center - 12/width, center + 12/width}, {}},"HorizontalPlotRangeType" -> "ShowGlobalShape"]
      } 
     ] /; a > 0 && n < 0


(* ::Input:: *)
(*PRF["RationalLinearExponential", (1+E^(-((x-100)/2)))^(-1), x] *)


PRF["RationalLinearExponential", expr_, x_] :=
Module[{(* verticalRangeFactor = 1/3, horizontalRangeFactor = 0.02, tMax = 0.1, *)
        (*theOccurringExps, theExpLinArguments, auxM, minExp, baseExp, expBase, exprNew, prfData, 
        finitize, revS, revS1, prfData1, cRange,
        int1o, int2o, int1, int2, a, b, c, d, int1N, int2N,
        realZeros, zeroPoint, zeroPointScale*)},

       theOccurringExps = Cases[expr, Power[_?NumericQ, _?(MemberQ[#, x, {0, Infinity}]&)], Infinity]; 
       theExpLinArguments = Last /@ theOccurringExps;
       auxM = theExpLinArguments /. x -> 1;

  Which[(* purely real arguments *) 
        Max[Abs[Im[auxM]]] == 0.,
        minExp = Min[auxM];
        baseExp = theExpLinArguments[[Position[auxM, minExp][[1, 1]]]];
        expBase = theOccurringExps[[1, 1]];
        exprNew = expr /. expBase^v_?(MemberQ[#, x, {0, Infinity}]&) :> y^(Cancel[v/baseExp]);

       If[(* purely linear in new variables *) 
          TrueQ[PolynomialQ[exprNew, y] && Exponent[exprNew, y] === 1 ],
          realZeros = Reduce[expr == 0, x, Reals];
          If[realZeros === False, 
             {PRData[{{-1., 1.}}," HorizontalPlotRangeType" -> "ShowZerosExtremasInflections"],
              PRData[{{-5., 5.},{}}, "HorizontalPlotRangeType" -> "ShowGlobalShape"]},
          zeroPoint = (x /. {ToRules[realZeros]})[[1]];
          zeroPointScale = Max[ Abs[1. zeroPoint], 1];
             {PRData[{zeroPoint + {-0.5, 0.5} zeroPointScale}, "HorizontalPlotRangeType"->"ShowZerosExtremasInflections"],
              PRData[{zeroPoint + {-2.5, 2.5} zeroPointScale, {}}, "HorizontalPlotRangeType"->"ShowGlobalShape"]} 
             ],
        (* more complicated case *)
        prfData = DeleteCases[
                  Which[PolynomialQ[exprNew, y], PRF["Polynomial", exprNew, y],
                        True, PRF["RationalFunction", exprNew, y]
                        ], ("Zeros" | "Extrema" | "Poles" | 
                            "InflectionPoints" | PlotRange)  -> _, Infinity];
        finitize[{x1_, x2_}] := Which[x1 === -Infinity && NumberQ[x2], {x2 - 2 Abs[x2], x2}, 
                                      NumberQ[x1] && x2 === Infinity, {x1, x1 + 2 Abs[x1]}, 
                                      True, {x1, x2}];
        revS[y_] = N[xz /. Solve[(Power[expBase, baseExp] /. x -> xz) == y, xz][[1]]] /. ConditionalExpression[b_, __] :> (b /. _C :> 0);
        revS1[{y1_, y2_}] := finitize @ Sort[N[{If[Head[#] === Real, #, Min[Re[#] - Im[#], Re[#] + Im[#]]]&[revS[y1]],
                                                If[Head[#] === Real, #, Min[Re[#] - Im[#], Re[#] + Im[#]]]&[revS[y2]]}], 
                                             #1 < #2&];
        (* symmetric/antisymmetric functions *)
        prfData1 = If[Abs[Cancel[(expr /. x -> -x)/expr]] === 1,  
                      prfData /. {c1_?NumberQ, c2_?NumberQ} :> Mean[Abs[revS1[SortBy[{c1, c2}, Less]]]] {-1, 1}, 
                      prfData /. {c1_?NumberQ, c2_?NumberQ} :> revS1[SortBy[{c1, c2}, Less]]
                     ] ; 
        If[Length[prfData1] === 1, prfData1,
          int1o = prfData1[[1, 1, 1]]; int2o = prfData1[[2, 1, 1]];
          (* symmetrize *)
          If[TimeConstrained[Together[expr - (expr /. x -> -x)] == 0, 0.05, False],
             (* in the symmetric case, the inner interval must be strictly inside the outer one *)
             If[Abs[Subtract @@ int1o] < Abs[Subtract @@ int2o], 
                int1o = {-Max[Abs[int1o]], Max[Abs[int1o]]};
                int2o = {-Max[Abs[int2o]], Max[Abs[int2o]]},
                int1o = {-Max[Abs[int1o]], Max[Abs[int1o]]};
                int2o = 2 int1o
               ]
             ];
           int1N = int1o;
           int2N = int2o;
          {a, b, c, d} = Sort[Flatten[{int1N, int2N}], Less];
          int1N = If[b =!= c, {b, c}, {(a + b)/2, (c + d)/2}];
          int2N = {a, d};
          MapAt[int2N&, MapAt[int1N&, prfData1, {1, 1, 1}], {2, 1, 1}]
         ]
        ],

   (* purely imaginary arguments *)   
    Max[Abs[Re[auxM]]] == 0.,
    cRange = {-2Pi, 2Pi}/Max[Abs[auxM]];
    {PRData[{cRange},"HorizontalPlotRangeType" -> "ShowZerosExtremasInflections"],
     PRData[{3 cRange, {}},"HorizontalPlotRangeType" -> "ShowGlobalShape"]
     },

    True, $Failed
        ]
]


(* ::Input:: *)
(*PRF["RationalLinearExponential", 80E^(-2*x)+92*(1-E^(-2*x)), x] *)


(* ::Input:: *)
(*PRF["RationalLinearExponential", (1+Exp[3x] +Exp[1 x])^(-1), x] *)


(* ::Input:: *)
(*PRF["RationalLinearExponential", 1/(1-2^x),  x] *)


(* ::Input:: *)
(*PRF["RationalLinearExponential", 1/(1-E^x),  x] *)


(* ::Input:: *)
(*PRF["RationalLinearExponential", -((4*E^(2*x))/(1+E^x)^2)+(4*E^x)/(1+E^x), x] *)


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Rational linear exponential  integrated*)


RationalLinearExponentialIntegratedQ[expr_, x_] := 
Not[MatchQ[expr, f_./(a_ + b_. base_^(c_. x)) /; Im[a] == 0 && Im[b] == 0 && Im[c] == 0 && Im[base] == 0]] &&
RationalLinearExponentialQ[D[expr, x], x] 


(* ::Input:: *)
(*RationalLinearExponentialIntegratedQ[((-x)*Log[2]+Log[2^x+3^x])/Log[3/2], x]*)


(* ::Input:: *)
(*RationalLinearExponentialIntegratedQ[3^3^x, x]*)


PRF["RationalLinearExponentialIntegrated", expr_, x_] := PRF["RationalLinearExponential", D[expr, x], x]


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Nested exponentials (to do)*)


(* ::Input:: *)
(*(**)
(*  innerExponentials = Union[Cases[expr = TrigToExp[expr], Power[E, _. x], \[Infinity]]]*)
(*  Union[1/innerExponentials]===innerExponentials*)
(*  expr /. {innerExponentials[[1]] -> Y, innerExponentials[[2]] ->1/ Y}*)
(**)*)


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Table look-up   *)


Clear[scale, scaleCache, bothScales, specialEnlargementFactor];


(* ::Text:: *)
(*Bases cases:*)


scale[Sin[x_], x_] := {-2Pi, 2Pi}
scale[Cos[x_], x_] := {-2Pi, 2Pi}
scale[Sec[x_], x_] := {-2Pi, 2Pi}
scale[Csc[x_], x_] := {-2Pi, 2Pi}
scale[Tan[x_], x_] := {-Pi, Pi}
scale[Cot[x_], x_] := {-Pi, Pi}

scale[Sinc[x_], x_] := {-2Pi, 2Pi}

scale[Sinh[x_], x_] := {-2, 2}
scale[Cosh[x_], x_] := {-2, 2}
scale[Sech[x_], x_] := {-2, 2}
scale[Csch[x_], x_] := {-2, 2}
scale[Tanh[x_], x_] := {-2, 2}
scale[Coth[x_], x_] := {-2, 2}

scale[ArcSin[x_], x_] := {-1, 1}
scale[ArcCos[x_], x_] := {-1, 1}
scale[ArcSec[x_], x_] := {-4, 4}
scale[ArcCsc[x_], x_] := {-2, 2}
scale[ArcTan[x_], x_] := {-2, 2}
scale[ArcCot[x_], x_] := {-2, 2}

scale[ArcSinh[x_], x_] := {-2, 2}
scale[ArcCosh[x_], x_] := {0, 5}
scale[ArcSech[x_], x_] := {0, 1}
scale[ArcCsch[x_], x_] := {-2, 2}
scale[ArcTanh[x_], x_] := {-1, 1}
scale[ArcCoth[x_], x_] := {-2, 2}

scale[ArcTan[x_, y_], x_] := {-Pi, Pi}

scale[ArcTan[x_, y_], y_] := {-Pi, Pi}


scale[Exp[x_], x_] := {-3, 3}
scale[Log[x_], x_] := {-3, 3}
scale[Log[a_, x_], x_] := {-3, 3}/Log[Abs[a]]


scale[UnitStep[x_], x_] := {-1, 1}
scale[HeavisideTheta[x_], x_] := {-1, 1}


scale[Exp[a_?NumericQ/x_], x_] := Re[{-a, a}]
scale[Log[x_], x_] := {-3, 3}
scale[Log[a_, x_], x_] := {-3, 3}/Log[Abs[a]]


scale[Abs[x_], x_] := {-2, 2}
scale[AiryAi[x_], x_] := {-11, 5}
scale[AiryAiPrime[x_], x_] := {-11, 5}
scale[AiryBi[x_], x_] := {-11, 2}
scale[AiryBiPrime[x_], x_] := {-11, 2}
scale[ArithmeticGeometricMean[x_, b_], x_] := {-3, 3} /; FreeQ[b, x]
scale[ArithmeticGeometricMean[a_,x_], x_] := {-3, 3} /; FreeQ[a, x]


scale[Erf[x_], x_] := {-2, 2}
scale[Erfc[x_], x_] := {-2, 2}
scale[Erfi[x_], x_] := {-1, 1}


scale[PolyLog[_, x_], x_] := {-2, 2}


scale[Gamma[x_], x_] := {-5/2, 3}
scale[Factorial[x_], x_] := {-5/2, 3} + 1


scale[EllipticK[x_], x_] := {-1, 1}
scale[EllipticE[x_], x_] := {-1, 1}


scale[WeierstrassP[x_, {g2_, g3_}], x_] := {-2, 2} Max[Abs[WeierstrassHalfPeriods[{g2, g3}]]]
scale[WeierstrassPPrime[x_, {g2_, g3_}], x_] := {-2, 2} Max[Abs[WeierstrassHalfPeriods[{g2, g3}]]]


ellipticModuleRange[m_] := 
        Which[m == 0, {-2Pi, 2Pi},
              m == 1, {-2, 2},
              Im[m] == 0., Re @ {-4 EllipticK[m],4 EllipticK[m]},
              True, {-1, 1} Max[Re[EllipticK[m]], Re[EllipticK[1 - m]]]]


scale[EllipticF[x_, m_], x_] := ellipticModuleRange[m]
scale[EllipticE[x_, m_], x_] := ellipticModuleRange[m]


scale[JacobiCN[x_, m_], x_] := ellipticModuleRange[m]
scale[JacobiSN[x_, m_], x_] := ellipticModuleRange[m]
scale[JacobiDN[x_, m_], x_] := ellipticModuleRange[m]
scale[JacobiCD[x_, m_], x_] := ellipticModuleRange[m]
scale[JacobiCS[x_, m_], x_] := ellipticModuleRange[m]
scale[JacobiDC[x_, m_], x_] := ellipticModuleRange[m]
scale[JacobiDS[x_, m_], x_] := ellipticModuleRange[m]
scale[JacobiNC[x_, m_], x_] := ellipticModuleRange[m]
scale[JacobiND[x_, m_], x_] := ellipticModuleRange[m]
scale[JacobiNS[x_, m_], x_] := ellipticModuleRange[m]
scale[JacobiSC[x_, m_], x_] := ellipticModuleRange[m]
scale[JacobiSD[x_, m_], x_] := ellipticModuleRange[m]


ellipticModuleXRange[x_] := 100/(Abs[x]+ 1/200) {-1, 1}


scale[JacobiCN[x_, m_], m_] := ellipticModuleXRange[x]
scale[JacobiSN[x_, m_], m_] := ellipticModuleXRange[x]
scale[JacobiDN[x_, m_], m_] := ellipticModuleXRange[x]
scale[JacobiCD[x_, m_], m_] := ellipticModuleXRange[x]
scale[JacobiCS[x_, m_], m_] := ellipticModuleXRange[x]
scale[JacobiDC[x_, m_], m_] := ellipticModuleXRange[x]
scale[JacobiDS[x_, m_], m_] := ellipticModuleXRange[x]
scale[JacobiNC[x_, m_], m_] := ellipticModuleXRange[x]
scale[JacobiND[x_, m_], m_] := ellipticModuleXRange[x]
scale[JacobiNS[x_, m_], m_] := ellipticModuleXRange[x]
scale[JacobiSC[x_, m_], m_] := ellipticModuleXRange[x]
scale[JacobiSD[x_, m_], m_] := ellipticModuleXRange[x]


scale[Zeta[x_], x_] := {-5, 5}
scale[Zeta[a_ + I x_], x_] := {-20, 20}
scale[Re[Zeta[a_ + I x_]], x_] := {-20, 20}
scale[Im[Zeta[a_ + I x_]], x_] := {-20, 20}


scale[f_. (HeavisideLambda | HeavisidePi | UnitBox | UnitTriangle)[a_. x_ + b_.] /; NumericQ[a] && NumericQ[b] && NumericQ[f], x_] := 
     Re[{-b/a - 2/a, -b/a + 2/a}] 


scale[x_^a_, x_] := {-Log[4, x], Log[4, x]} /; FreeQ[a, x, {0, \[Infinity]}]
scale[x_^a_, a_] := {-2, 2} /; FreeQ[a, x, {0, \[Infinity]}]
scale[x_^x_, x_] := {-1, 2}
scale[x_^(1/x_), x_] := {-1, 3}


scale[_?NumericQ] := {-2, 2}


(*   (1-10^(-10))^((625*x)/11)  *)
scale[ HoldPattern[expr:Times[(_?NumericQ)^(_?NumericQ x_) , (_?NumericQ)^(_?NumericQ x_)  ..]] , x_] := 
Module[{exp},
       exp = Total[N[PowerExpand[Log[List @@ expr]]]];
       1/Abs[Cancel[exp/x]] {-1/4, 1/4} 
        ]


(* ::Text:: *)
(*Special combinations:*)


scale[a_. (Sin | Cos | Tan | Cot | Sec | Csc)[b_. x_ + c_.] + poly_, x_] := 
Module[{min, x0, p0, xs}, 
       min = Cases[Chop[N[x /. Solve[D[poly, x] == 0]]], _Real];
       x0 = If[min === {}, 0, Sort[min, Abs[#1] < Abs[#2]&][[1]]];
       p0 = poly /. x -> x0;
       xs = Union[Chop[N[Re[x] /. Solve[poly == p0 + 2 a || poly == p0 - 2 a]]]];
       If[Length[xs] < 2, {x0 - 2 2Pi/b, x0 + 2 2Pi/b}, {Min[xs], Max[xs]}]
       ]/; PolynomialQ[poly, x] && NumericQ[a] && NumericQ[b] &&  NumericQ[c]


scale[Gamma[_, d_. (a_. x_ + b_.)^n_.] + c_., x_] := 
With[{center = -b/a}, Re[center + {-1, 1}/a/Abs[d^n]]] 


scale[a_. ArcTan[b_. Exp[c_. (x_ + x0_.)^n_. + d_.] + e_.], x_] := 
With[{center = Re[(-(d/c))^(1/n) - x0]}, center + {-2, 2}/Abs[b c]] 


scale[a_. + b_. Mod[c_. x_ + d_., e_] /; NumericQ[a] && NumericQ[b] && NumericQ[c] && NumericQ[d] && NumericQ[e], x_] := 
 Abs[e/c] {-1, 1}


(* ::Text:: *)
(*Special combinations that need both scales:*)


bothScales[a_ + b_ x_^c_ , x_] := 
Module[{center, bound}, 
       center = Re[(-a/b)^(1/c)];
       bound = Abs[b^(1/c)];
       
       (* Fix for bug 293599 *)
       If[Head[bound] =!= Real || bound > $MaxMachineNumber,
       		bound = 100000;
       		If[Abs[center] < 1000, center = 0]
       ];

       {center + bound{-1, 1}, Abs[center] {-2, 2}}
       ]/;  NumericQ[a] && NumericQ[b] && NumericQ[c] && Head[c] === Real


bothScales[a_. + b_. (c_)^(d_. x_^e_.), x_] := 
Module[{center}, 
       center = 0;
       {center +  Abs[1/d^(1/e)] {-1, 1}, Abs[1/d^(1/e)] {-8, 8}}
       ]/;   NumericQ[c] && NumericQ[d] && NumericQ[e] && Re[e] < 0


bothScales[BesselJ[\[Nu]_Integer | \[Nu]_Real, x_], x_] := {{-1, 1} BesselJZero[N[\[Nu]], 4], {-1, 1} BesselJZero[N[\[Nu]], 12]}
bothScales[BesselY[\[Nu]_Integer | \[Nu]_Real, x_], x_] := {{-1, 1} BesselYZero[N[\[Nu]], 4], {-1, 1} BesselYZero[N[\[Nu]], 12]}
bothScales[BesselI[\[Nu]_Integer | \[Nu]_Real, x_], x_] := {{-1, 1}, {-3, 3}} If[\[Nu] == 0, 1, Abs[\[Nu]]]
bothScales[BesselK[\[Nu]_Integer | \[Nu]_Real, x_], x_] := {{1, Max[\[Nu], 1/(Abs[\[Nu]] + 1)] + 2}, {-2, 3 Max[\[Nu], 1/(Abs[\[Nu]] + 1)] + 2}}


bothScales[a_. (Sin | Cos | Sec | Csc)[b_. x_ + c_.] + f_. x_ + d_., x_] := 
Module[{center}, 
       center = Mod[-c/b, 2Pi];
       Re[{center + 2Pi/b {-1, 1}, center + 16Pi/b {-1, 1}}]
       ]/;  NumericQ[a] && NumericQ[b] && NumericQ[c] && NumericQ[d] && NumericQ[f] 


(* ::Input:: *)
(*bothScales[2x + Cos[2x] - 3, x]*)


bothScales[a_. Haversine[b_. (x_ + c_.)], x_] := 
Module[{center}, 
       center = -c;
       Re[{center + 2Pi/b {-1, 1}, center + 16Pi/b {-1, 1}}]
       ]/;  NumericQ[a] && NumericQ[b] && NumericQ[c]


bothScales[(Sin | Cos | Tan | Cot | Sec | Csc)[b_. x_^n_. ] (poly_), x_] := 
Module[{g = Function[A, Abs[(-((-A )/b))^(1./n)]]},  
       {Sort[{-g[-3Pi], g[3Pi]}], Sort[{-g[-8Pi], g[8Pi]}]} 
       ]/; PolynomialQ[1/poly, x] &&  NumericQ[b] &&  IntegerQ[n]


(* ::Input:: *)
(*bothScales[(3 Cos[(2x^2)/3])/(3+2 x^2), x]*)


(* ::Input:: *)
(*bothScales[Sin[2 t^2]/(2 t^2), t]*)


bothScales[a_. Power[b_, c_. x_^x_] /; NumericQ[a] && NumericQ[b] && NumericQ[c], x_] := {{-3, 3}, {-10, 10}} (* for all b, c *)


bothScales[f_. Power[a_, Power[b_, c_. x_]] /; NumericQ[a] && NumericQ[b] && NumericQ[c] && NumericQ[f], x_] :=
With[{X1 = Max[Abs[Re[Log[Log[2.]/Log[a]]/(c*Log[b])]], Abs[Re[Log[Log[-2.]/Log[a]]/(c*Log[b])]]],
      X2 = Max[Abs[Re[Log[Log[10.]/Log[a]]/(c*Log[b])]], Abs[Re[Log[Log[-10.]/Log[a]]/(c*Log[b])]]]},
     If[X2 > 1.2 X1, {X1 {-1, 1}, X2 {-1, 1}},
       {Max[{X1, X2}] {-1, 1}, 2 Max[{X1, X2}] {-1, 1}}]
       ]


bothScales[a_. Power[b_, c_. x_^e_] /; e < 0, x_] := 
With[{f1 = Abs[1./E/c]^(1/e), f2 = Abs[0.1 c]^(1/e)},  {f1 {-1, 1}, f2 {-1, 1}}] /; NumericQ[a] && NumericQ[b] && NumericQ[c]


bothScales[a_. x_^e_Real?Positive, x_] := 
If[e > 1/2, 
   With[{f1 = Min[Max[Abs[1.5/a]^(1/e), 10^6 $MachineEpsilon], 10^10], 
 	    f2 = Min[Max[Abs[6/a]^(1/e), 10^8 $MachineEpsilon], 10^100]},    
        {f1 {-1, 1}, f2 {-1, 1}}],
  With[{f1 = Min[Max[Abs[0.5/a]^(1/e), 10^6 $MachineEpsilon], 10^10], 
 	   f2 = Min[Max[Abs[2/a]^(1/e), 10^8 $MachineEpsilon], 10^100]},    
        {f1 {-1, 1}, f2 {-1, 1}}]
  ] /; NumericQ[a] 


bothScales[f_. (1 + a_. x_^e1_)^e2_ x_^e3_. /; FreeQ[{a, f, e1, e2, e3}, x], x_] := 
With[{S = Abs[(-(1/a))^(1/e1)]}, {1.5 S {-1, 1}, 5 S {-1, 1}}] 


bothScales[a_. (Tan | Cot)[b_. x_ + c_.] + f_. x_ + d_., x_] := 
Module[{center}, 
       center = Mod[-c/b, 2Pi];
       Re[{center + 2 * 2Pi/b {-1, 1}, center + 4 * 2Pi/b {-1, 1}}]
       ]/;  NumericQ[a] && NumericQ[b] && NumericQ[c] && NumericQ[d] && NumericQ[f] 


bothScales[ x_^n_ (Sin | Cos | Tan | Cot | Sec | Csc)[b_. x_ + c_.]  , x_] := 
Module[{trigScale1, trigScale2, polyScale1, polyScale2}, 
       trigScale1 = 2. Pi/b;
       trigScale2 = 12. Pi/b;
       If[(* large n *)
          Abs[trigScale2^n] > 10^100, 
          polyScale1 = 3.^(1/n);
          polyScale2 = 12.^(1/n),
          polyScale1 = trigScale1;
          polyScale2 = trigScale2
         ];
       {Min[trigScale1, polyScale1] {-1, 1}, Min[trigScale2, polyScale2] {-1, 1} } 
       ]/; Im[n] == 0 && n > 0 NumericQ[b] &&  NumericQ[c]


(* ::Input:: *)
(*bothScales[x^2 Sin[x], x]*)


(* ::Input:: *)
(*bothScales[x^2000 Sin[x], x]*)


bothScales[HoldPattern[a_. Power[b_, c_. Power[x_ + d_, e_]/x_]]  /; 
           NumericQ[a] && NumericQ[b] && NumericQ[c] && NumericQ[d] && NumericQ[e], x_]  :=  
            {1/Abs[c] {-1, 1}, Abs[d] {-4, 4}}


bothScales[\[Alpha]_. (a_ + a2_. base_^(c_. (x_ + s_.)^n_ + d_.))^r_Rational  + \[Beta]_. /; FreeQ[{\[Alpha], \[Beta], a, a2, base, c, d, r, s}, x, \[Infinity]] && c < 0, x_] := 
With[{center = -s, scale1 = Abs[(Log[1/3.]/(c*Log[base]))^(1/n)], scale2 = Abs[(Log[1/100.]/(c*Log[base]))^(1/n)]},
     Re[{center + scale1 {-1, 1}, center + scale2 {-1, 1}}]] /; NumericQ[a]


(* ::Input:: *)
(*bothScales[Sqrt[1 - Exp[-x^2]], x]*)


(* ::Input:: *)
(*bothScales[Power[Plus[1,Times[-1,Power[E,Times[-1,Power[x,2]]]]],Rational[1,2]], x]*)


bothScales[Mod[a_. x_ + b_?NumericQ, c_?NumericQ] + d_., x_] := 
With[{center = -b/a},
     Re[{center + {-c, c}, center + 6 {-c, c}}]] /; NumericQ[a]


bothScales[((a_. x_ + b_.)^n_.)! + c_., x_] := 
With[{center = -b/a},
     Re[{center + {-2, 2.5}/a, center + {-5, 5}/a}]] 


bothScales[Log[a_^x_] /; (a > 0 || a < 0), x_] := {{-1, 1}, {-10, 10}}


bothScales[a_. Mod[e_^(c_. x_), b_. x_] /; NumericQ[a] && NumericQ[b] && NumericQ[c] && NumericQ[e], x_] := 
          1. Max[Abs[-(ProductLog[-((c*Log[e])/(4*b))]/(c*Log[e]))], 
                 Abs[-(ProductLog[-1, -((c*Log[e])/(5*b))]/(c*Log[e]))]] {{-1/4, 1/4}, {-1, 1}}


bothScales[a_./( b_ + c_^(d_. (f_. x_ + g_)) ), x_] := 
With[{center = Re[-g/f]},
     {center + {-3, 3}/Abs[d f], center + {-20, 20}/Abs[d f]}] 


bothScales[_. (a_. + b_. x_)/(d_. + e_. Exp[f_. x_]), x_] := 
With[{center = Log[-(d/e)]/f},
     Re[{center + {-5, 5}/f, center + {-50, 50}/f}]] 


bothScales[a_./(b_ + c_. Exp[e_. (x_ + x0_.)^2]) /; (FreeQ[{a, b, c, e}, x, \[Infinity]] && b > 0 && c > 0), x_] := 
With[{center = -x0},
     Re[{center + {-1, 1} Abs[Sqrt[Log[((-1 + 10)*b + 10*c)/c]]/Sqrt[e]], 
      center + {-1, 1} Abs[Sqrt[Log[((-1 + 1000)*b + 1000*c)/c]]/Sqrt[e]]}] // N] 


(* double exponential cases *)
bothScales[Sinh[a_. Cosh[b_. x_ + c_.]], x_] := Re[{-c/b + {-1, 1}, -c/b + {-2, 2}}] /; a > 0 
bothScales[Cosh[a_. Sinh[b_. x_ + c_.]], x_] := Re[{-c/b + {-1, 1}, -c/b + {-2, 2}}] /; a > 0 


bothScales[AiryBi[x_], x_] := {{-11, 2}, {-44, 5}}


bothScales[PrimeZetaP[x_], x_] := {{0.1, 1}, {0.1, 4}}


bothScales[a_. AiryAi[\[Alpha]_. x_]^ea_. + b_. AiryBi[\[Beta]_. x_]^eb_. + _., x_] := 
Module[{sc1}, 
       sc1 = Which[\[Beta] > 0, {-11, 2}/\[Beta], \[Beta] < 0 , {-2, 11}/\[Beta]];
       {sc1, {4, 3} sc1}
       ] /; Im[\[Alpha]] == 0 && Im[\[Beta]] == 0 && Re[eb] > 0


bothScales[DawsonF[a_. x_^r_. + b_.], x_] :=  {{-Abs[(-((b - (-4))/a))^(1/r)], Abs[(-((b - 4)/a))^(1/r)]}, 
                                               {-Abs[(-((b - (-15))/a))^(1/r)], Abs[(-((b - 15)/a))^(1/r)]}}


bothScales[(JacobiCD|JacobiCN|JacobiCS|JacobiDC|JacobiDN|JacobiDS|
            JacobiNC|JacobiND|JacobiNS|JacobiSC|JacobiSD|JacobiSN)[EllipticK[x_^2], x_], x_] := {{-2, 2}, {-10, 10}}


bothScales[PrimePi[x_], x_] := {{0, 12}, {0, 100}}


bothScales[Zeta'[x_], x_] := {{-2, 2}, {-20, 20}}


bothScales[Fibonacci[x_], x_] := {{-3, 3}, {-12, 14}}


bothScales[Fibonacci[a_. x_ + b_.], x_] := 
With[{center = -b/a},
     {center + {-3, 3}/a, center + {-12, 14}/a}] /; Im[a] == 0 && Im[b] == 0

(*Bug 307937: center was being set to a complex number and being passed unmodified through to Plot, which only plots on a range of reals. Looking at other functions where this might be the case.*)
bothScales[Gamma[d_. (a_. x_ + b_.)^n_.] + c_., x_] := 
With[{center = -b/a},
     Re[{center + {-1, 3.5}/a/Abs[d^n], center + {-4, 6}/a/Abs[d^n]}]] 


bothScales[Gamma[_, d_. (a_. x_ + b_.)^n_.] + c_., x_] := 
With[{center = -b/a},
     Re[{center + {-1, 1}/a/Abs[d^n], center + {-3, 3}/a/Abs[d^n]}]] 


bothScales[LerchPhi[_, f_. s_, _], s_] := 
With[{center = 0}, 
     {center + 1/Abs[f] {-2, 2}, center + 1/Abs[f] {-8, 8}}] /; NumericQ[f]


(* Integrate[x^a Exp[x^b], x] *) 
bothScales[f_?(NumericQ[PowerExpand[#]]&) Gamma[_, a_. x_^b_] , x_] := 
With[{center = 0},
     {center + {-1, 1}, center + Max[2, Abs[3^(1/b)]]{-1, 1}}]  


bothScales[f_./(a_ + b_. base_^(c_. x_)) /; Im[a] == 0 && Im[b] == 0 && Im[c] == 0 && Im[base] == 0, x_] := 
Which[Sign[a] === Sign[b],  {Abs[Log[2.]/c Log[base]] {-1, 1}, Abs[Log[8.]/c Log[base]] {-1, 1}},
      True, Max[Abs[Log[-a/b]/c], 1] {{-0.5, 0.5}, 2. {-1, 1}}
       ]


(* ::Input:: *)
(*bothScales[44/(1 + 3 E^(-0.22` x)), x]*)


(* ::Input:: *)
(*bothScales[1/(4 - 2^x), x]*)


(* ::Input:: *)
(*bothScales[1/(1 + 2^x), x]*)


bothScales[f_./(a_ + b_. base_^(c_. x_)) + _. x_^_. /; Im[a] == 0 && Im[b] == 0 && Im[c] == 0 && Im[base] == 0, x_] := 
Which[Sign[a] === Sign[b],  {Abs[Log[2.]/c Log[base]] {-1, 1}, Abs[Log[8.]/c Log[base]] {-1, 1}},
      True, Max[Abs[Log[-a/b]/c], 1] {{-0.5, 0.5}, 5. {-1, 1}}
       ]


(* ::Input:: *)
(*bothScales[1/(-1+E^x)-1/x, x]*)


(* ::Input:: *)
(*bothScales[1/(-1+E^x)+1/x, x]*)


(* ::Input:: *)
(*bothScales[1/(-1+E^c)+1/c, c]*)


(* ::Input:: *)
(*bothScales[1/c + 1/(-1+E^c ), c]*)


bothScales[f_. ((a_. (x_ + s1_.)!)^(b_. (x_ + s2_.)!)) , x_] := 
With[{center = Re[s1]},
     {center + {-2, 2}/Abs[a], center + {-6, 6}/Abs[a]}]  /; NumericQ[f] && NumericQ[a] && NumericQ[b] && NumericQ[s1] && NumericQ[s2]


bothScales[a_. (SinIntegral | CosIntegral)[b_. + c_. Exp[x_]], x_] :=  {{-2, 2}/c, {-5, 5}/c}


bothScales[_. (b1_ + b2_. c_^(d_. (e_. (x_ + x0_.) + f_.)^g_))^a_, x_] :=  
Module[{center = Re[(-f - e x0)/e],  width = Abs[((1/d)^(1/g) - f)/e]}, 
        If[width == 0, width = Max[Abs[((1/d)^(1/g))/e], Abs[(f)/e]]]; 
        Re[{{center - width, center + width}, {center - 8 width, center + 8 width}}]
      ] /; a < 0 && g < 0


(* ::Input:: *)
(*bothScales[(1+E^(1+x)^(-1))^(-1), x]*)


bothScales[a_. (BesselJ[n_, b_. x_] BesselY[n_, c_. x_] - BesselJ[n_, c_. x_] BesselY[n_, b_. x_] ) /; 
           NumericQ[a] && NumericQ[b] && NumericQ[c], x_] :=  
          {{-4, 4} 2Pi, {-8, 8} 2Pi} /Max[Abs[{b, c}]]


bothScales[f_. Log[a_ + b_. Exp[c_. x_^n_.]] /;  
           NumericQ[a] && NumericQ[b] && NumericQ[c] && NumericQ[f] && NumericQ[n], x_] :=
          {{-1, 1}, {-3, 3}} * Abs[(Log[-((a - 6.)/b)]/c)^(1/n)]


bothScales[_. poly1_/(a_. Exp[poly2_] + b_.), x_] :=  
With[{roots1 = x /. Union[{ToRules[Reduce[N[ poly1 == 0], x, Reals]]}],
      roots2 = If[b == 0, {},  x /. Union[{ToRules[Reduce[N[ poly2 == Log[-b/a]], x, Reals]]}]]}, 
      Module[{min2, max2, center2, \[CapitalDelta]2, allRoots, min, max, center, \[CapitalDelta]} ,  
             {min2, max2} = {Min[roots2], Max[roots2]};
             center2 = Mean[{min2, max2}];
             \[CapitalDelta]2 = (max2 - min2)/2;
             allRoots = Union[Flatten[{roots1, roots2}]];
             {min, max} = {Min[allRoots], Max[allRoots]};
             center = Mean[{min, max}];
             \[CapitalDelta] = (max - min)/2;
             Which[roots1 === {},  {center2 + 3/2 {-\[CapitalDelta]2, \[CapitalDelta]2}, center2 + 5 {-\[CapitalDelta]2, \[CapitalDelta]2}},
                   True, {center2 + 3/2 {-\[CapitalDelta]2, \[CapitalDelta]2}, center + 1.5 {-\[CapitalDelta], \[CapitalDelta]}}]  
            ]
      ] /; PolynomialQ[poly1, x] && PolynomialQ[poly2, x] && FreeQ[a, x, {0, \[Infinity]}] && FreeQ[b, x, {0, \[Infinity]}]  &&
           Length[Union @ {ToRules[Reduce[N[ poly2 == 0], x, Reals]]}] >= 2


bothScales[f_. BesselJZero[a_. x_ + b_., _Integer | _Rational | _Real] /;  NumericQ[a] && NumericQ[b] && NumericQ[f], x_] :=  
          {5 {-b/a, b/a}, 10 {-b/a, b/a}}


bothScales[f_. HypergeometricPFQ[l1_, l2_, z_] /; NumericQ[f], x_] /; MemberQ[{l1, l2}, x, \[Infinity]] := {{-1, 1}, {-5, 5}}


bothScales[f_. HypergeometricPFQRegularized[l1_, l2_, z_] /; NumericQ[f], x_] /; MemberQ[{l1, l2}, x, \[Infinity]] := {{-1, 1}, {-5, 5}}


bothScales[f_. IteratedLog[a_. x_ + b_.] /;   NumericQ[a] && NumericQ[b] && NumericQ[f], x_] :=  
          Re[{{-b/a, (1.5 E^E - b)/a}, {-b/a, (1.5 E^E^E - b)/a}}]


bothScales[f_. IteratedBinaryLog[a_. x_ + b_.] /;   NumericQ[a] && NumericQ[b] && NumericQ[f], x_] :=  
          Re[{{-b/a, (1.5 2^2 - b)/a}, {-b/a, (1.5 2^2^2 - b)/a}}]


bothScales[f_. DickmanRho[a_. x_ + b_.] /;   NumericQ[a] && NumericQ[b] && NumericQ[f], x_] :=
          Re[{{-b/a, (3 - b)/a}, {-b/a, (10 - b)/a}}] 


bothScales[f_. Logit[a_. x_ + b_.] /;   NumericQ[a] && NumericQ[b] && NumericQ[f], x_] :=
          Re[{{-b/a, (1 - b)/a}, {-b/a - 2, (1 - b)/a + 2}}]


bothScales[f_. FaddeevaFunction[a_. x_ + b_.] /;   NumericQ[a] && NumericQ[b] && NumericQ[f], x_] :=
          Re[{ {-b/a - 3, -b/a + 3}, {-b/a - 12, -b/a + 12}}]


bothScales[f_. VoigtV[a_. x_ + b_., \[Gamma]_] /;   NumericQ[a] && NumericQ[b] && NumericQ[f], x_] :=
         Re[{ {-b/a - 3 \[Gamma], -b/a + 3 \[Gamma]}, {-b/a - 12 \[Gamma], -b/a + 12 \[Gamma]}}] 


bothScales[f_. PseudoVoigtV[a_. x_ + b_., \[Gamma]_] /;   NumericQ[a] && NumericQ[b] && NumericQ[f], x_] :=
          Re[{ {-b/a - 3 \[Gamma], -b/a + 3 \[Gamma]}, {-b/a - 12 \[Gamma], -b/a + 12 \[Gamma]}}]


bothScales[f_. ThomaeFunction[a_. x_ + b_.] /;   NumericQ[a] && NumericQ[b] && NumericQ[f], x_] :=
          Re[{{-b/a, (1 - b)/a}, {-b/a - 2, (1 - b)/a + 2}}] 


bothScales[f_. RieszR[a_. x_ + b_.] /; NumericQ[a] && NumericQ[b] && NumericQ[f], x_] :=
          Re[{{-b/a, -b/a + 5}, {-b/a, -b/a + 100}}] 


bothScales[f_. TakagiT[a_. x_ + b_.] /; NumericQ[a] && NumericQ[b] && NumericQ[f], x_] :=
          Re[{{-b/a + 0.399, -b/a + 0.401}, {-b/a, -b/a + 1}}] 


bothScales[f_. UehlingPotential[a_. x_ + b_.] /; NumericQ[a] && NumericQ[b] && NumericQ[f], x_] :=
          {{1, 2}, {1/2, 4}} 


bothScales[f_. RogersRamanujanR[a_. q_ + b_.] /; NumericQ[a] && NumericQ[b] && NumericQ[f], q_] :=
          With[{c = -b/a}, {{c - 1/4, c + 1/4}, {c - 3/4, c + 3/4}}] 
          
bothScales[f_. RamanujanCubicContinuedFractionV[a_. q_ + b_.] /; NumericQ[a] && NumericQ[b] && NumericQ[f], q_] :=
          With[{c = -b/a}, {{c - 1/4, c + 1/4}, {c - 3/4, c + 3/4}}]


With[{R = Evaluate[RogersRamanujanR[#]]&},
bothScales[f_. R[a_. q_ + b_.] /; NumericQ[a] && NumericQ[b] && NumericQ[f], q_] :=
          With[{c = -b/a}, Re[{{c - 1/8, c + 1/8}, {c - 99/100, c + 99/100}}]]  
]


With[{R = Evaluate[RogersRamanujanS[#]]&},
bothScales[f_. R[a_. q_ + b_.] /; NumericQ[a] && NumericQ[b] && NumericQ[f], q_] :=
          With[{c = -b/a}, Re[{{c - 1/8, c + 1/8}, {c - 99/100, c + 99/100}}]]  
]


With[{R = Evaluate[RogersRamanujanG[#]]&},
bothScales[f_. R[a_. q_ + b_.] /; NumericQ[a] && NumericQ[b] && NumericQ[f], q_] :=
          With[{c = -b/a}, Re[{{c - 1/8, c + 1/8}, {c - 99/100, c + 99/100}}]]  
]


With[{R = Evaluate[RogersRamanujanH[#]]&},
bothScales[f_. R[a_. q_ + b_.] /; NumericQ[a] && NumericQ[b] && NumericQ[f], q_] :=
          With[{c = -b/a}, Re[{{c - 1/8, c + 1/8}, {c - 99/100, c + 99/100}}]]  
]


bothScales[__ QPochhammer[(q_)^_., _], q_] :=
          With[{c = 0}, Re[{{c - 1/8, c + 1/8}, {c - 99/100, c + 99/100}}]]  


bothScales[__ QPochhammer[(q_)^_., _]^_., q_] :=
          With[{c = 0}, Re[{{c - 1/8, c + 1/8}, {c - 99/100, c + 99/100}}]]  


(* ::Input:: *)
(*bothScales[QPochhammer[q,q^5]*QPochhammer[q^4,q^5], q]*)


(* ::Text:: *)
(*Scale arithmetic:*)


scale[(f:Except[Plus | Times])[Longest[a___], poly_, b___], x_] := 
         innerPolyScale[f[a, x, b], poly, x]  /;
                          (poly =!= x &&
                           Head[scale[f[a, x, b], x]] === List &&
                           MemberQ[poly, x, {0, \[Infinity]}] && PolynomialQ[poly, x])


scale[(f:Except[Plus | Times])[a___, x_^\[Alpha]_ + \[Beta]_., b___], x_] := 
         innerPowerScale[f[a, x, b], x^\[Alpha] + \[Beta], x]  /;
                          (Head[scale[f[a, x, b], x]] === List &&
                           FreeQ[\[Alpha], x, {0, \[Infinity]}] && FreeQ[\[Beta], x, {0, \[Infinity]}] )


scale[((f:Except[Plus | Times])[a___, poly_, b___])^e_Integer, x_] := 
      scale[f[a, poly, b], x] /; Head[scale[f[a, x, b],x]]===List


(* special evaluation cases *)
scale[HoldPattern[Power[E, poly_]], x_] := innerPolyScale[Exp[x], poly, x]  /;
                           poly =!= x && MemberQ[poly, x, {0, \[Infinity]}] && PolynomialQ[poly, x]


intervalSolve[poly_, {y1_, y2_}, x_] := 
Module[{roots},
       roots = Flatten[{realRoots[poly - y1, x], realRoots[poly - y2, x]}];
       Which[Length[roots] === 2, {roots},
             Length[roots] === 1, {{roots[[1]] -1, roots[[1]] + 1}},
             Length[roots] === 0, (* a hack for now *) {{-1, +1}},
             Length[roots] > 2, Sort[#, Abs[Mean[#1]] < Abs[Mean[#2]]&]& @ 
                   Select[Partition[roots, 2, 1],  (y1 < Mean[poly /. x -> #1] < y2)&]
             ]
       ] /; PolynomialQ[poly, x]


intervalPowerSolve[x_^\[Alpha]_ + \[Beta]_., {y1_, y2_}, x_] := 
Module[{roots},
       roots = Select[x /. Solve[x^\[Alpha] + \[Beta] == y1 || x^\[Alpha] + \[Beta] == y2 , x], Im[#] == 0&];
       If[Length[roots] > 0, {{Min[{0, roots}], Max[roots]}}, $Failed]
       ]


innerPolyScale[f_, p_, x_] := 
Module[{},
       outerScale = scale[f, x]; 
       If[Head[outerScale] === List && PolynomialQ[p, x],
          innerIntervals = intervalSolve[p, outerScale, x];
          innerIntervals[[1]],
          $Failed]
          ]


innerPowerScale[f_, p_, x_] := 
Module[{},
       outerScale = scale[f, x]; 
       If[Head[outerScale] === List,
          innerIntervals = intervalPowerSolve[p, outerScale, x];
          innerIntervals[[1]],
          $Failed]
          ]


scale[p_Plus, x_] := unitScales[scale[#, x]& /@ (List @@ p)]
scale[t_Times, x_] := unitScales[scale[#, x]& /@ (List @@ t)]


scale[f_?NumericQ fun_, x_] := scale[fun, x]


scale[Power[e_?(# > 3/2&), a_. x_^n_Integer] p_, x_] :=
Module[{saux, avV, xM, res, specialEnlargementFactor},
       (saux = scale[p, x]; 
        avV = Max[Table[Abs[p /. x -> RandomReal[saux]], {24}]];
        xM = Abs[(Log[(* use a reasonable factor *) 6 avV]/a Log[e])^(1/n)];
        res = Which[Min[saux] < xM <Max[xM], saux,
                  a < 0 && OddQ[n], {Max[Min[saux], -xM], Max[saux]},
                  a > 0 && OddQ[n], {Min[saux], Min[Max[saux], xM]},
                  a < 0 && EvenQ[n], saux,
                  a > 0 && EvenQ[n], {Max[Min[saux], xM], Min[Max[saux], xM]}];
        specialEnlargementFactor = Max[Abs[Abs[(Log[100 avV]/a Log[e])^(1/n)]/xM] - 1, 0.1]; 
        res /;  Head[saux] === List)] /; Im[a] == 0 && n >= 3 


(* fix for bug 70511 *)
scale[x_^f_, x_] :=
Module[{fScale = scale[f, x]},
      fScale /; Head[fScale] =!= scale
      ]


unitScales[scales_] := 
 Module[{sortedScales},
         sortedScales = Sort /@ N[Cases[scales, {_?NumericQ, _?NumericQ}]];
         If[sortedScales === {}, $Failed, 
            If[(IntervalIntersection @@ (Interval /@ sortedScales)) =!= 
               Interval[],
               Sort[{Mean[#1], Mean[#2]} & @@ Transpose[sortedScales]],
               {Min[#], Max[#]} &[ Transpose[sortedScales] ]
                ]
            ]
        ]


Clear[scaleCache];
scaleCache[expr_, x_] := scaleCache[expr, x] = scale[expr, x]


TableLookUpQ[expr_, x_] := Not[TrueQ[AlgebraicQ[expr, x]]] &&
                           TimeConstrained[Head[bothScales[expr, x]] === List || Head[scaleCache[expr, x]] === List, 0.3, False]


PRF["TableLookUp", expr_, x_] := 
Module[{aux, aux2, factor}, 
       (* special asymmetric cases *)
       aux2 = bothScales[expr, x];
       If[MatchQ[aux2, {{_?NumericQ, _?NumericQ}, {_?NumericQ, _?NumericQ}}], 
          {PRData[{aux2[[1]], {}}, "HorizontalPlotRangeType" -> "ShowCenterPart"],
           PRData[{aux2[[2]], {}}, "HorizontalPlotRangeType" -> "ShowEnlargedMore"]}, 
        (* generic cases *)
        aux = scaleCache[expr, x];
        If[Head[aux] === List &&( And @@ (NumericQ /@ aux)), 
           If[ValueQ[specialEnlargementFactor], 
              factor = specialEnlargementFactor; Clear[specialEnlargementFactor], 
              factor = 2.5];  
             {PRData[{aux, {}}, "HorizontalPlotRangeType" -> "ShowCenterPart"],
              PRData[{enlargePlotRange[{aux, 0}, factor] /. {{a_, b_}, {}} :> {a, b},  {}}, 
                      "HorizontalPlotRangeType" -> "ShowEnlargedMore"]},
             $Failed]]]


(* ::Input:: *)
(*PRF["TableLookUp",(1/c) - (1/((E^c) - 1)), c]*)


(* ::Input:: *)
(*PRF["TableLookUp",44/(1+3 E^(-0.22` x)),  x]*)


(* ::Input:: *)
(*PRF["TableLookUp",Log[1+E^-x^2],  x]*)


(* ::Input:: *)
(*PRF["TableLookUp",Sqrt[1 - Exp[-x^2]],  x]*)


(* ::Input:: *)
(*PRF["TableLookUp",E^(Sqrt[x-1]/x),  x]*)


(* ::Input:: *)
(*PRF["TableLookUp",BesselJ[1,8333.3` x] BesselY[1,x]-BesselJ[1,x] BesselY[1,8333.3` x],  x]*)


(* ::Input:: *)
(*PRF["TableLookUp",x + Cos[3 x] + 1,  x]*)


(* ::Input:: *)
(*PRF["TableLookUp",-1 + E^x - x, x]*)


(* ::Input:: *)
(*PRF["TableLookUp",Fibonacci[x], x]*)


(* ::Input:: *)
(*PRF["TableLookUp",-((x^3*Gamma[3/4,-x^4])/(4*(-x^4)^(3/4))), x]*)


(* ::Input:: *)
(*PRF["TableLookUp",Sin[1/x], x]*)


(* ::Input:: *)
(*PRF["TableLookUp",(1/(-1+E^x)-1/x), x]*)


(* ::Input:: *)
(*findMatchingRule[expr_, x_] := *)
(*Select[DownValues[scale] /. scale -> scaleDebug,*)
(*                ( MatchQ[scaleDebug[expr, x], #[[1]]] )&] /.scaleDebug -> scale*)


(* ::Input:: *)
(*findMatchingRule[1/(-1+E^x)-1/x, x]*)


(* ::Input:: *)
(*findMatchingRule[(1+E^(1+x)^(-1))^(-1), x]*)


(* ::Input:: *)
(*findMatchingRule[E^(Sqrt[x-1]/x), x]*)


(* ::Input:: *)
(*findMatchingRule[BesselJ[1,8333.3` x] BesselY[1,x]-BesselJ[1,x] BesselY[1,8333.3` x], x]*)


(* ::Input:: *)
(*findMatchingRule[Exp[Sqrt[x]], x]*)


(* ::Input:: *)
(*findMatchingRule[Exp[x], x]*)


(* ::Input:: *)
(*findMatchingRule[Exp[-x^2], x]*)


(* ::Input:: *)
(*findMatchingBothScalesRule[expr_, x_] := *)
(*Select[DownValues[bothScales] /. bothScales -> scaleDebug,*)
(*                ( MatchQ[scaleDebug[expr, x], #[[1]]] )&] /.scaleDebug -> bothScales*)


(* ::Input:: *)
(*findMatchingBothScalesRule[(1/c) - (1/(E^c - 1)), c]*)


(* ::Input:: *)
(*findMatchingBothScalesRule[(1+E^(1+x)^(-1))^(-1), x]*)


(* ::Input:: *)
(*findMatchingBothScalesRule[E^(Sqrt[x-1]/x), x]*)


(* ::Input:: *)
(*findMatchingBothScalesRule[Exp[Sqrt[x]], x]*)


(* ::Input:: *)
(*findMatchingBothScalesRule[Exp[x], x]*)


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Rational trigonometric   *)


Clear[RationalTrigonometricQ];
RationalTrigonometricQ[expr_Plus, x_] := 
         And @@ (RationalTrigonometricQ[#, x]& /@ (List @@ expr))

RationalTrigonometricQ[a_/b_, x_] := RationalTrigonometricQ[a, x] && 
                                     RationalTrigonometricQ[b, x]

RationalTrigonometricQ[a_ b_, x_] := RationalTrigonometricQ[a, x] && 
                                     RationalTrigonometricQ[b, x]

RationalTrigonometricQ[Power[a_, b_Integer], x_] := RationalTrigonometricQ[a, x]

RationalTrigonometricQ[_?NumericQ, x_] := True

RationalTrigonometricQ[(head:(Sin | Cos | Tan | Cot | Sec | Csc))[arg_], x_] :=
                       PolynomialQ[arg, x]


guessedCenter[f_, \[CapitalDelta]_, x_] := 
Module[{},
ntf = Numerator[Together[#]]&;
CalculateTimeConstrained[
     If[NumericQ[f + \[CapitalDelta]], {},
      roots = realRoots[ntf[f + \[CapitalDelta]], x];
      If[roots =!= {}, roots,
         dRoots = realRoots[D[ntf[f], x] + \[CapitalDelta], x]]], 0.1, {}]]


(* ::Input:: *)
(*guessedCenter[42+x^2, + 2 Pi, x]*)


PRF["RationalTrigonometric", expr_, x_] :=
Module[{(*verticalRangeFactor = 1/3, *) horizontalRangeFactor = 0.02 (*, tMax = 0.1 *)},
       theOccurringTrigs = Cases[expr, _Cos | _Sin | _Tan | _Cot | _Csc | _Sec, {0, Infinity}];
       theTrigArguments = Select[Union[First /@ theOccurringTrigs], MemberQ[#, x, {0, \[Infinity]}]&];
       (* special, but relevant linear case *)
       prs = 
         Function[R, 
         TimeConstrained[
             If[(And @@ (PolynomialQ[#, x]& /@ theTrigArguments)) && Max[Exponent[theTrigArguments, x]] === 1,
                absCoeffs = Abs[Coefficient[theTrigArguments, x, 1]];
                minAbsC = Which[Length[absCoeffs] == 1, R 2 Pi {1, -1}/absCoeffs[[1]],
                                Max[absCoeffs]/Min[absCoeffs] <= 3, R 2 Pi {1, -1}/Min[absCoeffs], 
                                True, R 2 Pi {1, -1} Max[absCoeffs]
                             ];
                padRange[minAbsC, horizontalRangeFactor],

                guessedIntervals = {guessedCenter[#, -R 2Pi, x], guessedCenter[#, 0, x],  guessedCenter[#, R 2Pi, x]}& /@ theTrigArguments;
                averageCenter = Mean[Flatten[guessedIntervals]]; 
                averageRangeLength = Mean[Abs[Max[#] - Min[#]]& /@ guessedIntervals];
                (* use symmetry *)
                If[expr === (expr /. x -> -x) || expr === -(expr /. x -> -x) && Abs[averageCenter] < 2 averageRangeLength , averageCenter = 0.];
                padRange[{averageCenter - averageRangeLength/2, 
                          averageCenter + averageRangeLength/2}, horizontalRangeFactor] 
                 ], 0.6, {}]] /@ (* horizontal periods goal *) {1, 3};

       {If[prs[[1]] === {}, {},
           PRData[{prs[[1]], {}}, "HorizontalPlotRangeType" -> "ShowFewPeriods"]],
        If[prs[[2]] === {}, {},
           PRData[{prs[[2]], {}}, "HorizontalPlotRangeType" -> "ShowMorePeriods"]]}
       
       ]


(* ::Input:: *)
(*PRF["RationalTrigonometric", Sin[x^4],x]*)


(* ::Input:: *)
(*PRF["RationalTrigonometric", Sec[1/2]*Sin[x^2]+Tan[x^2],x]*)


(* ::Input:: *)
(*PRF["RationalTrigonometric", Sin[42 + x^2],x]*)


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Periodic   *)


Clear[PeriodicQ];
PeriodicQ[expr_, x_] := (periodInX[expr, x] =!=$Failed)


(* optimization *)
PeriodicQ[f_ (_Sin | _Cos | _Tan | _Cot | _Sec | _Csc)[a_. x_ + b_. ], x_] := False /; MemberQ[f, x, {0, \[Infinity]}] && PolynomialQ[f, x]


Clear[periodicTerms];
periodicTerms[expr_, x_, bag_:{}] := 
Module[{},
       parts = Cases[expr, _Sin | _Cos | _Tan | _Cot | _Sec | _Csc, {0, Infinity}];
       args = First /@ parts;
       linearParts = Cases[args, _?(NumericQ) x | x];
       rest = Complement[args, linearParts];
       {bag, linearParts, periodicTerms[#, x, linearParts]& /@ rest}
     ]


(* ::Input:: *)
(*periodicTerms[Sin[3 x], x]*)


(* ::Input:: *)
(*periodicTerms[Abs[Cos[x-Pi/2]] (UnitStep[x+Pi]-UnitStep[x-Pi]), x]*)


Clear[periodInX];
periodInX[expr_, x_] := periodInX[expr, x] =
Module[{},
       periodicArgs = Flatten[periodicTerms[expr, x] /. x -> 1];
       If[periodicArgs === {}, $Failed, 
          conjecturedPeriod = 2Pi (PolynomialLCM @@ Denominator[periodicArgs]);
          valN1 = expr /. x -> 1./(Pi + 2 E); 
          valN2 = (expr /. x -> x + conjecturedPeriod) /. x -> 1./(Pi + 2 E); 
          If[Quiet[TrueQ[1 - Abs[valN1]/Abs[valN2] < 10^-10]], 
             \[Delta] = Simplify[ExpandAll[expr - (expr /. x -> x + conjecturedPeriod)], Element[x, Reals]];
             If[\[Delta] === 0, conjecturedPeriod, $Failed],
             $Failed]
         ]
      ]


PRF["Periodic", expr_, x_] :=
Module[{(* verticalRangeFactor = 1/3, horizontalRangeFactor = 0.02, tMax = 0.1*)},
       xPeriod = periodInX[expr, x];
       If[xPeriod === $Failed, $Failed, pr0 = {-xPeriod, xPeriod};
       {PRData[ {pr0, {}}, "HorizontalPlotRangeType" -> "ShowFewPeriods"],
        PRData[ enlargePlotRange[{pr0, {}}, 2] , "HorizontalPlotRangeType" -> "ShowMorePeriods"]}]
       ]


(* ::Input:: *)
(*PRF["Periodic", Abs[Cos[x-Pi/2]] (UnitStep[x+Pi]-UnitStep[x-Pi]), x]*)


(* ::Input:: *)
(*PRF["Periodic", x^20 Sin[x], x]*)


(* ::Input:: *)
(*PRF["Periodic", Sin[3x], x]*)


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Suggest other plot type   *)


Clear[SuggestOtherPlotTypeQ];
SuggestOtherPlotTypeQ[expr_, x_] := suggestOtherPlotCache[expr, x] =!= $Failed


solveLogRescaledODEFromInfinity[expr_, x_, sign_] := 
Module[{}, f1 = PowerExpand[1/Log[Log[expr /. x -> sign 1/x]]];
              nds = NDSolve[{F'[x] == If[Im[#] == 0, #, "NaN"]&[f1], 
                             F[0] == (f1/. x -> sign 10^-3)} , F, 
                             {x, sign 10^-3,  sign 2},
                             PrecisionGoal -> 3, MaxSteps -> 1200];
             If[MemberQ[nds, _InterpolatingFunction, Infinity],
                1/nds[[1, 1, 2, 1, 1]], $Failed]
       ]


suggestOtherPlotCache[expr_, x_] := 
Module[{},
       aux = TimeConstrained[
             Which[TrueQ[Limit[Log[Log[expr]]/x, x -> +Infinity] > 0],
                   solveLogRescaledODEFromInfinity[expr, x, +1],
                   TrueQ[Limit[Log[Log[expr]]/x, x -> -Infinity] > 0],
                   solveLogRescaledODEFromInfinity[expr, x, -1],
                   True, $Failed], 0.3];
      If[aux === $Aborted, $Failed, aux] 
      ]


PRF["SuggestOtherPlotType", expr_, x_] :=
    {{suggestOtherPlotCache[expr, x], {}}, "SuggestedPlotType" -> LogPlot}


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Trigonometrics of rationals   *)


Clear[TrigonometricsOfRationalQ];
TrigonometricsOfRationalQ[expr_, x_] := 
With[{expr1 = expr /. Exp[Complex[0 | 0., _] f_] :> Cos[f] + I Sin[f]},
If[# === {}, False, (And @@
  (TrueQ[RationalFunctionQ[First[#], x]]& /@ #)) &&
     FreeQ[expr1 /.  (_Sin | _Cos | _Tan | _Cot | _Sec | _Csc ) :> Random[],
            x, {0, Infinity}]]&[ 
     Cases[expr1, _Sin | _Cos | _Tan | _Cot | _Sec | _Csc, {0, Infinity}]
                     ]
]


(* ::Input:: *)
(*TrigonometricsOfRationalQ[Exp[I/x^2], x]*)


Clear[goodTrigInterval];
goodTrigInterval[ratf_, x_, R_] := 
Module[{tMax = 0.3},
       ratF[y_] = (ratf /. x -> y);
       tog = TimeConstrained[Factor @ Together[ratf], tMax];
       If[tog === $Aborted, {padRange[{-1, 1}, 0.01], {}},
          (* should be able to differentiate and together *)
          {num, den} = {Numerator[#], Denominator[#]}&[tog];
          {numDeg, denDeg} = Exponent[{num, den}, x];
          zeros = realRoots[num, x];
          poles = realRoots[den, x];
          d1Num = Numerator[Together[D[ratf, x]]];
          d2Num = Numerator[Together[D[ratf, x, x]]];
          extremas = realRoots[d1Num, x];
          inflections = realRoots[d2Num, x];
          {maxRightStructures, minLeftStructures} = 
             {Max[#], Min[#]}&[{zeros, poles, extremas, inflections}];
          rightAsymptoticValue = Limit[tog, x -> +Infinity];
          leftAsymptoticValue = Limit[tog, x -> -Infinity];
          CalculateTimeConstrained[
          Which[(* 1/x and friends *) 
                NumericQ[rightAsymptoticValue] && 
                extremas === {} && inflections === {} && poles =!= {}, 
                If[debugQ, Print["trig of rational 1"]];
                aux = {realRoots[num - (rightAsymptoticValue + 2 Pi) den, x], 
                       realRoots[num - (rightAsymptoticValue + R 6 Pi) den, x]}; 
                Sort @ Flatten @ 
                If[Min[aux] > maxRightStructures, {Min[aux], Max[aux]},
                 {Min[#], Max[#]}& @ 
                   {realRoots[num - (rightAsymptoticValue - 1/R 2 Pi) den, x],
                    realRoots[num + (rightAsymptoticValue - 1/R 2 Pi) den, x],
                    realRoots[num - (rightAsymptoticValue - R 6 Pi) den, x],
                    realRoots[num + (rightAsymptoticValue - R 6 Pi) den, x]}],
                (* 1/(x^2 + 1) and friends *)
                NumericQ[rightAsymptoticValue] && extremas =!= {} && poles === {},
                If[debugQ, Print["trig of rational 2"]];
                iLength = Max[inflections] - Min[inflections];
                {Min[inflections] - R iLength/4, Max[inflections] + R iLength/4},
                (* 1/(x^2 - 1) and friends *)
                Length[poles] >= 2,
                If[debugQ, Print["trig of rational 3"]];
                poleRanges = Partition[Sort[poles], 2, 1];
                poleRangesAndLengths = {Abs[Subtract @@ #], #}& /@ poleRanges;
                maxPoleIntervalLength = Max[First /@ poleRangesAndLengths];
                maxPoleInterval = (Last /@ 
                        Cases[poleRangesAndLengths, {maxPoleIntervalLength, _}])[[1]];
                poleIntervalExtremas = Select[extremas, 
                                        IntervalMemberQ[Interval[maxPoleInterval], #]&];
                poleIntervalReferenceValue = 
                     If[poleIntervalExtremas ==={}, 0, Max[ratF /@ poleIntervalExtremas]];
                rootPoints = Flatten @ 
                             {realRoots[num - (poleIntervalReferenceValue - R 6 Pi) den, x],
                              realRoots[num - (poleIntervalReferenceValue + R 6 Pi) den, x]};
                selRootPoints = Select[rootPoints, 
                                       IntervalMemberQ[Interval[maxPoleInterval], #]&];
                {Min[selRootPoints], Max[selRootPoints]}, 
                (* 1/x + polynomial friends *)
                Length[poles] === 1 && Length[extremas] + Length[inflections] > 0,
                If[debugQ, Print["trig of rational 4"]];
                poleIntervalReferenceValue = Max[ratF /@ Join[extremas, inflections]];
                rootPoints = Flatten @ 
                             {realRoots[num - (poleIntervalReferenceValue - R 6 Pi) den, x],
                              realRoots[num - (poleIntervalReferenceValue + R 6 Pi) den, x]};
                selIntervals = Select[Partition[Sort[rootPoints], 2, 1],
                                      Not[IntervalMemberQ[Interval[#], poles[[1]]]]&];
                If[selIntervals =!= {},
                   Sort[{Mean[#], #}& /@ selIntervals, Abs[#1[[1]]] < Abs[#2[[1]]]&][[1, 2]],
                   selIntervals2 = Select[Partition[Sort[Flatten[{rootPoints, extremas}]], 2, 1],
                                      Not[IntervalMemberQ[Interval[#], poles[[1]]]]&];
                   Sort[{Mean[#], #}& /@ selIntervals2, Abs[#1[[1]]] < Abs[#2[[1]]]&][[1, 2]]
                   ],
                True, print["Unexpected rational argument"]; R {-2Pi, 2Pi}
               ], 0.5, {}]
         ]
       ]


(* ::Input:: *)
(*goodTrigInterval[1/x^2, x,1]*)


PRF["TrigonometricsOfRational", expr_, x_] :=
With[{expr1 = expr /. Exp[Complex[0 | 0., ai_] f_] :> Cos[ai f] + I Sin[ai f]},
Module[{prs, individualIntervals, rationalTrigArguments, is, aux1, sel,
        funP, length1, length2, mp},   
       rationalTrigArguments = First /@ 
           Cases[expr1, (Sin | Cos | Tan | Cot | Sec | Csc | Sinc)[
                        arg_?(RationalFunctionQ[#, x]&)] , {0, Infinity}];  
      prs = Function[R,  TimeConstrained[
                          individualIntervals = goodTrigInterval[#, x, R]& /@ Union[rationalTrigArguments];   
      Which[Length[individualIntervals] === 1, 
            {individualIntervals[[1]], {}},
            is = IntervalIntersection @@ (Interval /@ individualIntervals);
            is === Interval[], {individualIntervals[[1]], {}},
            (* more than one *) 
            funP = Function @@ {x, Evaluate[{x, rationalTrigArguments}]};  
            aux1 = Union[Flatten[funP /@ (Interval /@ individualIntervals)]];  
            sel = Select[aux1, FreeQ[#, _DirectedInfinity, Infinity]&];     
            sel =!= {}, {sel[[1, 1]], {}},
            (* fts *)
            True, {individualIntervals[[1, 1]], {}}
           ], 0.5, {}]] /@ {1, 3};   
      (* potentially expand *)
      If[prs[[1]] =!= {} && prs[[2]] =!= {}, 
         length1 = Abs[Subtract @@ prs[[1, 1]]];
         length2 = Abs[Subtract @@ prs[[2, 1]]];
         If[length2/length1 < 1.1,  
            mp = Mean[prs[[1, 1]]]; 
            prs[[2, 1]] = {mp - 3/2 length1, mp + 3/2 length2}]
        ]; 
      {If[prs[[1]] =!= {},
          PRData[ prs[[1]], "HorizontalPlotRangeType" -> "ShowFewPeriods"], {}],
       If[prs[[2]] =!= {},
          PRData[ prs[[2]], "HorizontalPlotRangeType" -> "ShowMorePeriods"], {}]}
       ]
    ]


(* ::Input:: *)
(*PRF["TrigonometricsOfRational",Sin[1/x] , x]*)


(* ::Input:: *)
(*PRF["TrigonometricsOfRational",Sin[10Pi/x] , x]*)


(* ::Input:: *)
(*PRF["TrigonometricsOfRational",Sin[x]Tan[(1+x)/(1+x-x^2+x^3-x^4)], x]*)


(* very special cases *)
PRF["TrigonometricsOfRational", _. (Sin | Cos | Tan | Cot | Sec | Csc)[a_. (x_ + x0_.)^n_Integer?(#<0&)  /; 
                                                                          NumericQ[a] && NumericQ[x0]], x_] := 
{PRData[{-x0 + Abs[1/(a Pi/4)]^(1/n) {-1, 1}, {MaxRecursion -> 10}}, "HorizontalPlotRangeType" -> "ShowCenterPart"],
 PRData[{-x0 + Abs[1/(a 2 Pi)]^(1/n) {-1, 1}, {MaxRecursion -> 10}}, "HorizontalPlotRangeType" -> "ShowEnlargedMore"]}


(* very special cases *)
PRF["TrigonometricsOfRational", _. Exp[Complex[0 | 0., ai_] (x_ + x0_.)^n_Integer?(#<0&)  /; 
                                                                          NumericQ[ai] && NumericQ[x0]], x_] := 
{PRData[{-x0 + Abs[1/(ai Pi/4)]^(1/n) {-1, 1}, {MaxRecursion -> 10}}, "HorizontalPlotRangeType" -> "ShowCenterPart"],
 PRData[{-x0 + Abs[1/(ai 2 Pi/4)]^(1/n)  {-1, 1}, {MaxRecursion -> 10}}, "HorizontalPlotRangeType" -> "ShowEnlargedMore"]}


(* ::Input:: *)
(*PRF["TrigonometricsOfRational",Sin[10 Pi/x] , x]*)


(* ::Input:: *)
(*PRF["TrigonometricsOfRational",Tan[(1+x)/(1+x-x^2+x^3-x^4)], x]*)


(* ::Input:: *)
(*PRF["TrigonometricsOfRational",Sin[x]Tan[(1+x)/(1+x-x^2+x^3-x^4)], x]*)


(* ::Input:: *)
(*PRF["TrigonometricsOfRational",Sin[x/(x + 1)], x]*)


(* ::Input:: *)
(*PRF["TrigonometricsOfRational",Sin[1/(x-2)^2], x]*)


(* ::Input:: *)
(*PRF["TrigonometricsOfRational",Cos[1/x^2] + I  Sin[1/x^2], x]*)


(* ::Input:: *)
(*PRF["TrigonometricsOfRational",Exp[I/x^2], x]*)


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] NDSolve attempts to find oscillations   *)


getFirstNMaxima[ipo_, n_, rev_] := 
Module[{},
       ipoRXValues = rev[Re[First /@ InterpolatingFunctionGrid[ipo]]];
       ipoRYValues = rev[Re[InterpolatingFunctionValuesOnGrid[ipo]]];
       parti3 = Partition[ipoRYValues, 3, 1];
       pos = Position[parti3, _?((#1[[1]] < #[[2]] > #[[3]] || 
                                  #1[[1]] > #[[2]] < #[[3]])&), {1}, n, Heads -> False];
       If[pos === {}, {}, ipoRXValues[[Max[pos]]]]]


isFastGrowingQ[ipo_, ofm_] := 
Module[{},
       ipoRXValues = First /@ InterpolatingFunctionGrid[ipo];
       ipoRYValues = InterpolatingFunctionValuesOnGrid[ipo];
       Min[ipoRYValues] > 0 && 
       FreeQ[Log[ipoRYValues], _Complex] &&
       FreeQ[Log[Log[ipoRYValues]], _Complex] 
      ]


Clear[NDSolvableCache];
NDSolvableCache[expr_, x_] := (* NDSolvableCache[expr, x] = *)
Module[{},
       If[EvenOddFunctionQ[expr, x] && NumberQ[expr/. x -> 0], 
          (* wild guess *) xx0 = 0,
          deriv = D[expr, x];
          saddlePoints = nearOriginTranscendentalZerosN[deriv, x];
          If[saddlePoints === {}, 
             (* try 0 and 1 -- just as a guess *) 
             xx0 = Which[NumberQ[expr /. x -> 0], 0, 
                         NumberQ[expr /. x -> 1], 1],
             xx0 = Sort[saddlePoints, Abs[#1] < Abs[#2]&][[1]]
             ]
           ];
        df = D[expr, x] // Simplify;
        If[Not[NumberQ[xx0]] || MemberQ[df, _Derivative, {0, Infinity}, Heads -> True], 
           $Failed,
           ff0 = expr /. x -> xx0;
           ndsolR = Cases[#, _InterpolatingFunction, Infinity]& @
                    MemoryConstrained[NDSolve[{ff'[x] == df, ff[xx0] == 0}, ff, {x, xx0, xx0 + 10^3},
                            PrecisionGoal -> 2, MaxSteps -> 1000], 2^22];
           ndsolL = Cases[#, _InterpolatingFunction, Infinity]& @
                    MemoryConstrained[NDSolve[{ff'[x] == df, ff[xx0] == 0}, ff, {x, xx0, xx0 - 10^3},
                            PrecisionGoal -> 2, MaxSteps -> 1000], 2^22];
          (* analyze on grid *)
          xMax = If[ndsolR =!= {}, getFirstNMaxima[ndsolR[[1]], 6, Identity], {}];
          xMin = If[ndsolL =!= {}, getFirstNMaxima[ndsolL[[1]], 6, Reverse], {}];
          Which[xMin == xMax == {}, $Failed,
                xMin =!= {} && xMax =!= {}, {xMin, xMax},
                xMin === {} && xMax =!= {}, {xx0 - (xMax - xx0), xMax},
                xMin =!= {} && xMax === {}, {xMin, xx0 + (xx0 - xMin)}
               ]
         ] 
      ] // Quiet


NDSolvableQ[expr_, x_] := 
      TimeConstrained[Head[NDSolvableCache[expr, x]] === List, 0.3, False]


PRF["NDSolvable", expr_, x_] :={
	PRData[{NDSolvableCache[expr, x], {}}, "HorizontalPlotRangeType"->"ShowFewPeriods"]
}


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Trigonometrics of algebraic   *)


Clear[TrigonometricsOfAlgebraicQ];
TrigonometricsOfAlgebraicQ[expr_, x_] := 
(Cases[expr, (Sin | Cos | Tan | Cot | Sec | Csc | Sinc)[
             arg_?(AlgebraicQ[#, x]&)], {0, Infinity}] =!= {}) &&
FreeQ[expr, _Exp | _Log | 
            _Cosh| _Sinh | _Tanh | _Coth | _Sech | _Csch |
            _ArcCos| _ArcSin | _ArcTan | _ArcCot | _ArcSec | _ArcCsc | 
            _ArcCosh| _ArcSinh | _ArcTanh | _ArcCoth | _ArcSech | _ArcCsch, {0, \[Infinity]}]


PRF["TrigonometricsOfAlgebraic", expr_, x_] :=
Module[{theOccurringTrigs, theTrigArguments, prs},
       theOccurringTrigs = Cases[expr, _Cos | _Sin | _Tan | _Cot | _Csc | _Sec, {0, Infinity}];
       theTrigArguments = Union[First /@ theOccurringTrigs];
       prs = Function[R, TimeConstrained[guessRangeN[theTrigArguments, x, R 2Pi],
                                          2.5, {}]] /@ {1, 3};    
      Which[prs[[1]] === prs[[2]],
             prsM = Mean[prs[[1]]]; prs\[CapitalDelta] = Abs[Subtract @@ prs[[1]]];
            {PRData[{prs[[1]], {}}, "HorizontalPlotRangeType" -> "ShowFewPeriods"], 
             PRData[{{prsM - 2 prs\[CapitalDelta], prsM + 2 prs\[CapitalDelta]}, {}}, "HorizontalPlotRangeType" -> "ShowMorePeriods"]}, 
          
            prs[[1]] =!= {} && prs[[2]] =!= {},
            {PRData[ {prs[[1]], {}}, "HorizontalPlotRangeType" -> "ShowFewPeriods"],
             PRData[ {prs[[2]], {}}, "HorizontalPlotRangeType" -> "ShowMorePeriods"]},
           
            prs[[1]] =!= {},
            {PRData[ {prs[[1]], {}}, "HorizontalPlotRangeType" -> "ShowFewPeriods"]},

            prs[[2]] =!= {},
            {PRData[ {prs[[2]], {}}, "HorizontalPlotRangeType" -> "ShowMorePeriods"]} 
            ]
       ]


(* ::Input:: *)
(*PRF["TrigonometricsOfAlgebraic", 2*(1-x)*Cos[1/(1+x^2*(2-x)^2)], x] *)


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Log of algebraic   *)


LogOfAlgebraicQ[expr_, x_] := 
MatchQ[expr, Log[arg_?(AlgebraicQ[#, x]&)] | x^_. Log[arg_?(AlgebraicQ[#, x]&)]] ||
MatchQ[expr, ArcTan[_?((PolynomialQ[#, x] && Exponent[#, x] > 2)&)]]


PRF["LogOfAlgebraic", expr_, x_] :=
Module[{polyPRData, theLogArguments, zeros, poles, thePoints, prsInner, prs},
    If[MatchQ[expr, ArcTan[_?((PolynomialQ[#, x] && Exponent[#, x] > 2)&)]],  
       polyPRData = #[[1, 1]]& /@ Cases[PRF["Polynomial", expr[[1]], x], _PRData, {0, \[Infinity]}];
       If[Length[polyPRData] === 1,
          {PRData[ {polyPRData[[1]], {}}, "HorizontalPlotRangeType" -> "ShowSomething"],
           PRData[ enlargePlotRange[{polyPRData[[1]], {}}, 2], "HorizontalPlotRangeType" -> "ShowSomeMoreOfSomething"]},
          {PRData[ {polyPRData[[1]], {}}, "HorizontalPlotRangeType" -> "ShowSomething"],
           PRData[ enlargePlotRange[{polyPRData[[2]], {}}, 1], "HorizontalPlotRangeType" -> "ShowSomeMoreOfSomething"]} 
           ],
       (* generic *)
       theLogArguments = Select[First /@ Cases[expr, _Log, Infinity], AlgebraicQ[#, x]&];
       zeros = x /. Solve[Or @@ (# == 0& /@ theLogArguments), x];
       poles = x /. Solve[Or @@ (1/# == 0& /@ theLogArguments), x];
       thePoints = Union[Abs[Select[Union[Flatten[{0, zeros, poles}]], NumericQ]]];
       prsInner =  Which[thePoints === {}, {-1, 1}, 
                         thePoints == {0.}, {-1, 1},
                         Length[thePoints] == 1., thePoints {1/2, 3/2},
                         (Max[thePoints] - Min[thePoints]) < Abs[Mean[thePoints]],  Mean[thePoints] {1/2, 3/2},
                         True, 
                         iLength = Max[thePoints] - Min[thePoints];
                         {Min[thePoints] - iLength/4, Max[thePoints] + iLength/4}
                         ];

      If[prsInner === $Failed, $Failed, 

       prs = {prsInner, 6 prsInner};
  
        {If[prs[[1]] =!= {},
          PRData[ {prs[[1]], {}}, "HorizontalPlotRangeType" -> "ShowSomething"], {}],
         If[prs[[2]] =!= {},
          PRData[ {prs[[2]], {}}, "HorizontalPlotRangeType" -> "ShowSomeMoreOfSomething"], {}]}
       ]
       ]
      ]


(* ::Input:: *)
(*PRF["LogOfAlgebraic", Log[x + Sqrt[2] x^2], x]*)


(* ::Input:: *)
(*PRF["LogOfAlgebraic", ArcTan[x^3], x]  *)


(* ::Input:: *)
(*PRF["LogOfAlgebraic", ArcTan[x^12-3x+4], x]  *)


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Trigonometrics of other   *)


Clear[ContainsTrigonometricsQ];
ContainsTrigonometricsQ[expr_, x_] := 
Cases[expr, (Sin | Cos | Tan | Cot | Sec | Csc | Sinc)[
             arg_?(RationalFunctionQ[#, x]&)], {0, Infinity}] =!= {}


PRF["ContainsTrigonometrics", expr_, x_] :=
Module[{},
       theOccurringTrigs = Cases[expr, _Cos | _Sin | _Tan | _Cot | _Csc | _Sec, {0, Infinity}];
       theTrigArguments = Union[First /@ theOccurringTrigs];
       prs = Function[R, TimeConstrained[guessRangeN[theTrigArguments, x, R 2Pi],
                                          2.5, {}]] /@ {1, 3};       
        {If[prs[[1]] =!= {},
          PRData[ {prs[[1]], {}}, "HorizontalPlotRangeType" -> "ShowFewPeriods"], {}],
       If[prs[[2]] =!= {},
          PRData[ {prs[[2]], {}}, "HorizontalPlotRangeType" -> "ShowMorePeriods"], {}]}
       ]


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Analytic oscillatory  at infinity*)


Clear[AnalyticOscillatoryQ, analyticOscillatoryPRCached];
(* a hacky way *)
AnalyticOscillatoryQ[expr_, x_] := 
     NumericQ[expr /. x -> Pi^Catalan] && 
     FreeQ[expr, piecewiseFunctionPattern, {0, Infinity}] &&
     analyticOscillatoryPRCached[expr, x] =!= $Failed


mainAsymptoticTerm[expr_, {x_, x0_}, opts___] := 
Module[{}, order = -3;
       While[ns = Normal[ser = Series[expr, {x, x0, order}, opts]]; 
             (FreeQ[ser, _SeriesData, {0, Infinity}] || 
              NumericQ[ns]) && order < 6, order++];
       ns]       


Clear[analyticOscillatoryPRCached];
analyticOscillatoryPRCached[expr_, x_] := analyticOscillatoryPRCached[expr, x] =
Module[{(* verticalRangeFactor = 2/3, horizontalRangeFactor = 0.02*)},
        serMInf = mainAsymptoticTerm[expr, {x, -Infinity}];
        serPInf = mainAsymptoticTerm[expr, {x, +Infinity}];
        trigTerms = Cases[{serMInf, serPInf}, _Cos | _Sin | _Tan | _Cot | _Csc | _Sec, 
                          {0, Infinity}];
        (* use x-dependent arguments *)
        trigArguments = Select[Union[First /@ trigTerms], MemberQ[#, x, {0, \[Infinity]}]&];
         If[trigTerms =!= {}, 
            gtrNear = TimeConstrained[guessRangeN[trigArguments, x, 4Pi], 1, {}];
            If[gtrNear =!= {}, 
               gtrFar = TimeConstrained[guessRangeN[trigArguments, x, 16Pi], 1, {}]
               ]
            ];
       auxFMM = Function[gtr, {Min[gtr, -Max[gtr]], Max[gtr, -Min[gtr]]}];
       Which[trigArguments =!= {} && gtrNear =!= {} && gtrFar =!= {},
             {PRData[ {auxFMM[gtrNear], {}}, "HorizontalPlotRangeType" -> "ShowFewPeriods"],
              PRData[ {auxFMM[gtrFar], {}} , 
                      "HorizontalPlotRangeType" -> "ShowMorePeriods"]},
             trigArguments =!= {} && gtrNear =!= {},
              {PRData[ {auxFMM[gtrNear], {}}, "HorizontalPlotRangeType" -> "ShowFewPeriods"]},
             True, $Failed]
       ]


(* ::Input:: *)
(*analyticOscillatoryPRCached[DawsonF[x], x]*)


PRF["AnalyticOscillatory", expr_, x_] := analyticOscillatoryPRCached[expr, x]


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Analytic oscillatory  at origin*)


Clear[AnalyticOscillatoryAtOriginQ, analyticOscillatoryAtOriginPRCached];
(* a hacky way *)
AnalyticOscillatoryAtOriginQ[expr_, x_] := 
     Quiet[Not[NumericQ[expr /. x -> 0]]] && 
     FreeQ[expr, piecewiseFunctionPattern, {0, Infinity}] &&
     analyticOscillatoryAtOriginPRCached[expr, x] =!= $Failed


mainAsymptoticTermAtOrigin[expr_, {x_, x0_}, opts___] := 
Module[{}, order = -3;
       While[ns = Normal[ser = Series[expr, {x, x0, order}, opts]]; 
             (FreeQ[ser, _SeriesData, {0, Infinity}] || 
              NumericQ[ns]) && order < 6, order++];
       ns]       


Clear[analyticOscillatoryAtOriginPRCached];
analyticOscillatoryAtOriginPRCached[expr_, x_] := analyticOscillatoryAtOriginPRCached[expr, x] =
Module[{(* verticalRangeFactor = 2/3, horizontalRangeFactor = 0.02*)},
        serMInf = mainAsymptoticTerm[expr, {x, 0}, Assumptions -> x > 0];
        serPInf = mainAsymptoticTerm[expr, {x, 0}, Assumptions -> x < 0];
        trigTerms = Cases[{serMInf, serPInf}, _Cos | _Sin | _Tan | _Cot | _Csc | _Sec, 
                          {0, Infinity}];
        (* use x-dependent arguments *)
        trigArguments = Select[Union[First /@ trigTerms], MemberQ[#, x, {0, \[Infinity]}]&];
         If[trigTerms =!= {}, 
            gtrNear = TimeConstrained[guessRangeN[trigArguments, x, 4Pi], 1, {}];
            If[gtrNear =!= {}, 
               gtrFar = TimeConstrained[guessRangeN[trigArguments, x, 1/4 Pi], 1, {}]
               ]
            ];
       auxFMM = Function[gtr, {Min[gtr, -Max[gtr]], Max[gtr, -Min[gtr]]}];
       Which[trigArguments =!= {} && gtrNear =!= {} && gtrFar =!= {},
             {PRData[ {auxFMM[gtrNear], {}}, "HorizontalPlotRangeType" -> "ShowFewPeriods"],
              PRData[ {auxFMM[gtrFar], {}} , 
                      "HorizontalPlotRangeType" -> "ShowMorePeriods"]},
             trigArguments =!= {} && gtrNear =!= {},
              {PRData[ {auxFMM[gtrNear], {}}, "HorizontalPlotRangeType" -> "ShowFewPeriods"]},
             True, $Failed]
       ]


(* ::Input:: *)
(*analyticOscillatoryAtOriginPRCached[SphericalBesselJ[0, 1 + 1/x], x]*)


PRF["AnalyticOscillatoryAtOrigin", expr_, x_] := analyticOscillatoryAtOriginPRCached[expr, x]


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Elementary   *)


elementaryFunctionHeadPattern =  
            _Exp | _Log | _Power | _Sinc |
            _Sin | _Cos | _Tan | _Cot | _Sec | _Csc |
            _Cosh| _Sinh | _Tanh | _Coth | _Sech | _Csch |
            _ArcCos| _ArcSin | _ArcTan | _ArcCot | _ArcSec | _ArcCsc | 
            _ArcCosh| _ArcSinh | _ArcTanh | _ArcCoth | _ArcSech | _ArcCsch;

elementaryFunctionFunctionPattern = 
          Alternatives @@ (First /@ (List @@ (elementaryFunctionHeadPattern)));


Clear[ElementaryQ];
ElementaryQ[expr_, x_] :=
            (DeleteCases[Union[Level[expr, {-1}, Heads -> True]],
                        Plus | Times | x | _?NumericQ | 
                        elementaryFunctionFunctionPattern] === {}) &&
            ElementaryCache[expr, x] =!= $Failed


specialPoints[Power[x_, _], x_] := {0}
specialPoints[Power[_, x_], x_] := {}
specialPoints[Exp[x_], x_] := {}
specialPoints[Log[x_], x_] := {-1, 0, 1}
specialPoints[Sin[x_], x_] := {-Pi, 0, Pi}
specialPoints[Sinc[x_], x_] := {-Pi, 0, Pi}
specialPoints[Cos[x_], x_] := {-Pi/2, Pi/2}
specialPoints[Tan[x_], x_] := {-Pi/2, Pi/2}
specialPoints[Cot[x_], x_] := {-Pi, 0, Pi}
specialPoints[Sec[x_], x_] := {-Pi/2, Pi/2}
specialPoints[Csc[x_], x_] := {-Pi, 0, Pi}
specialPoints[ArcSin[x_], x_] := {-1, 1}
specialPoints[ArcCos[x_], x_] := {-1, 1}
specialPoints[ArcTan[x_], x_] := {0}
specialPoints[ArcCot[x_], x_] := {0}
specialPoints[ArcSec[x_], x_] := {-1, 1}
specialPoints[ArcCsc[x_], x_] := {-1, 1}
specialPoints[Cosh[x_], x_] := {}
specialPoints[Sinh[x_], x_] := {0}
specialPoints[Tanh[x_], x_] := {0}
specialPoints[Coth[x_], x_] := {0}
specialPoints[Coth[x_], x_] := {0}
specialPoints[Sech[x_], x_] := {}
specialPoints[Csch[x_], x_] := {0}
specialPoints[ArcSinh[x_], x_] := {0}
specialPoints[ArcCosh[x_], x_] := {1}
specialPoints[ArcTanh[x_], x_] := {-1, 1}
specialPoints[ArcCoth[x_], x_] := {-1, 1}
specialPoints[ArcSech[x_], x_] := {0, 1}
specialPoints[ArcCsch[x_], x_] := {0}


makeEquationSols[f_[args__], x_] := (x /. Solve[args == #, x])& /@ 
                                               (2 Flatten[specialPoints[f[args], #]& /@ {args}]) /.
                                ConditionalExpression[sol_, C[1] \[Element] Integers] :> First[SortBy[{sol /. C[1] -> 0, sol /. C[1] -> -1, sol /. C[1] -> 1}, Abs]]


(* ::Input:: *)
(*makeEquationSols[Cos[Exp[-x^2]], x]*)


Clear[ElementaryCache];
ElementaryCache[expr_, x_] := ElementaryCache[expr, x] =
Quiet[
Module[{},
       theOccurringElementaryFunctionsWithArguments = 
                   Cases[expr, elementaryFunctionHeadPattern, {0, Infinity}];
       eqsSols = Union[Select[Re[N[DeleteCases[Union[Flatten[makeEquationSols[#, x]& /@ 
                             theOccurringElementaryFunctionsWithArguments]], x] ]],
                              (NumericQ[#]  && Im[#] == 0.)&]];
       If[Length[eqsSols] > 1,
          aux = {Max[-$MaxMachineNumber/10^100, Min[eqsSols]], 
                 Min[+$MaxMachineNumber/10^100, Max[eqsSols]]};
          {PRData[{padRange[aux, 0.05], {}}, "HorizontalPlotRangeType" -> "ShowSomething"],
           PRData[enlargePlotRange[{padRange[aux, 0.05], {}}, 2], 
                 "HorizontalPlotRangeType" -> "ShowMoreOfSomething"]},
           $Failed]
      ] ]


(* ::Input:: *)
(*ElementaryCache[Cos[Exp[-x^2]], x]*)


(* ::Input:: *)
(*ElementaryCache[Sin[2^x], x]*)


(* ::Input:: *)
(*ElementaryCache[(1/10)*(Log[1+Exp[-10]]-Log[Exp[-10*x^2]+Exp[-10]]), x]*)


(* ::Input:: *)
(*ElementaryCache[Log[Sinh[Exp[x]]], x]*)


(* ::Input:: *)
(*ElementaryCache[Log[Log[x]], x]*)


(* ::Input:: *)
(*ElementaryCache[Log[Log[Log[x]]], x]*)


(* ::Input:: *)
(*ElementaryCache[Log[Log[Log[Log[x]]]], x]*)


PRF["Elementary", expr_, x_] := ElementaryCache[expr, x]


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Complex components   *)


Clear[ComplexComponentsQ];
ComplexComponentsQ[expr_, x_] :=
            MemberQ[expr, _Re | _Im _ | _Sign | _Conjugate, {0, Infinity}] &&
            FreeQ[ComplexExpand[expr], _Re | _Im | _Sign | _Conjugate, {0, Infinity}]


PRF["ComplexComponents", expr_, x_]  := 
Module[{},
        goodRealValuedPart = ComplexExpand[expr];
        proposedPlotRangeAndFunctionFeatures[goodRealValuedPart, x]
        ]


piecewiseFunctionPattern =  _Round | _Floor | _Ceiling | _UnitStep | _Abs | _Min | _Max | _Sign | _Piecewise |
                            _SquareWave | _TriangleWave | _SawtoothWave;


Clear[PiecewiseQ];
PiecewiseQ[expr_, x_] := 
         Cases[expr, piecewiseFunctionPattern, {0, Infinity}] =!= {}


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Piecewise   *)


periodictyBI[f_, x_] := 
Block[{Periodic`Private`PDWellDefinedFunctionQ}, Periodic`Private`PDWellDefinedFunctionQ[___] := True;
       Assuming[Element[x, Reals], Periodic`PeriodicFunctionPeriod[TrigToExp @ f, x]]]


PRF["Piecewise", expr_, x_] :=
Module[{(*verticalRangeFactor = 2/3, horizontalRangeFactor = 0.02*)
         (*period, sols, args, eExpList, eExpPos, eExp, pwe, allCondsNumbers, aux*)},
   If[(* special named functions *)
      MemberQ[expr, _SquareWave | _TriangleWave | _SawtoothWave, {0, \[Infinity]}],  
      period = periodictyBI[expr, x];  
      If[NumericQ[period] && Im[period] == 0., sols = period {-1, 1},
         (* aperiodic *)
         args = Last /@ Cases[expr,_SquareWave | _TriangleWave | _SawtoothWave, {0, \[Infinity]}];
         sols = Union[Cases[Flatten[Chop[N[x /. Solve[# == -1 || # == 1, x]& /@ args]]], _Real]]
         ];
       If[Length[sols] === 1,
          {PRData[ {{-1, 1}, {}}, "HorizontalPlotRangeType" -> "ShowCentralPart"],
           PRData[ {{-3, 3}, {}}, "HorizontalPlotRangeType" -> "ShowOverall"]},  
           {min, max} = {Min[sols], Max[sols]};
           {PRData[ {{min, max}, {}}, "HorizontalPlotRangeType" -> "ShowCentralPart"],
            PRData[ 3 {{min, max}, {}}, "HorizontalPlotRangeType" -> "ShowOverall"]}
            ],
       (* try to resolve into explicit piecewise *)
       eExpList = {-2, -1, 0, Log[10, 2], Log[10, 3], Log[10, 5], 1, 2, 3};
       (* remove distributions *)
       exprN  = expr /. {HeavisideTheta -> UnitStep, HeavisidePi -> UnitBox, HeavisideLambda -> UnitTriangle};
       eExpPos = 1;
       While[eExp = eExpList[[eExpPos]];
             pwe = PiecewiseExpand[exprN, -10^eExp < x < 10^eExp];
             (Head[pwe] =!= Piecewise || Length[pwe[[1]]] < 3) && 
             eExpPos < Length[eExpList],  eExpPos++];
       If[Head[pwe] === Piecewise,
          allCondsNumbers = Select[
               Union[Flatten[DeleteCases[List @@@ 
                       Flatten[(Last /@ pwe[[1]]) /. {Or -> List, And -> List}], x]]],
                                   NumericQ];
         aux = If[Length[allCondsNumbers] === 1, 
                  {allCondsNumbers[[1]] - 1, allCondsNumbers[[1]] + 1},
                  {Min[allCondsNumbers], Max[allCondsNumbers]}];
        
        {PRData[ enlargePlotRange[{aux, {}}, 0.12], "HorizontalPlotRangeType" -> "ShowCentralPart"],
         PRData[ enlargePlotRange[{aux, {}}, 1.6], "HorizontalPlotRangeType" -> "ShowOverall"]},
        $Failed]

      ]
       ]


(* ::Input:: *)
(*PRF["Piecewise",SawtoothWave[5 Exp[x]]+Sin[3 SquareWave[Exp[x]]], x]*)


(* ::Input:: *)
(*PRF["Piecewise",HeavisideLambda[x+Pi/2], x]*)


(* ::Input:: *)
(*PRF["Piecewise",UnitTriangle[x+Pi/2], x]*)


(* ::Input:: *)
(*PRF["Piecewise",x+4 (-1+Ceiling[x/4]), x]*)


(* ::Input:: *)
(*PRF["Piecewise", SquareWave[3x] + Sin[Pi x] SawtoothWave[2x], x]*)


(* ::Input:: *)
(*PRF["Piecewise", SquareWave[3x] + x SawtoothWave[2x], x]*)


(* ::Input:: *)
(*PRF["Piecewise", SquareWave[3x], x]*)


(* ::Input:: *)
(*PRF["Piecewise", TriangleWave[3x], x]*)


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Table desperate   *)


Clear[TableDesperateCache]
TableDesperateCache[expr_, x_] := 
Module[{},
       allBuiltInFunctions = DeleteCases[#, _?(AlgebraicQ[#, x]&)]& @ 
                              (Cases[expr, _?((# =!= Plus && # =!= Times &&
                                             MemberQ[Attributes[#], NumericFunction])&)[__], 
                                   {0, Infinity}] // PowerExpand);
       goodScales = Cases[scale[#, x]& /@ allBuiltInFunctions, {_?NumberQ, _?NumberQ}, {-2}];
       Which[Flatten[goodScales] === {}, $Failed,
             Length[Union[Flatten[goodScales]]] >= 2,
             aux = {Min[goodScales], Max[goodScales]};
             {PRData[ {aux, {}}, "HorizontalPlotRangeType" -> "ShowSomething"],
              PRData[ enlargePlotRange[{aux, {}}, 1] , 
                                 "HorizontalPlotRangeType" -> "ShowSomeMoreOfSomething"]}]
      ]


TableDesperateQ[expr_, x_] := TableDesperateCache[expr, x] =!= $Failed


PRF["TableDesperate", expr_, x_] := TableDesperateCache[expr, x]


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] ReIm*)


realValuedRange[f_, x_] :=
Module[{},
       red = Reduce[Exists[x, y == f && Element[x | y, Reals] && y != 0], y];
       Minimize[{x, red}, x, Reals]



       ]


(* ::Input:: *)
(*(* use QE for algebraic functions *)*)
(*Reduce[Exists[x, y == Im[x^4 Sqrt[Sqrt[x-2]-x]] && Element[x | y, Reals] && y != 0], y, WorkingPrecision -> 50]//Timing*)


Clear[ReImQCache];
ReImQCache[expr_, x_] := ReImQCache[expr, x] =
Module[{  
         y, red, min, max, range, trk, flag 

       }, 
      trk =  Quiet @ 
      Which[MatchQ[expr, (Im | Re)[_]], 
            (* find range where Re/Im is nonvanishing *)
            red = Reduce[Exists[y, Element[x | y, Reals],  y == expr && y != 0], WorkingPrecision -> 50];
            min = Minimize[{x, red}, x, Reals];
            max = Maximize[{x, red}, x, Reals]; 
            range = If[Head[min] === List @@ Head[max] === List,  
                       Which[NumberQ[min[[1]]] && NumberQ[max[[1]]], flag = All; {min[[1]], max[[1]]},
                             NumberQ[min[[1]]], {min[[1]], min[[1]] + 2},
                             NumberQ[max[[1]]], {max[[1]] - 2, max[[1]]}, 
                             True, $Failed 
                             ],
                       True],
            True, $Failed 
           ]; 
       If[trk =!= $Failed,
         (* flag: if there is a finite range only, do not extend too far *)
         {PRData[{trk, {}}, "HorizontalPlotRangeType" -> "ShowSomething"],
          PRData[enlargePlotRange[{trk, 0}, If[flag === All, 0.5, 1.]], "HorizontalPlotRangeType" -> "ShowMoreOfSomething"] 
         },
         $Failed
         ]
      ] 


(* ::Input:: *)
(*ReImQCache[Im[x^4 Sqrt[x^3 - x]], x]*)


(* ::Input:: *)
(*ReImQCache[Im[  Sqrt[x^2 - 4]], x]*)


(* ::Input:: *)
(*(* not RE/Im input *)*)
(*ReImQCache[x^4 Sqrt[x^3 - x], x]*)


ReImQ[expr_, x_] := If[ReImQCache[expr, x] === $Failed, False, True]


PRF["ReIm", expr_, x_] := ReImQCache[expr, x]


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Monotonic   *)


MonotonicQ[expr_, x_] := 
CalculateTimeConstrained[Reduce[D[expr, x] > 0, x,  Reals] || Reduce[D[expr, x] < 0, x, Reals], 0.5, False]


(* ::Input:: *)
(*MonotonicQ[((-x)*Log[2]+Log[2^x+3^x])/Log[3/2], x]*)


(* ::Input:: *)
(*MonotonicQ[E^2^x*(1 + E^2^x)^3, x]*)


PRF["Monotonic", expr_, x_] :=
Module[{(* verticalRangeFactor = 1/3, horizontalRangeFactor = 0.02, tMax = 0.1, *)
        order, gc, vals, mean, spread},
        TimeConstrained[bag = {}; order = 0; gc = 0;
                        While[order < 6 && gc < 3,
                              order++; red = Reduce[D[expr, {x, order}]== 0, x, Reals]; 
                              If[red =!= False, gc++ ]; AppendTo[bag, red]
                              ],
                             0.5];
       vals = Last /@ Cases[DeleteCases[Flatten[(bag) /. Or -> List], False], x == _];
       If[vals === {}, $Failed,
          mean = Mean[N[vals]];
          spread = If[Length[vals]=== 1, 1, StandardDeviation[N[vals]]];
          {PRData[{{mean - spread, mean + spread}, {}},"HorizontalPlotRangeType" -> "ShowZerosExtremasInflections"],
           PRData[{{mean - 3 spread, mean + 3 spread}, {}},"HorizontalPlotRangeType" -> "ShowGlobalShape"]
           }
         ] 
 
    ]      


(* ::Input:: *)
(*(* crash*)
(*  PRF["Monotonic", E^2^x*(1 + E^2^x)^3, x]*)
(**)*)


(* ::Input:: *)
(*PRF["Monotonic", ((-x)*Log[2]+Log[2^x+3^x])/Log[3/2], x]  *)


(* ::Input:: *)
(*PRF["Monotonic", 3^x/(3^x+2^x) , x]*)


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] General   *)


Clear[GeneralQCache];
GeneralQCache[expr_, x_] := GeneralQCache[expr, x] =
Module[{(*verticalRangeFactor = 2/3,horizontalRangeFactor = 0.02*)
        (*singularitiesS, singularitiesInvPre, singularitiesInv, zerosS, zeros,
        extremas, scaleCoeffs, specialXPoints1, avScale, specialXPointsPre, specialXPoints,
        nearOriginPoints, aux*)
       }, 
       singularitiesS = nearOriginTranscendentalZerosS[1/expr, x];  
       singularitiesInvPre = nearOriginTranscendentalZerosN[1/expr, x, 3]; 
       (* avoid nested exponentially decaying functions *)
       singularitiesInv = Select[singularitiesInvPre, ((Abs[1/expr] /. x -> #) < (Abs[1/expr] /. x -> 2#))&]; 
       zerosS = nearOriginTranscendentalZerosN[expr, x]; 
       zeros = nearOriginTranscendentalZerosN[expr, x, 4]; 
       deriv = If[MatchQ[expr, (Re | Im | Abs | Arg)[_]], expr[[0]][D[expr[[1]], x]], D[expr, x]];
       extremas = nearOriginTranscendentalZerosN[deriv, x, 4]; 
       extremaSings = nearOriginTranscendentalZerosN[1/deriv, x, 4]; 
       scaleCoeffs = N[Mean[Abs[If[# === {}, {1}, #]& @ Cases[expr, _?NumericQ, {0, \[Infinity]}]]]];
       If[NumericQ[scale] || scale == 0, Null, scaleCoeffs = 1];
       specialXPoints1 = Select[Sort[Flatten[Union @ Re @ {singularitiesS, singularitiesInv, zerosS, zeros, extremas, extremaSings}]],
                                (NumericQ[#] && Abs[#] < 10^6)&] // N;  
       avScale = Mean[Abs[specialXPoints1]];
       (* try to remove multiples *)
       specialXPointsPre = Union[Union @ specialXPoints1, SameTest -> ((Abs[#1 - #2]/Max[Abs[#1], Abs[#2] ] < 0.01 ||
                                                             Abs[#1 - #2] < 10^-3 avScale)&)];
      (* avoid using slightly different numerical results as a 'real' result *)
       specialXPoints = N[Union[Union[Round[specialXPointsPre, 10^(-1/(4 avScale))]]]];
       If[Length[specialXPoints] > 3, 
          specialXPoints = DeleteCases[specialXPoints, _?(Abs[#] < 10^-6 scaleCoeffs&)]
          ];
       If[Length[specialXPoints] < 2, $Failed,
          nearOriginPoints = Sort[specialXPoints, Abs[#1] < Abs[#2]&]; 
          aux = Sort[{nearOriginPoints[[1]], nearOriginPoints[[2]]}]; 
          If[(expr /. x -> -x) === expr, aux[[1]] = -aux[[2]]];
        {PRData[{aux, {}}, "HorizontalPlotRangeType" -> "ShowSomething"],
         PRData[enlargePlotRange[{aux, 0}, 2], "HorizontalPlotRangeType" -> "ShowMoreOfSomething"]}]
      ]


(* ::Input:: *)
(*GeneralQCache[((-x)*Log[2]+Log[2^x+3^x])/Log[3/2], x]*)


(* ::Input:: *)
(*GeneralQCache[Im[x^4 Sqrt[x^3 - x]], x]*)


(* ::Input:: *)
(*GeneralQCache[x/2/(1-E^(-x/2)), x]*)


(* ::Input:: *)
(*GeneralQCache[1/(1+E^x^2), x]*)


(* ::Input:: *)
(*GeneralQCache[Sinh[Cosh[x]], x]*)


GeneralQ[expr_, x_] := If[GeneralQCache[expr, x] === $Failed, False, True]


PRF["General", expr_, x_] := GeneralQCache[expr, x]


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Tan-rescaled Plot call*)


Clear[TanRescaledPlotCache];
TanRescaledPlotCache[expr_, x_] := TanRescaledPlotCache[expr, x] =
Module[{
        \[CurlyEpsilon], sf, pl, lines, rescaledLine, minMaxPositions, pr0, range, minMaxPositionsFrom\[CapitalDelta]s, minMaxPositionsMain,
        minMaxPositionsInner, minMaxPositionsOuter, \[CapitalDelta]i, \[CapitalDelta]o, symmetrize

       },
       sf[\[Xi]_] = Abs[D[Tan[\[Xi]], \[Xi]]];
       \[CurlyEpsilon] = 10^-6;
       pl = CalculateTimeConstrained[
             MemoryConstrained[Plot[Evaluate[Abs[expr] /. x -> Tan[y]], {y, -Pi/2 + \[CurlyEpsilon], Pi/2 - \[CurlyEpsilon]}, 
                                          ColorFunction -> None, MaxRecursion -> 6, WorkingPrecision -> 20] // Quiet, 
                               MemoryInUse[] + 20 * 10^6], 
                                     1];
       lines = First /@ Cases[Normal[pl], _Line, {0, \[Infinity]}];
       If[lines === {}, $Failed,
          rescaledLine = {Tan[#1], #2}& @@@ lines[[1]];
          minMaxPositions = Join[
             #[[2, 1]]& /@ 
             Select[Partition[{#1, Abs[#2]}& @@@ rescaledLine, 3, 1],
                    ((#[[1, 2]] < #[[2, 2]] && #[[3, 2]] < #[[2, 2]]) ||
                     (#[[1, 2]] > #[[2, 2]] && #[[3, 2]] > #[[2, 2]]))&],
             (* poles and other singularities where line is potentially cut *)
             If[Length[lines] === 1, {},
                Mean[{Last[#1], First[#2]}][[1]]& @@@ Partition[lines, 2, 1]
               ]];
          pr0 = Which[Length[minMaxPositions] < 2, $Failed,
                      (* just a few bumps *)
                      Length[minMaxPositions] < 5,  
                      range = Max[minMaxPositions] - Min[minMaxPositions];
                      {{Min[minMaxPositions] - range/6, Max[minMaxPositions] + range/6}},
                      (* more bumps *)
                      Length[minMaxPositions] < 12,  
                      minMaxPositionsFrom\[CapitalDelta]s = Last /@ Identity[Sort[{1/sf[#2] Mean[{Abs[#2 - #1], Abs[#3 - #1]}], #2}& @@@ 
                                                                              Partition[minMaxPositions, 3, 1]]];
                      minMaxPositionsMain = Take[minMaxPositionsFrom\[CapitalDelta]s, Min[Length[minMaxPositionsFrom\[CapitalDelta]s], 6]];
                      range = Max[minMaxPositionsMain] - Min[minMaxPositionsMain];
                      {{Min[minMaxPositionsMain] - range/6, Max[minMaxPositionsMain] + range/6}},
                      (* many bumps *)
                      True,  
                      minMaxPositionsFrom\[CapitalDelta]s = Last /@ Identity[Sort[{1/sf[#2] Mean[{Abs[#2 - #1], Abs[#3 - #1]}], #2}& @@@ 
                                                                              Partition[minMaxPositions, 3, 1]]];
                      minMaxPositionsInner = Take[minMaxPositionsFrom\[CapitalDelta]s, Min[Length[minMaxPositionsFrom\[CapitalDelta]s], 6]];  
                      minMaxPositionsOuter = Take[minMaxPositionsFrom\[CapitalDelta]s, Min[Length[minMaxPositionsFrom\[CapitalDelta]s], 12]]; 
                      \[CapitalDelta]i = {Min[minMaxPositionsInner], Max[minMaxPositionsInner]};
                      \[CapitalDelta]o = {Min[minMaxPositionsOuter], Max[minMaxPositionsOuter]};
                      If[\[CapitalDelta]i == \[CapitalDelta]o, {\[CapitalDelta]i}, {\[CapitalDelta]i, \[CapitalDelta]o}]
                     ]
              ];
       Which[pr0 === $Failed, $Failed,
             symmetrize[{\[Xi]1_, \[Xi]2_}] = If[TrueQ[((expr) - (expr /. x -> -x)) == 0], {-1, 1} Mean[Abs[{\[Xi]1, \[Xi]2}]], {\[Xi]1, \[Xi]2}];
             Length[pr0] === 1, 
             {PRData[{symmetrize @ pr0[[1]], {}}, "HorizontalPlotRangeType" -> "ShowSomething"],
              PRData[enlargePlotRange[{symmetrize @ pr0[[1]], 0}, 2], "HorizontalPlotRangeType" -> "ShowMoreOfSomething"]},
             Length[pr0] === 2, 
             {PRData[{symmetrize @ pr0[[1]], {}}, "HorizontalPlotRangeType" -> "ShowSomething"],
              PRData[{symmetrize @ pr0[[2]], {}}, "HorizontalPlotRangeType" -> "ShowMoreOfSomething"]}
           ]
             
      ]


TanRescaledPlotQ[expr_, x_] := If[TanRescaledPlotCache[expr, x] === $Failed, False, True]


PRF["TanRescaledPlot", expr_, x_] := TanRescaledPlotCache[expr, x]


(* ::Input:: *)
(*PRF["TanRescaledPlot", (1+1/x^2)/(1+x^-x), x]*)


(* ::Input:: *)
(*PRF["TanRescaledPlot", Sin[x Exp[x]], x]*)


(* ::Input:: *)
(*PRF["TanRescaledPlot", Sin[x  + 1/(x^3 + Cos[x])], x]*)


(* ::Input:: *)
(*PRF["TanRescaledPlot", Sqrt[(1+x^2)/(1+x^4)], x]*)


(* ::Input:: *)
(*PRF["TanRescaledPlot", ((-x)*Log[2]+Log[2^x+3^x])/Log[3/2], x]*)


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Tan-rescaled derivatives Plot call*)


Clear[TanRescaledDerivativesPlotCache];
TanRescaledDerivativesPlotCache[expr_, x_] := TanRescaledDerivativesPlotCache[expr, x] =
Module[{der1, der2, firstDerivativeResult}, 
       der1 = D[expr, x];
       If[FreeQ[der1, _Derivative, {0, \[Infinity]}], 
          firstDerivativeResult = PRF["TanRescaledPlot", der1, x]; 
          If[firstDerivativeResult =!= $Failed,
             firstDerivativeResult,
             der2 = D[expr, x, x];
             If[FreeQ[der2, _Derivative, {0, \[Infinity]}], 
                PRF["TanRescaledPlot", der2, x]
                 (* no third derivative *)]
            ], 
          $Failed
       ]
      ]


TanRescaledDerivativesPlotQ[expr_, x_] := If[TanRescaledDerivativesPlotCache[expr, x] === $Failed, False, True]


PRF["TanRescaledDerivativesPlot", expr_, x_] := TanRescaledDerivativesPlotCache[expr, x]


(* ::Input:: *)
(*PRF["TanRescaledDerivativesPlot", ((-x)*Log[2]+Log[2^x+3^x])/Log[3/2], x]*)


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] FindRoot attempts to find singularities (to be done)   *)


(* ::Text:: *)
(*search for mimimum of -Log[expr]^2 with initial values \[Tilde] log(|x|)*)


(* ::Subsubsection::Closed:: *)
(*   \[DoubleLongRightArrow] Fall through   *)


FallThroughQ[expr_, x_] := True


PRF["FallThrough", expr_, x_] :=
Module[{parts = DeleteCases[Cases[expr, _?NumericQ, {0, Infinity}],
                            _?(#==0.&)] // Union // N // Abs},
       mm = Min[Max[parts], 1/Min[parts]]; print[parts];
       aux =  If[parts =!= {}, {-2 mm, 2 mm}, {-1, 1}];
       {PRData[{aux, {}}, "HorizontalPlotRangeType" -> "ShowSomething"],
        PRData[enlargePlotRange[{aux, 0}, 2], "HorizontalPlotRangeType" -> "ShowMoreOfSomething"]}
       ]


(* ::Subsubsection:: *)
(*   \[DoubleLongRightArrow] ListPlot case*)


isPositiveQ[expr_, x_, {-\[Infinity], xM_}] := (expr /. x -> xM - 5) > 0

isPositiveQ[expr_, x_, {xM_, \[Infinity]}] := (expr /. x -> xM + 5) > 0

isPositiveQ[expr_, x_, {x1_, x2_}] := x2 - x1 > 10 && (expr /. x -> Round[Mean[{x1, x2}]]) > 0


positiveIntervals[expr_, x_] := 
Module[{deltaK = 160},
       sol = Solve[expr == 0, x];
       Which[Head[sol] === Solve, $Failed,
             sol === {}, $Failed,
             sol === {{}}, Interval[{-\[Infinity], \[Infinity]}],
             True,
             xs = Round[Select[Chop[N[x /. sol]], Im[#] == 0&]];
             If[xs === {}, 
                If[(expr /. x -> 3) > 0, Interval[{-\[Infinity], \[Infinity]}], $Failed],
                {min, max} = {Min[xs], Max[xs]};
                positiveArgumentIntervals = 
                {If[isPositiveQ[expr, x, {-\[Infinity], min}], IntervalRange[{min - deltaK, min}], Sequence @@ {}],
                 Sequence @@ (If[isPositiveQ[expr, x, #], #, Sequence @@ {}]& /@ Partition[xs, 2, 1]),
                 If[isPositiveQ[expr, x, {max, \[Infinity]}], IntervalRange[{max, max + deltaK}], Sequence @@ {}]
                };
             Which[positiveArgumentIntervals === {}, $Failed,
                   Max[Norm[Subtract @@ #[[1]]]& /@ positiveArgumentIntervals] < 10, $Failed,
                   True, Interval @@ (First /@ positiveArgumentIntervals)
                  ]
               ]
             ]
         ]


(* ::Input:: *)
(*{positiveIntervals[n^2 + 12, n], positiveIntervals[2n+3, n]}*)


(* ::Input:: *)
(*positiveIntervals[2n+3, n]*)


nontrivialIntervals[expr_Piecewise, j_] := 
Module[{fullRange}, 
Which[MatchQ[expr, HoldPattern[Piecewise[{{_, _}}, 0]]],
      fullRange = Cases[expr[[1, 1, 2]], LessEqual[_?NumericQ, j, _?NumericQ] | Less[_?NumericQ, j, _?NumericQ]];
      If[fullRange =!= {},  Interval[{Min[First /@ fullRange], Max[Last /@ fullRange]}],
         positiveIntervals[expr[[1, 1, 2]], j]
        ],
     (* general piecewise *)
     True, Interval[{0, 20}]
     ]
    ]


angularMomentumRanges[f_, x_] :=  
Module[{pwExpr},
       pwExpr = f /. {ThreeJSymbol\[DoubleStruckCapitalH] -> ThreeJSymbol, SixJSymbol\[DoubleStruckCapitalH] -> SixJSymbol, ClebschGordan\[DoubleStruckCapitalH] -> ClebschGordan}; 
       Flatten[{nontrivialIntervals[pwExpr, x]}]
     ]


(* ::Input:: *)
(*angularMomentumRanges[ThreeJSymbol\[DoubleStruckCapitalH][{1, 0}, {1, 0}, {j, 0}],  j]*)


(* ::Input:: *)
(*angularMomentumRanges[ThreeJSymbol\[DoubleStruckCapitalH][{1, 0}, {10, 0}, {j, 0}],  j]*)


angularMomentumTable[f_, {x_, {xMin_, xMax_}}] :=
Module[{heldExpr, preTab, goodTab},
       heldExpr = Hold[f] /. {ThreeJSymbol\[DoubleStruckCapitalH] -> ThreeJSymbol, SixJSymbol\[DoubleStruckCapitalH] -> SixJSymbol, ClebschGordan\[DoubleStruckCapitalH] -> ClebschGordan};
       preTab = Table[{x, Quiet @ Check[ReleaseHold[heldExpr], $Failed]}, {x, xMin, xMax, 1/2}];
       goodTab = DeleteCases[preTab, {_, $Failed}] 
     ]


(* ::Input:: *)
(*angularMomentumTable[ThreeJSymbol\[DoubleStruckCapitalH][{1,0},{10,0},{j,0}], {j,{ 8, 12}}]*)


(* ::Input:: *)
(*angularMomentumTable[ThreeJSymbol\[DoubleStruckCapitalH][{1,0},{1,0},{j,0}],  {j,{ 0, 2}}]*)


ListPlotData["ListPlotCase", expr_, x_] :=
Module[{\[CapitalDelta] = 60,
        expr1, intFuns, args, positiveDomains, intSect, ranges, finalRange},  
  angularMomentumFunctionLocalFlagQ = MemberQ[expr, _ThreeJSymbol\[DoubleStruckCapitalH] | _SixJSymbol\[DoubleStruckCapitalH] | _ClebschGordan\[DoubleStruckCapitalH], {0, \[Infinity]}];
  expr1 = resolveSumsAndProducts[expr];
  intFuns = Cases[expr1, Alternatives @@ integerDomainFunctionPatterns[x], {0, \[Infinity]}]; 
  args = DeleteCases[
           Flatten[Select[List @@@ intFuns, MemberQ[#, x, {0, \[Infinity]}]&]] /.
           {(EulerPhi | MoebiusMu | CarmichaelLambda | MultiplicativeOrder | DivisorSigma |
             BernoulliB | EulerE | BellB | NorlundB | MertensFunction | Composite |
             StirlingS1 | StirlingS2 | PartitionsP | PartitionsQ | SquaresR)[args__] :>
            (Sequence @@ Cases[{args}, _?(PolynomialQ[#, x]&), {0, \[Infinity]}])},
                       _?NumericQ, {1}];
   positiveDomains = If[angularMomentumFunctionLocalFlagQ,
                        angularMomentumRanges[expr, x],
                        If[args === {}, {{0, 120}}, positiveIntervals[#, x]& /@ args]
                       ];
   If[MemberQ[positiveDomains, $Failed], $Failed,
      intSect = IntervalIntersection @@ positiveDomains;
      ranges = List @@ intSect;
      finalRange = 
      Which[ranges === {{-\[Infinity], \[Infinity]}}, {-\[CapitalDelta], \[CapitalDelta]},
            MatchQ[ranges, {___, {_, \[Infinity]}}], {ranges[[1, 1]], ranges[[1, 1]] + 2\[CapitalDelta]},
            MatchQ[ranges, {{-\[Infinity], _}, ___}], {ranges[[1, 2]] - 2 \[CapitalDelta], ranges[[1, 2]]},
            True,
            If[Max[ranges] - Min[ranges] < 12 \[CapitalDelta] && Total[Subtract @@@ ranges] < 2 \[CapitalDelta], 
               {Min[ranges], Max[ranges]},
               sizesAndRanges  = Sort[{Abs[Subtract @@ #], #}& /@ ranges, #1[[1]] > #2[[1]]&];
               sizesAndRanges[[1, 2]] 
               ]
            ];
       listPlotData = CalculateTimeConstrained[
                      If[angularMomentumFunctionLocalFlagQ, 
                         angularMomentumTable[expr, {x, finalRange}],
                         Table[{x, expr}, {x, Range @@ finalRange}]
                        ], 1.25, $Failed];
       If[listPlotData === $Failed, $Failed,
          goodListPlotData = Select[Chop[N[listPlotData]], Im[#[[2]]]===0 && Abs[#[[2]]] < 10^6&]
          ]
      ]
    ] 


(* special cases *)
ListPlotData["ListPlotCase", BernoulliB[x_], x_] := N[Table[{n, BernoulliB[n]}, {n, 36}]]


ListPlotData["ListPlotCase", KroneckerDelta[x_], x_] := N[Table[{n, KroneckerDelta[n]}, {n, -5, 5}]]


ListPlotData["ListPlotCase", KroneckerDelta[x_, m_Integer], x_] := N[Table[{n, KroneckerDelta[n]}, {n, m - 5, m + 5}]]


ListPlotData["ListPlotCase", KroneckerDelta[ m_Integer, x_], x_] := N[Table[{n, KroneckerDelta[n]}, {n, m - 5, m + 5}]]


ListPlotData["ListPlotCase", ChampernowneNumber[b_], b_] := N[Table[{b, ChampernowneNumber[b]}, {b, 2, 36}]]


(* ::Input:: *)
(*ListPlotData["ListPlotCase", KroneckerDelta[x],x]*)


(* ::Input:: *)
(*ListPlotData["ListPlotCase", (x!)^(x!),x]*)


(* ::Input:: *)
(*ListPlotData["ListPlotCase", MertensFunction[n],n]*)


(* ::Input:: *)
(*ListPlotData["ListPlotCase", DivisorSigma[0,n]/n,n]*)


(* ::Input:: *)
(*ListPlotData["ListPlotCase", ThreeJSymbol\[DoubleStruckCapitalH][{1,0},{1,0},{j,0}],j]*)


(* ::Input:: *)
(*ListPlotData["ListPlotCase", BernoulliB[n],n]*)


(* ::Input:: *)
(*ListPlotData["ListPlotCase", EulerPhi[n],n]*)


(* ::Input:: *)
(*ListPlotData["ListPlotCase", EulerPhi[n-20],n]*)


(* ::Input:: *)
(*ListPlotData["ListPlotCase", EulerPhi[n^2 - 3 n],n]*)


(* ::Input:: *)
(*ListPlotData["ListPlotCase", EulerPhi[n^2 + 12],n]*)


(* ::Input:: *)
(*ListPlotData["ListPlotCase", EulerPhi[n^2 + 12] + MoebiusMu[2n+3],n]*)


(* ::Input:: *)
(*ListPlotData["ListPlotCase", EulerPhi[n+MoebiusMu[2n+3]] ,n]*)


(* ::Input:: *)
(*ListPlot[%, Filling -> Axis]*)


(* ::Input:: *)
(*ListPlotData["ListPlotCase",EulerPhi[n^2 + 2 n - 1] ,n]*)


(* ::Input:: *)
(*ListPlot[%, Filling -> Axis]*)


(* ::Input:: *)
(*ListPlotData["ListPlotCase",EulerPhi[3+EulerPhi[n + 2]] ,n]*)


(* ::Input:: *)
(*ListPlot[%, Filling -> Axis]*)


(* ::Input:: *)
(*ListPlotData["ListPlotCase",(EulerPhi[n + 2]/EulerPhi[n - 2])^(MoebiusMu[n^2]+2) ,n]*)


(* ::Input:: *)
(*ListPlot[%, Filling -> Axis]*)


(* ::Input:: *)
(*ListPlotData["ListPlotCase",CarmichaelLambda[u] + Exp[-u MoebiusMu[u+2u^2]] EulerPhi[u + 2]  ,u]*)


(* ::Input:: *)
(*ListPlot[%, Filling -> Axis]*)


(* ::Input:: *)
(*ListPlotData["ListPlotCase",Sum[EulerPhi[k], {k, n}]  ,n]*)


(* ::Input:: *)
(*ListPlotData["ListPlotCase",Sum[EulerPhi[k]/(2. + k + n^2), {k, n}]  ,n]*)


(* ::Input:: *)
(*ListPlot[%, Filling -> Axis]*)


(* ::Input:: *)
(*ListPlotData["ListPlotCase",Sum[Prime[k], {k, n}]  ,n]*)


(* ::Input:: *)
(*ListPlot[%, Filling -> Axis]*)


(* ::Input:: *)
(*ListPlotData["ListPlotCase",DivisorSigma[2, n]/DivisorSigma[n, 2]  ,n]*)


(* ::Input:: *)
(*ListPlot[%, Filling -> Axis]*)


(* ::Input:: *)
(*ListPlotData["ListPlotCase",MultiplicativeOrder[11, n]/Sum[Abs[MoebiusMu[k]], {k, 2n+5}]  ,n]*)


(* ::Input:: *)
(*ListPlot[%, Filling -> Axis]*)


(* ::Section::Closed:: *)
(*Tests for MakeScaledPlots*)


(* ::Subsection::Closed:: *)
(*   Debug switch   *)


(* ::Subsubsection::Closed:: *)
(*   Debug issues   *)


(* ::Input:: *)
(*   print["dispatch"] = Print["dispatch to: ", #]&;   *)
(*   (*print["trig of rational"] = Print["trig of rational", #]&;*)   *)


(* ::Input:: *)
(*   (* print["preDispatch"] = Print["preDispatch ", #]&;*)   *)


(* ::Subsection::Closed:: *)
(*   Base test set   *)


(* ::Input:: *)
(*lastPPRFFPBag = {};*)


(* ::Input:: *)
(*UserMetaData[_] := {}*)


(* ::Input:: *)
(*realPartPlotStyle={Blue};*)
(*imaginaryPartPlotStyle=Red;*)


(* ::Input:: *)
(*RealImaginaryPlotSketch[res_] := *)
(*Quiet @ *)
(*Block[{UserMetaData},*)
(*UserMetaData["UserStyles" -> "2DMathPlot"]={};*)
(*Module[{s1}, *)
(*s1 =Cases[ (RealImaginaryPlot@@@ #)& /@  res, _Graphics, \[Infinity]];*)
(*DeleteCases[s1, "UserStyles" -> _, \[Infinity]] /. Hold[Plot] :> Plot*)
(* ]*)
(*]*)


(* ::Subsubsection::Closed:: *)
(*Quick overall test 1*)


(* ::Input:: *)
(*MakeScaledPlots[{Cos[Exp[-x^2]],x, Plot}]    //Timing*)


(* ::Input:: *)
(*MakeScaledPlots[{(1/x)-(1/((E^x)-1)),x, Plot}]    *)


(* ::Input:: *)
(*MakeScaledPlots[{1/(E^x-1)-1/x,x, Plot}]    *)


(* ::Input:: *)
(*MakeScaledPlots[{3^3^x,x, Plot}]    *)


(* ::Input:: *)
(*MakeScaledPlots[{ArcTan[1/(1+Exp[-x])],x, Plot}]    *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(1+E^(1+x)^(-1))^(-1),x, Plot}]    *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(x^3+3Cos[x])^-4,x, Plot}]    *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{HypergeometricPFQRegularized[{x,x,1-x},{1+x,1+x},0.5],x, Plot}]    *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{1 + 1/x^2,x, Plot}]    *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{x^99999,x, Plot}]    *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(7/9)^(t/10)-(7/9)^(t 0.1`),t, Plot}]    *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{JacobiDN[EllipticK[k^2],k],k, Plot}]    *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{I x^29,x, Plot}]    *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{I x^30,x, Plot}]    *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{1000x^0.001,x, Plot}]    *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[2^x],x, Plot}]    *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{BesselJ[1, x],x, Plot}]    *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Log[3^x],x, Plot}]     *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{1/(1+(E^(-x^2))),x, Plot}]     *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{1/(2x^2-2x-2(x^2-x)),x, Plot}]    *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[x],x, Plot}]    *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Tan[x],x, Plot}]    *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{80E^(-2*x)+92*(1-E^(-2*x)),x, Plot}]    *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(1/10)*(Log[1+Exp[-10]]-Log[Exp[-10*x^2]+Exp[-10]]),x, Plot}]    *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{1/(1-2^x),x, Plot}]    *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{KroneckerDelta[F],F, Plot}]    *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{KroneckerDelta[n],n, Plot}]    *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[10Pi/x], x, Plot}]    *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(x^2-x)/(x^2-3x+2), x, Plot}]    *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Log[1+E^(-x^2)], x, Plot}]    *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{ArcTan[x^5-4x^2+12], x, Plot}]     *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{SawtoothWave[5 Exp[x]]+Sin[3 SquareWave[Exp[x]]], x, Plot}]     *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Log[1/x]^2-Log[x]^2, x, Plot}]     *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{-((4*E^(2*x))/(1+E^x)^2)+(4*E^x)/(1+E^x), x, Plot}]     *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{{x/.9,x/.8,x}, x, Plot}]     *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{x^2/(x-1), x, Plot}]     *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{{x/.9,x/.8,x}, x, Plot}]     *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{{y==x/.9,y==x/.8,y==x}, x, Plot}]     *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{{x/.9,x/.8,x}, x, Plot}]     *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sqrt[1-Exp[-x^2]], x, Plot}]     *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{x+3, x, Plot}]     *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sqrt[1-Exp[-x^2]], x, Plot}]     *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(4*x^3)/(4+x^4), x, Plot}]     *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(1+3x)^4/x, x, Plot}]     *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(1+3x)^4/x, x, Plot}]     *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Tan[x], x, Plot}]     *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[x], x, Plot}]     *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Tan[x], x, Plot}]     *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(x+I*Sqrt[4-x^2])^4-4*(x+I*Sqrt[4-x^2])^2, x, Plot}]     *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(Sqrt[1/(x^2+x)]+Sqrt[x/(1+x)])^2, x, Plot}]     *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{((-144*3.14+4*x^2)/3)*(36-x^2)^(1/2), x, Plot}]     *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(x-2)^(2/3)/x^(2/3)+x^(1/3)/(x-2)^3^(-1), x, Plot}]     *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(1-x)/(-2+x^3), x, Plot}]     *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Floor[x/(9.35 - x)], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(2*x+1)/(x^2+x-20)+(3*x+8)/(x^2+3*x-10)-(3*x-10)/(x^2-6*x+8), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{x Sqrt[1 - x^4], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Abs[Cos[x-Pi/2]] (UnitStep[x+Pi]-UnitStep[x-Pi]), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(x!)^(x!), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[2*Pi*x*440]+Sin[2*Pi*x*450], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{x^0.99, x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{E^(Sqrt[x-1]/x), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{{3, 5, x}, x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{36-36*x^80/36^80, x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{((-x)*Log[2]+Log[2^x+3^x])/Log[3/2], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{x^Log[2] + x^Log[3], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(1+Exp[2 x])^3, x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sqrt[x^3+5*x^2]/x, x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[2x], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sum[1/(2 k+1) Sin[(2 k+1) Pi x/(1/2)],{k,0,10}], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{0.25/(x-I*2^0.5-2)+0.25/(x+I*2^0.5-2)+ -1/(2*x), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{1/(x-1), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{5+Mod[13 x,7], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{BesselJ[1,8333.3` x] BesselY[1,x]-BesselJ[1,x] BesselY[1,8333.3` x], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{((x-x^9+1)^2)^(1/3), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(x^2+Cos[3x])^(1/2), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{x^2 Sin[x], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{x^20000 Sin[x], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(2x^2-x)/(x+5), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Im[x^4 Sqrt[x^3-x]], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{x^20000 Sin[x], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[x^2], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[42 + x^2], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[x] Tan[(1+x)/(1+x-x^2+x^3-x^4)], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[x] + Cos[2x], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(x-3)/(Sqrt[2 x+3]-3), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(Sqrt[1+x^6]-Sqrt[1-2 x^4])/(x+x^2), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sqrt[1+(4 x^(1/3)-1/(16 x^(1/3)))^2], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[2 x^2]/(2 x^2), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[4565667*x], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{MinkowskiQuestionMarkFunction[x], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Subsubsection::Closed:: *)
(*Quick overall test 2*)


(* ::Input:: *)
(*MakeScaledPlots[{(3 Cos[(2x^2)/3])/(3+2 x^2), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Log[(1*x)^x/(1*x-1)^(x-1)]+Log[(x-1)^(x-1)/x^x], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{4*1.51*^-22*((0.263/x)^12-(0.263/x)^6), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Exp[9.3 10^-8 x], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(1-10^(-10))^((625*x)/11), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*(* should fail *)*)
(*MakeScaledPlots[{y==x+2, x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[x/(x + 1)], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[1/x], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[x]^23, x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{HeavisideLambda[x+Pi/2], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[Pi*Ceiling[x]], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(x - 9)/Sqrt[4*x^2 + 3*x + 2], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(x-9)/Sqrt[4*x^2+3*x+2], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{2*(1-x)*Cos[1/(1+x^2*(2-x)^2)] , x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Log[x+I]+Log[x-I]-Log[x^2+1] , x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{(x + 1)^2 - x^2 - 2x - 1, x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{(x^243)/(x^2-1), x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{1/(1+E^(-(x/0.5`))), x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{1/(1+E^(-2.8284271247461903` x)), x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{{x+I}, x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{{x+I,x^2+I^3}, x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{{x^2+I^3}, x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{2^2^x, x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{Exp[x^x], x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{Exp[-1/(2 x^2)], x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{t+4 (-1+Ceiling[t/4]), t, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[x]/x^2, x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{Exp[I/x^2], x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{SawtoothWave[x], x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{SquareWave[x], x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{TriangleWave[x], x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{SquareWave[3x] + x SawtoothWave[2x], x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{TriangleWave[3x + TriangleWave[x]]  , x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{456973959814329828038934528*x^9002, x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{3. + 0. x, x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{(-0.75 -0.7 I) Sqrt[x], x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{(x^2+2x-3)/(x+3), x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{{(4+x)/(1+x)}, x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{{(7.7+x)/(22.090000000000003+x)}, x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{1 - Exp[-InverseErf[z^2]], z, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{{x, x + 5, 2x -4, -x}, x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{{Sin[x], Cos[x], Tan[x]}, x, Plot}]*)


(* ::Input:: *)
(*MakeScaledPlots[{{x, x + 5}, x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[1/x^2], x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{(x^2+19x)/(2E^(1-x^2)-1), x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{x^Log[x]^Log[x]^Log[x]^Log[x], x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{Mod[x+3*Pi,2*Pi]-Pi, x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{(x)^.5+3*(x)^(-.5)-10*(x)^(-3/2), x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{(x)^(1/2)+3*(x)^(-1/2)-10*(x)^(-3/2), x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[1/x], x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{{E^(I*x)*(1-I*x)}, x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{2 z^5 - 2 I z^3 + 4z - 12 I,z, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{(z + I)^3,z, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{E^(I*x)*(1-I*x), x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{Zeta'[n], n, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{(1-Exp[-x^(-1)])^(-1), x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{{-2x-3, 2x-5}, x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{261/(1+E^(-((Log[81]/50)*(x-34)))), x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{x/2/(1-E^(-x/2)), x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{-1+E^x-x, x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{{E^x,1+x,3+2*x}, x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{-1+E^x-x, x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{(x^2-4)/(x^4-x), x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{6/x^4, x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{(1+E^(-((x-100)/2)))^(-1), x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{Log[Log[Log[Log[x]]]], x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{Log[Log[Log[Log[Log[x]]]]], x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{(x^5-x^4+34x^3+1)/(3x^2+17), x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{Sqrt[Pi/6]*(Cos[11/12]*FresnelC[(1+6*x)/Sqrt[6*Pi]]-FresnelS[(1+6*x)/Sqrt[6*Pi]]*Sin[11/12]), x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{ArcTan[Exp[x]], x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{Log[Sinh[Exp[x]]], x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{Sec[1/2]*Sin[x^2] + Tan[x^2], x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{PrimeZetaP[s] , s, Plot}]   *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{SphericalBesselJ[0, 1 + 1/x] , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Piecewise[{{Sqrt[Pi/2]/2,\[Omega]==-1||\[Omega]==1},{Sqrt[Pi/2],-1<\[Omega]<1}},0] , \[Omega], Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{(Sqrt[Pi/2] (Sign[1-\[Omega]]+Sign[1+\[Omega]]))/2 , \[Omega], Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{(-(1/2))*x*(x-2*Log[E^x]) , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{LegendreP[19,x-1] , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{x^100-1 , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{(1+n)^10 , n, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{(1+x)^10 , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{x^5+6x+1 , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{x + Sin[x] , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{(Sin[x] + 2)/(Cos[4 x +1] - 3) , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{ JacobiSN[0.6, x], x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{1/(Exp[x^2] + 1) , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{(x^2)!-1 , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{(x^2)!!-1 , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Gamma[3 x^3]-1 , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{-(1/(E^x*2))+E^x/2 , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Sinh[x] , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Sinh[Cosh[x]] , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{DawsonF[x] , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Subsubsection::Closed:: *)
(*Quick overall test 3*)


(* ::Input:: *)
(*MakeScaledPlots[{((x^3+5*x^2)^.5)/x, x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[2x], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sum[1/(2 k+1) Sin[(2 k+1) Pi x/(1/2)],{k,0,10}], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{0.25/(x-I*2^0.5-2)+0.25/(x+I*2^0.5-2)+ -1/(2*x), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{1/(x-1), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{5+Mod[13 x,7], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{BesselJ[1,8333.3` x] BesselY[1,x]-BesselJ[1,x] BesselY[1,8333.3` x], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{((x-x^9+1)^2)^(1/3), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(x^2+Cos[3x])^(1/2), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{x^2 Sin[x], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{x^20000 Sin[x], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(2x^2-x)/(x+5), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Im[x^4 Sqrt[x^3-x]], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{x^20000 Sin[x], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[x^2], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[42 + x^2], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[x] Tan[(1+x)/(1+x-x^2+x^3-x^4)], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[x] + Cos[2x], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(x-3)/(Sqrt[2 x+3]-3), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(Sqrt[1+x^6]-Sqrt[1-2 x^4])/(x+x^2), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sqrt[1+(4 x^(1/3)-1/(16 x^(1/3)))^2], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[2 x^2]/(2 x^2), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[4565667*x], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{MinkowskiQuestionMarkFunction[x], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(3 Cos[(2x^2)/3])/(3+2 x^2), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Log[(1*x)^x/(1*x-1)^(x-1)]+Log[(x-1)^(x-1)/x^x], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{4*1.51*^-22*((0.263/x)^12-(0.263/x)^6), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Exp[9.3 10^-8 x], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(1-10^(-10))^((625*x)/11), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*(* should fail *)*)
(*MakeScaledPlots[{y==x+2, x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[x/(x + 1)], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[1/x], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[x]^23, x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{HeavisideLambda[x+Pi/2], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[Pi*Ceiling[x]], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(x - 9)/Sqrt[4*x^2 + 3*x + 2], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(x-9)/Sqrt[4*x^2+3*x+2], x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{2*(1-x)*Cos[1/(1+x^2*(2-x)^2)] , x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Log[x+I]+Log[x-I]-Log[x^2+1] , x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{(x + 1)^2 - x^2 - 2x - 1, x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{(x^243)/(x^2-1), x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{1/(1+E^(-(x/0.5`))), x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{1/(1+E^(-2.8284271247461903` x)), x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{{x+I}, x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{{x+I,x^2+I^3}, x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{{x^2+I^3}, x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{2^2^x, x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{Exp[x^x], x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{Exp[-1/(2 x^2)], x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{t+4 (-1+Ceiling[t/4]), t, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[x]/x^2, x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{Exp[I/x^2], x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{SawtoothWave[x], x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{SquareWave[x], x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{TriangleWave[x], x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{SquareWave[3x] + x SawtoothWave[2x], x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{TriangleWave[3x + TriangleWave[x]]  , x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{456973959814329828038934528*x^9002, x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{3. + 0. x, x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{(-0.75 -0.7 I) Sqrt[x], x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{(x^2+2x-3)/(x+3), x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{{(4+x)/(1+x)}, x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{{(7.7+x)/(22.090000000000003+x)}, x, Plot}]      *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{1 - Exp[-InverseErf[z^2]], z, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{{x, x + 5, 2x -4, -x}, x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{{Sin[x], Cos[x], Tan[x]}, x, Plot}]*)


(* ::Input:: *)
(*MakeScaledPlots[{{x, x + 5}, x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[1/x^2], x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{(x^2+19x)/(2E^(1-x^2)-1), x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{x^Log[x]^Log[x]^Log[x]^Log[x], x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{Mod[x+3*Pi,2*Pi]-Pi, x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{(x)^.5+3*(x)^(-.5)-10*(x)^(-3/2), x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{(x)^(1/2)+3*(x)^(-1/2)-10*(x)^(-3/2), x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[1/x], x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{{E^(I*x)*(1-I*x)}, x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{2 z^5 - 2 I z^3 + 4z - 12 I,z, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{(z + I)^3,z, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{E^(I*x)*(1-I*x), x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{Zeta'[n], n, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{(1-Exp[-x^(-1)])^(-1), x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{{-2x-3, 2x-5}, x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{261/(1+E^(-((Log[81]/50)*(x-34)))), x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{x/2/(1-E^(-x/2)), x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{-1+E^x-x, x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{{E^x,1+x,3+2*x}, x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{-1+E^x-x, x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{(x^2-4)/(x^4-x), x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{6/x^4, x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{(1+E^(-((x-100)/2)))^(-1), x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{Log[Log[Log[Log[x]]]], x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{Log[Log[Log[Log[Log[x]]]]], x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{(x^5-x^4+34x^3+1)/(3x^2+17), x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{Sqrt[Pi/6]*(Cos[11/12]*FresnelC[(1+6*x)/Sqrt[6*Pi]]-FresnelS[(1+6*x)/Sqrt[6*Pi]]*Sin[11/12]), x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{ArcTan[Exp[x]], x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{Log[Sinh[Exp[x]]], x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{Sec[1/2]*Sin[x^2] + Tan[x^2], x, Plot}]*)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{PrimeZetaP[s] , s, Plot}]   *)


(* ::Input:: *)
(*(RealImaginaryPlotSketch@@ #[[1]])& /@  %*)


(* ::Input:: *)
(*MakeScaledPlots[{SphericalBesselJ[0, 1 + 1/x] , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Piecewise[{{Sqrt[Pi/2]/2,\[Omega]==-1||\[Omega]==1},{Sqrt[Pi/2],-1<\[Omega]<1}},0] , \[Omega], Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{(Sqrt[Pi/2] (Sign[1-\[Omega]]+Sign[1+\[Omega]]))/2 , \[Omega], Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{(-(1/2))*x*(x-2*Log[E^x]) , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{LegendreP[19,x-1] , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{x^100-1 , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{(1+n)^10 , n, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{(1+x)^10 , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{x^5+6x+1 , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{x + Sin[x] , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{(Sin[x] + 2)/(Cos[4 x +1] - 3) , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{ JacobiSN[0.6, x], x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{1/(Exp[x^2] + 1) , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{(x^2)!-1 , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{(x^2)!!-1 , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Gamma[3 x^3]-1 , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{-(1/(E^x*2))+E^x/2 , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Sinh[x] , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Sinh[Cosh[x]] , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{DawsonF[x] , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Subsubsection::Closed:: *)
(*Intentionally failing cases*)


(* ::Input:: *)
(*MakeScaledPlots[{x  , x, Plot}]   *)


(* ::Input:: *)
(*MakeScaledPlots[{x + 1  , x, Plot}]   *)


(* ::Subsubsection::Closed:: *)
(*Very special cases   *)


(* ::Input:: *)
(*MakeScaledPlots[{x^x^x, x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{x^x^x^x, x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Subsubsection::Closed:: *)
(*Constants   *)


(* ::Input:: *)
(*MakeScaledPlots[{Cos[6x]^2+Sin[6x]^2, x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{ 2  , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Subsubsection::Closed:: *)
(*Polynomials   *)


(* ::Input:: *)
(*MakeScaledPlots[{3x+5  , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]]& /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{x^2(x+1)^2 , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]]& /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{(3x+5)^5  , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]]& /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{x^2+2x+1 , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{ 6.7 + 9 x, x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{x^3+2x+1 , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{x^5+6x+1 , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{ x^4  , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{ x^2 -3x+2, x, Plot}]   *)


(* ::Input:: *)
(*MakeScaledPlots[{ x^5, x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{ x^5+x^4-3 x^2 + 5 x -6, x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{ x^12 - 4 x + 3, x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{ Product[x-k,{k,12}], x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{x^40 + 3x - 1, x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{HermiteH[60, x], x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{ChebyshevT[20, x], x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{x^200, x, Plot}]   *)


(* ::Input:: *)
(*MakeScaledPlots[{x^200, x, Plot}]   *)


(* ::Input:: *)
(*MakeScaledPlots[{x^3+2x+1, x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{x^3-2x+1, x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{x^5+x, x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{x^3+2x+1, x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*(* should give only one plot *)*)
(*MakeScaledPlots[{(1 + x)^30, x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{LegendreP[10, x], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Subsubsection::Closed:: *)
(*Rational functions   *)


(* ::Input:: *)
(*MakeScaledPlots[{(x^2-4)/(x+2), x,Plot}]   *)


(* ::Input:: *)
(*MakeScaledPlots[{(-4+x^2)/(-x+x^4), x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{1/x, x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{1/(x + x^2), x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{(1-x)/(1 + x), x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{(1-x)/(1 + x^3+3x), x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{ChebyshevT[12, x]/ChebyshevT[11, x], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{1/(x^3-5x-1), x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{(x^5-x^4+34x^3+1)/(3x^2+17), x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Subsubsection::Closed:: *)
(*Rational linear trigonometric functions   *)


(* ::Input:: *)
(*MakeScaledPlots[{ Sin[x], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{ Sin[3 x] + Sin[5x], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[x] + Cos[2x], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[2 x] + Sin[3 x], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{(Sin[x+1]^3 + 1)/(Tan[5 x] + 2) + Cot[x + 1], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{1/(Sin[x] + 3) + Tan[2x]+Tan[3x]+Tan[7x]/4, x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Subsubsection::Closed:: *)
(*Rational linear exponential functions   *)


(* ::Input:: *)
(*MakeScaledPlots[{Exp[x]+2, x,Plot}]   *)


(* ::Input:: *)
(*MakeScaledPlots[{1/(3 + Exp[x]), x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{1/(1 - Exp[x]), x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Exp[3 x]/(Exp[-x] + 2 Exp[x]), x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Exp[x^2]/(1+ Exp[x^2]), x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Subsubsection::Closed:: *)
(*Rational trigonometric functions   *)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[x^2], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{(Sin[x+1]^3 + 1)/(Tan[2 x] + 2) + Cot[x^2+ 1], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[x^4 + 3 x -1], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[Sqrt[2] x] + Cos[Sqrt[3] x], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Subsubsection::Closed:: *)
(*Periodic functions   *)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[Sin[Sin[Cos[x]]]], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{1/Sqrt[2 + Cos[x]^2], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{1/Sqrt[4 + Cos[2x]^2 + Sin[3 x/5]^3], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Subsubsection::Closed:: *)
(*Algebraic functions   *)


(* ::Input:: *)
(*MakeScaledPlots[{ Sqrt[ x ] + x^(1/3), x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Sqrt[ x + (x + 1)^(1/3)], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Sqrt[ x - 2 + Sqrt[x^3 - 3]] + x^4, x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{(x + 1)^(1/3) - 3 (x^-1 - 1)^(2/3), x,Plot}]   *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sqrt[x-1]/(1+Sqrt[2 x]),x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Sqrt[x - 1]/(1 + Sqrt[2 x]), x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{(2+x)/Sqrt[9+x^2], x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{(1+x^2)/((1-x^2)*Sqrt[1+x^4]), x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{ Sqrt[(1+x^2)/(1+x^4)], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{(x-3)/(Sqrt[2 x+3]-3), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{(Sqrt[1+x^6]-Sqrt[1-2 x^4])/(x+x^2), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{((1+x-x^9)^2)^(1/3), x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Subsubsection::Closed:: *)
(*Trigonometric of rational   *)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[1/x], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[x/(x^2+1)], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[x/(x^2-1)], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[1/(x^2-1)], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[x/(x^3-1)], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Tan[x^4/(x^3-1)], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{1/(Sin[x]+1), x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[x] Tan[1/x], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[x/(1 - x)] Tan[1(2+3x)/(4 x^2 + 2)], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Subsubsection::Closed:: *)
(*Trigonometric of algebraics   *)


(* ::Input:: *)
(*MakeScaledPlots[{Cos[Sqrt[2-3x]] , x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Subsubsection::Closed:: *)
(*Logs of algebraics   *)


(* ::Input:: *)
(*MakeScaledPlots[{Log[(x + 1)/(x - 1)]/x , x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{x^2 Log[x + Sqrt[x]] , x, Plot}]    *)


(* ::Input:: *)
(*MakeScaledPlots[{x^2 Log[ (Sqrt[x] + 1)/(x^2 - 3)] , x, Plot}]    *)


(* ::Subsubsection::Closed:: *)
(*Table look-up cases   *)


(* ::Input:: *)
(*MakeScaledPlots[{Exp[x], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{PolyLog[2, x], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[x]/Exp[x^3], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Exp[x]/2, x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Exp[x]/2 - 3, x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Exp[-x^2], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{AiryAi[x], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{JacobiSN[x+3, 3], x,Plot}]   *)


(* ::Input:: *)
(*MakeScaledPlots[{1/Log[2 x+4], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{ArcCosh[x], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{x^(1/x), x,Plot}]   *)


(* ::Input:: *)
(*MakeScaledPlots[{AiryAi[Sqrt[x]], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Sinc[x], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Zeta[1/2 + I x], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{AiryAi[x] + AiryBi[x+1], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Gamma[x], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Gamma[x]Gamma[1-x], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Subsubsection::Closed:: *)
(*Contains trigonometric functions   *)


(* ::Input:: *)
(*   (* crashes on Mac *)   *)
(*MakeScaledPlots[{ Sin[x + x^12 Sin[x]], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Subsubsection::Closed:: *)
(*Analytic   *)


(* ::Input:: *)
(*MakeScaledPlots[{FresnelC[x], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Gamma[x], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{BesselJ[0,x], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Subsubsection::Closed:: *)
(*Complex components   *)


(* ::Input:: *)
(*MakeScaledPlots[{ Re[Sinh[I x + 3]], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{ Re[Sinh[I x + Cos[x]]], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Subsubsection::Closed:: *)
(*Elementary functions   *)


(* ::Input:: *)
(*MakeScaledPlots[{Cosh[x^2 + Sin[x]] + Log[ Sin[x]], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Subsubsection::Closed:: *)
(*Piecewise functions   *)


(* ::Input:: *)
(*MakeScaledPlots[{Floor[x^2 + 1], x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Subsubsection:: *)
(*Table desperate   *)


(* ::Subsubsection::Closed:: *)
(*General*)


(* ::Input:: *)
(*MakeScaledPlots[{(x^2)! - 1 ,x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Subsubsection::Closed:: *)
(*Tan rescaled plot*)


(* ::Input:: *)
(*MakeScaledPlots[{(x^2)! - Tan[x] ,x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Subsubsection::Closed:: *)
(*Fall through   *)


(* ::Input:: *)
(*MakeScaledPlots[{x^x^x, x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Subsubsection::Closed:: *)
(*ListPlots*)


(* ::Input:: *)
(*MakeScaledPlots[{GoldbachFunction[n],n,Plot}] *)


(* ::Input:: *)
(*MakeScaledPlots[{SquaresR[n, 4],n,Plot}] *)


(* ::Input:: *)
(*% /. Hold[ListPlot]["DataList"-> l_, rest___] :> ListPlot[l, rest]*)


(* ::Input:: *)
(*MakeScaledPlots[{MertensFunction[n],n,Plot}] *)


(* ::Input:: *)
(*% /. Hold[ListPlot]["DataList"-> l_, rest___] :> ListPlot[l, rest]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sum[MertensFunction[k], {k, n}]/EulerPhi[n^2],n,Plot}] *)


(* ::Input:: *)
(*% /. Hold[ListPlot]["DataList"-> l_, rest___] :> ListPlot[l, rest]*)


(* ::Input:: *)
(*MakeScaledPlots[{DivisorSigma[0,n]/n,n,Plot}] *)


(* ::Input:: *)
(*% /. Hold[ListPlot]["DataList"-> l_, rest___] :> ListPlot[l, rest]*)


(* ::Input:: *)
(*MakeScaledPlots[{AiryAiZero[n],n,Plot}] *)


(* ::Input:: *)
(*% /. Hold[ListPlot]["DataList"-> l_, rest___] :> ListPlot[l, rest]*)


(* ::Input:: *)
(*MakeScaledPlots[{EulerPhi[n],n,Plot}] *)


(* ::Input:: *)
(*% /. Hold[ListPlot]["DataList"-> l_, rest___] :> ListPlot[l, rest]*)


(* ::Input:: *)
(*MakeScaledPlots[{RamanujanTau[n],n,Plot}] *)


(* ::Input:: *)
(*% /. Hold[ListPlot]["DataList"-> l_, rest___] :> ListPlot[l, rest]*)


(* ::Input:: *)
(*MakeScaledPlots[{StieltjesGamma[n],n,Plot}] *)


(* ::Input:: *)
(*% /. Hold[ListPlot]["DataList"-> l_, rest___] :> ListPlot[l, rest]*)


(* ::Input:: *)
(*MakeScaledPlots[{EulerPhi[x], x,Plot}] *)


(* ::Input:: *)
(*% /. Hold[ListPlot]["DataList"-> l_, rest___] :> ListPlot[l, rest]*)


(* ::Input:: *)
(*MakeScaledPlots[{BernoulliB[x], x,Plot}] *)


(* ::Input:: *)
(*% /. Hold[ListPlot]["DataList"-> l_, rest___] :> ListPlot[l, rest]*)


(* ::Input:: *)
(*MakeScaledPlots[{EulerPhi[n^2 - 3 n], n,Plot}] *)


(* ::Input:: *)
(*% /. Hold[ListPlot]["DataList"-> l_, rest___] :> ListPlot[l, rest]*)


(* ::Input:: *)
(*MakeScaledPlots[{(EulerPhi[n + 2]/EulerPhi[n - 2])^(MoebiusMu[n^2]+2), n,Plot}] *)


(* ::Input:: *)
(*% /. Hold[ListPlot]["DataList"-> l_, rest___] :> ListPlot[l, rest]*)


(* ::Input:: *)
(*MakeScaledPlots[{Prime[n] PrimePi[n], n,Plot}] ;*)


(* ::Input:: *)
(*% /. Hold[ListPlot]["DataList"-> l_, rest___] :> ListPlot[l, rest]*)


(* ::Input:: *)
(*MakeScaledPlots[{Prime[n] PrimePi[n], n,Plot}] ;*)


(* ::Input:: *)
(*% /. Hold[ListPlot]["DataList"-> l_, rest___] :> ListPlot[l, rest]*)


(* ::Input:: *)
(*MakeScaledPlots[{Sum[Prime[k], {k, n}], n,Plot}] ;*)


(* ::Input:: *)
(*% /. Hold[ListPlot]["DataList"-> l_, rest___] :> ListPlot[l, rest]*)


(* ::Input:: *)
(* MakeScaledPlots[{ThreeJSymbol\[DoubleStruckCapitalH][{1,0},{1,0},{j,0}] , j, Plot}]   *)


(* ::Subsubsection::Closed:: *)
(*Mixed plots and listplots*)


(* ::Input:: *)
(*MakeScaledPlots[{Fibonacci[n], n,Plot}] *)


(* ::Input:: *)
(*% /. {Hold[ListPlot]["DataList"-> l_, rest___] :> ListPlot[l, rest],*)
(*       Hold[Plot] -> Plot}*)


(* ::Subsubsection::Closed:: *)
(*List of functions   *)


(* ::Input:: *)
(*MakeScaledPlots[{{3^x, (-3)^x}, x, Plot}]      *)


(* ::Input:: *)
(*RealImaginaryPlotSketch[%]*)


(* ::Input:: *)
(*MakeScaledPlots[{{Sin[x], Cos[x]}, x, Plot}]      *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{{Sin[x], Cos[x],2 Sin[x] Cos[x]}, x, Plot}]      *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{{1-x^2, -1+x^2, x^4/20}, x, Plot}]      *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{{x,x^2, x^3}, x, Plot}]      *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{{x,2 x}, x, Plot}]      *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{{x,8 x,27 x,64 x,125 x,216 x,343 x,512 x,729 x,1000 x}, x, Plot}]      *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{{(4+x)/(1+x),(7.7+x)/(22.090000000000003+x)}, x, Plot}]      *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{{Sqrt[Cos[x]],Sqrt[Sin[x]]}, x, Plot}]      *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(* MakeScaledPlots[{{7x+17+(-1)^(x+1),3x+1.5+0.5*(-1)^(x+1)}, x, Plot}]      *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{{Sqrt[Cos[x]],Sqrt[Sin[x]]}, x, Plot}]      *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*  MakeScaledPlots[{{E^(-1+x),E^(1-x)}, x, Plot}]      *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*  MakeScaledPlots[{{1,x,x^2,x^3}, x, Plot}]      *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*  MakeScaledPlots[{{x, x^2}, x, Plot}]      *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*  MakeScaledPlots[{{x^2, -x^2}, x, Plot}]      *)


(* ::Input:: *)
(*  MakeScaledPlots[{{LegendreP[0, x], LegendreP[1, x], LegendreP[2, x], LegendreP[3, x]}, x, Plot}]      *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{{Sin[x], Cos[x]}, x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*   GenerateGraphicsFromHeldPlots @ %%   *)


(* ::Input:: *)
(*MakeScaledPlots[{{Sin[x], Cos[x], Tan[x], Cot[x]}, x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*   GenerateGraphicsFromHeldPlots @ %%   *)


(* ::Input:: *)
(*MakeScaledPlots[{{Sin[x], Cos[x], 2, Cot[x]}, x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*   GenerateGraphicsFromHeldPlots @ %%   *)


(* ::Input:: *)
(*MakeScaledPlots[{{AiryAi[x], Sin[x]}, x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*   GenerateGraphicsFromHeldPlots @ %%   *)


(* ::Input:: *)
(*MakeScaledPlots[{{1, Sin[x], Exp[x]}, x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*   GenerateGraphicsFromHeldPlots @ %%   *)


(* ::Input:: *)
(*MakeScaledPlots[{{Sin[x], Sin[2x], Sin[3 x]}, x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*   GenerateGraphicsFromHeldPlots @ %%   *)


(* ::Input:: *)
(*MakeScaledPlots[{{(1/(1+x))^(1/x),0.6}, x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Subsubsection::Closed:: *)
(*Bug-inspired*)


(* ::Input:: *)
(*MakeScaledPlots[{8x^9895-334x^-32, x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Sqrt[(1+x^2)/(1+x^4)], x, Plot}]    *)


(* ::Input:: *)
(*MakeScaledPlots[{-((x^3*Gamma[3/4,-x^4])/(4*(-x^4)^(3/4))), x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{SphericalBesselJ[0, 1 + 1/x], x, Plot}]    *)


(* ::Input:: *)
(*MakeScaledPlots[{SinIntegral[Exp[x]], x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[Exp[x]], x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{PrimePi[x], x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Fibonacci[n], n, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{x^2 + x^4 + 2 Sin[3 x], x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Expand[(1 + x)^10], x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Exp[-x^2/2]/2 + 3, x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Log[(x + 1)/(x - 1)]/x, x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{x+Sqrt[1+x^2], x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{BesselJ[2, x], x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{BesselY[2, x], x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{BesselI[2, x], x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{BesselK[2, x], x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{x^AiryAi[x], x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{Sin[x] Tan[1/x], x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{BesselJ[0,1/(x+1)^2]+AiryAi[x^2], x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{(1/2)*Sqrt[Pi]*Erfi[x], x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{(1/2)*Sqrt[Pi]*Erfi[Sqrt[x]], x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{ Erf[x], x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{ JacobiSN[0.6, x], x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{ JacobiDN[0.6, x^2], x, Plot}]    *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{AiryAi[x] + AiryBi[x] + 2 , x, Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Subsubsection::Closed:: *)
(*Various (might work in more than one class)   *)


(* ::Input:: *)
(*MakeScaledPlots[{(1 + Exp[x])/(1 - Exp[-x^2]), x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*MakeScaledPlots[{1/(x^100 - Exp[x]), x,Plot}]   *)


(* ::Input:: *)
(*ReleaseHold[First[#]] & /@ %   *)


(* ::Subsubsection::Closed:: *)
(*Top level function test*)


(* ::Input:: *)
(*Block[{sow = Sow},*)
(*Reap[*)
(*PlotterScanner[..., Hold[TrigToExp[Sin[x]]], ...] ]*)
(*     ]*)


(* ::Subsection::Closed:: *)
(*   Debug test   *)


(* ::Subsubsection::Closed:: *)
(*   All methods   *)


(* ::Input:: *)
(*lastPPRFFPBag = {};*)


(* ::Input:: *)
(*UserMetaData[_] := {}*)


(* ::Input:: *)
(*debugMemoryQ = False;*)


(* ::Input:: *)
(*ttMax = 1;   *)
(*testFunction =4*1.51*^-22*((0.263/x)^12-(0.263/x)^6);     *)
(*testFunction =Abs[Cos[x-Pi/2]] (UnitStep[x+Pi]-UnitStep[x-Pi]);*)
(*testFunction =x Sqrt[1 - x^4];*)
(*testFunction = (1+1/(200*x))^200*x;*)
(*testFunction = -((4*E^(2*x))/(1+E^x)^2)+(4*E^x)/(1+E^x);*)
(*testFunction =SawtoothWave[5 Exp[x]]+Sin[3 SquareWave[Exp[x]]];*)
(*testFunction =Sin[10 Pi/x];*)
(*testFunction =1/(1-2^x);*)
(*testFunction =(1/10)*(Log[1+Exp[-10]]-Log[Exp[-10*x^2]+Exp[-10]]);*)
(*testFunction =Tan[x];*)
(*testFunction =1/(1+(E^(-x^2)));*)
(*testFunction =Log[3^x];*)
(*testFunction =BesselJ[4, x];*)
(*testFunction =1000x^0.001;*)
(*testFunction =JacobiDN[EllipticK[x^2],x];*)
(*testFunction =x^99999;*)
(*testFunction =Exp[Sqrt[x]];*)
(*testFunction =(1+E^(1+x)^(-1));*)
(*testFunction = x*((1.2)^(50/x))-x;*)
(*testFunction = 3^3^x; *)
(*testFunction =ArcTan[1/(1+Exp[-x])];*)
(*testFunction=With[{q=x},(q^(1/5)*QPochhammer[q,q^5]*QPochhammer[q^4,q^5])/(QPochhammer[q^2,q^5]*QPochhammer[q^3,q^5])];*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "Constant", ttMax]   // Timing*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "Polynomial", ttMax]    // Timing  *)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "RationalFunction", ttMax]      // Timing*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "RationalLinearTrigonometric", ttMax]      // Timing*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "RationalLinearFresnel", ttMax]      // Timing*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "RestrictedFunction", ttMax]      // Timing*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "RationalLinearExponential", ttMax]      // Timing*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "RationalLinearExponentialIntegrated", ttMax]      // Timing*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "RationalTrigonometric", ttMax]      // Timing*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "ComplexComponents", ttMax]     // Timing *)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "Periodic", ttMax]      // Timing*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "Algebraic", ttMax]      // Timing*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "TrigonometricsOfRational", ttMax]      // Timing*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "TableLookUp", ttMax]      // Timing*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "TrigonometricsOfAlgebraic", ttMax]      // Timing*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "LogOfAlgebraic", ttMax]      // Timing*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "Elementary", ttMax]      // Timing*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "ContainsTrigonometrics", ttMax]      // Timing*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "AnalyticOscillatory", ttMax]      // Timing*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "AnalyticOscillatoryAtOrigin", ttMax]      // Timing*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "Piecewise", ttMax]      // Timing*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "SuggestOtherPlotType", ttMax]      // Timing*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "NDSolvable", ttMax]      // Timing*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "SingularPoints", ttMax]      // Timing*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "Monotonic", ttMax]      // Timing*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "TanRescaledPlot", ttMax]      // Timing*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "General", ttMax]      // Timing*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "TableDesperate", ttMax]      // Timing*)


(* ::Input:: *)
(*classTry[testFunction, x, Null, "FallThrough", ttMax]      // Timing*)


(* ::Subsubsection::Closed:: *)
(*   Quick random test   *)


(* ::Input:: *)
(*   MakeScaledPlots[{(x^2)! - 1 ,x,Plot}]   *)


(* ::Input:: *)
(*   ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*   MakeScaledPlots[{Exp[x]/2+3 ,x,Plot}]   *)


(* ::Input:: *)
(*   ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*   MakeScaledPlots[{Exp[x]/2 ,x,Plot}]   *)


(* ::Input:: *)
(*   ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*   MakeScaledPlots[{Sin[x] Exp[-x^3] ,x,Plot}]   *)


(* ::Input:: *)
(*   ReleaseHold[First[#]] & /@ %   *)


(* ::Subsection::Closed:: *)
(*   Various   *)


(* ::Input:: *)
(*   MakeScaledPlots[{JacobiSN[0.6, x], x,Plot}]   *)


(* ::Input:: *)
(*   MakeScaledPlots[{TrigToExp[Sin[x]], x,Plot}]   *)


(* ::Input:: *)
(*   MakeScaledPlots[{Exp[I x], x,Plot}]   *)


(* ::Input:: *)
(*   MakeScaledPlots[{HypergeometricPFQRegularized[{1,.5},{2},1/x], x,Plot}]   *)


(* ::Input:: *)
(*   MakeScaledPlots[{HypergeometricPFQRegularized[{1,.5},{2},1/x], x,Plot}]   *)


(* ::Input:: *)
(*   MakeScaledPlots[{y''[x], x,Plot}]   *)


(* ::Input:: *)
(*   MakeScaledPlots[{-x Cos[x] + Sin[x], x,Plot}]   *)


(* ::Input:: *)
(*   ReleaseHold[First[#]] & /@ %   *)


(* ::Input:: *)
(*   MakeScaledPlots[{Floor[x^2 + 1], x,Plot}]   *)


(* ::Input:: *)
(*   Abs[f[x]]   *)


(* ::Input:: *)
(*   MakeScaledPlots[{ Sin[x + x^12 Sin[x]], x,Plot}]   *)


(* ::Input:: *)
(*   PolynomialQ[Gamma[x], x]   *)


(* ::Input:: *)
(*   SuggestOtherPlotTypeQ[Gamma[x], x]   *)


(* ::Input:: *)
(*   Plot[10^10^x,{x,-1,1}]   *)


(* ::Input:: *)
(*   MakeScaledPlots[{4*Exp[2/x], x, Plot}]    *)


(* ::Input:: *)
(*   MakeScaledPlots[{E^(3 x) Sinh[7+5 x], x, Plot}]    *)


(* ::Input:: *)
(*   MakeScaledPlots[{(2+x)/Sqrt[9+x^2], x, Plot}]   *)


(* ::Input:: *)
(*   MakeScaledPlots[{g[x], x,Plot}]   *)


(* ::Input:: *)
(*   x == Tan[\[Xi]]   *)
(*      *)


(* ::Input:: *)
(*   f[x_] := Sin[x]   *)


(* ::Input:: *)
(*   D[f[x], x] /. x -> Tan[\[Xi]]   *)


(* ::Input:: *)
(*   DSolve[{F'[\[Xi]] == 1/D[Tan[\[Xi]], \[Xi]] Cos[Tan[\[Xi]]]}, F, \[Xi]]   *)


(* ::Input:: *)
(*   nds=NDSolve[{1/D[Tan[\[Xi]], \[Xi]]  F'[\[Xi]] == Cos[Tan[\[Xi]]], F[0]==0}, F, {\[Xi], 0, 2}]   *)


(* ::Input:: *)
(*   Plot[F[\[Xi]] /. nds[[1]], {\[Xi], 0, 2}]   *)


(* ::Input:: *)
(*   ParametricPlot[{Tan[\[Xi]], F[\[Xi]] /. nds[[1]]}, {\[Xi], 0, 2}, AspectRatio -> 1]   *)


(* ::Input:: *)
(*   ParametricPlot[{Tan[\[Xi]], Cos[Tan[\[Xi]]]}, {\[Xi], 0, 2}, AspectRatio -> 1,   *)
(*   PlotPoints-> 30]   *)


(* ::Input:: *)
(*   ListPlot[Table[{Tan[\[Xi]], Cos[Tan[\[Xi]]]}, {\[Xi], 0, Pi/2, Pi/2/1000}], AspectRatio -> 1, PlotRange ->{{0, 100}, All}]   *)


(* ::Input:: *)
(*   MakeScaledPlots[{ArcCosh[x], x,Plot}]   *)


(* ::Input:: *)
(*   MakeScaledPlots[{Sinc[x], x,Plot}]   *)


(* ::Input:: *)
(*   MakeScaledPlots[{AiryAi[x], x,Plot}]   *)


(* ::Input:: *)
(*   MakeScaledPlots[{AiryAi[Sqrt[x]], x,Plot}]   *)


(* ::Chapter:: *)
(*epilog*)


End[];


EndPackage[];
