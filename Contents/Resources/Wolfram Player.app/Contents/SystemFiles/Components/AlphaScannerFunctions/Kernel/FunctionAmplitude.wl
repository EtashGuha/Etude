(* ::Package:: *)

(* ::Chapter:: *)
(*prolog*)


BeginPackage["AlphaScannerFunctions`", {"AlphaScannerFunctions`CommonFunctions`"}];


FunctionAmplitude::usage= "Computes the amplitude of a periodic function.";


Begin["`FunctionAmplitude`Private`"];


(* ::Chapter:: *)
(*main code*)


ClearAll[canonicalizePhasor]
canonicalizePhasor[expr_, x_]:= expr //.{
	Cos[arg_]:> I Sin[arg],
	Sin[arg_]:> Exp[I arg],
	Exp[I(arg_ + delta_)]:> Exp[I delta]~Inactive[Times]~Exp[I arg] /; FreeQ[delta, x]
}


ClearAll[FunctionAmplitude]
FunctionAmplitude[expr_, x_Symbol] /; phasorQ[expr, x]:= Module[
	{
		y=t2s[expr],
		props, amp
	},
Print["t2s[expr] ", y];
	props=phasorProperties[y,x];
Print["phasorProperties[y, x] ", props];
	amp="Amplitude"/.props
]


(* ::Subsection::Closed::*)
(*phasor properties*)
Clear[IntervalMean, intervalMean, ExponentialQ, t2s, phasorQ, phase, phasorQ, phasePlot, phasorProperties]

phasorProperties[y:(a_.*(f:(Sin|Cos))[b_.*x_+c_.]+d_.),x_]/;FreeQ[{a,b,c,d},x]:=phasorProperties[y,x]={"Period"->2Pi/b,"Amplitude"->a,"Midline"->d,"Range"->d+a{-1,1},"Phase"->If[NumericQ[#],Mod[#+Pi,2 Pi]-Pi,#]&@((Pi/2*Boole[f===Cos]-c)/b),"Frequency"->b/(2 Pi)}
phasorProperties[y_,x_]/;phasorQ[y,x]:=phasorProperties[t2s[y],x]
phasorProperties[__]:={}

phase[args__]:="Phase"/.phasorProperties[args]

phasePlot[y_,x_]/;phasorProperties[y,x]=!={}:=With[{hl=#1/(2 GoldenRatio)},Plot[{y,y/.x->x+#1}//Evaluate,{x,#1-#2,#1+#2},PlotStyle->{{Thick,Blue},{Thin,Blue}},Prolog->{{Thin,Black,Line[{{#1,#4-#3},{#1,#4+#3}}]},{Thick,Darker[Red],Opacity[.7],Line[{{{#1,#4-hl},{#1,#4+hl}},{{0,#4-hl},{0,#4+hl}},{{0,#4},{#1,#4}}}]}}]]&@@({"Phase","Period","Amplitude","Midline"}/.phasorProperties[y,x])

amplitudePlot[y_,x_]/;phasorProperties[y,x]=!={}:=With[{hl=#3/(2 GoldenRatio)},Plot[y,{x,-#2,#2},PlotStyle->{{Thick,Blue},{Thin,Blue}},Prolog->{{Thin,Black,Line[{{-#2,#4},{#2,#4}}]},{Thick,Opacity[.7],Darker[Red],Line[{{{#1+#2/4,#4},{#1+#2/4,#3+#4}},{{#1+#2/4-hl,#3+#4},{#1+#2/4+hl,#3+#4}},{{#1+#2/4-hl,#4},{#1+#2/4+hl,#4}}}]}}]]&@@({"Phase","Period","Amplitude","Midline"}/.phasorProperties[y,x])

IntervalMean[args__]:=TimeConstrained[intervalMean[args],$singlesteptimelimit,$Failed]

intervalMean[y_,{x_,Interval[{-\[Infinity],\[Infinity]}]}]:=If[NumericQ@Limit[y,x->Infinity]&&NumericQ@Limit[y,x->Infinity],Limit[Integrate[y,{x,-a,a}]/(2a),a->\[Infinity]],Indeterminate]
intervalMean[y_,{x_,Interval[{-\[Infinity],a_}]}]:=Limit[Integrate[y,{x,b,a}]/(a-b),b->-\[Infinity]]
intervalMean[y_,{x_,Interval[{a_,\[Infinity]}]}]:=Limit[Integrate[y,{x,a,b}]/(b-a),b->\[Infinity]]
intervalMean[y_,{x_,Interval[{a_,b_}]}]:=Integrate[y,{x,a,b}]/(b-a)
intervalMean[y_,{x_,Interval[a__]}]:=Mean[intervalMean[y,{x,Interval[#]}]&/@{a}]


t2s[y_]:=t2s[y]=ExpandAll[TrigReduce[y]]
possiblePhasorQ[y_]:=MatchQ[t2s[y],_.*(_Sin|_Cos)+_.]
phasorQ[y_,x_]:=MatchQ[t2s[y],(a_.*(Sin|Cos)[b_.*x+c_.]+d_.)/;FreeQ[{a,b,c,d},x]]

Clear[allvars]
allvars[expr_]:=allvars[expr]=({"Independents","Parameters"}/.chooseVariables[expr,"ReturnParameters"->True])

choosePhasorVariable[expr_,specifiedVars_: {}]/;possiblePhasorQ[expr]:=choosePhasorVariable[expr,specifiedVars]=With[{l=Length[#]},With[{lw=LengthWhile[#,!phasorQ[expr,#]&]},If[l===lw,$Failed,#[[lw+1]]]]]&[Join[specifiedVars,##]&@@allvars[expr]]

choosePhasorVariable[__]=$Failed;


(* ::Chapter::Closed:: *)
(*epilog*)


End[];


EndPackage[];
