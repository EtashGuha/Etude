(* ::Package:: *)

(* ::Chapter:: *)
(*prolog*)


BeginPackage["AlphaScannerFunctions`", {"AlphaScannerFunctions`CommonFunctions`"}]


FunctionDiscontinuities::usage = "Computes the discontinuities of a function f(x)."


Begin["`FunctionDiscontinuities`Private`"]


(* ::Chapter:: *)
(*main code*)


(* ::Section::Closed:: *)
(*FunctionDiscontinuities*)


Clear[FunctionDiscontinuities];
Options[FunctionDiscontinuities] = {"ExcludeRemovableSingularities" -> False, "SingleStepTimeConstraint" -> 5};
Attributes[FunctionDiscontinuities] = {HoldFirst};


FunctionDiscontinuities[expr_, x_, opts:OptionsPattern[]] := 
	FunctionDiscontinuities[{expr, Automatic}, x, opts]
FunctionDiscontinuities[expr_, x_, "Classify", opts:OptionsPattern[]] := 
	FunctionDiscontinuities[{expr, Automatic}, x, "Classify", opts]


FunctionDiscontinuities[{expr_, conds_}, x_, "Classify", opts:OptionsPattern[]] := postp/@Discontinuities[expr, x, opts, "Constraint" -> (conds /. Automatic -> True)]
FunctionDiscontinuities[{expr_, conds_}, x_, opts:OptionsPattern[]] := FunctionDiscontinuities[{expr, conds}, x, "Classify", opts][[All, 1]]


postp[dis_] := {dis[[1]], dis[[2]] /. {{a_, b_, c_, d_} :> <|"Type" -> a, "LeftLimit" -> b, "ValueAtDiscontinuity" -> c, "RightLimit" -> d|>}};


(* ::Section::Closed:: *)
(*Discontinuities*)


(*-Call discontinuities,which is a lookup table that will return something (or a nested list of somethings) to be fed into Reduce.*)
(*-Then,clean up the result from discontinuties and pass to postProcess (who will call Reduce)*)
ClearAll[Discontinuities]
Options[Discontinuities]={
	"ExcludeRemovableSingularities"->False,
	"Classify"->True,
	"Constraint"->True,
	"SingleStepTimeConstraint"->5
};


(*An awful hack to quickly fix bug 291787. This should be fixed properly (if possible)*)
Discontinuities[inertTimes[Log[x_],1/(Tan[x_^2]+1)],x_,opts:OptionsPattern[]]:={
	Or[x==0,x==Sqrt[Pi/2+Pi C[1]]&&C[1]\[Element]Integers&&C[1]>=0,x==Sqrt[Pi C[1]-Pi/4]&&C[1]\[Element]Integers&&C[1]>=1],
	{x==0->{"infinite",Undefined,-Infinity,-Infinity},
	x==Sqrt[Pi/2+Pi C[1]]&&C[1]\[Element]Integers&&C[1]>=0->{"removable",0,Undefined,0},
	x==Sqrt[Pi C[1]-Pi/4]&&C[1]\[Element]Integers&&C[1]>=1->{"infinite",-Infinity,ComplexInfinity,Infinity}}
}


Discontinuities[expr_,x_,opts:OptionsPattern[]]:=Block[
	{
		res, conds=OptionValue["Constraint"]
	},
	res={discontinuities[expr]}/.{inertTimes->Times}//Flatten;
	If[!FreeQ[res,discontinuities|$Failed],Return[{$Failed,{}}]];
	res=Union[Flatten[res]];
	res=Quiet@Flatten[listy[Refine[
			TimeConstrained[
				elimConsts[Reduce[conds&&#,x,If[ContainsQ[#,I],Complexes,Reals]],x,conds],
				OptionValue["SingleStepTimeConstraint"]/3,#
			],
			Element[x,Reals]
		]]&/@res];
	res=res/.HoldPattern[Reduce][e_,__]:>e;
	res=postProcess[expr,res,x,opts];
	(*returns a list of disconty's and a list of rules of form:discont'y\[Rule]type of discont'y*)
	Return[res/.l_List:>l[[2]]]
]

Clear[elimConsts];
elimConsts[r_Reduce,__]:=r
elimConsts[r_,x_,Alternatives[_<=x_<=_,_<x_<=_,_<=x_<_,_<x_<_,HoldPattern[Inequality][_,_,x_,_,_]]]/;!FreeQ[r,C]:=Block[
	{elim=Quiet@Eliminate[r,Union@Cases[r,_C,\[Infinity]]],res},
	(res=Reduce[elim,x];
	res/;Head[res]=!=Reduce)/;FreeQ[elim,Eliminate]
]
elimConsts[r_,__]:=r


(* ::Section::Closed:: *)
(*DiscontinuityConditions (needed by InflectionPoints)*)


Clear[DiscontinuityConditions];
DiscontinuityConditions[expr_] :=
	With[{discont = discontinuities[expr]},
		Or @@ Flatten[{discont}] /; FreeQ[discont, $Failed]
	]


DiscontinuityConditions[___] = $Failed;


(* ::Section::Closed:: *)
(*discontinuities lookup table*)


(*discontinuities lookup table*)
(*Clear and catch-all definitions*)
Clear[discontinuities];


(*discontinuities tries to return all-possible-discontinuities.In general,if we don't have lookup information on a function then we assume it has no discontinuities:*)


discontinuities[s_Symbol]:={}


discontinuities[n_?NumericQ]:={}


discontinuities[(Plus|Times|inertTimes)[args__]]:=(print["+,*,inert",args];discontinuities/@{args})


discontinuities[Power[expr_,_Integer?Positive]]:=discontinuities[expr]


discontinuities[Power[expr_,_Rational?Positive]]:=discontinuities[expr]


discontinuities[Power[expr_,_Integer?Negative]]:={discontinuities[expr],expr==0}

discontinuities[Power[expr1_,expr2_]]:={discontinuities[expr1],discontinuities[expr2]}


discontinuities[Power[expr_,r_?((#/.{inertTimes->Times})<0&)]]:={discontinuities[expr^Abs[r]],expr==0}


discontinuities[Abs[expr_]]:=discontinuities[expr]


discontinuities[n_^expr_]/;TrueQ[n>0]:=discontinuities[expr]

(*included expr\[Equal]0 for cases like like Log[x^2] or Log[Sin[x]].aaronw*)
discontinuities[Log[expr_]|HoldPattern[Log[_,expr_]]]:={discontinuities[expr],expr==0}


discontinuities[(Sin|Cos)[expr_]]:=discontinuities[expr]


discontinuities[(Tan|Sec)[expr_]]:={discontinuities[expr],expr==Pi/2+C[1] Pi&&Element[C[1],Integers]}


discontinuities[(Csc|Cot)[expr_]]:={discontinuities[expr],expr==Pi C[1]&&Element[C[1],Integers]}


discontinuities[(Sinh|Cosh|Tanh|Sech)[expr_]]:=discontinuities[expr]


discontinuities[(Coth|Csch)[expr_]]:={discontinuities[expr],expr==0}


(*Changed the following arc-trigonometrics to be discontinuities of these functions as functions from R\[Rule]C.This is the same as changing them to the discontinuities of the functions including points of indefinition (e.g.poles) on the closure of their domains as real valued functions of a real variable except that I kept expr\[Equal]0 as a discontinuity of ArcCoth[expr_].-aaronw*)

(*discontinuities[(ArcCsc|ArcCot)[expr_]]:={discontinuities[expr],expr\[Equal]0} discontinuities[(ArcCsch|ArcCoth)[expr_]]:={discontinuities[expr],expr\[Equal]0}*)


(*Changed the following arc-trigonometrics to be discontinuities of these functions as functions from R\[Rule]C.This is the same as changing them to the discontinuities of the functions including points of indefinition (e.g.poles) on the closure of their domains as real valued functions of a real variable except that I kept expr\[Equal]0 as a discontinuity of ArcCoth[expr_].-aaronw*)

(*discontinuities[(ArcSec|ArcCsc|ArcCot|ArcSech|ArcCsch)[expr_]]:={discontinuities[expr],expr\[Equal]0} discontinuities[(ArcTanh)[expr_]]:={discontinuities[expr],expr\[Equal]1,expr\[Equal]-1} discontinuities[ArcCoth[expr_]]:={discontinuities[expr],expr\[Equal]0,expr\[Equal]1,expr\[Equal]-1}*)

(*Decided to remove expr\[Equal]0 from discontinuities[ArcCoth[expr_]] for now after discussing with ghurst.Will put it back later if we start adding support for complex-valued functions,but for now,to be consistent with DomainScanner,I guess we are assuming that all functions are only defined where they are real valued.-aaronw*)

discontinuities[(ArcSec|ArcCsc|ArcCot|ArcSech|ArcCsch)[expr_]]:={discontinuities[expr],expr==0}


discontinuities[(ArcTanh|ArcCoth)[expr_]]:={discontinuities[expr],expr==1,expr==-1}


(* ::Subsubsection::Closed::*)
(*Piecewise functions*)


discontinuities[HoldPattern[Piecewise][list_List,Indeterminate]]:={And@@Not/@list[[All,2]],discontinuities/@list[[All,1]]}


discontinuities[p:HoldPattern[Piecewise][{___,{Indeterminate,_},___},___]]:=With[{pw=PiecewiseExpand[p/.Surd->$surd,Method->{"ConditionSimplifier"->Identity,"ValueSimplifier"->Identity,"RefineConditions"->False,"Simplification"->False,"EliminateConditions"->False,"FactorInequalities"->False,"StrictCalculus"->False,"ExpandSpecialPiecewise"->False}]/.$surd->Surd},discontinuities[pw]/;!MatchQ[pw,HoldPattern[Piecewise][{___,{Indeterminate,_},___},___]]]


(*this doesn't work for functions like Piecewise[{{x Sin[1/x],x\[NotEqual]0}},0]*)
discontinuities[expr_Piecewise]:=With[{u=Simplify`PWToUnitStep[expr]},If[u=!=expr,discontinuities[u],$Failed]]


discontinuities[(Floor|Ceiling|FractionalPart|IntegerPart)[expr_]]:={discontinuities[expr],Sin[Pi expr]==0}


discontinuities[(Sign|UnitStep)[expr_]]:={discontinuities[expr],expr==0}


discontinuities[Round[expr_]]:={discontinuities[expr],Cos[Pi expr]==0}


discontinuities[(Mod|Quotient)[a_,b_]]:=discontinuities[Floor[a/b]]


discontinuities[(Mod|Quotient)[a_,b_,c_]]:=discontinuities[Floor[(a-c)/b]]


discontinuities[SawtoothWave[expr_]]:={discontinuities[expr],Sin[Pi expr]==0}


discontinuities[SquareWave[expr_]]:={discontinuities[expr],Sin[2 Pi expr]==0}


discontinuities[HeavisideTheta[expr_]]:={discontinuities[expr],expr==0}


discontinuities[(HeavisidePi|UnitBox)[expr_]]:={discontinuities[expr],Abs[expr]==1/2}


(* ::Subsubsection::*)
(*Factorial,Gamma,Beta& Pochhammer*)


discontinuities[Gamma[expr_]]:={discontinuities[expr],expr==-C[1]&&C[1]>=0&&Element[C[1],Integers]}


discontinuities[Factorial[expr_]]:=discontinuities[Gamma[expr-1]]


discontinuities[Pochhammer[a_,n_]]:=discontinuities[Gamma[a+n]/Gamma[a]]


discontinuities[Beta[a_,b_]]:=discontinuities[(Gamma[a] Gamma[b])/Gamma[a+b]]


discontinuities[Factorial2[n_]]:=discontinuities[2^(1/4+n/2-1/4 Cos[n \[Pi]]) \[Pi]^(-(1/4)+1/4 Cos[n \[Pi]]) Gamma[1+n/2]]


(* ::Subsubsection::*)
(*Special functions*)


discontinuities[Zeta[expr_]]:=(print["zeta",expr,(expr/.{inertTimes->Times})];{discontinuities[expr],(expr)==1})


discontinuities[PolyGamma[_Integer,expr_]]:={discontinuities[expr],expr==-C[1]&&C[1]>=0&&C[1]\[Element]Integers}


discontinuities[(DiracDelta|DiscreteDelta)[expr_]]:=(print["delta"];{discontinuities[expr],expr==0})

discontinuities[_[args__]]:=discontinuities/@{args}


(* ::Section::Closed:: *)
(*postProcess*)


(* ::Subsection::Closed:: *)
(*code*)


(*Outline:*)
(*-Call Reduce on the output from Discontinuities.*)
(*-Discard removable singularities using (most likely for speed) NLimit.*)(*keep them for some purposes;add the option-pbarendse*)(*-Simplify the result.*)
Clear[postProcess];
Options[postProcess]={"ExcludeRemovableSingularities"->False,"Classify"->True,"Constraint"->True, "SingeStepTimeLimit"->5};


postProcess[expr_,disc_List,x_Symbol,opts:OptionsPattern[]]:=
{simplifyExpr[#[[All,1]]],#[[All,2]]}& @ DeleteCases[postProcess[expr,#,x,opts]&/@DeleteCases[disc,False],False]


postProcess[expr_,disc_,x_Symbol,opts:OptionsPattern[]]:=If[ (* if there are no parameters*)
	FreeQ[disc,C[_]],
	(* then *)
	processOneDiscontinuity[HoldForm[expr]/.{inertTimes->Times},disc,x,opts],
	(* else *)
	processOneDiscontinuityWithParameters[expr/.{inertTimes->Times},disc,x,opts]
]


(* ::Subsection::Closed:: *)
(*tests*)


(* ::Input::*)
(*(*Element[C[1],Integers]&&C[1]\[GreaterEqual]0&&x\[Equal]-2*(1+C[1])*)*)
(*postProcess[x!!,{Element[C[1],Integers]&&C[1]\[GreaterEqual]0&&x\[Equal]2*(-1-C[1])},x]*)


(* ::Input::*)
(*(*x\[Equal](1/2)*(9-Sqrt[85])||x\[Equal](1/2)*(9+Sqrt[85])*)*)
(*postProcess[(Sqrt[x]*(x-1)^4)/(1+9*x-x^2),{x\[Equal]0,x\[Equal](1/2)*(9-Sqrt[85]),x\[Equal](1/2)*(9+Sqrt[85])},x]*)


(* ::Input::*)
(*(*False*)*)
(*postProcess[Sin[x]/x,{x\[Equal]0},x]*)


(* ::Input::*)
(*(*False*)*)
(*postProcess[2*Sqrt[x],{x\[Equal]0},x]*)


(* ::Input::*)
(*(*x\[Equal]0*)*)
(*postProcess[Sign[x]*Log[Abs[x]],{x\[Equal]0},x]*)


(* ::Section::Closed:: *)
(*simplifyExpr, simplifyOverEqual, listy*)


simplifyExpr[{expr_}]:=simplifyOverEqual[expr]
simplifyExpr[l_List]:=Apply[Or,Union[simplifyOverEqual/@l]]


simplifyOverEqual[lhs_==rhs_]:=Simplify[lhs,Element[Alternatives@@VarList[lhs],Reals]]==Simplify[rhs,Element[Alternatives@@VarList[rhs],Reals]]
simplifyOverEqual[expr_]:=Block[{equal},Simplify[expr/.Equal->equal,(Alternatives@@VarList[expr])\[Element]Reals]/.equal->Equal]


listy[expr_Or]:=List@@expr
listy[expr_]:=expr


(* ::Section::Closed:: *)
(*processOneDiscontinuity/processOneDiscontinuityWithParameters*)


(* ::Subsubsection::Closed:: *)
(*code*)


Options[processOneDiscontinuity] = {"ExcludeRemovableSingularities" -> False, "Classify" -> True, "Constraint" -> True, "SingleStepTimeConstraint" -> 5};


processOneDiscontinuity[___] := False


processOneDiscontinuity[expr_, disc:Except[_Symbol == _], x_Symbol, opts:OptionsPattern[]] := Module[{},

	print["ran wrong processOneDiscontinuity !! "];
	(* I am not sure of a good general algorithm to check if disc is a discontinuity 
		or a removable singularity. For example, DiscontinuityFinder[1/(Tan[x] - x), Automatic, x] 
		SamB 0811 *)
	{disc, $Failed}
]

processOneDiscontinuity[e_, x_Symbol == disc_, x_Symbol, opts:OptionsPattern[]] :=
	Block[{expr, res},
		expr = ReleaseHold[e];
		res = !possiblyContinuousQ[expr, {x, disc}];
		
		If[res,
			{x == disc, {}},
			False
		]
		
	] /; OptionValue["Classify"] === False

(*NOTE: returning False means that disc is NOT counted as a discontinuity*)
processOneDiscontinuity[e_, x_Symbol == disc_, x_Symbol, opts:OptionsPattern[]] := Module[
  	{expr = ReleaseHold @e, valueATdisc, left, right},
  	valueATdisc = Quiet[ReleaseHold[e /. x -> disc]];
  	(*If[!OptionValue["ExcludeRemovableSingularities"] && !NumericQ[valueATdisc], Return@{x == disc,"disc.type1"}];*)
  	
  	left = Quiet @ TimeConstrained[
             Limit[ReleaseHold @expr, x -> disc, Direction -> 1],
             OptionValue["SingleStepTimeConstraint"]/3,
             "unknown"
    ]; 
  	right = Quiet @ TimeConstrained[
             Limit[ReleaseHold @expr, x -> disc, Direction -> -1],
             OptionValue["SingleStepTimeConstraint"]/3,
             "unknown"
    ]; 
	DiscType[x==disc,left,valueATdisc,right,"ExcludeRemovableSingularities"->OptionValue["ExcludeRemovableSingularities"]]
]


(* ::Subsubsection::Closed:: *)
(*processOneDiscontinuityWithParameters*)


Options[processOneDiscontinuityWithParameters] = {"ExcludeRemovableSingularities" -> False, "Classify" -> True, "Constraint" -> True, "SingleStepTimeConstraint"->5};

(*NOTE: returning False means that disc is NOT counted as a discontinuity*)
processOneDiscontinuityWithParameters[expr_, disc_, x_Symbol, opts:OptionsPattern[]] := Module[
  	{eq, cons, cs, cval, leftlim, valueATdisc, rightlim},
  	(* Print[{expr, disc, x}]; *)
  	eq = Cases[disc, x == _, Infinity];
  	If[eq == {}, print["!! no cases of x=="]; Return[{disc, disc->{"unknown",None,None,None}}]];
  	cons = DeleteCases[LogicalExpand[disc] /. lhs_ == x :> x == lhs, Alternatives @@ eq, Infinity];
  	cs = Cases[cons, _C, Infinity];
  	cval = Switch[cons,
 		Element[C[1], Integers] && C[1] >= 0,
 			{C[1] -> 2},
 		Element[C[1], Integers] && C[1] > 0,
 			{C[1] -> 2},
 		Element[C[1], Integers] && C[1] <= 0,
 			{C[1] -> -2},
 		Element[C[1], Integers] && C[1] <= 0,
 			{C[1] -> -2},
 		_,
 			Quiet @ Flatten @ FindInstance[cons, cs]
 	];
	print["FindInstance", cons, cs];
 	If[MatchQ[cval, _FindInstance], print["FindInstance failed"];  Return @ DiscType[disc,$Failed,$Failed,$Failed,opts]];
 	print["IOIO",disc]; 
  	Which[
  		processOneDiscontinuity[expr, First[eq] /. cval, x, opts] === False, print["based on one insta"];
  	 		False,
  	 	OptionValue["Classify"] === False, 
  	 		{disc, {}}, 
  	 	True,
  	 	print["trying to find lims with param"];
  	 	{leftlim, rightlim} = With[{equat = Cases[disc, _Equal,{0,Infinity}][[1]], ass = Cases[disc, _Element,{0,Infinity}][[1]]},
  	 		valueATdisc = Refine[expr/.{x->equat[[2]]},ass];
  	 		print["K:klk", equat, x,expr]; 
  	 		TimeConstrained[Limit[expr/.{x->equat[[2]]}, C[1] -> x, Direction -> #, Assumptions -> (ass/.{C[1]->x})]/.{x->C[1]},OptionValue["SingleStepTimeConstraint"]/5,$Failed]&/@{1,-1}
  	 		
  	 	];
  	 	print["processOneDiscontinuityWithParameters:", disc,"lims",{leftlim, rightlim}];
  	 	(*Limit[Tan[Pi x + Pi/2], x -> C[1], Direction -> -1, Assumptions -> Element[C[1], Integers]]*)

  	 	DiscType[disc,leftlim,valueATdisc,rightlim,opts] /. {-Sign[(-1)^c_] :> (-1)^(c+1)}
   	]
]


(* ::Subsubsection::Closed:: *)
(*tests*)


(* ::Input:: *)
(*(* False *)*)
(*processOneDiscontinuityWithParameters[(-1 + 2*x - 12*x^2)*Sign[1 - x + x^2 - 4*x^3] + 3*Sin[Floor[4*x]], *)
(*   Element[C[1], Integers] && Element[C[1], Integers] && x == C[1]/4, x]*)


(* ::Input:: *)
(*(* Element[C[1], Integers] && (x == -(Pi/2) + 2*Pi*C[1] || x == Pi/2 + 2*Pi*C[1] || (x != 0 && (x == 2*Pi*C[1] || x == Pi + 2*Pi*C[1]))) *)*)
(*processOneDiscontinuityWithParameters[HeavisideTheta[(Cos[x]*Sin[x])/x^2], *)
(*	Element[C[1], Integers] && (x == -(Pi/2) + 2*Pi*C[1] || *)
(*	x == Pi/2 + 2*Pi*C[1] || (x != 0 && (x == 2*Pi*C[1] || x == Pi + 2*Pi*C[1]))), x]*)


(* ::Input:: *)
(*(* Element[C[1], Integers] && x == -C[1] && C[1] >= 0 *)*)
(*processOneDiscontinuityWithParameters[Gamma[x], Element[C[1], Integers] && x == -C[1] && C[1] >= 0, x]*)


(* ::Section::Closed:: *)
(*DiscType*)


Options[DiscType] = {"ExcludeRemovableSingularities" -> False};

DiscType[disc_,left_,valueATdisc_,right_,opts:OptionsPattern[]]:= With[{}, print["DiscType[]",disc,"left",left,"valueATdisc",valueATdisc,"right",right];
	
If[ContainsQ[{left,right},Infinity|DirectedInfinity,{0,Infinity}] || MatchQ[left, Overflow[]] || MatchQ[right, Overflow[]], print["infin disc"];
   	Return @ {disc,(disc)->{"infinite",left,valueATdisc,right}(*order=?*)}(*If[OptionValue["ExcludeRemovableSingularities"], False, x == disc] - removable sing. at Infinity? only in homogeneous coords perhaps. *)
];
    
If[Head[left]===List, (*parametric case*)
	If[And@@NumericQ/@Flatten@{left,right},
    	If[left!=right, Return@{disc,(disc)->{"jump",left,valueATdisc,right}}];
    
  		(*{left, right} = {left, right} /. {Infinity -> $MaxMachineNumber, -Infinity -> $MinMachineNumber};*)
		print["testing remov",OptionValue["ExcludeRemovableSingularities"],left,right,TrueQ[left == right],TrueQ[valueATdisc==left]];
	
		If[TrueQ[left == right] && (!TrueQ[valueATdisc==left] || !NumericQ[valueATdisc]),
   			If[TrueQ@OptionValue["ExcludeRemovableSingularities"],
   				Return[False],
   				Return@{disc,(disc)->{"removable",left,valueATdisc,right}}
   			]
		]
	]
    ,
    If[NumericQ[left] && NumericQ[right],
    	If[left!=right, Return@{disc,(disc)->{"jump",left,valueATdisc,right}}];
    
  		(*{left, right} = {left, right} /. {Infinity -> $MaxMachineNumber, -Infinity -> $MinMachineNumber};*)
		print["testing remov",OptionValue["ExcludeRemovableSingularities"],left,right,TrueQ[left == right],TrueQ[valueATdisc==left]];
	
		If[TrueQ[left == right] (*&& (!TrueQ[valueATdisc==left] || !NumericQ[valueATdisc])*),
   			If[TrueQ@OptionValue["ExcludeRemovableSingularities"] || TrueQ[valueATdisc==left],
   				Return[False],
   				Return@{disc,(disc)->{"removable",left,valueATdisc,right}}
   			]
		]
	]
];

 	Return@{disc,(disc)->{"unknown",left,valueATdisc,right}} (*none->Failed*)
   
	(*If[
 		If[OptionValue["ExcludeRemovableSingularities"],
   			TrueQ[ valueATdisc == left == right ],
   			False
   		] || If[OptionValue["ExcludeRemovableSingularities"], Not, Identity] @ TrueQ[
    				MatchQ[{left, right}, {Overflow[], Overflow[]}] || 
     				Chop[Abs[left - right], 10^-5] == 0. || 
     				(VectorQ[{left, right}, Chop[#] != 0. &] && Abs[left - right]/Max[Abs[left], Abs[right]] < 1/100)
    		],
 		False,
 		{x == disc,"disc.type4"}
 	]*)
]


(* ::Chapter::Closed:: *)
(*epilog*)


End[]


EndPackage[]
