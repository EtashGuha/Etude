(* ::Package:: *)

(* ::Chapter:: *)
(*prolog*)


Needs["NumericalCalculus`"]


BeginPackage["AlphaScannerFunctions`", {"AlphaScannerFunctions`CommonFunctions`", "NumericalCalculus`"}]


Asymptotes::usage= "Compute the asymptotes to a plane curves";


Begin["`Asymptotes`Private`"]


(* ::Chapter:: *)
(*main code*)


(* ::Subsection:: *)
(*Asymptotes*)


ClearAll[Asymptotes, asymptotesOfImplicitEquations, horizontalAsymptotes, verticalAsymptotes]
Asymptotes::usage = "Computes horizontal, vertical, oblique(slant or skew) and parabolic asymptotes.";


Options[Asymptotes]= {"SingleStepTimeConstraint" -> 5};


(* ::Subsubsection::Closed:: *)
(*implicit function case*)


Asymptotes[expr:Equal[lhs_, rhs_], indepVar_Symbol, depVar_Symbol, opts: OptionsPattern[]] /; 
	!(Head[lhs] === Symbol && Length[VarList[{lhs, rhs}]] == 1) && !MatchQ[lhs, _?usersymbolQ[_]] := Module[
	{
		vars, x, y, asym, display, asymFnX, asymFnY, asymSameQFunc,
		horiz, vert, parab, oblique, trueParab, other
	},

	vars = VarList[expr];
	If[Length[vars] != 2, 
		Return[],
		{x, y} = vars
	];
	
	asymFnX = TimeConstrained[
		asymptotesOfImplicitEquations[expr, {x, y}] /. $Failed -> {}, (* y[x] = <fn of x> *)
		5 OptionValue["SingleStepTimeConstraint"],
		{}
	];
	
	asymFnY = Reverse[#, 4]& @ TimeConstrained[
		asymptotesOfImplicitEquations[expr, {y, x}] /. $Failed -> {}, (* x[y] = <fn of y> *)
		5 OptionValue["SingleStepTimeConstraint"],
		{}
	];
	
	asymSameQFunc = Quiet[TrueQ[#1 === #2 || Eliminate[{#1, #2}, x] || #1 === Reverse[#2, 2]]]&;
	horiz = (DeleteDuplicates[Join[asymFnX[[1]], asymFnY[[2]]], asymSameQFunc] // combinePlusMinus // DeleteDuplicates // First[#, {}]&);
	vert = (DeleteDuplicates[Join[asymFnX[[2]], asymFnY[[1]]], asymSameQFunc] // combinePlusMinus // DeleteDuplicates // First[#, {}]&);
	parab = (DeleteDuplicates[Join[asymFnX[[3]], asymFnY[[3]]], asymSameQFunc] // combinePlusMinus // DeleteDuplicates // First[#, {}]&);

	oblique= Cases[parab /. $Failed -> {}, Alternatives[
		{depVar -> _?(PolynomialQ[#,indepVar]&&Exponent[#,indepVar]===1&) | PlusMinus[_?(PolynomialQ[#,indepVar]&&Exponent[#,indepVar]===1&)], indepVar -> _},
		{indepVar -> _?(PolynomialQ[#,depVar]&&Exponent[#,depVar]===1&)| PlusMinus[_?(PolynomialQ[#,depVar]&&Exponent[#,depVar]===1&)], depVar -> _}
	]];
	trueParab = Cases[parab /. $Failed -> {}, Alternatives[
		{depVar -> _?(PolynomialQ[#,indepVar]&&Exponent[#,indepVar]===2&), indepVar -> _},
		{indepVar -> _?(PolynomialQ[#,depVar]&&Exponent[#,depVar]===2&), depVar -> _}
	]];
	other= Complement[parab /. $Failed -> {}, Join[oblique, trueParab]];
	
	asym= <|"Horizontal" -> horiz, "Vertical" -> vert, "Oblique" -> oblique, "Parabolic" -> trueParab, "Other"-> other|> // DeleteCases[{}]
]


(* ::Subsubsection::Closed:: *)
(*main case*)


Asymptotes[exp_, indepVar_Symbol, depVar_Symbol, opts: OptionsPattern[]] := Module[
	{expr, horiz, vert, pmHoriz, pmVert, parabolic, pmParabolic, res, oblique, trueParab, other},

	expr = preprocessUserInput[exp];
	(*vars = VarList[expr];
	If[Length[vars] =!= 1, 
		Return[],
		x = First @ vars
	];*)

	horiz = horizontalAsymptotes[expr, indepVar, depVar, opts];
	pmHoriz = combinePlusMinus[horiz];
	
	vert = verticalAsymptotes[expr, indepVar, depVar, opts];
	pmVert = combinePlusMinus[vert];

	parabolic = parabolicAsymptotes[expr, indepVar, depVar, opts];
	pmParabolic = combinePlusMinus[parabolic];

	oblique= Cases[pmParabolic /. $Failed -> {}, Alternatives[
		{depVar -> _?(PolynomialQ[#,indepVar]&&Exponent[#,indepVar]===1&), indepVar -> _},
		{indepVar -> _?(PolynomialQ[#,depVar]&&Exponent[#,depVar]===1&), depVar -> _}
	]];
	trueParab = Cases[pmParabolic /. $Failed -> {}, Alternatives[
		{depVar -> _?(PolynomialQ[#,indepVar]&&Exponent[#,indepVar]===2&), indepVar -> _},
		{indepVar -> _?(PolynomialQ[#,depVar]&&Exponent[#,depVar]===2&), depVar -> _}
	]];
	other= Complement[pmParabolic /. $Failed -> {}, Join[oblique, trueParab]];
	
	res= <|"Horizontal" -> pmHoriz, "Vertical" -> pmVert, "Oblique" -> oblique, "Parabolic" -> trueParab, "Other"-> other|>;
	res= DeleteCases[res, $Failed | {}]
]


(* ::Subsubsection::Closed:: *)
(*typed-case*)


Asymptotes[expr_, indepVar_Symbol, depVar_Symbol, All, opts: OptionsPattern[]]:= Flatten[Values[Asymptotes[expr, indepVar, depVar, opts]], 1]
Asymptotes[expr_, indepVar_Symbol, depVar_Symbol, type_String, opts: OptionsPattern[]]:= First[Values[KeySelect[Asymptotes[expr, indepVar, depVar, opts], #===type&]], {}]


(* ::Subsection::Closed:: *)
(*selectBestForm, combinePlusMinus*)


selectBestForm[a_, b : Times[-1, _]] := a

selectBestForm[a : Times[-1, _], b_] := b

selectBestForm[a_, b_] := If[LeafCount[a] < LeafCount[b], a, b]

combinePlusMinus[$Failed]:= $Failed
combinePlusMinus[l_List] := Quiet[l //. {
		{a___, {as_, x_ -> la_}, b___, {as_, x_ -> lb_}, c___} :> {a, b, c, {as, Rule[x, PlusMinus[Abs[lb]]]}} /; (PossibleZeroQ[la + lb] || (la === Infinity && lb === -Infinity) || (la === -Infinity && lb === Infinity)),
		{a___, {y_ -> rLim_, x_ -> limPt_}, b___, {y_-> lLim_, x_ -> limPt_}, c___} :> {a, b, c, {y -> PlusMinus[rLim], Rule[x, limPt]}} /; (PossibleZeroQ[rLim + lLim] || (rLim === Infinity && lLim === -Infinity)),
		{a___, {y_ -> rLim_, x_ -> limPt_}, b___, {y_ -> lLim_, x_ -> limPt_}, c___} :> {a, b, c, {y -> PlusMinus[lLim], Rule[x, limPt]}} /; (PossibleZeroQ[rLim + lLim] || (rLim === -Infinity && lLim === Infinity))
}]


(* ::Subsection::Closed:: *)
(*horizontalAsymptotes*)


(* ::Subsubsection::Closed:: *)
(*code*)


Options[horizontalAsymptotes]= {"SingleStepTimeConstraint" -> 5};
horizontalAsymptotes[expr_, x_, y_, opts: OptionsPattern[]] := Block[{right, left, asym},
  	asym = Flatten[Reap[
    	left = Limit[expr, x -> -Infinity];
     	If[FreeQ[left, DirectedInfinity] && NumericQ[left] && Element[left, Reals],
      		Sow[{left, x -> -Infinity}]
      	];
     	right = Limit[expr, x -> Infinity];
     	If[FreeQ[right, DirectedInfinity] && NumericQ[right] && Element[right, Reals],
      		Sow[{right, x -> Infinity}]
      	]
	][[-1]], 1];
	If[asym === {}, Return[$Failed]];
	asym = MapAt[y->#&, asym, {All, 1}]
]


(* ::Subsubsection::Closed:: *)
(*tests*)


(* ::Input:: *)
(*horizontalAsymptotes[1/x, x]*)


(* ::Input:: *)
(*horizontalAsymptotes[(2*x^2 + 3*x + 1)/x^2, x]*)


(* ::Input:: *)
(*horizontalAsymptotes[(x^2 + 3*x + 1)/(4*x^2 - 9), x]*)


(* ::Input:: *)
(*horizontalAsymptotes[(x + 3)/(x^2 + 9), x]*)


(* ::Input:: *)
(*horizontalAsymptotes[1 - Exp[-2*x], x]*)


(* ::Input:: *)
(*horizontalAsymptotes[ArcTan[x], x]*)


(* ::Input:: *)
(*horizontalAsymptotes[(5 - 3*x^2)/(1 - x^2), x]*)


(* ::Input:: *)
(*horizontalAsymptotes[200/(1 + x)^2, x]*)


(* ::Subsection::Closed:: *)
(*processInfinities, infinityRules*)


processInfinities[expr_] := Block[{lims},
	lims = Select[expr, FreeQ[#, Complex]&];
	Which[
		lims === {},
			Return[{}],
		MatchQ[lims, {Infinity} | {-Infinity}],
			First @ lims,
		True,
			(lims /. C[_] -> 1) //. infinityRules
	]
]

infinityRules = {
					{-Infinity, Infinity} -> PlusMinus[Infinity], 
					{Infinity, -Infinity} -> PlusMinus[Infinity],
					{Infinity, Infinity} -> Infinity,
					{-Infinity, -Infinity} -> -Infinity,
					{a_, b_} /; MatchQ[a, Infinity] || MatchQ[b, Infinity] :> Infinity,
					{a_, b_} /; MatchQ[a, -Infinity] || MatchQ[b, -Infinity] :> -Infinity
				};


(* ::Subsection::Closed:: *)
(*verticalAsymptotes, checkVerticalAsymptotes*)


(* ::Subsubsection::Closed:: *)
(*code*)


checkVerticalAsymptotes[expr_, x_ -> p_] := 
	ContainsQ[{
		Assuming[Element[C[1], Integers], Limit[expr, x -> p, Direction -> -1]], 
		Assuming[Element[C[1], Integers], Limit[expr, x -> p, Direction -> 1]]
		}, DirectedInfinity]

Options[verticalAsymptotes] = {"SingleStepTimeConstraint" -> 5};
verticalAsymptotes[expr_, x_, y_, opts: OptionsPattern[]] := Block[{pts, otherPoints, res},
  	pts = Quiet @ TimeConstrained[Reduce[1/expr == 0, x, Reals], 2 OptionValue["SingleStepTimeConstraint"]];
  	otherPoints = Quiet @ DeleteCases[TimeConstrained[Reduce[# == 0, x], OptionValue["SingleStepTimeConstraint"]/5, False]& /@ Level[expr, {0, Infinity}], False]; 
      If[pts === $Aborted && otherPoints === {}, Return[$Failed]];
  	pts = Cases[{pts, otherPoints}, _?usersymbolQ == _, {0, Infinity}];
  	pts = DeleteDuplicates[Rule @@@ pts];
  	(* Check the points are really vertical asymptotes. SamB 0510 *)
  	pts = Cases[pts, pt_ /; checkVerticalAsymptotes[expr, pt]];
  	If[pts === {}, Return[$Failed]];
  	res = {
  		processInfinities[{
  			Assuming[Element[C[1], Integers], Limit[expr, #, Direction -> -1]], 
  			Assuming[Element[C[1], Integers], Limit[expr, #, Direction -> 1]]
  		}], #} & /@ pts;
  	res = DeleteCases[res, l_ /; ListQ[First[l]] || ContainsQ[l, _Complex]];
  	If[res === {}, Return[$Failed]];
  	MapAt[y->#&, res, {All, 1}]
]


(* ::Subsubsection::Closed:: *)
(*tests*)


(*
In[97]:= verticalAsymptotes[(x^2 + 3*x + 1)/(4*x^2 - 9), x]
Out[97]= {{Infinity, x -> -(3/2)}, {Infinity, x -> 3/2}}

In[98]:= verticalAsymptotes[Tan[t], t]
Out[98]= {{-Cot[Pi*C[1]], t -> Pi/2 + Pi*C[1]}}

In[99]:= verticalAsymptotes[2*x + 3/x, x]
Out[99]= {{Infinity, x -> 0}}

In[100]:= verticalAsymptotes[(3*x + 6)/((x + 2)*(x - 4)), x]
Out[100]= {{Infinity, x -> 4}}

In[101]:= verticalAsymptotes[(5 - 3*x^2)/(1 - x^2), x]
Out[101]= {{Infinity, x -> -1}, {-Infinity, x -> 1}}

In[103]:= verticalAsymptotes[(x^2 - 5 x + 6)/(x^3 - 3 x^2 + 2 x), x]
Out[103]= {{Infinity, x -> 0}, {-Infinity, x -> 1}}

In[397]:= verticalAsymptotes[x*Csc[x], x]
Out[397]= {{\[PlusMinus]Infinity, x -> 2*Pi*C[1]}, {\[PlusMinus]Infinity, x -> Pi + 2*Pi*C[1]}}
*)


(* ::Subsection::Closed:: *)
(*univariateRationalQ*)


univariateRationalQ[Power[expr_, _Integer]] := univariateRationalQ[expr]
univariateRationalQ[expr_Plus] := VectorQ[List @@ expr, univariateRationalQ]
univariateRationalQ[expr_Times] := VectorQ[List @@ expr, univariateRationalQ]
univariateRationalQ[expr_] := With[{vars = VarList[expr]}, Length[vars] <= 1 && PolynomialQ[expr, vars]]
univariateRationalQ[n_] /; NumericQ[n] = True;
univariateRationalQ[__] = False;

(*
In[88]:= univariateRationalQ[Pi/E/Sqrt[x] - (1 + 4 Pi)^(3/2) x^2]
Out[88]= False

In[89]:= univariateRationalQ[Pi/E /x - (1 + 4 Pi)^(3/2) x^2]
Out[89]= True
*)


(* ::Subsection::Closed:: *)
(*parabolicAsymptotes*)


Clear[parabolicAsymptotes];
Options[parabolicAsymptotes]= {"SingleStepTimeConstraint" -> 5};
parabolicAsymptotes[expr_, x_, y_, opts: OptionsPattern[]] /; univariateRationalQ[expr] := Block[{rat, num, den},
  	rat = Together[expr];
  	num = Numerator[rat];
  	den = Denominator[rat];
  	If[Exponent[num, x] - Exponent[den, x] >= 1,
   		{{y-> PolynomialQuotient[num, den, x], x->PlusMinus[Infinity]}},
   		$Failed
   	]
]

(*
In[158]:= parabolicAsymptotes[(x^2 + x + 1)/(x + 1), x]
Out[158]= x

In[159]:= parabolicAsymptotes[(2 x^3 + 4 x^2 - 9)/(3 - x^2), x]
Out[159]= -4 - 2 x

In[160]:= parabolicAsymptotes[(x^3 + 2 x^2 + 3 x + 4)/x, x]
Out[160]= 3 + 2 x + x^2

In[161]:= parabolicAsymptotes[(37 x^5 + 2 x^2 + 3 x + 4)/(x^2 - x - 1), x]
Out[161]= 113 + 74 x + 37 x^2 + 37 x^3

In[174]:= parabolicAsymptotes[(x^4 - 2 x^3 + 1)/x^2, x]
Out[174]= -2 x + x^2

In[175]:= parabolicAsymptotes[(1 - x)^3/x^2, x]
Out[175]= 3 - x
*)

parabolicAsymptotes[c_. (a_. x_^n_Integer + k_.)^m_., x_, y_, opts: OptionsPattern[]] /; n == 1/m && FreeQ[{a, k, c}, x] && Im[a] == 0 := If[
	(* if n is even *)
	EvenQ[n],
	(* asymptotes in both directions are the same *)
	{{y -> c Sqrt[a] Abs[x], x-> PlusMinus[Infinity]}},
	(* else direction of asymtote depends on sign of a *)
	If[ a > 0,
		{{y -> c Sqrt[a] x, x-> Infinity}},
		{{y -> -c Sqrt[-a] x, x-> -Infinity}}
	]
]

(*)
In[902]:= parabolicAsymptotes[Sqrt[x^2 + 1], x]
Out[902]= Abs[x]

In[907]:= parabolicAsymptotes[7 Power[x^4 + 8, (4)^-1], x]
Out[907]= 7 Abs[x]
*)

parabolicAsymptotes[c_. (x_^n_ + k_.)^m_., x_, y_, opts: OptionsPattern[]] /; FreeQ[{k, c}, x] := {{y -> c x^(m n), x->PlusMinus[Infinity]}}

(*
In[200]:= parabolicAsymptotes[Sqrt[1 + x^3], x]
Out[200]= x^(3/2)
*)

parabolicAsymptotes[Sqrt[a_. x_^2 + b_. ] + c_. x_, x_, y_, opts: OptionsPattern[]] /; FreeQ[{a, b, c}, x] && c < 0 && Sqrt[a] == -c := 
	{{y -> -2 Sqrt[a] x, x -> Infinity}}

(*
In[211]:= Table[Limit[(Sqrt[k x^2 + 1] - Sqrt[k] x) - (-2 Sqrt[k] x), x -> -Infinity], {k, 5}]
Out[211]= {0, 0, 0, 0, 0}
*)

parabolicAsymptotes[expr_, x_, y_, opts: OptionsPattern[]] /; !PolynomialQ[expr, x] := Block[
	{
		leftAsym, rightAsym,
		series, rat, poly, limit, nlimit
	},
    If[FreeQ[expr, x], Return[$Failed]];
	(* find right asymptote (x\[Rule] +Infinity*)
  	series = Quiet @ Series[ToRadicals[expr], {x, Infinity, 1}];
  	rightAsym = If[FreeQ[series, HoldPattern @ SeriesData[_, _, {__?NumericQ}, ___]],
   		$Failed,
   		rat = rat = Together[Normal[
   			series //. {
   				Times[f_, s_SeriesData] :> Normal[s, f]
   			}
   		]];
   		If[PolynomialQ[Numerator[rat], x] && PolynomialQ[Denominator[rat], x],
    		poly = Factor @ Simplify @ PolynomialQuotient[Numerator[rat], Denominator[rat], x];
    		limit = {Limit[poly - ToRadicals[expr], x -> Infinity], Limit[poly - ToRadicals[expr], x -> -Infinity]} /. Interval[__] :> 0;
    		If[ContainsQ[limit, 0], 
    			poly,
 				nlimit = Quiet @ TimeConstrained[Chop[#, 10^-3] & /@ {NLimit[poly - expr, x -> Infinity], NLimit[poly - expr, x -> -Infinity]},
    					OptionValue["SingleStepTimeConstraint"]
    			];
 				If[ContainsQ[limit, 0], 
 					poly, 
 					$Failed
 				]
 			],
    		$Failed
    	]
   	];
   	
   (* find left asymptote (x\[Rule] -Infinity*)
  	series = Quiet @ Series[ToRadicals[expr], {x, -Infinity, 1}];
  	leftAsym = If[FreeQ[series, HoldPattern @ SeriesData[_, _, {__?NumericQ}, ___]],
   		$Failed,
   		rat = Together[Normal[
   			series //. {
   				Times[f_, s_SeriesData] :> Normal[s, f]
   			}
   		]];	

   		If[PolynomialQ[Numerator[rat], x] && PolynomialQ[Denominator[rat], x],
    		poly = Factor @ Simplify @ PolynomialQuotient[Numerator[rat], Denominator[rat], x];
    		
    		limit = {Limit[poly - ToRadicals[expr], x -> Infinity], Limit[poly - ToRadicals[expr], x -> -Infinity]} /. Interval[__] :> 0;
    		If[ContainsQ[limit, 0], 
    			poly,
 				nlimit = Quiet @ TimeConstrained[Chop[#, 10^-3] & /@ {NLimit[poly - expr, x -> Infinity], NLimit[poly - expr, x -> -Infinity]},
    					OptionValue["SingleStepTimeConstraint"]
    			];
 				If[ContainsQ[limit, 0], 
 					poly, 
 					$Failed
 				]
 			],
    		$Failed
    	]
  	];
  	{
  		If[leftAsym === $Failed, Nothing, {y->leftAsym, x->-Infinity}],
  		If[rightAsym === $Failed, Nothing, {y->rightAsym, x->Infinity}]
  	} /. {} -> $Failed
]

(*
In[636]:= parabolicAsymptotes[Sqrt[1 + x^4], x]
Out[636]= x^2

In[637]:= parabolicAsymptotes[(x^3 + Sin[x])/x, x]
Out[637]= x^2

In[638]:= parabolicAsymptotes[Sqrt[Exp[-x^2] + (1 + x)^4], x](* Here's one that we don't do well on... *)
Out[638]= $Failed

In[639]:= parabolicAsymptotes[Sin[x] + (1 + x)^4, x]
Out[639]= (1 + x)^4

In[640]:= parabolicAsymptotes[Log[x] + (1 + x)^4, x]
Out[640]= $Failed

In[641]:= parabolicAsymptotes[Sqrt[(1 + x)^4 + x/(x (x + 1)^2)], x]
Out[641]= (1 + x)^2

In[642]:= parabolicAsymptotes[Sqrt[(1 + x)^4 + x/(x (x + 1)^2)] + x^10, x]
Out[642]= 1 + 2 x + x^2 + x^10

In[643]:= parabolicAsymptotes[(17 + 27 x + 3 x^2 - 9 x^3 + 5 x^4 + 12 x^5 + 4 x^6)/(2 + 3 x + x^2), x]
Out[643]= 9 - 3 x^2 + 4 x^4

In[644]:= parabolicAsymptotes[Sqrt[1 + Sqrt[x] + x^4], x]
Out[644]= x^2

In[645]:= parabolicAsymptotes[ArcTan[x] + x, x](* We had to pick one.... *)
Out[645]= 1/2 (\[Pi] + 2 x)
*)


(* ::Subsection::Closed:: *)
(*asymptotesOfImplicitEquations*)


Clear[asymptotesOfImplicitEquations];
Options[asymptotesOfImplicitEquations] = {"SingleStepTimeConstraint" -> 5};
asymptotesOfImplicitEquations[cx_. x_^2 + cy_. y_^2 == k_, {x_, y_} | {y_, x_}, opts: OptionsPattern[]] /; (cx cy < 0 && (NumericQ[k] || (usersymbolQ[k] && FreeQ[{x,y},k]))) := 
	{
		{}, (* no horizontal asymptotes *)
		{}, (* no vertical asymptotes *)
		{{{y -> Sqrt[Abs[cx]/Abs[cy]] x, x-> PlusMinus[Infinity]}, {y -> -Sqrt[Abs[cx]/Abs[cy]] x, x-> PlusMinus[Infinity]}}} (*2 oblique asymptotes*)
	}

asymptotesOfImplicitEquations[eqn_Equal, {x_, y_}, opts: OptionsPattern[]] := Module[
	{
		soln, fnsOfx, asymptotes, res,
		parabolic = {}, vertical = {}, horizontal = {}
	},
  	soln = TimeConstrained[Reduce[eqn, {x, y}, Reals], 2 OptionValue["SingleStepTimeConstraint"]];

  	If[MatchQ[soln, _Reduce | $Aborted], Return[$Failed]];
  	fnsOfx = DeleteDuplicates @ Cases[soln, e_Equal /; ContainsQ[Last[e], x] :> Last[e], {0, Infinity}];
  	
  	If[fnsOfx == {}, Return @ $Failed];
  	
    (* try to find parabolic asymptotes *)
    asymptotes = parabolicAsymptotes[#, x, y] & /@ fnsOfx;
    asymptotes = DeleteDuplicates[DeleteCases[asymptotes, $Failed]];
    parabolic = Select[asymptotes, FreeQ[N[#], Complex]&];

	(* try to find vertical asymptotes *)
	asymptotes = verticalAsymptotes[#, x, y] & /@ fnsOfx;
	asymptotes = DeleteCases[asymptotes, $Failed];
	vertical = Select[asymptotes, FreeQ[N[#], Complex]&];

	(* try to find horizontal asymptotes *)
	asymptotes = horizontalAsymptotes[#, x, y] & /@ fnsOfx;
	asymptotes = DeleteCases[asymptotes, $Failed];
	horizontal = Select[asymptotes, FreeQ[N[#], Complex]&];

	res = {horizontal, vertical, parabolic}
]
(*
In[431]:= asymptotesOfImplicitEquations[x^2 + 3 y^2 == (x y)^2, {x, y}]
Out[431]= {y == -1, y == 1}

In[432]:= asymptotesOfImplicitEquations[x^2/25 - y^2/36 == 1, {x, y}]
Out[432]= {y == -((6 x)/5), y == (6 x)/5}

In[27]:= asymptotesOfImplicitEquations[x y - y == x^2/8 + y^2, {x, y}]
Out[27]= {y == 1/4 (-2 + 2 Sqrt[2]) + 1/4 (2 - Sqrt[2]) x, y == 1/4 (-2 - 2 Sqrt[2]) + 1/4 (2 + Sqrt[2]) x}

In[99]:= asymptotesOfImplicitEquations[x^4/y^2 - (x y)^2 == 1/2, {y, x}](* note the ordering of the variables *)
Out[99]= {x == -y^2, x == y^2}

In[115]:= asymptotesOfImplicitEquations[x^2/y^2 + x y - 1/(x y) == 1/5, {y, x}]
Out[115]= {x == -y^3}

In[34]:= asymptotesOfImplicitEquations[x y == 1, {x, y}]
Out[34]= {x == 0}

In[35]:= asymptotesOfImplicitEquations[x^2 y + y == 2, {x, y}]
Out[35]= {y == 0}

In[27]:= asymptotesOfImplicitEquations[x^2 - y^3 == (x y)^2, {y, x}]
Out[27]= {y == -1, y == 1}

*)


(* ::Chapter::Closed:: *)
(*epilog*)


End[]


EndPackage[]
