(* ::Package:: *)

(* :Name: NumericalIntegrationAnalysis`Butcher` *)

(* :Title: Runge-Kutta Order Conditions and Butcher Tree Functions *)

(* :Author: Mark Sofroniou (original version by Jerry B. Keiper) *)

(* :Summary:
This package gives the order conditions that a Runge-Kutta method
must satisfy to be of a particular order. It works for both implicit
and explicit methods. The package also calculates and plots Butcher trees
and implements the functions of trees defined in Butcher's book.
Butcher's row and column simplifying conditions assist in the derivation
of high order methods. A more compact and efficient stage-independent
tensor notation has also been implemented. *)

(* :Context: NumericalIntegrationAnalysis`Butcher` *)

(* :Package Version: 2.2 *)

(* :Copyright: Copyright 1990-2007, Wolfram Research, Inc. *)

(* :History:
 Version 1.2 by Jerry B. Keiper, 1989.
 Updated by Jerry B. Keiper, December 1990.
 Updated by Jerry B. Keiper, August, November 1992. Implemented
 suggestions of Mark Sofroniou, Loughborough University.
 Re-written and updated by Mark Sofroniou, October 1993.
 Updated by Mark Sofroniou July 1994 (acknowledgements to
 A. Harrison and S. N. Papakostas for suggesting the tensor
 notation).
 Updated by Mark Sofroniou January 1998. Performance enhancements to
 numerical functions on trees.
 Added support for generating order conditions for continuous output
 Mark Sofroniou January 2004.
*)

(* :Keywords:
 Butcher trees, Runge-Kutta, order conditions. *)

(* :Source:
 John C. Butcher: The Numerical Analysis of Ordinary Differential
 Equations: Runge-Kutta and General Linear Methods, John Wiley &
 Sons, New York, 1987.
 J. D. Lambert: Numerical Methods for Ordinary Differential Systems,
 The Initial Value Problem, John Wiley & Sons, New York, 1991.
 Other useful references are:
 E. Hairer, S. P. Norsett and G. Wanner, Solving Ordinary Differential
 Equations I: Nonstiff Problems, Second edition, Springer Verlag,
 New York, 1993.
 E. Hairer, and G. Wanner, Solving Ordinary Differential Equations II:
 Stiff and Differential Algebraic Problems, Second edition, Springer Verlag,
 New York, 1996.
 *)

(* :Mathematica Version: 3.0 *)

(* :Limitation:
 This package only derives the order conditions; it does not try
 to solve them for the required coefficients.

 Combinatorial explosion effectively limits the order for which
 order conditions can be found, e.g., 7813 (highly nonlinear)
 conditions are required for a 12th-order method.

 Butcher's simplifying conditions have been added to partly
 address this issue. *)

(* :Discussion:
 The method is due to John C. Butcher and is explained in Chapter
 3 of Butcher's book. It relies heavily on graph theory and
 recursive decomposition of Butcher trees. Pattern matching is
 used extensively. Tensor notation (see Lambert, Chapter 5.12
 for its use in Albrecht's formalism) is more efficient and
 is used for evaluation of the order conditions with numeric stages. *)

If[Not@ValueQ[ButcherAlpha::usage],ButcherAlpha::usage =
"ButcherAlpha[tree] gives the number of ways of labeling the vertices of \
the tree tree with a totally ordered set of labels such that if (m, n) is \
an edge, then m < n."];

If[Not@ValueQ[ButcherBeta::usage],ButcherBeta::usage =
"ButcherBeta[tree] gives the number of ways of labelling the tree tree \
with ButcherOrder[tree]-1 distinct labels such that the root is not \
labelled, but every other vertex is labelled. ButcherBeta[n, tree] \
gives the number of ways of labelling n of the vertices of the tree with \
n distinct labels such that every leaf is labelled and the root is not \
labelled."];

If[Not@ValueQ[ButcherBetaBar::usage],ButcherBetaBar::usage =
"ButcherBetaBar[tree] gives the number of ways of labelling the tree \
tree with ButcherOrder[tree] distinct labels such that every vertex \
is labelled. ButcherBetaBar[n, tree] gives the number of ways of \
labelling n of the vertices of the tree with n distinct labels such that \
every leaf is labelled."];

If[Not@ValueQ[ButcherColumnConditions::usage],ButcherColumnConditions::usage =
"ButcherColumnConditions[p, s] gives the column simplifying conditions up to \
and including order p for s stages. ButcherColumnConditions[p] gives \
the column simplifying conditions in stage-independent tensor notation. \
The option RungeKuttaMethod controls the form of the result."];

If[Not@ValueQ[ButcherGamma::usage],ButcherGamma::usage =
"ButcherGamma[tree] gives the density of the tree tree. This is also the \
reciprocal of the right-hand side of the order condition imposed by tree."];

If[Not@ValueQ[ButcherHeight::usage],ButcherHeight::usage =
"ButcherHeight[tree] gives the height of the tree tree."];

If[Not@ValueQ[ButcherOrder::usage],ButcherOrder::usage =
"ButcherOrder[tree] gives the order r of the tree tree."];

If[Not@ValueQ[ButcherPhi::usage],ButcherPhi::usage =
"ButcherPhi[tree, s] gives the left-hand side of the order condition \
imposed by tree of an s-stage Runge-Kutta method. ButcherPhi[tree] gives \
the weight function in stage-independent tensor notation involving dot \
products and scalar multiplications. Using this notation, \[FormalA] is an s by s \
matrix, \[FormalB] and \[FormalC] are vectors of length s, and \[FormalE] is the length-s vector with \
unit components.  The option RungeKuttaMethod controls the form of the result."];

If[Not@ValueQ[ButcherPlot::usage],ButcherPlot::usage =
"ButcherPlot[tree,options] gives a plot of the tree tree.\n \
ButcherPlot[{tree1, tree2, ...},options] gives an array of plots of \
the various trees.  Valid options are ButcherPlotColumns, \
ButcherPlotNodeSize, ButcherPlotRootSize and ButcherPlotLabel."];

If[Not@ValueQ[ButcherPlotColumns::usage],ButcherPlotColumns::usage = "ButcherPlotColumns is an option to ButcherPlot that \
specifies the number of columns in the GraphicsArray plot of a list of trees."];

If[Not@ValueQ[ButcherPlotLabel::usage],ButcherPlotLabel::usage = "ButcherPlotLabel is an option to ButcherPlot that \
specifies a list of plot labels, which should be the same length as the \
list of trees to be plotted."];

If[Not@ValueQ[ButcherPlotNodeSize::usage],ButcherPlotNodeSize::usage = "ButcherPlotNodeSize is an option to ButcherPlot that \
specifies a scaling factor to be applied to the default size for nodes in the plot."];

If[Not@ValueQ[ButcherPlotRootSize::usage],ButcherPlotRootSize::usage = "ButcherPlotRootSize is an option to ButcherPlot that \
specifies a scaling factor to be applied to the default size for encircling of tree \
roots in the plot. The root is not highlighted if the value is zero."];

If[Not@ValueQ[ButcherPrincipalError::usage],ButcherPrincipalError::usage =
"ButcherPrincipalError[p, s] calculates the principal error for a \
method of order p with s stages. ButcherPrincipalError[p] gives \
the principal error using stage-independent tensor notation. The result \
is a list of the order p+1 terms appearing in a Taylor series expansion \
of the error. The option RungeKuttaMethod controls the form of the result."];

If[Not@ValueQ[ButcherQuadratureConditions::usage],ButcherQuadratureConditions::usage =
"ButcherQuadratureConditions[p, s] gives the quadrature conditions up to \
and including order p for s stages. ButcherQuadratureConditions[p] gives \
the quadrature conditions in stage-independent tensor notation. The option \
RungeKuttaMethod controls the form of the result."];

If[Not@ValueQ[ButcherRowConditions::usage],ButcherRowConditions::usage =
"ButcherRowConditions[p, s] gives the row simplifying conditions up to \
and including order p for s stages. ButcherRowConditions[p] gives \
the row simplifying conditions in stage-independent tensor notation. \
The option RungeKuttaMethod controls the form of the result."];

If[Not@ValueQ[ButcherRowSum::usage],ButcherRowSum::usage =
"ButcherRowSum is an option to RungeKuttaOrderConditions stating whether \
the row-sum conditions for the \!\(\*SubscriptBox[\(\[FormalC]\), \(i\)]\) should be added to the list of order \
conditions. ButcherRowSum may be set to True or False."];

If[Not@ValueQ[ButcherSimplify::usage],ButcherSimplify::usage =
"ButcherSimplify is an option to RungeKuttaOrderConditions specifying \
whether to apply Butcher's row and column simplifying conditions. The \
conditions reduce the set of necessary and sufficient trees for an order \
p method with s stages. This is especially useful for deriving high order \
methods. The form of the simplifying assumptions used depends upon whether \
the method is implicit or explicit. For Implicit methods, all trees are \
reduced to quadrature type together with subsidiary row and column simplifying \
conditions (which are sufficient but by no means necessary for order p). For \
explicit methods, the restriction \!\(\*SubscriptBox[\(\[FormalB]\), \(2\)]\)==0 for p>=5 implies positivity of the \
weights \!\(\*SubscriptBox[\(\[FormalB]\), \(i\)]\). If simplification is to be performed the number of stages \
must be a number. Valid settings are True and False (standard order \
conditions)."];

If[Not@ValueQ[ButcherSigma::usage],ButcherSigma::usage =
"ButcherSigma[tree] gives the order of the symmetry group of isomorphisms \
of the tree tree with itself."];

If[Not@ValueQ[ButcherTrees::usage],ButcherTrees::usage =
"ButcherTrees[p] gives a list of lists of the trees for any Runge-Kutta \
method of order p. The inner lists group the trees by order."];

If[Not@ValueQ[ButcherTreeCount::usage],ButcherTreeCount::usage = "ButcherTreeCount[p] gives a list of the number of \
trees through order p."];

If[Not@ValueQ[ButcherTreeQ::usage],ButcherTreeQ::usage =
"ButcherTreeQ[tree] tests for valid syntax of the tree or trees tree."];

If[Not@ValueQ[ButcherTreeSimplify::usage],ButcherTreeSimplify::usage =
"ButcherTreeSimplify[p, eta, xi] gives the set of trees through order p that \
are not reduced by Butcher's simplifying assumptions, assuming that the \
quadrature conditions through order p, the row simplifying conditions through \
order eta, and the column simplifying conditions through order xi all hold. \
The result is grouped by order, starting with the first non-vanishing trees."];

If[Not@ValueQ[ButcherWidth::usage],ButcherWidth::usage =
"ButcherWidth[tree] gives the width of the tree tree."];

If[Not@ValueQ[$ContinuousExtension::usage],$ContinuousExtension::usage =
"$ContinuousExtension is a global environment setting, specifying whether \
to generate conditions for continuous extensions of Runge-Kutta methods."];

If[Not@ValueQ[ContinuousExtension::usage],ContinuousExtension::usage =
"ContinuousExtension is an option to RungeKuttaOrderConditions and related \
functions specifying whether to generate order conditions for continuous \
extensions of Runge-Kutta methods."];

If[Not@ValueQ[RungeKuttaOrderConditions::usage],RungeKuttaOrderConditions::usage =
"RungeKuttaOrderConditions[p, s] gives a list of lists of the order conditions \
that any s-stage Runge-Kutta method of order p must satisfy. The inner lists \
group the conditions by order. The conditions are expressed in terms of the \
variables \[FormalA], \[FormalB], \[FormalC], and the resulting Runge-Kutta method is\n\n\
Y[i,x] = y[x0] + (x - x0) Sum[\!\(\*SubscriptBox[\(\[FormalA]\), \(i, j\)]\) f[Y[j,x]], {j, 1, s}], i = 1, ..., s\n\n\
Y[x] = y[x0] + (x - x0) Sum[\!\(\*SubscriptBox[\(\[FormalB]\), \(j\)]\) f[Y[j,x]], {j, 1, s}]\n\n\
where the row-sum conditions \!\(\*SubscriptBox[\(\[FormalC]\), \(i\)]\) == Sum[\!\(\*SubscriptBox[\(\[FormalA]\), \(i, j\)]\), {j, 1, s}] are usually \
assumed to hold. RungeKuttaOrderConditions[p] expresses the order conditions \
using stage-independent tensor notation. In this notation, \[FormalA] is an s by s \
matrix, \[FormalB] and \[FormalC] are vectors of length s, and \[FormalE] is the length-s vector with \
unit components. The options ButcherRowSum, ButcherSimplify, and \
RungeKuttaMethod control the form of the result."];

If[Not@ValueQ[RungeKuttaMethod::usage],RungeKuttaMethod::usage=
"RungeKuttaMethod is an option to ButcherPhi and related functions \
specifying the type of method to be generated. Valid settings are \
Explicit, DiagonallyImplicit and Implicit."];

If[Not@ValueQ[$RungeKuttaMethod::usage],$RungeKuttaMethod::usage=
"$RungeKuttaMethod is a global environment setting, specifying the \
type of method to be generated by ButcherPhi and related functions. \
Valid settings are Explicit, DiagonallyImplicit and Implicit."];

If[Not@ValueQ[Explicit::usage],Explicit::usage=
"Explicit is a setting for the option RungeKuttaMethod specifying \
the type of Runge-Kutta method to be generated."];

If[Not@ValueQ[DiagonallyImplicit::usage],DiagonallyImplicit::usage=
"DiagonallyImplicit is a setting for the option RungeKuttaMethod \
specifying the type of Runge-Kutta method to be generated."];

If[Not@ValueQ[Implicit::usage],Implicit::usage=
"Explicit is a setting for the option RungeKuttaMethod specifying \
the type of Runge-Kutta method to be generated."];

If[Not@ValueQ[SymbolicTable::usage],SymbolicTable::usage =
"SymbolicTable is an alternative to Table that allows symbolic ranges."];

Unprotect[ButcherAlpha, ButcherBeta, ButcherBetaBar,
ButcherColumnConditions, ButcherGamma, ButcherHeight, ButcherOrder,
ButcherPhi, ButcherPlot, ButcherPlotColumns, ButcherPlotLabel,
ButcherPlotNodeSize, ButcherPlotRootSize, ButcherPrincipalError,
ButcherQuadratureConditions, ButcherRowConditions, ButcherRowSum,
ButcherSigma, ButcherSimplify, ButcherTrees, ButcherTreeCount,
ButcherTreeQ, ButcherTreeSimplify, ButcherWidth, ContinuousExtension, Explicit,
DiagonallyImplicit, Implicit, RungeKuttaMethod, RungeKuttaOrderConditions,
SymbolicTable];

Begin["`Private`"];

PMIntegerQ = Internal`PositiveMachineIntegerQ;

(* Global environment setting for continous output *)

$ContinuousExtension = False;

(* Global environment setting specifying the type of method to be
 generated by the weight function. *)

$RungeKuttaMethod = Implicit;

CheckCommonOptions[{cont_, mtd_}, func_]:=
  Catch[
    If[ !MemberQ[{True, False}, cont],
      Message[ContinuousExtension::optval, cont, func];
      Throw[False];
    ];
    If[ !MemberQ[{Explicit,DiagonallyImplicit,Implicit}, mtd],
      Message[RungeKuttaMethod::optval, mtd, func];
      Throw[False];
    ];
    True
  ];

ContinuousExtension::optval =
"The option ContinuousExtension->`1` in `2` did not evaluate to True or False.";

RungeKuttaMethod::optval =
"The option RungeKuttaMethod->`1` in `2` did not evaluate to Explicit, \
DiagonallyImplicit or Implicit.";


(* Consider all possible combinations of lower order trees. *)

maketrees[{p_}]:= dups[p]; (* Partition with one element. *)
maketrees[p_]:= Apply[ Outer[Times, ##]&, Map[dups, p] ];

(* Use multiplicity to avoid duplication. The number of trees is
 used at each order. *)

genindx[] := If[!ValueQ@ K, K,
	ToExpression["System`K$" ~~ ToString@ NestWhile[# + 1&, 1,
		ToExpression["System`K$" ~~ ToString@ #, InputForm, ValueQ]&]]];


dups[{i_, 1}]:= trees[i];
dups[{i:(1|2), m_}]:= trees[i]^m;
dups[{i_,m_}]:= dups[{i, m}] = Block[
    {K = genindx[]},
    With[{indx = Array[K, m], trs = trees[i]},
      Apply[
        Flatten[ Table[ Apply[ Times, Part[ trs, indx ] ], ## ] ]&,
        Thread[ {indx, Join[ {btc[i]}, Drop[indx, -1] ] } ]
      ]]];

(* Trees grouped by order. Lower order trees are grafted onto a new
 root by applying f. *)

trees[1] = {\[FormalF]};
trees[p_]:= trees[p] = Block[{\[FormalF]},
	Attributes[\[FormalF]] = Listable;
	\[FormalF][ Flatten[ Map[ maketrees, Internal`IntegerPartitions[-1+p] ] ] ] ];

btrees[n1_, n2_]:= Table[trees[i], {i, n1, n2}];
ButcherTrees[p_?PMIntegerQ]:= btrees[1, p];


(* Test input syntax for a tree. *)

ButcherTreeQ[{}] = False;
ButcherTreeQ[t_]:= forestQ[t];

forestQ[t_List]:= Apply[And, Map[forestQ, t]];
forestQ[\[FormalF]] = True;
forestQ[t_\[FormalF]]:= treeQ[t];
forestQ[_] = False;

SetAttributes[treeQ, Listable];

treeQ[\[FormalF]]=True;
treeQ[t_Times]:= Apply[And, treeQ[ Apply[List, t] ] ];
treeQ[t_^_?PMIntegerQ]:= treeQ[t];
treeQ[t:\[FormalF][subt_]]:= treeQ[t] = treeQ[subt];
treeQ[_]:= False;


(* Enumeration of rooted trees. See for example section 2.3.4.4,
 exercise 2 of D. E. Knuth (1974), Fundamental Algorithms:
 The Art of Computer Programming, Volume 1 (2nd edition),
 Addison Wesley, Massachusetts. The following is a more efficient
 implementation avoiding the use of a double indexed function. *)

ButcherTreeCount[n_?PMIntegerQ]:= Array[btc, n];

btc[1]=1;
btc[n_]:= btc[n]=
  Quotient[
    Sum[i btc[i] Sum[btc[-i j+n],{j, Quotient[#,i]}],{i, #}],
    #
  ]& @ (-1+n);


(**** Numerical functions on trees ****)

(* Define the order of a tree. *)

SetAttributes[order, Listable];
order[\[FormalF]] = 1;
order[t_Times]:= Apply[Plus, order[ Apply[List, t] ] ];
order[t_^n_]:= n*order[t];
order[t_\[FormalF]]:= order[t] = 1 + order[First[t]];

ButcherOrder[forest_?ButcherTreeQ]:= order[forest];

(* Define the density of a tree. *)

SetAttributes[gamma, Listable];
gamma[\[FormalF]] = 1;
gamma[t_Times]:= Map[gamma, t];
gamma[t_^n_]:= gamma[t]^n;
gamma[t_\[FormalF]]:= gamma[t] = order[t]*gamma[First[t]];

ButcherGamma[forest_?ButcherTreeQ]:= gamma[forest];

(* Define the symmetry of a tree *)

SetAttributes[sigma, Listable];
sigma[\[FormalF]] = 1;
sigma[t_Times]:= Map[sigma, t];
sigma[t_^n_]:= n!*sigma[t]^n;
sigma[t_\[FormalF]]:= sigma[t] = sigma[First[t]];

ButcherSigma[forest_?ButcherTreeQ]:= sigma[forest];

(* Define the number of monotonic labellings *)

(* We use a variant of Hairer's definition for alpha, since it is
 more efficient than Butcher's definition in terms of order, gamma and
 sigma. *)

(* Multinomial coefficient for equal arguments m, repeated n times and
 divided by n! *)

mult[1, _]:= 1;
mult[m_, n_]:= mult[m, n]= Quotient[Pochhammer[n+1, n*(m-1)], m!^n];

SetAttributes[alpha, Listable];
alpha[\[FormalF]] = 1;
alpha[t_Times]:=
  Times[
    Apply[ Multinomial, order[Apply[List, t]] ],
    Map[ alpha, t ]
  ];
alpha[t_^n_]:= Times[ mult[order[t], n], alpha[t]^n ];
alpha[t_\[FormalF]]:= alpha[t] = alpha[First[t]];

ButcherAlpha[forest_?ButcherTreeQ]:= alpha[forest];

(* Define the height of a tree *)

SetAttributes[height, Listable];
height[\[FormalF]] = 1;
height[t_Times]:= Max[height[Apply[List, t]]];
height[t_^_]:= height[t];
height[t_\[FormalF]]:= height[t] = 1 + height[First[t]];

ButcherHeight[forest_?ButcherTreeQ]:= height[forest];

(* Define the width of a tree *)

SetAttributes[{width, widthbar}, Listable];
width[\[FormalF]] = 0;
width[t_\[FormalF]]:= widthbar[t];
widthbar[\[FormalF]] = 1;
widthbar[t_Times]:= Apply[Plus, widthbar[ Apply[List, t] ] ];
widthbar[t_^n_]:= n*widthbar[t];
widthbar[t_\[FormalF]]:= widthbar[t] = widthbar[First[t]];

ButcherWidth[forest_?ButcherTreeQ]:= widthbar[forest];

(* Define the number of ways of labelling the tree *)

SetAttributes[beta, Listable];
beta[\[FormalF]] = 1;
beta[t_]:= beta[t] = Quotient[ (order[t]-1)!, sigma[t] ];
beta[0, \[FormalF]] = 1;
beta[0, _] = 0;
beta[n_, t_]:=
  beta[n, t] =
    Quotient[
      n!*(order[t]-1-width[t])!,
      sigma[t]*(n-width[t])!*(order[t]-1-n)!
    ];

ButcherBeta[tree_?ButcherTreeQ]:= beta[tree];

ButcherBeta[n:(0|_?PMIntegerQ), tree_?ButcherTreeQ]:= beta[n, tree];


SetAttributes[betabar, Listable];
betabar[\[FormalF]] = 1;
betabar[t_]:= order[t]*beta[t];
betabar[0, _] = 0;
betabar[n_, t_]:=
  betabar[n, t] =
    Quotient[
      n!*(order[t]-widthbar[t])!,
      sigma[t]*(n-widthbar[t])!*(order[t]-n)!
    ];

ButcherBetaBar[tree_?ButcherTreeQ]:= betabar[tree];

ButcherBetaBar[n:(0|_?PMIntegerQ), tree_?ButcherTreeQ]:= betabar[n, tree];


(**** Order conditions ****)

(* Butcher's index notation. *)

(* The function phi requires bookkeeping for the various indices of summation.
 It is more efficient to collect components in individual sums, since the
 conditions are maintained in factored form when evaluated numerically. *)

(* Indexing Function. *)

indx[i_, symb, Explicit]:= {i,2,symb};
indx[i_,  j_,  Explicit]:= {i,2,j-1};
indx[i_,  j_,  DiagonallyImplicit]:= {i,1,j};
indx[i_,   _,  Implicit]:= {i,1,symb};

(* Terminal vertices. *)

sumterms[depth_,_,\[FormalF]^n_.] := Subscript[\[FormalC],K[depth]]^n;

(* Non-terminal vertices. Current depth of the tree is the ith index in
 the summation. *)

sumterms[depth_, mtd_, \[FormalF][subt_]^n_.]:=
  With[{sumindex = K[++index]},
    Sum @@ {Subscript[\[FormalA], K[depth], sumindex] sumterms[index, mtd, subt],
          indx[sumindex, K[depth], mtd]}^n
  ];

(* Multiple branching. *)

sumterms[depth_, mtd_, subt_Times]:= Map[sumterms[depth, mtd, #]&, subt];

SetAttributes[iphi, Listable];

(* Special case for the tree with one vertex. *)

iphi[\[FormalF], _] := Block[
	{K = genindx[]},
	Sum @@ {Subscript[\[FormalB], K[1]], {K[1], 1, symb}}];

(* General case for a rooted tree. index is the jth index in the
 summation. *)

iphi[\[FormalF][subt_], mtd_] := Block[
    {index = 1, K = genindx[]},
    Sum @@ {Subscript[\[FormalB], K[1]] sumterms[1, mtd, subt],
          indx[K[1], symb, mtd]}];

(* Tensor notation. *)

dotterms[\[FormalF]] = \[FormalC];
dotterms[t_Times]:= Map[dotterms, t];
dotterms[t_^n_]:= dotterms[t]^n;
dotterms[t_\[FormalF]]:= dotterms[t]= Dot[\[FormalA], dotterms[First[t]]];

SetAttributes[tphi, Listable];

(* Special case for the tree with one vertex. *)

tphi[\[FormalF]] = Dot[\[FormalB], \[FormalE]];

(* General case for a rooted tree. *)

tphi[t_\[FormalF]]:= Dot[\[FormalB], dotterms[First[t]]];

(* Switch between the various forms *)

phidispatch[forest_, stages_Symbol, mtd_]:=
  iphi[forest, mtd] /. symb->stages;

phidispatch[forest_, stages_?PMIntegerQ, mtd_]:=
  nphi[forest, stages, mtd];

phidispatch[forest_, mtd_]:=
  tphi[forest];

Options[ButcherPhi] = {RungeKuttaMethod:>$RungeKuttaMethod};
BPhiOptionsValues = Map[First, Options[ButcherPhi]];


(* Tensor notation *)

ButcherPhi[forest_?ButcherTreeQ, opts___?OptionQ]:=
  Module[{mtd},
    {mtd} = BPhiOptionsValues /. Flatten@ {opts, Options[ButcherPhi]};
    (phidispatch[forest, mtd]) /; CheckCommonOptions[{False, mtd}, ButcherPhi]
  ];

(*
 Butcher's index notation for symbolic stages. Definition enables the storage of
 symbolic sums and substitutes for summation values.
 Tensor evaluation for efficiency for numeric stages.
 *)

ButcherPhi[forest_?ButcherTreeQ, stages:(_?PMIntegerQ|_Symbol), opts___?OptionQ]:=
  Module[{mtd},
    {mtd} = BPhiOptionsValues /. Flatten@ {opts, Options[ButcherPhi]};
    (phidispatch[forest, stages, mtd]) /; CheckCommonOptions[{False, mtd}, ButcherPhi]
  ];

(* Create local elements for phi recursion with numeric stages. *)

nphi[forest_, s_, mtd:(Implicit|DiagonallyImplicit|Explicit)]:=
  Module[{amat, bvec = Array[Subscript[\[FormalB], #]&, s], cvec, dotterms, phi},

    Switch[mtd,
      Explicit,
      amat = Table[If[i<=j,0,Subscript[\[FormalA],i,j]],{i,s},{j,s}];
      cvec = Join[{0},Array[Subscript[\[FormalC],#]&,s-1,2]],
      DiagonallyImplicit,
      amat = Table[If[i<j,0,Subscript[\[FormalA],i,j]],{i,s},{j,s}]; cvec = Array[Subscript[\[FormalC],#]&,s],
      Implicit,
      amat = Array[Subscript[\[FormalA],##]&,{s,s}]; cvec = Array[Subscript[\[FormalC],#]&,s]
    ];

    dotterms[\[FormalF]] = cvec;
    dotterms[t_Times]:= Map[dotterms, t];
    dotterms[t_^n_]:= dotterms[t]^n;
    dotterms[t_\[FormalF]]:= dotterms[t]= Dot[amat, dotterms[First[t]]];

    SetAttributes[phi, Listable];
    phi[\[FormalF]] = Dot[bvec, Table[1, {s}]];
    phi[t_\[FormalF]]:= Dot[bvec, dotterms[First[t]]];

    phi[forest]
  ];


(* Principal error function. *)

Options[ButcherPrincipalError] =
  {ContinuousExtension:>$ContinuousExtension, RungeKuttaMethod:>$RungeKuttaMethod};
BPEOptionsValues = Map[First, Options[ButcherPrincipalError]];

ButcherPrincipalError[p_?PMIntegerQ, s:(_?PMIntegerQ|_Symbol), opts___?OptionQ]:=
  Module[{cont, mtd},
    {cont, mtd} = BPEOptionsValues /. Flatten@ {opts, Options[ButcherPrincipalError]};
    ((phidispatch[#, s, mtd] - rhs[cont, #])/sigma[#])& @
      trees[p + 1] /; CheckCommonOptions[{cont, mtd}, ButcherPrincipalError]
  ];

ButcherPrincipalError[p_?PMIntegerQ, opts___?OptionQ]:=
  Module[{cont, mtd},
    {cont, mtd} = BPEOptionsValues /. Flatten@ {opts, Options[ButcherPrincipalError]};
    ((phidispatch[#, mtd] - rhs[cont, #])/sigma[#])& @
      trees[p + 1] /; CheckCommonOptions[{cont, mtd}, ButcherPrincipalError]
  ];


(* Display a Table with symbolic range. *)

SymbolicTable[expr_, {i_, i0_:1, i1_?PMIntegerQ}]:= Table[expr, {i, i0, i1}];

Options[RungeKuttaOrderConditions] =
  {ButcherRowSum->False, ButcherSimplify->False,
   ContinuousExtension:>$ContinuousExtension, RungeKuttaMethod:>$RungeKuttaMethod};
RKOCOptionValues = Map[First, Options[RungeKuttaOrderConditions]];

RungeKuttaOrderConditions::opts =
"The option `1` in RungeKuttaOrderConditions did not evaluate to `2`.";

RKOCtest =
  If[#1,
    True,
    Message[RungeKuttaOrderConditions::opts, Apply[Sequence,#2]]; False
  ]&;

RKOCmessages = {
  {ButcherRowSum, "True or False"},
  {ButcherSimplify, "False or True"},
  {ButcherSimplify, "False for symbolic number of stages"},
  {ContinuousExtension, "False or True"},
  {RungeKuttaMethod, "Explicit, DiagonallyImplicit or Implicit"}};

RKOCOptionsTest[s_, opts___]:=
  Module[{butchersimpQ, cont, datatypes, mtd, rowsumQ},
    {rowsumQ, butchersimpQ, cont, mtd} = RKOCOptionValues /. Flatten@ {opts,
         Options[RungeKuttaOrderConditions]};
    datatypes = {
      MemberQ[{True, False}, rowsumQ],
      MemberQ[{True, False}, butchersimpQ],
      butchersimpQ =!= True || PMIntegerQ[s],
      MemberQ[{True, False}, cont],
      MemberQ[{Explicit, DiagonallyImplicit, Implicit}, mtd]};
    If[Apply[And, MapThread[RKOCtest, {datatypes, RKOCmessages}]],
      {rowsumQ, butchersimpQ, cont, mtd},
      $Failed
    ]
  ];

RungeKuttaOrderConditions[p_?PMIntegerQ, s:(_Symbol|_?PMIntegerQ), opts___?OptionQ]:=
  Module[{ans},
    ans /; ((ans = RKOCOptionsTest[s,opts]) =!= $Failed) &&
              (ans = rkconds[p, s, ans]; True)
  ];

RungeKuttaOrderConditions[p_?PMIntegerQ, opts___?OptionQ]:=
  Module[{ans},
    ans /; ((ans = RKOCOptionsTest[symb, opts]) =!= $Failed) &&
              (ans = rkconds[p, ans]; True)
  ];

(* Thread Equal through equation lists - avoids problems with
 True and False. *)

eq[a_List, b_List]:= MapThread[eq, {a, b}];
eq[a_, b_]:= Equal[a, b];

(* Whether to use the continuous extension of the order conditions *) 

SetAttributes[rhs, Listable];
rhs[True, t_]:= Power[\[FormalT], order[t]]/gamma[t];
rhs[False, t_]:= 1/gamma[t];

(* Order conditions in tensor notation. *)

rkconds[p_, {rowsumQ_ , False, cont_, _}]:=
  With[{forest = btrees[1, p]},
    If[rowsumQ, Append[#, {\[FormalA].\[FormalE]==\[FormalC]}]&, (* else *) #& ] @
      eq[ tphi[forest], rhs[cont, forest] ]
  ];

(* Numeric stages. *)

rkconds[p_, s_?PMIntegerQ, {rowsumQ_, False, cont_, mtd_}]:=
  With[{forest = btrees[1, p]},
    If[rowsumQ,
      Append[#, First[ButcherRowConditions[1, s, RungeKuttaMethod->mtd]]]&,
    #& ] @
      eq[ phidispatch[forest, s, mtd], rhs[cont, forest] ]
  ];

(* Order conditions in Butcher's index notation. *)

rkconds[p_, s_Symbol, {rowsumQ_ , False, cont_, mtd_}]:=
  With[{forest = btrees[1, p]},
    If[rowsumQ,
      Append[#, First[ButcherRowConditions[1, s, RungeKuttaMethod->mtd]]]&,
      #& (* else *)
    ] @ eq[ phidispatch[forest, s, mtd], rhs[cont, forest] ]
  ];


(* Implicit row simplifying conditions A (eta). *)

Options[ButcherRowConditions] = {RungeKuttaMethod :> $RungeKuttaMethod};
BRCOptionValues = Map[First, Options[ButcherRowConditions]];

ButcherRowConditions[p_?PMIntegerQ, s:(_Symbol|_?PMIntegerQ), opts___?OptionQ]:=
  Module[{mtd},
    {mtd} = BRCOptionValues /. Flatten@ {opts, Options[ButcherRowConditions]};
    Table[rowsimpc[j, s, mtd],{j, p}] /; CheckCommonOptions[{False, mtd}, ButcherRowConditions]
  ];

rindx[i_,j_,_,Explicit]:= {i,2,j-1};
rindx[i_,j_,_,DiagonallyImplicit]:= {i,1,j};
rindx[i_,_,s_,Implicit]:= {i,1,s};

ritab[i_,s_,Explicit]:= {i,2,s};
ritab[i_,s_,_]:= {i,1,s};

rowsimpc[1, s_, Explicit] := Block[
  {K = genindx[]},
  SymbolicTable[
    Sum @@ {Subscript[\[FormalA],K[1],K[2]], {K[2],K[1]-1}} == Subscript[\[FormalC],K[1]],
  {K[1],2,s}]];

rowsimpc[p_, s_, mtd_] := Block[
  {K = genindx[]},
  SymbolicTable[
    Sum @@ {Subscript[\[FormalA],K[1],K[2]] Subscript[\[FormalC],K[2]]^(p-1),
      rindx[K[2],K[1],s,mtd]} == Subscript[\[FormalC],K[1]]^p/p,
  ritab[K[1],s,mtd]]];


ButcherRowConditions[p_?PMIntegerQ, opts___?OptionQ]:= Array[brc, p];

brc[1] = Dot[\[FormalA], \[FormalE]] == \[FormalC];
brc[q_]:= brc[q] = Dot[\[FormalA], \[FormalC]^(q-1)] == \[FormalC]^q/q;


(* Implicit column simplifying conditions D (xi). *)

Options[ButcherColumnConditions] = {RungeKuttaMethod:>$RungeKuttaMethod};
BCCOptionValues = Map[First, Options[ButcherColumnConditions]];

ButcherColumnConditions[p_?PMIntegerQ, s:(_Symbol|_?PMIntegerQ), opts___?OptionQ]:=
  Module[{mtd},
    {mtd} = BCCOptionValues /. Flatten@ {opts, Options[ButcherColumnConditions]};
    Table[colsimpc[j, s, mtd], {j, p}] /; CheckCommonOptions[{False, mtd}, ButcherColumnConditions]
  ];

cindx[i_,j_,s_,Explicit]:= {i,j+1,s};
cindx[i_,j_,s_,DiagonallyImplicit]:= {i,j,s};
cindx[i_,_,s_,Implicit]:= {i,s};

colsimpc[p_, s_, Explicit] := Block[
  {K = genindx[]},
  SymbolicTable[
    Sum @@
      {Subscript[\[FormalA],K[1],K[2]] Subscript[\[FormalB],K[1]] Subscript[\[FormalC],K[1]]^(p-1),
        cindx[K[1],K[2],s,Explicit]} == 
          (Apply[If[#1==1,#2,#2 #3]&,{K[2],Subscript[\[FormalB],K[2]]/p,1-Subscript[\[FormalC],K[2]]^p}])
  ,{K[2],1,s}]];

colsimpc[p_, s_, mtd_] := Block[
  {K = genindx[]},
  SymbolicTable[
    Sum @@
      {Subscript[\[FormalA],K[1],K[2]] Subscript[\[FormalB],K[1]] Subscript[\[FormalC],K[1]]^(p-1),
        cindx[K[1],K[2],s,mtd]} == Subscript[\[FormalB],K[2]] (1-Subscript[\[FormalC],K[2]]^p)/p
  ,{K[2],1,s}]];

ButcherColumnConditions[p_?PMIntegerQ, opts___?OptionQ]:= Array[bcc, p];

bcc[1] = Dot[\[FormalB], \[FormalA]] == \[FormalB] (\[FormalE]-\[FormalC]);
bcc[q_]:= bcc[q] = Dot[(\[FormalB] \[FormalC]^(q-1)), \[FormalA]] == \[FormalB] (\[FormalE]-\[FormalC]^q)/q;


(* Quadrature conditions (trees of type B(p)). *)

bushytrees[1] = {\[FormalF]};
bushytrees[p_]:= Join[{{\[FormalF]}}, Table[{\[FormalF][\[FormalF]^(i-1)]},{i, 2, p}]]

bqcdispatch[p_, {cont_, mtd_}]:=
  eq[ phidispatch[#, mtd], rhs[cont, #] ]& @ bushytrees[p];

bqcdispatch[p_, s_, {cont_, mtd_}]:=
  eq[ phidispatch[#, s, mtd], rhs[cont, #] ]& @ bushytrees[p];

Options[ButcherQuadratureConditions] =
  {ContinuousExtension:>$ContinuousExtension, RungeKuttaMethod:>$RungeKuttaMethod};
BQCOptionValues = Map[First, Options[ButcherQuadratureConditions]];

ButcherQuadratureConditions[p_?PMIntegerQ, s:(_?PMIntegerQ|_Symbol), opts___?OptionQ]:=
  Module[{cont, mtd},
    {cont, mtd} = BQCOptionValues /. Flatten@ {opts, Options[ButcherQuadratureConditions]};
    bqcdispatch[p, s, {cont, mtd}] /;
        CheckCommonOptions[{cont, mtd}, ButcherQuadratureConditions]
  ];

ButcherQuadratureConditions[p_?PMIntegerQ, opts___?OptionQ]:=
  Module[{cont, mtd},
    {cont, mtd} = BQCOptionValues /. Flatten@ {opts, Options[ButcherQuadratureConditions]};
    bqcdispatch[p, {cont, mtd}] /;
        CheckCommonOptions[{cont, mtd}, ButcherQuadratureConditions]
  ];

(* Simplifying conditions for implicit schemes. *)

rkconds[p_, s_?PMIntegerQ, {rowsumQ_, True, cont_, mtd:(DiagonallyImplicit|Implicit)}]:=
  Module[{eta, xi},
    eta = Max[1,Quotient[p-1,2]];
    xi = Max[0,p-eta-1];
    Join[
      bqcdispatch[p, s, {cont, mtd}],
      If[rowsumQ, #&, Drop[#,1]&] @
        ButcherRowConditions[eta, s, RungeKuttaMethod->mtd],
      If[xi==0,{},ButcherColumnConditions[xi, s, RungeKuttaMethod->mtd]]
    ]
  ];


(* Derive trees not reduced by simplifying conditions. *)

VanishTree[t_,t_]:=t;
VanishTree[__]:=Sequence[];

SetAttributes[isimptree,Listable];

(* Trees of type B (p). *)

isimptree[\[FormalF]|\[FormalF][\[FormalF]^x_.],__]:= Sequence[];

(* One-leg trees, reduced by D (1). *)

isimptree[t:\[FormalF][\[FormalF][x_]],_,xi_]:= VanishTree[t] /; xi>=1;

(* D (xi). *)

isimptree[t:\[FormalF][\[FormalF]^x_. \[FormalF][y_]],_,xi_]:= VanishTree[t] /; x<=xi-1;

(* A (eta). *)

isimptree[t:\[FormalF][subt_],eta_,_]:=
  VanishTree[t, \[FormalF][ subt /. \[FormalF][\[FormalF]^x_.]:>\[FormalF]^(x+1) /; x<=eta-1 ]];

ButcherTreeSimplify[p_?PMIntegerQ, eta:(0|_?PMIntegerQ), xi:(0|_?PMIntegerQ)]:=
  DeleteCases[ isimptree[ btrees[1, p], eta, xi ], {}];


(* Explicit row simplifying conditions A (eta). *)

(* Include explicit row-sum conditions. *)

ExplicitRowSimplify[s_, True, eta_]:=
  Join[
    ButcherRowConditions[1,s,RungeKuttaMethod->Explicit],
    ExplicitRowSimplify[s,False,eta]
  ];

ExplicitRowSimplify[1, False, _]:= Sequence[]; (* One stage case. *)

(* No additional row conditions for p<=4. *)

ExplicitRowSimplify[_, False, 1]:= Sequence[];

(* Special cases for explicit schemes of order p>=5, Butcher's equations
 330e and 330f (b[2]==0 and remaining 2nd row conditions redundant). *)

rowsimpc[i_,q_]:= Sum[Subscript[\[FormalA],i,j] Subscript[\[FormalC],j]^(q-1), {j, 2, i-1}] == Subscript[\[FormalC],i]^q/q;

ExplicitRowSimplify[s_, False, eta_]:=
  Insert[Table[rowsimpc[i,q],{q,2,eta},{i,3,s}],Subscript[\[FormalB],2]==0,{1,1}];


(* Explicit column simplifying conditions D (1).
 The first relation s=1 is neglected, since it is implied by imposing
 the row sum conditions, the order one and two trees and the remaining
 column conditions (e.g. Hairer, Norsett and Wanner's proof in lemma 1.4).
 The second relation can be neglected similarly using b[2]=0 for order p>=5.
 Higher order row simplifying conditions (xi>1) are not considered for
 explicit methods since they contradict order p>=1 (Butcher section 331). *)

dropecs[_,  1, _]:= Sequence[];
dropecs[3|4, _,  conds_]:= {Drop[Flatten[conds],1]};
dropecs[_,  2, _]:= Sequence[];
dropecs[_,  s_,  conds_]:= {Drop[Flatten[conds],2] /. Subscript[\[FormalB],2]->0};

ExplicitColumnSimplify[p_,s_]:=
  dropecs[p,s,ButcherColumnConditions[1,s,RungeKuttaMethod->Explicit]];


(* Additional trees not reduced by simplifying conditions for
 explicit schemes (Hairer, Norsett and Wanner II .6). *)

SetAttributes[esimptree, Listable];

(* Trees of type B(p) are already defined by ButcherQuadratureConditions. *)

esimptree[\[FormalF][\[FormalF]^x_.],_]:= Sequence[];

(* One-leg trees are reduced by D (1). *)

esimptree[\[FormalF][\[FormalF][x_]],_]:= Sequence[];

(* Reduce trees with one vertex leading to at most
 (eta-1) terminal vertices. *)

esimptree[t:\[FormalF][\[FormalF][\[FormalF]^x_.]^y_. z_.], eta_]:= VanishTree[t] /; x<=(eta-1);

(* All remaining trees are not reduced. *)

esimptree[t_,_]:= t;

AdditionalConditions[p_, s_, eta_, cont_]:=
  addconds[ DeleteCases[ esimptree[btrees[4, p], eta], {} ], s, cont ];

addconds[{}, _, _]:= {};
addconds[forest_, s_, cont_]:=
  eq[ phidispatch[forest, s, Explicit], rhs[cont, forest] ];

(* Simplifying conditions for explicit schemes. *)

(* Use standard order conditions for p=1,2. *)

rkconds[p:(1|2), s_?PMIntegerQ, {rowsumQ_, True, cont_, Explicit}]:=
  rkconds[p, s,{rowsumQ, False, cont, Explicit}];

rkconds[p_, s_?PMIntegerQ, {rowsumQ_, True, cont_, Explicit}]:=
  With[{eta = Quotient[p-1,2]},
    Join[
      If[p>=5, (# /. Subscript[\[FormalB],2]->0)&, #& ] @
        bqcdispatch[p, s, {cont, Explicit}],
      ExplicitRowSimplify[s, rowsumQ, eta],
      ExplicitColumnSimplify[p, s],
      If[p>=5, (# /. Subscript[\[FormalB],2]->0)&, #& ] @ AdditionalConditions[p, s, eta, cont]
    ]
  ];

(**** Tree plotting ****)

(* Find the list of subtrees (one level up) of a tree. *)

SetAttributes[splitpower,Listable];
splitpower[subt_^n_.]:= Table[subt,{n}];
split[\[FormalF][subt_Times]]:= Flatten[splitpower[Apply[List, subt]]];
split[\[FormalF][subt_]]:= splitpower[subt];

buildtree[\[FormalF], _, _, x_, y_]:= {Point[{x, y}]};
buildtree[t_\[FormalF], h_, w_, x_, y_]:=
  Module[{nx, ny=y+h, nw, sp = split[t]},
    nw = w/Length[sp];
    nx = x - w/2 - nw/2;
    Flatten[
      {Point[{x, y}],
       Map[ {Line[{{x,y}, {nx+=nw,ny}}], buildtree[#, h, nw, nx, ny]}&, sp]}
    ]
  ];

(* Encircle root. *)

root[_?(#==0&), __]:= Point[{0,0}];
root[_, br_, ar_]:= {Point[{0,0}], Circle[{0,0}, Scaled[{br ar, br}]]};

Options[ButcherPlot] =
  {ButcherPlotColumns -> Automatic, ButcherPlotNodeSize -> 1,
   ButcherPlotRootSize -> 1, ButcherPlotLabel->{}};
BPlotOptionValues = Map[First, Options[ButcherPlot]];

ButcherPlot::opts =
"The option `1` in ButcherPlot did not evaluate to `2`.";

BTPmessages = {
  {"ButcherPlotNodeSize", "a positive number"},
  {"ButcherPlotRootSize", "zero or a positive number"},
  {"ButcherPlotColumns", "a positive integer or Automatic"},
  {"ButcherPlotLabel", "a list of plot labels"}};

BTPtest =
  If[#1,
    True,
    Message[ButcherPlot::opts,Apply[Sequence,#2]]; False
  ]&;

BTPOptionsTest[ff_,opts___]:=
  Module[{ns, rs, col, lbl, datatypes},
    {col, ns, rs, lbl} = BPlotOptionValues /. {opts} /. Options[ButcherPlot];
    lbl = Flatten[lbl];
    datatypes = {
      TrueQ[Positive[ns]],
      MatchQ[rs,_?(#>=0&)],
      MatchQ[col, Automatic | (_?PMIntegerQ)],
      MatchQ[lbl,{}|_?(VectorQ[#]&&Length[#]===Length[ff]&)]};
    If[Apply[And, MapThread[BTPtest, {datatypes, BTPmessages}]],
      {ns,rs,col,lbl},
      $Failed
    ]
  ];

ButcherPlot[forest_?ButcherTreeQ, opts___?OptionQ]:=
  Module[{ans,ff = Flatten[{forest}]},
    ans /; ((ans = BTPOptionsTest[ff,opts]) =!= $Failed &&
                (ans = BTPlot[ff, ans]; True))
  ];

BTPlot[ff_, {ns_, rs_, col_, lbl_}]:=
  Module[{ar, br, h, lenff, m, n, pf, pr, ps},
    lenff = Length[ff];
    h = Max[1, Map[height, ff] - 1];
    n = If[col === Automatic, Ceiling[Sqrt[N[lenff]]], col];
    m = Ceiling[lenff/n];
    ar = Sqrt[N[h]];
    ps = ns (.0001 m n)^(.4);
    br = rs ps/ar;
    pr = {{-.52-br, .52+br}, {-.02-br, 1.02+br}};
    pf = MapThread[
           Graphics[
             {PointSize[ps], root[rs, br, ar], buildtree[#1, 1/h, .95, 0, 0]},
             PlotRange -> pr, AspectRatio -> ar, PlotLabel -> #2
           ]&,
           {ff, If[lbl==={}, Table[None,{lenff}], lbl]}
         ];
    h = Partition[pf, n];
    If[Length[h] < m, AppendTo[h, Drop[pf, n (m-1)]]];
    Show[GraphicsGrid[h], AspectRatio -> 1]
  ];

End[ ]; (* End `Private` Context. *)

SetAttributes[{ButcherAlpha, ButcherBeta,
ButcherBetaBar, ButcherColumnConditions, ButcherGamma, ButcherHeight,
ButcherOrder, ButcherPhi, ButcherPlot, ButcherPrincipalError,
ButcherQuadratureConditions, ButcherRowConditions, ButcherRowSum,
ButcherSigma, ButcherSimplify, ButcherTrees, ButcherTreeCount,
ButcherTreeQ, ButcherTreeSimplify, ButcherWidth,
RungeKuttaOrderConditions, SymbolicTable}, ReadProtected];

Protect[ButcherAlpha, ButcherBeta, ButcherBetaBar,
ButcherColumnConditions, ButcherGamma, ButcherHeight, ButcherOrder,
ButcherPhi, ButcherPlot, ButcherPlotColumns, ButcherPlotLabel,
ButcherPlotNodeSize, ButcherPlotRootSize, ButcherPrincipalError,
ButcherQuadratureConditions, ButcherRowConditions, ButcherRowSum,
ButcherSigma, ButcherSimplify, ButcherTrees, ButcherTreeCount,
ButcherTreeQ, ButcherTreeSimplify, ButcherWidth, ContinuousExtension, Explicit,
DiagonallyImplicit, Implicit, RungeKuttaMethod, RungeKuttaOrderConditions,
SymbolicTable];
