(* :Name: NDSolveProblems` *)

(* :Title: Examples problems for NDSolve. *)

(* :Author: Mark Sofroniou *)

(* :Summary:
 This package adds a number of numerical examples for use in NDSolve.
*)

(* :Context: DifferentialEquations`NDSolveProblems` *)

(* :Package Version: 1.0 *)

(* :Copyright: Copyright 2003, Wolfram Research, Inc. *)

(* :History:
 Version 1.0 by Mark Sofroniou, April 2003.
*)

(* :Keywords:
 Numerical differential equation problems, Test problems,
 Initial Value Problems.
*)

(* :Source:
 DETEST, Bari Test set
*)

(* :Mathematica Version: 5.0 *)

(* :Limitation:
 Should allow the integration range specification through GetNDSolveProblem,
 rather than a second rule for NDSolve.
 Should allow differently named variables for each problem.
 Should allow the specification of parameters for example in
 the Van der Pol problem.
 Should allow the specification of different initial conditions
 and vector initial conditions.
*)

(* :Discussion:
 A number of common differential equation test problems are useful for
 testing and demonstrating the functionality and implementation of NDSolve.
*)

(*
 Still need to add more problems from:
 DETEST
 Bari Test set
 Hairer, Lubich and Wanner
 Hairer, Norsett and Wanner
 Hairer and Wanner
 Lambert

 Specific problems to add:
 Spherical pendulum
 *)

BeginPackage["DifferentialEquations`NDSolveProblems`"];

GetNDSolveProblem::usage =
"GetNDSolveProblem[name] returns an NDSolveProblem data structure for the problem name.";

NDSolveProblem::usage =
"NDSolveProblem is a data structure containing an example problem for use in NDSolve.";

T::usage = "T is the default symbol used for independent variables in NDSolveProblem objects.";
X::usage = "X is the default symbol used for spatial variables in NDSolveProblem objects.";
Y::usage = "Y is the default symbol used for dependent variables in NDSolveProblem objects.";

$NDSolveProblems = {
"AkzoNobel",
"ArnoldBeltramiChildress",
"Arenstorf",
"BrusselatorODE",
"Brusselator-PDE",
"Brusselator-PDE-Discretized",
"Brusselator-PDE-2D",
"Burgers-PDE",
"CartesianPendulum",
"CurtissHirschfelder",
"CUSP-Discretized",
"DuffingOscillator",
"ElectricalEngineering",
"HarmonicOscillator",
"HenonHeiles",
"HIRES",
"Kepler",
"Korteweg-deVries-PDE",
"LinearTest",
"Lorenz",
"LotkaVolterra",
"Pendulum",
"PerturbedKepler",
"Pleiades",
"Pollution",
"RigidBody",
"RingModulator",
"Robertson",
"VanderPol"
};

Unprotect[ GetNDSolveProblem, NDSolveProblem];

Begin["`Private`"];

GetNDSolveProblem["AkzoNobel"] :=
NDSolveProblem[{
{Subscript[Y, 1]'[T] == -2*Subscript[r, 1] + Subscript[r, 2] - Subscript[r, 3] - Subscript[r, 4], 
 Subscript[Y, 2]'[T] == - Subscript[r, 1]/2 - Subscript[r, 4] - Subscript[r, 5]/2 + 
   (33*(9/7370 - Subscript[Y, 2][T]))/10,
 Subscript[Y, 3]'[T] == Subscript[r, 1] - Subscript[r, 2] + Subscript[r, 3], 
 Subscript[Y, 4]'[T] == -Subscript[r, 2] + Subscript[r, 3] - 2*Subscript[r, 4], 
 Subscript[Y, 5]'[T] == Subscript[r, 2] - Subscript[r, 3] + Subscript[r, 5], 
 0 == (11583*Subscript[Y, 1][T]*Subscript[Y, 4][T])/100 - Subscript[Y, 6][T]} /.
  {Subscript[r, 1] -> Subscript[k, 1]*Subscript[Y, 1][T]^4*Sqrt[Subscript[Y, 2][T]], 
   Subscript[r, 2] -> Subscript[k, 2]*Subscript[Y, 3][T]*Subscript[Y, 4][T], 
   Subscript[r, 3] -> (5*Subscript[k, 2]*Subscript[Y, 1][T]*Subscript[Y, 5][T])/172, 
   Subscript[r, 4] -> Subscript[k, 3]*Subscript[Y, 1][T]*Subscript[Y, 4][T]^2, 
   Subscript[r, 5] -> Subscript[k, 4]*Sqrt[Subscript[Y, 2][T]]*Subscript[Y, 6][T]^2} /.
  {Subscript[k, 1] -> 187/10, 
  Subscript[k, 2] -> 29/50, 
  Subscript[k, 3] -> 9/100, 
  Subscript[k, 4] -> 21/50},
{Subscript[Y, 1][0] == 111/250,
 Subscript[Y, 2][0] == 123/100000, 
 Subscript[Y, 3][0] == 0, 
 Subscript[Y, 4][0] == 7/1000,
 Subscript[Y, 5][0] == 0,
 Subscript[Y, 6][0] == 11583/100*111/250*7/1000},
{Subscript[Y, 1][T], 
 Subscript[Y, 2][T], 
 Subscript[Y, 3][T], 
 Subscript[Y, 4][T], 
 Subscript[Y, 5][T], 
 Subscript[Y, 6][T]},
{T, 0, 180},
{},
{},
{}
}];

(* Arnold, Beltrami and Childress flow *)

GetNDSolveProblem["ArnoldBeltramiChildress"] :=
NDSolveProblem[{
{Subscript[Y, 1]'[T] ==
 c*Cos[Subscript[Y, 2][T]] + a*Sin[Subscript[Y, 3][T]], 
 Subscript[Y, 2]'[T] ==
 a*Cos[Subscript[Y, 3][T]] + b*Sin[Subscript[Y, 1][T]], 
 Subscript[Y, 3]'[T] ==
 b*Cos[Subscript[Y, 1][T]] + c*Sin[Subscript[Y, 2][T]]} /.
   {a -> 1, b -> 1, c -> 3/4},
{Subscript[Y, 1][0] == 1/4, 
 Subscript[Y, 2][0] == 1/3, 
 Subscript[Y, 3][0] == 1/2},
{Subscript[Y, 1][T], 
 Subscript[Y, 2][T], 
 Subscript[Y, 3][T]},
{T, 0, 100},
{},
{},
{}
}];

(* Restricted three body problem *)

GetNDSolveProblem["Arenstorf"] :=
NDSolveProblem[{
{Subscript[Y, 1]''[T] == Subscript[Y, 1][T] - 
   (muprime*(mu + Subscript[Y, 1][T]))/D1 - 
   (mu*(-muprime + Subscript[Y, 1][T]))/D2 + 
   2*Subscript[Y, 2]'[T], 
 Subscript[Y, 2]''[T] == Subscript[Y, 2][T] - 
   (mu*Subscript[Y, 2][T])/D2 - (muprime*Subscript[Y, 2][T])/D1 - 
   2*Subscript[Y, 1]'[T]} /.
   {D1 -> ((mu + Subscript[Y, 1][T])^2 + Subscript[Y, 2][T]^2)^(3/2), 
    D2 -> ((-muprime + Subscript[Y, 1][T])^2 + Subscript[Y, 2][T]^2)^(3/2)} /.
   {mu -> Rationalize[0.012277471, 0], muprime -> 1 - Rationalize[0.012277471, 0]},
{Subscript[Y, 1][0] == Rationalize[0.994, 0], Subscript[Y, 1]'[0] == 
  0, Subscript[Y, 2][0] == 0, Subscript[Y, 2]'[0] == 
  Rationalize[-2.00158510637908252240537862224, 0]},
{Subscript[Y, 1][T], Subscript[Y, 2][T]},
{T, 0, Rationalize[17.0652165601579625588917206249, 0]},
{},
{},
{}
}];

GetNDSolveProblem["BrusselatorODE"] :=
NDSolveProblem[{
{Subscript[Y, 1]'[T] == 1 - 4*Subscript[Y, 1][T] + Subscript[Y, 1][T]^2*Subscript[Y, 2][T],
  Subscript[Y, 2]'[T] == 3*Subscript[Y, 1][T] - Subscript[Y, 1][T]^2*Subscript[Y, 2][T]},
{Subscript[Y, 1][0] == 3/2, Subscript[Y, 2][0] == 3},
{Subscript[Y, 1][T], Subscript[Y, 2][T]},
{T, 0, 20},
{},
{},
{}
}];

GetNDSolveProblem["Brusselator-PDE"] :=
NDSolveProblem[{
{Derivative[1, 0][Subscript[Y, 1]][T, X] == A - (B + 1)*Subscript[Y, 1][T, X] + Subscript[Y, 1][T, X]^2*Subscript[Y, 2][T, X] + \[Alpha]*Derivative[0, 2][Subscript[Y, 1]][T, X], 
 Derivative[1, 0][Subscript[Y, 2]][T, X] == B*Subscript[Y, 1][T, X] - Subscript[Y, 1][T, X]^2*Subscript[Y, 2][T, X] + \[Alpha]*Derivative[0, 2][Subscript[Y, 2]][T, X]} /.
 {A -> 1, B -> 3, \[Alpha] -> 1/50},
{Subscript[Y, 1][T, 0] == 1, Subscript[Y, 1][T, 1] == 1, Subscript[Y, 2][T, 0] == 3, Subscript[Y, 2][T, 1] == 3, Subscript[Y, 1][0, X] == 1 + Sin[2*Pi*X], Subscript[Y, 2][0, X] == 3},
{Subscript[Y, 1][T, X], Subscript[Y, 2][T, X]},
{T, 0, 10},
{{X, 0, 1}},
{},
{}
}];

GetNDSolveProblem["Brusselator-PDE-Discretized"] :=
Module[{alpha, dx, n, eqs, ics, sub, vars},
alpha = 1/50;
n = 40;
dx = 1/(n+1);
eqs = Flatten[Table[{Derivative[1][Subscript[Y, 1][i]][T] == 
     1 + Subscript[Y, 1][i][T]^2*Subscript[Y, 2][i][T] - 
      4*Subscript[Y, 1][i][T] + (alpha/dx^2)*(Subscript[Y, 1][i - 1][T] - 2*Subscript[Y, 1][i][T] + 
         Subscript[Y, 1][i + 1][T]), 
    Derivative[1][Subscript[Y, 2][i]][T] == 
     3*Subscript[Y, 1][i][T] - 
      Subscript[Y, 1][i][T]^2*
       Subscript[Y, 2][i][T] + (alpha/dx^2)*(Subscript[Y, 2][i - 1][T] - 2*Subscript[Y, 2][i][T] + 
         Subscript[Y, 2][i + 1][T])}, {i, n}]];
ics = Flatten[Table[{Subscript[Y, 1][i][0] == 1 + Sin[2*Pi*(i/(n + 1))], Subscript[Y, 2][i][0] == 3}, {i, n}]];
(* Should extract from the initial conditions to allow changes to initial/boundary values *)
sub = {Subscript[Y, 1][0][T] -> 1, Subscript[Y, 1][n+1][T] -> 1, Subscript[Y, 2][0][T] -> 3, Subscript[Y, 2][n+1][T] -> 3};
eqs = eqs /. sub;
vars = Flatten[Table[{Subscript[Y, 1][i][T], Subscript[Y, 2][i][T]}, {i, n}]];
NDSolveProblem[{
eqs,
ics,
vars,
{T, 0, 10},
{},
{},
{}
}]
];

GetNDSolveProblem["Brusselator-PDE-2D"] :=
NDSolveProblem[{
{Derivative[1, 0, 0][Subscript[Y, 1]][T, Subscript[X, 1], Subscript[X, 2]] == 1 - (22*Subscript[Y, 1][T, Subscript[X, 1], Subscript[X, 2]])/5 + 
  Subscript[Y, 1][T, Subscript[X, 1], Subscript[X, 2]]^2*Subscript[Y, 2][T, Subscript[X, 1], Subscript[X, 2]] +
  \[Alpha]*(Derivative[0, 0, 2][Subscript[Y, 1]][T, Subscript[X, 1], Subscript[X, 2]] +
  Derivative[0, 2, 0][Subscript[Y, 1]][T, Subscript[X, 1], Subscript[X, 2]]) +
  If[(T >= 11/10) && ((Subscript[X, 1] - 3/10)^2 + (Subscript[X, 2] - 3/5)^2 <= 1/100), 5, 0], 
 Derivative[1, 0, 0][Subscript[Y, 2]][T, Subscript[X, 1], Subscript[X, 2]] == (17*Subscript[Y, 1][T, Subscript[X, 1], Subscript[X, 2]])/5 -
  Subscript[Y, 1][T, Subscript[X, 1], Subscript[X, 2]]^2*Subscript[Y, 2][T, Subscript[X, 1], Subscript[X, 2]] + 
   \[Alpha]*(Derivative[0, 0, 2][Subscript[Y, 2]][T, Subscript[X, 1], Subscript[X, 2]] +
  Derivative[0, 2, 0][Subscript[Y, 2]][T, Subscript[X, 1], Subscript[X, 2]])} /.
 {\[Alpha] -> 1/10},
{Subscript[Y, 1][0, Subscript[X, 1], Subscript[X, 2]] == 22 Subscript[X, 2] (1 - Subscript[X, 2])^(3/2), 
 Subscript[Y, 2][0, Subscript[X, 1], Subscript[X, 2]] == 27 Subscript[X, 1] (1 - Subscript[X, 1])^(3/2), 
 Subscript[Y, 1][T, 0, Subscript[X, 2]] == Subscript[Y, 1][T, 1, Subscript[X, 2]], 
 Subscript[Y, 2][T, 0, Subscript[X, 2]] == Subscript[Y, 2][T, 1, Subscript[X, 2]], 
 Subscript[Y, 1][T, Subscript[X, 1], 0] == Subscript[Y, 1][T, Subscript[X, 1], 1], 
 Subscript[Y, 2][T, Subscript[X, 1], 0] == Subscript[Y, 2][T, Subscript[X, 1], 1]},
{Subscript[Y, 1][T, Subscript[X, 1], Subscript[X, 2]], Subscript[Y, 2][T, Subscript[X, 1], Subscript[X, 2]]},
{T, 0, 23/2},
{{Subscript[X, 1], 0, 1}, {Subscript[X, 2], 0, 1}},
{},
{}
}]

GetNDSolveProblem["Burgers-PDE"] :=
NDSolveProblem[{
{Derivative[1, 0][Y][T, X] == 2*Derivative[0, 1][Y][T, X] + v*Derivative[0, 2][Y][T, X]} /. v -> 5/1000,
{Y[T, 0] == Y[T, 1], Y[0, X] == 1 + Cos[2*Pi*X]},
{Y[T, X]},
{T, 0, 10},
{{X, 0, 1}},
{},
{}
}];

(* Stiff scalar example of Curtiss and Hirschfelder *)

GetNDSolveProblem["CurtissHirschfelder"] :=
NDSolveProblem[{
{Y'[T] == -\[Alpha]*(Y[T] - Cos[T])} /. \[Alpha]->2000,
{Y[0] == 0},
{Y[T]},
{T, 0, 3/2},
{},
{},
{}
}];

(* Discretized CUSP PDE *)

GetNDSolveProblem["CUSP-Discretized"] :=
Module[{n = 32, psub, usub, vsub},
vsub = v[i_][T] :> u[i][T]/(u[i][T] + 1/10);
usub = u[i_][T] :> (Subscript[Y, 3*i-2][T] - 7/10)*(Subscript[Y, 3*i-2][T] - 13/10);
psub =
 {Subscript[Y, -2][T] -> Subscript[Y, 3*n-2][T],
  Subscript[Y, 3*n + 1][T] -> Subscript[Y, 1][T],
  Subscript[Y, -1][T] -> Subscript[Y, 3*n-1][T],
  Subscript[Y, 3*n + 2][T] -> Subscript[Y, 2][T],
  Subscript[Y, 0][T] -> Subscript[Y, 3*n][T], 
  Subscript[Y, 3*n + 3][T] -> Subscript[Y, 3][T]};
NDSolveProblem[{
Flatten[
Table[
{Subscript[Y, 3*i-2]'[T] == -1*^4 (Subscript[Y, 3*i-2][T] (Subscript[Y, 3*i-2][T]^2 + Subscript[Y, 3*i-1][T]) + Subscript[Y, 3*i][T]) + 
   d (Subscript[Y, 3*i-5][T] - 2 Subscript[Y, 3*i-2][T] + Subscript[Y, 3*i+1][T]),
 Subscript[Y, 3*i-1]'[T] == Subscript[Y, 3*i][T] + 7/100 v[i][T] + d (Subscript[Y, 3*i-4][T] - 2 Subscript[Y, 3*i-1][T] + Subscript[Y, 3*i+2][T]),
 Subscript[Y, 3*i]'[T] == (1 - Subscript[Y, 3*i-1][T]^2) Subscript[Y, 3*i][T] - Subscript[Y, 3*i-1][T] - 2/5 Subscript[Y, 3*i-2][T] + 
   35/1000 v[i][T] + d (Subscript[Y, 3*i-3][T] - 2 Subscript[Y, 3*i][T] + Subscript[Y, 3*i+3][T])
 },{i, n}]
] /. d -> n^2/144 /. vsub /. usub /. psub,
Flatten[
  Table[
   {Subscript[Y, 3*i-2][0] == 0,
   Subscript[Y, 3*i-1][0] == -2 Cos[2 i Pi/n], 
   Subscript[Y, 3*i][0] == 2 Sin[2 i Pi/n]}
   , {i, n}]
  ],
Table[Subscript[Y, i][T], {i, 3*n}],
{T, 0, 11/10},
{},
{},
{}
}]
];

(* Forced planar non-autonomous differential system *)

GetNDSolveProblem["DuffingOscillator"] :=
NDSolveProblem[{
{Subscript[Y, 1]'[T] == Subscript[Y, 2][T], 
 Subscript[Y, 2]'[T] == \[Gamma]*Cos[T] + 
   Subscript[Y, 1][T] - Subscript[Y, 1][T]^3 - 
   \[Delta]*Subscript[Y, 2][T]} /. {\[Delta] -> -1/4, \[Gamma] -> 3/10},
{Subscript[Y, 1][0] == 0, Subscript[Y, 2][0] == 1},
{Subscript[Y, 1][T], Subscript[Y, 2][T]},
{T, 0, 10},
{},
{},
{}
}];

(* Tests dynamic type changes from real to complex arithmetic *)

GetNDSolveProblem["ElectricalEngineering"] :=
NDSolveProblem[{
{Subscript[Y, 1]'[T] == (2 + 4I)*Subscript[Y, 1][T] + (-2 - 2I)*Subscript[Y, 2][T] + (4 + 2 I)*Subscript[Y, 3][T], 
 Subscript[Y, 2]'[T] == (3/4 + 11/4 I)*Subscript[Y, 1][T] + (-1 - I)*Subscript[Y, 2][T] + (7/2 + 2I)*Subscript[Y, 3][T], 
 Subscript[Y, 3]'[T] == (-1/2 + 3/2I)*Subscript[Y, 1][T] - 2I Subscript[Y, 2][T] + (2 + 3I)*Subscript[Y, 3][T]},
{Subscript[Y, 1][0] == 1, Subscript[Y, 2][0] == 1 + I, Subscript[Y, 3][0] == 1 + 2I},
{Subscript[Y, 1][T], Subscript[Y, 2][T], Subscript[Y, 3][T]},
{T, 0, 10},
{},
{},
{}
}];

GetNDSolveProblem["HIRES"] :=
NDSolveProblem[{
{Subscript[Y, 1]'[T] == 7/10000 - 171/100*Subscript[Y, 1][T] + 43/100*Subscript[Y, 2][T] + 208/25*Subscript[Y, 3][T], 
 Subscript[Y, 2]'[T] == 171/100*Subscript[Y, 1][T] - 35/4*Subscript[Y, 2][T], 
 Subscript[Y, 3]'[T] == -1003/100*Subscript[Y, 3][T] + 43/100*Subscript[Y, 4][T] + 7/200*Subscript[Y, 5][T], 
 Subscript[Y, 4]'[T] == 208/25*Subscript[Y, 2][T] + 171/100*Subscript[Y, 3][T] - 28/25*Subscript[Y, 4][T], 
 Subscript[Y, 5]'[T] == -349/200*Subscript[Y, 5][T] + 43/100*Subscript[Y, 6][T] + 43/100*Subscript[Y, 7][T], 
 Subscript[Y, 6]'[T] == 69/100*Subscript[Y, 4][T] + 171/100*Subscript[Y, 5][T] - 43/100*Subscript[Y, 6][T] + 
   69/100*Subscript[Y, 7][T] - 280*Subscript[Y, 6][T]*Subscript[Y, 8][T], 
 Subscript[Y, 7]'[T] == -181/100*Subscript[Y, 7][T] + 280*Subscript[Y, 6][T]*Subscript[Y, 8][T], 
 Subscript[Y, 8]'[T] == 181/100*Subscript[Y, 7][T] - 280*Subscript[Y, 6][T]*Subscript[Y, 8][T]},
{Subscript[Y, 1][0] == 1,
 Subscript[Y, 2][0] == 0,
 Subscript[Y, 3][0] == 0,
 Subscript[Y, 4][0] == 0,
 Subscript[Y, 5][0] == 0,
 Subscript[Y, 6][0] == 0,
 Subscript[Y, 7][0] == 0,
 Subscript[Y, 8][0] == 57/10000},
{Subscript[Y, 1][T], Subscript[Y, 2][T], Subscript[Y, 3][T], Subscript[Y, 4][T],
Subscript[Y, 5][T], Subscript[Y, 6][T], Subscript[Y, 7][T], Subscript[Y, 8][T]},
{T, 0, 1609061/5000},
{},
{},
{}
}];

GetNDSolveProblem["Korteweg-deVries-PDE"] :=
NDSolveProblem[{
{Derivative[1, 0][Y][T, X] == Derivative[0, 3][Y][T, X] + 6*Y[T, X]*Derivative[0, 1][Y][T, X]},
{Y[0, X] == E^(-X^2), Y[T, -5] == Y[T, 5]},
{Y[T, X]},
{T, 0, 1},
{{X, -5, 5}},
{},
{}
}];

GetNDSolveProblem["LinearTest"] :=
NDSolveProblem[{
{Y'[T] == -Y[T]},
{Y[0] == 1},
{Y[T]},
{T, 0, 10},
{},
{},
{}
}];

GetNDSolveProblem["Lorenz"] :=
NDSolveProblem[{
{Subscript[Y, 1]'[T] == 
  10*(-Subscript[Y, 1][T] + Subscript[Y, 2][T]), 
 Subscript[Y, 2]'[T] == 28*Subscript[Y, 1][T] - 
   Subscript[Y, 2][T] - Subscript[Y, 1][T]*Subscript[Y, 3][T], 
 Subscript[Y, 3]'[T] == 
  Subscript[Y, 1][T]*Subscript[Y, 2][T] - (8*Subscript[Y, 3][T])/3} /.
 {\[Sigma] -> 10, r -> 28, b -> 8/3},
{Subscript[Y, 1][0] == -8, Subscript[Y, 2][0] == 8, 
 Subscript[Y, 3][0] == 27},
{Subscript[Y, 1][T], Subscript[Y, 2][T], Subscript[Y, 3][T]},
{T, 0, 16},
{},
{},
{}
}];

(* Celestial mechanics problem - seven stars in the plane *)

GetNDSolveProblem["Pleiades"] :=
NDSolveProblem[{
Table[
  {Subscript[Y, 1, i]''[T] == 
      Sum[m[j](Subscript[Y, 1, j][T] - Subscript[Y, 1, i][T])/r[i, j], {j, i - 1}] + 
        Sum[m[j](Subscript[Y, 1, j][T] - Subscript[Y, 1, i][T])/r[i, j], {j, i + 1, 7}],
    Subscript[Y, 2, i]''[T] == 
      Sum[m[j](Subscript[Y, 2, j][T] - Subscript[Y, 2, i][T])/r[i, j], {j, i - 1}] + 
        Sum[m[j](Subscript[Y, 2, j][T] - Subscript[Y, 2, i][T])/r[i, j], {j, i + 1, 7}]},
  {i, 7}] /.
{r[i_, j_]:> ((Subscript[Y, 1, i][T] - Subscript[Y, 1, j][T])^2 + (Subscript[Y, 2, i][T] - Subscript[Y, 2, j][T])^2)^(3/2), m[i_]:> i},
{Subscript[Y, 1, 1][0] == 3, Subscript[Y, 1, 2][0] == 3, 
 Subscript[Y, 1, 3][0] == -1, Subscript[Y, 1, 4][0] == -3, 
 Subscript[Y, 1, 5][0] == 2, Subscript[Y, 1, 6][0] == -2, 
 Subscript[Y, 1, 7][0] == 2, Subscript[Y, 1, 1]'[0] == 
  0, Subscript[Y, 1, 2]'[0] == 0, 
 Subscript[Y, 1, 3]'[0] == 0, 
 Subscript[Y, 1, 4]'[0] == 0, 
 Subscript[Y, 1, 5]'[0] == 0, 
 Subscript[Y, 1, 6]'[0] == 7/4, 
 Subscript[Y, 1, 7]'[0] == -3/2, 
 Subscript[Y, 2, 1][0] == 3, Subscript[Y, 2, 2][0] == -3, 
 Subscript[Y, 2, 3][0] == 2, Subscript[Y, 2, 4][0] == 0, 
 Subscript[Y, 2, 5][0] == 0, Subscript[Y, 2, 6][0] == -4, 
 Subscript[Y, 2, 7][0] == 4, Subscript[Y, 2, 1]'[0] == 
  0, Subscript[Y, 2, 2]'[0] == 0, 
 Subscript[Y, 2, 3]'[0] == 0, 
 Subscript[Y, 2, 4]'[0] == -5/4, 
 Subscript[Y, 2, 5]'[0] == 1, 
 Subscript[Y, 2, 6]'[0] == 0, 
 Subscript[Y, 2, 7]'[0] == 0},
Join[Table[Subscript[Y, 1, i][T], {i, 7}], Table[Subscript[Y, 2, i][T], {i, 7}]],
{T, 0, 3},
{},
{},
{}
}];

GetNDSolveProblem["Pollution"] :=
NDSolveProblem[{
{Subscript[Y, 1]'[T] == 
  -(Subscript[k, 1]*Subscript[Y, 1][T]) - 
   Subscript[k, 23]*Subscript[Y, 1][T]*Subscript[Y, 4][T] + 
   Subscript[k, 2]*Subscript[Y, 2][T]*Subscript[Y, 4][T] + 
   Subscript[k, 3]*Subscript[Y, 2][T]*Subscript[Y, 5][T] - 
   Subscript[k, 14]*Subscript[Y, 1][T]*Subscript[Y, 6][T] + 
   Subscript[k, 12]*Subscript[Y, 2][T]*Subscript[Y, 10][T] - 
   Subscript[k, 10]*Subscript[Y, 1][T]*Subscript[Y, 11][T] + 
   Subscript[k, 9]*Subscript[Y, 2][T]*Subscript[Y, 11][T] + 
   Subscript[k, 11]*Subscript[Y, 13][T] + 
   Subscript[k, 22]*Subscript[Y, 19][T] - 
   Subscript[k, 24]*Subscript[Y, 1][T]*Subscript[Y, 19][T] + 
   Subscript[k, 25]*Subscript[Y, 20][T], 
Subscript[Y, 2]'[T] == 
  Subscript[k, 1]*Subscript[Y, 1][T] - 
   Subscript[k, 2]*Subscript[Y, 2][T]*Subscript[Y, 4][T] - 
   Subscript[k, 3]*Subscript[Y, 2][T]*Subscript[Y, 5][T] - 
   Subscript[k, 12]*Subscript[Y, 2][T]*Subscript[Y, 10][T] - 
   Subscript[k, 9]*Subscript[Y, 2][T]*Subscript[Y, 11][T] + 
   Subscript[k, 21]*Subscript[Y, 19][T], 
Subscript[Y, 3]'[T] == 
  Subscript[k, 1]*Subscript[Y, 1][T] - 
   Subscript[k, 15]*Subscript[Y, 3][T] + 
   Subscript[k, 17]*Subscript[Y, 4][T] + 
   Subscript[k, 19]*Subscript[Y, 16][T] + 
   Subscript[k, 22]*Subscript[Y, 19][T], 
Subscript[Y, 4]'[T] == 
  Subscript[k, 15]*Subscript[Y, 3][T] - 
   Subscript[k, 16]*Subscript[Y, 4][T] - 
   Subscript[k, 17]*Subscript[Y, 4][T] - 
   Subscript[k, 23]*Subscript[Y, 1][T]*Subscript[Y, 4][T] - 
   Subscript[k, 2]*Subscript[Y, 2][T]*Subscript[Y, 4][T], 
Subscript[Y, 5]'[T] == 
  -(Subscript[k, 3]* Subscript[Y, 2][T]* Subscript[Y, 5][T]) + 
   2*Subscript[k, 4]*Subscript[Y, 7][T] + 
   Subscript[k, 6]*Subscript[Y, 6][T]*Subscript[Y, 7][T] + 
   Subscript[k, 7]*Subscript[Y, 9][T] + 
   Subscript[k, 13]*Subscript[Y, 14][T] + 
   Subscript[k, 20]*Subscript[Y, 6][T]*Subscript[Y, 17][T], 
Subscript[Y, 6]'[T] == 
  Subscript[k, 3]*Subscript[Y, 2][T]*Subscript[Y, 5][T] - 
   Subscript[k, 14]*Subscript[Y, 1][T]*Subscript[Y, 6][T] - 
   Subscript[k, 6]*Subscript[Y, 6][T]*Subscript[Y, 7][T] - 
   Subscript[k, 8]*Subscript[Y, 6][T]*Subscript[Y, 9][T] + 
   2*Subscript[k, 18]*Subscript[Y, 16][T] - 
   Subscript[k, 20]*Subscript[Y, 6][T]*Subscript[Y, 17][T], 
Subscript[Y, 7]'[T] == 
  -(Subscript[k, 4]* Subscript[Y, 7][T]) - 
   Subscript[k, 5]*Subscript[Y, 7][T] - 
   Subscript[k, 6]*Subscript[Y, 6][T]*Subscript[Y, 7][T] + 
   Subscript[k, 13]*Subscript[Y, 14][T], 
Subscript[Y, 8]'[T] == 
  Subscript[k, 4]*Subscript[Y, 7][T] + 
   Subscript[k, 5]*Subscript[Y, 7][T] + 
   Subscript[k, 6]*Subscript[Y, 6][T]*Subscript[Y, 7][T] + 
   Subscript[k, 7]*Subscript[Y, 9][T], 
Subscript[Y, 9]'[T] == 
  -(Subscript[k, 7]* Subscript[Y, 9][T]) - 
   Subscript[k, 8]*Subscript[Y, 6][T]*Subscript[Y, 9][T], 
Subscript[Y, 10]'[T] == 
  Subscript[k, 7]*Subscript[Y, 9][T] - 
   Subscript[k, 12]*Subscript[Y, 2][T]*Subscript[Y, 10][T] + 
   Subscript[k, 9]*Subscript[Y, 2][T]*Subscript[Y, 11][T], 
Subscript[Y, 11]'[T] == 
  Subscript[k, 8]*Subscript[Y, 6][T]*Subscript[Y, 9][T] - 
   Subscript[k, 10]*Subscript[Y, 1][T]*Subscript[Y, 11][T] - 
   Subscript[k, 9]*Subscript[Y, 2][T]*Subscript[Y, 11][T] + 
   Subscript[k, 11]*Subscript[Y, 13][T], 
Subscript[Y, 12]'[T] == 
  Subscript[k, 9]*Subscript[Y, 2][T]*Subscript[Y, 11][T], 
Subscript[Y, 13]'[T] == 
  Subscript[k, 10]*Subscript[Y, 1][T]*Subscript[Y, 11][T] - 
   Subscript[k, 11]*Subscript[Y, 13][T], 
Subscript[Y, 14]'[T] == 
  Subscript[k, 12]*Subscript[Y, 2][T]*Subscript[Y, 10][T] - 
   Subscript[k, 13]*Subscript[Y, 14][T], 
Subscript[Y, 15]'[T] == 
  Subscript[k, 14]*
   Subscript[Y, 1][T]*
   Subscript[Y, 6][T], 
Subscript[Y, 16]'[T] == 
  Subscript[k, 16]*Subscript[Y, 4][T] - 
   Subscript[k, 18]*Subscript[Y, 16][T] - 
   Subscript[k, 19]*Subscript[Y, 16][T], 
Subscript[Y, 17]'[T] == 
  -(Subscript[k, 20]*Subscript[Y, 6][T]*Subscript[Y, 17][T]), 
Subscript[Y, 18]'[T] == 
  Subscript[k, 20]*Subscript[Y, 6][T]*Subscript[Y, 17][T], 
Subscript[Y, 19]'[T] == 
  Subscript[k, 23]*Subscript[Y, 1][T]*Subscript[Y, 4][T] - 
   Subscript[k, 21]*Subscript[Y, 19][T] - 
   Subscript[k, 22]*Subscript[Y, 19][T] - 
   Subscript[k, 24]*Subscript[Y, 1][T]*Subscript[Y, 19][T] + 
   Subscript[k, 25]*Subscript[Y, 20][T], 
Subscript[Y, 20]'[T] == 
  Subscript[k, 24]*Subscript[Y, 1][T]*Subscript[Y, 19][T] - 
   Subscript[k, 25]*Subscript[Y, 20][T]} /.
{Subscript[k, 1] -> 7/20, 
 Subscript[k, 2] -> 133/5, 
 Subscript[k, 3] -> 12300, 
 Subscript[k, 4] -> 43/50000, 
 Subscript[k, 5] -> 41/50000, 
 Subscript[k, 6] -> 15000, 
 Subscript[k, 7] -> 13/100000, 
 Subscript[k, 8] -> 24000, 
 Subscript[k, 9] -> 16500, 
 Subscript[k, 10] -> 9000, 
 Subscript[k, 11] -> 11/500, 
 Subscript[k, 12] -> 12000, 
 Subscript[k, 13] -> 47/25, 
 Subscript[k, 14] -> 16300, 
 Subscript[k, 15] -> 4800000, 
 Subscript[k, 16] -> 7/20000, 
 Subscript[k, 17] -> 7/400, 
 Subscript[k, 18] -> 100000000, 
 Subscript[k, 19] -> 444000000000, 
 Subscript[k, 20] -> 1240, 
 Subscript[k, 21] -> 21/10, 
 Subscript[k, 22] -> 289/50, 
 Subscript[k, 23] -> 237/5000,
 Subscript[k, 24] -> 1780, 
 Subscript[k, 25] -> 78/25},
{Subscript[Y, 1][0] == 0, 
 Subscript[Y, 2][0] == 1/5, 
 Subscript[Y, 3][0] == 0, 
 Subscript[Y, 4][0] == 1/25, 
 Subscript[Y, 5][0] == 0, 
 Subscript[Y, 6][0] == 0, 
 Subscript[Y, 7][0] == 1/10, 
 Subscript[Y, 8][0] == 3/10, 
 Subscript[Y, 9][0] == 1/100, 
 Subscript[Y, 10][0] == 0, 
 Subscript[Y, 11][0] == 0, 
 Subscript[Y, 12][0] == 0, 
 Subscript[Y, 13][0] == 0, 
 Subscript[Y, 14][0] == 0, 
 Subscript[Y, 15][0] == 0, 
 Subscript[Y, 16][0] == 0, 
 Subscript[Y, 17][0] == 7/1000,
 Subscript[Y, 18][0] == 0,
 Subscript[Y, 19][0] == 0,
 Subscript[Y, 20][0] == 0},
{Subscript[Y, 1][T], Subscript[Y, 2][T], Subscript[Y, 3][T], Subscript[Y, 4][T],
Subscript[Y, 5][T], Subscript[Y, 6][T], Subscript[Y, 7][T], Subscript[Y, 8][T],
Subscript[Y, 9][T], Subscript[Y, 10][T], Subscript[Y, 11][T], Subscript[Y, 12][T],
Subscript[Y, 13][T], Subscript[Y, 14][T], Subscript[Y, 15][T], Subscript[Y, 16][T],
Subscript[Y, 17][T], Subscript[Y, 18][T], Subscript[Y, 19][T], Subscript[Y, 20][T]},
{T, 0, 60},
{},
{},
{}
}];

GetNDSolveProblem["RingModulator"] :=
NDSolveProblem[{
{Derivative[1][Subscript[Y, 1]][T] == 1/C (Subscript[Y, 8][T] - 1/2 Subscript[Y, 10][T] + 
    1/2 Subscript[Y, 11][T] + Subscript[Y, 14][T] - 
    Subscript[Y, 1][T]/R),
 Derivative[1][Subscript[Y, 2]][T] == 1/C (Subscript[Y, 9][T] - 1/2 Subscript[Y, 12][T] + 
    1/2 Subscript[Y, 13][T] + Subscript[Y, 15][T] - 
    Subscript[Y, 2][T]/R),
 Derivative[1][Subscript[Y, 3]][T] == 1/Subscript[C, s] (Subscript[Y, 10][T] - q[Subscript[U, D1]] + 
    q[Subscript[U, D4]]),
 Derivative[1][Subscript[Y, 4]][T] == 1/Subscript[C, s] (-Subscript[Y, 11][T] + q[Subscript[U, D2]] - 
    q[Subscript[U, D3]]),
 Derivative[1][Subscript[Y, 5]][T] == 1/Subscript[C, s] (Subscript[Y, 12][T] + q[Subscript[U, D1]] - 
    q[Subscript[U, D3]]),
 Derivative[1][Subscript[Y, 6]][T] == 1/Subscript[C, s] (-Subscript[Y, 13][T] - q[Subscript[U, D2]] + 
    q[Subscript[U, D4]]),
 Derivative[1][Subscript[Y, 7]][T] == 1/Subscript[C, p] (-1/Subscript[R, p] Subscript[Y, 7][T] + 
    q[Subscript[U, D1]] + q[Subscript[U, D2]] - q[Subscript[U, D3]] - 
    q[Subscript[U, D4]]),
 Derivative[1][Subscript[Y, 8]][T] == -1/Subscript[L, h] Subscript[Y, 1][T],
 Derivative[1][Subscript[Y, 9]][T] == -1/Subscript[L, h] Subscript[Y, 2][T],
 Derivative[1][Subscript[Y, 10]][T] == 1/Subscript[L, s2] (1/2 Subscript[Y, 1][T] - Subscript[Y, 3][T] - 
    Subscript[R, g2] Subscript[Y, 10][T]),
 Derivative[1][Subscript[Y, 11]][T] == 1/Subscript[L, s3] (-1/2 Subscript[Y, 1][T] + Subscript[Y, 4][T] - 
    Subscript[R, g3] Subscript[Y, 11][T]),
 Derivative[1][Subscript[Y, 12]][T] == 1/Subscript[L, s2] (1/2 Subscript[Y, 2][T] - Subscript[Y, 5][T] - 
    Subscript[R, g2] Subscript[Y, 12][T]),
 Derivative[1][Subscript[Y, 13]][T] == 1/Subscript[L, s3] (-1/2 Subscript[Y, 2][T] + Subscript[Y, 6][T] - 
    Subscript[R, g3] Subscript[Y, 13][T]),
 Derivative[1][Subscript[Y, 14]][T] == 1/Subscript[L, s1] (- Subscript[Y, 1][T] + 
    Subscript[U, in1] - (Subscript[R, i] + Subscript[R, g1]) Subscript[Y, 14][T]),
 Derivative[1][Subscript[Y, 15]][T] == 1/Subscript[L, s1] (- Subscript[Y, 2][T] -
 (Subscript[R, c] + Subscript[R, g1]) Subscript[Y, 15][T])} /.
 {q[U_] :> \[Gamma] (Exp[\[Delta] U] - 1)} /.
 {Subscript[U, D1] -> 
   Subscript[Y, 3][T] - Subscript[Y, 5][T] - Subscript[Y, 7][T] - 
    Subscript[U, in2],
  Subscript[U, D2] -> -Subscript[Y, 4][T] + Subscript[Y, 6][T] - 
    Subscript[Y, 7][T] - Subscript[U, in2],
  Subscript[U, D3] -> 
   Subscript[Y, 4][T] + Subscript[Y, 5][T] + Subscript[Y, 7][T] + 
    Subscript[U, in2],
  Subscript[U, D4] -> -Subscript[Y, 3][T] - Subscript[Y, 6][T] + 
    Subscript[Y, 7][T] + Subscript[U, in2]} /.
  {Subscript[U, in1] -> 1/2 Sin[2000 Pi T], Subscript[U, in2] -> 2 Sin[20000 Pi T]} /.
  {C -> 1/62500000, 
 Subscript[C, s] -> 1/500000000000, 
 Subscript[C, p] -> 1/100000000, 
 Subscript[L, h] -> 89/20, 
 Subscript[L, s1] -> 1/500, 
 Subscript[L, s2] -> 1/2000, 
 Subscript[L, s3] -> 1/2000, 
 \[Gamma] -> 111437/2739836563887, 
 R -> 25000, 
 Subscript[R, p] -> 50, 
 Subscript[R, g1] -> 363/10, 
 Subscript[R, g2] -> 173/10, 
 Subscript[R, g3] -> 173/10, 
 Subscript[R, i] -> 50, 
 Subscript[R, c] -> 600, 
 \[Delta] -> 44373333/2500000},
{Subscript[Y, 1][0] == 0, Subscript[Y, 2][0] == 0, Subscript[Y, 3][0] == 0, 
 Subscript[Y, 4][0] == 0, Subscript[Y, 5][0] == 0, Subscript[Y, 6][0] == 0, 
 Subscript[Y, 7][0] == 0, Subscript[Y, 8][0] == 0, Subscript[Y, 9][0] == 0, 
 Subscript[Y, 10][0] == 0, Subscript[Y, 11][0] == 0, Subscript[Y, 12][0] == 0, 
 Subscript[Y, 13][0] == 0, Subscript[Y, 14][0] == 0, Subscript[Y, 15][0] == 0},
{Subscript[Y, 1][T], Subscript[Y, 2][T], Subscript[Y, 3][T], 
 Subscript[Y, 4][T], Subscript[Y, 5][T], Subscript[Y, 6][T], 
 Subscript[Y, 7][T], Subscript[Y, 8][T], Subscript[Y, 9][T], 
 Subscript[Y, 10][T], Subscript[Y, 11][T], Subscript[Y, 12][T], 
 Subscript[Y, 13][T], Subscript[Y, 14][T], Subscript[Y, 15][T]},
{T, 0, 1/1000},
{},
{},
{}
}];

(* Stiff ODE modelling an electrical circuit *)

GetNDSolveProblem["VanderPol"] :=
NDSolveProblem[{
{Subscript[Y, 1]'[T] == Subscript[Y, 2][T],
 \[Epsilon]*Subscript[Y, 2]'[T] ==  -Subscript[Y, 1][T] + (1 - Subscript[Y, 1][T]^2)*Subscript[Y, 2][T]} /.
   \[Epsilon] -> 3/1000,
{Subscript[Y, 1][0] == 2, Subscript[Y, 2][0] == 0},
{Subscript[Y, 1][T], Subscript[Y, 2][T]},
{T, 0, 5/2},
{},
{},
{}
}];

(**** Problems with invariants ****)

(* Henon Heiles Hamiltonian *)

GetNDSolveProblem["HenonHeiles"] :=
Module[{ics, dvars, idata, ivar, ivar0, sdata},
idata = {T, 0, 100};
sdata = {};
ivar = Part[idata, 1];
ivar0 = Part[idata, 2];
dvars = {Subscript[Y, 1][ivar], Subscript[Y, 2][ivar], Subscript[Y, 3][ivar], Subscript[Y, 4][ivar]};
ics = {3/25, 3/25, 3/25, 3/25};
NDSolveProblem[{
Thread[Equal[D[dvars, ivar],
{Part[dvars, 3], Part[dvars, 4], -Part[dvars, 1]*(1 + 2*Part[dvars, 2]),
-Part[dvars, 1]^2 + Part[dvars, 2]*(Part[dvars, 2] - 1)}
]],
Thread[Equal[dvars /. ivar->ivar0, ics]],
dvars,
idata,
sdata,
{(Part[dvars, 3]^2 + Part[dvars, 4]^2)/2 + (Part[dvars, 1]^2 + Part[dvars, 2]^2)/2 +
Part[dvars, 1]^2 * Part[dvars, 2] - Part[dvars, 2]^3/3},
{}
}]
];

GetNDSolveProblem["CartesianPendulum"] :=
Module[{ics, dvars, idata, ivar, ivar0, sdata},
idata = {T, 0, 50};
sdata = {};
ivar = Part[idata, 1];
ivar0 = Part[idata, 2];
dvars = {Subscript[Y, 1][ivar], Subscript[Y, 2][ivar], Subscript[Y, 3][ivar], Subscript[Y, 4][ivar]};
ics = {1, 0, 0, 0};
NDSolveProblem[{
Thread[Equal[D[dvars, ivar],
{Part[dvars, 3], Part[dvars, 4],
-Part[dvars, 1]*(Part[dvars, 3]^2 + Part[dvars, 4]^2 - Part[dvars, 2])/(Part[dvars, 1]^2 + Part[dvars, 2]^2),
-1 - Part[dvars, 2]*(Part[dvars, 3]^2 + Part[dvars, 4]^2 - Part[dvars, 2])/(Part[dvars, 1]^2 + Part[dvars, 2]^2)}
]],
Thread[Equal[dvars /. ivar->ivar0, ics]],
dvars,
idata,
sdata,
{Part[dvars, 1]*Part[dvars, 3] + Part[dvars, 2]*Part[dvars, 4],
Part[dvars, 1]^2 + Part[dvars, 2]^2},
{}
}]
];

GetNDSolveProblem["HarmonicOscillator"] :=
Module[{ics, dvars, idata, ivar, ivar0, sdata},
idata = {T, 0, 10};
sdata = {};
ivar = Part[idata, 1];
ivar0 = Part[idata, 2];
dvars = {Subscript[Y, 1][ivar], Subscript[Y, 2][ivar]};
ics = {1, 0};
NDSolveProblem[{
Thread[Equal[D[dvars, ivar],
{Part[dvars, 2], -Part[dvars, 1]}
]],
Thread[Equal[dvars /. ivar->ivar0, ics]],
dvars,
idata,
sdata,
{dvars.dvars/2},
{}
}]
];

(* Two body problem *)

GetNDSolveProblem["Kepler"] :=
Module[{ics, dvars, idata, ivar, ivar0, params, sdata},
idata = {T, 0, 100};
sdata = {};
ivar = Part[idata, 1];
ivar0 = Part[idata, 2];
dvars = {Subscript[Y, 1][ivar], Subscript[Y, 2][ivar], Subscript[Y, 3][ivar], Subscript[Y, 4][ivar]};
(* Eccentricity e *)
params = {3/5};
ics = {1 - params[[1]], 0, 0, Sqrt[(1 + params[[1]])/(1 - params[[1]])]};
NDSolveProblem[{
Thread[Equal[D[dvars, ivar],
{Part[dvars, 3], Part[dvars, 4],
-(Part[dvars, 1]/(Part[dvars, 1]^2 + Part[dvars, 2]^2)^(3/2)), 
 -(Part[dvars, 2]/(Part[dvars, 1]^2 + Part[dvars, 2]^2)^(3/2))}
]],
Thread[Equal[dvars /. ivar->ivar0, ics]],
dvars,
idata,
sdata,
{(dvars[[3]]^2 + dvars[[4]]^2)/2 -1/Sqrt[dvars[[1]]^2 + dvars[[2]]^2],
-(dvars[[2]]*dvars[[3]]) + dvars[[1]]*dvars[[4]]},
{}
}]
];

(* Predator-prey model *)

GetNDSolveProblem["LotkaVolterra"] :=
Module[{ics, dvars, idata, ivar, ivar0, sdata},
idata = {T, 0, 10};
sdata = {};
ivar = Part[idata, 1];
ivar0 = Part[idata, 2];
dvars = {Subscript[Y, 1][ivar], Subscript[Y, 2][ivar]};
ics = {109/40, 1};
NDSolveProblem[{
Thread[Equal[D[dvars, ivar],
{Part[dvars, 1]*(Part[dvars, 2] - 1), Part[dvars, 2]*(2 - Part[dvars, 1])}
]],
Thread[Equal[dvars /. ivar->ivar0, ics]],
dvars,
idata,
sdata,
{Log[Part[dvars, 2]] - Part[dvars, 2] + 2*Log[Part[dvars, 1]] - Part[dvars, 1]},
{}
}]
];

(* Frictionless nonlinear pendulum *)

GetNDSolveProblem["Pendulum"] :=
Module[{ics, dvars, idata, ivar, ivar0, sdata},
idata = {T, 0, 100};
sdata = {};
ivar = Part[idata, 1];
ivar0 = Part[idata, 2];
dvars = {Subscript[Y, 1][ivar], Subscript[Y, 2][ivar]};
ics = {1/2, 0};
NDSolveProblem[{
Thread[Equal[D[dvars, ivar],
{Part[dvars, 2], -Sin[Part[dvars, 1]]}
]],
Thread[Equal[dvars /. ivar->ivar0, ics]],
dvars,
idata,
sdata,
{Part[dvars, 2]^2/2 - Cos[Part[dvars, 1]]},
{}
}]
];

(* Restricted two body problem *)

GetNDSolveProblem["PerturbedKepler"] :=
Module[{ics, dvars, idata, ivar, ivar0, params, sdata},
idata = {T, 0, 100};
sdata = {};
ivar = Part[idata, 1];
ivar0 = Part[idata, 2];
dvars = {Subscript[Y, 1][ivar], Subscript[Y, 2][ivar], Subscript[Y, 3][ivar], Subscript[Y, 4][ivar]};
(* Eccentricity e *)
params = {3/5};
ics = {1 - params[[1]], 0, 0, Sqrt[(1 + params[[1]])/(1 - params[[1]])]};
NDSolveProblem[{
Thread[Equal[D[dvars, ivar],
{dvars[[3]], dvars[[4]],
(-3*dvars[[1]])/(400*(dvars[[1]]^2 + dvars[[2]]^2)^(5/2)) - dvars[[1]]/(dvars[[1]]^2 + dvars[[2]]^2)^(3/2), 
 (-3*dvars[[2]])/(400*(dvars[[1]]^2 + dvars[[2]]^2)^(5/2)) - dvars[[2]]/(dvars[[1]]^2 + dvars[[2]]^2)^(3/2)}
]],
Thread[Equal[dvars /. ivar->ivar0, ics]],
dvars,
idata,
sdata,
{-1/(400*(dvars[[1]]^2 + dvars[[2]]^2)^(3/2)) - 1/Sqrt[dvars[[1]]^2 + dvars[[2]]^2] + (dvars[[3]]^2 + dvars[[4]]^2)/2, 
-(dvars[[2]]*dvars[[3]]) + dvars[[1]]*dvars[[4]]},
{}
}]
];

(* Euler's equations for rigid body motion *)

GetNDSolveProblem["RigidBody"] :=
Module[{ics, dvars, idata, ivar, ivar0, params, sdata},
idata = {T, 0, 32};
sdata = {};
ivar = Part[idata, 1];
ivar0 = Part[idata, 2];
dvars = {Subscript[Y, 1][ivar], Subscript[Y, 2][ivar], Subscript[Y, 3][ivar]};
ics = {Cos[11/10], 0,Sin[11/10]};
(* Principal moments of inertia {I[1], I[2], I[3]} *)
params = {2, 1, 2/3};
NDSolveProblem[{
Thread[Equal[D[dvars, ivar],
{(dvars[[2]]*dvars[[3]]*(params[[2]] - params[[3]]))/(params[[2]]*params[[3]]), 
 (dvars[[1]]*dvars[[3]]*(-params[[1]] + params[[3]]))/(params[[1]]*params[[3]]), 
 (dvars[[1]]*dvars[[2]]*(params[[1]] - params[[2]]))/(params[[1]]*params[[2]])}
]],
Thread[Equal[dvars /. ivar->ivar0, ics]],
dvars,
idata,
sdata,
{dvars.dvars, 1/2 dvars.(dvars/params)},
{}
}]
];

(* Stiff ODE modelling a chemical reaction *)

GetNDSolveProblem["Robertson"] :=
Module[{ics, dvars, idata, ivar, ivar0, params, sdata},
idata = {T, 0, 3/10};
sdata = {};
ivar = Part[idata, 1];
ivar0 = Part[idata, 2];
dvars = {Subscript[Y, 1][ivar], Subscript[Y, 2][ivar], Subscript[Y, 3][ivar]};
ics = {1, 0, 0};
params = {1/25, 1*^4, 3*^7};
NDSolveProblem[{
Thread[Equal[D[dvars, ivar],
{-Part[params, 1]*dvars[[1]] + Part[params, 2]*dvars[[2]]*dvars[[3]],
 Part[params, 1] dvars[[1]] - Part[params, 3]*dvars[[2]]^2 - Part[params, 2]*dvars[[2]]*dvars[[3]],
 Part[params, 3]*dvars[[2]]^2}
]],
Thread[Equal[dvars /. ivar->ivar0, ics]],
dvars,
idata,
sdata,
{Apply[Plus, dvars]},
{}
}]
];

(**** Data layout and communcation ****)

NDSolveProblem[_]["Methods"]:=
  {"DependentVariables", "ExactSolution", "InitialConditions", "Invariants", "Methods", "SpaceData",
   "System", "TimeData"}

NDSolveProblem[data_][("System" | "System"[])]:= Part[data, 1] /; (Length[data] == 7);
NDSolveProblem[data_][("InitialConditions" | "InitialConditions"[])]:= Part[data, 2] /; (Length[data] == 7);
NDSolveProblem[data_][("DependentVariables" | "DependentVariables"[])]:= Part[data, 3] /;(Length[data] == 7);
NDSolveProblem[data_][("TimeData" | "TimeData"[])]:= Part[data, 4] /; (Length[data] == 7);
NDSolveProblem[data_][("SpaceData" | "SpaceData"[])]:= Apply[Sequence, Part[data, 5]] /; (Length[data] == 7);
NDSolveProblem[data_][("Invariants" | "Invariants"[])]:= Part[data, 6] /; (Length[data] == 7);
NDSolveProblem[data_][("ExactSolution" | "ExactSolution"[])]:= Part[data, 7] /; (Length[data] == 7);

(* Overloaded definitions for NDSolve *)

IntegrationRangeQ[{_, _?NumberQ, _?NumberQ}]:= True;
IntegrationRangeQ[___]:= False;

NDSolveHeadQ[NDSolve] = True;
NDSolveHeadQ[NDSolveValue] = True;
NDSolveHeadQ[_] = False;

NDSolveProblem /:
(h_?NDSolveHeadQ)[ndp_NDSolveProblem, opts___?OptionQ]:=
  h[{ndp["System"], ndp["InitialConditions"]}, ndp["DependentVariables"],
    ndp["TimeData"], ndp["SpaceData"], opts];

NDSolveProblem /:
(h_?NDSolveHeadQ)[ndp_NDSolveProblem, tdata_?IntegrationRangeQ, opts___?OptionQ]:=
  h[{ndp["System"], ndp["InitialConditions"]}, ndp["DependentVariables"],
    tdata, opts];

NDSolveProblem /:
(h_?NDSolveHeadQ)[ndp_NDSolveProblem, tdata_?IntegrationRangeQ, sdata__?IntegrationRangeQ, opts___?OptionQ]:=
  h[{ndp["System"], ndp["InitialConditions"]}, ndp["DependentVariables"],
    tdata, sdata, opts];

(* Not yet correct for PDEs *)

NDSolveProblem /:
NDSolve`ProcessEquations[ndp_NDSolveProblem, opts___?OptionQ]:=
  NDSolve`ProcessEquations[{ndp["System"], ndp["InitialConditions"]}, ndp["DependentVariables"],
    First[ndp["TimeData"]], opts];

(* Rule for the NDSolveProblem object and NDSolve syntax *)

wasProtected = Unprotect[NDSolve];

ndspsyntax = {"ArgumentsPattern" -> {_, Optional[_], 
     Optional[{_, _, _}], Optional[{_, _, _}], OptionsPattern[]}};

remsyntax = If[#==={}, #, Rest[#]]& @ SyntaxInformation[NDSolve];

SyntaxInformation[NDSolve] = Join[ndspsyntax, remsyntax];

Apply[Protect, wasProtected];

End[ ]; (* End `Private` Context. *)

SetAttributes[{GetNDSolveProblem, NDSolveProblem}, {Protected, ReadProtected}];

EndPackage[ ]; (* End package Context. *)

