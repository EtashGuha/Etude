(* ::Package:: *)

(* Mathematica Package *)

(* Created by the Wolfram Workbench Aug 22, 2006 *)
(* :Name: VariationalMethods` *)
(* :Title: Variational Methods *)
(* :Author: Yu He *)
(* :Summary:
This package finds first functional derivatives,
Euler equations given the functional to be
extremized, and the first integrals corresponding to energy or to
ignorable coordinates. It also implements the Ritz
variational procedure given a trial function.
*)
(* :Context: VariationalMethods` *)
(* :Package Version: 1.2 *)
(* :Copyright: Copyright 1992-2007,  Wolfram Research, Inc.  *)
(* :History:
	Originally written by Yu He, 1992.
	Updated FirstIntegrals to give result in terms of
		FirstIntegral[] rules, Yu He, 1995.
		Integrated in to new context Packages`, 2006
*)
(* :Keywords: functional derivatives, calculus of variations, first integrals,
	Ritz variational method
*)
(* :Source: Basic calculus of variations texts. *)
(* :Warnings: None. *)
(* :Mathematica Version: 2.1 *)
(* :Limitations:
	The first variational derivative of a functional given here is defined
	for points inside the region of integration defining the functional
	and excludes the boundary. The extremal function solves the Euler
	equations (necessary condition) and satisfies boundary conditions
	that can depend on endpoint variations which are not included here.
	The function FirstIntegrals does not find all possible first integrals,
	only those corresponding to the energy and ignorable coordinates.  The
	function VariationalBound may be slow when multidimensional integrals
	are involved or if the number of variational parameters is more than two.
*)
(* :Discussion: *)

BeginPackage["VariationalMethods`"]
(* Exported symbols added here with SymbolName::usage *) 
If[Not@ValueQ[VariationalD::usage],VariationalD::usage=
"VariationalD[f, u[x], x] or VariationalD[f, u[x,y,...], {x,y,...}] returns the \
first variational derivative of the functional defined by the integrand f which \
is a function of u, its derivatives, and x,y,....  VariationalD[f, {u[x,y,...], \
v[x,y,...],...}, {x,y,...}] gives a list of first variational derivatives with \
respect to u, v, ... ."];
	
If[Not@ValueQ[EulerEquations::usage],EulerEquations::usage=
"EulerEquations[f, u[x], x] or EulerEquations[f, u[x,y,...], \
{x,y,...}] returns the Euler-(Lagrange) equation obeyed by u[x] for \
the functional defined by f.  EulerEquations[f, {u[x,y,...],v[x,y,...], ...}, \
{x,y,...}] returns a list of Euler equations obeyed by the \
functions {u[x,y,...], v[x,y,...], ...}."];
		
If[Not@ValueQ[FirstIntegrals::usage],FirstIntegrals::usage =
"FirstIntegrals[f, u[x], x] or FirstIntegrals[f, {u[x],v[x],...}, x] \
returns a list of first integrals corresponding to those coordinates u, v, ... \
of the integrand f that are ignorable; when f is independent of x and depends \
on the coordinates and their first derivatives only, the first integral \
corresponding to x is also returned."];

If[Not@ValueQ[FirstIntegral::usage],FirstIntegral::usage =
"FirstIntegral[u] represents the first integral associated with the variable \
u. It appears in the output of the function FirstIntegrals."];

If[Not@ValueQ[VariationalBound::usage],VariationalBound::usage =
"VariationalBound[f, u[x], {x,xmin,xmax}, ut, {a,amin,amax}, {b,bmin,bmax}, ...] \
finds the values of parameters {a,b,...} of the trial function ut that \
extremize Integrate[f, {x,xmin,xmax}], returning the extremal value \
of the functional and the optimal parameter values. \
VariationalBound[f, u[x,y,...], {{x,xmin, xmax},{y,ymin,ymax},...}, ut, \
{a,amin,amax}, {b,bmin,bmax}, ...] finds the values of parameters of a trial \
function of two or more variables. \
VariationalBound[{f,g}, u[x], {x,xmin,xmax}, ut, {a,amin,amax}, {b,bmin,bmax}, \
...] or VariationalBound[{f,g}, u[x,y,...], {{x,xmin, xmax},{y,ymin,ymax},...}, \
ut, {a,amin,amax}, {b,bmin,bmax}, ...] extremize the ratio \
Integrate[f,{x,xmin,xmax}]/Integrate[g,{x,xmin,xmax}]. \
Specifying {a} in place of {a,amin,amax} indicates that the parameter a may \
range over {-Infinity, Infinity}."];

If[Not@ValueQ[NVariationalBound::usage],NVariationalBound::usage =
"NVariationalBound[{f,g}, u[x], {x,xmin,xmax}, ut, {a,a0,amin,amax}, \
{b,b0,bmin,bmax}, ..., opts] numerically determines the parameters {a,b,...} \
of a trial function ut[x] that extremize \
Integrate[f, {x,xmin,xmax}]/Integrate[g, {x,xmin,xmax}] by starting from \
{a0,b0,...} and evaluating the integrals numerically.  It returns the \
extremal value of the functional and the optimal values of the parameters. \
NVariationalBound[{f,g}, u[x,y,...], {{x,xmin,xmax},{y,ymin,ymax},...}, ut, \
{a,a0,amin,amax}, {b,b0,bmin,bmax}, ..., opts] does the same for more than one \
independent variable.  If the argument g is absent the functional \
Integrate[f, {x,xmin,xmax}] is extremized."];


Begin["`Private`"]

(* utility function; currently excludes some common incorrect inputs *)
integrandQ[f_] := !ListQ[f] && (Head[f] =!= Equal)

(* Implementation of the package *)
VariationalD[f_, (y_)[x_, r___], w:{x_, r___}]/;integrandQ[f] :=
  Module[{Dfuncs, Dtimes, dummyfunc},
    Dfuncs = Union[Cases[{f}, Derivative[__][y][__], Infinity]];
    Dtimes = (Head[Head[#1]] & ) /@ Dfuncs /. Derivative -> List;
    Simplify[D[f, y[x, r]] + (ReleaseHold[Thread[dummyfunc[(D[f,
                 #1] & ) /@ Dfuncs,
             (Hold[Apply[Sequence, #1]] & ) /@
              (Thread[{w, #1}] & ) /@ Dtimes]]] /. dummyfunc -> D) .
        ((-1)^#1 & ) /@ (Apply[Plus, #1] & ) /@ Dtimes] ]

VariationalD[f_, v:{(y_)[x_, r___], ___}, w:{x_, r___}] :=
  (VariationalD[f, #1, w] & ) /@ v /;
   If[Apply[And, (MatchQ[#1, _[Apply[Sequence, w]]] & ) /@ v],
   True, Message[VariationalD::argx, w]]

VariationalD[f_, (y_)[x_], x_] := VariationalD[f, y[x], {x}]

VariationalD[f_, v:{(y_)[x_], ___}, x_] := VariationalD[f, v, {x}]

VariationalD[any___] := Null/;(Message[VariationalD::args]; False)

VariationalD::args =
"VariationalD takes a single integrand, a function or list of functions, and \
a list of variables as input."

VariationalD::argx =
"The second argument of VariationalD is a list of unknown functions \
depending on `1`."

EulerEquations[f_, funcs_, vars_]/;integrandQ[f] :=
	Module[{result=iEulerEquations[f, funcs, vars]},
		result /; result =!= $Failed]

EulerEquations[any___] := Null/;(Message[EulerEquations::args]; False)

EulerEquations::args =
"EulerEquations takes a single integrand, a function or list of functions, and \
a list of variables as input."

iEulerEquations[f_, funcs_, vars_] :=
	Module[{d=VariationalD[f, funcs, vars]},
		If[d===0, Return[$Failed], Thread[d == 0]]]

FirstIntegrals[f_, v:{(y_)[x_], ___}, x_]/;integrandQ[f] :=
  Module[{v1 = {}, dummy, Dfuncs, Dtimes},
    (If[FreeQ[f, #1], v1 = Append[v1, Rule[FirstIntegral[Head[#1]], 	
        Simplify[IntegratedEE[f, #1]]]]] & ) /@ v;
    If[FreeQ[f /. (t_)[x] -> t[dummy], x],
    Dfuncs = Union[Cases[{f}, Derivative[_][_][x], Infinity]];
    Dtimes = (Head[Head[#1]] & ) /@ Dfuncs /. Derivative -> List;
    If[Max[Flatten[Dtimes]] < 2,
      Append[v1, Rule[FirstIntegral[x], Simplify[-f + Apply[Plus,
      (D[#1, x]*D[f, D[#1, x]] & ) /@ v]]]]], v1]]\
     /; If[Apply[And, (MatchQ[#1, _[x]] & ) /@ v],
     True,
    Message[FirstIntegrals::argx, x]]

FirstIntegrals[f_, (y_)[x_], x_] := FirstIntegrals[f, {y[x]}, x]

FirstIntegrals[any___] := Null/;(Message[FirstIntegrals::args]; False)

FirstIntegrals::args =
"FirstIntegrals takes a single integrand, a function or list of functions, and \
a variable as input."

IntegratedEE[f_, (u_)[x_]] :=
  Module[{Dfuncs, Dtimes, dummyfunc},
   Dfuncs = Union[Cases[{f}, Derivative[_][u][x],
   Infinity]];
    Dtimes = (Head[Head[#1]] & ) /@ Dfuncs /.
    Derivative -> List;
    (ReleaseHold[Thread[dummyfunc[(D[f, #1] & ) /@ Dfuncs,
          (Hold[Apply[Sequence, #1]] & ) /@
            (Thread[{x, #1}] & ) /@ (#1 - 1 & ) /@ Dtimes]]] /.
        dummyfunc -> D) . ((-1)^#1 & ) /@ (Apply[Plus, #1] & ) /@
	Dtimes]

FirstIntegrals::argx =
"The second argument of FirstIntegrals is a list of unknown functions \
depending on `1`."


VariationalBound[{f_, g_}, (u_)[x_, y___],
   lim:{{x_, x1_, x2_}, ___List}, ut_, a__List]:=Module[{
   result=iVariationalBound[{f,g},u[x,y],lim,ut,a]},
   result/; result=!=$Failed]


VariationalBound[f_, (u_)[x_, y___], lim:{{x_, x1_, x2_}, ___List}, ut_,
   a__List]:= Module[{
   result=iVariationalBound[{f, Automatic},u[x,y],lim,ut,a]},
   result/; result=!=$Failed]

VariationalBound[obj_, (u_)[x_], lim:{x_, x1_, x2_}, ut_, a__List] :=
VariationalBound[obj, u[x], {lim}, ut, a]

iVariationalBound[{f_, g_}, (u_)[x_, y___],
   lim:{{x_, x1_, x2_}, ___List}, ut_, a__List]:=
  Module[{rr, num, den, intn, intd, cond1, cond2, v, v2, c,val, para},
   If[Apply[And, (MatchQ[Length[#1], 1 | 3] & ) /@ {a}], True, 
    	 Message[VariationalBound::argx];Return[$Failed]];
    para = First /@ {a};
       v = (#1[[2]] <= #1[[1]] <= #1[[3]] & ) /@
           (If[Length[#1] == 1, Flatten[{#1, -Infinity, Infinity}],
               #1] & ) /@ {a};
    rr = {u -> Function[Evaluate[{x, y}], ut]};
     intn = Integrate[Expand[f /. rr], Apply[Sequence, lim], GenerateConditions->False];
     intd = If[g === Automatic, 1,
      Integrate[Expand[g /. rr], Apply[Sequence, lim], GenerateConditions->False]];
  If[FreeQ[intn,If]&&(!FreeQ[intn,Integrate]), Message[VariationalBound::int];
  		Return[$Failed]];
  If[FreeQ[intd,If]&&(!FreeQ[intd,Integrate]), Message[VariationalBound::int];
  		Return[$Failed]];
     {num, cond1} = If[FreeQ[intn,If],{intn,True},{intn[[2]],intn[[1]]}];
     {den, cond2} = If[FreeQ[intd,If],{intd,True},{intd[[2]],intd[[1]]}];
      v2 = Chop[N[Solve[(Numerator[Together[D[num/den, #1]]] == 0 & )
    		 /@ para, para]]];
      v2 = Union[Delete[v2, {First[#]}& /@ Position[v2, _Complex]]];
      c=(cond1&&cond2)/.v2;
      Set[c[[#]],True]& /@ 
      Complement[Range[Length[v2]],Flatten[Position[c, True|False]]];
      v2 = v2[[Flatten[Position[c, True]]]];
         v2 = Delete[v2, Position[(Apply[And, #1] & ) /@ (v /. v2), False]];
      v2 = Delete[v2, Position[Chop[ut /. v2], 0]];
      If[v2 === {}, Message[VariationalBound::nonex];Return[$Failed],
           val = N[num/den /. v2];
           If[Apply[And, (NumberQ[#1] & ) /@ val] || Length[val] === 1,      
             {Min[val], Flatten[v2[[Flatten[Position[val, Min[val]]]]]]},
             Transpose[{val, v2}]]]]
  	
VariationalBound::int =
"The integral(s) involved cannot be evaluated."

VariationalBound::argx =
"The input for each variational parameter should be of the form {a, amin, amax} or {a}."

VariationalBound::nonex =
"No meaningful extremum found in the specified parameter interval(s)."
	
Options[NVariationalBound] = Options[FindMinimum]

NVariationalBound[{f_, g_:Automatic}, (u_)[x_, y___],
   lim:{{x_, x1_, x2_}, ___List}, ut_, para:{a_, a0_, ___}, m___List,
   opts___Rule] :=
  Module[{rr}, rr = {u -> Function[Evaluate[{x, y}], ut]};
    FindMinimum[Evaluate[Integrate[Expand[f /. rr], Apply[Sequence, lim]]/
      If[g === Automatic, 1, Integrate[Expand[g /. rr],
        Apply[Sequence, lim]]]], para, m, opts]]

NVariationalBound[f_, (u_)[x_, y___], lim:{{x_, x1_, x2_}, ___List}, ut_,
   para:{a_, a0_, ___}, m___List, opts___Rule] :=
  NVariationalBound[{f}, u[x, y], lim, ut, para, m, opts]

NVariationalBound[obj_, (u_)[x_], lim:{x_, x1_, x2_}, ut_, a__List,
   opts___Rule] := NVariationalBound[obj, u[x], {lim}, ut, a, opts]



End[]

EndPackage[]

(*:Examples:
VariationalD[y[x]Sqrt[1+y'[x]^2],y[x],x]

EulerEquations[Grad[phi[x,y,z]].Grad[phi[x,y,z]]/2, phi[x,y,z],{x,y,z}]

FirstIntegrals[m(r'[t]^2+r[t]^2 phi'[t]^2)/2-U[r], {r[t],phi[t]},t]

VariationalBound[{(-u[r] D[r^2 u'[r],r]/r^2-2u[r]^2/r)r^2, u[r]^2 r^2},u[r],
	{r,0,Infinity},(a-r)E^(-b r),{a},{b}]

NVariationalBound[{u'[x]^2+(x^2+x^4)u[x]^2/4,u[x]^2},u[x],
	{x,-Infinity,Infinity},E^(-a x^2)(1+b x^2),{a,0.5},{b,0.1}]

*)
