(* ::Package:: *)

(* ::Chapter::Closed:: *)
(*prolog*)


BeginPackage["AlphaScannerFunctions`", {"AlphaScannerFunctions`CommonFunctions`"}]


CompleteSquare::usage = "Factors an even-ordered polynomial function into a square polynomial plus a constant"


Begin["CompleteSquare`Private`"]


(* ::Chapter::Closed:: *)
(*main code*)


ClearAll[CompleteSquare];
CompleteSquare[lhs_ == rhs_, x_Symbol] := With[{res = CompleteSquare[lhs - rhs, x]}, res == 0 /; FreeQ[res, CompleteSquare]]

CompleteSquare[a_. x_Symbol^2 + b_. x_Symbol y_Symbol + c_. y_Symbol^2 + d_., x_Symbol] /;
	VectorQ[{a, b, c, d}, NumericQ] := a (x + (b y)/(2 a))^2 + (4 a d - b^2 y^2 + 4 a c y^2)/(4 a)

CompleteSquare[p_, var_Symbol] := Block[{isQuadraticQ, csq},
  	isQuadraticQ = !NumericQ[p] && PolynomialQ[p, {var}] && Max[Exponent[p, {var}]] > 1 && biQuadraticQ[p, {var}];
  	csq = p //. {
  			a_. x_Symbol^q_ + b_. x_Symbol^r_. + c_. :> a (x^r + b/(2 a))^2 + (4 a c - b^2)/(4 a) /; 2r == q && FreeQ[{a, b, c, r, q}, var],
  			a_. x_Symbol^q_ + b_. x_Symbol^r_. :> a (x^r + b/(2 a))^2 + -b^2/(4 a) /; 2r == q && FreeQ[{a, b, r, q}, var]
  		};
  	csq /; isQuadraticQ && p == Expand[csq]
]

CompleteSquare[args___]:= $Failed


(*
In[77]:= CompleteSquare[x^2 - 6 x + 13]
Out[77]= 4 + (-3 + x)^2

In[69]:= CompleteSquare[x^4 - 10 x^2 + 1]
Out[69]= -24 + (-5 + x^2)^2

In[76]:= CompleteSquare[x^2 - 6 x + 12 - y - y^2]
Out[76]= 13/4 + (-3 + x)^2 - (1/2 + y)^2

In[104]:= CompleteSquare[-x^2 + 12 x]
Out[104]= 36 - (-6 + x)^2

In[46]:= CompleteSquare[2 x^2 - 3 x y + 4 y^2 + 6 x - 3 y - 4 == 0]
Out[46]= -(145/16) + 2 (3/2 + x)^2 + 4 (-(3/8) + y)^2 - 3 x y == 0
*)


biQuadraticQ[expr_, vars_] := TrueQ @ Apply[And, Exponent[Subtract @@ Eliminate[{expr == 0, t - #^Exponent[expr, #] == 0}, #], t] == 2 & /@ vars]


(*
In[98]:= biQuadraticQ[x^3 - 4 x^2 + 6 x - 24]
Out[98]= False

In[99]:= biQuadraticQ[x^4 - 4 x^2 - 24]
Out[99]= True

In[104]:= biQuadraticQ[x^4 - 4 x^2 - 24 + y^6 - 5 y^3 + 1]
Out[104]= True
*)


ReciprocalPolynomial /: MakeBoxes[ReciprocalPolynomial[expr_, z_], TraditionalForm] := MakeBoxes[CalculateBoxForm[Localize["reciprocal polynomial",27152], expr], TraditionalForm]
ReciprocalPolynomial[poly_, z_] /; PolynomialQ[poly, z] := Conjugate[Reverse[CoefficientList[poly, z]]].(z^Range[0, Exponent[poly, z]])


(* ::Chapter::Closed:: *)
(*epilog*)


End[]


EndPackage[]
