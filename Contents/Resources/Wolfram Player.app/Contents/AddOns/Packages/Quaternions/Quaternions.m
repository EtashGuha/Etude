(* ::Package:: *)

(* :Context: Quaternions` *)

(* :Title: Quaternions *)

(* :Author: Jason Kastner *)

(* :Version: Mathematica 5.0 *)

(* :Package Version: 1.2 *)

(* :Keywords:
    quaternions, Hamilton
*)

(* :History:
Version 1.0 by Jason Kastner, 1992-1993.
Revised by Matthew Markert, November 1993.
Version 1.2 by Brett Champion, April 2003.
*)

(* :Copyright: Copyright 1993-2007,  Wolfram Research, Inc. *)

(* :Requirements: *)

(* :Warnings:

    "Quaternions came from Hamilton after his really good work had
     been done; and though beautifully ingenious, have been an unmixed
     evil to those who have touched them in any way."
                        - Lord Kelvin

    Adds functionality to the following functions:
    Log, Exp, Cos, Sin, Tan, Sec, Csc, Cot,
    ArcCos, ArcSin, ArcTan, ArcSec, ArcCsc, ArcCot,
    Cosh, Sinh, Tanh, Sech, Csch, Coth,
    ArcCosh, ArcSinh, ArcTanh, ArcSech, ArcCsch, ArcCoth,
    Round, Mod, Sqrt, Power, Plus, Times, NonCommutativeMultiply,
    PrimeQ, Abs, Divide, EvenQ, OddQ, Re, Sign, K

    In this version of the package, the quaternions must
    have real valued entries. i.e.
    QuaternionQ[Quaternion[w,x,y,z]]        (* Symbolic entries *)
    QuaternionQ[Quaternion[1+I, 2, 3+I, 4]] (* Bi-quaternions *)
    will both return False.

    More support is given to manipulating objects with the
    Head Quaternion than is given to objects of the form
    a + b I + c J + d K. In general, the I,J,K form is only
    usable when doing basic (i.e. addition and multiplication)
    quaternion mathematics.

*)

(* :Summary:
This package implements Hamilton's quaternion algebra.
*)

(* :Sources: The Mathematical Papers of Sir William Rowan Hamilton,
        Sir William Rowan Hamilton, Cambridge University Press, 1967.

         Algebras and their Arithmetics, Leonard Dickson, Dover, 1960.

         An Introduction to the Theory of Numbers, G.H.Hardy and E.M.
        Wright, Clarendon Press, 1965.

         Numbers, H.D. Ebbinghaus et al., Springer-Verlag 1991.

         A History of Vector Analysis, Michael Crowe, University of
        Notre Dame Press, 1967.

*)

BeginPackage["Quaternions`"]

If[Not@ValueQ[AbsIJK::usage],AbsIJK::usage = "AbsIJK[q] gives the absolute value of the pure quaternion \
part of q."];

If[Not@ValueQ[AdjustedSignIJK::usage],AdjustedSignIJK::usage = "AdjustedSignIJK[q] gives the Sign of the pure \
quaternion part of q, adjusted so its first non-zero part is positive."];

If[Not@ValueQ[LeftAssociates::usage],LeftAssociates::usage = "LeftAssociates[q] gives a list of the 24 \
left associates of the quaternion q."];

If[Not@ValueQ[RightAssociates::usage],RightAssociates::usage = "RightAssociates[q] gives a list of the 24 \
right associates of the quaternion q."];

If[Not@ValueQ[PrimaryLeftAssociate::usage],PrimaryLeftAssociate::usage = "PrimaryLeftAssociate[q] gives the \
left associate of the quaternion q with the largest scalar component."];

If[Not@ValueQ[PrimaryRightAssociate::usage],PrimaryRightAssociate::usage = "PrimaryRightAssociate[q] gives the \
right associate of the quaternion q with the largest scalar component."];

If[Not@ValueQ[LeftGCD::usage],LeftGCD::usage= "LeftGCD[a,b] gives the greatest common left divisor \
of the two quaternions a and b."];

If[Not@ValueQ[RightGCD::usage],RightGCD::usage= "RightGCD[a,b] gives the greatest common right divisor \
of the two quaternions a and b."];

If[Not@ValueQ[IntegerQuaternionQ::usage],IntegerQuaternionQ::usage = "IntegerQuaternionQ[q] gives True if q is \
an integer quaternion and False otherwise."];

If[Not@ValueQ[J::usage],J::usage = "J represents a quaternion unit with J^2 == -1."];

If[Head[K::usage] === MessageName,
   K::usage = "K represents a quaternion unit with K^2 == -1.",
   If[StringPosition[K::usage, "quaternion"] === {},
      K::usage = K::usage <> " " <>
     "K also represents a quaternion unit with K^2 == -1."
   ]
]

If[Not@ValueQ[NonCommutativeMultiply::usage],NonCommutativeMultiply::usage = "a ** b ** c is a general associative, \
but non-commutative, form of multiplication. It currently implements \
quaternion multiplication."];

If[Head[Norm::usage] =!= String,
    Norm::usage = "Norm[q] returns Abs[q]^2 when q is a quaternion."
]

If[
    StringPosition[PrimeQ::usage, "quaternions"] === {},
    PrimeQ::usage =
    PrimeQ::usage <> " " <>
    "PrimeQ[q, Quaternions -> True] yields True " <>
    "if q is a prime number with respect to the quaternions, " <>
    "and yields False otherwise."
]

If[Not@ValueQ[Quaternion::usage],Quaternion::usage = "Quaternion[a,b,c,d] represents the quaternion \
a + b I + c J + d K."];

If[Not@ValueQ[QuaternionQ::usage],QuaternionQ::usage = "QuaternionQ[q] gives True if q is a quaternion, \
and False otherwise."];

If[Not@ValueQ[Quaternions::usage],Quaternions::usage = "Quaternions is an option for PrimeQ which specifies \
whether factorization should be done over the quaternions."];

If[Not@ValueQ[ScalarQ::usage],ScalarQ::usage = "ScalarQ[q] gives True if q is a scalar, and False \
otherwise."];

If[Not@ValueQ[ToQuaternion::usage],ToQuaternion::usage = "ToQuaternion[q] transforms q into a Quaternion \
object if at all possible."];

If[Not@ValueQ[FromQuaternion::usage],FromQuaternion::usage = "FromQuaternion[q] transforms the quaternion q \
from Quaternion[a,b,c,d] into the form a + b I + c J + d K."];

If[Not@ValueQ[UnitQuaternionQ::usage],UnitQuaternionQ::usage = "UnitQuaternionQ[q] gives True if q is a \
unit quaternion."];

If[Not@ValueQ[UnitQuaternions::usage],UnitQuaternions::usage = "UnitQuaternions is a list of the 24 units \
in the ring of integer quaternions."];

Begin["`Private`"]

(* ScalarQ is a careful test for real numbers *)

ScalarQ[x_]:= (NumericQ[x] && Head[x] =!= Complex)

QuaternionQ[a_]:= Head[a] === Quaternion && Apply[And,Map[ScalarQ,a]]

MessageQuaternionQ[a_, fun_] :=
    If[QuaternionQ[a],
        True,
    (* else *)
    Message[Quaternion::notquat,1,fun];
    False
    ]

(* *************** Rules for Plus *************** *)

Quaternion /:
    Quaternion[a_,b_,c_,d_] + Quaternion[e_,f_,g_,h_]:=
    Quaternion[a+e,b+f,c+g,d+h] // QSimplify

Quaternion /:
    Complex[x_,y_] + Quaternion[a_,b_,c_,d_]:=
    Quaternion[a+x,b+y,c,d] // QSimplify

Quaternion /:
    x_?ScalarQ + Quaternion[a_,b_,c_,d_]:= Quaternion[a+x,b,c,d]

Quaternion /:
    x_. J + Quaternion[a_,b_,c_,d_]:= Quaternion[a,b,c+x,d]

Quaternion /:
    x_. K + Quaternion[a_,b_,c_,d_]:= Quaternion[a,b,c,d+x]

(* *************** Rules for Times *************** *)

Quaternion /:
    x_?ScalarQ * Quaternion[a_,b_,c_,d_]:= Quaternion[x a,x b,x c,x d]

(* *************** Rules for NonCommutativeMultiply *************** *)

Quaternion /:
    Quaternion[a_,b_,c_,d_] ** Quaternion[w_,x_,y_,z_]:=
    Quaternion[
        a w - b x - c y - d z,
        a x + b w + c z - d y,
        a y - b z + c w + d x,
        a z + b y - c x + d w
    ] // QSimplify

Unprotect[NonCommutativeMultiply]
SetAttributes[NonCommutativeMultiply,Listable]

(* Try to take care of the numerous patterns possible when using
   Complex[] numbers
*)

((a_:1)*Complex[b_,c_]) ** ((x_:1)*Complex[y_,z_]) =
    a b x y - a c x z + (a c x y + a b x z) I

((a_:1)*J) ** ((b_:1)*J) = -a b
((a_:1)*K) ** ((b_:1)*K) = -a b
((a_:1)*Complex[b_,c_]) ** ((d_:1)*J) = a b d J + a c d K
((d_:1)*J) ** ((a_:1)*Complex[b_,c_]) = a b d J - a c d K
((a_:1)*J) ** ((b_:1)* K) = a b I
((a_:1)*K) ** ((b_:1)* J) = - a b I
((d_:1)*K) ** ((a_:1)* Complex[b_,c_]) = a b d K + a c d J
((a_:1)*Complex[b_,c_]) ** ((d_:1)* K) = a b d K - a c d J
a_?ScalarQ ** b_ = a b
a_ ** b_?ScalarQ = a b
(x_ + y_) ** a_:= x**a + y**a
a_ ** (x_ + y_):= a**x + a**y

Protect[NonCommutativeMultiply]

(* *************** Rules for simple functions *************** *)

Quaternion /:
    Conjugate[Quaternion[a_, b_, c_, d_]]:=
    Quaternion[a,-b,-c,-d]

Quaternion /:
    Norm[a:Quaternion[__?ScalarQ]]:=
    Apply[Plus, (Apply[List,a])^2]

Quaternion /:
    Abs[  a:Quaternion[__?ScalarQ]]:= Sqrt[Norm[a]]

Quaternion /:
    EvenQ[a:Quaternion[__?ScalarQ]]:= EvenQ[Norm[a]]

Quaternion /:
    OddQ[ a:Quaternion[__?ScalarQ]]:= OddQ[Norm[a]]

IntegerQuaternionQ[a_]:=
    If[QuaternionQ[a],
    Mod[2 * List @@ a, 1] == {0,0,0,0},
    False
    ]

Quaternion /:
    Round[a:Quaternion[__?ScalarQ]]:=
    Module[{h,j},
        h = Map[Round,a];
        j = Map[Floor,a] + Quaternion[1/2,1/2,1/2,1/2];
    If[Norm[a-h] <= Norm[a-j],
            h,
            j
        ]
    ]

Quaternion /:
    Mod[a:Quaternion[__?ScalarQ], b_]:= a - b ** Round[(b^-1) ** a]

Quaternion /:
    Mod[a_, b:Quaternion[__?ScalarQ]]:= a - b ** Round[(b^-1) ** a]

UnitQuaternions = {
    Quaternion[1,0,0,0], Quaternion[-1,0,0,0],
    Quaternion[0,1,0,0], Quaternion[0,-1,0,0],
    Quaternion[0,0,1,0], Quaternion[0,0,-1,0],
    Quaternion[0,0,0,1], Quaternion[0,0,0,-1],
    Quaternion[ 1/2, 1/2, 1/2, 1/2], Quaternion[-1/2,-1/2,-1/2,-1/2],
    Quaternion[-1/2, 1/2, 1/2, 1/2], Quaternion[ 1/2,-1/2,-1/2,-1/2],
    Quaternion[ 1/2,-1/2, 1/2, 1/2], Quaternion[-1/2, 1/2,-1/2,-1/2],
    Quaternion[ 1/2, 1/2,-1/2, 1/2], Quaternion[-1/2,-1/2, 1/2,-1/2],
    Quaternion[ 1/2, 1/2, 1/2,-1/2], Quaternion[-1/2,-1/2,-1/2, 1/2],
    Quaternion[-1/2,-1/2, 1/2, 1/2], Quaternion[ 1/2, 1/2,-1/2,-1/2],
    Quaternion[-1/2, 1/2,-1/2, 1/2], Quaternion[ 1/2,-1/2, 1/2,-1/2],
    Quaternion[-1/2, 1/2, 1/2,-1/2], Quaternion[ 1/2,-1/2,-1/2, 1/2]
}

UnitQuaternionQ[a_]:= MemberQ[UnitQuaternions, a]

LeftAssociates[a_]:=
    Sort[UnitQuaternions**a] /;
    MessageQuaternionQ[a, LeftAssociates[]]

RightAssociates[a_]:=
    Sort[a**UnitQuaternions] /;
    MessageQuaternionQ[a, RightAssociates[]]

PrimaryLeftAssociate[a:Quaternion[__?ScalarQ]]:=
    Last[Sort[LeftAssociates[a]]]

PrimaryRightAssociate[a:Quaternion[__?ScalarQ]]:=
    Last[Sort[RightAssociates[a]]]

RightGCD[a_,Quaternion[0,0,0,0]]:= PrimaryRightAssociate[a]

RightGCD[a_,b_]:= RightGCD[b,Mod[a,b]]

LeftGCD[a_,Quaternion[0,0,0,0]]:=
    Conjugate[PrimaryRightAssociate[Conjugate[a]]]

LeftGCD[a_,b_]:= Conjugate[RightGCD[Conjugate[b],Conjugate[Mod[a,b]]]]

quatprime[x_]:= If[IntegerQ[x],False,PrimeQ[Norm[x /. toquat]]]

(* toquat is a List of replacement rules that are used to transform exprs into
   Quaternion[] objects. This list could be improved.
*)

toquat = {
    Complex[x_,y_]:> Quaternion[x,y,0,0],
    Plus[x_, Times[Complex[0, 1], y_]]:> Quaternion[x,y,0,0],
    Times[Complex[0, 1], x_]:> Quaternion[0,x,0,0],
    Times[Complex[0,x_], y_]:> Quaternion[0,x y,0,0],
    x_. J:> Quaternion[0,0,x,0],
    x_. K:> Quaternion[0,0,0,x],
    Quaternion[w_,x_,y_,z_]:> Quaternion[w,x,y,z],
    x_:> Quaternion[x,0,0,0] /; ScalarQ[x]
}

ToQuaternion[a_]:= a /. toquat // QSimplify

FromQuaternion[a_]:=
    a[[1]] + a[[2]] I + a[[3]] J + a[[4]] K /;
    MessageQuaternionQ[a, FromQuaternion[]]

Unprotect[PrimeQ]

PrimeQ[num_, opt1___,Quaternions->False,opt2___]:=
    PrimeQ[num, Evaluate[
    Sequence @@ FilterRules[Flatten[{opt1, opt2}], Options[PrimeQ][[All, 1]]]]]

PrimeQ[num_, opt1___, Quaternions->True, opt2___]:=
    quatprime[num] /;
    If[(GaussianIntegers /. {opt1,opt2}) === True,
        Message[PrimeQ::incopt]; False,
        True
    ]

(* syntax coloring *)
SyntaxInformation[PrimeQ] =
 Append[DeleteCases[SyntaxInformation[PrimeQ], "OptionNames" -> _],
  "OptionNames" -> {"GaussianIntegers", "Quaternions"}]

Protect[PrimeQ]

Quaternion /:
    Re[a:Quaternion[__?ScalarQ]]:= First[a]

Quaternion /:
    Sign[a:Quaternion[__?ScalarQ]]:= a / Abs[a]

AdjustedSignIJK[a_Complex | a_?ScalarQ]:= I

AdjustedSignIJK[Quaternion[a_?ScalarQ, b_?ScalarQ, c_?ScalarQ, d_?ScalarQ]]:=
Which[
    b != 0,
    Sign[b] * Sign[Quaternion[0,b,c,d]],
    c != 0,
    Sign[c] * Sign[Quaternion[0,b,c,d]],
    d != 0,
    Sign[d] * Sign[Quaternion[0,b,c,d]],
    True,
    I  (* This seems very wrong. *)
]

AbsIJK[Quaternion[a_?ScalarQ, b_?ScalarQ, c_?ScalarQ, d_?ScalarQ]]:=
    Sqrt[b^2+c^2+d^2]

AbsIJK[a_?NumericQ] := Im[a]

(* ***************  Exponential, Trig, and Hyperbolics *************** *)
(* One can extend de Moivre's formula to work on the Quaternions.  One can
   then define the various Trig and Hyperbolic functions in terms of this.
*)


(* fix bug 66775 -- charlesp *)

SignIJK[Quaternion[a_?ScalarQ, b_?ScalarQ, c_?ScalarQ, d_?ScalarQ]] :=
	Sign[Quaternion[0, b, c, d]]

Quaternion /:
    Exp[a:Quaternion[__?ScalarQ]]:=
    Exp[Re[a]]*(Cos[AbsIJK[a]]+ Sin[AbsIJK[a]]*SignIJK[a]) // QSimplify

ExtendableFunctions = {
    Log,
    Cos, Sin, Tan, Sec, Csc, Cot,
    ArcCos, ArcSin, ArcTan, ArcSec, ArcCsc, ArcCot,
    Cosh, Sinh, Tanh, Sech, Csch, Coth,
    ArcCosh, ArcSinh, ArcTanh, ArcSech, ArcCsch, ArcCoth
}

extfunc[func_, a_, b_] :=
    Re[b] + AbsIJK[b]*AdjustedSignIJK[a]

Block[{extend, $Output={}},

    extend[func_]:= (
    	Unprotect[func];
        Quaternion/:func[a:Quaternion[__?ScalarQ]]:=
        extfunc[func, a, func[Re[a] + AbsIJK[a] * I]];
        Protect[func];
    );

    Map[extend,ExtendableFunctions];
]

toAngle[a_]:=
Module[{r,theta,mu,phi,gamma},
    r = AbsIJK[a];
    theta = If[a[[1]] =!= 0, ArcTan[r/a[[1]]], Pi/2];
    mu = If[Sin[theta] =!= 0, r/Sin[theta],Pi/2];
    phi = If[r =!= 0, ArcCos[a[[2]]/r],0];
    gamma = If[Sin[phi] =!= 0,
        ArcCos[a[[3]]/(r Sin[phi])] Sign[a[[4]]],
        ArcCos[a[[3]]/(r Pi /2)] Sign[a[[4]]]
    ];
    {r,mu,theta,phi,gamma}
]

(* *************** Rules for Power *************** *)
(* Power is essentially de Moivre's theorem for quaternions. *)

Quaternion /:
    Power[a:Quaternion[__?ScalarQ],0]:= 1

Quaternion /:
    Power[a:Quaternion[__?ScalarQ],n_]:= Power[1/a, -n] /; n < 0 && n != -1

Quaternion /:
    Power[a:Quaternion[__?ScalarQ],-1]:= Conjugate[a] ** 1/Norm[a]

Quaternion /:
    Power[a:Quaternion[__?ScalarQ], n_]:=
    Module[{angle, pqradius},
    angle = toAngle[a];
    pqradius = angle[[2]]^n Sin[n angle[[3]]];
    Quaternion[
        angle[[2]]^n Cos[n angle[[3]]],
        Cos[angle[[4]]] pqradius,
        Sin[angle[[4]]] Cos[angle[[5]]] pqradius,
        Sin[angle[[4]]] Sin[angle[[5]]] pqradius
    ]  // QSimplify
    ] /; ScalarQ[n] && n > 0

(* special case *)
Quaternion /:
    Power[E, a:Quaternion[__?ScalarQ]] := Exp[a]

(* Do the rules for Divide and Sqrt actually do anything?
   These are quickly turned into Power anyway.*)

Quaternion /:
    Sqrt[a:Quaternion[__?ScalarQ]] := Power[a,1/2]

(* It is possible to call Divide in such a way that this pattern
   is not matched. It is preferable to use x**(1/y) instead of
   x/y when dividing two quaternions .*)


Quaternion /:
    Divide[a:Quaternion[__?ScalarQ], b:Quaternion[__?ScalarQ]]:=
    a ** (Conjugate[b] ** 1/Norm[b])


QSimplify[a:Quaternion[__?ScalarQ]]:=Simplify[TrigExpand/@a]
QSimplify[a_] := a

(* *************** Error messages *************** *)

PrimeQ::incopt = "Incompatible options given to PrimeQ. \
GaussianIntegers and Quaternions can not both be set to True"

Quaternion::arg = "`1` called with the wrong number of arguments. \
One argument is expected."

Quaternion::argc = "`1` called with the wrong number of arguments. \
`2`arguments are expected."

Quaternion::notquat = "Quaternion expected at position `1` in `2`"

End[]  (* Quaternions`Private`*)

EndPackage[] (* Quaternions`*)

