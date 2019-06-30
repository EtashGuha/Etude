(* ::Package:: *)

(* Mathematica Package *)

(* Created by the Wolfram Workbench Aug 1, 2006 *)

(* :Name: ComputerArithmetic Package *)

(* :Title: Fixed Precision, Correctly Rounded Computer Arithmetic *)

(* :Author: Jerry B. Keiper *)

(* :Summary:
This package implements fixed precision, rounded arithmetic.  The 
arithmetic can be in any base from 2 to 16, and any of several 
rounding schemes can be used.  The range of the exponent can also be
varied within limits.  This package is not suitable for
computational purposes; it is much too slow.  Its real use is
educational, but it can be used in conjunction with other
packages such as LinearAlgebra`GaussianElimination`.
*)

(* :Context: ComputerArithmetic` *)

(* :Package Version: 1.0 *)

(* :Copyright: Copyright 1990-2007,  Wolfram Research, Inc.
*)

(* :History:
	Originally by Jerry B. Keiper, May 1990.
	Revised by Jerry B. Keiper, December 1990.
    Replaced option form of IdealDivide by new symbol IdealDivision.
      IdealDivide as an option is still supported, but should be phased
      out, possibly for Mathematica 7.0.
      - Brian Van Vertloo, January 2007.
*)

(* :Keywords: arithmetic, rounding, fixed precision *)

(* :Source:
	Any elementary numerical analysis textbook.
*)

(* :Mathematica Version: 2.0 *)

(* :Warning:
	The function Normal is extended to convert ComputerNumbers
	to their exact rational equivalent.
*)

(* :Limitations:
	Changing the arithmetic can result in previous ComputerNumbers
	becoming invalid.

	Only one type of arithmetic is allowed at a time.  It would not
	be difficult to extend this package to support different data
	types simultaneously, e.g., ComputerInteger, ComputerNumber,
	DoubleComputerNumber, etc.
*)

(* :Discussion:
	ComputerNumber[sign, mantissa, exp, value, x] is a data object that
	represents a computer number, but its default print format is a
	simple number in the base specified by SetArithmetic.  The fifth
	element of a ComputerNumber is the value that the number would have
	if the arithmetic had been done with high-precision arithmetic,
	i.e., by comparing the value of the ComputerNumber with this value
	you get the total accumulated roundoff error in the number.
	ComputerNumber[x] gives the complete data object representing
	the number x.  ComputerNumber[sign, mantissa, exp] gives the
	complete data object ComputerNumber[sign, mantissa, exp, value, x],
	where value and x have the value sign * mantissa * base^exp, where
	base is the the base specified by SetArithmetic.  The sign must
	be +1 or -1, the integer mantissa must be between base^(digits-1)
	and base^digits - 1, and the exponent must be an integer within a
	range specified by SetArithmetic.  The fourth element, value,
	is used only for efficiency.

	Basic arithmetic with these objects is automatic using any of
	several rounding schemes.  Although the rounding within the
	basic operations is correct, numerically converting a complicated
	expression involving transcendental functions to a ComputerNumber
	can result in incorrect rounding.  (Such errors are unavoidable.
	Although the package could be designed to correctly round any
	particular expression, there would always be some expressions
	for which the rounding would be incorrect.)  For typical expressions
	the rounding error will be less than .50000000000000000001 ulps
	for rounding and less than 1.00000000000000000001 ulps for
	truncation.  The basic arithmetic is easily extensible to the
	elementary functions and even to special functions.

	Note that the default division is really two operations:
	multiplication by the reciprocal.  True division is implemented
	by the function IdealDivide[x, y].  These two forms of division
	can give different results.
	
	The arithmetic implemented in this package is slightly better
	than most computer arithmetic: integers and rational numbers
	used in multiplication and division, and integers used as
	exponents are NOT first converted to ComputerNumbers, but
	rather they are used in their given form, and the final result is
	then converted to a ComputerNumber.
*)

(* :Examples:

In[1]:= << NumericalMath`ComputerArithmetic`	(* read in the package *)

Out[1]= NumericalMath`ComputerArithmetic`

In[2]:= Arithmetic[ ]

Out[2]= {4, 10, RoundingRule -> RoundToEven, ExponentRange -> {-50, 50},

>	MixedMode -> False}

		(* The default arithmetic is four digits in base 10 with a
		rounding rule of RoundToEven and numbers between
		10^-50 and .9999 10^50 allowed.  Mixed-mode arithmetic
		is not allowed. *)

In[3]:= ComputerNumber[Pi]	(* Expressions that can be interpreted as
				numbers are converted to ComputerNumbers. *)

Out[3]= 3.142

In[4]:= FullForm[%]

Out[4]//FullForm= ComputerNumber[1, 3142, -3, Rational[1571, 500],

>		3.14159265358979323846264]

In[5]:= {ComputerNumber[-1, 1234, -6], ComputerNumber[-1, 123, 7]}

Out[5]= {-0.001234, NaN}	(* You also can enter ComputerNumbers in
			terms of a sign, mantissa, and an exponent, but only
			if it forms a valid ComputerNumber.  (In this
			example the one mantissa was not four digits.) *)

In[6]:= ComputerNumber[Pi - 22/7]
		(* Expressions are evaluated numerically before
				they become "ComputerNumbers". *)

Out[6]= -0.001264

In[7]:= ComputerNumber[Pi] - ComputerNumber[22/7] (* This result is different. *)

Out[7]= -0.001

In[8]:= sum = 0 

Out[8]= 0

In[9]:= Do[sum += ComputerNumber[i]^(-2), {i, 200}]; FullForm[sum]

Out[9]= FullForm=

>   ComputerNumber[1, 1625, -3, Rational[13, 8], 1.63994654601499726794569]

In[10]:= (sum = 0; Do[sum += ComputerNumber[i]^(-2), {i, 200, 1, -1}];
	FullForm[sum])

Out[10]//FullForm=

>   ComputerNumber[1, 1640, -3, Rational[41, 25], 1.63994654601499726794569]

	(* As a general rule, it is better to sum the smaller terms first,
	but it does not guarantee a better result. *)

In[11]:= sum = 0; Do[sum += 1/ComputerNumber[i], {i, 300}]; FullForm[sum]

Out[11]//FullForm=

>   ComputerNumber[1, 6281, -3, Rational[6281, 1000], 6.28266388029950346191949]

In[12]:= sum = 0; Do[sum += 1/ComputerNumber[i], {i,300,1,-1}]; FullForm[sum]

Out[12]//FullForm=

>   ComputerNumber[1, 6280, -3, Rational[157, 25], 6.28266388029950346191949]

	(* The difference is slight, and such examples are rare. *)

In[13]:= ComputerNumber[Sin[Pi/7]]

Out[13]= 0.4339

In[14]:= FullForm[Sin[ComputerNumber[N[Pi]/7]]]

Out[14]//FullForm= Sin[ComputerNumber[1, 4488, -4, Rational[561, 1250],

>		0.448798950512827587999709]]

	(* Basic arithmetic is all that is implemented in the package.
	   We could easily extend things to include elementary functions. *)

In[15]:= sq = ComputerNumber[Sqrt[47]]

Out[15]= 6.856

In[16]:= sq sq

Out[16]= 47.

	(* It is a theorem that correctly rounded square roots of small
	integers will always square back to the original integer if
	the arithmetic is correct.  Such is not the case for cube roots. *)

In[17]:= cr = ComputerNumber[3^(1/3)]

Out[17]= 1.442

In[18]:= cr cr cr

Out[18]= 2.998

	(* We now want to work with seven significant digits. *)

In[19]:= SetArithmetic[7]

Out[19]= {7, 10, RoundingRule -> RoundToEven, ExponentRange -> {-50, 50}, 
 
>    MixedMode -> False}

	(* The arithmetic is now seven digits in base 10 with a rounding rule
	of RoundToEven and an exponent range of -50 to 50. *)

In[20]:= ComputerNumber[.9999999499999999999999999]

Out[20]= 0.9999999

In[21]:= ComputerNumber[.9999999500000000000000001]

Out[21]= 1.

	(* Note that correct rounding is performed even near the discontinuity
	in the exponent. *)


	(* The reciprocal of the reciprocal is not the original number; in
	fact it may be quite different. *)

In[22]:= x = ComputerNumber[9010004]

                    6
Out[22]= 9.010004 10

In[23]:= y = 1/x

		    -7
Out[23]= 1.109877 10

In[24]:= z = 1/y

		    6
Out[24]= 9.010007 10

	(* Likewise division (as a single operation) is different from
	multiplication by the reciprocal. *)

In[25]:= ComputerNumber[2]/x	(* this is multiplication by the reciprocal. *)

		    -7
Out[25]= 2.219754 10

In[26]:= IdealDivide[ComputerNumber[2], x]	(* this is true division. *)

		    -7
Out[26]= 2.219755 10

	(* Note: you can set the arithmetic to use IdealDivide
	automatically whenever it encounters the `/' divide symbol.
	This is an option to SetArithmetic, but it uses $PreRead
	and may interfere with other behavior that also uses $PreRead. *)

	(* Division by 0 and overflow conditions result in NaN: *)

In[27]:= IdealDivide[ComputerNumber[1], ComputerNumber[0]]

ComputerNumber::divzer: Division by 0 occurred in computation.

Out[27]= NaN

In[28]:= x = ComputerNumber[1000000000]

	      9
Out[28]= 1. 10

In[29]:= x^7

Out[29]= NaN

In[30]:= Pi < 22/7

	      22
Out[30]= Pi < --
	      7

In[31]:= ComputerNumber[Pi] < ComputerNumber[22/7]

Out[31]= True

In[32]:= SetArithmetic[3, 2, RoundingRule -> Truncation,
	ExponentRange -> {-3,3}]

	(* Set the arithmetic to be three digits in base 2 using the rounding
		rule of Truncation and allowed numbers between -7 and -1/8
		and 1/8 and 7. *)

Out[32] = {3, 2, RoundingRule -> Truncation, ExponentRange -> {-3, 3},

>	MixedMode -> False}

In[33]:= ComputerNumber[Pi]

Out[33]= 11.
	    2

	(* Pi is simply 3 in this very limited arithmetic. *)

In[34]:= Plot[Normal[ComputerNumber[x]] - x, {x, -10, 10}, PlotPoints -> 47]

	(* Plot the representation error of the numbers between -10 and 10. *)

In[35]:= Plot[Normal[ComputerNumber[x]] - x, {x, -1, 1}, PlotPoints -> 47]

	(* Resolve the wiggles near the hole at zero. *)

In[36]:= 2 ComputerNumber[Pi] - 4

Out[36]= -4 + 2 11.
		   2

	(* The default is to disallow mixed-mode arithmetic, i.e.,
	    arithmetic between ComputerNumbers and ordinary numbers.
	    (Integer and Rational exponents are allowed however.) *)

In[37]:= SetArithmetic[3, 2, RoundingRule -> Truncation, 
	ExponentRange -> {-3,3}, MixedMode -> True];

In[38]:= 2 ComputerNumber[Pi] - 4

Out[38]= 10.		(* Now mixed-mode arithmetic is allowed. *)
	    2
*)

BeginPackage["ComputerArithmetic`"]

Get[ToFileName["ComputerArithmetic","Microscope.m"]]

If[Not@ValueQ[MixedMode::usage],MixedMode::usage =
"MixedMode is an option to SetArithmetic that can be either True or False. \
It specifies whether mixed-mode arithmetic is to be allowed. The default is \
False."];

If[Not@ValueQ[RoundingRule::usage],RoundingRule::usage =
"RoundingRule is an option to SetArithmetic and specifies the rounding \
scheme to use. The choices are RoundToEven, RoundToInfinity, and Truncation. \
The default is RoundToEven."];

If[Not@ValueQ[RoundToEven::usage],RoundToEven::usage =
"RoundToEven is a choice for the option RoundingRule of SetArithmetic. \
It specifies that the rounding is to be to the nearest representable number \
and, in the case of a tie, round to the one represented by an even mantissa."];

If[Not@ValueQ[RoundToInfinity::usage],RoundToInfinity::usage =
"RoundToInfinity is a choice for the option RoundingRule of SetArithmetic. \
It specifies that the rounding is to be to the nearest representable number \
and, in the case of a tie, round away from 0."];

If[Not@ValueQ[Truncation::usage],Truncation::usage =
"Truncation is a choice for the option RoundingRule of SetArithmetic. \
It specifies that the ``rounding '' is to simply discard excess digits, as \
Floor does for positive numbers."];

If[Not@ValueQ[ExponentRange::usage],ExponentRange::usage =
"ExponentRange is an option to SetArithmetic and specifies the range of \
exponents that are to be allowed. The exponent range must be of the form \
{minexp, maxexp} where -1000 <= minexp < maxexp <= 1000.  The default \
ExponentRange is {-50, 50}."];

If[Not@ValueQ[SetArithmetic::usage],SetArithmetic::usage =
"SetArithmetic[dig] evaluates certain global constants used in the package \
ComputerArithmetic.m to make the arithmetic work properly with dig digits \
precision. The value of dig must be an integer between 1 and 10, inclusive, \
and the default value is 4. SetArithmetic[dig, base] causes the arithmetic \
to be dig digits in base base. The value of base must be an integer between \
2 and 16, and the default is 10. Changing the arithmetic and then attempting \
to refer to ComputerNumbers that were defined prior to the change can \
lead to unpredictable results."];

If[Not@ValueQ[Arithmetic::usage],Arithmetic::usage =
"Arithmetic[ ] gives a list containing the number of digits, the base, \
the rounding rule, and the exponent range that are currently in effect."];

If[Not@ValueQ[ComputerNumber::usage],ComputerNumber::usage =
"ComputerNumber[sign, mantissa, exp, value, x] is a data object that represents \
a computer number, but its default print format is a simple number in the base \
specified by SetArithmetic. ComputerNumber[x] gives the complete data object \
representing the number x. ComputerNumber[sign, mantissa, exp] likewise gives \
the complete data object ComputerNumber[sign, mantissa, exp, value, x] where \
value and x have the value sign * mantissa * base^exp. The arithmetic with \
value is computer arithmetic; the arithmetic with x is ordinary high-precision \
arithmetic. Normal[ComputerNumber[...]] gives value."];

If[Not@ValueQ[NaN::usage],NaN::usage =
"NaN is the symbol used in ComputerArithmetic.m to represent a nonrepresentable \
number. NaN stands for Not-a-Number."];

If[Not@ValueQ[IdealDivide::usage],IdealDivide::usage =
"IdealDivide[x, y] gives the correctly rounded result of x divided by \
y. The default `/' division operator in Mathematica is, in fact, multiplication \
by the reciprocal, involves two rounding errors, and can result in an \
incorrectly rounded quotient."];

If[Not@ValueQ[IdealDivision::usage],IdealDivision::usage =
"IdealDivision is an option to SetArithmetic that can be set to True or \
False indicating whether $PreRead should be used to translate the default \
'/' division operator to use IdealDivide."];

Unprotect[SetArithmetic, Arithmetic, ComputerNumber, NaN, IdealDivide, 
    IdealDivision, RoundToEven, RoundToInfinity, Truncation, RoundingRule, 
    ExponentRange];

Options[SetArithmetic] =
	{RoundingRule -> RoundToEven,
	ExponentRange -> {-50, 50},
	MixedMode -> False,
	IdealDivide -> False,
    IdealDivision -> False}

Begin["ComputerArithmetic`Private`"]

Arithmetic[ ] := {$digits, $base, RoundingRule -> $roundrule,
	ExponentRange -> {$minexp - 1, $maxexp} + $digits,
	MixedMode -> $mixedmode, IdealDivide -> $idealdivide,
    IdealDivision -> $idealdivide}

SetArithmetic::digs =
"The number of digits requested is not an integer between 1 and 10."

SetArithmetic::base = "The base requested is not an integer between 2 and 16."

SetArithmetic::rr =
"The rounding rule `1` is not RoundToEven, RoundToInfinity, or Truncation."

SetArithmetic::er =
"The exponent range `1` is not a pair of integers between -1000 and 1000."

SetArithmetic::mm =
"The option MixedMode -> `1` is neither True nor False."

SetArithmetic::id =
"The option IdealDivision -> `1` is neither True nor False."

SetArithmetic[n_Integer:4, base_Integer:10, opts___] :=
    Module[{tmp, er, mm, id, idold},
	If[!(1 <= n <= 10), Message[SetArithmetic::digs]; Return[$Failed]];
	If[!(2 <= base <= 16), Message[SetArithmetic::base]; Return[$Failed]];
	tmp = (RoundingRule /. {opts} /. Options[SetArithmetic]);
	If[!MemberQ[{RoundToEven, RoundToInfinity, Truncation}, tmp],
		Message[SetArithmetic::rr, tmp];
		Return[$Failed]];
	er = (ExponentRange /. {opts} /. Options[SetArithmetic]);
	If[!ListQ[er] || (Length[er] != 2) || !IntegerQ[er[[1]]] ||
			!IntegerQ[er[[2]]] || (er[[1]] >= er[[2]]) ||
			(er[[1]] < -1000) || (er[[2]] > 1000),
		Message[SetArithmetic::er, er];
		Return[$Failed]];
	mm = (MixedMode /. {opts} /. Options[SetArithmetic]);
	If[(mm =!= True) && (mm =!= False),
		Message[SetArithmetic::mm, mm];
		Return[$Failed]];
    id = (IdealDivision /. {opts} /. Options[SetArithmetic]);
	idold = (IdealDivide /. {opts} /. Options[SetArithmetic]);
	If[(id =!= True) && (id =!= False),
		Message[SetArithmetic::id, id];
		Return[$Failed]];
    If[(idold =!= True) && (idold =!= False),
        Message[SetArithmetic::id, idold];
        Return[$Failed]];
	$mixedmode = mm;
	$idealdivide = (id || idold);
	$roundrule = tmp;
	$digits = n;
	$base = base;
	$minman = base^(n-1);
	$maxman = base $minman - 1;
	$minexp = er[[1]] + 1 - n;
	$maxexp = er[[2]] - n;
	$prec = Round[n Log[10., base]];
	$nfdigits = Max[$prec+1, $MachinePrecision+3];
	$prec += 20;
	If[$idealdivide,
		$PreRead = DivideReplace,
		If[$PreRead === DivideReplace, $PreRead = .]
		];
	Update[ ];
	Arithmetic[ ]
    ]

DivideReplace[s_String] :=
	Module[{ss = StringReplace[s , "/" -> "~IdealDivide~"]},
		StringReplace[ss, {"~IdealDivide~~IdealDivide~" -> "//",
		"~IdealDivide~;" -> "/;", "~IdealDivide~:" -> "/:",
		"~IdealDivide~@" -> "/@", "~IdealDivide~." -> "/.",
		"~IdealDivide~=" -> "/="}]];

DivideReplace[b_] := (* boxes *)
   b//.{ "/" -> "~IdealDivide~",
         FractionBox[x_, y_] :> RowBox[{x, "~IdealDivide~", y}]}

If[!NumberQ[$digits], SetArithmetic[]];	(* initialization *)

ComputerNumber::undflw = "Underflow occurred in computation.  The exponent is ``."

ComputerNumber::ovrflw = "Overflow occurred in computation.  The exponent is ``."

ComputerNumber[sign_Integer, mantissa_Integer, exp_Integer] := 
	Module[{tmp, value},
		tmp = ((sign == 1) || (sign == -1));
		If[tmp,
			If[$minexp > exp,
				Message[ComputerNumber::undflw, exp + $digits];
				tmp = False];
			If[exp > $maxexp,
				Message[ComputerNumber::ovrflw, exp + $digits];
				tmp = False];
			];
		If[tmp && ($minman <= mantissa <= $maxman),
			value = sign mantissa $base^exp;
			tmp = SetPrecision[value, $prec];
			tmp = ComputerNumber[sign, mantissa, exp, value, tmp],
		    (* else *)
			If[tmp && (mantissa == 0),
				tmp = ComputerNumber[1,0,0,0,0],
			    (* else *)
				tmp = NaN
			]
		];
		tmp
	]

round[x_] :=
	Module[{rndx1, rndx2, d},
		If[$roundrule === Truncation, Return[Floor[x]]];
		rndx1 = Round[x];
		rndx2 = rndx1 + If[rndx1 < x, 1, -1];
		d = Abs[x - rndx1] - Abs[x - rndx2];
		Which[
			TrueQ[Negative[d]], rndx1,
			TrueQ[Positive[d]], rndx2,
			True, If[$roundrule === RoundToEven,
					If[ EvenQ[rndx1], rndx1, rndx2],
					Max[{rndx1, rndx2}]]
		]
	]

ComputerNumber[x_] := x /; !NumberQ[N[x]]

ComputerNumber[x_] :=
	Module[{mantissa, exp, nx = x, absnx, tmp, ok},
		If[!NumberQ[nx], nx = SetPrecision[N[nx,$prec],$prec]];
		If[!NumberQ[nx] || (Head[nx] === Complex), Return[NaN]];
		If[nx == 0, Return[ComputerNumber[1,0,0,0,0]]];
		absnx = Abs[nx];
		exp = Floor[Log[$base,N[absnx]]]-$digits+1;
		tmp = $base^exp;
		absnx /= tmp;
		mantissa = round[absnx];
		If[mantissa > $maxman,
			exp++;
			mantissa = round[absnx/$base]
		];
		ok = ($minman <= mantissa <= $maxman);
		If[ok,
			If[$minexp > exp,
				Message[ComputerNumber::undflw, exp + $digits];
				ok = False];
			If[exp > $maxexp,
				Message[ComputerNumber::ovrflw, exp + $digits];
				ok = False];
			];
		If[ok,
			nx = SetPrecision[nx, $prec];
			tmp = Sign[nx] mantissa $base^exp;
			ComputerNumber[Sign[nx], mantissa, exp, tmp, nx],
		    (* else *)
			NaN
		]
	]

Format[ComputerNumber[s_, m_, e_, v_, x_]] ^:= BaseForm[N[v, $nfdigits], $base]

Normal[ComputerNumber[s_, m_, e_, v_, x_]] ^:= v

NaN /: Abs[NaN] = NaN;
NaN /: Less[NaN, _] := False
NaN /: Less[_, NaN] := False
NaN /: LessEqual[NaN, _] := False
NaN /: LessEqual[_, NaN] := False
NaN /: Greater[NaN, _] := False
NaN /: GreaterEqual[_, NaN] := False
NaN /: GreaterEqual[NaN, _] := False
NaN /: Greater[_, NaN] := False
NaN /: Equal[NaN, _] := False
NaN /: Equal[_, NaN] := False
NaN /: Unequal[NaN, _] := True
NaN /: Unequal[_, NaN] := True
NaN /: Plus[NaN, ___] := NaN
NaN /: Times[NaN, ___] := NaN
NaN /: Power[NaN, _] := NaN
NaN /: Power[_, NaN] := NaN
NaN /: IdealDivide[NaN, _] := NaN
NaN /: IdealDivide[_, NaN] := NaN

ComputerNumber/:
    Abs[x_ComputerNumber] :=
	ComputerNumber[1, x[[2]], x[[3]], Abs[x[[4]]], Abs[x[[5]]]];

ComputerNumber/:
    Less[x_ComputerNumber, y_ComputerNumber] := x[[4]] < y[[4]]
ComputerNumber/:
    LessEqual[x_ComputerNumber, y_ComputerNumber] := x[[4]] <= y[[4]]
ComputerNumber/:
    Greater[x_ComputerNumber, y_ComputerNumber] := x[[4]] > y[[4]]
ComputerNumber/:
    GreaterEqual[x_ComputerNumber, y_ComputerNumber] := x[[4]] >= y[[4]]
ComputerNumber/:
    Equal[x_ComputerNumber, y_ComputerNumber] := x[[4]] == y[[4]]
ComputerNumber/:
    Unequal[x_ComputerNumber, y_ComputerNumber] := x[[4]] != y[[4]]

ComputerNumber/:
    Max[x_ComputerNumber, y_ComputerNumber] := If[x[[4]] > y[[4]], x, y]
ComputerNumber/:
    Min[x_ComputerNumber, y_ComputerNumber] := If[x[[4]] < y[[4]], x, y]

ComputerNumber/:
    Floor[x_ComputerNumber] := ComputerNumber[Floor[x[[4]]]]
ComputerNumber/:
    Ceiling[x_ComputerNumber] := ComputerNumber[Ceiling[x[[4]]]]
ComputerNumber/:
    Round[x_ComputerNumber] := ComputerNumber[Round[x[[4]]]]


ComputerNumber/:
Plus[x_ComputerNumber, y_ComputerNumber] :=
	Module[{tmp},
		tmp = ComputerNumber[x[[4]] + y[[4]]];
		If[Head[tmp] === ComputerNumber,
			tmp[[5]] = x[[5]] + y[[5]],
		    (* else *)
			tmp = NaN
		];
		tmp
	]

ComputerNumber/:
Times[-1, ComputerNumber[s_, m_, e_, v_, x_]] :=
	ComputerNumber[-s, m, e, -v, -x];

ComputerNumber/:
Times[x_ComputerNumber, y_ComputerNumber] :=
	Module[{tmp},
		tmp = ComputerNumber[x[[4]] y[[4]]];
		If[Head[tmp] === ComputerNumber,
			tmp[[5]] = x[[5]] y[[5]],
		    (* else *)
			tmp = NaN
		];
		tmp
	]

ComputerNumber::divzer = "Division by 0 occurred in computation."

IdealDivide[x_ComputerNumber, y_ComputerNumber] :=
	Module[{tmp},
		If[y[[2]] == 0,
			Message[ComputerNumber::divzer];
			Return[NaN]];
		tmp = ComputerNumber[x[[4]]/y[[4]]];
		If[Head[tmp] === ComputerNumber,
			tmp[[5]] = x[[5]]/y[[5]],
		    (* else *)
			tmp = NaN
		];
		tmp
	]

ComputerNumber/:
Power[x_ComputerNumber, (n_Integer | n_Rational)] :=
	Module[{tmp},
		If[(x[[2]] == 0) && (n <= 0),
			Message[ComputerNumber::divzer];
			Return[NaN]];
		tmp = ComputerNumber[x[[4]]^n];
		If[Head[tmp] === ComputerNumber,
			tmp[[5]] = x[[5]]^n,
		    (* else *)
			tmp = NaN
		];
		tmp
	]

ComputerNumber/:
Power[x_ComputerNumber, y_ComputerNumber] :=
	Module[{tmp},
		If[y[[2]] == 0, Return[ComputerNumber[1]]];
		tmp = ComputerNumber[x[[4]]^y[[4]]];
		If[Head[tmp] === ComputerNumber,
			tmp[[5]] = x[[5]]^y[[5]],
		    (* else *)
			tmp = NaN
		];
		tmp
	]

(* the following are only active under mixed-mode arithmetic. *)

ComputerNumber/:
Times[(n_Integer | n_Rational), y_ComputerNumber] :=
	Module[{tmp},
		tmp = ComputerNumber[n y[[4]]];
		If[Head[tmp] === ComputerNumber,
			tmp[[5]] = n y[[5]],
		    (* else *)
			tmp = NaN
		];
		tmp
	] /; $mixedmode

IdealDivide[(n_Integer | n_Rational), y_ComputerNumber] :=
	Module[{tmp},
		If[y[[2]] == 0,
			Message[ComputerNumber::divzer];
			Return[NaN]];
		tmp = ComputerNumber[n/y[[4]]];
		If[Head[tmp] === ComputerNumber,
			tmp[[5]] = n/y[[5]],
		    (* else *)
			tmp = NaN
		];
		tmp
	] /; $mixedmode

IdealDivide[x_ComputerNumber, (n_Integer | n_Rational)] :=
	Module[{tmp},
		If[n == 0,
			Message[ComputerNumber::divzer];
			Return[NaN]];
		tmp = ComputerNumber[x[[4]]/n];
		If[Head[tmp] === ComputerNumber,
			tmp[[5]] = x[[5]]/n,
		    (* else *)
			tmp = NaN
		];
		tmp
	] /; $mixedmode

ComputerNumber/:
Plus[x_ComputerNumber, y_?NumberQ] := x + ComputerNumber[y] /; $mixedmode

ComputerNumber/:
Times[x_ComputerNumber, y_?NumberQ] := x ComputerNumber[y] /; $mixedmode

IdealDivide[x_ComputerNumber, y_?NumberQ] :=
	IdealDivide[x,ComputerNumber[y]] /; $mixedmode

IdealDivide[x_?NumberQ, y_ComputerNumber] :=
	IdealDivide[ComputerNumber[x],y] /; $mixedmode

IdealDivide[x_, y_] := Divide[x, y];	(* all other cases *)

ComputerNumber/:
Power[x_ComputerNumber, y_?NumberQ] := x^ComputerNumber[y] /; $mixedmode

ComputerNumber/:
Power[x_?NumberQ, y_ComputerNumber] := ComputerNumber[x]^y /; $mixedmode

End[ ] (* "ComputerArithmetic`Private`" *)

Protect[SetArithmetic, Arithmetic, ComputerNumber, NaN, IdealDivide,
    IdealDivision, RoundToEven, RoundToInfinity, Truncation, RoundingRule, 
    ExponentRange];

EndPackage[ ] (* "ComputerArithmetic`" *)


