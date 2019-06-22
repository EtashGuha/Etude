(* ::Package:: *)

(*:Name: FiniteFields` *)

(*:Author: Matthew Markert,
    with ideas from Emily Martin, Ilan Vardi, Stephen Wolfram *)

(*:Context: FiniteFields` *)

(*:Package Version: 1.2 *)

(*:Copyright: Copyright 1993-2007,  Wolfram Research, Inc.*)

(*:History:
   Version 1.1 by Matthew Markert, 1993
   Version 1.2 by Dan Lichtblau with John Novak, February 2000 --
       modifies the algorithm used for PowerList
*)

(*:Keywords:
    algebra, finite fields, discrete mathematics, coding theory *)

(*:Requirements: none. *)

(*:Warnings:
    There are probably functions in Mathematica which will be
    confused by the redefinition of basic arithmetic.  It would
    be prudent to avoid passing GF objects to numerical functions
    or to Integrate for example.
*)

(*:Source: Any graduate-level introductory algebra text. *)

(*:Limitations:
    Many Mathematica functions assume they are working over a
    specific ring, usually the integers, the rationals, or the
    complex numbers.  They will not necessarily produce correct
    answers with finite field elements.  For instance, Eigensystem
    sometimes will work, but in general won't.
*)

(*:Summary:
This package defines arithmetic in finite fields, along with
various supporting functions for working with finite field
objects.
*)

(*:Discussion:
The data structure GF[prime,poly][data] represents finite field
elements. Once the rules of arithmetic have been generalized to
this data structure, other functions should operate on them
normally.  Thus PolynomialQuotient[f, g, x] will give the correct
answer when applied to polynomials f, g, in x with coeffients of
the form GF[prime,irred][data].
*)

BeginPackage["FiniteFields`"]

(* Usage Messages *)

If[Not@ValueQ[GF::usage],GF::usage =
"GF[p, ilist][elist] represents an element of a finite field (Galois field) \
with prime characteristic p and where the coefficients of the irreducible \
polynomial are given by ilist (constant term listed first). The particular \
element is specified by elist, the list of coefficients of the polynomial \
representation of the element. GF[p, d], p prime and d a positive integer, \
is a special form giving a Galois field that is a degree d extension of the \
prime field of p elements. The resulting field has p^d elements."];

If[Not@ValueQ[SetFieldFormat::usage],SetFieldFormat::usage =
"SetFieldFormat[f] sets the output form of elements in field f. The default \
(FormatType -> Subscripted) is to set the OutputForm to be a subscripted vector. \
FormatType -> FullForm sets the OutputForm to FullForm. SetFieldFormat can \
also be used to set the input form. FormatType -> FunctionOfCoefficients[g] \
sets both input and output to have the functional notation g[c0, c1, ...] where \
c0, c1, ... are the coefficients in the polynomial representation of an \
element. FormatType -> FunctionOfCode[g] sets both input and output to have \
the functional notation g[c] where c is the integer code specifying an element."];
Options[SetFieldFormat] = {FormatType -> Subscripted}

If[Not@ValueQ[FunctionOfCoefficients::usage],FunctionOfCoefficients::usage =
"FunctionOfCoefficients[g] is a value of the option FormatType, used to \
specify the format g[c0, c1, ...] for an element in field f, where c0, c1, ... \
are the coefficients in the polynomial representation of the element. g can \
be any symbol."];

If[Not@ValueQ[FunctionOfCode::usage],FunctionOfCode::usage =
"FunctionOfCode[g] is a value of the option FormatType, used to specify the \
format g[c] for an element in field f, where c is the integer code for the \
element. g can be any symbol."];

If[Not@ValueQ[PerfectPowerQ::usage],PerfectPowerQ::usage =
"PerfectPowerQ[element, n] returns True if element is a perfect nth power in \
its field. Otherwise it returns False."];


(* ======================= field parameter functions ===================== *)

If[Not@ValueQ[IrreduciblePolynomial::usage],IrreduciblePolynomial::usage =
"IrreduciblePolynomial[s, p, d] finds an irreducible polynomial in the symbol s \
of degree d over the integers mod prime p."];

If[Not@ValueQ[FieldIrreducible::usage],FieldIrreducible::usage =
"FieldIrreducible[f, s] gives the irreducible polynomial in symbol s associated \
with the field f."];

If[Not@ValueQ[ExtensionDegree::usage],ExtensionDegree::usage =
"ExtensionDegree[f] gives the degree of the extension of the field f over its \
base field."];

If[Not@ValueQ[Characteristic::usage],Characteristic::usage =
"Characteristic[f] gives the characteristic of field f. The characteristic \
must be prime."];

If[StringQ[FieldSize::usage] &&
      StringPosition[FieldSize::usage, "finite field"] === {},
FieldSize::usage = FieldSize::usage <>
" FieldSize[f] gives the number of elements in the finite field f.";
]


(* ==================== use of discrete exponential table ================== *)

If[Not@ValueQ[PowerList::usage],PowerList::usage =
"PowerList[f] returns a list of the data parts of the non-zero \
elements of the field f. The first element in the list represents the \
multiplicative identity. The second element represents a primitive element of \
the field. The rest of the list represents successive powers of the primitive \
element."];

If[Not@ValueQ[PowerListQ::usage],PowerListQ::usage =
"PowerListQ[f] gives True if the list representing the powers of a primitive \
element of the field is used to do field arithmetic, False otherwise. Setting \
PowerListQ[f] = True computes the list if it does not yet exist, and enables \
arithmetic based on FieldExp and FieldInd. Setting PowerListQ[f] = False \
disables this arithmetic, but does not destroy the definitions of FieldExp \
and FieldInd."];

If[Not@ValueQ[FieldExp::usage],FieldExp::usage =
"FieldExp[f, n] gives the value of the discrete exponential function associated \
with the field f for integer n. The value of this function depends on the \
choice of primitive element, which depends on the choice of irreducible \
polynomial. FieldExp is defined only if PowerListQ has been set True for \
the field. FieldExp[f, 1] gives the primitive element used by the function. \
FieldExp[f, -Infinity] gives 0."];

If[Not@ValueQ[FieldInd::usage],FieldInd::usage =
"FieldInd[element] gives the power to which the primitive element must be \
raised in order to get the specified element. FieldInd is defined only if \
PowerListQ has been set True for the field. FieldInd[0] gives -Infinity."];


(* =================== field element manipulation ================ *)

If[Not@ValueQ[ReduceElement::usage],ReduceElement::usage =
"ReduceElement[element] gives a field element in reduced form. It is \
applied automatically to the results of arithmetic."];

If[Not@ValueQ[PowerListToField::usage],PowerListToField::usage =
"PowerListToField[list] gives a field object based on a list of the type \
produced by PowerList. PowerListQ for this field is set to True."];

If[Not@ValueQ[ToElementCode::usage],ToElementCode::usage =
"ToElementCode[element] gives a non-negative integer code, less than the \
field size, associated with the specified field element."];

If[Not@ValueQ[FromElementCode::usage],FromElementCode::usage =
"FromElementCode[f, code] gives the field element of f associated with code, a \
non-negative integer less than FieldSize[f]."];

If[Not@ValueQ[Successor::usage],Successor::usage =
"Successor[element] gives the next element in a canonical ordering of the \
field elements. This function does not wrap. The largest element has no \
successor."];

If[Not@ValueQ[PolynomialToElement::usage],PolynomialToElement::usage =
"PolynomialToElement[f, poly] gives an element in the field f corresponding \
to the polynomial in one symbol with integer coefficients given by poly."];

If[Not@ValueQ[ElementToPolynomial::usage],ElementToPolynomial::usage =
"ElementToPolynomial[element, s] gives a polynomial in the symbol s \
corresponding to the specified field element. ElementToPolynomial[f, s] \
gives the irreducible polynomial in s of the field f."];


Begin["`Private`"]

(* ************** Basic parameters of a finite field ************* *)

ExtensionDegree[GF[char_,irred_List]] :=
ExtensionDegree[GF[char,irred]] =
    (Length[irred] - 1)

GF /: FieldSize[GF[char_,irred_List]] :=
FieldSize[GF[char,irred]] ^=
    char^ExtensionDegree[GF[char,irred]]

Characteristic[GF[char_,irred_List]] :=
Characteristic[GF[char,irred]] = char

FieldIrreducible[GF[char_,irred_List]] :=
FieldIrreducible[GF[char,irred]] =
    VectorToPolynomial[irred, FieldSymbol[GF[char,irred]]]

FieldIrreducible[GF[char_,irred_List], s_Symbol] :=
    VectorToPolynomial[irred, s]

FieldSymbol[GF[char_,irred_List]] :=
FieldSymbol[GF[char,irred]] = Unique["FiniteFields`gf"]

DotVector[GF[char_,irred_List]] :=
DotVector[GF[char,irred]] =
    FieldSymbol[GF[char,irred]]^Range[0,Length[irred]-2]

(* ************ General cases for PowerList Functions ************ *)

PowerListExistsQ[_] := False

UsePowerListQ[_] := False

PowerListQ[gf_GF] := UsePowerListQ[gf]

PowerListQ /:
    Set[PowerListQ[f_GF], True] :=
     (
        If[TrueQ[PowerListExistsQ[f]],
            UsePowerListQ[f] = True,
            ConvertToExpTable[f]
    ]
     )

PowerListQ /:
    Set[PowerListQ[f_GF], False] := (UsePowerListQ[f] = False;)

(*  PowerListExistsQ is set universally to False when the package
    is loaded. It is only set to True in MakeExpTableField, which
    is called by both ConvertToExpTable and PowerListToField.
    Once it is set True, it is never set False.

    UsePowerListQ is also set universally to False at load time.
    It is also set to True by ConvertToExpTable.
    In addition it can be set to True or False by setting
    PowerListQ to True or False. *)

(* ********************* Automatic Reduction ********************* *)

GF[char_Integer?Positive, irred_List][data_List] :=
    polyReduce[
    GF[char,irred],
    VectorToPolynomial[data, FieldSymbol[GF[char,irred]]]
    ] /; Length[data] =!= ExtensionDegree[GF[char, irred]]

(* ********************* Define Data Object ********************** *)

Min2[n_] := Positive[n] && (n=!=1)

PositivePrimeQ[n_] := (Positive[n]  && PrimeQ[n])

GF::form =
"In GF[`1`], `1` must be prime or a power of a prime.";

GF::form2 =
"In GF[`1`, `2`], the first argument `1` must be prime and the second argument `2` must be a positive integer.";

GF[char_Integer?Min2] :=                (* e.g. GF[81] --> GF[3,4] *)
Block[{f = FactorInteger[char, Automatic]},
    (GF[char] = GF @@ First[f])/;If[ListQ[f] && Length[f] === 1 && PrimeQ[f[[1,1]]], True, Message[GF::form, char];False ]
]

GF[char_Integer?PositivePrimeQ, 1] :=   (* GF[7,1] --> GF[7,{0,1}] *)
GF[char, 1] = GF[char,{0,1}]

GF[char_Integer, degree_Integer] :=
(GF[char, degree] =
Module[{x,irred},
    irred = CoefficientList[IrreduciblePolynomial[x,char,degree],x];
    GF[char,irred]
])/;If[PositivePrimeQ[char] && Min2[degree], True, Message[GF::form2, char, degree];False]

(* ***************** Find Irreducible Polynomial ***************** *)

(* Find an irreducible polynomial modulo p to generate the finite
   field of p^n elements.  Ideally we will have an irreducible
   polynomial so that {0,1} is a primitive element.

   'x' is the symbol in which the polynomial is expressed
   'p' is the characteristic of the field
   'degree' is the degree of the polynomial and so the extension *)

IrreduciblePolynomial[x_Symbol, p_Integer, 1] := x;

IrreduciblePolynomial[
    x_Symbol,
    p_Integer?PrimeQ,
    degree_Integer?Positive] :=
Module[{irred},
    irred = OneIrreducible[x,p,degree];
    TransformIrreducible[ irred, findprim[irred,p], p]
]

(*  OneIrreducible looks for a polynomial which is irreducible mod p
    by a brute force search.  Since the density of such polynomials
    is reasonably high, the method is not too bad.  The old method,
    factoring a large cyclotomic polynomial, was ok for small fields.
    But when the number of elements in the field grew larger than
    about 1000, it took too much time and space to factor the
    polynomial.*)

OneIrreducible[
    x_Symbol,
    p_Integer?PrimeQ,
    degree_Integer?Positive] :=
Module[{dottbl,i},
    dottbl = x^Range[0,degree-1];
    i = (p^(degree-1));
    While[
        Head[
            Factor[x^degree + dottbl . IntegerDigits[i,p],Modulus->p]
        ] =!= Plus,
        i++
    ];
    x^degree + dottbl . IntegerDigits[i,p]
]

(*  Here we assume we have a monic irreducible polynomial 'irred'
    and a polynomial 'prim' which is a primitive field element with
    respect to irred.  To simplify the discussion, let us assume that
    the polynomials are both in x. We are looking for an irreducible
    polynomial which has x as a primitive element.  So, we want an
    automorphism of the field which maps the prim to x.  An
    automorphism of the field is a multiplication-preserving linear
    transformation of the vector space.  First, we find the linear
    transformation, then we use it to find the new irreducible.
    Since prim is primitive, {1,prim,...,prim^(degree-1)} is a basis
    for the vector space.  The transformation mapping {1,prim,...} to
    {1,x,...} is given by the inverse of the matrix which maps
    {1,x,...} to {1,prim,...}.  It is somewhat more natural in
    Mathematica to work with the transposes of the conventional
    transformation matrices.Once we have the transformation, we use
    it to find the image of prim^degree.  The irreducible
    polynomial we want is then just x^degree - image[prim^degree].*)

TransformIrreducible[ irred_, prim_, char_ ] :=
Module[{sym,deg,mat},
    sym = First[Variables[irred]];
    deg = Exponent[irred,sym];
    mat =
        Map[
            Drop[CoefficientList[sym^(deg) + #,sym],-1]&,
            NestList[PolynomialMod[prim #, {irred,char}]&, 1, deg]
        ];
    sym^deg +
        (Mod[-(Last[mat] . Inverse[Drop[mat,-1],Modulus->char]),char] .
        Map[(sym^#)&,Range[0,deg-1]])
]

(* **************** The Basic Reduction Function ***************** *)

(* An element in the finite field of p^n elements is a polynomial
   modulo an irreducible polynomial and modulo p. *)

ReduceElement[0] = 0;

ReduceElement[GF[char_,irred_List][f_]] :=
Module[{f1,x},
    f1 = VectorToPolynomial[Mod[f,char],x];
    If[ Exponent[f1,x] >= ExtensionDegree[GF[char,irred]],
        f1 = PolynomialMod[f1,{VectorToPolynomial[irred,x],char}]
    ];
    If[ f1 === 0, 0, GF[char,irred][CoefficientList[f1,x]] ]
]

(* vReduce assumes it is dealing with an integer vector of the
   right length.*)

vReduce[ GF[char_,irred_List], data_ ] :=
    Mod[data,char] // If[(Plus@@#)===0,0,GF[char,irred][#]]&

(* polyReduce does more error checking. *)

FieldPolyMod[ GF[char_,irred_List], poly_] :=
    PolynomialMod[poly, {FieldIrreducible[GF[char,irred]],char}]

polyReduce[ GF[char_,irred_List], poly_] :=
    FieldPolyMod[GF[char,irred],poly] //
        If[#===0,0,
            GF[char,irred][
                Drop[
            CoefficientList[
            FieldSymbol[GF[char,irred]]^(Length[irred]-1)+#,
            FieldSymbol[GF[char,irred]]
            ],
        -1
        ]
        ]
    ]&

ePowerMod[ GF[char_,irred_List][f_], k_Integer?Positive ] :=
Block[{nl,id},
    Which[
    k < 4,
        polyReduce[GF[char,irred],
        (f . DotVector[GF[char,irred]])^k
        ],
        k == 4,
        polyReduce[GF[char,irred],
        FieldPolyMod[GF[char,irred],
            (f . DotVector[GF[char,irred]])^2]^2
        ],
        k > 4,
        id = Reverse[IntegerDigits[k,2]];
        nl = NestList[
        FieldPolyMod[GF[char,irred],#^2]&,
        f . DotVector[GF[char,irred]],
        Length[id]-1
            ];
        polyReduce[GF[char,irred],
            Inner[Power,nl,id,Times]
        ]
    ]
]

(* ******** The field laws for elements in a finite field ******** *)

(* Addition *)
(* Addition works the same way in all field representations we use.*)

GF /: GF[char_,irred_List][f_] + m_Integer :=
    vReduce[GF[char,irred], Prepend[Rest[f],First[f]+m]]

GF /: GF[char_,irred_List][f_] + GF[char_,irred_][g_] :=
    vReduce[GF[char,irred], f+g]

(* Here is an important distributive law.  Unfortunately it must be
   associated with Plus since GF is buried too deeply inside. *)

Unprotect[Plus];
w_ * GF[char_,irred_List][f_] + w_ * GF[char_,irred_][g_] :=
     w * vReduce[GF[char,irred], f+g];
Protect[Plus];

(* Multiplication *)

GF /: GF[char_,irred_List][f_] * m_Integer :=
    vReduce[GF[char,irred], f*m]

GF /:
GF[char_,irred_List][{f_Integer}] * GF[char_,irred_List][{g_Integer}] :=
    GF[char,irred][{Mod[f*g,char]}]

GF /: GF[char_,irred_List][f_] * GF[char_,irred_List][g_] :=
    polyReduce[
    GF[char,irred],
    (f . DotVector[GF[char,irred]])*
        (g . DotVector[GF[char,irred]])
    ] /; Not[UsePowerListQ[GF[char,irred]]]

GF /: GF[char_,irred_List][f_] * GF[char_,irred_List][g_] :=
    FieldExp[GF[char,irred],
    Mod[
        FieldInd[GF[char,irred][f]] +
            FieldInd[GF[char,irred][g]],
        FieldSize[GF[char,irred]]-1
    ]
    ] /; UsePowerListQ[GF[char,irred]]

(* Powers and Reciprocals *)

GF /: GF[char_,irred_List][{f_Integer}]^k_Integer?Positive :=
    GF[char,irred][{PowerMod[f,k,char]}]

GF /: GF[char_,irred_List][{f_Integer}]^k_Integer?Negative :=
    GF[char,irred][{PowerMod[f,k,char]}] /;
    (Mod[f,char] =!= 0)

GF /: GF[char_,irred_List][f_]^k_Integer?Positive :=
    ePowerMod[GF[char,irred][f], k ] /;
    Not[UsePowerListQ[GF[char,irred]]]

(* For the mod-irreducible-polynomial case, taking a reciprocal
   means finding the ExtendedGCD with the irreducible polynomial. *)

GF /: Power[GF[char_,irred_List][f_], -1] :=
    polyReduce[GF[char,irred], gcdinverse[GF[char,irred],f]] /;
    Not[UsePowerListQ[GF[char,irred]]]

GF /: Power[GF[char_,irred_List][f_], k_Integer?Negative] :=
    polyReduce[
    GF[char,irred],
    gcdinverse[GF[char,irred],f]^(-k)
    ] /; Not[UsePowerListQ[GF[char,irred]]]

GF /: Power[GF[char_,irred_List][f_], k_Integer] :=
    FieldExp[ GF[char,irred],
    Mod[
        k*FieldInd[GF[char,irred][f]],
        FieldSize[GF[char,irred]]-1
    ]
    ] /; UsePowerListQ[GF[char,irred]]

(* ********* Computing the inverse w.r.t the irreducible ********* *)

(* PolynomialRemainder[f, g, x], and PolynomialQuotient[f, g, x]
   don't have the option Modulus -> p, so if g is not monic we
   have to compute inverses modulo p to make them monic. *)

gcdinverse[FieldTag_, f_] :=
Module[{fr = f . DotVector[FieldTag]},
    If[ IntegerQ[fr],
    PowerMod[fr, -1, Characteristic[FieldTag]],
    gcdi[FieldTag,fr]
    ]
]

gcdi[FieldTag_, f_] :=
Module[{
    fm = f,
    gm = FieldIrreducible[FieldTag],
    fv = {1, 0},
    gv = {0, 1},
    y = FieldSymbol[FieldTag],
    p = Characteristic[FieldTag],
    q,
    r = FieldSymbol[FieldTag],
    leading,
    monic
    }, (* end of local variables *)

    If[Exponent[fm, y] == 0,
    PowerMod[fm, -1, p],
    While[Exponent[r, y] > 0,
        leading = Coefficient[fm, y, Exponent[fm, y]];
        monic = PowerMod[leading, -1, p];
        q = PolynomialQuotient[
            PolynomialMod[monic gm, p],
            PolynomialMod[monic fm, p],
            y
            ];
        r = leading *
            PolynomialRemainder[
            PolynomialMod[monic gm, p],
            PolynomialMod[monic fm, p],
            y
            ];
        {gm, fm, gv, fv}  =
        PolynomialMod[#, p] & /@ {fm, r, fv, gv - q fv};
    ];
    If[fm === 0,
        leading = Coefficient[gm, y, Exponent[gm, y]];
        PolynomialMod[PowerMod[leading, -1, p] First[gv], p],
        PolynomialMod[PowerMod[fm, -1, p] First[fv],p]
    ]
    ]
]

(* ********************** Fractional Powers ********************** *)

(* Let p be the characteristic of a finite field F.  If F is a prime
   field then every element is its own pth root (Fermat's Little
   Theorem).  In general, if the field size is p^n then for all a in
   F, a^(p^n) == a, so a^(p^(n-1)) is the unique pth root of a. *)

GF /: Power[GF[char_,irred_List][f_], Rational[1,char_]] :=
    GF[char,irred][f] /; irred === {0,1}

GF /: Power[GF[char_,irred_List][f_], Rational[1,char_]] :=
    Power[GF[char,irred][f], char^(Length[irred]-2)]

PerfectPowerQ[GF[char_,irred_List][f_], m_Integer?Positive] :=
Block[{e,s,gcd},
    e = GF[char,irred][f];
    s = FieldSize[GF[char,irred]]-1;
    gcd = GCD[m,s];
    If[gcd === 1,
    True,
    e === e^(1+s/gcd)
    ]
]

(* *********************** Field Formatting ********************** *)

(* Default OutputForm *)

DefaultFormatOn[] := (
    Format[GF[c_Integer,irred_][data_List]] :=
    Subscript[data, c]
)

DefaultFormatOff[] :=
    (Format[GF[c_Integer,irred_][data_List]] =.)

DefaultFormatOn[];

(* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *)
(* visible formatting procedure *)

SetFieldFormat::ukft =  "Unknown FormatType `1` in SetFieldFormat.";

SetFieldFormat[ field:GF[_,_List], opts___ ] :=
Block[{ft},
    ft = FormatType /. {opts} /. Options[SetFieldFormat];

    Which[
    ft === FullForm,
        SetFieldFormat0[ field ],
    ft === FormatType || ft === Subscripted,
        SetFieldFormat1[ field ],
    MatchQ[ft, FunctionOfCoefficients[_Symbol]    ],
        SetFieldFormat2[ field, ft[[1]] ],
    MatchQ[ft, FunctionOfCode[_Symbol]   ],
        SetFieldFormat3[ field, ft[[1]] ],
    True,
        Message[SetFieldFormat::ukft, ft]
    ]
]

(* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *)
(* FullForm formatting *)

SetFieldFormat0[ field_ ] := (
    DefaultFormatOff[];
    Format[ field[data_List] ] := FullForm[ field[data] ];
    DefaultFormatOn[]; (* restore for all other fields *)
)

(* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *)
(* reassert default formatting *)

SetFieldFormat1[ field_ ] := (
    DefaultFormatOff[];
    Format[ field[data_List] ] = Subscript[data, field[[1]]] ;
    DefaultFormatOn[];
)

(* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *)
(* coefficient sequence format *)

SetFieldFormat2[ field_, IOHead_Symbol ] := (

    Clear[IOHead];
    IOHead[data___] := ReduceElement[field[{data}]];

    FieldOutputForm[ field, data_ ] :=
    ToExpression[
        "HoldForm[" <> ToString[IOHead]<>"["<>
        StringDrop[StringDrop[
        ToString[data //. {a___, 0} -> {a} ],1],-1] <> "]]"
    ];

    DefaultFormatOff[];
    Format[ field[data_List] ] :=
    FieldOutputForm[ field, data ];
    DefaultFormatOn[];
)

(* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *)
(* integer-encoded format *)

SetFieldFormat3[ field_, IOHead_Symbol ] := (

    Clear[IOHead];
    IOHead[data_Integer] :=
    FromElementCode[field,Mod[data,FieldSize[field]]];

    FieldOutputForm[ field, data_ ] :=
    ToExpression[
        "HoldForm[" <> ToString[IOHead]<>"["<>
        ToString[ToElementCode[field[data]]] <> "]]"
    ];

    DefaultFormatOff[];
    Format[ field[data_List] ] :=
    FieldOutputForm[ field, data ];
    DefaultFormatOn[];
)

(* ******************* Declare ExpTable Field ******************** *)

(* In the following we allow the user to enter an ExpTable directly.
   This is a little dangerous since there are many ways such a table
   can be in error. We do some error checking, but it is by no means
   complete.  One special precaution is that since we represent
   polynomials with the constant term first, the sublists
   representing the elements will be reversed with respect to tables
   such as those in Lidl and Niederreiter. The program tries to
   detect this and correct it by mapping Reverse over the table. *)

GF::etsz =
  "A discrete exponential table must have one less thana prime power number of sublists.";

GF::etnim =
  "A discrete exponential table must be a rectangular matrix (list of lists) of integers in the range 0 to p-1.  For this table p == `1`.";

GF::etdeg =
  "In a discrete exponential table with `1` entries, each entry must have length `2`.";

GF::etxnf =
  "An element is missing from the proposed discrete exponential table.";

GF::etbfe =
  "The first element of a discrete exponential table must be the multiplicative identity.";

PowerListToField[eTable_List] :=
  ReallyUglyGlobalToReturnFieldHead /; PowerListValidQ[eTable]

PowerListValidQ[eTable_] :=
Module[
    {size, characteristic, degree, fTable, xindex, t},

    (* elementary error checking *)

    size = FactorInteger[Length[eTable]+1];
    If[ Length[size] =!= 1,
    Message[GF::etsz]; Return[False]
    ];

    {{characteristic, degree}} = size;
    size = characteristic^degree;

    If[Not[MatrixQ[eTable,((0 <= # < characteristic) && IntegerQ[#])&]],
    Message[GF::etnim,characteristic]; Return[False]
    ];

    If[Length[eTable[[1]]] =!= degree,
    Message[GF::etdeg,size-1,degree]; Return[False]
    ];

    (* A good-but-expensive test would be
       Length[eTable] === Length[Union[eTable]] *)

    unit = Prepend[Table[0,{degree-1}],1];
    Which[
    eTable[[1]] === unit,
        fTable = eTable,
    eTable[[1]] === Reverse[unit],
        fTable = Map[Reverse,eTable],
    True,
        Message[GF::etbfe]; Return[False]
    ];

    xindex = Position[fTable,RotateRight[unit,1]];
    If[ Not[MatchQ[xindex,{{_Integer}}]],
    Message[GF::etxnf]; Return[False]
    ];

    (* Create the field head *)

(* ReallyUglyGlobalToReturnFieldHead is used to pass the field head
   created below back to PowerListToField.
   PowerListValidQ (this routine) is just an elaborate Condition
   for PowerListToField.  Most of the error conditions are reality
   tests of the various parameters we need to compute anyway.*)

    ReallyUglyGlobalToReturnFieldHead =
    GF[
        characteristic,
        CoefficientList[
        t^degree +
        Map[ (t^#)&, Range[0,degree-1] ] .
        Mod[
            -fTable[[1+Mod[degree*(xindex[[1,1]]-1),size-1]]],
            characteristic
        ],
        t
        ]
    ];
    MakeExpTable[ReallyUglyGlobalToReturnFieldHead,fTable];
    True
]

(* ********************** ConvertToExpTable ********************** *)

(* ConvertToExpTable computes the discrete exponential and logarithm
   functions, and sets UsePowerListQ and PowerListExistsQ for the
   field to True. *)

ConvertToExpTable[GF[char_,irred_List]] :=
 (
  MakeExpTable[
    GF[char,irred],
    PowerList[GF[char,irred]]
  ]
 ) /;  Not[PowerListExistsQ[GF[char,irred]]]

FieldInd[0] = -Infinity

FieldInd[GF[char_,irred_List][0]] = -Infinity

FieldExp[GF[char_,irred_List], -Infinity] = 0

(* MakeExpTable creates the discrete exponential (FieldExp) and
   discrete logarithm (FieldInd) functions for the field.  Using
   these to implement multiplication and division should be much
   more efficient than using the irreducible polynomial.  In order
   to avoid making a special case of 0, we declare the logarithm
   of 0 to be -Infinity and the exponential of -Infinity to be 0.
   Leave MakeExpTable as an internal function.*)

MakeExpTable[GF[char_,irred_List],eTable_List] :=
Block[{},
    PowerList[GF[char,irred]] = eTable; (* memo-ize this *)
    FieldExp[GF[char,irred], n_Integer] :=
      GF[char,irred][
        PowerList[GF[char,irred]][[
      (Mod[n,FieldSize[GF[char,irred]]-1]+1)
    ]]
      ];
    Scan[
       (FieldInd[GF[char,irred][#[[2]]]] = #[[1]])&,
       Transpose[
         {Range[0, Length[PowerList[GF[char,irred]]]-1],
      PowerList[GF[char,irred]]}
       ]
    ];
    PowerListExistsQ[GF[char,irred]] = True;
    UsePowerListQ[GF[char,irred]] = True;
]

(* ******************* The Discrete Logarithm ******************** *)

(* We find a primitive element by taking successive polynomials
and checking that no power smaller than t-1 gives 1, where t is the size
of our field. 2/00 DANL (modified primitiveQ to functional form by JMN, 2/00) *)

primitiveQ[poly_, defpoly_, prime_, deg_, facs_] :=
    Catch[
      Quiet[Scan[
        (pm = Algebra`PolynomialPowerMod`PolynomialPowerMod[
          poly, #, {defpoly, prime}];
        If[pm === 1 || Head[pm] === Algebra`PolynomialPowerMod`PolynomialPowerMod,
           Throw[False, "primitiveQ" (* throw tag *)]]) &,
        facs
      ]];True,
      "primitiveQ" (* catch tag *)
    ]

getprim[x_, defpoly_, prime_, deg_] := Module[
    {j, facs, pow=prime^deg, xpowers, poly, start},
    facs = (pow-1)/Map[First,FactorInteger[pow-1]];
    xpowers = Table[x^j, {j,deg-1,0,-1}];
    start = If[deg==1, 1, prime];
    For [j=start, j<pow, j++,
        poly = IntegerDigits[j, prime, deg] . xpowers;
        If [primitiveQ[poly, defpoly, prime, deg, facs],
            Return[poly]];
        ];
  Return[$Failed]
    ]

findprim[gen_, chr_Integer] := Module[{deg, var},
    var = Variables[gen][[1]];
    deg = Exponent[gen,var];
    getprim[var, gen, chr, deg]
    ]

(* A primitive element of a field is one whose powers include all
   elements of the field except 0. PowerList finds a
   primitive element, then returns a list with the identity first,
   the primitive element next, and successive powers of the primitive
   element following. The elements of the list are only the data part
   of the field elements, without all the field information in the
   head. *)

PowerList[ FieldTag:GF[_,_List] ] :=
Module[{pe},
    pe = PolynomialToElement[
    FieldTag,
    findprim[
        FieldIrreducible[FieldTag],
        Characteristic[FieldTag]
    ]
    ];
    NestList[(#*pe)&,FieldTag[{1}],FieldSize[FieldTag]-2] /.
        FieldTag[a_] -> a
]

(* **************** Element Encoding and Decoding **************** *)

(* Encode a field element into an integer.
  (Compare to ToChacterCode.) *)

ToIntegerVector[GF[char_,irred_List]] :=
ToIntegerVector[GF[char,irred]] =
    char^Range[0,ExtensionDegree[GF[char,irred]]-1]

ToElementCode[0] = 0;

ToElementCode[1] = 1;

ToElementCode[GF[char_,irred_List][data_]] :=
    (data . ToIntegerVector[GF[char,irred]])

(* Decode an integer into a field element. (Compare to FromCharacterCode.) *)

FromElementCode[GF[char_,irred_List], 0] = 0;

FromElementCode[GF[char_,irred_List], i_Integer?Positive] :=
    GF[char,irred][Reverse[IntegerDigits[i,char]]] /;
    (i < FieldSize[GF[char,irred]])

(* Successor imposes a cyclic ordering on the elements of the field
   so that given one element it can find the next in the ordering.
   This makes it possible to try something with each element in
   succession. *)

Successor[GF[char_,irred_List][data_]] :=
    FromElementCode[
    GF[char,irred],
    ToElementCode[GF[char,irred][data]]+1
    ]

(* Extract a polynomial from a data object. *)

VectorToPolynomial[ vector_List, s_ ] :=
    (vector . (s^Range[0,Length[vector]-1]))

ElementToPolynomial[ GF[char_, irred_List], s_Symbol ] :=
    VectorToPolynomial[irred,s]

ElementToPolynomial[ GF[char_, irred_List][data_], s_Symbol ] :=
    VectorToPolynomial[ data, s ]

(* Convert a polynomial to a field element. *)
(* Assume the polynomial's degree < ExtensionDegree. *)

PolynomialToElement[ GF[char_Integer,irred_List], poly_Integer ] :=
    GF[char,irred][{poly}]

PolynomialToElement[ GF[char_Integer,irred_List], poly_ ] :=
    PolynomialToElement[
    GF[char,irred],
    poly,
    First[Variables[poly]]
    ] /;
    (Length[Variables[poly]] === 1)

PolynomialToElement[ GF[char_,irred_List], poly_, s_Symbol ] :=
    GF[char,irred][
    Drop[CoefficientList[s^(Length[irred]-1)+poly,s],-1]
    ]

(* The following definition patches D so that it doesn't get
   confused by the element-list in the compound-function GF
   structure.  *)

Unprotect[D];

D[e_,v_] :=
Module[{u = Unique[GF], pm, ip, s},
    (D[e /. GF[pm_Integer, ip_List][s_] -> u[pm, ip, s],v]) /.
    (u[pm_Integer, ip_List, s_List] -> (GF[pm, ip][s]))
] /; !FreeQ[e,GF]

Protect[D];

(* ***************************** End ***************************** *)

End[] (* `Private` context *)

EndPackage[]   (* FiniteFields` *)
