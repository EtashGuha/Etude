(* ::Package:: *)

(*:Mathematica Version: 3.0 *)

(*:Package Version: 1.2 *)

(*:Copyright: Copyright 1992-2007,  Wolfram Research, Inc. *)

(*:Name: PrimalityProving` *)

(*:Context: PrimalityProving` *)

(* :Title:  PrimalityProving *)

(* :Author: Ilan Vardi *)

(*:Keywords:
    Primality Proving, Certificate of Primality, Pratt's Certificate,
    Elliptic Curves, Complex Multiplication, Goldwasser-Kilian,
    Atkin-Morain
*)

(*:Summary:
This package implements primality proving.  The functions provided in this
package not only prove primality, but also generate a certificate of
primality (i.e., a relatively short set of data that makes the primality
proof easy).
*)

(*:History:
Original package by Ilan Vardi, Wolfram Research, Inc., 1992.
Support added for cases where PrimeQ gives True for composites, added 2nd
    arg to HilbertPolynomial, removed export of symbol x,
    ECM, Wolfram Research, Inc., 1993.
Fixed precision problem in QuadraticRepresentation4, improved messages,
    verified that PrimeQCertificate[10^199 + 153] works, ECM, 1997.
*)


(* :Source: Francois Morain, ``Implementation of the Atkin-Goldwasser-Kilian
            Primality Testing Algorithm,'' INRIA Research Report #911,
            October 1988.

            A.O.L. Atkin and F. Morain, ``Elliptic Curves and
            Primality Proving,'' Mathematics of Computation,
        Vol. 61, No. 203, July 1993, pp. 26-68.

            D.A. Cox, Primes of the Form x^2 + n y^2, Wiley, 1989.

            Stan Wagon, Mathematica in Action, W.H. Freeman, 1991.

            D. Bressoud, Factorization and Primality Testing,
            Springer-Verlag, 1989.
*)

(* :Warning:

    ProvablePrimeQ[p] has to wait until PrimeQCertificate has
    finished to return its answer, i.e., until a certificate has been
    generated. Since PrimeQCertificate caches its value, running
    PrimeQCertificate[p] after ProvablePrimeQ[p] will not take
    any extra time. (There have been problems with caching of
    certificates.)

    In the case of small numbers, a false True from PrimeQ will be
    detected by generating a counterexample.  PrimeQCertificate
    returns the counterexample and ProvablePrimeQ returns False.
    In the case of large numbers, a false True from PrimeQ can be
    suspected by the behavior of the algorithm.  If a counterexample is
    found, PrimeQCertificate returns the counterexample and
    ProvablePrimeQ returns False.  If a counterexample is not found,
    both PrimeQCertificate and ProvablePrimeQ will return $Failed.

    Thus ProvablePrimeQ can be used to find counterexamples to PrimeQ.
*)


(* :Limitation:  The Elliptic Curves method will not work for very
                 small primes.  The option SmallPrime (indicating when
         the Elliptic Curves method takes over from Pratt's method)
         has a minimum value of 1000.

                 I. Vardi:  This method will return an answer in
                 reasonable time for inputs of up to 200 digits, i.e.,
                 this takes about six hours on a 1990 DEC workstation.
*)

(* :Discussion:

      This package does primality proving. In fact, the package will
      produce a certificate of primality, i.e., a short set of data
      that is a witness to the primality of a prime. This certificate
      can easily be checked in the sense that the theory involved in
      implementing a check is much easier than the theory that produced
      the certificate. This is analogous to the certificate of
      compositeness given by exhibiting a nontrivial factorization.
      The package also includes a certificate of compositeness, though
      it is more subtle than just giving a factor.

*********************************************************************
               Pratt's certificate of Primality
*********************************************************************

      For small prime p <= SmallPrime use Pratt's certificate of primality.

      Pratt's certificate gives a proof that a is a generator
      (primitive root) of the multiplicative group (mod p) which,
      along with the fact that a has order p-1, proves that p is a
      prime.  You need to factor p-1, which is feasible for p with
      20 digits or less.

      The certificate shows that PowerMod[a, p-1, p] = 1 and
      PowerMod[ a, (p-1)/q, p] != 1 for all primes q dividing p-1
      (=> a is a primitive root). These primes q are the children of
      {p,a} in the tree and are proved prime recursively. You have
      to check that all prime factors of p - 1 are included.

      This part of the program uses some suggestions of Ferrell Wheeler
      as in the PrimitiveRoot[]  program in NumberTheoryFunctions.m
      package.

      See Stan Wagon's book for details on this test.

        Examples:

        PrimeQCertificate[2]
        2
        (This can be considered a short-hand for {2, 1, {}}.)

        PrimeQCertificate[3]
                {3, 2, {2}}
        Certificate is not a vector so 3 is prime.  The certificate
            is in the form {p, a, {q}}.
                    This implies: p = 3, a = 2, q = 2.
                Check: PowerMod[2, 2, 3] = 1
                        PowerMod[2, 1, 3] = 2 != 1

    PrimeQCertificate[4]
        {2, 3, 4}
        Certificate is a vector so 4 is composite.  The certificate
            is in the form {a, p-1, p}.
            This implies: p = 4, p-1 = 3, a = 2
        Check: PowerMod[2, 3, 4] = 0 != 1

        PrimeQCertificate[5]
                {5, 2, {2}}
        Certificate is not a vector so 5 is prime.  The certificate
                        is in the form {p, a, {q}}.
                    This implies: p = 5, a = 2, q = 2.
                Check: PowerMod[2, 4, 5] = 1
                        PowerMod[2, 2, 5] = 4 != 1.

    PrimeQCertificate[6]
        {2, 5, 6}
        Certificate is a vector so 6 is composite.  The certificate
                        is in the form {a, p-1, p}.
        This implies: p = 6, p-1 = 5, a = 2
        Check: PowerMod[2, 5, 6] = 2 != 1

    PrimeQCertificate[7]
        {7, 3, {2, {3, 2, {2}}}}
        Certificate is not a vector so 7 is prime.  The certificate
                        is in the form {p, a, {q1, {q2, a, {q3}}}.
        This implies: p = 7, a = 3, q1 = 2, q2 = 3.
        Check: PowerMod[3, 6, 7] = 1
            PowerMod[3, 3, 7] = 6 != 1
            PowerMod[3, 2, 7] = 2 != 1

        PrimeQCertificate[17]
                {17, 3, {2}}
        Certificate is not a vector so 17 is prime.  The certificate
                        is in the form {p, a, {q}}.
                This implies: p = 17, a = 3, q = 2.
                Check: PowerMod[3, 16, 17] = 1
                        PowerMod[3, 8, 17] = 16 != 1.

    PrimeQCertificate[341]
        {32, 2, 341}
        Certificate is a vector so 341 is composite.  The certificate
                        is in the form {a, 2, p}.
        This implies that: a = 32, p = 341
        Check: PowerMod[32, 2, 341] = 1


***************************************************************
    Atkin-Goldwasser-Kilian-Morain certificate of primality
***************************************************************

      For large primes p > SmallPrime use the Atkin-Morain test.

      The Atkin-Morain test returns a recursive certificate for the
      prime p. This consists of a list of:

 (i)  A point (CertificatePoint) on an Elliptic Curve: PointEC[x,y,g2,g3,p],
      in other words, a solution of the equation

          y^2 = x^3 + g2 x + g3  (mod p)

      for some numbers g2 and g3.

 (ii) A prime q with q > (p^(1/4) + 1)^2, such that for some
      other number k (CertificateK), and m = k q (CertificateM)
      such that k is not equal to 1 and such that

           m PointEC[x,y,g2,g3,p]

      is the identity on the curve, but

           k PointEC[x,y,g2,g3,p]

       is not the identity. This guarantees primality of p
       by a theorem of Goldwasser and Kilian (see Lenstra,
       Elliptic Curve algorithms, World Congress, 1986). So one needs
       only to show that q is also prime in order to show that p is
       prime, and q is shown prime using the same method.

 (iii) Each q has its recursive certificate following it. So if
       the smallest q is known to be prime, all the numbers are
       certified prime up the chain.

   Thus to check the certificate at level q, do the following:

      CertificateK CertificatePoint /.
        Select[PrimeQCertificate[p, opts], #[[1]] == q &] [[2]]

      CertificateM CertificatePoint /.
        Select[PrimeQCertificate[p, opts], #[[1]] == q &] [[2]]

      CertificateNextPrime == CertificateM /CertificateK &&
        CertificateNextPrime > ((N[q]^(1/4) + 1)^2) /.
           Select[PrimeQCertificate[p, opts], #[[1]] == q &] [[2]]

   The first should not be the identity, the second should be the
   identity, and the third should be True.

   This algorithm is described in Francois Morain's INRIA technical
   report #911.  A more recent version is in the Atkin-Morain paper.


***************************************************************
              Certificate of Compositeness
***************************************************************

   The certificate of compositeness shows that either
   PowerMod[a, p - 1, p] != 1, or else a != 1 or -1 and
   PowerMod[a, 2, p] == 1, which proves that p is not a prime.

   Applying PowerMod to PrimeQCertificate[p] proves p composite
   by checking this condition for the value of a given by the
   certificate.

***************************************************************
              Implementation Notes
***************************************************************

   The ZeroPrecisionError flag is shared among routines and is not declared
    in any variable list.
*)


BeginPackage["PrimalityProving`"]


If[Not@ValueQ[ProvablePrimeQ::usage],ProvablePrimeQ::usage =
"ProvablePrimeQ[n] determines whether or not n is prime, based on \
a certificate of primality or compositeness."];

If[Not@ValueQ[PrimeQCertificate::usage],PrimeQCertificate::usage =
"PrimeQCertificate[n] gives a certificate that n is prime or that n is \
composite."];

If[Not@ValueQ[PrimeQCertificateCheck::usage],PrimeQCertificateCheck::usage = "PrimeQCertificateCheck[cert, n] \
checks whether the certificate cert is a correct certificate for the \
primality or compositeness of n. It returns True if the certificate \
is correct and False otherwise."];

If[Not@ValueQ[ModularInvariantj::usage],ModularInvariantj::usage = "ModularInvariantj[z] gives the j invariant \
of the complex number z, Im[z] > 0."];

If[Not@ValueQ[HilbertPolynomial::usage],HilbertPolynomial::usage = "HilbertPolynomial[d, x] gives the irreducible \
polynomial in x of ModularInvariantj[(-1 + Sqrt[d])/2], where d < 0 is a  \
square-free integer of the form 4 n + 1. The splitting field is \
the Hilbert class field (maximal unramified Abelian extension) of \
Q(Sqrt[d])."];

If[Not@ValueQ[PointEC::usage],PointEC::usage = "PointEC[a, b, g2, g3, p] specifies a point {x, y} = {a, b} \
on the elliptic curve y^2 = x^3 + g2 x + g3 (mod p)."];

If[Not@ValueQ[PointECQ::usage],PointECQ::usage = "PointECQ[PointEC[a, b, g2, g3, p]]  yields True if \
point {x, y} = {a, b} satisfies the elliptic curve \
y^2 = x^3 + g2 x + g3 (mod p), and yields False otherwise."];

If[Not@ValueQ[CertificatePrime::usage],CertificatePrime::usage =
"CertificatePrime is used to certify primality."];

If[Not@ValueQ[CertificatePoint::usage],CertificatePoint::usage =
"CertificatePoint is used to certify primality."];

If[Not@ValueQ[CertificateK::usage],CertificateK::usage =
"CertificateK is used to certify primality."];

If[Not@ValueQ[CertificateM::usage],CertificateM::usage =
"CertificateM is used to certify primality."];

If[Not@ValueQ[CertificateNextPrime::usage],CertificateNextPrime::usage =
"CertificateNextPrime is used to certify primality."];

If[Not@ValueQ[CertificateDiscriminant::usage],CertificateDiscriminant::usage =
"CertificateDiscriminant is used to certify primality."];

Unprotect[ProvablePrimeQ, PrimeQCertificateCheck, (* PrimeQCertificate, *)
        ModularInvariantj, HilbertPolynomial, fact,
        CertificatePrime, CertificatePoint, CertificateK, CertificateM,
        CertificateNextPrime, CertificateDiscriminant, PointEC, PointECQ]

Begin["`Private`"]

Options[PrimeQCertificate] =
{"SmallPrime" -> 10^50, "Certificate" -> False, "PrimeQMessages" ->False,
 "PollardPTest" -> Automatic, "PollardRhoTest" -> Automatic,
 "TrialDivisionLimit" -> Automatic}

Options[ProvablePrimeQ] = Options[PrimeQCertificate]

Attributes[ProvablePrimeQ] = {Listable}


ProvablePrimeQ::certfail =
"Unable to find a certificate for primality or compositeness."
ProvablePrimeQ::badcert =
"Certificate is neither a valid certificate for primality nor a valid \
certificate for compositeness."
ProvablePrimeQ::nonopt =
"Options expected (instead of `1`) beyond position 1 in \
ProvablePrimeQ[`2`, `3`].  An option must be a rule or a list of rules."


ProvablePrimeQ[x_, anything__] :=
 Null /; (If[!OptionQ[Last[{anything}]],
        Message[ProvablePrimeQ::nonopt, Last[{anything}],
            x, anything]
        ];
      False)

(* All argument other than integers should give False. *)
ProvablePrimeQ[x_, opts___?OptionQ] := False /; !IntegerQ[x]

ProvablePrimeQ[x_, opts___?OptionQ] :=
    ProvablePrimeQ[-x, opts] /; IntegerQ[x] && x < 0

ProvablePrimeQ[p_Integer, opts___?OptionQ]:=
   Block[{certflag, certresult, primeq},
      certflag = "Certificate" /. {opts} /. Options[ProvablePrimeQ];
      certresult = PrimeQCertificate[p, opts];
      If[certresult === $Failed && Abs[p] > 1,
            Message[ProvablePrimeQ::certfail];
        Return[$Failed]
      ];
      If[VectorQ[certresult] (* NOTE: && Length[certresult] == 3 *),
         (* certificate IS vector-valued: this means the
        number is a composite. *)
         If[((Rest[certresult] == {p-1, p} &&
          Apply[PowerMod, certresult] != 1) ||
         (Rest[certresult] == {2, p} &&
          First[certresult] != 1 && First[certresult] != p-1 &&
          Apply[PowerMod, certresult] == 1)
        ),
             primeq = False,
        (* NOTE: the following case should never happen. *)
        Message[ProvablePrimeQ::badcert];
             Return[$Failed]
             ],
         (* certificate is NOT vector-valued: this means the
        certificate is (a) a list containing lists, (b) the integer
        2, or (c) $Failed (with Abs[p] <= 1).  Cases (a) and (b)
        mean that p is prime.  Case (c) means that p is not prime. *)
         primeq = Not[certresult === $Failed]
      ];

          If[!TrueQ[certflag],
        primeq,
        {primeq, certresult}
      ]
   ]


PrimeQCertificate::zero =
"The number 0 is considered neither prime nor composite, so no \
certificate for the primality or compositeness exists."
PrimeQCertificate::one =
"The number 1 is considered neither prime nor composite, so no \
certificate for the primality or compositeness exists."
PrimeQCertificate::false =
"Warning: PrimeQCertificate has detected a counterexample to PrimeQ: ``."
PrimeQCertificate::smprm =
"Warning: SmallPrime is set to ``."
PrimeQCertificate::trdiv =
"Warning: TrialDivisionLimit is set to default ``."
PrimeQCertificate::pollp =
"Warning: PollardPTest is set to default ``."
PrimeQCertificate::pollr =
"Warning: PollardRhoTest is set to default ``."
PrimeQCertificate::qrprec =
"Failure of elliptic curves certificate for p = ``. Unable to find quadratic \
representation 4*p = u^2 + ``*v^2 due to insufficient precision. Try \
increasing $MaxPrecision."
(* NOTE: that SqrtMod is in the context NumberTheory`NumberTheoryFunctions` *)
PrimeQCertificate::qrsqrtmod =
"Failure of elliptic curves certificate for p = ``. Unable to find quadratic \
representation 4*p = u^2 + ``*v^2 due to failure of SqrtMod[``, p]."
PrimeQCertificate::gcdmod =
"Unable to find the GCD of `` and `` mod ``."
PrimeQCertificate::rootmod =
"Unable to find a root of `` mod ``."

$SmallPrimeMin = 1000

PrimeQCertificate[0, ___?OptionQ] :=
    (Message[PrimeQCertificate::zero]; $Failed);

PrimeQCertificate[1 | -1, ___?OptionQ] :=
    (Message[PrimeQCertificate::one]; $Failed);

PrimeQCertificate[ppp_Integer, opts___?OptionQ] :=
     PrimeQCertificate[ppp, opts]=
 Block[{primeq, smallprime},

  Which[
    (* cases 0, 1, and -1 are handled by separate rules *)

    Abs[ppp] == 2,
    2, (* Unlike the certificates for other primes which are
        all in the form of a list,
         the certificate for 2 and -2 is simply 2. *)

    Not[primeq = PrimeQ[ppp]],

    (* ======================== ppp Composite ========================= *)
    compositeCertificate[Abs[ppp]],
    (* ============================================================== *)

    primeq,

    (* =========================== ppp Prime =========================== *)
         smallprime = "SmallPrime" /. {opts} /. Options[PrimeQCertificate];

       (* restrict SmallPrime to >= $SmallPrimeMin *)
         If[!NumberQ[smallprime],
            Message[PrimeQCertificate::smprm, $SmallPrimeMin];
            smallprime = $SmallPrimeMin
        ];
         If[smallprime < $SmallPrimeMin,
        (* PrimeQCertificate::smprm message is annoying in the case of
            smallprime < $SmallPrimeMin, so commenting it out *)
            (* Message[PrimeQCertificate::smprm, $SmallPrimeMin]; *)
            smallprime = $SmallPrimeMin
        ];

    If[ppp <= smallprime,

            (* ===================  Pratt's certificate ================== *)

       Module[{p = Abs[ppp], q, a = 2, flist},
         q = p-1;
         flist = First /@ FactorInteger[p-1];
            While[JacobiSymbol[a, p] == 1 ||
                Scan[If[PowerMod[a, q/#, p] == 1,
                           Return[True]
                ]&, flist],
                    a++];
             If[PowerMod[a, q, p] == 1,
        (* p is a prime *)
             {p, a, PrimeQCertificate[#, opts] & /@ flist},
        (* p is a composite *)
        compositeCertificate[p]
             ]
       ],

       (* ================ Elliptic curves certificate ============  *)

       Module[{p = Abs[ppp],
           messages = "PrimeQMessages" /. {opts} /.
           Options[PrimeQCertificate],
           di = 1, d = -7, m, q},

         If[messages,
             Print[StringForm[
             "Prime candidate p = `` .", InputForm[p] ]]
         ];

         Module[{trialdivisionlimit, pollardptest,
             pollardrhotest, trialvector, triallen,
             tableq = True,
             qr, t, s, qplustr, qminustr, tdmod, qplus, qminus},

              {trialdivisionlimit, pollardptest, pollardrhotest} =
            {"TrialDivisionLimit", "PollardPTest", "PollardRhoTest"} /.
                {opts} /. Options[PrimeQCertificate];

          (* First build a vector of p + 1 mod small primes in order to
               facilitate trial division. *)


          If[trialdivisionlimit =!= Automatic &&
               (!NumberQ[trialdivisionlimit] ||
             Head[trialdivisionlimit] =!= Integer ||
                 trialdivisionlimit <= 0),
           trialdivisionlimit = Which[p > 10^150, 20000,
                   p > 10^60, 10000, True, 5000];
               Message[PrimeQCertificate::trdiv, trialdivisionlimit]
          ];
          If[trialdivisionlimit === Automatic,
           trialdivisionlimit = Which[p > 10^150, 20000,
                                   p > 10^60, 10000, True, 5000]
          ];
          trialdivisionlimit = Min[trialdivisionlimit,
                                 2 Sqrt[N[p]]/Log[N[p]]];
          trialvector = Mod[p+1, Prime[Range[trialdivisionlimit]]];
          triallen = Length[trialvector];
          If[messages,
             Print[StringForm["Trial division vector has length ``.",
             triallen]]
          ];

          (* Determine factoring tests to be used in LargeFactor. *)

          If[!MemberQ[{Automatic, True, False}, pollardptest],
            pollardptest = (p < 10^150);
            Message[PrimeQCertificate::pollp, pollardptest]
          ];
          If[pollardptest === Automatic, pollardptest = p < 10^150];

          If[!MemberQ[{Automatic, True, False}, pollardrhotest],
            pollardrhotest = (p < 10^60);
            Message[PrimeQCertificate::pollr, pollardrhotest]
          ];
          If[pollardrhotest === Automatic, pollardrhotest = p < 10^60];


                  (* ======================================================= *)
                  (* WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW *)
              While[True,

                 While[JacobiSymbol[d,p] != 1,
                      If[di < $classTableLength,
                 di++; d = 1 - 4 ClassTable[[di]],
                         If[di == $classTableLength || tableq,
                              d = -7; di++,
                tableq = False; {d, di} += {-4, 1}
                 ]
              ];
                      If[di > $classTableLength,
                         While[MemberQ[ClassTable, (1 - d)/4] ||
                               !SquareFreeQ[d],
               {d, di} += {-4, 1}
                 ]
              ]
                    ];  (* end While JacobiSymbol[d,p] != 1 *)

                  If[(di > $classTableLength) &&
                     (NumberTheory`ClassNumber[d] > 30),
                       {d, di} += {-4, 1};
                       While[MemberQ[ClassTable, (1 - d)/4] &&
                             !SquareFreeQ[d],
                 {d, di} += {-4, 1}
               ];
               Continue[]
            ]; (* end If (di > $classTableLength) &&
                 (ClassNumber[d] > 30) *)

                    If[messages,
               Print[StringForm[
            "Class table index = ``, discriminant = `` .",
                InputForm[di], InputForm[d] ]]
            ];

            ZeroPrecisionError = False;
                    qr = QuadraticRepresentation4[d, p];

            (* ====================================================== *)
            If[ZeroPrecisionError,
               (* Error in computing qr, return $Failed. *)
               Return[$Failed] ];
            If[qr === $Failed,
              (* Return either composite certificate or $Failed. *)
              With[{cert = compositeCertificate[p]},
                     If[Rest[cert] == {p-1, p} &&
               Apply[PowerMod, cert] != 1,
               Message[PrimeQCertificate::false,
                 StringForm["PowerMod[``, ``, ``] != 1",
                cert[[1]], cert[[2]], cert[[3]] ] ];
               Return[cert],
               If[Rest[cert] == {2, p} &&
                  First[cert] != 1 && First[cert] != p-1 &&
                  Apply[PowerMod, cert] == 1,
                  Message[PrimeQCertificate::false,
                StringForm["PowerMod[``, ``, ``] == 1",
                    cert[[1]], cert[[2]], cert[[3]] ] ];
                      Return[cert],
                  Return[$Failed]
                   ]
                    ]
              ]
            ];
                   If[qr === False,
               (* Update di and d and go to top of "While True" loop. *)
                        If[di < $classTableLength,
              di++; d = 1 - 4 ClassTable[[di]],
                       If[di == $classTableLength || tableq,
                             d = -7; di++; tableq = False,
                             {d, di} += {-4, 1}
              ]
            ];
                      If[di > $classTableLength,
                        While[MemberQ[ClassTable, (1 - d)/4] ||
                                 !SquareFreeQ[d],
                 {d, di} += {-4, 1}
               ]
            ];
                        Continue[]
            ];  (* end If qr === False *)
            (* ====================================================== *)

            (* At this point, qr is a valid vector of length 2. *)
            If[messages,
             Print[StringForm[
            "4*p = u^2 + ``*v^2.  Trying trial division.",
            InputForm[-d] ]]
            ];

                    t = Abs[2 qr[[1]]]; s = Abs[2 qr[[2]]];
                    qplustr = p + 1 + t;
                    qminustr = p + 1 - t;

            (* Trial division of p + 1 + t, p + 1 - t, is done in
            parallel by comparing t mod small primes with the stored
            values of p + 1 mod small primes in trialvector. This
            allows one to do both trial divisions at once and by
            modular operations with a number of size sqrt[p].
            See Atkin-Morain for details. *)

              tdmod = Mod[t, Prime[Range[triallen]]];
              Scan[While[Mod[qminustr, #] == 0,
                qminustr = Quotient[qminustr, #]
                 ]&,
                      Prime[Flatten[Position[tdmod - trialvector, 0]]]
              ];
              Scan[While[Mod[qplustr, #] == 0,
                qplustr = Quotient[qplustr, #]
                 ]&,
                      Prime[Flatten[Position[
                      MapIndexed[If[#1 == 0 || {#1} == Prime[#2],0,1]&,
                                       tdmod + trialvector], 0]]]
                    ];

              qplus = LargeFactor[qplustr, p+1+t, p, pollardptest,
            pollardrhotest];
              If[IntegerQ[qplus],
            m = p + 1 + t; q = qplus,
              qminus = LargeFactor[qminustr, p+1-t, p, pollardptest,
                pollardrhotest];
                    If[IntegerQ[qminus],
               m = p + 1 - t; q = qminus,
                           If[di < $classTableLength,
                  di++; d = 1 - 4 ClassTable[[di]],
                              If[di == $classTableLength || tableq,
                             d = -7; di++; tableq = False,
                              {d, di} += {-4, 1}
                  ]
                   ];
                           If[di > $classTableLength,
                              While[MemberQ[ClassTable, (1 - d)/4] ||
                    !SquareFreeQ[d],
                               {d, di} += {-4, 1}
                      ]
                   ];
                           Continue[]
                    ] (* end If IntegerQ[qminus] *)
                    ]; (* end If IntegerQ[qplus] *)
                    Break[]

                  ]; (* end FIRST While True *)

                  (* MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM *)
                  (* ======================================================== *)

         ]; (* end FIRST interior Module *)

             If[messages,
               Print[StringForm[
            "Trial division successful." ]]
         ];
         If[ZeroPrecisionError, Return[$Failed] ];

         Module[{hp, x, exp, hr,
             temp, ell, g2, g3, xec, sqrtmod, yec2, point,
             moverq, p1, p2, c, curve},
                  hp = If[di > $hilbertPolynomialTableLength ||
                          di > $classTableLength,
                         HilbertPolynomial[d, x],
                         (HilbertPolynomialTable[[di]])[x]
          ];
                 SeedRandom[101];                   (* Same random factor  *)
          (* NOTE: Added the following message, because PolyPowerMod
            (sometimes called by RootMod for polynomials of
                order 3 or more) can be lengthy. *)
          exp = Exponent[hp, x];
          If[messages,
                 Print[StringForm[
             "Finding root of a polynomial of order `` mod p.",
                InputForm[exp] ]]
          ];
                  hr = RootMod[hp, p, x];
          If[hr === $Failed,
             (* error in computing root mod p *)
                     Message[PrimeQCertificate::rootmod, hp, p];
             Return[$Failed]];
          temp = PowerMod[1728 - hr,-1,p];
                  ell = Mod[hr temp, p];
                  g2 = Mod[3 ell,p];
          g3= Mod[2 ell,p];
                  xec = 2;

                  While[True,

            temp = PowerMod[xec,3,p];
                    yec2 = Mod[temp + g2 xec + g3, p];
                    While[JacobiSymbol[yec2,p] == -1,
                      xec++;
              temp = PowerMod[xec,3,p];
                      yec2 = Mod[temp + g2 xec + g3, p]
                    ];
            sqrtmod = SqrtMod[yec2,p];
            point = PointEC[xec, sqrtmod, g2, g3, p];
                    moverq = m/q;

            (* Point will be expressed in projective coordinates so as
               to avoid doing GCD operations. The X coordinate will be
               written as a ratio x:z. The identity element occurs when
               z = 0. See Bressoud's book, p. 213.
                  One has to check that no nontrivial factors of p
               occur, i.e., that 1 < GCD[z, p] < p does not occur. *)

                    p1 = mult[{xec, 1}, moverq, g2, g3, p];
                    If[p1[[2]] == 0, xec++; Continue[]];
                    p2 = multcheckprime[p1, q, g2, g3, p];

                    If[GCD[p2[[1]], p] > 1,
               With[{cert = compositeCertificate[p]},
                         Message[PrimeQCertificate::false,
                  StringForm["`` is a factor of ``",
              GCD[p2[[1]], p], p]];
                         Return[cert]
               ]
            ];

            (* Check that second to last operation did not give a
               nontrivial factor of p. If it does then PrimeQ[p] is
                   wrong.  p2[[1]] : p2[[2]] gives X coordinate of q * pt
            which should be the identity (p2[[2]] == 0),
            p2[[3]] : p2[[4]] gives the X coordinate of
            (q + 1) * pt.  If a nontrivial divisor of
               p occured in the computation of q * pt, then it would
               remain in the the Z coordinate of (q+1) * pt
            (i.e., GCD[p2[[4]], p]>1). *)

                    If[p2[[3]] == 0,
              Break[]
            ];
                    c = 2;
            While[JacobiSymbol[c,p] == 1, c++];
                    g2 = Mod[g2 c^2, p];
                    g3 = Mod[g3 c^3, p];
                    curve = 2;
                    xec = 2

                  ]; (* end SECOND While True *)

          cert = PrimeQCertificate[q, opts];
          If[cert =!= $Failed,
             cert = Prepend[cert,
                  {CertificatePrime -> p,
                    CertificatePoint -> point,
                    CertificateK -> m / q,
                    CertificateM -> m,
                    CertificateNextPrime -> q,
                    CertificateDiscriminant -> d}]
          ];
          cert
         ] (* end SECOND interior Module *)

       ] (* end main Module *)

    ] (* end If ppp <= smallprime *)
    (* ============================================================== *)

  ] (* end Which ppp==0, Abs[ppp]==1, Abs[ppp]==2, ppp Composite, ppp Prime *)

 ] (* End of PrimeQCertificate[ppp] *)



compositeCertificate[p_] :=
  Module[{a = 1, q = (p - 1)/2,  s = -1,  exp = 2, r},
        While[EvenQ[q], q /= 2];
        While[Mod[s + 1, p] == 0,
                    a++;
                    If[PowerMod[a, p - 1, p] != 1,
                       exp = p - 1;
                       s = a;
                       Break[],
                       r = PowerMod[a, q, p];
                       If[r != 1,
                          While[r != 1,
                                s = r;
                                r = PowerMod[s, 2, p]
                          ]  (* end While *)
                       ]  (* end If *)
                    ]  (* end If *)
        ];  (* end While *)
        {s, exp, p}
  ]




(*  *************************************************************
               Checking the certificate
    ************************************************************

*)

(*
    If p is composite check that PowerMod[a, p - 1, p] != 1 or
    that ( PowerMod[ a, 2, p] == 1 for a != -1, 1 ), for some a.
*)

PrimeQCertificateCheck::badnum =
"The number `` does not correspond to the number `` in the certificate."

PrimeQCertificateCheck::vectorc =
"The number `` is prime, but a valid prime certificate is not vector-valued."

PrimeQCertificateCheck::notcompc =
"The number `` is composite, but the certificate is not a valid composite \
certificate."

PrimeQCertificateCheck::false =
"Warning: PrimeQCertificateCheck has detected a counterexample to PrimeQ; \
`1` is a factor of `2`."

(* 0 is considered neither prime nor composite, so by definition, no
    certificate for the primality or compositeness of 0 can be correct *)

PrimeQCertificateCheck[_, 0] :=
    (Message[PrimeQCertificateCheck::zero];
     False)

(* 1 is considered neither prime nor composite, so by definition, no
    certificate for the primality or compositeness of 1 can be correct *)

PrimeQCertificateCheck[_, x_] :=
    (Message[PrimeQCertificateCheck::one];
     False) /; Abs[x] == 1

(* 2 is the only prime that does not have a certificate in the form
     of a list. *)

PrimeQCertificateCheck[2, x_] := True /; Abs[x] == 2

PrimeQCertificateCheck[_, x_] := False /; Abs[x] == 2

PrimeQCertificateCheck::zero =
"The number 0 is considered neither prime nor composite, so no \
certificate for the primality or compositeness of 0 is correct."

PrimeQCertificateCheck::one =
"The number 1 is considered neither prime nor composite, so no \
certificate for the primality or compositeness of 1 is correct."

PrimeQCertificateCheck[list_List, ppp_Integer] :=
  Block[{p = Abs[ppp], am, pratt, certnumber},

    (* Check whether list is a valid composite certificate of p.
       Make a point of NOT using PrimeQ at this stage. *)

    If[VectorQ[list] (* NOTE: && Length[list] == 3 *),
       If[p != Last[list],
      Message[PrimeQCertificateCheck::badnum, p, Last[list]];
      Return[False],
          Return[
       (
         (* Either a^(p-1) != 1 (mod p)  *)
            (Rest[list] == {p-1, p} && Apply[PowerMod, list] != 1) ||
         (* or there is an element a != 1 or -1 such that a^2 = 1 (mod p). *)
            (Rest[list] == {2, p} && (First[list] != 1 && First[list] != p-1) &&
                Apply[PowerMod, list] == 1)
       )
          ]
       ]
    ];

    (* If the certificate is not a composite certificate, but p is
    composite, then return False: the certificate is wrong. *)

    If[!PrimeQ[p],
       Message[PrimeQCertificateCheck::notcompc, p];
       Return[False]
    ];

    (* Check whether list is a valid prime certificate of p. *)

    PrimeQCounterexample = Null;

    (* Checking for primality:
         pratt is the Pratt portion of the test (i.e., the last 3 entries in
     the certificate) and am the rest of the certificate (i.e., the
     Atkin-Morain portion). *)

    (* Check whether list is a valid prime certificate of p. *)

     If[VectorQ[list],
    (* A valid prime certificate cannot be vector-valued. *)
       Message[PrimeQCertificateCheck::vectorc, p];
    Return[False]
     ];

     pratt = Take[list, -3];  am = Drop[list, -3];

     (* Make sure that the certificate corresponds to the number. *)

     If[    (am === {} && (certnumber = First[pratt];  p != certnumber)) ||
        (Length[am] >= 1 &&
         (certnumber = (CertificatePrime /. First[am]); p != certnumber)),
    (* The number must agree with the number in the certificate! *)
       Message[PrimeQCertificateCheck::badnum, p, certnumber];
    Return[False]
     ];



     (* Testing the Pratt certificate is equivalent to recomputing it.

         A nontrivial failed inversion during elliptic curve addition
         means that a counterexample to PrimeQ has been found, i.e.,
         PrimeQ[p] gives True, but a nontrivial factor of p is found.
         This test is implemented using a Throw/Catch construction. *)

     Catch[answer =
               (     (* only Pratt's test *)
        p == First[pratt] &&
                pratt == PrimeQCertificate[First[pratt], "SmallPrime"-> p + 1]
        ) ||
                (     (* Elliptic curve test *)
         (((CertificateNextPrime /. Last[am]) == First[pratt])) &&
          (pratt == PrimeQCertificate[First[pratt],
            "SmallPrime" ->First[pratt+1]]) &&
                  (p == First[pratt] || (And @@
              Map[PointECQ[(CertificatePoint /. #)] &&
                   (!((CertificateK CertificatePoint /. #) === IdentityEC)) &&
                    (CertificateM CertificatePoint /. #) === IdentityEC &&
                    (CertificateM /. #) > (CertificateNextPrime /. #) >
                    ((N[CertificatePrime /. #]^(1/4) + 1)^2) &&
                    (CertificateK /. #)  (CertificateNextPrime /. #) ==
            (CertificateM /. #)&,
                   am])) &&
                  (Table[CertificateNextPrime, {Length[am] -1}] /.
            Drop[am, -1]) ==
                  (Table[CertificatePrime, {Length[am] -1}] /. Rest[am])
        )
     ];

     If[PrimeQCounterexample =!= Null,
    Message[PrimeQCertificateCheck::false, PrimeQCounterexample, p];
    False,
           answer
     ]
   ]

PrimeQCertificateCheck[list_, p_] := False /; Head[list] =!= List




   (* Computing the j invariant using the Dedekind eta function.
      See Atkin--Morain, Section 3.5. *)

ModularEta[z_, prec_Integer, range_Integer]:=
   Block[{q = Exp[2 N[Pi, prec] I N[z, prec]], i},
       Exp[N[Pi, prec] I N[z, prec]/12] *
       (1 +
       Plus @@ Table[(-1)^i (q^(i (3 i -1)/2) + q^(i (3 i + 1)/2)),
                     {i, range}])
   ]

ModularInvariantj[z_, prec_Integer, range_Integer] :=
  Block[{f2 =  2^12 (ModularEta[2 z, prec, range] /
              ModularEta[z, prec, range])^24},
      (f2 + 16)^3/f2
  ]


(* Computing the Hilbert Polynomial. The specifications for
   the precision in computing the j-invariant and the number
   of terms to use in the series for eta(z) are given in
   Atkin-Morain, Section 7.1.

   Note that p. 43 of Atkin-Morain gives HilbertPolynomial[-23, x] as
    12771880859375 - 5151296875 x + 3491750 x^2  + x^3
*)


HilbertPolynomial[d_Integer?Negative, y_] :=
  Block[{hprec, hrange, s, classlist = ClassList[d], x, poly},
      hprec = Ceiling[10 + 2 Sqrt[N[-d]] *
                            (Plus @@ N[1/(First /@ classlist)])];
      hrange = 3;
      poly = Expand[Fold[#1 x + #2&, 1,
           Rest[Reverse[Round[N[CoefficientList[Expand[Apply[Times,
           x - Map[ModularInvariantj[
               (- #[[2]] + Sqrt[N[d, hprec]]) / (2 #[[1]]) ,
                    hprec, Round[hrange Sqrt[N[#[[1]]]]]]&,
           classlist]]], {x}], hprec]]]]]];
      poly //. {x->y}
  ] /; Mod[d-1, 4] == 0


ClassList[d_Integer  /; d < 0 ] :=
       Block[{a,b,c,list},
           a = 1; list = {};
           While[a <= N[Sqrt[-d/3]],
                 b = 1 - a;
                 While[b <= a,
                       c= (b^2 - d)/(4 a);
                       If[Mod[c,1] == 0 && c >= a && GCD[a,b,c] == 1 &&
                            Not[a == c && b < 0],
                          AppendTo[list,{a,b,c}]
                         ];
                        b++
                       ];
                   a++
                  ];
             Return[list]
            ]

(*
   Find a root of the monic reducible polynomial f[x] (mod p).
   This doesn't need Berlekamp's algorithm since Class Field Theory
   guarantees that the Hilbert Polynomial splits completely (mod p).
   However this seems to be unacceptable for larger degrees, e.g., 11.
*)

(* (According to Ilan) PolynomialMod does the wrong thing,
     so use PolyMod instead. *)

PolyMod[f_, p_]:=
  (
  PolynomialMod[Expand[f], p]
  )



RootMod[f_, p_Integer, x_]:= Mod[-Coefficient[f,x,0], p]  /; Exponent[f,x] == 1

RootMod[f_, p_Integer, x_]:=
  Block[{cl,b,c},
    cl = Map[Mod[#, p] &, CoefficientList[f,x]];
    b = cl[[2]]; c = cl[[1]];
    Mod[(-b + SqrtMod[b b - 4 c, p]) PowerMod[2, -1,p],p]
  ]   /; Exponent[f,x] == 2

(* NOTE:  This RootMod (for f of order 3 or more) could be a real time sink.
    There is a "While[True,]" that could be quite time consuming
    and PolyPowerMod[s, (p-1)/2, {fp, p}, x] involves a Fold over
    a list having length Length[IntegerDigits[(p-1)/2, 2]].
*)
RootMod[f_, p_Integer, x_] :=
  Block[{exp = Exponent[f, x], fp, i, s, t, ppm, q},
    fp = PolyMod[f,p];
    While[True,
        s = Plus @@ Table[Random[Integer,{0,p}] x^i, {i,0,exp-2}] +
            x^(exp - 1);
        t = GCDMod[s, fp, p, x];
    If[t === $Failed,
       Return[$Failed]];
        If[0 < Exponent[t,x] < exp,
       Break[],
       ppm = PolyPowerMod[s, (p-1)/2, {fp, p}, x];
           t =  GCDMod[fp, ppm - 1, p, x];
       If[t === $Failed,
              Return[$Failed]];
           If[0 < Exponent[t,x] < exp,
          Break[]
       ]
        ]
    ];
    q = PolynomialQuotient[fp, t, x];
    q = PolyMod[q, p];
    t = If[Exponent[q, x] < Exponent[t, x], q, t];
    RootMod[t, p, x]
  ]       /; Exponent[f,x] > 2


(* You need a different GCD for polynomials (mod p). E.g.,
   GCD[x^3 - 1, x-2] = 1, but PolynomialRemainder[x^3 -1, x - 2, x] = 7.
   In other words, using the built-in GCD would not allow you to find
   the root 2 of x^3 - 1 == 0 (mod 7).      *)


GCDMod[f_, g_, p_Integer, x_] :=
    GCDMod[g, f, p, x] /; Exponent[f,x] < Exponent[g,x]

GCDMod[f_, g_, p_Integer, x_] := f  /;  Exponent[g,x] === -Infinity

GCDMod[f_, g_, p_Integer, x_] :=
   Block[{fp, gp, monic, pm, q, r},
     fp = PolyMod[f, p];
     gp = PolyMod[g, p];
         monic = Coefficient[gp, x, Exponent[gp,x]];
         If[monic != 1,
        pm = PowerMod[monic, -1, p];
        If[Head[pm] === PowerMod,
          Message[PrimeQCertificate::gcdmod, f, g, p];
         Return[$Failed]
        ];
            gp = PolyMod[pm gp, p]
         ];
     q = PolynomialQuotient[fp, gp, x];
         q = PolyMod[q, p];
     r = PolynomialRemainder[fp, gp, x];
         r = PolyMod[r, p];
         If[Exponent[r, x] <= 0,
            If[r == 0, gp, 1],
            GCDMod[gp, r, p, x]
         ]
   ]  /; Exponent[g,x] >= 0


PolynomialRemainderMod[f_, g_, x_, p_Integer] :=
  With[{f1 = PolyMod[f, p], g1 = PolyMod[g, p]},
   PolyMod[PolynomialRemainder[f1, g1, x], p]
  ]

(* NOTE: PolyPowerMod is called by only RootMod *)
PolyPowerMod[f_, n_Integer, {g_,p_Integer}, x_] :=
 Block[{prm = PolynomialRemainderMod[f, g, x, p], id = IntegerDigits[n, 2]},
  Fold[PolynomialRemainderMod[#1^2 #2, g, x, p] &,
     1,
     If[# == 0, 1, prm]& /@ id
  ]
 ] /; n > 0



(*
   LargeFactor tries to find a nontrivial large prime factor of n.
   (This part of the program is the bottleneck since large numbers require
   nontrivial factoring routines). The problem in this implementation
   is the fact that trial division is so slow. This part of the
   program should be speeded up a lot when the complete version of
   FactorInteger is implemented in the kernel.

   The larger the number, the simpler the factoring routine used.
   For very large numbers, one uses only trial division.
*)


LargeFactor[qq_, m_Integer,n_Integer, pptest_, prtest_] :=
         Block[{q = qq, pp, pr, nn = (N[n]^(1/4)+1)^2},
                If[PrimeQ[q] &&  nn < q < m, Return[q]];
                If[pptest,
                   If[messages,
              Print["Trying Pollard p-1 method of finding a large prime factor."]];
                   pp = GCD[q, PowerMod[2, LCM @@ Range[3000], q]-1];
                   q = Quotient[q, pp];
                   If[PrimeQ[q] &&  nn < q < m,
                      Return[q]
           ]
        ];
                If[prtest,
                   If[messages,
              Print["Trying Pollard rho method of finding a large prime factor."]];
                   If[!(N[10^5 nn] < q < m), Return[$Failed]];
                   pr = PollardRhoPrimeQ[q, 2, Min[2000, Round[N[n]^(1/4)]]];
                   If[pr != q, q = Quotient[q, pr]];
                   If[PrimeQ[q] &&  nn < q < m,
                      Return[q]
           ]
        ]
         ]


(* Finding a solution of  4p = x^2 + |d| y^2  if it exists.  *)
(* This is Section 8.4.2 of the Atkin-Morain paper. *)

QuadraticRepresentation4[d_Integer, p_Integer] :=
      Block[{b, prec, u, v, w, x, y, lr, inv},
            b = SqrtMod[d, p];
        If[!NumberQ[b],
           Message[PrimeQCertificate::qrsqrtmod, p, -d, d];
           Return[$Failed]];
            If[EvenQ[b], b = p - b];
            prec = Max[Round[N[4/3 Log[10,p]]], 50];
        v = HighPoint[ N[ (-b +  Sqrt[d])/(2 p), prec ]];
        While[v === $Failed && prec <= $MaxPrecision,
        (* Increase the precision by 20 % each iteration. *)
        prec = Ceiling[prec 1.2];
        v = HighPoint[ N[ (-b +  Sqrt[d])/(2 p), prec ]]
        ];
        If[v === $Failed,
        Message[PrimeQCertificate::qrprec, p, -d];
        ZeroPrecisionError = True;
                Return[$Failed]];
            x = v[[2]]; y = v[[1]];
            If[(x p - b y/2)^2 - y^2 d/4 == p,
               {x p - b y/2, y/2},
               False]
      ]    /;     d < 0 && Mod[d,4] == 1

HighPoint[z_Complex] :=
   Block[{g = IdentityMatrix[2], abs, t=z, round, den},
         While[True,
               round = Round[Re[t]];
               If[round != 0,
                  t = t - round;
                  g = {{1,-round},{0,1}}. g,
              den = Re[t]^2 + Im[t]^2;
          (* If precision is sufficiently lowered so that
            den==0 then the loop cannot continue.
            One solution may be to restart the loop with a higher
            precision, but here we just return with an error
            message.  The effect of low precision causing den==1
            does not seem to be a problem. *)
          If[den==0,
             Return[$Failed]
          ];
                  abs = 1/den;
                  If[abs > 1,
                     t =  - Conjugate[t] abs;
                     g = {{0,-1},{1,0}} . g,
                     Break[]
                    ]
                  ]
               ];
          g[[2]]
        ]

(* This version of PollardRho doesn't do intermediate GCD's,,
   since you want to find all the small factors of a large
   number. *)

PollardRhoPrimeQ[n_, c_, max_]:=
  Block[{x1 = 2, x2 = 2^2 + c, i = 0, j, gcd,
         range = 1, prod = 1, factor = 1},
         While[i <= max,
               Do[x2 = Mod[x2^2 + c, n];
                  prod = Mod[prod (x1 - x2), n], {range}];
               i += range;
               x1 = x2; range *= 2;
               x2 = Nest[Mod[#^2 + c, n]&, x2, range]];
           factor = GCD[prod, n];
        If[factor > 1, factor, n]]


(* ClassTable stores the fundamental discriminants d > -24000 with
   class number 20 or less. They are ordered by class number and by
   size. Since d is negative of the form 4k + 1, d is stored as
   (1 - d)/4 in order to save space. This large list of 1186 numbers
   is required so that weaker factoring algorithms can be used on
   large numbers (so that the algorithms searches more numbers).
   If this is too large, you can reduce it by taking the first 489
   elements (class number <= 12).
*)

ClassTable =
  {2, 3, 5, 11, 17, 41, 4, 9, 13, 23, 29, 31, 47, 59, 67, 101, 107, 6, 8, 15,
   21, 27, 35, 53, 71, 77, 83, 95, 125, 137, 161, 221, 227, 10, 14, 39, 49,
   51, 55, 65, 73, 81, 89, 109, 121, 139, 149, 157, 167, 179, 181, 191, 199,
   239, 251, 257, 307, 311, 347, 353, 359, 377, 389, 12, 20, 26, 32, 33, 45,
   57, 87, 111, 131, 143, 155, 171, 173, 185, 197, 237, 263, 281, 431, 437,
   467, 551, 587, 671, 22, 62, 85, 103, 113, 129, 177, 193, 209, 211, 265,
   275, 287, 301, 305, 317, 329, 337, 341, 391, 401, 461, 479, 491, 557, 571,
   611, 629, 641, 697, 731, 809, 857, 881, 941, 18, 38, 56, 63, 116, 117,
   122, 147, 203, 207, 215, 291, 293, 371, 381, 407, 447, 497, 503, 521, 545,
   563, 617, 677, 755, 767, 797, 977, 1151, 1277, 1481, 24, 28, 46, 74, 75,
   93, 99, 145, 146, 163, 229, 235, 245, 247, 249, 261, 283, 289, 299, 325,
   335, 361, 409, 413, 415, 433, 443, 449, 451, 485, 487, 499, 509, 515, 517,
   535, 541, 577, 581, 599, 605, 613, 647, 653, 667, 679, 689, 707, 737, 749,
   751, 811, 829, 839, 851, 877, 899, 947, 971, 991, 1031, 1049, 1067, 1081,
   1097, 1187, 1211, 1217, 1271, 1367, 1397, 1427, 1487, 1577, 50, 92, 105,
   123, 141, 206, 272, 297, 323, 356, 395, 501, 701, 791, 815, 827, 887, 911,
   1007, 1061, 1091, 1121, 1181, 1247, 1361, 1511, 1607, 1691, 1721, 1931,
   2141, 2201, 2267, 2657, 30, 36, 40, 76, 80, 104, 153, 159, 175, 195, 201,
   213, 231, 279, 411, 427, 445, 455, 459, 473, 481, 591, 661, 725, 773, 785,
   787, 823, 909, 917, 921, 953, 965, 1021, 1057, 1109, 1145, 1157, 1229,
   1283, 1291, 1379, 1403, 1417, 1451, 1529, 1565, 1601, 1667, 1781, 1841,
   1847, 1859, 1871, 1907, 2057, 2237, 2327, 2537, 2621, 3461, 42, 68, 165,
   242, 321, 326, 327, 365, 383, 425, 507, 567, 635, 683, 713, 743, 801, 837,
   875, 935, 983, 1013, 1295, 1421, 1541, 1637, 1757, 1877, 1901, 1967, 2111,
   2321, 2351, 2411, 2447, 2747, 3251, 3317, 3527, 3671, 3917, 58, 64, 82,
   136, 164, 172, 183, 189, 253, 267, 309, 314, 339, 343, 355, 373, 379, 387,
   441, 452, 489, 523, 539, 559, 589, 597, 623, 639, 649, 657, 659, 681, 699,
   717, 739, 757, 761, 779, 847, 863, 865, 905, 931, 937, 959, 989, 1009,
   1037, 1039, 1073, 1103, 1117, 1133, 1139, 1147, 1175, 1189, 1193, 1199,
   1207, 1227, 1237, 1241, 1259, 1289, 1315, 1325, 1327, 1343, 1349, 1381,
   1399, 1439, 1441, 1453, 1459, 1547, 1559, 1567, 1571, 1621, 1651, 1661,
   1679, 1697, 1711, 1733, 1739, 1741, 1747, 1777, 1823, 1889, 1921, 1973,
   2033, 2039, 2081, 2087, 2099, 2197, 2207, 2251, 2285, 2339, 2381, 2417,
   2461, 2501, 2651, 2677, 2687, 2699, 2729, 2789, 2837, 2927, 2951, 3077,
   3161, 3611, 3791, 3821, 4001, 4451, (* last classno = 12 *)
   48, 66, 152, 158, 182, 255, 363, 375,
   417, 477, 533, 536, 593, 665, 741, 771, 923, 1001, 1127, 1161, 1337, 1355,
   1445, 1655, 1811, 1991, 2387, 2435, 2867, 2897, 2957, 2981, 3011, 3587,
   3947, 4241, 5141, 54, 72, 98, 112, 128, 134, 176, 202, 225, 303, 351, 382,
   463, 471, 531, 537, 543, 584, 607, 627, 643, 695, 733, 807, 895, 927, 929,
   967, 1047, 1079, 1111, 1165, 1201, 1257, 1273, 1313, 1317, 1431, 1493,
   1625, 1631, 1745, 1767, 1775, 1787, 1979, 2009, 2047, 2153, 2225, 2279,
   2309, 2357, 2531, 2579, 2591, 2603, 2807, 3037, 3167, 3197, 3257, 3359,
   3371, 3401, 3551, 4217, 4547, 4637, 4661, 5057, 5387, 5771, 60, 110, 188,
   243, 315, 332, 357, 392, 405, 561, 662, 675, 711, 833, 893, 951, 1025,
   1055, 1251, 1307, 1331, 1391, 1457, 1497, 1517, 1523, 1553, 1643, 1805,
   1865, 1887, 2117, 2177, 2195, 2261, 2477, 2561, 2567, 2615, 2663, 2681,
   2771, 2993, 3041, 3191, 3287, 3491, 3581, 3707, 3713, 3797, 3911, 3977,
   4151, 4211, 4367, 4481, 4511, 4631, 4847, 4967, 5177, 5501, 100, 102, 118,
   140, 166, 200, 224, 226, 236, 254, 256, 262, 285, 290, 345, 399, 496, 505,
   524, 549, 553, 595, 685, 703, 766, 793, 841, 845, 859, 861, 883, 901, 949,
   955, 985, 1011, 1043, 1045, 1063, 1095, 1107, 1129, 1171, 1172, 1225,
   1235, 1243, 1279, 1297, 1299, 1341, 1351, 1471, 1477, 1499, 1525, 1531,
   1549, 1579, 1589, 1599, 1649, 1687, 1693, 1731, 1751, 1759, 1763, 1799,
   1829, 1837, 1849, 1873, 1895, 1927, 1937, 1939, 1949, 1955, 1961, 1981,
   1999, 2011, 2021, 2071, 2075, 2129, 2137, 2159, 2161, 2171, 2179, 2209,
   2215, 2305, 2371, 2377, 2399, 2441, 2459, 2467, 2489, 2549, 2551, 2557,
   2597, 2641, 2647, 2659, 2701, 2711, 2741, 2767, 2777, 2795, 2801, 2881,
   2891, 2909, 2929, 2999, 3007, 3065, 3097, 3131, 3149, 3187, 3209, 3215,
   3281, 3299, 3331, 3341, 3377, 3449, 3455, 3457, 3539, 3593, 3601, 3637,
   3677, 3691, 3749, 3767, 3847, 3851, 3887, 3929, 4007, 4049, 4087, 4133,
   4139, 4181, 4307, 4331, 4337, 4357, 4379, 4601, 4679, 4721, 4727, 4787,
   4799, 4987, 4997, 5039, 5099, 5351, 5429, 5459, 5561, 5711, 5849, 5897,
   96, 248, 273, 393, 416, 446, 633, 831, 987, 1085, 1112, 1137, 1163, 1371,
   1551, 1595, 1613, 1707, 1727, 1971, 2135, 2183, 2471, 2813, 2861, 3227,
   3407, 3521, 3695, 3737, 4175, 4457, 4577, 4991, 5267, 5891, 84, 130, 132,
   170, 284, 302, 346, 422, 423, 482, 512, 513, 542, 573, 579, 687, 715, 759,
   777, 886, 913, 1075, 1077, 1205, 1221, 1329, 1385, 1475, 1557, 1583, 1597,
   1685, 1709, 1831, 1835, 1893, 1929, 1943, 1957, 2051, 2083, 2101, 2127,
   2147, 2281, 2303, 2391, 2407, 2421, 2587, 2631, 2855, 2911, 2921, 2963,
   3017, 3047, 3059, 3071, 3163, 3181, 3203, 3307, 3329, 3347, 3437, 3487,
   3497, 3541, 3557, 3629, 3667, 3779, 3811, 4031, 4043, 4097, 4157, 4259,
   4283, 4351, 4409, 4571, 4757, 4781, 4913, 5009, 5207, 5261, 5417, 5477,
   5567, 5611, 5627, 5737, 5837, 5867, 5921, 5981, 78, 90, 230, 266, 386,
   458, 525, 585, 615, 836, 866, 867, 902, 1005, 1035, 1082, 1265, 1287,
   1382, 1415, 1701, 2105, 2231, 2243, 2405, 2723, 2825, 3773, 3833, 4091,
   4187, 4253, 4325, 4385, 4421, 4877, 5297, 5303, 5321, 5801, 114, 154, 316,
   374, 424, 435, 472, 562, 602, 621, 622, 651, 686, 746, 747, 752, 775, 776,
   783, 789, 805, 842, 849, 879, 892, 914, 939, 995, 1003, 1004, 1089, 1093,
   1099, 1101, 1135, 1183, 1191, 1214, 1223, 1255, 1333, 1433, 1489, 1513,
   1535, 1585, 1617, 1689, 1783, 1791, 1793, 1867, 1903, 1909, 1913, 1917,
   1963, 1985, 2017, 2063, 2089, 2123, 2189, 2249, 2263, 2287, 2299, 2333,
   2335, 2341, 2361, 2393, 2423, 2439, 2449, 2495, 2507, 2521, 2539, 2543,
   2573, 2575, 2627, 2629, 2705, 2839, 2845, 2849, 2857, 2885, 2915, 2939,
   2971, 2987, 2989, 3005, 3035, 3079, 3083, 3089, 3091, 3117, 3125, 3147,
   3151, 3221, 3233, 3239, 3241, 3289, 3311, 3389, 3413, 3451, 3577, 3583,
   3617, 3623, 3665, 3689, 3809, 3839, 3901, 3923, 3941, 3971, 3989, 4037,
   4099, 4109, 4121, 4127, 4171, 4177, 4229, 4231, 4267, 4297, 4391, 4411,
   4441, 4477, 4517, 4541, 4549, 4589, 4591, 4771, 4861, 4889, 4981, 5021,
   5051, 5147, 5171, 5189, 5221, 5273, 5309, 5327, 5347, 5399, 5431, 5441,
   5471, 5597, 5617, 5639, 5651, 5681, 5861, 5987};

$classTableLength = Length[ClassTable];

(* Table of Hilbert polynomials corresponding to fundamental
   discriminants of class number <= 3. *)

HilbertPolynomialTable =
 Map[(Function[{x}, y]/.y->#)&,
  {3375 + x, 32768 + x, 884736 + x, 884736000 + x, 147197952000 + x,
   262537412640768000 + x, -121287375 + 191025*x + x^2,
   -134217728000 + 117964800*x + x^2, 6262062317568 + 5541101568*x + x^2,
   -3845689020776448 + 10359073013760*x + x^2,
   130231327260672000 + 427864611225600*x + x^2,
   148809594175488000000 + 1354146840576000*x + x^2,
   -3845689020776448000000 + 4545336381788160000*x + x^2,
   11946621170462723407872000 + 823177419449425920000*x + x^2,
   531429662672621376897024000000 + 19683091854079488000000*x + x^2,
   -108844203402491055833088000000 + 2452811389229331391979520000*x + x^2,
   155041756222618916546936832000000 + 15611455512523783919812608000*x + x^2,
   12771880859375 - 5151296875*x + 3491750*x^2 + x^3,
   1566028350940383 - 58682638134*x + 39491307*x^2 + x^3,
   374643194001883136 - 140811576541184*x + 30197678080*x^2 + x^3,
   549755813888000000000 - 41490055168000000*x + 2691907584000*x^2 + x^3,
   337618789203968000000000 - 6764523159552000000*x + 129783279616000*x^2 +
    x^3, 67408489017571610198016 - 53041786755137667072*x +
    12183160834031616*x^2 + x^3,
   5310823021408898698117644288 + 277390576406111100862464*x +
    65873587288630099968*x^2 + x^3,
   201371843156955365376000000000 + 90839236535446929408000000*x +
    89611323386832801792000*x^2 + x^3,
   8987619631060626702336000000000 - 5083646425734146162688000000*x +
    805016812009981390848000*x^2 + x^3,
   56176242840389398230218488594563072 + 368729929041040103875232661504*x +
    6647404730173793386463232*x^2 + x^3,
   15443600047689011948024601807415148544 -
    121567791009880876719538528321536*x + 364395404104624239018246144*x^2 +
    x^3, 4671133182399954782798673154437441310949376 -
    6063717825494266394722392560011051008*x +
    3005101108071026200706725969920*x^2 + x^3,
   83303937570678403968635240448000000000 -
    139712328431787827943469744128000000*x +
    81297395539631654721637478400000*x^2 + x^3,
   308052554652302847380880841299197952000000000 -
    6300378505047247876499651797450752000000*x +
    39545575162726134099492467011584000*x^2 + x^3,
   167990285381627318187575520800123387904000000000 -
    151960111125245282033875619529124478976000000*x +
    34903934341011819039224295011933392896000*x^2 + x^3,
   149161274746524841328545894969274007552000000000 +
    39181594208014819617565811575376314368000000*x +
    123072080721198402394477590506838687744000*x^2 + x^3}
 ]

$hilbertPolynomialTableLength = Length[HilbertPolynomialTable];



(* Elliptic Curves *)


PointEC /: PointECQ[PointEC[pt___]] := PointECQ @@ PointEC[pt]


PointECQ[x_,y_,g2_,g3_,p_] := Mod[y^2 - x^3 - g2 x - g3, p] === 0

PointEC /:  PointEC[x1_,y1_,g2_,g3_,p_] == PointEC[x2_,y2_,g2_,g3_,p_]:=
              (Mod[x1,p] == Mod[x2,p]) && (Mod[y1,p] == Mod[y2,p])


(*
    Addition on the Elliptic curve y^2 = x^3 + g2 x + g3 (mod p).
    This is the part of the algorithm that gets used the most.
*)

PointEC /:  PointEC[x_Integer,y_Integer,g2_,g3_,p_] + IdentityEC :=
                      PointEC[x,y,g2,g3,p]

PointEC /: PointEC[x1_,y1_,g2_,g3_,p_] + PointEC[x2_,y2_,g2_,g3_,p_] :=
  Block[{slope, IdentityECQ = False, gcd},
       If[Mod[x1 - x2, p] == 0,
          If[Mod[y1 + y2, p] == 0,
             IdentityECQ = True,
             slope = Mod[(3 x1^2 + g2) PowerMod[2 y1, -1, p], p]
      ],
          slope = Mod[(y2 - y1) PowerMod[x2 - x1, -1, p], p]
       ];
       (* A failed inversion implies a counterexample to PrimeQ. *)

       If[!IntegerQ[slope],
             gcd = GCD[y1, p];
             If[1 < gcd < p,
                Message[PrimeQCertificate::false];
                PrimeQCounterexample = gcd;
                Throw[True]
         ];
             gcd = GCD[x2 - x1, p];
             If[1 < gcd < p,
                Message[PrimeQCertificate::false];
                PrimeQCounterexample = gcd;
                Throw[True]
         ]
       ];

       (*   Otherwise continue with addition. *)

       If[SameQ[IdentityECQ, True],
             IdentityEC,
             x3 =  Mod[slope^2 - x1 - x2, p];
             y3 =  Mod[slope (x1-x3) - y1, p];
             PointEC[x3,y3,g2,g3,p]
       ]
  ]

 (* Multiply a point on the curve by an integer by repeated doubling,
    exactly like PowerMod[a,b,c].  *)

IdentityEC /:  n_Integer * IdentityEC := IdentityEC

PointEC /:  0 * PointEC[x_,y_,g2_,g3_,p_]:= IdentityEC

PointEC /:  1 * PointEC[x_,y_,g2_,g3_,p_]:= PointEC[x,y,g2,g3,p]

PointEC /: 2 * PointEC[x_, y_, g2_, g3_, p_]:=
                PointEC[x, y, g2, g3, p] + PointEC[x, y, g2, g3, p]

PointEC /: n_ * PointEC[x_, y_, g2_, g3_, p_]:=
                 (-n) * PointEC[x, -y, g2, g3, p] /; n < 0

PointEC /: n_ * PointEC[x_, y_, g2_, g3_, p_]:=
  Fold[2 #1 + #2&, IdentityEC,
       IntegerDigits[n, 2] PointEC[x, y, g2, g3, p]]  /; 2 < n  <= 2^100

(* For large n use base 16 to speed up the calculation by 25-> 50%.
   If used repeatedly, precompute digits in main algorithm. *)


PointEC /: n_ * PointEC[x_, y_, g2_, g3_, p_]:=
  Fold[16 #1 + #2&, IdentityEC,
       NestList[# + PointEC[x, y, g2, g3, p]&,
                IdentityEC, 15] [[IntegerDigits[n, 16] + 1]]] /; n > 2^100



(* Elliptic curve multiplication in homogeneous form (no GCD's),
   as in Bressoud's book, p. 213. *)

(* X coordinate of 2*P. *)

double[{x_, z_, g2_, g3_, m_}]:=
Block[{x2 = Mod[x^2, m], z2 = Mod[z^2, m], bz3},
       bz3 = g3 Mod[z2 z, m];
       {Mod[(x2 - Mod[g2 z2, m])^2 - 8 x bz3, m],
        Mod[4 z Mod[Mod[x2 x, m] + g2 Mod[x z2, m] + bz3, m], m]}
      ]


(* X coordinate of (2i+1)*P from i*P and (i+1)*P. *)

doubleplusone[{x1_, z1_, xi_, zi_, xi1_, zi1_, g2_, g3_, m_}]:=
Block[{zizi1 = Mod[zi zi1, m], xi1zi = Mod[xi1 zi, m],
       xizi1 = Mod[xi zi1, m]},
      {Mod[z1 Mod[PowerMod[xi xi1 - g2 zizi1, 2, m] -
                  4 g3 Mod[zizi1 (xizi1 + xi1zi), m], m], m],
       Mod[x1 PowerMod[xi1zi - xizi1, 2, m], m]}
     ]


(* X coordinate of {2i*P, (2i+1)*P} from {i*P, (i+1)*P}. *)

even[{x1_, z1_, xn_, zn_, xm_, zm_, g2_, g3_, m_}]:=
Join[{x1, z1}, double[{xn, zn, g2, g3, m}],
     doubleplusone[{x1, z1, xn, zn, xm, zm, g2, g3, m}], {g2, g3, m}]

(* X coordinate of {(2i+1)*P, (2i+2)*P} from {i*P, (i+1)*P} *)

odd[{x1_, z1_, xn_, zn_, xm_, zm_, g2_, g3_, m_}]:=
Join[{x1, z1}, doubleplusone[{x1, z1, xn, zn, xm, zm, g2, g3, m}],
     double[{xm, zm, g2, g3,  m}], {g2, g3, m}]

(* Compute n*P, where P has X coordinate x1:z1. *)

mult[{x1_, z1_}, n_, g2_, g3_,  m_]:=
     Fold[If[#2 == 0, even[#1], odd[#1]]&,
          Join[{x1, z1}, {x1,z1},double[{x1, z1, g2, g3, m}], {g2, g3, m}],
               Rest[IntegerDigits[n, 2]]] [[{3,4}]]

(* This is the same as mult but multiplies together all the
   intermediate zi's in order to see whether a spurious
   factor of p exists. *)

multcheckprime[{x1_, z1_}, n_, g2_, g3_,  m_]:=
  Fold[Join[{Mod[#1[[1]] #1[[5]] #1[[7]], m]},
             If[#2 == 0, even[Rest[#1]], odd[Rest[#1]]]]&,
       Join[{1}, {x1, z1}, {x1,z1}, double[{x1, z1, g2, g3, m}], {g2, g3, m}],
            Rest[IntegerDigits[n, 2]]] [[{1, 4, 5}]]


(* SqrtMod added from NumberTheoryFunctions.m *)

sm[n_, m_] :=

    Module[{sml, res},

        If[ !IntegerQ[n] || !IntegerQ[m] || m <= 0,
            Return[$Failed]
        ];

        sml = Internal`SqrtModList[n, m];

        If[ sml == {},
            Message[PowerMod::root, 2, n, m];
            res = $Failed,

            res = Internal`SqrtMod[n,m];
        ];

        res
    ];


SqrtMod[n_, m_] :=
    With[{res = sm[n,m]},
        res /; FreeQ[res, $Failed | Internal`SqrtMod]
    ]


Protect[ProvablePrimeQ, PrimeQCertificateCheck, (* PrimeQCertificate, *)
        ModularInvariantj, HilbertPolynomial, fact,
        CertificatePrime, CertificatePoint, CertificateK, CertificateM,
        CertificateNextPrime, CertificateDiscriminant, PointEC, PointECQ]


End[]


EndPackage[]
