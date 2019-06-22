(*:Mathematica Version: 8.0 *)

(*:Package Version: 1.2 *)

(*:Name: Statistics`MultinormalDistribution` *)

(*:Context: Statistics`MultinormalDistribution` *)

(*:Title: Statistical Distributions Derived from the Multinormal Distribution *)

(*:Author: E. C. Martin *)

(*:Copyright: Copyright 1993-2010, Wolfram Research, Inc. *)

(* :History:
Original version by ECM (Wolfram Research), October 1993, May 1994.
Added error messages for various functions
	of the distributions MultinormalDistribution and
	MultivariateTDistribution;  added support for singular
	MultinormalDistribution: ECM, April 1997.
Added RandomReal for all distributions, Darren Glosemeyer (Wolfram Research), 
	August 2005.
Extended MultivariateTDistribution to allow for scale matrices and location parameters,
	e.g. MultivariateTDistribution[sigma, nu] and MultivariateTDistribution[mu, sigma, nu],
	Darren Glosemeyer (Wolfram Research) June 2008.
Updated for kernelization of distributions in version 8.0, 
	2010, Darren Glosemeyer (Wolfram Research).
*)

(*:Summary:
This package provides properties and functionals of five standard
probability distributions derived from the multinormal (multivariate Gaussian)
distribution.  The five distributions are the multinormal distribution, the 
Wishart distribution, Hotelling's T-square distribution, the multivariate
Student t distribution, and the distribution of the quadratic form of a
multinormal variate.
*)

(*:Keywords: Continuous distribution, multinormal distribution,
	multivariate normal, multivariate Gaussian, Wishart, Hotelling,
	quadratic form, multivariate student t *)

(*:Requirements: No special system requirements. *)

(*:Warning:
This package extends the definition of several descriptive statistics
functions and the definition of Random.  If the original usage messages
are reloaded, this change will not be reflected in the usage message,
although the extended functionality will remain.
This package also adds rules to Series.
*)

(*:Discussion:
MultinormalDistribution[{mu}, {{sigma^2}}] is NormalDistribution[mu, sigma].
Random variable X ~ WishartDistribution[{{sigma^2}}, m] is the same as
	random variable (Y * sigma^2) where Y ~ ChiSquareDistribution[m].
Random variable X ~ HotellingTSquareDistribution[1, m] is the same as
	random variable Y^2 where Y ~ StudentTDistribution[m].
Random variable X ~ HotellingTSquareDistribution[p, m] is the same as
	random variable (Y * (m p/(m-p+1))) where
	Y ~ FRatioDistribution[p, m-p+1].
Mean[dist], where dist is the distribution of a random vector, yields a
vector of means, where each element is the mean of the corresponding
element in the random vector.  Similarly, Mean[dist], where dist is the
distribution of a random matrix, yields a matrix of means, where each
element is the mean of the corresponding element in the random matrix.
The same goes for Variance, Skewness, Kurtosis, and KurtosisExcess.
*)

(*:Reference:
  Norman L. Johnson and Samuel Kotz, Continuous Multivariate Distributions,
    Wiley, 1970.
  K. V. Mardia, J. T. Kent, and J. M. Bibby, Multivariate Analysis, 
    Academic Press, 1979.
  Y. L. Tong, The Multivariate Normal Distribution, Springer, 1990.
  A. M. Mathai and Serge B. Provost, Quadratic Forms in Random Variables:
    Theory and Applications, Marcel Dekker, Inc., 1992.
  J. Wishart, The generalized product moment distribution in samples from a
    normal multivariate population, Biometrika, v. 20A, 1928.
  W. B. Smith and R. R. Hocking, "Algorithm AS 53: Wishart Variate Generator", 
  	Applied Statistics, 1972, Vol. 21, No. 3. 
  Dagpunar, John. Principles of Random Variate Generation. Oxford University 
  	Press, 1988. 
*)

(*:Limitations:
Singular multinormal distribution (i.e., determinant of covariance matrix is
zero) is not fully supported.

EllipsoidProbability[MultinormalDistribution[], ellipsoid] is supported only
for ellipsoids corresponding to constant-probability contours of nonsingular 
multinormal distributions.

EllipsoidProbability[MultivariateTDistribution[], ellipsoid] is supported only
for ellipsoids corresponding to constant-probability contours of the 
multivariate t distribution.

PDF[QuadraticFormDistribution[], x] and CDF[QuadraticFormDistribution[], x]
must be expanded using Series if a symbolic expression in x is sought.

Quantile is not implemented for
MultinormalDistribution, WishartDistribution, QuadraticFormDistribution, and
MultivariateTDistribution.

EllipsoidQuantile and EllipsoidProbability are not implemented for 
singular multinormal distributions.  CDF is not implemented for singular
multinormal distributions of dimensionality greater than two.
*)


(* matrix generalization of ChiSquareDistribution *)
If[ Not@ValueQ[WishartDistribution::usage],
WishartDistribution::usage =
"WishartDistribution[sigma, m] represents the Wishart distribution with scale \
matrix sigma and degrees of freedom parameter m. For a p x p symmetric \
positive definite random matrix to be distributed WishartDistribution[sigma, m], \
sigma must be a p x p symmetric positive definite matrix, and m must be an \
integer satisfying m >= p+1."]

If[ Not@ValueQ[QuadraticFormDistribution::usage],
QuadraticFormDistribution::usage =
"QuadraticFormDistribution[{a, b, c}, {mu, sigma}] represents the (univariate) \
distribution of the quadratic form z.a.z + b.z + c, where z is distributed \
MultinormalDistribution[mu, sigma]. For the random variable z.a.z + b.z + c \
to be distributed QuadraticFormDistribution[{a, b, c}, {mu, sigma}], a must be \
a p x p symmetric matrix, b must be a p-variate vector, and c must be a scalar."]


If[ Not@ValueQ[EllipsoidProbability::usage],
EllipsoidProbability::usage =
"EllipsoidProbability[distribution, ellipse] gives the integral of the \
probability density function of the specified distribution over the \
specified ellipse centered at the mean of the distribution."]


(* Unprotect MultinormalDistribution.m symbols. *)
Unprotect[MultinormalDistribution, WishartDistribution,
	  HotellingTSquareDistribution, QuadraticFormDistribution,
	  MultivariateTDistribution]

(* Unprotect descriptive statistics symbols. *)
Unprotect[Mean, Variance, StandardDeviation, Skewness, Kurtosis,
         PDF, CDF, ExpectedValue, DistributionParameterQ,
         DistributionDomainQ,DistributionDomain,Covariance,
         Correlation,MultivariateSkewness,MultivariateKurtosis,
         CharacteristicFunction,Quantile,EllipsoidQuantile,
         EllipsoidProbability]

(* ======================================================================= *)
Begin["`Private`"]

(* Gives output similar to old SingularValues command *)
CompactSVD[mat_, opts___] := 
Module[{sv, U, W, V, n},
    (* Compute SVD *)
    sv = SingularValueDecomposition[mat, Min[Dimensions[mat]], opts];
    If[!ListQ[sv], Return[$Failed]];
    {U, W, V} = sv;
    (*extract the diagonal vector*)
    sv = Tr[W, List];
    (* determine the number of positive singular values *)
    n = Length[sv] - Count[sv, _?(#==0&)];
    If [n == 0,
        {U, W, V},
        {Take[U, All, n], Take[sv, n], Take[V, All, n]}
    ]
]

CholDecomp[mat_] := If[Not[MatrixQ[N[mat], NumberQ]] &&
   	Transpose[mat] === mat && FreeQ[N[mat], Complex],
  (* for symmetric matrices containing symbolic values, 
  	 assume symbolic values are real *)
  CholeskyDecomposition[Re[mat]
  	] /. {Conjugate[x_?(FreeQ[#, Complex] &)] :> x} /. {Re[x_] :> x},
  CholeskyDecomposition[mat]];

Delta::usage =
"Delta[num, accgoal, precgoal] uses the number num, the accuracy goal accgoal, \
and the precision goal precgoal to determine the delta to use in  \
Chop[num, delta].  The complete result is a list {delta, accuracyGoalWarningQ, \
precisionGoalWarningQ}, where the final two elements of the list indicate \
whether the Automatic setting of AccuracyGoal or PrecisionGoal was used in \
place of accgoal or precgoal."

Delta[num_, accgoal_, precgoal_] :=
  Module[{acc, prec},
  	   acc = Max[Accuracy[num], $MachinePrecision]-3;
	   prec = Max[Precision[num], $MachinePrecision]-3;
	   Which[accgoal === Automatic,
		 Null,
		 validGoalQ[accgoal],
		 acc = accgoal,
		 True,
		 AccuracyGoalWarning = True];
	   Which[precgoal === Automatic,
		 Null,
		 validGoalQ[precgoal],
		 prec = precgoal,
		 True,
		 PrecisionGoalWarning = True];
	   (* use the largest delta *)
	   {Max[10^(-acc), Abs[num] 10^(-prec)],
		TrueQ[AccuracyGoalWarning], TrueQ[PrecisionGoalWarning]}
  ] 

validGoalQ[goal_] := (goal === Infinity ||
	(NumberQ[goal] && FreeQ[goal, Complex]))



(* routine to find sign of determinant, with proper
   adjustments for machine-precision input in zero-testing
   per Rob Knapp advice.
   Det may be precomputed as second arg if needed elsewhere. *)
(* use of CholeskyDecomposition in borderline cases is an addition 
to the testing suggested by Rob *)

detSign[mat_, idet_:None] :=
    Module[{det = If[idet === None, Det[mat], idet], posdef},
       Which[
         (* case zero *)
           If[MachineNumberQ[det],
             (* if machine prec, use test based on matrix size *)
             (* the inequality test alone can fail to detect positive definiteness
                for high dimensional matrices, so use CholeskyDecomposition as a 
                second check if the inequality is met *)
               Abs[det] < 2 * $MachineEpsilon * matrixnorm[mat]&&
               (Not[posdef=FreeQ[
               	Internal`DeactivateMessages[CholeskyDecomposition[mat]]
               	, 
               	CholeskyDecomposition]]),
             (* otherwise if bignum, just use ZeroQ *)
               PossibleZeroQ[det]
           ]
         , 
         0
         ,
         (* positive definiteness determined by Cholesky *)
         TrueQ[posdef]
         ,
         1
         ,
         (* case non-zero *)
         True
         , 
         Sign[det]
        ]
    ]

matrixnorm[mat_] := Norm[mat, Infinity]


(* assumes a square matrix *)

AboveDiagonalElements[matrix_] :=
    Flatten[MapIndexed[Drop[#1, #2[[1]]]&, matrix]]

ZeroVectorQ[v_] := Apply[And, Map[TrueQ[#==0]&, v]]

ZeroMatrixQ[m_] := Apply[And, Map[TrueQ[#==0]&, Flatten[m]]]

ellipsoidalLocus[mu_, sigma_] := Module[{esystem, acc, prec, delta, r},
    (* (x-mu).Inverse[sigma].(x-mu) == 1 *)
    If[Apply[And, Map[(Head[#] === DirectedInfinity)&, Flatten[sigma] ]],
        Return[Ellipsoid[mu, Table[Infinity, {Length[mu]}]]]
    ];
    If[Head[esystem = Eigensystem[sigma]] === Eigensystem,
        Return[$Failed]
    ];

    acc = Max[Accuracy[esystem], $MachinePrecision]-1;
    prec = Max[Precision[esystem], $MachinePrecision]-1;
    delta = Max[10^(-acc), Max[Abs[First[esystem]]] 10^(-prec)];

    (* radii are square roots of eigenvalues *)
    r = First[esystem];
    esystem = Last[esystem];
    If[NumberQ[delta], r = Chop[r, delta]];
    r = Sqrt[r];

    (* sort semi-axes from largest to smallest *)
    If[OrderedQ[r, Greater] =!= True,
        With[{ind = Ordering[r, Length[r], Greater]},
            r = r[[ind]];
            esystem = esystem[[ind]]
        ]
    ];

    Ellipsoid[mu, r, esystem]
]

(* ======================== Multinormal Distribution ====================== *)


(* Internal function; same as DistributionParameterQ except that check is for
	 Det[sigma] > 0 .
*)
nonsingularMultinormalQ[mu_, sigma_, warnflag_:False] :=
  (
   MatrixQ[sigma] && VectorQ[mu] &&
   Dimensions[sigma] === {Length[mu], Length[mu]} &&
   If[SymmetricMatrixQ[sigma], True,
      If[warnflag, Message[MultinormalDistribution::cmsym]];False] &&
  Module[{nsigma, dets},
   (
    nsigma = If[Precision[sigma] === Infinity, N[sigma], sigma];
    If[Apply[And, Map[NumberQ, Flatten[nsigma] ]],
       Apply[And, Map[TrueQ[# > 0]&, Diagonal[sigma] ]] &&
       (
	    dets = detSign[nsigma];
        If[dets > 0, True,
            If[TrueQ[warnflag]&&dets!=0, 
            	Message[MultinormalDistribution::cmdet]];False]),
       (* If sigma isn't numeric, just assume it is OK. *)
      True]) 
  ]
  )

MultinormalDistribution::cmsym =
"The covariance matrix must be symmetric.";

MultinormalDistribution::cmdet =
"The covariance matrix must have a non-negative determinant.";

(* Internal function; same as DistributionParameterQ except that check is for
         Det[sigma] == 0 and no message is issued for a singular distribution.
*)
singularMultinormalQ[mu_, sigma_] :=
 (
  MatrixQ[sigma] && VectorQ[mu] &&
   Dimensions[sigma] === {Length[mu], Length[mu]} &&
   SymmetricMatrixQ[sigma] &&
  Module[{nsigma},
   (
    nsigma = If[Precision[sigma] === Infinity, N[sigma], sigma];
    If[Apply[And, Map[NumberQ, Flatten[nsigma] ]],
       Apply[And, Map[TrueQ[# > 0]&, Diagonal[sigma] ]] &&
       (detSign[nsigma] === 0),
       (* If sigma isn't numeric, just assume it is OK. *)
      True])
  ] 
 )

(* Internal function; same as DistributionParameterQ except that check is for
         Det[sigma] >= 0 and no message is issued for a singular distribution.
*)
validMultinormalQ[mu_, sigma_] :=
  (
   MatrixQ[sigma] && VectorQ[mu] &&
   Dimensions[sigma] === {Length[mu], Length[mu]} &&
   SymmetricMatrixQ[sigma] &&
  Module[{nsigma},
   (nsigma = If[Precision[sigma] === Infinity, N[sigma], sigma];
    If[Apply[And, Map[NumberQ, Flatten[nsigma] ]],
       Apply[And, Map[TrueQ[# > 0]&, Diagonal[sigma] ]] &&
       TrueQ[detSign[nsigma] >= 0],
       (* If sigma isn't numeric, just assume it is OK. *)
      True])
  ]
  ) 


(* - - - - - - - - -  DistributionDomainQ[MultinormalDistribution[]] - - - - - - - - - - *)
(* NOTE: DistributionDomainQ should be fast so that a large data set, expected to
	follow a particular distribution, can be quickly checked 
	to see whether each point falls in the prescribed domain.
	Unfortunately, it is not possible to ignore the parameters
	of the multinormal (and avoid checking them), because the
	domain depends on whether the multinormal distribution is
	singular or not.

	To make the domain check of a data set as efficient as possible, 
		use DistributionDomainQ[data], rather than Map[DistributionDomainQ, data].
	The latter will check the distribution validity and singularity
	for every single data point, which is very wasteful.
*)
MultinormalDistribution/: DistributionDomainQ[MultinormalDistribution[mu_, sigma_]?DistributionParameterQ,
        list_?MatrixQ] :=
   (
   MatchQ[Dimensions[list], {_, Length[mu]}] && FreeQ[N[list], Complex]
   )

MultinormalDistribution/: DistributionDomainQ[MultinormalDistribution[mu_, sigma_]?DistributionParameterQ,
        x_?VectorQ] :=
   (
   TrueQ[Length[x] == Length[mu] && FreeQ[N[x], Complex]]
   )

MultinormalDistribution/: MultivariateSkewness[MultinormalDistribution[
	mu_, sigma_]?DistributionParameterQ] := 0

MultinormalDistribution/: MultivariateKurtosis[MultinormalDistribution[
	mu_, sigma_]?DistributionParameterQ] :=
	  With[{p = Length[mu]}, p(p + 2)]

(* - - - - -  EllipsoidQuantile[MultinormalDistribution[], ellipsoid] - - - - *)
MultinormalDistribution/: EllipsoidQuantile[MultinormalDistribution[
	mu_?VectorQ, sigma_?MatrixQ]?DistributionParameterQ, q_] :=
  Module[{ellipsoid, quantile},
    (
    ellipsoid
    ) /; (quantile = Quantile[ChiSquareDistribution[Length[mu]], q];
	  ellipsoid = ellipsoidalLocus[mu, quantile sigma];
	  If[ellipsoid === $Failed,
		Message[EllipsoidQuantile::mnormeig, quantile sigma] ];
	  ellipsoid =!= $Failed)
  ] /; QuantileQ[q] 

EllipsoidQuantile::mnormeig =
"Unable to find eigensystem of ``."

(* - - - - -  EllipsoidProbability[MultinormalDistribution[], ellipsoid] - - - - *)
MultinormalDistribution/: EllipsoidProbability[MultinormalDistribution[
	mu_?VectorQ, sigma_?MatrixQ]?DistributionParameterQ, EllipsoidQuantile[mu1_, radii_]] :=
		0 /;radii === Table[0, {Length[mu]}]

MultinormalDistribution/: EllipsoidProbability[MultinormalDistribution[
	mu_?VectorQ, sigma_?MatrixQ]?DistributionParameterQ, EllipsoidQuantile[mu1_, radii_]] :=
		1 /;radii === Table[Infinity, {Length[mu]}]

MultinormalDistribution/: EllipsoidProbability[MultinormalDistribution[
	mu_, sigma_]?DistributionParameterQ,
	Ellipsoid[mu_, radii_?VectorQ, dir___?MatrixQ]] :=
   Module[{p, internaldir, normalized, diagonalizedMatrix, acc, prec, delta,
		 scaled, quantile, diagonal},
     (
	CDF[ ChiSquareDistribution[p], quantile ]
     ) /; (
	   ( p = Length[mu];
	     internaldir = If[{dir}==={}, IdentityMatrix[p], dir];
	     normalized = sigma.Transpose[internaldir];	
	     diagonalizedMatrix = internaldir.normalized;
	     acc = Max[Accuracy[diagonalizedMatrix], $MachinePrecision]-1;
             prec = Max[Precision[diagonalizedMatrix], $MachinePrecision]-1;
             delta = Max[10^(-acc), Abs[Det[diagonalizedMatrix]] 10^(-prec)];
	
	     diagonalizedMatrix = Simplify[Chop[diagonalizedMatrix, delta]];
	     If[MatchQ[diagonalizedMatrix, DiagonalMatrix[Table[_, {p}]] ],
		True,
		Message[EllipsoidProbability::mnormell,
			 mu, radii, internaldir, sigma];
		False] &&	
	     (diagonal = Diagonal[diagonalizedMatrix]/radii^2;
	      (* Consider
		  (dist = MultinormalDistribution[{1, 2, 3},
			{{1/3, 1/4, 1/5}, {1/4, 1/5, 1/6}, {1/5, 1/6, 1/7}}];
		  ellipsoid = EllipsoidQuantile[dist, .5];
		  p = EllipsoidProbability[dist, ellipsoid];
		  Chop[p-.5])
		in determining acc and prec below. *)
              acc = Max[Accuracy[diagonal], $MachinePrecision]-4;
              prec = Max[Precision[diagonal], $MachinePrecision]-4;
              delta = Max[10^(-acc), Apply[Times, Abs[diagonal]] 10^(-prec)];	

	      Apply[And, Map[(Chop[#, delta]==0)&,
		 (diagonal - RotateRight[diagonal, 1]) ]])    ) &&
	   ( scaled = Transpose[internaldir].DiagonalMatrix[radii^2];
	     quantile = Scan[If[ !TrueQ[#[[1]]==0], Return[#[[2]]/#[[1]]] ]&,
	  	       Transpose[{Flatten[normalized], Flatten[scaled]}] ];
	     quantile =!= Null  )
	  )	
   ]

EllipsoidProbability::mnormell = 
"Ellipsoid[`1`, `2`, `3`] does not correspond to a constant-probability \
contour of MultinormalDistribution[`1`, `4`]."


(* - - - - - - -  ExpectedValue[f, MultinormalDistribution[], x] - - - - - - *)
MultinormalDistribution/: ExpectedValue[f_Function,
	 MultinormalDistribution[mu_, sigma_]?DistributionParameterQ, opts___?OptionQ] :=
  Module[{n, (* number of arguments of function f *)
	  m = Length[mu], (* dimensionality of MultinormalDistribution *) 
	  xvec, x, assmp = Assumptions /. {opts} /. Options[ExpectedValue],
	  arglist, integral, unique},
   (
    xvec = Array[x, m]; 
    assmp = Flatten[Join[	Map[(Im[#]==0)&, mu],
         			Map[(# > 0)&, Diagonal[sigma]],
	 			{Det[sigma]>0},
	 			assmp			] /. True -> {}];
    arglist = Prepend[ Map[{#, -Infinity, Infinity}&, xvec],
		       Apply[f, xvec] *
			PDF[MultinormalDistribution[mu, sigma], xvec] ];	
    If[assmp =!= {}, AppendTo[arglist, Assumptions -> assmp]];
    If[FreeQ[integral = Apply[Integrate, arglist], Integrate],
       integral,
       unique = Table[Unique[], {m}];	
       integral /. Thread[Rule[xvec, unique]]		] (* end If *)
   ) /;(If[Length[f]==1,
	    (* Function with only a body *)
	    n = Max[Cases[{f}, Slot[z_]->z, Infinity]],
	    (* Function with a list of formal parameters *)
	    n = Length[f[[1]]]	];
	 n <= m)
  ]
	
MultinormalDistribution/: ExpectedValue[f_,
	MultinormalDistribution[mu_, sigma_]?DistributionParameterQ, xvec_?VectorQ,
	opts___?OptionQ] :=			
  Module[{assmp = Assumptions /. {opts} /. Options[ExpectedValue], arglist},
    assmp = Flatten[Join[	Map[(Im[#]==0)&, AboveDiagonalElements[sigma]],
		 	        Map[(Im[#]==0)&, mu],
                                Map[(# > 0)&, Diagonal[sigma]],
                                {Det[sigma]>0},
                                assmp                   ] /. True -> {}];
    arglist = Prepend[ Map[{#, -Infinity, Infinity}&, xvec],
		       f PDF[MultinormalDistribution[mu, sigma], xvec] ];	
    If[assmp =!= {}, AppendTo[arglist, Assumptions -> assmp]];
    Apply[Integrate, arglist]
  ] /;Length[xvec] <= Length[mu]
	

(* single random vector from a nonsingular multinormal *)
MultinormalDistribution/: Random[MultinormalDistribution[mu_?(VectorQ[#, NumericQ]&), sigma_?(MatrixQ[#, NumericQ]&)]] :=
  Module[{z = normalarray[0, 1, Length[mu]], lower},
  	lower=Transpose[CholDecomp[sigma]];
	mu + lower.z  
  ]/;nonsingularMultinormalQ[mu, sigma]
  
      		
(* ========================== Wishart Distribution ======================== *)

WishartDistribution/: DistributionParameterQ[WishartDistribution[sigma_, m_, l_]] :=
   MatrixQ[sigma] &&
   If[NumberQ[m], TrueQ[m > Length[sigma]], True] &&
   SymmetricMatrixQ[sigma] &&
   If[Apply[And, Map[NumberQ, Flatten[N[sigma]] ]],
      TrueQ[Det[N[sigma]] > 0] &&
         Apply[And, Map[TrueQ[# > 0]&, N[Diagonal[sigma]] ]],
      True]

validWishartQ[sigma_, m_, warnflag_:False] :=
  (
   MatrixQ[sigma] &&
   If[NumberQ[m], TrueQ[m > Length[sigma]], True] &&
   If[SymmetricMatrixQ[sigma], True,
       If[TrueQ[warnflag], Message[WishartDistribution::cmsym]];False] &&
   If[Apply[And, Map[NumberQ, Flatten[N[sigma]] ]],
      TrueQ[Det[N[sigma]] > 0] &&
         Apply[And, Map[TrueQ[# > 0]&, N[Diagonal[sigma]] ]],
      True]
  )

WishartDistribution::cmsym =
"The covariance matrix must be symmetric.";

(* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *)
(* NOTE: do not condition DistributionDomainQ on DistributionParameterQ so that DistributionDomainQ will always
        evaluate and be efficient. *)
WishartDistribution/: DistributionDomainQ[WishartDistribution[sigma_, m_, l___],
	 list_List] :=
  Module[{p},
    (
	Scan[Module[{matrix = #},
	       If[!TrueQ[SymmetricMatrixQ[matrix] &&
	                 Det[N[matrix]] > 0 &&
	                 Apply[And, Map[(# > 0)&,
				 N[Diagonal[matrix]] ]]		],
		  Return[False]
	       ]
	     ]&, list] =!= False	
    ) /; (p = Length[sigma];  MatchQ[Dimensions[list], {_, p, p}])
  ] 
WishartDistribution/: DistributionDomainQ[WishartDistribution[sigma_, m_, l___],
	 x_?MatrixQ] :=
  Module[{p = Length[sigma]},
       TrueQ[Dimensions[x] == {p, p} && x === Transpose[x] &&
	     Det[N[x]] > 0 &&
	     Apply[And, Map[(# > 0)&, N[Diagonal[x]] ]]
       ]
  ]

(* The following iDomainQ allows for symbolic arguments *)
iDomainQ[WishartDistribution[sigma_, m_, l___], 
  x_?(MatrixQ[N[#], NumericQ] &)] := 
 SymmetricMatrixQ[x] && PositiveDefiniteMatrixQ[x]

iDomainQ[WishartDistribution[sigma_, m_, l___], x_?MatrixQ] := 
 SymmetricMatrixQ[x] && 
  Not[Apply[And, Map[(# > 0) &, N[Diagonal[x]]]] === False || 
    Reduce[Det[x] > 0] === False]

iDomainQ[WishartDistribution[sigma_, m_, l___], _] := False

(* - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - *)
(* compute lower triangular matrix for use in calculations involving
	WishartDistribution *)
WishartDistribution[sigma_:{{1}}, m_:1] :=
  Module[{u},
    (
    WishartDistribution[sigma, m, Transpose[u]
    ]	
    ) /; FreeQ[u = CholDecomp[sigma], CholDecomp]
  ] /; validWishartQ[sigma, m, True]

(* hide lower triangular matrix using Format *)
WishartDistribution /: Format[WishartDistribution[sigma_, m_, l_]] :=
	WishartDistribution[Short[sigma], m]

(* special rule for sigma = IdentityMatrix[p] *)
WishartDistribution[sigma_?MatrixQ, m_?IntegerQ] :=
  Module[{p},
    (
  	WishartDistribution[sigma, m, 
		Table[Prepend[Table[0, {i}], 1], {i, 0, p-1}]]
    ) /; (p = Length[sigma];
	  (m >= p + 1) && sigma === IdentityMatrix[p])
  ] /; Apply[Equal, Dimensions[sigma]]

(* - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - *)

(* The following indicates that the domain of WishartDistribution is the
	set of all appropriately-sized symmetric positive definite matrices. *)
WishartDistribution/: DistributionDomain[WishartDistribution[sigma_, m_, l___]] :=
 (
 (Dimensions[#] == Dimensions[sigma] && # === Transpose[#] && Det[#1] > 0)&
 ) /; ({l} =!= {} || validWishartQ[sigma, m]) 

WishartDistribution/: PDF[WishartDistribution[sigma_, m_, l___], x_?MatrixQ] :=
  0 /; ({l} =!= {} || validWishartQ[sigma, m]) &&
       !iDomainQ[WishartDistribution[sigma, m], x]

WishartDistribution/: PDF[WishartDistribution[sigma_, m_, l___], x_?MatrixQ] :=
  With[{invsigma = Inverse[sigma], p = Length[sigma]},
    iK[m, invsigma] Det[x]^((m-p-1)/2) Exp[Tr[-1/2 invsigma.x]]
  ] /; ({l} =!= {} || validWishartQ[sigma, m]) &&
       iDomainQ[WishartDistribution[sigma, m], x]

(*  - - - - - - - - - -  CDF[WishartDistribution[], ]  - - - - - - - - - - - *)

WishartDistribution/: CDF[WishartDistribution[sigma_, m_, l___], x_?MatrixQ] :=
  (
  0
  ) /; ({l} =!= {} || validWishartQ[sigma, m]) &&
       !iDomainQ[WishartDistribution[sigma, m], x]

WishartDistribution/: CDF[WishartDistribution[sigma_, m_, l___], x_?MatrixQ] :=
   (
   GammaRegularized[ m/2, 0, x[[1, 1]]/2/sigma[[1, 1]] ]
   ) /; Length[x] == 1 &&
       ({l} =!= {} || validWishartQ[sigma, m]) &&
       iDomainQ[WishartDistribution[sigma, m], x]


(* Johnson & Kotz, p. 162, eq. (11) *)
iK[nu_, C_] :=
  With[{p = Length[C]},
    Det[C]^(nu/2) ( 2^(nu p/2) MultivariateGamma[p, nu/2] )^(-1)
  ]	

(* Johnson & Kotz, p. 162, eq. (12) *)
MultivariateGamma[p_, z_] := 
	 Pi^(p(p-1)/4) Product[Gamma[(2z - j + 1)/2], {j, 1, p}]

(* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *)
 
WishartDistribution/: Mean[WishartDistribution[sigma_, m_, l___]] := m sigma /;
	({l} =!= {} || validWishartQ[sigma, m])

WishartDistribution/: Variance[WishartDistribution[sigma_, m_, l___]] :=
   With[{diag = Diagonal[sigma]},
	m (sigma^2 + Map[List, diag].{diag})
   ] /; ({l} =!= {} || validWishartQ[sigma, m])  

WishartDistribution/: StandardDeviation[
   WishartDistribution[sigma_, m_, l___]] :=
	Module[{var},
	  (
          Sqrt[var]	
	  ) /; FreeQ[var = Variance[WishartDistribution[sigma, m, l]], Variance]
	] /; ({l} =!= {} || validWishartQ[sigma, m])

(* NOTE: Covariance and Correlation are not implemented for the
	WishartDistribution.  They could be if the p(p+1)/2 unique entries of
	a Wishart matrix were ordered.  Currently, the distribution is
	considered to be matrix-valued, rather than vector-valued with the 
	entries taken from a matrix. *)

WishartDistribution/: Skewness[WishartDistribution[sigma_, m_, l___]] :=
    With[{outer = (Outer[Times, #, #] &[Diagonal[sigma]]), V2 = sigma^2},
        2 m sigma (V2 + 3 outer)/Power[m (V2 + outer), 3/2]
    ] /; ({l} =!= {} || validWishartQ[sigma, m])  

(* NOTE: MultivariateSkewness[WishartDistribution[sigma, m]] not implemented. *)

WishartDistribution/: Kurtosis[WishartDistribution[sigma_, m_, l___]] :=
    Module[{outer, V2, var},
        outer = Outer[Times, #, #] &[Diagonal[sigma]];
        V2 = sigma^2;
        var = V2 + outer;
        (3/m) ((m + 2) + 8 outer V2/var^2)
    ] /; ({l} =!= {} || validWishartQ[sigma, m]) 

(* NOTE: MultivariateKurtosis[WishartDistribution[sigma, m]] not implemented. *)

(* NOTE:
  CharacteristicFunction[dist, t] for a random matrix x is defined as the
	expected value of
  Exp[Tr[-I(t + DiagonalMatrix[Diagonal[t]]).x]],
	where Dimensions[t] == Dimensions[x].
  Johnson & Kotz, Chap. 38, eq. 18.2
*)
WishartDistribution/: CharacteristicFunction[
                WishartDistribution[sigma_, m_, l___], t_?MatrixQ] :=
  With[{invsigma = Inverse[sigma]},
    ( Det[invsigma] )^(m/2) /
    ( Det[invsigma - I (t + DiagonalMatrix[Diagonal[t]])] )^(m/2)
  ] /; ({l} =!= {} || validWishartQ[sigma, m]) &&
       TrueQ[Dimensions[t] == Dimensions[sigma]] && t === Transpose[t]

(* NOTE:
Quantile is not implemented for WishartDistribution[sigma, m].
Actually since there is no unique matrix x such that
q = CDF[WishartDistribution[sigma, m], x], it is difficult to even
define Quantile[WishartDistribution[sigma, m], q]. *)

(* NOTE: ExpectedValue of WishartDistribution not implemented. *)


(* the following uses Smith and Hocking's method for Wishart generation
   W. B. Smith and R. R. Hocking, "Algorithm AS 53: Wishart Variate Generator", 
  	Applied Statistics, 1972, Vol. 21, No. 3. *)
WishartDistribution /: 
 Random`DistributionVector[WishartDistribution[sigma_, m_, l_], n_Integer, 
  prec_] := Internal`DeactivateMessages[
  Block[{chol = CholeskyDecomposition[sigma], dim = Length[sigma[[1]]], 
  	rmat, wrap}, 
  	If[(#<prec||(#===MachinePrecision&&NumberQ[prec]))&[Precision[sigma]],
  		Message[RandomReal::precw,sigma,prec]];
  	rmat = SparseArray[
  	  {{i_, j_} /; i < j :> wrap[RandomReal[NormalDistribution[0, 1], n, WorkingPrecision -> prec]], 
  	  {i_, i_} :> wrap[Sqrt[RandomReal[ChiSquareDistribution[(m - i + 1)], n, WorkingPrecision -> prec]]]}, 
  	  {dim, dim}, wrap[If[prec===MachinePrecision,Table[0., {n}],Table[0,{n}]]]]; 
	rmat = Developer`ToPackedArray[Normal[rmat] /. wrap -> Identity]; 
	rmat = Flatten[rmat, {{3}, {1}, {2}}]; 
	rmat = rmat.chol; 
	Map[Transpose[#].# &, rmat]]
	,
	Precision::"mxprec"]/;(validWishartQ[sigma, m] && VectorQ[N[Flatten[{m,sigma}]],NumberQ]&&
   	Dimensions[sigma]===Dimensions[l])

(WishartDistribution[sigma_, m_, l_])@"RandomType"=RandomReal;

WishartDistribution/: Random[WishartDistribution[sigma_, m_, l_]] :=
   Module[{dist = MultinormalDistribution[Table[0, {Length[sigma]}], sigma],
	   dataMatrix},
     dataMatrix = RandomArray[dist, m];
     Transpose[dataMatrix].dataMatrix
   ] /; (IntegerQ[m] && m >= Length[sigma]+1 && MatrixQ[sigma] && 
   	Dimensions[sigma]===Dimensions[l])
   	  
(* ======================== QuadraticForm Distribution ====================== *)


QuadraticFormDistribution/: DistributionParameterQ[QuadraticFormDistribution[{a_, b_, c_},
	 {mu_, sigma_}, {lambda_, beta_, alpha_}, rest___]] := 
   DistributionParameterQ[MultinormalDistribution[mu, sigma]] && 
   MatrixQ[a] && VectorQ[b] && (Head[c] =!= List) &&
   (Dimensions[a] === {Length[b], Length[b]} === Dimensions[sigma]) &&
   SymmetricMatrixQ[a]

validQuadraticFormQ[{a_, b_, c_}, {mu_, sigma_}, warnflag_:False] :=
   If[DistributionParameterQ[MultinormalDistribution[mu, sigma]],
      True,
      If[TrueQ[warnflag], Message[QuadraticFormDistribution::multinorm]];False] && 
   If[MatrixQ[a] && VectorQ[b] && (Head[c] =!= List),
      True,
      If[TrueQ[warnflag], Message[QuadraticFormDistribution::badcoefs]];False] &&
   If[(Dimensions[a] === {Length[b], Length[b]} === Dimensions[sigma]),
      True,
      If[TrueQ[warnflag], Message[QuadraticFormDistribution::coefdims]];False] &&
   If[SymmetricMatrixQ[a],
      True,
      If[TrueQ[warnflag], Message[QuadraticFormDistribution::coefsym]];False]

QuadraticFormDistribution::multinorm =
"Second argument must be a list of valid MultinormalDistribution arguments.";

QuadraticFormDistribution::badcoefs =
"The quadratic coefficients should be a matrix, a vector, and a scalar, \
respectively.";

QuadraticFormDistribution::coefdims =
"The quadratric coefficients must have dimensions commensurate with \
the covariance matrix of the multinormal distribution.";

QuadraticFormDistribution::coefsym =
"The first quadratic coefficient must be a symmetric matrix.";

(* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *)
(* NOTE: do not condition DistributionDomainQ on DistributionParameterQ so that DistributionDomainQ will always
        evaluate and be efficient. *)
QuadraticFormDistribution/: DistributionDomainQ[QuadraticFormDistribution[{a_, b_, c_},
        {mu_, sigma_}, {lambda_, beta_, alpha_}, lowertri_], list_?VectorQ] :=
        FreeQ[N[list], Complex] &&
 	If[Det[a] > 0,
	   With[{low = alpha - Apply[Plus, beta^2/lambda]/4}, 
	     Scan[If[!TrueQ[# >= low], Return[False]]&, list] =!= False  ],
	   True,
	   False]
QuadraticFormDistribution/: DistributionDomainQ[QuadraticFormDistribution[{a_, b_, c_},
        {mu_, sigma_}, {lambda_, beta_, alpha_}, lowertri_], x_] :=
        FreeQ[N[x], Complex] &&
 	If[Det[a] > 0, x >= alpha - Apply[Plus, beta^2/lambda]/4, True, False]

(* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *)
(* assume a = {{1}}, b = {0}, c = 0, mu = {0}, sigma = {{1}} *)
QuadraticFormDistribution[] := ChiSquareDistribution[1]

(* assume mu = Table[0, {Length[b]}] and sigma = IdentityMatrix[Length[b]] *)
QuadraticFormDistribution[{a_?MatrixQ, b_?VectorQ, c_}] :=
  Module[{p = Length[b]},
    QuadraticFormDistribution[{a, b, c}, {Table[0, {p}], IdentityMatrix[p]}]
  ]  /; Dimensions[a] === {Length[b], Length[b]} && Det[a] > 0

(* assume a = IdentityMatrix[Length[mu]] and b = Table[0, {Length[mu]}] and
	 c = 0 *)
QuadraticFormDistribution[{mu_, sigma_}] :=
  Module[{p = Length[mu]},
    QuadraticFormDistribution[{IdentityMatrix[p],
	 Table[0, {p}], 0}, {mu, sigma}]
  ] /; DistributionParameterQ[MultinormalDistribution[mu, sigma]] 
	

(* QuadraticFormDistribution -> NormalDistribution for linear forms *)
QuadraticFormDistribution[{a_?MatrixQ, b_?VectorQ, c_}, {mu_, sigma_}] :=
	NormalDistribution[b.mu + c, Sqrt[b.sigma.b]] /; ZeroMatrixQ[a] &&
		DistributionParameterQ[MultinormalDistribution[mu, sigma]] &&
		(Head[c] =!= List) && Length[b]==Length[mu]

(* compute reparametrization for use in calculations involving
	QuadraticFormDistribution *)
QuadraticFormDistribution[{a_?MatrixQ, b_?VectorQ, c_}, {mu_, sigma_}] :=
  Module[{u, esys, lambda, beta, alpha, evecs},
    (
    (* use ComplexExpand to simplify expressions involving Conjugate 
       for symbolic parameters; for valid a, b, c, mu, and sigma,
       there will be no imaginary terms *)
    esys=ComplexExpand[esys];
    {lambda, evecs} = esys;
    evecs = (#/Sqrt[#.#])& /@ evecs; (* P P' = I *)
    beta = evecs.(u.b + 2u.a.mu);
    alpha = mu.a.mu + b.mu + c;
    QuadraticFormDistribution[
	{a, b, c}, {mu, sigma}, 
	{lambda, beta, alpha},	(* reparameterization *)
	Transpose[u] (* for random num gen *)
    ]
    ) /; FreeQ[u = CholDecomp[sigma], CholDecomp] &&
	 FreeQ[esys = Eigensystem[u.a.Transpose[u]], Eigensystem]
  ] /; validQuadraticFormQ[{a, b, c}, {mu, sigma}, True]

(* hide additional parameters using Format *)
QuadraticFormDistribution /: Format[QuadraticFormDistribution[
   {a_, b_, c_}, {mu_, sigma_}, {lambda_, beta_, alpha_}, lowertri_]] :=
	QuadraticFormDistribution[{Short[a], b, c}, {mu, Short[sigma]}]	

(* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *)

(* Mathai & Provost, p. 97 *)
(* if {lambda, beta, alpha} exist, assume valid distribution *)
QuadraticFormDistribution/: DistributionDomain[QuadraticFormDistribution[{a_, b_, c_},
	{mu_, sigma_}, {lambda_, beta_, alpha_}, lowertri_]] :=
	If[Det[a] > 0, 
	   Interval[{alpha - Apply[Plus, beta^2/lambda]/4, Infinity}],
	   Interval[{-Infinity, Infinity}]
	]

(*  - - - - - - - - - - - PDF of QuadraticFormDistribution  - - - - - - - - *)

QuadraticFormDistribution/: PDF[QuadraticFormDistribution[{a_, b_, c_},
  {mu_, sigma_}, {lambda_, beta_, alpha_}, lowertri_], x_] :=
     Module[{noncentralparm, gamma, delta},
	gamma = a[[1, 1]] sigma[[1, 1]];
 	delta = c - b[[1]]^2/(4 a[[1, 1]]);	
	noncentralparm = (b[[1]]/(2 a[[1, 1]]) + mu[[1]])^2 / sigma[[1, 1]];
	If[Evaluate[noncentralparm == 0],
	   Evaluate[PDF[ChiSquareDistribution[1], (x-delta)/gamma]],
	   Evaluate[PDF[NoncentralChiSquareDistribution[1, noncentralparm],
		(x-delta)/gamma]]	
        ] / Abs[gamma]
     ] /; Length[b] == 1

QuadraticFormDistribution/: PDF[QuadraticFormDistribution[{a_?MatrixQ, b_, c_},
    {mu_, sigma_}, {lambda_, beta_, alpha_}, lowertri_], x_] :=
	  (
	  0
	  ) /; TrueQ[x < alpha - Apply[Plus, beta^2/lambda]/4] &&
  		TrueQ[Det[a] > 0] 


Unprotect[PDF]

(* Mathai & Provost, p. 93 *)
(* A LaguerreL polynomial expansion is preferable to the power series
	expansion provided by Series and should be implemented someday! *)
PDF/: Series[PDF[QuadraticFormDistribution[{a_?MatrixQ, b_, c_}, {mu_, sigma_},
	{lambda_, beta_, alpha_}, lowertri_], x_],
	{x_Symbol, x0_, n_Integer}] :=
  Module[{kmax, nmin, nmax, cvec, cc, dvec, dd, r, p = Length[mu], cutoff},
    (
    kmax = Max[0, Floor[n-p/2+1]];
			 (* kmax+1 is the number of nonzero non-order terms *)
    nmin = p/2 - 1;	   (* smallest exponent of (x-x0) *)
    nmax = p/2 + kmax - 1; (* largest exponent of (x-x0) *)
    cvec = Array[cc, n+1, 0]; dvec = Array[dd, n];
    dvec = Table[(-1)^r/2 Apply[Plus, (1 - r (beta/lambda)^2/4)/(2 lambda)^r],
		 {r, kmax}];
    cc[0] = Exp[-Apply[Plus, (beta/lambda)^2]/8] / Sqrt[Apply[Times, 2 lambda]];
    Scan[(cc[#] = (Reverse[Take[dvec, #]].Take[cvec, #])/#)&, Range[n]];
    cvec /= Table[Gamma[p/2 + r], {r, 0, kmax}];
    If[OddQ[p],	
       SeriesData[x, x0, Flatten[Map[{#, 0}&, cvec]],
		 Numerator[nmin], Numerator[nmax+1], 2],
       SeriesData[x, x0, cvec, nmin, nmax+1, 1] 
    ]
    ) /; (cutoff = alpha - Apply[Plus, beta^2/lambda]/4;
	  If[!TrueQ[N[x0-cutoff]==0||Simplify[x0-cutoff,TimeConstraint->1]===0],
	    Message[Series::qforig,
		    x, cutoff, 
		    PDF[QuadraticFormDistribution[{a, b, c}, {mu, sigma},
			{lambda, beta, alpha}, lowertri], x],
		    {x, cutoff, n}]; False, 
	    True])
  ]  /; n >= 0 && Apply[Equal, Dimensions[a]] &&
	If[TrueQ[Det[a] <= 0], Message[Series::qfpos]; False, True]

(*  - - - - - - - - - - - CDF of QuadraticFormDistribution  - - - - - - - - *)

QuadraticFormDistribution/: CDF[QuadraticFormDistribution[{a_, b_, c_},
  {mu_, sigma_}, {lambda_, beta_, alpha_}, lowertri_], x_] :=
     Module[{noncentralparm, gamma, delta},
	gamma = a[[1, 1]] sigma[[1, 1]];
 	delta = c - b[[1]]^2/(4 a[[1, 1]]);	
	noncentralparm = (b[[1]]/(2 a[[1, 1]]) + mu[[1]])^2 / sigma[[1, 1]];
	If[Evaluate[noncentralparm == 0],
	   Evaluate[If[Evaluate[a[[1, 1]] > 0],
	      Evaluate[CDF[ChiSquareDistribution[1], (x-delta)/gamma]],
	      Evaluate[1 - CDF[ChiSquareDistribution[1], (x-delta)/gamma]]	
	   ]],
	   Evaluate[If[Evaluate[a[[1, 1]] > 0],
	      Evaluate[CDF[NoncentralChiSquareDistribution[1, noncentralparm],
		(x-delta)/gamma]],
	      Evaluate[1 - CDF[NoncentralChiSquareDistribution[1,
		 noncentralparm], (x-delta)/gamma]]
	   ]]
	]
     ] /; Length[b] == 1

QuadraticFormDistribution/: CDF[QuadraticFormDistribution[{a_?MatrixQ, b_, c_},
    {mu_, sigma_}, {lambda_, beta_, alpha_}, lowertri_], x_] :=
	  (
	  0
	  ) /; TrueQ[x < alpha - Apply[Plus, beta^2/lambda]/4] &&
  		TrueQ[Det[a] > 0] 


Unprotect[CDF]

CDF/: Series[CDF[QuadraticFormDistribution[{a_?MatrixQ, b_, c_}, {mu_, sigma_},
	{lambda_, beta_, alpha_}, lowertri_], x_],
	{x_Symbol, x0_, n_Integer}] :=
  Module[{pdfseries, cutoff},
    (
        pdfseries = Series[PDF[QuadraticFormDistribution[{a, b, c},
		{mu, sigma}, {lambda, beta, alpha}, lowertri], x],
		{x, x0, n-1}];
	Integrate[pdfseries, x]
    ) /; (cutoff = alpha - Apply[Plus, beta^2/lambda]/4;
	  If[!TrueQ[N[x0-cutoff]==0||Simplify[x0-cutoff,TimeConstraint->1]===0],
	    Message[Series::qforig,
		    x, cutoff, 
		    CDF[QuadraticFormDistribution[{a, b, c}, {mu, sigma},
			{lambda, beta, alpha}, lowertri], x],
		    {x, cutoff, n}]; False, 
	    True])
  ] /; n >= 1 && Apply[Equal, Dimensions[a]] &&
	If[TrueQ[Det[a] <= 0], Message[Series::qfpos]; False, True] 
	
CDF/: Series[CDF[QuadraticFormDistribution[{a_?MatrixQ, b_, c_}, {mu_, sigma_},
	{lambda_, beta_, alpha_}, lowertri_], x_], {x_Symbol, x0_, 0}] :=
  Module[{cutoff},
    (
    SeriesData[x, 0, List[], 1, 1, 1]
    ) /; (cutoff = alpha - Apply[Plus, beta^2/lambda]/4;
	  If[!TrueQ[N[x0-cutoff]==0],
             Message[Series::qforig,
		     x, cutoff,
		     CDF[QuadraticFormDistribution[{a, b, c}, {mu, sigma},	
                       {lambda, beta, alpha}, lowertri], x],
		     {x, cutoff, 0}]; False,
             True])
  ] /; Apply[Equal, Dimensions[a]] &&
	If[TrueQ[Det[a] <= 0], Message[Series::qfpos]; False, True]

Series::qfpos =
"The Series expansions of PDF[QuadraticFormDistribution[{A, B, C}, {mu, sigma}], \
x] or CDF[QuadraticFormDistribution[{A, B, C}, {mu, sigma}], x] are implemented \
only for the case of positive definite A matrix."

Series::qforig =
"The Series expansions of the PDF and CDF of the QuadraticFormDistribution \
must be about the lower point of the domain `1` = `2`. \
Try Series[`3`, `4`]."

(* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *)

(* Mathai & Provost, eq. 3.2b.8, used for Mean, StandardDeviation, Variance,
	Skewness, Kurtosis *)
QuadraticFormDistribution/: Mean[QuadraticFormDistribution[
   {a_, b_, c_}, {mu_, sigma_}, {lambda_, beta_, alpha_}, ___]] :=
	(
       Apply[Plus, lambda] + alpha
	) (* if {lambda, beta, alpha} exist, assume valid distribution *)

QuadraticFormDistribution/: StandardDeviation[
	QuadraticFormDistribution[{a_, b_, c_}, {mu_, sigma_},
		 {lambda_, beta_, alpha_}, rest___]] :=
   Module[{var},
     (
     Sqrt[var]
     ) /; FreeQ[var = Variance[QuadraticFormDistribution[{a, b, c},
		{mu, sigma}, {lambda, beta, alpha}, rest]], Variance]
   ]

QuadraticFormDistribution/: Variance[QuadraticFormDistribution[
   {a_, b_, c_}, {mu_, sigma_}, {lambda_, beta_, alpha_}, ___]] :=
	(
	Together[2 Apply[Plus, lambda^2] + Apply[Plus, beta^2]]
	) (* if {lambda, beta, alpha} exist, assume valid distribution *) 

QuadraticFormDistribution/: Skewness[QuadraticFormDistribution[
	{a_, b_, c_}, {mu_, sigma_}, {lambda_, beta_, alpha_}, rest___]] := 
  Module[{var},
    (
      Together[(8 Apply[Plus, lambda^3] + 6 Apply[Plus, beta^2 lambda])] var^(-3/2)
    ) /; FreeQ[var = Variance[QuadraticFormDistribution[{a, b, c},
                {mu, sigma}, {lambda, beta, alpha}, rest]], Variance] 
  ]

QuadraticFormDistribution/: Kurtosis[qfd:QuadraticFormDistribution[
   {a_, b_, c_}, {mu_, sigma_}, {lambda_, beta_, alpha_}, rest___]] := 
  Module[{var},
    (Together[(48 Apply[Plus, lambda^4 + beta^2 lambda^2])]/var^2
    + 3)/; FreeQ[var = Variance[qfd], Variance]
  ]  (* if {lambda, beta, alpha} exist, assume valid distribution *)


QuadraticFormDistribution/: CharacteristicFunction[QuadraticFormDistribution[
   {a_, b_, c_}, {mu_, sigma_}, {lambda_, beta_, alpha_}, rest___], t_] :=
   Module[{terms = Map[(1 - 2 t I #)&, lambda]},
     Exp[alpha t I - t^2/2 (beta^2) . (terms^(-1))] *
       Apply[Times, terms^(-1/2)]
   ] (* if {lambda, beta, alpha} exist, assume valid distribution *)


(* ================== Quantile for QuadraticFormDistribution =============== *)

(* NOTE:
  Quantile is not implemented for QuadraticFormDistribution[{a, b, c},
  {mu, sigma}, {lambda, beta, alpha}] for p > 1.  Since
  QuadraticFormDistribution is univariate,
  Quantile[QuadraticFormDistribution[], q] does have a unique definition.
  Unfortunately, only various approximations are available for p > 1. *)

QuadraticFormDistribution/: Quantile[QuadraticFormDistribution[{a_, b_, c_},
  {mu_, sigma_}, {lambda_, beta_, alpha_}, lowertri_], q_] :=
     Module[{noncentralparm, gamma, delta},
	gamma = a[[1, 1]] sigma[[1, 1]];
	delta = c - b[[1]]^2/(4 a[[1, 1]]);
	noncentralparm = (b[[1]]/(2 a[[1, 1]]) + mu[[1]])^2 / sigma[[1, 1]];
	If[Evaluate[noncentralparm == 0],
	   Evaluate[gamma Quantile[ChiSquareDistribution[1], q] + delta],
	   Evaluate[gamma Quantile[NoncentralChiSquareDistribution[1,
		    noncentralparm], q] + delta]
	]		
     ] /; Length[b] == 1 && NumberQ[q] && 0 <= q <= 1


(* NOTE: ExpectedValue of QuadraticFormDistribution is not implemented. *)


QuadraticFormDistribution /: 
 Random`DistributionVector[
  QuadraticFormDistribution[{a_, b_, c_}, {mu_, sigma_}, 
  	{lambda_, beta_, alpha_}, lowertri_], m_Integer, prec_] := 
  Module[{array}, 
  	array = RandomReal[MultinormalDistribution[mu, sigma], m, 
    	WorkingPrecision -> prec]; 
  	Map[(#.a.# + b.# + c) &, array] 
  	] /;validQuadraticFormQ[{a, b, c}, {mu, sigma}, {lambda, beta, alpha}]&&
  		VectorQ[N[Flatten[{a,b,c,mu,sigma,lambda,beta,alpha}]],NumberQ]
  	
(QuadraticFormDistribution[{a_, b_, c_}, {mu_, sigma_}, 
  	{lambda_, beta_, alpha_}, lowertri_])@"RandomType"=RandomReal;

QuadraticFormDistribution/: Random[QuadraticFormDistribution[{a_, b_, c_},
	{mu_, sigma_}, {lambda_, beta_, alpha_}, lowertri_]] :=
  Module[{z = Random[MultinormalDistribution[mu, sigma]]},
    z.a.z + b.z + c
  ] (* if {lambda, beta, alpha} exist, assume valid distribution *)
  
(* ==================== Hotelling T-squared Distribution =================== *)

(* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *)
(* NOTE: do not condition DistributionDomainQ on DistributionParameterQ so that DistributionDomainQ will always
        evaluate and be efficient. *)
HotellingTSquareDistribution/: DistributionDomainQ[HotellingTSquareDistribution[p_, m_],
	 list_?VectorQ] :=
	FreeQ[N[list], Complex] &&
	Scan[If[!TrueQ[# >= 0], Return[False]]&, list] =!= False

HotellingTSquareDistribution/: DistributionDomainQ[HotellingTSquareDistribution[p_, m_],
	 x_] :=
	FreeQ[N[x], Complex] && TrueQ[x >= 0]



HotellingTSquareDistribution/: ExpectedValue[f_Function,
		 HotellingTSquareDistribution[p_, m_], opts___?OptionQ] :=
   Module[{x, integral,
           assmp = Assumptions /. {opts} /. Options[ExpectedValue]},
     If[FreeQ[integral = Integrate[ f[x] *
				 PDF[HotellingTSquareDistribution[p, m], x],
                                        {x, 0, Infinity},
                        Assumptions -> Join[{p > 0, m > 0, m-p+1 > 0}, Flatten[{assmp}]]],
              Integrate],
        integral,
        integral /. x -> Unique[]       ]
   ] /; DistributionParameterQ[HotellingTSquareDistribution[p, m]]

HotellingTSquareDistribution/: ExpectedValue[f_,
	 HotellingTSquareDistribution[p_, m_], x_, opts___?OptionQ] :=
  Module[{assmp = Assumptions /. {opts} /. Options[ExpectedValue]},
   Integrate[ f PDF[HotellingTSquareDistribution[p, m], x],
                {x, 0, Infinity},
                Assumptions -> Join[{p > 0, m > 0, m-p+1 > 0}, Flatten[{assmp}]]]
  ] /; DistributionParameterQ[HotellingTSquareDistribution[p, m]]

HotellingTSquareDistribution/: Random[HotellingTSquareDistribution[p_, m_]] :=
	(m p/(m-p+1)) Random[FRatioDistribution[p, m-p+1]]

(* =================== Multivariate StudentT Distribution =================  *)

validMultivariateTQ[sigma_, m_, warnflag_: False] := 
 And[If[MatrixQ[sigma] && SymmetricMatrixQ[sigma] && FreeQ[N[sigma], Complex] &&
    If[MatrixQ[N[sigma], NumberQ],
     	TrueQ[Det[N[sigma]] > 0], 
     	True],
   	True,
   	If[TrueQ[warnflag], 
    	Message[MultivariateTDistribution::posdef, sigma]];
   		False],
  	If[If[NumberQ[m], m > 0, True],
   		True,
   		If[TrueQ[warnflag], 
    		Message[MultivariateTDistribution::posparm, m]];
   			False]
   	]

MultivariateTDistribution::posdef =
"The scale matrix must be symmetric and positive definite.";

(* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *)
(* NOTE: do not condition DistributionDomainQ on DistributionParameterQ so that DistributionDomainQ will always
        evaluate and be efficient. *)
MultivariateTDistribution/: DistributionDomainQ[MultivariateTDistribution[r_, m_],
        list_?MatrixQ] :=
   MatchQ[Dimensions[list], {_, Length[r]}] && FreeQ[N[list], Complex]
   
MultivariateTDistribution/: DistributionDomainQ[MultivariateTDistribution[r_, m_],
        x_?VectorQ] :=
   TrueQ[Length[x] == Length[r] && FreeQ[N[x], Complex]]


MultivariateTDistribution/: MultivariateSkewness[MultivariateTDistribution[
	r_, m_]] := Piecewise[{{0,m>3}},Indeterminate] /; validMultivariateTQ[r, m]

MultivariateTDistribution/: MultivariateKurtosis[MultivariateTDistribution[
	r_?MatrixQ, m_]] := 
	Piecewise[{{With[{p = Length[r]},
		p(p + 2)(m - 2)/(m - 4)], m>4}},
		Indeterminate] /; validMultivariateTQ[r, m]

MultivariateTDistribution/: EllipsoidQuantile[MultivariateTDistribution[
	r_, m_], q_] :=
  Module[{ellipsoid, quantile, p},
    (
    ellipsoid
    ) /; (p = Length[r];
	  quantile = Quantile[FRatioDistribution[p, m], q];
	  ellipsoid = ellipsoidalLocus[Table[0, {p}], quantile p r];
	  If[ellipsoid === $Failed,
		Message[EllipsoidQuantile::mvarteig, quantile p r] ];
	  ellipsoid =!= $Failed)
  ] /; QuantileQ[q] && validMultivariateTQ[r, m]

EllipsoidQuantile::mvarteig =
"Unable to find eigensystem of ``."

MultivariateTDistribution/: EllipsoidProbability[MultivariateTDistribution[
	r_?MatrixQ, m_], EllipsoidQuantile[mu1_, radii_]] :=
		0 /; validMultivariateTQ[r, m]&&radii === Table[0, {Length[r]}]

MultivariateTDistribution/: EllipsoidProbability[MultivariateTDistribution[
	r_?MatrixQ, m_], EllipsoidQuantile[mu1_, radii_]] :=
		1 /; validMultivariateTQ[r, m]&&radii === Table[Infinity, {Length[r]}]
	
MultivariateTDistribution/: EllipsoidProbability[MultivariateTDistribution[
        r_, m_],
        Ellipsoid[mu_, radii_?VectorQ, dir___?MatrixQ]] :=
   Module[{p, internaldir, normalized, diagonalizedMatrix, acc, prec, delta,
		 scaled, quantile, diagonal},
     (
        CDF[ FRatioDistribution[p, m], quantile ]
     ) /; validMultivariateTQ[r, m] &&
          (p = Length[r];  TrueQ[mu == Table[0, {p}]]) &&
          (  internaldir = If[{dir}==={}, IdentityMatrix[p], dir];
             normalized = r.Transpose[internaldir];
	     diagonalizedMatrix = internaldir.normalized;
	     acc = Max[Accuracy[diagonalizedMatrix], $MachinePrecision]-1;
             prec = Max[Precision[diagonalizedMatrix], $MachinePrecision]-1;
             delta = Max[10^(-acc), Abs[Det[diagonalizedMatrix]] 10^(-prec)];	
             diagonalizedMatrix = Simplify[Chop[diagonalizedMatrix, delta]];
             If[ MatchQ[ diagonalizedMatrix, DiagonalMatrix[Table[_, {p}]] ],
		True,
		Message[EllipsoidProbability::mvartell,
			 mu, radii, internaldir, r, m];
		False] &&
          (diagonal = Diagonal[diagonalizedMatrix]/radii^2;
	   acc = Max[Accuracy[diagonal], $MachinePrecision]-3;
           prec = Max[Precision[diagonal], $MachinePrecision]-3;
           delta = Max[10^(-acc), Apply[Times, Abs[diagonal]] 10^(-prec)];
	   Apply[Equal, Chop[diagonal, delta]]) &&
          (  scaled = Transpose[internaldir].DiagonalMatrix[radii^2];
             quantile = Scan[If[ !TrueQ[#[[1]]==0], Return[#[[2]]/#[[1]]/p] ]&,
                       Transpose[{Flatten[normalized], Flatten[scaled]}] ];
             quantile =!= Null)
          )
   ]

EllipsoidProbability::mvartell = 
"Ellipsoid[`1`, `2`, `3`] does not correspond to a constant-probability \
contour of MultivariateTDistribution[`4`, `5`]."

(* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *)

MultivariateTDistribution/: ExpectedValue[f_Function,
	 MultivariateTDistribution[r_, m_], opts___?OptionQ] :=
  Module[{n, (* number of arguments of function f *)
	  p = Length[r], (* dimensionality of MultinormalDistribution *)
	  xvec, x, assmp = Assumptions /. {opts} /. Options[ExpectedValue],
	  correlations = AboveDiagonalElements[r],
	  arglist, integral,unique},
   (
    xvec = Array[x, p]; 
    assmp = Flatten[Join[	Map[(#^2 < 1)&, correlations],
	 			{Det[r] > 0, m > 0},
	 			assmp			] /. True -> {}];
    arglist = Prepend[ Map[{#, -Infinity, Infinity}&, xvec],
		       Apply[f, xvec] *
		        PDF[MultivariateTDistribution[r, m], xvec] ];
    If[assmp =!= {}, AppendTo[arglist, Assumptions -> assmp]];
    If[FreeQ[integral = Apply[Integrate, arglist], Integrate],
       integral,
       unique = Table[Unique[], {p}];	
       integral /. Thread[Rule[xvec, unique]]		] (* end If *)
   ) /; validMultivariateTQ[r, m]&&(
	 n = If[Length[f]==1,
	        (* Function with only a body *)
	        Max[Cases[{f}, Slot[z_]->z, Infinity]],
	        (* Function with a list of formal parameters *)
	        Length[f[[1]]]	];
	 n <= p)
  ]
	
MultivariateTDistribution/: ExpectedValue[f_,
	MultivariateTDistribution[r_, m_], xvec_?VectorQ,
	opts___?OptionQ] :=			
  Module[{assmp = Assumptions /. {opts} /. Options[ExpectedValue],
	  correlations = AboveDiagonalElements[r], arglist},
    assmp = Flatten[Join[	Map[(#^2 < 1)&, correlations],      	
                                {Det[r] > 0, m > 0},
                                assmp			] /. True -> {}];
    arglist = Prepend[ Map[{#, -Infinity, Infinity}&, xvec],
	 	       f PDF[MultivariateTDistribution[r, m], xvec] ];
    If[assmp =!= {}, AppendTo[arglist, Assumptions -> assmp]];
    Apply[Integrate, arglist]
  ] /; validMultivariateTQ[r, m]&&Length[xvec] <= Length[r] 
	
MultivariateTDistribution/: Random[MultivariateTDistribution[r_, m_]] :=
   Module[{x = Random[MultinormalDistribution[Table[0, {Length[r]}], r]],
	   s = Random[ChiSquareDistribution[m]]},
	x Sqrt[m/s]
   ] (* if l exists, assume valid distribution *)     


(* three parameter multivariate t distribution *)

validMultivariateT2Q[mu_, sigma_, m_, warnflag_: False] := And[
	If[Length[mu]===Length[sigma],
			True,
			If[TrueQ[warnflag],
				Message[MultivariateTDistribution::lsdims, mu, sigma]];
			False],
	If[VectorQ[mu]&&FreeQ[N[mu],Complex],
			True,
			If[TrueQ[warnflag],
				Message[MultivariateTDistribution::realparm, mu]];
			False],
	If[MatrixQ[sigma] && SymmetricMatrixQ[sigma] && FreeQ[N[sigma], Complex] &&
    	If[MatrixQ[N[sigma], NumberQ],
     		TrueQ[Det[N[sigma]] > 0], 
     		True],
   		True,
   		If[TrueQ[warnflag], 
    		Message[MultivariateTDistribution::posdef, sigma]];
   		False],
   	If[If[NumberQ[m], m > 0, True],
   			True,
   			If[TrueQ[warnflag],
				Message[MultivariateTDistribution::posparm, m]];
			False]
	]

MultivariateTDistribution::lsdims="The location parameter `1` and scale matrix `2` have incompatible dimensions."

MultivariateTDistribution /: 
 DistributionDomainQ[
  MultivariateTDistribution[mu_, sigma_, m_], 
  list_?MatrixQ] := 
 MatchQ[Dimensions[list], {_, Length[sigma]}] && 
  FreeQ[N[list], Complex]/;DistributionParameterQ[
  MultivariateTDistribution[mu, sigma, m]]

MultivariateTDistribution /: 
 DistributionDomainQ[
  MultivariateTDistribution[mu_, sigma_, m_], 
  x_?VectorQ] := 
 TrueQ[Length[x] == Length[sigma] && FreeQ[N[x], Complex]]/;DistributionParameterQ[
  MultivariateTDistribution[mu, sigma, m]]


MultivariateTDistribution /: 
 MultivariateSkewness[
  MultivariateTDistribution[mu_, sigma_, m_]] := 
 Piecewise[{{Table[0,{Length[mu]}], m > 3}}, Indeterminate] /; 
  validMultivariateT2Q[mu, sigma, m, True]

MultivariateTDistribution /: 
 MultivariateKurtosis[
  MultivariateTDistribution[mu_, sigma_, m_]] := 
 With[{p = Length[mu]}, p (p + 2) (m - 2)/(m - 4)] /; !TrueQ[IntegerQ[m] && m <= 4] /; 
  validMultivariateT2Q[mu, sigma, m, True]

MultivariateTDistribution/: EllipsoidQuantile[MultivariateTDistribution[
	mu_, sigma_, m_], q_] :=
  Module[{ellipsoid, quantile, p},
    (
    ellipsoid
    ) /; (p = Length[mu];
	  quantile = Quantile[FRatioDistribution[p, m], q];
	  ellipsoid = ellipsoidalLocus[mu, quantile p sigma];
	  If[ellipsoid === $Failed,
		Message[EllipsoidQuantile::mvarteig, quantile p sigma] ];
	  ellipsoid =!= $Failed)
  ] /; QuantileQ[q] && validMultivariateT2Q[mu, sigma, m]
  

MultivariateTDistribution/: EllipsoidProbability[MultivariateTDistribution[
	mu_, sigma_, m_], EllipsoidQuantile[mu1_, radii_]] :=
		0 /; validMultivariateT2Q[mu, sigma, m]&&radii === Table[0, {Length[mu]}]

MultivariateTDistribution/: EllipsoidProbability[MultivariateTDistribution[
	mu_, sigma_, m_], EllipsoidQuantile[mu1_, radii_]] :=
		1 /; validMultivariateT2Q[mu, sigma, m]&&radii === Table[Infinity, {Length[mu]}]
	
MultivariateTDistribution/: EllipsoidProbability[MultivariateTDistribution[
        mu_, sigma_, m_],
        Ellipsoid[mu1_, radii_?VectorQ, dir___?MatrixQ]] :=
   Module[{p, internaldir, normalized, diagonalizedMatrix, acc, prec, delta,
		 scaled, quantile, diagonal},
     (
        CDF[ FRatioDistribution[p, m], quantile ]
     ) /; validMultivariateT2Q[mu, sigma, m] &&
          (p = Length[mu];  TrueQ[mu1 == Table[0, {p}]]) &&
          (  internaldir = If[{dir}==={}, IdentityMatrix[p], dir];
             normalized = sigma.Transpose[internaldir];
	     diagonalizedMatrix = internaldir.normalized;
	     acc = Max[Accuracy[diagonalizedMatrix], $MachinePrecision]-1;
             prec = Max[Precision[diagonalizedMatrix], $MachinePrecision]-1;
             delta = Max[10^(-acc), Abs[Det[diagonalizedMatrix]] 10^(-prec)];	
             diagonalizedMatrix = Simplify[Chop[diagonalizedMatrix, delta]];
             If[ MatchQ[ diagonalizedMatrix, DiagonalMatrix[Table[_, {p}]] ],
		True,
		Message[EllipsoidProbability::mvartell,
			 mu1, radii, internaldir, sigma, m];
		False] &&
          (diagonal = Diagonal[diagonalizedMatrix]/radii^2;
	   acc = Max[Accuracy[diagonal], $MachinePrecision]-3;
           prec = Max[Precision[diagonal], $MachinePrecision]-3;
           delta = Max[10^(-acc), Apply[Times, Abs[diagonal]] 10^(-prec)];
	   Apply[Equal, Chop[diagonal, delta]]) &&
          (  scaled = Transpose[internaldir].DiagonalMatrix[radii^2];
             quantile = Scan[If[ !TrueQ[#[[1]]==0], Return[#[[2]]/#[[1]]/p] ]&,
                       Transpose[{Flatten[normalized], Flatten[scaled]}] ];
             quantile =!= Null)
          )
   ]


MultivariateTDistribution/: ExpectedValue[f_Function,
	 MultivariateTDistribution[mu_, sigma_, m_], opts___?OptionQ] :=
  Module[{n, (* number of arguments of function f *)
	  p = Length[mu], (* dimensionality of MultinormalDistribution *)
	  xvec, x, assmp = Assumptions /. {opts} /. Options[ExpectedValue],
	  arglist, integral,unique},
   (
    xvec = Array[x, p]; 
    assmp = Flatten[Join[{Det[sigma] > 0, m > 0, Element[mu,Reals]},
	 			{assmp}			] /. True -> {}];
    arglist = Prepend[ Map[{#, -Infinity, Infinity}&, xvec],
		       Apply[f, xvec] *
		        PDF[MultivariateTDistribution[mu, sigma, m], xvec] ];
    If[assmp =!= {}, AppendTo[arglist, Assumptions -> assmp]];
    If[FreeQ[integral = Apply[Integrate, arglist], Integrate],
       integral,
       unique = Table[Unique[], {p}];	
       integral /. Thread[Rule[xvec, unique]]		] (* end If *)
   ) /; (
	 n = If[Length[f]==1,
	        (* Function with only a body *)
	        Max[Cases[{f}, Slot[z_]->z, Infinity]],
	        (* Function with a list of formal parameters *)
	        Length[f[[1]]]	];
	 n <= p)&&validMultivariateT2Q[mu, sigma, m]
  ]
	
MultivariateTDistribution/: ExpectedValue[f_,
	MultivariateTDistribution[mu_, sigma_, m_], xvec_?VectorQ,
	opts___?OptionQ] :=			
  Module[{assmp = Assumptions /. {opts} /. Options[ExpectedValue],
	  arglist},
    assmp = Flatten[Join[{Det[sigma] > 0, m > 0, Element[mu,Reals]},
                                {assmp}			] /. True -> {}];
    arglist = Prepend[ Map[{#, -Infinity, Infinity}&, xvec],
	 	       f PDF[MultivariateTDistribution[mu, sigma, m], xvec] ];
    If[assmp =!= {}, AppendTo[arglist, Assumptions -> assmp]];
    Apply[Integrate, arglist]
  ] /; Length[xvec] === Length[mu]&&validMultivariateT2Q[mu, sigma, m]
  
  
(* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *)

 
normal = Compile[{{mu, _Real}, {sigma, _Real}, {q1, _Real}, {q2, _Real}},
		mu + sigma Sqrt[-2 Log[q1]] Cos[2Pi q2]	]

(*
generation of array using listability is as fast as an equivalent compilation
*)

normalarray[mu_, sigma_, dim_] :=
  Module[{n, array},
    n = If[VectorQ[dim], Apply[Times, dim], dim];
    array = mu + sigma Flatten[
    	Block[{mat = Table[Random[], {Quotient[n, 2]}, {2}], q1, q2},
 		{q1, q2} = {mat[[All, 1]], mat[[All, 2]]};
 		Sqrt[-2 Log[q1]]*Transpose[({Cos[2Pi#], Sin[2Pi#]} &)[q2]]]];
    If[OddQ[n],
       AppendTo[array, normal[mu, sigma, Random[], Random[]] ]  ];
    If[VectorQ[dim] && Length[dim] > 1,
       Fold[Partition[#1, #2]&, array, Reverse[Drop[dim, 1]] ],	
       array  ]
  ] /; DistributionParameterQ[NormalDistribution[mu, sigma]]&&VectorQ[{mu,sigma},NumericQ]&&
  	(IntegerQ[dim] && dim > 0) || VectorQ[dim, (IntegerQ[#] && # > 0)&]


(* list of random vectors from a nonsingular multinormal *)
MultinormalDistribution/: RandomArray[
	MultinormalDistribution[mu_, sigma_], dim_] :=
  Module[{m, array, lower},
	m = If[VectorQ[dim], Apply[Times, dim], dim];
	array = normalarray[0, 1, {m, Length[mu]}];
	lower= Transpose[CholDecomp[sigma]];
	array = Map[mu + lower . # &, array];
	If[VectorQ[dim] && Length[dim] > 1,
           Fold[Partition[#1, #2]&, array, Reverse[Drop[dim, 1]] ],
           array  ]
  ] /; nonsingularMultinormalQ[mu, sigma]&&(IntegerQ[dim] && dim > 0) || VectorQ[dim, (IntegerQ[#] && # > 0)&]



(* ====================================================================== *)
End[]


SetAttributes[ WishartDistribution, ReadProtected];    
SetAttributes[ QuadraticFormDistribution, ReadProtected];    
  

(* Protect descriptive statistics symbols *)
Protect[Mean, Variance, StandardDeviation, Skewness, Kurtosis,
         PDF, CDF, ExpectedValue, DistributionParameterQ,
         DistributionDomainQ,DistributionDomain,Covariance,
         Correlation,MultivariateSkewness,MultivariateKurtosis,
         CharacteristicFunction,Quantile,EllipsoidQuantile,
         EllipsoidProbability]

(* Protect distributions. *)
Protect[MultinormalDistribution, WishartDistribution,
	HotellingTSquareDistribution, QuadraticFormDistribution,
	MultivariateTDistribution]


