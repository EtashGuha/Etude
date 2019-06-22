(* ::Package:: *)

(*:Mathematica Version: 6.0 *)

(*:Package Version: 2.0 *)

(*:Context: LinearRegression` *)

(*:Name: Linear Regression Package *)

(*:Title: Linear Regression Analysis *)

(*:Author: Wolfram Research, Inc. *)

(*:Copyright: Copyright 1990-2007, Wolfram Research, Inc. *)

(*:Summary:
This package provides least squares or weighted least squares linear
regression for data whose errors are assumed to be normally and
independently distributed.  It supports a number of commonly used statistics
such as RSquared, EstimatedVariance, and an ANOVATable.  It also provides
diagnostics for leverage, influence, collinearity, and correlation.
*)

(*:History:
  Version 1.0, Wolfram Research, Inc., 1990. 
  Version 1.1, ECM (Wolfram Research), 1994: mostly rewritten,
  Replaced options OutputList, OutputControl, BestFitCoefficients, and
     ConfidenceIntervalTable (and option value NoPrint) with options
     RegressionReport, BestFitParameters, and ParameterCITable (and option
     value SummaryReport);
  Added options BestFitParametersDelta, CatcherMatrix, CoefficientOfVariation,
     CookD, StudentizedResiduals, JackknifedVariance,
     CovarianceMatrixDetRatio, DurbinWatsonD, EigenstructureTable, HatDiagonal,
     MeanPredictionCITable, ParameterConfidenceRegion,
     PartialSumOfSquares, PredictedResponseDelta,
     SequentialSumOfSquares, SinglePredictionCITable, StandardizedResiduals,
     and VarianceInflation.
  Added ability for DesignedRegress to accept either an unweighted design
     matrix or the singular value decomposition of an unweighted design
     matrix.  Added IncludeConstant option to DesignMatrix.
  Added RegressionReportValues so the possible values of the option
     RegressionReport are easily discovered.
  Version 1.2, ECM (Wolfram Research), 1998: modified Weights option so it could
        be set to a pure function of the entire observation vector... not
        just the response.  A pure function of a single argument is still
	applied to the response as before.  Some speedups implemented based
	on suggestions of D. Withoff and experiments by ECM.  Modified
	StandardizedResiduals and StudentizedResiduals to take account of
	Weights.  (Note that the definitions for StandardizedResiduals and
	StudentizedResiduals for the unweighted case follow the definitions
	on p. 157 of "Modern Applied Statistics with S-Plus", Venables and
	Ripley, Springer-Verlag, 1994.  There is not uniform agreement on
	what to call these diagnostics.)
  Version 1.3, Daniel Lichtblau, Serguei Chebalov, and John Novak;
    assorted optimizations and bugfixes, November 1999.
  Version 2.0, Darren Glosemeyer, moved from Statistics`LinearRegression` 
  	to LinearRegression`, 2006.
  Obsoleted in Mathematica version 7.0, Darren Glosemeyer;
  	Functionality is replaced by LinearModelFit and related FittedModel functionality.
*)

(* :Discussion:

(1) Weights -> vector

  When Weights -> {w1, ..., wn}, then the
        weighted residual sum of squares is minimized:
                Sum[wi (yi - g[params, xi1, ..., xik])^2, {i, 1, n}]
        (where g[params, ] represents the linear model).

(2) Weights -> pureFcn

  (i) In Regress[{y1, ..., yn}, basisFcns, x, Weights -> (w[#] &)],
	the weight function is applied to the response y, and the weighted sum 
		Sum[w[yi] (yi - g[params, i])^2, {i, 1, n}]
	is minimized.
      In Regress[{y1, ..., yn}, basisFcns, x, Weights -> (w[#1, #2] &)],
	the weight function is applied to the response y and the implied
	independent variable x, and the weighted sum
		Sum[w[yi, i] (yi - g[params, i])^2, {i, 1, n}]
	is minimized.

  (ii) In Regress[{{x1, y1}, ..., {xn, yn}}, basisFcns, x, Weights -> (w[#] &)],
	 the weight function is applied to the response y, and the weighted sum
		Sum[w[yi] (yi - g[params, xi])^2, {i, 1, n}]
	 is minimized.
       In Regress[{{x1, y1}, ..., {xn, yn}}, basisFcns, x,
	 					Weights -> (w[#1, #2] &)],
	 the weight function is applied to the response y and the explicit
	 independent variable x, and the weighted sum
		Sum[w[yi, xi] (yi - g[params, xi])^2, {i, 1, n}]
	 is minimized.

  (iii) In Regress[{{x11, ..., x1k, y1}, ..., {xn1, ..., xnk, yn}}, basisFcns,
	  				{x1, ..., xk}, Weights -> (w[#] &)],
	  the weight function is applied to the response y, and the weighted sum
                Sum[w[yi] (yi - g[params, xi, ..., xk])^2, {i, 1, n}]
          is minimized.
	In Regress[{{x11, ..., x1k, y1}, ..., {xn1, ..., xnk, yn}}, basisFcns,
                         {x1, ..., xk}, Weights -> (w[#1, ..., # (k+1)] &)],
	  the weight function is applied to the response y and the explicit
          independent variables x1, ..., xk, and the weighted sum
		Sum[w[yi, x1i, ..., xki] (yi - g[params, x1i, ..., xki])^2,
								 {i, 1, n}]
	  is minimized.
	

  Note that while Regress will apply a pure function to the ith observation
	{yi, xi1, ..., xik}, DesignedRegress does not have
	access to the independent variables {xi1, ..., xik}, so it will
	apply the pure function to the ith response yi joined with the ith
	entry in the design matrix:
	{y1, f1[xi1, ..., xik], ..., fp[xi1, ..., xik]}.

  Warning:  Note that while DesignedRegress[mat, response] or
	DesignedRegress[svd, response] is supposed to
	save time because the design matrix or the singular value
	decomposition of the design matrix is already computed (and
	may be used with different response vectors), there is
	a sanity check done in vDR to calculate constantPresent that
	can eliminate the potential time savings.  This sanity check is
	not needed if you use Regress instead of DesignedRegress.
*)

(*:Keywords: linear regression, design matrix  *)

(*:Requirements: No special system requirements. *)

(*:Warning:
Options OutputList, OutputControl, BestFitCoefficients, and
ConfidenceIntervalTable and option value NoPrint are obsolete.

V2.2 syntax:
  Regress[d, f, v, OutputList -> Null, OutputControl -> Automatic]
  Regress[d, f, v, OutputList -> {BestFitCoefficients, ConfidenceIntervalTable},
	 OutputControl -> Automatic]
  Regress[d, f, v, OutputList -> {BestFitCoefficients, ConfidenceIntervalTable},
	 OutputControl -> NoPrint]
V3 .0 syntax:
  Regress[d, f, v, RegressionReport -> SummaryReport]
  Regress[d, f, v, RegressionReport -> {BestFitParameters, ParameterCITable,
	 SummaryReport}]
  Regress[d, f, v, RegressionReport -> {BestFitParameters, ParameterCITable}]
*)

(*:Sources:
F. Mosteller & J. W. Tukey, Data Analysis and Regression, 1977, Addison-Wesley.
P. F. Velleman & R. E. Welsch, Efficient Computing of Regression Diagnostics,
	The American Statistician, 1981, Vol. 35, No. 4.
D. A. Belsey, E. Kuh, & R. E. Welsch, Regression Diagnostics, 1980, Wiley.
Weisberg, S., Applied Linear Regression, 1985, Wiley.
*)

Message[General::"obspkg", "LinearRegression`"]

BeginPackage["LinearRegression`",
			(* needed for confidence intervals and p-values *)
            { "HypothesisTesting`",
	      	(* needed for Weights, RegressionReport,
		 	   RegressionReportValues, and all symbols allowed by
			   RegressionReportValues[Regress] and
		 	   RegressionReportValues[DesignedRegress] *)
	     	"RegressionCommon`",
	     	(*needed for Elliposid in version 7*)
	     	"MultivariateStatistics`"}]

Unprotect[{Regress, DesignedRegress}];

If[ Not@ValueQ[Regress::usage],
Regress::usage =
"Regress[data, funs, vars] finds a least-squares fit to a list of data as \
a linear combination of the functions funs of variables vars. The data \
can have the form {{x1, y1, ..., f1}, {x2, y2, ..., f2}, ...}, where the \
number of coordinates x, y, ... is equal to the number of variables in \
the list vars. The data can also be of the form {f1, f2, ...}, with a \
single coordinate assumed to take values 1, 2, .... The argument funs can \
be any list of functions that depend only on the objects vars. The result \
is a list of rules identifying summary statistics about the fit, or other \
statistics as specified by the option RegressionReport."]

Options[Regress] = {RegressionReport -> SummaryReport,
 IncludeConstant -> True, BasisNames -> Automatic, Weights -> Automatic} ~Join~
   Options[SingularValueDecomposition] ~Join~ Options[StudentTCI]

If[ Not@ValueQ[DesignedRegress::usage],
DesignedRegress::usage =
"DesignedRegress[designmatrix, response] finds a least-squares fit given a \
design matrix and a vector of the response data. The result is a list of \
rules identifying summary statistics about the fit, or other statistics as \
specified by the option RegressionReport."]

Options[DesignedRegress] = {RegressionReport -> SummaryReport,
   BasisNames -> Automatic, Weights -> Automatic} ~Join~
   Options[SingularValueDecomposition] ~Join~ Options[StudentTCI]

RegressionReportValues[DesignedRegress] =
{AdjustedRSquared, ANOVATable, BestFitParameters, 
BestFitParametersDelta, CatcherMatrix, CoefficientOfVariation,
CookD, CorrelationMatrix, CovarianceMatrix,
CovarianceMatrixDetRatio, DurbinWatsonD, EigenstructureTable,
EstimatedVariance, FitResiduals, HatDiagonal, JackknifedVariance,
MeanPredictionCITable, ParameterCITable, ParameterConfidenceRegion,
ParameterTable, PartialSumOfSquares, PredictedResponse, PredictedResponseDelta,
RSquared, SequentialSumOfSquares, SinglePredictionCITable,
StandardizedResiduals, StudentizedResiduals, SummaryReport, VarianceInflation}

RegressionReportValues[Regress] =
	Sort[ RegressionReportValues[DesignedRegress] ~Join~ {BestFit} ]


	(* options unique to Regress and DesignedRegress *)

If[ Not@ValueQ[IncludeConstant::usage],
IncludeConstant::usage =
"IncludeConstant is an option to Regress and DesignMatrix and specifies whether \
a constant term is automatically included in the model. If the set of basis \
functions does not include a constant term and IncludeConstant->True, then a \
constant term is added. If the set of basis functions includes a constant term \
and IncludeConstant -> False, then the set of basis functions is left \
unchanged."]

If[ Not@ValueQ[BasisNames::usage],
BasisNames::usage =
"BasisNames is an option to regression functions and is used to specify \
headings for each of the basis functions (or predictors) in the output."]


Begin["`Private`"]

(* Gives output similar to old SingularValues command;
   matrices U and V are transposed *)
CompactSVD[mat_, opts___] := 
Module[{sv, U, W, V, n},
    (* Compute SVD *)
    sv = SingularValueDecomposition[mat, Min[Dimensions[mat]], opts];
    If[!ListQ[sv], Return[$Failed]];
    svdtocompact[sv]
]

svdtocompact[{U_, W_, V_}] :=
Module[{sv},
    (*extract the diagonal vector*)
    sv = Tr[W, List];
    (* determine the number of positive singular values *)
    n = Length[sv] - Count[sv, _?(#==0&)];
    If [n == 0,
        {U, W, V},
        {Take[U, All, n], Take[sv, n], Take[V, All, n]}
    ]
]

SVDRank[w_List] := Length[w];



(* We'll normalize columns before doing an SVD on the matrix in order to
improve precision. Then we use the same column weights to recover the
correct least-squares solution vector for the unnormalized matrix.
9/99 DANL *)

normalizeColumns[mat_] := Module[{weights, tmp},
    weights = Table[tmp = Abs[Part[mat, All, i]];
        Sqrt[tmp.tmp], {i, Length[First[mat]]}];
    weights = Map[If[# == 0, N[0, Precision[mat]], 1/#] &, weights];
     {Developer`ToPackedArray[weights], weights*# & /@ mat}]

(* ============================== Regress ================================= *)

Regress[data_, basis_, vars_, options___Rule] :=
  Block[{answer, basislist, varlist, varcount, coordcount},
    answer /; (varlist = If[VectorQ[vars], vars, {vars}];
           basislist = If[VectorQ[basis], basis, {basis}];
	       varcount = Length[varlist];
	       answer = Which[
		  Apply[Or, Map[NumberQ, N[varlist]]],
		     Message[Regress::crdvl, varlist];
		     $Failed,	
 	          MatrixQ[data], (* data is a matrix *)
		     If[varcount <= (coordcount = Length[data[[1]]]-1),
		        vRegress[data, basislist, varlist, options],
		        Message[Regress::coord, coordcount, varcount];
		        $Failed],
		  True, (* data is a vector *)
		     If[varcount === 1,
		        vRegress[Transpose[{Range[Length[data]],data}],
				basislist, varlist, options],
		        Message[Regress::coord, 1, varcount];
		        $Failed]
	       ];
	       answer =!= $Failed) 	
  ] /; If[VectorQ[N[data],NumericQ] || MatrixQ[N[data],NumericQ],
          True, Message[Regress::notdata]; False] &&
       If[FreeQ[basis, List] || VectorQ[basis],
          True, Message[Regress::notfcn]; False] &&
       If[(FreeQ[vars, List] || VectorQ[vars]) && Head[vars] =!= Rule,
          True, Message[Regress::notvar]; False]

vRegress[data_, basis_, varlist_, options___] :=
    Module[{response = Map[Last, data], cbasis = basis,
	    tempdata, tempweights,  
	    designMatrix, optionlist = {options},
	    dmPrecision, nResponse, nDesignMatrix,
	    report, includeC, names, weights, tol, cLevel, outlist, control, one},
	(* establish the values of these options according to the defaults
		of Regress *)
	{includeC, names, weights, tol, cLevel} =
		{IncludeConstant, BasisNames, Weights,
		 Tolerance, ConfidenceLevel} /. optionlist /. Options[Regress];
	
	dmPrecision = Precision[{data, cbasis}];
	one = N[1, If[NumberQ[dmPrecision], dmPrecision, MachinePrecision]];	
 
	If[!(Or @@ Apply[And, Outer[FreeQ, basis, varlist], {1}]) && includeC,
            PrependTo[cbasis, one];
            If[names===Automatic, names=ReplacePart[cbasis,1,1]]
        ];

	(* check and set Weights *)
        (* This must be done before calling vDesignedRegress, because
		if Weights -> pureFcn, then the Weights setting must be
		converted to Weight -> vector while the data is still 
		available.  vDesignedRegress can't do this conversion,
		because it has access to the design matrix,
		not the raw data. *)
	Which[
         weights === Equal || weights === Automatic,
                weights = Table[one, {Length[response]}],
         Head[weights] === Function,
		(* Rearrange data so that response is first... if weights
			has only one arg, then it is applied to the response.
		   Note that as of May 1998, the ability to use a weights
			function of more than one arg is not documented. *)
		(*tempdata = Map[Join[{Last[#]}, Drop[#, -1]]&, data];*)
		tempdata = Map[RotateRight, data];
        tempweights = Map[Apply[weights, #]&, tempdata];
        If[!VectorQ[N[tempweights], NumberQ],
        	Message[Regress::bdwghts, weights];
            weights = Table[one, {Length[response]}],
            weights = tempweights
            ],
         VectorQ[weights],
                If[!VectorQ[N[weights], NumberQ] ||
                   Length[weights] =!= Length[response],
                        Message[Regress::bdwghts, weights];
                        weights = Table[one, {Length[response]}]
                ],
         True,
                Message[Regress::bdwghts, weights];
                weights = Table[one, {Length[response]}]
        ];
	(* calculate designMatrix *)
	If[!FreeQ[designMatrix =
	   DesignMatrix[data, cbasis, varlist, IncludeConstantBasis -> includeC],
	   DesignMatrix],
	   Message[Regress::nodm];
	   Return[$Failed]];
	If[Not[MatrixQ[N[designMatrix],NumericQ]],
		Message[Regress::invdm];
		Return[$Failed]];
	(* if designMatrix is exact and response is neither exact nor MachinePrecision,
	   designMatrix must be coerced to the appropriate precision *)
	designMatrix=N[designMatrix, dmPrecision];
	If[Length[response] <= Length[cbasis],
            Message[Regress::mindata]
        ];
	If[names === Automatic, 
		names = cbasis,
              (* else names provided *)
		If[!VectorQ[names],
		   Message[Regress::invnam]; names = cbasis,
		   If[Length[names] =!= Length[cbasis],
		      Message[Regress::namnum]; names = cbasis
		   ]
		] 
	];


	dmPrecision = Precision[designMatrix];
	(* make nDesignMatrix suitable for SingularValueDecomposition *)
	{nDesignMatrix, nResponse} = If[dmPrecision === Infinity,
                      N[{designMatrix, response}],
			{designMatrix, response} ];

	(* make sure that the default options of Regress are used instead of
		the default options of DesignedRegress *)report = RegressionReport /. optionlist /. Options[Regress];
           vDesignedRegress[nDesignMatrix, nResponse, cbasis,
		RegressionReport -> report, IncludeConstant -> includeC,
		BasisNames -> names, Weights -> weights, Tolerance -> tol,
		ConfidenceLevel -> cLevel]

    ] (* end vRegress *)


(* ================= DesignedRegress[designMatrix, ...] =================== *)

DesignedRegress[designMatrix_?MatrixQ, response_?VectorQ, options___Rule] :=
  Block[{answer, dmPrecision, nDesignMatrix, nResponse, optionlist,
	 report, pos}, (* ttt does not belong here *)
    (
    answer
    ) /; (dmPrecision = Precision[{designMatrix, response}];
	  (* make nDesignMatrix suitable for SingularValueDecomposition *)
	  {nDesignMatrix, nResponse} = If[dmPrecision === Infinity,
          	      N[{designMatrix, response}],
          	      {designMatrix, response} ];
	  optionlist = {options};
	  If[MemberQ[optionlist, IncludeConstant->_],
	     (* IncludeConstant should be an option of DesignMatrix,
			not DesignedRegress. *)
	     Message[DesignedRegress::invopt, IncludeConstant];
	     optionlist = Delete[optionlist, Position[optionlist,
		IncludeConstant->_]]	
	  ];
	  (* filter out invalid report items *)
          report = RegressionReport /. optionlist /. Options[DesignedRegress];
	  If[ListQ[report],
	     If[MemberQ[report, BestFit],
		(* we can't return a symbolic expression when the input is
			entirely numeric *)
		Message[DesignedRegress::invrep, BestFit];
		report = Delete[report, Position[report, BestFit]]
  	     ];
	     pos = Position[optionlist, RegressionReport -> _];
	     optionlist = Delete[optionlist, pos];	
	     optionlist = Insert[optionlist, RegressionReport->report, pos]
	  ]; 
	  (* ttt array is just a place holder *)
	  answer = vDesignedRegress[nDesignMatrix, nResponse,
		 Array[ttt, Length[nDesignMatrix[[1]]]],
		 Apply[Sequence, optionlist] ];
	  answer =!= $Failed)	
  ] /; If[MatrixQ[N[designMatrix], NumberQ],
	  True,
	  Message[DesignedRegress::invdm]; False] &&
       If[Length[response] == Length[designMatrix],
	  True,
	  Message[DesignedRegress::invresp, Length[response],
		 Length[designMatrix]];
	  False]&&
	  If[VectorQ[N[response],NumericQ],
	  	True,
	  	Message[DesignedRegress::numrsp];False] (* end of DesignedRegress[designMatrix, etc.] *)


vDesignedRegress[designMatrix_?MatrixQ, response_,
	 basis_, (* this is either a valid list of basis functions
			or a place holder {ttt[1], ttt[2], ...} *)
	 options___] :=
    Module[{temp, constantQ, n, p, dmPrecision,
	    optionlist = {options}, tol, weights,
	    tempdata, tempweights, nsqrtweights, weightedDesignMatrix,
		colweights, reweightedmat, svd, u, w, v},
	If[Not[VectorQ[N[response],NumericQ]],
	  	Message[DesignedRegress::numrsp];Return[$Failed]];
	If[FreeQ[basis, ttt],
           (* basis list is not just a place-holder *)	
	   With[{temp = Apply[Plus, Map[If[NumberQ[#], 1, 0]&, basis]]},
		Which[temp == 1,
		      constantQ = True,
		      temp == 0,
		      constantQ = False,
		      temp > 1,
		      Message[DesignedRegress::twocon];
		      Return[$Failed],
		      True,
		      Message[DesignedRegress::uncon];
		      Return[$Failed]
		]	
	   ],
	   (* It is unknown whether constantQ = True or False, that is
	      whether there is a single constant basis fcn represented in the
	      design matrix or not. *)
	   constantQ = UNKNOWN];

	{n, p} = Dimensions[designMatrix];

        {tol, weights} = {Tolerance, Weights} /.
		 optionlist /. Options[DesignedRegress];
	(* set and check weights *)
	Which[
         weights === Equal || weights === Automatic,
                weights = Table[1, {Length[response]}],
         Head[weights] === Function,
		(* Rearrange data so that response is first... if weights
                        has only one arg, then it is applied to the response.
                   Note that as of May 1998, the ability to use a weights
                        function of more than one arg is not documented. *)
		(* vDesignedRegress has no access to the independent
		   variables, so a pure function for computing
		   weights must be applied to the response vector. *)
		tempdata = Map[Prepend[designMatrix[[#]], response[[#]]]&,
                                Range[n]];
                tempweights = Map[Apply[weights, #]&, tempdata];
                If[!VectorQ[N[tempweights], NumberQ],
                        Message[DesignedRegress::bdwghts, weights];
                        weights = Table[1, {Length[response]}],
                        weights = tempweights
                ],
         VectorQ[weights],
                If[!VectorQ[N[weights], NumberQ] ||
                   Length[weights] =!= Length[response],
                        Message[DesignedRegress::bdwghts, weights];
                        weights = Table[1, {Length[response]}]
                ],
         True,
                Message[DesignedRegress::bdwghts, weights];
                weights = Table[1, {Length[response]}]
        ];
	dmPrecision = Precision[designMatrix];
	If[Precision[weights] === Infinity,
		 weights = N[weights, If[NumberQ[dmPrecision], dmPrecision, MachinePrecision]]
	];
	sqrtweights = Developer`ToPackedArray[Sqrt[weights]];
	weightedDesignMatrix = sqrtweights*designMatrix;
	
	{colweights, reweightedmat} =
	  normalizeColumns[weightedDesignMatrix];
        svd = CompactSVD[reweightedmat, Tolerance->tol];
        If[Head[svd]===List && Length[svd]===3,
            {u, w, v} = svd; u = Transpose[u]; v = Transpose[v],
            Message[DesignedRegress::nosvd];
            Return[$Failed]
        ];
        If[SVDRank[w] < p, Message[DesignedRegress::rank,
		 SVDRank[w], p, If[SVDRank[w]==1, "is", "are"] ] ]; 
	vDR[weightedDesignMatrix, {u, w, v}, colweights,
	  sqrtweights*response, basis, sqrtweights, constantQ, options]
    ] (* end of vDesignedRegress[designMatrix, etc] *)


(* =================== DesignedRegress[{u, w, v}, ...] ===================== *)

(* to work with SingularValueDecomposition *)
DesignedRegress[svd:{u_?MatrixQ, w_?MatrixQ, v_?MatrixQ}, response_?VectorQ,
	 options___Rule] := 
     DesignedRegress[{Transpose[#1], #2, #3}& @@ svdtocompact[svd],
                     response, options];
         

DesignedRegress[{u_?MatrixQ, w_?VectorQ, v_?MatrixQ}, response_?VectorQ,
	 options___Rule] :=
  Block[{answer, optionlist, report, pos}, (* ttt does not belong here *)
    (
    answer
    ) /; (optionlist = {options};
	  If[MemberQ[optionlist, IncludeConstant->_],
	     (* IncludeConstant should be an option of DesignMatrix,
			not DesignedRegress. *)
	     Message[DesignedRegress::invopt, IncludeConstant];
	     optionlist = Delete[optionlist, Position[optionlist,
		IncludeConstant->_]]	
	  ];
	  (* filter out invalid report items *)
          report = RegressionReport /. optionlist /. Options[DesignedRegress];
	  If[ListQ[report],
	     If[MemberQ[report, BestFit],
		(* we can't return a symbolic expression when the input is
			entirely numeric *)
		Message[DesignedRegress::invrep, BestFit];
		report = Delete[report, Position[report, BestFit]]
  	     ];
	     pos = Position[optionlist, RegressionReport -> _];
	     optionlist = Delete[optionlist, pos];	
	     optionlist = Insert[optionlist, RegressionReport->report, pos]
	  ]; 
	  (* ttt matrix array is just a place holder *)
	  answer = vDesignedRegress[{u, w, v}, response, Array[ttt, Length[w]],
		 Apply[Sequence, optionlist] ];
	  answer =!= $Failed)	
  ] /; If[MatrixQ[u, NumberQ] && VectorQ[w, NumberQ] && MatrixQ[v, NumberQ],
	  True,
	  Message[DesignedRegress::invsvd]; False] &&
       If[Length[response] == Length[u[[1]]],
	  True,
	  Message[DesignedRegress::invresp, Length[response], Length[u[[1]]]];
	  False] (* end of DesignedRegress[{u, w, v}, etc.] *)
	

vDesignedRegress[{u_?MatrixQ, w_?VectorQ, v_?MatrixQ}, response_,
	 basis_, (* this is a place holder {ttt[1], ttt[2], ...} *)
	 options___] :=
	If[(* check response first to avoid computing designMatrix if 
	      the response is non-numeric *)
		Not[VectorQ[N[response],NumericQ]],
	  	Message[DesignedRegress::numrsp];Return[$Failed],
    Module[{n, p, optionlist = {options}, tol, weights, tempdata, tempweights,
		wPrecision,  sqrtweights, colweights, reweightedmat,
	  	designMatrix = Transpose[u] . (v w),
	  	weightedDesignMatrix, svd, uprime, wprime, vprime},
	{n, p} = Dimensions[designMatrix];

        {tol, weights} = {Tolerance, Weights} /.
		 optionlist /. Options[DesignedRegress];
	(* set and check weights *)
        Which[
         weights === Equal || weights === Automatic,
                weights = Table[1, {Length[response]}],
         Head[weights] === Function,
                (* Rearrange data so that response is first... if weights
                        has only one arg, then it is applied to the response.
                   Note that as of May 1998, the ability to use a weights
                        function of more than one arg is not documented. *)
                (* vDesignedRegress has no access to the independent
                   variables, so a pure function for computing
                   weights must be applied to the response vector. *)
                tempdata = Map[Prepend[designMatrix[[#]], response[[#]]]&,
                                Range[n]];
                tempweights = Map[Apply[weights, #]&, tempdata];
                If[!VectorQ[N[tempweights], NumberQ],
                        Message[DesignedRegress::bdwghts, weights];
                        weights = Table[1, {Length[response]}],
                        weights = tempweights
                ],
         VectorQ[weights],
                If[!VectorQ[N[weights], NumberQ] ||
                   Length[weights] =!= Length[response],
                        Message[DesignedRegress::bdwghts, weights];
                        weights = Table[1, {Length[response]}]
                ],
         True,
                Message[DesignedRegress::bdwghts, weights];
                weights = Table[1, {Length[response]}]
        ];
	wPrecision = Precision[w];
	If[Precision[weights] === Infinity,
		 weights = N[weights, If[NumberQ[wPrecision], wPrecision, MachinePrecision]]
	];
	sqrtweights = Developer`ToPackedArray[Sqrt[weights]];
	weightedDesignMatrix = sqrtweights*designMatrix;
	

	(* Here it is unknown whether constantQ = True or False, that is
	   whether there is a single constant basis fcn represented in the
	   design matrix or not.  The problem with this is that vDR
	   will check the designMatrix and assign a value to constantQ,
	   and this can be time-consuming.  Perhaps all the time saved
	   in reusing the singular value decomposition of the design matrix
	   is lost in making this sanity check. *)
	constantQ = UNKNOWN;
	If [TrueQ[Apply[Equal, sqrtweights]],
	   (* use existing svd *)
	   vDR[weightedDesignMatrix, {u, (sqrtweights[[1]])^2*w, v},
	     Table[1,{Length[w]}], sqrtweights*response, basis, sqrtweights,
		 constantQ, options],
	   (* else compute new svd *)
		Message[DesignedRegress::newsvd];
		{colweights, reweightedmat} =
		  normalizeColumns[weightedDesignMatrix];
		svd = CompactSVD[reweightedmat, Tolerance->tol];
		If [Head[svd]===List && Length[svd]===3,
			{uprime, wprime, vprime} = svd;
                        uprime = Transpose[uprime];
                        vprime = Transpose[vprime];
                        ,
			Message[DesignedRegress::nosvd];
			Return[$Failed]
           ];
		If[SVDRank[wprime] < p,
			Message[DesignedRegress::rank, SVDRank[wprime], p,
			  If[SVDRank[w]==1, "is", "are"] ]]; 
		vDR[weightedDesignMatrix, {uprime, wprime, vprime}, colweights,
		  sqrtweights*response, basis, sqrtweights, constantQ, options]
		]]
    ] (* end of vDesignedRegress[{u, w, v}, etc] *)


(* ================================ vDR ================================= *)

vDR[weightedDesignMatrix_, {uu_, ww_, vv_}, colweights_, weightedResponse_,
  basis_, sqrtweights_, constantQ_, options___] :=
    (* weightedDesignMatrix:   weighted design matrix for regression         *)
    (* {u, w, v}:      singular value decomposition of weightedDesignMatrix *)
    (* weightedResponse:       vector of response values                     *)
    (* basis:								     *)
    (*   symbolic basis functions if vDesignedRegress is called by vRegress,  *)
    (*   place-holding array if vDesignedRegress is called by DesignedRegress *)
    (* optionlist:     list of options                                    *)
    Module[{n, p, optionlist = {options}, constantPresent = constantQ,
	  meanWeightedResponse,
          modelDOF, totalDOF, errorDOF, modelSS, totalSS, errorSS,
          pValue, fRatio,
	  weights = sqrtweights^2,
	  designMatrix = weightedDesignMatrix/sqrtweights,
	  response = weightedResponse/sqrtweights,
          tol,			(* tolerance for SingularValueDecomposition *)
	  names, 		(* basis fcn names for table headings *)
	  catcherMatrix, 	(* terminology of Mosteller & Tukey *)
          fit, 			(* best fit parameters *)
          responseMean, predictedResponse, fitResiduals,
	  rSquared, adjRSquared, coeffvar,
          totalResiduals, 	(* residuals from fitting a constant *)
          responseVariance, 	(* estimated variance from best fit *)
          report, 		(* output accumulation *)
          designMatrixT, headings, cLevel, singlepredci, meanpredci, varinf,
	  sosDiagnostics,	(* flag indicating whether design matrix is
				   suitable for computing PartialSumOfSquares
				   or SequentialSumOfSquares *)
      startreport, finishreport  (* constructs to prevent unpacking during
                                    replacement of report parts *)
	 },
	{n, p} = Dimensions[designMatrix];

     (* READ IN OPTIONS *)

       	optionlist = {options};
        {tol, names, cLevel} =
		 {Tolerance, BasisNames, ConfidenceLevel} /.
		 optionlist /. Options[DesignedRegress];
 

    (*  COMPUTE PARAMETERS *) 

	(* If X = design matrix and Y = response, then
		X = Transpose[uu].DiagonalMatrix[ww].vv      
		fit = Inverse[Transpose[X].X].Transpose[X].Y
		    = Transpose[vv].DiagonalMatrix[1/ww].uu.Y
		    = catcherMatrix.Y = Transpose[vv].((uu.Y)/ww)		*)

	(* NOTE: The following manner of calculating catcherMatrix unnecessarily
		constructs DiagonalMatrix[sqrtweights] (an n x n matrix) and is
		slow.
	   catcherMatrix =
	     Transpose[vv].DiagonalMatrix[1/ww].uu.DiagonalMatrix[sqrtweights]; *)
(* Since this is a suboptimal way to do the computation we'll rearrange
the furniture. We compute fit one way that avoids catcherMatrix, then
compute catcherMatrix separately. Moreover, we only find the latter if
it is needed. 9/99 DANL *)

		fit = colweights *
		  (Transpose[vv] . ((1/ww) * (uu . (sqrtweights * response))));

        predictedResponse = designMatrix . fit;
        fitResiduals = response - predictedResponse;
        errorDOF = n - p;
        errorSS = weights . (fitResiduals^2);

	designMatrixT = Transpose[designMatrix];

     (*  COMPARING MODELS  *)

	If[!(constantPresent === True || constantPresent === False),
	   (* Examine design matrix to determine whether the model included
		a single constant basis function.  This is time-consuming
		for a large sample. *)
	   With[{temp = Tr[
			 Map[If[Apply[Equal, #], 1, 0]&, designMatrixT]]},
		Which[temp == 1,
		      constantPresent = True,
		      temp == 0,
		      constantPresent = False,
		      temp > 1,
		      Message[DesignedRegress::twocon];
		      Return[$Failed],
		      True,
		      Message[DesignedRegress::uncon];
		      Return[$Failed]
		]	
	   ]
	];
	
        If[constantPresent,
	   meanWeightedResponse =
	      sqrtweights (weightedResponse.sqrtweights)/Tr[weights];
	   totalResiduals = weightedResponse - meanWeightedResponse;
           totalDOF = n - 1,
           (* else if !constantPresent *)
           totalResiduals = weightedResponse;
           totalDOF = n
        ];
        totalSS = totalResiduals . totalResiduals;
        modelDOF = totalDOF - errorDOF;
        modelSS = totalSS - errorSS;
	If[modelSS < 0||modelSS===Indeterminate,
	   Message[DesignedRegress::badfit];
	   rSquared = $Failed;  adjRSquared = $Failed,
	   rSquared = modelSS/totalSS;
	   adjRSquared = 1-(1-rSquared)(totalDOF/errorDOF)];

        responseVariance = errorSS/errorDOF;


     (*  REPORT GENERATION *)

        report = GetOutputList[optionlist];

      (* following split is to prevent unpacking in generated report
         objects when subjected to MemberQ test *)
       (* To check only names in MemberQ *)
        startreport = report;
       (* To add results without replacement *)
        finishreport = report;

	(* ================= compute ANOVA table ================ *)

 	If [MemberQ[startreport, ANOVATable],
	   If[modelSS >= 0,
	      If[!constantPresent,
	         (* Use message to remind user of the changed interpretation 
		    of total sum of squares when there is no constant in
		    the model. *)
		 Message[DesignedRegress::tsos];
	         (* Use "U Total" sum of squares heading to remind user of 
		    the changed interpretation when there is no constant term
		    in the model. *)
		 headings = {{"Model", "Error", "U Total"},
			{"DF", "SumOfSq", "MeanSq", "FRatio", "PValue"}},
		 headings = {{"Model", "Error", "Total"},
			{"DF", "SumOfSq", "MeanSq", "FRatio", "PValue"}}
	      ];
              fRatio = (modelSS/modelDOF)/responseVariance;
              pValue = 1 - CDF[FRatioDistribution[modelDOF, errorDOF], fRatio];
              (* in case of complete cancellation, recompute the pvalue using significance arithmetic *)
              If[Precision[fRatio]===MachinePrecision&&PossibleZeroQ[pValue], 
              	pValue=N[1 - CDF[FRatioDistribution[modelDOF, errorDOF], SetPrecision[fRatio,16]]]];
              finishreport = Replace[finishreport, ANOVATable -> (ANOVATable -> TableForm[
              	{{modelDOF, modelSS, modelSS/modelDOF, fRatio, pValue},
               	{errorDOF, errorSS, errorSS/errorDOF},
               	{totalDOF, totalSS}}, TableHeadings->headings ]), 1],
	      (* modelSS < 0 *)
	      If[!constantPresent,
	         Message[DesignedRegress::tsos];
		 headings = {{"Error", "U Total"}, {"DF", "SumOfSq", "MeanSq"}},
		 headings = {{"Error", "Total"}, {"DF", "SumOfSq", "MeanSq"}}
	      ]; 
	      finishreport = Replace[finishreport, ANOVATable -> (ANOVATable -> TableForm[
		{{errorDOF, errorSS, errorSS/errorDOF},
		 {totalDOF, totalSS}}, TableHeadings->headings ]), 1]	
	   ] (* end If modelSS >= 0 *)
        ]; 

        (* ========= compute things involving the covariance matrix ======== *)

       If [Or @@ Map[MemberQ[startreport, #]&, {CovarianceMatrix,
             CorrelationMatrix, MeanPredictionCITable,
	     ParameterTable, ParameterCITable,
	     ParameterConfidenceRegion, ParameterConfidenceRegion[_?VectorQ],
	     PartialSumOfSquares, SinglePredictionCITable}],
           Block[{unscaledcovMatrix, covMatrix, corMatrix, parmci, predci,
		   infy, indet, cvt,
                   parmse, (* list of standard errors of parm estimates *)
                   predse, (* list of standard errors of predicted response *)
		   tRatio, tPValues},
	    (* expression for covariance matrix does not depend on whether the
		regression is weighted or not, except that v and w are from
		the svd of a weighted design matrix *)
(* Even the columns can be weighted, but then you need to handle those
multipliers as well. The gist is we want to get
Inverse[Transpose[weightedDesignMatrix].weightedDesignMatrix]. 9/99 DANL *)

		cvt = colweights * Transpose[vv];
		unscaledcovMatrix = cvt . (Transpose[cvt] / ww^2);

	    covMatrix = responseVariance*unscaledcovMatrix;
	    finishreport = Replace[finishreport, CovarianceMatrix ->
                   (CovarianceMatrix -> MatrixForm[covMatrix]), 1];
            parmse = Sqrt[Diagonal[covMatrix]];
	    If[MemberQ[startreport, CorrelationMatrix],
               infy = (Head[Power::infy] === $Off);  Off[Power::infy];	
	       indet = (Head[Infinity::indet] === $Off);  Off[Infinity::indet]; 
               corMatrix = Transpose[Transpose[covMatrix/parmse] / parmse];
	       If[!infy, On[Power::infy]];  If[!indet, On[Infinity::indet]];
 	       finishreport = Replace[finishreport, CorrelationMatrix ->
                   (CorrelationMatrix -> MatrixForm[corMatrix]), 1];
	    ]; 
	    If[MemberQ[startreport, ParameterTable],
	        infy = (Head[Power::infy] === $Off);  Off[Power::infy];
	        indet = (Head[Infinity::indet] === $Off);  Off[Infinity::indet];
            	tRatio = fit / parmse; 
	        If[!infy, On[Power::infy]];  If[!indet, On[Infinity::indet]];
		tPValues = Map[If[# === Indeterminate, Indeterminate,
			CDF[StudentTDistribution[errorDOF],-Abs[#]]]&, tRatio];
		tPValues = Map[If[# === Indeterminate, Indeterminate,
				  2 #]&, tPValues];
	        (* Note: when names is not defined, the default row headings
			are used (i.e., 1, 2, 3, ....) *)
	    	finishreport = Replace[finishreport, ParameterTable ->
             		(ParameterTable -> TableForm[
			   Transpose[{fit, parmse, tRatio, tPValues}],
              		TableHeadings ->
			 {names, {"Estimate", "SE", "TStat", "PValue"}}]), 1]
	    ];
	    If[MemberQ[startreport, PartialSumOfSquares],
 	       sosDiagnostics = p > 1 &&
			TrueQ[equalvectorQ[First[designMatrixT]] &&
              Not[Apply[Or, 
                  Table[equalvectorQ[designMatrixT[[n]]], {n, 2, 
                        Length[designMatrixT]}]
              ]]
            ];
	       If[sosDiagnostics,
		        partialsos =
			 Rest[fit^2/Diagonal[unscaledcovMatrix]];
			finishreport = Replace[finishreport, PartialSumOfSquares ->
			  (PartialSumOfSquares -> partialsos), 1],
			Message[DesignedRegress::parsos];
			finishreport = Replace[finishreport, {PartialSumOfSquares ->
                                (PartialSumOfSquares -> $Failed)}, 1]	]
  	    ];
	    If[MemberQ[startreport, ParameterCITable],
            	parmci = Apply[N[StudentTCI[#1, #2, errorDOF,
                                   ConfidenceLevel -> cLevel]]&,
                           Transpose[{fit, parmse}], 1];        
                finishreport = Replace[finishreport, ParameterCITable ->
             		(ParameterCITable -> TableForm[
				Transpose[{fit, parmse, parmci}],
			 TableDepth -> 2,
	      		TableHeadings -> {names, {"Estimate", "SE", "CI"}}]), 1]
	    ];
	    If[MemberQ[startreport, MeanPredictionCITable] ||
	       MemberQ[startreport, SinglePredictionCITable],
	       	predse = Sqrt[Map[(#.covMatrix.#)&, designMatrix]] ];	
            If[MemberQ[startreport, MeanPredictionCITable],
	        predci = Apply[N[StudentTCI[#1, #2, n-p,
				ConfidenceLevel -> cLevel]]&,
				Transpose[{predictedResponse, predse}], 1];
	        finishreport = Replace[finishreport, MeanPredictionCITable ->
	     	   (MeanPredictionCITable -> TableForm[
		    Transpose[{response, predictedResponse, predse, predci}],
			TableDepth -> 2, TableHeadings ->
		 	{None, {"Observed", "Predicted", "SE", "CI"}}]), 1]
            ];
            If[MemberQ[startreport, SinglePredictionCITable],
		predse = Sqrt[responseVariance + predse^2];
	        predci = Apply[N[StudentTCI[#1, #2, n-p,
				ConfidenceLevel -> cLevel]]&,
				Transpose[{predictedResponse, predse}], 1];
	        finishreport = Replace[finishreport, SinglePredictionCITable ->
	     	   (SinglePredictionCITable -> TableForm[
		    Transpose[{response, predictedResponse, predse, predci}],
			TableDepth -> 2, TableHeadings ->
		 	{None, {"Observed", "Predicted", "SE", "CI"}}]), 1]
            ];
	    If[MemberQ[startreport, ParameterConfidenceRegion] ||
	       MemberQ[startreport, ParameterConfidenceRegion[_?VectorQ]],
               (* Weisberg, p. 97 *)
	       Block[{ellipsoid, arglist, pos, 
		      basisSubset, fitSubset, covMatrixSubset, pSubset},
	          If[MemberQ[startreport, ParameterConfidenceRegion],
		     ellipsoid = ellipsoidalLocus[fit, covMatrix * p *
			Quantile[FRatioDistribution[p, n-p], cLevel]];
	             finishreport = Replace[finishreport, ParameterConfidenceRegion ->
                  	(ParameterConfidenceRegion -> ellipsoid), 1]
	          ];
	          If[MemberQ[startreport, ParameterConfidenceRegion[_?VectorQ]],
 		     arglist = Map[First, Select[startreport,
			 MatchQ[#, ParameterConfidenceRegion[l_List] /;
			 	 Apply[And, Map[MemberQ[basis, #]&, l]]]& ]];
	             Scan[(basisSubset = #;
		           pos = Flatten[Map[Position[basis, #, 1]&,
				 basisSubset]];
		   	   fitSubset = fit[[pos]];
		           covMatrixSubset = covMatrix[[pos, pos]];
(*				 Transpose[Transpose[covMatrix[[pos]]][[pos]]];*)
		   	   pSubset = Length[basisSubset];
		   	   ellipsoid = ellipsoidalLocus[fitSubset,
			     covMatrixSubset * pSubset * Quantile[
			     FRatioDistribution[pSubset, n-p], cLevel]];
		   	   finishreport = Replace[finishreport,
				 (ParameterConfidenceRegion[basisSubset] ->
		     	         (ParameterConfidenceRegion[basisSubset] ->
				 ellipsoid)), 1]
		          )&, arglist]
	          ]  (* end If *)
	       ] (* end Block *)
	    ] (* end If *)


	   ]  (* end Block *)
        ];	(* end of covariance matrix calculations *)

(* See if we need to compute catcherMatrix. 9/99 DANL *)
		If [Or @@ Map[MemberQ[startreport,#]&, {CatcherMatrix, VarianceInflation,
		  HatDiagonal, StandardizedResiduals, StudentizedResiduals,
		  PredictedResponseDelta, CookD, CovarianceMatrixDetRatio,
		  BestFitParametersDelta, JackknifedVariance}],
			catcherMatrix = colweights *
			  (Transpose[vv] . (Map[(sqrtweights*#)&, uu] / ww));
			];


        (* =========== compute things involving the hat matrix ========== *)

        If [Or @@ Map[MemberQ[startreport, #]&, {HatDiagonal,
		StandardizedResiduals, StudentizedResiduals,
		PredictedResponseDelta, CookD, CovarianceMatrixDetRatio,
		BestFitParametersDelta, JackknifedVariance}],
	   Block[{hatdiag, standResiduals, cookD, jackvar, studResiduals,
	          dfit, covratio, parmInfo, obsrInfo, dbeta, catcherMatrixT},

	   (*  hatMatrix = designMatrix . catcherMatrix;
	     hatdiag = Diagonal[hatMatrix]; *)

		 catcherMatrixT = Transpose[catcherMatrix];
		 hatdiag = Table[designMatrix[[j]].catcherMatrixT[[j]],
		   {j,Length[designMatrix]}];

	     (* 8/98: changed standResiduals from
		fitResiduals/Sqrt[responseVariance (1-hatdiag)]
		to
		fitResiduals/Sqrt[responseVariance (1-hatdiag)/weights]
		*)
	     standResiduals = fitResiduals/
		Sqrt[responseVariance (1-hatdiag)/weights];
	     cookD = hatdiag/(1 - hatdiag) standResiduals^2 / p;
	     (* Exhibit 2B .1, p. 75, Belsley, Kuh, Welsch *)
	     jackvar = ((n-p) responseVariance -
		fitResiduals^2/(1 - hatdiag)) / (n-p-1);
	     (* 8/98: changed studResiduals from
	        studResiduals = fitResiduals/Sqrt[jackvar (1-hatdiag)];
		to
	        studResiduals = fitResiduals/Sqrt[jackvar (1-hatdiag)/weights];
		*)
	     studResiduals = fitResiduals/Sqrt[jackvar (1-hatdiag)/weights];
	     dfit = Sqrt[hatdiag/(1 - hatdiag)] studResiduals;
	     covratio = (n-p)^p/( (n-p-1+studResiduals^2)^p (1-hatdiag) ); 	
	     If [MemberQ[startreport, BestFitParametersDelta],
		parmInfo = 1/Sqrt[Map[(Tr[#^2])&, catcherMatrix]];
		obsrInfo = dfit / Sqrt[hatdiag];
(*		dbeta = DiagonalMatrix[obsrInfo] . catcherMatrixT .
				DiagonalMatrix[parmInfo]
changed for efficiency 9/99 DANL 
changed again 11/99 JMN (per Serguei C.) *)
        dbeta = Transpose[catcherMatrix * parmInfo] * obsrInfo;
	     ];

	     finishreport = Replace[finishreport, {
		HatDiagonal -> (HatDiagonal -> hatdiag),
		StandardizedResiduals -> (StandardizedResiduals ->
			 standResiduals),
		CookD -> (CookD -> cookD),
		JackknifedVariance -> (JackknifedVariance -> jackvar),
		StudentizedResiduals ->
		 (StudentizedResiduals -> studResiduals),
		PredictedResponseDelta -> (PredictedResponseDelta -> dfit),
		CovarianceMatrixDetRatio -> (CovarianceMatrixDetRatio ->
			 covratio),
		BestFitParametersDelta -> (BestFitParametersDelta -> dbeta) }, 1]
	   ]
	];
	
 	(* ==================== compute DurbinWatsonD =================== *)	
	
	If [MemberQ[startreport, DurbinWatsonD],
	   dw = Tr[(Rest[fitResiduals-RotateRight[fitResiduals]])^2] /
                Tr[fitResiduals^2];
	   finishreport = Replace[finishreport,
            DurbinWatsonD -> (DurbinWatsonD -> dw), 1]
	];

	(* ================= compute sequential sum of squares ============ *)
	If [MemberQ[startreport, SequentialSumOfSquares],
	   If[!ValueQ[sosDiagnostics],
	      sosDiagnostics = p > 1 &&
			TrueQ[equalvectorQ[First[designMatrixT]] &&
              Not[Apply[Or, 
                  Table[equalvectorQ[designMatrixT[[n]]], {n, 2, 
                        Length[designMatrixT]}]
              ]]
            ]
       ];
	   If[sosDiagnostics,
                 With[{modelSS = ModelSS[designMatrix, response,
				   sqrtweights, tol]},
		   (* compute vector of model sum of squares partition *)
	      	   finishreport = Replace[finishreport, {SequentialSumOfSquares ->
			(SequentialSumOfSquares -> modelSS)}, 1] 
		 ],
	         Message[DesignedRegress::seqsos];
	         finishreport = Replace[finishreport, {SequentialSumOfSquares ->
		 	(SequentialSumOfSquares -> $Failed)}, 1]	]
	]; (* end SequentialSumOfSquares *)


	(* ================= add predictedResponse to report ============ *)

	If [MemberQ[startreport, PredictedResponse],
		finishreport = Replace[finishreport, PredictedResponse ->
        	 	(PredictedResponse -> predictedResponse), 1]
	];


	(* ================= add fitResiduals to report ================= *)

	If [MemberQ[startreport, FitResiduals],
		finishreport = Replace[finishreport, FitResiduals ->
        	 	(FitResiduals -> fitResiduals), 1]
	];


	(* ================= add catcherMatrix to report ================ *)

	If [MemberQ[startreport, CatcherMatrix],
	   finishreport = Replace[finishreport, CatcherMatrix ->
		 (CatcherMatrix -> MatrixForm[catcherMatrix]), 1]
	];


	(* ================= compute variance inflation factors ============ *)

 	If [MemberQ[startreport, VarianceInflation],
           varinf = Map[(Tr[#^2])&, catcherMatrix] *
	      If[constantPresent, 	
	  	Map[(Tr[(#-Mean[#])^2])&, designMatrixT],
	  	Map[(Tr[#^2])&, designMatrixT]
	      ];
	   finishreport = Replace[finishreport,
            VarianceInflation -> (VarianceInflation -> varinf), 1]
	];

	(* ================= compute eigenstructure table ============ *)
 	If [MemberQ[startreport, EigenstructureTable],
	   Block[{weightedDesignMatrixT, pos, carrierscale, scaledDesignMatrix,
		eval, condnum, partition, proportion, svd, uuu, vvv, www, localNames},
             localNames = names;	
	     (* weightDesignMatrix is n x p, where p is the # of basis fcns,
			including the constant term *)
	     weightedDesignMatrixT = Transpose[weightedDesignMatrix];
	     (* Don't assume that the constant term corresponds to the
			first column of designMatrix. *)
	     If[constantPresent,
             (* originally:
               pos = Position[Transpose[designMatrix],
                             x_List /; Apply[Equal, x]];
                 this alternative formulation is not at all elegant,
                 but it is much faster for packed arrays. --JMN, 11/99 *)
               pos = {};
               Do[If[equalvectorQ[designMatrixT[[n]]], pos = {pos, n}],
                  {n, Length[designMatrixT]}];
               If[pos =!= {}, pos = Transpose[{Flatten[pos]}]];
	           localNames = Delete[names, pos];	
	           weightedDesignMatrixT = Delete[weightedDesignMatrixT, pos]
	     ];
	     (* now weightedDesignMatrixT is (p-1) x n *)
	     localNames = Map[If[Head[#]=!=String, InputForm[#], #]&,
		 	localNames];
	     weightedDesignMatrixT -= Map[Mean, weightedDesignMatrixT];
	     carrierscale = Map[Sqrt[#.#]&, weightedDesignMatrixT];	
	     scaledDesignMatrix = Transpose[weightedDesignMatrixT /
		 	carrierscale];
	     (* scaledDesignMatrix is n x (p-1); each column is scaled *)	
   	     (* Note the singular values found are identical to the eigenvalues
		of CorrelationMatrix[Transpose[weightedDesignMatrixT]],
		where CorrelationMatrix is defined in
		MultiDescriptiveStatistics.m *)	
	     svd = CompactSVD[scaledDesignMatrix];	
	     If[!(Head[svd]===List && Length[svd]===3),
		    finishreport = Replace[finishreport, EigenstructureTable ->
				(EigenstructureTable -> $Failed), 1],
		   {uuu, www, vvv} = svd;   
                   eval = www^2;   
                   condnum = First[www]/www;
		   (* rows of vvv are the eigenvectors of Transpose[X].X,
			where X is scaledDesignMatrix *)
		   (* partition = Transpose[Transpose[vvv]/www]^2; *)
                   partition = Map[(vvv[[#]]/www)&, Range[Length[vvv]]]^2;
                   (* each row of
                      ( responseVariance partition / carrierscale^2 )
                      gives the partition of the variance of the corresponding
                      parameter *)
		   proportion = Map[(#/Tr[#])&, partition];
		   (* each row of proportion gives the proportion of the 
		      variance attributable to the various eigenvalues for 
		      the corresponding parameter *)
                   finishreport = Replace[finishreport, EigenstructureTable ->
		     (EigenstructureTable -> TableForm[
		     Transpose[
			Join[{eval, condnum}, Chop[proportion, 10^(-4)]] ],
		     TableHeadings ->
			 {None, Join[{"EigenV", "Index"}, localNames]}
			]), 1]
	     ]
	   ] (* end Block *)
        ]; 
  
	(* ================= compute coefficient of variation  ============ *)

 	If [MemberQ[startreport, CoefficientOfVariation],
	  coeffvar = If [VectorQ[weightedResponse, Positive],
		      responseMean =
		  	     Tr[weights response] / Tr[weights];
		      Sqrt[responseVariance] / responseMean,
		      $Failed];
	  finishreport = Replace[finishreport, CoefficientOfVariation ->
	         (CoefficientOfVariation -> coeffvar), 1]
	];

	(* ========== warn user of the changed interpretation of =========== *)
	(* === RSquared and AdjustedRSquared when there is no constant === *)

	If [!constantPresent,
	   If[(MemberQ[startreport, RSquared] || MemberQ[startreport, AdjustedRSquared]),
	      Message[DesignedRegress::rsqr] ];
	   If [MemberQ[startreport, VarianceInflation],
              Message[DesignedRegress::vif] ]
	];	
	

	(* ================================================================ *)
	finishreport = Replace[finishreport, {
		 EstimatedVariance -> (EstimatedVariance -> responseVariance),
        	 BestFitParameters -> (BestFitParameters -> fit),
		 RSquared -> (RSquared -> rSquared),
		 AdjustedRSquared -> (AdjustedRSquared -> adjRSquared),
		 (* Note: BestFit is not a valid output of DesignedRegress
			since it involves basis functions *)
		 BestFit -> (BestFit -> fit.basis)
		}, 1];

        finishreport
    ] (* end of vDR *)


(* ======================================================================= *)

(* used by ParameterConfidenceRegion option *)
ellipsoidalLocus[mu_, sigma_] :=
  Module[{esystem, esystemT, sorted, r, dir},
    (* (x-mu).Inverse[sigma].(x-mu) == 1 *)
    If[Head[esystem = Eigensystem[sigma]] === Eigensystem, Return[$Failed]];
    (* radii are square roots of eigenvalues *)
    esystemT = Transpose[MapAt[Sqrt[Chop[#]]&, esystem, 1]];
    (* sort semi-axes from largest to smallest *)
    sorted = Sort[esystemT, #1[[1]] > #2[[1]]&];
    {r, dir} = Transpose[sorted];
    Ellipsoid[mu, r, dir]
  ]

(* used to process BasisNames option *)
ChangeValue[list_, name_, newval_]:=
   If[FreeQ[list,name],
      Prepend[list,(name->newval)],
      Map[If[!FreeQ[#,name], Replace[name->_,(name->_)->(name->newval)],#]&,
	  list]
   ]

GetOutputList[optionlist_] :=
    Module[{localReport, control, outlist, regReport, summary, optlist},
      summary = {ParameterTable, RSquared, AdjustedRSquared, EstimatedVariance,
	ANOVATable};
    (* split out test for option presence to prevent inadvertent unpacking
        of packed arrays in rhs of an option *)
      optlist = Map[First,optionlist];
   (* ======= ignore obsolete options and support current option ====== *)
	regReport = RegressionReport /. optionlist;
        (* fixing mistakes user made in RegressionReport spec *)
        If[Head[regReport] === List && MemberQ[regReport, RegressionReport] &&
                regReport =!= {RegressionReport},
           Message[DesignedRegress::regrep];
           regReport = DeleteCases[regReport, RegressionReport]
        ];
        regReport = regReport /. Options[DesignedRegress];
	If[Head[regReport] =!= List,
	   regReport = {regReport}];
        If[MemberQ[regReport, SummaryReport],
	   regReport = DeleteCases[regReport, SummaryReport];
	   (* delete from `summary' items that are individually listed
		in `regReport' *)	
	   Scan[(summary = DeleteCases[summary, #])&,
		Intersection[regReport,
			{ParameterTable, RSquared, AdjustedRSquared,
			EstimatedVariance, ANOVATable}]];
           localReport = Join[regReport, summary],
	   localReport = regReport]; 
      localReport
    ]

ModelSS[designMatrix_, response_, sqrtweights_, tol_] :=        
  Module[{includeConstant, n, p, mean, nsqrtweights, weightedDesignMatrixT,
	  weightedResponse, totalSS, partialWeightedDesignMatrix,
 	  svd, u, w, v, pred, scan, modelSS = {}, colweights},
    includeConstant = equalvectorQ[First[Transpose[designMatrix]]];
    {n, p} = Dimensions[designMatrix];
    mean = Tr[response]/n;
	If [VectorQ[N[sqrtweights],NumberQ] && Length[sqrtweights]==n,
		(* numericalize weights *)
		If [Precision[sqrtweights] === Infinity,
			nsqrtweights = N[sqrtweights],
			(* else *) nsqrtweights = sqrtweights
			],
		(* else *) nsqrtweights = Table[1., {n}]
		];
    weightedDesignMatrixT = Transpose[nsqrtweights*designMatrix];
    weightedResponse = nsqrtweights*response;
    If[includeConstant,
	 p--;
	 totalSS = (response-mean).(response-mean),
	 totalSS = response.response
    ];
    scan = Scan[
         (partialWeightedDesignMatrix =
		 Transpose[Drop[weightedDesignMatrixT, -#]];
		 {colweights, partialWeightedDesignMatrix} =
		   normalizeColumns[partialWeightedDesignMatrix];
          svd = CompactSVD[partialWeightedDesignMatrix, Tolerance->tol];
          If[Head[svd] === List && Length[svd] === 3,
            	{u, w, v} = svd,
                Return[$Failed]
          ];

(* Replaced:
	  pred = partialWeightedDesignMatrix . (Transpose[v] .
			 DiagonalMatrix[1/w] . u) . response;
Reason: avoid O (n^3) operations in favor of O (n^2).
9/99 DANL *)
(* Notice that if C is the diagonal matrix of column weights, A is our design
matrix, and we want to compute A . Psi[A] . response (where Psi[A] is the
generalized inverse of A) then, since Psi[A] == C.Psi[A.C] (proof: exercise)
we obtain A . Psi[A] == (A.C) . Psi[A.C].
So after forming A.C we need not explicitly multiply by the column weights
anywhere. 9/99 DANL *)
	  pred = partialWeightedDesignMatrix .
	    (v . ((Transpose[u] . response)/w));

          modelSS = {modelSS,
		  	totalSS-((response-pred).(response-pred))})&, Reverse[Range[p]-1]];
    If[scan === $Failed, Return[$Failed]];
    modelSS = Flatten[modelSS];
    (* leave the model sum of squares as is for the first nonconstant basis
       function included in model; for each additional basis function,
       compute additional model sum of squares that this function contributes *)
    ReplacePart[modelSS-RotateRight[modelSS],modelSS[[1]],1]
  ] 

(* utility function for testing that a vector is made up of identical
   elements, optimized for packed arrays - note that for a non-packed
   array, if the most common case is that this will return True, it would
   be faster to use the Min/Max, but if the usual response is False, the
   Equal form is faster. For the packed case, the Min/Max form is always
   faster in current versions. *)
equalvectorQ[v_?Developer`PackedArrayQ] :=
  Min[v] == Max[v]

equalvectorQ[v_?VectorQ] := Equal @@ v

(* ======================================================================== *)

DesignMatrix::fitc="Number of coordinates (`1`) is not equal to the number of variables (`2`)."

DesignMatrix::fitd="First argument `1` in Fit is not a list or a rectangular array."

DesignMatrix::fitm="Unable to solve for the fit parameters; the design matrix is non-rectangular, non-numerical, or could not be inverted."

Regress::invdm=DesignedRegress::invdm =
"The design matrix is not a numerical matrix."
(* if Regress called DesignedRegress, this typically means that the basis
 functions yielded non-numerical results when evaluated at the data points *)
 
DesignedRegress::numrsp="The response vector is not a list of numeric values."

DesignedRegress::invsvd =
"The singular valid decomposition argument is invalid."

DesignedRegress::invresp =
"The response vector has length `1` but should have length `2`."

DesignedRegress::invrep =
"The item `` is an invalid RegressionReport item for DesignedRegress. \
Deleting item from RegressionReport."

DesignedRegress::invopt =
"The option `` is an invalid option for DesignedRegress."

DesignedRegress::bdwghts =
"Warning: Value of option Weights -> `1` is not Automatic, a pure function \
mapping a response to a numerical weight, or a numerical vector having \
the same length as the data. Setting all weights to 1 (Weights -> Automatic)."

DesignedRegress::newsvd =
"Warning: DesignedRegress was unable to use the input singular value \
decomposition of the design matrix due to unequal weights. Decomposing \
the weighted design matrix."

DesignedRegress::nosvd =
"DesignedRegress was unable to obtain the singular value decomposition for the \
design matrix of this problem."

DesignedRegress::rank =
"Warning: the rank of the design matrix is `1`, less than full rank `2`. \
Only `1` of the `2` basis functions `3` needed to provide this fit. \
Try using a different model or greater precision."

DesignedRegress::badfit =
"Warning: unable to find a fit that is better than the mean response."

DesignedRegress::twocon =
"DesignedRegress determined that two or more constant terms are present \
in the model." (* (This should not happen under normal conditions.) *)

DesignedRegress::uncon =
"DesignedRegress was unable to determine whether a single constant term is \
present in the model." (* (This should not happen under normal conditions.) *)

DesignedRegress::seqsos =
"Basis functions are unsuitable for sequential sum of squares analysis. \
Constant should be listed first if included in basis function list."

DesignedRegress::parsos =
"Basis functions are unsuitable for partial sum of squares analysis. \
Constant should be listed first if included in basis function list."

DesignedRegress::obs =
"Warning: the option `` is an obsolete option of Regress and DesignedRegress. \
It is superseded by RegressionReport."

DesignedRegress::optx = "Unknown option `1` in `2`."

DesignedRegress::regrep =
"RegressionReport may not be included in the list specified by the \
RegressionReport option. Deleting from list."

DesignedRegress::tsos =
"Warning: the total sum of squares in the ANOVATable is uncorrected (not \
centered on the response mean) when there is no constant term in the model; \
it is designated U Total."

DesignedRegress::rsqr =
"Warning: the RSquared and AdjustedRSquared diagnostics are redefined when \
there is no constant term in the model."

DesignedRegress::vif =
"Warning: the VarianceInflation collinearity diagnostics are redefined when \
there is no constant term in the model."

(* ======================================================================== *)
Regress::bdwghts =
"Warning: Value of option Weights -> `1` is not Automatic, a pure function \
mapping an observation to a numerical weight, or a numerical vector having \
the same length as the data. Setting all weights to 1 (Weights -> Automatic)."

Regress::notfcn =
"The second argument to Regress is not a list of functions or a \
single function."

Regress::notvar =
"The third argument to Regress is not a list of variables or a \
single variable."

Regress::notdata =
"The first argument to Regress is not a list or rectangular array of numeric data."

Regress::coord = "The number of independent coordinates (`1`) in the data is \
less than the number of variables (`2`) in the model."

Regress::crdvl = "One or more of the coordinates `1` has a value. The \
coordinates must be symbols."

Regress::nodm = "Unable to compute design matrix."

Regress::namnum =
"Warning: number of basis function names conflicts with number of basis \
functions. Using default names."

Regress::invnam =
"Warning: invalid BasisNames option value. Using default names."

Regress::mindata = "The number of parameters to be estimated is greater than \
or equal to the number of data points. Subsequent results may be misleading."

End[]

SetAttributes[
    {Regress, DesignedRegress},
    {Protected, ReadProtected}
];

EndPackage[]

(* :Example:
x: number of days material is stored
y: amount of methane gas produced

data = {{1, 8.2}, {1, 6.6}, {1, 9.8}, {2, 19.7}, {2, 15.7}, {2, 16.0},
        {3, 28.6}, {3, 25.0}, {3, 31.9}, {4, 30.8}, {4, 37.8}, {4, 40.2},
        {5, 40.3}, {5, 42.9}, {5, 32.6}}

Regress[data, {1, x, x^2}, x, Weights ->
        1/{1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5}^2 ]

{ParameterTable ->

>          Estimate    SE         TStat      PValue     ,
      1    -3.62995    2.53839    -1.43002   0.178228

      x    12.421      2.67432    4.64453    0.000565756

       2
      x    -0.717542   0.520765   -1.37786   0.193398

>    RSquared -> 0.948947, AdjustedRSquared -> 0.940438,

>    EstimatedVariance -> 1.52011,

>    ANOVATable ->         DF   SumOfSq   MeanSq    FRatio    PValue      }
                                                                        -8
                   Model   2    339.059   169.53    111.525   1.77066 10

                   Error   12   18.2413   1.52011

                   Total   14   357.3


Example of undocumented feature... pure function pointed to by Weights
may be a function of independent variable x....


Regress[data, {1, x, x^2}, x, Weights -> (1/#2^2 &)]

	same as above

Regress[data, {1, x, x^2}, {x}, Weights -> (1/#2^2 &)]

	same as above

*)


(* :Example:
x1, x2, y: simulated data

data = Table[(x1 = i;
	      x2 = Prime[i];
	      {x1, x2, 3 x1 + 2 x2 + Random[NormalDistribution[0, 
			Sqrt[x1+x2]/2]]}), {i, 20}];
	
data = {{1, 2, 6.71949}, {2, 3, 11.5972}, {3, 5, 18.0687}, {4, 7, 30.2149}, 
    {5, 11, 36.7307}, {6, 13, 42.6693}, {7, 17, 55.28}, {8, 19, 61.2664}, 
    {9, 23, 72.7406}, {10, 29, 81.9494}, {11, 31, 96.9331}, 
    {12, 37, 108.132}, {13, 41, 119.574}, {14, 43, 130.867}, 
    {15, 47, 138.833}, {16, 53, 157.414}, {17, 59, 177.146}, 
    {18, 61, 175.105}, {19, 67, 191.956}, {20, 71, 212.905}}

Example of undocumented feature... pure function pointed to by Weights
may be a function of independent variables x1 and x2....

Regress[data, {1, x1, x2}, {x1, x2}, Weights -> ((4/(#2 + #3))&)]

{ParameterTable -> 
 
>          Estimate   SE         TStat        PValue      , 
      1    -0.03908   1.05484    -0.0370483   0.970878

      x1   2.70723    0.692708   3.90818      0.00113143

                                                        -9
      x2   2.12577    0.203008   10.4714      7.84933 10
 
>    RSquared -> 0.998064, AdjustedRSquared -> 0.997836, 
 
>    EstimatedVariance -> 1.17182, 
 
>    ANOVATable ->         DF   SumOfSq   MeanSq    FRatio   PValue}
                   Model   2    10268.4   5134.2    4381.4   0.

                   Error   17   19.9209   1.17182

                   Total   19   10288.3


Regress[data, {1, x1, x2}, {x1, x2},
	 Weights -> Table[4/(i + Prime[i]), {i, 20}]]

	same as above
	
*)
