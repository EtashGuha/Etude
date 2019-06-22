(* :Title: Nonlinear Regression *)

(* :Context: NonlinearRegression` *)

(* :Author: John M. Novak & E. C. Martin *)

(* :Summary:
This package performs nonlinear least squares curve fitting and
statistical regression analysis.  
*)

(* :Copyright: Copyright 1991-2007, Wolfram Research, Inc. *)

(* :Package Version: 3.1 *)

(* :Mathematica Version: 6.0 *)

(* :History:
    Version 1.0, October 1991 by John M. Novak.
    Version 1.1, February 1992 by John M. Novak--major revisions.
    Version 1.2, February 1994 by ECM--major revisions and additions:
        added NonlinearRegress to compute a variety of diagnostics for a
        nonlinear fit; revised output of NonlinearFit to give a function
        (as does the output of Fit); replaced use of LinearSolve and
        termination upon encountering an ill-conditioned "alpha" matrix with 
        a more robust procedure for finding the next step in the parameter
        search; eliminated specifying weights in data matrix, because they can 
        be specified using Weights and the data matrix format conflicts with 
        the obvious representation of multiple-response data; tried to make 
        option interpretation more uniform across methods; added two more
        parameter formats: {parameter, {start0, start1}} and
        {parameter, {start0, start1}, minr, max}... currently these are only
        effective at eliminating the need for symbolic derivatives if
        Method -> FindMinimum;  added support for Gradient option, also
        relevent when Method -> FindMinimum.
    Version 1.3, 1.4, October 1996 by ECM.
    Version 1.5, January 1997 by Sirpa Saarinen -- implemented
        LevenbergMarquardt method using System`Private`SLM. 
    Version 1.6, May 1997 by ECM -- reimplemented LevenbergMarquardt method
        (and methods Gradient, Newton, and QuasiNewton) using FindMinimum.
        The setting Method -> FindMinimum is now obsolete since
        the V4.0 function FindMinimum supports several methods (Automatic,
        Gradient, LevenbergMarquardt, Newton, and QuasiNewton) .  To get the
        V3.0 default behavior of Method -> FindMinimum (steepest descent), use
        Method -> Gradient.  
    Version 1.7, May 1998 by ECM -- modified Weights option so it could
        be set to a pure function of the entire observation vector... not
        just the response.  A pure function of a single argument is still
        applied to the response as before.
    Version 1.8, July 1998 by ECM -- added support for
        ParameterConfidenceRegion (based on AsymptoticCovarianceMatrix).
        Modified StandardizedResiduals to take account of Weights.
    Version 1.9, November 1999 by Daniel Lichtblau and Serguei Chebalov
      with John M. Novak, modifications for improved performance with
      matrix operations
    Version 2.0, March 2003 by John M. Novak -- adapt package to the
      new kernel function FindFit
    Version 3.0, October 2005 by Darren Glosemeyer, extended NonlinearFit 
      and NonlinearRegress to handle constrained models
   	Version 3.1, 2006 by Darren Glosemeyer, moved from Statistics`NonlinearFit` 
   	  standard add-on to NonlinearRegression` package and changed argument 
   	  order to be consistent with FindFit
	Obsoleted in Mathematica version 7.0, Darren Glosemeyer;
  	  Functionality is replaced by LinearModelFit and related FittedModel functionality.
*)

(* :Keywords:
    curve fitting, nonlinear regression
*)

(* :Sources:
    Janhunen, Pekka, NonlinearFit`, (a Mathematica package), 1990.
    Press, William, et. al., "Numerical Recipes in Pascal",
        pp. 572-580, Cambridge University Press (Cambridge, 1989).
    Withoff, Dave, DataAnalysis`NonlinearRegression`,
        (a Mathematica package), 1989.
    Bates and Watts, "Relative Curvature Measures of Nonlinearity",
        J.R.Stat.Soc.B, 42 No.1, pp.1-25.
    Ratkowsky, David A., "Nonlinear Regression Modeling", Marcel Dekker, Inc.
        (New York, 1983).
    Kennedy, W. J. and Gentle, J. E., "Statistical Computing", Marcel Dekker,
        Inc. (New York, 1980).
*)

(* :Discussion:

   The actual nonlinear fit is performed by the kernel FindFit
    function; the function NonlinearFit is obsolete. NonlinearRegress
    uses FindFit to get the fit.

   What the curvature diagnostics in FitCurvatureTable mean:
    A maximum intrinsic curvature that is small compared to the curvature of the
    joint parameter confidence region indicates that the curved solution locus 
    is well approximated by a plane.  A maximum parameter-effects curvature that
    is small compared to the curvature of the joint parameter confidence region
    indicates that the curved parameter lines on the approximating plane are
    well approximated by a grid of straight, parallel, equispaced lines.  The
    validity of inferences made regarding parameters of the nonlinear model 
    depend on how well the `planar' and `uniform coordinate' assumptions are
    satisfied.  Model reparameterization can reduce the parameter-effects
    curvature, but not the intrinsic curvature.  (The confidence region
    curvature, which provides a basis for comparison, is for a region whose
    confidence level is specified by the option ConfidenceLevel.)

The Weights option....

(1) Weights -> vector

  When Weights -> {w1, ..., wn}, then the
        weighted residual sum of squares is minimized:
                Sum[wi (yi - g[params, xi1, ..., xik])^2, {i, 1, n}]
        (where g[params, ] represents the nonlinear model).

(2) Weights -> pureFcn

  (i) In NonlinearRegress[{y1, ..., yn}, model, x, parms, Weights -> (w[#] &)],
        the weight function is applied to the response y, and the weighted sum
                Sum[w[yi] (yi - g[params, i])^2, {i, 1, n}]
        is minimized.
      In NonlinearRegress[{y1, ..., yn}, model, x, parms,
                 Weights -> (w[#1, #2] &)],
        the weight function is applied to the response y and the implied
        independent variable x, and the weighted sum
                Sum[w[yi, i] (yi - g[params, i])^2, {i, 1, n}]
        is minimized.

  (ii) In NonlinearRegress[{{x1, y1}, ..., {xn, yn}}, model, x, parms,
                                                 Weights -> (w[#] &)],
         the weight function is applied to the response y, and the weighted sum
                Sum[w[yi] (yi - g[params, xi])^2, {i, 1, n}]
         is minimized.
       In NonlinearRegress[{{x1, y1}, ..., {xn, yn}}, model, x, parms,
                                                Weights -> (w[#1, #2] &)],
         the weight function is applied to the response y and the explicit
         independent variable x, and the weighted sum
                Sum[w[yi, xi] (yi - g[params, xi])^2, {i, 1, n}]
         is minimized.

  (iii) In NonlinearRegress[{{x11, ..., x1k, y1}, ..., {xn1, ..., xnk, yn}},
                     model, {x1, ..., xk}, parms, Weights -> (w[#] &)],
          the weight function is applied to the response y, and the weighted sum
                Sum[w[yi] (yi - g[params, xi, ..., xk])^2, {i, 1, n}]
          is minimized.
        In NonlinearRegress[{{x11, ..., x1k, y1}, ..., {xn1, ..., xnk, yn}},
              model, {x1, ..., xk}, parms, Weights -> (w[#1, ..., #(k+1)] &)],
          the weight function is applied to the response y and the explicit
          independent variables x1, ..., xk, and the weighted sum
                Sum[w[yi, x1i, ..., xki] (yi - g[params, x1i, ..., xki])^2,
                                                                 {i, 1, n}]
          is minimized.




   Unlike some nonlinear fitting procedures, it is not possible to use the
   Weights option to specify a function of a residual, so that the weights can
   be updated each iteration, and iteratively reweighted least squares
   (preferably using more robust criteria than the l2 norm) can be performed.
   However, the FindFit function can support this kind of operation
   via the NormFunction and EvaluationMonitor options.
*)

(* :Warning:
    Use FindFit[data, model, params, vars] to get the list of parameter
        replacement rules previously given by NonlinearFit[data, model, vars,
        params] in V2.2.
    Currently there is no way to utilize an estimate for the standard error
        based on replications.  For example, it should be possible to
        calculate both the maximum relative curvatures in FitCurvatureTable
        and the bias given by ParameterBias using a standard error estimate
        input by the user.  Currently the estimate for the standard error
        is always taken to be Sqrt[EstimatedVariance] which is based on the
        fit residuals.  The capability to input and use a separate estimate
        of the standard error should be added in a future version of the
        package.
*)

Message[General::"obspkg", "NonlinearRegression`"]

BeginPackage["NonlinearRegression`",        
        (* needed for StudentTCI: *)
       {"HypothesisTesting`",
        (* needed for Weights, RegressionReport,
           RegressionReportValues, and many symbols allowed by
           RegressionReportValues[NonlinearRegress] *)
       "RegressionCommon`",
	   (*needed for Ellipsoid in version 7*)
	   "MultivariateStatistics`"}]

If[ Not@ValueQ[NonlinearRegress::usage],
NonlinearRegress::usage =
"NonlinearRegress[data, model, pars, vars] searches for a least-squares \
fit to a list of data according to the model containing the variables vars \
and the parameters pars. NonlinearRegress[data, {model, cons}, pars, vars] fits \
model subject to the constraints cons. Parameter specifications are the same as in FindFit. \
The data can have the form {{x1, y1, ..., f1}, {x2, y2, ..., f2}, ...}, where \
the number of coordinates x, y, ... is equal to the number of variables in the \
list vars. The data can also be of the form {f1, f2, ...}, with a single \
coordinate assumed to take values 1, 2, .... The Method option specifies the \
LevenbergMarquardt (default), Gradient (steepest descent), Newton, \
QuasiNewton or Automatic search methods. The Automatic method does linear \
fitting for linear models and LevenbergMarquardt nonlinear fitting for \
nonlinear models."]

Options[NonlinearRegress] = Sort[Join[
      {RegressionReport -> SummaryReport,
       Tolerance -> Automatic},
      {AccuracyGoal -> Automatic,
    	Compiled -> Automatic,
    	Gradient -> Automatic,
    	MaxIterations -> 100,
    	Method -> Automatic,
    	PrecisionGoal -> Automatic,
    	Weights -> Automatic,
    	WorkingPrecision -> MachinePrecision},
      {ConfidenceLevel->.95} ]]
(* NOTE: Tolerance is used in SingularValueDecomposition and maxcurve.  
        Also note that FitCurvatureTable uses
        ConfidenceLevel from Options[StudentTCI]. *)

RegressionReportValues[NonlinearRegress] =
{ANOVATable, AsymptoticCorrelationMatrix, AsymptoticCovarianceMatrix,
BestFit, BestFitParameters,
EstimatedVariance, FitCurvatureTable, FitResiduals, 
HatDiagonal, MeanPredictionCITable, ParameterBias, ParameterConfidenceRegion,
ParameterCITable, ParameterTable, PredictedResponse,
SinglePredictionCITable, StartingParameters, StandardizedResiduals,
SummaryReport};


        (* symbols unique to nonlinear regression *)
        (* updating usage of shared symbols *)


Begin["`Private`"]

(* ===================== NonlinearRegress warning messages ================ *)

NonlinearRegress::bdargs =
"Arguments for NonlinearRegress must be in the form \
NonlinearRegress[data, model, variables, parameters, (options)]."

NonlinearRegress::bddata =
"The data argument of NonlinearRegress must be a matrix. Weighted regression \
is specified using the Weights option. The data format \
{{x11, x12, ..., {y11, ..., y1m}}, {x21, x22, ..., {y21, ..., y2m}}, ..., \
{{xn1, xn2, ..., {yn1, ..., ynm}}} is reserved for multiple response data, \
which will be supported in the future."

NonlinearRegress::bdwghts =
"Warning: Value of option Weights -> `1` is not Automatic, a pure function \
mapping a response to a non-negative numerical weight, or a non-negative \
numerical vector having the same length as the data. Setting all weights to 1 \
(Weights -> Automatic)."

NonlinearRegress::nosvd =
"NonlinearRegress was unable to obtain the singular value decomposition for the \
design matrix corresponding to the linearized problem."

NonlinearRegress::noqrd =
"NonlinearRegress was unable to obtain the QR decomposition of the design \
matrix corresponding to the linearized problem. Unable to compute curvature \
diagnostics or parameter bias."

NonlinearRegress::noinv =
"NonlinearRegress was unable to obtain an inverse or pseudoinverse for the \
R matrix in the QR decomposition of the design matrix corresponding to the \
linearized problem. Unable to compute curvature diagnostics or parameter bias."

NonlinearRegress::nodm =
"NonlinearRegress was unable to compute a design matrix, \
because it could not find derivatives of the model. Some \
requested diagnostics may not be able to be generated."

NonlinearRegress::pefail =
"Maximum parameter-effects curvature search failed in `` iterations to find the \
direction vector that when dotted with the gradient vector gives unity to \
within a tolerance of ``. Parameter-effects curvature returned may not be \
maximum value."

NonlinearRegress::infail =
"Maximum intrinsic curvature search failed in `` iterations to find the \
direction vector that when dotted with the gradient vector gives unity to \
within a tolerance of ``. Intrinsic curvature returned may not be maximum \
value."

NonlinearRegress::regrep =
"RegressionReport may not be included in the list specified by the \
RegressionReport option. Deleting from list."

NonlinearRegress::zerovar =
"`` cannot be computed in the case of zero EstimatedVariance."

NonlinearRegress::zerocov =
"`` cannot be computed in the case of zero elements on the diagonal of \
AsymptoticCovarianceMatrix."

NonlinearRegress::rank =
"Warning: Parameter solution locus has dimensionality of only `` at the \
parameter estimate, less than full dimensionality ``. Confidence region \
curvature given by 1/Sqrt[Quantile[FRatioDistribution[``, ``], ``]]."

NonlinearRegress::crvtab =
"Warning: Computing the orthogonal complement of a `` x `` matrix for \
quantities needed by FitCurvatureTable and ParameterBias can be lengthy. \
To avoid the wait, abort this calculation and repeat the regression, \
omitting FitCurvatureTable and ParameterBias from the items specified by \
RegressionReport."

NonlinearRegress::bdfit =
"Warning: unable to find a fit that is better than the mean response."

NonlinearRegress::bdcl =
"Warning: Value of option ConfidenceLevel -> `1` is not a number between 0 and 1. \
Setting to `2`."

NonlinearRegress::linfit =
"Warning: the model `1` is linear. Regression diagnostics may not be \
reliable, try using Regress from the `2` package."; (* use of `2` here is a hack around a bug. *)

NonlinearRegress::constr = 
"The report values `1` assume an unconstrained model. The results for these \
report values may not be valid, particularly if the fitted parameters are near \
a constraint boundary.";

NonlinearRegress::spnmin="Unable to extract StartingParameters from NMinimize when \
an EvaluationMonitor is given. Try extracting the parameter values as part of the \
EvaluationMonitor.";

NonlinearRegress::dof:="The number of parameters `1` is not less than the number of \
data values `2`. Some report values cannot be computed.";

(* ========= NonlinearRegress fatal error messages ========= *)

NonlinearRegress::tmny =
"Too many variables have been provided for the size of the data; \
given `1`-tuples for data, given `2` variables."

NonlinearRegress::bdpar =
"The form of the parameters `1` given to NonlinearRegress is not correct. \
Parameters should be symbols or lists containing a symbol and between 0 and \
4 real values."

NonlinearRegress::bdgrid =
"Specifying parameters in the form `1` means that a search of \
`2` grid points in the `3`-dimensional parameter space `4` is conducted to find \
the best starting point (i.e., the one minimizing the residual sum of squares). \
Unfortunately, each of the `2` points makes the model complex valued at one or \
more observations. Try a different starting point for the parameters."
(* NOTE:
 or try specifying a new model (it is possible that one of the parameters
is not meant to take on real values) *)

NonlinearRegress::srange =
"The starting point `1` for the parameter `2` is outside the prescribed parameter \
range `3`. Try another starting point for the parameter."

NonlinearRegress::unevldr =
"One or more derivatives of the model with respect to parameters did not \
evaluate."

NonlinearRegress::nonnum =
"The model is not numerical at ``. Check that all model parameters are \
included in the parameter list."

NonlinearRegress::fitfail =
"The fitting algorithm failed."

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


(* ============================= NonlinearRegress =========================== *)

(* NonlinearRegress arguments are the same as those of NonlinearFit *)

NonlinearRegress[args___] := 
        With[{result = Catch[nonlinearFit[NonlinearRegress, args]]},
                result /; result =!= $Failed && FreeQ[result, nonlinearFit]];


(* =============================== nonlinearFit ============================= *)

(* nonlinearFit calls FindFit *)
nonlinearFit[msghead_, xdata_, model_, xparams_, xvars_,
             opts___?OptionQ] :=
    Block[{accgoal, grad, maxits, precgoal, tol, weights,
            tweights, workprec, progress,
            params, fixedparams, w, s1,
            ddims, datalen, tempdata, varlen,
            pts, tempweights, response, scan,
            start, out, p, select, step, stepmonitor,
            npts, nresponse, normfunc,
            temp, data = xdata,
            inputparams = xparams, vars = xvars, stepcounter, stringp, csf,
            modelfunction=If[Head[model]===List,model[[1]],model],
            evalmonitor, eval, evalcounter, spnminflag,method},
        {accgoal, grad, maxits, precgoal, tol, weights,
                 workprec, progress, method} =
            {AccuracyGoal, Gradient, MaxIterations, 
                PrecisionGoal, Tolerance, Weights, WorkingPrecision,
                "ShowProgress",Method} /. 
                 	(Flatten[{opts, Options[msghead]}]/. 
                 	{(_?(StringMatchQ[ToString[#], "ShowProgress"] &)) -> "ShowProgress"}); 
	(* Option and Argument checking *)
        (* Check the variables *)
        If[!ListQ[vars], vars = {vars}];
        varlen = Length[vars];
        Map[
                If[!symbolQ[#],
                        Message[MessageName[msghead,"bdvar"],xvars];
                        Throw[$Failed]],
                vars];
        (* Check parameter specifications which include starting conditions *)
        If[parameterQ[inputparams], inputparams = {inputparams}];
        Map[
                If[!parameterQ[#],
                        Message[MessageName[msghead,"bdpar"], xparams];
                        Throw[$Failed]],
                inputparams];
    (* - check input data, separate out response and coords *)
        If[!ListQ[data],
                Message[MessageName[msghead,"bddata"]];
                Throw[$Failed]];
        (* try to pack the data if using machine precision *)
        If[SameQ[workprec, MachinePrecision],
                data = Developer`ToPackedArray[data, Real],
        (* else *) 
        (* check this before using N *)
        workprec = N[workprec];
        If[!(NumberQ[workprec] && Positive[workprec]), 
            Message[NonlinearRegress::wprec, workprec];
            Throw[$Failed]];
                data = N[data, workprec]
        ];
        datalen = Length[data];
        ddims = Dimensions[data];
        Switch[Length[ddims],
                1,
                    nresponse = data;
                    temp = Range[First[ddims]];
                    pts = Developer`ToPackedArray[Transpose[Table[temp,{varlen}]]];
                    If[True, (* Do we really need this ? *)
                            response = xdata];
                    npts = N[pts, workprec],
                2,
                    If[ddims[[2]] - varlen < 1,
                            Message[MessageName[msghead, "tmny"], ddims[[2]], varlen];
                            Throw[$Failed]];
                    (* - assume univariate response *)
                    nresponse = data[[All, -1]];
                    npts = Take[data, All, {1,varlen}];
                    If[True, (* Do we really need this ? *)
                            response = xdata[[All,-1]];
                            pts = Take[data, All, {1,varlen}]
                    ],
                _,
                    Message[MessageName[msghead,"bddata"]];
                    Throw[$Failed]];
        If[!Developer`PackedArrayQ[nresponse],
                If[!VectorQ[nresponse, NumberQ],
                        Message[MessageName[msghead,"bddata"]];
                        Throw[$Failed]]];
        If[!Developer`PackedArrayQ[npts],
                If[!MatrixQ[npts, NumberQ],
                        Message[MessageName[msghead,"bddata"]];
                        Throw[$Failed]]];

    (* - check and set weights *)
        Which[
         weights === Equal || weights === Automatic,
                weights = 1,
         Head[weights] === Function,
                (* Rearrange data so that response is first... if weights
                        has only one arg, then it is applied to the response.
                   Note that as of May 1998, the ability to use a weights
                        function of more than one arg is not documented. *)
                tempdata = Map[Join[{Last[#]}, Drop[#, -1]]&, data];
                tempweights = Map[Apply[weights, #]&, tempdata];
                If[!VectorQ[N[tempweights], NumberQ[#]&&NonNegative[#]&],
                        Message[NonlinearRegress::bdwghts, weights];
                        weights = 1,
                        weights = tempweights
                ],
         VectorQ[weights],
                If[Length[weights] =!= Length[nresponse] ||
                   !VectorQ[N[weights],  NumberQ[#]&&NonNegative[#]&],
                        Message[NonlinearRegress::bdwghts, weights];
                        weights = 1
                ],
         True,
                Message[NonlinearRegress::bdwghts, weights];        
                weights = 1
        ];
          (* note that some computations involving weights are better
         done as scalars for efficiency, but others need a vector;
         thus, we maintain both and pass the appropriate version to
         the appropriate routines. *)
        tweights = If[NumberQ[weights],
                        Table[weights, {Length[nresponse]}],
            weights
        ];


    (* - fill in default values for start, start0, start1, minr, and maxr *)
        inputparams = Map[(# /. 
			{symb_?symbolQ, {num1_?numberQ, num2_?numberQ}, y___} :> {symb, num1, num2, y}) &, inputparams];
		If[Not[VectorQ[inputparams,
   			MatchQ[#,((_?symbolQ) | {_?symbolQ, _?numberQ} | 
   			{_?symbolQ, _?numberQ, _?numberQ} | 
   			{_?symbolQ, _?numberQ, _?numberQ, _?numberQ, _?numberQ} | 
   			{_?symbolQ, _?numberQ, _?numberQ, _?numberQ})]&]]
   			, 
 			Message[NonlinearRegress::bdpar, inputparams];
 			Return[$Failed]
 			];
		If[Catch[Map[Switch[Length[#],
   			4,
   			If[validateParameterBounds[#, 3, 4, {2}] === $Failed, 
    			Throw[$Failed]],
   			5,
   			If[validateParameterBounds[#, 4, 5, {2, 3}] === $Failed, 
    			Throw[$Failed]],
   			_,
   			Null] &, inputparams]]===$Failed,
   			Return[$Failed]];
    (* - check assorted options *)
        If[precgoal === Automatic, 
            precgoal = Max[6 workprec/$MachinePrecision, workprec - 10.]];
        If[accgoal === Automatic, 
            accgoal = Max[6 workprec/$MachinePrecision, workprec - 10.]];
        If[!NumberQ[precgoal] && precgoal =!= Infinity,
           Message[NonlinearRegress::precg, precgoal];        
           Throw[$Failed]];
        If[!NumberQ[accgoal] && accgoal =!= Infinity, 
           Message[NonlinearRegress::accg, accgoal];         
           Throw[$Failed]];
        If[method === FindMinimum,
            Message[NonlinearRegress::obsmtd];
            method = Gradient
        ];
        progress = TrueQ[progress];
        (* Note that tol is used by MakeRegressionReport.
               It is used by SingularValueDecomposition
           and maxcurve. *)
        If[(!(tol === Automatic ||
                (NumberQ[tol] && FreeQ[tol, Complex] && 0 < tol < 1))),
           Message[NonlinearRegress::bdtol, tol, Automatic];
           tol = Automatic];
                
      (* set up handling for options and args that are different between
         NonlinearRegress and FindFit *)
        normfunc = If[weights === 1,
            Norm,
            With[{w = Sqrt[weights]}, (Norm[# * w]) &]
        ];
        params=Map[If[symbolQ[#],#,#[[1]]]&,inputparams];
        stepmonitor = Which[
            (* check if step monitor passed in; if so, it has precedent
               over ShowProgress. This is not a NonlinearRegress option,
               though, so it isn't set in Options. *)
			(step = Position[First/@{opts}, StepMonitor]) =!= {},
				{opts}[[First[step]]],
			
            progress, 
                      stringp = ToString /@ params;
                      csf[p_] := makechisq[model, vars, pts, response,
                                           tweights, Thread[stringp -> p]];
                      StepMonitor :> Print["Iteration:", stepcounter++,
                                              " ChiSquared:", csf[params],
                                              " Parameters:", params],
            True,     StepMonitor -> None
        ];
        
        evalmonitor=Which[
        	spnminflag=(Not[FreeQ[RegressionReport/.{opts}/.Options[NonlinearRegress],
        		StartingParameters]]&&
        		Not[FreeQ[Method/.{opts}/.Options[NonlinearRegress],NMinimize]]&&
        		(eval = Position[First/@{opts}, EvaluationMonitor]) =!= {})
        		,
        		Message[NonlinearRegress::spnmin];
        		{opts}[[First[eval]]]
        		,
        		Not[FreeQ[RegressionReport/.{opts}/.Options[NonlinearRegress],
        		StartingParameters]]&&
        		Not[FreeQ[Method/.{opts}/.Options[NonlinearRegress],NMinimize]]
        		,
        		EvaluationMonitor:>(If[evalcounter===1,start=params];evalcounter++)
        		,
        		True
        		,
        		EvaluationMonitor->None];
      
     (* do the FindFit call *)
        stepcounter = evalcounter = 1;
        out = If[Head[Unevaluated[#]]===FindFit, $Failed, #]&[
                FindFit[xdata, model, inputparams, vars,
                 FilterRules[
                 	Join[{stepmonitor, evalmonitor,NormFunction -> normfunc,opts},
                 		Options[NonlinearRegress]],
                 	Options[FindFit]]]
        ];
    (* check output of fitting routine *)
        If[out === $Failed || Head[out] === FindFit,
            Message[NonlinearRegress::fitfail];        
            Return[$Failed]
        ]; 

        MakeRegressionReport[
              {makechisq[modelfunction, vars, pts, response, tweights, out],
               out},
              pts, response, tweights, model, vars, params, 
              Which[spnminflag,
              	$Failed,
              	VectorQ[start, numberQ],
              	start,
              	Head[inputparams[[1]]]===List&&MemberQ[{2,4},Length[inputparams[[1]]]],
              	inputparams[[All,2]],
              	Head[inputparams[[1]]]===List&&MemberQ[{3,5},Length[inputparams[[1]]]],
              	Transpose[inputparams[[All,{2,3}]]],
              	True,
              	Table[1,{Length[inputparams]}]],
              tol, maxits, workprec, opts]
        
    ] (* end nonlinearFit *)

nonlinearFit[h_, ___] := $Failed/;(Message[MessageName[h, "bdargs"]]; False)

(* =============================== nonnumMessage ========================= *)

nonnumMessage[tempRule_] :=
           Message[NonlinearRegress::nonnum, tempRule];

(* ============================== makechisq  ============================= *)
(* following is a utility to generate the chi^2 value after the fit has
   been performed, for use in MakeRegressionReport. *)
makechisq[model_, vars_, pts_, response_, tweights_, out_] :=
    Module[{modelF, chi},
        modelF = (Evaluate[model /. Join[Thread[Rule[vars, 
        	Map[Slot, Range[Length[vars]]]]],out]]) &;
        chi = response - Apply[modelF, pts, {1}];
        (tweights chi).chi
    ]


(* ========================== GetRegressionReport ========================= *)

GetRegressionReport[options_] :=
    Module[{localReport, control, outlist, regReport, summary},
      (* NOTE: the summary list for Regress:
         summary = {ParameterTable,RSquared,AdjustedRSquared,EstimatedVariance,
         ANOVATable} *)
      summary = {BestFitParameters, ParameterCITable, EstimatedVariance,
                 ANOVATable, AsymptoticCorrelationMatrix, FitCurvatureTable};
      regReport = RegressionReport /. options;
      (* fixing mistakes user made in RegressionReport spec *)
      If[Head[regReport] === List && MemberQ[regReport, RegressionReport] &&
                regReport =!= {RegressionReport},
         Message[NonlinearRegress::regrep];
         regReport = DeleteCases[regReport, RegressionReport]
      ];
      regReport = regReport /. Options[NonlinearRegress];
      If[Head[regReport] =!= List,
           regReport = {regReport}];
      If[MemberQ[regReport, SummaryReport],
         regReport = DeleteCases[regReport, SummaryReport];
         (* delete from `summary' items that are individually listed
                in `regReport' *)
         Scan[(summary = DeleteCases[summary, #])&,
              Intersection[regReport,
                         {BestFitParameters, ParameterCITable,
                          EstimatedVariance,
                          ANOVATable, AsymptoticCorrelationMatrix,
                          FitCurvatureTable}
                        ]];
           localReport = Join[regReport, summary],
           localReport = regReport];
      localReport
    ]

(* ========================== MakeRegressionReport ========================= *)

(* MakeRegressionReport is called by nonlinearFit. *)

MakeRegressionReport[out_, pts_, response_, weights_, model_, vars_,
        params_, start_, tol_, maxits_, wp_, opts___] :=
        Block[{cLevel,
               report, errorSS, parameterRules, parameterEst, uv, uvars,
               variableRules, modelF,
               modelprime, modelFprime, modelprimeprime, modelFprimeprime,
               INcurve, PEcurve, bias, rinv, locusDOF,        
               predictedResponse, fitResiduals, totalDOF, modelDOF, errorDOF,
               responseVariance, npts, ndesignMatrix, pP,
               sqrtweights = Sqrt[weights], nresponse,
               weightedResponse, nweightedResponse,
               meanWeightedResponse, nmeanWeightedResponse,
               uncorrectedTotalSS, correctedTotalSS, modelSS,
               arglist, basisSubset, startreport, finishreport,
               modelfunction=If[Head[model]===List,model[[1]],model],
               insufficientdataflag, cnf, cnfopts, grad},

          cLevel = ConfidenceLevel /. {opts} /. Options[NonlinearRegress];
          If[ !(NumberQ[cLevel] && FreeQ[cLevel, Complex] && 0 < cLevel < 1),
             Message[NonlinearRegress::bdcl, cLevel, 0.95];
             cLevel = .95];

          report = GetRegressionReport[{opts}];
          (* for constrained problems, results based on asymptotic normality 
          	 will not be valid in general; issue a warning if the model is constrained 
          	 and any such results are a part of the RegressionReport *)
          If[Head[model]===List&&Length[model]>1&&Not[TrueQ[model[[2]]]],
          	If[#=!={}, Message[NonlinearRegress::constr, #]]&[
          		Intersection[Flatten[{report}], 
          			{ANOVATable, AsymptoticCorrelationMatrix, AsymptoticCovarianceMatrix,
					EstimatedVariance, FitCurvatureTable, HatDiagonal, MeanPredictionCITable, 
					ParameterBias, ParameterConfidenceRegion, ParameterCITable, 
					ParameterTable, SinglePredictionCITable, StandardizedResiduals}]]
			];

     (* split report into two components, one for checking the presence
        of an option, the other for holding the results. This is used
        to prevent unpacking of packed arrays in results in tests for
        MemberQ *)
      startreport = report;
      finishreport = report;

          {errorSS, parameterRules} = out;
          If[!FreeQ[errorSS,Complex]&&
          	Im[errorSS]<(10^-AccuracyGoal+Re[errorSS]*10^-PrecisionGoal)/.{opts}/.Options[NonlinearRegress]/.
          		{Automatic->wp/2}/.{MachinePrecision->$MachinePrecision},
          	errorSS=Re[errorSS]];
          parameterEst = params /. parameterRules;
          uvars = Map[Slot,Range[Length[vars]]];
          variableRules = Thread[Rule[vars, uvars]];
          modelF = 
                Evaluate[(modelfunction /. parameterRules) /. variableRules]&;
          cnfopts = Flatten[{WorkingPrecision -> wp, 
   				FilterRules[{opts}, Options[Experimental`CreateNumericalFunction]]}];
   		  grad=Gradient/.{opts}/.Options[NonlinearRegress];
   		  If[grad===Automatic,
   		  	(* Gradient is Automatic or is dimensionally invalid *)
   		  	If[FreeQ[(modelprime = Map[D[modelfunction, #]&, params]), Derivative],
             (* Dimensions[modelFprime[[2]]] = {p} *)
             modelFprime = 
                  Evaluate[(modelprime /. parameterRules) /. variableRules]&;
             modelprimeprime = Map[D[modelprime, #]&, params];
             If[FreeQ[modelprimeprime,Derivative],
                (* Dimensions[modelFprime[[2]]] = {p, p} *)
                modelFprimeprime = Evaluate[(modelprimeprime /. parameterRules) /.
                                variableRules]&,
                (* modelprimeprime is NOT FreeQ of Derivative, so evaluate the Hessian numerically *)
                (* quiet nnum messages that will correctly be generated in the process of setting up a pure functions for the derivatives *)
                modelFprimeprime=Quiet[Evaluate[(cnf[params, 
 					modelfunction/. variableRules, {},
 					Apply[Sequence,cnfopts]]/.cnf->Experimental`CreateNumericalFunction)&[##]["Hessian"[params/. parameterRules]]],
 					{Experimental`NumericalFunction::nlnum, Experimental`NumericalFunction::nnum,Experimental`NumericalFunction::symd}] ;
             ],
             (* modelprime is not free of Derivative, so evaluate derivatives numerically *)
             (* quiet nnum messages that will correctly be generated in the process of setting up pure functions for the derivatives *)
             Quiet[modelFprime=Evaluate[(cnf[
             	params, 
 				modelfunction/. variableRules, {},
 				Apply[Sequence,cnfopts]]/.cnf->Experimental`CreateNumericalFunction)&[##]["Gradient"[params/. parameterRules]]]&;
 			 modelFprimeprime=Evaluate[(cnf[
             	params, 
 				modelfunction/. variableRules, {},
 				Apply[Sequence,cnfopts]]/.cnf->Experimental`CreateNumericalFunction)&[##]["Hessian"[params/. parameterRules]]] &;,
 				{Experimental`NumericalFunction::nlnum, Experimental`NumericalFunction::nnum,Experimental`NumericalFunction::symd}]
 			],
 			(* grad is not Automatic and is of valid length *)
 			modelFprime = 
                  Evaluate[(grad /. parameterRules) /. variableRules]&;
            modelprimeprime = If[VectorQ[grad]&&Length[grad]===Length[params],
            	Map[D[grad, #]&, params],False];
            If[modelprimeprime=!=False&&FreeQ[modelprimeprime, Derivative],
                (* Dimensions[modelFprime[[2]]] = {p, p} *)
                modelFprimeprime = Evaluate[(modelprimeprime /. parameterRules) /.
                                variableRules]&,
                (* modelprimeprime is NOT FreeQ of Derivative, so evaluate the Hessian numerically *)
                (* quiet nnum messages that will correctly be generated in the process of setting up a pure functions for the derivatives *)
                modelFprimeprime=Quiet[Evaluate[(cnf[params, 
 					modelfunction/. variableRules, {},
 					Gradient->(grad/.variableRules),
 					Apply[Sequence,cnfopts]]/.cnf->Experimental`CreateNumericalFunction)&[##]["Hessian"[params/. parameterRules]]]&,
 					{Experimental`NumericalFunction::nlnum, Experimental`NumericalFunction::nnum,Experimental`NumericalFunction::symd}];
             ]];
         
         
          (* Dimensions[predictedResponse] = {n} *)
          predictedResponse = Apply[modelF, pts, {1}];
          fitResiduals = response - predictedResponse;
          {totalDOF, modelDOF} = {Length[pts], Length[params]};
          errorDOF = totalDOF - modelDOF;
          insufficientdataflag=(errorDOF <= 0 && 
  			Intersection[{ANOVATable, AsymptoticCorrelationMatrix, 
     			AsymptoticCovarianceMatrix, EstimatedVariance, FitCurvatureTable, 
     			MeanPredictionCITable, ParameterBias, ParameterCITable, 
     			ParameterConfidenceRegion, ParameterTable, SinglePredictionCITable, 
     			StandardizedResiduals}, startreport] =!= {});
          If[insufficientdataflag
     		, 
 			Message[NonlinearRegress::dof, modelDOF, totalDOF]; 
 			finishreport=Replace[finishreport, 
  				Map[Rule[#, Rule[#, $Failed]] &, 
  				{ANOVATable, AsymptoticCorrelationMatrix, 
  				AsymptoticCovarianceMatrix, EstimatedVariance, FitCurvatureTable, 
    			MeanPredictionCITable, ParameterBias, ParameterCITable, 
    			ParameterConfidenceRegion, ParameterTable, SinglePredictionCITable, 
    			StandardizedResiduals}], 1]
    		,
    		If[errorDOF>0,
          	responseVariance = errorSS/errorDOF;

          	(* Dimensions[ndesignMatrix] = {n, p} *)
          	pP = Precision[pts];
          	npts = If[pP === Infinity, N[pts, wp], pts];
          	Quiet[If[modelFprime === $Failed,
             ndesignMatrix = $Failed,
             ndesignMatrix = Block[{modelderval},
             	Catch[Apply[If[FreeQ[modelderval = modelFprime[##], 
             		Indeterminate|Underflow[]|Overflow[] ], modelderval, Throw[$Failed]] &, npts, {1}]]];
             pP = Precision[ndesignMatrix];
             If[pP === Infinity, ndesignMatrix = N[ndesignMatrix, wp]]                
          	],Experimental`NumericalFunction::symd];

          	pP = Internal`EffectivePrecision[response];
          	weightedResponse = sqrtweights response;
          	nweightedResponse = If[pP === Infinity, 
                N[weightedResponse, wp], weightedResponse];
          	meanWeightedResponse = 
                sqrtweights (response.weights) / Apply[Plus,weights];
          	nmeanWeightedResponse = If[pP === Infinity,
                N[meanWeightedResponse, wp], meanWeightedResponse];
          
          	uncorrectedTotalSS = nweightedResponse.nweightedResponse;
          	If[!FreeQ[uncorrectedTotalSS,Complex]&&
          	  Im[uncorrectedTotalSS]<(10^-AccuracyGoal+Re[modelSS]*10^-PrecisionGoal)/.
          		{opts}/.Options[NonlinearRegress]/.{Automatic->wp/2}/.{MachinePrecision->$MachinePrecision},
          	uncorrectedTotalSS=Re[uncorrectedTotalSS]];
          	modelSS = uncorrectedTotalSS - errorSS;
          	If[!FreeQ[modelSS,Complex]||modelSS < 0,
             	Message[NonlinearRegress::bdfit]];

          	correctedTotalSS = (nweightedResponse - nmeanWeightedResponse).
                (nweightedResponse - nmeanWeightedResponse)]];
            finishreport = Replace[finishreport, {
                EstimatedVariance ->
                 (EstimatedVariance -> responseVariance),
                BestFit -> (BestFit -> (modelfunction /. parameterRules)),
                BestFitParameters ->
                 (BestFitParameters -> parameterRules),
                StartingParameters -> (StartingParameters ->
                        Which[start===$Failed,
                           {},
                           VectorQ[start, numberQ],
                            (* single start point *)
                           Thread[Rule[params, start]],
                           True,
                           (* two start points *)
                           Map[Thread[Rule[params, #]] &, start] ])
               }, 1];

          (* ================= add predictedResponse to report ============ *)

          (* Only add to report if necessary.... depending on the length of
                the data, predictedResponse could be a very long vector. *)
          If[MemberQ[startreport, PredictedResponse],
                finishreport = Replace[finishreport, PredictedResponse ->
                        (PredictedResponse -> predictedResponse), 1]
          ];


          (* ================= add fitResiduals to report ================= *)

          (* Only add to report if necessary.... depending on the length of
                the data, fitResiduals could be a very long vector. *)
          If[MemberQ[startreport, FitResiduals],
                finishreport = Replace[finishreport, FitResiduals ->
                        (FitResiduals -> fitResiduals), 1]
          ];


          (* ===================== compute ANOVA table ===================== *)
        
          If[MemberQ[startreport, ANOVATable]&&Not[insufficientdataflag],
             finishreport = Replace[finishreport, ANOVATable -> (ANOVATable ->
                 TableForm[{{modelDOF, modelSS, modelSS/modelDOF},
                                    {errorDOF, errorSS, errorSS/errorDOF},
                                    {totalDOF, uncorrectedTotalSS},
                                   {totalDOF-1, correctedTotalSS}},
                                 TableHeadings->
                {{"Model", "Error", "Uncorrected Total", "Corrected Total"},
                 {"DF", "SumOfSq", "MeanSq"}} ]), 1]
          ];


          (* ================== compute things involving ================== *)
          (* ============= the SVD of the asymp design matrix ============= *)

          If[Or @@ Map[MemberQ[startreport, #]&,
                {AsymptoticCovarianceMatrix, AsymptoticCorrelationMatrix,
                 ParameterConfidenceRegion,
                 ParameterConfidenceRegion[_?VectorQ],
                 HatDiagonal, ParameterTable, ParameterCITable,
                 SinglePredictionCITable, MeanPredictionCITable,
                 StandardizedResiduals}],

           (* Regardless of whether ndesignMatrix === $Failed or the
              singular value decomposition of nweightedDesignMatrix works
              or not, we need to know for which subsets of the
              parameters joint confidence regions are required.  If
              ndesignMatrix and the singular value decomposition work,
              we will return  meaningful confidence regions.  If not, we will
              return $Failed. *)
           If[MemberQ[startreport, ParameterConfidenceRegion[_?VectorQ]],
                  arglist = Map[First, Select[startreport,
                              MatchQ[#,
                                 ParameterConfidenceRegion[l_List] /;
                                 Apply[And, Map[MemberQ[params, #]&, l]]]& ]];
           ];

		  If[Not[insufficientdataflag||
		  		(errorDOF<=0&&MemberQ[startreport,HatDiagonal])],
           If[ndesignMatrix =!= $Failed,
             Block[{names, nweightedDesignMatrix, 
                    svd, u, w, v, covMatrix, corMatrix, hatMatrix,
                    hatdiag, standResiduals, parmci, predci,
                    parmse, (* list of asympt. stan. errors of parm estimates *)
                    predse, (* list of asympt. stan. errors of pred response *)
                    tRatio, tPValues},

               (* names = Map[ToString, params]; *)         
               names = params; 
               nweightedDesignMatrix = sqrtweights ndesignMatrix;
               pP = Precision[nweightedDesignMatrix];
               If[pP === Infinity,
                  nweightedDesignMatrix = N[nweightedDesignMatrix, wp]];        
               svd = CompactSVD[nweightedDesignMatrix, Tolerance -> tol];
               If[!(Head[svd] === List && Length[svd] === 3),
                  
                  (* BAD SVD *)
                  Message[NonlinearRegress::nosvd];
                  (* Need to assign everything that depended on a 
                        good singular value decomposition to $Failed. *)
                  finishreport = Replace[finishreport, {
                          AsymptoticCovarianceMatrix -> 
                                    (AsymptoticCovarianceMatrix -> $Failed),
                          AsymptoticCorrelationMatrix ->
                                    (AsymptoticCorrelationMatrix -> $Failed),
                        HatDiagonal -> (HatDiagonal -> $Failed),
                        MeanPredictionCITable ->
                                (MeanPredictionCITable -> $Failed),
                        ParameterCITable ->
                                (ParameterCITable -> $Failed),
                          ParameterConfidenceRegion ->
                                (ParameterConfidenceRegion -> $Failed),
                          ParameterTable ->
                                (ParameterTable -> $Failed),
                          SinglePredictionCITable ->
                                (SinglePredictionCITable -> $Failed),
                          StandardizedResiduals ->
                                (StandardizedResiduals -> $Failed)
                        }, 1];
                  Scan[(basisSubset = #;
                        finishreport = Replace[finishreport,
                                   (ParameterConfidenceRegion[
                                        basisSubset] ->
                                   (ParameterConfidenceRegion[
                                        basisSubset] -> $Failed)), 1]
                                )&, arglist],

                  (* GOOD SVD *)
                  {u, w, v} = svd;
                  covMatrix = responseVariance (v . (Transpose[v] / w^2));
                  (* if imaginary zeros occur, remove them to avoid messages 
                  	 generated by inequalities for later report values *)
                  If[Not[FreeQ[covMatrix,Complex]]&&covMatrix==Re[covMatrix],
                  	covMatrix=Re[covMatrix]];
                  finishreport = Replace[finishreport,AsymptoticCovarianceMatrix ->
                       (AsymptoticCovarianceMatrix -> MatrixForm[covMatrix]), 1];
                  parmse = Sqrt[Diagonal[covMatrix]];
                  If[Apply[Or, Map[TrueQ[#==0]&, parmse]],
                     Message[NonlinearRegress::zerocov,
                         AsymptoticCorrelationMatrix];
                     finishreport = Replace[finishreport, AsymptoticCorrelationMatrix ->
                        (AsymptoticCorrelationMatrix -> $Failed), 1],
                     corMatrix = Transpose[Transpose[covMatrix/parmse] /
                         parmse];
                     finishreport = Replace[finishreport, AsymptoticCorrelationMatrix ->
                        (AsymptoticCorrelationMatrix -> MatrixForm[corMatrix]), 1]
                  ];        
           

                  (* NOTE: Calculate hatMatrix only when you know that the
                           HatDiagonal diagnostic is needed.
                           hatMatrix is an n x n matrix, which can be
                           quite large. *)
                  If[MemberQ[startreport, HatDiagonal] ||
                     MemberQ[startreport, StandardizedResiduals],
                     (* NOTE:  The following manner of calculating hatMatrix
                                unnecessarily constructs
                                DiagonalMatrix[sqrtweights]
                                (an n x n matrix) and is slow.
                       hatMatrix = ndesignMatrix .
                                (Transpose[v].DiagonalMatrix[1/w].u.
                                 DiagonalMatrix[sqrtweights])
                     *)
                     hatMatrix = Map[sqrtweights # &, Transpose[u]];
                     hatMatrix =
                         v.DiagonalMatrix[1/w].hatMatrix;
                     hatMatrix = ndesignMatrix . hatMatrix;
                     hatdiag = Diagonal[hatMatrix];
                     (* 8/98: changed standResiduals from
                        fitResiduals/Sqrt[responseVariance (1-hatdiag)]
                        to
                        fitResiduals/Sqrt[responseVariance (1-hatdiag)/weights]
                     *)
                     standResiduals = If[TrueQ[responseVariance==0],
                          Message[NonlinearRegress::zerovar,
                                 StandardizedResiduals];
                          $Failed,
                          fitResiduals /
                                 Sqrt[responseVariance (1-hatdiag)/weights]
                     ]; 
                     finishreport = Replace[finishreport, {HatDiagonal ->
                        (HatDiagonal -> hatdiag),
                                           StandardizedResiduals ->
                         (StandardizedResiduals -> standResiduals)}, 1]
                  ];         (* end HatDiagonal & StandardizedResiduals *)
                  If[MemberQ[startreport, ParameterTable],
                        tRatio = Map[If[ TrueQ[#[[2]]==0],
                                         Infinity, #[[1]]/#[[2]] ]&,
                                     Transpose[{parameterEst, parmse}] ];
                        tPValues = Map[CDF[StudentTDistribution[errorDOF],-Abs[#]]&,
                                 tRatio];
                        tPValues = 2 tPValues;
                        finishreport = Replace[finishreport, ParameterTable ->
                                (ParameterTable -> TableForm[
                           Transpose[{parameterEst, parmse, tRatio, tPValues}],
                                TableHeadings -> {names,
                                 {"Estimate", "Asymp. SE", "TStat", "PValue"}}]), 1]
                      ];        (* end ParameterTable *)
                  If[MemberQ[startreport, ParameterCITable],
                     parmci = Apply[N[StudentTCI[#1, #2, errorDOF,
                                   ConfidenceLevel -> cLevel]]&,
                           Transpose[{parameterEst, parmse}], 1];
                     finishreport = Replace[finishreport, ParameterCITable ->
                        (ParameterCITable -> TableForm[
                          Transpose[{parameterEst, parmse, parmci}], 
                          TableDepth -> 2,
                          TableHeadings -> {names,
                                 {"Estimate", "Asymptotic SE", "CI"}}]), 1]
                  ];        (* end ParameterCITable *)
                  If[MemberQ[startreport, MeanPredictionCITable] ||
                     MemberQ[startreport, SinglePredictionCITable],
                     pP = Precision[response];
                     nresponse = If[pP === Infinity, N[response, wp], response];
                     predse = Sqrt[Map[(#.covMatrix.#)&, ndesignMatrix]] ];
                  If[MemberQ[startreport, MeanPredictionCITable],
                     predci = Apply[N[StudentTCI[#1, #2, errorDOF,
                                ConfidenceLevel -> cLevel]]&,
                                Transpose[{predictedResponse, predse}], 1];
                     finishreport = Replace[finishreport, MeanPredictionCITable ->
                      (MeanPredictionCITable -> TableForm[
                      Transpose[{nresponse, predictedResponse, predse, predci}],
                                      TableDepth -> 2, TableHeadings -> {None,
                          {"Observed", "Predicted", "Asymptotic SE", "CI"}}]), 1]
                  ];        (* end MeanPredictionCITable *)
                  If[MemberQ[startreport, SinglePredictionCITable],
                     predse = Sqrt[responseVariance + predse^2];
                     predci = Apply[N[StudentTCI[#1, #2, errorDOF,
                                ConfidenceLevel -> cLevel]]&,
                                Transpose[{predictedResponse, predse}], 1];
                     finishreport = Replace[finishreport, SinglePredictionCITable ->
                      (SinglePredictionCITable -> TableForm[
                      Transpose[{nresponse, predictedResponse, predse, predci}],
                                      TableDepth -> 2, TableHeadings -> {None,
                          {"Observed", "Predicted", "Asymptotic SE", "CI"}}]), 1]
                  ];        (* end SinglePredictionCITable *)
                  If[MemberQ[startreport, ParameterConfidenceRegion] ||
                     MemberQ[startreport,
                         ParameterConfidenceRegion[_?VectorQ]],
                     (* Weisberg, p. 97 *)
                     Block[{ellipsoid, pos,
                         fit = parameterEst, p = modelDOF, n = totalDOF,
                         basisSubset, fitSubset, covMatrixSubset, pSubset},
                        If[MemberQ[startreport, ParameterConfidenceRegion],
                           ellipsoid = ellipsoidalLocus[fit,
                                covMatrix * p  *
                                Quantile[FRatioDistribution[p, n-p], cLevel]];
                           finishreport = Replace[finishreport,
                                 ParameterConfidenceRegion ->
                              (ParameterConfidenceRegion -> ellipsoid), 1]
                        ];
                        If[MemberQ[startreport,
                                ParameterConfidenceRegion[_?VectorQ]],
                           Scan[(basisSubset = #;
                                 pos = Flatten[Map[Position[params, #, 1]&,
                                         basisSubset]];
                                 fitSubset = fit[[pos]];
                                 covMatrixSubset =
                                  Transpose[Transpose[covMatrix[[pos]]][[pos]]];
                                 pSubset = Length[basisSubset];
                                 ellipsoid = ellipsoidalLocus[fitSubset,
                                   covMatrixSubset * pSubset * Quantile[
                                   FRatioDistribution[pSubset, n-p], cLevel]];
                                 finishreport = Replace[finishreport,
                                   (ParameterConfidenceRegion[
                                        basisSubset] ->
                                   (ParameterConfidenceRegion[
                                        basisSubset] ->
                                     ellipsoid)), 1]
                                )&, arglist]
                        ]  (* end If *)
                     ] (* end Block *)
                  ] (* end If *)

                     

               ];        (* end If SVD *)


             ], (* end Block *)



             (* ndesignMatrix === $Failed *)
         Message[NonlinearRegress::nodm]; (* warn user that some diagnostics are bad *)
             finishreport = Replace[finishreport, {
               AsymptoticCovarianceMatrix ->
                (AsymptoticCovarianceMatrix -> $Failed),
               AsymptoticCorrelationMatrix ->
                (AsymptoticCorrelationMatrix -> $Failed),
               HatDiagonal -> (HatDiagonal -> $Failed),
               MeanPredictionCITable -> (MeanPredictionCITable -> $Failed),        
               ParameterCITable -> (ParameterCITable -> $Failed),
               ParameterConfidenceRegion ->
                (ParameterConfidenceRegion -> $Failed),
               ParameterTable -> (ParameterTable -> $Failed),
               SinglePredictionCITable -> (SinglePredictionCITable -> $Failed),
               StandardizedResiduals -> (StandardizedResiduals -> $Failed)
                                 }, 1];
             Scan[(basisSubset = #;
                   finishreport = Replace[finishreport,
                                   (ParameterConfidenceRegion[
                                        basisSubset] ->
                                   (ParameterConfidenceRegion[
                                        basisSubset] -> $Failed)), 1]
                                )&, arglist]
           ](* end If ndesignMatrix =!= $Failed *),
           (* if there are fewer data points than parameters, 
              the hat diagonals are all 1 *)
           If[MemberQ[startreport,HatDiagonal],
           	finishreport=Replace[finishreport,
           		HatDiagonal->(HatDiagonal->N[Table[1,{totalDOF}],wp]),1]
           	]  	
           	](* end If Not[insufficientdataflag]*)
        ];        (* end of SVD of asymptotic design matrix calculations *)
        

          (* ============= compute curvature diagnostics table =========== *)
        
          If[MemberQ[startreport, FitCurvatureTable] ||
             MemberQ[startreport, ParameterBias],
           If[!(responseVariance==0 || modelprimeprime===$Failed),
            If[modelprimeprime === Table[0, {modelDOF}, {modelDOF}],
              {INcurve, PEcurve} = {0, 0};
              locusDOF = modelDOF;        
              bias = Table[0, {modelDOF}],
              (* modelprimeprime is NOT a zero matrix *)
              Block[{radius, V1, V2, qrd, q, r, rsvd, u, w, v,
                     accel, digits, maxcurvetol},
                radius = Sqrt[modelDOF responseVariance];
                If[!FreeQ[radius, Sqrt], radius = N[radius, wp]];
                V1 = radius^(-1) ndesignMatrix;
                V2 = Quiet[radius^(-1) Apply[modelFprimeprime, npts, {1}],
                	Experimental`NumericalFunction::symd];
                pP = Precision[V2];
                If[pP === Infinity, V2 = N[V2, wp]];
                If[Not[FreeQ[V2,Complex]]&&V2==Re[V2],
                	V2=Re[V2]];
                If[Head[qrd = QRDecomposition[V1]] =!= List,
                   Message[NonlinearRegress::noqrd];
                   rinv = $Failed,
                   (* {p x n, p x p} OR {k x n, k x p} where k < p *)        
                   (* clean up imaginary zeros if they occur *)
                   If[Not[FreeQ[qrd,Complex]]&&qrd==Re[qrd],
                   		qrd=Re[qrd]];
                   {q, r} = qrd;
                   locusDOF = Length[r];
                   If[locusDOF =!= modelDOF,
                      Message[NonlinearRegress::rank, locusDOF, modelDOF,
                        locusDOF, errorDOF, cLevel]
                   		];
                   rinv=inverseFromSVD[r, tol];
                      If[rinv === $Failed,
                         Message[NonlinearRegress::noinv]
                         ]
                ]; (* end calc of rinv, q, locusDOF *)
                If[rinv =!= $Failed,
                   If[(locusDOF totalDOF - locusDOF(locusDOF+1)/2) > 500,
                      (* NOTE: for large problems,
                         inform user what the wait is about. *)        
                      Message[NonlinearRegress::crvtab, Length[q],
                                Length[q[[1]]] ] ];
                   q = Join[q, OrthogonalComplement[q]]; (* n x n *)         
                   (* {n x p x p} OR {n x k x k} where k < p *)
                   accel =   q.Map[(Transpose[rinv].#.rinv)&, V2  ];
                   maxcurvetol = If[tol === Automatic,
              digits = Precision[accel];
              If[digits === MachinePrecision || digits < $MachinePrecision,
                  digits = MachinePrecision
              ];
                      SetPrecision[10^(-digits+10), wp],
            (* otherwise tol was passed in *)
                      tol ];
                   INcurve = maxcurve[Drop[accel, locusDOF],
                         maxcurvetol, maxits,Min[wp,Max[Precision[accel],Accuracy[accel]]]];
                   If[Head[INcurve] === List,
                      Message[NonlinearRegress::infail, maxits, maxcurvetol];
                      INcurve = INcurve[[2]]
                   ];
                   PEcurve = maxcurve[Take[accel, locusDOF],
                         maxcurvetol, maxits,Min[wp,Max[Precision[accel],Accuracy[accel]]]];
                   If[Head[PEcurve] === List,
                      Message[NonlinearRegress::pefail, maxits, maxcurvetol];
                      PEcurve = PEcurve[[2]]
                   ];
                   bias = (-1/(2 modelDOF)) rinv . Map[
                                Tr[Diagonal[#]]&,
                                        Take[accel, locusDOF] ]
                ]        (* end If rinv =!= $Failed *)
              ]        (* end Block calc of rinv, INcurve, PEcurve, bias,
                           and locusDOF *)
            ]; (* end If modelprimeprime === Table[0, {modelDOF}, {modelDOF}] *)
            If[rinv === $Failed,
              finishreport = Replace[finishreport, {FitCurvatureTable ->
                        (FitCurvatureTable->$Failed),
                   ParameterBias -> (ParameterBias->$Failed)}, 1],
              finishreport = Replace[finishreport, {FitCurvatureTable ->
                        (FitCurvatureTable -> TableForm[
                           {{INcurve}, {PEcurve},
                             {1/Sqrt[Quantile[FRatioDistribution[locusDOF,
                                 errorDOF], cLevel]]}}, 
                         TableDepth -> 2, TableHeadings -> {
                          {"Max Intrinsic", "Max Parameter-Effects",
                           ToString[100 cLevel]<>" % Confidence Region"},
                          {"Curvature"}}]),
                   ParameterBias -> (ParameterBias->bias)}, 1]
            ], (* end If rinv === $Failed *)
            (* responseVariance==0 || modelprimeprime===$Failed *)
            finishreport = Replace[finishreport, {FitCurvatureTable ->
                        (FitCurvatureTable->$Failed),
                        ParameterBias -> (ParameterBias->$Failed)}, 1]
           ] (* end If !(responseVariance==0 || modelprimeprime===$Failed) *)
          ];        (* end If MemberQ[startreport, FitCurvatureTable] ||
                          MemberQ[startreport, ParameterBias] *)

          If[TrueQ[responseVariance==0],
             If[MemberQ[startreport, FitCurvatureTable],
                Message[NonlinearRegress::zerovar, FitCurvatureTable];        
                finishreport = Replace[finishreport, FitCurvatureTable ->
                        (FitCurvatureTable -> $Failed), 1]];
             If[MemberQ[startreport, ParameterBias],
                Message[NonlinearRegress::zerovar, ParameterBias];        
                finishreport = Replace[finishreport, ParameterBias ->
                        (ParameterBias -> $Failed), 1]]
          ];    (* end If responseVariance==0 *)
        
          finishreport

        ] (* end MakeRegressionReport *)


inverseFromSVD[mat_, tol_] := Block[{svd, newmat, wts, u, w, v},
  {wts, newmat} = normalizeColumns[mat];
  svd = CompactSVD[newmat, Tolerance -> tol];
  If[Not[Head[svd] === List && Length[svd] === 3],
   $Failed,
   (*clean up imaginary zeros if they occur*)
   If[Not[FreeQ[svd, Complex]] && svd == Re[svd], svd = Re[svd]];
   {u, w, v} = svd;
   wts*(v.DiagonalMatrix[1/w].Transpose[u])
   ]]
   
   
normalizeColumns[mat_] := Module[{weights, tmp},
  weights = Table[tmp = Abs[Part[mat, All, i]];
    Sqrt[tmp.tmp], {i, Length[First[mat]]}];
  weights = Map[If[# == 0, N[0, Precision[mat]], 1/#] &, weights];
  {Developer`ToPackedArray[weights], weights*# & /@ mat}]
  
  
(* ================================= maxcurve =============================== *)

(* maxcurve returns the maximum curvature of an acceleration matrix when
   convergence occurs or {$Failed, approximate maximum curvature} if the
   maximum # of iterations is exceeded. *)

maxcurve[A_, tol_, maxiter_,wp_] :=
  Module[{dir, grad, Ad, dAd, thresh = 1 - tol, mc = $Failed},
    (* A is a m x p x p tensor, where p = # of parameters, m = d.o.f. *)
    (* initialize parameter space direction vector *)
    dir = Append[Table[0, {Length[ A[[1,1]] ] - 1}], 1];
	(* quiet messages such as division by 0;
	   in these cases, failed convergence messages will be issued from 
	   the report generation code *)
    Quiet[Block[{$MaxPrecision=wp,$MinPrecision=wp},
    Do[  (* set grad to [dirT A dir]T [A dir] and normalize *)
         Ad = Map[(#.dir)&, A];      (* m x p matrix *)
         dAd = Map[(dir.#)&, Ad];    (* m vector *)
         grad = dAd . Ad;            (* p vector *)
         grad /= Sqrt[grad.grad];    (* normalize *)

         (* at convergence, dir points along grad *)
         If[ Abs[grad.dir] <= thresh,
           dir = 3 grad + dir;       (* update dir *)
           dir /= Sqrt[dir.dir],     (* normalize *)
           mc = Sqrt[dAd.dAd];
           Break[]
         ],

         {maxiter}];

    If[mc === $Failed,
      {$Failed, Sqrt[dAd.dAd]},
      mc]]] 
  ]

(* ============================ OrthogonalComplement ======================== *)

(* OrthogonalComplement[vectors] computes the (n-p) length n vectors
        that form the orthogonal complement to the p length n
        orthonormal input vectors.
   The ith input vector is rotated in the plane defined by the ith and jth
   axes, until it lines up with the ith axis.  This same rotation is
   performed on all input vectors, as well as the rotation
   matrix (originally the n x n identity matrix).  A rotation is done for
   entries j=i+1 to j=n in the ith vector until the ith vector is equal to
   the ith axis.  Since the vectors are orthonormal to begin with, any
   rotations will maintain this property.  Once the ith vector is lined up
   with the ith axis, none of the other vectors will have a component in the
   ith direction.
   This same step is performed for each of the p input vectors.
   The resulting rotation matrix gives the inverse of the matrix composed
   of the original p vectors plus the (n-p) orthogonal complement.  In the
   case of orthonormal matrices, Transpose gives the Inverse.
*)
(* this code has a compiled version broken out for speed. This works
   by testing if the vectors are in the form of a packed array; if so,
   then use the compiled version directly. If not, check if the input
   is machine numbers; if so, pack and try again. If not machine numbers,
   then use the original uncompiled version. *)
OrthogonalComplement[vectors_?Developer`PackedArrayQ] :=
    orthcompcompiled[vectors]

OrthogonalComplement[vectors:{{__?MachineNumberQ}..}] :=
    orthcompcompiled[Developer`ToPackedArray[vectors]]

OrthogonalComplement[vectors_] :=
 Module[{p = Length[vectors], n = Length[vectors[[1]]],
        tempvT = Transpose[vectors], rotationT, i, j, theta, cos, sin, rot},

        (* rotation matrix starts out as identity *)
        rotationT = IdentityMatrix[n];

        (* line up each vector in turn: *)
        Do[
          (* zero out each vector element by rotating the vectors: *)
          Do[
            (
            If[tempvT[[j,i]] != 0,
              (* compute rotation angle to zero jth element in ith vector *)
              theta = ArcTan[tempvT[[i,i]],tempvT[[j,i]]];
                cos = Cos[theta];  sin = Sin[theta];
                rot = {{cos, sin}, {-sin, cos}};
              (* rotate all the vectors by this amount *)
              tempvT[[{i, j}]] = rot.tempvT[[{i, j}]];
              (* rotate the matrix also *)
              rotationT[[{i, j}]] = rot.rotationT[[{i, j}]]
            ] ),
          {j,i+1,n}],
        {i,p}];

        Drop[rotationT, p]

 ] (* end OrthogonalComplement *)

(* most significant different in compiled version is wrapping N
   around the identity matrix to prevent any unwanted unpacking;
   otherwise code is essentially identical *)
orthcompcompiled = Compile[{{vectors, _Real, 2}}, 
    Module[{p = Length[vectors], n = Length[vectors[[1]]], 
        tempvT = Transpose[vectors], rotationT, i, j, theta, cos, sin, rot}, 
      rotationT = N[IdentityMatrix[n]];
      Do[Do[
          If[tempvT[[j, i]] != 0, 
            theta = ArcTan[tempvT[[i, i]], tempvT[[j, i]]];
            cos = Cos[theta];  sin = Sin[theta];
            rot = {{cos, sin}, {-sin, cos}};
            tempvT[[{i, j}]] = rot.tempvT[[{i, j}]];
            rotationT[[{i, j}]] = rot.rotationT[[{i, j}]]],
          {j, i + 1, n}], {i, p}];
      Drop[rotationT, p]]]

(* ====================== miscellaneous utility functions =================== *)

mylength[x_] := If[Head[x]===List,
                   Length[x],
                   (* assume symbolQ *)
                   1]

numberQ[n_] := NumberQ[N[n]]

parameterQ[param_] :=
              MatchQ[param, (_?symbolQ)|
              		{_?symbolQ, start_?numberQ}|
             		{_?symbolQ, _?numberQ,_?numberQ}|
            		{_?symbolQ, {_?numberQ, _?numberQ}}|
            		{_?symbolQ, {_?numberQ, _?numberQ},
                 		_?((numberQ[#] || # === -Infinity)&),
                 		_?((numberQ[#] || # === Infinity)&)}|
             		{_?symbolQ, _?numberQ, _?((numberQ[#] || # === -Infinity)&),
                  		_?((numberQ[#] || # === Infinity)&)}|
                 	{_?symbolQ, _?numberQ, _?numberQ, 
                 		_?((numberQ[#] || # === -Infinity)&),
                  		_?((numberQ[#] || # === Infinity)&)}]

symbolQ[x_] := (Head[x] === Symbol) ||
                (!numberQ[x] && !ListQ[x]) (* can't check for head Symbol 
                                because what about Subscript[a, 1] ? *)
                                
                                
validateParameterBounds[inputparams_, minpos_, maxpos_, valpos_] := 
 Scan[If[Not[inputparams[[minpos]] <= inputparams[[#]] <= 
        	inputparams[[maxpos]]]
      	,
    	Message[NonlinearRegress::srange, inputparams[[#]], inputparams[[1]],
     		inputparams[[minpos]] <= inputparams[[1]] <= inputparams[[maxpos]]]; 
       	Return[$Failed]] &, 
       	valpos]

(* utility to choose starting positions based on a minimum and a maximum
    for the parameter *)

startposns[min_,max_] := 
        (
        {min + (1/3)(max - min), min + (2/3)(max - min)}
        )

(* utility to choose a starting position from those derived with the
    above utility by picking the point with the smallest chi-square. *)

findbeststart[starts_, data_, response_, weights_, model_, vars_, params_] :=
    Module[{up, uparams, paramRules, uv, uvars, varRules, umodel,
            chis, chisq, pairs,
            modelfunction=If[Head[model]===List,model[[1]],model]},
        (* need to use Unique symbols because Function doesn't like
                variables like a[1] or Subscript[a, 1] *)
        uparams = Table[Unique[up], {Length[params]}];
        paramRules = Thread[Rule[params, uparams]];
        uvars = Table[Unique[uv], {Length[vars]}];
        varRules = Thread[Rule[vars, uvars]];
        umodel = modelfunction /. paramRules /. varRules;
        chisq = Function[Evaluate[uparams], 
           Evaluate[Plus @@ MapThread[
             (#3 (#2 - Function[Evaluate[uvars], Evaluate[umodel]] @@ #1)^2) &,
             {data,response,weights}]
           ]
        ];
        pairs = Transpose[{starts,chis = N[Apply[chisq, starts,{1}]]}];
        pairs = Select[pairs, FreeQ[#[[2]], Complex]&];
        If[pairs === {},
                $Failed,
                First[First[Select[pairs, #[[2]] == Min[chis] &]]]
        	]
    ]

(* used by ParameterConfidenceRegion option *)
ellipsoidalLocus[mu_, sigma_] :=
  Module[{esystem, esystemT, sorted, r, dir},
    (* (x-mu).Inverse[sigma].(x-mu) == 1 *)
    If[!FreeQ[esystem = Eigensystem[sigma], Eigensystem], Return[$Failed]];
    (* radii are square roots of eigenvalues *)
    esystemT = Transpose[MapAt[Sqrt[#]&, esystem, 1]];
    (* sort semi-axes from largest to smallest *)
    sorted = Sort[esystemT, #1[[1]] > #2[[1]]&];
    {r, dir} = Transpose[sorted];
    Ellipsoid[mu, r, dir]
  ]


(* ====================================================================== *)

End[]

EndPackage[]


(* :Example:
x: number of days material is stored
y: amount of methane gas produced

data = {{1, 8.2}, {1, 6.6}, {1, 9.8}, {2, 19.7}, {2, 15.7}, {2, 16.0},
        {3, 28.6}, {3, 25.0}, {3, 31.9}, {4, 30.8}, {4, 37.8}, {4, 40.2},
        {5, 40.3}, {5, 42.9}, {5, 32.6}}

NonlinearRegress[data, b0 + b1 x + b2 x^2, {b0, b1, b2}, x, Weights ->
        1/{1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5}^2 ]

{BestFitParameters -> {b0 -> -3.62995, b1 -> 12.421, b2 -> -0.717542},
 
>    ParameterCITable -> 
 
>          Estimate    Asymptotic SE   CI                  , 
      b0   -3.62995    2.53839         {-9.16063, 1.90072}

      b1   12.421      2.67432         {6.59412, 18.2478}

      b2   -0.717542   0.520765        {-1.85219, 0.417108}
 
>    EstimatedVariance -> 1.52011, 
 
>    ANOVATable ->                     DF   SumOfSq   MeanSq , 
                   Model               3    1115.33   371.775

                   Error               12   18.2413   1.52011

                   Uncorrected Total   15   1133.57

                   Corrected Total     14   357.3
 
>    AsymptoticCorrelationMatrix -> 1.          -0.960244   0.901754 , 

                                    -0.960244   1.          -0.975052

                                    0.901754    -0.975052   1.
 
>    FitCurvatureTable ->                           Curvature}
                          Max Intrinsic             0

                          Max Parameter-Effects     0

                          95. % Confidence Region   0.535265

Compare with result from Regress...

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

NonlinearRegress[data, b0 + b1 x + b2 x^2, {b0, b1, b2}, x, Weights ->
        (1/#2^2 &)]

*)


(* :Example:
x: time
y: radiation count

data = {{0, 383.}, {14, 373.}, {43, 348.}, {61, 328.}, {69, 324.}, {74, 317.},
        {86, 307.}, {90, 302.}, {92, 298.}, {117, 280.}, {133, 268.},
        {138, 261.}, {165, 244.}, {224, 200.}, {236, 197.}, {253, 185.},
        {265, 180.}, {404, 120.5}, {434, 112.5}};

Regress[data, {1, time, time^2}, time]

{ParameterTable -> 
 
>             Estimate      SE             TStat      PValue       , 
      1       386.918       1.11748        346.241    0.

      time    -1.01988      0.0134187      -76.0041   0.

          2                                                     -15
      time    0.000891277   0.0000303768   29.3407    2.44249 10
 
>    RSquared -> 0.999428, AdjustedRSquared -> 0.999357, 
 
>    EstimatedVariance -> 4.07858, 
 
>    ANOVATable ->         DF   SumOfSq   MeanSq    FRatio    PValue}
                   Model   2    114087.   57043.3   13986.1   0.

                   Error   16   65.2572   4.07858

                   Total   18   114152.


This model is linear in the parameters, so NonlinearRegress is not needed...

NonlinearRegress[data, b0 + b1 time + b2 time^2, {b0, b1, b2}, time]

{BestFitParameters -> 
 
>     {b0 -> 386.918, b1 -> -1.01988, b2 -> 0.000891277}, 
 
>    ParameterCITable -> 
 
>          Estimate      Asymptotic SE   CI                        , 
      b0   386.918       1.11748         {384.549, 389.287}

      b1   -1.01988      0.0134187       {-1.04832, -0.991432}

      b2   0.000891277   0.0000303768    {0.000826881, 0.000955673}
 
>    EstimatedVariance -> 4.07858, 
 
>    ANOVATable ->                     DF   SumOfSq       MeanSq , 
                                                      6
                   Model               3    1.44465 10    481551.

                   Error               16   65.2572       4.07858

                                                      6
                   Uncorrected Total   19   1.44472 10

                   Corrected Total     18   114152.
 
>    AsymptoticCorrelationMatrix -> 1.          -0.860713   0.73743  , 

                                    -0.860713   1.          -0.956687

                                    0.73743     -0.956687   1.
 
>    FitCurvatureTable ->                           Curvature}
                          Max Intrinsic             0

                          Max Parameter-Effects     0

                          95. % Confidence Region   0.555652

*)

(* :Example:
x: fish age
y: fish length 

data = {{14, 590}, {21, 910}, {28, 1305}, {35, 1730}, {42, 2140}, {49, 2725},
        {56, 2890}, {63, 3685}, {70, 3920}, {77, 4325}, {84, 4410}, {91, 4485},
        {98, 4515}, {105, 4480}, {112, 4520}, {119, 4545}, {126, 4525},
        {133, 4560}, {140, 4565}, {147, 4626}, {154, 4566}};

data1 = Map[({age, length} = #;
        {age, age^2, Max[age-80, 0], Max[age-80, 0]^2, length})&, data]

spline regression: known knot (= 80)

If the knot is known, then the model is linear in the parameters...

Regress[data1, {1, x1, x2, x3, x4}, {x1, x2, x3, x4}]

{ParameterTable -> 
 
>          Estimate     SE          TStat      PValue      , 
      1    -320.994     130.734     -2.45532   0.0258957

                                                       -8
      x1   59.6324      6.13763     9.71586    4.096 10

      x2   0.00832024   0.0626651   0.132773   0.896028

                                                         -7
      x3   -61.4221     6.9967      -8.77873   1.62856 10

      x4   0.0167096    0.0623096   0.268171   0.791995
 
>    RSquared -> 0.997143, AdjustedRSquared -> 0.996428, 
 
>    EstimatedVariance -> 6868.4, 
 
>    ANOVATable ->         DF   SumOfSq       MeanSq        FRatio    PValue}
                                          7             6
                   Model   4    3.83521 10    9.58803 10    1395.96   0.

                   Error   16   109894.       6868.4

                                         7
                   Total   20   3.8462 10

spline regression: unknown knot (= k)

If the knot is unknown, then NonlinearFit/NonlinearRegress are required...

Derivative[0, 1][Max][0, x_] := x UnitStep[x]

NonlinearRegress[data, b0 + b1 x + b2 x^2 + b3 Max[x-k, 0] + b4 Max[x-k, 0]^2, 
	{{b0, -321}, {b1, 60}, {b2, .0083}, {b3, -61}, {b4, .017}, {k, 80}}, x]
        
{BestFitParameters -> 
 
>     {b0 -> -317.91, b1 -> 59.4066, b2 -> 0.0112144, b3 -> -62.0161, 
 
>      b4 -> 0.0179745, k -> 80.0283}, 
 
>    ParameterCITable -> 
 
>          Estimate    Asymptotic SE   CI                   , 
      b0   -317.91     135.204         {-606.092, -29.7289}

      b1   59.4066     6.3458          {45.8808, 72.9323}

      b2   0.0112144   0.0647726       {-0.126845, 0.149274}

      b3   -62.0161    0.00189227      {-62.0201, -62.0121}

      b4   0.0179745   0.0654434       {-0.121515, 0.157464}

      k    80.0283     0.116694        {79.7796, 80.277}
 
>    EstimatedVariance -> 7351.58, 
 
>    ANOVATable ->                     DF   SumOfSq       MeanSq     , 
                                                      8             7
                   Model               6    2.99233 10    4.98722 10

                   Error               15   110274.       7351.58

                                                      8
                   Uncorrected Total   21   2.99344 10

                                                     7
                   Corrected Total     20   3.8462 10
 
>    AsymptoticCorrelationMatrix -> 
 
>     1.          -0.947631   0.883162    -0.591827   -0.736338   -0.580316, 

      -0.947631   1.          -0.982463   0.742232    0.779545    0.730698

      0.883162    -0.982463   1.          -0.835067   -0.738681   -0.824879

      -0.591827   0.742232    -0.835067   1.          0.288683    0.999814

      -0.736338   0.779545    -0.738681   0.288683    1.          0.270148

      -0.580316   0.730698    -0.824879   0.999814    0.270148    1.
 
>    FitCurvatureTable ->                           Curvature}
                          Max Intrinsic             0.0181284

                          Max Parameter-Effects     4.89117

                          95. % Confidence Region   0.587089



*)
