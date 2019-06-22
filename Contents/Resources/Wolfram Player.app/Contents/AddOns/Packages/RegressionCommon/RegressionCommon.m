(* ::Package:: *)

(*:Version: Mathematica 6.0 *)

(*:Name: Regression Common Functions Package *)

(*:Context: RegressionCommon` *)

(*:Title: Regression Common Functions Package *)

(*:Author:
  Darren Glosemeyer (Wolfram Research), 2006
*)

(*:Copyright: Copyright 2006-2007, Wolfram Research, Inc. *)

(*:History:
   Original version, 2006.
*)

(*:Reference: Usage messages only. *)

(*:Summary:
This package defines symbols used in multiple statistical packages. 
It is effectively a merger of Statistics`Common`* files used in Mathematica
versions 5.2 and prior with symbols that have been moved to the kernel or 
made obsolete removed.
*)

(*:Requirements: No special system requirements. *)

(*:Warning: None. *)

(*:Sources: Basic statistics texts. *)

BeginPackage["RegressionCommon`", "MultivariateStatistics`"]

If[ Not@ValueQ[CovarianceMatrix::usage], 
CovarianceMatrix::usage="CovarianceMatrix is a possible RegressionReport \
value for Regress and DesignedRegress."]

If[ Not@ValueQ[CorrelationMatrix::usage],
CorrelationMatrix::usage="CorrelationMatrix is a possible RegressionReport \
value for Regress and DesignedRegress."]


(* ===================== common regression functions ====================== *)

If[ Not@ValueQ[RegressionReportValues::usage],
RegressionReportValues::usage =
"RegressionReportValues[regfcn] gives a list of valid values that may be \
included in the RegressionReport list input as an option to regfcn; \
regfcn specifies a regression function like Regress or NonlinearRegress."]


(* ===================== regression function options ====================== *)


If[ Not@ValueQ[RegressionReport::usage],
RegressionReport::usage =
"RegressionReport is an option to regression functions and specifies a \
statistic or a list of statistics to be reported about the fit. \
RegressionReportValues[regfcn] specifies valid statistics for the \
regression function regfcn."]


(* ==================== RegressionReport option values ===================== *)

If[ Not@ValueQ[SummaryReport::usage],
SummaryReport::usage="SummaryReport is a possible value for the RegressionReport option \
for regression functions."]

If[ Not@ValueQ[AdjustedRSquared::usage],
AdjustedRSquared::usage =
"AdjustedRSquared is used in the output of regression functions to identify \
the multiple correlation coefficient adjusted for the number of degrees of \
freedom in the fit."]

If[ Not@ValueQ[ANOVATable::usage],
ANOVATable::usage =
"ANOVATable is used in the output of regression and ANOVA functions to identify \
the analysis of variance table."]

If[ Not@ValueQ[BestFit::usage],
BestFit::usage =
"BestFit is used in the output of regression functions to identify the best \
fit."]

If[ Not@ValueQ[BestFitParameters::usage],
BestFitParameters::usage =
"BestFitParameters is used in the output of regression functions to identify the \
list of parameter estimates that give the best (least squares) fit."]

If[ Not@ValueQ[BestFitParametersDelta::usage],
BestFitParametersDelta::usage =
"BestFitParametersDelta is used in the output of linear regression functions \
to identify a list of parameter estimate influence diagnostics, p associated \
with each of the n data points, where p is the number of parameters in the \
model. The ith p-vector gives the standardized differences in the parameter
estimates resulting from the omission of a data point. If \
PredictedResponseDelta indicates that the ith point is influential, then a \
large absolute value (> 2/Sqrt[n]) for the jth element in the ith p-vector of \
BestFitParametersDelta indicates that the jth parameter is heavily \
influenced by the ith point. (Kuh and Welsch call this diagnostic matrix \
DFBETAS.)"]

If[ Not@ValueQ[CatcherMatrix::usage],
CatcherMatrix::usage =
"CatcherMatrix is used in the output of linear regression functions to identify \
the so-called `matrix of catchers' C. If y is the response vector and b is the \
estimated parameter vector, then b = C . y. This matrix can be used to \
compute regression diagnostics. Each row of C catches all the information \
the predictors provide about the corresponding element of the parameter \
vector b."]

If[ Not@ValueQ[CoefficientOfVariation::usage],
CoefficientOfVariation::usage="CoefficientOfVariation is used in the output of linear regression functions \
to identify the ratio of the estimated error standard deviation (residual \
root mean square) to the mean of the response variable."]

If[ Not@ValueQ[CookD::usage],
CookD::usage =
"CookD is used in the output of linear regression functions to identify the \
list of Cook's D influence diagnostics, one associated with each of the data \
points. This diagnostic combines a measure of the remoteness of the point in \
the space of basis functions with a measure of the fit at that point, and \
is a squared distance. Values greater than Quantile[FRatioDistribution[p, n-p], \
.5] may indicate influential points, where n is the number of points and p is \
the number of estimated parameters."]

If[ Not@ValueQ[CovarianceMatrixDetRatio::usage],
CovarianceMatrixDetRatio::usage =
"CovarianceMatrixDetRatio is used in the output of regression functions to \
identify a list of determinant ratio influence diagnostics, one associated with \
each of the data points. The ith diagnostic is given by the ratio of the \
determinant of the parameter covariance matrix obtained by deleting the ith \
row in the original data to the determinant of the parameter covariance matrix \
for the original data. Values outside the interval {1 - 3p/n, 1 + 3p/n} may \
indicate influential points, where n is the number of points and p is the number \
of estimated parameters. (Kuh and Welsch call this diagnostic list COVRATIO.)"]

If[ Not@ValueQ[DurbinWatsonD::usage],
DurbinWatsonD::usage =
"DurbinWatsonD is used in the output of regression functions to identify the \
Durbin-Watson d statistic for testing the existence of a first order \
autoregressive process. A value close to 0 indicates positive correlation and \
a value close to 4 indicates negative correlation. To test positive \
correlation, Durbin-Watson tables are entered with d; to test negative \
correlation, Durbin-Watson tables are entered with (4-d)."]

If[ Not@ValueQ[EigenstructureTable::usage],
EigenstructureTable::usage =
"EigenstructureTable is used in the output of linear regression functions to \
identify a table of information about the eigenstructure of the correlation \
matrix of the nonconstant predictors (basis functions). The table includes \
eigenvalues listed from largest to smallest, the associated condition indices, \
and for each predictor, the proportion of the variance attributable to each \
eigenvalue. Predictors that indicate a large proportion of variance due to a \
particular eigenvalue may be involved in a collinear relationship. Indices \
of 30 to 100 indicate moderate to strong collinearities. When Weights are \
specified, the correlation matrix is based on the weighted observations."]

If[ Not@ValueQ[EstimatedVariance::usage],
EstimatedVariance::usage =
"EstimatedVariance is used in the output of regression functions to \
identify the estimated error variance, or the residual mean square."]

If[ Not@ValueQ[JackknifedVariance::usage],
JackknifedVariance::usage =
"JackknifedVariance is used in the output of regression functions to \
identify the jackknifed estimated error variance vector, each element \
giving the estimated error variance resulting from the omission of the \
corresponding data point. It is given by \
((n-p)*EstimatedVariance - FitResiduals^2/(1-HatDiagonal))/(n-p-1), \
where n is the number of observations and p is the number of estimated \
parameters."]

If[ Not@ValueQ[FitResiduals::usage],
FitResiduals::usage =
"FitResiduals is used in the output of regression functions \
to identify the list of differences between the response data and \
the best fit evaluated at the same abscissa points."]

If[ Not@ValueQ[HatDiagonal::usage],
HatDiagonal::usage =
"HatDiagonal is used in the output of regression functions to identify \
the diagonal of the projection or `hat' matrix H. If y is the response vector \
and yhat is the predicted response vector, then yhat = H . y. The leverage of \
a data point is given by the associated element in the HatDiagonal vector. A \
leverage of zero indicates a point with no influence on the fit, and a leverage \
of one indicates that a degree of freedom has been lost to fitting that point. \
If n is the number of points and p is the number of parameters, 2*p/n is \
often used as the threshold for determining which points have significant \
leverage. For a linear model, the elements of this vector sum to p."]
(* NOTE: from 
  D. A. Belsey, E. Kuh, & R. E. Welsch, Regression Diagnostics, 1980, Wiley.
  "Assume the explanatory variables are independently distributed multinormal.
  While these assumptions are often not valid in practice, they allow one to
  show that (n-p)(h[i] - (1/n))/((1-h[i])(p-1)) is distributed F with p-1 and
  n-p degrees of freedom.  For p>10 and n-p>50 the 95% value for F is < 2,
  making 2*p/n a good rough cutoff.  When p/n > .04, there are so few degrees of
  freedom per parameter that all observations become suspect.  For small p,
  2*p/n tends to call a few too many points to our attentions."
*)

If[ Not@ValueQ[MeanPredictionCITable::usage],
MeanPredictionCITable::usage =
"MeanPredictionCITable is used in the output of regression functions to \
identify a table of confidence intervals for the mean predicted responses, \
one interval for each row in the data or design matrix. The level of the \
confidence interval is specified using the option ConfidenceLevel. The \
interval is found using StudentTCI."]

If[ Not@ValueQ[ParameterCITable::usage],
ParameterCITable::usage =
"ParameterCITable is used in the output of regression functions to \
identify a table of confidence intervals for the parameters. \
The level of the confidence interval is specified using the option \
ConfidenceLevel. The interval is found using StudentTCI."]

If[ Not@ValueQ[ParameterConfidenceRegion::usage],
ParameterConfidenceRegion::usage =
"ParameterConfidenceRegion is used in the output of regression functions \
to specify an elliptically shaped joint confidence region for the parameters. \
It is based on CovarianceMatrix in the case of Regress and \
AsymptoticCovarianceMatrix in the case of NonlinearRegress. The level of the \
confidence interval is specified using the option ConfidenceLevel. \
The option ParameterConfidenceRegion alone specifies the joint confidence \
region of all parameters in the case of Regress, and the asymptotic joint \
confidence region in the case of NonlinearRegress. In the case of Regress, the \
option ParameterConfidenceRegion[list] specifies the confidence region of the \
parameters associated with the basis functions in list, conditioned on the rest \
of the model. In the case of NonlinearRegress, the option \
ParameterConfidenceRegion[list] specifies the asymptotic confidence region \
of the parameters in list, conditioned on the rest of the model."]

If[ Not@ValueQ[ParameterTable::usage],
ParameterTable::usage =
"ParameterTable is used in the output of regression functions to identify a \
table of information about the parameter estimates."]

If[ Not@ValueQ[PartialSumOfSquares::usage],
PartialSumOfSquares::usage =
"PartialSumOfSquares is used in the output of Regress[data, funs, vars] to \
identify a list giving the increase in the model sum of squares \
due to adding the corresponding (nonconstant) basis function (predictor) in \
funs to a model consisting of the remaining basis functions. Partial sum of \
squares is also referred to as type II sum of squares."]

If[ Not@ValueQ[PredictedResponse::usage],
PredictedResponse::usage =
"PredictedResponse is used in the output of regression functions \
to identify the best fit evaluated at the data points."]

If[ Not@ValueQ[PredictedResponseDelta::usage],
PredictedResponseDelta::usage =
"PredictedResponseDelta is used in the output of linear regression functions to \
identify a list of predicted response influence diagnostics, one associated \
with each of the data points. The ith diagnostic gives the standardized \
difference in the predicted response for the ith point, resulting from the \
omission of the ith data point. An absolute value greater than 2*Sqrt[p/n] \
may indicate an influential point, where n is the number of points and p is the \
number of estimated parameters. (Kuh and Welsch call this diagnostic list \
DFFITS.)"]

If[ Not@ValueQ[RSquared::usage],
RSquared::usage =
"RSquared is used in the output of regression functions to identify the square \
of the multiple correlation coefficient."]

If[ Not@ValueQ[SequentialSumOfSquares::usage],
SequentialSumOfSquares::usage =
"SequentialSumOfSquares is used in the output of Regress[data, funs, vars] to \
identify a list giving a partitioning of the model sum of squares into \
component sums of squares due to each (nonconstant) basis function (predictor) \
as it is added sequentially to the model, in the order it appears in funs. \
Sequential sum of squares is also referred to as type I sum of squares."]

If[ Not@ValueQ[SinglePredictionCITable::usage],
SinglePredictionCITable::usage =
"SinglePredictionCITable is used in the output of regression functions to \
identify a table of confidence intervals for the predicted response of \
single observations, one interval for each row in the data or design matrix. \
The level of the confidence interval is specified using the option \
ConfidenceLevel. The interval is found using StudentTCI."]

If[ Not@ValueQ[StandardizedResiduals::usage],
StandardizedResiduals::usage =
"StandardizedResiduals is used in the output of regression functions to \
identify the list of standardized residuals, where the ith residual is \
divided by the standard error for that residual: \
FitResiduals / Sqrt[EstimatedVariance * (1 - HatDiagonal)]. \
In the case of linear regression, each standardized \
residual follows the beta distribution with unity variance."]

If[ Not@ValueQ[StudentizedResiduals::usage],
StudentizedResiduals::usage =
"StudentizedResiduals is used in the output of linear regression functions to \
identify the list of studentized residuals, where the ith residual is \
divided by the standard error for that residual resulting from the omission of \
a data point: \
FitResiduals / Sqrt[JackknifedVariance * (1 - HatDiagonal)]. \
Each studentized residual follows StudentTDistribution[n-p-2]."]

If[ Not@ValueQ[VarianceInflation::usage],
VarianceInflation::usage =
"VarianceInflation is used in the output of linear regression functions to \
identify the list of variance inflation collinearity diagnostics, one associated \
with each of the parameters to be estimated. Values greater than \
1/(1-RSquared) indicate basis functions that may be involved in a collinear \
relationship."]

If[ Not@ValueQ[AsymptoticCovarianceMatrix::usage],
AsymptoticCovarianceMatrix::usage = 
"AsymptoticCovarianceMatrix is used in the output of NonlinearRegress to \
identify the estimated covariance matrix of the fit parameters, for the \
linear model approximating the original nonlinear model. It is given by \
(EstimatedVariance * Inverse[Transpose[J].J]), where J is the n x p Jacobian \
for the nonlinear model (i.e., the design matrix for the linear approximation \
to the nonlinear model)."]

If[ Not@ValueQ[AsymptoticCorrelationMatrix::usage],
AsymptoticCorrelationMatrix::usage = 
"AsymptoticCorrelationMatrix is used in the output of NonlinearRegress to \
identify the estimated correlation matrix of the fit parameters, for the \
linear model approximating the original nonlinear model."]

(* Instead of having an AsymptoticParameterConfidenceRegion symbol based on
    AsymptoticCovarianceMatrix,  NonlinearRegress instead makes use of the
    symbol ParameterConfidenceRegion, just as Regress does.  I thought that
    the symbol AsymptoticParameterConfidenceRegion was simply too long.  ECM *)

If[ Not@ValueQ[FitCurvatureTable::usage],
FitCurvatureTable::usage =
"FitCurvatureTable is used in the output of NonlinearRegress to identify a \
table of information about the maximum relative intrinsic and parameter-effects \
curvatures associated with the least squares solution of a nonlinear fit."]

If[ Not@ValueQ[ParameterBias::usage],
ParameterBias::usage =
"ParameterBias is used in the output of NonlinearRegress to identify a vector \
giving the bias in the parameter estimates. It represents the discrepancy \
between the parameter estimates and the true parameter values."]

If[ Not@ValueQ[StartingParameters::usage],
StartingParameters::usage =
"StartingParameters is used in the output of NonlinearRegress to identify \
the starting point used in the search for parameter values minimizing the \
sum of squares."]


Begin["`Private`"]


(* With the move of Ellipsoid to MultivariateStatistics in version 7, this package only instantiates symbols. *)


End[]

EndPackage[]
