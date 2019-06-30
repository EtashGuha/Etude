(* :Title: Multivariate Statistics *)

(* :Author: Darren Glosemeyer *)

(* :Summary:
This package computes descriptive statistics (location, dispersion, shape,
and association statistics) for a sample represented as a data matrix.
The data matrix is a list of independent identically distributed
vector-valued or multivariate observations.
*)

(* :Context: MultivariateStatistics` *)

(* :Package Version: 1.2 *)

(* :Copyright: Copyright 1993-2010 by Wolfram Research, Inc. *)

(* :History:
	Version 1.0 Merged Statistics`MultiDescriptiveStatistics`, Statistics`MultinormalDistribution`,
		and Statistics`MultiDiscreteDistributions` standard add-ons into this file, 2006,
		Darren Glosemeyer. 
	Version 1.1 moved Ellipsoid to MultivariateStatistics` context from RegressionCommon` 
		in Mathematica version 7.0, 2008,
		Darren Glosemeyer.
	Version 1.2 updates for version 8.0, 2010,
		Darren Glosemeyer.	
	*)

(* :Mathematica Version: 8.0 *)

(* force the MV variant of Ellipsoid usage to be ignored *)
MessageName[Ellipsoid, "usage"];
MessageName[MultivariateStatistics`Ellipsoid, "usage"];
Remove["MultivariateStatistics`Ellipsoid"];

BeginPackage["MultivariateStatistics`",
	{(* needed for ConvexHull *)
	"ComputationalGeometry`"}]

(* following forces loading of kernel files so the symbols can be unprotected *)

Statistics`Library`iComplain; (* load statistics library *)

(* load system file for overloaded distribution *)
MultinormalDistribution; 
MultivariateTDistribution;
HotellingTSquareDistribution;
MultinomialDistribution;

If[FileType[ToFileName[{System`Private`$MessagesDir,$Language},"Usage.m"]]===File,
Select[FindList[ToFileName[{System`Private`$MessagesDir,$Language},"Usage.m"],"MultivariateStatistics`"],
(StringMatchQ[#,StartOfString~~"MultivariateStatistics`*"] &&
!StringMatchQ[#,"MultivariateStatistics`Ellipsoid:*"])&]//ToExpression;
];

Get[ToFileName["MultivariateStatistics","MultiDescriptiveStatistics.m"]]

Get[ToFileName["MultivariateStatistics","MultinormalDistribution.m"]]

Get[ToFileName["MultivariateStatistics","MultiDiscreteDistributions.m"]]

EndPackage[]

