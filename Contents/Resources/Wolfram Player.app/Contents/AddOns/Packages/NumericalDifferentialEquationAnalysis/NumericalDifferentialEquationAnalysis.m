(* ::Package:: *)

(* :Title: Numerical Differential Equation Analysis *)

(* :Author: Jerry B. Keiper, Mark Sofroniou *)

(* :Summary:
This package calculates weights, abscissas and errors for Newton-Cotes quadrature formulas of arbitrary order and for the elementary Gaussian quadrature rule and also finds the order conditions that a Runge-Kutta method must satisfy to be of a particular order.
*)

(* :Context: NumericalDifferentialEquationAnalysis` *)

(* :Package Version: 1.0 *)

(* :Copyright: Copyright 1990-2007 by Wolfram Research, Inc. *)

(* :History:
  Merged NumericalMath`Butcher` and NumericalMath`NewtonCotes` into this file. *)

(* :Mathematica Version: 6.0 *)

BeginPackage["NumericalDifferentialEquationAnalysis`"]

Get[ToFileName["NumericalDifferentialEquationAnalysis", "Butcher.m"]]

Get[ToFileName["NumericalDifferentialEquationAnalysis", "NewtonCotes.m"]]

Get[ToFileName["NumericalDifferentialEquationAnalysis", "GaussianQuadrature.m"]]

EndPackage[]
