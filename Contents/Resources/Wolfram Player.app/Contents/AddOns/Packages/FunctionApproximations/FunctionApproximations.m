(* :Title: Function Approximations *)

(* :Author: Jerry B. Keiper, Mark Sofroniou,
      David Jacobson, Hewlett-Packard Laboratories *)

(* :Summary:
This package provides tools for finding a rational approximation to a differentiable function; numerically finding a root of an analytic function of a single variable; finding an approximate integral from a list of discrete points; numerically integrating expressions containing InterpolatingFunction objects; plotting the order star of an approximating function; finding economized rational approximations.
*)

(* :Context: FunctionApproximations` *)

(* :Package Version: 1.0 *)

(* :Copyright: Copyright 1990-2007 by Wolfram Research, Inc. *)

(* :History:
  Merged NumericalMath`Approximations`, NumericalMath`InterpolateRoot`,
    NumericalMath`ListIntegrate`, NumericalMath`NIntegrateInterpolatingFunct`,
    NumericalMath`OrderStar` into this file. *)

(* :Mathematica Version: 6.0 *)

BeginPackage["FunctionApproximations`"]

Get[ToFileName["FunctionApproximations", "Approximations.m"]]

Get[ToFileName["FunctionApproximations","InterpolateRoot.m"]]

Get[ToFileName["FunctionApproximations","NIntegrateInterpolatingFunct.m"]]

Get[ToFileName["FunctionApproximations","OrderStar.m"]]

Get[ToFileName["FunctionApproximations","EconomizedRationalApproximation.m"]]

EndPackage[]
