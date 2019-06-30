(* 
   Defines functions for extracting information out of InterpolatingFunction
   objects.
*)

BeginPackage["DifferentialEquations`InterpolatingFunctionAnatomy`"];

InterpolatingFunctionDomain::usage = "InterpolatingFunctionDomain[ifun] gives the domain for the InterpolatingFunction object ifun."

InterpolatingFunctionCoordinates::usage = "InterpolatingFunctionCoordinates[ifun] gives the list of coordinates in each dimension for an InterpolatingFunction object ifun."

InterpolatingFunctionDerivativeOrder::usage = "InterpolatingFunctionDerivativeOrder[ifun] gives the derivative of the interpolated function which will be computed at args when ifun[args] is evaluated for the InterpolatingFunction object ifun."

InterpolatingFunctionInterpolationOrder::usage = "InterpolatingFunctionInterpolationOrder[ifun] gives the interpolation order used in each dimension for computing the interpolated function args when ifun[args] is evaluated for the InterpolatingFunction object ifun."

InterpolatingFunctionGrid::usage = "InterpolatingFunctionGrid[ifun] gives the grid of all of the interpolating points."

InterpolatingFunctionValuesOnGrid::usage = "InterpolatingFunctionValuesOnGrid[ifun] gives the function values on the grid of all of the interpolating points."

Begin["`Private`"]

InterpolatingFunctionDomain[ifun_InterpolatingFunction] := ifun["Domain"]

InterpolatingFunctionCoordinates[ifun_InterpolatingFunction] := ifun["Coordinates"]

InterpolatingFunctionDerivativeOrder[ifun_InterpolatingFunction] := ifun["DerivativeOrder"]

InterpolatingFunctionInterpolationOrder[ifun_InterpolatingFunction] := ifun["InterpolationOrder"]

InterpolatingFunctionGrid[ifun_InterpolatingFunction] := ifun["Grid"]

InterpolatingFunctionValuesOnGrid[ifun_InterpolatingFunction] := ifun["ValuesOnGrid"]

End[(* Private *)]

EndPackage[]
