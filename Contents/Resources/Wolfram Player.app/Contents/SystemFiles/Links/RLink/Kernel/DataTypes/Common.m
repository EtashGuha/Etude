(* :Title: Common.m *)

(* :Author:
        Leonid Shifrin
        leonids@wolfram.com
*)

(* :Package Version: 1.0 *)

(* :Mathematica Version: 8.0 *)

(* :Copyright: RLink source code (c) 2011-2012, Wolfram Research, Inc. All rights reserved. *)

(* :Discussion:

   This package contains inert heads representing containers for a number of common
   data elements (parts) in R, and symbols representing accessor functions to extract 
   those types of data from various user-defined Mathematica representations of R objects.
   For the accessor functions, it is assumed that specific implementations of extended
   data types (higher - level Mathematica representations of them) will overload these
   heads via UpValues (meaning that the rules governng their behavior are attached to 
   heads representing those data types, not to the accessor symbols themselves). Any 
   other modifications of these symbols are STRONGLY DISCOURAGED, since those can break 
   their functionality.   
   
   This mechanism may allow Mathematica representation of various R functions behave as
   R generic functions.
   
*)




BeginPackage["RLink`DataTypes`Common`"]
(* Exported symbols added here with SymbolName::usage *)  

RNames::usage = "RNames[names__] is a container for a sequence of strings representing names \
for list or vector entries in R and corresponding to \"names\" R attribute";

RRowNames::usage = "RRowNames[data__] is a container for a sequence of strings representing \ 
row names for a table  or a data frame in R and corresponding to \"rownames\" R attribute";

RData::usage = "RData[data__] is a general container for a sequence of data entries for \
various R data types (e.g. data frames, factors, tables)";

RGetNames::usage = "RGetNames[expr_] is a general accessor function to retrieve a list of names \
from expr, which is a Mathematica representation of some R object with attribute \"names\". \
Any specific user-defined type implementation is free to (re)define this function (via UpValues) \
to work properly with this type.";

RGetRowNames::usage = "RGetRowNames[expr_] is a general accessor function to retrieve a list of \
row names from expr, which is a Mathematica representation of some R object with attribute \
\"rownames\". Any specific user-defined type implementation is free to (re)define this function \
(via UpValues) to work properly with this type.";

RGetData::usage = "RGetData[expr_] is a general accessor function to retrieve the main data \
from expr, which is a Mathematica representation of some R object. Any specific user-defined \
type implementation is free to (re)define this function (via UpValues) to work properly with \
this type.";

RGetAttributes::usage = "RGetAttributes[expr_] is a general accessor function to retrieve a set \ 
of attributes from expr, which is a Mathematica representation of some R object. Any specific \
user-defined type implementation is free to (re)define this function (via UpValues) to work \
properly with this type.";


EndPackage[]