
(* :Title: RDataTypesTools.m *)

(* :Author:
        Leonid Shifrin
        leonids@wolfram.com
*)

(* :Package Version: 1.0 *)

(* :Mathematica Version: 8.0 *)

(* :Copyright: RLink source code (c) 2011-2012, Wolfram Research, Inc. All rights reserved. *)

(* :Discussion:

	This package contains functions useful for construction and registraction of 
	new Mathematica representations for certain R data types. In addition to functions
	used to define and register such types with RLink, it contains a number of
	convenience (helper) functions, which are handy for construction of such new
	representations.
	
	This package is NOT a part of public API, in the sense that while it is necessary 
	to import it into files where new types definitions and implementations reside,
	it is not necessary to export it on the $ContextPath, so private imports are 
	recommended.   
	
*)




BeginPackage["RLink`RDataTypeTools`",{"RLink`"}]
(* Exported symbols added here with SymbolName::usage *)  

RExtractAttribute::usage = "RExtractAttribute[att_RAttributes, attrib_String] extracts a value \
of an attribute named attrib from an attribute container  RAttributes. If an attribute with \
this name is not present, $Failed is returned";

RInstanceOf::usage = "RInstanceOf[cl_String][expr_]  returns True if an R Object represented \
by expr (which should have a head RObject) has the value of its \"class\" attribute equal to \
cl, and False otherwise";

RRemoveAttributes::usage = "RRemoveAttributes[atts_RAttributes, attNames_List] removes key-value \
pairs (delayed rules) corresponding to attributes in the attNames list, from the attributes \
container atts";

RRemoveAttributesComplete::usage = "RRemoveAttributesComplete[atts_RAttributes, attNames_List] \
does the same as RRemoveAttributes, but also removes an attribute container itself in cases \
when it only contains the attributes in attNames list, and no other attributes";

RAddAttributes::usage = "RAddAttributes[a_RAttributes, atts:{(_String:>_)..}] adds a number of \
attributes (key - value pairs represented as delayed rules) in atts to an attributes container \
a";

RGetAllAttributes::usage = "RGetAllAttributes[a : (_RAttributes | None) : None] extracts all \
attributes (key - value pairs represented as delayed rules) from an attribute container a, \
and places them in a list. If the attribute container is empty or None, an empty list is 
returned.";


Begin["`Private`"] (* Begin Private Context *) 



ClearAll[RExtractAttribute];
RExtractAttribute[att_RAttributes, attrib_String] :=
  attrib /. Append[List @@ att, _ -> $Failed];
  
RExtractAttribute[RObject[_, att_RAttributes], attrib_String] :=
  RExtractAttribute[att, attrib];


ClearAll[RInstanceOf];
RInstanceOf[cl_String][x : (_RObject | _RAttributes)] :=
  RExtractAttribute[x, "class"] === {cl};
 
 
ClearAll[RRemoveAttributes]; 
RRemoveAttributes[atts_RAttributes, attNames_List] :=  
  	DeleteCases[atts, (Alternatives @@ attNames) :> _]

  	
  
ClearAll[RRemoveAttributesComplete];
RRemoveAttributesComplete[atts_RAttributes, attNames_List] :=
  Replace[
  	RRemoveAttributes[atts, attNames],
	RAttributes[]:>Sequence[]
  ];
  
  
ClearAll[RAddAttributes]  
RAddAttributes[a_RAttributes, atts:{(_String:>_)..}]:=
	Join[RAttributes@@atts,a];
	
RAddAttributes[None, atts:{(_String:>_)..}]:=
	RAttributes@@atts;
	

ClearAll[RGetAllAttributes];
RGetAllAttributes[a : (_RAttributes | None) : None]:=	
	Replace[a,{None->{},RAttributes[atts__]:>{atts}}];
  

End[] (* End Private Context *)

EndPackage[]


