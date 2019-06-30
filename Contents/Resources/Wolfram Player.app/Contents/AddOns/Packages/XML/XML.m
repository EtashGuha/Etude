(* :Title: XML *)

(* :Author: Wolfram Research *)

(* :Copyright: Copyright 2001-2007 Wolfram Research, Inc *)

(* :Mathematica Version: 6.0 *)

(* :Package Version: 1.0 *)

(* :History: 

*)

(* :Summary: 
	This package simply adds XML`, XML`Parser, XML`NotebookML`, XML`MathML` contexts
	to $ContextPath so that user has access functions used for importing, exporting,
	and manipulating XML.
*)

(* :Context: XML`*)

(* :Keywords: 
	
	XML`FromSymbolicXML, XML`RawXML, XML`SymbolicXMLErrors, XML`SymbolicXMLQ, 
	XML`ToCompactXML, XML`ToSymbolicXML, XML`ToVerboseXML, 
	
	XML`Parser`InitializeXMLParser, XML`Parser`ReleaseXMLParser, XML`Parser`XMLGet,
	XML`Parser`XMLGetString, XML`Parser`XMLParser, 
	
	XML`NotebookML`ExpressionToSymbolicExpressionML,
	XML`NotebookML`NotebookToSymbolicNotebookML, 
	XML`NotebookML`SymbolicExpressionMLToExpression, 
	XML`NotebookML`SymbolicNotebookMLToNotebook,
	
	XML`MathML`BoxesToMathML, XML`MathML`BoxesToSymbolicMathML, XML`MathML`ExpressionToMathML,
	XML`MathML`ExpressionToSymbolicMathML, XML`MathML`MathMLToBoxes,
	XML`MathML`MathMLToExpression, XML`MathML`SymbolicMathMLToBoxes, 
	XML`MathML`SymbolicMathMLToExpression

*)

(* :Requirements: *)

(* :Warnings: *)

(* :Discussion: 
	The source code for the functions in the contexts mentioned above are not
	available for the public.
*)

(* :Sources: *)

BeginPackage["XML`"]
(* intentionally left blank *)
EndPackage[]

(*	This code must remain after the BeginPackage/EndPackage *)
If[Not[MemberQ[System`$ContextPath, #]], PrependTo[System`$ContextPath, #]] & /@ 
	{"XML`MathML`", "XML`NotebookML`", "XML`Parser`", "XML`"};
