BeginPackage["GraphStore`SPARQL`Operator`", {"GraphStore`", "GraphStore`SPARQL`"}];

Needs["GraphStore`RDF`"];

Begin["`Private`"];


(* 17.3 Operator Mapping *)

(* SPARQL Unary Operators *)
(* XQuery Unary Operators *)
(* ! A *)
EvaluateSPARQLOperator[Not, a_] := fn["not"][a];
(* + A *)
(* - A *)

(* SPARQL Binary Operators *)
(* Logical Connectives *)
(* A || B *)
(* A && B *)

(* XPath Tests *)
(* A = B *)
EvaluateSPARQLOperator[Equal, a_IRI, b_IRI] := a === b;
EvaluateSPARQLOperator[Equal, a_?numericQ, b_?numericQ] := a == b;
EvaluateSPARQLOperator[Equal, a_?StringQ, b_?StringQ] := a === b;
EvaluateSPARQLOperator[Equal, a_?BooleanQ, b_?BooleanQ] := a === b;
EvaluateSPARQLOperator[Equal, a_?DateObjectQ, b_?DateObjectQ] := a == b;
(* A != B *)
EvaluateSPARQLOperator[Unequal, a_IRI, b_IRI] := a =!= b;
EvaluateSPARQLOperator[Unequal, a_?numericQ, b_?numericQ] := a != b;
EvaluateSPARQLOperator[Unequal, a_?StringQ, b_?StringQ] := a =!= b;
EvaluateSPARQLOperator[Unequal, a_?BooleanQ, b_?BooleanQ] := a =!= b;
EvaluateSPARQLOperator[Unequal, a_?DateObjectQ, b_?DateObjectQ] := a != b;
(* A < B *)
EvaluateSPARQLOperator[Less, a_?numericQ, b_?numericQ] := a < b;
EvaluateSPARQLOperator[Less, a_?StringQ, b_?StringQ] := fn["compare"][a, b] == -1;
EvaluateSPARQLOperator[Less, a_?BooleanQ, b_?BooleanQ] := op["boolean-less-than"][a, b];
EvaluateSPARQLOperator[Less, a_?DateObjectQ, b_?DateObjectQ] := a < b;
(* A > B *)
EvaluateSPARQLOperator[Greater, a_?numericQ, b_?numericQ] := a > b;
EvaluateSPARQLOperator[Greater, a_?StringQ, b_?StringQ] := fn["compare"][a, b] == 1;
EvaluateSPARQLOperator[Greater, a_?BooleanQ, b_?BooleanQ] := op["boolean-greater-than"][a, b];
EvaluateSPARQLOperator[Greater, a_?DateObjectQ, b_?DateObjectQ] := a > b;
(* A <= B *)
EvaluateSPARQLOperator[LessEqual, a_?numericQ, b_?numericQ] := a <= b;
EvaluateSPARQLOperator[LessEqual, a_?StringQ, b_?StringQ] := fn["not"][fn["compare"][a, b] == 1];
EvaluateSPARQLOperator[LessEqual, a_?BooleanQ, b_?BooleanQ] := fn["not"][op["boolean-greater-than"][a, b]];
EvaluateSPARQLOperator[LessEqual, a_?DateObjectQ, b_?DateObjectQ] := a <= b;
(* A >= B *)
EvaluateSPARQLOperator[GreaterEqual, a_?numericQ, b_?numericQ] := a >= b;
EvaluateSPARQLOperator[GreaterEqual, a_?StringQ, b_?StringQ] := fn["not"][fn["compare"][a, b] == -1];
EvaluateSPARQLOperator[GreaterEqual, a_?BooleanQ, b_?BooleanQ] := fn["not"][op["boolean-less-than"][a, b]];
EvaluateSPARQLOperator[GreaterEqual, a_?DateObjectQ, b_?DateObjectQ] := a >= b;

(* XPath Arithmetic *)
(* A * B *)
EvaluateSPARQLOperator[Times, a_, 1 / b_] := EvaluateSPARQLOperator[Divide, a, b];
EvaluateSPARQLOperator[Times, a_?numericQ, b_?numericQ] := a * b;
(* A / B *)
EvaluateSPARQLOperator[Divide, a_?numericQ, b_?numericQ] := N[a / b];
(* A + B *)
EvaluateSPARQLOperator[Plus, a_, -b_] := EvaluateSPARQLOperator[Subtract, a, b];
EvaluateSPARQLOperator[Plus, a_?numericQ, b_?numericQ] := a + b;
(* A - B *)
EvaluateSPARQLOperator[Subtract, a_?numericQ, b_?numericQ] := a - b;

(* SPARQL Tests *)
(* A = B *)
EvaluateSPARQLOperator[Equal, a_, b_] := EvaluateSPARQLFunction["RDFTERM-EQUAL", a, b];
(* A != B *)
EvaluateSPARQLOperator[Unequal, a_, b_] := fn["not"][EvaluateSPARQLFunction["RDFTERM-EQUAL", a, b]];

EvaluateSPARQLOperator[o_, x___, r_Rational, y___] := EvaluateSPARQLOperator[o, x, N[r], y];
EvaluateSPARQLOperator[o_, x___, RDFLiteral[s_, IRI[dt_?StringQ]], y___] := EvaluateSPARQLOperator[o, x, RDFLiteral[s, dt], y];
EvaluateSPARQLOperator[o_, x___, lt_RDFLiteral, y___] := With[{nlt = FromRDFLiteral[lt]}, EvaluateSPARQLOperator[o, x, nlt, y] /; nlt =!= lt];

EvaluateSPARQLOperator[__] := $Failed;


numericQ[_Integer] := True;
numericQ[_Real] := True;
numericQ[_Rational] := True;
numericQ[_?QuantityQ] := True;
numericQ[_] := False;


(* XPath functions *)
fn[f_String][args___] := EvaluateSPARQLFunction[IRI["http://www.w3.org/2005/xpath-functions#" <> f], args];
fn[___][___] := $Failed;


(* XPath operators *)

(* 9.2.2 op:boolean-less-than *)
op["boolean-less-than"][arg1_?BooleanQ, arg2_?BooleanQ] := arg1 === False && arg2 === True;
(* 9.2.3 op:boolean-greater-than *)
op["boolean-greater-than"][arg1_?BooleanQ, arg2_?BooleanQ] := arg1 === True && arg2 === False;

op[___][___] := $Failed;

End[];
EndPackage[];
