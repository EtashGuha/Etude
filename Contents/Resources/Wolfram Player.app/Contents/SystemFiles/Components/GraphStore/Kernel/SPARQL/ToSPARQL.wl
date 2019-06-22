BeginPackage["GraphStore`SPARQL`ToSPARQL`", {"GraphStore`", "GraphStore`SPARQL`"}];
Begin["`Private`"];

ToSPARQL[expr_] := expr //. {
	(* 17.4.1 Functional Forms *)
	(* 17.4.1.1 bound *)
	(* 17.4.1.2 IF *)
	(SPARQLEvaluation[If] | If)[cond_, t_, f_] :> SPARQLEvaluation["IF"][cond, t, f],
	(* 17.4.1.3 COALESCE *)
	(* 17.4.1.4 NOT EXISTS and EXISTS *)
	(* 17.4.1.5 logical-or *)
	(* 17.4.1.6 logical-and *)
	(* 17.4.1.7 RDFterm-equal *)
	(* 17.4.1.8 sameTerm *)
	(* 17.4.1.9 IN *)
	(* 17.4.1.10 NOT IN *)

	(* 17.4.2 Functions on RDF Terms *)
	(* 17.4.2.1 isIRI *)
	(* 17.4.2.2 isBlank *)
	(* 17.4.2.3 isLiteral *)
	(* 17.4.2.4 isNumeric *)
	(* 17.4.2.5 str *)
	(* 17.4.2.6 lang *)
	(* 17.4.2.7 datatype *)
	(* 17.4.2.8 IRI *)
	(* 17.4.2.9 BNODE *)
	(* 17.4.2.10 STRDT *)
	(* 17.4.2.11 STRLANG *)
	(* 17.4.2.12 UUID *)
	(* 17.4.2.13 STRUUID *)
	SPARQLEvaluation[CreateUUID][] :> SPARQLEvaluation["STRUUID"][],

	(* 17.4.3 Functions on Strings *)
	(* 17.4.3.2 STRLEN *)
	SPARQLEvaluation[StringLength][var_] :> SPARQLEvaluation["STRLEN"][var],
	(* 17.4.3.3 SUBSTR *)
	(* 17.4.3.4 UCASE *)
	SPARQLEvaluation[ToUpperCase][var_] :> SPARQLEvaluation["UCASE"][var],
	(* 17.4.3.5 LCASE *)
	SPARQLEvaluation[ToLowerCase][var_] :> SPARQLEvaluation["LCASE"][var],
	(* 17.4.3.6 STRSTARTS *)
	SPARQLEvaluation[StringStartsQ][var_, patt_String] | SPARQLEvaluation[StringStartsQ[patt_String]][var_] :> SPARQLEvaluation["STRSTARTS"][var, patt],
	(* 17.4.3.7 STRENDS *)
	SPARQLEvaluation[StringEndsQ][var_, patt_String] | SPARQLEvaluation[StringEndsQ[patt_String]][var_] :> SPARQLEvaluation["STRENDS"][var, patt],
	(* 17.4.3.8 CONTAINS *)
	SPARQLEvaluation[StringContainsQ][var_, patt_String] | SPARQLEvaluation[StringContainsQ[patt_String]][var_] :> SPARQLEvaluation["CONTAINS"][var, patt],
	(* 17.4.3.9 STRBEFORE *)
	(* 17.4.3.10 STRAFTER *)
	(* 17.4.3.11 ENCODE_FOR_URI *)
	SPARQLEvaluation[URLEncode][var_] :> SPARQLEvaluation["ENCODE_FOR_URI"][var],
	(* 17.4.3.12 CONCAT *)
	SPARQLEvaluation[StringJoin][args___] :> SPARQLEvaluation["CONCAT"][args],
	(* 17.4.3.13 langMatches *)
	(* 17.4.3.14 REGEX *)
	SPARQLEvaluation[StringContainsQ][var_, RegularExpression[regex_String]] | SPARQLEvaluation[StringContainsQ[RegularExpression[regex_String]]][var_] :> SPARQLEvaluation["REGEX"][var, regex],
	(* 17.4.3.15 REPLACE *)

	(* 17.4.4 Functions on Numerics *)
	(* 17.4.4.1 abs *)
	(SPARQLEvaluation[Abs | RealAbs] | Abs | RealAbs)[var_] :> SPARQLEvaluation["abs"][var],
	(* 17.4.4.2 round *)
	(SPARQLEvaluation[Round] | Round)[var_] :> SPARQLEvaluation["round"][var],
	(* 17.4.4.3 ceil *)
	(SPARQLEvaluation[Ceiling] | Ceiling)[var_] :> SPARQLEvaluation["ceil"][var],
	(* 17.4.4.4 floor *)
	(SPARQLEvaluation[Floor] | Floor)[var_] :> SPARQLEvaluation["floor"][var],
	(* 17.4.4.5 RAND *)
	SPARQLEvaluation[RandomReal][] :> SPARQLEvaluation["RAND"][]

	(* 17.4.5 Functions on Dates and Times *)

	(* 17.4.6 Hash Functions *)
};

End[];
EndPackage[];
