(* SQL:2003 *)
(* https://github.com/ronsavage/SQL *)
(* https://ronsavage.github.io/SQL/sql-2003-2.bnf.html *)

BeginPackage["GraphStore`Formats`SQL`", {"GraphStore`"}];

Needs["GraphStore`Parsing`"];
Needs["GraphStore`SQL`"];

ExportSQL;
ImportSQL;

Begin["`Private`"];

ExportSQL[args___] := Catch[iExportSQL[args], $failTag, (Message[Export::fmterr, "SQL"]; #) &];
ImportSQL[file_, opts : OptionsPattern[]] := Catch[iImportSQL[file, FilterRules[{opts}, Options[ImportSQL]]], $failTag, (Message[Import::fmterr, "SQL"]; #) &];


fail[___] := Throw[$Failed, $failTag];
SetAttributes[clear, HoldAll];
clear[s_Symbol] := (s[args___] := fail[s, args]);


(* -------------------------------------------------- *)
(* export *)

clear[iExportSQL];
iExportSQL[file_, data_, OptionsPattern[]] := fail[];

(* end export *)
(* -------------------------------------------------- *)


(* -------------------------------------------------- *)
(* import *)

clear[iImportSQL];
iImportSQL[file_String, OptionsPattern[]] := {
	"Data" -> (GrammarApply1[$SQLGrammar, File[file]] // Replace[_?FailureQ :> fail[]])
};

$SQLGrammar := GrammarRules[
	{
		GrammarToken["query specification"],
		GrammarToken["direct SQL statement"]
	},
	{
		(* 5 Lexical Elements *)

		(* 5.1 <SQL terminal character> (p151) *)
		"digit" -> d : DigitCharacter :> d,
		"left paren" -> "(" -> "(",
		"right paren" -> ")" -> ")",
		"asterisk" -> "*" -> "*",
		"plus sign" -> "+" -> "+",
		"comma" -> "," -> ",",
		"minus sign" -> "-" -> "-",
		"period" -> "." -> ".",
		"solidus" -> "/" -> "/",
		"semicolon" -> ";" -> ";",
		"less than operator" -> "<" -> Less,
		"equals operator" -> "=" -> Equal,
		"greater than operator" -> ">" -> Greater,

		(* 5.2 <token> and <separator> (p134) *)
		"regular identifier" -> id : (LetterCharacter ~~ (WordCharacter | "_") ...) :> id,
		"delimited identifier" -> "\"" ~~ id : (LetterCharacter ~~ (WordCharacter | "_") ...) ~~ "\"" :> id,
		"not equals operator" -> "<>" -> Unequal,
		"greater than or equals operator" -> ">=" -> GreaterEqual,
		"less than or equals operator" -> "<=" -> LessEqual,
		"concatenation operator" -> "||" -> "||",

		(* 5.3 <literal> (p143) *)
		"unsigned literal" -> {
			GrammarToken["unsigned numeric literal"],
			GrammarToken["general literal"]
		},
		"general literal" -> {
			GrammarToken["character string literal"](*,
			GrammarToken["national character string literal"],
			GrammarToken["Unicode character string literal"],
			GrammarToken["binary string literal"],
			GrammarToken["datetime literal"],
			GrammarToken["interval literal"],
			GrammarToken["boolean literal"]*)
		},
		"character string literal" -> "'" ~~ lit : Except["'"] .. ~~ "'" :> lit,
		"unsigned numeric literal" -> {
			GrammarToken["exact numeric literal"](*,
			GrammarToken["approximate numeric literal"]*)
		},
		"exact numeric literal" -> {
			GrammarToken["unsigned integer"]
		},
		"sign" -> {
			GrammarToken["plus sign"],
			GrammarToken["minus sign"]
		},
		"unsigned integer" -> d : GrammarToken["digit"] .. :> ToExpression[StringJoin[d]],

		(* 5.4 Names and identifiers (p151) *)
		"identifier" -> id : GrammarToken["actual identifier"] :> SQLIdentifier1[id],
		"actual identifier" -> {
			GrammarToken["regular identifier"],
			GrammarToken["delimited identifier"]
		},
		"table name" -> GrammarToken["local or schema qualified name"],
		"local or schema qualified name" -> GrammarToken["qualified identifier"],
		"qualified identifier" -> GrammarToken["identifier"],
		"column name" -> GrammarToken["identifier"],
		"correlation name" -> GrammarToken["identifier"],
		"query name" -> GrammarToken["identifier"],


		(* 6 Scalar expressions *)

		(* 6.3 <value expression primary> (p174) *)
		"value expression primary" -> {
			GrammarToken["parenthesized value expression"],
			GrammarToken["nonparenthesized value expression primary"]
		},
		"parenthesized value expression" -> FixedOrder[GrammarToken["left paren"], expr : GrammarToken["value expression"], GrammarToken["right paren"]] :> expr,
		"nonparenthesized value expression primary" -> {
			GrammarToken["unsigned value specification"],
			GrammarToken["column reference"],
			GrammarToken["set function specification"],
			(*GrammarToken["window function"],
			GrammarToken["scalar subquery"],*)
			GrammarToken["case expression"](*,
			GrammarToken["cast specification"],
			GrammarToken["field reference"],
			GrammarToken["subtype treatment"],
			GrammarToken["method invocation"],
			GrammarToken["static method invocation"],
			GrammarToken["new specification"],
			GrammarToken["attribute or method reference"],
			GrammarToken["reference resolution"],
			GrammarToken["collection value constructor"],
			GrammarToken["array element reference"],
			GrammarToken["multiset element reference"],
			GrammarToken["routine invocation"],
			GrammarToken["next value expression"]*)
		},

		(* 6.4 <value specification> and <target specification> (p176) *)
		"unsigned value specification" -> {
			GrammarToken["unsigned literal"](*,
			GrammarToken["general value specification"]*)
		},

		(* 6.6 <identifier chain> (p183) *)
		"identifier chain" -> seq : FixedOrder[GrammarToken["identifier"], FixedOrder[GrammarToken["period"], GrammarToken["identifier"]] ...] :> Replace[{seq}[[;; ;; 2]], {x_} :> x],
		"basic identifier chain" -> GrammarToken["identifier chain"],

		(* 6.7 <column reference> (p187) *)
		"column reference" -> {
			GrammarToken["basic identifier chain"]
		},

		(* 6.9 <set function specification> (p191) *)
		"set function specification" -> {
			GrammarToken["aggregate function"](*,
			GrammarToken["grouping operation"]*)
		},

		(* 6.11 <case expression> (p197) *)
		"case expression" -> {
			(*GrammarToken["case abbreviation"],*)
			GrammarToken["case specification"]
		},
		"case specification" -> {
			GrammarToken["simple case"](*,
			GrammarToken["searched case"]*)
		},
		"simple case" -> FixedOrder[
			"CASE",
			case : GrammarToken["case operand"],
			when : GrammarToken["simple when clause"] ...,
			else : Repeated[GrammarToken["else clause"], {0, 1}],
			"END"
		] :> SQLCase1[case, when, else],
		"simple when clause" -> FixedOrder["WHEN", when : GrammarToken["when operand"], "THEN", then : GrammarToken["result"]] :> Sequence[when, then],
		"else clause" -> FixedOrder["ELSE", else : GrammarToken["result"]] :> else,
		"case operand" -> {
			GrammarToken["row value predicand"](*,
			GrammarToken["overlaps predicate part"]*)
		},
		"when operand" -> {
			GrammarToken["row value predicand"]
			(* to do *)
		},
		"result" -> {
			GrammarToken["result expression"],
			"NULL" -> Null
		},
		"result expression" -> GrammarToken["value expression"],

		(* 6.25 <value expression> (p236) *)
		"value expression" -> {
			GrammarToken["common value expression"],
			GrammarToken["boolean value expression"],
			GrammarToken["row value expression"]
		},
		"common value expression" -> {
			GrammarToken["numeric value expression"],
			GrammarToken["string value expression"](*,
			GrammarToken["datetime value expression"],
			GrammarToken["interval value expression"],
			GrammarToken["user-defined type value expression"],
			GrammarToken["reference value expression"],
			GrammarToken["collection value expression"]*)
		},

		(* 6.26 <numeric value expression> (p240) *)
		"numeric value expression" -> {
			GrammarToken["term"],
			FixedOrder[a : GrammarToken["numeric value expression"], GrammarToken["plus sign"], b : GrammarToken["term"]] :> Plus[a, b],
			FixedOrder[a : GrammarToken["numeric value expression"], GrammarToken["minus sign"], b : GrammarToken["term"]] :> Subtract[a, b]
		},
		"term" -> {
			GrammarToken["factor"],
			FixedOrder[a : GrammarToken["term"], GrammarToken["asterisk"], b : GrammarToken["factor"]] :> Times[a, b],
			FixedOrder[a : GrammarToken["term"], GrammarToken["solidus"], b : GrammarToken["factor"]] :> Divide[a, b]
		},
		"factor" -> FixedOrder[
			s : Repeated[GrammarToken["sign"], {0, 1}],
			n : GrammarToken["numeric primary"]
		] :> ({s} // Replace[{
			{"-"} :> -n,
			_ :> n
		}]),
		"numeric primary" -> {
			GrammarToken["value expression primary"](*,
			GrammarToken["numeric value function"]*)
		},

		(* 6.28 <string value expression> (p251) *)
		"string value expression" -> {
			GrammarToken["character value expression"](*,
			GrammarToken["blob value expression"]*)
		},
		"character value expression" -> {
			GrammarToken["concatenation"],
			GrammarToken["character factor"]
		},
		"concatenation" -> FixedOrder[a : GrammarToken["character value expression"], GrammarToken["concatenation operator"], b : GrammarToken["character factor"]] :> Inactive[StringJoin][a, b],
		"character factor" -> GrammarToken["character primary"],
		"character primary" -> {
			GrammarToken["value expression primary"](*,
			GrammarToken["string value function"]*)
		},

		(* 6.34 <boolean value expression> (p277) *)
		"boolean value expression" -> {
			GrammarToken["boolean term"],
			FixedOrder[a : GrammarToken["boolean term"], "OR", b : GrammarToken["boolean value expression"]] :> Or[a, b]
		},
		"boolean term" -> {
			GrammarToken["boolean factor"],
			FixedOrder[a : GrammarToken["boolean factor"], "AND", b : GrammarToken["boolean term"]] :> And[a, b]
		},
		"boolean factor" -> FixedOrder[not : Repeated["NOT", {0, 1}], test : GrammarToken["boolean test"]] :> If[{not} === {}, test, Not[test]],
		"boolean test" -> FixedOrder[
			b : GrammarToken["boolean primary"],
			is : Repeated[FixedOrder["IS", Repeated["NOT", {0, 1}], GrammarToken["truth value"]], {0, 1}]
		] :> ({is} // Replace[{
			{} :> b,
			{_, t_} :> Equal[b, t],
			{_, _, t_} :> Unequal[b, t]
		}]),
		"truth value" -> {
			"TRUE" -> True,
			"FALSE" -> False,
			"UNKNOWN" -> Undefined
		},
		"boolean primary" -> {
			GrammarToken["predicate"],
			GrammarToken["boolean predicand"]
		},
		"boolean predicand" -> {
			GrammarToken["parenthesized boolean value expression"],
			GrammarToken["nonparenthesized value expression primary"]
		},
		"parenthesized boolean value expression" -> FixedOrder[GrammarToken["left paren"], expr : GrammarToken["boolean value expression"], GrammarToken["right paren"]] :> expr,


		(* 7 Query expressions *)

		(* 7.1 <row value constructor> (p293) *)
		"row value constructor predicand" -> {
			GrammarToken["common value expression"],
			GrammarToken["boolean predicand"](*,
			GrammarToken["explicit row value constructor"]*)
		},

		(* 7.2 <row value expression> (p296) *)
		"row value expression" -> {
			GrammarToken["row value special case"](*,
			GrammarToken["explicit row value constructor"]*)
		},
		"row value predicand" -> {
			GrammarToken["row value special case"](*,
			GrammarToken["row value constructor predicand"]*)
		},
		"row value special case" -> GrammarToken["nonparenthesized value expression primary"],

		(* 7.4 <table expression> (p300) *)
		"table expression" -> seq : FixedOrder[
			GrammarToken["from clause"],
			Repeated[GrammarToken["where clause"], {0, 1}],
			Repeated[GrammarToken["group by clause"], {0, 1}]
		] :> seq,

		(* 7.5 <from clause> (p301) *)
		"from clause" -> FixedOrder["FROM", trl : GrammarToken["table reference list"]] :> trl,
		"table reference list" -> seq : FixedOrder[GrammarToken["table reference"], FixedOrder[GrammarToken["comma"], GrammarToken["table reference"]] ...] :> {seq}[[;; ;; 2]],

		(* 7.6 <table reference> (p303) *)
		"table reference" -> GrammarToken["table primary or joined table"],
		"table primary or joined table" -> {
			GrammarToken["table primary"](*,
			GrammarToken["joined table"]*)
		},
		"table primary" -> {
			FixedOrder[t : GrammarToken["table or query name"], cn : Repeated[FixedOrder["AS", GrammarToken["correlation name"]], {0, 1}]] :> If[{cn} === {}, t, Last[{cn}] -> t]
		},
		"table or query name" -> {
			GrammarToken["table name"],
			GrammarToken["query name"]
		},

		(* 7.8 <where clause> (p319) *)
		"where clause" -> FixedOrder["WHERE", cond : GrammarToken["search condition"]] :> cond,

		(* 7.9 <group by clause> (p320) *)
		"group by clause" -> FixedOrder["GROUP", "BY", (*s : Repeated[GrammarToken["set quantifier"], {0, 1}],*) g : GrammarToken["grouping element list"]] :> "GroupBy" -> g,
		"grouping element list" -> seq : FixedOrder[GrammarToken["grouping element"], FixedOrder[GrammarToken["comma"], GrammarToken["grouping element"]] ...] :> {seq}[[;; ;; 2]],
		"grouping element" -> {
			GrammarToken["ordinary grouping set"],
			(*GrammarToken["rollup list"],
			GrammarToken["cube list"],
			GrammarToken["grouping sets specification"],*)
			GrammarToken["empty grouping set"]
		},
		"ordinary grouping set" -> {
			GrammarToken["grouping column reference"],
			FixedOrder[GrammarToken["left paren"], l : GrammarToken["grouping column reference list"], GrammarToken["right paren"]] :> l
		},
		"grouping column reference" -> GrammarToken["column reference"],
		"grouping column reference list" -> seq : FixedOrder[GrammarToken["grouping column reference"], FixedOrder[GrammarToken["comma"], GrammarToken["grouping column reference"]] ...] :> {seq}[[;; ;; 2]],
		"empty grouping set" -> FixedOrder[GrammarToken["left paren"], GrammarToken["right paren"]] :> {},

		(* 7.12 <query specification> (p341) *)
		"query specification" -> FixedOrder["SELECT", sl : GrammarToken["select list"], te : GrammarToken["table expression"]] :> SQLSelect1[sl, te],
		"select list" -> {
			GrammarToken["asterisk"] -> All,
			seq : FixedOrder[GrammarToken["select sublist"], FixedOrder[GrammarToken["comma"], GrammarToken["select sublist"]] ...] :> {seq}[[;; ;; 2]]
		},
		"select sublist" -> {
			GrammarToken["derived column"],
			GrammarToken["qualified asterisk"]
		},
		"qualified asterisk" -> {
			FixedOrder[id : GrammarToken["asterisked identifier chain"], GrammarToken["period"], GrammarToken["asterisk"]] :> {id, All}(*,
			GrammarToken["all fields reference"]*)
		},
		"asterisked identifier chain" -> GrammarToken["asterisked identifier"],
		"asterisked identifier" -> GrammarToken["identifier"],
		"derived column" -> FixedOrder[ve : GrammarToken["value expression"], alias : Repeated[GrammarToken["as clause"], {0, 1}]] :> If[{alias} === {}, ve, Evaluate[alias] -> ve],
		"as clause" -> FixedOrder["AS", alias : GrammarToken["column name"]] :> alias,

		(* 7.13 <query expression> (p350) *)
		"query expression" -> GrammarToken["query expression body"],
		"query expression body" -> {
			GrammarToken["non-join query expression"](*,
			GrammarToken["joined table"]*)
		},
		"non-join query expression" -> GrammarToken["non-join query term"],
		"non-join query term" -> GrammarToken["non-join query primary"],
		"non-join query primary" -> {
			GrammarToken["simple table"],
			FixedOrder[GrammarToken["left paren"], x : GrammarToken["non-join query expression"], GrammarToken["right paren"]] :> x
		},
		"simple table" -> {
			GrammarToken["query specification"](*,
			GrammarToken["table value constructor"],
			GrammarToken["explicit table"]*)
		},


		(* 8 Predicates *)

		(* 8.1 <predicate> (p371) *)
		"predicate" -> {
			GrammarToken["comparison predicate"]
		},

		(* 8.2 <comparison predicate> (p373) *)
		"comparison predicate" -> FixedOrder[a : GrammarToken["row value predicand"], op : GrammarToken["comp op"], b : GrammarToken["row value predicand"]] :> op[a, b],
		"comp op" -> {
			GrammarToken["equals operator"],
			GrammarToken["not equals operator"],
			GrammarToken["less than operator"],
			GrammarToken["greater than operator"],
			GrammarToken["less than or equals operator"],
			GrammarToken["greater than or equals operator"]
		},

		(* 8.19 <search condition> (p416) *)
		"search condition" -> GrammarToken["boolean value expression"],


		(* 10 Additional common elements *)

		(* 10.9 <aggregate function> (p503) *)
		"aggregate function" -> {
			FixedOrder["COUNT", GrammarToken["left paren"], GrammarToken["asterisk"], GrammarToken["right paren"]] :> SQLEvaluation1["COUNT"][],
			GrammarToken["general set function"](*,
			GrammarToken["binary set function"],
			GrammarToken["ordered set function"]*)
		},
		"general set function" -> FixedOrder[
			f : GrammarToken["set function type"],
			GrammarToken["left paren"],
			s : Repeated[GrammarToken["set quantifier"], {0, 1}],
			v : GrammarToken["value expression"],
			GrammarToken["right paren"]
		] :> f[v, s],
		"set function type" -> GrammarToken["computational operation"],
		"computational operation" -> x : Alternatives[
			"AVG" | "MAX" | "MIN" | "SUM",
			"EVERY" | "ANY" | "SOME",
			"COUNT",
			"STDDEV_POP" | "STDDEV_SAMP" | "VAR_SAMP" | "VAR_POP",
			"COLLECT" | "FUSION" | "INTERSECTION"
		] :> SQLEvaluation1[x],
		"set quantifier" -> {
			"DISTINCT" -> "Distinct" -> True,
			"ALL" -> Sequence[]
		},


		(* 14 Data manipulation *)

		(* 14.1 <declare cursor> (p807) *)
		"cursor specification" -> GrammarToken["query expression"],


		(* 21 Direct invocation of SQL *)

		(* 21.1 <direct SQL statement> (p1047) *)
		"direct SQL statement" -> FixedOrder[x : GrammarToken["directly executable statement"], GrammarToken["semicolon"]] :> x,
		"directly executable statement" -> {
			GrammarToken["direct SQL data statement"](*,
			GrammarToken["SQL schema statement"],
			GrammarToken["SQL transaction statement"],
			GrammarToken["SQL connection statement"],
			GrammarToken["SQL session statement"],
			GrammarToken["direct implementation-defined statement"]*)
		},
		"direct SQL data statement" -> {
			(*GrammarToken["delete statement: searched"],*)
			GrammarToken["direct select statement: multiple rows"](*,
			GrammarToken["insert statement"],
			GrammarToken["update statement: searched"],
			GrammarToken["merge statement"],
			GrammarToken["temporary table declaration"]*)
		},

		(* 21.2 <direct select statement: multiple rows> (p1051) *)
		"direct select statement: multiple rows" -> GrammarToken["cursor specification"]
	}
];

(* end import *)
(* -------------------------------------------------- *)


End[];
EndPackage[];
