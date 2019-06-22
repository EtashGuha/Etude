Package["NeuralNetworks`"]



PackageScope["IndefiniteTypeForm"]
PackageScope["IndefiniteTypeString"]

IndefiniteTypeString[type_, pluralq_:False] := toString @ IndefiniteTypeForm[type, pluralq];

IndefiniteTypeForm[type_, pluralq_:False] := removeDefinite @ TypeForm[type, pluralq];

removeDefinite[StringForm["a ``" | "an ``", x_]] := x;
removeDefinite[StringForm[str_, x_, rest___]] := StringForm[str, removeDefinite[x], rest];
removeDefinite[e_] := e;


PackageExport["TypeString"]

TypeString[a_, pluralq_:False] := toString @ TypeForm[a, pluralq];


toString[e_] := TextString @ ReplaceAll[e, s_StringForm :> RuleCondition @ StringFormToString[s]];


PackageExport["TypeForm"]

SetUsage @ "
TypeForm[type$] gives an expression that formats as a human-readable string decsribing a type.
"

TypeForm[type_] := TypeForm[type, False];

TypeForm[type_, _] /; $DebugMode := type;

TypeForm[type_, pluralq_] := ReplaceRepeated[
	If[pluralq, plural, Identity] @ form[type], {
	plural[e_] :> ReplaceAll[e, {
		blockplural -> Identity,
		pform -> pluralform,
		psform -> psingleform,
		ppform -> pluralform,
		s_single :> s
	}],
	single[e_] :> ReplaceAll[e, {
		pform -> singleform,
		psform -> singleform,
		ppform -> psingleform,
		p_plural :> p
	}],
	blockplural -> Identity,
	pform -> singleform,
	psform -> singleform,
	ppform -> psingleform
}];

pluralform[p_String] := p <> If[StringEndsQ[p, "s"], "es", "s"];
pluralform[a_, b_] := b;

psingleform[a_, b_] := a;
psingleform[a_] := a;

singleform[a_, b_] := singleform[a];
singleform[s_String] := 
	If[StringStartsQ[s, "a"|"e"|"i"|"o"|"u"], 
		StringForm["an ``", s],
		StringForm["a ``", s]
	];

singleform[s_] := StringForm["a ``", s];

form[type_] := Match[type,
	ListT[1, t_] :> StringForm["`` containing ``", pform["list"], % @ t],
	ListT[2, t_] :> of["pair", plural[% @ t]],
	ListT[SizeT, t_] :> of["non-empty list", plural[% @ t]],
	ListT[n_, t_] :> of["list", num[n, % @ t]],
	AssocT[k_, v_] :> from["association", % @ k, % @ v],
	RuleT[k_, v_] :> from["rule", % @ k, % @ v],
	StructT[rules_] :> structForm[rules],
	EitherT[list_] :> eitherForm[list],
	EnumT[list_ /; Length[list] > 8] :> StringForm["one of ``, ... (`` other items)", Row[form /@ Take[list, 8], ","], Length[list] - 8],
	EnumT[list_] :> % @ EitherT[list],
	s_String :> "\"" <> s <> "\"",
	ScalarT :> pform["number"],
	IntervalScalarT[min_, max_] :> StringForm["`` between `` and ``", pform["number"], min, max],	
	t_TensorT :> tensorForm[t],
	StringT :> pform["string"],
	BooleanT :> pform["boolean"],
	IntegerT :> pform["integer"],

	RealT :> pform["real number"],
	IndexIntegerT[max_Integer] :> StringForm["`` between 1 and ``", pform["integer"], max],
	IndexIntegerT[Infinity] :> pform["positive integer"],
	IndexIntegerT[All] :> pform["integer"],
	IndexIntegerT[_] :> adj["bounded", "integer"],
	(* The following is a hack needed to produce an appropriate message when attempting to do 
	   unify[IndexIntegerT[All], IndexIntegerT[SizeT]], which triggers unify[All, SizeT] and 
	   then fails producing the message. TODO: try to remove this rule and have 
	   unify[IndexIntegerT[All], IndexIntegerT[SizeT]] fail immediately *)
	All :> "integer",
	SizeT|PosIntegerT|LengthVar :> adj["positive", "integer"],
	NaturalT :> adj["non-negative", "integer"],
	ImageT :> pform["image"],
	Image3DT :> pform["image3d"],
	FunctionT :> pform["function"],
	MatchT[sym_Symbol] :> sym,
	MatchT[Infinity] :> "Infinity",
	MatchT[patt_] :> StringForm["`` matching ``", pform["expression"], patt],
	TypeT :> pform["type"],
	TypeExpressionT :> pform["type expression"],
	ExpressionT :> pform["expression"],
	DurationT :> pform["duration"],
	Defaulting[t_, _] :> %[t],
	Nullable[t_] :> %[t],
	_ImageT :> pform["image"], (* TODO: special cases for known size, colspace *)
	_Image3DT :> pform["image3d"], (* TODO: special cases for known size, colspace *)
	sym_Symbol /; Context[sym] === "System`" :> sym,
	cod:CoderP :> pform[CoderKind[cod]],
	TypeAliasP :> %[ResolveAlias @ type],
	l_List :> Map[%, l],
	DistributionT :> adj["univariate", "distribution"],
	NetT[] :> pform["net"],
	NetT[ins_, outs_] :> pform["net"], (* TODO: make more specific *)
	ValidatedParameter[t_] :> %[t],
	CustomType[desc_, _] :> fromCustomDesc[desc],
	type
];

fromCustomDesc[str_String] := Scope[
	If[StringFreeQ[str, "["], Return @ str];
	args = {};
	form = StringReplace[str, {
		"plural[" ~~ Shortest[a__] ~~ "]" :> (AppendTo[args, pform[a]]; "``")
	}];
	StringForm[form, Sequence @@ args]
];


tensorForm[t_] := Match[t,
	TensorT[dims__, r:RealT] :> If[dims === {}, form[r], StringForm["`` of ``", %[TensorT[dims,AtomT]], plural@form[r]]],
	TensorT[dims_, i_IndexIntegerT] :> If[dims === {}, form[i], StringForm["`` of ``", %[TensorT[dims,AtomT]], plural@form[i]]],
	TensorT[{m_, n_, o_} ? hasOneIntQ, AtomT] :> adj[size[{m, n, o}], pform["array"]],
	TensorT[{m_, n_, o_, p_} ? hasOneIntQ, AtomT] :> adj[size[{m, n, o, p}], pform["array"]],
	TensorT[{m_, n_} ? hasOneIntQ, AtomT] :> adj[size[{m, n}], pform["matrix", "matrices"]],
	TensorT[{_, _}|ListT[2, SizeT], AtomT] :> pform["matrix", "matrices"],
	TensorT[{n_Integer}, AtomT] :> dashadj["length", n, "vector"],
	TensorT[{_}|ListT[1, SizeT], AtomT] :> pform["vector"],
	TensorT[dims:{__Integer}, AtomT] :> adj[size[dims], "array"],
	TensorT[{n_Integer}, _TensorT] :> StringForm["`` of length ``", pform["array"], n],
	TensorT[SizeListT[], TensorT[dims:{__Integer}, AtomT]] :> StringForm["`` with final dimensions ``", pform["array"], size[dims]],
	TensorT[dims:{__Integer}, _TensorT] :> StringForm["`` with initial dimensions ``", pform["array"], size[dims]],
	TensorT[_, _TensorT] :> StringForm["`` of rank \[GreaterEqual] ``", pform["array"], minRank[t]],
	Match[
		TRank[t], 
		n_Integer :> dashadj["rank", n, "array"], 
		pform["array"]
	]
];

minRank[TensorT[ListT[SizeT, _], t_]] := 1 + minRank[t];
minRank[TensorT[ListT[NaturalT, _], t_]] := minRank[t];
minRank[TensorT[ListT[n_Integer, _], t_]] := n + minRank[t];
minRank[TensorT[list_List, t_]] := Length[list] + minRank[t];
minRank[_] := 0;

(* REFACTOR: make strings for arrays of types other than RealT *)


hasOneIntQ[a_List] := !FreeQ[a, _Integer];

eitherForm[{a_, b_}] := 
	StringForm["either `` or ``", form[a], form[b]];
eitherForm[list_] := 
	StringForm["either ``, or ``", Row[form /@ Most[list], ","], form @ Last[list]];

tupleForm[list_] := Panic["NotImplemented"];

structForm[rules_] := Scope[
	StringForm[
		"`` containing the keys `` and ``", 
		pform["association"],
		Row[structEntry @@@ Most[rules], ","], structEntry @@ Last[rules]
	]
];

structEntry[key_, t_] := 
	StringForm["`` (``)", QuotedString[key], single[form @ t]]; 

size[n_List] := Row[n /. {
	lv_LengthVar :> FormatLengthVar[lv],
	SizeT -> "\[DottedSquare]",
	All -> "\[PlusMinus]\[Infinity]"
	}, "\[Times]"];

ppform[p_pform] := ppform @@ p;

adj[a_, b_] := StringForm["`` ``", psform[a], ppform[b]];
dashadj[a_, b_, c_] := StringForm["``-`` ``", psform[a], b, ppform[c]];

num[SizeT|NaturalT, b_] := plural[b];
num[1, b_] := b;
num[a_, b_] := StringForm["`` ``", a, plural[b]];

of[a_, b_] := StringForm["`` of ``", pform[a], b];
from[a_, k_, v_] := StringForm["`` from `` to ``", pform[a], plural[k], plural[v]];


