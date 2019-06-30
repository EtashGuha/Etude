Package["NeuralNetworks`"]



PackageScope["TRoll"]
PackageScope["TUnroll"]

TRoll[a_, b___] := TensorT[Replace[a, {i_Integer :> {i}, All -> SizeListT[]}], TRoll[b]];
TRoll[b_] := b;

TUnroll[TensorT[d_List, p:TTypeP]] := {d, p}; (* <- fast path *)
TUnroll[$Failed] := $Failed;
TUnroll[t_] := unroll[t] //. {L___, a_List, b_List, R___} :> {L, Join[a, b], R};

unroll[TensorT[t1_, t2_]] := Prepend[
	unroll[t2],
	Replace[t1, ListT[n_, z_] :> If[IntegerQ[n], Table[z, n], All]]
];

unroll[t_] := {t};


PackageScope["TToNormalForm"]
PackageScope["TFromNormalForm"]

(* NormalForm is {dimsOnLeft, hasArbitraryGap, dimsOnRight, innerType} *)

midP = All | _RepeatedInteger;
TToNormalForm[t_TensorT] := Match[TUnroll[t],
	{a_List, it_} :> {a, False, {}, it},
	{a_List, All, it_} :> {a, True, {}, it},
	{All, it_} :> {{}, True, {}, it},
	{All, b_List, it_} :> {{}, True, b, it},
	{a_List, All, b_List, it_} :> {a, True, b, it},
	{a_List, r_RepeatedInteger, it_} :> {a, r, {}, it},
	z_ /; !FreeQ[z, RepeatedInteger] :> (Print[z]; Panic["UnsupportedRepeatedIntegerRole", "``", z]),
	{$Failed, False, $Failed, $Failed}
];

(*
(* this ensures that TensorT[{5}, RealTensorT] and TensorT[SizeListT[], VectorT[5]] get
normalized to the same thing. probably better handled by making the second
arg of the canon form a minimum and maximum rank, and always gobbling SizeTs into this value. *)
canonNF[{l_List, True, {s:SizeT.., r___}, t_}] := {Join[l, {s}], True, {r}, t};
canonNF[e_] := e;
*)

TFromNormalForm[{a_List, False, _, it_}] := TensorT[a, it];
TFromNormalForm[{a_List, True, {}, it_}] := TensorT[a, TensorT[SizeListT[], it]];
TFromNormalForm[{a_List, True, b_List, it_}] := TensorT[a, TensorT[SizeListT[], TensorT[b, it]]];
TFromNormalForm[e:{_, rp_RepeatedInteger, _, _}] := TFromNormalForm[e /. rp -> True] /. SizeListT[] -> rp;


PackageScope["TDimensions"]

SetUsage @ "
TDimensions[type$] returns the list of dimensions of a numeric array type (TensorT), \
or $Failed if the rank is not fixed or the type is not an array type. Note: the list returned \
could contain SizeT. 
Certain pseudo-scalar values like IndexInteger, are treated as scalars (dims {})."

SetHoldRest[TDimensions];

TDimensions = MatchValues[
	TensorT[dims_List, t_] := joinDims[dims, %[t]];
	TensorT[SizeListT[rank_Integer], t_] := joinDims[Table[SizeT, rank], %[t]];
	TTypeP := {};
	Nullable[t_] := %[t];
	TypeAliasP := %[ResolveAlias[type]];
	cod:CoderP := %[CoderType[cod]];
	list_List := Map[%, list];
	$Failed
];

joinDims[a_, b_] := ToList[a, b];
joinDims[_, $Failed] := $Failed;

TDimensions[t_, r_] := ReplaceAll[TDimensions[t], $Failed :> r];


PackageScope["TFirstDim"]

TFirstDim = MatchValues[
	TensorT[dims_List, _] := First[dims, $Failed];
	TensorT[_SizeListT, _] := SizeT;
	cod:CoderP := %[CoderType[cod]];
	$Failed
];


PackageScope["TType"]

SetUsage @ "
TType[tensor$] returns the innermost type of the array tensor$.
* This is usually a RealT."

Clear[TType]
TType[TensorT[_ , t_]] := TType[t];
TType[t:TTypeP] := t;
TType[encoder_NetEncoder] := TType[CoderType[encoder]];
TType[decoder_NetDecoder] := TType[CoderType[decoder]];
TType[notsupported_] := $Failed;

PackageScope["TTypeDefaulted"]
TTypeDefaulted[t_] := 

PackageScope["TRank"]

SetHoldRest[TRank];

SetUsage @ "
TRank[type$] returns the rank of a numeric array type (TensorT), or $Failed if \
this isn't known or the type isn't an array type."

TRank = MatchValues[
	TensorT[dims_List, t_] := addRank[Length[dims], %[t]];
	TensorT[SizeListT[rank_Integer], t_] := addRank[rank, %[t]];
	TensorT[SizeListT[], t_] := If[$minRankQ, %[t], $Failed];
	TTypeP := 0;
	Nullable[t_] := %[t];
	t:TypeAliasP := %[ResolveAlias[t]];
	cod:CoderP := %[CoderType[cod]];
	$Failed
];

$minRankQ = False;
addRank[n_, $Failed] := $Failed;
addRank[n_, m_] := n + m;

TRank[t_, r_] := Replace[TRank[t], $Failed :> r];


